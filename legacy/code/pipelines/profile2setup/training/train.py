"""Stage 5 training pipeline for profile2setup v2."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from profile2setup.inference.routing import route_setup_prediction
from profile2setup.models import build_model_from_config
from profile2setup.schema import VARIABLE_ORDER
from profile2setup.training.dataset import (
    Profile2SetupDataset,
    profile2setup_collate_fn,
)
from profile2setup.training.losses import compute_profile2setup_loss
from profile2setup.training.text import build_vocab_from_jsonl, load_vocab, save_vocab
from profile2setup.training.history import (
    append_history_row,
    plot_training_history,
    save_history_csv,
    save_history_json,
)
from profile2setup.training.utils import (
    count_parameters,
    count_task_types_from_records_or_dataset,
    ensure_dir,
    format_per_variable_table,
    get_device,
    load_yaml,
    move_batch_to_device,
    save_checkpoint,
    save_json,
    set_seed,
)


def _deep_update(base: dict, updates: dict | None) -> dict:
    if not updates:
        return base
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _as_float(value: torch.Tensor | float) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def _metric_average(total: float, count: int) -> float:
    if count <= 0:
        return float("nan")
    return float(total / count)


def _load_training_checkpoint(path) -> dict:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Resume checkpoint must be a dict, got {type(checkpoint).__name__}")
    return checkpoint


def _load_history(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r") as f:
        history = json.load(f)
    if not isinstance(history, list):
        raise ValueError(f"History JSON must contain a list: {path}")
    return [dict(row) for row in history if isinstance(row, dict)]


def _best_val_loss_from_history(history: list[dict]) -> float:
    values = []
    for row in history:
        value = row.get("val_loss")
        if isinstance(value, (int, float)):
            values.append(float(value))
    return min(values) if values else float("inf")


def resolve_data_path(config, key_primary, key_alias=None):
    """Resolve a dataset path, accepting a legacy alias when needed."""
    data_cfg = config.get("data") or {}
    path = data_cfg.get(key_primary)
    if path is None and key_alias is not None:
        path = data_cfg.get(key_alias)
    if path is None:
        alias_text = f" or data.{key_alias}" if key_alias else ""
        raise KeyError(f"Missing required config key data.{key_primary}{alias_text}")
    return str(path)


def build_or_load_vocab(config) -> dict:
    """Load an existing vocab or build one from the training JSONL."""
    data_cfg = config.get("data") or {}
    vocab_path = Path(data_cfg["vocab_path"])
    if vocab_path.exists():
        vocab = load_vocab(vocab_path)
    else:
        train_path = resolve_data_path(config, "train_path", "train_jsonl")
        vocab = build_vocab_from_jsonl(train_path, min_freq=1)
    save_vocab(vocab, vocab_path)
    return vocab


def build_datasets(config, vocab) -> tuple[Profile2SetupDataset, Profile2SetupDataset]:
    """Build train and validation datasets using the Stage 3 dataset API."""
    data_cfg = config.get("data") or {}
    common_kwargs = {
        "variables_config_path": data_cfg["variables_config"],
        "vocab": vocab,
        "input_size": data_cfg.get("input_size", 128),
        "max_text_len": data_cfg.get("max_text_len", 32),
        "normalize_mode": data_cfg.get("normalize_mode", "max_log"),
        "task_filter": data_cfg.get("task_filter"),
        "strict": data_cfg.get("strict", True),
        "change_threshold": data_cfg.get("change_threshold", 1.0e-6),
    }
    train_dataset = Profile2SetupDataset(
        jsonl_path=resolve_data_path(config, "train_path", "train_jsonl"),
        limit=data_cfg.get("train_limit"),
        **common_kwargs,
    )
    val_dataset = Profile2SetupDataset(
        jsonl_path=resolve_data_path(config, "val_path", "val_jsonl"),
        limit=data_cfg.get("val_limit"),
        **common_kwargs,
    )
    return train_dataset, val_dataset


def build_dataloaders(config, train_dataset, val_dataset):
    """Build train and validation DataLoaders."""
    opt_cfg = config.get("optimization") or {}
    batch_size = int(opt_cfg.get("batch_size", 32))
    num_workers = int(opt_cfg.get("num_workers", 0))
    generator = torch.Generator()
    generator.manual_seed(int((config.get("experiment") or {}).get("seed", 42)))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=profile2setup_collate_fn,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=profile2setup_collate_fn,
    )
    return train_loader, val_loader


def _forward_model(model, batch):
    return model(
        batch["profile"],
        batch["prompt_tokens"],
        batch["current_setup"],
        setup_present=batch["setup_present"],
        intent_features=batch.get("intent_features"),
    )


def _loss_from_outputs(outputs, batch, config):
    loss_cfg = dict(config.get("loss") or {})
    loss_cfg.update(config.get("losses") or {})
    return compute_profile2setup_loss(
        outputs,
        batch,
        absolute_weight=loss_cfg.get("absolute_weight", 1.0),
        delta_weight=loss_cfg.get("delta_weight", 1.0),
        change_weight=loss_cfg.get("change_weight", 0.5),
        constraint_weight=loss_cfg.get("constraint_weight", 0.0),
        fixed_change_mask=batch.get("fixed_change_mask"),
    )


def train_one_epoch(model, loader, optimizer, device, config, epoch):
    """Train for one epoch and return averaged losses."""
    model.train()
    opt_cfg = config.get("optimization") or {}
    log_every = int((config.get("logging") or {}).get("log_every_n_steps", 20))
    grad_clip_norm = opt_cfg.get("grad_clip_norm")
    totals = {
        "loss": 0.0,
        "absolute_loss": 0.0,
        "delta_loss": 0.0,
        "change_loss": 0.0,
        "constraint_loss": 0.0,
    }
    seen = 0

    for step, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = _forward_model(model, batch)
        loss_dict = _loss_from_outputs(outputs, batch, config)
        loss_dict["loss"].backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
        optimizer.step()

        batch_size = int(batch["profile"].shape[0])
        seen += batch_size
        for key in totals:
            totals[key] += _as_float(loss_dict[key]) * batch_size

        if log_every > 0 and step % log_every == 0:
            print(
                f"epoch {epoch} step {step}/{len(loader)} "
                f"train_loss={_as_float(loss_dict['loss']):.6f}"
            )

    return {f"train_{key}": _metric_average(value, seen) for key, value in totals.items()}


def compute_masked_mae_per_variable(pred, target, mask) -> dict:
    """Compute masked normalized MAE for each canonical variable."""
    if mask.ndim == 1:
        mask = mask.unsqueeze(-1)
    mask = mask.to(dtype=pred.dtype, device=pred.device)
    if mask.shape[-1] == 1:
        mask = mask.expand_as(pred)
    if mask.shape != pred.shape:
        raise ValueError(f"mask shape must broadcast to pred shape, got {tuple(mask.shape)} vs {tuple(pred.shape)}")

    abs_error = torch.abs(pred - target) * mask
    denom = mask.sum(dim=0)
    sums = abs_error.sum(dim=0)
    values = []
    for idx in range(len(VARIABLE_ORDER)):
        if float(denom[idx].detach().cpu().item()) <= 0.0:
            values.append(float("nan"))
        else:
            values.append(float((sums[idx] / denom[idx]).detach().cpu().item()))
    return {name: values[idx] for idx, name in enumerate(VARIABLE_ORDER)}


def compute_validation_metrics(outputs, batch, config) -> dict:
    """Compute batch-level validation MAE summaries and active mask counts."""
    val_cfg = config.get("validation") or {}
    routed = route_setup_prediction(
        outputs,
        batch["current_setup"],
        batch["setup_present"],
        prefer_absolute_when_setup_missing=val_cfg.get("prefer_absolute_when_setup_missing", True),
    )
    return {
        "absolute_mae": compute_masked_mae_per_variable(
            outputs["absolute"], batch["target_setup"], batch["absolute_loss_mask"]
        ),
        "delta_mae": compute_masked_mae_per_variable(
            outputs["delta"], batch["target_delta"], batch["delta_loss_mask"]
        ),
        "routed_setup_mae": compute_masked_mae_per_variable(
            routed, batch["target_setup"], batch["absolute_loss_mask"]
        ),
        "active_counts": {
            "absolute_loss_mask": int(batch["absolute_loss_mask"].sum().detach().cpu().item()),
            "delta_loss_mask": int(batch["delta_loss_mask"].sum().detach().cpu().item()),
            "change_loss_mask": int(batch["change_loss_mask"].sum().detach().cpu().item()),
        },
        "routed": routed,
    }


def _accumulate_masked_mae(accumulator, name, pred, target, mask) -> None:
    if mask.ndim == 1:
        mask = mask.unsqueeze(-1)
    mask = mask.to(dtype=pred.dtype, device=pred.device)
    if mask.shape[-1] == 1:
        mask = mask.expand_as(pred)
    abs_error = torch.abs(pred - target) * mask
    accumulator[name]["sum"] += abs_error.sum(dim=0).detach().cpu()
    accumulator[name]["count"] += mask.sum(dim=0).detach().cpu()


def _finalize_mae(accumulator, name) -> dict:
    sums = accumulator[name]["sum"]
    counts = accumulator[name]["count"]
    out = {}
    for idx, variable in enumerate(VARIABLE_ORDER):
        count = float(counts[idx].item())
        out[variable] = float(sums[idx].item() / count) if count > 0.0 else float("nan")
    return out


def _to_jsonable_float_list(tensor: torch.Tensor, index: int) -> list[float]:
    return [float(x) for x in tensor[index].detach().cpu().tolist()]


def validate_one_epoch(model, loader, device, config):
    """Validate for one epoch and return averaged losses and MAE metrics."""
    model.eval()
    totals = {
        "loss": 0.0,
        "absolute_loss": 0.0,
        "delta_loss": 0.0,
        "change_loss": 0.0,
        "constraint_loss": 0.0,
    }
    seen = 0
    active_counts = {"absolute_loss_mask": 0, "delta_loss_mask": 0, "change_loss_mask": 0}
    mae_acc = {
        name: {
            "sum": torch.zeros(len(VARIABLE_ORDER), dtype=torch.float64),
            "count": torch.zeros(len(VARIABLE_ORDER), dtype=torch.float64),
        }
        for name in ("absolute_mae", "delta_mae", "routed_setup_mae")
    }
    examples = []
    max_examples = int((config.get("validation") or {}).get("save_examples", 0) or 0)

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            outputs = _forward_model(model, batch)
            loss_dict = _loss_from_outputs(outputs, batch, config)
            metrics = compute_validation_metrics(outputs, batch, config)
            routed = metrics["routed"]

            batch_size = int(batch["profile"].shape[0])
            seen += batch_size
            for key in totals:
                totals[key] += _as_float(loss_dict[key]) * batch_size
            for key, value in metrics["active_counts"].items():
                active_counts[key] += int(value)

            _accumulate_masked_mae(
                mae_acc, "absolute_mae", outputs["absolute"], batch["target_setup"], batch["absolute_loss_mask"]
            )
            _accumulate_masked_mae(
                mae_acc, "delta_mae", outputs["delta"], batch["target_delta"], batch["delta_loss_mask"]
            )
            _accumulate_masked_mae(
                mae_acc, "routed_setup_mae", routed, batch["target_setup"], batch["absolute_loss_mask"]
            )

            if len(examples) < max_examples:
                remaining = max_examples - len(examples)
                for idx in range(min(batch_size, remaining)):
                    item = {
                        "record_id": batch["record_id"][idx],
                        "task_type": batch["task_type"][idx],
                        "prompt": batch["prompt"][idx],
                        "setup_present": float(batch["setup_present"][idx].detach().cpu().reshape(-1)[0].item()),
                        "predicted_absolute": _to_jsonable_float_list(outputs["absolute"], idx),
                        "predicted_delta": _to_jsonable_float_list(outputs["delta"], idx),
                        "routed_prediction": _to_jsonable_float_list(routed, idx),
                        "target_setup": _to_jsonable_float_list(batch["target_setup"], idx),
                    }
                    if float(batch["delta_loss_mask"][idx].detach().cpu().reshape(-1)[0].item()) > 0.0:
                        item["target_delta"] = _to_jsonable_float_list(batch["target_delta"], idx)
                    examples.append(item)

    out = {f"val_{key}": _metric_average(value, seen) for key, value in totals.items()}
    out["absolute_mae"] = _finalize_mae(mae_acc, "absolute_mae")
    out["delta_mae"] = _finalize_mae(mae_acc, "delta_mae")
    out["routed_setup_mae"] = _finalize_mae(mae_acc, "routed_setup_mae")
    out["active_counts"] = active_counts
    out["examples"] = examples
    return out


def _print_startup(config, device, train_dataset, val_dataset, vocab, model) -> None:
    train_path = resolve_data_path(config, "train_path", "train_jsonl")
    val_path = resolve_data_path(config, "val_path", "val_jsonl")
    print(f"experiment: {(config.get('experiment') or {}).get('name')}")
    print(f"device: {device}")
    print(f"train path: {train_path}")
    print(f"val path: {val_path}")
    print(f"train dataset length: {len(train_dataset)}")
    print(f"val dataset length: {len(val_dataset)}")
    print(f"vocab size: {len(vocab)}")
    print(f"model parameter count: {count_parameters(model)}")
    print(f"train task type counts: {count_task_types_from_records_or_dataset(train_dataset)}")
    print(f"val task type counts: {count_task_types_from_records_or_dataset(val_dataset)}")


def _print_epoch_summary(epoch, train_metrics, val_metrics) -> None:
    print(
        f"epoch {epoch} "
        f"train_loss={train_metrics['train_loss']:.6f} "
        f"train_absolute_loss={train_metrics['train_absolute_loss']:.6f} "
        f"train_delta_loss={train_metrics['train_delta_loss']:.6f} "
        f"train_change_loss={train_metrics['train_change_loss']:.6f} "
        f"train_constraint_loss={train_metrics.get('train_constraint_loss', 0.0):.6f} "
        f"val_loss={val_metrics['val_loss']:.6f} "
        f"val_absolute_loss={val_metrics['val_absolute_loss']:.6f} "
        f"val_delta_loss={val_metrics['val_delta_loss']:.6f} "
        f"val_change_loss={val_metrics['val_change_loss']:.6f} "
        f"val_constraint_loss={val_metrics.get('val_constraint_loss', 0.0):.6f}"
    )
    print(f"validation active supervision counts: {val_metrics['active_counts']}")
    print(format_per_variable_table("absolute_mae", VARIABLE_ORDER, val_metrics["absolute_mae"]))
    print(format_per_variable_table("delta_mae", VARIABLE_ORDER, val_metrics["delta_mae"]))
    print(format_per_variable_table("routed_setup_mae", VARIABLE_ORDER, val_metrics["routed_setup_mae"]))


def _apply_smoke_test_config(config: dict, smoke_test: bool) -> dict:
    smoke_cfg = config.get("smoke_test") or {}
    if not smoke_test and not smoke_cfg.get("enabled", False):
        return config
    config.setdefault("data", {})
    config.setdefault("optimization", {})
    config["data"]["train_limit"] = smoke_cfg.get("train_limit", 32)
    config["data"]["val_limit"] = smoke_cfg.get("val_limit", 16)
    config["optimization"]["epochs"] = smoke_cfg.get("epochs", 1)
    config["optimization"]["batch_size"] = smoke_cfg.get("batch_size", 4)
    return config


def _checkpoint_dir(config) -> Path:
    experiment_name = (config.get("experiment") or {}).get("name", "profile2setup_v2")
    checkpoint_cfg = config.get("checkpoint") or {}
    base_dir = checkpoint_cfg.get("save_dir")
    if base_dir is None:
        base_dir = (config.get("logging") or {}).get("save_dir", "profile2setup/checkpoints")
    return ensure_dir(Path(base_dir) / str(experiment_name))


def train_from_config(
    config_path: str,
    overrides: dict | None = None,
    smoke_test: bool = False,
    resume_checkpoint_path: str | None = None,
    additional_epochs: int | None = None,
) -> dict:
    """Train Profile2SetupModel from a YAML config and return run metadata."""
    resume_checkpoint = _load_training_checkpoint(resume_checkpoint_path) if resume_checkpoint_path else None
    if resume_checkpoint is not None and isinstance(resume_checkpoint.get("config"), dict):
        config = copy.deepcopy(resume_checkpoint["config"])
    else:
        config = load_yaml(config_path)
    config = _deep_update(copy.deepcopy(config), overrides)
    config = _apply_smoke_test_config(config, smoke_test)

    set_seed(int((config.get("experiment") or {}).get("seed", 42)))
    device = get_device((config.get("optimization") or {}).get("device", "auto"))
    if resume_checkpoint is not None and isinstance(resume_checkpoint.get("vocab"), dict):
        vocab = {str(key): int(value) for key, value in resume_checkpoint["vocab"].items()}
        save_vocab(vocab, Path((config.get("data") or {})["vocab_path"]))
    else:
        vocab = build_or_load_vocab(config)
    train_dataset, val_dataset = build_datasets(config, vocab)
    if len(train_dataset) == 0:
        raise RuntimeError("Training dataset is empty")
    if len(val_dataset) == 0:
        raise RuntimeError("Validation dataset is empty")
    train_loader, val_loader = build_dataloaders(config, train_dataset, val_dataset)

    model = build_model_from_config(config, vocab_size=len(vocab)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float((config.get("optimization") or {}).get("learning_rate", 1.0e-3)),
        weight_decay=float((config.get("optimization") or {}).get("weight_decay", 1.0e-4)),
    )
    if resume_checkpoint is not None:
        if list(resume_checkpoint.get("variable_order") or VARIABLE_ORDER) != list(VARIABLE_ORDER):
            raise ValueError(f"Resume checkpoint variable_order must match {VARIABLE_ORDER}")
        model_state = resume_checkpoint.get("model_state_dict")
        if model_state is None:
            raise ValueError("Resume checkpoint is missing model_state_dict")
        model.load_state_dict(model_state)
        optimizer_state = resume_checkpoint.get("optimizer_state_dict")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(device)

    _print_startup(config, device, train_dataset, val_dataset, vocab, model)

    checkpoint_dir = _checkpoint_dir(config)
    epochs = int((config.get("optimization") or {}).get("epochs", 20))
    save_every_epoch = bool((config.get("checkpoint") or {}).get("save_every_epoch", True))
    best_val_loss = float("inf")
    best_checkpoint_path = checkpoint_dir / "best.pt"
    latest_checkpoint_path = checkpoint_dir / "latest.pt"
    history_csv_path = checkpoint_dir / "history.csv"
    history_json_path = checkpoint_dir / "history.json"
    history = _load_history(history_json_path) if resume_checkpoint is not None else []
    start_epoch = 1
    if resume_checkpoint is not None:
        start_epoch = int(resume_checkpoint.get("epoch", 0)) + 1
        if additional_epochs is not None:
            if int(additional_epochs) <= 0:
                raise ValueError("--additional-epochs must be positive")
            epochs = start_epoch + int(additional_epochs) - 1
        print(f"resuming from checkpoint: {resume_checkpoint_path}")
        print(f"resume start epoch: {start_epoch}")
        print(f"resume final epoch: {epochs}")
    elif additional_epochs is not None:
        raise ValueError("--additional-epochs requires --resume-checkpoint")
    best_val_loss = _best_val_loss_from_history(history) if resume_checkpoint is not None else best_val_loss

    if start_epoch > epochs:
        print(f"no training needed: start epoch {start_epoch} is greater than final epoch {epochs}")

    for epoch in range(start_epoch, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, config, epoch)
        val_metrics = validate_one_epoch(model, val_loader, device, config)
        epoch_metrics = append_history_row(history, epoch, train_metrics, val_metrics)
        save_history_csv(history, history_csv_path)
        save_history_json(history, history_json_path)
        plot_training_history(history, checkpoint_dir)
        _print_epoch_summary(epoch, train_metrics, val_metrics)

        if int((config.get("validation") or {}).get("save_examples", 0) or 0) > 0:
            save_json(val_metrics["examples"], checkpoint_dir / f"validation_examples_epoch_{epoch:03d}.json")

        save_checkpoint(
            latest_checkpoint_path,
            model,
            optimizer,
            epoch,
            config,
            vocab,
            epoch_metrics,
            VARIABLE_ORDER,
        )
        if save_every_epoch:
            save_checkpoint(
                checkpoint_dir / f"epoch_{epoch:03d}.pt",
                model,
                optimizer,
                epoch,
                config,
                vocab,
                epoch_metrics,
                VARIABLE_ORDER,
            )
        current_val_loss = float(val_metrics["val_loss"])
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            save_checkpoint(
                best_checkpoint_path,
                model,
                optimizer,
                epoch,
                config,
                vocab,
                epoch_metrics,
                VARIABLE_ORDER,
            )

    return {
        "config": config,
        "checkpoint_dir": str(checkpoint_dir),
        "latest_checkpoint_path": str(latest_checkpoint_path),
        "best_checkpoint_path": str(best_checkpoint_path),
        "best_val_loss": best_val_loss,
        "history_csv_path": str(history_csv_path),
        "history_json_path": str(history_json_path),
        "history": history,
    }
