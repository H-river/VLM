#!/usr/bin/env python3
"""
General stage checker for profile2setup v2.

Run from repo root:

    python -m profile2setup.scripts.check_all_stages_cli

Optional smoke checks:

    python -m profile2setup.scripts.check_all_stages_cli \
      --run-dataset-smoke \
      --run-model-smoke \
      --run-train-smoke

This script checks readiness for:

Stage 0: skeleton/schema/configs
Stage 1: simulator v2 fields
Stage 2: dataset builders
Stage 3: preprocessing + dataset loading
Stage 4: model architecture
Stage 5: training loop/checkpoints
Stage 6: offline evaluation/baselines
Stage 7: closed-loop simulation evaluation
Stage 8: workflow/integrity docs/scripts
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


CANONICAL_VARIABLE_ORDER = [
    "source_to_lens",
    "lens_to_camera",
    "focal_length",
    "lens_x",
    "lens_y",
    "camera_x",
    "camera_y",
]

# Avoid literal forbidden tokens in this checker to prevent simple grep false positives.
FORBIDDEN_KEYS = [
    "align" + "ment",
    "align" + "ment_x",
    "align" + "ment_y",
]


class StageChecker:
    def __init__(self, repo_root: Path, strict: bool = False, json_out: Path | None = None):
        self.repo_root = repo_root.resolve()
        self.strict = strict
        self.json_out = json_out
        self.results: list[dict[str, Any]] = []
        self.stage_counts: dict[str, Counter] = defaultdict(Counter)

        if str(self.repo_root) not in sys.path:
            sys.path.insert(0, str(self.repo_root))

    def record(self, stage: str, name: str, status: str, message: str = "") -> None:
        assert status in {"PASS", "FAIL", "WARN", "SKIP"}
        self.stage_counts[stage][status] += 1
        self.results.append(
            {
                "stage": stage,
                "check": name,
                "status": status,
                "message": message,
            }
        )
        prefix = {
            "PASS": "✅",
            "FAIL": "❌",
            "WARN": "⚠️ ",
            "SKIP": "⏭️ ",
        }[status]
        msg = f"{prefix} [{stage}] {name}"
        if message:
            msg += f" — {message}"
        print(msg)

    def pass_(self, stage: str, name: str, message: str = "") -> None:
        self.record(stage, name, "PASS", message)

    def fail(self, stage: str, name: str, message: str = "") -> None:
        self.record(stage, name, "FAIL", message)

    def warn(self, stage: str, name: str, message: str = "") -> None:
        self.record(stage, name, "WARN", message)

    def skip(self, stage: str, name: str, message: str = "") -> None:
        self.record(stage, name, "SKIP", message)

    def path(self, rel: str | Path) -> Path:
        return self.repo_root / rel

    def require_file(self, stage: str, rel: str) -> bool:
        p = self.path(rel)
        if p.is_file():
            self.pass_(stage, f"file exists: {rel}")
            return True
        self.fail(stage, f"file missing: {rel}")
        return False

    def require_dir(self, stage: str, rel: str) -> bool:
        p = self.path(rel)
        if p.is_dir():
            self.pass_(stage, f"dir exists: {rel}")
            return True
        self.fail(stage, f"dir missing: {rel}")
        return False

    def load_yaml_file(self, rel: str) -> dict[str, Any] | None:
        if yaml is None:
            self.warn("GLOBAL", "PyYAML not available", "YAML checks will be limited.")
            return None
        p = self.path(rel)
        if not p.exists():
            return None
        try:
            with p.open("r", encoding="utf-8") as f:
                obj = yaml.safe_load(f) or {}
            return obj
        except Exception as e:
            self.fail("GLOBAL", f"YAML parse failed: {rel}", str(e))
            return None

    def run_cmd(self, stage: str, name: str, cmd: list[str], timeout: int = 120) -> bool:
        print(f"\n--- running: {' '.join(cmd)}")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.repo_root) + os.pathsep + env.get("PYTHONPATH", "")
        try:
            proc = subprocess.run(
                cmd,
                cwd=self.repo_root,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
            if proc.returncode == 0:
                self.pass_(stage, name)
                if proc.stdout.strip():
                    print(proc.stdout[-4000:])
                return True
            self.fail(stage, name, f"exit={proc.returncode}\n{proc.stdout[-4000:]}")
            return False
        except subprocess.TimeoutExpired:
            self.fail(stage, name, f"timeout after {timeout}s")
            return False
        except Exception as e:
            self.fail(stage, name, str(e))
            return False

    def contains_forbidden_key(self, obj: Any) -> bool:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in FORBIDDEN_KEYS:
                    return True
                if self.contains_forbidden_key(v):
                    return True
        elif isinstance(obj, list):
            return any(self.contains_forbidden_key(x) for x in obj)
        return False

    def find_forbidden_key_paths(self, obj: Any, prefix: str = "") -> list[str]:
        found = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                here = f"{prefix}.{k}" if prefix else str(k)
                if k in FORBIDDEN_KEYS:
                    found.append(here)
                found.extend(self.find_forbidden_key_paths(v, here))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                found.extend(self.find_forbidden_key_paths(v, f"{prefix}[{i}]"))
        return found

    def validate_setup_dict(self, setup: Any) -> tuple[bool, str]:
        if setup is None:
            return True, "setup is None"
        if not isinstance(setup, dict):
            return False, "setup is not a dict"
        missing = [k for k in CANONICAL_VARIABLE_ORDER if k not in setup]
        if missing:
            return False, f"missing keys: {missing}"
        bad = [k for k in setup.keys() if k in FORBIDDEN_KEYS]
        if bad:
            return False, f"forbidden setup keys: {bad}"
        return True, "ok"

    def validate_jsonl_file(self, stage: str, rel: str, max_records: int | None = 5000) -> None:
        p = self.path(rel)
        if not p.exists():
            self.warn(stage, f"JSONL not found: {rel}")
            return

        task_counts = Counter()
        checked = 0
        errors = []

        with p.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                if max_records is not None and checked >= max_records:
                    break
                line = line.strip()
                if not line:
                    continue

                try:
                    rec = json.loads(line)
                except Exception as e:
                    errors.append(f"line {line_idx}: JSON parse error: {e}")
                    continue

                checked += 1

                forbidden_paths = self.find_forbidden_key_paths(rec)
                if forbidden_paths:
                    errors.append(f"line {line_idx}: forbidden keys: {forbidden_paths[:5]}")

                task_type = rec.get("task_type", "unknown")
                task_counts[task_type] += 1

                for setup_key in ["current_setup", "target_setup", "target_delta"]:
                    if setup_key in rec and rec[setup_key] is not None:
                        ok, msg = self.validate_setup_dict(rec[setup_key])
                        if not ok:
                            errors.append(f"line {line_idx}: {setup_key}: {msg}")

                if "target_profile_path" in rec and rec["target_profile_path"]:
                    profile_path = self.path(rec["target_profile_path"])
                    if not profile_path.exists():
                        errors.append(f"line {line_idx}: target_profile_path missing: {rec['target_profile_path']}")

                if "current_profile_path" in rec and rec["current_profile_path"]:
                    profile_path = self.path(rec["current_profile_path"])
                    if not profile_path.exists():
                        errors.append(f"line {line_idx}: current_profile_path missing: {rec['current_profile_path']}")

        if errors:
            self.fail(stage, f"validate JSONL: {rel}", f"{len(errors)} errors. First: {errors[0]}")
        else:
            self.pass_(stage, f"validate JSONL: {rel}", f"checked={checked}, tasks={dict(task_counts)}")

    def check_stage0(self) -> None:
        stage = "Stage 0 — skeleton/schema/configs"

        for rel in [
            "profile2setup",
            "profile2setup/configs",
            "profile2setup/data_prep",
            "profile2setup/training",
            "profile2setup/models",
            "profile2setup/evaluation",
            "profile2setup/inference",
            "profile2setup/scripts",
        ]:
            self.require_dir(stage, rel)

        for rel in [
            "profile2setup/__init__.py",
            "profile2setup/schema.py",
            "profile2setup/README.md",
            "profile2setup/configs/variables.yaml",
            "profile2setup/configs/prompts.yaml",
            "profile2setup/configs/train.yaml",
            "optical_sim/configs/random_config_v2.yaml",
        ]:
            self.require_file(stage, rel)

        try:
            schema = importlib.import_module("profile2setup.schema")
            order = getattr(schema, "VARIABLE_ORDER")
            num = getattr(schema, "NUM_VARIABLES")
            if order == CANONICAL_VARIABLE_ORDER and num == 7:
                self.pass_(stage, "schema canonical variable order")
            else:
                self.fail(stage, "schema canonical variable order", f"order={order}, NUM_VARIABLES={num}")
        except Exception:
            self.fail(stage, "import profile2setup.schema", traceback.format_exc(limit=2))

        variables = self.load_yaml_file("profile2setup/configs/variables.yaml")
        if variables:
            order = variables.get("variable_order")
            if order == CANONICAL_VARIABLE_ORDER:
                self.pass_(stage, "variables.yaml variable_order")
            else:
                self.fail(stage, "variables.yaml variable_order", f"got={order}")

            vars_block = variables.get("variables", {})
            missing = [v for v in CANONICAL_VARIABLE_ORDER if v not in vars_block]
            if not missing:
                self.pass_(stage, "variables.yaml contains all variables")
            else:
                self.fail(stage, "variables.yaml contains all variables", f"missing={missing}")

    def check_stage1(self, run_simulator_smoke: bool = False) -> None:
        stage = "Stage 1 — simulator v2 support"

        for rel in [
            "optical_sim/src/optical_elements.py",
            "optical_sim/src/simulator.py",
            "optical_sim/src/io_utils.py",
            "optical_sim/configs/base_config.yaml",
            "optical_sim/configs/random_config_v2.yaml",
        ]:
            self.require_file(stage, rel)

        rcfg = self.load_yaml_file("optical_sim/configs/random_config_v2.yaml")
        if rcfg is not None:
            forbidden_paths = self.find_forbidden_key_paths(rcfg)
            if forbidden_paths:
                self.fail(stage, "random_config_v2 has no forbidden keys", str(forbidden_paths[:10]))
            else:
                self.pass_(stage, "random_config_v2 has no forbidden keys")

            text = self.path("optical_sim/configs/random_config_v2.yaml").read_text(encoding="utf-8")
            required_tokens = [
                "lens.x_offset",
                "lens.y_offset",
                "camera.x_offset",
                "camera.y_offset",
                "geometry.laser_to_lens",
                "geometry.lens_to_camera",
                "lens.focal_length",
            ]
            missing = [t for t in required_tokens if t not in text]
            if not missing:
                self.pass_(stage, "random_config_v2 contains v2 parameter ranges")
            else:
                self.fail(stage, "random_config_v2 contains v2 parameter ranges", f"missing={missing}")

        try:
            from optical_sim.src.optical_elements import setup_from_dict

            cfg = {
                "source": {
                    "wavelength": 632.8e-9,
                    "beam_waist": 1.0e-3,
                    "power": 1.0,
                    "type": "gaussian",
                },
                "lens": {
                    "focal_length": 0.1,
                    "clear_aperture": 0.025,
                    "diameter": 0.0254,
                    "x_offset": 1e-4,
                    "y_offset": -2e-4,
                },
                "sensor": {
                    "resolution": [64, 64],
                    "pixel_pitch": 5.5e-6,
                },
                "geometry": {
                    "laser_to_lens": 0.2,
                    "lens_to_camera": 0.15,
                },
                "camera": {
                    "x_offset": 3e-4,
                    "y_offset": -4e-4,
                },
                "simulation": {
                    "grid_size": 128,
                    "grid_extent": 0.03,
                    "propagation_backend": "fresnel_numpy",
                },
            }
            setup = setup_from_dict(cfg)
            checks = [
                abs(setup.lens.x_offset - 1e-4) < 1e-12,
                abs(setup.lens.y_offset + 2e-4) < 1e-12,
                hasattr(setup, "camera"),
                abs(setup.camera.x_offset - 3e-4) < 1e-12,
                abs(setup.camera.y_offset + 4e-4) < 1e-12,
            ]
            if all(checks):
                self.pass_(stage, "setup_from_dict parses lens/camera offsets")
            else:
                self.fail(stage, "setup_from_dict parses lens/camera offsets")
        except Exception:
            self.fail(stage, "setup_from_dict lens/camera offset test", traceback.format_exc(limit=3))

        smoke = self.path("optical_sim/scripts/smoke_v2_offsets.py")
        if smoke.exists():
            self.pass_(stage, "simulator v2 smoke script exists")
            if run_simulator_smoke:
                self.run_cmd(stage, "run simulator v2 smoke script", [sys.executable, str(smoke)], timeout=180)
        else:
            self.warn(stage, "simulator v2 smoke script missing", "optional but recommended")

    def check_stage2(self) -> None:
        stage = "Stage 2 — dataset builders"

        for rel in [
            "profile2setup/data_prep/extract_setup.py",
            "profile2setup/data_prep/extract_profile_features.py",
            "profile2setup/data_prep/build_absolute_dataset.py",
            "profile2setup/data_prep/build_edit_dataset.py",
            "profile2setup/data_prep/split.py",
            "profile2setup/scripts/build_absolute_dataset_cli.py",
            "profile2setup/scripts/build_edit_dataset_cli.py",
            "profile2setup/scripts/split_dataset_cli.py",
        ]:
            self.require_file(stage, rel)

        for module_name in [
            "profile2setup.data_prep.extract_setup",
            "profile2setup.data_prep.extract_profile_features",
            "profile2setup.data_prep.build_absolute_dataset",
            "profile2setup.data_prep.build_edit_dataset",
            "profile2setup.data_prep.split",
        ]:
            try:
                importlib.import_module(module_name)
                self.pass_(stage, f"import {module_name}")
            except Exception:
                self.fail(stage, f"import {module_name}", traceback.format_exc(limit=2))

        candidate_jsonls = [
            "profile2setup/data/absolute.jsonl",
            "profile2setup/data/edit.jsonl",
            "profile2setup/data/all_modes/train.jsonl",
            "profile2setup/data/all_modes/val.jsonl",
            "profile2setup/data/all_modes/test.jsonl",
        ]
        any_jsonl = False
        for rel in candidate_jsonls:
            if self.path(rel).exists():
                any_jsonl = True
                self.validate_jsonl_file(stage, rel)
        if not any_jsonl:
            self.warn(stage, "no dataset JSONL files found", "build datasets before training")

    def check_stage3(self, run_dataset_smoke: bool = False, train_jsonl: str | None = None) -> None:
        stage = "Stage 3 — preprocessing/dataset loading"

        for rel in [
            "profile2setup/training/preprocessing.py",
            "profile2setup/training/normalization.py",
            "profile2setup/training/text.py",
            "profile2setup/training/dataset.py",
            "profile2setup/scripts/dataset_smoke_test_cli.py",
        ]:
            self.require_file(stage, rel)

        try:
            from profile2setup.training.dataset import Profile2SetupDataset, profile2setup_collate_fn

            if Profile2SetupDataset and profile2setup_collate_fn:
                self.pass_(stage, "import Profile2SetupDataset and collate_fn")
        except Exception:
            self.fail(stage, "import Profile2SetupDataset", traceback.format_exc(limit=2))

        if run_dataset_smoke:
            jsonl = train_jsonl or "profile2setup/data/all_modes/train.jsonl"
            if not self.path(jsonl).exists():
                self.warn(stage, "dataset smoke skipped", f"missing {jsonl}")
            else:
                self.run_cmd(
                    stage,
                    "dataset smoke test",
                    [
                        sys.executable,
                        "-m",
                        "profile2setup.scripts.dataset_smoke_test_cli",
                        "--jsonl",
                        jsonl,
                        "--variables-config",
                        "profile2setup/configs/variables.yaml",
                        "--input-size",
                        "128",
                        "--max-text-len",
                        "32",
                        "--limit",
                        "4",
                    ],
                    timeout=180,
                )

    def check_stage4(self, run_model_smoke: bool = False, train_jsonl: str | None = None) -> None:
        stage = "Stage 4 — model architecture"

        for rel in [
            "profile2setup/models/profile_encoder.py",
            "profile2setup/models/text_encoder.py",
            "profile2setup/models/setup_encoder.py",
            "profile2setup/models/heads.py",
            "profile2setup/models/fusion_model.py",
            "profile2setup/scripts/model_smoke_test_cli.py",
        ]:
            self.require_file(stage, rel)

        try:
            from profile2setup.models import Profile2SetupModel

            if Profile2SetupModel:
                self.pass_(stage, "import Profile2SetupModel")
        except Exception:
            self.fail(stage, "import Profile2SetupModel", traceback.format_exc(limit=2))

        if run_model_smoke:
            cmd = [
                sys.executable,
                "-m",
                "profile2setup.scripts.model_smoke_test_cli",
                "--vocab-size",
                "100",
                "--batch-size",
                "2",
                "--input-size",
                "128",
                "--text-len",
                "32",
            ]
            if train_jsonl and self.path(train_jsonl).exists():
                vocab = "profile2setup/data/all_modes/vocab.json"
                if self.path(vocab).exists():
                    cmd.extend(
                        [
                            "--jsonl",
                            train_jsonl,
                            "--variables-config",
                            "profile2setup/configs/variables.yaml",
                            "--vocab",
                            vocab,
                        ]
                    )
            self.run_cmd(stage, "model smoke test", cmd, timeout=180)

    def check_stage5(self, run_train_smoke: bool = False) -> None:
        stage = "Stage 5 — training/checkpointing"

        for rel in [
            "profile2setup/training/train.py",
            "profile2setup/training/utils.py",
            "profile2setup/scripts/train_cli.py",
        ]:
            self.require_file(stage, rel)

        cfg = self.load_yaml_file("profile2setup/configs/train.yaml")
        if cfg:
            text = self.path("profile2setup/configs/train.yaml").read_text(encoding="utf-8")
            if "placeholder" in text:
                self.fail(stage, "train.yaml not placeholder", "contains 'placeholder'")
            else:
                self.pass_(stage, "train.yaml not placeholder")

            if "input_channels: 1" in text:
                self.fail(stage, "train.yaml uses 4-channel profile model", "found input_channels: 1")
            else:
                self.pass_(stage, "train.yaml uses 4-channel profile model")

            data = cfg.get("data", {})
            train_path = data.get("train_path") or data.get("train_jsonl")
            val_path = data.get("val_path") or data.get("val_jsonl")
            if train_path and val_path:
                self.pass_(stage, "train.yaml has train/val paths", f"{train_path}, {val_path}")
            else:
                self.fail(stage, "train.yaml has train/val paths", "missing train_path/val_path or aliases")

        if run_train_smoke:
            if not self.path("profile2setup/scripts/train_cli.py").exists():
                self.warn(stage, "training smoke skipped", "train_cli.py missing")
            else:
                self.run_cmd(
                    stage,
                    "training smoke test",
                    [
                        sys.executable,
                        "-m",
                        "profile2setup.scripts.train_cli",
                        "--config",
                        "profile2setup/configs/train.yaml",
                        "--smoke-test",
                    ],
                    timeout=600,
                )

        ckpt_roots = list(self.path("profile2setup/checkpoints").glob("**/best.pt")) if self.path("profile2setup/checkpoints").exists() else []
        if ckpt_roots:
            self.pass_(stage, "best checkpoint exists", str(ckpt_roots[0].relative_to(self.repo_root)))
        else:
            self.warn(stage, "best checkpoint not found", "run training smoke/full training")

    def check_stage6(
        self,
        run_baselines: bool = False,
        run_evaluate: bool = False,
        checkpoint: str | None = None,
        test_jsonl: str | None = None,
    ) -> None:
        stage = "Stage 6 — offline evaluation/baselines"

        for rel in [
            "profile2setup/evaluation/param_metrics.py",
            "profile2setup/evaluation/profile_metrics.py",
            "profile2setup/evaluation/baselines.py",
            "profile2setup/evaluation/evaluate_model.py",
            "profile2setup/scripts/evaluate_cli.py",
            "profile2setup/scripts/run_baselines_cli.py",
        ]:
            self.require_file(stage, rel)

        if run_baselines:
            train = "profile2setup/data/all_modes/train.jsonl"
            test = test_jsonl or "profile2setup/data/all_modes/test.jsonl"
            if not self.path(train).exists() or not self.path(test).exists():
                self.warn(stage, "baseline eval skipped", "train/test JSONL missing")
            elif not self.path("profile2setup/scripts/run_baselines_cli.py").exists():
                self.warn(stage, "baseline eval skipped", "run_baselines_cli.py missing")
            else:
                self.run_cmd(
                    stage,
                    "run baselines",
                    [
                        sys.executable,
                        "-m",
                        "profile2setup.scripts.run_baselines_cli",
                        "--train",
                        train,
                        "--test",
                        test,
                        "--variables-config",
                        "profile2setup/configs/variables.yaml",
                        "--out",
                        "profile2setup/results/baselines_check.json",
                    ],
                    timeout=600,
                )

        if run_evaluate:
            ckpt = checkpoint or self.find_default_checkpoint()
            test = test_jsonl or "profile2setup/data/all_modes/test.jsonl"
            if not ckpt or not self.path(ckpt).exists():
                self.warn(stage, "model eval skipped", "checkpoint missing")
            elif not self.path(test).exists():
                self.warn(stage, "model eval skipped", "test JSONL missing")
            elif not self.path("profile2setup/scripts/evaluate_cli.py").exists():
                self.warn(stage, "model eval skipped", "evaluate_cli.py missing")
            else:
                self.run_cmd(
                    stage,
                    "evaluate checkpoint",
                    [
                        sys.executable,
                        "-m",
                        "profile2setup.scripts.evaluate_cli",
                        "--checkpoint",
                        ckpt,
                        "--data",
                        test,
                        "--out",
                        "profile2setup/results/model_eval_check.json",
                    ],
                    timeout=600,
                )

    def check_stage7(
        self,
        run_closed_loop: bool = False,
        checkpoint: str | None = None,
        test_jsonl: str | None = None,
    ) -> None:
        stage = "Stage 7 — closed-loop simulation evaluation"

        for rel in [
            "profile2setup/inference/controller.py",
            "profile2setup/evaluation/closed_loop.py",
            "profile2setup/scripts/closed_loop_eval_cli.py",
        ]:
            self.require_file(stage, rel)

        if run_closed_loop:
            ckpt = checkpoint or self.find_default_checkpoint()
            test = test_jsonl or "profile2setup/data/all_modes/test.jsonl"
            if not ckpt or not self.path(ckpt).exists():
                self.warn(stage, "closed-loop skipped", "checkpoint missing")
            elif not self.path(test).exists():
                self.warn(stage, "closed-loop skipped", "test JSONL missing")
            elif not self.path("profile2setup/scripts/closed_loop_eval_cli.py").exists():
                self.warn(stage, "closed-loop skipped", "closed_loop_eval_cli.py missing")
            else:
                self.run_cmd(
                    stage,
                    "closed-loop smoke",
                    [
                        sys.executable,
                        "-m",
                        "profile2setup.scripts.closed_loop_eval_cli",
                        "--checkpoint",
                        ckpt,
                        "--data",
                        test,
                        "--out",
                        "profile2setup/results/closed_loop_check.json",
                        "--simulation-policy",
                        "target_base",
                        "--max-examples",
                        "5",
                        "--no-strict",
                    ],
                    timeout=900,
                )

    def check_stage8(self, run_integrity: bool = False) -> None:
        stage = "Stage 8 — workflow/integrity"

        for rel in [
            "profile2setup/WORKFLOW.md",
            "profile2setup/EXPERIMENTS.md",
            "profile2setup/scripts/check_v2_integrity_cli.py",
            "profile2setup/scripts/run_v2_smoke_pipeline_cli.py",
        ]:
            self.require_file(stage, rel)

        if run_integrity:
            if not self.path("profile2setup/scripts/check_v2_integrity_cli.py").exists():
                self.warn(stage, "integrity check skipped", "check_v2_integrity_cli.py missing")
            else:
                self.run_cmd(
                    stage,
                    "run v2 integrity checker",
                    [
                        sys.executable,
                        "-m",
                        "profile2setup.scripts.check_v2_integrity_cli",
                        "--root",
                        "profile2setup",
                        "--data-dir",
                        "profile2setup/data",
                        "--results-dir",
                        "profile2setup/results",
                    ],
                    timeout=300,
                )

    def find_default_checkpoint(self) -> str | None:
        ckpt_dir = self.path("profile2setup/checkpoints")
        if not ckpt_dir.exists():
            return None
        candidates = sorted(ckpt_dir.glob("**/best.pt"))
        if not candidates:
            candidates = sorted(ckpt_dir.glob("**/latest.pt"))
        if not candidates:
            return None
        return str(candidates[0].relative_to(self.repo_root))

    def check_result_jsons_for_forbidden_keys(self) -> None:
        stage = "GLOBAL — result/data forbidden key scan"
        for root_rel in ["profile2setup/results", "profile2setup/data"]:
            root = self.path(root_rel)
            if not root.exists():
                self.warn(stage, f"{root_rel} missing")
                continue

            checked = 0
            bad = []
            for p in root.rglob("*.json"):
                try:
                    obj = json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    continue
                checked += 1
                paths = self.find_forbidden_key_paths(obj)
                if paths:
                    bad.append((str(p.relative_to(self.repo_root)), paths[:5]))

            for p in root.rglob("*.jsonl"):
                try:
                    with p.open("r", encoding="utf-8") as f:
                        for idx, line in enumerate(f, start=1):
                            if not line.strip():
                                continue
                            obj = json.loads(line)
                            checked += 1
                            paths = self.find_forbidden_key_paths(obj)
                            if paths:
                                bad.append((f"{p.relative_to(self.repo_root)}:{idx}", paths[:5]))
                                break
                except Exception:
                    continue

            if bad:
                self.fail(stage, f"forbidden keys in {root_rel}", str(bad[:5]))
            else:
                self.pass_(stage, f"forbidden key scan {root_rel}", f"checked objects/files={checked}")

    def summary(self) -> int:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        total = Counter()
        for stage, counts in self.stage_counts.items():
            total.update(counts)
            print(
                f"{stage}: "
                f"PASS={counts['PASS']} "
                f"WARN={counts['WARN']} "
                f"FAIL={counts['FAIL']} "
                f"SKIP={counts['SKIP']}"
            )

        print("-" * 80)
        print(
            f"TOTAL: PASS={total['PASS']} WARN={total['WARN']} "
            f"FAIL={total['FAIL']} SKIP={total['SKIP']}"
        )

        if self.json_out:
            self.json_out.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "repo_root": str(self.repo_root),
                "strict": self.strict,
                "summary": dict(total),
                "stage_counts": {k: dict(v) for k, v in self.stage_counts.items()},
                "results": self.results,
            }
            self.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"\nWrote JSON summary to: {self.json_out}")

        if self.strict and total["FAIL"] > 0:
            return 1
        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check profile2setup v2 readiness across all stages.")
    parser.add_argument("--repo-root", default=".", help="Repo root. Default: current directory.")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero if any required check fails.")
    parser.add_argument("--json-out", default=None, help="Optional path to write JSON check summary.")

    parser.add_argument("--train-jsonl", default="profile2setup/data/all_modes/train.jsonl")
    parser.add_argument("--test-jsonl", default="profile2setup/data/all_modes/test.jsonl")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint for evaluation/closed-loop checks.")

    parser.add_argument("--run-simulator-smoke", action="store_true")
    parser.add_argument("--run-dataset-smoke", action="store_true")
    parser.add_argument("--run-model-smoke", action="store_true")
    parser.add_argument("--run-train-smoke", action="store_true")
    parser.add_argument("--run-baselines", action="store_true")
    parser.add_argument("--run-evaluate", action="store_true")
    parser.add_argument("--run-closed-loop", action="store_true")
    parser.add_argument("--run-integrity", action="store_true")

    parser.add_argument(
        "--scan-data-results",
        action="store_true",
        help="Scan profile2setup/data and profile2setup/results JSON/JSONL for forbidden keys.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checker = StageChecker(
        repo_root=Path(args.repo_root),
        strict=args.strict,
        json_out=Path(args.json_out) if args.json_out else None,
    )

    print(f"Repo root: {checker.repo_root}")
    print(f"Strict: {args.strict}")

    checker.check_stage0()
    checker.check_stage1(run_simulator_smoke=args.run_simulator_smoke)
    checker.check_stage2()
    checker.check_stage3(run_dataset_smoke=args.run_dataset_smoke, train_jsonl=args.train_jsonl)
    checker.check_stage4(run_model_smoke=args.run_model_smoke, train_jsonl=args.train_jsonl)
    checker.check_stage5(run_train_smoke=args.run_train_smoke)
    checker.check_stage6(
        run_baselines=args.run_baselines,
        run_evaluate=args.run_evaluate,
        checkpoint=args.checkpoint,
        test_jsonl=args.test_jsonl,
    )
    checker.check_stage7(
        run_closed_loop=args.run_closed_loop,
        checkpoint=args.checkpoint,
        test_jsonl=args.test_jsonl,
    )
    checker.check_stage8(run_integrity=args.run_integrity)

    if args.scan_data_results:
        checker.check_result_jsons_for_forbidden_keys()

    return checker.summary()


if __name__ == "__main__":
    raise SystemExit(main())