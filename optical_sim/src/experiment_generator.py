"""
experiment_generator.py
Produce lists of OpticalSetup objects from sweep or random-sampling configs.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import List, Tuple

import yaml
import numpy as np

from .optical_elements import OpticalSetup, setup_from_dict


def _set_nested(d: dict, dotpath: str, value) -> dict:
    """Set a value in a nested dict using a dot-separated path."""
    keys = dotpath.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur[k]
    cur[keys[-1]] = value
    return d


def _resolve_path(path: str, *, parent_config_path: str | None = None) -> Path:
    """Resolve config paths robustly across different launch working directories."""
    requested = Path(path).expanduser()
    if requested.is_absolute():
        return requested

    candidates: List[Path] = []
    if parent_config_path:
        parent = Path(parent_config_path).resolve().parent
        candidates.append(parent / requested)
        if parent.parent != parent:
            candidates.append(parent.parent / requested)
    # Fallback: resolve relative to the optical_sim package root.
    package_root = Path(__file__).resolve().parents[1]
    candidates.append(package_root / requested)
    candidates.append(Path.cwd() / requested)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    tried = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(
        f"Could not resolve config path '{path}'. Tried:\n{tried}"
    )


def load_yaml(path: str, *, parent_config_path: str | None = None) -> dict:
    resolved = _resolve_path(path, parent_config_path=parent_config_path)
    with open(resolved, "r") as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────────────────────
# Sweep generator
# ──────────────────────────────────────────────────────────────

def generate_sweep_experiments(sweep_config_path: str) -> List[Tuple[str, OpticalSetup]]:
    """
    Read a sweep YAML and return a list of (run_name, OpticalSetup) pairs.
    """
    scfg = load_yaml(sweep_config_path)
    base = load_yaml(scfg["base_config"], parent_config_path=sweep_config_path)

    experiments: List[Tuple[str, OpticalSetup]] = []
    for sweep in scfg["sweeps"]:
        name = sweep["name"]
        param = sweep["param_path"]
        for i, val in enumerate(sweep["values"]):
            cfg = copy.deepcopy(base)
            _set_nested(cfg, param, val)
            run_name = f"{name}_{i:03d}"
            experiments.append((run_name, setup_from_dict(cfg)))
    return experiments


# ──────────────────────────────────────────────────────────────
# Random-sampling generator
# ──────────────────────────────────────────────────────────────

def generate_random_experiments(random_config_path: str) -> List[Tuple[str, OpticalSetup]]:
    """
    Read a random-sampling YAML and return (run_name, OpticalSetup) pairs.
    """
    rcfg = load_yaml(random_config_path)
    base = load_yaml(rcfg["base_config"], parent_config_path=random_config_path)
    n = rcfg["num_samples"]
    seed = rcfg.get("random_seed", 42)

    rng = np.random.default_rng(seed)
    experiments: List[Tuple[str, OpticalSetup]] = []

    for i in range(n):
        cfg = copy.deepcopy(base)
        for param_path, (lo, hi) in rcfg["parameter_ranges"].items():
            val = rng.uniform(lo, hi)
            _set_nested(cfg, param_path, float(val))
        experiments.append((f"rand_{i:05d}", setup_from_dict(cfg)))

    return experiments
