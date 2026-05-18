"""
describe.py
Generate natural-language descriptions from beam features using templates.
"""
from __future__ import annotations

import random
from typing import Dict, Any, List

import yaml
from pathlib import Path

_DEFAULT_TEMPLATES_PATH = Path(__file__).parent.parent / "configs" / "templates.yaml"


def load_templates_config(path: str | Path = _DEFAULT_TEMPLATES_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _classify_feature(value: float, levels: Dict[str, list]) -> str:
    """Return the label whose [min, max) range contains value."""
    for label, (lo, hi) in levels.items():
        if lo <= value < hi:
            return label
    # fallback: return last label
    return list(levels.keys())[-1]


def compute_descriptors(features: Dict[str, Any],
                        templates_cfg: dict) -> Dict[str, str]:
    """
    Map numeric features to qualitative labels.

    Returns e.g. {"width": "narrow", "position_x": "centered", ...}
    """
    desc_cfg = templates_cfg["descriptors"]
    label_text = templates_cfg["label_text"]
    result = {}

    for desc_name, desc_info in desc_cfg.items():
        field = desc_info["field"]
        source = desc_info.get("source", "metrics")

        # Get the raw value
        if source == "setup":
            value = features.get(field, 0.0)
        else:
            value = features.get(field, 0.0)

        # Skip percentile-based descriptors (handled at dataset level)
        if desc_info.get("mode") == "percentile":
            result[desc_name] = "moderate"  # placeholder
            continue

        raw_label = _classify_feature(value, desc_info["levels"])
        result[desc_name] = label_text[desc_name][raw_label]

    return result


def generate_descriptions(features: Dict[str, Any],
                          templates_cfg: dict | None = None,
                          num_variants: int | None = None,
                          rng: random.Random | None = None) -> List[str]:
    """
    Generate natural-language descriptions for a single sample.

    Parameters
    ----------
    features : flat feature dict from extract_features
    templates_cfg : loaded templates.yaml (loads default if None)
    num_variants : how many text variants to produce (default from config)
    rng : random.Random instance for reproducibility

    Returns
    -------
    List of description strings.
    """
    if templates_cfg is None:
        templates_cfg = load_templates_config()
    if num_variants is None:
        num_variants = templates_cfg.get("variants_per_sample", 4)
    if rng is None:
        rng = random.Random()

    descriptors = compute_descriptors(features, templates_cfg)
    templates = templates_cfg["templates"]

    # Sample templates without replacement (or with, if more variants than templates)
    chosen = rng.sample(templates, min(num_variants, len(templates)))

    descriptions = []
    for tmpl in chosen:
        try:
            text = tmpl.format(**descriptors)
            descriptions.append(text)
        except KeyError:
            # Template uses a descriptor not computed; skip
            continue

    return descriptions
