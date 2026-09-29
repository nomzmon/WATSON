"""WICS computation (thesis Section 5.2.1, Equation 5.1).

Weights come from configs/rubrics/wics.yaml so the weight-variance
experiment can swap in its equal-weight and inverted variants.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from watson.common.config import load_yaml

WICS_CRITERIA = ("OOP", "TR", "HC", "GIP")


def load_wics_weights(path: str | Path = "configs/rubrics/wics.yaml") -> dict[str, float]:
    weights = load_yaml(path)["weights"]
    if set(weights) != set(WICS_CRITERIA):
        raise ValueError(f"WICS weights must cover exactly {WICS_CRITERIA}, got {sorted(weights)}")
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"WICS weights must sum to 1.0, got {total}")
    return {code: float(weights[code]) for code in WICS_CRITERIA}


def compute_wics(scores: Mapping[str, float], weights: Mapping[str, float]) -> float:
    # Rounded so float error can't push a boundary score (e.g. exactly 8.0) into a lower band.
    return round(sum(weights[code] * scores[code] for code in WICS_CRITERIA), 4)