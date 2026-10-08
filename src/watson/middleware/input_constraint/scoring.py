"""WICS computation (thesis Section 5.2.1, Equation 5.1).

Weights come from configs/rubrics/wics.yaml so the weight-variance
experiment can swap in its equal-weight and inverted variants.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from watson.common.rubric import load_rubric_weights, weighted_score

WICS_CRITERIA = ("OOP", "TR", "HC", "GIP")


def load_wics_weights(path: str | Path = "configs/rubrics/wics.yaml") -> dict[str, float]:
    return load_rubric_weights(path, WICS_CRITERIA, "WICS")


def compute_wics(scores: Mapping[str, float], weights: Mapping[str, float]) -> float:
    return weighted_score(scores, weights)