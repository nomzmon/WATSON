"""Shared pieces of the three judge rubrics (WICS, DFMS, OVS).

Each rubric scores a fixed set of criteria from 1 to 10 and combines them
with weights loaded from configs/rubrics/<rubric>.yaml.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel, Field

from watson.common.config import load_yaml


class CriterionJudgement(BaseModel):
    reasoning: str
    score: int = Field(ge=1, le=10)


def load_rubric_weights(path: str | Path, criteria: Sequence[str], name: str) -> dict[str, float]:
    weights = load_yaml(path)["weights"]
    if set(weights) != set(criteria):
        raise ValueError(f"{name} weights must cover exactly {tuple(criteria)}, got {sorted(weights)}")
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"{name} weights must sum to 1.0, got {total}")
    return {code: float(weights[code]) for code in criteria}


def weighted_score(scores: Mapping[str, float], weights: Mapping[str, float]) -> float:
    # Rounded so float error can't push a boundary score (e.g. exactly 8.0) into a lower band.
    return round(sum(weight * scores[code] for code, weight in weights.items()), 4)