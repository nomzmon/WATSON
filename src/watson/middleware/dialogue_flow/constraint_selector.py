"""Constraint selection from the Dialogue Constraint Library (thesis Section 5.2.2).

Each DFMS criterion maps to one constraint category. The current stage's
dialogue-state constraint is always included. When the DFMS is low, the
categories of the weak criteria are added, and a very low DFMS raises them to
strength level 2. Level 3 is left for Output Validation's constraint refinement.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel

from watson.common.config import load_yaml, resolve_path
from watson.common.schemas import Constraint, ConstraintCategory, DialogueStage, RubricScore

CRITERION_CATEGORY = {
    "CST": ConstraintCategory.DIALOGUE_STATE,
    "NP": ConstraintCategory.NARRATIVE_PROGRESSION,
    "CC": ConstraintCategory.CONTEXT_CONTINUITY,
    "TBC": ConstraintCategory.TOPIC_BOUNDARY,
}


class SelectionThresholds(BaseModel):
    guidance_below: float
    weak_criterion_below: float
    strong_below: float


class SelectedConstraint(BaseModel):
    id: str
    category: ConstraintCategory
    level: int
    text: str


def load_selection_thresholds(path: str | Path = "configs/rubrics/dfms.yaml") -> SelectionThresholds:
    return SelectionThresholds.model_validate(load_yaml(path)["selection"])


class ConstraintLibrary:
    def __init__(self, constraints: Sequence[Constraint]) -> None:
        ids = [c.id for c in constraints]
        if len(ids) != len(set(ids)):
            raise ValueError("constraint ids must be unique")
        self.constraints = list(constraints)

    @classmethod
    def from_dir(
        cls, path: str | Path = "src/watson/middleware/dialogue_flow/constraint_library"
    ) -> ConstraintLibrary:
        constraints = []
        for file in sorted(resolve_path(path).glob("*.yaml")):
            data = load_yaml(file)
            constraints += [Constraint(category=data["category"], **c) for c in data["constraints"]]
        return cls(constraints)

    def find(self, category: ConstraintCategory, stage: DialogueStage) -> list[Constraint]:
        return [c for c in self.constraints if c.category == category and c.applies_to(stage)]


def select_constraints(
    dfms: RubricScore,
    stage: DialogueStage,
    library: ConstraintLibrary,
    thresholds: SelectionThresholds,
) -> list[SelectedConstraint]:
    weak: list[str] = []
    if dfms.weighted_total < thresholds.guidance_below:
        weak = [code for code, c in dfms.criteria.items() if c.score < thresholds.weak_criterion_below]
        weak = weak or [min(dfms.criteria.values(), key=lambda c: c.score).code]
    strength = 2 if dfms.weighted_total < thresholds.strong_below else 1

    levels = {ConstraintCategory.DIALOGUE_STATE: max(strength, 2) if "CST" in weak else 1}
    for code in weak:
        levels.setdefault(CRITERION_CATEGORY[code], strength)

    selected = []
    for category, level in levels.items():
        for constraint in library.find(category, stage):
            applied = min(level, constraint.max_level)
            selected.append(
                SelectedConstraint(id=constraint.id, category=category, level=applied, text=constraint.text_for(applied))
            )
    return selected