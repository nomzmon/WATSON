"""Constraint refinement (thesis Sections 5.1.6 and 5.2.3).

When a candidate reply fails validation, the constraints tied to its weak
criteria are strengthened before the next attempt: the first retry uses
strength level 2 and later retries level 3 (the strictest wording, kept for
this module). Constraints already in the prompt at a lower level are replaced
by the stronger wording, so the same rule is never given twice.
"""

from __future__ import annotations

from collections.abc import Sequence

from watson.common.schemas import ConstraintCategory, DialogueStage, RubricScore
from watson.middleware.dialogue_flow.constraint_selector import ConstraintLibrary, SelectedConstraint

# Each OVS criterion maps to the constraint categories that address it; None means every constraint in the category.
CRITERION_CONSTRAINTS: dict[str, list[tuple[ConstraintCategory, set[str] | None]]] = {
    "PCC": [(ConstraintCategory.PERSONA, None)],
    "NC": [(ConstraintCategory.CONTEXT_CONTINUITY, None), (ConstraintCategory.NARRATIVE_PROGRESSION, None)],
    "HWB": [(ConstraintCategory.HISTORICAL, None)],
    "SS": [(ConstraintCategory.PERSONA, {"P-VOICE", "P-SPOKEN-ONLY"})],
}


def weak_criteria(ovs: RubricScore, below: float) -> list[str]:
    weak = [code for code, c in ovs.criteria.items() if c.score < below]
    return weak or [min(ovs.criteria.values(), key=lambda c: c.score).code]


def refinement_level(retry: int) -> int:
    """Strength level for the given retry (1 = first regeneration)."""
    return 2 if retry <= 1 else 3


def refine_constraints(
    current: Sequence[SelectedConstraint],
    weak: Sequence[str],
    level: int,
    stage: DialogueStage,
    library: ConstraintLibrary,
) -> list[SelectedConstraint]:
    """The current constraints plus the weak criteria's constraints at `level`, keeping the stronger of duplicates."""
    merged = {c.id: c for c in current}
    for code in weak:
        for category, ids in CRITERION_CONSTRAINTS[code]:
            for constraint in library.find(category, stage):
                if ids is not None and constraint.id not in ids:
                    continue
                applied = min(level, constraint.max_level)
                if constraint.id not in merged or merged[constraint.id].level < applied:
                    merged[constraint.id] = SelectedConstraint(
                        id=constraint.id, category=category, level=applied, text=constraint.text_for(applied)
                    )
    return list(merged.values())