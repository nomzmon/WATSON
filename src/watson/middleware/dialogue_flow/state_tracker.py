"""Dialogue state tracking (thesis Sections 3.2.1 and 5.2.2).

The judge proposes the stage the investigation has reached, but the official
stage only ever advances one step per turn and never moves back. This keeps
the conversation from skipping ahead or reopening completed stages.
"""

from __future__ import annotations

from watson.common.schemas import DialogueStage

STAGE_ORDER = list(DialogueStage)


def next_stage(current: DialogueStage, proposed: DialogueStage) -> DialogueStage:
    position = STAGE_ORDER.index(current)
    if STAGE_ORDER.index(proposed) > position:
        return STAGE_ORDER[position + 1]
    return current