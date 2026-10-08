"""Dialogue Flow Manager (thesis Sections 5.1.4 and 5.2.2).

Receives the structured prompt from the Prompt Builder, evaluates the flow of
the conversation, advances the investigation stage, and adds the selected
dialogue constraints to the prompt before it goes to the Response Generation
Layer.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from watson.common.schemas import DialogueStage, RubricScore
from watson.llm.base import LLMClient
from watson.middleware.dialogue_flow.constraint_selector import (
    ConstraintLibrary,
    SelectedConstraint,
    SelectionThresholds,
    load_selection_thresholds,
    select_constraints,
)
from watson.middleware.dialogue_flow.flow_evaluator import FlowEvaluator
from watson.middleware.dialogue_flow.state_tracker import next_stage
from watson.middleware.prompt_builder.builder import StructuredPrompt


class FlowResult(BaseModel):
    dfms: RubricScore
    proposed_stage: DialogueStage
    stage: DialogueStage
    constraints: list[SelectedConstraint]


class DialogueFlowManager:
    def __init__(
        self,
        evaluator: FlowEvaluator,
        library: ConstraintLibrary,
        thresholds: SelectionThresholds,
    ) -> None:
        self.evaluator = evaluator
        self.library = library
        self.thresholds = thresholds

    @classmethod
    def from_config(
        cls,
        judge: LLMClient,
        rubric_path: str | Path = "configs/rubrics/dfms.yaml",
        prompt_path: str | Path = "prompts/judge/dfms.txt",
        library_dir: str | Path = "src/watson/middleware/dialogue_flow/constraint_library",
    ) -> DialogueFlowManager:
        return cls(
            FlowEvaluator.from_config(judge, rubric_path, prompt_path),
            ConstraintLibrary.from_dir(library_dir),
            load_selection_thresholds(rubric_path),
        )

    def apply(self, prompt: StructuredPrompt) -> tuple[StructuredPrompt, FlowResult]:
        current = prompt.dialogue_stage or DialogueStage.INTRODUCTION
        evaluation = self.evaluator.evaluate(prompt.model_copy(update={"dialogue_stage": current}))
        stage = next_stage(current, evaluation.proposed_stage)
        selected = select_constraints(evaluation.dfms, stage, self.library, self.thresholds)
        guided = prompt.model_copy(
            update={"dialogue_stage": stage, "constraints": [*prompt.constraints, *(c.text for c in selected)]}
        )
        return guided, FlowResult(
            dfms=evaluation.dfms, proposed_stage=evaluation.proposed_stage, stage=stage, constraints=selected
        )