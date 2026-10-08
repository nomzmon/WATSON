"""Dialogue Flow Evaluation (thesis Section 5.2.2).

Before Holmes replies, the judge scores the conversation so far, plus the
visitor's new message, on CST, NP, CC and TBC, and proposes the stage the
investigation has reached.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel

from watson.common.config import load_text
from watson.common.rubric import CriterionJudgement, load_rubric_weights, weighted_score
from watson.common.schemas import CriterionScore, DialogueStage, RubricName, RubricScore
from watson.llm.base import LLMClient
from watson.middleware.prompt_builder.builder import StructuredPrompt, render_history

DFMS_CRITERIA = ("CST", "NP", "CC", "TBC")


class FlowJudgement(BaseModel):
    CST: CriterionJudgement
    NP: CriterionJudgement
    CC: CriterionJudgement
    TBC: CriterionJudgement
    stage: DialogueStage


class FlowEvaluation(BaseModel):
    dfms: RubricScore
    proposed_stage: DialogueStage


def load_dfms_weights(path: str | Path = "configs/rubrics/dfms.yaml") -> dict[str, float]:
    return load_rubric_weights(path, DFMS_CRITERIA, "DFMS")


class FlowEvaluator:
    def __init__(self, judge: LLMClient, weights: Mapping[str, float], prompt_template: str) -> None:
        self.judge = judge
        self.weights = dict(weights)
        self.prompt_template = prompt_template

    @classmethod
    def from_config(
        cls,
        judge: LLMClient,
        rubric_path: str | Path = "configs/rubrics/dfms.yaml",
        prompt_path: str | Path = "prompts/judge/dfms.txt",
    ) -> FlowEvaluator:
        return cls(judge, load_dfms_weights(rubric_path), load_text(prompt_path))

    def build_prompt(self, prompt: StructuredPrompt) -> str:
        return self.prompt_template.format(
            user_message=prompt.user_message,
            history=render_history(prompt.history),
            dialogue_stage=prompt.dialogue_stage.value if prompt.dialogue_stage else "not yet tracked",
            case_context=prompt.case_context or "No case has been presented yet.",
        )

    def evaluate(self, prompt: StructuredPrompt) -> FlowEvaluation:
        judgement = self.judge.generate_structured(self.build_prompt(prompt), FlowJudgement)
        criteria = {
            code: CriterionScore(
                code=code,
                score=getattr(judgement, code).score,
                rationale=getattr(judgement, code).reasoning,
            )
            for code in DFMS_CRITERIA
        }
        total = weighted_score({code: c.score for code, c in criteria.items()}, self.weights)
        return FlowEvaluation(
            dfms=RubricScore(rubric=RubricName.DFMS, criteria=criteria, weighted_total=total),
            proposed_stage=judgement.stage,
        )