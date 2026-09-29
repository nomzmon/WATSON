"""Input Constraint Engine (thesis Section 5.2.1).

Before a user message reaches the Prompt Builder, the judge scores it on the
four WICS criteria. The engine only evaluates; the Decision Layer
(middleware/decision/decision.py) decides what to do with the score.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel, Field

from watson.common.config import load_text
from watson.common.schemas import CriterionScore, DialogueStage, HistoryTurn, RubricName, RubricScore
from watson.llm.base import LLMClient
from watson.middleware.input_constraint.scoring import WICS_CRITERIA, compute_wics, load_wics_weights


class CriterionJudgement(BaseModel):
    reasoning: str
    score: int = Field(ge=1, le=10)


class WICSJudgement(BaseModel):
    OOP: CriterionJudgement
    TR: CriterionJudgement
    HC: CriterionJudgement
    GIP: CriterionJudgement


class InputConstraintEngine:
    def __init__(
        self,
        judge: LLMClient,
        weights: Mapping[str, float],
        prompt_template: str,
        history_turns: int = 6,
    ) -> None:
        self.judge = judge
        self.weights = dict(weights)
        self.prompt_template = prompt_template
        self.history_turns = history_turns

    @classmethod
    def from_config(
        cls,
        judge: LLMClient,
        rubric_path: str | Path = "configs/rubrics/wics.yaml",
        prompt_path: str | Path = "prompts/judge/wics.txt",
        history_turns: int = 6,
    ) -> InputConstraintEngine:
        return cls(judge, load_wics_weights(rubric_path), load_text(prompt_path), history_turns)

    def build_prompt(
        self,
        user_message: str,
        history: Sequence[HistoryTurn] = (),
        dialogue_stage: DialogueStage | None = None,
        case_context: str | None = None,
    ) -> str:
        recent = history[-self.history_turns :] if self.history_turns else []
        speaker = {"user": "User", "holmes": "Holmes"}
        return self.prompt_template.format(
            user_message=user_message,
            history="\n".join(f"{speaker[t.role]}: {t.text}" for t in recent) or "(no previous turns)",
            dialogue_stage=dialogue_stage.value if dialogue_stage else "not yet tracked",
            case_context=case_context or "not provided",
        )

    def evaluate(
        self,
        user_message: str,
        history: Sequence[HistoryTurn] = (),
        dialogue_stage: DialogueStage | None = None,
        case_context: str | None = None,
    ) -> RubricScore:
        prompt = self.build_prompt(user_message, history, dialogue_stage, case_context)
        judgement = self.judge.generate_structured(prompt, WICSJudgement)
        criteria = {
            code: CriterionScore(
                code=code,
                score=getattr(judgement, code).score,
                rationale=getattr(judgement, code).reasoning,
            )
            for code in WICS_CRITERIA
        }
        total = compute_wics({code: c.score for code, c in criteria.items()}, self.weights)
        return RubricScore(rubric=RubricName.WICS, criteria=criteria, weighted_total=total)