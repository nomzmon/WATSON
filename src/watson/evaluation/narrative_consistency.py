"""Narrative consistency score for regression testing.

A judge, separate from the middleware components under test, rates each
generated Holmes reply from 1 to 10 on whether it stays true to the case and
the conversation, moves the investigation forward, and is naturally paced.
This measures what the Prompt Builder's case context and the Dialogue Flow
Manager are meant to improve; the persona score measures Holmes' character.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from watson.common.config import load_text
from watson.common.rubric import CriterionJudgement
from watson.common.schemas import HistoryTurn
from watson.llm.base import LLMClient
from watson.middleware.prompt_builder.builder import render_history


class NarrativeConsistencyJudge:
    def __init__(self, judge: LLMClient, template: str, history_turns: int = 6) -> None:
        self.judge = judge
        self.template = template
        self.history_turns = history_turns

    @classmethod
    def from_config(
        cls, judge: LLMClient, prompt_path: str | Path = "prompts/judge/narrative_consistency.txt"
    ) -> NarrativeConsistencyJudge:
        return cls(judge, load_text(prompt_path))

    def score(
        self, case_context: str, history: Sequence[HistoryTurn], user_message: str, reply: str
    ) -> CriterionJudgement:
        recent = history[-self.history_turns :] if self.history_turns else []
        prompt = self.template.format(
            case_context=case_context,
            history=render_history(recent),
            user_message=user_message,
            reply=reply,
        )
        return self.judge.generate_structured(prompt, CriterionJudgement)