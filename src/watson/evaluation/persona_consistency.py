"""Persona consistency score for regression testing.

A judge, separate from the middleware components under test, rates each
generated Holmes reply from 1 to 10 against the persona profile.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, Field

from watson.common.config import load_text
from watson.common.schemas import HistoryTurn, PersonaProfile
from watson.llm.base import LLMClient
from watson.middleware.prompt_builder.builder import render_persona

_SPEAKERS = {"user": "Visitor", "holmes": "Holmes"}


class PersonaJudgement(BaseModel):
    reasoning: str
    score: int = Field(ge=1, le=10)


class PersonaConsistencyJudge:
    def __init__(self, judge: LLMClient, profile: PersonaProfile, template: str, history_turns: int = 6) -> None:
        self.judge = judge
        self.template = template
        self.history_turns = history_turns
        self._persona_text = render_persona(profile)

    @classmethod
    def from_config(
        cls,
        judge: LLMClient,
        profile_path: str | Path,
        prompt_path: str | Path = "prompts/judge/persona_consistency.txt",
    ) -> PersonaConsistencyJudge:
        profile = PersonaProfile.model_validate_json(load_text(profile_path))
        return cls(judge, profile, load_text(prompt_path))

    def score(self, history: Sequence[HistoryTurn], user_message: str, reply: str) -> PersonaJudgement:
        recent = history[-self.history_turns :] if self.history_turns else []
        prompt = self.template.format(
            persona_profile=self._persona_text,
            history="\n".join(f"{_SPEAKERS[t.role]}: {t.text}" for t in recent) or "(This is the start of the conversation.)",
            user_message=user_message,
            reply=reply,
        )
        return self.judge.generate_structured(prompt, PersonaJudgement)