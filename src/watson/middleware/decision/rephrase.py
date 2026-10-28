"""Automatic rephrasing for the Rephrase decision (thesis Section 5.2.1.1).

A message in the Rephrase band is understandable but only partly fits the
role-play, for example vague, informal or loosely tied to the case. The judge
model rewrites it as a clear message from Holmes' assistant, keeping its
meaning, and Holmes answers the rewrite. The conversation history keeps what
the visitor actually wrote.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel

from watson.common.config import load_text
from watson.common.schemas import DialogueStage, HistoryTurn
from watson.llm.base import LLMClient


class Rephrasing(BaseModel):
    rephrased: str


class Rephraser:
    def __init__(self, llm: LLMClient, prompt_template: str, history_turns: int = 2) -> None:
        self.llm = llm
        self.prompt_template = prompt_template
        self.history_turns = history_turns

    @classmethod
    def from_config(cls, llm: LLMClient, prompt_path: str | Path = "prompts/decision/rephrase.txt") -> Rephraser:
        return cls(llm, load_text(prompt_path))

    def build_prompt(
        self,
        user_message: str,
        history: Sequence[HistoryTurn] = (),
        dialogue_stage: DialogueStage | None = None,
        case_context: str | None = None,
    ) -> str:
        recent = history[-self.history_turns :] if self.history_turns else []
        speaker = {"user": "Assistant", "holmes": "Holmes"}
        return self.prompt_template.format(
            user_message=user_message,
            history="\n".join(f"{speaker[t.role]}: {t.text}" for t in recent) or "(no previous turns)",
            dialogue_stage=dialogue_stage.value if dialogue_stage else "not yet tracked",
            case_context=case_context or "not provided",
        )

    def rephrase(
        self,
        user_message: str,
        history: Sequence[HistoryTurn] = (),
        dialogue_stage: DialogueStage | None = None,
        case_context: str | None = None,
    ) -> str:
        """The rewritten message, or the original if the model returns nothing."""
        prompt = self.build_prompt(user_message, history, dialogue_stage, case_context)
        return self.llm.generate_structured(prompt, Rephrasing).rephrased.strip() or user_message