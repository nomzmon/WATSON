"""Prompt Builder (thesis Section 5.1.3).

Combines the validated user message with the persona profile, conversation
history, dialogue state and case context into a StructuredPrompt. The Dialogue
Flow Manager (Section 5.1.4) then adds its selected constraints before the
prompt is rendered and sent to the Response Generation Layer.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, Field

from watson.common.config import load_text
from watson.common.schemas import DialogueStage, HistoryTurn, PersonaProfile

_LAYER_HEADINGS = {
    "personality": "Personality",
    "expression": "How you speak",
    "knowledge": "What you know",
    "moral": "Your moral principles",
    "belief": "Your beliefs",
}
_SPEAKERS = {"user": "Visitor", "holmes": "Holmes"}


class StructuredPrompt(BaseModel):
    user_message: str
    history: list[HistoryTurn] = Field(default_factory=list)
    dialogue_stage: DialogueStage | None = None
    case_context: str | None = None
    constraints: list[str] = Field(default_factory=list)


class PromptBuilder:
    def __init__(self, persona_profile: PersonaProfile, template: str, history_turns: int = 10) -> None:
        self.persona_profile = persona_profile
        self.template = template
        self.history_turns = history_turns
        self._persona_text = render_persona(persona_profile)

    @classmethod
    def from_config(
        cls,
        profile_path: str | Path = "data/persona_profiles/holmes_profile_final.json",
        template_path: str | Path = "prompts/generation/structured_prompt.txt",
        history_turns: int = 10,
    ) -> PromptBuilder:
        profile = PersonaProfile.model_validate_json(load_text(profile_path))
        return cls(profile, load_text(template_path), history_turns)

    def build(
        self,
        user_message: str,
        history: Sequence[HistoryTurn] = (),
        dialogue_stage: DialogueStage | None = None,
        case_context: str | None = None,
    ) -> StructuredPrompt:
        recent = list(history[-self.history_turns :]) if self.history_turns else []
        return StructuredPrompt(
            user_message=user_message,
            history=recent,
            dialogue_stage=dialogue_stage,
            case_context=case_context,
        )

    def render(self, prompt: StructuredPrompt) -> str:
        """Templates may use any subset of the placeholders; unused ones are ignored."""
        return self.template.format(
            persona_profile=self._persona_text,
            user_message=prompt.user_message,
            history="\n".join(f"{_SPEAKERS[t.role]}: {t.text}" for t in prompt.history)
            or "(This is the start of the conversation.)",
            dialogue_stage=prompt.dialogue_stage.value if prompt.dialogue_stage else "not yet tracked",
            case_context=prompt.case_context or "No case has been presented yet.",
            constraints="\n".join(f"- {c}" for c in prompt.constraints) or "None for this reply.",
        )


def render_persona(profile: PersonaProfile) -> str:
    sections = []
    for layer, heading in _LAYER_HEADINGS.items():
        traits = getattr(profile, layer).traits
        if traits:
            sections.append(f"{heading}:\n" + "\n".join(f"- {trait}" for trait in traits))
    return "\n\n".join(sections)