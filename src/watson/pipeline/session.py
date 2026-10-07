"""State of one conversation: case context, investigation stage and history."""

from __future__ import annotations

from pydantic import BaseModel, Field

from watson.common.schemas import DialogueStage, HistoryTurn
from watson.common.utils import new_id


class ConversationSession(BaseModel):
    session_id: str = Field(default_factory=lambda: new_id("session"))
    case_context: str | None = None
    dialogue_stage: DialogueStage | None = None
    history: list[HistoryTurn] = Field(default_factory=list)

    def add_exchange(self, user_message: str, reply: str) -> None:
        self.history += [HistoryTurn(role="user", text=user_message), HistoryTurn(role="holmes", text=reply)]