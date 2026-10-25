"""A live conversation with one configuration, logged turn by turn (thesis Section 1.4, Phase 4).

Each turn is written as a TurnLog line (JSONL) as soon as it finishes, so an interrupted session keeps every
completed turn. The configuration is recorded in the log only; the person chatting is never shown it.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from watson.common.schemas import TurnLog
from watson.pipeline.orchestrator import Pipeline, TurnResult
from watson.pipeline.session import ConversationSession


def to_turn_log(result: TurnResult, session: ConversationSession, config_id: str,
                participant_id: str | None = None) -> TurnLog:
    flow, validation = result.flow, result.validation
    return TurnLog(
        session_id=session.session_id,
        participant_id=participant_id,
        config_id=config_id,
        user_input=result.user_message,
        dialogue_stage=flow.stage if flow else None,
        wics=result.wics,
        input_decision=result.decision,
        dfms=flow.dfms if flow else None,
        selected_constraint_ids=[c.id for c in flow.constraints] if flow else [],
        ovs=validation.attempts[-1].ovs if validation else None,
        retries=len(validation.attempts) - 1 if validation else 0,
        final_response=result.reply,
        latencies=result.latencies,
    )


def chat(
    pipeline: Pipeline,
    session: ConversationSession,
    config_id: str,
    log_path: Path,
    read: Callable[[], str] = input,
    write: Callable[[str], None] = print,
    participant_id: str | None = None,
) -> int:
    """Runs turns until an empty message or end of input; returns the number of turns logged."""
    turns = 0
    while True:
        try:
            message = read().strip()
        except EOFError:
            break
        if not message:
            break
        result = pipeline.run_turn(session, message)
        write(result.reply)
        log_turn(log_path, to_turn_log(result, session, config_id, participant_id))
        turns += 1
    return turns


def log_turn(log_path: Path, log: TurnLog) -> None:
    """Appends one turn as a JSON line, so every finished turn is kept even if the session is interrupted."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(log.model_dump_json() + "\n")