"""Quick check of the Input Constraint Engine, without generating any replies.

Re-scores every scripted message against the conversation histories recorded
in an earlier regression run (one history per step), using the current WICS
prompt and decision rules. This shows in a few minutes whether a change to the
input check gets more decisions right, and whether the same message gets the
same decision whatever conversation came before it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime

from pydantic import BaseModel

from watson.common.schemas import DecisionType
from watson.evaluation.middleware_regression import (
    ScriptedDialogue,
    TurnKind,
    TurnRecord,
    history_before_each_turn,
)
from watson.middleware.decision.decision import DecisionThresholds, decide
from watson.middleware.input_constraint.engine import InputConstraintEngine
from watson.middleware.input_constraint.scoring import WICS_CRITERIA


class Check(BaseModel):
    """One scripted message scored after one recorded history."""

    history_step: int
    wics: float
    criteria: dict[str, int]
    decision: DecisionType


class TurnCheck(BaseModel):
    dialogue_id: str
    turn: int
    kind: TurnKind
    message: str
    expected: list[DecisionType]
    checks: list[Check]

    @property
    def consistent(self) -> bool:
        return len({c.decision for c in self.checks}) == 1

    def as_expected(self, history_step: int) -> bool | None:
        if not self.expected:
            return None
        return next(c.decision for c in self.checks if c.history_step == history_step) in self.expected


def check_input_stability(
    records: list[TurnRecord],
    steps: Sequence[int],
    engine: InputConstraintEngine,
    thresholds: DecisionThresholds,
    dialogues: Sequence[ScriptedDialogue],
    expected_decisions: dict[TurnKind, list[DecisionType]],
    on_check: Callable[[TurnCheck, Check], None] = lambda turn, check: None,
) -> list[TurnCheck]:
    histories = history_before_each_turn(records)
    cases = {d.id: d.case_context for d in dialogues}
    results = []
    for r in sorted((r for r in records if r.step == steps[0]), key=lambda r: (r.dialogue_id, r.turn)):
        turn = TurnCheck(dialogue_id=r.dialogue_id, turn=r.turn, kind=r.kind, message=r.message,
                         expected=expected_decisions.get(r.kind, []), checks=[])
        for step in steps:
            wics = engine.evaluate(r.message, histories[(step, r.dialogue_id, r.turn)], case_context=cases[r.dialogue_id])
            check = Check(
                history_step=step,
                wics=wics.weighted_total,
                criteria={code: int(wics.criteria[code].score) for code in WICS_CRITERIA},
                decision=decide(wics, thresholds).decision,
            )
            turn.checks.append(check)
            on_check(turn, check)
        results.append(turn)
    return results


def render_stability_report(started: datetime, code_version: str, source: str, steps: Sequence[int],
                            results: Sequence[TurnCheck]) -> str:
    lines = [
        f"# Input check stability: {started:%Y-%m-%d %H:%M}",
        "",
        f"- **Code version:** {code_version}",
        f"- **Histories from:** `{source}` (steps {', '.join(map(str, steps))})",
        f"- **Messages:** {len(results)}, each scored once after each step's recorded history "
        "(without the investigation stage, as in steps 2 and 3)",
        "",
        "## Summary",
        "",
        "| History from step | Decisions as expected |",
        "|---|---|",
    ]
    rated = [r for r in results if r.expected]
    for step in steps:
        lines.append(f"| {step} | {sum(r.as_expected(step) for r in rated)}/{len(rated)} |")
    inconsistent = [r for r in results if not r.consistent]
    lines += [
        "",
        f"**Same decision after every history:** {len(results) - len(inconsistent)}/{len(results)} messages",
        "",
        "## Decisions per message",
        "",
        "| Dialogue | Turn | Kind | Expected | " + " | ".join(f"After step {s}" for s in steps) + " | Same? |",
        "|---" * (5 + len(steps)) + "|",
    ]
    for r in results:
        cells = []
        for c in r.checks:
            criteria = ", ".join(f"{code} {score}" for code, score in c.criteria.items())
            cells.append(f"{c.decision.value} ({c.wics:.2f}: {criteria})")
        expected = " or ".join(d.value for d in r.expected) or "-"
        lines.append(f"| {r.dialogue_id} | {r.turn} | {r.kind.value.replace('_', ' ')} | {expected} | "
                     + " | ".join(cells) + f" | {'yes' if r.consistent else '**no**'} |")
    lines += ["", "## Messages", ""]
    lines += [f"- {r.dialogue_id} turn {r.turn}: {r.message}" for r in results]
    return "\n".join(lines) + "\n"