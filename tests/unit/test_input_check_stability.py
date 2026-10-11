import json
from datetime import datetime

from watson.common.schemas import DecisionType
from watson.evaluation.input_check_stability import check_input_stability, render_stability_report
from watson.evaluation.middleware_regression import (
    ScriptedDialogue,
    TurnKind,
    TurnRecord,
    history_before_each_turn,
)
from watson.llm.base import LLMClient
from watson.middleware.decision.decision import load_decision_thresholds
from watson.middleware.input_constraint.engine import InputConstraintEngine


class HistorySensitiveJudge(LLMClient):
    """Rejects the rushing message only when Holmes was rude just before it; accepts everything else."""

    model_name = "fake-judge"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def _generate(self, prompt: str, **kwargs) -> str:
        self.prompts.append(prompt)
        message = prompt.split("<<<")[1].split(">>>")[0]
        score = 2 if "thief" in message and "Holmes: Bah." in prompt else 9
        return json.dumps({code: {"reasoning": "r", "score": score} for code in ("OOP", "TR", "HC", "GIP")})


DIALOGUE = ScriptedDialogue(id="d1", title="t", case_context="The pearl necklace affair",
                            turns=[{"kind": "normal", "message": "Hello."}])


def record(step: int, turn: int, kind: TurnKind, message: str, source: str, reply: str) -> TurnRecord:
    return TurnRecord(step=step, step_name="s", dialogue_id="d1", turn=turn, kind=kind, message=message,
                      decision=None, source=source, reply=reply, persona_score=None, persona_reasoning="",
                      latency_ms=0)


RECORDS = [
    record(2, 1, TurnKind.NORMAL, "Good evening.", "generator", "Bah."),
    record(2, 2, TurnKind.OFF_TOPIC, "Lovely weather?", "reject_message", "[System notice]"),
    record(2, 3, TurnKind.RUSHING, "Who is the thief?", "generator", "Patience."),
    record(3, 1, TurnKind.NORMAL, "Good evening.", "generator", "Good evening to you."),
    record(3, 2, TurnKind.OFF_TOPIC, "Lovely weather?", "redirect_message", "Let us return to the case."),
    record(3, 3, TurnKind.RUSHING, "Who is the thief?", "generator", "Patience."),
]
EXPECTED = {TurnKind.NORMAL: [DecisionType.ACCEPT], TurnKind.RUSHING: [DecisionType.ACCEPT]}


def run_check(judge: HistorySensitiveJudge):
    engine = InputConstraintEngine.from_config(judge, history_turns=2)
    return check_input_stability(RECORDS, [2, 3], engine, load_decision_thresholds(), [DIALOGUE], EXPECTED)


def test_history_keeps_generated_and_redirected_replies_but_not_rejected_turns():
    histories = history_before_each_turn(RECORDS)

    assert [t.text for t in histories[(2, "d1", 3)]] == ["Good evening.", "Bah."]
    assert [t.text for t in histories[(3, "d1", 3)]][-2:] == ["Lovely weather?", "Let us return to the case."]
    assert histories[(2, "d1", 1)] == []


def test_each_message_is_scored_after_every_step_s_history():
    judge = HistorySensitiveJudge()
    results = run_check(judge)

    assert [(r.turn, [c.history_step for c in r.checks]) for r in results] == [(1, [2, 3]), (2, [2, 3]), (3, [2, 3])]
    assert len(judge.prompts) == 6
    assert "The pearl necklace affair" in judge.prompts[0]


def test_a_decision_that_depends_on_the_history_is_flagged_as_inconsistent():
    rushing = run_check(HistorySensitiveJudge())[2]

    assert [c.decision for c in rushing.checks] == [DecisionType.REJECT, DecisionType.ACCEPT]
    assert not rushing.consistent
    assert (rushing.as_expected(2), rushing.as_expected(3)) == (False, True)


def test_turns_without_an_expectation_are_not_rated():
    off_topic = run_check(HistorySensitiveJudge())[1]

    assert off_topic.consistent
    assert off_topic.as_expected(2) is None


def test_report_summarises_expected_and_consistent_decisions():
    report = render_stability_report(datetime(2026, 10, 10, 9, 0), "abc1234", "runs/x/turns.csv", [2, 3],
                                     run_check(HistorySensitiveJudge()))

    assert "| 2 | 1/2 |" in report and "| 3 | 2/2 |" in report
    assert "**Same decision after every history:** 2/3 messages" in report
    assert "| d1 | 3 | rushing | accept | reject (2.00: OOP 2, TR 2, HC 2, GIP 2) | accept (9.00:" in report
    assert "| **no** |" in report