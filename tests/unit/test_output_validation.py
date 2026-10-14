import json

import pytest

from watson.common.schemas import CriterionScore, DialogueStage, HistoryTurn, RubricName, RubricScore
from watson.llm.base import LLMClient
from watson.middleware.dialogue_flow.constraint_selector import ConstraintLibrary, SelectedConstraint
from watson.middleware.output_validation.constraint_refiner import (
    refine_constraints,
    refinement_level,
    weak_criteria,
)
from watson.middleware.output_validation.retry import OutputValidationEngine
from watson.middleware.output_validation.validator import (
    OutputValidator,
    ValidationSettings,
    load_ovs_weights,
    load_validation_settings,
    passes,
)
from watson.middleware.prompt_builder.builder import StructuredPrompt

PROFILE = "tests/fixtures/holmes_profile_sample.json"
LIBRARY = ConstraintLibrary.from_dir()
SETTINGS = ValidationSettings(pass_threshold=7.0, min_criterion=5, max_retries=2, weak_criterion_below=7)


class ScriptedJudge(LLMClient):
    """Returns the given OVS scores in turn, one set per call."""

    model_name = "fake-judge"

    def __init__(self, *scores: dict[str, int]) -> None:
        self.scores = list(scores)
        self.prompts: list[str] = []

    def _generate(self, prompt: str, **kwargs) -> str:
        self.prompts.append(prompt)
        scores = self.scores.pop(0) if len(self.scores) > 1 else self.scores[0]
        return json.dumps({code: {"reasoning": f"{code} reasoning", "score": s} for code, s in scores.items()})


GOOD = {"PCC": 9, "NC": 9, "HWB": 10, "SS": 9}
WEAK_NARRATIVE = {"PCC": 9, "NC": 4, "HWB": 10, "SS": 8}


def make_ovs(pcc: int, nc: int, hwb: int, ss: int) -> RubricScore:
    scores = {"PCC": pcc, "NC": nc, "HWB": hwb, "SS": ss}
    return RubricScore(rubric=RubricName.OVS, weighted_total=0,
                       criteria={code: CriterionScore(code=code, score=s) for code, s in scores.items()})


def make_prompt(stage: DialogueStage = DialogueStage.EVIDENCE) -> StructuredPrompt:
    return StructuredPrompt(
        user_message="What of the footprints?",
        history=[HistoryTurn(role="user", text="A necklace was stolen."), HistoryTurn(role="holmes", text="Go on.")],
        dialogue_stage=stage,
        case_context="The pearl necklace affair",
    )


# --- Scoring --------------------------------------------------------------------------


def test_repo_weights_follow_equation_5_3():
    assert load_ovs_weights() == {"PCC": 0.40, "NC": 0.30, "HWB": 0.20, "SS": 0.10}


def test_validator_scores_the_reply_with_the_ovs_weights():
    judge = ScriptedJudge({"PCC": 10, "NC": 8, "HWB": 6, "SS": 4})

    ovs = OutputValidator.from_config(judge, PROFILE).evaluate(make_prompt(), "Elementary.")

    assert ovs.rubric == RubricName.OVS
    assert ovs.weighted_total == 8.0  # 0.40*10 + 0.30*8 + 0.20*6 + 0.10*4
    assert ovs.criteria["SS"].rationale == "SS reasoning"


def test_validator_prompt_contains_profile_case_conversation_and_reply():
    judge = ScriptedJudge(GOOD)

    OutputValidator.from_config(judge, PROFILE).evaluate(make_prompt(), "The mud is telling.")

    prompt = judge.prompts[0]
    assert "Relentlessly analytical" in prompt
    assert "The pearl necklace affair" in prompt
    assert "Current stage of the investigation: evidence" in prompt
    assert "Visitor: A necklace was stolen.\nHolmes: Go on." in prompt
    assert "Candidate reply as Holmes\n<<<\nThe mud is telling.\n>>>" in prompt


@pytest.mark.parametrize(
    ("scores", "expected"),
    [((9, 9, 9, 9), True), ((7, 7, 7, 7), True), ((8, 8, 4, 9), False), ((6, 7, 7, 7), False)],
)
def test_a_reply_passes_only_with_a_high_enough_ovs_and_no_very_low_criterion(scores, expected):
    ovs = make_ovs(*scores)
    ovs.weighted_total = sum(w * s for w, s in zip((0.4, 0.3, 0.2, 0.1), scores))

    assert passes(ovs, SETTINGS) is expected


def test_repo_settings_allow_two_retries():
    assert load_validation_settings().max_retries == 2


# --- Constraint refinement ---------------------------------------------------------------


def test_weak_criteria_are_those_below_the_limit_or_else_the_lowest():
    assert weak_criteria(make_ovs(9, 4, 6, 8), below=7) == ["NC", "HWB"]
    assert weak_criteria(make_ovs(9, 8, 7, 8), below=7) == ["HWB"]


def test_first_retry_uses_level_2_and_later_retries_level_3():
    assert [refinement_level(r) for r in (1, 2, 3)] == [2, 3, 3]


def test_weak_criteria_add_their_constraints_at_the_given_level():
    refined = refine_constraints([], ["HWB", "SS"], 2, DialogueStage.EVIDENCE, LIBRARY)

    assert {c.id: c.level for c in refined} == {"H-VICTORIAN-ERA": 2, "P-VOICE": 2}


def test_a_constraint_already_in_the_prompt_is_strengthened_not_repeated():
    current = [SelectedConstraint(id="CC-REFER-BACK", category="context_continuity", level=1, text="weak")]

    refined = refine_constraints(current, ["NC"], 3, DialogueStage.EVIDENCE, LIBRARY)

    assert [c.id for c in refined].count("CC-REFER-BACK") == 1
    assert next(c for c in refined if c.id == "CC-REFER-BACK").level == 3
    assert "NP-NO-PREMATURE-CONCLUSION" in [c.id for c in refined]


def test_stage_limited_constraints_are_skipped_outside_their_stages():
    refined = refine_constraints([], ["NC"], 2, DialogueStage.DEDUCTION, LIBRARY)

    assert "NP-NO-PREMATURE-CONCLUSION" not in [c.id for c in refined]


# --- Retry loop and fallback -------------------------------------------------------------


def run_loop(judge: ScriptedJudge, constraints=()):
    engine = OutputValidationEngine.from_config(judge, PROFILE)
    seen: list[tuple[int, list[str]]] = []

    def generate(prompt: StructuredPrompt, attempt: int) -> str:
        seen.append((attempt, prompt.constraints))
        return f"reply {attempt + 1}"

    return engine.run(make_prompt(), list(constraints), generate), seen


def test_a_good_first_reply_is_delivered_without_retrying():
    result, seen = run_loop(ScriptedJudge(GOOD))

    assert (result.reply, result.used_fallback, len(result.attempts)) == ("reply 1", False, 1)
    assert len(seen) == 1


def test_a_failed_reply_is_regenerated_with_strengthened_constraints():
    stage = SelectedConstraint(id="DS-EVIDENCE", category="dialogue_state", level=1, text="Stay at evidence.")

    result, seen = run_loop(ScriptedJudge(WEAK_NARRATIVE, GOOD), [stage])

    assert (result.reply, result.used_fallback) == ("reply 2", False)
    assert [a.passed for a in result.attempts] == [False, True]
    assert seen[0][1] == ["Stay at evidence."]
    retry_constraints = {c.id: c.level for c in result.attempts[1].constraints}
    assert retry_constraints["DS-EVIDENCE"] == 1  # the Dialogue Flow Manager's constraint is kept
    assert retry_constraints["CC-NO-INVENTED-EVIDENCE"] == 2


def test_the_fallback_is_returned_when_every_attempt_fails():
    result, seen = run_loop(ScriptedJudge(WEAK_NARRATIVE))

    assert result.used_fallback
    assert result.reply.startswith("Forgive me")
    assert [attempt for attempt, _ in seen] == [0, 1, 2]  # first reply plus two retries
    assert {c.level for c in result.attempts[2].constraints} == {3}