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
from watson.middleware.output_validation.safety_filter import FilterSettings, SafetyFilter
from watson.middleware.output_validation.validator import (
    OutputValidator,
    OVSJudgement,
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

class DetailJudge(ScriptedJudge):
    """Like ScriptedJudge, but lists the given unsupported details on the first call only."""

    def __init__(self, details: list[str], *scores: dict[str, int]) -> None:
        super().__init__(*scores)
        self.details = details

    def _generate(self, prompt: str, **kwargs) -> str:
        judgement = json.loads(super()._generate(prompt, **kwargs))
        details, self.details = self.details, []
        return json.dumps({"unsupported_details": details, **judgement})


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

def test_unsupported_details_cap_narrative_coherence_and_fail_the_reply():
    judge = DetailJudge(["the window latch was broken", "boots with a moderate amount of tread"], GOOD)

    ovs = OutputValidator.from_config(judge, PROFILE).evaluate(make_prompt(), "The latch was broken.")

    assert ovs.criteria["NC"].score == 3  # repo cap
    assert "the window latch was broken; boots with a moderate amount of tread" in ovs.criteria["NC"].rationale
    assert ovs.weighted_total == pytest.approx(0.40 * 9 + 0.30 * 3 + 0.20 * 10 + 0.10 * 9)
    assert not passes(ovs, SETTINGS)


def test_a_reply_without_unsupported_details_keeps_its_narrative_score():
    ovs = OutputValidator.from_config(DetailJudge([], GOOD), PROFILE).evaluate(make_prompt(), "Elementary.")

    assert ovs.criteria["NC"].score == 9


def test_repo_prompt_asks_for_unsupported_details_before_the_scores():
    prompt = OutputValidator.from_config(ScriptedJudge(GOOD), PROFILE).build_prompt(make_prompt(), "Elementary.")

    assert 'First give the key "unsupported_details"' in prompt


def test_the_judge_must_write_the_unsupported_details_list_first():
    schema = OVSJudgement.model_json_schema()

    assert list(schema["properties"])[0] == "unsupported_details"
    assert "unsupported_details" in schema["required"]



def test_repo_settings_allow_two_retries():
    assert load_validation_settings().max_retries == 2

def test_repo_fallback_message_is_only_holmes_words():
    # A comment pasted with the wrong indentation once became part of the fallback reply.
    fallback = OutputValidationEngine.from_config(ScriptedJudge(GOOD), PROFILE).fallback_message

    assert fallback.startswith("Forgive me") and fallback.endswith("what you observed.")
    assert "#" not in fallback and "\n" not in fallback

# --- Constraint refinement ---------------------------------------------------------------


def test_weak_criteria_are_those_below_the_limit_or_else_the_lowest():
    assert weak_criteria(make_ovs(9, 4, 6, 8), below=7) == ["NC", "HWB"]
    assert weak_criteria(make_ovs(9, 8, 7, 8), below=7) == ["HWB"]


def test_first_retry_uses_level_2_and_later_retries_level_3():
    assert [refinement_level(r) for r in (1, 2, 3)] == [2, 3, 3]


def test_weak_criteria_add_their_constraints_at_the_given_level():
    refined = refine_constraints([], ["HWB", "SS"], 2, DialogueStage.EVIDENCE, LIBRARY)

    assert {c.id: c.level for c in refined} == {"H-VICTORIAN-ERA": 2, "P-VOICE": 2, "P-SPOKEN-ONLY": 2}


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

def test_a_reply_with_invented_evidence_is_retried_with_the_continuity_rules_strengthened():
    result, _ = run_loop(DetailJudge(["the window latch was broken"], GOOD))

    assert (result.reply, [a.passed for a in result.attempts]) == ("reply 2", [False, True])
    assert {c.id: c.level for c in result.attempts[1].constraints}["CC-NO-INVENTED-EVIDENCE"] == 2

def test_the_fallback_is_returned_when_every_attempt_fails():
    result, seen = run_loop(ScriptedJudge(WEAK_NARRATIVE))

    assert result.used_fallback
    assert result.reply.startswith("Forgive me")
    assert [attempt for attempt, _ in seen] == [0, 1, 2]  # first reply plus two retries
    assert {c.level for c in result.attempts[2].constraints} == {3}

# --- Safety filter -----------------------------------------------------------------------

FILTER = SafetyFilter.from_config()

# Replies from regression run 5 (2026-10-14) that the judge passed with OVS 8.1-9.2.
RUN5_SECURITY_FOOTAGE = (
    "My dear visitor, I have examined the security footage, but alas, it offers naught but a blurry glimpse of a "
    "fleeting shadow. The grainy image does not reveal the culprit's identity, though it does confirm the hasty "
    "nature of the escape. We must delve deeper into the investigation, my dear fellow, to uncover the truth."
)
RUN5_STAGE_DIRECTION = (
    '"Good evening, sir. You have requested my assistance in a matter of grave concern. Please, do tell me '
    'precisely what happened."\n\n(Holmes leans forward, his eyes narrowing)'
)
RUN5_OPEN_QUOTE = (
    '"Indeed, my dear visitor. The gardener\'s dismissal last week certainly raises suspicion. The investigation '
    "continues, my dear visitor, and the truth, like a pearl, will be unearthed."
)


def flagged(reply: str) -> list[tuple[str, str, str]]:
    return [(v.rule, v.criterion, v.evidence) for v in FILTER.check(reply)]


def test_repo_config_turns_the_safety_filter_on():
    assert FILTER is not None
    assert FILTER.settings.score_cap == 2


def test_filter_catches_the_problems_the_judge_missed_in_run_5():
    assert flagged(RUN5_SECURITY_FOOTAGE) == [("modern term", "HWB", "footage")]
    assert flagged(RUN5_STAGE_DIRECTION) == [
        ("stage direction", "SS", "(Holmes leans forward, his eyes narrowing)")
    ]
    assert flagged(RUN5_OPEN_QUOTE) == [("open quotation", "SS", '"')]


@pytest.mark.parametrize(
    ("reply", "rule"),
    [
        ("*leans back in his chair* The mud is telling.", "stage direction"),
        ("[He lights his pipe.] Go on.", "stage direction"),
        ("Holmes: The mud is telling.", "name label"),
        ("The mud is telling.\nVisitor: And then?", "name label"),
        ("The mud is telling, Holmes said.", "narration"),
        ("Holmes nodded. The mud is telling.", "narration"),
        ("\u201cThe mud is telling.", "open quotation"),
        ("Have you checked the CCTV?", "modern term"),
        ("Send it by e-mail.", "modern term"),
        ("No computers were involved.", "modern term"),
        # From regression runs 6 and 7: first-person narration and the visitor's line written by Holmes.
        ('"Good evening, sir," I replied, my voice as dry as a bone. "What brings you here?"', "narration"),
        ('"Indeed," said I. "Most curious."', "narration"),
        ("The visitor's voice was low and urgent.", "narration"),
        ("We must tread cautiously, my dear Watson, and gather further evidence.", "calls the visitor Watson"),
        ("Well, Watson, the footprints lead us outside the study.", "calls the visitor Watson"),
        ("Watson, fetch my coat.", "calls the visitor Watson"),
    ],
)
def test_filter_rules(reply, rule):
    assert rule in [v.rule for v in FILTER.check(reply)]


@pytest.mark.parametrize(
    "reply",
    [
        "The mud is telling (as I suspected it would be), and the window was forced.",  # Holmes' own aside
        "Send a wire, or ring them on the telephone.",  # the telephone existed in 1890s London
        "Mud of that colour comes from the flower beds, so our visitor crossed the garden.",
        "'The butler,' you say? Then we must see his boots.",  # single quotes and apostrophes
        "You must *see*, not merely look.",  # one-word emphasis
        '"The game is afoot," as I have said before.',
        "As I said to the inspector, the mud is telling.",  # "I said" in Holmes' own speech
        "My friend Watson is away in the country this week.",  # mentioning Watson is not calling the visitor Watson
        'The butler told me, "I heard nothing," and I believe him.',
    ],
)
def test_filter_passes_ordinary_holmes_speech(reply):
    assert FILTER.check(reply) == []


def test_filter_caps_flagged_criteria_and_recomputes_the_ovs():
    judged = make_ovs(9, 9, 10, 9)
    weights = load_ovs_weights()

    capped = FILTER.apply(judged, FILTER.check(RUN5_SECURITY_FOOTAGE), weights)

    assert capped.criteria["HWB"].score == 2
    assert capped.weighted_total == 7.6  # 0.40*9 + 0.30*9 + 0.20*2 + 0.10*9
    assert capped.criteria["HWB"].rationale.startswith('Filter: modern term "footage". Judge (10)')
    assert not passes(capped, SETTINGS)  # HWB 2 is below the minimum criterion score


def test_filter_can_be_turned_off(tmp_path):
    config = tmp_path / "ovs.yaml"
    config.write_text("safety_filter:\n  enabled: false\n  score_cap: 2\n  modern_terms: [camera]\n")

    assert SafetyFilter.from_config(config) is None
    assert SafetyFilter(FilterSettings(score_cap=2, modern_terms=[])).check("A video.") == []


def test_a_reply_the_judge_passes_is_retried_when_the_filter_flags_it():
    engine = OutputValidationEngine.from_config(ScriptedJudge(GOOD), PROFILE)
    replies = iter([RUN5_STAGE_DIRECTION, "Pray, tell me precisely what happened."])

    result = engine.run(make_prompt(), [], lambda prompt, attempt: next(replies))

    first, second = result.attempts
    assert (first.passed, first.ovs.criteria["SS"].score, first.violations[0].rule) == (False, 2, "stage direction")
    assert {c.id: c.level for c in second.constraints} == {"P-VOICE": 2, "P-SPOKEN-ONLY": 2}
    assert (result.reply, second.passed, second.violations) == ("Pray, tell me precisely what happened.", True, [])
    
def test_calling_the_visitor_watson_strengthens_the_persona_constraints():
    engine = OutputValidationEngine.from_config(ScriptedJudge(GOOD), PROFILE)
    replies = iter(["We must tread cautiously, my dear Watson.", "We must tread cautiously, my dear fellow."])

    result = engine.run(make_prompt(), [], lambda prompt, attempt: next(replies))

    first, second = result.attempts
    assert (first.passed, first.ovs.criteria["PCC"].score) == (False, 2)
    assert first.violations[0].describe() == 'calls the visitor Watson "my dear Watson"'
    assert second.constraints[[c.id for c in second.constraints].index("P-ADDRESS-VISITOR")].level == 2
    assert result.reply == "We must tread cautiously, my dear fellow."
