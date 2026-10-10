import json

import pytest
from pydantic import ValidationError

from watson.common.schemas import DecisionType, DialogueStage, HistoryTurn, ModuleToggles
from watson.evaluation.middleware_regression import (
    StepConfig,
    StepSummary,
    TurnKind,
    build_pipeline,
    find_regressions,
    load_dialogues,
    load_experiment_config,
    repetition,
    run_step,
    summarize_dialogue_flow,
    summarize_input_checks,
    summarize_step,
)
from watson.evaluation.narrative_consistency import NarrativeConsistencyJudge
from watson.evaluation.persona_consistency import PersonaConsistencyJudge
from watson.llm.base import LLMClient
from watson.middleware.generation.generator import ResponseGenerator

DRIFT_WORDS = ("opera", "violin", "AI assistant", "ChatGPT", "camera", "FBI")


class FakeGemma(LLMClient):
    model_name = "fake-gemma"

    def __init__(self) -> None:
        self.seeds: list[int] = []

    def _generate(self, prompt: str, **kwargs) -> str:
        self.seeds.append(kwargs["seed"])
        return "Elementary."


class FakeJudge(LLMClient):
    """Plays every judge. WICS gives drift messages 2 and others 9; the persona score is always 7 and the
    narrative score 6; the dialogue flow judge reports weak narrative progression and proposes the suspects stage."""

    model_name = "fake-judge"

    def __init__(self) -> None:
        self.persona_prompts: list[str] = []
        self.narrative_prompts: list[str] = []

    def _generate(self, prompt: str, **kwargs) -> str:
        if "persona profile" in prompt:
            self.persona_prompts.append(prompt)
            return json.dumps({"reasoning": "in character", "score": 7})
        if "narrative consistency" in prompt:
            self.narrative_prompts.append(prompt)
            return json.dumps({"reasoning": "consistent", "score": 6})
        if "Conversation State Tracking" in prompt:
            scores = {"CST": 9, "NP": 4, "CC": 9, "TBC": 9}
            return json.dumps({**{c: {"reasoning": "r", "score": s} for c, s in scores.items()}, "stage": "suspects"})
        message = prompt.split("<<<")[1].split(">>>")[0]
        score = 2 if any(word in message for word in DRIFT_WORDS) else 9
        return json.dumps({code: {"reasoning": "r", "score": score} for code in ("OOP", "TR", "HC", "GIP")})


CONFIG = load_experiment_config()
DIALOGUES = load_dialogues(CONFIG.dialogues)


def run(step_index: int, gemma: FakeGemma | None = None, judge: FakeJudge | None = None):
    gemma, judge = gemma or FakeGemma(), judge or FakeJudge()
    step = CONFIG.steps[step_index - 1]
    pipeline = build_pipeline(step, CONFIG.profile, ResponseGenerator(gemma), judge)
    persona_judge = PersonaConsistencyJudge.from_config(judge, CONFIG.profile)
    return run_step(step_index, step, pipeline, persona_judge, DIALOGUES, CONFIG.seed,
                    narrative_judge=NarrativeConsistencyJudge.from_config(judge))


# --- Repo config and scripted dialogues ----------------------------------------------


def test_repo_config_adds_one_component_per_step():
    assert [step.modules.input_constraint for step in CONFIG.steps] == [False, True, True, True]
    assert [step.modules.dialogue_flow for step in CONFIG.steps] == [False, False, False, True]
    assert [step.template.split("/")[-1] for step in CONFIG.steps] == [
        "baseline_persona.txt", "baseline_persona.txt", "structured_prompt.txt", "structured_prompt.txt"
    ]


def test_repo_config_expects_a_decision_for_every_turn_kind():
    assert set(CONFIG.expected_decisions) == set(TurnKind)
    assert CONFIG.expected_decisions[TurnKind.OFF_TOPIC] == [DecisionType.REDIRECT]
    assert DecisionType.ACCEPT in CONFIG.expected_decisions[TurnKind.RUSHING]


def test_every_dialogue_mixes_normal_and_drift_turns():
    assert len(DIALOGUES) == 3
    for dialogue in DIALOGUES:
        kinds = {turn.kind for turn in dialogue.turns}
        assert TurnKind.NORMAL in kinds, dialogue.id
        assert any(kind.is_drift for kind in kinds), dialogue.id


def test_steps_cannot_enable_components_that_are_not_built():
    with pytest.raises(ValidationError, match="not built yet"):
        StepConfig(name="x", template="t", modules=ModuleToggles(output_validation=True))


def test_dialogue_flow_step_needs_a_template_with_a_guidance_section():
    with pytest.raises(ValidationError, match="constraints"):
        StepConfig(name="x", template="prompts/generation/baseline_persona.txt",
                   modules=ModuleToggles(dialogue_flow=True))


def test_input_constraint_and_decision_layer_must_be_enabled_together():
    with pytest.raises(ValidationError, match="together"):
        StepConfig(name="x", template="t", modules=ModuleToggles(input_constraint=True))


# --- Running steps -------------------------------------------------------------------


def test_baseline_step_sends_every_turn_to_holmes_and_scores_it():
    records = run(1)

    assert len(records) == sum(len(d.turns) for d in DIALOGUES)
    assert all(r.source == "generator" and r.persona_score == 7 for r in records)


def test_input_constraint_step_stops_drift_turns_and_skips_scoring_them():
    records = run(2)

    for r in records:
        if r.kind.is_drift:
            assert r.source in ("redirect_message", "reject_message")
            assert r.persona_score is None
        else:
            assert r.source == "generator"


def test_input_check_step_records_the_wics_scores_behind_each_decision():
    records = run(2)

    drift = next(r for r in records if r.kind.is_drift)
    assert (drift.wics_total, drift.wics_scores()) == (2.0, {"OOP": 2, "TR": 2, "HC": 2, "GIP": 2})
    assert "reject band" in drift.input_check_reasons
    assert drift.wics_reasoning.splitlines() == ["OOP 2: r", "TR 2: r", "HC 2: r", "GIP 2: r"]
    assert all(r.wics_total == 9.0 for r in records if not r.kind.is_drift)


def test_dialogue_flow_step_records_stage_dfms_and_guidance():
    records = run(4)

    reached = [r for r in records if r.source == "generator"]
    assert all(r.dfms_scores() == {"CST": 9, "NP": 4, "CC": 9, "TBC": 9} for r in reached)
    # The judge proposes "suspects" every turn, but the stage only advances one step per turn.
    first_two = [r.stage for r in reached if r.dialogue_id == "dialogue_01"][:2]
    assert first_two == [DialogueStage.EVIDENCE, DialogueStage.SUSPECTS]
    assert "NP-ONE-STEP (level 1)" in reached[0].guidance
    assert all(r.stage is None and r.guidance == "" for r in records if r.source != "generator")


def test_steps_without_the_dialogue_flow_manager_record_no_flow():
    assert all(r.stage is None and r.dfms_total is None for r in run(3))


def test_every_generated_reply_gets_a_narrative_score_with_the_case():
    judge = FakeJudge()
    records = run(1, judge=judge)

    assert all(r.narrative_score == 6 and r.narrative_reasoning == "consistent" for r in records)
    assert DIALOGUES[0].case_context in judge.narrative_prompts[0]


def test_baseline_step_records_no_wics_scores():
    assert all(r.wics_total is None and r.input_check_reasons == "" for r in run(1))


def test_each_step_replays_turns_with_the_same_seeds():
    baseline, guarded = FakeGemma(), FakeGemma()
    run(1, gemma=baseline)
    run(2, gemma=guarded)

    # Step 2 skips drift turns, so its seeds are the baseline's seeds for the remaining turns.
    assert set(guarded.seeds) < set(baseline.seeds)
    assert len(set(baseline.seeds)) == len(baseline.seeds)


def test_persona_judge_sees_the_profile_history_and_reply():
    judge = FakeJudge()
    run(1, judge=judge)

    second_turn_prompt = judge.persona_prompts[1]
    assert "Relentlessly analytical" in second_turn_prompt
    assert "Visitor: Good evening, Mr. Holmes." in second_turn_prompt
    assert "replied as Holmes\n<<<\nElementary.\n>>>" in second_turn_prompt


# --- Summaries and regressions ---------------------------------------------------------


def test_summary_counts_caught_drift_and_blocked_normal_turns():
    summary = summarize_step(2, "guarded", run(2))

    drift_turns = sum(t.kind.is_drift for d in DIALOGUES for t in d.turns)
    assert (summary.drift_caught, summary.drift_total) == (drift_turns, drift_turns)
    assert summary.normal_blocked == 0
    assert summary.persona_normal == 7.0
    assert summary.persona_drift is None  # every drift turn was stopped before reaching Holmes


def test_input_check_summary_gives_mean_scores_and_decisions_per_turn_kind():
    summaries = {s.kind: s for s in summarize_input_checks(run(2))}

    normal, breaking = summaries[TurnKind.NORMAL], summaries[TurnKind.PERSONA_BREAKING]
    assert (normal.turns, normal.wics, normal.criteria["GIP"]) == (13, 9.0, 9.0)
    assert normal.decisions[DecisionType.ACCEPT] == 13
    assert (breaking.turns, breaking.wics, breaking.decisions[DecisionType.REJECT]) == (2, 2.0, 2)
    assert list(summaries) == list(TurnKind)  # in turn-kind order


def test_input_check_summary_counts_decisions_that_were_as_expected():
    # The fake judge rejects every drift turn: right for anachronisms, wrong for off-topic turns (redirect).
    summaries = {s.kind: s for s in summarize_input_checks(run(2), CONFIG.expected_decisions)}

    assert summaries[TurnKind.ANACHRONISM].as_expected == 2
    assert summaries[TurnKind.OFF_TOPIC].as_expected == 0
    assert summaries[TurnKind.NORMAL].as_expected == 13
    assert summaries[TurnKind.OFF_TOPIC].expected == [DecisionType.REDIRECT]


def test_without_expectations_nothing_is_counted_as_expected():
    assert all(s.as_expected is None for s in summarize_input_checks(run(2)))


def test_input_check_summary_is_empty_without_the_input_constraint_engine():
    assert summarize_input_checks(run(1)) == []


class RepetitiveGemma(FakeGemma):
    def _generate(self, prompt: str, **kwargs) -> str:
        super()._generate(prompt, **kwargs)
        return "The footprints tell us a great deal."


def test_summary_includes_narrative_score_and_repetition():
    records = run(1, gemma=RepetitiveGemma())
    summary = summarize_step(1, "baseline", records)

    assert summary.narrative_normal == 6.0
    # Each dialogue's first reply is new; every later one repeats it word for word.
    assert [r.repetition for r in records[:3]] == [0.0, 1.0, 1.0]
    assert summary.repetition == round(18 / 21, 2)


def test_dialogue_flow_summary_gives_mean_dfms_guidance_and_final_stages():
    [summary] = summarize_dialogue_flow(run(4))

    generated = sum(not t.kind.is_drift for d in DIALOGUES for t in d.turns)
    assert (summary.step, summary.turns, summary.guided_turns) == (4, generated, generated)
    assert (summary.dfms, summary.criteria["NP"]) == (7.5, 4.0)  # 0.35*9 + 0.30*4 + 0.20*9 + 0.15*9
    assert summary.final_stages["dialogue_01"] == DialogueStage.SUSPECTS  # never beyond the stage the judge proposes


def test_dialogue_flow_summary_is_empty_without_the_dialogue_flow_manager():
    assert summarize_dialogue_flow(run(2)) == []


def test_repetition_is_the_share_of_reused_word_trigrams():
    assert repetition("Let us examine the window.", []) == 0.0
    assert repetition("Let us examine the window.", ["We must examine the window at once."]) == 0.33
    assert repetition("Quite so.", ["Quite so."]) == 0.0  # too short to have a trigram


def make_summary(step: int, persona_normal: float | None) -> StepSummary:
    return StepSummary(step=step, name="s", persona_normal=persona_normal, persona_drift=None, drift_caught=0,
                       drift_total=0, normal_blocked=0, normal_total=0, mean_latency_ms=0)


def test_regression_is_a_persona_drop_beyond_tolerance():
    summaries = [make_summary(1, 7.5), make_summary(2, 7.0), make_summary(3, 5.5)]

    [regression] = find_regressions(summaries, tolerance=1.0)

    assert (regression.step, regression.drop) == (3, 1.5)


def test_improvement_is_not_a_regression():
    assert find_regressions([make_summary(1, 6.0), make_summary(2, 8.0)], tolerance=1.0) == []


def test_history_before_the_turn_is_what_the_judge_sees():
    judge = FakeJudge()
    persona_judge = PersonaConsistencyJudge.from_config(judge, CONFIG.profile)

    persona_judge.score([HistoryTurn(role="user", text="Earlier.")], "Now?", "Quite.")

    assert "Visitor: Earlier." in judge.persona_prompts[0]