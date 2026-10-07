import json

import pytest
from pydantic import ValidationError

from watson.common.schemas import HistoryTurn, ModuleToggles
from watson.evaluation.middleware_regression import (
    StepConfig,
    StepSummary,
    TurnKind,
    build_pipeline,
    find_regressions,
    load_dialogues,
    load_experiment_config,
    run_step,
    summarize_step,
)
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
    """Plays both judges: WICS gives drift messages 2 and others 9; the persona score is always 7."""

    model_name = "fake-judge"

    def __init__(self) -> None:
        self.persona_prompts: list[str] = []

    def _generate(self, prompt: str, **kwargs) -> str:
        if "persona profile" in prompt:
            self.persona_prompts.append(prompt)
            return json.dumps({"reasoning": "in character", "score": 7})
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
    return run_step(step_index, step, pipeline, persona_judge, DIALOGUES, CONFIG.seed)


# --- Repo config and scripted dialogues ----------------------------------------------


def test_repo_config_has_the_two_built_steps():
    assert [step.modules.input_constraint for step in CONFIG.steps] == [False, True]


def test_every_dialogue_mixes_normal_and_drift_turns():
    assert len(DIALOGUES) == 3
    for dialogue in DIALOGUES:
        kinds = {turn.kind for turn in dialogue.turns}
        assert TurnKind.NORMAL in kinds, dialogue.id
        assert any(kind.is_drift for kind in kinds), dialogue.id


def test_steps_cannot_enable_components_that_are_not_built():
    with pytest.raises(ValidationError, match="not built yet"):
        StepConfig(name="x", template="t", modules=ModuleToggles(dialogue_flow=True))


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