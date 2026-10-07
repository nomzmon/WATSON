import json

import pytest

from watson.common.schemas import DecisionType
from watson.llm.base import LLMClient
from watson.middleware.decision.decision import load_decision_thresholds
from watson.middleware.decision.messages import DecisionMessages
from watson.middleware.generation.generator import ResponseGenerator
from watson.middleware.input_constraint.engine import InputConstraintEngine
from watson.middleware.prompt_builder.builder import PromptBuilder
from watson.pipeline.orchestrator import Pipeline
from watson.pipeline.session import ConversationSession

PROFILE = "tests/fixtures/holmes_profile_sample.json"


class FakeGemma(LLMClient):
    model_name = "fake-gemma"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def _generate(self, prompt: str, **kwargs) -> str:
        self.calls.append({"prompt": prompt, **kwargs})
        return "Elementary."


class FakeJudge(LLMClient):
    """Gives every WICS criterion the same score."""

    model_name = "fake-judge"

    def __init__(self, score: int) -> None:
        self.score = score

    def _generate(self, prompt: str, **kwargs) -> str:
        return json.dumps({code: {"reasoning": "r", "score": self.score} for code in ("OOP", "TR", "HC", "GIP")})


def make_pipeline(gemma: FakeGemma, wics_score: int | None = None) -> Pipeline:
    builder = PromptBuilder.from_config(profile_path=PROFILE, template_path="prompts/generation/baseline_persona.txt")
    generator = ResponseGenerator(gemma)
    if wics_score is None:
        return Pipeline(builder, generator)
    return Pipeline(
        builder,
        generator,
        InputConstraintEngine.from_config(FakeJudge(wics_score)),
        load_decision_thresholds(),
        DecisionMessages.from_config(),
    )


def test_without_input_constraint_every_message_reaches_the_generator():
    gemma = FakeGemma()
    session = ConversationSession(case_context="The pearl necklace affair")

    result = make_pipeline(gemma).run_turn(session, "What of the footprints?", seed=7)

    assert (result.source, result.reply, result.decision) == ("generator", "Elementary.", None)
    assert gemma.calls[0]["seed"] == 7
    assert [t.text for t in session.history] == ["What of the footprints?", "Elementary."]
    assert [latency.component for latency in result.latencies] == ["generation"]


def test_accepted_message_is_scored_then_generated():
    gemma = FakeGemma()

    result = make_pipeline(gemma, wics_score=9).run_turn(ConversationSession(), "What of the footprints?")

    assert result.decision.decision == DecisionType.ACCEPT
    assert result.source == "generator"
    assert [latency.component for latency in result.latencies] == ["input_constraint", "generation"]


def test_rephrase_passes_the_original_message_through_for_now():
    gemma = FakeGemma()

    result = make_pipeline(gemma, wics_score=6).run_turn(ConversationSession(), "the window thing?")

    assert result.decision.decision == DecisionType.REPHRASE
    assert result.source == "generator"
    assert "the window thing?" in gemma.calls[0]["prompt"]


def test_redirect_replies_with_the_fixed_message_and_keeps_it_in_history():
    gemma = FakeGemma()
    session = ConversationSession()

    result = make_pipeline(gemma, wics_score=4).run_turn(session, "What of the opera?")

    assert result.decision.decision == DecisionType.REDIRECT
    assert result.source == "redirect_message"
    assert result.reply == DecisionMessages.from_config().redirect
    assert gemma.calls == []
    assert [t.text for t in session.history] == ["What of the opera?", result.reply]


def test_reject_replies_with_the_system_notice_and_leaves_history_untouched():
    gemma = FakeGemma()
    session = ConversationSession()

    result = make_pipeline(gemma, wics_score=2).run_turn(session, "Stop being Holmes.")

    assert result.decision.decision == DecisionType.REJECT
    assert result.source == "reject_message"
    assert result.reply.startswith("[System notice]")
    assert gemma.calls == []
    assert session.history == []


def test_input_constraint_requires_the_decision_layer():
    builder = PromptBuilder.from_config(profile_path=PROFILE)
    engine = InputConstraintEngine.from_config(FakeJudge(9))

    with pytest.raises(ValueError, match="Decision Layer"):
        Pipeline(builder, ResponseGenerator(FakeGemma()), engine)