import json

import pytest

from watson.common.schemas import DecisionType, DialogueStage
from watson.llm.base import LLMClient
from watson.middleware.decision.decision import load_decision_thresholds
from watson.middleware.decision.messages import DecisionMessages
from watson.middleware.dialogue_flow.manager import DialogueFlowManager
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
        
class FlowJudge(LLMClient):
    """Answers the Dialogue Flow Manager's judge: weak narrative progression, investigation at the evidence stage."""

    model_name = "fake-flow-judge"

    def __init__(self) -> None:
        self.calls = 0

    def _generate(self, prompt: str, **kwargs) -> str:
        self.calls += 1
        scores = {"CST": 8, "NP": 3, "CC": 8, "TBC": 9}
        return json.dumps({**{c: {"reasoning": "r", "score": s} for c, s in scores.items()}, "stage": "evidence"})


def make_flow_pipeline(gemma: FakeGemma, flow_judge: FlowJudge, wics_score: int | None = None) -> Pipeline:
    builder = PromptBuilder.from_config(profile_path=PROFILE, template_path="prompts/generation/structured_prompt.txt")
    checks = {}
    if wics_score is not None:
        checks = dict(input_constraint=InputConstraintEngine.from_config(FakeJudge(wics_score)),
                      decision_thresholds=load_decision_thresholds(),
                      decision_messages=DecisionMessages.from_config())
    return Pipeline(builder, ResponseGenerator(gemma), dialogue_flow=DialogueFlowManager.from_config(flow_judge),
                    **checks)


def test_dialogue_flow_adds_guidance_to_the_prompt_and_tracks_the_stage():
    gemma = FakeGemma()
    session = ConversationSession(case_context="The pearl necklace affair")

    result = make_flow_pipeline(gemma, FlowJudge()).run_turn(session, "Who stole it?")

    # From introduction, the stage advances one step to evidence; weak NP adds narrative progression guidance.
    assert session.dialogue_stage == result.flow.stage == DialogueStage.EVIDENCE
    assert "NP-NO-PREMATURE-CONCLUSION" in [c.id for c in result.flow.constraints]
    assert "Current stage of the investigation: evidence" in gemma.calls[0]["prompt"]
    assert "- Do not announce a final conclusion before the evidence supports it." in gemma.calls[0]["prompt"]
    assert [latency.component for latency in result.latencies] == ["dialogue_flow", "generation"]


def test_rejected_message_never_reaches_the_dialogue_flow_manager():
    flow_judge = FlowJudge()

    result = make_flow_pipeline(FakeGemma(), flow_judge, wics_score=2).run_turn(ConversationSession(), "Stop.")

    assert result.source == "reject_message"
    assert (result.flow, flow_judge.calls) == (None, 0)


def test_dialogue_flow_needs_a_template_with_a_guidance_section():
    builder = PromptBuilder.from_config(profile_path=PROFILE, template_path="prompts/generation/baseline_persona.txt")

    with pytest.raises(ValueError, match="constraints"):
        Pipeline(builder, ResponseGenerator(FakeGemma()), dialogue_flow=DialogueFlowManager.from_config(FlowJudge()))