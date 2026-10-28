import json

import pytest

from watson.common.schemas import DecisionType, DialogueStage
from watson.llm.base import LLMClient
from watson.middleware.decision.decision import load_decision_thresholds
from watson.middleware.decision.messages import DecisionMessages
from watson.middleware.decision.rephrase import Rephraser   
from watson.middleware.dialogue_flow.manager import DialogueFlowManager
from watson.middleware.generation.generator import ResponseGenerator
from watson.middleware.input_constraint.engine import InputConstraintEngine
from watson.middleware.output_validation.retry import OutputValidationEngine
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

def test_the_session_visitor_reaches_the_generation_prompt():
    gemma = FakeGemma()
    builder = PromptBuilder.from_config(profile_path=PROFILE, template_path="prompts/generation/structured_prompt.txt")
    session = ConversationSession(case_context="The pearl necklace affair", visitor="Lady Ashworth, your client.")

    Pipeline(builder, ResponseGenerator(gemma)).run_turn(session, "What of the footprints?")

    assert "Lady Ashworth, your client." in gemma.calls[0]["prompt"]

def test_accepted_message_is_scored_then_generated():
    gemma = FakeGemma()

    result = make_pipeline(gemma, wics_score=9).run_turn(ConversationSession(), "What of the footprints?")

    assert result.decision.decision == DecisionType.ACCEPT
    assert result.source == "generator"
    assert [latency.component for latency in result.latencies] == ["input_constraint", "generation"]


class RephraseJudge(FakeJudge):
    """Scores every message in the Rephrase band (6) and rewrites it when asked."""

    def __init__(self, rewrite: str) -> None:
        super().__init__(6)
        self.rewrite = rewrite
        self.rephrase_prompts: list[str] = []

    def _generate(self, prompt: str, **kwargs) -> str:
        if "User message to rewrite" in prompt:
            self.rephrase_prompts.append(prompt)
            return json.dumps({"rephrased": self.rewrite})
        return super()._generate(prompt, **kwargs)


def make_rephrasing_pipeline(gemma: FakeGemma, judge: RephraseJudge) -> Pipeline:
    builder = PromptBuilder.from_config(profile_path=PROFILE, template_path="prompts/generation/baseline_persona.txt")
    return Pipeline(builder, ResponseGenerator(gemma), InputConstraintEngine.from_config(judge),
                    load_decision_thresholds(), DecisionMessages.from_config(), rephraser=Rephraser.from_config(judge))


def test_rephrase_gives_holmes_the_rewritten_message_and_keeps_the_original_in_history():
    gemma, judge = FakeGemma(), RephraseJudge("What do you make of the open study window?")
    session = ConversationSession(case_context="The pearl necklace affair")
    session.add_exchange("The study window stands open.", "Indeed.")

    result = make_rephrasing_pipeline(gemma, judge).run_turn(session, "the window thing?")

    assert result.decision.decision == DecisionType.REPHRASE
    assert result.decision.rephrased_input == "What do you make of the open study window?"
    assert "What do you make of the open study window?" in gemma.calls[0]["prompt"]
    assert "the window thing?" not in gemma.calls[0]["prompt"]
    assert "Assistant: The study window stands open." in judge.rephrase_prompts[0]
    assert "The pearl necklace affair" in judge.rephrase_prompts[0]
    assert [t.text for t in session.history][-2:] == ["the window thing?", "Elementary."]
    assert [latency.component for latency in result.latencies] == ["input_constraint", "rephrase", "generation"]


def test_an_empty_rewrite_keeps_the_original_message():
    gemma = FakeGemma()

    result = make_rephrasing_pipeline(gemma, RephraseJudge("  ")).run_turn(ConversationSession(), "the window thing?")

    assert result.decision.rephrased_input == "the window thing?"
    assert "the window thing?" in gemma.calls[0]["prompt"]


def test_without_a_rephraser_a_rephrase_message_reaches_holmes_unchanged():
    gemma = FakeGemma()

    result = make_pipeline(gemma, wics_score=6).run_turn(ConversationSession(), "the window thing?")

    assert result.decision.decision == DecisionType.REPHRASE
    assert result.decision.rephrased_input is None
    assert "the window thing?" in gemma.calls[0]["prompt"]


def test_an_accepted_message_is_not_rephrased():
    gemma, judge = FakeGemma(), RephraseJudge("unused")
    judge.score = 9

    result = make_rephrasing_pipeline(gemma, judge).run_turn(ConversationSession(), "What of the footprints?")

    assert (result.decision.decision, result.decision.rephrased_input) == (DecisionType.ACCEPT, None)
    assert judge.rephrase_prompts == []


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
        
class OVSJudge(LLMClient):
    """Answers the Output Validation judge with the given OVS scores for every reply."""

    model_name = "fake-ovs-judge"

    def __init__(self, score: int) -> None:
        self.score = score
        self.calls = 0

    def _generate(self, prompt: str, **kwargs) -> str:
        self.calls += 1
        return json.dumps({c: {"reasoning": "r", "score": self.score} for c in ("PCC", "NC", "HWB", "SS")})


def make_validated_pipeline(gemma: FakeGemma, ovs_score: int, with_flow: bool = False) -> Pipeline:
    builder = PromptBuilder.from_config(profile_path=PROFILE, template_path="prompts/generation/structured_prompt.txt")
    flow = {"dialogue_flow": DialogueFlowManager.from_config(FlowJudge())} if with_flow else {}
    return Pipeline(builder, ResponseGenerator(gemma),
                    output_validation=OutputValidationEngine.from_config(OVSJudge(ovs_score), PROFILE), **flow)


def test_a_reply_that_passes_validation_is_delivered():
    gemma = FakeGemma()
    session = ConversationSession()

    result = make_validated_pipeline(gemma, ovs_score=9).run_turn(session, "What of the footprints?", seed=7)

    assert (result.source, result.reply) == ("generator", "Elementary.")
    assert len(result.validation.attempts) == 1
    assert [latency.component for latency in result.latencies] == ["generation", "output_validation"]


def test_failed_replies_are_retried_with_new_seeds_then_replaced_by_the_fallback():
    gemma = FakeGemma()
    session = ConversationSession()

    result = make_validated_pipeline(gemma, ovs_score=3).run_turn(session, "What of the footprints?", seed=7)

    assert result.source == "fallback_message"
    assert result.reply.startswith("Forgive me")
    assert [call["seed"] for call in gemma.calls] == [7, 1007, 2007]
    assert "- Every sentence must sound like Holmes" in gemma.calls[2]["prompt"]  # persona at level 3 by then
    assert session.history[-1].text == result.reply  # the fallback is in Holmes' voice, so it stays in the history


def test_validation_keeps_the_dialogue_flow_guidance_without_repeating_it():
    gemma = FakeGemma()

    make_validated_pipeline(gemma, ovs_score=9, with_flow=True).run_turn(ConversationSession(), "Who stole it?")

    prompt = gemma.calls[0]["prompt"]
    assert prompt.count("- Do not announce a final conclusion before the evidence supports it.") == 1
    assert "Current stage of the investigation: evidence" in prompt

class StageDirectionGemma(FakeGemma):
    def _generate(self, prompt: str, **kwargs) -> str:
        return f"(Holmes leans forward) {super()._generate(prompt, **kwargs)}"


def test_stage_directions_are_cut_before_the_reply_is_shown():
    session = ConversationSession()

    result = make_pipeline(StageDirectionGemma()).run_turn(session, "What of the footprints?")

    assert (result.reply, result.removed_from_reply) == ("Elementary.", ["(Holmes leans forward)"])
    assert session.history[-1].text == "Elementary."


def test_stage_directions_are_cut_before_validation_so_no_retry_is_needed():
    gemma = StageDirectionGemma()

    result = make_validated_pipeline(gemma, ovs_score=9).run_turn(ConversationSession(), "What of the footprints?")

    assert (result.reply, result.removed_from_reply) == ("Elementary.", ["(Holmes leans forward)"])
    assert len(gemma.calls) == 1
    assert result.validation.attempts[0].violations == []

def test_output_validation_needs_a_template_with_a_guidance_section():
    builder = PromptBuilder.from_config(profile_path=PROFILE, template_path="prompts/generation/baseline_persona.txt")

    with pytest.raises(ValueError, match="constraints"):
        Pipeline(builder, ResponseGenerator(FakeGemma()),
                 output_validation=OutputValidationEngine.from_config(OVSJudge(9), PROFILE))