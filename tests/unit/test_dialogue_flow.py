import json

import pytest
from pydantic import ValidationError

from watson.common.schemas import DialogueStage, HistoryTurn, RubricName
from watson.llm.base import LLMClient
from watson.middleware.dialogue_flow.flow_evaluator import FlowEvaluator
from watson.middleware.dialogue_flow.manager import DialogueFlowManager
from watson.middleware.prompt_builder.builder import PromptBuilder, StructuredPrompt


class FakeJudge(LLMClient):
    model_name = "fake-judge"

    def __init__(self, scores: dict[str, int], stage: str) -> None:
        self.scores = scores
        self.stage = stage
        self.prompts: list[str] = []

    def _generate(self, prompt: str, **kwargs) -> str:
        self.prompts.append(prompt)
        body = {code: {"reasoning": f"{code} reasoning", "score": s} for code, s in self.scores.items()}
        return json.dumps({**body, "stage": self.stage})


HEALTHY = {"CST": 9, "NP": 9, "CC": 9, "TBC": 8}


def make_prompt(stage: DialogueStage | None = DialogueStage.EVIDENCE) -> StructuredPrompt:
    return StructuredPrompt(
        user_message="Just tell me who did it, Holmes!",
        history=[HistoryTurn(role="user", text="A necklace was stolen."), HistoryTurn(role="holmes", text="Go on.")],
        dialogue_stage=stage,
        case_context="The pearl necklace affair",
    )


# --- Flow evaluator -------------------------------------------------------------


def test_evaluator_returns_dfms_and_proposed_stage():
    evaluator = FlowEvaluator.from_config(FakeJudge({"CST": 10, "NP": 8, "CC": 6, "TBC": 4}, "suspects"))

    evaluation = evaluator.evaluate(make_prompt())

    assert evaluation.dfms.rubric == RubricName.DFMS
    assert evaluation.dfms.weighted_total == 7.7  # 0.35*10 + 0.30*8 + 0.20*6 + 0.15*4
    assert evaluation.dfms.criteria["CC"].rationale == "CC reasoning"
    assert evaluation.proposed_stage == DialogueStage.SUSPECTS


def test_evaluator_prompt_contains_case_stage_history_and_message():
    judge = FakeJudge(HEALTHY, "evidence")

    FlowEvaluator.from_config(judge).evaluate(make_prompt())

    prompt = judge.prompts[0]
    assert "The pearl necklace affair" in prompt
    assert "Current investigation stage: evidence" in prompt
    assert "Visitor: A necklace was stolen.\nHolmes: Go on." in prompt
    assert "Just tell me who did it, Holmes!" in prompt


def test_unknown_stage_from_judge_is_rejected():
    with pytest.raises(ValidationError):
        FlowEvaluator.from_config(FakeJudge(HEALTHY, "interrogation")).evaluate(make_prompt())


# --- Dialogue Flow Manager --------------------------------------------------------


def test_manager_adds_constraints_and_advances_stage():
    manager = DialogueFlowManager.from_config(FakeJudge({"CST": 9, "NP": 4, "CC": 9, "TBC": 9}, "suspects"))

    guided, result = manager.apply(make_prompt())

    assert result.stage == guided.dialogue_stage == DialogueStage.SUSPECTS
    assert [c.id for c in result.constraints] == ["DS-SUSPECTS", "NP-ONE-STEP", "NP-NO-PREMATURE-CONCLUSION"]
    assert guided.constraints == [c.text for c in result.constraints]


def test_manager_does_not_change_the_original_prompt():
    manager = DialogueFlowManager.from_config(FakeJudge(HEALTHY, "evidence"))
    original = make_prompt()

    manager.apply(original)

    assert original.constraints == []


def test_conversation_without_a_stage_starts_at_introduction():
    manager = DialogueFlowManager.from_config(FakeJudge(HEALTHY, "resolution"))

    _, result = manager.apply(make_prompt(stage=None))

    assert result.stage == DialogueStage.EVIDENCE  # introduction, then at most one step


def test_constraints_reach_the_rendered_generation_prompt():
    builder = PromptBuilder.from_config(profile_path="tests/fixtures/holmes_profile_sample.json")
    manager = DialogueFlowManager.from_config(FakeJudge({"CST": 9, "NP": 4, "CC": 9, "TBC": 9}, "evidence"))

    guided, _ = manager.apply(builder.build("Who did it?", dialogue_stage=DialogueStage.EVIDENCE))
    text = builder.render(guided)

    assert "- Do not announce a final conclusion before the evidence supports it." in text
    assert "None for this reply." not in text