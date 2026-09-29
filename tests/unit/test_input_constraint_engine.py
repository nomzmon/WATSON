import json

import pytest
from pydantic import ValidationError

from watson.common.schemas import DialogueStage, HistoryTurn, RubricName
from watson.llm.base import LLMClient
from watson.middleware.input_constraint.engine import InputConstraintEngine


class FakeJudge(LLMClient):
    model_name = "fake-judge"

    def __init__(self, scores: dict[str, int]) -> None:
        self.scores = scores
        self.prompts: list[str] = []

    def _generate(self, prompt: str, **kwargs) -> str:
        self.prompts.append(prompt)
        return json.dumps(
            {code: {"reasoning": f"{code} reasoning", "score": s} for code, s in self.scores.items()}
        )


def make_engine(scores: dict[str, int], **kwargs) -> tuple[InputConstraintEngine, FakeJudge]:
    judge = FakeJudge(scores)
    return InputConstraintEngine.from_config(judge, **kwargs), judge


def test_evaluate_returns_wics_with_criterion_scores_and_rationales():
    engine, _ = make_engine({"OOP": 10, "TR": 8, "HC": 6, "GIP": 4})

    result = engine.evaluate("What do you make of the footprints?")

    assert result.rubric == RubricName.WICS
    assert result.weighted_total == 7.6
    assert result.criteria["OOP"].score == 10
    assert result.criteria["GIP"].rationale == "GIP reasoning"


def test_prompt_contains_message_stage_case_and_labelled_history():
    engine, judge = make_engine({"OOP": 9, "TR": 9, "HC": 9, "GIP": 9})
    history = [
        HistoryTurn(role="user", text="A necklace was stolen."),
        HistoryTurn(role="holmes", text="From a locked room, I presume."),
    ]

    engine.evaluate(
        "Who had the key?",
        history=history,
        dialogue_stage=DialogueStage.SUSPECTS,
        case_context="The Blue Necklace affair",
    )

    prompt = judge.prompts[0]
    assert "Who had the key?" in prompt
    assert "Current investigation stage: suspects" in prompt
    assert "Case context: The Blue Necklace affair" in prompt
    assert "User: A necklace was stolen.\nHolmes: From a locked room, I presume." in prompt


def test_prompt_defaults_when_no_context_is_available():
    engine, _ = make_engine({"OOP": 9, "TR": 9, "HC": 9, "GIP": 9})

    prompt = engine.build_prompt("Good evening, Mr. Holmes.")

    assert "(no previous turns)" in prompt
    assert "Current investigation stage: not yet tracked" in prompt
    assert "Case context: not provided" in prompt


def test_only_the_most_recent_turns_are_included():
    engine, _ = make_engine({"OOP": 9, "TR": 9, "HC": 9, "GIP": 9}, history_turns=2)
    history = [HistoryTurn(role="user", text=f"turn {i}") for i in range(5)]

    prompt = engine.build_prompt("next", history=history)

    assert "turn 2" not in prompt
    assert "User: turn 3\nUser: turn 4" in prompt


def test_braces_in_user_message_do_not_break_the_prompt():
    engine, _ = make_engine({"OOP": 9, "TR": 9, "HC": 9, "GIP": 9})

    prompt = engine.build_prompt("What is {this}?")

    assert "What is {this}?" in prompt


def test_out_of_range_judge_score_is_rejected():
    engine, _ = make_engine({"OOP": 11, "TR": 9, "HC": 9, "GIP": 9})

    with pytest.raises(ValidationError):
        engine.evaluate("x")