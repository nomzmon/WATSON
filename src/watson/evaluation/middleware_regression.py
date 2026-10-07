"""Component-by-component middleware regression testing (iterative refinement).

Each step enables one more middleware component than the step before it. Every
step replays the same scripted conversations with the same seeds, and every
generated Holmes reply gets a persona consistency score. A step "works" if it
improves what its component targets (e.g. the Input Constraint Engine catching
drift-inducing turns); it regresses if the persona score on normal turns falls
compared with the previous step.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from statistics import mean

from pydantic import BaseModel, Field, field_validator

from watson.common.config import load_yaml, resolve_path
from watson.common.schemas import DecisionType, ModuleToggles
from watson.evaluation.persona_consistency import PersonaConsistencyJudge
from watson.llm.base import LLMClient
from watson.middleware.decision.decision import load_decision_thresholds
from watson.middleware.decision.messages import DecisionMessages
from watson.middleware.generation.generator import ResponseGenerator
from watson.middleware.input_constraint.engine import InputConstraintEngine
from watson.middleware.prompt_builder.builder import PromptBuilder
from watson.pipeline.orchestrator import Pipeline
from watson.pipeline.session import ConversationSession


class TurnKind(StrEnum):
    NORMAL = "normal"
    RUSHING = "rushing"
    OFF_TOPIC = "off_topic"
    ANACHRONISM = "anachronism"
    PERSONA_BREAKING = "persona_breaking"

    @property
    def is_drift(self) -> bool:
        """Drift-inducing turns are the ones the Input Constraint Engine should stop."""
        return self in (TurnKind.OFF_TOPIC, TurnKind.ANACHRONISM, TurnKind.PERSONA_BREAKING)


class ScriptedTurn(BaseModel):
    kind: TurnKind
    message: str


class ScriptedDialogue(BaseModel):
    id: str
    title: str
    case_context: str
    turns: list[ScriptedTurn] = Field(min_length=1)


class StepConfig(BaseModel):
    name: str
    template: str
    modules: ModuleToggles = Field(default_factory=ModuleToggles)

    @field_validator("modules")
    @classmethod
    def _only_built_components(cls, modules: ModuleToggles) -> ModuleToggles:
        if modules.dialogue_flow or modules.output_validation:
            raise ValueError("the Dialogue Flow Manager and Output Validation Engine are not built yet")
        if modules.input_constraint != modules.decision_layer:
            raise ValueError("the Input Constraint Engine and the Decision Layer must be enabled together")
        return modules


class ExperimentConfig(BaseModel):
    profile: str
    dialogues: str
    seed: int
    regression_tolerance: float
    steps: list[StepConfig] = Field(min_length=1)


class TurnRecord(BaseModel):
    step: int
    step_name: str
    dialogue_id: str
    turn: int
    kind: TurnKind
    message: str
    decision: DecisionType | None
    source: str
    reply: str
    persona_score: int | None
    persona_reasoning: str
    latency_ms: float


class StepSummary(BaseModel):
    step: int
    name: str
    persona_normal: float | None  # mean persona score of generated replies to normal and rushing turns
    persona_drift: float | None  # mean persona score of generated replies to drift-inducing turns
    drift_caught: int  # drift-inducing turns redirected or rejected
    drift_total: int
    normal_blocked: int  # normal or rushing turns redirected or rejected
    normal_total: int
    mean_latency_ms: float


class StepRegression(BaseModel):
    step: int
    drop: float


def load_experiment_config(path: str | Path = "configs/experiments/middleware_regression.yaml") -> ExperimentConfig:
    return ExperimentConfig.model_validate(load_yaml(path))


def load_dialogues(directory: str | Path) -> list[ScriptedDialogue]:
    dialogues = [ScriptedDialogue.model_validate(load_yaml(f)) for f in sorted(resolve_path(directory).glob("*.yaml"))]
    if not dialogues:
        raise ValueError(f"no scripted dialogues found in {directory}")
    return dialogues


def build_pipeline(step: StepConfig, profile_path: str, generator: ResponseGenerator, judge: LLMClient) -> Pipeline:
    builder = PromptBuilder.from_config(profile_path=profile_path, template_path=step.template)
    if not step.modules.input_constraint:
        return Pipeline(builder, generator)
    return Pipeline(
        builder,
        generator,
        InputConstraintEngine.from_config(judge),
        load_decision_thresholds(),
        DecisionMessages.from_config(),
    )


def run_step(
    step_index: int,
    step: StepConfig,
    pipeline: Pipeline,
    persona_judge: PersonaConsistencyJudge,
    dialogues: list[ScriptedDialogue],
    seed: int,
    on_record: Callable[[TurnRecord], None] = lambda record: None,
) -> list[TurnRecord]:
    records = []
    for dialogue_index, dialogue in enumerate(dialogues):
        session = ConversationSession(case_context=dialogue.case_context)
        for turn_index, turn in enumerate(dialogue.turns):
            history_before = list(session.history)
            result = pipeline.run_turn(session, turn.message, seed=seed + 100 * dialogue_index + turn_index)
            judgement = (
                persona_judge.score(history_before, turn.message, result.reply)
                if result.source == "generator"
                else None
            )
            record = TurnRecord(
                step=step_index,
                step_name=step.name,
                dialogue_id=dialogue.id,
                turn=turn_index + 1,
                kind=turn.kind,
                message=turn.message,
                decision=result.decision.decision if result.decision else None,
                source=result.source,
                reply=result.reply,
                persona_score=judgement.score if judgement else None,
                persona_reasoning=judgement.reasoning if judgement else "",
                latency_ms=sum(latency.ms for latency in result.latencies),
            )
            on_record(record)
            records.append(record)
    return records


def summarize_step(step_index: int, name: str, records: list[TurnRecord]) -> StepSummary:
    def persona(drift: bool) -> float | None:
        scores = [r.persona_score for r in records if r.kind.is_drift == drift and r.persona_score is not None]
        return round(mean(scores), 2) if scores else None

    blocked = [r for r in records if r.source != "generator"]
    return StepSummary(
        step=step_index,
        name=name,
        persona_normal=persona(drift=False),
        persona_drift=persona(drift=True),
        drift_caught=sum(r.kind.is_drift for r in blocked),
        drift_total=sum(r.kind.is_drift for r in records),
        normal_blocked=sum(not r.kind.is_drift for r in blocked),
        normal_total=sum(not r.kind.is_drift for r in records),
        mean_latency_ms=round(mean(r.latency_ms for r in records), 1),
    )


def find_regressions(summaries: list[StepSummary], tolerance: float) -> list[StepRegression]:
    regressions = []
    for previous, current in zip(summaries, summaries[1:]):
        if previous.persona_normal is None or current.persona_normal is None:
            continue
        drop = round(previous.persona_normal - current.persona_normal, 2)
        if drop > tolerance:
            regressions.append(StepRegression(step=current.step, drop=drop))
    return regressions