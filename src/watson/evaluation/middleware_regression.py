"""Component-by-component middleware regression testing (iterative refinement).

Each step enables one more middleware component than the step before it. Every
step replays the same scripted conversations with the same seeds, and every
generated Holmes reply gets a persona consistency score and a narrative
consistency score. A step "works" if it improves what its component targets
(e.g. the Input Constraint Engine catching drift-inducing turns, the Dialogue
Flow Manager keeping the investigation coherent); it regresses if the persona
score on normal turns falls compared with the previous step.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from statistics import mean

from pydantic import BaseModel, Field, field_validator, model_validator

from watson.common.config import load_text, load_yaml, resolve_path
from watson.common.schemas import DecisionType, DialogueStage, ModuleToggles, RubricScore
from watson.evaluation.narrative_consistency import NarrativeConsistencyJudge
from watson.evaluation.persona_consistency import PersonaConsistencyJudge
from watson.llm.base import LLMClient
from watson.middleware.decision.decision import load_decision_thresholds
from watson.middleware.decision.messages import DecisionMessages
from watson.middleware.dialogue_flow.flow_evaluator import DFMS_CRITERIA
from watson.middleware.dialogue_flow.manager import DialogueFlowManager, FlowResult
from watson.middleware.generation.generator import ResponseGenerator
from watson.middleware.input_constraint.engine import InputConstraintEngine
from watson.middleware.input_constraint.scoring import WICS_CRITERIA
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
        if modules.output_validation:
            raise ValueError("the Output Validation Engine is not built yet")
        if modules.input_constraint != modules.decision_layer:
            raise ValueError("the Input Constraint Engine and the Decision Layer must be enabled together")
        return modules

    @model_validator(mode="after")
    def _flow_needs_guidance_section(self) -> StepConfig:
        # Checked here so a bad step fails before the run starts, not halfway through it.
        if self.modules.dialogue_flow and "{constraints}" not in load_text(self.template):
            raise ValueError(f"the Dialogue Flow Manager needs a template with a {{constraints}} section: {self.template}")
        return self


class ExperimentConfig(BaseModel):
    profile: str
    dialogues: str
    seed: int
    regression_tolerance: float
    expected_decisions: dict[TurnKind, list[DecisionType]] = Field(default_factory=dict)
    steps: list[StepConfig] = Field(min_length=1)


class TurnRecord(BaseModel):
    step: int
    step_name: str
    dialogue_id: str
    turn: int
    kind: TurnKind
    message: str
    decision: DecisionType | None
    # Input check: the WICS total and criterion scores behind the decision (empty without the Input Constraint Engine)
    wics_total: float | None = None
    wics_oop: int | None = None
    wics_tr: int | None = None
    wics_hc: int | None = None
    wics_gip: int | None = None
    input_check_reasons: str = ""
    wics_reasoning: str = ""
    # Dialogue flow: the DFMS, the investigation stage and the guidance added (empty without the Dialogue Flow Manager)
    stage: DialogueStage | None = None
    dfms_total: float | None = None
    dfms_cst: int | None = None
    dfms_np: int | None = None
    dfms_cc: int | None = None
    dfms_tbc: int | None = None
    guidance: str = ""  # selected constraints as "ID (level N)", comma-separated
    dfms_reasoning: str = ""
    source: str
    reply: str
    persona_score: int | None
    persona_reasoning: str
    narrative_score: int | None = None
    narrative_reasoning: str = ""
    repetition: float | None = None  # share of the reply's word trigrams already used in Holmes' earlier replies
    latency_ms: float

    def wics_scores(self) -> dict[str, int | None]:
        return {code: getattr(self, f"wics_{code.lower()}") for code in WICS_CRITERIA}

    def dfms_scores(self) -> dict[str, int | None]:
        return {code: getattr(self, f"dfms_{code.lower()}") for code in DFMS_CRITERIA}


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
    narrative_normal: float | None = None  # mean narrative score of generated replies to normal and rushing turns
    repetition: float | None = None  # mean repetition of all generated replies


class StepRegression(BaseModel):
    step: int
    drop: float


class InputCheckSummary(BaseModel):
    """Mean WICS scores and decisions for one turn kind in one step."""

    step: int
    kind: TurnKind
    turns: int
    wics: float
    criteria: dict[str, float]  # mean score per WICS criterion
    decisions: dict[DecisionType, int]
    expected: list[DecisionType] = Field(default_factory=list)  # empty when the config sets no expectation
    as_expected: int | None = None  # turns whose decision was one of the expected ones


class FlowSummary(BaseModel):
    """What the Dialogue Flow Manager did in one step."""

    step: int
    turns: int  # turns evaluated by the Dialogue Flow Manager
    dfms: float
    criteria: dict[str, float]  # mean score per DFMS criterion
    guided_turns: int  # turns that received guidance beyond the stage constraint
    final_stages: dict[str, DialogueStage]  # stage each dialogue ended at


def load_experiment_config(path: str | Path = "configs/experiments/middleware_regression.yaml") -> ExperimentConfig:
    return ExperimentConfig.model_validate(load_yaml(path))


def load_dialogues(directory: str | Path) -> list[ScriptedDialogue]:
    dialogues = [ScriptedDialogue.model_validate(load_yaml(f)) for f in sorted(resolve_path(directory).glob("*.yaml"))]
    if not dialogues:
        raise ValueError(f"no scripted dialogues found in {directory}")
    return dialogues


def build_pipeline(step: StepConfig, profile_path: str, generator: ResponseGenerator, judge: LLMClient) -> Pipeline:
    builder = PromptBuilder.from_config(profile_path=profile_path, template_path=step.template)
    components = {}
    if step.modules.input_constraint:
        components.update(
            input_constraint=InputConstraintEngine.from_config(judge),
            decision_thresholds=load_decision_thresholds(),
            decision_messages=DecisionMessages.from_config(),
        )
    if step.modules.dialogue_flow:
        components["dialogue_flow"] = DialogueFlowManager.from_config(judge)
    return Pipeline(builder, generator, **components)


def run_step(
    step_index: int,
    step: StepConfig,
    pipeline: Pipeline,
    persona_judge: PersonaConsistencyJudge,
    dialogues: list[ScriptedDialogue],
    seed: int,
    on_record: Callable[[TurnRecord], None] = lambda record: None,
    narrative_judge: NarrativeConsistencyJudge | None = None,
) -> list[TurnRecord]:
    records = []
    for dialogue_index, dialogue in enumerate(dialogues):
        session = ConversationSession(case_context=dialogue.case_context)
        for turn_index, turn in enumerate(dialogue.turns):
            history_before = list(session.history)
            result = pipeline.run_turn(session, turn.message, seed=seed + 100 * dialogue_index + turn_index)
            generated = result.source == "generator"
            persona = persona_judge.score(history_before, turn.message, result.reply) if generated else None
            narrative = (
                narrative_judge.score(dialogue.case_context, history_before, turn.message, result.reply)
                if generated and narrative_judge
                else None
            )
            earlier_replies = [t.text for t in history_before if t.role == "holmes"]
            record = TurnRecord(
                step=step_index,
                step_name=step.name,
                dialogue_id=dialogue.id,
                turn=turn_index + 1,
                kind=turn.kind,
                message=turn.message,
                decision=result.decision.decision if result.decision else None,
                **_input_check_fields(result.wics, result.decision.reasons if result.decision else []),
                **_flow_fields(result.flow),
                source=result.source,
                reply=result.reply,
                persona_score=persona.score if persona else None,
                persona_reasoning=persona.reasoning if persona else "",
                narrative_score=narrative.score if narrative else None,
                narrative_reasoning=narrative.reasoning if narrative else "",
                repetition=repetition(result.reply, earlier_replies) if generated else None,
                latency_ms=sum(latency.ms for latency in result.latencies),
            )
            on_record(record)
            records.append(record)
    return records


def _input_check_fields(wics: RubricScore | None, reasons: list[str]) -> dict:
    if wics is None:
        return {}
    criteria = [wics.criteria[code] for code in WICS_CRITERIA]
    return {
        "wics_total": wics.weighted_total,
        **{f"wics_{c.code.lower()}": int(c.score) for c in criteria},
        "input_check_reasons": "; ".join(reasons),
        "wics_reasoning": "\n".join(f"{c.code} {c.score:g}: {c.rationale}" for c in criteria),
    }


def _flow_fields(flow: FlowResult | None) -> dict:
    if flow is None:
        return {}
    criteria = [flow.dfms.criteria[code] for code in DFMS_CRITERIA]
    return {
        "stage": flow.stage,
        "dfms_total": flow.dfms.weighted_total,
        **{f"dfms_{c.code.lower()}": int(c.score) for c in criteria},
        "guidance": ", ".join(f"{c.id} (level {c.level})" for c in flow.constraints),
        "dfms_reasoning": "\n".join(f"{c.code} {c.score:g}: {c.rationale}" for c in criteria),
    }


def _trigrams(text: str) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z']+", text.lower())
    return {tuple(words[i : i + 3]) for i in range(len(words) - 2)}


def repetition(reply: str, earlier_replies: list[str]) -> float:
    """Share of the reply's word trigrams that already appeared in Holmes' earlier replies (0 = all new)."""
    trigrams = _trigrams(reply)
    if not trigrams:
        return 0.0
    earlier = set().union(*(_trigrams(r) for r in earlier_replies))
    return round(len(trigrams & earlier) / len(trigrams), 2)


def summarize_input_checks(
    records: list[TurnRecord], expected_decisions: dict[TurnKind, list[DecisionType]] | None = None
) -> list[InputCheckSummary]:
    """Per step and turn kind, the mean WICS scores, the decisions they led to and how many were as expected."""
    expected_decisions = expected_decisions or {}
    summaries = []
    checked = [r for r in records if r.wics_total is not None]
    for step in sorted({r.step for r in checked}):
        for kind in TurnKind:
            group = [r for r in checked if r.step == step and r.kind == kind]
            if not group:
                continue
            expected = expected_decisions.get(kind, [])
            summaries.append(InputCheckSummary(
                step=step,
                kind=kind,
                turns=len(group),
                wics=round(mean(r.wics_total for r in group), 2),
                criteria={code: round(mean(r.wics_scores()[code] for r in group), 2) for code in WICS_CRITERIA},
                decisions={d: sum(r.decision == d for r in group) for d in DecisionType},
                expected=expected,
                as_expected=sum(r.decision in expected for r in group) if expected else None,
            ))
    return summaries


def summarize_dialogue_flow(records: list[TurnRecord]) -> list[FlowSummary]:
    """Per step with the Dialogue Flow Manager: mean DFMS, how often it added guidance and the stages reached."""
    summaries = []
    evaluated = [r for r in records if r.dfms_total is not None]
    for step in sorted({r.step for r in evaluated}):
        group = [r for r in evaluated if r.step == step]
        summaries.append(FlowSummary(
            step=step,
            turns=len(group),
            dfms=round(mean(r.dfms_total for r in group), 2),
            criteria={code: round(mean(r.dfms_scores()[code] for r in group), 2) for code in DFMS_CRITERIA},
            # The stage constraint (DS-...) is always added; anything else means the DFMS asked for guidance.
            guided_turns=sum(any(not g.startswith("DS-") for g in r.guidance.split(", ") if g) for r in group),
            final_stages={r.dialogue_id: r.stage for r in group},  # later turns overwrite earlier ones
        ))
    return summaries


def summarize_step(step_index: int, name: str, records: list[TurnRecord]) -> StepSummary:
    def average(values: list[float | None]) -> float | None:
        values = [v for v in values if v is not None]
        return round(mean(values), 2) if values else None

    blocked = [r for r in records if r.source != "generator"]
    normal = [r for r in records if not r.kind.is_drift]
    return StepSummary(
        step=step_index,
        name=name,
        persona_normal=average([r.persona_score for r in normal]),
        persona_drift=average([r.persona_score for r in records if r.kind.is_drift]),
        drift_caught=sum(r.kind.is_drift for r in blocked),
        drift_total=sum(r.kind.is_drift for r in records),
        normal_blocked=sum(not r.kind.is_drift for r in blocked),
        normal_total=len(normal),
        mean_latency_ms=round(mean(r.latency_ms for r in records), 1),
        narrative_normal=average([r.narrative_score for r in normal]),
        repetition=average([r.repetition for r in records]),
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