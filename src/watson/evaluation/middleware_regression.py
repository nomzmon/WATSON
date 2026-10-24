"""Component-by-component middleware regression testing (iterative refinement).

Each step enables one more middleware component than the step before it. Every
step replays the same scripted conversations with the same seeds, and every
Holmes reply the visitor sees gets a persona consistency score and a narrative
consistency score. A step "works" if it improves what its component targets
(e.g. the Input Constraint Engine catching drift-inducing turns, the Dialogue
Flow Manager keeping the investigation coherent, the Output Validation Engine
catching bad replies); it regresses if the persona score on normal turns falls
compared with the previous step.
"""

from __future__ import annotations

import csv
import re
import types
import typing
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from statistics import mean

from pydantic import BaseModel, Field, field_validator, model_validator

from watson.common.config import load_text, load_yaml, resolve_path
from watson.common.schemas import DecisionType, DialogueStage, HistoryTurn, ModuleToggles, RubricScore
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
from watson.middleware.output_validation.retry import OutputValidationEngine, ValidationResult
from watson.middleware.output_validation.safety_filter import FilterViolation
from watson.middleware.output_validation.validator import OVS_CRITERIA
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
    visitor: str | None = None  # who Holmes is speaking with; only the generation prompt sees it
    turns: list[ScriptedTurn] = Field(min_length=1)


class StepConfig(BaseModel):
    name: str
    template: str
    modules: ModuleToggles = Field(default_factory=ModuleToggles)

    @field_validator("modules")
    @classmethod
    def _only_built_components(cls, modules: ModuleToggles) -> ModuleToggles:
        if modules.input_constraint != modules.decision_layer:
            raise ValueError("the Input Constraint Engine and the Decision Layer must be enabled together")
        return modules

    @model_validator(mode="after")
    def _guidance_needs_a_section_in_the_template(self) -> StepConfig:
        # Checked here so a bad step fails before the run starts, not halfway through it.
        guided = self.modules.dialogue_flow or self.modules.output_validation
        if guided and "{constraints}" not in load_text(self.template):
            raise ValueError("the Dialogue Flow Manager and Output Validation Engine need a template with a "
                             f"{{constraints}} section: {self.template}")
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
    proposed_stage: DialogueStage | None = None  # the stage the judge proposed; the tracker moves at most one step
    dfms_total: float | None = None
    dfms_cst: int | None = None
    dfms_np: int | None = None
    dfms_cc: int | None = None
    dfms_tbc: int | None = None
    guidance: str = ""  # selected constraints as "ID (level N)", comma-separated
    dfms_reasoning: str = ""
    # Output validation of the reply shown (or, after a fallback, of the last attempt); empty without the engine
    ovs_total: float | None = None
    ovs_pcc: int | None = None
    ovs_nc: int | None = None
    ovs_hwb: int | None = None
    ovs_ss: int | None = None
    ovs_first: float | None = None  # OVS of the first attempt
    ovs_first_scores: str = ""  # criterion scores of the first attempt, as "PCC 7, NC 4, HWB 10, SS 8"
    ovs_attempts: int | None = None  # replies generated, including the first
    filter_flags: str = ""  # what the safety filter flagged in the first reply, separated by "; "
    ovs_reasoning: str = ""
    rejected_replies: str = ""  # replies that failed validation, with their OVS, separated by blank lines
    source: str
    reply: str
    removed_from_reply: str = ""  # stage directions the Response Generation Layer cut from the reply, "; "-separated
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

    def ovs_scores(self) -> dict[str, int | None]:
        return {code: getattr(self, f"ovs_{code.lower()}") for code in OVS_CRITERIA}

    @property
    def reached_holmes(self) -> bool:
        """The visitor got a reply in Holmes' voice from the generator (or its fallback), not a fixed message."""
        return self.source in ("generator", "fallback_message")


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
    fallbacks: int = 0  # turns where every reply failed validation and the fallback was shown
    directions_removed: int = 0  # replies shown with stage directions cut out


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
    proposed_stages: dict[DialogueStage, int] = Field(default_factory=dict)  # how often the judge proposed each stage


class ValidationSummary(BaseModel):
    """What the Output Validation Engine did in one step."""

    step: int
    turns: int  # turns whose reply was validated
    passed_first: int  # first reply passed
    passed_after_retry: int  # a regenerated reply passed
    fallbacks: int  # every reply failed, so the fallback was shown
    filtered_first: int = 0  # first replies the safety filter flagged
    mean_attempts: float
    first_ovs: float  # mean OVS of the first reply
    final_ovs: float  # mean OVS of the reply shown (or the last attempt, after a fallback)
    first_criteria: dict[str, float]  # mean criterion scores of the first reply


def load_experiment_config(path: str | Path = "configs/experiments/middleware_regression.yaml") -> ExperimentConfig:
    return ExperimentConfig.model_validate(load_yaml(path))


def load_dialogues(directory: str | Path) -> list[ScriptedDialogue]:
    dialogues = [ScriptedDialogue.model_validate(load_yaml(f)) for f in sorted(resolve_path(directory).glob("*.yaml"))]
    if not dialogues:
        raise ValueError(f"no scripted dialogues found in {directory}")
    return dialogues


def load_turn_records(path: str | Path) -> list[TurnRecord]:
    """Read back the turns.csv written by scripts/run_middleware_regression.py (empty cells become None)."""
    nullable = {
        name for name, field in TurnRecord.model_fields.items()
        if isinstance(field.annotation, types.UnionType) and type(None) in typing.get_args(field.annotation)
    }
    with resolve_path(path).open(encoding="utf-8", newline="") as f:
        return [
            TurnRecord.model_validate(
                {k: None if v == "" and k in nullable else v for k, v in row.items() if k in TurnRecord.model_fields}
            )
            for row in csv.DictReader(f)
        ]


def history_before_each_turn(records: list[TurnRecord]) -> dict[tuple[int, str, int], list[HistoryTurn]]:
    """The conversation history each turn of a run was answered after, keyed by (step, dialogue_id, turn).

    Rebuilt the way the pipeline keeps it: generated, fallback and redirected replies enter the history, rejected
    turns don't.
    """
    histories, running = {}, {}
    for r in sorted(records, key=lambda r: (r.step, r.dialogue_id, r.turn)):
        history = running.setdefault((r.step, r.dialogue_id), [])
        histories[(r.step, r.dialogue_id, r.turn)] = list(history)
        if r.source in ("generator", "fallback_message", "redirect_message"):
            history += [HistoryTurn(role="user", text=r.message), HistoryTurn(role="holmes", text=r.reply)]
    return histories


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
    if step.modules.output_validation:
        components["output_validation"] = OutputValidationEngine.from_config(judge, profile_path)
    generator = ResponseGenerator(generator.client, generator.stop, clean=step.modules.response_generation)
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
        session = ConversationSession(case_context=dialogue.case_context, visitor=dialogue.visitor)
        for turn_index, turn in enumerate(dialogue.turns):
            history_before = list(session.history)
            result = pipeline.run_turn(session, turn.message, seed=seed + 100 * dialogue_index + turn_index)
            generated = result.source in ("generator", "fallback_message")  # replies in Holmes' voice
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
                **_validation_fields(result.validation),
                source=result.source,
                reply=result.reply,
                removed_from_reply="; ".join(result.removed_from_reply),
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
        "proposed_stage": flow.proposed_stage,
        "dfms_total": flow.dfms.weighted_total,
        **{f"dfms_{c.code.lower()}": int(c.score) for c in criteria},
        "guidance": ", ".join(f"{c.id} (level {c.level})" for c in flow.constraints),
        "dfms_reasoning": "\n".join(f"{c.code} {c.score:g}: {c.rationale}" for c in criteria),
    }


def _validation_fields(validation: ValidationResult | None) -> dict:
    if validation is None:
        return {}
    final = validation.attempts[-1].ovs
    criteria = [final.criteria[code] for code in OVS_CRITERIA]
    rejected = [a for a in validation.attempts if not a.passed]
    return {
        "ovs_total": final.weighted_total,
        **{f"ovs_{c.code.lower()}": int(c.score) for c in criteria},
        "ovs_first": validation.attempts[0].ovs.weighted_total,
        "ovs_first_scores": _format_scores(validation.attempts[0].ovs),
        "ovs_attempts": len(validation.attempts),
        "filter_flags": "; ".join(v.describe() for v in validation.attempts[0].violations),
        "ovs_reasoning": "\n".join(f"{c.code} {c.score:g}: {c.rationale}" for c in criteria),
        "rejected_replies": "\n\n".join(
            f"OVS {a.ovs.weighted_total:.2f} ({_format_scores(a.ovs)}){_flagged(a.violations)}: {_one_line(a.reply)}"
            for a in rejected
        ),
    }

def _one_line(reply: str) -> str:
    """Line breaks joined, so a rejected reply with blank lines stays one entry in rejected_replies."""
    return " ".join(reply.split())

def _flagged(violations: list[FilterViolation]) -> str:
    return f" [filter: {'; '.join(v.describe() for v in violations)}]" if violations else ""


def _format_scores(ovs: RubricScore) -> str:
    return ", ".join(f"{code} {ovs.criteria[code].score:g}" for code in OVS_CRITERIA)


def _parse_scores(text: str) -> dict[str, float]:
    """The reverse of _format_scores: "PCC 7, NC 4" -> {"PCC": 7.0, "NC": 4.0}."""
    return {code: float(score) for code, score in (part.split() for part in text.split(", "))}


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
            proposed_stages={
                stage: count for stage in DialogueStage
                if (count := sum(r.proposed_stage == stage for r in group))
            },
        ))
    return summaries


def summarize_output_validation(records: list[TurnRecord]) -> list[ValidationSummary]:
    """Per step with the Output Validation Engine: first-try passes, retries, fallbacks and mean OVS."""
    summaries = []
    validated = [r for r in records if r.ovs_total is not None]
    for step in sorted({r.step for r in validated}):
        group = [r for r in validated if r.step == step]
        fallbacks = sum(r.source == "fallback_message" for r in group)
        passed_first = sum(r.ovs_attempts == 1 and r.source == "generator" for r in group)
        first_scores = [_parse_scores(r.ovs_first_scores) for r in group]
        summaries.append(ValidationSummary(
            step=step,
            turns=len(group),
            passed_first=passed_first,
            passed_after_retry=len(group) - passed_first - fallbacks,
            fallbacks=fallbacks,
            filtered_first=sum(bool(r.filter_flags) for r in group),
            mean_attempts=round(mean(r.ovs_attempts for r in group), 2),
            first_ovs=round(mean(r.ovs_first for r in group), 2),
            final_ovs=round(mean(r.ovs_total for r in group), 2),
            first_criteria={code: round(mean(s[code] for s in first_scores), 2) for code in OVS_CRITERIA},
        ))
    return summaries


def summarize_step(step_index: int, name: str, records: list[TurnRecord]) -> StepSummary:
    def average(values: list[float | None]) -> float | None:
        values = [v for v in values if v is not None]
        return round(mean(values), 2) if values else None

    blocked = [r for r in records if r.source in ("redirect_message", "reject_message")]
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
        fallbacks=sum(r.source == "fallback_message" for r in records),
        directions_removed=sum(bool(r.removed_from_reply) for r in records),
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