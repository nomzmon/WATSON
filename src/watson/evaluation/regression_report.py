"""Markdown report of a middleware regression run, for documenting progress."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from datetime import datetime

from pydantic import BaseModel

from watson.common.config import PROJECT_ROOT
from watson.common.schemas import DecisionType
from watson.evaluation.middleware_regression import (
    ExperimentConfig,
    InputCheckSummary,
    ScriptedDialogue,
    StepRegression,
    StepSummary,
    TurnRecord,
)
from watson.middleware.input_constraint.scoring import WICS_CRITERIA


class RunDetails(BaseModel):
    started: datetime
    code_version: str
    generator_model: str
    judge_model: str


def code_version() -> str:
    """Short git commit of the working copy, flagged when there are uncommitted changes."""
    try:
        commit = _git("rev-parse", "--short", "HEAD")
        dirty = _git("status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.CalledProcessError):
        return "unknown (git not available)"
    return f"{commit} (with uncommitted changes)" if dirty else commit


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def render_report(
    details: RunDetails,
    config: ExperimentConfig,
    dialogues: Sequence[ScriptedDialogue],
    summaries: Sequence[StepSummary],
    regressions: Sequence[StepRegression],
    records: Sequence[TurnRecord],
    input_checks: Sequence[InputCheckSummary] = (),
) -> str:
    turns = sum(len(d.turns) for d in dialogues)
    lines = [
        f"# Middleware regression report: {details.started:%Y-%m-%d %H:%M}",
        "",
        "## Run details",
        "",
        f"- **Code version:** {details.code_version}",
        f"- **Generator:** {details.generator_model}",
        f"- **Judge:** {details.judge_model}",
        f"- **Persona profile:** `{config.profile}`",
        f"- **Conversations:** {len(dialogues)} ({turns} turns), from `{config.dialogues}`",
        f"- **Seed:** {config.seed}",
        f"- **Regression tolerance:** {config.regression_tolerance:g} (persona score, 1-10 scale)",
        "",
        "## Results by step",
        "",
        "| Step | Name | Components enabled | Persona (normal turns) | Persona (drift turns) | Drift caught "
        "| Normal blocked | Avg time per turn |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for summary, step in zip(summaries, config.steps):
        enabled = [name.replace("_", " ") for name, on in step.modules.model_dump().items() if on]
        lines.append(
            f"| {summary.step} | {summary.name} | {', '.join(enabled) or 'none'} "
            f"| {_score(summary.persona_normal)} | {_score(summary.persona_drift)} "
            f"| {summary.drift_caught}/{summary.drift_total} | {summary.normal_blocked}/{summary.normal_total} "
            f"| {summary.mean_latency_ms / 1000:.1f} s |"
        )
    lines += [
        "",
        "- **Persona (normal turns):** mean persona score (1-10) of Holmes' replies to normal and rushing turns.",
        "- **Persona (drift turns):** the same for drift-inducing turns that still reached Holmes.",
        "- **Drift caught:** drift-inducing turns the middleware redirected or rejected.",
        "- **Normal blocked:** normal turns the middleware wrongly redirected or rejected.",
        "",
    ]
    if input_checks:
        lines += _input_check_table(input_checks)
    lines += ["## Regressions", ""]
    if regressions:
        lines += [
            f"- Step {r.step}: persona score on normal turns dropped {r.drop:.2f} from the previous step."
            for r in regressions
        ]
    else:
        lines.append(f"None: no step lowered the persona score on normal turns by more than "
                     f"{config.regression_tolerance:g}.")
    lines += ["", "## Observations", "", "_To be filled in by the group: what changed, what to refine next._", ""]

    titles = {d.id: d.title for d in dialogues}
    lines += ["## Transcripts", ""]
    for summary in summaries:
        lines += [f"### Step {summary.step}: {summary.name}", ""]
        step_records = [r for r in records if r.step == summary.step]
        for dialogue_id in dict.fromkeys(r.dialogue_id for r in step_records):
            lines += [f"#### {dialogue_id}: {titles.get(dialogue_id, '')}", ""]
            for r in (r for r in step_records if r.dialogue_id == dialogue_id):
                lines += _turn(r)
    return "\n".join(lines)


def _input_check_table(summaries: Sequence[InputCheckSummary]) -> list[str]:
    decisions = list(DecisionType)
    lines = [
        "## Input check scores",
        "",
        "Mean WICS and criterion scores (1-10) per turn kind, and the decisions they led to.",
        "",
        f"| Step | Turn kind | Turns | WICS | {' | '.join(WICS_CRITERIA)} | "
        f"{' | '.join(d.value.capitalize() for d in decisions)} | Expected | As expected |",
        "|---" * (6 + len(WICS_CRITERIA) + len(decisions)) + "|",
    ]
    for s in summaries:
        lines.append(
            f"| {s.step} | {s.kind.value.replace('_', ' ')} | {s.turns} | {s.wics:.2f} | "
            + " | ".join(f"{s.criteria[code]:.1f}" for code in WICS_CRITERIA)
            + " | " + " | ".join(str(s.decisions[d]) for d in decisions)
            + f" | {' or '.join(d.value for d in s.expected) or '-'} | {_as_expected(s.as_expected, s.turns)} |"
        )
    lines += [
        "",
        "OOP = out-of-persona compatibility, TR = topic relevance, HC = historical consistency, "
        "GIP = guided input compliance. Expected decisions are set in the experiment config.",
        "",
    ]
    for step in dict.fromkeys(s.step for s in summaries):
        rated = [s for s in summaries if s.step == step and s.as_expected is not None]
        if rated:
            lines += [f"**Decisions as expected in step {step}:** "
                      f"{sum(s.as_expected for s in rated)}/{sum(s.turns for s in rated)}", ""]
    return lines


def _as_expected(count: int | None, turns: int) -> str:
    return f"{count}/{turns}" if count is not None else "-"


def _turn(record: TurnRecord) -> list[str]:
    outcome = [record.decision.value if record.decision else "no input check"]
    if record.wics_total is not None:
        criteria = ", ".join(f"{code} {score}" for code, score in record.wics_scores().items())
        outcome.append(f"WICS {record.wics_total:.2f} ({criteria})")
    if record.persona_score is not None:
        outcome.append(f"persona {record.persona_score}/10")
    speaker = "Holmes" if record.source in ("generator", "redirect_message") else "System"
    lines = [
        f"**Turn {record.turn}** · {record.kind.value.replace('_', ' ')} · {' · '.join(outcome)}",
        "",
        *_quote(f"**Visitor:** {record.message}"),
        ">",
        *_quote(f"**{speaker}:** {record.reply}"),
        "",
    ]
    if record.decision not in (None, DecisionType.ACCEPT):
        # Shows why a turn was not accepted, to guide refinement of the Input Constraint Engine.
        lines += [f"_Input check: {record.input_check_reasons}._", ""]
        lines += [f"- {line}" for line in record.wics_reasoning.splitlines()] + [""]
    if record.persona_reasoning:
        lines += [f"_Judge: {record.persona_reasoning}_", ""]
    return lines


def _quote(text: str) -> list[str]:
    return [f"> {line}" if line else ">" for line in text.splitlines()]


def _score(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else "-"