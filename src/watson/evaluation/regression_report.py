"""Markdown report of a middleware regression run, for documenting progress."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from datetime import datetime

from pydantic import BaseModel

from watson.common.config import PROJECT_ROOT
from watson.evaluation.middleware_regression import (
    ExperimentConfig,
    ScriptedDialogue,
    StepRegression,
    StepSummary,
    TurnRecord,
)


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
        "## Regressions",
        "",
    ]
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


def _turn(record: TurnRecord) -> list[str]:
    outcome = [record.decision.value if record.decision else "no input check"]
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
    if record.persona_reasoning:
        lines += [f"_Judge: {record.persona_reasoning}_", ""]
    return lines


def _quote(text: str) -> list[str]:
    return [f"> {line}" if line else ">" for line in text.splitlines()]


def _score(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else "-"