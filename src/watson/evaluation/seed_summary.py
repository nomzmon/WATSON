"""Combine repeated runs of an experiment made with different seeds (thesis Section 1.4, Phase 3).

A changed reply alters the rest of its conversation, so the conversation, not the turn, is the unit: each step's
persona, narrative and repetition scores are first averaged per conversation, then reported as the mean and
standard deviation across every conversation of every run. Counts (drift caught, normal turns blocked, fallbacks)
are summed over the runs.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from statistics import mean, stdev

from pydantic import BaseModel

from watson.evaluation.middleware_regression import TurnRecord, summarize_step


class Spread(BaseModel):
    mean: float
    sd: float  # sample standard deviation across conversations; 0 with a single conversation
    n: int  # conversations with a value

    def __str__(self) -> str:
        return f"{self.mean:.2f} ± {self.sd:.2f}"


class CombinedStep(BaseModel):
    step: int
    name: str
    runs: int
    persona_normal: Spread | None
    narrative_normal: Spread | None
    repetition: Spread | None
    drift_caught: int
    drift_total: int
    normal_blocked: int
    normal_total: int
    fallbacks: int
    mean_latency_s: float


def _spread(per_conversation: list[float | None]) -> Spread | None:
    values = [v for v in per_conversation if v is not None]
    if not values:
        return None
    return Spread(mean=round(mean(values), 2), sd=round(stdev(values), 2) if len(values) > 1 else 0.0, n=len(values))


def combine_runs(runs: Sequence[list[TurnRecord]]) -> list[CombinedStep]:
    combined = []
    for step in sorted({r.step for run in runs for r in run}):
        groups = [[r for r in run if r.step == step] for run in runs]
        groups = [g for g in groups if g]
        names = {g[0].step_name for g in groups}
        if len(names) > 1:
            raise ValueError(f"step {step} has different names in different runs: {sorted(names)}")
        summaries = [summarize_step(step, g[0].step_name, g) for g in groups]
        conversations = [[r for r in g if r.dialogue_id == d] for g in groups for d in sorted({r.dialogue_id for r in g})]

        def per_conversation(field: str, normal_only: bool) -> list[float | None]:
            values = []
            for conversation in conversations:
                scores = [getattr(r, field) for r in conversation
                          if getattr(r, field) is not None and not (normal_only and r.kind.is_drift)]
                values.append(mean(scores) if scores else None)
            return values

        combined.append(CombinedStep(
            step=step,
            name=summaries[0].name,
            runs=len(groups),
            persona_normal=_spread(per_conversation("persona_score", normal_only=True)),
            narrative_normal=_spread(per_conversation("narrative_score", normal_only=True)),
            repetition=_spread(per_conversation("repetition", normal_only=False)),
            drift_caught=sum(s.drift_caught for s in summaries),
            drift_total=sum(s.drift_total for s in summaries),
            normal_blocked=sum(s.normal_blocked for s in summaries),
            normal_total=sum(s.normal_total for s in summaries),
            fallbacks=sum(s.fallbacks for s in summaries),
            mean_latency_s=round(mean(s.mean_latency_ms for s in summaries) / 1000, 1),
        ))
    return combined


def render_combined(started: datetime, sources: Sequence[str], combined: Sequence[CombinedStep]) -> str:
    lines = [
        f"# Combined runs: {started:%Y-%m-%d %H:%M}",
        "",
        "Runs combined:",
        "",
        *[f"- `{source}`" for source in sources],
        "",
        "Scores are averaged per conversation first, then given as mean ± standard deviation across every "
        "conversation of every run (n = number of conversations). Counts are summed over the runs.",
        "",
        "| Step | Name | Runs | Persona (normal turns) | Narrative (normal turns) | Repetition | Drift caught "
        "| Normal blocked | Fallbacks | Avg time per turn |",
        "|---" * 10 + "|",
    ]

    def cell(spread: Spread | None) -> str:
        return f"{spread} (n={spread.n})" if spread else "-"

    for c in combined:
        lines.append(
            f"| {c.step} | {c.name} | {c.runs} | {cell(c.persona_normal)} | {cell(c.narrative_normal)} "
            f"| {cell(c.repetition)} | {c.drift_caught}/{c.drift_total} | {c.normal_blocked}/{c.normal_total} "
            f"| {c.fallbacks} | {c.mean_latency_s:.1f} s |"
        )
    return "\n".join(lines) + "\n"