"""Compare the rubric weights with another set of weights on finished runs (thesis Sections 5.2.1-5.2.3).

The weights only combine the judges' criterion scores into the WICS, DFMS and OVS totals; the judges never see
them. So every decision a run made can be recomputed from the criterion scores in its turns.csv under other
weights, without calling the models again:
- WICS: the Decision Layer's accept / rephrase / redirect / reject;
- DFMS: whether the Dialogue Flow Manager adds guidance, and at which strength;
- OVS: whether the first reply passes output validation.
This shows how many decisions the weights decide. It does not show how a changed decision would have changed the
rest of the conversation; a live run with the other weights would be needed for that.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from statistics import mean

from pydantic import BaseModel

from watson.common.rubric import weighted_score
from watson.common.schemas import CriterionScore, RubricName, RubricScore
from watson.evaluation.middleware_regression import TurnRecord, _parse_scores
from watson.middleware.decision.decision import decide, load_decision_thresholds
from watson.middleware.dialogue_flow.constraint_selector import load_selection_thresholds
from watson.middleware.dialogue_flow.flow_evaluator import load_dfms_weights
from watson.middleware.input_constraint.scoring import load_wics_weights
from watson.middleware.output_validation.validator import load_ovs_weights, load_validation_settings, passes

Weights = Mapping[str, float]


class RubricComparison(BaseModel):
    rubric: str
    decision: str  # what the weighted total decides
    weights: dict[str, float]
    variant_weights: dict[str, float]
    turns: int
    mean_total: float
    variant_mean_total: float
    changed: list[str]  # "2026-11-02_2236, step 4, dialogue_01 turn 7 (normal): accept -> rephrase"


def _rubric_score(name: RubricName, scores: Mapping[str, float], weights: Weights) -> RubricScore:
    criteria = {code: CriterionScore(code=code, score=score) for code, score in scores.items()}
    return RubricScore(rubric=name, criteria=criteria, weighted_total=weighted_score(scores, weights))


def _compare(rubric: str, decision: str, runs: Mapping[str, Sequence[TurnRecord]], judged: Callable[[TurnRecord], bool],
             scores: Callable[[TurnRecord], dict], weights: Weights, variant: Weights,
             decide_one: Callable[[dict, Weights], str]) -> RubricComparison:
    records = [(run, r) for run, run_records in runs.items() for r in run_records if judged(r)]
    changed = []
    for run, r in records:
        before, after = decide_one(scores(r), weights), decide_one(scores(r), variant)
        if before != after:
            changed.append(f"{run}, step {r.step}, {r.dialogue_id} turn {r.turn} ({r.kind.value}): {before} -> {after}")
    totals = [(weighted_score(scores(r), weights), weighted_score(scores(r), variant)) for _, r in records]
    return RubricComparison(
        rubric=rubric, decision=decision, weights=dict(weights), variant_weights=dict(variant), turns=len(records),
        mean_total=round(mean(t for t, _ in totals), 2) if totals else 0.0,
        variant_mean_total=round(mean(v for _, v in totals), 2) if totals else 0.0,
        changed=changed,
    )


def compare_weights(runs: Mapping[str, Sequence[TurnRecord]], wics_variant: Weights, dfms_variant: Weights,
                    ovs_variant: Weights) -> list[RubricComparison]:
    """`runs` maps a run's name (its folder) to its turns."""
    thresholds, selection, validation = load_decision_thresholds(), load_selection_thresholds(), load_validation_settings()

    def wics_decision(scores: dict, weights: Weights) -> str:
        return decide(_rubric_score(RubricName.WICS, scores, weights), thresholds).decision.value

    def dfms_guidance(scores: dict, weights: Weights) -> str:
        total = weighted_score(scores, weights)
        if total >= selection.guidance_below:
            return "no guidance"
        return "strong guidance" if total < selection.strong_below else "guidance"

    def ovs_result(scores: dict, weights: Weights) -> str:
        return "pass" if passes(_rubric_score(RubricName.OVS, scores, weights), validation) else "fail"

    return [
        _compare("WICS", "input decision", runs, lambda r: r.wics_total is not None,
                 TurnRecord.wics_scores, load_wics_weights(), wics_variant, wics_decision),
        _compare("DFMS", "guidance added", runs, lambda r: r.dfms_total is not None,
                 TurnRecord.dfms_scores, load_dfms_weights(), dfms_variant, dfms_guidance),
        _compare("OVS", "first reply passes", runs, lambda r: bool(r.ovs_first_scores),
                 lambda r: _parse_scores(r.ovs_first_scores), load_ovs_weights(), ovs_variant, ovs_result),
    ]


def render_weight_comparison(started: datetime, sources: Sequence[str | Path], variant: str,
                             comparisons: Sequence[RubricComparison]) -> str:
    def weights_text(weights: Mapping[str, float]) -> str:
        return ", ".join(f"{code} {weight:g}" for code, weight in weights.items())

    lines = [
        f"# Weight comparison: current weights against {variant} weights, {started:%Y-%m-%d %H:%M}",
        "",
        "Runs:",
        "",
        *[f"- `{source}`" for source in sources],
        "",
        "Each decision is recomputed from the judges' criterion scores recorded in the runs, once with the current "
        "weights and once with the variant; the judges' scores are the same in both. A changed decision shows where "
        "the weights matter; how it would have changed the rest of the conversation is not shown.",
        "",
        "| Rubric | Decision | Current weights | Variant weights | Turns | Mean total (current) "
        "| Mean total (variant) | Decisions changed |",
        "|---" * 8 + "|",
    ]
    for c in comparisons:
        lines.append(f"| {c.rubric} | {c.decision} | {weights_text(c.weights)} | {weights_text(c.variant_weights)} "
                     f"| {c.turns} | {c.mean_total:.2f} | {c.variant_mean_total:.2f} | {len(c.changed)} |")
    for c in comparisons:
        lines += ["", f"## {c.rubric}: changed decisions", ""]
        lines += [f"- {change}" for change in c.changed] or ["None."]
    return "\n".join(lines) + "\n"