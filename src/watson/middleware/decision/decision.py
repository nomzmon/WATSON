"""Decision Layer (thesis Section 5.2.1.1, Table 5.2).

Two decisions are computed from a WICS result: one from the overall WICS and
one from the lowest single criterion. The stricter of the two wins, so a high
overall score cannot hide one critically low criterion. This also covers the
cases Table 5.2 leaves open, e.g. all criteria at 7 (WICS 7.0) -> Rephrase.

Optionally (`low_topic_relevance_redirects`), a message whose only criterion in
the reject band is Topic Relevance is redirected instead: it is off-topic but
keeps Holmes in his role and in the period, so the conversation can recover.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, model_validator

from watson.common.config import load_yaml
from watson.common.schemas import DecisionResult, DecisionType, RubricScore

_SEVERITY = (DecisionType.ACCEPT, DecisionType.REPHRASE, DecisionType.REDIRECT, DecisionType.REJECT)


class BandThresholds(BaseModel):
    """Minimum score for each decision; anything below `redirect` is Reject."""

    accept: float
    rephrase: float
    redirect: float

    @model_validator(mode="after")
    def _check_order(self) -> BandThresholds:
        if not self.accept > self.rephrase > self.redirect:
            raise ValueError("thresholds must satisfy accept > rephrase > redirect")
        return self

    def band(self, score: float) -> DecisionType:
        if score >= self.accept:
            return DecisionType.ACCEPT
        if score >= self.rephrase:
            return DecisionType.REPHRASE
        if score >= self.redirect:
            return DecisionType.REDIRECT
        return DecisionType.REJECT


class DecisionThresholds(BaseModel):
    wics: BandThresholds
    lowest_criterion: BandThresholds
    low_topic_relevance_redirects: bool = False


def load_decision_thresholds(path: str | Path = "configs/rubrics/wics.yaml") -> DecisionThresholds:
    return DecisionThresholds.model_validate(load_yaml(path)["decision"])


def decide(wics: RubricScore, thresholds: DecisionThresholds) -> DecisionResult:
    by_wics = thresholds.wics.band(wics.weighted_total)
    lowest = min(wics.criteria.values(), key=lambda criterion: criterion.score)
    by_criterion = thresholds.lowest_criterion.band(lowest.score)
    reasons = [
        f"WICS {wics.weighted_total:g} is in the {by_wics.value} band",
        f"lowest criterion {lowest.code}={lowest.score:g} is in the {by_criterion.value} band",
    ]
    rejected_by = [c.code for c in wics.criteria.values()
                   if thresholds.lowest_criterion.band(c.score) == DecisionType.REJECT]
    if thresholds.low_topic_relevance_redirects and rejected_by == ["TR"]:
        by_criterion = DecisionType.REDIRECT
        reasons.append("only TR is in the reject band, so the off-topic message is redirected")
    decision = max(by_wics, by_criterion, key=_SEVERITY.index)
    return DecisionResult(decision=decision, reasons=reasons)