"""Secondary filtering pass of the Output Validation Engine (thesis Section 3.5.3).

After the judge has scored a candidate reply, simple rules screen it for
content outside Holmes' knowledge and behavioural boundaries that can be
checked directly, without relying on the judge noticing it:

- Style and Syntax: stage directions in brackets or asterisks, name labels,
  narration such as "Holmes said", and quotation marks left open.
- Historical and World Boundary: modern terms from a list in the OVS config.

A flagged criterion is capped at a low score, so the reply fails validation
and is regenerated with that criterion's constraints strengthened.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel

from watson.common.config import load_yaml
from watson.common.rubric import weighted_score
from watson.common.schemas import CriterionScore, RubricScore

# Bracketed text with no first- or second-person word is Holmes being described, not Holmes speaking.
_BRACKETED = re.compile(r"\(([^()]*)\)|\[([^\[\]]*)\]|\*([^*\n]+\s[^*\n]+)\*")
_SPEAKER_WORDS = re.compile(r"\b(?:I|me|my|myself|we|us|our|you|your)\b", re.IGNORECASE)
_NAME_LABEL = re.compile(
    r"^\s*(?:Sherlock\s+Holmes|Holmes|Dr\.?\s+Watson|Watson|Visitor|User|Assistant)\s*:", re.IGNORECASE | re.MULTILINE
)
_NARRATION = re.compile(
    r"\b(?:said|says|replied|replies|asked|asks)\s+Holmes\b"
    r"|\bHolmes\s+(?:said|says|replied|replies|asked|asks|continued|continues|added|adds|leans|leaned|nods|nodded"
    r"|smiles|smiled|gestures|gestured|pauses|paused|sighs|sighed)\b"
)


class FilterViolation(BaseModel):
    rule: str  # stage direction, name label, narration, open quotation or modern term
    criterion: str  # the OVS criterion it breaks
    evidence: str  # the text that broke it

    def describe(self) -> str:
        return f'{self.rule} "{self.evidence}"'


class FilterSettings(BaseModel):
    enabled: bool = True
    score_cap: float  # a flagged criterion scores at most this
    modern_terms: list[str]


class SafetyFilter:
    def __init__(self, settings: FilterSettings) -> None:
        self.settings = settings
        terms = "|".join(re.escape(term) for term in sorted(settings.modern_terms, key=len, reverse=True))
        self._modern = re.compile(rf"\b(?:{terms})s?\b", re.IGNORECASE) if terms else None

    @classmethod
    def from_config(cls, path: str | Path = "configs/rubrics/ovs.yaml") -> SafetyFilter | None:
        """None when the config has no safety_filter section or turns it off."""
        section = load_yaml(path).get("safety_filter")
        if not section:
            return None
        settings = FilterSettings.model_validate(section)
        return cls(settings) if settings.enabled else None

    def check(self, reply: str) -> list[FilterViolation]:
        violations = []
        for match in _BRACKETED.finditer(reply):
            inside = next(group for group in match.groups() if group is not None)
            if not _SPEAKER_WORDS.search(inside):
                violations.append(FilterViolation(rule="stage direction", criterion="SS", evidence=match.group()))
        violations += [FilterViolation(rule="name label", criterion="SS", evidence=m.group().strip())
                       for m in _NAME_LABEL.finditer(reply)]
        outside_brackets = _BRACKETED.sub(" ", reply)  # a stage direction is not counted again as narration
        violations += [FilterViolation(rule="narration", criterion="SS", evidence=m.group())
                       for m in _NARRATION.finditer(outside_brackets)]
        if open_quote := _open_quotation(reply):
            violations.append(FilterViolation(rule="open quotation", criterion="SS", evidence=open_quote))
        if self._modern:
            found = dict.fromkeys(m.group().lower() for m in self._modern.finditer(reply))
            violations += [FilterViolation(rule="modern term", criterion="HWB", evidence=term) for term in found]
        return violations

    def apply(self, ovs: RubricScore, violations: list[FilterViolation], weights: Mapping[str, float]) -> RubricScore:
        """The judge's OVS with each flagged criterion capped and the reason added to its rationale."""
        if not violations:
            return ovs
        criteria = dict(ovs.criteria)
        for code in dict.fromkeys(v.criterion for v in violations):
            judged = criteria[code]
            found = "; ".join(v.describe() for v in violations if v.criterion == code)
            criteria[code] = CriterionScore(
                code=code,
                score=min(judged.score, self.settings.score_cap),
                rationale=f"Filter: {found}. Judge ({judged.score:g}): {judged.rationale}",
            )
        total = weighted_score({code: c.score for code, c in criteria.items()}, weights)
        return ovs.model_copy(update={"criteria": criteria, "weighted_total": total})


def _open_quotation(reply: str) -> str:
    """The quotation mark left open, if any (apostrophes are not counted)."""
    if reply.count('"') % 2:
        return '"'
    if reply.count("\u201c") != reply.count("\u201d"):
        return "\u201c" if reply.count("\u201c") > reply.count("\u201d") else "\u201d"
    return ""