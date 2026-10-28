"""Output Validation Evaluation (thesis Sections 5.1.6 and 5.2.3).

Before a candidate reply reaches the visitor, the judge scores it on Persona
Consistency, Narrative Coherence, Historical and World Boundary, and Style
and Syntax. The weighted total is the Output Validation Score (OVS); a reply
passes when the OVS reaches the threshold and no criterion is too low.

Before scoring, the judge lists the details the reply states as fact that
appear in neither the case nor the conversation. If it lists any, Narrative
Coherence is capped, so a reply with invented evidence fails and is retried.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from watson.common.config import load_text, load_yaml
from watson.common.rubric import CriterionJudgement, load_rubric_weights, weighted_score
from watson.common.schemas import CriterionScore, PersonaProfile, RubricName, RubricScore
from watson.llm.base import LLMClient
from watson.middleware.prompt_builder.builder import StructuredPrompt, render_history, render_persona

OVS_CRITERIA = ("PCC", "NC", "HWB", "SS")


class OVSJudgement(BaseModel):
    # Required in the schema sent to the judge, so it cannot skip the list; parsing still accepts its absence.
    model_config = ConfigDict(json_schema_extra={"required": ["unsupported_details", *OVS_CRITERIA]})

    # First, so the judge lists unsupported details before it scores.
    unsupported_details: list[str] = Field(default_factory=list)
    PCC: CriterionJudgement
    NC: CriterionJudgement
    HWB: CriterionJudgement
    SS: CriterionJudgement


class ValidationSettings(BaseModel):
    pass_threshold: float
    min_criterion: float
    max_retries: int
    weak_criterion_below: float
    unsupported_detail_cap: float | None = None  # Narrative Coherence scores at most this if details are listed


def load_ovs_weights(path: str | Path = "configs/rubrics/ovs.yaml") -> dict[str, float]:
    return load_rubric_weights(path, OVS_CRITERIA, "OVS")


def load_validation_settings(path: str | Path = "configs/rubrics/ovs.yaml") -> ValidationSettings:
    return ValidationSettings.model_validate(load_yaml(path)["validation"])


def passes(ovs: RubricScore, settings: ValidationSettings) -> bool:
    lowest = min(c.score for c in ovs.criteria.values())
    return ovs.weighted_total >= settings.pass_threshold and lowest >= settings.min_criterion


class OutputValidator:
    def __init__(
        self,
        judge: LLMClient,
        profile: PersonaProfile,
        weights: Mapping[str, float],
        prompt_template: str,
        history_turns: int = 6,
        unsupported_detail_cap: float | None = None,
    ) -> None:
        self.judge = judge
        self.weights = dict(weights)
        self.prompt_template = prompt_template
        self.history_turns = history_turns
        self.unsupported_detail_cap = unsupported_detail_cap
        self._persona_text = render_persona(profile)

    @classmethod
    def from_config(
        cls,
        judge: LLMClient,
        profile_path: str | Path,
        rubric_path: str | Path = "configs/rubrics/ovs.yaml",
        prompt_path: str | Path = "prompts/judge/ovs.txt",
    ) -> OutputValidator:
        profile = PersonaProfile.model_validate_json(load_text(profile_path))
        cap = load_validation_settings(rubric_path).unsupported_detail_cap
        return cls(judge, profile, load_ovs_weights(rubric_path), load_text(prompt_path), unsupported_detail_cap=cap)

    def build_prompt(self, prompt: StructuredPrompt, reply: str) -> str:
        recent = prompt.history[-self.history_turns :] if self.history_turns else []
        return self.prompt_template.format(
            persona_profile=self._persona_text,
            case_context=prompt.case_context or "No case has been presented yet.",
            dialogue_stage=prompt.dialogue_stage.value if prompt.dialogue_stage else "not yet tracked",
            history=render_history(recent),
            user_message=prompt.user_message,
            reply=reply,
        )

    def evaluate(self, prompt: StructuredPrompt, reply: str) -> RubricScore:
        judgement = self.judge.generate_structured(self.build_prompt(prompt, reply), OVSJudgement)
        criteria = {
            code: CriterionScore(code=code, score=getattr(judgement, code).score,
                                 rationale=getattr(judgement, code).reasoning)
            for code in OVS_CRITERIA
        }
        details = [d for d in judgement.unsupported_details if d.strip()]
        if details and self.unsupported_detail_cap is not None:
            nc = criteria["NC"]
            criteria["NC"] = nc.model_copy(update={
                "score": min(nc.score, self.unsupported_detail_cap),
                "rationale": f"{nc.rationale} Unsupported details: {'; '.join(details)}",
            })
        total = weighted_score({code: c.score for code, c in criteria.items()}, self.weights)
        return RubricScore(rubric=RubricName.OVS, criteria=criteria, weighted_total=total)