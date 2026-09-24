"""Shared Pydantic models used across persona construction, middleware, and logging.

Rubric criterion codes (OOP/TR/HC/GIC for WICS, CST/NP/CC/TBC for DFMS,
PCC/NC/HWB/SS for OVS) are kept as opaque strings rather than hardcoded
fields: weights and criterion sets live in configs/rubrics/*.yaml (see
configs/rubrics/CONTENTS.txt) and their full names/definitions are in the
thesis (Section 5.2). Confirm against the thesis text before renaming.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from watson.common.utils import new_id, utc_now


# --- Persona Knowledge Base (Chapter 4) -------------------------------------


class PersonaLayer(BaseModel):
    """One of the five persona layers: a set of trait statements."""

    traits: list[str] = Field(default_factory=list)


class PersonaProfile(BaseModel):
    id: str = Field(default_factory=lambda: new_id("persona"))
    version: str = "v001"
    personality: PersonaLayer
    expression: PersonaLayer
    knowledge: PersonaLayer
    moral: PersonaLayer
    belief: PersonaLayer
    pcs_score: float | None = None
    status: Literal["draft", "validated"] = "draft"
    created_at: datetime = Field(default_factory=utc_now)


# --- Rubric scoring (WICS / DFMS / OVS / PCS) --------------------------------


class RubricName(StrEnum):
    WICS = "wics"
    DFMS = "dfms"
    OVS = "ovs"
    PCS = "pcs"


class CriterionScore(BaseModel):
    code: str
    score: float
    rationale: str | None = None


class RubricScore(BaseModel):
    rubric: RubricName
    criteria: dict[str, CriterionScore] = Field(default_factory=dict)
    weighted_total: float
    passed: bool | None = None


# --- Decision Layer (5.1.2, Table 5.2) ---------------------------------------


class DecisionType(StrEnum):
    ACCEPT = "accept"
    REPHRASE = "rephrase"
    REDIRECT = "redirect"
    REJECT = "reject"


class DecisionResult(BaseModel):
    decision: DecisionType
    reasons: list[str] = Field(default_factory=list)
    rephrased_input: str | None = None
    redirect_message: str | None = None


# --- Dialogue Flow Manager (5.1.4) -------------------------------------------


class DialogueStage(StrEnum):
    INTRODUCTION = "introduction"
    EVIDENCE = "evidence"
    SUSPECTS = "suspects"
    DEDUCTION = "deduction"
    RESOLUTION = "resolution"


# --- Dialogue Constraint Library ---------------------------------------------


class ConstraintCategory(StrEnum):
    DIALOGUE_STATE = "dialogue_state"
    NARRATIVE_PROGRESSION = "narrative_progression"
    TOPIC_BOUNDARY = "topic_boundary"
    CONTEXT_CONTINUITY = "context_continuity"
    PERSONA = "persona"
    HISTORICAL = "historical"


class ConstraintStrengthText(BaseModel):
    level: int = Field(ge=1)
    text: str


class Constraint(BaseModel):
    id: str
    category: ConstraintCategory
    strengths: list[ConstraintStrengthText]

    def text_for(self, level: int) -> str:
        for strength in self.strengths:
            if strength.level == level:
                return strength.text
        raise ValueError(f"constraint {self.id} has no strength level {level}")

    @property
    def max_level(self) -> int:
        return max(s.level for s in self.strengths)


# --- Conversation history -----------------------------------------------------


class HistoryTurn(BaseModel):
    role: Literal["user", "holmes"]
    text: str
    timestamp: datetime = Field(default_factory=utc_now)


# --- System configuration (four evaluated configurations) --------------------


class ModuleToggles(BaseModel):
    input_constraint: bool = False
    dialogue_flow: bool = False
    output_validation: bool = False
    decision_layer: bool = False


class SystemConfiguration(BaseModel):
    id: str
    name: str
    modules: ModuleToggles
    model_configs: dict[str, str] = Field(default_factory=dict)
    rubric_configs: dict[str, str] = Field(default_factory=dict)


# --- Interaction logging (per-turn JSONL records) -----------------------------


class ComponentLatency(BaseModel):
    component: str
    ms: float


class TurnLog(BaseModel):
    turn_id: str = Field(default_factory=lambda: new_id("turn"))
    session_id: str
    participant_id: str | None = None
    config_id: str
    timestamp: datetime = Field(default_factory=utc_now)
    user_input: str
    dialogue_stage: DialogueStage | None = None
    wics: RubricScore | None = None
    input_decision: DecisionResult | None = None
    dfms: RubricScore | None = None
    selected_constraint_ids: list[str] = Field(default_factory=list)
    ovs: RubricScore | None = None
    retries: int = 0
    final_response: str
    latencies: list[ComponentLatency] = Field(default_factory=list)

    @property
    def total_latency_ms(self) -> float:
        return sum(latency.ms for latency in self.latencies)