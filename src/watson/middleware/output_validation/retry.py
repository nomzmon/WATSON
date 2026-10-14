"""Validation loop with retry counter and fallback reply (thesis Sections 5.1.6 and 5.2.3).

Each candidate reply is validated. A reply that fails is regenerated with
strengthened constraints, up to the retry limit; if every attempt fails, the
predefined fallback reply is returned so the visitor always gets an answer.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from pydantic import BaseModel

from watson.common.config import load_yaml
from watson.common.schemas import DialogueStage, RubricScore
from watson.llm.base import LLMClient
from watson.middleware.dialogue_flow.constraint_selector import ConstraintLibrary, SelectedConstraint
from watson.middleware.output_validation.constraint_refiner import refine_constraints, refinement_level, weak_criteria
from watson.middleware.output_validation.validator import (
    OutputValidator,
    ValidationSettings,
    load_validation_settings,
    passes,
)
from watson.middleware.prompt_builder.builder import StructuredPrompt

# Generates a reply for a prompt; the second argument is the attempt number (0 = first reply).
GenerateReply = Callable[[StructuredPrompt, int], str]


class ValidationAttempt(BaseModel):
    reply: str
    ovs: RubricScore
    passed: bool
    constraints: list[SelectedConstraint]  # the constraints this reply was generated with


class ValidationResult(BaseModel):
    reply: str
    used_fallback: bool
    attempts: list[ValidationAttempt]


class OutputValidationEngine:
    def __init__(
        self,
        validator: OutputValidator,
        library: ConstraintLibrary,
        settings: ValidationSettings,
        fallback_message: str,
    ) -> None:
        self.validator = validator
        self.library = library
        self.settings = settings
        self.fallback_message = fallback_message

    @classmethod
    def from_config(
        cls,
        judge: LLMClient,
        profile_path: str | Path,
        rubric_path: str | Path = "configs/rubrics/ovs.yaml",
        prompt_path: str | Path = "prompts/judge/ovs.txt",
        library_dir: str | Path = "src/watson/middleware/dialogue_flow/constraint_library",
    ) -> OutputValidationEngine:
        return cls(
            OutputValidator.from_config(judge, profile_path, rubric_path, prompt_path),
            ConstraintLibrary.from_dir(library_dir),
            load_validation_settings(rubric_path),
            load_yaml(rubric_path)["fallback_message"],
        )

    def run(
        self,
        prompt: StructuredPrompt,
        constraints: Sequence[SelectedConstraint],
        generate: GenerateReply,
    ) -> ValidationResult:
        """prompt is the Prompt Builder's prompt without constraints; constraints are the Dialogue Flow Manager's."""
        stage = prompt.dialogue_stage or DialogueStage.INTRODUCTION
        current = list(constraints)
        attempts = []
        for attempt in range(self.settings.max_retries + 1):
            guided = prompt.model_copy(update={"constraints": [*prompt.constraints, *(c.text for c in current)]})
            reply = generate(guided, attempt)
            ovs = self.validator.evaluate(guided, reply)
            passed = passes(ovs, self.settings)
            attempts.append(ValidationAttempt(reply=reply, ovs=ovs, passed=passed, constraints=current))
            if passed:
                return ValidationResult(reply=reply, used_fallback=False, attempts=attempts)
            if attempt < self.settings.max_retries:
                weak = weak_criteria(ovs, self.settings.weak_criterion_below)
                current = refine_constraints(current, weak, refinement_level(attempt + 1), stage, self.library)
        return ValidationResult(reply=self.fallback_message, used_fallback=True, attempts=attempts)