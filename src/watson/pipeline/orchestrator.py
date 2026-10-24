"""Runs one conversational turn through the enabled middleware components.

Components are added here one at a time as they are built and validated:
the Input Constraint Engine with the Decision Layer, the Prompt Builder, the
Dialogue Flow Manager, the Response Generation Layer and the Output Validation
Engine.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from watson.common.schemas import ComponentLatency, DecisionResult, DecisionType, RubricScore
from watson.common.utils import timer
from watson.middleware.decision.decision import DecisionThresholds, decide
from watson.middleware.decision.messages import DecisionMessages
from watson.middleware.dialogue_flow.manager import DialogueFlowManager, FlowResult
from watson.middleware.generation.generator import ResponseGenerator
from watson.middleware.input_constraint.engine import InputConstraintEngine
from watson.middleware.output_validation.retry import OutputValidationEngine, ValidationResult
from watson.middleware.prompt_builder.builder import PromptBuilder, StructuredPrompt
from watson.pipeline.session import ConversationSession

ReplySource = Literal["generator", "redirect_message", "reject_message", "fallback_message"]


class TurnResult(BaseModel):
    user_message: str
    reply: str
    source: ReplySource
    wics: RubricScore | None = None
    decision: DecisionResult | None = None
    flow: FlowResult | None = None
    validation: ValidationResult | None = None
    removed_from_reply: list[str] = Field(default_factory=list)  # stage directions cut from the reply shown
    latencies: list[ComponentLatency]


class Pipeline:
    def __init__(
        self,
        builder: PromptBuilder,
        generator: ResponseGenerator,
        input_constraint: InputConstraintEngine | None = None,
        decision_thresholds: DecisionThresholds | None = None,
        decision_messages: DecisionMessages | None = None,
        dialogue_flow: DialogueFlowManager | None = None,
        output_validation: OutputValidationEngine | None = None,
    ) -> None:
        if input_constraint and not (decision_thresholds and decision_messages):
            raise ValueError("the Input Constraint Engine needs the Decision Layer's thresholds and messages")
        if (dialogue_flow or output_validation) and "{constraints}" not in builder.template:
            raise ValueError("the Dialogue Flow Manager and Output Validation Engine need a prompt template "
                             "with a {constraints} section")
        self.builder = builder
        self.generator = generator
        self.input_constraint = input_constraint
        self.decision_thresholds = decision_thresholds
        self.decision_messages = decision_messages
        self.dialogue_flow = dialogue_flow
        self.output_validation = output_validation

    def run_turn(self, session: ConversationSession, user_message: str, **generation_options: Any) -> TurnResult:
        """Extra keyword arguments go to the generator, e.g. seed=42."""
        latencies = []
        wics = decision = flow = validation = None

        if self.input_constraint:
            with timer() as elapsed:
                wics = self.input_constraint.evaluate(
                    user_message, session.history, session.dialogue_stage, session.case_context
                )
            latencies.append(ComponentLatency(component="input_constraint", ms=elapsed()))
            decision = decide(wics, self.decision_thresholds)

            if decision.decision == DecisionType.REJECT:
                # A rejected message never enters the role-play, so it is kept out of the history.
                reply = self.decision_messages.reject
                return TurnResult(user_message=user_message, reply=reply, source="reject_message",
                                  wics=wics, decision=decision, latencies=latencies)
            if decision.decision == DecisionType.REDIRECT:
                reply = self.decision_messages.redirect
                session.add_exchange(user_message, reply)
                return TurnResult(user_message=user_message, reply=reply, source="redirect_message",
                                  wics=wics, decision=decision, latencies=latencies)
            # Rephrase passes the original message through until rephrasing is designed.

        prompt = self.builder.build(
            user_message, session.history, session.dialogue_stage, session.case_context, session.visitor
        )
        guided, selected = prompt, []
        if self.dialogue_flow:
            with timer() as elapsed:
                guided, flow = self.dialogue_flow.apply(prompt)
            latencies.append(ComponentLatency(component="dialogue_flow", ms=elapsed()))
            session.dialogue_stage = flow.stage
            selected = flow.constraints

        if not self.output_validation:
            with timer() as elapsed:
                generated = self.generator.generate(self.builder.render(guided), **generation_options)
            latencies.append(ComponentLatency(component="generation", ms=elapsed()))
            session.add_exchange(user_message, generated.text)
            return TurnResult(user_message=user_message, reply=generated.text, source="generator", wics=wics,
                              decision=decision, flow=flow, removed_from_reply=generated.removed, latencies=latencies)

        generation_ms = 0.0
        removed: dict[int, list[str]] = {}  # stage directions cut from each attempt's reply

        def generate(candidate: StructuredPrompt, attempt: int) -> str:
            nonlocal generation_ms
            options = dict(generation_options)
            if "seed" in options:
                options["seed"] += 1000 * attempt  # a retry with the same seed could repeat the rejected reply
            with timer() as elapsed:
                generated = self.generator.generate(self.builder.render(candidate), **options)
            generation_ms += elapsed()
            removed[attempt] = generated.removed
            return generated.text

        with timer() as elapsed:
            validation = self.output_validation.run(guided.model_copy(update={"constraints": prompt.constraints}),
                                                    selected, generate)
        latencies += [ComponentLatency(component="generation", ms=generation_ms),
                      ComponentLatency(component="output_validation", ms=elapsed() - generation_ms)]
        session.add_exchange(user_message, validation.reply)  # the fallback is in Holmes' voice, so it stays
        shown = [] if validation.used_fallback else removed[len(validation.attempts) - 1]
        return TurnResult(user_message=user_message, reply=validation.reply,
                          source="fallback_message" if validation.used_fallback else "generator",
                          wics=wics, decision=decision, flow=flow, validation=validation,
                          removed_from_reply=shown, latencies=latencies)