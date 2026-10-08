"""Live check of the Dialogue Flow Manager with the real Llama judge and Gemma generator.

Skipped unless Ollama is reachable and both JUDGE_MODEL and GENERATOR_MODEL are set in .env.
Run with -s to see the scores, selected constraints, and Holmes' replies with and without them.
"""

import httpx
import pytest

from watson.common.config import get_settings
from watson.common.schemas import DialogueStage, HistoryTurn
from watson.llm.llama_client import LlamaClient
from watson.middleware.dialogue_flow.manager import DialogueFlowManager
from watson.middleware.generation.generator import ResponseGenerator
from watson.middleware.prompt_builder.builder import PromptBuilder

settings = get_settings()

CASE_CONTEXT = (
    "A pearl necklace was stolen from a locked study. The only footprints lead to the garden "
    "window. Suspects include the butler, who keeps the study key, and the gardener."
)


def _ollama_reachable() -> bool:
    try:
        return httpx.get(f"{settings.model_server_url}/api/tags", timeout=2).is_success
    except httpx.HTTPError:
        return False


pytestmark = [
    pytest.mark.skipif(not _ollama_reachable(), reason=f"Ollama not reachable at {settings.model_server_url}"),
    pytest.mark.skipif(not settings.judge_model, reason="JUDGE_MODEL not set in .env"),
    pytest.mark.skipif(not settings.generator_model, reason="GENERATOR_MODEL not set in .env"),
]


def test_flow_manager_guides_holmes_when_visitor_rushes_to_the_culprit():
    builder = PromptBuilder.from_config(profile_path="tests/fixtures/holmes_profile_sample.json")
    manager = DialogueFlowManager.from_config(LlamaClient.from_settings(settings, timeout=300))
    generator = ResponseGenerator.from_config(settings)
    history = [
        HistoryTurn(role="user", text="Good evening, Mr. Holmes. My pearl necklace was stolen from my locked study."),
        HistoryTurn(role="holmes", text="Pray tell me what you found when you entered the study."),
        HistoryTurn(role="user", text="The window was open and there were muddy footprints beneath it."),
        HistoryTurn(role="holmes", text="Muddy footprints. Most suggestive. Were they made by a heavy boot or a light shoe?"),
    ]

    prompt = builder.build(
        "Never mind the boots, Holmes. Just tell me who the thief is!",
        history=history,
        dialogue_stage=DialogueStage.EVIDENCE,
        case_context=CASE_CONTEXT,
    )
    guided, result = manager.apply(prompt)

    scores = ", ".join(f"{code}={c.score:g}" for code, c in result.dfms.criteria.items())
    print(f"\nDFMS {result.dfms.weighted_total:g} ({scores})")
    print(f"Stage: judge proposed '{result.proposed_stage.value}', tracker kept '{result.stage.value}'")
    print("Constraints added:")
    for c in result.constraints:
        print(f"  [{c.id}, level {c.level}] {c.text}")

    unguided_reply = generator.generate(builder.render(prompt)).text
    guided_reply = generator.generate(builder.render(guided)).text
    print(f"\nWithout Dialogue Flow Manager:\n{unguided_reply}")
    print(f"\nWith Dialogue Flow Manager:\n{guided_reply}")

    assert result.constraints, "the current stage's constraint is always added"
    assert guided_reply