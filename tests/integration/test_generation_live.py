"""Live check of the Prompt Builder and Response Generation Layer against the real Gemma model.

Skipped unless Ollama is reachable at MODEL_SERVER_URL and GENERATOR_MODEL is set in .env.
Uses the unvalidated sample profile in tests/fixtures until the Chapter 4 profile exists.
"""

import httpx
import pytest

from watson.common.config import get_settings
from watson.common.schemas import DialogueStage, HistoryTurn
from watson.middleware.generation.generator import ResponseGenerator
from watson.middleware.prompt_builder.builder import PromptBuilder

settings = get_settings()

CASE_CONTEXT = "A pearl necklace was stolen from a locked study; the only footprints lead to the garden window."


def _ollama_reachable() -> bool:
    try:
        return httpx.get(f"{settings.model_server_url}/api/tags", timeout=2).is_success
    except httpx.HTTPError:
        return False


pytestmark = [
    pytest.mark.skipif(not _ollama_reachable(), reason=f"Ollama not reachable at {settings.model_server_url}"),
    pytest.mark.skipif(not settings.generator_model, reason="GENERATOR_MODEL not set in .env"),
]


@pytest.mark.parametrize(
    "template",
    ["prompts/generation/baseline_persona.txt", "prompts/generation/structured_prompt.txt"],
)
def test_holmes_replies_from_structured_prompt(template):
    builder = PromptBuilder.from_config(profile_path="tests/fixtures/holmes_profile_sample.json", template_path=template)
    generator = ResponseGenerator.from_config(settings)
    history = [
        HistoryTurn(role="user", text="Good evening, Mr. Holmes. A necklace has been stolen from my study."),
        HistoryTurn(role="holmes", text="Pray sit down and tell me everything, omitting no detail."),
    ]

    prompt = builder.build(
        "What do you make of the footprints beneath the window?",
        history=history,
        dialogue_stage=DialogueStage.EVIDENCE,
        case_context=CASE_CONTEXT,
    )
    response = generator.generate(builder.render(prompt))

    print(f"\n[{template.split('/')[-1]}, {response.latency_ms:.0f} ms]\n{response.text}")
    assert response.text
    assert not response.text.lower().startswith("holmes:")