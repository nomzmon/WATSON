"""Live checks against a real Ollama server.

Skipped unless Ollama is reachable at MODEL_SERVER_URL and the model names are
set in .env (GENERATOR_MODEL / JUDGE_MODEL). The models must already be pulled.
"""

import httpx
import pytest
from pydantic import BaseModel, Field

from watson.common.config import get_settings
from watson.llm.gemma_client import GemmaClient
from watson.llm.llama_client import LlamaClient

settings = get_settings()


def _ollama_reachable() -> bool:
    try:
        return httpx.get(f"{settings.model_server_url}/api/tags", timeout=2).is_success
    except httpx.HTTPError:
        return False


pytestmark = pytest.mark.skipif(
    not _ollama_reachable(), reason=f"Ollama not reachable at {settings.model_server_url}"
)


@pytest.mark.skipif(not settings.generator_model, reason="GENERATOR_MODEL not set in .env")
def test_generator_replies():
    client = GemmaClient.from_settings(settings)

    response = client.generate("Greet Dr. Watson in one short sentence.", num_predict=60)

    print(f"\n[{response.model}, {response.latency_ms:.0f} ms] {response.text}")
    assert response.text.strip()


@pytest.mark.skipif(not settings.judge_model, reason="JUDGE_MODEL not set in .env")
def test_judge_returns_structured_score():
    class FormalityScore(BaseModel):
        score: int = Field(ge=1, le=5)

    client = LlamaClient.from_settings(settings)

    result = client.generate_structured(
        "Rate how formal this sentence is from 1 (casual) to 5 (very formal): "
        "'Good evening, my dear Watson. Pray, take a seat.' "
        "Respond with JSON containing a single integer field named score.",
        FormalityScore,
    )

    print(f"\njudge score: {result.score}")
    assert 1 <= result.score <= 5