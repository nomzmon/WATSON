import json

import httpx
import pytest
from pydantic import BaseModel

from watson.common.config import Settings
from watson.llm.gemma_client import GemmaClient
from watson.llm.llama_client import LlamaClient
from watson.llm.ollama import LLMError


def fake_ollama(reply: str, captured: list[dict]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append({"path": request.url.path, "body": json.loads(request.content)})
        return httpx.Response(200, json={"model": "test", "response": reply, "done": True})

    return httpx.MockTransport(handler)


def make_client(cls, transport: httpx.MockTransport, **kwargs):
    http = httpx.Client(transport=transport, base_url="http://ollama.test")
    return cls("test-model", http_client=http, **kwargs)


def test_generate_sends_prompt_to_ollama_and_returns_text():
    captured: list[dict] = []
    client = make_client(GemmaClient, fake_ollama("Elementary.", captured))

    response = client.generate("Who did it?", system="You are Sherlock Holmes.")

    assert response.text == "Elementary."
    assert response.model == "test-model"
    request = captured[0]
    assert request["path"] == "/api/generate"
    assert request["body"]["model"] == "test-model"
    assert request["body"]["prompt"] == "Who did it?"
    assert request["body"]["system"] == "You are Sherlock Holmes."
    assert request["body"]["stream"] is False


def test_generator_and_judge_use_different_default_temperatures():
    captured: list[dict] = []
    make_client(GemmaClient, fake_ollama("", captured)).generate("x")
    make_client(LlamaClient, fake_ollama("{}", captured)).generate("x")

    assert captured[0]["body"]["options"]["temperature"] == 0.7
    assert captured[1]["body"]["options"]["temperature"] == 0.0


def test_per_call_options_override_defaults():
    captured: list[dict] = []
    client = make_client(GemmaClient, fake_ollama("", captured), max_tokens=256)

    client.generate("x", temperature=0.2, seed=42)

    assert captured[0]["body"]["options"] == {"temperature": 0.2, "num_predict": 256, "seed": 42}


def test_judge_generate_structured_sends_schema_and_parses_messy_json():
    class Verdict(BaseModel):
        score: float
        rationale: str

    captured: list[dict] = []
    reply = '```json\n{"score": 4.5, "rationale": "stays in character",}\n```'
    client = make_client(LlamaClient, fake_ollama(reply, captured))

    verdict = client.generate_structured("Score this reply.", Verdict)

    assert verdict == Verdict(score=4.5, rationale="stays in character")
    assert captured[0]["body"]["format"] == Verdict.model_json_schema()


def test_http_error_raises_llm_error_with_ollama_message():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model 'test-model' not found, try pulling it first"})

    client = make_client(GemmaClient, httpx.MockTransport(handler))

    with pytest.raises(LLMError, match="not found"):
        client.generate("x")


def test_unreachable_server_raises_llm_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = make_client(GemmaClient, httpx.MockTransport(handler))

    with pytest.raises(LLMError, match="ollama serve"):
        client.generate("x")


def test_from_settings_reads_model_and_server_url():
    settings = Settings(
        generator_model="gemma:7b-instruct-q4_0",
        judge_model="llama3:8b-instruct-q4_0",
        model_server_url="http://gpu-box:11434",
    )

    generator = GemmaClient.from_settings(settings)
    judge = LlamaClient.from_settings(settings)

    assert (generator.model_name, generator.base_url) == ("gemma:7b-instruct-q4_0", "http://gpu-box:11434")
    assert (judge.model_name, judge.base_url) == ("llama3:8b-instruct-q4_0", "http://gpu-box:11434")


def test_from_settings_requires_model_name():
    with pytest.raises(ValueError, match="GENERATOR_MODEL"):
        GemmaClient.from_settings(Settings())
    with pytest.raises(ValueError, match="JUDGE_MODEL"):
        LlamaClient.from_settings(Settings())