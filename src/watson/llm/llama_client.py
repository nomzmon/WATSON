"""Judge model client: scores inputs and outputs against the WICS/DFMS/OVS/PCS rubrics."""

from __future__ import annotations

from typing import Any

from watson.common.config import Settings, get_settings
from watson.llm.base import T
from watson.llm.json_parsing import parse_json_object
from watson.llm.ollama import OllamaClient


class LlamaClient(OllamaClient):
    def __init__(self, model_name: str, temperature: float = 0.0, **kwargs: Any) -> None:
        super().__init__(model_name, temperature=temperature, **kwargs)

    @classmethod
    def from_settings(cls, settings: Settings | None = None, **kwargs: Any) -> LlamaClient:
        settings = settings or get_settings()
        if not settings.judge_model:
            raise ValueError("JUDGE_MODEL is not set in .env (e.g. JUDGE_MODEL=llama3:8b-instruct-q4_0)")
        return cls(settings.judge_model, base_url=settings.model_server_url, **kwargs)

    def generate_structured(self, prompt: str, schema: type[T], **kwargs: Any) -> T:
        """Constrain output to `schema` via Ollama's `format`, then repair and validate it."""
        response = self.generate(prompt, format=schema.model_json_schema(), **kwargs)
        return schema.model_validate(parse_json_object(response.text))