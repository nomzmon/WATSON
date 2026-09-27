"""Generator model client: produces Holmes's in-character responses."""

from __future__ import annotations

from typing import Any

from watson.common.config import Settings, get_settings
from watson.llm.ollama import OllamaClient


class GemmaClient(OllamaClient):
    def __init__(self, model_name: str, temperature: float = 0.7, **kwargs: Any) -> None:
        super().__init__(model_name, temperature=temperature, **kwargs)

    @classmethod
    def from_settings(cls, settings: Settings | None = None, **kwargs: Any) -> GemmaClient:
        settings = settings or get_settings()
        if not settings.generator_model:
            raise ValueError("GENERATOR_MODEL is not set in .env (e.g. GENERATOR_MODEL=gemma:7b-instruct-q4_0)")
        return cls(settings.generator_model, base_url=settings.model_server_url, **kwargs)