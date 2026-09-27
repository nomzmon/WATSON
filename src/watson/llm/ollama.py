"""Shared client for models served by a local Ollama server.

Both the generator (Gemma) and the judge (Llama) run on Ollama, so the HTTP
call lives here and gemma_client.py / llama_client.py only set defaults.
Uses Ollama's /api/generate endpoint with streaming off.
"""

from __future__ import annotations

from typing import Any

import httpx

from watson.llm.base import LLMClient


class LLMError(RuntimeError):
    pass


class OllamaClient(LLMClient):
    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int | None = None,
        context_window: int | None = None,
        timeout: float = 120.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url
        self.default_options: dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            self.default_options["num_predict"] = max_tokens
        if context_window is not None:
            self.default_options["num_ctx"] = context_window
        self._http = http_client or httpx.Client(base_url=base_url, timeout=timeout)

    def _generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        format: str | dict[str, Any] | None = None,
        **options: Any,
    ) -> str:
        """Extra keyword arguments are Ollama options (e.g. temperature, seed, top_p)."""
        payload: dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {**self.default_options, **options},
        }
        if system is not None:
            payload["system"] = system
        if format is not None:
            payload["format"] = format

        try:
            response = self._http.post("/api/generate", json=payload)
        except httpx.ConnectError as exc:
            raise LLMError(
                f"could not reach Ollama at {self.base_url}; is `ollama serve` running?"
            ) from exc
        except httpx.TimeoutException as exc:
            raise LLMError(
                f"Ollama timed out on model {self.model_name!r}; the first call after "
                "startup can be slow while the model loads, so consider a larger timeout"
            ) from exc
        if response.is_error:
            raise LLMError(
                f"Ollama returned {response.status_code} for model {self.model_name!r}: {response.text}"
            )
        return response.json()["response"]