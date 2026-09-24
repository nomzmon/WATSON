"""Common interface every model client wrapper implements.

Concrete clients (gemma_client.py for the generator, llama_client.py for the
judge, openai_client.py for GPT-5 extraction) subclass LLMClient and
implement `_generate`. This gives every client the same LLMResponse shape and
latency measurement, which TurnLog relies on.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypeVar

from pydantic import BaseModel

from watson.common.utils import timer

T = TypeVar("T", bound=BaseModel)


class LLMResponse(BaseModel):
    text: str
    model: str
    latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    raw: dict[str, Any] | None = None


class LLMClient(ABC):
    model_name: str

    @abstractmethod
    def _generate(self, prompt: str, **kwargs: Any) -> str:
        ...

    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        with timer() as elapsed:
            text = self._generate(prompt, **kwargs)
        return LLMResponse(text=text, model=self.model_name, latency_ms=elapsed())

    def generate_structured(self, prompt: str, schema: type[T], **kwargs: Any) -> T:
        """Generate then validate the response as `schema`.

        Default implementation parses generate()'s text as JSON directly.
        Clients whose output needs repair (e.g. the judge model) or that
        support native structured output should override this.
        """
        response = self.generate(prompt, **kwargs)
        return schema.model_validate_json(response.text)