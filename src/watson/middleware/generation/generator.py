"""Response Generation Layer (thesis Section 5.1.5).

Sends the rendered structured prompt to Gemma and returns the candidate
response, which the Output Validation Engine checks before the user sees it.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from watson.common.config import Settings, load_yaml
from watson.llm.base import LLMClient, LLMResponse
from watson.llm.gemma_client import GemmaClient

_NAME_LABEL = re.compile(r"^\s*(?:Sherlock\s+)?Holmes\s*:\s*", re.IGNORECASE)
_QUOTE_PAIRS = {'"': '"', "\u201c": "\u201d"}


class ResponseGenerator:
    def __init__(self, client: LLMClient, stop: Sequence[str] = ()) -> None:
        self.client = client
        self.stop = list(stop)

    @classmethod
    def from_config(
        cls,
        settings: Settings | None = None,
        config_path: str | Path = "configs/models/generator.yaml",
    ) -> ResponseGenerator:
        config = load_yaml(config_path)
        client = GemmaClient.from_settings(
            settings,
            temperature=config["temperature"],
            max_tokens=config.get("max_tokens"),
            context_window=config.get("context_window"),
            timeout=config.get("timeout", 120.0),
        )
        return cls(client, stop=config.get("stop", ()))

    def generate(self, prompt: str, **options: Any) -> LLMResponse:
        """Extra keyword arguments are passed to the model, e.g. seed=42 for a reproducible reply."""
        if self.stop:
            options = {"stop": self.stop, **options}
        response = self.client.generate(prompt, **options)
        return response.model_copy(update={"text": clean_response(response.text)})


def clean_response(text: str) -> str:
    """Drop a leading 'Holmes:' label and quotation marks wrapping the whole reply.

    Quotes are only removed when they enclose everything; a reply quoting several
    separate phrases is left as is, since stripping its ends would break it.
    """
    text = _NAME_LABEL.sub("", text.strip(), count=1).strip()
    opening, closing, inner = text[:1], text[-1:], text[1:-1]
    if len(text) >= 2 and _QUOTE_PAIRS.get(opening) == closing and opening not in inner and closing not in inner:
        text = inner.strip()
    return text