"""Response Generation Layer (thesis Section 5.1.5).

Sends the rendered structured prompt to Gemma and returns the candidate
response, which the Output Validation Engine checks before the user sees it.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from watson.common.config import Settings, load_yaml
from watson.llm.base import LLMClient, LLMResponse
from watson.llm.gemma_client import GemmaClient

_NAME_LABEL = re.compile(r"^\s*(?:Sherlock\s+)?Holmes\s*:\s*", re.IGNORECASE)


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

    def generate(self, prompt: str) -> LLMResponse:
        options = {"stop": self.stop} if self.stop else {}
        response = self.client.generate(prompt, **options)
        return response.model_copy(update={"text": clean_response(response.text)})


def clean_response(text: str) -> str:
    """Drop a leading 'Holmes:' label, which models often copy from the history format."""
    return _NAME_LABEL.sub("", text.strip(), count=1).strip()