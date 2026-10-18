"""Response Generation Layer (thesis Section 5.1.5).

Sends the rendered structured prompt to Gemma and returns the candidate
response, which the Output Validation Engine checks before the user sees it.
Formatting Gemma adds around Holmes' words (a name label, stage directions,
quotation marks around the speech) is removed here, so it does not cost a
regeneration later.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import Field

from watson.common.config import Settings, load_yaml
from watson.llm.base import LLMClient, LLMResponse
from watson.llm.gemma_client import GemmaClient

_NAME_LABEL = re.compile(r"^\s*(?:Sherlock\s+)?Holmes\s*:\s*", re.IGNORECASE)
# Text in brackets, or two or more words between asterisks.
_BRACKETED = re.compile(r"\(([^()]*)\)|\[([^\[\]]*)\]|\*([^*\n]+\s[^*\n]+)\*")
# Bracketed text with one of these is Holmes speaking (an aside); without, it describes him (a stage direction).
_SPEAKER_WORDS = re.compile(r"\b(?:I|me|my|myself|we|us|our|you|your)\b", re.IGNORECASE)
_QUOTED = re.compile(r'"([^"]*)"|\u201c([^\u201c\u201d]*)\u201d')


class GeneratedReply(LLMResponse):
    removed: list[str] = Field(default_factory=list)  # stage directions cut from the reply


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

    def generate(self, prompt: str, **options: Any) -> GeneratedReply:
        """Extra keyword arguments are passed to the model, e.g. seed=42 for a reproducible reply."""
        if self.stop:
            options = {"stop": self.stop, **options}
        response = self.client.generate(prompt, **options)
        text, removed = remove_stage_directions(response.text)
        return GeneratedReply(**response.model_dump(), removed=removed).model_copy(
            update={"text": clean_response(text)}
        )


def stage_directions(text: str) -> list[str]:
    """Bracketed or asterisked passages that describe Holmes instead of being his words."""
    return [
        match.group()
        for match in _BRACKETED.finditer(text)
        if not _SPEAKER_WORDS.search(next(group for group in match.groups() if group is not None))
    ]


def remove_stage_directions(text: str) -> tuple[str, list[str]]:
    """The text without its stage directions, and the directions removed.

    Nothing is removed if that would leave no words, so the Output Validation
    Engine still sees the reply and can ask for a new one.
    """
    removed = stage_directions(text)
    if not removed:
        return text, []
    remaining = text
    for direction in removed:
        remaining = remaining.replace(direction, " ", 1)
    lines = [re.sub(r"[ \t]{2,}", " ", line).strip() for line in remaining.splitlines()]
    remaining = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    return (remaining, removed) if remaining else (text, [])


def clean_response(text: str) -> str:
    """Drop a leading 'Holmes:' label, and quotation marks when the reply is only quoted speech.

    Quotes are only removed when nothing but whitespace lies outside them; a
    reply that quotes a phrase or narrates around its quotes is left as is. A
    single opening quotation mark that is never closed is dropped too.
    """
    text = _NAME_LABEL.sub("", text.strip(), count=1).strip()
    if _QUOTED.search(text) and not _QUOTED.sub("", text).strip():
        text = _QUOTED.sub(lambda m: next(g for g in m.groups() if g is not None).strip(), text).strip()
    quotes = text.count('"') + text.count("\u201c") + text.count("\u201d")
    if quotes == 1 and text[:1] in ('"', "\u201c"):
        text = text[1:].lstrip()
    return text