"""Recover a JSON object from judge model output.

Local models often wrap JSON in markdown fences, add prose around it, or
leave trailing commas. parse_json_object tries progressively looser
repairs and raises JSONParseError if none of them yield a JSON object.
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_TRAILING_COMMA = re.compile(r",\s*([}\]])")


class JSONParseError(ValueError):
    pass


def parse_json_object(text: str) -> dict[str, Any]:
    for candidate in _candidates(text):
        for attempt in (candidate, _TRAILING_COMMA.sub(r"\1", candidate)):
            try:
                value = json.loads(attempt)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    raise JSONParseError(f"no JSON object found in model output: {text[:200]!r}")


def _candidates(text: str) -> list[str]:
    candidates = [text.strip()]
    fenced = _FENCE.search(text)
    if fenced:
        candidates.append(fenced.group(1).strip())
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    return candidates