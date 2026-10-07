"""Fixed replies for the Redirect and Reject decisions (thesis Section 5.2.1.1)."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from watson.common.config import load_yaml


class DecisionMessages(BaseModel):
    redirect: str
    reject: str

    @classmethod
    def from_config(cls, path: str | Path = "prompts/decision/fixed_messages.yaml") -> DecisionMessages:
        return cls.model_validate(load_yaml(path))