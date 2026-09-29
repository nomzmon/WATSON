"""YAML config loading and .env-backed settings.

Model ids, rubric weights, and system toggles live in configs/*.yaml
(see configs/CONTENTS.txt) and are loaded via load_yaml/load_configs.
Secrets never go in YAML; they come from .env via get_settings().
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseModel):
    openai_api_key: str | None = None
    generator_model: str | None = None
    judge_model: str | None = None
    model_server_url: str = "http://localhost:11434"
    log_level: str = "INFO"

    @classmethod
    def load(cls, env_file: Path | None = None) -> Settings:
        load_dotenv(env_file or PROJECT_ROOT / ".env", override=False)
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            generator_model=os.getenv("GENERATOR_MODEL") or None,
            judge_model=os.getenv("JUDGE_MODEL") or None,
            model_server_url=os.getenv("MODEL_SERVER_URL", "http://localhost:11434"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )


@lru_cache
def get_settings() -> Settings:
    return Settings.load()


def resolve_path(path: str | Path) -> Path:
    """Relative paths are resolved from the repository root, not the current directory."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_yaml(path: str | Path) -> dict[str, Any]:
    with resolve_path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_text(path: str | Path) -> str:
    return resolve_path(path).read_text(encoding="utf-8")


def load_configs(*paths: str | Path) -> dict[str, Any]:
    """Load and shallow-merge multiple YAML files; later paths take precedence."""
    merged: dict[str, Any] = {}
    for path in paths:
        merged.update(load_yaml(path))
    return merged