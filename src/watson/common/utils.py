from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def new_id(prefix: str = "") -> str:
    token = uuid.uuid4().hex[:12]
    return f"{prefix}_{token}" if prefix else token


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@contextmanager
def timer() -> Iterator[Callable[[], float]]:
    """Yields a callable returning elapsed milliseconds since the `with` block started."""
    start = time.perf_counter()
    yield lambda: (time.perf_counter() - start) * 1000


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path