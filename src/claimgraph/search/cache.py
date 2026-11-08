"""Disk cache for online search responses, in data/cache."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from ..config import get_settings


def _path(namespace: str, key: str):
    folder = get_settings().cache_dir / namespace
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{hashlib.sha1(key.encode()).hexdigest()}.json"


def cache_get(namespace: str, key: str) -> Any | None:
    path = _path(namespace, key)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def cache_put(namespace: str, key: str, value: Any) -> None:
    _path(namespace, key).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
