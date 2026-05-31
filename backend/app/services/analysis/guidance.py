from __future__ import annotations

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def report_guidance() -> str:
    path = Path(__file__).with_name("report_guidance.md")
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""

