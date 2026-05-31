from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from time import monotonic
from typing import Any, Iterator


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class TimelineRecorder:
    def __init__(self) -> None:
        self._origin = monotonic()
        self.events: list[dict[str, Any]] = []

    @contextmanager
    def step(self, name: str, **metadata: Any) -> Iterator[None]:
        started = monotonic()
        event: dict[str, Any] = {
            "step": str(name),
            "started_at": utc_timestamp(),
            "offset_seconds": round(started - self._origin, 4),
        }
        clean_metadata = {key: value for key, value in metadata.items() if value is not None}
        if clean_metadata:
            event["metadata"] = clean_metadata
        try:
            yield
        except Exception as exc:
            event["status"] = "failed"
            event["error_type"] = type(exc).__name__
            event["error"] = str(exc)[:500]
            raise
        else:
            event["status"] = "completed"
        finally:
            event["ended_at"] = utc_timestamp()
            event["duration_seconds"] = round(monotonic() - started, 4)
            self.events.append(event)

    def snapshot(self) -> list[dict[str, Any]]:
        return [dict(event) for event in self.events]
