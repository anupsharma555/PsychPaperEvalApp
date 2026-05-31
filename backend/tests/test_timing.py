from __future__ import annotations

import pytest

from app.services.timing import TimelineRecorder, utc_timestamp


def test_utc_timestamp_uses_z_suffix() -> None:
    assert utc_timestamp().endswith("Z")


def test_timeline_recorder_tracks_completed_and_failed_steps() -> None:
    timeline = TimelineRecorder()

    with timeline.step("completed", document_id=3):
        pass

    with pytest.raises(ValueError):
        with timeline.step("failed"):
            raise ValueError("bad input")

    events = timeline.snapshot()
    assert events[0]["step"] == "completed"
    assert events[0]["status"] == "completed"
    assert events[0]["metadata"] == {"document_id": 3}
    assert events[0]["duration_seconds"] >= 0
    assert events[1]["step"] == "failed"
    assert events[1]["status"] == "failed"
    assert events[1]["error_type"] == "ValueError"
