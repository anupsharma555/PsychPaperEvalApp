from __future__ import annotations

from pathlib import Path

from app.services import jobs
from app.services import pipeline


def test_job_runner_init_does_not_recover_jobs(monkeypatch) -> None:
    def fail_recovery() -> int:
        raise AssertionError("job recovery should not run during construction")

    monkeypatch.setattr(jobs, "_recover_stale_running_jobs", fail_recovery)

    runner = jobs.JobRunner()

    assert runner.status()["stale_jobs_recovered"] == 0


def test_pipeline_job_execution_lock_blocks_duplicate_runner(monkeypatch, tmp_path) -> None:
    def fake_artifacts_dir(document_id: int) -> Path:
        return tmp_path / f"doc_{document_id}" / "artifacts"

    monkeypatch.setattr(pipeline, "artifacts_dir", fake_artifacts_dir)

    first = pipeline._acquire_job_execution_lock(17, 23)
    assert first is not None
    try:
        assert pipeline._acquire_job_execution_lock(17, 23) is None
    finally:
        pipeline._release_job_execution_lock(first)

    second = pipeline._acquire_job_execution_lock(17, 23)
    assert second is not None
    pipeline._release_job_execution_lock(second)
