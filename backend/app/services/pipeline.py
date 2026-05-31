from __future__ import annotations

from datetime import datetime
import json
import traceback
from pathlib import Path
from time import monotonic

from sqlmodel import Session

from app.core.config import settings
from app.db.models import Job, JobStatus
from app.db.session import engine
from app.services.analysis.runner import run_full_analysis
from app.services.parser import parse_document_assets
from app.services.report_retention import enforce_report_retention
from app.services.storage import artifacts_dir
from app.services.timing import TimelineRecorder, utc_timestamp


def _friendly_job_failure_message(exc: Exception, error_path: Path) -> str:
    raw = str(exc or "").strip()
    lowered = raw.lower()

    if not raw:
        raw = "Analysis failed with an unknown error."

    if "publisher blocked automated access" in lowered or "access-restricted" in lowered:
        detail = (
            "Publisher blocked automated fetch of this paper. "
            "Use From Upload and choose the main PDF."
        )
    elif "could not resolve a pdf" in lowered:
        detail = (
            "Could not find a downloadable PDF from this URL/DOI. "
            "Use From Upload and choose the main PDF."
        )
    elif "rate-limited" in lowered or "http 429" in lowered:
        detail = (
            "Remote site rate-limited this request. "
            "Retry later, or use From Upload with the main PDF."
        )
    elif "openai run budget would be exceeded" in lowered or "openai daily budget would be exceeded" in lowered:
        detail = f"OpenAI usage guardrail stopped this run. {raw}"
    else:
        detail = raw

    return f"{detail} (see {error_path})"


def run_pipeline(job_id: int) -> None:
    timeline = TimelineRecorder()

    def _write_run_timeline(document_id: int) -> None:
        _write_artifact(
            document_id,
            "run_timeline.json",
            {
                "document_id": document_id,
                "timeline": timeline.snapshot(),
                "generated_at": utc_timestamp(),
            },
        )

    with Session(engine) as session:
        job = session.get(Job, job_id)
        if not job:
            return
        try:
            with timeline.step("prepare_analysis", job_id=job.id, document_id=job.document_id):
                job.status = JobStatus.running
                job.progress = 0.05
                job.message = "Preparing analysis"
                job.updated_at = datetime.utcnow()
                session.add(job)
                session.commit()

                job.progress = 0.12
                job.message = "Parsing document assets"
                job.updated_at = datetime.utcnow()
                session.add(job)
                session.commit()

            parse_started_at = monotonic()
            parse_started_ts = utc_timestamp()
            with timeline.step("parse_document_assets", job_id=job.id, document_id=job.document_id):
                parse_counts = parse_document_assets(session, job.document_id)
            parse_seconds = monotonic() - parse_started_at
            parse_ended_ts = utc_timestamp()
            _write_artifact(
                job.document_id,
                "parse_diagnostics.json",
                {
                    "document_id": job.document_id,
                    "parse_seconds": round(parse_seconds, 4),
                    "started_at": parse_started_ts,
                    "ended_at": parse_ended_ts,
                    "counts": parse_counts,
                    "ts": utc_timestamp(),
                },
            )
            _write_run_timeline(job.document_id)

            with timeline.step("initialize_modality_analysis", job_id=job.id, document_id=job.document_id):
                job.progress = 0.48
                job.message = "Parsed assets; initializing modality analysis"
                job.updated_at = datetime.utcnow()
                session.add(job)
                session.commit()

            def _update_progress(progress: float, message: str) -> None:
                job.progress = max(job.progress or 0.0, min(float(progress), 0.99))
                job.message = str(message or job.message or "")
                job.updated_at = datetime.utcnow()
                session.add(job)
                session.commit()

            with timeline.step("run_full_analysis", job_id=job.id, document_id=job.document_id):
                analysis_diag = run_full_analysis(
                    session,
                    job.document_id,
                    progress_callback=_update_progress,
                    job_id=job.id,
                )
            if isinstance(analysis_diag, dict):
                analysis_diag["parse_timing"] = {
                    "parse_total_seconds": round(parse_seconds, 4),
                    "started_at": parse_started_ts,
                    "ended_at": parse_ended_ts,
                }
                analysis_diag["pipeline_timeline"] = timeline.snapshot()
            _log_model_timing(job.id, analysis_diag)
            with timeline.step("write_analysis_diagnostics", job_id=job.id, document_id=job.document_id):
                _write_artifact(
                    job.document_id,
                    "analysis_diagnostics.json",
                    {
                        "document_id": job.document_id,
                        "diagnostics": analysis_diag,
                        "ts": utc_timestamp(),
                    },
                )

            with timeline.step("mark_job_completed", job_id=job.id, document_id=job.document_id):
                job.status = JobStatus.completed
                job.progress = 1.0
                job.message = "Completed"
                job.updated_at = datetime.utcnow()
                session.add(job)
                session.commit()

            if settings.report_retention_enabled:
                with timeline.step("report_retention", job_id=job.id, document_id=job.document_id):
                    retention = enforce_report_retention(
                        session,
                        keep_latest=max(1, int(settings.report_retention_limit)),
                    )
                if retention.get("pruned_count", 0):
                    print(
                        "[pipeline] retention pruned documents:",
                        retention.get("pruned_document_ids", []),
                    )
            _write_run_timeline(job.document_id)
        except Exception as exc:
            error_text = traceback.format_exc()
            error_path = artifacts_dir(job.document_id) / "error.log"
            try:
                error_path.parent.mkdir(parents=True, exist_ok=True)
                error_path.write_text(error_text)
            except Exception:
                pass
            _write_run_timeline(job.document_id)
            job.status = JobStatus.failed
            job.message = _friendly_job_failure_message(exc, error_path)
            job.updated_at = datetime.utcnow()
            session.add(job)
            session.commit()
            print("[pipeline] job failed:", exc)
            print(error_text)


def _write_artifact(document_id: int, filename: str, payload: dict) -> None:
    path = artifacts_dir(document_id) / filename
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))
    except Exception:
        return


def _log_model_timing(job_id: int, analysis_diag: dict) -> None:
    usage = analysis_diag.get("model_usage", {}) if isinstance(analysis_diag, dict) else {}
    if not isinstance(usage, dict):
        return
    text_seconds = float(usage.get("text_total_seconds", 0.0) or 0.0)
    deep_seconds = float(usage.get("deep_total_seconds", 0.0) or 0.0)
    vision_seconds = float(usage.get("vision_total_seconds", 0.0) or 0.0)
    slowest_model = str(usage.get("slowest_model", "none"))
    slowest_seconds = float(usage.get("slowest_seconds", 0.0) or 0.0)
    print(
        "[pipeline] model_timing",
        f"job_id={job_id}",
        f"text={text_seconds:.3f}s",
        f"deep={deep_seconds:.3f}s",
        f"vision={vision_seconds:.3f}s",
        f"slowest={slowest_model}:{slowest_seconds:.3f}s",
    )
