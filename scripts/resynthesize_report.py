#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from sqlmodel import Session, select  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.db.models import Job, JobStatus, Report  # noqa: E402
from app.db.session import engine  # noqa: E402
from app.services.analysis.llm import set_openai_usage_context  # noqa: E402
from app.services.analysis.synthesis import _build_executive_report  # noqa: E402
from app.services.storage import artifacts_dir  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a new report version by rerunning executive synthesis from saved evidence."
    )
    parser.add_argument("--document-id", type=int, default=None, help="Document id to resynthesize.")
    parser.add_argument("--job-id", type=int, default=None, help="Use the document from a prior job id.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate saved evidence without writing a report or calling the LLM.",
    )
    parser.add_argument(
        "--allow-extractive-fallback",
        action="store_true",
        help="Allow writing the deterministic extractive fallback if LLM synthesis is not applied.",
    )
    parser.add_argument(
        "--create-job",
        action="store_true",
        help="Create a completed Job row for this resynthesis so it appears in job history.",
    )
    args = parser.parse_args()

    with Session(engine) as session:
        document_id = _resolve_document_id(session, document_id=args.document_id, job_id=args.job_id)
        source_report = _latest_report(session, document_id)
        payload = _load_report_payload(source_report)
        evidence = _extractive_evidence_from_payload(payload)
        evidence_counts = {
            section: len(rows)
            for section, rows in evidence.items()
            if isinstance(rows, list)
        }
        if not any(evidence_counts.values()):
            raise SystemExit(f"Report {source_report.id} has no saved extractive evidence to resynthesize.")

        if args.dry_run:
            print(
                json.dumps(
                    {
                        "document_id": document_id,
                        "source_report_id": source_report.id,
                        "evidence_counts": evidence_counts,
                        "would_call_llm": bool(settings.analysis_narrative_overrides_enabled),
                    },
                    indent=2,
                )
            )
            return 0

        job: Job | None = None
        if args.create_job:
            job = Job(
                document_id=document_id,
                status=JobStatus.running,
                progress=0.9,
                message="Resynthesizing saved report evidence",
            )
            session.add(job)
            session.commit()
            session.refresh(job)

        try:
            set_openai_usage_context(job_id=job.id if job else None, document_id=document_id, stage="resynthesis")
            executive_report = _build_executive_report(
                extractive_evidence=evidence,
                fallback_summary=str(payload.get("executive_summary", "") or ""),
            )
        except Exception:
            if job:
                job.status = JobStatus.failed
                job.progress = 1.0
                job.message = "Resynthesis failed"
                session.add(job)
                session.commit()
            raise

        if (
            settings.llm_provider_normalized == "openai"
            and not args.allow_extractive_fallback
            and not executive_report.get("synthesis_applied")
        ):
            if job:
                job.status = JobStatus.failed
                job.progress = 1.0
                job.message = "OpenAI resynthesis did not produce an accepted LLM synthesis"
                session.add(job)
                session.commit()
            raise SystemExit(
                "OpenAI resynthesis did not produce an accepted LLM synthesis; no report was written. "
                "Use --allow-extractive-fallback to write the deterministic fallback."
            )

        next_payload = dict(payload)
        overview = str(executive_report.get("overview", "") or "").strip()
        next_payload["executive_report"] = executive_report
        if overview:
            next_payload["executive_summary"] = overview
        next_payload["resynthesis"] = {
            "source_report_id": source_report.id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "mode": "executive_from_saved_extracts",
            "llm_synthesis_applied": bool(executive_report.get("synthesis_applied")),
            "evidence_counts": evidence_counts,
        }

        new_report = Report(document_id=document_id, payload=json.dumps(next_payload))
        session.add(new_report)
        session.commit()
        session.refresh(new_report)

        if job:
            job.status = JobStatus.completed
            job.progress = 1.0
            job.message = "Completed"
            session.add(job)
            session.commit()

        artifact_path = artifacts_dir(document_id) / f"resynthesis_report_{new_report.id}.json"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(
            json.dumps(
                {
                    "document_id": document_id,
                    "job_id": job.id if job else None,
                    "source_report_id": source_report.id,
                    "new_report_id": new_report.id,
                    "resynthesis": next_payload["resynthesis"],
                    "executive_report": executive_report,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        print(
            json.dumps(
                {
                    "document_id": document_id,
                    "job_id": job.id if job else None,
                    "source_report_id": source_report.id,
                    "new_report_id": new_report.id,
                    "artifact": str(artifact_path),
                    "llm_synthesis_applied": bool(executive_report.get("synthesis_applied")),
                    "evidence_counts": evidence_counts,
                },
                indent=2,
            )
        )
    return 0


def _resolve_document_id(session: Session, *, document_id: int | None, job_id: int | None) -> int:
    if document_id is not None:
        return int(document_id)
    if job_id is not None:
        job = session.get(Job, int(job_id))
        if not job:
            raise SystemExit(f"No job found for id {job_id}.")
        return int(job.document_id)
    job = session.exec(
        select(Job)
        .where(Job.status == JobStatus.completed)
        .order_by(Job.updated_at.desc())
    ).first()
    if not job:
        raise SystemExit("No completed jobs found. Pass --document-id explicitly.")
    return int(job.document_id)


def _latest_report(session: Session, document_id: int) -> Report:
    report = session.exec(
        select(Report)
        .where(Report.document_id == document_id)
        .order_by(Report.created_at.desc())
    ).first()
    if not report:
        raise SystemExit(f"No report found for document {document_id}.")
    return report


def _load_report_payload(report: Report) -> dict[str, Any]:
    try:
        payload = json.loads(report.payload or "{}")
    except Exception as exc:
        raise SystemExit(f"Report {report.id} payload is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"Report {report.id} payload is not a JSON object.")
    return payload


def _extractive_evidence_from_payload(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    evidence = payload.get("extractive_evidence")
    if isinstance(evidence, dict):
        return {
            str(section): [row for row in rows if isinstance(row, dict)]
            for section, rows in evidence.items()
            if isinstance(rows, list)
        }

    sections = payload.get("sections")
    if not isinstance(sections, dict):
        return {}

    out: dict[str, list[dict[str, Any]]] = {}
    for section, block in sections.items():
        if not isinstance(block, dict):
            continue
        rows: list[dict[str, Any]] = []
        for item in block.get("items", []):
            if not isinstance(item, dict):
                continue
            statement = str(item.get("statement", "") or "").strip()
            if not statement:
                continue
            rows.append(
                {
                    "statement": statement,
                    "rephrased_statement": statement,
                    "verbatim_text": statement,
                    "evidence_refs": item.get("evidence_refs", []),
                    "anchor": item.get("anchor", ""),
                    "confidence": item.get("confidence", 0.0),
                    "section_confidence": item.get("section_confidence", item.get("confidence", 0.0)),
                    "source_modality": item.get("source_modality", "text"),
                }
            )
        out[str(section)] = rows
    return out


if __name__ == "__main__":
    raise SystemExit(main())
