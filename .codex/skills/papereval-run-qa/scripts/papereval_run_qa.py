#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "app.db"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_json_any(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _job_row(job_id: int) -> dict[str, Any]:
    if not DB_PATH.exists():
        return {}
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            select id, document_id, status, progress, message, created_at, updated_at
            from job
            where id = ?
            """,
            (job_id,),
        ).fetchone()
    return dict(row) if row else {}


def _job_document_id(job_id: int) -> int | None:
    row = _job_row(job_id)
    document_id = row.get("document_id")
    return int(document_id) if document_id is not None else None


def _latest_document_id() -> int | None:
    candidates: list[tuple[float, int]] = []
    for path in DATA_DIR.glob("doc_*/artifacts/analysis_diagnostics.json"):
        try:
            document_id = int(path.parents[1].name.removeprefix("doc_"))
            candidates.append((path.stat().st_mtime, document_id))
        except Exception:
            continue
    if not candidates:
        return None
    return max(candidates)[1]


def _artifact_dir(document_id: int) -> Path:
    return DATA_DIR / f"doc_{document_id}" / "artifacts"


def _duration_rows(document_id: int) -> list[tuple[str, float, str, str]]:
    artifacts = _artifact_dir(document_id)
    rows: list[tuple[str, float, str, str]] = []
    timeline = _load_json(artifacts / "run_timeline.json").get("timeline", [])
    if isinstance(timeline, list):
        for event in timeline:
            if isinstance(event, dict):
                rows.append(
                    (
                        f"pipeline:{event.get('step', 'unknown')}",
                        float(event.get("duration_seconds", 0.0) or 0.0),
                        str(event.get("started_at", "")),
                        str(event.get("ended_at", "")),
                    )
                )
    diagnostics = _load_json(artifacts / "analysis_diagnostics.json").get("diagnostics", {})
    if isinstance(diagnostics, dict):
        analysis_timeline = diagnostics.get("analysis_timeline", [])
        if isinstance(analysis_timeline, list):
            for event in analysis_timeline:
                if isinstance(event, dict):
                    rows.append(
                        (
                            f"analysis:{event.get('stage', 'unknown')}",
                            float(event.get("duration_seconds", 0.0) or 0.0),
                            str(event.get("started_at", "")),
                            str(event.get("ended_at", "")),
                        )
                    )
    return sorted(rows, key=lambda row: row[1], reverse=True)


def _diagnostic_summary(document_id: int) -> dict[str, Any]:
    diagnostics = _load_json(_artifact_dir(document_id) / "analysis_diagnostics.json").get("diagnostics", {})
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    return {
        "analysis_total_seconds": diagnostics.get("analysis_timing", {}).get("analysis_total_seconds"),
        "parse_timing": diagnostics.get("parse_timing", {}),
        "model_usage": diagnostics.get("model_usage", {}),
        "openai_usage": diagnostics.get("openai_usage", {}),
        "coverage": diagnostics.get("coverage", {}),
        "fallback_counts_by_reason": diagnostics.get("fallback_counts_by_reason", {}),
        "sections_fallback_used": diagnostics.get("sections_fallback_used"),
        "vision_input_diagnostics": diagnostics.get("vision_input_diagnostics", {}),
    }


def _file_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"present": False}
    try:
        stat = path.stat()
    except OSError:
        return {"present": True}
    return {"present": True, "bytes": stat.st_size, "mtime": int(stat.st_mtime)}


def _stage_statuses(document_id: int, job_id: int | None = None) -> dict[str, Any]:
    artifacts = _artifact_dir(document_id)
    parse_diag = _load_json(artifacts / "parse_diagnostics.json")
    parser_assets = _load_json(artifacts / "parser_asset_diagnostics.json")
    source_manifest = _load_json(artifacts / "source_manifest.json")
    analysis_diag = _load_json(artifacts / "analysis_diagnostics.json").get("diagnostics", {})
    if not isinstance(analysis_diag, dict):
        analysis_diag = {}
    retention = _load_json(artifacts / "information_retention_audit.json")
    pdffigures_stats = _load_json_any(artifacts / "pdffigures2" / "stats.json")
    if not isinstance(pdffigures_stats, list):
        pdffigures_stats = []

    timeline_events = _load_json(artifacts / "run_timeline.json").get("timeline", [])
    completed_steps = {
        str(event.get("step", "")): event
        for event in timeline_events
        if isinstance(event, dict) and str(event.get("status", "")) == "completed"
    }
    analysis_timing = analysis_diag.get("analysis_timing", {})
    if not isinstance(analysis_timing, dict):
        analysis_timing = {}
    model_usage = analysis_diag.get("model_usage", {})
    if not isinstance(model_usage, dict):
        model_usage = {}
    stage_model_usage = analysis_diag.get("stage_model_usage", {})
    if not isinstance(stage_model_usage, dict):
        stage_model_usage = {}

    parse_counts = parse_diag.get("counts", {}) if isinstance(parse_diag.get("counts"), dict) else {}
    parser_counts: dict[str, int] = {}
    for asset in parser_assets.get("assets", []) if isinstance(parser_assets.get("assets"), list) else []:
        counts = asset.get("counts_delta", {}) if isinstance(asset, dict) else {}
        if not isinstance(counts, dict):
            continue
        for key, value in counts.items():
            parser_counts[str(key)] = parser_counts.get(str(key), 0) + int(value or 0)

    source_assets = source_manifest.get("selected_assets", [])
    supplements = source_manifest.get("supplements", [])
    stage_metrics = retention.get("compact_summary", {}).get("stage_metrics", [])
    if not isinstance(stage_metrics, list):
        stage_metrics = []

    modality_rows: dict[str, dict[str, Any]] = {}
    for modality in ("text", "table", "figure", "supplement"):
        usage_key = "vision" if modality == "figure" else "deep" if modality in {"table", "supplement"} else "text"
        usage = stage_model_usage.get(modality, {})
        if not isinstance(usage, dict):
            usage = {}
        modality_rows[modality] = {
            "parsed_items": int(parse_counts.get("supp" if modality == "supplement" else modality, 0) or 0),
            "analysis_seconds": analysis_timing.get(modality),
            "model_calls": int(usage.get(f"{usage_key}_calls", model_usage.get(f"{usage_key}_calls", 0)) or 0),
            "model_errors": int(usage.get(f"{usage_key}_errors", model_usage.get(f"{usage_key}_errors", 0)) or 0),
        }

    return {
        "job": _job_row(job_id) if job_id is not None else {},
        "document_id": document_id,
        "artifacts": str(artifacts),
        "artifact_state": {
            "source_manifest": _file_state(artifacts / "source_manifest.json"),
            "parse_diagnostics": _file_state(artifacts / "parse_diagnostics.json"),
            "parser_asset_diagnostics": _file_state(artifacts / "parser_asset_diagnostics.json"),
            "analysis_diagnostics": _file_state(artifacts / "analysis_diagnostics.json"),
            "information_retention_audit": _file_state(artifacts / "information_retention_audit.json"),
            "error_log": _file_state(artifacts / "error.log"),
        },
        "source": {
            "selected_asset_count": len(source_assets) if isinstance(source_assets, list) else 0,
            "supplement_count": len(supplements) if isinstance(supplements, list) else 0,
        },
        "parse": {
            "completed": "parse_document_assets" in completed_steps or bool(parse_diag),
            "seconds": parse_diag.get("parse_seconds"),
            "counts": parse_counts,
            "parser_asset_counts": parser_counts,
            "pdffigures": pdffigures_stats[:3],
        },
        "modalities": modality_rows,
        "analysis": {
            "diagnostics_present": bool(analysis_diag),
            "analysis_total_seconds": analysis_timing.get("analysis_total_seconds"),
            "fallback_counts_by_reason": analysis_diag.get("fallback_counts_by_reason", {}),
            "run_validity": analysis_diag.get("run_validity", {}),
            "latency_profile_present": bool(analysis_diag.get("latency_profile")),
        },
        "retention": {
            "audit_present": bool(retention),
            "stage_metrics": stage_metrics,
        },
        "timeline_completed_steps": sorted(completed_steps),
    }


def _run_checks() -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "backend")
    commands = [
        [
            str(ROOT / ".venv" / "bin" / "python"),
            "-m",
            "py_compile",
            "backend/app/services/pipeline.py",
            "backend/app/services/analysis/runner.py",
            "backend/app/services/analysis/synthesis.py",
        ],
        [
            str(ROOT / ".venv" / "bin" / "python"),
            "-m",
            "pytest",
            "backend/tests/test_timing.py",
            "backend/tests/test_desktop_api.py::test_report_endpoint_exposes_analysis_diagnostics_field",
        ],
    ]
    status = 0
    for cmd in commands:
        print(f"\n$ {' '.join(cmd)}")
        proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True)
        status = max(status, int(proc.returncode))
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize PaperEval run QA artifacts.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--job-id", type=int)
    group.add_argument("--document-id", type=int)
    group.add_argument("--latest", action="store_true")
    parser.add_argument("--run-checks", action="store_true")
    parser.add_argument(
        "--stage-diagnostics",
        action="store_true",
        help="Print partial stage diagnostics for running, failed, or completed jobs without requiring a finished report.",
    )
    args = parser.parse_args()

    if args.job_id is not None:
        document_id = _job_document_id(args.job_id)
        if document_id is None:
            print(f"Job {args.job_id} was not found in {DB_PATH}.", file=sys.stderr)
            return 2
    elif args.document_id is not None:
        document_id = int(args.document_id)
    else:
        document_id = _latest_document_id()
        if document_id is None:
            print("No analysis diagnostics artifacts found.", file=sys.stderr)
            return 2

    artifacts = _artifact_dir(document_id)
    print(f"document_id: {document_id}")
    print(f"artifacts: {artifacts}")

    if args.stage_diagnostics:
        print("\nStage diagnostics:")
        print(json.dumps(_stage_statuses(document_id, job_id=args.job_id), indent=2, sort_keys=True))
        if args.run_checks:
            return _run_checks()
        return 0

    rows = _duration_rows(document_id)
    if rows:
        print("\nSlowest steps:")
        for name, seconds, started, ended in rows[:10]:
            print(f"- {name}: {seconds:.3f}s ({started} -> {ended})")
    else:
        print("\nNo timeline rows found.")

    print("\nDiagnostics:")
    print(json.dumps(_diagnostic_summary(document_id), indent=2, sort_keys=True))

    if args.run_checks:
        return _run_checks()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
