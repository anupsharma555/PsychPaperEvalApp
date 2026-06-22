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


def _latest_report_payload(document_id: int) -> dict[str, Any]:
    if not DB_PATH.exists():
        return {}
    try:
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute(
                """
                select payload
                from report
                where document_id = ?
                order by created_at desc
                limit 1
                """,
                (document_id,),
            ).fetchone()
    except sqlite3.Error:
        return {}
    if not row:
        return {}
    try:
        payload = json.loads(row[0] or "{}")
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _count_present(rows: list[Any], key: str) -> int:
    return sum(1 for row in rows if isinstance(row, dict) and row.get(key) not in (None, "", [], {}))


def _artifact_organization_audit(document_id: int) -> dict[str, Any]:
    artifacts = _artifact_dir(document_id)
    stage_index = _load_json(artifacts / "intermediate_stage_index.json")
    report = _latest_report_payload(document_id)
    source_manifest = _load_json(artifacts / "source_manifest.json")
    retention = _load_json(artifacts / "information_retention_audit.json")
    analysis_diag = _load_json(artifacts / "analysis_diagnostics.json").get("diagnostics", {})
    if not isinstance(analysis_diag, dict):
        analysis_diag = {}

    packets = report.get("evidence_packets")
    packets = packets if isinstance(packets, list) else []
    native_organization = report.get("artifact_organization")
    native_organization = native_organization if isinstance(native_organization, dict) else {}
    llm_input_inventory = (
        native_organization.get("llm_input_inventory")
        if isinstance(native_organization.get("llm_input_inventory"), dict)
        else {}
    )
    packet_total = len(packets)
    packet_modalities: dict[str, int] = {}
    packet_sections: dict[str, int] = {}
    for packet in packets:
        if not isinstance(packet, dict):
            continue
        modality = str(packet.get("modality") or "missing")
        section = str(packet.get("section_label") or "missing")
        packet_modalities[modality] = packet_modalities.get(modality, 0) + 1
        packet_sections[section] = packet_sections.get(section, 0) + 1

    completeness = {
        "finding_id": _count_present(packets, "finding_id"),
        "anchor": _count_present(packets, "anchor"),
        "section_label": _count_present(packets, "section_label"),
        "modality": _count_present(packets, "modality"),
        "category": _count_present(packets, "category"),
        "confidence": _count_present(packets, "confidence"),
        "statement": _count_present(packets, "statement"),
        "source_excerpt": _count_present(packets, "source_excerpt"),
        "evidence_refs": _count_present(packets, "evidence_refs"),
        "detail_types": _count_present(packets, "detail_types"),
        "usable_for_gold_comparison": _count_present(packets, "usable_for_gold_comparison"),
    }

    coverage = report.get("coverage")
    if not isinstance(coverage, dict):
        coverage = {}
    missing_supplement_refs = []
    for key in ("supp_figures", "supp_tables"):
        row = coverage.get(key)
        if isinstance(row, dict):
            refs = row.get("missing_refs")
            if isinstance(refs, list):
                missing_supplement_refs.extend(str(ref) for ref in refs)

    source_supplements = source_manifest.get("supplements")
    source_supplement_count = len(source_supplements) if isinstance(source_supplements, list) else 0
    supplement_packet_count = packet_modalities.get("supplement", 0)
    figure_packet_count = packet_modalities.get("figure", 0)
    usable_figure_packet_count = sum(
        1
        for packet in packets
        if isinstance(packet, dict)
        and packet.get("modality") == "figure"
        and bool(packet.get("usable_for_gold_comparison"))
    )

    stage_metrics = retention.get("compact_summary", {}).get("stage_metrics", [])
    if not isinstance(stage_metrics, list):
        stage_metrics = []
    worst_loss = None
    for metric in stage_metrics:
        if not isinstance(metric, dict):
            continue
        if worst_loss is None or int(metric.get("lost_here_count", 0) or 0) > int(
            worst_loss.get("lost_here_count", 0) or 0
        ):
            worst_loss = metric

    issues: list[dict[str, Any]] = []
    if report:
        final_keys = {"executive_summary", "executive_report", "sections", "key_findings"}
        intermediate_keys = {"evidence_packets", "extractive_evidence", "presentation_evidence", "section_diagnostics"}
        if final_keys.intersection(report) and intermediate_keys.intersection(report):
            issues.append(
                {
                    "code": "flat_payload_mixes_final_and_intermediate_fields",
                    "severity": "medium",
                    "why": "The report payload exposes generated report fields beside evidence and diagnostic fields without a stage wrapper.",
                }
            )
    if packet_total and completeness["source_excerpt"] < packet_total:
        issues.append(
            {
                "code": "evidence_packets_missing_source_excerpt",
                "severity": "medium",
                "count": packet_total - completeness["source_excerpt"],
                "why": "Packets without source excerpts are harder to trace back to raw extraction and can obscure why facts were selected.",
            }
        )
    if packet_total and completeness["detail_types"] < packet_total:
        issues.append(
            {
                "code": "evidence_packets_missing_detail_types",
                "severity": "medium",
                "count": packet_total - completeness["detail_types"],
                "why": "Untyped packets weaken selection for statistics, sensitivity analyses, secondary findings, and other benchmark slots.",
            }
        )
    if packet_sections.get("unknown", 0):
        issues.append(
            {
                "code": "evidence_packets_unknown_section",
                "severity": "medium",
                "count": packet_sections["unknown"],
                "why": "Unknown sections make it difficult to preserve section boundaries during synthesis and scoring.",
            }
        )
    if figure_packet_count and usable_figure_packet_count == 0:
        issues.append(
            {
                "code": "figure_packets_not_usable_for_gold_comparison",
                "severity": "medium",
                "count": figure_packet_count,
                "why": "Figure evidence exists but is not marked usable for comparison, so visual evidence may be undercounted downstream.",
            }
        )
    if source_supplement_count == 0 and supplement_packet_count:
        issues.append(
            {
                "code": "main_text_supplement_references_are_labeled_as_supplement_packets",
                "severity": "medium",
                "count": supplement_packet_count,
                "why": "Supplement-referencing main text is useful, but labeling it as supplement modality can confuse source availability diagnostics.",
            }
        )
    if missing_supplement_refs and not str(report.get("supplement_availability_note") or "").strip():
        issues.append(
            {
                "code": "missing_supplements_without_availability_note",
                "severity": "high",
                "count": len(missing_supplement_refs),
                "why": "The source manifest has no supplement files while coverage reports missing supplement refs, but the final supplement note is empty.",
            }
        )

    return {
        "report_payload_present": bool(report),
        "top_level_key_count": len(report),
        "stage_boundaries": {
            "intermediate_stage_index_present": bool(stage_index),
            "source_manifest_present": bool(source_manifest),
            "report_payload_present": bool(report),
            "evidence_packets_present": bool(packet_total),
            "native_artifact_organization_present": bool(native_organization),
            "section_diagnostics_present": isinstance(report.get("section_diagnostics"), dict),
            "analysis_diagnostics_present": bool(analysis_diag),
            "retention_audit_present": bool(retention),
        },
        "native_artifact_organization_quality_flags": [
            str(flag)
            for flag in native_organization.get("quality_flags", [])
            if str(flag).strip()
        ][:16],
        "llm_input_inventory": _llm_input_inventory_summary(llm_input_inventory),
        "intermediate_stage_index": _stage_index_summary(stage_index),
        "evidence_packet_total": packet_total,
        "evidence_packet_field_completeness": completeness,
        "evidence_packet_modalities": dict(sorted(packet_modalities.items())),
        "evidence_packet_sections": dict(sorted(packet_sections.items())),
        "supplement_source_consistency": {
            "uploaded_supplement_count": source_supplement_count,
            "supplement_packet_count": supplement_packet_count,
            "missing_supplement_ref_count": len(missing_supplement_refs),
            "supplement_availability_note_present": bool(
                str(report.get("supplement_availability_note") or "").strip()
            ),
        },
        "figure_source_consistency": {
            "figure_packet_count": figure_packet_count,
            "usable_figure_packet_count": usable_figure_packet_count,
        },
        "retention_worst_loss_stage": {
            "stage": str(worst_loss.get("stage") or ""),
            "lost_here_count": int(worst_loss.get("lost_here_count", 0) or 0),
            "wrong_section_rate": worst_loss.get("wrong_section_rate"),
        }
        if isinstance(worst_loss, dict)
        else {},
        "issues": issues,
    }


def _llm_input_inventory_summary(inventory: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(inventory, dict) or not inventory:
        return {}
    records = inventory.get("selected_detail_records")
    record_rows = records if isinstance(records, list) else []
    return {
        "schema_version": inventory.get("schema_version"),
        "eligible_scientific_detail_count": int(inventory.get("eligible_scientific_detail_count", 0) or 0),
        "selected_prompt_detail_count": int(inventory.get("selected_prompt_detail_count", 0) or 0),
        "omitted_candidate_count": int(inventory.get("omitted_candidate_count", 0) or 0),
        "selected_quality": inventory.get("selected_quality") if isinstance(inventory.get("selected_quality"), dict) else {},
        "focus_slot_counts": inventory.get("focus_slot_counts") if isinstance(inventory.get("focus_slot_counts"), dict) else {},
        "quality_flags": [
            str(flag)
            for flag in inventory.get("quality_flags", [])
            if str(flag).strip()
        ][:16]
        if isinstance(inventory.get("quality_flags"), list)
        else [],
        "selected_detail_refs": [
            {
                "prompt_index": int(row.get("prompt_index", 0) or 0),
                "section_label": str(row.get("section_label") or ""),
                "source_modality": str(row.get("source_modality") or ""),
                "detail_types": [
                    str(item)
                    for item in row.get("detail_types", [])
                    if str(item).strip()
                ][:8]
                if isinstance(row.get("detail_types"), list)
                else [],
                "evidence_refs": [
                    str(ref)
                    for ref in row.get("evidence_refs", [])
                    if str(ref).strip()
                ][:5]
                if isinstance(row.get("evidence_refs"), list)
                else [],
                "statement_sha256": str(row.get("statement_sha256") or ""),
                "source_excerpt_sha256": str(row.get("source_excerpt_sha256") or ""),
            }
            for row in record_rows
            if isinstance(row, dict)
        ][:12],
    }


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
            "intermediate_stage_index": _file_state(artifacts / "intermediate_stage_index.json"),
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
        "artifact_organization": _artifact_organization_audit(document_id),
        "timeline_completed_steps": sorted(completed_steps),
    }


def _stage_index_summary(stage_index: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(stage_index, dict) or not stage_index:
        return {}
    stages = stage_index.get("stages")
    stage_rows = stages if isinstance(stages, list) else []
    transitions = stage_index.get("stage_transitions")
    transition_rows = transitions if isinstance(transitions, list) else []
    readiness = stage_index.get("llm_input_readiness")
    return {
        "schema_version": stage_index.get("schema_version"),
        "stage_order": [
            str(stage.get("stage_id"))
            for stage in stage_rows
            if isinstance(stage, dict) and str(stage.get("stage_id") or "").strip()
        ],
        "quality_flags": [
            str(flag)
            for flag in stage_index.get("quality_flags", [])
            if str(flag).strip()
        ][:16]
        if isinstance(stage_index.get("quality_flags"), list)
        else [],
        "stage_transitions": [
            {
                "transition_id": str(row.get("transition_id") or ""),
                "from_stage": str(row.get("from_stage") or ""),
                "to_stage": str(row.get("to_stage") or ""),
                "loss_count": int(row.get("loss_count", 0) or 0),
                "loss_rate": row.get("loss_rate"),
                "diagnostic_flags": [
                    str(flag)
                    for flag in row.get("diagnostic_flags", [])
                    if str(flag).strip()
                ][:8]
                if isinstance(row.get("diagnostic_flags"), list)
                else [],
            }
            for row in transition_rows
            if isinstance(row, dict)
        ][:12],
        "llm_input_readiness": readiness if isinstance(readiness, dict) else {},
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
