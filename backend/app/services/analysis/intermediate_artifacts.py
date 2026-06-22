from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any

from sqlmodel import Session, select

from app.db.models import Asset, Chunk, Report
from app.services.storage import artifacts_dir


STAGE_INDEX_FILENAME = "intermediate_stage_index.json"
LLM_INPUT_INVENTORY_FILENAME = "llm_input_inventory.json"
STAGE_INDEX_VERSION = 1


def write_intermediate_stage_index(
    session: Session,
    document_id: int,
    *,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifact_root = artifacts_dir(document_id)
    artifact_root.mkdir(parents=True, exist_ok=True)
    _write_llm_input_inventory_sidecar(session, document_id, artifact_root=artifact_root)
    payload = build_intermediate_stage_index(
        session,
        document_id,
        diagnostics=diagnostics,
    )
    path = artifact_root / STAGE_INDEX_FILENAME
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, default=str), encoding="utf-8")
    return payload


def build_intermediate_stage_index(
    session: Session,
    document_id: int,
    *,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifact_root = artifacts_dir(document_id)
    source_manifest = _load_json(artifact_root / "source_manifest.json")
    parse_diagnostics = _load_json(artifact_root / "parse_diagnostics.json")
    parser_asset_diagnostics = _load_json(artifact_root / "parser_asset_diagnostics.json")
    analysis_diagnostics = diagnostics if isinstance(diagnostics, dict) else _diagnostics_payload(artifact_root)
    retention = _load_json(artifact_root / "information_retention_audit.json")
    timeline = _load_json(artifact_root / "run_timeline.json")
    report = _latest_report_payload(session, document_id)
    assets = list(session.exec(select(Asset).where(Asset.document_id == document_id)).all())
    chunks = list(session.exec(select(Chunk).where(Chunk.document_id == document_id)).all())

    stage_records = _stage_records(
        source_manifest=source_manifest,
        parse_diagnostics=parse_diagnostics,
        parser_asset_diagnostics=parser_asset_diagnostics,
        analysis_diagnostics=analysis_diagnostics,
        retention=retention,
        timeline=timeline,
        report=report,
        assets=assets,
        chunks=chunks,
        artifact_root=artifact_root,
    )
    quality_flags = _index_quality_flags(stage_records)

    return {
        "schema_version": STAGE_INDEX_VERSION,
        "document_id": document_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "purpose": (
            "Ordered index of intermediate PaperEval data products for diagnostics and LLM-input "
            "readiness review. This index is derived from saved artifacts and does not change synthesis."
        ),
        "stage_order": [stage["stage_id"] for stage in stage_records],
        "stages": stage_records,
        "stage_transitions": _stage_transitions(stage_records),
        "quality_flags": quality_flags,
        "llm_input_readiness": _llm_input_readiness(stage_records, quality_flags),
    }


def _stage_records(
    *,
    source_manifest: dict[str, Any],
    parse_diagnostics: dict[str, Any],
    parser_asset_diagnostics: dict[str, Any],
    analysis_diagnostics: dict[str, Any],
    retention: dict[str, Any],
    timeline: dict[str, Any],
    report: dict[str, Any],
    assets: list[Asset],
    chunks: list[Chunk],
    artifact_root: Path,
) -> list[dict[str, Any]]:
    report_packets = _report_evidence_packets(report)
    artifact_organization = _artifact_organization(report)
    stage_metrics = _retention_stage_metrics(retention)
    chunk_counts = _chunk_counts(chunks)
    modality_counts = _analysis_modality_counts(analysis_diagnostics)

    return [
        {
            "stage_id": "source_manifest",
            "label": "Source Manifest",
            "role": "Declares selected source assets and supplement availability.",
            "artifact_paths": [_artifact_state(artifact_root / "source_manifest.json", artifact_root)],
            "record_counts": {
                "assets": len(assets),
                "selected_assets": _list_len(source_manifest.get("selected_assets")),
                "supplements": _list_len(source_manifest.get("supplements")),
            },
            "quality": {
                "present": bool(source_manifest),
                "source_type": str(source_manifest.get("source_type") or ""),
                "status": str(source_manifest.get("status") or ""),
            },
        },
        {
            "stage_id": "parsed_chunks",
            "label": "Parsed Chunks",
            "role": "Structured text/table/figure/supplement chunks before modality analysis.",
            "artifact_paths": [
                _artifact_state(artifact_root / "parse_diagnostics.json", artifact_root),
                _artifact_state(artifact_root / "parser_asset_diagnostics.json", artifact_root),
            ],
            "record_counts": {
                "chunks_total": len(chunks),
                **{f"{key}_chunks": value for key, value in chunk_counts.items()},
                **{f"{key}_parsed": value for key, value in _int_dict(parse_diagnostics.get("counts")).items()},
            },
            "quality": {
                "present": bool(chunks) or bool(parse_diagnostics),
                "parser_asset_status_counts": _parser_asset_status_counts(parser_asset_diagnostics),
            },
        },
        {
            "stage_id": "modality_packets",
            "label": "Modality Packets",
            "role": "Text/table/figure/supplement evidence packets before final synthesis.",
            "artifact_paths": [_artifact_state(artifact_root / "analysis_diagnostics.json", artifact_root)],
            "record_counts": {
                **{f"{key}_packets": value for key, value in modality_counts.items()},
                "evidence_packets": len(report_packets),
            },
            "quality": _packet_quality(report_packets),
        },
        {
            "stage_id": "audited_evidence_packets",
            "label": "Audited Evidence Packets",
            "role": "Report-level evidence packets with section, typing, source excerpt, and comparison readiness.",
            "artifact_paths": [],
            "record_counts": _artifact_int_block(artifact_organization, "audited_packet_quality"),
            "quality": {
                "native_artifact_organization": bool(artifact_organization),
                "quality_flags": _as_str_list(artifact_organization.get("quality_flags")),
            },
        },
        {
            "stage_id": "synthesis_inputs",
            "label": "Synthesis Inputs",
            "role": "Scientific details and focus slots selected for final narrative synthesis.",
            "artifact_paths": [_artifact_state(artifact_root / LLM_INPUT_INVENTORY_FILENAME, artifact_root)],
            "record_counts": _synthesis_input_counts(report, artifact_organization),
            "quality": {
                "llm_input_inventory": _llm_input_inventory_quality(report, artifact_organization),
                "synthesis_evidence_warnings": _as_str_list(
                    ((report.get("section_diagnostics") or {}).get("synthesis_evidence_warnings"))
                    if isinstance(report.get("section_diagnostics"), dict)
                    else []
                ),
            },
        },
        {
            "stage_id": "retention_audit",
            "label": "Information Retention Audit",
            "role": "Maps expected source/retrieval items across intermediate stages.",
            "artifact_paths": [_artifact_state(artifact_root / "information_retention_audit.json", artifact_root)],
            "record_counts": {
                "stage_metrics": len(stage_metrics),
                "worst_loss_count": int(_worst_retention_stage(stage_metrics).get("lost_here_count", 0) or 0),
            },
            "quality": {
                "present": bool(retention),
                "worst_loss_stage": _worst_retention_stage(stage_metrics),
            },
        },
        {
            "stage_id": "final_report",
            "label": "Final Report",
            "role": "User-facing structured report assembled from intermediate evidence.",
            "artifact_paths": [],
            "record_counts": _final_report_counts(report),
            "quality": {
                "present": bool(report),
                "top_level_key_count": len(report),
                "has_artifact_organization": bool(artifact_organization),
                "has_supplement_availability_note": bool(str(report.get("supplement_availability_note") or "").strip()),
            },
        },
        {
            "stage_id": "runtime_diagnostics",
            "label": "Runtime Diagnostics",
            "role": "Timing, fallback, model usage, and run-validity context.",
            "artifact_paths": [
                _artifact_state(artifact_root / "analysis_diagnostics.json", artifact_root),
                _artifact_state(artifact_root / "run_timeline.json", artifact_root),
                _artifact_state(artifact_root / "latency_profile.json", artifact_root),
            ],
            "record_counts": {
                "timeline_events": _list_len(timeline.get("timeline")),
                "fallback_reasons": len(
                    _dict_path(analysis_diagnostics, "run_validity", "fallback_reasons", default=[])
                    if isinstance(_dict_path(analysis_diagnostics, "run_validity", "fallback_reasons", default=[]), list)
                    else []
                ),
            },
            "quality": {
                "run_validity": str(_dict_path(analysis_diagnostics, "run_validity", "run_validity", default="")),
                "run_valid": bool(_dict_path(analysis_diagnostics, "run_validity", "valid", default=False)),
            },
        },
    ]


def _write_llm_input_inventory_sidecar(session: Session, document_id: int, *, artifact_root: Path) -> Path | None:
    report = _latest_report_payload(session, document_id)
    inventory, source = _llm_input_inventory_from_report(report)
    if not inventory:
        return None
    payload = {
        "schema_version": 1,
        "document_id": document_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "source": source,
        "inventory": inventory,
    }
    path = artifact_root / LLM_INPUT_INVENTORY_FILENAME
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, default=str), encoding="utf-8")
    return path


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _diagnostics_payload(artifact_root: Path) -> dict[str, Any]:
    payload = _load_json(artifact_root / "analysis_diagnostics.json")
    diagnostics = payload.get("diagnostics")
    return diagnostics if isinstance(diagnostics, dict) else {}


def _latest_report_payload(session: Session, document_id: int) -> dict[str, Any]:
    row = session.exec(
        select(Report)
        .where(Report.document_id == document_id)
        .order_by(Report.created_at.desc())
    ).first()
    if row is None:
        return {}
    try:
        payload = json.loads(row.payload or "{}")
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _artifact_state(path: Path, artifact_root: Path) -> dict[str, Any]:
    try:
        stat = path.stat()
    except OSError:
        return {"path": str(path.relative_to(artifact_root)), "present": False}
    return {
        "path": str(path.relative_to(artifact_root)),
        "present": True,
        "bytes": int(stat.st_size),
        "mtime": int(stat.st_mtime),
    }


def _chunk_counts(chunks: list[Chunk]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for chunk in chunks:
        key = str(chunk.modality or "missing")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _analysis_modality_counts(diagnostics: dict[str, Any]) -> dict[str, int]:
    counts = diagnostics.get("modality_packet_counts")
    return _int_dict(counts)


def _parser_asset_status_counts(payload: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    assets = payload.get("assets")
    if not isinstance(assets, list):
        return counts
    for item in assets:
        if not isinstance(item, dict):
            continue
        status = str(item.get("status") or "missing")
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def _report_evidence_packets(report: dict[str, Any]) -> list[dict[str, Any]]:
    packets = report.get("evidence_packets")
    if not isinstance(packets, list):
        return []
    return [packet for packet in packets if isinstance(packet, dict)]


def _artifact_organization(report: dict[str, Any]) -> dict[str, Any]:
    value = report.get("artifact_organization")
    return value if isinstance(value, dict) else {}


def _llm_input_inventory_from_report(report: dict[str, Any]) -> tuple[dict[str, Any], str]:
    organization = _artifact_organization(report)
    value = organization.get("llm_input_inventory")
    if isinstance(value, dict) and value:
        return value, "report.artifact_organization.llm_input_inventory"
    fallback = _fallback_llm_input_inventory(report)
    if fallback:
        return fallback, "report.scientific_details_and_section_diagnostics"
    return {}, ""


def _fallback_llm_input_inventory(report: dict[str, Any]) -> dict[str, Any]:
    details = [item for item in report.get("scientific_details", []) if isinstance(item, dict)]
    if not details:
        return {}
    diagnostics = report.get("section_diagnostics") if isinstance(report.get("section_diagnostics"), dict) else {}
    prompt_count = int(diagnostics.get("scientific_details_prompt_count", 0) or 0)
    if prompt_count <= 0:
        prompt_count = min(12, len(details))
    selected = details[:prompt_count]
    selected_records = [
        _llm_input_detail_record(detail, index=index)
        for index, detail in enumerate(selected, start=1)
    ]
    selected_records = [record for record in selected_records if record]
    missing_excerpt = sum(1 for record in selected_records if not record.get("source_excerpt_sha256"))
    missing_types = sum(1 for record in selected_records if not record.get("detail_types"))
    unknown_section = sum(1 for record in selected_records if record.get("section_label") == "unknown")
    evidence_plan = diagnostics.get("synthesis_evidence_plan") if isinstance(diagnostics.get("synthesis_evidence_plan"), dict) else {}
    critical_missing = evidence_plan.get("critical_missing_focus_slots")
    quality_flags: list[str] = ["recovered_from_report_scientific_details"]
    if missing_excerpt:
        quality_flags.append("selected_prompt_details_missing_source_excerpt")
    if missing_types:
        quality_flags.append("selected_prompt_details_missing_detail_types")
    if unknown_section:
        quality_flags.append("selected_prompt_details_unknown_section")
    if isinstance(critical_missing, list) and critical_missing:
        quality_flags.append("critical_focus_slots_missing")
    return {
        "schema_version": 1,
        "eligible_scientific_detail_count": len(details),
        "selected_prompt_detail_count": len(selected_records),
        "omitted_candidate_count": max(0, len(details) - len(selected_records)),
        "selected_detail_records": selected_records,
        "selected_detail_counts": {
            "by_section": _count_values(str(record.get("section_label") or "unknown") for record in selected_records),
            "by_modality": _count_values(str(record.get("source_modality") or "unknown") for record in selected_records),
            "by_detail_type": _count_values(
                detail_type
                for record in selected_records
                for detail_type in record.get("detail_types", [])
                if str(detail_type).strip()
            ),
        },
        "selected_quality": {
            "missing_source_excerpt": missing_excerpt,
            "missing_detail_types": missing_types,
            "unknown_section": unknown_section,
        },
        "focus_slot_counts": {
            "total": _list_len(evidence_plan.get("focus_slots")),
            "missing": int(evidence_plan.get("missing_focus_slot_count", 0) or 0),
            "critical_missing": _list_len(critical_missing),
        },
        "quality_flags": _unique_strings(quality_flags),
    }


def _llm_input_detail_record(detail: dict[str, Any], *, index: int) -> dict[str, Any]:
    statement = str(detail.get("statement") or "").strip()
    source_excerpt = str(detail.get("source_excerpt") or "").strip()
    if not statement:
        return {}
    detail_types = _as_str_list(detail.get("detail_types"))
    evidence_refs = _as_str_list(detail.get("evidence_refs"))[:5]
    return {
        "selection_index": index,
        "statement_sha256": _sha256_text(statement),
        "statement_preview": statement[:240],
        "source_excerpt_sha256": _sha256_text(source_excerpt) if source_excerpt else "",
        "source_excerpt_preview": source_excerpt[:240],
        "evidence_refs": evidence_refs,
        "source_modality": str(detail.get("source_modality") or "unknown"),
        "section_label": str(detail.get("section_label") or "unknown"),
        "category": str(detail.get("category") or "other"),
        "detail_types": detail_types,
        "confidence": detail.get("confidence", 0.0),
    }


def _packet_quality(packets: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(packets)
    source_excerpt = sum(1 for packet in packets if str(packet.get("source_excerpt") or "").strip())
    typed = sum(1 for packet in packets if packet.get("detail_types"))
    unknown = sum(1 for packet in packets if str(packet.get("section_label") or "").strip().lower() == "unknown")
    usable = sum(1 for packet in packets if packet.get("usable_for_gold_comparison"))
    return {
        "total": total,
        "source_excerpt_present": source_excerpt,
        "source_excerpt_missing": max(0, total - source_excerpt),
        "typed_packet_count": typed,
        "untyped_packet_count": max(0, total - typed),
        "unknown_section_count": unknown,
        "usable_for_gold_comparison": usable,
    }


def _synthesis_input_counts(report: dict[str, Any], artifact_organization: dict[str, Any]) -> dict[str, int]:
    native = _artifact_int_block(artifact_organization, "synthesis_input_counts")
    inventory = _artifact_dict_block(artifact_organization, "llm_input_inventory") or _fallback_llm_input_inventory(report)
    if inventory:
        focus_slot_counts = _int_dict(inventory.get("focus_slot_counts"))
        native = {
            **native,
            "llm_input_selected_prompt_details": int(inventory.get("selected_prompt_detail_count", 0) or 0),
            "llm_input_eligible_scientific_details": int(inventory.get("eligible_scientific_detail_count", 0) or 0),
            "llm_input_omitted_candidates": int(inventory.get("omitted_candidate_count", 0) or 0),
        }
        native.setdefault("critical_missing_focus_slots", int(focus_slot_counts.get("critical_missing", 0) or 0))
    if native:
        return native
    return {
        "scientific_details": _list_len(report.get("scientific_details")),
        "prompt_scientific_details": 0,
        "critical_missing_focus_slots": _list_len(
            _dict_path(report, "section_diagnostics", "synthesis_evidence_plan", "critical_missing_focus_slots", default=[])
        ),
    }


def _llm_input_inventory_quality(report: dict[str, Any], artifact_organization: dict[str, Any]) -> dict[str, Any]:
    inventory = _artifact_dict_block(artifact_organization, "llm_input_inventory") or _fallback_llm_input_inventory(report)
    if not inventory:
        return {}
    return {
        "present": True,
        "quality_flags": _as_str_list(inventory.get("quality_flags")),
        "selected_quality": _int_dict(inventory.get("selected_quality")),
        "focus_slot_counts": _int_dict(inventory.get("focus_slot_counts")),
    }


def _retention_stage_metrics(retention: dict[str, Any]) -> list[dict[str, Any]]:
    rows = _dict_path(retention, "compact_summary", "stage_metrics", default=[])
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _worst_retention_stage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    worst: dict[str, Any] = {}
    for row in rows:
        if int(row.get("lost_here_count", 0) or 0) > int(worst.get("lost_here_count", 0) or 0):
            worst = row
    return {
        "stage": str(worst.get("stage") or ""),
        "lost_here_count": int(worst.get("lost_here_count", 0) or 0),
        "wrong_section_rate": worst.get("wrong_section_rate"),
    } if worst else {}


def _final_report_counts(report: dict[str, Any]) -> dict[str, int]:
    sections = report.get("sections")
    key_findings = report.get("key_findings")
    details = report.get("scientific_details")
    packets = report.get("evidence_packets")
    return {
        "sections": _list_len(sections),
        "key_findings": _list_len(key_findings),
        "scientific_details": _list_len(details),
        "evidence_packets": _list_len(packets),
    }


def _index_quality_flags(stages: list[dict[str, Any]]) -> list[str]:
    flags: list[str] = []
    by_id = {str(stage.get("stage_id")): stage for stage in stages}
    modality_quality = by_id.get("modality_packets", {}).get("quality", {})
    retention_quality = by_id.get("retention_audit", {}).get("quality", {})
    final_quality = by_id.get("final_report", {}).get("quality", {})
    if int(modality_quality.get("source_excerpt_missing", 0) or 0):
        flags.append("packet_source_excerpts_incomplete")
    if int(modality_quality.get("untyped_packet_count", 0) or 0):
        flags.append("packet_detail_typing_incomplete")
    if int(modality_quality.get("unknown_section_count", 0) or 0):
        flags.append("packet_section_assignment_incomplete")
    worst = retention_quality.get("worst_loss_stage")
    if isinstance(worst, dict) and int(worst.get("lost_here_count", 0) or 0):
        flags.append(f"retention_loss_at_{worst.get('stage')}")
    if not final_quality.get("has_artifact_organization"):
        flags.append("final_report_missing_native_artifact_organization")
    return _unique_strings(flags)


def _stage_transitions(stages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {str(stage.get("stage_id")): stage for stage in stages}
    source_counts = _stage_counts(by_id, "source_manifest")
    parsed_counts = _stage_counts(by_id, "parsed_chunks")
    modality_counts = _stage_counts(by_id, "modality_packets")
    modality_quality = _stage_quality(by_id, "modality_packets")
    audited_counts = _stage_counts(by_id, "audited_evidence_packets")
    synthesis_counts = _stage_counts(by_id, "synthesis_inputs")
    retention_counts = _stage_counts(by_id, "retention_audit")
    retention_quality = _stage_quality(by_id, "retention_audit")
    final_counts = _stage_counts(by_id, "final_report")

    selected_assets = int(source_counts.get("selected_assets", 0) or 0)
    chunks_total = int(parsed_counts.get("chunks_total", 0) or 0)
    evidence_packets = int(modality_counts.get("evidence_packets", 0) or 0)
    source_missing = int(modality_quality.get("source_excerpt_missing", 0) or 0)
    untyped = int(modality_quality.get("untyped_packet_count", 0) or 0)
    unknown_sections = int(modality_quality.get("unknown_section_count", 0) or 0)
    usable_packets = int(modality_quality.get("usable_for_gold_comparison", 0) or 0)
    if not usable_packets:
        usable_packets = int(audited_counts.get("usable_for_gold_comparison", 0) or 0)
    scientific_details = int(synthesis_counts.get("scientific_details", 0) or 0)
    focus_slots = int(synthesis_counts.get("critical_missing_focus_slots", 0) or 0)
    key_findings = int(final_counts.get("key_findings", 0) or 0)
    sections = int(final_counts.get("sections", 0) or 0)
    retention_loss = int(retention_counts.get("worst_loss_count", 0) or 0)
    worst_loss_stage = retention_quality.get("worst_loss_stage")
    worst_loss_name = (
        str(worst_loss_stage.get("stage") or "")
        if isinstance(worst_loss_stage, dict)
        else ""
    )

    return [
        _transition_record(
            "source_to_parsed_chunks",
            "source_manifest",
            "parsed_chunks",
            input_count=selected_assets,
            output_count=chunks_total,
            input_unit="selected_assets",
            output_unit="chunks",
            diagnostic_flags=["no_parsed_chunks_from_selected_assets"]
            if selected_assets and not chunks_total
            else [],
        ),
        _transition_record(
            "parsed_chunks_to_modality_packets",
            "parsed_chunks",
            "modality_packets",
            input_count=chunks_total,
            output_count=evidence_packets,
            input_unit="chunks",
            output_unit="evidence_packets",
            diagnostic_flags=["no_packets_from_chunks"] if chunks_total and not evidence_packets else [],
        ),
        _transition_record(
            "modality_packets_to_audited_packets",
            "modality_packets",
            "audited_evidence_packets",
            input_count=evidence_packets,
            output_count=usable_packets,
            input_unit="evidence_packets",
            output_unit="gold_usable_packets",
            diagnostic_flags=[
                flag
                for flag, count in (
                    ("source_excerpts_missing", source_missing),
                    ("detail_types_missing", untyped),
                    ("section_labels_unknown", unknown_sections),
                )
                if count
            ],
            quality_gaps={
                "source_excerpt_missing": source_missing,
                "untyped_packet_count": untyped,
                "unknown_section_count": unknown_sections,
            },
        ),
        _transition_record(
            "audited_packets_to_synthesis_inputs",
            "audited_evidence_packets",
            "synthesis_inputs",
            input_count=max(usable_packets, evidence_packets),
            output_count=scientific_details,
            input_unit="candidate_packets",
            output_unit="scientific_details",
            diagnostic_flags=["critical_focus_slots_missing"] if focus_slots else [],
            quality_gaps={"critical_missing_focus_slots": focus_slots},
        ),
        _transition_record(
            "synthesis_inputs_to_final_report",
            "synthesis_inputs",
            "final_report",
            input_count=scientific_details,
            output_count=key_findings + sections,
            input_unit="scientific_details",
            output_unit="sections_plus_key_findings",
            diagnostic_flags=["final_report_sparse"] if scientific_details and not (key_findings or sections) else [],
            quality_gaps={
                "sections": sections,
                "key_findings": key_findings,
            },
        ),
        _transition_record(
            "retention_audit_to_final_report",
            "retention_audit",
            "final_report",
            input_count=retention_loss,
            output_count=0,
            input_unit="worst_stage_lost_items",
            output_unit="unrecovered_items",
            diagnostic_flags=[f"retention_loss_at_{worst_loss_name}"] if worst_loss_name and retention_loss else [],
            quality_gaps={"worst_loss_count": retention_loss},
        ),
    ]


def _stage_counts(by_id: dict[str, dict[str, Any]], stage_id: str) -> dict[str, int]:
    return _int_dict(by_id.get(stage_id, {}).get("record_counts"))


def _stage_quality(by_id: dict[str, dict[str, Any]], stage_id: str) -> dict[str, Any]:
    quality = by_id.get(stage_id, {}).get("quality")
    return quality if isinstance(quality, dict) else {}


def _transition_record(
    transition_id: str,
    from_stage: str,
    to_stage: str,
    *,
    input_count: int,
    output_count: int,
    input_unit: str,
    output_unit: str,
    diagnostic_flags: list[str] | None = None,
    quality_gaps: dict[str, int] | None = None,
) -> dict[str, Any]:
    loss_count = max(0, int(input_count or 0) - int(output_count or 0))
    return {
        "transition_id": transition_id,
        "from_stage": from_stage,
        "to_stage": to_stage,
        "input_count": int(input_count or 0),
        "output_count": int(output_count or 0),
        "input_unit": input_unit,
        "output_unit": output_unit,
        "loss_count": loss_count,
        "loss_rate": _safe_rate(loss_count, int(input_count or 0)),
        "diagnostic_flags": _unique_strings(diagnostic_flags or []),
        "quality_gaps": _int_dict(quality_gaps or {}),
    }


def _safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(float(numerator) / float(denominator), 4)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _count_values(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value or "").strip()
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _llm_input_readiness(stages: list[dict[str, Any]], quality_flags: list[str]) -> dict[str, Any]:
    blocking = {
        "packet_source_excerpts_incomplete",
        "packet_detail_typing_incomplete",
        "packet_section_assignment_incomplete",
    }
    flag_set = set(quality_flags)
    return {
        "ready": not bool(flag_set & blocking),
        "blocking_flags": sorted(flag_set & blocking),
        "advisory_flags": sorted(flag_set - blocking),
        "stage_count": len(stages),
    }


def _artifact_int_block(artifact_organization: dict[str, Any], key: str) -> dict[str, int]:
    value = artifact_organization.get(key)
    return _int_dict(value)


def _artifact_dict_block(artifact_organization: dict[str, Any], key: str) -> dict[str, Any]:
    value = artifact_organization.get(key)
    return value if isinstance(value, dict) else {}


def _int_dict(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, int] = {}
    for key, raw in value.items():
        try:
            out[str(key)] = int(raw or 0)
        except Exception:
            continue
    return dict(sorted(out.items()))


def _dict_path(value: Any, *keys: str, default: Any = None) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def _list_len(value: Any) -> int:
    return len(value) if isinstance(value, list) else 0


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return _unique_strings(str(item).strip() for item in value if str(item).strip())


def _unique_strings(values: Any, *, max_items: int = 32) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) >= max_items:
            break
    return out
