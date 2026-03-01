from __future__ import annotations

import json
import multiprocessing as mp
import os
from queue import Empty as QueueEmpty
import re
from time import monotonic
from typing import Any, Callable

from sqlalchemy import delete
from sqlmodel import Session, select

from app.db.models import Asset, Chunk, Discrepancy, Document, Finding, Report
from app.core.config import settings
from app.services.author_utils import sanitize_author_list
from app.services.analysis.figure_analysis import analyze_figures
from app.services.analysis.llm import reset_model_usage_counters, snapshot_model_usage_counters
from app.services.analysis.reconcile import reconcile_reports
from app.services.analysis.synthesis import synthesize_report
from app.services.analysis.supp_analysis import analyze_supplements
from app.services.analysis.table_analysis import analyze_tables
from app.services.analysis.text_analysis import analyze_text
from app.services.analysis.utils import extract_expected_refs, extract_refs_from_text

SUPPLEMENT_MARKER_RE = re.compile(
    r"\b(supplement(?:ary|al)?|suppl|appendix|extended data|supporting (?:information|info|data))\b",
    re.IGNORECASE,
)


def run_full_analysis(
    session: Session,
    document_id: int,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    _analysis_trace("run_full_analysis:start")
    reset_model_usage_counters()
    analysis_started_at = monotonic()
    stage_timings: dict[str, float] = {}
    document = session.get(Document, document_id)
    document_source_url = str(document.source_url or "").strip() if document else ""

    assets = session.exec(select(Asset).where(Asset.document_id == document_id)).all()
    asset_kind = {asset.id: asset.kind for asset in assets}

    chunks = session.exec(select(Chunk).where(Chunk.document_id == document_id)).all()
    main_chunks = [chunk for chunk in chunks if asset_kind.get(chunk.asset_id) == "main"]
    supp_chunks = [chunk for chunk in chunks if asset_kind.get(chunk.asset_id) == "supp"]

    text_chunks = [c for c in main_chunks if c.modality == "text"]
    table_chunks = [c for c in main_chunks if c.modality == "table"]
    figure_chunks = [c for c in main_chunks if c.modality == "figure"]
    meta_chunks = [c for c in main_chunks if c.modality == "meta"]

    supp_text = [c for c in supp_chunks if c.modality in {"text", "table", "figure"}]
    supplement_proxy_from_main = [
        c
        for c in main_chunks
        if c.modality in {"text", "table", "figure"} and _looks_like_supplement_chunk(c)
    ]
    supp_analysis_chunks = _dedupe_chunks_by_id(supp_text)
    supp_text_chunks = [c for c in supp_chunks if c.modality == "text"]
    supp_table_chunks = [c for c in supp_chunks if c.modality == "table"]
    supp_figure_chunks = [c for c in supp_chunks if c.modality == "figure"]
    main_expected_text_chunks = [c for c in text_chunks if not _looks_like_supplement_chunk(c)]
    supp_expected_text_chunks = _dedupe_chunks_by_id(
        supp_text_chunks + [c for c in supplement_proxy_from_main if c.modality == "text"]
    )
    stage_usage_samples: dict[str, dict[str, Any]] = {}
    stage_fallback_reasons: dict[str, str] = {}

    _emit_progress(progress_callback, 0.56, "Analyzing text modality")
    _analysis_trace("run_full_analysis:text:start")
    stage_started = monotonic()
    text_inputs = [_chunk_to_dict(c, asset_kind, document_source_url) for c in text_chunks]
    text_report, text_usage, text_fallback = _analyze_text_guarded(text_inputs)
    stage_usage_samples["text"] = text_usage
    if text_fallback:
        stage_fallback_reasons["text"] = text_fallback
    stage_timings["text"] = monotonic() - stage_started
    _analysis_trace(f"run_full_analysis:text:done packets={len(text_report.get('evidence_packets', []))}")
    _emit_progress(progress_callback, 0.64, "Analyzing table modality")
    _analysis_trace("run_full_analysis:table:start")
    stage_started = monotonic()
    table_inputs = [_chunk_to_dict(c, asset_kind, document_source_url) for c in table_chunks]
    table_report, table_usage, table_fallback = _analyze_tables_guarded(table_inputs)
    stage_usage_samples["table"] = table_usage
    if table_fallback:
        stage_fallback_reasons["table"] = table_fallback
    stage_timings["table"] = monotonic() - stage_started
    _analysis_trace(f"run_full_analysis:table:done packets={len(table_report.get('evidence_packets', []))}")
    _emit_progress(progress_callback, 0.72, "Analyzing figure modality")
    _analysis_trace("run_full_analysis:figure:start")
    stage_started = monotonic()
    figure_inputs = [_chunk_to_dict(c, asset_kind, document_source_url) for c in figure_chunks]
    figure_report, figure_usage, figure_fallback = _analyze_figures_guarded(figure_inputs)
    stage_usage_samples["figure"] = figure_usage
    if figure_fallback:
        stage_fallback_reasons["figure"] = figure_fallback
    stage_timings["figure"] = monotonic() - stage_started
    _analysis_trace(f"run_full_analysis:figure:done packets={len(figure_report.get('evidence_packets', []))}")
    _emit_progress(progress_callback, 0.78, "Analyzing supplements")
    _analysis_trace("run_full_analysis:supplement:start")
    stage_started = monotonic()
    supp_inputs = [_chunk_to_dict(c, asset_kind, document_source_url) for c in supp_analysis_chunks]
    supp_report, supp_usage, supp_fallback = _analyze_supplements_guarded(supp_inputs)
    stage_usage_samples["supplement"] = supp_usage
    if supp_fallback:
        stage_fallback_reasons["supplement"] = supp_fallback
    stage_timings["supplement"] = monotonic() - stage_started
    _analysis_trace(f"run_full_analysis:supplement:done packets={len(supp_report.get('evidence_packets', []))}")

    _emit_progress(progress_callback, 0.84, "Reconciling cross-modal evidence")
    _analysis_trace("run_full_analysis:reconcile:start")
    stage_started = monotonic()
    reconcile_report = reconcile_reports(text_report, table_report, figure_report, supp_report)
    stage_timings["reconcile"] = monotonic() - stage_started
    _analysis_trace("run_full_analysis:reconcile:done")
    coverage = _compute_coverage(
        text_chunks=main_expected_text_chunks,
        table_chunks=table_chunks,
        figure_chunks=figure_chunks,
        supp_expected_text_chunks=supp_expected_text_chunks,
        supp_table_chunks=supp_table_chunks,
        supp_figure_chunks=supp_figure_chunks,
    )
    paper_meta = _extract_meta(meta_chunks)
    _emit_progress(progress_callback, 0.9, "Synthesizing executive report")
    _analysis_trace("run_full_analysis:synthesis:start")
    stage_started = monotonic()

    def _synthesis_progress(local_progress: float, local_message: str) -> None:
        bounded_local = max(0.0, min(float(local_progress or 0.0), 1.0))
        absolute_progress = 0.90 + (0.05 * bounded_local)
        _emit_progress(
            progress_callback,
            absolute_progress,
            str(local_message or "Synthesizing executive report"),
        )

    summary = synthesize_report(
        text_report,
        table_report,
        figure_report,
        supp_report,
        reconcile_report,
        paper_meta=paper_meta,
        coverage=coverage,
        text_chunk_records=[_chunk_to_dict(c, asset_kind, document_source_url) for c in text_chunks],
        progress_callback=_synthesis_progress,
    )
    stage_timings["synthesis"] = monotonic() - stage_started
    _analysis_trace("run_full_analysis:synthesis:done")

    _emit_progress(progress_callback, 0.95, "Saving findings and report")
    _analysis_trace("run_full_analysis:store:start")
    stage_started = monotonic()
    _clear_existing(session, document_id)
    _store_findings(session, document_id, text_report)
    _store_findings(session, document_id, table_report)
    _store_findings(session, document_id, figure_report)
    _store_findings(session, document_id, supp_report)
    _store_discrepancies(session, document_id, reconcile_report)
    _store_report(session, document_id, summary)

    session.commit()
    stage_timings["store"] = monotonic() - stage_started
    stage_timings["analysis_total_seconds"] = monotonic() - analysis_started_at
    _analysis_trace("run_full_analysis:done")
    model_usage = _merge_usage_counts(list(stage_usage_samples.values()))
    section_diagnostics = summary.get("section_diagnostics", {})
    section_confidence_distribution = _aggregate_section_confidence_distribution(summary.get("sections"))
    fallback_counts_by_reason = _collect_fallback_reason_counts(
        section_diagnostics=section_diagnostics if isinstance(section_diagnostics, dict) else {},
        stage_fallbacks=stage_fallback_reasons,
    )
    figure_diag = figure_report.get("diagnostics", {}) if isinstance(figure_report, dict) else {}
    supp_diag = supp_report.get("diagnostics", {}) if isinstance(supp_report, dict) else {}
    ocr_calls = 0
    if isinstance(figure_diag, dict):
        ocr_calls += int(figure_diag.get("ocr_fallback_calls", 0) or 0)
    if isinstance(supp_diag, dict):
        ocr_calls += int(supp_diag.get("ocr_fallback_calls", 0) or 0)
    analysis_timing = {
        "text": round(stage_timings.get("text", 0.0), 4),
        "table": round(stage_timings.get("table", 0.0), 4),
        "figure": round(stage_timings.get("figure", 0.0), 4),
        "supplement": round(stage_timings.get("supplement", 0.0), 4),
        "reconcile": round(stage_timings.get("reconcile", 0.0), 4),
        "synthesis": round(stage_timings.get("synthesis", 0.0), 4),
        "store": round(stage_timings.get("store", 0.0), 4),
        "analysis_total_seconds": round(stage_timings.get("analysis_total_seconds", 0.0), 4),
    }
    return {
        "analysis_timing": analysis_timing,
        "coverage": coverage,
        "summary_schema_version": summary.get("schema_version", 1),
        "sectioned_report_version": summary.get("sectioned_report_version", 0),
        "section_diagnostics": summary.get("section_diagnostics", {}),
        "section_packet_counts": summary.get("section_diagnostics", {}).get("section_packet_counts", {}),
        "section_conflict_count": summary.get("section_diagnostics", {}).get("section_conflict_count", 0),
        "unknown_section_count": summary.get("section_diagnostics", {}).get("unknown_section_count", 0),
        "slot_fill_rates": summary.get("section_diagnostics", {}).get("slot_fill_rates", {}),
        "cross_section_rejections": summary.get("section_diagnostics", {}).get("cross_section_rejections", 0),
        "sections_fallback_used": bool(summary.get("sections_fallback_used", False)),
        "sections_fallback_notes": summary.get("sections_fallback_notes", []),
        "section_confidence_distribution": section_confidence_distribution,
        "fallback_counts_by_reason": fallback_counts_by_reason,
        "text_llm_calls": int(model_usage.get("text_calls", 0)),
        "deep_vision_calls": int(
            float(model_usage.get("deep_calls", 0) or 0) + float(model_usage.get("vision_calls", 0) or 0)
        ),
        "ocr_calls": ocr_calls,
        "metadata_diagnostics": {
            "authors_extracted_count": int(paper_meta.get("authors_extracted_count", 0) or 0),
            "authors_display_count": int(paper_meta.get("authors_display_count", 0) or 0),
            "metadata_source": str(paper_meta.get("metadata_source", "")),
        },
        "modality_packet_counts": {
            "text": len(text_report.get("evidence_packets", [])),
            "table": len(table_report.get("evidence_packets", [])),
            "figure": len(figure_report.get("evidence_packets", [])),
            "supplement": len(supp_report.get("evidence_packets", [])),
        },
        "model_usage": model_usage,
        "vision_input_diagnostics": {
            "figure": figure_report.get("diagnostics", {}),
            "supplement": supp_report.get("diagnostics", {}),
        },
        "stage_model_usage": stage_usage_samples,
        "reconcile": reconcile_report.get("stats", {}),
    }


def _analysis_trace(step: str) -> None:
    trace_path = str(os.getenv("ANALYSIS_TRACE_FILE", "")).strip()
    if not trace_path:
        return
    try:
        with open(trace_path, "a", encoding="utf-8") as handle:
            handle.write(f"{step}\n")
    except Exception:
        return


def _snapshot_counter_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    def _read_int(payload: dict[str, Any], key: str) -> int:
        try:
            return int(payload.get(key, 0))
        except Exception:
            return 0

    def _read_float(payload: dict[str, Any], key: str) -> float:
        try:
            return float(payload.get(key, 0.0))
        except Exception:
            return 0.0

    text_calls = max(0, _read_int(after, "text_calls") - _read_int(before, "text_calls"))
    deep_calls = max(0, _read_int(after, "deep_calls") - _read_int(before, "deep_calls"))
    vision_calls = max(0, _read_int(after, "vision_calls") - _read_int(before, "vision_calls"))
    text_seconds = max(0.0, _read_float(after, "text_total_seconds") - _read_float(before, "text_total_seconds"))
    deep_seconds = max(0.0, _read_float(after, "deep_total_seconds") - _read_float(before, "deep_total_seconds"))
    vision_seconds = max(0.0, _read_float(after, "vision_total_seconds") - _read_float(before, "vision_total_seconds"))
    return {
        "text_calls": text_calls,
        "text_errors": max(0, _read_int(after, "text_errors") - _read_int(before, "text_errors")),
        "text_total_seconds": round(text_seconds, 4),
        "text_avg_seconds": round((text_seconds / text_calls), 4) if text_calls else 0.0,
        "deep_calls": deep_calls,
        "deep_errors": max(0, _read_int(after, "deep_errors") - _read_int(before, "deep_errors")),
        "deep_total_seconds": round(deep_seconds, 4),
        "deep_avg_seconds": round((deep_seconds / deep_calls), 4) if deep_calls else 0.0,
        "vision_calls": vision_calls,
        "vision_errors": max(0, _read_int(after, "vision_errors") - _read_int(before, "vision_errors")),
        "vision_total_seconds": round(vision_seconds, 4),
        "vision_avg_seconds": round((vision_seconds / vision_calls), 4) if vision_calls else 0.0,
    }


def _merge_usage_counts(usages: list[dict[str, Any]]) -> dict[str, Any]:
    total = {
        "text_calls": 0,
        "text_errors": 0,
        "text_total_seconds": 0.0,
        "text_avg_seconds": 0.0,
        "deep_calls": 0,
        "deep_errors": 0,
        "deep_total_seconds": 0.0,
        "deep_avg_seconds": 0.0,
        "vision_calls": 0,
        "vision_errors": 0,
        "vision_total_seconds": 0.0,
        "vision_avg_seconds": 0.0,
    }
    for usage in usages:
        if not isinstance(usage, dict):
            continue
        for key in total:
            current = usage.get(key, 0)
            try:
                total[key] = float(total[key]) + float(current)
            except Exception:
                total[key] = float(total[key])
    for key in ("text_total_seconds", "deep_total_seconds", "vision_total_seconds"):
        total[key] = round(float(total[key]), 4)
    text_calls = float(total.get("text_calls", 0) or 0)
    deep_calls = float(total.get("deep_calls", 0) or 0)
    vision_calls = float(total.get("vision_calls", 0) or 0)
    total["text_avg_seconds"] = round(float(total["text_total_seconds"]) / text_calls, 4) if text_calls else 0.0
    total["deep_avg_seconds"] = round(float(total["deep_total_seconds"]) / deep_calls, 4) if deep_calls else 0.0
    total["vision_avg_seconds"] = round(float(total["vision_total_seconds"]) / vision_calls, 4) if vision_calls else 0.0
    return total


def _aggregate_section_confidence_distribution(sections_payload: Any) -> dict[str, Any]:
    if not isinstance(sections_payload, dict):
        return {"total_items": 0, "bucket_counts": {"high": 0, "medium": 0, "low": 0, "unknown": 0}}
    bucket_counts = {"high": 0, "medium": 0, "low": 0, "unknown": 0}
    for key in ("introduction", "methods", "results", "discussion", "conclusion"):
        section_block = sections_payload.get(key, {})
        if not isinstance(section_block, dict):
            continue
        for item in section_block.get("items", []) if isinstance(section_block.get("items", []), list) else []:
            if not isinstance(item, dict):
                continue
            try:
                value = float(item.get("section_confidence", 0.0))
            except Exception:
                value = 0.0
            if value >= 0.75:
                bucket_counts["high"] += 1
            elif value >= 0.55:
                bucket_counts["medium"] += 1
            elif value > 0.0:
                bucket_counts["low"] += 1
            else:
                bucket_counts["unknown"] += 1
    total = sum(bucket_counts.values())
    buckets = {
        name: {
            "count": count,
            "share": round((count / total), 3) if total else 0.0,
        }
        for name, count in bucket_counts.items()
    }
    return {"total_items": total, "bucket_counts": buckets}


def _collect_fallback_reason_counts(
    section_diagnostics: dict[str, Any] | None,
    stage_fallbacks: dict[str, str] | None,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    if isinstance(section_diagnostics, dict):
        for note in section_diagnostics.get("fallback_notes", []) or []:
            if not isinstance(note, str):
                continue
            parts = note.split(":", 1)
            if len(parts) != 2:
                continue
            reason = str(parts[1]).strip().lower().replace(" ", "_")
            counts[reason] = counts.get(reason, 0) + 1
            section = str(parts[0]).strip().lower().replace(" ", "_")
            key = f"{section}_fallback"
            counts[key] = counts.get(key, 0) + 1
    if isinstance(stage_fallbacks, dict):
        for stage, reason in stage_fallbacks.items():
            if not reason:
                continue
            key = f"{str(stage)}_{str(reason)}"
            normalized_key = str(key).strip().replace(" ", "_")
            counts[normalized_key] = counts.get(normalized_key, 0) + 1
    return counts


def _analysis_worker(kind: str, chunks: list[dict[str, Any]], out_queue: Any) -> None:
    try:
        kind_key = str(kind or "").strip().lower()
        if kind_key == "text":
            report = analyze_text(chunks)
        elif kind_key == "table":
            report = analyze_tables(chunks)
        elif kind_key == "figure":
            report = analyze_figures(chunks)
        elif kind_key == "supplement":
            report = analyze_supplements(chunks)
        else:
            raise RuntimeError(f"unsupported analysis modality: {kind}")
        out_queue.put({"ok": True, "report": report, "model_usage": snapshot_model_usage_counters()})
    except Exception as exc:
        out_queue.put({"ok": False, "error": str(exc), "model_usage": snapshot_model_usage_counters()})


def _empty_usage_payload() -> dict[str, Any]:
    return {
        "text_calls": 0,
        "text_errors": 0,
        "text_total_seconds": 0.0,
        "text_avg_seconds": 0.0,
        "deep_calls": 0,
        "deep_errors": 0,
        "deep_total_seconds": 0.0,
        "deep_avg_seconds": 0.0,
        "vision_calls": 0,
        "vision_errors": 0,
        "vision_total_seconds": 0.0,
        "vision_avg_seconds": 0.0,
    }


def _run_analysis_sync(
    analyze_fn,
    chunks: list[dict[str, Any]],
    **kwargs: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    before = snapshot_model_usage_counters()
    if kwargs:
        report = analyze_fn(chunks, **kwargs)
    else:
        report = analyze_fn(chunks)
    after = snapshot_model_usage_counters()
    return report, _snapshot_counter_delta(before, after)


def _dedupe_analysis_notes(notes: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in notes:
        note = " ".join(str(raw or "").split()).strip()
        if not note:
            continue
        key = note.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(note)
    return out


def _run_analysis_subprocess(
    kind: str,
    chunks: list[dict[str, Any]],
    *,
    timeout_seconds: int,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    context = mp.get_context("spawn")
    out_queue = context.Queue(maxsize=1)
    proc = context.Process(target=_analysis_worker, args=(kind, chunks, out_queue))
    proc.start()
    proc.join(timeout_seconds)

    timed_out = proc.is_alive()
    if timed_out:
        proc.terminate()
        proc.join(5)

    payload: dict[str, Any] = {}
    if not timed_out and int(proc.exitcode or 0) == 0:
        try:
            maybe_payload = out_queue.get_nowait()
            if isinstance(maybe_payload, dict):
                payload = maybe_payload
        except QueueEmpty:
            payload = {}
        except Exception:
            payload = {}

    usage = payload.get("model_usage", {})
    if not isinstance(usage, dict):
        usage = {}

    if payload.get("ok") and isinstance(payload.get("report"), dict):
        return payload["report"], usage, ""

    fail_reason = "timeout" if timed_out else f"exitcode={int(proc.exitcode or 0)}"
    if payload.get("error"):
        fail_reason = f"{fail_reason}; {payload.get('error')}"
    return {}, usage, fail_reason


def _empty_modality_report(kind: str, fail_reason: str) -> dict[str, Any]:
    label = str(kind or "modality").strip().lower()
    note = f"{label.capitalize()} analysis failed in isolated subprocess; fallback used ({fail_reason})."
    base: dict[str, Any] = {
        "findings": [],
        "results": [],
        "evidence_packets": [],
        "analysis_notes": [note],
    }
    if label in {"figure", "supplement"}:
        base["diagnostics"] = {"guarded_fallback": True, "reason": fail_reason}
    return base


def _analyze_text_guarded(chunks: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], str]:
    if not settings.analysis_text_llm_enabled or not settings.analysis_text_subprocess_guard_enabled:
        report, usage = _run_analysis_sync(analyze_text, chunks)
        return report, usage, ""

    timeout_seconds = max(15, int(settings.analysis_text_subprocess_timeout_sec or 0))
    report, usage, fail_reason = _run_analysis_subprocess("text", chunks, timeout_seconds=timeout_seconds)
    if report:
        return report, usage, ""

    retry_usage: dict[str, Any] = {}
    retry_fail_reason = ""
    if "timeout" in str(fail_reason or "").lower():
        retry_timeout_seconds = max(timeout_seconds + 180, timeout_seconds * 2)
        retry_report, retry_usage, retry_fail_reason = _run_analysis_subprocess(
            "text",
            chunks,
            timeout_seconds=retry_timeout_seconds,
        )
        if retry_report:
            retry_notes = list(retry_report.get("analysis_notes", []))
            retry_notes.append(
                "Text LLM stage timed out once in isolated subprocess; recovered on extended-timeout retry."
            )
            retry_report["analysis_notes"] = _dedupe_analysis_notes(retry_notes)
            return retry_report, _merge_usage_counts([usage, retry_usage]), ""

    combined_fail_reason = str(fail_reason or "").strip()
    retry_fail = str(retry_fail_reason or "").strip()
    if retry_fail and retry_fail != combined_fail_reason:
        combined_fail_reason = (
            f"{combined_fail_reason}; retry={retry_fail}" if combined_fail_reason else f"retry={retry_fail}"
        )

    fallback_report, fallback_usage = _run_analysis_sync(analyze_text, chunks, force_llm_enabled=False)
    notes = list(fallback_report.get("analysis_notes", []))
    notes.append(
        "Text LLM stage failed in isolated subprocess; "
        f"used deterministic section extraction fallback ({combined_fail_reason or 'unknown failure'})."
    )
    fallback_report["analysis_notes"] = _dedupe_analysis_notes(notes)
    return (
        fallback_report,
        _merge_usage_counts([usage, retry_usage, fallback_usage]),
        f"subprocess_fallback:{combined_fail_reason or 'unknown failure'}",
    )


def _analyze_tables_guarded(chunks: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], str]:
    if not chunks:
        return {"findings": [], "results": [], "evidence_packets": []}, _empty_usage_payload(), ""
    if not settings.analysis_nontext_llm_enabled:
        return _empty_modality_report("table", "non-text llm disabled"), _empty_usage_payload(), ""
    if not settings.analysis_modality_subprocess_guard_enabled:
        report, usage = _run_analysis_sync(analyze_tables, chunks)
        return report, usage, ""
    timeout_seconds = max(15, int(settings.analysis_modality_subprocess_timeout_sec or 0))
    report, usage, fail_reason = _run_analysis_subprocess("table", chunks, timeout_seconds=timeout_seconds)
    if report:
        return report, usage, ""
    return _empty_modality_report("table", fail_reason), usage, fail_reason


def _analyze_figures_guarded(chunks: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], str]:
    if not chunks:
        return {
            "findings": [],
            "results": [],
            "evidence_packets": [],
            "diagnostics": {},
        }, _empty_usage_payload(), ""
    if not settings.analysis_nontext_llm_enabled:
        return _empty_modality_report("figure", "non-text llm disabled"), _empty_usage_payload(), ""
    if not settings.analysis_modality_subprocess_guard_enabled:
        report, usage = _run_analysis_sync(analyze_figures, chunks)
        return report, usage, ""
    timeout_seconds = max(15, int(settings.analysis_modality_subprocess_timeout_sec or 0))
    report, usage, fail_reason = _run_analysis_subprocess("figure", chunks, timeout_seconds=timeout_seconds)
    if report:
        return report, usage, ""
    return _empty_modality_report("figure", fail_reason), usage, fail_reason


def _analyze_supplements_guarded(chunks: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], str]:
    if not chunks:
        return {
            "findings": [],
            "results": [],
            "evidence_packets": [],
            "diagnostics": {},
        }, _empty_usage_payload(), ""
    if not settings.analysis_nontext_llm_enabled:
        return _empty_modality_report("supplement", "non-text llm disabled"), _empty_usage_payload(), ""
    if not settings.analysis_modality_subprocess_guard_enabled:
        report, usage = _run_analysis_sync(analyze_supplements, chunks)
        return report, usage, ""
    timeout_seconds = max(15, int(settings.analysis_modality_subprocess_timeout_sec or 0))
    report, usage, fail_reason = _run_analysis_subprocess("supplement", chunks, timeout_seconds=timeout_seconds)
    if report:
        return report, usage, ""
    return _empty_modality_report("supplement", fail_reason), usage, fail_reason


def _emit_progress(progress_callback: Callable[[float, str], None] | None, progress: float, message: str) -> None:
    if progress_callback is None:
        return
    try:
        progress_callback(progress, message)
    except Exception:
        return


def _chunk_to_dict(chunk: Chunk, asset_kind: dict[int, str], document_source_url: str = "") -> dict[str, Any]:
    return {
        "anchor": chunk.anchor,
        "content": chunk.content,
        "meta": chunk.meta,
        "modality": chunk.modality,
        "asset_kind": asset_kind.get(chunk.asset_id or -1, "main"),
        "document_source_url": document_source_url,
    }


def _extract_meta(meta_chunks: list[Chunk]) -> dict:
    for chunk in meta_chunks:
        try:
            data = json.loads(chunk.content)
            if isinstance(data, dict):
                return _sanitize_meta(data)
        except Exception:
            continue
    return {}


def _sanitize_meta(meta: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(meta, dict):
        return {}
    out = dict(meta)
    out["metadata_source"] = str(out.get("metadata_source") or "meta_chunk")
    authors_raw = out.get("authors")
    if isinstance(authors_raw, list):
        display, extracted_count = sanitize_author_list(authors_raw, max_items=24)
        out["authors"] = display
        out["authors_extracted_count"] = extracted_count
        out["authors_display_count"] = len(display)
    return out


def _clear_existing(session: Session, document_id: int) -> None:
    session.exec(delete(Finding).where(Finding.document_id == document_id))
    session.exec(delete(Discrepancy).where(Discrepancy.document_id == document_id))
    session.exec(delete(Report).where(Report.document_id == document_id))


def _store_findings(session: Session, document_id: int, report: dict) -> None:
    findings = []
    if isinstance(report, dict):
        findings = report.get("findings", [])
    elif isinstance(report, list):
        findings = report
    for finding in findings:
        if not isinstance(finding, dict):
            continue
        session.add(
            Finding(
                document_id=document_id,
                category=finding.get("category", "other"),
                summary=finding.get("summary", ""),
                evidence=json.dumps(finding.get("evidence", [])),
                confidence=float(finding.get("confidence", 0.0)),
            )
        )


def _store_discrepancies(session: Session, document_id: int, report: dict) -> None:
    for item in report.get("discrepancies", []):
        evidence_payload = {
            "evidence": item.get("evidence", []),
            "reason": item.get("reason", "unsupported"),
            "linked_packet_ids": item.get("linked_packet_ids", []),
        }
        session.add(
            Discrepancy(
                document_id=document_id,
                claim=item.get("claim", ""),
                evidence=json.dumps(evidence_payload),
                severity=item.get("severity", "medium"),
                confidence=float(item.get("confidence", 0.0)),
            )
        )


def _store_report(session: Session, document_id: int, summary: dict) -> None:
    session.add(
        Report(
            document_id=document_id,
            payload=json.dumps(summary),
        )
    )


def _compute_coverage(
    text_chunks: list[Chunk],
    table_chunks: list[Chunk],
    figure_chunks: list[Chunk],
    supp_expected_text_chunks: list[Chunk],
    supp_table_chunks: list[Chunk],
    supp_figure_chunks: list[Chunk],
) -> dict:
    main_expected = extract_expected_refs([c.content for c in text_chunks])
    supp_expected = extract_expected_refs([c.content for c in supp_expected_text_chunks])

    extracted_fig_refs = _extract_refs_from_meta(figure_chunks)
    extracted_table_refs = _extract_refs_from_meta(table_chunks)
    supp_extracted_fig_refs = _extract_refs_from_meta(supp_figure_chunks)
    supp_extracted_table_refs = _extract_refs_from_meta(supp_table_chunks)

    return {
        "figures": _coverage_block(main_expected["figure_refs"], extracted_fig_refs, len(figure_chunks)),
        "tables": _coverage_block(main_expected["table_refs"], extracted_table_refs, len(table_chunks)),
        "supp_figures": _coverage_block(
            supp_expected["figure_refs"], supp_extracted_fig_refs, len(supp_figure_chunks)
        ),
        "supp_tables": _coverage_block(
            supp_expected["table_refs"], supp_extracted_table_refs, len(supp_table_chunks)
        ),
    }


def _coverage_block(expected_refs: list[str], extracted_refs: set[str], extracted_count: int) -> dict:
    expected_set = set(expected_refs)
    missing = sorted(expected_set - extracted_refs)
    return {
        "expected": len(expected_set),
        "expected_refs": sorted(expected_set),
        "extracted": extracted_count,
        "extracted_refs": sorted(extracted_refs),
        "missing_refs": missing,
    }


def _extract_refs_from_meta(chunks: list[Chunk]) -> set[str]:
    refs: set[str] = set()
    for chunk in chunks:
        text_parts = []
        if chunk.meta:
            try:
                meta = json.loads(chunk.meta)
                caption = meta.get("caption")
                ocr = meta.get("ocr_text")
                if caption:
                    text_parts.append(str(caption))
                if ocr:
                    text_parts.append(str(ocr))
            except Exception:
                pass
        if chunk.content:
            text_parts.append(chunk.content)
        if text_parts:
            refs |= extract_refs_from_text(" ".join(text_parts))
    return refs


def _dedupe_chunks_by_id(chunks: list[Chunk]) -> list[Chunk]:
    out: list[Chunk] = []
    seen: set[int] = set()
    for chunk in chunks:
        if chunk.id is None:
            out.append(chunk)
            continue
        if chunk.id in seen:
            continue
        seen.add(chunk.id)
        out.append(chunk)
    return out


def _looks_like_supplement_chunk(chunk: Chunk) -> bool:
    anchor = str(chunk.anchor or "").lower()
    if SUPPLEMENT_MARKER_RE.search(anchor):
        return True
    if chunk.meta:
        try:
            meta_obj = json.loads(chunk.meta)
        except Exception:
            meta_obj = {}
        if isinstance(meta_obj, dict):
            marker_blob = " ".join(
                [
                    str(meta_obj.get("caption") or ""),
                    str(meta_obj.get("source_url") or ""),
                    str(meta_obj.get("source_page_url") or ""),
                    str(meta_obj.get("figure_id") or ""),
                    str(meta_obj.get("table_id") or ""),
                    str(meta_obj.get("source") or ""),
                ]
            ).lower()
            if (
                SUPPLEMENT_MARKER_RE.search(marker_blob)
                or "/doi/suppl/" in marker_blob
                or "suppl_file" in marker_blob
            ):
                return True
    content = str(chunk.content or "").lower()
    if SUPPLEMENT_MARKER_RE.search(content):
        return True
    return False
