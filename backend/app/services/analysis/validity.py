from __future__ import annotations

from typing import Any

SECTION_KEYS = ("introduction", "methods", "results", "discussion", "conclusion")


def build_run_validity(
    *,
    summary_json: dict[str, Any] | None = None,
    diagnostics: dict[str, Any] | None = None,
    run_note: str = "",
    job_status: str | None = "completed",
    provider: str = "openai",
    require_source_provenance: bool = False,
) -> dict[str, Any]:
    summary = summary_json if isinstance(summary_json, dict) else {}
    diag = _unwrap_diagnostics(diagnostics)
    diagnostics_missing = not bool(diag)

    fallback_audit = _build_fallback_audit(summary, diag, run_note)
    quality_backend_audit = _build_quality_backend_audit(summary, diag)
    source_provenance_audit = _build_source_provenance_audit(summary)
    prompt_budget_audit = _build_prompt_budget_audit(diag)
    provider_key = _canonical(provider)
    text_cache_reused = bool(quality_backend_audit.get("text_cache_reused"))
    narrative_synthesis_missing = bool(
        quality_backend_audit.get("blockers", {}).get("narrative_synthesis_calls_zero")
    ) if isinstance(quality_backend_audit.get("blockers"), dict) else False

    status = _canonical(job_status or "completed")
    status_completed = status == "completed" or status.endswith(".completed")
    source_provenance_missing = require_source_provenance and not bool(source_provenance_audit.get("passed"))
    fallback_blocks = bool(fallback_audit.get("execution_fallback_used"))

    blockers = {
        "job_not_completed": not status_completed,
        "fallback_engaged": fallback_blocks,
        "text_analysis_cache_reused": text_cache_reused,
        "text_llm_calls_zero": (
            bool(quality_backend_audit.get("blockers", {}).get("text_calls_zero")) and not text_cache_reused
        )
        if isinstance(quality_backend_audit.get("blockers"), dict)
        else False,
        "section_extraction_disabled": bool(
            quality_backend_audit.get("blockers", {}).get("section_extraction_disabled")
        )
        if isinstance(quality_backend_audit.get("blockers"), dict)
        else False,
        "narrative_synthesis_calls_zero": narrative_synthesis_missing,
        "diagnostics_missing": diagnostics_missing,
        "source_provenance_missing": source_provenance_missing,
    }
    provider_failure = _provider_failure(summary, diag)
    budget_blocked = _budget_blocked(summary, diag, run_note)
    if provider_key == "openai":
        blockers["provider_failure"] = provider_failure
        blockers["budget_blocked"] = budget_blocked

    reasons = [key for key, value in blockers.items() if value]
    valid = not reasons
    fallback_reasons = _fallback_reasons(fallback_audit)
    warning_reasons = _warning_reasons(prompt_budget_audit, fallback_audit=fallback_audit, provider=provider_key)
    return {
        "run_validity": "valid" if valid else "invalid",
        "valid": valid,
        "benchmark_valid": valid,
        "failure_type": None if valid else "infrastructure",
        "reasons": reasons,
        "warning_reasons": warning_reasons,
        "blockers": blockers,
        "warnings": {
            "prompt_budget_pressure": not bool(prompt_budget_audit.get("passed")),
            "section_fallback_used": bool(fallback_audit.get("sections_fallback_used")),
        },
        "fallback_used": not bool(fallback_audit.get("passed")),
        "fallback_reasons": fallback_reasons,
        "fallback_sections": fallback_audit.get("fallback_sections", []),
        "provider_failure": provider_failure,
        "budget_blocked": budget_blocked,
        "rerun_required": not valid,
        "fallback_audit": fallback_audit,
        "quality_backend_audit": quality_backend_audit,
        "source_provenance_audit": source_provenance_audit,
        "prompt_budget_audit": prompt_budget_audit,
    }


def invalid_report_reason(validity: dict[str, Any] | None, *, provider: str = "openai") -> str:
    if not isinstance(validity, dict):
        return ""
    if not bool(validity.get("rerun_required")):
        return ""
    provider_key = _canonical(provider)
    reasons = set(str(reason) for reason in validity.get("reasons", []) if str(reason).strip())
    if (
        provider_key != "openai"
        and "diagnostics_missing" in reasons
        and not (reasons - {"diagnostics_missing", "text_llm_calls_zero"})
    ):
        return ""
    if provider_key == "openai" and "provider_failure" in reasons:
        return "OpenAI model calls failed during analysis."
    if provider_key == "openai" and "budget_blocked" in reasons:
        return "OpenAI usage guardrail stopped this run."
    if "job_not_completed" in reasons:
        return "Analysis did not complete, so the report validity cannot be verified."
    if "fallback_engaged" in reasons:
        if provider_key == "openai":
            return "OpenAI analysis used fallback outputs, so the report should be rerun for a verified result."
        if provider_key in {"local", "local_gpu", "llama_cpp"}:
            return (
                "Local model analysis used fallback outputs, so rerun after local model and GPU diagnostics are clean."
            )
        return "Analysis used fallback outputs, so the report should be rerun for a verified result."
    if "text_analysis_cache_reused" in reasons:
        return (
            "Cached local text-analysis output was reused; rerun with the text cache disabled or cleared "
            "for fresh local-model execution and latency validation."
        )
    if "text_llm_calls_zero" in reasons:
        return _provider_display_name(provider_key) + " text analysis did not run."
    if "section_extraction_disabled" in reasons:
        return "Section extraction was disabled, so the report should be rerun with section-level extraction enabled."
    if "narrative_synthesis_calls_zero" in reasons:
        return _provider_display_name(provider_key) + " narrative synthesis did not run."
    if "diagnostics_missing" in reasons:
        return "Analysis diagnostics are missing, so the report validity cannot be verified."
    if "source_provenance_missing" in reasons:
        return "Report source provenance is missing, so the report validity cannot be verified."
    return _provider_display_name(provider_key) + "-backed report validity could not be verified."


def _unwrap_diagnostics(diagnostics: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(diagnostics, dict):
        return {}
    nested = diagnostics.get("diagnostics")
    if isinstance(nested, dict):
        return nested
    return diagnostics


def _canonical(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "_")


def _provider_display_name(provider: str) -> str:
    normalized = _canonical(provider)
    if normalized == "openai":
        return "OpenAI"
    if normalized in {"local", "local_gpu", "llama_cpp"}:
        return "Local model"
    if not normalized:
        return "Model"
    return normalized.replace("_", " ").title()


def _numeric(value: Any) -> float:
    try:
        return float(value or 0)
    except Exception:
        return 0.0


def _contains_fallback_marker(value: Any) -> bool:
    return isinstance(value, str) and "fallback" in value.lower()


def _is_execution_fallback_reason(value: Any) -> bool:
    normalized = str(value or "").strip().lower()
    if not normalized:
        return False
    execution_markers = (
        "subprocess_fallback",
        "guarded_fallback",
        "openai_request_failed",
        "model_crashed",
        "model crashed",
        "analysis_failed",
        "analysis failed",
        "pipeline_failure",
        "timeout",
        "exitcode=",
    )
    return any(marker in normalized for marker in execution_markers)


def _sum_numeric_mapping_values(payload: Any) -> int:
    if not isinstance(payload, dict):
        return 0
    total = 0
    for value in payload.values():
        try:
            total += int(value or 0)
        except Exception:
            continue
    return total


def _build_fallback_audit(summary: dict[str, Any], diagnostics: dict[str, Any], run_note: str) -> dict[str, Any]:
    section_diagnostics = diagnostics.get("section_diagnostics", {})
    if not isinstance(section_diagnostics, dict):
        section_diagnostics = summary.get("section_diagnostics", {})
    if not isinstance(section_diagnostics, dict):
        section_diagnostics = {}

    fallback_counts = diagnostics.get("fallback_counts_by_reason", {})
    fallback_count_total = _sum_numeric_mapping_values(fallback_counts)
    fallback_notes = diagnostics.get("sections_fallback_notes", [])
    if not fallback_notes:
        fallback_notes = summary.get("sections_fallback_notes", [])
    if not isinstance(fallback_notes, list):
        fallback_notes = []

    fallback_sections: list[str] = []
    for section in SECTION_KEYS:
        section_payload = section_diagnostics.get(section, {})
        if isinstance(section_payload, dict) and bool(section_payload.get("fallback_used")):
            fallback_sections.append(section)

    pipeline_failure = diagnostics.get("pipeline_failure", {})
    pipeline_reason = pipeline_failure.get("reason", "") if isinstance(pipeline_failure, dict) else ""
    execution_note = " ".join(str(part or "") for part in (run_note, pipeline_reason))
    execution_fallback_used = _contains_fallback_marker(execution_note) or _is_execution_fallback_reason(execution_note)
    if isinstance(fallback_counts, dict):
        for reason, count in fallback_counts.items():
            try:
                count_int = int(count or 0)
            except Exception:
                count_int = 0
            if count_int > 0 and _is_execution_fallback_reason(reason):
                execution_fallback_used = True
                break
    sections_fallback_used = bool(
        diagnostics.get("sections_fallback_used")
        or summary.get("sections_fallback_used")
        or fallback_notes
        or fallback_sections
        or fallback_count_total
    )

    return {
        "passed": not execution_fallback_used and not sections_fallback_used,
        "execution_fallback_used": execution_fallback_used,
        "sections_fallback_used": sections_fallback_used,
        "fallback_sections": fallback_sections,
        "fallback_count_total": fallback_count_total,
        "fallback_counts_by_reason": fallback_counts if isinstance(fallback_counts, dict) else {},
        "sections_fallback_notes": fallback_notes,
        "run_note": run_note,
    }


def _read_nested_int(payload: dict[str, Any], path: list[str]) -> int:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return 0
        current = current.get(key)
    try:
        return int(current or 0)
    except Exception:
        return 0


def _build_quality_backend_audit(summary: dict[str, Any], diagnostics: dict[str, Any]) -> dict[str, Any]:
    model_usage = diagnostics.get("model_usage", {})
    if not isinstance(model_usage, dict):
        model_usage = summary.get("model_usage", {})
    if not isinstance(model_usage, dict):
        model_usage = {}

    text_calls = int(model_usage.get("text_calls", 0) or 0)
    if text_calls <= 0:
        text_calls = _read_nested_int(diagnostics, ["llm_status", "text_llm_calls"])

    section_diagnostics = diagnostics.get("section_diagnostics", {})
    if not isinstance(section_diagnostics, dict):
        section_diagnostics = summary.get("section_diagnostics", {})
    if not isinstance(section_diagnostics, dict):
        section_diagnostics = {}

    section_extraction_enabled = section_diagnostics.get("section_extraction_enabled")
    if section_extraction_enabled is None:
        section_extraction_enabled = summary.get("section_extraction_enabled")
    section_extraction_counts = section_diagnostics.get("section_extraction_counts", {})
    if not isinstance(section_extraction_counts, dict):
        section_extraction_counts = {}

    text_cache_reused = _text_analysis_cache_reused(summary, diagnostics)
    synthesis_audit = _synthesis_model_audit(summary, diagnostics)

    blockers = {
        "text_calls_zero": text_calls <= 0 and not text_cache_reused,
        "section_extraction_disabled": section_extraction_enabled is False,
        "narrative_synthesis_calls_zero": bool(synthesis_audit.get("narrative_synthesis_required"))
        and int(synthesis_audit.get("narrative_synthesis_deep_calls", 0) or 0) <= 0,
    }
    return {
        "passed": not any(bool(value) for value in blockers.values()),
        "text_calls": text_calls,
        "text_cache_reused": text_cache_reused,
        **synthesis_audit,
        "section_extraction_enabled": section_extraction_enabled,
        "section_extraction_counts": section_extraction_counts,
        "blockers": blockers,
    }


def _synthesis_model_audit(summary: dict[str, Any], diagnostics: dict[str, Any]) -> dict[str, Any]:
    synthesis_diag = diagnostics.get("synthesis_diagnostics", {})
    if not isinstance(synthesis_diag, dict):
        synthesis_diag = summary.get("synthesis_diagnostics", {})
    if not isinstance(synthesis_diag, dict):
        synthesis_diag = {}

    narrative_enabled = synthesis_diag.get("narrative_overrides_enabled")
    narrative_required = synthesis_diag.get("narrative_synthesis_required")
    stage_usage = diagnostics.get("stage_model_usage", {})
    synthesis_usage = stage_usage.get("synthesis", {}) if isinstance(stage_usage, dict) else {}
    if not isinstance(synthesis_usage, dict):
        synthesis_usage = {}

    deep_calls = int(_numeric(synthesis_diag.get("narrative_synthesis_deep_calls")))
    if deep_calls <= 0:
        deep_calls = int(_numeric(synthesis_usage.get("deep_calls")))
    deep_errors = int(_numeric(synthesis_diag.get("narrative_synthesis_deep_errors")))
    if deep_errors <= 0:
        deep_errors = int(_numeric(synthesis_usage.get("deep_errors")))

    required = narrative_required if isinstance(narrative_required, bool) else narrative_enabled
    return {
        "narrative_overrides_enabled": narrative_enabled if isinstance(narrative_enabled, bool) else None,
        "narrative_synthesis_required": bool(required) if isinstance(required, bool) else False,
        "narrative_synthesis_deep_calls": deep_calls,
        "narrative_synthesis_deep_errors": deep_errors,
        "narrative_synthesis_ran": deep_calls > 0,
    }


def _text_analysis_cache_reused(summary: dict[str, Any], diagnostics: dict[str, Any]) -> bool:
    for stage in diagnostics.get("analysis_timeline", []) if isinstance(diagnostics.get("analysis_timeline"), list) else []:
        if not isinstance(stage, dict):
            continue
        if _canonical(stage.get("stage")) != "text":
            continue
        metadata = stage.get("metadata")
        if isinstance(metadata, dict) and bool(metadata.get("cache_hit")):
            return True

    for source in (diagnostics.get("latency_profile"), summary.get("latency_profile")):
        if not isinstance(source, dict):
            continue
        cache_summary = source.get("cache_summary")
        if isinstance(cache_summary, dict):
            stages = cache_summary.get("cache_hit_stages")
            if isinstance(stages, list) and any(_canonical(stage) == "text" for stage in stages):
                return True
        for flag in source.get("quality_flags", []) if isinstance(source.get("quality_flags"), list) else []:
            if _canonical(flag) == "text_cache_hit":
                return True
    return False


def _has_section_source_provenance(row: dict[str, Any]) -> bool:
    for key in ("evidence_refs", "evidence", "anchors"):
        values = row.get(key)
        if isinstance(values, list) and any(str(value).strip() for value in values):
            return True
    for key in ("anchor", "source_anchor"):
        if str(row.get(key) or "").strip():
            return True
    return False


def _section_source_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows_out: list[dict[str, Any]] = []
    for source_key, text_key in (("presentation_evidence", "statement"), ("sections_compact", "statement")):
        source = summary.get(source_key, {})
        if not isinstance(source, dict):
            continue
        for section in SECTION_KEYS:
            rows = source.get(section, [])
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                if source_key == "sections_compact" and _canonical(row.get("status")) != "found":
                    continue
                if str(row.get(text_key) or "").strip():
                    rows_out.append(row)

    sections = summary.get("sections", {})
    if isinstance(sections, dict):
        for section in SECTION_KEYS:
            block = sections.get(section, {})
            if not isinstance(block, dict):
                continue
            items = block.get("items", [])
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, dict) and str(item.get("statement") or "").strip():
                    rows_out.append(item)
    return rows_out


def _build_source_provenance_audit(summary: dict[str, Any]) -> dict[str, Any]:
    rows = _section_source_rows(summary)
    missing_count = sum(1 for row in rows if not _has_section_source_provenance(row))
    return {
        "passed": bool(rows) and missing_count == 0,
        "section_source_rows": len(rows),
        "missing_provenance_rows": missing_count,
    }


def _build_prompt_budget_audit(diagnostics: dict[str, Any]) -> dict[str, Any]:
    prompt_budget = diagnostics.get("prompt_budget_diagnostics", {})
    if not isinstance(prompt_budget, dict):
        prompt_budget = {}
    totals = prompt_budget.get("totals", {})
    if not isinstance(totals, dict):
        totals = {}
    flags = [
        str(flag).strip()
        for flag in prompt_budget.get("quality_flags", [])
        if str(flag).strip()
    ] if isinstance(prompt_budget.get("quality_flags", []), list) else []
    prompt_calls = int(_numeric(totals.get("prompt_calls")))
    max_prompt_chars = int(_numeric(totals.get("max_prompt_chars")))
    max_prompt_modality = str(totals.get("max_prompt_modality") or "").strip()
    warning_reasons: list[str] = []
    if flags:
        warning_reasons.extend(flags)
    if prompt_calls >= 8 and "many_local_prompt_batches" not in warning_reasons:
        warning_reasons.append("many_local_prompt_batches")
    if max_prompt_chars >= 12000:
        warning_reasons.append(
            f"{max_prompt_modality}_large_prompt" if max_prompt_modality else "large_prompt"
        )
    warning_reasons = list(dict.fromkeys(warning_reasons))
    return {
        "passed": not warning_reasons,
        "warning_reasons": warning_reasons,
        "prompt_calls": prompt_calls,
        "max_prompt_chars": max_prompt_chars,
        "max_prompt_modality": max_prompt_modality,
    }


def _warning_reasons(
    prompt_budget_audit: dict[str, Any],
    *,
    fallback_audit: dict[str, Any] | None = None,
    provider: str = "",
) -> list[str]:
    reasons: list[str] = []
    if isinstance(prompt_budget_audit.get("warning_reasons"), list):
        reasons.extend(str(reason) for reason in prompt_budget_audit["warning_reasons"] if str(reason).strip())
    fallback_payload = fallback_audit if isinstance(fallback_audit, dict) else {}
    if provider != "openai" and bool(fallback_payload.get("sections_fallback_used")):
        reasons.append("section_recall_fallback_used")
    return list(dict.fromkeys(reasons))


def _provider_failure(summary: dict[str, Any], diagnostics: dict[str, Any]) -> bool:
    model_usage = diagnostics.get("model_usage", {})
    if not isinstance(model_usage, dict):
        model_usage = summary.get("model_usage", {})
    if isinstance(model_usage, dict):
        if sum(_numeric(model_usage.get(key)) for key in ("text_errors", "deep_errors", "vision_errors")) > 0:
            return True

    fallback_counts = diagnostics.get("fallback_counts_by_reason", {})
    if isinstance(fallback_counts, dict):
        for key in fallback_counts:
            normalized = str(key).lower()
            if "openai_request_failed" in normalized or "text_subprocess_fallback" in normalized:
                return True

    stage_usage = diagnostics.get("stage_model_usage", {})
    if isinstance(stage_usage, dict):
        for usage in stage_usage.values():
            if not isinstance(usage, dict):
                continue
            if sum(_numeric(usage.get(key)) for key in ("text_errors", "deep_errors", "vision_errors")) > 0:
                return True

    uncertainty_gaps = summary.get("uncertainty_gaps", [])
    if isinstance(uncertainty_gaps, list):
        for item in uncertainty_gaps:
            normalized = str(item).lower()
            if "openai request failed" in normalized or "incorrect api key" in normalized:
                return True
    return False


def _budget_blocked(summary: dict[str, Any], diagnostics: dict[str, Any], run_note: str) -> bool:
    text = " ".join(str(value) for value in (run_note, summary.get("report_invalid_reason", "")))
    for value in diagnostics.get("fallback_counts_by_reason", {}) if isinstance(diagnostics.get("fallback_counts_by_reason"), dict) else {}:
        text += f" {value}"
    normalized = text.lower()
    return "budget" in normalized and ("exceeded" in normalized or "guardrail" in normalized)


def _fallback_reasons(fallback_audit: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    counts = fallback_audit.get("fallback_counts_by_reason", {})
    if isinstance(counts, dict):
        reasons.extend(str(key) for key, value in counts.items() if value)
    notes = fallback_audit.get("sections_fallback_notes", [])
    if isinstance(notes, list):
        reasons.extend(str(note) for note in notes if str(note).strip())
    if fallback_audit.get("execution_fallback_used"):
        note = str(fallback_audit.get("run_note") or "").strip()
        reasons.append(note or "execution_fallback")
    return list(dict.fromkeys(reasons))
