from __future__ import annotations

from app.services.analysis.validity import build_run_validity, invalid_report_reason


def test_run_validity_marks_provider_failure_as_rerun_required() -> None:
    validity = build_run_validity(
        summary_json={},
        diagnostics={
            "model_usage": {"text_calls": 1, "text_errors": 1},
            "section_diagnostics": {"section_extraction_enabled": True},
        },
        provider="openai",
        require_source_provenance=False,
    )

    assert validity["valid"] is False
    assert validity["rerun_required"] is True
    assert "provider_failure" in validity["reasons"]
    assert invalid_report_reason(validity, provider="openai") == "OpenAI model calls failed during analysis."


def test_run_validity_benchmark_can_require_source_provenance() -> None:
    summary = {
        "presentation_evidence": {
            "methods": [{"statement": "Methods statement without evidence."}],
        }
    }

    validity = build_run_validity(
        summary_json=summary,
        diagnostics={
            "model_usage": {"text_calls": 1},
            "section_diagnostics": {"section_extraction_enabled": True},
        },
        provider="openai",
        require_source_provenance=True,
    )

    assert validity["valid"] is False
    assert "source_provenance_missing" in validity["reasons"]


def test_run_validity_warns_on_prompt_pressure_without_invalidating_local_run() -> None:
    validity = build_run_validity(
        summary_json={},
        diagnostics={
            "model_usage": {"text_calls": 1},
            "section_diagnostics": {"section_extraction_enabled": True},
            "prompt_budget_diagnostics": {
                "totals": {"prompt_calls": 8, "max_prompt_chars": 15000, "max_prompt_modality": "figure"},
                "quality_flags": ["many_local_prompt_batches", "figure_large_prompt"],
            },
        },
        provider="local",
        require_source_provenance=False,
    )

    assert validity["valid"] is True
    assert validity["rerun_required"] is False
    assert validity["warnings"]["prompt_budget_pressure"] is True
    assert "many_local_prompt_batches" in validity["warning_reasons"]
    assert "figure_large_prompt" in validity["warning_reasons"]


def test_run_validity_warns_on_local_section_fallback_without_invalidating_run() -> None:
    validity = build_run_validity(
        summary_json={},
        diagnostics={
            "model_usage": {"text_calls": 1},
            "section_diagnostics": {
                "section_extraction_enabled": True,
                "introduction": {"fallback_used": True},
                "methods": {"fallback_used": False},
            },
            "sections_fallback_notes": [
                "Introduction: Explicit introduction anchors were sparse; supplemented with early-body text."
            ],
            "fallback_counts_by_reason": {"introduction_fallback": 1},
        },
        provider="local",
        require_source_provenance=False,
    )

    assert validity["valid"] is True
    assert validity["rerun_required"] is False
    assert validity["fallback_used"] is True
    assert validity["warnings"]["section_fallback_used"] is True
    assert "section_recall_fallback_used" in validity["warning_reasons"]
    assert "fallback_engaged" not in validity["reasons"]


def test_run_validity_blocks_local_execution_fallback_from_timeout() -> None:
    validity = build_run_validity(
        summary_json={},
        diagnostics={
            "model_usage": {"text_calls": 1},
            "section_diagnostics": {"section_extraction_enabled": True},
            "fallback_counts_by_reason": {"text_subprocess_fallback:timeout": 1},
        },
        provider="local",
        require_source_provenance=False,
    )

    assert validity["valid"] is False
    assert validity["rerun_required"] is True
    assert "fallback_engaged" in validity["reasons"]
    assert validity["fallback_audit"]["execution_fallback_used"] is True


def test_invalid_report_reason_reports_local_text_analysis_failure() -> None:
    validity = build_run_validity(
        summary_json={},
        diagnostics={
            "model_usage": {"text_calls": 0},
            "section_diagnostics": {"section_extraction_enabled": True},
        },
        provider="local",
        require_source_provenance=False,
    )

    assert validity["valid"] is False
    assert "text_llm_calls_zero" in validity["reasons"]
    assert invalid_report_reason(validity, provider="local") == "Local model text analysis did not run."


def test_invalid_report_reason_distinguishes_cached_local_text_reuse() -> None:
    validity = build_run_validity(
        summary_json={},
        diagnostics={
            "model_usage": {"text_calls": 0},
            "analysis_timeline": [
                {
                    "stage": "text",
                    "metadata": {"cache_hit": True, "cache_source": "global"},
                }
            ],
            "section_diagnostics": {"section_extraction_enabled": True},
        },
        provider="local",
        require_source_provenance=False,
    )

    assert validity["valid"] is False
    assert "text_analysis_cache_reused" in validity["reasons"]
    assert "text_llm_calls_zero" not in validity["reasons"]
    assert validity["quality_backend_audit"]["text_cache_reused"] is True
    assert (
        invalid_report_reason(validity, provider="local")
        == "Cached local text-analysis output was reused; rerun with the text cache disabled or cleared "
        "for fresh local-model execution and latency validation."
    )


def test_run_validity_blocks_when_required_local_narrative_synthesis_did_not_run() -> None:
    validity = build_run_validity(
        summary_json={},
        diagnostics={
            "model_usage": {"text_calls": 1, "deep_calls": 0},
            "stage_model_usage": {"synthesis": {"deep_calls": 0}},
            "section_diagnostics": {"section_extraction_enabled": True},
            "synthesis_diagnostics": {
                "narrative_overrides_enabled": True,
                "narrative_synthesis_required": True,
                "narrative_synthesis_deep_calls": 0,
            },
        },
        provider="local",
        require_source_provenance=False,
    )

    assert validity["valid"] is False
    assert "narrative_synthesis_calls_zero" in validity["reasons"]
    assert validity["quality_backend_audit"]["narrative_synthesis_required"] is True
    assert validity["quality_backend_audit"]["narrative_synthesis_ran"] is False
    assert invalid_report_reason(validity, provider="local") == "Local model narrative synthesis did not run."


def test_run_validity_accepts_required_local_narrative_synthesis_call() -> None:
    validity = build_run_validity(
        summary_json={},
        diagnostics={
            "model_usage": {"text_calls": 1, "deep_calls": 1},
            "stage_model_usage": {"synthesis": {"deep_calls": 1}},
            "section_diagnostics": {"section_extraction_enabled": True},
            "synthesis_diagnostics": {
                "narrative_overrides_enabled": True,
                "narrative_synthesis_required": True,
                "narrative_synthesis_deep_calls": 1,
            },
        },
        provider="local",
        require_source_provenance=False,
    )

    assert validity["valid"] is True
    assert "narrative_synthesis_calls_zero" not in validity["reasons"]
    assert validity["quality_backend_audit"]["narrative_synthesis_ran"] is True


def test_invalid_report_reason_does_not_hide_local_fallback_blocker_when_diagnostics_missing() -> None:
    validity = {
        "rerun_required": True,
        "reasons": ["diagnostics_missing", "text_llm_calls_zero", "fallback_engaged"],
    }

    assert (
        invalid_report_reason(validity, provider="local")
        == "Local model analysis used fallback outputs, so rerun after local model and GPU diagnostics are clean."
    )
