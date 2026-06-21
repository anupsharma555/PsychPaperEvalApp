from __future__ import annotations

import json

from app.services.analysis import runner


def _allow_sparse_text_cache(monkeypatch) -> None:
    monkeypatch.setattr(runner.settings, "analysis_local_text_cache_min_packets", 1)
    monkeypatch.setattr(runner.settings, "analysis_local_text_cache_min_sections", 0)
    monkeypatch.setattr(runner.settings, "analysis_local_text_cache_max_missing_evidence_rate", 1.0)


def test_merge_synthesis_report_usage_uses_narrative_usage_when_counters_are_zero() -> None:
    usage = runner._merge_synthesis_report_usage(  # noqa: SLF001 - diagnostic contract coverage
        {"deep_calls": 0, "deep_errors": 0, "deep_total_seconds": 0.0},
        {
            "section_diagnostics": {
                "narrative_synthesis_usage": {
                    "deep_calls": 2,
                    "deep_errors": 1,
                    "deep_total_seconds": 12.5,
                    "guarded_calls": 2,
                }
            }
        },
    )

    assert usage["deep_calls"] == 2
    assert usage["deep_errors"] == 1
    assert usage["deep_total_seconds"] == 12.5
    assert usage["deep_avg_seconds"] == 6.25
    assert usage["narrative_synthesis_usage"]["guarded_calls"] == 2


def test_multimodal_quality_summary_counts_caption_ocr_and_skipped_assets() -> None:
    summary = runner._build_multimodal_quality_summary(  # noqa: SLF001 - diagnostic contract coverage
        figure_report={
            "evidence_packets": [{"finding_id": "fig-1"}],
            "diagnostics": {
                "vision_calls": 2,
                "vision_success": 1,
                "vision_failures": 1,
                "caption_only_calls": 1,
                "caption_first_skipped_vision": 2,
                "ocr_fallback_calls": 1,
                "downstream_text_sources": {"caption": 1, "caption_plus_ocr_fallback": 1},
                "downstream_text_source_by_anchor": {
                    "F1": "caption",
                    "F2": "caption_plus_ocr_fallback",
                    "F3": "missing",
                },
                "vision_input_sources": {"remote_url": 1},
                "vision_skipped": {"download_error": 1},
            },
        },
        supp_report={
            "evidence_packets": [],
            "diagnostics": {
                "vision_calls": 0,
                "vision_failures": 0,
                "caption_only_calls": 0,
                "caption_first_skipped_vision": 1,
                "ocr_fallback_calls": 0,
                "downstream_text_sources": {},
                "downstream_text_source_by_anchor": {},
                "vision_skipped": {},
            },
        },
    )

    figure = summary["by_modality"]["figure"]
    assert figure["caption_anchored_count"] == 1
    assert figure["caption_first_skipped_vision"] == 2
    assert figure["ocr_dependent_count"] == 1
    assert figure["missing_text_count"] == 1
    assert figure["ocr_dependency_rate"] == 0.3333
    assert figure["skipped_assets"] == {"download_error": 1}
    assert summary["totals"]["vision_calls"] == 2
    assert summary["totals"]["caption_first_skipped_vision"] == 3
    assert summary["totals"]["ocr_fallback_calls"] == 1
    assert "media_assets_skipped" in summary["quality_flags"]


def test_multimodal_quality_summary_flags_high_risk_runs() -> None:
    summary = runner._build_multimodal_quality_summary(  # noqa: SLF001 - diagnostic contract coverage
        figure_report={
            "evidence_packets": [],
            "diagnostics": {
                "vision_calls": 2,
                "vision_success": 0,
                "vision_failures": 2,
                "caption_only_calls": 0,
                "ocr_fallback_calls": 2,
                "downstream_text_source_by_anchor": {
                    "F1": "ocr_fallback",
                    "F2": "caption_plus_ocr_fallback",
                },
                "vision_skipped": {},
            },
        },
        supp_report={"evidence_packets": [], "diagnostics": {}},
    )

    assert summary["totals"]["vision_failure_rate"] == 1.0
    assert summary["totals"]["ocr_dependency_rate"] == 0.5
    assert "high_vision_failure_rate" in summary["quality_flags"]
    assert "high_ocr_dependency" in summary["quality_flags"]
    assert "figure_analysis_produced_no_evidence" in summary["quality_flags"]


def test_prompt_budget_diagnostics_summarizes_local_context_pressure() -> None:
    summary = runner._build_prompt_budget_diagnostics(  # noqa: SLF001 - diagnostic contract coverage
        text_report={
            "diagnostics": {
                "llm_prompt_chars": [900, 1200],
                "llm_prompt_blocks": [2, 1],
                "llm_batch_seconds": [1.5, 3.25],
                "llm_batch_details": [
                    {
                        "batch_index": 1,
                        "duration_seconds": 1.5,
                        "prompt_chars": 900,
                        "prompt_blocks": 2,
                        "first_anchor": "section:Intro:1",
                        "last_anchor": "section:Methods:2",
                    },
                    {
                        "batch_index": 2,
                        "duration_seconds": 3.25,
                        "prompt_chars": 1200,
                        "prompt_blocks": 1,
                        "first_anchor": "section:Results:3",
                        "last_anchor": "section:Results:3",
                    },
                ],
            }
        },
        table_report={"diagnostics": {"prompt_chars": [800], "prompt_blocks": [3]}},
        figure_report={"diagnostics": {"prompt_chars": [15000]}},
        supp_report={"diagnostics": {"text_prompt_chars": 700, "text_prompt_blocks": 2}},
    )

    assert summary["totals"]["prompt_calls"] == 5
    assert summary["totals"]["total_prompt_chars"] == 18600
    assert summary["totals"]["max_prompt_chars"] == 15000
    assert summary["totals"]["max_prompt_modality"] == "figure"
    assert summary["totals"]["total_prompt_seconds"] == 4.75
    assert summary["totals"]["max_prompt_seconds"] == 3.25
    assert summary["totals"]["max_prompt_seconds_modality"] == "text"
    assert summary["by_modality"]["text"]["max_prompt_blocks"] == 2
    assert summary["by_modality"]["text"]["slowest_prompt_batch"]["batch_index"] == 2
    assert summary["by_modality"]["text"]["slowest_prompt_batch"]["first_anchor"] == "section:Results:3"
    assert "figure_large_prompt" in summary["quality_flags"]


def test_prompt_budget_diagnostics_flags_many_local_batches() -> None:
    summary = runner._build_prompt_budget_diagnostics(  # noqa: SLF001 - diagnostic contract coverage
        text_report={"diagnostics": {"llm_prompt_chars": [500] * 6, "llm_prompt_blocks": [1] * 6}},
        table_report={"diagnostics": {"prompt_chars": [600]}},
        figure_report={"diagnostics": {"prompt_chars": [700]}},
        supp_report={"diagnostics": {}},
    )

    assert summary["totals"]["prompt_calls"] == 8
    assert "many_local_prompt_batches" in summary["quality_flags"]
    assert "text_many_batches" in summary["quality_flags"]


def test_usage_merge_preserves_subprocess_execution_attempts() -> None:
    merged = runner._merge_usage_counts(  # noqa: SLF001 - diagnostic contract coverage
        [
            {
                "text_calls": 0,
                "text_total_seconds": 0.0,
                "execution": {
                    "mode": "subprocess_guard",
                    "kind": "text",
                    "timeout_seconds": 600,
                    "elapsed_seconds": 600.2,
                    "timed_out": True,
                    "exitcode": -15,
                    "payload_received": False,
                    "ok": False,
                },
            },
            {
                "text_calls": 0,
                "text_total_seconds": 0.0,
            },
        ]
    )

    assert merged["text_calls"] == 0
    assert merged["execution_attempts"][0]["kind"] == "text"
    assert merged["execution_attempts"][0]["timed_out"] is True
    assert merged["execution_attempts"][0]["timeout_seconds"] == 600


def test_usage_merge_preserves_model_load_timing() -> None:
    before = {
        "text_calls": 0,
        "text_total_seconds": 0.0,
        "text_model_load_calls": 0,
        "text_model_load_errors": 0,
        "text_model_load_seconds": 0.0,
    }
    after = {
        "text_calls": 1,
        "text_total_seconds": 7.0,
        "text_model_load_calls": 1,
        "text_model_load_errors": 0,
        "text_model_load_seconds": 2.5,
    }

    delta = runner._snapshot_counter_delta(before, after)  # noqa: SLF001
    merged = runner._merge_usage_counts(  # noqa: SLF001
        [delta, {"deep_model_load_calls": 1, "deep_model_load_seconds": 3.25}]
    )

    assert delta["text_model_load_calls"] == 1
    assert delta["text_model_load_seconds"] == 2.5
    assert merged["text_model_load_calls"] == 1
    assert merged["text_model_load_seconds"] == 2.5
    assert merged["deep_model_load_calls"] == 1
    assert merged["deep_model_load_seconds"] == 3.25


def test_modality_heartbeat_message_names_active_stages_and_elapsed_time() -> None:
    message = runner._modality_heartbeat_message(  # noqa: SLF001 - progress contract coverage
        ["text", "figure", "text"],
        elapsed_seconds=125,
    )

    assert message == "Still analyzing figure, text with local model/runtime (2 min elapsed)"


def test_modality_stage_heartbeat_progress_stays_inside_stage_range() -> None:
    early = runner._modality_stage_heartbeat_progress(  # noqa: SLF001 - progress contract coverage
        0.56,
        0.64,
        elapsed_seconds=30,
    )
    late = runner._modality_stage_heartbeat_progress(  # noqa: SLF001 - progress contract coverage
        0.56,
        0.64,
        elapsed_seconds=900,
    )

    assert 0.56 < early < late
    assert late < 0.64


def test_local_text_analysis_cache_reuses_matching_report(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner.settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(runner.settings, "llm_provider", "local")
    monkeypatch.setattr(runner.settings, "analysis_text_llm_enabled", True)
    monkeypatch.setattr(runner.settings, "analysis_local_text_cache_enabled", True)
    _allow_sparse_text_cache(monkeypatch)

    chunks = [
        {
            "anchor": "section:Methods:1",
            "content": "Participants received fluoxetine 20 mg daily for eight weeks.",
            "meta": json.dumps({"section_norm": "methods"}),
        }
    ]
    report = {
        "evidence_packets": [
            {
                "finding_id": "text-1",
                "statement": "Participants received fluoxetine 20 mg daily.",
                "evidence_refs": ["section:Methods:1"],
            }
        ],
        "diagnostics": {"llm_batches": 1},
    }

    runner._write_local_text_analysis_cache(17, chunks, report)  # noqa: SLF001
    cached, usage, fallback, decisions = runner._maybe_read_local_text_analysis_cache(17, chunks)  # noqa: SLF001

    assert cached is not None
    assert decisions[0]["status"] == "accepted"
    assert cached["evidence_packets"][0]["finding_id"] == "text-1"
    assert cached["diagnostics"]["llm_batches"] == 0
    assert cached["diagnostics"]["local_text_analysis_cache"]["hit"] is True
    assert cached["diagnostics"]["local_text_analysis_cache"]["source"] == "document"
    assert cached["diagnostics"]["local_text_analysis_cache"]["cached_prompt_diagnostics"]["llm_batches"] == 1
    assert usage["text_calls"] == 0
    assert fallback == ""


def test_local_text_analysis_cache_invalidates_when_settings_change(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner.settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(runner.settings, "llm_provider", "local")
    monkeypatch.setattr(runner.settings, "analysis_text_llm_enabled", True)
    monkeypatch.setattr(runner.settings, "analysis_local_text_cache_enabled", True)
    _allow_sparse_text_cache(monkeypatch)

    chunks = [{"anchor": "section:Results:1", "content": "MADRS response improved.", "meta": "{}"}]
    report = {"evidence_packets": [{"finding_id": "text-1"}], "diagnostics": {}}

    runner._write_local_text_analysis_cache(18, chunks, report)  # noqa: SLF001
    monkeypatch.setattr(
        runner.settings,
        "analysis_local_text_preselection_max_chunks",
        int(runner.settings.analysis_local_text_preselection_max_chunks) + 1,
    )
    cached, _usage, _fallback, decisions = runner._maybe_read_local_text_analysis_cache(18, chunks)  # noqa: SLF001

    assert cached is None
    assert decisions[0]["status"] == "signature_mismatch"


def test_local_text_analysis_global_cache_reuses_matching_report_across_documents(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(runner.settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(runner.settings, "llm_provider", "local")
    monkeypatch.setattr(runner.settings, "analysis_text_llm_enabled", True)
    monkeypatch.setattr(runner.settings, "analysis_local_text_cache_enabled", True)
    monkeypatch.setattr(runner.settings, "analysis_local_text_global_cache_enabled", True)
    _allow_sparse_text_cache(monkeypatch)

    chunks = [
        {
            "anchor": "section:Results:2",
            "content": "Sertraline remission was 48% versus 31% for placebo.",
            "meta": json.dumps({"section_norm": "results"}),
        }
    ]
    report = {
        "evidence_packets": [{"finding_id": "text-1", "evidence_refs": ["section:Results:2"]}],
        "diagnostics": {"llm_batches": 1, "llm_prompt_chars": [1200]},
    }

    runner._write_local_text_analysis_cache(21, chunks, report)  # noqa: SLF001
    cached, usage, fallback, decisions = runner._maybe_read_local_text_analysis_cache(22, chunks)  # noqa: SLF001

    assert cached is not None
    assert decisions[0]["status"] == "accepted"
    assert cached["evidence_packets"][0]["finding_id"] == "text-1"
    assert cached["diagnostics"]["llm_batches"] == 0
    assert cached["diagnostics"]["local_text_analysis_cache"]["source"] == "global"
    assert cached["diagnostics"]["local_text_analysis_cache"]["cached_prompt_diagnostics"]["llm_prompt_chars"] == [1200]
    assert usage["text_calls"] == 0
    assert fallback == ""


def test_local_text_analysis_global_cache_can_be_disabled(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner.settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(runner.settings, "llm_provider", "local")
    monkeypatch.setattr(runner.settings, "analysis_text_llm_enabled", True)
    monkeypatch.setattr(runner.settings, "analysis_local_text_cache_enabled", True)
    monkeypatch.setattr(runner.settings, "analysis_local_text_global_cache_enabled", True)
    _allow_sparse_text_cache(monkeypatch)

    chunks = [{"anchor": "section:Methods:3", "content": "Participants used lithium 900 mg/day.", "meta": "{}"}]
    report = {"evidence_packets": [{"finding_id": "text-1"}], "diagnostics": {}}

    runner._write_local_text_analysis_cache(23, chunks, report)  # noqa: SLF001
    monkeypatch.setattr(runner.settings, "analysis_local_text_global_cache_enabled", False)
    cached, _usage, _fallback, decisions = runner._maybe_read_local_text_analysis_cache(24, chunks)  # noqa: SLF001

    assert cached is None
    assert decisions == []


def test_local_text_analysis_cache_rejects_sparse_cached_report(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner.settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(runner.settings, "llm_provider", "local")
    monkeypatch.setattr(runner.settings, "analysis_text_llm_enabled", True)
    monkeypatch.setattr(runner.settings, "analysis_local_text_cache_enabled", True)
    monkeypatch.setattr(runner.settings, "analysis_local_text_cache_min_packets", 2)
    monkeypatch.setattr(runner.settings, "analysis_local_text_cache_min_sections", 2)
    monkeypatch.setattr(runner.settings, "analysis_local_text_cache_max_missing_evidence_rate", 0.25)

    chunks = [
        {
            "anchor": "section:Methods:1",
            "content": "Participants received lithium 900 mg/day.",
            "meta": json.dumps({"section_norm": "methods"}),
        }
    ]
    report = {
        "evidence_packets": [
            {
                "finding_id": "text-1",
                "statement": "Participants received lithium 900 mg/day.",
                "evidence_refs": ["section:Methods:1"],
                "section_label": "methods",
            }
        ],
        "diagnostics": {"llm_batches": 1},
    }

    runner._write_local_text_analysis_cache(25, chunks, report)  # noqa: SLF001
    cached, usage, fallback, decisions = runner._maybe_read_local_text_analysis_cache(25, chunks)  # noqa: SLF001

    assert cached is None
    assert decisions[0]["status"] == "rejected_quality"
    assert decisions[0]["quality"]["reject_reasons"] == ["too_few_packets", "too_few_sections"]
    assert usage["text_calls"] == 0
    assert fallback == ""


def test_local_text_analysis_cache_accepts_quality_checked_report(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner.settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(runner.settings, "llm_provider", "local")
    monkeypatch.setattr(runner.settings, "analysis_text_llm_enabled", True)
    monkeypatch.setattr(runner.settings, "analysis_local_text_cache_enabled", True)
    monkeypatch.setattr(runner.settings, "analysis_local_text_cache_min_packets", 3)
    monkeypatch.setattr(runner.settings, "analysis_local_text_cache_min_sections", 2)
    monkeypatch.setattr(runner.settings, "analysis_local_text_cache_max_missing_evidence_rate", 0.25)

    chunks = [
        {
            "anchor": "section:Methods:1",
            "content": "Participants were randomized to sertraline or placebo.",
            "meta": json.dumps({"section_norm": "methods"}),
        },
        {
            "anchor": "section:Results:1",
            "content": "Remission was 48% for sertraline versus 31% for placebo.",
            "meta": json.dumps({"section_norm": "results"}),
        },
    ]
    report = {
        "evidence_packets": [
            {
                "finding_id": "text-1",
                "statement": "Participants were randomized to sertraline or placebo.",
                "evidence_refs": ["section:Methods:1"],
                "section_label": "methods",
            },
            {
                "finding_id": "text-2",
                "statement": "Remission was 48% for sertraline versus 31% for placebo.",
                "evidence_refs": ["section:Results:1"],
            },
            {
                "finding_id": "text-3",
                "statement": "The report includes both methods and results evidence.",
                "evidence_refs": ["section:Methods:1", "section:Results:1"],
                "section_label": "unknown",
            },
        ],
        "diagnostics": {"llm_batches": 2},
    }

    runner._write_local_text_analysis_cache(26, chunks, report)  # noqa: SLF001
    cached, usage, fallback, decisions = runner._maybe_read_local_text_analysis_cache(26, chunks)  # noqa: SLF001

    assert cached is not None
    assert decisions[0]["status"] == "accepted"
    assert cached["diagnostics"]["llm_batches"] == 0
    cache_info = cached["diagnostics"]["local_text_analysis_cache"]
    assert cache_info["hit"] is True
    assert cache_info["quality"]["accepted"] is True
    assert cache_info["quality"]["packet_count"] == 3
    assert cache_info["quality"]["section_count"] == 2
    assert cache_info["quality"]["sections"] == ["methods", "results"]
    assert usage["text_calls"] == 0
    assert fallback == ""
