from __future__ import annotations

from app.services.analysis.latency import build_latency_profile


def test_latency_profile_ranks_slowest_stages_and_preserves_prompt_model_context() -> None:
    profile = build_latency_profile(
        {
            "parse_timing": {"parse_total_seconds": 4.5, "started_at": "2026-06-21T19:00:00Z"},
            "analysis_timing": {
                "text": 520.25,
                "table": 75.0,
                "figure": 35.0,
                "supplement": 0.0,
                "reconcile": 12.0,
                "synthesis": 44.0,
                "store": 1.5,
                "analysis_total_seconds": 690.0,
            },
            "model_usage": {
                "text_calls": 9,
                "text_total_seconds": 520.25,
                "deep_calls": 2,
                "deep_total_seconds": 56.0,
                "vision_calls": 1,
                "vision_total_seconds": 35.0,
                "slowest_model": "text",
                "slowest_seconds": 520.25,
            },
            "stage_model_usage": {
                "text": {
                    "text_calls": 9,
                    "text_total_seconds": 520.25,
                    "text_model_load_calls": 1,
                    "text_model_load_seconds": 22.5,
                    "slowest_model": "text",
                },
                "synthesis": {"deep_calls": 2, "deep_total_seconds": 44.0, "slowest_model": "deep"},
            },
            "prompt_budget_diagnostics": {
                "totals": {
                    "prompt_calls": 12,
                    "total_prompt_chars": 104000,
                    "max_prompt_chars": 15000,
                    "average_prompt_chars": 8666.7,
                    "max_prompt_modality": "text",
                    "total_prompt_seconds": 612.0,
                    "max_prompt_seconds": 140.0,
                    "max_prompt_seconds_modality": "text",
                    "average_prompt_seconds": 51.0,
                },
                "by_modality": {
                    "text": {
                        "prompt_calls": 9,
                        "total_prompt_chars": 90000,
                        "max_prompt_chars": 15000,
                        "average_prompt_chars": 10000.0,
                        "max_prompt_blocks": 3,
                        "total_prompt_seconds": 520.25,
                        "max_prompt_seconds": 140.0,
                        "average_prompt_seconds": 57.8056,
                        "slowest_prompt_batch": {
                            "batch_index": 4,
                            "duration_seconds": 140.0,
                            "first_anchor": "section:Methods:4",
                            "last_anchor": "section:Results:5",
                        },
                    }
                },
                "quality_flags": ["many_local_prompt_batches", "text_many_batches"],
            },
        },
        document_id=179,
    )

    assert profile["latency_profile_version"] == 1
    assert profile["document_id"] == 179
    assert profile["total_known_seconds"] == 694.5
    assert profile["slowest_stage"] == "text"
    assert profile["top_bottlenecks"][0]["stage"] == "text"
    assert profile["top_bottlenecks"][0]["duration_seconds"] == 520.25
    assert profile["top_bottlenecks"][0]["model_usage"]["calls"] == 9
    assert profile["top_bottlenecks"][0]["model_usage"]["model_load_calls"] == 1
    assert profile["top_bottlenecks"][0]["model_usage"]["model_load_seconds"] == 22.5
    assert profile["top_bottlenecks"][0]["prompt_budget"]["prompt_calls"] == 9
    assert profile["top_bottlenecks"][0]["prompt_budget"]["max_prompt_seconds"] == 140.0
    assert profile["top_bottlenecks"][0]["prompt_budget"]["slowest_prompt_batch"]["batch_index"] == 4
    assert profile["model_totals"]["text"]["total_seconds"] == 520.25
    assert profile["prompt_totals"]["max_prompt_modality"] == "text"
    assert profile["prompt_totals"]["max_prompt_seconds_modality"] == "text"
    assert profile["quality_flags"][0] == "text_slowest_stage"
    assert "many_local_prompt_batches" in profile["quality_flags"]
    assert "text_many_batches" in profile["quality_flags"]
    assert "text_model_cold_start" in profile["quality_flags"]
    assert "table_slow_without_model_usage" in profile["quality_flags"]


def test_latency_profile_uses_timeline_when_stage_timing_is_missing() -> None:
    profile = build_latency_profile(
        {
            "analysis_timeline": [
                {
                    "stage": "figure",
                    "duration_seconds": 10.25,
                    "started_at": "2026-06-21T19:02:00Z",
                    "ended_at": "2026-06-21T19:02:10Z",
                    "metadata": {"chunks": 3},
                }
            ],
            "pipeline_timeline": [
                {
                    "step": "prepare_analysis",
                    "duration_seconds": 0.75,
                    "metadata": {"job_id": 1},
                }
            ],
        }
    )

    assert profile["total_known_seconds"] == 11.0
    assert [stage["stage"] for stage in profile["stages"]] == ["figure", "prepare_analysis"]
    assert profile["stages"][0]["timing_source"] == "analysis_timeline"
    assert profile["stages"][0]["metadata"] == {"chunks": 3}
    assert "model_usage_missing" in profile["quality_flags"]


def test_latency_profile_does_not_treat_missing_timing_as_zero() -> None:
    profile = build_latency_profile({"model_usage": {"text_calls": 0}})

    assert profile["total_known_seconds"] is None
    assert profile["stages"] == []
    assert profile["slowest_stage"] == ""
    assert "latency_timing_missing" in profile["quality_flags"]
    assert "model_calls_zero" in profile["quality_flags"]


def test_latency_profile_preserves_subprocess_timeout_execution() -> None:
    profile = build_latency_profile(
        {
            "analysis_timing": {
                "text": 600.2,
                "analysis_total_seconds": 600.2,
            },
            "stage_model_usage": {
                "text": {
                    "text_calls": 0,
                    "text_total_seconds": 0.0,
                    "execution_attempts": [
                        {
                            "mode": "subprocess_guard",
                            "kind": "text",
                            "chunk_count": 18,
                            "timeout_seconds": 600,
                            "elapsed_seconds": 600.2,
                            "timed_out": True,
                            "exitcode": -15,
                            "payload_received": False,
                            "ok": False,
                        }
                    ],
                }
            },
            "model_usage": {"text_calls": 0},
        }
    )

    text_stage = profile["top_bottlenecks"][0]
    assert text_stage["stage"] == "text"
    assert text_stage["execution"]["timed_out"] is True
    assert text_stage["execution"]["attempts"][0]["timeout_seconds"] == 600
    assert "text_subprocess_timeout" in profile["quality_flags"]
    assert "text_slow_without_model_calls" in profile["quality_flags"]


def test_latency_profile_tolerates_structured_prompt_totals() -> None:
    profile = build_latency_profile(
        {
            "analysis_timing": {
                "synthesis": 30.0,
                "analysis_total_seconds": 30.0,
            },
            "prompt_budget_diagnostics": {
                "totals": {
                    "prompt_calls": 1,
                    "max_prompt_modality": {"modality": "synthesis"},
                    "max_prompt_seconds_modality": ["synthesis"],
                }
            },
        }
    )

    assert profile["prompt_totals"]["prompt_calls"] == 1
    assert profile["prompt_totals"]["max_prompt_modality"] == "{'modality': 'synthesis'}"
    assert profile["prompt_totals"]["max_prompt_seconds_modality"] == "['synthesis']"


def test_latency_profile_surfaces_parser_reuse_and_text_cache_hits() -> None:
    profile = build_latency_profile(
        {
            "parse_timing": {
                "parse_total_seconds": 0.35,
                "parser_reuse": True,
                "reused_assets": 1,
                "asset_status_counts": {"reused": 2},
            },
            "analysis_timing": {
                "text": 0.12,
                "analysis_total_seconds": 0.5,
            },
            "analysis_timeline": [
                {
                    "stage": "text",
                    "duration_seconds": 0.12,
                    "metadata": {
                        "cache_hit": True,
                        "input_chunks": 42,
                    },
                }
            ],
            "model_usage": {"text_calls": 0},
        }
    )

    stages = {stage["stage"]: stage for stage in profile["stages"]}
    assert stages["parse"]["metadata"]["parser_reuse"] is True
    assert stages["parse"]["metadata"]["reused_assets"] == 1
    assert stages["text"]["metadata"]["cache_hit"] is True
    assert stages["text"]["metadata"]["input_chunks"] == 42
    assert profile["cache_summary"] == {
        "parse_reused": True,
        "cache_hit_stages": ["parse", "text"],
        "cache_hit_count": 2,
    }
    assert "parse_reused" in profile["quality_flags"]
    assert "text_cache_hit" in profile["quality_flags"]
