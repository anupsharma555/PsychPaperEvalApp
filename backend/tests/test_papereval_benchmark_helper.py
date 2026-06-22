from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HELPER_PATH = PROJECT_ROOT / ".codex" / "skills" / "papereval-benchmark" / "scripts" / "papereval_benchmark_qa.py"


def _load_helper_module():
    spec = importlib.util.spec_from_file_location("papereval_benchmark_qa", HELPER_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    old_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old_dont_write_bytecode
    return module


def _args(**overrides):
    defaults = {
        "allow_runtime_not_ready": False,
        "allow_surface_not_ready": False,
        "frontend_port": 5184,
        "max_concurrent": 1,
        "start_app": False,
        "disable_local_text_cache": False,
        "allow_local_text_cache": False,
        "llm_provider": "local",
        "surface": "api",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_single_mode_can_sample_diagnostic_without_making_it_release_fatal(tmp_path: Path) -> None:
    helper = _load_helper_module()
    manifest = helper._load_json(helper.MANIFEST)

    selected = helper._selected_cases(
        manifest,
        mode="single",
        tier="all",
        case_ids=[],
        include_unscored=True,
        random_seed=1,
    )

    assert selected[0]["scoring"] == "diagnostic_coverage"
    record = helper._new_record(selected[0], _args(), "http://127.0.0.1:8000/api", 1)
    helper._mark_failed(record, tmp_path / "diagnostic", "gold comparison mismatch")

    assert record["release_gate"] is False
    assert record["decision"] == "diagnostic_fail"
    assert helper._process_exit_status(
        [record],
        fail_on_incompatible=True,
        fail_on_diagnostic=False,
    ) == 0
    assert helper._process_exit_status(
        [record],
        fail_on_incompatible=True,
        fail_on_diagnostic=True,
    ) == 1


def test_reference_scored_failure_remains_release_fatal(tmp_path: Path) -> None:
    helper = _load_helper_module()
    manifest = helper._load_json(helper.MANIFEST)
    selected = helper._selected_cases(
        manifest,
        mode="suite",
        tier="smoke",
        case_ids=[],
        include_unscored=False,
        random_seed=None,
    )

    record = helper._new_record(selected[0], _args(), "http://127.0.0.1:8000/api", 1)
    helper._mark_failed(record, tmp_path / "release", "report invalid")

    assert record["release_gate"] is True
    assert record["decision"] == "fail"
    assert helper._process_exit_status(
        [record],
        fail_on_incompatible=True,
        fail_on_diagnostic=False,
    ) == 1


def test_queue_timeout_is_separate_from_running_timeout(tmp_path: Path) -> None:
    helper = _load_helper_module()
    state = helper.ActiveCaseState(case={}, record={"job_id": 7}, case_dir=tmp_path, submitted_at=100.0)

    assert helper._update_active_timing(
        state,
        {"id": 7, "status": "queued"},
        now=109.0,
        queue_timeout=10.0,
        timeout_per_case=30.0,
    ) is None
    assert "did not start" in helper._update_active_timing(
        state,
        {"id": 7, "status": "queued"},
        now=111.0,
        queue_timeout=10.0,
        timeout_per_case=30.0,
    )

    running = helper.ActiveCaseState(case={}, record={"job_id": 8}, case_dir=tmp_path, submitted_at=100.0)
    assert helper._update_active_timing(
        running,
        {"id": 8, "status": "running"},
        now=120.0,
        queue_timeout=10.0,
        timeout_per_case=30.0,
    ) is None
    assert running.running_deadline == 150.0
    assert helper._update_active_timing(
        running,
        {"id": 8, "status": "running"},
        now=149.0,
        queue_timeout=10.0,
        timeout_per_case=30.0,
    ) is None
    assert "did not finish" in helper._update_active_timing(
        running,
        {"id": 8, "status": "running"},
        now=151.0,
        queue_timeout=10.0,
        timeout_per_case=30.0,
    )


def test_runtime_preflight_reports_missing_required_services() -> None:
    helper = _load_helper_module()
    status = {
        "processing": {
            "running": False,
            "paused": True,
            "executor_enabled": False,
            "worker_capacity": 0,
            "last_error": "pool crashed",
        },
        "grobid": {"ready": False, "error": "ConnectError"},
        "provider": "local",
        "provider_ready": False,
    }

    failures = helper._runtime_readiness_failures(status, allow_runtime_not_ready=False)

    assert "backend job runner is not running" in failures
    assert "backend job runner is paused" in failures
    assert "backend job runner executor is disabled" in failures
    assert "backend job runner has no worker capacity" in failures
    assert "backend job runner reports error: pool crashed" in failures
    assert any("GROBID is not ready" in failure for failure in failures)
    assert "LLM provider is not ready (local)" in failures
    assert helper._runtime_readiness_failures(status, allow_runtime_not_ready=True) == []


def test_diagnostic_extract_preserves_latency_profile_summary() -> None:
    helper = _load_helper_module()
    report = {
        "latency_profile": {
            "total_known_seconds": 1327.0451,
            "slowest_stage": "text",
            "top_bottlenecks": [
                {
                    "stage": "text",
                    "duration_seconds": 600.2,
                    "timing_source": "analysis_timing",
                    "execution": {
                        "attempt_count": 1,
                        "timed_out": True,
                        "total_elapsed_seconds": 600.2,
                    },
                },
                {
                    "stage": "supplement",
                    "duration_seconds": 74.8305,
                    "timing_source": "analysis_timing",
                    "prompt_budget": {
                        "prompt_calls": 1,
                        "max_prompt_chars": 9333,
                        "max_prompt_seconds": 74.159,
                    },
                },
            ],
            "quality_flags": ["text_subprocess_timeout", "text_slow_without_model_calls"],
            "cache_summary": {"cache_hit_stages": ["text"], "cache_hit_count": 1},
        },
        "analysis_diagnostics": {"diagnostics": {"run_validity": {"valid": True}}},
        "summary_json": {"model_usage": {"text_calls": 1}},
    }

    diagnostics = helper._diagnostic_extract(report, {"status": "completed", "message": "Completed"})

    latency = diagnostics["latency_profile"]
    assert latency["total_known_seconds"] == 1327.0451
    assert latency["slowest_stage"] == "text"
    assert latency["top_bottlenecks"][0]["execution"]["timed_out"] is True
    assert latency["top_bottlenecks"][1]["prompt_budget"]["max_prompt_chars"] == 9333
    assert "text_subprocess_timeout" in latency["quality_flags"]
    assert latency["cache_hit_stages"] == ["text"]
    assert latency["cache_hit_count"] == 1


def test_diagnostic_extract_reads_nested_run_validity() -> None:
    helper = _load_helper_module()
    report = {
        "analysis_diagnostics": {
            "diagnostics": {
                "run_validity": {
                    "valid": False,
                    "reasons": ["text_llm_calls_zero"],
                    "fallback_audit": {"text_llm_calls": 0},
                }
            }
        },
        "summary_json": {},
    }

    diagnostics = helper._diagnostic_extract(report, {"status": "completed", "message": "Completed"})

    assert diagnostics["run_validity"]["valid"] is False
    assert diagnostics["run_validity"]["reasons"] == ["text_llm_calls_zero"]
    assert diagnostics["fallback_audit"]["text_llm_calls"] == 0


def test_iteration_diagnostics_identifies_next_focus_from_latency_and_comparison() -> None:
    helper = _load_helper_module()
    record = {
        "case_id": "paper-1",
        "decision": "pass",
        "release_gate": True,
        "failures": [],
        "diagnostics": {
            "latency_profile": {
                "slowest_stage": "text",
                "top_bottlenecks": [
                    {"stage": "text", "duration_seconds": 529.913},
                ],
                "quality_flags": ["text_slowest_stage", "many_local_prompt_batches"],
            }
        },
            "comparison": {
                "comparison": {
                    "compatible": True,
                    "overall_benchmark_score": 0.941,
                    "benchmark_content_score": 0.941,
                    "benchmark_content_score_basis": {
                        "matched_slots": 16,
                        "expected_slots": 17,
                    },
                    "usable_packet_rate": 0.976,
                    "critical_claim_candidate_rate": 1.0,
                }
        },
    }

    diagnostics = helper._iteration_diagnostics(record)

    assert diagnostics["case_id"] == "paper-1"
    assert diagnostics["compatible"] is True
    assert diagnostics["overall_benchmark_score"] == 0.941
    assert diagnostics["benchmark_content_score"] == 0.941
    assert diagnostics["benchmark_matched_slots"] == 16.0
    assert diagnostics["benchmark_expected_slots"] == 17.0
    assert diagnostics["benchmark_missing_slots"] == 1.0
    assert diagnostics["usable_packet_rate"] == 0.976
    assert diagnostics["slowest_stage"] == "text"
    assert diagnostics["slowest_stage_seconds"] == 529.913
    assert diagnostics["next_focus"] == [
        "optimize_stage:text",
        "diagnose_flag:text_slowest_stage",
        "diagnose_flag:many_local_prompt_batches",
    ]


def test_iteration_diagnostics_flags_text_cache_validity_conflict() -> None:
    helper = _load_helper_module()
    record = {
        "case_id": "paper-cache",
        "decision": "diagnostic_fail",
        "failures": ["Local model text analysis did not run."],
        "diagnostics": {
            "report_invalid_reason": "Local model text analysis did not run.",
            "latency_profile": {
                "slowest_stage": "parse",
                "top_bottlenecks": [{"stage": "parse", "duration_seconds": 10.5}],
                "quality_flags": ["parse_slowest_stage", "text_cache_hit", "model_calls_zero"],
                "cache_hit_stages": ["text"],
                "cache_hit_count": 1,
            },
        },
        "comparison": {"comparison": {"compatible": False}},
    }

    diagnostics = helper._iteration_diagnostics(record)

    assert diagnostics["text_analysis_cache_validity_conflict"] is True
    assert diagnostics["cache_hit_stages"] == ["text"]
    assert diagnostics["cache_hit_count"] == 1
    assert diagnostics["next_focus"][:3] == [
        "fix_current_failure",
        "resolve_text_cache_validity_conflict",
        "optimize_stage:parse",
    ]


def test_merged_iteration_diagnostics_refreshes_derived_cache_conflict_fields() -> None:
    helper = _load_helper_module()
    record = {
        "case_id": "paper-cache",
        "decision": "diagnostic_fail",
        "failures": ["Local model text analysis did not run."],
        "diagnostics": {
            "report_invalid_reason": "Local model text analysis did not run.",
            "latency_profile": {
                "quality_flags": ["text_cache_hit", "model_calls_zero"],
            },
        },
        "comparison": {"comparison": {"compatible": False}},
        "iteration_diagnostics": {
            "next_focus": ["fix_current_failure", "diagnose_flag:model_calls_zero"],
            "text_analysis_cache_validity_conflict": False,
        },
    }

    diagnostics = helper._merged_iteration_diagnostics(record)

    assert diagnostics["text_analysis_cache_validity_conflict"] is True
    assert diagnostics["next_focus"][:2] == [
        "fix_current_failure",
        "resolve_text_cache_validity_conflict",
    ]


def test_iteration_diagnostics_flags_missing_narrative_synthesis() -> None:
    helper = _load_helper_module()
    record = {
        "case_id": "paper-no-synthesis",
        "decision": "diagnostic_fail",
        "failures": ["Local model narrative synthesis did not run."],
        "diagnostics": {
            "report_invalid_reason": "Local model narrative synthesis did not run.",
            "run_validity": {"reasons": ["narrative_synthesis_calls_zero"]},
            "latency_profile": {
                "slowest_stage": "synthesis",
                "top_bottlenecks": [{"stage": "synthesis", "duration_seconds": 14.2}],
                "quality_flags": [],
            },
        },
        "comparison": {"comparison": {"compatible": False}},
    }

    diagnostics = helper._iteration_diagnostics(record)

    assert diagnostics["narrative_synthesis_missing"] is True
    assert diagnostics["next_focus"][:3] == [
        "fix_current_failure",
        "resolve_missing_narrative_synthesis",
        "optimize_stage:synthesis",
    ]


def test_merged_iteration_diagnostics_refreshes_missing_narrative_synthesis_fields() -> None:
    helper = _load_helper_module()
    record = {
        "case_id": "paper-no-synthesis",
        "decision": "diagnostic_fail",
        "failures": ["Local model narrative synthesis did not run."],
        "diagnostics": {
            "report_invalid_reason": "Local model narrative synthesis did not run.",
            "run_validity": {"reasons": ["narrative_synthesis_calls_zero"]},
            "latency_profile": {"quality_flags": []},
        },
        "iteration_diagnostics": {
            "next_focus": ["fix_current_failure"],
            "narrative_synthesis_missing": False,
        },
    }

    diagnostics = helper._merged_iteration_diagnostics(record)

    assert diagnostics["narrative_synthesis_missing"] is True
    assert diagnostics["next_focus"][:2] == [
        "fix_current_failure",
        "resolve_missing_narrative_synthesis",
    ]


def test_benchmark_score_summary_counts_zero_score_as_scored() -> None:
    helper = _load_helper_module()

    summary = helper._benchmark_score_summary(
        [
            {
                "overall_benchmark_score": 0.0,
                "benchmark_matched_slots": 0,
                "benchmark_expected_slots": 10,
            },
            {
                "overall_benchmark_score": 1.0,
                "benchmark_matched_slots": 10,
                "benchmark_expected_slots": 10,
            },
        ]
    )

    assert summary == {
        "scored_cases": 2,
        "total_cases": 2,
        "weighted_overall_benchmark_score": 0.5,
        "mean_overall_benchmark_score": 0.5,
        "matched_slots": 10,
        "expected_slots": 20,
        "missing_slots": 10,
        "extra_content_penalized": False,
    }


def test_score_priority_rows_rank_lowest_scored_cases_first() -> None:
    helper = _load_helper_module()

    ranked = helper._score_priority_rows(
        [
            {
                "case_id": "paper-high",
                "overall_benchmark_score": 0.9,
                "benchmark_missing_slots": 2,
                "claim_requirement_gap_count": 1,
            },
            {
                "case_id": "paper-unscored",
                "overall_benchmark_score": None,
            },
            {
                "case_id": "paper-low",
                "overall_benchmark_score": 0.1,
                "benchmark_missing_slots": 12,
                "claim_requirement_gap_count": 4,
            },
        ]
    )

    assert [row["case_id"] for row in ranked] == ["paper-low", "paper-high"]
    assert ranked[0]["overall_benchmark_score"] == 0.1
    assert ranked[0]["benchmark_missing_slots"] == 12.0
    assert ranked[0]["claim_requirement_gap_count"] == 4


def test_benchmark_definition_fingerprint_tracks_gold_manifest_inputs() -> None:
    helper = _load_helper_module()

    fingerprint = helper._benchmark_definition_fingerprint()

    assert fingerprint["algorithm"] == "sha256"
    assert len(fingerprint["digest"]) == 64
    paths = {item["path"] for item in fingerprint["files"]}
    assert "benchmarks/app_evaluation_standard.json" in paths
    assert "benchmarks/multi_paper_benchmark.json" in paths
    assert "benchmarks/gold_standards/sharma_2017_reward_deficits.json" in paths
    assert fingerprint["file_count"] == len(fingerprint["files"])


def test_output_change_path_classification_separates_harness_from_output_risk() -> None:
    helper = _load_helper_module()

    audit = helper._classify_output_change_paths(
        [
            ".codex/skills/papereval-benchmark/scripts/papereval_benchmark_qa.py",
            ".codex/skills/papereval-run-qa/scripts/papereval_run_qa.py",
            ".gitignore",
            "backend/tests/test_papereval_benchmark_helper.py",
            "backend/tests/test_desktop_api.py",
            "docs/BACKLOG.md",
            "scripts/validate_gold_claims.py",
            "scripts/compare_upstream_ab.py",
            "backend/app/services/report_synthesis.py",
            "assets/icons/PaperEval-icon.png",
            "desktop_ui/src/App.jsx",
            "desktop_shell/src-tauri/src/main.rs",
            "backend/requirements.txt",
            "scripts/run_app.py",
            "README.md",
            "scratch/notes.txt",
        ]
    )

    assert audit["benchmark_only"] == [
        ".codex/skills/papereval-benchmark/scripts/papereval_benchmark_qa.py",
        ".codex/skills/papereval-run-qa/scripts/papereval_run_qa.py",
        ".gitignore",
        "README.md",
        "backend/tests/test_desktop_api.py",
        "backend/tests/test_papereval_benchmark_helper.py",
        "docs/BACKLOG.md",
        "scripts/compare_upstream_ab.py",
        "scripts/validate_gold_claims.py",
    ]
    assert audit["output_risk"] == [
        "assets/icons/PaperEval-icon.png",
        "backend/app/services/report_synthesis.py",
        "backend/requirements.txt",
        "desktop_shell/src-tauri/src/main.rs",
        "desktop_ui/src/App.jsx",
        "scripts/run_app.py",
    ]
    assert audit["unknown"] == ["scratch/notes.txt"]
    assert audit["benchmark_only_category_counts"] == {
        "docs_or_metadata": 3,
        "qa_skill": 2,
        "scripts": 2,
        "tests": 2,
    }
    assert audit["output_risk_category_counts"] == {
        "app_assets": 1,
        "backend_app": 1,
        "dependency_or_env": 1,
        "desktop_launcher": 1,
        "desktop_ui": 1,
        "scripts": 1,
    }
    assert audit["unknown_category_counts"] == {"other": 1}
    assert audit["diagnostic_only"] is False


def test_output_change_audit_json_can_fail_on_output_risk(monkeypatch, capsys) -> None:
    helper = _load_helper_module()
    monkeypatch.setattr(
        helper,
        "_git_changed_paths",
        lambda root: [
            ".codex/skills/papereval-benchmark/SKILL.md",
            "backend/app/analysis/runner.py",
        ],
    )

    assert helper._output_change_audit(json_output=True, fail_on_output_risk=True) == 1

    payload = __import__("json").loads(capsys.readouterr().out)
    assert payload["benchmark_only"] == [".codex/skills/papereval-benchmark/SKILL.md"]
    assert payload["output_risk"] == ["backend/app/analysis/runner.py"]
    assert payload["output_risk_count"] == 1
    assert payload["diagnostic_only"] is False


def test_suggested_active_command_can_disable_local_text_cache() -> None:
    helper = _load_helper_module()
    case = {
        "case_id": "pmc7440080_covid_economic_policy",
        "scoring": "diagnostic_coverage",
        "next_focus": ["resolve_text_cache_validity_conflict"],
    }

    command = helper._suggested_active_command(case, disable_local_text_cache=True)

    assert helper._needs_text_cache_disabled_run(case) is True
    assert "--disable-local-text-cache" in command
    assert "--include-unscored" in command


def test_start_app_disables_local_text_cache_by_default_for_local_provider(monkeypatch, tmp_path: Path) -> None:
    helper = _load_helper_module()
    captured: dict[str, object] = {}

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env")
        return SimpleNamespace()

    monkeypatch.setattr(helper.subprocess, "Popen", fake_popen)

    helper._start_app(
        _args(
            start_app=True,
            backend_port=8765,
            llm_provider="local",
            force_start=False,
        ),
        "http://127.0.0.1:8765/api",
        tmp_path,
    )

    env = captured["env"]
    assert isinstance(env, dict)
    assert env["ANALYSIS_LOCAL_TEXT_CACHE_ENABLED"] == "false"
    assert env["ANALYSIS_LOCAL_TEXT_GLOBAL_CACHE_ENABLED"] == "false"


def test_start_app_can_allow_local_text_cache(monkeypatch, tmp_path: Path) -> None:
    helper = _load_helper_module()
    captured: dict[str, object] = {}

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env")
        return SimpleNamespace()

    monkeypatch.setattr(helper.subprocess, "Popen", fake_popen)

    helper._start_app(
        _args(
            start_app=True,
            backend_port=8765,
            llm_provider="local",
            force_start=False,
            allow_local_text_cache=True,
        ),
        "http://127.0.0.1:8765/api",
        tmp_path,
    )

    env = captured["env"]
    assert isinstance(env, dict)
    assert "ANALYSIS_LOCAL_TEXT_CACHE_ENABLED" not in env
    assert "ANALYSIS_LOCAL_TEXT_GLOBAL_CACHE_ENABLED" not in env


def test_active_run_require_diagnostic_only_refuses_before_backend_start(monkeypatch, tmp_path: Path) -> None:
    helper = _load_helper_module()
    started = {"called": False}
    monkeypatch.setattr(
        helper,
        "_git_changed_paths",
        lambda root: ["backend/app/services/analysis/synthesis.py"],
    )

    def fake_start_app(*args, **kwargs):
        started["called"] = True
        raise AssertionError("backend should not start when diagnostic-only gate fails")

    monkeypatch.setattr(helper, "_start_app", fake_start_app)

    args = _args(
        active_run=True,
        api_base="http://127.0.0.1:8000/api",
        backend_port=8000,
        case_ids=["sharma_2017_reward_deficits"],
        include_unscored=False,
        mode="suite",
        out_dir=str(tmp_path / "active"),
        random_seed=None,
        require_diagnostic_only=True,
        tier="smoke",
    )

    with pytest.raises(SystemExit) as excinfo:
        helper._run_active_benchmark(args)

    assert "requires a diagnostic-only worktree" in str(excinfo.value)
    assert "output_risk_categories=backend_app:1" in str(excinfo.value)
    assert started["called"] is False
    assert not (tmp_path / "active").exists()


def test_active_run_summary_records_output_change_audit_on_preflight_failure(monkeypatch, tmp_path: Path) -> None:
    helper = _load_helper_module()
    monkeypatch.setattr(
        helper,
        "_git_changed_paths",
        lambda root: [".codex/skills/papereval-benchmark/SKILL.md"],
    )
    monkeypatch.setattr(helper, "_start_app", lambda args, api_base, run_dir: None)
    monkeypatch.setattr(
        helper,
        "_wait_api_ready",
        lambda api_base, *, timeout_seconds: {"processing": {"worker_capacity": 1}},
    )
    monkeypatch.setattr(
        helper,
        "_run_preflight",
        lambda api_base, api_status, args: (_ for _ in ()).throw(RuntimeError("runtime unavailable")),
    )
    run_dir = tmp_path / "active"
    args = _args(
        active_run=True,
        api_base="http://127.0.0.1:8000/api",
        backend_port=8000,
        case_ids=["sharma_2017_reward_deficits"],
        include_unscored=False,
        mode="suite",
        out_dir=str(run_dir),
        queue_timeout=600.0,
        random_seed=None,
        require_diagnostic_only=False,
        startup_timeout=1.0,
        tier="smoke",
        timeout_per_case=1800.0,
    )

    assert helper._run_active_benchmark(args) == 1

    summary = __import__("json").loads((run_dir / "active_benchmark_summary.json").read_text(encoding="utf-8"))
    assert summary["preflight_ok"] is False
    assert summary["preflight_failure"] == "runtime unavailable"
    assert summary["output_change_audit"]["diagnostic_only"] is True
    assert summary["output_change_audit"]["benchmark_only"] == [
        ".codex/skills/papereval-benchmark/SKILL.md"
    ]


def test_summarize_existing_run_prints_non_live_iteration_summary(tmp_path: Path, capsys) -> None:
    helper = _load_helper_module()
    summary = {
        "generated_at": "2026-06-21T20:12:59Z",
        "surface": "web",
        "output_change_audit": {
            "diagnostic_only": True,
            "changed_count": 2,
            "output_risk_count": 0,
            "unknown_count": 0,
        },
        "records": [
            {
                "case_id": "paper-1",
                "decision": "pass",
                "diagnostics": {
                    "latency_profile": {
                        "slowest_stage": "text",
                        "top_bottlenecks": [{"stage": "text", "duration_seconds": 100.0}],
                    }
                },
                "comparison": {
                    "comparison": {
                        "compatible": True,
                        "overall_benchmark_score": 0.9,
                        "benchmark_content_score_basis": {
                            "matched_slots": 9,
                            "expected_slots": 10,
                        },
                        "usable_packet_rate": 0.9,
                        "critical_claim_candidate_rate": 1.0,
                    }
                },
            }
        ],
    }
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "active_benchmark_summary.json").write_text(__import__("json").dumps(summary), encoding="utf-8")

    assert helper._summarize_existing_run(run_dir) == 0

    out = capsys.readouterr().out
    assert "records: 1" in out
    assert "output_change_audit: available=True diagnostic_only=True output_risk=0 unknown=0" in out
    assert "benchmark_score_summary: weighted=0.9 mean=0.9 scored=1/1 slots=9/10" in out
    assert "paper-1: decision=pass compatible=True usable=0.9 claims=1.0 slowest=text(100.0s) score=0.9" in out
    assert "next_focus: optimize_stage:text" in out


def test_summarize_existing_run_json_outputs_compact_records(tmp_path: Path, capsys) -> None:
    helper = _load_helper_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "active_benchmark_summary.json").write_text(
        __import__("json").dumps(
            {
                "generated_at": "2026-06-21T20:12:59Z",
                "surface": "web",
                "benchmark_definition": {
                    "algorithm": "sha256",
                    "digest": "abc123",
                    "file_count": 13,
                    "files": [{"path": "benchmarks/multi_paper_benchmark.json", "sha256": "def456"}],
                },
                "output_change_audit": {
                    "diagnostic_only": False,
                    "changed_count": 7,
                    "output_risk_count": 2,
                    "unknown_count": 1,
                },
                "records": [
                    {
                        "case_id": "paper-1",
                        "decision": "diagnostic_fail",
                        "failures": ["evidence-to-gold comparison is incompatible"],
                        "diagnostics": {
                            "latency_profile": {
                                "slowest_stage": "text",
                                "top_bottlenecks": [{"stage": "text", "duration_seconds": 273.8}],
                            }
                        },
                        "comparison": {
                            "comparison": {
                                "compatible": False,
                                "overall_benchmark_score": 0.443,
                                "benchmark_content_score_basis": {
                                    "matched_slots": 27,
                                    "expected_slots": 61,
                                },
                                "claim_requirement_gaps": [
                                    {
                                        "claim_id": "claim-001",
                                        "section": "results",
                                        "missing_entities": ["gut dysbiosis"],
                                    }
                                ],
                            }
                        },
                        "artifacts": {
                            "detailed_analysis_html": "/tmp/run/paper-1/report.html",
                            "slack_summary_markdown": "/tmp/run/paper-1/slack_summary.md",
                        },
                        "summary_json": {"should": "not appear"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert helper._summarize_existing_run(run_dir, json_output=True) == 0

    payload = __import__("json").loads(capsys.readouterr().out)
    assert payload["summary_path"] == str(run_dir / "active_benchmark_summary.json")
    assert payload["generated_at"] == "2026-06-21T20:12:59Z"
    assert payload["surface"] == "web"
    assert payload["output_change_audit_summary"] == {
        "available": True,
        "diagnostic_only": False,
        "changed_count": 7,
        "output_risk_count": 2,
        "unknown_count": 1,
        "output_risk_category_counts": {},
        "unknown_category_counts": {},
    }
    assert payload["benchmark_score_summary"] == {
        "scored_cases": 1,
        "total_cases": 1,
        "weighted_overall_benchmark_score": 0.443,
        "mean_overall_benchmark_score": 0.443,
        "matched_slots": 27,
        "expected_slots": 61,
        "missing_slots": 34,
        "extra_content_penalized": False,
    }
    assert len(payload["records"]) == 1
    record = payload["records"][0]
    assert record["case_id"] == "paper-1"
    assert record["iteration_diagnostics"]["decision"] == "diagnostic_fail"
    assert record["iteration_diagnostics"]["overall_benchmark_score"] == 0.443
    assert record["iteration_diagnostics"]["benchmark_missing_slots"] == 34.0
    assert record["iteration_diagnostics"]["claim_requirement_gap_count"] == 1
    assert record["iteration_diagnostics"]["benchmark_gap_summary"] == (
        "claim-001 (results): missing entities: gut dysbiosis"
    )
    assert record["artifacts"]["detailed_analysis_html"] == "/tmp/run/paper-1/report.html"
    assert record["artifacts"]["slack_summary_markdown"] == "/tmp/run/paper-1/slack_summary.md"
    assert "summary_json" not in record
    assert "report_json" not in record
    assert "evidence_packets" not in record


def test_summarize_history_reports_latest_by_case_and_focus_counts(tmp_path: Path, capsys) -> None:
    helper = _load_helper_module()
    first = {
        "generated_at": "2026-06-21T20:00:00Z",
        "surface": "web",
        "records": [
            {
                "case_id": "paper-1",
                "decision": "fail",
                "failures": ["text timeout"],
                "diagnostics": {
                    "latency_profile": {
                        "slowest_stage": "text",
                        "top_bottlenecks": [{"stage": "text", "duration_seconds": 600.0}],
                    }
                },
                "comparison": {"comparison": {"compatible": False}},
            }
        ],
    }
    second = {
        "generated_at": "2026-06-21T20:30:00Z",
        "surface": "web",
        "output_change_audit": {
            "diagnostic_only": True,
            "changed_count": 3,
            "output_risk_count": 0,
            "unknown_count": 0,
        },
        "records": [
            {
                "case_id": "paper-1",
                "decision": "pass",
                "diagnostics": {
                    "latency_profile": {
                        "slowest_stage": "text",
                        "top_bottlenecks": [{"stage": "text", "duration_seconds": 529.9}],
                        "quality_flags": ["text_many_batches"],
                    }
                },
                "comparison": {
                    "comparison": {
                        "compatible": True,
                        "overall_benchmark_score": 0.941,
                        "benchmark_content_score": 0.941,
                        "usable_packet_rate": 0.976,
                        "critical_claim_candidate_rate": 1.0,
                    }
                },
                "artifacts": {
                    "detailed_analysis_url": "http://127.0.0.1:61311/report.html",
                    "slack_summary_markdown": "/tmp/run/paper-1/slack_summary.md",
                },
            },
            {
                "case_id": "paper-2",
                "decision": "diagnostic_fail",
                "failures": ["missing table evidence"],
            },
        ],
    }
    for name, payload in {"one": first, "two": second}.items():
        run_dir = tmp_path / name
        run_dir.mkdir()
        (run_dir / "active_benchmark_summary.json").write_text(__import__("json").dumps(payload), encoding="utf-8")

    assert helper._summarize_history(tmp_path) == 0

    out = capsys.readouterr().out
    assert "runs: 2" in out
    assert "records: 3" in out
    assert "latest_output_change_summary: diagnostic_only=2 output_risk=0 unknown=0 missing_audit=0" in out
    assert (
        "paper-1: decision=pass compatible=True usable=0.976 claims=1.0 "
        "slowest=text(529.9s) score=0.941 gaps=0 missing_slots=None diagnostic_only=True"
    ) in out
    assert "Lowest score cases:" in out
    assert "paper-1: score=0.941 missing_slots=None gaps=0 decision=pass" in out
    assert "paper-2: decision=diagnostic_fail" in out
    assert "detailed_analysis: http://127.0.0.1:61311/report.html" in out
    assert "slack_summary: /tmp/run/paper-1/slack_summary.md" in out
    assert "fix_current_failure: 2" in out
    assert "optimize_stage:text: 2" in out


def test_summarize_history_json_outputs_latest_by_case_and_focus_counts(tmp_path: Path, capsys) -> None:
    helper = _load_helper_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "active_benchmark_summary.json").write_text(
        __import__("json").dumps(
            {
                "generated_at": "2026-06-21T20:30:00Z",
                "surface": "web",
                "output_change_audit": {
                    "diagnostic_only": False,
                    "changed_count": 9,
                    "output_risk_count": 1,
                    "unknown_count": 2,
                },
                "records": [
                    {
                        "case_id": "paper-1",
                        "decision": "diagnostic_fail",
                        "failures": ["evidence-to-gold comparison is incompatible"],
                        "diagnostics": {
                            "latency_profile": {
                                "slowest_stage": "text",
                                "top_bottlenecks": [{"stage": "text", "duration_seconds": 273.8}],
                                "quality_flags": ["text_slowest_stage"],
                            }
                        },
                        "comparison": {
                            "comparison": {
                                "compatible": False,
                                "overall_benchmark_score": 0.443,
                                "benchmark_content_score_basis": {
                                    "matched_slots": 27,
                                    "expected_slots": 61,
                                },
                                "claim_requirement_gaps": [
                                    {
                                        "claim_id": "claim-001",
                                        "section": "results",
                                        "missing_entities": ["gut dysbiosis"],
                                    }
                                ],
                            }
                        },
                        "artifacts": {
                            "detailed_analysis_html": "/tmp/run/paper-1/report.html",
                            "slack_summary_markdown": "/tmp/run/paper-1/slack_summary.md",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert helper._summarize_history(tmp_path, json_output=True) == 0

    payload = __import__("json").loads(capsys.readouterr().out)
    assert payload["history_root"] == str(tmp_path)
    assert payload["runs"] == 1
    assert payload["records"] == 1
    assert payload["latest_benchmark_score_summary"] == {
        "scored_cases": 1,
        "total_cases": 1,
        "weighted_overall_benchmark_score": 0.443,
        "mean_overall_benchmark_score": 0.443,
        "matched_slots": 27,
        "expected_slots": 61,
        "missing_slots": 34,
        "extra_content_penalized": False,
    }
    assert payload["latest_benchmark_definition_summary"] == {
        "total_cases": 1,
        "matching_current": 0,
        "mismatched_current": 0,
        "missing_definition": 1,
        "scored_matching_current": 0,
        "scored_mismatched_current": 0,
        "scored_missing_definition": 1,
    }
    assert payload["current_definition_benchmark_score_summary"] == {
        "scored_cases": 0,
        "total_cases": 0,
        "weighted_overall_benchmark_score": None,
        "mean_overall_benchmark_score": None,
        "matched_slots": 0,
        "expected_slots": 0,
        "missing_slots": 0,
        "extra_content_penalized": False,
    }
    refresh_case = payload["current_definition_refresh_cases"][0]
    assert refresh_case["case_id"] == "paper-1"
    assert refresh_case["reason"] == "missing_definition"
    assert refresh_case["definition_match_current"] is None
    assert refresh_case["overall_benchmark_score"] == 0.443
    assert refresh_case["benchmark_missing_slots"] == 34.0
    assert refresh_case["claim_requirement_gap_count"] == 1
    assert refresh_case["decision"] == "diagnostic_fail"
    assert refresh_case["next_focus"][:3] == [
        "refresh_current_benchmark_definition",
        "fix_current_failure",
        "optimize_stage:text",
    ]
    assert refresh_case["run_dir"] == str(run_dir.resolve())
    assert payload["benchmark_score_trends"] == {
        "scored_trend_cases": 0,
        "comparable_cases": 0,
        "improved": 0,
        "regressed": 0,
        "unchanged": 0,
        "comparable_improved": 0,
        "comparable_regressed": 0,
        "comparable_unchanged": 0,
        "definition_matched": 0,
        "definition_mismatched": 0,
        "definition_missing": 0,
        "mean_score_delta": None,
        "all_score_delta_mean": None,
        "cases": [],
    }
    assert payload["latest_output_change_summary"] == {
        "diagnostic_only": 0,
        "output_risk": 1,
        "unknown": 0,
        "missing_audit": 0,
    }
    assert payload["score_priority_cases"][0]["case_id"] == "paper-1"
    assert payload["score_priority_cases"][0]["overall_benchmark_score"] == 0.443
    assert payload["score_priority_cases"][0]["benchmark_missing_slots"] == 34.0
    assert payload["score_priority_cases"][0]["claim_requirement_gap_count"] == 1
    latest = payload["latest_by_case"]["paper-1"]
    assert latest["output_change_diagnostic_only"] is False
    assert latest["output_change_output_risk_count"] == 1
    assert latest["output_change_unknown_count"] == 2
    assert latest["decision"] == "diagnostic_fail"
    assert latest["overall_benchmark_score"] == 0.443
    assert latest["benchmark_matched_slots"] == 27.0
    assert latest["benchmark_expected_slots"] == 61.0
    assert latest["benchmark_missing_slots"] == 34.0
    assert latest["claim_requirement_gap_count"] == 1
    assert latest["benchmark_gap_summary"] == "claim-001 (results): missing entities: gut dysbiosis"
    assert latest["detailed_analysis_html"] == "/tmp/run/paper-1/report.html"
    assert latest["slack_summary_markdown"] == "/tmp/run/paper-1/slack_summary.md"
    assert payload["focus_counts"]["fix_current_failure"] == 1
    assert payload["focus_counts"]["optimize_stage:text"] == 1
    assert "report_json" not in latest
    assert "summary_json" not in latest
    assert "evidence_packets" not in latest


def test_summarize_history_tracks_score_trends_between_iterations(tmp_path: Path, capsys) -> None:
    helper = _load_helper_module()
    runs = [
        (
            "first",
            {
                "generated_at": "2026-06-21T20:00:00Z",
                "surface": "web",
                "records": [
                    {
                        "case_id": "paper-1",
                        "decision": "diagnostic_fail",
                        "comparison": {
                            "comparison": {
                                "compatible": False,
                                "overall_benchmark_score": 0.4,
                                "benchmark_content_score_basis": {
                                    "matched_slots": 4,
                                    "expected_slots": 10,
                                },
                            }
                        },
                    }
                ],
            },
        ),
        (
            "second",
            {
                "generated_at": "2026-06-21T20:30:00Z",
                "surface": "web",
                "records": [
                    {
                        "case_id": "paper-1",
                        "decision": "diagnostic_fail",
                        "comparison": {
                            "comparison": {
                                "compatible": False,
                                "overall_benchmark_score": 0.7,
                                "benchmark_content_score_basis": {
                                    "matched_slots": 7,
                                    "expected_slots": 10,
                                },
                            }
                        },
                    }
                ],
            },
        ),
    ]
    for name, payload in runs:
        run_dir = tmp_path / name
        run_dir.mkdir()
        (run_dir / "active_benchmark_summary.json").write_text(
            __import__("json").dumps(payload),
            encoding="utf-8",
        )

    assert helper._summarize_history(tmp_path, json_output=True) == 0

    payload = __import__("json").loads(capsys.readouterr().out)
    trends = payload["benchmark_score_trends"]
    assert trends["scored_trend_cases"] == 1
    assert trends["comparable_cases"] == 0
    assert trends["improved"] == 1
    assert trends["regressed"] == 0
    assert trends["unchanged"] == 0
    assert trends["comparable_improved"] == 0
    assert trends["comparable_regressed"] == 0
    assert trends["comparable_unchanged"] == 0
    assert trends["definition_matched"] == 0
    assert trends["definition_mismatched"] == 0
    assert trends["definition_missing"] == 1
    assert trends["mean_score_delta"] is None
    assert trends["all_score_delta_mean"] == 0.3
    assert trends["cases"][0]["case_id"] == "paper-1"
    assert trends["cases"][0]["direction"] == "improved"
    assert trends["cases"][0]["previous_score"] == 0.4
    assert trends["cases"][0]["latest_score"] == 0.7
    assert trends["cases"][0]["score_delta"] == 0.3
    assert trends["cases"][0]["missing_slots_delta"] == -3.0
    assert trends["cases"][0]["benchmark_definition_match"] is None

    assert helper._summarize_history(tmp_path) == 0
    out = capsys.readouterr().out
    assert (
        "benchmark_score_trends: scored_trends=1 comparable=0 improved=1 regressed=0 unchanged=0 "
        "comparable_improved=0 comparable_regressed=0 comparable_unchanged=0 "
        "definition_matched=0 definition_mismatched=0 definition_missing=1 "
        "comparable_mean_delta=None all_mean_delta=0.3"
    ) in out
    assert "paper-1: improved 0.4 -> 0.7 delta=0.3 missing_slots_delta=-3.0 definition_match=None" in out


def test_score_trends_only_count_matching_benchmark_definitions_as_comparable() -> None:
    helper = _load_helper_module()
    trends = helper._score_trend_summary(
        {
            "matched-paper": [
                {
                    "overall_benchmark_score": 0.4,
                    "benchmark_missing_slots": 6,
                    "benchmark_definition_digest": "same",
                },
                {
                    "overall_benchmark_score": 0.6,
                    "benchmark_missing_slots": 4,
                    "benchmark_definition_digest": "same",
                },
            ],
            "missing-definition-paper": [
                {
                    "overall_benchmark_score": 0.4,
                    "benchmark_missing_slots": 6,
                },
                {
                    "overall_benchmark_score": 0.7,
                    "benchmark_missing_slots": 3,
                },
            ],
        }
    )

    assert trends["scored_trend_cases"] == 2
    assert trends["comparable_cases"] == 1
    assert trends["improved"] == 2
    assert trends["comparable_improved"] == 1
    assert trends["definition_matched"] == 1
    assert trends["definition_missing"] == 1
    assert trends["mean_score_delta"] == 0.2
    assert trends["all_score_delta_mean"] == 0.25


def test_record_history_writes_redacted_sqlite_rows(tmp_path: Path, capsys) -> None:
    helper = _load_helper_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "active_benchmark_summary.json").write_text(
        __import__("json").dumps(
            {
                "generated_at": "2026-06-21T20:12:59Z",
                "surface": "web",
                "benchmark_definition": {
                    "algorithm": "sha256",
                    "digest": "abc123",
                    "file_count": 13,
                    "files": [{"path": "benchmarks/multi_paper_benchmark.json", "sha256": "def456"}],
                },
                "output_change_audit": {
                    "diagnostic_only": True,
                    "changed_count": 4,
                    "output_risk_count": 0,
                    "unknown_count": 0,
                },
                "records": [
                    {
                        "case_id": "paper-1",
                        "decision": "pass",
                        "diagnostics": {
                            "latency_profile": {
                                "slowest_stage": "text",
                                "top_bottlenecks": [{"stage": "text", "duration_seconds": 529.913}],
                            }
                        },
                        "comparison": {
                            "comparison": {
                                "compatible": True,
                                "overall_benchmark_score": 0.941,
                                "benchmark_content_score": 0.941,
                                "benchmark_content_score_basis": {
                                    "matched_slots": 19,
                                    "expected_slots": 21,
                                },
                                "claim_requirement_gaps": [
                                    {
                                        "claim_id": "claim-001",
                                        "section": "methods",
                                        "missing_entities": ["healthy controls"],
                                    }
                                ],
                                "usable_packet_rate": 0.976,
                                "critical_claim_candidate_rate": 1.0,
                            }
                        },
                        "artifacts": {
                            "detailed_analysis_html": "/tmp/run/paper-1/report.html",
                            "detailed_analysis_url": "http://127.0.0.1:61311/report.html",
                            "webapp_detailed_analysis_url": "http://127.0.0.1:8000/web/?job_id=1&document_id=2&view=detailed_analysis",
                            "slack_summary_markdown": "/tmp/run/paper-1/slack_summary.md",
                            "media_json": "/tmp/run/paper-1/media.json",
                            "static_media_dir": "/tmp/run/paper-1/media",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    db_path = tmp_path / "history.sqlite"

    assert helper._record_history(tmp_path, db_path) == 0

    out = capsys.readouterr().out
    assert "recorded_rows: 1" in out
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            select
                case_id,
                decision,
                benchmark_definition_available,
                benchmark_definition_algorithm,
                benchmark_definition_digest,
                benchmark_definition_file_count,
                output_change_audit_available,
                output_change_diagnostic_only,
                output_change_changed_count,
                output_change_output_risk_count,
                output_change_unknown_count,
                compatible,
                overall_benchmark_score,
                benchmark_content_score,
                benchmark_matched_slots,
                benchmark_expected_slots,
                benchmark_missing_slots,
                claim_requirement_gap_count,
                benchmark_gap_summary,
                usable_packet_rate,
                critical_claim_candidate_rate,
                slowest_stage,
                slowest_stage_seconds,
                next_focus_json,
                detailed_analysis_html,
                detailed_analysis_url,
                webapp_detailed_analysis_url,
                slack_summary_markdown,
                media_json,
                static_media_dir
            from active_benchmark_history
            """
        ).fetchone()
        columns = [info[1] for info in conn.execute("pragma table_info(active_benchmark_history)")]

    assert row == (
        "paper-1",
        "pass",
        1,
        "sha256",
        "abc123",
        13,
        1,
        1,
        4,
        0,
        0,
        1,
        0.941,
        0.941,
        19.0,
        21.0,
        2.0,
        1,
        "claim-001 (methods): missing entities: healthy controls",
        0.976,
        1.0,
        "text",
        529.913,
        '["optimize_stage:text"]',
        "/tmp/run/paper-1/report.html",
        "http://127.0.0.1:61311/report.html",
        "http://127.0.0.1:8000/web/?job_id=1&document_id=2&view=detailed_analysis",
        "/tmp/run/paper-1/slack_summary.md",
        "/tmp/run/paper-1/media.json",
        "/tmp/run/paper-1/media",
    )
    assert "report_json" not in columns
    assert "summary_json" not in columns
    assert "evidence_packets" not in columns
    assert "detailed_analysis_html" in columns
    assert "slack_summary_markdown" in columns
    assert "media_json" in columns
    assert "static_media_dir" in columns
    assert "benchmark_definition_available" in columns
    assert "benchmark_definition_algorithm" in columns
    assert "benchmark_definition_digest" in columns
    assert "benchmark_definition_file_count" in columns
    assert "output_change_audit_available" in columns
    assert "output_change_diagnostic_only" in columns
    assert "output_change_changed_count" in columns
    assert "output_change_output_risk_count" in columns
    assert "output_change_unknown_count" in columns
    assert "overall_benchmark_score" in columns
    assert "benchmark_content_score" in columns
    assert "benchmark_matched_slots" in columns
    assert "benchmark_expected_slots" in columns
    assert "benchmark_missing_slots" in columns
    assert "claim_requirement_gap_count" in columns
    assert "benchmark_gap_summary" in columns


def test_history_rows_discovers_backfilled_artifacts_without_record_schema(tmp_path: Path) -> None:
    helper = _load_helper_module()
    run_dir = tmp_path / "run"
    case_dir = run_dir / "paper-1"
    case_dir.mkdir(parents=True)
    (run_dir / "active_benchmark_summary.json").write_text(
        __import__("json").dumps(
            {
                "generated_at": "2026-06-21T20:12:59Z",
                "surface": "web",
                "records": [{"case_id": "paper-1", "decision": "pass"}],
            }
        ),
        encoding="utf-8",
    )
    (case_dir / "report.html").write_text("<html></html>", encoding="utf-8")
    (case_dir / "slack_summary.md").write_text("*Paper*", encoding="utf-8")
    (case_dir / "media.json").write_text("{}", encoding="utf-8")
    (case_dir / "media").mkdir()

    rows = helper._history_rows(tmp_path)

    assert rows[0]["detailed_analysis_html"] == str(case_dir / "report.html")
    assert rows[0]["slack_summary_markdown"] == str(case_dir / "slack_summary.md")
    assert rows[0]["media_json"] == str(case_dir / "media.json")
    assert rows[0]["static_media_dir"] == str(case_dir / "media")


def test_plan_next_prefers_unrun_manifest_case(tmp_path: Path, capsys) -> None:
    helper = _load_helper_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "active_benchmark_summary.json").write_text(
        __import__("json").dumps(
            {
                "generated_at": "2026-06-21T20:12:59Z",
                "surface": "web",
                "records": [
                    {
                        "case_id": "sharma_2017_reward_deficits",
                        "decision": "pass",
                        "diagnostics": {
                            "latency_profile": {
                                "slowest_stage": "text",
                                "top_bottlenecks": [{"stage": "text", "duration_seconds": 529.913}],
                            }
                        },
                        "comparison": {
                            "comparison": {
                                "compatible": True,
                                "overall_benchmark_score": 0.941,
                                "usable_packet_rate": 0.976,
                                "critical_claim_candidate_rate": 1.0,
                            }
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert helper._plan_next_case(tmp_path, tier="all", case_ids=[]) == 0

    out = capsys.readouterr().out
    assert "selected_cases: 11" in out
    assert "prefer_needs_fix: False" in out
    assert "next_case: pmc7439296_alzheimer_mouse_model state=unrun" in out
    assert "--case pmc7439296_alzheimer_mouse_model" in out


def test_plan_next_prioritizes_failed_case_before_passed_case(tmp_path: Path, capsys) -> None:
    helper = _load_helper_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    records = []
    for case_id, decision in [
        ("sharma_2017_reward_deficits", "pass"),
        ("pmc7439296_alzheimer_mouse_model", "fail"),
    ]:
        records.append(
            {
                "case_id": case_id,
                "decision": decision,
                "failures": ["timeout"] if decision == "fail" else [],
                "diagnostics": {
                    "latency_profile": {
                        "slowest_stage": "text",
                        "top_bottlenecks": [{"stage": "text", "duration_seconds": 600.0}],
                    }
                },
                "comparison": {
                    "comparison": {
                        "compatible": decision == "pass",
                        "overall_benchmark_score": 0.941 if decision == "pass" else 0.5,
                        "benchmark_content_score_basis": {
                            "matched_slots": 7 if decision == "pass" else 4,
                            "expected_slots": 7,
                        },
                        "claim_requirement_gaps": []
                        if decision == "pass"
                        else [
                            {
                                "claim_id": "claim-001",
                                "section": "results",
                                "missing_entities": ["Morris water maze"],
                            }
                        ],
                    }
                },
            }
        )
    (run_dir / "active_benchmark_summary.json").write_text(
        __import__("json").dumps(
            {
                "generated_at": "2026-06-21T20:12:59Z",
                "surface": "web",
                "records": records,
            }
        ),
        encoding="utf-8",
    )

    assert helper._plan_next_case(tmp_path, tier="release", case_ids=["sharma_2017_reward_deficits", "pmc7439296_alzheimer_mouse_model"]) == 0

    out = capsys.readouterr().out
    assert "selected_cases: 2" in out
    assert "next_case: pmc7439296_alzheimer_mouse_model state=needs_fix" in out
    assert "score=0.5" in out
    assert "gaps=1" in out
    assert "missing_slots=3.0" in out
    assert (
        "pmc7439296_alzheimer_mouse_model: state=needs_fix decision=fail "
        "score=0.5 definition_current=None gaps=1 missing_slots=3.0"
    ) in out
    assert "next_focus: refresh_current_benchmark_definition, fix_current_failure" in out
    assert "latest_benchmark_gaps: claim-001 (results): missing entities: Morris water maze" in out
    assert "benchmark_gaps: claim-001 (results): missing entities: Morris water maze" in out


def test_plan_next_can_prioritize_needs_fix_before_unrun_cases(tmp_path: Path, capsys) -> None:
    helper = _load_helper_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "active_benchmark_summary.json").write_text(
        __import__("json").dumps(
            {
                "generated_at": "2026-06-21T20:12:59Z",
                "surface": "web",
                "records": [
                    {
                        "case_id": "pmc7439296_alzheimer_mouse_model",
                        "decision": "diagnostic_fail",
                        "failures": ["text timeout"],
                        "diagnostics": {
                            "latency_profile": {
                                "slowest_stage": "text",
                                "top_bottlenecks": [{"stage": "text", "duration_seconds": 360.0}],
                            }
                        },
                        "comparison": {
                            "comparison": {
                                "compatible": False,
                                "overall_benchmark_score": 0.377,
                                "benchmark_content_score_basis": {
                                    "matched_slots": 23,
                                    "expected_slots": 61,
                                },
                                "claim_requirement_gaps": [
                                    {
                                        "claim_id": "ad-microbiome-001",
                                        "section": "results",
                                        "missing_entities": ["5xFAD"],
                                    }
                                ],
                            }
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert helper._plan_next_case(tmp_path, tier="release", case_ids=[], prefer_needs_fix=True) == 0

    out = capsys.readouterr().out
    assert "prefer_needs_fix: True" in out
    assert "next_case: pmc7439296_alzheimer_mouse_model state=needs_fix" in out
    assert "score=0.377" in out
    assert "gaps=1" in out
    assert "missing_slots=38.0" in out
    assert "--case pmc7439296_alzheimer_mouse_model" in out
    assert "latest_benchmark_gaps: ad-microbiome-001 (results): missing entities: 5xFAD" in out


def test_plan_next_json_outputs_machine_readable_handoff(tmp_path: Path, capsys) -> None:
    helper = _load_helper_module()
    helper._current_output_change_audit = lambda: {
        "diagnostic_only": False,
        "changed_count": 3,
        "output_risk_count": 1,
        "unknown_count": 0,
        "benchmark_only": ["backend/tests/test_papereval_benchmark_helper.py"],
        "output_risk": ["backend/app/services/analysis/text_analysis.py"],
        "unknown": [],
        "output_risk_category_counts": {"backend_app": 1},
        "unknown_category_counts": {},
    }
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "active_benchmark_summary.json").write_text(
        __import__("json").dumps(
            {
                "generated_at": "2026-06-21T20:12:59Z",
                "surface": "web",
                "records": [
                    {
                        "case_id": "pmc7439296_alzheimer_mouse_model",
                        "decision": "diagnostic_fail",
                        "failures": ["evidence-to-gold comparison is incompatible"],
                        "diagnostics": {
                            "latency_profile": {
                                "slowest_stage": "text",
                                "top_bottlenecks": [{"stage": "text", "duration_seconds": 273.8}],
                            }
                        },
                        "comparison": {
                            "comparison": {
                                "compatible": False,
                                "overall_benchmark_score": 0.443,
                                "benchmark_content_score_basis": {
                                    "matched_slots": 27,
                                    "expected_slots": 61,
                                },
                                "claim_requirement_gaps": [
                                    {
                                        "claim_id": "ad-microbiome-001",
                                        "section": "results",
                                        "missing_entities": ["gut dysbiosis"],
                                    }
                                ],
                            }
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert helper._plan_next_case(
        tmp_path,
        tier="all",
        case_ids=[],
        prefer_needs_fix=True,
        json_output=True,
    ) == 0

    payload = __import__("json").loads(capsys.readouterr().out)
    assert payload["prefer_needs_fix"] is True
    assert payload["output_change_audit"] == {
        "available": True,
        "diagnostic_only": False,
        "changed_count": 3,
        "output_risk_count": 1,
        "unknown_count": 0,
        "output_risk_category_counts": {"backend_app": 1},
        "unknown_category_counts": {},
    }
    assert payload["selected_cases"] == 11
    assert len(payload["planned_cases"]) == payload["selected_cases"]
    assert len(payload["queue_preview"]) == 10
    assert payload["state_counts"] == {"needs_fix": 1, "unrun": 10}
    assert payload["score_priority_cases"][0]["case_id"] == "pmc7439296_alzheimer_mouse_model"
    assert payload["score_priority_cases"][0]["overall_benchmark_score"] == 0.443
    assert payload["score_priority_cases"][0]["state"] == "needs_fix"
    assert payload["next_case"]["case_id"] == "pmc7439296_alzheimer_mouse_model"
    assert payload["next_case"]["state"] == "needs_fix"
    assert payload["next_case"]["latest_overall_benchmark_score"] == 0.443
    assert payload["next_case"]["latest_benchmark_definition_match_current"] is None
    assert payload["next_case"]["next_focus"][0] == "refresh_current_benchmark_definition"
    assert payload["current_definition_refresh_cases"][0]["case_id"] == "pmc7439296_alzheimer_mouse_model"
    assert payload["current_definition_refresh_cases"][0]["reason"] == "missing_definition"
    assert payload["next_case"]["latest_benchmark_missing_slots"] == 34.0
    assert payload["next_case"]["latest_claim_requirement_gap_count"] == 1
    assert payload["next_case"]["latest_benchmark_gap_summary"] == (
        "ad-microbiome-001 (results): missing entities: gut dysbiosis"
    )
    assert "--case" in payload["next_case"]["suggested_command"]
    assert "pmc7439296_alzheimer_mouse_model" in payload["next_case"]["suggested_command"]
    assert "--require-diagnostic-only" not in payload["next_case"]["suggested_command"]
    assert "--require-diagnostic-only" in payload["next_case"]["suggested_diagnostic_only_command"]
    assert payload["queue_preview"][0]["case_id"] == "pmc7439296_alzheimer_mouse_model"
    assert payload["planned_cases"][0]["case_id"] == "pmc7439296_alzheimer_mouse_model"


def test_plan_next_can_prioritize_lowest_benchmark_score(tmp_path: Path, capsys) -> None:
    helper = _load_helper_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "active_benchmark_summary.json").write_text(
        __import__("json").dumps(
            {
                "generated_at": "2026-06-21T20:12:59Z",
                "surface": "web",
                "records": [
                    {
                        "case_id": "pmc7439296_alzheimer_mouse_model",
                        "decision": "diagnostic_fail",
                        "comparison": {
                            "comparison": {
                                "compatible": False,
                                "overall_benchmark_score": 0.443,
                                "benchmark_content_score_basis": {
                                    "matched_slots": 27,
                                    "expected_slots": 61,
                                },
                                "claim_requirement_gaps": [
                                    {
                                        "claim_id": "ad-microbiome-001",
                                        "section": "results",
                                        "missing_entities": ["gut dysbiosis"],
                                    }
                                ],
                            }
                        },
                    },
                    {
                        "case_id": "pmc7437112_covid_logistic_growth",
                        "decision": "diagnostic_fail",
                        "failures": ["Local model text analysis did not run."],
                        "diagnostics": {
                            "report_invalid_reason": "Local model text analysis did not run.",
                            "latency_profile": {
                                "quality_flags": ["text_cache_hit", "model_calls_zero"],
                            },
                        },
                        "comparison": {
                            "comparison": {
                                "compatible": False,
                                "overall_benchmark_score": 0.0,
                                "benchmark_content_score_basis": {
                                    "matched_slots": 0,
                                    "expected_slots": 20,
                                },
                                "claim_requirement_gaps": [
                                    {
                                        "claim_id": "covid-logistic-supp-001",
                                        "section": "methods",
                                        "missing_entities": ["Supplementary Material"],
                                    }
                                ],
                            }
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    assert helper._plan_next_case(
        tmp_path,
        tier="all",
        case_ids=[],
        prefer_needs_fix=True,
        prefer_lowest_score=True,
        json_output=True,
    ) == 0

    payload = __import__("json").loads(capsys.readouterr().out)
    assert payload["prefer_needs_fix"] is True
    assert payload["prefer_lowest_score"] is True
    assert payload["next_case"]["case_id"] == "pmc7437112_covid_logistic_growth"
    assert payload["next_case"]["latest_overall_benchmark_score"] == 0.0
    assert "fix_current_failure" in payload["next_case"]["next_focus"]
    assert "resolve_text_cache_validity_conflict" in payload["next_case"]["next_focus"]
    assert "--disable-local-text-cache" in payload["next_case"]["suggested_fresh_text_command"]
    assert "--require-diagnostic-only" in payload["next_case"]["suggested_fresh_text_diagnostic_only_command"]
    assert payload["queue_preview"][0]["case_id"] == "pmc7437112_covid_logistic_growth"
    assert payload["score_priority_cases"][0]["case_id"] == "pmc7437112_covid_logistic_growth"


def test_plan_next_can_prioritize_current_definition_refresh(tmp_path: Path, capsys) -> None:
    helper = _load_helper_module()
    current_digest = helper._benchmark_definition_fingerprint()["digest"]
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "active_benchmark_summary.json").write_text(
        __import__("json").dumps(
            {
                "generated_at": "2026-06-21T20:12:59Z",
                "surface": "web",
                "benchmark_definition": {
                    "algorithm": "sha256",
                    "digest": "older-target",
                    "file_count": 13,
                    "files": [],
                },
                "records": [
                    {
                        "case_id": "pmc7439296_alzheimer_mouse_model",
                        "decision": "diagnostic_fail",
                        "comparison": {
                            "comparison": {
                                "compatible": False,
                                "overall_benchmark_score": 0.443,
                                "benchmark_content_score_basis": {
                                    "matched_slots": 27,
                                    "expected_slots": 61,
                                },
                            }
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert helper._plan_next_case(
        tmp_path,
        tier="all",
        case_ids=[],
        prefer_current_definition_refresh=True,
        json_output=True,
    ) == 0

    payload = __import__("json").loads(capsys.readouterr().out)
    assert payload["prefer_current_definition_refresh"] is True
    assert payload["current_benchmark_definition_summary"]["digest"] == current_digest
    assert payload["next_case"]["case_id"] == "pmc7439296_alzheimer_mouse_model"
    assert payload["next_case"]["latest_benchmark_definition_digest"] == "older-target"
    assert payload["next_case"]["latest_benchmark_definition_match_current"] is False
    assert payload["next_case"]["next_focus"][0] == "refresh_current_benchmark_definition"
    assert payload["current_definition_refresh_cases"][0]["case_id"] == "pmc7439296_alzheimer_mouse_model"
    assert payload["current_definition_refresh_cases"][0]["reason"] == "definition_mismatch"


def test_plan_next_human_output_includes_current_definition_refresh_preview(
    tmp_path: Path,
    capsys,
) -> None:
    helper = _load_helper_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "active_benchmark_summary.json").write_text(
        __import__("json").dumps(
            {
                "generated_at": "2026-06-21T20:12:59Z",
                "surface": "web",
                "records": [
                    {
                        "case_id": "pmc7439296_alzheimer_mouse_model",
                        "decision": "diagnostic_fail",
                        "comparison": {
                            "comparison": {
                                "compatible": False,
                                "overall_benchmark_score": 0.443,
                                "benchmark_content_score_basis": {
                                    "matched_slots": 27,
                                    "expected_slots": 61,
                                },
                                "claim_requirement_gaps": [
                                    {
                                        "claim_id": "ad-microbiome-001",
                                        "section": "results",
                                        "missing_entities": ["gut dysbiosis"],
                                    }
                                ],
                            }
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert helper._plan_next_case(tmp_path, tier="all", case_ids=[]) == 0

    output = capsys.readouterr().out
    assert "Current definition refresh preview:" in output
    assert (
        "- pmc7439296_alzheimer_mouse_model: reason=missing_definition "
        "score=0.443 missing_slots=34.0 gaps=1 decision=diagnostic_fail"
    ) in output


def test_summarize_stage_diagnostics_aggregates_latest_saved_reports(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    helper = _load_helper_module()
    case_id = "case_a"
    run_dir = tmp_path / "run"
    case_dir = run_dir / case_id
    case_dir.mkdir(parents=True)
    report_path = case_dir / "report.json"
    report_path.write_text("{}", encoding="utf-8")
    gold_path = tmp_path / "gold.json"
    gold_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        helper,
        "_history_rows",
        lambda path: [
            {
                "case_id": case_id,
                "run_dir": str(run_dir),
                "generated_at": "2026-06-21T22:26:44Z",
            }
        ],
    )

    def fake_load_json(path: Path):
        if path == helper.MANIFEST:
            return {"cases": [{"id": case_id, "gold_standard": str(gold_path)}]}
        if path == report_path:
            return {
                "artifact_organization": {
                    "schema_version": 1,
                    "quality_flags": ["selected_prompt_details_missing_source_excerpt"],
                    "audited_packet_quality": {"total": 2, "typed_packet_count": 2},
                    "supplement_source_consistency": {"missing_supplement_ref_count": 0},
                    "llm_input_inventory": {
                        "schema_version": 1,
                        "selected_quality": {"missing_source_excerpt": 1},
                        "quality_flags": ["selected_prompt_details_missing_source_excerpt"],
                    },
                }
            }
        return {}

    monkeypatch.setattr(helper, "_load_json", fake_load_json)
    fake_validator = SimpleNamespace(load_gold_standard=lambda path: {"case_id": case_id})
    fake_comparator = SimpleNamespace(
        evidence_packets_from_payload=lambda payload: [],
        evidence_metadata_from_payload=lambda payload: {},
        compare_evidence_to_gold=lambda packets, gold, evidence_metadata=None: {
            "overall_benchmark_score": 0.25,
            "benchmark_content_score_basis": {"matched_slots": 1, "expected_slots": 4},
        },
        _build_artifact_stage_diagnostics=lambda report, gold, comparison: {
            "stage_presence_counts": {
                "extracted_text": {"present": 1, "total_expected_items": 3},
                "evidence_packets": {"present": 1, "total_expected_items": 3},
                "synthesis_inputs": {"present": 0, "total_expected_items": 3},
                "final_report": {"present": 0, "total_expected_items": 3},
            },
            "failure_point_counts": {
                "absent_from_saved_artifact": 2,
                "dropped_before_synthesis_selection": 1,
            },
            "item_type_counts": {"entity": 2, "number": 1},
            "failure_point_by_item_type": {
                "entity": {"absent_from_saved_artifact": 2},
                "number": {"dropped_before_synthesis_selection": 1},
            },
            "missing_items": [
                {
                    "claim_id": "claim-1",
                    "item_type": "entity",
                    "term": "missing source entity",
                    "failure_point": "absent_from_saved_artifact",
                    "improvement_lane": "source_recall_or_extraction_visibility",
                    "source_visibility": {
                        "classification": "weak_term_candidate",
                        "term_score": 0.5,
                    },
                    "diagnostic_reason": "No exact diagnostic match was found.",
                    "nearest_stage_candidates": {
                        "extracted_text": [
                            {
                                "path": "summary.extractive_evidence.methods.0",
                                "score": 0.5,
                                "snippet": "nearby source text",
                            }
                        ]
                    },
                },
                {
                    "claim_id": "claim-2",
                    "item_type": "number",
                    "term": "20 mg",
                    "failure_point": "dropped_before_synthesis_selection",
                    "improvement_lane": "synthesis_evidence_selection_or_ranking",
                    "source_visibility": {
                        "classification": "exact_present",
                        "term_score": 1.0,
                    },
                    "diagnostic_reason": "Present in evidence packets but not synthesis inputs.",
                    "stage_matches": {
                        "evidence_packets": [
                            {
                                "path": "summary.modalities.text.findings.0",
                                "snippet": "20 mg was present in packet text",
                            }
                        ]
                    },
                },
            ],
        },
    )

    def fake_load_module(path: Path, name: str):
        return fake_comparator if path.name == "compare_evidence_to_gold.py" else fake_validator

    monkeypatch.setattr(helper, "_load_module", fake_load_module)

    assert helper._summarize_stage_diagnostics(tmp_path) == 0

    output = capsys.readouterr().out
    assert "cases_analyzed: 1" in output
    assert "- absent_from_saved_artifact: 2" in output
    assert "- dropped_before_synthesis_selection: 1" in output
    assert "- source_recall_or_extraction_visibility: 1" in output
    assert "- synthesis_evidence_selection_or_ranking: 1" in output
    assert "- exact_present: 1" in output
    assert "- weak_term_candidate: 1" in output
    assert "component_candidate_summary:" in output
    assert "artifact_organization_summary:" in output
    assert "- native_cases: 1/1" in output
    assert (
        "- llm_input_inventory: 1/1 quality=missing_source_excerpt=1 "
        "flags=selected_prompt_details_missing_source_excerpt=1"
    ) in output
    assert (
        "- normalization / alias handling: weighted_items=1 "
        "causal=needs_manual_confirmation"
    ) in output
    assert (
        "- synthesis focus-slot selection: weighted_items=1 "
        "causal=candidate_causal"
    ) in output
    assert "source_recall_drilldown:" in output
    assert "- total_items: 1" in output
    assert "- by_visibility: weak_term_candidate=1" in output
    assert "- by_db_visibility: db_not_checked=1" in output
    assert "- by_pdf_visibility: pdf_not_checked=1" in output
    assert "- by_chunk_payload_visibility: " in output
    assert "- by_loss_mode: db_or_pdf_not_checked=1" in output
    assert (
        "case_a: items=1 visibility=weak_term_candidate=1 "
        "db=db_not_checked=1 pdf=pdf_not_checked=1 "
        "chunk_payload= loss=db_or_pdf_not_checked=1"
    ) in output
    assert "loss_mode_examples:" in output
    assert "- db_or_pdf_not_checked:" in output
    assert (
        "case_a claim-1 entity=missing source entity "
        "db=db_not_checked pdf=pdf_not_checked"
    ) in output
    assert "recommended_inspection_queue:" in output
    assert (
        "source_recall_or_extraction_visibility: items=1 visibility=weak_term_candidate=1"
    ) in output
    assert "inspect: Inspect normalization, aliases, OCR cleanup" in output
    assert "do not add paper-specific deterministic expected-fact rules" in output
    assert "components: normalization / alias handling" in output
    assert (
        "synthesis_evidence_selection_or_ranking: items=1 visibility=exact_present=1"
    ) in output
    assert "components: synthesis focus-slot selection, synthesis evidence plan coverage" in output
    assert "case_a: score=0.25 missing_items=3" in output
    assert (
        "example: source_recall_or_extraction_visibility claim-1 entity=missing source entity "
        "failure=absent_from_saved_artifact visibility=weak_term_candidate:0.5"
    ) in output
    assert "trace: extracted_text summary.extractive_evidence.methods.0: nearby source text" in output
    assert (
        "example: synthesis_evidence_selection_or_ranking claim-2 number=20 mg "
        "failure=dropped_before_synthesis_selection visibility=exact_present:1.0"
    ) in output
    assert "trace: evidence_packets summary.modalities.text.findings.0: 20 mg was present in packet text" in output

    assert helper._summarize_stage_diagnostics(tmp_path, json_output=True) == 0
    payload = capsys.readouterr().out
    assert '"aggregate_lane_source_visibility_counts"' in payload
    assert '"recommended_inspection_queue"' in payload
    assert '"component_candidate_summary"' in payload
    assert '"source_recall_drilldown"' in payload
    assert '"source_recall_item_type_counts"' in payload
    assert '"source_recall_db_visibility_counts"' in payload
    assert '"source_recall_pdf_visibility_counts"' in payload
    assert '"source_recall_chunk_payload_visibility_counts"' in payload
    assert '"llm_input_inventory_cases": 1' in payload
    assert '"llm_input_inventory_quality_flag_counts"' in payload
    assert '"by_loss_mode"' in payload
    assert '"by_pdf_visibility"' in payload
    assert '"by_chunk_payload_visibility"' in payload
    assert '"examples_by_loss_mode"' in payload
    assert '"loss_mode_counts"' in payload
    assert '"db_not_checked": 1' in payload
    assert '"pdf_not_checked": 1' in payload
    assert '"db_or_pdf_not_checked": 1' in payload
    assert '"db_visibility": "db_not_checked"' in payload
    assert '"pdf_visibility": "pdf_not_checked"' in payload
    assert '"source_pdf"' in payload
    assert '"weighted_items": 1' in payload
    assert '"component_candidates"' in payload
    assert '"causal_status": "candidate_causal"' in payload
    assert '"causal_status": "needs_manual_confirmation"' in payload
    assert '"example_cases"' in payload


def test_source_recall_pdf_visibility_helpers_classify_loss_modes() -> None:
    helper = _load_helper_module()
    item = {"term": "Bergson-type welfare functions"}
    pdf_match = helper._pdf_match_for_stage_item(
        "The paper discusses Bergson-type welfare functions and QALY assumptions.",
        item,
    )

    assert pdf_match["status"] == "pdf_exact_term_present"
    assert "Bergson-type welfare functions" in pdf_match["trace"]
    range_match = helper._pdf_match_for_stage_item(
        "The model compares 20–49 years, 50–64 years, and 65 and over.",
        {"term": "20-49 years"},
    )
    assert range_match["status"] == "pdf_exact_term_present"
    assert range_match["trace"].startswith("pdf(normalized):")
    assert (
        helper._source_recall_loss_mode("db_not_found", "pdf_exact_term_present")
        == "pdf_text_present_but_chunk_missing"
    )
    assert (
        helper._source_recall_loss_mode("db_not_found", "pdf_not_found")
        == "pdf_text_absent_or_alias_needed"
    )
    assert (
        helper._source_recall_loss_mode("db_exact_term_present", "pdf_exact_term_present")
        == "parsed_chunk_present_lost_before_report_artifact"
    )

    examples = helper._source_recall_examples_by_loss_mode(
        [
            {
                "case_id": "case_a",
                "examples": [
                    {
                        "claim_id": "claim-1",
                        "item_type": "entity",
                        "term": "Bergson-type welfare functions",
                        "db_visibility": "db_not_found",
                        "pdf_visibility": "pdf_exact_term_present",
                        "chunk_payload_visibility": "report_chunk_payload_missing",
                        "pdf_trace": "pdf: ...Bergson-type welfare functions...",
                    }
                ],
            }
        ],
        {"pdf_text_present_but_chunk_missing": 1},
    )
    assert examples["pdf_text_present_but_chunk_missing"][0]["case_id"] == "case_a"
    assert examples["pdf_text_present_but_chunk_missing"][0]["trace"].startswith("pdf:")
    assert (
        examples["pdf_text_present_but_chunk_missing"][0]["chunk_payload_visibility"]
        == "report_chunk_payload_missing"
    )
    assert (
        helper._chunk_payload_match_for_stage_item(
            ["The model uses Bergson-type welfare functions."],
            item,
        )
        == "report_chunk_payload_contains_term"
    )
    assert (
        helper._chunk_payload_match_for_stage_item([], item)
        == "report_chunk_payload_missing"
    )


def test_source_chunk_inventory_sidecar_is_bounded_and_searchable(tmp_path: Path) -> None:
    helper = _load_helper_module()
    db_path = tmp_path / "app.db"
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            """
            create table chunk (
                id integer primary key,
                document_id integer not null,
                anchor text,
                modality text,
                content text
            )
            """
        )
        conn.execute(
            "insert into chunk (id, document_id, anchor, modality, content) values (?, ?, ?, ?, ?)",
            (
                1,
                7,
                "section:Methods:1",
                "text",
                "The model uses Bergson-type welfare functions and QALY assumptions.",
            ),
        )
        conn.commit()

    sidecar_path = helper._write_source_chunk_inventory(
        {"document": {"id": 7}},
        tmp_path / "source_chunks.json",
        db_path=db_path,
        max_chunks=1,
        max_chars=32,
    )

    assert sidecar_path == tmp_path / "source_chunks.json"
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["chunk_count"] == 1
    assert payload["stored_chunk_count"] == 1
    assert payload["chunks"][0]["content_sha256"]
    assert payload["chunks"][0]["excerpt"] == "The model uses Bergson-type welf"
    assert helper._source_chunk_inventory_texts(sidecar_path) == [
        "The model uses Bergson-type welf"
    ]
    assert (
        helper._chunk_payload_match_for_stage_item(
            helper._source_chunk_inventory_texts(sidecar_path),
            {"term": "Bergson-type"},
        )
        == "report_chunk_payload_contains_term"
    )
    stage_index_path = tmp_path / "intermediate_stage_index.json"
    llm_inventory_path = tmp_path / "llm_input_inventory.json"
    metadata = helper._artifact_metadata(
        report_path=tmp_path / "report.json",
        html_path=tmp_path / "report.html",
        slack_summary_path=tmp_path / "slack_summary.md",
        source_chunks_json_path=sidecar_path,
        intermediate_stage_index_json_path=stage_index_path,
        llm_input_inventory_json_path=llm_inventory_path,
    )
    assert metadata["source_chunks_json"] == str(sidecar_path)
    assert metadata["intermediate_stage_index_json"] == str(stage_index_path)
    assert metadata["llm_input_inventory_json"] == str(llm_inventory_path)


def test_write_detailed_analysis_html_is_reader_first_and_links_webapp(tmp_path: Path) -> None:
    helper = _load_helper_module()
    report = {
        "document": {"id": 181},
        "summary_json": {
            "paper_meta": {
                "title": "Test Paper",
                "authors": ["A. Author", "B. Author"],
                "journal": "Journal",
                "date": "2026-01-01",
            },
            "executive_summary": "Introduction: Why it matters.\nMethods: What was done.",
            "scientific_details": ["Statistical Result / Results: primary endpoint improved."],
            "sections": {
                "methods": {
                    "items": [{"statement": "Participants completed baseline imaging."}],
                    "evidence_refs": ["section:Methods:1"],
                },
                "results": {
                    "items": [{"statement": "The main result was detected."}],
                    "evidence_refs": ["section:Results:2"],
                },
            },
            "table_results": [{"result": "Table 1 summarized baseline characteristics.", "evidence": ["table:1"]}],
            "figure_results": [{"result": "Figure 1 showed the primary effect.", "evidence": ["figure:1"]}],
            "coverage": {"supp_tables": {"expected": 2, "extracted": 0, "missing_refs": ["S1", "S2"]}},
        },
    }

    html_path = helper._write_detailed_analysis_html(
        report,
        tmp_path / "report.html",
        webapp_url="http://127.0.0.1:8000/web/?job_id=1&document_id=2&view=detailed_analysis",
        media={
            "figures": [
                {
                    "chunk_id": 11,
                    "anchor": "figure:1",
                    "legend": "Figure 1 showed the primary effect.",
                    "page": 3,
                    "asset_kind": "main",
                    "static_image_url": "media/figure-01-11.png",
                    "source_proxy_url": "/api/documents/2/media/11/source",
                }
            ],
            "tables": [
                {
                    "chunk_id": 12,
                    "anchor": "table:1",
                    "legend": "Table 1 summarized baseline characteristics.",
                    "table_preview": {"columns": ["Group", "N"], "rows": [["Control", "53"]]},
                }
            ],
        },
        api_base="http://127.0.0.1:8000/api",
        comparison={
            "comparison": {
                "compatible": True,
                "overall_benchmark_score": 0.905,
                "benchmark_content_score_basis": {
                    "matched_slots": 19,
                    "expected_slots": 21,
                    "components": {
                        "critical_claim_candidates": {"matched": 3, "expected": 3},
                        "expected_entities": {"matched": 9, "expected": 11},
                        "expected_numbers": {"matched": 1, "expected": 1},
                        "expected_detail_types": {"matched": 4, "expected": 4},
                        "required_sections": {"matched": 2, "expected": 2},
                    },
                },
                "claim_requirement_gaps": [
                    {
                        "claim_id": "claim-001",
                        "section": "methods",
                        "missing_entities": ["healthy controls"],
                        "missing_numbers": [],
                        "missing_detail_types": [],
                    }
                ],
            }
        },
    )

    html = html_path.read_text(encoding="utf-8")
    assert "Open in PaperEval Webapp" in html
    assert "Benchmark Score" in html
    assert ">0.905<" in html
    assert "Matched 19 of 21 expected benchmark content slots." in html
    assert "Expected entities 9/11" in html
    assert "claim-001 (methods): missing entities: healthy controls" in html
    assert "Executive Summary" in html
    assert "Participants completed baseline imaging." in html
    assert "Table 1 summarized baseline characteristics." in html
    assert "Missing 2 supplementary tables: S1, S2." in html
    assert "Embedded Figures" in html
    assert 'src="media/figure-01-11.png"' in html
    assert "http://127.0.0.1:8000/api/documents/2/media/11/source" in html
    assert "<td>Control</td>" in html
    assert html.index("Executive Summary") < html.index("Report Metadata")


def test_slack_summary_markdown_contains_summary_and_links(tmp_path: Path) -> None:
    helper = _load_helper_module()
    report = {
        "summary_json": {
            "paper_meta": {"title": "Test Paper"},
            "executive_summary": "Introduction: short summary.",
        }
    }

    path = helper._write_slack_summary_markdown(
        report,
        tmp_path / "slack_summary.md",
        detailed_analysis_url="http://127.0.0.1:61311/report.html",
        webapp_url="http://127.0.0.1:8000/web/?job_id=1&document_id=2&view=detailed_analysis",
        comparison={
            "comparison": {
                "overall_benchmark_score": 0.905,
                "benchmark_content_score_basis": {
                    "matched_slots": 19,
                    "expected_slots": 21,
                    "components": {
                        "critical_claim_candidates": {"matched": 3, "expected": 3},
                        "expected_entities": {"matched": 9, "expected": 11},
                    },
                },
                "claim_requirement_gaps": [
                    {
                        "claim_id": "claim-001",
                        "section": "methods",
                        "missing_entities": ["healthy controls"],
                    }
                ],
            }
        },
    )

    text = path.read_text(encoding="utf-8")
    assert "*Test Paper*" in text
    assert "*Introduction*\nshort summary." in text
    assert "*Benchmark score*: 0.905 (19/21 expected content slots)" in text
    assert "Score components: Claim candidates 3/3; Expected entities 9/11" in text
    assert "Benchmark gaps: claim-001 (methods): missing entities: healthy controls" in text
    assert "Static detailed analysis: http://127.0.0.1:61311/report.html" in text
    assert "Open in PaperEval webapp: http://127.0.0.1:8000/web/?job_id=1&document_id=2&view=detailed_analysis" in text


def test_write_detailed_report_from_path_backfills_html_and_slack_summary(tmp_path: Path, capsys) -> None:
    helper = _load_helper_module()
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "report.json").write_text(
        __import__("json").dumps(
            {
                "summary_json": {
                    "paper_meta": {"title": "Backfill Paper"},
                    "executive_summary": "Introduction: cached report summary.",
                    "sections": {"introduction": {"items": [{"statement": "Background line."}]}},
                }
            }
        ),
        encoding="utf-8",
    )

    assert helper._write_detailed_report_from_path(
        case_dir,
        detailed_analysis_url="http://127.0.0.1:61311/report.html",
        webapp_url="http://127.0.0.1:8000/web/?job_id=9&document_id=10&view=detailed_analysis",
    ) == 0

    assert (case_dir / "report.html").exists()
    assert (case_dir / "slack_summary.md").exists()
    html = (case_dir / "report.html").read_text(encoding="utf-8")
    slack = (case_dir / "slack_summary.md").read_text(encoding="utf-8")
    out = capsys.readouterr().out
    assert "detailed_analysis_html=" in out
    assert "slack_summary_markdown=" in out
    assert "Open in PaperEval Webapp" in html
    assert "*Introduction*\ncached report summary." in slack


def test_write_detailed_report_backfills_record_and_summary_artifacts(tmp_path: Path) -> None:
    helper = _load_helper_module()
    run_dir = tmp_path / "run"
    case_dir = run_dir / "paper-1"
    case_dir.mkdir(parents=True)
    (case_dir / "report.json").write_text(
        __import__("json").dumps(
            {
                "summary_json": {
                    "paper_meta": {"title": "Backfill Paper"},
                    "executive_summary": "Introduction: cached report summary.",
                }
            }
        ),
        encoding="utf-8",
    )
    (case_dir / "record.json").write_text(
        __import__("json").dumps({"case_id": "paper-1", "decision": "pass"}),
        encoding="utf-8",
    )
    (run_dir / "active_benchmark_summary.json").write_text(
        __import__("json").dumps(
            {
                "generated_at": "2026-06-21T20:12:59Z",
                "surface": "web",
                "records": [{"case_id": "paper-1", "decision": "pass"}],
            }
        ),
        encoding="utf-8",
    )

    assert helper._write_detailed_report_from_path(
        case_dir,
        detailed_analysis_url="http://127.0.0.1:61311/paper-1/report.html",
        webapp_url="http://127.0.0.1:8000/web/?job_id=9&document_id=10&view=detailed_analysis",
        media_json="",
        fetch_media_assets=False,
        api_base="",
    ) == 0

    record = __import__("json").loads((case_dir / "record.json").read_text(encoding="utf-8"))
    summary = __import__("json").loads((run_dir / "active_benchmark_summary.json").read_text(encoding="utf-8"))
    for artifacts in (record["artifacts"], summary["records"][0]["artifacts"]):
        assert artifacts["report_json"] == str(case_dir / "report.json")
        assert artifacts["detailed_analysis_html"] == str(case_dir / "report.html")
        assert artifacts["slack_summary_markdown"] == str(case_dir / "slack_summary.md")
        assert artifacts["detailed_analysis_url"] == "http://127.0.0.1:61311/paper-1/report.html"
        assert artifacts["webapp_detailed_analysis_url"] == (
            "http://127.0.0.1:8000/web/?job_id=9&document_id=10&view=detailed_analysis"
        )


def test_write_detailed_report_refreshes_existing_comparison_metadata(tmp_path: Path, monkeypatch) -> None:
    helper = _load_helper_module()
    monkeypatch.setattr(
        helper,
        "_compare_report_to_gold",
        lambda case, report: {
            "comparison": {
                "compatible": True,
                "overall_benchmark_score": 0.75,
                "benchmark_content_score": 0.75,
                "usable_packet_rate": 1.0,
                "critical_claim_candidate_rate": 1.0,
            }
        },
    )
    run_dir = tmp_path / "run"
    case_dir = run_dir / "sharma_2017_reward_deficits"
    case_dir.mkdir(parents=True)
    (case_dir / "report.json").write_text(
        __import__("json").dumps(
            {
                "summary_json": {
                    "paper_meta": {"title": "Backfill Paper"},
                    "executive_summary": "Introduction: cached report summary.",
                }
            }
        ),
        encoding="utf-8",
    )
    (case_dir / "record.json").write_text(
        __import__("json").dumps({"case_id": "sharma_2017_reward_deficits", "decision": "pass"}),
        encoding="utf-8",
    )
    (run_dir / "active_benchmark_summary.json").write_text(
        __import__("json").dumps(
            {
                "generated_at": "2026-06-21T20:12:59Z",
                "surface": "web",
                "records": [{"case_id": "sharma_2017_reward_deficits", "decision": "pass"}],
            }
        ),
        encoding="utf-8",
    )

    assert helper._write_detailed_report_from_path(case_dir) == 0

    record = __import__("json").loads((case_dir / "record.json").read_text(encoding="utf-8"))
    summary = __import__("json").loads((run_dir / "active_benchmark_summary.json").read_text(encoding="utf-8"))
    for item in (record, summary["records"][0]):
        assert item["comparison"]["comparison"]["overall_benchmark_score"] == 0.75
        assert item["iteration_diagnostics"]["overall_benchmark_score"] == 0.75


def test_write_detailed_report_uses_existing_media_json(tmp_path: Path) -> None:
    helper = _load_helper_module()
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "report.json").write_text(
        __import__("json").dumps(
            {
                "summary_json": {
                    "paper_meta": {"title": "Media Paper"},
                    "executive_summary": "Results: media available.",
                }
            }
        ),
        encoding="utf-8",
    )
    (case_dir / "media.json").write_text(
        __import__("json").dumps(
            {
                "figures": [
                    {
                        "chunk_id": 7,
                        "anchor": "figure:2",
                        "legend": "Figure 2 displayed extracted media.",
                        "static_image_url": "media/figure-01-7.png",
                    }
                ],
                "tables": [],
            }
        ),
        encoding="utf-8",
    )

    assert helper._write_detailed_report_from_path(case_dir) == 0

    html = (case_dir / "report.html").read_text(encoding="utf-8")
    assert "Embedded Figures" in html
    assert "Figure 2 displayed extracted media." in html
    assert 'src="media/figure-01-7.png"' in html


def test_webapp_detailed_analysis_url_uses_job_and_document_ids() -> None:
    helper = _load_helper_module()

    assert helper._webapp_detailed_analysis_url(
        "http://127.0.0.1:8000/api",
        {"job_id": 175, "document_id": 181},
    ) == "http://127.0.0.1:8000/web/?job_id=175&document_id=181&view=detailed_analysis"


def test_surface_preflight_distinguishes_web_and_desktop(monkeypatch) -> None:
    helper = _load_helper_module()
    monkeypatch.setattr(helper, "_url_ready", lambda url, *, timeout=5.0: False)

    web_failures = helper._surface_preflight_failures(
        "http://127.0.0.1:8000/api",
        {"frontend_target": 7777},
        _args(surface="web"),
    )
    desktop_failures = helper._surface_preflight_failures(
        "http://127.0.0.1:8000/api",
        {},
        _args(surface="desktop", start_app=True),
    )

    assert web_failures == ["web frontend is not ready at http://127.0.0.1:8000/web/"]
    assert desktop_failures == [
        "--surface desktop requires an existing/launched desktop app, not --start-app api-only mode"
    ]
