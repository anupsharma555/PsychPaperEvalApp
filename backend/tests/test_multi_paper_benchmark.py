from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_multi_paper_benchmark.py"
GOLD_VALIDATOR_PATH = PROJECT_ROOT / "scripts" / "validate_gold_standards.py"
MANIFEST_PATH = PROJECT_ROOT / "benchmarks" / "multi_paper_benchmark.json"
EXPECTATIONS_PATH = PROJECT_ROOT / "benchmarks" / "ten_paper_expectations.json"


def _load_runner_module():
    spec = importlib.util.spec_from_file_location("run_multi_paper_benchmark", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_gold_validator_module():
    spec = importlib.util.spec_from_file_location("validate_gold_standards", GOLD_VALIDATOR_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_multi_paper_manifest_is_valid_and_has_release_depth() -> None:
    runner = _load_runner_module()
    manifest = runner._read_json(MANIFEST_PATH)

    errors, warnings = runner.validate_manifest(manifest, root=PROJECT_ROOT)
    assert errors == []
    assert warnings == []

    smoke = runner.select_cases(manifest, tier="smoke", include_unscored=True)
    release = runner.select_cases(manifest, tier="release", include_unscored=True)
    deep = runner.select_cases(manifest, tier="deep", include_unscored=True)

    assert len(smoke) >= 1
    assert len(release) >= 5
    assert len(deep) >= 8
    assert any(case["scoring"] == "reference_comparison" for case in smoke)
    assert any(case["reference_status"] == "needed" for case in release)
    assert any(gap["id"] == "supplement_heavy_reference_case" for gap in manifest["known_gaps"])


def test_every_manifest_case_has_one_valid_gold_standard() -> None:
    runner = _load_runner_module()
    validator = _load_gold_validator_module()
    manifest = runner._read_json(MANIFEST_PATH)

    cases = manifest["cases"]
    paths = []
    for case in cases:
        gold_standard = case.get("gold_standard")
        assert isinstance(gold_standard, str) and gold_standard.endswith(".json")
        assert case.get("gold_standard_status") in {
            "codex_drafted_needs_review",
            "reviewed_gold_standard",
            "reviewed_reference_available",
        }
        path = PROJECT_ROOT / gold_standard
        assert path.exists(), case["id"]
        payload = validator.load_gold_standard(path)
        assert payload["case_id"] == case["id"]
        expectations = payload["final_report_expectations"]
        assert expectations["supplement_availability"]
        for field in (
            "secondary_findings",
            "sensitivity_analysis",
            "statistical_tests_used",
            "uniqueness",
        ):
            assert isinstance(expectations[field], list)
            assert expectations[field], f"{case['id']} missing {field}"
        paths.append(path)

    assert len(paths) == len(cases)
    assert len(set(paths)) == len(cases)


def test_release_manifest_covers_processing_stages_and_paper_components() -> None:
    runner = _load_runner_module()
    manifest = runner._read_json(MANIFEST_PATH)
    release = runner.select_cases(manifest, tier="release", include_unscored=True)

    coverage = runner.coverage_summary(release)

    for stage in [
        "source_ingestion",
        "source_manifest",
        "parsing",
        "text_analysis",
        "table_analysis",
        "figure_analysis",
        "reconcile",
        "synthesis",
        "diagnostics",
    ]:
        assert coverage["processing_stages"].get(stage, 0) > 0
    for component in [
        "title_abstract",
        "introduction",
        "methods",
        "results",
        "discussion",
        "conclusion",
        "tables",
        "figures",
        "references_metadata",
    ]:
        assert coverage["paper_components"].get(component, 0) > 0

    release_gaps = runner.known_gaps(manifest, tier="release")
    assert any(gap["id"] == "supplement_heavy_reference_case" for gap in release_gaps)


def test_default_selection_runs_only_reference_scored_smoke_case() -> None:
    runner = _load_runner_module()
    manifest = runner._read_json(MANIFEST_PATH)

    selected = runner.select_cases(manifest, tier="smoke", include_unscored=False)

    assert [case["id"] for case in selected] == ["sharma_2017_reward_deficits"]
    command = runner._command_for_case(
        selected[0],
        manifest=manifest,
        python_executable="python3",
        mode=None,
        parser_engine=None,
        backend_profile=None,
        matching_mode=None,
        matching_threshold=None,
        retain_runs=None,
        out_dir=None,
        db_dir=Path("/tmp"),
        stamp="20260621_120000",
    )
    assert "scripts/compare_pdf_against_reference.py" in command[1]
    assert "--reference-md" in command
    assert "--gold-standard-json" in command
    assert "benchmarks/gold_standards/sharma_2017_reward_deficits.json" in command[command.index("--gold-standard-json") + 1]
    assert "sharma_2017_reward_deficits" in command[command.index("--db-path") + 1]
    assert runner._case_gold_standard_path(selected[0]).name == "sharma_2017_reward_deficits.json"


def test_release_tier_validation_fails_when_unscored_cases_are_filtered() -> None:
    runner = _load_runner_module()
    manifest = runner._read_json(MANIFEST_PATH)
    selected = runner.select_cases(manifest, tier="release", include_unscored=False)

    errors, warnings = runner.validate_tier_selection(
        manifest,
        selected,
        tier="release",
        include_unscored=False,
    )

    assert any("minimum_cases" in error for error in errors)
    assert any("--include-unscored" in warning for warning in warnings)


def test_release_tier_validation_passes_with_diagnostic_coverage_cases() -> None:
    runner = _load_runner_module()
    manifest = runner._read_json(MANIFEST_PATH)
    selected = runner.select_cases(manifest, tier="release", include_unscored=True)

    errors, warnings = runner.validate_tier_selection(
        manifest,
        selected,
        tier="release",
        include_unscored=True,
    )

    assert errors == []
    assert any("target_reference_scored_cases" in warning for warning in warnings)


def test_deep_manifest_covers_all_ten_paper_expectation_pdfs() -> None:
    runner = _load_runner_module()
    manifest = runner._read_json(MANIFEST_PATH)
    expectations = runner._read_json(EXPECTATIONS_PATH)
    selected = runner.select_cases(manifest, tier="deep", include_unscored=True)

    manifest_pdfs = {case["pdf"] for case in selected}
    expectation_pdfs = {case["pdf_file"] for case in expectations["cases"]}

    assert expectation_pdfs <= manifest_pdfs


def test_child_timeout_records_structured_failed_case() -> None:
    runner = _load_runner_module()
    manifest = runner._read_json(MANIFEST_PATH)
    case = runner.select_cases(manifest, tier="smoke", include_unscored=False)[0]

    child = runner._run_child_command(
        [sys.executable, "-c", "import time; time.sleep(1)"],
        timeout_seconds=0.05,
    )
    record = runner._score_reference_case(
        case=case,
        manifest=manifest,
        paths={},
        returncode=child["returncode"],
        stdout=child["stdout"],
        stderr=child["stderr"],
        timed_out=child["timed_out"],
        timeout_seconds=child["timeout_seconds"],
    )

    assert child["timed_out"] is True
    assert record["ok"] is False
    assert record["decision"] == "fail"
    assert record["timed_out"] is True
    assert any("timed out" in failure for failure in record["failures"])


def test_reference_case_records_evidence_gold_compatibility(tmp_path: Path) -> None:
    runner = _load_runner_module()
    manifest = runner._read_json(MANIFEST_PATH)
    case = runner.select_cases(manifest, tier="smoke", include_unscored=False)[0]
    comparison_path = tmp_path / "comparison.json"
    run_path = tmp_path / "run.json"
    comparison_path.write_text(
        json.dumps(
            {
                "overall_recall": 0.95,
                "sections": {
                    section: {"recall": 0.95}
                    for section in runner.SECTION_KEYS
                },
            }
        ),
        encoding="utf-8",
    )
    run_path.write_text(
        json.dumps(
            {
                "run_mode": "pipeline",
                "runtime_seconds": 12.0,
                "summary_json": {
                    "section_diagnostics": {
                        "evidence_packet_coverage": {
                            "packet_total": 3,
                            "usable_packets": 3,
                            "usable_packet_rate": 1.0,
                            "sections_present": ["methods", "results"],
                            "missing_core_sections": ["discussion", "conclusion"],
                            "by_section": {"methods": 2, "results": 1},
                            "by_modality": {"text": 2, "figure": 1},
                            "by_detail_type": {"data_source_or_design": 1, "cross_modal_result": 1},
                            "cross_modal_packet_count": 1,
                            "typed_packet_count": 2,
                            "quality_flags": [],
                        },
                        "synthesis_evidence_plan": {
                            "missing_focus_slot_count": 1,
                            "quality_flags": ["critical_focus_slots_missing", "safety_or_adverse_events_missing"],
                            "critical_missing_focus_slots": [
                                {
                                    "slot_key": "safety_or_adverse_events",
                                    "label": "Safety or Adverse Events",
                                }
                            ],
                        }
                    },
                    "modalities": {
                        "text": {
                            "findings": [
                                {
                                    "finding_id": "methods-sample",
                                    "modality": "text",
                                    "section_label": "methods",
                                    "statement": (
                                        "The final analytic sample included 225 adults across major depressive disorder, "
                                        "bipolar disorder, schizophrenia, psychosis risk, and healthy controls."
                                    ),
                                    "evidence_refs": ["section:Methods:1"],
                                },
                                {
                                    "finding_id": "methods-bas",
                                    "modality": "text",
                                    "section_label": "methods",
                                    "statement": (
                                        "Reward responsivity was measured dimensionally with the Behavioral Activation Scale "
                                        "reward sensitivity subscale and resting-state functional connectivity."
                                    ),
                                    "evidence_refs": ["section:Methods:2"],
                                },
                            ]
                        },
                        "figure": {
                            "findings": [
                                {
                                    "finding_id": "results-network",
                                    "modality": "figure",
                                    "section_label": "results",
                                    "statement": (
                                        "Reward deficits were linked to nucleus accumbens, default mode network, and "
                                        "cingulo-opercular network dysconnectivity."
                                    ),
                                    "evidence_refs": ["figure:2"],
                                }
                            ]
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    record = runner._score_reference_case(
        case=case,
        manifest=manifest,
        paths={"comparison_json": str(comparison_path), "run_json": str(run_path)},
        returncode=0,
        stdout="",
        stderr="",
    )

    assert record["ok"] is True
    assert record["evidence_gold_compatibility"]["compatible"] is True
    assert record["evidence_gold_compatibility"]["critical_claim_candidate_rate"] == 1.0
    synthesis_diagnostics = record["evidence_gold_compatibility"]["synthesis_evidence_diagnostics"]
    assert synthesis_diagnostics["critical_missing_focus_slot_count"] == 1
    assert synthesis_diagnostics["critical_missing_focus_slots"][0]["slot_key"] == "safety_or_adverse_events"
    assert "safety_or_adverse_events_missing" in synthesis_diagnostics["synthesis_quality_flags"]
    packet_coverage = synthesis_diagnostics["evidence_packet_coverage"]
    assert packet_coverage["available"] is True
    assert packet_coverage["cross_modal_packet_count"] == 1
    assert packet_coverage["missing_core_sections"] == ["discussion", "conclusion"]


def test_evidence_gold_failure_message_includes_detail_type_rate() -> None:
    runner = _load_runner_module()

    message = runner._evidence_gold_failure_message(
        {
            "available": True,
            "compatible": False,
            "usable_packet_rate": 1.0,
            "section_coverage_rate": 1.0,
            "critical_claim_candidate_rate": 1.0,
            "expected_entity_observability_rate": 1.0,
            "expected_number_observability_rate": 1.0,
            "expected_detail_type_observability_rate": 0.25,
            "failure_reasons": ["expected detail-type observability rate 0.250 < 0.800"],
            "claim_requirement_gaps": [
                {
                    "claim_id": "claim-1",
                    "missing_entities": ["fluoxetine"],
                    "missing_numbers": ["dose_mg"],
                    "missing_detail_types": ["dose_schedule"],
                }
            ],
        }
    )

    assert "detail_types=0.250" in message
    assert "expected detail-type observability rate" in message
    assert "claim=claim-1" in message
    assert "numbers=dose_mg" in message


def test_evidence_gold_failure_message_includes_critical_synthesis_gaps() -> None:
    runner = _load_runner_module()

    message = runner._evidence_gold_failure_message(
        {
            "available": True,
            "compatible": False,
            "usable_packet_rate": 1.0,
            "section_coverage_rate": 1.0,
            "critical_claim_candidate_rate": 0.0,
            "expected_entity_observability_rate": 0.0,
            "expected_number_observability_rate": 1.0,
            "expected_detail_type_observability_rate": 1.0,
            "synthesis_evidence_diagnostics": {
                "critical_missing_focus_slots": [
                    {
                        "slot_key": "safety_or_adverse_events",
                        "label": "Safety or Adverse Events",
                    }
                ]
            },
        }
    )

    assert "critical synthesis gaps: Safety or Adverse Events" in message


def test_evidence_gold_failure_message_includes_packet_coverage_gaps() -> None:
    runner = _load_runner_module()

    message = runner._evidence_gold_failure_message(
        {
            "available": True,
            "compatible": False,
            "usable_packet_rate": 0.5,
            "section_coverage_rate": 0.5,
            "critical_claim_candidate_rate": 0.0,
            "expected_entity_observability_rate": 0.0,
            "expected_number_observability_rate": 1.0,
            "expected_detail_type_observability_rate": 1.0,
            "synthesis_evidence_diagnostics": {
                "evidence_packet_coverage": {
                    "available": True,
                    "missing_core_sections": ["methods", "results"],
                    "cross_modal_packet_count": 0,
                    "typed_packet_count": 0,
                }
            },
        }
    )

    assert "packet coverage:" in message
    assert "missing_sections=methods,results" in message
    assert "cross_modal_packets=0" in message
    assert "typed_packets=0" in message


def test_reference_case_fails_when_evidence_gold_compatibility_is_missing(tmp_path: Path) -> None:
    runner = _load_runner_module()
    manifest = runner._read_json(MANIFEST_PATH)
    case = runner.select_cases(manifest, tier="smoke", include_unscored=False)[0]
    comparison_path = tmp_path / "comparison.json"
    run_path = tmp_path / "run.json"
    comparison_path.write_text(
        json.dumps(
            {
                "overall_recall": 0.95,
                "sections": {
                    section: {"recall": 0.95}
                    for section in runner.SECTION_KEYS
                },
            }
        ),
        encoding="utf-8",
    )
    run_path.write_text(json.dumps({"run_mode": "pipeline", "runtime_seconds": 12.0}), encoding="utf-8")

    record = runner._score_reference_case(
        case=case,
        manifest=manifest,
        paths={"comparison_json": str(comparison_path), "run_json": str(run_path)},
        returncode=0,
        stdout="",
        stderr="",
    )

    assert record["ok"] is False
    assert record["evidence_gold_compatibility"]["available"] is False
    assert any("evidence/gold compatibility unavailable" in failure for failure in record["failures"])


def test_parse_compare_stdout_extracts_artifact_paths() -> None:
    runner = _load_runner_module()

    paths = runner._parse_compare_stdout(
        "\n".join(
            [
                "some log line",
                "run_json=/tmp/run.json",
                "comparison_json=/tmp/comparison.json",
                "information_retention_json=/tmp/retention.json",
            ]
        )
    )

    assert paths == {
        "run_json": "/tmp/run.json",
        "comparison_json": "/tmp/comparison.json",
        "information_retention_json": "/tmp/retention.json",
    }
