from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[2]


def _load_compare_script() -> ModuleType:
    script_path = ROOT / "scripts" / "compare_pdf_against_reference.py"
    spec = importlib.util.spec_from_file_location("compare_pdf_against_reference", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _valid_benchmark_result(compare: ModuleType) -> dict:
    presentation_evidence = {
        section: [
            {
                "statement": f"{section} statement with source provenance.",
                "evidence_refs": [f"text:{section}:1"],
            }
        ]
        for section in compare.SECTION_KEYS
    }
    return {
        "job_status": "completed",
        "summary_json": {
            "presentation_evidence": presentation_evidence,
            "model_usage": {"text_calls": 2},
            "section_diagnostics": {
                "section_extraction_enabled": True,
                "section_extraction_counts": {section: 1 for section in compare.SECTION_KEYS},
            },
        },
        "diagnostics": {
            "model_usage": {"text_calls": 2},
            "section_diagnostics": {
                "section_extraction_enabled": True,
                "section_extraction_counts": {section: 1 for section in compare.SECTION_KEYS},
            },
        },
        "fallback_note": "",
    }


def test_benchmark_validity_gate_passes_valid_run() -> None:
    compare = _load_compare_script()

    gate = compare._build_benchmark_validity_gate(_valid_benchmark_result(compare), "")

    assert gate["valid"] is True
    assert gate["failure_type"] is None
    assert gate["reasons"] == []


def test_benchmark_validity_gate_defaults_missing_provider_to_local() -> None:
    compare = _load_compare_script()
    result = _valid_benchmark_result(compare)
    result["diagnostics"]["model_usage"]["text_errors"] = 1
    result["summary_json"]["model_usage"]["text_errors"] = 1

    gate = compare._build_benchmark_validity_gate(result, "")

    assert "provider_failure" not in gate["reasons"]
    assert gate["canonical_run_validity"]["provider_failure"] is True
    assert gate["canonical_run_validity"]["blockers"].get("provider_failure") is None


def test_benchmark_validity_gate_allows_explicit_nonlocal_provider_path() -> None:
    compare = _load_compare_script()
    result = _valid_benchmark_result(compare)
    result["provider"] = "openai"
    result["diagnostics"]["model_usage"]["text_errors"] = 1
    result["summary_json"]["model_usage"]["text_errors"] = 1

    gate = compare._build_benchmark_validity_gate(result, "")

    assert "provider_failure" in gate["reasons"]


def test_benchmark_validity_gate_flags_fallback_as_infrastructure_failure() -> None:
    compare = _load_compare_script()

    gate = compare._build_benchmark_validity_gate(
        _valid_benchmark_result(compare),
        "Pipeline fallback engaged after parser failure.",
    )

    assert gate["valid"] is False
    assert gate["failure_type"] == "infrastructure"
    assert "fallback_engaged" in gate["reasons"]


def test_benchmark_validity_gate_flags_zero_text_calls_as_infrastructure_failure() -> None:
    compare = _load_compare_script()
    result = _valid_benchmark_result(compare)
    result["diagnostics"]["model_usage"]["text_calls"] = 0
    result["summary_json"]["model_usage"]["text_calls"] = 0

    gate = compare._build_benchmark_validity_gate(result, "")

    assert gate["valid"] is False
    assert gate["failure_type"] == "infrastructure"
    assert "text_llm_calls_zero" in gate["reasons"]


def test_benchmark_validity_gate_flags_missing_required_narrative_synthesis() -> None:
    compare = _load_compare_script()
    result = _valid_benchmark_result(compare)
    result["diagnostics"]["stage_model_usage"] = {"synthesis": {"deep_calls": 0}}
    result["diagnostics"]["synthesis_diagnostics"] = {
        "narrative_overrides_enabled": True,
        "narrative_synthesis_required": True,
        "narrative_synthesis_deep_calls": 0,
    }

    gate = compare._build_benchmark_validity_gate(result, "")

    assert gate["valid"] is False
    assert gate["failure_type"] == "infrastructure"
    assert "narrative_synthesis_calls_zero" in gate["reasons"]
    assert gate["quality_backend_audit"]["blockers"]["narrative_synthesis_calls_zero"] is True


def test_benchmark_validity_gate_flags_missing_diagnostics_as_infrastructure_failure() -> None:
    compare = _load_compare_script()
    result = _valid_benchmark_result(compare)
    result["diagnostics"] = {}

    gate = compare._build_benchmark_validity_gate(result, "")

    assert gate["valid"] is False
    assert gate["failure_type"] == "infrastructure"
    assert "diagnostics_missing" in gate["reasons"]


def test_benchmark_validity_gate_flags_status_section_and_provenance_blockers() -> None:
    compare = _load_compare_script()
    result = _valid_benchmark_result(compare)
    result["job_status"] = "failed"
    result["diagnostics"]["section_diagnostics"]["section_extraction_enabled"] = False
    result["summary_json"]["section_diagnostics"]["section_extraction_enabled"] = False
    first_row = result["summary_json"]["presentation_evidence"]["introduction"][0]
    first_row["evidence_refs"] = []

    gate = compare._build_benchmark_validity_gate(result, "")

    assert gate["valid"] is False
    assert gate["failure_type"] == "infrastructure"
    assert "job_not_completed" in gate["reasons"]
    assert "section_extraction_disabled" in gate["reasons"]
    assert "source_provenance_missing" in gate["reasons"]


def test_extraction_audit_passes_when_sections_present_without_fallback() -> None:
    compare = _load_compare_script()
    app_sections = {
        "introduction": ["The trial tested a reward-processing hypothesis."],
        "methods": ["Participants completed fMRI reward processing tasks."],
        "results": ["Reward-network connectivity was significantly reduced."],
        "discussion": ["The authors interpret findings as transdiagnostic reward impairment."],
        "conclusion": ["Future longitudinal studies are needed."],
    }
    ref_sections = {key: list(value) for key, value in app_sections.items()}
    comparison = compare._compare_sections(app_sections, ref_sections, match_threshold=0.1, matching_mode="lexical")
    comparison["discrepancy_diagnostics"] = compare._build_discrepancy_diagnostics(
        app_sections,
        ref_sections,
        comparison,
    )

    audit = compare._build_extraction_audit(
        app_sections,
        ref_sections,
        comparison,
        {
            "summary_json": {},
            "diagnostics": {
                "model_usage": {"text_calls": 2},
                "section_diagnostics": {
                    "section_extraction_enabled": True,
                    "section_extraction_counts": {"introduction": 1, "methods": 1, "results": 1},
                },
            },
        },
        "",
    )

    assert audit["passed"] is True
    assert audit["fallback_audit"]["passed"] is True
    assert audit["quality_backend_audit"]["passed"] is True
    assert audit["blockers"]["empty_relevant_sections"] == []


def test_extraction_audit_quality_backend_flags_missing_required_narrative_synthesis() -> None:
    compare = _load_compare_script()
    app_sections = {
        "introduction": ["The trial tested a reward-processing hypothesis."],
        "methods": ["Participants completed fMRI reward processing tasks."],
        "results": ["Reward-network connectivity was significantly reduced."],
        "discussion": [],
        "conclusion": [],
    }
    ref_sections = {key: list(value) for key, value in app_sections.items()}
    comparison = compare._compare_sections(app_sections, ref_sections, match_threshold=0.1, matching_mode="lexical")
    comparison["discrepancy_diagnostics"] = compare._build_discrepancy_diagnostics(
        app_sections,
        ref_sections,
        comparison,
    )

    audit = compare._build_extraction_audit(
        app_sections,
        ref_sections,
        comparison,
        {
            "summary_json": {},
            "diagnostics": {
                "model_usage": {"text_calls": 2},
                "section_diagnostics": {
                    "section_extraction_enabled": True,
                    "section_extraction_counts": {"introduction": 1, "methods": 1, "results": 1},
                },
                "stage_model_usage": {"synthesis": {"deep_calls": 0}},
                "synthesis_diagnostics": {
                    "narrative_overrides_enabled": True,
                    "narrative_synthesis_required": True,
                    "narrative_synthesis_deep_calls": 0,
                },
            },
        },
        "",
    )

    assert audit["passed"] is False
    assert audit["blockers"]["narrative_synthesis_calls_zero"] is True
    assert audit["quality_backend_audit"]["passed"] is False
    assert audit["quality_backend_audit"]["blockers"]["narrative_synthesis_calls_zero"] is True


def test_extraction_audit_flags_fallback_and_missing_relevant_sections() -> None:
    compare = _load_compare_script()
    app_sections = {key: [] for key in compare.SECTION_KEYS}
    app_sections["methods"] = ["Participants completed fMRI reward processing tasks."]
    ref_sections = {
        "introduction": ["The trial tested a reward-processing hypothesis."],
        "methods": ["Participants completed fMRI reward processing tasks."],
        "results": ["Reward-network connectivity was significantly reduced."],
        "discussion": [],
        "conclusion": [],
    }
    comparison = compare._compare_sections(app_sections, ref_sections, match_threshold=0.1, matching_mode="lexical")
    comparison["discrepancy_diagnostics"] = compare._build_discrepancy_diagnostics(
        app_sections,
        ref_sections,
        comparison,
    )
    result = {
        "summary_json": {},
        "diagnostics": {
            "sections_fallback_used": True,
            "sections_fallback_notes": ["Results: deterministic fallback used for minimum coverage."],
            "fallback_counts_by_reason": {"results_fallback": 1},
        },
    }

    audit = compare._build_extraction_audit(
        app_sections,
        ref_sections,
        comparison,
        result,
        "Pipeline fallback engaged after parser failure.",
    )

    assert audit["passed"] is False
    assert audit["fallback_audit"]["passed"] is False
    assert audit["fallback_audit"]["execution_fallback_used"] is True
    assert "introduction" in audit["blockers"]["empty_relevant_sections"]
    assert "results" in audit["blockers"]["empty_relevant_sections"]


def test_discrepancy_diagnostics_identifies_cross_section_and_missing_causes() -> None:
    compare = _load_compare_script()
    app_sections = {key: [] for key in compare.SECTION_KEYS}
    app_sections["results"] = ["Reward-network connectivity was significantly reduced."]
    ref_sections = {
        "introduction": ["Reward-network connectivity was significantly reduced."],
        "methods": ["Participants completed fMRI reward processing tasks."],
        "results": [],
        "discussion": [],
        "conclusion": [],
    }

    comparison = compare._compare_sections(app_sections, ref_sections, match_threshold=0.1, matching_mode="lexical")
    diagnostics = compare._build_discrepancy_diagnostics(app_sections, ref_sections, comparison)

    intro = diagnostics["section_diagnostics"]["introduction"]
    methods = diagnostics["section_diagnostics"]["methods"]
    assert intro["cause_counts"]["cross_section_only"] == 1
    assert intro["cross_section_top"][0]["best_any_section"] == "results"
    assert methods["cause_counts"]["missing_any_section"] == 1
    assert "introduction" in diagnostics["cause_details"]["cross_section_loss_sections"]
    assert "methods" in diagnostics["cause_details"]["missing_any_section_sections"]


def test_failure_mode_metrics_split_quality_dimensions() -> None:
    compare = _load_compare_script()
    app_sections = {
        "introduction": ["The paper studies reward deficits."],
        "methods": ["The final sample included 225 participants."],
        "results": ["Figure 1 showed diagnostic group variability."],
        "discussion": ["Invented unsupported moon result."],
        "conclusion": [],
    }
    ref_sections = {
        "introduction": ["The paper studies reward deficits."],
        "methods": ["The final sample included 225 participants."],
        "results": ["Figure 1 and Table S1 showed diagnostic group variability and robustness checks."],
        "discussion": ["The authors interpret findings as transdiagnostic reward impairment."],
        "conclusion": [],
    }
    comparison = compare._compare_sections(app_sections, ref_sections, match_threshold=0.3, matching_mode="lexical")
    metrics = compare._build_failure_mode_metrics(
        comparison=comparison,
        app_sections=app_sections,
        ref_sections=ref_sections,
        information_retention_summary={
            "stage_metrics": [{"stage": "executive_report", "retained_rate": 0.5}]
        },
        gold_claims=[
            {
                "claim_id": "methods-sample",
                "section": "methods",
                "evidence_quote": "The final sample included 225 participants.",
                "expected_numbers": [{"label": "n", "value": 225}],
            },
            {
                "claim_id": "discussion-interpretation",
                "section": "discussion",
                "evidence_quote": "The authors interpret findings as transdiagnostic reward impairment.",
                "expected_numbers": [{"label": "missing_marker", "value": 999}],
            },
        ],
        match_threshold=0.3,
    )

    dimensions = metrics["primary_dimensions"]
    assert set(dimensions) == {
        "parser_recall",
        "section_assignment",
        "claim_recall",
        "claim_precision",
        "numeric_fidelity",
        "unsupported_claim_rate",
        "table_figure_supplement_recall",
        "synthesis_retention",
    }
    assert dimensions["claim_recall"] == 0.5
    assert dimensions["numeric_fidelity"] == 0.5
    assert dimensions["synthesis_retention"] == 0.5
    assert metrics["gold_claims"]["missing_claims"][0]["claim_id"] == "discussion-interpretation"
    assert metrics["modality_recall"]["supplement"]["missing_refs"] == ["S1"]
    assert metrics["top_unsupported_claims"][0]["statement"] == "Invented unsupported moon result."


def test_compare_script_builds_final_report_gold_compatibility() -> None:
    compare = _load_compare_script()
    gold_path = ROOT / "benchmarks" / "gold_standards" / "sharma_2017_reward_deficits.json"
    summary_json = {
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
    }
    summary_json["section_diagnostics"] = {
        "evidence_packet_coverage": {
            "packet_total": 3,
            "usable_packets": 3,
            "usable_packet_rate": 1.0,
            "sections_present": ["methods", "results"],
            "missing_core_sections": ["discussion"],
            "by_section": {"methods": 2, "results": 1},
            "by_modality": {"text": 2, "figure": 1},
            "by_detail_type": {"data_source_or_design": 1, "cross_modal_result": 1},
            "cross_modal_packet_count": 1,
            "typed_packet_count": 2,
            "quality_flags": ["critical_focus_slots_missing"],
        }
    }

    compatibility = compare._build_evidence_gold_compatibility(summary_json, gold_path)

    assert compatibility["available"] is True
    assert compatibility["compatible"] is True
    assert compatibility["critical_claim_candidate_rate"] == 1.0
    coverage = compatibility["synthesis_evidence_diagnostics"]["evidence_packet_coverage"]
    assert coverage["available"] is True
    assert coverage["cross_modal_packet_count"] == 1
    assert coverage["missing_core_sections"] == ["discussion"]


def test_comparison_markdown_reports_detail_type_observability(tmp_path: Path) -> None:
    compare = _load_compare_script()
    output_path = tmp_path / "comparison.md"

    compare._write_comparison_markdown(
        output_path,
        {
            "evidence_gold_compatibility": {
                "available": True,
                "compatible": False,
                "expected_detail_type_observability_rate": 0.25,
                "synthesis_evidence_diagnostics": {
                    "evidence_packet_coverage": {
                        "available": True,
                        "packet_total": 4,
                        "usable_packets": 3,
                        "usable_packet_rate": 0.75,
                        "sections_present": ["methods", "results"],
                        "missing_core_sections": ["discussion", "conclusion"],
                        "by_section": {"methods": 2, "results": 1},
                        "by_modality": {"text": 2, "figure": 1},
                        "by_detail_type": {"dose_schedule": 1},
                        "cross_modal_packet_count": 1,
                        "typed_packet_count": 1,
                        "quality_flags": ["critical_focus_slots_missing"],
                    }
                },
                "claim_requirement_gaps": [
                    {
                        "claim_id": "claim-1",
                        "section": "methods",
                        "missing_entities": ["fluoxetine"],
                        "missing_numbers": ["dose_mg"],
                        "missing_detail_types": ["dose_schedule"],
                    }
                ],
            }
        },
        {"runtime_seconds": 1.2, "job_status": "completed", "document_id": 1, "job_id": 2},
    )

    rendered = output_path.read_text(encoding="utf-8")
    assert "- expected_detail_type_observability_rate: 0.25" in rendered
    assert "- claim_requirement_gaps:" in rendered
    assert "claim_id=claim-1" in rendered
    assert "missing_numbers=dose_mg" in rendered
    assert "- evidence_packet_coverage:" in rendered
    assert "  - usable_packet_rate: 0.75" in rendered
    assert "  - missing_core_sections: discussion, conclusion" in rendered
    assert "  - by_modality: text=2, figure=1" in rendered
