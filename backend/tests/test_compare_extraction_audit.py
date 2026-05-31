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
