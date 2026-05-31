from __future__ import annotations

import json

from app.services.analysis import synthesis
from app.services.analysis import reconcile
from app.services.analysis import runner
from app.services.analysis import text_analysis
from app.services.analysis.reconcile import reconcile_reports
from app.services.analysis.synthesis import synthesize_report
from app.services.analysis.utils import normalize_evidence_packets
from app.db.models import Chunk
from app.services import validated_pipeline


def test_normalize_evidence_packets_clamps_and_flags_missing() -> None:
    packets = normalize_evidence_packets(
        [
            {
                "finding_id": "x1",
                "anchor": "section:intro:1",
                "statement": "Treatment improved symptoms by 25%",
                "evidence_refs": ["section:intro:1"],
                "confidence": 1.7,
                "category": "clinical",
            },
            {
                "finding_id": "x2",
                "anchor": "missing:1",
                "statement": "Secondary claim",
                "evidence_refs": ["missing:1"],
                "confidence": -0.2,
                "category": "other",
            },
        ],
        "text",
        {"section:intro:1"},
    )
    assert len(packets) == 2
    assert packets[0]["confidence"] == 1.0
    assert packets[1]["confidence"] == 0.0
    assert "missing_evidence" in packets[1]["quality_flags"]


def test_normalize_evidence_packets_resolves_near_anchor_variants() -> None:
    packets = normalize_evidence_packets(
        [
            {
                "finding_id": "x1",
                "anchor": "section:results:2",
                "statement": "Connectivity increased in reward-related networks.",
                "evidence_refs": ["section:Results:2"],
                "confidence": 0.81,
                "category": "results",
            },
            {
                "finding_id": "x2",
                "anchor": "section:conclusions:38",
                "statement": "Overall findings suggest shared circuitry targets.",
                "evidence_refs": ["section:conclusion:38"],
                "confidence": 0.74,
                "category": "conclusion",
            },
        ],
        "text",
        {"section:Results::2", "section:CONCLUSIONS:38"},
    )
    assert len(packets) == 2
    assert packets[0]["anchor"] == "section:Results::2"
    assert packets[0]["evidence_refs"] == ["section:Results::2"]
    assert "missing_evidence" not in packets[0]["quality_flags"]
    assert packets[1]["anchor"] == "section:CONCLUSIONS:38"
    assert packets[1]["evidence_refs"] == ["section:CONCLUSIONS:38"]
    assert "missing_evidence" not in packets[1]["quality_flags"]


def test_reconcile_reports_produces_typed_mismatch_reason() -> None:
    text_report = {
        "claim_packets": [
            {
                "finding_id": "claim-1",
                "statement": "The intervention reduced symptoms by 40%",
                "evidence_refs": ["section:results:3"],
                "confidence": 0.8,
                "value": 40.0,
                "unit": "%",
                "category": "claim",
            }
        ]
    }
    table_report = {
        "evidence_packets": [
            {
                "finding_id": "table-1",
                "modality": "table",
                "anchor": "table:1",
                "statement": "Observed reduction was 10%",
                "evidence_refs": ["section:results:3"],
                "confidence": 0.9,
                "value": 10.0,
                "unit": "%",
                "category": "stats",
            }
        ]
    }
    report = reconcile_reports(text_report, table_report, {"evidence_packets": []}, {"evidence_packets": []})
    reasons = {item["reason"] for item in report["discrepancies"]}
    assert "magnitude_mismatch" in reasons
    assert report["cross_modal_claims"][0]["status"] in {"contradicted", "partial"}


def test_reconcile_stats_flag_deep_review_invocation(monkeypatch) -> None:
    calls: list[dict] = []

    def _fake_chat(prompt: str, system: str | None = None, temperature: float = 0.2) -> str:
        calls.append({"prompt": prompt})
        return '{"discrepancies":[]}'

    monkeypatch.setattr(reconcile, "chat_text_deep", _fake_chat)
    monkeypatch.setattr(reconcile.settings, "analysis_narrative_overrides_subprocess_guard_enabled", False)

    text_report = {
        "claim_packets": [
            {
                "finding_id": "claim-1",
                "statement": "Intervention reduced symptoms.",
                "evidence_refs": ["section:results:1"],
                "confidence": 0.8,
                "category": "claim",
            }
        ]
    }
    table_report = {
        "evidence_packets": [
            {
                "finding_id": "table-1",
                "modality": "table",
                "anchor": "table:1",
                "statement": "No overlap evidence",
                "evidence_refs": ["table:1"],
                "confidence": 0.8,
                "category": "stats",
            }
        ]
    }
    report = reconcile_reports(text_report, table_report, {"evidence_packets": []}, {"evidence_packets": []})
    assert calls
    assert report["stats"]["llm_review_invoked"] is True
    assert report["stats"]["llm_review_inputs"] >= 1


def test_synthesize_report_emits_schema_v2_and_compat_fields() -> None:
    text_report = {
        "evidence_packets": [
            {
                "finding_id": "text-1",
                "modality": "text",
                "anchor": "section:methods:1",
                "statement": "Randomization was clearly described.",
                "evidence_refs": ["section:methods:1"],
                "confidence": 0.9,
                "quality_flags": [],
                "category": "methods",
            }
        ]
    }
    table_report = {"evidence_packets": []}
    figure_report = {"evidence_packets": []}
    supp_report = {"evidence_packets": []}
    reconcile_report = {"cross_modal_claims": [], "discrepancies": []}
    summary = synthesize_report(
        text_report,
        table_report,
        figure_report,
        supp_report,
        reconcile_report,
        paper_meta={"title": "Sample Trial"},
        coverage={
            "figures": {"expected": 0, "extracted": 0, "missing_refs": []},
            "tables": {"expected": 0, "extracted": 0, "missing_refs": []},
            "supp_figures": {"expected": 0, "extracted": 0, "missing_refs": []},
            "supp_tables": {"expected": 0, "extracted": 0, "missing_refs": []},
        },
    )
    assert summary["schema_version"] == 2
    assert "modalities" in summary and "text" in summary["modalities"]
    assert "figure_coverage" in summary
    assert "key_findings" in summary
    assert "Introduction:" in summary["executive_summary"]
    assert "Methods:" in summary["executive_summary"]
    assert "Results:" in summary["executive_summary"]
    assert "Discussion:" in summary["executive_summary"]
    assert "Conclusion:" in summary["executive_summary"]
    assert len(summary["executive_summary"]) < 1400


def test_synthesize_report_applies_deep_overrides_to_methods_and_interpretation(monkeypatch) -> None:
    def _fake_chat(prompt: str, system: str | None = None, temperature: float = 0.2) -> str:
        return (
            '{"executive_summary":"Deep summary.",'
            '"interpretation":"Deep interpretation.",'
            '"methods_strengths":["Deep strength"],'
            '"methods_weaknesses":["Deep weakness"],'
            '"reproducibility_ethics":["Deep ethics"],'
            '"uncertainty_gaps":["Deep gap"]}'
        )

    monkeypatch.setattr(synthesis, "chat_text_deep", _fake_chat)
    monkeypatch.setattr(synthesis.settings, "analysis_narrative_overrides_enabled", True)

    text_report = {
        "evidence_packets": [
            {
                "finding_id": "text-1",
                "modality": "text",
                "anchor": "section:methods:1",
                "statement": "Fallback strength",
                "evidence_refs": ["section:methods:1"],
                "confidence": 0.9,
                "quality_flags": [],
                "category": "methods",
            }
        ]
    }
    summary = synthesize_report(
        text_report,
        {"evidence_packets": []},
        {"evidence_packets": []},
        {"evidence_packets": []},
        {"cross_modal_claims": [], "discrepancies": []},
        paper_meta={},
        coverage={
            "figures": {"expected": 0, "extracted": 0, "missing_refs": []},
            "tables": {"expected": 0, "extracted": 0, "missing_refs": []},
            "supp_figures": {"expected": 0, "extracted": 0, "missing_refs": []},
            "supp_tables": {"expected": 0, "extracted": 0, "missing_refs": []},
        },
    )

    assert "Introduction:" in summary["executive_summary"]
    assert "Introduction:" in summary["executive_summary"]
    assert "Methods:" in summary["executive_summary"]
    assert "Results:" in summary["executive_summary"]
    assert "Discussion:" in summary["executive_summary"]
    assert "Conclusion:" in summary["executive_summary"]
    assert summary["interpretation"] == "Deep interpretation."
    assert summary["methods_strengths"] == ["Deep strength"]
    assert summary["methods_weaknesses"] == ["Deep weakness"]
    assert summary["reproducibility_ethics"] == ["Deep ethics"]
    assert summary["uncertainty_gaps"] == ["Deep gap"]


def test_synthesize_report_uses_raw_chunks_for_intro_when_packets_sparse() -> None:
    summary = synthesize_report(
        {"evidence_packets": []},
        {"evidence_packets": []},
        {"evidence_packets": []},
        {"evidence_packets": []},
        {"cross_modal_claims": [], "discrepancies": []},
        paper_meta={"title": "Sparse Intro Test"},
        coverage={
            "figures": {"expected": 0, "extracted": 0, "missing_refs": []},
            "tables": {"expected": 0, "extracted": 0, "missing_refs": []},
            "supp_figures": {"expected": 0, "extracted": 0, "missing_refs": []},
            "supp_tables": {"expected": 0, "extracted": 0, "missing_refs": []},
        },
        text_chunk_records=[
            {
                "anchor": "section:body:0",
                "content": (
                    "Major depressive and psychotic disorders share impairments in reward responsivity, "
                    "yet the neural architecture supporting this transdiagnostic deficit remains unclear. "
                    "This study aimed to test whether connectome-wide signatures track dimensional reward deficits."
                ),
            },
            {
                "anchor": "section:Methods:1",
                "content": "Participants underwent MRI and multivariate analysis.",
            },
        ],
    )
    intro_items = summary.get("sections", {}).get("introduction", {}).get("items", [])
    assert isinstance(intro_items, list) and intro_items
    assert any("section:body:0" in ",".join(item.get("evidence_refs", [])) for item in intro_items if isinstance(item, dict))
    intro_slots = summary.get("sections_compact", {}).get("introduction", [])
    assert any(str(row.get("status", "")).strip().lower() == "found" for row in intro_slots if isinstance(row, dict))


def test_synthesize_report_verifier_pass_can_correct_narrative(monkeypatch) -> None:
    responses = iter(
        [
            (
                '{"executive_summary":"Initial draft summary.",'
                '"interpretation":"Initial draft interpretation.",'
                '"methods_strengths":["Initial strength"],'
                '"methods_weaknesses":["Initial weakness"],'
                '"reproducibility_ethics":["Initial ethics"],'
                '"uncertainty_gaps":["Initial gap"]}'
            ),
            (
                '{"executive_summary":"Verified summary with corrected wording.",'
                '"interpretation":"Verified interpretation with cautious language.",'
                '"methods_strengths":["Verified strength"],'
                '"methods_weaknesses":["Verified weakness"],'
                '"reproducibility_ethics":["Verified ethics"],'
                '"uncertainty_gaps":["Verified gap"]}'
            ),
        ]
    )

    def _fake_chat(prompt: str, system: str | None = None, temperature: float = 0.2) -> str:
        return next(responses)

    monkeypatch.setattr(synthesis, "chat_text_deep", _fake_chat)
    monkeypatch.setattr(synthesis.settings, "analysis_verifier_enabled", True)
    monkeypatch.setattr(synthesis.settings, "analysis_narrative_overrides_enabled", True)

    summary = synthesize_report(
        {
            "evidence_packets": [
                {
                    "finding_id": "text-1",
                    "modality": "text",
                    "anchor": "section:methods:1",
                    "statement": "Method signal",
                    "evidence_refs": ["section:methods:1"],
                    "confidence": 0.9,
                    "quality_flags": [],
                    "category": "methods",
                }
            ]
        },
        {"evidence_packets": []},
        {"evidence_packets": []},
        {"evidence_packets": []},
        {"cross_modal_claims": [], "discrepancies": []},
        paper_meta={},
        coverage={
            "figures": {"expected": 0, "extracted": 0, "missing_refs": []},
            "tables": {"expected": 0, "extracted": 0, "missing_refs": []},
            "supp_figures": {"expected": 0, "extracted": 0, "missing_refs": []},
            "supp_tables": {"expected": 0, "extracted": 0, "missing_refs": []},
        },
    )

    assert "Introduction:" in summary["executive_summary"]
    assert "Introduction:" in summary["executive_summary"]
    assert "Methods:" in summary["executive_summary"]
    assert "Results:" in summary["executive_summary"]
    assert "Discussion:" in summary["executive_summary"]
    assert "Conclusion:" in summary["executive_summary"]
    assert summary["interpretation"] == "Verified interpretation with cautious language."
    assert summary["methods_strengths"] == ["Verified strength"]
    assert summary["methods_weaknesses"] == ["Verified weakness"]
    assert summary["reproducibility_ethics"] == ["Verified ethics"]
    assert summary["uncertainty_gaps"] == ["Verified gap"]


def test_methodology_details_prioritizes_methods_and_filters_outcome_summary() -> None:
    details = synthesis._methodology_details(
        [
            {
                "anchor": "section:Results:5",
                "statement": "Results showed symptoms improved by 25% in the treatment arm.",
                "evidence_refs": ["section:Results:5"],
                "confidence": 0.9,
                "category": "clinical",
            },
            {
                "anchor": "section:Methods:2",
                "statement": "Randomized allocation used concealed sequence generation and assessor blinding.",
                "evidence_refs": ["section:Methods:2"],
                "confidence": 0.8,
                "category": "methods",
            },
            {
                "anchor": "section:Analysis:7",
                "statement": "Mixed-effects regression adjusted for baseline severity and site.",
                "evidence_refs": ["section:Analysis:7"],
                "confidence": 0.7,
                "category": "stats",
            },
        ]
    )
    assert len(details) == 2
    assert details[0]["statement"].startswith("Randomized allocation")
    statements = {item["statement"] for item in details}
    assert "Results showed symptoms improved by 25% in the treatment arm." not in statements


def test_methodology_details_promotes_method_signal_without_category() -> None:
    details = synthesis._methodology_details(
        [
            {
                "anchor": "section:Body:9",
                "statement": "Inclusion criteria required DSM-5 diagnosis and a baseline PHQ-9 score >= 10.",
                "evidence_refs": ["section:Body:9"],
                "confidence": 0.6,
                "category": "other",
            }
        ]
    )
    assert len(details) == 1
    assert details[0]["category"] == "methods"


def test_synthesize_report_includes_analysis_notes_in_uncertainty() -> None:
    summary = synthesize_report(
        {
            "evidence_packets": [],
            "analysis_notes": [
                "Source text appears access-limited (publisher landing/subscription content).",
            ],
        },
        {"evidence_packets": []},
        {"evidence_packets": []},
        {"evidence_packets": []},
        {"cross_modal_claims": [], "discrepancies": []},
        paper_meta={},
        coverage={
            "figures": {"expected": 0, "extracted": 0, "missing_refs": []},
            "tables": {"expected": 0, "extracted": 0, "missing_refs": []},
            "supp_figures": {"expected": 0, "extracted": 0, "missing_refs": []},
            "supp_tables": {"expected": 0, "extracted": 0, "missing_refs": []},
        },
    )
    assert any("access-limited" in item for item in summary["uncertainty_gaps"])


def test_methods_compact_emits_twelve_slots_and_not_found_defaults() -> None:
    summary = synthesize_report(
        {"evidence_packets": []},
        {"evidence_packets": []},
        {"evidence_packets": []},
        {"evidence_packets": []},
        {"cross_modal_claims": [], "discrepancies": []},
        paper_meta={},
        coverage={
            "figures": {"expected": 0, "extracted": 0, "missing_refs": []},
            "tables": {"expected": 0, "extracted": 0, "missing_refs": []},
            "supp_figures": {"expected": 0, "extracted": 0, "missing_refs": []},
            "supp_tables": {"expected": 0, "extracted": 0, "missing_refs": []},
        },
    )
    slots = summary["methods_compact"]
    assert len(slots) == 12
    assert all(item["status"] == "not_found" for item in slots)
    assert all(item["statement"] == "N/A - not found in parsed text." for item in slots)


def test_methods_compact_uses_access_limited_status_when_noted() -> None:
    summary = synthesize_report(
        {
            "evidence_packets": [],
            "analysis_notes": ["Source text appears access-limited (publisher landing/subscription content)."],
        },
        {"evidence_packets": []},
        {"evidence_packets": []},
        {"evidence_packets": []},
        {"cross_modal_claims": [], "discrepancies": []},
        paper_meta={},
        coverage={
            "figures": {"expected": 0, "extracted": 0, "missing_refs": []},
            "tables": {"expected": 0, "extracted": 0, "missing_refs": []},
            "supp_figures": {"expected": 0, "extracted": 0, "missing_refs": []},
            "supp_tables": {"expected": 0, "extracted": 0, "missing_refs": []},
        },
    )
    slots = summary["methods_compact"]
    assert len(slots) == 12
    assert all(item["status"] == "access_limited" for item in slots)


def test_methods_compact_prefers_method_anchor_and_caps_statement_length() -> None:
    long_statement = (
        "MDMR was used with nuisance covariates and multiple network modules, while additional details describe "
        "parameterization, thresholds, harmonization, and extra context that should be truncated for compact display."
    )
    summary = synthesize_report(
        {
            "evidence_packets": [
                {
                    "finding_id": "m1",
                    "modality": "text",
                    "anchor": "section:Methods:3",
                    "statement": long_statement,
                    "evidence_refs": ["section:Methods:3"],
                    "confidence": 0.9,
                    "quality_flags": [],
                    "category": "methods",
                },
                {
                    "finding_id": "r1",
                    "modality": "text",
                    "anchor": "section:Results:10",
                    "statement": "Results showed improvements in outcomes.",
                    "evidence_refs": ["section:Results:10"],
                    "confidence": 0.9,
                    "quality_flags": [],
                    "category": "results",
                },
            ]
        },
        {"evidence_packets": []},
        {"evidence_packets": []},
        {"evidence_packets": []},
        {"cross_modal_claims": [], "discrepancies": []},
        paper_meta={},
        coverage={
            "figures": {"expected": 0, "extracted": 0, "missing_refs": []},
            "tables": {"expected": 0, "extracted": 0, "missing_refs": []},
            "supp_figures": {"expected": 0, "extracted": 0, "missing_refs": []},
            "supp_tables": {"expected": 0, "extracted": 0, "missing_refs": []},
        },
    )
    slots = summary["methods_compact"]
    study_design = next(item for item in slots if item["slot_key"] == "study_design")
    assert study_design["status"] == "found"
    assert len(study_design["statement"]) <= 160
    assert all("Results showed improvements in outcomes." not in item["statement"] for item in slots)


def test_text_section_annotation_prefers_anchor_over_category() -> None:
    packets = text_analysis._annotate_text_packet_sections(
        [
            {
                "finding_id": "t1",
                "anchor": "section:Methods:11",
                "statement": "Randomization and covariate handling were described.",
                "evidence_refs": ["section:Methods:11"],
                "confidence": 0.9,
                "category": "results",
            }
        ]
    )
    assert packets[0]["section_label"] == "methods"
    assert packets[0]["section_source"] == "anchor"
    assert packets[0]["section_confidence"] >= 0.9


def test_hydrate_anchor_metadata_prefers_section_source_and_carries_order_fields() -> None:
    packets = [
        {
            "finding_id": "t1",
            "anchor": "section:Methods:11",
            "statement": "Randomization and covariate handling were described.",
            "evidence_refs": ["section:Methods:11"],
            "confidence": 0.9,
            "category": "results",
        }
    ]
    chunks = [
        {
            "anchor": "section:Methods:11",
            "meta": json.dumps(
                {
                    "source": "docling",
                    "section_source": "heading",
                    "section_norm": "methods",
                    "section_confidence": 0.94,
                    "paragraph_index": 11,
                }
            ),
        }
    ]
    hydrated = text_analysis._hydrate_anchor_metadata(packets, chunks)
    assert hydrated[0]["anchor_meta_source"] == "heading"
    assert hydrated[0]["anchor_meta_paragraph_index"] == 11
    assert hydrated[0]["anchor_meta_anchor_order"] == 0


def test_text_section_annotation_uses_document_order_not_input_order() -> None:
    chunks = [
        {
            "anchor": "section:Body:12",
            "meta": json.dumps(
                {
                    "source": "docling",
                    "section_source": "position",
                    "section_norm": "unknown",
                    "section_confidence": 0.24,
                    "paragraph_index": 12,
                }
            ),
        },
        {
            "anchor": "section:Conclusion:90",
            "meta": json.dumps(
                {
                    "source": "docling",
                    "section_source": "heading",
                    "section_norm": "conclusion",
                    "section_confidence": 0.95,
                    "paragraph_index": 90,
                }
            ),
        },
    ]
    packets = text_analysis._hydrate_anchor_metadata(
        [
            {
                "finding_id": "c1",
                "anchor": "section:Conclusion:90",
                "statement": "Overall, these findings support network-level effects.",
                "evidence_refs": ["section:Conclusion:90"],
                "confidence": 0.9,
                "category": "other",
            },
            {
                "finding_id": "m1",
                "anchor": "section:Body:12",
                "statement": "Participants completed standardized interviews and covariate-adjusted analyses.",
                "evidence_refs": ["section:Body:12"],
                "confidence": 0.85,
                "category": "methods",
            },
        ],
        chunks,
    )
    annotated = text_analysis._annotate_text_packet_sections(packets)
    assert annotated[1]["section_label"] == "methods"


def test_text_section_annotation_meta_results_override_late_conclusion_lock() -> None:
    chunks = [
        {
            "anchor": "section:Conclusions:1",
            "meta": json.dumps(
                {
                    "source": "grobid_tei",
                    "section_norm": "conclusion",
                    "section_raw_title": "Conclusions",
                }
            ),
        },
        {
            "anchor": "section:Individual Deviations in Topological Properties:2",
            "meta": json.dumps(
                {
                    "source": "grobid_tei",
                    "section_norm": "results",
                    "section_raw_title": "Individual Deviations in Topological Properties",
                }
            ),
        },
    ]
    packets = text_analysis._hydrate_anchor_metadata(
        [
            {
                "finding_id": "c1",
                "anchor": "section:Conclusions:1",
                "statement": "Overall, these findings support translational relevance.",
                "evidence_refs": ["section:Conclusions:1"],
                "confidence": 0.85,
                "category": "conclusion",
            },
            {
                "finding_id": "r1",
                "anchor": "section:Individual Deviations in Topological Properties:2",
                "statement": "Children with ADHD showed significantly greater extreme nodal deviations versus controls.",
                "evidence_refs": ["section:Individual Deviations in Topological Properties:2"],
                "confidence": 0.82,
                "category": "results",
            },
        ],
        chunks,
    )
    annotated = text_analysis._annotate_text_packet_sections(packets)
    labels = {item["finding_id"]: item.get("section_label") for item in annotated}
    assert labels["c1"] == "conclusion"
    assert labels["r1"] == "results"


def test_unlabeled_methods_position_chunk_with_observed_outcome_partitions_as_results() -> None:
    chunks = [
        {
            "anchor": "section:Body:7",
            "meta": json.dumps(
                {
                    "source": "docling",
                    "section_source": "position",
                    "section_norm": "methods",
                    "section_confidence": 0.24,
                    "paragraph_index": 7,
                }
            ),
        }
    ]
    packets = text_analysis._hydrate_anchor_metadata(
        [
            {
                "finding_id": "r1",
                "anchor": "section:Body:7",
                "statement": (
                    "The MDMR analysis identified two significant connectivity clusters associated "
                    "with lower reward responsiveness (p<0.01)."
                ),
                "evidence_refs": ["section:Body:7"],
                "confidence": 0.82,
                "category": "other",
            }
        ],
        chunks,
    )
    annotated = text_analysis._annotate_text_packet_sections(packets)
    assert annotated[0]["section_label"] == "results"
    assert annotated[0]["section_source"] in {"semantic", "lexical"}


def test_unlabeled_result_media_context_partitions_as_results() -> None:
    chunks = [
        {
            "anchor": "section:Body:22",
            "meta": json.dumps(
                {
                    "source": "docling",
                    "section_source": "position",
                    "section_norm": "discussion",
                    "section_confidence": 0.24,
                    "paragraph_index": 22,
                }
            ),
        }
    ]
    packets = text_analysis._hydrate_anchor_metadata(
        [
            {
                "finding_id": "r1",
                "anchor": "section:Body:22",
                "statement": "Table 2 showed higher symptom severity in the low-reward group than controls.",
                "evidence_refs": ["section:Body:22"],
                "confidence": 0.8,
                "category": "other",
            }
        ],
        chunks,
    )
    annotated = text_analysis._annotate_text_packet_sections(packets)
    assert annotated[0]["section_label"] == "results"


def test_unlabeled_procedural_analysis_sentence_stays_methods() -> None:
    section = text_analysis._infer_chunk_section(
        {
            "anchor": "section:Body:4",
            "content": "We used MDMR analysis to identify clusters while adjusting for age, sex, and scanner site.",
            "meta": json.dumps({"section_source": "position", "section_norm": "unknown"}),
        },
        idx=4,
        total_chunks=20,
    )
    assert section == "methods"


def test_results_fidelity_gate_rejects_generic_visual_and_method_lines() -> None:
    packets = [
        {
            "anchor": "section:Results:8",
            "statement": "Figure 3 shows connectivity maps.",
            "evidence_refs": ["section:Results:8"],
            "confidence": 0.8,
        },
        {
            "anchor": "section:Results:9",
            "statement": "Connectivity increased in the default mode network (p<0.01).",
            "evidence_refs": ["section:Results:9"],
            "confidence": 0.8,
        },
        {
            "anchor": "section:Results:10",
            "statement": "MDMR was used to test covariates and scanner effects.",
            "evidence_refs": ["section:Results:10"],
            "confidence": 0.8,
        },
    ]
    filtered = synthesis._filter_result_text_packets(packets)
    statements = {item["statement"] for item in filtered}
    assert "Connectivity increased in the default mode network (p<0.01)." in statements
    assert "Figure 3 shows connectivity maps." not in statements
    assert "MDMR was used to test covariates and scanner effects." not in statements


def test_section_blocks_emit_hybrid_fallback_for_intro_and_conclusion() -> None:
    summary = synthesize_report(
        {
            "evidence_packets": [
                {
                    "finding_id": "x1",
                    "modality": "text",
                    "anchor": "section:Background:1",
                    "statement": "The objective was to evaluate transdiagnostic reward deficits.",
                    "evidence_refs": ["section:Background:1"],
                    "confidence": 0.8,
                    "quality_flags": [],
                    "category": "other",
                    "section_label": "unknown",
                },
                {
                    "finding_id": "x2",
                    "modality": "text",
                    "anchor": "section:Methods:2",
                    "statement": "Participants completed BAS reward sensitivity assessments.",
                    "evidence_refs": ["section:Methods:2"],
                    "confidence": 0.8,
                    "quality_flags": [],
                    "category": "methods",
                    "section_label": "methods",
                },
                {
                    "finding_id": "x3",
                    "modality": "text",
                    "anchor": "section:Results:3",
                    "statement": "Reward deficits were associated with higher DMN connectivity (p<0.05).",
                    "evidence_refs": ["section:Results:3"],
                    "confidence": 0.8,
                    "quality_flags": [],
                    "category": "results",
                    "section_label": "results",
                },
                {
                    "finding_id": "x4",
                    "modality": "text",
                    "anchor": "section:Discussion:4",
                    "statement": "Overall, these findings support a shared network-level mechanism across diagnoses.",
                    "evidence_refs": ["section:Discussion:4"],
                    "confidence": 0.8,
                    "quality_flags": [],
                    "category": "discussion",
                    "section_label": "discussion",
                },
            ]
        },
        {"evidence_packets": []},
        {"evidence_packets": []},
        {"evidence_packets": []},
        {"cross_modal_claims": [], "discrepancies": []},
        paper_meta={},
        coverage={
            "figures": {"expected": 0, "extracted": 0, "missing_refs": []},
            "tables": {"expected": 0, "extracted": 0, "missing_refs": []},
            "supp_figures": {"expected": 0, "extracted": 0, "missing_refs": []},
            "supp_tables": {"expected": 0, "extracted": 0, "missing_refs": []},
        },
    )
    sections = summary["sections"]
    assert sections["introduction"]["fallback_used"] is True
    assert sections["conclusion"]["fallback_used"] is True
    assert summary["sections_fallback_used"] is True
    assert any("Introduction" in note for note in summary["sections_fallback_notes"])
    assert any("Conclusion" in note for note in summary["sections_fallback_notes"])


def test_intro_section_uses_pre_methods_positional_fallback() -> None:
    summary = synthesize_report(
        {
            "evidence_packets": [
                {
                    "finding_id": "i1",
                    "modality": "text",
                    "anchor": "section:Body:1",
                    "statement": "Reward dysfunction spans diagnostic boundaries and motivates a transdiagnostic framework.",
                    "evidence_refs": ["section:Body:1"],
                    "confidence": 0.8,
                    "quality_flags": [],
                    "category": "other",
                    "section_label": "unknown",
                },
                {
                    "finding_id": "i2",
                    "modality": "text",
                    "anchor": "section:Body:2",
                    "statement": "Prior studies have not fully resolved shared network mechanisms across disorders.",
                    "evidence_refs": ["section:Body:2"],
                    "confidence": 0.8,
                    "quality_flags": [],
                    "category": "other",
                    "section_label": "unknown",
                },
                {
                    "finding_id": "m1",
                    "modality": "text",
                    "anchor": "section:Methods:3",
                    "statement": "Participants completed BAS reward sensitivity assessments.",
                    "evidence_refs": ["section:Methods:3"],
                    "confidence": 0.8,
                    "quality_flags": [],
                    "category": "methods",
                    "section_label": "methods",
                },
                {
                    "finding_id": "r1",
                    "modality": "text",
                    "anchor": "section:Results:4",
                    "statement": "Reward deficits were associated with higher DMN connectivity (p<0.05).",
                    "evidence_refs": ["section:Results:4"],
                    "confidence": 0.8,
                    "quality_flags": [],
                    "category": "results",
                    "section_label": "results",
                },
            ]
        },
        {"evidence_packets": []},
        {"evidence_packets": []},
        {"evidence_packets": []},
        {"cross_modal_claims": [], "discrepancies": []},
        paper_meta={},
        coverage={
            "figures": {"expected": 0, "extracted": 0, "missing_refs": []},
            "tables": {"expected": 0, "extracted": 0, "missing_refs": []},
            "supp_figures": {"expected": 0, "extracted": 0, "missing_refs": []},
            "supp_tables": {"expected": 0, "extracted": 0, "missing_refs": []},
        },
    )
    introduction = summary["sections"]["introduction"]
    intro_statements = [str(item.get("statement", "")).lower() for item in introduction["items"]]
    assert introduction["fallback_used"] is True
    assert len(intro_statements) >= 2
    assert any("transdiagnostic framework" in statement for statement in intro_statements)
    assert any("shared network mechanisms" in statement for statement in intro_statements)


def test_media_counts_line_reports_text_cited_supp_tables() -> None:
    line = synthesis._media_counts_line(
        {
            "figures": {"expected": 5, "extracted": 5},
            "tables": {"expected": 2, "extracted": 1},
            "supp_figures": {"expected": 3, "extracted": 0},
            "supp_tables": {"expected": 4, "extracted": 0},
        }
    )
    assert "Main tables (text-cited/extracted): 2/1" in line
    assert "Supplementary tables (text-cited/extracted): 4/0" in line


def test_compute_coverage_counts_supp_refs_from_text_mentions() -> None:
    text_chunk = Chunk(
        document_id=1,
        asset_id=1,
        anchor="section:results:1",
        modality="text",
        content="See Supplementary Figure S1 and Supplementary Table S2 for additional analyses.",
        meta=None,
    )
    coverage = runner._compute_coverage(
        text_chunks=[text_chunk],
        table_chunks=[],
        figure_chunks=[],
        supp_expected_text_chunks=[text_chunk],
        supp_table_chunks=[],
        supp_figure_chunks=[],
    )
    assert coverage["supp_figures"]["expected"] == 1
    assert coverage["supp_tables"]["expected"] == 1


def test_extract_tei_metadata_author_dedupe_and_source() -> None:
    tei = """
    <TEI xmlns="http://www.tei-c.org/ns/1.0">
      <teiHeader>
        <fileDesc>
          <titleStmt>
            <title>Sample Paper</title>
            <author><persName><forename>Ignored</forename><surname>Header</surname></persName></author>
          </titleStmt>
          <sourceDesc>
            <biblStruct>
              <analytic>
                <author><persName><forename>Ana</forename><surname>Smith</surname></persName></author>
                <author><persName><forename>Ana</forename><surname>Smith</surname></persName></author>
                <author><persName><forename>Ben</forename><surname>Jones</surname></persName></author>
              </analytic>
            </biblStruct>
          </sourceDesc>
        </fileDesc>
      </teiHeader>
    </TEI>
    """
    meta = validated_pipeline._extract_tei_metadata(tei)
    assert meta["metadata_source"] == "tei_analytic"
    assert meta["authors"] == ["Ana Smith", "Ben Jones"]
    assert meta["authors_extracted_count"] == 2
    assert meta["authors_display_count"] == 2


def test_extract_tei_metadata_filters_affiliation_spillover() -> None:
    tei = """
    <TEI xmlns="http://www.tei-c.org/ns/1.0">
      <teiHeader>
        <fileDesc>
          <sourceDesc>
            <biblStruct>
              <analytic>
                <author><persName><forename>Nanfang</forename><surname>Pan</surname></persName></author>
                <author><persName><forename>Qiyong</forename><surname>Gong</surname></persName></author>
                <author>Department of Radiology Research JAMA Psychiatry Xiamen Hospital</author>
              </analytic>
            </biblStruct>
          </sourceDesc>
        </fileDesc>
      </teiHeader>
    </TEI>
    """
    meta = validated_pipeline._extract_tei_metadata(tei)
    assert meta["authors"] == ["Nanfang Pan", "Qiyong Gong"]
    assert meta["authors_extracted_count"] == 2
    assert meta["authors_display_count"] == 2


def test_runner_sanitize_meta_recomputes_author_counts() -> None:
    meta = {
        "authors": [
            "Nanfang Pan",
            "Qiyong Gong",
            "Department of Radiology Research JAMA Psychiatry Xiamen Hospital",
        ],
        "authors_extracted_count": 23,
        "authors_display_count": 23,
        "metadata_source": "tei_analytic",
    }
    sanitized = runner._sanitize_meta(meta)
    assert sanitized["authors"] == ["Nanfang Pan", "Qiyong Gong"]
    assert sanitized["authors_extracted_count"] == 2
    assert sanitized["authors_display_count"] == 2


def test_section_dedupe_strips_citation_prefixes_and_drops_fragments() -> None:
    rows = [
        {
            "statement": "7 Normative modeling incorporating neuroimaging metrics offers a robust framework.",
            "anchor": "section:Discussion:1",
            "evidence_refs": ["section:Discussion:1"],
            "source_modality": "text",
            "confidence": 0.8,
            "section_confidence": 0.8,
            "flags": [],
        },
        {
            "statement": "4, 5 In contrast, data-driven clustering may offer a superior solution.",
            "anchor": "section:Discussion:2",
            "evidence_refs": ["section:Discussion:2"],
            "source_modality": "text",
            "confidence": 0.8,
            "section_confidence": 0.8,
            "flags": [],
        },
        {
            "statement": "In contrast, biotype 2 demonstrated higher hyperactivity/impulsivity (mean [SD], 0.",
            "anchor": "section:Discussion:3",
            "evidence_refs": ["section:Discussion:3"],
            "source_modality": "text",
            "confidence": 0.7,
            "section_confidence": 0.7,
            "flags": [],
        },
    ]
    deduped = synthesis._dedupe_section_items(rows, max_items=10)
    statements = [str(item.get("statement", "")) for item in deduped]
    assert any(statement.startswith("Normative modeling incorporating neuroimaging metrics") for statement in statements)
    assert any(statement.startswith("In contrast, data-driven clustering") for statement in statements)
    assert all("4, 5 " not in statement for statement in statements)
    assert all("7 " not in statement[:3] for statement in statements)
    assert all("mean [SD], 0." not in statement for statement in statements)


def test_section_cleaner_drops_truncated_visual_and_generic_fragments() -> None:
    truncated = (
        "Connectome-wide analysis relating a dimensional reward responsivity measure to resting-state "
        "functional connectivity was conducted in 225 adults across five diagnostic groups "
        "(major depressive disorder, N=32"
    )
    assert synthesis._clean_section_statement(truncated) == ""
    assert synthesis._summary_fragment(truncated, max_chars=180) == ""

    visual_axis_soup = (
        "A B C r Mean connectivity BAS reward sensitivity subscale Nucleus accumbens Right inferior "
        "temporal cortex Right temporoparietal junction Left temporoparietal junction Left orbitofrontal "
        "cortex Right insula Dorsomedial frontal cortex -0.1 0.1 -0.4 0.4 z -1.64 1.64 -4.0 4.0 "
        "Left superior temporal cortex a The multivariate results of the connectome-wide association "
        "study identified the nucleus accumbens, default mode network regions, and cingulo-opercular "
        "network regions where connectivity was related to reward responsivity."
    )
    cleaned = synthesis._clean_section_statement(visual_axis_soup)
    assert cleaned.startswith("The multivariate results")
    assert "Mean connectivity BAS" not in cleaned

    assert synthesis._is_noise_statement("However, certain limitations of our approach should be noted.")


def test_detailed_sections_are_enriched_with_executive_narrative() -> None:
    sections = {
        "methods": {
            "items": [
                {
                    "statement": "Cells were cultured at 37 °C in a humidified atmosphere.",
                    "anchor": "section:Methods:4",
                    "evidence_refs": ["section:Methods:4"],
                    "confidence": 0.7,
                    "section_confidence": 0.7,
                }
            ],
            "evidence_refs": ["section:Methods:4"],
        }
    }
    executive_report = {
        "sections": [
            {
                "section": "methods",
                "summary": (
                    "The method uses a two-vector Double-Flp-In strategy in which complementary frameshift "
                    "edits restore antibiotic resistance only after correct sequential integration."
                ),
                "bullets": [{"anchors": ["section:Methods:1"]}],
            }
        ]
    }
    enriched = synthesis._enrich_detailed_sections_with_executive_report(
        sections,
        executive_report,
        fallback_summary=(
            "Introduction: The paper introduces the Double-Flp-In system as a way to co-express two genes from "
            "a defined genomic locus. Methods: The authors engineered two complementary vectors that restore "
            "antibiotic resistance only after correct sequential integration. Results: The method enabled "
            "selection and validation of double-integrated cells. Discussion: The authors frame the system as "
            "adaptable while noting selection pressure as a limitation. Conclusion: The method may help study "
            "gene-gene interactions."
        ),
    )
    narrative = enriched["methods"]["narrative_items"]
    assert narrative[0]["statement"].startswith("The authors engineered two complementary vectors")
    assert narrative[0]["evidence_refs"] == ["section:Methods:1"]
    assert synthesis._section_item_passes_fidelity(
        "results",
        {"statement": "Cells were cultured at 37 °C in a humidified atmosphere.", "anchor": "section:Results:9"},
    ) is False


def test_filter_result_text_packets_keeps_high_signal_non_numeric_findings() -> None:
    packets = [
        {
            "statement": "Data-driven clustering identified three ADHD biotypes with distinct network patterns.",
            "anchor": "section:Results:7",
            "evidence_refs": ["section:Results:7"],
            "section_label": "results",
            "confidence": 0.76,
            "section_confidence": 0.8,
        }
    ]
    filtered = synthesis._filter_result_text_packets(packets)
    assert len(filtered) == 1
    assert "identified three adhd biotypes" in filtered[0]["statement"].lower()


def test_filter_result_text_packets_drops_visual_annotation_lines() -> None:
    packets = [
        {
            "statement": "Solid dots indicate statistically significant correlations after correction for spatial autocorrelation.",
            "anchor": "section:Results:27",
            "evidence_refs": ["section:Results:27"],
            "section_label": "results",
            "confidence": 0.56,
            "section_confidence": 0.56,
        },
        {
            "statement": "Patterns of attention problems in the validation cohort paralleled those of the discovery sample without statistical significance (H = 0.49; P = .78).",
            "anchor": "section:Results:25",
            "evidence_refs": ["section:Results:25"],
            "section_label": "results",
            "confidence": 0.7,
            "section_confidence": 0.7,
        },
    ]
    filtered = synthesis._filter_result_text_packets(packets)
    assert len(filtered) == 1
    assert "patterns of attention problems" in filtered[0]["statement"].lower()


def test_filter_section_items_by_fidelity_prunes_panel_labels_from_results() -> None:
    items = [
        {
            "statement": "C and D, Nodes exhibiting significant case-control differences in extreme deviation patterns.",
            "anchor": "section:Results:19",
            "evidence_refs": ["section:Results:19"],
            "source_modality": "text",
            "confidence": 0.56,
            "section_confidence": 0.56,
            "flags": ["fallback", "raw_chunk"],
            "result_evidence_type": "text_primary",
        },
        {
            "statement": "The robustness of the identified ADHD biotypes was validated via permutation testing.",
            "anchor": "section:Results:7",
            "evidence_refs": ["section:Results:7"],
            "source_modality": "text",
            "confidence": 0.82,
            "section_confidence": 0.82,
            "flags": [],
            "result_evidence_type": "text_primary",
        },
    ]
    filtered = synthesis._filter_section_items_by_fidelity("results", items, max_items=10)
    assert len(filtered) == 1
    assert "validated via permutation testing" in filtered[0]["statement"].lower()


def test_filter_section_items_by_fidelity_prunes_method_like_conclusion_rows() -> None:
    items = [
        {
            "statement": "A multimetric approach that incorporated 3 topological metrics was used to assess each region's MSN hubness.",
            "anchor": "section:Individual Deviations in Topological Properties:14",
            "evidence_refs": ["section:Individual Deviations in Topological Properties:14"],
            "source_modality": "text",
            "confidence": 0.82,
            "section_confidence": 0.82,
            "flags": ["llm_section_extract"],
        },
        {
            "statement": "Overall, these findings support stratification-oriented models for ADHD heterogeneity.",
            "anchor": "section:Conclusion:1",
            "evidence_refs": ["section:Conclusion:1"],
            "source_modality": "text",
            "confidence": 0.82,
            "section_confidence": 0.82,
            "flags": ["llm_section_extract"],
        },
    ]
    filtered = synthesis._filter_section_items_by_fidelity("conclusion", items, max_items=10)
    assert len(filtered) == 1
    assert "overall, these findings support" in filtered[0]["statement"].lower()


def test_build_presentation_evidence_applies_fidelity_and_anchor_diversity() -> None:
    evidence = {
        "results": [
            {
                "statement": "C and D, Nodes exhibiting significant case-control differences in extreme deviation patterns.",
                "anchor": "section:Results:19",
                "confidence": 0.7,
                "section_confidence": 0.7,
                "source_modality": "text",
                "evidence_refs": ["section:Results:19"],
                "flags": ["fallback", "raw_chunk"],
            },
            {
                "statement": "Data-driven clustering identified three ADHD biotypes with distinct network patterns.",
                "anchor": "section:Results:7",
                "confidence": 0.8,
                "section_confidence": 0.8,
                "source_modality": "text",
                "evidence_refs": ["section:Results:7"],
                "flags": [],
            },
            {
                "statement": "Split-half cross-validation supported stable biotype assignment in independent folds.",
                "anchor": "section:Results:7",
                "confidence": 0.79,
                "section_confidence": 0.79,
                "source_modality": "text",
                "evidence_refs": ["section:Results:7"],
                "flags": [],
            },
            {
                "statement": "Validation cohort symptom trajectories did not differ significantly by biotype over follow-up.",
                "anchor": "section:Results:25",
                "confidence": 0.78,
                "section_confidence": 0.78,
                "source_modality": "text",
                "evidence_refs": ["section:Results:25"],
                "flags": [],
            },
        ],
        "conclusion": [
            {
                "statement": "To examine longitudinal changes, we analyzed follow-up data using linear mixed models.",
                "anchor": "section:Biotype Identification:9",
                "confidence": 0.82,
                "section_confidence": 0.82,
                "source_modality": "text",
                "evidence_refs": ["section:Biotype Identification:9"],
                "flags": ["llm_section_extract"],
            },
            {
                "statement": "Overall, these findings support stratification-oriented models for ADHD heterogeneity.",
                "anchor": "section:Conclusion:1",
                "confidence": 0.82,
                "section_confidence": 0.82,
                "source_modality": "text",
                "evidence_refs": ["section:Conclusion:1"],
                "flags": ["llm_section_extract"],
            },
        ],
    }
    presentation = synthesis._build_presentation_evidence(extractive_evidence=evidence)
    results = presentation["results"]
    conclusion = presentation["conclusion"]
    assert len(results) == 3
    assert all("nodes exhibiting" not in str(item.get("statement", "")).lower() for item in results)
    assert sum(1 for item in results if str(item.get("anchor", "")) == "section:Results:7") <= 2
    assert len(conclusion) == 1
    assert "overall, these findings support" in str(conclusion[0].get("statement", "")).lower()


def test_structured_abstract_split_emits_section_norms() -> None:
    tei = """
    <TEI xmlns="http://www.tei-c.org/ns/1.0">
      <text>
        <front>
          <abstract>
            Objective: Evaluate transdiagnostic reward deficits. Method: Analyze resting-state connectivity.
            Results: Identified network-level dysconnectivity. Conclusions: Corticostriatal abnormalities were central.
          </abstract>
        </front>
      </text>
    </TEI>
    """
    chunks = validated_pipeline._tei_to_text_chunks(tei)
    norms = []
    for chunk in chunks:
        meta = json.loads(chunk["meta"])
        norms.append(meta.get("section_norm"))
    assert norms[:4] == ["introduction", "methods", "results", "conclusion"]


def test_tei_figure_legend_extracts_caption_only_figures() -> None:
    tei = """
    <TEI xmlns="http://www.tei-c.org/ns/1.0">
      <text>
        <body>
          <figure type="figure" coords="5,100,100,300,200">
            <head>Figure 4.</head>
            <figDesc>Multiplex and single PCR confirmed vector integration in the intended order.</figDesc>
          </figure>
        </body>
      </text>
    </TEI>
    """
    figures = validated_pipeline._tei_figure_legend_extract(tei)

    assert len(figures) == 1
    assert figures[0].path is None
    assert figures[0].page == 5
    assert figures[0].meta["source"] == "grobid_tei_figure"
    assert "Multiplex and single PCR" in str(figures[0].caption)


def test_wet_lab_method_heading_is_classified_as_methods() -> None:
    text = (
        "The second vector was built to include the same promoter and FRT site. "
        "Fragments were generated by PCR using hybrid primers and recombined into plasmids."
    )

    assert validated_pipeline._infer_section_from_text(text, idx=20, total_chunks=32) == "methods"


def test_text_section_conflict_override_prefers_statement_prefix() -> None:
    packets = text_analysis._annotate_text_packet_sections(
        [
            {
                "finding_id": "p1",
                "anchor": "section:Results:2",
                "statement": "Objective: Evaluate transdiagnostic reward deficits across disorders.",
                "evidence_refs": ["section:Results:2"],
                "confidence": 0.9,
                "category": "results",
                "quality_flags": [],
            }
        ]
    )
    assert packets[0]["section_label"] == "introduction"
    assert "section_conflict_resolved" in packets[0]["quality_flags"]


def test_cross_section_dedupe_keeps_higher_confidence_section_owner() -> None:
    sections = {
        "introduction": {"items": [], "evidence_refs": [], "fallback_used": False, "fallback_reason": None},
        "methods": {"items": [], "evidence_refs": [], "fallback_used": False, "fallback_reason": None},
        "results": {
            "items": [
                {
                    "statement": "Connectivity increased in the default mode network (p<0.01).",
                    "anchor": "section:Results:4",
                    "evidence_refs": ["section:Results:4"],
                    "source_modality": "text",
                    "section_source": "anchor",
                    "confidence": 0.92,
                    "section_confidence": 0.94,
                    "flags": [],
                }
            ],
            "evidence_refs": ["section:Results:4"],
            "fallback_used": False,
            "fallback_reason": None,
        },
        "discussion": {"items": [], "evidence_refs": [], "fallback_used": False, "fallback_reason": None},
        "conclusion": {
            "items": [
                {
                    "statement": "Connectivity increased in the default mode network (p<0.01).",
                    "anchor": "section:Discussion:20",
                    "evidence_refs": ["section:Discussion:20"],
                    "source_modality": "text",
                    "section_source": "fallback",
                    "confidence": 0.61,
                    "section_confidence": 0.60,
                    "flags": ["fallback"],
                }
            ],
            "evidence_refs": ["section:Discussion:20"],
            "fallback_used": True,
            "fallback_reason": "No explicit conclusion heading detected.",
        },
    }
    deduped, diagnostics = synthesis._dedupe_items_across_sections(sections)
    assert diagnostics["removed_count"] == 1
    assert len(deduped["results"]["items"]) == 1
    assert deduped["results"]["items"][0]["anchor"] == "section:Results:4"
    assert deduped["conclusion"]["items"] == []


def test_sections_compact_cross_section_dedupe_replaces_duplicate_found_slot() -> None:
    compact = {
        "introduction": [],
        "methods": [
            {
                "section_key": "methods",
                "slot_key": "study_design",
                "label": "Study Design",
                "statement": "Our results corroborate previous research and emphasize corticostriatal dysconnectivity.",
                "status": "found",
                "evidence_refs": ["section:Methods:5"],
                "confidence": 0.58,
            }
        ],
        "results": [],
        "discussion": [],
        "conclusion": [
            {
                "section_key": "conclusion",
                "slot_key": "takeaway",
                "label": "Takeaway",
                "statement": "Our results corroborate previous research and emphasize corticostriatal dysconnectivity.",
                "status": "found",
                "evidence_refs": ["section:Conclusion:1"],
                "confidence": 0.84,
            }
        ],
    }
    deduped, diagnostics = synthesis._dedupe_sections_compact_rows(compact, access_limited=False)
    assert diagnostics["removed_count"] == 1
    assert deduped["conclusion"][0]["status"] == "found"
    assert deduped["methods"][0]["status"] == "not_found"
    assert deduped["methods"][0]["evidence_refs"] == []


def test_sections_compact_dedupe_keeps_protected_conclusion_slot() -> None:
    compact = {
        "introduction": [
            {
                "section_key": "introduction",
                "slot_key": "background_gap",
                "label": "Background/Gap",
                "statement": "Overall, the study supports a shared transdiagnostic network mechanism.",
                "status": "found",
                "evidence_refs": ["section:Introduction:2"],
                "confidence": 0.76,
            }
        ],
        "methods": [],
        "results": [],
        "discussion": [],
        "conclusion": [
            {
                "section_key": "conclusion",
                "slot_key": "main_takeaway",
                "label": "Main Takeaway",
                "statement": "Overall, the study supports a shared transdiagnostic network mechanism.",
                "status": "found",
                "evidence_refs": ["section:Conclusion:1"],
                "confidence": 0.86,
            }
        ],
    }
    deduped, diagnostics = synthesis._dedupe_sections_compact_rows(compact, access_limited=False)
    assert diagnostics["removed_count"] == 0
    assert deduped["introduction"][0]["status"] == "found"
    assert deduped["conclusion"][0]["status"] == "found"


def test_methods_compact_avoids_duplicate_fallback_reuse() -> None:
    packet = {
        "finding_id": "m1",
        "anchor": "section:Methods:1",
        "statement": (
            "Participants were enrolled in a randomized design with predefined inclusion and exclusion criteria, "
            "and analyses used covariate-adjusted regression models."
        ),
        "evidence_refs": ["section:Methods:1"],
        "confidence": 0.9,
        "quality_flags": [],
        "section_label": "methods",
        "category": "methods",
    }
    rows = synthesis._methods_compact([packet], analysis_notes=[])
    found = [row for row in rows if str(row.get("status", "")).lower() == "found"]
    keys = {synthesis._canonical_statement_text(str(row.get("statement", ""))) for row in found}
    assert len(keys) == len(found)


def test_methods_compact_prefers_distinct_anchor_when_alternative_exists() -> None:
    packets = [
        {
            "finding_id": "m1",
            "anchor": "section:Methods:1",
            "statement": "Study design used a randomized cross-sectional trial framework with preregistered hypotheses.",
            "evidence_refs": ["section:Methods:1"],
            "confidence": 0.94,
            "quality_flags": [],
            "section_label": "methods",
            "category": "methods",
        },
        {
            "finding_id": "m2",
            "anchor": "section:Methods:1",
            "statement": "Participants in the sample included five diagnostic groups with balanced demographics.",
            "evidence_refs": ["section:Methods:1"],
            "confidence": 0.96,
            "quality_flags": [],
            "section_label": "methods",
            "category": "methods",
        },
        {
            "finding_id": "m3",
            "anchor": "section:Methods:2",
            "statement": "The cohort included 244 participants across five diagnostic groups assessed with harmonized procedures.",
            "evidence_refs": ["section:Methods:2"],
            "confidence": 0.72,
            "quality_flags": [],
            "section_label": "methods",
            "category": "methods",
        },
    ]
    rows = synthesis._methods_compact(packets, analysis_notes=[])
    study_design = next(row for row in rows if row["slot_key"] == "study_design")
    sample_population = next(row for row in rows if row["slot_key"] == "sample_population")
    assert study_design["status"] == "found"
    assert sample_population["status"] == "found"
    assert study_design["evidence_refs"][0] != sample_population["evidence_refs"][0]


def test_build_detailed_sections_conclusion_recovers_from_discussion_pool() -> None:
    text_packets = [
        {
            "finding_id": "d1",
            "anchor": "section:Discussion:10",
            "statement": "Overall, these findings suggest a shared reward-network mechanism across diagnoses.",
            "evidence_refs": ["section:Discussion:10"],
            "confidence": 0.85,
            "section_confidence": 0.9,
            "section_label": "discussion",
            "section_source": "anchor",
            "category": "discussion",
        },
        {
            "finding_id": "d2",
            "anchor": "section:Discussion:11",
            "statement": "Clinical implications include targeting nucleus accumbens circuitry in intervention design.",
            "evidence_refs": ["section:Discussion:11"],
            "confidence": 0.83,
            "section_confidence": 0.88,
            "section_label": "discussion",
            "section_source": "anchor",
            "category": "discussion",
        },
        {
            "finding_id": "d3",
            "anchor": "section:Discussion:12",
            "statement": "Future longitudinal work should test whether these biomarkers predict relapse trajectories.",
            "evidence_refs": ["section:Discussion:12"],
            "confidence": 0.81,
            "section_confidence": 0.86,
            "section_label": "discussion",
            "section_source": "anchor",
            "category": "discussion",
        },
        {
            "finding_id": "d4",
            "anchor": "section:Discussion:13",
            "statement": "Interpretation should remain cautious given cross-sectional design and potential residual confounding.",
            "evidence_refs": ["section:Discussion:13"],
            "confidence": 0.78,
            "section_confidence": 0.82,
            "section_label": "discussion",
            "section_source": "anchor",
            "category": "discussion",
        },
    ]
    sections, diagnostics, _fallback_notes = synthesis._build_detailed_sections(
        text_packets=text_packets,
        table_packets=[],
        figure_packets=[],
        supp_packets=[],
        methods_compact=[],
        analysis_notes=[],
        text_chunk_records=[],
        sections_extracted={},
    )
    assert len(sections["discussion"]["items"]) >= 4
    assert len(sections["conclusion"]["items"]) >= 3
    assert diagnostics["conclusion"]["fallback_used"] is True


def test_cross_section_dedupe_preserves_conclusion_min_keep_when_dense() -> None:
    shared_statements = [
        "Overall, these findings suggest a shared transdiagnostic mechanism.",
        "Clinical implications support targeting reward-circuit connectivity.",
        "Future longitudinal studies are needed to test causal pathways.",
        "Interpretation should remain cautious due to cross-sectional design.",
    ]
    discussion_items = [
        {
            "statement": text,
            "anchor": f"section:Discussion:{idx + 1}",
            "evidence_refs": [f"section:Discussion:{idx + 1}"],
            "confidence": 0.82,
            "section_confidence": 0.82,
            "section_source": "anchor",
            "flags": [],
        }
        for idx, text in enumerate(shared_statements)
    ]
    conclusion_items = [
        {
            "statement": text,
            "anchor": f"section:Conclusion:{idx + 1}",
            "evidence_refs": [f"section:Conclusion:{idx + 1}"],
            "confidence": 0.74,
            "section_confidence": 0.74,
            "section_source": "anchor",
            "flags": [],
        }
        for idx, text in enumerate(shared_statements)
    ]
    sections = {
        "introduction": {"items": [], "evidence_refs": [], "fallback_used": False, "fallback_reason": None},
        "methods": {"items": [], "evidence_refs": [], "fallback_used": False, "fallback_reason": None},
        "results": {"items": [], "evidence_refs": [], "fallback_used": False, "fallback_reason": None},
        "discussion": {"items": discussion_items, "evidence_refs": [], "fallback_used": False, "fallback_reason": None},
        "conclusion": {"items": conclusion_items, "evidence_refs": [], "fallback_used": False, "fallback_reason": None},
    }
    deduped, diagnostics = synthesis._dedupe_items_across_sections(sections)
    assert diagnostics["removed_count"] >= 1
    assert len(deduped["conclusion"]["items"]) >= 3


def test_section_verifier_rejects_destructive_empty_output(monkeypatch) -> None:
    def _item(section: str, idx: int, statement: str) -> dict:
        anchor_section = section.title()
        return {
            "statement": statement,
            "anchor": f"section:{anchor_section}:{idx}",
            "evidence_refs": [f"section:{anchor_section}:{idx}"],
            "confidence": 0.82,
            "section_confidence": 0.86,
            "section_source": "anchor",
            "flags": [],
        }

    sections = {
        "introduction": {"items": [], "evidence_refs": [], "fallback_used": False, "fallback_reason": None},
        "methods": {
            "items": [
                _item("methods", 1, "Cells were transfected with expression vectors using a Flp-In protocol."),
                _item("methods", 2, "Stable clones were selected with hygromycin and puromycin before validation."),
                _item("methods", 3, "Expression was quantified by quantitative PCR after vector integration."),
                _item("methods", 4, "Transport experiments were performed in engineered cell lines."),
            ],
            "evidence_refs": [],
            "fallback_used": False,
            "fallback_reason": None,
        },
        "results": {
            "items": [
                _item("results", 1, "Transport activity increased by 40% in the double-transfected cells."),
                _item("results", 2, "Vector-order validation found the expected integration pattern in selected clones."),
                _item("results", 3, "Gene expression was higher and remained stable over repeated culture passages."),
                _item("results", 4, "Drug conversion was significantly higher between engineered cell lines."),
                _item("results", 5, "Figure assays showed higher metabolite formation in CYP2C19-expressing cells."),
            ],
            "evidence_refs": [],
            "fallback_used": False,
            "fallback_reason": None,
        },
        "discussion": {
            "items": [
                _item("discussion", 1, "These findings suggest the double-Flp-In strategy can support paired gene studies."),
                _item("discussion", 2, "The authors note important limitations for multiplex PCR screening."),
                _item("discussion", 3, "The work may help interpret transporter and enzyme interactions."),
                _item("discussion", 4, "Further validation is needed before broader pharmacogenomic use."),
            ],
            "evidence_refs": [],
            "fallback_used": False,
            "fallback_reason": None,
        },
        "conclusion": {
            "items": [
                _item("conclusion", 1, "Overall, the findings support a durable platform for studying dual gene effects."),
                _item("conclusion", 2, "Taken together, these findings indicate the approach can clarify gene interactions."),
                _item("conclusion", 3, "In conclusion, the platform provides a basis for future transport studies."),
            ],
            "evidence_refs": [],
            "fallback_used": False,
            "fallback_reason": None,
        },
    }

    monkeypatch.setattr(synthesis.settings, "analysis_section_verifier_enabled", True)
    monkeypatch.setattr(
        synthesis,
        "_run_deep_json_prompt",
        lambda **_kwargs: {"sections": {section: [] for section in synthesis.EXEC_REPORT_SECTION_ORDER}},
    )

    verified, diagnostics = synthesis._verify_section_fidelity_with_llm(sections, payload={})

    assert diagnostics["applied"] is False
    assert diagnostics["rejected"] is True
    assert diagnostics["emptied_sections"] == ["methods", "results", "discussion"]
    assert len(verified["methods"]["items"]) == 4
    assert len(verified["results"]["items"]) == 5
    assert len(verified["discussion"]["items"]) == 4
    assert len(verified["conclusion"]["items"]) == 3


def test_llm_section_extraction_falls_back_to_rows_when_llm_errors(monkeypatch) -> None:
    def _raise_chat(prompt: str, system: str | None = None, temperature: float = 0.2) -> str:
        raise RuntimeError("llm unavailable")

    monkeypatch.setattr(synthesis, "chat_text_deep", _raise_chat)
    monkeypatch.setattr(synthesis.settings, "analysis_section_extraction_subprocess_guard_enabled", False)

    payload = {
        "text_packets": [
            {
                "finding_id": "m1",
                "anchor": "section:Methods:1",
                "statement": "Participants were recruited into five diagnostic groups.",
                "evidence_refs": ["section:Methods:1"],
                "confidence": 0.82,
                "category": "methods",
                "section_label": "methods",
            },
            {
                "finding_id": "r1",
                "anchor": "section:Results:1",
                "statement": "Higher reward sensitivity was associated with increased connectivity (p<0.01).",
                "evidence_refs": ["section:Results:1"],
                "confidence": 0.84,
                "category": "results",
                "section_label": "results",
            },
            {
                "finding_id": "d1",
                "anchor": "section:Discussion:1",
                "statement": "These findings suggest a shared transdiagnostic mechanism with important limitations.",
                "evidence_refs": ["section:Discussion:1"],
                "confidence": 0.8,
                "category": "discussion",
                "section_label": "discussion",
            },
            {
                "finding_id": "c1",
                "anchor": "section:Conclusion:1",
                "statement": "Overall, the study supports a network-level conclusion and future longitudinal work.",
                "evidence_refs": ["section:Conclusion:1"],
                "confidence": 0.8,
                "category": "conclusion",
                "section_label": "conclusion",
            },
        ],
        "table_packets": [],
        "figure_packets": [],
        "supp_packets": [],
    }
    extracted = synthesis._llm_section_extraction(payload)
    assert len(extracted.get("methods", [])) >= 1
    assert len(extracted.get("results", [])) >= 1
    assert len(extracted.get("discussion", [])) >= 1
    assert len(extracted.get("conclusion", [])) >= 1


def test_sections_compact_uses_sections_extracted_candidates() -> None:
    methods_compact = synthesis._methods_compact([], analysis_notes=[])
    sections_compact = synthesis._sections_compact(
        text_packets=[],
        methods_compact=methods_compact,
        analysis_notes=[],
        text_chunk_records=[],
        result_support_packets=[],
        sections_extracted={
            "discussion": [
                {
                    "statement": "These findings suggest a shared mechanism and important implications for interpretation.",
                    "evidence_refs": ["section:Discussion:1"],
                    "kind": "interpretation",
                }
            ],
            "conclusion": [
                {
                    "statement": "Overall, the conclusion emphasizes transdiagnostic network dysfunction and future research.",
                    "evidence_refs": ["section:Conclusion:1"],
                    "kind": "takeaway",
                }
            ],
        },
    )
    assert any(str(row.get("status", "")).lower() == "found" for row in sections_compact["discussion"])
    assert any(str(row.get("status", "")).lower() == "found" for row in sections_compact["conclusion"])


def test_section_compact_candidates_filters_results_like_line_from_methods() -> None:
    candidates = synthesis._section_compact_candidates(
        "methods",
        [
            {
                "finding_id": "x1",
                "anchor": "section:Methods:1",
                "statement": "Our results corroborate prior studies and emphasize network dysconnectivity.",
                "evidence_refs": ["section:Methods:1"],
                "confidence": 0.88,
                "quality_flags": [],
            }
        ],
    )
    assert candidates == []


def test_sections_compact_has_fixed_slot_counts_without_cross_section_fallback() -> None:
    summary = synthesize_report(
        {
            "evidence_packets": [
                {
                    "finding_id": "m1",
                    "modality": "text",
                    "anchor": "section:Methods:1",
                    "statement": "Randomized cohort design with MDMR and covariates.",
                    "evidence_refs": ["section:Methods:1"],
                    "confidence": 0.9,
                    "quality_flags": [],
                    "category": "methods",
                    "section_label": "methods",
                    "section_confidence": 0.95,
                    "section_source": "anchor",
                }
            ]
        },
        {"evidence_packets": []},
        {"evidence_packets": []},
        {"evidence_packets": []},
        {"cross_modal_claims": [], "discrepancies": []},
        paper_meta={},
        coverage={"figures": {}, "tables": {}, "supp_figures": {}, "supp_tables": {}},
    )
    compact = summary["sections_compact"]
    assert len(compact["introduction"]) == 3
    assert len(compact["methods"]) == 12
    assert len(compact["results"]) == 5
    assert len(compact["discussion"]) == 4
    assert len(compact["conclusion"]) == 2
    assert all(slot["status"] == "not_found" for slot in compact["introduction"])
    assert all(slot["status"] == "not_found" for slot in compact["results"])


def test_sections_compact_statement_length_and_sentence_cap() -> None:
    long_statement = (
        "Reward deficits were associated with increased default mode connectivity and decreased cingulo-opercular integration "
        "with p<0.01 and robust subgroup effects. Additional narrative sentence that should be excluded."
    )
    summary = synthesize_report(
        {
            "evidence_packets": [
                {
                    "finding_id": "r1",
                    "modality": "text",
                    "anchor": "section:Results:10",
                    "statement": long_statement,
                    "evidence_refs": ["section:Results:10"],
                    "confidence": 0.9,
                    "quality_flags": [],
                    "category": "results",
                    "section_label": "results",
                    "section_confidence": 0.95,
                    "section_source": "anchor",
                }
            ]
        },
        {"evidence_packets": []},
        {"evidence_packets": []},
        {"evidence_packets": []},
        {"cross_modal_claims": [], "discrepancies": []},
        paper_meta={},
        coverage={"figures": {}, "tables": {}, "supp_figures": {}, "supp_tables": {}},
    )
    slot = summary["sections_compact"]["results"][0]
    assert len(slot["statement"]) <= 160
    assert "Additional narrative sentence" not in slot["statement"]


def test_sections_compact_is_deterministic_for_same_input() -> None:
    text_report = {
        "evidence_packets": [
            {
                "finding_id": "m1",
                "modality": "text",
                "anchor": "section:Methods:1",
                "statement": "Randomized cohort design with covariate-adjusted mixed-effects regression.",
                "evidence_refs": ["section:Methods:1"],
                "confidence": 0.9,
                "quality_flags": [],
                "category": "methods",
                "section_label": "methods",
                "section_confidence": 0.95,
                "section_source": "anchor",
            }
        ]
    }
    left = synthesize_report(
        text_report,
        {"evidence_packets": []},
        {"evidence_packets": []},
        {"evidence_packets": []},
        {"cross_modal_claims": [], "discrepancies": []},
        paper_meta={},
        coverage={"figures": {}, "tables": {}, "supp_figures": {}, "supp_tables": {}},
    )
    right = synthesize_report(
        text_report,
        {"evidence_packets": []},
        {"evidence_packets": []},
        {"evidence_packets": []},
        {"cross_modal_claims": [], "discrepancies": []},
        paper_meta={},
        coverage={"figures": {}, "tables": {}, "supp_figures": {}, "supp_tables": {}},
    )
    assert left["sections_compact"] == right["sections_compact"]
    assert left["executive_summary"] == right["executive_summary"]


def test_summary_polish_validator_rejects_numeric_drift() -> None:
    original = (
        "Introduction: objective defined. Methods: cohort and regression. Results: effect size 0.4. "
        "Discussion: interpretation cautious. Conclusion: provisional."
    )
    candidate = (
        "Introduction: objective defined. Methods: cohort and regression. Results: effect size 999. "
        "Discussion: interpretation cautious. Conclusion: provisional."
    )
    assert synthesis._summary_polish_valid(original, candidate) is False


def test_executive_summary_strips_confidence_annotations() -> None:
    text_report = {
        "evidence_packets": [
            {
                "finding_id": "i1",
                "modality": "text",
                "anchor": "section:Introduction:1",
                "statement": "Objective was to evaluate reward deficits. (confidence 88%)",
                "evidence_refs": ["section:Introduction:1"],
                "confidence": 0.9,
                "quality_flags": [],
                "category": "introduction",
                "section_label": "introduction",
                "section_confidence": 0.95,
                "section_source": "anchor",
            },
            {
                "finding_id": "m1",
                "modality": "text",
                "anchor": "section:Methods:1",
                "statement": "Randomized cohort design with covariate-adjusted model. (confidence 75%)",
                "evidence_refs": ["section:Methods:1"],
                "confidence": 0.9,
                "quality_flags": [],
                "category": "methods",
                "section_label": "methods",
                "section_confidence": 0.95,
                "section_source": "anchor",
            },
            {
                "finding_id": "r1",
                "modality": "text",
                "anchor": "section:Results:1",
                "statement": "Results identified higher connectivity with p<0.01. (70%)",
                "evidence_refs": ["section:Results:1"],
                "confidence": 0.9,
                "quality_flags": [],
                "category": "results",
                "section_label": "results",
                "section_confidence": 0.95,
                "section_source": "anchor",
            },
            {
                "finding_id": "d1",
                "modality": "text",
                "anchor": "section:Discussion:1",
                "statement": "Interpretation was cautious due to cross-sectional design. (confidence 65%)",
                "evidence_refs": ["section:Discussion:1"],
                "confidence": 0.9,
                "quality_flags": [],
                "category": "discussion",
                "section_label": "discussion",
                "section_confidence": 0.95,
                "section_source": "anchor",
            },
            {
                "finding_id": "c1",
                "modality": "text",
                "anchor": "section:Conclusion:1",
                "statement": "Conclusion suggests provisional clinical relevance. (confidence 60%)",
                "evidence_refs": ["section:Conclusion:1"],
                "confidence": 0.9,
                "quality_flags": [],
                "category": "conclusion",
                "section_label": "conclusion",
                "section_confidence": 0.95,
                "section_source": "anchor",
            },
        ]
    }
    summary = synthesize_report(
        text_report,
        {"evidence_packets": []},
        {"evidence_packets": []},
        {"evidence_packets": []},
        {"cross_modal_claims": [], "discrepancies": []},
        paper_meta={},
        coverage={"figures": {}, "tables": {}, "supp_figures": {}, "supp_tables": {}},
    )
    assert "(confidence" not in summary["executive_summary"].lower()
    assert "(70%)" not in summary["executive_summary"].lower()


def test_results_compact_uses_media_support_when_text_results_sparse() -> None:
    summary = synthesize_report(
        {
            "evidence_packets": [
                {
                    "finding_id": "m1",
                    "modality": "text",
                    "anchor": "section:Methods:1",
                    "statement": "Randomized cohort with covariate-adjusted analysis.",
                    "evidence_refs": ["section:Methods:1"],
                    "confidence": 0.9,
                    "quality_flags": [],
                    "category": "methods",
                    "section_label": "methods",
                    "section_confidence": 0.95,
                    "section_source": "anchor",
                }
            ]
        },
        {"evidence_packets": []},
        {
            "evidence_packets": [
                {
                    "finding_id": "f1",
                    "modality": "figure",
                    "anchor": "figure:2",
                    "statement": "Figure 2 identified increased reward-network connectivity with significant effects (p<0.01).",
                    "evidence_refs": ["figure:2"],
                    "confidence": 0.9,
                    "quality_flags": [],
                    "category": "results",
                }
            ]
        },
        {"evidence_packets": []},
        {"cross_modal_claims": [], "discrepancies": []},
        paper_meta={},
        coverage={"figures": {}, "tables": {}, "supp_figures": {}, "supp_tables": {}},
    )
    result_slots = summary["sections_compact"]["results"]
    assert any(slot["status"] == "found" for slot in result_slots)
    assert any(
        slot["status"] == "found" and any(str(ref).lower().startswith("figure:2") for ref in slot.get("evidence_refs", []))
        for slot in result_slots
    )


def test_section_extraction_drives_executive_summary_when_enabled(monkeypatch) -> None:
    def _fake_chat(prompt: str, system: str | None = None, temperature: float = 0.2) -> str:
        if system == synthesis.SECTION_EXTRACTION_SYSTEM:
            return json.dumps(
                {
                    "introduction": [
                        {"statement": "Background framing emphasized transdiagnostic reward impairment.", "evidence_refs": ["section:Introduction:1"]}
                    ],
                    "methods": [
                        {"statement": "Methods used a covariate-adjusted randomized cohort design.", "evidence_refs": ["section:Methods:1"]}
                    ],
                    "results": [
                        {"statement": "Results identified higher connectivity with p<0.01.", "evidence_refs": ["section:Results:1"]}
                    ],
                    "discussion": [
                        {"statement": "Discussion interpreted findings cautiously due to cross-sectional data.", "evidence_refs": ["section:Discussion:1"]}
                    ],
                    "conclusion": [
                        {"statement": "Conclusion emphasized provisional clinical implications.", "evidence_refs": ["section:Conclusion:1"]}
                    ],
                }
            )
        return "{}"

    monkeypatch.setattr(synthesis, "chat_text_deep", _fake_chat)
    monkeypatch.setattr(synthesis.settings, "analysis_section_extraction_enabled", True)
    monkeypatch.setattr(synthesis.settings, "analysis_section_extraction_subprocess_guard_enabled", False)
    summary = synthesize_report(
        {
            "evidence_packets": [
                {
                    "finding_id": "i1",
                    "modality": "text",
                    "anchor": "section:Introduction:1",
                    "statement": "Background focused on transdiagnostic reward impairment.",
                    "evidence_refs": ["section:Introduction:1"],
                    "confidence": 0.9,
                    "quality_flags": [],
                    "category": "introduction",
                    "section_label": "introduction",
                    "section_confidence": 0.95,
                    "section_source": "anchor",
                },
                {
                    "finding_id": "m1",
                    "modality": "text",
                    "anchor": "section:Methods:1",
                    "statement": "Randomized cohort design with covariate-adjusted model.",
                    "evidence_refs": ["section:Methods:1"],
                    "confidence": 0.9,
                    "quality_flags": [],
                    "category": "methods",
                    "section_label": "methods",
                    "section_confidence": 0.95,
                    "section_source": "anchor",
                },
                {
                    "finding_id": "r1",
                    "modality": "text",
                    "anchor": "section:Results:1",
                    "statement": "Higher connectivity was associated with anhedonia (p<0.01).",
                    "evidence_refs": ["section:Results:1"],
                    "confidence": 0.9,
                    "quality_flags": [],
                    "category": "results",
                    "section_label": "results",
                    "section_confidence": 0.95,
                    "section_source": "anchor",
                },
                {
                    "finding_id": "d1",
                    "modality": "text",
                    "anchor": "section:Discussion:1",
                    "statement": "Interpretation was constrained by cross-sectional design.",
                    "evidence_refs": ["section:Discussion:1"],
                    "confidence": 0.9,
                    "quality_flags": [],
                    "category": "discussion",
                    "section_label": "discussion",
                    "section_confidence": 0.95,
                    "section_source": "anchor",
                },
                {
                    "finding_id": "c1",
                    "modality": "text",
                    "anchor": "section:Conclusion:1",
                    "statement": "Conclusion suggested provisional clinical relevance.",
                    "evidence_refs": ["section:Conclusion:1"],
                    "confidence": 0.9,
                    "quality_flags": [],
                    "category": "conclusion",
                    "section_label": "conclusion",
                    "section_confidence": 0.95,
                    "section_source": "anchor",
                },
            ]
        },
        {"evidence_packets": []},
        {"evidence_packets": []},
        {"evidence_packets": []},
        {"cross_modal_claims": [], "discrepancies": []},
        paper_meta={},
        coverage={"figures": {}, "tables": {}, "supp_figures": {}, "supp_tables": {}},
    )
    assert summary["sections_extracted_version"] == 1
    assert "Background framing emphasized transdiagnostic reward impairment" in summary["executive_summary"]
    assert "Methods used a covariate-adjusted randomized cohort design" in summary["executive_summary"]


def test_executive_report_synthesis_keeps_focus_fields_inside_sections(monkeypatch) -> None:
    def _fake_chat(prompt: str, system: str | None = None, temperature: float = 0.2) -> str:
        if system == synthesis.EXECUTIVE_REPORT_SYNTHESIS_SYSTEM:
            assert "section:Introduction:1" in prompt
            return json.dumps(
                {
                    "sections": [
                        {
                            "section": "introduction",
                            "summary": "This data brief presents findings from the 2024 NSDUH on GAD symptoms among adults.",
                            "study_purpose": "This data brief presents findings from the 2024 NSDUH on GAD symptoms among adults.",
                            "study_hypothesis": "",
                            "central_finding": "",
                        },
                        {
                            "section": "methods",
                            "summary": "Responses to the GAD-7 questions were compiled into a score ranging from 0 to 21.",
                            "study_purpose": "",
                            "study_hypothesis": "",
                            "central_finding": "",
                        },
                        {
                            "section": "results",
                            "summary": "In 2024, 7.4% of adults had moderate or severe symptoms of GAD in the past 2 weeks.",
                            "study_purpose": "",
                            "study_hypothesis": "",
                            "central_finding": "In 2024, 7.4% of adults had moderate or severe symptoms of GAD in the past 2 weeks.",
                        },
                        {
                            "section": "discussion",
                            "summary": "The brief advises caution in interpreting the high percentage.",
                            "study_purpose": "",
                            "study_hypothesis": "",
                            "central_finding": "",
                        },
                        {
                            "section": "conclusion",
                            "summary": "The brief noted sociodemographic and geographic differences in moderate or severe GAD symptoms.",
                            "study_purpose": "",
                            "study_hypothesis": "",
                            "central_finding": "",
                        },
                    ]
                }
            )
        return "{}"

    monkeypatch.setattr(synthesis, "chat_text_deep", _fake_chat)
    monkeypatch.setattr(synthesis.settings, "analysis_narrative_overrides_enabled", True)
    monkeypatch.setattr(synthesis.settings, "analysis_narrative_overrides_subprocess_guard_enabled", False)
    monkeypatch.setattr(synthesis.settings, "analysis_section_extraction_enabled", False)
    monkeypatch.setattr(synthesis.settings, "analysis_verifier_enabled", False)

    summary = synthesize_report(
        {
            "evidence_packets": [
                {
                    "finding_id": "i1",
                    "modality": "text",
                    "anchor": "section:Introduction:1",
                    "statement": "This data brief presents findings from the 2024 NSDUH on GAD symptoms among adults.",
                    "evidence_refs": ["section:Introduction:1"],
                    "confidence": 0.9,
                    "quality_flags": [],
                    "category": "introduction",
                    "section_label": "introduction",
                    "section_confidence": 0.95,
                    "section_source": "anchor",
                },
                {
                    "finding_id": "m1",
                    "modality": "text",
                    "anchor": "section:Methods:1",
                    "statement": "Responses to the GAD-7 questions were compiled into a score ranging from 0 to 21.",
                    "evidence_refs": ["section:Methods:1"],
                    "confidence": 0.9,
                    "quality_flags": [],
                    "category": "methods",
                    "section_label": "methods",
                    "section_confidence": 0.95,
                    "section_source": "anchor",
                },
                {
                    "finding_id": "r1",
                    "modality": "text",
                    "anchor": "section:Results:1",
                    "statement": "In 2024, 7.4% of adults had moderate or severe symptoms of GAD in the past 2 weeks.",
                    "evidence_refs": ["section:Results:1"],
                    "confidence": 0.9,
                    "quality_flags": [],
                    "category": "results",
                    "section_label": "results",
                    "section_confidence": 0.95,
                    "section_source": "anchor",
                },
                {
                    "finding_id": "d1",
                    "modality": "text",
                    "anchor": "section:Discussion:1",
                    "statement": "The brief advises caution in interpreting the high percentage.",
                    "evidence_refs": ["section:Discussion:1"],
                    "confidence": 0.9,
                    "quality_flags": [],
                    "category": "discussion",
                    "section_label": "discussion",
                    "section_confidence": 0.95,
                    "section_source": "anchor",
                },
                {
                    "finding_id": "c1",
                    "modality": "text",
                    "anchor": "section:Conclusion:1",
                    "statement": "The brief noted sociodemographic and geographic differences in moderate or severe GAD symptoms.",
                    "evidence_refs": ["section:Conclusion:1"],
                    "confidence": 0.9,
                    "quality_flags": [],
                    "category": "conclusion",
                    "section_label": "conclusion",
                    "section_confidence": 0.95,
                    "section_source": "anchor",
                },
            ]
        },
        {"evidence_packets": []},
        {"evidence_packets": []},
        {"evidence_packets": []},
        {"cross_modal_claims": [], "discrepancies": []},
        paper_meta={},
        coverage={"figures": {}, "tables": {}, "supp_figures": {}, "supp_tables": {}},
    )

    executive_report = summary["executive_report"]
    assert executive_report["style"] == "llm_section_synthesis_v1"
    intro = next(row for row in executive_report["sections"] if row["section"] == "introduction")
    results = next(row for row in executive_report["sections"] if row["section"] == "results")
    assert intro["study_purpose"].startswith("This data brief presents findings")
    assert "study_hypothesis" not in intro
    assert results["central_finding"].startswith("In 2024, 7.4%")
    assert "Study Purpose:" not in summary["executive_summary"]


def test_section_synthesis_v2_uses_extracted_sections_before_backfill() -> None:
    summary = {
        "section_diagnostics": {"paper_type": "laboratory/preclinical study"},
        "sections_extracted": {
            "methods": [
                {
                    "statement": "Cells were engineered with a Flp-In strategy using FRT sites.",
                    "evidence_refs": ["section:Methods:1"],
                },
                {
                    "statement": "Stable clones were selected and validated by PCR.",
                    "evidence_refs": ["section:Methods:2"],
                },
                {
                    "statement": "Gene expression was quantified before transport experiments.",
                    "evidence_refs": ["section:Methods:3"],
                },
                {
                    "statement": "Transport experiments used engineered cell lines to assess substrate handling.",
                    "evidence_refs": ["section:Methods:4"],
                },
                {
                    "statement": "Vector integration order was evaluated with genomic validation assays.",
                    "evidence_refs": ["section:Methods:5"],
                },
            ]
        },
        "presentation_evidence": {
            "methods": [
                {
                    "statement": "This fallback method row should not replace primary extracted methods evidence.",
                    "anchor": "section:Methods:6",
                    "evidence_refs": ["section:Methods:6"],
                }
            ]
        },
    }
    parsed_chunks = [
        {
            "anchor": "section:Methods:1",
            "content": "Cells were engineered with a Flp-In strategy using FRT sites.",
            "modality": "text",
        }
    ]

    report = synthesis.build_section_synthesis_v2(summary_json=summary, parsed_chunks=parsed_chunks, use_llm=False)
    methods = next(row for row in report["sections"] if row["section"] == "methods")

    assert report["style"] == "section_synthesis_v2_experimental"
    assert methods["source_counts"]["sections_extracted"] == 5
    assert methods["source_counts"]["heuristic_backfill"] == 0
    assert any(term["term"] == "Flp-In" for term in methods["key_terms"])


def test_detailed_methods_reconstructs_from_source_method_chunks() -> None:
    early_packets = [
        {
            "finding_id": "m1",
            "anchor": "section:Figure 2.:3",
            "statement": "The illustration shows the construct and frame shift used for double transfection.",
            "evidence_refs": ["section:Figure 2.:3"],
            "confidence": 0.72,
            "category": "methods",
            "section_label": "methods",
        }
    ]
    chunk_records = [
        {
            "anchor": "section:Transfection protocol optimization.:7",
            "content": "Cells were transfected with two expression plasmids and selected with hygromycin and puromycin.",
            "modality": "text",
            "meta": json.dumps({"section_raw_title": "Transfection protocol optimization.", "section_norm": "methods"}),
        },
        {
            "anchor": "section:Generation of second vector:32",
            "content": "The stable genomic integration of both plasmids was validated by PCR using genomic DNA isolated from cells.",
            "modality": "text",
            "meta": json.dumps({"section_raw_title": "Generation of second vector", "section_norm": "methods"}),
        },
        {
            "anchor": "section:Generation of second vector:36",
            "content": "Expression analyses used RNA isolation, cDNA synthesis, and real-time qPCR to quantify gene expression.",
            "modality": "text",
            "meta": json.dumps({"section_raw_title": "Generation of second vector", "section_norm": "methods"}),
        },
        {
            "anchor": "section:Generation of second vector:39",
            "content": "Functional validation was performed with transport experiments measuring proguanil transport and metabolism.",
            "modality": "text",
            "meta": json.dumps({"section_raw_title": "Generation of second vector", "section_norm": "methods"}),
        },
    ]

    sections, diagnostics, _notes = synthesis._build_detailed_sections(
        text_packets=early_packets,
        table_packets=[],
        figure_packets=[],
        supp_packets=[],
        methods_compact=[],
        analysis_notes=[],
        text_chunk_records=chunk_records,
        sections_extracted={"methods": []},
    )

    methods_text = " ".join(item["statement"] for item in sections["methods"]["items"])
    assert "selected with hygromycin and puromycin" in methods_text
    assert "validated by PCR" in methods_text
    assert "real-time qPCR" in methods_text
    assert diagnostics["methods"]["final_item_count"] >= 4


def test_section_synthesis_v2_backfills_sparse_extracted_sections() -> None:
    summary = {
        "sections_extracted": {
            "discussion": [
                {
                    "statement": "The findings suggest the approach can clarify gene interactions.",
                    "evidence_refs": ["section:Discussion:1"],
                }
            ]
        },
        "presentation_evidence": {
            "discussion": [
                {
                    "statement": "The authors note important limitations for multiplex PCR screening.",
                    "anchor": "section:Discussion:2",
                    "evidence_refs": ["section:Discussion:2"],
                },
                {
                    "statement": "Further validation is needed before broader pharmacogenomic use.",
                    "anchor": "section:Discussion:3",
                    "evidence_refs": ["section:Discussion:3"],
                },
                {
                    "statement": "These findings may help interpret transporter and enzyme interactions.",
                    "anchor": "section:Discussion:4",
                    "evidence_refs": ["section:Discussion:4"],
                },
            ]
        },
    }

    report = synthesis.build_section_synthesis_v2(summary_json=summary, parsed_chunks=[], use_llm=False)
    discussion = next(row for row in report["sections"] if row["section"] == "discussion")

    assert discussion["source_counts"]["sections_extracted"] == 1
    assert discussion["source_counts"]["heuristic_backfill"] >= 1


def test_section_extraction_rejects_invalid_or_cross_section_refs(monkeypatch) -> None:
    def _fake_chat(prompt: str, system: str | None = None, temperature: float = 0.2) -> str:
        if system == synthesis.SECTION_EXTRACTION_SYSTEM:
            return json.dumps(
                {
                    "introduction": [
                        {"statement": "Invalid intro line with fabricated ref.", "evidence_refs": ["section:Fake:99"]}
                    ],
                    "methods": [
                        {"statement": "Wrongly assigned methods line.", "evidence_refs": ["section:Results:1"]}
                    ],
                    "results": [
                        {"statement": "Valid results line.", "evidence_refs": ["section:Results:1"]}
                    ],
                    "discussion": [],
                    "conclusion": [],
                }
            )
        return "{}"

    monkeypatch.setattr(synthesis, "chat_text_deep", _fake_chat)
    monkeypatch.setattr(synthesis.settings, "analysis_section_extraction_enabled", True)
    monkeypatch.setattr(synthesis.settings, "analysis_section_extraction_subprocess_guard_enabled", False)
    summary = synthesize_report(
        {
            "evidence_packets": [
                {
                    "finding_id": "i1",
                    "modality": "text",
                    "anchor": "section:Introduction:1",
                    "statement": "Introduction objective statement.",
                    "evidence_refs": ["section:Introduction:1"],
                    "confidence": 0.9,
                    "quality_flags": [],
                    "category": "introduction",
                    "section_label": "introduction",
                    "section_confidence": 0.95,
                    "section_source": "anchor",
                },
                {
                    "finding_id": "r1",
                    "modality": "text",
                    "anchor": "section:Results:1",
                    "statement": "Results statement with concrete outcome p<0.01.",
                    "evidence_refs": ["section:Results:1"],
                    "confidence": 0.9,
                    "quality_flags": [],
                    "category": "results",
                    "section_label": "results",
                    "section_confidence": 0.95,
                    "section_source": "anchor",
                },
            ]
        },
        {"evidence_packets": []},
        {"evidence_packets": []},
        {"evidence_packets": []},
        {"cross_modal_claims": [], "discrepancies": []},
        paper_meta={},
        coverage={"figures": {}, "tables": {}, "supp_figures": {}, "supp_tables": {}},
    )
    extracted = summary.get("sections_extracted", {})
    assert extracted.get("introduction", []) == []
    assert extracted.get("methods", []) == []
    assert len(extracted.get("results", [])) == 1
