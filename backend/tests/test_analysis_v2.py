from __future__ import annotations

import json

import pytest

from app.services.analysis import synthesis
from app.services.analysis import figure_analysis
from app.services.analysis import reconcile
from app.services.analysis import runner
from app.services.analysis import prompts
from app.services.analysis import supp_analysis
from app.services.analysis import table_analysis
from app.services.analysis import text_analysis
from app.services.analysis.reconcile import reconcile_reports
from app.services.analysis.schemas import StructuredDossierV2
from app.services.analysis.synthesis import synthesize_report
from app.services.analysis.utils import filter_grounded_evidence_packets, normalize_evidence_packets
from app.db.models import Chunk
from app.services import validated_pipeline


@pytest.fixture(autouse=True)
def _legacy_llm_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Most tests in this module exercise the explicit LLM-analysis path."""
    for module in (figure_analysis, supp_analysis, table_analysis, synthesis):
        monkeypatch.setattr(module.settings, "llm_provider", "openai")
        monkeypatch.setattr(module.settings, "analysis_local_evidence_first_enabled", False)
    monkeypatch.setattr(figure_analysis.settings, "analysis_local_figure_caption_first_enabled", False)
    monkeypatch.setattr(supp_analysis.settings, "analysis_local_supplement_caption_first_enabled", False)


def test_normalize_evidence_packets_clamps_and_flags_missing() -> None:
    packets = normalize_evidence_packets(
        [
            {
                "finding_id": "x1",
                "anchor": "section:intro:1",
                "statement": "Treatment improved symptoms by 25%",
                "source_excerpt": "Methods text: participants received treatment before symptoms improved by 25%.",
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
    assert packets[0]["source_excerpt"].startswith("Methods text")
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


def test_normalize_evidence_packets_repairs_invalid_anchor_from_valid_ref() -> None:
    packets = normalize_evidence_packets(
        [
            {
                "finding_id": "x1",
                "anchor": "section:missing:99",
                "statement": "The methods section described the study design.",
                "evidence_refs": ["section:Methods:1"],
                "confidence": 0.8,
            }
        ],
        "text",
        {"section:Methods:1"},
    )

    assert packets[0]["anchor"] == "section:Methods:1"
    assert packets[0]["evidence_refs"] == ["section:Methods:1"]
    assert "missing_evidence" not in packets[0]["quality_flags"]


def test_normalize_evidence_packets_preserves_high_signal_section_sources() -> None:
    packets = normalize_evidence_packets(
        [
            {
                "finding_id": "x1",
                "anchor": "section:Methods:1",
                "statement": "The methods section described the study design.",
                "evidence_refs": ["section:Methods:1"],
                "confidence": 0.8,
                "section_source": "explicit_heading",
            },
            {
                "finding_id": "x2",
                "anchor": "section:Results:2",
                "statement": "The results section reported the primary outcome.",
                "evidence_refs": ["section:Results:2"],
                "confidence": 0.8,
                "section_source": "heading_style",
            },
            {
                "finding_id": "x3",
                "anchor": "section:Discussion:3",
                "statement": "The section ledger corrected a boundary issue.",
                "evidence_refs": ["section:Discussion:3"],
                "confidence": 0.8,
                "section_source": "section_boundary_ledger:forward_fill",
            },
        ],
        "text",
        {"section:Methods:1", "section:Results:2", "section:Discussion:3"},
    )

    assert [packet["section_source"] for packet in packets] == [
        "explicit_heading",
        "heading_style",
        "section_boundary_ledger",
    ]


def test_filter_grounded_evidence_packets_removes_invalid_anchor_packets() -> None:
    packets = normalize_evidence_packets(
        [
            {
                "finding_id": "valid",
                "anchor": "table:1",
                "statement": "Table 1 reports sample characteristics.",
                "evidence_refs": ["table:1"],
                "confidence": 0.8,
            },
            {
                "finding_id": "invalid",
                "anchor": "table:missing",
                "statement": "Invented table claim.",
                "evidence_refs": ["table:missing"],
                "confidence": 0.8,
            },
        ],
        "table",
        {"table:1"},
    )

    grounded = filter_grounded_evidence_packets(packets)

    assert [packet["finding_id"] for packet in grounded] == ["valid"]


def test_table_analysis_drops_ungrounded_llm_packets(monkeypatch) -> None:
    def _fake_chat(prompt: str, system: str | None = None, temperature: float = 0.2) -> str:
        return json.dumps(
            {
                "evidence_packets": [
                    {
                        "finding_id": "table-valid",
                        "anchor": "table:1",
                        "statement": "Table 1 reports the sample size.",
                        "evidence_refs": ["table:1"],
                        "confidence": 0.8,
                    },
                    {
                        "finding_id": "table-invalid",
                        "anchor": "table:missing",
                        "statement": "A missing table reports an unsupported result.",
                        "evidence_refs": ["table:missing"],
                        "confidence": 0.8,
                    },
                ]
            }
        )

    monkeypatch.setattr(table_analysis, "chat_text_fast", _fake_chat)

    report = table_analysis.analyze_tables(
        [
            {
                "anchor": "table:1",
                "content": json.dumps({"columns": ["n"], "data": [[42]]}),
            }
        ]
    )

    assert [packet["finding_id"] for packet in report["evidence_packets"]] == ["table-valid"]
    assert report["diagnostics"]["dropped_ungrounded_packets"] == 1
    assert report["diagnostics"]["prompt_chars"]
    assert report["diagnostics"]["prompt_blocks"] == [1]


def test_table_analysis_extractive_fallback_when_llm_returns_empty(monkeypatch) -> None:
    def _fake_chat(prompt: str, system: str | None = None, temperature: float = 0.2) -> str:
        return json.dumps({"evidence_packets": [], "findings": [], "results": []})

    monkeypatch.setattr(table_analysis, "chat_text_fast", _fake_chat)

    report = table_analysis.analyze_tables(
        [
            {
                "anchor": "table:1",
                "content": json.dumps(
                    {
                        "columns": ["arm", "dose", "outcome", "p_value"],
                        "data": [["fluoxetine", "20 mg", "symptom improvement", "0.03"]],
                    }
                ),
            }
        ]
    )

    packets = report["evidence_packets"]
    assert len(packets) == 2
    assert all(packet["anchor"] == "table:1" for packet in packets)
    assert any("fluoxetine" in packet["source_excerpt"] for packet in packets)
    assert any("Table row reports" in packet["statement"] for packet in packets)
    row_packet = next(packet for packet in packets if "Table row reports" in packet["statement"])
    assert "20 mg" in row_packet["statement"]
    assert "0.03" in row_packet["statement"]
    assert row_packet["section_label"] == "results"
    assert row_packet["category"] == "stats"
    assert "table_row_extraction" in row_packet["quality_flags"]
    assert all("extractive_fallback" in packet["quality_flags"] for packet in packets)
    assert report["diagnostics"]["extractive_fallback_packets"] == 2


def test_figure_analysis_extractive_fallback_when_caption_llm_returns_empty(monkeypatch) -> None:
    def _fake_chat(prompt: str, system: str | None = None, temperature: float = 0.2) -> str:
        return json.dumps({"evidence_packets": [], "findings": [], "results": []})

    monkeypatch.setattr(figure_analysis, "chat_text_fast", _fake_chat)

    report = figure_analysis.analyze_figures(
        [
            {
                "anchor": "figure:2",
                "meta": json.dumps(
                    {
                        "caption": (
                            "Figure 2. Fluoxetine 20 mg reduced immobility in the forced swim test "
                            "relative to vehicle; p = 0.03."
                        )
                    }
                ),
            }
        ]
    )

    packets = report["evidence_packets"]
    assert len(packets) == 1
    assert packets[0]["anchor"] == "figure:2"
    assert "Fluoxetine 20 mg" in packets[0]["source_excerpt"]
    assert "extractive_fallback" in packets[0]["quality_flags"]
    assert report["diagnostics"]["extractive_fallback_packets"] == 1


def test_table_analysis_prompt_preserves_complete_anchored_blocks_under_budget() -> None:
    blocks = [
        "[TABLE table:1]\n" + "\n".join(f"row {idx}\tfluoxetine\t20 mg\tp < 0.05" for idx in range(30)),
        "[TABLE table:2]\n" + "\n".join(f"row {idx}\tplacebo\t0 mg\tp = 0.80" for idx in range(30)),
        "[TABLE table:3]\n" + "\n".join(f"row {idx}\tadverse events\tcount {idx}" for idx in range(30)),
    ]

    prompt = table_analysis._table_analysis_prompt(blocks, max_chars=900)  # noqa: SLF001

    assert len(prompt) <= 900
    assert "[TABLE table:1]" in prompt
    assert prompt.count("[TABLE table:") <= 2
    assert prompt.endswith("...") or "[TABLE table:2]" not in prompt


def test_supplement_analysis_prompt_preserves_anchor_blocks_under_budget() -> None:
    blocks = [
        "[SUPP supp:1]\n" + "fluoxetine 20 mg daily adverse events p = 0.04 " * 60,
        "[SUPP TABLE supp:table:1]\n" + "\n".join(f"row {idx}\tbiomarker\t{idx}" for idx in range(40)),
        "[SUPP supp:2]\n" + "model system validation assay readout " * 60,
    ]

    prompt = supp_analysis._supplement_analysis_prompt(blocks, max_chars=900)  # noqa: SLF001

    assert len(prompt) <= 900
    assert "[SUPP supp:1]" in prompt
    assert prompt.count("[SUPP") <= 2
    assert prompt.endswith("...") or "[SUPP TABLE supp:table:1]" not in prompt


def test_supplement_text_analysis_adds_source_excerpts(monkeypatch) -> None:
    def _fake_chat(prompt: str, system: str | None = None, temperature: float = 0.2) -> str:
        assert "[SUPP supp:1]\n" in prompt
        return json.dumps(
            {
                "evidence_packets": [
                    {
                        "finding_id": "supp-valid",
                        "anchor": "supp:1",
                        "statement": "Supplement reports fluoxetine 20 mg daily.",
                        "evidence_refs": ["supp:1"],
                        "confidence": 0.82,
                    }
                ]
            }
        )

    monkeypatch.setattr(supp_analysis, "chat_text_fast", _fake_chat)

    report = supp_analysis.analyze_supplements(
        [
            {
                "anchor": "supp:1",
                "modality": "text",
                "content": "Supplement methods define fluoxetine 20 mg daily before endpoint assessment.",
            }
        ]
    )

    assert report["evidence_packets"][0]["finding_id"] == "supp-valid"
    assert "fluoxetine 20 mg daily" in report["evidence_packets"][0]["source_excerpt"]


def test_text_analysis_prompt_preserves_anchors_under_budget() -> None:
    blocks = [
        "[section:Methods:1]\n" + "fluoxetine 20 mg daily randomized comparator outcome week 8 " * 60,
        "[section:Results:2]\n" + "response was higher with p < 0.05 adverse events were similar " * 60,
    ]

    prompt = text_analysis._text_analysis_prompt(blocks, max_chars=900)  # noqa: SLF001

    assert len(prompt) <= 900
    assert "[section:Methods:1]" in prompt
    assert prompt.count("[section:") <= 2
    assert prompt.endswith("...") or "[section:Results:2]" not in prompt


def test_text_analysis_uses_bounded_anchor_prompt_and_source_excerpts(monkeypatch) -> None:
    captured: list[str] = []

    def _fake_chat(prompt: str, system: str | None = None, temperature: float = 0.2) -> str:
        captured.append(prompt)
        assert "[section:Methods:1]\n" in prompt
        return json.dumps(
            {
                "evidence_packets": [
                    {
                        "finding_id": "text-valid",
                        "anchor": "section:Methods:1",
                        "statement": "Participants received fluoxetine 20 mg daily for 8 weeks.",
                        "evidence_refs": ["section:Methods:1"],
                        "confidence": 0.86,
                        "category": "methods",
                    }
                ]
            }
        )

    monkeypatch.setattr(text_analysis, "chat_text_fast", _fake_chat)
    timing = iter([10.0, 12.5])
    monkeypatch.setattr(text_analysis, "monotonic", lambda: next(timing))
    monkeypatch.setattr(text_analysis.settings, "analysis_max_text_chars", 900)
    monkeypatch.setattr(text_analysis.settings, "llm_n_ctx", 1000)

    report = text_analysis.analyze_text(
        [
            {
                "anchor": "section:Methods:1",
                "content": "Participants received fluoxetine 20 mg daily for 8 weeks before outcome assessment. " * 40,
            }
        ],
        force_llm_enabled=True,
    )

    assert captured
    assert all(len(prompt) <= 900 for prompt in captured)
    assert report["evidence_packets"][0]["finding_id"] == "text-valid"
    assert "fluoxetine 20 mg daily" in report["evidence_packets"][0]["source_excerpt"]
    assert report["diagnostics"]["llm_batches"] == 1
    assert report["diagnostics"]["llm_batch_seconds"] == [2.5]
    assert report["diagnostics"]["llm_batch_details"] == [
        {
            "batch_index": 1,
            "prompt_chars": len(captured[0]),
            "prompt_blocks": 1,
            "duration_seconds": 2.5,
            "first_anchor": "section:Methods:1",
            "last_anchor": "section:Methods:1",
        }
    ]


def test_analysis_prompts_preserve_medical_and_scientific_detail_contract() -> None:
    joined = "\n".join(
        [
            prompts.TEXT_ANALYSIS_SYSTEM,
            prompts.TABLE_ANALYSIS_SYSTEM,
            prompts.FIGURE_ANALYSIS_SYSTEM,
            prompts.SUPP_ANALYSIS_SYSTEM,
        ]
    ).lower()

    for required in (
        "dose",
        "route",
        "duration",
        "comparator",
        "adverse events",
        "model systems",
        "assays",
        "effect directions",
    ):
        assert required in joined


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


def test_reconcile_prompt_compacts_to_valid_json() -> None:
    payload = {
        "claim_reviews": [
            {
                "claim_id": f"claim-{idx}",
                "claim": "The paper reports a cross-modal result needing careful review. " * 12,
                "evidence": [f"section:Results:{idx}", f"table:{idx}"],
                "related_packets": [
                    {
                        "finding_id": f"packet-{idx}-{packet_idx}",
                        "statement": "Related table or figure evidence with detailed context. " * 10,
                        "evidence_refs": [f"figure:{packet_idx}", f"table:{packet_idx}"],
                        "modality": "figure",
                        "confidence": 0.8,
                    }
                    for packet_idx in range(6)
                ],
            }
            for idx in range(16)
        ]
    }

    prompt = reconcile._reconcile_prompt_text(payload, max_chars=2400)  # noqa: SLF001
    parsed_payload = json.loads(prompt.split("\n\n", 1)[1])

    assert len(prompt) <= 2400
    assert "claim_reviews" in parsed_payload
    assert parsed_payload["claim_reviews"]


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


def test_scientific_details_inventory_is_broad_and_evidence_grounded() -> None:
    details = synthesis._scientific_details(  # noqa: SLF001 - deterministic synthesis helper coverage
        [
            {
                "finding_id": "med-1",
                "modality": "text",
                "anchor": "section:Methods:1",
                "statement": "Participants received oral fluoxetine 20 mg daily for 8 weeks compared with placebo.",
                "evidence_refs": ["section:Methods:1"],
                "confidence": 0.9,
                "category": "medication",
                "section_label": "methods",
            },
            {
                "finding_id": "assay-1",
                "modality": "figure",
                "anchor": "figure:fig2",
                "statement": "Figure 2 shows qPCR validation of gene expression in engineered cell lines.",
                "source_excerpt": "Figure 2. qPCR validation confirmed gene expression changes in engineered cell lines.",
                "evidence_refs": ["figure:fig2"],
                "confidence": 0.85,
                "category": "assay",
                "section_label": "results",
            },
            {
                "finding_id": "review-1",
                "modality": "text",
                "anchor": "section:Methods:2",
                "statement": "The review used eligibility criteria and risk of bias assessment for included studies.",
                "evidence_refs": ["section:Methods:2"],
                "confidence": 0.8,
                "category": "methods",
                "section_label": "methods",
            },
            {
                "finding_id": "bad-1",
                "modality": "text",
                "anchor": "section:Methods:3",
                "statement": "Unsupported invented detail.",
                "evidence_refs": [],
                "confidence": 0.9,
                "quality_flags": ["missing_evidence"],
                "category": "methods",
            },
        ],
        paper_type="laboratory/preclinical study",
    )

    detail_types = {detail_type for detail in details for detail_type in detail["detail_types"]}
    assert "medication_or_therapeutic" in detail_types
    assert "dose_schedule" in detail_types
    assert "assay_readout" in detail_types
    assert "model_system" in detail_types
    assert "data_source_or_design" in detail_types
    assert all(detail["evidence_refs"] for detail in details)
    figure_detail = next(detail for detail in details if detail["source_modality"] == "figure")
    assert "qPCR validation confirmed" in figure_detail["source_excerpt"]


def test_scientific_details_inventory_captures_section_level_reasoning_roles() -> None:
    details = synthesis._scientific_details(  # noqa: SLF001 - deterministic synthesis helper coverage
        [
            {
                "finding_id": "intro-objective",
                "modality": "text",
                "anchor": "section:Introduction:1",
                "statement": "The objective was to test whether reward responsivity explains transdiagnostic symptoms.",
                "evidence_refs": ["section:Introduction:1"],
                "confidence": 0.82,
                "category": "objective",
                "section_label": "introduction",
            },
            {
                "finding_id": "discussion-interpretation",
                "modality": "text",
                "anchor": "section:Discussion:1",
                "statement": "The authors interpreted the network findings as clinically relevant but cautioned about cross-sectional design limitations.",
                "evidence_refs": ["section:Discussion:1"],
                "confidence": 0.78,
                "category": "interpretation",
                "section_label": "discussion",
            },
            {
                "finding_id": "conclusion-takeaway",
                "modality": "text",
                "anchor": "section:Conclusion:1",
                "statement": "Overall, these findings highlight a transdiagnostic reward-network signature.",
                "evidence_refs": ["section:Conclusion:1"],
                "confidence": 0.76,
                "category": "conclusion",
                "section_label": "conclusion",
            },
        ],
        paper_type="observational study",
    )

    detail_types = {detail_type for detail in details for detail_type in detail["detail_types"]}

    assert "rationale_or_objective" in detail_types
    assert "interpretation_or_implication" in detail_types
    assert "limitation_or_caution" in detail_types
    assert "conclusion_or_takeaway" in detail_types
    assert {detail["section_label"] for detail in details} == {"introduction", "discussion", "conclusion"}


def test_scientific_details_prompt_selection_preserves_roles_and_modalities() -> None:
    details = [
        {
            "detail_types": ["statistical_result"],
            "statement": f"Text result {idx} reported a statistically significant symptom change with p < 0.01.",
            "source_excerpt": f"Results excerpt {idx}.",
            "evidence_refs": [f"section:Results:{idx}"],
            "source_modality": "text",
            "section_label": "results",
            "category": "stats",
            "confidence": 0.99,
        }
        for idx in range(8)
    ]
    details += [
        {
            "detail_types": ["medication_or_therapeutic", "dose_schedule"],
            "statement": "Participants received oral fluoxetine 20 mg daily for 8 weeks.",
            "source_excerpt": "Methods excerpt with fluoxetine 20 mg daily for 8 weeks.",
            "evidence_refs": ["section:Methods:1"],
            "source_modality": "text",
            "section_label": "methods",
            "category": "medication",
            "confidence": 0.52,
        },
        {
            "detail_types": ["cross_modal_result", "statistical_result"],
            "statement": "Table 2 reported adverse events by treatment arm.",
            "source_excerpt": "Table 2 adverse event counts by arm.",
            "evidence_refs": ["table:2"],
            "source_modality": "table",
            "section_label": "results",
            "category": "stats",
            "confidence": 0.48,
        },
        {
            "detail_types": ["assay_readout", "cross_modal_result"],
            "statement": "Figure 3 showed biomarker assay validation in engineered cells.",
            "source_excerpt": "Figure 3 legend described biomarker assay validation.",
            "evidence_refs": ["figure:3"],
            "source_modality": "figure",
            "section_label": "results",
            "category": "assay",
            "confidence": 0.46,
        },
        {
            "detail_types": ["statistical_result"],
            "statement": "Supplementary Table S1 reported subgroup sensitivity estimates.",
            "source_excerpt": "Supplementary Table S1 subgroup sensitivity estimates.",
            "evidence_refs": ["supp:S1"],
            "source_modality": "supplement",
            "section_label": "results",
            "category": "stats",
            "confidence": 0.44,
        },
    ]

    selected = synthesis._scientific_details_for_prompt(details, max_items=5)  # noqa: SLF001
    selected_text = json.dumps(selected)
    selected_modalities = {item["source_modality"] for item in selected}

    assert "fluoxetine 20 mg daily" in selected_text
    assert {"table", "figure", "supplement"}.issubset(selected_modalities)
    assert "Supplementary Table S1" in selected_text


def test_scientific_details_prompt_selection_preserves_major_section_roles() -> None:
    details = [
        {
            "detail_types": ["statistical_result"],
            "statement": f"Results row {idx} reported a statistically significant association with p < 0.01.",
            "source_excerpt": f"Results excerpt {idx}.",
            "evidence_refs": [f"section:Results:{idx}"],
            "source_modality": "text",
            "section_label": "results",
            "category": "stats",
            "confidence": 0.99,
        }
        for idx in range(10)
    ]
    details += [
        {
            "detail_types": ["rationale_or_objective"],
            "statement": "The objective was to evaluate a transdiagnostic reward-network hypothesis.",
            "source_excerpt": "Introduction objective excerpt.",
            "evidence_refs": ["section:Introduction:1"],
            "source_modality": "text",
            "section_label": "introduction",
            "category": "objective",
            "confidence": 0.45,
        },
        {
            "detail_types": ["data_source_or_design"],
            "statement": "The cohort included adults across diagnostic groups.",
            "source_excerpt": "Methods cohort excerpt.",
            "evidence_refs": ["section:Methods:1"],
            "source_modality": "text",
            "section_label": "methods",
            "category": "methods",
            "confidence": 0.44,
        },
        {
            "detail_types": ["interpretation_or_implication", "limitation_or_caution"],
            "statement": "The discussion interpreted the association cautiously because cross-sectional design limits causal inference.",
            "source_excerpt": "Discussion limitation excerpt.",
            "evidence_refs": ["section:Discussion:1"],
            "source_modality": "text",
            "section_label": "discussion",
            "category": "limitations",
            "confidence": 0.43,
        },
        {
            "detail_types": ["conclusion_or_takeaway"],
            "statement": "Overall, the paper concluded that reward-network signatures may be transdiagnostic.",
            "source_excerpt": "Conclusion takeaway excerpt.",
            "evidence_refs": ["section:Conclusion:1"],
            "source_modality": "text",
            "section_label": "conclusion",
            "category": "conclusion",
            "confidence": 0.42,
        },
    ]

    selected = synthesis._scientific_details_for_prompt(details, max_items=5)  # noqa: SLF001
    selected_sections = {item["section_label"] for item in selected}
    selected_text = json.dumps(selected)

    assert {"introduction", "methods", "results", "discussion", "conclusion"}.issubset(selected_sections)
    assert "transdiagnostic reward-network hypothesis" in selected_text
    assert "cross-sectional design limits causal inference" in selected_text
    assert "reward-network signatures may be transdiagnostic" in selected_text


def test_synthesis_evidence_plan_marks_focus_slots_for_diverse_paper_details() -> None:
    scientific_details = [
        {
            "detail_types": ["medication_or_therapeutic", "dose_schedule"],
            "statement": "Participants received oral fluoxetine 20 mg daily for 8 weeks.",
            "source_excerpt": "Methods excerpt with fluoxetine 20 mg daily for 8 weeks.",
            "evidence_refs": ["section:Methods:1"],
            "source_modality": "text",
            "section_label": "methods",
            "category": "medication",
            "confidence": 0.82,
        },
        {
            "detail_types": ["outcome_measure"],
            "statement": "The primary endpoint was change in MADRS score at week 8.",
            "source_excerpt": "Methods excerpt naming MADRS at week 8.",
            "evidence_refs": ["section:Methods:2"],
            "source_modality": "text",
            "section_label": "methods",
            "category": "clinical",
            "confidence": 0.78,
        },
        {
            "detail_types": ["cross_modal_result", "statistical_result"],
            "statement": "Table 2 reported higher response in the active arm with p < 0.05.",
            "source_excerpt": "Table 2 response counts and p < 0.05.",
            "evidence_refs": ["table:2"],
            "source_modality": "table",
            "section_label": "results",
            "category": "stats",
            "confidence": 0.75,
        },
        {
            "detail_types": ["adverse_event"],
            "statement": "Adverse events were tabulated by treatment arm.",
            "source_excerpt": "Safety table listed adverse events by arm.",
            "evidence_refs": ["table:3"],
            "source_modality": "table",
            "section_label": "results",
            "category": "clinical",
            "confidence": 0.7,
        },
    ]

    plan = synthesis._synthesis_evidence_plan(  # noqa: SLF001 - prompt contract coverage
        paper_type="randomized trial",
        scientific_details=scientific_details,
        section_inputs={
            "introduction": [
                {
                    "statement": "The objective was to test a treatment for persistent depressive symptoms.",
                    "evidence_refs": ["section:Introduction:1"],
                    "source_modality": "text",
                    "confidence": 0.7,
                }
            ],
            "discussion": [
                {
                    "statement": "The authors noted limited generalizability and a need for replication.",
                    "evidence_refs": ["section:Discussion:1"],
                    "source_modality": "text",
                    "confidence": 0.7,
                }
            ],
        },
    )
    slots = {slot["slot_key"]: slot for slot in plan["focus_slots"]}

    assert slots["objective_or_rationale"]["status"] == "found"
    assert slots["objective_or_rationale"]["source"] == "section_input"
    assert slots["intervention_exposure_protocol_or_assay"]["status"] == "found"
    assert "fluoxetine 20 mg daily" in slots["intervention_exposure_protocol_or_assay"]["statement"]
    assert slots["outcome_readout_or_endpoint"]["status"] == "found"
    assert "MADRS" in slots["outcome_readout_or_endpoint"]["statement"]
    assert slots["primary_results_statistics_or_effects"]["source_modality"] == "table"
    assert slots["figure_table_or_supplement_evidence"]["status"] == "found"
    assert slots["limitations_interpretation_or_implications"]["status"] == "found"
    assert slots["safety_or_adverse_events"]["status"] == "found"
    assert plan["critical_missing_focus_slots"] == []
    assert "critical_focus_slots_missing" not in plan["quality_flags"]


def test_synthesis_evidence_plan_keeps_split_cross_modal_support_candidates() -> None:
    plan = synthesis._synthesis_evidence_plan(  # noqa: SLF001 - prompt contract coverage
        paper_type="observational cohort",
        scientific_details=[
            {
                "detail_types": ["cross_modal_result", "statistical_result"],
                "statement": "Table 2 reported adjusted odds ratio OR = 1.26 for symptom persistence.",
                "source_excerpt": "Table 2 adjusted model: OR = 1.26.",
                "evidence_refs": ["table:2"],
                "source_modality": "table",
                "section_label": "results",
                "category": "stats",
                "confidence": 0.82,
            },
            {
                "detail_types": ["cross_modal_result", "statistical_result"],
                "statement": "Figure 3 showed the same symptom-persistence direction across sensitivity analyses.",
                "source_excerpt": "Figure 3 sensitivity analyses showed the same direction.",
                "evidence_refs": ["figure:3"],
                "source_modality": "figure",
                "section_label": "results",
                "category": "stats",
                "confidence": 0.78,
            },
            {
                "detail_types": ["outcome_measure"],
                "statement": "The outcome was symptom persistence at follow-up.",
                "source_excerpt": "Methods defined symptom persistence at follow-up.",
                "evidence_refs": ["section:Methods:4"],
                "source_modality": "text",
                "section_label": "methods",
                "category": "clinical",
                "confidence": 0.75,
            },
        ],
        section_inputs={
            "introduction": [
                {
                    "statement": "The objective was to evaluate predictors of symptom persistence.",
                    "evidence_refs": ["section:Introduction:1"],
                    "source_modality": "text",
                    "confidence": 0.7,
                }
            ],
            "methods": [
                {
                    "statement": "The cohort design used adjusted regression models.",
                    "evidence_refs": ["section:Methods:1"],
                    "source_modality": "text",
                    "confidence": 0.7,
                }
            ],
            "discussion": [
                {
                    "statement": "The authors cautioned that residual confounding may limit causal interpretation.",
                    "evidence_refs": ["section:Discussion:1"],
                    "source_modality": "text",
                    "confidence": 0.7,
                }
            ],
        },
    )
    slots = {slot["slot_key"]: slot for slot in plan["focus_slots"]}
    primary_results = slots["primary_results_statistics_or_effects"]

    assert primary_results["status"] == "found"
    assert primary_results["source_modality"] == "table"
    assert "supporting_candidates" in primary_results
    assert {
        candidate["source_modality"]
        for candidate in primary_results["supporting_candidates"]
    } >= {"table", "figure"}

    compact = synthesis._compact_evidence_plan_for_prompt(  # noqa: SLF001
        plan,
        focus_slot_limit=8,
        focus_statement_chars=160,
    )
    compact_slots = {slot["slot_key"]: slot for slot in compact["focus_slots"]}
    compact_support = compact_slots["primary_results_statistics_or_effects"]["supporting_candidates"]

    assert len(compact_support) == 2
    assert compact_support[0]["evidence_refs"] == ["table:2"]
    assert compact_support[1]["evidence_refs"] == ["figure:3"]
    assert "OR = 1.26" in compact_support[0]["statement"]
    assert "sensitivity analyses" in compact_support[1]["statement"]


def test_synthesis_evidence_plan_exposes_missing_focus_slots_without_fabricating() -> None:
    plan = synthesis._synthesis_evidence_plan(  # noqa: SLF001 - prompt contract coverage
        paper_type="systematic review",
        scientific_details=[],
        section_inputs={"methods": []},
    )
    slots = {slot["slot_key"]: slot for slot in plan["focus_slots"]}

    assert slots["review_search_synthesis_quality"]["status"] == "missing"
    assert "No selected scientific detail" in slots["review_search_synthesis_quality"]["reason"]
    critical_missing = {slot["slot_key"] for slot in plan["critical_missing_focus_slots"]}
    assert "review_search_synthesis_quality" in critical_missing
    assert "critical_focus_slots_missing" in plan["quality_flags"]
    assert "review_search_or_bias_details_missing" in plan["quality_flags"]


def test_synthesis_evidence_plan_flags_missing_trial_safety_without_blocking_synthesis() -> None:
    plan = synthesis._synthesis_evidence_plan(  # noqa: SLF001 - prompt contract coverage
        paper_type="randomized trial",
        scientific_details=[
            {
                "detail_types": ["medication_or_therapeutic", "dose_schedule"],
                "statement": "Participants received oral fluoxetine 20 mg daily for 8 weeks.",
                "source_excerpt": "Methods excerpt with fluoxetine 20 mg daily for 8 weeks.",
                "evidence_refs": ["section:Methods:1"],
                "source_modality": "text",
                "section_label": "methods",
                "category": "medication",
                "confidence": 0.82,
            },
            {
                "detail_types": ["outcome_measure"],
                "statement": "The primary endpoint was change in MADRS score at week 8.",
                "source_excerpt": "Methods excerpt naming MADRS at week 8.",
                "evidence_refs": ["section:Methods:2"],
                "source_modality": "text",
                "section_label": "methods",
                "category": "clinical",
                "confidence": 0.78,
            },
            {
                "detail_types": ["statistical_result"],
                "statement": "The active arm had a larger symptom reduction than placebo with p < 0.05.",
                "source_excerpt": "Results excerpt reporting p < 0.05.",
                "evidence_refs": ["section:Results:1"],
                "source_modality": "text",
                "section_label": "results",
                "category": "stats",
                "confidence": 0.76,
            },
        ],
        section_inputs={
            "introduction": [
                {
                    "statement": "The objective was to test fluoxetine for persistent depressive symptoms.",
                    "evidence_refs": ["section:Introduction:1"],
                    "source_modality": "text",
                    "confidence": 0.7,
                }
            ],
            "discussion": [
                {
                    "statement": "The authors noted limited generalizability and a need for replication.",
                    "evidence_refs": ["section:Discussion:1"],
                    "source_modality": "text",
                    "confidence": 0.7,
                }
            ],
        },
    )

    critical_missing = {slot["slot_key"] for slot in plan["critical_missing_focus_slots"]}

    assert "safety_or_adverse_events" in critical_missing
    assert "safety_or_adverse_events_missing" in plan["quality_flags"]
    assert plan["missing_focus_slot_count"] >= len(plan["critical_missing_focus_slots"])


def test_synthesize_report_exposes_scientific_details_without_extra_llm_routing(monkeypatch) -> None:
    monkeypatch.setattr(synthesis.settings, "analysis_section_extraction_enabled", False)
    text_report = {
        "evidence_packets": [
            {
                "finding_id": "med-1",
                "modality": "text",
                "anchor": "section:Methods:1",
                "statement": "Participants received oral fluoxetine 20 mg daily for 8 weeks compared with placebo.",
                "evidence_refs": ["section:Methods:1"],
                "confidence": 0.9,
                "quality_flags": [],
                "category": "medication",
                "section_label": "methods",
            }
        ]
    }
    table_report = {
        "evidence_packets": [
            {
                "finding_id": "table-1",
                "modality": "table",
                "anchor": "table:1",
                "statement": "Table 1 reports a higher response rate in the fluoxetine arm with p < 0.05.",
                "evidence_refs": ["table:1"],
                "confidence": 0.8,
                "quality_flags": [],
                "category": "stats",
                "section_label": "results",
            }
        ]
    }
    summary = synthesize_report(
        text_report,
        table_report,
        {"evidence_packets": []},
        {"evidence_packets": []},
        {"cross_modal_claims": [], "discrepancies": []},
        paper_meta={"title": "Medication Trial"},
        coverage={"figures": {}, "tables": {}, "supp_figures": {}, "supp_tables": {}},
    )

    assert summary["scientific_details_version"] == 1
    assert len(summary["scientific_details"]) >= 2
    detail_types = {detail_type for detail in summary["scientific_details"] for detail_type in detail["detail_types"]}
    assert "medication_or_therapeutic" in detail_types
    assert "statistical_result" in detail_types
    assert summary["evidence_packets_version"] == 1
    assert len(summary["evidence_packets"]) == 2
    assert summary["evidence_packet_coverage_version"] == 1
    assert summary["evidence_packet_coverage"]["by_section"]["methods"] == 1
    assert summary["evidence_packet_coverage"]["by_section"]["results"] == 1
    assert summary["evidence_packet_coverage"]["by_modality"]["table"] == 1
    assert summary["evidence_packet_coverage"]["cross_modal_packet_count"] == 1
    assert "medication_or_therapeutic" in summary["evidence_packet_coverage"]["by_detail_type"]
    audit_by_id = {packet["finding_id"]: packet for packet in summary["evidence_packets"]}
    assert audit_by_id["med-1"]["section_label"] == "methods"
    assert "medication_or_therapeutic" in audit_by_id["med-1"]["detail_types"]
    assert audit_by_id["table-1"]["modality"] == "table"
    assert audit_by_id["table-1"]["usable_for_gold_comparison"] is True
    assert summary["section_diagnostics"]["evidence_packet_count"] == 2
    assert summary["section_diagnostics"]["evidence_packet_usable_count"] == 2
    assert summary["section_diagnostics"]["evidence_packet_coverage"]["usable_packets"] == 2
    assert summary["section_diagnostics"]["scientific_details_count"] == len(summary["scientific_details"])
    assert any(
        "Critical synthesis evidence not found" in item
        for item in summary["section_diagnostics"]["synthesis_evidence_warnings"]
    )
    assert any(
        "Critical synthesis evidence not found" in item
        for item in summary["uncertainty_gaps"]
    )
    validated = StructuredDossierV2.model_validate(summary).model_dump()
    assert validated["scientific_details"][0]["evidence_refs"]
    assert validated["evidence_packets"][0]["evidence_refs"]
    assert validated["evidence_packet_coverage"]["usable_packets"] == 2


def test_evidence_packet_coverage_excludes_unknown_sections_from_gold_usable(monkeypatch) -> None:
    monkeypatch.setattr(synthesis.settings, "analysis_section_extraction_enabled", False)
    summary = synthesize_report(
        {
            "evidence_packets": [
                {
                    "finding_id": "unknown-1",
                    "modality": "text",
                    "anchor": "section:Unknown:1",
                    "statement": "This packet is grounded but not assigned to a real paper section.",
                    "evidence_refs": ["section:Unknown:1"],
                    "confidence": 0.8,
                    "quality_flags": [],
                    "category": "other",
                    "section_label": "unknown",
                }
            ]
        },
        {"evidence_packets": []},
        {"evidence_packets": []},
        {"evidence_packets": []},
        {"cross_modal_claims": [], "discrepancies": []},
        paper_meta={"title": "Unknown Section"},
        coverage={"figures": {}, "tables": {}, "supp_figures": {}, "supp_tables": {}},
    )

    assert summary["evidence_packets"][0]["usable_for_gold_comparison"] is False
    assert summary["evidence_packet_coverage"]["usable_packets"] == 0
    assert "no_usable_evidence_packets" in summary["evidence_packet_coverage"]["quality_flags"]


def test_synthesis_evidence_plan_uses_section_extraction_rows_in_diagnostics(monkeypatch) -> None:
    monkeypatch.setattr(synthesis.settings, "analysis_local_evidence_first_enabled", False)
    monkeypatch.setattr(synthesis.settings, "analysis_section_extraction_enabled", True)
    monkeypatch.setattr(synthesis.settings, "analysis_narrative_overrides_enabled", False)
    monkeypatch.setattr(
        synthesis,
        "_llm_section_extraction",
        lambda _payload: {
            "introduction": [
                {
                    "statement": "The objective was to test whether fluoxetine improved depressive symptoms.",
                    "evidence_refs": ["section:Introduction:1"],
                    "confidence": 0.86,
                }
            ],
            "discussion": [
                {
                    "statement": "The authors noted limited generalizability and a need for replication.",
                    "evidence_refs": ["section:Discussion:1"],
                    "confidence": 0.78,
                }
            ],
        },
    )
    text_report = {
        "evidence_packets": [
            {
                "finding_id": "med-1",
                "modality": "text",
                "anchor": "section:Methods:1",
                "statement": "Participants received oral fluoxetine 20 mg daily for 8 weeks compared with placebo.",
                "evidence_refs": ["section:Methods:1"],
                "confidence": 0.9,
                "quality_flags": [],
                "category": "medication",
                "section_label": "methods",
            }
        ]
    }

    summary = synthesize_report(
        text_report,
        {"evidence_packets": []},
        {"evidence_packets": []},
        {"evidence_packets": []},
        {"cross_modal_claims": [], "discrepancies": []},
        paper_meta={"title": "Medication Trial"},
        coverage={"figures": {}, "tables": {}, "supp_figures": {}, "supp_tables": {}},
    )
    plan = summary["section_diagnostics"]["synthesis_evidence_plan"]
    slots = {slot["slot_key"]: slot for slot in plan["focus_slots"]}

    assert plan["section_input_counts"]["introduction"] == 1
    assert slots["objective_or_rationale"]["status"] == "found"
    assert slots["objective_or_rationale"]["source"] == "section_input"
    assert slots["limitations_interpretation_or_implications"]["status"] == "found"


def test_local_evidence_first_skips_llm_section_extraction_but_keeps_section_compact(monkeypatch) -> None:
    monkeypatch.setattr(synthesis.settings, "llm_provider", "local")
    monkeypatch.setattr(synthesis.settings, "analysis_local_evidence_first_enabled", True)
    monkeypatch.setattr(synthesis.settings, "analysis_section_extraction_enabled", True)
    monkeypatch.setattr(synthesis.settings, "analysis_narrative_overrides_enabled", False)

    def _raise_section_extraction(_payload: dict[str, object]) -> dict[str, list[dict[str, object]]]:
        raise AssertionError("local evidence-first should not call deep section extraction")

    monkeypatch.setattr(synthesis, "_llm_section_extraction", _raise_section_extraction)
    summary = synthesize_report(
        {
            "evidence_packets": [
                {
                    "finding_id": "m1",
                    "modality": "text",
                    "anchor": "section:Methods:1",
                    "statement": "Participants received fluoxetine 20 mg daily for 8 weeks.",
                    "evidence_refs": ["section:Methods:1"],
                    "confidence": 0.9,
                    "quality_flags": [],
                    "category": "medication",
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
        paper_meta={"title": "Medication Trial"},
        coverage={"figures": {}, "tables": {}, "supp_figures": {}, "supp_tables": {}},
    )

    diagnostics = summary["section_diagnostics"]
    assert diagnostics["section_extraction_enabled"] is True
    assert diagnostics["section_extraction_llm_enabled"] is False
    assert diagnostics["section_extraction_skip_reason"] == "local_evidence_first"
    assert summary["sections_compact"]["methods"]
    assert "fluoxetine" in summary["executive_summary"].lower()


def test_local_evidence_first_can_run_deep_narrative_synthesis(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def _fake_run(prompt: str, system_prompt: str, **_: object) -> dict[str, object]:
        calls.append({"prompt": prompt, "system_prompt": system_prompt})
        return {
            "executive_summary": (
                "Introduction: The report evaluates a medication intervention.\n"
                "Methods: Participants received fluoxetine 20 mg daily for 8 weeks.\n"
                "Results: The evidence packet did not provide a result estimate.\n"
                "Discussion: Interpretation should remain cautious.\n"
                "Conclusion: The synthesized report is grounded in the medication evidence."
            ),
            "methods_strengths": ["The synthesis preserved the medication dose and duration."],
            "methods_weaknesses": [],
            "reproducibility_ethics": [],
            "uncertainty_gaps": ["The result estimate was not available in the provided evidence."],
        }

    def _raise_section_extraction(_payload: dict[str, object]) -> dict[str, list[dict[str, object]]]:
        raise AssertionError("local evidence-first should not call deep section extraction")

    monkeypatch.setattr(synthesis, "_run_deep_json_prompt", _fake_run)
    monkeypatch.setattr(synthesis, "_llm_section_extraction", _raise_section_extraction)
    monkeypatch.setattr(synthesis.settings, "llm_provider", "local")
    monkeypatch.setattr(synthesis.settings, "analysis_local_evidence_first_enabled", True)
    monkeypatch.setattr(synthesis.settings, "analysis_narrative_overrides_enabled", True)
    monkeypatch.setattr(synthesis.settings, "analysis_section_extraction_enabled", True)

    summary = synthesize_report(
        {
            "evidence_packets": [
                {
                    "finding_id": "m1",
                    "modality": "text",
                    "anchor": "section:Methods:1",
                    "statement": "Participants received fluoxetine 20 mg daily for 8 weeks.",
                    "evidence_refs": ["section:Methods:1"],
                    "confidence": 0.9,
                    "quality_flags": [],
                    "category": "medication",
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
        paper_meta={"title": "Medication Trial"},
        coverage={"figures": {}, "tables": {}, "supp_figures": {}, "supp_tables": {}},
    )

    assert calls
    system_prompts = {call["system_prompt"] for call in calls}
    assert synthesis.EXECUTIVE_REPORT_SYNTHESIS_SYSTEM in system_prompts
    assert summary["section_diagnostics"]["section_extraction_llm_enabled"] is False
    assert summary["methods_strengths"] == ["The synthesis preserved the medication dose and duration."]
    assert "fluoxetine 20 mg daily" in summary["executive_summary"]


def test_synthesize_report_records_narrative_synthesis_usage(monkeypatch) -> None:
    def _fake_run(prompt: str, system_prompt: str, **_: object) -> dict[str, object]:
        synthesis._record_narrative_synthesis_usage(  # noqa: SLF001 - diagnostic contract coverage
            guarded=True,
            elapsed_seconds=2.5,
        )
        return {
            "executive_summary": "The local narrative synthesis ran and preserved the methods detail.",
            "methods_strengths": ["The narrative synthesis used the supplied evidence packet."],
            "methods_weaknesses": [],
            "reproducibility_ethics": [],
            "uncertainty_gaps": [],
        }

    monkeypatch.setattr(synthesis, "_run_deep_json_prompt", _fake_run)
    monkeypatch.setattr(synthesis.settings, "llm_provider", "local")
    monkeypatch.setattr(synthesis.settings, "analysis_local_evidence_first_enabled", True)
    monkeypatch.setattr(synthesis.settings, "analysis_narrative_overrides_enabled", True)
    monkeypatch.setattr(synthesis.settings, "analysis_verifier_enabled", False)

    summary = synthesize_report(
        {
            "evidence_packets": [
                {
                    "finding_id": "m1",
                    "modality": "text",
                    "anchor": "section:Methods:1",
                    "statement": "Participants received fluoxetine 20 mg daily for 8 weeks.",
                    "evidence_refs": ["section:Methods:1"],
                    "confidence": 0.9,
                    "quality_flags": [],
                    "category": "medication",
                    "section_label": "methods",
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

    usage = summary["section_diagnostics"]["narrative_synthesis_usage"]
    assert usage["deep_calls"] >= 1
    assert usage["guarded_calls"] == usage["deep_calls"]
    assert usage["deep_total_seconds"] == usage["deep_calls"] * 2.5


def test_scientific_details_are_available_to_narrative_prompts(monkeypatch) -> None:
    captured: list[dict[str, object]] = []

    def _fake_run(prompt: str, system_prompt: str, **_: object) -> dict[str, object]:
        captured.append({"prompt": prompt, "system_prompt": system_prompt})
        return {}

    monkeypatch.setattr(synthesis, "_run_deep_json_prompt", _fake_run)
    payload = {
        "paper_meta": {"title": "Trial"},
        "coverage": {},
        "text_packets": [],
        "table_packets": [],
        "figure_packets": [],
        "supp_packets": [],
        "cross_modal_claims": [],
        "discrepancies": [],
        "sections_extracted": {
            "introduction": [
                {
                    "statement": "The objective was to test fluoxetine for depressive symptoms.",
                    "evidence_refs": ["section:Introduction:1"],
                    "confidence": 0.8,
                }
            ]
        },
        "scientific_details": [
            {
                "detail_types": ["medication_or_therapeutic", "dose_schedule"],
                "statement": "Participants received fluoxetine 20 mg daily for 8 weeks.",
                "source_excerpt": "Methods: participants received fluoxetine 20 mg daily for 8 weeks before outcome assessment.",
                "evidence_refs": ["section:Methods:1"],
                "source_modality": "text",
                "section_label": "methods",
                "category": "medication",
                "confidence": 0.9,
            }
        ],
    }

    synthesis._llm_synthesis_overrides(payload)  # noqa: SLF001 - prompt contract coverage
    synthesis._llm_verifier_overrides(payload, {"executive_summary": "Draft"})  # noqa: SLF001

    prompts = "\n".join(str(item["prompt"]) for item in captured)
    assert "evidence_plan" in prompts
    assert "scientific_detail_candidates" in prompts
    assert "fluoxetine 20 mg daily for 8 weeks" in prompts
    assert "before outcome assessment" in prompts
    assert "medication_or_therapeutic" in prompts
    assert "focus_slots" in prompts
    assert "The objective was to test fluoxetine" in prompts


def test_narrative_and_verifier_prompts_compact_to_valid_json(monkeypatch) -> None:
    captured: list[str] = []

    def _fake_run(prompt: str, system_prompt: str, **_: object) -> dict[str, object]:
        captured.append(prompt)
        return {}

    monkeypatch.setattr(synthesis, "_run_deep_json_prompt", _fake_run)
    monkeypatch.setattr(synthesis.settings, "llm_n_ctx", 1700)
    long_findings = [
        {
            "statement": f"Finding {idx} reports medication response with p < 0.05 and detailed context. " * 8,
            "evidence_refs": [f"section:Results:{idx}"],
        }
        for idx in range(18)
    ]
    payload = {
        "paper_meta": {"title": "Large local context trial"},
        "coverage": {},
        "text_packets": long_findings,
        "table_packets": long_findings,
        "figure_packets": long_findings,
        "supp_packets": long_findings,
        "cross_modal_claims": [{"claim": f"Claim {idx} " * 12} for idx in range(12)],
        "discrepancies": [{"description": f"Discrepancy {idx} " * 12} for idx in range(12)],
        "sections_extracted": {
            "introduction": [
                {
                    "statement": "The objective was to test a local-model synthesis path.",
                    "evidence_refs": ["section:Introduction:1"],
                    "confidence": 0.8,
                }
            ]
        },
        "scientific_details": [
            {
                "detail_types": ["medication_or_therapeutic", "dose_schedule", "statistical_result"],
                "statement": "Participants received fluoxetine 20 mg daily and improved with p < 0.05. " * 6,
                "source_excerpt": "Verbose medication/statistics excerpt. " * 30,
                "evidence_refs": ["section:Methods:1", "table:1"],
                "source_modality": "table",
                "section_label": "results",
                "category": "stats",
                "confidence": 0.9,
            }
        ],
    }

    synthesis._llm_synthesis_overrides(payload)  # noqa: SLF001
    synthesis._llm_verifier_overrides(payload, {"executive_summary": "Draft summary. " * 200})  # noqa: SLF001

    assert len(captured) == 2
    for prompt in captured:
        assert len(prompt) <= synthesis.max_chars_for_ctx(synthesis.settings.llm_n_ctx)
        parsed = json.loads(prompt.split("\n\n", 1)[1])
        assert "evidence_plan" in parsed
        assert "scientific_detail_candidates" in parsed
        assert "critical_missing_focus_slots" in parsed["evidence_plan"]
        assert "quality_flags" in parsed["evidence_plan"]


def test_executive_synthesis_payload_uses_scientific_detail_support() -> None:
    payload = synthesis._executive_report_synthesis_payload(  # noqa: SLF001 - deterministic prompt payload coverage
        {
            "introduction": [
                {
                    "statement": "The objective was to test whether fluoxetine improved depressive symptoms.",
                    "evidence_refs": ["section:Introduction:1"],
                    "confidence": 0.8,
                    "section_confidence": 0.8,
                    "source_modality": "text",
                }
            ],
            "results": [],
        },
        scientific_details=[
            {
                "detail_types": ["statistical_result"],
                "statement": "The fluoxetine arm had higher response than placebo with p < 0.05.",
                "source_excerpt": "Table 1. Response was higher in the fluoxetine arm than placebo; p < 0.05.",
                "evidence_refs": ["table:1"],
                "source_modality": "table",
                "section_label": "results",
                "category": "stats",
                "confidence": 0.82,
            }
        ],
    )

    assert "evidence_plan" in payload
    assert payload["sections"]["introduction"][0]["evidence_refs"] == ["section:Introduction:1"]
    assert payload["evidence_plan"]["focus_slots"]
    assert any(slot["status"] == "found" for slot in payload["evidence_plan"]["focus_slots"])
    assert payload["scientific_detail_candidates"][0]["statement"].startswith("The fluoxetine arm")
    assert "Table 1" in payload["scientific_detail_candidates"][0]["source_excerpt"]
    assert synthesis._executive_section_payload_supports(  # noqa: SLF001
        "results",
        "The fluoxetine arm had higher response than placebo with p < 0.05.",
        payload,
    )


def test_executive_synthesis_prompt_text_compacts_to_valid_json() -> None:
    extractive_evidence = {
        section: [
            {
                "statement": (
                    f"{section.title()} evidence row {idx} reports a detailed scientific point with methods, "
                    "statistics, modalities, and interpretation that should be compacted before prompt use."
                ),
                "verbatim_text": "This deliberately verbose source excerpt should be shortened. " * 20,
                "evidence_refs": [f"section:{section.title()}:{idx}"],
                "confidence": 0.85,
                "section_confidence": 0.9,
                "source_modality": "text",
            }
            for idx in range(12)
        ]
        for section in synthesis.EXEC_REPORT_SECTION_ORDER
    }
    payload = synthesis._executive_report_synthesis_payload(  # noqa: SLF001 - prompt budget contract
        extractive_evidence,
        scientific_details=[
            {
                "detail_types": ["cross_modal_result", "statistical_result"],
                "statement": "Figure 2 and Table 1 reported convergent response findings with p < 0.05.",
                "source_excerpt": "Figure and table excerpt. " * 30,
                "evidence_refs": ["figure:2", "table:1"],
                "source_modality": "figure",
                "section_label": "results",
                "category": "stats",
                "confidence": 0.82,
            }
        ],
    )

    prompt = synthesis._executive_report_synthesis_prompt_text(payload, max_chars=4500)  # noqa: SLF001
    parsed_payload = json.loads(prompt.split("\n\n", 1)[1])

    assert len(prompt) <= 4500
    assert "evidence_plan" in parsed_payload
    assert set(parsed_payload["sections"]) == set(synthesis.EXEC_REPORT_SECTION_ORDER)


def test_section_synthesis_v2_prompt_payload_is_budgeted_and_valid_json() -> None:
    section_inputs = {}
    for section in synthesis.EXEC_REPORT_SECTION_ORDER:
        section_inputs[section] = [
            {
                "statement": f"{section.title()} statement {idx} reports a paper-specific clinical or scientific detail.",
                "source_excerpt": (
                    f"{section.title()} source excerpt {idx}. "
                    + "This deliberately long excerpt should be trimmed before prompt serialization. " * 18
                ),
                "evidence_refs": [f"section:{section.title()}:{idx}"],
                "source": "sections_extracted",
                "source_modality": "text",
                "confidence": 0.8,
                "section_confidence": 0.9,
            }
            for idx in range(1, 10)
        ]
    scientific_details = [
        {
            "detail_types": ["medication_or_therapeutic", "dose_schedule"],
            "statement": "Participants received sertraline 50 mg daily before outcome assessment.",
            "source_excerpt": "Methods detail: sertraline 50 mg daily was administered before outcome assessment.",
            "evidence_refs": ["section:Methods:2"],
            "source_modality": "text",
            "section_label": "methods",
            "category": "medication",
            "confidence": 0.91,
        },
        {
            "detail_types": ["cross_modal_result", "statistical_result"],
            "statement": "Figure 2 reported higher response with p < 0.05.",
            "source_excerpt": "Figure 2 legend: response was higher in the active group, p < 0.05.",
            "evidence_refs": ["figure:2"],
            "source_modality": "figure",
            "section_label": "results",
            "category": "stats",
            "confidence": 0.86,
        },
    ]

    payload = synthesis._section_synthesis_v2_prompt_payload(  # noqa: SLF001 - prompt budget contract
        section_inputs=section_inputs,
        paper_type="randomized trial",
        scientific_details=scientific_details,
        max_chars=5200,
    )
    prompt = synthesis._section_synthesis_v2_prompt_text(payload, max_chars=5200)  # noqa: SLF001
    parsed_payload = json.loads(prompt.split("\n\n", 1)[1])

    assert len(prompt) <= 5200
    assert set(parsed_payload["sections"]) == set(synthesis.EXEC_REPORT_SECTION_ORDER)
    assert parsed_payload["evidence_plan"]["paper_type"] == "randomized trial"
    assert parsed_payload["evidence_plan"]["selected_detail_counts"]["by_modality"]["figure"] == 1
    assert parsed_payload["evidence_plan"]["focus_slots"]
    assert "critical_missing_focus_slots" in parsed_payload["evidence_plan"]
    assert "quality_flags" in parsed_payload["evidence_plan"]
    assert any(
        slot["slot_key"] == "safety_or_adverse_events"
        for slot in parsed_payload["evidence_plan"]["critical_missing_focus_slots"]
    )
    assert all(parsed_payload["sections"][section] for section in synthesis.EXEC_REPORT_SECTION_ORDER)
    assert "sertraline 50 mg daily" in json.dumps(parsed_payload["scientific_detail_candidates"])
    assert "figure" in json.dumps(parsed_payload["scientific_detail_candidates"]).lower()
    assert all(
        "source_modality" in row
        for rows in parsed_payload["sections"].values()
        for row in rows
    )


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


def test_synthesize_report_discloses_unavailable_supplements_in_uncertainty() -> None:
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
            "supp_figures": {"expected": 1, "extracted": 0, "missing_refs": ["Fig S1"]},
            "supp_tables": {"expected": 1, "extracted": 0, "missing_refs": ["Table S1"]},
        },
    )

    note = summary["supplement_availability_note"]
    assert "Supplement availability:" in note
    assert "not extracted or reviewed" in note
    assert any("not extracted or reviewed" in item for item in summary["uncertainty_gaps"])
    assert summary["section_diagnostics"]["supplement_availability_note"] == note


def test_synthesize_report_marks_supplement_only_source() -> None:
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
        text_chunk_records=[
            {
                "anchor": "page:1",
                "content": (
                    "Supplementary Material. This supplementary material provides detailed fitting statistics. "
                    "Table S1 provides useful statistics for each province."
                ),
            }
        ],
    )

    assert "supplementary material rather than the main article" in summary["supplement_availability_note"]
    assert any("rather than the main article" in item for item in summary["uncertainty_gaps"])


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


def test_tei_table_text_extracts_table_heading_divs() -> None:
    tei = """
    <TEI xmlns="http://www.tei-c.org/ns/1.0">
      <text>
        <body>
          <div>
            <head>TA B L E 1 Baseline clinical characteristics</head>
            <p>Arm Placebo Sertraline P-value</p>
            <p>Adverse events 4 7 .04</p>
          </div>
        </body>
      </text>
    </TEI>
    """
    tables = validated_pipeline._tei_table_text_extract(tei)

    assert len(tables) == 1
    assert tables[0].table_id == "1"
    assert tables[0].caption == "TA B L E 1 Baseline clinical characteristics"
    assert "Adverse events" in tables[0].text
    assert tables[0].meta["source"] == "grobid_tei_table_text"


def test_tei_table_text_extracts_table_like_figure_nodes() -> None:
    tei = """
    <TEI xmlns="http://www.tei-c.org/ns/1.0">
      <text>
        <body>
          <figure coords="6,100,200,300,120">
            <figDesc>TA B L E 5 Haplotype frequencies by disease group</figDesc>
          </figure>
        </body>
      </text>
    </TEI>
    """
    tables = validated_pipeline._tei_table_text_extract(tei)

    assert len(tables) == 1
    assert tables[0].table_id == "5"
    assert tables[0].page == 6
    assert "Haplotype frequencies" in tables[0].caption
    assert tables[0].meta["source"] == "grobid_tei_table_figure"


def test_table_text_rows_preserve_caption_and_content_for_preview() -> None:
    rows = validated_pipeline._table_text_rows(
        "TA B L E 1 Baseline characteristics Arm Placebo Sertraline P-value Adverse events 4 7 .04",
        caption="TA B L E 1 Baseline characteristics",
    )

    assert rows[0] == "TA B L E 1 Baseline characteristics"
    assert any("Adverse events" in row for row in rows)


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

    monkeypatch.setattr(synthesis.settings, "analysis_local_evidence_first_enabled", False)
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


def test_section_verifier_prompt_compacts_to_valid_json(monkeypatch) -> None:
    captured: list[str] = []

    def _fake_run(prompt: str, system_prompt: str, **_: object) -> dict[str, object]:
        captured.append(prompt)
        return {"sections": {}}

    def _item(section: str, idx: int) -> dict:
        anchor_section = section.title()
        return {
            "statement": (
                f"{anchor_section} statement {idx} includes detailed methods, results, limitations, "
                "and source evidence that should be compacted before local prompt use. " * 4
            ),
            "anchor": f"section:{anchor_section}:{idx}",
            "evidence_refs": [f"section:{anchor_section}:{idx}"],
            "confidence": 0.82,
            "section_confidence": 0.86,
            "section_source": "anchor",
            "flags": [],
        }

    monkeypatch.setattr(synthesis.settings, "analysis_local_evidence_first_enabled", False)
    monkeypatch.setattr(synthesis.settings, "analysis_section_verifier_enabled", True)
    monkeypatch.setattr(synthesis.settings, "llm_n_ctx", 1500)
    monkeypatch.setattr(synthesis, "_run_deep_json_prompt", _fake_run)
    sections = {
        section: {
            "items": [_item(section, idx) for idx in range(12)],
            "evidence_refs": [],
            "fallback_used": False,
            "fallback_reason": None,
        }
        for section in synthesis.EXEC_REPORT_SECTION_ORDER
    }
    payload = {
        "text_packets": [_item("results", idx) for idx in range(12)],
        "table_packets": [_item("results", idx) for idx in range(4)],
        "figure_packets": [_item("results", idx) for idx in range(4)],
        "supp_packets": [_item("results", idx) for idx in range(4)],
    }

    synthesis._verify_section_fidelity_with_llm(sections, payload=payload)  # noqa: SLF001

    assert captured
    prompt = captured[0]
    assert len(prompt) <= synthesis.max_chars_for_ctx(synthesis.settings.llm_n_ctx)
    parsed_payload = json.loads(prompt.split("\n\n", 1)[1])
    assert set(parsed_payload["draft_sections"]) == set(synthesis.EXEC_REPORT_SECTION_ORDER)
    assert "evidence_digest" in parsed_payload


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


def test_section_extraction_prompt_compacts_to_valid_json(monkeypatch) -> None:
    captured: list[str] = []

    def _fake_chat(prompt: str, system: str | None = None, temperature: float = 0.2) -> str:
        captured.append(prompt)
        return "{}"

    def _packet(section: str, idx: int) -> dict:
        title = section.title()
        return {
            "finding_id": f"{section}-{idx}",
            "anchor": f"section:{title}:{idx}",
            "statement": (
                f"{title} statement {idx} contains a detailed scientific extraction candidate with methods, "
                "results, statistics, implications, and source grounding that should be compacted. " * 4
            ),
            "evidence_refs": [f"section:{title}:{idx}"],
            "confidence": 0.84,
            "category": section,
            "section_label": section,
        }

    monkeypatch.setattr(synthesis, "chat_text_deep", _fake_chat)
    monkeypatch.setattr(synthesis.settings, "llm_n_ctx", 1600)
    payload = {
        "paper_type": "observational study",
        "text_packets": [
            _packet(section, idx)
            for section in synthesis.EXEC_REPORT_SECTION_ORDER
            for idx in range(14)
        ],
        "table_packets": [_packet("results", idx) for idx in range(4)],
        "figure_packets": [_packet("results", idx) for idx in range(4)],
        "supp_packets": [_packet("results", idx) for idx in range(4)],
        "scientific_details": [
            {
                "detail_types": ["statistical_result", "cross_modal_result"],
                "statement": "Table 2 and Figure 3 reported adjusted outcome effects with p < 0.05. " * 4,
                "source_excerpt": "Verbose statistical result excerpt. " * 20,
                "evidence_refs": ["table:2", "figure:3"],
                "source_modality": "table",
                "section_label": "results",
                "category": "stats",
                "confidence": 0.9,
            }
        ],
    }

    synthesis._llm_section_extraction_direct(payload)  # noqa: SLF001

    assert captured
    prompt = captured[0]
    assert len(prompt) <= synthesis.max_chars_for_ctx(synthesis.settings.llm_n_ctx)
    parsed_payload = json.loads(prompt.split("\n\n", 1)[1])
    assert set(parsed_payload["section_rows"]) == set(synthesis.EXEC_REPORT_SECTION_ORDER)
    assert "evidence_plan" in parsed_payload
    assert "scientific_detail_candidates" in parsed_payload


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
    monkeypatch.setattr(synthesis.settings, "analysis_local_evidence_first_enabled", False)
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
    monkeypatch.setattr(synthesis.settings, "analysis_local_evidence_first_enabled", False)
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
