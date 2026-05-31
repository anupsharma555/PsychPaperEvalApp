from __future__ import annotations

import json

from app.services.analysis.information_retention import build_information_retention_audit


def _chunk(anchor: str, section: str, content: str, modality: str = "text") -> dict:
    return {
        "anchor": anchor,
        "content": content,
        "meta": json.dumps({"section_norm": section, "section_raw_title": section.title()}),
        "modality": modality,
        "asset_kind": "main",
    }


def test_source_sentence_bank_uses_parsed_chunks_and_filters_noise() -> None:
    audit = build_information_retention_audit(
        document_id=1,
        source_assets=[],
        parsed_chunks=[
            _chunk(
                "section:Methods:1",
                "methods",
                (
                    "Participants completed a reward learning task during functional MRI acquisition. "
                    "doi: 10.1000/example copyright all rights reserved."
                ),
            )
        ],
        summary_json={},
    )

    assert audit["source_basis"] == "parsed_chunks"
    assert audit["source_sentence_count"] == 1
    assert audit["source_sentences"][0]["section"] == "methods"
    assert "doi:" not in audit["source_sentences"][0]["sentence"].lower()
    assert audit["compact_summary"]["source_basis_warning"]


def test_stage_extraction_counts_synthetic_report_payload() -> None:
    statement = "Participants completed a reward learning task during functional MRI acquisition."
    audit = build_information_retention_audit(
        document_id=1,
        source_assets=[],
        parsed_chunks=[_chunk("section:Methods:1", "methods", statement)],
        summary_json={
            "modalities": {
                "text": {
                    "findings": [
                        {
                            "statement": statement,
                            "section_label": "methods",
                            "evidence_refs": ["section:Methods:1"],
                        }
                    ]
                }
            },
            "sections_extracted": {"methods": [{"statement": statement, "evidence_refs": ["section:Methods:1"]}]},
            "sections_compact": {
                "methods": [{"statement": statement, "status": "found", "evidence_refs": ["section:Methods:1"]}]
            },
            "sections": {"methods": {"items": [{"statement": statement, "anchor": "section:Methods:1"}]}},
            "extractive_evidence": {"methods": [{"statement": statement, "anchor": "section:Methods:1"}]},
            "presentation_evidence": {"methods": [{"statement": statement, "anchor": "section:Methods:1"}]},
            "executive_report": {
                "sections": [
                    {
                        "section": "methods",
                        "summary": statement,
                        "bullets": [{"text": statement, "anchors": ["section:Methods:1"]}],
                    }
                ]
            },
        },
    )

    metrics = {row["stage"]: row for row in audit["stage_metrics"]}
    assert metrics["parsed_chunks"]["stage_item_count"] == 1
    assert metrics["text_packets"]["stage_item_count"] == 1
    assert metrics["sections"]["stage_item_count"] == 1
    assert metrics["executive_report"]["retained_rate"] == 1.0
    assert audit["first_loss_counts"]["retained_all_stages"] == 1


def test_first_loss_attribution_reports_section_drop_after_compact_stage() -> None:
    statement = "Participants completed a reward learning task during functional MRI acquisition."
    audit = build_information_retention_audit(
        document_id=1,
        source_assets=[],
        parsed_chunks=[_chunk("section:Methods:1", "methods", statement)],
        summary_json={
            "modalities": {
                "text": {
                    "findings": [
                        {
                            "statement": statement,
                            "section_label": "methods",
                            "evidence_refs": ["section:Methods:1"],
                        }
                    ]
                }
            },
            "sections_extracted": {"methods": [{"statement": statement, "evidence_refs": ["section:Methods:1"]}]},
            "sections_compact": {
                "methods": [{"statement": statement, "status": "found", "evidence_refs": ["section:Methods:1"]}]
            },
            "sections": {"methods": {"items": []}},
        },
    )

    source = audit["source_sentences"][0]
    assert source["stage_status"]["sections"] == "missing"
    assert source["first_lost_after"] == "sections"
    metrics = {row["stage"]: row for row in audit["stage_metrics"]}
    assert metrics["sections"]["lost_here_count"] == 1


def test_wrong_section_detection_counts_retained_but_misassigned_sentence() -> None:
    statement = "Participants completed a reward learning task during functional MRI acquisition."
    audit = build_information_retention_audit(
        document_id=1,
        source_assets=[],
        parsed_chunks=[_chunk("section:Methods:1", "methods", statement)],
        summary_json={
            "modalities": {
                "text": {
                    "findings": [
                        {
                            "statement": statement,
                            "section_label": "results",
                            "evidence_refs": ["section:Methods:1"],
                        }
                    ]
                }
            }
        },
    )

    source = audit["source_sentences"][0]
    assert source["stage_status"]["text_packets"] == "present_wrong_section"
    metrics = {row["stage"]: row for row in audit["stage_metrics"]}
    assert metrics["text_packets"]["retained_count"] == 1
    assert metrics["text_packets"]["wrong_section_count"] == 1
