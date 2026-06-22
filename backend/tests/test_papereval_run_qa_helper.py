from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sqlite3
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HELPER_PATH = PROJECT_ROOT / ".codex" / "skills" / "papereval-run-qa" / "scripts" / "papereval_run_qa.py"


def _load_helper_module():
    spec = importlib.util.spec_from_file_location("papereval_run_qa", HELPER_PATH)
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


def test_llm_input_inventory_summary_keeps_bounded_trace_fields() -> None:
    helper = _load_helper_module()

    summary = helper._llm_input_inventory_summary(
        {
            "schema_version": 1,
            "eligible_scientific_detail_count": 3,
            "selected_prompt_detail_count": 2,
            "omitted_candidate_count": 1,
            "selected_quality": {"missing_source_excerpt": 1},
            "focus_slot_counts": {"critical_missing": 1},
            "quality_flags": ["selected_prompt_details_missing_source_excerpt"],
            "selected_detail_records": [
                {
                    "prompt_index": 1,
                    "section_label": "results",
                    "source_modality": "table",
                    "detail_types": ["statistical_result"],
                    "evidence_refs": ["table:1"],
                    "statement_sha256": "a" * 64,
                    "source_excerpt_sha256": "b" * 64,
                    "statement": "raw text should not be surfaced",
                }
            ],
        }
    )

    assert summary["eligible_scientific_detail_count"] == 3
    assert summary["selected_quality"] == {"missing_source_excerpt": 1}
    assert summary["selected_detail_refs"] == [
        {
            "prompt_index": 1,
            "section_label": "results",
            "source_modality": "table",
            "detail_types": ["statistical_result"],
            "evidence_refs": ["table:1"],
            "statement_sha256": "a" * 64,
            "source_excerpt_sha256": "b" * 64,
        }
    ]


def test_artifact_organization_audit_flags_traceability_issues(tmp_path: Path) -> None:
    helper = _load_helper_module()
    data_dir = tmp_path / "data"
    artifacts = data_dir / "doc_42" / "artifacts"
    artifacts.mkdir(parents=True)
    helper.DATA_DIR = data_dir
    helper.DB_PATH = data_dir / "app.db"

    with sqlite3.connect(helper.DB_PATH) as conn:
        conn.execute(
            """
            create table report (
                id integer primary key,
                document_id integer not null,
                payload text not null,
                created_at text not null
            )
            """
        )
        payload = {
            "executive_summary": "Final summary.",
            "executive_report": {"sections": []},
            "sections": {"results": {"items": []}},
            "evidence_packets": [
                {
                    "finding_id": "fig-1",
                    "modality": "figure",
                    "section_label": "unknown",
                    "anchor": "figure:1",
                    "statement": "Figure 1 shows the endpoint.",
                    "evidence_refs": ["figure:1"],
                    "category": "figure_extractive_summary",
                    "confidence": 0.45,
                    "usable_for_gold_comparison": False,
                },
                {
                    "finding_id": "supp-1",
                    "modality": "supplement",
                    "section_label": "methods",
                    "anchor": "section:Methods:1",
                    "statement": "Supplementary methods are referenced.",
                    "evidence_refs": ["section:Methods:1"],
                    "category": "supplement_extractive_summary",
                    "confidence": 0.45,
                    "usable_for_gold_comparison": True,
                },
            ],
            "extractive_evidence": {"results": []},
            "section_diagnostics": {},
            "coverage": {
                "supp_figures": {"missing_refs": ["S1"]},
                "supp_tables": {"missing_refs": ["S2"]},
            },
            "supplement_availability_note": "",
        }
        conn.execute(
            "insert into report (id, document_id, payload, created_at) values (?, ?, ?, ?)",
            (1, 42, json.dumps(payload), "2026-06-21T00:00:00"),
        )

    (artifacts / "source_manifest.json").write_text(
        json.dumps({"schema_version": 1, "document_id": 42, "supplements": []}),
        encoding="utf-8",
    )
    (artifacts / "analysis_diagnostics.json").write_text(
        json.dumps({"diagnostics": {"analysis_timing": {"analysis_total_seconds": 1.0}}}),
        encoding="utf-8",
    )
    (artifacts / "information_retention_audit.json").write_text(
        json.dumps(
            {
                "compact_summary": {
                    "stage_metrics": [
                        {"stage": "text_packets", "lost_here_count": 7, "wrong_section_rate": 0.5}
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    (artifacts / "intermediate_stage_index.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stage_order": ["source_manifest", "parsed_chunks", "modality_packets"],
                "quality_flags": ["packet_detail_typing_incomplete"],
                "llm_input_readiness": {
                    "ready": False,
                    "blocking_flags": ["packet_detail_typing_incomplete"],
                },
                "stage_transitions": [
                    {
                        "transition_id": "modality_packets_to_audited_packets",
                        "from_stage": "modality_packets",
                        "to_stage": "audited_evidence_packets",
                        "loss_count": 4,
                        "loss_rate": 0.5,
                        "diagnostic_flags": ["detail_types_missing"],
                    }
                ],
                "stages": [
                    {"stage_id": "source_manifest"},
                    {"stage_id": "parsed_chunks"},
                    {"stage_id": "modality_packets"},
                ],
            }
        ),
        encoding="utf-8",
    )

    audit = helper._artifact_organization_audit(42)

    assert audit["report_payload_present"] is True
    assert audit["stage_boundaries"]["intermediate_stage_index_present"] is True
    assert audit["stage_boundaries"]["native_artifact_organization_present"] is False
    assert audit["native_artifact_organization_quality_flags"] == []
    assert audit["intermediate_stage_index"]["stage_order"] == [
        "source_manifest",
        "parsed_chunks",
        "modality_packets",
    ]
    assert audit["intermediate_stage_index"]["llm_input_readiness"]["ready"] is False
    assert audit["intermediate_stage_index"]["stage_transitions"] == [
        {
            "transition_id": "modality_packets_to_audited_packets",
            "from_stage": "modality_packets",
            "to_stage": "audited_evidence_packets",
            "loss_count": 4,
            "loss_rate": 0.5,
            "diagnostic_flags": ["detail_types_missing"],
        }
    ]
    assert audit["evidence_packet_field_completeness"]["source_excerpt"] == 0
    assert audit["supplement_source_consistency"] == {
        "uploaded_supplement_count": 0,
        "supplement_packet_count": 1,
        "missing_supplement_ref_count": 2,
        "supplement_availability_note_present": False,
    }
    assert audit["figure_source_consistency"] == {
        "figure_packet_count": 1,
        "usable_figure_packet_count": 0,
    }
    assert audit["retention_worst_loss_stage"]["stage"] == "text_packets"
    assert {
        "flat_payload_mixes_final_and_intermediate_fields",
        "evidence_packets_missing_source_excerpt",
        "evidence_packets_missing_detail_types",
        "evidence_packets_unknown_section",
        "figure_packets_not_usable_for_gold_comparison",
        "main_text_supplement_references_are_labeled_as_supplement_packets",
        "missing_supplements_without_availability_note",
    }.issubset({issue["code"] for issue in audit["issues"]})
