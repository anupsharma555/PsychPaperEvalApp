from __future__ import annotations

import json

from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from app.db.models import Asset, Chunk, Report
from app.services import storage
from app.services.analysis.intermediate_artifacts import write_intermediate_stage_index


def test_build_intermediate_stage_index_organizes_run_stages(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(storage.settings, "data_dir", tmp_path)
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    artifact_dir = storage.artifacts_dir(42)
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "source_manifest.json").write_text(
        json.dumps(
            {
                "document_id": 42,
                "source_type": "upload",
                "status": "uploaded",
                "selected_assets": [{"asset_id": 1}],
                "supplements": [],
            }
        ),
        encoding="utf-8",
    )
    (artifact_dir / "parse_diagnostics.json").write_text(
        json.dumps({"counts": {"text": 2, "table": 1, "figure": 0, "supp": 0}}),
        encoding="utf-8",
    )
    (artifact_dir / "parser_asset_diagnostics.json").write_text(
        json.dumps({"assets": [{"status": "parsed"}, {"status": "reused"}]}),
        encoding="utf-8",
    )
    (artifact_dir / "information_retention_audit.json").write_text(
        json.dumps(
            {
                "compact_summary": {
                    "stage_metrics": [
                        {
                            "stage": "text_packets",
                            "lost_here_count": 9,
                            "wrong_section_rate": 0.4,
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    (artifact_dir / "run_timeline.json").write_text(
        json.dumps({"timeline": [{"step": "parse_document_assets"}]}),
        encoding="utf-8",
    )

    report_payload = {
        "sections": [{"section": "methods"}],
        "key_findings": ["Finding"],
        "scientific_details": [{"statement": "Dose was 20 mg."}],
        "evidence_packets": [
            {
                "finding_id": "p1",
                "modality": "text",
                "section_label": "methods",
                "statement": "Dose was 20 mg.",
                "source_excerpt": "Dose was 20 mg.",
                "detail_types": ["dose_schedule"],
                "usable_for_gold_comparison": True,
            },
            {
                "finding_id": "p2",
                "modality": "figure",
                "section_label": "unknown",
                "statement": "Figure shows response.",
                "usable_for_gold_comparison": False,
            },
        ],
        "section_diagnostics": {
            "synthesis_evidence_plan": {
                "critical_missing_focus_slots": ["sensitivity_analysis"],
            }
        },
        "artifact_organization": {
            "llm_input_inventory": {
                "schema_version": 1,
                "eligible_scientific_detail_count": 2,
                "selected_prompt_detail_count": 1,
                "omitted_candidate_count": 1,
                "selected_quality": {"missing_source_excerpt": 0},
                "focus_slot_counts": {"critical_missing": 1},
                "quality_flags": ["critical_focus_slots_missing"],
                "selected_detail_records": [
                    {
                        "prompt_index": 1,
                        "statement_sha256": "a" * 64,
                        "source_excerpt_sha256": "b" * 64,
                        "evidence_refs": ["section:methods:1"],
                        "section_label": "methods",
                        "source_modality": "text",
                        "detail_types": ["dose_schedule"],
                    }
                ],
            }
        },
    }
    diagnostics = {
        "modality_packet_counts": {"text": 1, "table": 0, "figure": 1, "supplement": 0},
        "run_validity": {
            "run_validity": "valid",
            "valid": True,
            "fallback_reasons": ["section_recall_fallback_used"],
        },
    }

    with Session(engine) as session:
        session.add(Asset(document_id=42, kind="main", filename="paper.pdf", path="/tmp/paper.pdf"))
        session.add(Chunk(document_id=42, asset_id=1, anchor="section:methods:1", modality="text", content="Dose"))
        session.add(Chunk(document_id=42, asset_id=1, anchor="table:1", modality="table", content="{}"))
        session.add(Report(document_id=42, payload=json.dumps(report_payload)))
        session.commit()

        index = write_intermediate_stage_index(session, 42, diagnostics=diagnostics)

    assert index["schema_version"] == 1
    assert index["stage_order"] == [
        "source_manifest",
        "parsed_chunks",
        "modality_packets",
        "audited_evidence_packets",
        "synthesis_inputs",
        "retention_audit",
        "final_report",
        "runtime_diagnostics",
    ]
    stage_by_id = {stage["stage_id"]: stage for stage in index["stages"]}
    assert stage_by_id["source_manifest"]["record_counts"]["selected_assets"] == 1
    assert stage_by_id["parsed_chunks"]["record_counts"]["text_chunks"] == 1
    assert stage_by_id["modality_packets"]["quality"]["source_excerpt_missing"] == 1
    assert stage_by_id["modality_packets"]["quality"]["unknown_section_count"] == 1
    assert stage_by_id["synthesis_inputs"]["record_counts"]["critical_missing_focus_slots"] == 1
    assert stage_by_id["synthesis_inputs"]["record_counts"]["llm_input_selected_prompt_details"] == 1
    assert stage_by_id["synthesis_inputs"]["quality"]["llm_input_inventory"]["present"] is True
    assert stage_by_id["synthesis_inputs"]["quality"]["llm_input_inventory"]["quality_flags"] == [
        "critical_focus_slots_missing"
    ]
    assert stage_by_id["synthesis_inputs"]["artifact_paths"][0]["path"] == "llm_input_inventory.json"
    assert stage_by_id["synthesis_inputs"]["artifact_paths"][0]["present"] is True
    assert stage_by_id["retention_audit"]["quality"]["worst_loss_stage"]["stage"] == "text_packets"
    transition_by_id = {row["transition_id"]: row for row in index["stage_transitions"]}
    assert transition_by_id["parsed_chunks_to_modality_packets"]["input_count"] == 2
    assert transition_by_id["parsed_chunks_to_modality_packets"]["output_count"] == 2
    assert transition_by_id["modality_packets_to_audited_packets"]["quality_gaps"] == {
        "source_excerpt_missing": 1,
        "unknown_section_count": 1,
        "untyped_packet_count": 1,
    }
    assert set(transition_by_id["modality_packets_to_audited_packets"]["diagnostic_flags"]) == {
        "source_excerpts_missing",
        "detail_types_missing",
        "section_labels_unknown",
    }
    assert transition_by_id["audited_packets_to_synthesis_inputs"]["diagnostic_flags"] == [
        "critical_focus_slots_missing"
    ]
    assert transition_by_id["retention_audit_to_final_report"]["diagnostic_flags"] == [
        "retention_loss_at_text_packets"
    ]
    assert index["llm_input_readiness"]["ready"] is False
    assert set(index["llm_input_readiness"]["blocking_flags"]) == {
        "packet_source_excerpts_incomplete",
        "packet_detail_typing_incomplete",
        "packet_section_assignment_incomplete",
    }
    sidecar = json.loads((artifact_dir / "llm_input_inventory.json").read_text(encoding="utf-8"))
    assert sidecar["document_id"] == 42
    assert sidecar["source"] == "report.artifact_organization.llm_input_inventory"
    assert sidecar["inventory"]["selected_prompt_detail_count"] == 1


def test_intermediate_stage_index_recovers_llm_input_inventory_from_report_details(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(storage.settings, "data_dir", tmp_path)
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    artifact_dir = storage.artifacts_dir(43)
    artifact_dir.mkdir(parents=True)

    report_payload = {
        "scientific_details": [
            {
                "statement": "Definitive pneumococcal pneumonia cases were adjudicated.",
                "source_excerpt": "",
                "evidence_refs": ["section:methods:1"],
                "source_modality": "text",
                "section_label": "methods",
                "category": "methods",
                "detail_types": ["design_source_sample_or_model"],
                "confidence": 0.8,
            },
            {
                "statement": "Serotype 19A was summarized by vaccination status.",
                "source_excerpt": "Serotype 19A was summarized by vaccination status.",
                "evidence_refs": ["table:3"],
                "source_modality": "table",
                "section_label": "results",
                "category": "results",
                "detail_types": ["primary_results_statistics_or_effects"],
                "confidence": 0.7,
            },
        ],
        "section_diagnostics": {
            "scientific_details_prompt_count": 2,
            "synthesis_evidence_plan": {
                "focus_slots": ["Objective/Rationale", "Safety or Adverse Events"],
                "missing_focus_slot_count": 1,
                "critical_missing_focus_slots": ["Safety or Adverse Events"],
            },
        },
    }

    with Session(engine) as session:
        session.add(Report(document_id=43, payload=json.dumps(report_payload)))
        session.commit()

        index = write_intermediate_stage_index(session, 43, diagnostics={})

    stage_by_id = {stage["stage_id"]: stage for stage in index["stages"]}
    synthesis_inputs = stage_by_id["synthesis_inputs"]
    assert synthesis_inputs["artifact_paths"][0]["present"] is True
    assert synthesis_inputs["record_counts"]["llm_input_selected_prompt_details"] == 2
    assert synthesis_inputs["record_counts"]["llm_input_eligible_scientific_details"] == 2
    assert synthesis_inputs["quality"]["llm_input_inventory"]["present"] is True
    assert synthesis_inputs["quality"]["llm_input_inventory"]["selected_quality"] == {
        "missing_detail_types": 0,
        "missing_source_excerpt": 1,
        "unknown_section": 0,
    }

    sidecar = json.loads((artifact_dir / "llm_input_inventory.json").read_text(encoding="utf-8"))
    assert sidecar["source"] == "report.scientific_details_and_section_diagnostics"
    assert sidecar["inventory"]["selected_prompt_detail_count"] == 2
    assert sidecar["inventory"]["omitted_candidate_count"] == 0
    assert "recovered_from_report_scientific_details" in sidecar["inventory"]["quality_flags"]
