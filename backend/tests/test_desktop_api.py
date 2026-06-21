from __future__ import annotations

from datetime import datetime, timedelta
import json
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
from sqlmodel import Session, SQLModel, create_engine

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("PAPER_EVAL_ROOT", str(PROJECT_ROOT))
os.environ.setdefault("DB_PATH", str(PROJECT_ROOT / "data" / "app.db"))
os.environ.setdefault("ANALYSIS_USE_PROCESS_POOL", "false")

from app.api import routes as api_routes
from app.db.models import Asset, Chunk, Document, Job, JobStatus, Report
from app.services import ingest as ingest_service


class _FakeRunner:
    def status(self) -> dict:
        return {
            "running": True,
            "paused": False,
            "inflight": 1,
            "worker_capacity": 2,
            "stale_jobs_recovered": 3,
            "last_recovery_at": "2026-02-09T12:00:00Z",
            "orphan_cleanup_last_run": {"checked": 2, "killed": 1, "skipped_mismatch": 0},
            "executor_enabled": True,
            "last_error": None,
        }

    def cleanup_orphans(self) -> dict:
        return {"checked": 0, "killed": 0, "skipped_mismatch": 0}

    def recover_state(self) -> dict:
        return {
            "recovered_jobs": 0,
            "total_recovered_jobs": 0,
            "last_recovery_at": None,
            "orphan_cleanup": {"checked": 0, "killed": 0, "skipped_mismatch": 0},
        }

    def pause(self) -> None:
        return None

    def resume(self) -> None:
        return None


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    SQLModel.metadata.create_all(engine)

    def get_test_session():
        with Session(engine) as session:
            yield session

    app = FastAPI()
    app.include_router(api_routes.router)
    app.dependency_overrides[api_routes.get_session] = get_test_session

    monkeypatch.setattr(api_routes, "job_runner", _FakeRunner())
    monkeypatch.setattr(
        api_routes,
        "read_pids",
        lambda: {"frontend_port": 3000, "backend_pid": 12345},
    )
    monkeypatch.setattr(
        api_routes,
        "read_runtime_events",
        lambda limit=25: [
            {
                "event": "backend_startup",
                "timestamp": "2026-02-09T10:00:00Z",
                "details": {"message": "Backend started"},
            },
            {
                "event": "startup_failed",
                "timestamp": "2026-02-09T10:05:00Z",
                "details": {"message": "Frontend failed"},
            },
            {
                "event": "shutdown_completed",
                "timestamp": "2026-02-09T10:10:00Z",
                "details": {"message": "Shutdown complete"},
            },
        ],
    )
    monkeypatch.setattr(
        api_routes,
        "artifacts_dir",
        lambda document_id: tmp_path / "artifacts" / str(document_id),
    )

    with Session(engine) as session:
        base = datetime(2026, 2, 9, 9, 0, 0)
        doc_1 = Document(title="Paper A", source_url="https://example.org/a")
        doc_2 = Document(title="Paper B", source_url=None)
        session.add(doc_1)
        session.add(doc_2)
        session.commit()
        session.refresh(doc_1)
        session.refresh(doc_2)

        job_1 = Job(
            document_id=doc_1.id,
            status=JobStatus.completed,
            progress=1.0,
            message="done",
            created_at=base,
            updated_at=base + timedelta(minutes=10),
        )
        job_2 = Job(
            document_id=doc_2.id,
            status=JobStatus.running,
            progress=0.6,
            message="running",
            created_at=base + timedelta(minutes=1),
            updated_at=base + timedelta(minutes=5),
        )
        session.add(job_1)
        session.add(job_2)

        report_payload = {
            "schema_version": 2,
            "executive_summary": "Desktop summary with concise findings. (77%)",
            "key_findings": [
                "Legacy intro line [section:body:0] (id:text-fallback-introduction-1)",
                "Legacy intro line [section:body:0] (id:text-fallback-introduction-2)",
            ],
            "methods_compact": [
                {"slot_key": "study_design", "label": "Study Design", "statement": "Randomized cohort design. (confidence 88%)", "status": "found"},
                {"slot_key": "sample_population", "label": "Sample/Population", "statement": "225 participants.", "status": "found"},
                {"slot_key": "inclusion_criteria", "label": "Inclusion Criteria", "statement": "DSM-based enrollment.", "status": "found"},
                {"slot_key": "exclusion_criteria", "label": "Exclusion Criteria", "statement": "Neurologic disorders excluded.", "status": "found"},
                {"slot_key": "intervention_or_exposure", "label": "Intervention/Exposure", "statement": "Resting-state connectome exposure.", "status": "found"},
                {"slot_key": "outcomes_measures", "label": "Outcomes/Measures", "statement": "BAS reward sensitivity.", "status": "found"},
                {"slot_key": "statistical_model", "label": "Statistical Model", "statement": "MDMR with covariate controls.", "status": "found"},
            ],
            "sections_compact_version": 1,
            "sections_compact": {
                "introduction": [
                    {
                        "section_key": "introduction",
                        "slot_key": "objective_hypothesis",
                        "label": "Objective/Hypothesis",
                        "statement": "Evaluate reward deficits across diagnoses. (80%)",
                        "status": "found",
                    }
                ],
                "methods": [],
                "results": [
                    {
                        "section_key": "results",
                        "slot_key": "primary_findings",
                        "label": "Primary Findings",
                        "statement": "Identified network dysconnectivity linked to reward deficits.",
                        "status": "found",
                    }
                ],
                "discussion": [],
                "conclusion": [],
            },
            "section_diagnostics": {
                "evidence_packet_coverage": {
                    "packet_total": 2,
                    "usable_packets": 2,
                    "usable_packet_rate": 1.0,
                    "sections_present": ["methods"],
                    "missing_core_sections": ["results"],
                    "by_section": {"methods": 2},
                    "by_modality": {"text": 2},
                    "by_detail_type": {"medication_or_therapeutic": 1},
                    "cross_modal_packet_count": 0,
                    "typed_packet_count": 1,
                    "quality_flags": ["no_cross_modal_packets", "no_results_packets"],
                },
                "synthesis_evidence_warnings": [
                    "Critical synthesis evidence not found for this paper type: Safety or Adverse Events."
                ],
                "synthesis_evidence_plan": {
                    "critical_missing_focus_slots": [
                        {
                            "slot_key": "safety_or_adverse_events",
                            "label": "Safety or Adverse Events",
                        }
                    ],
                    "quality_flags": ["critical_focus_slots_missing", "safety_or_adverse_events_missing"],
                },
            },
            "scientific_details_version": 1,
            "scientific_details": [
                {
                    "detail_types": ["medication_or_therapeutic", "outcome_measure"],
                    "statement": "Sertraline dosing was compared with placebo for depressive symptom change. (confidence 91%)",
                    "evidence_refs": ["section:Methods:1", "id:text-fallback-methods-1", "section:Methods:1"],
                    "source_modality": "text",
                    "section_label": "methods",
                    "category": "clinical",
                    "confidence": 0.91,
                },
                {
                    "detail_types": ["medication_or_therapeutic"],
                    "statement": "Sertraline dosing was compared with placebo for depressive symptom change.",
                    "evidence_refs": ["section:Methods:1"],
                    "source_modality": "text",
                    "section_label": "methods",
                    "category": "clinical",
                    "confidence": 0.9,
                },
                {
                    "detail_types": ["assay_readout"],
                    "statement": "This ungrounded item should not be displayed.",
                    "evidence_refs": [],
                    "source_modality": "text",
                    "section_label": "results",
                    "category": "assay",
                    "confidence": 0.8,
                },
            ],
            "coverage_snapshot_line": "Figure/Table Coverage: Main figures (text-cited/extracted): 0/0.",
            "modalities": {
                "text": {
                    "findings": [],
                    "highlights": [
                        "Legacy intro line [section:body:0] (id:text-fallback-introduction-1)",
                        "Legacy intro line [section:body:0]",
                    ],
                    "coverage_gaps": [],
                }
            },
            "sections": {
                "methods": {
                    "items": [
                        {"statement": "Sample size was 225 participants. [section:Methods:1]"},
                        {"statement": "Sample size was 225 participants. [section:Methods:1] (id:text-fallback-methods-2)"},
                    ],
                    "evidence_refs": ["section:Methods:1"],
                    "fallback_used": False,
                    "fallback_reason": "",
                }
            },
            "discrepancies": [],
            "overall_confidence": 0.8,
        }
        session.add(Report(document_id=doc_1.id, payload=json.dumps(report_payload)))
        session.add(
            Report(
                document_id=doc_2.id,
                payload=json.dumps({"schema_version": 2, "executive_summary": "Legacy payload"}),
            )
        )
        main_asset = Asset(
            document_id=doc_1.id,
            kind="main",
            filename="paper.pdf",
            content_type="application/pdf",
            path=str(tmp_path / "paper.pdf"),
        )
        session.add(main_asset)
        session.commit()
        session.refresh(main_asset)
        session.add_all(
            [
                Chunk(
                    document_id=doc_1.id,
                    asset_id=main_asset.id,
                    anchor="figure:10",
                    modality="figure",
                    content="",
                    meta=json.dumps(
                        {
                            "figure_id": "Figure10",
                            "caption": "Figure 10. Later image",
                            "source_url": "https://example.org/fig10.png",
                        }
                    ),
                ),
                Chunk(
                    document_id=doc_1.id,
                    asset_id=main_asset.id,
                    anchor="figure:2",
                    modality="figure",
                    content="",
                    meta=json.dumps(
                        {
                            "figure_id": "Figure2",
                            "caption": "Figure 2. Earlier image",
                            "source_url": "https://example.org/fig2.png",
                        }
                    ),
                ),
                Chunk(
                    document_id=doc_1.id,
                    asset_id=main_asset.id,
                    anchor="image:yoi260001f2_mock.png",
                    modality="figure",
                    content="",
                    meta=json.dumps(
                        {
                            "source": "image_file",
                            "caption": "",
                            "ocr_text": "Noisy OCR content from chart labels only.",
                        }
                    ),
                ),
                Chunk(
                    document_id=doc_1.id,
                    asset_id=main_asset.id,
                    anchor="section:Results:3",
                    modality="text",
                    content=(
                        "Figure 2 shows reduced ventral striatal coupling in the patient group and highlights "
                        "a stronger frontostriatal dysconnectivity pattern compared with controls."
                    ),
                    meta=json.dumps({"source": "docling"}),
                ),
                Chunk(
                    document_id=doc_1.id,
                    asset_id=main_asset.id,
                    anchor="table:9",
                    modality="table",
                    content=json.dumps({"columns": ["c1"], "data": [[1]]}),
                    meta=json.dumps(
                        {
                            "table_id": "Table9",
                            "caption": "Table 9. Later table",
                        }
                    ),
                ),
                Chunk(
                    document_id=doc_1.id,
                    asset_id=main_asset.id,
                    anchor="table:1",
                    modality="table",
                    content=json.dumps({"columns": ["c1"], "data": [[2]]}),
                    meta=json.dumps(
                        {
                            "table_id": "Table1",
                            "caption": "Table 1. Earlier table",
                        }
                    ),
                ),
            ]
        )
        session.commit()

    with TestClient(app) as test_client:
        yield test_client


def test_jobs_sort_and_filter(client: TestClient) -> None:
    resp = client.get("/api/jobs", params={"sort": "created_at:asc"})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["sort"] == "created_at:asc"
    assert payload["count"] == 2
    assert payload["items"][0]["message"] == "done"
    assert str(payload["items"][0]["created_at"]).endswith("+00:00")
    assert str(payload["items"][0]["updated_at"]).endswith("+00:00")

    filtered = client.get("/api/jobs", params={"status": "running"})
    assert filtered.status_code == 200
    filtered_payload = filtered.json()
    assert filtered_payload["count"] == 1
    assert filtered_payload["items"][0]["status"] == "running"


def test_get_job_returns_raw_job_id_contract(client: TestClient) -> None:
    resp = client.get("/api/jobs/1")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["id"] == 1
    assert payload["document_id"] == 1
    assert payload["status"] == "completed"
    assert payload["progress"] == 1.0
    assert payload["message"] == "done"


def test_jobs_invalid_sort_returns_remediation(client: TestClient) -> None:
    resp = client.get("/api/jobs", params={"sort": "unknown:desc"})
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert detail["error_code"] == "invalid_sort"
    assert "sort=updated_at:desc" in detail["next_action"]


def test_runtime_events_since_filter(client: TestClient) -> None:
    resp = client.get("/api/runtime/events", params={"since": "2026-02-09T10:04:59Z"})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["count"] == 2
    assert payload["events"][0]["kind"] == "startup_failed"
    assert payload["events"][0]["severity"] == "error"


def test_runtime_events_since_filter_accepts_epoch_seconds(client: TestClient) -> None:
    resp = client.get("/api/runtime/events", params={"since": "1777825749"})

    assert resp.status_code == 200
    assert "events" in resp.json()


def test_desktop_bootstrap_contract(client: TestClient) -> None:
    resp = client.get("/api/desktop/bootstrap")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["backend_ready"] is True
    assert payload["processing"]["worker_capacity"] == 2
    assert payload["models"]["model_path"]
    assert "text_model_path" in payload["models"]
    assert "deep_model_path" in payload["models"]
    assert "vision_model_path" in payload["models"]
    assert "vision_mmproj_path" in payload["models"]
    assert "provider" in payload["models"]
    assert "provider_ready" in payload["models"]
    assert "local_gpu_mode" in payload["models"]
    assert "local_gpu_effective_layers" in payload["models"]
    assert isinstance(payload.get("runtime_build"), dict)
    assert "git_commit" in payload["runtime_build"]
    assert "build_marker" in payload["runtime_build"]
    assert isinstance(payload["latest_jobs"], list)
    assert payload["lifecycle"]["latest_event"]["kind"] == "shutdown_completed"


def test_status_exposes_split_model_fields(client: TestClient) -> None:
    resp = client.get("/api/status")
    assert resp.status_code == 200
    payload = resp.json()
    assert "text_model_path" in payload
    assert "text_model_exists" in payload
    assert "deep_model_path" in payload
    assert "deep_model_exists" in payload
    assert "vision_model_path" in payload
    assert "vision_model_exists" in payload
    assert "vision_mmproj_path" in payload
    assert "vision_mmproj_exists" in payload
    assert "provider" in payload
    assert "provider_ready" in payload
    assert "local_gpu_mode" in payload
    assert "local_gpu_effective_layers" in payload
    assert "grobid" in payload
    assert isinstance(payload.get("runtime_build"), dict)
    assert "git_commit" in payload["runtime_build"]
    assert "build_marker" in payload["runtime_build"]


def test_from_url_blocked_access_returns_upload_remediation(client: TestClient, monkeypatch) -> None:
    def _blocked_ingest(*args, **kwargs):
        raise ValueError("Publisher blocked automated access (HTTP 403). Use From Upload and choose the main PDF.")

    monkeypatch.setattr(api_routes, "ingest_url", _blocked_ingest)

    resp = client.post(
        "/api/documents/from-url",
        json={"url": "https://example.org/paper", "doi": "10.1234/example", "fetch_supplements": True},
    )
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert detail["error_code"] == "ingest_url_failed"
    assert "publisher blocked automated access" in detail["user_message"].lower()
    assert "from upload" in detail["next_action"].lower()


def test_report_endpoint_exposes_analysis_diagnostics_field(client: TestClient) -> None:
    resp = client.get("/api/documents/1/report")
    assert resp.status_code == 200
    payload = resp.json()
    assert "analysis_diagnostics" in payload
    assert "summary_schema_version" in payload
    assert "sectioned_report_version" in payload


def test_report_endpoint_exposes_latency_profile_from_diagnostics(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(
        api_routes,
        "_load_analysis_diagnostics",
        lambda _document_id: {
            "diagnostics": {
                "latency_profile": {
                    "total_known_seconds": 694.5,
                    "slowest_stage": "text",
                    "top_bottlenecks": [
                        {"stage": "text", "duration_seconds": 420.0},
                        {"stage": "figure", "duration_seconds": 210.0},
                    ],
                }
            }
        },
    )

    resp = client.get("/api/documents/1/report")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["latency_profile"]["slowest_stage"] == "text"
    assert payload["latency_profile"]["top_bottlenecks"][0]["duration_seconds"] == 420.0


def test_report_endpoint_sanitizes_legacy_fallback_ids_and_dedupes(client: TestClient) -> None:
    resp = client.get("/api/documents/1/report")
    assert resp.status_code == 200
    payload = resp.json()
    summary = payload["summary_json"]

    key_findings = summary.get("key_findings", [])
    assert len(key_findings) == 1
    assert "text-fallback" not in key_findings[0].lower()

    text_highlights = summary.get("modalities", {}).get("text", {}).get("highlights", [])
    assert len(text_highlights) == 1
    assert "text-fallback" not in text_highlights[0].lower()

    methods_items = summary.get("sections", {}).get("methods", {}).get("items", [])
    assert len(methods_items) == 1
    assert "text-fallback" not in str(methods_items[0].get("statement", "")).lower()


def test_report_summary_includes_methods_card(client: TestClient) -> None:
    resp = client.get("/api/documents/1/report/summary")
    assert resp.status_code == 200
    payload = resp.json()
    assert "methods_card" in payload
    assert len(payload["methods_card"]) == 6
    assert payload["methods_card"][0].startswith("Study Design:")
    assert all("(confidence" not in line.lower() for line in payload["methods_card"])
    assert "sections_card" in payload
    assert payload["sections_card"][0].startswith("Objective/Hypothesis:")
    assert all("(confidence" not in line.lower() for line in payload["sections_card"])
    assert all("(80%)" not in line.lower() for line in payload["sections_card"])
    assert payload["scientific_card"] == [
        "Medication Or Therapeutic, Outcome Measure / Methods: Sertraline dosing was compared with placebo for depressive symptom change. [section:Methods:1]"
    ]
    assert payload["rerun_recommended"] is False
    assert payload["report_capabilities"]["sections_compact"] is True
    assert payload["report_capabilities"]["scientific_details"] is True
    assert any(
        "Safety or Adverse Events" in warning
        for warning in payload["report_warnings"]
    )
    assert any("missing Results packets" in warning for warning in payload["report_warnings"])
    assert any("No table, figure, or supplement evidence packets" in warning for warning in payload["report_warnings"])
    assert "safety_or_adverse_events_missing" in payload["synthesis_quality_flags"]
    assert payload["critical_missing_focus_slots"] == [
        {"slot_key": "safety_or_adverse_events", "label": "Safety or Adverse Events"}
    ]
    assert payload["overall_confidence"] is None
    assert "(confidence" not in str(payload.get("executive_summary", "")).lower()
    assert "(77%)" not in str(payload.get("executive_summary", "")).lower()
    assert all(
        "text-fallback" not in " ".join(str(item) for item in card.get("highlights", [])).lower()
        for card in payload.get("modality_cards", [])
    )


def test_report_summary_uses_analysis_diagnostics_validity(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(api_routes.settings, "llm_provider", "openai")
    monkeypatch.setattr(
        api_routes,
        "_load_analysis_diagnostics",
        lambda _document_id: {
            "diagnostics": {
                "model_usage": {"text_calls": 1, "text_errors": 1},
                "section_diagnostics": {"section_extraction_enabled": True},
            }
        },
    )

    resp = client.get("/api/documents/1/report/summary")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["report_status"] == "failed"
    assert payload["rerun_recommended"] is True
    assert payload["report_capabilities"]["openai_valid"] is False
    assert "OpenAI model calls failed" in str(payload["executive_summary"])


def test_report_summary_marks_local_invalid_run_failed(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(api_routes.settings, "llm_provider", "local")
    monkeypatch.setattr(
        api_routes,
        "_load_analysis_diagnostics",
        lambda _document_id: {
            "diagnostics": {
                "model_usage": {"text_calls": 0},
                "section_diagnostics": {"section_extraction_enabled": True},
            }
        },
    )

    resp = client.get("/api/documents/1/report/summary")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["report_status"] == "failed"
    assert payload["rerun_recommended"] is True
    assert payload["report_capabilities"]["analysis_valid"] is False
    assert payload["report_capabilities"]["local_valid"] is False
    assert "Local model text analysis did not run" in str(payload["executive_summary"])


def test_report_validity_refreshes_stale_cached_local_text_run(monkeypatch) -> None:
    monkeypatch.setattr(api_routes.settings, "llm_provider", "local")
    summary = {
        "analysis_diagnostics": {
            "diagnostics": {
                "model_usage": {"text_calls": 0},
                "analysis_timeline": [
                    {
                        "stage": "text",
                        "metadata": {"cache_hit": True, "cache_source": "global"},
                    }
                ],
                "section_diagnostics": {"section_extraction_enabled": True},
                "run_validity": {
                    "run_validity": "invalid",
                    "valid": False,
                    "rerun_required": True,
                    "reasons": ["text_llm_calls_zero"],
                    "blockers": {"text_llm_calls_zero": True},
                    "quality_backend_audit": {
                        "text_calls": 0,
                        "text_cache_reused": False,
                        "blockers": {"text_calls_zero": True},
                    },
                },
            }
        }
    }

    validity = api_routes._report_validity(summary, None)

    assert "text_analysis_cache_reused" in validity["reasons"]
    assert "text_llm_calls_zero" not in validity["reasons"]
    assert validity["blockers"]["text_analysis_cache_reused"] is True
    assert validity["blockers"]["text_llm_calls_zero"] is False
    assert validity["quality_backend_audit"]["text_cache_reused"] is True
    assert "Cached local text-analysis output was reused" in api_routes._openai_report_invalid_reason(summary)


def test_report_summary_exposes_prompt_budget_warnings(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(api_routes.settings, "llm_provider", "local")
    monkeypatch.setattr(
        api_routes,
        "_load_analysis_diagnostics",
        lambda _document_id: {
            "diagnostics": {
                "model_usage": {"text_calls": 1},
                "section_diagnostics": {"section_extraction_enabled": True},
                "prompt_budget_diagnostics": {
                    "totals": {"prompt_calls": 8, "max_prompt_chars": 15000, "max_prompt_modality": "figure"},
                    "quality_flags": ["many_local_prompt_batches", "figure_large_prompt"],
                },
            }
        },
    )

    resp = client.get("/api/documents/1/report/summary")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["report_status"] == "ready"
    assert payload["rerun_recommended"] is False
    assert any("Local context pressure" in warning for warning in payload["report_warnings"])
    assert any("many local-model prompt batches" in warning for warning in payload["report_warnings"])


def test_report_summary_exposes_latency_profile(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(api_routes.settings, "llm_provider", "local")
    monkeypatch.setattr(
        api_routes,
        "_load_analysis_diagnostics",
        lambda _document_id: {
            "diagnostics": {
                "model_usage": {"text_calls": 1},
                "section_diagnostics": {"section_extraction_enabled": True},
                "latency_profile": {
                    "total_known_seconds": 694.5,
                    "slowest_stage": "text",
                    "top_bottlenecks": [
                        {"stage": "text", "duration_seconds": 420.0},
                        {"stage": "figure", "duration_seconds": 210.0},
                    ],
                    "quality_flags": ["long_text_stage"],
                },
            }
        },
    )

    resp = client.get("/api/documents/1/report/summary")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["report_status"] == "ready"
    assert payload["latency_profile"]["total_known_seconds"] == 694.5
    assert payload["latency_profile"]["slowest_stage"] == "text"
    assert payload["report_capabilities"]["latency_profile"] is True


def test_report_summary_exposes_compact_run_validity(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(api_routes.settings, "llm_provider", "local")
    monkeypatch.setattr(
        api_routes,
        "_load_analysis_diagnostics",
        lambda _document_id: {
            "diagnostics": {
                "model_usage": {"text_calls": 1, "deep_calls": 0},
                "stage_model_usage": {"synthesis": {"deep_calls": 0}},
                "section_diagnostics": {"section_extraction_enabled": True},
                "synthesis_diagnostics": {
                    "narrative_overrides_enabled": True,
                    "narrative_synthesis_required": True,
                    "narrative_synthesis_deep_calls": 0,
                },
            }
        },
    )

    resp = client.get("/api/documents/1/report/summary")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["report_status"] == "failed"
    assert payload["rerun_recommended"] is True
    assert payload["report_validity"]["valid"] is False
    assert payload["report_validity"]["narrative_synthesis_required"] is True
    assert payload["report_validity"]["narrative_synthesis_ran"] is False
    assert "narrative_synthesis_calls_zero" in payload["report_validity"]["reasons"]


def test_report_summary_warns_on_local_section_fallback_without_failing(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(api_routes.settings, "llm_provider", "local")
    monkeypatch.setattr(
        api_routes,
        "_load_analysis_diagnostics",
        lambda _document_id: {
            "diagnostics": {
                "model_usage": {"text_calls": 1},
                "section_diagnostics": {
                    "section_extraction_enabled": True,
                    "introduction": {"fallback_used": True},
                },
                "sections_fallback_notes": [
                    "Introduction: Explicit introduction anchors were sparse; supplemented with early-body text."
                ],
                "fallback_counts_by_reason": {"introduction_fallback": 1},
            }
        },
    )

    resp = client.get("/api/documents/1/report/summary")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["report_status"] == "ready"
    assert payload["rerun_recommended"] is False
    assert any("Section-level evidence required" in warning for warning in payload["report_warnings"])


def test_report_summary_warns_from_diagnostics_evidence_packet_coverage(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(api_routes.settings, "llm_provider", "local")
    monkeypatch.setattr(
        api_routes,
        "_load_analysis_diagnostics",
        lambda _document_id: {
            "diagnostics": {
                "model_usage": {"text_calls": 1},
                "section_diagnostics": {
                    "section_extraction_enabled": True,
                    "evidence_packet_coverage": {
                        "packet_total": 1,
                        "usable_packets": 0,
                        "usable_packet_rate": 0.0,
                        "missing_core_sections": ["methods", "results"],
                        "cross_modal_packet_count": 0,
                        "typed_packet_count": 0,
                        "quality_flags": ["no_usable_evidence_packets", "no_typed_evidence_packets"],
                    },
                },
            }
        },
    )

    resp = client.get("/api/documents/1/report/summary")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["report_status"] == "ready"
    assert any("No usable evidence packets" in warning for warning in payload["report_warnings"])
    assert any("missing Methods, Results packets" in warning for warning in payload["report_warnings"])
    assert any("No typed scientific evidence packets" in warning for warning in payload["report_warnings"])


def test_report_summary_enriches_stored_validity_with_prompt_budget_warnings(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(api_routes.settings, "llm_provider", "local")
    monkeypatch.setattr(
        api_routes,
        "_load_analysis_diagnostics",
        lambda _document_id: {
            "diagnostics": {
                "model_usage": {"text_calls": 1},
                "section_diagnostics": {"section_extraction_enabled": True},
                "prompt_budget_diagnostics": {
                    "totals": {"prompt_calls": 8, "max_prompt_chars": 15000, "max_prompt_modality": "figure"},
                    "quality_flags": ["many_local_prompt_batches", "figure_large_prompt"],
                },
                "run_validity": {
                    "valid": True,
                    "run_validity": "valid",
                    "rerun_required": False,
                    "reasons": [],
                },
            }
        },
    )

    resp = client.get("/api/documents/1/report/summary")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["report_status"] == "ready"
    assert payload["rerun_recommended"] is False
    assert any("Local context pressure" in warning for warning in payload["report_warnings"])


def test_report_summary_marks_legacy_payload_for_rerun(client: TestClient) -> None:
    resp = client.get("/api/documents/2/report/summary")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["report_status"] == "ready"
    assert payload["rerun_recommended"] is True
    assert payload["report_capabilities"]["sections_compact"] is False


def test_sanitize_summary_payload_strips_fallback_ids_from_evidence_refs() -> None:
    raw = {
        "methods_compact": [
            {
                "label": "Study Design",
                "statement": "Randomized design.",
                "evidence_refs": ["id:text-fallback-methods-1", "section:Methods:1", "section:Methods:1"],
            }
        ],
        "sections_compact": {
            "conclusion": [
                {
                    "label": "Takeaway",
                    "statement": "Main conclusion.",
                    "evidence_refs": ["id:text-fallback-conclusion-1", "section:Conclusions:1"],
                }
            ]
        },
        "sections": {
            "conclusion": {
                "items": [
                    {
                        "statement": "Main conclusion. (id:text-fallback-conclusion-2)",
                        "evidence_refs": ["id:text-fallback-conclusion-2", "section:Conclusions:1"],
                    }
                ]
            }
        },
    }

    sanitized = api_routes._sanitize_summary_payload(raw)
    assert sanitized["methods_compact"][0]["evidence_refs"] == ["section:Methods:1"]
    assert sanitized["sections_compact"]["conclusion"][0]["evidence_refs"] == ["section:Conclusions:1"]
    conclusion_items = sanitized["sections"]["conclusion"]["items"]
    assert len(conclusion_items) == 1
    assert conclusion_items[0]["evidence_refs"] == ["section:Conclusions:1"]
    assert "text-fallback" not in conclusion_items[0]["statement"].lower()


def test_sanitize_summary_payload_filters_authors_and_recomputes_counts() -> None:
    raw = {
        "paper_meta": {
            "title": "Sample",
            "authors": [
                "Ana Smith",
                "Ben Jones",
                "Department of Radiology Research JAMA Psychiatry Xiamen Hospital",
            ],
            "authors_extracted_count": 23,
            "authors_display_count": 23,
        }
    }
    sanitized = api_routes._sanitize_summary_payload(raw)
    meta = sanitized["paper_meta"]
    assert meta["authors"] == ["Ana Smith", "Ben Jones"]
    assert meta["authors_extracted_count"] == 2
    assert meta["authors_display_count"] == 2


def test_media_endpoint_sorts_assets_by_ordinal_number(client: TestClient) -> None:
    resp = client.get("/api/documents/1/media")
    assert resp.status_code == 200
    payload = resp.json()
    figures = payload.get("figures", [])
    figure_anchors = [str(item.get("anchor", "")) for item in payload.get("figures", [])]
    table_anchors = [str(item.get("anchor", "")) for item in payload.get("tables", [])]
    tables = payload.get("tables", [])
    assert "figure:2" in figure_anchors
    assert "figure:10" in figure_anchors
    assert figure_anchors.index("figure:2") < figure_anchors.index("figure:10")
    assert table_anchors[:2] == ["table:1", "table:9"]
    assert all(not str(item.get("legend", "")).lstrip().startswith('{"columns"') for item in tables)
    assert tables[0]["legend_source"] == "missing"
    assert tables[0]["table_preview"]["columns"] == ["c1"]
    assert figures[0]["legend_source"] == "linked_text"
    assert "ventral striatal coupling" in str(figures[0]["legend"]).lower()
    assert str(figures[0]["legend"]).strip().lower() != str(figures[0]["caption"]).strip().lower()
    linked = next((item for item in figures if str(item.get("source", "")).lower() == "image_file"), None)
    assert linked is not None
    assert linked["caption"] == "Figure 2. Earlier image"
    assert linked["legend_source"] in {"missing", "ocr", "linked_text"}


def test_media_legend_uses_ocr_when_caption_unavailable() -> None:
    chunk = Chunk(
        document_id=1,
        anchor="figure:3",
        modality="figure",
        content="",
    )
    legend, source = api_routes._media_legend(
        chunk,
        {
            "caption": "",
            "ocr_text": "Figure 3. Connectivity maps show reduced ventral striatal coupling in major depression.",
        },
    )
    assert source == "ocr"
    assert "connectivity maps show reduced ventral striatal coupling" in legend.lower()


def test_extract_source_figure_legend_maps_from_html_text_blocks() -> None:
    html = """
    <html>
      <body>
        <div>Figure 2. Nodal Heterogeneity of Extreme Topological Deviations</div>
        <div>View Large</div>
        <div>Download</div>
        <div>Go to Figure in Article</div>
        <div>This figure shows case-control spatial overlap in nodal deviations across topological metrics.</div>
        <div>Abbreviations: ADHD, attention-deficit/hyperactivity disorder.</div>
        <div>Figure 3. Multivariate Distance-Based Regression</div>
      </body>
    </html>
    """
    maps = ingest_service._extract_source_figure_legend_maps(html)
    caption_map = maps.get("caption_map", {})
    legend_map = maps.get("legend_map", {})
    assert caption_map.get("2", "").startswith("Figure 2.")
    assert "case-control spatial overlap" in legend_map.get("2", "").lower()


def test_load_source_figure_legend_maps_normalizes_tokens(tmp_path, monkeypatch) -> None:
    def _fake_artifacts_dir(document_id: int) -> Path:
        path = tmp_path / f"doc_{document_id}" / "artifacts"
        path.mkdir(parents=True, exist_ok=True)
        return path

    monkeypatch.setattr(api_routes, "artifacts_dir", _fake_artifacts_dir)
    payload = {
        "caption_map": {"Figure 2": "Figure 2. Nodal map"},
        "legend_map": {"2A": "Panel A shows a stronger frontostriatal dysconnectivity pattern."},
    }
    (_fake_artifacts_dir(99) / "source_figure_legends.json").write_text(json.dumps(payload), encoding="utf-8")
    caption_map, legend_map = api_routes._load_source_figure_legend_maps(99)
    assert caption_map.get("2") == "Figure 2. Nodal map"
    assert "frontostriatal dysconnectivity" in legend_map.get("2A", "").lower()


def test_openai_report_invalid_reason_detects_model_errors(monkeypatch) -> None:
    monkeypatch.setattr(api_routes.settings, "llm_provider", "openai")
    summary = {
        "analysis_diagnostics": {
            "diagnostics": {
                "model_usage": {"text_calls": 1, "text_errors": 1, "vision_calls": 0, "vision_errors": 0}
            }
        }
    }

    reason = api_routes._openai_report_invalid_reason(summary)

    assert "OpenAI model calls failed" in reason


def test_openai_report_invalid_reason_detects_invalid_key_fallback(monkeypatch) -> None:
    monkeypatch.setattr(api_routes.settings, "llm_provider", "openai")
    summary = {
        "uncertainty_gaps": [
            "Text LLM stage failed in isolated subprocess; used deterministic section extraction fallback "
            "(OpenAI request failed with HTTP 401: Incorrect API key provided)."
        ]
    }

    reason = api_routes._openai_report_invalid_reason(summary)

    assert "OpenAI model calls failed" in reason
