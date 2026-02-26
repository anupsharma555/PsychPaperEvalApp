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
    assert payload["rerun_recommended"] is False
    assert payload["report_capabilities"]["sections_compact"] is True
    assert payload["overall_confidence"] is None
    assert "(confidence" not in str(payload.get("executive_summary", "")).lower()
    assert "(77%)" not in str(payload.get("executive_summary", "")).lower()
    assert all(
        "text-fallback" not in " ".join(str(item) for item in card.get("highlights", [])).lower()
        for card in payload.get("modality_cards", [])
    )


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


def test_media_endpoint_sorts_assets_by_ordinal_number(client: TestClient) -> None:
    resp = client.get("/api/documents/1/media")
    assert resp.status_code == 200
    payload = resp.json()
    figure_anchors = [str(item.get("anchor", "")) for item in payload.get("figures", [])]
    table_anchors = [str(item.get("anchor", "")) for item in payload.get("tables", [])]
    assert figure_anchors[:2] == ["figure:2", "figure:10"]
    assert table_anchors[:2] == ["table:1", "table:9"]
