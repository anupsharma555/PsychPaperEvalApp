from __future__ import annotations

from datetime import datetime, timedelta
import json
from pathlib import Path

from sqlmodel import Session, SQLModel, create_engine, select

from app.core.config import settings
from app.db.models import Document, Report
from app.services.report_retention import enforce_report_retention, save_report_export, saved_reports_dir


def test_retention_keeps_max_ten_live_reports(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "data_dir", Path(tmp_path / "data"))
    settings.data_dir.mkdir(parents=True, exist_ok=True)

    db_path = tmp_path / "retention.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)

    now = datetime(2026, 2, 21, 12, 0, 0)
    with Session(engine) as session:
        doc_ids: list[int] = []
        for idx in range(12):
            doc = Document(title=f"Paper {idx + 1}")
            session.add(doc)
            session.commit()
            session.refresh(doc)
            assert doc.id is not None
            doc_ids.append(doc.id)
            report = Report(
                document_id=doc.id,
                payload=json.dumps({"schema_version": 2, "executive_summary": f"Summary {idx + 1}"}),
                created_at=now + timedelta(minutes=idx),
            )
            session.add(report)
            session.commit()

        # Save one old report; saved export should survive even if live document is pruned.
        old_saved_doc_id = doc_ids[0]
        saved_path = save_report_export(session, old_saved_doc_id)
        assert saved_path.exists()

        retention = enforce_report_retention(session, keep_latest=99)
        assert retention["keep_latest"] == 10
        assert retention["tracked_reports"] == 12
        assert retention["pruned_count"] == 2

        remaining_doc_ids = session.exec(select(Document.id)).all()
        assert len(remaining_doc_ids) == 10
        assert old_saved_doc_id not in remaining_doc_ids

    # Saved copy remains available in saved_reports even after live document prune.
    assert saved_reports_dir().exists()
    assert saved_path.exists()


def test_saved_report_uses_friendly_naming_and_writes_html_markdown(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "data_dir", Path(tmp_path / "data"))
    settings.data_dir.mkdir(parents=True, exist_ok=True)

    db_path = tmp_path / "retention_labels.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        doc = Document(title="Common Dimensional Reward Deficits Across Mood and Psychotic Disorders")
        session.add(doc)
        session.commit()
        session.refresh(doc)
        assert doc.id is not None

        payload = {
            "schema_version": 2,
            "paper_meta": {
                "title": "Common Dimensional Reward Deficits Across Mood and Psychotic Disorders",
                "journal": "American Journal of Psychiatry",
                "date": "2017-07-01",
            },
            "executive_summary": "Summary text.",
            "sections": {"results": {"focus": "Results focus", "items": []}},
        }
        session.add(Report(document_id=doc.id, payload=json.dumps(payload)))
        session.commit()

        saved_path = save_report_export(session, doc.id)

    assert saved_path.exists()
    assert "common_dimensional_reward_deficits_across" in saved_path.as_posix()
    assert "american_journal_of_2017" in saved_path.name
    assert saved_path.with_suffix(".md").exists()
    assert saved_path.with_suffix(".html").exists()
    assert "Results focus" not in saved_path.with_suffix(".md").read_text()
    assert "Results focus" not in saved_path.with_suffix(".html").read_text()
