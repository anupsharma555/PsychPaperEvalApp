from __future__ import annotations

import json
from pathlib import Path
import zipfile

import pytest
from sqlmodel import Session, SQLModel, create_engine, select

from app.db.models import Asset, Chunk, Document, SourceManifest
from app.services import parser
from app.services.parser import parse_document_assets
from app.services.source_manifest import record_source_manifest


def _session(tmp_path: Path) -> Session:
    engine = create_engine(
        f"sqlite:///{tmp_path / 'test.db'}",
        connect_args={"check_same_thread": False},
    )
    SQLModel.metadata.create_all(engine)
    return Session(engine)


def test_record_source_manifest_persists_table_and_artifact(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(parser.settings, "data_dir", tmp_path / "data")
    session = _session(tmp_path)
    try:
        document = Document(source_url="https://example.org/paper", doi="10.1000/test")
        session.add(document)
        session.commit()
        session.refresh(document)

        record_source_manifest(
            session,
            document.id,
            source_type="url",
            status="resolved",
            payload={"selected_assets": [{"filename": "main.pdf"}]},
        )
        session.commit()

        rows = session.exec(select(SourceManifest).where(SourceManifest.document_id == document.id)).all()
        assert len(rows) == 1
        payload = json.loads(rows[0].payload)
        assert payload["schema_version"] == 1
        assert payload["selected_assets"][0]["filename"] == "main.pdf"

        artifact = tmp_path / "data" / f"doc_{document.id}" / "artifacts" / "source_manifest.json"
        assert artifact.exists()
        assert json.loads(artifact.read_text())["status"] == "resolved"
    finally:
        session.close()


def test_parse_document_assets_writes_per_asset_diagnostics(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(parser.settings, "data_dir", tmp_path / "data")
    session = _session(tmp_path)
    try:
        document = Document(source_url=None)
        session.add(document)
        session.commit()
        session.refresh(document)

        asset_dir = tmp_path / "assets"
        asset_dir.mkdir()
        text_path = asset_dir / "paper.txt"
        text_path.write_text("Methods: Participants completed structured symptom ratings.", encoding="utf-8")
        asset = Asset(
            document_id=document.id,
            kind="main",
            filename=text_path.name,
            content_type="text/plain",
            path=str(text_path),
        )
        session.add(asset)
        session.commit()

        counts = parse_document_assets(session, document.id)
        assert counts["text"] == 1

        diag_path = tmp_path / "data" / f"doc_{document.id}" / "artifacts" / "parser_asset_diagnostics.json"
        payload = json.loads(diag_path.read_text())
        assert payload["assets"][0]["status"] == "parsed"
        assert payload["assets"][0]["sniffed_kind"] == "text"
        assert payload["assets"][0]["counts_delta"]["text"] == 1
    finally:
        session.close()


def test_parse_document_assets_is_idempotent_for_chunks(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(parser.settings, "data_dir", tmp_path / "data")
    session = _session(tmp_path)
    try:
        document = Document(source_url=None)
        session.add(document)
        session.commit()
        session.refresh(document)

        asset_dir = tmp_path / "assets"
        asset_dir.mkdir()
        text_path = asset_dir / "paper.txt"
        text_path.write_text("Results: The primary analysis reported improved symptom ratings.", encoding="utf-8")
        asset = Asset(
            document_id=document.id,
            kind="main",
            filename=text_path.name,
            content_type="text/plain",
            path=str(text_path),
        )
        session.add(asset)
        session.commit()

        parse_document_assets(session, document.id)
        parse_document_assets(session, document.id)

        chunks = session.exec(select(Chunk).where(Chunk.document_id == document.id)).all()
        assert len(chunks) == 1
        assert chunks[0].modality == "text"
    finally:
        session.close()


def test_zip_member_limit_blocks_main_archive(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(parser.settings, "archive_max_members", 1)
    zip_path = tmp_path / "too_many.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("a.txt", "one")
        archive.writestr("b.txt", "two")

    asset = Asset(document_id=1, kind="main", filename=zip_path.name, path=str(zip_path))
    with pytest.raises(ValueError, match="member count exceeds limit"):
        parser._parse_zip_file(  # noqa: SLF001 - intentional parser guardrail test
            session=None,
            document_id=1,
            asset=asset,
            path=zip_path,
            counts={"text": 0, "table": 0, "figure": 0, "supp": 0},
        )
