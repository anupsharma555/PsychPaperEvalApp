from __future__ import annotations

from pathlib import Path

from app.core.config import settings


def document_dir(document_id: int) -> Path:
    return settings.data_dir / f"doc_{document_id}"


def assets_dir(document_id: int) -> Path:
    return document_dir(document_id) / "assets"


def artifacts_dir(document_id: int) -> Path:
    return document_dir(document_id) / "artifacts"


def ensure_document_dirs(document_id: int) -> None:
    assets_dir(document_id).mkdir(parents=True, exist_ok=True)
    artifacts_dir(document_id).mkdir(parents=True, exist_ok=True)


def asset_path(document_id: int, filename: str) -> Path:
    return assets_dir(document_id) / filename
