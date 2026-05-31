from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

from sqlmodel import Session

from app.db.models import SourceManifest
from app.services.storage import artifacts_dir


def record_source_manifest(
    session: Session,
    document_id: int,
    *,
    source_type: str,
    status: str,
    payload: dict[str, Any],
) -> SourceManifest:
    normalized = _json_safe(
        {
            "schema_version": 1,
            "document_id": document_id,
            "source_type": source_type,
            "status": status,
            "created_at": datetime.utcnow().isoformat(),
            **payload,
        }
    )
    manifest = SourceManifest(
        document_id=document_id,
        source_type=source_type,
        status=status,
        payload=json.dumps(normalized, ensure_ascii=True, sort_keys=True),
    )
    session.add(manifest)
    _write_manifest_artifact(document_id, normalized)
    return manifest


def _write_manifest_artifact(document_id: int, payload: dict[str, Any]) -> None:
    try:
        path = artifacts_dir(document_id) / "source_manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    except Exception:
        return


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
