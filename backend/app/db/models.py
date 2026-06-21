from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
import json
from typing import Optional

from sqlmodel import Field, SQLModel


def utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"


class Document(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    title: Optional[str] = None
    source_url: Optional[str] = None
    doi: Optional[str] = None
    created_at: datetime = Field(default_factory=utc_now)


class Asset(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    document_id: int = Field(index=True)
    kind: str = Field(index=True)  # main | supp
    filename: str
    content_type: Optional[str] = None
    path: str
    created_at: datetime = Field(default_factory=utc_now)


class Job(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    document_id: int = Field(index=True)
    status: JobStatus = Field(default=JobStatus.queued)
    progress: float = 0.0
    message: Optional[str] = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class Chunk(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    document_id: int = Field(index=True)
    asset_id: Optional[int] = Field(default=None, index=True)
    anchor: str = Field(index=True)  # page/section/figure/table/supp identifiers
    modality: str = Field(index=True)  # text|figure|table|supp
    content: str  # text or json string for structured data
    meta: Optional[str] = None


class SourceManifest(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    document_id: int = Field(index=True)
    source_type: str = Field(index=True)  # url|upload
    status: str = Field(index=True)  # resolved|uploaded|failed
    payload: str
    created_at: datetime = Field(default_factory=utc_now)


class Finding(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    document_id: int = Field(index=True)
    category: str = Field(index=True)
    summary: str
    evidence: str  # json string of anchors
    confidence: float = 0.0


class Discrepancy(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    document_id: int = Field(index=True)
    claim: str
    evidence: str  # json string with anchors and comparison
    severity: str = "medium"
    confidence: float = 0.0


class Report(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    document_id: int = Field(index=True)
    payload: str  # json string for executive summary + scores
    created_at: datetime = Field(default_factory=utc_now)

    @property
    def schema_version(self) -> int:
        try:
            parsed = json.loads(self.payload or "{}")
        except Exception:
            return 1
        version = parsed.get("schema_version", 1)
        try:
            return int(version)
        except Exception:
            return 1
