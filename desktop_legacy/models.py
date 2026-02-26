from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class BackendStatus:
    backend_ready: bool
    processing_paused: bool
    processing_running: bool
    inflight: int
    worker_capacity: int
    model_exists: bool
    mmproj_exists: bool
    stale_jobs_recovered: int
    orphan_cleanup_last_run: dict | None
    last_error: Optional[str]
    last_recovery_at: Optional[str]


@dataclass
class JobRow:
    job_id: int
    document_id: int
    source_kind: str
    status: str
    progress: float
    message: str
    updated_at: str
    created_at: str


@dataclass
class ReportSummaryCard:
    modality: str
    highlights: list[str] = field(default_factory=list)
    finding_count: int = 0
    coverage_gaps: list[str] = field(default_factory=list)


@dataclass
class DesktopReport:
    document_id: int
    summary_version: int
    report_status: str
    executive_summary: Optional[str]
    modality_cards: list[ReportSummaryCard] = field(default_factory=list)
    discrepancy_count: int = 0
    overall_confidence: Optional[float] = None
    export_url: Optional[str] = None


@dataclass
class ValidationResult:
    valid: bool
    message: str
    normalized_url: Optional[str] = None
    normalized_doi: Optional[str] = None


@dataclass
class DesktopRuntimeInfo:
    ui_python_path: str
    backend_python_path: str
    desktop_instance_pid: int
    backend_pid: Optional[int]
    startup_state: str
    last_start_error: Optional[str]


def now_ts() -> str:
    return datetime.utcnow().isoformat()
