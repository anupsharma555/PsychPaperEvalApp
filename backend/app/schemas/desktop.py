from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _SchemaModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=())


class ApiRemediation(_SchemaModel):
    error_code: str
    user_message: str
    next_action: str


class DesktopRuntimeEvent(_SchemaModel):
    event_id: str
    timestamp: datetime | str
    kind: str
    severity: Literal["info", "warning", "error"] = "info"
    message: str
    context: dict[str, Any] = Field(default_factory=dict)


class DesktopJobRow(_SchemaModel):
    job_id: int
    document_id: int
    source_kind: Literal["url", "upload"]
    status: str
    progress: float
    message: str = ""
    created_at: datetime
    updated_at: datetime


class DesktopModelReadiness(_SchemaModel):
    model_path: str
    mmproj_path: str
    model_exists: bool
    mmproj_exists: bool
    text_model_path: str = ""
    text_model_exists: bool = False
    deep_model_path: str = ""
    deep_model_exists: bool = False
    vision_model_path: str = ""
    vision_model_exists: bool = False
    vision_mmproj_path: str = ""
    vision_mmproj_exists: bool = False


class DesktopProcessingStatus(_SchemaModel):
    running: bool
    paused: bool
    inflight: int
    worker_capacity: int
    stale_jobs_recovered: int = 0
    orphan_cleanup_last_run: dict[str, Any] | None = None
    last_recovery_at: str | None = None
    last_error: str | None = None


class DesktopLifecycleStatus(_SchemaModel):
    event_count: int
    latest_event: DesktopRuntimeEvent | None = None


class DesktopBootstrap(_SchemaModel):
    backend_ready: bool
    processing: DesktopProcessingStatus
    models: DesktopModelReadiness
    lifecycle: DesktopLifecycleStatus
    latest_jobs: list[DesktopJobRow] = Field(default_factory=list)


class DesktopModalityCard(_SchemaModel):
    modality: str
    highlights: list[str] = Field(default_factory=list)
    finding_count: int = 0
    coverage_gaps: list[str] = Field(default_factory=list)


class DesktopReportSummary(_SchemaModel):
    document_id: int
    summary_version: int
    report_status: Literal["ready", "not_ready", "failed"] = "not_ready"
    executive_summary: str | None = None
    modality_cards: list[DesktopModalityCard] = Field(default_factory=list)
    methods_card: list[str] = Field(default_factory=list)
    sections_card: list[str] = Field(default_factory=list)
    rerun_recommended: bool = False
    report_capabilities: dict[str, bool] = Field(default_factory=dict)
    discrepancy_count: int = 0
    overall_confidence: float | None = None
    export_url: str
    saved: bool = False
