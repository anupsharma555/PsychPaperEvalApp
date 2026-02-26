from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Optional

from desktop_legacy.models import ValidationResult


DOI_RE = re.compile(r"10\.\d{4,9}/\S+", re.IGNORECASE)
URL_RE = re.compile(r"^https?://", re.IGNORECASE)
SUPP_ALLOWED = {".pdf", ".zip", ".csv", ".tsv", ".xlsx", ".xls", ".txt", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


class WorkflowStep:
    SELECT_SOURCE = 1
    VALIDATE_INPUT = 2
    SUBMIT_JOB = 3
    MONITOR = 4
    REVIEW = 5


@dataclass
class WorkflowState:
    step: int = WorkflowStep.SELECT_SOURCE
    source_mode: str = "url"
    validated: bool = False
    validation_message: str = "Choose a source and validate input."
    selected_job_id: Optional[int] = None
    selected_document_id: Optional[int] = None
    report_ready: bool = False
    last_events: list[str] = field(default_factory=list)

    def step_label(self, step: int) -> str:
        state = "Pending"
        if self.step > step:
            state = "Done"
        elif self.step == step:
            state = "Active"
        return state

    def mark_event(self, message: str) -> None:
        self.last_events.append(message)
        self.last_events = self.last_events[-20:]

    def set_source_mode(self, mode: str) -> None:
        self.source_mode = mode
        self.validated = False
        self.report_ready = False
        self.step = WorkflowStep.SELECT_SOURCE

    def set_validation(self, result: ValidationResult) -> None:
        self.validated = result.valid
        self.validation_message = result.message
        self.step = WorkflowStep.VALIDATE_INPUT if not result.valid else WorkflowStep.SUBMIT_JOB

    def on_job_submitted(self, job_id: int, document_id: int) -> None:
        self.selected_job_id = job_id
        self.selected_document_id = document_id
        self.step = WorkflowStep.MONITOR
        self.report_ready = False
        self.mark_event(f"Queued job {job_id} for document {document_id}.")

    def on_job_status(self, status: str) -> None:
        if status == "completed":
            self.step = WorkflowStep.REVIEW
            self.report_ready = True
        elif status == "failed":
            self.report_ready = False
            self.step = WorkflowStep.MONITOR

    def can_submit(self) -> bool:
        return self.validated and self.step >= WorkflowStep.SUBMIT_JOB

    def can_load_report(self, job_status: str | None) -> bool:
        return bool(self.selected_document_id and job_status == "completed")


def validate_source_input(
    source_mode: str,
    *,
    url_text: str,
    doi_text: str,
    main_file: str | None,
    supplement_files: list[str] | None,
) -> ValidationResult:
    supplements = supplement_files or []
    cleaned_doi = normalize_doi(doi_text)

    if source_mode == "url":
        normalized_url = url_text.strip()
        if not normalized_url and cleaned_doi:
            normalized_url = f"https://doi.org/{cleaned_doi}"
        if not normalized_url:
            return ValidationResult(False, "URL or DOI is required before validation.")
        if not URL_RE.search(normalized_url):
            return ValidationResult(False, "URL must start with http:// or https://.")
        return ValidationResult(True, "Input validated. You can submit analysis.", normalized_url=normalized_url, normalized_doi=cleaned_doi)

    if not main_file:
        return ValidationResult(False, "Main PDF is required. Choose a file in Step 1.")
    if not main_file.lower().endswith(".pdf"):
        return ValidationResult(False, "Main file must be a PDF.")
    for path in supplements:
        suffix = path.lower().rsplit(".", 1)
        ext = "." + suffix[-1] if len(suffix) > 1 else ""
        if ext not in SUPP_ALLOWED:
            return ValidationResult(False, f"Unsupported supplement type: {ext or 'unknown'}")
    return ValidationResult(True, "Input validated. You can submit analysis.")


def normalize_doi(value: str) -> Optional[str]:
    match = DOI_RE.search((value or "").strip())
    if not match:
        return None
    return match.group(0)
