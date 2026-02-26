from __future__ import annotations

import json
import mimetypes
from pathlib import Path
import time
from typing import Any, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request

from desktop_legacy.models import BackendStatus, DesktopReport, JobRow, ReportSummaryCard


class DesktopApiError(RuntimeError):
    pass


class DesktopApiClient:
    def __init__(self, api_base: str, *, timeout_sec: float = 15.0) -> None:
        self.api_base = api_base.rstrip("/")
        self.timeout_sec = timeout_sec

    def get_status(self) -> BackendStatus:
        data = self._request_json("/status", method="GET")
        processing = data.get("processing", {}) if isinstance(data, dict) else {}
        return BackendStatus(
            backend_ready=bool(data.get("backend_ready")),
            processing_paused=bool(processing.get("paused")),
            processing_running=bool(processing.get("running")),
            inflight=int(processing.get("inflight", 0) or 0),
            worker_capacity=int(processing.get("worker_capacity", 0) or 0),
            model_exists=bool(data.get("model_exists")),
            mmproj_exists=bool(data.get("mmproj_exists")),
            stale_jobs_recovered=int(data.get("stale_jobs_recovered", 0) or 0),
            orphan_cleanup_last_run=data.get("orphan_cleanup_last_run"),
            last_error=data.get("last_error"),
            last_recovery_at=data.get("last_recovery_at"),
        )

    def list_jobs(self, *, status: Optional[str] = None, limit: int = 100, offset: int = 0) -> list[JobRow]:
        query = f"?limit={limit}&offset={offset}"
        if status:
            query += f"&status={status}"
        data = self._request_json(f"/jobs{query}", method="GET")
        rows = []
        for item in data.get("items", []):
            rows.append(
                JobRow(
                    job_id=int(item.get("job_id")),
                    document_id=int(item.get("document_id")),
                    source_kind=str(item.get("source_kind", "upload")),
                    status=str(item.get("status", "queued")),
                    progress=float(item.get("progress", 0.0) or 0.0),
                    message=str(item.get("message", "") or ""),
                    updated_at=str(item.get("updated_at", "")),
                    created_at=str(item.get("created_at", "")),
                )
            )
        return rows

    def get_job(self, job_id: int) -> dict:
        return self._request_json(f"/jobs/{job_id}", method="GET")

    def create_from_url(self, url: str, doi: Optional[str] = None, fetch_supplements: bool = True) -> dict:
        payload = {"url": url, "doi": doi, "fetch_supplements": fetch_supplements}
        return self._request_json("/documents/from-url", method="POST", payload=payload, timeout=45.0)

    def create_from_upload(self, main_file: Path, supp_files: list[Path]) -> dict:
        parts: list[tuple[str, Path]] = [("main_file", main_file)]
        for item in supp_files:
            parts.append(("supp_files", item))
        return self._request_multipart("/documents/upload", file_parts=parts, timeout=180.0)

    def get_report_summary(self, document_id: int) -> DesktopReport:
        data = self._request_json(f"/documents/{document_id}/report/summary", method="GET")
        cards = []
        for item in data.get("modality_cards", []):
            cards.append(
                ReportSummaryCard(
                    modality=str(item.get("modality", "unknown")),
                    highlights=item.get("highlights", []) if isinstance(item.get("highlights"), list) else [],
                    finding_count=int(item.get("finding_count", 0) or 0),
                    coverage_gaps=item.get("coverage_gaps", []) if isinstance(item.get("coverage_gaps"), list) else [],
                )
            )
        return DesktopReport(
            document_id=int(data.get("document_id", document_id)),
            summary_version=int(data.get("summary_version", 0) or 0),
            report_status=str(data.get("report_status", "not_ready")),
            executive_summary=data.get("executive_summary"),
            modality_cards=cards,
            discrepancy_count=int(data.get("discrepancy_count", 0) or 0),
            overall_confidence=data.get("overall_confidence"),
            export_url=data.get("export_url"),
        )

    def get_report(self, document_id: int) -> dict:
        return self._request_json(f"/documents/{document_id}/report", method="GET")

    def get_runtime_events(self, limit: int = 25) -> list[dict]:
        data = self._request_json(f"/runtime/events?limit={limit}", method="GET")
        return data.get("events", []) if isinstance(data, dict) else []

    def processing_pause(self) -> dict:
        return self._request_json("/processing/stop", method="POST", payload={})

    def processing_resume(self) -> dict:
        return self._request_json("/processing/start", method="POST", payload={})

    def processing_cleanup(self) -> dict:
        return self._request_json("/processing/cleanup", method="POST", payload={})

    def processing_recover(self) -> dict:
        return self._request_json("/processing/recover", method="POST", payload={})

    def stop_app(self) -> dict:
        return self._request_json("/stop", method="POST", payload={}, timeout=8.0)

    def _request_json(
        self,
        path: str,
        *,
        method: str = "GET",
        payload: Optional[dict] = None,
        timeout: Optional[float] = None,
    ) -> dict:
        retries = 2
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                return self._request_json_once(path, method=method, payload=payload, timeout=timeout)
            except Exception as exc:
                last_error = exc
                if attempt >= retries:
                    break
                time.sleep(0.2 * (attempt + 1))
        raise DesktopApiError(str(last_error) if last_error else "Unknown API error")

    def _request_json_once(
        self,
        path: str,
        *,
        method: str,
        payload: Optional[dict],
        timeout: Optional[float],
    ) -> dict:
        url = f"{self.api_base}{path}"
        headers = {"Accept": "application/json"}
        data = None
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib_request.Request(url, data=data, headers=headers, method=method)
        effective_timeout = timeout if timeout is not None else self.timeout_sec
        try:
            with urllib_request.urlopen(req, timeout=effective_timeout) as response:
                body = response.read().decode("utf-8")
        except urllib_error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            raise DesktopApiError(f"HTTP {exc.code}: {body}") from exc
        except Exception as exc:
            raise DesktopApiError(str(exc)) from exc
        if not body:
            return {}
        try:
            return json.loads(body)
        except Exception:
            return {"raw": body}

    def _request_multipart(self, path: str, *, file_parts: list[tuple[str, Path]], timeout: float) -> dict:
        boundary = f"----PaperEvalBoundary{int(time.time() * 1000)}"
        body = self._encode_multipart(file_parts=file_parts, boundary=boundary)
        headers = {
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Accept": "application/json",
        }
        req = urllib_request.Request(f"{self.api_base}{path}", data=body, headers=headers, method="POST")
        try:
            with urllib_request.urlopen(req, timeout=timeout) as response:
                content = response.read().decode("utf-8")
        except urllib_error.HTTPError as exc:
            content = exc.read().decode("utf-8", errors="ignore")
            raise DesktopApiError(f"HTTP {exc.code}: {content}") from exc
        except Exception as exc:
            raise DesktopApiError(str(exc)) from exc
        if not content:
            return {}
        return json.loads(content)

    def _encode_multipart(self, *, file_parts: list[tuple[str, Path]], boundary: str) -> bytes:
        lines: list[bytes] = []
        for field_name, file_path in file_parts:
            filename = file_path.name
            mime_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
            lines.append(f"--{boundary}".encode("utf-8"))
            lines.append(
                f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"'.encode("utf-8")
            )
            lines.append(f"Content-Type: {mime_type}".encode("utf-8"))
            lines.append(b"")
            lines.append(file_path.read_bytes())
        lines.append(f"--{boundary}--".encode("utf-8"))
        lines.append(b"")
        return b"\r\n".join(lines)
