from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urljoin, urlparse, urlunparse

import httpx
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, RedirectResponse, Response
from pydantic import BaseModel
from sqlmodel import Session, select

from app.core.config import ROOT_DIR, settings
from app.db.models import Asset, Chunk, Discrepancy, Document, Finding, Job, JobStatus, Report
from app.db.session import get_session
from app.schemas.desktop import (
    DesktopBootstrap,
    DesktopJobRow,
    DesktopLifecycleStatus,
    DesktopModelReadiness,
    DesktopProcessingStatus,
    DesktopReportSummary,
    DesktopRuntimeEvent,
)
from app.services.ingest import ingest_upload, ingest_url
from app.services.jobs import enqueue_job, job_runner
from app.services.author_utils import sanitize_author_list
from app.services.report_retention import (
    is_document_saved,
    list_saved_reports,
    save_report_export,
    saved_status,
)
from app.services.runtime import read_pids, read_runtime_events, stop_app
from app.services.storage import artifacts_dir, document_dir

router = APIRouter(prefix="/api")


_RUNTIME_EVENT_SEVERITY = {
    "startup_failed": "error",
    "backend_shutdown": "warning",
    "stop_requested": "warning",
    "stop_completed": "info",
    "backend_startup": "info",
    "shutdown_requested": "warning",
    "shutdown_completed": "info",
}

SUPPLEMENT_MARKER_RE = re.compile(
    r"\b("
    r"supplement(?:ary|al)?|suppl|appendix|extended data|supporting (?:information|info|data)|"
    r"figure\s*s\d+|fig(?:ure)?\s*s\d+|table\s*s\d+|s\d+\s*(?:fig(?:ure)?|table)"
    r")\b",
    flags=re.IGNORECASE,
)

CONFIDENCE_TAG_RE = re.compile(
    r"\s*(?:\((?:model\s+)?confidence[:\s]*\d{1,3}(?:\.\d+)?%?\)|(?:model\s+)?confidence[:\s]+\d{1,3}(?:\.\d+)?%|\(\d{1,3}(?:\.\d+)?%\))\s*",
    flags=re.IGNORECASE,
)
FIGURE_ORDER_RE = re.compile(r"(?:\bfig(?:ure)?\b|^f)\s*[_:\s-]*(s?\d+)([a-z]?)", flags=re.IGNORECASE)
TABLE_ORDER_RE = re.compile(r"\btable\b\s*[_:\s-]*(s?\d+)([a-z]?)", flags=re.IGNORECASE)
FIGURE_TOKEN_RE = re.compile(r"(?:^|[^a-z0-9])f(?:ig(?:ure)?)?[_:\s-]*(s?\d+[a-z]?)(?:[^a-z0-9]|$)", flags=re.IGNORECASE)
FIGURE_FILENAME_TOKEN_RE = re.compile(r"f(?:ig(?:ure)?)?[_:\s-]?(s?\d+[a-z]?)(?:[^a-z0-9]|$)", flags=re.IGNORECASE)
FIGURE_TEXT_REF_RE = re.compile(r"\bfig(?:ure)?\.?\s*(s?\d+)([a-z]?)\b", flags=re.IGNORECASE)
FALLBACK_ID_CONTEXT_RE = re.compile(r"\s*\(id:text-fallback-[^)]+\)\s*", flags=re.IGNORECASE)
FALLBACK_ID_BRACKET_RE = re.compile(r"\s*\[id:text-fallback-[^\]]+\]\s*", flags=re.IGNORECASE)
FALLBACK_ID_TOKEN_RE = re.compile(r"\bid:text-fallback-[a-z0-9_.:-]+\b", flags=re.IGNORECASE)
STATEMENT_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
LEGEND_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
LEGEND_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")
LEGEND_STOPWORD_RE = re.compile(r"\b(the|and|of|in|to|with|for|that|this|was|were|from|across|between)\b", flags=re.IGNORECASE)
LEGEND_VERB_RE = re.compile(
    r"\b(show|shows|showed|indicate|indicates|indicated|reveal|reveals|revealed|demonstrate|demonstrates|demonstrated|identify|identifies|identified|compare|compares|compared|was|were|is|are)\b",
    flags=re.IGNORECASE,
)
LEGEND_LINK_VERB_RE = re.compile(
    r"\b(show|shows|showed|indicate|indicates|indicated|reveal|reveals|revealed|demonstrate|demonstrates|demonstrated|identify|identifies|identified|associated|correlated|difference|differences|pattern|patterns|characteriz(?:e|ed|es|ing)|relat(?:ed|es|ing)|correspond(?:ed|ence|ing)?|validate(?:d|s|ion)?)\b",
    flags=re.IGNORECASE,
)
GENERIC_LEGEND_RE = re.compile(
    r"^(?:page\s+\d+\s+(?:raster(?:\s+fallback)?|image(?:\s+fallback)?)|supplementary material|image)$",
    flags=re.IGNORECASE,
)
MEDIA_LEGEND_MAX_CHARS = max(1200, int(getattr(settings, "media_legend_max_chars", 6000) or 6000))
OCR_LEGEND_MAX_CHARS = max(900, int(getattr(settings, "media_ocr_legend_max_chars", 1800) or 1800))


def _safe_git_output(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT_DIR), *args],
            capture_output=True,
            text=True,
            timeout=1.5,
            check=False,
        )
    except Exception:
        return ""
    if result.returncode != 0:
        return ""
    return str(result.stdout or "").strip()


def _load_build_manifest() -> dict[str, Any]:
    path = ROOT_DIR / ".run" / "build_manifest.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


@lru_cache(maxsize=1)
def _runtime_build_info() -> dict[str, Any]:
    version = ""
    try:
        version = (ROOT_DIR / "VERSION").read_text(encoding="utf-8").strip()
    except Exception:
        version = ""

    manifest = _load_build_manifest()
    manifest_git_sha = str(manifest.get("git_sha") or "").strip()
    git_commit = manifest_git_sha[:12] if manifest_git_sha else _safe_git_output("rev-parse", "--short=12", "HEAD")
    git_dirty = bool(_safe_git_output("status", "--porcelain", "--untracked-files=no"))
    build_timestamp = str(manifest.get("built_at") or "").strip()

    marker_parts: list[str] = []
    if version:
        marker_parts.append(f"v{version}")
    marker_parts.append(git_commit or "no-git")
    if git_dirty:
        marker_parts.append("dirty")

    return {
        "app_version": version,
        "git_commit": git_commit or "unknown",
        "git_dirty": git_dirty,
        "build_timestamp": build_timestamp,
        "source_root": str(ROOT_DIR),
        "build_marker": "-".join(marker_parts),
    }


def _raise_bad_request(error_code: str, user_message: str, next_action: str) -> None:
    raise HTTPException(
        status_code=400,
        detail={
            "error_code": error_code,
            "user_message": user_message,
            "next_action": next_action,
        },
    )


def _strip_upload_guidance(user_message: str) -> str:
    message = str(user_message or "").strip()
    if not message:
        return ""
    patterns = [
        r"use from upload and choose the main pdf\.?",
        r"use from upload, choose the main pdf, then submit again\.?",
        r"retry later, or use from upload with the main pdf\.?",
    ]
    for pattern in patterns:
        message = re.sub(pattern, "", message, flags=re.IGNORECASE).strip()
    message = re.sub(r"\s{2,}", " ", message).strip()
    if message.endswith(":"):
        message = message[:-1].strip()
    return message


def _parse_iso_timestamp(value: str) -> datetime:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(normalized)
    except Exception:
        _raise_bad_request(
            "invalid_since",
            "Invalid since timestamp. Use ISO-8601 format.",
            "Retry with a value like 2026-02-09T14:00:00Z.",
        )


def _to_desktop_runtime_event(raw: dict[str, Any], index: int) -> DesktopRuntimeEvent:
    timestamp = raw.get("timestamp") or datetime.utcnow().isoformat()
    kind = str(raw.get("event", "event"))
    details = raw.get("details") if isinstance(raw.get("details"), dict) else {}
    message = str(details.get("message") or kind.replace("_", " ").title())
    severity = _RUNTIME_EVENT_SEVERITY.get(kind, "info")
    event_id = f"{timestamp}:{kind}:{index}"
    return DesktopRuntimeEvent(
        event_id=event_id,
        timestamp=timestamp,
        kind=kind,
        severity=severity,
        message=message,
        context=details,
    )


def _load_runtime_events(*, limit: int, since: str | None = None) -> list[DesktopRuntimeEvent]:
    raw_events = read_runtime_events(limit=max(limit * 4, limit))
    desktop_events: list[DesktopRuntimeEvent] = []
    since_dt = _parse_iso_timestamp(since) if since else None

    for idx, raw in enumerate(raw_events):
        if not isinstance(raw, dict):
            continue
        event = _to_desktop_runtime_event(raw, idx)
        if since_dt is not None:
            try:
                event_dt = _parse_iso_timestamp(str(event.timestamp))
            except HTTPException:
                continue
            if event_dt <= since_dt:
                continue
        desktop_events.append(event)

    if len(desktop_events) > limit:
        return desktop_events[-limit:]
    return desktop_events


def _job_sort_clause(sort: str):
    value = (sort or "updated_at:desc").strip()
    if not value:
        value = "updated_at:desc"

    if ":" in value:
        field_name, direction = value.split(":", 1)
    else:
        field_name, direction = value, "desc"

    field_name = field_name.strip().lower()
    direction = direction.strip().lower()

    allowed_fields = {
        "updated_at": Job.updated_at,
        "created_at": Job.created_at,
        "progress": Job.progress,
    }
    if field_name not in allowed_fields:
        _raise_bad_request(
            "invalid_sort",
            "Unsupported sort field.",
            "Use sort=updated_at:desc, created_at:asc, or progress:desc.",
        )
    if direction not in {"asc", "desc"}:
        _raise_bad_request(
            "invalid_sort_direction",
            "Unsupported sort direction.",
            "Use :asc or :desc.",
        )

    clause = allowed_fields[field_name]
    if direction == "desc":
        clause = clause.desc()
    else:
        clause = clause.asc()

    return clause, f"{field_name}:{direction}"


def _job_status_filter(status: str | None) -> JobStatus | None:
    if status is None:
        return None
    try:
        return JobStatus(status)
    except Exception:
        _raise_bad_request(
            "invalid_status",
            "Invalid status filter.",
            "Use queued, running, completed, or failed.",
        )


def _job_source_kind(document: Document | None) -> str:
    return "url" if document and document.source_url else "upload"


def _load_job_rows(
    session: Session,
    *,
    status: JobStatus | None,
    limit: int,
    offset: int,
    sort: str,
) -> tuple[list[DesktopJobRow], str]:
    order_clause, normalized_sort = _job_sort_clause(sort)
    stmt = select(Job)
    if status is not None:
        stmt = stmt.where(Job.status == status)
    stmt = stmt.order_by(order_clause).offset(offset).limit(limit)

    jobs = session.exec(stmt).all()
    if not jobs:
        return [], normalized_sort

    doc_ids = sorted({job.document_id for job in jobs if job.document_id is not None})
    docs = session.exec(select(Document).where(Document.id.in_(doc_ids))).all()
    doc_map = {doc.id: doc for doc in docs}

    rows: list[DesktopJobRow] = []
    for job in jobs:
        rows.append(
            DesktopJobRow(
                job_id=int(job.id),
                document_id=job.document_id,
                source_kind=_job_source_kind(doc_map.get(job.document_id)),
                status=job.status.value if isinstance(job.status, JobStatus) else str(job.status),
                progress=float(job.progress or 0.0),
                message=str(job.message or ""),
                created_at=_as_utc_datetime(job.created_at),
                updated_at=_as_utc_datetime(job.updated_at),
            )
        )
    return rows, normalized_sort


def _as_utc_datetime(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_chunk_meta(raw_meta: str | None) -> dict[str, Any]:
    if not raw_meta:
        return {}
    try:
        parsed = json.loads(raw_meta)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _safe_doc_media_path(document_id: int, raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    try:
        candidate = Path(str(raw_path)).expanduser().resolve()
    except Exception:
        return None
    doc_root = document_dir(document_id).resolve()
    try:
        candidate.relative_to(doc_root)
    except ValueError:
        return None
    if not candidate.exists() or not candidate.is_file():
        return None
    return candidate


def _resolve_chunk_image_path(document_id: int, chunk: Chunk, meta_obj: dict[str, Any]) -> Path | None:
    if chunk.modality == "figure":
        return _safe_doc_media_path(document_id, str(meta_obj.get("path") or ""))
    if chunk.modality != "table":
        return None

    for key in ("table_image_path", "path"):
        resolved = _safe_doc_media_path(document_id, str(meta_obj.get(key) or ""))
        if resolved is not None:
            return resolved

    anchor = str(chunk.anchor or "")
    if anchor.startswith("table:"):
        stem = anchor.split("table:", 1)[1]
        inferred = document_dir(document_id) / "artifacts" / "pdffigures2" / f"{stem}.png"
        return _safe_doc_media_path(document_id, str(inferred))
    return None


def _remote_media_url(meta_obj: dict[str, Any]) -> str | None:
    for key in ("source_url", "image_url", "url", "href"):
        value = str(meta_obj.get(key) or "").strip()
        if value.startswith("http://") or value.startswith("https://"):
            parsed = urlparse(value)
            # Some DOI-hosted pages emit media URLs that should resolve on the publisher host.
            if parsed.netloc.lower() == "doi.org" and parsed.path.startswith("/cms/"):
                patched = parsed._replace(netloc="psychiatryonline.org")
                return urlunparse(patched._replace(scheme="https"))
            if parsed.netloc.lower() == "ajp.psychiatryonline.org":
                parsed = parsed._replace(netloc="psychiatryonline.org", scheme="https")
                return urlunparse(parsed)
            if parsed.netloc.lower().endswith("psychiatryonline.org") and parsed.scheme.lower() != "https":
                parsed = parsed._replace(scheme="https")
                return urlunparse(parsed)
            return value
    return None


def _looks_like_image_url(url: str | None) -> bool:
    value = str(url or "").lower().split("?", 1)[0]
    return value.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp", ".tif", ".tiff", ".svg"))


def _proxy_remote_with_redirect_guard(
    remote_url: str,
    *,
    headers: dict[str, str],
    timeout: int,
    max_redirects: int = 8,
) -> tuple[bytes | None, str | None, str]:
    current = str(remote_url or "").strip()
    visited: set[str] = {current.lower()}
    with httpx.Client(timeout=timeout, follow_redirects=False, headers=headers) as client:
        for _ in range(max_redirects + 1):
            response = client.get(current)
            if response.status_code in {301, 302, 303, 307, 308}:
                location = str(response.headers.get("location") or "").strip()
                if not location:
                    break
                next_url = urljoin(current, location)
                key = next_url.lower()
                if key in visited:
                    return None, None, next_url
                visited.add(key)
                current = next_url
                continue
            response.raise_for_status()
            content_type = response.headers.get("content-type", "application/octet-stream")
            return response.content, content_type, current
    return None, None, current


def _redirect_fallback_target(remote_url: str | None, exc: Exception | None = None) -> str | None:
    target = str(remote_url or "").strip()
    if not target:
        return None
    if isinstance(exc, httpx.TooManyRedirects):
        response = getattr(exc, "response", None)
        location = str(getattr(response, "headers", {}).get("location") or "").strip() if response is not None else ""
        if location:
            return urljoin(target, location)
    return target


def _normalize_media_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _normalize_media_legend(value: Any, *, max_chars: int = MEDIA_LEGEND_MAX_CHARS) -> str:
    text = str(value or "").replace("\x00", " ").strip()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(FALLBACK_ID_CONTEXT_RE, " ", text)
    text = re.sub(FALLBACK_ID_BRACKET_RE, " ", text)
    text = re.sub(FALLBACK_ID_TOKEN_RE, " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    trimmed = text[:max_chars]
    cut = trimmed.rfind(" ")
    if cut >= int(max_chars * 0.65):
        trimmed = trimmed[:cut]
    return f"{trimmed.rstrip(',;:.- ')}..."


def _is_generic_legend(value: str) -> bool:
    text = _normalize_media_text(value)
    if not text:
        return True
    if GENERIC_LEGEND_RE.match(text):
        return True
    if re.fullmatch(r"(?:fig(?:ure)?|table)[:\s_-]*[a-z0-9]+", text):
        return True
    if re.fullmatch(r"(?:figure|table):[a-z0-9:_-]+", text):
        return True
    return False


def _extract_legend_from_ocr(ocr_text: Any) -> str:
    def _looks_sentence_like_legend(value: str) -> bool:
        text = _normalize_media_legend(value, max_chars=OCR_LEGEND_MAX_CHARS)
        if not text:
            return False
        words = LEGEND_WORD_RE.findall(text)
        if len(words) < 10:
            return False
        if len(LEGEND_STOPWORD_RE.findall(text)) < 1:
            return False
        if not LEGEND_VERB_RE.search(text):
            return False
        if re.match(r"^\s*[A-H](?:\s*(?:and|,)\s*[A-H])?,\s+", text):
            return False
        return True

    raw = str(ocr_text or "").strip()
    if not raw:
        return ""
    lines = [_normalize_media_legend(line, max_chars=OCR_LEGEND_MAX_CHARS) for line in re.split(r"[\r\n]+", raw)]
    for line in lines:
        if not line or _is_generic_legend(line):
            continue
        words = LEGEND_WORD_RE.findall(line)
        if len(words) >= 8 and re.search(r"\bfig(?:ure)?\b", line, flags=re.IGNORECASE) and _looks_sentence_like_legend(line):
            return line
    for line in lines:
        if not line or _is_generic_legend(line):
            continue
        if _looks_sentence_like_legend(line):
            return line
    normalized = _normalize_media_legend(raw, max_chars=OCR_LEGEND_MAX_CHARS)
    for sentence in LEGEND_SENTENCE_SPLIT_RE.split(normalized):
        candidate = _normalize_media_legend(sentence, max_chars=OCR_LEGEND_MAX_CHARS)
        if not candidate or _is_generic_legend(candidate):
            continue
        if _looks_sentence_like_legend(candidate):
            return candidate
    return ""


def _media_legend(chunk: Chunk, meta_obj: dict[str, Any]) -> tuple[str, str]:
    candidate_fields = (
        ("legend", "legend"),
        ("legend_text", "legend"),
        ("description", "description"),
        ("alt_text", "alt_text"),
    )
    for field_name, source_name in candidate_fields:
        candidate = _normalize_media_legend(meta_obj.get(field_name))
        if not candidate:
            continue
        if _is_generic_legend(candidate):
            continue
        if field_name in {"description", "alt_text"}:
            words = LEGEND_WORD_RE.findall(candidate)
            if len(words) < 8 or not LEGEND_VERB_RE.search(candidate):
                continue
        return candidate, source_name

    content_candidate = _normalize_media_legend(chunk.content)
    if content_candidate and not _is_generic_legend(content_candidate):
        return content_candidate, "content"

    ocr_candidate = _extract_legend_from_ocr(meta_obj.get("ocr_text"))
    if ocr_candidate:
        return ocr_candidate, "ocr"
    return "", "missing"


def _normalize_figure_token(value: Any) -> str:
    raw = str(value or "").strip().upper()
    if not raw:
        return ""
    match = re.search(r"\b(S?)(\d+)([A-Z]?)\b", raw)
    if not match:
        return ""
    prefix = "S" if match.group(1) else ""
    number = str(int(match.group(2) or "0")) if str(match.group(2) or "").isdigit() else ""
    suffix = str(match.group(3) or "")
    if not number:
        return ""
    return f"{prefix}{number}{suffix}"


def _figure_token_from_chunk(chunk: Chunk, meta_obj: dict[str, Any]) -> str:
    candidates = [
        str(meta_obj.get("figure_id") or ""),
        str(meta_obj.get("caption") or ""),
        str(chunk.anchor or ""),
        str(meta_obj.get("source_url") or ""),
        str(meta_obj.get("path") or ""),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        for pattern in (FIGURE_ORDER_RE, FIGURE_TOKEN_RE):
            match = pattern.search(candidate)
            if not match:
                continue
            token = _normalize_figure_token(match.group(1) or "")
            if token:
                return token
        tail = Path(candidate).name
        filename_match = FIGURE_FILENAME_TOKEN_RE.search(tail)
        if filename_match:
            token = _normalize_figure_token(filename_match.group(1) or "")
            if token:
                return token
    return ""


def _figure_tokens_from_text(value: Any) -> list[str]:
    text = str(value or "")
    if not text:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for match in FIGURE_TEXT_REF_RE.finditer(text):
        base = _normalize_figure_token(match.group(1) or "")
        suffix = str(match.group(2) or "").upper()
        token = f"{base}{suffix}" if base else ""
        if base and base not in seen:
            seen.add(base)
            out.append(base)
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _legend_duplicates_caption(legend: str, caption: str) -> bool:
    normalized_legend = _normalize_media_text(legend)
    normalized_caption = _normalize_media_text(caption)
    if not normalized_legend or not normalized_caption:
        return False
    if normalized_legend == normalized_caption:
        return True
    if normalized_legend in normalized_caption or normalized_caption in normalized_legend:
        return True
    legend_tokens = set(STATEMENT_TOKEN_RE.findall(normalized_legend))
    caption_tokens = set(STATEMENT_TOKEN_RE.findall(normalized_caption))
    if not legend_tokens or not caption_tokens:
        return False
    overlap = len(legend_tokens & caption_tokens) / max(1, min(len(legend_tokens), len(caption_tokens)))
    return overlap >= 0.9


def _is_supplement_like_media(chunk: Chunk, meta_obj: dict[str, Any], asset_url: str | None) -> bool:
    source = _normalize_media_text(meta_obj.get("source"))
    caption = _normalize_media_text(meta_obj.get("caption"))
    anchor = _normalize_media_text(chunk.anchor)
    remote = _normalize_media_text(asset_url)
    figure_id = _normalize_media_text(meta_obj.get("figure_id"))
    table_id = _normalize_media_text(meta_obj.get("table_id"))
    marker_blob = " ".join([source, caption, anchor, remote, figure_id, table_id]).strip()

    if "/doi/suppl/" in remote or "suppl_file" in remote:
        return True
    if "supplement" in remote or "supplement" in caption:
        return True
    if "supplement" in anchor or "suppl" in anchor:
        return True
    if source == "html_link" and ("supplement" in caption or "suppl" in caption):
        return True
    if marker_blob and SUPPLEMENT_MARKER_RE.search(marker_blob):
        return True
    return False


def _media_signature(chunk: Chunk, meta_obj: dict[str, Any]) -> str:
    remote_url = _remote_media_url(meta_obj)
    if remote_url:
        parsed = urlparse(remote_url)
        normalized = urlunparse(parsed._replace(query="", fragment=""))
        return f"url:{normalized.lower()}"

    if chunk.modality == "figure":
        figure_id = _normalize_media_text(meta_obj.get("figure_id"))
        if figure_id:
            return f"figure_id:{figure_id}"
    if chunk.modality == "table":
        table_id = _normalize_media_text(meta_obj.get("table_id"))
        if table_id:
            return f"table_id:{table_id}"

    anchor = _normalize_media_text(chunk.anchor)
    if anchor:
        return f"anchor:{anchor}"
    caption = _normalize_media_text(meta_obj.get("caption"))
    if caption:
        return f"caption:{caption[:180]}"
    return f"chunk:{int(chunk.id) if chunk.id is not None else 'unknown'}"


def _strip_confidence_tag(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return " ".join(re.sub(CONFIDENCE_TAG_RE, " ", text).split()).strip()


def _strip_fallback_id_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(FALLBACK_ID_CONTEXT_RE, " ", text)
    text = re.sub(FALLBACK_ID_BRACKET_RE, " ", text)
    text = re.sub(FALLBACK_ID_TOKEN_RE, " ", text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    return " ".join(text.split()).strip()


def _is_evidence_ref_token(value: Any) -> bool:
    token = str(value or "").strip().lower()
    if not token:
        return False
    if token.startswith("id:"):
        return True
    if token.startswith("section:"):
        return True
    if token.startswith("figure:") or token.startswith("table:"):
        return True
    if re.fullmatch(r"[ft]\d+[a-z]?", token):
        return True
    if token.startswith("supp"):
        return True
    return False


def _sanitize_evidence_refs(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        ref = str(value or "").strip()
        if not ref:
            continue
        lowered = ref.lower()
        if lowered.startswith("id:"):
            continue
        if not _is_evidence_ref_token(ref):
            continue
        if lowered in seen:
            continue
        seen.add(lowered)
        out.append(ref)
    return out


def _statement_dedupe_key(value: Any) -> str:
    text = _strip_fallback_id_text(_strip_confidence_tag(value))
    if not text:
        return ""
    text = re.sub(
        r"\[([^\]]+)\]",
        lambda match: " " if _is_evidence_ref_token(match.group(1)) else match.group(0),
        text,
    )
    tokens = STATEMENT_TOKEN_RE.findall(text.lower().replace("...", " "))
    return " ".join(tokens).strip()


def _are_near_duplicate_statement_keys(left: str, right: str) -> bool:
    a = str(left or "").strip()
    b = str(right or "").strip()
    if not a or not b:
        return False
    if a == b:
        return True
    if a in b or b in a:
        return True
    a_tokens = [token for token in a.split(" ") if token]
    b_tokens = [token for token in b.split(" ") if token]
    if not a_tokens or not b_tokens:
        return False
    min_len = min(len(a_tokens), len(b_tokens))
    if min_len >= 10:
        prefix_matches = 0
        for idx in range(min_len):
            if a_tokens[idx] != b_tokens[idx]:
                break
            prefix_matches += 1
        if (prefix_matches / max(min_len, 1)) >= 0.86:
            return True
    a_set = set(a_tokens)
    b_set = set(b_tokens)
    intersection = len(a_set & b_set)
    if intersection <= 0:
        return False
    overlap_max = intersection / max(len(a_set), len(b_set), 1)
    overlap_min = intersection / max(min(len(a_set), len(b_set)), 1)
    return overlap_max >= 0.82 or (overlap_min >= 0.9 and overlap_max >= 0.56)


def _dedupe_statement_lines(lines: Any) -> list[str]:
    if not isinstance(lines, list):
        return []
    out: list[str] = []
    seen_keys: list[str] = []
    for value in lines:
        cleaned = _strip_fallback_id_text(_strip_confidence_tag(value))
        if not cleaned:
            continue
        key = _statement_dedupe_key(cleaned)
        if not key:
            continue
        if any(_are_near_duplicate_statement_keys(existing, key) for existing in seen_keys):
            continue
        seen_keys.append(key)
        out.append(cleaned)
    return out


def _dedupe_statement_rows(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    seen_keys: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        statement = row.get("statement", "")
        cleaned = _strip_fallback_id_text(_strip_confidence_tag(statement))
        if not cleaned:
            continue
        key = _statement_dedupe_key(cleaned)
        if not key:
            continue
        if any(_are_near_duplicate_statement_keys(existing, key) for existing in seen_keys):
            continue
        seen_keys.append(key)
        updated = dict(row)
        updated["statement"] = cleaned
        if "evidence_refs" in row:
            updated["evidence_refs"] = _sanitize_evidence_refs(row.get("evidence_refs"))
        out.append(updated)
    return out


def _sanitize_summary_payload(parsed: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        return parsed

    paper_meta = parsed.get("paper_meta")
    if isinstance(paper_meta, dict):
        authors_raw = paper_meta.get("authors")
        if isinstance(authors_raw, list):
            authors_display, authors_extracted_count = sanitize_author_list(authors_raw, max_items=24)
            paper_meta["authors"] = authors_display
            paper_meta["authors_extracted_count"] = authors_extracted_count
            paper_meta["authors_display_count"] = len(authors_display)

    if isinstance(parsed.get("key_findings"), list):
        parsed["key_findings"] = _dedupe_statement_lines(parsed.get("key_findings"))

    modalities = parsed.get("modalities")
    if isinstance(modalities, dict):
        for modality_name in ["text", "table", "figure", "supplement"]:
            block = modalities.get(modality_name)
            if not isinstance(block, dict):
                continue
            if isinstance(block.get("highlights"), list):
                block["highlights"] = _dedupe_statement_lines(block.get("highlights"))

    methods_compact = parsed.get("methods_compact")
    if isinstance(methods_compact, list):
        normalized_methods: list[dict[str, Any]] = []
        for row in methods_compact:
            if not isinstance(row, dict):
                continue
            updated = dict(row)
            if "statement" in row:
                updated["statement"] = _strip_fallback_id_text(_strip_confidence_tag(row.get("statement", "")))
            if "evidence_refs" in row:
                updated["evidence_refs"] = _sanitize_evidence_refs(row.get("evidence_refs"))
            normalized_methods.append(updated)
        parsed["methods_compact"] = normalized_methods

    sections_compact = parsed.get("sections_compact")
    if isinstance(sections_compact, dict):
        for section_name in ["introduction", "methods", "results", "discussion", "conclusion"]:
            rows = sections_compact.get(section_name)
            if not isinstance(rows, list):
                continue
            normalized_rows: list[dict[str, Any]] = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                updated = dict(row)
                if "statement" in row:
                    updated["statement"] = _strip_fallback_id_text(_strip_confidence_tag(row.get("statement", "")))
                if "evidence_refs" in row:
                    updated["evidence_refs"] = _sanitize_evidence_refs(row.get("evidence_refs"))
                normalized_rows.append(updated)
            sections_compact[section_name] = normalized_rows

    for block_name in ["extractive_evidence", "presentation_evidence"]:
        section_map = parsed.get(block_name)
        if not isinstance(section_map, dict):
            continue
        for section_name in ["introduction", "methods", "results", "discussion", "conclusion"]:
            rows = section_map.get(section_name)
            if isinstance(rows, list):
                section_map[section_name] = _dedupe_statement_rows(rows)

    sections = parsed.get("sections")
    if isinstance(sections, dict):
        for section_name in ["introduction", "methods", "results", "discussion", "conclusion"]:
            block = sections.get(section_name)
            if not isinstance(block, dict):
                continue
            rows = block.get("items")
            if isinstance(rows, list):
                block["items"] = _dedupe_statement_rows(rows)

    return parsed


def _media_order_key(item: dict[str, Any], *, prefer: str) -> tuple[int, int, int, str, str]:
    token_type_rank = 0 if prefer == "figure" else 1
    if str(item.get("asset_kind", "")).strip().lower() == "supp":
        token_type_rank += 10
    patterns = [FIGURE_ORDER_RE] if prefer == "figure" else [TABLE_ORDER_RE]
    if prefer == "figure":
        patterns.append(TABLE_ORDER_RE)
    else:
        patterns.append(FIGURE_ORDER_RE)

    candidates = [
        str(item.get("figure_id", "")).strip(),
        str(item.get("table_id", "")).strip(),
        str(item.get("anchor", "")).strip(),
        str(item.get("caption", "")).strip(),
    ]
    for idx, candidate in enumerate(candidates):
        lowered = candidate.lower()
        for pattern in patterns:
            match = pattern.search(lowered)
            if not match:
                continue
            raw_num = str(match.group(1) or "").strip().upper()
            suffix = str(match.group(2) or "").strip().lower()
            supp_penalty = 1 if raw_num.startswith("S") else 0
            digits = re.findall(r"\d+", raw_num)
            number = int(digits[0]) if digits else 10_000_000
            suffix_key = suffix if suffix else ""
            return (
                token_type_rank + supp_penalty,
                number,
                idx,
                suffix_key,
                lowered,
            )
    fallback_anchor = str(item.get("anchor", "")).strip().lower()
    fallback_caption = str(item.get("caption", "")).strip().lower()
    fallback_chunk = int(item.get("chunk_id", 10_000_000) or 10_000_000)
    return (token_type_rank + 99, fallback_chunk, 99, "", f"{fallback_anchor}|{fallback_caption}")


def _table_preview(content: str, *, max_rows: int = 120, max_cols: int = 40) -> dict[str, Any] | None:
    try:
        parsed = json.loads(content or "{}")
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None
    columns = parsed.get("columns")
    rows = parsed.get("data")
    if not isinstance(columns, list) or not isinstance(rows, list):
        return None

    total_cols = len(columns)
    total_rows = len(rows)
    preview_cols = [str(value) for value in columns[:max_cols]]
    preview_rows: list[list[str]] = []
    for row in rows[:max_rows]:
        if not isinstance(row, list):
            continue
        preview_rows.append([str(value) for value in row[:max_cols]])
    return {
        "columns": preview_cols,
        "rows": preview_rows,
        "total_rows": total_rows,
        "total_cols": total_cols,
    }


def _parse_summary_payload(summary_payload: str | None) -> tuple[dict[str, Any] | None, int, int]:
    if not summary_payload:
        return None, 1, 0
    try:
        parsed = json.loads(summary_payload)
    except Exception:
        return None, 1, 0
    if not isinstance(parsed, dict):
        return None, 1, 0
    parsed = _sanitize_summary_payload(parsed)
    try:
        summary_version = int(parsed.get("schema_version", 1))
    except Exception:
        summary_version = 1
    try:
        sectioned_report_version = int(parsed.get("sectioned_report_version", 0) or 0)
    except Exception:
        sectioned_report_version = 0
    return parsed, summary_version, sectioned_report_version


def _load_analysis_diagnostics(document_id: int) -> dict[str, Any] | None:
    diagnostics_path = artifacts_dir(document_id) / "analysis_diagnostics.json"
    if not diagnostics_path.exists():
        return None
    try:
        parsed = json.loads(diagnostics_path.read_text())
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _load_source_figure_legend_maps(document_id: int) -> tuple[dict[str, str], dict[str, str]]:
    path = artifacts_dir(document_id) / "source_figure_legends.json"
    if not path.exists():
        return {}, {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}, {}
    if not isinstance(parsed, dict):
        return {}, {}

    raw_caption = parsed.get("caption_map", {})
    raw_legend = parsed.get("legend_map", {})
    caption_map: dict[str, str] = {}
    legend_map: dict[str, str] = {}

    if isinstance(raw_caption, dict):
        for key, value in raw_caption.items():
            token = _normalize_figure_token(key)
            text = _normalize_media_legend(value)
            if not token or not text:
                continue
            existing = caption_map.get(token, "")
            if len(text) > len(existing):
                caption_map[token] = text

    if isinstance(raw_legend, dict):
        for key, value in raw_legend.items():
            token = _normalize_figure_token(key)
            text = _normalize_media_legend(value)
            if not token or not text:
                continue
            existing = legend_map.get(token, "")
            if len(text) > len(existing):
                legend_map[token] = text

    return caption_map, legend_map


@router.get("/status")
def get_status():
    processing = job_runner.status()
    pids = read_pids()
    text_model_path = settings.resolved_llm_text_model_path
    deep_model_path = settings.resolved_llm_deep_model_path
    vision_model_path = settings.resolved_llm_vision_model_path
    vision_mmproj_path = settings.resolved_llm_vision_mmproj_path
    return {
        # Legacy fields retained for compatibility with existing frontends.
        "model_path": str(vision_model_path),
        "mmproj_path": str(vision_mmproj_path),
        "model_exists": vision_model_path.exists(),
        "mmproj_exists": vision_mmproj_path.exists(),
        # Explicit multi-model readiness fields.
        "text_model_path": str(text_model_path),
        "text_model_exists": text_model_path.exists(),
        "deep_model_path": str(deep_model_path),
        "deep_model_exists": deep_model_path.exists(),
        "vision_model_path": str(vision_model_path),
        "vision_model_exists": vision_model_path.exists(),
        "vision_mmproj_path": str(vision_mmproj_path),
        "vision_mmproj_exists": vision_mmproj_path.exists(),
        "data_dir": str(settings.data_dir),
        "models_dir": str(settings.models_dir),
        "processing": processing,
        "backend_ready": True,
        "frontend_target": pids.get("frontend_port"),
        "worker_inflight": processing.get("inflight", 0),
        "stale_jobs_recovered": processing.get("stale_jobs_recovered", 0),
        "last_recovery_at": processing.get("last_recovery_at"),
        "last_error": processing.get("last_error"),
        "orphan_cleanup_last_run": processing.get("orphan_cleanup_last_run"),
        "runtime_build": _runtime_build_info(),
        "runtime_events": [item.model_dump() for item in _load_runtime_events(limit=10)],
    }


@router.get("/runtime/events")
def get_runtime_events(
    limit: int = Query(default=25, ge=1, le=200),
    since: str | None = Query(default=None),
):
    events = _load_runtime_events(limit=limit, since=since)
    return {"events": [item.model_dump() for item in events], "count": len(events)}


@router.post("/stop")
def stop_application():
    cleanup = job_runner.cleanup_orphans()
    pids = stop_app()
    return {"status": "stopping", "pids": pids, "cleanup": cleanup}


@router.get("/processing/status")
def processing_status():
    return job_runner.status()


@router.post("/processing/stop")
def processing_stop():
    job_runner.pause()
    cleanup = job_runner.cleanup_orphans()
    return {"status": "paused", "cleanup": cleanup}


@router.post("/processing/start")
def processing_start():
    job_runner.resume()
    return {"status": "running"}


@router.post("/processing/cleanup")
def processing_cleanup():
    result = job_runner.cleanup_orphans()
    return {"status": "cleaned", **result}


@router.post("/processing/recover")
def processing_recover():
    result = job_runner.recover_state()
    return {"status": "recovered", **result}


@router.get("/desktop/bootstrap", response_model=DesktopBootstrap)
def desktop_bootstrap(
    limit: int = Query(default=12, ge=1, le=50),
    session: Session = Depends(get_session),
):
    processing_raw = job_runner.status()
    jobs, _ = _load_job_rows(
        session,
        status=None,
        limit=limit,
        offset=0,
        sort="updated_at:desc",
    )
    events = _load_runtime_events(limit=20)

    latest_event = events[-1] if events else None
    text_model_path = settings.resolved_llm_text_model_path
    deep_model_path = settings.resolved_llm_deep_model_path
    vision_model_path = settings.resolved_llm_vision_model_path
    vision_mmproj_path = settings.resolved_llm_vision_mmproj_path
    return DesktopBootstrap(
        backend_ready=True,
        processing=DesktopProcessingStatus(
            running=bool(processing_raw.get("running", False)),
            paused=bool(processing_raw.get("paused", False)),
            inflight=int(processing_raw.get("inflight", 0)),
            worker_capacity=int(processing_raw.get("worker_capacity", 1)),
            stale_jobs_recovered=int(processing_raw.get("stale_jobs_recovered", 0)),
            orphan_cleanup_last_run=processing_raw.get("orphan_cleanup_last_run"),
            last_recovery_at=processing_raw.get("last_recovery_at"),
            last_error=processing_raw.get("last_error"),
        ),
        models=DesktopModelReadiness(
            # Legacy fields retained for compatibility with existing frontends.
            model_path=str(vision_model_path),
            mmproj_path=str(vision_mmproj_path),
            model_exists=vision_model_path.exists(),
            mmproj_exists=vision_mmproj_path.exists(),
            text_model_path=str(text_model_path),
            text_model_exists=text_model_path.exists(),
            deep_model_path=str(deep_model_path),
            deep_model_exists=deep_model_path.exists(),
            vision_model_path=str(vision_model_path),
            vision_model_exists=vision_model_path.exists(),
            vision_mmproj_path=str(vision_mmproj_path),
            vision_mmproj_exists=vision_mmproj_path.exists(),
        ),
        runtime_build=_runtime_build_info(),
        lifecycle=DesktopLifecycleStatus(
            event_count=len(events),
            latest_event=latest_event,
        ),
        latest_jobs=jobs,
    )


class UrlIngestRequest(BaseModel):
    url: str
    doi: Optional[str] = None
    fetch_supplements: bool = True


@router.post("/documents/from-url")
def create_document_from_url(
    payload: UrlIngestRequest,
    session: Session = Depends(get_session),
):
    try:
        document = ingest_url(
            session=session,
            input_url=payload.url,
            doi=payload.doi,
            fetch_supplements=payload.fetch_supplements,
        )
    except Exception as exc:
        user_message = str(exc).strip()
        next_action = "Verify URL/DOI and retry."
        lowered = user_message.lower()
        upload_fallback_hints = (
            "use from upload",
            "choose the main pdf",
            "downloadable pdf",
            "resolved pdf",
            "valid pdf",
            "publisher blocked automated access",
        )
        if any(hint in lowered for hint in upload_fallback_hints):
            user_message = _strip_upload_guidance(user_message) or "Could not fetch a usable main PDF from the URL/DOI."
            next_action = "Use From Upload, choose the main PDF, then submit again."
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": "ingest_url_failed",
                "user_message": user_message,
                "next_action": next_action,
            },
        ) from exc

    job = enqueue_job(session, document.id)
    return {"document_id": document.id, "job_id": job.id}


@router.post("/documents/upload")
async def create_document_from_upload(
    main_file: UploadFile = File(...),
    supp_files: list[UploadFile] | None = File(default=None),
    session: Session = Depends(get_session),
):
    try:
        document = await ingest_upload(
            session=session,
            main_file=main_file,
            supp_files=supp_files,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": "ingest_upload_failed",
                "user_message": str(exc),
                "next_action": "Verify the main PDF and supplement types, then retry.",
            },
        ) from exc
    job = enqueue_job(session, document.id)
    return {"document_id": document.id, "job_id": job.id}


@router.get("/documents")
def list_documents(session: Session = Depends(get_session)):
    docs = session.exec(select(Document).order_by(Document.created_at.desc())).all()
    return docs


@router.get("/jobs")
def list_jobs(
    status: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    sort: str = Query(default="updated_at:desc"),
    session: Session = Depends(get_session),
):
    status_filter = _job_status_filter(status)
    rows, normalized_sort = _load_job_rows(
        session,
        status=status_filter,
        limit=limit,
        offset=offset,
        sort=sort,
    )
    return {
        "items": [item.model_dump() for item in rows],
        "count": len(rows),
        "offset": offset,
        "limit": limit,
        "sort": normalized_sort,
    }


@router.get("/documents/{document_id}/media")
def get_document_media(document_id: int, session: Session = Depends(get_session)):
    document = session.get(Document, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    chunks = session.exec(
        select(Chunk)
        .where(Chunk.document_id == document_id)
        .where(Chunk.modality.in_(["figure", "table"]))
        .order_by(Chunk.id)
    ).all()
    asset_ids = sorted({chunk.asset_id for chunk in chunks if chunk.asset_id is not None})
    asset_kind_map: dict[int, str] = {}
    if asset_ids:
        assets = session.exec(select(Asset).where(Asset.id.in_(asset_ids))).all()
        asset_kind_map = {int(asset.id): str(asset.kind) for asset in assets if asset.id is not None}

    # Some publisher-scraped image files do not carry captions. Re-link by figure
    # ordinal token so those images can inherit parsed figure captions.
    figure_caption_lookup: dict[str, str] = {}
    figure_legend_lookup: dict[str, str] = {}
    source_caption_lookup, source_legend_lookup = _load_source_figure_legend_maps(document_id)
    for token, caption in source_caption_lookup.items():
        existing = figure_caption_lookup.get(token, "")
        if len(caption) > len(existing):
            figure_caption_lookup[token] = caption
    for token, legend in source_legend_lookup.items():
        existing = figure_legend_lookup.get(token, "")
        if len(legend) > len(existing):
            figure_legend_lookup[token] = legend

    for chunk in chunks:
        if chunk.modality != "figure":
            continue
        meta_obj = _parse_chunk_meta(chunk.meta)
        caption = _normalize_media_legend(meta_obj.get("caption"))
        if not caption or _is_generic_legend(caption):
            continue
        token = _figure_token_from_chunk(chunk, meta_obj)
        if not token:
            continue
        existing = figure_caption_lookup.get(token, "")
        if len(caption) > len(existing):
            figure_caption_lookup[token] = caption

    text_chunks = session.exec(
        select(Chunk)
        .where(Chunk.document_id == document_id)
        .where(Chunk.modality == "text")
        .order_by(Chunk.id)
    ).all()
    for text_chunk in text_chunks:
        sentence_candidates = LEGEND_SENTENCE_SPLIT_RE.split(str(text_chunk.content or ""))
        for sentence in sentence_candidates:
            candidate = _normalize_media_legend(sentence, max_chars=OCR_LEGEND_MAX_CHARS)
            if not candidate or _is_generic_legend(candidate):
                continue
            if len(LEGEND_WORD_RE.findall(candidate)) < 8:
                continue
            if not LEGEND_LINK_VERB_RE.search(candidate):
                continue
            tokens = _figure_tokens_from_text(candidate)
            if not tokens:
                continue
            for token in tokens:
                existing = figure_legend_lookup.get(token, "")
                if len(candidate) > len(existing):
                    figure_legend_lookup[token] = candidate

    figures: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []
    seen_any_figure: set[str] = set()
    seen_any_table: set[str] = set()
    seen_main_figure: set[str] = set()
    seen_main_table: set[str] = set()
    for chunk in chunks:
        if chunk.id is None:
            continue
        meta_obj = _parse_chunk_meta(chunk.meta)
        image_path = _resolve_chunk_image_path(document_id, chunk, meta_obj)
        remote_url = _remote_media_url(meta_obj)
        has_image = image_path is not None or _looks_like_image_url(remote_url)
        image_url = f"/api/documents/{document_id}/media/{chunk.id}/image" if has_image else None
        asset_url = remote_url or image_url
        prefer_direct_source = bool(
            asset_url
            and (
                ("/doi/suppl/" in str(asset_url).lower())
                or ("suppl_file" in str(asset_url).lower())
            )
            and _is_supplement_like_media(chunk, meta_obj, asset_url)
        )
        source_proxy_url = (
            str(asset_url)
            if prefer_direct_source
            else (f"/api/documents/{document_id}/media/{chunk.id}/source" if asset_url else None)
        )
        raw_asset_kind = asset_kind_map.get(int(chunk.asset_id), "main") if chunk.asset_id is not None else "main"
        asset_kind = "supp" if (raw_asset_kind == "supp" or _is_supplement_like_media(chunk, meta_obj, asset_url)) else "main"
        quality_flags = meta_obj.get("quality_flags", [])
        if not isinstance(quality_flags, list):
            quality_flags = []
        signature = _media_signature(chunk, meta_obj)
        legend_text, legend_source = _media_legend(chunk, meta_obj)
        caption_text = _normalize_media_legend(meta_obj.get("caption"))
        figure_token = _figure_token_from_chunk(chunk, meta_obj)
        linked_caption = figure_caption_lookup.get(figure_token, "") if figure_token else ""
        source_linked_legend = source_legend_lookup.get(figure_token, "") if figure_token else ""
        linked_legend = figure_legend_lookup.get(figure_token, "") if figure_token else ""
        if linked_caption and not caption_text:
            caption_text = linked_caption
        if source_linked_legend and (not legend_text or legend_source in {"missing", "ocr"}):
            legend_text = source_linked_legend
            legend_source = "linked_source_page"
        elif linked_legend and (not legend_text or legend_source in {"missing", "ocr"}):
            legend_text = linked_legend
            legend_source = "linked_text"
        if legend_text and caption_text and _legend_duplicates_caption(legend_text, caption_text):
            legend_text = ""
            legend_source = "missing"

        if chunk.modality == "figure":
            if asset_kind == "supp" and signature in seen_main_figure:
                continue
            if signature in seen_any_figure:
                continue
            if asset_kind == "main":
                seen_main_figure.add(signature)
            seen_any_figure.add(signature)
            figures.append(
                {
                    "chunk_id": int(chunk.id),
                    "anchor": chunk.anchor,
                    "caption": caption_text,
                    "page": meta_obj.get("page"),
                    "figure_id": str(meta_obj.get("figure_id") or ""),
                    "source": str(meta_obj.get("source") or ""),
                    "legend": legend_text,
                    "legend_source": legend_source,
                    "asset_kind": asset_kind,
                    "quality_flags": [str(item) for item in quality_flags],
                    "image_url": image_url,
                    "asset_url": asset_url,
                    "source_proxy_url": source_proxy_url,
                }
            )
            continue

        table_preview = _table_preview(chunk.content)
        source = str(meta_obj.get("source") or "").strip().lower()
        # Supplement links often arrive as table-link placeholders with no table payload.
        # Surface them in supplementary figures so users can open the supplement directly.
        if (
            chunk.modality == "table"
            and asset_kind == "supp"
            and table_preview is None
            and source == "html_link"
            and asset_url
        ):
            if signature in seen_any_figure:
                continue
            seen_any_figure.add(signature)
            figures.append(
                {
                    "chunk_id": int(chunk.id),
                    "anchor": chunk.anchor,
                    "caption": caption_text or "Supplementary material",
                    "page": meta_obj.get("page"),
                    "figure_id": str(meta_obj.get("figure_id") or ""),
                    "source": source or "html_link",
                    "legend": legend_text,
                    "legend_source": legend_source,
                    "asset_kind": "supp",
                    "quality_flags": [str(item) for item in quality_flags],
                    "image_url": image_url,
                    "asset_url": asset_url,
                    "source_proxy_url": source_proxy_url,
                }
            )
            continue

        if asset_kind == "supp" and signature in seen_main_table:
            continue
        if signature in seen_any_table:
            continue
        if asset_kind == "main":
            seen_main_table.add(signature)
        seen_any_table.add(signature)
        tables.append(
            {
                "chunk_id": int(chunk.id),
                "anchor": chunk.anchor,
                "caption": str(meta_obj.get("caption") or ""),
                "page": meta_obj.get("page"),
                "source": str(meta_obj.get("source") or ""),
                "legend": legend_text,
                "legend_source": legend_source,
                "asset_kind": asset_kind,
                "quality_flags": [str(item) for item in quality_flags],
                "image_url": image_url,
                "asset_url": asset_url,
                "source_proxy_url": source_proxy_url,
                "table_preview": table_preview,
            }
        )

    figures.sort(key=lambda item: _media_order_key(item, prefer="figure"))
    tables.sort(key=lambda item: _media_order_key(item, prefer="table"))
    return {"figures": figures, "tables": tables}


@router.get("/documents/{document_id}/media/{chunk_id}/image")
def get_document_media_image(
    document_id: int,
    chunk_id: int,
    session: Session = Depends(get_session),
):
    chunk = session.get(Chunk, chunk_id)
    if not chunk or chunk.document_id != document_id:
        raise HTTPException(status_code=404, detail="Media not found")
    if chunk.modality not in {"figure", "table"}:
        raise HTTPException(status_code=404, detail="Media not found")

    meta_obj = _parse_chunk_meta(chunk.meta)
    image_path = _resolve_chunk_image_path(document_id, chunk, meta_obj)
    if image_path is not None:
        return FileResponse(path=image_path, filename=image_path.name)

    remote_url = _remote_media_url(meta_obj)
    if not _looks_like_image_url(remote_url):
        raise HTTPException(status_code=404, detail="Media not found")

    document = session.get(Document, document_id)
    headers = {
        "User-Agent": settings.fetch_user_agent,
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }
    if document and document.source_url:
        headers["Referer"] = str(document.source_url)

    try:
        content, content_type, final_url = _proxy_remote_with_redirect_guard(
            remote_url,
            headers=headers,
            timeout=20,
            max_redirects=8,
        )
        if content is None:
            return RedirectResponse(url=final_url or remote_url, status_code=307)
        return Response(content=content, media_type=content_type or "image/jpeg")
    except Exception as exc:
        fallback_url = _redirect_fallback_target(remote_url, exc)
        if fallback_url:
            return RedirectResponse(url=fallback_url, status_code=307)
        raise HTTPException(status_code=404, detail=f"Media fetch failed: {exc}") from exc


@router.get("/documents/{document_id}/media/{chunk_id}/source")
def get_document_media_source(
    document_id: int,
    chunk_id: int,
    session: Session = Depends(get_session),
):
    chunk = session.get(Chunk, chunk_id)
    if not chunk or chunk.document_id != document_id:
        raise HTTPException(status_code=404, detail="Media not found")
    if chunk.modality not in {"figure", "table"}:
        raise HTTPException(status_code=404, detail="Media not found")

    meta_obj = _parse_chunk_meta(chunk.meta)
    image_path = _resolve_chunk_image_path(document_id, chunk, meta_obj)
    if image_path is not None:
        return FileResponse(path=image_path, filename=image_path.name)

    remote_url = _remote_media_url(meta_obj)
    if not remote_url:
        raise HTTPException(status_code=404, detail="Media not found")

    document = session.get(Document, document_id)
    headers = {
        "User-Agent": settings.fetch_user_agent,
        "Accept": "application/pdf,image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }
    if document and document.source_url:
        headers["Referer"] = str(document.source_url)

    try:
        content, content_type, final_url = _proxy_remote_with_redirect_guard(
            remote_url,
            headers=headers,
            timeout=25,
            max_redirects=8,
        )
        if content is None:
            return RedirectResponse(url=final_url or remote_url, status_code=307)
        filename = Path(urlparse(final_url or remote_url).path).name or "asset"
        response_headers = {"Content-Disposition": f'inline; filename="{filename}"'}
        return Response(content=content, media_type=content_type or "application/octet-stream", headers=response_headers)
    except Exception as exc:
        fallback_url = _redirect_fallback_target(remote_url, exc)
        if fallback_url:
            return RedirectResponse(url=fallback_url, status_code=307)
        raise HTTPException(status_code=404, detail=f"Media fetch failed: {exc}") from exc


@router.get("/documents/{document_id}")
def get_document(document_id: int, session: Session = Depends(get_session)):
    document = session.get(Document, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


@router.get("/jobs/{job_id}")
def get_job(job_id: int, session: Session = Depends(get_session)):
    job = session.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/documents/{document_id}/report")
def get_report(document_id: int, session: Session = Depends(get_session)):
    document = session.get(Document, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    findings = session.exec(
        select(Finding).where(Finding.document_id == document_id)
    ).all()
    discrepancies = session.exec(
        select(Discrepancy).where(Discrepancy.document_id == document_id)
    ).all()
    report = session.exec(
        select(Report)
        .where(Report.document_id == document_id)
        .order_by(Report.created_at.desc())
    ).first()
    summary_payload = report.payload if report else None
    summary_json, summary_version, sectioned_report_version = _parse_summary_payload(summary_payload)
    analysis_diagnostics = _load_analysis_diagnostics(document_id)

    return {
        "document": document,
        "findings": findings,
        "discrepancies": discrepancies,
        "summary": summary_payload,
        "summary_json": summary_json,
        "summary_version": summary_version,
        "summary_schema_version": summary_version,
        "sectioned_report_version": sectioned_report_version,
        "analysis_diagnostics": analysis_diagnostics,
    }


@router.get("/documents/{document_id}/report/summary", response_model=DesktopReportSummary)
def get_report_summary(document_id: int, session: Session = Depends(get_session)):
    document = session.get(Document, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    report = session.exec(
        select(Report)
        .where(Report.document_id == document_id)
        .order_by(Report.created_at.desc())
    ).first()
    if not report:
        return DesktopReportSummary(
            document_id=document_id,
            summary_version=0,
            report_status="not_ready",
            executive_summary=None,
            modality_cards=[],
            methods_card=[],
            sections_card=[],
            rerun_recommended=False,
            report_capabilities={},
            discrepancy_count=0,
            export_url=f"/api/documents/{document_id}/export",
            saved=is_document_saved(document_id),
        )

    summary_json, summary_version, _sectioned_report_version = _parse_summary_payload(report.payload)
    summary_json = summary_json or {}

    modalities = summary_json.get("modalities", {})
    cards = []
    for modality_name in ["text", "table", "figure", "supplement"]:
        section = modalities.get(modality_name, {})
        highlights = section.get("highlights", []) if isinstance(section, dict) else []
        cards.append(
            {
                "modality": modality_name,
                "highlights": highlights[:5] if isinstance(highlights, list) else [],
                "finding_count": len(section.get("findings", [])) if isinstance(section, dict) else 0,
                "coverage_gaps": section.get("coverage_gaps", []) if isinstance(section, dict) else [],
            }
        )
    sections_compact = summary_json.get("sections_compact", {})
    presentation_evidence = summary_json.get("presentation_evidence", {})
    has_presentation_evidence = isinstance(presentation_evidence, dict) and any(
        isinstance(presentation_evidence.get(section), list) and len(presentation_evidence.get(section, [])) > 0
        for section in ["introduction", "methods", "results", "discussion", "conclusion"]
    )
    if not has_presentation_evidence:
        extractive = summary_json.get("extractive_evidence", {})
        if isinstance(extractive, dict):
            derived: dict[str, list[dict[str, Any]]] = {}
            limits = {"introduction": 5, "methods": 10, "results": 10, "discussion": 8, "conclusion": 6}
            for section in ["introduction", "methods", "results", "discussion", "conclusion"]:
                rows = extractive.get(section, [])
                if not isinstance(rows, list):
                    continue
                seen: set[str] = set()
                deduped_rows: list[dict[str, Any]] = []
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    statement = _strip_confidence_tag(row.get("statement", ""))
                    if not statement:
                        continue
                    dedupe_key = " ".join(str(statement).lower().split())
                    if not dedupe_key or dedupe_key in seen:
                        continue
                    seen.add(dedupe_key)
                    deduped_rows.append({"statement": str(statement)})
                    if len(deduped_rows) >= int(limits.get(section, 8)):
                        break
                if deduped_rows:
                    derived[section] = deduped_rows
            if derived:
                presentation_evidence = derived
                has_presentation_evidence = True

    methods_card: list[str] = []
    sections_card: list[str] = []
    if has_presentation_evidence:
        seen_method_keys: list[str] = []
        for item in presentation_evidence.get("methods", []):
            if not isinstance(item, dict):
                continue
            statement = _strip_confidence_tag(item.get("statement", ""))
            if not statement:
                continue
            dedupe_key = _statement_dedupe_key(statement)
            if not dedupe_key:
                continue
            if any(_are_near_duplicate_statement_keys(existing, dedupe_key) for existing in seen_method_keys):
                continue
            seen_method_keys.append(dedupe_key)
            methods_card.append(str(statement))
            if len(methods_card) >= 6:
                break

        section_order = ["introduction", "results", "discussion", "conclusion"]
        section_limits = {
            "introduction": 4,
            "results": 5,
            "discussion": 4,
            "conclusion": 4,
        }
        sections_card_target = 16
        section_headers = {
            "introduction": "Introduction",
            "results": "Results",
            "discussion": "Discussion",
            "conclusion": "Conclusion",
        }
        seen_section_keys: list[str] = []
        section_candidates: dict[str, list[str]] = {key: [] for key in section_order}
        for section_key in section_order:
            rows = presentation_evidence.get(section_key, [])
            if not isinstance(rows, list):
                continue
            prefix = section_headers.get(section_key, section_key.title())
            for item in rows:
                if not isinstance(item, dict):
                    continue
                statement = _strip_confidence_tag(item.get("statement", ""))
                if not statement:
                    continue
                dedupe_key = _statement_dedupe_key(statement)
                if not dedupe_key:
                    continue
                if any(_are_near_duplicate_statement_keys(existing, dedupe_key) for existing in seen_section_keys):
                    continue
                seen_section_keys.append(dedupe_key)
                section_candidates[section_key].append(f"{prefix}: {statement}")

        for section_key in section_order:
            limit = int(section_limits.get(section_key, 0) or 0)
            if limit <= 0:
                continue
            for line in section_candidates.get(section_key, [])[:limit]:
                sections_card.append(line)
                if len(sections_card) >= sections_card_target:
                    break
            if len(sections_card) >= sections_card_target:
                break
        if len(sections_card) < sections_card_target:
            for section_key in section_order:
                limit = int(section_limits.get(section_key, 0) or 0)
                for line in section_candidates.get(section_key, [])[limit:]:
                    sections_card.append(line)
                    if len(sections_card) >= sections_card_target:
                        break
                if len(sections_card) >= sections_card_target:
                    break
    else:
        for slot in summary_json.get("methods_compact", []) if isinstance(summary_json.get("methods_compact"), list) else []:
            if not isinstance(slot, dict):
                continue
            label = str(slot.get("label", "")).strip()
            statement = _strip_confidence_tag(slot.get("statement", ""))
            if not label or not statement:
                continue
            methods_card.append(f"{label}: {statement}")
            if len(methods_card) >= 6:
                break
        if isinstance(sections_compact, dict):
            for section_key in ["introduction", "results", "discussion", "conclusion"]:
                rows = sections_compact.get(section_key, [])
                if not isinstance(rows, list):
                    continue
                for slot in rows:
                    if not isinstance(slot, dict):
                        continue
                    label = str(slot.get("label", "")).strip()
                    statement = _strip_confidence_tag(slot.get("statement", ""))
                    if not label or not statement:
                        continue
                    sections_card.append(f"{label}: {statement}")
                    if len(sections_card) >= 12:
                        break
                if len(sections_card) >= 12:
                    break
    has_methods_compact = isinstance(summary_json.get("methods_compact"), list) and len(summary_json.get("methods_compact", [])) > 0
    sections_compact_version = int(summary_json.get("sections_compact_version", 0) or 0)
    has_sections_compact = isinstance(sections_compact, dict) and bool(sections_compact) and sections_compact_version >= 1
    sections_extracted = summary_json.get("sections_extracted", {})
    sections_extracted_version = int(summary_json.get("sections_extracted_version", 0) or 0)
    has_sections_extracted = isinstance(sections_extracted, dict) and bool(sections_extracted) and sections_extracted_version >= 1
    report_capabilities = {
        "methods_compact": has_methods_compact,
        "sections_compact": has_sections_compact,
        "sections_extracted": has_sections_extracted,
        "presentation_evidence": has_presentation_evidence,
        "coverage_snapshot_line": bool(str(summary_json.get("coverage_snapshot_line", "")).strip()),
    }
    rerun_recommended = not (has_sections_compact or has_presentation_evidence)
    discrepancies = summary_json.get("discrepancies", [])
    return DesktopReportSummary(
        document_id=document_id,
        summary_version=summary_version,
        report_status="ready",
        executive_summary=_strip_confidence_tag(summary_json.get("executive_summary")),
        modality_cards=cards,
        methods_card=methods_card,
        sections_card=sections_card,
        rerun_recommended=rerun_recommended,
        report_capabilities=report_capabilities,
        discrepancy_count=len(discrepancies) if isinstance(discrepancies, list) else 0,
        overall_confidence=None,
        export_url=f"/api/documents/{document_id}/export",
        saved=is_document_saved(document_id),
    )


@router.get("/documents/{document_id}/report/save-status")
def get_report_save_status(document_id: int, session: Session = Depends(get_session)):
    document = session.get(Document, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return saved_status(document_id)


@router.get("/reports/saved")
def get_saved_reports():
    return {"items": list_saved_reports()}


@router.post("/documents/{document_id}/report/save")
def save_report(document_id: int, session: Session = Depends(get_session)):
    document = session.get(Document, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    try:
        saved_path = save_report_export(session, document_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "status": "saved",
        "document_id": document_id,
        "saved_path": str(saved_path),
        "saved": True,
    }


@router.get("/documents/{document_id}/export")
def export_report(document_id: int, session: Session = Depends(get_session)):
    document = session.get(Document, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    findings = session.exec(
        select(Finding).where(Finding.document_id == document_id)
    ).all()
    discrepancies = session.exec(
        select(Discrepancy).where(Discrepancy.document_id == document_id)
    ).all()
    report = session.exec(
        select(Report)
        .where(Report.document_id == document_id)
        .order_by(Report.created_at.desc())
    ).first()
    summary_payload = report.payload if report else None
    summary_json, summary_version, sectioned_report_version = _parse_summary_payload(summary_payload)
    analysis_diagnostics = _load_analysis_diagnostics(document_id)

    return {
        "document": document,
        "summary": summary_payload,
        "summary_json": summary_json,
        "summary_version": summary_version,
        "summary_schema_version": summary_version,
        "sectioned_report_version": sectioned_report_version,
        "findings": findings,
        "discrepancies": discrepancies,
        "analysis_diagnostics": analysis_diagnostics,
    }
