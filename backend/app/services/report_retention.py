from __future__ import annotations

from datetime import datetime
import html
import json
from pathlib import Path
import re
import shutil
from typing import Any

from sqlalchemy import delete
from sqlmodel import Session, select

from app.core.config import settings
from app.db.models import Asset, Chunk, Discrepancy, Document, Finding, Job, JobStatus, Report
from app.services.storage import document_dir


_SAVED_INDEX_VERSION = 1
_MAX_LIVE_REPORTS = 10
_CONFIDENCE_TEXT_RE = re.compile(
    r"\s*(?:\((?:model\s+)?confidence[:\s]*\d{1,3}(?:\.\d+)?%?\)|(?:model\s+)?confidence[:\s]+\d{1,3}(?:\.\d+)?%|\(\d{1,3}(?:\.\d+)?%\))\s*",
    re.IGNORECASE,
)


def save_report_export(session: Session, document_id: int) -> Path:
    document = session.get(Document, document_id)
    if not document:
        raise ValueError("Document not found")

    report = session.exec(
        select(Report)
        .where(Report.document_id == document_id)
        .order_by(Report.created_at.desc())
    ).first()
    if not report:
        raise ValueError("Report not found")

    findings = session.exec(select(Finding).where(Finding.document_id == document_id)).all()
    discrepancies = session.exec(select(Discrepancy).where(Discrepancy.document_id == document_id)).all()

    summary_json, summary_version = _parse_summary_payload(report.payload)
    now = datetime.utcnow()
    payload = {
        "document": _model_dump(document),
        "summary": report.payload,
        "summary_json": summary_json,
        "summary_version": summary_version,
        "findings": [_model_dump(item) for item in findings],
        "discrepancies": [_model_dump(item) for item in discrepancies],
        "saved_at": now.isoformat(),
    }

    label = _saved_label(document_id, document, summary_json)
    doc_dir = saved_reports_dir() / label["dir_name"]
    doc_dir.mkdir(parents=True, exist_ok=True)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    stem = _next_available_stem(
        doc_dir,
        f"report_{timestamp}_{label['file_suffix']}",
    )
    out_path = doc_dir / f"{stem}.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    md_path = doc_dir / f"{stem}.md"
    md_path.write_text(_render_saved_markdown(payload))
    html_path = doc_dir / f"{stem}.html"
    html_path.write_text(_render_saved_html(payload))

    index = _load_saved_index()
    key = str(document_id)
    entry = index["documents"].get(key)
    if not isinstance(entry, dict):
        entry = {"saved_at": datetime.utcnow().isoformat(), "files": []}
        index["documents"][key] = entry
    files = entry.get("files", [])
    if not isinstance(files, list):
        files = []
    for path in (out_path, md_path, html_path):
        rel = str(path.relative_to(saved_reports_dir()))
        if rel not in files:
            files.append(rel)
    entry["files"] = files
    entry["saved_at"] = datetime.utcnow().isoformat()
    entry["label"] = label["display_label"]
    _write_saved_index(index)

    return out_path


def is_document_saved(document_id: int) -> bool:
    index = _load_saved_index()
    entry = index["documents"].get(str(document_id))
    return isinstance(entry, dict) and bool(entry.get("files"))


def saved_status(document_id: int) -> dict[str, Any]:
    index = _load_saved_index()
    entry = index["documents"].get(str(document_id))
    if not isinstance(entry, dict):
        return {"document_id": document_id, "saved": False, "saved_files": []}
    files = entry.get("files", [])
    if not isinstance(files, list):
        files = []
    return {
        "document_id": document_id,
        "saved": bool(files),
        "saved_files": files,
        "saved_at": entry.get("saved_at"),
    }


def list_saved_reports() -> list[dict[str, Any]]:
    index = _load_saved_index()
    out: list[dict[str, Any]] = []
    for key, entry in index["documents"].items():
        if not isinstance(entry, dict):
            continue
        try:
            document_id = int(key)
        except Exception:
            continue
        files = entry.get("files", [])
        if not isinstance(files, list):
            files = []
        out.append(
            {
                "document_id": document_id,
                "saved_at": entry.get("saved_at"),
                "saved_files": files,
                "label": entry.get("label"),
            }
        )
    out.sort(key=lambda item: str(item.get("saved_at") or ""), reverse=True)
    return out


def enforce_report_retention(session: Session, *, keep_latest: int) -> dict[str, Any]:
    if keep_latest <= 0:
        keep_latest = 1
    keep_latest = min(keep_latest, _MAX_LIVE_REPORTS)

    latest_doc_ids = _latest_report_document_ids(session)
    keep_ids = set(latest_doc_ids[:keep_latest])
    candidate_ids = [doc_id for doc_id in latest_doc_ids if doc_id not in keep_ids]

    pruned: list[int] = []
    skipped_active: list[int] = []
    for document_id in candidate_ids:
        if _has_active_jobs(session, document_id):
            skipped_active.append(document_id)
            continue
        _delete_document_data(session, document_id)
        pruned.append(document_id)

    if pruned:
        session.commit()
        for document_id in pruned:
            shutil.rmtree(document_dir(document_id), ignore_errors=True)

    return {
        "keep_latest": keep_latest,
        "tracked_reports": len(latest_doc_ids),
        "pruned_count": len(pruned),
        "pruned_document_ids": pruned,
        "skipped_active_document_ids": skipped_active,
    }


def saved_reports_dir() -> Path:
    return settings.data_dir / "saved_reports"


def _saved_index_path() -> Path:
    return saved_reports_dir() / "saved_reports_index.json"


def _load_saved_index() -> dict[str, Any]:
    path = _saved_index_path()
    default = {"version": _SAVED_INDEX_VERSION, "documents": {}}
    if not path.exists():
        return default
    try:
        parsed = json.loads(path.read_text())
    except Exception:
        return default
    if not isinstance(parsed, dict):
        return default
    docs = parsed.get("documents")
    if not isinstance(docs, dict):
        docs = {}
    return {"version": parsed.get("version", _SAVED_INDEX_VERSION), "documents": docs}


def _write_saved_index(index: dict[str, Any]) -> None:
    path = _saved_index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(index, indent=2, default=str))
    tmp.replace(path)


def _latest_report_document_ids(session: Session) -> list[int]:
    rows = session.exec(select(Report).order_by(Report.created_at.desc())).all()
    doc_ids: list[int] = []
    seen: set[int] = set()
    for row in rows:
        if row.document_id in seen:
            continue
        seen.add(row.document_id)
        doc_ids.append(row.document_id)
    return doc_ids


def _has_active_jobs(session: Session, document_id: int) -> bool:
    stmt = (
        select(Job.id)
        .where(Job.document_id == document_id)
        .where(Job.status.in_([JobStatus.queued, JobStatus.running]))
        .limit(1)
    )
    return session.exec(stmt).first() is not None


def _delete_document_data(session: Session, document_id: int) -> None:
    session.exec(delete(Finding).where(Finding.document_id == document_id))
    session.exec(delete(Discrepancy).where(Discrepancy.document_id == document_id))
    session.exec(delete(Report).where(Report.document_id == document_id))
    session.exec(delete(Chunk).where(Chunk.document_id == document_id))
    session.exec(delete(Asset).where(Asset.document_id == document_id))
    session.exec(delete(Job).where(Job.document_id == document_id))
    session.exec(delete(Document).where(Document.id == document_id))


def _parse_summary_payload(summary_payload: str | None) -> tuple[dict[str, Any] | None, int]:
    if not summary_payload:
        return None, 1
    try:
        parsed = json.loads(summary_payload)
    except Exception:
        return None, 1
    if not isinstance(parsed, dict):
        return None, 1
    try:
        version = int(parsed.get("schema_version", 1))
    except Exception:
        version = 1
    return parsed, version


def _model_dump(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    return {}


def _saved_label(document_id: int, document: Document, summary_json: dict[str, Any] | None) -> dict[str, str]:
    meta = summary_json.get("paper_meta", {}) if isinstance(summary_json, dict) else {}
    title = str(meta.get("title") or document.title or f"document-{document_id}").strip()
    journal = str(meta.get("journal") or "").strip()
    date_text = str(meta.get("date") or "").strip()
    year_match = re.search(r"\b(19|20)\d{2}\b", date_text)
    year = year_match.group(0) if year_match else ""

    title_short = _slug_words(title, max_words=5, fallback=f"document-{document_id}")
    journal_short = _slug_words(journal, max_words=3, fallback="")
    suffix_parts = [title_short]
    if journal_short:
        suffix_parts.append(journal_short)
    if year:
        suffix_parts.append(year)
    suffix = "_".join(part for part in suffix_parts if part)[:120].strip("_")
    if not suffix:
        suffix = f"document-{document_id}"

    display_bits = [title]
    if journal:
        display_bits.append(journal)
    if year:
        display_bits.append(year)
    display_label = " | ".join(display_bits)
    return {
        "dir_name": f"doc_{document_id}_{title_short}",
        "file_suffix": suffix,
        "display_label": display_label,
    }


def _slug_words(value: str, *, max_words: int, fallback: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", str(value or "").lower())
    if not tokens:
        return fallback
    return "_".join(tokens[:max_words]).strip("_") or fallback


def _next_available_stem(doc_dir: Path, base_stem: str) -> str:
    stem = re.sub(r"_+", "_", str(base_stem or "").strip("_"))
    if not stem:
        stem = f"report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    candidate = stem
    idx = 2
    while (doc_dir / f"{candidate}.json").exists() or (doc_dir / f"{candidate}.md").exists():
        candidate = f"{stem}_{idx}"
        idx += 1
    return candidate


def _render_saved_markdown(payload: dict[str, Any]) -> str:
    summary_json = payload.get("summary_json") if isinstance(payload.get("summary_json"), dict) else {}
    meta = summary_json.get("paper_meta", {}) if isinstance(summary_json, dict) else {}
    title = str(meta.get("title") or payload.get("document", {}).get("title") or "Paper Report").strip()
    lines: list[str] = [f"# {title}", ""]

    journal = str(meta.get("journal") or "").strip()
    date = str(meta.get("date") or "").strip()
    authors = meta.get("authors", [])
    if journal:
        lines.append(f"- Journal: {journal}")
    if date:
        lines.append(f"- Date: {date}")
    if isinstance(authors, list) and authors:
        lines.append(f"- Authors: {', '.join(str(item) for item in authors[:12])}")
    lines.append("")

    executive = _strip_confidence_text(str(summary_json.get("executive_summary") or "").strip())
    if executive:
        lines.extend(["## Executive Summary", "", executive, ""])

    sections = summary_json.get("sections", {}) if isinstance(summary_json, dict) else {}
    for section_key in ("introduction", "methods", "results", "discussion", "conclusion"):
        block = sections.get(section_key) if isinstance(sections, dict) else None
        if not isinstance(block, dict):
            continue
        section_title = section_key.title()
        lines.extend([f"## {section_title}", ""])
        items = block.get("items", [])
        if isinstance(items, list) and items:
            for idx, item in enumerate(items[:30], start=1):
                if not isinstance(item, dict):
                    continue
                statement = _strip_confidence_text(str(item.get("statement") or "").strip())
                if not statement:
                    continue
                refs = item.get("evidence_refs", [])
                src = f" [{', '.join(str(ref) for ref in refs[:3])}]" if isinstance(refs, list) and refs else ""
                lines.append(f"{idx}. {statement}{src}")
        else:
            lines.append("_No extracted items._")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _render_saved_html(payload: dict[str, Any]) -> str:
    summary_json = payload.get("summary_json") if isinstance(payload.get("summary_json"), dict) else {}
    meta = summary_json.get("paper_meta", {}) if isinstance(summary_json, dict) else {}
    title = str(meta.get("title") or payload.get("document", {}).get("title") or "Paper Report").strip()
    executive = _strip_confidence_text(str(summary_json.get("executive_summary") or "").strip())
    sections = summary_json.get("sections", {}) if isinstance(summary_json, dict) else {}

    def esc(value: Any) -> str:
        return html.escape(str(value or ""))

    html_parts = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8' />",
        f"<title>{esc(title)}</title>",
        "<style>body{font-family:Arial,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;line-height:1.45;color:#13261a}h1{margin-bottom:8px}section{border:1px solid #bfd4c6;border-radius:10px;padding:12px;margin:12px 0}small{color:#3f5d49}ol{margin:8px 0 0 18px}li{margin:4px 0}</style>",
        "</head><body>",
        f"<h1>{esc(title)}</h1>",
    ]
    journal = str(meta.get("journal") or "").strip()
    date = str(meta.get("date") or "").strip()
    if journal or date:
        html_parts.append(f"<small>{esc(' | '.join(part for part in [journal, date] if part))}</small>")

    if executive:
        executive_html = "<br/>".join(esc(part) for part in executive.split("\n"))
        html_parts.append(f"<section><h2>Executive Summary</h2><p>{executive_html}</p></section>")

    for section_key in ("introduction", "methods", "results", "discussion", "conclusion"):
        block = sections.get(section_key) if isinstance(sections, dict) else None
        if not isinstance(block, dict):
            continue
        html_parts.append(f"<section><h2>{esc(section_key.title())}</h2>")
        items = block.get("items", [])
        if isinstance(items, list) and items:
            html_parts.append("<ol>")
            for item in items[:30]:
                if not isinstance(item, dict):
                    continue
                statement = _strip_confidence_text(str(item.get("statement") or "").strip())
                if not statement:
                    continue
                refs = item.get("evidence_refs", [])
                source = ""
                if isinstance(refs, list) and refs:
                    source = f" <small>(source: {esc(', '.join(str(ref) for ref in refs[:3]))})</small>"
                html_parts.append(f"<li>{esc(statement)}{source}</li>")
            html_parts.append("</ol>")
        else:
            html_parts.append("<p><em>No extracted items.</em></p>")
        html_parts.append("</section>")

    html_parts.append("</body></html>")
    return "".join(html_parts)


def _strip_confidence_text(value: str) -> str:
    return " ".join(_CONFIDENCE_TEXT_RE.sub(" ", str(value or "")).split()).strip()
