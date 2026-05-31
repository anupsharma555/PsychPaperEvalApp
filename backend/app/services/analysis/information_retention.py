from __future__ import annotations

from datetime import datetime, timezone
import html
import json
from pathlib import Path
import re
from typing import Any


SECTION_KEYS = ["introduction", "methods", "results", "discussion", "conclusion"]
AUDIT_STAGES = [
    "parsed_chunks",
    "text_packets",
    "table_packets",
    "figure_packets",
    "supplement_packets",
    "sections_extracted",
    "sections_compact",
    "sections",
    "extractive_evidence",
    "presentation_evidence",
    "executive_report",
]
STATUS_PRESENT_CORRECT = "present_correct_section"
STATUS_PRESENT_WRONG = "present_wrong_section"
STATUS_MISSING = "missing"
MATCH_THRESHOLD = 0.42
MAX_SOURCE_SENTENCES = 300
MAX_STAGE_ITEMS = 900
MAX_TEXT_CHARS = 420
TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-]*")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
HEADING_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+){0,3}\s+)?"
    r"(abstract|introduction|background|objective|objectives|aim|aims|method|methods|"
    r"materials?\s+and\s+methods?|participants?|results?|discussion|conclusions?|summary)"
    r"[\s\-:.;\)]*$",
    re.IGNORECASE,
)
STRUCTURED_PREFIX_RE = re.compile(
    r"^\s*(objective|objectives|background|aim|aims|method|methods|design|results|discussion|conclusion|conclusions)\s*:\s*(.+)$",
    re.IGNORECASE,
)
HTML_TAG_RE = re.compile(r"<[^>]+>")
REFERENCE_NOISE_RE = re.compile(
    r"\b("
    r"references?|bibliography|doi:|copyright|all rights reserved|"
    r"received [a-z]+ \d{1,2}, \d{4}|accepted [a-z]+ \d{1,2}, \d{4}|"
    r"published online|correspondence|supplementary material is available"
    r")\b",
    re.IGNORECASE,
)
HEADER_FOOTER_RE = re.compile(
    r"\b("
    r"downloaded from|licensed under|creative commons|journal homepage|"
    r"www\.|http://|https://|pmcid:|pmid:"
    r")\b",
    re.IGNORECASE,
)
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "these",
    "this",
    "to",
    "was",
    "were",
    "with",
}


def build_information_retention_audit(
    *,
    document_id: int | None,
    source_assets: list[dict[str, Any]] | None,
    parsed_chunks: list[dict[str, Any]] | None,
    summary_json: dict[str, Any] | None,
    max_source_sentences: int = MAX_SOURCE_SENTENCES,
) -> dict[str, Any]:
    source_basis, source_warning, source_sentences = _build_source_sentence_bank(
        source_assets=source_assets or [],
        parsed_chunks=parsed_chunks or [],
        max_source_sentences=max_source_sentences,
    )
    stages = _extract_stage_items(
        parsed_chunks=parsed_chunks or [],
        summary_json=summary_json or {},
    )
    source_sentences = _score_source_sentences(source_sentences, stages)
    stage_metrics = _stage_metrics(source_sentences, stages)
    first_loss_counts = _first_loss_counts(source_sentences)
    section_metrics = _section_metrics(source_sentences)
    audit = {
        "schema_version": 1,
        "document_id": int(document_id or 0),
        "source_basis": source_basis,
        "source_basis_warning": source_warning,
        "generated_at": _utc_timestamp(),
        "source_sentence_count": len(source_sentences),
        "stage_metrics": stage_metrics,
        "first_loss_counts": first_loss_counts,
        "section_metrics": section_metrics,
        "source_sentences": source_sentences,
    }
    audit["compact_summary"] = compact_information_retention_summary(audit)
    return audit


def compact_information_retention_summary(audit: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(audit, dict):
        return {}
    section_metrics = audit.get("section_metrics", {})
    worst_sections: list[dict[str, Any]] = []
    if isinstance(section_metrics, dict):
        for section, payload in section_metrics.items():
            if not isinstance(payload, dict):
                continue
            worst_sections.append(
                {
                    "section": section,
                    "source_sentence_count": int(payload.get("source_sentence_count", 0) or 0),
                    "final_retained_rate": float(payload.get("final_retained_rate", 0.0) or 0.0),
                    "final_cumulative_lost_count": int(payload.get("final_cumulative_lost_count", 0) or 0),
                }
            )
    worst_sections.sort(key=lambda item: (-int(item["final_cumulative_lost_count"]), str(item["section"])))
    return {
        "schema_version": int(audit.get("schema_version", 1) or 1),
        "source_basis": str(audit.get("source_basis", "")),
        "source_basis_warning": str(audit.get("source_basis_warning", "")),
        "source_sentence_count": int(audit.get("source_sentence_count", 0) or 0),
        "stage_metrics": audit.get("stage_metrics", []),
        "first_loss_counts": audit.get("first_loss_counts", {}),
        "worst_sections_by_cumulative_loss": worst_sections[:5],
    }


def _build_source_sentence_bank(
    *,
    source_assets: list[dict[str, Any]],
    parsed_chunks: list[dict[str, Any]],
    max_source_sentences: int,
) -> tuple[str, str, list[dict[str, Any]]]:
    direct_warning = ""
    for asset in _rank_source_assets(source_assets):
        path = Path(str(asset.get("path") or "")).expanduser()
        if not path.exists() or not path.is_file():
            continue
        is_pdf = _is_pdf_asset(asset, path)
        if is_pdf:
            try:
                sentences = _source_sentences_from_pdf(path, max_source_sentences=max_source_sentences)
            except Exception as exc:
                direct_warning = f"Original PDF text extraction failed: {_normalize_text(str(exc))}"
                sentences = []
            if sentences:
                return "original_pdf", "", sentences
            continue
        try:
            text = _read_text_asset(path)
        except Exception as exc:
            direct_warning = f"Original asset text extraction failed: {_normalize_text(str(exc))}"
            text = ""
        if text:
            sentences = _source_sentences_from_text(
                text,
                anchor_prefix=f"asset:{_safe_anchor(path.name)}",
                max_source_sentences=max_source_sentences,
            )
            if sentences:
                return "original_asset_text", "", sentences

    sentences = _source_sentences_from_parsed_chunks(parsed_chunks, max_source_sentences=max_source_sentences)
    if sentences:
        warning = direct_warning or "Original source asset was unavailable or unreadable; parse-stage loss is unmeasured."
        return "parsed_chunks", warning, sentences
    warning = direct_warning or "No source sentences could be extracted from the original asset or parsed chunks."
    return "none", warning, []


def _rank_source_assets(source_assets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def rank(asset: dict[str, Any]) -> tuple[int, int, str]:
        kind = str(asset.get("kind") or "").lower()
        filename = str(asset.get("filename") or asset.get("path") or "")
        is_main = 0 if kind == "main" else 1
        is_pdf = 0 if filename.lower().endswith(".pdf") or "pdf" in str(asset.get("content_type") or "").lower() else 1
        return is_main, is_pdf, filename

    return sorted([asset for asset in source_assets if isinstance(asset, dict)], key=rank)


def _is_pdf_asset(asset: dict[str, Any], path: Path) -> bool:
    content_type = str(asset.get("content_type") or "").lower()
    return path.suffix.lower() == ".pdf" or "pdf" in content_type


def _source_sentences_from_pdf(path: Path, *, max_source_sentences: int) -> list[dict[str, Any]]:
    try:
        import pypdfium2 as pdfium
    except Exception as exc:
        raise RuntimeError("pypdfium2 is required for direct PDF source extraction") from exc

    pdf = pdfium.PdfDocument(str(path))
    rows: list[dict[str, Any]] = []
    active_section = "unknown"
    try:
        total_pages = len(pdf)
        for page_idx in range(total_pages):
            page = pdf.get_page(page_idx)
            text_page = None
            try:
                text_page = page.get_textpage()
                raw_text = str(text_page.get_text_bounded() or "")
            finally:
                try:
                    if text_page is not None:
                        text_page.close()
                except Exception:
                    pass
                try:
                    page.close()
                except Exception:
                    pass
            if not raw_text.strip():
                continue
            active_section = _append_pdf_page_sentences(
                rows,
                raw_text,
                page_idx=page_idx,
                total_pages=total_pages,
                active_section=active_section,
                max_source_sentences=max_source_sentences,
            )
            if len(rows) >= max_source_sentences:
                break
    finally:
        try:
            pdf.close()
        except Exception:
            pass
    return _finalize_source_rows(rows, max_source_sentences=max_source_sentences)


def _append_pdf_page_sentences(
    rows: list[dict[str, Any]],
    raw_text: str,
    *,
    page_idx: int,
    total_pages: int,
    active_section: str,
    max_source_sentences: int,
) -> str:
    buffer: list[str] = []
    buffer_section = active_section

    def flush() -> None:
        nonlocal buffer
        if not buffer:
            return
        text = _normalize_candidate_text(" ".join(buffer))
        buffer = []
        for sentence in _split_sentences(text):
            if len(rows) >= max_source_sentences:
                return
            section = buffer_section
            if section == "unknown":
                section = _position_section(page_idx, total_pages)
            rows.append(
                {
                    "section": section,
                    "modality": "text",
                    "anchor": f"page:{page_idx + 1}",
                    "sentence": sentence,
                    "source_order": len(rows),
                }
            )

    for line in raw_text.splitlines():
        clean = _normalize_candidate_text(line)
        if not clean:
            continue
        heading = HEADING_RE.match(clean)
        if heading:
            flush()
            active_section = _normalize_section_label(heading.group(1))
            buffer_section = active_section
            continue
        structured = STRUCTURED_PREFIX_RE.match(clean)
        if structured:
            flush()
            section = _normalize_section_label(structured.group(1))
            statement = _normalize_candidate_text(structured.group(2))
            for sentence in _split_sentences(statement):
                if len(rows) >= max_source_sentences:
                    return active_section
                rows.append(
                    {
                        "section": section,
                        "modality": "text",
                        "anchor": f"page:{page_idx + 1}",
                        "sentence": sentence,
                        "source_order": len(rows),
                    }
                )
            continue
        if _is_noise_sentence(clean):
            continue
        buffer.append(clean)
    flush()
    return active_section


def _source_sentences_from_text(
    text: str,
    *,
    anchor_prefix: str,
    max_source_sentences: int,
) -> list[dict[str, Any]]:
    clean = _html_to_text(text)
    rows: list[dict[str, Any]] = []
    active_section = "unknown"
    paragraph_idx = 0
    for raw_line in clean.splitlines():
        line = _normalize_candidate_text(raw_line)
        if not line:
            continue
        heading = HEADING_RE.match(line)
        if heading:
            active_section = _normalize_section_label(heading.group(1))
            continue
        for sentence in _split_sentences(line):
            if len(rows) >= max_source_sentences:
                break
            rows.append(
                {
                    "section": active_section,
                    "modality": "text",
                    "anchor": f"{anchor_prefix}:{paragraph_idx}",
                    "sentence": sentence,
                    "source_order": len(rows),
                }
            )
        paragraph_idx += 1
        if len(rows) >= max_source_sentences:
            break
    return _finalize_source_rows(rows, max_source_sentences=max_source_sentences)


def _source_sentences_from_parsed_chunks(
    parsed_chunks: list[dict[str, Any]],
    *,
    max_source_sentences: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, chunk in enumerate(parsed_chunks):
        if len(rows) >= max_source_sentences:
            break
        if not isinstance(chunk, dict):
            continue
        modality = str(chunk.get("modality") or "text").strip().lower()
        if modality == "meta":
            continue
        anchor = str(chunk.get("anchor") or f"chunk:{idx}").strip()
        section = _section_from_chunk(chunk, idx=idx, total=max(1, len(parsed_chunks)))
        content = _extract_text_from_value(chunk.get("content"))
        if not content:
            continue
        for sentence in _split_sentences(content):
            if len(rows) >= max_source_sentences:
                break
            rows.append(
                {
                    "section": section,
                    "modality": modality or "text",
                    "anchor": anchor,
                    "sentence": sentence,
                    "source_order": len(rows),
                }
            )
    return _finalize_source_rows(rows, max_source_sentences=max_source_sentences)


def _finalize_source_rows(rows: list[dict[str, Any]], *, max_source_sentences: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        sentence = _normalize_candidate_text(row.get("sentence"))
        if not sentence or _is_noise_sentence(sentence):
            continue
        canonical = _canonical_statement(sentence)
        if canonical in seen:
            continue
        seen.add(canonical)
        source_id = f"src_{len(out) + 1:04d}"
        out.append(
            {
                "source_id": source_id,
                "section": _normalize_section_label(str(row.get("section") or "")),
                "modality": str(row.get("modality") or "text").strip().lower() or "text",
                "anchor": str(row.get("anchor") or "").strip(),
                "sentence": sentence[:MAX_TEXT_CHARS],
                "source_order": int(row.get("source_order", len(out)) or len(out)),
            }
        )
        if len(out) >= max_source_sentences:
            break
    return out


def _extract_stage_items(
    *,
    parsed_chunks: list[dict[str, Any]],
    summary_json: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    stages: dict[str, list[dict[str, Any]]] = {stage: [] for stage in AUDIT_STAGES}
    stages["parsed_chunks"] = _stage_items_from_parsed_chunks(parsed_chunks)
    modalities = summary_json.get("modalities", {}) if isinstance(summary_json, dict) else {}
    if isinstance(modalities, dict):
        stages["text_packets"] = _stage_items_from_packets(_modality_findings(modalities, "text"), "text_packets", "text")
        stages["table_packets"] = _stage_items_from_packets(_modality_findings(modalities, "table"), "table_packets", "table")
        stages["figure_packets"] = _stage_items_from_packets(_modality_findings(modalities, "figure"), "figure_packets", "figure")
        stages["supplement_packets"] = _stage_items_from_packets(
            _modality_findings(modalities, "supplement"),
            "supplement_packets",
            "supplement",
        )
    stages["sections_extracted"] = _stage_items_from_section_rows(
        summary_json.get("sections_extracted", {}) if isinstance(summary_json, dict) else {},
        "sections_extracted",
    )
    stages["sections_compact"] = _stage_items_from_section_rows(
        summary_json.get("sections_compact", {}) if isinstance(summary_json, dict) else {},
        "sections_compact",
        require_found_status=True,
    )
    stages["sections"] = _stage_items_from_sections(summary_json.get("sections", {}) if isinstance(summary_json, dict) else {})
    stages["extractive_evidence"] = _stage_items_from_section_rows(
        summary_json.get("extractive_evidence", {}) if isinstance(summary_json, dict) else {},
        "extractive_evidence",
    )
    stages["presentation_evidence"] = _stage_items_from_section_rows(
        summary_json.get("presentation_evidence", {}) if isinstance(summary_json, dict) else {},
        "presentation_evidence",
    )
    stages["executive_report"] = _stage_items_from_executive_report(
        summary_json.get("executive_report", {}) if isinstance(summary_json, dict) else {}
    )
    return {stage: items[:MAX_STAGE_ITEMS] for stage, items in stages.items()}


def _stage_items_from_parsed_chunks(parsed_chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for idx, chunk in enumerate(parsed_chunks):
        if not isinstance(chunk, dict):
            continue
        modality = str(chunk.get("modality") or "text").strip().lower()
        if modality == "meta":
            continue
        anchor = str(chunk.get("anchor") or f"chunk:{idx}").strip()
        section = _section_from_chunk(chunk, idx=idx, total=max(1, len(parsed_chunks)))
        content = _extract_text_from_value(chunk.get("content"))
        for sentence in _split_sentences(content):
            items.append(_stage_item("parsed_chunks", section, modality, anchor, sentence, len(items)))
            if len(items) >= MAX_STAGE_ITEMS:
                return items
    return items


def _stage_items_from_packets(rows: list[dict[str, Any]], stage: str, modality: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        text = _row_statement(row)
        if not text:
            continue
        section = _normalize_section_label(
            str(row.get("section_label") or row.get("category") or row.get("section") or "")
        )
        if section == "unknown":
            section = _section_from_anchor(_first_anchor(row))
        anchor = _first_anchor(row)
        items.append(_stage_item(stage, section, modality, anchor, text, len(items)))
    return items


def _stage_items_from_section_rows(
    section_rows: Any,
    stage: str,
    *,
    require_found_status: bool = False,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if not isinstance(section_rows, dict):
        return items
    for section in SECTION_KEYS:
        rows = section_rows.get(section, [])
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            if require_found_status and row.get("status") is not None:
                if str(row.get("status") or "").strip().lower() != "found":
                    continue
            text = _row_statement(row)
            if not text:
                continue
            anchor = _first_anchor(row)
            items.append(_stage_item(stage, section, str(row.get("source_modality") or "text"), anchor, text, len(items)))
    return items


def _stage_items_from_sections(sections: Any) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if not isinstance(sections, dict):
        return items
    for section in SECTION_KEYS:
        block = sections.get(section, {})
        rows = block.get("items", []) if isinstance(block, dict) else []
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            text = _row_statement(row)
            if not text:
                continue
            items.append(
                _stage_item(
                    "sections",
                    section,
                    str(row.get("source_modality") or "text"),
                    _first_anchor(row),
                    text,
                    len(items),
                )
            )
    return items


def _stage_items_from_executive_report(report: Any) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if not isinstance(report, dict):
        return items
    for row in report.get("sections", []) if isinstance(report.get("sections", []), list) else []:
        if not isinstance(row, dict):
            continue
        section = _normalize_section_label(str(row.get("section") or ""))
        summary = _normalize_candidate_text(row.get("summary"))
        if summary:
            for sentence in _split_sentences(summary):
                items.append(_stage_item("executive_report", section, "text", "", sentence, len(items)))
        bullets = row.get("bullets", [])
        if isinstance(bullets, list):
            for bullet in bullets:
                if not isinstance(bullet, dict):
                    continue
                text = _normalize_candidate_text(bullet.get("text"))
                if not text:
                    continue
                anchors = bullet.get("anchors", [])
                anchor = str(anchors[0]).strip() if isinstance(anchors, list) and anchors else ""
                items.append(_stage_item("executive_report", section, "text", anchor, text, len(items)))
    if not items:
        overview = _normalize_candidate_text(report.get("overview"))
        for sentence in _split_sentences(overview):
            items.append(_stage_item("executive_report", "unknown", "text", "", sentence, len(items)))
    return items


def _score_source_sentences(
    source_sentences: list[dict[str, Any]],
    stages: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for source in source_sentences:
        source_section = _normalize_section_label(str(source.get("section") or ""))
        stage_status: dict[str, str] = {}
        best_matches: dict[str, dict[str, Any]] = {}
        for stage in AUDIT_STAGES:
            best = _best_stage_match(source, stages.get(stage, []))
            if best["score"] >= MATCH_THRESHOLD:
                target_section = _normalize_section_label(str(best.get("section") or ""))
                if _section_matches(source_section, target_section):
                    status = STATUS_PRESENT_CORRECT
                else:
                    status = STATUS_PRESENT_WRONG
                best_matches[stage] = {
                    "score": round(float(best["score"]), 3),
                    "section": target_section,
                    "anchor": str(best.get("anchor") or ""),
                    "text": _truncate_match_text(str(best.get("text") or "")),
                }
            else:
                status = STATUS_MISSING
                if best["score"] > 0:
                    best_matches[stage] = {
                        "score": round(float(best["score"]), 3),
                        "section": _normalize_section_label(str(best.get("section") or "")),
                        "anchor": str(best.get("anchor") or ""),
                        "text": _truncate_match_text(str(best.get("text") or "")),
                    }
            stage_status[stage] = status
        scored.append(
            {
                **source,
                "stage_status": stage_status,
                "first_lost_after": _first_lost_stage(stage_status, modality=str(source.get("modality") or "text")),
                "best_matches": best_matches,
            }
        )
    return scored


def _best_stage_match(source: dict[str, Any], items: list[dict[str, Any]]) -> dict[str, Any]:
    best: dict[str, Any] = {"score": 0.0, "section": "", "anchor": "", "text": ""}
    source_text = str(source.get("sentence") or "")
    source_anchor = str(source.get("anchor") or "")
    for item in items:
        score = _lexical_match_score(source_text, str(item.get("text") or ""))
        if source_anchor and _anchor_matches(source_anchor, str(item.get("anchor") or "")):
            score = min(1.0, score + 0.08)
        if score > float(best["score"]):
            best = {**item, "score": score}
    return best


def _lexical_match_score(left: str, right: str) -> float:
    left_tokens = _keyword_tokens(left)
    right_tokens = _keyword_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    left_set = set(left_tokens)
    right_set = set(right_tokens)
    overlap = len(left_set & right_set)
    if overlap <= 0:
        return 0.0
    precision = overlap / len(right_set)
    recall = overlap / len(left_set)
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    left_canonical = " ".join(left_tokens)
    right_canonical = " ".join(right_tokens)
    containment = 0.0
    if len(left_tokens) >= 6 and len(right_tokens) >= 6:
        if left_canonical in right_canonical or right_canonical in left_canonical:
            containment = 0.92
    return max(f1, containment)


def _stage_metrics(source_sentences: list[dict[str, Any]], stages: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    total = len(source_sentences)
    metrics: list[dict[str, Any]] = []
    for stage in AUDIT_STAGES:
        retained = {
            str(row.get("source_id"))
            for row in source_sentences
            if row.get("stage_status", {}).get(stage) != STATUS_MISSING
        }
        wrong = {
            str(row.get("source_id"))
            for row in source_sentences
            if row.get("stage_status", {}).get(stage) == STATUS_PRESENT_WRONG
        }
        lost_here = [row for row in source_sentences if str(row.get("first_lost_after") or "") == stage]
        metrics.append(
            {
                "stage": stage,
                "stage_item_count": len(stages.get(stage, [])),
                "retained_count": len(retained),
                "retained_rate": round((len(retained) / total), 3) if total else 0.0,
                "lost_here_count": len(lost_here),
                "cumulative_lost_count": max(0, total - len(retained)),
                "wrong_section_count": len(wrong),
                "wrong_section_rate": round((len(wrong) / len(retained)), 3) if retained else 0.0,
            }
        )
    return metrics


def _first_loss_counts(source_sentences: list[dict[str, Any]]) -> dict[str, int]:
    counts = {stage: 0 for stage in AUDIT_STAGES}
    counts["retained_all_stages"] = 0
    for row in source_sentences:
        key = str(row.get("first_lost_after") or "retained_all_stages")
        counts[key] = counts.get(key, 0) + 1
    return counts


def _section_metrics(source_sentences: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for section in SECTION_KEYS + ["unknown"]:
        rows = [row for row in source_sentences if _normalize_section_label(str(row.get("section") or "")) == section]
        if not rows:
            continue
        stage_rates: dict[str, Any] = {}
        for stage in AUDIT_STAGES:
            retained = [row for row in rows if row.get("stage_status", {}).get(stage) != STATUS_MISSING]
            wrong = [row for row in rows if row.get("stage_status", {}).get(stage) == STATUS_PRESENT_WRONG]
            stage_rates[stage] = {
                "retained_count": len(retained),
                "retained_rate": round((len(retained) / len(rows)), 3) if rows else 0.0,
                "cumulative_lost_count": max(0, len(rows) - len(retained)),
                "wrong_section_count": len(wrong),
            }
        final_payload = stage_rates.get("executive_report", {})
        out[section] = {
            "source_sentence_count": len(rows),
            "final_retained_count": int(final_payload.get("retained_count", 0) or 0),
            "final_retained_rate": float(final_payload.get("retained_rate", 0.0) or 0.0),
            "final_cumulative_lost_count": int(final_payload.get("cumulative_lost_count", 0) or 0),
            "stage_metrics": stage_rates,
        }
    return out


def _first_lost_stage(stage_status: dict[str, str], *, modality: str) -> str | None:
    previous_retained = True
    for stage in _source_stage_path(modality):
        retained = stage_status.get(stage) != STATUS_MISSING
        if previous_retained and not retained:
            return stage
        previous_retained = retained
    return None


def _source_stage_path(modality: str) -> list[str]:
    normalized = str(modality or "text").strip().lower()
    if normalized == "table":
        packet_stage = "table_packets"
    elif normalized == "figure":
        packet_stage = "figure_packets"
    elif normalized in {"supp", "supplement"}:
        packet_stage = "supplement_packets"
    else:
        packet_stage = "text_packets"
    return [
        "parsed_chunks",
        packet_stage,
        "sections_extracted",
        "sections_compact",
        "sections",
        "extractive_evidence",
        "presentation_evidence",
        "executive_report",
    ]


def _stage_item(stage: str, section: str, modality: str, anchor: str, text: str, index: int) -> dict[str, Any]:
    return {
        "stage": stage,
        "item_id": f"{stage}:{index}",
        "section": _normalize_section_label(section),
        "modality": str(modality or "text").strip().lower() or "text",
        "anchor": str(anchor or "").strip(),
        "text": _normalize_candidate_text(text)[:MAX_TEXT_CHARS],
    }


def _modality_findings(modalities: dict[str, Any], key: str) -> list[dict[str, Any]]:
    payload = modalities.get(key, {})
    if not isinstance(payload, dict):
        return []
    rows = payload.get("findings", [])
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _row_statement(row: dict[str, Any]) -> str:
    for key in ("statement", "summary", "text", "claim", "value"):
        value = _normalize_candidate_text(row.get(key))
        if value:
            return value
    return ""


def _first_anchor(row: dict[str, Any]) -> str:
    for key in ("anchor", "source_anchor"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    for key in ("evidence_refs", "anchors", "evidence"):
        value = row.get(key)
        if isinstance(value, list) and value:
            return str(value[0]).strip()
    return ""


def _section_from_chunk(chunk: dict[str, Any], *, idx: int, total: int) -> str:
    meta = _parse_meta(chunk.get("meta"))
    for key in ("section_norm", "section_label", "section", "section_raw_title"):
        section = _normalize_section_label(str(meta.get(key) or ""))
        if section != "unknown":
            return section
    section = _section_from_anchor(str(chunk.get("anchor") or ""))
    if section != "unknown":
        return section
    content_section = _normalize_section_label(_extract_text_from_value(chunk.get("content"))[:120])
    if content_section != "unknown":
        return content_section
    return _position_section(idx, total)


def _section_from_anchor(anchor: str) -> str:
    return _normalize_section_label(str(anchor or "").replace("_", " ").replace(":", " "))


def _normalize_section_label(value: str) -> str:
    text = _normalize_text(value).lower()
    if not text:
        return "unknown"
    if "conclusion" in text or "concluding" in text:
        return "conclusion"
    if "discussion" in text or "limitation" in text or "implication" in text:
        return "discussion"
    if "result" in text or "finding" in text or "outcome" in text:
        return "results"
    if any(token in text for token in ("method", "material", "participant", "procedure", "protocol", "analysis", "design")):
        return "methods"
    if any(token in text for token in ("abstract", "intro", "background", "objective", "aim", "hypoth", "rationale")):
        return "introduction"
    return "unknown"


def _position_section(idx: int, total: int) -> str:
    if total <= 0:
        return "unknown"
    ratio = idx / max(1, total - 1)
    if ratio <= 0.22:
        return "introduction"
    if ratio <= 0.52:
        return "methods"
    if ratio <= 0.72:
        return "results"
    if ratio <= 0.90:
        return "discussion"
    return "conclusion"


def _section_matches(source_section: str, target_section: str) -> bool:
    if source_section == "unknown" or target_section == "unknown":
        return True
    return source_section == target_section


def _anchor_matches(left: str, right: str) -> bool:
    left_norm = _normalize_text(left).lower()
    right_norm = _normalize_text(right).lower()
    if not left_norm or not right_norm:
        return False
    return left_norm == right_norm or left_norm in right_norm or right_norm in left_norm


def _split_sentences(text: Any) -> list[str]:
    clean = _normalize_candidate_text(text)
    if not clean:
        return []
    parts = SENTENCE_SPLIT_RE.split(clean)
    out: list[str] = []
    for part in parts:
        sentence = _normalize_candidate_text(part)
        if not sentence:
            continue
        if _is_noise_sentence(sentence):
            continue
        out.append(sentence)
    return out


def _is_noise_sentence(text: str) -> bool:
    clean = _normalize_text(text)
    if not clean:
        return True
    if len(clean) < 35:
        return True
    tokens = TOKEN_RE.findall(clean.lower())
    if len(tokens) < 7:
        return True
    if len(clean) > 1200:
        return True
    if REFERENCE_NOISE_RE.search(clean) or HEADER_FOOTER_RE.search(clean):
        return True
    digit_chars = sum(1 for char in clean if char.isdigit())
    if digit_chars / max(1, len(clean)) > 0.28:
        return True
    if len(set(tokens)) <= 4:
        return True
    return False


def _keyword_tokens(text: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(_normalize_text(text).lower()) if token not in STOPWORDS and len(token) > 2]


def _canonical_statement(text: str) -> str:
    return " ".join(_keyword_tokens(text))


def _normalize_candidate_text(value: Any) -> str:
    text = _extract_text_from_value(value)
    text = html.unescape(text)
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _extract_text_from_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and stripped[0] in "[{":
            try:
                return _extract_text_from_value(json.loads(stripped))
            except Exception:
                return stripped
        return stripped
    if isinstance(value, dict):
        parts: list[str] = []
        for key in ("text", "caption", "legend", "title", "summary", "statement", "content"):
            if key in value:
                parts.append(_extract_text_from_value(value.get(key)))
        if not parts:
            for nested in value.values():
                if isinstance(nested, (dict, list)):
                    parts.append(_extract_text_from_value(nested))
        return " ".join(part for part in parts if part)
    if isinstance(value, list):
        return " ".join(_extract_text_from_value(item) for item in value)
    return str(value)


def _parse_meta(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _read_text_asset(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8", "latin-1"):
        try:
            return raw.decode(encoding)
        except Exception:
            continue
    return raw.decode("utf-8", errors="ignore")


def _html_to_text(text: str) -> str:
    clean = HTML_TAG_RE.sub("\n", text)
    clean = re.sub(r"\n{3,}", "\n\n", clean)
    return clean


def _safe_anchor(value: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value or "").strip())
    return safe or "source"


def _truncate_match_text(text: str) -> str:
    clean = _normalize_candidate_text(text)
    if len(clean) <= 220:
        return clean
    return clean[:217].rstrip() + "..."


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
