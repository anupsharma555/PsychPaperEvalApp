from __future__ import annotations

import json
from functools import lru_cache
import os
from pathlib import Path
import re
from typing import Any, Optional
from urllib.parse import urljoin
import zipfile

from bs4 import BeautifulSoup
import pandas as pd
from sqlmodel import Session, select

from app.db.models import Asset, Chunk, Document
from app.services.analysis.utils import extract_refs_from_text
from app.services.storage import artifacts_dir
from app.services.validated_pipeline import parse_pdf_validated
from app.core.config import settings


STRUCTURED_ABSTRACT_PREFIX_RE = re.compile(
    r"(?i)\b(objective|objectives|background|aim|aims|method|methods|design|results|conclusion|conclusions)\s*:"
)
DOCLING_HEADING_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9\s&,()/:\-]{1,90}$"
)
DOCLING_SECTION_HEADING_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+){0,3}\s+)?(?P<label>"
    r"introduction|background|objective|objectives|aim|aims|purpose|method|methods|"
    r"materials?\s+and\s+methods?|materials?|participants?|analysis|results|"
    r"findings|discussion|conclusion|conclusions|limitation|implication|"
    r"abstract|summary|supplementary|supplementary\s+materials?|supplement|appendix|references?)\b[\s\-–—:.;\)]*\s*$",
    re.IGNORECASE,
)
DOCLING_SECTION_FALLBACK_HINT_RE = re.compile(
    r"\b(introduction|background|objective|aim|method|methods|materials|analysis|results|"
    r"finding|discussion|conclusion|summary|limitation)\b",
    re.IGNORECASE,
)
SECTION_HEADING_CANONICAL_MAP: list[tuple[str, str]] = [
    ("materials and methods", "methods"),
    ("materials", "methods"),
    ("analysis", "methods"),
    ("participants", "methods"),
    ("objective", "introduction"),
    ("objectives", "introduction"),
    ("aim", "introduction"),
    ("aims", "introduction"),
    ("purpose", "introduction"),
    ("background", "introduction"),
    ("abstract", "introduction"),
    ("introduction", "introduction"),
    ("method", "methods"),
    ("methods", "methods"),
    ("finding", "results"),
    ("findings", "results"),
    ("result", "results"),
    ("results", "results"),
    ("discussion", "discussion"),
    ("limitation", "discussion"),
    ("limitations", "discussion"),
    ("implication", "discussion"),
    ("conclusion", "conclusion"),
    ("conclusions", "conclusion"),
    ("summary", "conclusion"),
    ("supplementary materials", "unknown"),
    ("supplementary material", "unknown"),
    ("supplementary", "unknown"),
    ("supplement", "unknown"),
    ("appendix", "unknown"),
    ("references", "unknown"),
]
SECTION_HEADING_CANONICAL_INDEX: dict[str, str] = {alias: label for alias, label in SECTION_HEADING_CANONICAL_MAP}



def _clean_heading_text(value: str) -> str:
    text = " ".join(str(value or "").split()).strip()
    if text.endswith("."):
        text = text[:-1].strip()
    return text


def _numeric_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _extract_docling_heading_style_signal(item: Any, heading: str = "") -> tuple[float, Optional[int]]:
    style_score = 0.0
    heading_level: Optional[int] = None
    heading_text = _clean_heading_text(heading or getattr(item, "text", "") or "")

    for key in ("heading_level", "header_level", "outline_level", "level"):
        value = _numeric_value(getattr(item, key, None))
        if value is None:
            continue
        level = int(value)
        if level < 0:
            continue
        if heading_level is None or level < heading_level:
            heading_level = level
    if heading_level is not None:
        depth_bonus = max(0.0, min(0.26, 0.26 - (0.03 * min(heading_level, 6))))
        style_score += 0.20 + depth_bonus

    for key, bonus in (
        ("is_heading", 0.18),
        ("is_title", 0.18),
        ("bold", 0.09),
        ("is_bold", 0.09),
    ):
        if bool(getattr(item, key, False)):
            style_score += bonus

    prov_blocks = getattr(item, "prov", None) if isinstance(getattr(item, "prov", None), list) else []
    max_font_size = 0.0
    for source in [item, *prov_blocks]:
        for key in ("font_size", "font_size_pt", "size", "text_size"):
            value = _numeric_value(getattr(source, key, None))
            if value is not None:
                max_font_size = max(max_font_size, value)
    if max_font_size >= 15.0:
        style_score += 0.16
    elif max_font_size >= 13.0:
        style_score += 0.11
    elif max_font_size >= 11.8:
        style_score += 0.06

    style_text_parts: list[str] = []
    for source in [item, *prov_blocks]:
        for key in ("style", "text_style", "class_name", "label", "name", "role", "kind"):
            value = getattr(source, key, None)
            if isinstance(value, str) and value.strip():
                style_text_parts.append(value.strip().lower())
    style_text = " ".join(style_text_parts)
    if any(token in style_text for token in ("heading", "header", "title", "subtitle")):
        style_score += 0.14
    if "bold" in style_text:
        style_score += 0.06
    if re.search(r"\bh[1-6]\b", style_text):
        style_score += 0.08

    if heading_text:
        tokens = heading_text.split()
        if len(tokens) <= 6:
            style_score += 0.08
        if heading_text.isupper() and len(tokens) <= 10:
            style_score += 0.10
        if _extract_explicit_section_keyword(heading_text):
            style_score += 0.16

    return min(1.0, max(0.0, style_score)), heading_level


def _inline_heading_style_score(heading: str, raw_text: str) -> float:
    clean_heading = _clean_heading_text(heading)
    if not clean_heading:
        return 0.0
    score = 0.0
    if DOCLING_SECTION_HEADING_RE.match(clean_heading):
        score += 0.42
    if _extract_explicit_section_keyword(clean_heading):
        score += 0.26
    tokens = clean_heading.split()
    if len(tokens) <= 6:
        score += 0.10
    if clean_heading.isupper() and len(tokens) <= 10:
        score += 0.10
    lines = [line.strip() for line in str(raw_text or "").replace("\r", "\n").split("\n") if line.strip()]
    if lines and _clean_heading_text(lines[0]) == clean_heading:
        score += 0.10
        if len(lines) == 1:
            score += 0.12
        elif len(lines[1]) > 120:
            score += 0.05
    return min(1.0, max(0.0, score))


def _position_section_from_progress(idx: int, total: int) -> str:
    if total <= 0:
        return "unknown"
    ratio = idx / float(max(1, total - 1))
    if ratio <= 0.20:
        return "introduction"
    if ratio <= 0.52:
        return "methods"
    if ratio <= 0.74:
        return "results"
    if ratio <= 0.90:
        return "discussion"
    return "conclusion"


def _looks_like_heading_line(value: str) -> bool:
    text = _clean_heading_text(value)
    if not text:
        return False
    if len(text) < 4 or len(text) > 140:
        return False
    if re.search(r"[.!?]$", text):
        return False
    if DOCLING_SECTION_HEADING_RE.match(text):
        return True
    if text.endswith(":") and DOCLING_SECTION_FALLBACK_HINT_RE.search(text):
        return True
    return False


def _extract_heading_like_from_text(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    lines = [line.strip() for line in text.replace("\r", "\n").split("\n")]
    if not lines:
        return ""
    first = lines[0]
    if len(lines) > 1:
        compact = f"{first} {lines[1]}" if len(first) < 80 else first
    else:
        compact = first
    match = DOCLING_SECTION_HEADING_RE.match(first)
    if match:
        return _clean_heading_text(match.group(0))
    if compact:
        stripped = _clean_heading_text(compact)
        if _looks_like_heading_line(stripped) and " " in stripped:
            return stripped
    return ""


def _extract_explicit_section_keyword(value: str) -> str:
    text = _clean_heading_text(value)
    if not text:
        return ""
    normalized = re.sub(r"^\d+(?:\.\d+){0,3}\s*", "", text.lower())
    normalized = re.sub(r"\s*[:\-–—.);,]+$", "", normalized).strip()
    normalized = re.sub(r"\s+", " ", normalized)
    if not normalized:
        return ""
    if len(normalized.split()) > 18:
        return ""
    for alias in SECTION_HEADING_CANONICAL_INDEX:
        alias_clean = alias.lower()
        if normalized == alias_clean or re.match(rf"^{re.escape(alias_clean)}\b", normalized):
            return alias_clean
    return ""


def _extract_heading_body_tail(value: str, heading: str) -> str:
    raw = str(value or "")
    if not raw or not heading:
        return raw.strip()
    lines = [line.strip() for line in raw.replace("\r", "\n").split("\n")]
    lines = [line for line in lines if line]
    if not lines:
        return ""
    normalized_heading = _clean_heading_text(heading)
    if _clean_heading_text(lines[0]) != normalized_heading:
        return raw.strip()
    if len(lines) == 1:
        return ""
    return " ".join(_clean_heading_text(line) for line in lines[1:] if line).strip()


def _section_slug(value: str) -> str:
    text = _clean_heading_text(value).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or "body"


def _looks_like_heading_like(value: str) -> bool:
    text = _clean_heading_text(value)
    if not text:
        return False
    lowered = text.lower()
    if _normalize_section_title(lowered) != "unknown":
        return True
    word_count = len(text.split())
    if word_count == 1 and len(text) <= 4:
        return False
    if word_count > 16:
        return False
    if not DOCLING_HEADING_RE.search(text):
        return False
    if re.search(r"\b(and|of|to|from|that|this|with|using|we|were|are|is|was|were)\b", lowered):
        return False
    return True


def _looks_like_docling_heading(value: str) -> bool:
    text = _clean_heading_text(value)
    if not text or len(text) > 110:
        return False
    if text.endswith((".", ";", ",")):
        return False
    if _extract_explicit_section_keyword(text):
        return True
    if not _looks_like_heading_like(text):
        return False
    return True


def _extract_docling_heading_from_item(item: Any) -> str:
    candidates: list[str] = []
    for key in (
        "label",
        "title",
        "heading",
        "section",
        "name",
        "section_title",
        "raw_title",
        "text",
    ):
        value = getattr(item, key, None)
        if isinstance(value, str):
            candidates.append(value)
    if isinstance(getattr(item, "prov", None), list):
        for block in item.prov:
            for key in ("text", "label", "name", "section", "title", "value"):
                value = getattr(block, key, None)
                if isinstance(value, str):
                    candidates.append(value)
    for key in ("heading", "title", "section"):
        parent = getattr(item, "parent", None)
        if parent is None:
            continue
        try:
            value = getattr(parent, key, None)
        except Exception:
            value = None
        if isinstance(value, str):
            candidates.append(value)

    heading_like_text = _extract_heading_like_text_from_item(item)
    if heading_like_text:
        candidates.append(heading_like_text)

    seen: set[str] = set()
    for candidate in candidates:
        text = _clean_heading_text(candidate)
        if not text:
            continue
        lowered = text.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        keyword = _extract_explicit_section_keyword(text)
        if keyword and keyword in SECTION_HEADING_CANONICAL_INDEX:
            return text
        if _looks_like_docling_heading(text):
            return text
    return ""


def _normalize_section_title(title: str) -> str:
    text = " ".join(str(title or "").split()).strip().lower()
    if not text:
        return "unknown"
    explicit_keyword = _extract_explicit_section_keyword(text)
    if explicit_keyword:
        explicit_label = SECTION_HEADING_CANONICAL_INDEX.get(explicit_keyword, "unknown")
        if explicit_label != "unknown":
            return explicit_label
    if "conclusion" in text or "concluding" in text:
        return "conclusion"
    if "discussion" in text or "limitation" in text or "implication" in text:
        return "discussion"
    if "result" in text or "finding" in text:
        return "results"
    if any(token in text for token in ("method", "material", "participant", "procedure", "analysis", "design", "protocol", "acquisition")):
        return "methods"
    if any(token in text for token in ("intro", "background", "objective", "aim", "rationale", "hypoth", "abstract")):
        return "introduction"
    return "unknown"


def _extract_heading_like_text_from_item(item: Any) -> str:
    text = str(getattr(item, "text", "") or "").strip()
    if not text:
        return ""
    # Short, label-like first line is often a heading emitted as plain text by docling.
    first_line = re.split(r"[\n\r]+", text)[0].strip()
    if not first_line or len(first_line) > 120:
        return ""
    if not _looks_like_heading_like(first_line):
        return ""
    return first_line


def _split_structured_abstract(text: str) -> list[tuple[str, str]]:
    clean = " ".join(str(text or "").split()).strip()
    if not clean:
        return []
    matches = list(STRUCTURED_ABSTRACT_PREFIX_RE.finditer(clean))
    if not matches:
        return [("body", clean)]
    out: list[tuple[str, str]] = []
    for idx, match in enumerate(matches):
        label = str(match.group(1) or "").strip() or "body"
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(clean)
        statement = clean[start:end].strip(" -;,:")
        if statement:
            out.append((label, statement))
    return out or [("body", clean)]


def parse_document_assets(session: Session, document_id: int) -> dict[str, int]:
    counts = {"text": 0, "table": 0, "figure": 0, "supp": 0}
    document = session.get(Document, document_id)
    base_url = str(document.source_url or "").strip() if document else ""
    assets = session.exec(select(Asset).where(Asset.document_id == document_id)).all()
    for asset in assets:
        file_path = Path(asset.path)
        if not file_path.exists():
            continue
        if asset.kind == "supp":
            counts["supp"] += 1
        ext = file_path.suffix.lower()
        if ext in {".csv", ".tsv", ".xlsx", ".xls"}:
            counts["table"] += _parse_tabular_file(session, document_id, asset, file_path)
        elif ext in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            counts["figure"] += _parse_image_file(session, document_id, asset, file_path)
        elif ext in {".txt"}:
            counts["text"] += _parse_text_file(session, document_id, asset, file_path)
        elif ext in {".html", ".htm", ".xhtml"} or _looks_like_html_file(file_path):
            counts = _parse_html_file(session, document_id, asset, file_path, counts, base_url=base_url)
        elif ext == ".zip":
            counts = _parse_zip_file(session, document_id, asset, file_path, counts)
        else:
            if file_path.suffix.lower() == ".pdf" and settings.parser_engine == "validated":
                counts = parse_pdf_validated(session, document_id, asset, file_path, counts)
            else:
                # Default to docling for PDF/DOCX/PPTX/HTML and unknowns
                counts = _parse_with_docling(session, document_id, asset, file_path, counts)
        if (
            not settings.retain_source_files
            and asset.kind == "main"
            and file_path.suffix.lower() == ".pdf"
        ):
            try:
                file_path.unlink()
            except Exception:
                pass
    return counts


def _looks_like_html_file(path: Path) -> bool:
    try:
        head = path.read_bytes()[:4096]
    except Exception:
        return False
    text = head.decode("utf-8", errors="ignore").lower()
    html_markers = ("<!doctype html", "<html", "<body", "<article", "<head")
    return any(marker in text for marker in html_markers)


def _parse_text_file(
    session: Session,
    document_id: int,
    asset: Asset,
    path: Path,
    anchor_prefix: str = "",
) -> int:
    text = path.read_text(errors="ignore")
    chunk = Chunk(
        document_id=document_id,
        asset_id=asset.id,
        anchor=f"{anchor_prefix}file:{path.name}",
        modality="text",
        content=text,
        meta=json.dumps(
            {
                "source": "file",
                "asset_kind": asset.kind,
                "filename": path.name,
                "section_raw_title": "file",
                "section_norm": "unknown",
                "section_path": f"file/{path.name}",
            }
        ),
    )
    session.add(chunk)
    session.commit()
    return 1


def _clean_text(value: str) -> str:
    return " ".join(str(value or "").split()).strip()


def _resolve_html_href(href: str, base_url: str) -> str:
    raw = str(href or "").strip()
    if not raw:
        return ""
    if raw.startswith(("javascript:", "mailto:", "tel:")):
        return ""
    if raw.startswith("http://") or raw.startswith("https://"):
        return raw
    if raw.startswith("#"):
        return ""
    if base_url:
        return urljoin(base_url, raw)
    return raw


def _looks_like_content_image(url: str, alt_text: str = "") -> bool:
    value = str(url or "").lower()
    alt = str(alt_text or "").lower()
    blocked = ("logo", "icon", "favicon", "sprite", "social", "footer")
    if any(token in value for token in blocked) and not any(token in alt for token in ("figure", "table")):
        return False
    image_exts = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".tif", ".tiff", ".svg")
    has_image_ext = value.split("?", 1)[0].endswith(image_exts)
    if has_image_ext and ("/cms/" in value or "/asset/" in value):
        return True
    if any(token in value for token in ("/figure", "fig", "table")):
        return True
    if has_image_ext and len(alt) > 20:
        return True
    return False


def _parse_html_file(
    session: Session,
    document_id: int,
    asset: Asset,
    path: Path,
    counts: dict[str, int],
    *,
    base_url: str = "",
) -> dict[str, int]:
    try:
        html = path.read_text(errors="ignore")
    except Exception:
        return counts
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "template", "svg"]):
        tag.decompose()

    root = soup.find("article") or soup.find("main") or soup.body or soup

    text_added = 0
    seen_text: set[str] = set()
    current_heading = "body"
    for idx, node in enumerate(root.find_all(["h1", "h2", "h3", "h4", "p", "li"])):
        node_name = str(getattr(node, "name", "") or "").lower()
        text = _clean_text(node.get_text(" ", strip=True))
        if node_name in {"h1", "h2", "h3", "h4"}:
            if text:
                current_heading = text
            continue
        if len(text) < 30:
            continue
        dedupe_key = text[:500].lower()
        if dedupe_key in seen_text:
            continue
        seen_text.add(dedupe_key)
        section_title = current_heading or "body"
        parts = _split_structured_abstract(text)
        for part_idx, (part_title, part_text) in enumerate(parts):
            section_raw = part_title if len(parts) > 1 else section_title
            section_norm = _normalize_section_title(section_raw or section_title)
            anchor = f"html:para:{idx}" if part_idx == 0 else f"html:para:{idx}:{part_idx}"
            session.add(
                Chunk(
                    document_id=document_id,
                    asset_id=asset.id,
                    anchor=anchor,
                    modality="text",
                    content=part_text,
                    meta=json.dumps(
                        {
                            "source": "html",
                            "asset_kind": asset.kind,
                            "filename": path.name,
                            "section": section_title,
                            "section_raw_title": section_raw,
                            "section_norm": section_norm,
                            "section_path": f"{section_title}/{text_added}",
                            "paragraph_index": text_added,
                        }
                    ),
                )
            )
            counts["text"] += 1
            text_added += 1
            if text_added >= 400:
                break
        if text_added >= 400:
            break

    if text_added == 0:
        text = _clean_text(root.get_text(" ", strip=True))
        if text:
            session.add(
                Chunk(
                    document_id=document_id,
                    asset_id=asset.id,
                    anchor=f"file:{path.name}",
                    modality="text",
                    content=text,
                    meta=json.dumps(
                        {
                            "source": "html",
                            "asset_kind": asset.kind,
                            "filename": path.name,
                            "section": "body",
                            "section_raw_title": "body",
                            "section_norm": "unknown",
                            "section_path": "body/0",
                            "paragraph_index": 0,
                        }
                    ),
                )
            )
            counts["text"] += 1

    figure_idx = 0
    seen_figure_urls: set[str] = set()
    for figure in root.find_all("figure"):
        image = figure.find("img", src=True)
        image_url = _resolve_html_href(image.get("src") if image else "", base_url)
        caption = _clean_text(figure.find("figcaption").get_text(" ", strip=True) if figure.find("figcaption") else "")
        figure_id = str(figure.get("id") or "").strip()
        if not image_url and not caption:
            continue
        figure_idx += 1
        if image_url:
            seen_figure_urls.add(image_url)
        session.add(
            Chunk(
                document_id=document_id,
                asset_id=asset.id,
                anchor=figure_id or f"figure:html:{figure_idx}",
                modality="figure",
                content=caption or _clean_text(image.get("alt") if image else ""),
                meta=json.dumps(
                    {
                        "source": "html",
                        "asset_kind": asset.kind,
                        "caption": caption,
                        "figure_id": figure_id,
                        "source_url": image_url,
                        "source_page_url": base_url,
                    }
                ),
            )
        )
        counts["figure"] += 1

    for image in root.find_all("img", src=True):
        if image.find_parent("figure") is not None:
            continue
        image_url = _resolve_html_href(image.get("src"), base_url)
        alt_text = _clean_text(image.get("alt") or "")
        if not image_url or image_url in seen_figure_urls:
            continue
        if not _looks_like_content_image(image_url, alt_text):
            continue
        figure_idx += 1
        seen_figure_urls.add(image_url)
        session.add(
            Chunk(
                document_id=document_id,
                asset_id=asset.id,
                anchor=f"figure:image:{figure_idx}",
                modality="figure",
                content=alt_text,
                meta=json.dumps(
                    {
                        "source": "html",
                        "asset_kind": asset.kind,
                        "caption": alt_text,
                        "figure_id": f"img_{figure_idx}",
                        "source_url": image_url,
                        "source_page_url": base_url,
                    }
                ),
            )
        )
        counts["figure"] += 1

    table_idx = 0
    for table in root.find_all("table"):
        rows: list[list[str]] = []
        for tr in table.find_all("tr"):
            cells = [_clean_text(cell.get_text(" ", strip=True)) for cell in tr.find_all(["th", "td"])]
            if cells and any(cells):
                rows.append(cells)
        if not rows:
            continue

        table_idx += 1
        head_cells = [_clean_text(cell.get_text(" ", strip=True)) for cell in table.find_all("th")]
        if head_cells:
            columns = head_cells
            data_rows = rows[1:] if len(rows) > 1 else []
        else:
            max_cols = max(len(row) for row in rows)
            columns = [f"col_{idx + 1}" for idx in range(max_cols)]
            data_rows = rows

        normalized_rows = []
        for row in data_rows:
            normalized = row[: len(columns)] + [""] * max(0, len(columns) - len(row))
            normalized_rows.append(normalized)

        parent_figure = table.find_parent("figure")
        table_id = str((parent_figure.get("id") if parent_figure else "") or table.get("id") or "").strip()
        caption = _clean_text(parent_figure.find("figcaption").get_text(" ", strip=True) if parent_figure and parent_figure.find("figcaption") else "")
        session.add(
            Chunk(
                document_id=document_id,
                asset_id=asset.id,
                anchor=table_id or f"table:html:{table_idx}",
                modality="table",
                content=json.dumps({"columns": columns, "data": normalized_rows}),
                meta=json.dumps(
                    {
                        "source": "html",
                        "asset_kind": asset.kind,
                        "caption": caption,
                        "table_id": table_id,
                    }
                ),
            )
        )
        counts["table"] += 1

    seen_links: set[tuple[str, str]] = set()
    figure_link_idx = 0
    table_link_idx = 0
    for link in root.find_all("a", href=True):
        href = _resolve_html_href(link.get("href"), base_url)
        if not href:
            continue
        text = _clean_text(link.get_text(" ", strip=True))
        lower = f"{href} {text}".lower()
        modality = ""
        if _looks_like_content_image(href, text) or "figure" in lower:
            modality = "figure"
        elif ("supplement" in lower or "suppl" in lower) and "table" not in lower:
            modality = "figure"
        elif "table" in lower:
            modality = "table"
        if not modality:
            continue
        key = (modality, href)
        if key in seen_links:
            continue
        seen_links.add(key)
        if modality == "figure":
            figure_link_idx += 1
            anchor = f"figure:link:{figure_link_idx}"
            counts["figure"] += 1
        else:
            table_link_idx += 1
            anchor = f"table:link:{table_link_idx}"
            counts["table"] += 1
        session.add(
            Chunk(
                document_id=document_id,
                asset_id=asset.id,
                anchor=anchor,
                modality=modality,
                content=text,
                meta=json.dumps(
                    {
                        "source": "html_link",
                        "asset_kind": asset.kind,
                        "caption": text,
                        "source_url": href,
                        "source_page_url": base_url,
                    }
                ),
            )
        )

    session.commit()
    return counts


def _parse_tabular_file(
    session: Session,
    document_id: int,
    asset: Asset,
    path: Path,
    anchor_prefix: str = "",
) -> int:
    tables = 0
    if path.suffix.lower() in {".xlsx", ".xls"}:
        sheets = pd.read_excel(path, sheet_name=None)
        for sheet_name, df in sheets.items():
            _store_table(
                session,
                document_id,
                asset,
                df,
                f"{anchor_prefix}sheet:{sheet_name}",
                extra_meta={"source": "sheet", "asset_kind": asset.kind, "filename": path.name},
            )
            tables += 1
    else:
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
        _store_table(
            session,
            document_id,
            asset,
            df,
            f"{anchor_prefix}file:{path.name}",
            extra_meta={"source": "file", "asset_kind": asset.kind, "filename": path.name},
        )
        tables += 1
    session.commit()
    return tables


def _store_table(
    session: Session,
    document_id: int,
    asset: Asset,
    df: pd.DataFrame,
    anchor: str,
    extra_meta: dict[str, Any] | None = None,
) -> None:
    content = df.to_json(orient="split")
    meta_obj: dict[str, Any] = {"rows": len(df), "cols": len(df.columns), "asset_kind": asset.kind}
    if extra_meta:
        meta_obj.update(extra_meta)
    chunk = Chunk(
        document_id=document_id,
        asset_id=asset.id,
        anchor=anchor,
        modality="table",
        content=content,
        meta=json.dumps(meta_obj),
    )
    session.add(chunk)


def _parse_image_file(
    session: Session,
    document_id: int,
    asset: Asset,
    path: Path,
    anchor_prefix: str = "",
) -> int:
    ocr_text = _ocr_image(path)
    refs = extract_refs_from_text(f"{path.stem} {ocr_text}")
    meta = {
        "path": str(path),
        "asset_kind": asset.kind,
        "source": "image_file",
        "figure_refs": sorted(refs),
        "ocr_source": "easyocr" if ocr_text else None,
        "quality_flags": [],
    }
    if ocr_text:
        meta["ocr_text"] = ocr_text
    meta = json.dumps(meta)
    chunk = Chunk(
        document_id=document_id,
        asset_id=asset.id,
        anchor=f"{anchor_prefix}image:{path.name}",
        modality="figure",
        content="",
        meta=meta,
    )
    session.add(chunk)
    session.commit()
    return 1


def _parse_with_docling(
    session: Session,
    document_id: int,
    asset: Asset,
    path: Path,
    counts: dict[str, int],
) -> dict[str, int]:
    try:
        from docling.document_converter import DocumentConverter
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import PdfFormatOption
        try:
            from docling.datamodel.base_models import InputFormat
        except Exception:
            InputFormat = None
        try:
            from docling_core.types.doc import PictureItem
        except Exception:
            PictureItem = None
    except Exception as exc:
        raise RuntimeError("Docling is required to parse this asset.") from exc

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = bool(settings.docling_enable_ocr)
    pipeline_options.do_table_structure = bool(settings.docling_table_structure_enabled)
    pipeline_options.generate_picture_images = bool(settings.docling_extract_figures)
    pipeline_options.images_scale = 2.0
    if hasattr(pipeline_options, "ocr_options") and hasattr(pipeline_options.ocr_options, "use_gpu"):
        pipeline_options.ocr_options.use_gpu = False

    format_options = {}
    if InputFormat is not None:
        try:
            format_options[InputFormat.PDF] = PdfFormatOption(pipeline_options=pipeline_options)
        except Exception:
            pass

    converter = DocumentConverter(format_options=format_options if format_options else None)
    result = converter.convert(str(path))
    doc = result.document

    fig_dir = artifacts_dir(document_id) / "figures"
    if settings.docling_extract_figures:
        fig_dir.mkdir(parents=True, exist_ok=True)

    figures_added = 0
    current_section_raw = "body"
    current_section_norm = "unknown"
    current_section_confidence = 0.0
    current_section_source = "fallback"
    current_heading_style_score = 0.0
    current_heading_level: Optional[int] = None
    current_section_anchor_token = ""
    doc_items = list(doc.iterate_items())
    total_items = len(doc_items)
    for idx, (item, _level) in enumerate(doc_items):
        anchor = _anchor_for_item(item, idx)
        item_section_raw = current_section_raw
        item_section_norm = current_section_norm
        item_section_confidence = current_section_confidence
        item_section_source = current_section_source
        item_heading_style_score = current_heading_style_score
        item_heading_level = current_heading_level
        item_section_anchor_token = current_section_anchor_token
        heading = _extract_docling_heading_from_item(item)
        if heading and _looks_like_docling_heading(heading):
            heading_style_score, heading_level = _extract_docling_heading_style_signal(item, heading)
            section_anchor_token = _extract_explicit_section_keyword(heading)
            current_section_raw = heading
            current_section_norm = _normalize_section_title(heading)
            if current_section_norm != "unknown":
                section_bonus = 0.08 if current_section_norm in {"discussion", "conclusion"} else 0.0
                current_section_confidence = min(0.999, 0.90 + section_bonus + (0.08 * heading_style_score))
                current_section_source = "heading"
            else:
                current_section_confidence = min(0.82, 0.62 + (0.20 * heading_style_score))
                current_section_source = "heading"
            current_heading_style_score = heading_style_score
            current_heading_level = heading_level
            current_section_anchor_token = section_anchor_token
            # Heading markers typically should not be emitted as standalone text packets.
            if _looks_like_heading_like(heading):
                continue
            item_section_raw = current_section_raw
            item_section_norm = current_section_norm
            item_section_confidence = current_section_confidence
            item_section_source = current_section_source
            item_heading_style_score = current_heading_style_score
            item_heading_level = current_heading_level
            item_section_anchor_token = current_section_anchor_token
        if "TextItem" in type(item).__name__ or ("TextItem" in str(type(item))):
            text = getattr(item, "text", "")
            if not text:
                continue
            inline_heading = _extract_heading_like_from_text(text)
            if inline_heading:
                tail = _extract_heading_body_tail(text, inline_heading)
                heading_from_text = _normalize_section_title(inline_heading)
                if heading_from_text != "unknown":
                    inline_style_score = _inline_heading_style_score(inline_heading, text)
                    section_anchor_token = _extract_explicit_section_keyword(inline_heading)
                    current_section_raw = inline_heading
                    current_section_norm = heading_from_text
                    section_bonus = 0.08 if heading_from_text in {"discussion", "conclusion"} else 0.0
                    current_section_confidence = min(0.97, 0.82 + section_bonus + (0.10 * inline_style_score))
                    current_section_source = "heading"
                    current_heading_style_score = inline_style_score
                    current_heading_level = None
                    current_section_anchor_token = section_anchor_token
                    item_section_raw = current_section_raw
                    item_section_norm = current_section_norm
                    item_section_confidence = current_section_confidence
                    item_section_source = current_section_source
                    item_heading_style_score = current_heading_style_score
                    item_heading_level = current_heading_level
                    item_section_anchor_token = current_section_anchor_token
                if not tail:
                    continue
                text = tail
            if item_section_norm == "unknown":
                item_section_norm = _position_section_from_progress(idx, total_items)
                item_section_raw = item_section_norm
                item_section_confidence = 0.24
                item_section_source = "position"
                item_heading_style_score = 0.0
                item_heading_level = None
                item_section_anchor_token = ""
            chunk = Chunk(
                document_id=document_id,
                asset_id=asset.id,
                anchor=anchor,
                modality="text",
                content=text,
                meta=json.dumps(
                    {
                        "source": "docling",
                        "asset_kind": asset.kind,
                        "section_raw_title": item_section_raw,
                        "section_norm": item_section_norm,
                        "section_source": item_section_source,
                        "section_confidence": item_section_confidence,
                        "heading_text": item_section_raw if item_section_source == "heading" else "",
                        "heading_style_score": item_heading_style_score,
                        "heading_level": item_heading_level if item_heading_level is not None else -1,
                        "section_anchor_token": item_section_anchor_token,
                        "section_path": f"{_section_slug(item_section_raw)}/{idx}",
                        "paragraph_index": idx,
                    }
                ),
            )
            session.add(chunk)
            counts["text"] += 1
            continue
        if settings.docling_table_structure_enabled and "TableItem" in type(item).__name__:
            try:
                df = item.export_to_dataframe(doc=doc)
                _store_table(
                    session,
                    document_id,
                    asset,
                    df,
                    anchor,
                    extra_meta={
                        "source": "docling",
                        "asset_kind": asset.kind,
                        "section_raw_title": item_section_raw if item_section_norm == "unknown" else current_section_raw,
                        "section_norm": item_section_norm,
                        "section_source": item_section_source,
                        "section_confidence": item_section_confidence,
                        "heading_text": item_section_raw if item_section_source == "heading" else "",
                        "heading_style_score": item_heading_style_score,
                        "heading_level": item_heading_level if item_heading_level is not None else -1,
                        "section_anchor_token": item_section_anchor_token,
                    },
                )
                counts["table"] += 1
            except Exception:
                continue
            continue
        if settings.docling_extract_figures and ("PictureItem" in type(item).__name__ or (PictureItem and isinstance(item, PictureItem))):
            try:
                image = item.get_image(doc)
                image_path = fig_dir / f"figure_{asset.id}_{idx}.png"
                image.save(image_path, "PNG")
                caption = getattr(item, "caption", None)
                ocr_text = _ocr_image(image_path)
                refs = extract_refs_from_text(f"{caption or ''} {ocr_text or ''}")
                meta_obj = {
                    "path": str(image_path),
                    "caption": caption,
                    "asset_kind": asset.kind,
                    "source": "docling",
                    "section_raw_title": item_section_raw if item_section_norm == "unknown" else current_section_raw,
                    "section_norm": item_section_norm,
                    "section_source": item_section_source,
                    "section_confidence": item_section_confidence,
                    "heading_text": item_section_raw if item_section_source == "heading" else "",
                    "heading_style_score": item_heading_style_score,
                    "heading_level": item_heading_level if item_heading_level is not None else -1,
                    "section_anchor_token": item_section_anchor_token,
                    "figure_refs": sorted(refs),
                    "ocr_source": "easyocr" if ocr_text else None,
                    "quality_flags": [],
                }
                if ocr_text:
                    meta_obj["ocr_text"] = ocr_text
                meta = json.dumps(meta_obj)
                chunk = Chunk(
                    document_id=document_id,
                    asset_id=asset.id,
                    anchor=anchor,
                    modality="figure",
                    content="",
                    meta=meta,
                )
                session.add(chunk)
                counts["figure"] += 1
                figures_added += 1
            except Exception:
                continue

    if settings.docling_extract_figures and figures_added == 0 and path.suffix.lower() == ".pdf":
        counts["figure"] += _extract_page_images_pdf(session, document_id, asset, path)

    session.commit()
    return counts


def _anchor_for_item(item: Any, idx: int) -> str:
    prov = getattr(item, "prov", None)
    if prov and isinstance(prov, list) and len(prov) > 0:
        page = getattr(prov[0], "page_no", None)
        if page is not None:
            return f"page:{page}-item:{idx}"
    return f"item:{idx}"


def _extract_page_images_pdf(
    session: Session,
    document_id: int,
    asset: Asset,
    path: Path,
) -> int:
    try:
        import pypdfium2 as pdfium
    except Exception:
        return 0
    fig_dir = artifacts_dir(document_id) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    pdf = pdfium.PdfDocument(str(path))
    total_pages = len(pdf)
    max_pages = min(settings.figure_fallback_max_pages, total_pages)
    added = 0
    for i in range(max_pages):
        try:
            page = pdf.get_page(i)
            pil = page.render(scale=settings.figure_fallback_scale).to_pil()
            image_path = fig_dir / f"page_{asset.id}_{i + 1}.png"
            pil.save(image_path, "PNG")
            ocr_text = _ocr_image(image_path)
            refs = extract_refs_from_text(f"Page {i + 1} raster {ocr_text}")
            meta_obj = {
                "path": str(image_path),
                "caption": f"Page {i + 1} raster",
                "source": "page_raster_fallback",
                "asset_kind": asset.kind,
                "figure_refs": sorted(refs),
                "quality_flags": ["page_raster_fallback"],
                "ocr_source": "easyocr" if ocr_text else None,
            }
            if ocr_text:
                meta_obj["ocr_text"] = ocr_text
            chunk = Chunk(
                document_id=document_id,
                asset_id=asset.id,
                anchor=f"page-image:{i + 1}",
                modality="figure",
                content="",
                meta=json.dumps(meta_obj),
            )
            session.add(chunk)
            added += 1
        except Exception:
            continue
    return added


def _parse_zip_file(
    session: Session,
    document_id: int,
    asset: Asset,
    path: Path,
    counts: dict[str, int],
) -> dict[str, int]:
    extract_root = artifacts_dir(document_id) / f"zip_asset_{asset.id}"
    extract_root.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                safe_name = _safe_zip_name(member.filename)
                if not safe_name:
                    continue
                target = extract_root / safe_name
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as src, open(target, "wb") as dst:
                    dst.write(src.read())
                counts = _parse_embedded_asset(
                    session,
                    document_id,
                    asset,
                    target,
                    counts,
                    anchor_prefix=f"zip:{path.stem}:{safe_name}:",
                )
    except Exception:
        return counts
    return counts


def _parse_embedded_asset(
    session: Session,
    document_id: int,
    asset: Asset,
    file_path: Path,
    counts: dict[str, int],
    *,
    anchor_prefix: str,
) -> dict[str, int]:
    ext = file_path.suffix.lower()
    if ext in {".csv", ".tsv", ".xlsx", ".xls"}:
        counts["table"] += _parse_tabular_file(session, document_id, asset, file_path, anchor_prefix=anchor_prefix)
    elif ext in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        counts["figure"] += _parse_image_file(session, document_id, asset, file_path, anchor_prefix=anchor_prefix)
    elif ext in {".txt"}:
        counts["text"] += _parse_text_file(session, document_id, asset, file_path, anchor_prefix=anchor_prefix)
    elif ext == ".pdf":
        if settings.parser_engine == "validated":
            counts = parse_pdf_validated(session, document_id, asset, file_path, counts)
        else:
            counts = _parse_with_docling(session, document_id, asset, file_path, counts)
    return counts


def _safe_zip_name(member_name: str) -> str:
    # Prevent path traversal and normalize separators for nested supplement files.
    normalized = member_name.replace("\\", "/").strip("/")
    if not normalized or normalized.startswith("../") or "/../" in normalized:
        return ""
    cleaned = os.path.normpath(normalized).replace("\\", "/")
    if cleaned.startswith(".."):
        return ""
    return cleaned


@lru_cache(maxsize=1)
def _ocr_reader():
    try:
        import easyocr
    except Exception:
        return None
    langs = [lang.strip() for lang in settings.figure_ocr_langs.split(",") if lang.strip()]
    return easyocr.Reader(langs, gpu=False)


def _ocr_image(path: Path) -> str:
    if not settings.figure_ocr_enabled:
        return ""
    reader = _ocr_reader()
    if reader is None:
        return ""
    try:
        results = reader.readtext(str(path))
    except Exception:
        return ""
    # results: list of (bbox, text, confidence)
    texts = []
    for _bbox, text, _conf in results:
        if text:
            texts.append(text)
    if not texts:
        return ""
    joined = " ".join(texts)
    if len(joined) > settings.figure_ocr_max_chars:
        joined = joined[: settings.figure_ocr_max_chars]
    return joined
