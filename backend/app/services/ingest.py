from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, urlunparse

from fastapi import UploadFile
import httpx
from sqlmodel import Session
from bs4 import BeautifulSoup

from app.core.config import settings
from app.db.models import Asset, Document
from app.services.fetcher import (
    discover_additional_supplement_urls,
    download_file,
    fetch_url,
    filter_supp_urls,
    guess_filename,
    resolve_url,
)
from app.services.storage import asset_path, artifacts_dir, document_dir, ensure_document_dirs


def _looks_like_pdf(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            head = handle.read(1024)
    except Exception:
        return False
    return b"%PDF-" in head


def _looks_like_html(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            head = handle.read(2048)
    except Exception:
        return False
    lowered = head.decode("utf-8", errors="ignore").lower()
    return "<html" in lowered or "<!doctype html" in lowered or "<body" in lowered


def _looks_like_zip(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            head = handle.read(8)
    except Exception:
        return False
    return head.startswith((b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"))


def _detected_image_extension(path: Path) -> str | None:
    try:
        with path.open("rb") as handle:
            head = handle.read(32)
    except Exception:
        return None
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if head.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if head.startswith((b"II*\x00", b"MM\x00*")):
        return ".tiff"
    if head.startswith((b"GIF87a", b"GIF89a")):
        return ".gif"
    if len(head) >= 12 and head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return ".webp"
    return None


def _sanitize_stem(filename: str, fallback: str) -> str:
    stem = Path(str(filename or "")).stem.strip()
    if not stem:
        stem = fallback
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return cleaned or fallback


def _unique_asset_target(document_id: int, stem: str, extension: str) -> tuple[str, Path]:
    name = f"{stem}{extension}"
    target = asset_path(document_id, name)
    suffix = 2
    while target.exists():
        name = f"{stem}_{suffix}{extension}"
        target = asset_path(document_id, name)
        suffix += 1
    return name, target


def _normalize_downloaded_supplement(
    document_id: int,
    filename: str,
    path: Path,
) -> tuple[str, Path, str | None]:
    detected_ext = ""
    content_type: str | None = None
    if _looks_like_pdf(path):
        detected_ext = ".pdf"
        content_type = "application/pdf"
    elif _looks_like_html(path):
        detected_ext = ".html"
        content_type = "text/html"
    else:
        image_ext = _detected_image_extension(path)
        if image_ext:
            detected_ext = image_ext
            content_type = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".tiff": "image/tiff",
                ".gif": "image/gif",
                ".webp": "image/webp",
            }.get(image_ext, None)
        elif _looks_like_zip(path):
            detected_ext = ".zip"
            content_type = "application/zip"

    if not detected_ext or path.suffix.lower() == detected_ext:
        return filename, path, content_type

    stem = _sanitize_stem(filename, "supp")
    normalized_name, normalized_path = _unique_asset_target(document_id, stem, detected_ext)
    try:
        path.rename(normalized_path)
    except Exception:
        return filename, path, content_type
    return normalized_name, normalized_path, content_type


def _url_suggests_pdf(url: str) -> bool:
    parsed = urlparse(url or "")
    path = parsed.path.lower()
    query = parsed.query.lower()
    return path.endswith(".pdf") or "/doi/suppl/" in path or "pdf" in query or "suppl_file" in path


FIGURE_TITLE_LINE_RE = re.compile(r"^Figure\s+(S?\d+[A-Za-z]?)\.\s*(.+)$", re.IGNORECASE)
FIGURE_SKIP_LINE_RE = re.compile(
    r"^(?:view large|download|open in|go to figure in article|related|close)$",
    re.IGNORECASE,
)
FIGURE_STOP_LINE_RE = re.compile(
    r"^(?:key points|abstract|introduction|methods?|results?|discussion|conclusions?|article information|references?)$",
    re.IGNORECASE,
)


def _normalize_figure_token(value: str) -> str:
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


def _extract_source_figure_legend_maps(html: str | None) -> dict[str, dict[str, str]]:
    raw_html = str(html or "")
    if not raw_html:
        return {"caption_map": {}, "legend_map": {}}

    soup = BeautifulSoup(raw_html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    lines = [re.sub(r"\s+", " ", line).strip() for line in soup.get_text("\n").splitlines()]
    lines = [line for line in lines if line]

    caption_map: dict[str, str] = {}
    legend_map: dict[str, str] = {}
    idx = 0
    total = len(lines)
    while idx < total:
        line = lines[idx]
        match = FIGURE_TITLE_LINE_RE.match(line)
        if not match:
            idx += 1
            continue
        token = _normalize_figure_token(match.group(1) or "")
        title_tail = str(match.group(2) or "").strip()
        title = f"Figure {match.group(1)}. {title_tail}".strip()
        if token and title:
            existing_caption = caption_map.get(token, "")
            if len(title) > len(existing_caption):
                caption_map[token] = title

        legend_parts: list[str] = []
        probe = idx + 1
        while probe < total:
            candidate = lines[probe]
            if FIGURE_TITLE_LINE_RE.match(candidate):
                break
            if FIGURE_STOP_LINE_RE.match(candidate):
                break
            if FIGURE_SKIP_LINE_RE.match(candidate):
                probe += 1
                continue
            # Ignore bare urls and tiny control labels.
            if re.match(r"^https?://", candidate, flags=re.IGNORECASE) or len(candidate) < 8:
                probe += 1
                continue
            legend_parts.append(candidate)
            if len(" ".join(legend_parts)) >= int(getattr(settings, "media_legend_max_chars", 6000) or 6000):
                break
            probe += 1

        legend_text = re.sub(r"\s+", " ", " ".join(legend_parts)).strip()
        if token and legend_text:
            if title and legend_text.lower().startswith(title.lower()):
                legend_text = legend_text[len(title) :].strip(" .:-")
            if len(legend_text.split()) >= 8:
                existing_legend = legend_map.get(token, "")
                if len(legend_text) > len(existing_legend):
                    legend_map[token] = legend_text
        idx = max(idx + 1, probe)

    return {"caption_map": caption_map, "legend_map": legend_map}


def _persist_source_figure_legend_maps(document_id: int, html: str | None, *, source_url: str = "") -> None:
    maps = _extract_source_figure_legend_maps(html)
    caption_map = maps.get("caption_map", {}) if isinstance(maps, dict) else {}
    legend_map = maps.get("legend_map", {}) if isinstance(maps, dict) else {}
    if not isinstance(caption_map, dict):
        caption_map = {}
    if not isinstance(legend_map, dict):
        legend_map = {}
    if not caption_map and not legend_map:
        return
    payload = {
        "source": "html_text_scan_v1",
        "source_url": str(source_url or "").strip(),
        "caption_map": {str(k): str(v) for k, v in caption_map.items() if str(k).strip() and str(v).strip()},
        "legend_map": {str(k): str(v) for k, v in legend_map.items() if str(k).strip() and str(v).strip()},
    }
    path = artifacts_dir(document_id) / "source_figure_legends.json"
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _supplement_url_variants(url: str) -> list[str]:
    parsed = urlparse(url or "")
    if not parsed.scheme and not parsed.netloc:
        return [url]
    out: list[str] = []
    seen: set[str] = set()

    def _push(candidate: str) -> None:
        key = candidate.strip().lower()
        if not key or key in seen:
            return
        seen.add(key)
        out.append(candidate)

    netloc = (parsed.netloc or "").lower()
    scheme = (parsed.scheme or "https").lower()
    _push(urlunparse(parsed._replace(scheme=scheme, netloc=netloc)))

    if netloc == "ajp.psychiatryonline.org":
        _push(urlunparse(parsed._replace(scheme="https", netloc="psychiatryonline.org")))
    if netloc.endswith("psychiatryonline.org") and scheme != "https":
        _push(urlunparse(parsed._replace(scheme="https")))
    return out


def _download_supplement_with_resolution(
    *,
    supp_url: str,
    dest: Path,
    referer: str,
    doi: str | None = None,
) -> bool:
    attempted: set[str] = set()

    def _attempt(candidate_url: str, candidate_referer: str) -> bool:
        key = candidate_url.strip().lower()
        if not key or key in attempted:
            return False
        attempted.add(key)
        try:
            download_file(candidate_url, dest, referer=candidate_referer)
        except Exception:
            return False
        if _url_suggests_pdf(candidate_url) and not _looks_like_pdf(dest):
            try:
                dest.unlink()
            except Exception:
                pass
            return False
        return True

    for candidate in _supplement_url_variants(supp_url):
        if _attempt(candidate, referer):
            return True
        try:
            nested = fetch_url(candidate, doi=doi)
        except Exception:
            continue
        nested_urls: list[str] = []
        if nested.resolved_pdf_url:
            nested_urls.append(nested.resolved_pdf_url)
        nested_urls.extend(filter_supp_urls(nested.supplement_urls or [], main_url=nested.main_url))
        nested_urls.extend(
            discover_additional_supplement_urls(
                main_url=nested.main_url,
                doi=doi,
                resolved_pdf_url=nested.resolved_pdf_url,
            )
        )
        nested_referer = nested.main_url or referer
        for nested_url in nested_urls:
            for nested_candidate in _supplement_url_variants(nested_url):
                if _attempt(nested_candidate, nested_referer):
                    return True
    return False


def _extract_readable_html_text(html: str) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    root = soup.find("article") or soup.find("main") or soup.body or soup
    text = root.get_text("\n", strip=True) if root else ""
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _try_add_html_fallback_main_asset(
    session: Session,
    document_id: int,
    html: str | None,
) -> bool:
    raw_html = str(html or "")
    html_text = _extract_readable_html_text(raw_html)
    if len(html_text) < 1200:
        return False
    main_filename = "main_from_url.html"
    main_path = asset_path(document_id, main_filename)
    main_path.write_text(raw_html, encoding="utf-8")
    session.add(
        Asset(
            document_id=document_id,
            kind="main",
            filename=main_filename,
            content_type="text/html",
            path=str(main_path),
        )
    )
    return True


async def ingest_upload(
    session: Session,
    main_file: UploadFile,
    supp_files: Optional[list[UploadFile]] = None,
    source_url: Optional[str] = None,
    doi: Optional[str] = None,
) -> Document:
    document = Document(source_url=source_url, doi=doi)
    session.add(document)
    session.commit()
    session.refresh(document)

    ensure_document_dirs(document.id)

    main_path = asset_path(document.id, main_file.filename)
    with open(main_path, "wb") as f:
        f.write(await main_file.read())

    session.add(
        Asset(
            document_id=document.id,
            kind="main",
            filename=main_file.filename,
            content_type=main_file.content_type,
            path=str(main_path),
        )
    )

    for supp in supp_files or []:
        supp_path = asset_path(document.id, supp.filename)
        with open(supp_path, "wb") as f:
            f.write(await supp.read())
        session.add(
            Asset(
                document_id=document.id,
                kind="supp",
                filename=supp.filename,
                content_type=supp.content_type,
                path=str(supp_path),
            )
        )

    session.commit()
    session.refresh(document)
    return document


def ingest_url(
    session: Session,
    input_url: str,
    doi: Optional[str] = None,
    fetch_supplements: bool = True,
) -> Document:
    document = Document(source_url=input_url, doi=doi)
    session.add(document)
    session.commit()
    session.refresh(document)

    try:
        ensure_document_dirs(document.id)

        resolved = resolve_url(input_url, doi)
        fetch_result = fetch_url(resolved, doi=doi)
        if fetch_result.main_url:
            document.source_url = str(fetch_result.main_url)
        _persist_source_figure_legend_maps(
            document.id,
            fetch_result.html,
            source_url=fetch_result.main_url or resolved,
        )

        pdf_url = fetch_result.resolved_pdf_url
        if pdf_url:
            main_filename = guess_filename(pdf_url, "main.pdf")
            main_path = asset_path(document.id, main_filename)
            try:
                download_file(pdf_url, main_path, referer=fetch_result.main_url or resolved)
            except httpx.HTTPStatusError as exc:
                try:
                    if main_path.exists():
                        main_path.unlink()
                except Exception:
                    pass
                if _try_add_html_fallback_main_asset(session, document.id, fetch_result.html):
                    main_path = None
                else:
                    status_code = int(exc.response.status_code)
                    if status_code in (401, 403):
                        raise ValueError(
                            f"Resolved PDF is access-restricted by publisher (HTTP {status_code}). "
                            "Use From Upload and choose the main PDF."
                        ) from exc
                    if status_code == 429:
                        raise ValueError(
                            "Resolved PDF download was rate-limited (HTTP 429). "
                            "Retry later, or use From Upload with the main PDF."
                        ) from exc
                    raise ValueError(
                        f"Could not download resolved PDF (HTTP {status_code}). "
                        "Use From Upload and choose the main PDF."
                    ) from exc
            except Exception as exc:
                try:
                    if main_path.exists():
                        main_path.unlink()
                except Exception:
                    pass
                if _try_add_html_fallback_main_asset(session, document.id, fetch_result.html):
                    main_path = None
                else:
                    raise ValueError(
                        "Could not download resolved PDF. Use From Upload and choose the main PDF."
                    ) from exc

            if main_path and not _looks_like_pdf(main_path):
                try:
                    if main_path.exists():
                        main_path.unlink()
                except Exception:
                    pass
                if _try_add_html_fallback_main_asset(session, document.id, fetch_result.html):
                    main_path = None
                else:
                    raise ValueError(
                        "Resolved download did not return a valid PDF (likely publisher block page). "
                        "Use From Upload and choose the main PDF."
                    )

            if main_path:
                session.add(
                    Asset(
                        document_id=document.id,
                        kind="main",
                        filename=main_filename,
                        content_type="application/pdf",
                        path=str(main_path),
                    )
                )
        else:
            if not _try_add_html_fallback_main_asset(session, document.id, fetch_result.html):
                raise ValueError(
                    "Could not resolve a downloadable PDF, and the page does not expose enough full-text content. "
                    "Use From Upload and choose the main PDF."
                )

        supp_urls: list[str] = []
        if fetch_supplements:
            if fetch_result.supplement_urls:
                supp_urls.extend(fetch_result.supplement_urls)
            supp_urls.extend(
                discover_additional_supplement_urls(
                    main_url=fetch_result.main_url or resolved,
                    doi=doi,
                    resolved_pdf_url=pdf_url,
                )
            )
            supp_urls = filter_supp_urls(supp_urls, main_url=fetch_result.main_url)

        for idx, supp_url in enumerate(supp_urls):
            supp_filename = guess_filename(supp_url, f"supp_{idx + 1}.bin")
            supp_path = asset_path(document.id, supp_filename)
            try:
                downloaded = _download_supplement_with_resolution(
                    supp_url=supp_url,
                    dest=supp_path,
                    referer=fetch_result.main_url or resolved,
                    doi=doi,
                )
                if not downloaded:
                    raise ValueError("supplement download failed")
                expects_pdf = _url_suggests_pdf(supp_url)
                is_pdf = _looks_like_pdf(supp_path)
                if expects_pdf and not is_pdf:
                    try:
                        supp_path.unlink()
                    except Exception:
                        pass
                    continue
                supp_filename, supp_path, content_type = _normalize_downloaded_supplement(
                    document.id,
                    supp_filename,
                    supp_path,
                )
                session.add(
                    Asset(
                        document_id=document.id,
                        kind="supp",
                        filename=supp_filename,
                        content_type=content_type,
                        path=str(supp_path),
                    )
                )
            except Exception:
                # Skip failed supplements but keep main document
                continue

        session.commit()
        session.refresh(document)
        return document
    except Exception:
        session.rollback()
        try:
            session.delete(document)
            session.commit()
        except Exception:
            session.rollback()
        shutil.rmtree(document_dir(document.id), ignore_errors=True)
        raise
