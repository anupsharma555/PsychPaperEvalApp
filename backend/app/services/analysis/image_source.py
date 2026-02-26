from __future__ import annotations

import hashlib
import mimetypes
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

import httpx

from app.core.config import settings


MAX_REMOTE_IMAGE_BYTES = 15 * 1024 * 1024
ALLOWED_IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
}
DISALLOWED_IMAGE_SUFFIXES = {
    ".pdf",
    ".zip",
    ".csv",
    ".tsv",
    ".xlsx",
    ".xls",
    ".doc",
    ".docx",
    ".ppt",
    ".pptx",
    ".txt",
}


class NonImageResponseError(ValueError):
    pass


class UnsupportedImageTypeError(ValueError):
    pass


def resolve_image_path(
    meta_obj: dict[str, Any],
    cache_dir: Path,
    remote_cache: dict[str, str],
) -> tuple[str | None, str | None, str | None]:
    local_path = str(meta_obj.get("path") or "").strip()
    if local_path:
        path = Path(local_path).expanduser()
        if path.exists() and path.is_file():
            return str(path), "local_path", None
        if not str(meta_obj.get("source_url") or "").strip():
            return None, None, "local_path_missing"

    source_url = _normalize_remote_url(str(meta_obj.get("source_url") or "").strip())
    if not source_url:
        return None, None, "no_image_source"

    parsed = urlparse(source_url)
    if parsed.scheme not in {"http", "https"}:
        return None, None, "unsupported_source_url"
    suffix = Path(parsed.path).suffix.lower()
    if suffix in DISALLOWED_IMAGE_SUFFIXES:
        return None, None, "unsupported_image_type"

    cached = remote_cache.get(source_url)
    if cached and Path(cached).exists():
        return cached, "remote_url", None

    referer = str(
        meta_obj.get("source_page_url")
        or meta_obj.get("referer")
        or meta_obj.get("document_source_url")
        or ""
    ).strip()
    if referer:
        referer = _normalize_remote_url(referer)

    try:
        downloaded = _download_remote_image(source_url, cache_dir, referer=referer or None)
    except UnsupportedImageTypeError:
        return None, None, "unsupported_image_type"
    except NonImageResponseError:
        return None, None, "non_image_response"
    except Exception:
        return None, None, "download_error"

    remote_cache[source_url] = downloaded
    return downloaded, "remote_url", None


def _download_remote_image(source_url: str, cache_dir: Path, referer: str | None = None) -> str:
    cache_dir.mkdir(parents=True, exist_ok=True)
    headers = {
        "User-Agent": settings.fetch_user_agent,
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }
    if referer:
        headers["Referer"] = referer
    with httpx.Client(
        timeout=min(max(int(settings.fetch_timeout_sec), 5), 20),
        headers=headers,
        follow_redirects=True,
    ) as client:
        response = client.get(source_url)
        response.raise_for_status()
        content_type = str(response.headers.get("content-type") or "").split(";", 1)[0].strip().lower()
        payload = response.content

    if not payload:
        raise NonImageResponseError("empty payload")
    if len(payload) > MAX_REMOTE_IMAGE_BYTES:
        raise UnsupportedImageTypeError("payload too large")
    if content_type and not content_type.startswith("image/"):
        raise NonImageResponseError(content_type)

    suffix = _infer_suffix(source_url, content_type)
    if suffix == ".svg":
        raise UnsupportedImageTypeError("svg unsupported")

    digest = hashlib.sha1(source_url.encode("utf-8")).hexdigest()
    output_path = cache_dir / f"{digest}{suffix}"
    output_path.write_bytes(payload)
    return str(output_path)


def _infer_suffix(source_url: str, content_type: str) -> str:
    parsed = urlparse(source_url)
    path_suffix = Path(parsed.path).suffix.lower()
    if path_suffix in ALLOWED_IMAGE_SUFFIXES:
        return path_suffix
    if path_suffix == ".svg":
        return ".svg"

    mime = content_type if content_type else None
    if mime:
        guessed = mimetypes.guess_extension(mime)
        if guessed:
            guessed = guessed.lower()
            if guessed in ALLOWED_IMAGE_SUFFIXES:
                return guessed
            if guessed == ".svg":
                return ".svg"
    return ".jpg"


def _normalize_remote_url(value: str) -> str:
    if not value:
        return ""
    parsed = urlparse(value)
    if not parsed.scheme or not parsed.netloc:
        return value
    netloc = parsed.netloc.lower()
    scheme = parsed.scheme.lower()
    if netloc == "doi.org" and parsed.path.startswith("/cms/"):
        netloc = "psychiatryonline.org"
        scheme = "https"
    if netloc == "ajp.psychiatryonline.org":
        netloc = "psychiatryonline.org"
        scheme = "https"
    if netloc.endswith("psychiatryonline.org") and scheme != "https":
        scheme = "https"
    return urlunparse(parsed._replace(netloc=netloc, scheme=scheme))
