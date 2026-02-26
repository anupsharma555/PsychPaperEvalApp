from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import unquote, urljoin, urlparse, urlunparse

import httpx
from bs4 import BeautifulSoup

from app.core.config import settings


PDF_HINTS = (".pdf", "/pdf", "download=", "pdf?", "pdf=")
SUPP_KEYWORDS = (
    "supplement",
    "supplementary",
    "appendix",
    "additional file",
    "additional data",
    "supporting information",
    "supporting info",
    "supplemental",
    "dataset",
)
SUPP_EXTS = (
    ".pdf",
    ".zip",
    ".csv",
    ".tsv",
    ".xlsx",
    ".xls",
    ".docx",
    ".pptx",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
)

META_PDF_FIELDS = (
    "citation_pdf_url",
    "dc.identifier",
    "dc.relation",
    "dcterms.isformatof",
)

DOI_REGEX = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
SUPP_URL_REGEX = re.compile(
    r"""(?:
        (?P<abs>https?://[^\s"'<>\\]+(?:suppl_file[^\s"'<>\\]*|/doi/suppl/[^\s"'<>\\]*))
        |
        (?P<rel>/doi/suppl/[^\s"'<>\\]*)
    )""",
    re.IGNORECASE | re.VERBOSE,
)


@dataclass
class FetchResult:
    main_url: str
    content_type: str
    html: Optional[str] = None
    resolved_pdf_url: Optional[str] = None
    supplement_urls: Optional[list[str]] = None


def _http_client() -> httpx.Client:
    default_headers = {
        "User-Agent": settings.fetch_user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf;q=0.8,*/*;q=0.7",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }
    return httpx.Client(
        timeout=settings.fetch_timeout_sec,
        headers=default_headers,
        follow_redirects=True,
    )


def resolve_url(input_url: str, doi: Optional[str] = None) -> str:
    if doi:
        return f"https://doi.org/{doi}"
    return input_url


def _fallback_pdf_from_doi(doi: Optional[str]) -> Optional[str]:
    if not doi:
        return None
    return fetch_pdf_from_unpaywall(doi) or fetch_pdf_from_crossref(doi)


def _http_status_error_message(status_code: int) -> str:
    if status_code in (401, 403):
        return (
            f"Publisher blocked automated access (HTTP {status_code}). "
            "Use From Upload and choose the main PDF."
        )
    if status_code == 429:
        return (
            "Remote site rate-limited requests (HTTP 429). "
            "Retry later, or use From Upload with the main PDF."
        )
    return f"Failed to fetch URL (HTTP {status_code}). Verify URL/DOI and retry."


def fetch_url(url: str, doi: Optional[str] = None) -> FetchResult:
    with _http_client() as client:
        try:
            resp = client.get(url)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            status_code = int(exc.response.status_code)
            doi_candidate = doi or extract_doi_from_text(url)
            pdf_url = _fallback_pdf_from_doi(doi_candidate)
            if pdf_url:
                return FetchResult(
                    main_url=url,
                    content_type="",
                    html=None,
                    resolved_pdf_url=pdf_url,
                    supplement_urls=[],
                )
            raise ValueError(_http_status_error_message(status_code)) from exc
        except httpx.RequestError as exc:
            doi_candidate = doi or extract_doi_from_text(url)
            pdf_url = _fallback_pdf_from_doi(doi_candidate)
            if pdf_url:
                return FetchResult(
                    main_url=url,
                    content_type="",
                    html=None,
                    resolved_pdf_url=pdf_url,
                    supplement_urls=[],
                )
            raise ValueError("Network error while fetching URL. Verify URL/DOI and retry.") from exc

        content_type = resp.headers.get("content-type", "")
        resolved_main_url = str(resp.url)
        if "application/pdf" in content_type or url.lower().endswith(".pdf"):
            return FetchResult(main_url=resolved_main_url, content_type=content_type, resolved_pdf_url=resolved_main_url)
        html = resp.text
        soup = BeautifulSoup(html, "html.parser")
        doi = doi or extract_doi_from_text(url) or extract_doi_from_html(soup)
        pdf_url = extract_pdf_link_from_html(resolved_main_url, soup)
        if not pdf_url and doi:
            pdf_url = _fallback_pdf_from_doi(doi)
        supp_urls = extract_supplement_links_from_html(resolved_main_url, soup, raw_html=html)
        return FetchResult(
            main_url=resolved_main_url,
            content_type=content_type,
            html=html,
            resolved_pdf_url=pdf_url,
            supplement_urls=supp_urls,
        )


def extract_pdf_link_from_html(base_url: str, soup: BeautifulSoup) -> Optional[str]:
    links = []
    meta_pdf = extract_pdf_link_from_meta(soup)
    if meta_pdf:
        return urljoin(base_url, meta_pdf)
    typed_pdf = extract_pdf_link_from_links(base_url, soup)
    if typed_pdf:
        return typed_pdf
    for a in soup.find_all("a", href=True):
        href = a["href"]
        text = (a.get_text(" ") or "").lower()
        if any(hint in href.lower() for hint in PDF_HINTS) or "pdf" in text:
            links.append(urljoin(base_url, href))
    if not links:
        return None
    # Prefer links that look like direct pdf
    for link in links:
        if link.lower().endswith(".pdf"):
            return link
    return links[0]


def extract_pdf_link_from_meta(soup: BeautifulSoup) -> Optional[str]:
    for meta in soup.find_all("meta"):
        name = (meta.get("name") or meta.get("property") or "").lower()
        if name in META_PDF_FIELDS:
            content = meta.get("content")
            if content and ".pdf" in content.lower():
                return content
        if name == "citation_pdf_url":
            content = meta.get("content")
            if content:
                return content
    return None


def extract_pdf_link_from_links(base_url: str, soup: BeautifulSoup) -> Optional[str]:
    for link in soup.find_all("link", href=True):
        href = link["href"]
        type_attr = (link.get("type") or "").lower()
        rel = " ".join(link.get("rel") or []).lower()
        if "pdf" in type_attr or "pdf" in rel:
            return urljoin(base_url, href)
    return None


def extract_supplement_links_from_html(base_url: str, soup: BeautifulSoup, raw_html: str | None = None) -> list[str]:
    base_parsed = urlparse(base_url)
    base_canonical = urlunparse(base_parsed._replace(query="", fragment=""))
    links: list[str] = []
    for a in soup.find_all("a", href=True):
        href = str(a["href"] or "").strip()
        if not href:
            continue
        if href.startswith("#"):
            # Ignore same-page anchors like #supplementary-materials.
            continue
        text = (a.get_text(" ") or "").lower()
        resolved = _normalize_candidate_url(urljoin(base_url, href))
        parsed = urlparse(resolved)
        canonical = urlunparse(parsed._replace(query="", fragment=""))
        if canonical == base_canonical:
            continue
        href_lower = href.lower()
        resolved_lower = resolved.lower()
        if (
            any(keyword in text for keyword in SUPP_KEYWORDS)
            or any(ext in href_lower for ext in SUPP_EXTS)
            or "/doi/suppl/" in resolved_lower
            or "suppl_file" in resolved_lower
        ):
            links.append(resolved)
    links.extend(_extract_supplement_links_from_raw_html(base_url, raw_html or ""))
    # dedupe while preserving order
    seen: set[str] = set()
    ordered: list[str] = []
    for link in links:
        parsed = urlparse(link)
        canonical = urlunparse(parsed._replace(query="", fragment=""))
        if canonical == base_canonical:
            continue
        dedupe_key = _supp_dedupe_key(link)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        ordered.append(link)

    def _priority(link: str) -> tuple[int, int]:
        lower = link.lower()
        is_pdf = lower.split("?", 1)[0].endswith(".pdf")
        has_supp_path = "/doi/suppl/" in lower or "suppl_file" in lower
        return (1 if is_pdf else 0, 1 if has_supp_path else 0)

    ordered.sort(key=_priority, reverse=True)
    return ordered


def _extract_supplement_links_from_raw_html(base_url: str, raw_html: str) -> list[str]:
    if not raw_html:
        return []
    decoded = unquote(raw_html.replace("\\/", "/"))
    found: list[str] = []
    for match in SUPP_URL_REGEX.finditer(decoded):
        candidate = match.group("abs") or match.group("rel")
        if not candidate:
            continue
        found.append(_normalize_candidate_url(urljoin(base_url, candidate)))
    return found


def _supp_dedupe_key(url: str) -> str:
    parsed = urlparse(url)
    path = (parsed.path or "").rstrip("/").lower()
    query = (parsed.query or "").lower()
    if "/doi/suppl/" in path or "suppl_file" in path:
        return f"{path}?{query}".rstrip("?")
    return urlunparse(parsed._replace(query="", fragment="")).lower()


def _normalize_candidate_url(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.scheme and not parsed.netloc:
        return url
    netloc = (parsed.netloc or "").lower()
    scheme = (parsed.scheme or "https").lower()
    if netloc == "ajp.psychiatryonline.org":
        netloc = "psychiatryonline.org"
    if netloc.endswith("psychiatryonline.org"):
        scheme = "https"
    return urlunparse(parsed._replace(scheme=scheme, netloc=netloc))


def extract_doi_from_text(text: str) -> Optional[str]:
    match = DOI_REGEX.search(text or "")
    if match:
        return match.group(0)
    return None


def extract_doi_from_html(soup: BeautifulSoup) -> Optional[str]:
    # Try citation/doi meta tags
    for meta in soup.find_all("meta"):
        name = (meta.get("name") or meta.get("property") or "").lower()
        if "doi" in name:
            content = meta.get("content")
            doi = extract_doi_from_text(content or "")
            if doi:
                return doi
    # Fallback: scan text
    return extract_doi_from_text(soup.get_text(" "))


def fetch_pdf_from_unpaywall(doi: str) -> Optional[str]:
    if not settings.unpaywall_email:
        return None
    url = f"https://api.unpaywall.org/v2/{doi}?email={settings.unpaywall_email}"
    try:
        with _http_client() as client:
            resp = client.get(url)
            if resp.status_code != 200:
                return None
            data = resp.json()
    except Exception:
        return None

    best = data.get("best_oa_location") or {}
    return best.get("url_for_pdf") or best.get("url")


def fetch_pdf_from_crossref(doi: str) -> Optional[str]:
    url = f"https://api.crossref.org/works/{doi}"
    try:
        with _http_client() as client:
            resp = client.get(url)
            if resp.status_code != 200:
                return None
            data = resp.json()
    except Exception:
        return None

    message = data.get("message") if isinstance(data, dict) else {}
    links = message.get("link") if isinstance(message, dict) else []
    if not isinstance(links, list):
        return None

    for item in links:
        if not isinstance(item, dict):
            continue
        link_url = item.get("URL")
        content_type = str(item.get("content-type") or "").lower()
        if not isinstance(link_url, str) or not link_url:
            continue
        if "pdf" in content_type or link_url.lower().endswith(".pdf"):
            return link_url
    return None


def download_file(url: str, dest: Path, *, referer: str | None = None) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with _http_client() as client:
        headers = {"Referer": referer} if referer else None
        with client.stream("GET", url, headers=headers) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_bytes():
                    f.write(chunk)
    return str(dest)


def guess_filename(url: str, fallback: str) -> str:
    name = url.split("?")[0].rstrip("/").split("/")[-1]
    if not name:
        return fallback
    if "." not in name:
        return f"{name}.pdf"
    return name


def filter_supp_urls(urls: Iterable[str], *, main_url: str | None = None) -> list[str]:
    main_canonical = ""
    if main_url:
        parsed_main = urlparse(main_url)
        main_canonical = urlunparse(parsed_main._replace(query="", fragment=""))
    filtered = []
    seen: set[str] = set()
    for url in urls:
        normalized = _normalize_candidate_url(url)
        lower = normalized.lower().strip()
        if not lower:
            continue
        parsed = urlparse(normalized)
        canonical = urlunparse(parsed._replace(query="", fragment=""))
        if main_canonical and canonical == main_canonical:
            continue
        dedupe_key = _supp_dedupe_key(normalized)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        if any(ext in lower for ext in SUPP_EXTS):
            filtered.append(normalized)
        elif "/doi/suppl/" in lower or "suppl_file" in lower:
            filtered.append(normalized)
        elif any(keyword.replace(" ", "-") in lower for keyword in SUPP_KEYWORDS):
            filtered.append(normalized)
    return filtered


def discover_additional_supplement_urls(
    *,
    main_url: str | None = None,
    doi: str | None = None,
    resolved_pdf_url: str | None = None,
) -> list[str]:
    candidates = _candidate_article_urls(main_url=main_url, doi=doi, resolved_pdf_url=resolved_pdf_url)
    if not candidates:
        return []

    discovered: list[str] = []
    with _http_client() as client:
        for candidate in candidates[:10]:
            try:
                response = client.get(candidate)
            except Exception:
                continue
            if response.status_code != 200:
                continue
            content_type = str(response.headers.get("content-type", "")).lower()
            if "application/pdf" in content_type:
                continue
            html = response.text or ""
            if not html:
                continue
            soup = BeautifulSoup(html, "html.parser")
            discovered.extend(
                extract_supplement_links_from_html(
                    str(response.url),
                    soup,
                    raw_html=html,
                )
            )
    return filter_supp_urls(discovered, main_url=main_url)


def _candidate_article_urls(
    *,
    main_url: str | None,
    doi: str | None,
    resolved_pdf_url: str | None,
) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()

    def _push(url: str | None) -> None:
        value = str(url or "").strip()
        if not value:
            return
        key = value.lower()
        if key in seen:
            return
        seen.add(key)
        out.append(value)

    _push(main_url)
    if doi:
        normalized_doi = doi.strip()
        _push(f"https://doi.org/{normalized_doi}")
        _push(f"https://psychiatryonline.org/doi/{normalized_doi}")
        _push(f"https://psychiatryonline.org/doi/full/{normalized_doi}")

    for raw_url in [resolved_pdf_url, main_url]:
        parsed = urlparse(str(raw_url or "").strip())
        if not parsed.scheme or not parsed.netloc:
            continue
        path = parsed.path or ""
        variants = [path]
        if "/doi/pdf/" in path:
            variants.append(path.replace("/doi/pdf/", "/doi/", 1))
            variants.append(path.replace("/doi/pdf/", "/doi/full/", 1))
        if "/doi/epdf/" in path:
            variants.append(path.replace("/doi/epdf/", "/doi/", 1))
            variants.append(path.replace("/doi/epdf/", "/doi/full/", 1))
        if path.lower().endswith(".pdf"):
            variants.append(path[:-4])
        for variant in variants:
            _push(urlunparse(parsed._replace(path=variant, query="", fragment="")))

    return out
