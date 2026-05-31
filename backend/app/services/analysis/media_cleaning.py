from __future__ import annotations

import re
from typing import Any


GENERIC_CAPTION_RE = re.compile(r"^\s*(?:fig(?:ure)?\.?\s*)?[A-Za-z]?\d+[A-Za-z]?[.:]?\s*$", re.IGNORECASE)
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
SOFT_HYPHEN_RE = re.compile(r"([A-Za-z])-\s+([a-z])")
SPACED_PUNCT_RE = re.compile(r"\s+([,.;:)])")
MISSING_SPACE_RE = re.compile(r"([a-z])([A-Z])")
OCR_NUMERIC_SEPARATOR_RE = re.compile(r"(?<=\d)\s+[46]\s+(?=\d)")
REPEATED_SPACE_RE = re.compile(r"\s+")


def clean_figure_caption(value: Any) -> str:
    return _clean_media_text(value, max_chars=4000)


def clean_figure_ocr_text(value: Any) -> str:
    return _clean_media_text(value, max_chars=4000)


def usable_caption(value: Any, *, min_chars: int = 80) -> bool:
    text = clean_figure_caption(value)
    if len(text) < min_chars:
        return False
    return not GENERIC_CAPTION_RE.fullmatch(text)


def figure_downstream_text(
    *,
    caption: Any,
    ocr_text: Any,
    caption_first: bool = True,
    min_caption_chars: int = 80,
) -> tuple[str, str]:
    clean_caption = clean_figure_caption(caption)
    clean_ocr = clean_figure_ocr_text(ocr_text)
    if not caption_first:
        parts = [part for part in (clean_caption, clean_ocr) if part]
        return "\n".join(parts), "caption_plus_ocr" if clean_ocr else "caption"
    if usable_caption(clean_caption, min_chars=min_caption_chars):
        return clean_caption, "caption"
    parts = [part for part in (clean_caption, clean_ocr) if part]
    if parts:
        return "\n".join(parts), "caption_plus_ocr_fallback" if clean_caption and clean_ocr else "ocr_fallback"
    return "", "missing"


def _clean_media_text(value: Any, *, max_chars: int) -> str:
    text = str(value or "")
    if not text:
        return ""
    text = text.replace("\u00b1", "+/-")
    text = text.replace("\x00B1", "+/-")
    text = CONTROL_CHAR_RE.sub(" ", text)
    text = text.replace("\u00ad", "")
    text = SOFT_HYPHEN_RE.sub(r"\1\2", text)
    text = OCR_NUMERIC_SEPARATOR_RE.sub(" +/- ", text)
    text = MISSING_SPACE_RE.sub(r"\1 \2", text)
    text = SPACED_PUNCT_RE.sub(r"\1", text)
    text = REPEATED_SPACE_RE.sub(" ", text).strip()
    return text[:max_chars]
