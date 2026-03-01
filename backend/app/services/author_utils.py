from __future__ import annotations

import re
from typing import Any


_AFFILIATION_HINT_RE = re.compile(
    r"\b("
    r"department|university|hospital|institute|school|laborator(?:y|ies)|center|centre|"
    r"college|faculty|clinic|academy|medical|research|radiology|psychiatry|"
    r"original investigation|key laboratory|road|street|avenue|china|australia|usa|uk|"
    r"email|correspondence|psychoradiology|imaging"
    r")\b",
    re.IGNORECASE,
)


def normalize_author_name(value: Any) -> str:
    text = " ".join(str(value or "").replace("\u00a0", " ").split()).strip()
    if not text:
        return ""
    text = re.sub(r"^[,;:\-\.\s]+|[,;:\-\.\s]+$", "", text).strip()
    return text


def is_probable_author_name(value: Any) -> bool:
    name = normalize_author_name(value)
    if not name:
        return False
    lowered = name.lower()
    if len(name) < 3 or len(name) > 96:
        return False
    if "@" in name or "http://" in lowered or "https://" in lowered:
        return False
    if "|" in name or ";" in name:
        return False
    if any(char.isdigit() for char in name):
        return False
    if _AFFILIATION_HINT_RE.search(lowered):
        return False
    if "(" in name and ")" in name and len(name.split()) > 5:
        return False
    tokens = [token for token in re.split(r"\s+", name) if token]
    if len(tokens) < 2 or len(tokens) > 7:
        return False
    alpha_tokens = 0
    for token in tokens:
        cleaned = "".join(ch for ch in token if ch.isalpha() or ch in {"-", "'", "."})
        if cleaned and len(cleaned) > 32:
            return False
        if any(ch.isalpha() for ch in cleaned):
            alpha_tokens += 1
    return alpha_tokens >= 2


def sanitize_author_list(values: Any, *, max_items: int = 24) -> tuple[list[str], int]:
    if not isinstance(values, list):
        return [], 0
    cleaned_values = [normalize_author_name(value) for value in values]
    cleaned_values = [value for value in cleaned_values if value]

    deduped: list[str] = []
    seen: set[str] = set()
    for name in cleaned_values:
        if not is_probable_author_name(name):
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(name)
    extracted_count = len(deduped)
    return deduped[:max(1, int(max_items or 24))], extracted_count
