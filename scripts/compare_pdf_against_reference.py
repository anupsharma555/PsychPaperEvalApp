#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "backend"
SECTION_KEYS = ["introduction", "methods", "results", "discussion", "conclusion"]
SECTION_HEADERS = {
    "introduction": "INTRODUCTION",
    "methods": "METHODS",
    "results": "RESULTS",
    "discussion": "DISCUSSION",
    "conclusion": "CONCLUSION",
}
SECTION_HEADER_TO_KEY = {value: key for key, value in SECTION_HEADERS.items()}
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
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
}
TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-]*")
CANONICAL_STATEMENT_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
HEADER_RE = re.compile(r"^#\s*([A-Z][A-Z ]+)\s*$")
BULLET_RE = re.compile(r"^\s*(?:[-*]\s+|\d+\.\s+)(.+)\s*$")
HEADING_RE = re.compile(
    r"^\s*(introduction|background|methods?|materials? and methods?|participants?|results?|discussion|conclusions?)\s*$",
    re.IGNORECASE,
)
STRUCTURED_PREFIX_RE = re.compile(
    r"^\s*(objective|objectives|background|aim|aims|method|methods|design|results|conclusion|conclusions)\s*:\s*(.+)$",
    re.IGNORECASE,
)
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
REFERENCE_NOISE_RE = re.compile(r"\b(doi:|et al\b|ajp\.psychiatryonline\.org|copyright|all rights reserved)\b", re.IGNORECASE)
TABLE_NOISE_RE = re.compile(r"\b(characteristic|mean\s+sd|n\s*%|education\s*\(years\)|in-scanner motion)\b", re.IGNORECASE)
HEADER_NOISE_RE = re.compile(r"\b(am j psychiatry|connectome-wide analysis|multivariate distance-based matrix regression)\b", re.IGNORECASE)
STAT_RE = re.compile(r"\b(t|f|z|p)\s*[=<>]\s*[-+]?\d+(?:\.\d+)?", re.IGNORECASE)
CITATION_RE = re.compile(r"\(\d{1,3}(?:\s*,\s*\d{1,3})*\)")
MULTISPACE_DIGITS_RE = re.compile(r"(?:\d+(?:\.\d+)?\s+){6,}\d+(?:\.\d+)?")
METHOD_KEYWORD_RE = re.compile(
    r"\b(participants?|sample|cohort|scanner|acquisition|preprocess|covariate|regression|mdmr|seed-based|protocol|exclusion|inclusion)\b",
    re.IGNORECASE,
)
RESULT_KEYWORD_RE = re.compile(
    r"\b(found|identified|associated|increased|decreased|significant|connectivity|effect|cluster|module)\b",
    re.IGNORECASE,
)
RESULT_OUTCOME_RE = re.compile(
    r"\b(found|identified|revealed|associated|increased|decreased|higher|lower|significant|t\s*=|p\s*[<=>])\b",
    re.IGNORECASE,
)
DISCUSSION_KEYWORD_RE = re.compile(
    r"\b(suggest|interpret|implication|limitation|consistent|may reflect|speculate)\b",
    re.IGNORECASE,
)
CONCLUSION_KEYWORD_RE = re.compile(
    r"\b(in conclusion|overall|these findings|support|future|clinical implication|longitudinal)\b",
    re.IGNORECASE,
)
DISCUSSION_RESCUE_RE = re.compile(
    r"\b(to date|in our study|in contrast|consistent with|may reflect|limitations?|interpret|implication)\b",
    re.IGNORECASE,
)
CONCLUSION_RESCUE_RE = re.compile(
    r"\b(in conclusion|overall|these findings|these results suggest|these results support|future research|longitudinal)\b",
    re.IGNORECASE,
)
PAGE_ANCHOR_RE = re.compile(r"\bpage:(\d+)\b", re.IGNORECASE)


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _canonical(value: Any) -> str:
    return _normalize_text(value).lower()


def _canonical_statement_text(value: Any) -> str:
    tokens = CANONICAL_STATEMENT_TOKEN_RE.findall(_canonical(value))
    return " ".join(tokens)


def _are_near_duplicate_lines(left: str, right: str) -> bool:
    a = _canonical_statement_text(left)
    b = _canonical_statement_text(right)
    if not a or not b:
        return False
    if a in b or b in a:
        return True
    a_tokens = a.split()
    b_tokens = b.split()
    if not a_tokens or not b_tokens:
        return False
    min_len = min(len(a_tokens), len(b_tokens))
    if min_len >= 10:
        prefix_matches = 0
        for idx in range(min_len):
            if a_tokens[idx] != b_tokens[idx]:
                break
            prefix_matches += 1
        if (prefix_matches / float(min_len)) >= 0.86:
            return True
    a_set = set(a_tokens)
    b_set = set(b_tokens)
    inter = len(a_set & b_set)
    if inter <= 0:
        return False
    overlap_max = inter / max(len(a_set), len(b_set))
    overlap_min = inter / max(1, min(len(a_set), len(b_set)))
    if overlap_max >= 0.82:
        return True
    if overlap_min >= 0.90 and overlap_max >= 0.56:
        return True
    return False


def _strip_control_chars(text: str) -> str:
    return CONTROL_CHAR_RE.sub(" ", str(text or ""))


def _normalize_candidate_text(text: str) -> str:
    clean = _strip_control_chars(text)
    clean = clean.replace("\u00ad", "")
    clean = clean.replace("ﬁ", "fi").replace("ﬂ", "fl")
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def _digit_ratio(text: str) -> float:
    value = str(text or "")
    if not value:
        return 0.0
    digits = sum(1 for ch in value if ch.isdigit())
    return digits / max(1, len(value))


def _is_noise_sentence(text: str) -> bool:
    line = _normalize_candidate_text(text)
    if not line:
        return True
    if MULTISPACE_DIGITS_RE.search(line):
        return True
    digit_ratio = _digit_ratio(line)
    if digit_ratio > 0.34 and TABLE_NOISE_RE.search(line):
        return True
    if digit_ratio > 0.42 and len(line) > 220:
        return True
    if REFERENCE_NOISE_RE.search(line) and _token_count(line) < 10:
        return True
    if HEADER_NOISE_RE.search(line) and _token_count(line) < 10:
        return True
    return False


def _tokenize(text: str) -> set[str]:
    tokens = {tok for tok in TOKEN_RE.findall(_canonical(text)) if tok and tok not in STOPWORDS}
    return tokens


def _similarity(ref_text: str, app_text: str) -> float:
    ref_tokens = _tokenize(ref_text)
    app_tokens = _tokenize(app_text)
    if not ref_tokens or not app_tokens:
        return 0.0
    overlap = len(ref_tokens & app_tokens)
    if overlap == 0:
        return 0.0
    precision = overlap / len(app_tokens)
    recall = overlap / len(ref_tokens)
    if precision + recall <= 0:
        return 0.0
    return (2.0 * precision * recall) / (precision + recall)


_EMBEDDING_CACHE: dict[str, Any] = {}


def _keyword_overlap_similarity(ref_text: str, app_text: str) -> float:
    ref_tokens = _tokenize(ref_text)
    app_tokens = _tokenize(app_text)
    if not ref_tokens or not app_tokens:
        return 0.0
    overlap = len(ref_tokens & app_tokens)
    union = len(ref_tokens | app_tokens)
    if union <= 0:
        return 0.0
    return overlap / union


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except Exception:
        return default


def _normalize_matching_mode(raw: str) -> str:
    mode = str(raw or "").strip().lower()
    return mode if mode in {"lexical", "hybrid"} else "lexical"


def _load_embedding_matcher() -> Any | None:
    model_path = os.getenv("MATCHING_EMBEDDING_MODEL", "").strip()
    if not model_path:
        return None
    if model_path in _EMBEDDING_CACHE:
        return _EMBEDDING_CACHE[model_path]
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        _EMBEDDING_CACHE[model_path] = None
        return None
    try:
        model = SentenceTransformer(model_path)
    except Exception:
        _EMBEDDING_CACHE[model_path] = None
        return None
    _EMBEDDING_CACHE[model_path] = model
    return model


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    if len(left) != len(right):
        limit = min(len(left), len(right))
        left = left[:limit]
        right = right[:limit]
    numerator = 0.0
    left_sq = 0.0
    right_sq = 0.0
    for i, lv in enumerate(left):
        rv = right[i]
        lv = float(lv)
        rv = float(rv)
        numerator += lv * rv
        left_sq += lv * lv
        right_sq += rv * rv
    denom = (left_sq ** 0.5) * (right_sq ** 0.5)
    if denom <= 0.0:
        return 0.0
    return max(0.0, min(1.0, numerator / denom))


def _encode_text_batch(model: Any, texts: list[str], batch_size: int = 32) -> dict[int, list[float]]:
    if model is None:
        return {}
    if not texts:
        return {}
    source_texts = [str(text).strip() for text in texts if str(text).strip()]
    if not source_texts:
        return {}
    try:
        vectors = model.encode(source_texts, convert_to_numpy=False, batch_size=batch_size)
    except Exception:
        try:
            vectors = model.encode(source_texts, convert_to_tensor=False, batch_size=batch_size)
        except Exception:
            return {}
    if not vectors:
        return {}
    out: dict[int, list[float]] = {}
    for idx, vector in enumerate(vectors):
        if idx >= len(source_texts):
            break
        try:
            packed = [float(v) for v in vector]
        except Exception:
            continue
        out[idx] = packed
    return out


def _parse_reference_markdown(path: Path) -> dict[str, list[str]]:
    by_section: dict[str, list[str]] = {key: [] for key in SECTION_KEYS}
    current_key: str | None = None
    active_bullet: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        header_match = HEADER_RE.match(line.strip())
        if header_match:
            header = _normalize_text(header_match.group(1)).upper()
            current_key = SECTION_HEADER_TO_KEY.get(header)
            active_bullet = None
            continue
        if not current_key:
            continue
        bullet_match = BULLET_RE.match(line)
        if bullet_match:
            bullet = _normalize_text(bullet_match.group(1))
            if bullet:
                by_section[current_key].append(bullet)
                active_bullet = bullet
            continue
        if active_bullet and line.strip() and not line.strip().startswith("#"):
            continuation = _normalize_text(line)
            if continuation:
                merged = _normalize_text(f"{by_section[current_key][-1]} {continuation}")
                by_section[current_key][-1] = merged
                active_bullet = merged
    return by_section


def _normalize_section_label(value: str) -> str:
    text = _canonical(value)
    if not text:
        return "unknown"
    if "conclusion" in text:
        return "conclusion"
    if "discussion" in text or "limitation" in text or "implication" in text:
        return "discussion"
    if "result" in text or "finding" in text:
        return "results"
    if any(token in text for token in ("method", "material", "participant", "procedure", "protocol", "analysis", "design")):
        return "methods"
    if any(token in text for token in ("intro", "background", "objective", "aim", "hypoth", "rationale", "abstract")):
        return "introduction"
    return "unknown"


def _extract_pdf_text_rows_local(pdf_path: Path) -> list[dict[str, Any]]:
    try:
        import pypdfium2 as pdfium
    except Exception as exc:
        raise RuntimeError("pypdfium2 is required for lightweight local extraction.") from exc

    rows: list[dict[str, Any]] = []
    doc = pdfium.PdfDocument(str(pdf_path))
    paragraph_idx = 0
    active_section_raw = "unknown"

    total_pages = len(doc)
    for page_idx in range(total_pages):
        page = doc.get_page(page_idx)
        text_page = None
        try:
            text_page = page.get_textpage()
            raw_text = str(text_page.get_text_bounded() or "")
        except Exception:
            raw_text = ""
        finally:
            try:
                text_page.close()
            except Exception:
                pass
            try:
                page.close()
            except Exception:
                pass
        if not raw_text.strip():
            continue

        lines = raw_text.splitlines()
        normalized_lines: list[str] = []
        for line in lines:
            clean = _normalize_candidate_text(line)
            if not clean:
                continue
            heading = HEADING_RE.match(clean)
            if heading:
                active_section_raw = heading.group(1)
                continue
            structured = STRUCTURED_PREFIX_RE.match(clean)
            if structured:
                label = _normalize_candidate_text(structured.group(1))
                statement = _normalize_candidate_text(structured.group(2))
                if statement and not _is_noise_sentence(statement):
                    rows.append(
                        {
                            "anchor": f"page:{page_idx + 1}:structured:{paragraph_idx}",
                            "text": statement,
                            "modality": "text",
                            "section_raw_title": label,
                            "section_norm": _normalize_section_label(label),
                            "paragraph_index": paragraph_idx,
                            "page_index": page_idx,
                            "total_pages": total_pages,
                        }
                    )
                    paragraph_idx += 1
                continue
            if _is_noise_sentence(clean):
                continue
            normalized_lines.append(clean)

        if not normalized_lines:
            continue

        page_text = "\n".join(normalized_lines)
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n|(?<=\.)\s{2,}", page_text) if p.strip()]
        for para in paragraphs:
            compact = _normalize_candidate_text(para)
            if len(compact) < 40:
                continue
            if _is_noise_sentence(compact):
                continue
            rows.append(
                {
                    "anchor": f"page:{page_idx + 1}:p:{paragraph_idx}",
                    "text": compact,
                    "modality": "text",
                    "section_raw_title": active_section_raw,
                    "section_norm": _normalize_section_label(active_section_raw),
                    "paragraph_index": paragraph_idx,
                    "page_index": page_idx,
                    "total_pages": total_pages,
                }
            )
            paragraph_idx += 1
    try:
        doc.close()
    except Exception:
        pass
    return rows


def _anchor_page_index(anchor: str) -> int:
    match = PAGE_ANCHOR_RE.search(str(anchor or ""))
    if not match:
        return 0
    try:
        return max(0, int(match.group(1)) - 1)
    except Exception:
        return 0


def _rows_from_parsed_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total_pages = 1
    page_indices: list[int] = []
    for idx, chunk in enumerate(chunks):
        page_idx = _anchor_page_index(str(chunk.get("anchor", "")))
        page_indices.append(page_idx)
    if page_indices:
        total_pages = max(page_indices) + 1

    for idx, chunk in enumerate(chunks):
        modality = _canonical(chunk.get("modality", "text")) or "text"
        meta_obj: dict[str, Any] = {}
        try:
            raw_meta = chunk.get("meta")
            if raw_meta:
                parsed = json.loads(raw_meta)
                if isinstance(parsed, dict):
                    meta_obj = parsed
        except Exception:
            meta_obj = {}
        raw_content = str(chunk.get("content", "") or "")
        if modality == "figure" and not raw_content.strip():
            raw_content = str(meta_obj.get("ocr_text", "") or "")
        text = _normalize_candidate_text(raw_content)
        if len(text) < 30:
            continue
        if _is_noise_sentence(text):
            continue
        section_norm = _normalize_section_label(
            str(meta_obj.get("section_norm") or meta_obj.get("section_raw_title") or meta_obj.get("section") or "")
        )
        page_idx = _anchor_page_index(str(chunk.get("anchor", "")))
        rows.append(
            {
                "anchor": str(chunk.get("anchor", f"chunk:{idx}")),
                "text": text,
                "modality": modality,
                "section_raw_title": str(meta_obj.get("section_raw_title") or meta_obj.get("section") or "unknown"),
                "section_norm": section_norm,
                "paragraph_index": int(meta_obj.get("paragraph_index", idx) or idx),
                "page_index": page_idx,
                "total_pages": total_pages,
            }
        )
    return rows


def _sentence_split(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", _normalize_candidate_text(text))
    out: list[str] = []
    for part in parts:
        clean = _normalize_candidate_text(part)
        if len(clean) >= 30:
            if _is_noise_sentence(clean):
                continue
            out.append(clean)
    return out


def _infer_section_from_text(text: str) -> str:
    lower = _canonical(text)
    if any(tok in lower for tok in ("conclusion", "conclude", "takeaway", "overall", "future research")):
        return "conclusion"
    if any(tok in lower for tok in ("discussion", "limitation", "implication", "interpret", "to date", "in our study", "consistent with")):
        return "discussion"
    if any(tok in lower for tok in ("results", "found", "associated", "increased", "decreased", "significant", "t=", "p=", "effect size")):
        return "results"
    if any(tok in lower for tok in ("method", "participants", "sample", "scanner", "analysis", "covariate", "regression", "mdmr", "acquisition", "preprocess", "protocol")):
        return "methods"
    if any(tok in lower for tok in ("background", "objective", "aim", "hypothesis", "rationale", "anhedonia", "reward")):
        return "introduction"
    return "unknown"


def _is_section_compatible(section: str, sentence: str, source_section: str = "unknown") -> bool:
    text = _canonical(sentence)
    if not text:
        return False
    source = _canonical(source_section)
    if source == section:
        # Respect parser-provided section assignments unless clearly contradictory.
        if section == "introduction" and CONCLUSION_KEYWORD_RE.search(text):
            return False
        return True
    if section == "introduction":
        if CONCLUSION_KEYWORD_RE.search(text):
            return False
        return True
    if section == "methods":
        if CONCLUSION_KEYWORD_RE.search(text):
            return False
        if DISCUSSION_KEYWORD_RE.search(text) and not METHOD_KEYWORD_RE.search(text):
            return False
        return True
    if section == "results":
        if CONCLUSION_KEYWORD_RE.search(text) and not RESULT_KEYWORD_RE.search(text):
            return False
        return bool(RESULT_KEYWORD_RE.search(text) or STAT_RE.search(text) or METHOD_KEYWORD_RE.search(text))
    if section == "discussion":
        return bool(DISCUSSION_KEYWORD_RE.search(text) or DISCUSSION_RESCUE_RE.search(text))
    if section == "conclusion":
        return bool(CONCLUSION_KEYWORD_RE.search(text) or CONCLUSION_RESCUE_RE.search(text))
    return True


INTRO_KEYWORD_RE = re.compile(r"\b(background|objective|aim|hypothesis|rationale|anhedonia|reward)\b", re.IGNORECASE)


def _section_sentence_score(
    section: str,
    sentence: str,
    paragraph_index: int,
    page_index: int,
    total_pages: int,
) -> float:
    lower = _canonical(sentence)
    score = 0.0
    keyword_map: dict[str, tuple[str, ...]] = {
        "introduction": ("objective", "aim", "hypothesis", "rationale", "background", "reward", "anhedonia"),
        "methods": ("participants", "sample", "scanner", "acquisition", "analysis", "covariate", "regression", "mdmr"),
        "results": ("found", "associated", "significant", "increased", "decreased", "t=", "p=", "cluster"),
        "discussion": ("suggest", "interpret", "implication", "limitation", "consistent"),
        "conclusion": ("conclusion", "conclude", "takeaway", "implication", "future"),
    }
    for kw in keyword_map.get(section, ()):
        if kw in lower:
            score += 1.0
    if section == "results" and RESULT_OUTCOME_RE.search(lower):
        score += 1.2
    if section == "discussion" and DISCUSSION_RESCUE_RE.search(lower):
        score += 1.2
    if section == "conclusion" and CONCLUSION_RESCUE_RE.search(lower):
        score += 1.2
    score += max(0.0, 1.0 - min(paragraph_index, 40) / 40.0)
    score += min(len(lower), 220) / 400.0
    if section == "introduction":
        score += max(0.0, 1.0 - (page_index / max(1.0, total_pages - 1)))
    elif section in {"discussion", "conclusion"}:
        score += page_index / max(1.0, total_pages - 1)
    if CITATION_RE.search(sentence):
        score -= 0.25
    if _digit_ratio(sentence) > 0.24:
        score -= 0.35
    if _is_noise_sentence(sentence):
        score -= 1.0
    return score


def _looks_like_result_sentence(sentence: str) -> bool:
    text = _normalize_candidate_text(sentence)
    if not text:
        return False
    lower = _canonical(text)
    if RESULT_OUTCOME_RE.search(lower):
        return True
    if STAT_RE.search(text) and re.search(r"\b(connectivity|network|cluster|association|effect)\b", lower):
        return True
    return False


def _late_section_rescue_candidates(
    rows: list[dict[str, Any]],
    *,
    section: str,
    existing_keys: set[str],
    max_items: int,
) -> list[dict[str, Any]]:
    if max_items <= 0:
        return []
    ordered = sorted(
        rows,
        key=lambda row: (
            -int(row.get("page_index", 0) or 0),
            -int(row.get("paragraph_index", 0) or 0),
        ),
    )
    total_pages = max(1, max(int(row.get("total_pages", 1) or 1) for row in rows))
    out: list[dict[str, Any]] = []
    for row in ordered:
        anchor = str(row.get("anchor", ""))
        source_section = str(row.get("section_norm", "unknown"))
        page_index = int(row.get("page_index", 0) or 0)
        paragraph_index = int(row.get("paragraph_index", 0) or 0)
        for sentence in _sentence_split(str(row.get("text", ""))):
            canonical = _canonical(sentence)
            if not canonical or canonical in existing_keys:
                continue
            if section == "discussion" and not (DISCUSSION_KEYWORD_RE.search(canonical) or DISCUSSION_RESCUE_RE.search(canonical)):
                continue
            if section == "conclusion" and not (CONCLUSION_KEYWORD_RE.search(canonical) or CONCLUSION_RESCUE_RE.search(canonical)):
                continue
            if not _is_section_compatible(section, sentence, source_section):
                continue
            existing_keys.add(canonical)
            out.append(
                {
                    "statement": _normalize_candidate_text(sentence),
                    "anchor": anchor,
                    "score": _section_sentence_score(
                        section,
                        sentence,
                        paragraph_index,
                        page_index,
                        total_pages,
                    )
                    + 0.9,
                }
            )
            if len(out) >= max_items:
                return out
    return out


def _build_lightweight_summary(
    rows: list[dict[str, Any]],
    *,
    support_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    section_caps = {
        "introduction": 10,
        "methods": 18,
        "results": 14,
        "discussion": 10,
        "conclusion": 8,
    }
    candidates: dict[str, list[dict[str, Any]]] = {key: [] for key in SECTION_KEYS}
    seen: set[str] = set()
    for row in rows:
        text = str(row.get("text", ""))
        section = str(row.get("section_norm", "unknown"))
        paragraph_index = int(row.get("paragraph_index", 0) or 0)
        page_index = int(row.get("page_index", 0) or 0)
        total_pages = int(row.get("total_pages", 1) or 1)
        anchor = str(row.get("anchor", ""))
        for sentence in _sentence_split(text):
            inferred = section if section in SECTION_KEYS else _infer_section_from_text(sentence)
            if inferred not in SECTION_KEYS:
                continue
            if not _is_section_compatible(inferred, sentence, section):
                continue
            canonical = _canonical(sentence)
            if canonical in seen:
                continue
            seen.add(canonical)
            clean_sentence = _normalize_candidate_text(sentence)
            candidates[inferred].append(
                {
                    "statement": clean_sentence,
                    "anchor": anchor,
                    "score": _section_sentence_score(
                        inferred,
                        clean_sentence,
                        paragraph_index,
                        page_index,
                        total_pages,
                    ),
                }
            )

    # Supplement results with parsed table/figure text when available.
    for row in support_rows or []:
        text = str(row.get("text", ""))
        paragraph_index = int(row.get("paragraph_index", 0) or 0)
        page_index = int(row.get("page_index", 0) or 0)
        total_pages = int(row.get("total_pages", 1) or 1)
        anchor = str(row.get("anchor", ""))
        for sentence in _sentence_split(text):
            if not _looks_like_result_sentence(sentence):
                continue
            canonical = _canonical(sentence)
            if canonical in seen:
                continue
            seen.add(canonical)
            clean_sentence = _normalize_candidate_text(sentence)
            candidates["results"].append(
                {
                    "statement": clean_sentence,
                    "anchor": anchor,
                    "score": _section_sentence_score(
                        "results",
                        clean_sentence,
                        paragraph_index,
                        page_index,
                        total_pages,
                    )
                    + 0.7,
                }
            )

    sections_payload: dict[str, Any] = {}
    rescue_targets = {"discussion": 4, "conclusion": 2}
    for key in SECTION_KEYS:
        ranked = sorted(candidates[key], key=lambda x: (-float(x["score"]), str(x["anchor"]), str(x["statement"])))
        picked: list[dict[str, Any]] = []
        anchor_counts: dict[str, int] = {}
        for row in ranked:
            anchor = str(row["anchor"])
            count = anchor_counts.get(anchor, 0)
            if count >= 3:
                continue
            anchor_counts[anchor] = count + 1
            picked.append(row)
            if len(picked) >= section_caps[key]:
                break
        if key in rescue_targets and len(picked) < rescue_targets[key]:
            existing_keys = {_canonical(str(item.get("statement", ""))) for item in picked}
            rescued = _late_section_rescue_candidates(
                rows,
                section=key,
                existing_keys=existing_keys,
                max_items=min(section_caps[key] - len(picked), rescue_targets[key] - len(picked)),
            )
            if rescued:
                picked.extend(rescued)
        items = [{"statement": str(row["statement"]), "evidence": [str(row["anchor"])]} for row in picked]
        sections_payload[key] = {"items": items}

    return {
        "schema_version": 2,
        "sections": sections_payload,
        "sections_compact": {},
    }


def _dedupe_lines(lines: list[str], max_items: int) -> list[str]:
    out: list[str] = []
    seen: list[str] = []
    for line in lines:
        text = _normalize_text(line)
        key = _canonical_statement_text(text)
        if not text:
            continue
        if key and any(_are_near_duplicate_lines(key, existing) for existing in seen):
            continue
        seen.append(key or _canonical(text))
        out.append(text)
        if len(out) >= max_items:
            break
    return out


def _extract_app_sections(summary_json: dict[str, Any]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {key: [] for key in SECTION_KEYS}
    presentation = summary_json.get("presentation_evidence", {})
    sections = summary_json.get("sections", {})
    sections_compact = summary_json.get("sections_compact", {})

    for key in SECTION_KEYS:
        merged: list[str] = []

        # Primary source: presentation_evidence is the app-ready ranked set.
        if isinstance(presentation, dict):
            rows = presentation.get(key, [])
            if isinstance(rows, list):
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    statement = _normalize_text(row.get("statement"))
                    if statement:
                        merged.append(statement)

        # Secondary source: detailed section blocks.
        if isinstance(sections, dict):
            block = sections.get(key, {})
            if isinstance(block, dict):
                items = block.get("items", [])
                if isinstance(items, list):
                    for item in items:
                        if not isinstance(item, dict):
                            continue
                        statement = _normalize_text(item.get("statement"))
                        if statement:
                            merged.append(statement)

        # Fallback source: compact slots.
        if isinstance(sections_compact, dict):
            rows = sections_compact.get(key, [])
            if isinstance(rows, list):
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    if _canonical(row.get("status")) != "found":
                        continue
                    statement = _normalize_text(row.get("statement"))
                    if statement:
                        merged.append(statement)

        out[key] = merged

    out["methods"] = _dedupe_lines(out["methods"], max_items=36)
    for key in ("introduction", "results", "discussion", "conclusion"):
        out[key] = _dedupe_lines(out[key], max_items=30)
    return out


def _resolve_match_runtime(
    *,
    match_threshold: float,
    matching_mode: str,
) -> dict[str, Any]:
    mode = _normalize_matching_mode(matching_mode)
    effective_threshold = _float_env("MATCHING_HYBRID_THRESHOLD", float(match_threshold)) if mode == "hybrid" else float(match_threshold)
    match_config_note = ""

    embedding_model: Any | None = _load_embedding_matcher() if mode == "hybrid" else None
    component_weights = {
        "lexical": 1.0,
        "keyword": 0.0,
        "embedding": 0.0,
    }
    if mode == "hybrid":
        if embedding_model is not None:
            component_weights = {"lexical": 0.48, "keyword": 0.34, "embedding": 0.18}
        else:
            component_weights = {"lexical": 1.0, "keyword": 0.0, "embedding": 0.0}
            effective_threshold = _float_env(
                "MATCHING_HYBRID_NO_EMBEDDING_THRESHOLD",
                effective_threshold,
            )
            match_config_note = "hybrid_no_embedding_fallback"
    return {
        "mode": mode,
        "effective_threshold": effective_threshold,
        "match_config_note": match_config_note,
        "embedding_model": embedding_model,
        "component_weights": component_weights,
    }


def _score_candidate_pair(
    ref_text: str,
    app_text: str,
    *,
    mode: str,
    component_weights: dict[str, float],
    ref_vector: list[float] | None = None,
    app_vector: list[float] | None = None,
) -> tuple[float, dict[str, float], str]:
    lexical_score = _similarity(ref_text, app_text)
    keyword_score = _keyword_overlap_similarity(ref_text, app_text)
    embedding_score = 0.0
    if mode == "hybrid" and ref_vector is not None and app_vector is not None:
        embedding_score = _cosine_similarity(ref_vector, app_vector)

    if mode == "hybrid":
        score = (
            (component_weights["lexical"] * lexical_score)
            + (component_weights["keyword"] * keyword_score)
            + (component_weights["embedding"] * embedding_score)
        )
        if lexical_score >= keyword_score and lexical_score >= embedding_score:
            reason = "lexical_match"
        elif keyword_score >= embedding_score:
            reason = "keyword_match"
        else:
            reason = "embedding_match"
    else:
        score = lexical_score
        reason = "lexical_match"
    return score, {"lexical": lexical_score, "keyword": keyword_score, "embedding": embedding_score}, reason


def _compute_sentence_inclusion_metrics(
    app_sections: dict[str, list[str]],
    ref_sections: dict[str, list[str]],
    *,
    match_threshold: float = 0.42,
    matching_mode: str = "lexical",
    inclusion_threshold: float | None = None,
) -> dict[str, Any]:
    threshold = float(match_threshold if inclusion_threshold is None else inclusion_threshold)
    runtime = _resolve_match_runtime(match_threshold=threshold, matching_mode=matching_mode)
    mode = str(runtime["mode"])
    effective_threshold = float(runtime["effective_threshold"])
    match_config_note = str(runtime["match_config_note"])
    embedding_model = runtime["embedding_model"]
    component_weights = dict(runtime["component_weights"])

    all_app_pairs: list[tuple[str, str]] = []
    for section in SECTION_KEYS:
        for line in _dedupe_lines(app_sections.get(section, []), max_items=400):
            all_app_pairs.append((section, line))

    all_app_vectors: dict[int, list[float]] = {}
    if mode == "hybrid" and embedding_model is not None and all_app_pairs:
        all_app_vectors = _encode_text_batch(embedding_model, [line for _, line in all_app_pairs])

    section_metrics: dict[str, Any] = {}
    total_ref_sentences = 0
    total_included_sentences = 0
    total_included_any_section_sentences = 0
    matched_app_indices_global: set[int] = set()

    for section in SECTION_KEYS:
        refs = _dedupe_lines(ref_sections.get(section, []), max_items=400)
        section_app_indices = [idx for idx, (app_section, _) in enumerate(all_app_pairs) if app_section == section]
        ref_vectors: dict[int, list[float]] = {}
        if mode == "hybrid" and embedding_model is not None and refs:
            ref_vectors = _encode_text_batch(embedding_model, refs)

        included_count = 0
        included_any_section_count = 0
        matched_app_indices_section: set[int] = set()
        missing_samples: list[dict[str, Any]] = []

        for ref_idx, ref in enumerate(refs):
            ref_vector = ref_vectors.get(ref_idx)
            best_same_score = 0.0
            best_any_score = 0.0
            best_same_idx = -1
            best_any_idx = -1
            best_same_payload = {"lexical": 0.0, "keyword": 0.0, "embedding": 0.0}
            best_any_payload = {"lexical": 0.0, "keyword": 0.0, "embedding": 0.0}

            for app_idx, (_, app_line) in enumerate(all_app_pairs):
                app_vector = all_app_vectors.get(app_idx)
                score, payload, _reason = _score_candidate_pair(
                    ref,
                    app_line,
                    mode=mode,
                    component_weights=component_weights,
                    ref_vector=ref_vector,
                    app_vector=app_vector,
                )
                if app_idx in section_app_indices and score > best_same_score:
                    best_same_score = score
                    best_same_idx = app_idx
                    best_same_payload = payload
                if score > best_any_score:
                    best_any_score = score
                    best_any_idx = app_idx
                    best_any_payload = payload

            same_section_hit = best_same_idx >= 0 and best_same_score >= effective_threshold
            any_section_hit = best_any_idx >= 0 and best_any_score >= effective_threshold
            if same_section_hit:
                included_count += 1
                matched_app_indices_section.add(best_same_idx)
                matched_app_indices_global.add(best_same_idx)
            if any_section_hit:
                included_any_section_count += 1
            elif len(missing_samples) < 5:
                missing_samples.append(
                    {
                        "reference": ref,
                        "best_same_section_score": round(best_same_score, 3),
                        "best_any_section_score": round(best_any_score, 3),
                        "best_same_section_components": {
                            "lexical": round(float(best_same_payload["lexical"]), 3),
                            "keyword": round(float(best_same_payload["keyword"]), 3),
                            "embedding": round(float(best_same_payload["embedding"]), 3) if mode == "hybrid" else 0.0,
                        },
                        "best_any_section_components": {
                            "lexical": round(float(best_any_payload["lexical"]), 3),
                            "keyword": round(float(best_any_payload["keyword"]), 3),
                            "embedding": round(float(best_any_payload["embedding"]), 3) if mode == "hybrid" else 0.0,
                        },
                    }
                )

        ref_count = len(refs)
        app_count = len(section_app_indices)
        sentence_inclusion_recall = (included_count / ref_count) if ref_count else 1.0
        sentence_inclusion_any_section_recall = (included_any_section_count / ref_count) if ref_count else 1.0
        section_fidelity = (included_count / included_any_section_count) if included_any_section_count else (1.0 if ref_count == 0 else 0.0)
        inclusion_precision = (len(matched_app_indices_section) / app_count) if app_count else 0.0

        section_metrics[section] = {
            "reference_sentences": ref_count,
            "app_sentences": app_count,
            "included_sentences": included_count,
            "included_any_section_sentences": included_any_section_count,
            "sentence_inclusion_recall": round(sentence_inclusion_recall, 3),
            "sentence_inclusion_any_section_recall": round(sentence_inclusion_any_section_recall, 3),
            "section_fidelity": round(section_fidelity, 3),
            "inclusion_precision": round(inclusion_precision, 3),
            "missing_top": missing_samples,
        }

        total_ref_sentences += ref_count
        total_included_sentences += included_count
        total_included_any_section_sentences += included_any_section_count

    overall_sentence_inclusion_recall = (total_included_sentences / total_ref_sentences) if total_ref_sentences else 1.0
    overall_sentence_inclusion_any_section_recall = (
        (total_included_any_section_sentences / total_ref_sentences) if total_ref_sentences else 1.0
    )
    overall_section_fidelity = (
        (total_included_sentences / total_included_any_section_sentences) if total_included_any_section_sentences else 0.0
    )
    overall_inclusion_precision = (len(matched_app_indices_global) / len(all_app_pairs)) if all_app_pairs else 0.0

    return {
        "overall_reference_sentences": total_ref_sentences,
        "overall_included_sentences": total_included_sentences,
        "overall_included_any_section_sentences": total_included_any_section_sentences,
        "overall_sentence_inclusion_recall": round(overall_sentence_inclusion_recall, 3),
        "overall_sentence_inclusion_any_section_recall": round(overall_sentence_inclusion_any_section_recall, 3),
        "overall_section_fidelity": round(overall_section_fidelity, 3),
        "overall_inclusion_precision": round(overall_inclusion_precision, 3),
        "sections": section_metrics,
        "match_mode": mode,
        "match_mode_note": match_config_note,
        "match_threshold": effective_threshold,
        "inclusion_threshold": effective_threshold,
        "match_components": component_weights,
    }


def _compare_sections(
    app_sections: dict[str, list[str]],
    ref_sections: dict[str, list[str]],
    *,
    match_threshold: float = 0.42,
    matching_mode: str = "lexical",
) -> dict[str, Any]:
    runtime = _resolve_match_runtime(match_threshold=float(match_threshold), matching_mode=matching_mode)
    mode = str(runtime["mode"])
    effective_threshold = float(runtime["effective_threshold"])
    match_config_note = str(runtime["match_config_note"])
    embedding_model = runtime["embedding_model"]
    component_weights = dict(runtime["component_weights"])

    per_section: dict[str, Any] = {}
    total_ref = 0
    total_matched = 0

    for key in SECTION_KEYS:
        refs = _dedupe_lines(ref_sections.get(key, []), max_items=200)
        apps = _dedupe_lines(app_sections.get(key, []), max_items=200)

        app_vectors: dict[int, list[float]] = {}
        ref_vectors: dict[int, list[float]] = {}
        if mode == "hybrid" and embedding_model is not None and refs and apps:
            app_vectors = _encode_text_batch(embedding_model, apps)
            ref_vectors = _encode_text_batch(embedding_model, refs)

        matched: list[dict[str, Any]] = []
        missing: list[dict[str, Any]] = []
        used_app_idx: set[int] = set()

        for ref_index, ref in enumerate(refs):
            best_idx = -1
            best_score = 0.0
            best_payload = {"lexical": 0.0, "keyword": 0.0, "embedding": 0.0}
            best_reason = "no_signal"

            for app_index, app_line in enumerate(apps):
                ref_vector = ref_vectors.get(ref_index) if mode == "hybrid" else None
                app_vector = app_vectors.get(app_index) if mode == "hybrid" else None
                score, payload, reason = _score_candidate_pair(
                    ref,
                    app_line,
                    mode=mode,
                    component_weights=component_weights,
                    ref_vector=ref_vector,
                    app_vector=app_vector,
                )

                if score <= best_score:
                    continue
                best_idx = app_index
                best_score = score
                best_payload = {
                    "lexical": round(float(payload["lexical"]), 3),
                    "keyword": round(float(payload["keyword"]), 3),
                    "embedding": round(float(payload["embedding"]), 3),
                }
                best_reason = reason

            if best_idx >= 0 and best_score >= effective_threshold:
                if best_idx not in used_app_idx:
                    used_app_idx.add(best_idx)
                matched.append(
                    {
                        "reference": ref,
                        "app_match": apps[best_idx],
                        "match_confidence": round(best_score, 3),
                        "match_mode": mode,
                        "match_reason": best_reason,
                        "match_components": best_payload,
                    }
                )
            else:
                missing.append(
                    {
                        "reference": ref,
                        "best_score": round(best_score, 3),
                        "best_app_match": apps[best_idx] if best_idx >= 0 else "",
                        "match_components": {
                            "lexical": best_payload["lexical"],
                            "keyword": best_payload["keyword"],
                            "embedding": best_payload["embedding"] if mode == "hybrid" else 0.0,
                        },
                        "match_reason": "below_threshold" if best_idx >= 0 else "no_signal",
                    }
                )

        section_ref_count = len(refs)
        section_matched_count = len(matched)
        total_ref += section_ref_count
        total_matched += section_matched_count
        recall = (section_matched_count / section_ref_count) if section_ref_count else 1.0
        precision_proxy = (len(used_app_idx) / len(apps)) if apps else 0.0
        per_section[key] = {
            "reference_points": section_ref_count,
            "app_points": len(apps),
            "matched_points": section_matched_count,
            "recall": round(recall, 3),
            "precision_proxy": round(precision_proxy, 3),
            "match_mode": mode,
            "match_mode_note": match_config_note,
            "match_threshold": effective_threshold,
            "match_components": component_weights,
            "missing_top": missing[:8],
            "matched_top": matched[:8],
        }

    sentence_inclusion_threshold = _float_env(
        "SENTENCE_INCLUSION_THRESHOLD",
        max(0.18, float(match_threshold) - 0.20),
    )
    sentence_inclusion = _compute_sentence_inclusion_metrics(
        app_sections,
        ref_sections,
        match_threshold=float(match_threshold),
        matching_mode=mode,
        inclusion_threshold=sentence_inclusion_threshold,
    )
    sentence_sections = sentence_inclusion.get("sections", {})
    if isinstance(sentence_sections, dict):
        for key in SECTION_KEYS:
            section_sentence = sentence_sections.get(key, {})
            if not isinstance(section_sentence, dict) or key not in per_section:
                continue
            per_section[key]["sentence_inclusion_recall"] = section_sentence.get("sentence_inclusion_recall")
            per_section[key]["sentence_inclusion_any_section_recall"] = section_sentence.get("sentence_inclusion_any_section_recall")
            per_section[key]["section_fidelity"] = section_sentence.get("section_fidelity")
            per_section[key]["inclusion_precision"] = section_sentence.get("inclusion_precision")

    overall_recall = (total_matched / total_ref) if total_ref else 1.0
    return {
        "overall_reference_points": total_ref,
        "overall_matched_points": total_matched,
        "overall_recall": round(overall_recall, 3),
        "overall_sentence_inclusion_recall": sentence_inclusion.get("overall_sentence_inclusion_recall"),
        "overall_sentence_inclusion_any_section_recall": sentence_inclusion.get("overall_sentence_inclusion_any_section_recall"),
        "overall_section_fidelity": sentence_inclusion.get("overall_section_fidelity"),
        "overall_inclusion_precision": sentence_inclusion.get("overall_inclusion_precision"),
        "sentence_inclusion_threshold": sentence_inclusion.get("inclusion_threshold"),
        "sections": per_section,
        "match_mode": mode,
        "match_mode_note": match_config_note,
        "match_components": component_weights,
        "match_threshold": effective_threshold,
        "sentence_inclusion": sentence_inclusion,
    }


def _token_count(text: str) -> int:
    return len(TOKEN_RE.findall(_canonical(text)))


def _line_noise_flags(line: str) -> list[str]:
    flags: list[str] = []
    text = str(line or "")
    if not text:
        return flags
    if CONTROL_CHAR_RE.search(text):
        flags.append("control_chars")
    if REFERENCE_NOISE_RE.search(text):
        flags.append("reference_noise")
    if TABLE_NOISE_RE.search(text):
        flags.append("table_noise")
    if HEADER_NOISE_RE.search(text):
        flags.append("header_noise")
    return flags


def _build_discrepancy_diagnostics(
    app_sections: dict[str, list[str]],
    ref_sections: dict[str, list[str]],
    comparison: dict[str, Any],
) -> dict[str, Any]:
    per_section = comparison.get("sections", {})
    sentence_inclusion = comparison.get("sentence_inclusion", {})
    sentence_per_section = sentence_inclusion.get("sections", {}) if isinstance(sentence_inclusion, dict) else {}
    section_diag: dict[str, Any] = {}
    low_recall_sections: list[str] = []
    low_sentence_inclusion_sections: list[str] = []
    noisy_total = 0
    coverage_total_gap = 0

    for section in SECTION_KEYS:
        app_lines = app_sections.get(section, [])
        ref_lines = ref_sections.get(section, [])
        recall = float(per_section.get(section, {}).get("recall", 0.0) or 0.0)
        sentence_recall = float(sentence_per_section.get(section, {}).get("sentence_inclusion_recall", 0.0) or 0.0)
        sentence_any_recall = float(
            sentence_per_section.get(section, {}).get("sentence_inclusion_any_section_recall", 0.0) or 0.0
        )
        section_fidelity = float(sentence_per_section.get(section, {}).get("section_fidelity", 0.0) or 0.0)
        inclusion_precision = float(sentence_per_section.get(section, {}).get("inclusion_precision", 0.0) or 0.0)
        if recall < 0.2:
            low_recall_sections.append(section)
        if sentence_recall < 0.2:
            low_sentence_inclusion_sections.append(section)
        gap = max(0, len(ref_lines) - len(app_lines))
        coverage_total_gap += gap

        noisy_examples: list[dict[str, Any]] = []
        noisy_count = 0
        for line in app_lines:
            flags = _line_noise_flags(line)
            if not flags:
                continue
            noisy_count += 1
            if len(noisy_examples) < 3:
                noisy_examples.append({"line": _normalize_text(line), "flags": flags})
        noisy_total += noisy_count

        app_mean_tokens = round(sum(_token_count(line) for line in app_lines) / max(1, len(app_lines)), 1)
        ref_mean_tokens = round(sum(_token_count(line) for line in ref_lines) / max(1, len(ref_lines)), 1)
        section_diag[section] = {
            "recall": round(recall, 3),
            "app_points": len(app_lines),
            "reference_points": len(ref_lines),
            "coverage_gap": gap,
            "mean_app_tokens": app_mean_tokens,
            "mean_reference_tokens": ref_mean_tokens,
            "noisy_line_count": noisy_count,
            "noisy_line_examples": noisy_examples,
            "sentence_inclusion_recall": round(sentence_recall, 3),
            "sentence_inclusion_any_section_recall": round(sentence_any_recall, 3),
            "section_fidelity": round(section_fidelity, 3),
            "inclusion_precision": round(inclusion_precision, 3),
        }

    likely_causes: list[str] = []
    if any(section in low_recall_sections for section in ("discussion", "conclusion")):
        likely_causes.append("Late-section extraction is weak; section boundaries likely drift after methods/results-heavy text.")
    if section_diag.get("methods", {}).get("coverage_gap", 0) >= 10:
        likely_causes.append("Methods under-coverage remains high; ranking is selecting fewer protocol-detail lines than reference.")
    if noisy_total >= 3:
        likely_causes.append("PDF artifact noise (headers/table strings/control chars) is entering ranked sentences and displacing key content.")
    if float(comparison.get("overall_recall", 0.0) or 0.0) < 0.2:
        likely_causes.append("Current lexical matching threshold misses semantically similar lines; embedding-based matching would raise measured recall.")
    overall_sentence_inclusion_recall = float(comparison.get("overall_sentence_inclusion_recall", 0.0) or 0.0)
    if float(comparison.get("overall_recall", 0.0) or 0.0) < 0.2 and overall_sentence_inclusion_recall >= 0.3:
        likely_causes.append("Extraction coverage is stronger than strict point-match recall indicates; metric mismatch is suppressing headline score.")

    return {
        "overall_recall": float(comparison.get("overall_recall", 0.0) or 0.0),
        "overall_sentence_inclusion_recall": overall_sentence_inclusion_recall,
        "overall_sentence_inclusion_any_section_recall": float(
            comparison.get("overall_sentence_inclusion_any_section_recall", 0.0) or 0.0
        ),
        "overall_section_fidelity": float(comparison.get("overall_section_fidelity", 0.0) or 0.0),
        "overall_inclusion_precision": float(comparison.get("overall_inclusion_precision", 0.0) or 0.0),
        "low_recall_sections": low_recall_sections,
        "low_sentence_inclusion_sections": low_sentence_inclusion_sections,
        "coverage_total_gap": coverage_total_gap,
        "noisy_line_total": noisy_total,
        "section_diagnostics": section_diag,
        "likely_causes": likely_causes,
    }


def _write_section_markdown(path: Path, title: str, sections: dict[str, list[str]], meta: dict[str, Any]) -> None:
    lines: list[str] = [f"# {title}", ""]
    for key, value in meta.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    for section in SECTION_KEYS:
        lines.append(f"# {SECTION_HEADERS[section]}")
        lines.append("")
        rows = sections.get(section, [])
        if not rows:
            lines.append("* N/A")
            lines.append("")
            continue
        for row in rows:
            lines.append(f"* {_normalize_text(row)}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_comparison_markdown(path: Path, comparison: dict[str, Any], runtime_meta: dict[str, Any]) -> None:
    lines: list[str] = [
        "# Comparative Analysis",
        "",
        f"- runtime_seconds: {runtime_meta.get('runtime_seconds')}",
        f"- job_status: {runtime_meta.get('job_status')}",
        f"- document_id: {runtime_meta.get('document_id')}",
        f"- job_id: {runtime_meta.get('job_id')}",
        "",
        f"- overall_reference_points: {comparison.get('overall_reference_points')}",
        f"- overall_matched_points: {comparison.get('overall_matched_points')}",
        f"- overall_recall: {comparison.get('overall_recall')}",
        f"- overall_sentence_inclusion_recall: {comparison.get('overall_sentence_inclusion_recall')}",
        f"- overall_sentence_inclusion_any_section_recall: {comparison.get('overall_sentence_inclusion_any_section_recall')}",
        f"- overall_section_fidelity: {comparison.get('overall_section_fidelity')}",
        f"- overall_inclusion_precision: {comparison.get('overall_inclusion_precision')}",
        f"- sentence_inclusion_threshold: {comparison.get('sentence_inclusion_threshold')}",
        f"- match_mode: {comparison.get('match_mode')}",
        f"- match_threshold: {comparison.get('match_threshold')}",
        "",
    ]
    diagnostics = comparison.get("discrepancy_diagnostics", {})
    if isinstance(diagnostics, dict):
        lines.append("## Diagnostic Summary")
        lines.append(f"- coverage_total_gap: {diagnostics.get('coverage_total_gap')}")
        lines.append(f"- noisy_line_total: {diagnostics.get('noisy_line_total')}")
        low_recall_sections = diagnostics.get("low_recall_sections", [])
        if low_recall_sections:
            lines.append(f"- low_recall_sections: {', '.join(str(v) for v in low_recall_sections)}")
        low_sentence_sections = diagnostics.get("low_sentence_inclusion_sections", [])
        if low_sentence_sections:
            lines.append(f"- low_sentence_inclusion_sections: {', '.join(str(v) for v in low_sentence_sections)}")
        likely_causes = diagnostics.get("likely_causes", [])
        if likely_causes:
            lines.append("- likely_causes:")
            for cause in likely_causes:
                lines.append(f"  - {cause}")
        lines.append("")
    sections = comparison.get("sections", {})
    for section in SECTION_KEYS:
        section_payload = sections.get(section, {})
        lines.append(f"## {SECTION_HEADERS[section]}")
        lines.append(f"- reference_points: {section_payload.get('reference_points')}")
        lines.append(f"- app_points: {section_payload.get('app_points')}")
        lines.append(f"- matched_points: {section_payload.get('matched_points')}")
        lines.append(f"- recall: {section_payload.get('recall')}")
        lines.append(f"- precision_proxy: {section_payload.get('precision_proxy')}")
        lines.append(f"- sentence_inclusion_recall: {section_payload.get('sentence_inclusion_recall')}")
        lines.append(f"- sentence_inclusion_any_section_recall: {section_payload.get('sentence_inclusion_any_section_recall')}")
        lines.append(f"- section_fidelity: {section_payload.get('section_fidelity')}")
        lines.append(f"- inclusion_precision: {section_payload.get('inclusion_precision')}")
        missing_top = section_payload.get("missing_top", [])
        if missing_top:
            lines.append("- missing_top:")
            for item in missing_top:
                lines.append(f"  - ({item.get('best_score')}) {item.get('reference')}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _run_pipeline_for_pdf(pdf_path: Path) -> dict[str, Any]:
    os.chdir(ROOT)
    os.environ.setdefault("PAPER_EVAL_ROOT", str(ROOT))
    sys.path.insert(0, str(BACKEND_DIR))

    from sqlmodel import Session, select

    from app.db.models import Asset, Document, Job, JobStatus, Report
    from app.db.session import engine, init_db
    from app.services.pipeline import run_pipeline
    from app.services.storage import asset_path, artifacts_dir, ensure_document_dirs

    init_db()
    with Session(engine) as session:
        document = Document(title=pdf_path.stem, source_url=f"file://{pdf_path}")
        session.add(document)
        session.commit()
        session.refresh(document)
        ensure_document_dirs(int(document.id))
        dest = asset_path(int(document.id), pdf_path.name)
        shutil.copy2(pdf_path, dest)
        session.add(
            Asset(
                document_id=int(document.id),
                kind="main",
                filename=pdf_path.name,
                content_type="application/pdf",
                path=str(dest),
            )
        )
        job = Job(document_id=int(document.id), status=JobStatus.queued, progress=0.0, message="Queued")
        session.add(job)
        session.commit()
        session.refresh(job)
        doc_id = int(document.id)
        job_id = int(job.id)

    start = time.time()
    run_pipeline(job_id)
    runtime_seconds = round(time.time() - start, 3)

    with Session(engine) as session:
        job = session.get(Job, job_id)
        report = session.exec(
            select(Report).where(Report.document_id == doc_id).order_by(Report.created_at.desc())
        ).first()
        if not job:
            raise RuntimeError(f"Job {job_id} is missing after run.")
        if job.status != JobStatus.completed:
            raise RuntimeError(f"Job {job_id} did not complete: status={job.status} message={job.message}")
        if not report:
            raise RuntimeError(f"No report generated for document {doc_id}.")
        summary_json = json.loads(report.payload)

    diagnostics_path = artifacts_dir(doc_id) / "analysis_diagnostics.json"
    diagnostics: dict[str, Any] = {}
    if diagnostics_path.exists():
        try:
            diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        except Exception:
            diagnostics = {}
    return {
        "document_id": doc_id,
        "job_id": job_id,
        "runtime_seconds": runtime_seconds,
        "summary_json": summary_json,
        "diagnostics": diagnostics,
        "mode": "pipeline",
        "fallback_note": "",
    }


def _run_parse_only_for_pdf(pdf_path: Path) -> dict[str, Any]:
    os.chdir(ROOT)
    os.environ.setdefault("PAPER_EVAL_ROOT", str(ROOT))
    sys.path.insert(0, str(BACKEND_DIR))

    from sqlmodel import Session, select

    from app.db.models import Asset, Chunk, Document
    from app.db.session import engine, init_db
    from app.services.parser import parse_document_assets
    from app.services.storage import asset_path, ensure_document_dirs

    init_db()
    doc_id = 0
    with Session(engine) as session:
        document = Document(title=pdf_path.stem, source_url=f"file://{pdf_path}")
        session.add(document)
        session.commit()
        session.refresh(document)
        doc_id = int(document.id)
        ensure_document_dirs(doc_id)
        dest = asset_path(doc_id, pdf_path.name)
        shutil.copy2(pdf_path, dest)
        asset = Asset(
            document_id=doc_id,
            kind="main",
            filename=pdf_path.name,
            content_type="application/pdf",
            path=str(dest),
        )
        session.add(asset)
        session.commit()
        start = time.time()
        counts = parse_document_assets(session, doc_id)
        parsed_chunks = session.exec(
            select(Chunk)
            .where(Chunk.document_id == doc_id)
            .where(Chunk.modality.in_(["text", "table", "figure"]))
        ).all()
        runtime = round(time.time() - start, 3)
    chunk_rows = [
        {
            "anchor": str(chunk.anchor or ""),
            "content": str(chunk.content or ""),
            "meta": str(chunk.meta or ""),
            "modality": str(chunk.modality or "text"),
        }
        for chunk in parsed_chunks
    ]
    parsed_rows = _rows_from_parsed_chunks(chunk_rows)
    text_rows = [row for row in parsed_rows if _canonical(row.get("modality")) == "text"]
    support_rows = [row for row in parsed_rows if _canonical(row.get("modality")) in {"table", "figure"}]
    rows = text_rows
    summary_json = (
        _build_lightweight_summary(text_rows, support_rows=support_rows)
        if text_rows
        else {"schema_version": 2, "sections": {}, "sections_compact": {}}
    )
    return {
        "document_id": doc_id,
        "parse_counts": counts,
        "runtime_seconds": runtime,
        "rows_extracted": len(rows),
        "support_rows_extracted": len(support_rows),
        "summary_json": summary_json,
    }


def _tail_lines(text: str, max_lines: int = 40, max_chars: int = 4000) -> str:
    clean = str(text or "")
    if not clean:
        return ""
    lines = clean.splitlines()
    tail = "\n".join(lines[-max_lines:])
    if len(tail) > max_chars:
        tail = tail[-max_chars:]
    return tail


def _run_pipeline_for_pdf_isolated(pdf_path: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    with tempfile.TemporaryDirectory(prefix="paper_eval_pipeline_") as tmp_dir:
        out_path = Path(tmp_dir) / "pipeline_result.json"
        backend_profile = os.getenv("BACKEND_PROFILE", "section-sensitive")
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--internal-run-pipeline",
            "--backend-profile",
            backend_profile,
            "--pdf",
            str(pdf_path),
            "--internal-out",
            str(out_path),
        ]
        started = time.time()
        child_env = os.environ.copy()
        child_env["PYTHONFAULTHANDLER"] = "1"
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=child_env,
            capture_output=True,
            text=True,
        )
        elapsed = round(time.time() - started, 3)
        failure_diag = {
            "isolated_pipeline": True,
            "returncode": int(proc.returncode),
            "signal": int(-proc.returncode) if proc.returncode < 0 else 0,
            "runtime_seconds": elapsed,
            "stdout_tail": _tail_lines(proc.stdout),
            "stderr_tail": _tail_lines(proc.stderr),
        }
        if proc.returncode != 0:
            return None, failure_diag
        if not out_path.exists():
            failure_diag["reason"] = "pipeline child exited 0 but no output file was produced"
            return None, failure_diag
        try:
            payload = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception as exc:
            failure_diag["reason"] = f"failed to parse child output JSON: {_normalize_text(str(exc))}"
            return None, failure_diag
        if not isinstance(payload, dict):
            failure_diag["reason"] = "child output payload is not a JSON object"
            return None, failure_diag
        diag = payload.get("diagnostics")
        merged_diag: dict[str, Any] = {}
        if isinstance(diag, dict):
            merged_diag.update(diag)
        merged_diag.update(
            {
                "isolated_pipeline": True,
                "returncode": int(proc.returncode),
                "runtime_seconds": elapsed,
            }
        )
        payload["diagnostics"] = merged_diag
        return payload, None


def _run_parse_probe_isolated(pdf_path: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="paper_eval_parse_probe_") as tmp_dir:
        out_path = Path(tmp_dir) / "parse_probe_result.json"
        backend_profile = os.getenv("BACKEND_PROFILE", "section-sensitive")
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--internal-run-parse-probe",
            "--backend-profile",
            backend_profile,
            "--pdf",
            str(pdf_path),
            "--internal-out",
            str(out_path),
        ]
        started = time.time()
        child_env = os.environ.copy()
        child_env["PYTHONFAULTHANDLER"] = "1"
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=child_env,
            capture_output=True,
            text=True,
        )
        elapsed = round(time.time() - started, 3)
        probe: dict[str, Any] = {
            "returncode": int(proc.returncode),
            "signal": int(-proc.returncode) if proc.returncode < 0 else 0,
            "runtime_seconds": elapsed,
            "stdout_tail": _tail_lines(proc.stdout),
            "stderr_tail": _tail_lines(proc.stderr),
        }
        if out_path.exists():
            try:
                probe["payload"] = json.loads(out_path.read_text(encoding="utf-8"))
            except Exception:
                probe["payload"] = {}
        return probe


def _run_analysis_stage_for_document(document_id: int, stage: str) -> dict[str, Any]:
    os.chdir(ROOT)
    os.environ.setdefault("PAPER_EVAL_ROOT", str(ROOT))
    sys.path.insert(0, str(BACKEND_DIR))

    from sqlmodel import Session, select

    from app.db.models import Asset, Chunk
    from app.db.session import engine
    from app.services.analysis.figure_analysis import analyze_figures
    from app.services.analysis.reconcile import reconcile_reports
    from app.services.analysis import runner as runner_mod
    from app.services.analysis.runner import run_full_analysis
    from app.services.analysis.supp_analysis import analyze_supplements
    from app.services.analysis.synthesis import synthesize_report
    from app.services.analysis.table_analysis import analyze_tables
    from app.services.analysis.text_analysis import analyze_text

    trace_path = str(os.getenv("SYNTHESIS_TRACE_FILE", "")).strip()

    def _trace(step: str) -> None:
        if not trace_path:
            return
        try:
            with open(trace_path, "a", encoding="utf-8") as handle:
                handle.write(f"{step}\n")
        except Exception:
            return

    started = time.time()
    with Session(engine) as session:
        assets = session.exec(select(Asset).where(Asset.document_id == document_id)).all()
        asset_kind = {asset.id: asset.kind for asset in assets}
        chunks = session.exec(select(Chunk).where(Chunk.document_id == document_id)).all()
        main_chunks = [chunk for chunk in chunks if asset_kind.get(chunk.asset_id) == "main"]
        supp_chunks = [chunk for chunk in chunks if asset_kind.get(chunk.asset_id) == "supp"]
        text_chunks = [c for c in main_chunks if c.modality == "text"]
        table_chunks = [c for c in main_chunks if c.modality == "table"]
        figure_chunks = [c for c in main_chunks if c.modality == "figure"]
        supp_modal = [c for c in supp_chunks if c.modality in {"text", "table", "figure"}]

        def _to_dict(c: Chunk) -> dict[str, Any]:
            return {
                "anchor": c.anchor,
                "content": c.content,
                "meta": c.meta,
                "modality": c.modality,
                "asset_kind": asset_kind.get(c.asset_id or -1, "main"),
                "document_source_url": "",
            }

        stage_key = _canonical(stage)
        if stage_key == "text":
            report = analyze_text([_to_dict(c) for c in text_chunks])
            return {
                "stage": "text",
                "runtime_seconds": round(time.time() - started, 3),
                "chunks": len(text_chunks),
                "packets": len(report.get("evidence_packets", [])),
            }
        if stage_key == "table":
            report = analyze_tables([_to_dict(c) for c in table_chunks])
            return {
                "stage": "table",
                "runtime_seconds": round(time.time() - started, 3),
                "chunks": len(table_chunks),
                "packets": len(report.get("evidence_packets", [])),
            }
        if stage_key == "figure":
            report = analyze_figures([_to_dict(c) for c in figure_chunks])
            return {
                "stage": "figure",
                "runtime_seconds": round(time.time() - started, 3),
                "chunks": len(figure_chunks),
                "packets": len(report.get("evidence_packets", [])),
            }
        if stage_key in {"supp", "supplement"}:
            report = analyze_supplements([_to_dict(c) for c in supp_modal])
            return {
                "stage": "supplement",
                "runtime_seconds": round(time.time() - started, 3),
                "chunks": len(supp_modal),
                "packets": len(report.get("evidence_packets", [])),
            }
        if stage_key == "synthesis":
            text_report = analyze_text([_to_dict(c) for c in text_chunks])
            table_report = analyze_tables([_to_dict(c) for c in table_chunks])
            figure_report = analyze_figures([_to_dict(c) for c in figure_chunks])
            supp_report = analyze_supplements([_to_dict(c) for c in supp_modal])
            reconcile = reconcile_reports(text_report, table_report, figure_report, supp_report)
            summary = synthesize_report(
                text_report,
                table_report,
                figure_report,
                supp_report,
                reconcile,
                paper_meta={},
                coverage={},
                text_chunk_records=[_to_dict(c) for c in text_chunks],
            )
            return {
                "stage": "synthesis",
                "runtime_seconds": round(time.time() - started, 3),
                "summary_schema_version": int(summary.get("schema_version", 0) or 0),
            }
        if stage_key == "full_reconcile":
            text_report = analyze_text([_to_dict(c) for c in text_chunks])
            table_report = analyze_tables([_to_dict(c) for c in table_chunks])
            figure_report = analyze_figures([_to_dict(c) for c in figure_chunks])
            supp_report = analyze_supplements([_to_dict(c) for c in supp_modal])
            reconcile = reconcile_reports(text_report, table_report, figure_report, supp_report)
            return {
                "stage": "full_reconcile",
                "runtime_seconds": round(time.time() - started, 3),
                "claims_total": int(reconcile.get("stats", {}).get("claims_total", 0) or 0),
                "discrepancies_total": int(reconcile.get("stats", {}).get("discrepancies_total", 0) or 0),
            }
        if stage_key == "full_coverage":
            supplement_proxy_from_main = [
                c
                for c in main_chunks
                if c.modality in {"text", "table", "figure"} and runner_mod._looks_like_supplement_chunk(c)
            ]
            supp_text_chunks = [c for c in supp_chunks if c.modality == "text"]
            supp_table_chunks = [c for c in supp_chunks if c.modality == "table"]
            supp_figure_chunks = [c for c in supp_chunks if c.modality == "figure"]
            main_expected_text_chunks = [c for c in text_chunks if not runner_mod._looks_like_supplement_chunk(c)]
            supp_expected_text_chunks = runner_mod._dedupe_chunks_by_id(
                supp_text_chunks + [c for c in supplement_proxy_from_main if c.modality == "text"]
            )
            coverage = runner_mod._compute_coverage(
                text_chunks=main_expected_text_chunks,
                table_chunks=table_chunks,
                figure_chunks=figure_chunks,
                supp_expected_text_chunks=supp_expected_text_chunks,
                supp_table_chunks=supp_table_chunks,
                supp_figure_chunks=supp_figure_chunks,
            )
            return {
                "stage": "full_coverage",
                "runtime_seconds": round(time.time() - started, 3),
                "coverage_keys": sorted(coverage.keys()) if isinstance(coverage, dict) else [],
                "fig_expected": int(coverage.get("figures", {}).get("expected", 0) or 0) if isinstance(coverage, dict) else 0,
            }
        if stage_key == "synthesis_with_coverage_only":
            _trace("stage:synthesis_with_coverage_only:start")
            supplement_proxy_from_main = [
                c
                for c in main_chunks
                if c.modality in {"text", "table", "figure"} and runner_mod._looks_like_supplement_chunk(c)
            ]
            supp_analysis_chunks = runner_mod._dedupe_chunks_by_id(
                [c for c in supp_chunks if c.modality in {"text", "table", "figure"}]
            )
            supp_text_chunks = [c for c in supp_chunks if c.modality == "text"]
            supp_table_chunks = [c for c in supp_chunks if c.modality == "table"]
            supp_figure_chunks = [c for c in supp_chunks if c.modality == "figure"]
            main_expected_text_chunks = [c for c in text_chunks if not runner_mod._looks_like_supplement_chunk(c)]
            supp_expected_text_chunks = runner_mod._dedupe_chunks_by_id(
                supp_text_chunks + [c for c in supplement_proxy_from_main if c.modality == "text"]
            )
            _trace("stage:synthesis_with_coverage_only:prepared_chunks")
            text_report = analyze_text([_to_dict(c) for c in text_chunks])
            _trace("stage:synthesis_with_coverage_only:text_done")
            table_report = analyze_tables([_to_dict(c) for c in table_chunks])
            _trace("stage:synthesis_with_coverage_only:table_done")
            figure_report = analyze_figures([_to_dict(c) for c in figure_chunks])
            _trace("stage:synthesis_with_coverage_only:figure_done")
            supp_report = analyze_supplements([_to_dict(c) for c in supp_analysis_chunks])
            _trace("stage:synthesis_with_coverage_only:supp_done")
            reconcile = reconcile_reports(text_report, table_report, figure_report, supp_report)
            _trace("stage:synthesis_with_coverage_only:reconcile_done")
            coverage = runner_mod._compute_coverage(
                text_chunks=main_expected_text_chunks,
                table_chunks=table_chunks,
                figure_chunks=figure_chunks,
                supp_expected_text_chunks=supp_expected_text_chunks,
                supp_table_chunks=supp_table_chunks,
                supp_figure_chunks=supp_figure_chunks,
            )
            _trace("stage:synthesis_with_coverage_only:coverage_done")
            _trace("stage:synthesis_with_coverage_only:synthesize_begin")
            summary = synthesize_report(
                text_report,
                table_report,
                figure_report,
                supp_report,
                reconcile,
                paper_meta={},
                coverage=coverage,
                text_chunk_records=[_to_dict(c) for c in text_chunks],
            )
            _trace("stage:synthesis_with_coverage_only:synthesize_done")
            return {
                "stage": "synthesis_with_coverage_only",
                "runtime_seconds": round(time.time() - started, 3),
                "summary_schema_version": int(summary.get("schema_version", 0) or 0),
            }
        if stage_key == "synthesis_with_meta_only":
            supp_analysis_chunks = runner_mod._dedupe_chunks_by_id(
                [c for c in supp_chunks if c.modality in {"text", "table", "figure"}]
            )
            text_report = analyze_text([_to_dict(c) for c in text_chunks])
            table_report = analyze_tables([_to_dict(c) for c in table_chunks])
            figure_report = analyze_figures([_to_dict(c) for c in figure_chunks])
            supp_report = analyze_supplements([_to_dict(c) for c in supp_analysis_chunks])
            reconcile = reconcile_reports(text_report, table_report, figure_report, supp_report)
            paper_meta = runner_mod._extract_meta([c for c in main_chunks if c.modality == "meta"])
            summary = synthesize_report(
                text_report,
                table_report,
                figure_report,
                supp_report,
                reconcile,
                paper_meta=paper_meta,
                coverage={},
                text_chunk_records=[_to_dict(c) for c in text_chunks],
            )
            return {
                "stage": "synthesis_with_meta_only",
                "runtime_seconds": round(time.time() - started, 3),
                "summary_schema_version": int(summary.get("schema_version", 0) or 0),
            }
        if stage_key == "full_synthesis_from_full":
            supplement_proxy_from_main = [
                c
                for c in main_chunks
                if c.modality in {"text", "table", "figure"} and runner_mod._looks_like_supplement_chunk(c)
            ]
            supp_analysis_chunks = runner_mod._dedupe_chunks_by_id(
                [c for c in supp_chunks if c.modality in {"text", "table", "figure"}]
            )
            supp_text_chunks = [c for c in supp_chunks if c.modality == "text"]
            supp_table_chunks = [c for c in supp_chunks if c.modality == "table"]
            supp_figure_chunks = [c for c in supp_chunks if c.modality == "figure"]
            main_expected_text_chunks = [c for c in text_chunks if not runner_mod._looks_like_supplement_chunk(c)]
            supp_expected_text_chunks = runner_mod._dedupe_chunks_by_id(
                supp_text_chunks + [c for c in supplement_proxy_from_main if c.modality == "text"]
            )
            text_report = analyze_text([_to_dict(c) for c in text_chunks])
            table_report = analyze_tables([_to_dict(c) for c in table_chunks])
            figure_report = analyze_figures([_to_dict(c) for c in figure_chunks])
            supp_report = analyze_supplements([_to_dict(c) for c in supp_analysis_chunks])
            reconcile = reconcile_reports(text_report, table_report, figure_report, supp_report)
            coverage = runner_mod._compute_coverage(
                text_chunks=main_expected_text_chunks,
                table_chunks=table_chunks,
                figure_chunks=figure_chunks,
                supp_expected_text_chunks=supp_expected_text_chunks,
                supp_table_chunks=supp_table_chunks,
                supp_figure_chunks=supp_figure_chunks,
            )
            paper_meta = runner_mod._extract_meta([c for c in main_chunks if c.modality == "meta"])
            summary = synthesize_report(
                text_report,
                table_report,
                figure_report,
                supp_report,
                reconcile,
                paper_meta=paper_meta,
                coverage=coverage,
                text_chunk_records=[_to_dict(c) for c in text_chunks],
            )
            return {
                "stage": "full_synthesis_from_full",
                "runtime_seconds": round(time.time() - started, 3),
                "summary_schema_version": int(summary.get("schema_version", 0) or 0),
            }
        if stage_key == "full_components":
            supplement_proxy_from_main = [
                c
                for c in main_chunks
                if c.modality in {"text", "table", "figure"} and runner_mod._looks_like_supplement_chunk(c)
            ]
            supp_analysis_chunks = runner_mod._dedupe_chunks_by_id(
                [c for c in supp_chunks if c.modality in {"text", "table", "figure"}]
            )
            supp_text_chunks = [c for c in supp_chunks if c.modality == "text"]
            supp_table_chunks = [c for c in supp_chunks if c.modality == "table"]
            supp_figure_chunks = [c for c in supp_chunks if c.modality == "figure"]
            main_expected_text_chunks = [c for c in text_chunks if not runner_mod._looks_like_supplement_chunk(c)]
            supp_expected_text_chunks = runner_mod._dedupe_chunks_by_id(
                supp_text_chunks + [c for c in supplement_proxy_from_main if c.modality == "text"]
            )
            text_report = analyze_text([_to_dict(c) for c in text_chunks])
            table_report = analyze_tables([_to_dict(c) for c in table_chunks])
            figure_report = analyze_figures([_to_dict(c) for c in figure_chunks])
            supp_report = analyze_supplements([_to_dict(c) for c in supp_analysis_chunks])
            reconcile = reconcile_reports(text_report, table_report, figure_report, supp_report)
            coverage = runner_mod._compute_coverage(
                text_chunks=main_expected_text_chunks,
                table_chunks=table_chunks,
                figure_chunks=figure_chunks,
                supp_expected_text_chunks=supp_expected_text_chunks,
                supp_table_chunks=supp_table_chunks,
                supp_figure_chunks=supp_figure_chunks,
            )
            paper_meta = runner_mod._extract_meta([c for c in main_chunks if c.modality == "meta"])
            summary = synthesize_report(
                text_report,
                table_report,
                figure_report,
                supp_report,
                reconcile,
                paper_meta=paper_meta,
                coverage=coverage,
                text_chunk_records=[_to_dict(c) for c in text_chunks],
            )
            return {
                "stage": "full_components",
                "runtime_seconds": round(time.time() - started, 3),
                "summary_schema_version": int(summary.get("schema_version", 0) or 0),
                "text_packets": len(text_report.get("evidence_packets", [])),
                "coverage_keys": sorted(coverage.keys()) if isinstance(coverage, dict) else [],
            }
        if stage_key == "full":
            diag = run_full_analysis(session, document_id)
            return {
                "stage": "full",
                "runtime_seconds": round(time.time() - started, 3),
                "diagnostics_keys": sorted(diag.keys()) if isinstance(diag, dict) else [],
            }
        raise RuntimeError(f"Unsupported analysis stage: {stage}")


def _run_analysis_stage_probe_isolated(document_id: int, stage: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="paper_eval_analysis_probe_") as tmp_dir:
        out_path = Path(tmp_dir) / f"analysis_probe_{stage}.json"
        trace_path = Path(tmp_dir) / f"analysis_probe_{stage}.trace.log"
        backend_profile = os.getenv("BACKEND_PROFILE", "section-sensitive")
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--internal-run-analysis-stage",
            stage,
            "--backend-profile",
            backend_profile,
            "--internal-document-id",
            str(document_id),
            "--internal-out",
            str(out_path),
        ]
        started = time.time()
        child_env = os.environ.copy()
        child_env["PYTHONFAULTHANDLER"] = "1"
        child_env["SYNTHESIS_TRACE_FILE"] = str(trace_path)
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=child_env,
            capture_output=True,
            text=True,
        )
        elapsed = round(time.time() - started, 3)
        probe: dict[str, Any] = {
            "stage": stage,
            "returncode": int(proc.returncode),
            "signal": int(-proc.returncode) if proc.returncode < 0 else 0,
            "runtime_seconds": elapsed,
            "stdout_tail": _tail_lines(proc.stdout),
            "stderr_tail": _tail_lines(proc.stderr),
        }
        if out_path.exists():
            try:
                probe["payload"] = json.loads(out_path.read_text(encoding="utf-8"))
            except Exception:
                probe["payload"] = {}
        if trace_path.exists():
            try:
                probe["synthesis_trace_tail"] = _tail_lines(
                    trace_path.read_text(encoding="utf-8"),
                    max_lines=120,
                    max_chars=12000,
                )
            except Exception:
                probe["synthesis_trace_tail"] = ""
        return probe


def _run_analysis_probe_sequence(parse_probe: dict[str, Any]) -> dict[str, Any]:
    payload = parse_probe.get("payload", {}) if isinstance(parse_probe, dict) else {}
    if not isinstance(payload, dict):
        return {"available": False, "reason": "parse probe payload missing"}
    document_id = int(payload.get("document_id", 0) or 0)
    if document_id <= 0:
        return {"available": False, "reason": "parse probe has no document_id"}

    stages = [
        "text",
        "table",
        "figure",
        "supplement",
        "synthesis",
        "full_reconcile",
        "full_coverage",
        "synthesis_with_coverage_only",
        "synthesis_with_meta_only",
        "full_synthesis_from_full",
        "full_components",
        "full",
    ]
    results: list[dict[str, Any]] = []
    failing_stage = ""
    for stage in stages:
        probe = _run_analysis_stage_probe_isolated(document_id, stage)
        results.append(probe)
        if int(probe.get("returncode", 0)) != 0:
            failing_stage = stage
            break
    return {
        "available": True,
        "document_id": document_id,
        "failing_stage": failing_stage,
        "stages": results,
    }


def _result_from_parse_probe_payload(parse_probe: dict[str, Any]) -> dict[str, Any] | None:
    payload = parse_probe.get("payload", {}) if isinstance(parse_probe, dict) else {}
    if not isinstance(payload, dict):
        return None
    summary_json = payload.get("summary_json")
    if not isinstance(summary_json, dict) or not summary_json:
        return None
    return {
        "document_id": int(payload.get("document_id", 0) or 0),
        "job_id": 0,
        "runtime_seconds": round(float(payload.get("runtime_seconds", 0.0) or 0.0), 3),
        "summary_json": summary_json,
        "diagnostics": {
            "mode": "parse_probe_text_only",
            "rows_extracted": int(payload.get("rows_extracted", 0) or 0),
            "support_rows_extracted": int(payload.get("support_rows_extracted", 0) or 0),
            "parse_counts": payload.get("parse_counts", {}),
        },
        "mode": "lightweight",
        "fallback_note": "Used isolated parse-only output with deterministic section extraction after pipeline failure.",
    }


def _pipeline_failure_note(prefix: str, failure_diag: dict[str, Any]) -> str:
    code = failure_diag.get("returncode")
    signal_num = int(failure_diag.get("signal", 0) or 0)
    reason = _normalize_text(failure_diag.get("reason", ""))
    note = f"{prefix}: isolated pipeline failed (returncode={code}"
    if signal_num:
        note += f", signal={signal_num}"
    note += ")."
    if reason:
        note += f" {reason}"
    return note


def _run_lightweight_for_pdf(pdf_path: Path) -> dict[str, Any]:
    start = time.time()
    rows = _extract_pdf_text_rows_local(pdf_path)
    if not rows:
        raise RuntimeError("No text extracted from PDF using lightweight mode.")
    summary_json = _build_lightweight_summary(rows)
    runtime_seconds = round(time.time() - start, 3)

    return {
        "document_id": 0,
        "job_id": 0,
        "runtime_seconds": runtime_seconds,
        "summary_json": summary_json,
        "diagnostics": {
            "mode": "lightweight_text_only",
            "rows_extracted": len(rows),
        },
        "mode": "lightweight",
        "fallback_note": "Used local lightweight extraction (pypdfium2 text-only, deterministic scoring) to avoid external parser dependencies.",
    }


def _can_reach_grobid(url: str) -> bool:
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    if not host:
        return False
    port = int(parsed.port or (443 if parsed.scheme == "https" else 80))
    try:
        with socket.create_connection((host, port), timeout=0.6):
            return True
    except OSError:
        return False


def _apply_backend_profile_env(profile: str) -> None:
    normalized = _canonical(profile)
    if normalized == "fast":
        # Fast profile for comparisons: deterministic text-only behavior, no expensive LLM/vision.
        os.environ["ANALYSIS_TEXT_LLM_ENABLED"] = "false"
        os.environ["ANALYSIS_SECTION_EXTRACTION_ENABLED"] = "false"
        os.environ["ANALYSIS_VERIFIER_ENABLED"] = "false"
        os.environ["ANALYSIS_NARRATIVE_OVERRIDES_ENABLED"] = "false"
        os.environ["ANALYSIS_SUMMARY_POLISH_ENABLED"] = "false"
        os.environ["DOCLING_ENABLE_OCR"] = "false"
        os.environ["DOCLING_TABLE_STRUCTURE_ENABLED"] = "false"
        os.environ["DOCLING_EXTRACT_FIGURES"] = "false"
        os.environ["FIGURE_OCR_ENABLED"] = "false"
        os.environ["ANALYSIS_MAX_FIGURES"] = "0"
        return
    if normalized == "balanced":
        # Balanced profile: keep extraction quality improvements from text/deep models,
        # while still avoiding vision/ocr-heavy work for stable runtime.
        os.environ["ANALYSIS_TEXT_LLM_ENABLED"] = "true"
        os.environ["ANALYSIS_SECTION_EXTRACTION_ENABLED"] = "true"
        os.environ["ANALYSIS_VERIFIER_ENABLED"] = "false"
        os.environ["ANALYSIS_NARRATIVE_OVERRIDES_ENABLED"] = "false"
        os.environ["ANALYSIS_SUMMARY_POLISH_ENABLED"] = "false"
        os.environ["DOCLING_ENABLE_OCR"] = "false"
        os.environ["DOCLING_TABLE_STRUCTURE_ENABLED"] = "false"
        os.environ["DOCLING_EXTRACT_FIGURES"] = "false"
        os.environ["FIGURE_OCR_ENABLED"] = "false"
        os.environ["ANALYSIS_MAX_FIGURES"] = "0"
        return
    if normalized == "section-sensitive":
        os.environ["ANALYSIS_TEXT_LLM_ENABLED"] = "true"
        os.environ["ANALYSIS_SECTION_EXTRACTION_ENABLED"] = "true"
        os.environ["ANALYSIS_VERIFIER_ENABLED"] = "false"
        os.environ["ANALYSIS_NARRATIVE_OVERRIDES_ENABLED"] = "false"
        os.environ["ANALYSIS_SUMMARY_POLISH_ENABLED"] = "false"
        os.environ["DOCLING_ENABLE_OCR"] = "false"
        os.environ["DOCLING_TABLE_STRUCTURE_ENABLED"] = "false"
        os.environ["DOCLING_EXTRACT_FIGURES"] = "false"
        os.environ["FIGURE_OCR_ENABLED"] = "false"
        os.environ["ANALYSIS_MAX_FIGURES"] = "0"
        return
    if normalized == "high-recall":
        os.environ["ANALYSIS_TEXT_LLM_ENABLED"] = "true"
        os.environ["ANALYSIS_SECTION_EXTRACTION_ENABLED"] = "true"
        os.environ["ANALYSIS_VERIFIER_ENABLED"] = "false"
        os.environ["ANALYSIS_NARRATIVE_OVERRIDES_ENABLED"] = "false"
        os.environ["ANALYSIS_SUMMARY_POLISH_ENABLED"] = "false"
        os.environ["DOCLING_ENABLE_OCR"] = "true"
        os.environ["DOCLING_TABLE_STRUCTURE_ENABLED"] = "true"
        os.environ["DOCLING_EXTRACT_FIGURES"] = "true"
        os.environ["FIGURE_OCR_ENABLED"] = "true"
        os.environ["ANALYSIS_MAX_FIGURES"] = "20"
        return
    if normalized == "full":
        os.environ["ANALYSIS_TEXT_LLM_ENABLED"] = "true"
        os.environ["ANALYSIS_SECTION_EXTRACTION_ENABLED"] = "true"
        os.environ["ANALYSIS_VERIFIER_ENABLED"] = "false"
        os.environ["ANALYSIS_NARRATIVE_OVERRIDES_ENABLED"] = "false"
        os.environ["ANALYSIS_SUMMARY_POLISH_ENABLED"] = "false"
        os.environ["DOCLING_ENABLE_OCR"] = "true"
        os.environ["DOCLING_TABLE_STRUCTURE_ENABLED"] = "true"
        os.environ["DOCLING_EXTRACT_FIGURES"] = "true"
        os.environ["FIGURE_OCR_ENABLED"] = "true"
        os.environ["ANALYSIS_MAX_FIGURES"] = "30"
        return


def _prune_output_artifacts(out_dir: Path, *, stem: str, keep_latest: int) -> int:
    if keep_latest <= 0:
        return 0
    removed = 0
    families = ("app_extraction", "comparison", "run", "model_benchmark")
    for family in families:
        family_re = re.compile(rf"^{re.escape(stem)}_{re.escape(family)}_(\d{{8}}_\d{{6}})\..+$")
        stamped: dict[str, list[Path]] = {}
        for path in out_dir.glob(f"{stem}_{family}_*"):
            if not path.is_file():
                continue
            match = family_re.match(path.name)
            if not match:
                continue
            stamped.setdefault(match.group(1), []).append(path)
        if len(stamped) <= keep_latest:
            continue
        stale_stamps = sorted(stamped.keys(), reverse=True)[keep_latest:]
        for stamp in stale_stamps:
            for artifact in stamped.get(stamp, []):
                try:
                    artifact.unlink()
                    removed += 1
                except OSError:
                    continue
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a PDF end-to-end through local pipeline and compare app extraction to a ChatGPT extraction.",
    )
    parser.add_argument("--pdf", required=False, help="Path to PDF file.")
    parser.add_argument("--reference-md", required=False, help="Path to ChatGPT extraction markdown.")
    parser.add_argument("--out-dir", default=str(ROOT / "test" / "text"), help="Output directory for artifacts.")
    parser.add_argument(
        "--mode",
        choices=["auto", "pipeline", "lightweight"],
        default="auto",
        help="Execution mode: pipeline (full app pipeline), lightweight (local text-only), auto (pipeline then lightweight fallback).",
    )
    parser.add_argument(
        "--parser-engine",
        default=os.getenv("PARSER_ENGINE", "validated"),
        help="Parser engine for pipeline mode (e.g., docling or validated).",
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="Optional sqlite path override (for isolated runs).",
    )
    parser.add_argument(
        "--backend-profile",
        choices=["fast", "balanced", "section-sensitive", "high-recall", "full"],
        default=os.getenv("BACKEND_PROFILE", "section-sensitive"),
        help="Backend execution profile for pipeline mode.",
    )
    parser.add_argument(
        "--matching-mode",
        choices=["lexical", "hybrid"],
        default=_normalize_matching_mode(os.getenv("MATCHING_MODE", "hybrid")),
        help="Reference matching strategy: lexical or hybrid lexical+keyword+embedding.",
    )
    parser.add_argument(
        "--matching-threshold",
        type=float,
        default=_float_env("MATCHING_THRESHOLD", 0.42),
        help="Minimum matching score for a match to count as recall.",
    )
    parser.add_argument(
        "--retain-runs",
        type=int,
        default=1,
        help="Number of timestamped artifact sets to keep per PDF in out-dir (0 keeps all history).",
    )
    parser.add_argument("--internal-run-pipeline", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--internal-run-parse-probe", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--internal-run-analysis-stage", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--internal-document-id", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--internal-out", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.db_path:
        db_path = Path(args.db_path).expanduser().resolve()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        os.environ["DB_PATH"] = str(db_path)
    os.environ["BACKEND_PROFILE"] = str(args.backend_profile)
    _apply_backend_profile_env(args.backend_profile)
    os.environ.setdefault("PARSER_ENGINE", str(args.parser_engine))
    os.environ["MATCHING_MODE"] = _normalize_matching_mode(str(args.matching_mode))
    os.environ["MATCHING_THRESHOLD"] = str(float(args.matching_threshold))
    os.environ["MATCHING_HYBRID_THRESHOLD"] = str(float(args.matching_threshold))

    if args.internal_run_pipeline:
        if not args.pdf:
            raise SystemExit("--internal-run-pipeline requires --pdf")
        if not args.internal_out:
            raise SystemExit("--internal-run-pipeline requires --internal-out")
        pdf_path = Path(args.pdf).expanduser().resolve()
        if not pdf_path.exists():
            raise SystemExit(f"PDF not found: {pdf_path}")
        result = _run_pipeline_for_pdf(pdf_path)
        out_path = Path(str(args.internal_out)).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result), encoding="utf-8")
        print("internal_pipeline_ok")
        return
    if args.internal_run_parse_probe:
        if not args.pdf:
            raise SystemExit("--internal-run-parse-probe requires --pdf")
        if not args.internal_out:
            raise SystemExit("--internal-run-parse-probe requires --internal-out")
        pdf_path = Path(args.pdf).expanduser().resolve()
        if not pdf_path.exists():
            raise SystemExit(f"PDF not found: {pdf_path}")
        result = _run_parse_only_for_pdf(pdf_path)
        out_path = Path(str(args.internal_out)).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result), encoding="utf-8")
        print("internal_parse_probe_ok")
        return
    if args.internal_run_analysis_stage:
        if not args.internal_out:
            raise SystemExit("--internal-run-analysis-stage requires --internal-out")
        if args.internal_document_id is None:
            raise SystemExit("--internal-run-analysis-stage requires --internal-document-id")
        try:
            document_id = int(args.internal_document_id)
        except Exception as exc:
            raise SystemExit(f"invalid --internal-document-id: {args.internal_document_id}") from exc
        result = _run_analysis_stage_for_document(document_id, args.internal_run_analysis_stage)
        out_path = Path(str(args.internal_out)).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result), encoding="utf-8")
        print("internal_analysis_stage_ok")
        return

    if not args.pdf:
        raise SystemExit("--pdf is required")
    if not args.reference_md:
        raise SystemExit("--reference-md is required")
    pdf_path = Path(args.pdf).expanduser().resolve()
    ref_path = Path(args.reference_md).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")
    if not ref_path.exists():
        raise SystemExit(f"Reference extraction not found: {ref_path}")

    run_note = ""
    if args.mode == "pipeline":
        pipeline_result, failure_diag = _run_pipeline_for_pdf_isolated(pdf_path)
        if pipeline_result is not None:
            result = pipeline_result
        else:
            parse_probe = _run_parse_probe_isolated(pdf_path)
            analysis_probe = _run_analysis_probe_sequence(parse_probe)
            result = _result_from_parse_probe_payload(parse_probe) or _run_lightweight_for_pdf(pdf_path)
            diag = result.get("diagnostics", {})
            if not isinstance(diag, dict):
                diag = {}
            diag["pipeline_failure"] = failure_diag or {}
            diag["parse_probe"] = parse_probe
            diag["analysis_probe"] = analysis_probe
            result["diagnostics"] = diag
            run_note = _pipeline_failure_note(
                "Pipeline fallback engaged",
                failure_diag or {"reason": "unknown isolated pipeline failure"},
            )
    elif args.mode == "lightweight":
        result = _run_lightweight_for_pdf(pdf_path)
    else:
        parser_engine = _canonical(os.getenv("PARSER_ENGINE", args.parser_engine))
        grobid_url = os.getenv("GROBID_URL", "http://localhost:8070")
        if parser_engine == "validated" and not _can_reach_grobid(grobid_url):
            run_note = (
                f"Auto fallback engaged before pipeline run: GROBID not reachable at {grobid_url}. "
                "Using lightweight deterministic mode."
            )
            result = _run_lightweight_for_pdf(pdf_path)
        else:
            pipeline_result, failure_diag = _run_pipeline_for_pdf_isolated(pdf_path)
            if pipeline_result is not None:
                result = pipeline_result
            else:
                parse_probe = _run_parse_probe_isolated(pdf_path)
                analysis_probe = _run_analysis_probe_sequence(parse_probe)
                result = _result_from_parse_probe_payload(parse_probe) or _run_lightweight_for_pdf(pdf_path)
                diag = result.get("diagnostics", {})
                if not isinstance(diag, dict):
                    diag = {}
                diag["pipeline_failure"] = failure_diag or {}
                diag["parse_probe"] = parse_probe
                diag["analysis_probe"] = analysis_probe
                result["diagnostics"] = diag
                run_note = _pipeline_failure_note(
                    "Auto fallback engaged after pipeline failure",
                    failure_diag or {"reason": "unknown isolated pipeline failure"},
                )

    summary_json = result["summary_json"]
    app_sections = _extract_app_sections(summary_json)
    ref_sections = _parse_reference_markdown(ref_path)
    comparison = _compare_sections(
        app_sections,
        ref_sections,
        match_threshold=float(args.matching_threshold),
        matching_mode=_normalize_matching_mode(str(args.matching_mode)),
    )
    comparison["discrepancy_diagnostics"] = _build_discrepancy_diagnostics(app_sections, ref_sections, comparison)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    app_md_path = out_dir / f"{pdf_path.stem}_app_extraction_{stamp}.md"
    cmp_md_path = out_dir / f"{pdf_path.stem}_comparison_{stamp}.md"
    cmp_json_path = out_dir / f"{pdf_path.stem}_comparison_{stamp}.json"
    run_json_path = out_dir / f"{pdf_path.stem}_run_{stamp}.json"

    run_meta = {
        "document_id": result["document_id"],
        "job_id": result["job_id"],
        "runtime_seconds": result["runtime_seconds"],
        "job_status": "completed",
        "run_mode": result.get("mode", args.mode),
        "backend_profile": str(args.backend_profile),
        "run_note": run_note or result.get("fallback_note", ""),
        "matching_mode": _normalize_matching_mode(str(args.matching_mode)),
        "matching_threshold": float(args.matching_threshold),
    }
    _write_section_markdown(
        app_md_path,
        title="App Extraction",
        sections=app_sections,
        meta=run_meta,
    )
    _write_comparison_markdown(cmp_md_path, comparison, run_meta)
    cmp_json_path.parent.mkdir(parents=True, exist_ok=True)
    cmp_json_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    run_json_path.write_text(
        json.dumps(
            {
                **run_meta,
                "pdf": str(pdf_path),
                "reference_md": str(ref_path),
                "app_extraction_md": str(app_md_path),
                "comparison_md": str(cmp_md_path),
                "comparison_json": str(cmp_json_path),
                "analysis_diagnostics": result.get("diagnostics", {}),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    pruned_files = _prune_output_artifacts(out_dir, stem=pdf_path.stem, keep_latest=max(0, int(args.retain_runs)))

    print(f"document_id={run_meta['document_id']}")
    print(f"job_id={run_meta['job_id']}")
    print(f"runtime_seconds={run_meta['runtime_seconds']}")
    print(f"run_mode={run_meta['run_mode']}")
    if run_meta.get("run_note"):
        print(f"run_note={run_meta['run_note']}")
    print(f"app_extraction_md={app_md_path}")
    print(f"comparison_md={cmp_md_path}")
    print(f"comparison_json={cmp_json_path}")
    print(f"run_json={run_json_path}")
    print(f"artifacts_pruned={pruned_files}")


if __name__ == "__main__":
    main()
