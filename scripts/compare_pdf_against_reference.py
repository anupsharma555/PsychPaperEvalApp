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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.analysis.information_retention import AUDIT_STAGES, build_information_retention_audit
from app.services.analysis.validity import build_run_validity
from scripts.compare_evidence_to_gold import (
    compare_evidence_to_gold,
    evidence_metadata_from_payload,
    evidence_packets_from_payload,
)
from scripts.validate_gold_standards import load_gold_standard

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
NUMBER_VALUE_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", re.IGNORECASE)
FIGURE_REF_RE = re.compile(r"\b(?:fig(?:ure)?s?\.?\s*)(S?\d+[A-Z]?)\b", re.IGNORECASE)
TABLE_REF_RE = re.compile(r"\b(?:tables?\.?\s*)(S?\d+[A-Z]?)\b", re.IGNORECASE)


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
    total_cross_section_only_sentences = 0
    total_missing_any_section_sentences = 0
    matched_app_indices_global: set[int] = set()

    for section in SECTION_KEYS:
        refs = _dedupe_lines(ref_sections.get(section, []), max_items=400)
        section_app_indices = [idx for idx, (app_section, _) in enumerate(all_app_pairs) if app_section == section]
        ref_vectors: dict[int, list[float]] = {}
        if mode == "hybrid" and embedding_model is not None and refs:
            ref_vectors = _encode_text_batch(embedding_model, refs)

        included_count = 0
        included_any_section_count = 0
        cross_section_only_count = 0
        missing_any_section_count = 0
        matched_app_indices_section: set[int] = set()
        missing_samples: list[dict[str, Any]] = []
        cross_section_samples: list[dict[str, Any]] = []

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
                if not same_section_hit:
                    cross_section_only_count += 1
                    if len(cross_section_samples) < 5:
                        best_section = all_app_pairs[best_any_idx][0] if best_any_idx >= 0 else ""
                        best_line = all_app_pairs[best_any_idx][1] if best_any_idx >= 0 else ""
                        same_section_line = all_app_pairs[best_same_idx][1] if best_same_idx >= 0 else ""
                        cross_section_samples.append(
                            {
                                "reference": ref,
                                "best_any_section": best_section,
                                "best_any_section_score": round(best_any_score, 3),
                                "best_any_section_line": _normalize_text(best_line),
                                "best_same_section_score": round(best_same_score, 3),
                                "best_same_section_line": _normalize_text(same_section_line),
                                "best_any_section_components": {
                                    "lexical": round(float(best_any_payload["lexical"]), 3),
                                    "keyword": round(float(best_any_payload["keyword"]), 3),
                                    "embedding": round(float(best_any_payload["embedding"]), 3) if mode == "hybrid" else 0.0,
                                },
                            }
                        )
            else:
                missing_any_section_count += 1
                if len(missing_samples) < 5:
                    best_any_section = all_app_pairs[best_any_idx][0] if best_any_idx >= 0 else ""
                    best_any_line = all_app_pairs[best_any_idx][1] if best_any_idx >= 0 else ""
                    best_same_line = all_app_pairs[best_same_idx][1] if best_same_idx >= 0 else ""
                    missing_samples.append(
                        {
                            "reference": ref,
                            "best_same_section_score": round(best_same_score, 3),
                            "best_same_section_line": _normalize_text(best_same_line),
                            "best_any_section": best_any_section,
                            "best_any_section_score": round(best_any_score, 3),
                            "best_any_section_line": _normalize_text(best_any_line),
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
            "cross_section_only_sentences": cross_section_only_count,
            "missing_any_section_sentences": missing_any_section_count,
            "cause_counts": {
                "same_section_match": included_count,
                "cross_section_only": cross_section_only_count,
                "missing_any_section": missing_any_section_count,
            },
            "sentence_inclusion_recall": round(sentence_inclusion_recall, 3),
            "sentence_inclusion_any_section_recall": round(sentence_inclusion_any_section_recall, 3),
            "section_fidelity": round(section_fidelity, 3),
            "inclusion_precision": round(inclusion_precision, 3),
            "cross_section_top": cross_section_samples,
            "missing_top": missing_samples,
        }

        total_ref_sentences += ref_count
        total_included_sentences += included_count
        total_included_any_section_sentences += included_any_section_count
        total_cross_section_only_sentences += cross_section_only_count
        total_missing_any_section_sentences += missing_any_section_count

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
        "overall_cross_section_only_sentences": total_cross_section_only_sentences,
        "overall_missing_any_section_sentences": total_missing_any_section_sentences,
        "overall_cause_counts": {
            "same_section_match": total_included_sentences,
            "cross_section_only": total_cross_section_only_sentences,
            "missing_any_section": total_missing_any_section_sentences,
        },
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
            per_section[key]["cause_counts"] = section_sentence.get("cause_counts", {})
            per_section[key]["cross_section_top"] = section_sentence.get("cross_section_top", [])

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
    cross_section_loss_sections: list[str] = []
    under_selected_sections: list[str] = []
    missing_any_section_sections: list[str] = []
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
        sentence_payload = sentence_per_section.get(section, {})
        if not isinstance(sentence_payload, dict):
            sentence_payload = {}
        cause_counts = sentence_payload.get("cause_counts", {})
        if not isinstance(cause_counts, dict):
            cause_counts = {}
        cross_section_only = int(cause_counts.get("cross_section_only", 0) or 0)
        missing_any_section = int(cause_counts.get("missing_any_section", 0) or 0)
        if sentence_any_recall - sentence_recall >= 0.25 or section_fidelity < 0.6:
            cross_section_loss_sections.append(section)
        if gap >= 4 and len(app_lines) < max(1, int(0.65 * len(ref_lines))):
            under_selected_sections.append(section)
        if len(ref_lines) and (missing_any_section / max(1, len(ref_lines))) >= 0.35:
            missing_any_section_sections.append(section)

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
            "cause_counts": {
                "same_section_match": int(cause_counts.get("same_section_match", 0) or 0),
                "cross_section_only": cross_section_only,
                "missing_any_section": missing_any_section,
            },
            "cross_section_top": sentence_payload.get("cross_section_top", [])[:3],
        }

    likely_causes: list[str] = []
    cause_details: dict[str, Any] = {
        "cross_section_loss_sections": cross_section_loss_sections,
        "under_selected_sections": under_selected_sections,
        "missing_any_section_sections": missing_any_section_sections,
    }
    if cross_section_loss_sections:
        likely_causes.append(
            "Section assignment drift: reference-like sentences are present, but assigned to different sections "
            f"for {', '.join(cross_section_loss_sections)}."
        )
    if under_selected_sections:
        likely_causes.append(
            "Section under-selection: app output has substantially fewer section points than the reference "
            f"for {', '.join(under_selected_sections)}."
        )
    if missing_any_section_sections:
        likely_causes.append(
            "Content absent from extracted app sections: reference points have no close match in any section "
            f"for {', '.join(missing_any_section_sections)}."
        )
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
        "cause_details": cause_details,
        "likely_causes": likely_causes,
    }


def _walk_values(payload: Any) -> list[Any]:
    values: list[Any] = []
    stack = [payload]
    while stack:
        current = stack.pop()
        values.append(current)
        if isinstance(current, dict):
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)
    return values


def _contains_fallback_marker(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return "fallback" in value.lower()


def _sum_numeric_mapping_values(payload: Any) -> int:
    if not isinstance(payload, dict):
        return 0
    total = 0
    for value in payload.values():
        try:
            total += int(value or 0)
        except Exception:
            continue
    return total


def _build_fallback_audit(result: dict[str, Any], run_note: str) -> dict[str, Any]:
    diagnostics = result.get("diagnostics", {})
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    if isinstance(diagnostics.get("diagnostics"), dict):
        diagnostics = diagnostics["diagnostics"]

    summary_json = result.get("summary_json", {})
    if not isinstance(summary_json, dict):
        summary_json = {}

    section_diagnostics = diagnostics.get("section_diagnostics", {})
    if not isinstance(section_diagnostics, dict):
        section_diagnostics = summary_json.get("section_diagnostics", {})
    if not isinstance(section_diagnostics, dict):
        section_diagnostics = {}

    fallback_counts = diagnostics.get("fallback_counts_by_reason", {})
    fallback_count_total = _sum_numeric_mapping_values(fallback_counts)
    fallback_notes = diagnostics.get("sections_fallback_notes", [])
    if not fallback_notes:
        fallback_notes = summary_json.get("sections_fallback_notes", [])
    if not isinstance(fallback_notes, list):
        fallback_notes = []

    fallback_sections: list[str] = []
    for section in SECTION_KEYS:
        section_payload = section_diagnostics.get(section, {})
        if isinstance(section_payload, dict) and bool(section_payload.get("fallback_used")):
            fallback_sections.append(section)

    execution_note = " ".join(
        str(part or "")
        for part in (
            run_note,
            result.get("fallback_note", ""),
            diagnostics.get("pipeline_failure", {}).get("reason", "") if isinstance(diagnostics.get("pipeline_failure"), dict) else "",
        )
    )
    execution_fallback_used = _contains_fallback_marker(execution_note)

    diagnostics_fallback_marker = any(_contains_fallback_marker(value) for value in _walk_values(diagnostics))
    sections_fallback_used = bool(
        diagnostics.get("sections_fallback_used")
        or summary_json.get("sections_fallback_used")
        or fallback_notes
        or fallback_sections
        or fallback_count_total
        or diagnostics_fallback_marker
    )

    return {
        "passed": not execution_fallback_used and not sections_fallback_used,
        "execution_fallback_used": execution_fallback_used,
        "sections_fallback_used": sections_fallback_used,
        "fallback_sections": fallback_sections,
        "fallback_count_total": fallback_count_total,
        "fallback_counts_by_reason": fallback_counts if isinstance(fallback_counts, dict) else {},
        "sections_fallback_notes": fallback_notes,
        "run_note": run_note or result.get("fallback_note", ""),
    }


def _read_nested_int(payload: dict[str, Any], path: list[str]) -> int:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return 0
        current = current.get(key)
    try:
        return int(current or 0)
    except Exception:
        return 0


def _build_quality_backend_audit(result: dict[str, Any]) -> dict[str, Any]:
    diagnostics = result.get("diagnostics", {})
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    if isinstance(diagnostics.get("diagnostics"), dict):
        diagnostics = diagnostics["diagnostics"]

    summary_json = result.get("summary_json", {})
    if not isinstance(summary_json, dict):
        summary_json = {}

    provider = str(result.get("provider") or result.get("llm_provider") or "local")
    canonical_validity = build_run_validity(
        summary_json=summary_json,
        diagnostics=diagnostics,
        job_status=result.get("job_status", "completed"),
        provider=provider,
        require_source_provenance=False,
    )
    canonical_quality = canonical_validity.get("quality_backend_audit")
    if isinstance(canonical_quality, dict):
        return canonical_quality

    model_usage = diagnostics.get("model_usage", {})
    if not isinstance(model_usage, dict):
        model_usage = summary_json.get("model_usage", {})
    if not isinstance(model_usage, dict):
        model_usage = {}

    text_calls = int(model_usage.get("text_calls", 0) or 0)
    if text_calls <= 0:
        text_calls = _read_nested_int(diagnostics, ["llm_status", "text_llm_calls"])

    section_diagnostics = diagnostics.get("section_diagnostics", {})
    if not isinstance(section_diagnostics, dict):
        section_diagnostics = summary_json.get("section_diagnostics", {})
    if not isinstance(section_diagnostics, dict):
        section_diagnostics = {}

    section_extraction_enabled = section_diagnostics.get("section_extraction_enabled")
    if section_extraction_enabled is None:
        section_extraction_enabled = summary_json.get("section_extraction_enabled")
    section_extraction_counts = section_diagnostics.get("section_extraction_counts", {})
    if not isinstance(section_extraction_counts, dict):
        section_extraction_counts = {}

    blockers = {
        "text_calls_zero": text_calls <= 0,
        "section_extraction_disabled": section_extraction_enabled is False,
    }
    return {
        "passed": not any(bool(value) for value in blockers.values()),
        "text_calls": text_calls,
        "section_extraction_enabled": section_extraction_enabled,
        "section_extraction_counts": section_extraction_counts,
        "blockers": blockers,
    }


def _has_section_source_provenance(row: dict[str, Any]) -> bool:
    for key in ("evidence_refs", "evidence", "anchors"):
        values = row.get(key)
        if isinstance(values, list) and any(str(value).strip() for value in values):
            return True
    for key in ("anchor", "source_anchor"):
        if str(row.get(key) or "").strip():
            return True
    return False


def _section_source_rows(summary_json: dict[str, Any]) -> list[dict[str, Any]]:
    rows_out: list[dict[str, Any]] = []
    section_sources = (
        ("presentation_evidence", "statement"),
        ("sections_compact", "statement"),
    )
    for source_key, text_key in section_sources:
        source = summary_json.get(source_key, {})
        if not isinstance(source, dict):
            continue
        for section in SECTION_KEYS:
            rows = source.get(section, [])
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                if source_key == "sections_compact" and _canonical(row.get("status")) != "found":
                    continue
                if _normalize_text(row.get(text_key)):
                    rows_out.append(row)

    sections = summary_json.get("sections", {})
    if isinstance(sections, dict):
        for section in SECTION_KEYS:
            block = sections.get(section, {})
            if not isinstance(block, dict):
                continue
            items = block.get("items", [])
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, dict) and _normalize_text(item.get("statement")):
                    rows_out.append(item)
    return rows_out


def _build_source_provenance_audit(summary_json: dict[str, Any]) -> dict[str, Any]:
    rows = _section_source_rows(summary_json)
    missing_count = sum(1 for row in rows if not _has_section_source_provenance(row))
    return {
        "passed": bool(rows) and missing_count == 0,
        "section_source_rows": len(rows),
        "missing_provenance_rows": missing_count,
    }


def _build_benchmark_validity_gate(
    result: dict[str, Any],
    run_note: str,
    *,
    job_status: str | None = None,
) -> dict[str, Any]:
    diagnostics = result.get("diagnostics")
    diagnostics_missing = not isinstance(diagnostics, dict) or not diagnostics
    summary_json = result.get("summary_json", {})
    if not isinstance(summary_json, dict):
        summary_json = {}

    provider = str(result.get("provider") or result.get("llm_provider") or "local")
    canonical_validity = build_run_validity(
        summary_json=summary_json,
        diagnostics=diagnostics if isinstance(diagnostics, dict) else {},
        run_note=run_note,
        job_status=job_status if job_status is not None else result.get("job_status", "completed"),
        provider=provider,
        require_source_provenance=False,
    )
    fallback_audit = canonical_validity.get("fallback_audit", {})
    if not isinstance(fallback_audit, dict):
        fallback_audit = _build_fallback_audit(result, run_note)
    quality_backend_audit = canonical_validity.get("quality_backend_audit", {})
    if not isinstance(quality_backend_audit, dict):
        quality_backend_audit = _build_quality_backend_audit(result)
    source_provenance_audit = _build_source_provenance_audit(summary_json)

    status = _canonical(job_status if job_status is not None else result.get("job_status", "completed"))
    status_completed = status == "completed" or status.endswith(".completed")
    canonical_blockers = (
        dict(canonical_validity.get("blockers"))
        if isinstance(canonical_validity.get("blockers"), dict)
        else {}
    )
    blockers = {
        **canonical_blockers,
        "job_not_completed": not status_completed,
        "fallback_engaged": not bool(fallback_audit.get("passed")),
        "text_llm_calls_zero": bool(quality_backend_audit.get("blockers", {}).get("text_calls_zero"))
        if isinstance(quality_backend_audit.get("blockers"), dict)
        else False,
        "section_extraction_disabled": bool(
            quality_backend_audit.get("blockers", {}).get("section_extraction_disabled")
        )
        if isinstance(quality_backend_audit.get("blockers"), dict)
        else False,
        "diagnostics_missing": diagnostics_missing,
        "source_provenance_missing": not bool(source_provenance_audit.get("passed")),
    }
    reasons = [key for key, value in blockers.items() if value]
    valid = not reasons
    return {
        "valid": valid,
        "run_validity": "valid" if valid else "invalid",
        "benchmark_valid": valid,
        "failure_type": None if valid else "infrastructure",
        "reasons": reasons,
        "blockers": blockers,
        "canonical_run_validity": canonical_validity,
        "fallback_audit": fallback_audit,
        "quality_backend_audit": quality_backend_audit,
        "source_provenance_audit": source_provenance_audit,
    }


def _format_benchmark_validity_failure(gate: dict[str, Any]) -> str:
    reasons = gate.get("reasons", [])
    if not isinstance(reasons, list):
        reasons = []
    reason_text = ", ".join(str(reason) for reason in reasons) or "unknown"
    return f"Invalid benchmark run (infrastructure failure): {reason_text}"


def _load_gold_claims_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _default_gold_claims_path(reference_path: Path) -> Path | None:
    if reference_path.name == "sharma_2017_chatgpt_extraction.md":
        candidate = ROOT / "benchmarks" / "sharma_2017_gold_claims.jsonl"
        if candidate.exists():
            return candidate
    return None


def _number_tokens_from_text(text: str) -> set[str]:
    tokens: set[str] = set()
    for raw in NUMBER_VALUE_RE.findall(str(text or "")):
        try:
            number = float(raw)
        except ValueError:
            continue
        if number.is_integer():
            tokens.add(str(int(number)))
        tokens.add(f"{number:.8g}")
    return tokens


def _number_tokens_from_lines(lines: list[str]) -> set[str]:
    tokens: set[str] = set()
    for line in lines:
        tokens.update(_number_tokens_from_text(line))
    return tokens


def _number_tokens_from_gold_claims(gold_claims: list[dict[str, Any]]) -> set[str]:
    tokens: set[str] = set()
    for claim in gold_claims:
        expected_numbers = claim.get("expected_numbers", [])
        if not isinstance(expected_numbers, list):
            continue
        for item in expected_numbers:
            if not isinstance(item, dict):
                continue
            tokens.update(_number_tokens_from_text(str(item.get("value", ""))))
    return tokens


def _collect_modality_refs(sections: dict[str, list[str]]) -> dict[str, set[str]]:
    text = "\n".join(line for rows in sections.values() for line in rows)
    figure_refs = {match.upper() for match in FIGURE_REF_RE.findall(text)}
    table_refs = {match.upper() for match in TABLE_REF_RE.findall(text)}
    return {
        "figure": {ref for ref in figure_refs if not ref.startswith("S")},
        "table": {ref for ref in table_refs if not ref.startswith("S")},
        "supplement": {ref for ref in figure_refs | table_refs if ref.startswith("S")},
    }


def _recall_payload(expected: set[str], observed: set[str]) -> dict[str, Any]:
    matched = sorted(expected & observed)
    missing = sorted(expected - observed)
    return {
        "expected": len(expected),
        "observed": len(observed),
        "matched": len(matched),
        "recall": round((len(matched) / len(expected)) if expected else 1.0, 3),
        "missing_refs": missing[:12],
    }


def _top_unmatched_app_lines(
    app_sections: dict[str, list[str]],
    ref_sections: dict[str, list[str]],
    *,
    match_threshold: float,
    max_items: int = 8,
) -> list[dict[str, Any]]:
    refs = [line for rows in ref_sections.values() for line in rows]
    unsupported: list[dict[str, Any]] = []
    for section, rows in app_sections.items():
        for line in rows:
            best_score = 0.0
            best_ref = ""
            for ref in refs:
                score, _payload, _reason = _score_candidate_pair(
                    ref,
                    line,
                    mode="lexical",
                    component_weights={"lexical": 1.0, "keyword": 0.0, "embedding": 0.0},
                )
                if score > best_score:
                    best_score = score
                    best_ref = ref
            if best_score < match_threshold:
                unsupported.append(
                    {
                        "section": section,
                        "statement": _normalize_text(line),
                        "best_reference_score": round(best_score, 3),
                        "best_reference": _normalize_text(best_ref),
                    }
                )
    unsupported.sort(key=lambda item: float(item.get("best_reference_score", 0.0) or 0.0))
    return unsupported[:max_items]


def _match_gold_claims(
    gold_claims: list[dict[str, Any]],
    app_sections: dict[str, list[str]],
    *,
    match_threshold: float,
) -> dict[str, Any]:
    matched: list[str] = []
    missing: list[dict[str, Any]] = []
    for claim in gold_claims:
        claim_id = str(claim.get("claim_id") or "")
        section = _canonical(claim.get("section"))
        reference = _normalize_text(claim.get("evidence_quote"))
        if not reference:
            continue
        same_section_rows = app_sections.get(section, [])
        any_section_rows = [line for rows in app_sections.values() for line in rows]
        best_same = 0.0
        best_any = 0.0
        best_any_section = ""
        for line in same_section_rows:
            score, _payload, _reason = _score_candidate_pair(
                reference,
                line,
                mode="lexical",
                component_weights={"lexical": 1.0, "keyword": 0.0, "embedding": 0.0},
            )
            best_same = max(best_same, score)
        for candidate_section, rows in app_sections.items():
            for line in rows:
                score, _payload, _reason = _score_candidate_pair(
                    reference,
                    line,
                    mode="lexical",
                    component_weights={"lexical": 1.0, "keyword": 0.0, "embedding": 0.0},
                )
                if score > best_any:
                    best_any = score
                    best_any_section = candidate_section
        if best_same >= match_threshold:
            matched.append(claim_id)
        else:
            missing.append(
                {
                    "claim_id": claim_id,
                    "section": section,
                    "priority": claim.get("priority"),
                    "importance": claim.get("importance"),
                    "best_same_section_score": round(best_same, 3),
                    "best_any_section_score": round(best_any, 3),
                    "best_any_section": best_any_section,
                    "evidence_quote": reference,
                }
            )
    total = len(gold_claims)
    return {
        "gold_claims_total": total,
        "gold_claims_matched": len(matched),
        "gold_claim_recall": round((len(matched) / total) if total else 0.0, 3),
        "matched_claim_ids": matched[:50],
        "missing_claims": missing[:10],
    }


def _build_evidence_gold_compatibility(summary_json: dict[str, Any], gold_standard_path: Path | None) -> dict[str, Any]:
    if gold_standard_path is None:
        return {"available": False, "compatible": False, "reason": "no gold-standard JSON provided"}
    if not gold_standard_path.exists():
        return {"available": False, "compatible": False, "reason": f"gold standard not found: {gold_standard_path}"}
    try:
        gold_payload = load_gold_standard(gold_standard_path)
        packets = evidence_packets_from_payload(summary_json)
        metadata = evidence_metadata_from_payload(summary_json)
        return {"available": True, **compare_evidence_to_gold(packets, gold_payload, evidence_metadata=metadata)}
    except Exception as exc:
        return {"available": False, "compatible": False, "reason": str(exc)}


def _retention_stage_rate(retention_summary: dict[str, Any], stage: str) -> float | None:
    stage_metrics = retention_summary.get("stage_metrics", [])
    if not isinstance(stage_metrics, list):
        return None
    for item in stage_metrics:
        if isinstance(item, dict) and item.get("stage") == stage:
            try:
                return float(item.get("retained_rate"))
            except Exception:
                return None
    return None


def _build_failure_mode_metrics(
    *,
    comparison: dict[str, Any],
    app_sections: dict[str, list[str]],
    ref_sections: dict[str, list[str]],
    information_retention_summary: dict[str, Any],
    gold_claims: list[dict[str, Any]],
    match_threshold: float,
) -> dict[str, Any]:
    section_payload = comparison.get("sections", {})
    if not isinstance(section_payload, dict):
        section_payload = {}
    per_section = {
        section: {
            "claim_recall": float((section_payload.get(section, {}) or {}).get("recall", 0.0) or 0.0),
            "section_fidelity": float((section_payload.get(section, {}) or {}).get("section_fidelity", 0.0) or 0.0),
            "inclusion_precision": float((section_payload.get(section, {}) or {}).get("inclusion_precision", 0.0) or 0.0),
        }
        for section in SECTION_KEYS
    }

    gold_metrics = _match_gold_claims(gold_claims, app_sections, match_threshold=match_threshold) if gold_claims else {}
    claim_recall = (
        float(gold_metrics.get("gold_claim_recall", 0.0))
        if gold_metrics
        else float(comparison.get("overall_recall", 0.0) or 0.0)
    )

    unsupported_top = _top_unmatched_app_lines(
        app_sections,
        ref_sections,
        match_threshold=max(0.1, match_threshold - 0.12),
    )
    app_line_count = sum(len(rows) for rows in app_sections.values())
    unsupported_rate = round((len(unsupported_top) / app_line_count) if app_line_count else 0.0, 3)

    expected_numbers = _number_tokens_from_gold_claims(gold_claims)
    if not expected_numbers:
        expected_numbers = _number_tokens_from_lines([line for rows in ref_sections.values() for line in rows])
    observed_numbers = _number_tokens_from_lines([line for rows in app_sections.values() for line in rows])
    matched_numbers = expected_numbers & observed_numbers
    numeric_fidelity = round((len(matched_numbers) / len(expected_numbers)) if expected_numbers else 1.0, 3)

    expected_refs = _collect_modality_refs(ref_sections)
    observed_refs = _collect_modality_refs(app_sections)
    modality_recall = {
        key: _recall_payload(expected_refs[key], observed_refs[key])
        for key in ("figure", "table", "supplement")
    }
    table_figure_supplement_recall = round(
        (
            modality_recall["figure"]["recall"]
            + modality_recall["table"]["recall"]
            + modality_recall["supplement"]["recall"]
        )
        / 3.0,
        3,
    )

    synthesis_retention = _retention_stage_rate(information_retention_summary, "executive_report")
    if synthesis_retention is None:
        synthesis_retention = _retention_stage_rate(information_retention_summary, "sections")

    top_missing_claims = list(gold_metrics.get("missing_claims", [])) if gold_metrics else []
    if not top_missing_claims:
        for section in SECTION_KEYS:
            payload = section_payload.get(section, {})
            if not isinstance(payload, dict):
                continue
            for item in payload.get("missing_top", [])[:3]:
                if isinstance(item, dict):
                    top_missing_claims.append({"section": section, **item})
            if len(top_missing_claims) >= 10:
                break

    cross_section_top: list[dict[str, Any]] = []
    for section in SECTION_KEYS:
        payload = section_payload.get(section, {})
        if isinstance(payload, dict):
            for item in payload.get("cross_section_top", [])[:3]:
                if isinstance(item, dict):
                    cross_section_top.append({"section": section, **item})

    aggregate = float(comparison.get("overall_recall", 0.0) or 0.0)
    return {
        "aggregate_score": round(aggregate, 3),
        "primary_dimensions": {
            "parser_recall": float(comparison.get("overall_sentence_inclusion_any_section_recall", 0.0) or 0.0),
            "section_assignment": float(comparison.get("overall_section_fidelity", 0.0) or 0.0),
            "claim_recall": round(claim_recall, 3),
            "claim_precision": float(comparison.get("overall_inclusion_precision", 0.0) or 0.0),
            "numeric_fidelity": numeric_fidelity,
            "unsupported_claim_rate": unsupported_rate,
            "table_figure_supplement_recall": table_figure_supplement_recall,
            "synthesis_retention": round(float(synthesis_retention), 3) if synthesis_retention is not None else None,
        },
        "per_section": per_section,
        "gold_claims": gold_metrics,
        "numeric_fidelity": {
            "expected_numbers": len(expected_numbers),
            "matched_numbers": len(matched_numbers),
            "missing_numbers": sorted(expected_numbers - observed_numbers)[:20],
            "score": numeric_fidelity,
        },
        "modality_recall": modality_recall,
        "top_missing_claims": top_missing_claims[:10],
        "top_unsupported_claims": unsupported_top,
        "top_cross_section_misassignments": cross_section_top[:10],
    }


def _build_extraction_audit(
    app_sections: dict[str, list[str]],
    ref_sections: dict[str, list[str]],
    comparison: dict[str, Any],
    result: dict[str, Any],
    run_note: str,
) -> dict[str, Any]:
    diagnostics = comparison.get("discrepancy_diagnostics", {})
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    section_diag = diagnostics.get("section_diagnostics", {})
    if not isinstance(section_diag, dict):
        section_diag = {}

    section_audit: dict[str, Any] = {}
    empty_relevant_sections: list[str] = []
    low_sentence_recall_sections: list[str] = []
    cross_section_loss_sections: list[str] = []
    high_gap_sections: list[str] = []

    for section in SECTION_KEYS:
        ref_count = len(_dedupe_lines(ref_sections.get(section, []), max_items=400))
        app_count = len(_dedupe_lines(app_sections.get(section, []), max_items=400))
        payload = section_diag.get(section, {})
        if not isinstance(payload, dict):
            payload = {}
        sentence_recall = float(payload.get("sentence_inclusion_recall", 0.0) or 0.0)
        any_section_recall = float(payload.get("sentence_inclusion_any_section_recall", 0.0) or 0.0)
        section_fidelity = float(payload.get("section_fidelity", 0.0) or 0.0)
        coverage_gap = max(0, ref_count - app_count)

        if ref_count > 0 and app_count == 0:
            empty_relevant_sections.append(section)
        if ref_count > 0 and sentence_recall < 0.2:
            low_sentence_recall_sections.append(section)
        if ref_count > 0 and any_section_recall - sentence_recall >= 0.2:
            cross_section_loss_sections.append(section)
        if ref_count > 0 and coverage_gap >= max(3, int(ref_count * 0.5)):
            high_gap_sections.append(section)

        section_audit[section] = {
            "reference_points": ref_count,
            "app_points": app_count,
            "coverage_gap": coverage_gap,
            "sentence_inclusion_recall": round(sentence_recall, 3),
            "sentence_inclusion_any_section_recall": round(any_section_recall, 3),
            "section_fidelity": round(section_fidelity, 3),
            "needs_review": section in empty_relevant_sections
            or section in low_sentence_recall_sections
            or section in cross_section_loss_sections
            or section in high_gap_sections,
        }

    fallback_audit = _build_fallback_audit(result, run_note)
    quality_backend_audit = _build_quality_backend_audit(result)
    quality_blockers = (
        quality_backend_audit.get("blockers", {})
        if isinstance(quality_backend_audit.get("blockers"), dict)
        else {}
    )
    blockers = {
        "fallback_present": not bool(fallback_audit.get("passed")),
        "text_calls_zero": bool(quality_blockers.get("text_calls_zero")),
        "section_extraction_disabled": bool(quality_blockers.get("section_extraction_disabled")),
        "narrative_synthesis_calls_zero": bool(quality_blockers.get("narrative_synthesis_calls_zero")),
        "empty_relevant_sections": empty_relevant_sections,
        "low_sentence_recall_sections": low_sentence_recall_sections,
        "cross_section_loss_sections": cross_section_loss_sections,
        "high_gap_sections": high_gap_sections,
    }
    passed = (
        not blockers["fallback_present"]
        and bool(quality_backend_audit.get("passed"))
        and not empty_relevant_sections
        and not low_sentence_recall_sections
        and not cross_section_loss_sections
    )

    return {
        "passed": passed,
        "purpose": "Track no-fallback extraction completeness and section filtering quality for model comparison runs.",
        "fallback_audit": fallback_audit,
        "quality_backend_audit": quality_backend_audit,
        "section_audit": section_audit,
        "blockers": blockers,
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
        cause_details = diagnostics.get("cause_details", {})
        if isinstance(cause_details, dict):
            cross_section_loss = cause_details.get("cross_section_loss_sections", [])
            under_selected = cause_details.get("under_selected_sections", [])
            missing_any = cause_details.get("missing_any_section_sections", [])
            if cross_section_loss:
                lines.append(f"- cross_section_loss_sections: {', '.join(str(v) for v in cross_section_loss)}")
            if under_selected:
                lines.append(f"- under_selected_sections: {', '.join(str(v) for v in under_selected)}")
            if missing_any:
                lines.append(f"- missing_any_section_sections: {', '.join(str(v) for v in missing_any)}")
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
    failure_mode_metrics = comparison.get("failure_mode_metrics", {})
    if isinstance(failure_mode_metrics, dict) and failure_mode_metrics:
        lines.append("## Failure Mode Metrics")
        dimensions = failure_mode_metrics.get("primary_dimensions", {})
        if isinstance(dimensions, dict):
            for key in (
                "parser_recall",
                "section_assignment",
                "claim_recall",
                "claim_precision",
                "numeric_fidelity",
                "unsupported_claim_rate",
                "table_figure_supplement_recall",
                "synthesis_retention",
            ):
                lines.append(f"- {key}: {dimensions.get(key)}")
        lines.append(f"- aggregate_score: {failure_mode_metrics.get('aggregate_score')}")
        missing_claims = failure_mode_metrics.get("top_missing_claims", [])
        if isinstance(missing_claims, list) and missing_claims:
            lines.append("- top_missing_claims:")
            for item in missing_claims[:5]:
                if not isinstance(item, dict):
                    continue
                claim_id = item.get("claim_id")
                section = item.get("section")
                text = item.get("evidence_quote") or item.get("reference")
                prefix = f"{claim_id}: " if claim_id else ""
                lines.append(f"  - {section}: {prefix}{_normalize_text(text)}")
        unsupported = failure_mode_metrics.get("top_unsupported_claims", [])
        if isinstance(unsupported, list) and unsupported:
            lines.append("- top_unsupported_claims:")
            for item in unsupported[:5]:
                if isinstance(item, dict):
                    lines.append(f"  - {item.get('section')}: {_normalize_text(item.get('statement'))}")
        cross_section = failure_mode_metrics.get("top_cross_section_misassignments", [])
        if isinstance(cross_section, list) and cross_section:
            lines.append("- top_cross_section_misassignments:")
            for item in cross_section[:5]:
                if isinstance(item, dict):
                    lines.append(
                        f"  - {item.get('section')} -> {item.get('best_any_section')}: "
                        f"{_normalize_text(item.get('reference'))}"
                    )
        lines.append("")
    extraction_audit = comparison.get("extraction_audit", {})
    if isinstance(extraction_audit, dict):
        fallback_audit = extraction_audit.get("fallback_audit", {})
        quality_backend_audit = extraction_audit.get("quality_backend_audit", {})
        blockers = extraction_audit.get("blockers", {})
        lines.append("## Extraction Audit")
        lines.append(f"- passed: {extraction_audit.get('passed')}")
        if isinstance(fallback_audit, dict):
            lines.append(f"- no_fallback_passed: {fallback_audit.get('passed')}")
            lines.append(f"- execution_fallback_used: {fallback_audit.get('execution_fallback_used')}")
            lines.append(f"- sections_fallback_used: {fallback_audit.get('sections_fallback_used')}")
            fallback_sections = fallback_audit.get("fallback_sections", [])
            if fallback_sections:
                lines.append(f"- fallback_sections: {', '.join(str(v) for v in fallback_sections)}")
        if isinstance(quality_backend_audit, dict):
            lines.append(f"- quality_backend_passed: {quality_backend_audit.get('passed')}")
            lines.append(f"- text_calls: {quality_backend_audit.get('text_calls')}")
            lines.append(f"- section_extraction_enabled: {quality_backend_audit.get('section_extraction_enabled')}")
        if isinstance(blockers, dict):
            for key in ("text_calls_zero", "section_extraction_disabled"):
                if blockers.get(key):
                    lines.append(f"- {key}: true")
            for key in (
                "empty_relevant_sections",
                "low_sentence_recall_sections",
                "cross_section_loss_sections",
                "high_gap_sections",
            ):
                values = blockers.get(key, [])
                if values:
                    lines.append(f"- {key}: {', '.join(str(v) for v in values)}")
        lines.append("")
    benchmark_validity = comparison.get("benchmark_validity", {})
    if isinstance(benchmark_validity, dict):
        lines.append("## Benchmark Validity")
        lines.append(f"- valid: {benchmark_validity.get('valid')}")
        failure_type = benchmark_validity.get("failure_type")
        if failure_type:
            lines.append(f"- failure_type: {failure_type}")
        reasons = benchmark_validity.get("reasons", [])
        if isinstance(reasons, list) and reasons:
            lines.append(f"- reasons: {', '.join(str(reason) for reason in reasons)}")
        lines.append("")
    evidence_gold = comparison.get("evidence_gold_compatibility", {})
    if isinstance(evidence_gold, dict) and evidence_gold:
        lines.append("## Evidence/Gold Compatibility")
        lines.append(f"- available: {evidence_gold.get('available')}")
        lines.append(f"- compatible: {evidence_gold.get('compatible')}")
        if evidence_gold.get("reason"):
            lines.append(f"- reason: {evidence_gold.get('reason')}")
        failure_reasons = evidence_gold.get("failure_reasons", [])
        if isinstance(failure_reasons, list) and failure_reasons:
            lines.append(f"- failure_reasons: {', '.join(str(reason) for reason in failure_reasons)}")
        thresholds = evidence_gold.get("thresholds", {})
        if isinstance(thresholds, dict) and thresholds:
            threshold_parts = [
                f"{key}={value}"
                for key, value in thresholds.items()
            ]
            lines.append(f"- thresholds: {', '.join(threshold_parts)}")
        for key in (
            "usable_packet_rate",
            "section_coverage_rate",
            "critical_claim_candidate_rate",
            "expected_entity_observability_rate",
            "expected_number_observability_rate",
            "expected_detail_type_observability_rate",
        ):
            if key in evidence_gold:
                lines.append(f"- {key}: {evidence_gold.get(key)}")
        gaps = evidence_gold.get("schema_gaps", [])
        if isinstance(gaps, list) and gaps:
            lines.append(f"- schema_gaps: {', '.join(str(gap) for gap in gaps)}")
        requirement_gaps = evidence_gold.get("claim_requirement_gaps", [])
        if isinstance(requirement_gaps, list) and requirement_gaps:
            lines.append("- claim_requirement_gaps:")
            for gap in requirement_gaps[:5]:
                if not isinstance(gap, dict):
                    continue
                parts = [
                    f"claim_id={gap.get('claim_id')}",
                    f"section={gap.get('section')}",
                ]
                missing_entities = gap.get("missing_entities", [])
                if isinstance(missing_entities, list) and missing_entities:
                    parts.append(f"missing_entities={', '.join(str(item) for item in missing_entities[:5])}")
                missing_numbers = gap.get("missing_numbers", [])
                if isinstance(missing_numbers, list) and missing_numbers:
                    parts.append(f"missing_numbers={', '.join(str(item) for item in missing_numbers[:5])}")
                missing_detail_types = gap.get("missing_detail_types", [])
                if isinstance(missing_detail_types, list) and missing_detail_types:
                    parts.append(f"missing_detail_types={', '.join(str(item) for item in missing_detail_types[:5])}")
                lines.append(f"  - {'; '.join(parts)}")
        synthesis_diagnostics = evidence_gold.get("synthesis_evidence_diagnostics", {})
        if isinstance(synthesis_diagnostics, dict):
            packet_coverage = synthesis_diagnostics.get("evidence_packet_coverage", {})
            if isinstance(packet_coverage, dict) and packet_coverage.get("available"):
                lines.append("- evidence_packet_coverage:")
                for key in ("packet_total", "usable_packets", "usable_packet_rate", "cross_modal_packet_count", "typed_packet_count"):
                    if key in packet_coverage:
                        lines.append(f"  - {key}: {packet_coverage.get(key)}")
                for key in ("sections_present", "missing_core_sections", "quality_flags"):
                    values = packet_coverage.get(key, [])
                    if isinstance(values, list) and values:
                        lines.append(f"  - {key}: {', '.join(str(item) for item in values[:8])}")
                for key in ("by_section", "by_modality", "by_detail_type"):
                    counts = packet_coverage.get(key, {})
                    if isinstance(counts, dict) and counts:
                        count_parts = [f"{name}={count}" for name, count in list(counts.items())[:10]]
                        lines.append(f"  - {key}: {', '.join(count_parts)}")
        lines.append("")
    retention_summary = comparison.get("information_retention_summary", {})
    if isinstance(retention_summary, dict) and retention_summary:
        lines.append("## Information Retention Audit")
        lines.append(f"- source_basis: {retention_summary.get('source_basis')}")
        warning = str(retention_summary.get("source_basis_warning") or "").strip()
        if warning:
            lines.append(f"- source_basis_warning: {warning}")
        lines.append(f"- source_sentence_count: {retention_summary.get('source_sentence_count')}")
        first_loss = retention_summary.get("first_loss_counts", {})
        if isinstance(first_loss, dict):
            loss_parts = [
                f"{stage}={first_loss.get(stage, 0)}"
                for stage in AUDIT_STAGES
                if int(first_loss.get(stage, 0) or 0) > 0
            ]
            if int(first_loss.get("retained_all_stages", 0) or 0) > 0:
                loss_parts.append(f"retained_all_stages={first_loss.get('retained_all_stages', 0)}")
            if loss_parts:
                lines.append(f"- first_loss_counts: {', '.join(loss_parts)}")
        stage_metrics = retention_summary.get("stage_metrics", [])
        if isinstance(stage_metrics, list):
            lines.append("- stage_retention:")
            for item in stage_metrics:
                if not isinstance(item, dict):
                    continue
                lines.append(
                    "  - "
                    f"{item.get('stage')}: retained={item.get('retained_rate')} "
                    f"lost_here={item.get('lost_here_count')} "
                    f"wrong_section={item.get('wrong_section_count')}"
                )
        worst_sections = retention_summary.get("worst_sections_by_cumulative_loss", [])
        if isinstance(worst_sections, list) and worst_sections:
            lines.append("- worst_sections_by_cumulative_loss:")
            for item in worst_sections[:5]:
                if not isinstance(item, dict):
                    continue
                lines.append(
                    "  - "
                    f"{item.get('section')}: lost={item.get('final_cumulative_lost_count')} "
                    f"final_retained_rate={item.get('final_retained_rate')}"
                )
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
        cause_counts = section_payload.get("cause_counts", {})
        if isinstance(cause_counts, dict):
            lines.append(
                "- cause_counts: "
                f"same_section={cause_counts.get('same_section_match', 0)}, "
                f"cross_section_only={cause_counts.get('cross_section_only', 0)}, "
                f"missing_any_section={cause_counts.get('missing_any_section', 0)}"
            )
        cross_section_top = section_payload.get("cross_section_top", [])
        if cross_section_top:
            lines.append("- cross_section_top:")
            for item in cross_section_top[:3]:
                lines.append(
                    f"  - ({item.get('best_any_section_score')}, {item.get('best_any_section')}) "
                    f"{item.get('reference')}"
                )
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


def _load_parsed_chunks_for_document(document_id: int) -> list[dict[str, Any]]:
    if int(document_id or 0) <= 0:
        return []
    try:
        from sqlmodel import Session, select

        from app.db.models import Asset, Chunk
        from app.db.session import engine
    except Exception:
        return []
    try:
        with Session(engine) as session:
            assets = session.exec(select(Asset).where(Asset.document_id == int(document_id))).all()
            asset_kind = {asset.id: asset.kind for asset in assets}
            chunks = session.exec(select(Chunk).where(Chunk.document_id == int(document_id))).all()
            return [
                {
                    "anchor": str(chunk.anchor or ""),
                    "content": str(chunk.content or ""),
                    "meta": str(chunk.meta or ""),
                    "modality": str(chunk.modality or "text"),
                    "asset_kind": asset_kind.get(chunk.asset_id or -1, "main"),
                }
                for chunk in chunks
            ]
    except Exception:
        return []


def _parsed_chunks_from_pdf_rows(pdf_path: Path) -> list[dict[str, Any]]:
    try:
        rows = _extract_pdf_text_rows_local(pdf_path)
    except Exception:
        return []
    chunks: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        meta = {
            "section_norm": row.get("section_norm", "unknown"),
            "section_raw_title": row.get("section_raw_title", ""),
            "page_index": row.get("page_index"),
            "paragraph_index": row.get("paragraph_index"),
            "source": "comparator_pdf_text",
        }
        chunks.append(
            {
                "anchor": str(row.get("anchor") or f"page:p:{idx}"),
                "content": str(row.get("text") or ""),
                "meta": json.dumps(meta),
                "modality": str(row.get("modality") or "text"),
                "asset_kind": "main",
            }
        )
    return chunks


def _build_comparator_information_retention_audit(
    *,
    pdf_path: Path,
    result: dict[str, Any],
    summary_json: dict[str, Any],
) -> dict[str, Any]:
    document_id = int(result.get("document_id", 0) or 0)
    parsed_chunks = _load_parsed_chunks_for_document(document_id)
    if not parsed_chunks:
        parsed_chunks = _parsed_chunks_from_pdf_rows(pdf_path)
    return build_information_retention_audit(
        document_id=document_id,
        source_assets=[
            {
                "kind": "main",
                "filename": pdf_path.name,
                "content_type": "application/pdf",
                "path": str(pdf_path),
            }
        ],
        parsed_chunks=parsed_chunks,
        summary_json=summary_json,
    )


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
                ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
                handle.write(f"{ts} {step}\n")
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
        os.environ["ANALYSIS_NONTEXT_LLM_ENABLED"] = "false"
        os.environ["ANALYSIS_FORCE_NONTEXT_LLM_FOR_OPENAI"] = "false"
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
        os.environ["ANALYSIS_NONTEXT_LLM_ENABLED"] = "false"
        os.environ["ANALYSIS_FORCE_NONTEXT_LLM_FOR_OPENAI"] = "false"
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
        os.environ["ANALYSIS_NONTEXT_LLM_ENABLED"] = "false"
        os.environ["ANALYSIS_FORCE_NONTEXT_LLM_FOR_OPENAI"] = "false"
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
        os.environ["ANALYSIS_NONTEXT_LLM_ENABLED"] = "true"
        os.environ["ANALYSIS_FORCE_NONTEXT_LLM_FOR_OPENAI"] = "true"
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
        os.environ["ANALYSIS_NONTEXT_LLM_ENABLED"] = "true"
        os.environ["ANALYSIS_FORCE_NONTEXT_LLM_FOR_OPENAI"] = "true"
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
    families = ("app_extraction", "comparison", "run", "model_benchmark", "information_retention")
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
    parser.add_argument(
        "--gold-claims-jsonl",
        default="",
        help="Optional structured gold-claims JSONL file for claim and numeric fidelity scoring.",
    )
    parser.add_argument(
        "--gold-standard-json",
        default="",
        help="Optional final-report gold-standard JSON for evidence-packet compatibility scoring.",
    )
    parser.add_argument(
        "--allow-fallback-probes",
        action="store_true",
        default=os.getenv("ALLOW_FALLBACK_PROBES", "").strip().lower() in {"1", "true", "yes", "on"},
        help="Allow parser/analysis probe fallback after a pipeline failure. Disabled by default for report-quality benchmarks.",
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
            if not args.allow_fallback_probes:
                raise SystemExit(
                    _pipeline_failure_note(
                        "Pipeline failed and fallback probes are disabled",
                        failure_diag or {"reason": "unknown isolated pipeline failure"},
                    )
                )
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
    benchmark_validity = _build_benchmark_validity_gate(result, run_note)
    if not benchmark_validity["valid"]:
        raise SystemExit(_format_benchmark_validity_failure(benchmark_validity))

    app_sections = _extract_app_sections(summary_json)
    ref_sections = _parse_reference_markdown(ref_path)
    comparison = _compare_sections(
        app_sections,
        ref_sections,
        match_threshold=float(args.matching_threshold),
        matching_mode=_normalize_matching_mode(str(args.matching_mode)),
    )
    comparison["discrepancy_diagnostics"] = _build_discrepancy_diagnostics(app_sections, ref_sections, comparison)
    comparison["extraction_audit"] = _build_extraction_audit(
        app_sections,
        ref_sections,
        comparison,
        result,
        run_note,
    )
    comparison["benchmark_validity"] = benchmark_validity
    information_retention_audit = _build_comparator_information_retention_audit(
        pdf_path=pdf_path,
        result=result,
        summary_json=summary_json,
    )
    information_retention_summary = information_retention_audit.get("compact_summary", {})
    comparison["information_retention_summary"] = (
        information_retention_summary if isinstance(information_retention_summary, dict) else {}
    )
    gold_claims_path = Path(args.gold_claims_jsonl).expanduser().resolve() if args.gold_claims_jsonl else None
    if gold_claims_path is None:
        gold_claims_path = _default_gold_claims_path(ref_path)
    gold_claims = _load_gold_claims_jsonl(gold_claims_path)
    comparison["failure_mode_metrics"] = _build_failure_mode_metrics(
        comparison=comparison,
        app_sections=app_sections,
        ref_sections=ref_sections,
        information_retention_summary=comparison["information_retention_summary"],
        gold_claims=gold_claims,
        match_threshold=float(args.matching_threshold),
    )
    if gold_claims_path is not None:
        comparison["failure_mode_metrics"]["gold_claims_path"] = str(gold_claims_path)
    gold_standard_path = Path(args.gold_standard_json).expanduser().resolve() if args.gold_standard_json else None
    comparison["evidence_gold_compatibility"] = _build_evidence_gold_compatibility(summary_json, gold_standard_path)
    if gold_standard_path is not None:
        comparison["evidence_gold_compatibility"]["gold_standard_path"] = str(gold_standard_path)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    app_md_path = out_dir / f"{pdf_path.stem}_app_extraction_{stamp}.md"
    cmp_md_path = out_dir / f"{pdf_path.stem}_comparison_{stamp}.md"
    cmp_json_path = out_dir / f"{pdf_path.stem}_comparison_{stamp}.json"
    run_json_path = out_dir / f"{pdf_path.stem}_run_{stamp}.json"
    info_json_path = out_dir / f"{pdf_path.stem}_information_retention_{stamp}.json"

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
    info_json_path.write_text(json.dumps(information_retention_audit, indent=2), encoding="utf-8")
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
                "information_retention_json": str(info_json_path),
                "summary_json": summary_json,
                "analysis_diagnostics": result.get("diagnostics", {}),
                "benchmark_validity": comparison.get("benchmark_validity", {}),
                "extraction_audit": comparison.get("extraction_audit", {}),
                "information_retention_audit": comparison.get("information_retention_summary", {}),
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
    print(f"information_retention_json={info_json_path}")
    print(f"run_json={run_json_path}")
    print(f"artifacts_pruned={pruned_files}")


if __name__ == "__main__":
    main()
