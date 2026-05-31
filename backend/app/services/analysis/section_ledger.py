from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from sqlmodel import Session, select

from app.db.models import Chunk


SECTION_KEYS = {"introduction", "methods", "results", "discussion", "conclusion", "unknown"}
STRUCTURED_PREFIX_RE = re.compile(
    r"^\s*(objective|objectives|background|aim|aims|purpose|hypothesis|method|methods|design|results|conclusion|conclusions)\s*:",
    re.IGNORECASE,
)


def apply_section_boundary_ledger_to_session(session: Session, document_id: int) -> int:
    chunks = session.exec(select(Chunk).where(Chunk.document_id == document_id).order_by(Chunk.id)).all()
    changed = apply_section_boundary_ledger_to_chunks(chunks)
    if changed:
        session.commit()
    return changed


def apply_section_boundary_ledger_to_chunks(chunks: list[Chunk]) -> int:
    rows = [
        {
            "index": idx,
            "chunk": chunk,
            "anchor": chunk.anchor,
            "content": chunk.content,
            "meta": chunk.meta,
            "modality": chunk.modality,
        }
        for idx, chunk in enumerate(chunks)
    ]
    relabeled = apply_section_boundary_ledger_to_dicts(rows)
    changed = 0
    for row in relabeled:
        chunk = row.get("chunk")
        if not isinstance(chunk, Chunk):
            continue
        meta = row.get("meta")
        if isinstance(meta, dict):
            next_meta = json.dumps(meta, ensure_ascii=True)
        else:
            next_meta = str(meta or "")
        if next_meta and next_meta != (chunk.meta or ""):
            chunk.meta = next_meta
            changed += 1
    return changed


def apply_section_boundary_ledger_to_dicts(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = [dict(row) for row in chunks]
    text_indices = [idx for idx, row in enumerate(out) if str(row.get("modality") or "").lower() == "text"]
    blocks = _ledger_blocks(out, text_indices)
    ledger = _build_section_boundary_ledger(blocks)
    for entry in ledger:
        for idx in entry["indices"]:
            row = out[idx]
            meta = _parse_meta(row.get("meta"))
            original_label = _normalize_section_label(meta.get("section_norm") or meta.get("section_label") or meta.get("section"))
            meta.setdefault("section_original_norm", original_label)
            meta.setdefault("section_original_source", str(meta.get("section_source") or "parser"))
            meta.setdefault("section_original_confidence", meta.get("section_confidence"))
            meta["section_norm"] = entry["section"]
            meta["section_source"] = f"section_boundary_ledger:{entry['reason']}"
            meta["section_confidence"] = entry["confidence"]
            meta["section_ledger_title"] = entry["title"]
            meta["section_ledger_baseline"] = entry["baseline_section"]
            meta["section_ledger_version"] = 1
            row["meta"] = meta
    return out


def _ledger_blocks(chunks: list[dict[str, Any]], text_indices: list[int]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for order, idx in enumerate(text_indices):
        row = chunks[idx]
        meta = _parse_meta(row.get("meta"))
        raw_title = _clean_title(meta.get("section_raw_title") or meta.get("section") or "")
        if not raw_title:
            raw_title = _leading_heading_text(str(row.get("content") or "")) or "body"
        if current is None or raw_title != current["title"]:
            current = {
                "title": raw_title,
                "indices": [],
                "start_order": order,
                "texts": [],
                "baseline_sections": [],
            }
            blocks.append(current)
        current["indices"].append(idx)
        current["texts"].append(str(row.get("content") or ""))
        current["baseline_sections"].append(
            _normalize_section_label(meta.get("section_norm") or meta.get("section_label") or meta.get("section"))
        )
    return blocks


def _build_section_boundary_ledger(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ledger: list[dict[str, Any]] = []
    active = "unknown"
    seen_methods = False
    seen_results = False
    seen_discussion = False
    for block_idx, block in enumerate(blocks):
        title = str(block.get("title") or "")
        text = " ".join(block.get("texts", []))
        title_section, title_reason, title_confidence = _classify_ledger_title(title)
        content_section, content_reason, content_confidence = _classify_ledger_content(text)
        baseline_section, baseline_confidence = _majority_section(block.get("baseline_sections", []))

        if _is_abstract_title(title):
            chosen = "introduction"
            reason = "abstract_default"
            confidence = 0.82
        elif title_section != "unknown":
            chosen = title_section
            reason = title_reason
            confidence = title_confidence
        elif content_section != "unknown" and content_confidence >= 0.78:
            chosen = content_section
            reason = content_reason
            confidence = content_confidence
        elif active == "unknown":
            chosen = _pre_boundary_default(block_idx, blocks, baseline_section)
            reason = "pre_boundary_default"
            confidence = 0.56
        else:
            chosen = active
            reason = "carry_forward"
            confidence = 0.58

        if chosen == "results" and not seen_methods and content_section != "results":
            chosen = "introduction"
            reason = "early_results_title_without_body_result_signal"
            confidence = 0.62
        if chosen == "introduction" and seen_methods and active not in {"unknown", "introduction"}:
            chosen = active
            reason = "blocked_intro_after_methods"
            confidence = 0.46
        if (
            chosen == "methods"
            and seen_results
            and active in {"results", "discussion", "conclusion"}
            and title_reason != "title_direct"
        ):
            chosen = active
            reason = "blocked_methods_after_results"
            confidence = 0.46
        if chosen == "results" and seen_discussion and active in {"discussion", "conclusion"}:
            chosen = active
            reason = "blocked_results_after_discussion"
            confidence = 0.46

        if chosen != "unknown":
            active = chosen
        if active == "methods":
            seen_methods = True
        elif active == "results":
            seen_results = True
        elif active == "discussion":
            seen_discussion = True

        ledger.append(
            {
                "title": title,
                "indices": list(block.get("indices", [])),
                "section": active if active != "unknown" else baseline_section,
                "reason": reason,
                "confidence": round(float(confidence), 3),
                "baseline_section": baseline_section,
                "baseline_confidence": baseline_confidence,
            }
        )
    return ledger


def _classify_ledger_title(title: str) -> tuple[str, str, float]:
    text = _clean_title(title)
    lowered = text.lower()
    if not text or lowered in {"body", "file"} or lowered.startswith(("figure", "table")):
        return "unknown", "title_uninformative", 0.0
    direct = _normalize_section_label(text)
    if direct != "unknown":
        return direct, "title_direct", 0.92

    methods_patterns = (
        "participant",
        "sample",
        "cohort",
        "assessment",
        "acquisition",
        "processing",
        "analysis",
        "dataset",
        "data distribution",
        "severity analysis",
        "methods",
        "protocol",
        "optimization",
        "transfection",
        "vector",
        "plasmid",
        "cell line",
        "quantification",
        "lc-ms",
        "pcr",
        "sanger",
        "construct",
        "reagents",
        "specimen",
        "pathogen identification",
        "diagnostic criteria",
        "statistical",
        "statistics",
        "ethical",
        "ethics",
        "institutional review",
        "irb",
        "informed consent",
        "serotype",
        "microbiolog",
        "laborator",
    )
    result_patterns = (
        "result",
        "finding",
        "validation",
        "functional",
        "transport",
        "stability",
        "expression",
        "forecast",
        "prediction",
        "performance",
        "strategy",
        "generation",
        "generated",
        "connectivity",
        "network",
        "nucleus accumbens",
        "default mode",
        "hyperconnectivity",
        "reduced connectivity",
        "structural features",
        "allosteric",
        "oligomerization",
    )
    discussion_patterns = (
        "discussion",
        "limitation",
        "strength",
        "framework",
        "perspective",
        "suggestion",
        "consideration",
        "implication",
        "future",
    )
    if any(token in lowered for token in ("conclusion", "conclusions", "summary")):
        return "conclusion", "title_conclusion_pattern", 0.9
    if any(token in lowered for token in discussion_patterns):
        return "discussion", "title_discussion_pattern", 0.84
    if any(token in lowered for token in methods_patterns):
        return "methods", "title_methods_pattern", 0.82
    if any(token in lowered for token in result_patterns):
        return "results", "title_results_pattern", 0.72
    return "unknown", "title_unknown", 0.0


def _classify_ledger_content(text: str) -> tuple[str, str, float]:
    clean = _clean_text(text).lower()
    prefix = STRUCTURED_PREFIX_RE.match(clean)
    if prefix:
        token = prefix.group(1)
        if token in {"objective", "objectives", "background", "aim", "aims", "purpose", "hypothesis"}:
            return "introduction", "content_structured_intro", 0.9
        if token in {"method", "methods", "design"}:
            return "methods", "content_structured_methods", 0.9
        if token == "results":
            return "results", "content_structured_results", 0.9
        return "conclusion", "content_structured_conclusion", 0.9
    if re.search(r"\b(the rest of the paper is organized|we hypothesized|our goal was|we aimed|objective)\b", clean):
        return "introduction", "content_intro_signal", 0.82
    if re.search(
        r"\b(participants?|sample|recruited|enrolled|administered|acquired|preprocess|covariate|regression|protocol|"
        r"cells? were|plasmid|pcr|qpcr|transfected|selection|assay|lc-ms|specimens?|pathogen|serotype|"
        r"statistical|chi-square|fisher'?s exact|student'?s t|mann-whitney|irb|institutional review|"
        r"informed consent)\b",
        clean,
    ):
        return "methods", "content_methods_signal", 0.78
    if re.search(r"\b(results? showed|we found|identified|increased|decreased|higher|lower|significant|p\s*[<=>]|figure\s+\d+|table\s+\d+)\b", clean):
        return "results", "content_results_signal", 0.76
    if re.search(r"\b(limitation|consistent with|in contrast|suggests?|implications?|future studies|should confirm|to our knowledge)\b", clean):
        return "discussion", "content_discussion_signal", 0.78
    if re.search(r"\b(in conclusion|to sum up|overall,|taken together|we conclude)\b", clean):
        return "conclusion", "content_conclusion_signal", 0.84
    return "unknown", "content_unknown", 0.0


def _majority_section(sections: list[str]) -> tuple[str, float]:
    counts = Counter(section for section in sections if section and section != "unknown")
    if not counts:
        return "unknown", 0.0
    section, count = counts.most_common(1)[0]
    return section, round(count / max(1, len(sections)), 3)


def _pre_boundary_default(block_idx: int, blocks: list[dict[str, Any]], baseline_section: str) -> str:
    if block_idx <= 1:
        return "introduction" if baseline_section == "unknown" else baseline_section
    for future in blocks[block_idx + 1 : block_idx + 5]:
        section, _reason, _confidence = _classify_ledger_title(str(future.get("title") or ""))
        if section == "methods":
            return "introduction"
    return baseline_section if baseline_section != "unknown" else "introduction"


def _parse_meta(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _normalize_section_label(value: Any) -> str:
    text = _clean_text(value).lower()
    if not text:
        return "unknown"
    if "conclusion" in text or "concluding" in text or text == "summary":
        return "conclusion"
    if "discussion" in text or "limitation" in text or "implication" in text:
        return "discussion"
    if "result" in text or "finding" in text or "outcome" in text:
        return "results"
    if any(
        token in text
        for token in (
            "method",
            "material",
            "participant",
            "procedure",
            "protocol",
            "analysis",
            "statistical",
            "statistics",
            "design",
            "specimen",
            "pathogen",
            "ethical",
            "ethics",
            "irb",
            "consent",
        )
    ):
        return "methods"
    if any(token in text for token in ("abstract", "intro", "background", "objective", "aim", "hypoth", "rationale", "purpose")):
        return "introduction"
    return "unknown"


def _clean_title(value: Any) -> str:
    text = _clean_text(value).replace("_", " ")
    text = re.sub(r"^\d+(?:\.\d+)*\s*", "", text)
    text = text.strip(" -:.;)")
    return text or "body"


def _leading_heading_text(text: str) -> str:
    first_line = str(text or "").replace("\r", "\n").split("\n", 1)[0].strip()
    if len(first_line) > 120:
        return ""
    if re.search(r"[.!?]$", first_line) and not STRUCTURED_PREFIX_RE.match(first_line):
        return ""
    return first_line


def _is_abstract_title(title: str) -> bool:
    return _clean_title(title).lower() == "abstract"


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()
