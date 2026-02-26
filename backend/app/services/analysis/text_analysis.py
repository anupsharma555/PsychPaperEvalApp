from __future__ import annotations

import json
import re
from typing import Any

from app.core.config import settings
from app.services.analysis.llm import chat_text_fast
from app.services.analysis.prompts import TEXT_ANALYSIS_SYSTEM
from app.services.analysis.utils import (
    extract_json,
    max_chars_for_ctx,
    normalize_evidence_packets,
    packets_to_legacy_findings,
    truncate_text,
)

INTRO_SECTION_RE = re.compile(r"\b(intro|background|objective|aim|rationale|hypoth)\b", re.IGNORECASE)
METHOD_SECTION_RE = re.compile(
    r"\b(methods?|materials?|participants?|procedure|analysis|acquisition|design|protocol|covariate)\b",
    re.IGNORECASE,
)
RESULT_SECTION_RE = re.compile(
    r"\b(finding|outcome|revealed|identified|demonstrated|showed|significant|increased|decreased|higher|lower)\b",
    re.IGNORECASE,
)
DISCUSSION_SECTION_RE = re.compile(
    r"\b(discussion|interpretation|implication|limitation|to date|in our study|consistent with|may reflect|suggests that)\b",
    re.IGNORECASE,
)
CONCLUSION_SECTION_RE = re.compile(
    r"\b(conclusion|concluding|summary|overall|future research|longitudinal)\b",
    re.IGNORECASE,
)
SECTION_LABELS = {"introduction", "methods", "results", "discussion", "conclusion", "unknown"}
SECTION_SOURCE_WEIGHTS: dict[str, float] = {
    "meta": 1.00,
    "explicit_heading": 1.08,
    "structured_abstract": 0.98,
    "anchor": 0.92,
    "heading": 0.91,
    "heading_style": 0.90,
    "statement_prefix": 0.84,
    "category": 0.72,
    "position": 0.46,
    "lexical": 0.60,
    "fallback": 0.28,
}
SECTION_SOURCE_RANK: dict[str, int] = {
    "meta": 0,
    "explicit_heading": 1,
    "heading": 2,
    "heading_style": 3,
    "structured_abstract": 4,
    "anchor": 5,
    "statement_prefix": 6,
    "category": 7,
    "lexical": 8,
    "position": 9,
    "fallback": 10,
}
SECTION_MARGIN_THRESHOLD = 0.08
SECTION_CONFLICT_OVERRIDE_MARGIN = 0.12
SECTION_POSITION_BOUNDARIES = [
    ("introduction", 0.22),
    ("methods", 0.52),
    ("results", 0.70),
    ("discussion", 0.90),
    ("conclusion", 1.00),
]
SECTION_ORDER = ["introduction", "methods", "results", "discussion", "conclusion", "unknown"]


def _clamp_confidence(value: Any) -> float:
    try:
        score = float(value)
    except Exception:
        return 0.0
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score
STATEMENT_PREFIX_RE = re.compile(
    r"^\s*(objective|objectives|background|aim|aims|method|methods|design|results|conclusion|conclusions)\s*:",
    re.IGNORECASE,
)
EXPLICIT_HEADING_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+){0,3}\s+)?"
    r"(abstract|introduction|background|objective|objectives|aim|aims|method|methods|"
    r"materials?\s+and\s+methods?|results?|discussion|conclusions?|summary|supplementary|"
    r"supplementary\s+materials?|references?)"
    r"[\s\-–—:.;\)]*$",
    re.IGNORECASE,
)


def analyze_text(chunks: list[dict[str, Any]], *, force_llm_enabled: bool | None = None) -> dict[str, Any]:
    max_chars = min(settings.analysis_max_text_chars, max_chars_for_ctx(settings.llm_n_ctx))
    valid_anchors = {str(chunk.get("anchor", "unknown")) for chunk in chunks}
    raw_findings: list[dict[str, Any]] = []
    raw_claims: list[dict[str, Any]] = []
    raw_packets: list[dict[str, Any]] = []
    llm_enabled = settings.analysis_text_llm_enabled if force_llm_enabled is None else bool(force_llm_enabled)

    if llm_enabled:
        batch: list[str] = []
        batch_len = 0

        def flush_batch() -> None:
            nonlocal batch, batch_len, raw_findings, raw_claims, raw_packets
            if not batch:
                return
            prompt = (
                "Analyze the paper text for peer-review quality with strict section fidelity. "
                "Only use anchors that exactly match the bracketed snippet anchors in the input. "
                "Extract section-appropriate points from introduction, methods, results, discussion, and conclusion. "
                "For methods anchors, extract concrete protocol details (design, cohort/sample construction, "
                "inclusion/exclusion criteria, interventions/comparators, endpoints, measurement instruments, "
                "statistical models, covariates, missing-data handling, and sensitivity checks). "
                "For results anchors, extract observed findings only and state the actual outcome direction/magnitude "
                "reported in text (avoid vague phrasing like 'shows this or that'). "
                "Never emit generic result narration like 'the figure shows' without concrete reported outcomes. "
                "For discussion/conclusion anchors, extract interpretation/conclusions only from those sections. "
                "Do not mix section content (for example, do not place methods statements in results packets). "
                "Do not claim details are missing unless the text explicitly says they are missing/not reported. "
                "Cite evidence anchors.\n\n"
                + "\n\n".join(batch)
            )
            prompt = truncate_text(prompt, max_chars)
            response = chat_text_fast(prompt, system=TEXT_ANALYSIS_SYSTEM)
            data = _normalize_llm_payload(extract_json(response))
            raw_findings.extend(data.get("findings", []))
            raw_claims.extend(data.get("claims", []))
            raw_packets.extend(data.get("evidence_packets", []))
            batch = []
            batch_len = 0

        for chunk in chunks:
            text = chunk.get("content", "")
            anchor = chunk.get("anchor", "unknown")
            if not text:
                continue
            snippet = f"[{anchor}] {text}"
            if batch_len + len(snippet) > max_chars and batch:
                flush_batch()
            if len(snippet) > max_chars:
                snippet = snippet[:max_chars]
            batch.append(snippet)
            batch_len += len(snippet)

        flush_batch()

    packet_inputs = list(raw_packets)
    for finding in raw_findings:
        evidence = finding.get("evidence") or []
        anchor = evidence[0] if isinstance(evidence, list) and evidence else ""
        packet_inputs.append(
            {
                "finding_id": finding.get("finding_id"),
                "anchor": anchor,
                "statement": finding.get("summary", ""),
                "evidence_refs": evidence,
                "confidence": finding.get("confidence", 0.0),
                "category": finding.get("category", "other"),
            }
        )
    for claim in raw_claims:
        evidence = claim.get("evidence") or []
        anchor = evidence[0] if isinstance(evidence, list) and evidence else ""
        packet_inputs.append(
            {
                "finding_id": claim.get("finding_id"),
                "anchor": anchor,
                "statement": claim.get("claim", ""),
                "evidence_refs": evidence,
                "confidence": claim.get("confidence", 0.0),
                "category": "claim",
            }
        )

    evidence_packets = normalize_evidence_packets(
        packet_inputs,
        "text",
        valid_anchors,
        default_category="other",
    )
    evidence_packets = _filter_text_packets(evidence_packets)
    analysis_notes = _analysis_notes(chunks)
    if len(chunks) >= 3 and not analysis_notes and _needs_heuristic_fallback(evidence_packets):
        # Deterministic fallback when LLM output is empty/sparse for a full paper.
        fallback_packets = _heuristic_packets_from_chunks(chunks)
        if fallback_packets:
            normalized_fallback = normalize_evidence_packets(
                fallback_packets,
                "text",
                valid_anchors,
                default_category="other",
            )
            normalized_fallback = _filter_text_packets(normalized_fallback)
            evidence_packets = _dedupe_packets(evidence_packets + normalized_fallback)
    evidence_packets = _hydrate_anchor_metadata(evidence_packets, chunks)
    evidence_packets = _annotate_text_packet_sections(evidence_packets)

    findings_packets = [packet for packet in evidence_packets if packet.get("category") != "claim"]
    claim_packets = [packet for packet in evidence_packets if packet.get("category") == "claim"]
    claims = [
        {
            "claim": packet.get("statement", ""),
            "evidence": packet.get("evidence_refs", []),
            "confidence": packet.get("confidence", 0.0),
            "claim_id": packet.get("finding_id"),
        }
        for packet in claim_packets
    ]
    return {
        "findings": packets_to_legacy_findings(findings_packets),
        "claims": claims,
        "evidence_packets": evidence_packets,
        "claim_packets": claim_packets,
        "analysis_notes": analysis_notes,
    }


def _normalize_llm_payload(raw: Any) -> dict[str, list[dict[str, Any]]]:
    if isinstance(raw, list):
        return {
            "evidence_packets": [item for item in raw if isinstance(item, dict)],
            "findings": [],
            "claims": [],
        }
    if not isinstance(raw, dict):
        return {"evidence_packets": [], "findings": [], "claims": []}
    return {
        "evidence_packets": [item for item in raw.get("evidence_packets", []) if isinstance(item, dict)],
        "findings": [item for item in raw.get("findings", []) if isinstance(item, dict)],
        "claims": [item for item in raw.get("claims", []) if isinstance(item, dict)],
    }


ABSENCE_CLAIM_RE = re.compile(
    r"\b("
    r"not described|not reported|not mentioned|not provided|not detailed|"
    r"no mention|no details|absence of|lacking in detail|not explicitly described"
    r")\b",
    re.IGNORECASE,
)
PAYWALL_MARKER_RE = re.compile(
    r"\b("
    r"purchase this article|already a subscriber|access your subscription|full access to this article|"
    r"ppv articles|subscription options|customer service|citation manager"
    r")\b",
    re.IGNORECASE,
)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
METHOD_FALLBACK_RE = re.compile(
    r"\b(participants?|sample|cohort|scanner|acquisition|processing|cwas|mdmr|seed-based|covariate|regression|analysis)\b",
    re.IGNORECASE,
)
RESULT_FALLBACK_RE = re.compile(
    r"\b(found|identified|revealed|associated|increased|decreased|higher|lower|significant|p\s*[<=>]\s*0?\.\d+)\b",
    re.IGNORECASE,
)
DISCUSSION_FALLBACK_RE = re.compile(
    r"\b(suggest|implication|interpret|limitation|limitations|may reflect|consistent with|to date|in our study|in contrast)\b",
    re.IGNORECASE,
)
CONCLUSION_FALLBACK_RE = re.compile(
    r"\b(in conclusion|overall|these findings|these results (?:suggest|support|emphasize)|conclude|conclusions?|future research|longitudinal)\b",
    re.IGNORECASE,
)
INTRO_FALLBACK_RE = re.compile(
    r"\b(background|objective|aim|hypothesis|rationale|motivation|anhedonia|reward)\b",
    re.IGNORECASE,
)
METHOD_STRONG_RE = re.compile(
    r"\b(participants?|sample|cohort|scanner|acquisition|preprocess|processed|protocol|covariates?|regression|mdmr|quality assurance|inclusion|exclusion)\b",
    re.IGNORECASE,
)
RESULT_STRONG_RE = re.compile(
    r"\b(we found|revealed|identified|significant|p\s*[<=>]\s*0?\.\d+|t\s*=\s*-?\d+(?:\.\d+)?|increased|decreased|higher|lower|hyperconnectivity|dysconnectivity)\b",
    re.IGNORECASE,
)
DISCUSSION_STRONG_RE = re.compile(
    r"\b(to date|in our study|in contrast|consistent with|may reflect|we used a fully data-driven|limitations?)\b",
    re.IGNORECASE,
)
CONCLUSION_STRONG_RE = re.compile(
    r"\b(in conclusion|overall|these results (?:suggest|support|emphasize)|future research|longitudinal)\b",
    re.IGNORECASE,
)
STRUCTURED_PREFIX_LINE_RE = re.compile(
    r"^\s*(objective|objectives|background|aim|aims|method|methods|design|results|discussion|conclusion|conclusions)\s*:\s*(.+)$",
    re.IGNORECASE,
)
NOISE_STATEMENT_RE = re.compile(
    r"\b("
    r"ajp\.psychiatryonline\.org|doi:|received [a-z]+ \d{1,2}, \d{4}|revision received|accepted [a-z]+ \d{1,2}, \d{4}|"
    r"published online|address correspondence|presented at|supported by nih|financial relationships|customer service|"
    r"table\s+\d+|figure\s+\d+|^articles$|^sharma et al\.?$|^reward deficits across mood and psychotic disorders$"
    r")\b",
    re.IGNORECASE,
)


def _filter_text_packets(packets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for packet in packets:
        refs = [str(ref).strip() for ref in packet.get("evidence_refs", []) if str(ref).strip()]
        statement = str(packet.get("statement", "")).strip()
        confidence = float(packet.get("confidence", 0.0) or 0.0)
        quality_flags = {str(flag).strip().lower() for flag in packet.get("quality_flags", [])}

        # Drop any text packet without usable evidence refs; these are usually invented anchors.
        if not refs or "missing_evidence" in quality_flags:
            continue
        if _is_noise_statement(statement):
            continue
        # Drop low-confidence omission claims to prevent template spam when methods text is unavailable.
        if confidence < 0.6 and ABSENCE_CLAIM_RE.search(statement):
            continue
        filtered.append(packet)
    return filtered


def _analysis_notes(chunks: list[dict[str, Any]]) -> list[str]:
    texts = [str(chunk.get("content", "")).strip() for chunk in chunks if str(chunk.get("content", "")).strip()]
    if not texts:
        return ["No text chunks were available for methods extraction."]

    marker_hits = 0
    for text in texts:
        if PAYWALL_MARKER_RE.search(text):
            marker_hits += 1
    if marker_hits >= 2:
        return [
            "Source text appears access-limited (publisher landing/subscription content). "
            "Methods extraction may be incomplete; upload the full PDF for reliable methodology analysis."
        ]
    return []


def _heuristic_packets_from_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    section_targets = {
        "introduction": 3,
        "methods": 8,
        "results": 6,
        "discussion": 4,
        "conclusion": 2,
    }
    section_counts = {key: 0 for key in section_targets}
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    for idx, chunk in enumerate(chunks):
        anchor = str(chunk.get("anchor", "")).strip()
        if not anchor:
            continue
        section = _infer_chunk_section(chunk, idx=idx, total_chunks=len(chunks))
        if section == "unknown":
            continue
        if section_counts.get(section, 0) >= section_targets.get(section, 0):
            continue
        content = " ".join(str(chunk.get("content", "")).split()).strip()
        if not content:
            continue
        candidates = _select_fallback_sentences(content, section=section)
        for sentence in candidates:
            canonical = " ".join(sentence.lower().split())
            if not canonical or canonical in seen:
                continue
            seen.add(canonical)
            out.append(
                {
                    "finding_id": f"text-fallback-{section}-{len(out) + 1}",
                    "anchor": anchor,
                    "statement": sentence,
                    "evidence_refs": [anchor],
                    "confidence": 0.68 if section == "introduction" else 0.7,
                    "category": section,
                }
            )
            section_counts[section] = section_counts.get(section, 0) + 1
            if section_counts[section] >= section_targets.get(section, 0):
                break
        if all(section_counts[key] >= section_targets[key] for key in section_targets):
            break
    return out


def _needs_heuristic_fallback(packets: list[dict[str, Any]]) -> bool:
    if not packets:
        return True
    labels: set[str] = set()
    for packet in packets:
        section_label = str(packet.get("section_label", "")).strip().lower()
        if section_label in SECTION_LABELS and section_label != "unknown":
            labels.add(section_label)
            continue
        category = str(packet.get("category", "")).strip().lower()
        if category in SECTION_LABELS and category != "unknown":
            labels.add(category)
    return len(labels & {"methods", "results", "discussion", "conclusion"}) < 2


def _dedupe_packets(packets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, tuple[str, ...], str]] = set()
    out: list[dict[str, Any]] = []
    for packet in packets:
        if not isinstance(packet, dict):
            continue
        statement = " ".join(str(packet.get("statement", "")).lower().split()).strip()
        refs = tuple(sorted(str(ref).strip() for ref in packet.get("evidence_refs", []) if str(ref).strip()))
        modality = str(packet.get("modality", "text")).strip().lower() or "text"
        key = (statement, refs, modality)
        if not statement or key in seen:
            continue
        seen.add(key)
        out.append(packet)
    return out


def _infer_chunk_section(chunk: dict[str, Any], *, idx: int, total_chunks: int) -> str:
    meta = _parse_chunk_meta(chunk.get("meta"))
    meta_label = str(meta.get("section_norm", "") or "").strip().lower()
    if meta_label in SECTION_LABELS and meta_label != "unknown":
        return meta_label
    meta_raw_title = str(meta.get("section_raw_title", "") or "").strip()
    explicit_meta = _explicit_heading_section(meta_raw_title)
    if explicit_meta != "unknown":
        return explicit_meta
    meta_anchor_token = str(meta.get("section_anchor_token", "") or "").strip().lower()
    if meta_anchor_token in SECTION_LABELS and meta_anchor_token != "unknown":
        return meta_anchor_token

    anchor = str(chunk.get("anchor", "")).strip()
    heading = _section_heading_from_anchor(anchor)
    from_heading = _section_from_text(heading)
    if from_heading != "unknown":
        return from_heading

    content = str(chunk.get("content", "")).strip()
    if _is_noise_statement(content):
        return "unknown"
    structured_prefix = _statement_prefix_section(content)
    if structured_prefix != "unknown":
        return structured_prefix
    lowered = content.lower()
    pos = (float(idx) / float(max(1, total_chunks - 1))) if total_chunks > 1 else 0.0
    if CONCLUSION_STRONG_RE.search(lowered) and pos >= 0.55:
        return "conclusion"
    if DISCUSSION_STRONG_RE.search(lowered) and pos >= 0.45:
        return "discussion"
    if METHOD_STRONG_RE.search(lowered) and pos <= 0.62:
        return "methods"
    if RESULT_STRONG_RE.search(lowered) and pos >= 0.2:
        return "results"
    from_content = _section_from_text(content[:400])
    if from_content == "introduction":
        return "introduction"
    if from_content == "conclusion" and pos >= 0.5:
        return "conclusion"
    if from_content == "discussion" and DISCUSSION_STRONG_RE.search(lowered):
        return "discussion"
    if from_content == "methods" and METHOD_STRONG_RE.search(lowered):
        return "methods"
    if from_content == "results" and RESULT_STRONG_RE.search(lowered):
        return "results"

    if pos <= 0.22:
        return "introduction"
    if pos <= 0.45:
        return "methods"
    if pos <= 0.70:
        return "results"
    if pos <= 0.93:
        return "discussion"
    return "conclusion"


def _select_fallback_sentences(content: str, *, section: str) -> list[str]:
    text = str(content or "")
    prefix_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = STRUCTURED_PREFIX_LINE_RE.match(line)
        if not match:
            continue
        prefix = str(match.group(1) or "").strip().lower()
        body = " ".join(str(match.group(2) or "").split()).strip()
        mapped = _statement_prefix_section(f"{prefix}: {body}")
        if mapped == section and body:
            if _is_noise_statement(body):
                continue
            prefix_lines.append(body)
    if prefix_lines:
        return [truncate_text(line, 260) for line in prefix_lines[:2]]

    sentences = [part.strip() for part in SENTENCE_SPLIT_RE.split(" ".join(text.split())) if part.strip()]
    picked: list[str] = []
    for sentence in sentences:
        text = " ".join(sentence.split()).strip()
        if len(text) < 24:
            continue
        if _is_noise_statement(text):
            continue
        if section == "introduction":
            if INTRO_FALLBACK_RE.search(text) or text.lower().startswith(("background", "objective", "aim")):
                picked.append(text)
        elif section == "methods":
            if METHOD_FALLBACK_RE.search(text) and not RESULT_STRONG_RE.search(text):
                picked.append(text)
        elif section == "results":
            if RESULT_STRONG_RE.search(text) or (
                RESULT_FALLBACK_RE.search(text)
                and re.search(r"\b(connectivity|network|cluster|association|effect)\b", text, re.IGNORECASE)
            ):
                picked.append(text)
        elif section == "discussion":
            if DISCUSSION_FALLBACK_RE.search(text) and not RESULT_STRONG_RE.search(text):
                picked.append(text)
        elif section == "conclusion":
            if CONCLUSION_FALLBACK_RE.search(text):
                picked.append(text)
        if len(picked) >= 2:
            break

    if picked:
        return [truncate_text(item, 260) for item in picked]
    for sentence in sentences:
        text = " ".join(sentence.split()).strip()
        if len(text) >= 24:
            return [truncate_text(text, 260)]
    return []


def _parse_chunk_meta(raw_meta: Any) -> dict[str, Any]:
    if isinstance(raw_meta, dict):
        return raw_meta
    if isinstance(raw_meta, str) and raw_meta.strip():
        try:
            parsed = json.loads(raw_meta)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _hydrate_anchor_metadata(packets: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    anchor_meta: dict[str, dict[str, Any]] = {}
    anchor_order: dict[str, int] = {}
    for chunk_idx, chunk in enumerate(chunks):
        anchor = str(chunk.get("anchor", "")).strip()
        if not anchor:
            continue
        anchor_meta[anchor] = _parse_chunk_meta(chunk.get("meta"))
        if anchor not in anchor_order:
            anchor_order[anchor] = chunk_idx
    out: list[dict[str, Any]] = []
    for packet in packets:
        aligned = dict(packet)
        anchor = str(aligned.get("anchor", "")).strip()
        refs = [str(ref).strip() for ref in aligned.get("evidence_refs", []) if str(ref).strip()]
        meta = anchor_meta.get(anchor, {})
        order_idx = anchor_order.get(anchor, -1)
        if (not meta or order_idx < 0) and refs:
            for ref in refs:
                if not meta:
                    meta = anchor_meta.get(ref, {})
                if order_idx < 0 and ref in anchor_order:
                    order_idx = anchor_order[ref]
                if meta and order_idx >= 0:
                    break
        aligned["anchor_meta_section_norm"] = str(meta.get("section_norm", "") or "").strip().lower()
        aligned["anchor_meta_section_raw_title"] = str(meta.get("section_raw_title", "") or "").strip()
        aligned["anchor_meta_source"] = str(meta.get("section_source", meta.get("source", "")) or "").strip().lower()
        aligned["anchor_meta_section_confidence"] = float(meta.get("section_confidence", 0.0) or 0.0)
        aligned["anchor_meta_heading_style_score"] = _clamp_confidence(meta.get("heading_style_score", 0.0))
        try:
            aligned["anchor_meta_heading_level"] = int(float(meta.get("heading_level", -1) or -1))
        except Exception:
            aligned["anchor_meta_heading_level"] = -1
        try:
            aligned["anchor_meta_paragraph_index"] = int(float(meta.get("paragraph_index", -1) or -1))
        except Exception:
            aligned["anchor_meta_paragraph_index"] = -1
        aligned["anchor_meta_anchor_order"] = int(order_idx)
        aligned["anchor_meta_section_anchor_token"] = str(meta.get("section_anchor_token", "") or "").strip().lower()
        out.append(aligned)
    return out


def _anchor_trailing_index(anchor: str) -> int:
    value = str(anchor or "").strip()
    if not value:
        return -1
    match = re.search(r":(\d+)\s*$", value)
    if not match:
        return -1
    try:
        return int(match.group(1))
    except Exception:
        return -1


def _packet_document_sort_key(packet: dict[str, Any], fallback_idx: int) -> tuple[int, int, int, int]:
    try:
        paragraph_index = int(float(packet.get("anchor_meta_paragraph_index", -1) or -1))
    except Exception:
        paragraph_index = -1
    try:
        anchor_order = int(float(packet.get("anchor_meta_anchor_order", -1) or -1))
    except Exception:
        anchor_order = -1
    anchor_tail_idx = _anchor_trailing_index(str(packet.get("anchor", "") or ""))

    if paragraph_index >= 0:
        primary = paragraph_index
    elif anchor_order >= 0:
        primary = anchor_order
    elif anchor_tail_idx >= 0:
        primary = anchor_tail_idx
    else:
        primary = fallback_idx

    secondary = anchor_order if anchor_order >= 0 else fallback_idx
    tertiary = anchor_tail_idx if anchor_tail_idx >= 0 else fallback_idx
    return (int(primary), int(secondary), int(tertiary), int(fallback_idx))


def _annotate_text_packet_sections(packets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed_packets: list[tuple[int, dict[str, Any], tuple[int, int, int, int]]] = []
    for idx, packet in enumerate(packets):
        if not isinstance(packet, dict):
            continue
        aligned = dict(packet)
        indexed_packets.append((idx, aligned, _packet_document_sort_key(aligned, idx)))
    if not indexed_packets:
        return []

    ordered_packets = sorted(indexed_packets, key=lambda item: item[2])
    total_chunks = len(ordered_packets)
    previous_section = "unknown"
    previous_index = -1
    out_by_original_index: dict[int, dict[str, Any]] = {}
    for doc_idx, (original_idx, aligned, _sort_key) in enumerate(ordered_packets):
        section_label, section_confidence, section_source, extra_flags = _resolve_packet_section(
            aligned,
            idx=doc_idx,
            total_chunks=total_chunks,
            previous_section=previous_section,
            previous_index=previous_index,
        )
        aligned["section_label"] = section_label
        aligned["section_confidence"] = section_confidence
        aligned["section_source"] = section_source
        quality_flags = [str(flag).strip() for flag in aligned.get("quality_flags", []) if str(flag).strip()]
        aligned["quality_flags"] = sorted(set(quality_flags + extra_flags))

        if section_label != "unknown":
            previous_section = section_label
            previous_index = doc_idx

        category = str(aligned.get("category", "")).strip().lower()
        if section_label != "unknown":
            if category in {"", "other"}:
                aligned["category"] = section_label
            elif category in SECTION_LABELS and section_source in {"meta", "explicit_heading", "structured_abstract", "anchor", "heading"}:
                aligned["category"] = section_label
        out_by_original_index[original_idx] = aligned
    return [out_by_original_index[idx] for idx, _, _ in indexed_packets if idx in out_by_original_index]


def _position_section_hint(idx: int, total_chunks: int) -> str:
    if total_chunks <= 0:
        return "unknown"
    ratio = float(idx) / float(max(1, total_chunks - 1))
    for label, upper_bound in SECTION_POSITION_BOUNDARIES:
        if ratio <= upper_bound:
            return label
    return "conclusion"


def _section_transition_rank(section: str) -> int:
    return SECTION_ORDER.index(section) if section in SECTION_ORDER else len(SECTION_ORDER)


def _transition_satisfies(previous: str, candidate: str, ratio: float) -> bool:
    if previous == "unknown" or candidate == "unknown":
        return True
    prev_rank = _section_transition_rank(previous)
    cand_rank = _section_transition_rank(candidate)
    if cand_rank >= prev_rank:
        return True
    if ratio < 0.2:
        return False
    if ratio < 0.45 and cand_rank <= prev_rank:
        return False
    return True


def _pick_top_conflict_label(votes: dict[str, float], vote_sources: dict[str, set[str]]) -> tuple[str, float, int] | None:
    ranked = sorted(
        (
            (
                label,
                score,
                min(SECTION_SOURCE_RANK.get(source, 999) for source in vote_sources[label]) if vote_sources[label] else 999,
            )
            for label, score in votes.items()
            if score > 0
        ),
        key=lambda item: (-float(item[1]), int(item[2])),
    )
    if not ranked:
        return None
    return ranked[0]


def _resolve_packet_section(
    packet: dict[str, Any],
    *,
    idx: int,
    total_chunks: int,
    previous_section: str,
    previous_index: int,
) -> tuple[str, float, str, list[str]]:
    flags: list[str] = []
    anchor = str(packet.get("anchor", "") or "").strip()
    statement = str(packet.get("statement", "") or "")
    packet_confidence = float(packet.get("confidence", 0.0) or 0.0)
    packet_section_confidence = float(packet.get("section_confidence", packet_confidence) or 0.0)
    meta_section_confidence = _clamp_confidence(packet.get("anchor_meta_section_confidence", 0.0))
    meta_source = str(packet.get("anchor_meta_source", "") or "").strip().lower()
    meta_raw_title = str(packet.get("anchor_meta_section_raw_title", "") or "").strip()
    meta_heading_style_score = _clamp_confidence(packet.get("anchor_meta_heading_style_score", 0.0))
    try:
        meta_heading_level = int(packet.get("anchor_meta_heading_level", -1) or -1)
    except Exception:
        meta_heading_level = -1
    meta_anchor_token = str(packet.get("anchor_meta_section_anchor_token", "") or "").strip().lower()
    votes: dict[str, float] = {label: 0.0 for label in SECTION_LABELS if label != "unknown"}
    vote_sources: dict[str, set[str]] = {label: set() for label in SECTION_LABELS if label != "unknown"}

    def _vote(label: str, source: str, weight: float) -> None:
        key = str(label or "").strip().lower()
        if key not in votes:
            return
        votes[key] += float(weight)
        vote_sources[key].add(source)

    meta_label = str(packet.get("anchor_meta_section_norm", "") or "").strip().lower()
    if meta_label in votes:
        meta_weight = SECTION_SOURCE_WEIGHTS["meta"] * (0.30 + (0.70 * meta_section_confidence))
        if meta_source == "position":
            meta_weight *= 0.55
        elif meta_source == "fallback":
            meta_weight *= 0.45
        elif meta_source not in {"heading", "structured_abstract", "anchor", "meta"} and meta_section_confidence < 0.5:
            meta_weight *= 0.70
        if meta_source == "heading":
            meta_weight += (0.06 + (0.12 * meta_heading_style_score))
            if meta_heading_level >= 0 and meta_heading_level <= 2:
                meta_weight += 0.06
            if meta_label in {"discussion", "conclusion"}:
                meta_weight += 0.07
        _vote(meta_label, "meta", meta_weight)

    explicit_heading_label = _explicit_heading_section(meta_raw_title)
    if explicit_heading_label == "unknown" and meta_anchor_token in votes:
        explicit_heading_label = meta_anchor_token
    if explicit_heading_label in votes:
        explicit_weight = SECTION_SOURCE_WEIGHTS["explicit_heading"]
        explicit_weight += (0.12 * meta_heading_style_score)
        if meta_heading_level >= 0 and meta_heading_level <= 2:
            explicit_weight += 0.08
        if explicit_heading_label in {"discussion", "conclusion"}:
            explicit_weight += 0.08
        _vote(explicit_heading_label, "explicit_heading", explicit_weight)

    structured_label = _structured_abstract_section_from_statement(statement)
    if structured_label in votes:
        _vote(structured_label, "structured_abstract", SECTION_SOURCE_WEIGHTS["structured_abstract"])

    heading = _section_heading_from_anchor(anchor)
    anchor_label = "unknown"
    anchor_explicit_label = "unknown"
    if heading:
        anchor_label = _section_from_text(heading)
        anchor_explicit_label = _explicit_heading_section(heading)
        if anchor_label in votes:
            anchor_weight = SECTION_SOURCE_WEIGHTS["anchor"]
            if anchor_explicit_label == anchor_label:
                anchor_weight += 0.12
            _vote(anchor_label, "anchor", anchor_weight)
    if anchor.lower() == "abstract":
        _vote("introduction", "anchor", SECTION_SOURCE_WEIGHTS["anchor"])

    prefix_label = _statement_prefix_section(statement)
    if prefix_label in votes:
        _vote(prefix_label, "statement_prefix", SECTION_SOURCE_WEIGHTS["statement_prefix"])
    category = str(packet.get("category", "") or "").strip().lower()
    if category in SECTION_LABELS and category != "unknown":
        _vote(category, "category", SECTION_SOURCE_WEIGHTS["category"])

    lexical_label = _section_from_text(statement)
    if lexical_label in votes:
        _vote(lexical_label, "lexical", SECTION_SOURCE_WEIGHTS["lexical"])

    position_label = _position_section_hint(idx, total_chunks)
    if position_label in votes:
        _vote(position_label, "position", SECTION_SOURCE_WEIGHTS["position"])

    explicit_signal = False
    if explicit_heading_label in votes:
        explicit_signal = True
    elif anchor_explicit_label in votes and anchor_explicit_label == anchor_label:
        explicit_signal = True
    elif prefix_label in votes and STATEMENT_PREFIX_RE.match(statement):
        explicit_signal = True
    elif meta_source in {"heading", "structured_abstract"} and meta_label in votes:
        explicit_signal = True

    ranked = sorted(
        (
            (
                label,
                score,
                min(SECTION_SOURCE_RANK.get(source, 999) for source in vote_sources[label]) if vote_sources[label] else 999,
                sorted(vote_sources[label]) if vote_sources[label] else ["fallback"],
            )
            for label, score in votes.items()
            if score > 0
        ),
        key=lambda item: (-float(item[1]), int(item[2]), -packet_confidence, -packet_section_confidence, int(idx)),
    )
    if not ranked:
        return "unknown", 0.0, "fallback", flags

    winner, winner_score, _winner_rank, winner_sources = ranked[0]
    second_score = float(ranked[1][1]) if len(ranked) > 1 else 0.0

    if (
        anchor_label == "results"
        and prefix_label in {"introduction", "methods"}
        and (votes.get(prefix_label, 0.0) - votes.get("results", 0.0)) >= SECTION_CONFLICT_OVERRIDE_MARGIN
    ):
        winner = prefix_label
        winner_score = votes.get(prefix_label, winner_score)
        flags.append("section_conflict_resolved")

    prev_distance = abs(idx - previous_index) if previous_index >= 0 else total_chunks
    ratio = float(idx) / float(max(1, total_chunks - 1)) if total_chunks > 1 else 0.0
    if previous_section == "conclusion" and winner != "conclusion" and not explicit_signal:
        winner = "conclusion"
        winner_score = max(winner_score, votes.get("conclusion", SECTION_SOURCE_WEIGHTS["position"]))
        winner_sources = ["position"]
        flags.append("late_section_lock")
    elif previous_section == "discussion" and winner in {"introduction", "methods", "results"} and not explicit_signal:
        winner = "discussion"
        winner_score = max(winner_score, votes.get("discussion", SECTION_SOURCE_WEIGHTS["position"]))
        winner_sources = ["position"]
        flags.append("late_section_lock")

    if not _transition_satisfies(previous_section, winner, ratio):
        candidate = _pick_top_conflict_label(votes, vote_sources)
        if candidate is not None:
            alt_label, alt_score, _alt_rank = candidate[0], candidate[1], candidate[2]
            if alt_label == previous_section and (winner_score - alt_score) < (SECTION_MARGIN_THRESHOLD * 1.6):
                winner = alt_label
                winner_score = alt_score
                winner_sources = sorted(vote_sources.get(alt_label, {"position"}))
                flags.append("position_or_transition_smoothed")

    if prev_distance == 1 and winner == "unknown":
        prev_hint = _position_section_hint(previous_index, total_chunks)
        if prev_hint == previous_section:
            winner = previous_section
            winner_score = max(0.2, SECTION_SOURCE_WEIGHTS["position"])
            winner_sources = ["position"]
            flags.append("continuity_recovery")

    if (winner_score - second_score) < SECTION_MARGIN_THRESHOLD:
        if previous_section in {"discussion", "conclusion"} and not explicit_signal:
            winner = previous_section
            winner_score = max(winner_score, votes.get(previous_section, SECTION_SOURCE_WEIGHTS["position"]))
            winner_sources = ["position"]
            flags.append("margin_smoothed_to_previous")
        if (winner_score - second_score) < SECTION_MARGIN_THRESHOLD and position_label in votes:
            if position_label != "unknown" and (
                _transition_satisfies(previous_section, position_label, ratio)
                or previous_section == "unknown"
            ):
                winner = position_label
                winner_score = max(winner_score, votes.get(position_label, 0.0))
                winner_sources = ["position"]
                flags.append("margin_smoothed_to_position")
        if position_label in votes:
            if (winner_score - second_score) < SECTION_MARGIN_THRESHOLD:
                return "unknown", 0.0, "fallback", flags

    if winner_sources:
        winner_source = sorted(set(winner_sources), key=lambda source: SECTION_SOURCE_RANK.get(source, 999))[0]
    else:
        winner_source = "fallback"
    return winner, min(1.0, float(winner_score)), winner_source, flags


def _statement_prefix_section(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown"
    match = STATEMENT_PREFIX_RE.match(text)
    if not match:
        return "unknown"
    token = str(match.group(1) or "").lower()
    if token in {"objective", "objectives", "background", "aim", "aims"}:
        return "introduction"
    if token in {"method", "methods", "design"}:
        return "methods"
    if token in {"results"}:
        return "results"
    if token in {"discussion"}:
        return "discussion"
    if token in {"conclusion", "conclusions"}:
        return "conclusion"
    return "unknown"


def _structured_abstract_section_from_statement(value: str) -> str:
    return _statement_prefix_section(value)


def _explicit_heading_section(value: str) -> str:
    text = " ".join(str(value or "").split()).strip()
    if not text:
        return "unknown"
    match = EXPLICIT_HEADING_RE.match(text)
    if not match:
        return "unknown"
    token = str(match.group(1) or "").strip().lower()
    if token in {"abstract", "introduction", "background", "objective", "objectives", "aim", "aims"}:
        return "introduction"
    if token in {"method", "methods", "materials and methods"}:
        return "methods"
    if token in {"result", "results"}:
        return "results"
    if token == "discussion":
        return "discussion"
    if token in {"conclusion", "conclusions", "summary"}:
        return "conclusion"
    return "unknown"


def _section_from_text(value: str) -> str:
    lowered = str(value or "").lower()
    if not lowered:
        return "unknown"
    if CONCLUSION_SECTION_RE.search(lowered):
        return "conclusion"
    if DISCUSSION_SECTION_RE.search(lowered):
        return "discussion"
    if INTRO_SECTION_RE.search(lowered):
        return "introduction"
    if RESULT_SECTION_RE.search(lowered):
        return "results"
    if METHOD_SECTION_RE.search(lowered):
        return "methods"
    return "unknown"


def _is_noise_statement(value: str) -> bool:
    text = " ".join(str(value or "").split()).strip()
    if not text:
        return True
    if text.startswith("$^{"):
        return True
    if NOISE_STATEMENT_RE.search(text):
        return True
    if re.search(r"\bAm\s+J\s+Psychiatry\b", text):
        return True
    if re.search(r"\b(Drs?\.|Department of|Perelman School|Address correspondence)\b", text):
        return True
    if re.match(r"^(table|figure)\s+\d+\b", text, re.IGNORECASE):
        return True
    if re.fullmatch(r"\d{1,4}", text):
        return True
    letters = [ch for ch in text if ch.isalpha()]
    if letters:
        upper_ratio = sum(1 for ch in letters if ch.isupper()) / len(letters)
        if upper_ratio > 0.85 and len(text) >= 16:
            return True
    if re.search(r"(M\.D\.|Ph\.D\.|B\.A\.|B\.S\.)", text) and (text.count(",") >= 2 or len(text.split()) <= 18):
        return True
    return False


def _section_heading_from_anchor(anchor: str) -> str:
    value = str(anchor or "").strip()
    if not value.lower().startswith("section:"):
        return ""
    parts = value.split(":")
    if len(parts) <= 2:
        return parts[1] if len(parts) == 2 else ""
    return ":".join(parts[1:-1])
