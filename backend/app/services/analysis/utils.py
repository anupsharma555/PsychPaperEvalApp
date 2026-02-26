from __future__ import annotations

import json
import re
from typing import Any

from app.services.analysis.schemas import ModalityEvidence

FIGURE_REF_RE = re.compile(r"\bfig(?:ure)?\.?\s*(s?\d+)([a-z])?", re.IGNORECASE)
TABLE_REF_RE = re.compile(r"\btable\s*(s?\d+)([a-z])?", re.IGNORECASE)
RANGE_RE = re.compile(
    r"\b(?:fig(?:ure)?\.?|table)\s*(s?\d+)\s*[-–]\s*(s?\d+)",
    re.IGNORECASE,
)
SECTION_ANCHOR_RE = re.compile(r"^section:(.*?):(\d+)\s*$", re.IGNORECASE)
ANCHOR_COLON_CANON_RE = re.compile(r":+")
ANCHOR_TITLE_TOKEN_RE = re.compile(r"[a-z0-9]+")


def extract_json(text: str) -> Any:
    text = text.strip()
    if not text:
        return None

    fenced = _strip_fenced_json(text)
    if fenced != text:
        text = fenced.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    candidate = _extract_balanced_json(text)
    if candidate:
        try:
            return json.loads(candidate)
        except Exception:
            pass

    match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            return None
    return None


def _strip_fenced_json(text: str) -> str:
    if "```" not in text:
        return text
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1)
    return text


def _extract_balanced_json(text: str) -> str | None:
    starts = [i for i, ch in enumerate(text) if ch in "{["]
    if not starts:
        return None
    for start in starts:
        stack: list[str] = []
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == "\"":
                    in_string = False
                continue
            if ch == "\"":
                in_string = True
                continue
            if ch in "{[":
                stack.append(ch)
            elif ch in "}]":
                if not stack:
                    break
                open_ch = stack.pop()
                if open_ch == "{" and ch != "}":
                    break
                if open_ch == "[" and ch != "]":
                    break
                if not stack:
                    return text[start : i + 1]
    return None


def max_chars_for_ctx(n_ctx: int, chars_per_token: float = 3.5, safety: float = 0.75) -> int:
    if n_ctx <= 0:
        return 8000
    return int(n_ctx * chars_per_token * safety)


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def truncate_list(items: list[Any], max_items: int) -> list[Any]:
    if max_items <= 0:
        return []
    return items[:max_items]


def extract_expected_refs(texts: list[str]) -> dict[str, list[str]]:
    joined = " ".join(texts)
    figure_refs, figure_raw = _extract_refs(joined, FIGURE_REF_RE)
    table_refs, table_raw = _extract_refs(joined, TABLE_REF_RE)
    return {
        "figure_refs": sorted(figure_refs),
        "figure_raw": sorted(figure_raw),
        "table_refs": sorted(table_refs),
        "table_raw": sorted(table_raw),
    }


def extract_refs_from_text(text: str) -> set[str]:
    refs, _raw = _extract_refs(text, FIGURE_REF_RE)
    refs |= _extract_refs(text, TABLE_REF_RE)[0]
    return refs


def _extract_refs(text: str, pattern: re.Pattern[str]) -> tuple[set[str], set[str]]:
    refs: set[str] = set()
    raw: set[str] = set()
    if not text:
        return refs, raw

    # expand ranges like Figure 1-3 or Fig S1-S3
    for match in RANGE_RE.finditer(text):
        start = match.group(1)
        end = match.group(2)
        start_id = _normalize_ref(start)
        end_id = _normalize_ref(end, prefix=_prefix(start))
        if start_id and end_id:
            start_num = _num(start_id)
            end_num = _num(end_id)
            pref = _prefix(start_id)
            if start_num is not None and end_num is not None and start_num <= end_num:
                for num in range(start_num, end_num + 1):
                    refs.add(f"{pref}{num}")

    for match in pattern.finditer(text):
        raw_id = (match.group(1) or "") + (match.group(2) or "")
        if not raw_id:
            continue
        raw.add(raw_id.upper())
        norm = _normalize_ref(raw_id)
        if norm:
            refs.add(norm)

    return refs, raw


def _normalize_ref(value: str, prefix: str | None = None) -> str:
    if not value:
        return ""
    val = value.strip().upper()
    pref = prefix or _prefix(val)
    digits = re.findall(r"\d+", val)
    if not digits:
        return ""
    return f"{pref}{int(digits[0])}"


def _prefix(value: str) -> str:
    return "S" if value.upper().startswith("S") else ""


def _num(value: str) -> int | None:
    digits = re.findall(r"\d+", value)
    return int(digits[0]) if digits else None


NUMERIC_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*([a-zA-Z%]+)?")
PVALUE_RE = re.compile(r"\bp\s*[=<]\s*(0?\.\d+|\d+)", re.IGNORECASE)
EFFECT_RE = re.compile(r"\b(?:cohen'?s?\s*d|hedges'?g|or|rr)\s*[=:]?\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
SECTION_LABEL_SET = {"introduction", "methods", "results", "discussion", "conclusion", "unknown"}
SECTION_SOURCE_SET = {
    "meta",
    "structured_abstract",
    "anchor",
    "statement_prefix",
    "category",
    "heading",
    "position",
    "lexical",
    "fallback",
}
RESULT_EVIDENCE_TYPE_SET = {"text_primary", "media_support"}


def clamp_confidence(value: Any) -> float:
    try:
        score = float(value)
    except Exception:
        return 0.0
    if score < 0:
        return 0.0
    if score > 1:
        return 1.0
    return score


def ensure_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value).strip()]


def _map_unique_anchor(mapping: dict[Any, str | None], key: Any, value: str) -> None:
    if key in ("", None):
        return
    sentinel = object()
    current = mapping.get(key, sentinel)
    if current is sentinel:
        mapping[key] = value
        return
    if current != value:
        mapping[key] = None


def _anchor_colon_canonical(value: str) -> str:
    token = " ".join(str(value or "").split()).strip().lower()
    token = ANCHOR_COLON_CANON_RE.sub(":", token)
    return token


def _section_anchor_parts(anchor: str) -> tuple[int | None, str]:
    match = SECTION_ANCHOR_RE.match(str(anchor or "").strip())
    if not match:
        return None, ""
    try:
        idx = int(match.group(2))
    except Exception:
        return None, ""
    return idx, str(match.group(1) or "")


def _section_title_canonical(value: str) -> str:
    tokens = ANCHOR_TITLE_TOKEN_RE.findall(str(value or "").lower())
    return " ".join(tokens)


def _build_anchor_resolution_maps(valid_anchors: set[str]) -> dict[str, Any]:
    lower_map: dict[str, str | None] = {}
    colon_map: dict[str, str | None] = {}
    section_index_map: dict[int, str | None] = {}
    section_signature_map: dict[tuple[int, str], str | None] = {}
    for raw in valid_anchors:
        anchor = str(raw or "").strip()
        if not anchor:
            continue
        _map_unique_anchor(lower_map, anchor.lower(), anchor)
        _map_unique_anchor(colon_map, _anchor_colon_canonical(anchor), anchor)
        idx, title = _section_anchor_parts(anchor)
        if idx is None:
            continue
        _map_unique_anchor(section_index_map, idx, anchor)
        title_key = _section_title_canonical(title)
        if title_key:
            _map_unique_anchor(section_signature_map, (idx, title_key), anchor)
    return {
        "lower_map": lower_map,
        "colon_map": colon_map,
        "section_index_map": section_index_map,
        "section_signature_map": section_signature_map,
    }


def _resolve_anchor_ref(value: str, valid_anchors: set[str], maps: dict[str, Any]) -> str:
    anchor = str(value or "").strip()
    if not anchor:
        return ""
    if anchor in valid_anchors:
        return anchor

    lower_map = maps.get("lower_map", {})
    colon_map = maps.get("colon_map", {})
    section_index_map = maps.get("section_index_map", {})
    section_signature_map = maps.get("section_signature_map", {})

    resolved = lower_map.get(anchor.lower())
    if resolved:
        return str(resolved)

    resolved = colon_map.get(_anchor_colon_canonical(anchor))
    if resolved:
        return str(resolved)

    idx, title = _section_anchor_parts(anchor)
    if idx is not None:
        title_key = _section_title_canonical(title)
        if title_key:
            resolved = section_signature_map.get((idx, title_key))
            if resolved:
                return str(resolved)
        resolved = section_index_map.get(idx)
        if resolved:
            return str(resolved)

    return anchor


def normalize_evidence_packets(
    raw_items: list[dict[str, Any]],
    modality: str,
    valid_anchors: set[str],
    *,
    id_prefix: str | None = None,
    default_category: str = "other",
) -> list[dict[str, Any]]:
    packets: list[ModalityEvidence] = []
    seen: set[tuple[str, tuple[str, ...], str]] = set()
    prefix = id_prefix or modality
    valid_anchor_set = {str(anchor or "").strip() for anchor in valid_anchors if str(anchor or "").strip()}
    anchor_maps = _build_anchor_resolution_maps(valid_anchor_set)
    for idx, raw in enumerate(raw_items, start=1):
        if not isinstance(raw, dict):
            continue
        statement = (
            str(raw.get("statement") or raw.get("summary") or raw.get("result") or raw.get("claim") or "").strip()
        )
        if not statement:
            continue
        anchor = _resolve_anchor_ref(str(raw.get("anchor") or "").strip(), valid_anchor_set, anchor_maps)
        evidence_refs = [
            _resolve_anchor_ref(ref, valid_anchor_set, anchor_maps)
            for ref in ensure_str_list(raw.get("evidence_refs") or raw.get("evidence"))
        ]
        evidence_refs = [ref for ref in evidence_refs if ref]
        evidence_refs = list(dict.fromkeys(evidence_refs))
        quality_flags = ensure_str_list(raw.get("quality_flags"))
        if anchor and anchor not in evidence_refs:
            evidence_refs.append(anchor)
        if not anchor and evidence_refs:
            anchor = evidence_refs[0]
        if not anchor:
            anchor = "unknown"
        valid_refs = [ref for ref in evidence_refs if ref in valid_anchor_set]
        if anchor in valid_anchor_set and anchor not in valid_refs:
            valid_refs.append(anchor)
        if not valid_refs:
            quality_flags.append("missing_evidence")
            valid_refs = []

        value, unit = _extract_numeric_value(statement)
        p_value = _extract_p_value(statement)
        effect_size = _extract_effect_size(statement)
        if raw.get("value") is not None:
            value = _safe_float(raw.get("value"))
        if raw.get("unit") is not None:
            unit = str(raw.get("unit") or "").strip() or unit
        if raw.get("p_value") is not None:
            p_value = _safe_float(raw.get("p_value"))
        if raw.get("effect_size") is not None:
            effect_size = _safe_float(raw.get("effect_size"))

        dedupe_key = (
            _canonical_text(statement),
            tuple(sorted(set(valid_refs))),
            modality,
        )
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        packet = ModalityEvidence(
            finding_id=str(raw.get("finding_id") or f"{prefix}-{idx}"),
            modality=modality if modality != "supp" else "supplement",
            anchor=anchor,
            statement=statement,
            evidence_refs=sorted(set(valid_refs)),
            confidence=clamp_confidence(raw.get("confidence", 0.0)),
            quality_flags=sorted(set(flag for flag in quality_flags if flag)),
            value=value,
            unit=unit,
            p_value=p_value,
            effect_size=effect_size,
            category=str(raw.get("category") or default_category),
            section_label=_normalize_section_label(raw.get("section_label")),
            section_confidence=clamp_confidence(raw.get("section_confidence", raw.get("confidence", 0.0))),
            section_source=_normalize_section_source(raw.get("section_source")),
            result_evidence_type=_normalize_result_evidence_type(raw.get("result_evidence_type")),
        )
        packets.append(packet)
    return [packet.model_dump() for packet in packets]


def _normalize_section_label(value: Any) -> str:
    token = str(value or "").strip().lower()
    if token in SECTION_LABEL_SET:
        return token
    return "unknown"


def _normalize_section_source(value: Any) -> str:
    token = str(value or "").strip().lower()
    if token in SECTION_SOURCE_SET:
        return token
    return "fallback"


def _normalize_result_evidence_type(value: Any) -> str | None:
    token = str(value or "").strip().lower()
    if token in RESULT_EVIDENCE_TYPE_SET:
        return token
    return None


def packets_to_legacy_findings(packets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for packet in packets:
        findings.append(
            {
                "category": packet.get("category", "other"),
                "summary": packet.get("statement", ""),
                "evidence": packet.get("evidence_refs", []),
                "confidence": clamp_confidence(packet.get("confidence", 0.0)),
            }
        )
    return findings


def summarize_packet_statements(packets: list[dict[str, Any]], max_items: int = 6) -> list[str]:
    statements = []
    for packet in packets:
        statement = str(packet.get("statement") or "").strip()
        if statement:
            statements.append(_statement_with_provenance(packet, statement))
        if len(statements) >= max_items:
            break
    return statements


def _statement_with_provenance(packet: dict[str, Any], statement: str) -> str:
    finding_id = str(packet.get("finding_id") or "").strip()
    show_finding_id = bool(finding_id) and not finding_id.startswith("text-fallback-")
    anchor = str(packet.get("anchor") or "").strip()
    if anchor.lower() == "unknown":
        anchor = ""
    evidence_refs = [ref for ref in ensure_str_list(packet.get("evidence_refs") or packet.get("evidence")) if ref]
    refs: list[str] = []
    seen: set[str] = set()
    for ref in [anchor, *evidence_refs]:
        value = str(ref or "").strip()
        if not value or value.lower() == "unknown":
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        refs.append(value)

    primary_ref = refs[0] if refs else ""
    secondary_refs = refs[1:4]
    primary_label = _humanize_evidence_ref(primary_ref)
    secondary_labels = [
        label
        for label in (_humanize_evidence_ref(ref) for ref in secondary_refs)
        if label and label != primary_label
    ]
    statement_with_context = _contextualize_statement(statement, primary_label)
    parts: list[str] = []
    if show_finding_id and primary_ref:
        parts.append(f"id:{finding_id}")
    if secondary_labels:
        parts.append("refs:" + ", ".join(secondary_labels))

    line = statement_with_context
    if primary_ref:
        line = f"{line} [{primary_ref}]"
    elif show_finding_id:
        line = f"{line} [id:{finding_id}]"
    if parts:
        line = f"{line} ({'; '.join(parts)})"
    return line


def _humanize_evidence_ref(ref: str) -> str:
    value = str(ref or "").strip()
    if not value:
        return ""
    lowered = value.lower()
    plain_figure = re.fullmatch(r"(?:fig(?:ure)?[_:\s-]*)?f?(\d+[a-z]?)", value, flags=re.IGNORECASE)
    if plain_figure and lowered.startswith("f"):
        return f"Figure {plain_figure.group(1).upper()}"
    plain_table = re.fullmatch(r"(?:table[_:\s-]*)?t?(\d+[a-z]?)", value, flags=re.IGNORECASE)
    if plain_table and lowered.startswith("t"):
        return f"Table {plain_table.group(1).upper()}"
    supp_figure = re.fullmatch(r"s(?:upp(?:lement)?)?[_:\s-]*f(?:ig(?:ure)?)?[_:\s-]*(\d+[a-z]?)", value, flags=re.IGNORECASE)
    if supp_figure:
        return f"Supplement Figure {supp_figure.group(1).upper()}"
    supp_table = re.fullmatch(r"s(?:upp(?:lement)?)?[_:\s-]*t(?:able)?[_:\s-]*(\d+[a-z]?)", value, flags=re.IGNORECASE)
    if supp_table:
        return f"Supplement Table {supp_table.group(1).upper()}"
    if lowered.startswith("figure:"):
        token = value.split(":", 1)[1]
        page_match = re.search(r"page[_\s-]*(\d+)", token, flags=re.IGNORECASE)
        if page_match:
            return f"Figure (page {page_match.group(1)})"
        fig_match = re.search(r"(?:fig(?:ure)?[_\s-]*)?([a-z]?\d+[a-z]?|s\d+[a-z]?)", token, flags=re.IGNORECASE)
        if fig_match:
            return f"Figure {fig_match.group(1).upper()}"
        return f"Figure {token}"
    if lowered.startswith("table:"):
        token = value.split(":", 1)[1]
        page_match = re.search(r"page[_\s-]*(\d+)", token, flags=re.IGNORECASE)
        if page_match:
            return f"Table (page {page_match.group(1)})"
        table_match = re.search(r"(?:table[_\s-]*)?([a-z]?\d+[a-z]?|s\d+[a-z]?)", token, flags=re.IGNORECASE)
        if table_match:
            return f"Table {table_match.group(1).upper()}"
        return f"Table {token}"
    if lowered.startswith("section:"):
        section = value.split(":", 1)[1]
        return f"Section {section.replace(':', ' > ')}"
    if lowered.startswith("supp"):
        suffix = re.sub(r"^supp(?:lement)?[:_\s-]*", "", value, flags=re.IGNORECASE).strip()
        return f"Supplement {suffix or value}"
    return value


def _contextualize_statement(statement: str, primary_label: str) -> str:
    text = str(statement or "").strip()
    label = str(primary_label or "").strip()
    if not text or not label:
        return text
    lowered = label.lower()
    if lowered.startswith("figure"):
        text = re.sub(r"\b[Tt]he figure\b", label, text)
        text = re.sub(r"\b[Ff]igure\b(?!\s*[A-Za-z]?\d)", label, text)
    elif lowered.startswith("table"):
        text = re.sub(r"\b[Tt]he table\b", label, text)
        text = re.sub(r"\b[Tt]able\b(?!\s*[A-Za-z]?\d)", label, text)
    elif lowered.startswith("supplement"):
        text = re.sub(r"\b[Tt]he supplementary (?:figure|table|material)\b", label, text)
        text = re.sub(r"\b[Tt]he supplement(?:ary)?\b", label, text)
    return text


def _canonical_text(text: str) -> str:
    return " ".join((text or "").lower().split())


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _extract_numeric_value(text: str) -> tuple[float | None, str | None]:
    match = NUMERIC_RE.search(text or "")
    if not match:
        return None, None
    number = _safe_float(match.group(1))
    unit = (match.group(2) or "").strip() or None
    return number, unit


def _extract_p_value(text: str) -> float | None:
    match = PVALUE_RE.search(text or "")
    if not match:
        return None
    return _safe_float(match.group(1))


def _extract_effect_size(text: str) -> float | None:
    match = EFFECT_RE.search(text or "")
    if not match:
        return None
    return _safe_float(match.group(1))
