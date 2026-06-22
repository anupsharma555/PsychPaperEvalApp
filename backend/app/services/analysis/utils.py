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


def clean_source_excerpt(value: Any, *, max_chars: int = 900) -> str:
    text = ""
    if isinstance(value, list):
        text = " ".join(str(item or "") for item in value)
    else:
        text = str(value or "")
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    return text[:max_chars].rsplit(" ", 1)[0].strip() if len(text) > max_chars else text


def add_source_excerpts_to_packets(
    raw_items: list[dict[str, Any]],
    anchor_excerpts: dict[str, Any],
    *,
    max_chars: int = 900,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    clean_map = {
        str(anchor or "").strip(): clean_source_excerpt(excerpt, max_chars=max_chars)
        for anchor, excerpt in anchor_excerpts.items()
        if str(anchor or "").strip()
    }
    valid_anchors = set(clean_map)
    anchor_maps = _build_anchor_resolution_maps(valid_anchors)
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        updated = dict(item)
        if not clean_source_excerpt(updated.get("source_excerpt"), max_chars=max_chars):
            refs = ensure_str_list(updated.get("evidence_refs") or updated.get("evidence"))
            anchor_candidates = [str(updated.get("anchor") or "").strip()] + [str(ref or "").strip() for ref in refs]
            for anchor in anchor_candidates:
                excerpt = _source_excerpt_for_anchor(anchor, clean_map, valid_anchors, anchor_maps)
                if excerpt:
                    updated["source_excerpt"] = excerpt
                    break
        out.append(updated)
    return out


def _source_excerpt_for_anchor(
    anchor: str,
    clean_map: dict[str, str],
    valid_anchors: set[str],
    anchor_maps: dict[str, Any],
) -> str:
    token = str(anchor or "").strip()
    if not token:
        return ""
    if token in clean_map:
        return clean_map[token]
    resolved = _resolve_anchor_ref(token, valid_anchors, anchor_maps)
    return clean_map.get(resolved, "")


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
DETAIL_TYPE_SET = {
    "medication_or_therapeutic",
    "dose_schedule",
    "intervention_or_exposure",
    "outcome_measure",
    "adverse_event",
    "model_system",
    "assay_readout",
    "statistical_result",
    "data_source_or_design",
    "rationale_or_objective",
    "interpretation_or_implication",
    "limitation_or_caution",
    "conclusion_or_takeaway",
    "tool_or_algorithm",
    "cross_modal_result",
    "secondary_finding",
    "sensitivity_analysis",
}
DETAIL_CATEGORY_HINTS = {
    "medication": "medication_or_therapeutic",
    "therapeutic": "medication_or_therapeutic",
    "dose": "dose_schedule",
    "intervention": "intervention_or_exposure",
    "exposure": "intervention_or_exposure",
    "clinical": "outcome_measure",
    "outcome": "outcome_measure",
    "assay": "assay_readout",
    "model_system": "model_system",
    "model": "model_system",
    "stats": "statistical_result",
    "statistical": "statistical_result",
    "table_quality": "statistical_result",
    "data_consistency": "statistical_result",
    "objective": "rationale_or_objective",
    "rationale": "rationale_or_objective",
    "interpretation": "interpretation_or_implication",
    "implications": "interpretation_or_implication",
    "limitations": "limitation_or_caution",
    "limitation": "limitation_or_caution",
    "conclusion": "conclusion_or_takeaway",
    "tool": "tool_or_algorithm",
    "algorithm": "tool_or_algorithm",
}
DETAIL_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "medication_or_therapeutic",
        re.compile(
            r"\b(medication|drug|therapeutic|pharmacologic|antidepressant|antipsychotic|"
            r"ketamine|ssri|snri|lithium|clozapine|risperidone|olanzapine|quetiapine)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "dose_schedule",
        re.compile(
            r"\b(\d+(?:\.\d+)?\s*(?:mg|mcg|ug|g|ml|iu|units?)\b|dose|dosage|route|"
            r"daily|weekly|administered|duration|weeks?|months?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "intervention_or_exposure",
        re.compile(
            r"\b(intervention|treatment|therapy|exposure|comparator|control arm|placebo|"
            r"stimulation|training|randomi[sz]ed|trial arm)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "outcome_measure",
        re.compile(
            r"\b(outcome|endpoint|measure|scale|questionnaire|instrument|score|timepoint|"
            r"follow[- ]up|symptom)\b",
            re.IGNORECASE,
        ),
    ),
    ("adverse_event", re.compile(r"\b(adverse event|side effect|safety|tolerability|dropout)\b", re.IGNORECASE)),
    (
        "model_system",
        re.compile(
            r"\b(cell lines?|organoid|mouse|mice|rat|animal model|in vivo|in vitro|specimens?|"
            r"culture|construct|plasmid|vector|transfection|mutagenesis)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "assay_readout",
        re.compile(
            r"\b(assay|readout|qpcr|pcr|western blot|elisa|rna[- ]?seq|sequencing|"
            r"flow cytometry|microscopy|biomarker|protein|gene expression|validation)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "statistical_result",
        re.compile(
            r"\b(p\s*[<=>]\s*0?\.\d+|confidence interval|ci\b|odds ratio|hazard ratio|"
            r"effect size|regression|bayesian|subgroup|adjusted|covariate|significant|"
            r"increased|decreased|higher|lower)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "data_source_or_design",
        re.compile(
            r"\b(cohort|registry|claims database|survey|dataset|data source|cross[- ]sectional|"
            r"case[- ]control|longitudinal|systematic review|meta[- ]analysis|eligibility criteria|"
            r"search strategy|risk of bias)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "rationale_or_objective",
        re.compile(r"\b(objective|aim|hypothesis|rationale|research question|knowledge gap)\b", re.IGNORECASE),
    ),
    (
        "interpretation_or_implication",
        re.compile(
            r"\b(interpret(?:ation|ed)?|implication|may reflect|suggests?|supports?|"
            r"consistent with|clinical relevance|mechanism)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "limitation_or_caution",
        re.compile(
            r"\b(limitation|caution|generalizability|underpowered|bias|confounding|"
            r"cannot establish|future work|future research|replication needed)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "conclusion_or_takeaway",
        re.compile(r"\b(conclusion|in conclusion|overall|taken together|these findings|these results)\b", re.IGNORECASE),
    ),
    (
        "tool_or_algorithm",
        re.compile(r"\b(algorithm|software|pipeline|classifier|benchmark|validation dataset|model performance)\b", re.IGNORECASE),
    ),
    ("secondary_finding", re.compile(r"\b(secondary|exploratory|post hoc|additional analysis)\b", re.IGNORECASE)),
    ("sensitivity_analysis", re.compile(r"\b(sensitivity analysis|robustness check|sensitivity model)\b", re.IGNORECASE)),
]
SECTION_LABEL_SET = {"introduction", "methods", "results", "discussion", "conclusion", "unknown"}
SECTION_SOURCE_SET = {
    "meta",
    "explicit_heading",
    "structured_abstract",
    "anchor",
    "statement_prefix",
    "category",
    "semantic",
    "heading",
    "heading_style",
    "position",
    "lexical",
    "llm_section_extract",
    "llm_narrative_summary",
    "llm_narrative_study_purpose",
    "llm_narrative_study_hypothesis",
    "llm_narrative_central_finding",
    "section_boundary_ledger",
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
        elif anchor not in valid_anchor_set:
            anchor = valid_refs[0]

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

        category = str(raw.get("category") or default_category)
        section_label, section_source_hint = _infer_section_label_with_source(
            raw.get("section_label"),
            anchor=anchor,
            category=category,
            statement=statement,
        )
        section_source = _normalize_section_source(raw.get("section_source"))
        if section_source == "fallback" and section_label != "unknown" and section_source_hint:
            section_source = section_source_hint

        packet = ModalityEvidence(
            finding_id=str(raw.get("finding_id") or f"{prefix}-{idx}"),
            modality=modality if modality != "supp" else "supplement",
            anchor=anchor,
            statement=statement,
            source_excerpt=clean_source_excerpt(
                raw.get("source_excerpt")
                or raw.get("source_text")
                or raw.get("verbatim_text")
                or raw.get("caption")
                or raw.get("legend")
                or raw.get("ocr_text")
            ),
            evidence_refs=sorted(set(valid_refs)),
            confidence=clamp_confidence(raw.get("confidence", 0.0)),
            quality_flags=sorted(set(flag for flag in quality_flags if flag)),
            value=value,
            unit=unit,
            p_value=p_value,
            effect_size=effect_size,
            category=str(raw.get("category") or default_category),
            detail_types=infer_scientific_detail_types(
                raw.get("detail_types"),
                category=category,
                statement=statement,
                modality=modality if modality != "supp" else "supplement",
            ),
            section_label=section_label,
            section_confidence=clamp_confidence(raw.get("section_confidence", raw.get("confidence", 0.0))),
            section_source=section_source,
            result_evidence_type=_normalize_result_evidence_type(raw.get("result_evidence_type")),
        )
        packets.append(packet)
    return [packet.model_dump() for packet in packets]


def _normalize_section_label(value: Any) -> str:
    token = str(value or "").strip().lower()
    if token in SECTION_LABEL_SET:
        return token
    return "unknown"


def _infer_section_label_with_source(
    value: Any,
    *,
    anchor: str,
    category: str,
    statement: str,
) -> tuple[str, str]:
    label = _normalize_section_label(value)
    if label != "unknown":
        return label, ""

    anchor_label = _section_label_from_anchor(anchor)
    if anchor_label != "unknown":
        return anchor_label, "anchor"

    category_label = _section_label_from_category(category)
    if category_label != "unknown":
        return category_label, "category"

    statement_label = _section_label_from_statement_prefix(statement)
    if statement_label != "unknown":
        return statement_label, "statement_prefix"

    return "unknown", ""


def _section_label_from_anchor(anchor: str) -> str:
    _idx, title = _section_anchor_parts(anchor)
    return _section_label_from_tokens(_category_tokens(title))


def _section_label_from_category(category: str) -> str:
    tokens = _category_tokens(category)
    if "stats" in tokens or "statistical" in tokens:
        return "results"
    if "limitation" in tokens or "limitations" in tokens:
        return "discussion"
    if "objective" in tokens or "rationale" in tokens:
        return "introduction"
    return _section_label_from_tokens(tokens)


def _section_label_from_statement_prefix(statement: str) -> str:
    text = str(statement or "").strip().lower()
    match = re.match(r"^(introduction|background|methods?|results?|discussion|conclusions?)\s*[:\-]", text)
    if not match:
        return "unknown"
    return _section_label_from_tokens([match.group(1)])


def _section_label_from_tokens(tokens: list[str]) -> str:
    token_set = {str(token or "").strip().lower() for token in tokens if str(token or "").strip()}
    if token_set & {"introduction", "intro", "background"}:
        return "introduction"
    if token_set & {"method", "methods", "methodology", "design"}:
        return "methods"
    if token_set & {"result", "results", "finding", "findings"}:
        return "results"
    if token_set & {"discussion", "interpretation"}:
        return "discussion"
    if token_set & {"conclusion", "conclusions", "concluding"}:
        return "conclusion"
    return "unknown"


def infer_scientific_detail_types(
    raw_detail_types: Any,
    *,
    category: str,
    statement: str,
    modality: str,
) -> list[str]:
    found: list[str] = []
    for value in ensure_str_list(raw_detail_types):
        normalized = _normalize_detail_type(value)
        if normalized:
            found.append(normalized)
    for token in _category_tokens(category):
        detail_type = DETAIL_CATEGORY_HINTS.get(token)
        if detail_type:
            found.append(detail_type)
    text = str(statement or "")
    for detail_type, pattern in DETAIL_PATTERNS:
        if pattern.search(text):
            found.append(detail_type)
    if str(modality or "").strip().lower() in {"table", "figure", "supplement"}:
        if any(
            detail_type in found
            for detail_type in (
                "statistical_result",
                "outcome_measure",
                "assay_readout",
                "secondary_finding",
                "sensitivity_analysis",
            )
        ):
            found.append("cross_modal_result")
    return _unique_detail_types(found)


def _normalize_detail_type(value: Any) -> str:
    token = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return token if token in DETAIL_TYPE_SET else ""


def _category_tokens(value: str) -> list[str]:
    text = str(value or "").strip().lower()
    if not text:
        return []
    return [token for token in re.split(r"[^a-z0-9]+", text) if token]


def _unique_detail_types(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = _normalize_detail_type(value)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out[:8]


def _normalize_section_source(value: Any) -> str:
    token = str(value or "").strip().lower()
    if token.startswith("section_boundary_ledger:"):
        return "section_boundary_ledger"
    if token in SECTION_SOURCE_SET:
        return token
    return "fallback"


def _normalize_result_evidence_type(value: Any) -> str | None:
    token = str(value or "").strip().lower()
    if token in RESULT_EVIDENCE_TYPE_SET:
        return token
    return None


def filter_grounded_evidence_packets(packets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop normalized packets that cannot be traced back to parsed source anchors."""
    filtered: list[dict[str, Any]] = []
    for packet in packets:
        if not isinstance(packet, dict):
            continue
        flags = {str(flag or "").strip().lower() for flag in ensure_str_list(packet.get("quality_flags"))}
        refs = [str(ref or "").strip() for ref in ensure_str_list(packet.get("evidence_refs"))]
        refs = [ref for ref in refs if ref and ref.lower() != "unknown"]
        anchor = str(packet.get("anchor") or "").strip()
        if "missing_evidence" in flags:
            continue
        if not refs:
            continue
        if not anchor or anchor.lower() == "unknown":
            continue
        filtered.append(packet)
    return filtered


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
