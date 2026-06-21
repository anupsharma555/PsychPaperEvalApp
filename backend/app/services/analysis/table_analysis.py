from __future__ import annotations

import json
import re
from typing import Any

from app.core.config import settings
from app.services.analysis.llm import chat_text_fast
from app.services.analysis.prompts import TABLE_ANALYSIS_SYSTEM
from app.services.analysis.utils import (
    add_source_excerpts_to_packets,
    extract_json,
    filter_grounded_evidence_packets,
    max_chars_for_ctx,
    normalize_evidence_packets,
    packets_to_legacy_findings,
)

NUMERIC_SIGNAL_RE = re.compile(r"(?<![A-Za-z0-9])-?(?:\d+(?:\.\d+)?|\.\d+)(?:\s*(?:%|mg/kg|mg|kg|ml|iu|or|ci))?", re.IGNORECASE)
STAT_SIGNAL_RE = re.compile(
    r"\b(p\s*[<=>]\s*0?\.\d+|odds ratio|hazard ratio|confidence interval|ci\b|or\b|hr\b|"
    r"mape|ic50|auc|sensitivity|specificity|positive|negative|significant|increased|decreased|"
    r"higher|lower|response|risk|rate|prevalence)\b",
    re.IGNORECASE,
)
METHOD_SIGNAL_RE = re.compile(
    r"\b(sample|participants?|patients?|controls?|cohort|dataset|data source|model|models|"
    r"group|groups|arm|variant|variants|structure|pdb|construct|plasmid|vector|cell line|"
    r"clone|passage|intervention|exposure|criteria|enrolled|included)\b",
    re.IGNORECASE,
)
BASELINE_METHOD_RE = re.compile(
    r"\b(baseline|characteristics?|demographics?|study population|sample size|healthy controls?|"
    r"patients?|participants?|controls?|cohort|enrolled|included)\b",
    re.IGNORECASE,
)
RESULT_SIGNAL_RE = re.compile(
    r"\b(results?|outcome|effect|association|odds ratio|hazard ratio|confidence interval|"
    r"p[-_ ]?value|mape|ic50|expression|positive|negative|response|rate|significant|"
    r"increased|decreased|higher|lower|vs\.?|versus)\b",
    re.IGNORECASE,
)


def _table_preview(table_json: str, max_rows: int = 20) -> str:
    try:
        data = json.loads(table_json)
        if "data" in data and "columns" in data:
            rows = data["data"][:max_rows]
            cols = data["columns"]
            lines = ["\t".join(map(str, cols))]
            for row in rows:
                lines.append("\t".join(map(str, row)))
            return "\n".join(lines)
    except Exception:
        return table_json[:4000]
    return table_json[:4000]


def _table_extractive_preview(table_json: str, *, meta: Any = None, max_rows: int = 80) -> str:
    caption = _table_caption(meta)
    lines: list[str] = []
    if caption:
        lines.append(f"caption: {caption}")
    lines.extend(_table_structured_lines(table_json, max_rows=max_rows))
    if not lines:
        return _table_preview(table_json, max_rows=max_rows)
    return "\n".join(lines)[:8000]


def _table_caption(meta: Any) -> str:
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:
            return ""
    if not isinstance(meta, dict):
        return ""
    for key in ("caption", "label", "title", "table_id"):
        value = str(meta.get(key) or "").strip()
        if value:
            return re.sub(r"\s+", " ", value)
    return ""


def _table_structured_lines(table_json: str, *, max_rows: int = 80) -> list[str]:
    try:
        data = json.loads(table_json)
    except Exception:
        return _plain_table_lines(table_json, max_rows=max_rows)
    if not isinstance(data, dict) or "data" not in data or "columns" not in data:
        return _plain_table_lines(table_json, max_rows=max_rows)
    columns = [str(column or "").strip() for column in data.get("columns", [])]
    rows = data.get("data", [])
    if not isinstance(rows, list):
        return []
    out: list[str] = []
    if columns:
        out.append("columns: " + "; ".join(column for column in columns if column))
    for row in rows[:max_rows]:
        if not isinstance(row, list):
            text = re.sub(r"\s+", " ", str(row or "")).strip()
            if text:
                out.append(text)
            continue
        if columns and len(columns) == len(row):
            parts = [
                f"{column}: {value}"
                for column, value in zip(columns, row)
                if column and str(value or "").strip()
            ]
            text = "; ".join(parts)
        else:
            text = "; ".join(str(value or "").strip() for value in row if str(value or "").strip())
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            out.append(text)
    return out


def _plain_table_lines(value: str, *, max_rows: int = 80) -> list[str]:
    lines = [re.sub(r"\s+", " ", line).strip() for line in str(value or "").splitlines()]
    return [line for line in lines if line][:max_rows]


def _table_analysis_prompt(blocks: list[str], *, max_chars: int) -> str:
    prefix = (
        "Analyze the tables for key results and issues. "
        "Look for sample sizes, effect sizes, p-values, subgroup results, and inconsistencies. "
        "Preserve medication/intervention names, dose, route, duration, comparator arms, adverse events, "
        "assay readouts, model-system labels, units, confidence intervals, and outcome timepoints when present. "
        "Cite anchors.\n\n"
    )
    if not blocks:
        return prefix.rstrip()
    selected: list[str] = []
    current_len = len(prefix)
    separator_len = 2
    for block in blocks:
        block_text = str(block or "").strip()
        if not block_text:
            continue
        projected = current_len + len(block_text) + (separator_len if selected else 0)
        if projected <= max_chars:
            selected.append(block_text)
            current_len = projected
            continue
        remaining = max_chars - current_len - (separator_len if selected else 0)
        if remaining >= 240:
            selected.append(_compact_table_block(block_text, max_chars=remaining))
        break
    if not selected and blocks:
        selected.append(_compact_table_block(str(blocks[0]), max_chars=max(240, max_chars - len(prefix))))
    return prefix + "\n\n".join(selected)


def _compact_table_block(block: str, *, max_chars: int) -> str:
    text = str(block or "").strip()
    if len(text) <= max_chars:
        return text
    lines = text.splitlines()
    if not lines:
        return text[:max(0, max_chars - 3)].rstrip() + "..."
    header = lines[0]
    kept = [header]
    current_len = len(header)
    for line in lines[1:]:
        projected = current_len + 1 + len(line)
        if projected > max_chars - 3:
            break
        kept.append(line)
        current_len = projected
    compact = "\n".join(kept).strip()
    if len(compact) > max_chars:
        compact = compact[:max(0, max_chars - 3)].rstrip()
    return compact + "..."


def analyze_tables(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    max_tables = settings.analysis_max_tables
    valid_anchors = {str(chunk.get("anchor", "unknown")) for chunk in chunks}
    blocks: list[str] = []
    anchor_excerpts: dict[str, str] = {}
    for chunk in chunks[:max_tables]:
        anchor = chunk.get("anchor", "unknown")
        content = chunk.get("content", "")
        preview = _table_preview(content)
        anchor_excerpts[str(anchor)] = _table_extractive_preview(content, meta=chunk.get("meta"))
        blocks.append(f"[TABLE {anchor}]\n{preview}")

    if not blocks:
        return {"findings": [], "results": [], "evidence_packets": []}

    if settings.analysis_local_evidence_first_active:
        evidence_packets = filter_grounded_evidence_packets(
            normalize_evidence_packets(
                _extractive_table_packets(anchor_excerpts, reason="local_evidence_first"),
                "table",
                valid_anchors,
                default_category="table_extractive_summary",
            )
        )
        findings = packets_to_legacy_findings(evidence_packets)
        results = [
            {
                "result": packet.get("statement", ""),
                "evidence": packet.get("evidence_refs", []),
                "confidence": packet.get("confidence", 0.0),
            }
            for packet in evidence_packets
            if packet.get("statement")
        ]
        return {
            "findings": findings,
            "results": results,
            "evidence_packets": evidence_packets,
            "diagnostics": {
                "local_evidence_first": True,
                "extractive_fallback_packets": len(evidence_packets),
                "prompt_chars": [],
                "prompt_blocks": [len(blocks)],
            },
        }

    prompt = _table_analysis_prompt(blocks, max_chars=max_chars_for_ctx(settings.llm_n_ctx))
    prompt_chars = len(prompt)

    response = chat_text_fast(prompt, system=TABLE_ANALYSIS_SYSTEM)
    data = _normalize_llm_payload(extract_json(response))

    raw_packets = add_source_excerpts_to_packets(data.get("evidence_packets", []), anchor_excerpts)
    for finding in data.get("findings", []):
        evidence = finding.get("evidence") or []
        anchor = evidence[0] if isinstance(evidence, list) and evidence else ""
        raw_packets.append(
            {
                "finding_id": finding.get("finding_id"),
                "anchor": anchor,
                "statement": finding.get("summary", ""),
                "evidence_refs": evidence,
                "confidence": finding.get("confidence", 0.0),
                "category": finding.get("category", "table_quality"),
                "source_excerpt": anchor_excerpts.get(str(anchor), ""),
            }
        )
    for result in data.get("results", []):
        evidence = result.get("evidence") or []
        anchor = evidence[0] if isinstance(evidence, list) and evidence else ""
        raw_packets.append(
            {
                "finding_id": result.get("finding_id"),
                "anchor": anchor,
                "statement": result.get("result", ""),
                "evidence_refs": evidence,
                "confidence": result.get("confidence", 0.0),
                "category": "stats",
                "value": result.get("value"),
                "unit": result.get("unit"),
                "p_value": result.get("p_value"),
                "effect_size": result.get("effect_size"),
                "source_excerpt": anchor_excerpts.get(str(anchor), ""),
            }
        )

    evidence_packets = normalize_evidence_packets(
        raw_packets,
        "table",
        valid_anchors,
        default_category="table_quality",
    )
    grounded_packets = filter_grounded_evidence_packets(evidence_packets)
    dropped_ungrounded = len(evidence_packets) - len(grounded_packets)
    evidence_packets = grounded_packets
    extractive_fallback_packets = 0
    if not evidence_packets:
        fallback_packets = _extractive_table_packets(anchor_excerpts, reason="llm_empty_output")
        evidence_packets = filter_grounded_evidence_packets(
            normalize_evidence_packets(
                fallback_packets,
                "table",
                valid_anchors,
                default_category="table_extractive_summary",
            )
        )
        extractive_fallback_packets = len(evidence_packets)
    findings = packets_to_legacy_findings(evidence_packets)
    results = [
        {
            "result": packet.get("statement", ""),
            "evidence": packet.get("evidence_refs", []),
            "confidence": packet.get("confidence", 0.0),
        }
        for packet in evidence_packets
        if packet.get("statement")
    ]
    return {
        "findings": findings,
        "results": results,
        "evidence_packets": evidence_packets,
        "diagnostics": {
            "dropped_ungrounded_packets": dropped_ungrounded,
            "extractive_fallback_packets": extractive_fallback_packets,
            "prompt_chars": [prompt_chars],
            "prompt_blocks": [len(blocks)],
        },
    }


def _extractive_table_packets(anchor_excerpts: dict[str, str], *, reason: str) -> list[dict[str, Any]]:
    packets: list[dict[str, Any]] = []
    for idx, (anchor, preview) in enumerate(anchor_excerpts.items(), start=1):
        section_label = _infer_table_section(preview)
        statement = _extractive_table_statement(preview)
        if not statement:
            continue
        packets.append(
            {
                "finding_id": f"table-extractive-{idx}",
                "anchor": anchor,
                "statement": statement,
                "evidence_refs": [anchor],
                "confidence": 0.45,
                "category": "stats" if section_label == "results" else "table_extractive_summary",
                "section_label": section_label,
                "section_confidence": 0.55 if section_label != "unknown" else 0.0,
                "section_source": "semantic" if section_label != "unknown" else "fallback",
                "source_excerpt": preview,
                "quality_flags": ["extractive_fallback", reason],
            }
        )
        packets.extend(_extractive_table_row_packets(anchor, preview, table_index=idx, reason=reason))
    return packets


def _extractive_table_statement(preview: str) -> str:
    lines = [re.sub(r"\s+", " ", line).strip() for line in str(preview or "").splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return ""
    joined = "; ".join(lines[:5])
    if len(joined) > 360:
        joined = joined[:357].rsplit(" ", 1)[0].rstrip(" ,;:") + "..."
    return f"Extracted table content reports {joined}."


def _extractive_table_row_packets(
    anchor: str,
    preview: str,
    *,
    table_index: int,
    reason: str,
    max_rows: int = 8,
) -> list[dict[str, Any]]:
    rows = _ranked_signal_rows(preview, max_rows=max_rows)
    packets: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows, start=1):
        section_label = _infer_table_section(row, table_context=preview)
        category = _table_row_category(row, section_label=section_label)
        statement = _table_row_statement(row)
        if not statement:
            continue
        packets.append(
            {
                "finding_id": f"table-extractive-{table_index}-row-{row_index}",
                "anchor": anchor,
                "statement": statement,
                "evidence_refs": [anchor],
                "confidence": 0.58 if section_label != "unknown" else 0.48,
                "category": category,
                "section_label": section_label,
                "section_confidence": 0.62 if section_label != "unknown" else 0.0,
                "section_source": "semantic" if section_label != "unknown" else "fallback",
                "source_excerpt": row,
                "quality_flags": ["extractive_fallback", "table_row_extraction", reason],
            }
        )
    return packets


def _ranked_signal_rows(preview: str, *, max_rows: int) -> list[str]:
    candidates: list[tuple[int, int, str]] = []
    for idx, line in enumerate(_plain_table_lines(preview, max_rows=120)):
        cleaned = _clean_table_row_text(line)
        if not cleaned:
            continue
        score = _table_row_signal_score(cleaned)
        if score <= 0:
            continue
        candidates.append((-score, idx, cleaned))
    candidates.sort()
    selected = [line for _score, _idx, line in candidates[:max_rows]]
    return selected


def _clean_table_row_text(line: str) -> str:
    text = re.sub(r"\s+", " ", str(line or "")).strip()
    if not text:
        return ""
    lowered = text.lower()
    if lowered.startswith("columns:") or lowered.startswith("caption:"):
        return ""
    text = re.sub(r"^table_text:\s*", "", text, flags=re.IGNORECASE).strip()
    # Drop OCR-split title rows; the summary packet already preserves table title/caption context.
    compact = re.sub(r"\s+", "", text).lower()
    if compact.startswith("table") and len(text) < 160:
        return ""
    if compact.startswith("table") and len(text) < 240 and BASELINE_METHOD_RE.search(text):
        return ""
    if compact.startswith("table") and len(text) < 120 and not NUMERIC_SIGNAL_RE.search(text):
        return ""
    return text


def _table_row_signal_score(line: str) -> int:
    text = str(line or "")
    score = 0
    if NUMERIC_SIGNAL_RE.search(text):
        score += 3
    if STAT_SIGNAL_RE.search(text):
        score += 4
    if METHOD_SIGNAL_RE.search(text):
        score += 2
    if RESULT_SIGNAL_RE.search(text):
        score += 3
    if len(text) > 30:
        score += 1
    return score


def _table_row_statement(row: str) -> str:
    text = _clean_table_row_text(row).rstrip(".")
    if not text:
        return ""
    if len(text) > 420:
        text = text[:417].rsplit(" ", 1)[0].rstrip(" ,;:") + "..."
    return f"Table row reports {text}."


def _infer_table_section(text: str, *, table_context: str = "") -> str:
    combined = f"{text} {table_context}".strip()
    row_text = str(text or "")
    if BASELINE_METHOD_RE.search(row_text):
        return "methods"
    if RESULT_SIGNAL_RE.search(row_text) or STAT_SIGNAL_RE.search(row_text):
        return "results"
    if METHOD_SIGNAL_RE.search(row_text):
        return "methods"
    if re.search(r"\b(characteristics|baseline|demographics?|sample|patients?|controls?|methods?)\b", combined, re.IGNORECASE):
        return "methods"
    if RESULT_SIGNAL_RE.search(combined) or STAT_SIGNAL_RE.search(combined):
        return "results"
    return "unknown"


def _table_row_category(row: str, *, section_label: str) -> str:
    if section_label == "methods":
        return "table_method_detail"
    if STAT_SIGNAL_RE.search(row) or RESULT_SIGNAL_RE.search(row):
        return "stats"
    return "table_extractive_summary"


def _normalize_llm_payload(raw: Any) -> dict[str, list[dict[str, Any]]]:
    if isinstance(raw, list):
        return {
            "evidence_packets": [item for item in raw if isinstance(item, dict)],
            "findings": [],
            "results": [],
        }
    if not isinstance(raw, dict):
        return {"evidence_packets": [], "findings": [], "results": []}
    return {
        "evidence_packets": [item for item in raw.get("evidence_packets", []) if isinstance(item, dict)],
        "findings": [item for item in raw.get("findings", []) if isinstance(item, dict)],
        "results": [item for item in raw.get("results", []) if isinstance(item, dict)],
    }
