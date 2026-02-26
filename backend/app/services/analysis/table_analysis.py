from __future__ import annotations

import json
from typing import Any

from app.core.config import settings
from app.services.analysis.llm import chat_text_fast
from app.services.analysis.prompts import TABLE_ANALYSIS_SYSTEM
from app.services.analysis.utils import (
    extract_json,
    max_chars_for_ctx,
    normalize_evidence_packets,
    packets_to_legacy_findings,
    truncate_text,
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


def analyze_tables(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    max_tables = settings.analysis_max_tables
    valid_anchors = {str(chunk.get("anchor", "unknown")) for chunk in chunks}
    blocks: list[str] = []
    for chunk in chunks[:max_tables]:
        anchor = chunk.get("anchor", "unknown")
        content = chunk.get("content", "")
        preview = _table_preview(content)
        blocks.append(f"[TABLE {anchor}]\n{preview}")

    if not blocks:
        return {"findings": [], "results": [], "evidence_packets": []}

    prompt = (
        "Analyze the tables for key results and issues. "
        "Look for sample sizes, effect sizes, p-values, subgroup results, and inconsistencies. "
        "Cite anchors.\n\n"
        + "\n\n".join(blocks)
    )
    prompt = truncate_text(prompt, max_chars_for_ctx(settings.llm_n_ctx))

    response = chat_text_fast(prompt, system=TABLE_ANALYSIS_SYSTEM)
    data = _normalize_llm_payload(extract_json(response))
    if not data["evidence_packets"] and not data["findings"] and not data["results"]:
        return {"findings": [], "results": [], "evidence_packets": []}

    raw_packets = list(data.get("evidence_packets", []))
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
            }
        )

    evidence_packets = normalize_evidence_packets(
        raw_packets,
        "table",
        valid_anchors,
        default_category="table_quality",
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
    return {"findings": findings, "results": results, "evidence_packets": evidence_packets}


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
