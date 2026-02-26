from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from app.core.config import settings
from app.services.analysis.image_source import resolve_image_path
from app.services.analysis.llm import chat_text_fast, chat_with_images
from app.services.analysis.ocr import ocr_image_text
from app.services.analysis.prompts import FIGURE_ANALYSIS_SYSTEM
from app.services.analysis.utils import (
    extract_json,
    normalize_evidence_packets,
    packets_to_legacy_findings,
)


def analyze_figures(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    max_figures = settings.analysis_max_figures
    valid_anchors = {str(chunk.get("anchor", "unknown")) for chunk in chunks}
    raw_packets: list[dict[str, Any]] = []

    skipped: defaultdict[str, int] = defaultdict(int)
    source_counts: defaultdict[str, int] = defaultdict(int)
    diagnostics: dict[str, Any] = {
        "chunks_considered": 0,
        "vision_calls": 0,
        "vision_success": 0,
        "vision_failures": 0,
        "vision_input_sources": {},
        "vision_skipped": {},
        "ocr_fallback_calls": 0,
        "ocr_fallback_success": 0,
    }

    with TemporaryDirectory(prefix="paper_eval_fig_") as cache_tmp:
        cache_dir = Path(cache_tmp)
        remote_cache: dict[str, str] = {}

        for chunk in chunks[:max_figures]:
            diagnostics["chunks_considered"] += 1
            anchor = chunk.get("anchor", "unknown")
            meta = chunk.get("meta")
            if not meta:
                skipped["missing_meta"] += 1
                continue
            try:
                meta_obj = json.loads(meta)
            except Exception:
                skipped["invalid_meta_json"] += 1
                continue
            document_source_url = str(chunk.get("document_source_url") or "").strip()
            if document_source_url:
                meta_obj.setdefault("document_source_url", document_source_url)

            caption = meta_obj.get("caption")
            ocr_text = meta_obj.get("ocr_text")
            image_path, source_kind, skip_reason = resolve_image_path(meta_obj, cache_dir, remote_cache)
            if image_path and source_kind:
                source_counts[source_kind] += 1
            if not image_path:
                skipped[skip_reason or "missing_image_source"] += 1
                if not ocr_text:
                    continue
                ocr_text = str(ocr_text)
                diagnostics["ocr_fallback_calls"] += 1
                ocr_prompt = (
                    "Image input is unavailable. Use only OCR text and caption to infer figure content. "
                    f"Anchor: {anchor}\nCaption: {caption or 'N/A'}\nOCR Text: {ocr_text}"
                )
                try:
                    response = chat_text_fast(ocr_prompt, system=FIGURE_ANALYSIS_SYSTEM)
                    diagnostics["ocr_fallback_success"] += 1
                except Exception:
                    continue
            else:
                prompt = (
                    "Analyze this figure. Extract key quantitative or qualitative results. "
                    "Check if axes/legends are clear and if the caption matches the visual content. "
                    f"Anchor: {anchor}\nCaption: {caption or 'N/A'}"
                )
                if not ocr_text and image_path:
                    ocr_text = _safe_ocr_text(image_path)
                if ocr_text:
                    prompt += f"\nOCR Text: {ocr_text}"

                diagnostics["vision_calls"] += 1
                try:
                    response = chat_with_images(prompt, [image_path], system=FIGURE_ANALYSIS_SYSTEM)
                    diagnostics["vision_success"] += 1
                except Exception:
                    diagnostics["vision_failures"] += 1
                    if not ocr_text and image_path:
                        ocr_text = _safe_ocr_text(image_path)
                    if ocr_text:
                        diagnostics["ocr_fallback_calls"] += 1
                        ocr_prompt = (
                            "Image analysis failed. Use only the OCR text and caption to infer the figure content. "
                            f"Anchor: {anchor}\nCaption: {caption or 'N/A'}\nOCR Text: {ocr_text}"
                        )
                        try:
                            response = chat_text_fast(ocr_prompt, system=FIGURE_ANALYSIS_SYSTEM)
                            diagnostics["ocr_fallback_success"] += 1
                        except Exception:
                            continue
                    else:
                        continue

            data = _normalize_llm_payload(extract_json(response))
            if not data["evidence_packets"] and not data["findings"] and not data["results"]:
                continue
            raw_packets.extend(data.get("evidence_packets", []))
            for finding in data.get("findings", []):
                raw_packets.append(
                    {
                        "finding_id": finding.get("finding_id"),
                        "anchor": anchor,
                        "statement": finding.get("summary", ""),
                        "evidence_refs": finding.get("evidence", []) or [anchor],
                        "confidence": finding.get("confidence", 0.0),
                        "category": finding.get("category", "figure_quality"),
                    }
                )
            for result in data.get("results", []):
                raw_packets.append(
                    {
                        "finding_id": result.get("finding_id"),
                        "anchor": anchor,
                        "statement": result.get("result", ""),
                        "evidence_refs": result.get("evidence", []) or [anchor],
                        "confidence": result.get("confidence", 0.0),
                        "category": "stats",
                        "value": result.get("value"),
                        "unit": result.get("unit"),
                        "p_value": result.get("p_value"),
                        "effect_size": result.get("effect_size"),
                    }
                )

    diagnostics["vision_input_sources"] = dict(source_counts)
    diagnostics["vision_skipped"] = dict(skipped)

    evidence_packets = normalize_evidence_packets(
        raw_packets,
        "figure",
        valid_anchors,
        default_category="figure_quality",
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
        "diagnostics": diagnostics,
    }


def _safe_ocr_text(image_path: str | Path | None) -> str:
    if not image_path:
        return ""
    try:
        return ocr_image_text(image_path, max_chars=settings.figure_ocr_max_chars)
    except Exception:
        return ""


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
