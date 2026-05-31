from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from app.core.config import settings
from app.services.analysis.image_source import resolve_image_path
from app.services.analysis.llm import chat_text_fast, chat_with_images
from app.services.analysis.media_cleaning import clean_figure_caption, figure_downstream_text
from app.services.analysis.openai_usage import OpenAIBudgetExceeded
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
        "caption_only_calls": 0,
        "caption_only_success": 0,
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

            caption = clean_figure_caption(meta_obj.get("caption"))
            ocr_text = meta_obj.get("ocr_text")
            downstream_text, downstream_source = figure_downstream_text(caption=caption, ocr_text=ocr_text)
            diagnostics.setdefault("downstream_text_sources", {})
            diagnostics["downstream_text_sources"][downstream_source] = (
                int(diagnostics["downstream_text_sources"].get(downstream_source, 0) or 0) + 1
            )
            if _is_page_raster_fallback(meta_obj):
                skipped["page_raster_fallback"] += 1
                continue
            image_path, source_kind, skip_reason = resolve_image_path(meta_obj, cache_dir, remote_cache)
            if image_path and source_kind:
                source_counts[source_kind] += 1
            if not image_path:
                skipped[skip_reason or "missing_image_source"] += 1
                if caption and not _is_generic_caption(caption):
                    diagnostics["caption_only_calls"] += 1
                    caption_prompt = (
                        "Image input is unavailable, but a figure legend/caption was extracted. "
                        "Use only this caption to summarize what the figure contributes. "
                        "Do not repeat OCR artifacts, page headers, URLs, or malformed tokens. "
                        "Write a concise, figure-specific result or quality note.\n"
                        f"Anchor: {anchor}\nCaption: {caption}"
                    )
                    try:
                        response = chat_text_fast(caption_prompt, system=FIGURE_ANALYSIS_SYSTEM)
                        diagnostics["caption_only_success"] += 1
                    except Exception as exc:
                        if isinstance(exc, OpenAIBudgetExceeded):
                            raise
                        continue
                elif not ocr_text:
                    continue
                else:
                    diagnostics["ocr_fallback_calls"] += 1
                    ocr_prompt = (
                        "Image input is unavailable. Use only OCR text and caption to infer figure content. "
                        "Ignore page headers, URLs, malformed tokens, and obvious OCR artifacts. "
                        f"Anchor: {anchor}\nCaption/OCR Evidence: {downstream_text}"
                    )
                    try:
                        response = chat_text_fast(ocr_prompt, system=FIGURE_ANALYSIS_SYSTEM)
                        diagnostics["ocr_fallback_success"] += 1
                    except Exception as exc:
                        if isinstance(exc, OpenAIBudgetExceeded):
                            raise
                        continue
            else:
                prompt = (
                    "Analyze this figure. Extract key quantitative or qualitative results. "
                    "Check if axes/legends are clear and if the caption matches the visual content. "
                    "Ignore page headers, URLs, malformed tokens, and obvious OCR artifacts. "
                    f"Anchor: {anchor}\nCaption: {caption or 'N/A'}"
                )
                if not ocr_text and image_path:
                    ocr_text = _safe_ocr_text(image_path)
                    downstream_text, downstream_source = figure_downstream_text(caption=caption, ocr_text=ocr_text)
                if downstream_text and downstream_source != "caption":
                    prompt += f"\nCaption/OCR Evidence: {downstream_text}"

                diagnostics["vision_calls"] += 1
                try:
                    response = chat_with_images(prompt, [image_path], system=FIGURE_ANALYSIS_SYSTEM)
                    diagnostics["vision_success"] += 1
                except Exception as exc:
                    if isinstance(exc, OpenAIBudgetExceeded):
                        raise
                    diagnostics["vision_failures"] += 1
                    if not ocr_text and image_path:
                        ocr_text = _safe_ocr_text(image_path)
                    if ocr_text:
                        diagnostics["ocr_fallback_calls"] += 1
                        ocr_prompt = (
                            "Image analysis failed. Use only the OCR text and caption to infer the figure content. "
                            f"Anchor: {anchor}\nCaption/OCR Evidence: {downstream_text or ocr_text}"
                        )
                        try:
                            response = chat_text_fast(ocr_prompt, system=FIGURE_ANALYSIS_SYSTEM)
                            diagnostics["ocr_fallback_success"] += 1
                        except Exception as fallback_exc:
                            if isinstance(fallback_exc, OpenAIBudgetExceeded):
                                raise
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


def _is_page_raster_fallback(meta_obj: dict[str, Any]) -> bool:
    source = str(meta_obj.get("source") or "").strip().lower()
    fig_type = str(meta_obj.get("fig_type") or "").strip().lower()
    extra = meta_obj.get("extra")
    extra_source = ""
    if isinstance(extra, dict):
        extra_source = str(extra.get("source") or "").strip().lower()
    return source == "page_raster_fallback" or extra_source == "page_raster_fallback" or fig_type == "page"


def _is_generic_caption(value: Any) -> bool:
    text = " ".join(str(value or "").split()).strip()
    if not text:
        return True
    return bool(re.fullmatch(r"(?:fig(?:ure)?\.?\s*)?[A-Za-z]?\d+[A-Za-z]?[.:]?", text, flags=re.IGNORECASE))


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
