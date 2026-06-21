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
from app.services.analysis.media_cleaning import clean_figure_caption, figure_downstream_text, usable_caption
from app.services.analysis.openai_usage import OpenAIBudgetExceeded
from app.services.analysis.ocr import ocr_image_text
from app.services.analysis.prompts import FIGURE_ANALYSIS_SYSTEM
from app.services.analysis.utils import (
    add_source_excerpts_to_packets,
    extract_json,
    filter_grounded_evidence_packets,
    max_chars_for_ctx,
    normalize_evidence_packets,
    packets_to_legacy_findings,
)


def _figure_prompt_text(
    instruction: str,
    *,
    anchor: Any,
    caption: Any = "",
    downstream_text: Any = "",
    downstream_label: str = "Caption/OCR Evidence",
    max_chars: int,
) -> str:
    prefix = " ".join(str(instruction or "").split()).strip()
    if not prefix:
        prefix = "Analyze this figure using only the provided visual and text evidence."
    lines = [prefix, f"Anchor: {anchor}"]
    caption_text = " ".join(str(caption or "").split()).strip()
    downstream = " ".join(str(downstream_text or "").split()).strip()
    if caption_text:
        lines.append(f"Caption: {caption_text}")
    if downstream:
        lines.append(f"{downstream_label}: {downstream}")
    prompt = "\n".join(lines)
    if len(prompt) <= max_chars:
        return prompt

    fixed_lines = lines[:2]
    fixed_len = len("\n".join(fixed_lines))
    remaining = max(0, max_chars - fixed_len - 2)
    evidence_parts: list[str] = []
    if caption_text:
        evidence_parts.append(f"Caption: {caption_text}")
    if downstream:
        evidence_parts.append(f"{downstream_label}: {downstream}")
    compact_evidence = _compact_visual_evidence("\n".join(evidence_parts), max_chars=remaining)
    if compact_evidence:
        fixed_lines.append(compact_evidence)
    prompt = "\n".join(fixed_lines)
    if len(prompt) > max_chars:
        prompt = prompt[: max(0, max_chars - 3)].rstrip() + "..."
    return prompt


def _compact_visual_evidence(text: str, *, max_chars: int) -> str:
    clean = " ".join(str(text or "").split()).strip()
    if not clean or max_chars <= 0:
        return ""
    if len(clean) <= max_chars:
        return clean
    window = clean[: max_chars - 3]
    clipped_at_boundary = False
    for marker in (". ", "; ", ": ", ", "):
        idx = window.rfind(marker)
        if idx >= int(max_chars * 0.55):
            window = window[: idx + (1 if marker.startswith(".") else 0)]
            clipped_at_boundary = True
            break
    if not clipped_at_boundary and " " in window:
        window = window.rsplit(" ", 1)[0]
    return window.rstrip(" ,;:") + "..."


def analyze_figures(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    max_figures = settings.analysis_max_figures
    max_prompt_chars = max_chars_for_ctx(settings.llm_n_ctx)
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
        "downstream_text_sources": {},
        "downstream_text_source_by_anchor": {},
        "caption_only_calls": 0,
        "caption_only_success": 0,
        "ocr_fallback_calls": 0,
        "ocr_fallback_success": 0,
        "caption_first_skipped_vision": 0,
        "caption_first_skip_anchors": [],
        "extractive_fallback_packets": 0,
        "local_evidence_first": bool(settings.analysis_local_evidence_first_active),
        "local_evidence_first_packets": 0,
        "prompt_chars": [],
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
            if _is_page_raster_fallback(meta_obj):
                skipped["page_raster_fallback"] += 1
                continue
            if settings.analysis_local_evidence_first_active:
                _record_downstream_source(diagnostics, anchor, downstream_source)
                caption_first = _use_local_caption_first(caption)
                fallback_packet = _extractive_figure_packet(
                    anchor=anchor,
                    caption=caption,
                    downstream_text=downstream_text,
                    ocr_text=ocr_text,
                    reason="local_caption_first_skipped_vision" if caption_first else "local_evidence_first",
                    index=diagnostics["extractive_fallback_packets"] + 1,
                )
                if fallback_packet:
                    flags = fallback_packet.setdefault("quality_flags", [])
                    if isinstance(flags, list) and "local_evidence_first" not in flags:
                        flags.append("local_evidence_first")
                    raw_packets.append(fallback_packet)
                    diagnostics["extractive_fallback_packets"] += 1
                    diagnostics["local_evidence_first_packets"] += 1
                    if caption_first:
                        diagnostics["caption_first_skipped_vision"] += 1
                        diagnostics["caption_first_skip_anchors"].append(str(anchor))
                else:
                    skipped["local_evidence_first_no_usable_text"] += 1
                continue
            image_path, source_kind, skip_reason = resolve_image_path(meta_obj, cache_dir, remote_cache)
            if image_path and source_kind:
                source_counts[source_kind] += 1
            if not image_path:
                _record_downstream_source(diagnostics, anchor, downstream_source)
                skipped[skip_reason or "missing_image_source"] += 1
                if caption and not _is_generic_caption(caption):
                    diagnostics["caption_only_calls"] += 1
                    caption_prompt = _figure_prompt_text(
                        "Image input is unavailable, but a figure legend/caption was extracted. "
                        "Use only this caption to summarize what the figure contributes. "
                        "Do not repeat OCR artifacts, page headers, URLs, or malformed tokens. "
                        "Write a concise, figure-specific result or quality note. Preserve panel labels, "
                        "legend-defined groups, medication/intervention names, doses, biomarkers, assays, "
                        "model systems, effect directions, and statistics when stated.",
                        anchor=anchor,
                        caption=caption,
                        max_chars=max_prompt_chars,
                    )
                    diagnostics["prompt_chars"].append(len(caption_prompt))
                    try:
                        response = chat_text_fast(caption_prompt, system=FIGURE_ANALYSIS_SYSTEM)
                        diagnostics["caption_only_success"] += 1
                    except Exception as exc:
                        if isinstance(exc, OpenAIBudgetExceeded):
                            raise
                        fallback_packet = _extractive_figure_packet(
                            anchor=anchor,
                            caption=caption,
                            downstream_text=downstream_text,
                            ocr_text=ocr_text,
                            reason="caption_llm_failed",
                            index=diagnostics["extractive_fallback_packets"] + 1,
                        )
                        if fallback_packet:
                            raw_packets.append(fallback_packet)
                            diagnostics["extractive_fallback_packets"] += 1
                        continue
                elif not ocr_text:
                    continue
                else:
                    diagnostics["ocr_fallback_calls"] += 1
                    ocr_prompt = _figure_prompt_text(
                        "Image input is unavailable. Use only OCR text and caption to infer figure content. "
                        "Ignore page headers, URLs, malformed tokens, and obvious OCR artifacts. "
                        "Preserve legend-defined groups, medication/intervention names, doses, biomarkers, "
                        "assays, model systems, effect directions, and statistics when stated.",
                        anchor=anchor,
                        downstream_text=downstream_text,
                        max_chars=max_prompt_chars,
                    )
                    diagnostics["prompt_chars"].append(len(ocr_prompt))
                    try:
                        response = chat_text_fast(ocr_prompt, system=FIGURE_ANALYSIS_SYSTEM)
                        diagnostics["ocr_fallback_success"] += 1
                    except Exception as exc:
                        if isinstance(exc, OpenAIBudgetExceeded):
                            raise
                        fallback_packet = _extractive_figure_packet(
                            anchor=anchor,
                            caption=caption,
                            downstream_text=downstream_text,
                            ocr_text=ocr_text,
                            reason="ocr_llm_failed",
                            index=diagnostics["extractive_fallback_packets"] + 1,
                        )
                        if fallback_packet:
                            raw_packets.append(fallback_packet)
                            diagnostics["extractive_fallback_packets"] += 1
                        continue
            else:
                if _use_local_caption_first(caption):
                    _record_downstream_source(diagnostics, anchor, downstream_source)
                    fallback_packet = _extractive_figure_packet(
                        anchor=anchor,
                        caption=caption,
                        downstream_text=downstream_text,
                        ocr_text=ocr_text,
                        reason="local_caption_first_skipped_vision",
                        index=diagnostics["extractive_fallback_packets"] + 1,
                    )
                    if fallback_packet:
                        raw_packets.append(fallback_packet)
                        diagnostics["extractive_fallback_packets"] += 1
                        diagnostics["caption_first_skipped_vision"] += 1
                        diagnostics["caption_first_skip_anchors"].append(str(anchor))
                        continue

                prompt = _figure_prompt_text(
                    "Analyze this figure. Extract key quantitative or qualitative results. "
                    "Check if axes/legends are clear and if the caption matches the visual content. "
                    "Ignore page headers, URLs, malformed tokens, and obvious OCR artifacts. "
                    "Preserve panel labels, legend-defined groups, medication/intervention names, doses, "
                    "biomarkers, assays, model systems, effect directions, and statistics when stated.",
                    anchor=anchor,
                    caption=caption or "N/A",
                    max_chars=max_prompt_chars,
                )
                if not ocr_text and image_path:
                    ocr_text = _safe_ocr_text(image_path)
                    downstream_text, downstream_source = figure_downstream_text(caption=caption, ocr_text=ocr_text)
                _record_downstream_source(diagnostics, anchor, downstream_source)
                if downstream_text and downstream_source != "caption":
                    prompt = _figure_prompt_text(
                        "Analyze this figure. Extract key quantitative or qualitative results. "
                        "Check if axes/legends are clear and if the caption matches the visual content. "
                        "Ignore page headers, URLs, malformed tokens, and obvious OCR artifacts. "
                        "Preserve panel labels, legend-defined groups, medication/intervention names, doses, "
                        "biomarkers, assays, model systems, effect directions, and statistics when stated.",
                        anchor=anchor,
                        caption=caption or "N/A",
                        downstream_text=downstream_text,
                        max_chars=max_prompt_chars,
                    )
                diagnostics["prompt_chars"].append(len(prompt))

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
                        downstream_text, downstream_source = figure_downstream_text(caption=caption, ocr_text=ocr_text)
                    if ocr_text:
                        diagnostics["ocr_fallback_calls"] += 1
                        ocr_prompt = _figure_prompt_text(
                            "Image analysis failed. Use only the OCR text and caption to infer the figure content. "
                            "Ignore page headers, URLs, malformed tokens, and obvious OCR artifacts.",
                            anchor=anchor,
                            downstream_text=downstream_text or ocr_text,
                            max_chars=max_prompt_chars,
                        )
                        diagnostics["prompt_chars"].append(len(ocr_prompt))
                        try:
                            response = chat_text_fast(ocr_prompt, system=FIGURE_ANALYSIS_SYSTEM)
                            diagnostics["ocr_fallback_success"] += 1
                        except Exception as fallback_exc:
                            if isinstance(fallback_exc, OpenAIBudgetExceeded):
                                raise
                            fallback_packet = _extractive_figure_packet(
                                anchor=anchor,
                                caption=caption,
                                downstream_text=downstream_text,
                                ocr_text=ocr_text,
                                reason="vision_and_ocr_llm_failed",
                                index=diagnostics["extractive_fallback_packets"] + 1,
                            )
                            if fallback_packet:
                                raw_packets.append(fallback_packet)
                                diagnostics["extractive_fallback_packets"] += 1
                            continue
                    else:
                        fallback_packet = _extractive_figure_packet(
                            anchor=anchor,
                            caption=caption,
                            downstream_text=downstream_text,
                            ocr_text=ocr_text,
                            reason="vision_llm_failed",
                            index=diagnostics["extractive_fallback_packets"] + 1,
                        )
                        if fallback_packet:
                            raw_packets.append(fallback_packet)
                            diagnostics["extractive_fallback_packets"] += 1
                        continue

            data = _normalize_llm_payload(extract_json(response))
            if not data["evidence_packets"] and not data["findings"] and not data["results"]:
                fallback_packet = _extractive_figure_packet(
                    anchor=anchor,
                    caption=caption,
                    downstream_text=downstream_text,
                    ocr_text=ocr_text,
                    reason="llm_empty_output",
                    index=diagnostics["extractive_fallback_packets"] + 1,
                )
                if fallback_packet:
                    raw_packets.append(fallback_packet)
                    diagnostics["extractive_fallback_packets"] += 1
                continue
            source_excerpt = downstream_text or caption or ocr_text or ""
            raw_packets.extend(add_source_excerpts_to_packets(data.get("evidence_packets", []), {str(anchor): source_excerpt}))
            for finding in data.get("findings", []):
                raw_packets.append(
                    {
                        "finding_id": finding.get("finding_id"),
                        "anchor": anchor,
                        "statement": finding.get("summary", ""),
                        "evidence_refs": finding.get("evidence", []) or [anchor],
                        "confidence": finding.get("confidence", 0.0),
                        "category": finding.get("category", "figure_quality"),
                        "source_excerpt": source_excerpt,
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
                        "source_excerpt": source_excerpt,
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
    grounded_packets = filter_grounded_evidence_packets(evidence_packets)
    diagnostics["dropped_ungrounded_packets"] = len(evidence_packets) - len(grounded_packets)
    evidence_packets = grounded_packets
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


def _use_local_caption_first(caption: Any) -> bool:
    if settings.llm_provider_normalized != "local":
        return False
    if not bool(settings.analysis_local_figure_caption_first_enabled):
        return False
    try:
        min_chars = max(40, int(settings.analysis_local_figure_caption_first_min_chars or 80))
    except Exception:
        min_chars = 80
    return usable_caption(caption, min_chars=min_chars)


def _record_downstream_source(diagnostics: dict[str, Any], anchor: Any, source: str) -> None:
    diagnostics["downstream_text_sources"][source] = (
        int(diagnostics["downstream_text_sources"].get(source, 0) or 0) + 1
    )
    diagnostics["downstream_text_source_by_anchor"][str(anchor)] = source


def _extractive_figure_packet(
    *,
    anchor: Any,
    caption: Any,
    downstream_text: Any,
    ocr_text: Any,
    reason: str,
    index: int,
) -> dict[str, Any] | None:
    excerpt = _best_figure_excerpt(caption=caption, downstream_text=downstream_text, ocr_text=ocr_text)
    if not excerpt or _is_generic_caption(excerpt):
        return None
    statement = _extractive_figure_statement(excerpt)
    if not statement:
        return None
    return {
        "finding_id": f"figure-extractive-{index}",
        "anchor": str(anchor or "unknown"),
        "statement": statement,
        "evidence_refs": [str(anchor or "unknown")],
        "confidence": 0.45,
        "category": "figure_extractive_summary",
        "source_excerpt": excerpt,
        "quality_flags": ["extractive_fallback", reason],
    }


def _best_figure_excerpt(*, caption: Any, downstream_text: Any, ocr_text: Any) -> str:
    for value in (caption, downstream_text, ocr_text):
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        if text and not _is_generic_caption(text):
            return text
    return ""


def _extractive_figure_statement(excerpt: str) -> str:
    text = re.sub(r"\s+", " ", str(excerpt or "")).strip()
    if not text:
        return ""
    if len(text) > 360:
        text = text[:357].rsplit(" ", 1)[0].rstrip(" ,;:") + "..."
    return f"Extracted figure legend/OCR content available for synthesis: {text}"


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
