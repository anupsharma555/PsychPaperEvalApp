from __future__ import annotations

import json
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
from app.services.analysis.prompts import SUPP_ANALYSIS_SYSTEM
from app.services.analysis.utils import (
    add_source_excerpts_to_packets,
    extract_json,
    filter_grounded_evidence_packets,
    max_chars_for_ctx,
    normalize_evidence_packets,
    packets_to_legacy_findings,
)
from app.services.analysis.table_analysis import _table_preview


def _supplement_prompt_prefix() -> str:
    return (
        "Analyze supplementary materials for key results, data quality, and issues. "
        "Preserve medication/intervention definitions, dose/route/duration details, comparator arms, "
        "model systems, assay/readout definitions, adverse events, sensitivity analyses, subgroup results, "
        "effect directions, statistics, units, figure legends, and table legends when present. "
        "Cite anchors.\n\n"
    )


def _supplement_analysis_prompt(blocks: list[str], *, max_chars: int) -> str:
    prefix = _supplement_prompt_prefix()
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
            selected.append(_compact_supplement_block(block_text, max_chars=remaining))
        break
    if not selected and blocks:
        selected.append(_compact_supplement_block(str(blocks[0]), max_chars=max(240, max_chars - len(prefix))))
    return prefix + "\n\n".join(selected)


def _compact_supplement_block(block: str, *, max_chars: int) -> str:
    text = str(block or "").strip()
    if len(text) <= max_chars:
        return text
    lines = text.splitlines() or [text]
    anchor_line = lines[0].strip()
    if len(anchor_line) > max_chars - 3:
        return anchor_line[: max(0, max_chars - 3)].rstrip() + "..."
    kept = [anchor_line]
    current_len = len(anchor_line)
    for line in lines[1:]:
        cleaned = line.strip()
        if not cleaned:
            continue
        projected = current_len + 1 + len(cleaned)
        if projected <= max_chars - 3:
            kept.append(cleaned)
            current_len = projected
            continue
        remaining = max_chars - current_len - 4
        if remaining > 32:
            kept.append(cleaned[:remaining].rstrip())
        break
    compact = "\n".join(kept).strip()
    if len(compact) > max_chars:
        compact = compact[: max(0, max_chars - 3)].rstrip()
    return compact + "..."


def analyze_supplements(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    max_items = settings.analysis_max_supp_items
    valid_anchors = {str(chunk.get("anchor", "unknown")) for chunk in chunks}
    text_blocks: list[str] = []
    anchor_excerpts: dict[str, str] = {}
    fig_chunks: list[dict[str, Any]] = []
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
        "caption_first_skipped_vision": 0,
        "caption_first_skip_anchors": [],
        "ocr_fallback_calls": 0,
        "ocr_fallback_success": 0,
        "local_evidence_first": bool(settings.analysis_local_evidence_first_active),
        "local_evidence_first_packets": 0,
    }

    for chunk in chunks[:max_items]:
        modality = chunk.get("modality")
        anchor = chunk.get("anchor", "unknown")
        if modality == "text":
            text = chunk.get("content", "")
            excerpt = str(text)[:4000]
            anchor_excerpts[str(anchor)] = excerpt
            text_blocks.append(f"[SUPP {anchor}]\n{excerpt}")
        elif modality == "table":
            preview = _table_preview(chunk.get("content", ""))
            anchor_excerpts[str(anchor)] = preview
            text_blocks.append(f"[SUPP TABLE {anchor}]\n{preview}")
        elif modality == "figure":
            fig_chunks.append(chunk)

    raw_packets: list[dict[str, Any]] = []

    if text_blocks and settings.analysis_local_evidence_first_active:
        packets = _extractive_supplement_text_packets(
            anchor_excerpts,
            start_index=1,
            reason="local_evidence_first",
        )
        raw_packets.extend(packets)
        diagnostics["text_prompt_blocks"] = len(text_blocks)
        diagnostics["text_prompt_chars"] = 0
        diagnostics["local_evidence_first_packets"] += len(packets)
    elif text_blocks:
        prompt = _supplement_analysis_prompt(text_blocks, max_chars=max_chars_for_ctx(settings.llm_n_ctx))
        diagnostics["text_prompt_blocks"] = len(text_blocks)
        diagnostics["text_prompt_chars"] = len(prompt)
        response = chat_text_fast(prompt, system=SUPP_ANALYSIS_SYSTEM)
        data = _normalize_llm_payload(extract_json(response))
        raw_packets.extend(add_source_excerpts_to_packets(data.get("evidence_packets", []), anchor_excerpts))
        for finding in data.get("findings", []):
            evidence = finding.get("evidence", [])
            anchor = evidence[0] if isinstance(evidence, list) and evidence else ""
            raw_packets.append(
                {
                    "finding_id": finding.get("finding_id"),
                    "anchor": anchor,
                    "statement": finding.get("summary", ""),
                    "evidence_refs": evidence,
                    "confidence": finding.get("confidence", 0.0),
                    "category": finding.get("category", "supplement_quality"),
                    "source_excerpt": anchor_excerpts.get(str(anchor), ""),
                }
            )
        for result in data.get("results", []):
            evidence = result.get("evidence", [])
            anchor = evidence[0] if isinstance(evidence, list) and evidence else ""
            raw_packets.append(
                {
                    "finding_id": result.get("finding_id"),
                    "anchor": anchor,
                    "statement": result.get("result", ""),
                    "evidence_refs": evidence,
                    "confidence": result.get("confidence", 0.0),
                    "category": "stats",
                    "source_excerpt": anchor_excerpts.get(str(anchor), ""),
                }
            )

    with TemporaryDirectory(prefix="paper_eval_supp_fig_") as cache_tmp:
        cache_dir = Path(cache_tmp)
        remote_cache: dict[str, str] = {}

        for chunk in fig_chunks:
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
            if settings.analysis_local_evidence_first_active:
                _record_downstream_source(diagnostics, anchor, downstream_source)
                caption_first = _use_local_caption_first(caption)
                fallback_packet = _extractive_supplement_packet(
                    anchor=anchor,
                    caption=caption,
                    downstream_text=downstream_text,
                    ocr_text=ocr_text,
                    reason="local_caption_first_skipped_vision" if caption_first else "local_evidence_first",
                    index=diagnostics["local_evidence_first_packets"] + 1,
                )
                if fallback_packet:
                    flags = fallback_packet.setdefault("quality_flags", [])
                    if isinstance(flags, list) and "local_evidence_first" not in flags:
                        flags.append("local_evidence_first")
                    raw_packets.append(fallback_packet)
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
                if not downstream_text:
                    continue
                if downstream_source == "caption":
                    diagnostics["caption_only_calls"] += 1
                else:
                    diagnostics["ocr_fallback_calls"] += 1
                fallback_prompt = (
                    "Image input is unavailable. Use only the selected clean caption/OCR evidence to infer "
                    "supplementary figure content. Prioritize the caption when it contains a usable legend; "
                    "ignore page headers, URLs, malformed tokens, and obvious OCR artifacts.\n"
                    "Preserve panel labels, legend-defined groups, medication/intervention names, doses, "
                    "biomarkers, assays, model systems, effect directions, and statistics when stated.\n"
                    f"Anchor: {anchor}\nCaption/OCR Evidence: {downstream_text}"
                )
                try:
                    response = chat_text_fast(fallback_prompt, system=SUPP_ANALYSIS_SYSTEM)
                    if downstream_source == "caption":
                        diagnostics["caption_only_success"] += 1
                    else:
                        diagnostics["ocr_fallback_success"] += 1
                except Exception as exc:
                    if isinstance(exc, OpenAIBudgetExceeded):
                        raise
                    continue
            else:
                if _use_local_caption_first(caption):
                    _record_downstream_source(diagnostics, anchor, downstream_source)
                    fallback_packet = _extractive_supplement_packet(
                        anchor=anchor,
                        caption=caption,
                        downstream_text=downstream_text,
                        ocr_text=ocr_text,
                        reason="local_caption_first_skipped_vision",
                        index=int(diagnostics["caption_first_skipped_vision"]) + 1,
                    )
                    if fallback_packet:
                        raw_packets.append(fallback_packet)
                        diagnostics["caption_first_skipped_vision"] += 1
                        diagnostics["caption_first_skip_anchors"].append(str(anchor))
                        continue

                prompt = (
                    "Analyze this supplementary figure for key results or issues. "
                    "Ignore page headers, URLs, malformed tokens, and obvious OCR artifacts. "
                    "Preserve panel labels, legend-defined groups, medication/intervention names, doses, "
                    "biomarkers, assays, model systems, effect directions, and statistics when stated. "
                    f"Anchor: {anchor}\nCaption: {caption or 'N/A'}"
                )
                if not ocr_text and image_path:
                    ocr_text = _safe_ocr_text(image_path)
                    downstream_text, downstream_source = figure_downstream_text(caption=caption, ocr_text=ocr_text)
                _record_downstream_source(diagnostics, anchor, downstream_source)
                if downstream_text and downstream_source != "caption":
                    prompt += f"\nCaption/OCR Evidence: {downstream_text}"
                diagnostics["vision_calls"] += 1
                try:
                    response = chat_with_images(prompt, [image_path], system=SUPP_ANALYSIS_SYSTEM)
                    diagnostics["vision_success"] += 1
                except Exception as exc:
                    if isinstance(exc, OpenAIBudgetExceeded):
                        raise
                    diagnostics["vision_failures"] += 1
                    if not ocr_text and image_path:
                        ocr_text = _safe_ocr_text(image_path)
                        downstream_text, downstream_source = figure_downstream_text(caption=caption, ocr_text=ocr_text)
                    if not downstream_text:
                        continue
                    if downstream_source == "caption":
                        diagnostics["caption_only_calls"] += 1
                    else:
                        diagnostics["ocr_fallback_calls"] += 1
                    fallback_prompt = (
                        "Image analysis failed. Use only the selected clean caption/OCR evidence to infer "
                        "the supplementary figure content. Prioritize the caption when it contains a usable legend; "
                        "ignore page headers, URLs, malformed tokens, and obvious OCR artifacts.\n"
                        f"Anchor: {anchor}\nCaption/OCR Evidence: {downstream_text}"
                    )
                    try:
                        response = chat_text_fast(fallback_prompt, system=SUPP_ANALYSIS_SYSTEM)
                        if downstream_source == "caption":
                            diagnostics["caption_only_success"] += 1
                        else:
                            diagnostics["ocr_fallback_success"] += 1
                    except Exception as fallback_exc:
                        if isinstance(fallback_exc, OpenAIBudgetExceeded):
                            raise
                        continue

            data = _normalize_llm_payload(extract_json(response))
            source_excerpt = downstream_text or caption or str(ocr_text or "")
            raw_packets.extend(
                add_source_excerpts_to_packets(data.get("evidence_packets", []), {str(anchor): source_excerpt})
            )
            for finding in data.get("findings", []):
                raw_packets.append(
                    {
                        "finding_id": finding.get("finding_id"),
                        "anchor": anchor,
                        "statement": finding.get("summary", ""),
                        "evidence_refs": finding.get("evidence", []) or [anchor],
                        "confidence": finding.get("confidence", 0.0),
                        "category": finding.get("category", "supplement_quality"),
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
                        "source_excerpt": source_excerpt,
                    }
                )

    diagnostics["vision_input_sources"] = dict(source_counts)
    diagnostics["vision_skipped"] = dict(skipped)

    evidence_packets = normalize_evidence_packets(
        raw_packets,
        "supplement",
        valid_anchors,
        default_category="supplement_quality",
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


def _record_downstream_source(diagnostics: dict[str, Any], anchor: Any, source: str) -> None:
    diagnostics["downstream_text_sources"][source] = (
        int(diagnostics["downstream_text_sources"].get(source, 0) or 0) + 1
    )
    diagnostics["downstream_text_source_by_anchor"][str(anchor)] = source


def _use_local_caption_first(caption: Any) -> bool:
    if settings.llm_provider_normalized != "local":
        return False
    if not bool(settings.analysis_local_supplement_caption_first_enabled):
        return False
    try:
        min_chars = max(40, int(settings.analysis_local_supplement_caption_first_min_chars or 80))
    except Exception:
        min_chars = 80
    return usable_caption(caption, min_chars=min_chars)


def _extractive_supplement_packet(
    *,
    anchor: Any,
    caption: Any,
    downstream_text: Any,
    ocr_text: Any,
    reason: str,
    index: int,
) -> dict[str, Any] | None:
    excerpt = _best_supplement_excerpt(caption=caption, downstream_text=downstream_text, ocr_text=ocr_text)
    if not excerpt:
        return None
    statement = _extractive_supplement_statement(excerpt)
    if not statement:
        return None
    return {
        "finding_id": f"supplement-extractive-{index}",
        "anchor": str(anchor or "unknown"),
        "statement": statement,
        "evidence_refs": [str(anchor or "unknown")],
        "confidence": 0.45,
        "category": "supplement_extractive_summary",
        "source_excerpt": excerpt,
        "quality_flags": ["extractive_fallback", reason],
    }


def _best_supplement_excerpt(*, caption: Any, downstream_text: Any, ocr_text: Any) -> str:
    for value in (caption, downstream_text, ocr_text):
        text = " ".join(str(value or "").split()).strip()
        if text:
            return text
    return ""


def _extractive_supplement_statement(excerpt: str) -> str:
    text = " ".join(str(excerpt or "").split()).strip()
    if not text:
        return ""
    if len(text) > 360:
        text = text[:357].rsplit(" ", 1)[0].rstrip(" ,;:") + "..."
    return f"Extracted supplementary figure legend/OCR content available for synthesis: {text}"


def _extractive_supplement_text_packets(
    anchor_excerpts: dict[str, str],
    *,
    start_index: int,
    reason: str,
) -> list[dict[str, Any]]:
    packets: list[dict[str, Any]] = []
    for offset, (anchor, excerpt) in enumerate(anchor_excerpts.items(), start=start_index):
        statement = _extractive_supplement_text_statement(excerpt)
        if not statement:
            continue
        packets.append(
            {
                "finding_id": f"supplement-extractive-{offset}",
                "anchor": anchor,
                "statement": statement,
                "evidence_refs": [anchor],
                "confidence": 0.45,
                "category": "supplement_extractive_summary",
                "source_excerpt": excerpt,
                "quality_flags": ["extractive_fallback", reason],
            }
        )
    return packets


def _extractive_supplement_text_statement(excerpt: str) -> str:
    text = " ".join(str(excerpt or "").split()).strip()
    if not text:
        return ""
    if len(text) > 360:
        text = text[:357].rsplit(" ", 1)[0].rstrip(" ,;:") + "..."
    return f"Extracted supplementary text/table content available for synthesis: {text}"


def _safe_ocr_text(image_path: str | Path | None) -> str:
    if not image_path:
        return ""
    try:
        return ocr_image_text(image_path, max_chars=settings.figure_ocr_max_chars)
    except Exception:
        return ""
