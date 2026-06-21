#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from sqlmodel import Session, select  # noqa: E402

from app.db.models import Asset, Chunk, Document, Report  # noqa: E402
from app.db.session import engine  # noqa: E402
from app.services.analysis.information_retention import build_information_retention_audit  # noqa: E402
from app.services.analysis.media_cleaning import clean_figure_caption, clean_figure_ocr_text, figure_downstream_text  # noqa: E402
from app.services.analysis.section_ledger import apply_section_boundary_ledger_to_dicts  # noqa: E402
from app.services.analysis.utils import extract_expected_refs  # noqa: E402
from app.services.parser import _normalize_section_title  # noqa: E402
from app.services.storage import artifacts_dir  # noqa: E402


SECTION_KEYS = ["introduction", "methods", "results", "discussion", "conclusion", "unknown"]
EXPLICIT_SECTION_SOURCES = {"heading", "anchor", "structured_abstract", "meta"}
OCR_ARTIFACT_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]|(?:\b[a-z]{12,}\b)|(?:\d+\s+[46]\s+\d+)")
GENERIC_CAPTION_RE = re.compile(r"^\s*(?:fig(?:ure)?\.?\s*)?\d+[a-z]?\s*$", re.IGNORECASE)
UPSTREAM_AB_REGRESSION_THRESHOLDS = {
    "section_boundary_ledger_mean_wrong_section_rate_max": 0.24,
    "section_boundary_ledger_document_wrong_section_rate_max": 0.40,
    "clean_caption_first_mean_artifact_text_rate_max": 0.46,
}
MEDIA_RECALL_KEYS = (
    "figure_ref_recall",
    "table_ref_recall",
    "supplementary_figure_ref_recall",
    "supplementary_table_ref_recall",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Offline A/B comparison for upstream extraction pathways. "
            "Does not call OpenAI and does not mutate stored reports."
        )
    )
    parser.add_argument("--document-id", type=int, required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    with Session(engine) as session:
        document = session.get(Document, args.document_id)
        if document is None:
            raise SystemExit(f"Document {args.document_id} not found")
        assets = session.exec(select(Asset).where(Asset.document_id == args.document_id)).all()
        chunks = session.exec(select(Chunk).where(Chunk.document_id == args.document_id).order_by(Chunk.id)).all()
        report = session.exec(
            select(Report).where(Report.document_id == args.document_id).order_by(Report.id.desc())
        ).first()

    parsed_chunks = [_chunk_to_dict(chunk) for chunk in chunks]
    source_assets = [_asset_to_dict(asset) for asset in assets]
    summary_json = _load_report_payload(report)

    variants = {
        "baseline": parsed_chunks,
        "source_first_sections": _source_first_section_relabel(parsed_chunks),
        "heading_boundary_sections": _heading_boundary_section_relabel(parsed_chunks),
        "imrad_guarded_sections": _imrad_guarded_section_relabel(parsed_chunks),
        "section_boundary_ledger": _section_boundary_ledger_relabel(parsed_chunks),
    }
    variant_metrics = {
        name: _variant_metrics(
            document_id=args.document_id,
            source_assets=source_assets,
            parsed_chunks=variant_chunks,
        )
        for name, variant_chunks in variants.items()
    }

    media_metrics = {
        "current_caption_plus_ocr": _media_metrics(parsed_chunks, mode="caption_plus_ocr"),
        "caption_first": _media_metrics(parsed_chunks, mode="caption_first"),
        "clean_caption_first": _media_metrics(parsed_chunks, mode="clean_caption_first"),
    }

    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "document_id": args.document_id,
        "label": args.label or str(document.title or ""),
        "document": {
            "title": document.title,
            "source_url": document.source_url,
            "doi": document.doi,
        },
        "inputs": {
            "asset_count": len(source_assets),
            "chunk_count": len(parsed_chunks),
            "report_id": report.id if report else None,
            "has_summary": bool(summary_json),
        },
        "comparison": _comparison_delta(variant_metrics, media_metrics),
        "variants": variant_metrics,
        "media_variants": media_metrics,
    }

    out_dir = Path(args.out_dir).expanduser() if args.out_dir else artifacts_dir(args.document_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"upstream_ab_comparison_{stamp}.json"
    md_path = out_dir / f"upstream_ab_comparison_{stamp}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(payload), encoding="utf-8")

    print(f"document_id={args.document_id}")
    print(f"comparison_json={json_path}")
    print(f"comparison_md={md_path}")
    for name, metrics in variant_metrics.items():
        parsed = metrics.get("parsed_chunks", {})
        print(
            f"{name}: retained={parsed.get('retained_rate')} "
            f"wrong={parsed.get('wrong_section_count')} "
            f"wrong_rate={parsed.get('wrong_section_rate')}"
        )
    for name, metrics in media_metrics.items():
        print(
            f"{name}: usable_figures={metrics.get('usable_figure_rate')} "
            f"artifact_rate={metrics.get('artifact_text_rate')} "
            f"mean_chars={metrics.get('mean_downstream_text_chars')}"
        )


def _chunk_to_dict(chunk: Chunk) -> dict[str, Any]:
    return {
        "id": chunk.id,
        "document_id": chunk.document_id,
        "asset_id": chunk.asset_id,
        "anchor": chunk.anchor,
        "modality": chunk.modality,
        "content": chunk.content,
        "meta": chunk.meta,
    }


def _asset_to_dict(asset: Asset) -> dict[str, Any]:
    return {
        "id": asset.id,
        "document_id": asset.document_id,
        "kind": asset.kind,
        "filename": asset.filename,
        "content_type": asset.content_type,
        "path": asset.path,
    }


def _load_report_payload(report: Report | None) -> dict[str, Any]:
    if report is None:
        return {}
    try:
        parsed = json.loads(report.payload or "{}")
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _variant_metrics(
    *,
    document_id: int,
    source_assets: list[dict[str, Any]],
    parsed_chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    audit = build_information_retention_audit(
        document_id=document_id,
        source_assets=source_assets,
        parsed_chunks=parsed_chunks,
        summary_json={},
    )
    parsed_stage = next(
        (row for row in audit.get("stage_metrics", []) if row.get("stage") == "parsed_chunks"),
        {},
    )
    text_chunks = [row for row in parsed_chunks if str(row.get("modality") or "").lower() == "text"]
    return {
        "source_basis": audit.get("source_basis"),
        "source_sentence_count": audit.get("source_sentence_count"),
        "parsed_chunks": {
            "stage_item_count": parsed_stage.get("stage_item_count", 0),
            "retained_count": parsed_stage.get("retained_count", 0),
            "retained_rate": parsed_stage.get("retained_rate", 0.0),
            "wrong_section_count": parsed_stage.get("wrong_section_count", 0),
            "wrong_section_rate": parsed_stage.get("wrong_section_rate", 0.0),
        },
        "section_counts": dict(Counter(_chunk_section(row, idx, len(parsed_chunks)) for idx, row in enumerate(text_chunks))),
        "low_confidence_section_count": _low_confidence_section_count(text_chunks),
        "unknown_section_count": sum(1 for idx, row in enumerate(text_chunks) if _chunk_section(row, idx, len(text_chunks)) == "unknown"),
    }


def _source_first_section_relabel(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = deepcopy(chunks)
    text_indices = [idx for idx, row in enumerate(out) if str(row.get("modality") or "").lower() == "text"]
    active_section = "unknown"
    for order, idx in enumerate(text_indices):
        row = out[idx]
        meta = _parse_meta(row.get("meta"))
        baseline = _normalize_known_section(meta.get("section_norm") or meta.get("section_label") or meta.get("section"))
        source = str(meta.get("section_source") or "").strip().lower()
        confidence = _float(meta.get("section_confidence"), default=0.0)
        raw_title = str(meta.get("section_raw_title") or meta.get("raw_title") or "").strip()
        heading_section = _normalize_known_section(raw_title)
        content_heading = _leading_heading_section(str(row.get("content") or ""))

        if heading_section != "unknown":
            chosen = heading_section
            chosen_source = "raw_title"
            confidence = max(confidence, 0.92)
        elif content_heading != "unknown":
            chosen = content_heading
            chosen_source = "content_heading"
            confidence = max(confidence, 0.88)
        elif baseline != "unknown" and (source in EXPLICIT_SECTION_SOURCES or confidence >= 0.58):
            chosen = baseline
            chosen_source = source or "baseline_high_confidence"
        elif active_section != "unknown":
            chosen = active_section
            chosen_source = "section_continuity"
            confidence = max(confidence, 0.48)
        else:
            chosen = _position_section(order, len(text_indices))
            chosen_source = "position_seed"
            confidence = max(confidence, 0.35)

        active_section = chosen if chosen != "unknown" else active_section
        meta["section_norm"] = chosen
        meta["section_source"] = f"ab_source_first:{chosen_source}"
        meta["section_confidence"] = round(float(confidence), 3)
        row["meta"] = json.dumps(meta, ensure_ascii=True)
    return out


def _heading_boundary_section_relabel(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = deepcopy(chunks)
    text_indices = [idx for idx, row in enumerate(out) if str(row.get("modality") or "").lower() == "text"]
    active_section = "unknown"
    for order, idx in enumerate(text_indices):
        row = out[idx]
        meta = _parse_meta(row.get("meta"))
        baseline = _normalize_known_section(meta.get("section_norm") or meta.get("section_label") or meta.get("section"))
        raw_title = str(meta.get("section_raw_title") or meta.get("raw_title") or "").strip()
        heading_section = _normalize_known_section(raw_title)
        content_heading = _leading_heading_section(str(row.get("content") or ""))
        paragraph_index = _int(meta.get("paragraph_index"), default=order)

        if heading_section != "unknown":
            active_section = heading_section
            chosen = heading_section
            reason = "raw_title_boundary"
            confidence = 0.94
        elif content_heading != "unknown":
            active_section = content_heading
            chosen = content_heading
            reason = "content_heading_boundary"
            confidence = 0.9
        elif active_section != "unknown":
            chosen = active_section
            reason = "active_boundary"
            confidence = 0.62
        elif baseline != "unknown":
            chosen = baseline
            reason = "baseline_before_boundary"
            confidence = 0.5
        else:
            chosen = _position_section(paragraph_index, len(text_indices))
            reason = "position_before_boundary"
            confidence = 0.34

        meta["section_norm"] = chosen
        meta["section_source"] = f"ab_heading_boundary:{reason}"
        meta["section_confidence"] = confidence
        row["meta"] = json.dumps(meta, ensure_ascii=True)
    return out


def _imrad_guarded_section_relabel(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = deepcopy(chunks)
    text_indices = [idx for idx, row in enumerate(out) if str(row.get("modality") or "").lower() == "text"]
    order_map = {"introduction": 0, "methods": 1, "results": 2, "discussion": 3, "conclusion": 4}
    active_section = "unknown"
    active_rank = -1
    for order, idx in enumerate(text_indices):
        row = out[idx]
        meta = _parse_meta(row.get("meta"))
        baseline = _normalize_known_section(meta.get("section_norm") or meta.get("section_label") or meta.get("section"))
        raw_title = str(meta.get("section_raw_title") or meta.get("raw_title") or "").strip()
        candidate = _normalize_known_section(raw_title)
        reason = "continuity"
        confidence = 0.55
        if candidate == "unknown":
            candidate = _leading_heading_section(str(row.get("content") or ""))
            if candidate != "unknown":
                reason = "content_heading"
                confidence = 0.86
        else:
            reason = "raw_title"
            confidence = 0.92

        if candidate != "unknown":
            candidate_rank = order_map.get(candidate, active_rank)
            if candidate_rank >= active_rank or _allowed_backward_transition(active_section, candidate):
                active_section = candidate
                active_rank = max(active_rank, candidate_rank)
                chosen = candidate
            else:
                chosen = active_section if active_section != "unknown" else baseline
                reason = f"blocked_backward_{reason}"
                confidence = 0.46
        elif baseline != "unknown" and active_section == "unknown":
            chosen = baseline
            active_section = baseline
            active_rank = order_map.get(baseline, active_rank)
            reason = "baseline_seed"
            confidence = 0.48
        elif active_section != "unknown":
            chosen = active_section
        else:
            chosen = _position_section(order, len(text_indices))
            reason = "position_seed"
            confidence = 0.34

        meta["section_norm"] = chosen or "unknown"
        meta["section_source"] = f"ab_imrad_guarded:{reason}"
        meta["section_confidence"] = confidence
        row["meta"] = json.dumps(meta, ensure_ascii=True)
    return out


def _section_boundary_ledger_relabel(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return apply_section_boundary_ledger_to_dicts(chunks)
    out = deepcopy(chunks)
    text_indices = [idx for idx, row in enumerate(out) if str(row.get("modality") or "").lower() == "text"]
    blocks = _ledger_blocks(out, text_indices)
    ledger = _build_section_boundary_ledger(blocks)
    for entry in ledger:
        for idx in entry["indices"]:
            row = out[idx]
            meta = _parse_meta(row.get("meta"))
            meta["section_norm"] = entry["section"]
            meta["section_source"] = f"ab_section_boundary_ledger:{entry['reason']}"
            meta["section_confidence"] = entry["confidence"]
            meta["section_ledger_title"] = entry["title"]
            row["meta"] = json.dumps(meta, ensure_ascii=True)
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
                "sources": [],
            }
            blocks.append(current)
        current["indices"].append(idx)
        current["texts"].append(str(row.get("content") or ""))
        current["baseline_sections"].append(
            _normalize_known_section(meta.get("section_norm") or meta.get("section_label") or meta.get("section"))
        )
        current["sources"].append(str(meta.get("source") or ""))
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
    if not text or lowered in {"body", "file"} or lowered.startswith("figure") or lowered.startswith("table"):
        return "unknown", "title_uninformative", 0.0
    direct = _normalize_known_section(text)
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
    conclusion_patterns = ("conclusion", "conclusions", "summary")
    if any(token in lowered for token in conclusion_patterns):
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
    prefix = re.match(
        r"^\s*(objective|objectives|background|aim|aims|purpose|hypothesis|method|methods|design|results|conclusion|conclusions)\s*:",
        clean,
    )
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
    if re.search(r"\b(participants?|sample|administered|acquired|preprocess|covariate|regression|protocol|cells? were|plasmid|pcr|qpcr|transfected|selection|assay|lc-ms)\b", clean):
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


def _clean_title(value: Any) -> str:
    text = " ".join(str(value or "").replace("_", " ").split()).strip()
    text = re.sub(r"^\d+(?:\.\d+)*\s*", "", text)
    text = text.strip(" -:.;)")
    return text or "body"


def _leading_heading_text(text: str) -> str:
    first_line = str(text or "").replace("\r", "\n").split("\n", 1)[0].strip()
    if len(first_line) > 120:
        return ""
    if re.search(r"[.!?]$", first_line) and not re.match(r"^(objective|method|results?|conclusions?)\s*:", first_line, re.IGNORECASE):
        return ""
    return first_line


def _is_abstract_title(title: str) -> bool:
    return _clean_title(title).lower() == "abstract"


def _media_metrics(chunks: list[dict[str, Any]], *, mode: str) -> dict[str, Any]:
    text_chunks = [str(row.get("content") or "") for row in chunks if str(row.get("modality") or "").lower() == "text"]
    expected = extract_expected_refs(text_chunks)
    figure_refs_expected = set(expected.get("figure_refs", []))
    table_refs_expected = set(expected.get("table_refs", []))
    figures = [row for row in chunks if str(row.get("modality") or "").lower() == "figure"]
    tables = [row for row in chunks if str(row.get("modality") or "").lower() == "table"]

    figure_ids: set[str] = set()
    downstream_texts: list[str] = []
    usable = 0
    captions = 0
    artifact_rows = 0
    for fig in figures:
        meta = _parse_meta(fig.get("meta"))
        raw_caption = meta.get("caption") or fig.get("content") or ""
        raw_ocr_text = meta.get("ocr_text") or ""
        caption = _clean_text(raw_caption)
        ocr_text = _clean_text(raw_ocr_text)
        if mode == "clean_caption_first":
            downstream, _source = figure_downstream_text(caption=raw_caption, ocr_text=raw_ocr_text, caption_first=True)
            caption = clean_figure_caption(raw_caption)
            ocr_text = clean_figure_ocr_text(raw_ocr_text)
        else:
            downstream = caption
        figure_id = _normalize_ref_token(meta.get("figure_id") or fig.get("anchor") or caption)
        if figure_id:
            figure_ids.add(figure_id)
        if caption and not GENERIC_CAPTION_RE.match(caption):
            captions += 1
        if mode == "caption_plus_ocr" and ocr_text:
            downstream = f"{caption}\n{ocr_text}".strip()
        elif mode == "caption_first" and (not caption or len(caption) < 80) and ocr_text:
            downstream = f"{caption}\n{ocr_text}".strip()
        downstream_texts.append(downstream)
        if len(caption) >= 80 or len(downstream) >= 140:
            usable += 1
        if _has_ocr_artifacts(downstream):
            artifact_rows += 1

    table_ids = {
        token
        for row in tables
        if (
            token := _normalize_ref_token(
                _parse_meta(row.get("meta")).get("table_id")
                or _parse_meta(row.get("meta")).get("figure_id")
                or row.get("anchor")
                or row.get("content")
            )
        )
    }
    matched_figures = figure_refs_expected & figure_ids
    matched_tables = table_refs_expected & table_ids
    total_downstream_chars = sum(len(text) for text in downstream_texts)
    return {
        "figure_count": len(figures),
        "table_count": len(tables),
        "expected_figure_refs": sorted(figure_refs_expected),
        "extracted_figure_ids": sorted(figure_ids),
        "figure_ref_recall": round(len(matched_figures) / len(figure_refs_expected), 3) if figure_refs_expected else 1.0,
        "missing_figure_refs": sorted(figure_refs_expected - figure_ids),
        "expected_table_refs": sorted(table_refs_expected),
        "extracted_table_ids": sorted(table_ids),
        "table_ref_recall": round(len(matched_tables) / len(table_refs_expected), 3) if table_refs_expected else 1.0,
        "missing_table_refs": sorted(table_refs_expected - table_ids),
        "supplementary_figure_ref_recall": _ref_recall(
            {ref for ref in figure_refs_expected if ref.startswith("S")},
            {ref for ref in figure_ids if ref.startswith("S")},
        ),
        "supplementary_table_ref_recall": _ref_recall(
            {ref for ref in table_refs_expected if ref.startswith("S")},
            {ref for ref in table_ids if ref.startswith("S")},
        ),
        "caption_nonempty_rate": round(captions / len(figures), 3) if figures else 0.0,
        "usable_figure_rate": round(usable / len(figures), 3) if figures else 0.0,
        "artifact_text_count": artifact_rows,
        "artifact_text_rate": round(artifact_rows / len(figures), 3) if figures else 0.0,
        "total_downstream_text_chars": total_downstream_chars,
        "mean_downstream_text_chars": round(total_downstream_chars / len(figures), 1) if figures else 0.0,
    }


def evaluate_regression_thresholds(
    payload: dict[str, Any],
    thresholds: dict[str, float] | None = None,
) -> list[str]:
    limits = dict(UPSTREAM_AB_REGRESSION_THRESHOLDS)
    if thresholds:
        limits.update(thresholds)

    failures: list[str] = []
    aggregate = payload.get("aggregate", {}) if isinstance(payload.get("aggregate"), dict) else {}
    section_variants = aggregate.get("section_variants", {}) if aggregate else payload.get("variants", {})
    media_variants = aggregate.get("media_variants", {}) if aggregate else payload.get("media_variants", {})

    ledger_metrics = section_variants.get("section_boundary_ledger", {})
    ledger_parsed = ledger_metrics.get("parsed_chunks", ledger_metrics) if isinstance(ledger_metrics, dict) else {}
    ledger_rate_key = "mean_wrong_section_rate" if aggregate else "wrong_section_rate"
    ledger_rate = _optional_float(ledger_parsed.get(ledger_rate_key))
    ledger_max = limits["section_boundary_ledger_mean_wrong_section_rate_max"]
    if ledger_rate is None:
        failures.append("section_boundary_ledger wrong_section_rate metric is missing")
    elif ledger_rate > ledger_max:
        failures.append(f"section_boundary_ledger wrong_section_rate {ledger_rate:.3f} > {ledger_max:.3f}")

    per_doc_max = limits["section_boundary_ledger_document_wrong_section_rate_max"]
    for row in payload.get("documents", []) if isinstance(payload.get("documents"), list) else []:
        parsed = (
            row.get("variants", {})
            .get("section_boundary_ledger", {})
            .get("parsed_chunks", {})
        )
        doc_rate = _optional_float(parsed.get("wrong_section_rate"))
        if doc_rate is not None and doc_rate > per_doc_max:
            failures.append(
                f"document {row.get('document_id')} section_boundary_ledger wrong_section_rate "
                f"{doc_rate:.3f} > {per_doc_max:.3f}"
            )

    clean_media = media_variants.get("clean_caption_first", {})
    for key in MEDIA_RECALL_KEYS:
        aggregate_key = f"mean_{key}"
        if aggregate and aggregate_key not in clean_media:
            failures.append(f"clean_caption_first {aggregate_key} metric is missing")
        elif not aggregate and key not in clean_media:
            failures.append(f"clean_caption_first {key} metric is missing")

    artifact_key = "mean_artifact_text_rate" if aggregate else "artifact_text_rate"
    artifact_rate = _optional_float(clean_media.get(artifact_key))
    artifact_max = limits["clean_caption_first_mean_artifact_text_rate_max"]
    if artifact_rate is None:
        failures.append(f"clean_caption_first {artifact_key} metric is missing")
    elif artifact_rate > artifact_max:
        failures.append(f"clean_caption_first artifact_text_rate {artifact_rate:.3f} > {artifact_max:.3f}")

    return failures


def _comparison_delta(variant_metrics: dict[str, Any], media_metrics: dict[str, Any]) -> dict[str, Any]:
    base = variant_metrics.get("baseline", {}).get("parsed_chunks", {})
    current_media = media_metrics.get("current_caption_plus_ocr", {})
    clean_caption_first = media_metrics.get("clean_caption_first", {})
    section_deltas: dict[str, Any] = {}
    for name, metrics in variant_metrics.items():
        if name == "baseline":
            continue
        parsed = metrics.get("parsed_chunks", {})
        section_deltas[name] = {
            "retained_rate_delta": round(
                float(parsed.get("retained_rate", 0.0) or 0.0) - float(base.get("retained_rate", 0.0) or 0.0),
                3,
            ),
            "wrong_section_delta": int(parsed.get("wrong_section_count", 0) or 0)
            - int(base.get("wrong_section_count", 0) or 0),
            "wrong_section_rate_delta": round(
                float(parsed.get("wrong_section_rate", 0.0) or 0.0)
                - float(base.get("wrong_section_rate", 0.0) or 0.0),
                3,
            ),
        }
    return {
        "section_deltas": section_deltas,
        "clean_caption_first_artifact_rate_delta": round(
            float(clean_caption_first.get("artifact_text_rate", 0.0) or 0.0)
            - float(current_media.get("artifact_text_rate", 0.0) or 0.0),
            3,
        ),
        "clean_caption_first_text_char_delta": int(clean_caption_first.get("total_downstream_text_chars", 0) or 0)
        - int(current_media.get("total_downstream_text_chars", 0) or 0),
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Upstream A/B Comparison",
        "",
        f"- document_id: {payload.get('document_id')}",
        f"- label: {payload.get('label') or ''}",
        f"- report_id: {payload.get('inputs', {}).get('report_id')}",
        "",
        "## Section Pathway",
    ]
    for name, metrics in payload.get("variants", {}).items():
        parsed = metrics.get("parsed_chunks", {})
        lines.append(
            "- "
            f"{name}: retained={parsed.get('retained_rate')}, "
            f"wrong={parsed.get('wrong_section_count')}, "
            f"wrong_rate={parsed.get('wrong_section_rate')}, "
            f"sections={metrics.get('section_counts')}"
        )
    lines += ["", "## Media Pathway"]
    for name, metrics in payload.get("media_variants", {}).items():
        lines.append(
            "- "
            f"{name}: figure_recall={metrics.get('figure_ref_recall')}, "
            f"table_recall={metrics.get('table_ref_recall')}, "
            f"supp_figure_recall={metrics.get('supplementary_figure_ref_recall')}, "
            f"supp_table_recall={metrics.get('supplementary_table_ref_recall')}, "
            f"usable_figures={metrics.get('usable_figure_rate')}, "
            f"artifact_rate={metrics.get('artifact_text_rate')}, "
            f"mean_chars={metrics.get('mean_downstream_text_chars')}, "
            f"missing_figures={metrics.get('missing_figure_refs')}"
        )
    lines += ["", "## Deltas", "```json", json.dumps(payload.get("comparison", {}), indent=2), "```", ""]
    return "\n".join(lines)


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


def _chunk_section(chunk: dict[str, Any], idx: int, total: int) -> str:
    meta = _parse_meta(chunk.get("meta"))
    for key in ("section_norm", "section_label", "section", "section_raw_title"):
        section = _normalize_known_section(meta.get(key))
        if section != "unknown":
            return section
    anchor_section = _normalize_known_section(str(chunk.get("anchor") or "").replace("_", " ").replace(":", " "))
    if anchor_section != "unknown":
        return anchor_section
    return _position_section(idx, total)


def _low_confidence_section_count(chunks: list[dict[str, Any]]) -> int:
    count = 0
    for row in chunks:
        meta = _parse_meta(row.get("meta"))
        section = _normalize_known_section(meta.get("section_norm") or meta.get("section"))
        confidence = _float(meta.get("section_confidence"), default=1.0 if section != "unknown" else 0.0)
        if section == "unknown" or confidence < 0.5:
            count += 1
    return count


def _normalize_known_section(value: Any) -> str:
    section = _normalize_section_title(str(value or ""))
    return section if section in SECTION_KEYS else "unknown"


def _leading_heading_section(text: str) -> str:
    first_line = str(text or "").replace("\r", "\n").split("\n", 1)[0].strip()
    if len(first_line) > 120:
        first_line = first_line[:120]
    return _normalize_known_section(first_line)


def _position_section(idx: int, total: int) -> str:
    if total <= 0:
        return "unknown"
    ratio = idx / float(max(1, total - 1))
    if ratio <= 0.22:
        return "introduction"
    if ratio <= 0.52:
        return "methods"
    if ratio <= 0.72:
        return "results"
    if ratio <= 0.90:
        return "discussion"
    return "conclusion"


def _normalize_ref_token(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    match = re.search(r"(?:fig(?:ure)?|table)?\s*:?\s*(s?\d+)", text, re.IGNORECASE)
    if not match:
        return ""
    token = match.group(1).upper()
    if token.startswith("S"):
        return f"S{int(re.sub(r'[^0-9]', '', token) or 0)}"
    return str(int(re.sub(r"[^0-9]", "", token) or 0))


def _ref_recall(expected: set[str], extracted: set[str]) -> float:
    return round(len(expected & extracted) / len(expected), 3) if expected else 1.0


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _has_ocr_artifacts(text: str) -> bool:
    clean = str(text or "")
    if OCR_ARTIFACT_RE.search(clean):
        return True
    if not clean:
        return False
    replacement_chars = clean.count("\ufffd")
    return replacement_chars > 0


def _allowed_backward_transition(active_section: str, candidate: str) -> bool:
    return active_section == "conclusion" and candidate == "discussion"


def _int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
