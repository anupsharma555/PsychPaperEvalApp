#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from sqlmodel import Session, select  # noqa: E402

from app.db.models import Asset, Chunk, Job, JobStatus, Report  # noqa: E402
from app.db.session import engine  # noqa: E402
from app.services.analysis.information_retention import build_information_retention_audit  # noqa: E402
from app.services.analysis.synthesis import apply_section_synthesis_v2_payload  # noqa: E402
from app.services.storage import artifacts_dir  # noqa: E402


SECTION_KEYS = ["introduction", "methods", "results", "discussion", "conclusion"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare production section synthesis with experimental section_synthesis_v2.",
    )
    parser.add_argument("--document-id", type=int, default=None)
    parser.add_argument("--job-id", type=int, default=None)
    parser.add_argument("--reference-md", default=None, help="Optional gold-standard extraction markdown.")
    parser.add_argument("--use-llm", action="store_true", help="Call the experimental V2 LLM synthesis path.")
    parser.add_argument("--out-dir", default=None, help="Output directory. Defaults to document artifacts dir.")
    args = parser.parse_args()

    compare = _load_compare_module()
    with Session(engine) as session:
        document_id = _resolve_document_id(session, document_id=args.document_id, job_id=args.job_id)
        report = _latest_report(session, document_id)
        summary = _load_report_payload(report)
        assets = session.exec(select(Asset).where(Asset.document_id == document_id)).all()
        chunks = session.exec(select(Chunk).where(Chunk.document_id == document_id)).all()

    parsed_chunks = [_chunk_to_dict(chunk) for chunk in chunks]
    source_assets = [_asset_to_dict(asset) for asset in assets]
    v2_summary = apply_section_synthesis_v2_payload(
        summary,
        parsed_chunks=parsed_chunks,
        use_llm=bool(args.use_llm),
    )

    v1_audit = build_information_retention_audit(
        document_id=document_id,
        source_assets=source_assets,
        parsed_chunks=parsed_chunks,
        summary_json=summary,
    )
    v2_audit = build_information_retention_audit(
        document_id=document_id,
        source_assets=source_assets,
        parsed_chunks=parsed_chunks,
        summary_json=v2_summary,
    )
    v1_sections = _sections_from_summary_for_comparison(compare, summary)
    v2_sections = _sections_from_v2_report(v2_summary.get("executive_report", {}))
    reference_metrics = {}
    if args.reference_md:
        ref_path = Path(args.reference_md).expanduser().resolve()
        ref_sections = compare._parse_reference_markdown(ref_path)
        reference_metrics = {
            "v1": _comparison_summary(compare, v1_sections, ref_sections),
            "v2": _comparison_summary(compare, v2_sections, ref_sections),
        }

    quality = {
        "v1": _report_quality_summary(summary.get("executive_report", {})),
        "v2": _report_quality_summary(v2_summary.get("executive_report", {})),
    }
    retention = {
        "v1": v1_audit.get("compact_summary", {}),
        "v2": v2_audit.get("compact_summary", {}),
        "delta": _retention_delta(v1_audit, v2_audit),
    }
    result = {
        "document_id": document_id,
        "source_report_id": report.id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "v2_used_llm": bool(args.use_llm),
        "quality": quality,
        "reference_metrics": reference_metrics,
        "retention": retention,
        "v1_report": summary.get("executive_report", {}),
        "v2_report": v2_summary.get("executive_report", {}),
    }

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else artifacts_dir(document_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"section_synthesis_v2_comparison_{stamp}.json"
    md_path = out_dir / f"section_synthesis_v2_comparison_{stamp}.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    md_path.write_text(_comparison_markdown(result), encoding="utf-8")

    print(f"document_id={document_id}")
    print(f"source_report_id={report.id}")
    print(f"comparison_json={json_path}")
    print(f"comparison_md={md_path}")
    print(f"v1_final_retained={_final_retained_rate(v1_audit)}")
    print(f"v2_final_retained={_final_retained_rate(v2_audit)}")
    if reference_metrics:
        print(f"v1_gold_recall={reference_metrics['v1'].get('overall_sentence_inclusion_recall')}")
        print(f"v2_gold_recall={reference_metrics['v2'].get('overall_sentence_inclusion_recall')}")
    return 0


def _load_compare_module() -> Any:
    path = ROOT / "scripts" / "compare_pdf_against_reference.py"
    spec = importlib.util.spec_from_file_location("compare_pdf_against_reference", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not load comparator module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_document_id(session: Session, *, document_id: int | None, job_id: int | None) -> int:
    if document_id is not None:
        return int(document_id)
    if job_id is not None:
        job = session.get(Job, int(job_id))
        if not job:
            raise SystemExit(f"No job found for id {job_id}.")
        return int(job.document_id)
    job = session.exec(
        select(Job).where(Job.status == JobStatus.completed).order_by(Job.updated_at.desc())
    ).first()
    if not job:
        raise SystemExit("No completed jobs found. Pass --document-id or --job-id.")
    return int(job.document_id)


def _latest_report(session: Session, document_id: int) -> Report:
    report = session.exec(
        select(Report).where(Report.document_id == document_id).order_by(Report.created_at.desc())
    ).first()
    if not report:
        raise SystemExit(f"No report found for document {document_id}.")
    return report


def _load_report_payload(report: Report) -> dict[str, Any]:
    try:
        payload = json.loads(report.payload or "{}")
    except Exception as exc:
        raise SystemExit(f"Report {report.id} payload is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"Report {report.id} payload is not a JSON object.")
    return payload


def _chunk_to_dict(chunk: Chunk) -> dict[str, Any]:
    return {
        "anchor": str(chunk.anchor or ""),
        "content": str(chunk.content or ""),
        "meta": chunk.meta,
        "modality": str(chunk.modality or "text"),
    }


def _asset_to_dict(asset: Asset) -> dict[str, Any]:
    return {
        "kind": asset.kind,
        "filename": asset.filename,
        "content_type": asset.content_type,
        "path": asset.path,
    }


def _sections_from_summary_for_comparison(compare: Any, summary: dict[str, Any]) -> dict[str, list[str]]:
    return compare._extract_app_sections(summary)


def _sections_from_v2_report(report: Any) -> dict[str, list[str]]:
    out = {section: [] for section in SECTION_KEYS}
    if not isinstance(report, dict):
        return out
    for row in report.get("sections", []) if isinstance(report.get("sections"), list) else []:
        if not isinstance(row, dict):
            continue
        section = str(row.get("section", "")).strip().lower()
        if section not in out:
            continue
        summary = " ".join(str(row.get("summary", "") or "").split()).strip()
        if summary:
            out[section].append(summary)
        for point in row.get("salient_points", []) if isinstance(row.get("salient_points"), list) else []:
            text = " ".join(str(point or "").split()).strip()
            if text:
                out[section].append(text)
        for bullet in row.get("bullets", []) if isinstance(row.get("bullets"), list) else []:
            if not isinstance(bullet, dict):
                continue
            text = " ".join(str(bullet.get("text", "") or "").split()).strip()
            if text:
                out[section].append(text)
    return out


def _comparison_summary(compare: Any, app_sections: dict[str, list[str]], ref_sections: dict[str, list[str]]) -> dict[str, Any]:
    comparison = compare._compare_sections(app_sections, ref_sections, match_threshold=0.42, matching_mode="hybrid")
    return {
        "overall_reference_points": comparison.get("overall_reference_points"),
        "overall_matched_points": comparison.get("overall_matched_points"),
        "overall_recall": comparison.get("overall_recall"),
        "overall_sentence_inclusion_recall": comparison.get("overall_sentence_inclusion_recall"),
        "overall_sentence_inclusion_any_section_recall": comparison.get("overall_sentence_inclusion_any_section_recall"),
        "overall_section_fidelity": comparison.get("overall_section_fidelity"),
        "overall_inclusion_precision": comparison.get("overall_inclusion_precision"),
        "sections": {
            section: {
                "reference_points": payload.get("reference_points"),
                "app_points": payload.get("app_points"),
                "matched_points": payload.get("matched_points"),
                "sentence_inclusion_recall": payload.get("sentence_inclusion_recall"),
                "section_fidelity": payload.get("section_fidelity"),
            }
            for section, payload in comparison.get("sections", {}).items()
            if isinstance(payload, dict)
        },
    }


def _report_quality_summary(report: Any) -> dict[str, Any]:
    if not isinstance(report, dict):
        return {}
    sections = [row for row in report.get("sections", []) if isinstance(row, dict)] if isinstance(report.get("sections"), list) else []
    summaries = [str(row.get("summary", "") or "").strip() for row in sections]
    bullets = sum(len(row.get("bullets", [])) for row in sections if isinstance(row.get("bullets"), list))
    key_terms = sum(len(row.get("key_terms", [])) for row in sections if isinstance(row.get("key_terms"), list))
    populated = sum(1 for text in summaries if text)
    avg_summary_words = round(
        sum(len(text.split()) for text in summaries if text) / max(1, populated),
        1,
    )
    return {
        "style": str(report.get("style", "")),
        "section_count": len(sections),
        "populated_sections": populated,
        "bullet_count": bullets,
        "key_term_count": key_terms,
        "avg_summary_words": avg_summary_words,
        "overview_words": len(str(report.get("overview", "") or "").split()),
        "synthesis_applied": bool(report.get("synthesis_applied")),
    }


def _final_retained_rate(audit: dict[str, Any]) -> float:
    metrics = audit.get("compact_summary", {}).get("stage_metrics", [])
    if not isinstance(metrics, list):
        return 0.0
    for row in metrics:
        if isinstance(row, dict) and row.get("stage") == "executive_report":
            return float(row.get("retained_rate", 0.0) or 0.0)
    return 0.0


def _retention_delta(v1: dict[str, Any], v2: dict[str, Any]) -> dict[str, Any]:
    v1_metrics = _stage_metric_map(v1)
    v2_metrics = _stage_metric_map(v2)
    return {
        "executive_report_retained_rate_delta": round(_final_retained_rate(v2) - _final_retained_rate(v1), 3),
        "executive_report_wrong_section_delta": int(
            v2_metrics.get("executive_report", {}).get("wrong_section_count", 0) or 0
        )
        - int(v1_metrics.get("executive_report", {}).get("wrong_section_count", 0) or 0),
        "stage_deltas": {
            stage: {
                "retained_rate_delta": round(
                    float(v2_metrics.get(stage, {}).get("retained_rate", 0.0) or 0.0)
                    - float(v1_metrics.get(stage, {}).get("retained_rate", 0.0) or 0.0),
                    3,
                ),
                "wrong_section_delta": int(v2_metrics.get(stage, {}).get("wrong_section_count", 0) or 0)
                - int(v1_metrics.get(stage, {}).get("wrong_section_count", 0) or 0),
            }
            for stage in sorted(set(v1_metrics) | set(v2_metrics))
        },
    }


def _stage_metric_map(audit: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = audit.get("compact_summary", {}).get("stage_metrics", [])
    if not isinstance(rows, list):
        return {}
    return {str(row.get("stage", "")): row for row in rows if isinstance(row, dict)}


def _comparison_markdown(result: dict[str, Any]) -> str:
    lines = ["# Section Synthesis V2 Comparison", ""]
    lines.append(f"- document_id: {result.get('document_id')}")
    lines.append(f"- source_report_id: {result.get('source_report_id')}")
    lines.append(f"- v2_used_llm: {result.get('v2_used_llm')}")
    lines.append("")
    lines.append("## Report Quality")
    for label in ("v1", "v2"):
        payload = result.get("quality", {}).get(label, {})
        lines.append(
            f"- {label}: style={payload.get('style')}, populated_sections={payload.get('populated_sections')}, "
            f"bullets={payload.get('bullet_count')}, key_terms={payload.get('key_term_count')}, "
            f"avg_summary_words={payload.get('avg_summary_words')}, overview_words={payload.get('overview_words')}"
        )
    lines.append("")
    if result.get("reference_metrics"):
        lines.append("## Gold Similarity")
        for label in ("v1", "v2"):
            payload = result["reference_metrics"].get(label, {})
            lines.append(
                f"- {label}: recall={payload.get('overall_recall')}, "
                f"sentence_recall={payload.get('overall_sentence_inclusion_recall')}, "
                f"section_fidelity={payload.get('overall_section_fidelity')}, "
                f"precision={payload.get('overall_inclusion_precision')}"
            )
        lines.append("")
    lines.append("## Information Retention")
    delta = result.get("retention", {}).get("delta", {})
    lines.append(f"- executive_report_retained_rate_delta: {delta.get('executive_report_retained_rate_delta')}")
    lines.append(f"- executive_report_wrong_section_delta: {delta.get('executive_report_wrong_section_delta')}")
    for label in ("v1", "v2"):
        summary = result.get("retention", {}).get(label, {})
        lines.append(f"- {label}_source_sentences: {summary.get('source_sentence_count')}")
        for row in summary.get("stage_metrics", []) if isinstance(summary.get("stage_metrics"), list) else []:
            if row.get("stage") in {"sections_extracted", "sections", "executive_report"}:
                lines.append(
                    f"  - {label}.{row.get('stage')}: retained={row.get('retained_rate')} "
                    f"wrong={row.get('wrong_section_count')} lost_here={row.get('lost_here_count')}"
                )
    lines.append("")
    lines.append("## V2 Executive Report")
    report = result.get("v2_report", {})
    for row in report.get("sections", []) if isinstance(report, dict) and isinstance(report.get("sections"), list) else []:
        lines.append(f"### {str(row.get('section', '')).title()}")
        lines.append(str(row.get("summary", "") or ""))
        terms = row.get("key_terms", [])
        if isinstance(terms, list) and terms:
            lines.append("")
            lines.append("Key terms:")
            for term in terms:
                if isinstance(term, dict):
                    lines.append(f"- {term.get('term')}: {term.get('explanation')}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
