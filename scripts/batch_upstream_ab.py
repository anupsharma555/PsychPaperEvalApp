#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from sqlmodel import Session, select  # noqa: E402

import compare_upstream_ab as upstream_ab  # noqa: E402
from app.db.models import Asset, Chunk, Document, Report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run offline upstream A/B/C/D extraction comparisons across parsed documents."
    )
    parser.add_argument("--document-id", action="append", type=int, default=[])
    parser.add_argument("--test-dir", default="test")
    parser.add_argument("--include-all-parsed", action="store_true")
    parser.add_argument("--out-dir", default="test/upstream_ab")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    from app.db.session import engine  # noqa: WPS433

    test_names = _test_pdf_names(Path(args.test_dir))
    with Session(engine) as session:
        doc_ids = args.document_id or _discover_document_ids(
            session,
            test_names=test_names,
            include_all_parsed=args.include_all_parsed,
        )
        if args.limit > 0:
            doc_ids = doc_ids[: args.limit]
        rows = [_compare_document(session, doc_id) for doc_id in doc_ids]

    rows = [row for row in rows if row]
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "document_count": len(rows),
        "documents": rows,
        "aggregate": _aggregate(rows),
    }

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"upstream_ab_batch_{stamp}.json"
    md_path = out_dir / f"upstream_ab_batch_{stamp}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(payload), encoding="utf-8")

    print(f"document_count={len(rows)}")
    print(f"batch_json={json_path}")
    print(f"batch_md={md_path}")
    for variant, metrics in payload["aggregate"].get("section_variants", {}).items():
        print(
            f"{variant}: retained={metrics.get('mean_retained_rate')} "
            f"wrong_rate={metrics.get('mean_wrong_section_rate')} "
            f"wins={metrics.get('wrong_section_wins')}"
        )
    for variant, metrics in payload["aggregate"].get("media_variants", {}).items():
        print(
            f"{variant}: artifact_rate={metrics.get('mean_artifact_text_rate')} "
            f"chars={metrics.get('mean_downstream_text_chars')} "
            f"fig_recall={metrics.get('mean_figure_ref_recall')}"
        )


def _test_pdf_names(test_dir: Path) -> set[str]:
    root = test_dir if test_dir.is_absolute() else PROJECT_ROOT / test_dir
    return {path.name for path in root.glob("*.pdf")}


def _discover_document_ids(
    session: Session,
    *,
    test_names: set[str],
    include_all_parsed: bool,
) -> list[int]:
    assets = session.exec(select(Asset).where(Asset.kind == "main")).all()
    by_filename: dict[str, int] = {}
    for asset in assets:
        filename = str(asset.filename or Path(asset.path or "").name)
        if test_names and filename not in test_names and not include_all_parsed:
            continue
        chunks = session.exec(select(Chunk.id).where(Chunk.document_id == asset.document_id).limit(1)).all()
        if not chunks:
            continue
        by_filename[filename] = max(int(asset.document_id), by_filename.get(filename, 0))
    return sorted(set(by_filename.values()))


def _compare_document(session: Session, doc_id: int) -> dict[str, Any]:
    document = session.get(Document, doc_id)
    if document is None:
        return {}
    assets = session.exec(select(Asset).where(Asset.document_id == doc_id)).all()
    chunks = session.exec(select(Chunk).where(Chunk.document_id == doc_id).order_by(Chunk.id)).all()
    if not chunks:
        return {}
    report = session.exec(select(Report).where(Report.document_id == doc_id).order_by(Report.id.desc())).first()
    parsed_chunks = [upstream_ab._chunk_to_dict(chunk) for chunk in chunks]
    source_assets = [upstream_ab._asset_to_dict(asset) for asset in assets]
    variants = {
        "baseline": parsed_chunks,
        "source_first_sections": upstream_ab._source_first_section_relabel(parsed_chunks),
        "heading_boundary_sections": upstream_ab._heading_boundary_section_relabel(parsed_chunks),
        "imrad_guarded_sections": upstream_ab._imrad_guarded_section_relabel(parsed_chunks),
        "section_boundary_ledger": upstream_ab._section_boundary_ledger_relabel(parsed_chunks),
    }
    variant_metrics = {
        name: upstream_ab._variant_metrics(
            document_id=doc_id,
            source_assets=source_assets,
            parsed_chunks=variant_chunks,
        )
        for name, variant_chunks in variants.items()
    }
    media_metrics = {
        "current_caption_plus_ocr": upstream_ab._media_metrics(parsed_chunks, mode="caption_plus_ocr"),
        "caption_first": upstream_ab._media_metrics(parsed_chunks, mode="caption_first"),
        "clean_caption_first": upstream_ab._media_metrics(parsed_chunks, mode="clean_caption_first"),
    }
    return {
        "document_id": doc_id,
        "title": document.title,
        "filename": next((asset.filename for asset in assets if asset.kind == "main"), ""),
        "report_id": report.id if report else None,
        "chunk_count": len(parsed_chunks),
        "variants": variant_metrics,
        "media_variants": media_metrics,
        "comparison": upstream_ab._comparison_delta(variant_metrics, media_metrics),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    section_names = sorted({name for row in rows for name in row.get("variants", {})})
    media_names = sorted({name for row in rows for name in row.get("media_variants", {})})
    section_variants = {name: _aggregate_section_variant(rows, name) for name in section_names}
    media_variants = {name: _aggregate_media_variant(rows, name) for name in media_names}
    return {
        "section_variants": section_variants,
        "media_variants": media_variants,
        "best_section_by_wrong_rate": _best_variant(section_variants, "mean_wrong_section_rate"),
        "best_media_by_artifact_rate": _best_variant(media_variants, "mean_artifact_text_rate"),
    }


def _aggregate_section_variant(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    parsed_rows = [
        row.get("variants", {}).get(name, {}).get("parsed_chunks", {})
        for row in rows
        if row.get("variants", {}).get(name)
    ]
    baseline_rows = [
        row.get("variants", {}).get("baseline", {}).get("parsed_chunks", {})
        for row in rows
        if row.get("variants", {}).get(name)
    ]
    wins = 0
    ties = 0
    for parsed, baseline in zip(parsed_rows, baseline_rows, strict=False):
        wrong = float(parsed.get("wrong_section_rate", 0.0) or 0.0)
        base_wrong = float(baseline.get("wrong_section_rate", 0.0) or 0.0)
        if wrong < base_wrong:
            wins += 1
        elif wrong == base_wrong:
            ties += 1
    return {
        "documents": len(parsed_rows),
        "mean_retained_rate": _mean(parsed_rows, "retained_rate"),
        "mean_wrong_section_rate": _mean(parsed_rows, "wrong_section_rate"),
        "mean_wrong_section_count": _mean(parsed_rows, "wrong_section_count"),
        "wrong_section_wins": wins,
        "wrong_section_ties": ties,
    }


def _aggregate_media_variant(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    media_rows = [
        row.get("media_variants", {}).get(name, {})
        for row in rows
        if row.get("media_variants", {}).get(name)
    ]
    baseline_rows = [
        row.get("media_variants", {}).get("current_caption_plus_ocr", {})
        for row in rows
        if row.get("media_variants", {}).get(name)
    ]
    wins = 0
    ties = 0
    for media, baseline in zip(media_rows, baseline_rows, strict=False):
        artifact = float(media.get("artifact_text_rate", 0.0) or 0.0)
        base_artifact = float(baseline.get("artifact_text_rate", 0.0) or 0.0)
        if artifact < base_artifact:
            wins += 1
        elif artifact == base_artifact:
            ties += 1
    return {
        "documents": len(media_rows),
        "mean_figure_ref_recall": _mean(media_rows, "figure_ref_recall"),
        "mean_table_ref_recall": _mean(media_rows, "table_ref_recall"),
        "mean_artifact_text_rate": _mean(media_rows, "artifact_text_rate"),
        "mean_downstream_text_chars": _mean(media_rows, "mean_downstream_text_chars"),
        "artifact_rate_wins": wins,
        "artifact_rate_ties": ties,
    }


def _best_variant(metrics: dict[str, dict[str, Any]], key: str) -> str:
    candidates = [(name, payload.get(key)) for name, payload in metrics.items()]
    candidates = [(name, float(value)) for name, value in candidates if value is not None]
    if not candidates:
        return ""
    return min(candidates, key=lambda item: item[1])[0]


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row.get(key, 0.0) or 0.0) for row in rows]
    return round(sum(values) / len(values), 3) if values else 0.0


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Upstream A/B/C/D Batch",
        "",
        f"- document_count: {payload.get('document_count')}",
        "",
        "## Aggregate Section Variants",
    ]
    for name, metrics in payload.get("aggregate", {}).get("section_variants", {}).items():
        lines.append(
            f"- {name}: retained={metrics.get('mean_retained_rate')}, "
            f"wrong_rate={metrics.get('mean_wrong_section_rate')}, "
            f"wins={metrics.get('wrong_section_wins')}, ties={metrics.get('wrong_section_ties')}"
        )
    lines += ["", "## Aggregate Media Variants"]
    for name, metrics in payload.get("aggregate", {}).get("media_variants", {}).items():
        lines.append(
            f"- {name}: fig_recall={metrics.get('mean_figure_ref_recall')}, "
            f"artifact_rate={metrics.get('mean_artifact_text_rate')}, "
            f"mean_chars={metrics.get('mean_downstream_text_chars')}, "
            f"wins={metrics.get('artifact_rate_wins')}, ties={metrics.get('artifact_rate_ties')}"
        )
    lines += ["", "## Documents"]
    for row in payload.get("documents", []):
        baseline = row.get("variants", {}).get("baseline", {}).get("parsed_chunks", {})
        best_media = row.get("media_variants", {}).get("clean_caption_first", {})
        lines.append(
            f"- doc {row.get('document_id')} ({row.get('filename')}): "
            f"baseline_retained={baseline.get('retained_rate')}, "
            f"baseline_wrong_rate={baseline.get('wrong_section_rate')}, "
            f"clean_media_artifact_rate={best_media.get('artifact_text_rate')}, "
            f"missing_figures={best_media.get('missing_figure_refs')}"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
