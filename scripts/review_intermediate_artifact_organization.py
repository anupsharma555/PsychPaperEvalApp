#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import html
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_OUTPUT_DIR = ROOT / "test" / "artifact_organization_review" / "latest"
STAGE_ORDER = [
    "source_manifest",
    "parsed_chunks",
    "modality_packets",
    "audited_evidence_packets",
    "synthesis_inputs",
    "retention_audit",
    "final_report",
    "runtime_diagnostics",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Review PaperEval intermediate artifact organization.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-document-id", type=int, default=0)
    parser.add_argument("--max-documents", type=int, default=40)
    args = parser.parse_args()

    doc_rows = _collect_documents(
        args.data_dir,
        min_document_id=args.min_document_id,
        max_documents=args.max_documents,
    )
    payload = _summary_payload(doc_rows, data_dir=args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "artifact_organization_review.json"
    html_path = args.output_dir / "artifact_organization_review.html"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    html_path.write_text(_render_html(payload), encoding="utf-8")
    print(f"review_json={json_path}")
    print(f"review_html={html_path}")
    print(f"documents={payload['document_count']}")
    print(f"ready={payload['aggregate']['llm_ready_count']}/{payload['document_count']}")
    print(f"stage_index_present={payload['aggregate']['stage_index_present']}/{payload['document_count']}")
    print(f"llm_input_inventory_present={payload['aggregate']['llm_input_inventory_present']}/{payload['document_count']}")
    return 0


def _collect_documents(data_dir: Path, *, min_document_id: int, max_documents: int) -> list[dict[str, Any]]:
    artifact_roots = sorted(data_dir.glob("doc_*/artifacts"), key=_doc_sort_key, reverse=True)
    rows: list[dict[str, Any]] = []
    for artifact_root in artifact_roots:
        document_id = _document_id_from_artifact_root(artifact_root)
        if document_id is None or document_id < min_document_id:
            continue
        rows.append(_document_row(document_id, artifact_root))
        if len(rows) >= max_documents:
            break
    return rows


def _doc_sort_key(path: Path) -> int:
    return _document_id_from_artifact_root(path) or -1


def _document_id_from_artifact_root(path: Path) -> int | None:
    name = path.parent.name
    if not name.startswith("doc_"):
        return None
    try:
        return int(name.split("_", 1)[1])
    except Exception:
        return None


def _document_row(document_id: int, artifact_root: Path) -> dict[str, Any]:
    stage_index_path = artifact_root / "intermediate_stage_index.json"
    llm_inventory_path = artifact_root / "llm_input_inventory.json"
    stage_index = _load_json(stage_index_path)
    llm_inventory = _load_json(llm_inventory_path)
    stages = stage_index.get("stages") if isinstance(stage_index.get("stages"), list) else []
    stage_by_id = {str(stage.get("stage_id")): stage for stage in stages if isinstance(stage, dict)}
    readiness = stage_index.get("llm_input_readiness") if isinstance(stage_index.get("llm_input_readiness"), dict) else {}
    transitions = stage_index.get("stage_transitions") if isinstance(stage_index.get("stage_transitions"), list) else []
    row = {
        "document_id": document_id,
        "artifact_root": str(artifact_root),
        "stage_index_present": bool(stage_index),
        "llm_input_inventory_present": bool(llm_inventory),
        "llm_ready": bool(readiness.get("ready")),
        "blocking_flags": _str_list(readiness.get("blocking_flags")),
        "advisory_flags": _str_list(readiness.get("advisory_flags")),
        "stage_order_complete": _stage_order_complete(stage_index.get("stage_order")),
        "stage_presence": {
            stage_id: bool(stage_by_id.get(stage_id))
            for stage_id in STAGE_ORDER
        },
        "stage_counts": {
            stage_id: _int_dict((stage_by_id.get(stage_id) or {}).get("record_counts"))
            for stage_id in STAGE_ORDER
        },
        "stage_quality_flags": {
            stage_id: _stage_quality_flags(stage_by_id.get(stage_id) or {})
            for stage_id in STAGE_ORDER
        },
        "transitions": [_transition_summary(row) for row in transitions if isinstance(row, dict)],
        "artifact_paths": _artifact_path_summary(stages),
        "llm_input_inventory": _llm_inventory_summary(llm_inventory),
    }
    row["organization_score"] = _organization_score(row)
    return row


def _summary_payload(rows: list[dict[str, Any]], *, data_dir: Path) -> dict[str, Any]:
    flag_counts: Counter[str] = Counter()
    transition_loss: defaultdict[str, int] = defaultdict(int)
    transition_flags: Counter[str] = Counter()
    stage_artifact_missing: Counter[str] = Counter()
    for row in rows:
        flag_counts.update(row.get("blocking_flags", []))
        flag_counts.update(row.get("advisory_flags", []))
        for transition in row.get("transitions", []):
            transition_loss[str(transition.get("transition_id") or "unknown")] += int(
                transition.get("loss_count", 0) or 0
            )
            transition_flags.update(transition.get("diagnostic_flags", []))
        for artifact in row.get("artifact_paths", []):
            if not artifact.get("present"):
                stage_artifact_missing[str(artifact.get("stage_id") or "unknown")] += 1
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir),
        "document_count": len(rows),
        "aggregate": {
            "stage_index_present": sum(1 for row in rows if row.get("stage_index_present")),
            "llm_input_inventory_present": sum(1 for row in rows if row.get("llm_input_inventory_present")),
            "llm_ready_count": sum(1 for row in rows if row.get("llm_ready")),
            "mean_organization_score": round(
                sum(float(row.get("organization_score", 0.0) or 0.0) for row in rows) / len(rows),
                3,
            )
            if rows
            else 0.0,
            "stage_order_complete": sum(1 for row in rows if row.get("stage_order_complete")),
            "blocking_and_advisory_flags": dict(sorted(flag_counts.items())),
            "transition_loss_totals": dict(sorted(transition_loss.items())),
            "transition_flag_counts": dict(sorted(transition_flags.items())),
            "missing_artifact_paths_by_stage": dict(sorted(stage_artifact_missing.items())),
        },
        "documents": rows,
    }


def _transition_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "transition_id": str(row.get("transition_id") or ""),
        "from_stage": str(row.get("from_stage") or ""),
        "to_stage": str(row.get("to_stage") or ""),
        "input_count": _int_or_zero(row.get("input_count")),
        "output_count": _int_or_zero(row.get("output_count")),
        "loss_count": _int_or_zero(row.get("loss_count")),
        "loss_rate": row.get("loss_rate"),
        "diagnostic_flags": _str_list(row.get("diagnostic_flags")),
        "quality_gaps": _int_dict(row.get("quality_gaps")),
    }


def _artifact_path_summary(stages: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        for artifact in stage.get("artifact_paths", []) if isinstance(stage.get("artifact_paths"), list) else []:
            if not isinstance(artifact, dict):
                continue
            rows.append(
                {
                    "stage_id": str(stage.get("stage_id") or ""),
                    "path": str(artifact.get("path") or ""),
                    "present": bool(artifact.get("present")),
                    "bytes": _int_or_zero(artifact.get("bytes")),
                }
            )
    return rows


def _llm_inventory_summary(payload: dict[str, Any]) -> dict[str, Any]:
    inventory = payload.get("inventory") if isinstance(payload.get("inventory"), dict) else {}
    if not inventory:
        return {}
    return {
        "selected_prompt_detail_count": _int_or_zero(inventory.get("selected_prompt_detail_count")),
        "eligible_scientific_detail_count": _int_or_zero(inventory.get("eligible_scientific_detail_count")),
        "omitted_candidate_count": _int_or_zero(inventory.get("omitted_candidate_count")),
        "selected_quality": _int_dict(inventory.get("selected_quality")),
        "focus_slot_counts": _int_dict(inventory.get("focus_slot_counts")),
        "quality_flags": _str_list(inventory.get("quality_flags")),
    }


def _render_html(payload: dict[str, Any]) -> str:
    aggregate = payload.get("aggregate", {})
    docs = payload.get("documents", [])
    cards = "\n".join(_doc_card(row) for row in docs)
    stage_nodes = "\n".join(_stage_node(stage) for stage in STAGE_ORDER)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PaperEval Artifact Organization Review</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #17201b;
      --muted: #5b665f;
      --line: #c8d4cd;
      --panel: #f6f8f6;
      --stage: #e8f1ec;
      --warn: #9a5b00;
      --bad: #a63838;
      --good: #226c45;
    }}
    body {{
      margin: 0;
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: #ffffff;
    }}
    main {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 24px;
    }}
    h1 {{ margin: 0 0 4px; font-size: 28px; letter-spacing: 0; }}
    h2 {{ margin: 26px 0 10px; font-size: 18px; }}
    .subtle {{ color: var(--muted); }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(4, minmax(140px, 1fr));
      gap: 12px;
      margin-top: 18px;
    }}
    .metric {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: var(--panel);
    }}
    .metric strong {{ display: block; font-size: 24px; }}
    .pipeline {{
      display: grid;
      grid-template-columns: repeat(8, minmax(118px, 1fr));
      gap: 8px;
      align-items: stretch;
      overflow-x: auto;
      padding-bottom: 8px;
    }}
    .stage {{
      min-height: 86px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: var(--stage);
      position: relative;
    }}
    .stage:not(:last-child)::after {{
      content: "→";
      position: absolute;
      right: -12px;
      top: 31px;
      color: var(--muted);
      font-weight: 700;
    }}
    .stage-title {{ font-weight: 700; margin-bottom: 6px; }}
    .docs {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
      gap: 14px;
    }}
    .doc {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fff;
    }}
    .doc.ready {{ border-color: #7fb794; }}
    .doc.blocked {{ border-color: #d8a7a7; }}
    .row {{ display: flex; justify-content: space-between; gap: 12px; border-top: 1px solid #edf1ee; padding: 6px 0; }}
    .pill {{
      display: inline-block;
      padding: 2px 7px;
      border: 1px solid var(--line);
      border-radius: 999px;
      margin: 2px 4px 2px 0;
      color: var(--muted);
      background: #fff;
      font-size: 12px;
    }}
    .bad {{ color: var(--bad); }}
    .warn {{ color: var(--warn); }}
    .good {{ color: var(--good); }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }}
    @media (max-width: 820px) {{
      main {{ padding: 16px; }}
      .metrics {{ grid-template-columns: repeat(2, 1fr); }}
      .pipeline {{ grid-template-columns: repeat(8, 140px); }}
      .docs {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
<main>
  <h1>PaperEval Artifact Organization Review</h1>
  <div class="subtle">Generated {html.escape(str(payload.get("generated_at", "")))} from <code>{html.escape(str(payload.get("data_dir", "")))}</code></div>
  <section class="metrics">
    {_metric("Documents", payload.get("document_count", 0), "")}
    {_metric("Stage Indexes", f"{aggregate.get('stage_index_present', 0)}/{payload.get('document_count', 0)}", "")}
    {_metric("LLM Inventories", f"{aggregate.get('llm_input_inventory_present', 0)}/{payload.get('document_count', 0)}", "")}
    {_metric("LLM Ready", f"{aggregate.get('llm_ready_count', 0)}/{payload.get('document_count', 0)}", "")}
    {_metric("Mean Org Score", aggregate.get('mean_organization_score', 0), "0-1 diagnostic")}
  </section>
  <h2>Data Flow Standard</h2>
  <section class="pipeline">{stage_nodes}</section>
  <h2>Aggregate Diagnostics</h2>
  <div class="doc">
    {_kv_block("Transition Loss Totals", aggregate.get("transition_loss_totals", {}))}
    {_kv_block("Transition Flags", aggregate.get("transition_flag_counts", {}))}
    {_kv_block("Readiness Flags", aggregate.get("blocking_and_advisory_flags", {}))}
    {_kv_block("Missing Artifact Paths", aggregate.get("missing_artifact_paths_by_stage", {}))}
  </div>
  <h2>Documents</h2>
  <section class="docs">{cards}</section>
</main>
</body>
</html>
"""


def _metric(label: str, value: Any, note: str) -> str:
    return f'<div class="metric"><strong>{html.escape(str(value))}</strong><span>{html.escape(label)}</span><div class="subtle">{html.escape(note)}</div></div>'


def _stage_node(stage_id: str) -> str:
    title = stage_id.replace("_", " ").title()
    return f'<div class="stage"><div class="stage-title">{html.escape(title)}</div><div class="subtle"><code>{html.escape(stage_id)}</code></div></div>'


def _doc_card(row: dict[str, Any]) -> str:
    ready = bool(row.get("llm_ready"))
    class_name = "doc ready" if ready else "doc blocked"
    transitions = row.get("transitions", [])
    biggest = sorted(transitions, key=lambda item: int(item.get("loss_count", 0) or 0), reverse=True)[:3]
    flags = row.get("blocking_flags", []) + row.get("advisory_flags", [])
    inventory = row.get("llm_input_inventory") if isinstance(row.get("llm_input_inventory"), dict) else {}
    return f"""
    <article class="{class_name}">
      <h3>Document {html.escape(str(row.get("document_id", "")))}</h3>
      <div class="row"><span>Organization score</span><strong>{html.escape(str(row.get('organization_score', 0)))}</strong></div>
      <div class="row"><span>Readiness</span><strong class="{'good' if ready else 'bad'}">{'ready' if ready else 'not ready'}</strong></div>
      <div class="row"><span>Stage order</span><strong>{'complete' if row.get('stage_order_complete') else 'incomplete'}</strong></div>
      <div class="row"><span>LLM inventory</span><strong>{'present' if row.get('llm_input_inventory_present') else 'missing'}</strong></div>
      <div class="row"><span>Selected prompt details</span><strong>{html.escape(str(inventory.get('selected_prompt_detail_count', 'n/a')))}</strong></div>
      <div><span class="subtle">Flags</span><br>{_pills(flags)}</div>
      <div><span class="subtle">Largest losses</span>{_loss_rows(biggest)}</div>
    </article>
    """


def _loss_rows(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<div class="subtle">No transition records.</div>'
    return "".join(
        f'<div class="row"><span><code>{html.escape(str(row.get("transition_id", "")))}</code></span><strong>{html.escape(str(row.get("loss_count", 0)))}</strong></div>'
        for row in rows
    )


def _pills(values: list[Any]) -> str:
    if not values:
        return '<span class="pill good">none</span>'
    return "".join(f'<span class="pill warn">{html.escape(str(value))}</span>' for value in values[:12])


def _organization_score(row: dict[str, Any]) -> float:
    score = 0.0
    if row.get("stage_index_present"):
        score += 0.15
    if row.get("stage_order_complete"):
        score += 0.10
    if row.get("llm_input_inventory_present"):
        score += 0.20

    artifacts = row.get("artifact_paths") if isinstance(row.get("artifact_paths"), list) else []
    if artifacts:
        present = sum(1 for item in artifacts if isinstance(item, dict) and item.get("present"))
        score += 0.10 * (present / len(artifacts))

    blocking_flags = row.get("blocking_flags") if isinstance(row.get("blocking_flags"), list) else []
    score += 0.25 * max(0.0, 1.0 - (len(blocking_flags) / 3.0))

    inventory = row.get("llm_input_inventory") if isinstance(row.get("llm_input_inventory"), dict) else {}
    selected_count = int(inventory.get("selected_prompt_detail_count", 0) or 0) if inventory else 0
    selected_quality = inventory.get("selected_quality") if isinstance(inventory.get("selected_quality"), dict) else {}
    if selected_count > 0:
        issue_count = (
            int(selected_quality.get("missing_source_excerpt", 0) or 0)
            + int(selected_quality.get("missing_detail_types", 0) or 0)
            + int(selected_quality.get("unknown_section", 0) or 0)
        )
        score += 0.15 * max(0.0, 1.0 - (issue_count / max(1, selected_count * 3)))

    flags = row.get("blocking_flags", []) + row.get("advisory_flags", [])
    score += 0.05 if "retention_loss_at_text_packets" not in flags else 0.025
    return round(min(1.0, max(0.0, score)), 3)


def _kv_block(title: str, values: Any) -> str:
    if not isinstance(values, dict) or not values:
        return f'<h3>{html.escape(title)}</h3><div class="subtle">None recorded.</div>'
    rows = "".join(
        f'<div class="row"><span><code>{html.escape(str(key))}</code></span><strong>{html.escape(str(value))}</strong></div>'
        for key, value in sorted(values.items(), key=lambda item: str(item[0]))
    )
    return f'<h3>{html.escape(title)}</h3>{rows}'


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _stage_order_complete(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    return [str(item) for item in value] == STAGE_ORDER


def _stage_quality_flags(stage: dict[str, Any]) -> list[str]:
    quality = stage.get("quality") if isinstance(stage.get("quality"), dict) else {}
    flags = quality.get("quality_flags") if isinstance(quality.get("quality_flags"), list) else []
    return _str_list(flags)


def _str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _int_dict(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, int] = {}
    for key, raw in value.items():
        out[str(key)] = _int_or_zero(raw)
    return out


def _int_or_zero(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
