#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
COMPARE_SCRIPT = ROOT / "scripts" / "compare_pdf_against_reference.py"


def _prune_benchmark_artifacts(out_dir: Path, *, stem: str, keep_latest: int) -> int:
    if keep_latest <= 0:
        return 0
    family = "model_benchmark"
    family_re = re.compile(rf"^{re.escape(stem)}_{family}_(\d{{8}}_\d{{6}})\.json$")
    stamped: dict[str, list[Path]] = {}
    for path in out_dir.glob(f"{stem}_{family}_*"):
        if not path.is_file():
            continue
        match = family_re.match(path.name)
        if not match:
            continue
        stamped.setdefault(match.group(1), []).append(path)
    if len(stamped) <= keep_latest:
        return 0
    removed = 0
    stale_stamps = sorted(stamped.keys(), reverse=True)[keep_latest:]
    for stamp in stale_stamps:
        for artifact in stamped.get(stamp, []):
            try:
                artifact.unlink()
                removed += 1
            except OSError:
                continue
    return removed


def _run_one(
    *,
    pdf: Path,
    reference_md: Path,
    out_dir: Path,
    db_path: Path,
    parser_engine: str,
    backend_profile: str,
    matching_mode: str,
    matching_threshold: float,
    retain_runs: int,
    text_model_path: Path,
    deep_model_path: Path,
) -> dict[str, Any]:
    env = os.environ.copy()
    env["PARSER_ENGINE"] = parser_engine
    env["LLM_TEXT_MODEL_PATH"] = str(text_model_path)
    env["LLM_DEEP_MODEL_PATH"] = str(deep_model_path)
    env["MATCHING_MODE"] = matching_mode
    env["MATCHING_THRESHOLD"] = str(matching_threshold)
    env["MATCHING_HYBRID_THRESHOLD"] = str(matching_threshold)
    cmd = [
        str(ROOT / ".venv" / "bin" / "python"),
        str(COMPARE_SCRIPT),
        "--mode",
        "pipeline",
        "--parser-engine",
        parser_engine,
        "--backend-profile",
        backend_profile,
        "--pdf",
        str(pdf),
        "--reference-md",
        str(reference_md),
        "--out-dir",
        str(out_dir),
        "--retain-runs",
        str(max(0, int(retain_runs))),
        "--matching-mode",
        matching_mode,
        "--matching-threshold",
        str(matching_threshold),
        "--db-path",
        str(db_path),
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True)
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    if proc.returncode != 0:
        return {
            "ok": False,
            "returncode": int(proc.returncode),
            "stdout": stdout,
            "stderr": stderr,
        }

    run_json_path = ""
    comparison_json_path = ""
    for raw in stdout.splitlines():
        line = raw.strip()
        if line.startswith("run_json="):
            run_json_path = line.split("=", 1)[1].strip()
        elif line.startswith("comparison_json="):
            comparison_json_path = line.split("=", 1)[1].strip()
    if not run_json_path or not comparison_json_path:
        return {
            "ok": False,
            "returncode": int(proc.returncode),
            "stdout": stdout,
            "stderr": stderr,
            "error": "missing output paths in compare stdout",
        }

    run_json = json.loads(Path(run_json_path).read_text(encoding="utf-8"))
    comparison_json = json.loads(Path(comparison_json_path).read_text(encoding="utf-8"))
    per_section = comparison_json.get("sections", {}) if isinstance(comparison_json, dict) else {}
    analysis_diag = run_json.get("analysis_diagnostics", {})
    run_mode = str(run_json.get("run_mode", "") or "")
    pipeline_ok = run_mode == "pipeline"
    failing_stage = ""
    if isinstance(analysis_diag, dict):
        analysis_probe = analysis_diag.get("analysis_probe", {})
        if isinstance(analysis_probe, dict):
            failing_stage = str(analysis_probe.get("failing_stage", "") or "")
    return {
        "ok": True,
        "run_json_path": run_json_path,
        "comparison_json_path": comparison_json_path,
        "run_mode": run_mode,
        "pipeline_ok": pipeline_ok,
        "run_note": str(run_json.get("run_note", "") or ""),
        "failing_stage": failing_stage,
        "matching_mode": str(run_json.get("matching_mode", matching_mode) or matching_mode),
        "matching_threshold": float(run_json.get("matching_threshold", matching_threshold) or matching_threshold),
        "runtime_seconds": float(run_json.get("runtime_seconds", 0.0) or 0.0),
        "overall_recall": float(comparison_json.get("overall_recall", 0.0) or 0.0),
        "section_recalls": {
            key: float((per_section.get(key, {}) or {}).get("recall", 0.0) or 0.0)
            for key in ("introduction", "methods", "results", "discussion", "conclusion")
        },
        "stdout": stdout,
        "stderr": stderr,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark compare pipeline across candidate LLM models.")
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--reference-md", required=True)
    parser.add_argument("--out-dir", default=str(ROOT / "test" / "text"))
    parser.add_argument("--db-dir", default="/tmp")
    parser.add_argument("--parser-engine", default="docling")
    parser.add_argument(
        "--backend-profile",
        default="balanced",
        choices=["fast", "balanced", "section-sensitive", "high-recall", "full"],
    )
    parser.add_argument(
        "--matching-mode",
        default="hybrid",
        choices=["lexical", "hybrid"],
        help="Matching strategy used by the compare step.",
    )
    parser.add_argument(
        "--matching-threshold",
        type=float,
        default=0.42,
        help="Minimum matching score for a point to count as matched.",
    )
    parser.add_argument(
        "--retain-runs",
        type=int,
        default=1,
        help="Number of compare run artifact sets to keep in out-dir (0 keeps all history).",
    )
    parser.add_argument(
        "--retain-benchmarks",
        type=int,
        default=1,
        help="Number of benchmark summary JSON files to keep per PDF (0 keeps all history).",
    )
    parser.add_argument(
        "--model",
        action="append",
        help="Model spec name=PATH. Can be passed multiple times. If omitted, defaults to Qwen3-8B and Qwen3-14B in models/.",
    )
    args = parser.parse_args()

    pdf = Path(args.pdf).expanduser().resolve()
    ref = Path(args.reference_md).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    db_dir = Path(args.db_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    db_dir.mkdir(parents=True, exist_ok=True)

    if not pdf.exists():
        raise SystemExit(f"PDF not found: {pdf}")
    if not ref.exists():
        raise SystemExit(f"Reference markdown not found: {ref}")

    model_specs: list[tuple[str, Path]] = []
    if args.model:
        for raw in args.model:
            if "=" not in raw:
                raise SystemExit(f"Invalid --model spec: {raw}. Expected name=PATH")
            name, raw_path = raw.split("=", 1)
            path = Path(raw_path).expanduser().resolve()
            model_specs.append((name.strip(), path))
    else:
        default_models = [
            ("qwen3-8b", ROOT / "models" / "Qwen3-8B-Q4_K_M.gguf"),
            ("qwen3-14b", ROOT / "models" / "Qwen3-14B-Q4_K_M.gguf"),
        ]
        model_specs.extend((name, path.resolve()) for name, path in default_models)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    records: list[dict[str, Any]] = []
    for idx, (name, model_path) in enumerate(model_specs):
        if not model_path.exists():
            records.append(
                {
                    "model": name,
                    "model_path": str(model_path),
                    "ok": False,
                    "error": "model file not found",
                }
            )
            continue
        db_path = db_dir / f"paper_eval_benchmark_{name}_{stamp}_{idx}.db"
        result = _run_one(
            pdf=pdf,
            reference_md=ref,
            out_dir=out_dir,
            db_path=db_path,
            parser_engine=args.parser_engine,
            backend_profile=args.backend_profile,
            matching_mode=args.matching_mode,
            matching_threshold=float(args.matching_threshold),
            retain_runs=args.retain_runs,
            text_model_path=model_path,
            deep_model_path=model_path,
        )
        record = {
            "model": name,
            "model_path": str(model_path),
            **result,
        }
        records.append(record)
        status = "ok" if record.get("ok") else "failed"
        print(f"{name}: {status}")
        if record.get("ok"):
            mode_label = str(record.get("run_mode", "unknown"))
            fidelity = "pipeline" if record.get("pipeline_ok") else f"fallback({record.get('failing_stage') or 'unknown'})"
            print(
                f"  recall={record['overall_recall']:.3f} runtime={record['runtime_seconds']:.2f}s "
                f"methods={record['section_recalls']['methods']:.3f} results={record['section_recalls']['results']:.3f} "
                f"mode={mode_label} fidelity={fidelity}"
            )
        else:
            print(f"  error={record.get('error') or 'non-zero return'}")

    successful = [r for r in records if r.get("ok")]
    best: dict[str, Any] | None = None
    if successful:
        pipeline_success = [r for r in successful if bool(r.get("pipeline_ok"))]
        ranking_pool = pipeline_success if pipeline_success else successful
        best = sorted(
            ranking_pool,
            key=lambda r: (-float(r.get("overall_recall", 0.0)), float(r.get("runtime_seconds", 0.0))),
        )[0]

    payload = {
        "timestamp_utc": stamp,
        "pdf": str(pdf),
        "reference_md": str(ref),
        "backend_profile": args.backend_profile,
        "matching_mode": args.matching_mode,
        "matching_threshold": float(args.matching_threshold),
        "parser_engine": args.parser_engine,
        "records": records,
        "best_model": best,
    }
    out_path = out_dir / f"{pdf.stem}_model_benchmark_{stamp}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pruned = _prune_benchmark_artifacts(out_dir, stem=pdf.stem, keep_latest=max(0, int(args.retain_benchmarks)))
    print(f"benchmark_json={out_path}")
    print(f"benchmarks_pruned={pruned}")
    if best:
        mode_note = "pipeline" if best.get("pipeline_ok") else "fallback_only"
        print(
            "best_model="
            f"{best['model']} recall={float(best.get('overall_recall', 0.0)):.3f} "
            f"runtime={float(best.get('runtime_seconds', 0.0)):.2f}s mode={mode_note}"
        )


if __name__ == "__main__":
    main()
