#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "benchmarks" / "multi_paper_benchmark.json"
COMPARE_SCRIPT = ROOT / "scripts" / "compare_pdf_against_reference.py"
GOLD_STANDARD_VALIDATOR = ROOT / "scripts" / "validate_gold_standards.py"
EVIDENCE_GOLD_COMPARATOR = ROOT / "scripts" / "compare_evidence_to_gold.py"
SECTION_KEYS = ("introduction", "methods", "results", "discussion", "conclusion")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"JSON file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected JSON object in {path}")
    return payload


def _resolve_path(value: str, *, root: Path = ROOT) -> Path:
    path = Path(str(value or "")).expanduser()
    if path.is_absolute():
        return path
    return root / path


def _case_reference_path(case: dict[str, Any]) -> Path | None:
    raw = str(case.get("reference_md") or "").strip()
    if not raw:
        return None
    return _resolve_path(raw)


def _case_gold_standard_path(case: dict[str, Any]) -> Path | None:
    raw = str(case.get("gold_standard") or "").strip()
    if not raw:
        return None
    return _resolve_path(raw)


def _validate_gold_standard(path: Path) -> list[str]:
    spec = importlib.util.spec_from_file_location("validate_gold_standards", GOLD_STANDARD_VALIDATOR)
    if spec is None or spec.loader is None:
        return [f"could not load gold-standard validator: {GOLD_STANDARD_VALIDATOR}"]
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        module.load_gold_standard(path)
    except Exception as exc:
        return [str(exc)]
    return []


def _load_script_module(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {module_name}: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_manifest(manifest: dict[str, Any], *, root: Path = ROOT) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if int(manifest.get("schema_version", 0) or 0) != 1:
        errors.append("schema_version must be 1")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        errors.append("manifest must define at least one case")
        return errors, warnings

    seen: set[str] = set()
    for index, raw_case in enumerate(cases):
        if not isinstance(raw_case, dict):
            errors.append(f"case {index} must be an object")
            continue
        case_id = str(raw_case.get("id") or "").strip()
        if not case_id:
            errors.append(f"case {index} is missing id")
            continue
        if case_id in seen:
            errors.append(f"duplicate case id: {case_id}")
        seen.add(case_id)

        pdf_path = _resolve_path(str(raw_case.get("pdf") or ""), root=root)
        if not pdf_path.exists():
            errors.append(f"{case_id}: PDF not found: {pdf_path}")

        gold_standard_path = _case_gold_standard_path(raw_case)
        gold_standard_status = str(raw_case.get("gold_standard_status") or "needed").strip()
        gold_standard_required = gold_standard_status in {
            "available",
            "reviewed_gold_standard",
            "reviewed_reference_available",
        }
        if gold_standard_path is None:
            if gold_standard_required:
                errors.append(f"{case_id}: missing gold_standard")
        elif not gold_standard_path.exists():
            errors.append(f"{case_id}: gold standard not found: {gold_standard_path}")
        else:
            for gold_error in _validate_gold_standard(gold_standard_path):
                errors.append(f"{case_id}: {gold_error}")
            try:
                gold_payload = _read_json(gold_standard_path)
            except SystemExit as exc:
                errors.append(f"{case_id}: {exc}")
            else:
                if str(gold_payload.get("case_id") or "") != case_id:
                    errors.append(f"{case_id}: gold_standard case_id mismatch in {gold_standard_path}")

        scoring = str(raw_case.get("scoring") or "").strip()
        reference_path = _case_reference_path(raw_case)
        if scoring == "reference_comparison":
            if reference_path is None:
                errors.append(f"{case_id}: reference_comparison requires reference_md")
            elif not reference_path.exists():
                errors.append(f"{case_id}: reference markdown not found: {reference_path}")
        elif scoring == "diagnostic_coverage":
            if reference_path is not None and not reference_path.exists():
                warnings.append(f"{case_id}: unscored reference path is missing: {reference_path}")
        else:
            errors.append(f"{case_id}: unsupported scoring mode `{scoring}`")

        tiers = raw_case.get("tiers")
        if not isinstance(tiers, list) or not tiers:
            errors.append(f"{case_id}: tiers must be a non-empty list")
        if not isinstance(raw_case.get("processing_stages"), list) or not raw_case.get("processing_stages"):
            warnings.append(f"{case_id}: processing_stages not declared")
        if not isinstance(raw_case.get("paper_components"), list) or not raw_case.get("paper_components"):
            warnings.append(f"{case_id}: paper_components not declared")

    return errors, warnings


def validate_tier_selection(
    manifest: dict[str, Any],
    selected: list[dict[str, Any]],
    *,
    tier: str,
    include_unscored: bool,
    allow_diagnostic_tier_gap: bool = False,
) -> tuple[list[str], list[str]]:
    if tier == "all":
        return [], []
    tiers = manifest.get("tiers", {}) if isinstance(manifest.get("tiers"), dict) else {}
    tier_spec = tiers.get(tier, {}) if isinstance(tiers.get(tier), dict) else {}
    if not tier_spec:
        return [], []

    errors: list[str] = []
    warnings: list[str] = []
    selected_count = len(selected)
    reference_scored_count = sum(1 for case in selected if str(case.get("scoring")) == "reference_comparison")
    minimum_cases = int(tier_spec.get("minimum_cases", 0) or 0)
    required_reference_cases = int(tier_spec.get("requires_reference_scored_cases", 0) or 0)
    target_reference_cases = int(tier_spec.get("target_reference_scored_cases", 0) or 0)

    if selected_count < minimum_cases:
        message = f"{tier}: selected {selected_count} case(s), minimum_cases is {minimum_cases}"
        if allow_diagnostic_tier_gap:
            warnings.append(message)
        else:
            errors.append(message)
    if reference_scored_count < required_reference_cases:
        message = (
            f"{tier}: selected {reference_scored_count} reference-scored case(s), "
            f"requires_reference_scored_cases is {required_reference_cases}"
        )
        if allow_diagnostic_tier_gap:
            warnings.append(message)
        else:
            errors.append(message)
    if target_reference_cases and reference_scored_count < target_reference_cases:
        warnings.append(
            f"{tier}: selected {reference_scored_count} reference-scored case(s), "
            f"target_reference_scored_cases is {target_reference_cases}"
        )
    if not include_unscored and selected_count < minimum_cases:
        warnings.append(f"{tier}: pass --include-unscored to validate diagnostic-coverage cases")

    required_domains_any = tier_spec.get("required_domains_any", [])
    if isinstance(required_domains_any, list):
        selected_domains = {
            str(domain)
            for case in selected
            for domain in case.get("domains", [])
            if isinstance(domain, str)
        }
        missing_domain_groups = [
            [str(domain) for domain in group]
            for group in required_domains_any
            if isinstance(group, list) and not (selected_domains & {str(domain) for domain in group})
        ]
        if missing_domain_groups:
            errors.append(f"{tier}: required domain mix missing groups: {missing_domain_groups}")

    return errors, warnings


def select_cases(
    manifest: dict[str, Any],
    *,
    tier: str,
    case_ids: list[str] | None = None,
    include_unscored: bool = False,
) -> list[dict[str, Any]]:
    cases = [case for case in manifest.get("cases", []) if isinstance(case, dict)]
    requested = set(case_ids or [])
    if requested:
        selected = [case for case in cases if str(case.get("id")) in requested]
        missing = sorted(requested - {str(case.get("id")) for case in selected})
        if missing:
            raise SystemExit(f"Unknown benchmark case id(s): {', '.join(missing)}")
    elif tier == "all":
        selected = cases
    else:
        selected = [case for case in cases if tier in [str(item) for item in case.get("tiers", [])]]

    if include_unscored:
        return selected
    return [case for case in selected if str(case.get("scoring")) == "reference_comparison"]


def coverage_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    stages: dict[str, int] = {}
    components: dict[str, int] = {}
    domains: dict[str, int] = {}
    for case in cases:
        for key, target in (
            ("processing_stages", stages),
            ("paper_components", components),
            ("domains", domains),
        ):
            values = case.get(key, [])
            if not isinstance(values, list):
                continue
            for raw in values:
                value = str(raw).strip()
                if value:
                    target[value] = target.get(value, 0) + 1
    return {
        "processing_stages": dict(sorted(stages.items())),
        "paper_components": dict(sorted(components.items())),
        "domains": dict(sorted(domains.items())),
    }


def known_gaps(manifest: dict[str, Any], *, tier: str) -> list[dict[str, Any]]:
    gaps = manifest.get("known_gaps", [])
    if not isinstance(gaps, list):
        return []
    if tier == "all":
        return [gap for gap in gaps if isinstance(gap, dict)]
    selected: list[dict[str, Any]] = []
    for gap in gaps:
        if not isinstance(gap, dict):
            continue
        needed_for = gap.get("needed_for", [])
        if isinstance(needed_for, list) and tier in [str(item) for item in needed_for]:
            selected.append(gap)
    return selected


def _python_executable(raw: str | None) -> str:
    if raw:
        return raw
    venv_python = ROOT / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def _command_for_case(
    case: dict[str, Any],
    *,
    manifest: dict[str, Any],
    python_executable: str,
    mode: str | None,
    parser_engine: str | None,
    backend_profile: str | None,
    matching_mode: str | None,
    matching_threshold: float | None,
    retain_runs: int | None,
    out_dir: Path | None,
    db_dir: Path | None,
    stamp: str,
) -> list[str]:
    defaults = manifest.get("defaults", {}) if isinstance(manifest.get("defaults"), dict) else {}
    case_id = str(case.get("id"))
    pdf = _resolve_path(str(case.get("pdf") or ""))
    reference = _case_reference_path(case)
    if reference is None:
        raise ValueError(f"{case_id} has no reference markdown")
    gold_standard = _case_gold_standard_path(case)
    resolved_out_dir = out_dir or _resolve_path(str(defaults.get("out_dir") or "test/multi_paper_benchmark"))
    resolved_db_dir = db_dir or Path(str(defaults.get("db_dir") or "/tmp")).expanduser()
    db_path = resolved_db_dir / f"papereval_multi_{case_id}_{stamp}.db"
    command = [
        python_executable,
        str(COMPARE_SCRIPT),
        "--mode",
        str(mode or defaults.get("mode") or "pipeline"),
        "--parser-engine",
        str(parser_engine or defaults.get("parser_engine") or "validated"),
        "--backend-profile",
        str(backend_profile or defaults.get("backend_profile") or "section-sensitive"),
        "--pdf",
        str(pdf),
        "--reference-md",
        str(reference),
        "--out-dir",
        str(resolved_out_dir),
        "--retain-runs",
        str(retain_runs if retain_runs is not None else defaults.get("retain_runs", 1)),
        "--matching-mode",
        str(matching_mode or defaults.get("matching_mode") or "hybrid"),
        "--matching-threshold",
        str(matching_threshold if matching_threshold is not None else defaults.get("matching_threshold", 0.42)),
        "--db-path",
        str(db_path),
    ]
    if gold_standard is not None:
        command.extend(["--gold-standard-json", str(gold_standard)])
    return command


def _parse_compare_stdout(stdout: str) -> dict[str, str]:
    paths: dict[str, str] = {}
    for raw in stdout.splitlines():
        line = raw.strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key in {"run_json", "comparison_json", "app_md", "comparison_md", "information_retention_json"}:
            paths[key] = value.strip()
    return paths


def _child_output_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


def _run_child_command(command: list[str], *, timeout_seconds: float | None) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            command,
            cwd=str(ROOT),
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        timeout_text = f"case command timed out after {timeout_seconds:g}s" if timeout_seconds else "case command timed out"
        return {
            "returncode": 124,
            "stdout": _child_output_text(exc.stdout),
            "stderr": "\n".join(part for part in [_child_output_text(exc.stderr), timeout_text] if part),
            "timed_out": True,
            "timeout_seconds": timeout_seconds,
        }
    return {
        "returncode": int(proc.returncode),
        "stdout": proc.stdout or "",
        "stderr": proc.stderr or "",
        "timed_out": False,
        "timeout_seconds": timeout_seconds,
    }


def _build_evidence_gold_compatibility(case: dict[str, Any], run_json: dict[str, Any]) -> dict[str, Any]:
    gold_standard = _case_gold_standard_path(case)
    if gold_standard is None:
        return {"available": False, "compatible": False, "reason": "missing gold_standard in manifest case"}
    if not gold_standard.exists():
        return {"available": False, "compatible": False, "reason": f"gold standard not found: {gold_standard}"}
    summary_json = run_json.get("summary_json")
    if not isinstance(summary_json, dict):
        return {"available": False, "compatible": False, "reason": "run_json is missing summary_json"}
    try:
        comparator = _load_script_module(EVIDENCE_GOLD_COMPARATOR, "compare_evidence_to_gold")
        gold_validator = _load_script_module(GOLD_STANDARD_VALIDATOR, "validate_gold_standards_for_evidence")
        gold_payload = gold_validator.load_gold_standard(gold_standard)
        packets = comparator.evidence_packets_from_payload(summary_json)
        metadata = comparator.evidence_metadata_from_payload(summary_json)
        comparison = comparator.compare_evidence_to_gold(packets, gold_payload, evidence_metadata=metadata)
    except Exception as exc:
        return {"available": False, "compatible": False, "reason": str(exc)}
    return {"available": True, **comparison}


def _evidence_gold_failure_message(payload: dict[str, Any]) -> str:
    if not payload.get("available"):
        return f"evidence/gold compatibility unavailable: {payload.get('reason') or 'unknown reason'}"
    reasons = payload.get("failure_reasons", [])
    reason_text = ""
    if isinstance(reasons, list) and reasons:
        reason_text = f"; reasons: {', '.join(str(reason) for reason in reasons[:5])}"
    gaps = payload.get("schema_gaps", [])
    gap_text = f"; schema gaps: {', '.join(gaps)}" if isinstance(gaps, list) and gaps else ""
    requirement_gaps = payload.get("claim_requirement_gaps", [])
    requirement_gap_text = ""
    if isinstance(requirement_gaps, list) and requirement_gaps:
        first_gap = requirement_gaps[0] if isinstance(requirement_gaps[0], dict) else {}
        gap_parts = [
            f"claim={first_gap.get('claim_id') or 'unknown'}",
            f"entities={','.join(str(item) for item in first_gap.get('missing_entities', [])[:3])}",
            f"numbers={','.join(str(item) for item in first_gap.get('missing_numbers', [])[:3])}",
            f"detail_types={','.join(str(item) for item in first_gap.get('missing_detail_types', [])[:3])}",
        ]
        requirement_gap_text = f"; first requirement gap: {'; '.join(gap_parts)}"
    synthesis_diagnostics = payload.get("synthesis_evidence_diagnostics", {})
    synthesis_text = ""
    if isinstance(synthesis_diagnostics, dict):
        critical_missing = synthesis_diagnostics.get("critical_missing_focus_slots", [])
        if isinstance(critical_missing, list) and critical_missing:
            labels = [
                str(slot.get("label") or slot.get("slot_key") or "").strip()
                for slot in critical_missing
                if isinstance(slot, dict) and str(slot.get("label") or slot.get("slot_key") or "").strip()
            ]
            if labels:
                synthesis_text = f"; critical synthesis gaps: {', '.join(labels[:4])}"
        packet_coverage = synthesis_diagnostics.get("evidence_packet_coverage", {})
        if isinstance(packet_coverage, dict) and packet_coverage.get("available"):
            coverage_parts: list[str] = []
            missing_sections = [
                str(section).strip()
                for section in packet_coverage.get("missing_core_sections", [])
                if str(section).strip()
            ]
            if missing_sections:
                coverage_parts.append(f"missing_sections={','.join(missing_sections[:4])}")
            cross_modal = int(packet_coverage.get("cross_modal_packet_count", 0) or 0)
            typed = int(packet_coverage.get("typed_packet_count", 0) or 0)
            if cross_modal == 0:
                coverage_parts.append("cross_modal_packets=0")
            if typed == 0:
                coverage_parts.append("typed_packets=0")
            if coverage_parts:
                synthesis_text += f"; packet coverage: {'; '.join(coverage_parts)}"
    return (
        "evidence/gold compatibility failed: "
        f"usable={float(payload.get('usable_packet_rate', 0.0) or 0.0):.3f}, "
        f"sections={float(payload.get('section_coverage_rate', 0.0) or 0.0):.3f}, "
        f"claims={float(payload.get('critical_claim_candidate_rate', 0.0) or 0.0):.3f}, "
        f"entities={float(payload.get('expected_entity_observability_rate', 0.0) or 0.0):.3f}, "
        f"numbers={float(payload.get('expected_number_observability_rate', 0.0) or 0.0):.3f}, "
        f"detail_types={float(payload.get('expected_detail_type_observability_rate', 0.0) or 0.0):.3f}"
        f"{reason_text}"
        f"{gap_text}"
        f"{requirement_gap_text}"
        f"{synthesis_text}"
    )


def _score_reference_case(
    *,
    case: dict[str, Any],
    manifest: dict[str, Any],
    paths: dict[str, str],
    returncode: int,
    stdout: str,
    stderr: str,
    timed_out: bool = False,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    case_id = str(case.get("id"))
    record: dict[str, Any] = {
        "case_id": case_id,
        "title": str(case.get("title") or case_id),
        "scoring": str(case.get("scoring") or ""),
        "returncode": int(returncode),
        "stdout_tail": "\n".join(stdout.splitlines()[-20:]),
        "stderr_tail": "\n".join(stderr.splitlines()[-20:]),
        "timed_out": bool(timed_out),
        "timeout_seconds": timeout_seconds,
        "artifact_paths": paths,
        "ok": False,
        "decision": "fail",
        "failures": [],
    }
    if timed_out:
        record["failures"].append(f"compare command timed out after {timeout_seconds:g}s")
        return record
    if returncode != 0:
        record["failures"].append(f"compare command exited {returncode}")
        return record
    comparison_path = paths.get("comparison_json")
    run_path = paths.get("run_json")
    if not comparison_path or not run_path:
        record["failures"].append("compare output did not include run_json and comparison_json paths")
        return record

    try:
        comparison = _read_json(Path(comparison_path))
        run_json = _read_json(Path(run_path))
    except SystemExit as exc:
        record["failures"].append(str(exc))
        return record

    sections = comparison.get("sections", {}) if isinstance(comparison.get("sections"), dict) else {}
    section_recalls = {
        key: float((sections.get(key, {}) or {}).get("recall", 0.0) or 0.0)
        for key in SECTION_KEYS
    }
    thresholds = manifest.get("reference_thresholds", {}) if isinstance(manifest.get("reference_thresholds"), dict) else {}
    overall_recall = float(comparison.get("overall_recall", 0.0) or 0.0)
    overall_min = float(thresholds.get("overall_recall_min", 0.68))
    methods_min = float(thresholds.get("methods_recall_min", 0.55))
    results_min = float(thresholds.get("results_recall_min", 0.55))
    section_floor = float(thresholds.get("section_recall_floor", 0.35))
    failures: list[str] = []
    if overall_recall < overall_min:
        failures.append(f"overall recall {overall_recall:.3f} < {overall_min:.3f}")
    if section_recalls["methods"] < methods_min:
        failures.append(f"methods recall {section_recalls['methods']:.3f} < {methods_min:.3f}")
    if section_recalls["results"] < results_min:
        failures.append(f"results recall {section_recalls['results']:.3f} < {results_min:.3f}")
    low_sections = [key for key, value in section_recalls.items() if value < section_floor]
    if low_sections:
        failures.append(f"section recall below floor for: {', '.join(low_sections)}")

    run_mode = str(run_json.get("run_mode") or "")
    if run_mode and run_mode != "pipeline":
        failures.append(f"run_mode is {run_mode}, not pipeline")
    evidence_gold_compatibility = _build_evidence_gold_compatibility(case, run_json)
    if not evidence_gold_compatibility.get("compatible"):
        failures.append(_evidence_gold_failure_message(evidence_gold_compatibility))

    record.update(
        {
            "ok": not failures,
            "decision": "pass" if not failures else "fail",
            "overall_recall": overall_recall,
            "section_recalls": section_recalls,
            "evidence_gold_compatibility": evidence_gold_compatibility,
            "runtime_seconds": float(run_json.get("runtime_seconds", 0.0) or 0.0),
            "run_mode": run_mode,
            "failures": failures,
        }
    )
    return record


def _case_plan(case: dict[str, Any], *, command: list[str] | None = None) -> dict[str, Any]:
    reference = _case_reference_path(case)
    gold_standard = _case_gold_standard_path(case)
    return {
        "case_id": str(case.get("id")),
        "title": str(case.get("title") or case.get("id")),
        "scoring": str(case.get("scoring") or ""),
        "reference_status": str(case.get("reference_status") or ""),
        "gold_standard": str(case.get("gold_standard") or ""),
        "gold_standard_exists": bool(gold_standard and gold_standard.exists()),
        "gold_standard_status": str(case.get("gold_standard_status") or ""),
        "pdf_exists": _resolve_path(str(case.get("pdf") or "")).exists(),
        "reference_exists": bool(reference and reference.exists()),
        "tiers": case.get("tiers", []),
        "domains": case.get("domains", []),
        "processing_stages": case.get("processing_stages", []),
        "paper_components": case.get("paper_components", []),
        "expected_stressors": case.get("expected_stressors", []),
        "command": command or [],
    }


def _write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate or execute the PsychPaperEvalApp multi-paper benchmark manifest.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--tier", default="smoke", choices=["smoke", "release", "deep", "all"])
    parser.add_argument("--case", action="append", dest="case_ids", help="Run or list one case id. Can be passed multiple times.")
    parser.add_argument("--include-unscored", action="store_true", help="Include diagnostic-coverage cases without reference markdown.")
    parser.add_argument("--list", action="store_true", help="List selected cases and exit.")
    parser.add_argument("--validate-only", action="store_true", help="Validate manifest and selected cases without executing.")
    parser.add_argument("--execute", action="store_true", help="Execute reference-scored cases.")
    parser.add_argument("--summary-json", default="", help="Optional path for benchmark summary JSON.")
    parser.add_argument("--python", default="", help="Python executable for child compare runs.")
    parser.add_argument("--mode", choices=["auto", "pipeline", "lightweight"], default=None)
    parser.add_argument("--parser-engine", default=None)
    parser.add_argument("--backend-profile", choices=["fast", "balanced", "section-sensitive", "high-recall", "full"], default=None)
    parser.add_argument("--matching-mode", choices=["lexical", "hybrid"], default=None)
    parser.add_argument("--matching-threshold", type=float, default=None)
    parser.add_argument("--retain-runs", type=int, default=None)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--db-dir", default="")
    parser.add_argument("--timeout-per-case", type=float, default=1800.0, help="Seconds before a benchmark child run is failed as timed out.")
    parser.add_argument("--allow-diagnostic-tier-gap", action="store_true", help="Warn instead of failing when a benchmark tier is underfilled.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    manifest = _read_json(manifest_path)
    errors, warnings = validate_manifest(manifest)
    selected = select_cases(
        manifest,
        tier=args.tier,
        case_ids=args.case_ids,
        include_unscored=bool(args.include_unscored),
    )
    tier_errors, tier_warnings = validate_tier_selection(
        manifest,
        selected,
        tier=args.tier,
        include_unscored=bool(args.include_unscored),
        allow_diagnostic_tier_gap=bool(args.allow_diagnostic_tier_gap),
    )
    errors.extend(tier_errors)
    warnings.extend(tier_warnings)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    python_executable = _python_executable(args.python or None)
    out_dir = _resolve_path(args.out_dir) if args.out_dir else None
    db_dir = Path(args.db_dir).expanduser() if args.db_dir else None

    plans: list[dict[str, Any]] = []
    for case in selected:
        command: list[str] | None = None
        if str(case.get("scoring")) == "reference_comparison":
            command = _command_for_case(
                case,
                manifest=manifest,
                python_executable=python_executable,
                mode=args.mode,
                parser_engine=args.parser_engine,
                backend_profile=args.backend_profile,
                matching_mode=args.matching_mode,
                matching_threshold=args.matching_threshold,
                retain_runs=args.retain_runs,
                out_dir=out_dir,
                db_dir=db_dir,
                stamp=stamp,
            )
        plans.append(_case_plan(case, command=command))

    summary: dict[str, Any] = {
        "schema_version": 1,
        "benchmark_id": str(manifest.get("benchmark_id") or ""),
        "manifest": str(manifest_path),
        "tier": args.tier,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "errors": errors,
        "warnings": warnings,
        "selected_case_count": len(selected),
        "coverage": coverage_summary(selected),
        "known_gaps": known_gaps(manifest, tier=args.tier),
        "plans": plans,
        "records": [],
    }

    if args.list or args.validate_only or not args.execute:
        for plan in plans:
            status = "scored" if plan["scoring"] == "reference_comparison" else "needs-reference"
            gold_status = str(plan.get("gold_standard_status") or "unknown")
            print(f"{plan['case_id']}: {status}; gold={gold_status} | {', '.join(plan.get('domains', []))}")
        coverage = summary["coverage"]
        print(f"processing_stages={','.join(coverage['processing_stages'].keys())}")
        print(f"paper_components={','.join(coverage['paper_components'].keys())}")
        for gap in summary["known_gaps"]:
            print(f"known_gap={gap.get('id', '')}: {gap.get('reason', '')}")
        if errors:
            for error in errors:
                print(f"ERROR: {error}", file=sys.stderr)
        for warning in warnings:
            print(f"WARNING: {warning}", file=sys.stderr)
        if args.summary_json:
            _write_summary(_resolve_path(args.summary_json), summary)
        return 1 if errors else 0

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    records: list[dict[str, Any]] = []
    for case, plan in zip(selected, plans):
        if str(case.get("scoring")) != "reference_comparison":
            records.append(
                {
                    "case_id": str(case.get("id")),
                    "ok": False,
                    "decision": "skipped",
                    "reason": "diagnostic_coverage case has no reference markdown yet",
                }
            )
            continue
        command = list(plan.get("command", []))
        print(f"running {case['id']}...")
        child = _run_child_command(command, timeout_seconds=float(args.timeout_per_case))
        paths = _parse_compare_stdout(str(child.get("stdout") or ""))
        record = _score_reference_case(
            case=case,
            manifest=manifest,
            paths=paths,
            returncode=int(child.get("returncode", 1) or 1),
            stdout=str(child.get("stdout") or ""),
            stderr=str(child.get("stderr") or ""),
            timed_out=bool(child.get("timed_out")),
            timeout_seconds=child.get("timeout_seconds"),
        )
        records.append(record)
        status = "pass" if record.get("ok") else str(record.get("decision") or "fail")
        detail = f" recall={float(record.get('overall_recall', 0.0) or 0.0):.3f}" if "overall_recall" in record else ""
        print(f"  {status}{detail}")

    summary["records"] = records
    passed = sum(1 for record in records if record.get("ok"))
    failed = sum(1 for record in records if record.get("decision") == "fail")
    skipped = sum(1 for record in records if record.get("decision") == "skipped")
    summary["totals"] = {"passed": passed, "failed": failed, "skipped": skipped, "executed": len(records) - skipped}
    if args.summary_json:
        summary_path = _resolve_path(args.summary_json)
    else:
        default_out = out_dir or _resolve_path(str((manifest.get("defaults") or {}).get("out_dir") or "test/multi_paper_benchmark"))
        summary_path = default_out / f"multi_paper_benchmark_{stamp}.json"
    _write_summary(summary_path, summary)
    print(f"summary_json={summary_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
