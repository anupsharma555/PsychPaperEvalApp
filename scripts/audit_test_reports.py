#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import mimetypes
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Any
import urllib.error
import urllib.request
import uuid


ROOT = Path(__file__).resolve().parents[1]


class ApiError(RuntimeError):
    def __init__(self, status_code: int, payload: Any):
        self.status_code = status_code
        self.payload = payload
        super().__init__(f"API error {status_code}: {payload}")


def _json_loads_maybe(payload: bytes) -> Any:
    text = payload.decode("utf-8", errors="replace")
    try:
        return json.loads(text)
    except Exception:
        return text


def api_json(
    url: str,
    *,
    method: str = "GET",
    timeout: float = 20.0,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> Any:
    req = urllib.request.Request(url=url, method=method, data=body)
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return _json_loads_maybe(response.read())
    except urllib.error.HTTPError as exc:
        payload = _json_loads_maybe(exc.read())
        raise ApiError(exc.code, payload) from exc


def encode_upload_multipart(file_path: Path, field_name: str = "main_file") -> tuple[bytes, str]:
    boundary = f"----PaperEvalBoundary{uuid.uuid4().hex}"
    filename = file_path.name
    content_type = mimetypes.guess_type(filename)[0] or "application/pdf"
    file_bytes = file_path.read_bytes()
    parts = [
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'.encode(),
        f"Content-Type: {content_type}\r\n\r\n".encode(),
        file_bytes,
        b"\r\n",
        f"--{boundary}--\r\n".encode(),
    ]
    return b"".join(parts), boundary


def wait_backend_ready(api_base: str, timeout_sec: int = 60) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            payload = api_json(f"{api_base}/status", timeout=2.0)
            if isinstance(payload, dict) and payload.get("backend_ready") is True:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def terminate_process_tree(proc: subprocess.Popen[Any]) -> None:
    try:
        pgid = os.getpgid(proc.pid)
    except Exception:
        pgid = None
    if isinstance(pgid, int) and pgid > 0:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except Exception:
            pass
        time.sleep(0.4)
        try:
            os.killpg(pgid, signal.SIGKILL)
        except Exception:
            pass
        return
    try:
        proc.terminate()
    except Exception:
        return
    time.sleep(0.4)
    try:
        proc.kill()
    except Exception:
        return


def _result(ok: bool, detail: str) -> dict[str, Any]:
    return {"ok": bool(ok), "detail": detail}


def _norm_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _extract_report_text(summary_json: dict[str, Any]) -> dict[str, Any]:
    executive_summary = _norm_text(summary_json.get("executive_summary"))
    methods_rows = summary_json.get("methods_compact", [])
    methods_statements: list[str] = []
    if isinstance(methods_rows, list):
        for row in methods_rows:
            if not isinstance(row, dict):
                continue
            statement = _norm_text(row.get("statement"))
            if statement:
                methods_statements.append(statement)

    section_rows = summary_json.get("sections_compact", {})
    section_statements: list[str] = []
    if isinstance(section_rows, dict):
        for key in ["introduction", "methods", "results", "discussion", "conclusion"]:
            rows = section_rows.get(key, [])
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                statement = _norm_text(row.get("statement"))
                if statement:
                    section_statements.append(statement)

    all_text_parts = [executive_summary] + methods_statements + section_statements
    all_text = " ".join(part for part in all_text_parts if part).strip()
    return {
        "executive_summary": executive_summary,
        "methods_statements": methods_statements,
        "section_statements": section_statements,
        "all_text": all_text,
    }


def _safe_ratio(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return num / den


def _grade(score: int) -> str:
    if score >= 90:
        return "excellent"
    if score >= 75:
        return "good"
    if score >= 60:
        return "fair"
    return "needs_improvement"


def _score_content_quality(summary_json: dict[str, Any], findings: Any) -> dict[str, Any]:
    extracted = _extract_report_text(summary_json)
    text = extracted["all_text"]
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9'-]*", text)
    token_count = len(tokens)
    unique_count = len({tok.lower() for tok in tokens})
    lexical_diversity = _safe_ratio(unique_count, token_count)

    raw_sentences = [seg.strip() for seg in re.split(r"[.!?]+", text) if seg.strip()]
    sentence_count = len(raw_sentences)
    avg_sentence_tokens = _safe_ratio(token_count, sentence_count)
    unique_sentences = len({s.lower() for s in raw_sentences})
    duplicate_sentence_ratio = 1.0 - _safe_ratio(unique_sentences, sentence_count)

    placeholder_hits = 0
    placeholder_patterns = [
        r"\bn/a\b",
        r"\bnot found\b",
        r"\baccess-limited\b",
        r"\bplaceholder\b",
        r"\blorem ipsum\b",
        r"\btbd\b",
    ]
    lowered = text.lower()
    for pattern in placeholder_patterns:
        placeholder_hits += len(re.findall(pattern, lowered))

    sensical_score = 100
    sensical_notes: list[str] = []
    if token_count < 80:
        sensical_score -= 30
        sensical_notes.append("Very little report text extracted.")
    elif token_count < 150:
        sensical_score -= 10
        sensical_notes.append("Limited report text; interpretation may be shallow.")
    if lexical_diversity < 0.25:
        sensical_score -= 20
        sensical_notes.append("Low lexical diversity suggests repetitive content.")
    if avg_sentence_tokens < 4 or avg_sentence_tokens > 45:
        sensical_score -= 15
        sensical_notes.append("Sentence lengths are atypical, reducing readability.")
    if duplicate_sentence_ratio > 0.35:
        sensical_score -= 15
        sensical_notes.append("High repetition across report sentences.")
    if placeholder_hits >= 3:
        sensical_score -= 20
        sensical_notes.append("Many placeholder/access-limited statements present.")
    elif placeholder_hits >= 1:
        sensical_score -= 10
        sensical_notes.append("Some placeholder/access-limited statements present.")
    if isinstance(findings, list) and len(findings) == 0:
        sensical_score -= 10
        sensical_notes.append("No findings listed in detailed report.")
    sensical_score = max(0, min(100, sensical_score))

    return {
        "sensical_score": sensical_score,
        "sensical_grade": _grade(int(round(sensical_score))),
        "sensical_notes": sensical_notes,
        "text_metrics": {
            "token_count": token_count,
            "unique_token_count": unique_count,
            "lexical_diversity": round(lexical_diversity, 3),
            "sentence_count": sentence_count,
            "avg_sentence_tokens": round(avg_sentence_tokens, 2),
            "duplicate_sentence_ratio": round(duplicate_sentence_ratio, 3),
            "placeholder_hits": placeholder_hits,
        },
    }


def evaluate_report(
    report_payload: dict[str, Any],
    report_summary_payload: dict[str, Any],
    job_status: str,
) -> dict[str, Any]:
    summary_json = report_payload.get("summary_json")
    if not isinstance(summary_json, dict):
        summary_json = {}

    sections = summary_json.get("sections_compact")
    modalities = summary_json.get("modalities")
    methods_compact = summary_json.get("methods_compact")
    coverage_line = str(summary_json.get("coverage_snapshot_line", "") or "").strip()
    executive_summary = str(summary_json.get("executive_summary", "") or "").strip()
    findings = report_payload.get("findings")
    discrepancies = report_payload.get("discrepancies")
    diagnostics = report_payload.get("analysis_diagnostics")

    required = {
        "job_completed": _result(job_status == "completed", f"job status={job_status}"),
        "report_summary_ready": _result(
            report_summary_payload.get("report_status") == "ready",
            f"report_status={report_summary_payload.get('report_status')}",
        ),
        "summary_json_present": _result(bool(summary_json), f"summary_json keys={len(summary_json)}"),
        "executive_summary_present": _result(
            len(executive_summary) >= 40,
            f"executive_summary chars={len(executive_summary)}",
        ),
        "methods_compact_present": _result(
            isinstance(methods_compact, list) and len(methods_compact) > 0,
            (
                f"methods_compact count={len(methods_compact)}"
                if isinstance(methods_compact, list)
                else "methods_compact missing"
            ),
        ),
        "sections_compact_complete": _result(
            isinstance(sections, dict)
            and all(key in sections for key in ["introduction", "methods", "results", "discussion", "conclusion"]),
            (
                f"sections keys={sorted(sections.keys())[:10]}"
                if isinstance(sections, dict)
                else "sections_compact missing"
            ),
        ),
        "modalities_present": _result(
            isinstance(modalities, dict)
            and all(key in modalities for key in ["text", "table", "figure", "supplement"]),
            (
                f"modalities keys={sorted(modalities.keys())[:10]}"
                if isinstance(modalities, dict)
                else "modalities missing"
            ),
        ),
        "coverage_line_present": _result(bool(coverage_line), f"coverage_line chars={len(coverage_line)}"),
        "detailed_lists_present": _result(
            isinstance(findings, list) and isinstance(discrepancies, list),
            f"findings_type={type(findings).__name__}, discrepancies_type={type(discrepancies).__name__}",
        ),
        "diagnostics_present": _result(
            diagnostics is not None,
            f"analysis_diagnostics_type={type(diagnostics).__name__}",
        ),
    }

    quality_warnings: list[str] = []
    if isinstance(methods_compact, list):
        found_methods = sum(1 for row in methods_compact if isinstance(row, dict) and str(row.get("status", "")).lower() == "found")
        if found_methods == 0:
            quality_warnings.append("No methods_compact rows marked as status=found.")
    else:
        found_methods = 0

    section_found_count = 0
    if isinstance(sections, dict):
        for rows in sections.values():
            if not isinstance(rows, list):
                continue
            section_found_count += sum(1 for row in rows if isinstance(row, dict) and str(row.get("status", "")).lower() == "found")
        if section_found_count == 0:
            quality_warnings.append("No sections_compact rows marked as status=found.")
    else:
        section_found_count = 0

    if isinstance(report_summary_payload, dict) and report_summary_payload.get("rerun_recommended") is True:
        quality_warnings.append("report_summary indicates rerun_recommended=true.")

    if isinstance(findings, list) and len(findings) == 0:
        quality_warnings.append("Detailed report has zero findings entries.")

    passes_required = all(item["ok"] for item in required.values())
    required_count = len(required)
    required_pass_count = sum(1 for item in required.values() if item["ok"])
    required_pass_ratio = _safe_ratio(required_pass_count, required_count)

    methods_rows_count = len(methods_compact) if isinstance(methods_compact, list) else 0
    section_rows_count = 0
    if isinstance(sections, dict):
        for rows in sections.values():
            if isinstance(rows, list):
                section_rows_count += len(rows)

    completeness_score = int(round(
        min(100.0, (
            required_pass_ratio * 70.0
            + _safe_ratio(found_methods, max(1, methods_rows_count)) * 15.0
            + _safe_ratio(section_found_count, max(1, section_rows_count)) * 10.0
            + (5.0 if isinstance(findings, list) and len(findings) > 0 else 0.0)
        ))
    ))

    organization_score = 0
    if isinstance(sections, dict):
        ordered_keys = ["introduction", "methods", "results", "discussion", "conclusion"]
        present_count = sum(1 for key in ordered_keys if key in sections)
        organization_score += int(round(_safe_ratio(present_count, len(ordered_keys)) * 45))
    if isinstance(methods_compact, list) and len(methods_compact) > 0:
        organization_score += 20
    if section_found_count > 0:
        organization_score += min(20, section_found_count * 3)
    if found_methods > 0:
        organization_score += min(15, found_methods * 2)
    organization_score = max(0, min(100, organization_score))

    quality = _score_content_quality(summary_json, findings)
    sensical_score = int(round(quality["sensical_score"]))
    overall_score = int(round(
        completeness_score * 0.5
        + organization_score * 0.25
        + sensical_score * 0.25
    ))

    improvement_points: list[str] = []
    for check_name, check in required.items():
        if not check["ok"]:
            improvement_points.append(f"Fix `{check_name}`: {check['detail']}")
    if sensical_score < 70:
        improvement_points.append("Improve report clarity/coherence; reduce repetitive or placeholder wording.")
    if organization_score < 70:
        improvement_points.append("Improve section organization and ensure key section slots are filled with concrete statements.")
    if isinstance(findings, list) and len(findings) == 0:
        improvement_points.append("Increase extraction quality so the detailed report includes concrete findings.")
    if isinstance(report_summary_payload, dict) and report_summary_payload.get("rerun_recommended") is True:
        improvement_points.append("Report flags rerun recommended; review extraction failures and rerun pipeline.")

    return {
        "required_checks": required,
        "passes_required": passes_required,
        "quality_warnings": quality_warnings,
        "scores": {
            "completeness": completeness_score,
            "organization": organization_score,
            "sensical_content": sensical_score,
            "overall": overall_score,
            "overall_grade": _grade(overall_score),
        },
        "improvement_points": improvement_points,
        "sensical_notes": quality["sensical_notes"],
        "derived_metrics": {
            "methods_found_count": found_methods,
            "methods_rows_count": methods_rows_count,
            "sections_found_count": section_found_count,
            "sections_rows_count": section_rows_count,
            "executive_summary_chars": len(executive_summary),
            "findings_count": len(findings) if isinstance(findings, list) else None,
            "discrepancies_count": len(discrepancies) if isinstance(discrepancies, list) else None,
            **quality["text_metrics"],
        },
    }


def poll_job_until_done(api_base: str, job_id: int, timeout_sec: int, poll_sec: float) -> dict[str, Any]:
    deadline = time.time() + timeout_sec
    last_payload: dict[str, Any] | None = None
    while time.time() < deadline:
        payload = api_json(f"{api_base}/jobs/{job_id}", timeout=10.0)
        if not isinstance(payload, dict):
            raise RuntimeError(f"Unexpected /jobs payload for job {job_id}: {payload!r}")
        last_payload = payload
        status = str(payload.get("status", ""))
        if status in {"completed", "failed"}:
            return payload
        time.sleep(poll_sec)
    raise TimeoutError(f"Timed out waiting for job {job_id}. Last payload: {last_payload}")


@dataclass
class BackendHandle:
    process: subprocess.Popen[Any] | None
    started_here: bool


def ensure_backend(args: argparse.Namespace) -> BackendHandle:
    if wait_backend_ready(args.api_base, timeout_sec=4):
        return BackendHandle(process=None, started_here=False)
    if args.no_start_backend:
        raise SystemExit("Backend not ready and --no-start-backend was set.")

    py = ROOT / ".venv" / "bin" / "python"
    if not py.exists():
        py = Path(sys.executable)
    cmd = [str(py), "scripts/run_app.py", "--api-only", "--backend-port", str(args.backend_port)]
    if args.force_start:
        cmd.append("--force")
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if not wait_backend_ready(args.api_base, timeout_sec=90):
        terminate_process_tree(proc)
        raise SystemExit("Backend did not become ready in time.")
    return BackendHandle(process=proc, started_here=True)


def maybe_stop_backend(api_base: str, handle: BackendHandle, keep_running: bool) -> None:
    if not handle.started_here or keep_running:
        return
    try:
        api_json(f"{api_base}/stop", method="POST", timeout=5.0)
    except Exception:
        pass
    if handle.process is None:
        return
    deadline = time.time() + 10
    while time.time() < deadline:
        if handle.process.poll() is not None:
            return
        time.sleep(0.2)
    terminate_process_tree(handle.process)


def write_outputs(
    out_dir: Path,
    run_payload: dict[str, Any],
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    json_path = out_dir / f"report_audit_{stamp}.json"
    md_path = out_dir / f"report_audit_{stamp}.md"

    json_path.write_text(json.dumps(run_payload, indent=2))

    lines: list[str] = []
    lines.append("# Report Audit")
    lines.append("")
    lines.append(f"- Generated UTC: `{run_payload['generated_utc']}`")
    lines.append(f"- PDF directory: `{run_payload['pdf_dir']}`")
    lines.append(f"- Total processed: `{run_payload['summary']['total_processed']}`")
    lines.append(f"- Required-check pass: `{run_payload['summary']['passes_required']}`")
    lines.append(f"- Required-check fail: `{run_payload['summary']['fails_required']}`")
    lines.append(f"- Avg overall score: `{run_payload['summary']['avg_overall_score']}`")
    lines.append(f"- Avg completeness score: `{run_payload['summary']['avg_completeness_score']}`")
    lines.append(f"- Avg organization score: `{run_payload['summary']['avg_organization_score']}`")
    lines.append(f"- Avg sensical-content score: `{run_payload['summary']['avg_sensical_score']}`")
    lines.append("")
    lines.append("## Per Document")
    lines.append("")
    lines.append("| PDF | Job | Required | Overall | Completeness | Organization | Sensical | Warnings |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for row in run_payload["documents"]:
        required_state = "PASS" if row.get("passes_required") else "FAIL"
        scores = row.get("scores", {})
        warnings = "; ".join(row.get("quality_warnings", [])) or "-"
        lines.append(
            f"| `{row['pdf_name']}` | `{row.get('job_status', '-')}` | `{required_state}` "
            f"| `{scores.get('overall', '-')}` ({scores.get('overall_grade', '-')}) "
            f"| `{scores.get('completeness', '-')}` "
            f"| `{scores.get('organization', '-')}` "
            f"| `{scores.get('sensical_content', '-')}` "
            f"| {warnings} |"
        )

        failed_checks = [name for name, info in row.get("required_checks", {}).items() if not info.get("ok")]
        improvements = row.get("improvement_points", [])
        if failed_checks:
            lines.append("")
            lines.append(f"- `{row['pdf_name']}` failed checks: {', '.join(failed_checks)}")
            for name in failed_checks:
                detail = row["required_checks"][name].get("detail", "")
                lines.append(f"  - `{name}`: {detail}")
            lines.append("")
        if improvements:
            lines.append(f"- `{row['pdf_name']}` improvement points:")
            for point in improvements:
                lines.append(f"  - {point}")
            lines.append("")

    md_path.write_text("\n".join(lines).rstrip() + "\n")
    return json_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run test PDFs through PaperEval and audit whether reports are properly filled.",
    )
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/api")
    parser.add_argument("--backend-port", type=int, default=8000)
    parser.add_argument("--force-start", action="store_true", help="Start backend with --force.")
    parser.add_argument("--no-start-backend", action="store_true", help="Do not auto-start backend.")
    parser.add_argument("--keep-running", action="store_true", help="Do not stop backend if started by this script.")
    parser.add_argument("--timeout-per-doc", type=int, default=1800, help="Job timeout in seconds per PDF.")
    parser.add_argument("--poll-sec", type=float, default=2.0, help="Polling interval for job status.")
    parser.add_argument("--max-docs", type=int, default=None, help="Optional cap on number of PDFs to process.")
    parser.add_argument("--pdf-dir", default=str(ROOT / "test"))
    parser.add_argument("--out-dir", default=str(ROOT / "test" / "report_audit"))
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir).expanduser().resolve()
    if not pdf_dir.exists():
        raise SystemExit(f"PDF directory not found: {pdf_dir}")
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if args.max_docs is not None:
        pdfs = pdfs[: max(args.max_docs, 0)]
    if not pdfs:
        raise SystemExit(f"No PDFs found in {pdf_dir}")

    handle = ensure_backend(args)
    docs: list[dict[str, Any]] = []
    started_at = time.time()
    try:
        for pdf_path in pdfs:
            record: dict[str, Any] = {
                "pdf_name": pdf_path.name,
                "pdf_path": str(pdf_path),
            }
            doc_start = time.time()
            try:
                body, boundary = encode_upload_multipart(pdf_path, field_name="main_file")
                upload = api_json(
                    f"{args.api_base}/documents/upload",
                    method="POST",
                    timeout=120.0,
                    body=body,
                    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
                )
                if not isinstance(upload, dict):
                    raise RuntimeError(f"Unexpected upload response: {upload!r}")
                document_id = int(upload["document_id"])
                job_id = int(upload["job_id"])
                record["document_id"] = document_id
                record["job_id"] = job_id

                job = poll_job_until_done(
                    args.api_base,
                    job_id=job_id,
                    timeout_sec=args.timeout_per_doc,
                    poll_sec=args.poll_sec,
                )
                job_status = str(job.get("status", ""))
                record["job_status"] = job_status
                record["job_progress"] = job.get("progress")
                record["job_message"] = job.get("message")

                report = api_json(f"{args.api_base}/documents/{document_id}/report", timeout=20.0)
                report_summary = api_json(f"{args.api_base}/documents/{document_id}/report/summary", timeout=20.0)
                if not isinstance(report, dict) or not isinstance(report_summary, dict):
                    raise RuntimeError("Unexpected report payloads.")

                quality = evaluate_report(report, report_summary, job_status=job_status)
                record.update(quality)
            except Exception as exc:
                record["job_status"] = record.get("job_status", "failed")
                record["passes_required"] = False
                record["required_checks"] = {
                    "pipeline_execution": _result(False, str(exc)),
                }
                record["quality_warnings"] = ["Execution failed before report evaluation."]
                record["derived_metrics"] = {}
            finally:
                record["duration_sec"] = round(time.time() - doc_start, 2)
                docs.append(record)
    finally:
        maybe_stop_backend(args.api_base, handle, args.keep_running)

    passes_required = sum(1 for row in docs if row.get("passes_required"))
    overall_scores = [int(row.get("scores", {}).get("overall", 0)) for row in docs if isinstance(row.get("scores"), dict)]
    completeness_scores = [int(row.get("scores", {}).get("completeness", 0)) for row in docs if isinstance(row.get("scores"), dict)]
    organization_scores = [int(row.get("scores", {}).get("organization", 0)) for row in docs if isinstance(row.get("scores"), dict)]
    sensical_scores = [int(row.get("scores", {}).get("sensical_content", 0)) for row in docs if isinstance(row.get("scores"), dict)]
    summary = {
        "total_processed": len(docs),
        "passes_required": passes_required,
        "fails_required": len(docs) - passes_required,
        "avg_overall_score": round(sum(overall_scores) / len(overall_scores), 2) if overall_scores else 0.0,
        "avg_completeness_score": round(sum(completeness_scores) / len(completeness_scores), 2) if completeness_scores else 0.0,
        "avg_organization_score": round(sum(organization_scores) / len(organization_scores), 2) if organization_scores else 0.0,
        "avg_sensical_score": round(sum(sensical_scores) / len(sensical_scores), 2) if sensical_scores else 0.0,
        "total_runtime_sec": round(time.time() - started_at, 2),
    }
    output_payload = {
        "generated_utc": datetime.utcnow().isoformat() + "Z",
        "pdf_dir": str(pdf_dir),
        "api_base": args.api_base,
        "summary": summary,
        "documents": docs,
    }

    json_path, md_path = write_outputs(Path(args.out_dir).expanduser().resolve(), output_payload)
    print(f"Audit complete: {summary}")
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")


if __name__ == "__main__":
    main()
