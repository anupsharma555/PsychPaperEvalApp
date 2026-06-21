#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from html import escape
import importlib.util
import json
import os
from pathlib import Path
import random
import signal
import sqlite3
import subprocess
import sys
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request
import uuid


ROOT = Path(__file__).resolve().parents[4]
MANIFEST = ROOT / "benchmarks" / "multi_paper_benchmark.json"
STANDARD = ROOT / "benchmarks" / "app_evaluation_standard.json"
GOLD_STANDARDS = ROOT / "benchmarks" / "gold_standards"
DEFAULT_HISTORY_DB = ROOT / "test" / "active_benchmark" / "benchmark_history.sqlite"
MAX_STATIC_MEDIA_BYTES = 12 * 1024 * 1024
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


@dataclass
class ActiveCaseState:
    case: dict[str, Any]
    record: dict[str, Any]
    case_dir: Path
    submitted_at: float
    running_deadline: float | None = None


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Failed to read {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"{path} did not contain a JSON object.")
    return payload


def _benchmark_definition_fingerprint() -> dict[str, Any]:
    files = [STANDARD, MANIFEST, *sorted(GOLD_STANDARDS.glob("*.json"))]
    file_records: list[dict[str, str]] = []
    aggregate = hashlib.sha256()
    for path in files:
        rel_path = str(path.relative_to(ROOT))
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        aggregate.update(rel_path.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
        file_records.append({"path": rel_path, "sha256": digest})
    return {
        "algorithm": "sha256",
        "digest": aggregate.hexdigest(),
        "file_count": len(file_records),
        "files": file_records,
    }


def _compact_benchmark_definition(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {
            "available": False,
            "algorithm": "",
            "digest": "",
            "file_count": None,
        }
    digest = str(value.get("digest") or "")
    return {
        "available": bool(digest),
        "algorithm": str(value.get("algorithm") or ""),
        "digest": digest,
        "file_count": _int_or_none(value.get("file_count")),
    }


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_manifest() -> int:
    manifest = _load_json(MANIFEST)
    _load_json(STANDARD)
    cases = manifest.get("cases", [])
    if not isinstance(cases, list):
        raise SystemExit("Manifest field `cases` must be a list.")

    missing_files: list[str] = []
    reference_scored = 0
    diagnostic = 0
    gold_standards = 0
    tiers: dict[str, int] = {}
    for case in cases:
        if not isinstance(case, dict):
            continue
        case_id = str(case.get("id") or "unknown")
        pdf = ROOT / str(case.get("pdf") or "")
        if not pdf.exists():
            missing_files.append(f"{case_id}: missing PDF {pdf}")
        reference_md = str(case.get("reference_md") or "").strip()
        if case.get("scoring") == "reference_comparison":
            reference_scored += 1
            if not reference_md or not (ROOT / reference_md).exists():
                missing_files.append(f"{case_id}: missing reference {reference_md}")
        elif case.get("scoring") == "diagnostic_coverage":
            diagnostic += 1
        gold_standard = str(case.get("gold_standard") or "").strip()
        if gold_standard:
            gold_standards += 1
            if not (ROOT / gold_standard).exists():
                missing_files.append(f"{case_id}: missing gold standard {gold_standard}")
        else:
            missing_files.append(f"{case_id}: missing gold_standard manifest field")
        for tier in case.get("tiers", []):
            tiers[str(tier)] = tiers.get(str(tier), 0) + 1

    print(f"manifest: {MANIFEST}")
    print(f"cases: {len(cases)}")
    print(f"reference_scored: {reference_scored}")
    print(f"diagnostic_coverage: {diagnostic}")
    print(f"gold_standards: {gold_standards}")
    print("tiers:")
    for tier, count in sorted(tiers.items()):
        print(f"- {tier}: {count}")

    if missing_files:
        print("\nMissing files:")
        for item in missing_files:
            print(f"- {item}")
        return 1
    print("\nManifest file references are present.")
    return 0


def _run_checks() -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "backend")
    env["PYTHONPYCACHEPREFIX"] = "/tmp/papereval_benchmark_pycache"
    commands = [
        [sys.executable, "-m", "json.tool", str(STANDARD)],
        [sys.executable, "-m", "json.tool", str(MANIFEST)],
        [sys.executable, "scripts/validate_gold_standards.py", *[str(path) for path in sorted(GOLD_STANDARDS.glob("*.json"))]],
        [
            str(ROOT / ".venv" / "bin" / "python"),
            "-m",
            "py_compile",
            ".codex/skills/papereval-benchmark/scripts/papereval_benchmark_qa.py",
            "scripts/run_multi_paper_benchmark.py",
            "scripts/validate_gold_standards.py",
            "scripts/compare_pdf_against_reference.py",
            "scripts/compare_evidence_to_gold.py",
            "backend/tests/test_evidence_gold_compatibility.py",
            "backend/tests/test_multi_paper_benchmark.py",
            "backend/tests/test_papereval_benchmark_helper.py",
        ],
        [
            str(ROOT / ".venv" / "bin" / "python"),
            "-m",
            "pytest",
            "backend/tests/test_evidence_gold_compatibility.py",
            "backend/tests/test_multi_paper_benchmark.py",
            "backend/tests/test_papereval_benchmark_helper.py",
        ],
    ]
    status = 0
    for cmd in commands:
        print(f"\n$ {' '.join(cmd)}")
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.DEVNULL if "json.tool" in cmd else None,
        )
        status = max(status, int(proc.returncode))
    return status


def _selected_cases(
    manifest: dict[str, Any],
    *,
    mode: str,
    tier: str,
    case_ids: list[str],
    include_unscored: bool,
    random_seed: int | None,
) -> list[dict[str, Any]]:
    cases = [case for case in manifest.get("cases", []) if isinstance(case, dict)]
    requested = set(case_ids)
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
        filtered = selected
    else:
        filtered = [case for case in selected if str(case.get("scoring")) == "reference_comparison"]
    if mode == "single":
        if not filtered:
            raise SystemExit("No active benchmark cases selected for single-paper mode.")
        rng = random.Random(random_seed)
        return [rng.choice(filtered)]
    return filtered


def _normalize_api_base(api_base: str, port: int) -> str:
    value = (api_base or f"http://127.0.0.1:{port}/api").rstrip("/")
    if not value.endswith("/api"):
        value = value + "/api"
    return value


def _web_url_for_api_base(api_base: str) -> str:
    root = api_base.rstrip("/")
    if root.endswith("/api"):
        root = root[: -len("/api")]
    return f"{root}/web/"


def _request_json(
    api_base: str,
    path: str,
    *,
    method: str = "GET",
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 15.0,
) -> dict[str, Any]:
    url = f"{api_base.rstrip('/')}/{path.lstrip('/')}"
    req = urllib.request.Request(url=url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed HTTP {exc.code}: {detail[:800]}") from exc
    payload = json.loads(raw or "{}")
    if not isinstance(payload, dict):
        raise RuntimeError(f"{method} {url} did not return a JSON object")
    return payload


def _url_ready(url: str, *, timeout: float = 5.0) -> bool:
    try:
        req = urllib.request.Request(url=url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return 200 <= int(getattr(response, "status", 0) or 0) < 500
    except Exception:
        return False


def _wait_api_ready(api_base: str, *, timeout_seconds: float) -> dict[str, Any]:
    deadline = time.time() + timeout_seconds
    last_error = ""
    while time.time() < deadline:
        try:
            payload = _request_json(api_base, "status", timeout=2.0)
            if payload.get("backend_ready") is True:
                return payload
        except Exception as exc:
            last_error = str(exc)
        time.sleep(0.7)
    raise RuntimeError(f"API did not become ready at {api_base}: {last_error}")


def _runtime_readiness_failures(api_status: dict[str, Any], *, allow_runtime_not_ready: bool) -> list[str]:
    if allow_runtime_not_ready:
        return []
    failures: list[str] = []
    processing = api_status.get("processing") if isinstance(api_status.get("processing"), dict) else {}
    if processing and processing.get("running") is not True:
        failures.append("backend job runner is not running")
    if processing and processing.get("paused") is True:
        failures.append("backend job runner is paused")
    if processing and processing.get("executor_enabled") is False:
        failures.append("backend job runner executor is disabled")
    if processing:
        try:
            capacity = int(processing.get("worker_capacity") or 0)
        except Exception:
            capacity = 0
        if capacity < 1:
            failures.append("backend job runner has no worker capacity")
        last_error = str(processing.get("last_error") or api_status.get("last_error") or "").strip()
        if last_error:
            failures.append(f"backend job runner reports error: {last_error}")
    grobid = api_status.get("grobid") if isinstance(api_status.get("grobid"), dict) else {}
    if grobid and grobid.get("ready") is not True:
        detail = grobid.get("error") or grobid.get("status_code") or "not ready"
        failures.append(f"GROBID is not ready ({detail})")
    if api_status.get("provider_ready") is False:
        provider = str(api_status.get("provider") or "unknown")
        failures.append(f"LLM provider is not ready ({provider})")
    return failures


def _surface_preflight_failures(api_base: str, api_status: dict[str, Any], args: argparse.Namespace) -> list[str]:
    if getattr(args, "allow_surface_not_ready", False):
        return []
    failures: list[str] = []
    if args.surface == "desktop":
        if args.start_app:
            failures.append("--surface desktop requires an existing/launched desktop app, not --start-app api-only mode")
        return failures
    if args.surface == "web":
        url = _web_url_for_api_base(api_base)
        if not _url_ready(url, timeout=5.0):
            failures.append(f"web frontend is not ready at {url}")
    return failures


def _run_preflight(api_base: str, api_status: dict[str, Any], args: argparse.Namespace) -> None:
    failures = [
        *_runtime_readiness_failures(api_status, allow_runtime_not_ready=bool(args.allow_runtime_not_ready)),
        *_surface_preflight_failures(api_base, api_status, args),
    ]
    if args.surface == "desktop" and not bool(args.allow_surface_not_ready):
        try:
            _request_json(api_base, "desktop/bootstrap", timeout=10.0)
        except Exception as exc:
            failures.append(f"desktop bootstrap failed: {exc}")
    if failures:
        joined = "; ".join(failures)
        raise RuntimeError(f"Active benchmark preflight failed: {joined}")


def _python() -> Path:
    venv_python = ROOT / ".venv" / "bin" / "python"
    if venv_python.exists():
        return venv_python
    return Path(sys.executable)


def _terminate_process_tree(proc: subprocess.Popen[Any]) -> None:
    try:
        pgid = os.getpgid(proc.pid)
    except Exception:
        pgid = None
    if isinstance(pgid, int) and pgid > 0:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except Exception:
            pass
        time.sleep(0.6)
        try:
            os.killpg(pgid, signal.SIGKILL)
        except Exception:
            pass
        return
    try:
        proc.terminate()
    except Exception:
        return
    try:
        proc.wait(timeout=3)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _start_app(args: argparse.Namespace, api_base: str, run_dir: Path) -> subprocess.Popen[Any] | None:
    if not args.start_app:
        if args.launch_desktop:
            app_bundle = ROOT / "PaperEval.app"
            if not app_bundle.exists():
                raise RuntimeError(f"PaperEval.app not found at {app_bundle}")
            subprocess.Popen(["open", str(app_bundle)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return None
    cmd = [
        str(_python()),
        "scripts/run_app.py",
        "--backend-port",
        str(args.backend_port),
        "--llm-provider",
        args.llm_provider,
        "--log-file",
        str(run_dir / "active_app.log"),
    ]
    cmd.append("--api-only")
    if args.force_start:
        cmd.append("--force")
    env = os.environ.copy()
    if _effective_disable_local_text_cache(args):
        env["ANALYSIS_LOCAL_TEXT_CACHE_ENABLED"] = "false"
        env["ANALYSIS_LOCAL_TEXT_GLOBAL_CACHE_ENABLED"] = "false"
    print(f"starting app surface={args.surface} api_base={api_base}")
    return subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _effective_disable_local_text_cache(args: argparse.Namespace) -> bool:
    if bool(getattr(args, "disable_local_text_cache", False)):
        return True
    if bool(getattr(args, "allow_local_text_cache", False)):
        return False
    return bool(getattr(args, "start_app", False)) and str(getattr(args, "llm_provider", "") or "") == "local"


def _multipart_upload_body(pdf_path: Path) -> tuple[bytes, str]:
    boundary = f"----PaperEvalBenchmark{uuid.uuid4().hex}"
    content = pdf_path.read_bytes()
    filename = pdf_path.name
    parts = [
        f"--{boundary}\r\n".encode("utf-8"),
        (
            'Content-Disposition: form-data; name="main_file"; '
            f'filename="{filename}"\r\n'
            "Content-Type: application/pdf\r\n\r\n"
        ).encode("utf-8"),
        content,
        b"\r\n",
        f"--{boundary}--\r\n".encode("utf-8"),
    ]
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _upload_case(api_base: str, pdf_path: Path) -> dict[str, Any]:
    body, content_type = _multipart_upload_body(pdf_path)
    return _request_json(
        api_base,
        "documents/upload",
        method="POST",
        data=body,
        headers={"Content-Type": content_type, "Content-Length": str(len(body))},
        timeout=120.0,
    )


def _compare_report_to_gold(case: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    gold_path = ROOT / str(case.get("gold_standard") or "")
    comparator = _load_module(SCRIPTS_DIR / "compare_evidence_to_gold.py", "compare_evidence_to_gold_active")
    validator = _load_module(SCRIPTS_DIR / "validate_gold_standards.py", "validate_gold_standards_active")
    gold = validator.load_gold_standard(gold_path)
    summary_json = report.get("summary_json") if isinstance(report.get("summary_json"), dict) else {}
    packets = comparator.evidence_packets_from_payload(summary_json)
    comparison = comparator.compare_evidence_to_gold(packets, gold)
    return {
        "gold_standard": str(gold_path),
        "comparison": comparison,
        "evidence_packet_count": len(packets),
    }


def _summary_json_from_report(report: dict[str, Any]) -> dict[str, Any]:
    summary_json = report.get("summary_json")
    if isinstance(summary_json, dict):
        return summary_json
    summary = report.get("summary")
    if isinstance(summary, str):
        try:
            parsed = json.loads(summary)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _render_report_scalar(value: Any) -> str:
    if value is None:
        return '<span class="muted">null</span>'
    if isinstance(value, bool):
        return f'<span class="pill">{str(value).lower()}</span>'
    if isinstance(value, (int, float)):
        return f"<code>{value}</code>"
    text = str(value)
    if "\n" in text or len(text) > 220:
        return "<pre>" + escape(text) + "</pre>"
    return escape(text)


def _render_report_value(value: Any) -> str:
    if isinstance(value, dict):
        if not value:
            return '<span class="muted">empty</span>'
        rows = []
        for key, item in value.items():
            rows.append(
                "<tr><th>"
                + escape(str(key))
                + "</th><td>"
                + _render_report_value(item)
                + "</td></tr>"
            )
        return '<table class="kv"><tbody>' + "".join(rows) + "</tbody></table>"
    if isinstance(value, list):
        if not value:
            return '<span class="muted">empty</span>'
        if all(not isinstance(item, (dict, list)) for item in value):
            return "<ul>" + "".join("<li>" + _render_report_scalar(item) + "</li>" for item in value) + "</ul>"
        rendered = []
        for index, item in enumerate(value, start=1):
            label = f"Item {index}"
            if isinstance(item, dict):
                label = str(item.get("title") or item.get("finding_id") or item.get("result") or item.get("section") or label)
                if len(label) > 120:
                    label = label[:117] + "..."
            rendered.append(
                "<details><summary>"
                + escape(label)
                + "</summary>"
                + _render_report_value(item)
                + "</details>"
            )
        return "".join(rendered)
    return _render_report_scalar(value)


def _report_section(title: str, value: Any, *, collapsible: bool = False, open_section: bool = False) -> str:
    if collapsible:
        return (
            '<details class="section"'
            + (" open" if open_section else "")
            + "><summary><h2>"
            + escape(title)
            + "</h2></summary>"
            + _render_report_value(value)
            + "</details>"
        )
    return '<section class="section"><h2>' + escape(title) + "</h2>" + _render_report_value(value) + "</section>"


def _clean_report_text(value: Any) -> str:
    text = str(value or "").replace("\r", " ").strip()
    while "  " in text:
        text = text.replace("  ", " ")
    return text


def _statement_from_item(item: Any) -> str:
    if isinstance(item, dict):
        return _clean_report_text(item.get("statement") or item.get("result") or item.get("summary") or item.get("text"))
    return _clean_report_text(item)


def _section_lines(summary_json: dict[str, Any], section_name: str, *, limit: int = 14) -> list[str]:
    sections = summary_json.get("sections") if isinstance(summary_json.get("sections"), dict) else {}
    section = sections.get(section_name) if isinstance(sections.get(section_name), dict) else {}
    candidates = section.get("narrative_items") or section.get("items") or []
    if not isinstance(candidates, list):
        candidates = []
    lines = [_statement_from_item(item) for item in candidates]
    if not lines and section.get("summary"):
        lines = [_clean_report_text(section.get("summary"))]
    return _unique([line for line in lines if line])[:limit]


def _section_evidence_refs(summary_json: dict[str, Any], section_name: str) -> list[str]:
    sections = summary_json.get("sections") if isinstance(summary_json.get("sections"), dict) else {}
    section = sections.get(section_name) if isinstance(sections.get(section_name), dict) else {}
    refs = section.get("evidence_refs") if isinstance(section.get("evidence_refs"), list) else []
    return _unique([str(ref) for ref in refs if str(ref or "").strip()])


def _render_reader_list(lines: list[str], *, empty: str = "No extracted content available.") -> str:
    if not lines:
        return f'<p class="empty">{escape(empty)}</p>'
    return '<ol class="reader-list">' + "".join("<li>" + escape(line) + "</li>" for line in lines) + "</ol>"


def _render_reader_section(title: str, lines: list[str], refs: list[str] | None = None) -> str:
    refs = refs or []
    refs_html = ""
    if refs:
        refs_html = (
            '<details class="evidence-note"><summary>Evidence refs ('
            + str(len(refs))
            + ")</summary><div>"
            + escape(", ".join(refs))
            + "</div></details>"
        )
    return '<section class="section reader"><h2>' + escape(title) + "</h2>" + _render_reader_list(lines) + refs_html + "</section>"


def _render_executive_summary(summary_text: Any) -> str:
    text = _clean_report_text(summary_text)
    if not text:
        return _report_section("Executive Summary", "")
    rows: list[tuple[str, str]] = []
    for raw in str(summary_text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if ":" in line:
            label, value = line.split(":", 1)
            rows.append((label.strip(), value.strip()))
        else:
            rows.append(("Summary", line))
    if not rows:
        rows = [("Summary", text)]
    body = '<div class="exec-grid">' + "".join(
        '<article><h3>' + escape(label) + "</h3><p>" + escape(value) + "</p></article>"
        for label, value in rows
    ) + "</div>"
    return '<section class="section reader"><h2>Executive Summary</h2>' + body + "</section>"


def _render_metadata_grid(paper_meta: dict[str, Any]) -> str:
    if not paper_meta:
        return ""
    rows = []
    for key in ["title", "authors", "journal", "date", "metadata_source", "authors_extracted_count"]:
        if key not in paper_meta:
            continue
        value = paper_meta.get(key)
        if isinstance(value, list):
            value_text = "; ".join(str(item) for item in value)
        else:
            value_text = str(value or "")
        rows.append(
            '<div class="meta-row"><strong>'
            + escape(key.replace("_", " ").title())
            + "</strong><span>"
            + escape(value_text)
            + "</span></div>"
        )
    return '<section class="section reader"><h2>Paper Metadata</h2><div class="meta-grid">' + "".join(rows) + "</div></section>"


def _api_root_from_base(api_base: str) -> str:
    root = api_base.rstrip("/")
    if root.endswith("/api"):
        root = root[: -len("/api")]
    return root


def _api_base_from_webapp_url(webapp_url: str) -> str:
    if not webapp_url:
        return ""
    parsed = urllib.parse.urlparse(webapp_url)
    if not parsed.scheme or not parsed.netloc:
        return ""
    return urllib.parse.urlunparse((parsed.scheme, parsed.netloc, "/api", "", "", ""))


def _absolute_media_url(value: str, *, api_base: str) -> str:
    value = str(value or "").strip()
    if not value:
        return ""
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme and parsed.netloc:
        return value
    root = _api_root_from_base(api_base)
    if not root:
        return value
    return urllib.parse.urljoin(root.rstrip("/") + "/", value.lstrip("/"))


def _is_same_backend_url(value: str, *, api_base: str) -> bool:
    if not value or not api_base:
        return False
    parsed_value = urllib.parse.urlparse(_absolute_media_url(value, api_base=api_base))
    parsed_root = urllib.parse.urlparse(_api_root_from_base(api_base))
    return bool(parsed_value.scheme and parsed_value.netloc and parsed_value.netloc == parsed_root.netloc)


def _extension_from_response(url: str, content_type: str) -> str:
    path_suffix = Path(urllib.parse.urlparse(url).path).suffix.lower()
    if path_suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf"}:
        return path_suffix
    content_type = content_type.split(";", 1)[0].strip().lower()
    return {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/gif": ".gif",
        "image/webp": ".webp",
        "application/pdf": ".pdf",
    }.get(content_type, ".bin")


def _safe_media_filename(kind: str, index: int, item: dict[str, Any], extension: str) -> str:
    chunk_id = str(item.get("chunk_id") or index)
    safe_chunk = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in chunk_id).strip("-")
    safe_kind = "".join(char if char.isalnum() else "-" for char in kind.lower()).strip("-") or "media"
    return f"{safe_kind}-{index:02d}-{safe_chunk}{extension}"


def _fetch_document_media(api_base: str, document_id: int | str | None) -> dict[str, Any]:
    if not api_base or not document_id:
        return {}
    try:
        media = _request_json(api_base, f"documents/{int(document_id)}/media", timeout=20.0)
    except Exception:
        return {}
    return media if isinstance(media, dict) else {}


def _download_static_media_assets(api_base: str, case_dir: Path, media: dict[str, Any]) -> dict[str, Any]:
    if not api_base or not media:
        return media
    asset_dir = case_dir / "media"
    copied = json.loads(json.dumps(media))
    for kind in ("figures", "tables"):
        items = copied.get(kind) if isinstance(copied.get(kind), list) else []
        for index, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                continue
            image_url = str(item.get("image_url") or "").strip()
            if not image_url or not _is_same_backend_url(image_url, api_base=api_base):
                continue
            absolute_url = _absolute_media_url(image_url, api_base=api_base)
            try:
                req = urllib.request.Request(absolute_url, method="GET")
                with urllib.request.urlopen(req, timeout=20.0) as response:
                    content_type = str(response.headers.get("Content-Type") or "")
                    data = response.read(MAX_STATIC_MEDIA_BYTES + 1)
            except Exception as exc:
                item.setdefault("static_asset_errors", []).append(str(exc)[:240])
                continue
            if len(data) > MAX_STATIC_MEDIA_BYTES:
                item.setdefault("static_asset_errors", []).append("media asset exceeded static artifact size limit")
                continue
            asset_dir.mkdir(parents=True, exist_ok=True)
            filename = _safe_media_filename(kind[:-1], index, item, _extension_from_response(absolute_url, content_type))
            output_path = asset_dir / filename
            output_path.write_bytes(data)
            item["static_image_url"] = f"media/{filename}"
    return copied


def _media_item_label(kind: str, item: dict[str, Any], index: int) -> str:
    if kind == "tables":
        base = "Table"
    elif str(item.get("asset_kind") or "") == "supp":
        base = "Supplementary Figure"
    else:
        base = "Figure"
    anchor = str(item.get("anchor") or item.get("figure_id") or "").strip()
    if anchor:
        return anchor
    return f"{base} {index}"


def _render_table_preview(preview: Any, *, max_rows: int = 8, max_cols: int = 8) -> str:
    if not isinstance(preview, dict):
        return ""
    columns = preview.get("columns") if isinstance(preview.get("columns"), list) else []
    rows = preview.get("rows") if isinstance(preview.get("rows"), list) else []
    if not rows:
        return ""
    columns = [str(col) for col in columns[:max_cols]]
    body_rows = []
    for row in rows[:max_rows]:
        cells = row if isinstance(row, list) else []
        body_rows.append("<tr>" + "".join("<td>" + escape(str(cell)) + "</td>" for cell in cells[:max_cols]) + "</tr>")
    header = ""
    if columns:
        header = "<thead><tr>" + "".join("<th>" + escape(col) + "</th>" for col in columns) + "</tr></thead>"
    return '<div class="table-preview"><table>' + header + "<tbody>" + "".join(body_rows) + "</tbody></table></div>"


def _render_static_media_assets(media: dict[str, Any] | None, *, api_base: str = "") -> str:
    if not isinstance(media, dict):
        return ""
    sections: list[str] = []
    for kind, title in (("tables", "Embedded Tables"), ("figures", "Embedded Figures")):
        items = media.get(kind) if isinstance(media.get(kind), list) else []
        if not items:
            continue
        cards = []
        for index, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                continue
            label = _media_item_label(kind, item, index)
            legend = _clean_report_text(item.get("legend") or item.get("caption"))
            source = _absolute_media_url(str(item.get("source_proxy_url") or item.get("asset_url") or ""), api_base=api_base)
            static_image = str(item.get("static_image_url") or "").strip()
            live_image = _absolute_media_url(str(item.get("image_url") or ""), api_base=api_base)
            image_src = static_image or live_image
            meta = []
            if item.get("page"):
                meta.append(f"Page {item.get('page')}")
            if item.get("asset_kind"):
                meta.append(str(item.get("asset_kind")))
            card = (
                '<article class="media-card asset-card"><h3>'
                + escape(label)
                + "</h3>"
                + (f'<p>{escape(legend)}</p>' if legend else '<p class="empty">Legend unavailable from source extraction.</p>')
                + (f'<div class="asset-meta">{escape(" | ".join(meta))}</div>' if meta else "")
                + (f'<a class="asset-link" href="{escape(source, quote=True)}">Open source</a>' if source else "")
                + (f'<img class="asset-image" src="{escape(image_src, quote=True)}" alt="{escape(label, quote=True)}">' if image_src else "")
                + _render_table_preview(item.get("table_preview"))
                + "</article>"
            )
            cards.append(card)
        if cards:
            sections.append('<section class="section reader"><h2>' + escape(title) + '</h2><div class="media-grid">' + "".join(cards) + "</div></section>")
    return "".join(sections)


def _render_result_cards(title: str, rows: Any, *, empty: str) -> str:
    items = rows if isinstance(rows, list) else []
    if not items:
        return '<section class="section reader"><h2>' + escape(title) + f'</h2><p class="empty">{escape(empty)}</p></section>'
    cards = []
    for index, item in enumerate(items, start=1):
        if isinstance(item, dict):
            statement = _clean_report_text(item.get("result") or item.get("statement") or item.get("summary"))
            evidence = item.get("evidence") if isinstance(item.get("evidence"), list) else item.get("evidence_refs")
            evidence_list = [str(ref) for ref in evidence] if isinstance(evidence, list) else []
            confidence = item.get("confidence")
        else:
            statement = _clean_report_text(item)
            evidence_list = []
            confidence = None
        cards.append(
            '<article class="media-card"><h3>'
            + escape(f"{title[:-1] if title.endswith('s') else title} {index}")
            + "</h3><p>"
            + escape(statement or "No extracted statement available.")
            + "</p>"
            + ('<div class="asset-meta">Evidence: ' + escape(", ".join(evidence_list)) + "</div>" if evidence_list else "")
            + (f'<div class="asset-meta">Confidence: {escape(str(confidence))}</div>' if confidence is not None else "")
            + "</article>"
        )
    return '<section class="section reader"><h2>' + escape(title) + '</h2><div class="media-grid">' + "".join(cards) + "</div></section>"


def _render_supplement_note(summary_json: dict[str, Any]) -> str:
    note = _clean_report_text(summary_json.get("supplement_availability_note"))
    coverage = summary_json.get("coverage") if isinstance(summary_json.get("coverage"), dict) else {}
    parts = [note] if note else []
    for key, label in [("supp_figures", "supplementary figures"), ("supp_tables", "supplementary tables")]:
        block = coverage.get(key) if isinstance(coverage.get(key), dict) else {}
        missing = block.get("missing_refs") if isinstance(block.get("missing_refs"), list) else []
        expected = block.get("expected")
        extracted = block.get("extracted")
        if missing:
            parts.append(
                f"Missing {len(missing)} {label}: {', '.join(str(item) for item in missing)}."
            )
        elif expected is not None or extracted is not None:
            parts.append(f"{label.title()}: expected {expected or 0}, extracted {extracted or 0}.")
    if not parts:
        return ""
    return '<section class="section reader"><h2>Supplement Availability</h2>' + _render_reader_list(parts) + "</section>"


def _comparison_metrics(comparison: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(comparison, dict):
        return {}
    metrics = comparison.get("comparison") if isinstance(comparison.get("comparison"), dict) else comparison
    return metrics if isinstance(metrics, dict) else {}


def _format_benchmark_score(value: Any) -> str:
    numeric = _float_or_none(value)
    if numeric is None:
        return ""
    return f"{numeric:.3f}".rstrip("0").rstrip(".")


def _benchmark_score_value(metrics: dict[str, Any]) -> Any:
    score = metrics.get("overall_benchmark_score")
    if score is None:
        score = metrics.get("benchmark_content_score")
    return score


def _score_component_rows(basis: dict[str, Any]) -> list[tuple[str, Any, Any]]:
    components = basis.get("components") if isinstance(basis.get("components"), dict) else {}
    labels = [
        ("critical_claim_candidates", "Claim candidates"),
        ("expected_entities", "Expected entities"),
        ("expected_numbers", "Expected numbers"),
        ("expected_detail_types", "Expected detail types"),
        ("required_sections", "Required sections"),
    ]
    rows: list[tuple[str, Any, Any]] = []
    for key, label in labels:
        item = components.get(key) if isinstance(components.get(key), dict) else {}
        matched = item.get("matched")
        expected = item.get("expected")
        if matched is None or expected is None:
            continue
        rows.append((label, matched, expected))
    return rows


def _score_component_text(basis: dict[str, Any]) -> str:
    rows = _score_component_rows(basis)
    return "; ".join(f"{label} {matched}/{expected}" for label, matched, expected in rows)


def _benchmark_gap_rows(metrics: dict[str, Any], *, max_rows: int = 4) -> list[str]:
    raw_gaps = metrics.get("claim_requirement_gaps")
    if not isinstance(raw_gaps, list):
        return []
    rows: list[str] = []
    for gap in raw_gaps:
        if not isinstance(gap, dict):
            continue
        parts: list[str] = []
        missing_entities = [str(item) for item in gap.get("missing_entities", []) if str(item).strip()] if isinstance(gap.get("missing_entities"), list) else []
        missing_numbers = [str(item) for item in gap.get("missing_numbers", []) if str(item).strip()] if isinstance(gap.get("missing_numbers"), list) else []
        missing_detail_types = [str(item) for item in gap.get("missing_detail_types", []) if str(item).strip()] if isinstance(gap.get("missing_detail_types"), list) else []
        if gap.get("candidate_missing"):
            parts.append("candidate missing")
        if missing_entities:
            parts.append("missing entities: " + ", ".join(missing_entities[:4]))
        if missing_numbers:
            parts.append("missing numbers: " + ", ".join(missing_numbers[:4]))
        if missing_detail_types:
            parts.append("missing detail types: " + ", ".join(missing_detail_types[:4]))
        if not parts:
            continue
        claim = str(gap.get("claim_id") or "claim").strip()
        section = str(gap.get("section") or "").strip()
        prefix = claim + (f" ({section})" if section else "")
        rows.append(prefix + ": " + "; ".join(parts))
        if len(rows) >= max_rows:
            break
    if len(raw_gaps) > len(rows) and rows:
        rows.append(f"{len(raw_gaps) - len(rows)} additional benchmark gap(s) not shown")
    return rows


def _benchmark_gap_text(metrics: dict[str, Any], *, max_rows: int = 4) -> str:
    return " | ".join(_benchmark_gap_rows(metrics, max_rows=max_rows))


def _render_benchmark_score_html(comparison: dict[str, Any] | None) -> str:
    metrics = _comparison_metrics(comparison)
    score = _format_benchmark_score(_benchmark_score_value(metrics))
    if not score:
        return ""
    basis = metrics.get("benchmark_content_score_basis") if isinstance(metrics.get("benchmark_content_score_basis"), dict) else {}
    matched = basis.get("matched_slots")
    expected = basis.get("expected_slots")
    detail = ""
    if matched is not None and expected is not None:
        detail = f"Matched {matched} of {expected} expected benchmark content slots. Extra report content is not penalized."
    compatible = metrics.get("compatible")
    compatible_text = ""
    if compatible is not None:
        compatible_text = "Compatible with gold thresholds." if compatible else "Below one or more gold thresholds."
    parts = [part for part in (detail, compatible_text) if part]
    body = '<div class="score-card"><div class="score-value">' + escape(score) + "</div>"
    body += '<div><h2>Benchmark Score</h2>'
    if parts:
        body += "<p>" + escape(" ".join(parts)) + "</p>"
    component_text = _score_component_text(basis)
    if component_text:
        body += '<p class="score-components">' + escape(component_text) + "</p>"
    gap_rows = _benchmark_gap_rows(metrics)
    if gap_rows:
        body += '<ul class="score-gaps">' + "".join("<li>" + escape(row) + "</li>" for row in gap_rows) + "</ul>"
    body += "</div></div>"
    return '<section class="section reader score-section">' + body + "</section>"


def _write_detailed_analysis_html(
    report: dict[str, Any],
    output_path: Path,
    *,
    webapp_url: str = "",
    media: dict[str, Any] | None = None,
    api_base: str = "",
    comparison: dict[str, Any] | None = None,
) -> Path:
    summary_json = _summary_json_from_report(report)
    paper_meta = summary_json.get("paper_meta") if isinstance(summary_json.get("paper_meta"), dict) else {}
    document = report.get("document") if isinstance(report.get("document"), dict) else {}
    title = str(paper_meta.get("title") or document.get("title") or "PaperEval Detailed Analysis")
    authors = paper_meta.get("authors") if isinstance(paper_meta.get("authors"), list) else []
    author_line = "; ".join(str(author) for author in authors[:8])
    if len(authors) > 8:
        author_line += " et al."

    shown_fields = {
        "paper_meta",
        "executive_summary",
        "key_findings",
        "secondary_findings",
        "sensitivity_analysis",
        "statistical_tests_used",
        "uniqueness",
        "sections",
        "table_results",
        "figure_results",
        "supplement_availability_note",
        "methodology_details",
        "methods_compact",
        "scientific_details",
        "cross_modal_claims",
        "discrepancies",
        "strengths",
        "weaknesses",
        "overall_confidence",
        "scores",
        "coverage",
        "evidence_packet_coverage",
        "presentation_evidence",
        "evidence_packets",
    }
    metadata = {
        "paper": paper_meta,
        "document": document,
        "report_invalid": bool(report.get("report_invalid")),
        "report_invalid_reason": str(report.get("report_invalid_reason") or ""),
        "summary_schema_version": report.get("summary_schema_version"),
        "sectioned_report_version": report.get("sectioned_report_version"),
    }
    sections: list[str] = []
    benchmark_score_html = _render_benchmark_score_html(comparison)
    if benchmark_score_html:
        sections.append(benchmark_score_html)
    sections.append(_render_executive_summary(summary_json.get("executive_summary") or report.get("summary")))
    if summary_json.get("scientific_details"):
        sections.append(_report_section("Scientific Details", summary_json.get("scientific_details")))
    sections.append(_render_metadata_grid(paper_meta))
    for section_name, label in [
        ("introduction", "Introduction"),
        ("methods", "Methods"),
        ("results", "Results"),
    ]:
        sections.append(_render_reader_section(label, _section_lines(summary_json, section_name), _section_evidence_refs(summary_json, section_name)))
    sections.append(_render_result_cards("Tables", summary_json.get("table_results"), empty="No extracted table summaries available."))
    sections.append(_render_result_cards("Figures", summary_json.get("figure_results"), empty="No extracted figure summaries available."))
    media_assets_html = _render_static_media_assets(media, api_base=api_base)
    if media_assets_html:
        sections.append(media_assets_html)
    supplement_note = _render_supplement_note(summary_json)
    if supplement_note:
        sections.append(supplement_note)
    for section_name, label in [
        ("conclusion", "Conclusion"),
        ("discussion", "Discussion"),
    ]:
        sections.append(_render_reader_section(label, _section_lines(summary_json, section_name), _section_evidence_refs(summary_json, section_name)))
    for key, label in [
        ("secondary_findings", "Secondary Findings"),
        ("sensitivity_analysis", "Sensitivity Analysis"),
        ("statistical_tests_used", "Statistical Tests Used"),
        ("uniqueness", "Uniqueness"),
        ("discrepancies", "Discrepancies"),
        ("weaknesses", "Weaknesses"),
        ("coverage", "Coverage"),
        ("evidence_packet_coverage", "Evidence Packet Coverage"),
    ]:
        if key in summary_json:
            sections.append(_report_section(label, summary_json.get(key), collapsible=key in {"coverage", "evidence_packet_coverage"}))
    sections.append(_report_section("Report Metadata", metadata, collapsible=True))
    if "presentation_evidence" in summary_json:
        sections.append(_report_section("Presentation Evidence", summary_json.get("presentation_evidence"), collapsible=True))
    if "evidence_packets" in summary_json:
        sections.append(_report_section("Evidence Packets", summary_json.get("evidence_packets"), collapsible=True))
    sections.append(_report_section("Analysis Diagnostics", report.get("analysis_diagnostics"), collapsible=True))
    sections.append(_report_section("Latency Profile", report.get("latency_profile"), collapsible=True))
    remaining = {key: value for key, value in summary_json.items() if key not in shown_fields}
    sections.append(_report_section("Other Structured Fields", remaining, collapsible=True))

    invalid_notice = ""
    if report.get("report_invalid"):
        invalid_notice = (
            '<div class="notice">Report marked invalid: '
            + escape(str(report.get("report_invalid_reason") or ""))
            + "</div>"
        )

    webapp_link = (
        '<a class="button" href="' + escape(webapp_url, quote=True) + '">Open in PaperEval Webapp</a>'
        if webapp_url
        else ""
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)}</title>
<style>
:root {{ color-scheme: light; --ink:#18212f; --muted:#667085; --line:#d9e0e8; --soft:#f6f8fb; --accent:#0f766e; }}
* {{ box-sizing: border-box; }}
body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:var(--ink); background:#fff; line-height:1.5; }}
header {{ padding:32px clamp(18px,4vw,56px) 24px; border-bottom:1px solid var(--line); background:linear-gradient(180deg,#f8fbff,#ffffff); }}
h1 {{ margin:0 0 10px; font-size:clamp(26px,4vw,42px); line-height:1.1; letter-spacing:0; max-width:1100px; }}
.meta {{ color:var(--muted); max-width:1100px; }}
.actions {{ margin-top:18px; display:flex; gap:10px; flex-wrap:wrap; }}
a.button {{ color:#fff; background:var(--accent); padding:8px 12px; border-radius:6px; text-decoration:none; font-weight:600; }}
a.button.secondary {{ background:#344054; }}
main {{ max-width:1180px; margin:0 auto; padding:24px clamp(14px,3vw,36px) 56px; }}
.section {{ border-top:1px solid var(--line); padding:22px 0; }}
.section:first-child {{ border-top:0; }}
.reader h2 {{ font-size:21px; }}
.score-section {{ padding-top:0; }}
.score-card {{ display:flex; align-items:center; gap:18px; border:1px solid var(--line); border-radius:6px; padding:14px; background:#f8fbff; }}
.score-card h2 {{ margin:0 0 4px; }}
.score-card p {{ margin:0; color:var(--muted); }}
.score-components {{ margin-top:6px !important; font-size:13px; }}
.score-gaps {{ margin:8px 0 0 18px; color:var(--muted); font-size:13px; }}
.score-value {{ font-size:38px; line-height:1; font-weight:800; color:var(--accent); min-width:96px; }}
.exec-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:10px; }}
.exec-grid article,.media-card {{ border:1px solid var(--line); border-radius:6px; padding:12px; background:#fff; }}
.exec-grid h3,.media-card h3 {{ font-size:13px; text-transform:uppercase; margin:0 0 6px; color:#0f766e; letter-spacing:.04em; }}
.exec-grid p,.media-card p {{ margin:0; }}
.reader-list {{ margin:8px 0 8px 24px; padding:0; }}
.reader-list li {{ margin:8px 0; }}
.meta-grid {{ display:grid; gap:8px; }}
.meta-row {{ display:grid; grid-template-columns:minmax(140px,220px) 1fr; gap:10px; border-bottom:1px solid var(--line); padding:8px 0; }}
.meta-row span {{ overflow-wrap:anywhere; }}
.media-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:10px; }}
.asset-card {{ display:flex; flex-direction:column; gap:8px; }}
.asset-link {{ color:var(--accent); font-weight:600; text-decoration:none; }}
.asset-image {{ width:100%; max-height:360px; object-fit:contain; border:1px solid var(--line); border-radius:6px; background:#fff; }}
.table-preview {{ overflow:auto; border:1px solid var(--line); border-radius:6px; max-height:340px; }}
.table-preview table {{ width:100%; border-collapse:collapse; font-size:13px; }}
.table-preview th,.table-preview td {{ border:1px solid var(--line); padding:5px 6px; vertical-align:top; overflow-wrap:anywhere; }}
.table-preview th {{ background:var(--soft); text-align:left; }}
.asset-meta,.evidence-note {{ margin-top:8px; color:var(--muted); font-size:13px; }}
.empty {{ color:var(--muted); }}
h2 {{ font-size:20px; margin:0 0 12px; letter-spacing:0; }}
details.section > summary {{ cursor:pointer; list-style:none; }}
details.section > summary::-webkit-details-marker {{ display:none; }}
details.section > summary h2::before {{ content:'+'; display:inline-block; width:22px; color:var(--accent); }}
details.section[open] > summary h2::before {{ content:'-'; }}
details:not(.section) {{ border:1px solid var(--line); border-radius:6px; padding:10px 12px; margin:8px 0; background:#fff; }}
details:not(.section) > summary {{ cursor:pointer; color:#344054; font-weight:600; }}
table.kv {{ width:100%; border-collapse:collapse; margin:8px 0; table-layout:fixed; }}
.kv th {{ width:230px; text-align:left; vertical-align:top; color:#344054; background:var(--soft); border:1px solid var(--line); padding:8px; overflow-wrap:anywhere; }}
.kv td {{ vertical-align:top; border:1px solid var(--line); padding:8px; overflow-wrap:anywhere; }}
ul {{ margin:8px 0 8px 22px; padding:0; }}
li {{ margin:6px 0; }}
pre {{ white-space:pre-wrap; overflow-wrap:anywhere; background:#101828; color:#eef2f6; padding:12px; border-radius:6px; overflow:auto; max-height:520px; }}
code {{ background:#eef2f6; padding:1px 4px; border-radius:4px; }}
.pill {{ display:inline-block; padding:2px 7px; border-radius:999px; background:#ecfdf3; color:#067647; font-size:12px; font-weight:700; }}
.muted {{ color:var(--muted); }}
.notice {{ background:#fffbeb; border:1px solid #fedf89; color:#93370d; padding:10px 12px; border-radius:6px; margin-top:14px; max-width:980px; }}
@media (max-width:700px) {{ .kv th,.kv td,.meta-row {{ display:block; width:100%; }} header {{ padding-top:22px; }} }}
</style>
</head>
<body>
<header>
<h1>{escape(title)}</h1>
<div class="meta">{escape(author_line)}<br>{escape(str(paper_meta.get("journal") or ""))} {escape(str(paper_meta.get("date") or ""))}</div>
<div class="actions">{webapp_link}<a class="button secondary" href="report.json">Raw JSON</a></div>
{invalid_notice}
</header>
<main>
<div id="tables"></div>
{''.join(sections)}
</main>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
    return output_path


def _write_slack_summary_markdown(
    report: dict[str, Any],
    output_path: Path,
    *,
    detailed_analysis_url: str = "",
    webapp_url: str = "",
    comparison: dict[str, Any] | None = None,
) -> Path:
    summary_json = _summary_json_from_report(report)
    paper_meta = summary_json.get("paper_meta") if isinstance(summary_json.get("paper_meta"), dict) else {}
    title = str(paper_meta.get("title") or "PaperEval report")
    executive_lines: list[str] = []
    for raw in str(summary_json.get("executive_summary") or report.get("summary") or "").splitlines():
        line = _clean_report_text(raw)
        if not line:
            continue
        if ":" in line:
            label, value = line.split(":", 1)
            executive_lines.extend([f"*{label.strip()}*", value.strip(), ""])
        else:
            executive_lines.extend(["*Summary*", line, ""])
    if not executive_lines:
        executive_lines = ["Detailed analysis is available.", ""]
    joined = "\n".join(executive_lines).strip()
    if len(joined) > 1400:
        joined = joined[:1397].rstrip() + "..."
    lines = [f"*{title}*", "", joined]
    metrics = _comparison_metrics(comparison)
    score = _format_benchmark_score(_benchmark_score_value(metrics))
    if score:
        basis = metrics.get("benchmark_content_score_basis") if isinstance(metrics.get("benchmark_content_score_basis"), dict) else {}
        matched = basis.get("matched_slots")
        expected = basis.get("expected_slots")
        if matched is not None and expected is not None:
            lines.extend(["", f"*Benchmark score*: {score} ({matched}/{expected} expected content slots)"])
        else:
            lines.extend(["", f"*Benchmark score*: {score}"])
        component_text = _score_component_text(basis)
        if component_text:
            lines.append(f"Score components: {component_text}")
        gap_text = _benchmark_gap_text(metrics, max_rows=3)
        if gap_text:
            lines.append(f"Benchmark gaps: {gap_text}")
    if detailed_analysis_url:
        lines.extend(["", f"Static detailed analysis: {detailed_analysis_url}"])
    if webapp_url:
        lines.extend(["", f"Open in PaperEval webapp: {webapp_url}"])
    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return output_path


def _read_optional_json_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _artifact_metadata(
    *,
    report_path: Path,
    html_path: Path,
    slack_summary_path: Path,
    media_json_path: Path | None = None,
    detailed_analysis_url: str = "",
    webapp_url: str = "",
) -> dict[str, str]:
    metadata = {
        "report_json": str(report_path),
        "detailed_analysis_html": str(html_path),
        "slack_summary_markdown": str(slack_summary_path),
    }
    if media_json_path is not None:
        metadata["media_json"] = str(media_json_path)
        media_dir = media_json_path.parent / "media"
        if media_dir.exists():
            metadata["static_media_dir"] = str(media_dir)
    if detailed_analysis_url:
        metadata["detailed_analysis_url"] = detailed_analysis_url
    if webapp_url:
        metadata["webapp_detailed_analysis_url"] = webapp_url
    return metadata


def _merge_artifact_metadata(record: dict[str, Any], metadata: dict[str, str]) -> None:
    artifacts = record.get("artifacts") if isinstance(record.get("artifacts"), dict) else {}
    artifacts.update(metadata)
    record["artifacts"] = artifacts


def _backfill_artifact_metadata(case_dir: Path, metadata: dict[str, str]) -> list[Path]:
    updated: list[Path] = []
    record_path = case_dir / "record.json"
    record = _read_optional_json_object(record_path)
    case_id = case_dir.name
    if record is not None:
        case_id = str(record.get("case_id") or case_id)
        _merge_artifact_metadata(record, metadata)
        record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        updated.append(record_path)

    summary_path = case_dir.parent / "active_benchmark_summary.json"
    summary = _read_optional_json_object(summary_path)
    records = summary.get("records") if isinstance(summary, dict) else None
    if isinstance(records, list):
        changed = False
        for item in records:
            if not isinstance(item, dict):
                continue
            if str(item.get("case_id") or "") != case_id:
                continue
            _merge_artifact_metadata(item, metadata)
            changed = True
        if changed:
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            updated.append(summary_path)
    return updated


def _manifest_case_by_id(case_id: str) -> dict[str, Any] | None:
    if not case_id:
        return None
    manifest = _load_json(MANIFEST)
    for case in manifest.get("cases", []):
        if isinstance(case, dict) and str(case.get("id") or "") == case_id:
            return case
    return None


def _refresh_case_comparison_metadata(case_dir: Path, report: dict[str, Any]) -> list[Path]:
    updated: list[Path] = []
    record_path = case_dir / "record.json"
    record = _read_optional_json_object(record_path)
    if record is None:
        return updated
    case_id = str(record.get("case_id") or case_dir.name)
    case = _manifest_case_by_id(case_id)
    if case is None:
        return updated
    comparison = _compare_report_to_gold(case, report)
    record["comparison"] = comparison
    record["iteration_diagnostics"] = _iteration_diagnostics(record)
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    updated.append(record_path)

    summary_path = case_dir.parent / "active_benchmark_summary.json"
    summary = _read_optional_json_object(summary_path)
    records = summary.get("records") if isinstance(summary, dict) else None
    if isinstance(records, list):
        changed = False
        for item in records:
            if not isinstance(item, dict):
                continue
            if str(item.get("case_id") or "") != case_id:
                continue
            item["comparison"] = comparison
            item["iteration_diagnostics"] = _iteration_diagnostics(item)
            changed = True
        if changed:
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            updated.append(summary_path)
    return updated


def _artifact_url(base_url: str, case_id: str, filename: str) -> str:
    if not base_url:
        return ""
    return f"{base_url.rstrip('/')}/{case_id}/{filename}"


def _webapp_detailed_analysis_url(api_base: str, record: dict[str, Any]) -> str:
    root = api_base.rstrip("/")
    if root.endswith("/api"):
        root = root[: -len("/api")]
    job_id = record.get("job_id")
    document_id = record.get("document_id")
    if not job_id and not document_id:
        return ""
    params = []
    if job_id:
        params.append(f"job_id={job_id}")
    if document_id:
        params.append(f"document_id={document_id}")
    params.append("view=detailed_analysis")
    return f"{root}/web/?{'&'.join(params)}"


def _diagnostic_extract(report: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    diagnostics = report.get("analysis_diagnostics") if isinstance(report.get("analysis_diagnostics"), dict) else {}
    summary_json = report.get("summary_json") if isinstance(report.get("summary_json"), dict) else {}
    nested_diagnostics = diagnostics.get("diagnostics") if isinstance(diagnostics.get("diagnostics"), dict) else {}
    run_validity = diagnostics.get("run_validity") if isinstance(diagnostics.get("run_validity"), dict) else {}
    if not run_validity and isinstance(nested_diagnostics.get("run_validity"), dict):
        run_validity = nested_diagnostics["run_validity"]
    fallback_audit = run_validity.get("fallback_audit") if isinstance(run_validity.get("fallback_audit"), dict) else {}
    model_usage = summary_json.get("model_usage") if isinstance(summary_json.get("model_usage"), dict) else {}
    latency_profile = _latency_profile_from_report(report, diagnostics)
    return {
        "job_status": str(job.get("status") or ""),
        "job_message": str(job.get("message") or ""),
        "report_invalid": bool(report.get("report_invalid")),
        "report_invalid_reason": str(report.get("report_invalid_reason") or ""),
        "run_validity": run_validity,
        "fallback_audit": fallback_audit,
        "model_usage": model_usage,
        "latency_profile": _latency_record_summary(latency_profile),
        "analysis_diagnostics_keys": sorted(str(key) for key in diagnostics),
    }


def _iteration_diagnostics(record: dict[str, Any]) -> dict[str, Any]:
    diagnostics = record.get("diagnostics") if isinstance(record.get("diagnostics"), dict) else {}
    latency = diagnostics.get("latency_profile") if isinstance(diagnostics.get("latency_profile"), dict) else {}
    latency_quality_flags = [
        str(flag)
        for flag in latency.get("quality_flags", [])
        if str(flag or "").strip()
    ][:12] if isinstance(latency.get("quality_flags"), list) else []
    cache_hit_stages = [
        str(stage)
        for stage in latency.get("cache_hit_stages", [])
        if str(stage or "").strip()
    ][:12] if isinstance(latency.get("cache_hit_stages"), list) else []
    comparison_wrapper = record.get("comparison") if isinstance(record.get("comparison"), dict) else {}
    comparison = comparison_wrapper.get("comparison") if isinstance(comparison_wrapper.get("comparison"), dict) else {}
    score_basis = (
        comparison.get("benchmark_content_score_basis")
        if isinstance(comparison.get("benchmark_content_score_basis"), dict)
        else {}
    )
    claim_gaps = comparison.get("claim_requirement_gaps") if isinstance(comparison.get("claim_requirement_gaps"), list) else []
    bottlenecks = latency.get("top_bottlenecks", []) if isinstance(latency.get("top_bottlenecks"), list) else []
    first_bottleneck = bottlenecks[0] if bottlenecks and isinstance(bottlenecks[0], dict) else {}
    failures = [str(item) for item in record.get("failures", []) if str(item).strip()] if isinstance(record.get("failures"), list) else []
    expected_slots = _float_or_none(score_basis.get("expected_slots"))
    matched_slots = _float_or_none(score_basis.get("matched_slots"))
    missing_slots = None
    if expected_slots is not None and matched_slots is not None:
        missing_slots = max(0.0, expected_slots - matched_slots)
    report_invalid_reason = str(diagnostics.get("report_invalid_reason") or "")
    run_validity = diagnostics.get("run_validity") if isinstance(diagnostics.get("run_validity"), dict) else {}
    validity_reasons = [
        str(reason)
        for reason in run_validity.get("reasons", [])
        if str(reason or "").strip()
    ] if isinstance(run_validity.get("reasons"), list) else []
    text_cache_hit = "text_cache_hit" in latency_quality_flags or "text" in cache_hit_stages
    text_cache_validity_conflict = (
        text_cache_hit
        and "local model text analysis did not run" in report_invalid_reason.lower()
    )
    narrative_synthesis_missing = (
        "narrative_synthesis_calls_zero" in validity_reasons
        or "narrative synthesis did not run" in report_invalid_reason.lower()
    )

    next_focus: list[str] = []
    if failures:
        next_focus.append("fix_current_failure")
    if text_cache_validity_conflict:
        next_focus.append("resolve_text_cache_validity_conflict")
    if narrative_synthesis_missing:
        next_focus.append("resolve_missing_narrative_synthesis")
    if first_bottleneck.get("stage"):
        next_focus.append(f"optimize_stage:{first_bottleneck['stage']}")
    if comparison and comparison.get("compatible") is False:
        next_focus.append("inspect_evidence_to_gold_gaps")
    for flag in latency.get("quality_flags", []) if isinstance(latency.get("quality_flags"), list) else []:
        flag_text = str(flag).strip()
        if flag_text:
            next_focus.append(f"diagnose_flag:{flag_text}")

    return {
        "case_id": str(record.get("case_id") or ""),
        "decision": str(record.get("decision") or ""),
        "release_gate": bool(record.get("release_gate")),
        "failures": failures[:5],
        "compatible": comparison.get("compatible") if comparison else None,
        "overall_benchmark_score": _float_or_none(comparison.get("overall_benchmark_score")) if comparison else None,
        "benchmark_content_score": _float_or_none(comparison.get("benchmark_content_score")) if comparison else None,
        "benchmark_matched_slots": matched_slots,
        "benchmark_expected_slots": expected_slots,
        "benchmark_missing_slots": missing_slots,
        "claim_requirement_gap_count": len(claim_gaps),
        "benchmark_gap_summary": _benchmark_gap_text(comparison, max_rows=3) if comparison else "",
        "usable_packet_rate": _float_or_none(comparison.get("usable_packet_rate")) if comparison else None,
        "critical_claim_candidate_rate": _float_or_none(comparison.get("critical_claim_candidate_rate")) if comparison else None,
        "slowest_stage": str(latency.get("slowest_stage") or ""),
        "slowest_stage_seconds": _float_or_none(first_bottleneck.get("duration_seconds")),
        "latency_quality_flags": latency_quality_flags,
        "cache_hit_stages": cache_hit_stages,
        "cache_hit_count": int(_float_or_none(latency.get("cache_hit_count")) or 0),
        "text_analysis_cache_validity_conflict": text_cache_validity_conflict,
        "narrative_synthesis_missing": narrative_synthesis_missing,
        "next_focus": _unique(next_focus),
    }


def _merged_iteration_diagnostics(record: dict[str, Any]) -> dict[str, Any]:
    derived = _iteration_diagnostics(record)
    existing = record.get("iteration_diagnostics")
    if not isinstance(existing, dict):
        return derived
    merged = dict(existing)
    refresh_keys = {
        "cache_hit_count",
        "cache_hit_stages",
        "latency_quality_flags",
        "narrative_synthesis_missing",
        "next_focus",
        "text_analysis_cache_validity_conflict",
    }
    for key, value in derived.items():
        if key in refresh_keys or key not in merged or merged.get(key) is None:
            merged[key] = value
    return merged


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _latency_profile_from_report(
    report: dict[str, Any],
    diagnostics_wrapper: dict[str, Any],
) -> dict[str, Any]:
    for candidate in (
        report.get("latency_profile"),
        diagnostics_wrapper.get("latency_profile"),
        (diagnostics_wrapper.get("diagnostics") or {}).get("latency_profile")
        if isinstance(diagnostics_wrapper.get("diagnostics"), dict)
        else None,
    ):
        if isinstance(candidate, dict):
            return candidate
    return {}


def _latency_record_summary(profile: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(profile, dict) or not profile:
        return {}
    cache_summary = profile.get("cache_summary") if isinstance(profile.get("cache_summary"), dict) else {}
    bottlenecks: list[dict[str, Any]] = []
    for raw in profile.get("top_bottlenecks", []) if isinstance(profile.get("top_bottlenecks"), list) else []:
        if not isinstance(raw, dict):
            continue
        item = {
            "stage": str(raw.get("stage") or ""),
            "duration_seconds": _float_or_none(raw.get("duration_seconds")),
            "timing_source": str(raw.get("timing_source") or ""),
        }
        execution = raw.get("execution") if isinstance(raw.get("execution"), dict) else {}
        if execution:
            item["execution"] = {
                "timed_out": bool(execution.get("timed_out")),
                "attempt_count": int(_float_or_none(execution.get("attempt_count")) or 0),
                "total_elapsed_seconds": _float_or_none(execution.get("total_elapsed_seconds")),
            }
        prompt = raw.get("prompt_budget") if isinstance(raw.get("prompt_budget"), dict) else {}
        if prompt:
            item["prompt_budget"] = {
                "prompt_calls": int(_float_or_none(prompt.get("prompt_calls")) or 0),
                "max_prompt_chars": int(_float_or_none(prompt.get("max_prompt_chars")) or 0),
                "max_prompt_seconds": _float_or_none(prompt.get("max_prompt_seconds")),
            }
        bottlenecks.append({key: value for key, value in item.items() if value is not None and value != ""})
        if len(bottlenecks) >= 5:
            break
    return {
        "total_known_seconds": _float_or_none(profile.get("total_known_seconds")),
        "slowest_stage": str(profile.get("slowest_stage") or ""),
        "top_bottlenecks": bottlenecks,
        "quality_flags": [
            str(flag)
            for flag in profile.get("quality_flags", [])
            if str(flag or "").strip()
        ][:12]
        if isinstance(profile.get("quality_flags"), list)
        else [],
        "cache_hit_stages": [
            str(stage)
            for stage in cache_summary.get("cache_hit_stages", [])
            if str(stage or "").strip()
        ][:12]
        if isinstance(cache_summary.get("cache_hit_stages"), list)
        else [],
        "cache_hit_count": int(_float_or_none(cache_summary.get("cache_hit_count")) or 0),
    }


def _float_or_none(value: Any) -> float | None:
    try:
        return round(float(value), 4)
    except Exception:
        return None


def _rate_or_none(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 3)


def _benchmark_score_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    scores: list[float] = []
    matched_total = 0.0
    expected_total = 0.0
    for item in items:
        score = _float_or_none(_benchmark_score_value(item))
        if score is not None:
            scores.append(score)
        matched = _float_or_none(item.get("benchmark_matched_slots"))
        expected = _float_or_none(item.get("benchmark_expected_slots"))
        if matched is not None and expected is not None and expected > 0:
            matched_total += matched
            expected_total += expected
    weighted_score = _rate_or_none(matched_total, expected_total)
    mean_score = round(sum(scores) / len(scores), 3) if scores else None
    return {
        "scored_cases": len(scores),
        "total_cases": len(items),
        "weighted_overall_benchmark_score": weighted_score,
        "mean_overall_benchmark_score": mean_score,
        "matched_slots": int(matched_total) if matched_total.is_integer() else round(matched_total, 3),
        "expected_slots": int(expected_total) if expected_total.is_integer() else round(expected_total, 3),
        "missing_slots": int(max(0.0, expected_total - matched_total))
        if expected_total.is_integer() and matched_total.is_integer()
        else round(max(0.0, expected_total - matched_total), 3),
        "extra_content_penalized": False,
    }


def _first_present(row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in row:
            value = row.get(key)
            if value is not None and value != "":
                return value
    return None


def _score_priority_rows(rows: list[dict[str, Any]], *, limit: int = 8) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for row in rows:
        score = _float_or_none(
            _first_present(row, ["overall_benchmark_score", "latest_overall_benchmark_score"])
        )
        if score is None:
            continue
        missing_slots = _float_or_none(
            _first_present(row, ["benchmark_missing_slots", "latest_benchmark_missing_slots"])
        )
        gap_count = _int_or_none(
            _first_present(row, ["claim_requirement_gap_count", "latest_claim_requirement_gap_count"])
        )
        ranked.append(
            {
                "case_id": str(row.get("case_id") or ""),
                "decision": str(row.get("decision") or row.get("latest_decision") or ""),
                "state": str(row.get("state") or ""),
                "overall_benchmark_score": score,
                "benchmark_missing_slots": missing_slots,
                "claim_requirement_gap_count": gap_count,
                "benchmark_gap_summary": str(
                    row.get("benchmark_gap_summary") or row.get("latest_benchmark_gap_summary") or ""
                ),
                "next_focus": row.get("next_focus") if isinstance(row.get("next_focus"), list) else [],
                "run_dir": str(row.get("run_dir") or ""),
            }
        )
    ranked.sort(
        key=lambda row: (
            float(row["overall_benchmark_score"]),
            -float(row["benchmark_missing_slots"] or 0),
            -int(row["claim_requirement_gap_count"] or 0),
            str(row["case_id"]),
        )
    )
    return ranked[:limit]


def _score_trend_summary(by_case: dict[str, list[dict[str, Any]]], *, limit: int = 10) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for case_id, rows in by_case.items():
        scored = [
            row
            for row in rows
            if _float_or_none(row.get("overall_benchmark_score")) is not None
        ]
        if len(scored) < 2:
            continue
        previous = scored[-2]
        latest = scored[-1]
        previous_score = _float_or_none(previous.get("overall_benchmark_score"))
        latest_score = _float_or_none(latest.get("overall_benchmark_score"))
        if previous_score is None or latest_score is None:
            continue
        delta = round(latest_score - previous_score, 4)
        previous_missing = _float_or_none(previous.get("benchmark_missing_slots"))
        latest_missing = _float_or_none(latest.get("benchmark_missing_slots"))
        missing_delta = None
        if previous_missing is not None and latest_missing is not None:
            missing_delta = round(latest_missing - previous_missing, 4)
        previous_digest = str(previous.get("benchmark_definition_digest") or "")
        latest_digest = str(latest.get("benchmark_definition_digest") or "")
        if previous_digest and latest_digest:
            definition_match: bool | None = previous_digest == latest_digest
        else:
            definition_match = None
        if delta > 0:
            direction = "improved"
        elif delta < 0:
            direction = "regressed"
        else:
            direction = "unchanged"
        cases.append(
            {
                "case_id": str(case_id),
                "direction": direction,
                "previous_score": previous_score,
                "latest_score": latest_score,
                "score_delta": delta,
                "previous_missing_slots": previous_missing,
                "latest_missing_slots": latest_missing,
                "missing_slots_delta": missing_delta,
                "benchmark_definition_match": definition_match,
                "previous_benchmark_definition_digest": previous_digest,
                "latest_benchmark_definition_digest": latest_digest,
                "previous_decision": str(previous.get("decision") or ""),
                "latest_decision": str(latest.get("decision") or ""),
                "previous_generated_at": str(previous.get("generated_at") or ""),
                "latest_generated_at": str(latest.get("generated_at") or ""),
                "previous_run_dir": str(previous.get("run_dir") or ""),
                "latest_run_dir": str(latest.get("run_dir") or ""),
            }
        )
    cases.sort(
        key=lambda row: (
            {"regressed": 0, "unchanged": 1, "improved": 2}.get(str(row["direction"]), 3),
            float(row["score_delta"]),
            str(row["case_id"]),
        )
    )
    all_deltas = [float(row["score_delta"]) for row in cases]
    comparable_cases = [row for row in cases if row["benchmark_definition_match"] is True]
    comparable_deltas = [float(row["score_delta"]) for row in comparable_cases]
    return {
        "scored_trend_cases": len(cases),
        "comparable_cases": len(comparable_cases),
        "improved": sum(1 for row in cases if row["direction"] == "improved"),
        "regressed": sum(1 for row in cases if row["direction"] == "regressed"),
        "unchanged": sum(1 for row in cases if row["direction"] == "unchanged"),
        "comparable_improved": sum(1 for row in comparable_cases if row["direction"] == "improved"),
        "comparable_regressed": sum(1 for row in comparable_cases if row["direction"] == "regressed"),
        "comparable_unchanged": sum(1 for row in comparable_cases if row["direction"] == "unchanged"),
        "definition_matched": sum(1 for row in cases if row["benchmark_definition_match"] is True),
        "definition_mismatched": sum(1 for row in cases if row["benchmark_definition_match"] is False),
        "definition_missing": sum(1 for row in cases if row["benchmark_definition_match"] is None),
        "mean_score_delta": round(sum(comparable_deltas) / len(comparable_deltas), 4)
        if comparable_deltas
        else None,
        "all_score_delta_mean": round(sum(all_deltas) / len(all_deltas), 4) if all_deltas else None,
        "cases": cases[:limit],
    }


def _benchmark_definition_match_current(row: dict[str, Any], current_digest: str) -> bool | None:
    digest = str(row.get("benchmark_definition_digest") or "")
    if not digest or not current_digest:
        return None
    return digest == current_digest


def _latest_definition_summary(rows: list[dict[str, Any]], current_digest: str) -> dict[str, Any]:
    matching = 0
    mismatched = 0
    missing = 0
    scored_matching = 0
    scored_mismatched = 0
    scored_missing = 0
    for row in rows:
        score_present = _float_or_none(row.get("overall_benchmark_score")) is not None
        match = _benchmark_definition_match_current(row, current_digest)
        if match is True:
            matching += 1
            if score_present:
                scored_matching += 1
        elif match is False:
            mismatched += 1
            if score_present:
                scored_mismatched += 1
        else:
            missing += 1
            if score_present:
                scored_missing += 1
    return {
        "total_cases": len(rows),
        "matching_current": matching,
        "mismatched_current": mismatched,
        "missing_definition": missing,
        "scored_matching_current": scored_matching,
        "scored_mismatched_current": scored_mismatched,
        "scored_missing_definition": scored_missing,
    }


def _definition_match_for_row(row: dict[str, Any], current_digest: str) -> bool | None:
    if "latest_benchmark_definition_match_current" in row:
        value = row.get("latest_benchmark_definition_match_current")
        return value if isinstance(value, bool) else None
    return _benchmark_definition_match_current(row, current_digest)


def _definition_refresh_rows(rows: list[dict[str, Any]], current_digest: str, *, limit: int = 10) -> list[dict[str, Any]]:
    refresh: list[dict[str, Any]] = []
    for row in rows:
        match = _definition_match_for_row(row, current_digest)
        if match is True:
            continue
        score = _float_or_none(
            _first_present(row, ["overall_benchmark_score", "latest_overall_benchmark_score"])
        )
        if score is None:
            continue
        missing_slots = _float_or_none(
            _first_present(row, ["benchmark_missing_slots", "latest_benchmark_missing_slots"])
        )
        gap_count = _int_or_none(
            _first_present(row, ["claim_requirement_gap_count", "latest_claim_requirement_gap_count"])
        )
        reason = "definition_mismatch" if match is False else "missing_definition"
        next_focus = row.get("next_focus") if isinstance(row.get("next_focus"), list) else []
        refresh.append(
            {
                "case_id": str(row.get("case_id") or ""),
                "reason": reason,
                "definition_match_current": match,
                "overall_benchmark_score": score,
                "benchmark_missing_slots": missing_slots,
                "claim_requirement_gap_count": gap_count,
                "decision": str(row.get("decision") or row.get("latest_decision") or ""),
                "next_focus": _unique(["refresh_current_benchmark_definition", *next_focus]),
                "run_dir": str(row.get("run_dir") or ""),
            }
        )
    refresh.sort(
        key=lambda row: (
            0 if row["reason"] == "definition_mismatch" else 1,
            float(row["overall_benchmark_score"]),
            -float(row["benchmark_missing_slots"] or 0),
            -int(row["claim_requirement_gap_count"] or 0),
            str(row["case_id"]),
        )
    )
    return refresh[:limit]


def _needs_text_cache_disabled_run(row: dict[str, Any]) -> bool:
    next_focus = row.get("next_focus") if isinstance(row.get("next_focus"), list) else []
    return any(str(item) == "resolve_text_cache_validity_conflict" for item in next_focus)


def _history_output_change_summary(rows: list[dict[str, Any]]) -> dict[str, int]:
    summary = {
        "diagnostic_only": 0,
        "output_risk": 0,
        "unknown": 0,
        "missing_audit": 0,
    }
    for row in rows:
        if row.get("output_change_audit_available") is not True:
            summary["missing_audit"] += 1
        elif row.get("output_change_diagnostic_only") is True:
            summary["diagnostic_only"] += 1
        elif (row.get("output_change_output_risk_count") or 0) > 0:
            summary["output_risk"] += 1
        else:
            summary["unknown"] += 1
    return summary


BENCHMARK_ONLY_PREFIXES = (
    ".codex/skills/papereval-",
    "backend/tests/",
    "benchmarks/",
    "docs/",
)
BENCHMARK_ONLY_FILES = {
    ".gitignore",
    "README.md",
    "backend/tests/test_evidence_gold_compatibility.py",
    "backend/tests/test_multi_paper_benchmark.py",
    "backend/tests/test_papereval_benchmark_helper.py",
    "docs/app-evaluation-benchmark.md",
    "scripts/batch_upstream_ab.py",
    "scripts/compare_evidence_to_gold.py",
    "scripts/compare_pdf_against_reference.py",
    "scripts/compare_upstream_ab.py",
    "scripts/run_multi_paper_benchmark.py",
    "scripts/validate_gold_claims.py",
    "scripts/validate_gold_standards.py",
}
OUTPUT_RISK_PREFIXES = (
    "assets/icons/",
    "backend/app/",
    "desktop_packaging/",
    "desktop_shell/",
    "desktop_ui/",
    "frontend/",
    "app/",
    "desktop/",
    "macos/",
)
OUTPUT_RISK_FILES = {
    ".github/workflows/ci.yml",
    "Makefile",
    "PsychPaperEvalApp-Image.png",
    "PsychPaperEvalApp-Image2.png",
    "backend/.env.example",
    "backend/requirements.txt",
    "scripts/build_macos_app.py",
    "scripts/run_app.py",
}
OUTPUT_RISK_KEYWORDS = (
    "analysis",
    "extract",
    "grobid",
    "llm",
    "model",
    "parser",
    "prompt",
    "provider",
    "report",
    "synthesis",
)


def _normalize_changed_path(path: str) -> str:
    normalized = path.strip()
    if " -> " in normalized:
        normalized = normalized.rsplit(" -> ", 1)[-1].strip()
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _classify_output_change_path(path: str) -> str:
    normalized = _normalize_changed_path(path)
    if not normalized:
        return "unknown"
    if normalized in BENCHMARK_ONLY_FILES or normalized.startswith(BENCHMARK_ONLY_PREFIXES):
        return "benchmark_only"
    if normalized in OUTPUT_RISK_FILES or normalized.startswith(OUTPUT_RISK_PREFIXES):
        return "output_risk"
    lowered = normalized.lower()
    if any(keyword in lowered for keyword in OUTPUT_RISK_KEYWORDS):
        return "output_risk"
    return "unknown"


def _output_change_path_category(path: str) -> str:
    normalized = _normalize_changed_path(path)
    if normalized.startswith("backend/app/"):
        return "backend_app"
    if normalized.startswith(("desktop_ui/", "frontend/")):
        return "desktop_ui"
    if normalized.startswith(("desktop_shell/", "desktop_packaging/", "desktop/", "macos/")):
        return "desktop_launcher"
    if normalized.startswith("assets/icons/") or normalized.lower().endswith((".icns", ".ico", ".png")):
        return "app_assets"
    if normalized in {".github/workflows/ci.yml", "Makefile"}:
        return "ci_or_build"
    if normalized in {"backend/.env.example", "backend/requirements.txt"}:
        return "dependency_or_env"
    if normalized.startswith("scripts/"):
        return "scripts"
    if normalized.startswith(".codex/skills/"):
        return "qa_skill"
    if normalized.startswith("backend/tests/"):
        return "tests"
    if normalized.startswith("benchmarks/"):
        return "benchmark_fixture"
    if normalized.startswith("docs/") or normalized in {"README.md", ".gitignore"}:
        return "docs_or_metadata"
    return "other"


def _output_change_category_counts(paths: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in paths:
        category = _output_change_path_category(path)
        counts[category] = counts.get(category, 0) + 1
    return dict(sorted(counts.items()))


def _classify_output_change_paths(paths: list[str]) -> dict[str, Any]:
    groups: dict[str, list[str]] = {
        "benchmark_only": [],
        "output_risk": [],
        "unknown": [],
    }
    for raw_path in paths:
        path = _normalize_changed_path(raw_path)
        if not path:
            continue
        groups[_classify_output_change_path(path)].append(path)
    for key in groups:
        groups[key] = sorted(set(groups[key]))
    return {
        "benchmark_only": groups["benchmark_only"],
        "output_risk": groups["output_risk"],
        "unknown": groups["unknown"],
        "benchmark_only_category_counts": _output_change_category_counts(groups["benchmark_only"]),
        "output_risk_category_counts": _output_change_category_counts(groups["output_risk"]),
        "unknown_category_counts": _output_change_category_counts(groups["unknown"]),
        "changed_count": sum(len(items) for items in groups.values()),
        "output_risk_count": len(groups["output_risk"]),
        "unknown_count": len(groups["unknown"]),
        "diagnostic_only": not groups["output_risk"] and not groups["unknown"],
    }


def _git_changed_paths(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit((result.stderr or result.stdout or "git status failed").strip())
    paths: list[str] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        paths.append(line[3:] if len(line) > 3 else line.strip())
    return paths


def _current_output_change_audit() -> dict[str, Any]:
    return _classify_output_change_paths(_git_changed_paths(ROOT))


def _output_change_audit(*, json_output: bool = False, fail_on_output_risk: bool = False) -> int:
    audit = _current_output_change_audit()
    if json_output:
        print(json.dumps(audit, indent=2, sort_keys=True))
    else:
        print(f"changed_files: {audit['changed_count']}")
        print(f"diagnostic_only: {audit['diagnostic_only']}")
        print(f"output_risk_count: {audit['output_risk_count']}")
        print(f"unknown_count: {audit['unknown_count']}")
        for label in ("output_risk", "unknown", "benchmark_only"):
            counts = audit.get(f"{label}_category_counts")
            if counts:
                summary = ", ".join(f"{key}={value}" for key, value in counts.items())
                print(f"{label}_categories: {summary}")
        for label in ("output_risk", "unknown", "benchmark_only"):
            paths = audit.get(label, [])
            if not paths:
                continue
            print(f"\n{label}:")
            for path in paths[:30]:
                print(f"- {path}")
            if len(paths) > 30:
                print(f"- ... {len(paths) - 30} additional file(s)")
    if fail_on_output_risk and not audit["diagnostic_only"]:
        return 1
    return 0


def _require_diagnostic_only_worktree() -> dict[str, Any]:
    audit = _current_output_change_audit()
    if not audit["diagnostic_only"]:
        output_risk_categories = audit.get("output_risk_category_counts")
        unknown_categories = audit.get("unknown_category_counts")
        category_bits = []
        if isinstance(output_risk_categories, dict) and output_risk_categories:
            category_bits.append(
                "output_risk_categories="
                + ",".join(f"{key}:{value}" for key, value in output_risk_categories.items())
            )
        if isinstance(unknown_categories, dict) and unknown_categories:
            category_bits.append(
                "unknown_categories="
                + ",".join(f"{key}:{value}" for key, value in unknown_categories.items())
            )
        category_text = f" {'; '.join(category_bits)}. " if category_bits else " "
        raise SystemExit(
            "active benchmark requires a diagnostic-only worktree; "
            f"output_risk={audit['output_risk_count']} unknown={audit['unknown_count']}. "
            f"{category_text}"
            "Run --output-change-audit --json for details or omit --require-diagnostic-only."
        )
    return audit


def _compact_output_change_audit(audit: Any) -> dict[str, Any]:
    if not isinstance(audit, dict):
        return {
            "available": False,
            "diagnostic_only": None,
            "changed_count": None,
            "output_risk_count": None,
            "unknown_count": None,
            "output_risk_category_counts": {},
            "unknown_category_counts": {},
        }
    return {
        "available": True,
        "diagnostic_only": audit.get("diagnostic_only") if isinstance(audit.get("diagnostic_only"), bool) else None,
        "changed_count": _int_or_none(audit.get("changed_count")),
        "output_risk_count": _int_or_none(audit.get("output_risk_count")),
        "unknown_count": _int_or_none(audit.get("unknown_count")),
        "output_risk_category_counts": audit.get("output_risk_category_counts")
        if isinstance(audit.get("output_risk_category_counts"), dict)
        else {},
        "unknown_category_counts": audit.get("unknown_category_counts")
        if isinstance(audit.get("unknown_category_counts"), dict)
        else {},
    }


def _new_record(case: dict[str, Any], args: argparse.Namespace, api_base: str, queue_order: int) -> dict[str, Any]:
    scoring = str(case.get("scoring") or "")
    return {
        "case_id": str(case.get("id")),
        "scoring": scoring,
        "release_gate": scoring == "reference_comparison",
        "gold_standard": str(case.get("gold_standard") or ""),
        "gold_standard_status": str(case.get("gold_standard_status") or ""),
        "reference_status": str(case.get("reference_status") or ""),
        "surface": args.surface,
        "api_base": api_base,
        "queue_order": queue_order,
        "helper_max_concurrent": int(args.max_concurrent),
        "disable_local_text_cache": _effective_disable_local_text_cache(args),
        "ok": False,
        "decision": "queued",
        "failures": [],
    }


def _write_record(case_dir: Path, record: dict[str, Any]) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


def _failure_decision(record: dict[str, Any]) -> str:
    return "fail" if bool(record.get("release_gate")) else "diagnostic_fail"


def _mark_failed(record: dict[str, Any], case_dir: Path, message: str, *, decision: str = "fail") -> dict[str, Any]:
    if decision == "fail":
        decision = _failure_decision(record)
    record.update(
        {
            "ok": False,
            "decision": decision,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "failures": [message],
        }
    )
    _write_record(case_dir, record)
    print(f"  {decision} {record['case_id']} {message}", file=sys.stderr)
    return record


def _process_exit_status(
    records: list[dict[str, Any]],
    *,
    fail_on_incompatible: bool,
    fail_on_diagnostic: bool,
) -> int:
    for record in records:
        decision = str(record.get("decision") or "")
        if decision == "fail" and fail_on_incompatible:
            return 1
        if decision == "diagnostic_fail" and fail_on_diagnostic:
            return 1
    return 0


def _update_active_timing(
    state: ActiveCaseState,
    job: dict[str, Any],
    *,
    now: float,
    queue_timeout: float,
    timeout_per_case: float,
) -> str | None:
    job_id = job.get("id") or state.record.get("job_id") or "unknown"
    job_status = str(job.get("status") or "").lower()
    if job_status == "running" and state.running_deadline is None:
        state.running_deadline = now + timeout_per_case
        state.record["running_started_at"] = datetime.now(timezone.utc).isoformat()
        state.record["run_timeout_seconds"] = timeout_per_case
    if job_status == "queued" and now - state.submitted_at > queue_timeout:
        return f"job {job_id} did not start within {queue_timeout:g}s; last={job}"
    if job_status == "running" and state.running_deadline is not None and now > state.running_deadline:
        return f"job {job_id} did not finish within {timeout_per_case:g}s after it started; last={job}"
    return None


def _finish_uploaded_case(
    *,
    api_base: str,
    case: dict[str, Any],
    record: dict[str, Any],
    case_dir: Path,
    job: dict[str, Any],
    artifact_url_base: str = "",
) -> dict[str, Any]:
    document_id = int(record["document_id"])
    report: dict[str, Any] = {}
    comparison: dict[str, Any] = {}
    diagnostics = _diagnostic_extract(report, job)
    failures: list[str] = []

    if str(job.get("status") or "").lower() != "completed":
        failures.append(f"job status is {job.get('status')}")
    else:
        report = _request_json(api_base, f"documents/{document_id}/report", timeout=60.0)
        comparison = _compare_report_to_gold(case, report)
        diagnostics = _diagnostic_extract(report, job)
        compatible = bool(comparison.get("comparison", {}).get("compatible"))
        if report.get("report_invalid"):
            failures.append(str(report.get("report_invalid_reason") or "report invalid"))
        if not compatible:
            failures.append("evidence-to-gold comparison is incompatible")
        report_path = case_dir / "report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        webapp_url = _webapp_detailed_analysis_url(api_base, record)
        media = _fetch_document_media(api_base, document_id)
        media_json_path: Path | None = None
        if media:
            media = _download_static_media_assets(api_base, case_dir, media)
            media_json_path = case_dir / "media.json"
            media_json_path.write_text(json.dumps(media, indent=2), encoding="utf-8")
        detailed_analysis_url = _artifact_url(
            artifact_url_base,
            str(case.get("id") or record.get("case_id") or ""),
            "report.html",
        )
        html_path = _write_detailed_analysis_html(
            report,
            case_dir / "report.html",
            webapp_url=webapp_url,
            media=media,
            api_base=api_base,
            comparison=comparison,
        )
        slack_summary_path = _write_slack_summary_markdown(
            report,
            case_dir / "slack_summary.md",
            detailed_analysis_url=detailed_analysis_url,
            webapp_url=webapp_url,
            comparison=comparison,
        )
        record["artifacts"] = _artifact_metadata(
            report_path=report_path,
            html_path=html_path,
            slack_summary_path=slack_summary_path,
            media_json_path=media_json_path,
            detailed_analysis_url=detailed_analysis_url,
            webapp_url=webapp_url,
        )

    record.update(
        {
            "ok": not failures,
            "decision": "pass" if not failures else _failure_decision(record),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "job": job,
            "diagnostics": diagnostics,
            "comparison": comparison,
            "failures": failures,
        }
    )
    record["iteration_diagnostics"] = _iteration_diagnostics(record)
    _write_record(case_dir, record)
    metrics = comparison.get("comparison", {}) if isinstance(comparison.get("comparison"), dict) else {}
    print(
        "  "
        + record["decision"]
        + " "
        + record["case_id"]
        + " usable="
        + str(metrics.get("usable_packet_rate", "n/a"))
        + " claims="
        + str(metrics.get("critical_claim_candidate_rate", "n/a"))
        + " score="
        + str(metrics.get("overall_benchmark_score", "n/a"))
    )
    return record


def _write_detailed_report_from_path(
    path: Path,
    *,
    detailed_analysis_url: str = "",
    webapp_url: str = "",
    media_json: str = "",
    fetch_media_assets: bool = False,
    api_base: str = "",
) -> int:
    report_path = path
    if path.is_dir():
        report_path = path / "report.json"
    report = _load_json(report_path)
    media: dict[str, Any] = {}
    media_json_path: Path | None = None
    if media_json:
        media_json_path = Path(media_json).expanduser()
        media = _load_json(media_json_path)
    elif (report_path.parent / "media.json").exists():
        media_json_path = report_path.parent / "media.json"
        media = _load_json(media_json_path)
    if fetch_media_assets:
        resolved_api_base = api_base or _api_base_from_webapp_url(webapp_url)
        document = report.get("document") if isinstance(report.get("document"), dict) else {}
        document_id = document.get("id")
        fetched = _fetch_document_media(resolved_api_base, document_id)
        if fetched:
            media = _download_static_media_assets(resolved_api_base, report_path.parent, fetched)
            media_json_path = report_path.parent / "media.json"
            media_json_path.write_text(json.dumps(media, indent=2), encoding="utf-8")
    comparison: dict[str, Any] = {}
    record = _read_optional_json_object(report_path.parent / "record.json")
    if record and isinstance(record.get("comparison"), dict):
        comparison = record["comparison"]
    case_id = str(record.get("case_id") or report_path.parent.name) if record else report_path.parent.name
    case = _manifest_case_by_id(case_id)
    if case is not None:
        comparison = _compare_report_to_gold(case, report)
    html_path = _write_detailed_analysis_html(
        report,
        report_path.with_suffix(".html"),
        webapp_url=webapp_url,
        media=media,
        api_base=api_base or _api_base_from_webapp_url(webapp_url),
        comparison=comparison,
    )
    slack_summary_path = _write_slack_summary_markdown(
        report,
        report_path.with_name("slack_summary.md"),
        detailed_analysis_url=detailed_analysis_url,
        webapp_url=webapp_url,
        comparison=comparison,
    )
    updated_metadata = _backfill_artifact_metadata(
        report_path.parent,
        _artifact_metadata(
            report_path=report_path,
            html_path=html_path,
            slack_summary_path=slack_summary_path,
            media_json_path=media_json_path,
            detailed_analysis_url=detailed_analysis_url,
            webapp_url=webapp_url,
        ),
    )
    updated_comparison = _refresh_case_comparison_metadata(report_path.parent, report)
    print(f"detailed_analysis_html={html_path}")
    print(f"slack_summary_markdown={slack_summary_path}")
    if media_json_path is not None:
        print(f"media_json={media_json_path}")
    for updated_path in updated_metadata:
        print(f"updated_artifact_metadata={updated_path}")
    for updated_path in updated_comparison:
        print(f"updated_comparison_metadata={updated_path}")
    return 0


def _summarize_existing_run(path: Path, *, json_output: bool = False) -> int:
    summary_path = path
    if path.is_dir():
        summary_path = path / "active_benchmark_summary.json"
    summary = _load_json(summary_path)
    records = summary.get("records", [])
    if not isinstance(records, list):
        raise SystemExit(f"{summary_path} field `records` must be a list.")
    compact_records: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        iteration = _merged_iteration_diagnostics(record)
        artifacts = record.get("artifacts") if isinstance(record.get("artifacts"), dict) else {}
        compact_records.append(
            {
                "case_id": str(iteration.get("case_id") or record.get("case_id") or "unknown"),
                "iteration_diagnostics": iteration,
                "failures": iteration.get("failures") if isinstance(iteration.get("failures"), list) else [],
                "artifacts": {
                    "detailed_analysis_html": str(artifacts.get("detailed_analysis_html") or ""),
                    "detailed_analysis_url": str(artifacts.get("detailed_analysis_url") or ""),
                    "webapp_detailed_analysis_url": str(artifacts.get("webapp_detailed_analysis_url") or ""),
                    "slack_summary_markdown": str(artifacts.get("slack_summary_markdown") or ""),
                    "media_json": str(artifacts.get("media_json") or ""),
                    "static_media_dir": str(artifacts.get("static_media_dir") or ""),
                },
            }
        )
    score_summary = _benchmark_score_summary(
        [compact["iteration_diagnostics"] for compact in compact_records]
    )
    output_change_audit = summary.get("output_change_audit") if isinstance(summary.get("output_change_audit"), dict) else {}
    compact_audit = _compact_output_change_audit(output_change_audit)
    benchmark_definition = summary.get("benchmark_definition") if isinstance(summary.get("benchmark_definition"), dict) else {}
    compact_definition = _compact_benchmark_definition(benchmark_definition)

    if json_output:
        payload = {
            "summary_path": str(summary_path),
            "generated_at": str(summary.get("generated_at") or ""),
            "surface": str(summary.get("surface") or ""),
            "benchmark_definition": benchmark_definition,
            "benchmark_definition_summary": compact_definition,
            "output_change_audit": output_change_audit,
            "output_change_audit_summary": compact_audit,
            "benchmark_score_summary": score_summary,
            "records": compact_records,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    print(f"summary: {summary_path}")
    print(f"generated_at: {summary.get('generated_at', '')}")
    print(f"surface: {summary.get('surface', '')}")
    print(f"records: {len(compact_records)}")
    print(
        "benchmark_definition: "
        f"available={compact_definition.get('available')} "
        f"digest={compact_definition.get('digest')} "
        f"files={compact_definition.get('file_count')}"
    )
    print(
        "output_change_audit: "
        f"available={compact_audit.get('available')} "
        f"diagnostic_only={compact_audit.get('diagnostic_only')} "
        f"output_risk={compact_audit.get('output_risk_count')} "
        f"unknown={compact_audit.get('unknown_count')}"
    )
    print(
        "benchmark_score_summary: "
        f"weighted={score_summary.get('weighted_overall_benchmark_score')} "
        f"mean={score_summary.get('mean_overall_benchmark_score')} "
        f"scored={score_summary.get('scored_cases')}/{score_summary.get('total_cases')} "
        f"slots={score_summary.get('matched_slots')}/{score_summary.get('expected_slots')}"
    )
    for compact in compact_records:
        iteration = compact["iteration_diagnostics"]
        case_id = compact["case_id"]
        print(
            "- "
            f"{case_id}: decision={iteration.get('decision', '')} "
            f"compatible={iteration.get('compatible', None)} "
            f"usable={iteration.get('usable_packet_rate', None)} "
            f"claims={iteration.get('critical_claim_candidate_rate', None)} "
            f"slowest={iteration.get('slowest_stage', '')}"
            f"({iteration.get('slowest_stage_seconds', None)}s) "
            f"score={iteration.get('overall_benchmark_score', None)} "
            f"gaps={iteration.get('claim_requirement_gap_count', None)} "
            f"missing_slots={iteration.get('benchmark_missing_slots', None)}"
        )
        next_focus = iteration.get("next_focus") if isinstance(iteration.get("next_focus"), list) else []
        if next_focus:
            print(f"  next_focus: {', '.join(str(item) for item in next_focus[:8])}")
        failures = iteration.get("failures") if isinstance(iteration.get("failures"), list) else []
        if failures:
            print(f"  failures: {'; '.join(str(item) for item in failures[:3])}")
        gap_summary = str(iteration.get("benchmark_gap_summary") or "")
        if gap_summary:
            print(f"  benchmark_gaps: {gap_summary}")
    return 0


def _history_summary_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise SystemExit(f"History path does not exist: {path}")
    return sorted(path.glob("**/active_benchmark_summary.json"))


def _history_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in _history_summary_files(path):
        summary = _load_json(summary_path)
        generated_at = str(summary.get("generated_at") or "")
        records = summary.get("records", [])
        if not isinstance(records, list):
            continue
        output_change_audit = _compact_output_change_audit(summary.get("output_change_audit"))
        benchmark_definition = _compact_benchmark_definition(summary.get("benchmark_definition"))
        for record in records:
            if not isinstance(record, dict):
                continue
            iteration = _merged_iteration_diagnostics(record)
            artifacts = record.get("artifacts") if isinstance(record.get("artifacts"), dict) else {}
            case_id = str(iteration.get("case_id") or record.get("case_id") or "")
            case_dir = summary_path.parent / case_id if case_id else summary_path.parent
            fallback_html = case_dir / "report.html"
            fallback_slack = case_dir / "slack_summary.md"
            fallback_media_json = case_dir / "media.json"
            fallback_media_dir = case_dir / "media"
            rows.append(
                {
                    "summary_path": str(summary_path.resolve()),
                    "run_dir": str(summary_path.parent.resolve()),
                    "generated_at": generated_at,
                    "surface": str(summary.get("surface") or ""),
                    "benchmark_definition_available": benchmark_definition.get("available"),
                    "benchmark_definition_digest": benchmark_definition.get("digest"),
                    "benchmark_definition_algorithm": benchmark_definition.get("algorithm"),
                    "benchmark_definition_file_count": benchmark_definition.get("file_count"),
                    "output_change_audit_available": output_change_audit.get("available"),
                    "output_change_diagnostic_only": output_change_audit.get("diagnostic_only"),
                    "output_change_changed_count": output_change_audit.get("changed_count"),
                    "output_change_output_risk_count": output_change_audit.get("output_risk_count"),
                    "output_change_unknown_count": output_change_audit.get("unknown_count"),
                    "case_id": case_id,
                    "decision": str(iteration.get("decision") or record.get("decision") or ""),
                    "compatible": iteration.get("compatible"),
                    "overall_benchmark_score": iteration.get("overall_benchmark_score"),
                    "benchmark_content_score": iteration.get("benchmark_content_score"),
                    "benchmark_matched_slots": iteration.get("benchmark_matched_slots"),
                    "benchmark_expected_slots": iteration.get("benchmark_expected_slots"),
                    "benchmark_missing_slots": iteration.get("benchmark_missing_slots"),
                    "claim_requirement_gap_count": iteration.get("claim_requirement_gap_count"),
                    "benchmark_gap_summary": str(iteration.get("benchmark_gap_summary") or ""),
                    "usable_packet_rate": iteration.get("usable_packet_rate"),
                    "critical_claim_candidate_rate": iteration.get("critical_claim_candidate_rate"),
                    "slowest_stage": str(iteration.get("slowest_stage") or ""),
                    "slowest_stage_seconds": iteration.get("slowest_stage_seconds"),
                    "next_focus": iteration.get("next_focus") if isinstance(iteration.get("next_focus"), list) else [],
                    "failures": iteration.get("failures") if isinstance(iteration.get("failures"), list) else [],
                    "detailed_analysis_html": str(
                        artifacts.get("detailed_analysis_html")
                        or (fallback_html if fallback_html.exists() else "")
                    ),
                    "detailed_analysis_url": str(artifacts.get("detailed_analysis_url") or ""),
                    "webapp_detailed_analysis_url": str(artifacts.get("webapp_detailed_analysis_url") or ""),
                    "slack_summary_markdown": str(
                        artifacts.get("slack_summary_markdown")
                        or (fallback_slack if fallback_slack.exists() else "")
                    ),
                    "media_json": str(
                        artifacts.get("media_json")
                        or (fallback_media_json if fallback_media_json.exists() else "")
                    ),
                    "static_media_dir": str(
                        artifacts.get("static_media_dir")
                        or (fallback_media_dir if fallback_media_dir.exists() else "")
                    ),
                }
            )
    return sorted(rows, key=lambda row: (str(row.get("generated_at") or ""), str(row.get("case_id") or "")))


def _summarize_stage_diagnostics(path: Path, *, json_output: bool = False) -> int:
    rows = _history_rows(path)
    latest_by_case = _latest_rows_by_case(rows)
    manifest = _load_json(MANIFEST)
    gold_by_case = {
        str(case.get("id") or ""): str(case.get("gold_standard") or "")
        for case in manifest.get("cases", [])
        if isinstance(case, dict)
    }
    comparator = _load_module(SCRIPTS_DIR / "compare_evidence_to_gold.py", "compare_evidence_to_gold_stage_summary")
    validator = _load_module(SCRIPTS_DIR / "validate_gold_standards.py", "validate_gold_standards_stage_summary")

    case_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    aggregate_failure_counts: dict[str, int] = {}
    aggregate_item_type_counts: dict[str, int] = {}
    aggregate_failure_by_item_type: dict[str, dict[str, int]] = {}
    aggregate_improvement_lane_counts: dict[str, int] = {}
    aggregate_source_visibility_counts: dict[str, int] = {}
    aggregate_lane_source_visibility_counts: dict[str, dict[str, int]] = {}
    aggregate_stage_presence = {
        stage: {"present": 0, "total_expected_items": 0}
        for stage in ("extracted_text", "evidence_packets", "synthesis_inputs", "final_report")
    }

    for case_id in sorted(latest_by_case):
        row = latest_by_case[case_id]
        report_path = Path(str(row.get("run_dir") or "")) / case_id / "report.json"
        gold_rel = gold_by_case.get(case_id, "")
        gold_path = ROOT / gold_rel if gold_rel else Path("")
        if not report_path.exists():
            skipped.append({"case_id": case_id, "reason": f"missing report.json: {report_path}"})
            continue
        if not gold_rel or not gold_path.exists():
            skipped.append({"case_id": case_id, "reason": f"missing gold_standard: {gold_rel}"})
            continue
        report = _load_json(report_path)
        gold = validator.load_gold_standard(gold_path)
        comparison = comparator.compare_evidence_to_gold(
            comparator.evidence_packets_from_payload(report),
            gold,
            evidence_metadata=comparator.evidence_metadata_from_payload(report),
        )
        stage = comparator._build_artifact_stage_diagnostics(report, gold, comparison)
        failure_counts = _int_dict(stage.get("failure_point_counts"))
        item_type_counts = _int_dict(stage.get("item_type_counts"))
        failure_by_item_type = _nested_int_dict(stage.get("failure_point_by_item_type"))
        lane_counts = _count_stage_missing_item_field(stage, "improvement_lane")
        visibility_counts = _source_visibility_counts(stage)
        lane_visibility_counts = _stage_lane_source_visibility_counts(stage)
        representative_examples = _stage_representative_examples(stage)
        stage_presence = stage.get("stage_presence_counts") if isinstance(stage.get("stage_presence_counts"), dict) else {}
        for key, count in failure_counts.items():
            aggregate_failure_counts[key] = aggregate_failure_counts.get(key, 0) + count
        for key, count in item_type_counts.items():
            aggregate_item_type_counts[key] = aggregate_item_type_counts.get(key, 0) + count
        for key, count in lane_counts.items():
            aggregate_improvement_lane_counts[key] = aggregate_improvement_lane_counts.get(key, 0) + count
        for key, count in visibility_counts.items():
            aggregate_source_visibility_counts[key] = aggregate_source_visibility_counts.get(key, 0) + count
        _merge_nested_counts(aggregate_lane_source_visibility_counts, lane_visibility_counts)
        _merge_nested_counts(aggregate_failure_by_item_type, failure_by_item_type)
        for stage_name, counts in stage_presence.items():
            if not isinstance(counts, dict):
                continue
            target = aggregate_stage_presence.setdefault(stage_name, {"present": 0, "total_expected_items": 0})
            target["present"] += _int_or_zero(counts.get("present"))
            target["total_expected_items"] += _int_or_zero(counts.get("total_expected_items"))
        case_rows.append(
            {
                "case_id": case_id,
                "overall_benchmark_score": _float_or_none(comparison.get("overall_benchmark_score")),
                "matched_slots": _float_or_none(
                    (comparison.get("benchmark_content_score_basis") or {}).get("matched_slots")
                    if isinstance(comparison.get("benchmark_content_score_basis"), dict)
                    else None
                ),
                "expected_slots": _float_or_none(
                    (comparison.get("benchmark_content_score_basis") or {}).get("expected_slots")
                    if isinstance(comparison.get("benchmark_content_score_basis"), dict)
                    else None
                ),
                "failure_point_counts": failure_counts,
                "item_type_counts": item_type_counts,
                "failure_point_by_item_type": failure_by_item_type,
                "improvement_lane_counts": lane_counts,
                "source_visibility_counts": visibility_counts,
                "lane_source_visibility_counts": lane_visibility_counts,
                "representative_examples": representative_examples,
                "stage_presence_counts": stage_presence,
                "report_json": str(report_path),
                "gold_standard": str(gold_path),
            }
        )

    payload = {
        "history_root": str(path),
        "cases_analyzed": len(case_rows),
        "cases_skipped": len(skipped),
        "skipped": skipped,
        "aggregate_failure_point_counts": dict(
            sorted(aggregate_failure_counts.items(), key=lambda item: (-item[1], item[0]))
        ),
        "aggregate_item_type_counts": dict(
            sorted(aggregate_item_type_counts.items(), key=lambda item: (-item[1], item[0]))
        ),
        "aggregate_failure_point_by_item_type": aggregate_failure_by_item_type,
        "aggregate_improvement_lane_counts": dict(
            sorted(aggregate_improvement_lane_counts.items(), key=lambda item: (-item[1], item[0]))
        ),
        "aggregate_source_visibility_counts": dict(
            sorted(aggregate_source_visibility_counts.items(), key=lambda item: (-item[1], item[0]))
        ),
        "aggregate_lane_source_visibility_counts": {
            lane: dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
            for lane, counts in sorted(
                aggregate_lane_source_visibility_counts.items(),
                key=lambda item: (-sum(item[1].values()), item[0]),
            )
        },
        "aggregate_stage_presence_counts": aggregate_stage_presence,
        "cases": sorted(
            case_rows,
            key=lambda item: (
                -sum(_int_dict(item.get("failure_point_counts")).values()),
                str(item.get("case_id") or ""),
            ),
        ),
    }
    payload["recommended_inspection_queue"] = _recommended_stage_inspection_queue(payload)
    if json_output:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    print(f"stage_diagnostics_history: {path}")
    print(f"cases_analyzed: {payload['cases_analyzed']}")
    print(f"cases_skipped: {payload['cases_skipped']}")
    print("aggregate_stage_presence_counts:")
    for stage_name, counts in aggregate_stage_presence.items():
        print(f"- {stage_name}: {counts['present']}/{counts['total_expected_items']} expected items present")
    print("aggregate_failure_point_counts:")
    for key, count in payload["aggregate_failure_point_counts"].items():
        print(f"- {key}: {count}")
    print("aggregate_item_type_counts:")
    for key, count in payload["aggregate_item_type_counts"].items():
        print(f"- {key}: {count}")
    print("aggregate_improvement_lane_counts:")
    for key, count in payload["aggregate_improvement_lane_counts"].items():
        print(f"- {key}: {count}")
    print("aggregate_source_visibility_counts:")
    for key, count in payload["aggregate_source_visibility_counts"].items():
        print(f"- {key}: {count}")
    print("recommended_inspection_queue:")
    for entry in payload["recommended_inspection_queue"]:
        print(
            "- "
            f"{entry['rank']}. {entry['focus']}: "
            f"items={entry['count']} "
            f"visibility={_format_count_dict(_int_dict(entry.get('source_visibility_counts')))}"
        )
        print(f"  inspect: {entry['recommended_next_step']}")
        guardrail = str(entry.get("guardrail") or "").strip()
        if guardrail:
            print(f"  guardrail: {guardrail}")
    print("\nTop cases by missing expected items:")
    for case_row in payload["cases"][:10]:
        failure_counts = _int_dict(case_row.get("failure_point_counts"))
        print(
            "- "
            f"{case_row['case_id']}: "
            f"score={case_row.get('overall_benchmark_score')} "
            f"missing_items={sum(failure_counts.values())} "
            f"failures={_format_count_dict(failure_counts)}"
        )
        examples = case_row.get("representative_examples")
        if isinstance(examples, list) and examples:
            for example in examples[:2]:
                if not isinstance(example, dict):
                    continue
                print(
                    "  example: "
                    f"{example.get('improvement_lane', 'unknown')} "
                    f"{example.get('claim_id', 'claim')} "
                    f"{example.get('item_type', 'item')}={example.get('term', '')} "
                    f"failure={example.get('failure_point', 'unknown')} "
                    f"visibility={example.get('source_visibility', 'unknown')}"
                )
                trace = str(example.get("trace") or example.get("nearest") or "").strip()
                if trace:
                    print(f"    trace: {trace}")
    if skipped:
        print("\nSkipped cases:")
        for item in skipped[:10]:
            print(f"- {item['case_id']}: {item['reason']}")
    return 0


def _count_stage_missing_item_field(stage: dict[str, Any], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    items = stage.get("missing_items")
    if not isinstance(items, list):
        return counts
    for item in items:
        if not isinstance(item, dict):
            continue
        key = str(item.get(field) or "unknown").strip() or "unknown"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda row: (-row[1], row[0])))


def _source_visibility_counts(stage: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    items = stage.get("missing_items")
    if not isinstance(items, list):
        return counts
    for item in items:
        if not isinstance(item, dict):
            continue
        visibility = item.get("source_visibility")
        key = "unknown"
        if isinstance(visibility, dict):
            key = str(visibility.get("classification") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda row: (-row[1], row[0])))


def _stage_lane_source_visibility_counts(stage: dict[str, Any]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    items = stage.get("missing_items")
    if not isinstance(items, list):
        return counts
    for item in items:
        if not isinstance(item, dict):
            continue
        lane = str(item.get("improvement_lane") or "unknown").strip() or "unknown"
        visibility = item.get("source_visibility")
        visibility_key = "unknown"
        if isinstance(visibility, dict):
            visibility_key = str(visibility.get("classification") or "unknown").strip() or "unknown"
        lane_counts = counts.setdefault(lane, {})
        lane_counts[visibility_key] = lane_counts.get(visibility_key, 0) + 1
    return {
        lane: dict(sorted(lane_counts.items(), key=lambda row: (-row[1], row[0])))
        for lane, lane_counts in sorted(counts.items(), key=lambda row: (-sum(row[1].values()), row[0]))
    }


def _recommended_stage_inspection_queue(payload: dict[str, Any]) -> list[dict[str, Any]]:
    lane_visibility = _nested_int_dict(payload.get("aggregate_lane_source_visibility_counts"))
    lane_totals = _int_dict(payload.get("aggregate_improvement_lane_counts"))
    queue: list[dict[str, Any]] = []
    for lane, count in sorted(lane_totals.items(), key=lambda item: (-item[1], item[0])):
        visibility_counts = dict(
            sorted(lane_visibility.get(lane, {}).items(), key=lambda item: (-item[1], item[0]))
        )
        queue.append(
            {
                "rank": 0,
                "focus": lane,
                "count": count,
                "source_visibility_counts": visibility_counts,
                "recommended_next_step": _inspection_next_step_for_lane(lane, visibility_counts),
                "guardrail": (
                    "Use this to inspect generalizable pipeline behavior only; do not add "
                    "paper-specific deterministic expected-fact rules."
                ),
                "example_cases": _top_cases_for_inspection_lane(payload.get("cases"), lane),
            }
        )
    for index, entry in enumerate(queue, start=1):
        entry["rank"] = index
    return queue


def _inspection_next_step_for_lane(lane: str, visibility_counts: dict[str, int]) -> str:
    exact_count = visibility_counts.get("exact_present", 0)
    near_count = visibility_counts.get("near_term_candidate", 0)
    weak_count = visibility_counts.get("weak_term_candidate", 0)
    absent_count = visibility_counts.get("no_term_candidate", 0)
    if lane == "source_recall_or_extraction_visibility":
        if absent_count >= max(near_count + weak_count, exact_count):
            return (
                "Inspect source availability, parser coverage, figure/table OCR, and supplement ingestion "
                "for missing evidence surfaces."
            )
        if near_count or weak_count:
            return (
                "Inspect normalization, aliases, OCR cleanup, and benchmark wording before changing "
                "parser behavior."
            )
        return (
            "Inspect artifact routing and evidence visibility because expected terms appear somewhere "
            "but are still classified as source-recall losses."
        )
    if lane == "evidence_packetization_or_typing":
        return (
            "Inspect packet builder rules, modality labels, section anchors, and scientific-detail typing "
            "using reusable category logic."
        )
    if lane == "synthesis_evidence_selection_or_ranking":
        return (
            "Inspect focus-slot ranking, per-section quotas, cross-modal evidence promotion, and selection "
            "budgets because evidence reaches packets but not synthesis input."
        )
    if lane == "final_synthesis_instruction_or_coverage":
        return (
            "Inspect synthesis prompt coverage requirements, verifier checks, and merge behavior because "
            "the right evidence reaches synthesis input."
        )
    if lane == "benchmark_matching_or_claim_support_context":
        return (
            "Inspect benchmark matching, claim context, and report phrasing alignment before changing "
            "extraction or synthesis."
        )
    return "Inspect representative examples for a shared pipeline pattern before proposing a patch."


def _top_cases_for_inspection_lane(cases: Any, lane: str, *, limit: int = 3) -> list[dict[str, Any]]:
    if not isinstance(cases, list):
        return []
    rows: list[dict[str, Any]] = []
    for case in cases:
        if not isinstance(case, dict):
            continue
        lane_counts = _int_dict(case.get("improvement_lane_counts"))
        count = lane_counts.get(lane, 0)
        if count <= 0:
            continue
        lane_visibility = _nested_int_dict(case.get("lane_source_visibility_counts")).get(lane, {})
        rows.append(
            {
                "case_id": str(case.get("case_id") or ""),
                "count": count,
                "overall_benchmark_score": _float_or_none(case.get("overall_benchmark_score")),
                "source_visibility_counts": dict(
                    sorted(lane_visibility.items(), key=lambda item: (-item[1], item[0]))
                ),
            }
        )
    rows.sort(key=lambda row: (-_int_or_zero(row.get("count")), str(row.get("case_id") or "")))
    return rows[:limit]


def _stage_representative_examples(stage: dict[str, Any], *, max_examples: int = 5) -> list[dict[str, str]]:
    items = stage.get("missing_items")
    if not isinstance(items, list):
        return []
    examples: list[dict[str, str]] = []
    seen_lanes: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        lane = str(item.get("improvement_lane") or "unknown")
        if lane in seen_lanes:
            continue
        row = {
            "claim_id": str(item.get("claim_id") or ""),
            "item_type": str(item.get("item_type") or ""),
            "term": str(item.get("term") or ""),
            "failure_point": str(item.get("failure_point") or ""),
            "improvement_lane": lane,
            "diagnostic_reason": str(item.get("diagnostic_reason") or ""),
            "source_visibility": _compact_source_visibility(item.get("source_visibility")),
        }
        trace = _compact_stage_item_trace(item)
        if trace:
            row["trace"] = trace
        nearest = _compact_stage_item_nearest(item)
        if nearest:
            row["nearest"] = nearest
        examples.append(row)
        seen_lanes.add(lane)
        if len(examples) >= max_examples:
            break
    return examples


def _compact_source_visibility(value: Any) -> str:
    if not isinstance(value, dict):
        return "unknown"
    classification = str(value.get("classification") or "unknown")
    score = value.get("term_score")
    if score is None:
        return classification
    return f"{classification}:{score}"


def _compact_stage_item_trace(item: dict[str, Any]) -> str:
    matches = item.get("stage_matches")
    if not isinstance(matches, dict):
        return ""
    for stage in _trace_stage_order_for_failure(str(item.get("failure_point") or "")):
        stage_matches = matches.get(stage)
        if not isinstance(stage_matches, list) or not stage_matches:
            continue
        first = stage_matches[0]
        if not isinstance(first, dict):
            continue
        path = str(first.get("path") or "").strip()
        snippet = str(first.get("snippet") or "").strip()
        if path or snippet:
            return f"{stage} {path}: {snippet[:180]}"
    return ""


def _trace_stage_order_for_failure(failure_point: str) -> tuple[str, ...]:
    return {
        "dropped_before_evidence_packetization": (
            "extracted_text",
            "evidence_packets",
            "synthesis_inputs",
            "final_report",
        ),
        "dropped_before_synthesis_selection": (
            "evidence_packets",
            "extracted_text",
            "synthesis_inputs",
            "final_report",
        ),
        "dropped_during_final_synthesis": (
            "synthesis_inputs",
            "evidence_packets",
            "extracted_text",
            "final_report",
        ),
        "present_in_final_but_unmatched": (
            "final_report",
            "synthesis_inputs",
            "evidence_packets",
            "extracted_text",
        ),
    }.get(
        failure_point,
        ("extracted_text", "evidence_packets", "synthesis_inputs", "final_report"),
    )


def _compact_stage_item_nearest(item: dict[str, Any]) -> str:
    candidates = item.get("nearest_stage_candidates")
    if not isinstance(candidates, dict):
        return ""
    best: dict[str, Any] | None = None
    best_stage = ""
    best_score = -1.0
    for stage, stage_candidates in candidates.items():
        if not isinstance(stage_candidates, list) or not stage_candidates:
            continue
        first = stage_candidates[0]
        if not isinstance(first, dict):
            continue
        try:
            score = float(first.get("score") or 0)
        except Exception:
            score = 0.0
        if score > best_score:
            best = first
            best_stage = str(stage)
            best_score = score
    if not best:
        return ""
    path = str(best.get("path") or "").strip()
    snippet = str(best.get("snippet") or "").strip()
    return f"{best_stage} {path}: {snippet[:180]}"


def _latest_rows_by_case(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = str(row.get("case_id") or "")
        if case_id:
            latest[case_id] = row
    return latest


def _int_dict(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, int] = {}
    for key, count in value.items():
        result[str(key)] = _int_or_zero(count)
    return result


def _nested_int_dict(value: Any) -> dict[str, dict[str, int]]:
    if not isinstance(value, dict):
        return {}
    return {str(key): _int_dict(inner) for key, inner in value.items() if isinstance(inner, dict)}


def _merge_nested_counts(target: dict[str, dict[str, int]], source: dict[str, dict[str, int]]) -> None:
    for outer, inner_counts in source.items():
        target.setdefault(outer, {})
        for inner, count in inner_counts.items():
            target[outer][inner] = target[outer].get(inner, 0) + count


def _int_or_zero(value: Any) -> int:
    parsed = _int_or_none(value)
    return parsed if parsed is not None else 0


def _format_count_dict(value: dict[str, int]) -> str:
    return ", ".join(f"{key}={count}" for key, count in sorted(value.items(), key=lambda item: (-item[1], item[0])))


def _summarize_history(path: Path, *, json_output: bool = False) -> int:
    rows = _history_rows(path)
    current_definition = _benchmark_definition_fingerprint()
    compact_current_definition = _compact_benchmark_definition(current_definition)
    by_case: dict[str, list[dict[str, Any]]] = {}
    focus_counts: dict[str, int] = {}
    for row in rows:
        case_id = str(row.get("case_id") or "unknown")
        by_case.setdefault(case_id, []).append(row)
        for focus in row.get("next_focus", []):
            key = str(focus)
            focus_counts[key] = focus_counts.get(key, 0) + 1

    if json_output:
        latest_by_case = {
            case_id: by_case[case_id][-1]
            for case_id in sorted(by_case)
            if by_case[case_id]
        }
        latest_rows = list(latest_by_case.values())
        current_definition_rows = [
            row
            for row in latest_rows
            if _benchmark_definition_match_current(row, str(compact_current_definition.get("digest") or "")) is True
        ]
        payload = {
            "history_root": str(path),
            "runs": len({row["summary_path"] for row in rows}),
            "records": len(rows),
            "current_benchmark_definition": current_definition,
            "current_benchmark_definition_summary": compact_current_definition,
            "latest_benchmark_score_summary": _benchmark_score_summary(latest_rows),
            "latest_benchmark_definition_summary": _latest_definition_summary(
                latest_rows,
                str(compact_current_definition.get("digest") or ""),
            ),
            "current_definition_benchmark_score_summary": _benchmark_score_summary(current_definition_rows),
            "current_definition_refresh_cases": _definition_refresh_rows(
                latest_rows,
                str(compact_current_definition.get("digest") or ""),
                limit=10,
            ),
            "benchmark_score_trends": _score_trend_summary(by_case, limit=10),
            "latest_output_change_summary": _history_output_change_summary(latest_rows),
            "score_priority_cases": _score_priority_rows(latest_rows, limit=10),
            "latest_by_case": latest_by_case,
            "focus_counts": dict(sorted(focus_counts.items(), key=lambda item: (-item[1], item[0]))),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    print(f"history_root: {path}")
    print(f"runs: {len({row['summary_path'] for row in rows})}")
    print(f"records: {len(rows)}")
    print(
        "current_benchmark_definition: "
        f"digest={compact_current_definition.get('digest')} "
        f"files={compact_current_definition.get('file_count')}"
    )
    if not rows:
        return 0
    latest_by_case = {
        case_id: by_case[case_id][-1]
        for case_id in sorted(by_case)
        if by_case[case_id]
    }
    score_summary = _benchmark_score_summary(list(latest_by_case.values()))
    definition_summary = _latest_definition_summary(
        list(latest_by_case.values()),
        str(compact_current_definition.get("digest") or ""),
    )
    current_definition_score_summary = _benchmark_score_summary(
        [
            row
            for row in latest_by_case.values()
            if _benchmark_definition_match_current(row, str(compact_current_definition.get("digest") or "")) is True
        ]
    )
    definition_refresh_cases = _definition_refresh_rows(
        list(latest_by_case.values()),
        str(compact_current_definition.get("digest") or ""),
        limit=5,
    )
    trend_summary = _score_trend_summary(by_case, limit=5)
    output_change_summary = _history_output_change_summary(list(latest_by_case.values()))
    print(
        "latest_benchmark_score_summary: "
        f"weighted={score_summary.get('weighted_overall_benchmark_score')} "
        f"mean={score_summary.get('mean_overall_benchmark_score')} "
        f"scored={score_summary.get('scored_cases')}/{score_summary.get('total_cases')} "
        f"slots={score_summary.get('matched_slots')}/{score_summary.get('expected_slots')}"
    )
    print(
        "latest_benchmark_definition_summary: "
        f"matching_current={definition_summary.get('matching_current')} "
        f"mismatched_current={definition_summary.get('mismatched_current')} "
        f"missing_definition={definition_summary.get('missing_definition')} "
        f"scored_matching_current={definition_summary.get('scored_matching_current')}"
    )
    print(
        "current_definition_benchmark_score_summary: "
        f"weighted={current_definition_score_summary.get('weighted_overall_benchmark_score')} "
        f"mean={current_definition_score_summary.get('mean_overall_benchmark_score')} "
        f"scored={current_definition_score_summary.get('scored_cases')}/{current_definition_score_summary.get('total_cases')} "
        f"slots={current_definition_score_summary.get('matched_slots')}/{current_definition_score_summary.get('expected_slots')}"
    )
    if definition_refresh_cases:
        print("\nCurrent definition refresh cases:")
        for row in definition_refresh_cases:
            print(
                "- "
                f"{row['case_id']}: reason={row['reason']} "
                f"score={row['overall_benchmark_score']} "
                f"missing_slots={row['benchmark_missing_slots']} "
                f"gaps={row['claim_requirement_gap_count']}"
            )
    print(
        "benchmark_score_trends: "
        f"scored_trends={trend_summary.get('scored_trend_cases')} "
        f"comparable={trend_summary.get('comparable_cases')} "
        f"improved={trend_summary.get('improved')} "
        f"regressed={trend_summary.get('regressed')} "
        f"unchanged={trend_summary.get('unchanged')} "
        f"comparable_improved={trend_summary.get('comparable_improved')} "
        f"comparable_regressed={trend_summary.get('comparable_regressed')} "
        f"comparable_unchanged={trend_summary.get('comparable_unchanged')} "
        f"definition_matched={trend_summary.get('definition_matched')} "
        f"definition_mismatched={trend_summary.get('definition_mismatched')} "
        f"definition_missing={trend_summary.get('definition_missing')} "
        f"comparable_mean_delta={trend_summary.get('mean_score_delta')} "
        f"all_mean_delta={trend_summary.get('all_score_delta_mean')}"
    )
    print(
        "latest_output_change_summary: "
        f"diagnostic_only={output_change_summary.get('diagnostic_only')} "
        f"output_risk={output_change_summary.get('output_risk')} "
        f"unknown={output_change_summary.get('unknown')} "
        f"missing_audit={output_change_summary.get('missing_audit')}"
    )
    score_priority_cases = _score_priority_rows(list(latest_by_case.values()), limit=5)
    if score_priority_cases:
        print("\nLowest score cases:")
        for row in score_priority_cases:
            print(
                "- "
                f"{row['case_id']}: score={row['overall_benchmark_score']} "
                f"missing_slots={row['benchmark_missing_slots']} "
                f"gaps={row['claim_requirement_gap_count']} "
                f"decision={row['decision']}"
            )

    trend_cases = trend_summary.get("cases") if isinstance(trend_summary.get("cases"), list) else []
    if trend_cases:
        print("\nScore trends:")
        for row in trend_cases:
            print(
                "- "
                f"{row['case_id']}: {row['direction']} "
                f"{row['previous_score']} -> {row['latest_score']} "
                f"delta={row['score_delta']} "
                f"missing_slots_delta={row['missing_slots_delta']} "
                f"definition_match={row['benchmark_definition_match']}"
            )

    print("\nLatest by case:")
    for case_id in sorted(by_case):
        latest = by_case[case_id][-1]
        print(
            "- "
            f"{case_id}: decision={latest.get('decision', '')} "
            f"compatible={latest.get('compatible', None)} "
            f"usable={latest.get('usable_packet_rate', None)} "
            f"claims={latest.get('critical_claim_candidate_rate', None)} "
            f"slowest={latest.get('slowest_stage', '')}"
            f"({latest.get('slowest_stage_seconds', None)}s) "
            f"score={latest.get('overall_benchmark_score', None)} "
            f"gaps={latest.get('claim_requirement_gap_count', None)} "
            f"missing_slots={latest.get('benchmark_missing_slots', None)} "
            f"diagnostic_only={latest.get('output_change_diagnostic_only', None)} "
            f"run={latest.get('run_dir', '')}"
        )
        next_focus = latest.get("next_focus") if isinstance(latest.get("next_focus"), list) else []
        if next_focus:
            print(f"  next_focus: {', '.join(str(item) for item in next_focus[:8])}")
        failures = latest.get("failures") if isinstance(latest.get("failures"), list) else []
        if failures:
            print(f"  failures: {'; '.join(str(item) for item in failures[:3])}")
        gap_summary = str(latest.get("benchmark_gap_summary") or "")
        if gap_summary:
            print(f"  benchmark_gaps: {gap_summary}")
        artifact_url = str(
            latest.get("detailed_analysis_url")
            or latest.get("detailed_analysis_html")
            or latest.get("webapp_detailed_analysis_url")
            or ""
        )
        if artifact_url:
            print(f"  detailed_analysis: {artifact_url}")
        slack_summary = str(latest.get("slack_summary_markdown") or "")
        if slack_summary:
            print(f"  slack_summary: {slack_summary}")
        media_json = str(latest.get("media_json") or "")
        if media_json:
            print(f"  media_json: {media_json}")

    if focus_counts:
        print("\nFocus counts:")
        for focus, count in sorted(focus_counts.items(), key=lambda item: (-item[1], item[0])):
            print(f"- {focus}: {count}")
    return 0


def _record_history(path: Path, db_path: Path) -> int:
    rows = _history_rows(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            create table if not exists active_benchmark_history (
                summary_path text not null,
                generated_at text not null,
                surface text not null,
                benchmark_definition_available integer,
                benchmark_definition_algorithm text not null default '',
                benchmark_definition_digest text not null default '',
                benchmark_definition_file_count integer,
                output_change_audit_available integer,
                output_change_diagnostic_only integer,
                output_change_changed_count integer,
                output_change_output_risk_count integer,
                output_change_unknown_count integer,
                case_id text not null,
                decision text not null,
                compatible integer,
                overall_benchmark_score real,
                benchmark_content_score real,
                benchmark_matched_slots real,
                benchmark_expected_slots real,
                benchmark_missing_slots real,
                claim_requirement_gap_count integer,
                benchmark_gap_summary text not null default '',
                usable_packet_rate real,
                critical_claim_candidate_rate real,
                slowest_stage text not null,
                slowest_stage_seconds real,
                next_focus_json text not null,
                failures_json text not null,
                run_dir text not null,
                detailed_analysis_html text not null default '',
                detailed_analysis_url text not null default '',
                webapp_detailed_analysis_url text not null default '',
                slack_summary_markdown text not null default '',
                media_json text not null default '',
                static_media_dir text not null default '',
                primary key (summary_path, case_id, generated_at)
            )
            """
        )
        columns = {info[1] for info in conn.execute("pragma table_info(active_benchmark_history)")}
        migration_columns = {
            "detailed_analysis_html": "text not null default ''",
            "detailed_analysis_url": "text not null default ''",
            "webapp_detailed_analysis_url": "text not null default ''",
            "slack_summary_markdown": "text not null default ''",
            "media_json": "text not null default ''",
            "static_media_dir": "text not null default ''",
            "benchmark_definition_available": "integer",
            "benchmark_definition_algorithm": "text not null default ''",
            "benchmark_definition_digest": "text not null default ''",
            "benchmark_definition_file_count": "integer",
            "output_change_audit_available": "integer",
            "output_change_diagnostic_only": "integer",
            "output_change_changed_count": "integer",
            "output_change_output_risk_count": "integer",
            "output_change_unknown_count": "integer",
            "overall_benchmark_score": "real",
            "benchmark_content_score": "real",
            "benchmark_matched_slots": "real",
            "benchmark_expected_slots": "real",
            "benchmark_missing_slots": "real",
            "claim_requirement_gap_count": "integer",
            "benchmark_gap_summary": "text not null default ''",
        }
        for column, column_type in migration_columns.items():
            if column not in columns:
                conn.execute(f"alter table active_benchmark_history add column {column} {column_type}")
        for row in rows:
            conn.execute(
                """
                delete from active_benchmark_history
                where generated_at = ? and case_id = ? and summary_path != ?
                """,
                (
                    str(row.get("generated_at") or ""),
                    str(row.get("case_id") or ""),
                    str(row.get("summary_path") or ""),
                ),
            )
            conn.execute(
                """
                insert or replace into active_benchmark_history (
                    summary_path,
                    generated_at,
                    surface,
                    benchmark_definition_available,
                    benchmark_definition_algorithm,
                    benchmark_definition_digest,
                    benchmark_definition_file_count,
                    output_change_audit_available,
                    output_change_diagnostic_only,
                    output_change_changed_count,
                    output_change_output_risk_count,
                    output_change_unknown_count,
                    case_id,
                    decision,
                    compatible,
                    overall_benchmark_score,
                    benchmark_content_score,
                    benchmark_matched_slots,
                    benchmark_expected_slots,
                    benchmark_missing_slots,
                    claim_requirement_gap_count,
                    benchmark_gap_summary,
                    usable_packet_rate,
                    critical_claim_candidate_rate,
                    slowest_stage,
                    slowest_stage_seconds,
                    next_focus_json,
                    failures_json,
                    run_dir,
                    detailed_analysis_html,
                    detailed_analysis_url,
                    webapp_detailed_analysis_url,
                    slack_summary_markdown,
                    media_json,
                    static_media_dir
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(row.get("summary_path") or ""),
                    str(row.get("generated_at") or ""),
                    str(row.get("surface") or ""),
                    _bool_to_int_or_none(row.get("benchmark_definition_available")),
                    str(row.get("benchmark_definition_algorithm") or ""),
                    str(row.get("benchmark_definition_digest") or ""),
                    _int_or_none(row.get("benchmark_definition_file_count")),
                    _bool_to_int_or_none(row.get("output_change_audit_available")),
                    _bool_to_int_or_none(row.get("output_change_diagnostic_only")),
                    _int_or_none(row.get("output_change_changed_count")),
                    _int_or_none(row.get("output_change_output_risk_count")),
                    _int_or_none(row.get("output_change_unknown_count")),
                    str(row.get("case_id") or ""),
                    str(row.get("decision") or ""),
                    _bool_to_int_or_none(row.get("compatible")),
                    _float_or_none(row.get("overall_benchmark_score")),
                    _float_or_none(row.get("benchmark_content_score")),
                    _float_or_none(row.get("benchmark_matched_slots")),
                    _float_or_none(row.get("benchmark_expected_slots")),
                    _float_or_none(row.get("benchmark_missing_slots")),
                    _int_or_none(row.get("claim_requirement_gap_count")),
                    str(row.get("benchmark_gap_summary") or ""),
                    _float_or_none(row.get("usable_packet_rate")),
                    _float_or_none(row.get("critical_claim_candidate_rate")),
                    str(row.get("slowest_stage") or ""),
                    _float_or_none(row.get("slowest_stage_seconds")),
                    json.dumps(row.get("next_focus", []), sort_keys=True),
                    json.dumps(row.get("failures", []), sort_keys=True),
                    str(row.get("run_dir") or ""),
                    str(row.get("detailed_analysis_html") or ""),
                    str(row.get("detailed_analysis_url") or ""),
                    str(row.get("webapp_detailed_analysis_url") or ""),
                    str(row.get("slack_summary_markdown") or ""),
                    str(row.get("media_json") or ""),
                    str(row.get("static_media_dir") or ""),
                ),
            )
        conn.commit()
        count = conn.execute("select count(*) from active_benchmark_history").fetchone()[0]
    print(f"history_db: {db_path}")
    print(f"recorded_rows: {len(rows)}")
    print(f"stored_rows: {count}")
    return 0


def _bool_to_int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return 1 if value else 0
    return None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _plan_next_rows(
    *,
    manifest: dict[str, Any],
    history_path: Path,
    tier: str,
    case_ids: list[str],
    current_benchmark_definition_digest: str = "",
    prefer_needs_fix: bool = False,
    prefer_lowest_score: bool = False,
    prefer_current_definition_refresh: bool = False,
) -> dict[str, Any]:
    selected = _selected_cases(
        manifest,
        mode="suite",
        tier=tier,
        case_ids=case_ids,
        include_unscored=True,
        random_seed=None,
    )
    history = _history_rows(history_path) if history_path.exists() else []
    latest_by_case: dict[str, dict[str, Any]] = {}
    for row in history:
        case_id = str(row.get("case_id") or "")
        if case_id:
            latest_by_case[case_id] = row

    planned: list[dict[str, Any]] = []
    for index, case in enumerate(selected, start=1):
        case_id = str(case.get("id") or "")
        latest = latest_by_case.get(case_id)
        latest_score = _float_or_none(latest.get("overall_benchmark_score")) if latest else None
        latest_definition_digest = str(latest.get("benchmark_definition_digest") or "") if latest else ""
        latest_definition_match_current = (
            _benchmark_definition_match_current(latest, current_benchmark_definition_digest)
            if latest
            else None
        )
        next_focus = latest.get("next_focus", []) if latest else ["run_case"]
        if latest is not None and latest_definition_match_current is not True:
            next_focus = _unique(["refresh_current_benchmark_definition", *next_focus])
        if latest is None:
            state = "unrun"
            priority = 1 if prefer_needs_fix else 0
        elif str(latest.get("decision") or "") != "pass":
            state = "needs_fix"
            priority = 0 if prefer_needs_fix else 1
        else:
            state = "passed"
            priority = 2
        planned.append(
            {
                "case_id": case_id,
                "manifest_order": index,
                "scoring": str(case.get("scoring") or ""),
                "tiers": [str(item) for item in case.get("tiers", [])],
                "reference_status": str(case.get("reference_status") or ""),
                "gold_standard": str(case.get("gold_standard") or ""),
                "pdf": str(case.get("pdf") or ""),
                "state": state,
                "priority": priority,
                "latest_decision": str(latest.get("decision") or "") if latest else "",
                "latest_compatible": latest.get("compatible") if latest else None,
                "latest_overall_benchmark_score": latest_score,
                "latest_benchmark_definition_digest": latest_definition_digest,
                "latest_benchmark_definition_match_current": latest_definition_match_current,
                "latest_benchmark_missing_slots": latest.get("benchmark_missing_slots") if latest else None,
                "latest_claim_requirement_gap_count": latest.get("claim_requirement_gap_count") if latest else None,
                "latest_benchmark_gap_summary": str(latest.get("benchmark_gap_summary") or "") if latest else "",
                "latest_usable_packet_rate": latest.get("usable_packet_rate") if latest else None,
                "latest_claim_rate": latest.get("critical_claim_candidate_rate") if latest else None,
                "latest_slowest_stage": str(latest.get("slowest_stage") or "") if latest else "",
                "latest_slowest_stage_seconds": latest.get("slowest_stage_seconds") if latest else None,
                "next_focus": next_focus,
            }
        )
    if prefer_current_definition_refresh:
        def refresh_priority(row: dict[str, Any]) -> tuple[int, float, float, int, int]:
            score = _float_or_none(row.get("latest_overall_benchmark_score"))
            missing_slots = _float_or_none(row.get("latest_benchmark_missing_slots")) or 0.0
            gap_count = _int_or_none(row.get("latest_claim_requirement_gap_count")) or 0
            definition_match = row.get("latest_benchmark_definition_match_current")
            if definition_match is False and score is not None:
                state_priority = 0
            elif definition_match is None and score is not None:
                state_priority = 1
            elif row.get("state") == "unrun":
                state_priority = 2
            elif row.get("state") == "needs_fix":
                state_priority = 3
            else:
                state_priority = 4
            return (
                state_priority,
                score if score is not None else 999.0,
                -missing_slots,
                -gap_count,
                int(row["manifest_order"]),
            )

        planned.sort(key=refresh_priority)
    elif prefer_lowest_score:
        def score_priority(row: dict[str, Any]) -> tuple[int, float, float, int, int]:
            score = _float_or_none(row.get("latest_overall_benchmark_score"))
            missing_slots = _float_or_none(row.get("latest_benchmark_missing_slots")) or 0.0
            gap_count = _int_or_none(row.get("latest_claim_requirement_gap_count")) or 0
            if row.get("state") == "needs_fix" and score is not None:
                state_priority = 0
            elif row.get("state") == "needs_fix":
                state_priority = 1
            elif row.get("state") == "unrun":
                state_priority = 2
            else:
                state_priority = 3
            return (
                state_priority,
                score if score is not None else 999.0,
                -missing_slots,
                -gap_count,
                int(row["manifest_order"]),
            )

        planned.sort(key=score_priority)
    else:
        planned.sort(key=lambda row: (int(row["priority"]), int(row["manifest_order"])))
    state_counts: dict[str, int] = {}
    for row in planned:
        state = str(row.get("state") or "unknown")
        state_counts[state] = state_counts.get(state, 0) + 1
    return {
        "selected_count": len(selected),
        "history_record_count": len(history),
        "prefer_needs_fix": prefer_needs_fix,
        "prefer_lowest_score": prefer_lowest_score,
        "prefer_current_definition_refresh": prefer_current_definition_refresh,
        "planned": planned,
        "state_counts": state_counts,
    }


def _plan_next_case(
    path: Path,
    *,
    tier: str,
    case_ids: list[str],
    prefer_needs_fix: bool = False,
    prefer_lowest_score: bool = False,
    prefer_current_definition_refresh: bool = False,
    json_output: bool = False,
) -> int:
    manifest = _load_json(MANIFEST)
    benchmark_definition = _benchmark_definition_fingerprint()
    compact_benchmark_definition = _compact_benchmark_definition(benchmark_definition)
    output_change_audit = _compact_output_change_audit(_current_output_change_audit())
    plan = _plan_next_rows(
        manifest=manifest,
        history_path=path,
        tier=tier,
        case_ids=case_ids,
        current_benchmark_definition_digest=str(compact_benchmark_definition.get("digest") or ""),
        prefer_needs_fix=prefer_needs_fix,
        prefer_lowest_score=prefer_lowest_score,
        prefer_current_definition_refresh=prefer_current_definition_refresh,
    )
    planned = plan["planned"]
    if json_output:
        payload: dict[str, Any] = {
            "history_root": str(path),
            "selected_cases": plan["selected_count"],
            "history_records": plan["history_record_count"],
            "prefer_needs_fix": plan["prefer_needs_fix"],
            "prefer_lowest_score": plan["prefer_lowest_score"],
            "prefer_current_definition_refresh": plan["prefer_current_definition_refresh"],
            "current_benchmark_definition": benchmark_definition,
            "current_benchmark_definition_summary": compact_benchmark_definition,
            "output_change_audit": output_change_audit,
            "next_case": None,
            "state_counts": plan["state_counts"],
            "planned_cases": planned,
            "score_priority_cases": _score_priority_rows(planned, limit=10),
            "current_definition_refresh_cases": _definition_refresh_rows(
                planned,
                str(compact_benchmark_definition.get("digest") or ""),
                limit=10,
            ),
            "queue_preview": planned[:10],
        }
        if planned:
            next_case = dict(planned[0])
            next_case["suggested_command"] = _suggested_active_command(next_case)
            next_case["suggested_diagnostic_only_command"] = _suggested_active_command(
                next_case,
                require_diagnostic_only=True,
            )
            if _needs_text_cache_disabled_run(next_case):
                next_case["suggested_fresh_text_command"] = _suggested_active_command(
                    next_case,
                    disable_local_text_cache=True,
                )
                next_case["suggested_fresh_text_diagnostic_only_command"] = _suggested_active_command(
                    next_case,
                    require_diagnostic_only=True,
                    disable_local_text_cache=True,
                )
            payload["next_case"] = next_case
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    print(f"history_root: {path}")
    print(f"selected_cases: {plan['selected_count']}")
    print(f"history_records: {plan['history_record_count']}")
    print(f"prefer_needs_fix: {plan['prefer_needs_fix']}")
    print(f"prefer_lowest_score: {plan['prefer_lowest_score']}")
    print(f"prefer_current_definition_refresh: {plan['prefer_current_definition_refresh']}")
    print(
        "current_benchmark_definition: "
        f"digest={compact_benchmark_definition.get('digest')} "
        f"files={compact_benchmark_definition.get('file_count')}"
    )
    print(
        "output_change_audit: "
        f"diagnostic_only={output_change_audit['diagnostic_only']} "
        f"changed={output_change_audit['changed_count']} "
        f"output_risk={output_change_audit['output_risk_count']} "
        f"unknown={output_change_audit['unknown_count']}"
    )
    if not planned:
        print("next_case: none")
        return 0

    next_case = planned[0]
    print(
        "next_case: "
        f"{next_case['case_id']} "
        f"state={next_case['state']} "
        f"scoring={next_case['scoring']} "
        f"tiers={','.join(next_case['tiers'])}"
    )
    if next_case["latest_decision"]:
        print(
            "latest: "
            f"decision={next_case['latest_decision']} "
            f"compatible={next_case['latest_compatible']} "
            f"score={next_case['latest_overall_benchmark_score']} "
            f"definition_current={next_case['latest_benchmark_definition_match_current']} "
            f"gaps={next_case['latest_claim_requirement_gap_count']} "
            f"missing_slots={next_case['latest_benchmark_missing_slots']} "
            f"usable={next_case['latest_usable_packet_rate']} "
            f"claims={next_case['latest_claim_rate']} "
            f"slowest={next_case['latest_slowest_stage']}({next_case['latest_slowest_stage_seconds']}s)"
        )
    next_focus = next_case.get("next_focus") if isinstance(next_case.get("next_focus"), list) else []
    if next_focus:
        print(f"next_focus: {', '.join(str(item) for item in next_focus[:8])}")
    if next_case.get("latest_benchmark_gap_summary"):
        print(f"latest_benchmark_gaps: {next_case['latest_benchmark_gap_summary']}")
    print("suggested_command: " + " ".join(_suggested_active_command(next_case)))
    print(
        "suggested_diagnostic_only_command: "
        + " ".join(_suggested_active_command(next_case, require_diagnostic_only=True))
    )
    if _needs_text_cache_disabled_run(next_case):
        print(
            "suggested_fresh_text_command: "
            + " ".join(_suggested_active_command(next_case, disable_local_text_cache=True))
        )
        print(
            "suggested_fresh_text_diagnostic_only_command: "
            + " ".join(
                _suggested_active_command(
                    next_case,
                    require_diagnostic_only=True,
                    disable_local_text_cache=True,
                )
            )
        )

    refresh_rows = _definition_refresh_rows(
        planned,
        str(compact_benchmark_definition.get("digest") or ""),
        limit=5,
    )
    if refresh_rows:
        print("\nCurrent definition refresh preview:")
        for row in refresh_rows:
            print(
                "- "
                f"{row['case_id']}: reason={row['reason']} "
                f"score={row['overall_benchmark_score']} "
                f"missing_slots={row['benchmark_missing_slots'] if row['benchmark_missing_slots'] is not None else 'n/a'} "
                f"gaps={row['claim_requirement_gap_count'] if row['claim_requirement_gap_count'] is not None else 'n/a'} "
                f"decision={row['decision'] or 'n/a'}"
            )

    print("\nQueue preview:")
    for row in planned[:10]:
        print(
            "- "
            f"{row['case_id']}: state={row['state']} "
            f"decision={row['latest_decision'] or 'n/a'} "
            f"score={row['latest_overall_benchmark_score'] if row['latest_overall_benchmark_score'] is not None else 'n/a'} "
            f"definition_current={row['latest_benchmark_definition_match_current']} "
            f"gaps={row['latest_claim_requirement_gap_count'] if row['latest_claim_requirement_gap_count'] is not None else 'n/a'} "
            f"missing_slots={row['latest_benchmark_missing_slots'] if row['latest_benchmark_missing_slots'] is not None else 'n/a'} "
            f"slowest={row['latest_slowest_stage'] or 'n/a'}"
        )
        if row.get("latest_benchmark_gap_summary"):
            print(f"  benchmark_gaps: {row['latest_benchmark_gap_summary']}")
    return 0


def _suggested_active_command(
    next_case: dict[str, Any],
    *,
    require_diagnostic_only: bool = False,
    disable_local_text_cache: bool = True,
) -> list[str]:
    command = [
        ".venv/bin/python",
        ".codex/skills/papereval-benchmark/scripts/papereval_benchmark_qa.py",
        "--active-run",
        "--surface",
        "web",
        "--start-app",
        "--stop-app",
        "--llm-provider",
        "local",
        "--case",
        str(next_case.get("case_id") or ""),
        "--max-concurrent",
        "1",
        "--fail-on-incompatible",
    ]
    if str(next_case.get("scoring") or "") == "diagnostic_coverage":
        command.append("--include-unscored")
    if require_diagnostic_only:
        command.append("--require-diagnostic-only")
    if disable_local_text_cache:
        command.append("--disable-local-text-cache")
    return command


def _run_active_benchmark(args: argparse.Namespace) -> int:
    manifest = _load_json(MANIFEST)
    tier = args.tier or ("all" if args.mode == "single" else "smoke")
    include_unscored = bool(args.include_unscored or args.mode == "single")
    selected = _selected_cases(
        manifest,
        mode=args.mode,
        tier=tier,
        case_ids=args.case_ids or [],
        include_unscored=include_unscored,
        random_seed=args.random_seed,
    )
    if not selected:
        raise SystemExit("No active benchmark cases selected.")
    if bool(getattr(args, "require_diagnostic_only", False)):
        output_change_audit = _require_diagnostic_only_worktree()
    else:
        output_change_audit = _current_output_change_audit()
    benchmark_definition = _benchmark_definition_fingerprint()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_dir).expanduser() if args.out_dir else ROOT / "test" / "active_benchmark" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    api_base = _normalize_api_base(args.api_base, args.backend_port)
    app_proc: subprocess.Popen[Any] | None = None
    records: list[dict[str, Any]] = []
    started_here = False

    try:
        app_proc = _start_app(args, api_base, run_dir)
        started_here = app_proc is not None
        api_status = _wait_api_ready(api_base, timeout_seconds=args.startup_timeout)
        try:
            _run_preflight(api_base, api_status, args)
        except Exception as exc:
            summary_path = run_dir / "active_benchmark_summary.json"
            summary = {
                "schema_version": 1,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "surface": args.surface,
                "api_base": api_base,
                "started_app": started_here,
                "mode": args.mode,
                "tier": tier,
                "include_unscored": include_unscored,
                "helper_max_concurrent": int(args.max_concurrent),
                "disable_local_text_cache": _effective_disable_local_text_cache(args),
                "random_seed": args.random_seed,
                "benchmark_definition": benchmark_definition,
                "output_change_audit": output_change_audit,
                "queue_timeout_seconds": float(args.queue_timeout),
                "timeout_per_case_seconds": float(args.timeout_per_case),
                "preflight_ok": False,
                "preflight_failure": str(exc),
                "api_status": api_status,
                "totals": {"passed": 0, "failed": 0, "diagnostic_failed": 0, "skipped": len(selected), "executed": 0},
                "records": [],
            }
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"preflight_failed={exc}", file=sys.stderr)
            print(f"summary_json={summary_path}")
            return 1
        print(f"api_ready={api_base}")

        pending = list(enumerate(selected, start=1))
        in_flight: dict[int, ActiveCaseState] = {}
        stop_submitting = False
        backend_capacity = int(api_status.get("processing", {}).get("worker_capacity") or 1) if isinstance(api_status.get("processing"), dict) else 1
        print(
            f"selected_cases={','.join(str(case.get('id')) for case in selected)} "
            f"mode={args.mode} tier={tier} helper_max_concurrent={args.max_concurrent} "
            f"backend_worker_capacity={backend_capacity}"
        )

        while pending or in_flight:
            while pending and not stop_submitting and len(in_flight) < int(args.max_concurrent):
                queue_order, case = pending.pop(0)
                case_id = str(case.get("id"))
                case_dir = run_dir / case_id
                record = _new_record(case, args, api_base, queue_order)
                record["backend_worker_capacity_at_start"] = backend_capacity
                record["submitted_at"] = datetime.now(timezone.utc).isoformat()
                print(f"active submitting {case_id}...")
                try:
                    pdf_path = ROOT / str(case.get("pdf") or "")
                    uploaded = _upload_case(api_base, pdf_path)
                    document_id = int(uploaded["document_id"])
                    job_id = int(uploaded["job_id"])
                    record.update(
                        {
                            "decision": "running",
                            "document_id": document_id,
                            "job_id": job_id,
                            "upload": uploaded,
                        }
                    )
                    _write_record(case_dir, record)
                    in_flight[job_id] = ActiveCaseState(
                        case=case,
                        record=record,
                        case_dir=case_dir,
                        submitted_at=time.time(),
                    )
                except Exception as exc:
                    records.append(_mark_failed(record, case_dir, str(exc)))

            for job_id, state in list(in_flight.items()):
                case = state.case
                record = state.record
                case_dir = state.case_dir
                try:
                    job = _request_json(api_base, f"jobs/{job_id}", timeout=10.0)
                    record["last_seen_job"] = job
                    record["last_polled_at"] = datetime.now(timezone.utc).isoformat()
                    job_status = str(job.get("status") or "").lower()
                    if job_status in {"completed", "failed"}:
                        records.append(
                            _finish_uploaded_case(
                                api_base=api_base,
                                case=case,
                                record=record,
                                case_dir=case_dir,
                                job=job,
                                artifact_url_base=str(args.artifact_url_base or ""),
                            )
                        )
                        in_flight.pop(job_id, None)
                    else:
                        timeout_message = _update_active_timing(
                            state,
                            job,
                            now=time.time(),
                            queue_timeout=float(args.queue_timeout),
                            timeout_per_case=float(args.timeout_per_case),
                        )
                        runtime_status = _request_json(api_base, "status", timeout=10.0)
                        runtime_failures = _runtime_readiness_failures(
                            runtime_status,
                            allow_runtime_not_ready=bool(args.allow_runtime_not_ready),
                        )
                        if runtime_failures:
                            stop_submitting = True
                            records.append(
                                _mark_failed(
                                    record,
                                    case_dir,
                                    "backend runtime became unhealthy: " + "; ".join(runtime_failures),
                                )
                            )
                            in_flight.pop(job_id, None)
                            continue
                        if timeout_message is None:
                            _write_record(case_dir, record)
                            continue
                        stop_submitting = True
                        records.append(_mark_failed(record, case_dir, timeout_message))
                        in_flight.pop(job_id, None)
                except Exception as exc:
                    records.append(_mark_failed(record, case_dir, str(exc)))
                    in_flight.pop(job_id, None)

            if stop_submitting and not in_flight:
                break
            if pending or in_flight:
                time.sleep(float(args.poll_interval))

        for queue_order, case in pending:
            case_dir = run_dir / str(case.get("id"))
            record = _new_record(case, args, api_base, queue_order)
            records.append(_mark_failed(record, case_dir, "not submitted after earlier timeout", decision="skipped"))

        totals = {
            "passed": sum(1 for record in records if record.get("decision") == "pass"),
            "failed": sum(1 for record in records if record.get("decision") == "fail"),
            "diagnostic_failed": sum(1 for record in records if record.get("decision") == "diagnostic_fail"),
            "skipped": sum(1 for record in records if record.get("decision") == "skipped"),
            "executed": sum(1 for record in records if record.get("decision") != "skipped"),
        }
        iteration_diagnostics = [_iteration_diagnostics(record) for record in records]
        summary = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "surface": args.surface,
            "api_base": api_base,
            "started_app": started_here,
            "mode": args.mode,
            "tier": tier,
            "include_unscored": include_unscored,
            "helper_max_concurrent": int(args.max_concurrent),
            "backend_worker_capacity_at_start": backend_capacity,
            "disable_local_text_cache": _effective_disable_local_text_cache(args),
            "random_seed": args.random_seed,
            "benchmark_definition": benchmark_definition,
            "output_change_audit": output_change_audit,
            "queue_timeout_seconds": float(args.queue_timeout),
            "timeout_per_case_seconds": float(args.timeout_per_case),
            "api_status": api_status,
            "preflight_ok": True,
            "totals": totals,
            "records": records,
            "iteration_diagnostics": iteration_diagnostics,
            "benchmark_score_summary": _benchmark_score_summary(iteration_diagnostics),
        }
        summary_path = run_dir / "active_benchmark_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"summary_json={summary_path}")
        return _process_exit_status(
            records,
            fail_on_incompatible=bool(args.fail_on_incompatible),
            fail_on_diagnostic=bool(args.fail_on_diagnostic),
        )
    finally:
        if app_proc is not None and args.stop_app:
            _terminate_process_tree(app_proc)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate PaperEval benchmark configuration or run active app benchmarks.")
    parser.add_argument("--validate", action="store_true", help="Validate manifest file references.")
    parser.add_argument("--run-checks", action="store_true", help="Run local benchmark-focused checks.")
    parser.add_argument(
        "--output-change-audit",
        action="store_true",
        help="Classify current dirty files as benchmark-only, output-risk, or unknown without starting a benchmark.",
    )
    parser.add_argument("--active-run", action="store_true", help="Run selected PDFs through the active app API and compare reports to gold standards.")
    parser.add_argument("--summarize-run", default="", help="Summarize an existing active_benchmark_summary.json or run directory without starting a benchmark.")
    parser.add_argument(
        "--write-detailed-report",
        default="",
        help="Write report.html next to an existing report.json, or inside a directory containing report.json, without starting a benchmark.",
    )
    parser.add_argument("--detailed-analysis-url", default="", help="Optional static detailed-analysis URL to include in --write-detailed-report Slack markdown.")
    parser.add_argument("--webapp-url", default="", help="Optional PaperEval webapp detailed-analysis URL to include in generated report artifacts.")
    parser.add_argument("--media-json", default="", help="Optional document media JSON to embed in --write-detailed-report static HTML.")
    parser.add_argument(
        "--fetch-media-assets",
        action="store_true",
        help="For --write-detailed-report, fetch document media metadata/assets from --api-base or --webapp-url when the backend is available.",
    )
    parser.add_argument(
        "--summarize-history",
        nargs="?",
        const=str(ROOT / "test" / "active_benchmark"),
        default="",
        help="Summarize existing active benchmark runs under a directory without starting a benchmark.",
    )
    parser.add_argument(
        "--summarize-stage-diagnostics",
        nargs="?",
        const=str(ROOT / "test" / "active_benchmark"),
        default="",
        help="Aggregate saved report-vs-gold stage-loss diagnostics across latest benchmark cases without starting a benchmark.",
    )
    parser.add_argument(
        "--record-history",
        action="store_true",
        help="Record redacted history rows from --summarize-history into a local SQLite tracker.",
    )
    parser.add_argument(
        "--plan-next",
        nargs="?",
        const=str(ROOT / "test" / "active_benchmark"),
        default="",
        help="Plan the next benchmark case from manifest plus existing history without starting a benchmark.",
    )
    parser.add_argument(
        "--prefer-needs-fix",
        action="store_true",
        help="For --plan-next, prioritize previously failed/diagnostic-failed cases before unrun cases.",
    )
    parser.add_argument(
        "--prefer-lowest-score",
        action="store_true",
        help="For --plan-next, prioritize scored failed/diagnostic-failed cases with the lowest overall benchmark score.",
    )
    parser.add_argument(
        "--prefer-current-definition-refresh",
        action="store_true",
        help="For --plan-next, prioritize scored cases missing or mismatched against the current benchmark-definition fingerprint.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON for --plan-next, --summarize-history, or --summarize-run.")
    parser.add_argument(
        "--history-db",
        default=str(DEFAULT_HISTORY_DB),
        help="SQLite path for --record-history. Stores compact metrics only, not reports or evidence packets.",
    )
    parser.add_argument("--mode", choices=["suite", "single"], default="suite", help="Run the selected suite or one random selected paper.")
    parser.add_argument("--single", action="store_true", help="Shortcut for --mode single.")
    parser.add_argument("--surface", choices=["api", "web", "desktop"], default="api", help="Active app surface to exercise.")
    parser.add_argument("--start-app", action="store_true", help="Start the app with scripts/run_app.py for api/web active runs.")
    parser.add_argument("--force-start", action="store_true", help="Pass --force to scripts/run_app.py when starting the app.")
    parser.add_argument("--stop-app", action="store_true", help="Stop an app process started by this helper after the run.")
    parser.add_argument(
        "--disable-local-text-cache",
        action="store_true",
        help="When used with --start-app, explicitly disable document/global local text-analysis caches for a fresh text LLM diagnostic run.",
    )
    parser.add_argument(
        "--allow-local-text-cache",
        action="store_true",
        help="Allow local active benchmark runs started by this helper to reuse document/global text-analysis caches.",
    )
    parser.add_argument("--launch-desktop", action="store_true", help="Open PaperEval.app before using --surface desktop.")
    parser.add_argument("--allow-runtime-not-ready", action="store_true", help="Allow active runs to proceed when GROBID or provider readiness is not confirmed.")
    parser.add_argument("--allow-surface-not-ready", action="store_true", help="Allow active runs to proceed when web/desktop surface checks fail.")
    parser.add_argument(
        "--require-diagnostic-only",
        action="store_true",
        help="For --active-run, refuse to start if dirty files include output-risk or unknown paths.",
    )
    parser.add_argument("--api-base", default="", help="Existing API base, usually http://127.0.0.1:8000/api.")
    parser.add_argument("--backend-port", type=int, default=8765)
    parser.add_argument("--frontend-port", type=int, default=5184, help="Legacy Vite dev UI port; normal web surface uses the backend-served /web/ route.")
    parser.add_argument("--llm-provider", choices=["local", "openai", "env"], default="local")
    parser.add_argument("--tier", default=None, choices=["smoke", "release", "deep", "all"], help="Case tier. Defaults to smoke for suite mode and all for single mode.")
    parser.add_argument("--case", action="append", dest="case_ids", help="Benchmark case id. Can be passed multiple times.")
    parser.add_argument("--include-unscored", action="store_true", help="Include diagnostic-coverage cases with draft gold standards.")
    parser.add_argument("--out-dir", default="", help="Directory for active benchmark artifacts.")
    parser.add_argument(
        "--artifact-url-base",
        default="",
        help="Optional base URL serving the active benchmark run directory; records link <base>/<case_id>/report.html.",
    )
    parser.add_argument("--max-concurrent", type=int, default=1, help="Maximum active benchmark jobs submitted at once. Defaults to 1.")
    parser.add_argument("--random-seed", type=int, default=None, help="Seed for --mode single random selection.")
    parser.add_argument("--startup-timeout", type=float, default=120.0)
    parser.add_argument("--queue-timeout", type=float, default=600.0, help="Seconds a submitted app job may remain queued before the helper stops submitting more work.")
    parser.add_argument("--timeout-per-case", type=float, default=1800.0)
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--fail-on-incompatible", action="store_true", help="Return non-zero when release-gate gold comparison or report validity fails.")
    parser.add_argument("--fail-on-diagnostic", action="store_true", help="Return non-zero when diagnostic-coverage cases fail.")
    parser.add_argument("--fail-on-output-risk", action="store_true", help="Return non-zero for --output-change-audit when dirty files may affect generated report outputs.")
    args = parser.parse_args()
    if args.single:
        args.mode = "single"
    if args.max_concurrent < 1:
        parser.error("--max-concurrent must be >= 1")
    if args.poll_interval <= 0:
        parser.error("--poll-interval must be > 0")
    if args.queue_timeout <= 0:
        parser.error("--queue-timeout must be > 0")
    if args.timeout_per_case <= 0:
        parser.error("--timeout-per-case must be > 0")

    status = 0
    if args.output_change_audit:
        status = max(
            status,
            _output_change_audit(
                json_output=bool(args.json),
                fail_on_output_risk=bool(args.fail_on_output_risk),
            ),
        )
    elif args.plan_next:
        status = max(
            status,
            _plan_next_case(
                Path(args.plan_next).expanduser(),
                tier=args.tier or "all",
                case_ids=args.case_ids or [],
                prefer_needs_fix=bool(args.prefer_needs_fix),
                prefer_lowest_score=bool(args.prefer_lowest_score),
                prefer_current_definition_refresh=bool(args.prefer_current_definition_refresh),
                json_output=bool(args.json),
            ),
        )
    elif args.summarize_history:
        history_path = Path(args.summarize_history).expanduser()
        status = max(status, _summarize_history(history_path, json_output=bool(args.json)))
        if args.record_history:
            status = max(status, _record_history(history_path, Path(args.history_db).expanduser()))
    elif args.summarize_stage_diagnostics:
        status = max(
            status,
            _summarize_stage_diagnostics(
                Path(args.summarize_stage_diagnostics).expanduser(),
                json_output=bool(args.json),
            ),
        )
    elif args.record_history:
        status = max(
            status,
            _record_history(
                ROOT / "test" / "active_benchmark",
                Path(args.history_db).expanduser(),
            ),
        )
    elif args.summarize_run:
        status = max(status, _summarize_existing_run(Path(args.summarize_run).expanduser(), json_output=bool(args.json)))
    elif args.write_detailed_report:
        status = max(
            status,
            _write_detailed_report_from_path(
                Path(args.write_detailed_report).expanduser(),
                detailed_analysis_url=str(args.detailed_analysis_url or ""),
                webapp_url=str(args.webapp_url or ""),
                media_json=str(args.media_json or ""),
                fetch_media_assets=bool(args.fetch_media_assets),
                api_base=_normalize_api_base(args.api_base, args.backend_port) if args.api_base else "",
            ),
        )
    elif args.active_run:
        status = max(status, _run_active_benchmark(args))
    else:
        status = max(status, _validate_manifest() if args.validate or not args.run_checks else 0)
    if args.run_checks:
        status = max(status, _run_checks())
    return status


if __name__ == "__main__":
    raise SystemExit(main())
