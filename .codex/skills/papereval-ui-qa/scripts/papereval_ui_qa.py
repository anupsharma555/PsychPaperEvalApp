#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[4]


EXPECTED_FILES = [
    "Makefile",
    "scripts/run_app.py",
    "desktop_ui/package.json",
    "desktop_ui/src/App.jsx",
    "desktop_ui/src/api.js",
    "desktop_ui/src/styles.css",
    "desktop_shell/package.json",
    "desktop_shell/src-tauri/tauri.conf.json",
    "backend/app/api/routes.py",
]


def _static_check() -> int:
    missing = [path for path in EXPECTED_FILES if not (ROOT / path).exists()]
    print("UI static check:")
    for path in EXPECTED_FILES:
        status = "missing" if path in missing else "ok"
        print(f"- {path}: {status}")

    package_path = ROOT / "desktop_ui" / "package.json"
    if package_path.exists():
        payload = json.loads(package_path.read_text(encoding="utf-8"))
        scripts = payload.get("scripts", {})
        print("\ndesktop_ui scripts:")
        if isinstance(scripts, dict):
            for name, command in sorted(scripts.items()):
                print(f"- {name}: {command}")

    return 1 if missing else 0


def _probe(url: str) -> int:
    request = Request(url, headers={"User-Agent": "PaperEval-UI-QA/1.0"})
    try:
        with urlopen(request, timeout=5) as response:
            body = response.read(4000).decode("utf-8", errors="replace")
            print(f"probe: {url}")
            print(f"status: {response.status}")
            print(f"bytes_sampled: {len(body)}")
            title = ""
            lower = body.lower()
            if "<title>" in lower and "</title>" in lower:
                start = lower.index("<title>") + len("<title>")
                end = lower.index("</title>", start)
                title = body[start:end].strip()
            print(f"title: {title or 'not found'}")
            return 0 if 200 <= int(response.status) < 400 else 1
    except Exception as exc:
        print(f"probe failed: {url}: {exc}", file=sys.stderr)
        return 1


def _status(api_base: str) -> int:
    url = api_base.rstrip("/") + "/status"
    request = Request(url, headers={"User-Agent": "PaperEval-UI-QA/1.0"})
    try:
        with urlopen(request, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
    except Exception as exc:
        print(f"status failed: {url}: {exc}", file=sys.stderr)
        return 1
    fields = {
        "provider": payload.get("provider"),
        "backend_ready": payload.get("ready"),
        "model_ready": payload.get("model_ready"),
        "local_gpu_mode": payload.get("local_gpu_mode"),
        "local_gpu_smoke_status": payload.get("local_gpu_smoke_status"),
        "grobid": payload.get("grobid"),
        "processing": payload.get("processing"),
    }
    print(f"status: {url}")
    print(json.dumps(fields, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PaperEval UI QA static/probe checks.")
    parser.add_argument("--static", action="store_true", help="Check expected UI/desktop files.")
    parser.add_argument("--probe", help="Probe an already-running local UI URL.")
    parser.add_argument("--status", help="Probe an already-running backend API base URL, e.g. http://127.0.0.1:8000/api.")
    args = parser.parse_args()

    status = 0
    if args.static or not (args.probe or args.status):
        status = max(status, _static_check())
    if args.probe:
        status = max(status, _probe(args.probe))
    if args.status:
        status = max(status, _status(args.status))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
