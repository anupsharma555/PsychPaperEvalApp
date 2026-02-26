from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import signal
import threading
import time
from typing import Any


def _run_dir() -> Path:
    root = os.environ.get("PAPER_EVAL_ROOT")
    if root:
        return Path(root) / ".run"
    return Path(".run")


def _pid_file() -> Path:
    return _run_dir() / "pids.json"


def _events_file() -> Path:
    return _run_dir() / "runtime_events.jsonl"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def read_pids() -> dict:
    path = _pid_file()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def log_runtime_event(event: str, details: dict[str, Any] | None = None) -> None:
    payload = {
        "event": event,
        "timestamp": datetime.utcnow().isoformat(),
        "details": details or {},
    }
    path = _events_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("a") as handle:
            handle.write(json.dumps(payload) + "\n")
    except Exception:
        return


def read_runtime_events(limit: int = 25) -> list[dict[str, Any]]:
    paths = [_events_file(), _run_dir() / "lifecycle_events.jsonl"]
    lines: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        try:
            lines.extend(path.read_text().splitlines())
        except Exception:
            continue
    if not lines:
        return []
    events: list[dict[str, Any]] = []
    for raw in lines[-max(limit, 1) :]:
        try:
            parsed = json.loads(raw)
        except Exception:
            continue
        if isinstance(parsed, dict):
            events.append(parsed)
    return events


def stop_app(delay_sec: float = 0.5) -> dict:
    pids = read_pids()
    backend_pid = pids.get("backend_pid")
    frontend_pid = pids.get("frontend_pid")
    backend_pgid = pids.get("backend_pgid")
    frontend_pgid = pids.get("frontend_pgid")
    log_runtime_event(
        "stop_requested",
        {
            "backend_pid": backend_pid,
            "frontend_pid": frontend_pid,
            "backend_pgid": backend_pgid,
            "frontend_pgid": frontend_pgid,
        },
    )

    def _kill() -> None:
        _terminate_target(frontend_pid, frontend_pgid)
        _terminate_target(backend_pid, backend_pgid)
        # Ensure job runner is stopped if we are in-process.
        try:
            from app.services.jobs import job_runner

            job_runner.stop()
        except Exception:
            pass
        _clear_pid_file()
        log_runtime_event("stop_completed", {"status": "ok"})

    threading.Timer(delay_sec, _kill).start()
    return pids


def _terminate_target(pid: int | None, pgid: int | None) -> None:
    if isinstance(pgid, int) and pgid > 0:
        _terminate_process_group(pgid)
        return
    if not isinstance(pid, int) or pid <= 0:
        return
    if not _pid_alive(pid):
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except Exception:
        return
    time.sleep(0.4)
    if _pid_alive(pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:
            pass


def _terminate_process_group(pgid: int) -> None:
    try:
        os.killpg(pgid, signal.SIGTERM)
    except Exception:
        return
    time.sleep(0.5)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except Exception:
        return


def _clear_pid_file() -> None:
    path = _pid_file()
    try:
        if path.exists():
            path.unlink()
    except Exception:
        return
