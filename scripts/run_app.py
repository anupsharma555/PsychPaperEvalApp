from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def _resolve_python() -> Path:
    venv_python = ROOT / ".venv" / "bin" / "python"
    if venv_python.exists():
        return venv_python
    return Path(sys.executable)


def _resolve_uvicorn(py: Path, *, reload_backend: bool) -> list[str]:
    cmd = [str(py), "-m", "uvicorn", "app.main:app", "--app-dir", "backend"]
    if reload_backend:
        cmd.append("--reload")
    return cmd


def _is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _find_free_port(start: int, max_tries: int = 20) -> int:
    port = start
    for _ in range(max_tries):
        if not _is_port_in_use(port):
            return port
        port += 1
    return start


def _run_dir() -> Path:
    return ROOT / ".run"


def _pid_file() -> Path:
    return _run_dir() / "pids.json"


def _lifecycle_file() -> Path:
    return _run_dir() / "lifecycle_events.jsonl"


def _read_pid_file() -> dict:
    path = _pid_file()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _write_pid_file(payload: dict) -> None:
    path = _pid_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(path)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _notify(title: str, message: str) -> None:
    try:
        subprocess.Popen(["osascript", "-e", f'display alert "{title}" message "{message}"'])
    except Exception:
        pass


def _fingerprint(cmd: list[str]) -> str:
    return hashlib.sha1(" ".join(cmd).encode("utf-8")).hexdigest()


def _log_lifecycle(event: str, details: dict | None = None) -> None:
    payload = {
        "event": event,
        "timestamp": datetime.utcnow().isoformat(),
        "details": details or {},
    }
    path = _lifecycle_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("a") as handle:
            handle.write(json.dumps(payload) + "\n")
    except Exception:
        return


def _terminate_process(pid: int | None, pgid: int | None) -> None:
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
    if isinstance(pid, int) and pid > 0:
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            return
        time.sleep(0.4)
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:
            return


def _safe_getpgid(pid: int | None) -> int | None:
    if not isinstance(pid, int):
        return None
    try:
        return os.getpgid(pid)
    except Exception:
        return None


def _http_ready(url: str, timeout_sec: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout_sec) as resp:
            return 200 <= resp.status < 500
    except urllib.error.URLError:
        return False
    except Exception:
        return False


def _wait_ready(proc: subprocess.Popen, url: str, timeout_sec: int = 45) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        if _http_ready(url):
            return True
        time.sleep(0.5)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PaperEval backend API")
    parser.add_argument(
        "--api-only",
        action="store_true",
        help="Deprecated compatibility flag; backend-only mode is always used.",
    )
    parser.add_argument("--log-file", default=None, help="Append logs to this file")
    parser.add_argument("--notify", action="store_true", help="Show macOS alerts for status")
    parser.add_argument("--force", action="store_true", help="Start a new instance even if one is running")
    parser.add_argument(
        "--reload-backend",
        action="store_true",
        help="Run uvicorn with --reload.",
    )
    parser.add_argument(
        "--backend-port",
        type=int,
        default=None,
        help="Preferred backend port (defaults to first available from 8000).",
    )
    args = parser.parse_args()

    py = _resolve_python()
    env = os.environ.copy()
    env["PAPER_EVAL_ROOT"] = str(ROOT)
    _run_dir().mkdir(parents=True, exist_ok=True)

    log_handle = None
    if args.log_file:
        log_path = Path(args.log_file).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = open(log_path, "a", buffering=1)
        print(f"[run_app] logging to {log_path}", file=log_handle)

    existing = _read_pid_file()
    if existing:
        backend_pid = existing.get("backend_pid")
        frontend_pid = existing.get("frontend_pid")
        if (backend_pid and _pid_alive(backend_pid)) or (frontend_pid and _pid_alive(frontend_pid)):
            if not args.force:
                backend_port = existing.get("backend_port", 8000)
                if args.notify:
                    _notify("PaperEval already running", f"Backend at http://127.0.0.1:{backend_port}")
                print("PaperEval backend already running. Use --force to restart.")
                return
            _terminate_process(existing.get("backend_pid"), existing.get("backend_pgid"))
            _terminate_process(existing.get("frontend_pid"), existing.get("frontend_pgid"))

    procs: list[subprocess.Popen] = []
    backend_port = args.backend_port if args.backend_port else _find_free_port(8000)
    backend_proc: subprocess.Popen | None = None

    reload_backend = args.reload_backend

    try:
        backend_cmd = _resolve_uvicorn(py, reload_backend=reload_backend) + ["--port", str(backend_port)]
        print("Starting backend...")
        backend_proc = subprocess.Popen(
            backend_cmd,
            cwd=str(ROOT),
            env=env,
            stdout=log_handle,
            stderr=log_handle,
            start_new_session=True,
        )
        procs.append(backend_proc)
    except Exception as exc:
        for proc in procs:
            _terminate_process(proc.pid, _safe_getpgid(proc.pid))
        if log_handle:
            log_handle.close()
        raise SystemExit(f"Failed to start backend: {exc}")

    if backend_proc:
        ok = _wait_ready(backend_proc, f"http://127.0.0.1:{backend_port}/api/status")
        if not ok:
            _log_lifecycle("startup_failed", {"component": "backend"})
            for proc in procs:
                _terminate_process(proc.pid, _safe_getpgid(proc.pid))
            if log_handle:
                log_handle.close()
            raise SystemExit("Backend did not become ready in time.")

    pid_payload = {
        "backend_pid": backend_proc.pid if backend_proc else None,
        "frontend_pid": None,
        "backend_pgid": _safe_getpgid(backend_proc.pid if backend_proc else None),
        "frontend_pgid": None,
        "backend_port": backend_port,
        "frontend_port": None,
        "backend_cmd_fingerprint": _fingerprint(_resolve_uvicorn(py, reload_backend=reload_backend)),
        "frontend_cmd_fingerprint": None,
        "started_at": datetime.utcnow().isoformat(),
    }
    _write_pid_file(pid_payload)
    _log_lifecycle("startup_completed", pid_payload)

    def shutdown(*_sig) -> None:
        _log_lifecycle("shutdown_requested", {"signal": str(_sig[0]) if _sig else "manual"})
        for proc in procs:
            if proc.poll() is None:
                _terminate_process(proc.pid, _safe_getpgid(proc.pid))
        pid_path = _pid_file()
        if pid_path.exists():
            pid_path.unlink()
        _log_lifecycle("shutdown_completed", {})
        if log_handle:
            log_handle.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        while True:
            alive = [proc for proc in procs if proc.poll() is None]
            if not alive:
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        shutdown()
    finally:
        if log_handle and not log_handle.closed:
            log_handle.close()


if __name__ == "__main__":
    main()
