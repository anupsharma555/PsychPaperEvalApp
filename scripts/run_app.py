from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
GROBID_CONTAINER_NAME = "papereval-grobid"
GROBID_IMAGE = "grobid/grobid:0.8.2-crf"
GROBID_ADOPTABLE_CONTAINER_NAMES = (GROBID_CONTAINER_NAME, "grobid")
DOCKER_CLI_CANDIDATES = (
    "/Applications/Docker.app/Contents/Resources/bin/docker",
    "/usr/local/bin/docker",
    "/opt/homebrew/bin/docker",
)


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


def _grobid_url() -> str:
    return os.environ.get("GROBID_URL", "http://localhost:8070").strip() or "http://localhost:8070"


def _grobid_health_url() -> str:
    return _grobid_url().rstrip("/") + "/api/isalive"


def _manage_grobid_enabled() -> bool:
    value = os.environ.get("PAPER_EVAL_MANAGE_GROBID", "true").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _grobid_host_port() -> str:
    parsed = urlparse(_grobid_url())
    if parsed.port:
        return str(parsed.port)
    return "8070"


def _docker_cmd() -> str | None:
    configured = os.environ.get("DOCKER_BIN", "").strip()
    candidates: list[str] = [configured] if configured else []
    discovered = shutil.which("docker")
    if discovered:
        candidates.append(discovered)
    candidates.extend(DOCKER_CLI_CANDIDATES)

    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return discovered or "docker"


def _docker_context() -> str:
    configured = os.environ.get("PAPER_EVAL_DOCKER_CONTEXT", "").strip()
    if configured:
        return configured
    if sys.platform == "darwin":
        return "desktop-linux"
    return ""


def _docker_args(*args: str) -> list[str] | None:
    docker = _docker_cmd()
    if not docker:
        return None
    context = _docker_context()
    if context:
        return [docker, "--context", context, *args]
    return [docker, *args]


def _docker_ready() -> bool:
    cmd = _docker_args("info")
    if not cmd:
        return False
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=8, check=False)
    except Exception:
        return False
    return result.returncode == 0


def _start_docker_desktop() -> None:
    if sys.platform != "darwin":
        return
    try:
        subprocess.Popen(["open", "-a", "Docker"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        return


def _wait_for_docker(timeout_sec: int = 90) -> bool:
    if _docker_ready():
        return True
    _start_docker_desktop()
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if _docker_ready():
            return True
        time.sleep(2.0)
    return False


def _docker_container_exists(name: str) -> bool:
    cmd = _docker_args("ps", "-a", "--format", "{{.Names}}")
    if not cmd:
        return False
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except Exception:
        return False
    return name in {line.strip() for line in result.stdout.splitlines()}


def _docker_container_info(name: str) -> dict | None:
    cmd = _docker_args("inspect", name)
    if not cmd:
        return None
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    try:
        payload = json.loads(result.stdout)
    except Exception:
        return None
    if not isinstance(payload, list) or not payload:
        return None
    info = payload[0]
    return info if isinstance(info, dict) else None


def _is_expected_grobid_container(info: dict, port: str) -> bool:
    config = info.get("Config") if isinstance(info.get("Config"), dict) else {}
    image = str(config.get("Image") or "")
    if image != GROBID_IMAGE:
        return False

    host_config = info.get("HostConfig") if isinstance(info.get("HostConfig"), dict) else {}
    bindings = host_config.get("PortBindings") if isinstance(host_config.get("PortBindings"), dict) else {}
    bound_ports = bindings.get("8070/tcp") if isinstance(bindings.get("8070/tcp"), list) else []
    for binding in bound_ports:
        if isinstance(binding, dict) and str(binding.get("HostPort") or "") == port:
            return True
    return False


def _grobid_container_running(info: dict) -> bool:
    state = info.get("State") if isinstance(info.get("State"), dict) else {}
    return bool(state.get("Running"))


def _find_adoptable_grobid_container(port: str, *, include_stopped: bool) -> str | None:
    for name in GROBID_ADOPTABLE_CONTAINER_NAMES:
        info = _docker_container_info(name)
        if not info or not _is_expected_grobid_container(info, port):
            continue
        if include_stopped or _grobid_container_running(info):
            return name
    return None


def _start_grobid_container() -> str | None:
    if not _manage_grobid_enabled():
        _log_lifecycle("grobid_manage_disabled", {"url": _grobid_url()})
        return None
    port = _grobid_host_port()
    if _http_ready(_grobid_health_url(), timeout_sec=2.0):
        if _docker_ready():
            existing_running = _find_adoptable_grobid_container(port, include_stopped=False)
            if existing_running:
                _log_lifecycle("grobid_already_ready", {"container": existing_running, "url": _grobid_url()})
                return existing_running
        _log_lifecycle("grobid_already_ready_external", {"url": _grobid_url()})
        return None

    if not _wait_for_docker():
        if _http_ready(_grobid_health_url(), timeout_sec=2.0):
            _log_lifecycle("grobid_already_ready_external", {"url": _grobid_url(), "reason": "docker_unavailable"})
            return None
        _log_lifecycle("grobid_start_failed", {"reason": "docker_unavailable", "url": _grobid_url()})
        return None

    container_name = _find_adoptable_grobid_container(port, include_stopped=True)
    docker_start = _docker_args("start")
    docker_run = _docker_args("run")
    if not docker_start or not docker_run:
        _log_lifecycle("grobid_start_failed", {"reason": "docker_cli_unavailable", "url": _grobid_url()})
        return None
    if container_name:
        cmd = [*docker_start, container_name]
    elif _docker_container_exists(GROBID_CONTAINER_NAME):
        cmd = [*docker_start, GROBID_CONTAINER_NAME]
        container_name = GROBID_CONTAINER_NAME
    else:
        container_name = GROBID_CONTAINER_NAME
        cmd = [
            *docker_run,
            "-d",
            "--name",
            GROBID_CONTAINER_NAME,
            "--init",
            "--ulimit",
            "core=0",
            "-p",
            f"{port}:8070",
            GROBID_IMAGE,
        ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
    if result.returncode != 0:
        _log_lifecycle(
            "grobid_start_failed",
            {"reason": "docker_command_failed", "returncode": result.returncode, "stderr": result.stderr[-500:]},
        )
        return None

    deadline = time.time() + 90
    while time.time() < deadline:
        if _http_ready(_grobid_health_url(), timeout_sec=2.0):
            _log_lifecycle("grobid_ready", {"container": container_name, "url": _grobid_url()})
            return container_name
        time.sleep(2.0)

    _log_lifecycle("grobid_start_failed", {"reason": "readiness_timeout", "url": _grobid_url()})
    return container_name


def _stop_grobid_container(container_name: str | None) -> None:
    if not container_name or not _manage_grobid_enabled():
        return
    if container_name not in GROBID_ADOPTABLE_CONTAINER_NAMES:
        _log_lifecycle("grobid_stop_skipped", {"container": container_name, "reason": "not_adoptable"})
        return
    try:
        cmd = _docker_args("stop", container_name)
        if not cmd:
            _log_lifecycle("grobid_stop_failed", {"container": container_name, "error": "docker_cli_unavailable"})
            return
        subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
        _log_lifecycle("grobid_stopped", {"container": container_name})
    except Exception as exc:
        _log_lifecycle("grobid_stop_failed", {"container": container_name, "error": str(exc)})


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
    env.setdefault("LLM_PROVIDER", "openai")
    env.setdefault("PYTHONFAULTHANDLER", "1")
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
    grobid_container_name: str | None = None

    reload_backend = args.reload_backend

    try:
        grobid_container_name = _start_grobid_container()
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
        _stop_grobid_container(grobid_container_name)
        if log_handle:
            log_handle.close()
        raise SystemExit(f"Failed to start backend: {exc}")

    if backend_proc:
        ok = _wait_ready(backend_proc, f"http://127.0.0.1:{backend_port}/api/status")
        if not ok:
            _log_lifecycle("startup_failed", {"component": "backend"})
            for proc in procs:
                _terminate_process(proc.pid, _safe_getpgid(proc.pid))
            _stop_grobid_container(grobid_container_name)
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
        nonlocal grobid_container_name
        _log_lifecycle("shutdown_requested", {"signal": str(_sig[0]) if _sig else "manual"})
        for proc in procs:
            if proc.poll() is None:
                _terminate_process(proc.pid, _safe_getpgid(proc.pid))
        _stop_grobid_container(grobid_container_name)
        grobid_container_name = None
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
        _stop_grobid_container(grobid_container_name)
        if log_handle and not log_handle.closed:
            log_handle.close()


if __name__ == "__main__":
    main()
