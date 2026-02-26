from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request


def http_json(url: str, *, method: str = "GET", timeout: float = 2.0) -> dict:
    req = urllib.request.Request(url=url, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        payload = response.read().decode("utf-8")
    return json.loads(payload)


def wait_ready(url: str, timeout_sec: int = 40) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            payload = http_json(url, timeout=1.5)
            if payload.get("backend_ready") is True:
                return True
        except Exception:
            pass
        time.sleep(0.4)
    return False


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def terminate_process_tree(pid: int) -> None:
    try:
        pgid = os.getpgid(pid)
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
        os.kill(pid, signal.SIGTERM)
    except Exception:
        return
    time.sleep(0.4)
    try:
        os.kill(pid, signal.SIGKILL)
    except Exception:
        return


def read_pid_file(root: Path) -> dict:
    path = root / ".run" / "pids.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke checks for PaperEval desktop lifecycle")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--app-name", default="PaperEval")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    app_bundle = root / f"{args.app_name}.app"
    if not app_bundle.exists():
        raise SystemExit(f"App bundle not found: {app_bundle}")

    py = root / ".venv" / "bin" / "python"
    if not py.exists():
        py = Path(sys.executable)

    env = os.environ.copy()
    env["PAPER_EVAL_ROOT"] = str(root)
    cmd = [str(py), "scripts/run_app.py", "--api-only", "--force", "--backend-port", "8000"]

    proc = subprocess.Popen(
        cmd,
        cwd=str(root),
        env=env,
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    status_url = "http://127.0.0.1:8000/api/status"
    if not wait_ready(status_url, timeout_sec=45):
        terminate_process_tree(proc.pid)
        raise SystemExit("Smoke failed: backend did not become ready on port 8000")

    try:
        status = http_json(status_url)
    except Exception as exc:
        terminate_process_tree(proc.pid)
        raise SystemExit(f"Smoke failed: could not read /api/status: {exc}") from exc

    if "processing" not in status:
        terminate_process_tree(proc.pid)
        raise SystemExit("Smoke failed: /api/status missing processing payload")

    try:
        recover = http_json("http://127.0.0.1:8000/api/processing/recover", method="POST")
        if recover.get("status") != "recovered":
            raise RuntimeError("processing/recover did not return recovered")
    except Exception as exc:
        terminate_process_tree(proc.pid)
        raise SystemExit(f"Smoke failed: /api/processing/recover failed: {exc}") from exc

    try:
        stop_payload = http_json("http://127.0.0.1:8000/api/stop", method="POST")
        if stop_payload.get("status") != "stopping":
            raise RuntimeError("stop endpoint did not report stopping")
    except urllib.error.URLError:
        # The process can exit before response returns; treat as acceptable.
        pass

    deadline = time.time() + 10
    while time.time() < deadline:
        if proc.poll() is not None:
            break
        time.sleep(0.25)

    if proc.poll() is None:
        terminate_process_tree(proc.pid)

    pids = read_pid_file(root)
    leftover = []
    for key in ["backend_pid"]:
        pid = pids.get(key)
        if isinstance(pid, int) and pid > 0 and pid_alive(pid):
            leftover.append((key, pid))

    if leftover:
        raise SystemExit(f"Smoke failed: leftover processes detected: {leftover}")

    print("Smoke OK: startup/readiness/recover/stop passed with no leftover processes")


if __name__ == "__main__":
    main()
