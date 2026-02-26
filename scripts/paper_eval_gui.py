from __future__ import annotations

import atexit
import importlib
import json
import os
from pathlib import Path
import plistlib
import shutil
import signal
import subprocess
import sys
import time
import traceback

LOG_PATH = Path.home() / "Library" / "Logs" / "PaperEval" / "gui_boot.log"
APP_VERSION = "V7"
LEGACY_LAUNCHER_ENV = "PAPER_EVAL_ALLOW_LEGACY_UI"
V7_MANIFEST = ".run/build_manifest.json"
V7_DISPATCH_ENV = "PAPER_EVAL_SKIP_V7_DISPATCH"


def _log(message: str) -> None:
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a") as handle:
            handle.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message.rstrip()}\n")
    except Exception:
        return


def _show_dialog(title: str, message: str, *, error: bool) -> None:
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        if error:
            messagebox.showerror(title, message)
        else:
            messagebox.showinfo(title, message)
        root.destroy()
        return
    except Exception:
        pass

    # Fallback for environments where Tk isn't available.
    level = "critical" if error else "informational"
    escaped_title = title.replace('"', "'")
    escaped_message = message.replace('"', "'").replace("\n", "\\n")
    script = (
        f'display alert "{escaped_title}" '
        f'message "{escaped_message}" '
        f"as {level} buttons {{\"OK\"}} default button \"OK\""
    )
    try:
        subprocess.run(["osascript", "-e", script], check=False, timeout=5)
    except Exception:
        return


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _terminate_pid(pid: int, *, timeout_sec: float = 2.0) -> bool:
    try:
        os.kill(pid, signal.SIGTERM)
    except Exception:
        return not _pid_alive(pid)

    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if not _pid_alive(pid):
            return True
        time.sleep(0.1)

    try:
        os.kill(pid, signal.SIGKILL)
    except Exception:
        pass
    return not _pid_alive(pid)


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def _read_json(path: Path) -> dict:
    try:
        if not path.exists():
            return {}
        return json.loads(path.read_text())
    except Exception:
        return {}


def _bring_process_to_front(pid: int) -> None:
    script = (
        'tell application "System Events"\n'
        f"  set frontmost of (first process whose unix id is {pid}) to true\n"
        "end tell"
    )
    try:
        subprocess.run(["osascript", "-e", script], check=False, timeout=3)
    except Exception:
        return


def _is_repo_root(path: Path) -> bool:
    return (
        (path / "backend").exists()
        and (path / "scripts").exists()
        and ((path / "desktop_legacy").exists() or (path / "desktop_ui").exists())
    )


def _detect_root() -> Path:
    env_root = os.environ.get("PAPER_EVAL_ROOT", "").strip()
    if env_root:
        candidate = Path(env_root).expanduser().resolve()
        if _is_repo_root(candidate):
            return candidate

    this_file = Path(__file__).resolve()
    for parent in [this_file.parent, *this_file.parents]:
        if _is_repo_root(parent):
            return parent

    cwd = Path.cwd().resolve()
    for parent in [cwd, *cwd.parents]:
        if _is_repo_root(parent):
            return parent

    return this_file.parents[1]


def _cleanup_instance_file(path: Path, pid: int) -> None:
    payload = _read_json(path)
    if int(payload.get("pid", -1)) != pid:
        return
    try:
        path.unlink(missing_ok=True)
    except Exception:
        return


def _candidate_v7_bundles(root: Path) -> list[Path]:
    candidates: list[Path] = []
    manifest = _read_json(root / V7_MANIFEST)
    artifact = str(manifest.get("artifact", "")).strip()
    if artifact:
        try:
            candidates.append(Path(artifact).expanduser().resolve())
        except Exception:
            pass

    candidates.append(
        (root / "desktop_shell" / "src-tauri" / "target" / "release" / "bundle" / "macos" / "PaperEval.app").resolve()
    )
    candidates.append(
        (root / "desktop_shell" / "src-tauri" / "target" / "release" / "bundle" / "macos" / "PaperEval Desktop.app").resolve()
    )
    candidates.append((root / "dist" / "PaperEval.app").resolve())
    candidates.append((root / "build" / "PaperEval.app").resolve())

    search_roots = [
        root / "desktop_shell" / "src-tauri" / "target",
        root / "dist",
        root / "build",
    ]
    for search_root in search_roots:
        if not search_root.exists():
            continue
        for match in search_root.rglob("PaperEval*.app"):
            try:
                candidates.append(match.resolve())
            except Exception:
                continue

    deduped: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _is_py2app_bundle(bundle: Path) -> bool:
    info_plist = bundle / "Contents" / "Info.plist"
    if not info_plist.exists():
        return False
    try:
        with info_plist.open("rb") as handle:
            payload = plistlib.load(handle)
    except Exception:
        return False
    return "PyMainFileNames" in payload or "PythonInfoDict" in payload


def _dispatch_to_v7_if_available(root: Path) -> bool:
    if os.environ.get(V7_DISPATCH_ENV) == "1":
        _log("V7 dispatch skipped via environment override.")
        return False

    launcher_bundle = (root / "PaperEval.app").resolve()
    for bundle in _candidate_v7_bundles(root):
        if not bundle.exists():
            continue
        if bundle == launcher_bundle:
            # If this launcher itself is the V7 bundle, do not recurse.
            continue
        if not (bundle / "Contents" / "MacOS").exists():
            continue
        if _is_py2app_bundle(bundle):
            _log(f"Skipping legacy py2app bundle candidate: {bundle}")
            continue
        try:
            _log(f"Dispatching to V7 desktop bundle: {bundle}")
            subprocess.Popen(["open", str(bundle)])
            return True
        except Exception as exc:
            _log(f"Failed to dispatch to V7 bundle {bundle}: {exc}")
            continue
    return False


def main() -> None:
    root = _detect_root()
    instance_file = root / ".run" / "desktop_instance.json"

    if os.environ.get(LEGACY_LAUNCHER_ENV) != "1":
        if _dispatch_to_v7_if_available(root):
            return
        missing_tools = [tool for tool in ("cargo", "rustc") if shutil.which(tool) is None]
        tool_hint = ""
        if missing_tools:
            tool_hint = (
                "Missing desktop build tools detected: "
                f"{', '.join(missing_tools)}.\n"
                "Install Rust toolchain first (rustup), then build.\n\n"
            )
        _log("No V7 desktop bundle was found; blocking legacy Tk launcher.")
        message = (
            "PaperEval Desktop V7 was not found.\n\n"
            f"{tool_hint}"
            f"Build and install the canonical app in:\n{root}\n\n"
            "Run:\n"
            "make desktop-env\n"
            "make desktop-build\n\n"
            "Then relaunch this icon. To run the legacy Tk UI intentionally, set\n"
            "PAPER_EVAL_ALLOW_LEGACY_UI=1 before launching."
        )
        _show_dialog(
            "PaperEval",
            message,
            error=True,
        )
        return

    os.environ.setdefault("PAPER_EVAL_ROOT", str(root))
    os.environ.setdefault("PAPER_EVAL_DESKTOP_INSTANCE_FILE", str(instance_file))

    # Force repo root to highest import precedence so desktop modules always
    # resolve from the live source tree, not stale bundled copies.
    root_str = str(root)
    sys.path = [entry for entry in sys.path if entry != root_str]
    sys.path.insert(0, root_str)

    _log(f"Root: {root}")
    _log("Starting PaperEval desktop launcher")

    skip_single_instance = os.environ.get("PAPER_EVAL_SKIP_SINGLE_INSTANCE") == "1"
    if not skip_single_instance:
        existing = _read_json(instance_file)
        existing_pid = int(existing.get("pid", -1)) if existing else -1
        existing_version = str(existing.get("app_version", ""))
        if existing_pid > 0 and existing_pid != os.getpid() and _pid_alive(existing_pid):
            if existing_version != APP_VERSION:
                _log(
                    f"Existing instance detected with version '{existing_version or 'unknown'}'; "
                    f"replacing with {APP_VERSION}."
                )
                if not _terminate_pid(existing_pid):
                    _show_dialog(
                        "PaperEval",
                        "A previous PaperEval instance is still running and could not be replaced.\n"
                        "Please quit it from Activity Monitor, then relaunch.",
                        error=True,
                    )
                    return
                time.sleep(0.2)
            else:
                _log(f"Existing instance detected (pid={existing_pid}), bringing to front.")
                _bring_process_to_front(existing_pid)
                _show_dialog(
                    "PaperEval",
                    "PaperEval is already running. Brought the existing window to front.",
                    error=False,
                )
                return

    _write_json_atomic(
        instance_file,
        {
            "pid": os.getpid(),
            "app_version": APP_VERSION,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "root": str(root),
            "ui_python": sys.executable,
            "argv": sys.argv,
        },
    )
    atexit.register(_cleanup_instance_file, instance_file, os.getpid())

    try:
        from desktop_legacy.app import main as desktop_main
    except Exception as exc:
        err = traceback.format_exc()
        _log("Failed to import desktop app")
        _log(err)
        _show_dialog(
            "PaperEval",
            "PaperEval failed to start the desktop UI.\n\n"
            f"{exc}\n\n"
            f"See log: {LOG_PATH}",
            error=True,
        )
        raise

    try:
        desktop_app_mod = importlib.import_module("desktop_legacy.app")
        source_step_mod = importlib.import_module("desktop_legacy.views.source_step")
        _log(f"Loaded module desktop_legacy.app from: {getattr(desktop_app_mod, '__file__', 'unknown')}")
        _log(
            "Loaded module desktop_legacy.views.source_step from: "
            f"{getattr(source_step_mod, '__file__', 'unknown')}"
        )
    except Exception:
        _log("Failed to log loaded module paths")

    try:
        desktop_main()
    except KeyboardInterrupt:
        _log("Desktop interrupted by keyboard signal")
    except Exception as exc:
        err = traceback.format_exc()
        _log("Desktop app crashed")
        _log(err)
        _show_dialog(
            "PaperEval",
            "PaperEval desktop UI crashed.\n\n"
            f"{exc}\n\n"
            f"See log: {LOG_PATH}",
            error=True,
        )
        raise
    finally:
        _cleanup_instance_file(instance_file, os.getpid())


if __name__ == "__main__":
    main()
