from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox
from typing import Optional
import traceback

from desktop_legacy.api_client import DesktopApiClient, DesktopApiError
from desktop_legacy.models import BackendStatus, DesktopRuntimeInfo, JobRow, now_ts
from desktop_legacy.views.event_log import EventLogBar
from desktop_legacy.views.queue_panel import QueuePanel
from desktop_legacy.views.report_panel import ReportPanel
from desktop_legacy.views.source_step import SourceStepView
from desktop_legacy.views.status_bar import StatusBar
from desktop_legacy.workflow import WorkflowState, validate_source_input

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / ".run"
INSTANCE_FILE = RUN_DIR / "desktop_instance.json"
LOG_DIR = Path.home() / "Library" / "Logs" / "PaperEval"
LOG_PATH = LOG_DIR / "desktop_ui.log"
BACKEND_LOG_PATH = LOG_DIR / "backend_desktop.log"
DESKTOP_RUNTIME_PATH = LOG_DIR / "desktop_runtime.json"

STATE_BOOTING = "booting"
STATE_BACKEND_STARTING = "backend_starting"
STATE_BACKEND_READY = "backend_ready"
STATE_DEGRADED = "degraded"
STATE_STOPPING = "stopping"
UI_VERSION = "V6"


def _resolve_backend_python() -> Path:
    venv_python = ROOT / ".venv" / "bin" / "python"
    if venv_python.exists():
        return venv_python
    return Path("/usr/bin/python3")


def _build_backend_env() -> dict[str, str]:
    # Build a minimal clean environment instead of inheriting the full GUI process env.
    keep_keys = [
        "HOME",
        "PATH",
        "LANG",
        "LC_ALL",
        "TMPDIR",
        "SHELL",
        "USER",
        "LOGNAME",
        "TERM",
        "DYLD_LIBRARY_PATH",
        "DYLD_FRAMEWORK_PATH",
    ]
    env = {key: value for key, value in os.environ.items() if key in keep_keys}
    env["PAPER_EVAL_ROOT"] = str(ROOT)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _find_free_port(start: int = 8000, max_tries: int = 20) -> int:
    port = start
    for _ in range(max_tries):
        if not _is_port_in_use(port):
            return port
        port += 1
    return start


class PaperEvalDesktopApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(f"PaperEval Desktop {UI_VERSION}")
        self.geometry("1420x900")
        self.minsize(1220, 760)
        self.configure(bg="#eef1f5")
        self.option_add("*Font", "Helvetica 11")

        self.workflow = WorkflowState()
        self.backend_port = _find_free_port(8000)
        self.api_base = f"http://127.0.0.1:{self.backend_port}/api"
        self.api = DesktopApiClient(self.api_base)
        self.backend_proc: subprocess.Popen | None = None
        self.backend_log_handle = None
        self.backend_status: BackendStatus | None = None
        self.backend_ready = False
        self.backend_starting = False

        self.startup_state = STATE_BOOTING
        self.last_start_error: str | None = None
        self.next_retry_due_at: float | None = None
        self.last_runtime_event_ts: str | None = None
        self.last_runtime_event_key: str | None = None
        self.last_report_export_url: str | None = None
        self.last_poll_delay_ms = 2000

        self.jobs_by_id: dict[int, JobRow] = {}
        self._job_status_cache: dict[int, str] = {}
        self.selected_job_id: int | None = None
        self.selected_job_status: str | None = None
        self.selected_document_id: int | None = None

        self._ui_queue: queue.Queue = queue.Queue()
        self._event_dedupe: dict[str, float] = {}

        self.runtime_info = DesktopRuntimeInfo(
            ui_python_path=sys.executable,
            backend_python_path=str(_resolve_backend_python()),
            desktop_instance_pid=os.getpid(),
            backend_pid=None,
            startup_state=self.startup_state,
            last_start_error=None,
        )

        self._build_layout()
        self._initialize_visual_placeholders()
        self._apply_disabled_state(disabled=True)
        self._set_startup_state(STATE_BOOTING, "Initializing desktop workflow and health probes.")

        self._start_backend_async()
        self.after(80, self._drain_ui_queue)
        self.after(1000, self._tick_connection_status)
        self.after(1400, self._poll_backend)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_layout(self) -> None:
        header = tk.Frame(self, bg="#1f3548", padx=12, pady=10)
        header.pack(fill=tk.X)
        left = tk.Frame(header, bg="#1f3548")
        left.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(
            left,
            text=f"PaperEval Desktop {UI_VERSION}",
            fg="#ffffff",
            bg="#1f3548",
            anchor=tk.W,
            font=("Helvetica", 18, "bold"),
        ).pack(fill=tk.X)
        tk.Label(
            left,
            text="Guided workflow for source selection, analysis monitoring, and report review.",
            fg="#d6e3f0",
            bg="#1f3548",
            anchor=tk.W,
        ).pack(fill=tk.X, pady=(3, 0))

        right = tk.Frame(header, bg="#1f3548")
        right.pack(side=tk.RIGHT, padx=(8, 0))
        self.retry_btn = tk.Button(right, text="Retry Backend", command=self._retry_backend_now)
        self.retry_btn.pack(side=tk.RIGHT, padx=(6, 0))
        self.logs_btn = tk.Button(right, text="Open Logs", command=self._open_logs_folder)
        self.logs_btn.pack(side=tk.RIGHT, padx=(0, 6))

        self.status_bar = StatusBar(self)
        self.status_bar.pack(fill=tk.X, padx=10, pady=(10, 6))

        self.connection_card = tk.Frame(
            self,
            bg="#ffffff",
            bd=1,
            relief=tk.SOLID,
            highlightthickness=1,
            highlightbackground="#c7d1dc",
        )
        self.connection_card.pack(fill=tk.X, padx=10, pady=(0, 8))
        tk.Label(
            self.connection_card,
            text="Connection",
            bg="#dbe7f5",
            fg="#12304a",
            anchor=tk.W,
            padx=10,
            pady=6,
            font=("Helvetica", 11, "bold"),
        ).pack(fill=tk.X)
        conn_body = tk.Frame(self.connection_card, bg="#ffffff", padx=10, pady=8)
        conn_body.pack(fill=tk.X)

        self.connection_badge_var = tk.StringVar(value="Starting")
        self.connection_note_var = tk.StringVar(value="Preparing startup checks")
        self.connection_retry_var = tk.StringVar(value="")

        self.connection_badge_label = tk.Label(
            conn_body,
            textvariable=self.connection_badge_var,
            fg="#0d47a1",
            font=("Helvetica", 13, "bold"),
            anchor=tk.W,
            bg="#ffffff",
        )
        self.connection_badge_label.pack(side=tk.LEFT, padx=(2, 16))
        note = tk.Label(
            conn_body,
            textvariable=self.connection_note_var,
            fg="#1f2933",
            anchor=tk.W,
            justify=tk.LEFT,
            wraplength=700,
            bg="#ffffff",
        )
        note.pack(side=tk.LEFT, fill=tk.X, expand=True)
        retry = tk.Label(
            conn_body,
            textvariable=self.connection_retry_var,
            fg="#30475f",
            anchor=tk.E,
            bg="#ffffff",
        )
        retry.pack(side=tk.RIGHT, padx=(10, 6))

        main = tk.Frame(self, padx=10, pady=8, bg="#eef1f5")
        main.pack(fill=tk.BOTH, expand=True)
        main.grid_columnconfigure(0, weight=3, uniform="columns")
        main.grid_columnconfigure(1, weight=4, uniform="columns")
        main.grid_columnconfigure(2, weight=4, uniform="columns")
        main.grid_rowconfigure(0, weight=1)

        left_col = tk.Frame(main, bg="#eef1f5")
        left_col.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left_col.grid_columnconfigure(0, weight=1)
        left_col.grid_rowconfigure(0, weight=4)
        left_col.grid_rowconfigure(1, weight=2)

        self.source_step = SourceStepView(left_col, on_validate=self._validate_workflow_input, on_submit=self._submit_analysis)
        self.source_step.frame.grid(row=0, column=0, sticky="nsew", pady=(0, 8))

        controls = tk.Frame(
            left_col,
            bg="#ffffff",
            bd=1,
            relief=tk.SOLID,
            highlightthickness=1,
            highlightbackground="#c7d1dc",
        )
        controls.grid(row=1, column=0, sticky="nsew")
        tk.Label(
            controls,
            text="Processing Controls",
            bg="#dbe7f5",
            fg="#12304a",
            anchor=tk.W,
            padx=10,
            pady=6,
            font=("Helvetica", 11, "bold"),
        ).pack(fill=tk.X)
        controls_body = tk.Frame(controls, bg="#ffffff", padx=10, pady=10)
        controls_body.pack(fill=tk.BOTH, expand=True)
        tk.Label(
            controls_body,
            text="These actions control worker execution and recovery.",
            anchor=tk.W,
            justify=tk.LEFT,
            wraplength=330,
            fg="#1f2933",
            bg="#ffffff",
        ).pack(fill=tk.X, pady=(0, 8))
        self.pause_btn = tk.Button(controls_body, text="Pause Processing", command=self._pause_processing, state=tk.DISABLED)
        self.pause_btn.pack(fill=tk.X, pady=(0, 4))
        self.resume_btn = tk.Button(controls_body, text="Resume Processing", command=self._resume_processing, state=tk.DISABLED)
        self.resume_btn.pack(fill=tk.X, pady=(0, 4))
        self.clean_btn = tk.Button(controls_body, text="Clean Worker Orphans", command=self._clean_orphans, state=tk.DISABLED)
        self.clean_btn.pack(fill=tk.X, pady=(0, 4))
        self.recover_btn = tk.Button(controls_body, text="Recover Processing State", command=self._recover_processing, state=tk.DISABLED)
        self.recover_btn.pack(fill=tk.X)

        self.queue_panel = QueuePanel(main, on_select_job=self._on_select_job)
        self.queue_panel.frame.grid(row=0, column=1, sticky="nsew", padx=8)

        right_col = tk.Frame(main, bg="#eef1f5")
        right_col.grid(row=0, column=2, sticky="nsew", padx=(8, 0))
        right_col.grid_rowconfigure(0, weight=3)
        right_col.grid_rowconfigure(1, weight=2)
        right_col.grid_columnconfigure(0, weight=1)

        self.report_panel = ReportPanel(right_col, on_load_latest=self._load_latest_report, on_export=self._report_export_url)
        self.report_panel.frame.grid(row=0, column=0, sticky="nsew", pady=(0, 8))

        self.diagnostics_frame = tk.Frame(
            right_col,
            bg="#ffffff",
            bd=1,
            relief=tk.SOLID,
            highlightthickness=1,
            highlightbackground="#c7d1dc",
        )
        self.diagnostics_frame.grid(row=1, column=0, sticky="nsew")
        tk.Label(
            self.diagnostics_frame,
            text="Diagnostics",
            bg="#dbe7f5",
            fg="#12304a",
            anchor=tk.W,
            padx=10,
            pady=6,
            font=("Helvetica", 11, "bold"),
        ).pack(fill=tk.X)
        d_body = tk.Frame(self.diagnostics_frame, bg="#ffffff", padx=10, pady=10)
        d_body.pack(fill=tk.BOTH, expand=True)
        d_actions = tk.Frame(d_body, bg="#ffffff")
        d_actions.pack(fill=tk.X, pady=(0, 6))
        tk.Button(d_actions, text="Open Logs Folder", command=self._open_logs_folder).pack(side=tk.LEFT)
        self.diag_text = tk.Text(d_body, wrap=tk.WORD, relief=tk.SOLID, borderwidth=1, bg="#ffffff")
        self.diag_text.pack(fill=tk.BOTH, expand=True)
        self.diag_text.configure(state=tk.DISABLED)

        self.event_log = EventLogBar(self)
        self.event_log.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=(0, 10))

    def _initialize_visual_placeholders(self) -> None:
        self._log_event(f"Desktop app initialized ({UI_VERSION}).", dedupe_key="init", cooldown_sec=2.0)
        self.source_step.set_step_states(self.workflow.step)
        self.source_step.set_validation_message(
            "Step 1: choose source input. Step 2: validate before submission.",
            is_error=False,
        )
        self.source_step.set_next_step_hint(
            "Step 1: choose URL/DOI or upload a main PDF. Then click Validate."
        )
        self.queue_panel.set_details("No job selected. Submit analysis to populate queue details.")
        self.report_panel.clear()
        self.report_panel.set_report_status("Not Ready")
        self._update_diagnostics_panel()

    def _on_ui(self, func, *args, **kwargs) -> None:
        self._ui_queue.put((func, args, kwargs))

    def _drain_ui_queue(self) -> None:
        while True:
            try:
                func, args, kwargs = self._ui_queue.get_nowait()
            except queue.Empty:
                break
            try:
                func(*args, **kwargs)
            except Exception as exc:
                _append_log(f"UI queue callback failed: {exc}")
                _append_log(traceback.format_exc())
        self.after(80, self._drain_ui_queue)

    def _set_startup_state(self, state: str, note: str = "") -> None:
        self.startup_state = state
        if state == STATE_BACKEND_READY:
            badge_text = "Connection: Backend Ready"
            badge_color = "#1b5e20"
        elif state == STATE_BACKEND_STARTING:
            badge_text = "Connection: Starting"
            badge_color = "#0d47a1"
        elif state == STATE_DEGRADED:
            badge_text = "Connection: Degraded"
            badge_color = "#8b0000"
        elif state == STATE_STOPPING:
            badge_text = "Connection: Stopping"
            badge_color = "#6a1b9a"
        else:
            badge_text = "Connection: Booting"
            badge_color = "#30475f"

        self.connection_badge_var.set(badge_text)
        if note:
            self.connection_note_var.set(note)

        self.connection_badge_label.configure(fg=badge_color)
        if state == STATE_DEGRADED:
            self.source_step.set_next_step_hint("Backend disconnected. You can edit input and retry backend.")
        elif state == STATE_BACKEND_READY and self.workflow.can_submit():
            self.source_step.set_next_step_hint("Validation passed. Click Submit Analysis.")
        elif state == STATE_BACKEND_READY:
            self.source_step.set_next_step_hint("Step 2: click Validate. Submission is enabled after validation.")

        self.runtime_info.startup_state = state
        self._write_runtime_snapshot()
        self._update_diagnostics_panel()

    def _set_retry_deadline(self, seconds: int | None) -> None:
        if seconds is None:
            self.next_retry_due_at = None
            self.connection_retry_var.set("")
            return
        self.next_retry_due_at = time.time() + max(0, seconds)

    def _tick_connection_status(self) -> None:
        if self.next_retry_due_at is not None:
            remaining = max(0, int(self.next_retry_due_at - time.time()))
            if remaining <= 0:
                self.connection_retry_var.set("Retry: now")
            else:
                self.connection_retry_var.set(f"Retry in {remaining}s")
        else:
            self.connection_retry_var.set("")
        self.after(1000, self._tick_connection_status)

    def _log_event(self, message: str, dedupe_key: str | None = None, cooldown_sec: float = 0.0) -> None:
        now = time.time()
        if dedupe_key:
            last = self._event_dedupe.get(dedupe_key)
            if last is not None and now - last < cooldown_sec:
                return
            self._event_dedupe[dedupe_key] = now
        self.event_log.push(message)
        _append_log(message)

    def _update_diagnostics_panel(self) -> None:
        backend_pid = None
        if self.backend_proc and self.backend_proc.poll() is None:
            backend_pid = self.backend_proc.pid
        self.runtime_info.backend_pid = backend_pid
        self.runtime_info.last_start_error = self.last_start_error

        lines = [
            f"Startup state: {self.startup_state}",
            f"API base: {self.api_base}",
            f"UI python: {self.runtime_info.ui_python_path}",
            f"Backend python: {self.runtime_info.backend_python_path}",
            f"Desktop PID: {self.runtime_info.desktop_instance_pid}",
            f"Backend PID: {backend_pid if backend_pid else 'not running'}",
            f"Backend ready: {self.backend_ready}",
            f"Last start error: {self.last_start_error or 'none'}",
            f"Instance marker: {INSTANCE_FILE}",
            f"UI log: {LOG_PATH}",
            f"Backend log: {BACKEND_LOG_PATH}",
            f"Runtime snapshot: {DESKTOP_RUNTIME_PATH}",
        ]
        if self.backend_status:
            lines.append(f"Workers inflight: {self.backend_status.inflight}/{max(self.backend_status.worker_capacity, 1)}")
            lines.append(f"Processing paused: {self.backend_status.processing_paused}")
            lines.append(f"Stale jobs recovered: {self.backend_status.stale_jobs_recovered}")
            lines.append(f"Backend reported error: {self.backend_status.last_error or 'none'}")

        payload = "\n".join(lines)
        self.diag_text.configure(state=tk.NORMAL)
        self.diag_text.delete("1.0", tk.END)
        self.diag_text.insert("1.0", payload)
        self.diag_text.configure(state=tk.DISABLED)

        self._write_runtime_snapshot()

    def _write_runtime_snapshot(self) -> None:
        data = asdict(self.runtime_info)
        data["timestamp"] = now_ts()
        data["api_base"] = self.api_base
        data["backend_ready"] = self.backend_ready
        data["backend_log"] = str(BACKEND_LOG_PATH)
        data["ui_log"] = str(LOG_PATH)
        data["instance_file"] = str(INSTANCE_FILE)
        _write_json_atomic(DESKTOP_RUNTIME_PATH, data)

    def _open_logs_folder(self) -> None:
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            subprocess.Popen(["open", str(LOG_DIR)])
        except Exception as exc:
            messagebox.showerror("Open logs failed", str(exc))

    def _retry_backend_now(self) -> None:
        if self.backend_ready or self.backend_starting:
            self._log_event("Retry ignored: backend is already ready or starting.", dedupe_key="retry-ignore", cooldown_sec=1.0)
            return
        self._log_event("Manual retry requested.", dedupe_key="retry", cooldown_sec=1.0)
        self._start_backend_async()

    def _apply_disabled_state(self, *, disabled: bool) -> None:
        self.source_step.set_controls_enabled(True)
        self.source_step.set_submit_enabled(False if disabled else self.workflow.can_submit())
        self.pause_btn.config(state=tk.DISABLED if disabled else self.pause_btn.cget("state"))
        self.resume_btn.config(state=tk.DISABLED if disabled else self.resume_btn.cget("state"))
        self.clean_btn.config(state=tk.DISABLED if disabled else tk.NORMAL)
        self.recover_btn.config(state=tk.DISABLED if disabled else tk.NORMAL)
        self.report_panel.set_load_enabled(False if disabled else self.report_panel.load_btn.cget("state") == tk.NORMAL)
        if disabled:
            self.report_panel.set_export_enabled(False)

    def _set_processing_buttons(self, status: BackendStatus) -> None:
        if status.processing_paused:
            self.pause_btn.config(state=tk.DISABLED)
            self.resume_btn.config(state=tk.NORMAL)
        elif status.processing_running:
            self.pause_btn.config(state=tk.NORMAL)
            self.resume_btn.config(state=tk.DISABLED)
        else:
            self.pause_btn.config(state=tk.NORMAL)
            self.resume_btn.config(state=tk.NORMAL)
        self.clean_btn.config(state=tk.NORMAL)
        self.recover_btn.config(state=tk.NORMAL)

    def _preflight_checks(self) -> list[str]:
        warnings: list[str] = []
        data_dir = ROOT / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        probe = data_dir / ".desktop_write_probe"
        try:
            probe.write_text(now_ts())
            probe.unlink(missing_ok=True)
        except Exception:
            warnings.append(f"Data directory is not writable: {data_dir}")

        models_dir = ROOT / "models"
        model_files = list(models_dir.glob("*.gguf")) if models_dir.exists() else []
        if not model_files:
            warnings.append("No model GGUF files detected under models/.")

        py = _resolve_backend_python()
        self.runtime_info.backend_python_path = str(py)
        if not py.exists():
            warnings.append(f"Backend python runtime not found: {py}")
            return warnings

        if not _python_has_module(py, "uvicorn", env=_build_backend_env()):
            warnings.append(f"Selected backend python has no uvicorn: {py}")
            if py != Path(sys.executable):
                warnings.append(f"GUI is running on: {sys.executable}")

        return warnings

    def _recover_local_runtime(self) -> None:
        pids_path = RUN_DIR / "pids.json"
        if not pids_path.exists():
            return
        try:
            payload = json.loads(pids_path.read_text())
        except Exception:
            return

        for key in ["frontend_pgid", "backend_pgid"]:
            pgid = payload.get(key)
            if isinstance(pgid, int) and pgid > 0:
                try:
                    os.killpg(pgid, signal.SIGTERM)
                    time.sleep(0.2)
                    os.killpg(pgid, signal.SIGKILL)
                except Exception:
                    pass

        for key in ["frontend_pid", "backend_pid"]:
            pid = payload.get(key)
            if isinstance(pid, int) and pid > 0:
                try:
                    os.kill(pid, signal.SIGTERM)
                    time.sleep(0.2)
                    os.kill(pid, signal.SIGKILL)
                except Exception:
                    pass

        try:
            pids_path.unlink()
        except Exception:
            pass

    def _start_backend_async(self) -> None:
        if self.backend_starting:
            return
        self.backend_starting = True
        self._set_startup_state(STATE_BACKEND_STARTING, "Starting backend service and running readiness checks.")
        threading.Thread(target=self._start_backend_worker, daemon=True).start()

    def _start_backend_worker(self) -> None:
        try:
            self._on_ui(self._log_event, "Running startup preflight checks.", "preflight", 0.5)
            warnings = self._preflight_checks()
            for warning in warnings:
                self._on_ui(self._log_event, f"Warning: {warning}", f"warn-{warning}", 1.0)

            self._on_ui(self._log_event, "Recovering stale local runtime state.", "recover-local", 0.5)
            self._recover_local_runtime()

            attempts = 2
            for attempt in range(1, attempts + 1):
                if self._spawn_backend_once():
                    self.backend_ready = True
                    self.last_start_error = None
                    self._on_ui(self._on_backend_ready)
                    return
                self._on_ui(self._log_event, f"Backend startup attempt {attempt} failed.", f"start-attempt-{attempt}", 0.5)
                self._stop_backend_process(force=True)
                time.sleep(0.8)

            self.backend_ready = False
            self.last_start_error = self.last_start_error or "Backend failed to become ready"
            self._on_ui(self._on_backend_unavailable)
        finally:
            self.backend_starting = False
            self._on_ui(self._update_diagnostics_panel)

    def _spawn_backend_once(self) -> bool:
        py = _resolve_backend_python()
        env = _build_backend_env()
        cmd = [str(py), "-m", "uvicorn", "app.main:app", "--app-dir", "backend", "--port", str(self.backend_port)]
        try:
            BACKEND_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            self.backend_log_handle = BACKEND_LOG_PATH.open("a", buffering=1)
            self.backend_log_handle.write(f"\n---- {time.strftime('%Y-%m-%d %H:%M:%S')} backend start ----\n")
            self.backend_log_handle.write(f"cmd: {' '.join(cmd)}\n")
            self.backend_proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                env=env,
                stdout=self.backend_log_handle,
                stderr=self.backend_log_handle,
                start_new_session=True,
            )
        except Exception as exc:
            self.last_start_error = f"Backend launch failed: {exc}"
            self._on_ui(self._log_event, self.last_start_error, "launch-failed", 0.5)
            _append_log(traceback.format_exc())
            return False

        deadline = time.time() + 35
        while time.time() < deadline:
            if self.backend_proc and self.backend_proc.poll() is not None:
                rc = self.backend_proc.returncode
                self.last_start_error = f"Backend exited early (code {rc})"
                self._on_ui(
                    self._log_event,
                    f"{self.last_start_error}. See {BACKEND_LOG_PATH}.",
                    "backend-exit",
                    0.5,
                )
                return False
            try:
                status = self.api.get_status()
                self.backend_status = status
                self.runtime_info.backend_pid = self.backend_proc.pid if self.backend_proc else None
                return True
            except Exception:
                time.sleep(0.3)

        self.last_start_error = "Backend readiness timed out"
        return False

    def _on_backend_ready(self) -> None:
        self._set_retry_deadline(None)
        self._set_startup_state(STATE_BACKEND_READY, "All systems healthy. Validate input and submit analysis.")
        self._apply_disabled_state(disabled=False)
        self._log_event("Backend is ready.", dedupe_key="backend-ready", cooldown_sec=1.0)
        self.source_step.set_validation_message("Validate input before submission.", is_error=False)
        self.source_step.set_next_step_hint("Step 2: click Validate. Step 3 will unlock when validation passes.")

        try:
            result = self.api.processing_recover()
            self._log_event(
                (
                    f"Recovery complete: {result.get('recovered_jobs', 0)} jobs, "
                    f"cleaned {result.get('orphan_cleanup', {}).get('killed', 0)} workers."
                ),
                dedupe_key="recover-complete",
                cooldown_sec=1.0,
            )
        except Exception as exc:
            self._log_event(f"Recovery endpoint failed: {exc}", dedupe_key="recover-failed", cooldown_sec=2.0)

        self._update_diagnostics_panel()

    def _on_backend_unavailable(self) -> None:
        self.status_bar.set_backend_unavailable()
        self._apply_disabled_state(disabled=True)
        self._set_retry_deadline(3)
        detail = "Backend unavailable. Inputs remain editable; submit is blocked until reconnect."
        self._set_startup_state(STATE_DEGRADED, detail)
        self._log_event(
            f"Backend unavailable. Reconnect attempts will continue. Check {BACKEND_LOG_PATH}.",
            dedupe_key="backend-unavailable",
            cooldown_sec=2.0,
        )
        self.source_step.set_validation_message("Backend unavailable. Waiting for reconnect.", is_error=True)
        self.source_step.set_next_step_hint("You can still select files and edit source input while backend reconnects.")
        self._update_diagnostics_panel()

    def _poll_backend(self) -> None:
        try:
            status = self.api.get_status()
            self.backend_status = status
            self.backend_ready = True
            self.status_bar.update(status)
            self._set_processing_buttons(status)
            self._refresh_jobs()
            self._refresh_runtime_events()
            self._apply_disabled_state(disabled=False)
            if self.startup_state != STATE_BACKEND_READY:
                self._set_startup_state(STATE_BACKEND_READY, "All systems healthy. Validate input and submit analysis.")
            self._set_retry_deadline(None)

            running_jobs = any(row.status == "running" for row in self.jobs_by_id.values())
            self.last_poll_delay_ms = 1000 if running_jobs else 4000
        except Exception as exc:
            self.backend_ready = False
            self.status_bar.set_backend_unavailable()
            self._apply_disabled_state(disabled=True)
            self.last_poll_delay_ms = 3000
            self.last_start_error = str(exc)
            self._set_startup_state(STATE_DEGRADED, "Backend disconnected. Auto-reconnect is active.")
            self._set_retry_deadline(3)
            if self.backend_proc is None or (self.backend_proc and self.backend_proc.poll() is not None):
                self._start_backend_async()
        finally:
            self._update_diagnostics_panel()
            self.after(self.last_poll_delay_ms, self._poll_backend)

    def _refresh_jobs(self) -> None:
        rows = self.api.list_jobs(limit=200, offset=0)
        rows.sort(key=lambda item: item.updated_at, reverse=True)
        self.jobs_by_id = {row.job_id: row for row in rows}
        self.queue_panel.update_jobs(rows)

        current_ids = {row.job_id for row in rows}
        for stale_id in [jid for jid in self._job_status_cache if jid not in current_ids]:
            self._job_status_cache.pop(stale_id, None)

        for row in rows:
            old = self._job_status_cache.get(row.job_id)
            self._job_status_cache[row.job_id] = row.status
            if old == row.status:
                continue
            if row.status == "completed":
                self._log_event(
                    f"Job {row.job_id} completed. Report is ready.",
                    dedupe_key=f"job-{row.job_id}-completed",
                    cooldown_sec=120.0,
                )
            elif row.status == "failed":
                self._log_event(
                    f"Job {row.job_id} failed. {row.message or 'Check details in Active Work.'}",
                    dedupe_key=f"job-{row.job_id}-failed",
                    cooldown_sec=60.0,
                )

        if self.selected_job_id and self.selected_job_id in self.jobs_by_id:
            self._update_selected_job_details(self.jobs_by_id[self.selected_job_id])
        elif self.workflow.selected_job_id and self.workflow.selected_job_id in self.jobs_by_id:
            self.selected_job_id = self.workflow.selected_job_id
            self._update_selected_job_details(self.jobs_by_id[self.selected_job_id])
        else:
            self.report_panel.set_load_enabled(False)

    def _refresh_runtime_events(self) -> None:
        try:
            events = self.api.get_runtime_events(limit=8)
        except Exception:
            return
        if not events:
            return
        latest = events[-1]
        ts = str(latest.get("timestamp", ""))
        evt = str(latest.get("event", "event"))
        key = f"{evt}:{ts}"
        if key == self.last_runtime_event_key or not ts:
            return
        self.last_runtime_event_key = key
        self.last_runtime_event_ts = ts
        self._log_event(f"Runtime event: {evt}", dedupe_key=f"runtime-{key}", cooldown_sec=2.0)

    def _validate_workflow_input(self) -> None:
        payload = self.source_step.get_source_payload()
        result = validate_source_input(
            payload["source_mode"],
            url_text=payload["url"],
            doi_text=payload["doi"],
            main_file=payload["main_file"],
            supplement_files=payload["supp_files"],
        )
        self.workflow.set_source_mode(payload["source_mode"])
        self.workflow.set_validation(result)
        self.source_step.set_step_states(self.workflow.step)
        self.source_step.set_validation_message(result.message, is_error=not result.valid)
        self.source_step.set_submit_enabled(self.workflow.can_submit() and self.backend_ready)

        if result.valid:
            if self.backend_ready:
                self.source_step.set_next_step_hint("Step 3: click Submit Analysis to start processing.")
            else:
                self.source_step.set_next_step_hint(
                    "Validation passed. Waiting for backend reconnection before submission."
                )
            self._log_event("Input validated successfully.", dedupe_key="validate-success", cooldown_sec=0.5)
            if result.normalized_url:
                self.source_step.url_entry.delete(0, tk.END)
                self.source_step.url_entry.insert(0, result.normalized_url)
            if result.normalized_doi:
                self.source_step.doi_entry.delete(0, tk.END)
                self.source_step.doi_entry.insert(0, result.normalized_doi)
        else:
            self.source_step.set_next_step_hint("Resolve validation issue shown above, then click Validate again.")
            self._log_event("Input validation failed.", dedupe_key="validate-fail", cooldown_sec=0.5)

    def _submit_analysis(self) -> None:
        if not self.backend_ready:
            messagebox.showerror("Backend unavailable", "Backend is not ready yet.")
            return
        if not self.workflow.can_submit():
            messagebox.showerror("Validation required", "Please validate input before submitting.")
            return

        payload = self.source_step.get_source_payload()
        try:
            if payload["source_mode"] == "url":
                response = self.api.create_from_url(payload["url"], payload["doi"] or None, True)
            else:
                main_file = Path(payload["main_file"])
                supp_files = [Path(item) for item in payload["supp_files"]]
                response = self.api.create_from_upload(main_file, supp_files)
        except DesktopApiError as exc:
            self._log_event(f"Submission failed: {exc}", dedupe_key="submit-failed", cooldown_sec=0.5)
            messagebox.showerror("Submission failed", str(exc))
            return

        job_id = int(response["job_id"])
        document_id = int(response["document_id"])
        self.workflow.on_job_submitted(job_id, document_id)
        self.source_step.set_step_states(self.workflow.step)
        self.selected_job_id = job_id
        self.selected_document_id = document_id
        self.report_panel.set_report_status("Not Ready")
        self.report_panel.set_load_enabled(False)
        self.report_panel.set_export_enabled(False)
        self.source_step.set_next_step_hint(
            "Step 4: monitor job status in Active Work. Load report after status becomes completed."
        )
        self._log_event(f"Submitted analysis for document {document_id} (job {job_id}).")
        self._refresh_jobs()

    def _on_select_job(self, job_id: Optional[int]) -> None:
        self.selected_job_id = job_id
        if job_id is None:
            self.queue_panel.set_details("No job selected.")
            self.report_panel.set_load_enabled(False)
            return

        row = self.jobs_by_id.get(job_id)
        if not row:
            self.queue_panel.set_details("Selected job no longer available.")
            self.report_panel.set_load_enabled(False)
            return

        self._update_selected_job_details(row)

    def _update_selected_job_details(self, row: JobRow) -> None:
        detail = (
            f"Job ID: {row.job_id}\n"
            f"Document ID: {row.document_id}\n"
            f"Source: {row.source_kind}\n"
            f"Status: {row.status}\n"
            f"Progress: {int(row.progress * 100)}%\n"
            f"Updated: {row.updated_at}\n"
            f"Message: {row.message}\n"
        )
        self.queue_panel.set_details(detail)
        self.selected_job_status = row.status
        self.selected_document_id = row.document_id

        self.workflow.selected_job_id = row.job_id
        self.workflow.selected_document_id = row.document_id
        self.workflow.on_job_status(row.status)
        self.source_step.set_step_states(self.workflow.step)

        can_load = self.workflow.can_load_report(row.status)
        self.report_panel.set_load_enabled(can_load)
        self.report_panel.set_report_status("Ready" if can_load else "Not Ready")
        if row.status == "failed":
            self.report_panel.set_report_status("Failed")
            self.source_step.set_next_step_hint(
                "Selected job failed. Adjust source input and submit a new analysis run."
            )
        elif row.status == "completed":
            self.source_step.set_next_step_hint("Step 5: click Load Latest Report to review findings.")
        elif row.status == "running":
            self.source_step.set_next_step_hint("Analysis is running. Wait for completion, then load report.")
        else:
            self.source_step.set_next_step_hint("Job is queued. Monitor Active Work for status changes.")

    def _load_latest_report(self) -> None:
        if not self.selected_document_id:
            messagebox.showerror("No document selected", "Choose a completed job first.")
            return

        try:
            summary = self.api.get_report_summary(self.selected_document_id)
            full = self.api.get_report(self.selected_document_id)
        except Exception as exc:
            self._log_event(f"Failed to load report: {exc}", dedupe_key="load-report-failed", cooldown_sec=0.5)
            messagebox.showerror("Report load failed", str(exc))
            return

        self.last_report_export_url = summary.export_url
        self.report_panel.render_summary(summary, full_payload=full)
        self.report_panel.set_report_status("Ready" if summary.report_status == "ready" else "Not Ready")
        self.report_panel.set_export_enabled(summary.report_status == "ready")
        if summary.report_status == "ready":
            self.workflow.step = 5
        self.source_step.set_step_states(self.workflow.step)
        self.source_step.set_next_step_hint("Report loaded. Review modality sections and optionally export JSON.")
        self._log_event(f"Loaded report for document {self.selected_document_id}.", dedupe_key="load-report", cooldown_sec=0.5)

    def _report_export_url(self) -> Optional[str]:
        if not self.last_report_export_url:
            return None
        if self.last_report_export_url.startswith("http://") or self.last_report_export_url.startswith("https://"):
            return self.last_report_export_url
        return f"http://127.0.0.1:{self.backend_port}{self.last_report_export_url}"

    def _pause_processing(self) -> None:
        if not self.backend_ready:
            return
        try:
            self.api.processing_pause()
            self._log_event("Processing paused.", dedupe_key="pause", cooldown_sec=0.5)
        except Exception as exc:
            messagebox.showerror("Pause failed", str(exc))

    def _resume_processing(self) -> None:
        if not self.backend_ready:
            return
        try:
            self.api.processing_resume()
            self._log_event("Processing resumed.", dedupe_key="resume", cooldown_sec=0.5)
        except Exception as exc:
            messagebox.showerror("Resume failed", str(exc))

    def _clean_orphans(self) -> None:
        if not self.backend_ready:
            return
        try:
            result = self.api.processing_cleanup()
            self._log_event(f"Cleaned {result.get('killed', 0)} orphan workers.", dedupe_key="clean-orphans", cooldown_sec=0.5)
        except Exception as exc:
            messagebox.showerror("Cleanup failed", str(exc))

    def _recover_processing(self) -> None:
        if not self.backend_ready:
            return
        try:
            result = self.api.processing_recover()
            self._log_event(
                (
                    f"Recovered {result.get('recovered_jobs', 0)} jobs; "
                    f"cleaned {result.get('orphan_cleanup', {}).get('killed', 0)} workers."
                ),
                dedupe_key="recover-processing",
                cooldown_sec=0.5,
            )
        except Exception as exc:
            messagebox.showerror("Recover failed", str(exc))

    def _on_close(self) -> None:
        self._set_startup_state(STATE_STOPPING, "Stopping backend and cleaning runtime state.")
        self._log_event("Shutting down.", dedupe_key="shutdown", cooldown_sec=0.5)
        try:
            self.api.stop_app()
        except Exception:
            pass
        self._stop_backend_process(force=True)
        self._write_runtime_snapshot()
        self.destroy()

    def _stop_backend_process(self, *, force: bool) -> None:
        if not self.backend_proc or self.backend_proc.poll() is not None:
            self._close_backend_log_handle()
            return

        pid = self.backend_proc.pid
        pgid = None
        try:
            pgid = os.getpgid(pid)
        except Exception:
            pgid = None

        if pgid:
            try:
                os.killpg(pgid, signal.SIGTERM)
                time.sleep(0.4)
                if force:
                    os.killpg(pgid, signal.SIGKILL)
            except Exception:
                pass
            self._close_backend_log_handle()
            return

        try:
            self.backend_proc.terminate()
            time.sleep(0.3)
            if force:
                self.backend_proc.kill()
        except Exception:
            pass
        self._close_backend_log_handle()

    def _close_backend_log_handle(self) -> None:
        if self.backend_log_handle is None:
            return
        try:
            self.backend_log_handle.close()
        except Exception:
            pass
        self.backend_log_handle = None


def main() -> None:
    _append_log(f"Launching PaperEval Desktop {UI_VERSION} (guided layout).")
    app = PaperEvalDesktopApp()
    smoke_seconds = _env_int("PAPER_EVAL_SMOKE_SECONDS")
    if smoke_seconds and smoke_seconds > 0:
        _append_log(f"Smoke mode enabled for {smoke_seconds}s")
        app.after(smoke_seconds * 1000, app._on_close)
    app.mainloop()


def _append_log(message: str) -> None:
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a") as handle:
            handle.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
    except Exception:
        return


def _python_has_module(py: Path, module_name: str, env: dict[str, str] | None = None) -> bool:
    try:
        result = subprocess.run(
            [str(py), "-c", f"import {module_name}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
            timeout=5,
            check=False,
        )
        return result.returncode == 0
    except Exception:
        return False


def _write_json_atomic(path: Path, payload: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(path)
    except Exception:
        _append_log("Failed to write runtime snapshot")


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except Exception:
        return None
