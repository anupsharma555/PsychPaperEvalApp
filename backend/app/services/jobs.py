from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
from typing import Optional

from sqlmodel import Session, select

from app.core.config import settings
from app.db.models import Job, JobStatus
from app.db.session import engine
from app.services.pipeline import run_pipeline


class JobRunner:
    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._lock = threading.Lock()
        self._executor: Optional[ProcessPoolExecutor] = None
        self._futures: dict[Future[None], int] = {}
        self._max_inflight = max(1, settings.analysis_workers)
        self._stale_jobs_recovered = _recover_stale_running_jobs()
        self._last_recovery_at = datetime.utcnow().isoformat() if self._stale_jobs_recovered else None
        self._last_orphan_cleanup: dict | None = None
        self._last_error: str | None = None
        if settings.analysis_use_process_pool:
            try:
                self._executor = ProcessPoolExecutor(max_workers=self._max_inflight)
            except Exception as exc:
                # Some constrained environments (sandboxed desktop contexts) can
                # block process semaphore initialization. Fall back gracefully.
                self._executor = None
                self._max_inflight = 1
                self._last_error = f"Process pool disabled, fallback to inline worker: {exc}"
        else:
            self._max_inflight = 1
        if settings.analysis_cleanup_orphans:
            self._last_orphan_cleanup = _cleanup_orphan_workers()

    def start(self) -> None:
        self._stop.clear()
        self._paused.clear()
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._paused.clear()
        if self._thread:
            self._thread.join(timeout=3)
        if self._executor:
            try:
                self._executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                try:
                    self._executor.shutdown(wait=False)
                except Exception:
                    pass
        self._last_orphan_cleanup = _cleanup_orphan_workers()
        _clear_worker_pid_file()
        with self._lock:
            self._futures.clear()
        self._last_error = None

    def pause(self) -> None:
        self._paused.set()

    def resume(self) -> None:
        if not self._thread or not self._thread.is_alive():
            self.start()
        else:
            self._paused.clear()

    def status(self) -> dict:
        with self._lock:
            inflight = len(self._futures)
        return {
            "running": bool(self._thread and self._thread.is_alive()),
            "paused": self._paused.is_set(),
            "inflight": inflight,
            "worker_capacity": self._max_inflight,
            "stale_jobs_recovered": self._stale_jobs_recovered,
            "last_recovery_at": self._last_recovery_at,
            "orphan_cleanup_last_run": self._last_orphan_cleanup,
            "executor_enabled": bool(self._executor),
            "last_error": self._last_error,
        }

    def cleanup_orphans(self) -> dict:
        self._last_orphan_cleanup = _cleanup_orphan_workers()
        return self._last_orphan_cleanup

    def recover_state(self) -> dict:
        recovered = _recover_stale_running_jobs()
        cleanup = self.cleanup_orphans()
        if recovered:
            self._stale_jobs_recovered += recovered
            self._last_recovery_at = datetime.utcnow().isoformat()
        return {
            "recovered_jobs": recovered,
            "total_recovered_jobs": self._stale_jobs_recovered,
            "last_recovery_at": self._last_recovery_at,
            "orphan_cleanup": cleanup,
        }

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._collect_finished_futures()
                if self._paused.is_set():
                    time.sleep(0.4)
                    continue

                capacity = self._available_slots()
                if capacity <= 0:
                    time.sleep(0.3)
                    continue
                job_ids = self._next_job_ids(capacity)
                if not job_ids:
                    time.sleep(0.8)
                    continue

                if self._executor:
                    submit_failed = False
                    for job_id in job_ids:
                        try:
                            future = self._executor.submit(run_pipeline, job_id)
                            with self._lock:
                                self._futures[future] = job_id
                        except Exception as exc:
                            self._last_error = f"Worker submit failed for job {job_id}: {exc}"
                            _mark_job_queued(job_id, f"Recovered after worker submit failure: {exc}")
                            self._reset_executor(self._last_error)
                            submit_failed = True
                            break
                    if submit_failed:
                        time.sleep(0.2)
                        continue
                    _record_worker_pids(self._executor)
                else:
                    for job_id in job_ids:
                        run_pipeline(job_id)

                time.sleep(0.2)
            except Exception as exc:
                # Keep runner alive even if pool internals raise unexpectedly.
                self._last_error = f"Job runner loop error: {exc}"
                if _is_pool_failure(exc):
                    self._reset_executor(self._last_error)
                recovered = _recover_stale_running_jobs()
                if recovered:
                    self._stale_jobs_recovered += recovered
                    self._last_recovery_at = datetime.utcnow().isoformat()
                time.sleep(0.5)

    def _available_slots(self) -> int:
        if not self._executor:
            return 1
        with self._lock:
            return max(0, self._max_inflight - len(self._futures))

    def _collect_finished_futures(self) -> None:
        if not self._executor:
            return
        done: list[tuple[Future[None], int]] = []
        with self._lock:
            for future, job_id in list(self._futures.items()):
                if future.done():
                    done.append((future, job_id))
                    self._futures.pop(future, None)
        for future, job_id in done:
            try:
                future.result()
            except Exception as exc:
                if _is_pool_failure(exc):
                    self._last_error = f"Worker pool crashed while processing job {job_id}: {exc}"
                    _mark_job_queued(job_id, "Recovered after worker pool crash")
                    self._reset_executor(self._last_error)
                    continue
                self._last_error = f"Worker crashed for job {job_id}: {exc}"
                _mark_job_failed(job_id, f"Worker crashed: {exc}")

    def _reset_executor(self, reason: str) -> None:
        executor = self._executor
        if executor is None:
            return
        self._executor = None
        self._max_inflight = 1
        self._last_error = reason
        in_flight_job_ids: list[int] = []
        with self._lock:
            in_flight_job_ids = list(self._futures.values())
            self._futures.clear()
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            try:
                executor.shutdown(wait=False)
            except Exception:
                pass
        _clear_worker_pid_file()
        for job_id in in_flight_job_ids:
            _mark_job_queued(job_id, "Recovered after worker pool reset")

    def _next_job_ids(self, limit: int) -> list[int]:
        if limit <= 0:
            return []
        with Session(engine) as session:
            stmt = (
                select(Job)
                .where(Job.status == JobStatus.queued)
                .order_by(Job.created_at)
                .limit(limit)
            )
            jobs = session.exec(stmt).all()
            if not jobs:
                return []
            ids: list[int] = []
            now = datetime.utcnow()
            for job in jobs:
                job.status = JobStatus.running
                job.updated_at = now
                session.add(job)
                if job.id is not None:
                    ids.append(job.id)
            session.commit()
            return ids


def enqueue_job(session: Session, document_id: int) -> Job:
    job = Job(document_id=document_id, status=JobStatus.queued, progress=0.0)
    session.add(job)
    session.commit()
    session.refresh(job)
    return job


def _recover_stale_running_jobs() -> int:
    recovered = 0
    with Session(engine) as session:
        jobs = session.exec(select(Job).where(Job.status == JobStatus.running)).all()
        if not jobs:
            return 0
        now = datetime.utcnow()
        for job in jobs:
            job.status = JobStatus.queued
            job.message = "Recovered after unclean shutdown"
            job.updated_at = now
            session.add(job)
            recovered += 1
        session.commit()
    return recovered


def _mark_job_failed(job_id: int, message: str) -> None:
    with Session(engine) as session:
        job = session.get(Job, job_id)
        if not job:
            return
        job.status = JobStatus.failed
        job.message = message
        job.updated_at = datetime.utcnow()
        session.add(job)
        session.commit()


def _mark_job_queued(job_id: int, message: str) -> None:
    with Session(engine) as session:
        job = session.get(Job, job_id)
        if not job:
            return
        job.status = JobStatus.queued
        job.message = message
        job.updated_at = datetime.utcnow()
        session.add(job)
        session.commit()


def _is_pool_failure(exc: Exception) -> bool:
    label = str(exc.__class__.__name__).lower()
    text = str(exc).lower()
    if "brokenprocesspool" in label:
        return True
    if "process pool" in text and "terminated abruptly" in text:
        return True
    if "executor" in text and "shutdown" in text:
        return True
    return False


def _worker_pid_file() -> Path:
    root = os.environ.get("PAPER_EVAL_ROOT")
    if root:
        return Path(root) / ".run" / "workers.json"
    return Path(".run") / "workers.json"


def _record_worker_pids(executor: ProcessPoolExecutor) -> None:
    try:
        processes = getattr(executor, "_processes", {})
        pids = sorted([pid for pid in processes.keys() if isinstance(pid, int)])
    except Exception:
        return
    if not pids:
        return
    workers = []
    for pid in pids:
        cmd = _pid_command(pid)
        workers.append(
            {
                "pid": pid,
                "cmd_fingerprint": _fingerprint(cmd),
                "cmd": cmd,
            }
        )
    payload = {
        "workers": workers,
        "runner_pid": os.getpid(),
        "runner_cmd_fingerprint": _fingerprint(_pid_command(os.getpid())),
        "ts": time.time(),
    }
    path = _worker_pid_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(payload))
        tmp.replace(path)
    except Exception:
        return


def _clear_worker_pid_file() -> None:
    path = _worker_pid_file()
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


def _cleanup_orphan_workers() -> dict:
    path = _worker_pid_file()
    if not path.exists():
        return {"checked": 0, "killed": 0, "skipped_mismatch": 0}
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return {"checked": 0, "killed": 0, "skipped_mismatch": 0}

    workers: list[dict] = payload.get("workers", [])
    if not workers and isinstance(payload.get("pids"), list):
        workers = [{"pid": pid} for pid in payload.get("pids", [])]

    checked = 0
    killed = 0
    skipped_mismatch = 0
    for worker in workers:
        pid = worker.get("pid")
        if not isinstance(pid, int):
            continue
        checked += 1
        if not _pid_alive(pid):
            continue
        cmd = _pid_command(pid)
        if not cmd:
            continue
        expected_fp = worker.get("cmd_fingerprint")
        current_fp = _fingerprint(cmd)
        if expected_fp and expected_fp != current_fp:
            skipped_mismatch += 1
            continue
        if "python" not in cmd or "spawn_main" not in cmd:
            skipped_mismatch += 1
            continue
        _terminate_pid(pid)
        killed += 1
    _clear_worker_pid_file()
    return {"checked": checked, "killed": killed, "skipped_mismatch": skipped_mismatch}


def _pid_command(pid: int) -> str:
    try:
        out = subprocess.check_output(["ps", "-p", str(pid), "-o", "command="], text=True)
        return out.strip()
    except Exception:
        return ""


def _fingerprint(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()


def _terminate_pid(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except Exception:
        return
    time.sleep(0.3)
    if _pid_alive(pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:
            pass


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


job_runner = JobRunner()
