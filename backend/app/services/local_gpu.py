from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from app.core.config import ROOT_DIR, settings
from app.services.runtime import log_runtime_event

STATUS_SCHEMA_VERSION = 1
FALLBACK_ENV = "PAPEREVAL_LOCAL_GPU_FALLBACK"


def _status_path() -> Path:
    return ROOT_DIR / ".run" / "local_gpu_status.json"


def _provider_is_local() -> bool:
    return settings.llm_provider_normalized == "local"


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _model_signature() -> dict[str, Any]:
    model_path = settings.resolved_llm_text_model_path
    stat_payload: dict[str, Any] = {}
    try:
        stat = model_path.stat()
        stat_payload = {"mtime_ns": int(stat.st_mtime_ns), "size": int(stat.st_size)}
    except Exception:
        stat_payload = {"mtime_ns": None, "size": None}
    llama_version = "unavailable"
    llama_libs: list[dict[str, Any]] = []
    try:
        import llama_cpp

        llama_version = str(getattr(llama_cpp, "__version__", "unknown"))
        package_dir = Path(getattr(llama_cpp, "__file__", "")).resolve().parent
        lib_dir = package_dir / "lib"
        for name in ("libllama.dylib", "libggml-metal.dylib", "libggml-metal.0.dylib"):
            lib_path = lib_dir / name
            if not lib_path.exists():
                continue
            try:
                stat = lib_path.stat()
                llama_libs.append(
                    {
                        "name": name,
                        "mtime_ns": int(stat.st_mtime_ns),
                        "size": int(stat.st_size),
                    }
                )
            except Exception:
                llama_libs.append({"name": name, "mtime_ns": None, "size": None})
    except Exception:
        pass
    return {
        "schema_version": STATUS_SCHEMA_VERSION,
        "provider": settings.llm_provider_normalized,
        "text_model_path": str(model_path),
        "text_model": stat_payload,
        "n_gpu_layers": int(settings.llm_n_gpu_layers),
        "ggml_metal_devices": str(settings.ggml_metal_devices or ""),
        "n_ctx": int(settings.llm_n_ctx),
        "n_batch": int(settings.llm_n_batch),
        "python": sys.executable,
        "llama_cpp_version": llama_version,
        "llama_cpp_libs": llama_libs,
    }


def _read_status_file() -> dict[str, Any]:
    path = _status_path()
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_status_file(payload: dict[str, Any]) -> None:
    path = _status_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def _signature_matches(payload: dict[str, Any], signature: dict[str, Any]) -> bool:
    return payload.get("signature") == signature


def _apply_cpu_fallback(payload: dict[str, Any]) -> None:
    settings.llm_n_gpu_layers = 0
    os.environ["LLM_N_GPU_LAYERS"] = "0"
    os.environ[FALLBACK_ENV] = "1"
    payload["effective_n_gpu_layers"] = 0
    payload["fallback_active"] = True


def ensure_local_metal_device_env() -> str:
    if not _provider_is_local():
        return str(os.environ.get("GGML_METAL_DEVICES") or "")
    configured = str(settings.ggml_metal_devices or "").strip()
    if configured:
        os.environ["GGML_METAL_DEVICES"] = configured
    return str(os.environ.get("GGML_METAL_DEVICES") or "")


def _native_crash_result(result: dict[str, Any]) -> bool:
    returncode = result.get("returncode")
    if isinstance(returncode, int) and returncode < 0:
        return True
    reason = str(result.get("reason") or "").lower()
    return "segmentation" in reason or "sigsegv" in reason or reason.startswith("returncode:-")


def _llama_runtime_links_metal() -> bool:
    try:
        import llama_cpp
    except Exception:
        return False
    package_dir = Path(getattr(llama_cpp, "__file__", "")).resolve().parent
    libllama = package_dir / "lib" / "libllama.dylib"
    if not libllama.exists():
        return False
    try:
        result = subprocess.run(
            ["otool", "-L", str(libllama)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return (package_dir / "lib" / "libggml-metal.dylib").exists()
    return "libggml-metal" in str(result.stdout or "")


def local_llama_runtime_blocked_reason() -> str:
    payload = _read_status_file()
    if not payload.get("fallback_active"):
        return ""
    reason = str(payload.get("fallback_reason") or "local_gpu_fallback")
    explicit_block = not bool(payload.get("main_process_load_safe", True))
    legacy_native_crash = "main_process_load_safe" not in payload and reason.startswith("returncode:-")
    if not explicit_block and not (legacy_native_crash and _llama_runtime_links_metal()):
        return ""
    return (
        "Local llama.cpp runtime is blocked because the bundled Metal backend crashed "
        f"during the guarded model-load smoke check ({reason}). Reinstall or rebuild "
        "llama-cpp-python with a CPU-only fallback or a fixed Metal backend before "
        "running local model inference."
    )
    return ""


def local_gpu_runtime_status() -> dict[str, Any]:
    preferred_layers = int(settings.llm_n_gpu_layers)
    payload = _read_status_file()
    fallback_active = bool(payload.get("fallback_active")) or _truthy(os.environ.get(FALLBACK_ENV))
    effective_layers = int(payload.get("effective_n_gpu_layers", preferred_layers) or 0)

    if not _provider_is_local():
        mode = "openai"
    elif fallback_active:
        mode = "cpu_fallback"
        effective_layers = 0
    elif preferred_layers <= 0:
        mode = "cpu"
        effective_layers = 0
    elif payload.get("smoke_status") == "passed":
        mode = "gpu_active"
    else:
        mode = "gpu_preferred"

    return {
        "local_gpu_mode": mode,
        "local_gpu_preferred_layers": preferred_layers,
        "local_gpu_effective_layers": effective_layers,
        "local_gpu_metal_devices": str(os.environ.get("GGML_METAL_DEVICES") or settings.ggml_metal_devices or ""),
        "local_gpu_smoke_status": str(payload.get("smoke_status") or "not_run"),
        "local_gpu_fallback_reason": str(payload.get("fallback_reason") or ""),
        "local_gpu_status": payload,
    }


def ensure_local_gpu_ready(*, reason: str = "analysis_start") -> dict[str, Any]:
    if not _provider_is_local():
        return local_gpu_runtime_status()
    if not bool(settings.local_gpu_smoke_enabled):
        metal_devices = ensure_local_metal_device_env()
        payload = {
            "signature": _model_signature(),
            "smoke_status": "disabled",
            "effective_n_gpu_layers": int(settings.llm_n_gpu_layers),
            "ggml_metal_devices": metal_devices,
            "fallback_active": False,
            "fallback_reason": "",
        }
        _write_status_file(payload)
        return local_gpu_runtime_status()
    if int(settings.llm_n_gpu_layers) <= 0:
        payload = {
            "signature": _model_signature(),
            "smoke_status": "skipped",
            "effective_n_gpu_layers": 0,
            "ggml_metal_devices": str(os.environ.get("GGML_METAL_DEVICES") or ""),
            "fallback_active": False,
            "fallback_reason": "gpu_layers_disabled",
        }
        _write_status_file(payload)
        return local_gpu_runtime_status()

    signature = _model_signature()
    metal_devices = ensure_local_metal_device_env()
    cached = _read_status_file()
    if _signature_matches(cached, signature):
        if cached.get("fallback_active"):
            _apply_cpu_fallback(cached)
        return local_gpu_runtime_status()

    result = _run_gpu_smoke_subprocess(signature)
    payload = {
        "signature": signature,
        "smoke_status": "passed" if result["ok"] else "failed",
        "effective_n_gpu_layers": int(settings.llm_n_gpu_layers) if result["ok"] else 0,
        "ggml_metal_devices": metal_devices,
        "fallback_active": not result["ok"],
        "fallback_reason": "" if result["ok"] else str(result.get("reason") or "gpu_smoke_failed"),
        "main_process_load_safe": bool(result["ok"] or not (_native_crash_result(result) and _llama_runtime_links_metal())),
        "returncode": result.get("returncode"),
        "stderr_tail": result.get("stderr_tail", ""),
        "reason": reason,
    }
    if not result["ok"]:
        _apply_cpu_fallback(payload)
    _write_status_file(payload)
    log_runtime_event(
        "local_gpu_cpu_fallback" if payload["fallback_active"] else "local_gpu_smoke_completed",
        {
            "status": payload["smoke_status"],
            "mode": "cpu_fallback" if payload["fallback_active"] else "gpu_active",
            "fallback_reason": payload["fallback_reason"],
            "n_gpu_layers": payload["effective_n_gpu_layers"],
        },
    )
    return local_gpu_runtime_status()


def _run_gpu_smoke_subprocess(signature: dict[str, Any]) -> dict[str, Any]:
    env = dict(os.environ)
    env["PAPER_EVAL_ROOT"] = str(ROOT_DIR)
    env["LLM_PROVIDER"] = "local"
    metal_devices = str(settings.ggml_metal_devices or "").strip()
    if metal_devices:
        env["GGML_METAL_DEVICES"] = metal_devices
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in [str(ROOT_DIR / "backend"), env.get("PYTHONPATH", "")] if part
    )
    code = """
from __future__ import annotations
import inspect
from app.core.config import Settings
s = Settings()
if s.llm_provider_normalized != "local":
    raise SystemExit("provider_not_local")
if int(s.llm_n_gpu_layers) <= 0:
    raise SystemExit(0)
model_path = s.resolved_llm_text_model_path
if not model_path.exists():
    raise SystemExit(f"missing_model:{model_path}")
from llama_cpp import Llama
kwargs = {
    "model_path": str(model_path),
    "n_ctx": s.llm_n_ctx,
    "n_threads": s.llm_n_threads,
    "n_batch": s.llm_n_batch,
    "n_gpu_layers": s.llm_n_gpu_layers,
    "verbose": False,
}
sig = inspect.signature(Llama.__init__)
if "chat_format" in sig.parameters and s.llm_text_chat_format:
    kwargs["chat_format"] = s.llm_text_chat_format
llm = Llama(**kwargs)
del llm
"""
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(ROOT_DIR),
            env=env,
            capture_output=True,
            text=True,
            timeout=int(settings.local_gpu_smoke_timeout_sec),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "reason": "timeout",
            "returncode": None,
            "stderr_tail": str(exc.stderr or "")[-1000:],
            "signature": signature,
        }
    except Exception as exc:
        return {
            "ok": False,
            "reason": f"smoke_exception:{type(exc).__name__}",
            "returncode": None,
            "stderr_tail": str(exc)[-1000:],
            "signature": signature,
        }
    return {
        "ok": completed.returncode == 0,
        "reason": "ok" if completed.returncode == 0 else f"returncode:{completed.returncode}",
        "returncode": completed.returncode,
        "stderr_tail": str(completed.stderr or "")[-1000:],
        "signature": signature,
    }
