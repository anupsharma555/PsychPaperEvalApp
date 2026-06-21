from __future__ import annotations

import os

from app.services import local_gpu


def _configure_local_gpu(monkeypatch, tmp_path):
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"gguf")
    status_path = tmp_path / "local_gpu_status.json"
    monkeypatch.setattr(local_gpu, "_status_path", lambda: status_path)
    monkeypatch.setattr(local_gpu.settings, "llm_provider", "local")
    monkeypatch.setattr(local_gpu.settings, "llm_text_model_path", model_path)
    monkeypatch.setattr(local_gpu.settings, "llm_n_gpu_layers", 999)
    monkeypatch.setattr(local_gpu.settings, "ggml_metal_devices", "0")
    monkeypatch.setattr(local_gpu.settings, "llm_n_ctx", 8192)
    monkeypatch.setattr(local_gpu.settings, "llm_n_batch", 512)
    monkeypatch.setattr(local_gpu.settings, "local_gpu_smoke_enabled", True)
    monkeypatch.delenv("LLM_N_GPU_LAYERS", raising=False)
    monkeypatch.delenv("GGML_METAL_DEVICES", raising=False)
    monkeypatch.delenv(local_gpu.FALLBACK_ENV, raising=False)
    return status_path


def test_local_gpu_smoke_success_keeps_gpu(monkeypatch, tmp_path) -> None:
    _configure_local_gpu(monkeypatch, tmp_path)
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        local_gpu,
        "_run_gpu_smoke_subprocess",
        lambda signature: {"ok": True, "returncode": 0, "signature": signature},
    )
    monkeypatch.setattr(local_gpu, "log_runtime_event", lambda event, details=None: events.append((event, details or {})))

    status = local_gpu.ensure_local_gpu_ready(reason="test")

    assert status["local_gpu_mode"] == "gpu_active"
    assert status["local_gpu_effective_layers"] == 999
    assert status["local_gpu_metal_devices"] == "0"
    assert os.environ.get("LLM_N_GPU_LAYERS") is None
    assert os.environ["GGML_METAL_DEVICES"] == "0"
    assert events[0][0] == "local_gpu_smoke_completed"


def test_local_gpu_smoke_crash_falls_back_to_cpu(monkeypatch, tmp_path) -> None:
    _configure_local_gpu(monkeypatch, tmp_path)
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(local_gpu, "_llama_runtime_links_metal", lambda: True)
    monkeypatch.setattr(
        local_gpu,
        "_run_gpu_smoke_subprocess",
        lambda signature: {
            "ok": False,
            "reason": "returncode:-11",
            "returncode": -11,
            "stderr_tail": "",
            "signature": signature,
        },
    )
    monkeypatch.setattr(local_gpu, "log_runtime_event", lambda event, details=None: events.append((event, details or {})))

    status = local_gpu.ensure_local_gpu_ready(reason="test")

    assert status["local_gpu_mode"] == "cpu_fallback"
    assert status["local_gpu_effective_layers"] == 0
    assert local_gpu.settings.llm_n_gpu_layers == 0
    assert os.environ["LLM_N_GPU_LAYERS"] == "0"
    assert os.environ["GGML_METAL_DEVICES"] == "0"
    assert os.environ[local_gpu.FALLBACK_ENV] == "1"
    assert "Metal backend crashed" in local_gpu.local_llama_runtime_blocked_reason()
    assert events[0][0] == "local_gpu_cpu_fallback"
    assert events[0][1]["fallback_reason"] == "returncode:-11"


def test_local_gpu_non_native_failure_allows_cpu_fallback_load(monkeypatch, tmp_path) -> None:
    _configure_local_gpu(monkeypatch, tmp_path)
    monkeypatch.setattr(local_gpu, "_llama_runtime_links_metal", lambda: True)
    monkeypatch.setattr(
        local_gpu,
        "_run_gpu_smoke_subprocess",
        lambda signature: {
            "ok": False,
            "reason": "missing_model",
            "returncode": 1,
            "stderr_tail": "",
            "signature": signature,
        },
    )
    monkeypatch.setattr(local_gpu, "log_runtime_event", lambda *_args, **_kwargs: None)

    status = local_gpu.ensure_local_gpu_ready(reason="test")

    assert status["local_gpu_mode"] == "cpu_fallback"
    assert local_gpu.local_llama_runtime_blocked_reason() == ""


def test_local_gpu_legacy_native_crash_status_blocks_main_load(monkeypatch, tmp_path) -> None:
    status_path = _configure_local_gpu(monkeypatch, tmp_path)
    monkeypatch.setattr(local_gpu, "_llama_runtime_links_metal", lambda: True)
    status_path.write_text(
        '{"fallback_active": true, "fallback_reason": "returncode:-11", "effective_n_gpu_layers": 0}'
    )

    assert "Metal backend crashed" in local_gpu.local_llama_runtime_blocked_reason()


def test_local_gpu_metal_device_setting_is_applied(monkeypatch, tmp_path) -> None:
    _configure_local_gpu(monkeypatch, tmp_path)
    monkeypatch.setattr(local_gpu.settings, "ggml_metal_devices", "Apple M4 Max")

    assert local_gpu.ensure_local_metal_device_env() == "Apple M4 Max"
    assert os.environ["GGML_METAL_DEVICES"] == "Apple M4 Max"
