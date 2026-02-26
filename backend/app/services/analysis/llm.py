from __future__ import annotations

import base64
from collections import Counter
import inspect
import mimetypes
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from threading import Lock
from time import monotonic
from typing import Any, Iterable

from app.core.config import settings

_usage_lock = Lock()
_usage_counts: Counter[str] = Counter()
_usage_durations: Counter[str] = Counter()


def reset_model_usage_counters() -> None:
    with _usage_lock:
        _usage_counts.clear()
        _usage_durations.clear()


def snapshot_model_usage_counters() -> dict[str, int | float | str]:
    with _usage_lock:
        text_calls = int(_usage_counts.get("text_calls", 0))
        deep_calls = int(_usage_counts.get("deep_calls", 0))
        vision_calls = int(_usage_counts.get("vision_calls", 0))
        text_seconds = float(_usage_durations.get("text_total_seconds", 0.0))
        deep_seconds = float(_usage_durations.get("deep_total_seconds", 0.0))
        vision_seconds = float(_usage_durations.get("vision_total_seconds", 0.0))
        timing_seconds = {
            "text": round(text_seconds, 4),
            "deep": round(deep_seconds, 4),
            "vision": round(vision_seconds, 4),
        }
        slowest_model = max(timing_seconds, key=timing_seconds.get)
        slowest_seconds = float(timing_seconds[slowest_model])
        if slowest_seconds <= 0.0:
            slowest_model = "none"
        return {
            "text_calls": text_calls,
            "text_errors": int(_usage_counts.get("text_errors", 0)),
            "text_total_seconds": timing_seconds["text"],
            "text_avg_seconds": round(text_seconds / text_calls, 4) if text_calls else 0.0,
            "deep_calls": deep_calls,
            "deep_errors": int(_usage_counts.get("deep_errors", 0)),
            "deep_total_seconds": timing_seconds["deep"],
            "deep_avg_seconds": round(deep_seconds / deep_calls, 4) if deep_calls else 0.0,
            "vision_calls": vision_calls,
            "vision_errors": int(_usage_counts.get("vision_errors", 0)),
            "vision_total_seconds": timing_seconds["vision"],
            "vision_avg_seconds": round(vision_seconds / vision_calls, 4) if vision_calls else 0.0,
            "slowest_model": slowest_model,
            "slowest_seconds": slowest_seconds if slowest_model != "none" else 0.0,
        }


def _record_usage(counter_key: str) -> None:
    with _usage_lock:
        _usage_counts[counter_key] += 1


def _record_duration(counter_key: str, elapsed_seconds: float) -> None:
    safe_elapsed = max(0.0, float(elapsed_seconds))
    with _usage_lock:
        _usage_durations[counter_key] += safe_elapsed


@lru_cache(maxsize=1)
def _load_text_model():
    model_path = settings.resolved_llm_text_model_path
    if not model_path.exists():
        raise RuntimeError(f"Text model file not found: {model_path}")
    try:
        from llama_cpp import Llama
    except Exception as exc:
        raise RuntimeError("llama-cpp-python is required for model inference") from exc

    llm_kwargs: dict[str, Any] = {
        "model_path": str(model_path),
        "n_ctx": settings.llm_n_ctx,
        "n_threads": settings.llm_n_threads,
        "n_batch": settings.llm_n_batch,
        "n_gpu_layers": settings.llm_n_gpu_layers,
        "verbose": False,
    }
    sig = inspect.signature(Llama.__init__)
    if "chat_format" in sig.parameters and settings.llm_text_chat_format:
        llm_kwargs["chat_format"] = settings.llm_text_chat_format
    return Llama(**llm_kwargs)


@lru_cache(maxsize=1)
def _load_deep_model():
    model_path = settings.resolved_llm_deep_model_path
    if not model_path.exists():
        raise RuntimeError(f"Deep model file not found: {model_path}")
    try:
        from llama_cpp import Llama
    except Exception as exc:
        raise RuntimeError("llama-cpp-python is required for model inference") from exc

    llm_kwargs: dict[str, Any] = {
        "model_path": str(model_path),
        "n_ctx": settings.llm_n_ctx,
        "n_threads": settings.llm_n_threads,
        "n_batch": settings.llm_n_batch,
        "n_gpu_layers": settings.llm_n_gpu_layers,
        "verbose": False,
    }
    sig = inspect.signature(Llama.__init__)
    if "chat_format" in sig.parameters and settings.llm_deep_chat_format:
        llm_kwargs["chat_format"] = settings.llm_deep_chat_format
    return Llama(**llm_kwargs)


@lru_cache(maxsize=1)
def _load_vision_model():
    model_path = settings.resolved_llm_vision_model_path
    mmproj_path = settings.resolved_llm_vision_mmproj_path
    if not model_path.exists():
        raise RuntimeError(f"Vision model file not found: {model_path}")
    if not mmproj_path.exists():
        raise RuntimeError(f"Vision MMProj file not found: {mmproj_path}")
    try:
        from llama_cpp import Llama
    except Exception as exc:
        raise RuntimeError("llama-cpp-python is required for model inference") from exc

    llm_kwargs: dict[str, Any] = {
        "model_path": str(model_path),
        "n_ctx": settings.llm_n_ctx,
        "n_threads": settings.llm_n_threads,
        "n_batch": settings.llm_n_batch,
        "n_gpu_layers": settings.llm_n_gpu_layers,
        "verbose": False,
    }
    chat_handler = _build_vision_chat_handler(mmproj_path)
    if chat_handler is not None:
        llm_kwargs["chat_handler"] = chat_handler
    else:
        sig = inspect.signature(Llama.__init__)
        if "chat_format" in sig.parameters and settings.llm_vision_chat_format:
            llm_kwargs["chat_format"] = settings.llm_vision_chat_format
    return Llama(**llm_kwargs)


def _build_vision_chat_handler(mmproj_path: Path) -> Any | None:
    try:
        from llama_cpp import llama_chat_format as chat_format_mod
    except Exception:
        return None

    chat_format = str(settings.llm_vision_chat_format or "").strip().lower()
    candidates = _vision_handler_candidates(chat_format)
    for handler_name in candidates:
        handler_cls = getattr(chat_format_mod, handler_name, None)
        if handler_cls is None:
            continue
        kwargs = _vision_handler_kwargs(handler_cls, mmproj_path)
        try:
            return handler_cls(**kwargs)
        except Exception:
            continue
    return None


def _vision_handler_candidates(chat_format: str) -> list[str]:
    if "qwen3" in chat_format:
        return ["Qwen3VLChatHandler", "Qwen25VLChatHandler", "Qwen2VLChatHandler"]
    if "qwen2.5" in chat_format or "qwen25" in chat_format:
        return ["Qwen25VLChatHandler", "Qwen2VLChatHandler", "Qwen3VLChatHandler"]
    if "qwen2" in chat_format:
        return ["Qwen2VLChatHandler", "Qwen25VLChatHandler", "Qwen3VLChatHandler"]
    return ["Qwen3VLChatHandler", "Qwen25VLChatHandler", "Qwen2VLChatHandler"]


def _vision_handler_kwargs(handler_cls: Any, mmproj_path: Path) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    handler_sig = inspect.signature(handler_cls.__init__)
    if "clip_model_path" in handler_sig.parameters:
        kwargs["clip_model_path"] = str(mmproj_path)
    elif "mmproj_path" in handler_sig.parameters:
        kwargs["mmproj_path"] = str(mmproj_path)
    elif "mmproj" in handler_sig.parameters:
        kwargs["mmproj"] = str(mmproj_path)
    return kwargs


def _image_to_data_uri(image_path: str) -> str:
    path = Path(image_path)
    try:
        from PIL import Image
    except Exception:
        Image = None
    if Image is None:
        mime, _ = mimetypes.guess_type(path.name)
        if not mime:
            mime = "image/png"
        data = path.read_bytes()
        b64 = base64.b64encode(data).decode("utf-8")
        return f"data:{mime};base64,{b64}"

    image = Image.open(path).convert("RGB")
    width, height = image.size
    max_dim = settings.llm_image_max_dim
    max_pixels = settings.llm_image_max_pixels

    scale = 1.0
    if max_dim and max(width, height) > max_dim:
        scale = min(scale, max_dim / float(max(width, height)))
    if max_pixels and width * height > max_pixels:
        scale = min(scale, (max_pixels / float(width * height)) ** 0.5)

    if scale < 1.0:
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        image = image.resize(new_size, Image.LANCZOS)

    fmt = settings.llm_image_format.lower()
    if fmt not in {"jpeg", "png", "webp"}:
        fmt = "jpeg"
    buf = BytesIO()
    save_kwargs = {}
    if fmt == "jpeg":
        save_kwargs["quality"] = settings.llm_image_quality
        save_kwargs["optimize"] = True
    image.save(buf, format=fmt.upper(), **save_kwargs)
    data = buf.getvalue()
    mime = f"image/{fmt}"
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _chat_completion(
    llm: Any,
    messages: list[dict[str, Any]],
    temperature: float,
    *,
    max_tokens: int | None = None,
) -> str:
    kwargs: dict[str, Any] = {
        "messages": messages,
        "temperature": temperature,
    }
    if isinstance(max_tokens, int) and max_tokens > 0:
        kwargs["max_tokens"] = max_tokens
    try:
        response = llm.create_chat_completion(**kwargs)
    except TypeError:
        kwargs.pop("max_tokens", None)
        response = llm.create_chat_completion(**kwargs)
    return response["choices"][0]["message"]["content"]


def chat_text_fast(prompt: str, system: str | None = None, temperature: float = 0.2) -> str:
    llm = _load_text_model()
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    started = monotonic()
    _record_usage("text_calls")
    try:
        return _chat_completion(llm, messages, temperature, max_tokens=settings.llm_text_max_tokens)
    except Exception:
        _record_usage("text_errors")
        raise
    finally:
        _record_duration("text_total_seconds", monotonic() - started)


def chat_text_deep(prompt: str, system: str | None = None, temperature: float = 0.2) -> str:
    llm = _load_deep_model()
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    started = monotonic()
    _record_usage("deep_calls")
    try:
        return _chat_completion(llm, messages, temperature, max_tokens=settings.llm_deep_max_tokens)
    except Exception:
        _record_usage("deep_errors")
        raise
    finally:
        _record_duration("deep_total_seconds", monotonic() - started)


def chat_text(prompt: str, system: str | None = None, temperature: float = 0.2) -> str:
    # Backward-compatible alias used by scripts/tests.
    return chat_text_fast(prompt, system=system, temperature=temperature)


def chat_with_images(
    prompt: str,
    image_paths: Iterable[str],
    system: str | None = None,
    temperature: float = 0.2,
) -> str:
    llm = _load_vision_model()
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for path in image_paths:
        content.append({"type": "image_url", "image_url": {"url": _image_to_data_uri(path)}})
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})
    started = monotonic()
    _record_usage("vision_calls")
    try:
        return _chat_completion(llm, messages, temperature, max_tokens=settings.llm_vision_max_tokens)
    except Exception:
        _record_usage("vision_errors")
        raise
    finally:
        _record_duration("vision_total_seconds", monotonic() - started)
