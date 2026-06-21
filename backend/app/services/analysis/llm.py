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

import httpx

from app.core.config import settings
from app.services.local_gpu import ensure_local_metal_device_env, local_llama_runtime_blocked_reason
from app.services.analysis.openai_usage import (
    estimate_input_tokens,
    record_openai_result,
    reserve_openai_call,
    set_usage_context,
)

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
        load_seconds = {
            "text": round(float(_usage_durations.get("text_model_load_seconds", 0.0)), 4),
            "deep": round(float(_usage_durations.get("deep_model_load_seconds", 0.0)), 4),
            "vision": round(float(_usage_durations.get("vision_model_load_seconds", 0.0)), 4),
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
            "text_model_load_calls": int(_usage_counts.get("text_model_load_calls", 0)),
            "text_model_load_errors": int(_usage_counts.get("text_model_load_errors", 0)),
            "text_model_load_seconds": load_seconds["text"],
            "deep_calls": deep_calls,
            "deep_errors": int(_usage_counts.get("deep_errors", 0)),
            "deep_total_seconds": timing_seconds["deep"],
            "deep_avg_seconds": round(deep_seconds / deep_calls, 4) if deep_calls else 0.0,
            "deep_model_load_calls": int(_usage_counts.get("deep_model_load_calls", 0)),
            "deep_model_load_errors": int(_usage_counts.get("deep_model_load_errors", 0)),
            "deep_model_load_seconds": load_seconds["deep"],
            "vision_calls": vision_calls,
            "vision_errors": int(_usage_counts.get("vision_errors", 0)),
            "vision_total_seconds": timing_seconds["vision"],
            "vision_avg_seconds": round(vision_seconds / vision_calls, 4) if vision_calls else 0.0,
            "vision_model_load_calls": int(_usage_counts.get("vision_model_load_calls", 0)),
            "vision_model_load_errors": int(_usage_counts.get("vision_model_load_errors", 0)),
            "vision_model_load_seconds": load_seconds["vision"],
            "slowest_model": slowest_model,
            "slowest_seconds": slowest_seconds if slowest_model != "none" else 0.0,
            "openai_estimated_cost_usd": round(float(_usage_counts.get("openai_estimated_cost_usd", 0.0)), 8),
            "openai_input_tokens": int(_usage_counts.get("openai_input_tokens", 0)),
            "openai_cached_input_tokens": int(_usage_counts.get("openai_cached_input_tokens", 0)),
            "openai_output_tokens": int(_usage_counts.get("openai_output_tokens", 0)),
            "openai_reasoning_tokens": int(_usage_counts.get("openai_reasoning_tokens", 0)),
            "openai_total_tokens": int(_usage_counts.get("openai_total_tokens", 0)),
        }


def _record_usage(counter_key: str) -> None:
    with _usage_lock:
        _usage_counts[counter_key] += 1


def _record_duration(counter_key: str, elapsed_seconds: float) -> None:
    safe_elapsed = max(0.0, float(elapsed_seconds))
    with _usage_lock:
        _usage_durations[counter_key] += safe_elapsed


def _record_model_load(kind: str, elapsed_seconds: float, *, ok: bool) -> None:
    prefix = str(kind or "").strip().lower()
    if prefix not in {"text", "deep", "vision"}:
        return
    _record_usage(f"{prefix}_model_load_calls")
    if not ok:
        _record_usage(f"{prefix}_model_load_errors")
    _record_duration(f"{prefix}_model_load_seconds", elapsed_seconds)


def set_openai_usage_context(
    *,
    job_id: int | None = None,
    document_id: int | None = None,
    stage: str | None = None,
) -> None:
    set_usage_context(job_id=job_id, document_id=document_id, stage=stage)


def _record_openai_usage_totals(entry: dict[str, Any]) -> None:
    if entry.get("status") != "succeeded":
        return
    with _usage_lock:
        _usage_counts["openai_estimated_cost_usd"] += float(entry.get("estimated_cost_usd", 0.0) or 0.0)
        _usage_counts["openai_input_tokens"] += int(entry.get("input_tokens", 0) or 0)
        _usage_counts["openai_cached_input_tokens"] += int(entry.get("cached_input_tokens", 0) or 0)
        _usage_counts["openai_output_tokens"] += int(entry.get("output_tokens", 0) or 0)
        _usage_counts["openai_reasoning_tokens"] += int(entry.get("reasoning_tokens", 0) or 0)
        _usage_counts["openai_total_tokens"] += int(entry.get("total_tokens", 0) or 0)


def _use_openai_provider() -> bool:
    return settings.llm_provider_normalized == "openai"


def _openai_api_key() -> str:
    key = str(settings.resolved_openai_api_key or "").strip()
    if not key:
        raise RuntimeError("OpenAI provider is selected but OPENAI_API_KEY is not configured.")
    return key


def _wants_json_response(messages: list[dict[str, Any]]) -> bool:
    if not settings.openai_json_mode_enabled:
        return False
    text_parts: list[str] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            text_parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(str(item.get("text") or ""))
    combined = "\n".join(text_parts).lower()
    return "return json" in combined or "output json" in combined or "json only" in combined


def _openai_completion(
    *,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float,
    modality: str,
    max_tokens: int | None = None,
) -> str:
    api_mode = str(settings.openai_api_mode or "responses").strip().lower()
    if api_mode in {"chat", "chat_completions", "chat-completions"}:
        url = settings.openai_base_url.rstrip("/") + "/chat/completions"
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if isinstance(max_tokens, int) and max_tokens > 0:
            payload["max_completion_tokens"] = max_tokens
    else:
        instructions, responses_input = _messages_to_responses_payload(messages)
        url = settings.openai_base_url.rstrip("/") + "/responses"
        payload = {
            "model": model,
            "input": responses_input,
        }
        if instructions:
            payload["instructions"] = instructions
        if isinstance(max_tokens, int) and max_tokens > 0:
            payload["max_output_tokens"] = max_tokens
    if settings.openai_send_temperature:
        payload["temperature"] = temperature
    reasoning_effort = str(settings.openai_reasoning_effort or "").strip()
    if reasoning_effort:
        if api_mode in {"chat", "chat_completions", "chat-completions"}:
            payload["reasoning_effort"] = reasoning_effort
        else:
            payload["reasoning"] = {"effort": reasoning_effort}
    wants_json = _wants_json_response(messages)
    if wants_json:
        if api_mode in {"chat", "chat_completions", "chat-completions"}:
            payload["response_format"] = {"type": "json_object"}
        else:
            payload["input"] = _append_json_marker_to_responses_input(payload.get("input"))
            payload["text"] = {"format": {"type": "json_object"}}

    headers = {
        "Authorization": f"Bearer {_openai_api_key()}",
        "Content-Type": "application/json",
    }
    reservation_id = reserve_openai_call(
        model=model,
        modality=modality,
        max_output_tokens=max_tokens,
        estimated_input_tokens=estimate_input_tokens(messages),
    )
    try:
        response = httpx.post(
            url,
            headers=headers,
            json=payload,
            timeout=max(10, int(settings.openai_timeout_sec or 0)),
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = ""
        try:
            body = exc.response.json()
            if isinstance(body, dict):
                error = body.get("error")
                if isinstance(error, dict):
                    detail = str(error.get("message") or "")
        except Exception:
            detail = str(exc.response.text or "")[:200]
        message = f"OpenAI request failed with HTTP {exc.response.status_code}"
        if detail:
            message += f": {detail}"
        record_openai_result(
            reservation_id=reservation_id,
            model=model,
            modality=modality,
            status="failed",
            error=message,
        )
        raise RuntimeError(message) from exc
    except httpx.RequestError as exc:
        record_openai_result(
            reservation_id=reservation_id,
            model=model,
            modality=modality,
            status="failed",
            error=str(exc),
        )
        raise RuntimeError(f"OpenAI request failed: {exc}") from exc

    data = response.json()
    entry = record_openai_result(
        reservation_id=reservation_id,
        model=model,
        modality=modality,
        response_id=str(data.get("id") or ""),
        usage=data.get("usage") if isinstance(data.get("usage"), dict) else {},
    )
    _record_openai_usage_totals(entry)
    return _extract_openai_text(data, api_mode=api_mode)


def _messages_to_responses_payload(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    instructions: list[str] = []
    converted: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role") or "user")
        content = message.get("content")
        if role == "system":
            text = _content_to_text(content)
            if text:
                instructions.append(text)
            continue
        if isinstance(content, str):
            converted.append({"role": role, "content": [{"type": "input_text", "text": content}]})
            continue
        if not isinstance(content, list):
            converted.append({"role": role, "content": [{"type": "input_text", "text": str(content or "")}]})
            continue
        items: list[dict[str, Any]] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "")
            if item_type == "text":
                items.append({"type": "input_text", "text": str(item.get("text") or "")})
            elif item_type == "image_url":
                image_url = item.get("image_url")
                url = image_url.get("url") if isinstance(image_url, dict) else image_url
                if url:
                    items.append({"type": "input_image", "image_url": str(url), "detail": "auto"})
            else:
                items.append(item)
        converted.append({"role": role, "content": items})
    return "\n\n".join(instructions), converted


def _responses_input_mentions_json(responses_input: Any) -> bool:
    text_parts: list[str] = []
    for message in responses_input if isinstance(responses_input, list) else []:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            text_parts.append(content)
            continue
        for item in content if isinstance(content, list) else []:
            if isinstance(item, dict) and item.get("type") == "input_text":
                text_parts.append(str(item.get("text") or ""))
    return "json" in "\n".join(text_parts).lower()


def _append_json_marker_to_responses_input(responses_input: Any) -> list[dict[str, Any]]:
    converted = list(responses_input) if isinstance(responses_input, list) else []
    marker = "Return JSON."
    for message in reversed(converted):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, list):
            if any(
                isinstance(item, dict)
                and item.get("type") == "input_text"
                and str(item.get("text") or "").strip().lower() == marker.lower()
                for item in content
            ):
                return converted
            content.append({"type": "input_text", "text": marker})
            return converted
        if isinstance(content, str):
            message["content"] = [
                {"type": "input_text", "text": content},
                {"type": "input_text", "text": marker},
            ]
            return converted
    converted.append({"role": "user", "content": [{"type": "input_text", "text": marker}]})
    return converted


def _extract_openai_text(data: dict[str, Any], *, api_mode: str) -> str:
    if api_mode in {"chat", "chat_completions", "chat-completions"}:
        try:
            content = data["choices"][0]["message"]["content"]
        except Exception as exc:
            raise RuntimeError("OpenAI response did not include chat message content.") from exc
        return _content_to_text(content)

    output_text = data.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()
    text_parts: list[str] = []
    for item in data.get("output", []) if isinstance(data.get("output"), list) else []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []) if isinstance(item.get("content"), list) else []:
            if not isinstance(content, dict):
                continue
            if content.get("type") == "output_text":
                text_parts.append(str(content.get("text") or ""))
            elif content.get("type") == "refusal":
                text_parts.append(str(content.get("refusal") or ""))
    text = "\n".join(part for part in text_parts if part).strip()
    if not text:
        status = str(data.get("status") or "").strip()
        incomplete_reason = ""
        incomplete_details = data.get("incomplete_details")
        if isinstance(incomplete_details, dict):
            incomplete_reason = str(incomplete_details.get("reason") or "").strip()
        usage = data.get("usage")
        reasoning_tokens = 0
        output_tokens = 0
        if isinstance(usage, dict):
            output_tokens = _safe_int(usage.get("output_tokens"))
            output_details = usage.get("output_tokens_details")
            if isinstance(output_details, dict):
                reasoning_tokens = _safe_int(output_details.get("reasoning_tokens"))
        if reasoning_tokens and output_tokens and reasoning_tokens >= output_tokens:
            raise RuntimeError(
                "OpenAI response used the full output budget for reasoning and produced no visible text. "
                "Use a lower OPENAI_REASONING_EFFORT or increase the stage max output tokens."
            )
        if status or incomplete_reason:
            raise RuntimeError(
                "OpenAI response did not include output text"
                + (f" (status={status})" if status else "")
                + (f" (reason={incomplete_reason})" if incomplete_reason else "")
                + "."
            )
        raise RuntimeError("OpenAI response did not include output text.")
    return text


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _content_to_text(content: Any) -> str:
    if isinstance(content, list):
        text_parts = [
            str(item.get("text") or "")
            for item in content
            if isinstance(item, dict) and item.get("type") in {"text", "output_text"}
        ]
        return "\n".join(part for part in text_parts if part).strip()
    return str(content or "").strip()


@lru_cache(maxsize=1)
def _load_text_model():
    started = monotonic()
    ok = False
    try:
        model = _load_text_model_uncached()
        ok = True
        return model
    finally:
        _record_model_load("text", monotonic() - started, ok=ok)


def _load_text_model_uncached():
    ensure_local_metal_device_env()
    blocked_reason = local_llama_runtime_blocked_reason()
    if blocked_reason:
        raise RuntimeError(blocked_reason)
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
    started = monotonic()
    ok = False
    try:
        model = _load_deep_model_uncached()
        ok = True
        return model
    finally:
        _record_model_load("deep", monotonic() - started, ok=ok)


def _load_deep_model_uncached():
    ensure_local_metal_device_env()
    blocked_reason = local_llama_runtime_blocked_reason()
    if blocked_reason:
        raise RuntimeError(blocked_reason)
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
    started = monotonic()
    ok = False
    try:
        model = _load_vision_model_uncached()
        ok = True
        return model
    finally:
        _record_model_load("vision", monotonic() - started, ok=ok)


def _load_vision_model_uncached():
    ensure_local_metal_device_env()
    blocked_reason = local_llama_runtime_blocked_reason()
    if blocked_reason:
        raise RuntimeError(blocked_reason)
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


def _local_max_tokens(default_value: int, local_value: int) -> int:
    if _use_openai_provider():
        return default_value
    try:
        value = int(local_value or 0)
    except Exception:
        value = 0
    return value if value > 0 else default_value


def chat_text_fast(prompt: str, system: str | None = None, temperature: float = 0.2) -> str:
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    started = monotonic()
    _record_usage("text_calls")
    try:
        if _use_openai_provider():
            return _openai_completion(
                model=settings.openai_text_model,
                messages=messages,
                temperature=temperature,
                modality="text",
                max_tokens=settings.llm_text_max_tokens,
            )
        llm = _load_text_model()
        return _chat_completion(
            llm,
            messages,
            temperature,
            max_tokens=_local_max_tokens(settings.llm_text_max_tokens, settings.llm_local_text_max_tokens),
        )
    except Exception:
        _record_usage("text_errors")
        raise
    finally:
        _record_duration("text_total_seconds", monotonic() - started)


def chat_text_deep(prompt: str, system: str | None = None, temperature: float = 0.2) -> str:
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    started = monotonic()
    _record_usage("deep_calls")
    try:
        if _use_openai_provider():
            return _openai_completion(
                model=settings.openai_deep_model,
                messages=messages,
                temperature=temperature,
                modality="deep",
                max_tokens=settings.llm_deep_max_tokens,
            )
        llm = _load_deep_model()
        return _chat_completion(
            llm,
            messages,
            temperature,
            max_tokens=_local_max_tokens(settings.llm_deep_max_tokens, settings.llm_local_deep_max_tokens),
        )
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
        if _use_openai_provider():
            return _openai_completion(
                model=settings.openai_vision_model,
                messages=messages,
                temperature=temperature,
                modality="vision",
                max_tokens=settings.llm_vision_max_tokens,
            )
        llm = _load_vision_model()
        return _chat_completion(
            llm,
            messages,
            temperature,
            max_tokens=_local_max_tokens(settings.llm_vision_max_tokens, settings.llm_local_vision_max_tokens),
        )
    except Exception:
        _record_usage("vision_errors")
        raise
    finally:
        _record_duration("vision_total_seconds", monotonic() - started)
