from __future__ import annotations

from contextvars import ContextVar
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.core.config import settings

try:
    import fcntl
except Exception:  # pragma: no cover - Windows fallback, harmless on macOS/Linux.
    fcntl = None  # type: ignore[assignment]


_job_id: ContextVar[int | None] = ContextVar("openai_usage_job_id", default=None)
_document_id: ContextVar[int | None] = ContextVar("openai_usage_document_id", default=None)
_stage: ContextVar[str] = ContextVar("openai_usage_stage", default="unknown")


class OpenAIBudgetExceeded(RuntimeError):
    pass


def set_usage_context(
    *,
    job_id: int | None = None,
    document_id: int | None = None,
    stage: str | None = None,
) -> None:
    if job_id is not None:
        _job_id.set(int(job_id))
    if document_id is not None:
        _document_id.set(int(document_id))
    if stage:
        _stage.set(str(stage))


def current_usage_context() -> dict[str, Any]:
    return {
        "job_id": _job_id.get(),
        "document_id": _document_id.get(),
        "stage": _stage.get(),
    }


def estimate_input_tokens(messages: list[dict[str, Any]]) -> int:
    text_chars = 0
    image_count = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            text_chars += len(content)
            continue
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "")
            if item_type in {"text", "input_text"}:
                text_chars += len(str(item.get("text") or ""))
            elif item_type in {"image_url", "input_image"}:
                image_count += 1
    text_tokens = int(text_chars / 4) + 1 if text_chars else 0
    image_tokens = image_count * max(0, int(settings.openai_estimated_image_input_tokens or 0))
    return max(1, text_tokens + image_tokens)


def reserve_openai_call(
    *,
    model: str,
    modality: str,
    max_output_tokens: int | None,
    estimated_input_tokens: int,
) -> str | None:
    if not _guardrails_enabled():
        return None

    output_cap = max(0, int(max_output_tokens or 0))
    estimate = estimate_cost_usd(
        model=model,
        input_tokens=max(0, int(estimated_input_tokens or 0)),
        cached_input_tokens=0,
        output_tokens=output_cap,
    )
    entry = {
        "event_type": "reservation",
        "reservation_id": uuid4().hex,
        "created_at": _now_iso(),
        "created_day": _today_key(),
        "model": str(model or ""),
        "modality": str(modality or "unknown"),
        "max_output_tokens": output_cap,
        "estimated_input_tokens": max(0, int(estimated_input_tokens or 0)),
        "estimated_cost_usd": round(estimate, 8),
        **current_usage_context(),
    }

    path = _ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _locked_file(path) as handle:
        rows = _read_rows_from_handle(handle)
        summary = _summarize_rows(rows, job_id=entry.get("job_id"), day=entry["created_day"])
        _enforce_limits(summary, entry)
        handle.seek(0, 2)
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
        handle.flush()
    return str(entry["reservation_id"])


def record_openai_result(
    *,
    reservation_id: str | None,
    model: str,
    modality: str,
    response_id: str = "",
    usage: dict[str, Any] | None = None,
    status: str = "succeeded",
    error: str = "",
) -> dict[str, Any]:
    parsed = parse_openai_usage(usage or {})
    cost = estimate_cost_usd(model=model, **parsed)
    entry = {
        "event_type": "actual",
        "reservation_id": reservation_id or "",
        "created_at": _now_iso(),
        "created_day": _today_key(),
        "model": str(model or ""),
        "modality": str(modality or "unknown"),
        "response_id": str(response_id or ""),
        "status": str(status or "succeeded"),
        "error": str(error or "")[:500],
        "estimated_cost_usd": round(cost, 8),
        **parsed,
        **current_usage_context(),
    }
    if _guardrails_enabled():
        path = _ledger_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with _locked_file(path) as handle:
            handle.seek(0, 2)
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
            handle.flush()
    return entry


def mark_unmatched_openai_reservations_failed(
    *,
    job_id: int | None,
    document_id: int | None = None,
    stage: str | None = None,
    error: str = "",
) -> int:
    if not _guardrails_enabled() or job_id is None:
        return 0

    path = _ledger_path()
    if not path.exists():
        return 0

    marked = 0
    with _locked_file(path) as handle:
        rows = _read_rows_from_handle(handle)
        actual_reservations = {
            str(row.get("reservation_id") or "")
            for row in rows
            if row.get("event_type") == "actual"
        }
        markers: list[dict[str, Any]] = []
        for row in rows:
            if row.get("event_type") != "reservation":
                continue
            reservation_id = str(row.get("reservation_id") or "")
            if not reservation_id or reservation_id in actual_reservations:
                continue
            if row.get("job_id") != job_id:
                continue
            if document_id is not None and row.get("document_id") != document_id:
                continue
            if stage and str(row.get("stage") or "") != str(stage):
                continue
            markers.append(
                {
                    "event_type": "actual",
                    "reservation_id": reservation_id,
                    "created_at": _now_iso(),
                    "created_day": _today_key(),
                    "model": str(row.get("model") or ""),
                    "modality": str(row.get("modality") or "unknown"),
                    "response_id": "",
                    "status": "failed",
                    "error": str(error or "reservation abandoned")[:500],
                    "estimated_cost_usd": 0.0,
                    "input_tokens": 0,
                    "cached_input_tokens": 0,
                    "output_tokens": 0,
                    "reasoning_tokens": 0,
                    "total_tokens": 0,
                    "job_id": row.get("job_id"),
                    "document_id": row.get("document_id"),
                    "stage": row.get("stage"),
                }
            )

        if markers:
            handle.seek(0, 2)
            for marker in markers:
                handle.write(json.dumps(marker, sort_keys=True) + "\n")
            handle.flush()
            marked = len(markers)

    return marked


def parse_openai_usage(usage: dict[str, Any]) -> dict[str, int]:
    input_tokens = _int_value(usage.get("input_tokens"), usage.get("prompt_tokens"))
    output_tokens = _int_value(usage.get("output_tokens"), usage.get("completion_tokens"))
    total_tokens = _int_value(usage.get("total_tokens"), input_tokens + output_tokens)

    input_details = usage.get("input_tokens_details")
    if not isinstance(input_details, dict):
        input_details = usage.get("prompt_tokens_details")
    if not isinstance(input_details, dict):
        input_details = {}

    output_details = usage.get("output_tokens_details")
    if not isinstance(output_details, dict):
        output_details = usage.get("completion_tokens_details")
    if not isinstance(output_details, dict):
        output_details = {}

    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": _int_value(input_details.get("cached_tokens")),
        "output_tokens": output_tokens,
        "reasoning_tokens": _int_value(output_details.get("reasoning_tokens")),
        "total_tokens": total_tokens,
    }


def estimate_cost_usd(
    *,
    model: str,
    input_tokens: int,
    cached_input_tokens: int,
    output_tokens: int,
    reasoning_tokens: int = 0,
    total_tokens: int = 0,
) -> float:
    prices = _prices_for_model(model)
    cached = min(max(0, int(cached_input_tokens or 0)), max(0, int(input_tokens or 0)))
    uncached = max(0, int(input_tokens or 0) - cached)
    output = max(0, int(output_tokens or 0))
    return (
        uncached * prices["input"]
        + cached * prices["cached_input"]
        + output * prices["output"]
    ) / 1_000_000.0


def summarize_openai_usage(*, job_id: int | None = None, document_id: int | None = None) -> dict[str, Any]:
    if not _ledger_path().exists():
        return _empty_summary()
    rows = _read_rows(_ledger_path())
    filtered = []
    for row in rows:
        if job_id is not None and row.get("job_id") != job_id:
            continue
        if document_id is not None and row.get("document_id") != document_id:
            continue
        filtered.append(row)
    return _summarize_actuals(filtered)


def _guardrails_enabled() -> bool:
    return bool(settings.llm_provider_normalized == "openai" and settings.openai_usage_guardrails_enabled)


def _ledger_path() -> Path:
    path = settings.openai_usage_log_path
    if path.is_absolute():
        return path
    return settings.data_dir.parent / path if str(path).startswith("data/") else settings.data_dir / path


def _prices_for_model(model: str) -> dict[str, float]:
    key = str(model or "").strip().lower()
    if key.startswith("gpt-5-mini"):
        return {"input": 0.25, "cached_input": 0.025, "output": 2.00}
    if key.startswith("gpt-5-nano"):
        return {"input": 0.05, "cached_input": 0.005, "output": 0.40}
    if key.startswith("gpt-5.5"):
        return {"input": 5.00, "cached_input": 0.50, "output": 30.00}
    if key.startswith("gpt-5.4-mini"):
        return {"input": 0.75, "cached_input": 0.075, "output": 4.50}
    if key.startswith("gpt-5.4-nano"):
        return {"input": 0.20, "cached_input": 0.02, "output": 1.25}
    if key.startswith("gpt-5.4"):
        return {"input": 2.50, "cached_input": 0.25, "output": 15.00}
    if key.startswith("gpt-4.1-mini"):
        return {"input": 0.40, "cached_input": 0.10, "output": 1.60}
    if key.startswith("gpt-4.1"):
        return {"input": 2.00, "cached_input": 0.50, "output": 8.00}
    return {
        "input": float(settings.openai_cost_fallback_input_per_million or 0.0),
        "cached_input": float(settings.openai_cost_fallback_cached_input_per_million or 0.0),
        "output": float(settings.openai_cost_fallback_output_per_million or 0.0),
    }


def _enforce_limits(summary: dict[str, Any], reservation: dict[str, Any]) -> None:
    next_cost = float(reservation.get("estimated_cost_usd", 0.0) or 0.0)
    next_output = int(reservation.get("max_output_tokens", 0) or 0)
    run_cap = float(settings.openai_max_cost_per_run_usd or 0.0)
    day_cap = float(settings.openai_max_cost_per_day_usd or 0.0)
    call_cap = int(settings.openai_max_calls_per_run or 0)
    output_cap = int(settings.openai_max_output_tokens_per_run or 0)

    if run_cap > 0 and float(summary["run_cost_usd"]) + next_cost > run_cap:
        raise OpenAIBudgetExceeded(
            f"OpenAI run budget would be exceeded: ${summary['run_cost_usd']:.4f} + "
            f"${next_cost:.4f} > ${run_cap:.4f}."
        )
    if day_cap > 0 and float(summary["day_cost_usd"]) + next_cost > day_cap:
        raise OpenAIBudgetExceeded(
            f"OpenAI daily budget would be exceeded: ${summary['day_cost_usd']:.4f} + "
            f"${next_cost:.4f} > ${day_cap:.4f}."
        )
    if call_cap > 0 and int(summary["run_calls"]) + 1 > call_cap:
        raise OpenAIBudgetExceeded(f"OpenAI run call cap would be exceeded: {summary['run_calls']} + 1 > {call_cap}.")
    if output_cap > 0 and int(summary["run_output_tokens_reserved"]) + next_output > output_cap:
        raise OpenAIBudgetExceeded(
            "OpenAI run output-token cap would be exceeded: "
            f"{summary['run_output_tokens_reserved']} + {next_output} > {output_cap}."
        )


def _summarize_rows(rows: list[dict[str, Any]], *, job_id: int | None, day: str) -> dict[str, Any]:
    actual_reservations = {str(row.get("reservation_id") or "") for row in rows if row.get("event_type") == "actual"}
    run_cost = 0.0
    day_cost = 0.0
    run_calls = 0
    run_output_reserved = 0
    for row in rows:
        event_type = row.get("event_type")
        row_job = row.get("job_id")
        row_day = str(row.get("created_day") or "")
        cost = float(row.get("estimated_cost_usd", 0.0) or 0.0)
        if event_type == "reservation" and str(row.get("reservation_id") or "") in actual_reservations:
            continue
        if event_type not in {"reservation", "actual"}:
            continue
        if row_day == day:
            day_cost += cost
        if job_id is not None and row_job == job_id:
            run_cost += cost
            run_calls += 1
            run_output_reserved += int(row.get("max_output_tokens", row.get("output_tokens", 0)) or 0)
    return {
        "run_cost_usd": run_cost,
        "day_cost_usd": day_cost,
        "run_calls": run_calls,
        "run_output_tokens_reserved": run_output_reserved,
    }


def _summarize_actuals(rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals = _empty_summary()
    by_stage: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("event_type") != "actual" or row.get("status") != "succeeded":
            continue
        stage = str(row.get("stage") or row.get("modality") or "unknown")
        stage_totals = by_stage.setdefault(stage, _empty_usage_totals())
        _add_usage_row(totals, row)
        _add_usage_row(stage_totals, row)
    totals["by_stage"] = by_stage
    return totals


def _empty_summary() -> dict[str, Any]:
    payload = _empty_usage_totals()
    payload["by_stage"] = {}
    return payload


def _empty_usage_totals() -> dict[str, Any]:
    return {
        "calls": 0,
        "estimated_cost_usd": 0.0,
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
    }


def _add_usage_row(total: dict[str, Any], row: dict[str, Any]) -> None:
    total["calls"] = int(total.get("calls", 0) or 0) + 1
    total["estimated_cost_usd"] = round(
        float(total.get("estimated_cost_usd", 0.0) or 0.0) + float(row.get("estimated_cost_usd", 0.0) or 0.0),
        8,
    )
    for key in ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens", "total_tokens"):
        total[key] = int(total.get(key, 0) or 0) + int(row.get(key, 0) or 0)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with _locked_file(path) as handle:
        return _read_rows_from_handle(handle)


def _read_rows_from_handle(handle: Any) -> list[dict[str, Any]]:
    handle.seek(0)
    rows = []
    for line in handle:
        try:
            row = json.loads(line)
        except Exception:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


class _locked_file:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle: Any = None

    def __enter__(self) -> Any:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+", encoding="utf-8")
        if fcntl is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)
        return self.handle

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.handle is None:
            return
        if fcntl is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()


def _today_key() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _int_value(*values: Any) -> int:
    for value in values:
        try:
            return max(0, int(value or 0))
        except Exception:
            continue
    return 0
