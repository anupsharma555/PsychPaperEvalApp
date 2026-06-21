from __future__ import annotations

from typing import Any

LATENCY_PROFILE_VERSION = 1
ANALYSIS_STAGE_ORDER = ("text", "table", "figure", "supplement", "reconcile", "synthesis", "store")
MODEL_STAGE_KEYS = ("text", "deep", "vision")


def build_latency_profile(
    diagnostics: dict[str, Any] | None,
    *,
    document_id: int | None = None,
) -> dict[str, Any]:
    diag = diagnostics if isinstance(diagnostics, dict) else {}
    analysis_timing = _dict(diag.get("analysis_timing"))
    parse_timing = _dict(diag.get("parse_timing"))
    model_usage = _dict(diag.get("model_usage"))
    stage_model_usage = _dict(diag.get("stage_model_usage"))
    prompt_budget = _dict(diag.get("prompt_budget_diagnostics"))

    stages: list[dict[str, Any]] = []
    parse_seconds = _number(parse_timing.get("parse_total_seconds"))
    if parse_seconds is not None:
        parse_metadata = _parse_metadata(parse_timing)
        stages.append(
            _stage_row(
                "parse",
                duration_seconds=parse_seconds,
                timing_source="parse_timing",
                started_at=parse_timing.get("started_at"),
                ended_at=parse_timing.get("ended_at"),
                metadata=parse_metadata,
            )
        )

    for stage in ANALYSIS_STAGE_ORDER:
        duration = _number(analysis_timing.get(stage))
        if duration is None:
            continue
        stages.append(
            _stage_row(
                stage,
                duration_seconds=duration,
                timing_source="analysis_timing",
                model_usage=_dict(stage_model_usage.get(stage)),
                prompt_block=_prompt_modality_block(prompt_budget, stage),
            )
        )

    stage_by_name = {str(stage["stage"]): stage for stage in stages}
    for event in _list(diag.get("analysis_timeline")):
        if not isinstance(event, dict):
            continue
        name = str(event.get("stage", "") or "").strip()
        if not name:
            continue
        if name in stage_by_name:
            _merge_stage_metadata(stage_by_name[name], _dict(event.get("metadata")))
            continue
        duration = _number(event.get("duration_seconds"))
        if duration is None:
            continue
        row = _stage_row(
            name,
            duration_seconds=duration,
            timing_source="analysis_timeline",
            started_at=event.get("started_at"),
            ended_at=event.get("ended_at"),
            metadata=_dict(event.get("metadata")),
        )
        stages.append(row)
        stage_by_name[name] = row

    for event in _list(diag.get("pipeline_timeline")):
        if not isinstance(event, dict):
            continue
        name = _pipeline_step_stage_name(str(event.get("step", "") or ""))
        if not name or name in stage_by_name:
            continue
        duration = _number(event.get("duration_seconds"))
        if duration is None:
            continue
        row = _stage_row(
            name,
            duration_seconds=duration,
            timing_source="pipeline_timeline",
            started_at=event.get("started_at"),
            ended_at=event.get("ended_at"),
            metadata=_dict(event.get("metadata")),
        )
        stages.append(row)
        stage_by_name[name] = row

    stages.sort(key=lambda row: (-float(row.get("duration_seconds", 0.0) or 0.0), str(row.get("stage", ""))))
    for idx, stage in enumerate(stages, start=1):
        stage["rank"] = idx

    total_seconds = _profile_total_seconds(parse_seconds=parse_seconds, analysis_timing=analysis_timing, stages=stages)
    quality_flags = _quality_flags(stages=stages, prompt_budget=prompt_budget, model_usage=model_usage)
    return {
        "latency_profile_version": LATENCY_PROFILE_VERSION,
        "document_id": document_id,
        "total_known_seconds": total_seconds,
        "slowest_stage": stages[0]["stage"] if stages else "",
        "stages": stages,
        "top_bottlenecks": stages[:5],
        "model_totals": _model_totals(model_usage),
        "prompt_totals": _prompt_totals(prompt_budget),
        "cache_summary": _cache_summary(stages),
        "quality_flags": quality_flags,
    }


def _stage_row(
    stage: str,
    *,
    duration_seconds: float,
    timing_source: str,
    model_usage: dict[str, Any] | None = None,
    prompt_block: dict[str, Any] | None = None,
    started_at: Any = None,
    ended_at: Any = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "stage": stage,
        "duration_seconds": round(duration_seconds, 4),
        "timing_source": timing_source,
    }
    if started_at:
        row["started_at"] = str(started_at)
    if ended_at:
        row["ended_at"] = str(ended_at)
    if metadata:
        row["metadata"] = metadata
    model = _model_stage_summary(model_usage or {})
    if model:
        row["model_usage"] = model
    execution = _execution_summary(model_usage or {})
    if execution:
        row["execution"] = execution
    prompt = _prompt_block_summary(prompt_block or {})
    if prompt:
        row["prompt_budget"] = prompt
    return row


def _merge_stage_metadata(stage: dict[str, Any], metadata: dict[str, Any]) -> None:
    if not metadata:
        return
    existing = _dict(stage.get("metadata"))
    merged = {**existing, **metadata}
    if merged:
        stage["metadata"] = merged


def _parse_metadata(parse_timing: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in {
            "parser_reuse": bool(parse_timing.get("parser_reuse")),
            "reused_assets": int(_number(parse_timing.get("reused_assets")) or 0),
            "asset_status_counts": _dict(parse_timing.get("asset_status_counts")),
        }.items()
        if _has_profile_value(value) or key == "parser_reuse"
    }


def _cache_summary(stages: list[dict[str, Any]]) -> dict[str, Any]:
    cache_hit_stages: list[str] = []
    parse_reused = False
    for stage in stages:
        name = str(stage.get("stage", "") or "")
        metadata = _dict(stage.get("metadata"))
        if bool(metadata.get("cache_hit")):
            cache_hit_stages.append(name)
        if name == "parse" and bool(metadata.get("parser_reuse")):
            parse_reused = True
            cache_hit_stages.append("parse")
    cache_hit_stages = _unique(cache_hit_stages)
    return {
        "parse_reused": parse_reused,
        "cache_hit_stages": cache_hit_stages,
        "cache_hit_count": len(cache_hit_stages),
    }


def _model_stage_summary(usage: dict[str, Any]) -> dict[str, Any]:
    if not usage:
        return {}
    calls = sum(int(_number(usage.get(f"{key}_calls")) or 0) for key in MODEL_STAGE_KEYS)
    errors = sum(int(_number(usage.get(f"{key}_errors")) or 0) for key in MODEL_STAGE_KEYS)
    seconds = sum(float(_number(usage.get(f"{key}_total_seconds")) or 0.0) for key in MODEL_STAGE_KEYS)
    load_calls = sum(int(_number(usage.get(f"{key}_model_load_calls")) or 0) for key in MODEL_STAGE_KEYS)
    load_errors = sum(int(_number(usage.get(f"{key}_model_load_errors")) or 0) for key in MODEL_STAGE_KEYS)
    load_seconds = sum(float(_number(usage.get(f"{key}_model_load_seconds")) or 0.0) for key in MODEL_STAGE_KEYS)
    out: dict[str, Any] = {
        "calls": calls,
        "errors": errors,
        "total_seconds": round(seconds, 4),
    }
    if load_calls or load_seconds:
        out["model_load_calls"] = load_calls
        out["model_load_errors"] = load_errors
        out["model_load_seconds"] = round(load_seconds, 4)
    slowest = str(usage.get("slowest_model", "") or "").strip()
    slowest_seconds = _number(usage.get("slowest_seconds"))
    if slowest:
        out["slowest_model"] = slowest
    if slowest_seconds is not None:
        out["slowest_seconds"] = round(slowest_seconds, 4)
    return out


def _model_totals(usage: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in MODEL_STAGE_KEYS:
        calls = int(_number(usage.get(f"{key}_calls")) or 0)
        errors = int(_number(usage.get(f"{key}_errors")) or 0)
        seconds = _number(usage.get(f"{key}_total_seconds"))
        out[key] = {
            "calls": calls,
            "errors": errors,
            "total_seconds": round(seconds, 4) if seconds is not None else None,
            "model_load_calls": int(_number(usage.get(f"{key}_model_load_calls")) or 0),
            "model_load_errors": int(_number(usage.get(f"{key}_model_load_errors")) or 0),
            "model_load_seconds": round(float(_number(usage.get(f"{key}_model_load_seconds")) or 0.0), 4),
        }
    slowest = str(usage.get("slowest_model", "") or "").strip()
    slowest_seconds = _number(usage.get("slowest_seconds"))
    if slowest:
        out["slowest_model"] = slowest
    if slowest_seconds is not None:
        out["slowest_seconds"] = round(slowest_seconds, 4)
    return out


def _execution_summary(usage: dict[str, Any]) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    execution = _dict(usage.get("execution"))
    if execution:
        attempts.append(execution)
    for attempt in _list(usage.get("execution_attempts")):
        if isinstance(attempt, dict):
            attempts.append(attempt)
    if not attempts:
        return {}
    out_attempts: list[dict[str, Any]] = []
    for attempt in attempts[:5]:
        out_attempts.append(
            {
                key: value
                for key, value in {
                    "mode": str(attempt.get("mode", "") or ""),
                    "kind": str(attempt.get("kind", "") or ""),
                    "chunk_count": int(_number(attempt.get("chunk_count")) or 0),
                    "timeout_seconds": int(_number(attempt.get("timeout_seconds")) or 0),
                    "elapsed_seconds": _number(attempt.get("elapsed_seconds")),
                    "timed_out": bool(attempt.get("timed_out")),
                    "exitcode": int(_number(attempt.get("exitcode")) or 0),
                    "payload_received": bool(attempt.get("payload_received")),
                    "ok": bool(attempt.get("ok")),
                    "error": str(attempt.get("error", "") or ""),
                }.items()
                if _has_profile_value(value) or isinstance(value, bool)
            }
        )
    timed_out = any(bool(attempt.get("timed_out")) for attempt in attempts)
    total_seconds = sum(float(_number(attempt.get("elapsed_seconds")) or 0.0) for attempt in attempts)
    return {
        "attempt_count": len(attempts),
        "timed_out": timed_out,
        "total_elapsed_seconds": round(total_seconds, 4),
        "attempts": out_attempts,
    }


def _prompt_modality_block(prompt_budget: dict[str, Any], stage: str) -> dict[str, Any]:
    by_modality = _dict(prompt_budget.get("by_modality"))
    key = "supplement" if stage == "supplement" else stage
    return _dict(by_modality.get(key))


def _prompt_block_summary(block: dict[str, Any]) -> dict[str, Any]:
    if not block:
        return {}
    return {
        key: value
        for key, value in {
            "prompt_calls": int(_number(block.get("prompt_calls")) or 0),
            "total_prompt_chars": int(_number(block.get("total_prompt_chars")) or 0),
            "max_prompt_chars": int(_number(block.get("max_prompt_chars")) or 0),
            "average_prompt_chars": _number(block.get("average_prompt_chars")),
            "max_prompt_blocks": int(_number(block.get("max_prompt_blocks")) or 0),
            "total_prompt_seconds": _number(block.get("total_prompt_seconds")),
            "max_prompt_seconds": _number(block.get("max_prompt_seconds")),
            "average_prompt_seconds": _number(block.get("average_prompt_seconds")),
            "slowest_prompt_batch": _dict(block.get("slowest_prompt_batch")),
        }.items()
        if _has_profile_value(value)
    }


def _prompt_totals(prompt_budget: dict[str, Any]) -> dict[str, Any]:
    totals = _dict(prompt_budget.get("totals"))
    return {
        key: value
        for key, value in {
            "prompt_calls": int(_number(totals.get("prompt_calls")) or 0),
            "total_prompt_chars": int(_number(totals.get("total_prompt_chars")) or 0),
            "max_prompt_chars": int(_number(totals.get("max_prompt_chars")) or 0),
            "average_prompt_chars": _number(totals.get("average_prompt_chars")),
            "max_prompt_modality": str(totals.get("max_prompt_modality", "") or ""),
            "total_prompt_seconds": _number(totals.get("total_prompt_seconds")),
            "max_prompt_seconds": _number(totals.get("max_prompt_seconds")),
            "max_prompt_seconds_modality": str(totals.get("max_prompt_seconds_modality", "") or ""),
            "average_prompt_seconds": _number(totals.get("average_prompt_seconds")),
        }.items()
        if _has_profile_value(value)
    }


def _profile_total_seconds(
    *,
    parse_seconds: float | None,
    analysis_timing: dict[str, Any],
    stages: list[dict[str, Any]],
) -> float | None:
    analysis_total = _number(analysis_timing.get("analysis_total_seconds"))
    if parse_seconds is not None or analysis_total is not None:
        return round(float(parse_seconds or 0.0) + float(analysis_total or 0.0), 4)
    if not stages:
        return None
    return round(sum(float(stage.get("duration_seconds", 0.0) or 0.0) for stage in stages), 4)


def _quality_flags(
    *,
    stages: list[dict[str, Any]],
    prompt_budget: dict[str, Any],
    model_usage: dict[str, Any],
) -> list[str]:
    flags: list[str] = []
    if not stages:
        flags.append("latency_timing_missing")
    elif stages[0].get("stage"):
        flags.append(f"{stages[0]['stage']}_slowest_stage")
    for stage in stages:
        duration = _number(stage.get("duration_seconds")) or 0.0
        if duration < 60.0:
            continue
        model = _dict(stage.get("model_usage"))
        execution = _dict(stage.get("execution"))
        if bool(execution.get("timed_out")):
            flags.append(f"{stage.get('stage')}_subprocess_timeout")
        if not model:
            flags.append(f"{stage.get('stage')}_slow_without_model_usage")
            continue
        if int(_number(model.get("calls")) or 0) == 0:
            flags.append(f"{stage.get('stage')}_slow_without_model_calls")
        if int(_number(model.get("model_load_calls")) or 0) > 0 and (
            _number(model.get("model_load_seconds")) or 0.0
        ) >= 15.0:
            flags.append(f"{stage.get('stage')}_model_cold_start")
    prompt_flags = [
        str(flag).strip()
        for flag in _list(prompt_budget.get("quality_flags"))
        if str(flag).strip()
    ]
    flags.extend(prompt_flags[:8])
    for stage in stages:
        metadata = _dict(stage.get("metadata"))
        stage_name = str(stage.get("stage", "") or "")
        if bool(metadata.get("cache_hit")) and stage_name:
            flags.append(f"{stage_name}_cache_hit")
        if stage_name == "parse" and bool(metadata.get("parser_reuse")):
            flags.append("parse_reused")
    totals = _dict(prompt_budget.get("totals"))
    if (_number(totals.get("max_prompt_seconds")) or 0.0) >= 120.0:
        modality = str(totals.get("max_prompt_seconds_modality", "") or "").strip()
        flags.append(f"{modality}_slow_prompt_call" if modality else "slow_prompt_call")
    if not model_usage:
        flags.append("model_usage_missing")
    elif sum(int(_number(model_usage.get(f"{key}_calls")) or 0) for key in MODEL_STAGE_KEYS) == 0:
        flags.append("model_calls_zero")
    return _unique(flags)


def _pipeline_step_stage_name(step: str) -> str:
    normalized = step.strip().lower()
    if normalized == "parse_document_assets":
        return "parse"
    if normalized == "run_full_analysis":
        return ""
    if normalized in {"prepare_analysis", "initialize_modality_analysis", "write_analysis_diagnostics", "mark_job_completed", "report_retention"}:
        return normalized
    return ""


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _has_profile_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, dict):
        return bool(value)
    if isinstance(value, (list, tuple, set)):
        return bool(value)
    if value == "" or value == 0 or value == 0.0:
        return False
    return True


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except Exception:
        return None
    if parsed < 0:
        return None
    return parsed


def _unique(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out
