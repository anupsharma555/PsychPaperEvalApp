from __future__ import annotations

import json
import multiprocessing as mp
from queue import Empty as QueueEmpty
import re
from typing import Any

from app.core.config import settings
from app.services.analysis.llm import chat_text_deep
from app.services.analysis.prompts import RECONCILE_SYSTEM
from app.services.analysis.utils import (
    clamp_confidence,
    extract_json,
    extract_refs_from_text,
    max_chars_for_ctx,
    truncate_text,
)

POSITIVE_WORDS = {"increase", "improve", "higher", "reduction", "benefit", "better", "positive", "significant"}
NEGATIVE_WORDS = {"decrease", "worse", "lower", "null", "no effect", "negative", "nonsignificant"}
REASON_SET = {"unsupported", "contradicted", "magnitude_mismatch", "missing_modality"}
SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2}


def reconcile_reports(
    text_report: dict[str, Any],
    table_report: dict[str, Any],
    figure_report: dict[str, Any],
    supp_report: dict[str, Any],
) -> dict[str, Any]:
    claim_packets = _claim_packets(text_report)
    modality_packets = (
        list(table_report.get("evidence_packets", []))
        + list(figure_report.get("evidence_packets", []))
        + list(supp_report.get("evidence_packets", []))
    )
    cross_modal_claims: list[dict[str, Any]] = []
    discrepancies: list[dict[str, Any]] = []
    llm_review_inputs: list[dict[str, Any]] = []

    for index, claim_packet in enumerate(claim_packets, start=1):
        claim_id = str(claim_packet.get("finding_id") or f"claim-{index}")
        claim_text = str(claim_packet.get("statement") or "").strip()
        claim_refs = set(claim_packet.get("evidence_refs", []))
        claim_stmt_refs = extract_refs_from_text(claim_text)
        verifiable_claim = _claim_is_verifiable(claim_packet, claim_stmt_refs)

        related = [
            packet
            for packet in modality_packets
            if _is_related(claim_packet, packet, claim_stmt_refs)
        ]
        support_ids: list[str] = []
        conflict_ids: list[str] = []
        unresolved_ids: list[str] = []
        generated_discrepancies: list[dict[str, Any]] = []

        if not related:
            discrepancy = _build_discrepancy(
                claim=claim_text,
                reason="missing_modality",
                evidence=sorted(claim_refs | claim_stmt_refs),
                linked_packet_ids=[],
                confidence=0.45,
            )
            if discrepancy and verifiable_claim:
                generated_discrepancies.append(discrepancy)
        else:
            for packet in related:
                packet_id = str(packet.get("finding_id") or "")
                if not packet_id:
                    continue
                mismatch_reason = _numeric_mismatch_reason(claim_packet, packet)
                if mismatch_reason:
                    conflict_ids.append(packet_id)
                    discrepancy = _build_discrepancy(
                        claim=claim_text,
                        reason=mismatch_reason,
                        evidence=sorted(set(claim_refs | claim_stmt_refs | set(packet.get("evidence_refs", [])))),
                        linked_packet_ids=[packet_id],
                        confidence=0.7,
                    )
                    if discrepancy:
                        generated_discrepancies.append(discrepancy)
                    continue
                if _is_contradiction(claim_text, str(packet.get("statement") or "")):
                    conflict_ids.append(packet_id)
                    discrepancy = _build_discrepancy(
                        claim=claim_text,
                        reason="contradicted",
                        evidence=sorted(set(claim_refs | claim_stmt_refs | set(packet.get("evidence_refs", [])))),
                        linked_packet_ids=[packet_id],
                        confidence=0.65,
                    )
                    if discrepancy:
                        generated_discrepancies.append(discrepancy)
                    continue
                if _has_overlap(claim_packet, packet, claim_stmt_refs):
                    support_ids.append(packet_id)
                else:
                    unresolved_ids.append(packet_id)

            if not support_ids and not conflict_ids:
                discrepancy = _build_discrepancy(
                    claim=claim_text,
                    reason="unsupported",
                    evidence=sorted(claim_refs | claim_stmt_refs),
                    linked_packet_ids=unresolved_ids,
                    confidence=0.4,
                )
                if discrepancy and verifiable_claim:
                    generated_discrepancies.append(discrepancy)

        status = _status_for_claim(support_ids, conflict_ids, unresolved_ids)
        if status in {"unresolved", "partial", "contradicted"}:
            llm_review_inputs.append(
                {
                    "claim_id": claim_id,
                    "claim": claim_text,
                    "evidence": sorted(claim_refs),
                    "status": status,
                    "related_packets": related[:12],
                }
            )

        cross_modal_claims.append(
            {
                "claim_id": claim_id,
                "claim": claim_text,
                "evidence": sorted(claim_refs),
                "support_packets": sorted(set(support_ids)),
                "conflict_packets": sorted(set(conflict_ids)),
                "unresolved_packets": sorted(set(unresolved_ids)),
                "status": status,
                "confidence": clamp_confidence(claim_packet.get("confidence", 0.0)),
            }
        )
        discrepancies.extend(generated_discrepancies)

    llm_discrepancies = _llm_reconcile_unresolved(llm_review_inputs)
    discrepancies.extend(llm_discrepancies)
    discrepancies = _dedupe_discrepancies(discrepancies)
    stats = {
        "claims_total": len(cross_modal_claims),
        "claims_supported": sum(1 for claim in cross_modal_claims if claim["status"] == "supported"),
        "claims_contradicted": sum(1 for claim in cross_modal_claims if claim["status"] == "contradicted"),
        "claims_partial": sum(1 for claim in cross_modal_claims if claim["status"] == "partial"),
        "claims_unresolved": sum(1 for claim in cross_modal_claims if claim["status"] == "unresolved"),
        "discrepancies_total": len(discrepancies),
        "llm_review_invoked": bool(llm_review_inputs),
        "llm_review_inputs": len(llm_review_inputs),
        "llm_discrepancies_added": len(llm_discrepancies),
    }
    return {"cross_modal_claims": cross_modal_claims, "discrepancies": discrepancies, "stats": stats}


def _claim_packets(text_report: dict[str, Any]) -> list[dict[str, Any]]:
    packets = list(text_report.get("claim_packets", []))
    if packets:
        return packets
    mapped: list[dict[str, Any]] = []
    for idx, claim in enumerate(text_report.get("claims", []), start=1):
        evidence = claim.get("evidence") or []
        mapped.append(
            {
                "finding_id": claim.get("claim_id") or f"claim-{idx}",
                "statement": claim.get("claim", ""),
                "evidence_refs": evidence if isinstance(evidence, list) else [str(evidence)],
                "confidence": claim.get("confidence", 0.0),
                "category": "claim",
            }
        )
    return mapped


def _is_related(claim_packet: dict[str, Any], packet: dict[str, Any], claim_stmt_refs: set[str]) -> bool:
    claim_refs = set(claim_packet.get("evidence_refs", []))
    packet_refs = set(packet.get("evidence_refs", []))
    if claim_refs & packet_refs:
        return True
    packet_stmt_refs = extract_refs_from_text(str(packet.get("statement") or ""))
    if claim_stmt_refs and packet_stmt_refs and claim_stmt_refs & packet_stmt_refs:
        return True
    claim_value = _safe_float(claim_packet.get("value"))
    packet_value = _safe_float(packet.get("value"))
    if claim_value is not None and packet_value is not None:
        return True
    return False


def _has_overlap(claim_packet: dict[str, Any], packet: dict[str, Any], claim_stmt_refs: set[str]) -> bool:
    claim_refs = set(claim_packet.get("evidence_refs", []))
    packet_refs = set(packet.get("evidence_refs", []))
    if claim_refs & packet_refs:
        return True
    packet_stmt_refs = extract_refs_from_text(str(packet.get("statement") or ""))
    if claim_stmt_refs and packet_stmt_refs and claim_stmt_refs & packet_stmt_refs:
        return True
    return False


def _numeric_mismatch_reason(claim_packet: dict[str, Any], packet: dict[str, Any]) -> str | None:
    claim_value = _safe_float(claim_packet.get("value"))
    packet_value = _safe_float(packet.get("value"))
    if claim_value is None or packet_value is None:
        return None
    denom = max(abs(claim_value), abs(packet_value), 1e-6)
    rel = abs(claim_value - packet_value) / denom
    if rel > 0.35:
        return "magnitude_mismatch"
    return None


def _is_contradiction(claim_text: str, packet_text: str) -> bool:
    claim = _canonical(claim_text)
    packet = _canonical(packet_text)
    if not claim or not packet:
        return False
    claim_pos = any(word in claim for word in POSITIVE_WORDS)
    claim_neg = any(word in claim for word in NEGATIVE_WORDS)
    packet_pos = any(word in packet for word in POSITIVE_WORDS)
    packet_neg = any(word in packet for word in NEGATIVE_WORDS)
    if claim_pos and packet_neg:
        return True
    if claim_neg and packet_pos:
        return True
    # explicit negation mismatch on effect phrases
    has_effect_phrase = "effect" in claim or "effect" in packet
    if has_effect_phrase and (("no effect" in claim) ^ ("no effect" in packet)):
        return True
    return False


def _build_discrepancy(
    *,
    claim: str,
    reason: str,
    evidence: list[str],
    linked_packet_ids: list[str],
    confidence: float,
) -> dict[str, Any] | None:
    evidence_set = sorted(set(str(ref).strip() for ref in evidence if str(ref).strip()))
    linked_set = sorted(set(str(pid).strip() for pid in linked_packet_ids if str(pid).strip()))
    # Avoid surfacing discrepancies that do not carry any traceable evidence path.
    if not evidence_set and not linked_set:
        return None
    normalized_reason = reason if reason in REASON_SET else "unsupported"
    return {
        "claim": claim,
        "reason": normalized_reason,
        "evidence": evidence_set,
        "severity": _severity_for_reason(normalized_reason),
        "confidence": clamp_confidence(confidence),
        "linked_packet_ids": linked_set,
    }


def _status_for_claim(support_ids: list[str], conflict_ids: list[str], unresolved_ids: list[str]) -> str:
    if conflict_ids and not support_ids:
        return "contradicted"
    if conflict_ids and support_ids:
        return "partial"
    if support_ids:
        return "supported"
    if unresolved_ids:
        return "unresolved"
    return "unresolved"


def _severity_for_reason(reason: str) -> str:
    if reason in {"contradicted", "magnitude_mismatch"}:
        return "high"
    if reason == "missing_modality":
        return "medium"
    return "low"


def _llm_reconcile_unresolved(unresolved_inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not unresolved_inputs:
        return []
    prompt = (
        "Review non-supported cross-modal claims and return only discrepancies that are strongly justified.\n\n"
        + json.dumps({"claim_reviews": unresolved_inputs[:20]}, indent=2)
    )
    prompt = truncate_text(prompt, max_chars_for_ctx(settings.llm_n_ctx))
    data = _run_reconcile_prompt(prompt)
    if not isinstance(data, dict):
        return []
    output: list[dict[str, Any]] = []
    for item in data.get("discrepancies", []):
        if not isinstance(item, dict):
            continue
        discrepancy = _build_discrepancy(
            claim=str(item.get("claim", "")).strip(),
            reason=str(item.get("reason", "unsupported")).strip(),
            evidence=[str(ref) for ref in item.get("evidence", [])] if isinstance(item.get("evidence"), list) else [],
            linked_packet_ids=[
                str(ref) for ref in item.get("linked_packet_ids", [])
            ]
            if isinstance(item.get("linked_packet_ids"), list)
            else [],
            confidence=clamp_confidence(item.get("confidence", 0.0)),
        )
        if discrepancy:
            output.append(discrepancy)
    return output


def _llm_reconcile_worker(prompt: str, out_queue: Any) -> None:
    try:
        response = chat_text_deep(prompt, system=RECONCILE_SYSTEM)
        out_queue.put({"ok": True, "data": extract_json(response)})
    except Exception as exc:
        out_queue.put({"ok": False, "error": str(exc)})


def _run_reconcile_prompt(prompt: str) -> Any:
    guard_enabled = bool(settings.analysis_narrative_overrides_subprocess_guard_enabled)
    timeout_seconds = max(15, int(settings.analysis_narrative_overrides_subprocess_timeout_sec or 0))
    if not guard_enabled:
        try:
            response = chat_text_deep(prompt, system=RECONCILE_SYSTEM)
            return extract_json(response)
        except Exception:
            return {}

    context = mp.get_context("spawn")
    out_queue = context.Queue(maxsize=1)
    proc = context.Process(target=_llm_reconcile_worker, args=(prompt, out_queue))
    proc.start()
    proc.join(timeout_seconds)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        return {}
    if int(proc.exitcode or 0) != 0:
        return {}
    try:
        payload = out_queue.get_nowait()
    except QueueEmpty:
        return {}
    except Exception:
        return {}
    if not isinstance(payload, dict) or not payload.get("ok"):
        return {}
    return payload.get("data", {})


def _dedupe_discrepancies(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        claim = str(item.get("claim", "")).strip()
        if not claim:
            continue
        reason = str(item.get("reason", "unsupported")).strip()
        if reason not in REASON_SET:
            reason = "unsupported"
        key = (
            _canonical(claim),
            reason,
        )
        evidence = [str(ref) for ref in item.get("evidence", [])] if isinstance(item.get("evidence"), list) else []
        linked = (
            [str(ref) for ref in item.get("linked_packet_ids", [])]
            if isinstance(item.get("linked_packet_ids"), list)
            else []
        )
        confidence = clamp_confidence(item.get("confidence", 0.0))
        severity = str(item.get("severity", _severity_for_reason(reason))).strip().lower()
        if severity not in SEVERITY_ORDER:
            severity = _severity_for_reason(reason)

        existing = merged.get(key)
        if existing is None:
            merged[key] = {
                "claim": claim,
                "reason": reason,
                "evidence": sorted(set(evidence)),
                "severity": severity,
                "confidence": confidence,
                "linked_packet_ids": sorted(set(linked)),
            }
            continue

        existing["evidence"] = sorted(set(existing.get("evidence", []) + evidence))
        existing["linked_packet_ids"] = sorted(set(existing.get("linked_packet_ids", []) + linked))
        existing["confidence"] = max(clamp_confidence(existing.get("confidence", 0.0)), confidence)
        existing_severity = str(existing.get("severity", "low")).lower()
        if SEVERITY_ORDER.get(severity, 0) > SEVERITY_ORDER.get(existing_severity, 0):
            existing["severity"] = severity

    return [item for item in merged.values() if item.get("evidence") or item.get("linked_packet_ids")]


def _canonical(text: str) -> str:
    normalized = re.sub(r"\s+", " ", (text or "").strip().lower())
    return normalized


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _claim_is_verifiable(claim_packet: dict[str, Any], claim_stmt_refs: set[str]) -> bool:
    quality_flags = {str(flag).strip().lower() for flag in claim_packet.get("quality_flags", [])}
    if "missing_evidence" in quality_flags:
        return False
    if claim_packet.get("evidence_refs"):
        return True
    if claim_stmt_refs:
        return True
    if _safe_float(claim_packet.get("value")) is not None:
        return True
    if _safe_float(claim_packet.get("p_value")) is not None:
        return True
    if _safe_float(claim_packet.get("effect_size")) is not None:
        return True
    return False
