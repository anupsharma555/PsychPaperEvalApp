from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

try:
    from validate_gold_standards import load_gold_standard
except ModuleNotFoundError:  # pragma: no cover - exercised when imported as scripts.compare_evidence_to_gold
    from scripts.validate_gold_standards import load_gold_standard


VALID_SECTIONS = {"introduction", "methods", "results", "discussion", "conclusion"}
VALID_MODALITIES = {"text", "table", "figure", "supplement"}
TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
NUMBER_RE = re.compile(r"(?<![A-Za-z0-9])-?(?:\d+(?:\.\d+)?|\.\d+)")
ENTITY_ALIASES = {
    "cingulo opercular network": ["con"],
    "default mode network": ["dmn"],
    "nucleus accumbens": ["nac"],
}
DETAIL_TYPE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "medication_or_therapeutic",
        re.compile(
            r"\b("
            r"medication|drug|therapeutic|pharmacologic|antidepressant|antipsychotic|ssri|snri|"
            r"ketamine|esketamine|fluoxetine|sertraline|escitalopram|citalopram|paroxetine|"
            r"venlafaxine|duloxetine|bupropion|mirtazapine|lithium|clozapine|risperidone|"
            r"olanzapine|quetiapine|aripiprazole|methylphenidate|amphetamine|antibiotic|r13"
            r")\b",
            re.IGNORECASE,
        ),
    ),
    (
        "dose_schedule",
        re.compile(
            r"\b(\d+(?:\.\d+)?\s*(?:mg|mcg|ug|µg|g|kg|ml|iu|units?)\b|mg/kg|dose|dosage|"
            r"route|oral|intravenous|subcutaneous|intramuscular|daily|weekly|duration|weeks?|months?|"
            r"administered|treated)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "intervention_or_exposure",
        re.compile(
            r"\b(intervention|treatment|therapy|exposure|comparator|control arm|placebo|stimulation|"
            r"psychotherapy|training|randomi[sz]ed|trial arm|cohousing|dirty bedding|probiotic|antibiotic)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "outcome_measure",
        re.compile(
            r"\b(outcome|endpoint|measure|scale|questionnaire|instrument|score|gad-7|phq-9|ham-?d|"
            r"madrs|panss|ymrs|timepoint|follow[- ]up|symptom|morris water maze|fear conditioning)\b",
            re.IGNORECASE,
        ),
    ),
    ("adverse_event", re.compile(r"\b(adverse event|side effect|tolerability|safety|dropout|discontinuation)\b", re.IGNORECASE)),
    (
        "model_system",
        re.compile(
            r"\b(cell lines?|cell clones?|organoid|mouse|mice|rat|animal model|in vivo|in vitro|"
            r"specimens?|culture|construct|plasmid|vector|transfection|mutagenesis|mcf-7|mda-mb-231|"
            r"hek293|5xfad|3xtg)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "assay_readout",
        re.compile(
            r"\b(assay|readout|qpcr|pcr|western blot|elisa|rna[- ]?seq|sequencing|flow cytometry|"
            r"microscopy|immunostaining|biomarker|protein|gene expression|validation|ic50|fitc-dextran)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "statistical_result",
        re.compile(
            r"\b(p\s*[<=>]\s*0?\.\d+|confidence interval|ci\b|odds ratio|hazard ratio|effect size|"
            r"cohen'?s?\s*d|beta|regression|mixed[- ]effects|bayesian|sensitivity analysis|subgroup|"
            r"adjusted|covariate|significant|associated|association|linked|increased|decreased|higher|lower|ic50)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "data_source_or_design",
        re.compile(
            r"\b(cohort|registry|claims database|survey|dataset|data source|cross[- ]sectional|case[- ]control|"
            r"longitudinal|sample|participants?|subjects?|adults?|systematic review|meta[- ]analysis|"
            r"eligibility criteria|search strategy|risk of bias)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "rationale_or_objective",
        re.compile(
            r"\b(objective|aim|hypothesis|rationale|research question|background|motivat(?:e|ion)|"
            r"knowledge gap|unmet need|designed to test|sought to determine)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "interpretation_or_implication",
        re.compile(
            r"\b(interpret(?:ation|ed)?|implication|may reflect|suggests?|supports?|consistent with|"
            r"clinical relevance|biological significance|mechanism|context)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "limitation_or_caution",
        re.compile(
            r"\b(limitation|caution|generalizability|underpowered|bias|confounding|cannot establish|"
            r"cross[- ]sectional|future work|future research|replication needed)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "conclusion_or_takeaway",
        re.compile(
            r"\b(conclusion|in conclusion|overall|taken together|collectively|main takeaway|"
            r"these findings|these results|we conclude|the study concludes|highlight|underscore)\b",
            re.IGNORECASE,
        ),
    ),
    ("tool_or_algorithm", re.compile(r"\b(algorithm|software|pipeline|classifier|benchmark|validation dataset|model performance)\b", re.IGNORECASE)),
    ("cross_modal_result", re.compile(r"\b(figure|table|supplement|panel|legend)\b", re.IGNORECASE)),
]
DETAIL_TYPE_CATEGORY_HINTS = {
    "medication": "medication_or_therapeutic",
    "intervention": "intervention_or_exposure",
    "clinical": "outcome_measure",
    "assay": "assay_readout",
    "model_system": "model_system",
    "stats": "statistical_result",
    "table_quality": "statistical_result",
    "data_consistency": "statistical_result",
    "objective": "rationale_or_objective",
    "rationale": "rationale_or_objective",
    "interpretation": "interpretation_or_implication",
    "implications": "interpretation_or_implication",
    "limitations": "limitation_or_caution",
    "conclusion": "conclusion_or_takeaway",
    "algorithm": "tool_or_algorithm",
}
DEFAULT_MIN_USABLE_PACKET_RATE = 0.8
DEFAULT_MIN_SECTION_COVERAGE_RATE = 0.8
DEFAULT_MIN_CRITICAL_CLAIM_CANDIDATE_RATE = 0.8
DEFAULT_MIN_EXPECTED_ENTITY_OBSERVABILITY_RATE = 0.5
DEFAULT_MIN_EXPECTED_NUMBER_OBSERVABILITY_RATE = 0.8
DEFAULT_MIN_EXPECTED_DETAIL_TYPE_OBSERVABILITY_RATE = 0.8
DEFAULT_FORBIDDEN_CLAIM_THRESHOLD = 0.35


def load_evidence_packets(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return evidence_packets_from_payload(payload)


def load_evidence_metadata(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return evidence_metadata_from_payload(payload)


def evidence_packets_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    summary_json = payload.get("summary_json")
    if isinstance(summary_json, dict):
        return evidence_packets_from_payload(summary_json)
    direct_packets = payload.get("evidence_packets")
    if isinstance(direct_packets, list):
        return [item for item in direct_packets if isinstance(item, dict)]
    for report_key in ("v2_report", "v1_report"):
        report_packets = _packets_from_report_sections(payload.get(report_key), prefix=report_key)
        if report_packets:
            return report_packets
    direct_report_packets = _packets_from_report_sections(payload, prefix="report")
    if direct_report_packets:
        return direct_report_packets
    packets: list[dict[str, Any]] = []
    for modality, block in (payload.get("modalities") or {}).items():
        if not isinstance(block, dict):
            continue
        for packet in block.get("findings", []):
            if isinstance(packet, dict):
                packets.append({**packet, "modality": packet.get("modality") or modality})
    for detail in payload.get("scientific_details", []) or []:
        if isinstance(detail, dict):
            packets.append(
                {
                    "finding_id": detail.get("finding_id") or f"scientific-detail-{len(packets) + 1}",
                    "statement": detail.get("statement", ""),
                    "source_excerpt": detail.get("source_excerpt", ""),
                    "evidence_refs": detail.get("evidence_refs", []),
                    "modality": detail.get("source_modality") or detail.get("modality") or "text",
                    "section_label": detail.get("section_label") or "unknown",
                    "category": detail.get("category") or "other",
                    "detail_types": detail.get("detail_types", []),
                    "confidence": detail.get("confidence", 0.0),
                }
            )
    for section, rows in (payload.get("sections_extracted") or {}).items():
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, dict):
                packets.append(
                    {
                        "finding_id": row.get("finding_id") or f"section-extracted-{len(packets) + 1}",
                        "statement": row.get("statement", ""),
                        "evidence_refs": row.get("evidence_refs", []),
                        "modality": "text",
                        "section_label": section,
                        "category": row.get("kind") or "section_extracted",
                        "detail_types": row.get("detail_types", []),
                        "confidence": row.get("confidence", 0.0),
                    }
                )
    return packets


def evidence_metadata_from_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return _empty_evidence_metadata()
    summary_json = payload.get("summary_json")
    if isinstance(summary_json, dict):
        return evidence_metadata_from_payload(summary_json)

    diagnostics = payload.get("section_diagnostics", {})
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    evidence_plan = diagnostics.get("synthesis_evidence_plan", {})
    if not isinstance(evidence_plan, dict):
        evidence_plan = {}

    quality_flags = _clean_string_list(
        payload.get("synthesis_quality_flags")
        or evidence_plan.get("quality_flags")
    )
    critical_missing = _clean_missing_focus_slots(
        payload.get("critical_missing_focus_slots")
        or evidence_plan.get("critical_missing_focus_slots")
    )
    warnings = _clean_string_list(diagnostics.get("synthesis_evidence_warnings"))
    packet_coverage = _clean_packet_coverage(
        payload.get("evidence_packet_coverage")
        or diagnostics.get("evidence_packet_coverage")
    )
    return {
        "has_synthesis_evidence_plan": bool(evidence_plan),
        "synthesis_quality_flags": quality_flags,
        "critical_missing_focus_slots": critical_missing,
        "critical_missing_focus_slot_count": len(critical_missing),
        "missing_focus_slot_count": int(evidence_plan.get("missing_focus_slot_count", 0) or 0),
        "synthesis_evidence_warnings": warnings,
        "evidence_packet_coverage": packet_coverage,
    }


def compare_evidence_to_gold(
    evidence_packets: list[dict[str, Any]],
    gold_standard: dict[str, Any],
    *,
    evidence_metadata: dict[str, Any] | None = None,
    min_usable_packet_rate: float = DEFAULT_MIN_USABLE_PACKET_RATE,
    min_section_coverage_rate: float = DEFAULT_MIN_SECTION_COVERAGE_RATE,
    min_critical_claim_candidate_rate: float = DEFAULT_MIN_CRITICAL_CLAIM_CANDIDATE_RATE,
    min_expected_entity_observability_rate: float = DEFAULT_MIN_EXPECTED_ENTITY_OBSERVABILITY_RATE,
    min_expected_number_observability_rate: float = DEFAULT_MIN_EXPECTED_NUMBER_OBSERVABILITY_RATE,
    min_expected_detail_type_observability_rate: float = DEFAULT_MIN_EXPECTED_DETAIL_TYPE_OBSERVABILITY_RATE,
    forbidden_claim_threshold: float = DEFAULT_FORBIDDEN_CLAIM_THRESHOLD,
) -> dict[str, Any]:
    normalized_packets = [_normalize_packet(packet) for packet in evidence_packets if isinstance(packet, dict)]
    critical_claims = [
        claim for claim in gold_standard.get("critical_claims", []) if isinstance(claim, dict)
    ]
    required_sections = {
        str(claim.get("section") or "").lower()
        for claim in critical_claims
        if str(claim.get("section") or "").lower() in VALID_SECTIONS
    }
    packet_sections = {
        packet["section_label"] for packet in normalized_packets if packet["section_label"] in VALID_SECTIONS
    }
    usable_packets = [packet for packet in normalized_packets if packet["usable"]]
    claim_matches = [
        _match_critical_claim(claim, usable_packets)
        for claim in critical_claims
    ]
    claim_requirement_gaps = _claim_requirement_gaps(claim_matches)

    section_coverage_rate = _rate(len(required_sections & packet_sections), len(required_sections))
    usable_packet_rate = _rate(len(usable_packets), len(normalized_packets)) if normalized_packets else 0.0
    candidate_rate = _rate(
        sum(1 for item in claim_matches if item.get("has_candidate")),
        len(critical_claims),
    )
    entity_rate = _rate(
        sum(int(item.get("matched_entities", 0) or 0) for item in claim_matches),
        sum(int(item.get("expected_entities", 0) or 0) for item in claim_matches),
    )
    number_rate = _rate(
        sum(int(item.get("matched_numbers", 0) or 0) for item in claim_matches),
        sum(int(item.get("expected_numbers", 0) or 0) for item in claim_matches),
    )
    detail_type_rate = _rate(
        sum(int(item.get("matched_detail_types", 0) or 0) for item in claim_matches),
        sum(int(item.get("expected_detail_types", 0) or 0) for item in claim_matches),
    )
    content_score_basis = _benchmark_content_score_basis(
        claim_matches=claim_matches,
        required_sections=required_sections,
        packet_sections=packet_sections,
    )
    content_score = content_score_basis["score"]
    forbidden_violations = _match_forbidden_claims(
        gold_standard.get("report_should_not_claim", []),
        normalized_packets,
        threshold=forbidden_claim_threshold,
    )
    missing_fields = _packet_schema_gaps(normalized_packets)
    compatible = (
        usable_packet_rate >= min_usable_packet_rate
        and section_coverage_rate >= min_section_coverage_rate
        and candidate_rate >= min_critical_claim_candidate_rate
        and entity_rate >= min_expected_entity_observability_rate
        and number_rate >= min_expected_number_observability_rate
        and detail_type_rate >= min_expected_detail_type_observability_rate
        and not forbidden_violations
        and not missing_fields
    )
    failure_reasons = _compatibility_failure_reasons(
        usable_packet_rate=usable_packet_rate,
        min_usable_packet_rate=min_usable_packet_rate,
        section_coverage_rate=section_coverage_rate,
        min_section_coverage_rate=min_section_coverage_rate,
        candidate_rate=candidate_rate,
        min_critical_claim_candidate_rate=min_critical_claim_candidate_rate,
        entity_rate=entity_rate,
        min_expected_entity_observability_rate=min_expected_entity_observability_rate,
        number_rate=number_rate,
        min_expected_number_observability_rate=min_expected_number_observability_rate,
        detail_type_rate=detail_type_rate,
        min_expected_detail_type_observability_rate=min_expected_detail_type_observability_rate,
        forbidden_violations=forbidden_violations,
        schema_gaps=missing_fields,
    )
    return {
        "compatible": compatible,
        "case_id": gold_standard.get("case_id", ""),
        "thresholds": {
            "min_usable_packet_rate": min_usable_packet_rate,
            "min_section_coverage_rate": min_section_coverage_rate,
            "min_critical_claim_candidate_rate": min_critical_claim_candidate_rate,
            "min_expected_entity_observability_rate": min_expected_entity_observability_rate,
            "min_expected_number_observability_rate": min_expected_number_observability_rate,
            "min_expected_detail_type_observability_rate": min_expected_detail_type_observability_rate,
            "forbidden_claim_threshold": forbidden_claim_threshold,
        },
        "failure_reasons": failure_reasons,
        "packet_total": len(normalized_packets),
        "usable_packets": len(usable_packets),
        "usable_packet_rate": usable_packet_rate,
        "required_sections": sorted(required_sections),
        "packet_sections": sorted(packet_sections),
        "section_coverage_rate": section_coverage_rate,
        "overall_benchmark_score": content_score,
        "benchmark_content_score": content_score,
        "benchmark_content_score_basis": content_score_basis,
        "critical_claim_candidate_rate": candidate_rate,
        "expected_entity_observability_rate": entity_rate,
        "expected_number_observability_rate": number_rate,
        "expected_detail_type_observability_rate": detail_type_rate,
        "forbidden_claim_violations": forbidden_violations,
        "schema_gaps": missing_fields,
        "synthesis_evidence_diagnostics": _normalize_evidence_metadata(evidence_metadata),
        "claim_requirement_gaps": claim_requirement_gaps,
        "claim_matches": claim_matches,
    }


def _empty_evidence_metadata() -> dict[str, Any]:
    return {
        "has_synthesis_evidence_plan": False,
        "synthesis_quality_flags": [],
        "critical_missing_focus_slots": [],
        "critical_missing_focus_slot_count": 0,
        "missing_focus_slot_count": 0,
        "synthesis_evidence_warnings": [],
        "evidence_packet_coverage": _empty_packet_coverage(),
    }


def _benchmark_content_score_basis(
    *,
    claim_matches: list[dict[str, Any]],
    required_sections: set[str],
    packet_sections: set[str],
) -> dict[str, Any]:
    candidate_expected = len(claim_matches)
    candidate_matched = sum(1 for item in claim_matches if item.get("has_candidate"))
    entity_expected = sum(int(item.get("expected_entities", 0) or 0) for item in claim_matches)
    entity_matched = sum(int(item.get("matched_entities", 0) or 0) for item in claim_matches)
    number_expected = sum(int(item.get("expected_numbers", 0) or 0) for item in claim_matches)
    number_matched = sum(int(item.get("matched_numbers", 0) or 0) for item in claim_matches)
    detail_type_expected = sum(int(item.get("expected_detail_types", 0) or 0) for item in claim_matches)
    detail_type_matched = sum(int(item.get("matched_detail_types", 0) or 0) for item in claim_matches)
    section_expected = len(required_sections)
    section_matched = len(required_sections & packet_sections)
    expected_total = candidate_expected + entity_expected + number_expected + detail_type_expected + section_expected
    matched_total = candidate_matched + entity_matched + number_matched + detail_type_matched + section_matched
    return {
        "score": _rate(matched_total, expected_total),
        "matched_slots": matched_total,
        "expected_slots": expected_total,
        "extra_content_penalized": False,
        "components": {
            "critical_claim_candidates": {"matched": candidate_matched, "expected": candidate_expected},
            "expected_entities": {"matched": entity_matched, "expected": entity_expected},
            "expected_numbers": {"matched": number_matched, "expected": number_expected},
            "expected_detail_types": {"matched": detail_type_matched, "expected": detail_type_expected},
            "required_sections": {"matched": section_matched, "expected": section_expected},
        },
    }


def _normalize_evidence_metadata(value: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(value, dict):
        return _empty_evidence_metadata()
    critical_missing = _clean_missing_focus_slots(value.get("critical_missing_focus_slots"))
    return {
        "has_synthesis_evidence_plan": bool(value.get("has_synthesis_evidence_plan")),
        "synthesis_quality_flags": _clean_string_list(value.get("synthesis_quality_flags")),
        "critical_missing_focus_slots": critical_missing,
        "critical_missing_focus_slot_count": len(critical_missing),
        "missing_focus_slot_count": int(value.get("missing_focus_slot_count", 0) or 0),
        "synthesis_evidence_warnings": _clean_string_list(value.get("synthesis_evidence_warnings")),
        "evidence_packet_coverage": _clean_packet_coverage(value.get("evidence_packet_coverage")),
    }


def _compatibility_failure_reasons(
    *,
    usable_packet_rate: float,
    min_usable_packet_rate: float,
    section_coverage_rate: float,
    min_section_coverage_rate: float,
    candidate_rate: float,
    min_critical_claim_candidate_rate: float,
    entity_rate: float,
    min_expected_entity_observability_rate: float,
    number_rate: float,
    min_expected_number_observability_rate: float,
    detail_type_rate: float,
    min_expected_detail_type_observability_rate: float,
    forbidden_violations: list[dict[str, Any]],
    schema_gaps: list[str],
) -> list[str]:
    reasons: list[str] = []
    if usable_packet_rate < min_usable_packet_rate:
        reasons.append(f"usable packet rate {usable_packet_rate:.3f} < {min_usable_packet_rate:.3f}")
    if section_coverage_rate < min_section_coverage_rate:
        reasons.append(f"section coverage rate {section_coverage_rate:.3f} < {min_section_coverage_rate:.3f}")
    if candidate_rate < min_critical_claim_candidate_rate:
        reasons.append(f"critical claim candidate rate {candidate_rate:.3f} < {min_critical_claim_candidate_rate:.3f}")
    if entity_rate < min_expected_entity_observability_rate:
        reasons.append(f"expected entity observability rate {entity_rate:.3f} < {min_expected_entity_observability_rate:.3f}")
    if number_rate < min_expected_number_observability_rate:
        reasons.append(f"expected number observability rate {number_rate:.3f} < {min_expected_number_observability_rate:.3f}")
    if detail_type_rate < min_expected_detail_type_observability_rate:
        reasons.append(
            "expected detail-type observability rate "
            f"{detail_type_rate:.3f} < {min_expected_detail_type_observability_rate:.3f}"
        )
    if forbidden_violations:
        reasons.append(f"forbidden claim violations: {len(forbidden_violations)}")
    for gap in schema_gaps:
        reasons.append(f"schema gap: {gap}")
    return reasons


def _normalize_packet(packet: dict[str, Any]) -> dict[str, Any]:
    statement = _clean_text(packet.get("statement") or packet.get("summary") or packet.get("claim") or packet.get("result"))
    source_excerpt = _clean_text(packet.get("source_excerpt") or packet.get("verbatim_text") or packet.get("source_text"))
    refs = [str(ref).strip() for ref in _as_list(packet.get("evidence_refs") or packet.get("evidence")) if str(ref).strip()]
    section = str(packet.get("section_label") or packet.get("section") or "").strip().lower()
    if section not in VALID_SECTIONS:
        section = "unknown"
    modality = str(packet.get("modality") or packet.get("source_modality") or "").strip().lower()
    if modality == "supp":
        modality = "supplement"
    if modality not in VALID_MODALITIES:
        modality = "unknown"
    text = " ".join(part for part in (statement, source_excerpt) if part)
    category = str(packet.get("category") or "").strip()
    detail_types = _normalize_detail_types(
        packet.get("detail_types") or packet.get("scientific_detail_types"),
        text=" ".join(part for part in (modality, " ".join(refs), text) if part),
        category=category,
    )
    return {
        "finding_id": str(packet.get("finding_id") or "").strip(),
        "statement": statement,
        "source_excerpt": source_excerpt,
        "text": text,
        "evidence_refs": refs,
        "section_label": section,
        "modality": modality,
        "category": category,
        "detail_types": detail_types,
        "confidence": packet.get("confidence", 0.0),
        "usable": bool(statement and refs and (section in VALID_SECTIONS or modality in {"figure", "table", "supplement"})),
    }


def _packets_from_report_sections(report: Any, *, prefix: str) -> list[dict[str, Any]]:
    if not isinstance(report, dict):
        return []
    sections = report.get("sections")
    if not isinstance(sections, list):
        return []
    packets: list[dict[str, Any]] = []
    for section_index, section_block in enumerate(sections, start=1):
        if not isinstance(section_block, dict):
            continue
        section_label = str(section_block.get("section") or "").strip().lower()
        bullets = section_block.get("bullets")
        if not isinstance(bullets, list):
            continue
        for bullet_index, bullet in enumerate(bullets, start=1):
            if not isinstance(bullet, dict):
                continue
            statement = _clean_text(bullet.get("text") or bullet.get("statement") or bullet.get("summary"))
            refs = _as_list(
                bullet.get("evidence_refs")
                or bullet.get("evidence_ids")
                or bullet.get("anchors")
                or bullet.get("citations")
            )
            if not statement or not refs:
                continue
            packets.append(
                {
                    "finding_id": str(
                        bullet.get("finding_id")
                        or f"{prefix}-{section_index}-{bullet_index}"
                    ),
                    "statement": statement,
                    "evidence_refs": refs,
                    "section_label": section_label,
                    "modality": bullet.get("modality") or _infer_modality_from_refs(refs),
                    "category": bullet.get("source") or "report_section",
                    "detail_types": bullet.get("detail_types", []),
                    "confidence": bullet.get("confidence", 0.0),
                }
            )
    return packets


def _infer_modality_from_refs(refs: list[Any]) -> str:
    normalized = " ".join(str(ref).lower() for ref in refs)
    if "figure" in normalized or "fig" in normalized:
        return "figure"
    if "table" in normalized:
        return "table"
    if "supp" in normalized:
        return "supplement"
    return "text"


def _normalize_detail_types(value: Any, *, text: str, category: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []

    def _add(raw: Any) -> None:
        normalized = str(raw or "").strip()
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        out.append(normalized)

    for item in _as_list(value):
        _add(item)
    category_key = str(category or "").strip().lower()
    if category_key in DETAIL_TYPE_CATEGORY_HINTS:
        _add(DETAIL_TYPE_CATEGORY_HINTS[category_key])
    searchable = f"{category or ''} {text or ''}".strip()
    for detail_type, pattern in DETAIL_TYPE_PATTERNS:
        if pattern.search(searchable):
            _add(detail_type)
    return out


def _match_critical_claim(claim: dict[str, Any], packets: list[dict[str, Any]]) -> dict[str, Any]:
    section = str(claim.get("section") or "").lower()
    expected_entities = [str(item).strip() for item in claim.get("expected_entities", []) if str(item).strip()]
    expected_numbers = [
        item for item in claim.get("expected_numbers", []) if isinstance(item, dict) and item.get("value") is not None
    ]
    expected_detail_types = [
        str(item).strip()
        for item in claim.get("expected_detail_types", [])
        if str(item).strip()
    ]
    claim_text = _clean_text(claim.get("claim") or claim.get("evidence_quote"))
    same_section = [
        packet
        for packet in packets
        if packet.get("section_label") == section
        or (
            packet.get("section_label") == "unknown"
            and packet.get("modality") in {"figure", "table", "supplement"}
            and "cross_modal_result" in set(packet.get("detail_types", []))
        )
    ]
    best_packet: dict[str, Any] | None = None
    best_score = 0.0
    best_entities = 0
    best_numbers = 0
    best_detail_types = 0
    best_matched_entities: list[str] = []
    best_matched_numbers: list[str] = []
    best_matched_detail_types: list[str] = []
    support_packets: list[dict[str, Any]] = []
    for packet in same_section:
        matched_entities = [entity for entity in expected_entities if _contains_phrase(packet["text"], entity)]
        matched_numbers = [
            _expected_number_label(number)
            for number in expected_numbers
            if _contains_number(packet["text"], number)
        ]
        entity_hits = len(matched_entities)
        number_hits = len(matched_numbers)
        matched_detail_types = [
            detail_type
            for detail_type in expected_detail_types
            if detail_type in set(packet.get("detail_types", []))
        ]
        detail_type_hits = len(matched_detail_types)
        lexical = _token_overlap(claim_text, packet["text"])
        score = lexical + (0.12 * entity_hits) + (0.15 * number_hits) + (0.3 * detail_type_hits)
        if score > 0.12 or entity_hits or number_hits or detail_type_hits:
            support_packets.append(packet)
        if score > best_score:
            best_score = score
            best_packet = packet
            best_entities = entity_hits
            best_numbers = number_hits
            best_detail_types = detail_type_hits
            best_matched_entities = matched_entities
            best_matched_numbers = matched_numbers
            best_matched_detail_types = matched_detail_types
    if support_packets:
        best_matched_entities = [
            entity
            for entity in expected_entities
            if any(_contains_phrase(packet["text"], entity) for packet in support_packets)
        ]
        best_matched_numbers = [
            _expected_number_label(number)
            for number in expected_numbers
            if any(_contains_number(packet["text"], number) for packet in support_packets)
        ]
        best_matched_detail_types = [
            detail_type
            for detail_type in expected_detail_types
            if any(detail_type in set(packet.get("detail_types", [])) for packet in support_packets)
        ]
        best_entities = len(best_matched_entities)
        best_numbers = len(best_matched_numbers)
        best_detail_types = len(best_matched_detail_types)
    missing_entities = [entity for entity in expected_entities if entity not in set(best_matched_entities)]
    missing_numbers = [
        _expected_number_label(number)
        for number in expected_numbers
        if _expected_number_label(number) not in set(best_matched_numbers)
    ]
    missing_detail_types = [
        detail_type for detail_type in expected_detail_types if detail_type not in set(best_matched_detail_types)
    ]
    return {
        "claim_id": str(claim.get("claim_id") or ""),
        "section": section,
        "has_candidate": bool(support_packets) or (best_packet is not None and best_score > 0.12),
        "best_score": round(best_score, 3),
        "best_packet_id": best_packet.get("finding_id", "") if best_packet else "",
        "supporting_packet_ids": _supporting_packet_ids(support_packets),
        "expected_entities": len(expected_entities),
        "matched_entities": best_entities,
        "matched_entity_values": best_matched_entities,
        "missing_entity_values": missing_entities,
        "expected_numbers": len(expected_numbers),
        "matched_numbers": best_numbers,
        "matched_number_values": best_matched_numbers,
        "missing_number_values": missing_numbers,
        "expected_detail_types": len(expected_detail_types),
        "matched_detail_types": best_detail_types,
        "matched_detail_type_values": best_matched_detail_types,
        "missing_detail_type_values": missing_detail_types,
    }


def _supporting_packet_ids(packets: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for idx, packet in enumerate(packets, start=1):
        packet_id = str(packet.get("finding_id") or "").strip() or f"support-packet-{idx}"
        if packet_id in seen:
            continue
        seen.add(packet_id)
        out.append(packet_id)
        if len(out) >= 8:
            break
    return out


def _expected_number_label(number: dict[str, Any]) -> str:
    label = str(number.get("label") or "").strip()
    if label:
        return label
    value = str(number.get("value") or "").strip()
    unit = str(number.get("unit") or "").strip()
    return " ".join(part for part in (value, unit) if part) or "expected_number"


def _claim_requirement_gaps(claim_matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    for match in claim_matches:
        missing_entities = [str(item) for item in match.get("missing_entity_values", []) if str(item).strip()]
        missing_numbers = [str(item) for item in match.get("missing_number_values", []) if str(item).strip()]
        missing_detail_types = [
            str(item) for item in match.get("missing_detail_type_values", []) if str(item).strip()
        ]
        candidate_missing = not bool(match.get("has_candidate"))
        if not (candidate_missing or missing_entities or missing_numbers or missing_detail_types):
            continue
        gaps.append(
            {
                "claim_id": str(match.get("claim_id") or ""),
                "section": str(match.get("section") or ""),
                "best_packet_id": str(match.get("best_packet_id") or ""),
                "candidate_missing": candidate_missing,
                "missing_entities": missing_entities,
                "missing_numbers": missing_numbers,
                "missing_detail_types": missing_detail_types,
            }
        )
    return gaps


def _match_forbidden_claims(
    forbidden_claims: Any,
    packets: list[dict[str, Any]],
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    if not isinstance(forbidden_claims, list):
        return []
    violations: list[dict[str, Any]] = []
    usable_packets = [packet for packet in packets if packet.get("text")]
    for raw_claim in forbidden_claims:
        claim = _forbidden_claim_target_text(str(raw_claim or ""))
        if not claim:
            continue
        best_packet: dict[str, Any] | None = None
        best_score = 0.0
        for packet in usable_packets:
            score = _forbidden_claim_score(claim, str(packet.get("text") or ""))
            if score > best_score:
                best_score = score
                best_packet = packet
        if best_packet is not None and best_score >= threshold:
            violations.append(
                {
                    "forbidden_claim": str(raw_claim),
                    "target_text": claim,
                    "best_score": round(best_score, 3),
                    "best_packet_id": best_packet.get("finding_id", ""),
                    "section": best_packet.get("section_label", "unknown"),
                    "statement": best_packet.get("statement", ""),
                }
            )
    return violations


def _forbidden_claim_target_text(value: str) -> str:
    text = _clean_text(value)
    text = re.sub(r"^\s*do\s+not\s+claim\s+(?:that\s+)?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*do\s+not\s+describe\s+(?:this\s+as\s+)?", "", text, flags=re.IGNORECASE)
    return _clean_text(text)


def _forbidden_claim_score(forbidden: str, candidate: str) -> float:
    forbidden_tokens = set(TOKEN_RE.findall(forbidden.lower()))
    candidate_tokens = set(TOKEN_RE.findall(candidate.lower()))
    if not forbidden_tokens or not candidate_tokens:
        return 0.0
    overlap = len(forbidden_tokens & candidate_tokens) / len(forbidden_tokens)
    # Require at least one content-bearing token that usually distinguishes the prohibited claim.
    content_tokens = {
        token for token in forbidden_tokens
        if token not in {"the", "a", "an", "and", "or", "study", "paper", "this", "as", "is", "are", "was", "were"}
    }
    if content_tokens and not (content_tokens & candidate_tokens):
        return 0.0
    return overlap


def _packet_schema_gaps(packets: list[dict[str, Any]]) -> list[str]:
    gaps: list[str] = []
    if not packets:
        return ["no evidence packets"]
    if not any(packet["section_label"] in VALID_SECTIONS for packet in packets):
        gaps.append("no known section_label values")
    if not any(packet["evidence_refs"] for packet in packets):
        gaps.append("no evidence_refs")
    if not any(packet["statement"] for packet in packets):
        gaps.append("no statements")
    if not any(packet["modality"] in VALID_MODALITIES for packet in packets):
        gaps.append("no recognized modality values")
    return gaps


def _clean_string_list(value: Any, *, max_items: int = 16) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in _as_list(value):
        text = _clean_text(item)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) >= max_items:
            break
    return out


def _clean_missing_focus_slots(value: Any, *, max_items: int = 16) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for item in value if isinstance(value, list) else []:
        if not isinstance(item, dict):
            continue
        slot_key = _clean_text(item.get("slot_key"))
        label = _clean_text(item.get("label")) or slot_key
        reason = _clean_text(item.get("reason"))
        if not slot_key and not label:
            continue
        row = {"slot_key": slot_key, "label": label}
        if reason:
            row["reason"] = reason
        out.append(row)
        if len(out) >= max_items:
            break
    return out


def _empty_packet_coverage() -> dict[str, Any]:
    return {
        "available": False,
        "packet_total": 0,
        "usable_packets": 0,
        "usable_packet_rate": 0.0,
        "sections_present": [],
        "missing_core_sections": [],
        "by_section": {},
        "by_modality": {},
        "by_detail_type": {},
        "cross_modal_packet_count": 0,
        "typed_packet_count": 0,
        "quality_flags": [],
    }


def _clean_packet_coverage(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return _empty_packet_coverage()
    return {
        "available": True,
        "paper_type": _clean_text(value.get("paper_type")),
        "packet_total": _safe_int(value.get("packet_total")),
        "usable_packets": _safe_int(value.get("usable_packets")),
        "usable_packet_rate": _safe_float(value.get("usable_packet_rate")),
        "sections_present": _clean_string_list(value.get("sections_present"), max_items=8),
        "missing_core_sections": _clean_string_list(value.get("missing_core_sections"), max_items=8),
        "by_section": _clean_count_map(value.get("by_section"), valid_keys=VALID_SECTIONS),
        "by_modality": _clean_count_map(value.get("by_modality"), valid_keys=VALID_MODALITIES),
        "by_detail_type": _clean_count_map(value.get("by_detail_type")),
        "cross_modal_packet_count": _safe_int(value.get("cross_modal_packet_count")),
        "typed_packet_count": _safe_int(value.get("typed_packet_count")),
        "critical_missing_focus_slots": _clean_missing_focus_slots(value.get("critical_missing_focus_slots")),
        "quality_flags": _clean_string_list(value.get("quality_flags"), max_items=16),
    }


def _clean_count_map(value: Any, *, valid_keys: set[str] | None = None) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, int] = {}
    for raw_key, raw_count in value.items():
        key = _clean_text(raw_key).lower()
        if not key or (valid_keys is not None and key not in valid_keys):
            continue
        out[key] = _safe_int(raw_count)
    return out


def _safe_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except Exception:
        return 0


def _safe_float(value: Any) -> float:
    try:
        parsed = float(value)
    except Exception:
        return 0.0
    if parsed < 0:
        return 0.0
    if parsed > 1:
        return 1.0
    return round(parsed, 3)


def _token_overlap(left: str, right: str) -> float:
    left_tokens = set(TOKEN_RE.findall(left.lower()))
    right_tokens = set(TOKEN_RE.findall(right.lower()))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _contains_phrase(text: str, phrase: str) -> bool:
    normalized_text = " ".join(TOKEN_RE.findall(text.lower()))
    normalized_phrase = " ".join(TOKEN_RE.findall(phrase.lower()))
    if normalized_phrase and normalized_phrase in normalized_text:
        return True
    return any(
        f" {alias} " in f" {normalized_text} "
        for alias in ENTITY_ALIASES.get(normalized_phrase, [])
    )


def _contains_number(text: str, expected: Any) -> bool:
    expected_value = expected.get("value") if isinstance(expected, dict) else expected
    expected_unit = str(expected.get("unit") or "").strip() if isinstance(expected, dict) else ""
    comparator = str(expected.get("comparator") or "=").strip() if isinstance(expected, dict) else "="
    try:
        target = float(expected_value)
    except Exception:
        return False
    for match in NUMBER_RE.finditer(text):
        raw = match.group(0)
        try:
            observed = float(raw)
        except Exception:
            continue
        if expected_unit and not _unit_matches_number_context(text, match.start(), match.end(), expected_unit):
            continue
        observed_comparator = _number_comparator_prefix(text, match.start())
        if _number_matches_comparator(observed, target, comparator, observed_comparator):
            return True
    return False


def _number_matches_comparator(
    observed: float,
    target: float,
    comparator: str,
    observed_comparator: str,
) -> bool:
    normalized = comparator if comparator in {"=", "~", "<", "<=", ">", ">="} else "="
    exact_tolerance = max(1e-9, abs(target) * 0.001)
    approx_tolerance = max(exact_tolerance, abs(target) * 0.05)
    if normalized == "=":
        return abs(observed - target) <= exact_tolerance
    if normalized == "~":
        return abs(observed - target) <= approx_tolerance
    if normalized == "<":
        return observed < target or (abs(observed - target) <= exact_tolerance and observed_comparator in {"<", "<="})
    if normalized == "<=":
        return observed < target or abs(observed - target) <= exact_tolerance
    if normalized == ">":
        return observed > target or (abs(observed - target) <= exact_tolerance and observed_comparator in {">", ">="})
    if normalized == ">=":
        return observed > target or abs(observed - target) <= exact_tolerance
    return False


def _number_comparator_prefix(text: str, number_start: int) -> str:
    prefix = text[max(0, number_start - 12):number_start].lower()
    compact = " ".join(prefix.split())
    if re.search(r"(?:≤|<=|less than or equal to|not greater than)\s*$", compact):
        return "<="
    if re.search(r"(?:≥|>=|greater than or equal to|not less than)\s*$", compact):
        return ">="
    if re.search(r"(?:<|less than|below|under)\s*$", compact):
        return "<"
    if re.search(r"(?:>|greater than|above|over)\s*$", compact):
        return ">"
    if re.search(r"(?:=|equals?|equal to|was|is)\s*$", compact):
        return "="
    if re.search(r"(?:~|≈|approximately|approx\.?|about|around)\s*$", compact):
        return "~"
    return ""


def _unit_matches_number_context(text: str, number_start: int, number_end: int, expected_unit: str) -> bool:
    if _unit_follows_number(text, number_end, expected_unit):
        return True
    if _unit_precedes_number(text, number_start, expected_unit):
        return True
    return False


def _unit_follows_number(text: str, number_end: int, expected_unit: str) -> bool:
    tail = text[number_end:number_end + 48]
    if not tail:
        return False
    for alias in _unit_aliases(expected_unit):
        if _tail_starts_with_unit(tail, alias):
            return True
    return False


def _unit_precedes_number(text: str, number_start: int, expected_unit: str) -> bool:
    prefix = text[max(0, number_start - 48):number_start]
    if not prefix:
        return False
    for alias in _unit_aliases(expected_unit):
        if _prefix_ends_with_unit(prefix, alias):
            return True
    return False


def _tail_starts_with_unit(tail: str, alias: str) -> bool:
    normalized_alias = _normalize_unit_text(alias)
    if not normalized_alias:
        return False
    normalized_tail = _normalize_unit_context(tail)
    if normalized_alias == "%":
        return normalized_tail.startswith("%") or normalized_tail.startswith("percent")
    pattern = re.escape(normalized_alias).replace(r"\ ", r"\s+")
    return bool(re.match(rf"^\s*(?:[-–—]\s*)?{pattern}(?=\b|/|$)", normalized_tail))


def _prefix_ends_with_unit(prefix: str, alias: str) -> bool:
    normalized_alias = _normalize_unit_text(alias)
    if not normalized_alias:
        return False
    normalized_prefix = _normalize_unit_context(prefix).rstrip(" =:<>()[]{}")
    if normalized_alias == "%":
        return normalized_prefix.endswith("%") or normalized_prefix.endswith("percent")
    pattern = re.escape(normalized_alias).replace(r"\ ", r"\s+")
    return bool(re.search(rf"(?:^|\b){pattern}\s*(?:=|:|of|was|were|is|are)?\s*$", normalized_prefix))


def _unit_aliases(unit: str) -> set[str]:
    normalized = _normalize_unit_text(unit)
    aliases = {normalized} if normalized else set()
    alias_map = {
        "mg": {"mg", "milligram", "milligrams"},
        "mcg": {"mcg", "ug", "microgram", "micrograms"},
        "ug": {"mcg", "ug", "microgram", "micrograms"},
        "g": {"g", "gram", "grams"},
        "kg": {"kg", "kilogram", "kilograms"},
        "mg/kg": {"mg/kg", "mg per kg", "milligram/kg", "milligrams/kg", "milligrams per kilogram"},
        "ml": {"ml", "milliliter", "milliliters", "millilitre", "millilitres"},
        "l": {"l", "liter", "liters", "litre", "litres"},
        "iu": {"iu", "international unit", "international units"},
        "unit": {"unit", "units"},
        "units": {"unit", "units"},
        "%": {"%", "percent", "percentage"},
        "week": {"week", "weeks", "wk", "wks"},
        "weeks": {"week", "weeks", "wk", "wks"},
        "month": {"month", "months", "mo"},
        "months": {"month", "months", "mo"},
        "day": {"day", "days"},
        "days": {"day", "days"},
        "um": {"um", "u m", "micromolar", "micromol", "micromoles"},
        "mg/dl": {"mg/dl", "mg per dl", "milligrams/dl", "milligrams per deciliter", "milligrams per decilitre"},
        "mg/kg/day": {
            "mg/kg/day",
            "mg per kg per day",
            "mg/kg/d",
            "milligrams/kg/day",
            "milligrams per kilogram per day",
        },
        "cfu/ml": {"cfu/ml", "cfu per ml", "cfu/ml.", "colony-forming units/ml", "colony forming units/ml"},
        "k/cumm": {"k/cumm", "k/cu mm", "k/cubic mm", "thousand/cumm"},
        "odds_ratio": {"odds ratio", "or"},
        "d_prime": {"d prime", "d'", "d\u2032", "d-prime"},
        "proportion": {"proportion", "frequency", "allele frequency"},
        "fold": {"fold", "fold-change", "fold change", "times"},
        "participant": {"participant", "participants", "subject", "subjects", "adult", "adults", "patient", "patients"},
        "participants": {"participant", "participants", "subject", "subjects", "adult", "adults", "patient", "patients"},
        "subject": {"participant", "participants", "subject", "subjects", "adult", "adults", "patient", "patients"},
        "subjects": {"participant", "participants", "subject", "subjects", "adult", "adults", "patient", "patients"},
        "adult": {"participant", "participants", "subject", "subjects", "adult", "adults", "patient", "patients"},
        "adults": {"participant", "participants", "subject", "subjects", "adult", "adults", "patient", "patients"},
    }
    aliases.update(alias_map.get(normalized, set()))
    if normalized.endswith("s") and len(normalized) > 2:
        aliases.add(normalized[:-1])
    return {_normalize_unit_text(alias) for alias in aliases if _normalize_unit_text(alias)}


def _normalize_unit_text(value: str) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("µ", "u").replace("μ", "u")
    text = text.replace("micrograms", "ug").replace("microgram", "ug")
    text = text.replace("mcg", "ug")
    text = re.sub(r"\s+per\s+", "/", text)
    text = re.sub(r"\s*/\s*", "/", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,:;()[]{}")


def _normalize_unit_context(value: str) -> str:
    text = _normalize_unit_text(value)
    return text.lstrip(" -–—")


def _rate(numerator: int, denominator: int) -> float:
    return round((numerator / denominator) if denominator else 1.0, 3)


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _rate_arg(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not a number") from exc
    if parsed < 0.0 or parsed > 1.0:
        raise argparse.ArgumentTypeError(f"{value!r} must be between 0.0 and 1.0")
    return parsed


def _format_score_summary(comparison: dict[str, Any], *, max_gaps: int) -> str:
    basis = comparison.get("benchmark_content_score_basis")
    if not isinstance(basis, dict):
        basis = {}
    components = basis.get("components")
    if not isinstance(components, dict):
        components = {}

    lines = [
        f"case_id: {comparison.get('case_id', '')}",
        f"compatible: {str(bool(comparison.get('compatible'))).lower()}",
        f"overall_benchmark_score: {comparison.get('overall_benchmark_score')}",
        f"benchmark_content_score: {comparison.get('benchmark_content_score')}",
        f"matched_slots: {basis.get('matched_slots')}",
        f"expected_slots: {basis.get('expected_slots')}",
        f"extra_content_penalized: {str(bool(basis.get('extra_content_penalized'))).lower()}",
    ]

    component_labels = [
        ("critical_claim_candidates", "critical_claim_candidates"),
        ("expected_entities", "expected_entities"),
        ("expected_numbers", "expected_numbers"),
        ("expected_detail_types", "expected_detail_types"),
        ("required_sections", "required_sections"),
    ]
    component_lines: list[str] = []
    for key, label in component_labels:
        item = components.get(key)
        if not isinstance(item, dict):
            continue
        component_lines.append(f"- {label}: {item.get('matched')}/{item.get('expected')}")
    if component_lines:
        lines.append("components:")
        lines.extend(component_lines)

    failure_reasons = [str(item) for item in comparison.get("failure_reasons", []) if str(item).strip()]
    if failure_reasons:
        lines.append("failure_reasons:")
        lines.extend(f"- {item}" for item in failure_reasons)

    gap_lines = _format_claim_requirement_gaps(comparison.get("claim_requirement_gaps"), max_gaps=max_gaps)
    if gap_lines:
        lines.append("benchmark_gaps:")
        lines.extend(gap_lines)
    stage_diagnostics = comparison.get("artifact_stage_diagnostics")
    if isinstance(stage_diagnostics, dict):
        stage_lines = _format_stage_diagnostics(stage_diagnostics)
        if stage_lines:
            lines.append("stage_diagnostics:")
            lines.extend(stage_lines)
    return "\n".join(lines)


def _format_stage_diagnostics(value: dict[str, Any]) -> list[str]:
    rows: list[str] = []
    stage_counts = value.get("stage_presence_counts")
    if isinstance(stage_counts, dict):
        for stage in ("extracted_text", "evidence_packets", "synthesis_inputs", "final_report"):
            item = stage_counts.get(stage)
            if isinstance(item, dict):
                rows.append(f"- {stage}: {item.get('present')}/{item.get('total_expected_items')} expected items present")
    failure_counts = value.get("failure_point_counts")
    if isinstance(failure_counts, dict) and failure_counts:
        rows.append("- failure_points:")
        for key in (
            "absent_from_saved_artifact",
            "dropped_before_evidence_packetization",
            "dropped_before_synthesis_selection",
            "dropped_during_final_synthesis",
            "present_in_final_but_unmatched",
            "unknown",
        ):
            if key in failure_counts:
                rows.append(f"  - {key}: {failure_counts[key]}")
    examples = value.get("missing_item_examples")
    if isinstance(examples, list) and examples:
        rows.append("- examples:")
        for item in examples[:5]:
            if not isinstance(item, dict):
                continue
            present_in = ", ".join(item.get("present_in", [])) if isinstance(item.get("present_in"), list) else ""
            rows.append(
                "  - "
                + f"{item.get('claim_id', 'claim')} {item.get('item_type', 'item')}={item.get('term', '')} "
                + f"failure={item.get('failure_point', 'unknown')} present_in={present_in or 'none'}"
            )
            reason = _clean_text(item.get("diagnostic_reason"))
            if reason:
                rows.append(f"    why: {reason}")
            first_match = _first_stage_match(
                item.get("stage_matches"),
                failure_point=str(item.get("failure_point") or ""),
            )
            if first_match:
                rows.append(f"    trace: {first_match.get('path', '')}: {first_match.get('snippet', '')}")
            nearest = _first_nearest_candidate(item.get("nearest_stage_candidates"))
            if nearest and not first_match:
                rows.append(
                    "    nearest: "
                    + f"{nearest.get('stage', '')} {nearest.get('path', '')}: {nearest.get('snippet', '')}"
                )
    return rows


def _format_claim_requirement_gaps(value: Any, *, max_gaps: int) -> list[str]:
    if max_gaps <= 0 or not isinstance(value, list):
        return []
    rows: list[str] = []
    for gap in value:
        if not isinstance(gap, dict):
            continue
        parts: list[str] = []
        if gap.get("candidate_missing"):
            parts.append("candidate missing")
        for key, label in [
            ("missing_entities", "missing entities"),
            ("missing_numbers", "missing numbers"),
            ("missing_detail_types", "missing detail types"),
        ]:
            items = [str(item) for item in gap.get(key, []) if str(item).strip()] if isinstance(gap.get(key), list) else []
            if items:
                parts.append(f"{label}: {', '.join(items[:4])}")
        if not parts:
            continue
        claim_id = str(gap.get("claim_id") or "claim").strip()
        section = str(gap.get("section") or "").strip()
        prefix = claim_id + (f" ({section})" if section else "")
        rows.append(f"- {prefix}: {'; '.join(parts)}")
        if len(rows) >= max_gaps:
            break
    remaining = len([gap for gap in value if isinstance(gap, dict)]) - len(rows)
    if remaining > 0 and rows:
        rows.append(f"- {remaining} additional benchmark gap(s) not shown")
    return rows


def _build_artifact_stage_diagnostics(
    evidence_payload: Any,
    gold_standard: dict[str, Any],
    comparison: dict[str, Any],
) -> dict[str, Any]:
    expected_items = _stage_expected_items(gold_standard, comparison)
    stage_entries = _stage_entries(evidence_payload)
    claim_contexts = _stage_claim_contexts(comparison)
    rows: list[dict[str, Any]] = []
    stage_counts = {
        stage: {"present": 0, "total_expected_items": len(expected_items)}
        for stage in ("extracted_text", "evidence_packets", "synthesis_inputs", "final_report")
    }
    for item in expected_items:
        present_in: list[str] = []
        stage_matches: dict[str, list[dict[str, str]]] = {}
        for stage, entries in stage_entries.items():
            matches = _stage_item_matches(item, entries)
            if matches:
                present_in.append(stage)
                stage_counts[stage]["present"] += 1
                stage_matches[stage] = matches
        failure_point = _stage_failure_point(present_in)
        claim_context = claim_contexts.get(str(item.get("claim_id") or ""), {})
        rows.append(
            {
                **item,
                "present_in": present_in,
                "failure_point": failure_point,
                "source_visibility": _stage_source_visibility(
                    failure_point=failure_point,
                    present_in=present_in,
                    item=item,
                    stage_entries=stage_entries,
                ),
                "diagnostic_reason": _stage_diagnostic_reason(
                    failure_point=failure_point,
                    item=item,
                    claim_context=claim_context,
                    stage_matches=stage_matches,
                ),
                "improvement_lane": _stage_improvement_lane(failure_point),
                "next_probe": _stage_next_probe(failure_point),
                "claim_match_context": claim_context,
                "stage_matches": stage_matches,
                "nearest_stage_candidates": _nearest_stage_candidates(item, stage_entries, stage_matches),
            }
        )
    return {
        "expected_item_count": len(expected_items),
        "stage_presence_counts": stage_counts,
        "failure_point_counts": _count_by(rows, "failure_point"),
        "item_type_counts": _count_by(rows, "item_type"),
        "source_visibility_counts": _count_by_nested_key(rows, "source_visibility", "classification"),
        "failure_point_by_item_type": _nested_count_by(rows, "item_type", "failure_point"),
        "failure_point_by_claim": _nested_count_by(rows, "claim_id", "failure_point"),
        "missing_items": rows,
        "missing_item_examples": rows[:20],
        "notes": [
            "extracted_text searches source excerpts, verbatim text, OCR, captions, legends, and raw extraction-like fields.",
            "evidence_packets searches modality findings, scientific details, and evidence-packet-like fields.",
            "synthesis_inputs searches synthesis evidence plans, focus slots, selected details, and supporting candidates.",
            "final_report searches executive summary, section summaries, key findings, overview, conclusion, and final-report-like fields.",
        ],
    }


def _count_by(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get(field) or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return counts


def _nested_count_by(rows: list[dict[str, Any]], outer_field: str, inner_field: str) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        outer = str(row.get(outer_field) or "unknown")
        inner = str(row.get(inner_field) or "unknown")
        counts.setdefault(outer, {})
        counts[outer][inner] = counts[outer].get(inner, 0) + 1
    return counts


def _count_by_nested_key(rows: list[dict[str, Any]], field: str, nested_key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        nested = row.get(field)
        key = "unknown"
        if isinstance(nested, dict):
            key = str(nested.get(nested_key) or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return counts


def _stage_expected_items(gold_standard: dict[str, Any], comparison: dict[str, Any]) -> list[dict[str, Any]]:
    claims = {
        str(claim.get("claim_id") or ""): claim
        for claim in gold_standard.get("critical_claims", [])
        if isinstance(claim, dict)
    }
    rows: list[dict[str, Any]] = []
    for gap in comparison.get("claim_requirement_gaps", []):
        if not isinstance(gap, dict):
            continue
        claim_id = str(gap.get("claim_id") or "")
        claim = claims.get(claim_id, {})
        section = str(gap.get("section") or claim.get("section") or "")
        if gap.get("candidate_missing"):
            rows.append(
                {
                    "claim_id": claim_id,
                    "section": section,
                    "item_type": "claim_candidate",
                    "term": _clean_text(claim.get("claim") or claim_id),
                    "search_terms": _search_terms_for_text(claim.get("claim") or claim_id),
                    "claim_text": _clean_text(claim.get("claim")),
                }
            )
        for entity in gap.get("missing_entities", []) if isinstance(gap.get("missing_entities"), list) else []:
            rows.append(
                {
                    "claim_id": claim_id,
                    "section": section,
                    "item_type": "entity",
                    "term": _clean_text(entity),
                    "search_terms": _search_terms_for_text(entity),
                    "claim_text": _clean_text(claim.get("claim")),
                }
            )
        missing_number_labels = gap.get("missing_numbers", []) if isinstance(gap.get("missing_numbers"), list) else []
        number_lookup = _claim_number_lookup(claim)
        for label in missing_number_labels:
            rows.append(
                {
                    "claim_id": claim_id,
                    "section": section,
                    "item_type": "number",
                    "term": _clean_text(label),
                    "search_terms": _number_search_terms(label, number_lookup.get(str(label))),
                    "claim_text": _clean_text(claim.get("claim")),
                }
            )
        for detail_type in gap.get("missing_detail_types", []) if isinstance(gap.get("missing_detail_types"), list) else []:
            rows.append(
                {
                    "claim_id": claim_id,
                    "section": section,
                    "item_type": "detail_type",
                    "term": _clean_text(detail_type),
                    "search_terms": _search_terms_for_text(str(detail_type).replace("_", " ")),
                    "claim_text": _clean_text(claim.get("claim")),
                }
            )
    return [row for row in rows if row.get("search_terms")]


def _claim_number_lookup(claim: dict[str, Any]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    numbers = claim.get("expected_numbers")
    if not isinstance(numbers, list):
        return lookup
    for number in numbers:
        if not isinstance(number, dict):
            continue
        label = str(number.get("label") or "").strip()
        if label:
            lookup[label] = number
    return lookup


def _number_search_terms(label: Any, number: dict[str, Any] | None) -> list[str]:
    terms = _search_terms_for_text(str(label).replace("_", " "))
    if isinstance(number, dict):
        value = number.get("value")
        if value is not None:
            terms.append(_clean_text(value))
        unit = _clean_text(number.get("unit"))
        if value is not None and unit:
            terms.append(f"{_clean_text(value)} {unit}")
    return [term for term in dict.fromkeys(terms) if term]


def _search_terms_for_text(value: Any) -> list[str]:
    text = _clean_text(value)
    if not text:
        return []
    terms = [text]
    if "_" in text:
        terms.append(text.replace("_", " "))
    normalized_dash = re.sub(r"[-–—]+", " ", text)
    if normalized_dash != text:
        terms.append(_clean_text(normalized_dash))
    tokens = TOKEN_RE.findall(text.lower())
    if 2 <= len(tokens) <= 6:
        terms.append(" ".join(tokens))
    return [term for term in dict.fromkeys(terms) if term]


def _stage_item_present(item: dict[str, Any], stage_text: str) -> bool:
    return any(_term_present_in_text(term, stage_text) for term in item.get("search_terms", []) if str(term).strip())


def _stage_item_matches(
    item: dict[str, Any],
    entries: list[tuple[tuple[str, ...], str]],
    *,
    max_matches: int = 3,
) -> list[dict[str, str]]:
    matches: list[dict[str, str]] = []
    for path, text in entries:
        for term in item.get("search_terms", []):
            needle = str(term).strip()
            if not needle or not _term_present_in_text(needle, text):
                continue
            matches.append(
                {
                    "path": ".".join(path),
                    "term": needle,
                    "snippet": _match_snippet(text, needle),
                }
            )
            break
        if len(matches) >= max_matches:
            break
    return matches


def _stage_source_visibility(
    *,
    failure_point: str,
    present_in: list[str],
    item: dict[str, Any],
    stage_entries: dict[str, list[tuple[tuple[str, ...], str]]],
) -> dict[str, Any]:
    if present_in:
        return {
            "classification": "exact_present",
            "term_score": 1.0,
            "stage": present_in[0],
            "reason": "Exact or ordered-token diagnostic match was found in saved artifact text.",
        }
    best_stage = ""
    best_path = ""
    best_snippet = ""
    best_score = 0.0
    for stage, entries in stage_entries.items():
        for path, text in entries:
            score = _stage_term_candidate_score(item, text)
            if score > best_score:
                best_score = score
                best_stage = stage
                best_path = ".".join(path)
                best_snippet = text[:180]
    classification = _source_visibility_classification(best_score)
    reason = {
        "near_term_candidate": (
            "No exact diagnostic match, but a high-overlap term-level candidate exists. "
            "Inspect normalization, aliases, symbols, OCR, or benchmark wording before changing extraction."
        ),
        "weak_term_candidate": (
            "No exact diagnostic match; only a weak term-level candidate exists. "
            "Treat as possible source visibility issue until source chunks are inspected."
        ),
        "no_term_candidate": (
            "No exact diagnostic match or meaningful term-level candidate was found in saved artifact text."
        ),
    }.get(classification, "")
    row: dict[str, Any] = {
        "classification": classification,
        "term_score": round(best_score, 3),
        "reason": reason,
    }
    if best_stage:
        row.update({"stage": best_stage, "path": best_path, "snippet": best_snippet})
    if failure_point:
        row["failure_point"] = failure_point
    return row


def _source_visibility_classification(score: float) -> str:
    if score >= 0.74:
        return "near_term_candidate"
    if score >= 0.34:
        return "weak_term_candidate"
    return "no_term_candidate"


def _stage_term_candidate_score(item: dict[str, Any], text: str) -> float:
    terms = [
        _clean_text(term)
        for term in item.get("search_terms", [])
        if _clean_text(term)
    ]
    if not terms:
        terms = [_clean_text(item.get("term"))]
    scores = [_token_containment(term, text) for term in terms if term]
    return max(scores) if scores else 0.0


def _first_stage_match(value: Any, *, failure_point: str = "") -> dict[str, str] | None:
    if not isinstance(value, dict):
        return None
    stage_order = _failure_point_trace_stage_order(failure_point)
    for stage in stage_order:
        matches = value.get(stage)
        if isinstance(matches, list) and matches and isinstance(matches[0], dict):
            return {str(key): str(val) for key, val in matches[0].items()}
    return None


def _failure_point_trace_stage_order(failure_point: str) -> tuple[str, ...]:
    return {
        "dropped_before_evidence_packetization": (
            "extracted_text",
            "evidence_packets",
            "synthesis_inputs",
            "final_report",
        ),
        "dropped_before_synthesis_selection": (
            "evidence_packets",
            "extracted_text",
            "synthesis_inputs",
            "final_report",
        ),
        "dropped_during_final_synthesis": (
            "synthesis_inputs",
            "evidence_packets",
            "extracted_text",
            "final_report",
        ),
        "present_in_final_but_unmatched": (
            "final_report",
            "synthesis_inputs",
            "evidence_packets",
            "extracted_text",
        ),
    }.get(
        failure_point,
        ("extracted_text", "evidence_packets", "synthesis_inputs", "final_report"),
    )


def _first_nearest_candidate(value: Any) -> dict[str, str] | None:
    if not isinstance(value, dict):
        return None
    best: dict[str, str] | None = None
    best_score = -1.0
    for stage in ("extracted_text", "evidence_packets", "synthesis_inputs", "final_report"):
        candidates = value.get(stage)
        if isinstance(candidates, list) and candidates and isinstance(candidates[0], dict):
            row = {str(key): str(val) for key, val in candidates[0].items()}
            row["stage"] = stage
            try:
                score = float(row.get("score") or 0)
            except Exception:
                score = 0.0
            if score > best_score:
                best_score = score
                best = row
    return best


def _stage_claim_contexts(comparison: dict[str, Any]) -> dict[str, dict[str, Any]]:
    gaps = {
        str(gap.get("claim_id") or ""): gap
        for gap in comparison.get("claim_requirement_gaps", [])
        if isinstance(gap, dict)
    }
    contexts: dict[str, dict[str, Any]] = {}
    for match in comparison.get("claim_matches", []):
        if not isinstance(match, dict):
            continue
        claim_id = str(match.get("claim_id") or "")
        if not claim_id:
            continue
        gap = gaps.get(claim_id, {})
        contexts[claim_id] = {
            "section": str(match.get("section") or gap.get("section") or ""),
            "has_candidate": bool(match.get("has_candidate")),
            "best_score": match.get("best_score", 0),
            "best_packet_id": str(match.get("best_packet_id") or ""),
            "supporting_packet_ids": [
                str(item)
                for item in match.get("supporting_packet_ids", [])
                if str(item).strip()
            ][:8],
            "candidate_missing": bool(gap.get("candidate_missing")),
            "missing_entities": [
                str(item) for item in gap.get("missing_entities", []) if str(item).strip()
            ] if isinstance(gap.get("missing_entities"), list) else [],
            "missing_numbers": [
                str(item) for item in gap.get("missing_numbers", []) if str(item).strip()
            ] if isinstance(gap.get("missing_numbers"), list) else [],
            "missing_detail_types": [
                str(item) for item in gap.get("missing_detail_types", []) if str(item).strip()
            ] if isinstance(gap.get("missing_detail_types"), list) else [],
        }
    return contexts


def _stage_diagnostic_reason(
    *,
    failure_point: str,
    item: dict[str, Any],
    claim_context: dict[str, Any],
    stage_matches: dict[str, list[dict[str, str]]],
) -> str:
    term = _clean_text(item.get("term")) or "expected item"
    item_type = _clean_text(item.get("item_type")) or "item"
    claim_id = _clean_text(item.get("claim_id")) or "claim"
    if failure_point == "absent_from_saved_artifact":
        return (
            f"No exact diagnostic match for {item_type} `{term}` was found in saved extracted text, "
            "evidence packets, synthesis inputs, or final-report fields."
        )
    if failure_point == "dropped_before_evidence_packetization":
        return (
            f"`{term}` appears in extracted/source-like text but not in evidence-packet-like fields, "
            "so the loss occurred after extraction and before packetized evidence was available to scoring."
        )
    if failure_point == "dropped_before_synthesis_selection":
        packet_path = _first_match_path(stage_matches.get("evidence_packets"))
        detail = f" at {packet_path}" if packet_path else ""
        return (
            f"`{term}` appears in an evidence packet{detail} but not in synthesis-input fields, "
            "so selection/ranking for synthesis likely dropped it."
        )
    if failure_point == "dropped_during_final_synthesis":
        return (
            f"`{term}` reached synthesis-input fields but was not found in final-report fields, "
            "so this is a final synthesis/coverage issue rather than an extraction issue."
        )
    if failure_point == "present_in_final_but_unmatched":
        if claim_context.get("candidate_missing"):
            return (
                f"`{term}` appears in final-report text, but {claim_id} still has no qualifying "
                "same-claim evidence candidate in the scored packet set."
            )
        if claim_context:
            missing_bits = _claim_context_missing_bits(claim_context)
            suffix = f" Missing scored slot(s): {missing_bits}." if missing_bits else ""
            return (
                f"`{term}` appears in final-report text, but the benchmark slot is scored from "
                f"claim-support packets for {claim_id}, not keyword presence alone.{suffix}"
            )
        return (
            f"`{term}` appears in final-report text, but the scored evidence/claim context did not satisfy "
            "the benchmark matcher."
        )
    return f"The diagnostic stage for `{term}` could not be classified."


def _first_match_path(matches: Any) -> str:
    if isinstance(matches, list) and matches and isinstance(matches[0], dict):
        return _clean_text(matches[0].get("path"))
    return ""


def _claim_context_missing_bits(value: dict[str, Any]) -> str:
    parts: list[str] = []
    for key, label in (
        ("missing_entities", "entities"),
        ("missing_numbers", "numbers"),
        ("missing_detail_types", "detail_types"),
    ):
        items = [str(item) for item in value.get(key, []) if str(item).strip()] if isinstance(value.get(key), list) else []
        if items:
            parts.append(f"{label}: {', '.join(items[:4])}")
    return "; ".join(parts)


def _stage_improvement_lane(failure_point: str) -> str:
    return {
        "absent_from_saved_artifact": "source_recall_or_extraction_visibility",
        "dropped_before_evidence_packetization": "evidence_packetization_or_typing",
        "dropped_before_synthesis_selection": "synthesis_evidence_selection_or_ranking",
        "dropped_during_final_synthesis": "final_synthesis_instruction_or_coverage",
        "present_in_final_but_unmatched": "benchmark_matching_or_claim_support_context",
    }.get(failure_point, "unknown")


def _stage_next_probe(failure_point: str) -> str:
    return {
        "absent_from_saved_artifact": (
            "Inspect parser/full-text chunks and media OCR before the saved report artifact to determine "
            "whether source text was never extracted or was extracted but not persisted in the report artifact."
        ),
        "dropped_before_evidence_packetization": (
            "Inspect text/figure/table packet construction and detail typing for the matched source path."
        ),
        "dropped_before_synthesis_selection": (
            "Inspect synthesis evidence-plan selection, per-section quotas, ranking, and focus-slot coverage."
        ),
        "dropped_during_final_synthesis": (
            "Inspect the final synthesis prompt/context budget and section-specific inclusion requirements."
        ),
        "present_in_final_but_unmatched": (
            "Inspect whether the term is attached to the correct claim, section, co-occurring entities/numbers, "
            "and detail type; this may be a matcher/context issue rather than a report omission."
        ),
    }.get(failure_point, "Inspect saved artifact paths manually.")


def _nearest_stage_candidates(
    item: dict[str, Any],
    stage_entries: dict[str, list[tuple[tuple[str, ...], str]]],
    stage_matches: dict[str, list[dict[str, str]]],
    *,
    max_per_stage: int = 2,
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for stage, entries in stage_entries.items():
        if stage in stage_matches:
            continue
        candidates: list[dict[str, Any]] = []
        for path, text in entries:
            score = _stage_candidate_score(item, text)
            if score <= 0:
                continue
            candidates.append(
                {
                    "path": ".".join(path),
                    "score": round(score, 3),
                    "snippet": text[:180],
                }
            )
        candidates.sort(key=lambda row: float(row.get("score") or 0), reverse=True)
        if candidates:
            out[stage] = candidates[:max_per_stage]
    return out


def _stage_candidate_score(item: dict[str, Any], text: str) -> float:
    terms = [
        _clean_text(term)
        for term in item.get("search_terms", [])
        if _clean_text(term)
    ]
    if not terms:
        terms = [_clean_text(item.get("term"))]
    scores = [_token_containment(term, text) for term in terms if term]
    claim_text = _clean_text(item.get("claim_text"))
    if claim_text:
        scores.append(_token_containment(claim_text, text) * 0.5)
    return max(scores) if scores else 0.0


def _token_containment(needle: str, haystack: str) -> float:
    needle_tokens = set(TOKEN_RE.findall(str(needle).lower()))
    haystack_tokens = set(TOKEN_RE.findall(str(haystack).lower()))
    if not needle_tokens or not haystack_tokens:
        return 0.0
    return len(needle_tokens & haystack_tokens) / len(needle_tokens)


def _match_snippet(text: str, term: str, *, radius: int = 80) -> str:
    folded = text.casefold()
    index = folded.find(term.casefold())
    if index < 0:
        tokens = TOKEN_RE.findall(str(term).lower())
        for token in tokens:
            index = folded.find(token)
            if index >= 0:
                break
    if index < 0:
        return text[: radius * 2].strip()
    start = max(0, index - radius)
    end = min(len(text), index + len(term) + radius)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return (prefix + text[start:end].strip() + suffix).replace("\n", " ")


def _term_present_in_text(term: Any, text: str) -> bool:
    needle = str(term or "").strip()
    if not needle:
        return False
    if needle.casefold() in text.casefold():
        return True
    return _ordered_token_match(needle, text)


def _ordered_token_match(term: str, text: str, *, max_gap_tokens: int = 4) -> bool:
    term_tokens = TOKEN_RE.findall(str(term).lower())
    text_tokens = TOKEN_RE.findall(str(text).lower())
    if not term_tokens or not text_tokens:
        return False
    if len(term_tokens) == 1:
        return term_tokens[0] in set(text_tokens)
    first = term_tokens[0]
    for start, token in enumerate(text_tokens):
        if token != first:
            continue
        cursor = start
        ok = True
        for expected in term_tokens[1:]:
            found = False
            for idx in range(cursor + 1, min(len(text_tokens), cursor + max_gap_tokens + 2)):
                if text_tokens[idx] == expected:
                    cursor = idx
                    found = True
                    break
            if not found:
                ok = False
                break
        if ok:
            return True
    return False


def _stage_failure_point(present_in: list[str]) -> str:
    stages = set(present_in)
    if not stages:
        return "absent_from_saved_artifact"
    if "final_report" in stages:
        return "present_in_final_but_unmatched"
    if "synthesis_inputs" in stages:
        return "dropped_during_final_synthesis"
    if "evidence_packets" in stages:
        return "dropped_before_synthesis_selection"
    if "extracted_text" in stages:
        return "dropped_before_evidence_packetization"
    return "unknown"


def _stage_entries(value: Any) -> dict[str, list[tuple[tuple[str, ...], str]]]:
    buckets = {
        "extracted_text": [],
        "evidence_packets": [],
        "synthesis_inputs": [],
        "final_report": [],
    }
    for path, text in _flatten_artifact_strings(value):
        path_text = ".".join(path).casefold()
        if any(token in path_text for token in ("source_excerpt", "verbatim_text", "raw_text", "ocr", "caption", "legend")):
            buckets["extracted_text"].append((path, text))
        if any(token in path_text for token in ("modalities", "finding", "evidence_packet", "scientific_detail")):
            buckets["evidence_packets"].append((path, text))
        if any(token in path_text for token in ("synthesis_evidence_plan", "focus_slots", "supporting_candidates", "selected_detail")):
            buckets["synthesis_inputs"].append((path, text))
        if any(
            token in path_text
            for token in (
                "executive_summary",
                "key_findings",
                "section_snapshot",
                "overview",
                "conclusion",
                "final_report",
                "report_sections",
            )
        ):
            buckets["final_report"].append((path, text))
    return buckets


def _flatten_artifact_strings(value: Any, path: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], str]]:
    rows: list[tuple[tuple[str, ...], str]] = []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return rows
        if text[:1] in {"{", "["}:
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = None
            if parsed is not None:
                return _flatten_artifact_strings(parsed, (*path, "<parsed_json>"))
        if _reference_only_path(path, text):
            return rows
        cleaned = _clean_stage_text(text)
        if cleaned:
            rows.append((path, cleaned))
        return rows
    if isinstance(value, dict):
        for key, item in value.items():
            rows.extend(_flatten_artifact_strings(item, (*path, str(key))))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            rows.extend(_flatten_artifact_strings(item, (*path, str(index))))
    return rows


def _reference_only_path(path: tuple[str, ...], text: str) -> bool:
    leaf = path[-1].casefold() if path else ""
    if leaf in {"evidence", "evidence_refs", "anchor", "source_anchor"}:
        return True
    stripped = text.strip().casefold()
    return bool(re.match(r"^(section|figure|table|supplement):", stripped))


def _clean_stage_text(text: str) -> str:
    without_refs = re.sub(r"\[(?:section|figure|table|supplement):[^\]]+\]", " ", text, flags=re.IGNORECASE)
    return _clean_text(without_refs)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare evidence packets to a final-report gold standard.")
    parser.add_argument("--evidence-json", required=True, type=Path)
    parser.add_argument("--gold-standard", required=True, type=Path)
    parser.add_argument("--min-usable-packet-rate", type=_rate_arg, default=DEFAULT_MIN_USABLE_PACKET_RATE)
    parser.add_argument("--min-section-coverage-rate", type=_rate_arg, default=DEFAULT_MIN_SECTION_COVERAGE_RATE)
    parser.add_argument(
        "--min-critical-claim-candidate-rate",
        type=_rate_arg,
        default=DEFAULT_MIN_CRITICAL_CLAIM_CANDIDATE_RATE,
    )
    parser.add_argument(
        "--min-expected-entity-observability-rate",
        type=_rate_arg,
        default=DEFAULT_MIN_EXPECTED_ENTITY_OBSERVABILITY_RATE,
    )
    parser.add_argument(
        "--min-expected-number-observability-rate",
        type=_rate_arg,
        default=DEFAULT_MIN_EXPECTED_NUMBER_OBSERVABILITY_RATE,
    )
    parser.add_argument(
        "--min-expected-detail-type-observability-rate",
        type=_rate_arg,
        default=DEFAULT_MIN_EXPECTED_DETAIL_TYPE_OBSERVABILITY_RATE,
    )
    parser.add_argument("--forbidden-claim-threshold", type=_rate_arg, default=DEFAULT_FORBIDDEN_CLAIM_THRESHOLD)
    parser.add_argument("--fail-on-incompatible", action="store_true")
    parser.add_argument("--summary", action="store_true", help="Print a concise score summary instead of raw JSON.")
    parser.add_argument("--summary-gaps", type=int, default=5, help="Maximum benchmark gaps to print with --summary.")
    parser.add_argument(
        "--stage-diagnostics",
        action="store_true",
        help="Include artifact stage presence for missing gold-standard content.",
    )
    args = parser.parse_args()
    evidence_payload = json.loads(args.evidence_json.read_text(encoding="utf-8"))
    evidence_packets = evidence_packets_from_payload(evidence_payload)
    evidence_metadata = evidence_metadata_from_payload(evidence_payload)
    gold = load_gold_standard(args.gold_standard)
    comparison = compare_evidence_to_gold(
        evidence_packets,
        gold,
        evidence_metadata=evidence_metadata,
        min_usable_packet_rate=args.min_usable_packet_rate,
        min_section_coverage_rate=args.min_section_coverage_rate,
        min_critical_claim_candidate_rate=args.min_critical_claim_candidate_rate,
        min_expected_entity_observability_rate=args.min_expected_entity_observability_rate,
        min_expected_number_observability_rate=args.min_expected_number_observability_rate,
        min_expected_detail_type_observability_rate=args.min_expected_detail_type_observability_rate,
        forbidden_claim_threshold=args.forbidden_claim_threshold,
    )
    if args.stage_diagnostics:
        comparison["artifact_stage_diagnostics"] = _build_artifact_stage_diagnostics(
            evidence_payload,
            gold,
            comparison,
        )
    if args.summary:
        print(_format_score_summary(comparison, max_gaps=args.summary_gaps))
    else:
        print(json.dumps(comparison, indent=2))
    if args.fail_on_incompatible and not comparison["compatible"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
