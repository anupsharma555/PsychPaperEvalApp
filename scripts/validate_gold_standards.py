from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


VALID_REVIEW_STATUSES = {
    "codex_drafted_needs_review",
    "reviewed_gold_standard",
    "reviewed_reference_available",
}
VALID_SECTIONS = {"introduction", "methods", "results", "discussion", "conclusion"}
VALID_IMPORTANCE = {"P0", "P1", "P2", "P3"}
VALID_EXPECTED_DETAIL_TYPES = {
    "medication_or_therapeutic",
    "dose_schedule",
    "intervention_or_exposure",
    "outcome_measure",
    "adverse_event",
    "model_system",
    "assay_readout",
    "statistical_result",
    "data_source_or_design",
    "rationale_or_objective",
    "interpretation_or_implication",
    "limitation_or_caution",
    "conclusion_or_takeaway",
    "tool_or_algorithm",
    "cross_modal_result",
}
REQUIRED_EXPECTATION_FIELDS = {
    "research_question",
    "study_design",
    "population_or_materials",
    "methods",
    "primary_findings",
    "secondary_findings",
    "sensitivity_analysis",
    "statistical_tests_used",
    "interpretation",
    "limitations",
    "tables_figures_supplements",
    "supplement_availability",
    "uniqueness",
}


class GoldStandardValidationError(ValueError):
    """Raised when a final-report gold-standard fixture is structurally invalid."""


def load_gold_standard(path: str | Path) -> dict[str, Any]:
    fixture_path = Path(path)
    try:
        payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GoldStandardValidationError(f"{fixture_path}: invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise GoldStandardValidationError(f"{fixture_path}: root must be a JSON object")

    errors = validate_gold_standard(payload, source=str(fixture_path))
    if errors:
        raise GoldStandardValidationError("\n".join(errors))
    return payload


def validate_gold_standard(payload: dict[str, Any], *, source: str = "<gold_standard>") -> list[str]:
    errors: list[str] = []
    if payload.get("schema_version") != 1:
        errors.append(f"{source}: schema_version must be 1")
    for field in ("case_id", "source_pdf", "source_sha256", "gold_standard_type"):
        if not _non_empty_string(payload.get(field)):
            errors.append(f"{source}: {field} must be a non-empty string")
    if payload.get("gold_standard_type") != "final_report_expectations":
        errors.append(f"{source}: gold_standard_type must be final_report_expectations")

    errors.extend(_validate_authoring(payload.get("authoring"), source))
    errors.extend(_validate_paper_identity(payload.get("paper_identity"), source))
    errors.extend(_validate_final_report_expectations(payload.get("final_report_expectations"), source))
    errors.extend(_validate_critical_claims(payload.get("critical_claims"), source))
    errors.extend(_validate_string_list(payload.get("report_should_not_claim"), "report_should_not_claim", source))
    errors.extend(_validate_string_list(payload.get("scoring_focus"), "scoring_focus", source))
    return errors


def validate_gold_standard_file(path: str | Path) -> None:
    load_gold_standard(path)


def _validate_authoring(value: Any, source: str) -> list[str]:
    if not isinstance(value, dict):
        return [f"{source}: authoring must be an object"]
    errors: list[str] = []
    if value.get("method") != "codex_assisted_source_review":
        errors.append(f"{source}: authoring.method must be codex_assisted_source_review")
    if value.get("review_status") not in VALID_REVIEW_STATUSES:
        errors.append(
            f"{source}: authoring.review_status must be one of {', '.join(sorted(VALID_REVIEW_STATUSES))}"
        )
    if not _non_empty_string(value.get("created_at")):
        errors.append(f"{source}: authoring.created_at must be a non-empty string")
    errors.extend(_validate_string_list(value.get("source_material"), "authoring.source_material", source))
    return errors


def _validate_paper_identity(value: Any, source: str) -> list[str]:
    if not isinstance(value, dict):
        return [f"{source}: paper_identity must be an object"]
    errors: list[str] = []
    for field in ("title", "domain", "study_type"):
        if not _non_empty_string(value.get(field)):
            errors.append(f"{source}: paper_identity.{field} must be a non-empty string")
    return errors


def _validate_final_report_expectations(value: Any, source: str) -> list[str]:
    if not isinstance(value, dict):
        return [f"{source}: final_report_expectations must be an object"]
    errors: list[str] = []
    missing = sorted(REQUIRED_EXPECTATION_FIELDS - set(value))
    if missing:
        errors.append(f"{source}: missing final_report_expectations fields: {', '.join(missing)}")
    for field in ("research_question", "study_design", "population_or_materials", "supplement_availability"):
        if field in value and not _non_empty_string(value.get(field)):
            errors.append(f"{source}: final_report_expectations.{field} must be a non-empty string")
    for field in (
        "methods",
        "primary_findings",
        "secondary_findings",
        "sensitivity_analysis",
        "statistical_tests_used",
        "interpretation",
        "limitations",
        "tables_figures_supplements",
        "uniqueness",
    ):
        if field in value:
            errors.extend(_validate_string_list(value.get(field), f"final_report_expectations.{field}", source))
    return errors


def _validate_critical_claims(value: Any, source: str) -> list[str]:
    if not isinstance(value, list) or not value:
        return [f"{source}: critical_claims must be a non-empty list"]
    errors: list[str] = []
    seen: set[str] = set()
    for index, claim in enumerate(value):
        prefix = f"{source}: critical_claims[{index}]"
        if not isinstance(claim, dict):
            errors.append(f"{prefix} must be an object")
            continue
        claim_id = claim.get("claim_id")
        if not _non_empty_string(claim_id):
            errors.append(f"{prefix}.claim_id must be a non-empty string")
        elif claim_id in seen:
            errors.append(f"{prefix}.claim_id is duplicated")
        else:
            seen.add(claim_id)
        if claim.get("section") not in VALID_SECTIONS:
            errors.append(f"{prefix}.section must be one of {', '.join(sorted(VALID_SECTIONS))}")
        if claim.get("importance") not in VALID_IMPORTANCE:
            errors.append(f"{prefix}.importance must be one of {', '.join(sorted(VALID_IMPORTANCE))}")
        for field in ("claim", "source_anchor"):
            if not _non_empty_string(claim.get(field)):
                errors.append(f"{prefix}.{field} must be a non-empty string")
        errors.extend(_validate_string_list(claim.get("expected_entities"), f"{prefix}.expected_entities", source))
        errors.extend(_validate_expected_detail_types(claim.get("expected_detail_types"), prefix))
        errors.extend(_validate_expected_numbers(claim.get("expected_numbers"), prefix))
    return errors


def _validate_string_list(value: Any, field: str, source: str) -> list[str]:
    if not isinstance(value, list):
        return [f"{source}: {field} must be a list"]
    if any(not _non_empty_string(item) for item in value):
        return [f"{source}: {field} entries must be non-empty strings"]
    return []


def _validate_expected_numbers(value: Any, prefix: str) -> list[str]:
    if not isinstance(value, list):
        return [f"{prefix}.expected_numbers must be a list"]
    errors: list[str] = []
    for index, item in enumerate(value):
        item_prefix = f"{prefix}.expected_numbers[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{item_prefix} must be an object")
            continue
        if not _non_empty_string(item.get("label")):
            errors.append(f"{item_prefix}.label must be a non-empty string")
        number = item.get("value")
        if not isinstance(number, (int, float)) or isinstance(number, bool):
            errors.append(f"{item_prefix}.value must be a number")
        if "unit" in item and not isinstance(item["unit"], str):
            errors.append(f"{item_prefix}.unit must be a string")
        if "comparator" in item and item["comparator"] not in {"=", "~", "<", "<=", ">", ">="}:
            errors.append(f"{item_prefix}.comparator is invalid")
    return errors


def _validate_expected_detail_types(value: Any, prefix: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        return [f"{prefix}.expected_detail_types must be a list"]
    errors: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        if not _non_empty_string(item):
            errors.append(f"{prefix}.expected_detail_types[{index}] must be a non-empty string")
            continue
        normalized = str(item).strip()
        if normalized not in VALID_EXPECTED_DETAIL_TYPES:
            errors.append(
                f"{prefix}.expected_detail_types[{index}] must be one of "
                f"{', '.join(sorted(VALID_EXPECTED_DETAIL_TYPES))}"
            )
        if normalized in seen:
            errors.append(f"{prefix}.expected_detail_types[{index}] is duplicated")
        seen.add(normalized)
    return errors


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate final-report gold-standard JSON fixtures.")
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.paths:
        load_gold_standard(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
