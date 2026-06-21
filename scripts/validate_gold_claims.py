from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {
    "claim_id",
    "section",
    "claim_type",
    "importance",
    "evidence_quote",
    "page_or_anchor",
    "expected_entities",
    "expected_numbers",
    "priority",
}

VALID_SECTIONS = {"introduction", "methods", "results", "discussion", "conclusion"}
VALID_IMPORTANCE = {"low", "medium", "high"}
VALID_PRIORITIES = {"P0", "P1", "P2", "P3"}


class GoldClaimValidationError(ValueError):
    """Raised when a gold-claim fixture is structurally invalid."""


def load_gold_claims(path: str | Path) -> list[dict[str, Any]]:
    fixture_path = Path(path)
    claims: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    errors: list[str] = []

    with fixture_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                claim = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_number}: invalid JSON: {exc.msg}")
                continue
            if not isinstance(claim, dict):
                errors.append(f"line {line_number}: row must be a JSON object")
                continue

            claim_id = claim.get("claim_id")
            if isinstance(claim_id, str) and claim_id:
                if claim_id in seen_ids:
                    errors.append(f"line {line_number}: duplicate claim_id {claim_id!r}")
                seen_ids.add(claim_id)

            errors.extend(_validate_claim(claim, line_number))
            claims.append(claim)

    if not claims:
        errors.append("fixture contains no claims")

    if errors:
        raise GoldClaimValidationError("\n".join(errors))

    return claims


def validate_gold_claims(path: str | Path) -> None:
    load_gold_claims(path)


def _validate_claim(claim: dict[str, Any], line_number: int) -> list[str]:
    errors: list[str] = []
    missing = sorted(field for field in REQUIRED_FIELDS if field not in claim)
    if missing:
        errors.append(f"line {line_number}: missing required fields: {', '.join(missing)}")

    for field in ("claim_id", "claim_type", "evidence_quote", "page_or_anchor"):
        if field in claim and not _non_empty_string(claim[field]):
            errors.append(f"line {line_number}: {field} must be a non-empty string")

    section = claim.get("section")
    if section not in VALID_SECTIONS:
        errors.append(
            f"line {line_number}: section must be one of {', '.join(sorted(VALID_SECTIONS))}"
        )

    importance = claim.get("importance")
    if importance not in VALID_IMPORTANCE:
        errors.append(
            f"line {line_number}: importance must be one of {', '.join(sorted(VALID_IMPORTANCE))}"
        )

    priority = claim.get("priority")
    if priority not in VALID_PRIORITIES:
        errors.append(f"line {line_number}: priority must be one of {', '.join(sorted(VALID_PRIORITIES))}")

    errors.extend(_validate_string_list(claim.get("expected_entities"), "expected_entities", line_number))
    errors.extend(_validate_expected_numbers(claim.get("expected_numbers"), line_number))
    return errors


def _validate_string_list(value: Any, field: str, line_number: int) -> list[str]:
    if not isinstance(value, list):
        return [f"line {line_number}: {field} must be a list"]
    if any(not _non_empty_string(item) for item in value):
        return [f"line {line_number}: {field} entries must be non-empty strings"]
    return []


def _validate_expected_numbers(value: Any, line_number: int) -> list[str]:
    if not isinstance(value, list):
        return [f"line {line_number}: expected_numbers must be a list"]

    errors: list[str] = []
    for index, item in enumerate(value):
        prefix = f"line {line_number}: expected_numbers[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{prefix} must be an object")
            continue
        if not _non_empty_string(item.get("label")):
            errors.append(f"{prefix}.label must be a non-empty string")
        number = item.get("value")
        if not isinstance(number, (int, float)) or isinstance(number, bool):
            errors.append(f"{prefix}.value must be a number")
        if "unit" in item and not isinstance(item["unit"], str):
            errors.append(f"{prefix}.unit must be a string")
        if "comparator" in item and item["comparator"] not in {"=", "~", "<", "<=", ">", ">="}:
            errors.append(f"{prefix}.comparator is invalid")
    return errors


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a gold-claims JSONL fixture.")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    validate_gold_claims(args.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
