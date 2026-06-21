from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXPECTATIONS_PATH = PROJECT_ROOT / "benchmarks" / "ten_paper_expectations.json"

REQUIRED_FAILURE_MODES = {
    "parser_failure",
    "model_failure",
    "synthesis_failure",
    "benchmark_invalidity",
}
REQUIRED_REF_GROUPS = {
    "figure_refs",
    "table_refs",
    "supplementary_figure_refs",
    "supplementary_table_refs",
}
REQUIRED_FLOORS = {
    "parser_text_coverage",
    "section_availability_recall",
    "figure_ref_recall",
    "table_ref_recall",
    "source_manifest_completeness",
}
REF_PATTERN = re.compile(r"^S?\d+[A-Z]?$")


def load_benchmark_expectations(path: Path | None = None) -> dict[str, Any]:
    fixture_path = path or DEFAULT_EXPECTATIONS_PATH
    return json.loads(fixture_path.read_text(encoding="utf-8"))


def collect_benchmark_expectation_errors(
    payload: dict[str, Any],
    *,
    project_root: Path = PROJECT_ROOT,
) -> list[str]:
    errors: list[str] = []
    if payload.get("schema_version") != 1:
        errors.append("schema_version must be 1")

    canonical_sections = payload.get("canonical_sections")
    if not isinstance(canonical_sections, list) or not canonical_sections:
        errors.append("canonical_sections must be a non-empty list")
        canonical_section_set: set[str] = set()
    else:
        canonical_section_set = set(canonical_sections)

    failure_modes = set(payload.get("failure_modes") or [])
    if failure_modes != REQUIRED_FAILURE_MODES:
        errors.append(f"failure_modes must equal {sorted(REQUIRED_FAILURE_MODES)}")

    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        errors.append("cases must be a non-empty list")
        return errors

    seen_case_ids: set[str] = set()
    seen_pdf_files: set[str] = set()
    for index, case in enumerate(cases):
        prefix = f"cases[{index}]"
        if not isinstance(case, dict):
            errors.append(f"{prefix} must be an object")
            continue

        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not re.fullmatch(r"\d{2}", case_id):
            errors.append(f"{prefix}.case_id must be a two-digit string")
        elif case_id in seen_case_ids:
            errors.append(f"{prefix}.case_id duplicates {case_id}")
        else:
            seen_case_ids.add(case_id)

        pdf_file = case.get("pdf_file")
        if not isinstance(pdf_file, str) or not pdf_file.startswith("test/"):
            errors.append(f"{prefix}.pdf_file must be a test/ path")
        else:
            if pdf_file in seen_pdf_files:
                errors.append(f"{prefix}.pdf_file duplicates {pdf_file}")
            seen_pdf_files.add(pdf_file)
            if not (project_root / pdf_file).is_file():
                errors.append(f"{prefix}.pdf_file does not exist: {pdf_file}")

        sections = case.get("expected_sections")
        if not isinstance(sections, dict):
            errors.append(f"{prefix}.expected_sections must be an object")
        else:
            if set(sections) != canonical_section_set:
                errors.append(f"{prefix}.expected_sections must match canonical_sections")
            if not all(isinstance(value, bool) for value in sections.values()):
                errors.append(f"{prefix}.expected_sections values must be booleans")
            if sections and not any(sections.values()):
                errors.append(f"{prefix}.expected_sections must mark at least one section available")

        refs = case.get("expected_refs")
        if not isinstance(refs, dict):
            errors.append(f"{prefix}.expected_refs must be an object")
        else:
            if set(refs) != REQUIRED_REF_GROUPS:
                errors.append(f"{prefix}.expected_refs must contain {sorted(REQUIRED_REF_GROUPS)}")
            for key, values in refs.items():
                if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
                    errors.append(f"{prefix}.expected_refs.{key} must be a list of strings")
                    continue
                if values != sorted(values, key=_ref_sort_key):
                    errors.append(f"{prefix}.expected_refs.{key} must be sorted")
                if len(values) != len(set(values)):
                    errors.append(f"{prefix}.expected_refs.{key} contains duplicates")
                bad_refs = [item for item in values if not REF_PATTERN.fullmatch(item)]
                if bad_refs:
                    errors.append(f"{prefix}.expected_refs.{key} contains invalid refs: {bad_refs}")
                if key.startswith("supplementary_"):
                    non_supp = [item for item in values if not item.startswith("S")]
                    if non_supp:
                        errors.append(f"{prefix}.expected_refs.{key} must use S-prefixed refs: {non_supp}")

        floors = case.get("parser_coverage_floors")
        if not isinstance(floors, dict):
            errors.append(f"{prefix}.parser_coverage_floors must be an object")
        else:
            if set(floors) != REQUIRED_FLOORS:
                errors.append(f"{prefix}.parser_coverage_floors must contain {sorted(REQUIRED_FLOORS)}")
            for key, value in floors.items():
                if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                    errors.append(f"{prefix}.parser_coverage_floors.{key} must be between 0 and 1")

        ceiling = case.get("section_boundary_wrong_rate_ceiling")
        if not isinstance(ceiling, (int, float)) or not 0 <= ceiling <= 1:
            errors.append(f"{prefix}.section_boundary_wrong_rate_ceiling must be between 0 and 1")

        source = case.get("source_manifest_expectations")
        if not isinstance(source, dict):
            errors.append(f"{prefix}.source_manifest_expectations must be an object")
        else:
            _validate_source_manifest_expectations(source, prefix, errors)
            if isinstance(pdf_file, str) and source.get("expected_primary_asset") != pdf_file:
                errors.append(f"{prefix}.source_manifest_expectations.expected_primary_asset must match pdf_file")

        failure_metadata = case.get("failure_mode_metadata")
        if not isinstance(failure_metadata, dict):
            errors.append(f"{prefix}.failure_mode_metadata must be an object")
        else:
            if set(failure_metadata) != REQUIRED_FAILURE_MODES:
                errors.append(f"{prefix}.failure_mode_metadata must contain all failure modes")
            for key, values in failure_metadata.items():
                if not isinstance(values, list) or not values or not all(isinstance(item, str) for item in values):
                    errors.append(f"{prefix}.failure_mode_metadata.{key} must be a non-empty list of strings")

    return errors


def _validate_source_manifest_expectations(source: dict[str, Any], prefix: str, errors: list[str]) -> None:
    if source.get("required") is not True:
        errors.append(f"{prefix}.source_manifest_expectations.required must be true")
    for key in ("source_type_one_of", "status_one_of"):
        values = source.get(key)
        if not isinstance(values, list) or not values or not all(isinstance(item, str) for item in values):
            errors.append(f"{prefix}.source_manifest_expectations.{key} must be a non-empty list of strings")
    selected_assets_min = source.get("selected_assets_min")
    if not isinstance(selected_assets_min, int) or selected_assets_min < 1:
        errors.append(f"{prefix}.source_manifest_expectations.selected_assets_min must be a positive integer")
    expected_primary_asset = source.get("expected_primary_asset")
    if not isinstance(expected_primary_asset, str) or not expected_primary_asset.startswith("test/"):
        errors.append(f"{prefix}.source_manifest_expectations.expected_primary_asset must be a test/ path")


def _ref_sort_key(value: str) -> tuple[int, int, str]:
    match = re.fullmatch(r"(S?)(\d+)([A-Z]?)", value)
    if not match:
        return (2, 0, value)
    prefix, number, suffix = match.groups()
    return (1 if prefix else 0, int(number), suffix)
