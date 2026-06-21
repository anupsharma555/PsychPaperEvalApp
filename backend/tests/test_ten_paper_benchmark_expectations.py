from __future__ import annotations

import csv

from app.services.benchmark_expectations import (
    PROJECT_ROOT,
    REQUIRED_FAILURE_MODES,
    collect_benchmark_expectation_errors,
    load_benchmark_expectations,
)


def test_ten_paper_expectation_fixture_is_valid() -> None:
    payload = load_benchmark_expectations()

    errors = collect_benchmark_expectation_errors(payload)

    assert errors == []


def test_ten_paper_expectations_cover_metadata_corpus() -> None:
    payload = load_benchmark_expectations()
    cases = payload["cases"]
    expected_metadata = _metadata_rows()

    assert len(cases) == 10
    assert [case["case_id"] for case in cases] == [f"{index:02d}" for index in range(1, 11)]
    assert {case["pdf_file"] for case in cases} == {
        f"test/{row['pdf_file']}" for row in expected_metadata
    }
    assert {
        (case["pmid"], case["pmcid"], case["pdf_file"].removeprefix("test/")) for case in cases
    } == {(row["pmid"], row["pmcid"], row["pdf_file"]) for row in expected_metadata}


def test_expectations_define_parser_thresholds_and_section_boundaries() -> None:
    payload = load_benchmark_expectations()

    for case in payload["cases"]:
        floors = case["parser_coverage_floors"]
        assert floors["parser_text_coverage"] >= 0.7
        assert floors["source_manifest_completeness"] == 1.0
        assert case["section_boundary_wrong_rate_ceiling"] <= 0.25

        refs = case["expected_refs"]
        if not refs["figure_refs"] and not refs["supplementary_figure_refs"]:
            assert floors["figure_ref_recall"] == 1.0
        if not refs["table_refs"] and not refs["supplementary_table_refs"]:
            assert floors["table_ref_recall"] == 1.0


def test_source_manifest_expectations_point_to_primary_pdf_assets() -> None:
    payload = load_benchmark_expectations()

    for case in payload["cases"]:
        source = case["source_manifest_expectations"]
        assert source["required"] is True
        assert source["selected_assets_min"] >= 1
        assert source["expected_primary_asset"] == case["pdf_file"]
        assert "file" in source["source_type_one_of"]
        assert (PROJECT_ROOT / source["expected_primary_asset"]).is_file()


def test_failure_mode_metadata_distinguishes_benchmark_layers() -> None:
    payload = load_benchmark_expectations()

    assert set(payload["failure_modes"]) == REQUIRED_FAILURE_MODES
    for case in payload["cases"]:
        metadata = case["failure_mode_metadata"]
        assert set(metadata) == REQUIRED_FAILURE_MODES
        assert any("source_manifest" in item for item in metadata["benchmark_invalidity"])
        assert any(
            "section_boundary" in item or "section" in item for item in metadata["synthesis_failure"]
        )
        assert all("live_model" in item or "claim" in item for item in metadata["model_failure"])


def test_supplement_only_case_is_marked_without_hallucinated_main_sections() -> None:
    payload = load_benchmark_expectations()
    case = next(item for item in payload["cases"] if item["case_id"] == "04")

    assert case["expected_sections"] == {
        "abstract": False,
        "introduction": False,
        "methods": False,
        "results": False,
        "discussion": False,
        "conclusion": False,
        "references": False,
        "supplementary_material": True,
    }
    assert case["expected_refs"]["supplementary_table_refs"] == ["S1"]
    assert "case_is_supplement_only_without_case_metadata" in case["failure_mode_metadata"][
        "benchmark_invalidity"
    ]


def _metadata_rows() -> list[dict[str, str]]:
    path = PROJECT_ROOT / "test" / "metadata.tsv"
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))
