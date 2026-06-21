from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import compare_upstream_ab as upstream_ab  # noqa: E402


SAMPLE_BATCH_PATH = PROJECT_ROOT / "test" / "upstream_ab" / "upstream_ab_batch_20260503_183327.json"


def _chunk(modality: str, content: str, meta: dict[str, object], anchor: str = "") -> dict[str, object]:
    return {
        "anchor": anchor,
        "modality": modality,
        "content": content,
        "meta": json.dumps(meta),
    }


def _load_sample_batch() -> dict:
    return json.loads(SAMPLE_BATCH_PATH.read_text(encoding="utf-8"))


def _explicit_clean_media_metrics() -> dict[str, float]:
    return {
        "mean_figure_ref_recall": 0.795,
        "mean_table_ref_recall": 0.15,
        "mean_supplementary_figure_ref_recall": 0.75,
        "mean_supplementary_table_ref_recall": 0.1,
        "mean_artifact_text_rate": 0.456,
    }


def test_section_boundary_ledger_sample_batch_stays_within_observed_wrong_section_budget() -> None:
    payload = _load_sample_batch()
    ledger = payload["aggregate"]["section_variants"]["section_boundary_ledger"]
    threshold = upstream_ab.UPSTREAM_AB_REGRESSION_THRESHOLDS[
        "section_boundary_ledger_mean_wrong_section_rate_max"
    ]
    per_document_threshold = upstream_ab.UPSTREAM_AB_REGRESSION_THRESHOLDS[
        "section_boundary_ledger_document_wrong_section_rate_max"
    ]

    assert ledger["mean_wrong_section_rate"] <= threshold
    assert ledger["wrong_section_wins"] == payload["document_count"]
    assert all(
        row["variants"]["section_boundary_ledger"]["parsed_chunks"]["wrong_section_rate"] <= per_document_threshold
        for row in payload["documents"]
    )


def test_upstream_ab_regression_thresholds_accept_sample_batch_with_explicit_media_recall() -> None:
    payload = _load_sample_batch()
    payload = deepcopy(payload)
    payload["aggregate"]["media_variants"]["clean_caption_first"].update(_explicit_clean_media_metrics())

    assert upstream_ab.evaluate_regression_thresholds(payload) == []


def test_media_metrics_report_table_and_supplement_recall_explicitly() -> None:
    chunks = [
        _chunk(
            "text",
            "Results referenced Figure 1, Figure S1, Table 1, and Table S1.",
            {"section_norm": "results"},
        ),
        _chunk("figure", "Figure 1. Main result.", {"figure_id": "1", "caption": "Figure 1. Main result."}),
        _chunk("figure", "Figure S1. Supplement result.", {"figure_id": "S1", "caption": "Figure S1. Supplement result."}),
        _chunk("table", "Table 1. Main table.", {"table_id": "1"}),
    ]

    metrics = upstream_ab._media_metrics(chunks, mode="clean_caption_first")

    assert metrics["figure_ref_recall"] == 1.0
    assert metrics["table_ref_recall"] == 0.5
    assert metrics["supplementary_figure_ref_recall"] == 1.0
    assert metrics["supplementary_table_ref_recall"] == 0.0
    assert "missing_table_refs" in metrics


def test_artifact_text_rate_threshold_flags_noisy_figure_processing() -> None:
    noisy_metrics = upstream_ab._media_metrics(
        [
            _chunk("text", "The noisy extraction was attached to Figure 1.", {"section_norm": "results"}),
            _chunk(
                "figure",
                "Figure 1",
                {
                    "figure_id": "1",
                    "caption": "Figure 1",
                    "ocr_text": "123 4 567 aaaaaaaaaaaaaaaa",
                },
            ),
        ],
        mode="caption_plus_ocr",
    )
    payload = {
        "aggregate": {
            "section_variants": {
                "section_boundary_ledger": {
                    "mean_wrong_section_rate": 0.10,
                }
            },
            "media_variants": {
                "clean_caption_first": {
                    **_explicit_clean_media_metrics(),
                    "mean_artifact_text_rate": noisy_metrics["artifact_text_rate"],
                }
            },
        },
        "documents": [],
    }

    failures = upstream_ab.evaluate_regression_thresholds(payload)

    assert noisy_metrics["artifact_text_rate"] == 1.0
    assert any("clean_caption_first artifact_text_rate" in failure for failure in failures)
