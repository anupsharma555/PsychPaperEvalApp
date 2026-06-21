from __future__ import annotations

import json

import pytest

from app.services.analysis import figure_analysis, runner, supp_analysis, table_analysis


def _enable_local_evidence_first(monkeypatch: pytest.MonkeyPatch, module) -> None:
    monkeypatch.setattr(module.settings, "llm_provider", "local")
    monkeypatch.setattr(module.settings, "analysis_local_evidence_first_enabled", True)


def test_table_local_evidence_first_uses_extracts_without_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_local_evidence_first(monkeypatch, table_analysis)
    monkeypatch.setattr(
        table_analysis,
        "chat_text_fast",
        lambda *args, **kwargs: pytest.fail("local evidence-first table analysis should not call the LLM"),
    )

    report = table_analysis.analyze_tables(
        [
            {
                "anchor": "T1",
                "content": json.dumps(
                    {
                        "columns": ["Arm", "Dose", "Response"],
                        "data": [["sertraline", "50 mg", "62%"], ["placebo", "0 mg", "41%"]],
                    }
                ),
            }
        ]
    )

    assert report["diagnostics"]["local_evidence_first"] is True
    assert report["evidence_packets"]
    assert "sertraline" in report["evidence_packets"][0]["source_excerpt"]
    assert "local_evidence_first" in report["evidence_packets"][0]["quality_flags"]


def test_table_local_evidence_first_emits_method_and_result_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_local_evidence_first(monkeypatch, table_analysis)
    monkeypatch.setattr(
        table_analysis,
        "chat_text_fast",
        lambda *args, **kwargs: pytest.fail("local evidence-first table analysis should not call the LLM"),
    )

    report = table_analysis.analyze_tables(
        [
            {
                "anchor": "T2",
                "content": json.dumps(
                    {
                        "columns": ["Group", "Measure", "Value"],
                        "data": [
                            ["healthy controls", "sample size", "n = 50"],
                            ["treatment arm", "response rate", "62%"],
                            ["placebo arm", "response rate", "41%; p = 0.03"],
                        ],
                    }
                ),
                "meta": json.dumps({"caption": "Table 2. Baseline sample and treatment results."}),
            }
        ]
    )

    packets = report["evidence_packets"]
    row_statements = [packet["statement"] for packet in packets if "Table row reports" in packet["statement"]]
    assert any("healthy controls" in statement and "50" in statement for statement in row_statements)
    assert any("placebo arm" in statement and "0.03" in statement for statement in row_statements)
    assert any(packet["section_label"] == "methods" for packet in packets)
    assert any(packet["section_label"] == "results" for packet in packets)
    assert any(packet["category"] == "stats" for packet in packets)


def test_figure_local_evidence_first_uses_caption_without_resolving_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_local_evidence_first(monkeypatch, figure_analysis)
    monkeypatch.setattr(
        figure_analysis,
        "resolve_image_path",
        lambda *args, **kwargs: pytest.fail("local evidence-first figure analysis should not resolve image paths"),
    )
    monkeypatch.setattr(
        figure_analysis,
        "chat_with_images",
        lambda *args, **kwargs: pytest.fail("local evidence-first figure analysis should not call vision LLM"),
    )

    report = figure_analysis.analyze_figures(
        [
            {
                "anchor": "F1",
                "meta": json.dumps(
                    {
                        "caption": (
                            "Figure 1. Ketamine 0.5 mg/kg produced a larger MADRS reduction "
                            "than midazolam at 24 hours."
                        ),
                        "source_url": "https://example.org/figure1.png",
                    }
                ),
            }
        ]
    )

    assert report["diagnostics"]["local_evidence_first"] is True
    assert report["diagnostics"]["local_evidence_first_packets"] == 1
    assert report["diagnostics"]["vision_calls"] == 0
    assert "Ketamine 0.5 mg/kg" in report["evidence_packets"][0]["source_excerpt"]


def test_supplement_local_evidence_first_uses_text_and_figure_extracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_local_evidence_first(monkeypatch, supp_analysis)
    monkeypatch.setattr(
        supp_analysis,
        "resolve_image_path",
        lambda *args, **kwargs: pytest.fail("local evidence-first supplement analysis should not resolve image paths"),
    )
    monkeypatch.setattr(
        supp_analysis,
        "chat_text_fast",
        lambda *args, **kwargs: pytest.fail("local evidence-first supplement analysis should not call text LLM"),
    )
    monkeypatch.setattr(
        supp_analysis,
        "chat_with_images",
        lambda *args, **kwargs: pytest.fail("local evidence-first supplement analysis should not call vision LLM"),
    )

    report = supp_analysis.analyze_supplements(
        [
            {
                "anchor": "S1",
                "modality": "text",
                "content": "Supplementary methods specify escitalopram 10 mg daily for 8 weeks.",
            },
            {
                "anchor": "SF1",
                "modality": "figure",
                "meta": json.dumps(
                    {
                        "caption": (
                            "Supplementary Figure 1. Plasma IL-6 decreased after treatment "
                            "in responders but not nonresponders."
                        ),
                        "source_url": "https://example.org/supp_fig1.png",
                    }
                ),
            },
        ]
    )

    assert report["diagnostics"]["local_evidence_first"] is True
    assert report["diagnostics"]["local_evidence_first_packets"] == 2
    assert report["diagnostics"]["vision_calls"] == 0
    assert len(report["evidence_packets"]) == 2
    assert any("escitalopram 10 mg" in packet["source_excerpt"] for packet in report["evidence_packets"])
    assert any("Plasma IL-6 decreased" in packet["source_excerpt"] for packet in report["evidence_packets"])


def test_runner_allows_local_parallelism_for_evidence_first_extracts(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_local_evidence_first(monkeypatch, runner)
    monkeypatch.setattr(runner.settings, "analysis_parallel_modalities_enabled", True)
    monkeypatch.setattr(runner.settings, "analysis_text_subprocess_guard_enabled", True)
    monkeypatch.setattr(runner.settings, "analysis_modality_subprocess_guard_enabled", True)
    monkeypatch.setattr(runner.settings, "analysis_parallel_modality_workers", 4)

    assert runner._modalities_can_run_parallel() is True  # noqa: SLF001

    monkeypatch.setattr(runner.settings, "analysis_local_evidence_first_enabled", False)

    assert runner._modalities_can_run_parallel() is False  # noqa: SLF001
