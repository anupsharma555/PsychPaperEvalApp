from __future__ import annotations

import json
from pathlib import Path

from app.services.analysis import figure_analysis, image_source, llm, supp_analysis


def test_figure_analysis_uses_remote_source_when_local_path_missing(monkeypatch) -> None:
    seen_paths: list[list[str]] = []

    monkeypatch.setattr(
        figure_analysis,
        "resolve_image_path",
        lambda meta_obj, cache_dir, remote_cache: ("/tmp/figure_remote.jpg", "remote_url", None),
    )

    def _fake_chat_with_images(
        prompt: str,
        image_paths: list[str],
        system: str | None = None,
        temperature: float = 0.2,
    ) -> str:
        seen_paths.append(image_paths)
        return (
            '{"evidence_packets":[{"finding_id":"fig-1","anchor":"F1","statement":"Remote figure signal",'
            '"evidence_refs":["F1"],"confidence":0.9,"category":"figure_quality"}]}'
        )

    monkeypatch.setattr(figure_analysis, "chat_with_images", _fake_chat_with_images)

    report = figure_analysis.analyze_figures(
        [
            {
                "anchor": "F1",
                "meta": json.dumps({"source_url": "https://example.org/fig1.jpg", "caption": "Figure 1"}),
            }
        ]
    )
    diagnostics = report["diagnostics"]

    assert seen_paths == [["/tmp/figure_remote.jpg"]]
    assert diagnostics["vision_calls"] == 1
    assert diagnostics["vision_success"] == 1
    assert diagnostics["vision_input_sources"]["remote_url"] == 1
    assert report["evidence_packets"]


def test_figure_analysis_falls_back_to_ocr_when_image_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(
        figure_analysis,
        "resolve_image_path",
        lambda meta_obj, cache_dir, remote_cache: (None, None, "download_error"),
    )

    def _fake_chat_text(
        prompt: str,
        system: str | None = None,
        temperature: float = 0.2,
    ) -> str:
        return (
            '{"evidence_packets":[{"finding_id":"fig-ocr-1","anchor":"F2","statement":"OCR fallback signal",'
            '"evidence_refs":["F2"],"confidence":0.7,"category":"figure_quality"}]}'
        )

    monkeypatch.setattr(figure_analysis, "chat_text_fast", _fake_chat_text)

    report = figure_analysis.analyze_figures(
        [
            {
                "anchor": "F2",
                "meta": json.dumps(
                    {
                        "source_url": "https://example.org/missing.jpg",
                        "caption": "Figure 2",
                        "ocr_text": "Axis labels suggest improvement",
                    }
                ),
            }
        ]
    )
    diagnostics = report["diagnostics"]

    assert diagnostics["vision_calls"] == 0
    assert diagnostics["ocr_fallback_calls"] == 1
    assert diagnostics["ocr_fallback_success"] == 1
    assert diagnostics["vision_skipped"]["download_error"] == 1
    assert report["evidence_packets"]


def test_supplement_analysis_uses_remote_source_for_figure_chunks(monkeypatch) -> None:
    seen_paths: list[list[str]] = []

    monkeypatch.setattr(
        supp_analysis,
        "resolve_image_path",
        lambda meta_obj, cache_dir, remote_cache: ("/tmp/supp_remote.png", "remote_url", None),
    )

    def _fake_chat_with_images(
        prompt: str,
        image_paths: list[str],
        system: str | None = None,
        temperature: float = 0.2,
    ) -> str:
        seen_paths.append(image_paths)
        return (
            '{"evidence_packets":[{"finding_id":"supp-1","anchor":"S1","statement":"Supp figure signal",'
            '"evidence_refs":["S1"],"confidence":0.8,"category":"supplement_quality"}]}'
        )

    monkeypatch.setattr(supp_analysis, "chat_with_images", _fake_chat_with_images)

    report = supp_analysis.analyze_supplements(
        [
            {
                "anchor": "S1",
                "modality": "figure",
                "meta": json.dumps({"source_url": "https://example.org/s1.png", "caption": "Supp Fig S1"}),
            }
        ]
    )
    diagnostics = report["diagnostics"]

    assert seen_paths == [["/tmp/supp_remote.png"]]
    assert diagnostics["vision_calls"] == 1
    assert diagnostics["vision_success"] == 1
    assert diagnostics["vision_input_sources"]["remote_url"] == 1
    assert report["evidence_packets"]


def test_llm_model_usage_counters_capture_calls(monkeypatch) -> None:
    class _DummyLLM:
        def create_chat_completion(self, messages, temperature):
            return {"choices": [{"message": {"content": "{}"}}]}

    clock = iter([0.0, 0.25, 1.0, 1.8, 2.0, 4.5])

    monkeypatch.setattr(llm, "monotonic", lambda: next(clock))
    monkeypatch.setattr(llm, "_load_text_model", lambda: _DummyLLM())
    monkeypatch.setattr(llm, "_load_deep_model", lambda: _DummyLLM())
    monkeypatch.setattr(llm, "_load_vision_model", lambda: _DummyLLM())

    llm.reset_model_usage_counters()
    llm.chat_text_fast("text")
    llm.chat_text_deep("deep")
    llm.chat_with_images("vision", [])
    usage = llm.snapshot_model_usage_counters()

    assert usage["text_calls"] == 1
    assert usage["deep_calls"] == 1
    assert usage["vision_calls"] == 1
    assert usage["text_errors"] == 0
    assert usage["deep_errors"] == 0
    assert usage["vision_errors"] == 0
    assert usage["text_total_seconds"] == 0.25
    assert usage["deep_total_seconds"] == 0.8
    assert usage["vision_total_seconds"] == 2.5
    assert usage["text_avg_seconds"] == 0.25
    assert usage["deep_avg_seconds"] == 0.8
    assert usage["vision_avg_seconds"] == 2.5
    assert usage["slowest_model"] == "vision"
    assert usage["slowest_seconds"] == 2.5


def test_figure_analysis_passes_document_source_url_to_resolver(monkeypatch) -> None:
    captured_meta: list[dict] = []

    def _fake_resolve(meta_obj, cache_dir, remote_cache):
        captured_meta.append(meta_obj)
        return (None, None, "download_error")

    monkeypatch.setattr(figure_analysis, "resolve_image_path", _fake_resolve)

    report = figure_analysis.analyze_figures(
        [
            {
                "anchor": "F1",
                "document_source_url": "https://psychiatryonline.org/doi/10.1176/example",
                "meta": json.dumps({"source_url": "https://psychiatryonline.org/cms/asset/example.jpg"}),
            }
        ]
    )
    diagnostics = report["diagnostics"]

    assert captured_meta
    assert captured_meta[0]["document_source_url"] == "https://psychiatryonline.org/doi/10.1176/example"
    assert diagnostics["vision_skipped"]["download_error"] == 1


def test_image_source_skips_non_image_extensions_without_download(monkeypatch, tmp_path: Path) -> None:
    called = {"download": 0}

    def _fake_download(*args, **kwargs):
        called["download"] += 1
        return ""

    monkeypatch.setattr(image_source, "_download_remote_image", _fake_download)

    path, source_kind, skip_reason = image_source.resolve_image_path(
        {
            "source_url": "https://psychiatryonline.org/doi/suppl/10.1176/example/suppl_file/file.pdf",
            "source_page_url": "https://psychiatryonline.org/doi/10.1176/example",
        },
        cache_dir=tmp_path,
        remote_cache={},
    )

    assert path is None
    assert source_kind is None
    assert skip_reason == "unsupported_image_type"
    assert called["download"] == 0
