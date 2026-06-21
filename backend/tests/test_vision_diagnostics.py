from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.analysis import figure_analysis, image_source, llm, supp_analysis
from app.services.analysis.media_cleaning import clean_figure_caption, figure_downstream_text


@pytest.fixture(autouse=True)
def _legacy_vision_llm_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default these tests to the explicit vision/ocr LLM path."""
    for module in (figure_analysis, supp_analysis):
        monkeypatch.setattr(module.settings, "llm_provider", "openai")
        monkeypatch.setattr(module.settings, "analysis_local_evidence_first_enabled", False)
    monkeypatch.setattr(figure_analysis.settings, "analysis_local_figure_caption_first_enabled", False)
    monkeypatch.setattr(supp_analysis.settings, "analysis_local_supplement_caption_first_enabled", False)


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


def test_figure_prompt_text_preserves_anchor_under_budget() -> None:
    prompt = figure_analysis._figure_prompt_text(  # noqa: SLF001
        "Analyze this figure and preserve concrete scientific details.",
        anchor="F-long",
        caption="Figure 9. Fluoxetine 20 mg daily improved week 8 response, p=0.03. " * 20,
        downstream_text="OCR footer noise plus repeated axis labels. " * 80,
        max_chars=700,
    )

    assert len(prompt) <= 700
    assert "Anchor: F-long" in prompt
    assert "Fluoxetine 20 mg daily" in prompt
    assert prompt.endswith("...")


def test_figure_analysis_uses_bounded_prompt_for_long_caption_and_ocr(monkeypatch) -> None:
    prompts: list[str] = []

    monkeypatch.setattr(
        figure_analysis,
        "resolve_image_path",
        lambda meta_obj, cache_dir, remote_cache: ("/tmp/figure_remote.jpg", "remote_url", None),
    )
    monkeypatch.setattr(figure_analysis.settings, "llm_n_ctx", 300)

    def _fake_chat_with_images(
        prompt: str,
        image_paths: list[str],
        system: str | None = None,
        temperature: float = 0.2,
    ) -> str:
        prompts.append(prompt)
        return (
            '{"evidence_packets":[{"finding_id":"fig-long","anchor":"F-long",'
            '"statement":"Figure reports higher response with fluoxetine.",'
            '"evidence_refs":["F-long"],"confidence":0.8,"category":"stats"}]}'
        )

    monkeypatch.setattr(figure_analysis, "chat_with_images", _fake_chat_with_images)

    report = figure_analysis.analyze_figures(
        [
            {
                "anchor": "F-long",
                "meta": json.dumps(
                    {
                        "source_url": "https://example.org/fig-long.jpg",
                        "caption": "Figure 9. Fluoxetine 20 mg daily improved week 8 response, p=0.03. " * 50,
                        "ocr_text": "Axis labels and page footer tokens. " * 100,
                    }
                ),
            }
        ]
    )

    assert prompts
    assert all(len(prompt) <= figure_analysis.max_chars_for_ctx(figure_analysis.settings.llm_n_ctx) for prompt in prompts)
    assert "Anchor: F-long" in prompts[0]
    assert "Fluoxetine 20 mg daily" in prompts[0]
    assert report["diagnostics"]["prompt_chars"] == [len(prompts[0])]
    assert report["evidence_packets"][0]["source_excerpt"].startswith("Figure 9.")


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


def test_figure_analysis_uses_caption_only_when_legend_extracted(monkeypatch) -> None:
    prompts: list[str] = []

    def _fake_chat_text(
        prompt: str,
        system: str | None = None,
        temperature: float = 0.2,
    ) -> str:
        prompts.append(prompt)
        return (
            '{"evidence_packets":[{"finding_id":"fig-caption-1","anchor":"F3",'
            '"statement":"Figure legend describes PCR validation of vector integration.",'
            '"evidence_refs":["F3"],"confidence":0.75,"category":"figure_quality"}]}'
        )

    monkeypatch.setattr(figure_analysis, "chat_text_fast", _fake_chat_text)

    report = figure_analysis.analyze_figures(
        [
            {
                "anchor": "F3",
                "meta": json.dumps(
                    {
                        "caption": "Figure 4. Multiplex and single PCR confirmed vector integration order.",
                        "source": "grobid_tei_figure",
                    }
                ),
            }
        ]
    )
    diagnostics = report["diagnostics"]

    assert diagnostics["caption_only_calls"] == 1
    assert diagnostics["caption_only_success"] == 1
    assert diagnostics["ocr_fallback_calls"] == 0
    assert "figure legend/caption was extracted" in prompts[0]
    assert report["evidence_packets"][0]["statement"].startswith("Figure legend describes PCR")
    assert "Multiplex and single PCR confirmed" in report["evidence_packets"][0]["source_excerpt"]


def test_figure_analysis_drops_ungrounded_llm_packets(monkeypatch) -> None:
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
        return json.dumps(
            {
                "evidence_packets": [
                    {
                        "finding_id": "fig-valid",
                        "anchor": "F1",
                        "statement": "Figure 1 reports the measured response.",
                        "evidence_refs": ["F1"],
                        "confidence": 0.8,
                    },
                    {
                        "finding_id": "fig-invalid",
                        "anchor": "figure:missing",
                        "statement": "A missing figure reports an unsupported response.",
                        "evidence_refs": ["figure:missing"],
                        "confidence": 0.8,
                    },
                ]
            }
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

    assert [packet["finding_id"] for packet in report["evidence_packets"]] == ["fig-valid"]
    assert report["diagnostics"]["dropped_ungrounded_packets"] == 1


def test_media_cleaning_prefers_clean_caption_before_ocr() -> None:
    text, source = figure_downstream_text(
        caption=(
            "Figure 1. Bar graph shows mean \x00B1 SEM for cell uptake across engineered cell lines "
            "after stable vector integration."
        ),
        ocr_text="noisy page footer 150 6 250 www.example.org",
    )

    assert source == "caption"
    assert "+/- SEM" in text
    assert "www.example.org" not in text


def test_media_cleaning_falls_back_to_ocr_for_short_caption() -> None:
    text, source = figure_downstream_text(
        caption="Figure 2.",
        ocr_text="Dose response curves show higher transport activity in engineered cells.",
    )

    assert source == "caption_plus_ocr_fallback"
    assert "Dose response curves" in text
    assert clean_figure_caption("mean \x00B1 SEM") == "mean +/- SEM"


def test_media_cleaning_trusts_short_scientific_caption_before_noisy_ocr() -> None:
    text, source = figure_downstream_text(
        caption="Figure 2. MADRS response was higher with sertraline 50 mg, p=0.03.",
        ocr_text="Downloaded from www.example.org 150 6 250 page footer noisy tokens",
    )

    assert source == "caption"
    assert "sertraline 50 mg" in text
    assert "p=0.03" in text
    assert "www.example.org" not in text


def test_local_figure_analysis_uses_caption_first_for_scientific_caption(monkeypatch) -> None:
    monkeypatch.setattr(figure_analysis.settings, "llm_provider", "local")
    monkeypatch.setattr(figure_analysis.settings, "analysis_local_figure_caption_first_enabled", True)

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
        raise AssertionError("local caption-first path should not call vision")

    monkeypatch.setattr(figure_analysis, "chat_with_images", _fake_chat_with_images)

    report = figure_analysis.analyze_figures(
        [
            {
                "anchor": "F4",
                "meta": json.dumps(
                    {
                        "source_url": "https://example.org/fig4.jpg",
                        "caption": "Figure 4. MADRS response was higher with sertraline 50 mg, p=0.03.",
                        "ocr_text": "Downloaded from www.example.org 150 6 250 page footer noisy tokens",
                    }
                ),
            }
        ]
    )

    diagnostics = report["diagnostics"]
    assert diagnostics["vision_calls"] == 0
    assert diagnostics["caption_first_skipped_vision"] == 1
    assert diagnostics["caption_first_skip_anchors"] == ["F4"]
    assert diagnostics["downstream_text_sources"]["caption"] == 1
    assert diagnostics["downstream_text_source_by_anchor"]["F4"] == "caption"
    assert "local_caption_first_skipped_vision" in report["evidence_packets"][0]["quality_flags"]
    assert report["evidence_packets"][0]["source_excerpt"].startswith("Figure 4. MADRS response")


def test_openai_figure_analysis_keeps_vision_for_scientific_caption(monkeypatch) -> None:
    prompts: list[str] = []
    monkeypatch.setattr(figure_analysis.settings, "llm_provider", "openai")
    monkeypatch.setattr(figure_analysis.settings, "analysis_local_figure_caption_first_enabled", True)
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
        prompts.append(prompt)
        return (
            '{"evidence_packets":[{"finding_id":"fig-caption-signal","anchor":"F4",'
            '"statement":"Figure 4 reports higher MADRS response with sertraline.",'
            '"evidence_refs":["F4"],"confidence":0.8,"category":"stats"}]}'
        )

    monkeypatch.setattr(figure_analysis, "chat_with_images", _fake_chat_with_images)

    report = figure_analysis.analyze_figures(
        [
            {
                "anchor": "F4",
                "meta": json.dumps(
                    {
                        "source_url": "https://example.org/fig4.jpg",
                        "caption": "Figure 4. MADRS response was higher with sertraline 50 mg, p=0.03.",
                        "ocr_text": "Downloaded from www.example.org 150 6 250 page footer noisy tokens",
                    }
                ),
            }
        ]
    )

    diagnostics = report["diagnostics"]
    assert diagnostics["vision_calls"] == 1
    assert diagnostics["caption_first_skipped_vision"] == 0
    assert "sertraline 50 mg" in prompts[0]
    assert "www.example.org" not in prompts[0]
    assert report["evidence_packets"][0]["finding_id"] == "fig-caption-signal"


def test_figure_analysis_skips_page_raster_fallback() -> None:
    report = figure_analysis.analyze_figures(
        [
            {
                "anchor": "page1",
                "meta": json.dumps(
                    {
                        "caption": "Page 1 raster fallback",
                        "fig_type": "page",
                        "source": "page_raster_fallback",
                        "ocr_text": "jspodanynuaps/wopauneummm noisy page text",
                    }
                ),
            }
        ]
    )

    assert report["diagnostics"]["vision_skipped"]["page_raster_fallback"] == 1
    assert report["evidence_packets"] == []


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


def test_local_supplement_analysis_uses_caption_first_for_scientific_caption(monkeypatch) -> None:
    monkeypatch.setattr(supp_analysis.settings, "llm_provider", "local")
    monkeypatch.setattr(supp_analysis.settings, "analysis_local_supplement_caption_first_enabled", True)

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
        raise AssertionError("local supplement caption-first path should not call vision")

    monkeypatch.setattr(supp_analysis, "chat_with_images", _fake_chat_with_images)

    report = supp_analysis.analyze_supplements(
        [
            {
                "anchor": "S1",
                "modality": "figure",
                "meta": json.dumps(
                    {
                        "source_url": "https://example.org/s1.png",
                        "caption": (
                            "Supplementary Figure S1. Dose-dependent uptake increased in engineered cells "
                            "after stable vector integration, with means reported as mean \x00B1 SEM."
                        ),
                        "ocr_text": "Downloaded from www.example.org 150 6 250 noisy page footer",
                    }
                ),
            }
        ]
    )

    diagnostics = report["diagnostics"]
    assert diagnostics["vision_calls"] == 0
    assert diagnostics["caption_first_skipped_vision"] == 1
    assert diagnostics["caption_first_skip_anchors"] == ["S1"]
    assert diagnostics["downstream_text_sources"]["caption"] == 1
    assert diagnostics["downstream_text_source_by_anchor"]["S1"] == "caption"
    assert "local_caption_first_skipped_vision" in report["evidence_packets"][0]["quality_flags"]
    assert report["evidence_packets"][0]["source_excerpt"].startswith("Supplementary Figure S1")


def test_openai_supplement_analysis_keeps_vision_for_scientific_caption(monkeypatch) -> None:
    prompts: list[str] = []
    monkeypatch.setattr(supp_analysis.settings, "llm_provider", "openai")
    monkeypatch.setattr(supp_analysis.settings, "analysis_local_supplement_caption_first_enabled", True)
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
        prompts.append(prompt)
        return (
            '{"evidence_packets":[{"finding_id":"supp-caption-1","anchor":"S1",'
            '"statement":"Supplement S1 reports dose-dependent uptake in engineered cells.",'
            '"evidence_refs":["S1"],"confidence":0.8,"category":"supplement_quality"}]}'
        )

    monkeypatch.setattr(supp_analysis, "chat_with_images", _fake_chat_with_images)

    report = supp_analysis.analyze_supplements(
        [
            {
                "anchor": "S1",
                "modality": "figure",
                "meta": json.dumps(
                    {
                        "source_url": "https://example.org/s1.png",
                        "caption": (
                            "Supplementary Figure S1. Dose-dependent uptake increased in engineered cells "
                            "after stable vector integration, with means reported as mean \x00B1 SEM."
                        ),
                        "ocr_text": "Downloaded from www.example.org 150 6 250 noisy page footer",
                    }
                ),
            }
        ]
    )

    diagnostics = report["diagnostics"]
    assert diagnostics["vision_calls"] == 1
    assert diagnostics["caption_first_skipped_vision"] == 0
    assert diagnostics["downstream_text_sources"]["caption"] == 1
    assert diagnostics["downstream_text_source_by_anchor"]["S1"] == "caption"
    assert "mean +/- SEM" in prompts[0]
    assert "www.example.org" not in prompts[0]
    assert report["evidence_packets"][0]["finding_id"] == "supp-caption-1"


def test_supplement_analysis_uses_ocr_for_caption_poor_unavailable_image(monkeypatch) -> None:
    prompts: list[str] = []

    monkeypatch.setattr(
        supp_analysis,
        "resolve_image_path",
        lambda meta_obj, cache_dir, remote_cache: (None, None, "download_error"),
    )

    def _fake_chat_text(
        prompt: str,
        system: str | None = None,
        temperature: float = 0.2,
    ) -> str:
        prompts.append(prompt)
        return (
            '{"evidence_packets":[{"finding_id":"supp-ocr-1","anchor":"S2",'
            '"statement":"Supplement S2 reports higher transport activity in engineered cells.",'
            '"evidence_refs":["S2"],"confidence":0.7,"category":"supplement_quality"}]}'
        )

    monkeypatch.setattr(supp_analysis, "chat_text_fast", _fake_chat_text)

    report = supp_analysis.analyze_supplements(
        [
            {
                "anchor": "S2",
                "modality": "figure",
                "meta": json.dumps(
                    {
                        "caption": "Figure S2.",
                        "ocr_text": "Transport activity increased in engineered cells at the highest dose.",
                    }
                ),
            }
        ]
    )

    diagnostics = report["diagnostics"]
    assert diagnostics["ocr_fallback_calls"] == 1
    assert diagnostics["ocr_fallback_success"] == 1
    assert diagnostics["downstream_text_sources"]["caption_plus_ocr_fallback"] == 1
    assert diagnostics["downstream_text_source_by_anchor"]["S2"] == "caption_plus_ocr_fallback"
    assert "Transport activity increased" in prompts[0]
    assert report["evidence_packets"]


def test_supplement_analysis_drops_ungrounded_llm_packets(monkeypatch) -> None:
    def _fake_chat_text(
        prompt: str,
        system: str | None = None,
        temperature: float = 0.2,
    ) -> str:
        return json.dumps(
            {
                "evidence_packets": [
                    {
                        "finding_id": "supp-valid",
                        "anchor": "S1",
                        "statement": "Supplement S1 reports a sensitivity analysis.",
                        "evidence_refs": ["S1"],
                        "confidence": 0.8,
                    },
                    {
                        "finding_id": "supp-invalid",
                        "anchor": "S-missing",
                        "statement": "A missing supplement reports an unsupported result.",
                        "evidence_refs": ["S-missing"],
                        "confidence": 0.8,
                    },
                ]
            }
        )

    monkeypatch.setattr(supp_analysis, "chat_text_fast", _fake_chat_text)

    report = supp_analysis.analyze_supplements(
        [
            {
                "anchor": "S1",
                "modality": "text",
                "content": "Sensitivity analysis results are described here.",
            }
        ]
    )

    assert [packet["finding_id"] for packet in report["evidence_packets"]] == ["supp-valid"]
    assert report["diagnostics"]["dropped_ungrounded_packets"] == 1


def test_llm_model_usage_counters_capture_calls(monkeypatch) -> None:
    class _DummyLLM:
        def create_chat_completion(self, messages, temperature):
            return {"choices": [{"message": {"content": "{}"}}]}

    clock = iter([0.0, 0.25, 1.0, 1.8, 2.0, 4.5])

    monkeypatch.setattr(llm, "monotonic", lambda: next(clock))
    monkeypatch.setattr(llm.settings, "llm_provider", "local")
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


def test_llm_model_usage_counters_capture_local_model_loads() -> None:
    llm.reset_model_usage_counters()

    llm._record_model_load("text", 1.25, ok=True)  # noqa: SLF001
    llm._record_model_load("vision", 2.5, ok=False)  # noqa: SLF001
    usage = llm.snapshot_model_usage_counters()

    assert usage["text_model_load_calls"] == 1
    assert usage["text_model_load_errors"] == 0
    assert usage["text_model_load_seconds"] == 1.25
    assert usage["vision_model_load_calls"] == 1
    assert usage["vision_model_load_errors"] == 1
    assert usage["vision_model_load_seconds"] == 2.5


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
