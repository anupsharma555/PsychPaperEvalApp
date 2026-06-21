from __future__ import annotations

from app.core.config import Settings, settings


def test_test_harness_defaults_to_local_provider() -> None:
    assert settings.llm_provider_normalized == "local"


def test_local_text_timeout_defaults_to_shorter_fallback_budget() -> None:
    assert settings.analysis_text_subprocess_timeout_sec == 600
    assert settings.analysis_local_text_subprocess_timeout_sec == 360
    assert settings.analysis_local_text_subprocess_timeout_sec < settings.analysis_text_subprocess_timeout_sec


def test_local_text_preselection_defaults_enabled() -> None:
    assert settings.analysis_local_text_preselection_enabled is True
    assert settings.analysis_local_text_llm_batch_max_chars == 9000
    assert settings.analysis_local_text_preselection_max_chunks == 12
    assert settings.analysis_local_text_preselection_min_chunks_per_section == 2
    assert settings.analysis_local_text_cache_enabled is True
    assert settings.analysis_local_text_global_cache_enabled is False
    assert settings.llm_local_text_max_tokens == 320


def test_local_figure_caption_first_defaults_enabled() -> None:
    assert settings.analysis_local_figure_caption_first_enabled is True
    assert settings.analysis_local_figure_caption_first_min_chars == 80
    assert settings.analysis_local_supplement_caption_first_enabled is True
    assert settings.analysis_local_supplement_caption_first_min_chars == 80


def test_local_provider_defaults_to_evidence_first_analysis() -> None:
    local = Settings(llm_provider="local")

    assert local.analysis_local_evidence_first_enabled is True
    assert local.analysis_local_evidence_first_active is True
    assert local.effective_analysis_nontext_stage_enabled is True
    assert local.effective_analysis_nontext_llm_enabled is False
    assert local.analysis_narrative_overrides_enabled is True
    assert local.effective_analysis_narrative_overrides_enabled is True
    assert local.effective_analysis_verifier_enabled is False
    assert local.effective_analysis_section_verifier_enabled is False
    assert local.effective_analysis_summary_polish_enabled is False
    assert local.effective_analysis_section_extraction_llm_enabled is False
    assert local.effective_analysis_exec_summary_second_pass_enabled is False
    assert local.effective_analysis_section_synthesis_v2_llm_enabled is False


def test_local_evidence_first_does_not_disable_openai_deep_defaults() -> None:
    openai = Settings(llm_provider="openai", analysis_local_evidence_first_enabled=True)

    assert openai.analysis_local_evidence_first_active is False
    assert openai.effective_analysis_nontext_llm_enabled is True
