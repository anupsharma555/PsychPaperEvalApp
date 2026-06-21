from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(os.getenv("PAPER_EVAL_ROOT", Path(__file__).resolve().parents[3]))
ENV_FILES = [
    str(ROOT_DIR / "backend" / ".env"),
    str(ROOT_DIR / "secrets" / "openai.env"),
    str(ROOT_DIR / ".env"),
    ".env",
]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILES, env_nested_delimiter="__")

    app_name: str = "PaperEval"
    environment: str = "local"

    data_dir: Path = Field(default=Path("data"))
    models_dir: Path = Field(default=Path("models"))
    db_path: Path = Field(default=Path("data/app.db"))

    analysis_workers: int = 1
    analysis_use_process_pool: bool = True
    analysis_cleanup_orphans: bool = True
    report_retention_enabled: bool = True
    report_retention_limit: int = 10
    archive_max_members: int = 200
    archive_max_uncompressed_bytes: int = 250_000_000

    parser_engine: str = "validated"
    parser_reuse_unchanged_assets_enabled: bool = True

    llm_provider: str = "local"

    # Legacy single-model env vars retained for backward compatibility.
    llm_model_path: Path = Field(default=Path("models/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"))
    llm_mmproj_path: Path = Field(default=Path("models/mmproj-Qwen2.5-VL-7B-Instruct-Q8_0.gguf"))
    llm_chat_format: str = "qwen2.5-vl"

    llm_text_model_path: Optional[Path] = None
    llm_text_chat_format: str = "chatml"

    llm_deep_model_path: Optional[Path] = None
    llm_deep_chat_format: str = "chatml"

    llm_vision_model_path: Optional[Path] = None
    llm_vision_mmproj_path: Optional[Path] = None
    llm_vision_chat_format: str = "qwen2.5-vl"
    llm_n_ctx: int = 8192
    llm_n_threads: int = 8
    llm_n_batch: int = 512
    llm_n_gpu_layers: int = 999
    ggml_metal_devices: str = "0"
    local_gpu_smoke_enabled: bool = True
    local_gpu_smoke_timeout_sec: int = 120
    llm_text_max_tokens: int = 6000
    llm_deep_max_tokens: int = 3500
    llm_vision_max_tokens: int = 1200
    llm_local_text_max_tokens: int = 320
    llm_local_deep_max_tokens: int = 1200
    llm_local_vision_max_tokens: int = 800
    llm_image_max_dim: int = 1024
    llm_image_max_pixels: int = 1500000
    llm_image_format: str = "jpeg"
    llm_image_quality: int = 85

    psychpaper_openai_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_text_model: str = "gpt-5-mini"
    openai_deep_model: str = "gpt-5-mini"
    openai_vision_model: str = "gpt-5-mini"
    openai_timeout_sec: int = 120
    openai_json_mode_enabled: bool = True
    openai_send_temperature: bool = False
    openai_reasoning_effort: Optional[str] = "minimal"
    openai_api_mode: str = "responses"
    openai_usage_guardrails_enabled: bool = True
    openai_usage_log_path: Path = Field(default=Path("data/openai_usage_ledger.jsonl"))
    openai_run_mode: str = "standard"
    openai_max_cost_per_run_usd: float = 0.15
    openai_max_cost_per_day_usd: float = 2.00
    openai_max_calls_per_run: int = 36
    openai_max_output_tokens_per_run: int = 55000
    openai_estimated_image_input_tokens: int = 2500
    openai_cost_fallback_input_per_million: float = 5.0
    openai_cost_fallback_cached_input_per_million: float = 0.5
    openai_cost_fallback_output_per_million: float = 22.5

    fetch_timeout_sec: int = 60
    fetch_user_agent: str = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36 PaperEval/0.1"
    )
    unpaywall_email: Optional[str] = None

    grobid_url: str = "http://localhost:8070"
    grobid_timeout_sec: int = 120
    grobid_consolidate_header: bool = True
    grobid_consolidate_citations: bool = True
    grobid_include_coordinates: bool = True

    pdffigures2_cmd: Optional[str] = None
    pdffigures2_jar: Optional[Path] = None
    pdffigures2_timeout_sec: int = 180
    pdffigures2_headless: bool = True

    docling_enable_ocr: bool = True
    docling_ocr_lang: str = "eng"
    docling_extract_figures: bool = True
    docling_table_structure_enabled: bool = True

    analysis_max_text_chars: int = 120000
    analysis_max_tables: int = 50
    analysis_max_figures: int = 30
    analysis_max_supp_items: int = 50
    analysis_local_evidence_first_enabled: bool = True
    analysis_text_llm_enabled: bool = True
    analysis_text_llm_batch_max_chars: int = 9000
    analysis_local_text_llm_batch_max_chars: int = 9000
    analysis_local_text_preselection_enabled: bool = True
    analysis_local_text_preselection_max_chunks: int = 12
    analysis_local_text_preselection_min_chunks_per_section: int = 2
    analysis_local_text_cache_enabled: bool = True
    analysis_local_text_global_cache_enabled: bool = False
    analysis_local_text_cache_min_packets: int = 8
    analysis_local_text_cache_min_sections: int = 3
    analysis_local_text_cache_max_missing_evidence_rate: float = 0.25
    analysis_parallel_modalities_enabled: bool = True
    analysis_parallel_modality_workers: int = 4
    analysis_text_subprocess_guard_enabled: bool = True
    analysis_text_subprocess_timeout_sec: int = 600
    analysis_local_text_subprocess_timeout_sec: int = 360
    analysis_local_text_subprocess_retry_enabled: bool = False
    analysis_modality_subprocess_guard_enabled: bool = True
    analysis_modality_subprocess_timeout_sec: int = 240
    analysis_nontext_llm_enabled: bool = False
    analysis_force_nontext_llm_for_openai: bool = True
    analysis_local_figure_caption_first_enabled: bool = True
    analysis_local_figure_caption_first_min_chars: int = 80
    analysis_local_supplement_caption_first_enabled: bool = True
    analysis_local_supplement_caption_first_min_chars: int = 80
    analysis_verifier_enabled: bool = True
    analysis_section_verifier_enabled: bool = True
    analysis_summary_polish_enabled: bool = True
    analysis_summary_polish_subprocess_guard_enabled: bool = True
    analysis_summary_polish_subprocess_timeout_sec: int = 90
    analysis_narrative_overrides_enabled: bool = True
    analysis_narrative_overrides_subprocess_guard_enabled: bool = True
    analysis_narrative_overrides_subprocess_timeout_sec: int = 120
    analysis_section_extraction_enabled: bool = True
    analysis_section_extraction_max_points_per_section: int = 8
    analysis_section_extraction_subprocess_guard_enabled: bool = True
    analysis_section_extraction_subprocess_timeout_sec: int = 120
    analysis_section_synthesis_v2_enabled: bool = False
    analysis_section_synthesis_v2_llm_enabled: bool = False
    analysis_exec_summary_second_pass_enabled: bool = False
    analysis_schema_validation_enabled: bool = False
    sectioned_report_v3_enabled: bool = True
    media_legend_max_chars: int = 6000
    media_ocr_legend_max_chars: int = 1800

    figure_ocr_enabled: bool = True
    figure_ocr_langs: str = "en"
    figure_ocr_max_chars: int = 4000
    figure_ocr_parse_enabled: bool = True
    figure_fallback_max_pages: int = 6
    figure_fallback_scale: float = 2.0

    retain_source_files: bool = True

    doctr_enabled: bool = True
    doctr_det_arch: str = "db_resnet50"
    doctr_reco_arch: str = "crnn_vgg16_bn"
    doctr_max_chars: int = 4000
    torch_device: str = "mps"

    tatr_det_model: str = "microsoft/table-transformer-detection"
    tatr_struct_model: str = "microsoft/table-transformer-structure-recognition"
    tatr_threshold: float = 0.6

    @property
    def resolved_llm_text_model_path(self) -> Path:
        if self.llm_text_model_path is not None:
            return self.llm_text_model_path
        return self.models_dir / "Qwen2.5-7B-Instruct-Q4_K_M.gguf"

    @property
    def resolved_llm_deep_model_path(self) -> Path:
        if self.llm_deep_model_path is not None:
            return self.llm_deep_model_path
        return self.models_dir / "Qwen2.5-14B-Instruct-Q4_K_M.gguf"

    @property
    def resolved_llm_vision_model_path(self) -> Path:
        if self.llm_vision_model_path is not None:
            return self.llm_vision_model_path
        return self.llm_model_path

    @property
    def resolved_llm_vision_mmproj_path(self) -> Path:
        if self.llm_vision_mmproj_path is not None:
            return self.llm_vision_mmproj_path
        return self.llm_mmproj_path

    @property
    def llm_provider_normalized(self) -> str:
        provider = str(self.llm_provider or "local").strip().lower()
        if provider in {"openai", "api"}:
            return "openai"
        return "local"

    @property
    def openai_configured(self) -> bool:
        return bool(self.resolved_openai_api_key)

    @property
    def resolved_openai_api_key(self) -> str:
        def is_placeholder(value: str) -> bool:
            return value in {"", "YOUR_OPENAI_API_KEY", "OPENAI_API_KEY_PLACEHOLDER"}

        app_key = str(self.psychpaper_openai_api_key or "").strip()
        if not is_placeholder(app_key):
            return app_key
        key = str(self.openai_api_key or "").strip()
        if not is_placeholder(key):
            return key
        secrets_path = ROOT_DIR / "secrets" / "openai.env"
        try:
            for line in secrets_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("PSYCHPAPER_OPENAI_API_KEY="):
                    secret_key = line.split("=", 1)[1].strip()
                    if not is_placeholder(secret_key):
                        return secret_key
                    continue
                if not line.startswith("OPENAI_API_KEY="):
                    continue
                secret_key = line.split("=", 1)[1].strip()
                if not is_placeholder(secret_key):
                    return secret_key
        except Exception:
            return "" if is_placeholder(key) else key
        return "" if is_placeholder(key) else key

    @property
    def analysis_local_evidence_first_active(self) -> bool:
        return self.llm_provider_normalized == "local" and bool(self.analysis_local_evidence_first_enabled)

    @property
    def effective_analysis_nontext_llm_enabled(self) -> bool:
        if self.analysis_local_evidence_first_active:
            return False
        return bool(
            self.analysis_nontext_llm_enabled
            or (self.llm_provider_normalized == "openai" and self.analysis_force_nontext_llm_for_openai)
        )

    @property
    def effective_analysis_nontext_stage_enabled(self) -> bool:
        return bool(self.effective_analysis_nontext_llm_enabled or self.analysis_local_evidence_first_active)

    @property
    def effective_analysis_verifier_enabled(self) -> bool:
        return bool(self.analysis_verifier_enabled and not self.analysis_local_evidence_first_active)

    @property
    def effective_analysis_section_verifier_enabled(self) -> bool:
        return bool(self.analysis_section_verifier_enabled and not self.analysis_local_evidence_first_active)

    @property
    def effective_analysis_summary_polish_enabled(self) -> bool:
        return bool(self.analysis_summary_polish_enabled and not self.analysis_local_evidence_first_active)

    @property
    def effective_analysis_section_extraction_llm_enabled(self) -> bool:
        return bool(self.analysis_section_extraction_enabled and not self.analysis_local_evidence_first_active)

    @property
    def effective_analysis_narrative_overrides_enabled(self) -> bool:
        return bool(self.analysis_narrative_overrides_enabled)

    @property
    def effective_analysis_exec_summary_second_pass_enabled(self) -> bool:
        return bool(self.analysis_exec_summary_second_pass_enabled and not self.analysis_local_evidence_first_active)

    @property
    def effective_analysis_section_synthesis_v2_llm_enabled(self) -> bool:
        return bool(self.analysis_section_synthesis_v2_llm_enabled and not self.analysis_local_evidence_first_active)


settings = Settings()


def ensure_dirs() -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    (settings.data_dir / "saved_reports").mkdir(parents=True, exist_ok=True)
