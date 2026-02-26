from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(os.getenv("PAPER_EVAL_ROOT", Path(__file__).resolve().parents[3]))
ENV_FILES = [
    str(ROOT_DIR / "backend" / ".env"),
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

    parser_engine: str = "validated"

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
    llm_text_max_tokens: int = 900
    llm_deep_max_tokens: int = 700
    llm_vision_max_tokens: int = 700
    llm_image_max_dim: int = 1024
    llm_image_max_pixels: int = 1500000
    llm_image_format: str = "jpeg"
    llm_image_quality: int = 85

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
    analysis_text_llm_enabled: bool = True
    analysis_text_subprocess_guard_enabled: bool = True
    analysis_text_subprocess_timeout_sec: int = 180
    analysis_modality_subprocess_guard_enabled: bool = True
    analysis_modality_subprocess_timeout_sec: int = 240
    analysis_nontext_llm_enabled: bool = False
    analysis_verifier_enabled: bool = True
    analysis_summary_polish_enabled: bool = True
    analysis_summary_polish_subprocess_guard_enabled: bool = True
    analysis_summary_polish_subprocess_timeout_sec: int = 90
    analysis_narrative_overrides_enabled: bool = False
    analysis_narrative_overrides_subprocess_guard_enabled: bool = True
    analysis_narrative_overrides_subprocess_timeout_sec: int = 120
    analysis_section_extraction_enabled: bool = True
    analysis_section_extraction_max_points_per_section: int = 8
    analysis_section_extraction_subprocess_guard_enabled: bool = True
    analysis_section_extraction_subprocess_timeout_sec: int = 120
    analysis_exec_summary_second_pass_enabled: bool = False
    analysis_schema_validation_enabled: bool = False
    sectioned_report_v3_enabled: bool = False

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


settings = Settings()


def ensure_dirs() -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    (settings.data_dir / "saved_reports").mkdir(parents=True, exist_ok=True)
