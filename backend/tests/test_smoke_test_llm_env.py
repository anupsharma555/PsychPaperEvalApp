from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SMOKE_PATH = PROJECT_ROOT / "scripts" / "smoke_test_llm.py"


def _load_smoke_module():
    spec = importlib.util.spec_from_file_location("smoke_test_llm", SMOKE_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_smoke_test_llm_defaults_to_local_provider() -> None:
    smoke = _load_smoke_module()

    env = smoke._prepare_smoke_env("local", {"LLM_PROVIDER": "openai"})  # noqa: SLF001 - script contract coverage

    assert env["LLM_PROVIDER"] == "local"
    assert env["PAPER_EVAL_ROOT"] == str(PROJECT_ROOT)


def test_smoke_test_llm_can_preserve_environment_provider() -> None:
    smoke = _load_smoke_module()

    env = smoke._prepare_smoke_env("env", {"LLM_PROVIDER": "openai"})  # noqa: SLF001 - script contract coverage

    assert env["LLM_PROVIDER"] == "openai"


def test_smoke_test_llm_env_mode_still_defaults_to_local_when_unset() -> None:
    smoke = _load_smoke_module()

    env = smoke._prepare_smoke_env("env", {})  # noqa: SLF001 - script contract coverage

    assert env["LLM_PROVIDER"] == "local"
