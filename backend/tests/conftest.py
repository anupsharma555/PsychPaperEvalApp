from __future__ import annotations

import os
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Ensure settings are deterministic during tests regardless of invocation cwd.
os.environ.setdefault("PAPER_EVAL_ROOT", str(PROJECT_ROOT))
os.environ.setdefault("DB_PATH", str(PROJECT_ROOT / "data" / "app.db"))
os.environ.setdefault("ANALYSIS_USE_PROCESS_POOL", "false")
os.environ["LLM_PROVIDER"] = "local"
os.environ.setdefault("OPENAI_API_KEY", "OPENAI_API_KEY_FOR_TESTS")
os.environ.setdefault("OPENAI_BASE_URL", "http://127.0.0.1:9/v1")
os.environ.setdefault("OPENAI_TIMEOUT_SEC", "1")
os.environ.setdefault("ANALYSIS_NARRATIVE_OVERRIDES_SUBPROCESS_GUARD_ENABLED", "false")
os.environ.setdefault("ANALYSIS_VERIFIER_SUBPROCESS_GUARD_ENABLED", "false")
os.environ.setdefault("ANALYSIS_SECTION_VERIFIER_ENABLED", "false")


@pytest.fixture(autouse=True)
def _stub_local_llama_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep unit tests in local-provider mode without loading real GGUF models."""
    from app.services.analysis import llm

    class _StubLLM:
        def create_chat_completion(self, messages, temperature=0.2, **kwargs):
            return {"choices": [{"message": {"content": "{}"}}]}

    monkeypatch.setattr(llm, "_load_text_model", lambda: _StubLLM())
    monkeypatch.setattr(llm, "_load_deep_model", lambda: _StubLLM())
    monkeypatch.setattr(llm, "_load_vision_model", lambda: _StubLLM())
