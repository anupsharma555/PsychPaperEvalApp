from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Ensure settings are deterministic during tests regardless of invocation cwd.
os.environ.setdefault("PAPER_EVAL_ROOT", str(PROJECT_ROOT))
os.environ.setdefault("DB_PATH", str(PROJECT_ROOT / "data" / "app.db"))
os.environ.setdefault("ANALYSIS_USE_PROCESS_POOL", "false")
os.environ.setdefault("LLM_PROVIDER", "openai")
os.environ.setdefault("OPENAI_API_KEY", "sk-test")
os.environ.setdefault("OPENAI_BASE_URL", "http://127.0.0.1:9/v1")
os.environ.setdefault("OPENAI_TIMEOUT_SEC", "1")
os.environ.setdefault("ANALYSIS_NARRATIVE_OVERRIDES_SUBPROCESS_GUARD_ENABLED", "false")
os.environ.setdefault("ANALYSIS_VERIFIER_SUBPROCESS_GUARD_ENABLED", "false")
os.environ.setdefault("ANALYSIS_SECTION_VERIFIER_ENABLED", "false")
