from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Ensure settings are deterministic during tests regardless of invocation cwd.
os.environ.setdefault("PAPER_EVAL_ROOT", str(PROJECT_ROOT))
os.environ.setdefault("DB_PATH", str(PROJECT_ROOT / "data" / "app.db"))
os.environ.setdefault("ANALYSIS_USE_PROCESS_POOL", "false")
