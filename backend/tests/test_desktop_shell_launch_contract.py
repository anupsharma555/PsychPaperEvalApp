from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DESKTOP_MAIN = PROJECT_ROOT / "desktop_shell" / "src-tauri" / "src" / "main.rs"


def _backend_launch_block() -> str:
    source = DESKTOP_MAIN.read_text()
    start = source.index('.arg("scripts/run_app.py")')
    end = source.index("let child = cmd", start)
    return source[start:end]


def test_desktop_shell_forces_local_backend_provider() -> None:
    launch_block = _backend_launch_block()

    assert '.arg("--llm-provider")' in launch_block
    assert '.arg("local")' in launch_block
    assert '.env("LLM_PROVIDER", "local")' in launch_block
    assert "PAPER_EVAL_CLAIM_GROBID_ON_START" not in launch_block
