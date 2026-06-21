from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SMOKE_PROVIDER = "local"


def _prepare_smoke_env(provider: str, env: dict[str, str] | None = None) -> dict[str, str]:
    prepared = dict(os.environ if env is None else env)
    prepared["PAPER_EVAL_ROOT"] = str(ROOT)
    if provider != "env":
        prepared["LLM_PROVIDER"] = provider
    else:
        prepared.setdefault("LLM_PROVIDER", DEFAULT_SMOKE_PROVIDER)
    return prepared


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider",
        choices=["local", "openai", "env"],
        default=DEFAULT_SMOKE_PROVIDER,
        help="LLM provider to smoke test. Defaults to local to avoid API spend.",
    )
    parser.add_argument("--image", help="Path to image for multimodal test")
    args = parser.parse_args()

    os.environ.update(_prepare_smoke_env(args.provider))
    sys.path.insert(0, str(ROOT / "backend"))

    from app.core.config import settings  # noqa: WPS433
    from app.services.analysis.llm import chat_text, chat_with_images  # noqa: WPS433

    print(f"Testing provider: {settings.llm_provider_normalized}")
    print("Testing text prompt...")
    text = chat_text("Summarize the key result: a randomized trial found a 25% symptom reduction.")
    print(text)

    if args.image:
        print("Testing image prompt...")
        img = chat_with_images("Describe the chart in one sentence.", [args.image])
        print(img)


if __name__ == "__main__":
    main()
