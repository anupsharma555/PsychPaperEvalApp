from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.services.analysis.llm import chat_text, chat_with_images  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="Path to image for multimodal test")
    args = parser.parse_args()

    print("Testing text prompt...")
    text = chat_text("Summarize the key result: a randomized trial found a 25% symptom reduction.")
    print(text)

    if args.image:
        print("Testing image prompt...")
        img = chat_with_images("Describe the chart in one sentence.", [args.image])
        print(img)


if __name__ == "__main__":
    main()
