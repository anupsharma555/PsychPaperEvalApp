from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def pick_file(files: list[str], include: str, exclude: str | None = None) -> str:
    include_lower = include.lower()
    candidates = []
    for name in files:
        lower = name.lower()
        if not lower.endswith(".gguf"):
            continue
        if include_lower not in lower:
            continue
        if exclude and exclude.lower() in lower:
            continue
        candidates.append(name)
    if not candidates:
        raise RuntimeError(f"No file found for pattern '{include}'")
    return sorted(candidates)[0]


def list_gguf_files(api: HfApi, repo: str) -> list[str]:
    files = api.list_repo_files(repo, repo_type="model")
    return [name for name in files if name.lower().endswith(".gguf")]


def download_file(repo: str, filename: str, out_dir: Path) -> None:
    print(f"Downloading {repo} :: {filename}")
    hf_hub_download(
        repo_id=repo,
        filename=filename,
        local_dir=out_dir,
        local_dir_use_symlinks=False,
    )


def _download_vision(args: argparse.Namespace, api: HfApi, out_dir: Path) -> None:
    files = list_gguf_files(api, args.vision_repo)
    model_file = args.vision_model_file or pick_file(
        files,
        args.vision_model_pattern,
        exclude=args.vision_mmproj_pattern,
    )
    mmproj_file = args.vision_mmproj_file or pick_file(files, args.vision_mmproj_pattern)
    download_file(args.vision_repo, model_file, out_dir)
    download_file(args.vision_repo, mmproj_file, out_dir)


def _download_text(args: argparse.Namespace, api: HfApi, out_dir: Path) -> None:
    files = list_gguf_files(api, args.text_repo)
    model_file = args.text_model_file or pick_file(files, args.text_model_pattern)
    download_file(args.text_repo, model_file, out_dir)


def _download_deep(args: argparse.Namespace, api: HfApi, out_dir: Path) -> None:
    files = list_gguf_files(api, args.deep_repo)
    model_file = args.deep_model_file or pick_file(files, args.deep_model_pattern)
    download_file(args.deep_repo, model_file, out_dir)


def _list_selected(args: argparse.Namespace, api: HfApi) -> None:
    selected = [args.profile] if args.profile != "all" else ["vision", "text", "deep"]
    for role in selected:
        if role == "vision":
            repo = args.vision_repo
        elif role == "text":
            repo = args.text_repo
        else:
            repo = args.deep_repo
        print(f"\n[{role}] {repo}")
        for name in list_gguf_files(api, repo):
            print(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=["vision", "text", "deep", "all"],
        default="all",
        help="Which model set to download",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available GGUF files for the selected profile and exit",
    )
    parser.add_argument(
        "--out-dir",
        default="models",
        help="Output directory",
    )

    parser.add_argument("--vision-repo", default="ggml-org/Qwen2.5-VL-7B-Instruct-GGUF")
    parser.add_argument("--vision-model-pattern", default="Q4_K_M")
    parser.add_argument("--vision-model-file", default=None)
    parser.add_argument("--vision-mmproj-pattern", default="mmproj")
    parser.add_argument("--vision-mmproj-file", default=None)

    parser.add_argument("--text-repo", default="bartowski/Qwen2.5-7B-Instruct-GGUF")
    parser.add_argument("--text-model-pattern", default="Q4_K_M")
    parser.add_argument("--text-model-file", default=None)

    parser.add_argument("--deep-repo", default="bartowski/Qwen2.5-14B-Instruct-GGUF")
    parser.add_argument("--deep-model-pattern", default="Q4_K_M")
    parser.add_argument("--deep-model-file", default=None)

    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    api = HfApi()

    if args.list:
        _list_selected(args, api)
        return

    selected = [args.profile] if args.profile != "all" else ["vision", "text", "deep"]
    if "vision" in selected:
        _download_vision(args, api, out_dir)
    if "text" in selected:
        _download_text(args, api, out_dir)
    if "deep" in selected:
        _download_deep(args, api, out_dir)


if __name__ == "__main__":
    main()
