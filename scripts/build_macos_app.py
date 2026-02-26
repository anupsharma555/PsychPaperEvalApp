from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Iterable


UI_VERSION = "V8"


def run(cmd: list[str], *, cwd: Path | None = None, env: dict | None = None) -> None:
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None, env=env)


def check_tool(command: list[str], *, hint: str, env: dict | None = None) -> None:
    try:
        subprocess.check_call(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
    except Exception as exc:
        raise RuntimeError(hint) from exc


def ensure_npm_deps(project_dir: Path, *, env: dict | None = None) -> None:
    node_modules = project_dir / "node_modules"
    if node_modules.exists():
        return
    print(f"Installing npm dependencies in {project_dir} ...")
    run(["npm", "install"], cwd=project_dir, env=env)


def sha256_text(value: str) -> str:
    digest = hashlib.sha256()
    digest.update(value.encode("utf-8"))
    return digest.hexdigest()


def hash_paths(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.as_posix().encode("utf-8"))
        try:
            digest.update(path.read_bytes())
        except Exception:
            continue
    return digest.hexdigest()


def hash_tree(root: Path) -> str:
    if not root.exists():
        return sha256_text("")
    files = [path for path in root.rglob("*") if path.is_file()]
    return hash_paths(files)


def git_sha(root: Path) -> str:
    try:
        out = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True)
        return out.strip()
    except Exception:
        return "unknown"


def write_build_manifest(root: Path, *, app_name: str, ui_hash: str, backend_hash: str, shell_hash: str) -> Path:
    run_dir = root / ".run"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "build_manifest.json"
    payload = {
        "app_name": app_name,
        "desktop_version": UI_VERSION,
        "built_at": datetime.now(tz=timezone.utc).isoformat(),
        "git_sha": git_sha(root),
        "ui_hash": ui_hash,
        "backend_command_hash": backend_hash,
        "shell_hash": shell_hash,
        "artifact": str(root / f"{app_name}.app"),
    }
    manifest_path.write_text(json.dumps(payload, indent=2))
    return manifest_path


def sync_primary_bundle(root: Path, source_bundle: Path, app_name: str) -> Path:
    target_bundle = root / f"{app_name}.app"
    if target_bundle.exists():
        shutil.rmtree(target_bundle)
    shutil.copytree(source_bundle, target_bundle, symlinks=True)
    return target_bundle


def write_deprecated_markers(root: Path, *, app_name: str, kept_bundle: Path) -> None:
    dist_dir = root / "dist"
    dist_dir.mkdir(parents=True, exist_ok=True)

    deprecated_bundle = dist_dir / f"{app_name}.app"
    if deprecated_bundle.exists():
        shutil.rmtree(deprecated_bundle)

    marker = dist_dir / "DEPRECATED_DESKTOP_APP.txt"
    marker.write_text(
        "This directory no longer hosts the primary PaperEval desktop bundle.\n"
        f"Use only: {kept_bundle}\n"
    )



def find_tauri_app(shell_root: Path, app_name: str) -> Path:
    direct = shell_root / "src-tauri" / "target" / "release" / "bundle" / "macos" / f"{app_name}.app"
    if direct.exists():
        return direct

    candidates = list((shell_root / "src-tauri" / "target").glob(f"**/{app_name}.app"))
    if candidates:
        return sorted(candidates)[-1]

    raise RuntimeError("Tauri build completed but .app bundle was not found.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PaperEval macOS app using Tauri V2")
    parser.add_argument("--name", default="PaperEval")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()

    root = Path(args.root).resolve()
    desktop_ui = root / "desktop_ui"
    desktop_shell = root / "desktop_shell"
    build_env = os.environ.copy()
    cargo_bin = str(Path.home() / ".cargo" / "bin")
    build_env["PATH"] = f"{cargo_bin}:{build_env.get('PATH', '')}"

    if not desktop_ui.exists():
        raise RuntimeError(f"desktop_ui not found: {desktop_ui}")
    if not desktop_shell.exists():
        raise RuntimeError(f"desktop_shell not found: {desktop_shell}")

    check_tool(["node", "-v"], hint="Node.js is required. Install Node 20+.", env=build_env)
    check_tool(["npm", "-v"], hint="npm is required. Install Node.js/npm.", env=build_env)
    check_tool(["cargo", "--version"], hint="cargo is required. Install Rust toolchain (rustup).", env=build_env)
    check_tool(["rustc", "--version"], hint="rustc is required. Install Rust toolchain (rustup).", env=build_env)

    ensure_npm_deps(desktop_ui, env=build_env)
    ensure_npm_deps(desktop_shell, env=build_env)

    print("Building desktop UI (Vite)...")
    run(["npm", "run", "build"], cwd=desktop_ui, env=build_env)

    print("Building desktop shell (Tauri)...")
    run(["npm", "run", "build"], cwd=desktop_shell, env=build_env)

    tauri_bundle = find_tauri_app(desktop_shell, args.name)
    primary_bundle = sync_primary_bundle(root, tauri_bundle, args.name)
    write_deprecated_markers(root, app_name=args.name, kept_bundle=primary_bundle)

    ui_hash = hash_tree(desktop_ui / "dist")
    backend_hash = sha256_text(
        f"{root / '.venv' / 'bin' / 'python'} scripts/run_app.py --api-only --force --backend-port 8000"
    )
    shell_hash = hash_tree(desktop_shell / "src-tauri")
    manifest_path = write_build_manifest(
        root,
        app_name=args.name,
        ui_hash=ui_hash,
        backend_hash=backend_hash,
        shell_hash=shell_hash,
    )

    print(f"Desktop version: {UI_VERSION}")
    print(f"Primary bundle: {primary_bundle}")
    print(f"Source Tauri bundle: {tauri_bundle}")
    print(f"Build manifest: {manifest_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"build_macos_app.py failed: {exc}", file=sys.stderr)
        sys.exit(1)
