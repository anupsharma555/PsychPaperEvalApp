#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str, check: bool = False) -> str:
    result = subprocess.run(
        ["git", "-C", str(ROOT), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if check and result.returncode != 0:
        message = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(message or f"git {' '.join(args)} failed")
    if result.returncode != 0:
        return ""
    return str(result.stdout or "").strip()


def _status_counts(status_lines: list[str]) -> dict[str, int]:
    tracked_changed = 0
    untracked = 0
    staged = 0
    unstaged = 0
    deleted = 0
    for line in status_lines:
        if not line:
            continue
        if line.startswith("??"):
            untracked += 1
            continue
        tracked_changed += 1
        index_state = line[0]
        worktree_state = line[1] if len(line) > 1 else " "
        if index_state != " ":
            staged += 1
        if worktree_state != " ":
            unstaged += 1
        if "D" in (index_state, worktree_state):
            deleted += 1
    return {
        "tracked_changed": tracked_changed,
        "untracked": untracked,
        "staged": staged,
        "unstaged": unstaged,
        "deleted": deleted,
    }


def build_report(*, upstream: str, show_files: bool = False) -> dict[str, Any]:
    branch = _git("branch", "--show-current") or "(detached)"
    head = _git("rev-parse", "--short=12", "HEAD")
    head_full = _git("rev-parse", "HEAD")
    head_subject = _git("log", "-1", "--pretty=%s")
    upstream_sha = _git("rev-parse", "--short=12", upstream)
    upstream_full = _git("rev-parse", upstream)

    ahead = behind = 0
    ahead_behind = _git("rev-list", "--left-right", "--count", f"{upstream}...HEAD")
    if ahead_behind:
        parts = ahead_behind.split()
        if len(parts) == 2:
            behind = int(parts[0])
            ahead = int(parts[1])

    status_lines = _git("status", "--porcelain=v1", "--untracked-files=all").splitlines()
    report: dict[str, Any] = {
        "repo_root": str(ROOT),
        "branch": branch,
        "head": head,
        "head_full": head_full,
        "head_subject": head_subject,
        "upstream": upstream,
        "upstream_head": upstream_sha,
        "upstream_head_full": upstream_full,
        "ahead_of_upstream": ahead,
        "behind_upstream": behind,
        "working_tree": _status_counts(status_lines),
        "is_clean": not status_lines,
    }
    if show_files:
        report["status"] = status_lines
    return report


def print_human(report: dict[str, Any]) -> None:
    tree = report["working_tree"]
    print(f"repo: {report['repo_root']}")
    print(f"branch: {report['branch']}")
    print(f"HEAD: {report['head']} {report['head_subject']}")
    print(f"{report['upstream']}: {report['upstream_head']}")
    print(f"ahead/behind: +{report['ahead_of_upstream']} / -{report['behind_upstream']}")
    print(
        "working tree: "
        f"{tree['tracked_changed']} tracked changed, "
        f"{tree['staged']} staged, "
        f"{tree['unstaged']} unstaged, "
        f"{tree['untracked']} untracked"
    )
    if report.get("status"):
        print("status:")
        for line in report["status"]:
            print(f"  {line}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare local repo state with origin/main.")
    parser.add_argument("--upstream", default="origin/main", help="Upstream ref to compare against.")
    parser.add_argument("--fetch", action="store_true", help="Run git fetch origin before reporting.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument("--show-files", action="store_true", help="Include status filenames in the output.")
    args = parser.parse_args()

    try:
        if args.fetch:
            _git("fetch", "origin", "--prune", check=True)
        report = build_report(upstream=args.upstream, show_files=args.show_files)
    except Exception as exc:
        print(f"git version report failed: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_human(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
