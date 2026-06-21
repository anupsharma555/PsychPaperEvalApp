---
name: papereval-ui-qa
description: Repo-specific UI and UX review workflow for the PsychPaperEvalApp desktop/web app. Use when checking source-selection UX, validation flows, queue/job progress, report rendering, diagnostics visibility, logs access, error states, responsive layout, accessibility, or when using Computer Use, Browser Use, Playwright, screenshots, or static frontend checks to validate the PaperEval interface.
---

# PaperEval UI QA

## Core Workflow

Use this skill for app experience validation. Use `papereval-run-qa` for backend artifact diagnosis, `papereval-single-paper-review` for source-grounded report quality, and `papereval-benchmark` for corpus scoring.

1. Decide whether the surface is the desktop shell, canonical browser UI, browser/dev UI, or static frontend review.
2. Prefer observable evidence: screenshot, accessibility label, DOM state, API response, or log path.
3. Validate launch/readiness, source selection, queue progress, report display, diagnostics/logs, error recovery, and responsive layout.
4. Avoid destructive UI actions or live API submissions unless explicitly requested.
5. Keep proposed UI fixes work-focused, dense, and consistent with the existing app.
6. When the task mentions Docker, GROBID, backend retry, local model status, or startup/shutdown, verify visible UI state against `/api/status` or logs before concluding.

For the detailed checklist and tool-selection guidance, read `references/ui-qa-checklist.md`.

## Helper

Use the static helper before live UI work:

```bash
.venv/bin/python .codex/skills/papereval-ui-qa/scripts/papereval_ui_qa.py --static
.venv/bin/python .codex/skills/papereval-ui-qa/scripts/papereval_ui_qa.py --probe http://127.0.0.1:8000/web/
.venv/bin/python .codex/skills/papereval-ui-qa/scripts/papereval_ui_qa.py --probe http://127.0.0.1:5184/
```

The helper does not start servers. It checks expected frontend/desktop files and can probe an already-running local UI.

Use `http://127.0.0.1:8000/web/` as the canonical non-dev browser UI when `desktop_ui/dist` exists. Use `make web-app` to build and open that stable backend-served UI.

Use `make web-dev` and `http://127.0.0.1:5184/` only while editing frontend code or checking hot-reload behavior. When a frontend edit is complete, rebuild and verify the same change on `http://127.0.0.1:8000/web/` before calling the UI check done. Use the desktop app when the user wants packaged/Tauri behavior. Do not start or stop live servers just for static review.

## Computer Use Validation

When validating the desktop app with Computer Use:

1. Confirm the app is already running or ask before launching it.
2. Call `get_app_state` once before clicking.
3. Record visible screen state and accessibility labels relevant to the finding.
4. Avoid destructive actions such as deleting reports, changing settings, or submitting live API jobs unless the user explicitly requests it.

For launch/exit QA, record whether the desktop or web launcher owns the backend, whether GROBID is ready, and whether closing the visible Docker Desktop window is safe. Closing the Docker Desktop window is safe during a run; quitting Docker Engine is not.
