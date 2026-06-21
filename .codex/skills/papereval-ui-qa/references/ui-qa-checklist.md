# UI QA Checklist

Use this reference after reading `../SKILL.md`. It is for visible or static review of the PaperEval desktop/web interface.

## Test Surface

- Desktop shell: packaged `PaperEval.app` or Tauri shell.
- Browser/built UI: backend-served `desktop_ui/dist` at `http://127.0.0.1:8000/web/`; use this as the canonical non-dev browser surface.
- Browser/dev UI: `desktop_ui` served by Vite on `http://127.0.0.1:5184/` against the backend API; use this only for frontend development, hot reload, or diagnosing Vite-specific behavior. After frontend edits, rebuild and recheck the canonical browser surface.
- Static review: frontend source and package checks when no app is running.
- Managed browser launch: `make web-app`, which builds the frontend and opens the backend-served `http://127.0.0.1:8000/web/` surface with managed backend/GROBID lifecycle.
- Managed frontend-dev launch: `make web-dev`, which starts Vite on `http://127.0.0.1:5184/` for hot-reload work.

## Tool Selection

- Use Computer Use for an already-running macOS `PaperEval` desktop app. Capture app state, inspect accessibility labels, and click/type only as needed.
- Use Browser Use or Playwright for localhost browser UI, screenshots, responsive checks, and DOM-level assertions.
- Use shell/static checks for package metadata, source presence, lint/build/test commands, and API route expectations.

The bundled helper does not start servers:

```bash
.venv/bin/python .codex/skills/papereval-ui-qa/scripts/papereval_ui_qa.py --static
.venv/bin/python .codex/skills/papereval-ui-qa/scripts/papereval_ui_qa.py --probe http://127.0.0.1:8000/web/
.venv/bin/python .codex/skills/papereval-ui-qa/scripts/papereval_ui_qa.py --probe http://127.0.0.1:5184/
```

## Core Workflow Checks

- Launch/readiness: backend card, model card, GROBID card, processing state, connection banner, and disabled/enabled workflow controls.
- Source selection: URL, DOI, upload, main PDF and supplements, validation messaging.
- Queue/job progress: queued, running, completed, failed, progress text, stale run recovery.
- Report display: executive summary, sections, findings, diagnostics, coverage, cost/model usage, and timing.
- Diagnostics/logs: runtime status, GROBID health, provider readiness, local GPU/CPU fallback state, OpenAI cost fields, and open logs behavior.
- Error recovery: blocked fetch, missing PDF, OpenAI budget or call failure, parser failure, invalid report.
- Responsive layout: desktop and narrow viewport, no overlapping text, no hidden critical controls.

## Runtime Status Checks

For startup, backend retry, Docker/GROBID, or local-model UI issues, pair visible UI evidence with a lightweight API/log check when available:

```bash
curl -fsS http://127.0.0.1:8000/api/status
```

Confirm the UI does not contradict the API for:

- `provider` and model readiness.
- `local_gpu_mode`, including `gpu_preferred`, `gpu_active`, `cpu_fallback`, or blocked local runtime messaging.
- `grobid.ready` and the displayed GROBID card.
- `processing.running`, inflight count, and workflow action enablement.
- recent events for startup, retry backend, local GPU fallback, or shutdown.

For Docker Desktop, distinguish closing the visible window from quitting Docker Engine. The UI should not imply Docker Desktop must remain visibly open; GROBID only needs the Docker engine/container running.

## UX Review Criteria

Call out issues when there is observable evidence for:

- Unclear disabled controls or missing next actions.
- Stale progress messages or ambiguous run status.
- Poor contrast, overflow, cramped tables, or clipped text.
- Inconsistent terminology between source selection, job progress, reports, and diagnostics.
- Hidden critical controls on narrow screens.
- Accessibility labels that are missing, misleading, or too generic for repeated controls.

For proposed fixes, keep the UI operational and evidence-dense. Avoid marketing-style layouts, oversized hero treatments, and decorative elements that compete with review workflows.

## Desktop Safety

When validating with Computer Use:

1. Confirm the app is already running or ask before launching it.
2. Call `get_app_state` once before clicking.
3. Record visible screen state and accessibility labels relevant to the finding.
4. Avoid deleting reports, changing persistent settings, or submitting live API jobs unless explicitly requested.
5. For launch/exit flows, avoid force-quitting the app or Docker unless the user explicitly asks; prefer the app's Retry Backend or normal close path.
