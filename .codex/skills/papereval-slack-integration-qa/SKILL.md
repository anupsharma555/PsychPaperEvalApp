---
name: papereval-slack-integration-qa
description: Test and triage the headless PsychPaperEvalApp integration from the local keystone-slack repo. Use when validating @KNI psychpaperapp or @KNI psychpaperevalapp Slack workflows, PaperEval API-only backend startup/reuse, URL or allowlisted local PDF submission, job polling, Slack summary rendering, local /web report deep links, and API/export fallbacks without launching the desktop app, browser, or frontend dev server.
---

# PaperEval Slack Integration QA

## Overview

Use this skill to validate the Keystone Slack integration surface for PsychPaperEvalApp from the PsychPaperEvalApp repo. Keep proof focused on backend-only behavior, mocked Slack/API tests before live Slack, and evidence that the final Slack result includes a concise summary plus links to the full local report.

## Repo Boundaries

- PsychPaperEvalApp repo: current repository root.
- Keystone Slack repo: sibling/local checkout configured by the operator.
- Do not edit Keystone Slack unless the user explicitly asks in the current turn.
- Do not revert or overwrite unrelated work in either repo.
- Do not print Slack tokens, API keys, or full secret-bearing environment output.
- Do not launch the desktop app, open a browser, or start a frontend dev server. Use `scripts/run_app.py --api-only --backend-port <port>` for backend checks.

## Test Order

1. Inspect current state before running anything:

```bash
cd <PsychPaperEvalApp repo>
git status --short
rg -n "api-only|backend-port|/web|job_id|document_id" scripts backend desktop_ui/src
```

```bash
cd <keystone-slack repo>
git status --short
rg -n "psychpaper|PsychPaper|papereval|KNI_PSYCHPAPER" kni_integrations tests .env.example SLACK_APP_SETUP.md
```

2. Run mocked Keystone Slack tests before live Slack. Prefer the narrow tests first:

```bash
cd <keystone-slack repo>
python3 -m unittest tests.test_psychpaper
python3 -m py_compile kni_integrations/psychpaper.py kni_integrations/workflow_runner.py kni_integrations/slack_socket_mode.py
```

Expected evidence: tests prove URL parsing, allowlisted local PDF handling, backend ensure/submit calls through mocks, Slack started/final rendering, and stable `/web/?job_id=...&document_id=...` link formation.

3. Verify the PaperEval backend in API-only mode only when needed:

```bash
cd <PsychPaperEvalApp repo>
PYTHONPATH=backend .venv/bin/python scripts/run_app.py --api-only --backend-port 8765 --log-file .run/slack-integration-qa.log
```

In a second terminal, check readiness without opening a browser:

```bash
curl -fsS http://127.0.0.1:8765/api/status
curl -I http://127.0.0.1:8765/web/
```

Expected evidence: `/api/status` responds, `/web/` is served from the built static UI if `desktop_ui/dist` exists, and no desktop or frontend dev server starts. Stop the backend after the check unless the user asked to keep it running.

4. Rebuild static web assets only if frontend report deep-link files changed or `/web/` is missing:

```bash
cd <PsychPaperEvalApp repo>
npm --prefix desktop_ui run build
```

Expected evidence: the built UI can read `job_id` and `document_id` query parameters and load summary, full report JSON, and media metadata for completed jobs.

5. Run live Slack only after mocked tests pass and the user expects a live check. Use a non-sensitive public paper URL first:

```text
@KNI psychpaperapp https://example.org/path/to/paper.pdf
@KNI psychpaperevalapp https://example.org/path/to/paper.pdf
```

Expected Slack evidence: a quick started reply, then a final thread reply containing title when available, executive summary, methods or scientific highlights, figure/table counts when available, job and document IDs, a full local `/web` report link, and an API/export fallback link.

## Local PDF Checks

- Confirm the path exists and is a PDF before testing.
- Confirm Keystone config allowlists the parent directory through `KNI_PSYCHPAPER_ALLOWED_LOCAL_ROOTS` or the default roots.
- Treat rejected local paths as the expected secure behavior when they fall outside allowlisted roots.
- Prefer URL tests before local file tests when validating Slack rendering.

## Failure Triage

- Backend unavailable: check Keystone `KNI_PSYCHPAPER_REPO`, `KNI_PSYCHPAPER_PYTHON`, `KNI_PSYCHPAPER_API_BASE_URL`, `KNI_PSYCHPAPER_BACKEND_PORT`, and the configured backend log file.
- GUI or browser opened: verify Keystone starts PaperEval with `scripts/run_app.py --api-only --backend-port <port>` and does not call desktop or web dev server scripts.
- Missing report link: check `KNI_PSYCHPAPER_WEB_BASE_URL`, `desktop_ui/dist/index.html`, and `/web/` static serving from the backend.
- Slack has started reply but no final reply: inspect Keystone background polling timeout, job status endpoint, and Slack post errors without printing tokens.
- Local PDF rejected: confirm the path is absolute, exists, ends in `.pdf`, and is under an allowlisted root.
- Figures or tables absent in the web report: confirm the report summary, full report JSON, and media endpoint responses are loaded for the same `job_id` and `document_id`.

## Reporting

Report the smallest useful evidence set: commands run, pass/fail result, whether the workflow stayed headless, whether mocked tests passed before live Slack, the final Slack-visible fields verified, and any remaining unverified surface.
