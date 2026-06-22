---
name: papereval-run-qa
description: Repo-specific live paper-run QA workflow for PsychPaperEvalApp. Use when testing the app like a user by opening desktop/web, uploading or selecting a paper, validating input, submitting analysis, monitoring completion, loading the rendered report, then using backend artifacts/tests diagnostically to explain timing, failures, model/provider behavior, and output structure.
---

# PaperEval Run QA

## Core Workflow

Use this skill for one live paper run or a small set of recent runs. The default workflow is user-surface first: exercise the app through desktop or web as a user would, then use backend artifacts and tests diagnostically. It is not a corpus eval harness. Keep benchmark corpus scoring in `papereval-benchmark`, source-quality review in `papereval-single-paper-review`, and broad interface behavior in `papereval-ui-qa`.

There are two valid modes:

- User-surface E2E QA: default mode. Test like a user by opening the app, uploading/selecting a paper, validating input, submitting analysis, monitoring completion, loading the report, and then checking artifacts.
- Existing-run QA: diagnostic mode. Inspect a completed or failed job/document and optionally verify the rendered report when the user names a run or asks artifact-only questions.

1. Pick the run mode and target.
   - If the user says "do a run", "test the app", "test like a user", or gives no existing job/document target, default to `User-Surface E2E QA`.
   - If the user gives a job id, map it with `sqlite3 data/app.db`.
   - If the user gives a document id, inspect `data/doc_<id>/artifacts/`.
   - If no target is given and the user asked to test an existing run, inspect the newest `analysis_diagnostics.json`.
   - Never assume `job_id == document_id`.
   - Report both ids when available, plus the source mode (upload, DOI, URL) and final status.

2. Reuse or open an app surface early when rendered review is in scope.
   - If the user asks to "do a run" with this skill, assume rendered app review is in scope unless they say artifact-only.
   - First look for an open PaperEval desktop or web surface; reuse it when possible.
   - If no app surface is open, use `App Surface Discovery` to open the canonical web app by default.
   - Keep backend truth as the source of correctness, but establish the visible app surface early so stale/missing rendered state is caught.

3. Read artifacts before proposing fixes.
   - For running or failed jobs, first use the helper's stage diagnostics mode before rerunning:
     `.venv/bin/python .codex/skills/papereval-run-qa/scripts/papereval_run_qa.py --job-id <id> --stage-diagnostics`
   - Treat stage diagnostics as a checkpoint map: parse/source health, text/table/figure/supplement availability, model calls/errors, fallback flags, report-assembly artifacts, and whether final diagnostics exist yet.
   - Review `artifact_organization` in stage diagnostics when output quality is good but benchmark misses remain. It checks whether report payload stages are mixed, evidence packets are typed and source-backed, figure/supplement packets are consistently labeled, supplement availability is explicit, and retention losses point to weak LLM inputs.
   - Review `intermediate_stage_index.json` when present. It is the canonical ordered map of source manifest, parsed chunks, modality packets, audited evidence packets, synthesis inputs, retention audit, final report, and runtime diagnostics, including `llm_input_readiness` blockers.
   - Do not rerun the whole paper until the failing or missing stage is identified and a targeted fix or configuration change has been made.
   - `run_timeline.json`: pipeline step timestamps and durations.
   - `analysis_diagnostics.json`: `analysis_timing`, `analysis_timeline`, `model_usage`, `openai_usage`, `coverage`, fallbacks, invalid-report flags.
   - `parse_diagnostics.json`, `parser_asset_diagnostics.json`, `source_manifest.json`: parser/source health.
   - `runtime_events.jsonl` and `lifecycle_events.jsonl` when startup, provider, GROBID, local GPU, shutdown, or retry behavior is relevant.
   - `local_gpu_status.json` when local model loading, Metal/GPU fallback, or native crash behavior is relevant.
   - Report JSON/API payloads when the user asks about output quality.
   - Rendered report state in the desktop or web app when the user asks whether the output is visible, structured correctly, or usable in the app.

4. Validate runtime behavior.
   - Check whether modality stages overlap. Parallel OpenAI modality runs should show text/table/figure/supplement starting together; reconcile and synthesis should wait for modality completion.
   - Rank slowest stages by duration and identify the next optimization target.
   - Call out fallback use, missing supplement/table/figure coverage, model errors, cost/call cap issues, stale timestamps, and mismatched job/document ids.
   - Record effective provider and model mode: local/OpenAI, local GPU active/preferred, CPU fallback, or blocked native runtime.
   - Record GROBID readiness and parser mode when parsing or extraction failed.

5. Run focused backend checks only as diagnostics or regression evidence.
   - Start with `py_compile` for touched Python modules when code changed.
   - Run targeted pytest before broad suites when the live run exposes a failure or the task changed backend behavior.
   - Do not substitute tests for a user-surface run unless the user asked for artifact/test-only QA.
   - Avoid live OpenAI/network runs unless explicitly requested and approved.

6. Verify rendered report structure when app-surface review is in scope.
   - Use this only after backend artifacts identify the target report and when the app is already running or the user asked for app-surface validation.
   - First look for an open PaperEval desktop or web surface; reuse it when possible.
   - If no app surface is open and the user asked for rendered app review, open one using the rules in `App Surface Discovery`.
   - Use Computer Use for the desktop app and Browser for the web app.
   - Confirm visible report structure against backend artifacts/API; do not use UI rendering as the only source of truth.
   - If the finding is primarily layout, accessibility, button behavior, or launch UX, switch to `papereval-ui-qa`.

## Single-Run Checklist

For a normal one-run QA pass, answer these questions before recommending changes:

- Did the app inspect the intended job/document pair?
- Did parsing complete, and did GROBID/source diagnostics show usable main text, tables, figures, and supplements?
- Which pipeline and analysis stages took the longest?
- Did text, table, figure, and supplement analysis run in parallel when configured to do so?
- Did any modality use fallback content, skip LLM calls, hit a budget/call cap, or produce an invalid report marker?
- Was the effective model path expected for the run: local free mode versus explicit OpenAI mode?
- If local mode was used, did the run use GPU, CPU fallback, or a blocked native runtime after a Metal crash?
- Is the final report artifact present, internally valid, and connected to the diagnostics for that same document id?
- If app rendering was requested, does the visible desktop/web report show the same core structure as the backend report for that document?

## User-Surface E2E QA

Use this mode when the user asks to test the app like a user would, asks to "do a run" without naming an existing job, or wants confidence that the desktop/web workflow works end to end.

1. Preflight before submitting anything.
   - Reuse or open the app surface first; prefer canonical web at `http://127.0.0.1:8000/web/` unless desktop behavior is under test.
   - Check `/api/status` for backend, GROBID, processing, provider, and local GPU mode.
   - Do not run OpenAI-backed analysis unless the user explicitly opted into API cost for this run.
   - If local mode is selected but the local llama runtime is blocked after a native Metal crash, stop before submitting and report that the user-surface run cannot complete until local inference is fixed or an explicit OpenAI run is authorized.
   - Confirm a test paper source. Prefer an existing small local fixture or a user-provided PDF; do not guess a new external URL if network or source validity matters.

2. Drive the app as a user would.
   - Use Browser for the web app or Computer Use for desktop.
   - Upload/select the main PDF and supplements through the visible controls.
   - Click Validate and confirm the app accepts the input.
   - Submit analysis only after preflight passes.
   - Monitor queue/progress until completed or failed.
   - Click `Load Latest Report` or the relevant report action after completion.

3. Verify both surfaces of truth.
   - In the app, check selected job/document, status, report readiness, report sections, key findings, diagnostics, discrepancies, source coverage, model usage, and timing/cost/status indicators.
   - In backend artifacts, read `run_timeline.json`, `analysis_diagnostics.json`, parser/source diagnostics, and report JSON/API payloads for the new document id.
   - Compare rendered structure to backend report structure and call out stale UI, missing sections, disabled exports, incorrect selected job, or diagnostics mismatch.

4. Cleanup/report.
   - Do not delete reports or artifacts unless explicitly requested.
   - If this skill started a managed backend/web launcher, say whether it remains running or was stopped.
   - Report the exact source file/URL used, job id, document id, provider, model mode, GROBID status, final run status, slowest stages, and whether rendered report review passed.

## Rendered Report Review

Rendered report review is useful for a single run because backend artifacts can be valid while the app still shows stale, missing, disabled, or poorly structured report content. It is not benchmark/eval by default; it is a display-contract check for the selected run.

Use this sequence:

1. Establish backend truth first.
   - Identify job id and document id.
   - Read report JSON/API payloads and diagnostics.
   - Note expected top-level report sections, key findings, section snapshot, discrepancies, diagnostics, and invalid-report flags.

2. Choose the visible surface.
   - Desktop app: use Computer Use when `PaperEval Desktop` is already open or the user asked to inspect desktop behavior.
   - Web app: use Browser against the backend-served built UI at the stable URL `http://127.0.0.1:8000/web/` for normal report review.
   - Do not make run-specific web app URLs part of the workflow. Query strings such as `?_v=...` are temporary cache-busting tools only.
   - Use the Vite dev UI at `http://127.0.0.1:5184/` only for frontend development or hot-reload checks.
   - Prefer web for generic rendered-report review because it is easier to probe and less intrusive. Prefer desktop for Tauri lifecycle, packaged-app, Finder launch, logs button, or desktop-only behavior.

3. Compare visible state to backend truth.
   - Confirm the selected job/document shown in the UI matches the inspected backend target.
   - If needed, click `Load Latest Report` or the selected report action, but avoid submitting new analysis jobs unless explicitly requested.
   - Check that visible report areas are populated as expected: executive summary, key findings, section snapshot/sections, discrepancies, diagnostics/detailed analysis, source coverage, model usage, and timing/cost/status indicators.
   - Watch for stale state: completed job selected but `Report: Not Ready`, disabled export buttons for a present report, report from a different document, or diagnostics that do not match artifact timestamps.
   - Record whether the app rendering passed, failed, or was not checked.

4. Route findings correctly.
   - Backend report missing or invalid: stay in `papereval-run-qa`.
   - Report scientifically wrong or unsupported by the source paper: switch to `papereval-single-paper-review`.
   - Report visible but layout/controls/accessibility are poor: switch to `papereval-ui-qa`.
   - Report quality needs corpus/gold-standard evidence: switch to `papereval-benchmark`.

## App Surface Discovery

When rendered app review is requested, use this discovery order:

1. Check whether a usable web surface is already running.
   - Probe canonical built UI first: `http://127.0.0.1:8000/web/`.
   - Probe Vite dev UI only when frontend development or hot reload is relevant: `http://127.0.0.1:5184/`.
   - If a web surface is open/reachable, use Browser and do not start another one.

2. Check whether the desktop app is already open.
   - Use Computer Use to inspect the visible `PaperEval Desktop` app state when desktop behavior matters or the user has the desktop app open.
   - Do not click destructive controls, submit a new run, or change persistent state unless explicitly requested.

3. If no surface is open and the user asked for rendered app review, open one.
   - Default to the canonical web surface. If the backend is already running, open `http://127.0.0.1:8000/web/` in Browser.
   - If the backend/web surface is not running, start the managed web launcher with `make web-app` and then use Browser.
   - Open the desktop app only when the user asks for desktop/Tauri validation or when the bug is desktop-specific. Prefer the normal `PaperEval.app` launch item if present.
   - Report what was opened and whether it was newly launched or reused.

4. If startup fails, do not keep retrying blindly.
   - Check `/api/status`, GROBID readiness, lifecycle logs, and local GPU status.
   - Route launch, layout, and control issues to `papereval-ui-qa` if the finding is primarily UI/runtime experience rather than single-run artifact correctness.

## Eval Boundary

Do not tie a single run to benchmark/eval by default. Use this skill alone when the user is diagnosing a concrete run, checking timestamps, explaining a failure, validating a recent fix, or finding the slowest serialized stage.

Escalate to `papereval-benchmark` only when the user asks whether quality improved across papers, wants release-readiness evidence, wants comparison to gold/reference outputs, or needs a random/corpus paper selected by the benchmark harness.

Escalate to `papereval-single-paper-review` when the question is about scientific correctness, hallucination, evidence support, or whether the generated report faithfully represents the paper. `papereval-run-qa` can provide the artifact map first, but it should not substitute artifact timing for source-grounded quality review.

Escalate to `papereval-ui-qa` when the issue is visible app behavior: queue state, buttons, report rendering, diagnostics panels, logs access, layout, accessibility, or desktop/web launch experience.

## Helper

Use the bundled helper for a deterministic local summary:

```bash
.venv/bin/python .codex/skills/papereval-run-qa/scripts/papereval_run_qa.py --job-id 157
.venv/bin/python .codex/skills/papereval-run-qa/scripts/papereval_run_qa.py --job-id 157 --stage-diagnostics
.venv/bin/python .codex/skills/papereval-run-qa/scripts/papereval_run_qa.py --latest --run-checks
```

The helper reads local artifacts only. Use `--stage-diagnostics` for running, failed, or partially completed jobs; it does not require a final report or `analysis_diagnostics.json`. `--run-checks` runs local compile and focused pytest checks.
The stage diagnostics payload includes an `artifact_organization` audit. Use it to decide whether the intermediate data fed to synthesis is well structured enough before changing prompts or rerunning the whole paper.
Future completed runs write `intermediate_stage_index.json` under the document artifacts directory. This additive index organizes existing artifacts into stable stages and flags whether the data sent toward local LLM synthesis is source-backed, typed, and sectioned enough for reliable synthesis.
Newer reports also include `artifact_organization.llm_input_inventory`, a bounded diagnostic inventory of the scientific details selected for LLM-facing synthesis. It stores hashes, refs, sections, modalities, detail types, and quality counts rather than full duplicated prompt text, so use it to verify what synthesis could see before changing prompts or rerunning.

Manual commands that are often useful:

```bash
sqlite3 data/app.db 'select id, document_id, status, progress, updated_at from job order by id desc limit 10;'
curl -fsS http://127.0.0.1:8000/api/status
```

Use live `curl` only when the app/backend is already running or the user asked for runtime verification.

For rendered report review, pair UI observations with backend status:

```bash
curl -fsS http://127.0.0.1:8000/api/status
```

## Reporting

Lead with failures or risks. Include commands run, whether they passed, the job/document ids inspected, and artifact paths. For performance requests, include a short table of slowest stages and a concrete next step. If the answer is based only on stored artifacts, say that; if live status was checked, report the provider/GROBID/local GPU status observed. If rendered report review was performed, state the surface used (`desktop` via Computer Use or `web` via Browser), the visible selected job/document, and whether the displayed report matched backend artifacts.
