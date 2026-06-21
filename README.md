# PsychPaperEvalApp

Desktop + backend pipeline for evaluating psychiatric research papers from URL/DOI or manual PDF upload.

The app parses paper assets, runs section-grounded multimodal analysis, and writes a structured report with diagnostics for model usage, source coverage, and extraction quality.

## UI Snapshot

### Main Dashboard

![PaperEval Desktop Main Dashboard](docs/images/papereval-desktop-main-dashboard.png)

### Detailed Analysis Modal

![PaperEval Desktop Detailed Analysis](docs/images/papereval-desktop-detailed-analysis.png)

## Repo Scope

This repository tracks source code, tests, and configuration templates.

It intentionally does **not** track:
- local model weights (`models/*.gguf`, `mmproj`, etc.)
- runtime/job data (`data/`)
- OpenAI usage ledgers (`data/openai_usage_ledger.jsonl`)
- local DB files (`*.db`)
- local env secrets (`backend/.env`)
- build artifacts (`desktop_ui/dist`, `desktop_shell/src-tauri/target`, packaged `.app`)

## Quick Setup

1. Create Python env and install backend deps:
```bash
python3 -m venv .venv
.venv/bin/pip install -r backend/requirements.txt
```
2. Create local env config:
```bash
cp backend/.env.example backend/.env
```
3. Add local model files under `models/` (not tracked by git).
4. Start the backend directly when you do not need the desktop shell:
```bash
./run_app.sh
```

## Running PaperEval

Use the desktop app for the normal operator workflow. The desktop shell starts the backend in local-provider mode and supervises shutdown.

Use the backend-served browser UI for standard web review:

```bash
make web-app
```

The canonical browser URL is:

```text
http://127.0.0.1:8000/web/
```

Use this URL for report review, deep links, and app QA that should match the packaged UI. Avoid run-specific query strings as the project standard.

Use the Vite dev server only while editing frontend code:

```bash
make web-dev
```

The dev URL is `http://127.0.0.1:5184/`. After frontend edits, rebuild and verify the same behavior on `http://127.0.0.1:8000/web/`.

## Models and Providers

PaperEval is local-first. Normal launches force `LLM_PROVIDER=local` even if the parent shell has an OpenAI provider configured.

Local GPU/Metal is preferred with `LLM_N_GPU_LAYERS=999`. Backend startup runs a guarded local-model smoke check in a subprocess. If Metal crashes or times out, the app falls back to CPU and reports the effective GPU mode in `/api/status`.

Local model readiness, local live runs, and local report validation are the primary test path. OpenAI routing tests remain as secondary compatibility checks so the optional provider path does not regress, but they are not the default runtime or the main quality gate.

OpenAI is explicit opt-in only:

```bash
python scripts/run_app.py --llm-provider openai
```

Set provider credentials only in local env files that are ignored by git. Do not commit API keys.

## GROBID and Docker

`scripts/run_app.py` manages GROBID by default. It starts or adopts an expected GROBID container on `localhost:8070`, and stops app-owned/adopted containers when the launcher exits.

If Docker Engine is already running, Docker Desktop does not need to stay open. If the engine is off on macOS, the launcher can start Docker Desktop hidden in the background, wait for Docker, and then start GROBID.

## Diagnostics

Each run writes parse, source, timing, coverage, model-usage, and error diagnostics under the generated document artifact directory. Runtime data is intentionally ignored by git.

The UI surfaces backend readiness, GROBID status, model/provider status, source coverage, timing, media coverage, and report validity. Invalid or fallback-heavy reports are flagged instead of being presented as verified output.

## App Evaluation Benchmark

Use the benchmark assets when judging parser changes, prompt changes, model/provider changes, UI report rendering, or release readiness.

- Human-readable standard: `docs/app-evaluation-benchmark.md`
- Machine-readable scorecard: `benchmarks/app_evaluation_standard.json`
- Multi-paper manifest and runner: `benchmarks/multi_paper_benchmark.json`, `scripts/run_multi_paper_benchmark.py`
- Final-report gold standards: `benchmarks/gold_standards/*.json`
- Repo-local benchmark skill: `.codex/skills/papereval-benchmark/SKILL.md`

The benchmark focuses on source integrity, evidence retention, section fidelity, multimodal coverage, operational reliability, runtime visibility, and app usability.

## Pipeline Notes

- Parser and ingestion write source manifests and richer asset diagnostics.
- Text, table, figure, supplement, reconcile, and synthesis stages write timestamped diagnostics.
- Local GPU is preferred; CPU fallback is guarded and reported.
- Reconcile and synthesis wait for upstream modality work before producing the final report.
- Synthesis produces sectioned Introduction, Methods, Results, Discussion, and Conclusion summaries with evidence and warning metadata.
- Report UI review should use the canonical backend-served web app unless testing desktop/Tauri behavior or frontend hot reload.

## Git Version Check

Compare the local checkout with GitHub:

```bash
python3 scripts/git_version_report.py --fetch
```

Add `--show-files` only when you want the changed-file list in the output.
