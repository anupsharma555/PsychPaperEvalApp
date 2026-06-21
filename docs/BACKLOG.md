# PaperEval Diagnostic Backlog

This backlog tracks actionable work to improve PsychPaperEvalApp function, reduce errors, and make regressions easier to catch. It is diagnostic-only: tickets here are not fixed until a later implementation pass completes and validates them.

Related benchmark and testing guidance:

- `docs/app-evaluation-benchmark.md`
- `docs/test-quality-backlog.md`
- `benchmarks/app_evaluation_standard.json`

## Diagnostic Backlog - 2026-06-21

### PEA-DIAG-001 - Gate invalid benchmark runs before scoring app ability

- Status: done - fixed on 2026-06-21
- Severity: high
- Area: tests, benchmark, runtime
- Evidence: `scripts/compare_pdf_against_reference.py` can produce scored comparison artifacts after fallback paths in auto mode; `test/text/sharma-et-al-2017-common-dimensional-reward-deficits-across-mood-and-psychotic-disorders-a-connectome-wide-association_run_20260503_184049.json` records `Pipeline fallback engaged`, `fallback_present=true`, and `text_calls_zero=true`.
- Why it matters: Invalid benchmark runs can be mistaken for app-quality evidence, which hides environment failures and produces misleading conclusions about model or pipeline ability.
- Suggested fix: Added a benchmark validity gate that fails ability scoring when the job is not completed, fallback engaged, text LLM calls are zero, section extraction is disabled, source provenance is missing, or diagnostics are missing. Invalid runs are reported as infrastructure failures.
- Validation: Added focused tests in `backend/tests/test_compare_extraction_audit.py` for valid run, fallback run, zero-text-call run, missing diagnostics, status, section-extraction, and source-provenance blockers. Verified with `PYTHONPATH=backend .venv/bin/python -m pytest backend/tests`.
- Notes: This is also tracked in `docs/test-quality-backlog.md` as a P1 test-quality item.

### PEA-DIAG-002 - Convert gold annotations into structured claim fixtures

- Status: done - fixed on 2026-06-21
- Severity: high
- Area: tests, benchmark, scientific quality
- Evidence: `test/text/sharma_2017_chatgpt_extraction.md` contains useful sectioned bullets, but no stable claim IDs, evidence anchors, importance weights, claim taxonomy, numeric fields, or omission severity.
- Why it matters: The benchmark cannot reliably distinguish critical missed scientific claims from low-value paraphrase differences, making quality regressions hard to diagnose.
- Suggested fix: Created structured Sharma gold claims with `claim_id`, `section`, `claim_type`, `importance`, `evidence_quote`, `page_or_anchor`, `expected_entities`, `expected_numbers`, and `priority`. The markdown remains the reviewer-facing reference.
- Validation: Added a fixture validator test that rejects duplicate IDs, invalid sections, missing evidence, malformed numeric expectations, and unsupported priorities. Added comparator failure-mode coverage that maps app statements back to gold claim IDs.
- Notes: Use a small first pass of high-priority claims rather than trying to fully annotate the paper in one edit.

### PEA-DIAG-003 - Turn the 10-paper PDF corpus into acceptance coverage

- Status: partial - structural 10-paper expectations added on 2026-06-21; multi-paper priority claims remain open
- Severity: high
- Area: tests, benchmark, source coverage
- Evidence: `test/metadata.tsv` lists 10 PDFs, but `scripts/audit_test_reports.py` mostly checks structural report readiness, section key presence, modality key presence, text length, repetition, and warnings.
- Why it matters: The current 10-paper run can say the app produced something, but not whether the paper was understood, whether important claims were retained, or where extraction failed.
- Suggested fix: Added lightweight gold expectations for all 10 PDFs: expected section availability, expected figure/table refs, parser coverage floors, source-manifest requirements, and section-boundary wrong-rate ceilings. Still open: add priority claim annotations for at least 3-5 representative papers.
- Validation: Added deterministic fixture validation and executable-manifest consistency tests without live LLM calls. Still open: a script mode that evaluates parser outputs against the 10 local PDFs and emits per-document pass/fail reasons.
- Notes: Include psychiatry/neuroimaging cases in the acceptance set; the current 10 PDFs are broad OA robustness fixtures rather than domain-specific coverage.

### PEA-DIAG-004 - Promote section-boundary ledger gains into regression thresholds

- Status: done - fixed on 2026-06-21
- Severity: medium
- Area: parser, section fidelity, tests
- Evidence: `test/upstream_ab/upstream_ab_batch_20260503_183327.json` shows `section_boundary_ledger` had mean wrong-section rate `0.236` across four documents, while baseline was `0.517`; it also won wrong-section rate on all four sampled documents.
- Why it matters: Section drift causes Methods, Results, Discussion, and Conclusion content to be misread or mis-synthesized, which directly affects report correctness.
- Suggested fix: Added deterministic upstream A/B regression thresholds that assert `section_boundary_ledger` mean and per-document wrong-section rates stay under calibrated ceilings for the saved fixture set.
- Validation: Added `backend/tests/test_upstream_ab_thresholds.py` and verified it with `PYTHONPATH=backend:scripts .venv/bin/python -m pytest backend/tests/test_section_ledger.py backend/tests/test_upstream_ab_thresholds.py`.
- Notes: The threshold should be fixture-specific at first, then broaden as more gold data is added.

### PEA-DIAG-005 - Improve table and supplement recall diagnostics

- Status: partial - recall metrics added on 2026-06-21; supplement asset failure taxonomy remains open
- Severity: high
- Area: parser, multimodal analysis, benchmark
- Evidence: `test/upstream_ab/upstream_ab_batch_20260503_183327.json` reports mean table reference recall `0.15` across media variants. Sharma and 10-paper artifacts repeatedly show missing supplementary table and figure references such as `S1`, `S2`, and later supplement IDs.
- Why it matters: Tables and supplements often contain the methods, cohort details, secondary outcomes, or robustness checks needed for a trustworthy scientific evaluation.
- Suggested fix: Added explicit table/figure/supplement recall metrics to comparator failure-mode output and upstream A/B media metrics. Still open: improve supplement asset association and separate `not extracted`, `not downloaded`, `not parsed`, and `not analyzed` failure reasons in diagnostics.
- Validation: Added focused coverage in `backend/tests/test_compare_extraction_audit.py` and `backend/tests/test_upstream_ab_thresholds.py`; existing parser-ref tests remain in place.
- Notes: Avoid scoring missing supplements as model failures when the source asset was never available.

### PEA-DIAG-006 - Reduce figure OCR artifact text before downstream analysis

- Status: partial - regression guard, live caption-first cleanup, and analysis diagnostic roll-up added on 2026-06-21; UI/scored artifact-rate surfacing remains open
- Severity: medium
- Area: figure analysis, media cleaning, report quality
- Evidence: `test/upstream_ab/upstream_ab_batch_20260503_183327.json` shows `current_caption_plus_ocr` mean artifact text rate `0.606`, compared with `0.456` for caption-first variants. Individual documents show artifact text rates as high as `1.0`.
- Why it matters: Noisy OCR/page artifact text can displace useful figure evidence and pollute downstream synthesis with irrelevant strings.
- Suggested fix: Added deterministic artifact-text-rate thresholds for upstream A/B comparisons, explicit media metric reporting, live caption-first handling that trusts short scientific captions before noisy OCR, and a stable `multimodal_quality` roll-up in `analysis_diagnostics.json`. Still open: expose these quality signals in the desktop UI and scored benchmark summaries.
- Validation: Added synthetic noisy-figure coverage in `backend/tests/test_upstream_ab_thresholds.py`, live figure/supplement media-cleaning coverage in `backend/tests/test_vision_diagnostics.py`, and runner diagnostic coverage in `backend/tests/test_runner_multimodal_quality.py`.
- Notes: Keep OCR fallback for caption-poor figures, but make its use auditable.

### PEA-DIAG-007 - Split report quality scoring by failure mode

- Status: partial - comparator failure-mode split added on 2026-06-21; report-audit script integration remains open
- Severity: medium
- Area: benchmark, diagnostics, tests
- Evidence: `scripts/audit_test_reports.py` computes `completeness`, `organization`, `sensical_content`, and `overall`, but it does not separately score parser recall, section assignment, claim recall, claim precision, numeric fidelity, unsupported-claim rate, table/figure/supplement recall, or synthesis retention.
- Why it matters: A single broad quality score hides the subsystem that needs improvement and can reward readable but scientifically incomplete reports.
- Suggested fix: Added comparator `failure_mode_metrics` with parser recall, section assignment, claim recall, claim precision, numeric fidelity, unsupported-claim proxy, table/figure/supplement recall, synthesis retention, per-section metrics, top missing claims, unsupported claims, and cross-section misassignments. Still open: integrate the same dimensions into `scripts/audit_test_reports.py` if that script remains the canonical stored-report audit surface.
- Validation: Added a synthetic unit test in `backend/tests/test_compare_extraction_audit.py` that isolates the new metric dimensions and verified with the focused comparator tests.
- Notes: Preserve the aggregate score only as a summary, not as the primary diagnostic.

### PEA-DIAG-008 - Add a single supported backend test command

- Status: done - fixed directly on 2026-06-21
- Severity: medium
- Area: tooling, developer experience, CI
- Evidence: Running backend tests with system Python failed because `sqlmodel` was not installed, while `PYTHONPATH=backend .venv/bin/python -m pytest backend/tests` passed. `Makefile` currently has desktop targets but no backend test target.
- Why it matters: Contributors can get false failures or skip tests because the supported invocation is not encoded in tooling.
- Suggested fix: Added `backend-test` and `test-backend` Make targets that use `.venv/bin/python` when present and set `PYTHONPATH` to the backend package.
- Validation: `make backend-test` passed with `150 passed`, `87 warnings`.
- Notes: This was a minor direct fix completed during backlog creation.

### PEA-DIAG-009 - Centralize fallback state so UI, reports, and benchmarks agree

- Status: partial - canonical run validity helper added on 2026-06-21; full UI contract migration remains open
- Severity: high
- Area: runtime, synthesis, diagnostics, desktop UI
- Evidence: Fallback markers appear across `scripts/compare_pdf_against_reference.py`, `backend/app/services/analysis/synthesis.py`, report summary fields, extraction audit fields, and diagnostics. Existing code has many section fallback paths and separately detects provider/report invalidity.
- Why it matters: If fallback state is fragmented, a failed or partial run can be displayed as valid in one surface and invalid in another, increasing the risk of trusting a bad report.
- Suggested fix: Added a canonical `run_validity` builder used by analysis diagnostics, report APIs, and benchmark validity checks. Still open: migrate the React/UI banner logic fully to the same summary instead of section-level fallback inference.
- Validation: Added focused validity and API tests and verified `PYTHONPATH=backend:scripts .venv/bin/python -m pytest backend/tests/test_run_validity.py backend/tests/test_desktop_api.py backend/tests/test_compare_extraction_audit.py`.
- Notes: Keep report-invalid behavior narrower than benchmark-invalid behavior: benchmark evidence fails on fallback use, while the report UI should hard-fail on provider failure, budget stop, zero text calls, or unverifiable diagnostics.

### PEA-DIAG-010 - Add benchmark preflight checks for required services

- Status: partial - model load timing diagnostics added on 2026-06-21; persistent worker/context architecture remains open
- Severity: medium
- Area: benchmark, runtime, developer experience
- Evidence: The stored Sharma benchmark run failed because GROBID was not reachable at `http://localhost:8070`, then produced fallback diagnostic artifacts. `docs/app-evaluation-benchmark.md` recommends pipeline mode for smoke tier.
- Why it matters: Benchmark runs should fail early and clearly when required services are unavailable, instead of spending time producing invalid or misleading artifacts.
- Suggested fix: Add an explicit `--preflight` check or default preflight inside `scripts/compare_pdf_against_reference.py` for parser engine, GROBID reachability, model/provider readiness, output directory safety, and database path isolation.
- Validation: Add tests or a dry-run command that simulates missing GROBID and asserts the script exits before comparison scoring with a clear remediation message.
- Notes: Keep lightweight mode available for diagnostics, but label it as non-release evidence.

### PEA-DIAG-011 - Enforce benchmark tier minimums in the multi-paper runner

- Status: done - fixed on 2026-06-21
- Severity: high
- Area: benchmark, tests, release
- Evidence: `benchmarks/multi_paper_benchmark.json` declares release `minimum_cases=5` and deep `minimum_cases=8`, but `.venv/bin/python scripts/run_multi_paper_benchmark.py --tier release --validate-only` and `--tier deep --validate-only` both exit successfully while selecting only `sharma_2017_reward_deficits`. In `scripts/run_multi_paper_benchmark.py`, selection filters out diagnostic-coverage cases unless `--include-unscored` is set, and the tier metadata is not enforced before returning success.
- Why it matters: A release or deep benchmark can appear valid after exercising only one scored case, which weakens the benchmark gate and can let parser/model regressions escape.
- Suggested fix: Added tier validation that checks selected case count, required reference-scored case count, target reference-scored case count, and optional required domain mix. Release/deep validation now returns non-zero when minima are not met unless `--allow-diagnostic-tier-gap` is supplied.
- Validation: Added tests for smoke/default selection, release failure with one scored case, release pass with included diagnostic cases, and target-reference warnings. Verified `.venv/bin/python scripts/run_multi_paper_benchmark.py --tier release --validate-only` now exits non-zero with a clear minimum-cases error.
- Notes: This complements `PEA-DIAG-003`; it is about enforcing benchmark tier semantics, not creating new gold annotations.

### PEA-DIAG-012 - Add per-case timeout handling to benchmark child processes

- Status: done - fixed on 2026-06-21
- Severity: medium
- Area: benchmark, runtime, reliability
- Evidence: `scripts/run_multi_paper_benchmark.py` executes each comparison with `subprocess.run(..., capture_output=True, text=True)` and no timeout. The underlying comparison path can call GROBID, parser subprocesses, local models, or OpenAI-backed stages.
- Why it matters: A hung parser/model/provider call can stall an entire benchmark run indefinitely and leave no structured per-case timeout result.
- Suggested fix: Added configurable `--timeout-per-case`, passes it to child benchmark runs, catches `subprocess.TimeoutExpired`, and records a failed case with stdout/stderr tails plus a clear timeout reason.
- Validation: Added a unit test with a tiny child command that sleeps beyond the timeout and asserts the summary records `decision=fail` with a timeout failure. Verified with focused benchmark tests and the full backend suite.
- Notes: Keep the default generous enough for real paper runs, but never unbounded.

### PEA-DIAG-013 - Unify multi-paper manifest coverage with the 10-paper expectations fixture

- Status: done - fixed on 2026-06-21
- Severity: medium
- Area: benchmark, tests, docs
- Evidence: `benchmarks/ten_paper_expectations.json` covers all 10 rows from `test/metadata.tsv`, and `backend/tests/test_ten_paper_benchmark_expectations.py` asserts that coverage. `benchmarks/multi_paper_benchmark.json` currently defines 9 cases and omits `07_32843816_PMC7440080.pdf` and `08_32843834_PMC7440026.pdf`.
- Why it matters: Maintaining two partially overlapping benchmark manifests creates drift: one fixture says the 10-paper corpus is covered, while the executable multi-paper runner cannot list or run all 10 cases.
- Suggested fix: Added the missing `07_32843816_PMC7440080.pdf` and `08_32843834_PMC7440026.pdf` diagnostic cases to `multi_paper_benchmark.json`, preserving the 10-paper expectation fixture as the structural gold layer.
- Validation: Added a manifest consistency test comparing deep-tier executable cases against `ten_paper_expectations.json`. Verified `.venv/bin/python scripts/run_multi_paper_benchmark.py --tier deep --include-unscored --validate-only` lists all intended deep-tier cases.
- Notes: This should be resolved before relying on release/deep tier coverage summaries.

### PEA-DIAG-014 - Add CI coverage for backend, benchmark fixtures, and desktop build checks

- Status: done - fixed on 2026-06-21
- Severity: medium
- Area: CI, tooling, release
- Evidence: No `.github` workflow files are present. `Makefile` now has `backend-test`, but there is no CI job to run it. `desktop_ui/package.json` and `desktop_shell/package.json` expose build scripts but no test or lint scripts, so frontend/shell regressions are not automatically checked.
- Why it matters: A local-only verification path depends on the operator remembering to run commands; benchmark fixture drift, backend regressions, and desktop build breakage can land unnoticed.
- Suggested fix: Added `.github/workflows/ci.yml` with offline backend tests, benchmark/gold fixture validation, Vite UI build, and macOS Tauri `cargo check`.
- Validation: Locally validated the encoded gates: `make backend-test`, benchmark manifest validate-only commands, gold fixture validators, `npm --prefix desktop_ui run build`, and `cargo check --manifest-path desktop_shell/src-tauri/Cargo.toml --locked`.
- Notes: Keep CI free of live LLM, GROBID, Docker, and network-dependent benchmark execution unless those jobs are explicitly marked optional/manual.

### PEA-DIAG-015 - Create a PaperEval benchmark-running skill

- Status: partial - repo-local benchmark skill and reference workflow added on 2026-06-21; user-root install/quick-validate remains open
- Severity: medium
- Area: skills, benchmark, developer workflow
- Evidence: Existing local skills include generic test, review, Playwright, and repo-diagnostic skills, but no PaperEval-specific skill. This repo now has app-specific benchmark resources in `docs/app-evaluation-benchmark.md`, `benchmarks/app_evaluation_standard.json`, `benchmarks/multi_paper_benchmark.json`, `benchmarks/ten_paper_expectations.json`, `scripts/run_multi_paper_benchmark.py`, and `scripts/compare_pdf_against_reference.py`.
- Why it matters: Benchmark setup is easy to run incorrectly: pipeline versus lightweight mode, GROBID readiness, scored versus diagnostic cases, invalid fallback artifacts, output paths, and interpretation of comparison JSON all require app-specific process knowledge. A skill would reduce repeated rediscovery and prevent weak benchmark evidence from being treated as release evidence.
- Suggested fix: Added repo-local `.codex/skills/papereval-benchmark/` with a concise `SKILL.md`, helper script, and `references/benchmark-workflow.md`. Still open: decide whether to install/promote it to the user skill root and run the unavailable `quick_validate.py` equivalent.
- Validation: `.venv/bin/python .codex/skills/papereval-benchmark/scripts/papereval_benchmark_qa.py --validate` passed.
- Notes: This skill should reuse repo scripts rather than embedding long command logic in `SKILL.md`.

### PEA-DIAG-016 - Create a PaperEval desktop UI/UX QA skill

- Status: partial - repo-local UI QA skill and checklist added on 2026-06-21; live screenshot forward-test remains open
- Severity: medium
- Area: skills, UI/UX, desktop QA
- Evidence: Existing skills include generic Playwright and frontend QA skills, but no PaperEval-specific desktop QA workflow. The app has screenshots in `docs/images/`, desktop lifecycle scripts in `scripts/desktop_smoke.py` and `scripts/run_app.py`, Tauri/Vite surfaces in `desktop_ui/` and `desktop_shell/`, and UI requirements around readiness, GROBID/provider status, model calls, source coverage, estimated cost, and diagnostics in `docs/app-evaluation-benchmark.md`.
- Why it matters: Generic UI QA will not automatically check the operator-critical PaperEval states: backend readiness, source upload/URL states, report invalid/rerun banners, cost and model-call visibility, section coverage, diagnostics modal readability, and desktop lifecycle cleanup. Missing these can make the app look usable while hiding analysis failures.
- Suggested fix: Added repo-local `.codex/skills/papereval-ui-qa/` with a concise `SKILL.md`, helper script, and `references/ui-qa-checklist.md`. Still open: live Computer Use/Playwright forward-test with screenshots against the desktop report diagnostics flow.
- Validation: `.venv/bin/python .codex/skills/papereval-ui-qa/scripts/papereval_ui_qa.py --static` passed.
- Notes: This should complement, not replace, `PEA-DIAG-014` CI/build checks.

### PEA-DIAG-017 - Create a PaperEval single-paper report evaluation skill

- Status: partial - repo-local single-paper review skill and rubric added on 2026-06-21; forward-test remains open
- Severity: high
- Area: skills, report quality, scientific evaluation
- Evidence: The app produces rich artifacts (`analysis_diagnostics.json`, `information_retention_audit.json`, comparison JSON/markdown, report payloads, source manifests), and the backlog already tracks report-quality gaps such as unsupported claims, numeric fidelity, section fidelity, table/supplement recall, and fallback validity. No current skill encodes a repeatable workflow for evaluating one completed paper output end-to-end.
- Why it matters: Single-paper review is where users decide whether to trust an app output. Without a skill, Codex may inspect only the rendered report or only tests, missing whether the report is grounded, whether key claims/numbers were retained, whether it used fallback text, and what concrete app ability should improve next.
- Suggested fix: Added repo-local `.codex/skills/papereval-single-paper-review/` with `SKILL.md` and `references/single-paper-rubric.md` covering artifacts, section review, claim/numeric fidelity, multimodal coverage, fallback/invalidity checks, and trust-decision reporting. Still open: forward-test against an existing Sharma comparison/run artifact.
- Validation: No `quick_validate.py` helper was available; validation was limited to readback plus the existing benchmark/UI skill helpers.
- Notes: This should be the highest-priority skill to create because it directly supports user-facing report trust and ability-improvement diagnosis.

### PEA-DIAG-018 - Make desktop launch feel like one clear app open

- Status: partial - reduced the root to one desktop app and one web app link on 2026-06-21; deeper preflight/status testing remains open
- Severity: medium
- Area: desktop, launcher, onboarding, runtime UX
- Evidence: The launch contract is split across `run_app.sh`, `scripts/run_app.py`, `desktop_shell/src-tauri/src/main.rs`, `desktop_shell/README.md`, and `desktop_packaging/setup.py`. The Tauri shell hardcodes backend port 8000 and waits up to 180 seconds, while `scripts/run_app.py` can choose the first free port from 8000, manages GROBID/Docker, and has separate `--force`, `--notify`, and `--api-only` behavior. `desktop_shell/README.md` documents `make desktop-env`, `make desktop-build`, and launching `./PaperEval.app`, but the repo still exposes several plausible launch paths.
- Why it matters: The app can work technically while still feeling rough to open: users have to know which entry point is canonical, wait through backend/GROBID startup without a clear preflight path, and recover from fixed-port or stale-process issues if the sidecar cannot become ready.
- Suggested fix: Define one canonical "open PaperEval" contract for developers and desktop users. The repo root now keeps `PaperEval.app` as the desktop launch item and adds `PaperEval Web.app` as a link-style opener for the built static web UI. Older app bundles/backups are archived under `archive/desktop-apps/`. Still open: add a fuller preflight/status surface that checks `.venv`, backend port ownership, GROBID/Docker readiness, stale sidecar markers, and logs before or during first open.
- Validation: Add launch-contract tests or smoke checks that cover first launch, secondary launch focus, occupied backend port, missing `.venv`, Docker unavailable, GROBID already running, and clean shutdown. Run `scripts/desktop_smoke.py` plus a manual desktop open after implementation.
- Notes: Do this as a UX/runtime contract change, not just README cleanup; the goal is fewer decisions and clearer recovery when opening the app.

### PEA-DIAG-019 - Consolidate desktop and web icon assets under a canonical source

- Status: partial - removed two unreferenced root screenshot PNGs, moved raw icon assets out of the root, and wired the web favicon/manifest on 2026-06-21
- Severity: low
- Area: assets, desktop packaging, web packaging
- Evidence: The repo root contained four tracked app-image assets: `PaperEval.icns`, `PaperEval-icon.png`, `PsychPaperEvalApp-Image.png`, and `PsychPaperEvalApp-Image2.png`, plus multiple root-level app bundles/backups. `desktop_packaging/setup.py` previously referenced root `PaperEval.icns`, while Tauri references `desktop_shell/src-tauri/icons/icon.png` and `desktop_shell/src-tauri/icons/PaperEval.icns`. `rg` found no code references to the two `PsychPaperEvalApp-Image*.png` screenshots or to root `PaperEval-icon.png`.
- Why it matters: Repeated and ambiguously named image assets make it easy to package the wrong icon, keep stale screenshots in the root, or forget to wire a web favicon/manifest icon even though a web icon source exists.
- Suggested fix: Keep one canonical desktop icon and one canonical web icon source, then generate packaging-specific derivatives from those sources. Raw icon sources now live under `assets/icons/`, and the Vite web app serves `desktop_ui/public/PaperEval-icon.png` for favicon, Apple touch icon, and web manifest use. Remaining work: document the canonical generation flow for the packaging-specific Tauri and macOS derivatives instead of hand-maintaining parallel desktop icon files.
- Validation: Run `rg` for removed asset names, build the desktop shell/package, inspect the `.app` icon, and verify the web build exposes the intended favicon/manifest icon.
- Notes: Current cleanup removed the two large unreferenced root screenshots, moved raw root icon assets to `assets/icons/`, preserved the Tauri icon files needed by current packaging, and updated the legacy desktop packaging icon path.

### PEA-DIAG-020 - Build a latency profile artifact for every completed run

- Status: partial - latency profile artifact, report API exposure, summary field, diagnostics UI block, text prompt-call timing, subprocess timeout diagnostics, shorter local text timeout, local figure/supplement caption-first skips, explicit local fast-evidence workflow profile, local synthesis LLM-gate cleanup, cache/reuse summary fields, and focused tests added on 2026-06-21; benchmark roll-up remains open
- Severity: high
- Area: runtime, diagnostics, speed
- Evidence: The active local benchmark run `test/active_benchmark/iterative_20260621/04_sharma_2017_reward_deficits_clean_local_retry/sharma_2017_reward_deficits/record.json` showed job `173` still in text analysis after 8 minutes at progress `0.616`. The backend already records `parse_timing`, `pipeline_timeline`, `model_usage`, prompt budget diagnostics, and `[pipeline] model_timing`, but there is no single run-facing latency breakdown that ranks parse, text, table, figure, supplement, reconcile, synthesis, and write stages.
- Why it matters: Speed work needs a measurable bottleneck list. Without a stable profile artifact, optimization can drift toward easy code changes instead of the slowest stage for the current paper/model.
- Suggested fix: Added `latency_profile.json` beside `analysis_diagnostics.json`, and exposed the same summary in the report API, report summary API, and diagnostics panel. Text analysis now records per-batch prompt-call timing and the slowest batch anchors. Subprocess-guarded stages now preserve timeout/exit/payload diagnostics in stage usage and latency profiles. The local text subprocess fallback budget now defaults to 360s while the general text subprocess budget remains 600s and can be overridden through env config. Local figure and supplement analysis now skip expensive vision calls for caption-rich figures with scientific signal, preserving extractive packets and recording skipped-vision counts. Standard local analysis now runs evidence-first for table/figure/supplement stages, emitting extractive evidence while avoiding local nontext LLM calls and late synthesis polish/verifier passes on the critical path. Synthesis now also skips late LLM section extraction and section fidelity verification during local evidence-first runs while preserving deterministic section compaction and explicit skip diagnostics. Latency profiles now merge timeline metadata into ranked stages and expose parser reuse/text cache hits through `cache_summary` and quality flags. Still open: add local model kind, CPU/GPU mode, queue wait for all stages, non-text top slow calls, and a benchmark roll-up that compares profile deltas across runs.
- Validation: Added synthetic diagnostics tests proving the profile ranks slow stages deterministically, preserves model/prompt timing fields, marks missing timing as unknown rather than zero, and surfaces parser reuse/text cache hits. Added report API/summary tests for latency profile exposure. Still open: validate cache-hit wall-time savings on a repeated completed Sharma local run before using the profile to prioritize implementation.
- Notes: This should be the first speed item because it turns future latency discussions into evidence instead of estimates.

### PEA-DIAG-021 - Reduce serial local text-analysis batches before adding more model calls

- Status: partial - local text preselection and same-document local text report cache added on 2026-06-21; live benchmark calibration remains open
- Severity: high
- Area: text analysis, local LLM, speed
- Evidence: `backend/app/services/analysis/text_analysis.py` batches text chunks serially through `chat_text_fast`, records `llm_batches`, `llm_prompt_chars`, and `llm_prompt_blocks`, and the active run spent at least 8 minutes in text analysis. Current local mode intentionally limits batch size with `analysis_local_text_llm_batch_max_chars`, which improves reliability but can create many sequential calls.
- Why it matters: Text analysis is often the longest stage and blocks later table/figure/synthesis work. Reducing unnecessary text LLM calls is a no-cost latency improvement that also reduces local GPU contention.
- Suggested fix: Added a deterministic preselector before local text LLM batching: keep high-signal chunks by section, numeric/statistical density, medication/intervention/readout terms, table/figure references, and section-boundary confidence; skip low-signal boilerplate from the local LLM prompt path while preserving full chunks for deterministic fallback and section backfill. Added a same-document local text-analysis cache keyed by text chunks, local model settings, text-analysis settings, and source-file hashes; only successful local text reports are cached, while fallback reports force future retries. Still open: calibrate max chunks/section targets on completed local benchmark profiles and measure cache-hit wall-time savings on repeated app runs.
- Validation: Added focused unit coverage that verifies local preselection reduces LLM input chunks while preserving medication/statistical signal and does not apply to the OpenAI provider. Added cache tests proving matching local text signatures reuse reports without model calls and settings changes invalidate cached reports. Still open: compare old/new prompt batch counts, total prompt chars, claim recall, section coverage, and gold compatibility on the Sharma fixture plus at least two non-psychiatry corpus papers.
- Notes: Treat this as relevance-preserving compression, not generic truncation.

### PEA-DIAG-022 - Cache deterministic parse, OCR, and extraction intermediates by source hash

- Status: partial - same-document parser reuse manifest added on 2026-06-21; cross-document content-addressed cache remains open
- Severity: high
- Area: parser, media, benchmark, speed
- Evidence: Current repeated benchmark/debug runs reprocess the same local PDFs while code changes target synthesis, comparator, or UI behavior. Parser, GROBID TEI, section ledger, figure/table chunks, OCR/caption cleanup, and deterministic extractive evidence are expensive but often unchanged between runs.
- Why it matters: Iterating on local model prompts should not pay full parse/OCR cost for every run. Caching would shorten development runs and make benchmark retries less disruptive.
- Suggested fix: Added a same-document parser reuse manifest keyed by source asset SHA-256 plus parser-relevant settings. When a rerun sees matching assets, settings, existing chunks, and stored counts, parsing is skipped and diagnostics record `parser_reuse`; changed source bytes invalidate the manifest and force a fresh parse. Still open: add a cross-document content-addressed cache keyed by PDF bytes, parser version, GROBID version, section-ledger version, OCR/media-cleaning version, and settings that affect extraction. Reuse cached chunks and deterministic diagnostics when only LLM/synthesis/comparator code changed. Add an explicit cache-bypass flag for parser work.
- Validation: Added parser tests proving unchanged same-document assets reuse chunks without calling parsers and changed asset bytes force a reparse. Still open: run the same PDF twice in the full app and assert the second run records parse reuse, lowers wall time, and produces identical extracted chunks. Change a parser-version key and assert cache invalidation.
- Notes: Keep cache provenance visible in diagnostics so benchmark evidence remains auditable.

### PEA-DIAG-023 - Avoid repeated local model cold starts and duplicated model loads

- Status: open
- Severity: high
- Area: local LLM runtime, speed, memory
- Evidence: Active logs repeatedly show local llama context startup lines such as `llama_context: n_ctx_seq (8192) < n_ctx_train (40960)` during long local runs. The code routes text, deep synthesis, and vision through `chat_text_fast`, `chat_text_deep`, and `chat_with_images`; if contexts are loaded per call or per process without reuse, local runs pay avoidable startup latency and memory pressure.
- Why it matters: Local models avoid API cost, but cold starts can dominate perceived latency and make iterative paper runs feel stalled.
- Suggested fix: Added explicit lifecycle diagnostics showing local model load attempts, load errors, and load seconds separately from generation call seconds in usage snapshots, stage deltas, merged run totals, and latency profiles. Still open: audit completed local runs and add a persistent local inference worker or context pool per model path/config if cold starts are material. Keep fast text, deep text, and vision contexts warm where memory allows, with a safe teardown path for app stop.
- Validation: Added focused tests for model-load counters, runner usage merging, and latency-profile cold-start flags. Still open: compare first-call and subsequent-call latency on the same prompt in a live local run. Ensure no stale context survives model-path/settings changes before adding a persistent worker.
- Notes: This should be measured before changing model architecture because local memory pressure can offset warm-pool gains.

### PEA-DIAG-024 - Make modality parallelism adaptive rather than globally conservative

- Status: open
- Severity: medium
- Area: runner, local LLM, speed
- Evidence: `backend/app/services/analysis/runner.py` can run modality stages in parallel only when `_modalities_can_run_parallel()` allows it and `analysis_parallel_modality_workers > 1`; current status reports `worker_capacity=1` for the active backend, and local runs often serialize text, table, figure, and supplement work.
- Why it matters: Some modality work is CPU/parser-bound or uses already extracted text, while other work uses the local model. A single global worker limit leaves safe overlap unused.
- Suggested fix: Split stage scheduling into resource classes: parser/OCR, local text model, local vision model, deterministic extraction, and synthesis. Local fast-evidence mode now allows deterministic table/figure/supplement extraction to overlap with text LLM work while avoiding simultaneous local nontext model calls. Still open: add explicit resource-class scheduling, cache lookup overlap, and backpressure based on measured GPU/runtime state.
- Validation: Use a synthetic runner test to prove resource-class scheduling order, then compare wall time and failure rate on one paper with and without adaptive modality scheduling.
- Notes: Do not blindly increase worker count; the goal is safe overlap, not GPU thrash.

### PEA-DIAG-025 - Add early partial results and stage-level ETA to reduce perceived latency

- Status: open
- Severity: medium
- Area: UI, API, runtime UX, speed
- Evidence: Current job messages can report broad states like `Still analyzing text with local model/runtime (8 min elapsed)`, but the user cannot see which batch/stage is slow, whether tables/figures are already usable, or whether enough evidence exists for a preliminary readout.
- Why it matters: Even if total local runtime remains long, the app can feel much faster if it exposes useful partial evidence and a credible stage timeline.
- Suggested fix: Persist partial stage outputs as they complete and show a live diagnostics strip: current stage, batch index/total, elapsed time, recent model call duration, extracted section coverage, table/figure packet count, and preliminary report availability. Let users inspect completed parser/table/figure outputs while synthesis continues.
- Validation: Add API tests for partial diagnostics shape and UI tests for long-running job states. Confirm a running job exposes section/table/figure counts before final report completion.
- Notes: This complements backend speed work; it reduces waiting blindness without weakening final report quality.

### PEA-DIAG-026 - Protect the canonical normal app port from benchmark/editor stop commands

- Status: open
- Severity: medium
- Area: runtime, launcher, developer workflow
- Evidence: During live local runs on 2026-06-21, the standard `http://127.0.0.1:8000/web/` backend was repeatedly stopped by `POST /api/stop` while a separate benchmark backend continued on `19184`. The user expects `8000/web/` to be the normal app and other ports to be for benchmark/frontend editing.
- Why it matters: Repeated stop/restart cycles interrupt local jobs, recover stale jobs, and make the app appear unreliable even when the model pipeline is functioning.
- Suggested fix: Add port/session ownership to `/api/stop`: require a launcher token, role, or explicit force flag when stopping the canonical app port. Benchmark helpers should target their own API base and avoid controlling `8000` unless explicitly requested. The UI should label which backend it is controlling.
- Validation: Add route tests that a benchmark/editor client cannot stop an unrelated canonical backend without the expected token, and launcher tests that authorized stop still works.
- Notes: This is operational latency: avoiding accidental restarts is often faster than optimizing code.

### PEA-DIAG-027 - Benchmark smaller local synthesis models against quality floors

- Status: open
- Severity: medium
- Area: local LLM, benchmark, speed
- Evidence: Current local status reports Qwen3 text, Qwen3 deep, and Qwen3VL model files with GPU active. The active bottleneck appears local text generation, but no benchmark record currently compares a smaller/faster local text or synthesis model against gold compatibility and report quality floors.
- Why it matters: A smaller local model may be good enough for extraction or preliminary synthesis while the larger model is reserved for final reasoning, reducing wall time without API cost.
- Suggested fix: Add a benchmark matrix mode that runs the same paper with configurable local text/deep/vision model paths and generation limits. Evaluate latency profile, gold compatibility, numeric fidelity, unsupported claim rate, and section coverage. Only promote a faster model path if quality floors hold.
- Validation: Run at least Sharma plus two broad corpus papers; compare wall time and compatibility metrics. Store matrix results under `test/active_benchmark/` with model path/config provenance.
- Notes: Do not switch baselines only for speed unless scientific extraction quality is preserved.

### PEA-DIAG-028 - Add prompt-cost and latency budgets as first-class acceptance criteria

- Status: open
- Severity: medium
- Area: tests, benchmark, speed
- Evidence: Current diagnostics already count prompt chars and many-batch flags, but benchmark acceptance is still mainly validity/quality oriented. Long local runs can pass quality gates while remaining too slow for normal use.
- Why it matters: Speed regressions should be caught the same way section fidelity and gold compatibility regressions are caught.
- Suggested fix: Add optional benchmark thresholds for max wall time, max text batches, max total prompt chars, max single prompt chars, max model calls per modality, and max queue wait. Keep them warning-only until latency profile data is stable, then promote release-tier ceilings.
- Validation: Add tests for threshold evaluation and failure messaging. Use completed local benchmark profiles to calibrate realistic warning and failure floors.
- Notes: Keep latency thresholds tier-specific; smoke, release, and deep runs should not share the same time budget.
