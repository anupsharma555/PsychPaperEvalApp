# Benchmark Workflow

Use this reference after reading `../SKILL.md`. It keeps benchmark procedures in one place while the skill entrypoint stays concise.

## Inspect First

Review these files before proposing benchmark changes or running scoring:

- `benchmarks/app_evaluation_standard.json`
- `benchmarks/multi_paper_benchmark.json`
- `benchmarks/gold_standards/*.json`
- `docs/app-evaluation-benchmark.md`
- `scripts/run_multi_paper_benchmark.py`
- `scripts/validate_gold_standards.py`
- `scripts/compare_pdf_against_reference.py`
- `backend/tests/test_multi_paper_benchmark.py`

Benchmark, parser, analysis, and documentation files may be dirty during active evaluation. Work around unrelated changes and do not revert them.

## Validation And Execution

Prefer existing repo scripts over hand-written command logic:

```bash
.venv/bin/python .codex/skills/papereval-benchmark/scripts/papereval_benchmark_qa.py --validate
python3 scripts/run_multi_paper_benchmark.py --tier release --include-unscored --validate-only
```

For deeper local checks:

```bash
.venv/bin/python .codex/skills/papereval-benchmark/scripts/papereval_benchmark_qa.py --validate --run-checks
```

Only run live/scored benchmark execution when explicitly requested or clearly required:

```bash
python3 scripts/run_multi_paper_benchmark.py --tier smoke --execute
```

If local models, GROBID, provider credentials, or network access are unavailable, use validate-only checks and report the missing runtime dependency instead of presenting a benchmark pass.

## Active App Benchmark Runs

Use active app runs when the request is about real-world quality, desktop/webapp behavior, or improvement diagnostics. Active runs upload selected benchmark PDFs through the app API, wait for the app job to finish, fetch the generated report, and compare `summary_json` evidence packets to the case gold-standard JSON with `scripts/compare_evidence_to_gold.py`.

Start with validation:

```bash
.venv/bin/python .codex/skills/papereval-benchmark/scripts/papereval_benchmark_qa.py --validate
```

Choose the active surface:

- `--surface desktop`: use an already-open PaperEval Desktop backend, usually `--api-base http://127.0.0.1:8000/api`. Add `--launch-desktop` only when you need the helper to open `PaperEval.app`. Desktop bootstrap failure is fatal unless `--allow-surface-not-ready` is explicitly passed.
- `--surface web`: let the helper start the backend with the built UI served at `http://127.0.0.1:8000/web/`; use `--stop-app` for isolated benchmark runs. Web mode checks the backend-served web URL before uploads unless `--allow-surface-not-ready` is explicitly passed. Use Vite on `http://127.0.0.1:5184/` only for frontend hot-reload/debugging work, not benchmark pass/fail runs.
- `--surface api`: exercise the backend API only. This is useful for CI-like checks but does not prove desktop or web UI launch behavior.

Runtime preflight is required before upload. The helper checks the backend job runner, worker capacity, runner error state, GROBID readiness, and provider/model readiness. Use `--allow-runtime-not-ready` only for intentionally degraded diagnostic runs.

Provider policy:

- Normal active benchmark runs use `--llm-provider local`, matching the app's default local-LLM configuration.
- Gold standards are Codex-assisted reference targets; they are the comparison target, not the runtime provider under test.
- Use `--llm-provider openai` only for explicit non-local provider-comparison diagnostics.

Single-paper diagnostic run:

```bash
.venv/bin/python .codex/skills/papereval-benchmark/scripts/papereval_benchmark_qa.py --active-run --surface desktop --api-base http://127.0.0.1:8000/api --mode single --random-seed 1 --fail-on-incompatible
```

Add `--require-diagnostic-only` to any active-run command when benchmarking should refuse to start unless the current Git status is classified as benchmark-only. This runs the same output-change audit before app startup, upload, or queue submission. Refusal messages include output-risk/unknown category summaries so the blocker is visible without starting the backend.
Active-run summaries always include `output_change_audit`, even when the strict gate is not used, so later comparisons can distinguish diagnostic-only harness runs from runs made against a broader dirty app tree. `--summarize-run`, `--summarize-history`, and `--record-history` preserve compact audit provenance fields such as diagnostic-only status and output-risk/unknown counts.

Suite run against an isolated web/API process:

```bash
.venv/bin/python .codex/skills/papereval-benchmark/scripts/papereval_benchmark_qa.py --active-run --surface web --start-app --stop-app --llm-provider local --tier smoke --max-concurrent 1 --queue-timeout 600 --fail-on-incompatible
```

Release/deep diagnostic sweep:

```bash
.venv/bin/python .codex/skills/papereval-benchmark/scripts/papereval_benchmark_qa.py --active-run --surface api --start-app --stop-app --llm-provider local --tier release --include-unscored --max-concurrent 1 --queue-timeout 600
```

Queue and concurrency policy:

- Default to `--max-concurrent 1`. The backend default is `analysis_workers = 1`, so this avoids heating the machine and keeps failures attributable to one paper.
- Increase `--max-concurrent` only when stress-testing the app queue or when `/api/status` reports enough `processing.worker_capacity`.
- If `--max-concurrent` exceeds backend capacity, the helper submits only that many jobs at once and the app's job runner queues the excess internally.
- Queue timeout and run timeout are separate. `--queue-timeout` limits how long an app job may remain queued before the helper stops submitting more work. `--timeout-per-case` starts when the app job first reports `running`.
- After a queue or run timeout, stop submitting new queued papers and mark remaining helper-queued papers as skipped.

Single-paper mode:

- `--mode single` and `--single` select one random case from the selected pool.
- Without `--tier`, single mode samples from all gold-standard manifest cases and includes `diagnostic_coverage` cases so random checks can cover the broader corpus.
- Use `--random-seed` for repeatable selection.
- A sampled `reference_comparison` case is a release gate. A sampled `diagnostic_coverage` case is a diagnostic finding source until reviewed reference markdown exists.
- `--fail-on-incompatible` returns non-zero for release-gate failures. Add `--fail-on-diagnostic` only when diagnostic-coverage failures should also fail the command.

Artifacts are written under `test/active_benchmark/<timestamp>/` unless `--out-dir` is passed. Inspect:

- `active_benchmark_summary.json` for selected cases, active surface, preflight status, backend worker capacity, pass/fail totals, and per-case records.
- `benchmark_definition` inside `active_benchmark_summary.json` for the SHA-256 fingerprint of the benchmark manifest, app evaluation standard, and gold-standard JSON fixtures used for that run.
- `<case_id>/record.json` for job status, report validity, gold compatibility, model usage, and fallback/run-validity diagnostics.
- `<case_id>/report.json` for the generated report payload when the app job completes.
- `<case_id>/report.html` for a static, reader-first detailed-analysis artifact that can be opened from Slack/history even when the PaperEval backend is closed. When comparison metadata is available, the report starts with a benchmark score card.
- `<case_id>/media.json` and `<case_id>/media/` when local media metadata/assets were available; `report.html` embeds local image previews and table previews from these files where possible.
- `<case_id>/source_chunks.json` when the local chunk DB is available; this stores bounded source chunk excerpts and hashes for diagnostics so future stage-loss review does not depend entirely on the current `data/app.db`.
- `<case_id>/intermediate_stage_index.json` when the app's document artifact is available; this preserves the ordered stage map, transition losses, and LLM-input readiness flags beside the benchmark report.
- `<case_id>/slack_summary.md` for Slack-ready text: bold executive-summary headings, blank lines between sections, benchmark score when available, a static detailed-analysis link, and a live webapp detailed-analysis link when the backend URL is known.

Completed app runs also write `data/doc_<id>/artifacts/intermediate_stage_index.json`. Treat this as the canonical organized view of intermediate data products: source manifest, parsed chunks, modality packets, audited evidence packets, synthesis inputs, retention audit, final report, and runtime diagnostics. Its `llm_input_readiness` block identifies whether the LLM-facing data is missing source excerpts, detail typing, or section assignment before changing prompts or rerunning papers.
Active benchmark runs and static-report backfills copy that index into the case artifact folder when it is locally available, so history review can inspect stage order and transition losses without relying on the live app artifact directory.
For reports with `artifact_organization.llm_input_inventory`, inspect selected prompt-detail hashes, refs, sections, modalities, detail types, and quality flags to confirm what the synthesis LLM could see. Missing benchmark content that is absent from this inventory should be diagnosed upstream in packet shaping or synthesis selection before prompt wording is changed.

When a benchmark run directory is served by a static local server, pass `--artifact-url-base <base-url>` on active runs so `record.json` can store `<base-url>/<case_id>/report.html` as `artifacts.detailed_analysis_url`. Every completed active case also stores `artifacts.webapp_detailed_analysis_url`, shaped like:

```text
http://127.0.0.1:<backend-port>/web/?job_id=<job_id>&document_id=<document_id>&view=detailed_analysis
```

The webapp URL is a convenience link for an available backend; it is not durable after the backend closes. The static `report.html` is the durable history/Slack fallback.
Media capture is best-effort and must not determine benchmark pass/fail. If media capture fails, use the text/table/figure summaries and diagnostics rather than rerunning the paper solely for the static artifact.

Before any iterative rerun, summarize the existing run without starting the app:

```bash
.venv/bin/python .codex/skills/papereval-benchmark/scripts/papereval_benchmark_qa.py --output-change-audit
.venv/bin/python .codex/skills/papereval-benchmark/scripts/papereval_benchmark_qa.py --output-change-audit --json
.venv/bin/python .codex/skills/papereval-benchmark/scripts/papereval_benchmark_qa.py --summarize-history
.venv/bin/python .codex/skills/papereval-benchmark/scripts/papereval_benchmark_qa.py --summarize-history --json
.venv/bin/python .codex/skills/papereval-benchmark/scripts/papereval_benchmark_qa.py --summarize-stage-diagnostics
.venv/bin/python .codex/skills/papereval-benchmark/scripts/papereval_benchmark_qa.py --summarize-stage-diagnostics --json
.venv/bin/python .codex/skills/papereval-benchmark/scripts/papereval_benchmark_qa.py --summarize-history --record-history
.venv/bin/python .codex/skills/papereval-benchmark/scripts/papereval_benchmark_qa.py --plan-next
.venv/bin/python .codex/skills/papereval-benchmark/scripts/papereval_benchmark_qa.py --plan-next --prefer-needs-fix
.venv/bin/python .codex/skills/papereval-benchmark/scripts/papereval_benchmark_qa.py --plan-next --prefer-lowest-score
.venv/bin/python .codex/skills/papereval-benchmark/scripts/papereval_benchmark_qa.py --plan-next --prefer-current-definition-refresh
.venv/bin/python .codex/skills/papereval-benchmark/scripts/papereval_benchmark_qa.py --plan-next --prefer-needs-fix --json
.venv/bin/python .codex/skills/papereval-benchmark/scripts/papereval_benchmark_qa.py --summarize-run test/active_benchmark/<timestamp>
.venv/bin/python .codex/skills/papereval-benchmark/scripts/papereval_benchmark_qa.py --summarize-run test/active_benchmark/<timestamp> --json
```

To backfill reader artifacts from an existing `report.json` without rerunning analysis:

```bash
.venv/bin/python .codex/skills/papereval-benchmark/scripts/papereval_benchmark_qa.py --write-detailed-report test/active_benchmark/<timestamp>/<case_id> --detailed-analysis-url http://127.0.0.1:<port>/report.html --webapp-url 'http://127.0.0.1:8000/web/?job_id=<job_id>&document_id=<document_id>&view=detailed_analysis'
```

When sibling `record.json` or parent `active_benchmark_summary.json` files are present, this backfill also updates their compact `artifacts` pointers and recomputes comparison metadata from the existing `report.json`. That keeps `--summarize-history`, `--record-history`, Slack handoff text, and the 0-1 benchmark score aligned with regenerated static report links without requiring another paper run.

To enrich an older static report with media from a currently available backend:

```bash
.venv/bin/python .codex/skills/papereval-benchmark/scripts/papereval_benchmark_qa.py --write-detailed-report test/active_benchmark/<timestamp>/<case_id> --fetch-media-assets --api-base http://127.0.0.1:8000/api --detailed-analysis-url http://127.0.0.1:<port>/report.html --webapp-url 'http://127.0.0.1:8000/web/?job_id=<job_id>&document_id=<document_id>&view=detailed_analysis'
```

If a `media.json` file is already saved, `--write-detailed-report` uses it automatically. You can also pass `--media-json <path>` explicitly. These modes only regenerate static artifacts and metadata pointers; they do not rerun analysis or change synthesized report content.

Use `--output-change-audit` when a benchmarking session must prove it stayed diagnostic-only. The audit reads the current Git status and classifies dirty files as `benchmark_only`, `output_risk`, or `unknown`; add `--fail-on-output-risk` for strict gates. Benchmark fixtures, repo-local `papereval-*` QA skills, benchmark helpers, docs, repo metadata, and offline diagnostic scripts are non-output. Runtime app/service paths, launcher scripts, desktop shell/UI files, packaging files, dependency/env examples, and app icon/image assets are output-risk for active benchmarking because they can change generated reports, runtime behavior, or the surface under test. The audit also reports category counts, so reviewers can see whether a strict-run blocker is mainly backend app code, desktop UI, launcher/packaging, dependencies/env, assets, scripts, tests, or benchmark fixtures without reading every path. Treat `unknown` as not proven safe until reviewed.

Use `--summarize-history` to find the latest result, score trends, recurring bottlenecks, lowest-score cases, and available detailed-analysis/Slack-summary/media artifact pointers across prior iterations, then use `--summarize-run` for the specific artifact. Add `--json` when another Codex session or automation needs the same latest-by-case, focus-count, `benchmark_score_trends`, and advisory `score_priority_cases` state in machine-readable form. Add `--record-history` when you want a durable local SQLite tracker at `test/active_benchmark/benchmark_history.sqlite`; it stores compact redacted fields only: case id, benchmark-definition digest, decision, compatibility metrics, slowest stage, focus tags, failures, and artifact paths/URLs. It must not store raw report JSON, summary JSON payloads, media payloads, evidence packets, prompts, model text, source excerpts, or full benchmark fixture contents.
Use `--summarize-stage-diagnostics` when you need a corpus-level diagnosis of where benchmark-required content disappears. It scans the latest saved report per case, compares each report to its gold standard, and aggregates missing-item failure points across extraction/source recall, evidence packetization, synthesis selection, final synthesis, and comparator matching. It also reports aggregate `improvement_lane` and `source_visibility` counts, a recommended inspection queue, weighted candidate package/function areas, and representative examples without rerunning the app. This is read-only. Component candidates are attribution hypotheses from saved artifacts, not proof that a package is defective: `candidate_causal` means the stage trace makes that code family plausible, `needs_manual_confirmation` means inspect examples first, and `likely_not_app_output_causal` means compare/matching or claim context should be checked before app-output code. Component weights rank inspection targets by the number of missing benchmark items behind their lane; they are not unique defect counts. For source-recall misses, inspect `source_recall_drilldown.by_db_visibility` and `source_recall_drilldown.by_pdf_visibility`: `db_exact_term_present` points to downstream evidence selection/report-artifact visibility, `pdf_exact_term_present` with `db_not_found` points to parser/chunk persistence, `db_label_term_not_checked` or `pdf_label_term_not_checked` points to gold labels that need alias/context review rather than literal extraction, and `db_document_missing` means the old run cannot be checked against the current local DB. For DB-present misses, `by_chunk_payload_visibility` separates saved diagnostic artifacts that expose chunk payload text from reports where the term is only visible by querying the local DB; newer active runs use `source_chunks.json` for this artifact-level check. `source_recall_drilldown.by_loss_mode` converts paired DB/PDF statuses into next actions such as `parsed_chunk_present_lost_before_report_artifact`, `pdf_text_present_but_chunk_missing`, `pdf_text_absent_or_alias_needed`, `gold_label_or_alias_context_review`, `historical_db_unavailable_pdf_has_term`, `historical_db_unavailable_pdf_not_found`, or `db_or_pdf_not_checked`; `examples_by_loss_mode` gives compact representative terms and traces for each action bucket. Source-recall examples in JSON include compact `db_trace` and `pdf_trace` snippets when a parsed chunk or PDF text already contains the term. For a single case, `compare_evidence_to_gold.py --summary --stage-diagnostics` prints per-item why text, matched artifact paths/snippets, nearest candidate text, source-visibility classification, and an improvement lane. Use those fields to decide what to inspect next; do not infer that extraction failed just because the strict score missed a slot. Single-paper diagnostics may justify generalizable improvements to source visibility, normalization, packet shaping, selection, or synthesis contracts, but should not become paper-specific deterministic expected-fact rules.

`--summarize-stage-diagnostics` also reports `artifact_organization_summary`. For reports created before native stage indexes existed, it derives packet-quality totals from saved `report.json`; for newer app runs, prefer native `intermediate_stage_index.json` plus report-native `artifact_organization`.

Use `--plan-next` to choose the next manifest case from existing history without launching anything. It defaults to the full 11-paper gold-standard corpus, recommends unrun cases before failed cases, and then passed cases. Add `--prefer-needs-fix` when the next session should keep iterating on an existing failed or diagnostic-failed case before moving to unrun papers. Add `--prefer-lowest-score` when the next session should target the scored failed case furthest from the gold standard; it ranks scored failed cases by lowest `overall_benchmark_score`, then by missing slots and claim-gap count. Add `--prefer-current-definition-refresh` when the next session should rerun scored cases missing or mismatched against the current benchmark-definition fingerprint before expanding to unrun cases. Add `--json` when another Codex session or automation needs a machine-readable handoff with the next case, full `planned_cases` list, compact `state_counts`, 10-row `queue_preview`, advisory `score_priority_cases`, current `output_change_audit`, score/gap diagnostics, a normal suggested command, and `suggested_diagnostic_only_command`. Narrow it with `--tier release` or `--case <id>` when a later session needs a scoped plan. The printed commands are only handoffs for a future active run; do not execute them unless the user asks to resume live benchmarking. Use the diagnostic-only command when the run must prove the tree has not touched deterministic app-output paths.

Use the summary's `next_focus` values to patch or configure the current slow/failed section first. Do not use repeated full-paper reruns as the diagnostic tool when the current artifact already identifies a stage-level bottleneck or failure. Benchmark harness, comparator, timeout, and diagnostics changes are acceptable for this workflow; prompt/synthesis/output changes require a separate app-quality rationale because they can deterministically change generated reports.

Gold comparisons include `overall_benchmark_score` and `benchmark_content_score`, both on a 0-1 scale. The score is content coverage, not a fluency grade: it counts matched benchmark slots divided by expected benchmark slots. Slots are critical-claim candidates, expected entities, expected numbers, expected detail types, and required sections. Extra report content does not increase the denominator and is not penalized; forbidden claims and schema gaps remain separate compatibility diagnostics. Static report and Slack artifacts include the score component counts and a compact missing-slot summary when the comparison basis is available.
For one-off local checks against an existing `report.json` or evidence-packet file, add `--summary` to `scripts/compare_evidence_to_gold.py` to print the 0-1 score, matched/expected slots, score components, and top benchmark gaps without rerunning the app. Add `--stage-diagnostics` when the diagnostic needs to identify where missing gold content disappeared across saved extraction, evidence packets, synthesis inputs, and final report text.

For multi-paper runs and history summaries, use `benchmark_score_summary.weighted_overall_benchmark_score` or `latest_benchmark_score_summary.weighted_overall_benchmark_score` as the suite-level score. It sums matched slots across scored papers and divides by expected slots, so a paper with more benchmark obligations contributes proportionally. Use `mean_overall_benchmark_score` as a secondary quick read of average paper-level performance.
Use `latest_benchmark_definition_summary` to see how many latest case scores match, mismatch, or lack the current benchmark-definition fingerprint. Use `current_definition_benchmark_score_summary` when reporting scores that are directly comparable to the current gold-standard/manifest target; older runs without fingerprints remain visible in the regular latest summary but should be treated as provisional for current-target comparisons.
Use `current_definition_refresh_cases` from `--summarize-history --json` or `--plan-next --json` to choose scored cases that should be rerun against the current benchmark definition before their scores are treated as current-target evidence. Plain `--plan-next` output also prints a current-definition refresh preview for the highest-priority refresh candidates, including the previous score, missing-slot count, gap count, and decision.
Use `benchmark_score_trends` to check whether each paper's latest scored run improved, regressed, or stayed unchanged compared with its previous scored run. Score deltas are computed from `overall_benchmark_score`; missing-slot deltas are negative when the latest run covered more required benchmark content. `scored_trend_cases` counts every case with two scored runs, while `comparable_cases` and `mean_score_delta` count only rows where `benchmark_definition_match` is true. `all_score_delta_mean` is available as a looser historical signal, but it may mix old or changed benchmark targets. Treat trends as directly comparable only when `benchmark_definition_match` is true; false means the benchmark target changed, and null means one or both older summaries predate fingerprint recording.

Important diagnostic fields include `preflight_ok`, `preflight_failure`, `decision`, `release_gate`, `comparison.compatible`, `comparison.overall_benchmark_score`, `comparison.benchmark_content_score`, `comparison.usable_packet_rate`, `comparison.critical_claim_candidate_rate`, `diagnostics.report_invalid_reason`, `diagnostics.fallback_audit`, `diagnostics.model_usage`, and `diagnostics.analysis_diagnostics_keys`.
When `text_analysis_cache_validity_conflict` appears in iteration diagnostics or `resolve_text_cache_validity_conflict` appears in `next_focus`, the app report marked local text analysis as missing even though benchmark latency evidence shows a text-stage cache hit. Treat that as a validity-diagnostic issue, not a generated-content change. The API validity refresh should classify cached repeats as `text_analysis_cache_reused`. Helper-managed local `--start-app` active benchmark runs disable document/global local text-analysis caches by default so benchmark scores measure fresh local text LLM behavior; use `--allow-local-text-cache` only for explicit cache-reuse diagnostics.

## Score Gap Triage

Use this order before changing synthesis prompts:

1. Confirm run validity. If a case says `Local model text analysis did not run` while also showing `text_cache_hit` and `model_calls_zero`, treat that run as cache-contaminated for local model-quality measurement. Rerun the case with a helper-managed local app start, or verify `disable_local_text_cache: true` in the run summary.
2. Compare the existing report to gold with `scripts/compare_evidence_to_gold.py --summary --stage-diagnostics --summary-gaps 10`. Use the component counts to distinguish missing claim candidates, missing entities, missing numbers, and missing sections. For each missing item, read the `why`, `trace`, `nearest`, and improvement-lane fields before deciding which app stage needs work.
3. Run `--summarize-stage-diagnostics` to see whether the same failure point dominates across latest cases.
4. Search the saved `report.json` for the missing gold entities/numbers. If the terms are absent from the report, improve upstream extraction, table/figure/supplement capture, section recall, or evidence preselection before editing synthesis instructions.
5. If the missing terms are present in `report.json` but absent from evidence packets or have the wrong section/detail type, improve evidence packet shaping: section assignment, scientific-detail typing, number/entity preservation, and focus-slot selection.
6. If the required facts are present in well-typed evidence packets but the final narrative omits them, then improve synthesis instructions. The prompt should require explicit coverage of selected scientific details, expected numbers, sensitivity analyses, statistical tests, uniqueness, secondary findings, and supplement availability rather than asking for a generic concise summary.
7. Re-score the same saved report after diagnostic-only comparator changes. Only rerun the app after a concrete parser/evidence/synthesis fix is made, and prefer one paper at a time.

## Parallel Evidence Planning

When adding all-data LLM reasoning, prefer a parallel global evidence-planning lane rather than a single large final synthesis prompt. The planner should run alongside section synthesis and consume compact evidence inventories: section coverage, selected evidence packets, table/figure/supplement summaries, source availability notes, and stage diagnostics. Its output should be structured as coverage priorities, cross-modal links, missing-evidence requests, and verifier instructions. Keep final section synthesis partitioned by section/modality until the planner has been validated; then let a merge/verifier step use the global plan to check coverage without inventing unsupported content.

## Evaluation Scope

Check whether selected benchmark cases cover:

- Source ingestion and `source_manifest`.
- Parser extraction and parser diagnostics.
- Text, table, figure, supplement, reconcile, and synthesis.
- Run timeline, analysis diagnostics, provider/cost diagnostics, and desktop operator review.
- Paper components: abstract/title, Introduction, Methods, Results, Discussion, Conclusion, tables, figures, supplements, and references/metadata.

## Reference-Scored Cases

Use `scripts/compare_pdf_against_reference.py` through the multi-paper runner for scored cases. The current scored fixture is:

- `sharma_2017_reward_deficits`
- PDF: `test/sharma-et-al-2017-common-dimensional-reward-deficits-across-mood-and-psychotic-disorders-a-connectome-wide-association.pdf`
- Reference: `test/text/sharma_2017_chatgpt_extraction.md`

`diagnostic_coverage` cases are coverage/backlog, not release gates, until reviewed reference markdown exists. They still need exactly one final-report gold-standard JSON fixture so reviewers can compare app synthesis against a case-specific target.

## Codex-Assisted References

Codex can help draft benchmark gold-standard JSON and reference markdown for public or user-provided papers, but do not promote a reference until it is reviewed.

Gold-standard JSON rules:

- Maintain one fixture in `benchmarks/gold_standards/` for each manifest case.
- Use `authoring.review_status=codex_drafted_needs_review` until human review or reviewed reference material promotes it.
- Include `final_report_expectations.secondary_findings`, `sensitivity_analysis`, `statistical_tests_used`, and `uniqueness` for every paper.
- Include `final_report_expectations.supplement_availability` for every paper.
- Explicitly distinguish unavailable supplements, partially reviewed supplements, and supplement-only fixtures.
- Keep critical claims concise and source-grounded.

Reference markdown rules:

- Use section headers such as `# INTRODUCTION`, `# METHODS`, `# RESULTS`, `# DISCUSSION`, and `# CONCLUSION` when present.
- Write concise source-grounded bullets, not long copied passages.
- Include sample, design, measures, outcomes, effect directions, key limitations, and scientific implications.
- Represent tables, figures, and supplements when they contain important evidence.
- Avoid unsupported claims and avoid copying proprietary or copyrighted text wholesale.
- After review, update the case from `diagnostic_coverage` to `reference_comparison` and set `reference_status` to `available`.
