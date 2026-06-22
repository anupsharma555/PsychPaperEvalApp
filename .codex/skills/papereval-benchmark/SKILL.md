---
name: papereval-benchmark
description: Repo-specific benchmark and reference-evaluation workflow for PsychPaperEvalApp. Use when running or refining the single-paper or multi-paper benchmark, validating benchmark manifests, comparing app output against reference markdown, creating or reviewing Codex-assisted reference summaries, checking release/deep/smoke benchmark readiness, or evaluating parser/prompt/model/provider changes across the paper corpus.
---

# PaperEval Benchmark

## Core Workflow

Use this skill for corpus-level evidence and release-readiness. Use `papereval-run-qa` for one-off artifact review and `papereval-ui-qa` for UI/UX validation.

1. Inspect the manifest, gold standards, benchmark docs, and runner scripts before changing or running anything.
2. Preserve dirty user work; do not revert benchmark, parser, analysis, or docs changes unless explicitly asked.
3. Start with local validation helpers and validate-only benchmark modes.
4. When the user asks whether the app is improving, whether desktop/webapp behavior is correct, or asks for benchmark diagnostics, run active app benchmarks through the desktop/web/API surface and compare produced reports to gold standards.
5. Default active benchmarks to one paper at a time. Raise `--max-concurrent` only when explicitly useful; the backend defaults to one analysis worker and extra submitted jobs wait in the app queue.
6. Use `--mode single` or `--single` for one-paper random checks across the gold-standard corpus. Single mode is diagnostic by default when it samples `diagnostic_coverage` papers.
7. Treat `diagnostic_coverage` cases as coverage/backlog until reviewed reference markdown exists. `--fail-on-incompatible` is for release-gate failures; add `--fail-on-diagnostic` only when diagnostic coverage failures should fail the command.
8. Active runs must pass runtime and surface preflight checks before upload unless an explicit diagnostic override is used.
9. Normal active benchmark runs must use the local LLM provider. The gold standards are Codex-assisted reference targets, but the app output under test should be local-provider output unless the user explicitly asks for an OpenAI/provider-comparison run.

For the full procedure, scoring boundaries, and reference-authoring rules, read `references/benchmark-workflow.md`.

## Verification

For benchmark-only changes, run:

```bash
.venv/bin/python .codex/skills/papereval-benchmark/scripts/papereval_benchmark_qa.py --validate --run-checks
```

For real-world app benchmarking, use the same helper with `--active-run`. Prefer an existing desktop backend when the user already has the app open:

```bash
.venv/bin/python .codex/skills/papereval-benchmark/scripts/papereval_benchmark_qa.py --active-run --surface desktop --api-base http://127.0.0.1:8000/api --mode single --random-seed 1 --fail-on-incompatible
```

Add `--require-diagnostic-only` when a live benchmark must refuse to start unless the worktree audit is clean enough to prove harness-only changes.
Every active-run summary records `output_change_audit`, so later reviewers can see whether a result was produced from a diagnostic-only or output-risk dirty tree.
`--summarize-run` and `--summarize-history` surface the same audit provenance as compact fields for review and dashboard use.

Before rerunning after a slow or failed paper, summarize the existing run and patch the current bottleneck or failure section first:

```bash
.venv/bin/python .codex/skills/papereval-benchmark/scripts/papereval_benchmark_qa.py --summarize-run test/active_benchmark/<run_dir>
.venv/bin/python .codex/skills/papereval-benchmark/scripts/papereval_benchmark_qa.py --summarize-run test/active_benchmark/<run_dir> --json
```

To review prior iterations without launching the app:

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
```

The JSON planner handoff includes `planned_cases` for every selected paper, `state_counts`, a 10-row `queue_preview`, advisory `score_priority_cases`, score/gap diagnostics, current `output_change_audit`, a normal suggested command, and a `suggested_diagnostic_only_command` that adds `--require-diagnostic-only`.
Use `--prefer-lowest-score` when the next diagnostic iteration should target the scored failed case furthest from the gold standard instead of using manifest order.
Use `--prefer-current-definition-refresh` when the next diagnostic iteration should refresh scored cases that lack or mismatch the current benchmark-definition fingerprint before expanding to unrun cases.
Use `--output-change-audit --fail-on-output-risk` before comparing benchmark results across code changes when the session must prove it did not touch deterministic app output paths.

To regenerate reader artifacts from an existing completed report without rerunning the paper:

```bash
.venv/bin/python .codex/skills/papereval-benchmark/scripts/papereval_benchmark_qa.py --write-detailed-report test/active_benchmark/<run_dir>/<case_id> --detailed-analysis-url http://127.0.0.1:<port>/report.html --webapp-url 'http://127.0.0.1:8000/web/?job_id=<job_id>&document_id=<document_id>&view=detailed_analysis'
```

Use `slack_summary.md` for Slack/history snippets. It starts with bold executive-summary section headings, includes the benchmark score when comparison metadata is available, and then links to the static detailed-analysis artifact and, when available, the live webapp detailed-analysis deep link.
Backfill also refreshes sibling `record.json` and parent `active_benchmark_summary.json` artifact pointers and comparison metadata when those files are present, so later history summaries can find the same static/webapp links and score fields.
Completed active runs also try to save `<case_id>/media.json` plus local `media/` image previews for the static report. For older artifacts, add `--fetch-media-assets` only when the backend for that document is currently available.
Completed active runs also save `<case_id>/source_chunks.json` when the local chunk DB is available. This is a bounded diagnostic source inventory with chunk excerpts and hashes; it is for benchmark traceability and must not be treated as generated report content.
Completed active runs also save `<case_id>/intermediate_stage_index.json` when the local document artifact is available. This preserves stage order, transition loss summaries, and LLM-input readiness flags for later history diagnostics.
Newer reports expose `artifact_organization.llm_input_inventory` so stage diagnostics can aggregate whether selected synthesis inputs were source-backed, typed, and sectioned without storing full duplicate prompt text.
Gold comparisons expose `overall_benchmark_score`/`benchmark_content_score` from 0 to 1. It is the fraction of expected benchmark content slots observed in the report; extra report content is not penalized.
Run and history summaries also expose `benchmark_score_summary.weighted_overall_benchmark_score`, which aggregates matched slots across scored papers, plus a mean paper score for quick trend review.
Active runs record `benchmark_definition`, a SHA-256 fingerprint over the manifest, evaluation standard, and gold standards, so later score changes can be interpreted against the exact benchmark target.
History summaries expose `latest_benchmark_definition_summary` and `current_definition_benchmark_score_summary` so reviewers can separate all latest scores from latest scores produced against the current benchmark target.
History and planner JSON expose `current_definition_refresh_cases` for scored cases whose latest benchmark result is missing or mismatched against the current benchmark-definition fingerprint.
Plain `--plan-next` output also prints a short current-definition refresh preview with score, missing-slot, gap-count, and decision context for the highest-priority refresh candidates.
History summaries expose `benchmark_score_trends` to compare each case's latest scored run against its previous scored run, including score delta, missing-slot delta, and whether the benchmark-definition fingerprints match. `comparable_cases` and `mean_score_delta` only count matching benchmark-definition fingerprints; `scored_trend_cases` and `all_score_delta_mean` include looser historical trends.
`--record-history` stores the compact benchmark-definition digest with each row so longitudinal SQLite tracking keeps score provenance without storing raw reports or evidence.
History summaries expose `score_priority_cases` as a compact lowest-score-first diagnostic view for choosing what to inspect next without changing the default planner order.
`--summarize-stage-diagnostics` aggregates saved report-vs-gold stage-loss diagnostics across latest cases, showing whether missing benchmark content is absent from saved artifacts, dropped before packetization, dropped before synthesis selection, dropped during final synthesis, or present in final output but unmatched. It also aggregates `improvement_lane` and `source_visibility` counts and prints a recommended inspection queue plus weighted candidate package/function areas so future sessions can decide which stage to inspect first and whether app code is actually causal. For source-recall misses, the helper also probes local parsed chunks when a document id is available and the benchmark PDF text when a source PDF is available; `db_exact_term_present` means the parser/chunk DB already contains the term and the likely issue is downstream selection or report-artifact visibility, not PDF extraction. For DB-present misses, `by_chunk_payload_visibility` distinguishes whether saved diagnostic artifacts expose chunk payload text that contains the term or whether chunk-level evidence is only visible in the local DB. Newer active runs use `source_chunks.json` for that artifact-level check. `pdf_exact_term_present` with `db_not_found` means the PDF text contains the term but the saved chunk DB does not, which points to parser/chunk persistence rather than source availability. JSON examples include `db_trace` and `pdf_trace` snippets when visible. The source-recall drilldown also reports `by_loss_mode` and `examples_by_loss_mode`, including `parsed_chunk_present_lost_before_report_artifact`, `pdf_text_present_but_chunk_missing`, `pdf_text_absent_or_alias_needed`, `gold_label_or_alias_context_review`, `historical_db_unavailable_pdf_has_term`, `historical_db_unavailable_pdf_not_found`, and `db_or_pdf_not_checked`. `db_document_missing` means the old run cannot be checked against the current local DB. One-off `compare_evidence_to_gold.py --summary --stage-diagnostics` output includes per-item why text, artifact path/snippet traces, nearest candidates when absent, source-visibility classification, and the suggested improvement lane. Treat these as diagnostic evidence; they should guide whether to inspect extraction, normalization/aliases, packetization, synthesis selection, final synthesis, or benchmark matching without changing app output by themselves. Component weights rank inspection targets; they are not proof a package is defective. Do not convert a single-paper diagnostic into paper-specific deterministic expected-fact rules.
If adding all-data LLM reasoning, start as a parallel global evidence-planning diagnostic lane. It should consume compact evidence inventories, coverage diagnostics, and modality summaries while section synthesis proceeds normally; only after validation should its plan be used by a merge/verifier step to influence final reports.
If a cached repeat run reports `text_analysis_cache_validity_conflict` or `resolve_text_cache_validity_conflict`, diagnose the validity rule separately; do not treat zero text model calls as a text-stage failure when benchmark latency evidence shows a text cache hit. The API should refresh stale stored validity from current diagnostics so cache reuse is reported as `text_analysis_cache_reused`, not `text_llm_calls_zero`.
When a fresh local-text diagnostic is needed, use the planner's `suggested_fresh_text_command`/`suggested_fresh_text_diagnostic_only_command` or add `--disable-local-text-cache` to a helper-managed `--start-app` run.

For an isolated web/API run managed by the helper:

```bash
.venv/bin/python .codex/skills/papereval-benchmark/scripts/papereval_benchmark_qa.py --active-run --surface web --start-app --stop-app --llm-provider local --tier smoke --max-concurrent 1 --queue-timeout 600 --fail-on-incompatible
```

Do not run live model/API benchmarks unless the user asked for execution or the task clearly requires it. Record whether failures are release-gate failures, diagnostic findings, runtime preflight failures, or surface preflight failures.
