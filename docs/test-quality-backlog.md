# Test Quality Backlog

## Purpose

Track concrete improvements to the gold-standard annotation and 10-paper benchmark tests so they explain what PaperEval does well, where it fails, and which app strategies should be tried next.

This backlog complements `docs/app-evaluation-benchmark.md`. The benchmark defines the product-quality standard; this file lists implementation work needed to make that standard measurable.

## Current Findings

### P1 - Make Sharma Gold Annotation Machine-Checkable

The Sharma gold extraction is useful as human review material, but it is not yet a true machine-checkable gold standard.

Current limitations:

- No stable claim IDs.
- No source anchors, page references, or evidence quotes per claim.
- No importance weights.
- No claim taxonomy.
- No explicit numeric/entity fields.
- No omission severity.

Acceptance criteria:

- Create a structured `gold_claims.jsonl` or equivalent fixture for the Sharma paper.
- Each row has `claim_id`, `section`, `claim_type`, `importance`, `evidence_quote`, `page_or_anchor`, `expected_entities`, `expected_numbers`, and `priority`.
- Preserve the markdown file as a reviewer-facing reference, but make tests consume the structured file.
- Add a parser/validator test that fails on duplicate IDs, missing sections, missing evidence, invalid priorities, and malformed expected numeric fields.

### P1 - Add Valid Benchmark Run Gate

The latest stored Sharma comparison is diagnostic, not a valid app-quality benchmark, because it records pipeline fallback and zero text LLM calls.

Acceptance criteria:

- Add a benchmark validity gate before ability scoring.
- Hard-fail benchmark scoring when any of these are true: job not completed, fallback engaged, text LLM calls are zero, section extraction is disabled, diagnostics are missing, or required source provenance is missing.
- Report invalid runs as infrastructure failures rather than model/app ability failures.
- Add tests for valid run, fallback run, zero-text-call run, and missing-diagnostics run.

### P2 - Turn the 10-Paper Set Into Acceptance Tests

The files `test/01_...pdf` through `test/10_...pdf` are currently a corpus, not a gold-backed acceptance suite.

Current coverage is mostly structural:

- Completion status.
- Report presence.
- Section keys.
- Modalities keys.
- Basic text length/repetition checks.

Missing coverage:

- Factual claim recall.
- Unsupported claim rate.
- Section fidelity against source truth.
- Numeric fidelity.
- Table, figure, and supplement recall.
- Whether the report actually understood the paper.

Acceptance criteria:

- Add a lightweight gold layer for all 10 PDFs: expected section availability, expected figure/table refs, parser coverage floors, section-boundary wrong-rate ceilings, and source-manifest expectations.
- Add priority claim annotations for at least 3-5 representative papers.
- Include psychiatry/neuroimaging coverage in the acceptance set; the current 10-paper corpus is broad OA robustness material rather than psychiatry-specific capability coverage.
- Ensure test output separates parser failure, model failure, synthesis failure, and benchmark invalidity.

### P2 - Split Scores By Failure Mode

A single overall score hides the failures we need to fix.

Acceptance criteria:

- Report separate metrics for parser recall, section assignment, claim recall, claim precision, numeric fidelity, unsupported-claim rate, table/figure/supplement recall, and synthesis retention.
- Add per-section metrics for Introduction, Methods, Results, Discussion, and Conclusion.
- Include top missing claims, top unsupported claims, and top cross-section misassignments.
- Preserve aggregate score only as a convenience summary, not the primary diagnostic output.

### P2 - Promote Observed A/B Signals Into Regression Thresholds

Existing artifacts already show actionable behavior:

- The section-boundary ledger variant materially reduces wrong-section rate in the observed A/B sample.
- Table and supplement recall remain weak.
- Visual artifact text can displace useful evidence.

Acceptance criteria:

- Add regression thresholds for section-boundary wrong-section rate on the available benchmark cases.
- Add explicit table/supplement recall metrics and minimum floors.
- Add artifact-text-rate checks for figure/caption processing.
- Add tests that compare candidate strategies against baseline artifacts without requiring live LLM calls.

## Suggested Strategies To Evaluate

### Semantic Matching

Keep lexical overlap as a cheap baseline, but add semantic matching for paraphrase-tolerant claim recall.

Candidates:

- `sentence-transformers` for sentence/claim similarity and retrieve-rerank matching.
- BERTScore for paraphrase-sensitive reference comparison.
- A small cross-encoder reranker for high-confidence claim-to-output matching after cheap retrieval.

Acceptance criteria:

- Benchmark reports both lexical and semantic match scores.
- Semantic matching records matched claim ID, app statement, score components, and match reason.
- Thresholds are calibrated on human-reviewed positive and negative pairs, not guessed from a single run.

### Structured Annotation Workflow

Markdown is too loose for a growing benchmark.

Candidates:

- Label Studio for paper claim/evidence labeling.
- Argilla for reviewable claim, match, and feedback datasets.
- A repo-native JSONL schema if we want no service dependency.

Acceptance criteria:

- Gold annotations can be reviewed and diffed.
- Export format is deterministic.
- Annotation guidelines define atomic claim boundaries, section labels, priority, and evidence quote rules.

### Failure-Mode Fixtures

Add fixtures that intentionally exercise known weak spots.

Candidate cases:

- Section drift after Methods/Results-heavy text.
- Late Discussion and Conclusion under-selection.
- Table references in main text with missing extracted table chunks.
- Supplement references with unavailable supplement files.
- Figure captions contaminated by OCR/page artifact text.
- Provider failure or budget cap that must not produce a valid-looking OpenAI report.

Acceptance criteria:

- Each fixture has a named regression it catches.
- Each test asserts externally visible output, diagnostics, or benchmark status.
- No test depends on live network or live LLM unless explicitly marked as a benchmark/manual run.

## Implementation Order

1. Add the benchmark validity gate and tests.
2. Convert Sharma gold markdown into structured claim JSONL.
3. Add a validator for gold fixtures.
4. Add claim recall, section fidelity, numeric fidelity, and unsupported-claim metrics.
5. Add lightweight metadata/gold expectations for the 10 PDFs.
6. Add semantic matching as optional scoring with deterministic fallback to lexical mode.
7. Add strategy comparison reports for section ledger, media cleaning, table/supplement handling, and synthesis retention.

## Verification Baseline

Last checked locally with:

```bash
PYTHONPATH=backend .venv/bin/python -m pytest backend/tests
```

Observed result:

- `200 passed`
- `97 warnings`
- System Python could not collect the benchmark-focused tests because `sqlmodel` was missing; use the project virtualenv for backend tests.
