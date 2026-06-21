# PaperEval App Evaluation Benchmark

## Purpose

This benchmark evaluates whether PsychPaperEvalApp produces useful, trustworthy psychiatric paper analyses. Unit tests still protect code behavior. This benchmark is the product-quality standard for judging full app runs, model/provider changes, parser changes, desktop releases, and prompt or synthesis changes.

The benchmark should answer:

- Did the app preserve the important scientific content?
- Did it keep evidence attached to the right section and source?
- Did it handle tables, figures, supplements, and imperfect source material honestly?
- Did the operator get enough diagnostics to trust or reject the report?
- Did the run stay within practical time, cost, and reliability limits?

Each manifest case has one final-report gold-standard JSON fixture under `benchmarks/gold_standards/`. These fixtures define the expected paper identity, report content, secondary findings, sensitivity analyses, statistical tests used, uniqueness, critical claims, guardrails, optional critical-claim scientific detail types, and supplement availability disclosure. Codex-drafted fixtures are useful comparison targets, but only `reviewed_gold_standard` or `reviewed_reference_available` fixtures should be treated as release-grade gold without further review.

Non-goals:

- Replacing backend unit tests, parser tests, API tests, or desktop smoke tests.
- Treating diagnostic-only cases as release gates before reviewed reference markdown exists.
- Running live model/provider/API work without explicit benchmark execution intent.
- Committing private PDFs, secrets, runtime databases, OpenAI ledgers, or full proprietary paper text.

## Benchmark Tiers

### Smoke Tier

Use before merging a focused pipeline, prompt, parser, or desktop lifecycle change.

Required case:

- `test/sharma-et-al-2017-common-dimensional-reward-deficits-across-mood-and-psychotic-disorders-a-connectome-wide-association.pdf`
- `test/text/sharma_2017_chatgpt_extraction.md`

Recommended command:

```bash
.venv/bin/python scripts/compare_pdf_against_reference.py \
  --mode pipeline \
  --parser-engine validated \
  --backend-profile section-sensitive \
  --pdf test/sharma-et-al-2017-common-dimensional-reward-deficits-across-mood-and-psychotic-disorders-a-connectome-wide-association.pdf \
  --reference-md test/text/sharma_2017_chatgpt_extraction.md \
  --gold-standard-json benchmarks/gold_standards/sharma_2017_reward_deficits.json \
  --out-dir test/text \
  --retain-runs 1 \
  --matching-mode hybrid \
  --matching-threshold 0.42 \
  --db-path /tmp/papereval_benchmark_smoke.db
```

If local models or GROBID are unavailable, run the same command with `--mode lightweight` and mark the result as a benchmark limitation, not a release pass.

The same scored paper can be run through the benchmark manifest:

```bash
python3 scripts/run_multi_paper_benchmark.py --tier smoke --execute
```

### Evidence/Gold Compatibility

Reference comparison scores whether the final report resembles the sectioned reference. Evidence/gold compatibility checks a different failure mode: whether the evidence packet or synthesized report sections contain the details needed to support the reviewed final-report gold standard.

Run it directly without model/API calls:

```bash
python3 scripts/compare_evidence_to_gold.py \
  --evidence-json test/text/section_synthesis_v2_comparison_20260503_174925.json \
  --gold-standard benchmarks/gold_standards/sharma_2017_reward_deficits.json \
  --fail-on-incompatible
```

Accepted evidence inputs include direct `evidence_packets`, `summary_json` wrappers from comparison runs, `modalities.*.findings`, `scientific_details`, `sections_extracted`, and synthesized `v1_report`/`v2_report` section bullets with anchors.

The compatibility result is intentionally deterministic. It verifies:

- Usable packet rate: statements have source anchors and known section labels.
- Required section coverage: packet sections cover the gold standard's critical-claim sections.
- Critical-claim candidates: each critical gold claim has a same-section candidate packet.
- Entity, number, and detail-type observability: expected entities, numeric values, and scientific detail types are present in packets.
- Forbidden-claim guardrails: packets do not contain claims listed under `report_should_not_claim`.
- Schema gaps: no missing statements, anchors, known sections, or recognized modalities.

Thresholds are configurable for local iteration:

```bash
python3 scripts/compare_evidence_to_gold.py \
  --evidence-json path/to/run.json \
  --gold-standard benchmarks/gold_standards/sharma_2017_reward_deficits.json \
  --min-usable-packet-rate 0.8 \
  --min-section-coverage-rate 0.8 \
  --min-critical-claim-candidate-rate 0.8 \
  --min-expected-entity-observability-rate 0.5 \
  --min-expected-number-observability-rate 0.8 \
  --min-expected-detail-type-observability-rate 0.8 \
  --forbidden-claim-threshold 0.35 \
  --fail-on-incompatible
```

Use `critical_claims[].expected_detail_types` when the gold standard must verify the kind of evidence, not only lexical similarity. Supported values include `medication_or_therapeutic`, `dose_schedule`, `intervention_or_exposure`, `outcome_measure`, `adverse_event`, `model_system`, `assay_readout`, `statistical_result`, `data_source_or_design`, `tool_or_algorithm`, and `cross_modal_result`.

For scored benchmark runs, `compare_pdf_against_reference.py --gold-standard-json ...` embeds this result in the comparison report, and `run_multi_paper_benchmark.py` passes the case gold standard automatically for reference-scored cases.

### Release Tier

Use before packaging, sharing, or relying on a new app version.

Run at least five cases:

- One psychiatric/neuroimaging paper with known section-level reference coverage.
- One table-heavy clinical or epidemiologic paper.
- One figure-heavy paper where image/caption handling matters.
- One supplement-heavy paper.
- One URL/DOI ingestion case, or a manual upload plus recorded source manifest if external access is unavailable.

The Sharma 2017 case can count as the psychiatric/neuroimaging case. Other cases may be private local fixtures, but their benchmark scorecards should not include private paper text beyond short labels and aggregate metrics.

The current multi-paper manifest is `benchmarks/multi_paper_benchmark.json`. Validate the release corpus without spending model/API time:

```bash
python3 scripts/run_multi_paper_benchmark.py --tier release --include-unscored --validate-only
```

Cases marked `diagnostic_coverage` exercise processing-stage and paper-component coverage and now have final-report gold-standard JSON targets. They still need reviewed reference markdown before they become fully scored release gates.

The manifest also reports known gaps. In the current public fixture set, supplement handling is part of the benchmark standard, one fixture is supplement-only, and several fixtures cite online supplements. The benchmark should verify that synthesis explicitly notes unavailable, partial, or supplement-only evidence. A reference-scored supplement-heavy case is still needed before release/deep runs can claim full supplement scoring coverage.

### Deep Tier

Use for model swaps, parser changes, major prompt rewrites, or release candidates.

Run eight to twelve papers covering:

- Psychiatry, neuroscience, clinical trial, systematic review, and methods-heavy papers.
- Clean born-digital PDFs and degraded/scanned PDFs.
- Papers with meaningful tables, figures, and supplements.
- At least one intentional failure path: bad URL, unavailable supplement, budget cap, or provider/model unavailable.

## Scoring Dimensions

Use `benchmarks/app_evaluation_standard.json` as the machine-readable scorecard. The dimensions below sum to 100 points.

The benchmark should cover these processing stages across the selected case set:

- Source ingestion and source manifest creation.
- Parser extraction and parser diagnostics.
- Text, table, figure, and supplement analysis.
- Reconcile and synthesis.
- Analysis diagnostics, cost/runtime diagnostics, and desktop operator review.

It should cover these scientific paper components:

- Title/abstract, Introduction, Methods, Results, Discussion, and Conclusion.
- Tables, figures, supplements, and reference/metadata handling.

### 1. Source Integrity and Ingestion - 15 points

Pass evidence:

- Main source asset is identified in `source_manifest.json`.
- Parser diagnostics record selected parser, timing, page/chunk counts, and asset count deltas.
- URL/DOI/upload path records source provenance and does not silently switch sources.
- Missing or unreadable supplements are visible as warnings, not hidden success.

### 2. Evidence Retention and Grounding - 20 points

Pass evidence:

- `information_retention` or comparison output shows important reference statements retained.
- Every major report claim has source anchors or evidence references.
- The report avoids unsupported deterministic fallback text after failed LLM/provider calls.
- Missing evidence is labeled as missing or access-limited.

### 3. Section Fidelity and Scientific Structure - 15 points

Pass evidence:

- Introduction, Methods, Results, Discussion, and Conclusion are populated when present in the paper.
- Methods statements do not leak into Results as findings.
- Results include outcome/statistical content where present.
- Discussion and Conclusion emphasize interpretation, limitations, and implications rather than repeating methods.

### 4. Multimodal Coverage - 15 points

Pass evidence:

- Tables are extracted or explicitly diagnosed as unavailable.
- Figures are analyzed from valid figures/captions, not page-raster fallback artifacts unless labeled.
- Supplements are included, skipped, or failed with visible source-level reasons.
- Diagnostics show model calls and fallbacks by text, table, figure, and supplement stage.

### 5. Domain Usefulness - 15 points

Pass evidence:

- Psychiatric relevance is preserved: population, diagnosis/phenotype, measures, outcomes, effect direction, and limitations.
- The report distinguishes study design, sample, intervention/exposure, and endpoints.
- It surfaces limitations and clinical/research implications without overclaiming causality.
- The executive summary is readable and actionable for a psychiatric research reviewer.

### 6. Operational Reliability, Cost, and Runtime - 10 points

Pass evidence:

- `run_timeline.json` and `analysis_diagnostics.json` exist and include stage timings.
- OpenAI/local provider mode is visible.
- Cost and token estimates are present for OpenAI runs and respect configured caps.
- Failures are explicit with `error.log` or diagnostics, and do not produce valid-looking reports.

### 7. Desktop Operator Experience - 10 points

Pass evidence:

- Desktop/backend startup reaches readiness or reports the blocker.
- GROBID/model/provider readiness is visible.
- The UI exposes source coverage, model calls, estimated cost, and report diagnostics.
- Closing the app cleans up app-owned background services where configured.

## Gate Rules

Use the weighted score plus hard gates:

- Pass: score at least 80, no hard gate failures, no critical unsupported claims.
- Conditional: score 70 to 79, or one non-critical hard gate failure with a documented mitigation.
- Fail: score below 70, any critical hard gate failure, invalid-looking report after provider failure, or missing source provenance.

Hard gates:

- No silent successful report when required LLM/provider calls failed.
- No report without source provenance.
- No unsupported or hallucinated primary result in the executive summary.
- No hidden budget/cost overrun for OpenAI runs.
- No destructive or privacy-unsafe benchmark artifact committed to git.

## Evidence Ladder

Use the lowest-cost evidence that proves the criterion:

1. Repo evidence: backend tests, diagnostics JSON, source manifests, comparison output, desktop smoke output.
2. Focused proof: one benchmark run against a reference markdown, one API smoke, or one desktop smoke.
3. Broader proof: release/deep tier corpus, model comparison, browser/desktop screenshot review.
4. Manual judgment: only for domain usefulness, overclaiming, and operator experience.

## Review Notes

Each benchmark review should record:

- Git commit or local diff note.
- Environment: parser engine, backend profile, model/provider, cost caps, GROBID status.
- Cases run and artifact paths.
- Weighted score and hard gate result.
- Three strongest failures or regressions.
- Decision: pass, conditional, or fail.

Do not commit private PDFs, secrets, runtime databases, OpenAI ledgers, or full proprietary paper text. Store local-only run artifacts under ignored runtime or test output locations unless they are intentionally public fixtures.

## Reference Authoring

Reference markdown can be written by a human expert, by Codex with source-PDF review, or by another model. Codex-assisted references are useful for turning additional public PDFs into scored benchmark cases, but they should be promoted carefully:

- Extract or inspect the source paper directly.
- Maintain exactly one final-report gold-standard JSON fixture per manifest case.
- Include `secondary_findings`, `sensitivity_analysis`, `statistical_tests_used`, `supplement_availability`, and `uniqueness` in the final-report expectations.
- Write sectioned bullets under `# INTRODUCTION`, `# METHODS`, `# RESULTS`, `# DISCUSSION`, and `# CONCLUSION` when those sections exist.
- Preserve source-grounded claims, methods, sample details, outcomes, and limitations.
- Avoid copying long passages from copyrighted papers.
- Review for unsupported claims before changing a case from `diagnostic_coverage` to `reference_comparison`.
