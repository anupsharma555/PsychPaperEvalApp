# Single Paper Rubric

Use this rubric after reading `../SKILL.md`. It is for judging one PaperEval report against the paper and local run artifacts.

## Evidence To Inspect

- User-provided paper, DOI, URL, uploaded PDF, or local test PDF.
- `data/doc_<id>/artifacts/source_manifest.json`
- `data/doc_<id>/artifacts/parse_diagnostics.json`
- `data/doc_<id>/artifacts/parser_asset_diagnostics.json`
- `data/doc_<id>/artifacts/analysis_diagnostics.json`
- `data/doc_<id>/artifacts/run_timeline.json`
- Report JSON/API payloads or rendered report text supplied by the user.
- Reference markdown or gold-standard JSON when the paper is already in the benchmark.

Use `papereval-run-qa` helpers for artifact discovery and timing summaries. Do not run live provider calls unless the user explicitly asks and the environment is ready.

## Quality Dimensions

Assess the report on these dimensions:

- Source grounding: major claims are supported by the paper or clearly marked as unavailable.
- Coverage: abstract/title, Introduction, Methods, Results, Discussion, Conclusion, tables, figures, supplements, and references/metadata are represented when available.
- Method accuracy: sample, design, cohorts, measures, inclusion/exclusion criteria, statistical methods, and model details are correct.
- Results accuracy: effect directions, outcomes, comparisons, estimates, confidence intervals, p-values, and null findings are not inverted or overstated.
- Table/figure handling: important table and figure evidence is captured without treating labels, captions, or visual marks as unsupported conclusions.
- Supplement handling: supplement availability and use are explicit; missing supplements are not silently treated as negative evidence.
- Synthesis quality: limitations, implications, and cross-section conclusions follow from extracted evidence.
- Diagnostics: parser failures, fallback use, model errors, missing modalities, cost/call caps, and invalid-report flags are surfaced when they affect trust.

## Severity

- Critical: fabricated or inverted core finding; wrong population/intervention/exposure; unsafe clinical implication; report appears successful while major source coverage failed.
- High: missing or materially wrong primary outcome, sample/design, main table/figure, supplement-dependent claim, or limitation that changes interpretation.
- Medium: incomplete secondary result, unclear provenance, weak diagnostic visibility, confusing section organization, or non-blocking extraction omission.
- Low: wording, formatting, redundant phrasing, minor terminology mismatch, or polish issue that does not change scientific meaning.

## Reporting Format

Lead with the most important issues. For each finding include:

- Severity and concise title.
- The report claim or omission.
- The paper/artifact evidence used to check it.
- Why it matters for review quality.
- A practical fix or next validation step.

Keep conclusions bounded. If source material, artifacts, or supplements are missing, state the validation limit instead of guessing.
