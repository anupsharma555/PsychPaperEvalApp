---
name: papereval-single-paper-review
description: Repo-specific single-paper review workflow for PsychPaperEvalApp. Use when reviewing one paper's extracted report, comparing app output to source evidence, checking section/table/figure/supplement coverage, or preparing concise quality findings for a specific paper run.
---

# PaperEval Single Paper Review

## Core Workflow

Use this skill for one paper's output quality. Use `papereval-run-qa` for runtime artifacts and timing, `papereval-benchmark` for corpus scoring, and `papereval-ui-qa` for interface behavior.

1. Identify the paper, run, report, and available source material before judging quality.
2. Read local artifacts first; never assume `job_id == document_id`.
3. Compare claims against the paper, tables, figures, supplements, and app diagnostics.
4. Separate source limitations from app omissions, hallucinations, synthesis errors, and presentation issues.
5. Report only evidence-backed findings, with enough location detail for a reviewer to reproduce.

For the review rubric, severity guidance, and reporting format, read `references/single-paper-rubric.md`.

## Helpers

Use existing repo helpers rather than embedding long command logic:

```bash
.venv/bin/python .codex/skills/papereval-run-qa/scripts/papereval_run_qa.py --latest
.venv/bin/python .codex/skills/papereval-run-qa/scripts/papereval_run_qa.py --job-id 157
.venv/bin/python .codex/skills/papereval-run-qa/scripts/papereval_run_qa.py --document-id 157
```

When the review belongs in the benchmark harness, switch to `papereval-benchmark` and validate through the benchmark runner.
