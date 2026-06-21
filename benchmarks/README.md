# App Evaluation Benchmark

This directory stores the machine-readable benchmark standard for PsychPaperEvalApp.

- `app_evaluation_standard.json`: score dimensions, case tiers, gates, and verification evidence.
- `multi_paper_benchmark.json`: concrete single-paper and multi-paper case manifest, including processing-stage coverage, paper-component coverage, and known benchmark gaps.
- `gold_standards/*.json`: one Codex-assisted final-report gold-standard target per manifest case, including supplement availability and optional critical-claim detail-type expectations.

The human-readable guide is `docs/app-evaluation-benchmark.md`.
The repo-local Codex workflow skill is `.codex/skills/papereval-benchmark/SKILL.md`.

Use this benchmark alongside tests. Tests prove code behavior; the benchmark judges report usefulness, evidence grounding, section fidelity, multimodal coverage, diagnostics, cost, and operator experience.

Validate and inspect the current release benchmark plan without running model work:

```bash
python3 scripts/run_multi_paper_benchmark.py --tier release --include-unscored --validate-only
```

Execute reference-scored smoke cases:

```bash
python3 scripts/run_multi_paper_benchmark.py --tier smoke --execute
```

Compare an existing evidence packet or report payload to a gold standard without running a model:

```bash
python3 scripts/compare_evidence_to_gold.py \
  --evidence-json path/to/run_or_report.json \
  --gold-standard benchmarks/gold_standards/sharma_2017_reward_deficits.json \
  --summary \
  --stage-diagnostics \
  --fail-on-incompatible
```

This checks section coverage, anchored evidence, critical-claim candidates, expected entities/numbers/detail types, forbidden claims, and packet schema compatibility. `--summary` prints the 0-1 `overall_benchmark_score`, matched/expected content slots, score components, and top benchmark gaps; omit it when raw JSON is needed. `--stage-diagnostics` traces missing gold items through saved extraction, evidence-packet, synthesis-input, and final-report text so one-off diagnostics can distinguish upstream evidence loss from synthesis omissions. For each missing item, stage diagnostics also include the reason for the failure-point classification, matched artifact path/snippet when present, nearest candidate text when absent, source-visibility classification, and a suggested improvement lane. Source visibility separates exact-present losses from near/weak/no term candidates, so normalization and alias issues can be reviewed before parser changes. Corpus summaries add a recommended inspection queue for generalizable pipeline review. The score matcher remains conservative; these traces are for diagnosis, not for changing app output, adding paper-specific deterministic rules, or inflating scores.
