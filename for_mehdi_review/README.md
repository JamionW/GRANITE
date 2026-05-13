# GRANITE results for review

Curated subset of M0, M2, and M3 result artifacts. Mirrors the file layout
under `data/results/` and `output/` in the main repo. Refer to
`Research_Status.md` at repo root for the experimental framing and headline
numbers; this directory is the underlying evidence.

## Contents

### m0_n20_svi_parity/

Traditional-method parity check on the n20 stratified subset against
Dasymetric and Pycnophylactic at block-group resolution.

- `aggregate.csv` -- pooled BG r and per-tract median BG r per method, with
  bootstrap CIs.
- `per_tract.csv` -- per-tract per-method BG r values (the input to the
  aggregate).
- `pairwise_diffs.csv` -- bootstrap pairwise separability test on per-tract
  medians.
- `RESULTS.md` -- narrative summary.

Maps to: `Research_Status.md` M0 entry; locked-in findings 6.

### m2_n20_recovery/

Held-out engineered feature recovery harness. Three targets x two
architectures on n20.

- `summary/summary_stats.csv` -- median per-tract Pearson r, RMSE, and
  constraint error per (target, architecture). This is the headline table.
- `summary/pivot_pearson_r.csv`, `summary/pivot_rmse.csv` -- same data in
  pivot form.
- `{target}_{architecture}/per_tract_metrics.csv` -- per-tract metrics for
  each of the six experiment cells.
- `{target}_{architecture}/run_meta.json` -- configuration and run metadata.
- `employment_walk_effective_access_sage/predictions.csv` -- per-address
  predictions for the M2 tug-of-war exhibit (the only cell where address-
  level predictions are included).

Maps to: `Research_Status.md` M2 entry; locked-in finding 4
(constraint-vs-feature-signal tug-of-war).

### m3_n20_baselines/

Non-graph ridge and GBM baselines on the same M2 targets. No graph, no
constraint.

- `summary/baseline_summary_stats.csv` -- median ridge and GBM r per target.
- `summary/lift_table.csv`, `summary/lift_summary.csv` -- GRANITE-vs-baseline
  lift comparison.
- `summary/per_tract_metrics.csv` -- per-tract values across all three
  targets.
- `{target}/per_tract_metrics.csv` -- per-target per-tract metrics.
- `{target}/run_meta.json` -- configuration and run metadata.

Maps to: `Research_Status.md` M3 entry; locked-in finding 3 (within-tract
feature redundancy, established by M3.5 from this data).

## What is not here

- M2 per-address `predictions.csv` for cells other than the tug-of-war
  exhibit. The summary and per-tract metrics fully characterize those cells.
- M0 per-address predictions. The block-group aggregation is the unit of
  validation; per-address outputs do not add information at this scale.
- Plots. Regenerable from the CSVs.
- Cache files and intermediate artifacts.

## Reproducibility

Active branch: `main`. HEAD at time of bundle creation: see git log.
Entry points:
- M0: `granite/scripts/run_m0_parity.py`
- M2: `granite/disaggregation/recovery_harness.py:run_recovery`
- M3: `granite/evaluation/recovery_baselines.py:run_baselines`
