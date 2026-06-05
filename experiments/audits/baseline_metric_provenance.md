# Baseline Metric Provenance Audit

**Date:** 2026-06-05
**Scope:** Map BG-r-like metrics to their source files and validation contexts; resolve the
apparent conflict between 0.469 (legacy framing) and 0.769/0.772 (00_baseline framing).

---

## 1. Enumeration of all BG-r-like metrics

Three distinct metrics appear in the codebase. They are not interchangeable.

### Metric A: Pooled BG r (primary validation metric)

| Attribute | Detail |
|-----------|--------|
| Name in code | `pearson_r` in `block_group_validation.json`; `pooled_bg_r` in `aggregate.csv` |
| Defining file | `experiments/ablation/00_baseline/run_baseline.py`, function `_bg_metrics()`, line ~150 |
| Definition | Pearson r of predicted BG-mean SVI vs nationally-ranked ACS BG SVI, pooled across all 20 tracts simultaneously; each BG requires >= 10 addresses |
| n | 69 block groups (pooled from 20 tracts) |
| Validation ground truth | Nationally-ranked ACS block-group SVI from `data/processed/national_bg_svi.csv` |
| Computation | Address predictions aggregated to BG means; correlated against ACS values globally |
| Artifacts | `experiments/ablation/00_baseline/results/block_group_validation.json` (00_baseline); `data/results/m0_n20_svi_parity/aggregate.csv` (m0 parity) |

### Metric B: Per-tract BG r (within-tract divergence diagnostic)

| Attribute | Detail |
|-----------|--------|
| Name in code | `bg_r` column in per_tract_metrics.csv and per_tract.csv |
| Defining file | `experiments/ablation/00_baseline/run_baseline.py`, `_bg_metrics()` called with MIN_ADDRESSES_PER_BG=3; `granite/scripts/run_m0_parity.py` |
| Definition | Pearson r computed independently for each tract's block groups (min 3 addresses/BG); one r value per tract |
| n | 2-6 BGs per tract; effectively 19-20 r values across the n20 subset |
| Validation ground truth | Same nationally-ranked ACS BG SVI |
| Artifacts | `experiments/ablation/00_baseline/results/per_tract_metrics.csv`; `data/results/m0_n20_svi_parity/per_tract.csv` |
| Caveat | Highly variable (-1.0 to +1.0 range); most tracts have only 2-5 qualifying BGs; unreliable as standalone metric |

### Metric C: Per-tract BG r mean / median (aggregate of Metric B)

| Attribute | Detail |
|-----------|--------|
| Name in code | `bg_r_mean`, `bg_r_median` in `aggregate_metrics.json`; `median_bg_r` in `aggregate.csv` |
| Defining file | `experiments/ablation/00_baseline/run_baseline.py`, lines ~449-450 |
| Definition | Mean or median of the 20 per-tract BG r values (Metric B); only tracts with >= 1 qualifying BG included |
| n | Up to 20 tracts (19 in m0 parity run) |
| Artifacts | `experiments/ablation/00_baseline/results/aggregate_metrics.json`; `data/results/m0_n20_svi_parity/aggregate.csv` |

### Metric D: Global held-out BG r (legacy, superseded)

| Attribute | Detail |
|-----------|--------|
| Name in code | `pearson_r` in `bg_validation_summary.csv` (root) |
| Defining file | Unknown legacy script, not present in current codebase; output at repo root |
| Definition | Pearson r from an early global validation run; SESSION_LOG.md (line 166) labels this "global validation context" |
| n | n_predictions=192 (address or BG count unclear; context predates the structured n20 harness) |
| Note | `scripts/coord_artifact_bg_validation.py` writes to `output/coord_artifact/bg_validation_summary.csv`, a different file with different scope (feature mode comparison, not IDW vs GRANITE) |
| Status | Superseded by Metric A (pooled BG r from n20 stratified harness) |

---

## 2. Number-to-artifact mapping

| Number | Method | Metric type | Source file | Line | n | Validation context |
|--------|--------|-------------|-------------|------|---|-------------------|
| 0.469 | GRANITE | Metric D: global held-out BG r | `bg_validation_summary.csv` (root) | 2 | 192 (legacy) | Legacy global validation; unknown holdout strategy; predates n20 harness |
| 0.558 | IDW | Metric D: global held-out BG r | `bg_validation_summary.csv` (root) | 3 | 192 (legacy) | Same legacy context as 0.469 |
| 0.769 | GRANITE SAGE | Metric A: pooled BG r | `experiments/ablation/00_baseline/results/block_group_validation.json` | 3 | 69 BGs | n20 stratified tracts, single-tract SVI mode, nationally-ranked ACS SVI |
| 0.749 | GRANITE GCN-GAT | Metric A: pooled BG r | `experiments/ablation/00_baseline/results/block_group_validation.json` | 8 | 69 BGs | Same as 0.769 |
| 0.772 | IDW | Metric A: pooled BG r | `experiments/ablation/00_baseline/results/block_group_validation.json` | 12 | 69 BGs | Same as 0.769; IDW loaded from `graveyard/disaggregation_baselines_idw_kriging.py` |
| 0.768 | Kriging | Metric A: pooled BG r | `experiments/ablation/00_baseline/results/block_group_validation.json` | 17 | 69 BGs | Same as 0.769; Kriging loaded from `graveyard/disaggregation_baselines_idw_kriging.py` |

**M0 parity numbers (current canonical reference):**

| Number | Method | Source file | Line | n |
|--------|--------|-------------|------|---|
| 0.7692 | GRANITE (SAGE) | `data/results/m0_n20_svi_parity/aggregate.csv` | 2 | 69 BGs |
| 0.8018 | Dasymetric | `data/results/m0_n20_svi_parity/aggregate.csv` | 3 | 69 BGs |
| 0.7678 | Pycnophylactic | `data/results/m0_n20_svi_parity/aggregate.csv` | 4 | 69 BGs |

---

## 3. Why 0.469 and 0.769 differ

These are different metrics from different validation contexts, not two measurements of the same thing:

**0.469** (Metric D, global held-out BG r):
- Source: `bg_validation_summary.csv` (root), a legacy artifact from before the n20 structured harness
- Validation setup: unknown holdout strategy; n_predictions=192 suggests a small subset of tracts or a direct address-level comparison that predates the proper BG aggregation harness
- Written by: unknown legacy script (not traceable to any current file in `scripts/`)
- SESSION_LOG.md line 166 explicitly flags this: "initial run flagged stop on wrong reference (r=0.469 from bg_validation_summary.csv global validation context)"

**0.769** (Metric A, pooled BG r):
- Source: `experiments/ablation/00_baseline/results/block_group_validation.json`, frozen artifact from 2026-05-18
- Validation setup: all 20 stratified n20 tracts, single-tract SVI mode (no neighbor context), predictions pooled across 69 qualifying BGs, compared against nationally-ranked ACS BG SVI from `data/processed/national_bg_svi.csv`
- Confirmed: matches m0 parity reference (0.7692) to 3 decimal places

**Mechanism of the gap:** 0.469 was computed in a context where between-tract variance was either absent or not contributing (single tract or small subset), or where predictions were compared before proper BG aggregation. The pooled n20 harness captures between-tract variance (different tracts span the full SVI range from 0.114 to 0.891), which inflates the pooled correlation because tract-mean preservation alone produces substantial between-tract signal. The 0.769 pooled BG r is largely driven by the model correctly preserving tract means, not by within-tract allocation skill.

This is the pooled-vs-per-tract metric divergence finding (Research_Status.md, locked finding 6, 2026-05-09): per-tract BG r isolates within-tract allocation skill (SAGE median=0.390, CI -0.445 to 0.697), while pooled BG r compresses both levels of variance.

---

## 4. Baseline computation paths

### Ablation pooled-BG path (00_baseline)

File: `experiments/ablation/00_baseline/run_baseline.py`

| Baseline | Source | Import / call site | In ablation path? |
|----------|--------|--------------------|------------------|
| IDW | `graveyard/disaggregation_baselines_idw_kriging.py` | Dynamically loaded at lines ~378-397; `_bg_metrics()` called for pooled validation at lines ~497-523 | Yes |
| Kriging | Same graveyard file | Same call site | Yes |
| Dasymetric | `granite/evaluation/baselines.py`, `DasymetricDisaggregation` | NOT called in 00_baseline | No |
| Pycnophylactic | `granite/evaluation/baselines.py`, `PycnophylacticDisaggregation` | NOT called in 00_baseline | No |

The 00_baseline ablation run used IDW/Kriging because they were the active baselines at the time (2026-05-18). They were retired on 2026-04-18 per SESSION_LOG.md but retained in graveyard for the ablation frozen record.

### Main pipeline path (GRANITEPipeline)

File: `granite/disaggregation/pipeline.py`

| Baseline | Source | Call site |
|----------|--------|-----------|
| Dasymetric | `granite/evaluation/baselines.py` | `_run_disaggregation_baselines()` at line ~569; instantiated as `DasymetricDisaggregation(ancillary_column='nlcd_impervious_pct')` |
| Pycnophylactic | Same | `_run_disaggregation_baselines()` at line ~570; `PycnophylacticDisaggregation(n_iterations=50, k_neighbors=8)` |
| IDW | `graveyard/disaggregation_baselines_idw_kriging.py` | Retired; not called in main pipeline |
| Kriging | Same graveyard file | Retired; not called in main pipeline |

### Global held-out path (BlockGroupValidator)

File: `granite/validation/block_group_validation.py`

- `BlockGroupValidator` computes pooled BG r across all test tracts
- Baseline predictions extracted from `result['baseline_comparison']` (pipeline result dict)
- Since main pipeline now uses Dasymetric/Pycnophylactic, global held-out path receives those baselines
- IDW/Kriging do not appear in any live validation path

### M0 parity path

File: `granite/scripts/run_m0_parity.py`

- Calls `GRANITEPipeline._process_single_tract()` in single-tract SVI mode
- Dasymetric and Pycnophylactic extracted from pipeline's `_run_disaggregation_baselines`
- BG validation via `BlockGroupValidator` with `svi_ranking_scope='national'`

---

## 5. Per-tract BG r divergence (M0 Dasymetric vs GRANITE)

Source: `data/results/m0_n20_svi_parity/RESULTS.md` and `aggregate.csv`

### Per-tract median BG r (bootstrap, n=19 tracts with qualifying BGs)

| Method | median_bg_r | CI low 95 | CI high 95 |
|--------|-------------|-----------|------------|
| GRANITE | 0.3901 | -0.4452 | 0.6974 |
| Dasymetric | 0.7867 | 0.2531 | 0.8627 |
| Pycnophylactic | 0.2078 | -0.3526 | 0.5289 |

### Pairwise separability (bootstrap on per-tract paired differences)

| Pair | obs_median_diff | CI low 95 | CI high 95 | separable |
|------|----------------|-----------|------------|-----------|
| granite_vs_dasymetric | -0.1207 | -0.5356 | 0.108 | **False** |
| granite_vs_pycno | 0.0163 | -0.1068 | 0.2067 | False |
| dasymetric_vs_pycno | 0.4027 | 0.0435 | 0.6398 | **True** |

**Interpretation:** Dasymetric's within-tract advantage (median_bg_r = 0.787 vs GRANITE's 0.390) is not statistically separable from GRANITE at the 95% level (CI on difference spans zero). However, Dasymetric is separable from Pycnophylactic. The pooled BG r values (0.769 vs 0.802) are not separable by any test.

Per-tract BG r is the metric on which methods most visibly separate. It is also the most unreliable metric due to small per-tract BG counts (most tracts have 2-5 qualifying BGs).

---

## 6. Summary: which numbers to cite and from where

| Context | Metric to use | Canonical source |
|---------|--------------|-----------------|
| Headline performance claim | Pooled BG r | `data/results/m0_n20_svi_parity/aggregate.csv` |
| Architecture comparison | Pooled BG r (00_baseline) | `experiments/ablation/00_baseline/results/block_group_validation.json` |
| Within-tract divergence | Per-tract median BG r | `data/results/m0_n20_svi_parity/RESULTS.md` |
| Legacy IDW framing | Do not use | `bg_validation_summary.csv` is superseded |

Do not cite 0.469 or 0.558 as current results. They are from a superseded validation context with unknown holdout strategy. The current canonical numbers are 0.769 (GRANITE SAGE), 0.802 (Dasymetric), 0.768 (Pycnophylactic) from `data/results/m0_n20_svi_parity/aggregate.csv`.
