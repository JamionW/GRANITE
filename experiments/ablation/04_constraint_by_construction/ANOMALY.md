# ANOMALY.md -- BG r discrepancy investigation

**Date:** 2026-05-28
**Reported anomaly:** Step 4 baseline shows bg_r 0.170 (SAGE) and 0.054 (GCN-GAT) in
aggregate_metrics.json, against the claimed reference of 0.754 (SAGE) and 0.766 (GCN-GAT).

---

## 1. Measured r values verbatim

### 00_baseline (step 1)
`experiments/ablation/00_baseline/results/block_group_validation.json`
```
sage.pearson_r:    0.7691647597080817
gcn_gat.pearson_r: 0.7491051475476098
n_bgs: 69
```

### 01_per_tract_std (step 2a)
`experiments/ablation/01_per_tract_std/results/block_group_validation.json`
```
sage.pearson_r:    0.7536965003138819
gcn_gat.pearson_r: 0.7664314601020286
n_bgs: 69
```

### 00_baseline_for_step4 (step 4 baseline, constraint_mode: soft)
`experiments/ablation/04_constraint_by_construction/00_baseline_for_step4/results/block_group_validation.json`
```
sage.pearson_r:    0.7536965003138819
gcn_gat.pearson_r: 0.7664314601020286
n_bgs: 69
```

**The step 2a and step 4 baseline block_group_validation.json files are byte-for-byte identical.**
The 0.754 / 0.766 reference values are reproduced exactly.

---

## 2. Source of the apparent discrepancy

Two different BG r metrics exist in the artifact set. They measure different things and must not
be compared against each other.

**Metric A -- pooled BG r** (`block_group_validation.json` -> `pearson_r`)
Pearson r between the vector of 69 BG-level predicted SVI means and the vector of 69 BG-level
ground-truth SVI values. This is the canonical BG r. Values: ~0.754 SAGE, ~0.766 GCN-GAT.

**Metric B -- per-tract BG r mean** (`aggregate_metrics.json` -> `bg_r_mean`)
For each of the 20 tracts, compute Pearson r between the ~3-5 BGs that overlap that tract
(predicted means vs ground truth). Then average those 20 r values. With 3-5 data points per
correlation, this is high-variance and dominated by small-sample noise. Values: ~0.170 SAGE,
~0.054 GCN-GAT. These values also appear identically in `01_per_tract_std/results/aggregate_metrics.json`,
confirming they have been present since step 2a and have never been the primary BG r metric.

The anomaly report in the prior session compared Metric B (aggregate_metrics.json) against the
0.754 / 0.766 reference, which comes from Metric A (block_group_validation.json). This is an
apples-to-oranges comparison. There is no anomaly.

---

## 3. Config diff (01_per_tract_std vs 00_baseline_for_step4)

Notable differences:

- `feature_standardization`: `"global"` in 01_per_tract_std snapshot vs `per_tract` in step 4 snapshot.
  This is a config snapshot artifact. The 01_per_tract_std README explicitly states the run used
  per_tract z-score standardization (see `README.md` run metadata table: `feature_standardization: per_tract (z-score)`).
  The snapshot in that directory appears to have captured the default config.yaml rather than the
  runtime config. The identical numerical results confirm both runs used per_tract standardization.

- Step 4 snapshot contains additional keys absent from the 2a snapshot: `constraint_mode: soft`,
  `norm_layers`, `mixture`, `recovery`, `output`, `processing`. These are new config sections added
  since step 2a. They do not affect the soft-mode code path.

---

## 4. Git state diff

| field | 01_per_tract_std | 00_baseline_for_step4 |
|---|---|---|
| SHA | `486279248c872616d5574d1917a4027b3c3a4575` | `e988de442bbe8cf7bb587263be4817fb6a5b464b` |
| dirty files | none | `granite/models/gnn.py` (+71/-19 lines) |

The gnn.py modifications at step 4 HEAD add the `constraint_mode` branch. The soft-mode path
(which 00_baseline_for_step4 exercises) is reached only when `constraint_mode == 'soft'`, which
is the default. The identical block_group_validation.json values confirm the soft path is
correctly conditioned and does not alter existing behavior.

---

## 5. Origin of the 0.754 / 0.766 numbers

SESSION_LOG.md, step 2a entry (line 673):
> "SAGE BG r: -0.016 (0.769 -> 0.754); GCN-GAT BG r: +0.017 (0.749 -> 0.766); all within 0.05 flag threshold"

experiments/ablation/01_per_tract_std/README.md (block-group validation table):
> "| block-group r (pooled) | 0.7537 | 0.7664 |"

Both sources reference Metric A (pooled BG r across 69 BGs). The numbers are correct and were
correctly carried into the step 4 prompt.

---

## 6. Verdict

**The step 4 numbers are correct. Nothing broke between step 2a and step 4.**

The soft baseline (00_baseline_for_step4) reproduces the 2a baseline exactly:
- Pooled BG r: 0.7537 SAGE, 0.7664 GCN-GAT (identical to 01_per_tract_std).
- spatial_std_mean: 0.0823 SAGE, 0.0814 GCN-GAT (identical).
- morans_i_mean: 0.8776 SAGE, 0.8490 GCN-GAT (identical).

The sanity check passes. The step 4 implementation did not change the soft-mode code path.

The `bg_r_mean` field in aggregate_metrics.json (0.170 SAGE, 0.054 GCN-GAT) is a different,
noisier metric that has existed since step 2a. It should not be used as the primary BG r
reference. The canonical metric is `block_group_validation.json` -> `pearson_r`.

**Pre-condition for summary artifacts is satisfied. Step 4 may proceed to summary generation.**

---

## 7. Action items

1. The `aggregate_metrics.json` `bg_r_mean` field label is ambiguous. Future scripts should
   clearly distinguish it from the pooled BG r. No code change required for step 4.
2. The config snapshot for `01_per_tract_std` appears to have captured the pre-run default
   config rather than the runtime config. Not a reproducibility issue (results are deterministic
   and identical), but worth noting for future audit trails.
3. `pre_correction_constraint_error_mean` is NaN in all three step 4 aggregate_metrics.json
   files. This metric was not captured at the aggregate level. The per-tract values in
   per_tract_metrics.csv should be checked separately before summary generation.
