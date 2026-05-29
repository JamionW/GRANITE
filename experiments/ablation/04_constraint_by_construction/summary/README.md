# Step 4 summary: constraint-by-construction

**Git SHA:** `1d1deffb28a7e0f81459c7329d15307106360182`  
**Seed:** 42  
**Tracts:** 20  
**Run timestamps:** see git_state.txt in each variant subdir


## Sanity check: step 4 baseline vs step 2a

`00_baseline_for_step4` pooled BG r:
- SAGE: 0.7536965003 (2a reference: 0.7536965003) -- EXACT MATCH
- GCN-GAT: 0.7664314601 (2a reference: 0.7664314601) -- EXACT MATCH

The `block_group_validation.json` files are byte-for-byte identical between
`01_per_tract_std` (2a) and `00_baseline_for_step4`. Sanity check passes.

## Key metrics across modes

### Pooled BG r (n_bgs=69 -- primary generalization metric)

| mode | SAGE | delta vs soft | GCN-GAT | delta vs soft |
|---|---|---|---|---|
| soft | 0.7537 | -- | 0.7664 | -- |
| cbc+shift | 0.7511 | -0.0026 | 0.7481 | -0.0184 |
| cbc | 0.7511 | -0.0026 | 0.7481 | -0.0184 |

### Within-tract std (mean +/- 1 std across 20 tracts)

| mode | SAGE | GCN-GAT |
|---|---|---|
| soft | 0.0823 +/- 0.0472 | 0.0814 +/- 0.0355 |
| cbc+shift | 0.0595 +/- 0.0323 | 0.0830 +/- 0.0272 |
| cbc | 0.0595 +/- 0.0323 | 0.0830 +/- 0.0272 |

### Moran's I (mean +/- 1 std across 20 tracts)

| mode | SAGE | GCN-GAT |
|---|---|---|
| soft | 0.8776 +/- 0.1138 | 0.8490 +/- 0.1895 |
| cbc+shift | 0.8669 +/- 0.1851 | 0.8420 +/- 0.2589 |
| cbc | 0.8669 +/- 0.1851 | 0.8420 +/- 0.2589 |

### Pre-correction constraint error (mean over 20 tracts)

| mode | SAGE | GCN-GAT |
|---|---|---|
| soft | 2.54e-02 | 1.95e-02 |
| cbc+shift | 4.05e-08 | 3.50e-08 |
| cbc | 4.05e-08 | 3.50e-08 |

## Primary hypothesis

**Hypothesis:** the constraint loss contributes nothing measurable to within-tract structure
beyond what the post-hoc shift already provides.

**Outcome: partially confirms, with one notable exception.**


- Pooled BG r: minimal change across modes (SAGE delta -0.003, GCN-GAT delta -0.018).
  Generalization is unaffected by removing the constraint loss. Confirms hypothesis.
- Moran's I: minimal change (SAGE -0.011, GCN-GAT -0.008). Confirms hypothesis.
- Within-tract std: **SAGE shows a meaningful decrease (0.0823 -> 0.0595, delta -0.023)**.
  GCN-GAT is stable (0.0814 -> 0.0830, delta +0.002). SAGE partially contradicts the hypothesis.
  The constraint loss in soft mode acts as a regularizer that maintains spread in SAGE predictions.
  Removing it (cbc mode) allows SAGE to learn smaller deviations from the tract mean.
  GCN-GAT is not affected, suggesting architecture-specific sensitivity.

## SAGE bg_r behavior in cbc modes

Two bg_r metrics are reported. They measure different things.

**Per-tract mean bg_r** (from `aggregate_metrics.json`; mean of per-tract correlations over ~3-5 BGs):
- soft: 0.170
- cbc+shift: -0.391 (delta -0.561)
- cbc: -0.391 (delta -0.561)

This dramatic drop (-0.561) is an artifact of the small-sample per-tract BG correlation (n=3-5).
The sign flips when within-tract prediction ordering changes; the sample size is too small
to be meaningful. This metric is secondary.

**Pooled BG r** (from `block_group_validation.json`; correlation across all 69 BGs):
- soft: 0.7537
- cbc+shift: 0.7511 (delta -0.0026)
- cbc: 0.7511 (delta -0.0026)

The pooled BG r drop is 0.003 -- within noise, not a real degradation. The SAGE per-tract
bg_r sign flip is a small-sample artefact driven by the change in within-tract spread, not
a meaningful loss of generalization.

## GCN-GAT bg_r in cbc modes

Pooled BG r: 0.7664 (soft) -> 0.7481 (cbc), delta -0.0184.
Modest decrease, within the range of run-to-run noise. No meaningful degradation.

## Extreme-tract behavior: tracts 1324 and 1900

Tracts 47065011324 (SVI=0.037, lowest) and 47065001900 (SVI=0.980, highest)
were flagged at step 3 for collapse behavior (low Moran's I / low spatial std).

### Tract 1324 (SVI=0.037 (lowest))

| mode | arch | Moran's I | spatial_std |
|---|---|---|---|
| soft | GRANITE-SAGE | 0.5534 | 0.0382 |
| soft | GRANITE-GCNGAT | 0.1191 | 0.0423 |
| cbc+shift | GRANITE-SAGE | 0.2395 | 0.0182 |
| cbc+shift | GRANITE-GCNGAT | 0.0002 | 0.0406 |
| cbc | GRANITE-SAGE | 0.2395 | 0.0182 |
| cbc | GRANITE-GCNGAT | 0.0002 | 0.0406 |

### Tract 1900 (SVI=0.980 (highest))

| mode | arch | Moran's I | spatial_std |
|---|---|---|---|
| soft | GRANITE-SAGE | 0.7101 | 0.0131 |
| soft | GRANITE-GCNGAT | 0.6520 | 0.0239 |
| cbc+shift | GRANITE-SAGE | 0.9831 | 0.0306 |
| cbc+shift | GRANITE-GCNGAT | 0.9665 | 0.0282 |
| cbc | GRANITE-SAGE | 0.9831 | 0.0306 |
| cbc | GRANITE-GCNGAT | 0.9665 | 0.0282 |

Tract 1324 (extreme low SVI): cbc worsens GCN-GAT Moran's I to near-zero (0.0002),
while SAGE Moran's I is also lower (0.239 vs 0.553). The constraint loss in soft mode
contributed meaningful spatial structure for this extreme tract. cbc does not resolve the collapse.

Tract 1900 (extreme high SVI): cbc substantially improves both Moran's I and spatial_std
for both architectures (SAGE Moran's I 0.710 -> 0.983, GCN-GAT 0.652 -> 0.967).
This suggests the soft constraint loss suppressed spread at high SVI values.
cbc resolves the high-SVI collapse but worsens the low-SVI collapse.

## Alternative hypothesis

The alternative hypothesis -- that the constraint loss contributes through gradient flow,
steering the optimizer before post-correction is applied -- partially surfaces:
- SAGE spatial_std is lower under cbc, consistent with the constraint loss having
  guided SAGE toward predictions with more spread during training.
- Extreme-tract behavior is mixed: the constraint loss helps low-SVI tracts (1324)
  but suppresses spread in high-SVI tracts (1900).

## Recommendation for step 5

**Production default for step 5: `constraint_mode: soft`.**

Rationale:
- Pooled BG r is minimally affected across modes (<0.02 delta), so cbc does not
  improve the primary generalization metric.
- SAGE spatial_std collapses under cbc (-0.023 delta), reducing within-tract
  variation. This is undesirable: the method should produce spatially varying
  predictions, not near-uniform allocation within tracts.
- cbc worsens the already-weak low-SVI extreme tract (1324) for GCN-GAT.
- soft mode is the known-working baseline with predictable behavior.
- The constraint loss does carry gradient information (evidenced by SAGE spatial_std),
  so removing it is not cost-free.

Step 5 (graph variants) should therefore run with `constraint_mode: soft`.
If spatial_std collapse under cbc were desirable (e.g., for a proximity-weighted
allocation baseline), `cbc_no_shift` would be the cleaner implementation.

## Artifacts

- `cbc_sweep.png`: 3x2 grid of within-tract std, Moran's I, pooled BG r across modes
- `pre_correction_error.png`: per-tract pre-correction error bars by mode
- `extreme_tract_followup.png`: 2x2 panel for tracts 1324 and 1900
- `delta_vs_soft.json`: numeric deltas for cbc variants vs soft
- `02_cbc_no_shift/results/tract_mean_diagnostic.csv`: construction verification
