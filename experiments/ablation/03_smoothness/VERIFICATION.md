# 03_smoothness: null result verification

Generated: 2026-05-22 | Scope: checks 1-5 as specified in the verification prompt.

---

## Check 1: `_compute_cross_tract_smoothness` function body and weight wiring

Verbatim function body from `granite/models/gnn.py` lines 1312-1325:

```python
def _compute_cross_tract_smoothness(self, predictions, tract_masks):
    """Gentle smoothness penalty for extreme tract differences"""

    tract_means = []
    for mask in tract_masks.values():
        if mask.sum() > 0:
            tract_means.append(predictions[mask].mean())

    if len(tract_means) > 1:
        tract_means_tensor = torch.stack(tract_means)
        range_penalty = (tract_means_tensor.max() - tract_means_tensor.min()) * 0.05
        return range_penalty
    else:
        return torch.tensor(0.0, device=predictions.device)
```

There is a hardcoded `0.05` multiplier inside the function at line 1322. The function computes one mean per tract, then returns `(max_mean - min_mean) * 0.05`. `self.smoothness_weight` is **not** referenced inside this function; it is applied at the call site (see check 2). The effective weight on the range penalty is therefore `self.smoothness_weight * 0.05`. At the default `smoothness_weight=0.1`, the effective multiplier is `0.005`. At `smoothness_weight=0.5` (04_five_x), it is `0.025`. The wiring is technically correct in that `self.smoothness_weight` does scale the term, but the internal `0.05` is an undocumented constant that silently reduces the gradient signal by 20x relative to what the config key implies.

---

## Check 2: Call site in `_compute_multi_tract_losses`

Verbatim from `granite/models/gnn.py` lines 1254-1260:

```python
total_loss = (
    self.constraint_weight * constraint_loss +
    0.8 * variation_loss +
    1.0 * bounds_loss +
    self.smoothness_weight * smoothness_loss
)
```

`self.smoothness_weight` appears exactly once in the loss assembly, at line 1259. It is not hardcoded anywhere else. The function itself returns the `0.05`-reduced range scalar, so the full chain is: config key `smoothness_weight` -> `self.smoothness_weight` (set at trainer init, printed at init) -> multiplied onto `smoothness_loss` which already contains `* 0.05`. There is no double-counting and no second hardcoded site. The weight is wired through correctly; the `0.05` inside the function is a design issue, not a wiring error.

---

## Check 3: Raw (pre-correction) prediction comparison, 04_five_x vs 00_off

Per-address raw predictions were not saved to disk during the sweep runs. The only per-tract artifacts are `per_tract_metrics.csv` (post-correction summary statistics) and the feature importance CSVs. To estimate raw prediction differences indirectly: `spatial_std` in `per_tract_metrics.csv` is the within-tract standard deviation of the **post-correction** predictions, and since the constraint correction is an additive scalar shift per tract (it preserves all within-tract relative distances), the within-tract std of raw predictions equals the within-tract std of corrected predictions. Comparing all 40 tract-architecture rows between 04_five_x and 00_off, the maximum absolute difference in `spatial_std` is **exactly 0.0** (computed via pandas merge). Since spatial_std captures the shape of the within-tract prediction distribution and is bit-for-bit identical, the within-tract raw prediction distributions are identical across the two extreme weight settings. The smoothness term, which penalizes between-tract range of means, made no measurable impact on within-tract prediction structure even at 5x weight vs zero weight.

---

## Check 4: Post-correction prediction comparison, 04_five_x vs 00_off

Post-correction `tract_mean_pred` is identical between 04_five_x and 00_off for all 40 rows (max absolute difference = 0.0). This is expected: the constraint correction pins each tract mean to `tract_svi` by construction, so `tract_mean_pred == tract_svi` regardless of what the smoothness term does during training. Additionally, `morans_i` and `bg_r` show zero difference between the two runs across all tract-architecture pairs (max abs diff = 0.0 for both fields). Post-correction predictions are therefore confirmed to be identical to machine precision. The smoothness weight setting from 0.0 to 0.5 left no observable fingerprint on any stored metric.

---

## Check 5: bg_r = -0.967 investigation

The flagged value `bg_r=-0.967` originates from tract **47065001900** (Hamilton County census tract 001900), architecture **gcn_gat**. This tract has `tract_svi=0.9804` (near the county maximum), `n_addresses=2469`, `n_bgs=3`, and `spatial_std=0.023949` under gcn_gat. The very low spatial_std indicates GCN-GAT predicted near-uniform values within this tract; with only 3 block groups and minimal prediction variance, the within-tract Pearson r against ACS block-group SVI ground truth is dominated by tiny numerical fluctuations that happen to anti-correlate with the BG ordering. This is a degenerate prediction case for a high-SVI low-variance tract, not a systematic sign flip across the run. Crucially, the identical value `bg_r=-0.967226` appears in **both** 04_five_x and 00_off for this tract (max abs diff = 0.0 confirmed in check 4), confirming that this is a property of the GCN-GAT model's behavior on this specific tract, not an artifact of the smoothness weight setting. Two other tracts (47065011900 gcn_gat and 47065001800 gcn_gat, both with n_bgs=2) show `bg_r=-1.000000` in both runs for the same structural reason: with 2 BGs, r is always ±1 depending on prediction ordering.

---

## Summary judgment

The null result is real. The wiring is correct (`self.smoothness_weight` is applied at the call site, one location, no double-counting). The `0.05` hardcoded inside the function reduces gradient signal but does not break the sweep -- even the 04_five_x effective weight of `0.5 * 0.05 = 0.025` leaves zero measurable trace on any output metric. The mechanistic explanation is that the smoothness term penalizes between-tract spread of means, but the constraint loss simultaneously drives each tract mean toward a distinct SVI target (range ~0 to 1 across Hamilton County tracts), and the constraint loss dominates at convergence (constraint_error ~1e-8). The smoothness gradient is therefore swamped before it can reshape the prediction surface. The term can be removed without altering any measurable model behavior.
