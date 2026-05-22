# 03_smoothness: inspection finding and deletion record

git SHA at HEAD before deletion: `35cebcf5676cb5624fd300f52db5c15c15a4af24`

---

## 1. Verbatim function body (as it stood at HEAD before deletion)

From `granite/models/gnn.py`, the full body of `_compute_cross_tract_smoothness`:

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

---

## 2. Verbatim call-site line in `_compute_multi_tract_losses`

From the constrained-training branch of `_compute_multi_tract_losses`:

```python
self.smoothness_weight * smoothness_loss
```

And from the unconstrained branch (also present at deletion time):

```python
0.5 * smoothness_loss       # Spatial structure
```

The unconstrained branch had a separate hardcoded coefficient (0.5) that was never exposed to the config at all.

---

## 3. What the function actually did vs. what the name implied

The function name `_compute_cross_tract_smoothness` implies a graph-edge-level penalty on prediction differences at tract boundaries -- the kind of term that discourages discontinuities in the learned field where adjacent addresses happen to be in different tracts. The implementation contains no graph structure whatsoever: no edge indices, no adjacency, no spatial distance metric, no SVI ordering signal. It computes one scalar mean per tract, stacks them, and returns `(max - min) * 0.05`. This is a range penalty on tract-mean predictions. The only nodes it touches per training step are the two extreme tracts (the one with the highest mean and the one with the lowest mean). All intermediate tracts receive zero gradient from this term. The name "cross-tract smoothness" is therefore inaccurate on both counts: the term does not operate across tract boundaries in the graph sense, and it does not smooth predictions in any local or neighborhood sense.

---

## 4. Full weight chain and the anti-pattern

The config key `smoothness_weight` (float, default 0.1) is read at trainer init and stored as `self.smoothness_weight`. At the call site in `_compute_multi_tract_losses` it multiplies `smoothness_loss`, which is the return value of `_compute_cross_tract_smoothness`. That function applies an additional hardcoded `0.05` multiplier to the range before returning. The effective coefficient on the range is therefore:

```
smoothness_weight (config) * 0.05 (hardcoded inside function) = 0.005 at default
```

This two-layer coefficient pattern -- a tunable config weight at the call site combined with an undocumented constant inside the function body -- is the specific anti-pattern that the loss-term audit (step 3b, pending) should grep for in remaining terms. Any loss helper that returns a pre-scaled value without documenting its internal multiplier makes the effective training coefficient invisible to the config layer. Auditors and future sweep designers should check not just the call-site weight but also any literals inside each `_compute_*` helper.

---

## 5. Sweep results summary

From `summary/delta_vs_default.json` (all values rounded to 4 decimal places):

| smoothness_weight | arch    | within-tract std | cross-tract r | bg_r   |
|---|---|---|---|---|
| 0.0 (00_off)      | SAGE    | 0.0823           | 1.0000        | 0.7537 |
| 0.0 (00_off)      | GCN-GAT | 0.0814           | 1.0000        | 0.7664 |
| 0.1 (02_default)  | SAGE    | 0.0823           | 1.0000        | 0.7537 |
| 0.1 (02_default)  | GCN-GAT | 0.0814           | 1.0000        | 0.7664 |
| 0.5 (04_five_x)   | SAGE    | 0.0823           | 1.0000        | 0.7537 |
| 0.5 (04_five_x)   | GCN-GAT | 0.0814           | 1.0000        | 0.7664 |

Per VERIFICATION.md checks 3 and 4: max absolute difference across all 40 tract-architecture rows between 00_off and 04_five_x is **exactly 0.0** to machine precision for spatial_std, tract_mean_pred, morans_i, and bg_r. The term is behaviorally inert across a 500x range of its config weight.

Cross-tract signal r equals 1.0000 at all weights because the constraint correction pins each tract's mean prediction to the actual SVI value, making Pearson r between predicted tract means and actual SVI structural (both sides are the same vector).

---

## 6. Extreme-tract collapse finding (collateral to smoothness sweep)

Tract **47065001900** (Hamilton County, SVI = 0.9804) produced `bg_r = -0.967226` under GCN-GAT in both the 00_off and 04_five_x runs (identical values; confirmed independent of smoothness weight). The tract has `n_addresses = 2469`, `n_bgs = 3`, and `spatial_std = 0.023949` under GCN-GAT -- indicating near-uniform within-tract predictions. With 3 block groups and minimal prediction variance, the within-tract correlation against ACS block-group SVI is dominated by tiny numerical fluctuations whose ordering happens to anti-correlate with the ground truth ordering. A companion case at the low end of the SVI distribution is tract 47065011324 (SVI = 0.0374, 18 addresses, 1 block group), which cannot produce a meaningful bg_r and shows Moran's I collapse in the baseline run. These two tracts define the extremes of the SVI distribution in the 20-tract selection set: the low-SVI tract is too small for spatial statistics to converge; the high-SVI tract shows GCN-GAT predicting nearly flat values. Both are relevant design inputs for step 4 (constraint-by-construction): extreme-SVI tracts may warrant exclusion from the bg_r aggregate metric or a minimum-address threshold for inclusion.

---

## 7. Deletion summary

The following were removed from `granite/models/gnn.py`:
- `self.smoothness_weight = config.get('smoothness_weight', 0.1)` and associated print (trainer `__init__`)
- `smoothness_loss = self._compute_cross_tract_smoothness(predictions, tract_masks)` and comment (in `_compute_multi_tract_losses`)
- `self.smoothness_weight * smoothness_loss` from the constrained-mode total_loss sum
- `0.5 * smoothness_loss` from the unconstrained-mode total_loss sum
- `'smoothness': smoothness_loss` from the returned losses dict
- The `_compute_cross_tract_smoothness` method in its entirety

A fail-fast guard was added to trainer `__init__`: if `smoothness_weight` appears in the loaded config dict, a `ValueError` is raised pointing at this file.

`config.yaml`: the `smoothness_weight` key was removed.

Callers: `_compute_cross_tract_smoothness` had exactly one caller (`_compute_multi_tract_losses`). The evaluation script `granite/evaluation/compare_gnn_idw.py` contains a separate `smoothness_loss` local variable (edge-level MSE, lines 146 and 154) that is unrelated and was not touched.
