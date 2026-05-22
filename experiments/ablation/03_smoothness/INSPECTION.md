# INSPECTION.md: `_compute_cross_tract_smoothness`

Pre-run inspection required by Step 3 protocol.
Inspected from `granite/models/gnn.py` at commit `0d75890`.

---

## Verbatim function body

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

Called at loss-computation line 1224:
```python
smoothness_loss = self._compute_cross_tract_smoothness(predictions, tract_masks)
```

Weighted at line 1255 (constrained training branch):
```python
total_loss = (
    self.constraint_weight * constraint_loss +
    0.8 * variation_loss +
    1.0 * bounds_loss +
    0.1 * smoothness_loss      # <-- hardcoded 0.1
)
```

---

## Inspection answers

### 1. What pairs/nodes does the penalty operate over?

No node pairs. No edges. The function ignores graph structure entirely.
It operates on one scalar per tract: `predictions[mask].mean()`, i.e., the
unweighted mean prediction across all addresses in that tract. With 20 tracts
it produces 20 scalars.

### 2. Functional form

`(max_tract_mean - min_tract_mean) * 0.05`

This is a **range penalty on tract-level mean predictions**. It is not MSE.
It is not a sum of pairwise differences. It penalizes only the two extreme
tracts (highest and lowest mean prediction) and is insensitive to the
distribution in between.

The effective multiplier on the range at the default config is:
`0.1 (outer weight) * 0.05 (inner factor) = 0.005 * range`.

### 3. Is "cross-tract smoothness" an accurate name?

"Cross-tract" is accurate: the penalty operates purely on between-tract
comparisons. "Smoothness" is misleading: true graph smoothness penalizes
prediction differences between connected nodes. This function contains no
graph structure -- it is a **between-tract mean range suppressor**.

### 4. Stop condition check

The Step 3 stop condition is: "smoothness term is not actually cross-tract
(e.g., it is purely within-tract edge smoothing) -- stop."

This condition does NOT apply. The function is cross-tract. The sweep design
is valid.

---

## Implications for ablation design

The hypothesis in the prompt states: "The smoothness loss compresses prediction
differences between connected nodes." That is not what the function does. The
actual mechanism is more direct and more damaging:

- It directly penalizes `max(tract_means) - min(tract_means)`, which is
  exactly the quantity `cross_tract_signal_r` depends on.
- Any tract whose mean prediction is an outlier (high or low SVI) gets pulled
  toward the center. This is precisely what Mehdi flagged as "removes or
  correct."
- Setting weight to 0.0 removes all between-tract mean compression from the
  loss. The aggregate constraint (post-correction shift) is unaffected.

The sweep design remains correct. The predicted direction of effects holds:
- `cross_tract_signal_r` should increase monotonically as weight decreases
  toward 0.
- `between_tract_variance` should increase as weight decreases.
- `within_tract_std` effect is indirect (the function has no within-tract
  component) so any movement there is a secondary effect through shared
  gradient flow.

The "remove" recommendation from Mehdi is supported by the inspection:
the term adds no information (it does not encode graph structure or SVI
ordering) and directly suppresses the discriminability the model needs.
