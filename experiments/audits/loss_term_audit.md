# GRANITE loss term audit

Audit of all loss terms and penalty functions in the training codebase.
Each entry follows the same field structure.

For training losses: full checklist (functional form, gradient routing, weight chain,
name-vs-behavior verdict, recommendation).
For reporting metrics: reduced checklist (functional form, what it measures, name
accuracy, manuscript presence).

---

## Terms to audit

From `MultiTractGNNTrainer._compute_multi_tract_losses` (`granite/models/gnn.py`, lines 1183-1282):
- [x] constraint_loss (entry 002)
- [x] variation_loss (entry 003)
- [x] bounds_loss (entry 004)
- [x] bg_constraint_loss (entry 005)
- [x] ordering_loss (entry 006)

From `AccessibilityGNNTrainer._compute_losses` (`granite/models/gnn.py`, lines 588-640):
Note: the prompt referenced "single-tract `compute_loss`"; the actual function name is `_compute_losses`.
- [x] constraint_loss (entry 007)
- [x] variation_loss (entry 008)
- [x] bounds_loss (entry 009)
- [x] range_loss (entry 010)
- [x] accessibility_consistency_loss (entry 011)

From `compare_gnn_idw.py` (scope addendum):
- [x] smoothness_loss (entry 001)

---

## Entry 001: `smoothness_loss` in `granite/evaluation/compare_gnn_idw.py`

**Audited:** 2026-05-22
**Location:** `granite/evaluation/compare_gnn_idw.py`, lines 144-154, inside `train_constrained_gnn`
**Status:** active (not part of main pipeline -- see consumer note)

---

### Functional form

```python
src, dst = graph.edge_index
smoothness_loss = F.mse_loss(predictions[src], predictions[dst])
```

Mean squared error over all directed edges in the KNN graph (k=8 nearest neighbors,
bidirectional). For each address, approximately 16 edge slots (8 neighbors * 2 directions).
MSE is averaged over all edge pairs. This is a genuine graph-edge smoothness penalty: it
penalizes prediction differences between spatially adjacent addresses.

---

### Training loss or reporting metric

**Training loss.** Called inside a standard training loop (`optimizer.zero_grad()` /
`total_loss.backward()` / `optimizer.step()`). It is not returned, logged, or reported as
a metric anywhere. Its value is internal to the training loop and not visible outside
`train_constrained_gnn`.

---

### Gradient routing

Gradients flow to every node that participates in at least one edge (all nodes in a
connected graph). Each node receives gradient from the MSE on its outgoing edges and
its incoming edges (both directions are in `edge_index`). Gradient magnitude per node
scales with the sum of squared prediction differences to its k neighbors. In practice
this pushes spatially adjacent addresses toward similar predicted values, creating local
spatial smoothing.

---

### Weight chain

Hardcoded coefficient `0.5` at the call site (line 154). No config key. No external
visibility. No internal multiplier inside the computation. Effective coefficient on the
mean edge MSE is exactly `0.5`.

```python
total_loss = (
    3.0 * constraint_loss +
    1.0 * variation_loss +
    0.5 * smoothness_loss +   # <- this term
    1.0 * bounds_loss
)
```

No undocumented internal constant (compare: the deleted `_compute_cross_tract_smoothness`
applied `* 0.05` inside the function body, obscuring the effective coefficient from the
call site). This term is transparent.

---

### Consumer note

`train_constrained_gnn` is called only from `main()` in the same file (line 268). The
file is a standalone evaluation script, not imported by the main GRANITE pipeline
(`granite/disaggregation/pipeline.py`), not referenced in `config.yaml`, and not called
from any ablation runner. It is not part of `MultiTractGNNTrainer`. Its training loop,
model, optimizer, and losses are entirely separate from the production training path.

The script appears in git history from commit `596b90e` onward. It is not referenced in
`Research_Status.md`, `SESSION_LOG.md`, `docs/`, or `for_mehdi_review/`. No manuscript
framing depends on its behavior.

---

### Name vs behavior verdict

**Accurate.** The comment "Smoothness loss: nearby nodes should have similar values" and
the variable name `smoothness_loss` both correctly describe what the term does: it
penalizes prediction differences between connected (spatially proximate) nodes in the
graph. The functional form matches the intent. The name is accurate and non-misleading.

---

### Comparison with deleted `gnn.py` term

The deleted `_compute_cross_tract_smoothness` in `granite/models/gnn.py` shared the same
name ("cross-tract smoothness") and the same design intent (discouraging large prediction
differences across spatial units) but implemented a completely different functional form:
`(max_tract_mean - min_tract_mean) * 0.05`. That form had no graph structure, operated
on tract-mean aggregates, and penalized only the two extreme tracts per step.

The `compare_gnn_idw.py` term is what `_compute_cross_tract_smoothness` should have
implemented: edge-level MSE over a spatial graph. **Flag:** the deleted `gnn.py` term
was a misimplemented placeholder for the form that `compare_gnn_idw.py` correctly
instantiates. If a true graph-edge smoothness penalty is ever reintroduced into
`MultiTractGNNTrainer`, this implementation is the reference.

---

### Recommendation

**No action required.** This term is not part of the main pipeline, not auditable
against sweep data, and not referenced in any framing document. Its functional form is
correct. The only note for future reference: if `compare_gnn_idw.py` is ever promoted
to drive production evaluation, the hardcoded `0.5` should be surfaced as a config key
and subjected to a sweep equivalent to Step 3.

---

## Entry 002: `constraint_loss` in `MultiTractGNNTrainer._compute_multi_tract_losses`

**Audited:** 2026-05-27
**Location:** `granite/models/gnn.py`, lines 1196-1210, inside `_compute_multi_tract_losses`
**Status:** active (constrained path); zero-multiplied in unconstrained path

---

### Functional form

```python
for fips, target_svi in tract_targets.items():
    mask = tract_masks[fips]
    tract_predictions = predictions[mask]
    if len(tract_predictions) > 0:
        tract_mean = tract_predictions.mean()
        tract_loss = F.mse_loss(tract_mean.unsqueeze(0), target_svi)
        constraint_losses.append(tract_loss)
constraint_loss = torch.mean(torch.stack(constraint_losses))
```

Per-tract MSE between the mean of address-level predictions within each tract and the
known tract-level SVI target. Averaged (not summed) across tracts, so the magnitude is
independent of the number of tracts.

---

### Training loss or reporting metric

**Training loss.** Returned in the losses dict and consumed by the caller (`total_loss`).
Tracked in `training_history['constraint_errors']` as a percentage error metric
(separate from the loss value). `Research_Status.md` references this term explicitly:
"Soft MSE constraint term in the training loss (weight 2.0 in ... `_compute_multi_tract_losses`)."

---

### Gradient routing

Gradients flow to all address-level predictions through `predictions[mask].mean()`.
The mean operation distributes gradient equally across all addresses in the tract mask.
All addresses in all tracts receive gradient; the per-tract averaging means gradient
magnitude per address scales with `constraint_weight / n_addresses_in_tract`.

---

### Weight chain

Config key `constraint_weight` in the `training` config block; default `2.0`. Read at
`MultiTractGNNTrainer.__init__:710` via `config.get('constraint_weight', 2.0 if self.enforce_constraints else 0.0)`.
Applied at call site: `self.constraint_weight * constraint_loss`. No internal constant.
In unconstrained mode (`enforce_constraints=False`) the multiplier is explicitly `0.0`
in the weighted sum (not the stored attribute, which reads `0.0` from the config default).

All committed config snapshots use the default: `constraint_weight: 2.0`.

---

### Cross-checks

- Referenced outside the computing function: yes. `losses['constraint']` is logged in the
  multi-task verbose path (line 929) and used in `_compute_overall_constraint_error` (a
  separate reporting metric, not a gradient path). `Research_Status.md` line 43 describes
  the term and its weight.
- Config key: `training.constraint_weight`, default 2.0. All committed experiment snapshots
  use 2.0. `enforce_constraints: true` in all committed configs.
- Docstring: `_compute_multi_tract_losses` has a docstring that says "When enforce_constraints=False,
  the model learns purely from accessibility patterns without mean-matching pressure." Accurate.
- `Research_Status.md` passage: "Soft MSE constraint term in the training loss (weight 2.0 in
  single-tract `_compute_losses` at `granite/models/gnn.py:562`; same weight in multi-tract
  `_compute_multi_tract_losses` at `granite/models/gnn.py:1149`)." Line numbers are stale
  (current lines are ~593 and ~1204) but the description is accurate.
- DEFENSE_FRAMING.md: does not exist in repo.
- Ecological_Fallacy_Finding.md: does not exist in repo.
- GRANITE_GNN_Architecture_Spec.md: does not exist in repo.

---

### Name vs behavior verdict

**Yes.** `constraint_loss` is the per-tract mean-matching MSE. The name correctly
describes the methodological role: enforcing the aggregate constraint that predicted
address-level means equal the known tract-level SVI.

---

### Recommendation

**Keep.** Correct name, correct form, config-governed weight, no internal constants.

---

## Entry 003: `variation_loss` in `MultiTractGNNTrainer._compute_multi_tract_losses`

**Audited:** 2026-05-27
**Location:** `granite/models/gnn.py`, lines 1212-1226, inside `_compute_multi_tract_losses`
**Status:** active

---

### Functional form

```python
for fips, mask in tract_masks.items():
    tract_predictions = predictions[mask]
    if len(tract_predictions) > 10:
        tract_std = tract_predictions.std()
        min_variation = 0.02
        variation_loss = F.relu(min_variation - tract_std)
        variation_losses.append(variation_loss)
variation_loss = torch.mean(torch.stack(variation_losses))
```

Per-tract hinge loss that is nonzero only when within-tract prediction standard deviation
falls below `0.02`. The threshold `min_variation = 0.02` is hardcoded inside the
function body. For tracts with fewer than 11 addresses, no variation loss is computed.

---

### Training loss or reporting metric

**Training loss.** Applied with weight `0.8` in the constrained path and `2.0` in the
unconstrained path. Both are hardcoded at the call site. The config key `variation_weight`
(value `1.5` in all committed config snapshots) is defined in `config.yaml` but is
**never read** by `MultiTractGNNTrainer`. The effective coefficient is `0.8`, not `1.5`.

---

### Gradient routing

Gradient is nonzero only when `tract_std < 0.02`. When active, the gradient flows
through `std()` to all predictions within the tract, pushing them toward greater spread.
When the tract is already diverse enough (`std >= 0.02`), the hinge is inactive and no
gradient flows from this term.

---

### Weight chain

Hardcoded at the call site: `0.8 * variation_loss` (constrained) or `2.0 * variation_loss`
(unconstrained). Internal constant `min_variation = 0.02` controls activation threshold.
Config key `variation_weight: 1.5` exists in `config.yaml` and all committed experiment
config snapshots but is never consumed by `MultiTractGNNTrainer.__init__` or
`_compute_multi_tract_losses`. The config key is orphaned.

---

### Cross-checks

- Referenced outside the computing function: `losses['variation']` is returned and
  available to callers but is not logged or tracked in any training history entry.
- Config key: `training.variation_weight` = 1.5 in all committed configs. **Not consumed
  by the trainer.** The single-tract trainer also does not read this key. The config key
  is dead.
- Docstring: none on `_compute_multi_tract_losses` variation section. The function-level
  docstring does not mention this term.
- Research_Status.md: not mentioned.
- Internal constant `min_variation = 0.02`: not visible at call site.

**Flag: orphaned config key and hidden threshold.** `variation_weight: 1.5` in
`config.yaml` has no effect. The actual weight is `0.8` and is not configurable. The
activation threshold `0.02` is also not configurable.

---

### Name vs behavior verdict

**Yes.** `variation_loss` penalizes insufficient within-tract prediction variance.
The name matches the behavior. The implementation detail (threshold = 0.02) is not
communicated by the name, but "variation loss" correctly identifies the intent.

---

### Recommendation

**Keep, but flag two housekeeping items:** (1) remove the orphaned `variation_weight`
key from `config.yaml` or wire it to the trainer; (2) surface `min_variation = 0.02`
as a config key or at minimum a named constant.

---

## Entry 004: `bounds_loss` in `MultiTractGNNTrainer._compute_multi_tract_losses`

**Audited:** 2026-05-27
**Location:** `granite/models/gnn.py`, line 1229, inside `_compute_multi_tract_losses`
**Status:** active (both constrained and unconstrained paths)

---

### Functional form

```python
bounds_loss = torch.mean(F.relu(predictions - 1.0)) + torch.mean(F.relu(-predictions))
```

Sum of two hinge losses: one penalizing predictions above 1.0, one penalizing predictions
below 0.0. Each hinge is a mean over all addresses (not per-tract). No internal threshold
or multiplier; the bounds are implicit in the formula.

---

### Training loss or reporting metric

**Training loss.** Applied with weight `1.0` in both constrained and unconstrained paths
(hardcoded). No config key.

---

### Gradient routing

Gradients flow to all addresses, regardless of tract membership. Addresses with predictions
in `[0, 1]` contribute zero gradient from this term. Addresses outside the range receive
gradient proportional to the excess.

---

### Weight chain

Hardcoded at call site: `1.0 * bounds_loss` in both branches. No config key. No internal
constant (the bounds 0 and 1 are structural to the SVI domain definition, not arbitrary).

---

### Cross-checks

- Referenced outside the computing function: `losses['bounds']` returned but not logged
  or tracked separately.
- Config key: none.
- Docstring: the function-level comment says "3. Bounds enforcement (always active)"
  confirming both-path activation.
- Research_Status.md: not mentioned.
- No documents in the specified list mention `bounds_loss` by name.

---

### Name vs behavior verdict

**Yes.** Enforces predictions to lie within `[0, 1]`, which is the SVI domain.
The name matches. The activation in both constrained and unconstrained modes is
correctly described by the inline comment "always active."

---

### Recommendation

**Keep.** Correct name, correct form, no internal constants, no config key needed
(the SVI range is not a tunable parameter).

---

## Entry 005: `bg_constraint_loss` in `MultiTractGNNTrainer._compute_multi_tract_losses`

**Audited:** 2026-05-27
**Location:** `granite/models/gnn.py`, lines 1231-1244, inside `_compute_multi_tract_losses`
**Status:** conditional (zero when `block_group_targets` is None; inactive in unconstrained mode)

---

### Functional form

```python
if block_group_targets is not None and block_group_masks is not None:
    bg_losses = []
    for bg_id, bg_target in block_group_targets.items():
        bg_mask = block_group_masks[bg_id]
        bg_predictions = predictions[bg_mask]
        if len(bg_predictions) > 0:
            bg_mean = bg_predictions.mean()
            bg_loss = F.mse_loss(bg_mean.unsqueeze(0), bg_target)
            bg_losses.append(bg_loss)
    if len(bg_losses) > 0:
        bg_constraint_loss = torch.mean(torch.stack(bg_losses))
```

Per-block-group MSE between mean address predictions and target block group SVI.
Structurally identical to `constraint_loss` (entry 002) but at block-group resolution.
BG masks are subsets of tract masks; addresses can participate in both constraints.

---

### Training loss or reporting metric

**Training loss.** Added to `total_loss` only in the constrained path and only when
`block_group_targets` is not None (lines 1261-1263). Not added in unconstrained path.
Returned in losses dict as `'bg_constraint'`. Tracked in `training_history['bg_constraint_errors']`
as percentage error when BG targets are provided.

---

### Gradient routing

When active: gradients flow to all addresses within each BG mask via the mean operation.
Gradient magnitude per address scales with `bg_constraint_weight / n_addresses_in_bg`.
When block group targets are not provided, no gradient flows from this term.

---

### Weight chain

Config key `bg_constraint_weight`, default `1.0`. Read at `MultiTractGNNTrainer.__init__:712`
via `config.get('bg_constraint_weight', 1.0)`. Applied at call site:
`total_loss = total_loss + self.bg_constraint_weight * bg_constraint_loss`.

Non-default values seen in committed code:
- `recovery_harness.py:453`: `'bg_constraint_weight': 0.0` (BG constraint disabled for M1/M2 recovery)
- `run_m2_sweep.py:277`: `'bg_constraint_weight': 0.0`
- `scripts/multi_tract_experiment.py:295`: set to 1.0 if BG data provided, else 0.0
- `scripts/national_bg_convergence_experiment.py:216,218`: conditionally 1.0 or 0.0
- `scripts/bg_convergence_experiment.py:190,192`: conditionally 1.0 or 0.0

No internal constant.

---

### Cross-checks

- Referenced outside computing function: `losses['bg_constraint']` in returned dict.
  `training_history['bg_constraint_errors']` tracks this separately. `results['bg_constraint_error']`
  and `results['per_bg_errors']` reported in final results dict when BG targets present.
- Config key: `bg_constraint_weight`, default 1.0. Actively consumed.
- Docstring: `docs/recovery_harness_schema.md` line 119: "explicitly set to weight 0 for M1
  via `bg_constraint_weight=0.0` and `ordering_weight=0.0`." Confirms intentional deactivation.
- Research_Status.md: not mentioned by name.
- DEFENSE_FRAMING.md, Ecological_Fallacy_Finding.md, GRANITE_GNN_Architecture_Spec.md: do not exist.

---

### Name vs behavior verdict

**Yes.** `bg_constraint_loss` is the block-group-level analog of `constraint_loss`.
The name correctly identifies both the spatial scale (bg = block group) and the role
(constraint enforcement via mean-matching MSE).

---

### Recommendation

**Keep.** Correct name, correct form, config-governed weight, no internal constants.
The conditional activation (None check) is correct and necessary.

---

## Entry 006: `ordering_loss` in `MultiTractGNNTrainer._compute_multi_tract_losses`

**Audited:** 2026-05-27
**Location:** `granite/models/gnn.py`, lines 1246-1251 (call site) and 1172-1181 (implementation),
inside `_compute_multi_tract_losses` and `_compute_ordering_loss`
**Status:** conditional (zero when `ordering_pairs` is None)

---

### Functional form

```python
# _compute_ordering_loss:
pred_low = predictions[low_value_indices]
pred_high = predictions[high_value_indices]
pair_loss = torch.relu(pred_high - pred_low + margin)
return pair_loss.mean()
```

Margin-based ranking loss (hinge). `low_value_indices` are addresses with low property
appraisal value (log_appvalue); `high_value_indices` are addresses with high property
value. The loss is nonzero when a high-value address predicts higher SVI vulnerability
than a low-value address by more than `-margin` (i.e., when the ordering is violated
or nearly violated). Pairs are sampled within block groups (or tracts) with a minimum
value gap (`ordering_min_gap = 0.5` log-dollar units by default).

---

### Training loss or reporting metric

**Training loss.** Added to `total_loss` only in the constrained path and only when
`ordering_pairs` is not None. Logged verbosely during training: "Ordering:
loss=..., violated=N/M pairs." Tracked in `training_history['ordering_losses']`.

---

### Gradient routing

Gradients flow to both members of each active pair (where the ranking is violated or
within margin). `pred_low` receives negative gradient (push toward higher vulnerability
prediction); `pred_high` receives positive gradient (push toward lower vulnerability
prediction). Pairs with `pred_high - pred_low + margin <= 0` contribute zero gradient.

---

### Weight chain

Config key `ordering_weight`, default `0.5`. Read at `MultiTractGNNTrainer.__init__:715`
via `config.get('ordering_weight', 0.5)`. Applied at call site:
`total_loss = total_loss + self.ordering_weight * ordering_loss`.
`margin` parameter: `self.ordering_margin`, from `config.get('ordering_margin', 0.02)`.
`min_gap` parameter: `self.ordering_min_gap`, from `config.get('ordering_min_gap', 0.5)`.

All three parameters are fully config-governed. No internal constants. All committed
config snapshots use defaults: `ordering_weight: 0.5`, `ordering_min_gap: 0.5`,
`ordering_margin: 0.02`.

Non-default values in committed code:
- `recovery_harness.py:453`: `ordering_weight: 0.0` (deactivated for M1/M2)
- `run_m2_sweep.py:277`: `ordering_weight: 0.0`

---

### Cross-checks

- Referenced outside computing function: `losses['ordering']` in returned dict.
  `training_history['ordering_losses']` tracks per-epoch values. Verbose logging
  in training loop reports violated pair counts.
- Config key: `ordering_weight`, `ordering_min_gap`, `ordering_margin`. All consumed.
- Docstring on `_compute_ordering_loss`: "margin-based ranking loss: low-value properties
  should predict higher vulnerability." Accurate.
- Research_Status.md: not mentioned.
- DEFENSE_FRAMING.md, Ecological_Fallacy_Finding.md, GRANITE_GNN_Architecture_Spec.md: do not exist.

---

### Name vs behavior verdict

**Yes.** `ordering_loss` encodes a property-value-to-vulnerability ordering prior.
The name correctly communicates the intent (pairwise ordering constraint). The
implementation is the standard margin-based ranking loss. The semantic direction
(low value = high vulnerability) is documented in the function comment.

---

### Recommendation

**Keep.** Correct name, correct form, fully config-governed, no internal constants.

---

## Entry 007: `constraint_loss` in `AccessibilityGNNTrainer._compute_losses`

**Audited:** 2026-05-27
**Location:** `granite/models/gnn.py`, lines 591-593, inside `_compute_losses`
**Status:** active (constrained path); zero-multiplied in unconstrained path

---

### Functional form

```python
predicted_mean = predictions.mean()
constraint_loss = F.mse_loss(predicted_mean.unsqueeze(0), target_svi)
```

Single-tract MSE between the mean of all address-level predictions and the scalar
tract SVI target. Simpler than the multi-tract version (entry 002): no loop, no
masking, because this trainer handles one tract at a time.

---

### Training loss or reporting metric

**Training loss.** Returned in losses dict as `'constraint'`. Tracked in
`training_history['constraint_errors']` as a percentage (separate path from the
loss tensor).

---

### Gradient routing

Gradient flows to all predictions equally through the mean operation. Every address
receives identical gradient magnitude `constraint_weight / n_addresses`.

---

### Weight chain

Config key `constraint_weight`, default `2.0 if enforce_constraints else 0.0`.
Read at `AccessibilityGNNTrainer.__init__:445`. Applied at call site:
`self.constraint_weight * constraint_loss`. No internal constant.

In unconstrained mode the call site explicitly uses `0.0 * constraint_loss` regardless
of `self.constraint_weight`.

---

### Cross-checks

- Referenced outside computing function: `losses['constraint']` returned. Percentage
  error variant computed separately in training loop (line 531).
- Config key: `training.constraint_weight`, default 2.0. All committed configs use 2.0.
- Docstring: none on `_compute_losses`.
- Research_Status.md: "Soft MSE constraint term in the training loss (weight 2.0 in
  single-tract `_compute_losses` at `granite/models/gnn.py:562`)." Line 562 is stale
  (current line ~593) but the description is accurate.

---

### Name vs behavior verdict

**Yes.** Same verdict as entry 002: the name matches the methodological role.

---

### Recommendation

**Keep.** Correct name, correct form, config-governed weight, no internal constants.

---

## Entry 008: `variation_loss` in `AccessibilityGNNTrainer._compute_losses`

**Audited:** 2026-05-27
**Location:** `granite/models/gnn.py`, lines 595-598, inside `_compute_losses`
**Status:** active (both paths, different weights)

---

### Functional form

```python
spatial_std = predictions.std()
min_variation = 0.02
variation_loss = F.relu(min_variation - spatial_std)
```

Hinge loss on global prediction standard deviation (all addresses in the single tract).
Nonzero only when `std < 0.02`. No per-address masking (single-tract trainer covers one
tract). Structurally identical to entry 003 but without the per-tract loop.

---

### Training loss or reporting metric

**Training loss.** Applied with hardcoded `1.5` (constrained) or `2.0` (unconstrained).
`spatial_std` is tracked in `training_history['spatial_stds']` as a diagnostic metric
(line 536), independent of the loss.

---

### Gradient routing

When active: gradient flows through `std()` to all predictions, pushing them toward
greater spread. When `std >= 0.02`, no gradient from this term.

---

### Weight chain

Hardcoded at call site: `1.5 * variation_loss` (constrained) or `2.0 * variation_loss`
(unconstrained). Internal constant `min_variation = 0.02` controls activation threshold.
The value `1.5` matches `variation_weight: 1.5` in `config.yaml`, but `AccessibilityGNNTrainer`
does not read `variation_weight` from config. The match is coincidental.

**Flag: same orphaned config key as entry 003.** `variation_weight: 1.5` has no effect
on either trainer. Both trainers hardcode their variation weights.

---

### Cross-checks

- Referenced outside computing function: `losses['variation']` returned but not logged.
- Config key: `training.variation_weight: 1.5` in all committed configs. **Not consumed.**
- Research_Status.md: not mentioned.
- Internal constant `min_variation = 0.02`: not visible at call site.

---

### Name vs behavior verdict

**Yes.** Penalizes insufficient prediction spread. Name matches behavior.

---

### Recommendation

**Keep, same housekeeping as entry 003:** remove or wire the orphaned `variation_weight`
config key; surface `min_variation = 0.02` as a named constant.

---

## Entry 009: `bounds_loss` in `AccessibilityGNNTrainer._compute_losses`

**Audited:** 2026-05-27
**Location:** `granite/models/gnn.py`, line 601, inside `_compute_losses`
**Status:** active (both constrained and unconstrained paths)

---

### Functional form

```python
bounds_loss = torch.mean(F.relu(predictions - 1.0)) + torch.mean(F.relu(-predictions))
```

Identical formula to entry 004. Sum of two hinge losses enforcing predictions in `[0, 1]`.

---

### Training loss or reporting metric

**Training loss.** Applied with hardcoded weight `1.0` in both paths. No config key.

---

### Gradient routing

Identical to entry 004: all out-of-range addresses receive gradient; in-range addresses
contribute zero gradient from this term.

---

### Weight chain

Hardcoded `1.0` at call site. No config key. No internal constant beyond the domain
bounds 0 and 1.

---

### Cross-checks

- Same pattern as entry 004.
- `losses['bounds']` returned. Not logged separately.
- Research_Status.md: not mentioned.

---

### Name vs behavior verdict

**Yes.** Same verdict as entry 004.

---

### Recommendation

**Keep.** Correct, no action needed.

---

## Entry 010: `range_loss` in `AccessibilityGNNTrainer._compute_losses`

**Audited:** 2026-05-27
**Location:** `granite/models/gnn.py`, lines 604-609, inside `_compute_losses`
**Status:** conditional (skipped when `n_addresses <= 10`; different weights per path)

---

### Functional form

```python
if n_addresses > 10:
    prediction_range = predictions.max() - predictions.min()
    min_range = 0.05
    range_loss = F.relu(min_range - prediction_range)
else:
    range_loss = torch.tensor(0.0, device=predictions.device)
```

Hinge loss on the prediction range (max - min). Nonzero only when the full prediction
span falls below `0.05`. Applied only when `n_addresses > 10`; otherwise returns zero
tensor. Note: this term is present only in the single-tract trainer. The multi-tract
trainer (`_compute_multi_tract_losses`) does not include a range term.

---

### Training loss or reporting metric

**Training loss.** Applied with weight `0.3` (constrained) or `0.5` (unconstrained),
both hardcoded. No config key.

---

### Gradient routing

When active: gradient flows through `max()` and `min()` to the single highest-predicted
and single lowest-predicted address respectively. All other addresses receive zero
gradient from this term. The distribution of gradient is highly concentrated, unlike
the variation term which distributes through `std()`.

---

### Weight chain

Hardcoded at call site: `0.3 * range_loss` (constrained) or `0.5 * range_loss`
(unconstrained). Internal constant `min_range = 0.05` controls activation threshold.
No config key for weight or threshold.

---

### Cross-checks

- Referenced outside computing function: `losses['range']` returned but not logged.
- Config key: none.
- Docstring: inline comment "4. Distribution regularization." The function comment uses
  "distribution regularization" as the section label; the variable is `range_loss`.
  These names are different but not contradictory (range is one aspect of distribution).
- Research_Status.md: not mentioned.
- Present only in single-tract trainer; absent from multi-tract trainer. The multi-tract
  path relies on `variation_loss` alone for spread encouragement.

**Flag: gradient routing is degenerate.** `max()` and `min()` route all gradient to
exactly one address each, creating a maximally concentrated gradient. Compare to
`variation_loss` which distributes through `std()`. At small `n_addresses`, this
term is explicitly disabled (n > 10 guard), but at moderate tract sizes the sparse
gradient may not effectively encourage spread.

---

### Name vs behavior verdict

**Yes.** `range_loss` penalizes insufficient prediction range (max - min). The name
matches. The inline label "Distribution regularization" in the code is a less precise
alias, but the variable name is accurate.

---

### Recommendation

**Keep.** Functionally correct for its stated purpose. The degenerate gradient routing
note is informational; it is not a bug. If the term ever becomes active research
infrastructure (rather than a soft nudge in a secondary trainer), the gradient routing
deserves reconsideration.

---

## Entry 011: `accessibility_consistency_loss` in `AccessibilityGNNTrainer._compute_losses`

**Audited:** 2026-05-27
**Renamed:** `_compute_accessibility_consistency_loss` -> `_compute_min_spread_loss`; variable
`accessibility_consistency_loss` -> `min_spread_loss`; dict key `'accessibility'` -> `'min_spread'`
on 2026-05-27 (audit followup commit; see SESSION_LOG.md entry 2026-05-27).
**Location (post-rename):** `granite/models/gnn.py`, inside `_compute_losses`
and `_compute_min_spread_loss`
**Status:** active (both paths, different weights)

---

### Functional form

```python
# _compute_accessibility_consistency_loss:
if len(predictions) < 4:
    return torch.tensor(0.0, device=predictions.device)
sorted_preds = torch.sort(predictions)[0]
if len(sorted_preds) > 1:
    pred_gradient = sorted_preds[1:] - sorted_preds[:-1]
    gradient_loss = F.relu(0.001 - pred_gradient.mean())
else:
    gradient_loss = torch.tensor(0.0, device=predictions.device)
return gradient_loss
```

Sorts predictions and computes the mean of consecutive differences (the "gradient"
of the sorted sequence). Applies a hinge loss penalizing this mean when it falls
below `0.001`. In effect: penalizes prediction distributions where the sorted
values are nearly flat (all predictions nearly identical, or most predictions
clustered with a few outliers producing only a tiny mean increment per step).

**No accessibility features enter this computation.** The function takes only
`predictions` as input.

---

### Training loss or reporting metric

**Training loss.** Applied with weight `0.5` (constrained) or `1.0` (unconstrained),
both hardcoded. Returned in losses dict as `'accessibility'` (not
`'accessibility_consistency'` -- the dict key is further truncated from the variable
name).

---

### Gradient routing

When active: gradients flow through `sorted_preds[1:] - sorted_preds[:-1]`. The
sort operation is not differentiable at equal values; gradient routing through
`torch.sort` at non-degenerate configurations routes to the addresses with rank
`k` and `k+1` for each consecutive pair. The mean distributes gradient equally
across all N-1 consecutive pairs. This is a weak, diffuse gradient.

---

### Weight chain

Hardcoded at call site: `0.5 * accessibility_consistency_loss` (constrained) or
`1.0 * accessibility_consistency_loss` (unconstrained). Internal constant `0.001`
controls activation threshold. No config key for weight or threshold.

---

### Cross-checks

- Referenced outside computing function: returned in losses dict as `'accessibility'`
  (key mismatch with variable name `accessibility_consistency_loss`). Not logged or
  tracked in training history.
- Config key: none.
- Docstring on `_compute_accessibility_consistency_loss`: "Encourage structured
  predictions." This docstring is vague and does not mention accessibility features
  or explain the sorted-gradient mechanism.
- Research_Status.md: not mentioned.
- DEFENSE_FRAMING.md, Ecological_Fallacy_Finding.md, GRANITE_GNN_Architecture_Spec.md: do not exist.
- The name `accessibility_consistency` implies a consistency relationship between
  accessibility features and predictions. No accessibility features are involved.
  The function is effectively a minimum-spread penalty on the sorted prediction sequence,
  functionally similar to `variation_loss` and `range_loss` but operating on the mean
  consecutive difference rather than std or range.

---

### Name vs behavior verdict

**Partial.** The function name `accessibility_consistency_loss` and dict key
`'accessibility'` both suggest a relationship to accessibility features or
a consistency check between features and predictions. The actual behavior is a
minimum-spread penalty on sorted predictions. Accessibility features play no role.
The docstring ("Encourage structured predictions") is too vague to resolve the
discrepancy. The term does something valid (discourages collapsed prediction
distributions), but the name does not describe what it does.

---

### Recommendation

**Rename.** Suggested name: `min_spread_loss` or `sorted_gradient_loss`. The
functional form is legitimate but the name misleads any reader who expects
accessibility-feature involvement. The dict key `'accessibility'` in the returned
losses dict should also be updated to match.

---
