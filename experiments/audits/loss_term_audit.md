# GRANITE loss term audit

Audit of all loss terms and penalty functions in the training codebase.
Each entry follows the same field structure.

For training losses: full checklist (functional form, gradient routing, weight chain,
name-vs-behavior verdict, recommendation).
For reporting metrics: reduced checklist (functional form, what it measures, name
accuracy, manuscript presence).

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
