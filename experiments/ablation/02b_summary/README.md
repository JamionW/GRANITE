# 02b_summary: normalization-layer audit

Cross-variant summary for Step 2b of the GRANITE ablation series.
All three variants build on 2a (per-tract z-score standardization).

## Architecture-sensitivity prediction outcome

Hypothesis: GCN-GAT spread across 2b-i to 2b-iii exceeds SAGE spread on
both spatial std and Moran's I.

| metric | SAGE spread (max |delta|) | GCN-GAT spread | GCN-GAT > SAGE? |
|---|---|---|---|
| spatial std | 0.0658 | 0.0599 | NO |
| Moran's I   | 0.0293 | 0.1548 | YES |

**Prediction: NOT CONFIRMED** (both metrics require GCN-GAT > SAGE).

## Largest absolute movement per architecture

| metric | SAGE largest mover | GCN-GAT largest mover |
|---|---|---|
| spatial std | 2b_iii | 2b_iii |
| Moran's I   | 2b_iii | 2b_ii |

## Delta table vs 2a

### SAGE (GraphSAGE)

| variant | spatial std | delta | morans_i | delta | bg_r | delta |
|---|---|---|---|---|---|---|
| 2b-i (no input LN) | 0.0244 | -0.0579 | 0.8540 | -0.0236 | 0.7713 | +0.0176 |
| 2b-ii (Identity conv norm) | 0.0187 | -0.0637 | 0.8760 | -0.0016 | 0.7671 | +0.0134 |
| 2b-iii (LayerNorm conv norm) | 0.0165 | -0.0658 | 0.8483 | -0.0293 | 0.7593 | +0.0056 |

### GCN-GAT

| variant | spatial std | delta | morans_i | delta | bg_r | delta |
|---|---|---|---|---|---|---|
| 2b-i (no input LN) | 0.0348 | -0.0466 | 0.7207 | -0.1284 | 0.7702 | +0.0037 |
| 2b-ii (Identity conv norm) | 0.0257 | -0.0557 | 0.6943 | -0.1548 | 0.7598 | -0.0067 |
| 2b-iii (LayerNorm conv norm) | 0.0215 | -0.0599 | 0.9532 | +0.1042 | 0.7737 | +0.0072 |

## Artifacts

- `delta_vs_2a.json`: structured delta table, all three variants
- `norm_layer_sweep.png`: 2x2 panel figure resolving architecture-sensitivity prediction

## Next step

Step 3: cross-tract smoothness loss weight sweep.

git SHA: `486279248c872616d5574d1917a4027b3c3a4575` | seed: 42 | tracts: 20

