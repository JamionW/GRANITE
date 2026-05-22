# 03_smoothness/summary: cross-tract smoothness weight sweep

git SHA: `0d75890bbd43290209ae2cdb3de63f2be2b0846d` | seed: 42 | tracts: 20 | generated: 2026-05-22 03:56

## Inspection summary

`_compute_cross_tract_smoothness` is a between-tract mean range penalty.
It computes one scalar per tract (mean prediction over all addresses),
then returns `(max - min) * 0.05`. Effective weight on the range at
default config: `0.1 * 0.05 = 0.005`. The function contains no graph
structure. It directly suppresses the spread of tract mean predictions,
which is the same quantity `cross_tract_signal_r` depends on.
Mehdi's "remove or correct" recommendation is supported: the term
encodes no information and directly degrades between-tract discriminability.

## 02_default sanity check (must reproduce 2a within 1%)

| arch | std (run) | std (2a ref) | rel diff | bg_r (run) | bg_r (2a ref) | rel diff |
|---|---|---|---|---|---|---|
| sage | 0.0823 | 0.0823 | 0.00% | 0.7537 | 0.7537 | 0.00% |
| gcn_gat | 0.0814 | 0.0814 | 0.00% | 0.7664 | 0.7664 | 0.00% |

## Hypothesis: cross_tract_signal_r monotonic in smoothness weight

(lower weight -> less suppression -> higher r between tract mean pred and actual SVI)

**sage**: w=0.0=1.0000, w=0.025=1.0000, w=0.1=1.0000, w=0.2=1.0000, w=0.5=1.0000
**gcn_gat**: w=0.0=1.0000, w=0.025=1.0000, w=0.1=1.0000, w=0.2=1.0000, w=0.5=1.0000

## Best smoothness weight (maximize cross_tract_signal_r with bg_r >= 0.74)

- sage: `02_default` (cross_tract_signal_r = 1.0000)
- gcn_gat: `02_default` (cross_tract_signal_r = 1.0000)

## Delta vs default (02_default)

### SAGE

| variant | within-tract std | delta | cross-tract r | delta | bg_r | delta |
|---|---|---|---|---|---|---|
| 00_off | 0.0823 | +0.0000 | 1.0000 | -0.0000 | 0.7537 | +0.0000 |
| 01_quarter | 0.0823 | +0.0000 | 1.0000 | -0.0000 | 0.7537 | +0.0000 |
| 03_double | 0.0823 | +0.0000 | 1.0000 | -0.0000 | 0.7537 | +0.0000 |
| 04_five_x | 0.0823 | +0.0000 | 1.0000 | -0.0000 | 0.7537 | +0.0000 |

### GCN-GAT

| variant | within-tract std | delta | cross-tract r | delta | bg_r | delta |
|---|---|---|---|---|---|---|
| 00_off | 0.0814 | +0.0000 | 1.0000 | -0.0000 | 0.7664 | +0.0000 |
| 01_quarter | 0.0814 | +0.0000 | 1.0000 | -0.0000 | 0.7664 | +0.0000 |
| 03_double | 0.0814 | +0.0000 | 1.0000 | -0.0000 | 0.7664 | +0.0000 |
| 04_five_x | 0.0814 | +0.0000 | 1.0000 | -0.0000 | 0.7664 | +0.0000 |

## Artifacts

- `delta_vs_default.json`
- `smoothness_sweep.png`
- `between_tract_variance_sweep.png`
- `tract_pred_vs_actual.png`

## Next step

Step 4: constraint-by-construction.

