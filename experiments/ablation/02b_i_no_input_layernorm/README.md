# 02b_i_no_input_layernorm

Ablation 2b sub-experiment: drop input LayerNorm; keep BatchNorm on conv layers.

## Norm-layer configuration

| key | value |
|---|---|
| input_layernorm | False |
| conv_norm_type  | batchnorm |
| feature_standardization | per_tract (z-score, same as 2a) |

## Run metadata

| field | value |
|---|---|
| git SHA | `486279248c872616d5574d1917a4027b3c3a4575` |
| seed | 42 |
| tracts | 20 |
| architectures | sage, gcn_gat |

## Headline metrics

| metric | GRANITE-SAGE | GRANITE-GCNGAT |
|---|---|---|
| constraint error (mean) | 0.0000 | 0.0000 |
| spatial std (mean) | 0.0244 | 0.0348 |
| spatial std slope vs SVI | -0.03179 | -0.03472 |
| moran's I (mean) | 0.8540 | 0.7207 |
| block-group r (pooled) | 0.7713 | 0.7702 |

## Delta vs 2a (per-tract z-score baseline)

| metric | SAGE delta | GCN-GAT delta |
|---|---|---|
| spatial std (mean) | -0.0579 | -0.0466 |
| moran's I (mean)   | -0.0236 | -0.1284 |
| block-group r      | +0.0176 | +0.0037 |

## Artifacts

- `results/per_tract_metrics.csv`
- `results/aggregate_metrics.json`
- `results/block_group_validation.json`
- `results/feature_importance/sage_importance.csv`
- `results/feature_importance/gcngat_importance.csv`
- `figures/` (6 standard figures)

