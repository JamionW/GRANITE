# 02b_iii_layernorm_in_hidden

Ablation 2b sub-experiment: keep input LayerNorm; replace conv BatchNorm with LayerNorm.

## Norm-layer configuration

| key | value |
|---|---|
| input_layernorm | True |
| conv_norm_type  | layernorm |
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
| spatial std (mean) | 0.0165 | 0.0215 |
| spatial std slope vs SVI | 0.00208 | 0.01852 |
| moran's I (mean) | 0.8483 | 0.9532 |
| block-group r (pooled) | 0.7593 | 0.7737 |

## Delta vs 2a (per-tract z-score baseline)

| metric | SAGE delta | GCN-GAT delta |
|---|---|---|
| spatial std (mean) | -0.0658 | -0.0599 |
| moran's I (mean)   | -0.0293 | +0.1042 |
| block-group r      | +0.0056 | +0.0072 |

## Artifacts

- `results/per_tract_metrics.csv`
- `results/aggregate_metrics.json`
- `results/block_group_validation.json`
- `results/feature_importance/sage_importance.csv`
- `results/feature_importance/gcngat_importance.csv`
- `figures/` (6 standard figures)

