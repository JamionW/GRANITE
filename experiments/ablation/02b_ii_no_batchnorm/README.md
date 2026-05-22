# 02b_ii_no_batchnorm

Ablation 2b sub-experiment: keep input LayerNorm; replace conv BatchNorm with Identity.

## Norm-layer configuration

| key | value |
|---|---|
| input_layernorm | True |
| conv_norm_type  | identity |
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
| spatial std (mean) | 0.0187 | 0.0257 |
| spatial std slope vs SVI | -0.00577 | 0.00768 |
| moran's I (mean) | 0.8760 | 0.6943 |
| block-group r (pooled) | 0.7671 | 0.7598 |

## Delta vs 2a (per-tract z-score baseline)

| metric | SAGE delta | GCN-GAT delta |
|---|---|---|
| spatial std (mean) | -0.0637 | -0.0557 |
| moran's I (mean)   | -0.0016 | -0.1548 |
| block-group r      | +0.0134 | -0.0067 |

## Artifacts

- `results/per_tract_metrics.csv`
- `results/aggregate_metrics.json`
- `results/block_group_validation.json`
- `results/feature_importance/sage_importance.csv`
- `results/feature_importance/gcngat_importance.csv`
- `figures/` (6 standard figures)

