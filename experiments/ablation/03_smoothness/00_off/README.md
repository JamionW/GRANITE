# 03_smoothness/00_off

Ablation 3 sub-experiment: smoothness_weight=0.0 (term fully disabled).

## Smoothness weight configuration

| key | value |
|---|---|
| smoothness_weight | 0.0 |
| feature_standardization | per_tract (2a state) |
| norm_layers | input_layernorm=True, conv_norm_type=batchnorm (2a state) |

## Run metadata

| field | value |
|---|---|
| git SHA | `0d75890bbd43290209ae2cdb3de63f2be2b0846d` |
| seed | 42 |
| tracts | 20 |
| architectures | sage, gcn_gat |

## Headline metrics

| metric | GRANITE-SAGE | GRANITE-GCNGAT |
|---|---|---|
| constraint error (mean) | 0.0000 | 0.0000 |
| spatial std (mean) | 0.0823 | 0.0814 |
| moran's I (mean) | 0.8776 | 0.8490 |
| block-group r (pooled) | 0.7537 | 0.7664 |
| between-tract variance | 0.0830 | 0.0830 |
| cross-tract signal r | 1.0000 | 1.0000 |

## Artifacts

- `results/per_tract_metrics.csv`
- `results/aggregate_metrics.json`
- `results/block_group_validation.json`
- `results/feature_importance/sage_importance.csv`
- `results/feature_importance/gcngat_importance.csv`
- `figures/` (6 standard figures)

