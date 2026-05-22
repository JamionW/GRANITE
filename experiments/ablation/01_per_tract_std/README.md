# 01_per_tract_std

Ablation step 2a in the GRANITE series.

## What changed vs 00_baseline

Single change: feature standardization replaced from global RobustScaler
(fit across all addresses in a tract, using median/IQR) to per-tract z-score
(mean/std computed separately within each tract). All model architecture,
graph construction, training hyperparameters, tract list, and random seed are
identical to 00_baseline. For single-tract mode the practical effect is that
robust scaling (median/IQR) is replaced by z-score (mean/std); multi-tract
runs would additionally normalize each tract independently.

## Run metadata

| field | value |
|---|---|
| git SHA | `486279248c872616d5574d1917a4027b3c3a4575` |
| run timestamp | 2026-05-21 00:19:21 |
| elapsed | 33m 59s |
| seed | 42 |
| tracts | 20 |
| architectures | sage, gcn_gat |
| feature_standardization | per_tract (z-score) |

## Headline metrics

| metric | GRANITE-SAGE | GRANITE-GCNGAT |
|---|---|---|
| constraint error (mean) | 0.0000 | 0.0000 |
| spatial std (mean) | 0.0823 | 0.0814 |
| spatial std slope vs SVI | -0.01438 | -0.00644 |
| spatial std slope (baseline) | -0.01772 | -0.00798 |
| moran's I (mean) | 0.8776 | 0.8490 |
| block-group r (pooled) | 0.7537 | 0.7664 |

## Block-group validation

| method | pooled BG r |
|---|---|
| GRANITE-SAGE | 0.7537 |
| GRANITE-GCNGAT | 0.7664 |
| IDW | 0.7719 |
| Kriging | 0.7682 |

Note: IDW and kriging use no learned features and are unaffected by feature
standardization. Values are re-run on the same addresses for completeness.

## Artifacts

### Results
- `results/per_tract_metrics.csv`: one row per (tract, architecture)
- `results/aggregate_metrics.json`: mean/median per architecture
- `results/block_group_validation.json`: pooled BG Pearson r per method
- `results/delta_vs_baseline.json`: delta table vs 00_baseline
- `results/per_tract_scalers.npz`: per-tract mu/sigma for reproducibility
- `results/zero_var_columns.csv`: (tract, feature_idx) pairs where std was clamped
- `results/feature_importance/sage_importance.csv`
- `results/feature_importance/gcngat_importance.csv`

### Figures
- `figures/constraint_error_dist.png`
- `figures/spatial_std_by_svi.png`
- `figures/morans_i_by_tract.png`
- `figures/block_group_scatter.png`
- `figures/feature_importance_top20.png`
- `figures/architecture_overlap.png`
- `figures/comparison_spatial_std.png`: baseline vs per-tract overlay, slopes in legend
- `figures/comparison_morans_i.png`: side-by-side strip plot, same tract ordering

## Next step

Step 2b will swap normalization layers (LayerNorm -> BatchNorm or vice versa)
inside the GNN architecture while reverting feature standardization to global.

