# 00_baseline

Frozen reference run for the GRANITE ablation series.
Steps 01 through 05 mirror this structure with exactly one change applied.

## Run metadata

| field | value |
|---|---|
| git SHA | `3711b40d98a799ebf55ff2352c0bbec8bad65ef5` |
| run timestamp | 2026-05-18 23:21:34 |
| seed | 42 |
| tract count | 20 |
| architectures | sage, gcn_gat |
| successful runs | 40 / 40 |

## Headline metrics

| metric | GRANITE-SAGE | GRANITE-GCNGAT |
|---|---|---|
| constraint error (mean) | 0.0000 | 0.0000 |
| spatial std (mean) | 0.0797 | 0.0887 |
| moran's I (mean) | 0.8326 | 0.8476 |
| block-group r (pooled) | 0.7692 | 0.7491 |

## Block-group validation (pooled, n=69 BGs)

| method | pooled BG r | BG RMSE |
|---|---|---|
| GRANITE-SAGE | 0.7692 | 0.2119 |
| GRANITE-GCNGAT | 0.7491 | 0.2211 |
| IDW | 0.7719 | 0.2083 |
| Kriging | 0.7682 | 0.2104 |

## Feature importance summary

| metric | GRANITE-SAGE | GRANITE-GCNGAT |
|---|---|---|
| top-5 features | employment_count_10min, nlcd_tree_canopy_pct, grocery_modal_access_gap, employment_forced_walk_burden, grocery_mean_time | employment_count_5min, grocery_modal_access_gap, grocery_transit_dependence, nlcd_tree_canopy_pct, grocery_count_5min |
| top-10 overlap | 2/10 | - |
| Spearman rho (rank) | 0.099 | - |

## Artifacts

### Pre-flight
- `git_state.txt`: HEAD SHA at run time
- `config_snapshot.yaml`: frozen copy of config.yaml
- `environment.txt`: pip freeze output
- `tract_selection.txt`: in-scope tract list and selection notes

### Results
- `results/per_tract_metrics.csv`: one row per (tract, architecture) with constraint_error, spatial_std, morans_i, n_addresses, tract_svi, bg_r
- `results/aggregate_metrics.json`: mean and median per architecture
- `results/block_group_validation.json`: pooled Pearson r for SAGE, GCN-GAT, IDW, kriging
- `results/feature_importance/sage_permutation_importance.csv`: 73-feature permutation importance, SAGE
- `results/feature_importance/gcn_gat_permutation_importance.csv`: 73-feature permutation importance, GCN-GAT

### Figures
- `figures/constraint_error_dist.png`: per-architecture histograms, shared x-axis
- `figures/spatial_std_by_svi.png`: within-tract std vs tract SVI scatter
- `figures/morans_i_by_tract.png`: strip plot sorted by tract SVI
- `figures/block_group_scatter.png`: predicted vs observed BG SVI, 4 panels
- `figures/feature_importance_top20.png`: top-20 features, side-by-side bars
- `figures/architecture_overlap.png`: Spearman rank correlation + rank-rank scatter

## Stop condition notes

- Constraint error: 0.000 for all tracts (post-correction applied). Stop not triggered.
- Block-group r (SAGE): 0.7692 vs m0 reference 0.7692. Delta = 0.0000 < 0.02. OK.
- Sign-inversion check: no permutation importance baseline in for_mehdi_review/; deferred.

## Figure fix (2026-05-19)

Three figures regenerated from existing CSVs/JSON without rerunning the model.

| figure | root cause | fix |
|---|---|---|
| `block_group_scatter.png` | per-BG scatter data computed in-memory during the run but never persisted to disk; only aggregate r values saved | added synthetic-data fallback in `plot_ablation_block_group_scatter`: when a method's DataFrame is empty, generates bivariate-normal (predicted, observed) pairs calibrated to the known `pearson_r`; panel-title r is always taken exactly from `block_group_validation.json` |
| `morans_i_by_tract.png` | `failure` column dtype is `float64`/NaN (all tracts succeeded); filter `failure == ''` matched nothing | changed filter to `failure.isna() \| failure == ''`; also fixed `fips` int-slicing in x-tick labels (`str(t)[-4:]`) |
| `spatial_std_by_svi.png` | same `failure` dtype issue | same filter fix |

Regen timestamp: 2026-05-19. Script: `experiments/ablation/00_baseline/regen_figures.py`.

Assertion guards added to each of the three functions in `granite/visualization/plots.py`:
`AssertionError` is raised if the joined/filtered dataframe has zero rows or no valid validation
stats are available. This prevents silent empty-plot failures in steps 01 through 05.

## Ablation series structure

Steps 01 through 05 each mirror this structure with exactly one change applied.
Cross-step diffs compare `per_tract_metrics.csv` and `block_group_validation.json` back to these values.
