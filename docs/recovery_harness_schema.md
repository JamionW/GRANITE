# Recovery Harness Schema (M1)

`granite/disaggregation/recovery_harness.py` implements the held-out feature
recovery experiment. It drops one address-level feature from the input matrix,
replaces the per-tract SVI constraint with the per-tract mean of that feature,
trains the GNN, and writes three output files.

---

## CLI invocation

```
granite --recover-feature <FEATURE_NAME> --fips <FIPS> [--architecture sage|gcn_gat]
        [--neighbor-tracts N] [--epochs N] [--seed N] [--output DIR]
```

`--recover-feature` requires `--fips` and is incompatible with `--global-training`.
When `--output` is omitted the default is
`./output/recovery/<feature>_<arch>_<fips>_<seed>/`.

---

## Standardization step

Because the GNN's prediction head uses a sigmoid bounded in [0, 1], raw feature
values whose range differs significantly from [0, 1] would push the constraint
solver outside its working regime. Standardization maps each feature to
approximately zero mean and unit variance, so the per-tract means handed to the
trainer stay in a comparable scale.

Procedure (when `standardize_target: true`, the default):

1. Collect the raw pre-normalization values of `target_feature` across **all**
   addresses in the run (target tract + neighbor tracts).
2. Compute `mu = mean(values)` and `sigma = std(values)` across those addresses.
   If `sigma < 1e-8` it is clamped to 1.0 to avoid division by zero.
3. `target_values_std = (raw_values - mu) / sigma`
4. Per-tract constraint: `tract_target_mean[fips] = mean(target_values_std[mask_fips])`
5. The trainer receives `tract_svis = tract_target_means` in standardized units.
6. After training, predictions are de-standardized:
   `prediction_destandardized = prediction_standardized * sigma + mu`

`mu` and `sigma` are recorded in `run_meta.json` so predictions can always be
converted between scales.

---

## Output files

All three files are written to the run output directory.

### predictions.csv

One row per address in the training set (all tracts, including neighbors).

| Column | Type | Description |
|---|---|---|
| `tract_fips` | str | 11-digit census tract FIPS |
| `address_id` | int | Address identifier; matches `address_id` column if present in the address GeoDataFrame, otherwise the concatenated DataFrame index |
| `target_feature` | str | Name of the held-out feature (constant across the file) |
| `architecture` | str | GNN architecture used (`sage` or `gcn_gat`) |
| `seed` | int | Random seed |
| `prediction_standardized` | float | Raw GNN sigmoid output in standardized units |
| `prediction_destandardized` | float | Prediction converted back to native feature units via `prediction_standardized * sigma + mu` |
| `true_value` | float | Held-out true value of `target_feature` for this address in native units (pre-normalization value from the computed feature matrix) |
| `tract_target_mean` | float | Per-tract constraint value handed to the trainer, expressed in **native units** (`tract_target_means_std[fips] * sigma + mu`) |

### per_tract_metrics.csv

One row per census tract in the training set.

| Column | Type | Description |
|---|---|---|
| `tract_fips` | str | 11-digit census tract FIPS |
| `n_addresses` | int | Number of addresses in this tract |
| `pearson_r` | float | Pearson correlation between `prediction_destandardized` and `true_value` at address level within this tract |
| `spearman_rho` | float | Spearman rank correlation, same scope |
| `rmse` | float | Root mean squared error in native feature units |
| `constraint_error_pct` | float | `|predicted_tract_mean - tract_target_mean| / |tract_target_mean| * 100`. `predicted_tract_mean` is the mean of `prediction_destandardized` across the tract. `tract_target_mean` is the constraint value in native units. |

All correlation and RMSE values are computed over addresses where both
`prediction_destandardized` and `true_value` are finite. `nan` is written
when fewer than two finite pairs exist.

### run_meta.json

Captures configuration and standardization parameters for reproducibility.

| Key | Description |
|---|---|
| `target_feature` | Name of the held-out feature |
| `architecture` | GNN architecture |
| `seed` | Random seed |
| `n_features_after_drop` | Number of input features after removing `target_feature` |
| `standardize_target` | Whether standardization was applied |
| `target_mean_native` | `mu` used for standardization (in native feature units) |
| `target_std_native` | `sigma` used for standardization (in native feature units) |
| `git_sha` | Short git SHA at run time, or null if unavailable |
| `config` | Full config dict passed to the harness |

---

## Relationship to the SVI pipeline

The harness reuses the following pipeline methods without modification:

- `GRANITEPipeline._load_spatial_data()`
- `GRANITEPipeline._compute_accessibility_features()`
- `GRANITEPipeline._generate_feature_names()`
- `GRANITEPipeline._apply_feature_mode()`
- `DataLoader.create_context_features_for_addresses()`
- `DataLoader.normalize_context_features()`
- `DataLoader.create_spatial_accessibility_graph()`
- `MultiTractGNNTrainer.train()`

The trainer receives `tract_svis = tract_target_means` (feature means) in place
of the usual SVI dict. The trainer is unaware of the substitution; the constraint
math is identical. Block-group constraints and ordering constraints are
explicitly set to weight 0 for M1 via `bg_constraint_weight=0.0` and
`ordering_weight=0.0` in the trainer config.

The default SVI pipeline path (`granite --fips ...` without `--recover-feature`)
is not modified.

---

## Accepted feature names (M1)

Any name present in the computed feature matrix for the given tract, which
includes:

- Accessibility features: `employment_min_time`, `healthcare_count_10min`, etc.
- Modal features: `employment_transit_dependence`, etc.
- Socioeconomic controls: `pct_no_vehicle`, `pct_poverty`, etc.
- Building / parcel features: `log_appvalue`, `log_bldg_footprint_m2`,
  `bldg_vertex_count`, `in_sfha`, `is_residential`, `build_to_land_ratio`,
  `log_acres`, `lucode_*`, `proptype_*`
- NLCD features: `nlcd_land_cover`, `nlcd_impervious_pct`, `nlcd_tree_canopy_pct`

The harness validates the name against the computed feature matrix and fails
fast with an informative error listing available names if it is not found.
