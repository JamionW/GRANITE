# GRANITE Session Log

## 2026-04-27: Disk cleanup

**Files deleted:**
- `.venv/` (29 MB) — Python virtual environment removed to free disk space. Recreate with `pip install -e .`.
- `output/coord_artifact_test/`, `output/mehdi_review/`, `output/stage4_synthetic_eval/`, `output/architecture_comparison/`, `output/coord_artifact/`, `output/feature_importance/` (~133 MB total) — old experiment output directories, results already reviewed. `output/rank_consistency_run/` retained (active experiment).

**Cache invalidation:** none. Pipeline cache (`granite_cache/`) untouched.

---

## 2026-04-23: Coordinate-artifact experiment infrastructure

**Files changed:** `granite/disaggregation/pipeline.py`, `scripts/coord_artifact_experiment.py` (new), `scripts/coord_artifact_summary.py` (new)

**What changed and why:**
- `pipeline.py _apply_feature_mode`: new method substitutes the normalized feature matrix before graph construction. Supports four modes: `full` (pass-through), `coordinates_only` (z-scored lat/lon in cols 0-1, zeros elsewhere), `random_noise` (i.i.d. N(0,1) with fixed seed), `coords_plus_noise` (z-scored lat/lon + N(0,1) for remaining columns). Method is called after `normalize_accessibility_features` and reads `feature_mode` from config (default `'full'`, preserving baseline behavior).
- `pipeline.py _process_single_tract`: added call to `_apply_feature_mode` after normalization. Output shape (N, d) is unchanged so encoder architecture is unaffected.
- `scripts/coord_artifact_experiment.py`: runs the 4-condition x 5-tract design (Mehdi review tracts, 200 epochs, SAGE, fixed seed=42) and writes per-tract predictions and cross-condition metrics to `output/coord_artifact_test/`.
- `scripts/coord_artifact_summary.py`: reads experiment outputs and produces a 4-panel dashboard (spatial std, Moran's I, prediction-prediction correlation heatmap, constraint error by mode).

**Purpose:** tests whether GNN prediction quality is driven by coordinate information alone. If `coordinates_only` matches `full`, the 73-feature matrix adds no information beyond lat/lon; if `random_noise` matches, the model is fitting constraint correction only.

**Cache invalidation:** none. `feature_mode='full'` is the default and produces identical normalized features. Cache keys are unchanged.

## 2026-04-19: Stage 4 terminology fix and GIN integration audit

**Files changed:** `scripts/stage4_property_value_proxy_eval.py` (new), `graveyard/stage4_synthetic_eval.py.old` (retired), `gin_integration_audit.md` (new)

**What changed and why:**
- `scripts/stage4_synthetic_eval.py` renamed to `scripts/stage4_property_value_proxy_eval.py`. "Synthetic" was inaccurate: the evaluation target is log-transformed, min-max-normalized APPVALUE from the Hamilton County Assessor, not a generated signal. No features enter a generative function and no noise is injected.
- Module docstring updated to define the proxy transformation, document leakage audit (8 parcel-derived features excluded from feature set), rationale for proxy choice (escapes circularity), and add dasymetric citations (Mennis 2003; Maantay & Maroko 2009).
- OUTPUT_DIR updated from `./output/stage4_synthetic_eval` to `./output/stage4_property_value_proxy_eval`.
- Old file moved to `graveyard/` per convention.
- `gin_integration_audit.md` written at repo root: documents current GCN-GAT and GraphSAGE class structures, aggregation primitives, dispatch pattern (4 call sites in pipeline.py), and provides a diff-style plan for adding GIN (arch='gin') and standalone GCN (arch='gcn') as additional options. GINConv confirmed available. Estimated scope: ~245 lines across 2 files.

**Cache invalidation:** none. Rename does not affect pipeline caching logic.

## 2026-04-19: Fix post-training constraint correction with iterative bounded projection

**Files changed:** `granite/disaggregation/pipeline.py`, `config.yaml`

**What changed and why:**
- `config.yaml`: enabled `apply_post_correction: true` (was `false`). Without this, constraint errors were raw GNN training residuals with no post-hoc enforcement.
- `pipeline.py _finalize_predictions`: replaced single-pass additive shift + clip with iterative bounded projection (max_iter=50, tol=1e-8). Single-pass fails when clipping at 0 or 1 shifts the mean away from target; iterative loop re-applies the residual error until convergence.
- `pipeline.py` multi-tract holdout path (line ~4741): same fix applied to inline per-tract correction.
- `pipeline.py _apply_strong_constraint_correction`: same fix applied (was dead code but fixed for consistency).

**Verification (5 Mehdi tracts, 50 epochs each, cached):**

| FIPS        | SVI   | Iters | Residual  | Error  | Std    |
|-------------|-------|-------|-----------|--------|--------|
| 47065000700 | 0.114 | 2     | 7.45e-09  | 0.00%  | 0.0737 |
| 47065000600 | 0.224 | 3     | 0.00e+00  | 0.00%  | 0.0724 |
| 47065011326 | 0.510 | 1     | 0.00e+00  | 0.00%  | 0.1000 |
| 47065011321 | 0.696 | 2     | 0.00e+00  | 0.00%  | 0.0844 |
| 47065002400 | 0.891 | 4     | 0.00e+00  | 0.00%  | 0.1357 |

All constraint errors < 1e-6. Boundary tracts (0.114, 0.891) required 2-4 iterations due to clip-induced mean drift. Baseline methods (Dasymetric, Pycnophylactic) unaffected at machine epsilon.

**Cache invalidation:** none. Post-training correction only; no feature, routing, or cache key changes.

## 2026-04-19: Update visualization code for Dasymetric/Pycnophylactic baselines

**Files changed:** `granite/visualization/plots.py`, `granite/evaluation/post_training_validation.py`, `granite/scripts/run_granite.py`, `output/mehdi_review/_figures/README.md`

**What changed and why:**
- `plots.py`: replaced all IDW/Kriging/Naive_Uniform method references with Dasymetric/Pycnophylactic across dashboard panels (tradeoff scatter, variation bars, prediction distributions, prediction range, GNN-vs-baseline scatter, summary statistics, metrics table). Updated color scheme: dasymetric=#E65100 (warm orange), pycnophylactic=#1565C0 (cool blue). Removed overlap-handling code from tradeoff scatter (three distinct methods separate naturally).
- `post_training_validation.py`: updated color dicts, bootstrap difference test (now GRANITE vs Dasymetric), report text, summary checklist, docstrings, and help text to reference new baselines.
- `run_granite.py`: updated `--skip-baselines` help text.
- `README.md` in `_figures/`: updated dashboard description to reflect three-method display.

**Smoke test (tract 47065000600, 2089 addresses):**
- Dashboard: all three methods (GNN, Dasymetric, Pycnophylactic) appear in every panel.
- Metrics table: exactly three rows, no floor references.
- Tradeoff scatter: three distinct non-overlapping points.
- No KeyError, no missing-method warnings, no dangling IDW/Kriging strings in output artifacts.

**Cache invalidation:** none. No feature, routing, or pipeline logic changed.

## 2026-04-18: Retire IDW/Kriging baselines, add Dasymetric and Pycnophylactic

**Files changed:** `granite/evaluation/baselines.py`, `granite/disaggregation/pipeline.py`, `granite/evaluation/post_training_validation.py`, `graveyard/disaggregation_baselines_idw_kriging.py`

**What changed and why:**
- IDW and Kriging baselines retired to `graveyard/disaggregation_baselines_idw_kriging.py`. Point-interpolation methods collapse to tract mean under single-centroid interpolation; replaced by mass-preserving disaggregation baselines.
- `baselines.py`: added `DasymetricDisaggregation` (additive dasymetric using `nlcd_impervious_pct` ancillary) and `PycnophylacticDisaggregation` (Tobler 1979 iterative smoothing seeded from multi-tract gradient). Both satisfy `disaggregate()` interface. `DisaggregationComparison.run_comparison()` now passes `address_gdf` through to `disaggregate()` for ancillary column access. Base class `disaggregate()` signature extended with optional `address_gdf` parameter.
- `pipeline.py`: all IDW/Kriging imports, constructor calls, result key references (`IDW_p2.0`, `Kriging`), visualization labels, and block group validation collection updated to use `Dasymetric` and `Pycnophylactic`.
- `post_training_validation.py`: IDW/Kriging usage replaced with new baselines.

**Sanity check (tract 47065000600, 2089 addresses):**
- Dasymetric: constraint error 2.78e-17, spatial std 0.109, range [0.0, 0.47]
- Pycnophylactic: constraint error 2.78e-17, spatial std 0.065, range [0.10, 0.36]
- Both strictly in [0,1], all validation criteria pass.

**Cache invalidation:** none. No feature or routing logic changed.

## 2026-04-18: Add property_value as alternate disaggregation target

**Files changed:** `granite/data/loaders.py`, `granite/disaggregation/pipeline.py`, `granite/scripts/run_granite.py`, `config.yaml`

**What changed and why:**
- `loaders.py`: added `load_property_value_data()`, `get_tract_target_value()`, `get_address_truth_values()` to DataLoader. Property values loaded from `combined_address_features.csv`, normalized to [0,1] via county-wide min-max of log_appvalue. `hash` column now retained through address loading for property value joins.
- `pipeline.py _process_single_tract`: reads `config.data.target` (default 'svi') to select target. Uses `get_tract_target_value()` instead of reading RPL_THEMES directly. Logs active feature count. Attaches address-level truth vector to results. Skips IDW/Kriging baselines for non-SVI targets. Skips block group SVI constraints and pairwise ordering for property_value target.
- `pipeline.py _extract_building_features`: excludes `log_appvalue` and `build_to_land_ratio` when target=property_value to prevent target leakage (class attribute `_PROPERTY_VALUE_EXCLUDED_FEATURES`).
- `pipeline.py save_results`: writes `address_truth.csv` (predictions + truth) when truth vector is present; includes `target` field in `results_summary.json`.
- `run_granite.py`: added `--target` CLI argument (choices: svi, property_value).
- `config.yaml`: added `target: "svi"` under `data:` section.

**Verified:**
- `--target=svi` reproduces baseline behavior: 73 features, same constraint error and spatial std.
- `--target=property_value` runs to completion: 71 features (2 excluded), constraint satisfied, truth vector written to disk.

**Cache invalidation:** none. Accessibility caches store base+modal features which are identical across targets. Building features (where exclusion occurs) are recomputed fresh every run.

---

## 2026-04-17: Dashboard redesign and tract_fips bugfix

**Files changed:** `granite/visualization/plots.py`, `granite/disaggregation/pipeline.py`

**What changed and why:**
- `_plot_constraint_satisfaction` replaced with `_plot_tradeoff_scatter`: X-axis constraint error %, Y-axis spatial std. Shows each method as a labeled point -- GNN trades constraint precision for within-tract differentiation.
- `_plot_variation_comparison`, `_plot_prediction_distributions`, `_plot_prediction_range`: filtered to GNN and IDW_p2.0 only. Naive_Uniform, IDW_p3.0, Kriging are zero-valued in these panels and add no information. They still appear in the metrics table as floor references.
- `plot_spatial_analysis`: "Tract ?" fallback replaced -- title now omits tract reference when FIPS unavailable instead of showing a question mark.
- `pipeline.py _process_single_tract`: added `tract_fips` and `tract_svi` as top-level keys in return dict.
- `pipeline.py _create_research_visualizations` call site: added `tract_fips` to viz_data dict.
- `pipeline.py save_results` viz_data: added `tract_fips` via `results.get('tract_fips')`.
- Regenerated all figures for 5 tracts with SAGE architecture; collated into `output/mehdi_review/_figures/`.

**Cache invalidation:** none. Visualization-only and caller-side changes.

---

## 2026-04-16: Remove accessibility-centric framing from visualizations

**Files changed:** `granite/visualization/plots.py`, `granite/evaluation/spatial_diagnostics.py`

**What changed and why:**
- `plot_spatial_analysis`: replaced learned accessibility map and access-vulnerability scatter with within-tract deviation map (coolwarm) and prediction distribution histogram. title now shows tract FIPS, SVI, and address count.
- `_plot_accessibility_correlations` renamed to `_plot_prediction_range`: horizontal bar chart of (max - min) per method replaces accessibility correlation bars.
- `_plot_metrics_table`: "Access r" column replaced with "Moran's I" (displays n/a when not available in comparison_results).
- `_plot_spatial_patterns` (spatial_diagnostics.py): axes[1,0] replaced accessibility scatter with deviation-from-tract-mean geographic scatter.
- `_plot_accessibility_relationships` (spatial_diagnostics.py): short-circuited with deprecation warning; body retained for backward compatibility.
- `plot_accessibility_learning_validation`: marked deprecated in docstring; method body unchanged.
- regenerated all figures for 5 tracts (47065000700, 47065000600, 47065011326, 47065011321, 47065002400) with SAGE architecture; collated into `_figures/`.

**Cache invalidation:** none. visualization-only changes.

## 2026-04-14: National block group SVI data acquisition and validation

**Files changed:** `granite/data/block_group_loader.py`, `CLAUDE.md`

### Census API URL fix

`fetch_national_acs_data()` had `&in=state:{fips}&in=county:*` (two separate `in` params), which the Census API rejects for block group queries. Fixed to `&in=state:{fips}%20county:*` (space-separated single param), matching the working county-level fetch format.

### National data acquired

- Fetched ACS block group data for all 52 states/territories (242,335 block groups)
- Computed nationally-ranked SVI (239,346 with complete indicators)
- Cached to `data/processed/national_bg_acs_raw.csv` (63MB) and `data/processed/national_bg_svi.csv` (116MB)

### National ranking consistency results for tract 47065000600

- BG weighted mean SVI: 0.3046 (vs tract CDC SVI: 0.2235)
- Rescaling shift: -0.0811 (vs -0.1806 with county ranking, 55% reduction)
- BG1 nationally-ranked SVI: 0.1314 (above 0.05 floor, clipping problem resolved)
- All three BGs shifted downward under national ranking (county ranking inflated values because Hamilton County is low-vulnerability nationally)

### No cache invalidation

Training-only and data acquisition changes. No feature or routing modifications. Cache keys unchanged.

---

## 2026-04-13: Block group SVI rescaling for constraint consistency

**Files changed:** `granite/data/block_group_loader.py`, `granite/disaggregation/pipeline.py`, `scripts/bg_rescaled_convergence_experiment.py` (new), `scripts/diagnose_bg_consistency.py` (new)

### Diagnosis

Block group SVIs (ACS-derived, percentile-ranked within Hamilton County BGs) do not aggregate to the CDC tract SVI (national percentile ranks). For tract 47065000600: BG weighted mean = 0.4041, tract SVI = 0.2235, discrepancy = 80.8%. This explains why the prior convergence experiment saw tract constraint error blow up to 25-28% when BG constraints were added -- the optimizer had no feasible solution.

### Rescaling function

Added `rescale_block_group_svis()` to `block_group_loader.py`. Pure function: additive shift + clip to [0,1], iterates up to 10 times to handle clipping distortion. Converges within 0.001 of target weighted mean. For this tract: shift = -0.1806, BG1 clips to 0.0, BG2 = 0.290, BG3 = 0.370.

### Pipeline integration

Rescaling wired into `_train_multi_tract_gnn` in `pipeline.py`, gated by `rescale_bg_svi` config key (default True). Rescales per-tract: groups BGs by parent tract and rescales each group against its tract SVI.

### Rescaled convergence experiment results

Same 2x2 design: {GCN, SAGE} x {tract-only, tract+rescaled BG}. Key results:
- Tract error with BG constraints: 5.01% (GCN), 1.87% (SAGE) -- down from ~25-28% unrescaled
- BG error: 6.04% (GCN), 3.89% (SAGE) -- down from ~41-43% unrescaled
- GCN vs SAGE r: 0.693 (rescaled BG) vs 0.420 (tract-only) -- convergence preserved
- Prior unrescaled BG r was 0.788, but at cost of 25-28% tract error; rescaled achieves 0.693 with 3.4% avg tract error

Interpretation: rescaling resolved constraint tension while preserving convergence benefit.

### No cache invalidation

Training-only changes. No feature or routing modifications. Cache keys unchanged.

---

## 2026-04-12: Wire block group constraints into training

**Files changed:** `granite/models/gnn.py`, `granite/disaggregation/pipeline.py`, `scripts/setup_data.sh` (new)

### Block group mean constraints as additional training loss

- `MultiTractGNNTrainer.__init__`: added `bg_constraint_weight` attribute (default 1.0, configurable via config)
- `_compute_multi_tract_losses`: new optional `block_group_targets`/`block_group_masks` params; computes per-BG MSE constraint loss averaged across all block groups, returned as `bg_constraint` key in losses dict
- `train()`: accepts optional BG params, converts to tensors, passes to loss function, tracks `bg_constraint_errors` in training_history, reports BG constraint error at epoch intervals
- `_train_multi_tract_gnn` in pipeline: loads BG data via `BlockGroupLoader`, spatial-joins addresses, creates masks/targets filtered to `svi_complete==True`, >= 5 addresses, and nesting within training tracts; prints summary at training start
- Post-training diagnostic: reports per-BG residual mean error after tract correction (no BG correction applied)
- Backward compatible: all BG params default to `None`; behavior identical to prior code when not provided
- No cache invalidation: changes are training-only, no feature or routing changes

## 2026-04-12: Trim visualization output to 3 retained plots

**Files changed:** `granite/visualization/plots.py`, `granite/disaggregation/pipeline.py`, `granite/evaluation/accessibility_validator.py`, `granite/evaluation/feature_importance.py`, `granite/scripts/run_granite.py`

### Diagnostic plots gated behind --diagnostics flag

- 7 diagnostic plots (stage1_validation, stage2_validation, statistical_summary, and 4 accessibility_validation plots) now only generate when `--diagnostics` is passed
- `create_comprehensive_research_analysis` takes `diagnostics=False` parameter; diagnostic plots go to `output/visualizations/` when enabled
- `validate_granite_accessibility_features` takes `diagnostics=False` parameter; plots go to `output/accessibility_validation/` when enabled
- Code for generating diagnostic plots is preserved, not deleted

### Feature importance label bug fixed

- Cumulative importance annotations now handle the case where cumulative never reaches 80%/90% thresholds (falls back to total feature count)
- Fixed pluralization: "1 feature explains" vs "N features explain"

### spatial_analysis.png consolidated to 3-panel layout

- Changed from 2x2 (with text stats panel) to 1x3: SVI predictions, learned accessibility, access-vulnerability scatter
- SVI predictions panel now uses fixed 0-1 colorscale (RdYlGn_r) for cross-tract comparability
- Removed "Moderate Equity Pattern" text box from scatter panel; r value shown in title only

### plot_spatial_disaggregation updated

- Added `tract_results` parameter (backward-compatible with existing `multi_tract_data`)
- Auto-scale point size: s=30 (<5000 addresses), s=10 (<20000), s=3 (>=20000)
- Fixed TIGER shapefile path (was looking in nonexistent subdirectory)
- Single-tract mode now also uses fixed 0-1 SVI colorscale

### Output directory cleanup

- Default run produces: spatial_analysis.png, disaggregation_comparison.png, feature_importance/, CSVs, and results_summary.json
- output/visualizations/ and output/accessibility_validation/ only created with --diagnostics

### No cache invalidation

- No feature or routing changes. Cache keys unchanged.

---

## 2026-04-11: Visualization consolidation and multi-tract heatmap

**Files changed:** `granite/visualization/plots.py`, `granite/visualization/disaggregation_plots.py` (removed), `granite/disaggregation/pipeline.py`, `granite/scripts/run_granite.py`, `granite/scripts/run_holdout_validation.py`

### Consolidation

- `DisaggregationVisualizer` was duplicated identically in `plots.py` and `disaggregation_plots.py`. Removed `disaggregation_plots.py`, moved to `graveyard/disaggregation_plots_v2.old`.
- Updated imports in `pipeline.py` and `run_holdout_validation.py` to point to `granite.visualization.plots`.

### Multi-tract heatmap

- Extended `plot_spatial_disaggregation` to support multi-tract mode via `multi_tract_data` dict parameter. Single-tract mode unchanged.
- Multi-tract mode: unified SVI colorscale (0-1, RdYlGn_r), deviation panel per address relative to its own tract, tract boundary overlay from TIGER shapefiles, auto-scaled point size for large address counts.
- Pipeline now returns `address_gdf` in single-tract results dict.
- `run_multi_fips_experiment` collects per-tract GDFs and predictions, generates multi-tract heatmap automatically when 2+ tracts succeed.

### No cache invalidation

- No feature or routing changes. Cache keys unchanged.

---

## 2026-04-11: Bug fixes, modal feature refactor, cache improvements

**Files changed:** `granite/data/loaders.py`, `granite/models/gnn.py`, `granite/cache.py`, `granite/disaggregation/pipeline.py`, `granite/evaluation/spatial_diagnostics.py`, `granite/evaluation/accessibility_validator.py`, `granite/evaluation/morans_i_analysis.py`, `granite/evaluation/post_training_validation.py`, `granite/models/mixture_of_experts.py`, `granite/validation/block_group_validation.py`, `granite/features/modal_accessibility.py`, `README.md`, `docs/FEATURES.md`

### Bug fixes (12 bugs)

- **#7 CRS mismatch in spatial join** (`loaders.py`): Added CRS check before `within()` in `get_addresses_for_tract`
- **#10 Hardcoded county name** (`loaders.py`): `_get_county_name` now falls back to SVI CSV lookup; `load_svi_data` uses case-insensitive match and raises `ValueError` on empty result
- **#4 GPU device mismatch** (`gnn.py`, 8 locations): All CPU tensors now use `.to(device)` matching graph data device
- **#5 No cache invalidation** (`cache.py`): Added version tag (auto-invalidates on change), `invalidate()` method with optional TTL, and `_invalidate_all()`
- **#6 Bootstrap p-value**: Already fixed in previous session (verified)
- **#8 Single-county assumption** (`pipeline.py`): Global training now derives county from most common tract FIPS, warns if mixed
- **#9 MoE overlap documentation** (`mixture_of_experts.py`): Added comment documenting deliberate soft-boundary design
- **#11 FIPS strip inconsistency** (`loaders.py`): Added `str().strip()` at tract_fips assignment
- **#12 Zero-variance logging** (`pipeline.py`): Log level now INFO when near expected count (24 constants), WARN only when unexpectedly high
- **#13 Quality score inversion** (`spatial_diagnostics.py`): Changed to `1 - (bad/total)` so higher = better
- **#14 Silent address drop** (`pipeline.py`): Logs unmatched address count after spatial join
- **#15 Low BG threshold** (`block_group_validation.py`): Raised minimum from 5 to 10 addresses per block group

### Validator bug fix

- **NameError in accessibility_validator.py**: `count_5` referenced but never defined at lines 540-541 and 561-562; replaced with `count_10min` which was the intended variable. Validator now runs all 7 steps.

### Modal features refactored to per-address computation

- **Before:** 5 features per destination type computed at tract level from vehicle ownership rates and base features. All addresses in a tract received identical values (zero within-tract variance).
- **After:** Computed per address from OSRM driving and walking travel times. New semantics: avg_time (mean of drive/walk nearest), time_std (mode disparity), access_density (union of destinations reachable in 10min by either mode), equity_gap (|walk - drive| nearest), car_advantage (walk/drive ratio nearest).
- Pipeline extracts per-address travel time summaries via `_summarize_travel_times()` during base feature computation and caches them alongside per-destination features.
- Fallback to tract-level approximation when per-address times unavailable (partial cache from older runs).
- Output shape unchanged: (n_addresses, 15). Feature names unchanged for downstream compatibility.
- `employment_walk_effective_access` now ranks #3 in feature importance (was negligible as tract constant).

### Cache key changes

- Complete feature cache key changed from `_base_modal` to `_base_modal_v2` to invalidate stale entries with old modal features.
- New `modal_times` cache entries store per-address travel time summaries per destination type.
- Cache now stores base+modal only (not socioeco/building); those are recomputed fresh on cache hit.

### Documentation updates

- README and FEATURES.md updated to reflect per-address modal features, reduced zero-variance count (11 vs 24), cache invalidation API, and corrected feature count.

---

## 2026-04-09: Baseline and evaluation bug fixes

**Files changed:** `granite/evaluation/baselines.py`, `granite/evaluation/bootstrap_confidence_intervals.py`, `granite/evaluation/post_training_validation.py`

### Fix #1: IDW baseline was not excluding target tract from neighbors
- `baselines.py` IDWDisaggregation.disaggregate(): `target_idx` was computed but never used to filter KD-tree query results
- The target tract's own centroid was included as a neighbor, biasing IDW predictions toward the known tract mean
- Fix: filter `target_idx` out of each address's neighbor set after query, trim to `n_neighbors`
- Impact: IDW results will now reflect true interpolation from neighboring tracts only
- No cache invalidation needed (baselines don't use cache)

### Fix #6: Bootstrap p-value was one-tailed but labeled as p-value (implying two-tailed)
- `bootstrap_confidence_intervals.py` bootstrap_correlation_difference(): computed both two-tailed and one-tailed p-values, but returned the one-tailed value under the key 'p_value'
- `post_training_validation.py` line 407: same issue, used `np.mean(diff_dist <= 0)` which is one-tailed
- Fix: replaced both with proper two-tailed bootstrap pivotal test: shift bootstrap distribution to null (diff=0), then `np.mean(np.abs(boot_diffs_null) >= np.abs(diff_observed))`
- CI-based significance test was already correct; only the reported p-value number was wrong
- No cache invalidation needed

### Fix #3: IDW used raw lat/lon distances while Kriging used meters
- `baselines.py` IDWDisaggregation: KD-tree was built on raw lon/lat coordinates, meaning E-W distances were ~17% distorted at lat ~35 (Chattanooga)
- Kriging already converted to approximate meters with proper per-axis scaling
- Fix: IDW now converts centroids to meters in fit() and address coords in disaggregate(), using same scaling factors as Kriging (111320*cos(lat) for lon, 110540 for lat)
- Impact: IDW and Kriging now use consistent isotropic distance metrics
- No cache invalidation needed

### Fix #2: IDW and Kriging double-clipping broke aggregate constraint
- `baselines.py` both IDWDisaggregation and OrdinaryKrigingDisaggregation: scale-then-clip-then-shift-then-clip could leave the final mean != tract_svi
- Added shared `_enforce_constraint()` function that iteratively shifts predictions to target mean, clips to [0,1], and repeats until convergence (max 20 iterations, tol 1e-6)
- Both baselines now use this shared function
- Impact: baseline constraint satisfaction will now match the GNN's hard constraint, making comparison fair
- No cache invalidation needed

## 2026-04-16: Dr Mehdi review run -- five tracts spanning SVI spectrum

**Files changed:** none (pipeline run only)

### What was done

Ran `granite --architecture sage` sequentially on five Hamilton County tracts for Dr Mehdi's review. No code changes were required; all five tracts ran cleanly.

Outputs saved to `output/mehdi_review/<FIPS>/` with figures collated and renamed in `output/mehdi_review/_figures/`.

### Tracts and results

| FIPS        | SVI   | n_addresses | constraint_err% | spatial_std | morans_i |
|-------------|-------|-------------|-----------------|-------------|----------|
| 47065000700 | 0.114 | 1,784       | 8.75            | 0.0390      | 0.7774   |
| 47065000600 | 0.224 | 2,089       | 13.16           | 0.0771      | 0.9415   |
| 47065011326 | 0.510 | 2,738       | 0.80            | 0.1024      | 0.9858   |
| 47065011321 | 0.696 | 3,756       | 1.97            | 0.0997      | 0.9333   |
| 47065002400 | 0.891 | 1,918       | 2.68            | 0.0769      | 0.9392   |

### Cache notes

- 47065000600 was a full cache hit (62s); all others required OSRM modal feature recomputation (~1-2 hrs each).
- Feature count varies 71-73 across tracts due to zero-variance building features; existing behavior, no impact on comparability.
- No cache keys were invalidated.

## 2026-04-25 - Full rank-consistency experiment (8 tracts × 2 architectures)

**Files changed:**
- `scripts/run_rank_consistency_experiment.py` (new) - driver script running GRANITEPipeline for both GraphSAGE (`arch=sage`) and GCN-GAT (`arch=gcn_gat`) across all 8 inventory tracts; outputs to `output/rank_consistency_run/{graphsage,gcn_gat}/tract_{fips}/`

**Outputs generated:**
- `output/rank_consistency_run/graphsage/tract_*/` - 8 tracts, 73 named feature cols, raw_prediction present
- `output/rank_consistency_run/gcn_gat/tract_*/` - 8 tracts, 73 named feature cols, raw_prediction present
- `results/rank_consistency_full/summary.txt`, `per_tract_rho.csv`, `feature_summary.csv`

**Experiment parameters:** seed=42, epochs=200, apply_post_correction=True, 8 tracts spanning SVI 0.04-0.89, cv_threshold=0.10, min_tracts=5, min_addresses=50

**Key result:**
- Section A (SAGE only):    0
- Section B (GCN-GAT only): 2 features (healthcare_modal_access_gap, healthcare_car_effective_access; median_rho=0.18, n_tracts=6)
- Section C (both, same sign): 0
- Section D (sign-flippers): 0

**Interpretation:** No feature is rank-consistent under both architectures. Two healthcare modal features survive under GCN-GAT alone with a small positive correlation (rho~0.18), but no features are architecture-agnostic. The zero Section C/D counts mean there is no cross-architecture signal to report as a positive finding.

**Cache notes:** Runs used existing OSRM cache; no cache invalidation.
