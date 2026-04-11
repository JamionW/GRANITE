# GRANITE Session Log

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
