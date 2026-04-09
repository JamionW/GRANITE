# GRANITE Session Log

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
