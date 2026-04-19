# Session 2 Sanity Check: Property Value Pipeline

Tract: 47065000600 | 2,089 addresses | Single-tract run, 50 epochs
Output: `output/address_truth.csv` (columns: address_id, x, y, granite, truth, idw, kriging)

---

## Check 1: Unit Consistency

### Summary statistics

| Column  |   min    |   max    |   mean   |   std    |
|---------|----------|----------|----------|----------|
| granite | 0.293710 | 0.876975 | 0.661668 | 0.109293 |
| truth   | 0.000000 | 0.816191 | 0.652365 | 0.044572 |
| idw     | 0.621052 | 0.672438 | 0.652365 | 0.011021 |
| kriging | 0.627025 | 0.678364 | 0.652365 | 0.010024 |

Normalized tract mean (constraint target): **0.652365**

County-wide min-max normalization: log_appvalue range [0.0000, 20.5224], mapped to [0, 1].

### Sub-item verdicts

| Check | Result |
|-------|--------|
| All columns in [0, 1] | **PASS** -- no values outside bounds beyond floating-point tolerance |
| Truth mean matches normalized tract mean | **PASS** -- truth mean = 0.652365, identical to tract target |
| GRANITE mean within reported constraint error (~1.43%) | **PASS** -- GRANITE mean = 0.661668, deviation = 1.43% |
| IDW mean matches tract mean (within 1e-6) | **PASS** -- deviation = 0.00e+00 |
| Kriging mean matches tract mean (within 1e-6) | **PASS** -- deviation = 0.00e+00 |

### Pairwise Pearson correlations (truth vs method)

| Method  |   r    |    p     |
|---------|--------|----------|
| granite | 0.0621 | 4.51e-03 |
| idw     | 0.0853 | 9.42e-05 |
| kriging | 0.1599 | 1.98e-13 |

All correlations are positive but weak, consistent with the known difficulty of within-tract disaggregation for this target.

---

## Check 2: Spatial Basis

### GRANITE

The graph is constructed in `DataLoader._create_road_network_graph` (loaders.py:951). It combines two edge types:

- **Network edges** (`_create_road_connectivity` + `_extract_network_edges`): addresses are snapped to the nearest road-network node (within 500 m via BallTree/haversine). Pairs of road-connected addresses within 1,000 m Euclidean and 1,500 m road-network shortest-path distance are connected. Edge weight formula: `1.0 / (1.0 + path_length / 500.0)`.
- **Geographic edges** (`_create_geographic_edges`): Euclidean k-NN (k=6) on raw lon/lat coordinates, filtered to pairs within 1,000 m. Edge weight formula: `exp(-distance_m / 300.0)`.

Both edge sets are deduplicated and stored as bidirectional edges. Nodes are individual addresses. Edge counts are logged at runtime (exact counts depend on road coverage for this tract).

### IDW

- **Distance metric**: approximate meters. Raw lon/lat centroids are converted to meters using `111320 * cos(lat_center)` (lon) and `110540` (lat), then used in a `cKDTree`. This is a projected Euclidean distance, not road-network or travel-time distance.
- **Input granularity**: interpolation from **tract centroids**, not address-level points. Each tract centroid carries its mean normalized property value (`pv_mean` column, computed in `_run_property_value_baselines`). The target tract is excluded from its own neighbor set.
- **Parameters**: `power=2.0`, `k_neighbors=8`.
- **Features used**: none. IDW is purely proximity-based on per-tract mean property values. The 58 address-level features do not feed into IDW predictions.

### Kriging

- **Distance metric**: same approximate-meter projection as IDW (lon/lat scaled by `111320*cos(lat)` and `110540`).
- **Input granularity**: **tract centroids**, same as IDW. The `pv_mean` column is used via `svi_column='pv_mean'`.
- **Variogram**: exponential model with parameters `range=5000 m`, `sill=0.1`, `nugget=0.01`. Covariance = `sill - (nugget + sill * (1 - exp(-3h/range)))`.
- **Features used**: none. Kriging is purely spatial; the 58 address-level features do not participate.

---

## Check 3: Constraint Satisfaction Mechanism

### GRANITE

Constraint satisfaction is achieved through **learned loss minimization** during training. The `MultiTractGNNTrainer` includes a constraint loss term that penalizes deviation of the mean prediction from the known tract target value. Post-training, constraint error for this run was 1.43%, which the pipeline grades as "excellent". There is no analytic post-hoc rescaling; the model learns to satisfy the constraint during gradient descent.

### IDW

Constraint is enforced **analytically post-hoc** in two stages (baselines.py:183-190):
1. Multiplicative rescaling: `adjusted = raw * (tract_target / mean(raw))`.
2. Iterative shift-and-clip via `_enforce_constraint`: alternates additive shifts to match the target mean and clips to [0, 1], converging to tolerance 1e-6 within 20 iterations.

Result: exact constraint satisfaction (measured deviation = 0.00).

### Kriging

Same post-hoc enforcement as IDW (baselines.py:288-292): multiplicative rescaling followed by `_enforce_constraint` (shift-clip loop). Result: exact constraint satisfaction (measured deviation = 0.00).

---

## Summary

All Check 1 sub-items: **PASS**. No discrepancies found. All four columns share the same [0, 1] normalized space derived from county-wide min-max normalization of log_appvalue. Constraint satisfaction is within expected tolerances for all three methods.

The three methods differ intentionally in spatial basis (address-level graph vs tract-centroid interpolation) and constraint mechanism (learned vs analytic post-hoc). These asymmetries are by design and will be disclosed in the dissertation chapter.
