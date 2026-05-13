# GRANITE Research Status

**Project title:** Boundary Conditions for Constrained Graph Neural Network Spatial Disaggregation

**Codename:** GRANITE

**Last updated:** 2026-05-12

**Active branch:** `main` (HEAD: `6e5173c`)

---

## Current strategic position

**Primary contribution:** synthetic testbed (Phase B, M5 through M7). The boundary surface mapping where constrained GNN disaggregation succeeds versus collapses is the dissertation's primary evidence.

**Empirical complement (not headline):** Door 2 external-target recovery; tax delinquency in acquisition.

**Scaffolding (closed):** traditional-method parity check (M0) complete. Pooled BG r demonstrates constraint-preserving parity; per-tract BG r reveals Dasymetric's within-tract ancillary-variable advantage.

---

## Locked-in findings

1. **Architecture-dependent feature survival (n20 rank-consistency).** GraphSAGE preserves zero rank-consistent features at primary threshold (cv=0.10, min_tracts=12); GCN-GAT preserves 14. Methodological contribution independent of any performance claim.

2. **Ecological fallacy at address scale.** Tract-level SVI correlates do not carry within-tract signal. The 30-feature accessibility set hits r=0.033 at address level; raw x,y coordinates hit r=0.671.

3. **Within-tract feature redundancy (M3.5).** The 73-feature stack is internally redundant. GBM hit r approximately 0.957 to 1.000 across all three engineered held-out targets via functional proxy reconstruction. Exhibit: `employment_count_10min` reconstructed `employment_walk_effective_access` at GBM importance 0.9997. Killed feature engineering prospects for a positive outcome.

4. **Constraint-vs-feature-signal tug-of-war.** Under strong feature-target coupling, the soft constraint becomes a tax the trainer partially escapes. M2 exhibit: r=0.509 with 43.5% constraint error on `employment_walk_effective_access`. Methods-chapter material.

5. **Mechanism honesty (April 2026 audit).** Training uses multi-objective soft loss including a constraint term. Inference applies an additive mean-centering correction equal to the closed-form Euclidean projection onto the tract-mean constraint, also the base case of hierarchical forecast reconciliation. "Hard aggregate constraint enforcement acting as implicit regularizer" framing has been retired.

6. **Pooled-versus-per-tract metric divergence (M0, May 2026).** Pooled BG r compresses between-tract and within-tract performance; tract-mean preservation drives most of the pooled signal. Per-tract BG r isolates within-tract allocation skill and reveals different method behavior. 

---

## Mechanism reference

**Constraint enforcement.** Soft MSE constraint term in the training loss (weight 2.0 in single-tract `_compute_losses` at `granite/models/gnn.py:562`; same weight in multi-tract `_compute_multi_tract_losses` at `granite/models/gnn.py:1149`). Inference applies iterative bounded projection in `granite/disaggregation/pipeline.py:_finalize_predictions` (max_iter=50, tol=1e-8), gated by `apply_post_correction: true` in `config.yaml`.

**Graph construction.** Dual edge set in `granite/data/loaders.py:_create_road_network_graph`. Road-network edges: 500 m road snap, candidate pairs filtered to less than 1000 m Euclidean and less than 1500 m road-network shortest-path, weight `1 / (1 + path_length / 500)`. Geographic edges: k=6, less than 1000 m Euclidean filter, weight `exp(-distance_m / 300)`. Symmetric, deduplicated. No feature-similarity edges.

**Normalization.** `LayerNorm` at GNN input. `BatchNorm` after each graph convolution layer. `RobustScaler` on raw input features (`normalize_accessibility_features` in `granite/models/gnn.py`), fit globally across all addresses in the batch. Per-tract feature standardization is not in the GNN path; the M3 baselines do apply per-tract z-score at `granite/evaluation/recovery_baselines.py:139`.

**Cross-tract smoothness loss.** Active in the constrained multi-tract path at weight 0.1 (`granite/models/gnn.py:1198`). Weight is hardcoded; no config flag currently disables it.

**Features.** 73 address-level features. 30 base accessibility features. 15 modal features computed per-address from OSRM drive and walk times (tract-level fallback exists for cache-cold OSRM-unreachable cases and is not the active path under normal operation). 9 socioeconomic features broadcast as tract-level constants by design. 19 address-level attributes from parcel records, Microsoft Building Footprints, FEMA NFHL flood zones, and NLCD 2021. See docs/FEATURES.md for full list.

---

## Milestone status

### Closed

- **M0** (2026-05-09): Traditional-method parity check on the n20 stratified subset against Dasymetric and Pycnophylactic at block-group resolution; established pooled BG parity and per-tract divergence. See M0 entry below.
- **M1** (2026-04-29): Built the held-out feature recovery harness inside `granite/`, with one config switch to select a target feature, drop it from inputs, and use its tract aggregate as the soft constraint.
- **M2** (2026-05-03): Ran the recovery harness across three engineered targets (`log_appvalue`, `employment_walk_effective_access`, `nlcd_impervious_pct`) on n20 with both architectures; produced per-tract Pearson r, RMSE, and constraint error. See M2 entry below.
- **M3** (2026-05-03): Ran per-address ridge and gradient-boosted regression on the same retained features with no graph and no constraint, to measure how much recovery comes from feature correlations alone and compute GRANITE's lift over a non-graph baseline. See M3 entry below.
- **M3.5** (2026-04): Applied a redundancy filter to the 73-feature stack and confirmed within-tract functional proxy redundancy; GBM ceiling at r approximately 0.957 to 1.000 across all three engineered targets, with GRANITE never clearing ridge. Killed Phase A on engineered features.

### Active

- **M5** (in progress): Implementing the synthetic target generator that produces address-level "true" targets from controlled mechanisms (signal type, signal-to-noise ratio, spatial autocorrelation) over the real tract assignments, address coordinates, and k-NN graph topology. Smoke driver at `granite/scripts/run_m5_smoke.py`; generator at `granite/synthetic/generator.py`.
- **M3.6 through M3.8** (parallel track): Door 2 acquisition of an external target (tax delinquency) absent from the feature stack, with the redundancy filter gating whether it proceeds to GRANITE training.

### Pending

- **M6**: Run the synthetic parameter grid across 4 to 6 mechanisms x 3 SNR levels x 3 autocorrelation levels on both architectures, logging recovery r and constraint error per cell. Estimated 72 to 108 runs.
- **M7**: Characterize the boundary in (signal type, SNR, autocorrelation) space where constrained GNN disaggregation succeeds versus collapses toward proximity allocation, and cross-reference with Phase A real-data results.
- **M8**: Document mathematically why block-group r tests SVI's scale-decomposability rather than GRANITE's signal extraction, repositioning the BG result as a diagnostic on the index.
- **M9**: Fix the `name 'count_5' is not defined` validator bug that blocks downstream tooling.
- **M10**: Update `CLAUDE.md`, `README.md`, and remaining documentation to match the recovery-framework framing and retire residual "hard constraint as regularizer" language.
- **M11**: Produce a five-figure canonical suite: held-out recovery panel, boundary surface, architecture-dependent feature survival heatmap, ecological-fallacy bar, BG scale-decomposability diagnostic.
- **M12**: Build cross-experiment synthesis tables showing synthetic results predict real-data results and that architecture-dependent feature survival holds across both phases.
- **M13**: Pre-writing reproducibility audit; reproduce headline numbers from a clean checkout, lock seeds and configs, generate the chapter-ready artifact bundle.

### Retired or demoted

- IDW and Kriging (graveyard, 2026-04-18). Collapse to tract mean under single-centroid interpolation.
- "GRANITE beats baselines on SVI" as a headline.
- "Hard constraint enforcement as implicit regularizer" framing.
- Block-group r as a benchmark; demoted to scale-decomposability diagnostic.

---

## Milestone entries

### M0: n20 SVI parity check (2026-05-09)

**Status:** complete.

**Setup.** 20 stratified tracts, 39,535 addresses, GraphSAGE only. Three methods: GRANITE, Dasymetric (NLCD impervious surface ancillary), Pycnophylactic (k-NN adjacency). Validation: aggregate address predictions to BG centroids (min 10 addresses/BG), correlate against nationally-ranked ACS BG SVI from `data/processed/national_bg_svi.csv`. Bootstrap 1000 resamples for CIs.

**Pooled BG r (69 BGs combined):**

| Method | pooled_bg_r | CI 95% |
|--------|-------------|--------|
| GRANITE | 0.769 | [0.660, 0.853] |
| Dasymetric | 0.802 | [0.712, 0.871] |
| Pycnophylactic | 0.768 | [0.652, 0.858] |

**Per-tract median BG r (19 tracts with valid r):**

| Method | median_bg_r | CI 95% |
|--------|-------------|--------|
| Dasymetric | 0.787 | [0.253, 0.863] |
| GRANITE | 0.390 | [-0.445, 0.697] |
| Pycnophylactic | 0.208 | [-0.353, 0.529] |

**Pairwise separability (per-tract median difference, 95% bootstrap):**

| Pair | obs diff | CI 95% | separable |
|------|----------|--------|-----------|
| granite_vs_dasymetric | -0.121 | [-0.536, 0.108] | No |
| granite_vs_pycno | 0.016 | [-0.107, 0.207] | No |
| dasymetric_vs_pycno | 0.403 | [0.044, 0.640] | Yes |

**Decision.** Pooled parity holds; Narrative-A footnote survives in technical terms. Per-tract divergence reveals Dasymetric's ancillary-variable advantage on within-tract allocation. GRANITE matches Pycnophylactic at the within-tract level. The 0.30 absolute jump from prior records (BG r 0.469 to 0.769) is likely a metric-definition difference (per-tract or BG-internal vs current pooled), pending reconciliation.

**Constraint error sanity check.** All three methods at median 0.0000% (Dasymetric and Pycnophylactic by construction; GRANITE via post-hoc reconciliation).

**Artifacts.**
- `granite/scripts/run_m0_parity.py` (~660 lines, single-CLI driver)
- `data/results/m0_n20_svi_parity/per_tract.csv` (60 rows)
- `data/results/m0_n20_svi_parity/aggregate.csv`
- `data/results/m0_n20_svi_parity/pairwise_diffs.csv`
- `data/results/m0_n20_svi_parity/RESULTS.md`

**Open follow-ups.**
1. Reconcile the 0.469-to-0.769 jump against prior records by recomputing the old metric definition.
2. Per-tract distribution inspection: identify tracts where GRANITE wins decisively (if any) and characterize them (SVI band, address density, BG count).
3. Methodological note on pooled-vs-per-tract as a chapter-three contribution.

---

### M2: held-out engineered feature recovery, n20 (2026-05-03)

**Status:** complete.

**Setup.** Three engineered targets (`log_appvalue`, `employment_walk_effective_access`, `nlcd_impervious_pct`) across both architectures (GraphSAGE, GCN-GAT) on the n20 stratified subset. Target column dropped from inputs; per-tract mean of the held-out target replaces SVI as the soft training constraint. Entry point: `granite/disaggregation/recovery_harness.py:run_recovery`.

**Median per-tract Pearson r at address level:**

| Target | SAGE r | GCN-GAT r | SAGE constraint err |
|---|---|---|---|
| log_appvalue | 0.0387 | 0.1027 | 3.1% |
| employment_walk_effective_access | 0.5090 | 0.4819 | 43.5% |
| nlcd_impervious_pct | [fill from file] | [fill from file] | [fill from file] |

**Reading.** The `employment_walk_effective_access` cell is the constraint-vs-feature-signal tug-of-war exhibit (locked-in finding 4): nontrivial recovery r purchased by violating the constraint by 43.5%. `log_appvalue` near zero across both architectures; `nlcd_impervious_pct` to be filled in.

**Source.** `output/m2_n20_recovery/summary/summary_stats.csv` and per-target subdirectories `output/m2_n20_recovery/{target}_{architecture}/`.

---

### M3: non-graph leakage baselines, n20 (2026-05-03)

**Status:** complete.

**Setup.** Per-address ridge regression and gradient-boosted regression on the same three M2 targets with no graph and no constraint. Same retained features as the GNN path with target column dropped. Predictors per-tract z-scored. Entry point: `granite/evaluation/recovery_baselines.py:run_baselines`.

**Median per-tract Pearson r at address level:**

| Target | Ridge r | GBM r |
|---|---|---|
| log_appvalue | 0.8678 | 0.9759 |
| employment_walk_effective_access | 0.5870 | 0.9999 |
| nlcd_impervious_pct | 0.8245 | 0.9574 |

**Reading.** GBM ceiling at r approximately 0.957 to 1.000 across all three targets. Ridge clears 0.58 on the worst target. GRANITE never cleared ridge on any target. M3.5 then explained this mechanistically as within-tract feature redundancy (locked-in finding 3).

**Source.** `output/m3_n20_baselines/summary/baseline_summary_stats.csv` and per-target subdirectories `output/m3_n20_baselines/{target}/`. Lift table at `output/m3_n20_baselines/summary/lift_table.csv`.

---

## Where to find things

**Code.**

```
granite/
  models/gnn.py                          # GNN architectures, trainers, losses
  disaggregation/
    pipeline.py                          # Main pipeline; _finalize_predictions
    recovery_harness.py                  # M2 held-out feature recovery
  features/
    enhanced_accessibility.py            # 30 base accessibility features
    modal_accessibility.py               # 15 modal features (per-address path)
    osrm_router.py                       # OSRM interface
  data/loaders.py                        # Graph construction, address joins
  evaluation/
    recovery_baselines.py                # M3 ridge and GBM baselines
    redundancy_filter.py                 # M3.5 admissibility check
    run_ablation_study.py                # Feature-replacement ablations
  scripts/
    run_granite.py                       # CLI entry point
    run_m0_parity.py                     # M0 parity driver
    run_m5_smoke.py                      # M5 synthetic generator smoke driver
  synthetic/generator.py                 # M5 synthetic target generator
```

**Configuration.**

```
config.yaml                              # Constraint weights, scaling, seeds
CLAUDE.md                                # Working notes for Claude Code sessions
README.md                                # CLI usage and flags
```

**Results.**

```
data/results/m0_n20_svi_parity/
  aggregate.csv                          # Pooled and per-tract medians
  per_tract.csv                          # Per-tract per-method BG r
  pairwise_diffs.csv                     # Bootstrap pairwise separability
  RESULTS.md                             # Narrative summary

output/m2_n20_recovery/
  {target}_{architecture}/
    predictions.csv
    per_tract_metrics.csv
    run_meta.json
  summary/
    summary_stats.csv
    pivot_pearson_r.csv
    pivot_rmse.csv

output/m3_n20_baselines/
  {target}/
    per_tract_metrics.csv
    run_meta.json
  summary/
    baseline_summary_stats.csv
    lift_table.csv
    lift_summary.csv
    per_tract_metrics.csv
```