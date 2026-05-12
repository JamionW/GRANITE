# GRANITE Research Status

**Project title:** Boundary Conditions for Constrained Graph Neural Network Spatial Disaggregation

**Codename:** GRANITE

**Last updated:** 2026-05-11

---

## Current strategic position

**Primary contribution:** synthetic testbed (Phase B, M5 through M7). The boundary surface mapping where constrained GNN disaggregation succeeds versus collapses is the dissertation's primary evidence.

**Empirical complement (not headline):** Door 2 external-target recovery; tax delinquency in acquisition.

**Scaffolding (closed):** traditional-method parity check (M0) complete. Pooled BG r demonstrates constraint-preserving parity; per-tract BG r reveals Dasymetric's within-tract ancillary-variable advantage. Both readings are dissertation material.

---

## Locked-in findings

1. **Architecture-dependent feature survival (n20 rank-consistency).** GraphSAGE preserves zero rank-consistent features at primary threshold (cv=0.10, min_tracts=12); GCN-GAT preserves 14. Methodological contribution independent of any performance claim.

2. **Ecological fallacy at address scale.** Tract-level SVI correlates do not carry within-tract signal. The 30-feature accessibility set hits r=0.033 at address level; raw x,y coordinates hit r=0.671.

3. **Within-tract feature redundancy (M3.5).** The 73-feature stack is internally redundant. GBM hit r ~0.976 to 1.000 across all three engineered held-out targets via functional proxy reconstruction. Exhibit: `employment_count_10min` reconstructed `employment_walk_effective_access` at GBM importance 0.9997. Killed Phase A on engineered features.

4. **Constraint-vs-feature-signal tug-of-war.** Under strong feature-target coupling, the soft constraint becomes a tax the trainer partially escapes. M2 exhibit: r=0.509 with 43% constraint error on `employment_walk_effective_access`. Methods-chapter material.

5. **Mechanism honesty (April 2026 audit).** Training uses multi-objective soft loss including a constraint term. Inference applies an additive mean-centering correction equal to the closed-form Euclidean projection onto the tract-mean constraint, also the base case of hierarchical forecast reconciliation. "Hard aggregate constraint enforcement acting as implicit regularizer" framing has been retired.

6. **Pooled-versus-per-tract metric divergence (M0, May 2026).** Pooled BG r compresses between-tract and within-tract performance; tract-mean preservation drives most of the pooled signal. Per-tract BG r isolates within-tract allocation skill and reveals different method behavior. Methodological contribution on its own: pooled BG r is a misleading validation metric for disaggregation methods that preserve aggregate constraints.

---

## Milestone status

### Closed

- **M0** (2026-05-09): n20 stratified SVI parity check against Dasymetric and Pycnophylactic at BG resolution. See M0 entry below.
- **M1 through M3** (Q1 2026): Phase A recovery harness, three-target run, non-graph leakage baseline. GBM established the within-tract redundancy ceiling.
- **M3.5** (2026-04): Redundancy filter applied; engineered-feature recovery collapsed. Phase A reframed as empirical complement.

### Active

- **M5** (next): synthetic generator implementation. Prompt creation is the next conversation's task.
- **M3.6 through M3.8**: Door 2 tax delinquency acquisition in parallel.

### Pending

- **M6**: synthetic parameter grid.
- **M7**: boundary characterization.
- **M8, M10**: documentation alignment (post-direction-lock).
- **M9**: validator bug `name 'count_5' is not defined`.
- **M11 through M13**: figure suite, cross-experiment synthesis, reproducibility audit.

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

## Reference artifacts in project knowledge

- `granite_roadmap.md`: milestone definitions and sequencing.
- `CONVERSATION_PROTOCOL.md`: Claude operating contract.
- `DEFENSE_FRAMING.md`: dissertation defense framing.
- `Ecological_Fallacy_Finding.md`: detailed exhibit for finding 2.
- `tract_inventory.csv`: 85-tract roster with SVI values.
