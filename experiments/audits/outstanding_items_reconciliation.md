# Outstanding Items Reconciliation

**Date:** 2026-06-05
**Scope:** Inquiry only. No source, doc, or memory artifacts were modified.

---

## Item 1: Dasymetric pooled BG r -- 0.844 vs 0.802

### 1. Current canonical value

**File:** `data/results/m0_n20_svi_parity/aggregate.csv`, line 3, column `pooled_bg_r`

```
method,median_bg_r,ci_low_95,ci_high_95,n_tracts,pooled_bg_r,pooled_bg_r_ci_low,pooled_bg_r_ci_high,pooled_n_bgs
GRANITE,0.3901,-0.4452,0.6974,19,0.7692,0.6602,0.8527,69
Dasymetric,0.7867,0.2531,0.8627,19,0.8018,0.7118,0.871,69
Pycnophylactic,0.2078,-0.3526,0.5289,19,0.7678,0.6516,0.8579,69
```

Dasymetric pooled BG r = **0.8018**, n_bgs = 69. This file is dated 2026-05-09 (SESSION_LOG.md line 225, M0 parity run). The identical values appear in `for_mehdi_review/m0_n20_svi_parity/aggregate.csv:3` (exact copy).

The column `ci_high_95` for Dasymetric's per-tract median BG r is **0.8627** -- a plausible source of confusion with a remembered "0.84x" value, but this is a CI high bound for a different statistic (per-tract median, not pooled BG r).

### 2. Exhaustive search for 0.844 as a Dasymetric BG-r metric

Searched all `.csv`, `.json`, `.md`, and `.txt` files under `/workspaces/GRANITE` (excluding `.claude/` and `.git/`) for `0\.844` and `0\.84[0-9]` patterns where the value could be a Dasymetric aggregate BG-r.

**Every occurrence found:**

| Value | File | Line | What it is |
|-------|------|------|-----------|
| `0.844` | `results/convergence_experiment/summary.txt` | 19 | `constrained_r` for block group GEOID 470650006003, n_addr=586, county-ranked BG convergence experiment (single tract 47065000600, Tract+BG constraint condition). **Not Dasymetric; not pooled BG r; county-ranked ground truth.** |
| `0.8476` | `for_mehdi_review/m0_n20_svi_parity/per_tract.csv` | 18 | Dasymetric **per-tract** BG r for tract 47065002400, n_bgs=4. **Not pooled BG r.** |
| `0.8627` | `data/results/m0_n20_svi_parity/aggregate.csv` | 3 | Dasymetric per-tract median BG r **CI high 95%** bound. **Not pooled BG r.** |
| `0.8627` | `for_mehdi_review/m0_n20_svi_parity/aggregate.csv` | 3 | Same |
| `0.840x` (multiple) | `for_mehdi_review/m2_n20_recovery/.../predictions.csv` | many | Individual address-level predictions for M2 recovery experiment. Unrelated to Dasymetric or BG r. |
| `0.84x` (Moran's I) | `SESSION_LOG.md`, `Research_Status.md`, ablation READMEs | various | Moran's I spatial autocorrelation values (GCN-GAT architecture). Unrelated. |

**No occurrence of 0.844 exists anywhere in the repo as a Dasymetric aggregate or pooled BG r value.**

### 3. Classification of 0.844

**(c) Untraceable as a Dasymetric pooled BG r.** The value 0.844 occurs exactly once in the repo (`results/convergence_experiment/summary.txt:19`) and is `constrained_r` for a specific block group in a county-ranked single-tract constraint experiment -- a within-BG Pearson r between GNN address-level predictions (constrained condition) and the county-ranked BG SVI ground truth for addresses inside that block group. This is not Dasymetric, not pooled BG r, and not national-ranked ground truth.

The memory entry carrying 0.844 as Dasymetric pooled BG r is not present in the current `memory/` directory (only `step5_graph_contribution.md` is indexed in MEMORY.md). It may have been from a prior conversation session whose memory file no longer exists, or it was a misread of the 0.8627 CI high bound.

### 4. Verdict

**0.802 is canonical.** It is the Dasymetric pooled BG r from the current n20 national-ranked harness, grounded in `data/results/m0_n20_svi_parity/aggregate.csv:3`. **0.844 was not a Dasymetric pooled BG r in any run**; its only occurrence in the repo is a within-BG `constrained_r` for one block group in a county-ranked single-tract constraint experiment.

---

## Item 2: Coordinate-artifact experiment -- what it already establishes

### 1. `feature_mode` substitution in code

**File:** `granite/disaggregation/pipeline.py`

Call site (line 310-314):
```python
normalized_features = self._apply_feature_mode(
    normalized_features,
    tract_addresses,
    mode=self.config.get('feature_mode', 'full'),
    seed=self.config.get('seed', 42),
)
```

Method definition: `_apply_feature_mode`, lines 1720-1762. Four modes:

| Mode | Substitution | Lines |
|------|-------------|-------|
| `'full'` | Pass through unchanged | 1732-1733 |
| `'coordinates_only'` | Zeros matrix (N, d); column 0 = z-scored lat; column 1 = z-scored lon; all other columns = 0 | 1749-1753 |
| `'random_noise'` | i.i.d. N(0, 1) matrix of shape (N, d), fixed seed | 1755-1756 |
| `'coords_plus_noise'` | Columns 0-1 = z-scored lat/lon; columns 2-(d-1) = i.i.d. N(0, 1) | 1758-1760 |

Lat/lon z-scores computed as `(x - x.mean()) / (x.std() + 1e-8)` at lines 1746-1747. Called after `normalize_accessibility_features` in `_process_single_tract`.

### 2. Newest (and only surviving) result artifacts

**None exist.** The output directories produced by running the coord-artifact experiment were deleted on 2026-04-27 (SESSION_LOG.md:322):

> `output/coord_artifact_test/`, `output/coord_artifact/` (~133 MB total) -- old experiment output directories, results already reviewed.

Checked locations:
- `/workspaces/GRANITE/output/coord_artifact/` -- does not exist
- `/workspaces/GRANITE/output/coord_artifact_test/` -- does not exist
- `/workspaces/GRANITE/data/results/` -- no coord_artifact subdirectory
- `/workspaces/GRANITE/experiments/` -- no coord_artifact results

The scripts `scripts/coord_artifact_experiment.py` and `scripts/coord_artifact_bg_validation.py` exist, but neither has been run after 2026-04-27. `coord_artifact_bg_validation.py` cannot run at all without the `output/coord_artifact_test/` input (the predictions from the experiment).

### 3. The comparison the experiment was designed to run

**Design** (from `scripts/coord_artifact_experiment.py` and SESSION_LOG.md:333-338):
- Feature modes: `full`, `coordinates_only`, `random_noise`, `coords_plus_noise`
- Architectures: SAGE (gcn_gat not included in this script; coord_artifact_experiment.py runs SAGE only)
- Tracts: 5 Mehdi review tracts (47065000700, 47065000600, 47065011326, 47065011321, 47065002400)
- Epochs: 200, seed: 42
- Metric: `coord_artifact_bg_validation.py` computes Pearson r at block-group scale (`r_bg`) with bootstrap CIs, comparing each GNN mode against Dasymetric and Pycnophylactic
- Ground truth: nationally-ranked ACS BG SVI from `data/processed/national_bg_svi.csv` (line 88 of validation script)

**Decision logic** (coord_artifact_bg_validation.py lines 354-381): if `r(full) - r(coordinates_only) >= 0.05` AND CIs do not overlap, conclude "features contribute block-group-detectable vulnerability signal beyond coordinate structure." Otherwise: "features produce spatially structured variation not aligned with block-group boundaries."

### 4. What the experiment concluded -- with provenance

**The result artifacts no longer exist.** No CSV, JSON, or text file survives with per-condition metric values. The SESSION_LOG.md has no entry recording the results of running `coord_artifact_experiment.py` -- the 2026-04-23 entry (SESSION_LOG.md:328-340) only documents the infrastructure creation, not any run.

The values cited in other files:

| Claim | Source file | Line | Provenance |
|-------|------------|------|-----------|
| `r ~ 0.03` (accessibility features) | `docs/FEATURES.md` | 50 | Secondary summary; no surviving primary artifact |
| `r ~ 0.67` (spatial coordinates) | `docs/FEATURES.md` | 50 | Same |
| `r=0.033` (accessibility) | `Research_Status.md` | 27 | Secondary summary; no surviving primary artifact |
| `r=0.671` (coordinates) | `Research_Status.md` | 27 | Same |
| `r~0.67` (coordinates) | `CLAUDE.md` | 96 | Same |
| `r~0.03` (accessibility) | `CLAUDE.md` | 96, 116 | Same |

All of these trace to deleted `output/coord_artifact_test/` and `output/coord_artifact/` artifacts. The disk cleanup entry (SESSION_LOG.md:322) notes "results already reviewed" but does not record the values. There is no session log entry between the 2026-04-23 infrastructure creation and the 2026-04-27 deletion that quotes the metric values.

**Conclusion:** The README claim "raw spatial coordinates outperformed accessibility features" is not supported by any surviving artifact. The values r=0.671 and r=0.033 are summary claims propagated from an experiment whose output files were deleted. They may be accurate -- the design is sound -- but they are not grounded to any current file at a file:line level.

### 5. Overlap with step 5b

Step 5 (graph contribution, `experiments/ablation/05_graph_contribution/`) established via production vs mlp_floor comparison that the road-network graph contributes to Moran's I (~0.175 mean gap, ~7x seed std) but NOT to bg_r (mlp_floor bg_r slightly higher in some conditions). The conclusion is that **node features alone carry prediction accuracy**; the graph adds spatial clustering without improving BG-level ranking.

Step 5b (`experiments/ablation/05b_topology_specificity/`) appears to extend this by comparing specific graph topologies (road vs feature-similarity vs randomized wiring).

The coord-artifact experiment addresses a different and complementary question: **which node features carry the signal?** Specifically, whether spatial position (lat/lon) alone accounts for whatever prediction accuracy the node features provide, vs. the full 73-feature set vs. random noise. If `coordinates_only` matches `full` at BG r, then the 73 engineered features add nothing beyond location. If `random_noise` also matches, then even location adds nothing -- the model is fitting the tract-mean constraint correction only.

Step 5b does not answer this. Step 5b holds node features constant and varies graph topology. The coord-artifact experiment holds graph topology constant and varies node feature content. **The coord-artifact experiment already answers "does spatial position carry the signal" better than step 5b does** -- but its results are not currently available. Re-running `scripts/coord_artifact_experiment.py` followed by `scripts/coord_artifact_bg_validation.py` would produce the BG-r comparison across feature modes, which is currently ungrounded. Step 5b's additional contribution beyond coord-artifact is characterizing which wiring topology (road-network vs feature-similarity vs random) drives the Moran's I contribution that was already established in step 5.
