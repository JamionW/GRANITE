# M0 Discovery Report

**Date:** 2026-05-09
**Phase 1 conclusion:** M0 has NOT been run. No results, no driver script, no partial artifacts.

---

## 1. M0 existence check

**Result: NO**

No M0 artifacts found anywhere in the repo. Searched:
- `data/results/` (directory did not exist prior to this report)
- `results/` (repo root): contains only `convergence_experiment`, `rank_consistency_full`,
  `rank_consistency_n20`, `rank_consistency_n20_sens1`, `rank_consistency_n20_sens2`,
  `rank_consistency_smallpass`
- `output/`: contains `m2_n20_recovery`, `m3_n20_baselines`, `rank_consistency_run`,
  `recovery`, and single-tract outputs
- All `.md` files in repo root and `docs/`: no "M0", "parity check", "n20 SVI",
  "Dasymetric Pycnophylactic SVI" entries
- `SESSION_LOG.md`: no M0 entry; M2 and M3 runs are logged but M0 never appears
- `granite_roadmap.md`: does not exist
- `Research_Status.md`: does not exist

---

## 2. Resolved n20 FIPS list

Source: `tract_inventory.csv` (20 rows, no Status column). The `load_n20_tracts()`
function in `granite/scripts/run_m2_sweep.py` handles the no-Status case: uses all 20
rows, sorts alphabetically by FIPS. The sorted list is identical to
`output/m2_n20_recovery/summary/n20_tract_list.txt`, confirming consistency with M2.

| # | FIPS        | SVI (tract) |
|---|-------------|-------------|
|  1 | 47065000600 | 0.2235      |
|  2 | 47065000700 | 0.1140      |
|  3 | 47065001200 | 0.9066      |
|  4 | 47065001800 | 0.5471      |
|  5 | 47065001900 | 0.9804      |
|  6 | 47065002400 | 0.8910      |
|  7 | 47065003400 | 0.8797      |
|  8 | 47065010431 | 0.3848      |
|  9 | 47065010433 | 0.4540      |
| 10 | 47065010435 | 0.7153      |
| 11 | 47065011311 | 0.7464      |
| 12 | 47065011321 | 0.6960      |
| 13 | 47065011324 | 0.0374      |
| 14 | 47065011325 | 0.1713      |
| 15 | 47065011326 | 0.5100      |
| 16 | 47065011402 | 0.5759      |
| 17 | 47065011413 | 0.2735      |
| 18 | 47065011444 | 0.7088      |
| 19 | 47065011447 | 0.1650      |
| 20 | 47065011900 | 0.2822      |

SVI range: 0.0374 to 0.9804. Reasonable decile spread for stratified parity test.

**Admissibility flag:** `tract_inventory.csv` has no Status column. No formal ✓/✗
filter is applied; all 20 rows are used. Whether all 20 tracts are genuinely admissible
(sufficient addresses, OSRM coverage, BG containment) has not been independently
verified for M0 purposes. M2 ran all 20 successfully (39,535 addresses total per
`output_m2_sweep.log`), which is a strong proxy for admissibility.

---

## 3. Harness inspection

### 3a. CLI flag for SVI target

`run_granite.py` has `--target {svi,property_value}` (added per SESSION_LOG
2026-04-xx entry). Default is `svi`. For M0 the relevant invocation is the
standard SVI pipeline with no `--recover-feature` flag.

### 3b. CLI flag or config for n20 stratified set

No single flag selects the n20 set for SVI-mode runs. The n20 list is read by
`granite/scripts/run_m2_sweep.py::load_n20_tracts('tract_inventory.csv')` and
passed as an explicit list to `_run_recovery_explicit_tracts()`. For M0 a similar
batch-driver approach would be required; no existing script provides it.

### 3c. Deterministic n20 selector function and seed

`load_n20_tracts()` in `granite/scripts/run_m2_sweep.py` (lines 64-95):
reads `tract_inventory.csv`, optionally filters on `Status == '✓'`, sorts the
FIPS list alphabetically. No random seed involved: the n20 set is a fixed
inventory, not a random sample. GNN training seed is `SEED = 42` (also the
`processing.random_seed` default in `config.yaml`).

### 3d. How GRANITE, Dasymetric, and Pycnophylactic are dispatched in the same run

**They are not wired together anywhere.**

- GRANITE (SVI mode): `GRANITEPipeline.run()` via `run_granite.py --fips X`
  or `_run_recovery_explicit_tracts()` in `run_m2_sweep.py` (feature-recovery
  variant). SVI mode is the standard pipeline path, not via the recovery harness.
- Dasymetric: `DasymetricDisaggregation` in `granite/evaluation/baselines.py`
  (lines 84-124). Available as a class; takes pre-loaded `address_gdf` with
  `nlcd_impervious_pct` column and returns an address-level array.
- Pycnophylactic: `PycnophylacticDisaggregation` in `granite/evaluation/baselines.py`
  (lines 127-188). Available as a class; purely geometric, no features needed.
- `run_disaggregation_baselines()` (baselines.py lines 440-474) dispatches all
  three, but requires GNN predictions as an input argument and does not support
  BG-level output aggregation or n20 batch iteration.

**Gap:** No M0 driver script exists. One must be written.

### 3e. Output path convention

- M2: `output/m2_n20_recovery/{target}__{arch}/` per run; summary in
  `output/m2_n20_recovery/summary/`
- M3: `output/m3_n20_baselines/{target}/` per run; summary in
  `output/m3_n20_baselines/summary/`
- M0 (proposed, matching convention):
  `output/m0_n20_svi_parity/{method}/` per method;
  canonical output at `data/results/m0_n20_svi_parity/`

---

## 4. Exact CLI invocation for M0

No single CLI invocation exists. M0 requires a new driver script. The equivalent
batch command once that script is written would be:

```bash
python granite/scripts/run_m0_parity.py \
    --config config.yaml \
    --verbose
```

That script would need to:
1. Load n20 from `tract_inventory.csv` (reuse `load_n20_tracts()`).
2. For GRANITE (sage and optionally gcn_gat): call
   `_run_recovery_explicit_tracts()` adapted for SVI target (constraint =
   tract SVI from `tract_inventory.csv`), or call `GRANITEPipeline.run()` per
   tract. BG-level aggregation needed for final r computation.
3. For Dasymetric and Pycnophylactic: call `DasymetricDisaggregation.disaggregate()`
   and `PycnophylacticDisaggregation.disaggregate()` per tract, loading addresses
   and tract GDF via `GRANITEPipeline._load_spatial_data()`.
4. Aggregate per-address predictions to block-group means; join against BG-level
   ACS SVI (available at `data/processed/national_bg_svi.csv`) for Pearson r.
5. Compute bootstrap CIs and write to `data/results/m0_n20_svi_parity/`.

---

## 5. Estimated wall-clock time

Reference: M2 sweep (n20 x 3 targets x 2 architectures = 120 GRANITE runs) took
15.3 min with warm cache (SESSION_LOG 2026-05-03).

M0 run plan:
- GRANITE (sage): 20 tracts, 1 target (SVI), ~100 epochs = 20 runs.
  Proportional estimate: 15.3 x (20/120) = ~2.6 min (warm cache).
- GRANITE (gcn_gat, optional): +2.6 min.
- Dasymetric (20 tracts): seconds total (no training).
- Pycnophylactic (20 tracts): ~1-2 sec/tract x 20 = ~30 sec.
- BG aggregation + bootstrap (1000 resamples): ~1 min.
- **Total (GraphSAGE only, warm cache): ~5-7 min.**
- **Total (both architectures, warm cache): ~8-10 min.**
- Cold cache: multiply GRANITE portion by ~10-15x => ~30-50 min.

OSRM servers must be running for any cold-cache run.

---

## 6. Discrepancies and pre-existing issues

### Missing reference files (blocking for Phase 2 spec)

- `granite_roadmap.md`: does not exist. Phase 2 spec references it for the
  "Narrative-A parity footnote" decision. Cannot evaluate that framing without it.
- `Research_Status.md`: does not exist. Phase 2 spec says to append M0 entry.
  A new file would need to be created (or the SESSION_LOG used as substitute).

### Baseline naming vs. reported results

CLAUDE.md reports "IDW block-group correlation: r = 0.558". IDW was retired and
replaced by Dasymetric/Pycnophylactic (see `graveyard/disaggregation_baselines_idw_kriging.py`,
retired 2026-04-18). The r = 0.558 reference may correspond to old IDW, not the
current Dasymetric class. M0 would establish what the current baselines achieve
and whether the r = 0.558 reference is still load-bearing.

### Evaluation metric for SVI disaggregation

The Pearson r in Phase 2 spec ("per-tract metrics: Pearson r") is ambiguous for
SVI mode: there is no address-level ground truth. The interpretable r metric is at
block-group level (aggregate address predictions to BG means, compare against ACS
BG SVI), matching the existing bg_validation_results.csv approach. The script must
clarify this; per-tract Pearson r against BG-aggregated truth is the proxy.

### GRANITE SVI path vs. recovery harness

`_run_recovery_explicit_tracts()` in `run_m2_sweep.py` runs feature recovery (not
SVI disaggregation). For M0 GRANITE needs to run the standard SVI pipeline with
the tract-SVI constraint. The appropriate entry point is `GRANITEPipeline.run()`
or a refactored version of `_run_recovery_explicit_tracts()` that takes
`tract_svi_values` directly from `tract_inventory.csv` rather than computing a
feature mean.

### No Status column in tract_inventory.csv

All 20 entries are treated as admissible by default. No formal ✓/✗ filter is
possible with the current inventory file. M2 succeeded on all 20, so this is a
documentation gap, not a blocking issue.

---

## 7. Summary judgment

M0 has never been run. The n20 list is resolved, unambiguous, and verified against
M2. The three methods exist in the codebase but are not wired into a unified parity
runner. A new driver script (~150-200 lines, modeled on `run_m2_sweep.py`) is
required before Phase 2 can execute. Two reference files cited in the Phase 2 spec
(`granite_roadmap.md`, `Research_Status.md`) do not exist and must be addressed
before the RESULTS.md decision recommendation can reference them.

**Awaiting user confirmation before proceeding to Phase 2.**
