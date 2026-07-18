# Section 5 Verification Manifest

**Generated:** 2026-07-18
**Purpose:** Read-only verification recon of committed GRANITE artifacts. No source files were edited.
**Scope:** P1 through P18 as specified.

---

## P1 -- Tract counts: Hamilton County FIPS 47065

**Value:** 87 total tracts in Hamilton County, TN (FIPS 47); 85 carry valid SVI values (RPL_THEMES != -999).

**Artifact:** `/workspaces/GRANITE/data/raw/SVI_2020_US.csv`

**Command:**
```python
import pandas as pd
df = pd.read_csv('.../SVI_2020_US.csv', dtype={'FIPS': str, 'ST': str})
hamilton_tn = df[(df['ST'] == '47') & (df['COUNTY'] == 'Hamilton')]
print('Total Hamilton TN rows:', len(hamilton_tn))         # 87
print('With SVI (not -999):', (hamilton_tn['RPL_THEMES'] != -999).sum())  # 85
```

**Excerpt:** Script output: "Total Hamilton TN rows: 87 | With SVI (not -999): 85"

**Measurement definition:** Census 2020 SVI file (SVI_2020_US.csv), state FIPS 47, county name Hamilton. Rows with RPL_THEMES == -999 are excluded as having no valid SVI.

**Note:** The n20 stratified subset uses 20 tracts from this population; tract_inventory.csv lists these explicitly.

---

## P2 -- Address count: total address-level records

**Value:** 39,535 addresses in the n20 stratified subset (the active analysis dataset). The full county address file (combined_address_features.csv) contains 102,761 data rows (102,762 lines including header).

**Artifacts:**
- n20 count: `experiments/recovery/per_address_predictions/granite_m0.parquet` (39,535 rows)
- Full county: `data/raw/address_features/combined_address_features.csv`

**Command:**
```bash
wc -l /workspaces/GRANITE/data/raw/address_features/combined_address_features.csv
# 102762
python3 -c "import pandas as pd; print(len(pd.read_parquet('experiments/recovery/per_address_predictions/granite_m0.parquet')))"
# 39535
```

**Excerpt:** `wc -l` result: 102762; parquet len: 39535

**Measurement definition:** n20 count = all addresses falling within the 20 stratified n20 tracts (spatial join via `get_addresses_for_tract`). Full county = all records in the raw address features file.

---

## P3 -- Assessor-parcel match count

**Value:** 39,476 out of 39,535 n20 addresses matched to parcels (99.9% match rate). The unmatched 59 addresses are excluded from delinquency analysis.

**Artifact:** SESSION_LOG.md entry 2026-06-10 (Delinquency convergent validity)

**Excerpt from SESSION_LOG.md line 1048:**
> "Data join: n20 addresses (lat/lon) -> parcels.shp via gpd.sjoin(predicate='within'), CRS EPSG:6576. Parcel key MAP|GROUP_|PARCEL (strip+upper, blank group -> ''). Bill Year filter [2000, 2026]. Match rate: 39,476/39,535 (99.9%). Addresses with any delinquency: 1,866 (4.7%)."

**Measurement definition:** Spatial join of n20 address centroids (lat/lon) to Hamilton County parcel polygons using `gpd.sjoin(predicate='within')`. Context is the delinquency convergent validity experiment (M3.6-M3.8).

---

## P4 -- Feature count: total features per address

**Value:** 73 features per address.

**Composition:**
- 30 base accessibility features (travel times, counts, percentiles by mode/destination)
- 15 modal features (per-address from OSRM drive/walk times)
- 9 socioeconomic features (broadcast as tract-level constants)
- 19 address-level attributes: 7 individual parcel + 4 LUCODE one-hot + 5 PROPTYPE one-hot + 3 NLCD

**Artifacts:**
- Research_Status.md ("Features" section, line 53): "73 address-level features"
- SESSION_LOG.md 2026-06-07 verification: "Feature count grounded at pipeline.py: PROPTYPE_VOCAB has 5 entries; address-level = 7 individual + 4 LUCODE one-hot + 5 PROPTYPE one-hot + 3 NLCD = 19; full total = 30+15+9+19 = 73"
- experiments/ecological_fallacy/n20_feature_matrix.csv: 39,535 rows x 78 cols (fips, address_idx, lat, lon, tract_svi + 73 features)

**Excerpt from SESSION_LOG.md (2026-06-07):**
> "feature count reconciled to 73/19 across Overview, Features, and Configuration sections"

**Note:** Feature count varies 71-73 across some tracts due to zero-variance building features (existing behavior, no impact on comparability per SESSION_LOG 2026-04-25).

---

## P5 -- Pooled block-group r for CGD (GRANITE/GraphSAGE) with bootstrap CI

**Value:** pooled_bg_r = 0.769 [0.660, 0.853], n_BGs = 69

**Artifact:** `data/results/m0_n20_svi_parity/aggregate.csv`

**Command:** `cat data/results/m0_n20_svi_parity/aggregate.csv`

**Exact CSV excerpt:**
```
method,median_bg_r,ci_low_95,ci_high_95,n_tracts,pooled_bg_r,pooled_bg_r_ci_low,pooled_bg_r_ci_high,pooled_n_bgs
GRANITE,0.3901,-0.4452,0.6974,19,0.7692,0.6602,0.8527,69
```

**Measurement definition:** Pooled Pearson r between address-level SVI predictions (aggregated to BG means, min 10 addresses/BG) and nationally-ranked ACS BG SVI from `data/processed/national_bg_svi.csv`. Bootstrap CI: 1000 resamples over n_BGs=69. GraphSAGE, single-tract mode, seed=42, 100 epochs, apply_post_correction=True.

**Comparison to expectation:** Expected P5=0.769 [0.660, 0.853]. Observed 0.7692 [0.6602, 0.8527]. Values agree within rounding (0.769 = 0.7692 rounded; CI high 0.853 vs 0.8527 due to rounding). MATCH.

---

## P6 -- Pooled block-group r for Dasymetric with bootstrap CI

**Value:** pooled_bg_r = 0.802 [0.712, 0.871], n_BGs = 69

**Artifact:** `data/results/m0_n20_svi_parity/aggregate.csv`

**Exact CSV excerpt:**
```
Dasymetric,0.7867,0.2531,0.8627,19,0.8018,0.7118,0.871,69
```

**Measurement definition:** Same as P5 but using Dasymetric disaggregation (NLCD impervious surface ancillary). No GNN training; by-construction constraint satisfaction.

**Comparison to expectation:** Expected P6=0.802 [0.712, 0.871]. Observed 0.8018 [0.7118, 0.871]. MATCH.

---

## P7 -- Pooled block-group r for Pycnophylactic with CI

**Value:** pooled_bg_r = 0.768 [0.652, 0.858], n_BGs = 69

**Artifact:** `data/results/m0_n20_svi_parity/aggregate.csv`

**Exact CSV excerpt:**
```
Pycnophylactic,0.2078,-0.3526,0.5289,19,0.7678,0.6516,0.8579,69
```

**Measurement definition:** Same as P5 but using Pycnophylactic disaggregation (Tobler 1979 k-NN adjacency).

**Comparison to expectation:** Expected P7=0.768 [0.652, 0.858]. Observed 0.7678 [0.6516, 0.8579]. MATCH (0.768 = 0.7678 rounded; CI high 0.858 vs 0.8579).

---

## P8 -- Block-group n (number of BGs used in validation)

**Value:** n_BGs = 69

**Artifact:** `data/results/m0_n20_svi_parity/aggregate.csv` (column `pooled_n_bgs`; all three methods report 69)

**Excerpt from RESULTS.md:**
> "### Pooled BG r (all 20 tracts combined, min 10 addresses/BG) | GRANITE | 0.769 | 0.660 | 0.853 | 69 |"

**Measurement definition:** Block groups from the 20 n20 tracts with at least 10 address-level predictions. One tract (47065011324, n=18 addresses) contributed only 1 BG which may not meet the min-10 threshold; tract 47065011324 shows empty bg_r in per_tract.csv.

---

## P9 -- Per-tract median r for CGD (GRANITE)

**Value:** median_bg_r = 0.390 (exact: 0.3901), CI [-0.445, 0.697], n_tracts = 19

**Artifact:** `data/results/m0_n20_svi_parity/aggregate.csv`

**Exact CSV excerpt:**
```
GRANITE,0.3901,-0.4452,0.6974,19,0.7692,0.6602,0.8527,69
```

**Measurement definition:** Median of per-tract bg_r values (one per tract), bootstrapped over per-tract values (1000 resamples). 19 of 20 tracts have valid bg_r (47065011324 excluded: only 1 BG after min-10 filter).

**Comparison to expectation:** Expected P9=0.390. Observed 0.3901. MATCH.

---

## P10 -- Per-tract median r for Dasymetric

**Value:** median_bg_r = 0.787 (exact: 0.7867), CI [0.253, 0.863], n_tracts = 19

**Artifact:** `data/results/m0_n20_svi_parity/aggregate.csv`

**Exact CSV excerpt:**
```
Dasymetric,0.7867,0.2531,0.8627,19,0.8018,0.7118,0.871,69
```

**Comparison to expectation:** Expected P10=0.787. Observed 0.7867. MATCH.

---

## P11 -- CI on per-tract median difference (CGD minus Dasymetric) and whether it spans zero

**Value:** obs_median_diff = -0.1207, CI [-0.5356, 0.108], separable = False (CI spans zero)

**Artifact:** `data/results/m0_n20_svi_parity/pairwise_diffs.csv`

**Exact CSV excerpt:**
```
pair,obs_median_diff,ci_low_95,ci_high_95,n_pairs,separable
granite_vs_dasymetric,-0.1207,-0.5356,0.108,19,False
```

**Measurement definition:** Bootstrap (1000 resamples) on paired per-tract difference (GRANITE_r - Dasymetric_r) for 19 paired tracts. CI spans zero; methods are not statistically separable at per-tract level. Dasymetric IS separable from Pycnophylactic: [0.0435, 0.6398].

---

## P12 -- Address-level r for accessibility feature set

**Value:** MISSING -- the formerly reported r=0.033 (accessibility) and r=0.671 (coordinates) have been removed from all documentation as ungrounded.

**Removal audit:** SESSION_LOG.md 2026-06-09 entry documents the surgical excision. The values traced to `output/coord_artifact/bg_validation_report.txt` (gitignored, never committed) which contained a BG-level r, not an address-level r. `experiments/audits/outstanding_items_reconciliation.md` documents this at line 117: "r=0.033 (accessibility): no surviving primary artifact."

**What replaced it:** The variance decomposition experiment (M-ecological_fallacy, 2026-06-10) provides the current grounded finding: accessibility features have median within-tract variance share (within_share) of 0.230 -- they are not flat within tracts. No address-level r against SVI exists because no address-level SVI ground truth exists. The target variable `svi` returns `None` from `get_address_truth_values(target='svi')`.

**Artifact:** `experiments/ecological_fallacy/variance_decomposition.csv` (75 rows, one per feature)

**Relevant excerpt from Ecological_Fallacy_Finding.md:**
> "Accessibility features carry meaningful within-tract variance. Median within_share = 0.230 across 30 accessibility features."

---

## P13 -- Redundancy audit headline range and top exhibit pair

**Value:**
- Headline range: GBM r approximately 0.957 to 1.000 across all three engineered held-out targets
- Top exhibit pair: `employment_count_10min` reconstructed `employment_walk_effective_access` at GBM importance 0.9997

**Artifact:** Research_Status.md, locked-in finding 2 (line 27):
> "GBM hit r approximately 0.957 to 1.000 across all three engineered held-out targets via functional proxy reconstruction. Exhibit: employment_count_10min reconstructed employment_walk_effective_access at GBM importance 0.9997."

**Source data:** `output/m3_n20_baselines/summary/baseline_summary_stats.csv` (per SESSION_LOG.md 2026-05-03, M3 entry). This output directory exists and contains the three-target GBM results.

**Measurement definition:** 5-fold OOF GradientBoostingRegressor per tract, target feature dropped from inputs, all other 72 features as predictors. Median per-tract Pearson r reported. GBM importance is the single-feature share of total gain attributable to `employment_count_10min` when predicting `employment_walk_effective_access`.

---

## P14 -- Feature-survival screening thresholds and surviving counts per architecture

**Value:**
- cv threshold: 0.10
- min_tracts: 12
- GraphSAGE surviving features: 0
- GCN-GAT surviving features: 14

**Artifact:** `results/rank_consistency_n20/summary.txt`

**Full summary.txt excerpt:**
```
section A (GraphSAGE only):              0
section B (GCN-GAT only):                14
section C (both, same sign):             0
section D (both, sign-flip) [HEADLINE]:  0
```

**Command used to generate:**
```bash
python scripts/within_tract_rank_consistency.py \
    --base-dir output/rank_consistency_run \
    --cv-threshold 0.10 \
    --min-tracts 12 \
    --min-addresses 50 \
    --results-dir results/rank_consistency_n20
```
(From `/workspaces/GRANITE/scripts/run_n20_full_pipeline.sh`, line 101)

**Definition of consistent_flag:** sign_test_p < 0.05 AND median |rho| >= 0.10. Feature must appear as signal-bearing in at least min_tracts=12 tracts (n_tracts_signal_bearing >= 12). Signal-bearing requires coefficient of variation cv >= cv_threshold=0.10 on the feature column within the tract.

**Comparison to expectation:** Expected cv=0.10, min_tracts=12, GraphSAGE=0, GCN-GAT=14. MATCH.

**GCN-GAT surviving features (Section B):**
grocery_walk_effective_access, grocery_count_10min, employment_walk_effective_access, employment_count_10min, grocery_count_5min, grocery_percentile, employment_percentile, healthcare_min_time, healthcare_percentile, healthcare_count_5min, log_appvalue, healthcare_modal_access_gap, healthcare_car_effective_access, healthcare_transit_dependence

---

## P15 -- Graph construction constants: road-network pass parameters, fallback k, zero cross-tract edges

**Values:**
- Road snap distance: 500 m (address must be within 500 m of road node to be snapped)
- Candidate pair Euclidean filter: < 1000 m
- Road-network shortest-path filter: < 1500 m
- Road edge weight: `1 / (1 + path_length / 500)`
- Geographic fallback: k=6, Euclidean < 1000 m, weight `exp(-distance_m / 300)`
- Cross-tract edges: zero by construction (graph is built per-tract from single-tract address set)

**Artifact:** `granite/data/loaders.py`

**Key code references:**

Road snap (line 1288): `if distances[i] < 500:`

Euclidean filter (line 1332): `if geo_distance < 1000:`

Road path filter (line 1339): `if path_length > 1500: continue`

Road edge weight (line 1342): `weight = 1.0 / (1.0 + path_length / 500.0)`

Geographic fallback parameters (line 1361, `_create_geographic_edges`):
- `max_neighbors=6`
- `if distance_m < 1000:`
- `weight = np.exp(-distance_m / 300.0)`

Cross-tract confirmation: `get_addresses_for_tract()` (line 707) filters using `all_addresses[all_addresses.geometry.within(tract_geom)]` -- the graph is built from the single-tract address GeoDataFrame only. No mechanism exists for edges to span tracts.

**Research_Status.md summary (lines 47-48):**
> "Road-network edges: 500 m road snap, candidate pairs filtered to less than 1000 m Euclidean and less than 1500 m road-network shortest-path, weight 1 / (1 + path_length / 500). Geographic edges: k=6, less than 1000 m Euclidean filter, weight exp(-distance_m / 300). Symmetric, deduplicated. No feature-similarity edges."

---

## P16 -- Provenance of block-group SVI reference

**Construction:** ACS 5-year estimates (2020) fetched from the Census API for all US block groups across all 50 states plus DC. Raw counts are computed into 11 rate variables (EP_MHI, EP_PCI, EP_UNEMP, EP_NOHSDP, EP_AGE65, EP_AGE17, EP_SNGPNT, EP_MINRTY, EP_MUNIT, EP_MOBILE, EP_CROWD, EP_NOVEH) across 4 CDC SVI themes. Rates are ranked nationally by percentile (ascending for vulnerability) and averaged across themes to produce a composite SVI in [0,1]. This is derived independently from ACS components, not from the CDC SVI file and not from pipeline predictions.

**Code:** `granite/data/block_group_loader.py`
- `fetch_national_acs_data()` (line 491): fetches all US BGs from Census API, caches to `data/processed/national_bg_acs_raw.csv`
- `compute_national_svi()` (line 572): calls `_compute_demographic_rates()` and `_compute_block_group_svi()`, caches to `data/processed/national_bg_svi.csv`
- `_compute_block_group_svi()` (line ~420): ranks via `.rank(ascending=True, pct=True, na_option='keep')`

**Cache files:**
- `data/processed/national_bg_acs_raw.csv`: 242,335 rows (242,336 lines including header)
- `data/processed/national_bg_svi.csv`: 242,335 rows (242,336 lines including header)

**Scope note:** 242,335 total US block groups; 239,346 have complete SVI (svi_complete=True). Hamilton County has 73 block groups in this file (GEOID startswith '47065'); the M0 run used 69 BGs (after min-10 address filter). `svi_ranking_scope='national'` is the production mode used in M0.

**ACS variables fetched (11 EP_ indicators):**
```
B19013_001E (median_household_income -> EP_MHI)
B19301_001E (per_capita_income -> EP_PCI)
B23025_003/5E (unemployment -> EP_UNEMP)
B15003 series (education no HS diploma -> EP_NOHSDP)
B01001 series (age 65+ -> EP_AGE65; age <18 -> EP_AGE17)
B11012_010E (single-parent households -> EP_SNGPNT)
B03002_001/002E (race/ethnicity -> EP_MINRTY)
B25032 series (housing type -> EP_MUNIT, EP_MOBILE)
B25014 series (crowding -> EP_CROWD)
B25044 series (vehicle access -> EP_NOVEH)
```

---

## P17 -- Metadata concordance

Files searched: session_log.md (case-insensitive), research_status.md (case-insensitive), results.md (case-insensitive), config.yaml, dissertation_proposal_framework.md, any file with "manifest" in its name, any file with "roadmap" in its name, any file with "section5" or "section_5" in its name.

**Files found:**

| File | Repo Path |
|------|-----------|
| SESSION_LOG.md | `/workspaces/GRANITE/SESSION_LOG.md` |
| Research_Status.md | `/workspaces/GRANITE/Research_Status.md` |
| RESULTS.md (m0 parity) | `/workspaces/GRANITE/data/results/m0_n20_svi_parity/RESULTS.md` |
| RESULTS.md (5b topology) | `/workspaces/GRANITE/experiments/ablation/05b_topology_specificity/RESULTS.md` |
| RESULTS.md (mehdi review copy) | `/workspaces/GRANITE/for_mehdi_review/m0_n20_svi_parity/RESULTS.md` |
| config.yaml | `/workspaces/GRANITE/config.yaml` |
| dissertation_proposal_framework.md | NOT FOUND |
| manifest files | NOT FOUND (none with "manifest" in name prior to this run) |
| roadmap files | NOT FOUND |
| section5 / section_5 files | NOT FOUND (prior to this run) |

**Detailed metadata per file:**

### SESSION_LOG.md
- **Path:** `/workspaces/GRANITE/SESSION_LOG.md`
- **Bytes:** 97,124
- **Lines:** 1,276
- **Normalized SHA-256:** `e10c30bd3c6ce95a3072795c3f9c358f01252ab3951b6d516de48c381cd33553`
- **Final two lines:**
  ```

  **Cache invalidation:** none.
  ```
- **Last commit:** `b6f6b608a35c244a60d2fca047fb6b4b79db4b4f 2026-07-06 03:06:26 +0000 5b: correct RESULTS.md to fixed estimator values, retire 0.446 and 0.083`

### Research_Status.md
- **Path:** `/workspaces/GRANITE/Research_Status.md`
- **Bytes:** 32,743
- **Lines:** 441
- **Normalized SHA-256:** `0a5e5dcc6dc10e299485a85974135e460e2cc626f0b63d6ee4165d9d315f1696`
- **Final two lines:**
  ```

  PROJECT_PREAMBLE.md line 27 (Jamion quote) held verbatim, pending strategic review of coherence precision.
  ```
- **Last commit:** `b6f6b608a35c244a60d2fca047fb6b4b79db4b4f 2026-07-06 03:06:26 +0000 5b: correct RESULTS.md to fixed estimator values, retire 0.446 and 0.083`

### RESULTS.md (m0 parity -- canonical)
- **Path:** `/workspaces/GRANITE/data/results/m0_n20_svi_parity/RESULTS.md`
- **Bytes:** 2,399
- **Lines:** 53
- **Normalized SHA-256:** `774d9d8df03accba9996f5bb1220b1d5142f9585ce99218b6c8121e4499ce135`
- **Final two lines:**
  ```

  GRANITE constraint error reflects the soft-loss training penalty; values above 5% indicate the constraint was not well-enforced for that tract.
  ```
- **Last commit:** `5859ed0d77b7a5be374a9d192f14c993448b6b4e 2026-06-24 00:47:36 +0000 Track m0 parity results; anchor negations for trapped result CSVs`

### RESULTS.md (5b topology specificity)
- **Path:** `/workspaces/GRANITE/experiments/ablation/05b_topology_specificity/RESULTS.md`
- **Bytes:** 7,520
- **Lines:** 151
- **Normalized SHA-256:** `621a55ae7f8f2c0af0cc3221e64361d7be6a0adfe9435e14f2365c185e558f55`
- **Final two lines:**
  ```
  single-exhibit form of Finding 6: pooled BG r is invariant to within-tract
  quality by construction.
  ```
- **Last commit:** `dc419adc972dcec601a249f5c554d1e17a76473b 2026-07-06 03:12:09 +0000 5b: append Interpretation section to RESULTS.md`

### RESULTS.md (for_mehdi_review copy)
- **Path:** `/workspaces/GRANITE/for_mehdi_review/m0_n20_svi_parity/RESULTS.md`
- **Bytes:** 2,399
- **Lines:** 53
- **Normalized SHA-256:** `774d9d8df03accba9996f5bb1220b1d5142f9585ce99218b6c8121e4499ce135`
- **Note:** Byte-identical to canonical m0 parity RESULTS.md (same sha256). This is an intentional copy bundled for Mehdi review.
- **Last commit:** `3711b40d98a799ebf55ff2352c0bbec8bad65ef5 2026-05-13 00:41:25 +0000 Add for_mehdi_review/ bundle with M0, M2, and M3 result artifacts`

### config.yaml
- **Path:** `/workspaces/GRANITE/config.yaml`
- **Bytes:** 5,410
- **Lines:** 148
- **Normalized SHA-256:** `e2f43745a92ac31dbbda34c27f6015ec8816e17f05dee76cc0d4b07e5fd0be0b`
- **Final two lines:**
  ```
    # base directory for recovery outputs (overridden by --output CLI flag)
    output_dir: "./output/recovery"
  ```
- **Last commit:** `49a052b024fc5525702a4514d08b775cac1017de 2026-06-10 21:29:22 +0000 step05b: road_knn_k=9 calibrates road_network_uniform degree to spatial for honest topology parity`

**Missing files:** dissertation_proposal_framework.md -- NOT FOUND anywhere in repo. No manifest files found prior to this run. No roadmap files found. No section5/section_5 files found prior to this run.

---

## P18 -- Canonical trail uniqueness

**Command:**
```bash
find /workspaces/GRANITE -iname "*session_log*" -o -iname "*research_status*" -o -iname "*roadmap*"
```

**Full output:**
```
/workspaces/GRANITE/Research_Status.md
/workspaces/GRANITE/SESSION_LOG.md
```

**Confirmation:** Exactly one SESSION_LOG.md and exactly one Research_Status.md exist. No roadmap files found.

**Last 5 dated SESSION_LOG.md entries:**

1. `## 2026-07-06: 5b provenance sidecar and RESULTS.md value purge`
2. `## 2026-07-06: 5b topology specificity re-sweep under fixed Moran's I estimator`
3. `## 2026-07-05: M6 prediction files relocated to committed path`
4. `## 2026-06-26 -- M6 ceiling coordinate fix: UTM from generator frame`
5. `## 2026-06-25 -- M6 runner: ceiling_gbm method added`

---

## Summary of Missing Items

| Item | Status | Notes |
|------|--------|-------|
| P1 | FOUND | 87 tracts total, 85 with SVI |
| P2 | FOUND | 39,535 (n20); 102,761 full county |
| P3 | FOUND | 39,476/39,535 (99.9%) via SESSION_LOG |
| P4 | FOUND | 73 features |
| P5 | FOUND | 0.769 [0.660, 0.853] -- MATCHES expectation |
| P6 | FOUND | 0.802 [0.712, 0.871] -- MATCHES expectation |
| P7 | FOUND | 0.768 [0.652, 0.858] -- MATCHES expectation |
| P8 | FOUND | 69 BGs |
| P9 | FOUND | 0.390 -- MATCHES expectation |
| P10 | FOUND | 0.787 -- MATCHES expectation |
| P11 | FOUND | -0.121 [-0.536, 0.108], spans zero |
| P12 | MISSING | r=0.033 removed as ungrounded 2026-06-09; no address-level SVI ground truth exists |
| P13 | FOUND | 0.957-1.000 range; exhibit pair employment_count_10min / employment_walk_effective_access at importance 0.9997 |
| P14 | FOUND | cv=0.10, min_tracts=12, GraphSAGE=0, GCN-GAT=14 -- MATCHES expectation |
| P15 | FOUND | Road snap 500m, Euclidean <1000m, path <1500m, geo fallback k=6 <1000m; zero cross-tract edges |
| P16 | FOUND | ACS 5-year 2020 estimates, 11 variables, 4 themes, nationally ranked percentiles |
| P17 | FOUND | 6 files found; dissertation_proposal_framework.md MISSING; no manifest/roadmap/section5 files prior to this run |
| P18 | FOUND | Exactly 1 SESSION_LOG.md, 1 Research_Status.md; no roadmap files; no duplicates |

---

## Value Disagreement Check

Comparing observed values against stated expectations:

| Item | Expected | Observed | Status |
|------|----------|----------|--------|
| P5 CGD pooled BG r | 0.769 [0.660, 0.853] | 0.7692 [0.6602, 0.8527] | MATCH (rounding) |
| P6 Dasymetric pooled BG r | 0.802 [0.712, 0.871] | 0.8018 [0.7118, 0.871] | MATCH |
| P7 Pycnophylactic pooled BG r | 0.768 | 0.7678 | MATCH |
| P9 CGD per-tract median r | 0.390 | 0.3901 | MATCH |
| P10 Dasymetric per-tract median r | 0.787 | 0.7867 | MATCH |
| P12 accessibility address-level r | 0.033 | MISSING (removed as ungrounded) | MISMATCH -- value does not exist in committed artifacts |
| P14 cv/min_tracts/SAGE/GAT | 0.10/12/0/14 | 0.10/12/0/14 | MATCH |

**P12 disagreement note:** The expectation value of 0.033 refers to a claim removed from all documentation on 2026-06-09 because no committed artifact supports it. The value appeared in `Research_Status.md` (removed) and `docs/FEATURES.md` (removed). The experiment infrastructure exists in `scripts/coord_artifact_experiment.py` but has not been rerun to produce a committed artifact. This is a known gap documented in `experiments/audits/outstanding_items_reconciliation.md`.
