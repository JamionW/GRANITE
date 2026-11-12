# Paper 1 number-to-artifact trace (TRACE-P1)

Read-only recompute pass over every number Paper 1 may report. `CONVERSATION_PROTOCOL.md` and
`IMPORTS.md` do not exist in this repo (searched by name and case-insensitive glob); not read.
`SESSION_LOG.md` (tail), `Research_Status.md`, and the 2026-09-18 proposal trace
(`docs/reviews/proposal_variance_components_trace.md`) plus `data/results/variance_components/`
(commit `73e8638`) were read first; that trace's verdicts are reused below only where
re-verified against HEAD in this pass (all five of its claims were re-recomputed fresh, not
copied).

**TRACE-P1-CLOSE update.** A4, A5, A6, A7, A11, A12 re-verdicted and A14 added below against
three new committed artifacts: `experiments/audits/graph_edge_audit/` (B1, commit `3bcdf7d`),
`scripts/power_analysis_parity.py` rank-based sensitivity addition +
`data/results/m0_n20_svi_parity/power_analysis.json` (B2, commit `409dba0`), and
`data/results/bg_reference_manifest/` (B3, commit `1ae6905`). See RECON-P1c
(`/tmp/recon_p1c/RECON-P1c.md`, not committed -- scratch) for the read-only recon that produced
the B1 script and the Q1-Q3 findings behind these re-verdicts.

Docs convention: trace-style documents live at `docs/reviews/*_trace.md` (e.g.
`proposal_variance_components_trace.md`, `section4_5_figure_trace.md`); milestone-topic docs
live at `docs/proposal/`. Neither existing path conflicts with the milestone-specified
`docs/paper1/number_trace.md` (no `docs/paper1/` existed before this commit), so it was created
as specified.

## Claim trace table

| ID | Claim | Artifact path | Commit | Producing script | Recompute command (see Trace N below) | Measured value | Verdict |
|---|---|---|---|---|---|---|---|
| A1 | Pooled BG r 69 BGs: 0.769/0.802/0.768; CIs overlap | `data/results/m0_n20_svi_parity/aggregate.csv` | `de772b7` | `granite/scripts/run_m0_parity.py` | Trace 1 | GRANITE 0.7692 [0.660,0.853]; Dasymetric 0.8018 [0.712,0.871]; Pycno 0.7678 [0.652,0.858]; all three CIs overlap | CONFIRMED |
| A2 | Per-tract median BG r 19 tracts: 0.390/0.787/0.208 | `data/results/m0_n20_svi_parity/aggregate.csv` | `de772b7` | `granite/scripts/run_m0_parity.py` | Trace 1 | GRANITE 0.3901, Dasymetric 0.7867, Pycnophylactic 0.2078 | CONFIRMED |
| A3 | Paired diff vs Dasymetric: median -0.121, mean -0.270, bootstrap 95% CI [-0.536,0.108]; n resamples, seed | `data/results/m0_n20_svi_parity/pairwise_diffs.csv`, `.../per_tract.csv` | `de772b7` | `granite/scripts/run_m0_parity.py` (bootstrap), `scripts/power_analysis_parity.py` (mean/sd) | Trace 2 | median -0.1207, CI [-0.5356,0.108], n_boot=1000, seed=42; mean -0.2699 | CONFIRMED |
| A4 | 20 tracts, 39,535 addresses, 39,476 parcel-joined (99.9%) | `experiments/recovery/per_address_predictions/granite_m0.parquet` (addresses/tracts); parcel-join count has none | `10d0ee8` | `scripts/persist_per_address_predictions.py` | Trace 3 | 39,535 addresses, 20 tracts CONFIRMED; 39,476/99.9% parcel-join has no committed source | CONFIRMED (addresses, tracts) / **DROPPED** (parcel-join count removed from Paper 1 -- TRACE-P1-CLOSE; no committed artifact exists or was created to back it, and the underlying `parcels.shp` is gitignored by design, not by omission, so this is not expected to become traceable) |
| A5 | CDC method, 12 ACS indicators / 4 themes, ranked over 242,335 BGs | methodology: `granite/data/block_group_loader.py`; counts: `data/results/bg_reference_manifest/manifest.json` (B3) | methodology `HEAD`-tracked; manifest `1ae6905`; source CSVs still gitignored | `granite/data/block_group_loader.py`; `scripts/bg_reference_manifest.py` | Trace 4 | 4+3+1+4=12 variables across 4 `theme_vars` keys (code, CONFIRMED); fetched 242,335, but only **233,338** carry a non-null ranked `SVI` -- not 242,335 and not the previously-cited 239,346 (`svi_complete`, a 4-of-12-variable flag; 6,008 BGs pass it with `SVI` still `NaN`) | **CORRECTED** (methodology CONFIRMED; "ranked over 242,335" must become "233,338 of 242,335 fetched BGs carry a non-null ranked SVI"; row counts now sha256-anchored via B3 manifest, but source CSVs remain UNTRACKED-ONLY) |
| A6 | 5b: structured Moran's I 0.817-0.885; randomized SAGE 0.422; GCN-GAT 0.075; pooled BG r 0.763-0.772 all conditions | `data/results/m6_topology_5b/topology_specificity_metrics.json`; matched-pair degree table: `experiments/ablation/05b_topology_specificity/degree_calibration.md` and `experiments/audits/graph_edge_audit/graph_edge_audit.json` (B1) | `04f453e`; degree_calibration.md `a8e64de`; B1 `3bcdf7d` | `experiments/ablation/05b_topology_specificity/` sweep; `experiments/audits/graph_edge_audit/graph_edge_audit.py` | Trace 5 | structured mean range [0.8167, 0.8845]; randomized sage 0.4217, gcn_gat 0.0745; pooled_bg_r range [0.7628, 0.7716]. **Matched pair**: `spatial_knn_uniform` and `randomized` share identical mean degree 11.742 (232,109 undirected edges each, B1-reconstructed), i.e. the Moran's I gap (0.873/0.885 structured vs 0.422/0.075 randomized) is attributable to edge placement alone, not to degree | CONFIRMED (now also independently reproduced via the committed B1 graph audit, not just the frozen 5b metrics JSON) |
| A7 | Deck slide 13: median output Moran's I 0.94, coordinates-only, recovery near zero | `data/results/m6_recovery_grid/recovery_grid.csv` | `d01f5e4` | `experiments/m6_recovery_grid/run_grid.py` | Trace 6 | Row-level median over 1,620 sage/coordinates_only **draw-by-tract rows** = 0.988, NOT 0.94. Median, across the 81 draws, of each draw's own across-tract mean = 0.9407. recovery_r median (row-level) = 0.00088 | CORRECTED (0.94 = the median across the 81 draws of each draw's mean Moran's I across its 20 tracts, not a flat median over the 1,620 **draw by tract rows** -- each `recovery_grid.csv` row is one (draw, tract) pair, not one address, so the earlier "address-level rows" phrasing was also imprecise and is corrected here) |
| A8 | Methods within 0.034 of each other (derived from A1) | same as A1 | `de772b7` | — | Trace 1 | max(0.8018) - min(0.7678) = 0.0340 | CONFIRMED |
| A9 | Condition-mean = how many runs (seeds x rewiring draws), seed values | `data/results/m6_topology_5b/topology_specificity_metrics.json` | `04f453e` | 05b sweep | Trace 5 | 5 runs per condition-arch cell. Structured conditions: training_seed in [42,17,123,2024,7], graph_draw_seed fixed=42. randomized: training_seed fixed=42, graph_draw_seed in [42,17,123,2024,7] | CONFIRMED |
| A10 | Moran's I weights: fixed external k=8 row-standardized binary k-NN, separate from message-passing graph | `granite/evaluation/spatial_diagnostics.py:39-52` | tracked at `HEAD` | — | Trace 7 | `compute_spatial_autocorrelation(..., k_neighbors=8)` uses `libpysal.weights.KNN(..., k=8)`, `.transform='r'`, `esda.Moran`; independent of `granite/data/loaders.py` road-network/geographic edge builder | CONFIRMED |
| A11 | Graph: per-tract road-network hybrid, k=6 Gaussian fallback, mean degree 12.162 over 39,535 nodes, 0/480,808 cross-tract edges | `experiments/audits/graph_edge_audit/graph_edge_audit.json` (B1) | `3bcdf7d` | `experiments/audits/graph_edge_audit/graph_edge_audit.py` | Trace 8 | production: 39,535 nodes, 480,808 directed `edge_index` rows = **240,404 undirected edges**, mean degree 12.161578 (matches 12.162), 0 cross-tract edges, all 20 tracts `both_dir_ok=True` | **CONFIRMED, re-verdicted from UNTRACKED-ONLY** via the now-committed B1 artifact (independently reproduced from the shipped builder, not merely re-citing the still-gitignored `claude_XT_Edge_Audit.md`). Wording fix: "480,808 edges" must read "480,808 directed edge entries (240,404 undirected edges)" -- every edge is stored in both directions (`loaders.py:1009,1101,1114,1169,1211`, no `to_undirected` call) |
| A12 | Power analysis: SD 0.92, d 0.29, power 22%@19 / 76%@85, MDE 0.63, CI@85 [-0.47,-0.07], ~94 tracts for 80%; rank-based sensitivity exists | `data/results/m0_n20_svi_parity/per_tract.csv`, `data/results/m0_n20_svi_parity/power_analysis.json` (B2) | `de772b7`; B2 `409dba0` | `scripts/power_analysis_parity.py` (`bbaa5ff`, rank-based addition at `409dba0`) | Trace 2 | sd=0.9231, d=0.2924, power 22.7%/76.0%, MDE@19=0.627, CI@85=[-0.47,-0.07], n for 80%=94 (all CONFIRMED, unchanged); Wilcoxon signed-rank (two-sided, 2 zero-diffs dropped, n=17): statistic=46.0, p=0.1488; exact sign test (n_pos=6, n_neg=11): p=0.3323 | **CONFIRMED, re-verdicted from UNTRACED** -- the rank-based sensitivity analysis now exists (B2) and is null at n=19, consistent with the mean-based power analysis's own CI spanning zero |
| A13 | Dasymetric: `baselines.py:84`, `tract_svi + alpha*(impervious_pct - tract_mean)`, alpha largest value keeping [0,1] | `granite/evaluation/baselines.py` | tracked at `HEAD` | — | Trace 9 | Class `DasymetricDisaggregation` at line 84; `dev = a - a.mean()` then `predictions = tract_svi + alpha * dev` at line 122, `alpha = min(max_up, max_down)` chosen to keep output in [0,1] (lines 93-123) | CONFIRMED (logic and class line; additive-form line is 122, not 84) |
| A14 | Moran's I scope: computed per tract then aggregated, or pooled | `experiments/ablation/05b_topology_specificity/run_sweep.py`; `experiments/m6_recovery_grid/run_grid.py`; `granite/scripts/run_m0_parity.py` | 5b sweep tracked at `HEAD`; m6 `d01f5e4` | `run_sweep.py:_compute_morans_i`/`run_seed`; `run_grid.py:_compute_morans_i_true`/`_compute_morans_i_output` | Trace 11 | **5b**: per-tract Moran's I computed per tract (`run_sweep.py:516`), filtered to finite values (`:519-520`), then **unweighted arithmetic mean** across tracts per seed (`run_sweep.py:531`, `morans_i_mean = float(np.mean(per_tract_morans))`) -- not pooled, not weighted by tract size. **m6**: per-(draw,tract) Moran's I stored as one `recovery_grid.csv` row each (`run_grid.py:699`); no in-script aggregation -- downstream median-of-draw-means (see A7) is computed by the analysis consumer, not by `run_grid.py` itself. **m0**: `run_m0_parity.py` computes **no Moran's I at all** -- zero call sites for `compute_spatial_autocorrelation` or any Moran's-I helper; M0's pooled/per-tract BG r path does not touch spatial autocorrelation | CONFIRMED (none of the three paths pool Moran's I across tracts; m0 computes none) |
| B1 | M6 grid: 81 draws, 27 cells, seeds 42/17/123, 20 tracts, 7,200 rows | `data/results/m6_recovery_grid/recovery_grid.csv` | `d01f5e4` | `experiments/m6_recovery_grid/run_grid.py` | Trace 6 | rows 7200, draws 81, cells 27, seeds [17,42,123], tracts 20 | CONFIRMED |
| B2 | Coordinates: 97.86% / 97.88% between-tract, eta_sq 0.979 | `experiments/ecological_fallacy/variance_decomposition.csv` | `8aecdc1` | `scripts/variance_decomposition.py` | Trace 10 | lat eta_sq=0.97858187, lon eta_sq=0.97882321 | CONFIRMED |
| B3 | Between-tract variance share vs SVI coupling: Pearson -0.052, n=75 | `experiments/ecological_fallacy/variance_decomposition.csv` | `8aecdc1` | `scripts/variance_decomposition.py` | Trace 10 | pearsonr(eta_sq, abs(tract_svi_r)) = -0.0518, n=75 | CONFIRMED |

Note: table IDs `B1`-`B3` (Tier B context claims) and artifact names `B1`/`B2`/`B3` (TRACE-P1-CLOSE
Part B commits) are two separate numbering schemes that happen to share labels; context
disambiguates (Tier B rows are in this table; TRACE-P1-CLOSE B1/B2/B3 are the three commits
referenced in the "Verdict" column above and the SESSION_LOG entry below).

---

## Trace outputs

### Trace 1: A1/A2/A8 pooled and per-tract BG r

```
$ python3 -c "
import pandas as pd
agg = pd.read_csv('data/results/m0_n20_svi_parity/aggregate.csv')
print(agg.to_string())
"
          method  median_bg_r  ci_low_95  ci_high_95  n_tracts  pooled_bg_r  pooled_bg_r_ci_low  pooled_bg_r_ci_high  pooled_n_bgs
0        GRANITE       0.3901    -0.4452      0.6974        19       0.7692              0.6602               0.8527            69
1     Dasymetric       0.7867     0.2531      0.8627        19       0.8018              0.7118               0.8710            69
2 Pycnophylactic       0.2078    -0.3526      0.5289        19       0.7678              0.6516               0.8579            69

$ python3 -c "print(round(0.8018-0.7678,4))"
0.034
```

All three pooled-r CIs overlap: [0.660,0.853] / [0.712,0.871] / [0.652,0.858].

### Trace 2: A3/A12 paired difference and power

```
$ grep -n "n_boot=1000\|_bootstrap_diff_ci(paired" granite/scripts/run_m0_parity.py
146:def _bootstrap_median_ci(vals, n_boot=1000, seed=42):
159:def _bootstrap_diff_ci(a_vals, b_vals, n_boot=1000, seed=42):
175:def _bootstrap_pooled_r_ci(all_gdfs, all_preds_by_method, validator, bg_gdf, n_boot=1000, seed=42):
427:        n_boot=1000, seed=SEED,
466:        obs, lo, hi = _bootstrap_diff_ci(paired['ra'].values, paired['rb'].values,
36:SEED = 42

$ python3 -c "
import pandas as pd
pw = pd.read_csv('data/results/m0_n20_svi_parity/pairwise_diffs.csv')
print(pw.to_string())
"
                   pair  obs_median_diff  ci_low_95  ci_high_95  n_pairs  separable
0 granite_vs_dasymetric          -0.1207    -0.5356      0.1080       19      False
1      granite_vs_pycno           0.0163    -0.1068      0.2067       19      False
2   dasymetric_vs_pycno           0.4027     0.0435      0.6398       19       True

$ python3 scripts/power_analysis_parity.py
artifact              data/results/m0_n20_svi_parity/per_tract.csv
tracts (>= 2 bgs)      19
mean paired diff      -0.2699
median paired diff    -0.1207
sd of differences     0.9231
cohen's d (mean)      -0.2924  |d| = 0.2924
power at n=19         22.7 %
power at n=85         76.0 %
mde at n=19, 80% pwr  0.627  (d=0.680)
n for 80% power       94
projected 95% CI at n=85, on the mean effect   [-0.47, -0.07]  width 0.39

$ grep -rl "rank-based\|rank.based" --include=*.py --include=*.md .
(no output -- no rank-based sensitivity analysis script exists in the repo)
```
(TRACE-P1 finding; **superseded below**.)

**TRACE-P1-CLOSE addition (B2, commit `409dba0`):** `scripts/power_analysis_parity.py` now adds
a two-sided Wilcoxon signed-rank test and an exact sign test in place, writing all output to
`data/results/m0_n20_svi_parity/power_analysis.json`:

```
$ python3 scripts/power_analysis_parity.py
...
wilcoxon signed-rank (two-sided)  statistic=46.0000  p=0.1488  n_nonzero=17  (n_zero_dropped=2)
exact sign test (two-sided)       n_pos=6  n_neg=11  p=0.3323

wrote data/results/m0_n20_svi_parity/power_analysis.json
```
Wilcoxon drops 2 exact-zero paired differences (ties in `bg_r` at the csv's 4-decimal
precision), leaving n=17 nonzero differences. Both rank-based tests are null at n=19/17 (p=0.149,
p=0.332), consistent with the mean-based analysis's own CI spanning zero at n=19 -- the
rank-based result does not contradict the mean-based one, it corroborates "underpowered, not yet
separable" rather than adding new separating evidence.

### Trace 3: A4 address/tract/parcel counts

```
$ python3 -c "
import pandas as pd
df = pd.read_parquet('experiments/recovery/per_address_predictions/granite_m0.parquet')
print('rows', len(df)); print('n_tracts', df['fips'].nunique())
"
rows 39535
n_tracts 20

$ git log --oneline -1 -- experiments/recovery/per_address_predictions/granite_m0.parquet
10d0ee8 Persist per-address predictions for n20, provenance-anchored
```

39,476/39,535 (99.9%) parcel-join: only source is `SESSION_LOG.md` prose (2026-06-10 entry,
"Match rate: 39,476/39,535 (99.9%)") and `docs/proposal/section5_verification_manifest.md:56-61`,
both markdown, not artifacts. The underlying join (`gpd.sjoin` against `data/raw/parcels/parcels.shp`)
is not persisted anywhere as a committed per-address match-flag column, and `data/raw/parcels/`
is gitignored (`.gitignore:52`). Per the rule against taking values from markdown, and with no
committed artifact to recompute from: **UNTRACED** (TRACE-P1). **TRACE-P1-CLOSE: DROPPED from
Paper 1.** No artifact was created to back this count (none was requested), and the gap is not
expected to close later: `data/raw/parcels/` is gitignored by design (per `.gitignore:52`,
consistent with `CLAUDE.md`'s "`data/raw/` ... Not in git"), so persisting a committed
per-address match-flag column would require a deliberate scope change, not a missed trace.

### Trace 4: A5 national BG reference

```
$ sed -n '424,429p' granite/data/block_group_loader.py
theme_vars = {
    'theme1_socioeconomic': ['EP_MHI', 'EP_PCI', 'EP_UNEMP', 'EP_NOHSDP'],
    'theme2_household': ['EP_AGE65', 'EP_AGE17', 'EP_SNGPNT'],
    'theme3_minority': ['EP_MINRTY'],
    'theme4_housing': ['EP_MUNIT', 'EP_MOBILE', 'EP_CROWD', 'EP_NOVEH'],
}
```
4 + 3 + 1 + 4 = 12 variables, 4 themes. Code is tracked at `HEAD`.

```
$ git ls-files data/processed/national_bg_svi.csv
(no output)
$ git check-ignore -v data/processed/national_bg_svi.csv
.gitignore:53:data/processed/	data/processed/national_bg_svi.csv
$ wc -l data/processed/national_bg_svi.csv
242336 data/processed/national_bg_svi.csv   # header + 242335 data rows
```
File exists on disk and the row count matches the claim, but it is gitignored and was never
committed -- **UNTRACKED-ONLY** (TRACE-P1).

**TRACE-P1-CLOSE correction (B3, `data/results/bg_reference_manifest/manifest.json`):**

```
$ python3 -c "
import pandas as pd
svi = pd.read_csv('data/processed/national_bg_svi.csv')
print('total rows', len(svi))
print('SVI non-null (actually ranked)', svi['SVI'].notna().sum())
print('svi_complete=True (4-var flag)', (svi['svi_complete']==True).sum())
print('svi_complete=True but SVI is NaN', ((svi['svi_complete']==True) & svi['SVI'].isna()).sum())
"
total rows 242335
SVI non-null (actually ranked) 233338
svi_complete=True (4-var flag) 239346
svi_complete=True but SVI is NaN 6008
```

`svi_complete` (`block_group_loader.py:474-475`) checks only 4 of the 12 `EP_*` variables
(`EP_MHI, EP_UNEMP, EP_NOHSDP, EP_NOVEH`); `themes_sum` (`block_group_loader.py:462`) uses
`skipna=False` across all 4 themes, so a BG missing any of the other 8 variables still gets a
`NaN` final `SVI` despite `svi_complete=True`. **242,335 is the fetched row count, not the
ranked count; 239,346 is a weaker completeness flag than what the ranking actually requires; the
count that is actually ranked (non-null `SVI`) is 233,338.** This is now the canonical count,
sha256-anchored in `data/results/bg_reference_manifest/manifest.json` (B3, commit `1ae6905`).
The source CSVs themselves (`national_bg_acs_raw.csv`, `national_bg_svi.csv`) remain gitignored
and were not committed by B3 -- only the manifest's hashes and derived counts are committed.

### Trace 5: A6/A9 topology specificity (5b)

```
$ python3 -c "
import json
d = json.load(open('data/results/m6_topology_5b/topology_specificity_metrics.json'))
for cond in d:
    for arch in d[cond]:
        seeds = d[cond][arch]['seeds']
        print(cond, arch, 'n_seeds=', len(seeds),
              [(s['training_seed'], s['graph_draw_seed']) for s in seeds],
              'mean morans_i=', d[cond][arch]['mean']['morans_i'],
              'mean pooled_bg_r=', d[cond][arch]['mean']['pooled_bg_r'])
"
spatial_knn_uniform sage n_seeds= 5 [(42,42),(17,42),(123,42),(2024,42),(7,42)] mean morans_i= 0.8723029366180140 mean pooled_bg_r= 0.7634594643593754
spatial_knn_uniform gcn_gat n_seeds= 5 [(42,42),(17,42),(123,42),(2024,42),(7,42)] mean morans_i= 0.8844982987957074 mean pooled_bg_r= 0.7661961498946802
road_network_uniform sage n_seeds= 5 [(42,42),(17,42),(123,42),(2024,42),(7,42)] mean morans_i= 0.8347713198449048 mean pooled_bg_r= 0.7628256920864306
road_network_uniform gcn_gat n_seeds= 5 [(42,42),(17,42),(123,42),(2024,42),(7,42)] mean morans_i= 0.8166878016424070 mean pooled_bg_r= 0.7642542484694809
randomized sage n_seeds= 5 [(42,42),(42,17),(42,123),(42,2024),(42,7)] mean morans_i= 0.4217260271102307 mean pooled_bg_r= 0.7715511374687168
randomized gcn_gat n_seeds= 5 [(42,42),(42,17),(42,123),(42,2024),(42,7)] mean morans_i= 0.0745244497978178 mean pooled_bg_r= 0.7652436395647504
production sage n_seeds= 5 [(42,42),(17,42),(123,42),(2024,42),(7,42)] mean morans_i= 0.8478673308868924 mean pooled_bg_r= 0.7632204058517055
production gcn_gat n_seeds= 5 [(42,42),(17,42),(123,42),(2024,42),(7,42)] mean morans_i= 0.8261198266647423 mean pooled_bg_r= 0.7639487584155458
```
Structured-condition mean range [0.8167, 0.8845] matches "0.817 to 0.885". Randomized sage
0.4217 matches "0.422"; randomized gcn_gat 0.0745 matches "0.075". Pooled BG r across all
eight condition-arch cells ranges [0.7628, 0.7716], matching "0.763 to 0.772". Each
condition-arch mean is over 5 runs; structured conditions vary `training_seed` with
`graph_draw_seed` fixed at 42, randomized fixes `training_seed=42` and varies
`graph_draw_seed`.

### Trace 6: A7/B1 M6 recovery grid

```
$ python3 -c "
import pandas as pd
df = pd.read_csv('data/results/m6_recovery_grid/recovery_grid.csv')
print('rows total', len(df))
print('draws:', df.drop_duplicates(subset=['autocorr','snr','between_tract','seed']).shape[0])
print('cells:', df.drop_duplicates(subset=['autocorr','snr','between_tract']).shape[0])
print('seeds', sorted(df['seed'].unique()))
print('tracts', df['tract_fips'].nunique())
"
rows total 7200
draws: 81
cells: 27
seeds [17, 42, 123]
tracts 20

$ python3 -c "
import pandas as pd
df = pd.read_csv('data/results/m6_recovery_grid/recovery_grid.csv')
sub = df[(df['method']=='granite') & (df['feature_mode']=='coordinates_only') & (df['arch']=='sage')]
print('n rows', len(sub))
print('row-level median morans_i_output', sub['morans_i_output'].median())
print('row-level median recovery_r', sub['recovery_r'].median())
draw_means = sub.groupby(['autocorr','snr','between_tract','seed'])['morans_i_output'].mean()
print('n draws', len(draw_means))
print('median of per-draw means', draw_means.median())
"
n rows 1620
row-level median morans_i_output 0.9882638201106075
row-level median recovery_r 0.0008814611541136999
n draws 81
median of per-draw means 0.9407312808464919
```
The 0.94 figure in the deck is the median, across the 81 draws, of each draw's own mean
`morans_i_output` across its 20 tracts (sage, coordinates_only GRANITE rows) -- **not** the
median taken over the 1,620 individual address-level rows (which is 0.988). `recovery_r`
stays near zero under either aggregation (row-level median 0.00088).

### Trace 7: A10 Moran's I weights

```
$ sed -n '39,52p' granite/evaluation/spatial_diagnostics.py
    def compute_spatial_autocorrelation(self, predictions, coordinates, k_neighbors=8):
        """
        Moran's I with row-standardized k=8 k-NN weights.
        ...
        """
        import libpysal.weights as lw
        import esda as _esda
        n = len(predictions)
        if n < k_neighbors + 1:
            return 0.0
        w = lw.KNN.from_array(np.asarray(coordinates, dtype=float), k=k_neighbors)
        w.transform = 'r'
        return _esda.Moran(np.asarray(predictions, dtype=float), w, permutations=0).I
```
Default `k_neighbors=8`, row-standardized (`transform='r'`), binary k-NN via
`libpysal.weights.KNN`. This weight matrix is independent of the message-passing graph built
in `granite/data/loaders.py` (road-network + geographic k-NN hybrid).

### Trace 8: A11 cross-tract edge audit

```
$ git log --oneline -1 -- scripts/diag/xt_edge_audit.py claude_XT_Edge_Audit.md
(no output)
$ git ls-files scripts/diag/xt_edge_audit.py claude_XT_Edge_Audit.md
(no output)
$ git check-ignore -v scripts/diag/xt_edge_audit.py claude_XT_Edge_Audit.md
.gitignore:158:scripts/diag/	scripts/diag/xt_edge_audit.py
.gitignore:157:claude_XT_Edge_Audit.md	claude_XT_Edge_Audit.md
$ python3 -c "print(480808/39535)"
12.161578348298976
```
Arithmetic in the doc checks out (480,808/39,535 = 12.1616, matches "12.162"), but both the
script and the doc reporting the 0/480,808 cross-tract-edge result are gitignored and have
never been committed (`git log --all` returns nothing for either path). Per the milestone
rule, gitignored/untracked outputs are non-evidence: **UNTRACKED-ONLY** (TRACE-P1).

**TRACE-P1-CLOSE re-verdict (B1, `experiments/audits/graph_edge_audit/`, commit `3bcdf7d`):**

```
$ python3 experiments/audits/graph_edge_audit/graph_edge_audit.py
condition                nodes  directed_edges  undirected_edges   mean_degree both_dir_ok_all cross_tract_edges
production               39535          480808            240404     12.161578            True                 0
spatial_knn_uniform      39535          464218            232109     11.741950            True                 0
road_network_uniform     39535          464152            232076     11.740281            True                 0
randomized               39535          464218            232109     11.741950            True                 0
```
This is an independent reconstruction from the committed builder
(`granite.data.loaders.DataLoader.create_spatial_accessibility_graph`), not merely re-citing the
untracked doc. It confirms the untracked doc's numbers exactly for `production`
(480,808 == directed `edge_index.shape[1]`, 12.161578 mean degree, 0 cross-tract) and resolves
the directed/undirected ambiguity: **480,808 is the directed row count; the undirected edge
count is 240,404** (`loaders.py:1211` appends `[[u,v],[v,u]]` for every edge; no `to_undirected`
call anywhere in the file -- `grep -n "to_undirected" granite/data/loaders.py` is empty). Zero
cross-tract edges holds by construction for all four `graph_variant` conditions, not just
`production`: each graph is built per-tract via one `create_spatial_accessibility_graph` call
per `fips`, with no concatenation step for any variant, so node indices never span more than
one tract. **Re-verdict: CONFIRMED** (was UNTRACKED-ONLY in TRACE-P1; `claude_XT_Edge_Audit.md`
and `scripts/diag/xt_edge_audit.py` remain gitignored and un-cited going forward).

### Trace 9: A13 Dasymetric baseline

```
$ grep -n "class DasymetricDisaggregation\|predictions = tract_svi + alpha \* dev" granite/evaluation/baselines.py
84:class DasymetricDisaggregation(DisaggregationBaseline):
122:            predictions = tract_svi + alpha * dev

$ sed -n '108,123p' granite/evaluation/baselines.py
        a = pd.to_numeric(address_gdf[self.ancillary_column], errors='coerce').fillna(0.0).values
        dev = a - a.mean()
        if np.ptp(dev) < 1e-9 or a.sum() < 1e-9:
            predictions = np.full(n, tract_svi)
        else:
            max_up = (1.0 - tract_svi) / dev.max() if dev.max() > 0 else np.inf
            max_down = tract_svi / abs(dev.min()) if dev.min() < 0 else np.inf
            alpha = min(max_up, max_down)
            predictions = tract_svi + alpha * dev
```
Class starts at line 84 (matches the claim's file:line); the additive formula itself is at
line 122. `alpha` is the largest value keeping all predictions in [0,1] (`min` of the two
one-sided bounds). Logic confirmed.

### Trace 10: B2/B3 feature taxonomy

```
$ python3 -c "
import pandas as pd, scipy.stats as st
df = pd.read_csv('experiments/ecological_fallacy/variance_decomposition.csv')
print('n rows', len(df))
print(df[df['feature'].isin(['lat','lon'])][['feature','eta_sq']])
r,p = st.pearsonr(df['eta_sq'], df['tract_svi_r'].abs())
print('pearson eta_sq vs |tract_svi_r|:', r, 'n=', len(df))
"
n rows 75
  feature    eta_sq
0     lat  0.97858187
1     lon  0.97882321
pearson eta_sq vs |tract_svi_r|: -0.051812229772908516 n= 75
```

### Trace 11: A14 Moran's I scope (per-tract vs. aggregated vs. pooled)

```
$ grep -n "compute_spatial_autocorrelation\|_compute_morans_i" experiments/ablation/05b_topology_specificity/run_sweep.py
160:def _compute_morans_i(predictions, address_gdf, k=8):
163:        return SpatialLearningDiagnostics().compute_spatial_autocorrelation(
516:        morans_i = _compute_morans_i(preds_arr, address_gdf)

$ sed -n '513,531p' experiments/ablation/05b_topology_specificity/run_sweep.py
        constr_err = abs(np.mean(preds_arr) - tract_svi)
        sp_std = float(np.std(preds_arr))
        morans_i = _compute_morans_i(preds_arr, address_gdf)
        per_tract_stds.append(sp_std)
        if math.isfinite(morans_i):
            per_tract_morans.append(morans_i)
        ...
    morans_i_mean = float(np.mean(per_tract_morans)) if per_tract_morans else float('nan')
```
5b computes Moran's I **once per tract** (`:516`, called inside the per-`fips` loop), drops
non-finite values (`:519-520`), then takes an **unweighted arithmetic mean** across the
surviving tracts per seed (`:531`) -- not a pooled all-address computation, and not weighted by
tract size (a 18-address tract and a 3,756-address tract contribute equally to the mean).

```
$ grep -n "compute_spatial_autocorrelation\|morans_i" granite/scripts/run_m0_parity.py
(no output)
```
`run_m0_parity.py` has **zero** call sites for Moran's I or `compute_spatial_autocorrelation` --
confirmed also by its import list (`grep -n "^from\|^import" granite/scripts/run_m0_parity.py`
shows no `spatial_diagnostics` import). M0's pooled/per-tract BG r path never touches spatial
autocorrelation; Moran's I is a 5b/M6-only metric, not an M0 metric.

```
$ grep -n "_compute_morans_i_true\|_compute_morans_i_output" experiments/m6_recovery_grid/run_grid.py
214:def _compute_morans_i_true(addr_df, fips):
222:        return float(_SPATIAL_DIAG.compute_spatial_autocorrelation(vals, xy, k_neighbors=8))
227:def _compute_morans_i_output(predictions, address_gdf):
233:        return float(_SPATIAL_DIAG.compute_spatial_autocorrelation(
610:        mi_true = _compute_morans_i_true(addr_df, fips)
699:            mi_out = _compute_morans_i_output(preds, cur_gdf)
```
m6 computes Moran's I once per (draw, tract) pair and writes it as one `recovery_grid.csv` row
(`:699-712`) -- no aggregation happens inside `run_grid.py` itself; the median-of-81-draw-means
reduction cited in A7 is performed by the downstream analysis consumer, not by the generator.

---

## UNTRACED (as of TRACE-P1; superseded entries noted)

- **A4 (parcel-join count): still UNTRACED, now DROPPED.** 39,476/39,535 (99.9%) has no
  committed artifact and none was created for it. Only source is `SESSION_LOG.md` (markdown)
  and `docs/proposal/section5_verification_manifest.md` (markdown, itself citing the same
  `SESSION_LOG.md` entry). `data/raw/parcels/` is gitignored. The 39,535-address and 20-tract
  portions of A4 are separately CONFIRMED (see table). **Per TRACE-P1-CLOSE, this count is
  dropped from Paper 1 rather than carried forward as an open item.**
- ~~A12 (rank-based sensitivity analysis).~~ **Closed by TRACE-P1-CLOSE B2** (commit `409dba0`):
  `scripts/power_analysis_parity.py` now implements a Wilcoxon signed-rank test and an exact
  sign test; see Trace 2 and `data/results/m0_n20_svi_parity/power_analysis.json`.

## UNTRACKED-ONLY (as of TRACE-P1; superseded entries noted)

- **A5 row count: partially re-verdicted, still UNTRACKED-ONLY for the source files.**
  `data/processed/national_bg_svi.csv` and `data/processed/national_bg_acs_raw.csv` remain
  gitignored (`.gitignore:53`) and were never committed -- B3 (commit `1ae6905`) does not
  change this; it adds a committed manifest with the files' sha256 hashes and derived counts,
  not the files themselves. **The 242,335/239,346 counts are also now known to be the wrong
  numbers to cite regardless of commit status** -- see the A5 row's CORRECTED verdict (233,338
  actually-ranked). The 12-indicator/4-theme methodology is separately CONFIRMED from committed
  code.
- ~~A11 (cross-tract edge audit, mean degree).~~ **Closed by TRACE-P1-CLOSE B1** (commit
  `3bcdf7d`): `experiments/audits/graph_edge_audit/` independently reconstructs the same numbers
  from the committed builder. `scripts/diag/xt_edge_audit.py` and `claude_XT_Edge_Audit.md`
  themselves remain gitignored (`.gitignore:157-158`, likely swept up by commit `6280fa1`,
  "untrack agentic/AI-context helper files from source control") and should not be cited going
  forward -- cite `experiments/audits/graph_edge_audit/` instead.

## Phantom sweep (grep -e "0\.844" -e "11\.742" -e "0\.671" -e "0\.033" -e "Nguyen")

No un-flagged phantom citations found. All hits are either explicit retirement/audit text or
unrelated legitimate values:

- `0.844`: `DIRECTION.md:40` and `experiments/audits/outstanding_items_reconciliation.md` --
  both explicitly identify 0.844 as a different experiment's single-BG `constrained_r`, not a
  Dasymetric pooled BG r, and instruct not to cite it. One unrelated hit:
  `data/synthetic/calibration/svi_variance_decomposition.json:10`, a timestamp substring
  (`...2026-06-23T01:20:50.844719Z`).
- `11.742`: legitimate, unrelated value -- `experiments/ablation/05b_topology_specificity/degree_calibration.md`
  and `RESULTS.md:18` use it as the achieved mean degree for the **degree-parity k=10**
  construction in the `spatial_knn_uniform`/`randomized` conditions of that ablation. This is a
  different graph and a different claim from A11's 12.162 (production road-network hybrid,
  39,535 nodes); the two must not be conflated when citing either.
- `0.671`: `DIRECTION.md:42` and `outstanding_items_reconciliation.md` -- both explicitly flag
  r=0.671 as retired/ungrounded (deleted-experiment coordinate correlation) and instruct not to
  cite it.
- `0.033`: two senses, both non-phantom. `DIRECTION.md:43` / `outstanding_items_reconciliation.md`
  flag the retired address-level accessibility r=0.033 as not to cite. Separately,
  `experiments/ecological_fallacy/variance_decomposition_summary.md:89` uses 0.033 as the
  `eta_sq` for an unrelated feature (`grocery_forced_walk_burden`) -- a different statistic, not
  the retired correlation.
- `Nguyen`: zero hits anywhere in the repo (consistent with commit `b3d0fca`, "remove phantom
  Nguyen citation").

## Files read first (per milestone instructions)

`CONVERSATION_PROTOCOL.md` and `IMPORTS.md`: do not exist (confirmed via
`find . -iname "*conversation_protocol*" -o -iname "*imports*"`, no hits). `SESSION_LOG.md`
(tail), `Research_Status.md`, `docs/reviews/proposal_variance_components_trace.md`, and
`data/results/variance_components/` (`73e8638`) were read; the latter's five claims were
independently re-recomputed in this pass rather than copied (none changed; A1-A3 above restate
them under the Tier A/B numbering with fresh recompute commands).


## Reference v2 (REF-FIX)

The block-group SVI reference changed on 2026-10-31. v2 nulls Census annotation sentinels
(values at or below -100000000) before rates are computed (`f5202e5`), and it computes EP_MINRTY
as `100 * (1 - B03002_003E / B03002_001E)`, the block-group equivalent of the CDC/ATSDR SVI 2020
DP05 minority sum (`4366521`). v1 stays selectable as a frozen, cache-only reference
(`reference_version='v1'`, `8d32b78`). The entries above stay as written and record v1. This
section lists every Paper 1 number that changes under v2.

Reproduction: the v1 gate scores all 90 committed 5b and crossed cells to within 1e-12 of the
committed `pooled_bg_r` (max absdiff 3.3e-16; `62a7dcd`,
`data/results/bg_reference_v2/v1_reproduction_gate.csv`). M0 rescoring uses the persisted PATH B
parquets (`experiments/recovery/per_address_predictions/`). Their v1 recompute matches every
frozen M0 value at 4 decimals, so the v1 column below is the frozen value and the difference is
taken against the like-for-like v1 recompute.

Commands (results commit `3723706`):

```
python scripts/bg_reference_v2_manifest.py
python scripts/rescore_bg_reference.py --gate
python scripts/rescore_bg_reference.py --rescore
python scripts/power_analysis_parity.py --reference-version v2
python scripts/power_analysis_parity.py --reference-version v1 \
    --artifact data/results/bg_reference_v2/m0_per_tract_v1.csv \
    --output data/results/bg_reference_v2/power_analysis_v1_recompute.json
```

| ID | Number | v1 (frozen) | v2 | v2 minus v1 recompute | BGs v1 / v2 | Artifact |
|---|---|---|---|---|---|---|
| A1 | Pooled BG r, framework | 0.7692 [0.660, 0.853] | 0.7664 [0.656, 0.848] | -0.0028 | 69 / 67 | `bg_reference_v2/m0_aggregate_v1_v2.csv` |
| A1 | Pooled BG r, Dasymetric | 0.8018 [0.712, 0.871] | 0.7966 [0.697, 0.872] | -0.0052 | 69 / 67 | same |
| A1 | Pooled BG r, Pycnophylactic | 0.7678 [0.652, 0.858] | 0.7604 [0.644, 0.851] | -0.0074 | 69 / 67 | same |
| A1 | CIs overlap | yes | yes | none | 69 / 67 | same |
| A2 | Per-tract median r, framework | 0.3901 | 0.4610 | +0.0708 | 19 tracts | same |
| A2 | Per-tract median r, Dasymetric | 0.7867 | 0.7912 | +0.0045 | 19 tracts | same |
| A2 | Per-tract median r, Pycnophylactic | 0.2078 | 0.0139 | -0.1939 | 19 tracts | same |
| A3 | Paired diff vs Dasymetric, median [95% CI] | -0.1207 [-0.5356, 0.1080] | -0.1632 [-0.4057, 0.0787] | -0.0425 | 19 pairs | `bg_reference_v2/m0_pairwise_diffs_v1_v2.csv` |
| A3 | Paired diff, mean | -0.2699 | -0.2499 | +0.0200 | 19 pairs | `bg_reference_v2/power_analysis_v2.json` |
| A5 | National BGs with non-null ranked SVI | 233,338 of 242,335 | 221,944 of 242,335 | -11,394 | n/a | `bg_reference_v2/manifest.json` |
| A5 | Hamilton BGs with non-null SVI | 263 of 278 | 251 of 278 | -12 | n/a | same |
| A6 | 5b pooled BG r, condition-mean range | [0.7628, 0.7716] | [0.7575, 0.7650] | about -0.005 | 69 / 67 | `bg_reference_v2/rescored_condition_means.csv` |
| A6 | 5b randomized SAGE / GCN-GAT pooled r | 0.7716 / 0.7652 | 0.7650 / 0.7604 | -0.0065 / -0.0048 | 69 / 67 | same |
| A6 | Crossed randomized SAGE / GCN-GAT pooled r (25 cells) | 0.7721 / 0.7645 | 0.7675 / 0.7601 | -0.0046 / -0.0045 | 69 / 67 | same |
| A8 | Pooled r spread across methods | 0.0340 | 0.0362 | +0.0022 | 69 / 67 | `bg_reference_v2/m0_aggregate_v1_v2.csv` |
| A12 | SD of differences | 0.9231 | 0.9089 | -0.0142 | 19 tracts | `bg_reference_v2/power_analysis_v2.json` |
| A12 | Cohen's d | 0.2924 | 0.2750 | -0.0174 | 19 tracts | same |
| A12 | Power at n=19 / n=85 | 22.7% / 76.0% | 20.6% / 70.8% | -2.1 / -5.2 points | 19 tracts | same |
| A12 | MDE at n=19, 80% power | 0.627 | 0.618 | -0.009 | 19 tracts | same |
| A12 | Projected CI at n=85, mean | [-0.47, -0.07] | [-0.44, -0.06] | shifts toward zero | 19 tracts | same |
| A12 | Tracts for 80% power | 94 | 106 | +12 | 19 tracts | same |

Unchanged under v2: every Moran's I value (A6, A7, A10, A14; no reference dependence), A4, A9,
A11, A13, B1 to B3, and the CI-overlap verdict for A1. The 5b and crossed per-cell values are in
`bg_reference_v2/rescored_cells.csv` (90 rows, v1 and v2 side by side).

A12 rank-based tests. `scripts/power_analysis_parity.py` counts a paired difference as zero only
at exactly 0.0. Tract 47065001800 has two block groups, so both methods score r = +/-1 and the
difference is a tie in principle. The frozen run recorded 0.0. The parquet recompute records
+2.2e-16 under v1 and -2.2e-16 under v2, which the script counts as nonzero. Raw script output:
v1 recompute Wilcoxon 53.0, p=0.1570, sign 7/11, p=0.4807; v2 Wilcoxon 52.0, p=0.1446, sign 6/12,
p=0.2379. With ties at or below 1e-12 set to zero (computed outside the script, not committed),
v1 recompute and v2 both give Wilcoxon 46.0, p=0.1488, sign 6/11, p=0.3323, identical to the
frozen values. The reference change does not move the rank tests; the raw v2 figures reflect the
float tie. Whether the script should snap ties is open for Dr. Gao.

Hamilton reference comparison (`bg_reference_v2/manifest.json`): over the 67 scoring BGs both
versions score, v1 and v2 SVI agree at Pearson 0.99889 and Spearman 0.99904. The mean absolute
change is 0.018 and the maximum is 0.041 (470650114024, 0.463 to 0.504). 60 of the 67 rise,
consistent with the 15,700 sentinel BGs leaving the top of the national MHI ranking (not
isolated from the minority change). The two
named BGs 470650104314 and 470650104333 carry the MHI sentinel and have NaN SVI under v2.

## FIG-P1 figure numbers

Every plotted or captioned number in the Paper 1 figures, as written by `python scripts/build_paper1_figures.py` to `data/results/paper1_figures/figure_numbers.json` (full precision there). Intervals are 95% tract-clustered percentile bootstraps: the 20 tracts resampled with replacement, 2000 resamples, numpy default_rng seed 20261101. Display rounding is 3 decimals, ROUND_HALF_UP on repr(float). Source commit is the last commit touching the source path at build time.

| Figure | Name | Value | 95% interval | Source | Source commit |
|---|---|---|---|---|---|
| 1 | n_tracts_county | 87.0 |  | `data/results/paper1_figures/hamilton_tracts_svi.geojson` | `b0e2a57` |
| 1 | n_tracts_with_svi | 85.0 |  | `data/results/paper1_figures/hamilton_tracts_svi.geojson` | `b0e2a57` |
| 1 | n_tracts_paper1 | 20.0 |  | `data/results/paper1_figures/hamilton_tracts_svi.geojson` | `b0e2a57` |
| 1 | inset_n_nodes | 2089.0 |  | `data/results/paper1_figures/provenance.json` | `b0e2a57` |
| 1 | inset_n_undirected_edges | 13162.0 |  | `data/results/paper1_figures/provenance.json` | `b0e2a57` |
| 1 | inset_full_tract_mean_degree | 12.601244614648158 |  | `data/results/paper1_figures/provenance.json` | `b0e2a57` |
| 1 | inset_window_n_nodes | 200.0 |  | `data/results/paper1_figures/provenance.json` | `b0e2a57` |
| 1 | inset_window_induced_mean_degree | 10.37 |  | `data/results/paper1_figures/provenance.json` | `b0e2a57` |
| 2 | pooled_bg_r_GRANITE | 0.7663784437781533 | [0.6562008448504074, 0.8483177412358122] | `data/results/bg_reference_v2/m0_aggregate_v1_v2.csv` | `3723706` |
| 2 | pooled_bg_r_Dasymetric | 0.7966021224941551 | [0.6969647270417941, 0.8715551687715076] | `data/results/bg_reference_v2/m0_aggregate_v1_v2.csv` | `3723706` |
| 2 | pooled_bg_r_Pycnophylactic | 0.7603570144782591 | [0.6436486700187483, 0.850705948163584] | `data/results/bg_reference_v2/m0_aggregate_v1_v2.csv` | `3723706` |
| 2 | caption_wilcoxon_statistic | 46.0 |  | `data/results/bg_reference_v2/power_analysis_v2_tietol.json` | `9b14893` |
| 2 | caption_wilcoxon_p | 0.14879179284657834 |  | `data/results/bg_reference_v2/power_analysis_v2_tietol.json` | `9b14893` |
| 2 | caption_sign_test_n_pos | 6.0 |  | `data/results/bg_reference_v2/power_analysis_v2_tietol.json` | `9b14893` |
| 2 | caption_sign_test_n_neg | 11.0 |  | `data/results/bg_reference_v2/power_analysis_v2_tietol.json` | `9b14893` |
| 2 | caption_sign_test_p | 0.332305908203125 |  | `data/results/bg_reference_v2/power_analysis_v2_tietol.json` | `9b14893` |
| 3a | morans_i_structured_sage | 0.872302936618014 | [0.7873940462253142, 0.9306402500919304] | `data/results/m6_topology_5b/topology_specificity_metrics.json` | `04f453e` |
| 3a | morans_i_randomized_crossed_sage | 0.5036719618533596 | [0.4337121323994777, 0.5711953115778655] | `data/results/m6_topology_5b_crossed/trials_incremental.csv` | `9be51a6` |
| 3a | morans_i_structured_gcn_gat | 0.8844982987957074 | [0.7888894673251794, 0.9490831322455904] | `data/results/m6_topology_5b/topology_specificity_metrics.json` | `04f453e` |
| 3a | morans_i_randomized_crossed_gcn_gat | 0.07112015012780679 | [0.0550308239773037, 0.08930574958485718] | `data/results/m6_topology_5b_crossed/trials_incremental.csv` | `9be51a6` |
| 3b | pooled_bg_r_production_sage | 0.7578635160961603 |  | `data/results/bg_reference_v2/rescored_condition_means.csv` | `3723706` |
| 3b | pooled_bg_r_road_network_uniform_sage | 0.7574775988798064 |  | `data/results/bg_reference_v2/rescored_condition_means.csv` | `3723706` |
| 3b | pooled_bg_r_spatial_knn_uniform_sage | 0.758372306683623 |  | `data/results/bg_reference_v2/rescored_condition_means.csv` | `3723706` |
| 3b | pooled_bg_r_randomized_sage | 0.7675283573898029 |  | `data/results/bg_reference_v2/rescored_condition_means.csv` | `3723706` |
| 3b | pooled_bg_r_production_gcn_gat | 0.7595399761652136 |  | `data/results/bg_reference_v2/rescored_condition_means.csv` | `3723706` |
| 3b | pooled_bg_r_road_network_uniform_gcn_gat | 0.7605008243664212 |  | `data/results/bg_reference_v2/rescored_condition_means.csv` | `3723706` |
| 3b | pooled_bg_r_spatial_knn_uniform_gcn_gat | 0.7623512588708399 |  | `data/results/bg_reference_v2/rescored_condition_means.csv` | `3723706` |
| 3b | pooled_bg_r_randomized_gcn_gat | 0.7600588552630267 |  | `data/results/bg_reference_v2/rescored_condition_means.csv` | `3723706` |
| 3b | pooled_bg_r_range_min | 0.7574775988798064 |  | `data/results/bg_reference_v2/rescored_condition_means.csv` | `3723706` |
| 3b | pooled_bg_r_range_max | 0.7675283573898029 |  | `data/results/bg_reference_v2/rescored_condition_means.csv` | `3723706` |
| 4a | ceiling_gbm_seed42_tract_mean | 0.22580497938891897 |  | `data/results/m6_recovery_grid/recovery_grid.csv` | `d01f5e4` |
| 4a | recovery_r_spatial_knn_uniform_sage | 0.0017048611628437175 | [-0.014034549133982702, 0.014583164031781982] | `data/results/m6_fig4_randomization/tract_rows.csv` | `c9039a9` |
| 4a | morans_i_output_spatial_knn_uniform_sage | 0.9520572816056626 | [0.8675958471767207, 0.9950162587379163] | `data/results/m6_fig4_randomization/tract_rows.csv` | `c9039a9` |
| 4a | recovery_r_randomized_sage | 0.012340346991031289 | [0.0028344450520155383, 0.02320597013421232] | `data/results/m6_fig4_randomization/tract_rows.csv` | `c9039a9` |
| 4a | morans_i_output_randomized_sage | 0.7732004267874585 | [0.7014405817071755, 0.8193538094070724] | `data/results/m6_fig4_randomization/tract_rows.csv` | `c9039a9` |
| 4a | recovery_r_spatial_knn_uniform_gcn_gat | -0.007742632991789661 | [-0.01850681735407705, 0.0023571761309547647] | `data/results/m6_fig4_randomization/tract_rows.csv` | `c9039a9` |
| 4a | morans_i_output_spatial_knn_uniform_gcn_gat | 0.9541605378327782 | [0.8791418809942979, 0.9946291878504451] | `data/results/m6_fig4_randomization/tract_rows.csv` | `c9039a9` |
| 4a | recovery_r_randomized_gcn_gat | -0.0036855233496790956 | [-0.011497168500558937, 0.003274627801836087] | `data/results/m6_fig4_randomization/tract_rows.csv` | `c9039a9` |
| 4a | morans_i_output_randomized_gcn_gat | 0.14711036415201947 | [0.13057616169488612, 0.16541700539290558] | `data/results/m6_fig4_randomization/tract_rows.csv` | `c9039a9` |
| 4a | field_seed42_morans_i_true_current_estimator | 0.15334196178331091 |  | `data/synthetic/run_20260628_030416/addresses.csv` | `0cb231e` |
| 4a | field_seed42_morans_i_true_committed_old_estimator | 0.15958839305898267 |  | `data/results/m6_recovery_grid/recovery_grid.csv` | `d01f5e4` |
| 4b | morans_i_output_median_of_draw_means | 0.9407312808464919 |  | `data/results/m6_recovery_grid/recovery_grid.csv` | `d01f5e4` |
| 4b | recovery_r_median_of_draw_means | 8.464522132189956e-05 |  | `data/results/m6_recovery_grid/recovery_grid.csv` | `d01f5e4` |

Variance components (two-way crossed random effects without replication on the 25 cell means, balanced moment estimators, negatives truncated at 0; structured arms report the variance of the 5 training-seed cell means) are in `figure_numbers.json` under each Figure 3a and 4a entry.

Row-level versus per-draw medians: the 0.00088 recovery_r in Trace 6 (lines 35, 270 and 277 of this file) is the row-level median over the 1,620 sage coordinates_only rows. Figure 4b uses the median over the 81 draws of each draw's 20-tract mean, 8.464522132189956e-05 (8.5e-05).

Figure 4a input Moran's I: the seed-42 field scores 0.15334196178331091 under the current estimator (826c61a and later; recomputed from `data/synthetic/run_20260628_030416/addresses.csv`) and 0.15958839305898267 under the pre-826c61a estimator, as committed in the recovery_grid.csv morans_i_true column, which is not rescored.

Figure 3b: with the crossed randomized arm in place of seed 42, the condition-mean range is 0.7574775988798064 to 0.7675283573898029 (0.757 to 0.768). The earlier 0.757 to 0.765 used the seed-42 randomized arm.

## BOOT-P1 tract bootstrap

Every number below is written by `python scripts/paper1_bootstrap.py` to `data/results/paper1_bootstrap/bootstrap_results.json` (JSON key in the table, full precision there); per-resample values are in `bootstrap_draws.npz`, sha256 per array under `draws_sha256`. Margin and decision rule: `docs/paper1/equivalence_margin_spec.md` (`d6281ee`). Reference v2, 20 tracts, 67 scoring BGs, tracts resampled with replacement, B = 10,000, numpy default_rng seed 20261109, percentile CIs. Inputs: M0 parquets `10d0ee8`, structured 5b `c913306`, crossed 5b `9be51a6`, committed v2 values `3723706`. The script asserts every full-sample r equals the existing scorer to 1e-12 and the committed v2 value to 1e-9. Display rounding is 4 decimals.

| ID | Number | Value | 90% CI | 95% CI | JSON key |
|---|---|---|---|---|---|
| C1 | r_GRANITE (M0, one training seed) | 0.7664 | [0.6227, 0.8596] | [0.5908, 0.8762] | `C1.r_GRANITE` |
| C1 | r_Dasymetric | 0.7966 | [0.6675, 0.8779] | [0.6389, 0.8912] | `C1.r_Dasymetric` |
| C1 | d_r | -0.0302 | [-0.0712, 0.0024] | [-0.0806, 0.0081] | `C1.d_r` |
| C1 | d_z | -0.0778 | [-0.1640, 0.0065] | [-0.1807, 0.0231] | `C1.d_z` |
| C1 | verdict (q = 0.10) | small but inconclusive | | | `C1.verdict` |
| C1 | sensitivity verdict (r, 0.03) | small but inconclusive | | | `C1.sensitivity_verdict` |
| C1 | r-scale bounds | -0.0395 / +0.0337 | | | `margin.r_scale_lower`, `margin.r_scale_upper` |
| C2 | sage d_r, spatial_knn_uniform minus crossed randomized | -0.0092 | [-0.0196, 0.0004] | [-0.0218, 0.0021] | `C2.sage.d_r` |
| C2 | sage d_z | -0.0223 | [-0.0418, 0.0010] | [-0.0457, 0.0053] | `C2.sage.d_z` |
| C2 | gcn_gat d_r | 0.0023 | [-0.0034, 0.0114] | [-0.0045, 0.0139] | `C2.gcn_gat.d_r` |
| C2 | gcn_gat d_z | 0.0071 | [-0.0045, 0.0258] | [-0.0068, 0.0306] | `C2.gcn_gat.d_z` |

C2 variance components, reported separately (no pooling). Tract: bootstrap variance of the difference at training seed 42, graph draw 42. Seed: variance (ddof=1) of the 5 per-seed differences, randomized averaged over draws. Draw: variance (ddof=1) of the 5 per-draw randomized means; the structured arm has one draw.

| Arch | Scale | Tract | Training seed | Graph draw | JSON key |
|---|---|---|---|---|---|
| sage | r | 2.649e-04 | 8.216e-05 | 3.673e-06 | `C2.sage.variance_components.{tract.d_r.bootstrap_var, r_scale.training_seed.difference_var, r_scale.graph_draw.difference_var}` |
| sage | z | 1.308e-03 | 4.735e-04 | 2.309e-05 | `C2.sage.variance_components.{tract.d_z.bootstrap_var, z_scale.*.difference_var}` |
| gcn_gat | r | 1.655e-04 | 4.006e-04 | 1.646e-08 | `C2.gcn_gat.variance_components.{tract.d_r.bootstrap_var, r_scale.*.difference_var}` |
| gcn_gat | z | 8.049e-04 | 2.454e-03 | 9.214e-08 | `C2.gcn_gat.variance_components.{tract.d_z.bootstrap_var, z_scale.*.difference_var}` |

C1 has no seed component: M0 GRANITE carries one training seed (42). C2 carries no equivalence verdict (`C2.*.verdict` is null). Regeneration: two runs of the script produce byte-identical JSON (sha256 `8e2586746953b109944037e23b84628b6e27d814bf101c7536ab27ad1e01b2d5` at HEAD `d6281ee`); `git_head` in the JSON records the HEAD at generation.

## POSCTRL-P1 positive control

Every number below is written by `python experiments/m6_recovery_grid/run_positive_control.py` (pinned environment, `requirements.txt` at `fa3ba45`) to `data/results/paper1_figures/positive_control_morans_i.json` (JSON key in the table, full precision there). Field: seed-42 M6 cell latent/strong/medium/default, `data/synthetic/run_20260628_030416` (committed `0cb231e`, generator `0351298`). Ceiling: `run_grid._ceiling_gbm_r` unchanged (z-scored UTM x, y; KFold 5, shuffle, seed 42; GBM 200 trees, depth 4, rate 0.05, seed 42), out-of-fold predictions. Moran's I: the Figure 4a call `run_grid._compute_morans_i_output` (libpysal KNN k=8 on EPSG:4326 address coordinates, transform 'r', esda permutations 0). GRANITE comparator: spatial_knn_uniform sage from `data/results/m6_fig4_randomization/tract_rows.csv` (`c9039a9`), per-tract mean over 5 training seeds. Aggregation: unweighted mean over the 20 Paper 1 tracts. Bootstrap: tract-clustered, 2,000 resamples, numpy default_rng seed 20261102, same resample indices for every statistic, percentile CIs. Anchors as pre-registered: I_truth = 0.153, I_GRANITE = 0.952. Display rounding is 4 decimals.

| Number | Value | 95% CI | JSON key |
|---|---|---|---|
| ceiling recovery r | 0.2258 | [0.2049, 0.2465] | `aggregate.ceiling_recovery_r` |
| ceiling output Moran's I | 0.5006 | [0.4317, 0.5534] | `aggregate.ceiling_morans_i_output` |
| y_true Moran's I (Figure 4a call) | 0.1528 | [0.1231, 0.1717] | `aggregate.y_true_morans_i` |
| GRANITE output Moran's I | 0.9521 | [0.8678, 0.9950] | `aggregate.granite_morans_i_output` |
| paired difference, ceiling minus GRANITE | -0.4514 | [-0.4976, -0.4019] | `aggregate.diff_ceiling_minus_granite` |
| p = (I_ceil - 0.153) / (0.952 - 0.153) | 0.4351 | [0.3488, 0.5011] | `p.point`, `p.ci95_anchor_fixed` |
| p, truth and GRANITE also resampled | | [0.3918, 0.4806] | `p.ci95_joint_resampled_truth_and_granite` |
| band | C | | `p.band_reported` |
| signal component | not persisted | | `signal_component` |

Reproduction gate (decision 2026-11-09, `reproduction_gate.decision`): the recompute under `fa3ba45` is the reference of record; two in-process recomputes are bit-identical on all 20 tracts; the largest deviation from the committed June rows (`2dc231f`) is 3.770e-04 at 47065003400, within the 3.8e-04 bound; per-tract table under `reproduction_gate.per_tract`. The y_true Moran's I here uses EPSG:4326 coordinates (0.152815); the 0.153342 in the FIG-P1 table uses generator UTM x, y. Per-tract values under `per_tract`. Regeneration: two runs of the script produce byte-identical JSON (sha256 `8f3fd56c9bcd795ec5641f6469d93a57b637bca5b2c965c49c1bb3edafbb83bc` at HEAD `1200586`); `provenance.head` records the HEAD at generation.

## TAG-B

- M0 pooled code uses a 10-address block-group floor (`granite/scripts/run_m0_parity.py:40`) against the registered 3; on this data both select the same 67 scored block groups (every touched block group has at least 10 addresses; reproduction gate at `ccbc3ee`). BOOT-P1 used B = 10,000 with seed 20261109 (`data/results/paper1_bootstrap/bootstrap_results.json`).
- The registered clip share is 0.65% on the analysis set (20,586 of 3,162,800 values; frozen structured runs plus the crossed set, training-seed-42 randomized files counted once); 0.73% was the frozen 5b set alone (0.726%) or the ts42 double count (0.728%). Six tracts with SVI of 0.51 or less clip at 1 (`data/results/paper1_prereg_artifacts/clip_counts.json`, `b08132b`).
- The no-graph coherence values 0.682 and 0.675 (`59bd8e9`) predate the estimator fix in `826c61a` (inverse-distance, symmetrized weights) and cannot be rescored: those runs saved no predictions.
- The 83927c3 swap range 0.003 to 0.036 was measured under a drifted scikit-learn 1.9.1; the registered range under the pinned 1.8.0 is 0.0029 to 0.0393 for the 19 randomized tracts, 47065011324 at 1.000 (`data/results/paper1_prereg_artifacts/swap_rebuild.json`, `56ac806`).
