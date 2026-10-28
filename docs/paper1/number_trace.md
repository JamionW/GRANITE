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
