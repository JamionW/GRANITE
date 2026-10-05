# Paper 1 number-to-artifact trace (TRACE-P1)

Read-only recompute pass over every number Paper 1 may report. `CONVERSATION_PROTOCOL.md` and
`IMPORTS.md` do not exist in this repo (searched by name and case-insensitive glob); not read.
`SESSION_LOG.md` (tail), `Research_Status.md`, and the 2026-09-18 proposal trace
(`docs/reviews/proposal_variance_components_trace.md`) plus `data/results/variance_components/`
(commit `34199a1`) were read first; that trace's verdicts are reused below only where
re-verified against HEAD in this pass (all five of its claims were re-recomputed fresh, not
copied).

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
| A4 | 20 tracts, 39,535 addresses, 39,476 parcel-joined (99.9%) | `experiments/recovery/per_address_predictions/granite_m0.parquet` (addresses/tracts); parcel-join count has none | `10d0ee8` | `scripts/persist_per_address_predictions.py` | Trace 3 | 39,535 addresses, 20 tracts CONFIRMED; 39,476/99.9% parcel-join has no committed source | CONFIRMED (addresses, tracts) / UNTRACED (parcel-join count) |
| A5 | CDC method, 12 ACS indicators / 4 themes, ranked over 242,335 BGs | methodology: `granite/data/block_group_loader.py`; count: `data/processed/national_bg_svi.csv` | methodology `HEAD`-tracked; count file is gitignored, never committed | `granite/data/block_group_loader.py` | Trace 4 | 4+3+1+4=12 variables across 4 `theme_vars` keys (code); row count 242,335 read from disk but file is untracked | CONFIRMED (methodology) / UNTRACKED-ONLY (row count, `data/processed/national_bg_svi.csv`) |
| A6 | 5b: structured Moran's I 0.817-0.885; randomized SAGE 0.422; GCN-GAT 0.075; pooled BG r 0.763-0.772 all conditions | `data/results/m6_topology_5b/topology_specificity_metrics.json` | `04f453e` | `experiments/ablation/05b_topology_specificity/` sweep | Trace 5 | structured mean range [0.8167, 0.8845]; randomized sage 0.4217, gcn_gat 0.0745; pooled_bg_r range [0.7628, 0.7716] | CONFIRMED |
| A7 | Deck slide 13: median output Moran's I 0.94, coordinates-only, recovery near zero | `data/results/m6_recovery_grid/recovery_grid.csv` | `d01f5e4` | `experiments/m6_recovery_grid/run_grid.py` | Trace 6 | Row-level median over 1,620 sage/coordinates_only rows = 0.988, NOT 0.94. Median of per-draw (81 draws) across-tract-mean Moran's I = 0.9407. recovery_r median (row-level) = 0.00088 | CORRECTED (0.94 is the median-of-81-draw-means, not a per-row or per-tract median; must specify this when citing) |
| A8 | Methods within 0.034 of each other (derived from A1) | same as A1 | `de772b7` | — | Trace 1 | max(0.8018) - min(0.7678) = 0.0340 | CONFIRMED |
| A9 | Condition-mean = how many runs (seeds x rewiring draws), seed values | `data/results/m6_topology_5b/topology_specificity_metrics.json` | `04f453e` | 05b sweep | Trace 5 | 5 runs per condition-arch cell. Structured conditions: training_seed in [42,17,123,2024,7], graph_draw_seed fixed=42. randomized: training_seed fixed=42, graph_draw_seed in [42,17,123,2024,7] | CONFIRMED |
| A10 | Moran's I weights: fixed external k=8 row-standardized binary k-NN, separate from message-passing graph | `granite/evaluation/spatial_diagnostics.py:39-52` | tracked at `HEAD` | — | Trace 7 | `compute_spatial_autocorrelation(..., k_neighbors=8)` uses `libpysal.weights.KNN(..., k=8)`, `.transform='r'`, `esda.Moran`; independent of `granite/data/loaders.py` road-network/geographic edge builder | CONFIRMED |
| A11 | Graph: per-tract road-network hybrid, k=6 Gaussian fallback, mean degree 12.162 over 39,535 nodes, 0/480,808 cross-tract edges | `claude_XT_Edge_Audit.md`, `scripts/diag/xt_edge_audit.py` | never committed (both gitignored, `git log --all` empty for both paths) | `scripts/diag/xt_edge_audit.py` (not committed) | Trace 8 | 480,808 total edges / 39,535 nodes = 12.1616 (matches 12.162); 0 cross-tract edges reported in the doc, but neither file is a committed artifact | UNTRACKED-ONLY (`claude_XT_Edge_Audit.md`, `scripts/diag/xt_edge_audit.py`) |
| A12 | Power analysis: SD 0.92, d 0.29, power 22%@19 / 76%@85, MDE 0.63, CI@85 [-0.47,-0.07], ~94 tracts for 80%; rank-based sensitivity exists | `data/results/m0_n20_svi_parity/per_tract.csv` | `de772b7` | `scripts/power_analysis_parity.py` (`bbaa5ff`) | Trace 2 | sd=0.9231, d=0.2924, power 22.7%/76.0%, MDE@19=0.627, CI@85=[-0.47,-0.07], n for 80%=94 | CONFIRMED (power figures) / UNTRACED (no rank-based sensitivity analysis script exists anywhere in the repo; `grep -rl "rank-based"` is empty) |
| A13 | Dasymetric: `baselines.py:84`, `tract_svi + alpha*(impervious_pct - tract_mean)`, alpha largest value keeping [0,1] | `granite/evaluation/baselines.py` | tracked at `HEAD` | — | Trace 9 | Class `DasymetricDisaggregation` at line 84; `dev = a - a.mean()` then `predictions = tract_svi + alpha * dev` at line 122, `alpha = min(max_up, max_down)` chosen to keep output in [0,1] (lines 93-123) | CONFIRMED (logic and class line; additive-form line is 122, not 84) |
| B1 | M6 grid: 81 draws, 27 cells, seeds 42/17/123, 20 tracts, 7,200 rows | `data/results/m6_recovery_grid/recovery_grid.csv` | `d01f5e4` | `experiments/m6_recovery_grid/run_grid.py` | Trace 6 | rows 7200, draws 81, cells 27, seeds [17,42,123], tracts 20 | CONFIRMED |
| B2 | Coordinates: 97.86% / 97.88% between-tract, eta_sq 0.979 | `experiments/ecological_fallacy/variance_decomposition.csv` | `8aecdc1` | `scripts/variance_decomposition.py` | Trace 10 | lat eta_sq=0.97858187, lon eta_sq=0.97882321 | CONFIRMED |
| B3 | Between-tract variance share vs SVI coupling: Pearson -0.052, n=75 | `experiments/ecological_fallacy/variance_decomposition.csv` | `8aecdc1` | `scripts/variance_decomposition.py` | Trace 10 | pearsonr(eta_sq, abs(tract_svi_r)) = -0.0518, n=75 | CONFIRMED |

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
committed artifact to recompute from: **UNTRACED**.

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
committed -- **UNTRACKED-ONLY**.

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

### Trace 8: A11 cross-tract edge audit (untracked)

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
rule, gitignored/untracked outputs are non-evidence: **UNTRACKED-ONLY**.

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

---

## UNTRACED

- **A4 (parcel-join count).** 39,476/39,535 (99.9%) has no committed artifact. Only source is
  `SESSION_LOG.md` (markdown) and `docs/proposal/section5_verification_manifest.md` (markdown,
  itself citing the same `SESSION_LOG.md` entry). `data/raw/parcels/` is gitignored. The
  39,535-address and 20-tract portions of A4 are separately CONFIRMED (see table).
- **A12 (rank-based sensitivity analysis).** No script anywhere in the repo implements a
  rank-based sensitivity analysis (`grep -rl "rank-based"` across `.py`/`.md` is empty). The
  power-analysis numbers themselves are CONFIRMED from `scripts/power_analysis_parity.py`.

## UNTRACKED-ONLY

- **A5 row count (242,335 / 239,346 block groups).** `data/processed/national_bg_svi.csv` and
  `data/processed/national_bg_acs_raw.csv` are both gitignored (`.gitignore:53`) and were never
  committed. The file exists on disk and its row count matches the claim, but it is not a
  committed artifact. The 12-indicator/4-theme methodology is separately CONFIRMED from
  committed code.
- **A11 (cross-tract edge audit, mean degree).** `scripts/diag/xt_edge_audit.py` and
  `claude_XT_Edge_Audit.md` are both gitignored (`.gitignore:157-158`) and have no commit
  history (`git log --all` empty for both paths). This likely predates or was swept up by
  commit `6280fa1` ("untrack agentic/AI-context helper files from source control"). The
  480,808-edge / 12.162-mean-degree / 0-cross-tract numbers exist only in this untracked pair.

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
`data/results/variance_components/` (`34199a1`) were read; the latter's five claims were
independently re-recomputed in this pass rather than copied (none changed; A1-A3 above restate
them under the Tier A/B numbering with fresh recompute commands).
