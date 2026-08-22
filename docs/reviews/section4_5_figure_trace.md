# Section 4 and 5 Figure Trace

**Generated:** 2026-08-22
**Type:** Read-only verification. No source, data, or proposal files were edited. This report is the only new file.
**Rule applied:** verify the artifact, not the claim. Each figure is traced to a committed artifact, the computing command is run, and the reproduced value is recorded against the claimed value.
**Scope:** the sixteen figures inserted into Section 4 (Preliminary Work) and Section 5 (Proposed Research Program).

All commands are run from the repo root `/workspaces/GRANITE` on branch `main`. Variable shorthands used in the per-figure detail:

```
G=data/results/m6_recovery_grid/recovery_grid.csv
T=data/results/m6_topology_5b/topology_specificity_metrics.json
V=experiments/ecological_fallacy/variance_decomposition.csv
D=experiments/recovery/delinquency_convergent_validity/results.json
```

---

## Verdict summary

All sixteen figures reproduce against a committed artifact. Sixteen of sixteen are marked pass under the stated flag criterion (flag = missing artifact or non-reproducing number). Four figures carry characterization caveats where the surrounding proposal prose needs a qualifier even though the number itself reproduces: figures 5, 8, 15, and 16. These are enumerated in the Caveats section. None of the three phantom figures (0.844, 11.742, 0.671) appears as a stand-in for any of the sixteen.

---

## Trace table

| # | Figure | Claimed | Reproduced | Artifact | Command | Mark |
|---|--------|---------|-----------|----------|---------|------|
| 1 | grid holds 81 draws | 81 | 81 | `data/results/m6_recovery_grid/recovery_grid.csv` | `df[['autocorr','snr','between_tract','seed']].drop_duplicates().shape[0]` | pass |
| 2 | grid holds 7,200 per-tract rows | 7,200 | 7200 | `data/results/m6_recovery_grid/recovery_grid.csv` | `len(df)` | pass |
| 3 | three levels each of autocorr, SNR, between-tract | 3x3x3 = 27 cells | autocorr{weak,medium,strong}, snr{low,medium,high}, between_tract{low,default,high}, 27 cells | `data/results/m6_recovery_grid/recovery_grid.csv` | unique-level print | pass |
| 4 | GraphSAGE mean within-tract recovery near zero at every autocorr/SNR; max cell | near zero | all 9 cells in [-0.0007, 0.0052]; max cell (medium,medium)=0.0052 | `data/results/m6_recovery_grid/recovery_grid.csv` | groupby sage coords_only | pass |
| 5 | output Moran's I averaged 0.94 over grid | 0.94 | 0.9414 (sage coordinates_only sweep) | `data/results/m6_recovery_grid/recovery_grid.csv` | mean `morans_i_output` | pass (scope caveat) |
| 6 | GBM ceiling from coordinates r = 0.13 | 0.13 | 0.1309 (ceiling_gbm grand mean) | `data/results/m6_recovery_grid/recovery_grid.csv` | mean `recovery_r`, method=ceiling_gbm | pass |
| 7 | 97.9% of coordinate variance between tracts | 97.9% | 0.9787 median eta_sq -> 97.9% | `experiments/ecological_fallacy/variance_decomposition.csv` | median eta_sq, coordinate class | pass |
| 8 | structured-graph Moran's I in 0.817 to 0.885 | 0.817-0.885 | means 0.8167 to 0.8845 | `data/results/m6_topology_5b/topology_specificity_metrics.json` | per-condition mean morans_i | pass (rounding caveat) |
| 9 | randomized GraphSAGE Moran's I = 0.422 | 0.422 | 0.4217 | `data/results/m6_topology_5b/topology_specificity_metrics.json` | randomized.sage.mean.morans_i | pass |
| 10 | randomized GCN-GAT Moran's I = 0.075 | 0.075 | 0.0745 | `data/results/m6_topology_5b/topology_specificity_metrics.json` | randomized.gcn_gat.mean.morans_i | pass |
| 11 | pooled BG r within 0.763 to 0.772 across conditions | 0.763-0.772 | means 0.7628 to 0.7716 | `data/results/m6_topology_5b/topology_specificity_metrics.json` | per-condition mean pooled_bg_r | pass |
| 12 | median partial Spearman with delinquency = 0.003 | 0.003 | 0.002819 | `experiments/recovery/delinquency_convergent_validity/results.json` | primary_partial_rho.granite.median | pass |
| 13 | one-sided p = 0.372 | 0.372 | 0.37178 (one-sided Wilcoxon) | `experiments/recovery/delinquency_convergent_validity/results.json` | wilcoxon_vs_zero granite p | pass |
| 14 | 16 non-sparse tracts entered pilot | 16 | 16 (4 sparse excluded) | `experiments/recovery/delinquency_convergent_validity/results.json` | n_primary_tracts | pass |
| 15 | within-tract variance concentrates in building/parcel; report share by family | building/parcel highest | building 0.906 > modal 0.473 > accessibility 0.230 > coordinate 0.021 > socioeconomic ~0 | `experiments/ecological_fallacy/variance_decomposition.csv` | median within_share by feature_class | pass (taxonomy caveat) |
| 16 | coordinates support no supervised ceiling above 0.13; identity with fig 6 | 0.13, = fig 6 | 0.1309 grand mean, same artifact as fig 6 | `data/results/m6_recovery_grid/recovery_grid.csv` | mean recovery_r, method=ceiling_gbm | pass (supremum caveat) |

---

## Phantom figure check

The three known phantom figures were searched across committed `*.md *.json *.csv *.txt`. None appears as a stand-in for any of the sixteen figures, and none of the sixteen reproduced values equals a phantom.

```
$ git grep -n -E "0\.844|11\.742|0\.671" -- '*.md' '*.json' '*.csv' '*.txt'
```

- **0.844** occurs only as a documented retired value: a single-BG `constrained_r` from a county-ranked convergence experiment, explicitly not a Dasymetric or pooled BG r (`experiments/audits/outstanding_items_reconciliation.md`, `PROJECT_PREAMBLE.md`). It is not any of the sixteen figures. The `...50.844719Z` timestamp hit in `svi_variance_decomposition.json` is coincidental digits.
- **11.742** occurs as a genuine measured quantity, the spatial mean degree at k=10, in `experiments/ablation/05b_topology_specificity/RESULTS.md:18` and `degree_calibration.md`. This is its legitimate home; it is not standing in for any of the sixteen. It is a degree, not a Moran's I or a correlation, so it is not confusable with figures 8 to 11.
- **0.671** occurs only in removal and audit records (`SESSION_LOG.md`, `outstanding_items_reconciliation.md`, `section5_verification_manifest.md`) documenting its excision on 2026-06-09 as an ungrounded coordinate address-level r. It does not appear as a live figure. The `0.6719769...` hit in `02b_iii_layernorm_in_hidden/results/per_tract_metrics.csv` is an unrelated per-tract metric column, coincidental digits.

---

## Per-figure detail (exact command and raw output)

### Group 1: Synthetic recovery grid, Study 1 (figures 1 to 7)

**Figure 1: the grid holds 81 draws.**

```
$ python3 -c "import pandas as pd;d=pd.read_csv('data/results/m6_recovery_grid/recovery_grid.csv');print(d[['autocorr','snr','between_tract','seed']].drop_duplicates().shape[0])"
81
```

A draw is a unique (autocorr, snr, between_tract, seed) tuple, matching the scratch filenames `draw_latent__{autocorr}__{snr}__{between_tract}__{seed}.json` in `experiments/m6_recovery_grid/scratch/`. Reproduced 81. **PASS.**

**Figure 2: the grid holds 7,200 per-tract result rows.**

```
$ python3 -c "import pandas as pd;print(len(pd.read_csv('data/results/m6_recovery_grid/recovery_grid.csv')))"
7200
```

Row composition (one row per draw-tract-method): granite 2340, dasymetric 1620, pycnophylactic 1620, ceiling_gbm 1620; total 7200. Reproduced 7200. **PASS.**

**Figure 3: three levels each of spatial autocorrelation, SNR, and between-tract separation.**

```
$ python3 -c "import pandas as pd;d=pd.read_csv('data/results/m6_recovery_grid/recovery_grid.csv');print('autocorr',sorted(d.autocorr.unique()));print('snr',sorted(d.snr.unique()));print('between_tract',sorted(d.between_tract.unique()));print('cells',d[['autocorr','snr','between_tract']].drop_duplicates().shape[0])"
autocorr ['medium', 'strong', 'weak']
snr ['high', 'low', 'medium']
between_tract ['default', 'high', 'low']
cells 27
```

Three levels of each of the three factors, 27 crossed cells. **PASS.**

**Figure 4: GraphSAGE mean within-tract recovery stayed near zero at every autocorr and SNR; maximum cell.**

```
$ python3 -c "import pandas as pd;d=pd.read_csv('data/results/m6_recovery_grid/recovery_grid.csv');s=d[(d.method=='granite')&(d.arch=='sage')&(d.feature_mode=='coordinates_only')];c=s.groupby(['autocorr','snr']).recovery_r.mean();print(c.round(4).to_string());print('MAX',c.idxmax(),round(c.max(),4));print('MIN',c.idxmin(),round(c.min(),4))"
autocorr  snr
medium    high      0.0047
          low       0.0024
          medium    0.0052
strong    high      0.0034
          low       0.0005
          medium   -0.0007
weak      high      0.0035
          low       0.0006
          medium    0.0004
MAX ('medium', 'medium') 0.0052
MIN ('strong', 'medium') -0.0007
```

All nine autocorr-by-SNR cell means lie in [-0.0007, 0.0052]; the maximum cell is (medium, medium) at r = 0.0052. Near zero at every level. **PASS.** (Restricted to the swept mode `coordinates_only`, the only mode present at all nine cells. Pooling all sage feature modes gives max cell (medium,high)=0.0047, same conclusion.)

**Figure 5: output Moran's I averaged 0.94 over the grid.**

```
$ python3 -c "import pandas as pd;d=pd.read_csv('data/results/m6_recovery_grid/recovery_grid.csv');s=d[(d.method=='granite')&(d.arch=='sage')&(d.feature_mode=='coordinates_only')];print('sage/coords_only mean morans_i_output',round(s.morans_i_output.mean(),4));print('all granite mean morans_i_output',round(d[d.method=='granite'].morans_i_output.mean(),4))"
sage/coords_only mean morans_i_output 0.9414
all granite mean morans_i_output 0.8032
```

The primary sweep (GraphSAGE, coordinates_only, the 81 draws x 20 tracts) averages output Moran's I 0.9414, which rounds to 0.94. **PASS with scope caveat:** the 0.94 is the sage coordinates_only sweep, not an average over all `morans_i_output` rows. Averaging every granite row, which mixes in GCN-GAT and the degraded diagnostic feature modes (random_noise, coords_plus_noise), gives 0.8032. See Caveats.

**Figure 6: GBM fit on hidden address values from coordinates reached a supervised ceiling of r = 0.13.**

```
$ python3 -c "import pandas as pd;d=pd.read_csv('data/results/m6_recovery_grid/recovery_grid.csv');c=d[d.method=='ceiling_gbm'];print('n',len(c),'grand mean',round(c.recovery_r.mean(),4));print(c.groupby('autocorr').recovery_r.mean().round(4).to_string())"
n 1620 grand mean 0.1309
autocorr
medium    0.1273
strong    0.2253
weak      0.0402
```

Method `ceiling_gbm` is the 5-fold GradientBoostingRegressor fit on the generator-frame UTM coordinates predicting the hidden y_true, one row per draw-tract. Grand mean recovery_r = 0.1309, rounds to 0.13. **PASS.**

**Figure 7: 97.9 percent of coordinate variance lies between tracts.**

```
$ python3 -c "import pandas as pd;d=pd.read_csv('experiments/ecological_fallacy/variance_decomposition.csv');c=d[d.feature_class=='coordinate'];print(c[['feature','eta_sq','within_share']].to_string(index=False));print('median eta_sq',round(c.eta_sq.median(),4),'->',round(c.eta_sq.median()*100,1),'pct between-tract')"
feature   eta_sq  within_share
    lat 0.978582      0.021418
    lon 0.978823      0.021177
median eta_sq 0.9787 -> 97.9 pct between-tract
```

One-way ANOVA eta_sq is the between-tract variance share. Coordinate class (lat, lon) median eta_sq 0.9787, i.e. 97.9 percent between-tract, within_share 0.021. Matches `Research_Status.md:39` and `SESSION_LOG.md:106`. **PASS.**

### Group 2: Topology specificity, rewiring (figures 8 to 11)

Source artifact `data/results/m6_topology_5b/topology_specificity_metrics.json`, five seeds per condition-arch, Moran's I scored against a fixed external k=8 coordinate weight matrix (see `topology_specificity_metrics_meta.json`). One command produces figures 8 to 11:

```
$ python3 -c "
import json
m=json.load(open('data/results/m6_topology_5b/topology_specificity_metrics.json'))
struct=[]
for cond in ['spatial_knn_uniform','road_network_uniform','production']:
    for arch in ['sage','gcn_gat']:
        struct.append((cond,arch,round(m[cond][arch]['mean']['morans_i'],4)))
print('structured morans_i means:')
for x in struct: print(' ',x)
sv=[x[2] for x in struct]; print('structured range:',min(sv),'to',max(sv))
print('randomized sage morans_i mean:',round(m['randomized']['sage']['mean']['morans_i'],4))
print('randomized gcn_gat morans_i mean:',round(m['randomized']['gcn_gat']['mean']['morans_i'],4))
allbg=[]
for cond in m:
    for arch in m[cond]:
        allbg.append((cond,arch,round(m[cond][arch]['mean']['pooled_bg_r'],4)))
bv=[x[2] for x in allbg]; print('pooled_bg_r means range:',min(bv),'to',max(bv))
"
structured morans_i means:
  ('spatial_knn_uniform', 'sage', 0.8723)
  ('spatial_knn_uniform', 'gcn_gat', 0.8845)
  ('road_network_uniform', 'sage', 0.8348)
  ('road_network_uniform', 'gcn_gat', 0.8167)
  ('production', 'sage', 0.8479)
  ('production', 'gcn_gat', 0.8261)
structured range: 0.8167 to 0.8845
randomized sage morans_i mean: 0.4217
randomized gcn_gat morans_i mean: 0.0745
pooled_bg_r means range: 0.7628 to 0.7716
```

**Figure 8: structured-graph Moran's I falls in the 0.817 to 0.885 range.** Structured per-condition-arch means span 0.8167 (road_network_uniform, GCN-GAT) to 0.8845 (spatial_knn_uniform, GCN-GAT). Lower bound 0.8167 -> 0.817. **PASS with rounding caveat:** the upper-bound cell mean is 0.88450, which rounds to 0.884 at three decimals; the committed `experiments/ablation/05b_topology_specificity/RESULTS.md` (lines 38 and 52) displays and states 0.885. The claim 0.817 to 0.885 traces verbatim to that RESULTS.md text. See Caveats.

**Figure 9: randomized GraphSAGE Moran's I equals 0.422.** Reproduced 0.4217, rounds to 0.422. RESULTS.md:43 states 0.422 (supersedes retired 0.446). **PASS.**

**Figure 10: randomized GCN-GAT Moran's I equals 0.075.** Reproduced 0.0745, rounds to 0.075. RESULTS.md:44 states 0.075 (supersedes retired 0.083). **PASS.**

**Figure 11: pooled block-group correlation stays within 0.763 to 0.772 across every condition.** Per-condition-arch pooled_bg_r means span 0.7628 (road_network_uniform, SAGE) to 0.7716 (randomized, SAGE). 0.7628 -> 0.763, 0.7716 -> 0.772. RESULTS.md:146 states 0.763 to 0.772. **PASS.**

### Group 3: Delinquency pilot (figures 12 to 14)

Source artifact `experiments/recovery/delinquency_convergent_validity/results.json`, corroborated by the sibling `summary.txt`. One command produces figures 12 to 14:

```
$ python3 -c "
import json
d=json.load(open('experiments/recovery/delinquency_convergent_validity/results.json'))
print('n_primary_tracts',d['n_primary_tracts'],'| n_sparse_excluded',d['n_sparse_tracts_excluded_primary'])
print('granite median partial rho',d['primary_partial_rho']['granite']['median'],'-> round',round(d['primary_partial_rho']['granite']['median'],3))
w=[x for x in d['wilcoxon_vs_zero'] if x['label'].startswith('granite')][0]
print('granite wilcoxon vs 0 p',w['p'],'-> round',round(w['p'],3))
"
n_primary_tracts 16 | n_sparse_excluded 4
granite median partial rho 0.002819 -> round 0.003
granite wilcoxon vs 0 p 0.37178 -> round 0.372
```

**Figure 12: median partial Spearman with years of delinquency equals 0.003 after controlling for assessed value.** Reproduced 0.002819 -> 0.003. `summary.txt` confirms the control is partial Spearman on log_appvalue and the median is +0.0028. **PASS.**

**Figure 13: one-sided p equals 0.372.** Reproduced 0.37178 -> 0.372. `summary.txt` labels the test "Wilcoxon signed-rank (one-sided, vs 0, n=16): granite: W=75.0, p=0.3718". One-sided confirmed. **PASS.**

**Figure 14: 16 non-sparse tracts entered the pilot.** Reproduced n_primary_tracts = 16, with 4 sparse tracts excluded. `summary.txt`: "Analysis frame: 39,535 addresses, 16 primary tracts, 4 sparse excluded." **PASS.**

### Group 4: Synthesis claims (figures 15 and 16)

**Figure 15: within-tract variance concentrates in building and parcel attributes; variance share by feature family.**

```
$ python3 -c "import pandas as pd;d=pd.read_csv('experiments/ecological_fallacy/variance_decomposition.csv');print(d.groupby('feature_class').agg(n=('feature','size'),median_within_share=('within_share','median'),median_eta_sq=('eta_sq','median')).round(4).sort_values('median_within_share',ascending=False).to_string())"
                n  median_within_share  median_eta_sq
feature_class
building       19               0.9060         0.0940
modal          15               0.4727         0.5273
accessibility  30               0.2304         0.7696
coordinate      2               0.0213         0.9787
socioeconomic   9              -0.0000         1.0000
```

Within-tract variance share by family: building 0.906, modal 0.473, accessibility 0.230, coordinate 0.021, socioeconomic ~0. Building carries the highest within-tract share and the lowest between-tract share (eta_sq 0.094). **PASS with taxonomy caveat:** the committed taxonomy uses a single `building` class of 19 features that bundles parcel attributes (log_appvalue, log_acres, build_to_land_ratio, LUCODE and PROPTYPE one-hots), footprint, flood (in_sfha), and NLCD. There is no separate `parcel` feature_class, so "building and parcel" both map to this one family. See Caveats.

**Figure 16: coordinates support no supervised ceiling above 0.13; identity with figure 6.**

```
$ python3 -c "import pandas as pd;d=pd.read_csv('data/results/m6_recovery_grid/recovery_grid.csv');c=d[d.method=='ceiling_gbm'];print('ceiling_gbm grand mean recovery_r',round(c.recovery_r.mean(),4),'| strong-autocorr cell mean',round(c[c.autocorr=='strong'].recovery_r.mean(),4),'| max single',round(c.recovery_r.max(),4))"
ceiling_gbm grand mean recovery_r 0.1309 | strong-autocorr cell mean 0.2253 | max single 0.8697
```

The 0.13 is the same `ceiling_gbm` grand mean (0.1309) that produces figure 6, from the same artifact; the identity with figure 6 is confirmed exactly. **PASS with supremum caveat:** "no supervised ceiling above 0.13" holds only as a grid grand-mean statement. Per-autocorr the supervised ceiling rises to 0.2253 at strong autocorr (and single-tract ceilings reach 0.87), so 0.13 is an average, not an upper bound. `Research_Status.md:395` frames the strong-autocorr ceiling as 0.182 (3-tract probe) and describes it as modest but nonzero. See Caveats.

---

## Consistency answers

### A. Does 81 draws reconcile with three factors at three levels (27 cells)? What design yields 81?

**Yes: 27 cells times 3 replicate seeds equals 81. There is no fourth factor.** The seed axis supplies the three replicate draws per cell.

```
$ python3 -c "import pandas as pd;d=pd.read_csv('data/results/m6_recovery_grid/recovery_grid.csv');print('cells',d[['autocorr','snr','between_tract']].drop_duplicates().shape[0]);print('seeds',sorted(d.seed.unique()));print('draws',d[['autocorr','snr','between_tract','seed']].drop_duplicates().shape[0])"
cells 27
seeds [np.int64(17), np.int64(42), np.int64(123)]
draws 81
```

The three factors (autocorr, snr, between_tract) at three levels each give 27 cells (figure 3). Each cell is drawn three times under seeds {17, 42, 123}, giving 27 x 3 = 81 distinct draws (figure 1). This is corroborated by the 81 scratch draw files named `draw_latent__{autocorr}__{snr}__{between_tract}__{seed}.json` in `experiments/m6_recovery_grid/scratch/`. So figure 1 (81) and figure 3 (27 cells) reconcile through the seed replicate axis, not a fourth design factor.

### B. Is the 0.763 to 0.772 pooled range the rewiring experiment, and separate from the 0.769 pooled GRANITE parity figure? Confirm different runs.

**Yes. Two different runs, two different committed files, two different commits.**

The 0.763 to 0.772 range is the topology rewiring experiment (5b), from `data/results/m6_topology_5b/topology_specificity_metrics.json`, the per-condition mean `pooled_bg_r` across the four conditions and two architectures:

```
$ git log -1 --format="%h %ci %s" -- data/results/m6_topology_5b/topology_specificity_metrics.json
5c54c9d 2026-07-06 02:22:49 +0000 5b: corrected topology specificity metrics under fixed Moran's I estimator
```

The 0.769 pooled GRANITE parity figure is the M0 n20 SVI parity run, from `data/results/m0_n20_svi_parity/aggregate.csv`:

```
$ cat data/results/m0_n20_svi_parity/aggregate.csv
method,median_bg_r,ci_low_95,ci_high_95,n_tracts,pooled_bg_r,pooled_bg_r_ci_low,pooled_bg_r_ci_high,pooled_n_bgs
GRANITE,0.3901,-0.4452,0.6974,19,0.7692,0.6602,0.8527,69
Dasymetric,0.7867,0.2531,0.8627,19,0.8018,0.7118,0.871,69
Pycnophylactic,0.2078,-0.3526,0.5289,19,0.7678,0.6516,0.8579,69

$ git log -1 --format="%h %ci %s" -- data/results/m0_n20_svi_parity/aggregate.csv
5859ed0 2026-06-24 00:47:36 +0000 Track m0 parity results; anchor negations for trapped result CSVs
```

They are distinct: the parity figure is a single production GRANITE configuration giving pooled_bg_r 0.7692; the rewiring range is a five-seed sweep across four topology conditions giving per-condition means 0.7628 to 0.7716. Within the rewiring artifact the `production` condition (five-seed mean 0.7632) is the closest analog and is described in `RESULTS.md` as the anchor to the step-5 canonical result. The M0 parity value 0.769 and the 5b topology range 0.763 to 0.772 are different runs committed on different dates (2026-06-24 vs 2026-07-06).

### C. Does each experiment have a committed, dated artifact rather than an uncommitted notebook cell?

**Yes. All three experiments have git-tracked, dated artifacts.**

```
$ for f in data/results/m6_recovery_grid/recovery_grid.csv \
           data/results/m6_topology_5b/topology_specificity_metrics.json \
           experiments/ablation/05b_topology_specificity/RESULTS.md \
           experiments/recovery/delinquency_convergent_validity/results.json \
           experiments/ecological_fallacy/variance_decomposition.csv ; do
    git ls-files --error-unmatch "$f" >/dev/null 2>&1 && echo "TRACKED  $(git log -1 --format='%h %ci' -- "$f")  $f"
  done
TRACKED  fc212d4 2026-07-05 16:19:41 +0000  data/results/m6_recovery_grid/recovery_grid.csv
TRACKED  5c54c9d 2026-07-06 02:22:49 +0000  data/results/m6_topology_5b/topology_specificity_metrics.json
TRACKED  dc419ad 2026-07-06 03:12:09 +0000  experiments/ablation/05b_topology_specificity/RESULTS.md
TRACKED  5d047cd 2026-06-10 12:48:16 +0000  experiments/recovery/delinquency_convergent_validity/results.json
TRACKED  458cf42 2026-06-10 00:50:01 +0000  experiments/ecological_fallacy/variance_decomposition.csv
```

- Synthetic recovery grid: `recovery_grid.csv`, committed 2026-07-05 (fc212d4). Supervised-ceiling probe corroboration in `experiments/m6_recovery_grid/ceiling_probe_results.json`, committed 2026-06-26 (7844c93).
- Topology rewiring: `topology_specificity_metrics.json`, committed 2026-07-06 (5c54c9d), and its narrative `RESULTS.md`, committed 2026-07-06 (dc419ad).
- Delinquency pilot: `results.json` (with `summary.txt`, `per_tract_partial.csv`), committed 2026-06-10 (5d047cd).
- Variance decomposition (figures 7 and 15): `variance_decomposition.csv`, committed 2026-06-10 (458cf42).

No figure depends on an uncommitted notebook cell.

---

## Caveats

Four figures reproduce numerically but carry a prose qualifier the committee should see:

- **Figure 5 (scope).** Output Moran's I 0.94 is the GraphSAGE coordinates_only sweep (0.9414). Averaged over all `morans_i_output` rows, which include GCN-GAT and the degraded diagnostic feature modes, the figure is 0.803. The 0.94 is correct for the primary grid sweep but is not an average over every populated row.
- **Figure 8 (rounding).** The upper bound of the structured Moran's I range is the spatial_knn_uniform GCN-GAT mean 0.88450, which rounds to 0.884 at three decimals. The committed RESULTS.md displays 0.885 and states the range as "0.817 to 0.885"; the claim traces to that text. Difference is one unit in the third decimal from display rounding.
- **Figure 15 (taxonomy).** "Building and parcel" both map to the single `building` feature_class (19 features) in the committed taxonomy, which bundles parcel, footprint, flood, and NLCD attributes. There is no separate `parcel` class; the reported family share (within_share 0.906) is for the combined class.
- **Figure 16 (supremum vs mean).** "No supervised ceiling above 0.13" holds only as the grid grand mean (0.1309). Per-autocorr, the strong cell mean is 0.2253 and single-tract ceilings reach 0.87, so 0.13 is the average coordinate ceiling, not an upper bound. The identity with figure 6 is exact.
