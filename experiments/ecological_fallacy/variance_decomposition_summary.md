# Feature Taxonomy: Between-Tract Variance and SVI Coupling Are Near-Orthogonal

**Input:** `experiments/ecological_fallacy/n20_feature_matrix.csv` (39,535 rows, 20 tracts, 73 named features + lat + lon)
**Artifact:** `experiments/ecological_fallacy/variance_decomposition.csv` (75 rows, one per analyzed column)
**Method:** one-way ANOVA partition by tract FIPS. No normalization applied. Natural-unit values throughout.

---

## Primary finding

Aggregate SVI-predictive power does not imply within-tract disaggregation signal. Across 75 analyzed
columns, Pearson(eta_sq, |tract_svi_r|) = -0.052 (n=75). The relationship is near-zero and slightly
negative: features that correlate more strongly with tract-level SVI are not the features with more
within-tract variance -- if anything marginally less so.

The subset correlation excluding the two coordinate columns and the nine socioeconomic constants
(accessibility+modal+building, n=64) is -0.269. The direction holds after removing the two extreme
corners; the weak negative coupling is not an artifact of the anchor classes.

---

## Class-level summary

| feature_class | n | eta_sq median | eta_sq IQR [Q1, Q3] | within_share median | median \|tract_svi_r\| |
|---|---|---|---|---|---|
| coordinate | 2 | 0.9787 | [0.9786, 0.9788] | 0.0213 | 0.2219 |
| accessibility | 30 | 0.7696 | [0.5546, 0.8252] | 0.2304 | 0.3578 |
| modal | 15 | 0.5273 | [0.2121, 0.6588] | 0.4727 | 0.2435 |
| socioeconomic | 9 | 1.0000 | [1.0000, 1.0000] | 0.0000 | 0.5716 |
| building | 19 | 0.0940 | [0.0238, 0.1200] | 0.9060 | 0.4127 |

---

## The four corners

**Socioeconomic (eta_sq 1.000, |tract_svi_r| up to 0.800).** Pure aggregate case. All nine features
are ACS variables broadcast as tract-level constants; within-tract variance is zero by construction.
`pct_uninsured` (r=0.800), `pct_poverty` (r=0.772), `pct_no_vehicle` (r=0.712) are the strongest
SVI predictors in the dataset. They are also the features that provide zero within-tract
discrimination. This is not a failure of the features; it reflects that SVI is defined from these
same ACS variables at the tract level. The within-tract direction cannot be recovered from features
that carry no within-tract information.

**Coordinates (eta_sq 0.979).** Near-perfect tract proxies. lat eta_sq=0.979, lon eta_sq=0.979;
only 2.1% of coordinate variance is within-tract. A model trained on lat/lon as its only features
is primarily learning which tract an address belongs to. The aggregate constraint then snaps
predictions to the known tract-mean SVI -- which is already what the coordinate signal approximates.
High block-group-level validation r under the coordinates-only condition is therefore expected and
does not indicate within-tract disaggregation skill.

**Building (eta_sq median 0.094, within_share median 0.906, |tract_svi_r| median 0.413).** The cell
where within-tract variance and SVI coupling coexist. Parcel attributes (`log_appvalue`,
`build_to_land_ratio`, `log_acres`) and one-hot land-use and property-type codes are largely
determined by the individual parcel, not the tract. Building is the class most likely to carry
genuinely informative within-tract signal. See rows tagged `building` in
`experiments/ecological_fallacy/variance_decomposition.csv`.

**Accessibility (eta_sq median 0.770, within_share median 0.230, |tract_svi_r| median 0.358).**
Intermediate on both axes. Median 23% of accessibility variance is within-tract. The three
`_percentile` features (`employment_percentile` eta_sq=0.000006, `healthcare_percentile`
eta_sq=0.000013, `grocery_percentile` eta_sq=0.000014) are within-tract rank percentiles by
construction and function as within-tract ordinality features, not between-tract signals. Min-time
and count-5min features reach within_share of 0.50 to 0.78 for employment and grocery. Accessibility
is not flat within tracts.

---

## Coupling correlations

| scope | n | Pearson(eta_sq, \|tract_svi_r\|) |
|---|---|---|
| all 75 columns | 75 | **-0.052** |
| accessibility + modal + building only | 64 | **-0.269** |

The predicted ecological-fallacy pattern -- a positive coupling between between-tract concentration
and aggregate predictive power -- does not appear. The empirical relationship is near-zero to weakly
negative. The building class, which has the lowest eta_sq (most within-tract), has comparable
aggregate SVI coupling to socioeconomic (|r| medians 0.413 vs 0.572). The taxonomy does not cleanly
separate predictive and non-predictive feature classes.

---

## Modal features note

Modal features carry substantial within-tract variance: median eta_sq = 0.527, range [0.002, 0.861].
This is consistent with per-address OSRM computation (`docs/FEATURES.md:52-64`); modal features were
upgraded from tract-level constants to per-address drive/walk time comparisons. Feature class does not
partition the effect: modal eta_sq overlaps both accessibility and building ranges.
`employment_forced_walk_burden` (eta_sq=0.002) and `grocery_forced_walk_burden` (eta_sq=0.033) are
almost entirely within-tract; `employment_walk_effective_access` (eta_sq=0.861) is nearly as
between-tract as mean travel time features.

---

## Zero-variance features

None. All 75 analyzed columns have finite variance across the 39,535 rows.

---

## Caveat

eta_sq measures where variance is located -- between tracts or within tracts -- not whether
within-tract variation is informative about within-tract vulnerability. No address-level SVI ground
truth exists: `get_address_truth_values(target='svi')` in `granite/data/loaders.py` returns `None`
by design (SVI is defined only at the tract level). A feature with high within_share may vary within
tracts for reasons unrelated to individual-address vulnerability. A feature with low within_share may
still drive informative between-address ordering within a tract if the within-tract variation is
structured.

The taxonomy locates candidate features for within-tract disaggregation; it does not validate them.
Within-tract variance share is a necessary but not sufficient condition for within-tract predictive
validity. Validation requires an address-level proxy target -- an open test not yet run.
