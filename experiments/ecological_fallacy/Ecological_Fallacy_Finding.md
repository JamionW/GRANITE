# Ecological Fallacy: Feature Taxonomy Finding

**Status:** grounded (2026-06-10). Supersedes ungrounded address-level r claims removed 2026-06-09.
**Primary artifact:** `experiments/ecological_fallacy/variance_decomposition.csv`
**Summary:** `experiments/ecological_fallacy/variance_decomposition_summary.md`

---

## The generalizable point

Aggregate-level correlates do not guarantee sub-aggregate signal. A feature that correlates strongly
with SVI across tracts may carry almost no within-tract variance, in which case it cannot drive
within-tract disaggregation regardless of model architecture or constraint enforcement. The variance
decomposition makes this precise: eta_sq (between-tract variance share) and |tract_svi_r| (aggregate
SVI coupling) are near-orthogonal across the 73-feature stack plus coordinates.

Pearson(eta_sq, |tract_svi_r|) = -0.052 (n=75). The subset excluding the two coordinate columns
and nine socioeconomic constants gives -0.269 (n=64). The ecological-fallacy coupling -- SVI-predictive
features having less within-tract signal -- is weak and slightly negative, not strongly positive.

---

## Mechanism (revised)

The prior claim -- that accessibility features are approximately flat within tracts and that raw
spatial coordinates recover within-tract signal -- is not supported by the committed variance
decomposition.

**What the decomposition shows:**

Accessibility features carry meaningful within-tract variance. Median within_share = 0.230 across 30
accessibility features. Min-time and count-5min features reach within_share of 0.50 to 0.78
(`employment_min_time` within_share=0.783, `employment_time_range` within_share=0.787,
`grocery_min_time` within_share=0.503, `healthcare_count_5min` within_share=0.623).

Coordinates are near-perfect tract proxies. lat eta_sq=0.979, lon eta_sq=0.979. Only 2.1% of
coordinate variance is within-tract. A model trained on coordinates is learning which tract an
address belongs to, not where within the tract it sits. The aggregate constraint forces predictions
toward the known tract SVI, which is already approximated by the coordinate signal. High
block-group-level validation performance under the coordinates-only condition reflects tract-mean
recovery, not address-level disaggregation.

The most aggregate-SVI-predictive features are the socioeconomic constants (|r| up to 0.800), which
are broadcast at tract level and carry zero within-tract variance by construction. Among non-constant
features, building attributes (parcel, land-use, NLCD) have the highest within-tract share (median
0.906) and comparable aggregate SVI coupling (median |r| 0.413) to the accessibility class (0.358).

**Summary of the four-class picture:**

- socioeconomic: eta_sq 1.000, within_share 0.000 -- pure aggregate, no within-tract discrimination
- coordinate: eta_sq 0.979, within_share 0.021 -- tract proxy, not address-level signal
- accessibility: eta_sq median 0.770, within_share median 0.230 -- intermediate on both axes
- building: eta_sq median 0.094, within_share median 0.906 -- highest within-tract variance and
  non-trivial SVI coupling; most likely class to carry genuinely informative within-tract signal

Modal features (upgraded from tract-level constants to per-address OSRM computation; see
`docs/FEATURES.md:52-64`) span eta_sq 0.002 to 0.861, overlapping both accessibility and building
ranges. Feature class label does not determine within-tract concentration for modal features.

---

## What is not established

The decomposition does not validate that high-within_share features are informative about
individual-address vulnerability. No address-level SVI ground truth exists (`granite/data/loaders.py`:
`get_address_truth_values(target='svi')` returns `None`). Within-tract variance is a necessary
but not sufficient condition for within-tract predictive validity.

An external address-level proxy target (e.g., tax delinquency rate, property insurance claims,
or a fine-grained administrative welfare indicator) would be required to test whether the
within-tract variance in building and min-time accessibility features tracks actual within-tract
vulnerability. That test is not yet run.

---

## Provenance note

The values r=0.671 (coordinates) and r=0.033 (accessibility) cited in earlier documentation were
propagated from a deleted experiment run with no surviving committed artifact. They have been removed
from all documentation (commit 7dc0617, 2026-06-09). The current finding replaces them with numbers
grounded in `experiments/ecological_fallacy/variance_decomposition.csv` (committed 2026-06-10).
