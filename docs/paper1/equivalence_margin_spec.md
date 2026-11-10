# Equivalence margin spec for Paper 1 pooled block-group r (BOOT-P1)

Committed before any BOOT-P1 bootstrap runs. The bootstrap script and its results land in a later commit.

## Margin

The primary margin is Cohen's q = 0.10 on the Fisher z scale. The test statistic is the C1 z-difference, d_z = atanh(r_GRANITE) minus atanh(r_Dasymetric), on M0 pooled block-group r against reference v2 (67 scoring BGs, 20 tracts).

The anchor is r = 0.797, the Dasymetric pooled BG r. The comparator sets the margin, so the r-scale bounds are taken around the comparator's value.

r-scale bounds, computed with numpy 2.4.6:

```
atanh(0.797)                 = 1.090333953870886
tanh(1.090333953870886 + 0.10) = 0.8306824118481332
tanh(1.090333953870886 - 0.10) = 0.7575046869062133
upper bound: 0.8306824118481332 - 0.797 = +0.03368241184813314  (+0.034)
lower bound: 0.7575046869062133 - 0.797 = -0.03949531309378673  (-0.039)
```

The difference is reported on both scales: d_z against plus or minus 0.10, and d_r = r_GRANITE minus r_Dasymetric alongside the bounds -0.039 / +0.034.

## Decision rule

The interval is the 90% percentile CI of d_z from a tract-clustered bootstrap (tracts resampled with replacement, B = 10,000, seed 20261109).

- Equivalent: the 90% CI lies inside (-0.10, +0.10).
- Difference: the 90% CI excludes zero and crosses or lies outside the margin.
- Small but inconclusive: otherwise (the 90% CI contains zero and crosses the margin).

## Sensitivity margin

Plus or minus 0.03 on the r scale, applied to the 90% CI of d_r with the same three-way rule. It was the earlier primary margin and is dropped as primary because it was set after the gap was seen. It is reported as a sensitivity result only.

## Approval

Dr. Lani Gao, email of Nov 3, 2026 (q = 0.10 on the Fisher z scale, reported also on the r scale). Her Oct 21 refinements stand: tract-clustered bootstrap of the paired difference, seed and graph-draw variance reported separately, equivalence claimed only if the 90% CI sits inside the margin.

## Scope

The margin covers C1 only (M0 GRANITE minus Dasymetric). C2 (structured minus randomized pooled BG r, per architecture) is descriptive: differences and intervals are reported with no equivalence verdict.
