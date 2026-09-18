# Study 1 variance components (Milestone A)

**Input:** `data/results/m6_recovery_grid/recovery_grid.csv` (7,200 rows, generator_commit
`cdc860a4d36cbadb0e6e9865cd31c55d89aaa81b`). Restricted to the paired GRANITE-minus-Dasymetric
per-tract recovery_r difference: GRANITE rows filtered to `feature_mode=coordinates_only,
arch=sage` (the primary 81-draw x 20-tract grid), merged against the 1620 Dasymetric rows on
`(autocorr, snr, between_tract, seed, tract_fips)`. 1,620 paired rows: 20 tracts x 81 draws
(3 autocorr x 3 snr x 3 between_tract x 3 seeds), balanced (81 draws per tract, verified).

**Script:** `scripts/variance_components_study1.py` (run with no arguments; writes this
directory's JSON).

**Estimator:** one-way random-effects ANOVA (balanced), classic moment estimator on the paired
diff, grouped by `tract_fips`:

- `sigma_seed^2 = MS_within` — residual variance of the diff around each tract's own mean,
  pooled across all 81 draws for that tract.
- `sigma_tract^2 = max((MS_between - MS_within) / n, 0)` — variance of the true per-tract mean
  diff across the 20 tracts, n=81 draws per tract.

**Naming caveat.** "Seed" here is shorthand for *all* non-tract draw-level variation: the 81
draws vary seed (3 levels) **and** autocorr/snr/between_tract (27 cells), and the grid does not
hold a cell fixed while re-seeding only. The two-component decomposition (tract vs. "seed")
pools the cell-level and true-seed variation into one residual term, per the milestone's
two-term framing. This is a simplification, not a literal isolation of reseed-only noise; a
finer nested model (tract x cell x seed) would separate them further.

**MDE convention:** two-sided alpha 0.05, 80% power target, normal (z) approximation
(`z_0.975 + z_0.80 = 2.8016`), matching `scripts/power_analysis_parity.py`'s stated convention
for interval/sample-size figures. At `n` tracts with each tract's diff averaged over `s` seeds:

```
MDE(s) = (z_alpha + z_beta) * sqrt(sigma_tract^2 + sigma_seed^2 / s) / sqrt(n)
```

**Results** (see `study1_variance_components.json`):

- sigma_tract = 0.0468, sigma_seed = 0.0885, combined sd = 0.1001.
- MDE at 85 tracts: 0.0211 (3 seeds), 0.0186 (5 seeds), 0.0168 (9 seeds), 0.0158 (15 seeds).
- Seed variance share at 85 tracts falls below 25% of total per-tract variance at 11 seeds
  (continuous crossover 10.73).
- Threshold sigma_tract above which a 0.15 MDE target is unreachable at 85 tracts, even as
  seed count grows without bound: 0.4936 (spec reference 0.494; matches to 3 decimals).
  Observed sigma_tract (0.0468) does **not** exceed this threshold — the 0.15 target is
  reachable at 85 tracts on this metric, and comfortably so: even at 3 seeds the MDE (0.021)
  is roughly 7x smaller than 0.15.
