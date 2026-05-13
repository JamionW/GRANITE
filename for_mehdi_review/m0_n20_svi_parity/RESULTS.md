# M0 n20 SVI Parity Results

**Decision:** GRANITE (pooled BG r=0.769) is not statistically separable from Dasymetric (r=0.802) or Pycnophylactic (r=0.768) at BG resolution across the n20 subset; parity holds and the Narrative-A parity footnote survives.

Run date: 2026-05-09  |  n20 tracts attempted: 20  |  succeeded: 20  |  wall-clock: 13.2 min

---

## Aggregate metrics

### Pooled BG r (all 20 tracts combined, min 10 addresses/BG)

| Method | pooled_bg_r | CI low 95 | CI high 95 | n_BGs |
|--------|-------------|-----------|------------|-------|
| GRANITE | 0.769 | 0.660 | 0.853 | 69 |
| Dasymetric | 0.802 | 0.712 | 0.871 | 69 |
| Pycnophylactic | 0.768 | 0.652 | 0.858 | 69 |

### Per-tract median BG r (bootstrap on per-tract values)

| Method | median_bg_r | CI low 95 | CI high 95 | n_tracts_with_r |
|--------|-------------|-----------|------------|-----------------|
| GRANITE | 0.3901 | -0.4452 | 0.6974 | 19 |
| Dasymetric | 0.7867 | 0.2531 | 0.8627 | 19 |
| Pycnophylactic | 0.2078 | -0.3526 | 0.5289 | 19 |

## Pairwise comparisons

| Pair | obs_median_diff | CI low 95 | CI high 95 | n_pairs | separable |
|------|----------------|-----------|------------|---------|-----------|
| granite_vs_dasymetric | -0.1207 | -0.5356 | 0.108 | 19 | False |
| granite_vs_pycno | 0.0163 | -0.1068 | 0.2067 | 19 | False |
| dasymetric_vs_pycno | 0.4027 | 0.0435 | 0.6398 | 19 | True |

## Discussion

Primary metric is **pooled BG r**: all 20 tracts combined into one pool, addresses aggregated to BG means (min 10 addresses/BG), compared against nationally-ranked ACS BG SVI.

Per-tract BG r is also reported but has low statistical power: most tracts contain only 2-5 BGs after the min-address threshold, making per-tract r unreliable as a standalone metric.

Separability test: a pair is "separable" when the 95% bootstrap CI on median difference (from per-tract values) excludes zero.

### Constraint error sanity check

| Method | median_constraint_error_pct |
|--------|-----------------------------|
| GRANITE | 0.0000 |
| Dasymetric | 0.0000 |
| Pycnophylactic | 0.0000 |

Dasymetric and Pycnophylactic satisfy the aggregate constraint by construction (mean-preservation); nonzero values here indicate rounding or clipping effects only.

GRANITE constraint error reflects the soft-loss training penalty; values above 5% indicate the constraint was not well-enforced for that tract.
