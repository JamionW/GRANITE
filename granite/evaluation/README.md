# granite/evaluation -- redundancy filter gate

## Purpose

The redundancy filter (M3.6) determines whether a candidate external target
is linearly or non-linearly reconstructible from the existing 73-feature
stack. A target recoverable by ridge or GBM within-tract is not an
independent validation criterion and must be rejected before GRANITE
training (Door 2 closed).

## Procedure

1. Select `n_tracts` (default 5) tracts deterministically:
   sort FIPS ascending, apply seeded shuffle (default seed=42), take first n.
   The same selection is reused for both the filter and downstream training
   to prevent leakage between target admissibility and evaluation.

2. Per selected tract, fit:
   - RidgeCV: alphas=logspace(-3, 3, 13), 5-fold CV for alpha selection,
     closed-form LOO predictions at selected alpha.
   - GBM: n_estimators=200, max_depth=3, lr=0.05, subsample=0.8,
     5-fold OOF predictions. Skipped when n_addresses < 50.
   Predictor matrix: full 73-feature stack, per-tract z-scored,
   zero-variance columns dropped. Target: z-scored within tract.

3. Compute median Pearson r across the selected tracts for each model.

4. Apply threshold: `max(median_ridge_r, median_gbm_r) < REDUNDANCY_THRESHOLD`

## Constants

- `REDUNDANCY_THRESHOLD = 0.5`
- `n_tracts = 5` (default)
- `seed = 42` (default)

## Decision semantics

- `is_admissible=True`: max of medians is below threshold. Target is not
  near-reconstructible from the feature stack. Proceed to GRANITE training.
- `is_redundant=True`: negation of is_admissible. Ridge or GBM recovered
  the target with r >= 0.5 in the median tract. Door 2 is closed: the target
  would not provide independent signal relative to the feature stack.

## Rationale

M3.5 showed that syntactically independent features can be near-monotone
functional proxies of accessibility features within tract geography. A high
within-tract reconstruction r means GRANITE's 73-feature input already
encodes the target, so GRANITE's advantage over a non-constrained baseline
cannot be attributed to external information the model lacked.

## Outputs

Written to the run output directory:
- `redundancy_filter.json`: summary (medians, decision, threshold, tracts used)
- `redundancy_filter_per_tract.csv`: per-tract ridge_r, gbm_r, n_addresses
