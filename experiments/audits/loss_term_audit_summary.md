# GRANITE loss term audit -- summary

**Audited:** 2026-05-27
**Scope:** MultiTractGNNTrainer._compute_multi_tract_losses, AccessibilityGNNTrainer._compute_losses, plus compare_gnn_idw.py smoothness term (scope addendum).
**Total terms audited:** 11

## Flagged terms

- **variation_loss (entries 003, 008) -- orphaned config key.** `config.yaml` declares
  `training.variation_weight: 1.5` but neither trainer reads it. The multi-tract trainer
  uses hardcoded `0.8`; the single-tract trainer uses hardcoded `1.5` (coincidentally
  matching the config value but not reading it). The config key has no effect. See entries
  003 and 008.

- **variation_loss (entries 003, 008) -- hidden activation threshold.** Internal constant
  `min_variation = 0.02` is defined inside the function body, not visible at the call site
  and not configurable. Same pattern as the deleted smoothness term's `* 0.05`, though less
  severe (it is a threshold, not a scale multiplier). See entries 003 and 008.

- **range_loss (entry 010) -- degenerate gradient routing.** Gradient flows only through
  `max()` and `min()`, reaching exactly one address each. All other addresses receive zero
  gradient from this term. Informational flag; not a bug. See entry 010.

- **accessibility_consistency_loss (entry 011) -- misleading name, partial verdict.**
  The name implies a consistency check between accessibility features and predictions.
  No accessibility features enter the computation. The term is a minimum-spread penalty
  on sorted predictions, functionally overlapping with `variation_loss` and `range_loss`.
  The dict key (`'accessibility'`) further truncates the already-misleading name. See entry 011.

## Summary table

| # | Term | Function | Verdict | Internal constants | Recommendation |
|---|------|----------|---------|--------------------|----------------|
| 001 | smoothness_loss (compare_gnn_idw) | train_constrained_gnn | yes | no | keep |
| 002 | constraint_loss | _compute_multi_tract_losses | yes | no | keep |
| 003 | variation_loss | _compute_multi_tract_losses | yes | yes (min_variation=0.02; orphaned config key) | keep; fix config key |
| 004 | bounds_loss | _compute_multi_tract_losses | yes | no | keep |
| 005 | bg_constraint_loss | _compute_multi_tract_losses | yes | no | keep |
| 006 | ordering_loss | _compute_multi_tract_losses | yes | no | keep |
| 007 | constraint_loss | _compute_losses (single-tract) | yes | no | keep |
| 008 | variation_loss | _compute_losses (single-tract) | yes | yes (min_variation=0.02; orphaned config key) | keep; fix config key |
| 009 | bounds_loss | _compute_losses (single-tract) | yes | no | keep |
| 010 | range_loss | _compute_losses (single-tract) | yes | yes (min_range=0.05; degenerate gradient) | keep; informational |
| 011 | accessibility_consistency_loss | _compute_losses (single-tract) | partial | yes (threshold=0.001) | rename to min_spread_loss |

## Overall assessment

The loss terms collectively do what their names suggest with one exception:
`accessibility_consistency_loss` (entry 011) is misnamed -- it is a minimum-spread penalty
on sorted predictions with no connection to accessibility features, and it overlaps
functionally with `variation_loss` and `range_loss`. The single-trap trainer
(`AccessibilityGNNTrainer._compute_losses`) carries three distinct spread-encouragement
terms (variation, range, accessibility_consistency) that partially duplicate each other;
the multi-tract trainer uses only one (variation_loss). No interaction effects between
terms rise to the level of dissertation risk: the constraint term dominates at weight 2.0,
and all spread terms are hinge-based (zero gradient when conditions are already met), so
they cannot interfere with constraint satisfaction. The one systemic anti-pattern beyond
the removed smoothness term is the orphaned `variation_weight: 1.5` config key, which
creates false user-facing control surface. It should be removed or wired before any
hyperparameter documentation is written for step 4.
