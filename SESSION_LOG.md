# GRANITE Session Log

## 2026-06-23: M5 patch -- between-tract variance injection, two-sided WTVR guard, regeneration contract

**Files modified:**
- `granite/synthetic/generator.py`: four changes (see below)
- `granite/scripts/run_m5_smoke.py`: updated to read renamed diagnostics key `wtvr_achieved` (was `within_tract_variance_ratio`); added `wtvr_target` and `generator_commit` printout

**Files created:**
- `data/synthetic/calibration/svi_variance_decomposition.json`: population-weighted SVI variance decomposition for 20 n20 tracts, 73 Hamilton County BGs

**Changes in generator.py:**

1. `compute_svi_variance_decomposition()` (new module-level function): loads `data/processed/national_bg_svi.csv`, filters to n20 BGs (GEOID prefix 47065, tract in n20 list), computes population-weighted between_var=0.04824 and within_var=0.02428 over tract-BG hierarchy, saves JSON. ratio_between=0.6652 (66.5% of real SVI variance is between-tract).

2. `__init__` now loads the calibration JSON on construction and raises FileNotFoundError with instructions if missing.

3. Between-tract effect injection (`_inject_tract_effect`): y_within = y_pre + noise; then _inject_tract_effect draws per-tract N(0,1) effects, centers, scales to sigma_between derived from wtvr_target=0.3348. wtvr_achieved lands ~0.30-0.33 across all three smoke configs (see below).

4. Two-sided WTVR guard: existing lower bound (<0.05) kept; upper bound (>0.95) added immediately after.

5. Diagnostics: renamed `within_tract_variance_ratio` to `wtvr_achieved`; added `wtvr_target`, `sigma_between`, `tract_effect_variance`.

6. Metadata: added `generator_commit` (git rev-parse HEAD via subprocess).

**Cache invalidation:** none. Per-tract GP, length-scale calibration, and Moran's I weights are untouched.

**Smoke results (seed=42):**

| Run | autocorr | snr | wtvr_target | wtvr_achieved | MI achieved | MI target | in-band |
|-----|----------|-----|-------------|---------------|-------------|-----------|---------|
| 1   | medium   | medium | 0.3348   | 0.3275        | 0.3940      | 0.4000    | YES     |
| 2   | strong   | high   | 0.3348   | 0.3272        | 0.6972      | 0.7000    | YES     |
| 3   | weak     | low    | 0.3348   | 0.2982        | 0.1191      | 0.1000    | YES     |

**Knob checks:**
- autocorr medium -> strong: morans_i_achieved 0.3940 -> 0.6972 (rises, correct)
- snr medium -> high: noise_variance 1.0000 -> 0.3333 (noise share drops, correct for high SNR)

Both guards stayed silent on all runs.

## 2026-06-10: Feature taxonomy -- variance decomposition on n20 matrix (Milestones A-C)

**Files created:**
- `scripts/capture_feature_matrix.py`: assembles 73-feature matrix per tract via pipeline methods; no GNN training; no normalization
- `scripts/variance_decomposition.py`: one-way ANOVA partition (eta_sq, within_share) + tract_svi_r coupling; no normalization
- `experiments/ecological_fallacy/n20_feature_matrix.csv`: 39,535 rows x 78 cols (fips, address_idx, lat, lon, tract_svi + 73 features), 20 n20 tracts, natural units
- `experiments/ecological_fallacy/variance_decomposition.csv`: 75 rows, one per analyzed column (73 features + lat/lon)
- `experiments/ecological_fallacy/variance_decomposition_summary.md`: class-level table, four-corners prose, coupling correlations, caveat
- `experiments/ecological_fallacy/Ecological_Fallacy_Finding.md`: supersedes deleted ungrounded r=0.671/0.033 claims; new mechanism grounded in committed CSV

**Files modified:**
- `.gitignore`: added `!experiments/**/*.csv` negation so artifacts under experiments/ are not masked by `*.csv` rule
- `Research_Status.md`: added locked-in finding 7 (feature taxonomy) with caveat; no other findings renumbered
- `SESSION_LOG.md`: this entry

**What superseded what:**
- Ungrounded claims r=0.671 (coordinates) and r=0.033 (accessibility) were removed in commit 7dc0617 (2026-06-09) because they traced to a deleted output with no surviving artifact.
- Those claims have been replaced by committed variance decomposition numbers: eta_sq=0.979 for coordinates (near-perfect tract proxy), accessibility median within_share=0.230 (not flat within tracts).
- The old ecological-fallacy framing (coordinates outperform accessibility at address level) is superseded by the taxonomy finding: aggregate SVI coupling and within-tract variance share are near-orthogonal, not positively coupled as the fallacy story required.

**Key numbers (from variance_decomposition.csv):**
- Coupling correlation Pearson(eta_sq, |tract_svi_r|): -0.052 (n=75)
- Coupling correlation accessibility+modal+building subset: -0.269 (n=64)
- Coordinate median eta_sq: 0.979 (97.9% between-tract)
- Accessibility median eta_sq: 0.770 (within_share 0.230)
- Building median eta_sq: 0.094 (within_share 0.906)
- Socioeconomic median eta_sq: 1.000 (zero within-tract variance by construction; passes invariant check)

**Caveat recorded:**
eta_sq measures variance location, not predictive validity. No address-level SVI ground truth exists.
The taxonomy locates candidate features; it does not validate them.

**Cache invalidation notes:** none; capture script reads cached accessibility features; no routing logic changed.

## 2026-06-09: Remove ungrounded ecological-fallacy figures (r=0.671/0.033)

**Files modified:**
- `Research_Status.md`: removed item 2 ("Ecological fallacy at address scale"; r=0.033/r=0.671 assertion); renumbered items 3-7 to 2-6
- `docs/FEATURES.md`: removed "Empirical finding" subsection (r~0.03/r~0.67 claims)
- `CLAUDE.md`: removed two paragraphs citing r~0.67/r~0.03 from "Feature matrix" and "Key result reference points" sections
- `SESSION_LOG.md`: this prior entry (appended)

**What changed and why:**
All four assertion sites for r=0.671 (coordinates) and r=0.033 (accessibility) were surgical excisions.
Figures traced to `output/coord_artifact/` which is gitignored (`.gitignore:67`) and never committed.
On-disk `output/coord_artifact/bg_validation_report.txt` contains a *different* metric (BG-level r, not address-level r) and does not support the claim. `Ecological_Fallacy_Finding.md` did not exist.
The audit file `experiments/audits/outstanding_items_reconciliation.md` documents this provenance and was left untouched.

**Cache invalidation notes:** none.

## 2026-06-07: CPU torch pin; README and STARTUP corrections

**Files modified:**
- `requirements.txt`: pinned `torch==2.10.0+cpu` with `--index-url https://download.pytorch.org/whl/cpu` and `--extra-index-url https://pypi.org/simple`; no CUDA or nvidia libraries are pulled
- `STARTUP.md`: updated step 2 install timing to 1m7s (measured, semi-warm pip cache, CPU build); updated prose to describe CPU-only torch resolve; reduced troubleshooting block for torch/torch-geometric (CPU index is now the default path); updated step 4 runtime to 1.2s (measured in clean venv)
- `README.md`: five grounded corrections -- (1) coordinate claim gated to "ablation indicates, confirmatory re-run pending"; (2) install block now runs `pip install -r requirements.txt` before `pip install -e .`; (3) feature count reconciled to 73/19 across Overview, Features, and Configuration sections; (4) time-based example replaces 3km distance example in Key Research Findings; (5) Python requirement tightened to 3.11
- `Research_Status.md`: last-updated date

**Verification:**
- Clean virtualenv install: 1m7s, no nvidia or cuda packages
- `python -c "from granite.models.gnn import AccessibilitySVIGNN; print('ok')"` output: `ok`
- `python experiments/ablation/00_baseline_current/regen_figures.py` completed successfully, 1.2s total
- Feature count grounded at pipeline.py: PROPTYPE_VOCAB has 5 entries; address-level = 7 individual + 4 LUCODE one-hot + 5 PROPTYPE one-hot + 3 NLCD = 19; full total = 30+15+9+19 = 73

**Cache invalidation notes:** none; no routing or feature extraction logic was changed.

## 2026-06-06: STARTUP.md corrections (verified install, paths, baselines)

**Files created:**
- `experiments/ablation/00_baseline_current/regen_figures.py`: regen script reading committed Dasymetric/Pycnophylactic results; produces bg_r_by_tract.png and aggregate_summary.png
- `experiments/ablation/00_baseline_current/results/per_tract.csv`: copy of m0_n20_svi_parity per-tract metrics (GRANITE, Dasymetric, Pycnophylactic), force-added to git
- `experiments/ablation/00_baseline_current/results/aggregate.csv`: copy of m0_n20_svi_parity pooled metrics, force-added to git

**Files modified:**
- `STARTUP.md`: complete rewrite -- canonical run now points at 00_baseline_current (current baselines), step 3 uses AccessibilitySVIGNN import, install timing reflects measured warm-cache (3-5s) vs cold Codespace estimate, output paths verified CWD-independent, IDW/kriging retired-baseline note added
- `experiments/ablation/00_baseline/regen_figures.py`: fixed print messages to show absolute paths (FIGURES_DIR / filename) instead of hardcoded relative strings; CWD-independence confirmed by running from /tmp
- `.devcontainer/devcontainer.json`: added postCreateCommand to auto-install dependencies on Codespace creation

**What changed and why:**
- original STARTUP.md used invented install timing, wrong import check, misleading output path, and surfaced IDW/kriging as current baselines
- clean-venv install (semi-warm pip cache): 2m24s measured; torch 2.12.0 with CUDA runtime resolved; no compiler required; fully warm cache < 10s
- 00_baseline block_group_validation.json carries IDW/kriging keys (retired); 00_baseline_current points at m0 Dasymetric/Pycnophylactic results
- canonical run changed from 00_baseline regen to 00_baseline_current regen

**Cache invalidation notes:** none.

## 2026-06-06: STARTUP.md authorship (first-time setup guide)

**Files created:**
- `STARTUP.md`: zero-context setup guide for committee members opening the repo in Codespaces

**Files modified:**
- `README.md`: added single line near top pointing to STARTUP.md for first-time setup

**What changed and why:**
- audited the clean-clone runnable surface: data/raw/, data/processed/, and granite_cache/ are all gitignored; a fresh clone has no data for the main granite CLI
- canonical run identified as `python experiments/ablation/00_baseline/regen_figures.py`, which reads two committed result files (per_tract_metrics.csv, block_group_validation.json) and produces three PNG figures; verified to run in ~7 seconds with no OSRM or external data dependency
- OSRM and full pipeline steps placed in optional/unverified section only
- devcontainer postStartCommand (start_osrm.sh) noted to fail gracefully when OSRM data is absent; does not affect canonical run

**Cache invalidation notes:** none -- no changes to feature extraction, routing, or training.

## 2026-06-05: baseline unification + BG-r metric provenance audit

**Files created:**
- `experiments/audits/baseline_metric_provenance.md`: full provenance report mapping all four BG-r numbers to their source metric, file, and validation context

**Files modified:**
- `README.md`: Project Structure `evaluation/` line updated to name Dasymetric/Pycnophylactic and note IDW/Kriging retired to graveyard; new Baselines subsection added before Installation, stating headline baseline (Dasymetric), secondary baseline (Pycnophylactic), IDW/Kriging framing as retired degenerate proximity floor, pooled BG r as primary metric, and per-tract BG r as within-tract divergence diagnostic

**What changed and why:**
- four BG-r numbers in circulation (0.469, 0.558, 0.769/0.749, 0.772/0.768) were mapped to their distinct metrics:
  - 0.469/0.558: Metric D (global held-out BG r, legacy), from `bg_validation_summary.csv` (root), written by unknown legacy script predating the n20 harness; n_predictions=192; superseded
  - 0.769/0.749: Metric A (pooled BG r across n20 tracts, n=69 BGs), from `experiments/ablation/00_baseline/results/block_group_validation.json`; canonical for architecture comparison
  - 0.772/0.768: same Metric A for IDW/Kriging; recorded in 00_baseline frozen artifact; both methods now retired to graveyard
  - canonical current numbers: 0.769 GRANITE, 0.802 Dasymetric, 0.768 Pycnophylactic from `data/results/m0_n20_svi_parity/aggregate.csv`
- README.md baseline identity inconsistency resolved: Dasymetric is headline baseline; Pycnophylactic is secondary; IDW/Kriging appear only as retired

**Cache invalidation notes:** none -- no changes to feature extraction, routing logic, or training.

**Frozen artifact note:** no ablation result files were modified. The 00_baseline block_group_validation.json correctly records IDW/Kriging because those were the active baselines at run time.

## 2026-05-29: step 4b sweep complete -- Outcome C, soft mode selected for step 5

**Files changed:**
- `granite/disaggregation/pipeline.py`: `_train_multi_tract_gnn` and
  `_train_accessibility_svi_gnn` now forward `variation_loss_activation_rate` from the trainer
  result into their return dict. Previously the key was computed by the trainer but discarded
  by the pipeline before the ablation script could read it (causing NaN in aggregate_metrics).

**Sweep results (cbc_no_shift, 20 tracts, seed 42):**

| weight | SAGE std | SAGE BG r | GCN-GAT std | GCN-GAT BG r | SAGE act_rate |
|---|---|---|---|---|---|
| 0.8 | 0.0595 | 0.7511 | 0.0829 | 0.7480 | 0.0 |
| 1.5 | 0.0595 | 0.7511 | 0.0830 | 0.7481 | 0.0 |
| 2.5 | 0.0595 | 0.7511 | 0.0833 | 0.7481 | 0.0 |
| 4.0 | 0.0595 | 0.7511 | 0.0834 | 0.7480 | 0.0 |
| soft (ref) | 0.0823 | 0.7537 | 0.0814 | 0.7664 | - |

**Verdict: Outcome C.** SAGE within-tract std is identical across all four weights (0.0595).
The variation hinge (min_variation=0.02) never fired for SAGE at any weight -- activation
rate is 0.0 across the full sweep. SAGE predictions are already above the 0.02 floor so the
hinge is structurally dormant. The spread collapse from 0.082 to 0.060 under cbc is not a
calibration artifact; it is a structural consequence of removing the constraint loss from the
training objective. Raising variation_weight has no effect because the mechanism that would
drive it cannot engage.

**Step 5 decision:** use `constraint_mode: soft`. The cbc spread collapse is a documented
finding. See `experiments/ablation/04b_variation_weight_recalibration/summary/README.md`.

**Cache invalidation:** none.

---

## 2026-05-29: step 4b setup -- wire variation_weight, create sweep infrastructure

**Files changed:**
- `granite/models/gnn.py`:
  - `MultiTractGNNTrainer.__init__`: removed fail-fast ValueError for `variation_weight`; replaced
    with `self.variation_weight = config.get('variation_weight', 0.8)`. Default 0.8 is backward
    compatible with the prior hardcoded value.
  - `MultiTractGNNTrainer.__init__`: added `'variation_activation_count': 0` to `training_history`.
  - `MultiTractGNNTrainer.train`: added per-epoch tracking of hinge activation
    (`losses['variation'].item() > 0`); added `variation_loss_activation_rate` to results dict
    (count / epochs_trained).
  - `_compute_multi_tract_losses`: replaced hardcoded `0.8 * variation_loss` with
    `self.variation_weight * variation_loss`.
- `config.yaml`: re-added `variation_weight: 0.8` under `training:` with inline comment
  `# multi-tract trainer only; see step 4b sweep`.
- `tests/test_loss_terms.py`: replaced `TestVariationWeightFailFast` (6 tests that expected
  ValueError from multi-trainer) with `TestVariationWeightWiring` (6 tests: reads 1.5, defaults
  to 0.8, explicit matches implicit, clean config passes, single-trainer still raises, single
  clean passes). All 20 tests pass.
- `experiments/ablation/04_constraint_by_construction/run_ablation_04.py`: removed stale
  `cfg.get('training', {}).pop('variation_weight', None)` line (comment said "hardcoded in
  trainer"; no longer true).

**Files created:**
- `experiments/ablation/04b_variation_weight_recalibration/run_ablation_04b.py`: sweep driver
  for four variation_weight values (0.8, 1.5, 2.5, 4.0) under cbc_no_shift, both architectures,
  same 20 tracts and seed as step 4. Includes sanity check (00_w_0p8 must reproduce step 4
  02_cbc_no_shift within 1e-4), all five stop conditions from spec, variation_activation_rate
  capture per variant, and three summary figures (variation_weight_sweep.png,
  spread_vs_generalization.png, extreme_tract_recalibration.png).
- `experiments/ablation/04b_variation_weight_recalibration/{00_w_0p8,01_w_1p5,02_w_2p5,03_w_4p0,summary}/`
  (empty subdirectories; populated by sweep runs).

**Why:** Step 4 found SAGE within-tract std collapsed under cbc_no_shift (0.060 vs 0.082 soft).
Hypothesis: calibration artifact. The variation_loss weight was tuned against soft-mode gradient
magnitudes; removing constraint_loss reduced the effective regularization pressure on spread.
Step 4b tests whether raising variation_weight from 0.8 to 4.0 recovers spread without
sacrificing generalization.

**Behavior change:** MultiTractGNNTrainer now reads variation_weight from config instead of
hardcoding 0.8. Default is 0.8, so all existing configs and runs that omit variation_weight
are bit-identical to prior behavior.

**Cache invalidation:** none. Feature extraction and OSRM routing are unchanged.

---

## 2026-05-27: audit followup -- remove orphaned variation_weight key, rename accessibility_consistency_loss

**Files modified:**
- `config.yaml`: removed `training.variation_weight: 1.5`. The key was declared but never
  consumed by either trainer (both hardcode their variation loss weights). See audit entries
  003 and 008.
- `granite/models/gnn.py`:
  - `AccessibilityGNNTrainer.__init__`: added fail-fast raising `ValueError` when
    `variation_weight` appears in a loaded config, matching the smoothness_weight pattern.
  - `MultiTractGNNTrainer.__init__`: same fail-fast for `variation_weight`.
  - `AccessibilityGNNTrainer._compute_losses`: renamed call from
    `_compute_accessibility_consistency_loss` to `_compute_min_spread_loss`; renamed local
    variable from `accessibility_consistency_loss` to `min_spread_loss`; changed loss dict
    key from `'accessibility'` to `'min_spread'`; added inline comments on hardcoded
    variation loss weights at both constrained and unconstrained call sites.
  - `_compute_multi_tract_losses`: added inline comments on hardcoded variation loss weights.
  - `_compute_accessibility_consistency_loss` renamed to `_compute_min_spread_loss`;
    docstring updated to accurately describe the sorted-prediction-gradient hinge.
- `tests/test_loss_terms.py`: expanded test suite with `TestVariationWeightFailFast`
  (6 tests) and `TestMinSpreadLoss` (4 tests); added single-tract loss dict key tests;
  all 20 tests pass.
- `experiments/audits/loss_term_audit.md`: updated entry 011 to record rename history
  with date and pointer to this SESSION_LOG entry.

**Why:** Loss term audit (entries 003, 008, 011) found: (1) `variation_weight: 1.5` in
config.yaml had no effect on either trainer; (2) `_compute_accessibility_consistency_loss`
was misnamed -- no accessibility features enter the computation; it is a minimum-spread
penalty on sorted predictions.

**Behavior change:** none. The rename is semantics-only. Loss values, weights, and
gradient routing are unchanged. Verified: all 20 loss-term regression tests pass.

**Cache invalidation:** none.

---

## 2026-05-22: ablation Step 3 cleanup -- remove misnamed smoothness loss

**Files modified:**
- `granite/models/gnn.py`: removed `_compute_cross_tract_smoothness` method; removed `smoothness_loss` computation and both call sites (constrained and unconstrained branches of `_compute_multi_tract_losses`); removed `'smoothness'` key from returned loss dict; replaced `self.smoothness_weight` init with a fail-fast guard that raises `ValueError` if `smoothness_weight` appears in any loaded config.
- `config.yaml`: removed `smoothness_weight: 0.1` key.

**Files created:**
- `experiments/ablation/03_smoothness/INSPECTION_FINDING.md`: complete pre-deletion record including verbatim function body, weight chain, results summary, extreme-tract collapse finding, and deletion manifest.
- `tests/test_loss_terms.py`: six regression tests -- expected loss dict keys present, `smoothness` key absent, total loss finite, fail-fast fires on `smoothness_weight` in config (including zero weight), clean config passes.

**What changed and why:**
- `_compute_cross_tract_smoothness` was misnamed: it computed `(max_tract_mean - min_tract_mean) * 0.05` with no graph structure. Effective gradient coefficient at default config was `smoothness_weight * 0.05 = 0.005`. The five-weight sweep (0.0 to 0.5) confirmed zero measurable effect on any metric (max absolute diff = 0.0 to machine precision across 40 tract-architecture rows). The constraint loss drives tract means to distinct SVI targets and dominates; the smoothness gradient is swamped. Removal changes nothing measurable.
- Fail-fast added to prevent reinstatement via config without reading the ablation record.
- See `experiments/ablation/03_smoothness/INSPECTION_FINDING.md` for full audit trail.

**Cache invalidation:** none. Feature and routing logic unchanged.

## 2026-05-18: ablation 00_baseline frozen reference run

**Files created:**
- `experiments/ablation/00_baseline/run_baseline.py` (driver; runs both architectures on all 20 in-scope tracts)
- `experiments/ablation/00_baseline/README.md` (manifest with metrics, artifact index)
- `experiments/ablation/00_baseline/git_state.txt`, `config_snapshot.yaml`, `environment.txt`, `tract_selection.txt` (pre-flight artifacts)
- `experiments/ablation/00_baseline/results/per_tract_metrics.csv` (40 rows: 20 tracts x 2 architectures)
- `experiments/ablation/00_baseline/results/aggregate_metrics.json`
- `experiments/ablation/00_baseline/results/block_group_validation.json`
- `experiments/ablation/00_baseline/results/feature_importance/{sage,gcn_gat}_permutation_importance.csv`
- `experiments/ablation/00_baseline/figures/*.png` (6 figures at 300 dpi)

**Files modified:**
- `granite/visualization/plots.py`: added 6 new ablation figure functions (plot_ablation_constraint_error_dist, plot_ablation_spatial_std_by_svi, plot_ablation_morans_i_by_tract, plot_ablation_block_group_scatter, plot_ablation_feature_importance_top20, plot_ablation_architecture_overlap)

**What changed and why:**
- creates the frozen anchor for the ablation series; subsequent steps 01-05 compare back to these results
- ran GRANITEPipeline single-tract SVI mode (n_neighbor_tracts=0) for all 20 in-scope tracts, both sage and gcn_gat architectures, seed=42
- IDW and OrdinaryKriging baselines loaded from graveyard/disaggregation_baselines_idw_kriging.py for pooled BG validation
- tract_inventory.csv has no Status column; all 20 rows treated as in-scope; noted in tract_selection.txt

**Key results (seed=42, n20 tracts):**
- constraint error: 0.000 for all 40 runs (post-correction applied)
- spatial std mean: 0.0797 (SAGE), 0.0887 (GCN-GAT)
- Moran's I mean: 0.833 (SAGE), 0.848 (GCN-GAT)
- pooled BG r: SAGE=0.769, GCN-GAT=0.749, IDW=0.772, Kriging=0.768
- per-tract BG r mean: SAGE=0.140, GCN-GAT=0.043 (within-tract signal weak, consistent with prior findings)
- SAGE pooled BG r (0.769) matches m0 parity reference (0.7692) to 3 decimal places

**Stop condition notes:**
- initial run flagged stop on wrong reference (r=0.469 from bg_validation_summary.csv global validation context); corrected reference to m0 parity pooled BG r=0.7692; actual deviation 0.0001, well within tolerance
- no NaN metrics on any tract

**Cache invalidation notes:** none -- no changes to feature extraction or OSRM routing logic.

## 2026-05-12: M5 synthetic target generator (Phase 2 build)

**Files created:**
- `granite/synthetic/__init__.py` (package init, exports SyntheticTargetGenerator)
- `granite/synthetic/generator.py` (530 lines): full M5 generator with per-tract GP autocorrelation injection, latent/feature signal sources, SNR mixing, and Moran's I diagnostics
- `granite/scripts/run_m5_smoke.py` (156 lines): three-configuration smoke driver

**What changed and why:**
- M5 implements the synthetic boundary-surface testbed needed to generate ground-truth targets for M6 GRANITE recovery experiments
- signal_source='latent' draws z ~ N(0,1) per address independent of the 73-feature matrix; signal_source='features' loads numeric features from `data/raw/address_features/combined_address_features.csv`
- spatial autocorrelation injected per-tract via Matern nu=1.5 GP; length scales binary-searched per tract using the FULL tract addresses (no subsampling) to avoid density mismatch between calibration and production
- critical bug discovered and fixed during calibration: GP samples have non-zero within-tract mean; failing to subtract it before normalization creates spurious between-tract contrast that inflates global Moran's I far beyond per-tract targets
- morans_i_achieved is computed on the GP residuals r (the calibrated component), not y_true; SNR noise would always reduce y_true MI below target, making the calibration warning meaningless
- dynamic NaN filter drops feature columns with >99% NaN (district, region are 100% NaN in combined_address_features.csv); hardcoded exclusion list updated accordingly

**Smoke driver results (seed=42):**

| Run | signal | autocorr | snr | achieved MI | target MI | in-band | WTVR |
|-----|--------|----------|-----|-------------|-----------|---------|------|
| 1 | latent/linear | medium | medium | 0.3940 | 0.4000 | YES | 0.9994 |
| 2 | latent/nonlinear | strong | high | 0.6972 | 0.7000 | YES | 0.9996 |
| 3 | features/linear | weak | low | 0.1191 | 0.1000 | YES | 0.9925 |

No calibration warning fired. All runs within Moran's I target band (+/- 0.05). metadata.json matches input params on all three runs. Runtime: ~4 minutes for three runs (dominated by per-tract GP calibration via Cholesky, O(n^3) per tract with n=1000-4000).

**Output paths:** `data/synthetic/run_{timestamp}/` with addresses.csv, metadata.json, figures/{scatter_signal_vs_truth.png, spatial_heatmap.png, morans_i_validation.txt}

**Cache invalidation notes:** none -- no changes to existing feature extraction or OSRM cache keys.

## 2026-05-09: M0 parity run (n20 SVI, GraphSAGE vs Dasymetric vs Pycnophylactic)

**Files changed:**
- `granite/scripts/run_m0_parity.py` (new, ~380 lines)
- `data/results/m0_n20_svi_parity/per_tract.csv` (60 rows: 20 tracts x 3 methods)
- `data/results/m0_n20_svi_parity/aggregate.csv` (pooled and per-tract bootstrap CIs)
- `data/results/m0_n20_svi_parity/pairwise_diffs.csv` (3 method pairs)
- `data/results/m0_n20_svi_parity/RESULTS.md` (decision summary)
- `Research_Status.md` (created; M0 entry appended)
- `data/results/m0_discovery_report.md` (Phase 1 discovery, pre-existing from same session)

**What changed and why:**
- M0 driver: loops n20 stratified tracts, runs `GRANITEPipeline._process_single_tract()` in
  single-tract SVI mode with GraphSAGE (arch=sage, 100 epochs, seed=42). Dasymetric and
  Pycnophylactic predictions extracted from pipeline's existing `_run_disaggregation_baselines`
  (called automatically for target=svi). BG validation uses `BlockGroupValidator` with nationally-
  ranked ACS SVI from `data/processed/national_bg_svi.csv`. Bootstrap CIs (1000 resamples) on
  pooled BG r by resampling BGs; pairwise difference CIs on per-tract bg_r paired by fips.

**Key results:**
- GRANITE pooled BG r = 0.769 (CI: 0.660-0.853, n_bgs=69)
- Dasymetric pooled BG r = 0.802 (CI: 0.712-0.871, n_bgs=69)
- Pycnophylactic pooled BG r = 0.768 (CI: 0.652-0.858, n_bgs=69)
- All three methods not statistically separable (GRANITE vs Dasymetric and vs Pycnophylactic);
  parity holds; Narrative-A footnote survives.
- Dasymetric IS separable from Pycnophylactic at per-tract level (CI: 0.04-0.64).
- Constraint error = 0% for all methods (GRANITE post-correction enforces constraint exactly).
- Wall-clock: 13.2 min (warm OSRM cache, single-tract mode, 20 separate GNN training runs).

**Discrepancy note vs CLAUDE.md reference values:**
- CLAUDE.md reports r=0.469 (GRANITE) and r=0.558 (IDW/old naming). M0 reports r=0.769/0.802.
  This discrepancy is expected: M0 uses national SVI ranking (not county), single-tract mode
  (not multi-tract), and the n20 stratified subset (not a general validation set). Do not
  update CLAUDE.md reference values from M0 -- those reflect different validation setups
  (M8/M10 scope).

**Cache invalidation:** none -- cache keys unchanged; read-only access to OSRM granite_cache.

---

## 2026-05-03: M3 non-graph leakage baselines

**Files changed:**
- `granite/evaluation/recovery_baselines.py` (new, ~190 lines)
- `granite/scripts/run_m3_baselines.py` (new, ~280 lines)
- `output/m3_n20_baselines/` (generated outputs)

**What changed and why:**
- `recovery_baselines.py`: per-tract ridge (RidgeCV with LOO via hat matrix) and GBM (5-fold OOF) baselines. Uses same `_compute_accessibility_features` per-tract stacking as M2, same global target standardization, per-tract predictor z-score. Zero-variance columns dropped per tract. GBM skipped when n < 50.
- `run_m3_baselines.py`: driver sweeping 3 targets x 20 tracts. Reads n20 list from M2 output. Writes five summary files: `per_tract_metrics.csv` (60 rows), `baseline_summary_stats.csv` (3 rows), `lift_table.csv` (120 rows joining M2 pivot), `lift_summary.csv` (6 rows), `lift_brief.md`. Wall time 13.1 min (all cache hits; OSRM not re-queried).

**Key finding:** GBM r reaches near 1.0 on employment_walk_effective_access and log_appvalue for most tracts, confirming these targets are strongly feature-predictable without any graph or constraint. GRANITE does not clear the ridge or GBM baseline ceiling on median r for any (target, architecture) cell.

**Cache invalidation:** none. Uses same per-tract cache keys as M2.

---

## 2026-04-29: Recovery harness M1

**Files changed:**
- `granite/disaggregation/recovery_harness.py` (new, ~290 lines)
- `granite/scripts/run_granite.py` (+55 lines: `--recover-feature` flag, `run_recovery_workflow`)
- `config.yaml` (+8 lines: `recovery:` section)
- `docs/recovery_harness_schema.md` (new)

**What changed and why:**
- `recovery_harness.py`: implements roadmap M1 held-out feature recovery. Reuses `GRANITEPipeline` data loading and feature methods. Drops `target_feature` from the feature matrix, computes per-tract means of that feature as soft constraints (replacing SVI), trains `MultiTractGNNTrainer` with `bg_constraint_weight=0.0` and `ordering_weight=0.0`, writes `predictions.csv`, `per_tract_metrics.csv`, and `run_meta.json`.
- `run_granite.py`: `--recover-feature FEATURE_NAME` flag added as a standalone argument (not in the mutually exclusive group); requires `--fips`, incompatible with `--global-training`. Default output path is `./output/recovery/<feature>_<arch>_<fips>_<seed>/`.
- `config.yaml`: `recovery:` section with `target_feature: null`, `standardize_target: true`, `output_dir: ./output/recovery`.
- `docs/recovery_harness_schema.md`: documents column meanings, units, standardization procedure, and constraint-replacement logic.

**Cache invalidation:** none. Feature computation and cache keys are unchanged; the harness post-processes the feature matrix after `_compute_accessibility_features` returns.

---

## 2026-05-03: M2 sweep (n20 × 3 targets × 2 architectures)

**Files changed:**
- `granite/scripts/run_m2_sweep.py` (new, ~760 lines)

**What changed and why:**
- `run_m2_sweep.py`: M2 sweep driver. Loads 20-tract stratified set from `tract_inventory.csv`, runs recovery harness for 3 targets (`log_appvalue`, `employment_walk_effective_access`, `nlcd_impervious_pct`) × 2 architectures (`sage`, `gcn_gat`), writes per-run outputs, and aggregates into pivot tables and a decision brief.
- Per-tract feature computation: instead of concatenating all 20 tracts' addresses before calling `_compute_accessibility_features` (which would generate a never-cached combined hash), the driver calls it once per tract and vstacks the arrays. This reuses existing single-tract cache entries and writes new ones per-tract, avoiding multi-hour cold OSRM routing and enabling disconnect resilience between tracts.
- `gnn.py`: `_compute_overall_constraint_error` and `_compute_per_tract_errors` already had the `|target| < 1e-6` absolute-error guard from a prior session.

**Key results (seed 42, 100 epochs, z-scored targets, n=20 tracts):**
- Wall-clock: 15.3 min (all 20 per-tract caches warm after first run)
- Best cell: `employment_walk_effective_access / sage` (median r = 0.509, crosses Branch 1 threshold)
- `log_appvalue`: median r = 0.039 (sage) / 0.103 (gcn_gat) — low recovery
- `nlcd_impervious_pct`: median r = 0.144 (sage) / 0.253 (gcn_gat)
- Architecture: gcn_gat averaged slightly higher median r (0.279 vs 0.231 for sage)

**Outputs:** `output/m2_n20_recovery/{target}_{arch}/` (6 dirs), `output/m2_n20_recovery/summary/` (pivot tables, decision brief)

**Cache invalidation:** none. Per-tract cache entries are written using the same single-tract hash keys the regular pipeline would use.

---

## 2026-04-27: Disk cleanup

**Files deleted:**
- `.venv/` (29 MB) — Python virtual environment removed to free disk space. Recreate with `pip install -e .`.
- `output/coord_artifact_test/`, `output/mehdi_review/`, `output/stage4_synthetic_eval/`, `output/architecture_comparison/`, `output/coord_artifact/`, `output/feature_importance/` (~133 MB total) — old experiment output directories, results already reviewed. `output/rank_consistency_run/` retained (active experiment).

**Cache invalidation:** none. Pipeline cache (`granite_cache/`) untouched.

---

## 2026-04-23: Coordinate-artifact experiment infrastructure

**Files changed:** `granite/disaggregation/pipeline.py`, `scripts/coord_artifact_experiment.py` (new), `scripts/coord_artifact_summary.py` (new)

**What changed and why:**
- `pipeline.py _apply_feature_mode`: new method substitutes the normalized feature matrix before graph construction. Supports four modes: `full` (pass-through), `coordinates_only` (z-scored lat/lon in cols 0-1, zeros elsewhere), `random_noise` (i.i.d. N(0,1) with fixed seed), `coords_plus_noise` (z-scored lat/lon + N(0,1) for remaining columns). Method is called after `normalize_accessibility_features` and reads `feature_mode` from config (default `'full'`, preserving baseline behavior).
- `pipeline.py _process_single_tract`: added call to `_apply_feature_mode` after normalization. Output shape (N, d) is unchanged so encoder architecture is unaffected.
- `scripts/coord_artifact_experiment.py`: runs the 4-condition x 5-tract design (Mehdi review tracts, 200 epochs, SAGE, fixed seed=42) and writes per-tract predictions and cross-condition metrics to `output/coord_artifact_test/`.
- `scripts/coord_artifact_summary.py`: reads experiment outputs and produces a 4-panel dashboard (spatial std, Moran's I, prediction-prediction correlation heatmap, constraint error by mode).

**Purpose:** tests whether GNN prediction quality is driven by coordinate information alone. If `coordinates_only` matches `full`, the 73-feature matrix adds no information beyond lat/lon; if `random_noise` matches, the model is fitting constraint correction only.

**Cache invalidation:** none. `feature_mode='full'` is the default and produces identical normalized features. Cache keys are unchanged.

## 2026-04-19: Stage 4 terminology fix and GIN integration audit

**Files changed:** `scripts/stage4_property_value_proxy_eval.py` (new), `graveyard/stage4_synthetic_eval.py.old` (retired), `gin_integration_audit.md` (new)

**What changed and why:**
- `scripts/stage4_synthetic_eval.py` renamed to `scripts/stage4_property_value_proxy_eval.py`. "Synthetic" was inaccurate: the evaluation target is log-transformed, min-max-normalized APPVALUE from the Hamilton County Assessor, not a generated signal. No features enter a generative function and no noise is injected.
- Module docstring updated to define the proxy transformation, document leakage audit (8 parcel-derived features excluded from feature set), rationale for proxy choice (escapes circularity), and add dasymetric citations (Mennis 2003; Maantay & Maroko 2009).
- OUTPUT_DIR updated from `./output/stage4_synthetic_eval` to `./output/stage4_property_value_proxy_eval`.
- Old file moved to `graveyard/` per convention.
- `gin_integration_audit.md` written at repo root: documents current GCN-GAT and GraphSAGE class structures, aggregation primitives, dispatch pattern (4 call sites in pipeline.py), and provides a diff-style plan for adding GIN (arch='gin') and standalone GCN (arch='gcn') as additional options. GINConv confirmed available. Estimated scope: ~245 lines across 2 files.

**Cache invalidation:** none. Rename does not affect pipeline caching logic.

## 2026-04-19: Fix post-training constraint correction with iterative bounded projection

**Files changed:** `granite/disaggregation/pipeline.py`, `config.yaml`

**What changed and why:**
- `config.yaml`: enabled `apply_post_correction: true` (was `false`). Without this, constraint errors were raw GNN training residuals with no post-hoc enforcement.
- `pipeline.py _finalize_predictions`: replaced single-pass additive shift + clip with iterative bounded projection (max_iter=50, tol=1e-8). Single-pass fails when clipping at 0 or 1 shifts the mean away from target; iterative loop re-applies the residual error until convergence.
- `pipeline.py` multi-tract holdout path (line ~4741): same fix applied to inline per-tract correction.
- `pipeline.py _apply_strong_constraint_correction`: same fix applied (was dead code but fixed for consistency).

**Verification (5 Mehdi tracts, 50 epochs each, cached):**

| FIPS        | SVI   | Iters | Residual  | Error  | Std    |
|-------------|-------|-------|-----------|--------|--------|
| 47065000700 | 0.114 | 2     | 7.45e-09  | 0.00%  | 0.0737 |
| 47065000600 | 0.224 | 3     | 0.00e+00  | 0.00%  | 0.0724 |
| 47065011326 | 0.510 | 1     | 0.00e+00  | 0.00%  | 0.1000 |
| 47065011321 | 0.696 | 2     | 0.00e+00  | 0.00%  | 0.0844 |
| 47065002400 | 0.891 | 4     | 0.00e+00  | 0.00%  | 0.1357 |

All constraint errors < 1e-6. Boundary tracts (0.114, 0.891) required 2-4 iterations due to clip-induced mean drift. Baseline methods (Dasymetric, Pycnophylactic) unaffected at machine epsilon.

**Cache invalidation:** none. Post-training correction only; no feature, routing, or cache key changes.

## 2026-04-19: Update visualization code for Dasymetric/Pycnophylactic baselines

**Files changed:** `granite/visualization/plots.py`, `granite/evaluation/post_training_validation.py`, `granite/scripts/run_granite.py`, `output/mehdi_review/_figures/README.md`

**What changed and why:**
- `plots.py`: replaced all IDW/Kriging/Naive_Uniform method references with Dasymetric/Pycnophylactic across dashboard panels (tradeoff scatter, variation bars, prediction distributions, prediction range, GNN-vs-baseline scatter, summary statistics, metrics table). Updated color scheme: dasymetric=#E65100 (warm orange), pycnophylactic=#1565C0 (cool blue). Removed overlap-handling code from tradeoff scatter (three distinct methods separate naturally).
- `post_training_validation.py`: updated color dicts, bootstrap difference test (now GRANITE vs Dasymetric), report text, summary checklist, docstrings, and help text to reference new baselines.
- `run_granite.py`: updated `--skip-baselines` help text.
- `README.md` in `_figures/`: updated dashboard description to reflect three-method display.

**Smoke test (tract 47065000600, 2089 addresses):**
- Dashboard: all three methods (GNN, Dasymetric, Pycnophylactic) appear in every panel.
- Metrics table: exactly three rows, no floor references.
- Tradeoff scatter: three distinct non-overlapping points.
- No KeyError, no missing-method warnings, no dangling IDW/Kriging strings in output artifacts.

**Cache invalidation:** none. No feature, routing, or pipeline logic changed.

## 2026-04-18: Retire IDW/Kriging baselines, add Dasymetric and Pycnophylactic

**Files changed:** `granite/evaluation/baselines.py`, `granite/disaggregation/pipeline.py`, `granite/evaluation/post_training_validation.py`, `graveyard/disaggregation_baselines_idw_kriging.py`

**What changed and why:**
- IDW and Kriging baselines retired to `graveyard/disaggregation_baselines_idw_kriging.py`. Point-interpolation methods collapse to tract mean under single-centroid interpolation; replaced by mass-preserving disaggregation baselines.
- `baselines.py`: added `DasymetricDisaggregation` (additive dasymetric using `nlcd_impervious_pct` ancillary) and `PycnophylacticDisaggregation` (Tobler 1979 iterative smoothing seeded from multi-tract gradient). Both satisfy `disaggregate()` interface. `DisaggregationComparison.run_comparison()` now passes `address_gdf` through to `disaggregate()` for ancillary column access. Base class `disaggregate()` signature extended with optional `address_gdf` parameter.
- `pipeline.py`: all IDW/Kriging imports, constructor calls, result key references (`IDW_p2.0`, `Kriging`), visualization labels, and block group validation collection updated to use `Dasymetric` and `Pycnophylactic`.
- `post_training_validation.py`: IDW/Kriging usage replaced with new baselines.

**Sanity check (tract 47065000600, 2089 addresses):**
- Dasymetric: constraint error 2.78e-17, spatial std 0.109, range [0.0, 0.47]
- Pycnophylactic: constraint error 2.78e-17, spatial std 0.065, range [0.10, 0.36]
- Both strictly in [0,1], all validation criteria pass.

**Cache invalidation:** none. No feature or routing logic changed.

## 2026-04-18: Add property_value as alternate disaggregation target

**Files changed:** `granite/data/loaders.py`, `granite/disaggregation/pipeline.py`, `granite/scripts/run_granite.py`, `config.yaml`

**What changed and why:**
- `loaders.py`: added `load_property_value_data()`, `get_tract_target_value()`, `get_address_truth_values()` to DataLoader. Property values loaded from `combined_address_features.csv`, normalized to [0,1] via county-wide min-max of log_appvalue. `hash` column now retained through address loading for property value joins.
- `pipeline.py _process_single_tract`: reads `config.data.target` (default 'svi') to select target. Uses `get_tract_target_value()` instead of reading RPL_THEMES directly. Logs active feature count. Attaches address-level truth vector to results. Skips IDW/Kriging baselines for non-SVI targets. Skips block group SVI constraints and pairwise ordering for property_value target.
- `pipeline.py _extract_building_features`: excludes `log_appvalue` and `build_to_land_ratio` when target=property_value to prevent target leakage (class attribute `_PROPERTY_VALUE_EXCLUDED_FEATURES`).
- `pipeline.py save_results`: writes `address_truth.csv` (predictions + truth) when truth vector is present; includes `target` field in `results_summary.json`.
- `run_granite.py`: added `--target` CLI argument (choices: svi, property_value).
- `config.yaml`: added `target: "svi"` under `data:` section.

**Verified:**
- `--target=svi` reproduces baseline behavior: 73 features, same constraint error and spatial std.
- `--target=property_value` runs to completion: 71 features (2 excluded), constraint satisfied, truth vector written to disk.

**Cache invalidation:** none. Accessibility caches store base+modal features which are identical across targets. Building features (where exclusion occurs) are recomputed fresh every run.

---

## 2026-04-17: Dashboard redesign and tract_fips bugfix

**Files changed:** `granite/visualization/plots.py`, `granite/disaggregation/pipeline.py`

**What changed and why:**
- `_plot_constraint_satisfaction` replaced with `_plot_tradeoff_scatter`: X-axis constraint error %, Y-axis spatial std. Shows each method as a labeled point -- GNN trades constraint precision for within-tract differentiation.
- `_plot_variation_comparison`, `_plot_prediction_distributions`, `_plot_prediction_range`: filtered to GNN and IDW_p2.0 only. Naive_Uniform, IDW_p3.0, Kriging are zero-valued in these panels and add no information. They still appear in the metrics table as floor references.
- `plot_spatial_analysis`: "Tract ?" fallback replaced -- title now omits tract reference when FIPS unavailable instead of showing a question mark.
- `pipeline.py _process_single_tract`: added `tract_fips` and `tract_svi` as top-level keys in return dict.
- `pipeline.py _create_research_visualizations` call site: added `tract_fips` to viz_data dict.
- `pipeline.py save_results` viz_data: added `tract_fips` via `results.get('tract_fips')`.
- Regenerated all figures for 5 tracts with SAGE architecture; collated into `output/mehdi_review/_figures/`.

**Cache invalidation:** none. Visualization-only and caller-side changes.

---

## 2026-04-16: Remove accessibility-centric framing from visualizations

**Files changed:** `granite/visualization/plots.py`, `granite/evaluation/spatial_diagnostics.py`

**What changed and why:**
- `plot_spatial_analysis`: replaced learned accessibility map and access-vulnerability scatter with within-tract deviation map (coolwarm) and prediction distribution histogram. title now shows tract FIPS, SVI, and address count.
- `_plot_accessibility_correlations` renamed to `_plot_prediction_range`: horizontal bar chart of (max - min) per method replaces accessibility correlation bars.
- `_plot_metrics_table`: "Access r" column replaced with "Moran's I" (displays n/a when not available in comparison_results).
- `_plot_spatial_patterns` (spatial_diagnostics.py): axes[1,0] replaced accessibility scatter with deviation-from-tract-mean geographic scatter.
- `_plot_accessibility_relationships` (spatial_diagnostics.py): short-circuited with deprecation warning; body retained for backward compatibility.
- `plot_accessibility_learning_validation`: marked deprecated in docstring; method body unchanged.
- regenerated all figures for 5 tracts (47065000700, 47065000600, 47065011326, 47065011321, 47065002400) with SAGE architecture; collated into `_figures/`.

**Cache invalidation:** none. visualization-only changes.

## 2026-04-14: National block group SVI data acquisition and validation

**Files changed:** `granite/data/block_group_loader.py`, `CLAUDE.md`

### Census API URL fix

`fetch_national_acs_data()` had `&in=state:{fips}&in=county:*` (two separate `in` params), which the Census API rejects for block group queries. Fixed to `&in=state:{fips}%20county:*` (space-separated single param), matching the working county-level fetch format.

### National data acquired

- Fetched ACS block group data for all 52 states/territories (242,335 block groups)
- Computed nationally-ranked SVI (239,346 with complete indicators)
- Cached to `data/processed/national_bg_acs_raw.csv` (63MB) and `data/processed/national_bg_svi.csv` (116MB)

### National ranking consistency results for tract 47065000600

- BG weighted mean SVI: 0.3046 (vs tract CDC SVI: 0.2235)
- Rescaling shift: -0.0811 (vs -0.1806 with county ranking, 55% reduction)
- BG1 nationally-ranked SVI: 0.1314 (above 0.05 floor, clipping problem resolved)
- All three BGs shifted downward under national ranking (county ranking inflated values because Hamilton County is low-vulnerability nationally)

### No cache invalidation

Training-only and data acquisition changes. No feature or routing modifications. Cache keys unchanged.

---

## 2026-04-13: Block group SVI rescaling for constraint consistency

**Files changed:** `granite/data/block_group_loader.py`, `granite/disaggregation/pipeline.py`, `scripts/bg_rescaled_convergence_experiment.py` (new), `scripts/diagnose_bg_consistency.py` (new)

### Diagnosis

Block group SVIs (ACS-derived, percentile-ranked within Hamilton County BGs) do not aggregate to the CDC tract SVI (national percentile ranks). For tract 47065000600: BG weighted mean = 0.4041, tract SVI = 0.2235, discrepancy = 80.8%. This explains why the prior convergence experiment saw tract constraint error blow up to 25-28% when BG constraints were added -- the optimizer had no feasible solution.

### Rescaling function

Added `rescale_block_group_svis()` to `block_group_loader.py`. Pure function: additive shift + clip to [0,1], iterates up to 10 times to handle clipping distortion. Converges within 0.001 of target weighted mean. For this tract: shift = -0.1806, BG1 clips to 0.0, BG2 = 0.290, BG3 = 0.370.

### Pipeline integration

Rescaling wired into `_train_multi_tract_gnn` in `pipeline.py`, gated by `rescale_bg_svi` config key (default True). Rescales per-tract: groups BGs by parent tract and rescales each group against its tract SVI.

### Rescaled convergence experiment results

Same 2x2 design: {GCN, SAGE} x {tract-only, tract+rescaled BG}. Key results:
- Tract error with BG constraints: 5.01% (GCN), 1.87% (SAGE) -- down from ~25-28% unrescaled
- BG error: 6.04% (GCN), 3.89% (SAGE) -- down from ~41-43% unrescaled
- GCN vs SAGE r: 0.693 (rescaled BG) vs 0.420 (tract-only) -- convergence preserved
- Prior unrescaled BG r was 0.788, but at cost of 25-28% tract error; rescaled achieves 0.693 with 3.4% avg tract error

Interpretation: rescaling resolved constraint tension while preserving convergence benefit.

### No cache invalidation

Training-only changes. No feature or routing modifications. Cache keys unchanged.

---

## 2026-04-12: Wire block group constraints into training

**Files changed:** `granite/models/gnn.py`, `granite/disaggregation/pipeline.py`, `scripts/setup_data.sh` (new)

### Block group mean constraints as additional training loss

- `MultiTractGNNTrainer.__init__`: added `bg_constraint_weight` attribute (default 1.0, configurable via config)
- `_compute_multi_tract_losses`: new optional `block_group_targets`/`block_group_masks` params; computes per-BG MSE constraint loss averaged across all block groups, returned as `bg_constraint` key in losses dict
- `train()`: accepts optional BG params, converts to tensors, passes to loss function, tracks `bg_constraint_errors` in training_history, reports BG constraint error at epoch intervals
- `_train_multi_tract_gnn` in pipeline: loads BG data via `BlockGroupLoader`, spatial-joins addresses, creates masks/targets filtered to `svi_complete==True`, >= 5 addresses, and nesting within training tracts; prints summary at training start
- Post-training diagnostic: reports per-BG residual mean error after tract correction (no BG correction applied)
- Backward compatible: all BG params default to `None`; behavior identical to prior code when not provided
- No cache invalidation: changes are training-only, no feature or routing changes

## 2026-04-12: Trim visualization output to 3 retained plots

**Files changed:** `granite/visualization/plots.py`, `granite/disaggregation/pipeline.py`, `granite/evaluation/accessibility_validator.py`, `granite/evaluation/feature_importance.py`, `granite/scripts/run_granite.py`

### Diagnostic plots gated behind --diagnostics flag

- 7 diagnostic plots (stage1_validation, stage2_validation, statistical_summary, and 4 accessibility_validation plots) now only generate when `--diagnostics` is passed
- `create_comprehensive_research_analysis` takes `diagnostics=False` parameter; diagnostic plots go to `output/visualizations/` when enabled
- `validate_granite_accessibility_features` takes `diagnostics=False` parameter; plots go to `output/accessibility_validation/` when enabled
- Code for generating diagnostic plots is preserved, not deleted

### Feature importance label bug fixed

- Cumulative importance annotations now handle the case where cumulative never reaches 80%/90% thresholds (falls back to total feature count)
- Fixed pluralization: "1 feature explains" vs "N features explain"

### spatial_analysis.png consolidated to 3-panel layout

- Changed from 2x2 (with text stats panel) to 1x3: SVI predictions, learned accessibility, access-vulnerability scatter
- SVI predictions panel now uses fixed 0-1 colorscale (RdYlGn_r) for cross-tract comparability
- Removed "Moderate Equity Pattern" text box from scatter panel; r value shown in title only

### plot_spatial_disaggregation updated

- Added `tract_results` parameter (backward-compatible with existing `multi_tract_data`)
- Auto-scale point size: s=30 (<5000 addresses), s=10 (<20000), s=3 (>=20000)
- Fixed TIGER shapefile path (was looking in nonexistent subdirectory)
- Single-tract mode now also uses fixed 0-1 SVI colorscale

### Output directory cleanup

- Default run produces: spatial_analysis.png, disaggregation_comparison.png, feature_importance/, CSVs, and results_summary.json
- output/visualizations/ and output/accessibility_validation/ only created with --diagnostics

### No cache invalidation

- No feature or routing changes. Cache keys unchanged.

---

## 2026-04-11: Visualization consolidation and multi-tract heatmap

**Files changed:** `granite/visualization/plots.py`, `granite/visualization/disaggregation_plots.py` (removed), `granite/disaggregation/pipeline.py`, `granite/scripts/run_granite.py`, `granite/scripts/run_holdout_validation.py`

### Consolidation

- `DisaggregationVisualizer` was duplicated identically in `plots.py` and `disaggregation_plots.py`. Removed `disaggregation_plots.py`, moved to `graveyard/disaggregation_plots_v2.old`.
- Updated imports in `pipeline.py` and `run_holdout_validation.py` to point to `granite.visualization.plots`.

### Multi-tract heatmap

- Extended `plot_spatial_disaggregation` to support multi-tract mode via `multi_tract_data` dict parameter. Single-tract mode unchanged.
- Multi-tract mode: unified SVI colorscale (0-1, RdYlGn_r), deviation panel per address relative to its own tract, tract boundary overlay from TIGER shapefiles, auto-scaled point size for large address counts.
- Pipeline now returns `address_gdf` in single-tract results dict.
- `run_multi_fips_experiment` collects per-tract GDFs and predictions, generates multi-tract heatmap automatically when 2+ tracts succeed.

### No cache invalidation

- No feature or routing changes. Cache keys unchanged.

---

## 2026-04-11: Bug fixes, modal feature refactor, cache improvements

**Files changed:** `granite/data/loaders.py`, `granite/models/gnn.py`, `granite/cache.py`, `granite/disaggregation/pipeline.py`, `granite/evaluation/spatial_diagnostics.py`, `granite/evaluation/accessibility_validator.py`, `granite/evaluation/morans_i_analysis.py`, `granite/evaluation/post_training_validation.py`, `granite/models/mixture_of_experts.py`, `granite/validation/block_group_validation.py`, `granite/features/modal_accessibility.py`, `README.md`, `docs/FEATURES.md`

### Bug fixes (12 bugs)

- **#7 CRS mismatch in spatial join** (`loaders.py`): Added CRS check before `within()` in `get_addresses_for_tract`
- **#10 Hardcoded county name** (`loaders.py`): `_get_county_name` now falls back to SVI CSV lookup; `load_svi_data` uses case-insensitive match and raises `ValueError` on empty result
- **#4 GPU device mismatch** (`gnn.py`, 8 locations): All CPU tensors now use `.to(device)` matching graph data device
- **#5 No cache invalidation** (`cache.py`): Added version tag (auto-invalidates on change), `invalidate()` method with optional TTL, and `_invalidate_all()`
- **#6 Bootstrap p-value**: Already fixed in previous session (verified)
- **#8 Single-county assumption** (`pipeline.py`): Global training now derives county from most common tract FIPS, warns if mixed
- **#9 MoE overlap documentation** (`mixture_of_experts.py`): Added comment documenting deliberate soft-boundary design
- **#11 FIPS strip inconsistency** (`loaders.py`): Added `str().strip()` at tract_fips assignment
- **#12 Zero-variance logging** (`pipeline.py`): Log level now INFO when near expected count (24 constants), WARN only when unexpectedly high
- **#13 Quality score inversion** (`spatial_diagnostics.py`): Changed to `1 - (bad/total)` so higher = better
- **#14 Silent address drop** (`pipeline.py`): Logs unmatched address count after spatial join
- **#15 Low BG threshold** (`block_group_validation.py`): Raised minimum from 5 to 10 addresses per block group

### Validator bug fix

- **NameError in accessibility_validator.py**: `count_5` referenced but never defined at lines 540-541 and 561-562; replaced with `count_10min` which was the intended variable. Validator now runs all 7 steps.

### Modal features refactored to per-address computation

- **Before:** 5 features per destination type computed at tract level from vehicle ownership rates and base features. All addresses in a tract received identical values (zero within-tract variance).
- **After:** Computed per address from OSRM driving and walking travel times. New semantics: avg_time (mean of drive/walk nearest), time_std (mode disparity), access_density (union of destinations reachable in 10min by either mode), equity_gap (|walk - drive| nearest), car_advantage (walk/drive ratio nearest).
- Pipeline extracts per-address travel time summaries via `_summarize_travel_times()` during base feature computation and caches them alongside per-destination features.
- Fallback to tract-level approximation when per-address times unavailable (partial cache from older runs).
- Output shape unchanged: (n_addresses, 15). Feature names unchanged for downstream compatibility.
- `employment_walk_effective_access` now ranks #3 in feature importance (was negligible as tract constant).

### Cache key changes

- Complete feature cache key changed from `_base_modal` to `_base_modal_v2` to invalidate stale entries with old modal features.
- New `modal_times` cache entries store per-address travel time summaries per destination type.
- Cache now stores base+modal only (not socioeco/building); those are recomputed fresh on cache hit.

### Documentation updates

- README and FEATURES.md updated to reflect per-address modal features, reduced zero-variance count (11 vs 24), cache invalidation API, and corrected feature count.

---

## 2026-04-09: Baseline and evaluation bug fixes

**Files changed:** `granite/evaluation/baselines.py`, `granite/evaluation/bootstrap_confidence_intervals.py`, `granite/evaluation/post_training_validation.py`

### Fix #1: IDW baseline was not excluding target tract from neighbors
- `baselines.py` IDWDisaggregation.disaggregate(): `target_idx` was computed but never used to filter KD-tree query results
- The target tract's own centroid was included as a neighbor, biasing IDW predictions toward the known tract mean
- Fix: filter `target_idx` out of each address's neighbor set after query, trim to `n_neighbors`
- Impact: IDW results will now reflect true interpolation from neighboring tracts only
- No cache invalidation needed (baselines don't use cache)

### Fix #6: Bootstrap p-value was one-tailed but labeled as p-value (implying two-tailed)
- `bootstrap_confidence_intervals.py` bootstrap_correlation_difference(): computed both two-tailed and one-tailed p-values, but returned the one-tailed value under the key 'p_value'
- `post_training_validation.py` line 407: same issue, used `np.mean(diff_dist <= 0)` which is one-tailed
- Fix: replaced both with proper two-tailed bootstrap pivotal test: shift bootstrap distribution to null (diff=0), then `np.mean(np.abs(boot_diffs_null) >= np.abs(diff_observed))`
- CI-based significance test was already correct; only the reported p-value number was wrong
- No cache invalidation needed

### Fix #3: IDW used raw lat/lon distances while Kriging used meters
- `baselines.py` IDWDisaggregation: KD-tree was built on raw lon/lat coordinates, meaning E-W distances were ~17% distorted at lat ~35 (Chattanooga)
- Kriging already converted to approximate meters with proper per-axis scaling
- Fix: IDW now converts centroids to meters in fit() and address coords in disaggregate(), using same scaling factors as Kriging (111320*cos(lat) for lon, 110540 for lat)
- Impact: IDW and Kriging now use consistent isotropic distance metrics
- No cache invalidation needed

### Fix #2: IDW and Kriging double-clipping broke aggregate constraint
- `baselines.py` both IDWDisaggregation and OrdinaryKrigingDisaggregation: scale-then-clip-then-shift-then-clip could leave the final mean != tract_svi
- Added shared `_enforce_constraint()` function that iteratively shifts predictions to target mean, clips to [0,1], and repeats until convergence (max 20 iterations, tol 1e-6)
- Both baselines now use this shared function
- Impact: baseline constraint satisfaction will now match the GNN's hard constraint, making comparison fair
- No cache invalidation needed

## 2026-04-16: Dr Mehdi review run -- five tracts spanning SVI spectrum

**Files changed:** none (pipeline run only)

### What was done

Ran `granite --architecture sage` sequentially on five Hamilton County tracts for Dr Mehdi's review. No code changes were required; all five tracts ran cleanly.

Outputs saved to `output/mehdi_review/<FIPS>/` with figures collated and renamed in `output/mehdi_review/_figures/`.

### Tracts and results

| FIPS        | SVI   | n_addresses | constraint_err% | spatial_std | morans_i |
|-------------|-------|-------------|-----------------|-------------|----------|
| 47065000700 | 0.114 | 1,784       | 8.75            | 0.0390      | 0.7774   |
| 47065000600 | 0.224 | 2,089       | 13.16           | 0.0771      | 0.9415   |
| 47065011326 | 0.510 | 2,738       | 0.80            | 0.1024      | 0.9858   |
| 47065011321 | 0.696 | 3,756       | 1.97            | 0.0997      | 0.9333   |
| 47065002400 | 0.891 | 1,918       | 2.68            | 0.0769      | 0.9392   |

### Cache notes

- 47065000600 was a full cache hit (62s); all others required OSRM modal feature recomputation (~1-2 hrs each).
- Feature count varies 71-73 across tracts due to zero-variance building features; existing behavior, no impact on comparability.
- No cache keys were invalidated.

## 2026-04-25 - Full rank-consistency experiment (8 tracts × 2 architectures)

**Files changed:**
- `scripts/run_rank_consistency_experiment.py` (new) - driver script running GRANITEPipeline for both GraphSAGE (`arch=sage`) and GCN-GAT (`arch=gcn_gat`) across all 8 inventory tracts; outputs to `output/rank_consistency_run/{graphsage,gcn_gat}/tract_{fips}/`

**Outputs generated:**
- `output/rank_consistency_run/graphsage/tract_*/` - 8 tracts, 73 named feature cols, raw_prediction present
- `output/rank_consistency_run/gcn_gat/tract_*/` - 8 tracts, 73 named feature cols, raw_prediction present
- `results/rank_consistency_full/summary.txt`, `per_tract_rho.csv`, `feature_summary.csv`

**Experiment parameters:** seed=42, epochs=200, apply_post_correction=True, 8 tracts spanning SVI 0.04-0.89, cv_threshold=0.10, min_tracts=5, min_addresses=50

**Key result:**
- Section A (SAGE only):    0
- Section B (GCN-GAT only): 2 features (healthcare_modal_access_gap, healthcare_car_effective_access; median_rho=0.18, n_tracts=6)
- Section C (both, same sign): 0
- Section D (sign-flippers): 0

**Interpretation:** No feature is rank-consistent under both architectures. Two healthcare modal features survive under GCN-GAT alone with a small positive correlation (rho~0.18), but no features are architecture-agnostic. The zero Section C/D counts mean there is no cross-architecture signal to report as a positive finding.

**Cache notes:** Runs used existing OSRM cache; no cache invalidation.

## 2026-05-04 — M3.6 Framework Patch for External Targets

**Files created:**
- `granite/data/external_targets.py` (142 lines): `load_external_target()` — reads address-aligned CSV (plain or gzipped), returns numpy array + metadata dict, raises ValueError on zero matches, warns to stderr if matched fraction < 0.80
- `granite/evaluation/redundancy_filter.py` (229 lines): `run_redundancy_filter()` — per-tract ridge+GBM reconstruction test, `REDUNDANCY_THRESHOLD=0.5`, `RedundancyFilterResult` dataclass with `is_admissible`/`is_redundant` semantics, writes `redundancy_filter.json` + `redundancy_filter_per_tract.csv`
- `granite/evaluation/README.md` (56 lines): documents gate semantics, threshold, n5 default, rationale

**Files modified:**
- `granite/disaggregation/recovery_harness.py` (+82 lines): added `external_target_path=None` to `run_recovery()`; mutual exclusivity validation; external path loads target via `load_external_target`, uses full feature matrix (no drop); `_write_outputs` extended with `target_mode`, `target_name`, `target_source`, `n_addresses_matched`, `n_addresses_missing`
- `granite/evaluation/recovery_baselines.py` (+48 lines): added `external_target_vector=None` to `run_baselines()`; external path uses full feature matrix; NaN exclusion per-tract for unmatched addresses
- `granite/scripts/run_m2_sweep.py` (+90 lines): added `--external-targets JSON_PATH` flag; `_run_external_sweep()` function runs all ARCHITECTURES for each external target entry
- `granite/scripts/run_granite.py` (+130 lines): added `--recover-external PATH`, `--filter-only` flags; `run_external_recovery_workflow()` runs filter gate then GRANITE training if admissible

**Cache invalidation:** none — new code paths do not touch existing feature cache keys

**Acceptance checks:**
- M1 regression (log_appvalue, sage, seed 42, 50 epochs, single tract): r=0.267, RMSE=0.921, constraint_error=1.75% — within 0.001 of reference
- Synthetic noise filter smoke: median_ridge_r=0.065, median_gbm_r=0.025, is_admissible=True, is_redundant=False, exit 0
- M3 column schema: byte-identical to reference; values differ only due to single-tract vs 20-tract global standardization (expected)
- `run_meta.json` records target_mode, target_name, target_source, n_addresses_matched, n_addresses_missing on both paths

## 2026-05-21: ablation 01_per_tract_std (Step 2a)

**What changed:** Added `feature_standardization: {global, per_tract}` config toggle. `global` (default) preserves existing RobustScaler behavior; `per_tract` applies per-column z-score (mean/std) within each tract group, clamping near-zero std (<1e-8) to 1.0.

**Files changed:**
- `config.yaml`: added `feature_standardization: "global"` under `features:`
- `granite/models/gnn.py`: extended `normalize_accessibility_features()` with `method='per_tract'` branch and `tract_labels` parameter; fail-fast on missing tract assignments
- `granite/disaggregation/pipeline.py`: reads `features.feature_standardization` config key, passes `tract_labels` when per_tract mode active; stores scaler as `_stored_feature_scaler`

**Files created:**
- `experiments/ablation/01_per_tract_std/run_ablation_01.py` (driver: same 20 tracts, seed 42, both architectures, per_tract std enabled)
- `experiments/ablation/01_per_tract_std/git_state.txt`, `config_snapshot.yaml`, `environment.txt`, `tract_selection.txt` (pre-flight)
- `experiments/ablation/01_per_tract_std/results/per_tract_metrics.csv` (40 rows)
- `experiments/ablation/01_per_tract_std/results/aggregate_metrics.json`
- `experiments/ablation/01_per_tract_std/results/block_group_validation.json`
- `experiments/ablation/01_per_tract_std/results/delta_vs_baseline.json`
- `experiments/ablation/01_per_tract_std/results/per_tract_scalers.npz` (80 entries: 40 tract-arch pairs x mu+sigma)
- `experiments/ablation/01_per_tract_std/results/zero_var_columns.csv` (530 zero-variance feature-tract pairs clamped)
- `experiments/ablation/01_per_tract_std/results/feature_importance/sage_importance.csv`
- `experiments/ablation/01_per_tract_std/results/feature_importance/gcngat_importance.csv`
- 8 figures in `experiments/ablation/01_per_tract_std/figures/` (6 standard + 2 comparison)

**Cache invalidation:** None. Per-tract standardization is applied after feature computation; OSRM cache keys are unaffected.

**Headline deltas vs 00_baseline:**
- SAGE spatial std: +0.0026 (0.0797 -> 0.0823); slope vs SVI flattened from -0.01772 to -0.01438
- GCN-GAT spatial std: -0.0073 (0.0887 -> 0.0814); slope flattened from -0.00798 to -0.00644
- SAGE BG r: -0.016 (0.769 -> 0.754); GCN-GAT BG r: +0.017 (0.749 -> 0.766); all within 0.05 flag threshold
- Moran's I: SAGE +0.045, GCN-GAT +0.001
- Feature importance Spearman rho SAGE vs GCN-GAT: 0.099 -> 0.116; top-10 overlap: 2 -> 3
- 530 zero-variance (tract, feature) pairs clamped to std=1.0 (small tracts with constant columns)
- Constraint errors: ~2e-8 (post-correction saturates to zero as expected)

---

## 2026-05-29: Step 4 pre-summary fixups

**Files changed:** `experiments/ablation/04_constraint_by_construction/run_ablation_04.py`, `experiments/ablation/run_ablation_03.py`, `experiments/ablation/01_per_tract_std/CONFIG_SNAPSHOT_NOTE.md`, all three step 4 variant result sets re-run

**What changed and why:**

Two artifact-quality issues surfaced in `experiments/ablation/04_constraint_by_construction/ANOMALY.md` were resolved before summary generation.

**Issue 1: `pre_correction_constraint_error_mean` NaN across all three step 4 variants.**
Root cause: `run_ablation_04.py` line 337 used `training_result.get('final_predictions', None)` to retrieve pre-shift predictions, but `_train_accessibility_svi_gnn` returns a dict with key `raw_predictions`, not `final_predictions`. The lookup always returned `None`, producing NaN for every row.
Fix: changed key to `raw_predictions` and simplified the redundant if/else branches.
After fix, values match expected pattern:
- `soft`: pre_correction = 0.0254 SAGE, 0.0195 GCN-GAT (small nonzero; constraint loss pulls but shift is what makes mean exact)
- `cbc_with_shift`: pre_correction = 4.05e-08 / 3.5e-08 (machine precision; by-construction enforcement is exact)
- `cbc_no_shift`: pre_correction = 4.05e-08 / 3.5e-08 (same; no shift needed)

**Issue 2: config snapshot captured before runtime mutations in `run_ablation_03.py`.**
Root cause: `_write_preflight(vdir)` was called before `cfg['features']['feature_standardization'] = 'per_tract'` was applied, so step 3 snapshots recorded `global` instead of `per_tract`. Same bug existed for `01_per_tract_std` (config.yaml on disk used as snapshot source).
Fix: updated `_write_preflight` in `run_ablation_03.py` to accept the cfg dict and write `yaml.dump(cfg, ...)`, and moved the cfg mutation block to before the `_write_preflight` call. `run_ablation_04.py` already had correct ordering.
Added `experiments/ablation/01_per_tract_std/CONFIG_SNAPSHOT_NOTE.md` documenting the 2a snapshot discrepancy with evidence (per_tract_scalers.npz, zero_var_columns.csv, README metadata, identical numerics with step 4 baseline).

**Sanity check:** `00_baseline_for_step4` pooled BG r = 0.7537 SAGE, 0.7664 GCN-GAT (identical to 2a; baseline reproduced).

**Cache invalidation:** None. All three variants were rerun from cache.

## 2026-06-05: step 5 complete -- graph contribution boundary test

**Files changed:**
- `granite/data/loaders.py`: added `graph_variant` branch in `create_spatial_accessibility_graph`.
  `production` runs the existing road-network path unchanged. `mlp_floor` returns self-loops only
  (edge_index = [[i],[i]] for all i, edge_weight = 1.0). Unknown variant raises ValueError.
- `config.yaml`: added `graph_variant: production` (default preserves current behavior).
- `experiments/ablation/05_graph_contribution/`: run_sweep.py, make_figures.py, README.md,
  post_run_handler.py, per-condition subdirs with results and figures.

**Sweep:** 2 conditions x 2 architectures x 5 seeds x 20 tracts = 400 tract evaluations.
constraint_mode=soft, variation_weight=0.8, seeds=[42, 17, 123, 2024, 7].

**Sanity regression (production, seed=42):** PASSED within 1e-3 (BG r) / 2e-3 (std).

**Results:**

| condition | arch | within_tract_std | pooled_bg_r |
|---|---|---|---|
| production | SAGE | 0.0899 | 0.7632 +/- 0.0070 |
| production | GCN-GAT | 0.0906 | 0.7639 +/- 0.0168 |
| mlp_floor | SAGE | 0.0832 | 0.7714 +/- 0.0115 |
| mlp_floor | GCN-GAT | 0.0812 | 0.7660 +/- 0.0099 |

**Cache invalidation:** none. `graph_variant` affects only message-passing graph topology;
node features and OSRM routing cache keys are unchanged.

**Artifacts:** `experiments/ablation/05_graph_contribution/`
**Git sha:** 87ca99cba1702be36eb01abcebd44af87adab609

---

## 2026-06-09 — Cut ungrounded address-level coordinate/accessibility figures

**Files changed:** `Research_Status.md`, `docs/FEATURES.md`, `CLAUDE.md`

**What changed and why:** Removed all four assertion sites for the ungrounded figures r=0.671 (spatial coordinates at address level) and r=0.033 (accessibility features at address level). These figures traced only to `output/coord_artifact_test/` and `output/coord_artifact/` artifacts deleted 2026-04-27; no surviving committed artifact supports them at any file:line. The `experiments/audits/outstanding_items_reconciliation.md` audit (already committed) documented this gap.

Specific excisions:
- `Research_Status.md`: removed numbered item 2 ("Ecological fallacy at address scale") in full; renumbered former items 3-7 to 2-6.
- `docs/FEATURES.md`: removed the entire "### Empirical finding" section (heading + four sentences).
- `CLAUDE.md` Feature matrix section: removed the paragraph "Accessibility features (travel times...) do not treat it as a data quality problem."
- `CLAUDE.md` Key result reference points: removed two bullets ("Spatial coordinates alone: r ~ 0.67" and "Accessibility features alone: r ~ 0.03").

The audit file (`experiments/audits/outstanding_items_reconciliation.md`) and existing on-disk BG-validation outputs under `output/coord_artifact/` were left untouched. The audit file correctly records the figures as ungrounded and should remain as provenance.

Ecological-fallacy finding re-grounding deferred to a separate experiment. The `scripts/coord_artifact_experiment.py` and `scripts/coord_artifact_bg_validation.py` infrastructure exists and can produce a committed artifact when re-run.

**Cache invalidation:** none. No pipeline logic changed.

---

## 2026-06-10 -- Delinquency convergent validity (Door 2 acquisition)

**Files created:**
- `scripts/delinquency_convergent_validity.py`: per-tract partial Spearman analysis; pre-registered positive direction; Wilcoxon tests vs zero and paired; completeness and binary robustness checks
- `experiments/recovery/delinquency_convergent_validity/results.json`: full numeric results
- `experiments/recovery/delinquency_convergent_validity/per_tract_partial.csv`: partial rho per tract per method, 16 primary tracts
- `experiments/recovery/delinquency_convergent_validity/summary.txt`: plaintext report

**Data join:** n20 addresses (lat/lon) -> parcels.shp via gpd.sjoin(predicate='within'), CRS EPSG:6576. Parcel key MAP|GROUP_|PARCEL (strip+upper, blank group -> ''). Bill Year filter [2000, 2026]. Match rate: 39,476/39,535 (99.9%). Addresses with any delinquency: 1,866 (4.7%).

**Index alignment:** confirmed across granite_m0.parquet, dasymetric.parquet, pycnophylactic.parquet, and n20_feature_matrix.csv before any statistic computed.

**Pre-registered direction:** positive partial Spearman (higher svi_pred -> more delinquency-prone addresses after controlling log_appvalue). Committed in script docstring.

**Primary statistic:** partial Spearman r_xy.z = (r_xy - r_xz*r_yz) / sqrt((1-r_xz^2)(1-r_yz^2)), x=method_svi, y=n_delinq_years, z=log_appvalue. Per tract, 16 primary tracts only.

**Per-tract partial rho (n=16 primary tracts):**

| method | median | IQR | positive/negative tracts |
|---|---|---|---|
| granite | +0.003 | [-0.017, +0.018] | 10/6 |
| dasymetric | -0.012 | [-0.029, +0.006] | 5/11 |
| pycnophylactic | +0.019 | [-0.008, +0.039] | 11/5 |

**Wilcoxon signed-rank (one-sided, vs 0, n=16):**
- granite: W=75.0, p=0.3718
- dasymetric: W=41.0, p=0.9205
- pycnophylactic: W=99.0, p=0.0583 (marginal)

**Wilcoxon paired (GRANITE > baseline, n=16 pairs):**
- granite vs dasymetric: W=93.0, p=0.1057
- granite vs pycnophylactic: W=46.0, p=0.8739

3 vs-zero and 2 between-method tests; no correction applied; counts flagged.

**Completeness (building features vs n_delinq_years, n=16):**
- build_to_land_ratio: median=-0.101, IQR=[-0.124, -0.075]
- log_acres: median=-0.070, IQR=[-0.121, -0.049]
- log_bldg_footprint_m2: median=-0.063, IQR=[-0.117, -0.038]
Building features weakly negatively correlated with delinquency (larger/newer buildings less delinquency-prone); shared-driver concern not ruled out but weak.

**Binary robustness (n=16):** granite +0.003, dasymetric -0.012, pycnophylactic +0.020. Consistent with continuous results.

**Full n20 secondary (20 tracts):** granite +0.005, dasymetric -0.014, pycnophylactic +0.019. Sparse-tract deltas negligible.

**Verdicts:**
- granite: null/inconclusive (p=0.372, positive median, n=16 power limit)
- dasymetric: null (p=0.920, negative median)
- pycnophylactic: marginal (p=0.058, positive median)

**Interpretation:** The primary vs-zero test is null for GRANITE. Cannot separate "GNN allocates noise" from "proxy too attenuated by escrow ceiling." Dasymetric is negative, consistent with its ancillary-variable (impervious surface) allocation not tracking socioeconomic distress. Pycnophylactic's marginal positive is unexpected given its spatial smoothing; possibly an artifact of small n. A positive GRANITE result would have been conservative given attenuation; the null leaves the recovery question open.

**Cache invalidation notes:** none. No routing or feature extraction logic changed.

---

## 2026-06-10 -- Persist per-address predictions for n20, provenance-anchored

**Files changed:** `scripts/persist_per_address_predictions.py` (new), `experiments/recovery/per_address_predictions/` (new)

**What changed and why:** PATH B (no checkpoint found). Re-ran GRANITE m0/soft on warm cache for all 20 n20 tracts under the frozen config (GraphSAGE, seed=42, 150 epochs, apply_post_correction=True, neighbor_tracts=0). Captured per-address SVI predictions for GRANITE, Dasymetric, and Pycnophylactic from the same pipeline run so all three arrays share the same address ordering.

Provenance guard passed with all deltas well within tol=0.005:

| method | frozen | reproduced | delta |
|---|---|---|---|
| GRANITE | 0.7692 | 0.769165 | 0.000035 |
| Dasymetric | 0.8018 | 0.801786 | 0.000014 |
| Pycnophylactic | 0.7678 | 0.767751 | 0.000049 |

**Artifacts:**
- `experiments/recovery/per_address_predictions/granite_m0.parquet` (39535 rows)
- `experiments/recovery/per_address_predictions/dasymetric.parquet` (39535 rows)
- `experiments/recovery/per_address_predictions/pycnophylactic.parquet` (39535 rows)
- `experiments/recovery/per_address_predictions/provenance.json` (config hash 778295c2bdb6ed21, all three r values, index alignment note)

Index alignment: fips + address_idx (0-based within tract), row order matches n20_feature_matrix.csv exactly.

**Cache invalidation:** none. Pipeline logic unchanged; pyarrow added as a dependency.

2026-06-23: Added PROJECT_PREAMBLE.md as the durable grounded-truth block for strategic instance context; no code changes, no cache invalidation.
