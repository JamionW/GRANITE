# Commit Hash Rewrite Map

On 2026-09-27, commit `6280fa1` ("chore: untrack agentic/AI-context helper files from source control") ran `git-filter-repo` to purge `CLAUDE.md`, `SESSION_LOG.md`, `PROJECT_PREAMBLE.md`, and `Research_Status.md` from history entirely; filter-repo rewrote every earlier commit, so each such commit's hash changed. Documents written before that rewrite still cite the old (pre-rewrite) hashes, which no longer exist on `main`. This table maps each cited old hash to its rewritten equivalent on `main` (verified with `git merge-base --is-ancestor <new> main`). The table is generated from git history, not hand-maintained.

| old hash (pre-rewrite) | new hash (on `main`) | commit subject | files citing the old hash |
|---|---|---|---|
| `0d75890` | `b25fa3c` | Add ablation Steps 2a and 2b: per-tract standardization and norm-layer audit | `experiments/ablation/03_smoothness/00_off/README.md`<br>`experiments/ablation/03_smoothness/01_quarter/README.md`<br>`experiments/ablation/03_smoothness/02_default/README.md`<br>`experiments/ablation/03_smoothness/03_double/README.md`<br>`experiments/ablation/03_smoothness/04_five_x/README.md`<br>`experiments/ablation/03_smoothness/INSPECTION.md`<br>`experiments/ablation/03_smoothness/summary/README.md` |
| `1d1deff` | `3976b3f` | Step 4 fixups: correct pre_correction_constraint_error capture site, fix config snapshot timing (see experiments/ablation/04_constraint_by_construction/ANOMALY.md) | `experiments/ablation/04_constraint_by_construction/summary/README.md`<br>`experiments/ablation/04b_variation_weight_recalibration/RECOVERY.md` |
| `2a3e610` | `677f2fa` | Step 4b complete: variation_weight sweep under cbc_no_shift; Outcome C (see experiments/ablation/04b_variation_weight_recalibration/summary/README.md) | `experiments/ablation/04b_variation_weight_recalibration/RECOVERY.md` |
| `34199a1` | `73e8638` | Add Study 1 variance-components artifact; trace five disputed proposal figures | `docs/paper1/number_trace.md` |
| `35cebcf` | `df3d5a1` | Add ablation Step 3: cross-tract smoothness weight sweep and verification | `experiments/ablation/03_smoothness/INSPECTION_FINDING.md` |
| `3711b40` | `12302a4` | Add for_mehdi_review/ bundle with M0, M2, and M3 result artifacts | `docs/proposal/section5_verification_manifest.md`<br>`experiments/ablation/00_baseline/README.md` |
| `458cf42` | `8aecdc1` | Taxonomy finding supersedes ungrounded ecological-fallacy r-table | `docs/reviews/section4_5_figure_trace.md` |
| `4862792` | `4aeb279` | Fix three broken baseline figures; add regen script and assertion guards | `experiments/ablation/01_per_tract_std/CONFIG_SNAPSHOT_NOTE.md`<br>`experiments/ablation/01_per_tract_std/README.md`<br>`experiments/ablation/02b_i_no_input_layernorm/README.md`<br>`experiments/ablation/02b_ii_no_batchnorm/README.md`<br>`experiments/ablation/02b_iii_layernorm_in_hidden/README.md`<br>`experiments/ablation/02b_summary/README.md`<br>`experiments/ablation/04_constraint_by_construction/ANOMALY.md` |
| `49a052b` | `f84c7e8` | step05b: road_knn_k=9 calibrates road_network_uniform degree to spatial for honest topology parity | `docs/proposal/section5_verification_manifest.md`<br>`experiments/ablation/05b_topology_specificity/RESULTS.md` |
| `53d056d` | `41eab85` | Complete loss-term audit started after smoothness cleanup; see experiments/audits/loss_term_audit_summary.md | `experiments/ablation/04b_variation_weight_recalibration/RECOVERY.md` |
| `5859ed0` | `de772b7` | Track m0 parity results; anchor negations for trapped result CSVs | `docs/proposal/section5_verification_manifest.md`<br>`docs/reviews/section4_5_figure_trace.md` |
| `596b90e` | `eeff599` | Final try on this branch, time to scale down feature engineering | `experiments/audits/loss_term_audit.md` |
| `5c54c9d` | `04f453e` | 5b: corrected topology specificity metrics under fixed Moran's I estimator | `docs/reviews/section4_5_figure_trace.md` |
| `5d047cd` | `3e380c1` | Convergent-validity test: delinquency proxy, null at n=16 | `docs/reviews/section4_5_figure_trace.md` |
| `7844c93` | `c761d3d` | M6 recovery grid runner with supervised ceiling method | `docs/reviews/section4_5_figure_trace.md` |
| `7dc0617` | `c8ec3b7` | Cut ungrounded address-level coordinate/accessibility figures (r=0.671, r=0.033); traced only to deleted output, no surviving artifact | `DIRECTION.md`<br>`experiments/ecological_fallacy/Ecological_Fallacy_Finding.md` |
| `8644a43` | `1a30336` | step05b: remove graph_knn_k clobber, add production as fourth condition | `experiments/ablation/05b_topology_specificity/degree_calibration.md` |
| `87ca99c` | `3b648f8` | Step 4b recovery: document post-disconnect inspection in RECOVERY.md | `experiments/ablation/05_graph_contribution/README.md` |
| `b0db535` | `6af25c4` | Step 4 summary: constraint-by-construction sweep (see experiments/ablation/04_constraint_by_construction/summary/README.md) | `experiments/ablation/04b_variation_weight_recalibration/RECOVERY.md`<br>`experiments/ablation/04b_variation_weight_recalibration/summary/README.md` |
| `b6f6b60` | `8079334` | 5b: correct RESULTS.md to fixed estimator values, retire 0.446 and 0.083 | `docs/proposal/section5_verification_manifest.md` |
| `cdc860a` | `0351298` | M6 atomic CSV write, temp file plus os.replace | `data/results/variance_components/README.md` |
| `dc419ad` | `f1d6c13` | 5b: append Interpretation section to RESULTS.md | `docs/proposal/section5_verification_manifest.md`<br>`docs/reviews/section4_5_figure_trace.md` |
| `e988de4` | `812a02a` | Audit followup: remove orphaned variation_weight key, rename accessibility_consistency_loss to min_spread_loss (see experiments/audits/loss_term_audit.md entries 003, 008, 011) | `experiments/ablation/01_per_tract_std/CONFIG_SNAPSHOT_NOTE.md`<br>`experiments/ablation/04_constraint_by_construction/ANOMALY.md`<br>`experiments/ablation/04b_variation_weight_recalibration/RECOVERY.md` |
| `f16c01c` | `a8e64de` | step05b: freeze degree calibration artifact | `docs/paper1/number_trace.md`<br>`experiments/ablation/05b_topology_specificity/RESULTS.md` |
| `fc212d4` | `d01f5e4` | M6: corrected morans_i_output via fixed esda estimator, retire 5b presymmetrize artifact | `docs/reviews/section4_5_figure_trace.md` |

25 commits total. All new hashes verified as ancestors of `main`. The subjects of the old and rewritten commits are identical (filter-repo preserves commit messages); the hashes differ only because filter-repo rewrote the trees/parents during the file purge.
