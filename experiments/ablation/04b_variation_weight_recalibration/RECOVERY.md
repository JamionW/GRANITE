# Step 4b Recovery Inspection

Generated: 2026-06-03

## Working tree state

### git status (verbatim)
```
On branch main
Your branch is ahead of 'origin/main' by 8 commits.
  (use "git push" to publish your local commits)

Untracked files:
  (use "git add <file>..." to include in what will be committed)
        .claude/
```

### git log -5 --oneline (verbatim)
```
2a3e610 Step 4b complete: variation_weight sweep under cbc_no_shift; Outcome C (see experiments/ablation/04b_variation_weight_recalibration/summary/README.md)
b0db535 Step 4 summary: constraint-by-construction sweep (see experiments/ablation/04_constraint_by_construction/summary/README.md)
1d1deff Step 4 fixups: correct pre_correction_constraint_error capture site, fix config snapshot timing (see experiments/ablation/04_constraint_by_construction/ANOMALY.md)
e988de4 Audit followup: remove orphaned variation_weight key, rename accessibility_consistency_loss to min_spread_loss (see experiments/audits/loss_term_audit.md entries 003, 008, 011)
53d056d Complete loss-term audit started after smoothness cleanup; see experiments/audits/loss_term_audit_summary.md
```

### Uncommitted changes
Only `.claude/` (untracked). No staged or unstaged changes to tracked files.

### Branch sync status
8 commits ahead of origin/main. Not behind. No pending merges.

---

## Per-variant state

Note: `per_tract_metrics.csv` is stored at the variant root (not in `results/`). This is the actual layout used by the ablation script.

### 00_w_0p8
- directory exists: YES
- per_tract_metrics.csv: EXISTS at root (41 lines = 1 header + 40 rows = 20 tracts x 2 architectures)
- results/aggregate_metrics.json: EXISTS
- results/block_group_validation.json: EXISTS
- figures/: 6 files (architecture_overlap.png, block_group_scatter.png, constraint_error_dist.png, feature_importance_top20.png, morans_i_by_tract.png, spatial_std_by_svi.png)
- partial/truncated: NO

### 01_w_1p5
- directory exists: YES
- per_tract_metrics.csv: EXISTS at root (41 lines)
- results/aggregate_metrics.json: EXISTS
- results/block_group_validation.json: EXISTS
- figures/: 6 files
- partial/truncated: NO

### 02_w_2p5
- directory exists: YES
- per_tract_metrics.csv: EXISTS at root (41 lines)
- results/aggregate_metrics.json: EXISTS
- results/block_group_validation.json: EXISTS
- figures/: 6 files
- partial/truncated: NO

### 03_w_4p0
- directory exists: YES
- per_tract_metrics.csv: EXISTS at root (41 lines)
- results/aggregate_metrics.json: EXISTS
- results/block_group_validation.json: EXISTS
- figures/: 6 files
- partial/truncated: NO

---

## Summary state

- summary/delta_vs_cbc_baseline.json: EXISTS
- summary/README.md: EXISTS
- summary/variation_weight_sweep.png: EXISTS
- summary/spread_vs_generalization.png: EXISTS
- summary/extreme_tract_recalibration.png: EXISTS

All required summary artifacts present.

---

## Sanity check (00_w_0p8 vs step 4 cbc_no_shift baseline)

Expected (step 4 02_cbc_no_shift):
- SAGE: std=0.0595, Moran's I=0.8669, pooled BG r=0.7511
- GCN-GAT: std=0.0830, Moran's I=0.8420, pooled BG r=0.7481

Observed (00_w_0p8/results/aggregate_metrics.json + delta_vs_cbc_baseline.json):
- SAGE: std=0.059486, Moran's I=0.866872, pooled BG r=0.751064
- GCN-GAT: std=0.082895, Moran's I=0.842090, pooled BG r=0.748032

Deltas (delta_vs_cbc_baseline entries for weight=0.8 are all exactly 0.0 by construction -- the 0.8 variant IS the cbc_no_shift baseline):
- SAGE std delta: 0.0
- SAGE Moran's I delta: 0.0
- SAGE pooled BG r delta: 0.0
- GCN-GAT std delta: 0.0
- GCN-GAT Moran's I delta: 0.0
- GCN-GAT pooled BG r delta: 0.0

Sanity check: PASS (all deltas = 0 within floating-point representation).

---

## Code wiring verification

### variation_weight in MultiTractGNNTrainer.__init__
Present at granite/models/gnn.py line 774:
`self.variation_weight = config.get('variation_weight', 0.8)`

### self.variation_weight * variation_loss in _compute_multi_tract_losses
Present at granite/models/gnn.py line 1371:
`self.variation_weight * variation_loss +  # configurable; default 0.8; see step 4b sweep`

### variation_activation_count in training loop
Present in MultiTractGNNTrainer training loop (confirmed via grep).

### config.yaml
Contains `variation_weight: 0.8` under `training:` at line 69.

### TestVariationWeightWiring
Present in tests/test_loss_terms.py at line 133. Six test methods covering both trainers.

All wiring verified: PASS.

---

## Chosen path

**Path F+**: Everything complete including summary AND committed at HEAD (2a3e610). The sweep finished before the Codespaces disconnect and the commit landed. The only remaining action is to push the branch (8 commits ahead of origin/main) to origin.

No variant reruns needed. No artifact regeneration needed. No code changes needed.

---

## Action taken

Wrote this RECOVERY.md. Awaiting user confirmation to push 8 commits to origin/main.
