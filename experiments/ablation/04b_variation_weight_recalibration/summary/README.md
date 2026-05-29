# ablation step 4b: variation_weight recalibration under cbc_no_shift

git sha: b0db535fb5d5dd2ad46afe75a19416de79019c0b
seed: 42
generated: 2026-05-29 16:53

## context

Step 4 found that SAGE within-tract std collapsed under cbc_no_shift (0.060 vs 0.082 soft).
This sweep tests whether raising variation_weight recovers spread.

## 00_w_0p8 vs step 4 02_cbc_no_shift (informational)

Step 4 02_cbc_no_shift used variation_weight=1.5 (single-tract hardcoded default).
This sweep uses variation_weight=0.8. Delta is expected and non-zero.

- sage: spatial_std_mean got=0.0595 ref(step4,w=1.5)=0.0595 delta=-0.0000 | bg_r got=0.7511 ref=0.7511 delta=-0.0000
- gcn_gat: spatial_std_mean got=0.0829 ref(step4,w=1.5)=0.0830 delta=-0.0001 | bg_r got=0.7480 ref=0.7481 delta=-0.0000

## sweep results

| weight | SAGE std | SAGE BG r | GCN-GAT std | GCN-GAT BG r | SAGE act_rate | GCN-GAT act_rate |
|---|---|---|---|---|---|---|
| 0.8 | 0.0595 | 0.7511 | 0.0829 | 0.7480 | n/a | n/a |
| 1.5 | 0.0595 | 0.7511 | 0.0830 | 0.7481 | 0.0000 | 0.0005 |
| 2.5 | 0.0595 | 0.7511 | 0.0833 | 0.7481 | 0.0000 | 0.0010 |
| 4.0 | 0.0595 | 0.7511 | 0.0834 | 0.7480 | 0.0000 | 0.0010 |
| soft (ref) | 0.0823 | 0.7537 | 0.0814 | 0.7664 | - | - |

## primary question

Does any variation_weight in [0.8, 4.0] recover SAGE within-tract std to within 10% of soft mode
(0.0823)? Target: std >= 0.0738 with pooled BG r not dropping by > 0.02.

## verdict

**Outcome C**: no recovery. No variation_weight in [0.8, 4.0] restored SAGE spread without other damage. Recommend soft mode as step 5 default. Document cbc spread collapse as a finding.

## step 5 launch decision

Proceed with step 5 using `constraint_mode: soft`. cbc_no_shift limitation documented.

## figures

- `variation_weight_sweep.png`: 3x2 grid; rows=metric, cols=architecture; x-axis=weight (log); soft reference lines
- `spread_vs_generalization.png`: 2x1; twin y-axes (within-tract std, pooled BG r) vs variation_weight
- `extreme_tract_recalibration.png`: 2x2; extreme tracts x metrics, soft reference lines

