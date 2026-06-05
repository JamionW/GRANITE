# Step 5: graph contribution boundary test (two-pole)

Status: **complete**

## Design

Two conditions, both architectures, five seeds each (20 runs total).

| parameter | value |
|---|---|
| conditions | production, mlp_floor |
| architectures | SAGE, GCN-GAT |
| seeds | 42, 17, 123, 2024, 7 |
| constraint_mode | soft |
| variation_weight | 0.8 |
| epochs | 150 |

**production**: live hybrid road-network-plus-geographic graph, unchanged from prior steps.
**mlp_floor**: self-loops only (`edge_index = [[i],[i]] for all i`, `edge_weight = 1.0`). SAGE and GAT reduce to node-wise functions on the input features.

Moran's I computed from address coordinates (k=8, symmetrized, row-normalized), identical across conditions.

## Git sha

87ca99cba1702be36eb01abcebd44af87adab609

## Sanity regression

production / seed=42 must reproduce step 4 soft baseline within 1e-3 (BG r) / 2e-3 (std).

| arch | metric | reference | got | delta | pass |
|---|---|---|---|---|---|
| sage | pooled_bg_r | 0.7537 | 0.7537 | -0.0000 | PASS |
| sage | within_tract_std | 0.0823 | 0.0823 | +0.0000 | PASS |
| gcn_gat | pooled_bg_r | 0.7664 | 0.7664 | +0.0000 | PASS |
| gcn_gat | within_tract_std | 0.0814 | 0.0814 | +0.0000 | PASS |

Sanity regression: **PASSED**

## Metrics table

(mean +/- across-seed std; std computed over 5 seeds, not 20 tracts)

| condition | arch | within_tract_std | Moran's I | pooled_bg_r |
|---|---|---|---|---|
| production | sage | 0.0899 +/- 0.0190 | 0.8570 +/- 0.0267 | 0.7632 +/- 0.0070 |
| production | gcn_gat | 0.0906 +/- 0.0065 | 0.8368 +/- 0.0237 | 0.7639 +/- 0.0168 |
| mlp_floor | sage | 0.0832 +/- 0.0167 | 0.6820 +/- 0.0572 | 0.7714 +/- 0.0115 |
| mlp_floor | gcn_gat | 0.0812 +/- 0.0068 | 0.6747 +/- 0.0682 | 0.7660 +/- 0.0099 |

## Verdict

Primary question: does mlp_floor fall within the production seed band on Moran's I and within-tract std?

**SAGE: graph contributes.** mlp_floor falls outside the production seed band on Moran's I (0.6820 vs prod 0.8570+/-0.0267). The gap is the road-network graph's measurable contribution. A full construction sweep (road, feature-similarity, randomized) is the recommended follow-up to characterize which wiring type drives the gain.

**GCN_GAT: graph contributes.** mlp_floor falls outside the production seed band on within-tract std (0.0812 vs prod 0.0906+/-0.0065) and Moran's I (0.6747 vs prod 0.8368+/-0.0237). The gap is the road-network graph's measurable contribution. A full construction sweep (road, feature-similarity, randomized) is the recommended follow-up to characterize which wiring type drives the gain.

## Figure

`graph_contribution.png`: 3x2 grid, rows = within_tract_std / Moran's I / pooled BG r, cols = SAGE / GCN-GAT, x = {production, mlp_floor}, error bars = across-seed std. Production band drawn as horizontal reference span.
