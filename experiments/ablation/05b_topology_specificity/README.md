# Step 5b: Topology Specificity Sweep

**Status:** pending run

**Git SHA:** (fill after run)

## Purpose

Discriminate whether road-network wiring specifically raises output spatial
autocorrelation, or whether any within-tract local graph does it through generic
smoothing. Three conditions with uniform edge weight across all three so neighbor
selection is the only moving part.

## Conditions

| condition | neighbor selection | edge weight |
|-----------|-------------------|-------------|
| spatial_knn_uniform | euclidean k-NN, k=10 | 1.0 |
| road_network_uniform | road-snapped shortest-path k-NN, k=10 | 1.0 |
| randomized | degree-preserving swap on spatial_knn_uniform | 1.0 |

## Questions each condition answers

- **spatial_knn_uniform vs randomized**: does destroying spatial neighbor
  structure (degree held fixed) collapse Moran's I toward the step-5 floor? If
  yes, spatial wiring carries signal. If randomized stays near the structured
  graphs, the lift is generic smoothing.
- **road_network_uniform vs spatial_knn_uniform**: does road-path neighbor
  selection beat euclidean neighbor selection? A tie means spatial structure
  matters and road-specificity does not (weaker claim). A road win is the
  accessibility-earns-its-complexity result.

## Seeds

- structured (spatial_knn_uniform, road_network_uniform): training seeds [42, 17, 123, 2024, 7], one fixed graph
- randomized: training_seed=42, graph_draw_seeds [42, 17, 123, 2024, 7]

## Flagged choices

- k=10 is a single representative degree; production's hybrid uses up to 10 road
  plus 6 euclidean neighbors. k=10 uniform is not a reproduction of production degree.
- road_network_uniform strips the euclidean top-up that production carries, so the
  spatial-vs-road contrast is pure.

## Degree parity

(fill after run)

spatial_knn_uniform mean_degree: TBD
road_network_uniform mean_degree: TBD
randomized mean_degree: TBD
jaccard(spatial, road): TBD

## Metrics table

(fill after run)

| condition | arch | Moran's I | +/- | BG r | +/- | within_tract_std | +/- |
|-----------|------|-----------|-----|------|-----|-----------------|-----|

## Step-5 reference poles

| condition | arch | Moran's I | BG r | within_tract_std |
|-----------|------|-----------|------|-----------------|
| production (distance-weighted hybrid) | SAGE | 0.8570 | 0.7632 | 0.0899 |
| production (distance-weighted hybrid) | GCN-GAT | 0.8368 | 0.7639 | 0.0906 |
| mlp_floor (no-graph floor) | SAGE | 0.6820 | 0.7714 | 0.0832 |
| mlp_floor (no-graph floor) | GCN-GAT | 0.6747 | 0.7660 | 0.0812 |

## Verdict

(fill after run)

### SAGE

(a) Randomized vs step-5 floor: TBD
(b) Road vs spatial: TBD

### GCN-GAT

(a) Randomized vs step-5 floor: TBD
(b) Road vs spatial: TBD
