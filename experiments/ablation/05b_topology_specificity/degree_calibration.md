# Step 5b degree calibration

## Four-condition degree table at k=10

Source: `--check-only` run, commit 8644a43, all 20 n20 tracts pooled.

| condition | mean_degree | total_edges | n_tracts |
|-----------|-------------|-------------|----------|
| spatial_knn_uniform | 11.742 | 232109 | 20 |
| road_network_uniform | 12.985 | 256678 | 20 |
| randomized | 11.742 | 232109 | 20 |
| production | 12.162 | 240404 | 20 |

## Production as reference

Production is the reference graph: the road-network hybrid (`_create_road_network_graph`,
road edges plus geographic euclidean top-up at max 6 neighbors) that produced the canonical
pooled BG r reported in the dissertation. Pooled mean degree 12.162, median 12, std 3.149,
across 39535 nodes. This is the anchor the three variants are compared against.

## k=10 as the matched value

At k=10, spatial_knn_uniform reaches mean degree 11.742, which is the closest achievable
integer-k degree below production (12.162). k=11 would overshoot to approximately 12.8,
past production. The three variants therefore bracket production within about 10 percent at
a single shared k, so no per-variant k key is needed in config.yaml.

## Each condition's role

- **spatial_knn_uniform**: uniform euclidean k-NN topology at matched degree; the
  structured-graph baseline with edge weights all 1.0.
- **randomized**: degree-preserving double-edge swap applied to spatial_knn_uniform; shares
  spatial's exact degree sequence node by node and permutes only edge placement. Any Moran's
  I gap between spatial and randomized is attributable to edge structure alone, not degree.
- **road_network_uniform**: road-distance shortest-path k-NN at the same k, edge weights 1.0;
  tests whether road-path neighbor selection beats euclidean neighbor selection when degree is
  held approximately fixed.
- **production**: fixed reference; road-network hybrid with distance-weighted edges and
  euclidean top-up; not k-controlled; included to anchor all three uniform variants against
  the graph that generated the step-5 Moran's I result.

The primary degree-controlled contrast is spatial vs randomized: identical degree sequence,
different topology, same k, same weights. A Moran's I gap there isolates edge structure as
the causal factor.

## road_knn_k calibration (2026-06-10)

At k=10, road_network_uniform had mean degree 12.985 (total_edges=256678), exceeding
spatial by 1.243 -- above DEGREE_PARITY_TOL=0.5. This confounds topology with density.

A per-variant key `road_knn_k` was added to config.yaml and loaders.py so road_network_uniform
can be calibrated independently of spatial and randomized (which remain at graph_knn_k=10).

At road_knn_k=9, road_network_uniform reaches mean degree 11.740 (total_edges=232076), gap
0.002 from spatial's 11.742. Parity check passes with margin 0.498.

| condition | k | mean_degree | total_edges | n_tracts |
|-----------|---|-------------|-------------|----------|
| spatial_knn_uniform | 10 | 11.742 | 232109 | 20 |
| road_network_uniform | 9 | 11.740 | 232076 | 20 |
| randomized | 10 | 11.742 | 232109 | 20 |
| production | n/a | 12.162 | 240404 | 20 |

road_knn_k=9 is the committed value.

## Provenance of the retired 11.742 target

The value 11.742 appeared in a prior session as a "spatial mean degree target" without a
source. It traces to the spatial_knn_uniform degree at k=10 -- spatial's own output -- copied
into an ecological-fallacy CSV without provenance annotation. It was never a production
measurement. Production's real pooled mean degree is 12.162, measured by running
`_create_road_network_graph` on all 20 n20 tracts in commit 8644a43.
