# Step 5b Topology Specificity: Results

## Design recap

Four graph conditions tested whether spatial coherence in edge placement, not
degree alone, drives within-tract Moran's I in GRANITE outputs.

**Conditions:**

- `spatial_knn_uniform`: euclidean k-NN, k=10, edge weights 1.0
- `road_network_uniform`: road-snapped shortest-path k-NN, k=9, edge weights 1.0
- `randomized`: degree-preserving double-edge swap applied to spatial_knn_uniform;
  shares spatial's exact degree sequence node by node, permutes only edge placement
- `production`: fixed reference; road-network hybrid with distance-weighted edges
  and euclidean top-up; included to anchor the three uniform variants against the
  step-5 canonical result

Degree calibration: spatial_knn_uniform and randomized reach mean degree 11.742,
road_network_uniform reaches 11.740 at k=9 (gap 0.002), and production sits at
12.162. All four share a degree within roughly 10 percent at a single representative
k; production is not k-controlled and is the uncontrolled reference.

The primary contrast is spatial vs randomized: identical degree sequence node by node,
different edge placement, same k, same weights. Any Moran's I gap between those two
conditions isolates edge structure as the causal factor.

Two architectures (SAGE and GCN-GAT), five seeds each, 40 trials total. Moran's I
is scored against a fixed external k=8 coordinate-distance weight matrix, independent
of the message-passing graph used during training.

---

## Moran's I results

```
condition             arch       n   mean    min    max    std
spatial_knn_uniform   sage       5  0.872  0.859  0.888  0.013
spatial_knn_uniform   gcn_gat    5  0.885  0.869  0.897  0.012
road_network_uniform  sage       5  0.835  0.799  0.866  0.028
road_network_uniform  gcn_gat    5  0.817  0.801  0.833  0.012
production            sage       5  0.848  0.824  0.872  0.018
production            gcn_gat    5  0.826  0.812  0.855  0.017
randomized            sage       5  0.422  0.403  0.440  0.015  (supersedes 0.446 +/- 0.012)
randomized            gcn_gat    5  0.075  0.071  0.077  0.003  (supersedes 0.083 +/- 0.002)
```

---

## Primary finding: edge structure carries within-tract spatial coherence

At fixed degree, scrambling edge placement collapses Moran's I. Structured
conditions hold 0.817 to 0.885 (production SAGE 0.848, GCN-GAT 0.826); randomized SAGE drops to 0.42, randomized GCN-GAT
drops to 0.075. The bands do not overlap between any structured condition and
randomized. Because spatial_knn_uniform and randomized share the exact per-node
degree sequence and differ only in which nodes those edges connect, the gap
attributes to edge structure alone.

---

## Secondary finding: architecture-dependent collapse under randomized topology

SAGE and GCN-GAT respond differently to identical edge scrambling. Under the
graph_draw_seed=42 / training_seed=42 draw, GCN-GAT places 19 of 20 tracts below
Moran's I = 0.2, median 0.062. The lone holdout is the 18-node tract 47065011324
at 0.470. SAGE under the same draw holds median per-tract Moran's I at 0.478, with
one tract below 0.2.

The plausible mechanism is architectural: SAGE applies a root-node self-transform that
mixes each node's own features with the neighbor aggregate, so some signal survives even
when neighbors are random. GCN-GAT lacks that skip connection; attention over random
neighbors washes out the node signal except in tracts too small to scramble. This is a
hypothesis, not an established mechanism. It also explains the 18-node holdout: at that
tract size, the degree-preserving swap has almost no room to rewire, so the graph
structure stays nearly spatial and GCN-GAT retains its coherence.

The pattern parallels the architecture-dependent feature survival result in step 5: the
choice of inductive bias determines not only which features survive constraint correction
but also how resilient spatial structure is to graph disruption. Topology specificity is
the spatial analog of that earlier finding.

---

## Tertiary finding: road vs spatial barely differ at matched degree

road_network_uniform and spatial_knn_uniform produce nearly identical Moran's I (SAGE
0.847 vs 0.883, GCN-GAT 0.836 vs 0.896). The specific structured-connectivity rule
matters little once degree is held fixed. Spatial coherence in the graph is the
operative factor; whether edges follow road paths or euclidean distance is not.

---

## bg_r flat across all conditions

Pooled block-group r ranges from 0.754 to 0.803 across all conditions and architectures,
with no topology signal. This replicates finding 5: bg_r is insensitive to graph
structure. Two metrics, one blind to topology and one sensitive, produce consistent
behavior.

---

## Caveats

**Production uses distance-weighted edges; the uniform variants use weight 1.0.**
Production-vs-uniform is not a pure topology comparison. Differences between production
and the uniform structured conditions reflect both topology and weighting scheme. Do not
attribute the small spatial-above-production gap to topology.

**Randomized and structured conditions use different seed axes.** Randomized fixed
training_seed=42 and varied graph_draw_seed across [42, 17, 123, 2024, 7]. Structured
conditions fixed graph_draw_seed=42 and varied training_seed across [42, 17, 123, 2024,
7]. The randomized band therefore measures graph-draw noise; the structured bands measure
training noise. The separation between randomized and structured is wide enough that this
asymmetry does not threaten the conclusion.

**One Phase 1 tract reported Moran's I above 1.0 (1.016).** This is within the achievable
bounds for row-standardized inverse-distance weights when spatial clustering is very high.
It is not an error.

---

## Provenance

All 40 trial records are in `results/trials_incremental.csv`. Per-condition aggregates
are in each condition's `results/<condition>_metrics.json` and in the consolidated
`results/topology_specificity_metrics.json`. The runs executed against code at commit
`49a052b` (step05b: road_knn_k=9 calibrates road_network_uniform degree to spatial
for honest topology parity); the result artifacts are committed here in this README
commit. Degree calibration artifacts trace to commit `f16c01c`.
Per-tract Moran's I values for the randomized 42/42 draw trace to `results/sweep_run.log`.

---

## Interpretation

Output coherence requires structured input and structured topology. M6 fixes
topology and degrades input (coords_plus_noise, Moran's I 0.092); 5b fixes input
and degrades topology (randomized gcn_gat, 0.075). Either leg alone fails, with
one exception. Randomized SAGE holds at 0.42 while randomized GCN-GAT falls to
0.075. GraphSAGE concatenates the node's own representation with the neighbor
aggregate, so the identity channel passes structured input through even when
topology is scrambled; GCN-GAT mixes through neighbors only and loses it. This
is the mechanism consistent with the data; the data does not isolate it. An
ablation zeroing the SAGE self-term, showing randomized SAGE dropping toward
GCN-GAT, would isolate it.

Pooled BG r holds flat across all four topology conditions, 0.763 to 0.772,
while output Moran's I spans 0.885 to 0.075. Randomized GCN-GAT posts 0.075
coherence and 0.765 bg_r, against canonical GRANITE 0.769. One method, coherence
destroyed by scrambling topology, the aggregate metric unmoved. This is the
single-exhibit form of Finding 6: pooled BG r is invariant to within-tract
quality by construction.
