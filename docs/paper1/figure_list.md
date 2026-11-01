# Paper 1 figure list

Approved by Jamion Williams 2026-10-31. Every plotted value comes from a committed artifact; values marked pending or to trace are not yet fixed.

## Figure 1. Study area and graph
Panel a: Hamilton County tracts (85 of 87 carry SVI), with the 20 Paper 1 tracts highlighted.
Panel b: one tract's production graph (road-network shortest paths, geographic k-NN k=6 fallback) as an inset.
Reference figures for the caption: 39,535 addresses, 240,404 undirected edges, mean degree 12.162, zero cross-tract edges (3bcdf7d).
Source: to identify in recon.

## Figure 2. Agreement with the block-group reference
Panel a: pooled block-group r with 95% intervals, 67 block groups, reference v2: GRANITE 0.766 [0.656, 0.848], Dasymetric 0.797 [0.697, 0.872], Pycnophylactic 0.760 [0.644, 0.851].
Source: docs/paper1/number_trace.md, REF-FIX section (b965af4).
Panel b: paired difference GRANITE minus Dasymetric, 90% tract-bootstrap interval against the equivalence margin. Pending: the margin awaits Dr. Gao and is committed before the bootstrap runs.

## Figure 3. Graph randomization
Panel a: output Moran's I, degree-matched spatial k-NN graph (k=10, mean degree 11.742) against its degree-preserving randomizations, per architecture. Randomized arm: crossed 5 training seeds x 5 graph draws, about 0.504 (GraphSAGE) and 0.071 (GCN-GAT), data/results/m6_topology_5b_crossed/ (9be51a6). Structured arm: 0.872 (GraphSAGE) and 0.884 (GCN-GAT), provisional, to trace. Intervals: tract bootstrap, per FIG-P1 B1.
Panel b: pooled block-group r per condition under reference v2, condition means 0.757 to 0.765 (REF-FIX, b965af4).

## Figure 4. Coherence without recovery on known truth
Panel a: synthetic randomization run (FIG-P1 B2): within-tract recovery against known truth and output Moran's I, structured against randomized graph, per architecture. Pending.
Panel b: coordinate-only synthetic grid: median output Moran's I 0.94 (median over 81 draws of each draw's unweighted tract-mean Moran's I; docs/paper1/number_trace.md:274), with within-tract recovery on the same draws (value to trace).
