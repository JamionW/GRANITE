# Graph edge audit (TRACE-P1-CLOSE B1)

Committed version of the `/tmp/recon_p1c/q1_q2_graph_recon.py` read-only recon
(RECON-P1c, Q1/Q2). `graph_edge_audit.py` reconstructs all four 05b `graph_variant`
conditions (`production`, `spatial_knn_uniform`, `road_network_uniform`, `randomized`) over the
20 n20 tracts via the shipped builder
(`granite.data.loaders.DataLoader.create_spatial_accessibility_graph`) and writes
`graph_edge_audit.json`.

## Key citations (`granite/data/loaders.py`)

- **`randomized` rewires `spatial_knn_uniform`, not production or road_network_uniform**
  (`loaders.py:918-921`):
  ```python
  elif graph_variant == 'randomized':
      k = self.config.get('graph_knn_k', 10)
      graph_draw_seed = self.config.get('graph_draw_seed', 42)
      knn_ei, knn_ew = self._create_spatial_knn_uniform(addresses, k)
      edge_index, edge_weight = self._randomize_graph(knn_ei, knn_ew, n_addresses, graph_draw_seed)
  ```
  `_randomize_graph` (`loaders.py:1127`) runs `nx.double_edge_swap` (`loaders.py:1154`) on that
  parent graph, preserving its exact degree sequence.
- **Every edge is stored in both directions** (no `to_undirected` call anywhere in the file):
  `loaders.py:1009` (`_create_spatial_knn_uniform`), `1101`/`1114` (`_create_road_network_uniform`,
  snapped/fallback loops), `1169` (`_randomize_graph`), `1211` (`_create_road_network_graph`,
  production). `edge_index.shape[1]` is therefore twice the undirected edge count for all four
  conditions.
- **Zero cross-tract edges by construction, for all four conditions**: each graph is built from
  `DataLoader.get_addresses_for_tract(fips)` (one tract) and one
  `create_spatial_accessibility_graph` call per tract -- no concatenation step exists for any
  `graph_variant`. Node indices are local to that one call, so a cross-tract edge is not merely
  absent, it is not representable by the data structure. This extends RECON-XT1's
  production-only "zero by construction, not by filter" verdict to the three uniform/randomized
  variants.

## Measured (pooled over the 20 n20 tracts, `graph_knn_k=10`, `road_knn_k=9`, `graph_draw_seed=42`)

```
condition                nodes  directed_edges  undirected_edges   mean_degree both_dir_ok_all cross_tract_edges
production               39535          480808            240404     12.161578            True                 0
spatial_knn_uniform      39535          464218            232109     11.741950            True                 0
road_network_uniform     39535          464152            232076     11.740281            True                 0
randomized               39535          464218            232109     11.741950            True                 0
```

`spatial_knn_uniform` and `randomized` have identical undirected edge counts (232,109), as
expected for a degree-preserving swap. Matches the committed
`experiments/ablation/05b_topology_specificity/degree_calibration.md` table exactly, and matches
the untracked `claude_XT_Edge_Audit.md`'s production numbers (480,808 directed rows, 12.162 mean
degree, 0 cross-tract).

## `graph_edge_audit.json` schema

Top level: `tracts` (20 FIPS), `n_tracts`, `config` (`graph_knn_k`, `road_knn_k`,
`graph_draw_seed`), `conditions` (one key per condition, each with `n_nodes`,
`n_directed_edges`, `n_undirected_edges`, `mean_degree`, `both_dir_ok_all`,
`cross_tract_edges`, and a `per_tract` list of per-FIPS `{n_nodes, undirected_edges,
directed_edges, both_dir_ok}`).

## Scope note

`scripts/diag/xt_edge_audit.py` and the root-level `claude_XT_Edge_Audit.md` remain gitignored
and are not un-ignored by this commit (per instruction). This directory is an independent,
newly-committed reproduction of the same measurement from the shipped builder, not a relocation
of the untracked files.

## Does not modify

No file under `data/` or `experiments/` other than this directory's own
`graph_edge_audit.json` was touched. `granite/data/loaders.py` was read, not edited.
