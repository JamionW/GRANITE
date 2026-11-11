"""
Graph edge audit for the 05b topology-specificity conditions (production,
spatial_knn_uniform, road_network_uniform, randomized).

Reconstructs each condition's per-tract graph via the shipped builder
(granite.data.loaders.DataLoader.create_spatial_accessibility_graph) over the
20 n20 tracts and reports, per condition and per tract: node count, directed
edge_index row count, undirected edge count, mean degree, and whether every
directed row has a matching reverse row (both_dir_ok).

Read-only: does not touch any file under data/ or experiments/ other than
writing this script's own output JSON.
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from granite.data.loaders import DataLoader

TRACT_SELECTION = (
    REPO_ROOT / 'experiments' / 'ablation' / '00_baseline' / 'tract_selection.txt'
)


def load_tract_list():
    fips = []
    in_list = False
    with open(TRACT_SELECTION) as f:
        for line in f:
            line = line.strip()
            if line == 'fips_list:':
                in_list = True
                continue
            if in_list and line:
                fips.append(line.lstrip('- ').strip())
    return fips


CONDITIONS = ['production', 'spatial_knn_uniform', 'road_network_uniform', 'randomized']


def both_directions_present(edge_index):
    ei_np = edge_index.numpy()
    pairs = set()
    for col in range(ei_np.shape[1]):
        u, v = int(ei_np[0, col]), int(ei_np[1, col])
        pairs.add((u, v))
    for (u, v) in pairs:
        if u != v and (v, u) not in pairs:
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description='graph edge audit for the 05b conditions')
    parser.add_argument('--allow-unswapped', default='',
                        help=(
                            'comma-separated tract FIPS allowed to train on an unswapped '
                            'randomized graph (recorded randomized=False, warned). any other '
                            'unswapped tract raises GraphRandomizationError.'
                        ))
    args = parser.parse_args()
    allow_unswapped = [f.strip() for f in args.allow_unswapped.split(',') if f.strip()]

    tract_list = load_tract_list()
    assert len(tract_list) == 20, f"expected 20 tracts, got {len(tract_list)}"

    dl = DataLoader(str(REPO_ROOT / 'data'), config={'allow_unswapped': allow_unswapped})

    per_condition = {
        cond: {'n_nodes': 0, 'n_undirected_edges': 0, 'n_directed_edges': 0,
               'per_tract': []}
        for cond in CONDITIONS
    }

    for fips in tract_list:
        addresses = dl.get_addresses_for_tract(fips)
        n = len(addresses)
        dummy_features = np.zeros((n, 10), dtype=np.float32)

        for cond in CONDITIONS:
            dl.config['graph_variant'] = cond
            dl.config['graph_draw_seed'] = 42
            dl.config['graph_knn_k'] = 10
            # committed calibration value (experiments/ablation/05b_topology_specificity/
            # degree_calibration.md: "road_knn_k=9 is the committed value")
            dl.config['road_knn_k'] = 9 if cond == 'road_network_uniform' else 10
            if cond == 'randomized':
                dl._under_mixed_tracts = 0

            g = dl.create_spatial_accessibility_graph(addresses, dummy_features)
            ei = g.edge_index
            n_directed = ei.shape[1]
            n_undirected = n_directed // 2
            ok = both_directions_present(ei)

            per_condition[cond]['n_nodes'] += n
            per_condition[cond]['n_undirected_edges'] += n_undirected
            per_condition[cond]['n_directed_edges'] += n_directed
            per_condition[cond]['per_tract'].append({
                'fips': fips,
                'n_nodes': n,
                'undirected_edges': n_undirected,
                'directed_edges': n_directed,
                'both_dir_ok': ok,
            })

    summary = {}
    header = f"{'condition':<22}{'nodes':>8}{'directed_edges':>16}{'undirected_edges':>18}{'mean_degree':>14}{'both_dir_ok_all':>16}{'cross_tract_edges':>18}"
    print(header)
    for cond in CONDITIONS:
        r = per_condition[cond]
        mean_degree = 2 * r['n_undirected_edges'] / r['n_nodes']
        both_ok_all = all(t['both_dir_ok'] for t in r['per_tract'])
        # per-tract construction (get_addresses_for_tract + one graph call per tract,
        # no concatenation step for any graph_variant): a cross-tract edge is not
        # representable, since each tract's node indices are local to that call.
        cross_tract_edges = 0
        summary[cond] = {
            'n_nodes': r['n_nodes'],
            'n_directed_edges': r['n_directed_edges'],
            'n_undirected_edges': r['n_undirected_edges'],
            'mean_degree': mean_degree,
            'both_dir_ok_all': both_ok_all,
            'cross_tract_edges': cross_tract_edges,
            'per_tract': r['per_tract'],
        }
        print(f"{cond:<22}{r['n_nodes']:>8}{r['n_directed_edges']:>16}{r['n_undirected_edges']:>18}{mean_degree:>14.6f}{str(both_ok_all):>16}{cross_tract_edges:>18}")

    out_path = REPO_ROOT / 'experiments' / 'audits' / 'graph_edge_audit' / 'graph_edge_audit.json'
    with open(out_path, 'w') as f:
        json.dump({
            'tracts': tract_list,
            'n_tracts': len(tract_list),
            'config': {'graph_knn_k': 10, 'road_knn_k': 9, 'graph_draw_seed': 42},
            'conditions': summary,
        }, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == '__main__':
    main()
