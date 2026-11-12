#!/usr/bin/env python3
"""
swap-failure rebuild for paper 1 (preregistration v4.1, section 7).

rebuilds every randomized graph from its seed through
DataLoader.check_randomization (granite/data/loaders.py:1249): the
spatial_knn_uniform graph (k = 10) on get_addresses_for_tract, then the
degree-preserving swap at each graph-draw seed. one call per (tract, seed),
catching GraphRandomizationError, so every pair gets a row. trains nothing
and scores nothing.

one rebuild covers both arms: run_sweep.py (frozen 5b and crossed sets) and
run_fig4.py (synthetic figure 4 run) build randomized graphs through the
same path with k = 10, the same seeds and the same address coordinates; the
script asserts the coordinates against both arms' saved npz files.

gate (exit 1 on failure): 47065011324 (18 addresses, 111 edges) is the only
tract with randomized = False, at every seed, share_preserved 1.000; the
other 19 lie in [0.0029, 0.0393] inclusive, with 47065011900 at seed 123
reading 0.03927 (registered range under the pinned scikit-learn 1.8.0; the
83927c3 range 0.003 to 0.036 was measured under a drifted 1.9.1).

usage:
    python scripts/paper1_swap_rebuild.py
writes data/results/paper1_prereg_artifacts/swap_rebuild.json
"""
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from granite.data.loaders import DataLoader
from granite.errors import GraphRandomizationError

RESULTS = REPO_ROOT / 'data' / 'results'
OUT_PATH = RESULTS / 'paper1_prereg_artifacts' / 'swap_rebuild.json'
SWEEP_PATH = REPO_ROOT / 'experiments' / 'ablation' / '05b_topology_specificity' / 'run_sweep.py'
EDGE_AUDIT = REPO_ROOT / 'experiments' / 'audits' / 'graph_edge_audit' / 'graph_edge_audit.json'
GRAPH_DRAW_SEEDS = (42, 17, 123, 2024, 7)
GRAPH_KNN_K = 10
UNSWAPPED = '47065011324'
UNSWAPPED_N_ADDRESSES, UNSWAPPED_N_EDGES = 18, 111
SHARE_RANGE = (0.0029, 0.0393)
CHECK_ROW = ('47065011900', 123, 0.03927)
COORD_FILES = {
    'run_sweep.py (frozen 5b)': RESULTS / 'm6_topology_5b' / 'predictions' /
                                'spatial_knn_uniform__sage__ts42__gs42__{f}.npz',
    'run_sweep.py (crossed)': RESULTS / 'm6_topology_5b_crossed' / 'predictions' /
                              'randomized__sage__ts42__gs42__{f}.npz',
    'run_fig4.py': RESULTS / 'm6_fig4_randomization' / 'predictions' /
                   'randomized__sage__ts42__gs42__ds42__{f}.npz',
}


def _git(*args):
    return subprocess.check_output(['git', *args], cwd=str(REPO_ROOT)).decode().strip()


def _tract_list():
    spec = importlib.util.spec_from_file_location('run_sweep_05b', SWEEP_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._load_tract_list()


def main():
    import networkx
    import sklearn
    cfg = yaml.safe_load((REPO_ROOT / 'config.yaml').read_text()) or {}
    cfg['graph_knn_k'] = GRAPH_KNN_K
    cfg['allow_unswapped'] = []
    dl = DataLoader(str(REPO_ROOT / 'data'), config=cfg)
    tracts = _tract_list()
    audit = json.loads(EDGE_AUDIT.read_text())
    audit_edges = {t['fips']: t['undirected_edges']
                   for t in audit['conditions']['spatial_knn_uniform']['per_tract']}

    rows, coord_checks, edge_checks = [], {}, {}
    for fips in tracts:
        addresses = dl.get_addresses_for_tract(fips)
        xy = np.array([[g.x, g.y] for g in addresses.geometry])
        coord_checks[fips] = {arm: bool(np.array_equal(xy, np.load(str(p).format(f=fips))['coords']))
                              for arm, p in COORD_FILES.items()}
        ei, _ = dl._create_spatial_knn_uniform(addresses, GRAPH_KNN_K)
        edge_checks[fips] = {'rebuilt': int(ei.shape[1] // 2), 'graph_edge_audit': audit_edges[fips],
                             'edge_index_sha256': hashlib.sha256(
                                 np.ascontiguousarray(ei.numpy()).tobytes()).hexdigest(),
                             'edge_index_dtype': str(ei.numpy().dtype)}
        for seed in GRAPH_DRAW_SEEDS:
            try:
                rec = dl.check_randomization([fips], [seed])[0]
                raised = False
            except GraphRandomizationError as e:
                rec = e.records[0]
                raised = True
            rows.append({'arm': 'shared', 'tract': fips, 'seed': seed,
                         'n_addresses': int(len(addresses)), 'n_edges': int(rec['n_edges']),
                         'edges_preserved': int(rec['edges_preserved']),
                         'randomized': bool(rec['randomized']),
                         'share_preserved': float(rec['share_preserved']),
                         'swap_outcome': rec['swap_outcome'],
                         'guard_raised': raised})

    unswapped = sorted({(r['tract'], r['seed']) for r in rows if not r['randomized']})
    other = [r['share_preserved'] for r in rows if r['tract'] != UNSWAPPED]
    u_rows = [r for r in rows if r['tract'] == UNSWAPPED]
    gates = [
        {'gate': 'only 47065011324 has randomized = False, at every seed',
         'observed': unswapped, 'pass': unswapped == [(UNSWAPPED, s) for s in sorted(GRAPH_DRAW_SEEDS)]},
        {'gate': '47065011324 share_preserved 1.000 at every seed',
         'observed': [r['share_preserved'] for r in u_rows],
         'pass': all(r['share_preserved'] == 1.0 for r in u_rows) and len(u_rows) == 5},
        {'gate': '47065011324 has 18 addresses and 111 edges',
         'observed': sorted({(r['n_addresses'], r['n_edges']) for r in u_rows}),
         'pass': {(r['n_addresses'], r['n_edges']) for r in u_rows}
                 == {(UNSWAPPED_N_ADDRESSES, UNSWAPPED_N_EDGES)}},
        {'gate': 'the other 19 tracts lie in [0.0029, 0.0393] inclusive',
         'observed': [min(other), max(other)],
         'pass': (len({r['tract'] for r in rows} - {UNSWAPPED}) == 19
                  and round(min(other), 4) >= SHARE_RANGE[0] and round(max(other), 4) <= SHARE_RANGE[1])},
        {'gate': '47065011900 at seed 123 reads 0.03927',
         'observed': [r['share_preserved'] for r in rows
                      if (r['tract'], r['seed']) == CHECK_ROW[:2]],
         'pass': [round(r['share_preserved'], 5) for r in rows
                  if (r['tract'], r['seed']) == CHECK_ROW[:2]] == [CHECK_ROW[2]]},
        {'gate': 'rebuilt addresses equal the saved coordinates of both arms',
         'observed': sorted({a for c in coord_checks.values() for a, ok in c.items() if not ok}),
         'pass': all(all(c.values()) for c in coord_checks.values())},
        {'gate': 'rebuilt euclidean edge counts equal graph_edge_audit.json',
         'observed': sorted(f for f, e in edge_checks.items() if e['rebuilt'] != e['graph_edge_audit']),
         'pass': all(e['rebuilt'] == e['graph_edge_audit'] for e in edge_checks.values())},
    ]
    for g in gates:
        g['pass'] = bool(g['pass'])
    payload = {
        'milestone': 'TAG-B',
        'artifact': 'swap-failure rebuild (preregistration v4.1, section 7)',
        'path': ('DataLoader.check_randomization (granite/data/loaders.py:1249): '
                 '_create_spatial_knn_uniform(get_addresses_for_tract(fips), k=10), then '
                 '_randomize_graph(seed) degree-preserving double-edge swap; one call per (tract, seed); '
                 'nothing trained or scored'),
        'arms': ('one rebuild covers both arms (arm = shared): consumers run_sweep.py (frozen 5b and '
                 'crossed sets) and run_fig4.py (figure 4 synthetic run) build randomized graphs through '
                 'the same path with k = 10, the same seeds and the same address coordinates'),
        'consumers': ['experiments/ablation/05b_topology_specificity/run_sweep.py',
                      'experiments/m6_recovery_grid/run_fig4.py'],
        'graph_draw_seeds': list(GRAPH_DRAW_SEEDS),
        'graph_knn_k': GRAPH_KNN_K,
        'git_head': _git('rev-parse', 'HEAD'),
        'networkx_version': networkx.__version__,
        'scikit_learn_version': sklearn.__version__,
        'reference_report': ('83927c3 graph-only check (SWAP-GUARD): 0.003 to 0.036 and 47065011324 '
                             '1.000, measured under a drifted scikit-learn 1.9.1, whose k-NN tie-breaking '
                             'builds a different euclidean edge set for 47065011900 (same edge count); '
                             'registered range under the pinned 1.8.0: 0.0029 to 0.0393'),
        'partial_swap_share': ('not exposed: _randomize_graph swaps a local networkx graph and discards '
                               'it on the max_tries failure (granite/data/loaders.py:1181-1213); the '
                               'record compares input with the returned unswapped graph. not recorded '
                               'without editing loaders.py'),
        'gates': gates,
        'all_gates_pass': all(g['pass'] for g in gates),
        'excluded_tracts': sorted({r['tract'] for r in rows if not r['randomized']}),
        'coordinate_checks': coord_checks,
        'edge_count_checks': edge_checks,
        'rows': rows,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=1) + '\n')
    for g in gates:
        print(f"[{'PASS' if g['pass'] else 'FAIL'}] {g['gate']}: {g['observed']}")
    print(f"networkx {networkx.__version__}, scikit-learn {sklearn.__version__}; "
          f"wrote {OUT_PATH.relative_to(REPO_ROOT)}")
    return 0 if payload['all_gates_pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
