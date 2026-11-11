"""swap guard on the randomized graph variant (SWAP-GUARD).

synthetic in-memory graphs only; no tract data, no prediction files.
"""
import csv
import importlib.util
import tempfile
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pytest
import torch
from shapely.geometry import Point

from granite.data.loaders import DataLoader
from granite.errors import GraphRandomizationError

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_SWEEP = REPO_ROOT / 'experiments' / 'ablation' / '05b_topology_specificity' / 'run_sweep.py'


def _edge_index(G, offset=0):
    pairs = []
    for u, v in G.edges():
        pairs += [[u + offset, v + offset], [v + offset, u + offset]]
    return torch.LongTensor(pairs).t().contiguous()


def _undirected(ei):
    a = ei.numpy()
    return {(int(min(u, v)), int(max(u, v))) for u, v in zip(a[0], a[1])}


def _degrees(ei, n):
    return np.bincount(ei.numpy()[0], minlength=n)


def _loader(**config):
    # loader creates data_dir on init; give it a throwaway directory
    return DataLoader(data_dir=tempfile.mkdtemp(prefix='swapguard_'), config=dict(config))


def _load_run_sweep():
    spec = importlib.util.spec_from_file_location('run_sweep_05b', RUN_SWEEP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_swap_succeeds_on_sparse_graph():
    dl = _loader()
    tracts = {'T1': nx.random_regular_graph(4, 30, seed=1),
              'T2': nx.random_regular_graph(6, 25, seed=2)}
    offset = 0
    combined = []
    for fips, G in tracts.items():
        n = G.number_of_nodes()
        ei = _edge_index(G)
        out, _ = dl._randomize_graph(ei, None, n, 42, tract_fips=fips)
        rec = dl._last_swap_record
        assert rec['randomized'] is True
        assert rec['tract_fips'] == fips
        assert rec['n_edges'] == G.number_of_edges()
        assert rec['share_preserved'] < 1.0
        assert rec['swap_outcome'] == 'swapped'
        assert np.array_equal(_degrees(out, n), _degrees(ei, n))
        combined.append((offset, offset + n, out + offset))
        offset += n
    # per-tract graphs stacked: every edge stays inside its own tract block
    for lo, hi, ei in combined:
        a = ei.numpy()
        assert a.min() >= lo and a.max() < hi
    assert len(dl._swap_records) == 2


def test_complete_graph_raises_when_not_allowlisted():
    dl = _loader()
    G = nx.complete_graph(10)
    with pytest.raises(GraphRandomizationError) as exc:
        dl._randomize_graph(_edge_index(G), None, 10, 17, tract_fips='47065011324')
    assert '47065011324' in str(exc.value)
    assert 'seed=17' in str(exc.value)
    rec = exc.value.records[0]
    assert rec['randomized'] is False
    assert rec['share_preserved'] == 1.0
    assert rec['allowlisted'] is False


def test_complete_graph_proceeds_when_allowlisted(capsys):
    dl = _loader(allow_unswapped=['47065011324'])
    G = nx.complete_graph(10)
    ei = _edge_index(G)
    with pytest.warns(UserWarning, match='47065011324'):
        out, _ = dl._randomize_graph(ei, None, 10, 17, tract_fips='47065011324')
    assert _undirected(out) == _undirected(ei)
    rec = dl._last_swap_record
    assert rec['randomized'] is False
    assert rec['allowlisted'] is True
    assert dl._under_mixed_tracts == 1
    assert 'WARNING' in capsys.readouterr().out


def test_allowlist_accepts_comma_string():
    dl = _loader(allow_unswapped='47065000100, 47065011324')
    assert dl._allow_unswapped() == {'47065000100', '47065011324'}


def test_check_randomization_lists_every_failing_tract(monkeypatch):
    graphs = {'A': ('ring', 30), 'B': ('dense', 8), 'C': ('dense', 9)}

    def fake_addresses(fips):
        kind, n = graphs[fips]
        if kind == 'ring':
            pts = [Point(np.cos(t), np.sin(t)) for t in np.linspace(0, 2 * np.pi, n, endpoint=False)]
        else:
            pts = [Point(i, i * i) for i in range(n)]
        return gpd.GeoDataFrame({'tract_fips': [fips] * n}, geometry=pts)

    dl = _loader(graph_knn_k=10)
    monkeypatch.setattr(dl, 'get_addresses_for_tract', fake_addresses)
    with pytest.raises(GraphRandomizationError) as exc:
        dl.check_randomization(['A', 'B', 'C'], [42, 7])
    failing = {(r['tract_fips'], r['graph_draw_seed']) for r in exc.value.records}
    assert failing == {('B', 42), ('B', 7), ('C', 42), ('C', 7)}
    assert not hasattr(dl, '_swap_records')

    dl.config['allow_unswapped'] = ['B', 'C']
    with pytest.warns(UserWarning):
        recs = dl.check_randomization(['A', 'B', 'C'], [42, 7])
    assert len(recs) == 6
    assert all(r['randomized'] for r in recs if r['tract_fips'] == 'A')


def test_resume_round_trip(tmp_path, capsys):
    rs = _load_run_sweep()
    csv_path = tmp_path / 'trials_incremental.csv'
    seed_result = {'training_seed': 17, 'graph_draw_seed': 123, 'morans_i': 0.5,
                   'within_tract_std': 0.1, 'pooled_bg_r': 0.7, 'n_tracts': 20,
                   'runtime_s': 1.0, 'under_mixed_tracts': 1, 'n_unswapped_tracts': 1}
    rs._append_trial_csv(csv_path, 'randomized', 'gcn_gat', seed_result)
    done = rs._load_completed_trials(csv_path)
    row = done[('randomized', 'gcn_gat', 17, 123)]
    assert row['under_mixed_tracts'] == 1
    assert row['n_unswapped_tracts'] == 1
    assert 'MISSING' not in capsys.readouterr().out


def test_resume_missing_flags_reported_not_zero(tmp_path, capsys):
    rs = _load_run_sweep()
    csv_path = tmp_path / 'trials_incremental.csv'
    old = ['condition', 'arch', 'training_seed', 'graph_draw_seed', 'morans_i',
           'within_tract_std', 'pooled_bg_r', 'n_tracts', 'runtime_s']
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(old)
        w.writerow(['randomized', 'sage', 42, 7, 0.4, 0.1, 0.76, 20, 1.0])
    done = rs._load_completed_trials(csv_path)
    row = done[('randomized', 'sage', 42, 7)]
    assert row['under_mixed_tracts'] is None
    assert row['n_unswapped_tracts'] is None
    out = capsys.readouterr().out
    assert 'MISSING randomized/sage/42/7' in out
    # appending a new-schema row to a pre-guard csv is refused
    with pytest.raises(RuntimeError, match='predates SWAP-GUARD'):
        rs._append_trial_csv(csv_path, 'randomized', 'sage', {'training_seed': 1})


def test_same_seed_same_graph_and_record():
    G = nx.random_regular_graph(6, 40, seed=3)
    ei = _edge_index(G)
    dl1, dl2 = _loader(), _loader()
    out1, _ = dl1._randomize_graph(ei, None, 40, 2024, tract_fips='T')
    out2, _ = dl2._randomize_graph(ei, None, 40, 2024, tract_fips='T')
    assert torch.equal(out1, out2)
    assert dl1._last_swap_record == dl2._last_swap_record
    out3, _ = dl1._randomize_graph(ei, None, 40, 7, tract_fips='T')
    assert _undirected(out3) != _undirected(out1)
