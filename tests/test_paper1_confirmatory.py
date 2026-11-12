"""end-to-end tests for scripts/paper1_confirmatory.py on synthetic run sets.

run sets are written under tmp_path in the on-disk layout of the committed
sets (<cell dir>/predictions/{condition}__{arch}__ts{ts}__gs{gs}__{fips}.npz
with predictions and coords), with planted effects. nothing is read from the
repository's results tree. the script runs in a subprocess because it resets
the process warning filters at import.
"""
import ast
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from granite.evaluation.baselines import DasymetricDisaggregation

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / 'scripts' / 'paper1_confirmatory.py'
TS = (42, 17, 123, 2024, 7)
GS = (42, 17, 123, 2024, 7)
DIAG = [(i, i) for i in TS]
UNSWAPPED = '47065011324'
NAMED = ('47065001900', '47065010431')
TRACTS = [UNSWAPPED, *NAMED] + [f'4706590{i:04d}' for i in range(17)]
DRAWS, NPERM = 200, 60


# ---------------------------------------------------------------------------
# synthetic run set
# ---------------------------------------------------------------------------

def _w_matrix(coords, k=8):
    import libpysal.weights as lw
    w = lw.KNN.from_array(coords, k=k)
    w.transform = 'r'
    return w.sparse


def _moran(W, v):
    z = v - v.mean()
    return float(z @ (W @ z) / (z @ z))


def _calibrate(W, base, noise, target):
    """lam with moran(base + lam * noise) == target (bisection)."""
    lo, hi = 0.0, 1.0
    while _moran(W, base + hi * noise) > target:
        hi *= 2
        if hi > 1e6:
            raise RuntimeError('cannot reach target')
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _moran(W, base + mid * noise) > target else (lo, mid)
    return base + hi * noise


def _save(root, cell_dir, cond, arch, pairs, maps, coords):
    d = root / cell_dir / 'predictions'
    d.mkdir(parents=True, exist_ok=True)
    for ts, gs in pairs:
        for f in TRACTS:
            np.savez(d / f'{cond}__{arch}__ts{ts}__gs{gs}__{f}.npz',
                     predictions=maps[f], coords=coords[f])


def build_run_set(root, *, cost, h2_gap, interaction, with_no_graph, seed):
    """cost: {'sage': c, 'gcn_gat': c} planted randomization cost per tract;
    h2_gap: planted q of sage euclidean against dasymetric (0 means identical maps);
    interaction: planted self-path interaction for H3."""
    import geopandas as gpd
    from shapely.geometry import box
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    (root / 'tracts.txt').write_text('\n'.join(TRACTS) + '\n')
    coords, feats, polys = {}, [], []
    for t, f in enumerate(TRACTS):
        x0, y0 = -85.3 + (t % 5) * 0.012, 35.0 + (t // 5) * 0.012
        pts = []
        for b, (dx, dy) in enumerate([(0, 0), (0.005, 0), (0, 0.005), (0.005, 0.005)]):
            bx, by = x0 + dx, y0 + dy
            polys.append({'GEOID': f'{f}{b + 1}', 'geometry': box(bx, by, bx + 0.005, by + 0.005)})
            pts.append(np.column_stack([rng.uniform(bx + 1e-4, bx + 0.0049, 10),
                                        rng.uniform(by + 1e-4, by + 0.0049, 10)]))
        coords[f] = np.vstack(pts)
        c = coords[f]
        imp = 50 + 30 * np.sin((c[:, 0] - x0) * 300) + 30 * np.cos((c[:, 1] - y0) * 300)
        svi = float(rng.uniform(0.3, 0.7))
        feats.append(pd.DataFrame({'fips': f, 'address_idx': np.arange(len(c)), 'tract_svi': svi,
                                   'nlcd_impervious_pct': imp + rng.normal(0, 1, len(c))}))
    fm = pd.concat(feats, ignore_index=True)
    fm.to_csv(root / 'features.csv', index=False)
    dasy = DasymetricDisaggregation(ancillary_column='nlcd_impervious_pct')
    D = {}
    for f in TRACTS:
        sub = fm[fm.fips == f].reset_index(drop=True)
        D[f] = dasy.disaggregate(coords[f], f, float(sub.tract_svi.iloc[0]), address_gdf=sub)
    bgid = {f: np.repeat([f'{f}{b}' for b in range(1, 5)], 10) for f in TRACTS}
    # reference: block-group mean of dasymetric plus noise
    ref = {}
    for f in TRACTS:
        for b in range(1, 5):
            ref[f'{f}{b}'] = float(D[f][bgid[f] == f'{f}{b}'].mean() + rng.normal(0, 0.03))
    bg = gpd.GeoDataFrame(polys, crs='EPSG:4326')
    bg['SVI'] = bg['GEOID'].map(ref)
    bg.to_file(root / 'block_groups.geojson', driver='GeoJSON')
    (root / 'swap_rebuild.json').write_text(json.dumps(
        {'rows': [{'tract': f, 'seed': s, 'randomized': True} for f in TRACTS for s in GS]}))

    W = {f: _w_matrix(coords[f]) for f in TRACTS}
    # high-pass noise (negative autocorrelation) so low coherence targets are reachable
    noise = {f: (lambda n: n - W[f] @ n)(rng.normal(0, 1, len(coords[f]))) for f in TRACTS}
    bg_noise = {f: np.repeat(rng.normal(0, 1, 4), 10) for f in TRACTS}

    # sage euclidean: dasymetric, or dasymetric plus block-group noise at the planted gap
    def model_with_gap(gap):
        if gap == 0:
            return {f: D[f].copy() for f in TRACTS}
        from granite.evaluation import paper1_metrics as pm
        t = np.array([ref[g] for f in TRACTS for g in dict.fromkeys(bgid[f])])

        def bg_means(m):
            return np.concatenate([[m[f][bgid[f] == g].mean() for g in dict.fromkeys(bgid[f])]
                                   for f in TRACTS])
        r_d = pm.correlation(bg_means(D), t, 'pearson')
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            m = {f: D[f] + mid * bg_noise[f] for f in TRACTS}
            q = np.arctanh(pm.correlation(bg_means(m), t, 'pearson')) - np.arctanh(r_d)
            lo, hi = (mid, hi) if q > -gap else (lo, mid)
        return {f: D[f] + hi * bg_noise[f] for f in TRACTS}

    E = {'sage': model_with_gap(h2_gap),
         'gcn_gat': {f: 0.9 * D[f] + 0.05 for f in TRACTS}}
    for a in ('sage', 'gcn_gat'):
        e = E[a]
        rnd = {f: _calibrate(W[f], e[f], noise[f], _moran(W[f], e[f]) - cost[a]) for f in TRACTS}
        _save(root, 'm6_topology_5b', 'spatial_knn_uniform', a, [(t, 42) for t in TS], e, coords)
        _save(root, 'm6_topology_5b', 'production', a, [(t, 42) for t in TS], e, coords)
        _save(root, 'm6_topology_5b_crossed', 'randomized', a, [(t, g) for t in TS for g in GS],
              rnd, coords)
        if a == 'gcn_gat':
            # road network for gcn_gat: one within-tract permutation of the euclidean map
            prng = np.random.default_rng(seed + 1)
            road = {f: prng.permutation(e[f]) for f in TRACTS}
        else:
            road = e
        _save(root, 'm6_topology_5b', 'road_network_uniform', a, [(t, 42) for t in TS], road, coords)
        if with_no_graph:
            ng = {f: _calibrate(W[f], e[f], noise[f], _moran(W[f], e[f]) - 0.3) for f in TRACTS}
            _save(root, 'paper1_no_graph_rerun', 'mlp_floor', a, [(t, 42) for t in TS], ng, coords)
    # self path off: same euclidean map, randomization cost raised by the interaction
    e = E['sage']
    off = {f: _calibrate(W[f], e[f], noise[f], _moran(W[f], e[f]) - cost['sage'] - interaction)
           for f in TRACTS}
    _save(root, 'paper1_h3_sage_root_weight_false', 'spatial_knn_uniform', 'sage',
          [(t, 42) for t in TS], e, coords)
    _save(root, 'paper1_h3_sage_root_weight_false', 'randomized', 'sage', DIAG, off, coords)
    # one tract whose randomized edge list equals its input, recorded the post-guard way
    pd.DataFrame([{'condition': 'randomized', 'arch': a, 'training_seed': t, 'graph_draw_seed': g,
                   'tract_fips': f, 'randomized': f != UNSWAPPED}
                  for a in ('sage', 'gcn_gat') for t in TS for g in GS for f in TRACTS]
                 ).to_csv(root / 'm6_topology_5b_crossed' / 'swap_provenance.csv', index=False)
    return root


def run_script(run_set, out):
    r = subprocess.run([sys.executable, str(SCRIPT), '--synthetic', str(run_set), '--out', str(out),
                        '--draws', str(DRAWS), '--n-perm', str(NPERM)],
                       cwd=str(REPO), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    return out / 'confirmatory.json'


@pytest.fixture(scope='module')
def set_a(tmp_path_factory):
    """H1 sage cost 0.5 (pass), gcn_gat 0.1 (fail); H2 model equals dasymetric; H3 none;
    no no-graph cell; no residual cells."""
    root = build_run_set(tmp_path_factory.mktemp('a') / 'runs', cost={'sage': 0.5, 'gcn_gat': 0.1},
                         h2_gap=0.0, interaction=0.0, with_no_graph=False, seed=1)
    path = run_script(root, root.parent / 'out')
    return json.loads(path.read_text())


@pytest.fixture(scope='module')
def set_b(tmp_path_factory):
    """H1 sage cost 0.1; H2 planted gap 0.3 in z; H3 interaction 0.5; no-graph cell present.
    run twice for determinism."""
    root = build_run_set(tmp_path_factory.mktemp('b') / 'runs', cost={'sage': 0.1, 'gcn_gat': 0.3},
                         h2_gap=0.3, interaction=0.5, with_no_graph=True, seed=2)
    p1 = run_script(root, root.parent / 'out1')
    p2 = run_script(root, root.parent / 'out2')
    return json.loads(p1.read_text()), p1.read_bytes(), p2.read_bytes()


# ---------------------------------------------------------------------------
# hypotheses
# ---------------------------------------------------------------------------

def test_h1_planted_cost(set_a, set_b):
    a = set_a['analyses']
    assert abs(a['H1/sage']['estimate'] - 0.5) < 1e-6
    assert a['H1/sage']['verdicts'] == {'bootstrap': 'pass', 'jackknife': 'pass', 'agree': True}
    assert abs(a['H1/gcn_gat']['estimate'] - 0.1) < 1e-6
    assert a['H1/gcn_gat']['verdicts']['bootstrap'] == 'fail'
    assert a['H1/gcn_gat']['verdicts']['jackknife'] == 'fail'
    assert set_b[0]['analyses']['H1/sage']['verdicts']['bootstrap'] == 'fail'
    assert a['H1/sage']['bootstrap']['n_draws'] == DRAWS
    assert a['H1/sage']['bootstrap']['seed'] == 20261113


def test_h2_equal_and_gap(set_a, set_b):
    eq = set_a['analyses']['H2']
    assert abs(eq['estimate']['q_T']) < 1e-12 and abs(eq['estimate']['d_r_T']) < 1e-12
    assert eq['tost']['bootstrap']['verdict'] == 'equivalent'
    assert eq['tost']['jackknife']['verdict'] == 'equivalent'
    assert eq['verdicts']['agree'] is True
    gap = set_b[0]['analyses']['H2']
    assert abs(gap['estimate']['q_T'] + 0.3) < 1e-6
    assert gap['tost']['bootstrap']['verdict'] == 'different'
    assert gap['tost']['bootstrap']['tightest_margin'] > 0.1
    assert gap['bootstrap']['ci90']['d_r_T'][1] < 0   # difference in r reported beside q


def test_h3_planted_interaction(set_a, set_b):
    none = set_a['analyses']['H3']
    assert abs(none['estimate']) < 1e-6 and none['verdicts']['bootstrap'] == 'fail'
    yes = set_b[0]['analyses']['H3']
    assert abs(yes['estimate'] - 0.5) < 1e-6
    assert yes['verdicts'] == {'bootstrap': 'pass', 'jackknife': 'pass', 'agree': True}
    assert yes['pairs'] == [list(p) for p in DIAG]


# ---------------------------------------------------------------------------
# exclusion, sensitivity, supplement
# ---------------------------------------------------------------------------

def test_exclusion_from_swap_provenance(set_a):
    assert set_a['excluded_tracts'] == [UNSWAPPED]
    assert set_a['exclusion_sources'][UNSWAPPED] == ['m6_topology_5b_crossed/swap_provenance.csv']
    a = set_a['analyses']
    for name in ('H1/sage', 'H1/gcn_gat', 'H3', 'S3/sage', 'S3/gcn_gat'):
        assert UNSWAPPED not in a[name]['tracts'] and a[name]['n_tracts'] == 19
    assert UNSWAPPED in a['H2']['tracts'] and a['H2']['n_tracts'] == 20
    for name in ('supplement/H1/sage', 'supplement/H3', 'supplement/S3/gcn_gat'):
        assert a[name]['n_tracts'] == 20 and UNSWAPPED in a[name]['label']


def test_sensitivity_drops_named_tracts(set_a):
    a = set_a['analyses']
    s1 = a['sensitivity/without_47065001900/H1/sage']['tracts']
    assert NAMED[0] not in s1 and NAMED[1] in s1 and UNSWAPPED not in s1 and len(s1) == 18
    s2 = a['sensitivity/without_47065001900_47065010431/H1/sage']['tracts']
    assert not set(NAMED) & set(s2) and len(s2) == 17
    h = a['sensitivity/without_47065001900/H2']['tracts']
    assert NAMED[0] not in h and UNSWAPPED in h and len(h) == 19
    h2 = a['sensitivity/without_47065001900_47065010431/H2']['tracts']
    assert not set(NAMED) & set(h2) and len(h2) == 18


# ---------------------------------------------------------------------------
# secondary
# ---------------------------------------------------------------------------

def test_s2_permutation_null(set_a):
    s2 = set_a['analyses']['S2']
    assert s2['n_perm'] == NPERM
    # planted within-tract signal: sage euclidean maps are the dasymetric map
    assert s2['per_cell']['sage/euclidean'] == {'n_runs': 5, 'n_exceed': 5}
    assert s2['per_cell']['dasymetric'] == {'n_runs': 1, 'n_exceed': 1}
    # within-tract permuted map: gcn_gat road network
    assert s2['per_cell']['gcn_gat/road_network']['n_exceed'] == 0
    run = s2['per_run']['gcn_gat/road_network/ts42/gs42']
    assert run['observed_r_W'] <= run['p95'] and run['exceeds'] is False


def test_s1_s3_s5_present(set_b):
    a = set_b[0]['analyses']
    assert a['S1']['accuracy'] == 'r_W' and a['S1']['tost']['bootstrap']['verdict'] in (
        'equivalent', 'different', 'inconclusive')
    assert set(a['S3/sage']['components']) == {'q_T', 'd_r_T', 'q_W', 'd_r_W'}
    ng = a['S5/sage/no_graph_minus_randomized']
    assert ng['n_tracts'] == 19
    assert abs(a['S5/sage/euclidean_minus_no_graph']['estimate']['coherence_difference'] - 0.3) < 1e-6
    assert a['S5/sage/euclidean_minus_no_graph']['n_tracts'] == 20


def test_pending_cells_skipped(set_a):
    assert 'sage/no_graph' in set_a['pending_cells']
    assert 'gcn_gat_residual/euclidean' in set_a['pending_cells']
    skipped = {s['analysis']: s['pending_cells'] for s in set_a['skipped_analyses']}
    assert skipped['S4'] == ['gcn_gat_residual/euclidean', 'gcn_gat_residual/randomized']
    assert 'S5/sage/no_graph_minus_randomized' in skipped
    assert 'S4' not in set_a['analyses']
    assert 'S5/sage/road_network_minus_euclidean' in set_a['analyses']


def test_determinism(set_b):
    _, b1, b2 = set_b
    assert b1 == b2


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------

def _guard(repo):
    code = ('import sys; sys.path.insert(0, %r); import importlib.util as u; '
            's = u.spec_from_file_location("c", %r); m = u.module_from_spec(s); '
            's.loader.exec_module(m); print(m.confirmatory_guard(%r))') % (
        str(REPO), str(SCRIPT), str(repo))
    r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, cwd=str(REPO))
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout.strip().splitlines()[-1]


def test_confirmatory_refuses_without_tag(tmp_path):
    g = ['git', '-C', str(tmp_path)]
    subprocess.run([*g, 'init', '-q'], check=True)
    (tmp_path / 'f.txt').write_text('x')
    subprocess.run([*g, 'add', 'f.txt'], check=True)
    subprocess.run([*g, '-c', 'user.email=t@t', '-c', 'user.name=t', 'commit', '-qm', 'c'], check=True)
    out = _guard(tmp_path)
    assert out.startswith('(False') and 'not an ancestor' in out
    subprocess.run([*g, '-c', 'user.email=t@t', '-c', 'user.name=t', 'tag', '-a',
                    'paper1-prereg-v1', '-m', 't'], check=True)
    assert _guard(tmp_path).startswith('(True')
    (tmp_path / 'dirty.txt').write_text('y')
    assert 'not clean' in _guard(tmp_path)


def test_mode_flags_refused():
    r = subprocess.run([sys.executable, str(SCRIPT), '--dry-run-real', '--draws', '10'],
                       capture_output=True, text=True, cwd=str(REPO))
    assert r.returncode != 0 and 'synthetic only' in r.stderr


def test_dry_run_source_calls_no_statistic():
    tree = ast.parse(SCRIPT.read_text())
    funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    listed = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                  and getattr(n.targets[0], 'id', None) == 'DRY_RUN_FUNCTIONS')
    assert 'dry_run_real' in listed
    # every module-level function the dry run calls is in the list
    called = {c.func.id for c in ast.walk(funcs['dry_run_real'])
              if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
    assert called & set(funcs) <= set(listed) | {'real_inputs'}
    forbidden = ('pooled_r', 'within_tract_r', 'tract_bootstrap', 'tract_jackknife',
                 'within_tract_permutation_null', 'correlation', 'morans_i', 'bg_predictions',
                 'fisher_z_mean', 'q_difference', 'run_coherence', 'Analyses', 'RunData', 'BGTable')
    for name in listed:
        src = ast.get_source_segment(SCRIPT.read_text(), funcs[name])
        for word in forbidden:
            assert word not in src, (name, word)
