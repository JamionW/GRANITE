#!/usr/bin/env python3
"""
tract-clustered bootstrap of paper 1 pooled block-group r differences on
block-group reference v2 (BOOT-P1).

margin and decision rule: docs/paper1/equivalence_margin_spec.md.

scoring reuses the producing helpers, unchanged:
  m0 (C1):      granite.scripts.run_m0_parity._aggregate_to_bg (pooled
                threshold 10) and the _bg_metrics join, inputs via
                scripts/rescore_bg_reference.py._load_m0_inputs
  5b (C2):      run_sweep._aggregate_to_bg (threshold 3) and the _bg_metrics
                join, npz predictions + coords as in rescore_bg_reference

each scoring bg lies in one tract (geoid prefix equals source tract, asserted).
a resample draws 20 tracts with replacement; pooled r is pearson r over every
bg of the drawn tracts, duplicated tracts contributing duplicated bgs. this is
computed as a count-weighted pearson r, identical to duplicating rows; the
unweighted case is asserted equal to the scorer's bg_r.

C1: m0 GRANITE minus Dasymetric (d_r, d_z = atanh difference), verdict per spec.
C2: structured spatial_knn_uniform (5 training seeds, graph draw 42) minus
    crossed randomized (5 seeds x 5 draws), per architecture. descriptive only.

usage:
    python scripts/paper1_bootstrap.py
writes data/results/paper1_bootstrap/bootstrap_results.json and
bootstrap_draws.npz.
"""
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from granite.validation.block_group_validation import BlockGroupValidator
from granite.scripts import run_m0_parity as m0

RESCORE_PATH = REPO_ROOT / 'scripts' / 'rescore_bg_reference.py'
OUT_DIR = REPO_ROOT / 'data' / 'results' / 'paper1_bootstrap'
REFERENCE_VERSION = 'v2'
B = 10_000
SEED = 20261109
MARGIN_Z = 0.10
ANCHOR_R = 0.797
SENS_MARGIN_R = 0.03
ARCHS = ('sage', 'gcn_gat')
STRUCTURED = ('5b', 'spatial_knn_uniform')
RANDOMIZED = ('crossed', 'randomized')
SEEDS = (42, 17, 123, 2024, 7)
DRAWS = (42, 17, 123, 2024, 7)

INPUT_PATHS = {
    'm0_predictions': 'experiments/recovery/per_address_predictions',
    'structured_5b_predictions': 'data/results/m6_topology_5b',
    'crossed_5b_predictions': 'data/results/m6_topology_5b_crossed',
    'm0_committed_v2': 'data/results/bg_reference_v2/m0_aggregate_v1_v2.csv',
    'cells_committed_v2': 'data/results/bg_reference_v2/rescored_cells.csv',
    'margin_spec': 'docs/paper1/equivalence_margin_spec.md',
    'm0_scorer': 'granite/scripts/run_m0_parity.py',
    'sweep_scorer': 'experiments/ablation/05b_topology_specificity/run_sweep.py',
    'rescore_script': 'scripts/rescore_bg_reference.py',
    'bg_loader': 'granite/data/block_group_loader.py',
}


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _git(*args):
    return subprocess.run(['git', *args], cwd=REPO_ROOT, capture_output=True,
                          text=True, check=True).stdout.strip()


def _scored_table(bg_agg, bg_gdf, tract_index):
    """the _bg_metrics join, kept per bg with its tract index."""
    svi_lookup = bg_gdf[['GEOID', 'SVI']].dropna(subset=['SVI'])
    merged = bg_agg.merge(svi_lookup, on='GEOID', how='inner')
    p = merged['predicted_svi'].values.astype(float)
    t = merged['SVI'].values.astype(float)
    valid = np.isfinite(p) & np.isfinite(t)
    geoid = merged['GEOID'].astype(str).values[valid]
    tract = np.array([tract_index[g[:11]] for g in geoid], dtype=np.int64)
    return {'geoid': geoid, 'p': p[valid], 't': t[valid], 'tract': tract}


def _weighted_r(table, counts):
    """pearson r per resample row of counts (resamples x tracts)."""
    w = counts[:, table['tract']].astype(float)
    n = w.sum(axis=1)
    mp = (w @ table['p']) / n
    mt = (w @ table['t']) / n
    dp = table['p'][None, :] - mp[:, None]
    dt = table['t'][None, :] - mt[:, None]
    sxy = (w * dp * dt).sum(axis=1)
    sxx = (w * dp * dp).sum(axis=1)
    syy = (w * dt * dt).sum(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        r = sxy / np.sqrt(sxx * syy)
    n_bgs = (w > 0).sum(axis=1)
    r[(n_bgs < 2) | (sxx < 1e-18) | (syy < 1e-18)] = np.nan
    return r


def _ci(x, level):
    a = (1.0 - level) / 2.0 * 100.0
    return [float(np.nanpercentile(x, a)), float(np.nanpercentile(x, 100.0 - a))]


def _summary(point, draws):
    return {
        'point': float(point),
        'ci90': _ci(draws, 0.90),
        'ci95': _ci(draws, 0.95),
        'bootstrap_sd': float(np.nanstd(draws, ddof=1)),
        'bootstrap_var': float(np.nanvar(draws, ddof=1)),
        'n_nan_resamples': int(np.isnan(draws).sum()),
    }


def _verdict(ci, lo, hi):
    if lo < ci[0] and ci[1] < hi:
        return 'equivalent'
    if ci[0] > 0 or ci[1] < 0:
        return 'difference'
    return 'small but inconclusive'


def _sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main():
    rescore = _load_module('rescore_bg_reference', RESCORE_PATH)
    sweep = rescore._load_sweep_module()
    bg_gdf = rescore.load_bg_gdf(REFERENCE_VERSION)
    validator = BlockGroupValidator(bg_gdf, verbose=False)
    tract_list = sweep._load_tract_list()
    tract_index = {f: i for i, f in enumerate(tract_list)}
    n_tracts = len(tract_list)

    # m0 per-bg tables
    m0_inputs = rescore._load_m0_inputs(tract_list)
    combined = pd.concat([t['gdf'] for t in m0_inputs], ignore_index=True)
    assigned = validator._assign_to_block_groups(combined)
    src = np.concatenate([[t['fips']] * len(t['gdf']) for t in m0_inputs])
    m0_tables = {}
    for method in ('GRANITE', 'Dasymetric'):
        preds = np.concatenate([t['preds'][method] for t in m0_inputs])
        bg_agg = m0._aggregate_to_bg(validator, combined, preds, m0.MIN_ADDRESSES_PER_BG_POOLED)
        tab = _scored_table(bg_agg, bg_gdf, tract_index)
        scorer = m0._bg_metrics(bg_agg, bg_gdf)
        r_full = _weighted_r(tab, np.ones((1, n_tracts), dtype=np.int64))[0]
        assert len(tab['p']) == scorer['n_bgs'], method
        assert abs(r_full - scorer['bg_r']) <= 1e-12, (method, r_full, scorer['bg_r'])
        tab['scorer_r'] = scorer['bg_r']
        m0_tables[method] = tab
    scoring = set(m0_tables['GRANITE']['geoid'])
    assert scoring == set(m0_tables['Dasymetric']['geoid'])
    in_scoring = assigned['block_group_id'].isin(scoring).values
    assert (assigned['block_group_id'].values[in_scoring].astype(str).astype('U11')
            == src[in_scoring]).all(), 'scoring bg spans tracts'
    n_bgs = len(scoring)
    bgs_per_tract = {f: int((m0_tables['GRANITE']['tract'] == i).sum())
                     for f, i in tract_index.items()}

    # 5b per-bg tables
    def cell_table(set_name, condition, arch, ts, gs):
        cell = {'condition': condition, 'arch': arch, 'training_seed': ts, 'graph_draw_seed': gs}
        pred_dir = rescore.SETS[set_name] / 'predictions'
        preds, coords = [], []
        for fips in tract_list:
            z = np.load(pred_dir / f'{condition}__{arch}__ts{ts}__gs{gs}__{fips}.npz')
            preds.append(np.asarray(z['predictions'], dtype=float))
            coords.append(np.asarray(z['coords'], dtype=float))
        bg_agg = sweep._aggregate_to_bg(validator, rescore._points(np.concatenate(coords)),
                                        np.concatenate(preds), sweep.MIN_ADDRESSES_PER_BG)
        tab = _scored_table(bg_agg, bg_gdf, tract_index)
        scorer = sweep._bg_metrics(bg_agg, bg_gdf)
        r_full = _weighted_r(tab, np.ones((1, n_tracts), dtype=np.int64))[0]
        assert abs(r_full - scorer['bg_r']) <= 1e-12, (cell, r_full, scorer['bg_r'])
        assert set(tab['geoid']) == scoring, cell
        tab['scorer_r'] = scorer['bg_r']
        return tab

    committed_cells = pd.read_csv(REPO_ROOT / INPUT_PATHS['cells_committed_v2'])
    committed_m0 = pd.read_csv(REPO_ROOT / INPUT_PATHS['m0_committed_v2']).query(
        'reference_version == @REFERENCE_VERSION').set_index('method')['pooled_bg_r']
    for method, tab in m0_tables.items():
        assert abs(tab['scorer_r'] - committed_m0[method]) <= 1e-9, method

    # resamples, shared by every contrast
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n_tracts, size=(B, n_tracts))
    counts = np.zeros((B, n_tracts), dtype=np.int64)
    np.add.at(counts, (np.repeat(np.arange(B), n_tracts), idx.ravel()), 1)

    # C1
    r_g = _weighted_r(m0_tables['GRANITE'], counts)
    r_d = _weighted_r(m0_tables['Dasymetric'], counts)
    pg, pd_ = m0_tables['GRANITE']['scorer_r'], m0_tables['Dasymetric']['scorer_r']
    c1_dr = r_g - r_d
    c1_dz = np.arctanh(r_g) - np.arctanh(r_d)
    lo_r = float(np.tanh(np.arctanh(ANCHOR_R) - MARGIN_Z) - ANCHOR_R)
    hi_r = float(np.tanh(np.arctanh(ANCHOR_R) + MARGIN_Z) - ANCHOR_R)
    c1 = {
        'r_GRANITE': _summary(pg, r_g),
        'r_Dasymetric': _summary(pd_, r_d),
        'd_r': _summary(pg - pd_, c1_dr),
        'd_z': _summary(np.arctanh(pg) - np.arctanh(pd_), c1_dz),
    }
    c1['verdict'] = _verdict(c1['d_z']['ci90'], -MARGIN_Z, MARGIN_Z)
    c1['verdict_rule'] = '90% CI of d_z against (-0.10, +0.10), per equivalence_margin_spec.md'
    c1['d_r_ci90_within_r_scale_bounds'] = bool(lo_r < c1['d_r']['ci90'][0]
                                                and c1['d_r']['ci90'][1] < hi_r)
    c1['sensitivity_verdict'] = _verdict(c1['d_r']['ci90'], -SENS_MARGIN_R, SENS_MARGIN_R)
    c1['sensitivity_rule'] = '90% CI of d_r against (-0.03, +0.03)'
    c1['n_training_seeds'] = 1
    c1['training_seed'] = 42
    c1['seed_component'] = None
    c1['seed_component_note'] = 'm0 GRANITE carries one training seed; C1 has no seed component'
    draws = {'tract_index': idx.astype(np.int16), 'c1_r_granite': r_g, 'c1_r_dasymetric': r_d,
             'c1_d_r': c1_dr, 'c1_d_z': c1_dz}

    # C2
    c2 = {}
    for arch in ARCHS:
        s_tabs = {ts: cell_table(*STRUCTURED, arch, ts, 42) for ts in SEEDS}
        r_tabs = {(ts, gs): cell_table(*RANDOMIZED, arch, ts, gs) for ts in SEEDS for gs in DRAWS}
        for (set_name, cond), tabs in ((STRUCTURED, s_tabs), (RANDOMIZED, r_tabs)):
            for key, tab in tabs.items():
                ts, gs = key if isinstance(key, tuple) else (key, 42)
                row = committed_cells[(committed_cells.set == set_name)
                                      & (committed_cells.condition == cond)
                                      & (committed_cells.arch == arch)
                                      & (committed_cells.training_seed == ts)
                                      & (committed_cells.graph_draw_seed == gs)]
                assert len(row) == 1
                assert abs(tab['scorer_r'] - float(row.pooled_bg_r_v2.iloc[0])) <= 1e-9
        s_pt = np.array([s_tabs[ts]['scorer_r'] for ts in SEEDS])
        r_pt = np.array([[r_tabs[(ts, gs)]['scorer_r'] for gs in DRAWS] for ts in SEEDS])
        s_bs = np.stack([_weighted_r(s_tabs[ts], counts) for ts in SEEDS])
        r_bs = np.stack([np.stack([_weighted_r(r_tabs[(ts, gs)], counts) for gs in DRAWS])
                         for ts in SEEDS])

        dr = s_bs.mean(axis=0) - r_bs.mean(axis=(0, 1))
        dz = np.arctanh(s_bs).mean(axis=0) - np.arctanh(r_bs).mean(axis=(0, 1))
        dr_pt = s_pt.mean() - r_pt.mean()
        dz_pt = np.arctanh(s_pt).mean() - np.arctanh(r_pt).mean()

        # tract component at fixed seed 42, draw 42
        i42 = SEEDS.index(42)
        j42 = DRAWS.index(42)
        fx_dr = s_bs[i42] - r_bs[i42, j42]
        fx_dz = np.arctanh(s_bs[i42]) - np.arctanh(r_bs[i42, j42])
        fx_dr_pt = s_pt[i42] - r_pt[i42, j42]
        fx_dz_pt = np.arctanh(s_pt[i42]) - np.arctanh(r_pt[i42, j42])

        def seed_draw(sv, rv):
            # sv: structured per-seed values; rv: randomized seeds x draws
            per_seed_d = sv - rv.mean(axis=1)
            return {
                'training_seed': {
                    'difference_var': float(np.var(per_seed_d, ddof=1)),
                    'structured_var': float(np.var(sv, ddof=1)),
                    'randomized_var': float(np.var(rv.mean(axis=1), ddof=1)),
                    'per_seed_difference': [float(x) for x in per_seed_d],
                },
                'graph_draw': {
                    'difference_var': float(np.var(rv.mean(axis=0), ddof=1)),
                    'structured_var': None,
                    'randomized_var': float(np.var(rv.mean(axis=0), ddof=1)),
                    'per_draw_randomized_mean': [float(x) for x in rv.mean(axis=0)],
                    'note': 'structured arm has one graph draw (42); the difference inherits '
                            'the randomized per-draw variance',
                },
            }

        c2[arch] = {
            'structured_condition': STRUCTURED[1],
            'randomized_condition': 'randomized (crossed set)',
            'r_structured_mean': float(s_pt.mean()),
            'r_randomized_mean': float(r_pt.mean()),
            'd_r': _summary(dr_pt, dr),
            'd_z': _summary(dz_pt, dz),
            'variance_components': {
                'tract': {
                    'cell': 'training_seed 42, graph_draw 42',
                    'd_r': _summary(fx_dr_pt, fx_dr),
                    'd_z': _summary(fx_dz_pt, fx_dz),
                },
                'r_scale': seed_draw(s_pt, r_pt),
                'z_scale': seed_draw(np.arctanh(s_pt), np.arctanh(r_pt)),
                'estimator': 'tract: bootstrap variance (ddof=1) at the fixed cell; '
                             'training_seed: variance (ddof=1) of the 5 per-seed values, '
                             'randomized per-seed values averaged over draws; graph_draw: '
                             'variance (ddof=1) of the 5 per-draw randomized means averaged '
                             'over seeds. components are not pooled.',
            },
            'verdict': None,
            'verdict_note': 'descriptive only; the approved margin covers C1',
        }
        draws[f'c2_{arch}_d_r'] = dr
        draws[f'c2_{arch}_d_z'] = dz
        draws[f'c2_{arch}_fixed_cell_d_r'] = fx_dr
        draws[f'c2_{arch}_fixed_cell_d_z'] = fx_dz

    import geopandas, scipy
    results = {
        'milestone': 'BOOT-P1',
        'reference_version': REFERENCE_VERSION,
        'n_tracts': n_tracts,
        'n_bgs': n_bgs,
        'bgs_per_tract': bgs_per_tract,
        'B': B,
        'seed': SEED,
        'resampling': 'tracts with replacement, numpy default_rng(seed).integers(0, n_tracts, '
                      'size=(B, n_tracts)); pooled r over all bgs of the drawn tracts with '
                      'duplicates; percentile CIs via numpy nanpercentile (linear)',
        'margin': {
            'q_fisher_z': MARGIN_Z,
            'anchor_r': ANCHOR_R,
            'r_scale_lower': lo_r,
            'r_scale_upper': hi_r,
            'sensitivity_margin_r': SENS_MARGIN_R,
            'spec': 'docs/paper1/equivalence_margin_spec.md',
        },
        'C1': c1,
        'C2': c2,
        'draws_sha256': {k: _sha(v) for k, v in draws.items()},
        'git_head': _git('rev-parse', 'HEAD'),
        'input_commits': {k: _git('log', '-1', '--format=%H', '--', p)
                          for k, p in INPUT_PATHS.items()},
        'versions': {
            'python': platform.python_version(),
            'numpy': np.__version__,
            'pandas': pd.__version__,
            'geopandas': geopandas.__version__,
            'scipy': scipy.__version__,
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / 'bootstrap_results.json', 'w') as f:
        json.dump(results, f, indent=2)
        f.write('\n')
    np.savez(OUT_DIR / 'bootstrap_draws.npz', **draws)

    print(f'n_tracts {n_tracts} n_bgs {n_bgs} B {B} seed {SEED}')
    for k in ('r_GRANITE', 'r_Dasymetric', 'd_r', 'd_z'):
        s = c1[k]
        print(f'C1 {k:13s} point {s["point"]:.6f} ci90 [{s["ci90"][0]:.6f}, {s["ci90"][1]:.6f}] '
              f'ci95 [{s["ci95"][0]:.6f}, {s["ci95"][1]:.6f}]')
    print(f'C1 verdict: {c1["verdict"]}; sensitivity: {c1["sensitivity_verdict"]}')
    for arch, c in c2.items():
        for k in ('d_r', 'd_z'):
            s = c[k]
            print(f'C2 {arch:7s} {k} point {s["point"]:.6f} ci90 [{s["ci90"][0]:.6f}, '
                  f'{s["ci90"][1]:.6f}] ci95 [{s["ci95"][0]:.6f}, {s["ci95"][1]:.6f}]')
    print(f'wrote {OUT_DIR / "bootstrap_results.json"} and bootstrap_draws.npz')


if __name__ == '__main__':
    main()
