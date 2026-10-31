#!/usr/bin/env python3
"""
rescore paper 1 pooled block-group r from persisted predictions against a
chosen block-group svi reference version (REF-FIX).

scoring goes through the exact helpers the producing scripts use:
  5b / crossed: run_sweep._aggregate_to_bg, run_sweep._bg_metrics,
                run_sweep.MIN_ADDRESSES_PER_BG (3), tract order from
                run_sweep._load_tract_list, npz predictions + EPSG:4326 coords
  m0:           granite.scripts.run_m0_parity helpers (pooled threshold 10,
                per-tract threshold 3, bootstrap n=1000 seed=42), predictions
                from experiments/recovery/per_address_predictions/*.parquet,
                coordinates from the 5b npz files (identical address order)

modes:
  --gate     score every committed 5b and crossed cell under v1 and compare to
             the committed pooled_bg_r (csv and json); fail above 1e-12.
             writes data/results/bg_reference_v2/v1_reproduction_gate.csv
  --rescore  score 5b, crossed and m0 under v1 and v2; writes the comparison
             files under data/results/bg_reference_v2/. frozen files untouched.

usage:
    python scripts/rescore_bg_reference.py --gate
    python scripts/rescore_bg_reference.py --rescore
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from granite.data.block_group_loader import BlockGroupLoader
from granite.validation.block_group_validation import BlockGroupValidator
from granite.scripts import run_m0_parity as m0

SWEEP_PATH = REPO_ROOT / 'experiments' / 'ablation' / '05b_topology_specificity' / 'run_sweep.py'
RESULTS = REPO_ROOT / 'data' / 'results'
SETS = {
    '5b': RESULTS / 'm6_topology_5b',
    'crossed': RESULTS / 'm6_topology_5b_crossed',
}
M0_PRED_DIR = REPO_ROOT / 'experiments' / 'recovery' / 'per_address_predictions'
M0_PARQUETS = {
    'GRANITE': 'granite_m0.parquet',
    'Dasymetric': 'dasymetric.parquet',
    'Pycnophylactic': 'pycnophylactic.parquet',
}
M0_FROZEN = RESULTS / 'm0_n20_svi_parity'
COORD_CELL = 'production__sage__ts42__gs42__'
OUT_DIR = RESULTS / 'bg_reference_v2'
GATE_TOL = 1e-12
VERSIONS = ('v1', 'v2')


def _load_sweep_module():
    spec = importlib.util.spec_from_file_location('run_sweep_05b', SWEEP_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_bg_gdf(reference_version):
    """run_sweep._load_bg_gdf with the reference version made explicit."""
    loader = BlockGroupLoader(data_dir=str(REPO_ROOT / 'data'), verbose=False)
    bg_gdf = loader.get_block_groups_with_demographics(
        '47', '065', svi_ranking_scope='national', reference_version=reference_version)
    if bg_gdf is None or len(bg_gdf) == 0:
        raise RuntimeError('BlockGroupLoader returned empty GeoDataFrame')
    if bg_gdf.crs is None:
        bg_gdf = bg_gdf.set_crs('EPSG:4326')
    elif bg_gdf.crs.to_epsg() != 4326:
        bg_gdf = bg_gdf.to_crs('EPSG:4326')
    return bg_gdf


def _points(coords):
    return gpd.GeoDataFrame(geometry=gpd.points_from_xy(coords[:, 0], coords[:, 1]),
                            crs='EPSG:4326')


def _committed_cells(set_name):
    """committed pooled_bg_r per cell from trials_incremental.csv and the metrics json."""
    base = SETS[set_name]
    trials = pd.read_csv(base / 'trials_incremental.csv')
    with open(base / 'topology_specificity_metrics.json') as f:
        metrics = json.load(f)
    json_vals = {}
    for cond, by_arch in metrics.items():
        for arch, block in by_arch.items():
            for s in block['seeds']:
                json_vals[(cond, arch, int(s['training_seed']), int(s['graph_draw_seed']))] = \
                    float(s['pooled_bg_r'])
    rows = []
    for r in trials.itertuples(index=False):
        key = (r.condition, r.arch, int(r.training_seed), int(r.graph_draw_seed))
        rows.append({
            'set': set_name, 'condition': key[0], 'arch': key[1],
            'training_seed': key[2], 'graph_draw_seed': key[3],
            'committed_csv': float(r.pooled_bg_r),
            'committed_json': json_vals.get(key, float('nan')),
        })
    if len(rows) != len(json_vals):
        raise RuntimeError(f'{set_name}: csv has {len(rows)} cells, json {len(json_vals)}')
    return pd.DataFrame(rows)


def score_cell(sweep, validator, bg_gdf, set_name, cell, tract_list):
    """pooled bg r for one persisted cell, via run_sweep's aggregation path."""
    pred_dir = SETS[set_name] / 'predictions'
    preds, coords = [], []
    for fips in tract_list:
        p = pred_dir / (f"{cell['condition']}__{cell['arch']}__ts{cell['training_seed']}"
                        f"__gs{cell['graph_draw_seed']}__{fips}.npz")
        z = np.load(p)
        preds.append(np.asarray(z['predictions'], dtype=float))
        coords.append(np.asarray(z['coords'], dtype=float))
    all_preds = np.concatenate(preds)
    all_addr = _points(np.concatenate(coords))
    bg_agg = sweep._aggregate_to_bg(validator, all_addr, all_preds, sweep.MIN_ADDRESSES_PER_BG)
    bm = sweep._bg_metrics(bg_agg, bg_gdf)
    return bm['bg_r'], bm['n_bgs']


def score_sets(sweep, refs, set_names):
    tract_list = sweep._load_tract_list()
    frames = []
    for set_name in set_names:
        cells = _committed_cells(set_name)
        for v in refs:
            bg_gdf, validator = refs[v]
            out = [score_cell(sweep, validator, bg_gdf, set_name, c, tract_list)
                   for c in cells.to_dict(orient='records')]
            cells[f'pooled_bg_r_{v}'] = [o[0] for o in out]
            cells[f'n_bgs_{v}'] = [o[1] for o in out]
        frames.append(cells)
    return pd.concat(frames, ignore_index=True)


def run_gate(sweep, refs):
    df = score_sets(sweep, {'v1': refs['v1']}, ['5b', 'crossed'])
    df['absdiff_csv'] = (df['pooled_bg_r_v1'] - df['committed_csv']).abs()
    df['absdiff_json'] = (df['pooled_bg_r_v1'] - df['committed_json']).abs()
    df['pass'] = (df['absdiff_csv'] <= GATE_TOL) & (df['absdiff_json'] <= GATE_TOL)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / 'v1_reproduction_gate.csv', index=False)
    for (s, cond), g in df.groupby(['set', 'condition']):
        print(f'gate {s:8s} {cond:22s} cells={len(g):3d} pass={int(g["pass"].sum()):3d} '
              f'max_absdiff={max(g["absdiff_csv"].max(), g["absdiff_json"].max()):.3e}')
    n_pass = int(df['pass'].sum())
    print(f'gate total: {n_pass}/{len(df)} cells within {GATE_TOL:g}; '
          f'max absdiff {max(df["absdiff_csv"].max(), df["absdiff_json"].max()):.3e}; '
          f'n_bgs_v1 {sorted(df["n_bgs_v1"].unique().tolist())}')
    return n_pass == len(df)


# ---------------------------------------------------------------------------
# m0
# ---------------------------------------------------------------------------

def _load_m0_inputs(tract_order):
    frames = {m: pd.read_parquet(M0_PRED_DIR / f) for m, f in M0_PARQUETS.items()}
    ref = frames['GRANITE'][['fips', 'address_idx']].reset_index(drop=True)
    for m, df in frames.items():
        if not df[['fips', 'address_idx']].reset_index(drop=True).equals(ref):
            raise RuntimeError(f'm0 parquet {m} not aligned with GRANITE')
    fips_order = list(dict.fromkeys(ref['fips'].tolist()))
    if sorted(fips_order) != sorted(tract_order):
        raise RuntimeError('m0 tracts differ from 5b tract list')
    per_tract = []
    for fips in fips_order:
        mask = (ref['fips'] == fips).values
        coords = np.load(SETS['5b'] / 'predictions' / f'{COORD_CELL}{fips}.npz')['coords']
        if len(coords) != int(mask.sum()):
            raise RuntimeError(f'{fips}: coord count mismatch')
        per_tract.append({
            'fips': fips,
            'gdf': _points(np.asarray(coords, dtype=float)),
            'preds': {m: frames[m].loc[mask, 'svi_pred'].values.astype(float) for m in frames},
        })
    return per_tract


def score_m0(refs, tract_order):
    per_tract_inputs = _load_m0_inputs(tract_order)
    agg_rows, diff_rows, per_tract_frames = [], [], {}
    for v, (bg_gdf, validator) in refs.items():
        rows = []
        all_gdfs = [t['gdf'] for t in per_tract_inputs]
        preds_by_method = {m: [t['preds'][m] for t in per_tract_inputs] for m in m0.METHODS}
        for t in per_tract_inputs:
            for method in m0.METHODS:
                bg_agg = m0._aggregate_to_bg(validator, t['gdf'], t['preds'][method],
                                             m0.MIN_ADDRESSES_PER_BG_PERTRACT)
                bm = m0._bg_metrics(bg_agg, bg_gdf)
                rows.append({'fips': t['fips'], 'method': method,
                             'n_addresses': len(t['gdf']), 'n_bgs': bm['n_bgs'],
                             'bg_r': bm['bg_r'], 'bg_rmse': bm['bg_rmse']})
        pt = pd.DataFrame(rows)
        per_tract_frames[v] = pt

        combined = pd.concat(all_gdfs, ignore_index=True)
        boot = m0._bootstrap_pooled_r_ci(all_gdfs, preds_by_method, validator, bg_gdf,
                                         n_boot=1000, seed=m0.SEED)
        for method in m0.METHODS:
            pooled = m0._bg_metrics(
                m0._aggregate_to_bg(validator, combined, np.concatenate(preds_by_method[method]),
                                    m0.MIN_ADDRESSES_PER_BG_POOLED), bg_gdf)
            med, lo, hi = m0._bootstrap_median_ci(pt.loc[pt.method == method, 'bg_r'].values,
                                                  seed=m0.SEED)
            agg_rows.append({
                'reference_version': v, 'method': method,
                'pooled_bg_r': pooled['bg_r'], 'pooled_bg_r_ci_low': boot[method][1],
                'pooled_bg_r_ci_high': boot[method][2], 'pooled_n_bgs': pooled['n_bgs'],
                'median_bg_r': med, 'median_ci_low': lo, 'median_ci_high': hi,
                'n_tracts_with_r': int(np.isfinite(pt.loc[pt.method == method, 'bg_r']).sum()),
            })
        for label, ma, mb in [('granite_vs_dasymetric', 'GRANITE', 'Dasymetric'),
                              ('granite_vs_pycno', 'GRANITE', 'Pycnophylactic'),
                              ('dasymetric_vs_pycno', 'Dasymetric', 'Pycnophylactic')]:
            a = pt[pt.method == ma][['fips', 'bg_r']].rename(columns={'bg_r': 'ra'})
            b = pt[pt.method == mb][['fips', 'bg_r']].rename(columns={'bg_r': 'rb'})
            paired = a.merge(b, on='fips').dropna(subset=['ra', 'rb'])
            obs, lo, hi = m0._bootstrap_diff_ci(paired['ra'].values, paired['rb'].values,
                                                seed=m0.SEED)
            diff_rows.append({'reference_version': v, 'pair': label, 'obs_median_diff': obs,
                              'ci_low_95': lo, 'ci_high_95': hi, 'n_pairs': len(paired),
                              'separable': bool(np.isfinite(lo) and np.isfinite(hi)
                                                and (lo > 0 or hi < 0))})
    return pd.DataFrame(agg_rows), pd.DataFrame(diff_rows), per_tract_frames


def run_rescore(sweep, refs):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cells = score_sets(sweep, refs, ['5b', 'crossed'])
    cells['diff_v2_minus_v1'] = cells['pooled_bg_r_v2'] - cells['pooled_bg_r_v1']
    cells.to_csv(OUT_DIR / 'rescored_cells.csv', index=False)

    means = (cells.groupby(['set', 'condition', 'arch'], sort=False)
             .agg(n_cells=('pooled_bg_r_v1', 'size'),
                  mean_v1=('pooled_bg_r_v1', 'mean'), mean_v2=('pooled_bg_r_v2', 'mean'),
                  sd_v1=('pooled_bg_r_v1', 'std'), sd_v2=('pooled_bg_r_v2', 'std'),
                  n_bgs_v1=('n_bgs_v1', 'max'), n_bgs_v2=('n_bgs_v2', 'max'))
             .reset_index())
    means['diff_v2_minus_v1'] = means['mean_v2'] - means['mean_v1']
    means.to_csv(OUT_DIR / 'rescored_condition_means.csv', index=False)
    print(means.to_string())
    for s, g in cells.groupby('set'):
        print(f'{s}: pooled_bg_r range v1 [{g.pooled_bg_r_v1.min():.4f}, '
              f'{g.pooled_bg_r_v1.max():.4f}] v2 [{g.pooled_bg_r_v2.min():.4f}, '
              f'{g.pooled_bg_r_v2.max():.4f}]')

    tract_order = sweep._load_tract_list()
    agg, diffs, per_tract = score_m0(refs, tract_order)
    frozen_agg = pd.read_csv(M0_FROZEN / 'aggregate.csv').set_index('method')
    agg['frozen_pooled_bg_r'] = agg['method'].map(frozen_agg['pooled_bg_r'])
    agg['frozen_median_bg_r'] = agg['method'].map(frozen_agg['median_bg_r'])
    agg.to_csv(OUT_DIR / 'm0_aggregate_v1_v2.csv', index=False)
    diffs.to_csv(OUT_DIR / 'm0_pairwise_diffs_v1_v2.csv', index=False)
    for v, pt in per_tract.items():
        pt.to_csv(OUT_DIR / f'm0_per_tract_{v}.csv', index=False)
    print(agg.to_string())
    print(diffs.to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--gate', action='store_true')
    mode.add_argument('--rescore', action='store_true')
    args = ap.parse_args()

    sweep = _load_sweep_module()
    refs = {}
    for v in VERSIONS:
        bg_gdf = load_bg_gdf(v)
        refs[v] = (bg_gdf, BlockGroupValidator(bg_gdf, verbose=False))

    if args.gate:
        ok = run_gate(sweep, refs)
        print('GATE PASS' if ok else 'GATE FAIL')
        sys.exit(0 if ok else 1)
    run_rescore(sweep, refs)


if __name__ == '__main__':
    main()
