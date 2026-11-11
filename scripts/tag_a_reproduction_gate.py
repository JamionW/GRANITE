#!/usr/bin/env python3
"""
TAG-A reproduction gate: the paper 1 metric module against committed values.

scores only quantities already seen: per-tract output moran's i, run
coherence (unweighted tract mean), and pearson pooled block-group r_T on
reference v2, with the scored block-group count.

targets:
  5b and crossed, every committed cell: coherence vs trials_incremental.csv
    and topology_specificity_metrics.json; pooled r_T and n vs
    bg_reference_v2/rescored_cells.csv (REF-FIX).
  fig 4 randomization, every committed tract row: per-tract moran's i vs
    m6_fig4_randomization/tract_rows.csv.
  m0: pooled r_T and n for GRANITE, Dasymetric and Pycnophylactic vs
    bg_reference_v2/m0_aggregate_v1_v2.csv; Dasymetric also recomputed from
    granite/evaluation/baselines.py on the committed n20 feature matrix.

inputs come through the loaders of scripts/rescore_bg_reference.py (npz by
tract order with EPSG:4326 coords; m0 parquets aligned by fips and
address_idx to the 5b coords) and the block-group join of
BlockGroupValidator._assign_to_block_groups.

tolerance 1e-12. exits 1 if any target fails. writes
data/results/paper1_metrics_gate/reproduction_gate.json.

usage:
    python scripts/tag_a_reproduction_gate.py
"""
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

from granite.evaluation.baselines import DasymetricDisaggregation
from granite.evaluation.paper1_metrics import (bg_predictions, per_tract_morans_i, pooled_r,
                                               run_coherence)
from granite.validation.block_group_validation import BlockGroupValidator

RESCORE_PATH = REPO_ROOT / 'scripts' / 'rescore_bg_reference.py'
RESULTS = REPO_ROOT / 'data' / 'results'
OUT_PATH = RESULTS / 'paper1_metrics_gate' / 'reproduction_gate.json'
TOL = 1e-12
REFERENCE_VERSION = 'v2'
EXPECTED_BGS = 67

SRC = {
    '5b_trials': 'data/results/m6_topology_5b/trials_incremental.csv',
    '5b_metrics': 'data/results/m6_topology_5b/topology_specificity_metrics.json',
    'crossed_trials': 'data/results/m6_topology_5b_crossed/trials_incremental.csv',
    'crossed_metrics': 'data/results/m6_topology_5b_crossed/topology_specificity_metrics.json',
    'rescored_cells': 'data/results/bg_reference_v2/rescored_cells.csv',
    'm0_aggregate': 'data/results/bg_reference_v2/m0_aggregate_v1_v2.csv',
    'fig4_tract_rows': 'data/results/m6_fig4_randomization/tract_rows.csv',
    'm0_parquet_dasymetric': 'experiments/recovery/per_address_predictions/dasymetric.parquet',
    'n20_feature_matrix': 'experiments/ecological_fallacy/n20_feature_matrix.csv',
    'number_trace': 'docs/paper1/number_trace.md',
}
# number_trace.md, Reference v2 (REF-FIX) section, A1 row, v2 column (4 decimals)
NUMBER_TRACE_A1_V2 = {'GRANITE': 0.7664, 'Dasymetric': 0.7966, 'Pycnophylactic': 0.7604}


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _git(*args):
    return subprocess.check_output(['git', *args], cwd=str(REPO_ROOT)).decode().strip()


class Gate:
    def __init__(self):
        self.rows = []

    def check(self, target, committed, value, source, **keys):
        committed, value = float(committed), float(value)
        diff = abs(value - committed)
        self.rows.append({'target': target, **keys, 'committed_value': committed,
                          'module_value': value, 'abs_diff': diff,
                          'pass': bool(np.isfinite(diff) and diff <= TOL), 'source': source})


class BgAssigner:
    """block-group geoid per address via BlockGroupValidator, cached per coordinate array."""

    def __init__(self, validator, points):
        self.validator, self.points, self.cache = validator, points, {}

    def __call__(self, coords):
        key = (coords.shape, coords.tobytes())
        if key not in self.cache:
            joined = self.validator._assign_to_block_groups(self.points(coords))
            if len(joined) != len(coords) or not joined.index.equals(pd.RangeIndex(len(coords))):
                raise RuntimeError('block-group join changed the address rows')
            if joined['block_group_id'].isna().any():
                raise RuntimeError('addresses outside every block group')
            self.cache[key] = joined['block_group_id'].astype(str).to_numpy()
        return self.cache[key]


def address_frame(tract_arrays, assign=None):
    """one row per address: tract_geoid, x (lon), y (lat), value, bg_geoid."""
    parts = []
    for fips, coords, preds in tract_arrays:
        d = pd.DataFrame({'tract_geoid': fips, 'x': coords[:, 0], 'y': coords[:, 1],
                          'value': preds})
        if assign is not None:
            d['bg_geoid'] = assign(coords)
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def gate_sweep_sets(gate, rs, tract_list, assign, reference):
    rescored = pd.read_csv(REPO_ROOT / SRC['rescored_cells'])
    rescored = rescored.set_index(['set', 'condition', 'arch', 'training_seed', 'graph_draw_seed'])
    for set_name in ('5b', 'crossed'):
        trials = pd.read_csv(REPO_ROOT / SRC[f'{set_name}_trials'])
        metrics = json.loads((REPO_ROOT / SRC[f'{set_name}_metrics']).read_text())
        json_mi = {(c, a, int(s['training_seed']), int(s['graph_draw_seed'])): s['morans_i']
                   for c, by_arch in metrics.items() for a, blk in by_arch.items()
                   for s in blk['seeds']}
        pred_dir = rs.SETS[set_name] / 'predictions'
        for r in trials.itertuples(index=False):
            cell = (r.condition, r.arch, int(r.training_seed), int(r.graph_draw_seed))
            keys = dict(set=set_name, condition=cell[0], arch=cell[1],
                        training_seed=cell[2], graph_draw_seed=cell[3])
            arrays = []
            for fips in tract_list:
                z = np.load(pred_dir / f'{cell[0]}__{cell[1]}__ts{cell[2]}__gs{cell[3]}__{fips}.npz')
                arrays.append((fips, np.asarray(z['coords'], float),
                               np.asarray(z['predictions'], float)))
            df = address_frame(arrays, assign)
            coherence = run_coherence(per_tract_morans_i(df, 'value'))
            gate.check('coherence', r.morans_i, coherence, f'{set_name}_trials', **keys)
            gate.check('coherence', json_mi[cell], coherence, f'{set_name}_metrics', **keys)
            res = pooled_r(bg_predictions(df, 'value'), reference, method='pearson')
            ref_row = rescored.loc[(set_name,) + cell]
            gate.check('pooled_r_T_pearson_v2', ref_row['pooled_bg_r_v2'], res.r,
                       'rescored_cells', **keys)
            gate.check('n_scored_bgs', ref_row['n_bgs_v2'], res.n, 'rescored_cells', **keys)
            gate.check('n_scored_bgs_expected', EXPECTED_BGS, res.n, 'number_trace', **keys)
        print(f'[gate] {set_name}: {len(trials)} cells scored')


def gate_fig4(gate):
    rows = pd.read_csv(REPO_ROOT / SRC['fig4_tract_rows'], dtype={'tract_fips': str})
    pred_dir = RESULTS / 'm6_fig4_randomization' / 'predictions'
    for r in rows.itertuples(index=False):
        z = np.load(pred_dir / (f'{r.condition}__{r.arch}__ts{r.training_seed}__gs{r.graph_draw_seed}'
                                f'__ds{r.data_seed}__{r.tract_fips}.npz'))
        df = address_frame([(r.tract_fips, np.asarray(z['coords'], float),
                             np.asarray(z['predictions'], float))])
        mi = per_tract_morans_i(df, 'value')[r.tract_fips]
        gate.check('per_tract_morans_i', r.morans_i_output, mi, 'fig4_tract_rows',
                   set='fig4', condition=r.condition, arch=r.arch, training_seed=int(r.training_seed),
                   graph_draw_seed=int(r.graph_draw_seed), data_seed=int(r.data_seed),
                   tract=r.tract_fips)
    print(f'[gate] fig4: {len(rows)} tract rows scored')


def gate_m0(gate, rs, tract_list, assign, reference):
    per_tract = rs._load_m0_inputs(tract_list)
    committed = pd.read_csv(REPO_ROOT / SRC['m0_aggregate']).query(
        'reference_version == @REFERENCE_VERSION').set_index('method')
    coords = {t['fips']: np.array([[g.x, g.y] for g in t['gdf'].geometry]) for t in per_tract}
    out = {}
    for method in ('GRANITE', 'Dasymetric', 'Pycnophylactic'):
        df = address_frame([(t['fips'], coords[t['fips']], t['preds'][method]) for t in per_tract],
                           assign)
        res = pooled_r(bg_predictions(df, 'value'), reference, method='pearson')
        out[method] = res
        keys = dict(set='m0', method=method)
        gate.check('pooled_r_T_pearson_v2', committed.loc[method, 'pooled_bg_r'], res.r,
                   'm0_aggregate', **keys)
        gate.check('n_scored_bgs', committed.loc[method, 'pooled_n_bgs'], res.n,
                   'm0_aggregate', **keys)
        gate.check('n_scored_bgs_expected', EXPECTED_BGS, res.n, 'number_trace', **keys)

    # dasymetric recomputed from baselines.py as committed
    fm = pd.read_csv(REPO_ROOT / SRC['n20_feature_matrix'], dtype={'fips': str})
    dasy = DasymetricDisaggregation(ancillary_column='nlcd_impervious_pct')
    arrays, max_pred_diff = [], 0.0
    for t in per_tract:
        sub = fm[fm['fips'] == t['fips']].sort_values('address_idx')
        if not np.array_equal(sub['address_idx'].to_numpy(), np.arange(len(coords[t['fips']]))):
            raise RuntimeError(f"{t['fips']}: feature matrix rows do not match address order")
        tract_svi = sub['tract_svi'].unique()
        if len(tract_svi) != 1:
            raise RuntimeError(f"{t['fips']}: tract_svi not constant")
        p = dasy.disaggregate(coords[t['fips']], t['fips'], float(tract_svi[0]),
                              address_gdf=sub.reset_index(drop=True))
        max_pred_diff = max(max_pred_diff, float(np.max(np.abs(p - t['preds']['Dasymetric']))))
        arrays.append((t['fips'], coords[t['fips']], p))
    gate.check('dasymetric_recomputed_minus_parquet_max_abs', 0.0, max_pred_diff,
               'm0_parquet_dasymetric', set='m0', method='Dasymetric (baselines.py)')
    res = pooled_r(bg_predictions(address_frame(arrays, assign), 'value'), reference, method='pearson')
    gate.check('pooled_r_T_pearson_v2', committed.loc['Dasymetric', 'pooled_bg_r'], res.r,
               'm0_aggregate', set='m0', method='Dasymetric (baselines.py)')
    gate.check('n_scored_bgs', committed.loc['Dasymetric', 'pooled_n_bgs'], res.n,
               'm0_aggregate', set='m0', method='Dasymetric (baselines.py)')
    print('[gate] m0 scored')
    display = {m: {'number_trace_4dp': NUMBER_TRACE_A1_V2[m], 'module_value': out[m].r,
                   'module_rounded_4dp': round(out[m].r, 4),
                   'match': round(out[m].r, 4) == NUMBER_TRACE_A1_V2[m]}
               for m in NUMBER_TRACE_A1_V2}
    return display, {m: out[m].n_dropped for m in out}


def main():
    rs = _load_module('rescore_bg_reference', RESCORE_PATH)
    sweep = rs._load_sweep_module()
    tract_list = sweep._load_tract_list()
    bg_gdf = rs.load_bg_gdf(REFERENCE_VERSION)
    if bg_gdf['GEOID'].duplicated().any():
        raise RuntimeError('duplicated block-group GEOID in reference')
    reference = pd.Series(bg_gdf['SVI'].to_numpy(float), index=bg_gdf['GEOID'].astype(str))
    assign = BgAssigner(BlockGroupValidator(bg_gdf, verbose=False), rs._points)

    gate = Gate()
    gate_sweep_sets(gate, rs, tract_list, assign, reference)
    gate_fig4(gate)
    display, m0_dropped = gate_m0(gate, rs, tract_list, assign, reference)

    df = pd.DataFrame(gate.rows)
    sources = {k: {'file': SRC[k], 'commit': _git('log', '-1', '--format=%H', '--', SRC[k])}
               for k in sorted(set(df['source']))}
    summary = {
        'n_targets': int(len(df)), 'n_pass': int(df['pass'].sum()),
        'all_pass': bool(df['pass'].all()), 'max_abs_diff': float(df['abs_diff'].max()),
        'by_target': {t: {'n': int(len(g)), 'n_pass': int(g['pass'].sum()),
                          'max_abs_diff': float(g['abs_diff'].max())}
                      for t, g in df.groupby('target')},
        'n_unique_coordinate_arrays_joined': len(assign.cache),
        'm0_rows_dropped_no_reference': m0_dropped,
        'number_trace_display_check': display,
    }
    import esda, geopandas, libpysal, scipy
    payload = {
        'milestone': 'TAG-A',
        'gate': 'reproduction of committed Moran\'s I and Pearson pooled r_T',
        'tolerance': TOL,
        'reference_version': REFERENCE_VERSION,
        'git_head': _git('rev-parse', 'HEAD'),
        'module_commit': _git('log', '-1', '--format=%H', '--', 'granite/evaluation/paper1_metrics.py'),
        'module_functions': ['per_tract_morans_i', 'run_coherence', 'bg_predictions',
                             "pooled_r(method='pearson')"],
        'estimator': ("SpatialLearningDiagnostics.compute_spatial_autocorrelation: libpysal "
                      "KNN.from_array k=8 on EPSG:4326 lon, lat, transform 'r', esda.Moran, "
                      "no inference draws"),
        'bg_aggregation': ('BlockGroupValidator._assign_to_block_groups (sjoin within), plain '
                           'mean per block group, n_addresses >= 3'),
        'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                     'pandas': pd.__version__, 'scipy': scipy.__version__,
                     'geopandas': geopandas.__version__, 'libpysal': libpysal.__version__,
                     'esda': esda.__version__},
        'sources': sources,
        'summary': summary,
        'targets': gate.rows,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=1) + '\n')
    print(f"[gate] {summary['n_pass']}/{summary['n_targets']} within {TOL}; "
          f"max abs diff {summary['max_abs_diff']:.3e}")
    for t, s in summary['by_target'].items():
        print(f"  {t:48s} n={s['n']:5d} pass={s['n_pass']:5d} max={s['max_abs_diff']:.3e}")
    print(f'[gate] wrote {OUT_PATH.relative_to(REPO_ROOT)}')
    return 0 if summary['all_pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
