"""
POSCTRL-P1: output Moran's I for the address-supervised ceiling (positive control).

Scores ceiling_gbm out-of-fold predictions with the Figure 4a output Moran's I call
(run_grid._compute_morans_i_output: libpysal KNN k=8 on EPSG:4326 address coordinates,
transform 'r', esda.Moran permutations=0) on the committed seed-42 field, cell
latent/strong/medium/default.

Ceiling predictions come from run_grid._ceiling_gbm_r unchanged; the OOF prediction
vector is captured by wrapping run_grid._recovery_r for the duration of the call. The
predictions are in generator row order and are mapped by address hash onto the address
order and EPSG:4326 coordinates of the committed Figure 4a prediction files
(data/results/m6_fig4_randomization/predictions/spatial_knn_uniform__sage__ts42__*),
which equal DataLoader.get_addresses_for_tract geometry.

Pre-registered statistic (tracker, 2026-11-02):
  p = (I_ceil - I_truth) / (I_GRANITE - I_truth), I_truth = 0.153, I_GRANITE = 0.952
  A if p >= 0.8; C if 0.2 < p < 0.8; B if p <= 0.2; an interval crossing a band edge is C.

Output: data/results/paper1_figures/positive_control_morans_i.json

Usage (pinned environment, requirements.txt at fa3ba45):
    python experiments/m6_recovery_grid/run_positive_control.py
"""

import importlib.metadata as md
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))

import run_grid
from granite.data.loaders import DataLoader

FIELD_DIR = REPO_ROOT / 'data' / 'synthetic' / 'run_20260628_030416'
FIG4_PRED = REPO_ROOT / 'data' / 'results' / 'm6_fig4_randomization' / 'predictions'
FIG4_ROWS = REPO_ROOT / 'data' / 'results' / 'm6_fig4_randomization' / 'tract_rows.csv'
GRID_CSV = REPO_ROOT / 'data' / 'results' / 'm6_recovery_grid' / 'recovery_grid.csv'
FIG_NUMBERS = REPO_ROOT / 'data' / 'results' / 'paper1_figures' / 'figure_numbers.json'
OUT_JSON = REPO_ROOT / 'data' / 'results' / 'paper1_figures' / 'positive_control_morans_i.json'
TRACT_SEL = REPO_ROOT / 'experiments' / 'ablation' / '00_baseline' / 'tract_selection.txt'

CELL = {'signal_source': 'latent', 'autocorr': 'strong', 'snr': 'medium',
        'between_tract': 'default'}
DATA_SEED = 42
FOLD_SEED = 42                     # KFold and GBM random_state inside _ceiling_gbm_r
BOOT_SEED = 20261102               # date of pre-registration
N_BOOT = 2000
I_TRUTH = 0.153
I_GRANITE = 0.952
KNOWN_DEV_MAX = 3.8e-04           # bound on |recompute - committed june rows| per tract
FIELD_COMMIT = '0cb231eea2cbcab7e6cff102a1e58ca54576d36a'
GENERATOR_COMMIT = '0351298'       # = cdc860a before the 6280fa1 hash rewrite
GATE_DECISION = {
    'date': '2026-11-09',
    'by': 'Jamion (logged in the tracker before any ceiling Moran\'s I was seen)',
    'text': ('The reference of record for ceiling_gbm is the recompute under the pinned fa3ba45 '
             'environment (/workspaces/venv_figp1). The committed June rows (2dc231f) ran in an '
             'environment that was not pinned. Their deviation from the recompute is a known '
             'deviation: at most 3.77e-04 (47065003400) across the 11 tracts listed in the '
             'POSCTRL-P1 report. This follows the B2 precedent. The pre-registered statistic, '
             'bands, anchors (I_truth = 0.153, I_GRANITE = 0.952) and aggregation (unweighted '
             'mean over all 20 tracts, 47065011324 included) are unchanged.'),
}


def _git(*args):
    return subprocess.check_output(['git', *args], cwd=REPO_ROOT, text=True).strip()


def _tracts():
    return sorted(TRACT_SEL.read_text().split('fips_list:')[1].split())


def _ceiling_oof(coords, y_true):
    """call run_grid._ceiling_gbm_r unchanged; capture the OOF vector it scores."""
    captured = {}
    orig = run_grid._recovery_r

    def capture(predictions, y):
        captured['preds'] = np.asarray(predictions, dtype=float).copy()
        captured['y'] = np.asarray(y, dtype=float).copy()
        return orig(predictions, y)

    run_grid._recovery_r = capture
    try:
        r = run_grid._ceiling_gbm_r(coords, y_true)
    finally:
        run_grid._recovery_r = orig
    return r, captured['preds'], captured['y']


def _band(p):
    if p >= 0.8:
        return 'A'
    if p <= 0.2:
        return 'B'
    return 'C'


def _band_with_interval(p, lo, hi):
    edges = (0.2, 0.8)
    if any(lo < e < hi or lo == e or hi == e for e in edges):
        return 'C'
    return _band(p)


def main():
    tracts = _tracts()
    assert len(tracts) == 20
    draw_spec = dict(CELL, seed=DATA_SEED)
    _, addr_df, _, y_min, y_max, _ = run_grid._generate_draw(draw_spec, field_dir=FIELD_DIR)
    loader = DataLoader()   # raw data is not in git; default data_dir, as in run_fig4

    grid = pd.read_csv(GRID_CSV, dtype={'tract_fips': str})
    cg = grid[(grid['method'] == 'ceiling_gbm') & (grid['signal_source'] == CELL['signal_source'])
              & (grid['autocorr'] == CELL['autocorr']) & (grid['snr'] == CELL['snr'])
              & (grid['between_tract'] == CELL['between_tract']) & (grid['seed'] == DATA_SEED)]
    committed_r = cg.set_index('tract_fips')['recovery_r']
    assert committed_r.index.is_unique and set(tracts) <= set(committed_r.index)

    f4 = pd.read_csv(FIG4_ROWS, dtype={'tract_fips': str})
    f4 = f4[(f4['condition'] == 'spatial_knn_uniform') & (f4['arch'] == 'sage')]
    assert len(f4) == 100
    granite_tract = f4.groupby('tract_fips')['morans_i_output'].mean()

    rows, gate = [], []
    for f in tracts:
        sub = addr_df[addr_df['fips'] == f]
        r, preds, y_scored = _ceiling_oof(sub[['x', 'y']].values, sub['y_true'].values)
        r2, preds2, _ = _ceiling_oof(sub[['x', 'y']].values, sub['y_true'].values)
        assert len(preds) == len(sub) and np.isfinite(preds).all()
        assert np.array_equal(y_scored, sub['y_true'].to_numpy(float))
        deterministic = bool(r == r2 and np.array_equal(preds, preds2))
        dev = abs(r - float(committed_r[f]))
        gate.append({'tract_fips': f, 'recovery_r_committed_june': float(committed_r[f]),
                     'recovery_r_recomputed': r, 'recovery_r_recomputed_rerun': r2,
                     'deterministic': deterministic, 'abs_diff': dev,
                     'within_known_deviation': bool(dev <= KNOWN_DEV_MAX)})

        # map generator-order predictions onto the figure 4a address order and coordinates
        gdf = loader.get_addresses_for_tract(f)
        z = np.load(FIG4_PRED / f'spatial_knn_uniform__sage__ts42__gs42__ds42__{f}.npz')
        coords = np.array([[g.x, g.y] for g in gdf.geometry])
        assert np.array_equal(coords, z['coords'])
        pos = pd.Series(np.arange(len(sub)), index=sub['address_hash'].values)
        idx = pos.reindex(gdf['hash'].values).to_numpy()
        assert not np.isnan(idx).any() and len(set(idx)) == len(sub) == len(gdf)
        idx = idx.astype(int)
        ceil_aligned = preds[idx]
        y_aligned = sub['y_true'].to_numpy(float)[idx]
        assert np.array_equal(y_aligned, z['y_true'])

        mi_ceil = run_grid._compute_morans_i_output(ceil_aligned, gdf)
        mi_true = run_grid._compute_morans_i_output(y_aligned, gdf)
        rows.append({'tract_fips': f, 'n_addresses': int(len(sub)),
                     'ceiling_recovery_r': r,
                     'ceiling_morans_i_output': float(mi_ceil),
                     'y_true_morans_i': float(mi_true),
                     'granite_morans_i_output_seedmean': float(granite_tract[f]),
                     'diff_ceiling_minus_granite': float(mi_ceil - granite_tract[f])})
        print(f'[posctrl] {f} n={len(sub)} r={r:.6f} dev={dev:.3e} '
              f'I_ceil={mi_ceil:.4f} I_true={mi_true:.4f} I_gran={granite_tract[f]:.4f}')

    det_ok = all(g['deterministic'] for g in gate)
    dev_ok = all(g['within_known_deviation'] for g in gate)
    gate_ok = det_ok and dev_ok
    max_g = max(gate, key=lambda g: g['abs_diff'])
    print(f'[posctrl] determinism {"PASS" if det_ok else "FAIL"}; known deviation '
          f'{"PASS" if dev_ok else "FAIL"}; max dev {max_g["abs_diff"]:.3e} at '
          f'{max_g["tract_fips"]}')
    for g in gate:
        print(f'[posctrl] dev {g["tract_fips"]} june={g["recovery_r_committed_june"]!r} '
              f'recompute={g["recovery_r_recomputed"]!r} abs_diff={g["abs_diff"]!r} '
              f'det={g["deterministic"]}')
    if not gate_ok:
        for g in gate:
            if not (g['deterministic'] and g['within_known_deviation']):
                print('[posctrl] FAIL', g)
        print('[posctrl] STOP: nothing written')
        sys.exit(2)

    t = pd.DataFrame(rows)
    cols = {'ceiling_recovery_r': 'ceiling_recovery_r',
            'ceiling_morans_i_output': 'ceiling_morans_i_output',
            'y_true_morans_i': 'y_true_morans_i',
            'granite_morans_i_output': 'granite_morans_i_output_seedmean',
            'diff_ceiling_minus_granite': 'diff_ceiling_minus_granite'}
    M = np.column_stack([t[c].to_numpy(float) for c in cols.values()])
    point = M.mean(axis=0)

    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(t), size=(N_BOOT, len(t)))
    B = M[idx].mean(axis=1)                     # (N_BOOT, n_stats), same resamples for all

    def ci(a):
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

    agg = {k: {'mean': float(point[j]), 'ci95': ci(B[:, j])} for j, k in enumerate(cols)}

    j_c, j_t, j_g = 1, 2, 3
    p_point = (point[j_c] - I_TRUTH) / (I_GRANITE - I_TRUTH)
    p_boot = (B[:, j_c] - I_TRUTH) / (I_GRANITE - I_TRUTH)
    p_ci = ci(p_boot)
    p_joint = (B[:, j_c] - B[:, j_t]) / (B[:, j_g] - B[:, j_t])
    band_point = _band(p_point)
    band = _band_with_interval(p_point, *p_ci)
    print(f'[posctrl] I_ceil={point[j_c]:.6f} p={p_point:.6f} CI={p_ci} band={band}')

    pkgs = {}
    for name in ['scikit-learn', 'numpy', 'pandas', 'scipy', 'geopandas', 'shapely', 'pyogrio',
                 'libpysal', 'esda', 'torch']:
        try:
            pkgs[name] = md.version(name)
        except md.PackageNotFoundError:
            pkgs[name] = None

    fig_numbers = json.loads(FIG_NUMBERS.read_text())['numbers']
    fn = {n['name']: n['value'] for n in fig_numbers}

    out = {
        'milestone': 'POSCTRL-P1',
        'statistic': 'ceiling_gbm mean per-tract output Moran\'s I on out-of-fold predictions',
        'estimator': ("run_grid._compute_morans_i_output -> SpatialLearningDiagnostics."
                      "compute_spatial_autocorrelation: libpysal KNN.from_array k=8 on EPSG:4326 "
                      "address coordinates (address_gdf geometry), transform 'r', esda.Moran "
                      "permutations=0; identical call to experiments/m6_recovery_grid/run_fig4.py"),
        'aggregation': 'unweighted mean over the 20 Paper 1 tracts',
        'granite_comparator': ("spatial_knn_uniform sage, data/results/m6_fig4_randomization/"
                               "tract_rows.csv; per-tract mean over 5 training seeds"),
        'ceiling_method': ("run_grid._ceiling_gbm_r unchanged: z-scored generator UTM x, y; "
                           "KFold(n_splits=5, shuffle=True, random_state=42); "
                           "GradientBoostingRegressor(n_estimators=200, max_depth=4, "
                           "learning_rate=0.05, random_state=42); r is Pearson r of the "
                           "concatenated OOF predictions"),
        'alignment': ("OOF predictions (generator row order) mapped by address hash onto "
                      "DataLoader.get_addresses_for_tract order; coordinates asserted equal to "
                      "the Figure 4a npz coords and aligned y_true asserted equal to the npz "
                      "y_true for every tract"),
        'provenance': {
            'field_dir': str(FIELD_DIR.relative_to(REPO_ROOT)),
            'field_commit': FIELD_COMMIT,
            'generator_commit': GENERATOR_COMMIT,
            'head': _git('rev-parse', 'HEAD'),
            'cell': CELL, 'data_seed': DATA_SEED,
            'ytrue_rescale_min': y_min, 'ytrue_rescale_max': y_max,
            'fold_seed': FOLD_SEED, 'bootstrap_seed': BOOT_SEED, 'n_boot': N_BOOT,
            'bootstrap': 'tract-clustered percentile, same resample indices for every statistic',
            'packages': pkgs, 'requirements_commit': 'fa3ba45',
            'figure4_commits': ['5fe70f9', 'c9039a9', '3e63f25'],
        },
        'anchor': {
            'rule': 'pre-registered anchors and bands, no re-anchor (truth within 0.005 of 0.153)',
            'I_truth_used': I_TRUTH, 'I_granite_used': I_GRANITE,
            'I_truth_item4_call_recomputed': agg['y_true_morans_i']['mean'],
            'I_truth_utm_figure_numbers': fn['field_seed42_morans_i_true_current_estimator'],
            'I_granite_figure_numbers': fn['morans_i_output_spatial_knn_uniform_sage'],
        },
        'reproduction_gate': {
            'decision': GATE_DECISION,
            'criterion': ('determinism: two in-process recomputes bit-identical per tract (r and '
                          'OOF vector); known deviation: |recompute - committed june '
                          f'recovery_grid.csv (2dc231f) r| <= {KNOWN_DEV_MAX} per tract'),
            'reference_of_record': 'recompute under fa3ba45 (/workspaces/venv_figp1)',
            'determinism_pass': det_ok, 'known_deviation_pass': dev_ok, 'pass': gate_ok,
            'max_abs_diff': max_g['abs_diff'], 'max_abs_diff_tract': max_g['tract_fips'],
            'per_tract': gate,
        },
        'aggregate': agg,
        'signal_component': 'not persisted (field columns: fips, address_hash, x, y, y_true)',
        'p': {'point': float(p_point), 'ci95_anchor_fixed': p_ci,
              'ci95_joint_resampled_truth_and_granite': ci(p_joint),
              'band_point_estimate': band_point, 'band_reported': band,
              'band_rule': 'A if p >= 0.8; C if 0.2 < p < 0.8; B if p <= 0.2; interval '
                           'crossing a band edge reports as C (anchor-fixed interval)'},
        'per_tract': rows,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2) + '\n')
    print(f'[posctrl] wrote {OUT_JSON.relative_to(REPO_ROOT)}')


if __name__ == '__main__':
    main()
