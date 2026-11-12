#!/usr/bin/env python3
"""
paper 1 confirmatory analysis (preregistration v4.2, sections 4 to 7).

every metric comes from granite.evaluation.paper1_metrics and every interval
and decision from granite.evaluation.paper1_inference; run inputs load
through scripts/rescore_bg_reference.py (npz by tract order, EPSG:4326
coordinates, block groups via BlockGroupValidator). bootstrap statistics run
on the module's array cores (correlation, demean_within, fisher_z_mean,
q_difference, run_coherence) and every full-sample value is asserted equal to
pooled_r / within_tract_r within 1e-12.

modes:
  --synthetic DIR --out OUT   every analysis on a synthetic run set laid out
                              like data/results (tests only; OUT outside
                              data/results; --draws and --n-perm allowed here only)
  --dry-run-real [--out OUT]  real inputs, counts only (tracts, scored block
                              groups, runs per cell, pending cells, exclusion)
  --confirmatory              requires tag paper1-prereg-v1 as an ancestor of
                              HEAD and a clean tree; writes
                              data/results/paper1_confirmatory/<HEAD short>/
  --check-guard-only          with --confirmatory: exit after the guard

usage:
    python scripts/paper1_confirmatory.py --dry-run-real
    python scripts/paper1_confirmatory.py --confirmatory
"""
import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from granite.evaluation import paper1_inference as pi
from granite.evaluation import paper1_metrics as pm
from granite.evaluation.baselines import DasymetricDisaggregation
from granite.validation.block_group_validation import BlockGroupValidator

RESCORE_PATH = REPO_ROOT / 'scripts' / 'rescore_bg_reference.py'
_spec = importlib.util.spec_from_file_location('rescore_bg_reference', RESCORE_PATH)
rs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rs)

# spatial_diagnostics.py silences warnings process-wide; undo it so a nan path raises
warnings.resetwarnings()
warnings.simplefilter('error', RuntimeWarning)

# ---------------------------------------------------------------------------
# registered constants (preregistration v4.2)
# ---------------------------------------------------------------------------

BOOTSTRAP_SEED = 20261113
S2_SEED = 20261114            # within-tract permutation null; run i uses S2_SEED + i
N_PERM = 1000
LOWER_BOUND_THRESHOLD = 0.2   # H1, H3
TOST_MARGIN = 0.1             # H2, S1
ONE_SIDED_LEVEL = 0.95
TWO_SIDED_LEVEL = 0.90
JK_Z = pi.JACKKNIFE_Z
REFERENCE_VERSION = 'v2'
TAG = 'paper1-prereg-v1'
TRAINING_SEEDS = (42, 17, 123, 2024, 7)
GRAPH_DRAW_SEEDS = (42, 17, 123, 2024, 7)
STRUCTURED_GS = 42
SENSITIVITY = {
    'without_47065001900': ('47065001900',),
    'without_47065001900_47065010431': ('47065001900', '47065010431'),
}
EXPECTED_REAL_EXCLUSION = {'47065011324'}
ARCHS = ('sage', 'gcn_gat')

# real inputs (relative to the repo root)
REAL_RESULTS = Path('data/results')
REAL_SWAP_REBUILD = Path('data/results/paper1_prereg_artifacts/swap_rebuild.json')
REAL_FEATURES = Path('experiments/ecological_fallacy/n20_feature_matrix.csv')
CONFIRMATORY_DIR = REPO_ROOT / 'data' / 'results' / 'paper1_confirmatory'

# synthetic run set layout (relative to --synthetic DIR); cell dirs as in MANIFEST
SYN_TRACTS = 'tracts.txt'
SYN_BLOCK_GROUPS = 'block_groups.geojson'
SYN_FEATURES = 'features.csv'
SYN_SWAP_REBUILD = 'swap_rebuild.json'

_DIAG = [(i, i) for i in TRAINING_SEEDS]


def _structured(dir_, condition, arch):
    return {'dir': dir_, 'condition': condition, 'arch': arch,
            'seeds': [(ts, STRUCTURED_GS) for ts in TRAINING_SEEDS]}


# cell -> run directory (relative to the results root), condition, arch, (training seed,
# graph-draw seed) pairs. randomized cells read the crossed set only. the last three
# directories are post-tag runs; mlp_floor never reads the 59bd8e9 values (retired
# estimator, no saved predictions).
MANIFEST = {}
for _a in ARCHS:
    MANIFEST[f'{_a}/euclidean'] = _structured('m6_topology_5b', 'spatial_knn_uniform', _a)
    MANIFEST[f'{_a}/randomized'] = {'dir': 'm6_topology_5b_crossed', 'condition': 'randomized',
                                    'arch': _a, 'seeds': [(ts, gs) for ts in TRAINING_SEEDS
                                                          for gs in GRAPH_DRAW_SEEDS]}
    MANIFEST[f'{_a}/road_network'] = _structured('m6_topology_5b', 'road_network_uniform', _a)
    MANIFEST[f'{_a}/production'] = _structured('m6_topology_5b', 'production', _a)
    MANIFEST[f'{_a}/no_graph'] = _structured('paper1_no_graph_rerun', 'mlp_floor', _a)
MANIFEST['sage_root_weight_false/euclidean'] = _structured(
    'paper1_h3_sage_root_weight_false', 'spatial_knn_uniform', 'sage')
MANIFEST['sage_root_weight_false/randomized'] = {
    'dir': 'paper1_h3_sage_root_weight_false', 'condition': 'randomized', 'arch': 'sage',
    'seeds': list(_DIAG)}
MANIFEST['gcn_gat_residual/euclidean'] = _structured(
    'paper1_s4_gcn_gat_residual', 'spatial_knn_uniform', 'gcn_gat')
MANIFEST['gcn_gat_residual/randomized'] = {
    'dir': 'paper1_s4_gcn_gat_residual', 'condition': 'randomized', 'arch': 'gcn_gat',
    'seeds': list(_DIAG)}

DASYMETRIC = 'dasymetric'


def _versions():
    import esda
    import geopandas
    import libpysal
    import networkx
    import scipy
    import sklearn
    return {'python': sys.version.split()[0], 'numpy': np.__version__, 'pandas': pd.__version__,
            'scipy': scipy.__version__, 'geopandas': geopandas.__version__,
            'libpysal': libpysal.__version__, 'esda': esda.__version__,
            'scikit-learn': sklearn.__version__, 'networkx': networkx.__version__}


def _git(*args, cwd=REPO_ROOT):
    return subprocess.run(['git', *args], cwd=str(cwd), capture_output=True, text=True)


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

class Inputs:
    """tracts, reference, block-group assignment, dasymetric features, swap rebuild."""

    def __init__(self, root, tracts, bg_gdf, features_path, swap_rebuild_path):
        self.root = Path(root)
        self.tracts = [str(t) for t in tracts]
        self.bg_gdf = bg_gdf
        self.features_path = Path(features_path)
        self.swap_rebuild_path = Path(swap_rebuild_path)
        self.reference = pd.Series(bg_gdf['SVI'].to_numpy(float), index=bg_gdf['GEOID'].astype(str))
        if self.reference.index.duplicated().any():
            raise RuntimeError('duplicated block-group GEOID in reference')
        self._validator = BlockGroupValidator(bg_gdf, verbose=False)
        self._bg_cache = {}

    def assign_bg(self, coords):
        key = hashlib.sha256(np.ascontiguousarray(coords).tobytes()).hexdigest()
        if key not in self._bg_cache:
            joined = self._validator._assign_to_block_groups(rs._points(coords))
            if len(joined) != len(coords) or joined['block_group_id'].isna().any():
                raise RuntimeError('block-group join dropped or duplicated addresses')
            self._bg_cache[key] = joined['block_group_id'].astype(str).to_numpy()
        return self._bg_cache[key]


def real_inputs():
    tracts = rs._load_sweep_module()._load_tract_list()
    return Inputs(REPO_ROOT / REAL_RESULTS, tracts, rs.load_bg_gdf(REFERENCE_VERSION),
                  REPO_ROOT / REAL_FEATURES, REPO_ROOT / REAL_SWAP_REBUILD)


def synthetic_inputs(root):
    import geopandas as gpd
    root = Path(root)
    tracts = [l.strip() for l in (root / SYN_TRACTS).read_text().splitlines() if l.strip()]
    bg = gpd.read_file(root / SYN_BLOCK_GROUPS)
    bg['GEOID'] = bg['GEOID'].astype(str)
    if bg.crs is None:
        bg = bg.set_crs('EPSG:4326')
    return Inputs(root, tracts, bg, root / SYN_FEATURES, root / SYN_SWAP_REBUILD)


def cell_paths(inputs, spec):
    """{(ts, gs, tract): path}; None when no file exists (pending); raises when partial."""
    d = inputs.root / spec['dir'] / 'predictions'
    paths = {(ts, gs, f): d / f"{spec['condition']}__{spec['arch']}__ts{ts}__gs{gs}__{f}.npz"
             for ts, gs in spec['seeds'] for f in inputs.tracts}
    present = [p for p in paths.values() if p.exists()]
    if not present:
        return None
    if len(present) != len(paths):
        raise RuntimeError(f"partial cell under {d}: {len(present)} of {len(paths)} files")
    return paths


def manifest_status(inputs):
    status = {}
    for cell, spec in MANIFEST.items():
        paths = cell_paths(inputs, spec)
        status[cell] = {'dir': spec['dir'], 'condition': spec['condition'], 'arch': spec['arch'],
                        'n_runs_expected': len(spec['seeds']),
                        'status': 'pending' if paths is None else 'available',
                        'n_files': 0 if paths is None else len(paths)}
    return status


def derive_exclusion(inputs, status):
    """tracts with randomized = False in swap_rebuild.json or any randomized input's
    swap_provenance.csv."""
    excluded, sources = set(), {}
    rebuild = json.loads(inputs.swap_rebuild_path.read_text())
    for r in rebuild['rows']:
        if not bool(r['randomized']):
            excluded.add(str(r['tract']))
            sources.setdefault(str(r['tract']), set()).add(str(inputs.swap_rebuild_path.name))
    for cell, spec in MANIFEST.items():
        if spec['condition'] != 'randomized' or status[cell]['status'] != 'available':
            continue
        prov = inputs.root / spec['dir'] / 'swap_provenance.csv'
        if prov.exists():
            p = pd.read_csv(prov, dtype={'tract_fips': str})
            flag = p['randomized'].astype(str).str.lower().map({'true': True, 'false': False})
            if flag.isna().any():
                raise RuntimeError(f'{prov}: unreadable randomized flags')
            for t in p.loc[~flag.astype(bool), 'tract_fips']:
                excluded.add(str(t))
                sources.setdefault(str(t), set()).add(str(prov.relative_to(inputs.root)))
    return sorted(excluded), {k: sorted(v) for k, v in sorted(sources.items())}


# ---------------------------------------------------------------------------
# dry run (counts only)
# ---------------------------------------------------------------------------

def _canonical_coords(inputs, status):
    """coordinates per tract from the first available cell's first run."""
    for cell, spec in MANIFEST.items():
        if status[cell]['status'] == 'available':
            ts, gs = spec['seeds'][0]
            d = inputs.root / spec['dir'] / 'predictions'
            return {f: np.asarray(np.load(d / f"{spec['condition']}__{spec['arch']}__ts{ts}__gs{gs}"
                                             f"__{f}.npz")['coords'], dtype=float)
                    for f in inputs.tracts}
    raise RuntimeError('no available cell')


def _scored_bg_count(inputs, coords):
    """block groups with at least MIN_BG_ADDRESSES addresses and a finite reference."""
    bg = np.concatenate([inputs.assign_bg(coords[f]) for f in inputs.tracts])
    n = pd.Series(bg).value_counts()
    kept = n.index[n >= pm.MIN_BG_ADDRESSES]
    ref = inputs.reference.reindex(kept)
    return int(np.isfinite(ref.to_numpy(float)).sum()), int(len(kept))


DRY_RUN_FUNCTIONS = ('dry_run_real', 'manifest_status', 'cell_paths', 'derive_exclusion',
                     '_canonical_coords', '_scored_bg_count')


def dry_run_real():
    inputs = real_inputs()
    status = manifest_status(inputs)
    excluded, sources = derive_exclusion(inputs, status)
    if set(excluded) != EXPECTED_REAL_EXCLUSION:
        raise RuntimeError(f'real exclusion set {excluded} != {sorted(EXPECTED_REAL_EXCLUSION)}')
    coords = _canonical_coords(inputs, status)
    n_scored, n_kept = _scored_bg_count(inputs, coords)
    feats = pd.read_csv(inputs.features_path, usecols=['fips'], dtype={'fips': str})
    return {
        'mode': 'dry-run-real',
        'n_tracts': len(inputs.tracts),
        'n_addresses': int(sum(len(c) for c in coords.values())),
        'n_block_groups_with_min_addresses': n_kept,
        'n_scored_block_groups': n_scored,
        'n_dasymetric_feature_rows': int(feats['fips'].isin(inputs.tracts).sum()),
        'excluded_tracts': excluded,
        'exclusion_sources': sources,
        'n_tracts_randomized_contrasts': len(inputs.tracts) - len(excluded),
        'cells': status,
        'pending_cells': sorted(c for c, s in status.items() if s['status'] == 'pending'),
    }


# ---------------------------------------------------------------------------
# run data
# ---------------------------------------------------------------------------

class RunData:
    """per-run per-tract moran's i and block-group values for every available cell."""

    def __init__(self, inputs, status):
        self.inputs = inputs
        self.cells = {c: MANIFEST[c] for c, s in status.items() if s['status'] == 'available'}
        self.coords = {}
        self.moran = {}      # run id -> Series tract -> I
        self.bg = {}         # run id -> bg_predictions frame
        self.address = {}    # run id -> address frame (for S2)
        self.runs = {}       # cell -> [run ids]
        self._moran_cache = {}
        for cell, spec in self.cells.items():
            paths = cell_paths(inputs, spec)
            self.runs[cell] = []
            for ts, gs in spec['seeds']:
                rid = f'{cell}/ts{ts}/gs{gs}'
                arrays = []
                for f in inputs.tracts:
                    z = np.load(paths[(ts, gs, f)])
                    c = np.asarray(z['coords'], dtype=float)
                    if f in self.coords:
                        if not np.array_equal(self.coords[f], c):
                            raise RuntimeError(f'{rid} {f}: coordinates differ across runs')
                    else:
                        self.coords[f] = c
                    arrays.append((f, np.asarray(z['predictions'], dtype=float)))
                self._add_run(rid, arrays)
                self.runs[cell].append(rid)
        self._add_dasymetric()

    def _frame(self, arrays):
        parts = []
        for f, v in arrays:
            c = self.coords[f]
            if len(c) != len(v):
                raise RuntimeError(f'{f}: {len(v)} values for {len(c)} addresses')
            parts.append(pd.DataFrame({'tract_geoid': f, 'x': c[:, 0], 'y': c[:, 1], 'value': v,
                                       'bg_geoid': self.inputs.assign_bg(c)}))
        return pd.concat(parts, ignore_index=True)

    def _tract_moran(self, f, v):
        key = (f, hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest())
        if key not in self._moran_cache:
            i = pm.tract_morans_i(self.coords[f], v)
            if not np.isfinite(i):
                raise RuntimeError(f"tract {f}: undefined moran's i (zero prediction variance)")
            self._moran_cache[key] = i
        return self._moran_cache[key]

    def _add_run(self, rid, arrays):
        df = self._frame(arrays)
        self.address[rid] = df
        self.moran[rid] = pd.Series({f: self._tract_moran(f, v) for f, v in arrays})
        self.bg[rid] = pm.bg_predictions(df, 'value')

    def _add_dasymetric(self):
        fm = pd.read_csv(self.inputs.features_path, dtype={'fips': str})
        dasy = DasymetricDisaggregation(ancillary_column='nlcd_impervious_pct')
        arrays = []
        for f in self.inputs.tracts:
            sub = fm[fm['fips'] == f].sort_values('address_idx').reset_index(drop=True)
            if not np.array_equal(sub['address_idx'].to_numpy(), np.arange(len(self.coords[f]))):
                raise RuntimeError(f'{f}: feature rows do not match address order')
            svi = sub['tract_svi'].unique()
            if len(svi) != 1:
                raise RuntimeError(f'{f}: tract_svi not constant')
            arrays.append((f, dasy.disaggregate(self.coords[f], f, float(svi[0]), address_gdf=sub)))
        self._add_run(DASYMETRIC, arrays)
        self.runs[DASYMETRIC] = [DASYMETRIC]


class BGTable:
    """scored block groups aligned across runs, for fast resampled correlations."""

    def __init__(self, data):
        rids = list(data.bg)
        base = data.bg[rids[0]]
        for r in rids[1:]:
            if not data.bg[r]['bg_geoid'].equals(base['bg_geoid']):
                raise RuntimeError(f'{r}: scored block groups differ across runs')
        t = base['bg_geoid'].map(data.inputs.reference).to_numpy(float)
        keep = np.isfinite(t)
        self.bg = base['bg_geoid'].to_numpy()[keep]
        self.tract = base['tract_geoid'].to_numpy()[keep]
        self.t = t[keep]
        self.row = {r: i for i, r in enumerate(rids)}
        self.P = np.stack([data.bg[r]['value'].to_numpy(float)[keep] for r in rids])
        n_bg = pd.Series(self.bg).groupby(self.tract).nunique()
        self.rw_tracts = set(n_bg.index[n_bg >= 2])
        self.rows_T = {f: np.flatnonzero(self.tract == f) for f in data.inputs.tracts}
        self.rows_W = {f: (self.rows_T[f] if f in self.rw_tracts else np.array([], dtype=int))
                       for f in data.inputs.tracts}
        # full-sample parity with the module's pandas functions
        ref = data.inputs.reference
        for r in rids:
            for kind, fn in (('T', pm.pooled_r), ('W', pm.within_tract_r)):
                a = self.r([r], data.inputs.tracts, kind)[0]
                b = fn(data.bg[r], ref, 'pearson').r
                if not abs(a - b) <= 1e-12:
                    raise RuntimeError(f'{r} r_{kind}: fast path {a} != module {b}')

    def r(self, rids, tracts, kind):
        rows = self.rows_T if kind == 'T' else self.rows_W
        idx = np.concatenate([rows[f] for f in tracts])
        X = self.P[[self.row[x] for x in rids]][:, idx]
        y = self.t[idx]
        if kind == 'W':
            g = self.tract[idx]
            X, y = pm.demean_within(X, g), pm.demean_within(y, g)
        return np.atleast_1d(pm.correlation(X, y, 'pearson'))


# ---------------------------------------------------------------------------
# interval summaries
# ---------------------------------------------------------------------------

def _f(x):
    a = np.asarray(x, dtype=float)
    return float(a) if a.ndim == 0 else [float(v) for v in a]


def _boot(tracts, stat, n_draws):
    return pi.tract_bootstrap(list(tracts), stat, n_draws=n_draws, seed=BOOTSTRAP_SEED)


def _boot_meta(b):
    return {'n_draws': b.n_draws, 'seed': b.seed, 'quantile_method': b.quantile_method,
            'numpy_version': b.numpy_version, 'n_nonfinite': b.n_nonfinite,
            'draws_sd': _f(np.std(b.draws, axis=0, ddof=1))}


def lower_bound_test(tracts, stat, n_draws, names=None):
    b = _boot(tracts, stat, n_draws)
    j = pi.tract_jackknife(list(tracts), stat)
    bi = pi.percentile_interval(b.draws, ONE_SIDED_LEVEL, sided='lower')
    ji = pi.jackknife_interval(j.estimate, j.se, z=JK_Z, sided='lower')
    vb = pi.decide_lower_bound(bi.lower, LOWER_BOUND_THRESHOLD)
    vj = pi.decide_lower_bound(ji.lower, LOWER_BOUND_THRESHOLD)
    return {'estimate': _f(b.estimate), 'threshold': LOWER_BOUND_THRESHOLD,
            'bootstrap': {'lower_95': _f(bi.lower), 'quantile': bi.quantiles[0], **_boot_meta(b)},
            'jackknife': {'se': _f(j.se), 'lower_95': _f(ji.lower), 'z': JK_Z},
            'verdicts': pi.compare_verdicts(vb, vj)}


def two_sided(tracts, stat, n_draws, names):
    b = _boot(tracts, stat, n_draws)
    j = pi.tract_jackknife(list(tracts), stat)
    bi = pi.percentile_interval(b.draws, TWO_SIDED_LEVEL)
    ji = pi.jackknife_interval(j.estimate, j.se, z=JK_Z)
    out = {'components': list(names), 'estimate': dict(zip(names, np.atleast_1d(_f(b.estimate)))),
           'bootstrap': {'ci90': {n: [lo, hi] for n, lo, hi in
                                  zip(names, np.atleast_1d(_f(bi.lower)), np.atleast_1d(_f(bi.upper)))},
                         'quantiles': list(bi.quantiles), **_boot_meta(b)},
           'jackknife': {'se': dict(zip(names, np.atleast_1d(_f(j.se)))),
                         'ci90': {n: [lo, hi] for n, lo, hi in
                                  zip(names, np.atleast_1d(_f(ji.lower)), np.atleast_1d(_f(ji.upper)))},
                         'z': JK_Z}}
    return out


def tost_test(tracts, stat, n_draws, names):
    out = two_sided(tracts, stat, n_draws, names)
    q = names[0]
    db = pi.decide_tost(*out['bootstrap']['ci90'][q], TOST_MARGIN)
    dj = pi.decide_tost(*out['jackknife']['ci90'][q], TOST_MARGIN)
    out['tost'] = {'statistic': q, 'margin': TOST_MARGIN,
                   'bootstrap': {'verdict': db.verdict, 'tightest_margin': db.tightest_margin},
                   'jackknife': {'verdict': dj.verdict, 'tightest_margin': dj.tightest_margin}}
    out['verdicts'] = pi.compare_verdicts(db.verdict, dj.verdict)
    return out


# ---------------------------------------------------------------------------
# analyses
# ---------------------------------------------------------------------------

def _cell_mean_moran(data, rids):
    return pd.concat([data.moran[r] for r in rids], axis=1).mean(axis=1)


def _coherence_stat(per_tract):
    def stat(s):
        return pm.run_coherence(per_tract, s)
    return stat


def _corr_contrast_stat(table, rids_a, rids_b, kinds):
    """for each kind: q = artanh(zbar a) - artanh(zbar b) and d_r = zbar a - zbar b."""
    def stat(s):
        out = []
        for kind in kinds:
            za = pm.fisher_z_mean(table.r(rids_a, s, kind))
            zb = pm.fisher_z_mean(table.r(rids_b, s, kind))
            out += [pm.q_difference(za, zb), za - zb]
        return out
    return stat


def _seed_pair_runs(data, cell, pairs):
    want = {f'{cell}/ts{ts}/gs{gs}' for ts, gs in pairs}
    got = [r for r in data.runs[cell] if r in want]
    if len(got) != len(want):
        raise RuntimeError(f'{cell}: missing runs for pairs {pairs}')
    return got


class Analyses:
    def __init__(self, data, table, excluded, n_draws, n_perm):
        self.d, self.t = data, table
        self.all = list(data.inputs.tracts)
        self.excluded = list(excluded)
        self.rand_tracts = [f for f in self.all if f not in set(excluded)]
        self.n_draws, self.n_perm = n_draws, n_perm
        self.results, self.skipped = {}, []

    def have(self, *cells):
        return all(c in self.d.runs for c in cells)

    def run(self, name, cells, fn, **meta):
        if not self.have(*cells):
            missing = sorted(c for c in cells if c not in self.d.runs)
            self.skipped.append({'analysis': name, 'pending_cells': missing})
            return
        res = fn()
        self.results[name] = {**meta, **res}

    # H1 and its relatives: randomization cost on coherence
    def coherence_cost(self, arch, tracts):
        diff = (_cell_mean_moran(self.d, self.d.runs[f'{arch}/euclidean'])
                - _cell_mean_moran(self.d, self.d.runs[f'{arch}/randomized']))
        return {'tracts': list(tracts), 'n_tracts': len(tracts),
                'per_tract': {f: float(diff[f]) for f in tracts},
                **lower_bound_test(tracts, _coherence_stat(diff), self.n_draws)}

    def self_path_interaction(self, base, alt, tracts):
        """mean over diagonal pairs of cost(alt) - cost(base) per tract."""
        e_on = _seed_pair_runs(self.d, f'{base}/euclidean', [(i, STRUCTURED_GS) for i in TRAINING_SEEDS])
        r_on = _seed_pair_runs(self.d, f'{base}/randomized', _DIAG)
        e_off = _seed_pair_runs(self.d, f'{alt}/euclidean', [(i, STRUCTURED_GS) for i in TRAINING_SEEDS])
        r_off = _seed_pair_runs(self.d, f'{alt}/randomized', _DIAG)
        cost_on = _cell_mean_moran(self.d, e_on) - _cell_mean_moran(self.d, r_on)
        cost_off = _cell_mean_moran(self.d, e_off) - _cell_mean_moran(self.d, r_off)
        inter = cost_off - cost_on
        return {'tracts': list(tracts), 'n_tracts': len(tracts),
                'pairs': [list(p) for p in _DIAG],
                'per_tract': {f: float(inter[f]) for f in tracts},
                **lower_bound_test(tracts, _coherence_stat(inter), self.n_draws)}

    def h2(self, tracts, kind='T'):
        stat = _corr_contrast_stat(self.t, self.d.runs['sage/euclidean'], [DASYMETRIC], [kind])
        return {'tracts': list(tracts), 'n_tracts': len(tracts), 'accuracy': f'r_{kind}',
                **tost_test(tracts, stat, self.n_draws, [f'q_{kind}', f'd_r_{kind}'])}

    def s3(self, arch, tracts):
        stat = _corr_contrast_stat(self.t, self.d.runs[f'{arch}/euclidean'],
                                   self.d.runs[f'{arch}/randomized'], ['T', 'W'])
        return {'tracts': list(tracts), 'n_tracts': len(tracts),
                **two_sided(tracts, stat, self.n_draws, ['q_T', 'd_r_T', 'q_W', 'd_r_W'])}

    def coherence_contrast(self, cell_a, cell_b, tracts):
        diff = (_cell_mean_moran(self.d, self.d.runs[cell_a])
                - _cell_mean_moran(self.d, self.d.runs[cell_b]))
        return {'contrast': f'{cell_a} minus {cell_b}', 'tracts': list(tracts),
                'n_tracts': len(tracts), 'per_tract': {f: float(diff[f]) for f in tracts},
                **two_sided(tracts, _coherence_stat(diff), self.n_draws, ['coherence_difference'])}

    def s2(self):
        """within-tract permutation null on r_W for every available run and dasymetric."""
        rids = [r for c in self.d.runs for r in self.d.runs[c]]
        per_run, per_cell = {}, {}
        for i, rid in enumerate(rids):
            df = self.d.address[rid]
            score = self._fast_rw_score(df)
            obs_module = pm.within_tract_r(pm.bg_predictions(df, 'value'),
                                           self.d.inputs.reference, 'pearson').r
            if not abs(score(df) - obs_module) <= 1e-12:
                raise RuntimeError(f'{rid}: fast r_W {score(df)} != module {obs_module}')
            res = pi.within_tract_permutation_null(df, 'value', score, n_perm=self.n_perm,
                                                   seed=S2_SEED + i)
            per_run[rid] = {'observed_r_W': res.observed, 'p95': res.p95, 'exceeds': res.exceeds,
                            'seed': res.seed, 'n_perm': res.n_perm}
            cell = rid if rid == DASYMETRIC else rid.rsplit('/', 2)[0]
            c = per_cell.setdefault(cell, {'n_runs': 0, 'n_exceed': 0})
            c['n_runs'] += 1
            c['n_exceed'] += int(res.exceeds)
        return {'tracts': self.all, 'n_tracts': len(self.all), 'seed_base': S2_SEED,
                'n_perm': self.n_perm, 'quantile_method': pi.QUANTILE_METHOD,
                'per_cell': per_cell, 'per_run': per_run}

    def _fast_rw_score(self, df):
        """r_W from block-group means of a value column, block groups and tracts fixed."""
        tab = self.t
        pos = {b: i for i, b in enumerate(tab.bg)}
        keep_bg = np.array([tab.tract[i] in tab.rw_tracts for i in range(len(tab.bg))])
        codes = df['bg_geoid'].map(pos)
        mask = codes.notna().to_numpy()
        codes = codes[mask].astype(int).to_numpy()
        counts = np.bincount(codes, minlength=len(tab.bg))
        g = tab.tract[keep_bg]
        tdem = pm.demean_within(tab.t[keep_bg], g)

        def score(d):
            v = d['value'].to_numpy(float)[mask]
            means = np.bincount(codes, weights=v, minlength=len(tab.bg)) / counts
            return float(pm.correlation(pm.demean_within(means[keep_bg], g), tdem, 'pearson'))
        return score

    def run_all(self):
        R, A = self.rand_tracts, self.all
        excl_note = {'excluded_tracts': self.excluded}
        for a in ARCHS:
            self.run(f'H1/{a}', [f'{a}/euclidean', f'{a}/randomized'],
                     lambda a=a: self.coherence_cost(a, R), hypothesis='H1', **excl_note)
        self.run('H2', ['sage/euclidean'], lambda: self.h2(A), hypothesis='H2')
        self.run('H3', ['sage/euclidean', 'sage/randomized', 'sage_root_weight_false/euclidean',
                        'sage_root_weight_false/randomized'],
                 lambda: self.self_path_interaction('sage', 'sage_root_weight_false', R),
                 hypothesis='H3', **excl_note)
        self.run('S1', ['sage/euclidean'], lambda: self.h2(A, 'W'), secondary='S1',
                 note='reported as inconclusive with its supported margin when it is')
        self.run('S2', [], self.s2, secondary='S2')
        for a in ARCHS:
            self.run(f'S3/{a}', [f'{a}/euclidean', f'{a}/randomized'], lambda a=a: self.s3(a, R),
                     secondary='S3', **excl_note)
        self.run('S4', ['gcn_gat/euclidean', 'gcn_gat/randomized', 'gcn_gat_residual/euclidean',
                        'gcn_gat_residual/randomized'],
                 lambda: self.self_path_interaction('gcn_gat', 'gcn_gat_residual', R),
                 secondary='S4', **excl_note)
        for a in ARCHS:
            ng, rd = f'{a}/no_graph', f'{a}/randomized'
            self.run(f'S5/{a}/no_graph_minus_randomized', [ng, rd],
                     lambda ng=ng, rd=rd: self.coherence_contrast(ng, rd, R), secondary='S5',
                     **excl_note)
            for s in ('euclidean', 'road_network', 'production'):
                sc = f'{a}/{s}'
                self.run(f'S5/{a}/{s}_minus_no_graph', [sc, ng],
                         lambda sc=sc, ng=ng: self.coherence_contrast(sc, ng, A), secondary='S5')
            self.run(f'S5/{a}/road_network_minus_euclidean', [f'{a}/road_network', f'{a}/euclidean'],
                     lambda a=a: self.coherence_contrast(f'{a}/road_network', f'{a}/euclidean', A),
                     secondary='S5')
        for label, drop in SENSITIVITY.items():
            for a in ARCHS:
                t1 = [f for f in R if f not in drop]
                self.run(f'sensitivity/{label}/H1/{a}', [f'{a}/euclidean', f'{a}/randomized'],
                         lambda a=a, t1=t1: self.coherence_cost(a, t1), sensitivity=label,
                         dropped=list(drop), **excl_note)
            t2 = [f for f in A if f not in drop]
            self.run(f'sensitivity/{label}/H2', ['sage/euclidean'], lambda t2=t2: self.h2(t2),
                     sensitivity=label, dropped=list(drop))
        sup = {'supplement': True,
               'label': ('20-tract version of a randomized contrast, including '
                         + ', '.join(self.excluded) + ', which received no randomization')
               if self.excluded else '20-tract version of a randomized contrast'}
        for a in ARCHS:
            self.run(f'supplement/H1/{a}', [f'{a}/euclidean', f'{a}/randomized'],
                     lambda a=a: self.coherence_cost(a, A), **sup)
            self.run(f'supplement/S3/{a}', [f'{a}/euclidean', f'{a}/randomized'],
                     lambda a=a: self.s3(a, A), **sup)
            ng, rd = f'{a}/no_graph', f'{a}/randomized'
            self.run(f'supplement/S5/{a}/no_graph_minus_randomized', [ng, rd],
                     lambda ng=ng, rd=rd: self.coherence_contrast(ng, rd, A), **sup)
        self.run('supplement/H3', ['sage/euclidean', 'sage/randomized',
                                   'sage_root_weight_false/euclidean', 'sage_root_weight_false/randomized'],
                 lambda: self.self_path_interaction('sage', 'sage_root_weight_false', A), **sup)
        self.run('supplement/S4', ['gcn_gat/euclidean', 'gcn_gat/randomized',
                                   'gcn_gat_residual/euclidean', 'gcn_gat_residual/randomized'],
                 lambda: self.self_path_interaction('gcn_gat', 'gcn_gat_residual', A), **sup)
        return self.results, self.skipped


def analyze(inputs, n_draws, n_perm, real):
    status = manifest_status(inputs)
    excluded, sources = derive_exclusion(inputs, status)
    if real and set(excluded) != EXPECTED_REAL_EXCLUSION:
        raise RuntimeError(f'real exclusion set {excluded} != {sorted(EXPECTED_REAL_EXCLUSION)}')
    data = RunData(inputs, status)
    table = BGTable(data)
    results, skipped = Analyses(data, table, excluded, n_draws, n_perm).run_all()
    return {
        'preregistration': 'v4.2',
        'bootstrap': {'n_draws': n_draws, 'seed': BOOTSTRAP_SEED, 'quantile_method': pi.QUANTILE_METHOD},
        'jackknife': {'z': JK_Z, 'bias_correction': False},
        's2': {'n_perm': n_perm, 'seed_base': S2_SEED},
        'thresholds': {'lower_bound': LOWER_BOUND_THRESHOLD, 'tost_margin': TOST_MARGIN,
                       'one_sided_level': ONE_SIDED_LEVEL, 'two_sided_level': TWO_SIDED_LEVEL},
        'reference_version': REFERENCE_VERSION,
        'n_tracts': len(inputs.tracts), 'tracts': list(inputs.tracts),
        'n_scored_block_groups': int(len(table.bg)),
        'excluded_tracts': excluded, 'exclusion_sources': sources,
        'cells': status,
        'pending_cells': sorted(c for c, s in status.items() if s['status'] == 'pending'),
        'skipped_analyses': skipped,
        'versions': _versions(),
        'analyses': results,
    }


# ---------------------------------------------------------------------------
# guard and entry point
# ---------------------------------------------------------------------------

def confirmatory_guard(repo_root=REPO_ROOT):
    """(ok, message): tag is an ancestor of HEAD and the working tree is clean."""
    r = _git('merge-base', '--is-ancestor', TAG, 'HEAD', cwd=repo_root)
    if r.returncode != 0:
        return False, f'refused: {TAG} is not an ancestor of HEAD ({r.stderr.strip() or r.returncode})'
    s = _git('status', '--porcelain', cwd=repo_root)
    if s.returncode != 0 or s.stdout.strip():
        return False, 'refused: working tree is not clean'
    return True, 'guard passed'


def _outside_results(path):
    p = Path(path).resolve()
    res = (REPO_ROOT / 'data' / 'results').resolve()
    return res not in p.parents and p != res


def _write(out_dir, payload):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / 'confirmatory.json'
    path.write_text(json.dumps(payload, indent=1, sort_keys=True, allow_nan=False) + '\n')
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--synthetic', metavar='DIR')
    mode.add_argument('--dry-run-real', action='store_true')
    mode.add_argument('--confirmatory', action='store_true')
    ap.add_argument('--out')
    ap.add_argument('--draws', type=int)
    ap.add_argument('--n-perm', type=int)
    ap.add_argument('--check-guard-only', action='store_true')
    args = ap.parse_args(argv)

    if (args.draws is not None or args.n_perm is not None) and not args.synthetic:
        ap.error('--draws and --n-perm are allowed with --synthetic only')
    if args.check_guard_only and not args.confirmatory:
        ap.error('--check-guard-only goes with --confirmatory')

    if args.synthetic:
        if not args.out:
            ap.error('--synthetic needs --out')
        if not _outside_results(args.out):
            ap.error('--out must sit outside data/results/')
        payload = {'mode': 'synthetic',
                   **analyze(synthetic_inputs(args.synthetic),
                             args.draws or pi.BOOTSTRAP_DRAWS, args.n_perm or N_PERM, real=False)}
        print(f'wrote {_write(args.out, payload)}')
        return 0

    if args.dry_run_real:
        payload = dry_run_real()
        text = json.dumps(payload, indent=1, sort_keys=True)
        if args.out:
            if not _outside_results(args.out):
                ap.error('--out must sit outside data/results/')
            _write(args.out, payload)
        print(text)
        return 0

    ok, msg = confirmatory_guard()
    print(msg)
    if not ok:
        return 2
    if args.check_guard_only:
        return 0
    head = _git('rev-parse', 'HEAD').stdout.strip()
    out_dir = CONFIRMATORY_DIR / head[:7]
    if out_dir.exists():
        raise SystemExit(f'refused: {out_dir} exists; results are never rewritten')
    payload = {'mode': 'confirmatory', 'head': head, 'tag': TAG,
               'tag_commit': _git('rev-list', '-n', '1', TAG).stdout.strip(),
               **analyze(real_inputs(), pi.BOOTSTRAP_DRAWS, N_PERM, real=True)}
    print(f'wrote {_write(out_dir, payload)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
