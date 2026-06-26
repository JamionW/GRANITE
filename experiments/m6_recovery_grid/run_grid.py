"""
M6 recovery grid runner.

Sweeps synthetic-testbed knobs (autocorr, snr, between_tract), runs GRANITE
plus Dasymetric and Pycnophylactic against known synthetic ground truth, and
records per-tract recovery r (corr(method_estimates, y_true)) per cell.

Signal source: latent throughout. Feature modes: coordinates_only (full
27-cell grid), full/random_noise/coords_plus_noise (autocorr diagonal only).
Architectures: sage (all modes), gcn_gat (coordinates_only diagonal only).
Seeds: 3 per cell [42, 17, 123].

Draw key: (signal_source, autocorr, snr, between_tract, seed).
Comparators run once per draw. GRANITE runs once per (feature_mode, arch)
per draw. Draw scratch markers enable resume.

Output:
  scratch markers  -> experiments/m6_recovery_grid/scratch/   (gitignored)
  summary CSV      -> data/results/m6_recovery_grid/recovery_grid.csv (tracked)

Usage:
    python experiments/m6_recovery_grid/run_grid.py --smoke --tracts-limit 3
    python experiments/m6_recovery_grid/run_grid.py --tracts-limit 5
    python experiments/m6_recovery_grid/run_grid.py
"""

import argparse
import copy
import csv
import json
import math
import os
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from granite.disaggregation.pipeline import GRANITEPipeline
from granite.evaluation.baselines import DasymetricDisaggregation, PycnophylacticDisaggregation
from granite.evaluation.spatial_diagnostics import SpatialLearningDiagnostics
from granite.models.gnn import set_random_seed
from granite.synthetic.generator import SyntheticTargetGenerator

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

SEEDS = [42, 17, 123]
AUTOCORRS = ['weak', 'medium', 'strong']
SNRS = ['low', 'medium', 'high']
BETWEEN_TRACTS = ['low', 'default', 'high']

N20_LIST_PATH = REPO_ROOT / 'output' / 'm2_n20_recovery' / 'summary' / 'n20_tract_list.txt'
BASE_CONFIG_PATH = (
    REPO_ROOT / 'experiments' / 'ablation' / '03_smoothness' / '02_default' / 'config_snapshot.yaml'
)
SCRATCH_DIR = REPO_ROOT / 'experiments' / 'm6_recovery_grid' / 'scratch'
PIPELINE_SCRATCH = SCRATCH_DIR / 'pipeline'
OUTPUT_CSV = REPO_ROOT / 'data' / 'results' / 'm6_recovery_grid' / 'recovery_grid.csv'

CSV_FIELDS = [
    'signal_source', 'feature_mode', 'autocorr', 'snr', 'between_tract',
    'arch', 'seed', 'tract_fips', 'method', 'recovery_r',
    'morans_i_true', 'morans_i_output', 'wtvr_achieved', 'generator_commit',
]

# smoke cell: diagonal (snr=medium, between_tract=default), autocorr=medium, first seed
SMOKE_AUTOCORR = 'medium'
SMOKE_SNR = 'medium'
SMOKE_BETWEEN_TRACT = 'default'
SMOKE_SEED = SEEDS[0]  # 42

_SPATIAL_DIAG = SpatialLearningDiagnostics(verbose=False)


# ---------------------------------------------------------------------------
# grid construction
# ---------------------------------------------------------------------------

def _build_grid():
    """
    Return list of draw specs, one per (autocorr, snr, between_tract, seed).

    Each spec has:
        signal_source, autocorr, snr, between_tract, seed, fm_arch_list.

    fm_arch_list specifies which (feature_mode, arch) GRANITE runs belong to
    this draw. Comparators run once per draw regardless.

    Grid encoding:
      coordinates_only/sage: full 27-cell grid (all autocorr x snr x between_tract).
      On diagonal (snr=medium, between_tract=default):
        + coordinates_only/gcn_gat
        + full/sage, random_noise/sage, coords_plus_noise/sage
      full/random_noise/coords_plus_noise are diagonal-only (snr=medium, between_tract=default).
    """
    draws = []
    for autocorr in AUTOCORRS:
        for snr in SNRS:
            for between_tract in BETWEEN_TRACTS:
                on_diagonal = (snr == 'medium' and between_tract == 'default')
                fm_arch_list = [('coordinates_only', 'sage')]
                if on_diagonal:
                    fm_arch_list.append(('coordinates_only', 'gcn_gat'))
                    fm_arch_list.append(('full', 'sage'))
                    fm_arch_list.append(('random_noise', 'sage'))
                    fm_arch_list.append(('coords_plus_noise', 'sage'))
                for seed in SEEDS:
                    draws.append({
                        'signal_source': 'latent',
                        'autocorr': autocorr,
                        'snr': snr,
                        'between_tract': between_tract,
                        'seed': seed,
                        'fm_arch_list': fm_arch_list,
                    })
    return draws


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _draw_key(draw_spec):
    return (
        f"{draw_spec['signal_source']}__{draw_spec['autocorr']}"
        f"__{draw_spec['snr']}__{draw_spec['between_tract']}__{draw_spec['seed']}"
    )


def _marker_path(draw_key):
    return SCRATCH_DIR / f"draw_{draw_key}.json"


def _is_done(draw_key):
    return _marker_path(draw_key).exists()


def _json_default(obj):
    if isinstance(obj, float) and math.isnan(obj):
        return None
    raise TypeError(f"not JSON serializable: {type(obj)}")


def _write_marker(draw_key, draw_spec, rows, gen_result, generator_commit,
                  y_min=None, y_max=None):
    marker = {
        'draw_key': draw_key,
        'signal_source': draw_spec['signal_source'],
        'autocorr': draw_spec['autocorr'],
        'snr': draw_spec['snr'],
        'between_tract': draw_spec['between_tract'],
        'seed': draw_spec['seed'],
        'generator_output_dir': gen_result['output_dir'],
        'wtvr_achieved': gen_result['diagnostics']['wtvr_achieved'],
        'morans_i_achieved': gen_result['diagnostics']['morans_i_achieved'],
        'generator_commit': generator_commit,
        'ytrue_rescale_min': y_min,
        'ytrue_rescale_max': y_max,
        'rows': rows,
    }
    _marker_path(draw_key).write_text(
        json.dumps(marker, indent=2, default=_json_default)
    )


def _load_tract_list():
    if not N20_LIST_PATH.exists():
        raise FileNotFoundError(f"n20 tract list not found: {N20_LIST_PATH}")
    with open(N20_LIST_PATH) as f:
        return [line.strip() for line in f if line.strip()]


def _load_base_config():
    with open(BASE_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f) or {}
    for k in ('data', 'model', 'training', 'processing', 'features',
              'recovery', 'validation', 'norm_layers'):
        cfg.setdefault(k, {})
    cfg.get('training', {}).pop('smoothness_weight', None)
    cfg['data']['target'] = 'svi'
    cfg['data']['neighbor_tracts'] = 0
    cfg['data']['state_fips'] = '47'
    cfg['data']['county_fips'] = '065'
    cfg['processing']['skip_importance'] = True
    cfg['processing']['verbose'] = False
    cfg['processing']['enable_caching'] = True
    cfg['features']['feature_standardization'] = 'per_tract'
    cfg['training']['constraint_mode'] = 'soft'
    cfg['training']['apply_post_correction'] = True
    return cfg


def _git_sha():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=str(REPO_ROOT)
        ).decode().strip()
    except Exception:
        return 'unknown'


def _compute_morans_i_true(addr_df, fips):
    """Moran's I of y_true for one tract using generator UTM coordinates."""
    tract_df = addr_df[addr_df['fips'] == fips]
    if len(tract_df) < 9:
        return float('nan')
    xy = tract_df[['x', 'y']].values
    vals = tract_df['y_true'].values
    try:
        return float(_SPATIAL_DIAG.compute_spatial_autocorrelation(vals, xy, k_neighbors=8))
    except Exception:
        return float('nan')


def _compute_morans_i_output(predictions, address_gdf):
    """Moran's I of GRANITE output using EPSG:4326 coordinates (matches 05b ruler)."""
    if len(predictions) < 9:
        return float('nan')
    coords = np.array([[g.x, g.y] for g in address_gdf.geometry])
    try:
        return float(_SPATIAL_DIAG.compute_spatial_autocorrelation(
            predictions, coords, k_neighbors=8))
    except Exception:
        return float('nan')


def _recovery_r(predictions, y_true):
    """Pearson r between predictions and y_true."""
    p = np.asarray(predictions, dtype=float)
    t = np.asarray(y_true, dtype=float)
    valid = np.isfinite(p) & np.isfinite(t)
    if valid.sum() < 2:
        return float('nan')
    return float(np.corrcoef(p[valid], t[valid])[0, 1])


def _ceiling_gbm_r(coords, y_true):
    """
    5-fold CV GBM recovery_r using z-scored coordinates as the only features.

    coords : (N, 2) array of UTM zone 16N meters [x, y] for one tract (EPSG:32616)
    y_true : (N,) array of rescaled y_true aligned to coords rows

    GBM settings match the ceiling probe: n_estimators=200, max_depth=4,
    learning_rate=0.05 (probe used these; recorded here as the canonical choice).
    """
    coords = np.asarray(coords, dtype=float)
    y = np.asarray(y_true, dtype=float)
    valid = np.isfinite(coords).all(axis=1) & np.isfinite(y)
    if valid.sum() < 10:
        return float('nan')
    X = coords[valid]
    y = y[valid]
    # z-score per tract so GBM thresholds are on comparable scale
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-12] = 1.0
    X = (X - mu) / sd
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    preds = np.full(len(y), np.nan)
    for train_idx, val_idx in kf.split(X):
        gbm = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
        )
        gbm.fit(X[train_idx], y[train_idx])
        preds[val_idx] = gbm.predict(X[val_idx])
    return _recovery_r(preds, y)


def _align_ytrue(addr_df, fips, address_gdf):
    """
    Return y_true array aligned to address_gdf row order.
    Match on hash if available; fall back to position order.
    Generator uses address_hash column; pipeline uses hash column.
    """
    tract_gen = addr_df[addr_df['fips'] == fips].copy()
    if 'hash' in address_gdf.columns and 'address_hash' in tract_gen.columns:
        gen_map = tract_gen.set_index('address_hash')['y_true']
        hashes = address_gdf['hash'].values
        try:
            aligned = gen_map.reindex(hashes).values.astype(float)
            n_valid = int(np.isfinite(aligned).sum())
            if n_valid > len(aligned) // 2:
                return aligned
        except Exception:
            pass
    # fall back: position order
    y = tract_gen['y_true'].values.astype(float)
    n_pred = len(address_gdf)
    if len(y) == n_pred:
        return y
    if len(y) > n_pred:
        return y[:n_pred]
    return np.pad(y, (0, n_pred - len(y)), constant_values=float('nan'))


def _make_run_cfg(cfg_base, fips, feature_mode, arch):
    """Return a shallow-copied config for one GRANITE run."""
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in cfg_base.items()}
    cfg['data'] = dict(cfg_base['data'])
    cfg['model'] = dict(cfg_base['model'])
    cfg['processing'] = dict(cfg_base['processing'])
    cfg['data']['target_fips'] = fips
    cfg['feature_mode'] = feature_mode
    cfg['model']['architecture'] = arch
    return cfg


def _assemble_csv(completed_draw_keys):
    """Read all draw markers and write summary CSV."""
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for draw_key in completed_draw_keys:
        mp = _marker_path(draw_key)
        if not mp.exists():
            continue
        try:
            marker = json.loads(mp.read_text())
        except Exception as e:
            print(f"[m6] WARNING: failed to read marker {mp}: {e}")
            continue
        for row in marker.get('rows', []):
            all_rows.append(row)

    if not all_rows:
        print('[m6] WARNING: no rows to assemble into CSV')
        return

    tmp_path = OUTPUT_CSV.parent / (OUTPUT_CSV.name + '.tmp')
    with open(tmp_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in all_rows:
            csv_row = {}
            for k in CSV_FIELDS:
                v = row.get(k, '')
                if isinstance(v, float) and math.isnan(v):
                    v = ''
                elif v is None:
                    v = ''
                csv_row[k] = v
            writer.writerow(csv_row)
    os.replace(tmp_path, OUTPUT_CSV)

    print(f"[m6] CSV written: {OUTPUT_CSV} ({len(all_rows)} rows)")


# ---------------------------------------------------------------------------
# single draw execution
# ---------------------------------------------------------------------------

def run_draw(draw_spec, tract_list, cfg_base, pipeline, data, generator_commit):
    """
    Execute one draw: generate y_true once, run comparators and GRANITE.

    For each tract in tract_list:
      - run comparators (Dasymetric, Pycnophylactic) once per tract per draw
      - run GRANITE for each (feature_mode, arch) in draw_spec['fm_arch_list']

    Returns (list of CSV row dicts, gen_result dict).
    """
    draw_key = _draw_key(draw_spec)
    print(f"\n[m6] draw {draw_key}")

    gen_params = {
        'signal_source': draw_spec['signal_source'],
        'spatial_autocorrelation': draw_spec['autocorr'],
        'snr': draw_spec['snr'],
        'between_tract': draw_spec['between_tract'],
        'tract_list_source': 'auto',
    }
    set_random_seed(draw_spec['seed'])
    print(f"[m6]   generating (seed={draw_spec['seed']}, autocorr={draw_spec['autocorr']}, "
          f"snr={draw_spec['snr']}, between={draw_spec['between_tract']})")
    gen = SyntheticTargetGenerator(seed=draw_spec['seed'], params=gen_params)
    gen_result = gen.generate()

    addr_df = gen_result['addresses']          # fips, address_hash, x, y, y_true

    # rescale y_true globally into [0,1] so tract-mean constraints are in domain.
    # map is over all addresses in the draw (all tracts, not limited by --tracts-limit).
    y_min = float(np.nanmin(addr_df['y_true'].values))
    y_max = float(np.nanmax(addr_df['y_true'].values))
    if (y_max - y_min) < 1e-9:
        raise ValueError(
            f"draw {draw_key}: degenerate y_true range ({y_min:.4f}, {y_max:.4f})"
        )
    addr_df = addr_df.copy()
    addr_df['y_true'] = (addr_df['y_true'] - y_min) / (y_max - y_min)
    tract_means = addr_df.groupby('fips')['y_true'].mean().to_dict()
    wtvr = float(gen_result['diagnostics']['wtvr_achieved'])
    print(f"[m6]   wtvr={wtvr:.4f}  morans_i_achieved={gen_result['diagnostics']['morans_i_achieved']:.4f}")
    print(f"[m6]   y_true rescaled: y_min={y_min:.4f}  y_max={y_max:.4f}")

    # fit baselines once per draw on real tract geometry
    dasy = DasymetricDisaggregation(ancillary_column='nlcd_impervious_pct')
    dasy.fit(data['tracts'], svi_column='RPL_THEMES')
    pycno = PycnophylacticDisaggregation(n_iterations=50, k_neighbors=8)
    pycno.fit(data['tracts'], svi_column='RPL_THEMES')

    rows = []

    for fips in tract_list:
        if fips not in tract_means:
            print(f"[m6]   {fips}: absent from generator tract_means, skipping")
            continue

        svi_constraint = float(tract_means[fips])
        mi_true = _compute_morans_i_true(addr_df, fips)

        address_gdf_ref = None    # set on first successful GRANITE run
        y_true_ref = None
        address_coords_ref = None

        for feature_mode, arch in draw_spec['fm_arch_list']:
            run_cfg = _make_run_cfg(cfg_base, fips, feature_mode, arch)
            pipeline.config = run_cfg
            pipeline.data_loader.config['processing'] = run_cfg['processing']

            try:
                result = pipeline._process_single_tract(
                    fips, data, svi_override=svi_constraint
                )
            except Exception as e:
                print(f"[m6]   ERROR {fips}/{feature_mode}/{arch}: {str(e)[:120]}")
                traceback.print_exc()
                continue

            if not result.get('success'):
                print(f"[m6]   FAILED {fips}/{feature_mode}/{arch}: "
                      f"{result.get('error', '?')[:120]}")
                continue

            preds = result['predictions']['mean'].values.astype(float)
            cur_gdf = result['address_gdf']

            # on first success: cache address_gdf for comparators and y_true alignment
            if address_gdf_ref is None:
                address_gdf_ref = cur_gdf
                y_true_ref = _align_ytrue(addr_df, fips, address_gdf_ref)
                address_coords_ref = np.array([[g.x, g.y] for g in address_gdf_ref.geometry])

                # run comparators once per draw per tract
                for method_name, baseline in [('dasymetric', dasy), ('pycnophylactic', pycno)]:
                    cmp_preds = baseline.disaggregate(
                        address_coords_ref, fips, svi_constraint,
                        address_gdf=address_gdf_ref,
                    )
                    rr = _recovery_r(cmp_preds, y_true_ref)
                    rows.append({
                        'signal_source': draw_spec['signal_source'],
                        'feature_mode': 'na',
                        'autocorr': draw_spec['autocorr'],
                        'snr': draw_spec['snr'],
                        'between_tract': draw_spec['between_tract'],
                        'arch': 'na',
                        'seed': draw_spec['seed'],
                        'tract_fips': fips,
                        'method': method_name,
                        'recovery_r': rr,
                        'morans_i_true': mi_true,
                        'morans_i_output': float('nan'),
                        'wtvr_achieved': wtvr,
                        'generator_commit': generator_commit,
                    })
                    print(f"[m6]   {method_name} {fips}: recovery_r={rr:.4f}")

                # ceiling_gbm: supervised coordinate ceiling using 5-fold CV GBM
                # uses UTM (x, y) from the generator frame -- same space the GP drew in
                sub = addr_df[addr_df['fips'] == fips]
                ceiling_rr = _ceiling_gbm_r(sub[['x', 'y']].values, sub['y_true'].values)
                rows.append({
                    'signal_source': draw_spec['signal_source'],
                    'feature_mode': 'na',
                    'autocorr': draw_spec['autocorr'],
                    'snr': draw_spec['snr'],
                    'between_tract': draw_spec['between_tract'],
                    'arch': 'na',
                    'seed': draw_spec['seed'],
                    'tract_fips': fips,
                    'method': 'ceiling_gbm',
                    'recovery_r': ceiling_rr,
                    'morans_i_true': mi_true,
                    'morans_i_output': float('nan'),
                    'wtvr_achieved': wtvr,
                    'generator_commit': generator_commit,
                })
                print(f"[m6]   ceiling_gbm {fips}: recovery_r={ceiling_rr:.4f}")

            rr = _recovery_r(preds, y_true_ref)
            mi_out = _compute_morans_i_output(preds, cur_gdf)
            rows.append({
                'signal_source': draw_spec['signal_source'],
                'feature_mode': feature_mode,
                'autocorr': draw_spec['autocorr'],
                'snr': draw_spec['snr'],
                'between_tract': draw_spec['between_tract'],
                'arch': arch,
                'seed': draw_spec['seed'],
                'tract_fips': fips,
                'method': 'granite',
                'recovery_r': rr,
                'morans_i_true': mi_true,
                'morans_i_output': mi_out,
                'wtvr_achieved': wtvr,
                'generator_commit': generator_commit,
            })
            print(f"[m6]   granite {fips} fm={feature_mode} arch={arch}: "
                  f"recovery_r={rr:.4f} mi_out={mi_out:.4f}")

    return rows, gen_result, y_min, y_max


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='M6 recovery grid runner')
    parser.add_argument(
        '--smoke', action='store_true',
        help=('run only the smoke cell: latent/medium/medium/default, '
              'first seed (42), all feature_modes on the diagonal'),
    )
    parser.add_argument(
        '--tracts-limit', type=int, default=None,
        help='limit tract list to first N tracts',
    )
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    ts_start = datetime.now()
    print(f"[m6] start {ts_start.strftime('%Y-%m-%d %H:%M:%S')}")

    generator_commit = _git_sha()
    print(f"[m6] generator_commit={generator_commit}")

    # load tract list
    tract_list_full = _load_tract_list()
    if args.tracts_limit is not None:
        tract_list = tract_list_full[:args.tracts_limit]
        print(f"[m6] tract list limited to first {args.tracts_limit}: {tract_list}")
    else:
        tract_list = tract_list_full
    print(f"[m6] n_tracts={len(tract_list)}")

    # build full grid
    all_draws = _build_grid()
    all_draw_keys = [_draw_key(d) for d in all_draws]
    print(f"[m6] full grid: {len(all_draws)} draws "
          f"({len(AUTOCORRS)} autocorr x {len(SNRS)} snr x {len(BETWEEN_TRACTS)} between_tract "
          f"x {len(SEEDS)} seeds)")

    if args.smoke:
        smoke_key = (
            f"latent__{SMOKE_AUTOCORR}__{SMOKE_SNR}"
            f"__{SMOKE_BETWEEN_TRACT}__{SMOKE_SEED}"
        )
        draws_to_run = [d for d in all_draws if _draw_key(d) == smoke_key]
        if not draws_to_run:
            print(f"[m6] HALT: smoke draw not found in grid (key={smoke_key})")
            sys.exit(1)
        print(f"[m6] smoke mode: 1 draw ({smoke_key}), "
              f"fm_arch_list={draws_to_run[0]['fm_arch_list']}")
    else:
        draws_to_run = all_draws

    # set up directories
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    PIPELINE_SCRATCH.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # load config and initialize pipeline
    cfg_base = _load_base_config()
    cfg_init = _make_run_cfg(cfg_base, tract_list[0], 'full', 'sage')
    cfg_init['processing']['verbose'] = args.verbose

    print('[m6] initializing pipeline...')
    pipeline = GRANITEPipeline(cfg_init, output_dir=str(PIPELINE_SCRATCH))
    pipeline.verbose = args.verbose

    print('[m6] loading spatial data...')
    try:
        data = pipeline._load_spatial_data()
    except Exception as e:
        print(f"[m6] HALT: spatial data loading failed: {e}")
        sys.exit(1)

    # main loop
    completed_draw_keys = []
    for draw_spec in draws_to_run:
        draw_key = _draw_key(draw_spec)

        if _is_done(draw_key):
            print(f"[m6] draw {draw_key}: already done, skipping")
            completed_draw_keys.append(draw_key)
            continue

        t0 = datetime.now()
        try:
            rows, gen_result, y_min, y_max = run_draw(
                draw_spec, tract_list, cfg_base, pipeline, data, generator_commit
            )
        except Exception as e:
            print(f"[m6] draw {draw_key} FAILED: {e}")
            traceback.print_exc()
            continue

        _write_marker(draw_key, draw_spec, rows, gen_result, generator_commit,
                      y_min=y_min, y_max=y_max)
        completed_draw_keys.append(draw_key)
        elapsed = (datetime.now() - t0).total_seconds()
        n_granite = sum(1 for r in rows if r['method'] == 'granite')
        n_cmp = sum(1 for r in rows if r['method'] != 'granite')
        print(
            f"[m6] draw {draw_key} done in {elapsed:.0f}s: "
            f"{n_granite} granite rows, {n_cmp} comparator rows, marker written"
        )
        # reassemble after each draw so an interrupted run leaves a current CSV
        _assemble_csv(completed_draw_keys)

    # final reassemble (no-op if all draws called it above)
    _assemble_csv(completed_draw_keys)

    ts_end = datetime.now()
    elapsed_total = (ts_end - ts_start).total_seconds() / 60
    print(f"\n[m6] complete in {elapsed_total:.1f} min")
    if not args.smoke:
        n_done = len(completed_draw_keys)
        n_total = len(all_draws)
        print(f"[m6] {n_done}/{n_total} draws done")
        if n_done < n_total:
            print(f"[m6] {n_total - n_done} draws pending; rerun to resume")


if __name__ == '__main__':
    main()
