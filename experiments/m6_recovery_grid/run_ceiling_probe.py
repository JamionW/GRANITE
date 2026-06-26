"""
M6 ceiling probe.

Part A: 5-fold CV GBM predicting y_true from z-scored (x, y) per tract.
        This is the supervised coordinate ceiling -- how well any nonlinear
        coordinate model can recover within-tract structure.

Part B: GRANITE coordinates_only/sage recovery_r.
        Medium pulled from existing CSV; weak and strong run here.

Draws: seed=42, snr=medium, between_tract=default.
Tracts: 47065000600, 47065000700, 47065001200.

Usage:
    python experiments/m6_recovery_grid/run_ceiling_probe.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from granite.disaggregation.pipeline import GRANITEPipeline
from granite.models.gnn import set_random_seed

SMOKE_TRACTS = ['47065000600', '47065000700', '47065001200']

DRAWS = {
    'weak':   REPO_ROOT / 'data/synthetic/run_20260624_170206',
    'medium': REPO_ROOT / 'data/synthetic/run_20260624_121918',
    'strong': REPO_ROOT / 'data/synthetic/run_20260624_170403',
}

MEDIUM_CSV  = REPO_ROOT / 'data/results/m6_recovery_grid/recovery_grid.csv'
BASE_CONFIG = (
    REPO_ROOT / 'experiments/ablation/03_smoothness/02_default/config_snapshot.yaml'
)
PIPELINE_SCRATCH = REPO_ROOT / 'experiments/m6_recovery_grid/scratch/pipeline'
RESULTS_PATH = REPO_ROOT / 'experiments/m6_recovery_grid/ceiling_probe_results.json'


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def _load_draw(draw_dir):
    """Load addresses, globally rescale y_true to [0,1], z-score x/y."""
    df = pd.read_csv(draw_dir / 'addresses.csv', dtype={'fips': str})
    y_min = float(df['y_true'].min())
    y_max = float(df['y_true'].max())
    df = df.copy()
    df['y_scaled'] = (df['y_true'] - y_min) / (y_max - y_min)
    x_mean, x_std = df['x'].mean(), df['x'].std()
    y_mean, y_std = df['y'].mean(), df['y'].std()
    df['x_z'] = (df['x'] - x_mean) / x_std
    df['y_z'] = (df['y'] - y_mean) / y_std
    return df, y_min, y_max


def _recovery_r(preds, y_true):
    p = np.asarray(preds, dtype=float)
    t = np.asarray(y_true, dtype=float)
    valid = np.isfinite(p) & np.isfinite(t)
    if valid.sum() < 2:
        return float('nan')
    return float(np.corrcoef(p[valid], t[valid])[0, 1])


# ---------------------------------------------------------------------------
# part a: GBM ceiling
# ---------------------------------------------------------------------------

def _gbm_cv_r(sub):
    """5-fold CV GBM r on z-scored (x_z, y_z) -> y_scaled."""
    X = sub[['x_z', 'y_z']].values
    y = sub['y_scaled'].values
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    preds = np.full(len(y), np.nan)
    for train_idx, val_idx in kf.split(X):
        gbm = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
        )
        gbm.fit(X[train_idx], y[train_idx])
        preds[val_idx] = gbm.predict(X[val_idx])
    return _recovery_r(preds, y)


def run_part_a():
    print('[ceiling_probe] part_a: GBM supervised coordinate ceiling')
    results = {}
    for level in ['weak', 'medium', 'strong']:
        df, _, _ = _load_draw(DRAWS[level])
        tract_rs = {}
        for fips in SMOKE_TRACTS:
            sub = df[df['fips'] == fips]
            if len(sub) < 10:
                tract_rs[fips] = float('nan')
                print(f'  {level} {fips}: too few rows')
                continue
            r = _gbm_cv_r(sub)
            tract_rs[fips] = r
            print(f'  {level} {fips}: n={len(sub)}, ceiling_r={r:.4f}')
        results[level] = tract_rs
    return results


# ---------------------------------------------------------------------------
# part b: GRANITE coordinates_only/sage
# ---------------------------------------------------------------------------

def _medium_from_csv():
    df = pd.read_csv(MEDIUM_CSV)
    mask = (
        (df['feature_mode'] == 'coordinates_only') &
        (df['arch'] == 'sage') &
        (df['autocorr'] == 'medium') &
        (df['method'] == 'granite') &
        (df['seed'].astype(int) == 42)
    )
    rows = df[mask]
    result = {}
    for _, row in rows.iterrows():
        result[str(row['tract_fips'])] = float(row['recovery_r'])
    return result


def _load_base_config():
    with open(BASE_CONFIG) as f:
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


def _make_cfg(cfg_base, fips):
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in cfg_base.items()}
    cfg['data'] = dict(cfg_base['data'])
    cfg['model'] = dict(cfg_base['model'])
    cfg['processing'] = dict(cfg_base['processing'])
    cfg['data']['target_fips'] = fips
    cfg['feature_mode'] = 'coordinates_only'
    cfg['model']['architecture'] = 'sage'
    return cfg


def _align_ytrue(addr_df, fips, address_gdf):
    """Align y_scaled to pipeline address_gdf row order via address_hash."""
    tract_gen = addr_df[addr_df['fips'] == fips].copy()
    if 'hash' in address_gdf.columns and 'address_hash' in tract_gen.columns:
        gen_map = tract_gen.set_index('address_hash')['y_scaled']
        hashes = address_gdf['hash'].values
        try:
            aligned = gen_map.reindex(hashes).values.astype(float)
            n_valid = int(np.isfinite(aligned).sum())
            if n_valid > len(aligned) // 2:
                return aligned
        except Exception:
            pass
    # fall back: position order
    y = tract_gen['y_scaled'].values.astype(float)
    n_pred = len(address_gdf)
    if len(y) == n_pred:
        return y
    if len(y) > n_pred:
        return y[:n_pred]
    return np.pad(y, (0, n_pred - len(y)), constant_values=float('nan'))


def _run_granite_level(level, draw_dir, pipeline, data, cfg_base):
    """Run coordinates_only/sage on 3 tracts for one autocorr level."""
    df, _, _ = _load_draw(draw_dir)
    tract_means = df.groupby('fips')['y_scaled'].mean().to_dict()
    results = {}
    for fips in SMOKE_TRACTS:
        if fips not in tract_means:
            print(f'  {level} {fips}: not in tract_means, skipping')
            results[fips] = float('nan')
            continue
        svi_constraint = float(tract_means[fips])
        run_cfg = _make_cfg(cfg_base, fips)
        pipeline.config = run_cfg
        pipeline.data_loader.config['processing'] = run_cfg['processing']
        set_random_seed(42)
        try:
            result = pipeline._process_single_tract(fips, data, svi_override=svi_constraint)
        except Exception as e:
            print(f'  {level} {fips}: error {str(e)[:120]}')
            results[fips] = float('nan')
            continue
        if not result.get('success'):
            print(f'  {level} {fips}: failed {result.get("error","?")[:80]}')
            results[fips] = float('nan')
            continue
        preds = result['predictions']['mean'].values.astype(float)
        y_true = _align_ytrue(df, fips, result['address_gdf'])
        rr = _recovery_r(preds, y_true)
        print(f'  {level} {fips}: constraint={svi_constraint:.3f}  recovery_r={rr:.4f}')
        results[fips] = rr
    return results


def run_part_b():
    print('[ceiling_probe] part_b: GRANITE coordinates_only/sage')
    print('[ceiling_probe]   medium: from existing CSV')
    medium_r = _medium_from_csv()
    for fips in SMOKE_TRACTS:
        print(f'  medium {fips}: {medium_r.get(fips, float("nan")):.4f}')

    print('[ceiling_probe]   loading pipeline...')
    PIPELINE_SCRATCH.mkdir(parents=True, exist_ok=True)
    cfg_base = _load_base_config()
    cfg_init = _make_cfg(cfg_base, SMOKE_TRACTS[0])
    pipeline = GRANITEPipeline(cfg_init, output_dir=str(PIPELINE_SCRATCH))
    pipeline.verbose = False
    data = pipeline._load_spatial_data()

    part_b = {'medium': medium_r}
    for level in ['weak', 'strong']:
        print(f'[ceiling_probe]   running {level}...')
        part_b[level] = _run_granite_level(level, DRAWS[level], pipeline, data, cfg_base)
    return part_b


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def _mean(d):
    vals = [d.get(f, float('nan')) for f in SMOKE_TRACTS]
    return float(np.nanmean(vals))


def print_table(part_a, part_b):
    print()
    print('=== M6 CEILING PROBE RESULTS ===')
    print()
    print(f"{'autocorr':<10} | {'supervised ceiling r':<22} | {'GRANITE coords_only r':<22}")
    print(f"{'':10} | {'3-tract mean':<22} | {'3-tract mean':<22}")
    print('-' * 62)
    for level in ['weak', 'medium', 'strong']:
        a = _mean(part_a[level])
        b = _mean(part_b[level])
        print(f'{level:<10} | {a:<22.4f} | {b:<22.4f}')
    print()
    print('Per-tract detail:')
    print(f"{'autocorr':<10} {'tract':<15} {'ceiling_r':<12} {'granite_r':<12}")
    print('-' * 50)
    for level in ['weak', 'medium', 'strong']:
        for fips in SMOKE_TRACTS:
            ar = part_a[level].get(fips, float('nan'))
            br = part_b[level].get(fips, float('nan'))
            print(f'{level:<10} {fips:<15} {ar:<12.4f} {br:<12.4f}')
    print()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    part_a = run_part_a()
    part_b = run_part_b()

    print_table(part_a, part_b)

    # save JSON for session log
    out = {
        'part_a': part_a,
        'part_b': {k: v for k, v in part_b.items()},
    }
    RESULTS_PATH.write_text(json.dumps(out, indent=2, default=str))
    print(f'[ceiling_probe] results saved -> {RESULTS_PATH}')


if __name__ == '__main__':
    main()
