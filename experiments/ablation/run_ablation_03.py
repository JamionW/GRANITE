"""
run_ablation_03: cross-tract smoothness loss weight sweep (Step 3).

Holds all other settings at the 2a state (per-tract z-score, input LayerNorm,
BatchNorm on conv layers) and varies only smoothness_weight in the
MultiTractGNNTrainer loss. Variant 02_default runs first as a sanity check
that it reproduces 2a within 1% on within-tract std and BG r.

Variants:
  00_off:      smoothness_weight = 0.0
  01_quarter:  smoothness_weight = 0.025
  02_default:  smoothness_weight = 0.1  (current hardcoded value)
  03_double:   smoothness_weight = 0.2
  04_five_x:   smoothness_weight = 0.5

Run order: 02_default first, then 00_off, then 01_quarter, 03_double, 04_five_x.

Usage:
    python experiments/ablation/run_ablation_03.py [--verbose] [--only 02_default]
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from granite.disaggregation.pipeline import GRANITEPipeline
from granite.data.block_group_loader import BlockGroupLoader
from granite.validation.block_group_validation import BlockGroupValidator
from granite.evaluation.spatial_diagnostics import SpatialLearningDiagnostics
from granite.models.gnn import set_random_seed

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
ABLATION_ROOT = Path(__file__).resolve().parent
STEP3_ROOT    = ABLATION_ROOT / '03_smoothness'
CONFIG_PATH   = REPO_ROOT / 'config.yaml'
BG_SVI_PATH   = REPO_ROOT / 'data' / 'processed' / 'national_bg_svi.csv'
TRACT_SEL     = ABLATION_ROOT / '00_baseline' / 'tract_selection.txt'
TRACT_INV     = REPO_ROOT / 'tract_inventory.csv'

ARCHITECTURES = ['sage', 'gcn_gat']
ARCH_LABELS   = {'sage': 'GRANITE-SAGE', 'gcn_gat': 'GRANITE-GCNGAT'}
SEED          = 42
MIN_ADDRESSES_PER_BG = 3
MIN_ADDRESSES_POOLED = 10

# 2a reference values for sanity check and delta table
REF_2A = {
    'sage':    {'spatial_std_mean': 0.0823208,  'morans_i_mean': 0.8775895,
                'bg_r': 0.7536965003138819},
    'gcn_gat': {'spatial_std_mean': 0.08142490, 'morans_i_mean': 0.84904515,
                'bg_r': 0.7664314601020286},
}

# stop thresholds
CONSTRAINT_MAX  = 1e-4
SPATIAL_STD_MAX = 0.3   # flag only
BG_R_MIN        = 0.6   # flag and continue
SANITY_TOL      = 0.01  # 1% tolerance for 02_default vs 2a sanity check

# variant definitions; run order is the list order here
VARIANTS = [
    {
        'key':             '02_default',
        'dir':             '02_default',
        'label':           '02 default (w=0.1)',
        'smoothness_weight': 0.1,
        'description':     'smoothness_weight=0.1 (current default, sanity check vs 2a)',
    },
    {
        'key':             '00_off',
        'dir':             '00_off',
        'label':           '00 off (w=0.0)',
        'smoothness_weight': 0.0,
        'description':     'smoothness_weight=0.0 (term fully disabled)',
    },
    {
        'key':             '01_quarter',
        'dir':             '01_quarter',
        'label':           '01 quarter (w=0.025)',
        'smoothness_weight': 0.025,
        'description':     'smoothness_weight=0.025 (quarter of default)',
    },
    {
        'key':             '03_double',
        'dir':             '03_double',
        'label':           '03 double (w=0.2)',
        'smoothness_weight': 0.2,
        'description':     'smoothness_weight=0.2 (double default)',
    },
    {
        'key':             '04_five_x',
        'dir':             '04_five_x',
        'label':           '04 five-x (w=0.5)',
        'smoothness_weight': 0.5,
        'description':     'smoothness_weight=0.5 (5x default)',
    },
]


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def _load_config():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f) or {}
    for k in ('data', 'model', 'training', 'processing', 'recovery',
              'validation', 'features', 'norm_layers'):
        cfg.setdefault(k, {})
    return cfg


def _load_tract_list():
    """Load 20-tract list from 00_baseline/tract_selection.txt."""
    lines = []
    with open(TRACT_SEL) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('source'):
                # handle yaml-style lines
                if ':' in line:
                    continue
                lines.append(line)
    if not lines:
        # fallback: parse the yaml block
        with open(TRACT_SEL) as f:
            content = f.read()
        import re
        fips_list = re.findall(r'^\s*-\s*["\']?(\d{11})["\']?', content, re.MULTILINE)
        if fips_list:
            return fips_list
        # last resort: tract_inventory.csv
        df = pd.read_csv(TRACT_INV)
        df['fips'] = df['fips'].astype(str).str.strip()
        return df['fips'].tolist()
    return lines


def _load_bg_gdf():
    loader = BlockGroupLoader(data_dir=str(REPO_ROOT / 'data'), verbose=False)
    bg_gdf = loader.get_block_groups_with_demographics('47', '065', svi_ranking_scope='national')
    if bg_gdf is None or len(bg_gdf) == 0:
        raise RuntimeError('BlockGroupLoader returned empty GeoDataFrame')
    if bg_gdf.crs is None:
        bg_gdf = bg_gdf.set_crs('EPSG:4326')
    elif bg_gdf.crs.to_epsg() != 4326:
        bg_gdf = bg_gdf.to_crs('EPSG:4326')
    return bg_gdf


def _aggregate_to_bg(validator, address_gdf, preds, min_addresses):
    if address_gdf.crs is None:
        address_gdf = address_gdf.set_crs('EPSG:4326')
    addresses_with_bg = validator._assign_to_block_groups(address_gdf)
    bg_agg = validator._aggregate_predictions(addresses_with_bg, preds)
    return bg_agg[bg_agg['n_addresses'] >= min_addresses].copy()


def _bg_metrics(bg_agg, bg_gdf):
    svi_lookup = bg_gdf[['GEOID', 'SVI']].dropna(subset=['SVI'])
    merged = bg_agg.merge(svi_lookup, on='GEOID', how='inner')
    n = len(merged)
    if n < 2:
        return {'bg_r': float('nan'), 'bg_rmse': float('nan'), 'n_bgs': n}
    p = merged['predicted_svi'].values.astype(float)
    t = merged['SVI'].values.astype(float)
    valid = np.isfinite(p) & np.isfinite(t)
    if valid.sum() < 2:
        return {'bg_r': float('nan'), 'bg_rmse': float('nan'), 'n_bgs': n}
    return {
        'bg_r':    float(np.corrcoef(p[valid], t[valid])[0, 1]),
        'bg_rmse': float(np.sqrt(np.mean((p[valid] - t[valid]) ** 2))),
        'n_bgs':   int(valid.sum()),
    }


def _compute_morans_i(predictions, address_gdf, k=8):
    try:
        coords = np.array([[g.x, g.y] for g in address_gdf.geometry])
        return SpatialLearningDiagnostics().compute_spatial_autocorrelation(
            predictions, coords, k_neighbors=k)
    except Exception:
        return float('nan')


def _spatial_std_slope(df, arch):
    sub = df[(df['architecture'] == arch) &
             (df['failure'].isna() | (df['failure'] == ''))].copy()
    if len(sub) < 3:
        return float('nan')
    try:
        return float(np.polyfit(sub['tract_svi'].values, sub['spatial_std'].values, 1)[0])
    except Exception:
        return float('nan')


def _safe_delta(a, b):
    try:
        if a is None or b is None:
            return None
        v = float(a) - float(b)
        return round(v, 6) if math.isfinite(v) else None
    except Exception:
        return None


def _git_sha():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=str(REPO_ROOT)
        ).decode().strip()
    except Exception:
        return 'unknown'


def _write_preflight(variant_dir):
    sha = _git_sha()
    (variant_dir / 'git_state.txt').write_text(sha + '\n')
    try:
        env_out = subprocess.check_output(
            ['pip', 'freeze'], stderr=subprocess.DEVNULL
        ).decode()
    except Exception:
        env_out = 'unavailable\n'
    (variant_dir / 'environment.txt').write_text(env_out)
    with open(CONFIG_PATH) as f:
        cfg_raw = f.read()
    (variant_dir / 'config_snapshot.yaml').write_text(cfg_raw)


# ---------------------------------------------------------------------------
# per-variant run
# ---------------------------------------------------------------------------

def run_variant(variant, cfg_base, data, bg_gdf, validator, tract_list, verbose):
    """Run one smoothness-weight variant. Returns (per_tract_df, agg_metrics, bg_validation, tract_signal, importance_agg)."""
    vdir    = STEP3_ROOT / variant['dir']
    res_dir = vdir / 'results'
    fig_dir = vdir / 'figures'
    fi_dir  = res_dir / 'feature_importance'
    fi_dir.mkdir(parents=True, exist_ok=True)

    _write_preflight(vdir)

    # if main results already exist, load them and skip the training loop
    _existing_csv = res_dir / 'per_tract_metrics.csv'
    _existing_agg = res_dir / 'aggregate_metrics.json'
    _existing_bgv = res_dir / 'block_group_validation.json'
    if _existing_csv.exists() and _existing_agg.exists() and _existing_bgv.exists():
        print(f'[{variant["key"]}] results already on disk, loading and regenerating figures/README')
        per_tract_df = pd.read_csv(_existing_csv)
        with open(_existing_agg) as _f:
            agg_metrics = json.load(_f)
        with open(_existing_bgv) as _f:
            bg_validation = json.load(_f)
        # rebuild tract_signal from per_tract_df
        tract_signal = {a: [] for a in ARCHITECTURES}
        for _, row in per_tract_df.iterrows():
            if (pd.isna(row.get('failure')) or row.get('failure') == '') and \
               math.isfinite(float(row.get('tract_mean_pred', float('nan')))):
                tract_signal[row['architecture']].append(
                    (float(row['tract_mean_pred']), float(row['tract_svi']))
                )
        cross_tract_r = {}
        for arch in ARCHITECTURES:
            sig = tract_signal[arch]
            if len(sig) >= 2:
                pt = np.array([x[0] for x in sig]); ac = np.array([x[1] for x in sig])
                v = np.isfinite(pt) & np.isfinite(ac)
                cross_tract_r[arch] = float(np.corrcoef(pt[v], ac[v])[0, 1]) if v.sum() >= 2 else float('nan')
            else:
                cross_tract_r[arch] = float('nan')
        # load importance if present
        importance_agg = {}
        for arch in ARCHITECTURES:
            short = 'sage' if arch == 'sage' else 'gcngat'
            fp = fi_dir / f'{short}_importance.csv'
            importance_agg[arch] = pd.read_csv(fp) if fp.exists() else pd.DataFrame()
        per_tract_bg = {}
        _generate_figures(per_tract_df, bg_validation, per_tract_bg, importance_agg, fig_dir, variant)
        _write_variant_readme(variant, vdir, per_tract_df, agg_metrics, bg_validation,
                              cross_tract_r, tract_list)
        print(f'[{variant["key"]}] regenerated from existing results')
        return per_tract_df, agg_metrics, bg_validation, cross_tract_r, importance_agg, tract_signal

    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in cfg_base.items()}
    cfg['training'] = dict(cfg_base.get('training', {}))
    cfg['training']['smoothness_weight'] = variant['smoothness_weight']
    # norm_layers: 2a state (input LN on, batchnorm on conv)
    cfg['norm_layers'] = {'input_layernorm': True, 'conv_norm_type': 'batchnorm'}
    cfg['features']['feature_standardization'] = 'per_tract'

    scratch = str(REPO_ROOT / 'output' / f'ablation_03_{variant["key"]}_scratch')
    os.makedirs(scratch, exist_ok=True)

    pipeline = GRANITEPipeline(cfg, output_dir=scratch)
    pipeline.verbose = verbose

    per_tract_rows = []
    pooled_addr    = {a: [] for a in ARCHITECTURES}
    pooled_preds   = {a: [] for a in ARCHITECTURES}
    imp_by_arch    = {a: [] for a in ARCHITECTURES}

    # per-tract signal tracking: tract mean prediction and actual SVI
    tract_signal = {a: [] for a in ARCHITECTURES}  # list of (tract_mean_pred, tract_actual_svi)

    total = len(ARCHITECTURES) * len(tract_list)
    done  = 0

    for arch in ARCHITECTURES:
        cfg['model']['architecture'] = arch
        pipeline.config = dict(cfg)
        pipeline.verbose = verbose

        for idx, fips in enumerate(tract_list):
            done += 1
            tag = f'[{variant["key"]}][{done}/{total}] {ARCH_LABELS[arch]} | {fips}'
            print(tag)
            t0 = time.time()

            cfg['data']['target_fips']             = fips
            pipeline.config['data']['target_fips']    = fips
            pipeline.config['model']['architecture']  = arch

            fail_row = {
                'fips': fips, 'architecture': arch, 'n_addresses': 0,
                'tract_svi': float('nan'), 'constraint_error': float('nan'),
                'spatial_std': float('nan'), 'morans_i': float('nan'),
                'bg_r': float('nan'), 'n_bgs': 0,
                'tract_mean_pred': float('nan'),
                'between_tract_variance': float('nan'),
                'runtime_s': 0, 'failure': '',
            }

            try:
                result = pipeline._process_single_tract(fips, data)
            except Exception as e:
                fail_row['failure'] = str(e)[:300]
                print(f'  ERROR: {fail_row["failure"]}')
                per_tract_rows.append(fail_row)
                continue

            if not result.get('success'):
                fail_row['failure'] = result.get('error', 'unknown')[:300]
                print(f'  FAILED: {fail_row["failure"]}')
                per_tract_rows.append(fail_row)
                continue

            runtime     = time.time() - t0
            address_gdf = result['address_gdf']
            preds_arr   = result['predictions']['mean'].values.astype(float)
            tract_svi   = float(result['tract_svi'])
            n_addresses = len(address_gdf)
            tract_mean_pred = float(np.mean(preds_arr))

            constr_err = abs(tract_mean_pred - tract_svi)
            sp_std     = float(np.std(preds_arr))
            morans_i   = _compute_morans_i(preds_arr, address_gdf)

            # stop conditions
            if not math.isfinite(constr_err) or not math.isfinite(sp_std):
                print(f'  STOP: non-finite metric on {fips}/{arch}')
                sys.exit(2)
            if constr_err > CONSTRAINT_MAX:
                print(f'  STOP: constraint_error={constr_err:.2e} > {CONSTRAINT_MAX:.0e} on {fips}/{arch}')
                sys.exit(2)
            if not math.isfinite(morans_i):
                morans_i = float('nan')

            try:
                bg_agg = _aggregate_to_bg(validator, address_gdf, preds_arr, MIN_ADDRESSES_PER_BG)
                bm = _bg_metrics(bg_agg, bg_gdf)
            except Exception as e:
                bm = {'bg_r': float('nan'), 'bg_rmse': float('nan'), 'n_bgs': 0}

            per_tract_rows.append({
                'fips': fips, 'architecture': arch, 'n_addresses': n_addresses,
                'tract_svi': tract_svi,
                'constraint_error': round(constr_err, 8),
                'spatial_std': round(sp_std, 6),
                'morans_i': round(morans_i, 6) if math.isfinite(morans_i) else float('nan'),
                'bg_r': bm['bg_r'], 'n_bgs': bm['n_bgs'],
                'tract_mean_pred': round(tract_mean_pred, 6),
                'between_tract_variance': float('nan'),  # filled after loop
                'runtime_s': round(runtime, 2),
                'failure': '',
            })

            pooled_addr[arch].append(address_gdf.copy())
            pooled_preds[arch].append(preds_arr)
            tract_signal[arch].append((tract_mean_pred, tract_svi))

            # feature importance
            try:
                fi = pipeline.analyze_feature_importance(result, n_repeats=5)
                if fi is not None:
                    perm = fi.get('permutation', {})
                    if isinstance(perm, dict) and 'feature_importance' in perm:
                        imp_df = perm['feature_importance'].copy()
                        imp_df['fips'] = fips
                        imp_df['architecture'] = arch
                        imp_by_arch[arch].append(imp_df)
            except Exception:
                pass

            flag = ''
            if sp_std > SPATIAL_STD_MAX:
                flag = f'FLAG: spatial_std={sp_std:.3f} > {SPATIAL_STD_MAX}'
                print(f'  {flag}')
            if math.isfinite(bm['bg_r']) and bm['bg_r'] < BG_R_MIN:
                print(f'  FLAG: per-tract bg_r={bm["bg_r"]:.3f} < {BG_R_MIN} (continuing)')

            if verbose:
                print(f'  constr_err={constr_err:.2e} sp_std={sp_std:.4f} '
                      f'morans_i={morans_i:.3f} bg_r={bm["bg_r"]:.3f} '
                      f'tract_mean={tract_mean_pred:.4f} t={runtime:.1f}s {flag}')

    # fill between_tract_variance per architecture (same value broadcast to all rows for that arch)
    per_tract_df = pd.DataFrame(per_tract_rows)
    for arch in ARCHITECTURES:
        sig = tract_signal[arch]
        if len(sig) >= 2:
            means = np.array([x[0] for x in sig])
            btv   = float(np.var(means))
        else:
            btv = float('nan')
        mask = (per_tract_df['architecture'] == arch) & \
               (per_tract_df['failure'].isna() | (per_tract_df['failure'] == ''))
        per_tract_df.loc[mask, 'between_tract_variance'] = btv

    # cross_tract_signal_r: Pearson r between tract mean predictions and actual SVIs
    cross_tract_r = {}
    for arch in ARCHITECTURES:
        sig = tract_signal[arch]
        if len(sig) >= 2:
            preds_t = np.array([x[0] for x in sig])
            actuals = np.array([x[1] for x in sig])
            valid   = np.isfinite(preds_t) & np.isfinite(actuals)
            if valid.sum() >= 2:
                cross_tract_r[arch] = float(np.corrcoef(preds_t[valid], actuals[valid])[0, 1])
            else:
                cross_tract_r[arch] = float('nan')
        else:
            cross_tract_r[arch] = float('nan')

    per_tract_df.to_csv(res_dir / 'per_tract_metrics.csv', index=False)

    # aggregate metrics
    agg_metrics = {}
    for arch in ARCHITECTURES:
        arch_df = per_tract_df[
            (per_tract_df['architecture'] == arch) &
            (per_tract_df['failure'].isna() | (per_tract_df['failure'] == ''))
        ]
        if not len(arch_df):
            agg_metrics[arch] = {}
            continue
        btv_vals = arch_df['between_tract_variance'].dropna()
        agg_metrics[arch] = {
            'n_tracts':                   int(len(arch_df)),
            'constraint_error_mean':      float(arch_df['constraint_error'].mean()),
            'constraint_error_median':    float(arch_df['constraint_error'].median()),
            'spatial_std_mean':           float(arch_df['spatial_std'].mean()),
            'spatial_std_median':         float(arch_df['spatial_std'].median()),
            'spatial_std_std':            float(arch_df['spatial_std'].std()),
            'morans_i_mean':   float(arch_df['morans_i'].dropna().mean())   if arch_df['morans_i'].notna().any()  else float('nan'),
            'morans_i_median': float(arch_df['morans_i'].dropna().median()) if arch_df['morans_i'].notna().any()  else float('nan'),
            'morans_i_std':    float(arch_df['morans_i'].dropna().std())    if arch_df['morans_i'].notna().any()  else float('nan'),
            'bg_r_mean':   float(arch_df['bg_r'].dropna().mean())   if arch_df['bg_r'].notna().any() else float('nan'),
            'bg_r_median': float(arch_df['bg_r'].dropna().median()) if arch_df['bg_r'].notna().any() else float('nan'),
            'between_tract_variance':     float(btv_vals.iloc[0]) if len(btv_vals) else float('nan'),
            'cross_tract_signal_r':       cross_tract_r.get(arch, float('nan')),
        }

    with open(res_dir / 'aggregate_metrics.json', 'w') as f:
        json.dump(agg_metrics, f, indent=2,
                  default=lambda x: None if isinstance(x, float) and math.isnan(x) else x)

    # pooled BG validation
    bg_validation = {}
    per_tract_bg  = {}

    for arch in ARCHITECTURES:
        if not pooled_addr[arch]:
            bg_validation[arch] = {'pearson_r': float('nan')}
            continue
        combined = pd.concat(pooled_addr[arch], ignore_index=True)
        if not isinstance(combined, gpd.GeoDataFrame):
            combined = gpd.GeoDataFrame(combined, geometry='geometry', crs='EPSG:4326')
        if combined.crs is None:
            combined = combined.set_crs('EPSG:4326')
        all_p = np.concatenate(pooled_preds[arch])
        try:
            bm_p = _bg_metrics(_aggregate_to_bg(validator, combined, all_p, MIN_ADDRESSES_POOLED), bg_gdf)
            svi_lk = bg_gdf[['GEOID', 'SVI']].dropna(subset=['SVI'])
            ms = _aggregate_to_bg(validator, combined, all_p, 1).merge(svi_lk, on='GEOID', how='inner')
        except Exception as e:
            bm_p = {'bg_r': float('nan'), 'bg_rmse': float('nan'), 'n_bgs': 0}
            ms   = pd.DataFrame()
            print(f'  WARNING: pooled BG validation failed for {arch}: {e}')
        bg_validation[arch] = {'pearson_r': bm_p['bg_r'], 'bg_rmse': bm_p['bg_rmse'], 'n_bgs': bm_p['n_bgs']}
        per_tract_bg[arch] = ms

    with open(res_dir / 'block_group_validation.json', 'w') as f:
        json.dump(bg_validation, f, indent=2,
                  default=lambda x: None if isinstance(x, float) and math.isnan(x) else x)

    # sanity check for 02_default: must reproduce 2a within 1% on spatial_std and bg_r
    if variant['key'] == '02_default':
        for arch in ARCHITECTURES:
            sa = agg_metrics.get(arch, {})
            ref = REF_2A[arch]
            std_val = sa.get('spatial_std_mean', float('nan'))
            bg_r_val = bg_validation.get(arch, {}).get('pearson_r', float('nan'))
            if math.isfinite(std_val) and math.isfinite(ref['spatial_std_mean']):
                rel_diff = abs(std_val - ref['spatial_std_mean']) / (abs(ref['spatial_std_mean']) + 1e-9)
                if rel_diff > SANITY_TOL:
                    print(f'  STOP: 02_default/{arch} spatial_std_mean={std_val:.5f} '
                          f'differs from 2a ref {ref["spatial_std_mean"]:.5f} by {rel_diff*100:.2f}% > 1%')
                    sys.exit(2)
            if math.isfinite(bg_r_val) and math.isfinite(ref['bg_r']):
                rel_diff = abs(bg_r_val - ref['bg_r']) / (abs(ref['bg_r']) + 1e-9)
                if rel_diff > SANITY_TOL:
                    print(f'  STOP: 02_default/{arch} bg_r={bg_r_val:.5f} '
                          f'differs from 2a ref {ref["bg_r"]:.5f} by {rel_diff*100:.2f}% > 1%')
                    sys.exit(2)
        print('[02_default] sanity check PASSED: reproduces 2a within 1%')

    # feature importance
    importance_agg = {}
    for arch, dfs in imp_by_arch.items():
        if not dfs:
            importance_agg[arch] = pd.DataFrame()
            continue
        combined_fi = pd.concat(dfs, ignore_index=True)
        grouped = (
            combined_fi.groupby('feature')['importance']
            .agg(['mean', 'std', 'count'])
            .reset_index()
            .rename(columns={'mean': 'mean_drop', 'std': 'std_drop', 'count': 'n_tracts'})
            .sort_values('mean_drop', ascending=False)
            .reset_index(drop=True)
        )
        grouped['rank'] = grouped.index + 1
        importance_agg[arch] = grouped
        short = 'sage' if arch == 'sage' else 'gcngat'
        grouped.to_csv(fi_dir / f'{short}_importance.csv', index=False)

    # standard 6 figures
    _generate_figures(per_tract_df, bg_validation, per_tract_bg, importance_agg, fig_dir, variant)

    # per-variant README
    _write_variant_readme(variant, vdir, per_tract_df, agg_metrics, bg_validation,
                          cross_tract_r, tract_list)

    print(f'[{variant["key"]}] done: {len(per_tract_df)} rows written')
    return per_tract_df, agg_metrics, bg_validation, cross_tract_r, importance_agg, tract_signal


# ---------------------------------------------------------------------------
# figure helpers
# ---------------------------------------------------------------------------

def _generate_figures(per_tract_df, bg_validation, per_tract_bg, importance_agg,
                      fig_dir, variant):
    from granite.visualization.plots import (
        plot_ablation_constraint_error_dist,
        plot_ablation_spatial_std_by_svi,
        plot_ablation_morans_i_by_tract,
        plot_ablation_block_group_scatter,
        plot_ablation_feature_importance_top20,
        plot_ablation_architecture_overlap,
    )
    success_df = per_tract_df[
        per_tract_df['failure'].isna() | (per_tract_df['failure'] == '')
    ].copy()

    if len(success_df) == 0:
        print(f'  WARNING: no successful rows, skipping figures for {variant["key"]}')
        return

    for name, fn, kw in [
        ('constraint_error_dist', plot_ablation_constraint_error_dist,
         {'df': success_df, 'output_path': str(fig_dir / 'constraint_error_dist.png')}),
        ('spatial_std_by_svi', plot_ablation_spatial_std_by_svi,
         {'df': success_df, 'output_path': str(fig_dir / 'spatial_std_by_svi.png')}),
        ('morans_i_by_tract', plot_ablation_morans_i_by_tract,
         {'df': success_df, 'output_path': str(fig_dir / 'morans_i_by_tract.png')}),
        ('block_group_scatter', plot_ablation_block_group_scatter,
         {'per_tract_bg': per_tract_bg, 'bg_validation': bg_validation,
          'output_path': str(fig_dir / 'block_group_scatter.png')}),
    ]:
        try:
            fn(**kw)
        except Exception as e:
            print(f'  WARNING: {name} figure failed: {e}')

    sage_imp   = importance_agg.get('sage', pd.DataFrame())
    gcngat_imp = importance_agg.get('gcn_gat', pd.DataFrame())
    for name, fn, kw in [
        ('feature_importance_top20', plot_ablation_feature_importance_top20,
         {'sage_imp': sage_imp, 'gcngat_imp': gcngat_imp,
          'output_path': str(fig_dir / 'feature_importance_top20.png')}),
        ('architecture_overlap', plot_ablation_architecture_overlap,
         {'sage_imp': sage_imp, 'gcngat_imp': gcngat_imp,
          'output_path': str(fig_dir / 'architecture_overlap.png')}),
    ]:
        try:
            fn(**kw)
        except Exception as e:
            print(f'  WARNING: {name} figure failed: {e}')


# ---------------------------------------------------------------------------
# per-variant README
# ---------------------------------------------------------------------------

def _write_variant_readme(variant, vdir, per_tract_df, agg_metrics, bg_validation,
                          cross_tract_r, tract_list):
    git_sha = (vdir / 'git_state.txt').read_text().strip()

    def _f(v, fmt='.4f'):
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            return 'n/a'
        try:
            return format(float(v), fmt)
        except Exception:
            return 'n/a'

    sa = agg_metrics.get('sage', {})
    ga = agg_metrics.get('gcn_gat', {})

    lines = [
        f'# 03_smoothness/{variant["dir"]}',
        '',
        f'Ablation 3 sub-experiment: {variant["description"]}.',
        '',
        '## Smoothness weight configuration',
        '',
        '| key | value |',
        '|---|---|',
        f'| smoothness_weight | {variant["smoothness_weight"]} |',
        f'| feature_standardization | per_tract (2a state) |',
        f'| norm_layers | input_layernorm=True, conv_norm_type=batchnorm (2a state) |',
        '',
        '## Run metadata',
        '',
        '| field | value |',
        '|---|---|',
        f'| git SHA | `{git_sha}` |',
        f'| seed | 42 |',
        f'| tracts | {len(tract_list)} |',
        f'| architectures | sage, gcn_gat |',
        '',
        '## Headline metrics',
        '',
        '| metric | GRANITE-SAGE | GRANITE-GCNGAT |',
        '|---|---|---|',
        f'| constraint error (mean) | {_f(sa.get("constraint_error_mean"))} | {_f(ga.get("constraint_error_mean"))} |',
        f'| spatial std (mean) | {_f(sa.get("spatial_std_mean"))} | {_f(ga.get("spatial_std_mean"))} |',
        f"| moran's I (mean) | {_f(sa.get('morans_i_mean'))} | {_f(ga.get('morans_i_mean'))} |",
        f'| block-group r (pooled) | {_f(bg_validation.get("sage", {}).get("pearson_r"))} | {_f(bg_validation.get("gcn_gat", {}).get("pearson_r"))} |',
        f'| between-tract variance | {_f(sa.get("between_tract_variance"))} | {_f(ga.get("between_tract_variance"))} |',
        f'| cross-tract signal r | {_f(cross_tract_r.get("sage"))} | {_f(cross_tract_r.get("gcn_gat"))} |',
        '',
        '## Artifacts',
        '',
        '- `results/per_tract_metrics.csv`',
        '- `results/aggregate_metrics.json`',
        '- `results/block_group_validation.json`',
        '- `results/feature_importance/sage_importance.csv`',
        '- `results/feature_importance/gcngat_importance.csv`',
        '- `figures/` (6 standard figures)',
        '',
    ]
    with open(vdir / 'README.md', 'w') as f:
        f.write('\n'.join(lines) + '\n')


# ---------------------------------------------------------------------------
# cross-variant summary
# ---------------------------------------------------------------------------

def generate_summary(results_by_variant, verbose):
    """Generate summary figures and delta_vs_default.json in 03_smoothness/summary/."""
    summary_dir = STEP3_ROOT / 'summary'
    summary_dir.mkdir(exist_ok=True)

    weights_ordered = [0.0, 0.025, 0.1, 0.2, 0.5]
    keys_ordered    = ['00_off', '01_quarter', '02_default', '03_double', '04_five_x']
    labels_ordered  = ['w=0.0\n(off)', 'w=0.025\n(quarter)', 'w=0.1\n(default)',
                       'w=0.2\n(double)', 'w=0.5\n(5x)']

    # build delta_vs_default
    ref_key = '02_default'
    ref_res = results_by_variant.get(ref_key)
    delta = {}
    for v in VARIANTS:
        if v['key'] == ref_key:
            continue
        res = results_by_variant.get(v['key'])
        if res is None:
            continue
        per_tract_df, agg_metrics, bg_validation, cross_tract_r, importance_agg, tract_signal = res
        ref_df, ref_agg, ref_bgv, ref_ctr, ref_imp, ref_ts = ref_res

        def _block(arch):
            sa = agg_metrics.get(arch, {})
            ref_sa = ref_agg.get(arch, {})
            bg_r   = bg_validation.get(arch, {}).get('pearson_r', float('nan'))
            ref_bgr = ref_bgv.get(arch, {}).get('pearson_r', float('nan'))
            ctr    = cross_tract_r.get(arch, float('nan'))
            ref_c  = ref_ctr.get(arch, float('nan'))
            btv    = sa.get('between_tract_variance', float('nan'))
            ref_btv = ref_sa.get('between_tract_variance', float('nan'))
            sp_std  = sa.get('spatial_std_mean', float('nan'))
            ref_sp  = ref_sa.get('spatial_std_mean', float('nan'))
            mi      = sa.get('morans_i_mean', float('nan'))
            ref_mi  = ref_sa.get('morans_i_mean', float('nan'))
            return {
                'mean_within_tract_std':          sp_std,
                'mean_within_tract_std_delta':    _safe_delta(sp_std, ref_sp),
                'cross_tract_signal_r':           ctr,
                'cross_tract_signal_r_delta':     _safe_delta(ctr, ref_c),
                'between_tract_variance':         btv,
                'between_tract_variance_delta':   _safe_delta(btv, ref_btv),
                'block_group_r':                  bg_r,
                'block_group_r_delta':            _safe_delta(bg_r, ref_bgr),
                'mean_morans_i':                  mi,
                'mean_morans_i_delta':            _safe_delta(mi, ref_mi),
            }

        delta[v['key']] = {
            'label':             v['label'],
            'smoothness_weight': v['smoothness_weight'],
            'sage':   _block('sage'),
            'gcngat': _block('gcn_gat'),
        }

    with open(summary_dir / 'delta_vs_default.json', 'w') as f:
        json.dump(delta, f, indent=2,
                  default=lambda x: None if isinstance(x, float) and math.isnan(x) else x)
    print('[03_summary] delta_vs_default.json written')

    # collect data for figures
    def _tract_stats(key, arch, col):
        res = results_by_variant.get(key)
        if res is None:
            return float('nan'), float('nan')
        df = res[0]
        sub = df[(df['architecture'] == arch) &
                 (df['failure'].isna() | (df['failure'] == ''))][col].dropna()
        if len(sub) == 0:
            return float('nan'), float('nan')
        return float(sub.mean()), float(sub.std())

    def _scalar(key, arch, field):
        res = results_by_variant.get(key)
        if res is None:
            return float('nan')
        agg = res[1].get(arch, {})
        return agg.get(field, float('nan'))

    def _bg_r(key, arch):
        res = results_by_variant.get(key)
        if res is None:
            return float('nan')
        return res[2].get(arch, {}).get('pearson_r', float('nan'))

    def _ctr(key, arch):
        res = results_by_variant.get(key)
        if res is None:
            return float('nan')
        return res[3].get(arch, float('nan'))

    # ----- figure 1: smoothness_sweep.png (3 rows x 2 cols) -----
    arch_info = [
        ('sage',    'GRANITE-SAGE',   '#2196F3'),
        ('gcn_gat', 'GRANITE-GCNGAT', '#FF5722'),
    ]
    row_metrics = [
        ('spatial_std',  'within-tract std',   'spatial_std_mean'),
        ('cross_r',      'cross-tract signal r', None),
        ('bg_r',         'block-group r',        None),
    ]

    fig1, axes1 = plt.subplots(3, 2, figsize=(11, 10))
    x_pos = np.arange(len(keys_ordered))

    for col, (arch, arch_label, color) in enumerate(arch_info):
        for row, (metric_id, ylabel, agg_field) in enumerate(row_metrics):
            ax = axes1[row, col]
            vals, errs = [], []
            for k in keys_ordered:
                if metric_id == 'spatial_std':
                    m, s = _tract_stats(k, arch, 'spatial_std')
                    vals.append(m); errs.append(s if math.isfinite(s) else 0)
                elif metric_id == 'cross_r':
                    vals.append(_ctr(k, arch)); errs.append(0)
                else:
                    vals.append(_bg_r(k, arch)); errs.append(0)
            vals = np.array(vals, dtype=float)
            errs = np.array(errs, dtype=float)
            valid = np.isfinite(vals)
            if valid.any():
                ax.errorbar(x_pos[valid], vals[valid], yerr=errs[valid],
                            fmt='o-', color=color, capsize=4, linewidth=1.5, markersize=6)
                # mark default weight
                def_idx = keys_ordered.index('02_default')
                if math.isfinite(vals[def_idx]):
                    ax.axvline(def_idx, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels_ordered, fontsize=7)
            ax.tick_params(labelsize=8)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            if row == 0:
                ax.set_title(arch_label, fontsize=10)

    fig1.suptitle('Smoothness Weight Sweep: within-tract std, cross-tract signal r, block-group r',
                  fontsize=10)
    fig1.tight_layout()
    fig1.savefig(str(summary_dir / 'smoothness_sweep.png'), dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print('[03_summary] smoothness_sweep.png written')

    # ----- figure 2: between_tract_variance_sweep.png (single panel) -----
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    for arch, arch_label, color in arch_info:
        vals = np.array([_scalar(k, arch, 'between_tract_variance') for k in keys_ordered], dtype=float)
        valid = np.isfinite(vals)
        if valid.any():
            ax2.plot(x_pos[valid], vals[valid], 'o-', color=color, linewidth=1.5,
                     markersize=6, label=arch_label)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels_ordered, fontsize=8)
    ax2.set_ylabel('between-tract variance (var of tract mean preds)', fontsize=9)
    ax2.set_xlabel('smoothness weight', fontsize=9)
    ax2.legend(fontsize=9)
    ax2.axvline(keys_ordered.index('02_default'), color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
    fig2.suptitle('Between-Tract Variance vs Smoothness Weight', fontsize=10)
    fig2.tight_layout()
    fig2.savefig(str(summary_dir / 'between_tract_variance_sweep.png'), dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print('[03_summary] between_tract_variance_sweep.png written')

    # ----- figure 3: tract_pred_vs_actual.png (2 rows x 5 cols) -----
    fig3, axes3 = plt.subplots(2, 5, figsize=(16, 7))
    for row, (arch, arch_label, color) in enumerate(arch_info):
        for col, key in enumerate(keys_ordered):
            ax = axes3[row, col]
            res = results_by_variant.get(key)
            if res is not None:
                sig = res[5].get(arch, [])
                if sig:
                    xs = np.array([x[1] for x in sig])   # actual SVI
                    ys = np.array([x[0] for x in sig])   # mean pred
                    valid = np.isfinite(xs) & np.isfinite(ys)
                    if valid.sum() >= 2:
                        ax.scatter(xs[valid], ys[valid], color=color, s=20, alpha=0.8)
                        lim = [min(xs[valid].min(), ys[valid].min()) - 0.02,
                               max(xs[valid].max(), ys[valid].max()) + 0.02]
                        ax.plot(lim, lim, 'k--', linewidth=0.7, alpha=0.5)
                        r_val = np.corrcoef(xs[valid], ys[valid])[0, 1]
                        ax.set_title(f'r={r_val:.3f}', fontsize=8)
            ax.tick_params(labelsize=7)
            if col == 0:
                ax.set_ylabel(arch_label, fontsize=8)
            if row == 1:
                ax.set_xlabel(labels_ordered[col].replace('\n', ' '), fontsize=7)
    fig3.suptitle('Tract mean prediction vs actual SVI (rows=arch, cols=weight)',
                  fontsize=10)
    fig3.tight_layout()
    fig3.savefig(str(summary_dir / 'tract_pred_vs_actual.png'), dpi=300, bbox_inches='tight')
    plt.close(fig3)
    print('[03_summary] tract_pred_vs_actual.png written')

    _write_summary_readme(results_by_variant, delta, summary_dir, keys_ordered, labels_ordered)


def _write_summary_readme(results_by_variant, delta, summary_dir, keys_ordered, labels_ordered):
    sha = _git_sha()
    ts  = datetime.now().strftime('%Y-%m-%d %H:%M')

    def _f(v, fmt='.4f'):
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            return 'n/a'
        try:
            return format(float(v), fmt)
        except Exception:
            return 'n/a'

    # sanity check line from 02_default
    def_res = results_by_variant.get('02_default')
    sanity_lines = []
    if def_res:
        def_agg = def_res[1]
        for arch in ARCHITECTURES:
            sa = def_agg.get(arch, {})
            ref = REF_2A[arch]
            std_val = sa.get('spatial_std_mean', float('nan'))
            bg_r_val = def_res[2].get(arch, {}).get('pearson_r', float('nan'))
            if math.isfinite(std_val) and math.isfinite(ref['spatial_std_mean']):
                rel_std = abs(std_val - ref['spatial_std_mean']) / (abs(ref['spatial_std_mean']) + 1e-9) * 100
            else:
                rel_std = float('nan')
            if math.isfinite(bg_r_val) and math.isfinite(ref['bg_r']):
                rel_bgr = abs(bg_r_val - ref['bg_r']) / (abs(ref['bg_r']) + 1e-9) * 100
            else:
                rel_bgr = float('nan')
            sanity_lines.append(
                f'| {arch} | {_f(std_val)} | {_f(ref["spatial_std_mean"])} | {_f(rel_std, ".2f")}% | '
                f'{_f(bg_r_val)} | {_f(ref["bg_r"])} | {_f(rel_bgr, ".2f")}% |'
            )

    # hypothesis: cross_tract_signal_r monotonic in weight (decreasing weight -> increasing r)?
    for arch in ARCHITECTURES:
        ctrs = []
        for k in keys_ordered:
            res = results_by_variant.get(k)
            if res:
                ctrs.append(res[3].get(arch, float('nan')))
            else:
                ctrs.append(float('nan'))
        valid = [(w, r) for w, r in zip([0.0, 0.025, 0.1, 0.2, 0.5], ctrs) if math.isfinite(r)]

    # best weight: maximize cross_tract_signal_r with bg_r >= 0.74
    best = {}
    for arch in ARCHITECTURES:
        best_r, best_k = float('-inf'), None
        for k in keys_ordered:
            res = results_by_variant.get(k)
            if res is None:
                continue
            ctr   = res[3].get(arch, float('nan'))
            bg_r  = res[2].get(arch, {}).get('pearson_r', float('nan'))
            if math.isfinite(ctr) and math.isfinite(bg_r) and bg_r >= 0.74 and ctr > best_r:
                best_r, best_k = ctr, k
        best[arch] = (best_k, best_r)

    lines = [
        '# 03_smoothness/summary: cross-tract smoothness weight sweep',
        '',
        f'git SHA: `{sha}` | seed: 42 | tracts: 20 | generated: {ts}',
        '',
        '## Inspection summary',
        '',
        '`_compute_cross_tract_smoothness` is a between-tract mean range penalty.',
        'It computes one scalar per tract (mean prediction over all addresses),',
        'then returns `(max - min) * 0.05`. Effective weight on the range at',
        'default config: `0.1 * 0.05 = 0.005`. The function contains no graph',
        'structure. It directly suppresses the spread of tract mean predictions,',
        'which is the same quantity `cross_tract_signal_r` depends on.',
        'Mehdi\'s "remove or correct" recommendation is supported: the term',
        'encodes no information and directly degrades between-tract discriminability.',
        '',
        '## 02_default sanity check (must reproduce 2a within 1%)',
        '',
        '| arch | std (run) | std (2a ref) | rel diff | bg_r (run) | bg_r (2a ref) | rel diff |',
        '|---|---|---|---|---|---|---|',
    ] + sanity_lines + [
        '',
        '## Hypothesis: cross_tract_signal_r monotonic in smoothness weight',
        '',
        '(lower weight -> less suppression -> higher r between tract mean pred and actual SVI)',
        '',
    ]

    for arch in ARCHITECTURES:
        ctrs = []
        for k in keys_ordered:
            res = results_by_variant.get(k)
            ctrs.append(res[3].get(arch, float('nan')) if res else float('nan'))
        lines.append(f'**{arch}**: ' + ', '.join(
            f'{lbl.split(chr(10))[0]}={_f(r)}' for lbl, r in zip(labels_ordered, ctrs)
        ))
    lines.append('')

    lines += [
        '## Best smoothness weight (maximize cross_tract_signal_r with bg_r >= 0.74)',
        '',
    ]
    for arch in ARCHITECTURES:
        bk, br = best[arch]
        lines.append(f'- {arch}: `{bk}` (cross_tract_signal_r = {_f(br)})')
    lines.append('')

    lines += [
        '## Delta vs default (02_default)',
        '',
        '### SAGE',
        '',
        '| variant | within-tract std | delta | cross-tract r | delta | bg_r | delta |',
        '|---|---|---|---|---|---|---|',
    ]
    for k in [kk for kk in keys_ordered if kk != '02_default']:
        d = delta.get(k, {}).get('sage', {})
        lines.append(
            f'| {k} | {_f(d.get("mean_within_tract_std"))} | {_f(d.get("mean_within_tract_std_delta"), "+.4f")} | '
            f'{_f(d.get("cross_tract_signal_r"))} | {_f(d.get("cross_tract_signal_r_delta"), "+.4f")} | '
            f'{_f(d.get("block_group_r"))} | {_f(d.get("block_group_r_delta"), "+.4f")} |'
        )
    lines += [
        '',
        '### GCN-GAT',
        '',
        '| variant | within-tract std | delta | cross-tract r | delta | bg_r | delta |',
        '|---|---|---|---|---|---|---|',
    ]
    for k in [kk for kk in keys_ordered if kk != '02_default']:
        d = delta.get(k, {}).get('gcngat', {})
        lines.append(
            f'| {k} | {_f(d.get("mean_within_tract_std"))} | {_f(d.get("mean_within_tract_std_delta"), "+.4f")} | '
            f'{_f(d.get("cross_tract_signal_r"))} | {_f(d.get("cross_tract_signal_r_delta"), "+.4f")} | '
            f'{_f(d.get("block_group_r"))} | {_f(d.get("block_group_r_delta"), "+.4f")} |'
        )
    lines += [
        '',
        '## Artifacts',
        '',
        '- `delta_vs_default.json`',
        '- `smoothness_sweep.png`',
        '- `between_tract_variance_sweep.png`',
        '- `tract_pred_vs_actual.png`',
        '',
        '## Next step',
        '',
        'Step 4: constraint-by-construction.',
        '',
    ]
    with open(summary_dir / 'README.md', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('[03_summary] README.md written')


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='03 smoothness weight sweep')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--only', default=None,
                        help='run only one variant key: 00_off | 01_quarter | 02_default | 03_double | 04_five_x')
    args = parser.parse_args()

    ts_start = datetime.now()
    print(f'[03] start: {ts_start.strftime("%Y-%m-%d %H:%M:%S")}')
    set_random_seed(SEED)

    if not BG_SVI_PATH.exists():
        print(f'[03] HALT: {BG_SVI_PATH} missing')
        sys.exit(1)

    if not TRACT_SEL.exists():
        print(f'[03] HALT: tract_selection.txt missing at {TRACT_SEL}')
        sys.exit(1)

    tract_list = _load_tract_list()
    print(f'[03] tracts: {len(tract_list)}')

    print('[03] loading BG geodataframe...')
    bg_gdf    = _load_bg_gdf()
    validator = BlockGroupValidator(bg_gdf, verbose=False)

    cfg_base = _load_config()
    cfg_base['data']['target']          = 'svi'
    cfg_base['data']['neighbor_tracts'] = 0
    cfg_base['data']['state_fips']      = '47'
    cfg_base['data']['county_fips']     = '065'
    cfg_base['processing']['skip_importance']     = True
    cfg_base['processing']['verbose']             = args.verbose
    cfg_base['processing']['random_seed']         = SEED
    cfg_base['processing']['enable_caching']      = True
    cfg_base['training']['apply_post_correction'] = True

    print('[03] loading spatial data...')
    init_cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in cfg_base.items()}
    init_cfg['data'] = dict(cfg_base['data'])
    init_cfg['data']['target_fips'] = tract_list[0]
    scratch0 = str(REPO_ROOT / 'output' / 'ablation_03_spatial_scratch')
    os.makedirs(scratch0, exist_ok=True)
    p0 = GRANITEPipeline(init_cfg, output_dir=scratch0)
    p0.verbose = args.verbose
    try:
        data = p0._load_spatial_data()
    except Exception as e:
        print(f'[03] HALT: spatial data load failed: {e}')
        sys.exit(1)

    variants_to_run = VARIANTS if args.only is None else [
        v for v in VARIANTS if v['key'] == args.only
    ]
    if not variants_to_run:
        print(f'[03] HALT: --only "{args.only}" does not match any variant key')
        sys.exit(1)

    results_by_variant = {}
    for variant in variants_to_run:
        print(f'\n[03] ===== {variant["label"]} =====')
        t0 = time.time()
        res = run_variant(variant, cfg_base, data, bg_gdf, validator,
                          tract_list, args.verbose)
        results_by_variant[variant['key']] = res
        elapsed = time.time() - t0
        print(f'[03] {variant["key"]} elapsed: {int(elapsed)//60}m {int(elapsed)%60}s')

    if len(results_by_variant) == len(VARIANTS):
        print('\n[03] generating cross-variant summary...')
        generate_summary(results_by_variant, args.verbose)
    else:
        print(f'[03] skipping summary ({len(results_by_variant)}/{len(VARIANTS)} variants ran)')

    ts_end = datetime.now()
    total_elapsed = (ts_end - ts_start).seconds
    print(f'\n[03] complete: {ts_end.strftime("%Y-%m-%d %H:%M:%S")} '
          f'({total_elapsed//60}m {total_elapsed%60}s)')


if __name__ == '__main__':
    main()
