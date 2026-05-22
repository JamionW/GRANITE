"""
01_per_tract_std: per-tract z-score standardization ablation.

Replaces the global RobustScaler in 00_baseline with per-tract z-score
standardization. All other pipeline settings are identical to 00_baseline.

Hypothesis: global standardization compresses within-tract feature variance for
tracts whose features cluster away from the global median/IQR. Per-tract
standardization restores within-tract structure, increasing spatial std and
flattening the negative spatial-std vs SVI slope.

Usage:
    python experiments/ablation/01_per_tract_std/run_ablation_01.py [--verbose]
"""
import argparse
import json
import math
import os
import sys
import time
import traceback
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

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from granite.disaggregation.pipeline import GRANITEPipeline
from granite.data.block_group_loader import BlockGroupLoader
from granite.validation.block_group_validation import BlockGroupValidator
from granite.evaluation.spatial_diagnostics import SpatialLearningDiagnostics
from granite.models.gnn import set_random_seed

# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
ABLATION_DIR = Path(__file__).resolve().parent
RESULTS_DIR  = ABLATION_DIR / 'results'
FIGURES_DIR  = ABLATION_DIR / 'figures'
FI_DIR       = RESULTS_DIR  / 'feature_importance'
BASELINE_DIR = ABLATION_DIR.parent / '00_baseline'
BASELINE_RESULTS = BASELINE_DIR / 'results'

for d in (RESULTS_DIR, FIGURES_DIR, FI_DIR):
    d.mkdir(parents=True, exist_ok=True)

TRACT_INVENTORY = REPO_ROOT / 'tract_inventory.csv'
CONFIG_PATH     = REPO_ROOT / 'config.yaml'
BG_SVI_PATH     = REPO_ROOT / 'data' / 'processed' / 'national_bg_svi.csv'
GRAVEYARD       = REPO_ROOT / 'graveyard'

ARCHITECTURES = ['sage', 'gcn_gat']
ARCH_LABELS   = {'sage': 'GRANITE-SAGE', 'gcn_gat': 'GRANITE-GCNGAT'}
SEED          = 42
MIN_ADDRESSES_PER_BG = 3
MIN_ADDRESSES_POOLED = 10

# baseline reference values for delta_vs_baseline and stop conditions
BASELINE_AGG = {
    'sage':    {'spatial_std_mean': 0.0797295,  'morans_i_mean': 0.83260845},
    'gcn_gat': {'spatial_std_mean': 0.08870984999999999, 'morans_i_mean': 0.8476092},
}
BASELINE_BG_R = {
    'sage':    0.7691647597080817,
    'gcn_gat': 0.7491051475476098,
    'IDW':     0.7718927214530875,
    'kriging': 0.7681517600497690,
}
BASELINE_FI_SPEARMAN = 0.099
BASELINE_TOP10_OVERLAP = 2

# stop thresholds
CONSTRAINT_ERROR_MAX = 1e-4      # post-correction absolute error per tract
SPATIAL_STD_DROP_WARN = 0.02     # per-arch mean drop from baseline flagged
BG_R_CHANGE_WARN = 0.05          # per-method change from baseline flagged


# ---------------------------------------------------------------------------
# helpers (mirrors run_baseline.py)
# ---------------------------------------------------------------------------

def _load_config():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f) or {}
    for k in ('data', 'model', 'training', 'processing', 'recovery',
              'validation', 'features'):
        cfg.setdefault(k, {})
    return cfg


def _load_tract_inventory(verbose):
    df = pd.read_csv(TRACT_INVENTORY)
    df['fips'] = df['fips'].astype(str).str.strip()
    if 'Status' not in df.columns:
        if verbose:
            print('[ablation01] NOTE: no Status column; all rows treated as in-scope')
        return df
    documented = {'in-scope', 'in_scope', 'active', 'include', 'yes'}
    return df[df['Status'].str.lower().isin(documented)]


def _load_bg_gdf():
    loader = BlockGroupLoader(data_dir=str(REPO_ROOT / 'data'), verbose=False)
    bg_gdf = loader.get_block_groups_with_demographics(
        '47', '065', svi_ranking_scope='national'
    )
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
    n_bgs = len(merged)
    if n_bgs < 2:
        return {'bg_r': float('nan'), 'bg_rmse': float('nan'), 'n_bgs': n_bgs}
    p = merged['predicted_svi'].values.astype(float)
    t = merged['SVI'].values.astype(float)
    valid = np.isfinite(p) & np.isfinite(t)
    if valid.sum() < 2:
        return {'bg_r': float('nan'), 'bg_rmse': float('nan'), 'n_bgs': n_bgs}
    bg_r   = float(np.corrcoef(p[valid], t[valid])[0, 1])
    bg_rmse = float(np.sqrt(np.mean((p[valid] - t[valid]) ** 2)))
    return {'bg_r': bg_r, 'bg_rmse': bg_rmse, 'n_bgs': int(valid.sum())}


def _compute_morans_i(predictions, address_gdf, k_neighbors=8):
    try:
        coords = np.array([[g.x, g.y] for g in address_gdf.geometry])
        diag = SpatialLearningDiagnostics()
        return diag.compute_spatial_autocorrelation(
            predictions, coords, k_neighbors=k_neighbors
        )
    except Exception:
        return float('nan')


def _check_nan_stop(metric_name, value, fips, arch):
    if not math.isfinite(value):
        print(f'[ablation01] STOP: non-finite {metric_name}={value} on {fips}/{arch}')
        sys.exit(2)


def _check_constraint_stop(constraint_err, fips, arch):
    if constraint_err > CONSTRAINT_ERROR_MAX:
        print(
            f'[ablation01] STOP: constraint_error={constraint_err:.2e} > {CONSTRAINT_ERROR_MAX:.0e} '
            f'on {fips}/{arch}. Post-correction should saturate to zero.'
        )
        sys.exit(2)


def _load_idw_kriging(tracts_gdf):
    import importlib.util
    gv_path = GRAVEYARD / 'disaggregation_baselines_idw_kriging.py'
    spec = importlib.util.spec_from_file_location('idw_kriging_graveyard', gv_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    idw  = mod.IDWDisaggregation(power=2.0, n_neighbors=8)
    idw.fit(tracts_gdf, svi_column='RPL_THEMES')
    krig = mod.OrdinaryKrigingDisaggregation()
    krig.fit(tracts_gdf, svi_column='RPL_THEMES')
    return idw, krig


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='01_per_tract_std ablation run')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    ts_start = datetime.now()
    print(f'[ablation01] start: {ts_start.strftime("%Y-%m-%d %H:%M:%S")}')
    set_random_seed(SEED)

    if not BG_SVI_PATH.exists():
        print(f'[ablation01] HALT: {BG_SVI_PATH} missing')
        sys.exit(1)

    # tract selection (same 20 as baseline)
    inventory_df = _load_tract_inventory(args.verbose)
    tract_list   = inventory_df['fips'].tolist()
    print(f'[ablation01] in-scope tracts: {len(tract_list)}')

    # load BG data
    print('[ablation01] loading BG geodataframe...')
    bg_gdf    = _load_bg_gdf()
    validator = BlockGroupValidator(bg_gdf, verbose=False)

    # config: enable per-tract standardization, all else identical to baseline
    cfg = _load_config()
    cfg['data']['target']          = 'svi'
    cfg['data']['neighbor_tracts'] = 0
    cfg['data']['state_fips']      = '47'
    cfg['data']['county_fips']     = '065'
    cfg['processing']['skip_importance']   = True
    cfg['processing']['verbose']           = args.verbose
    cfg['processing']['random_seed']       = SEED
    cfg['processing']['enable_caching']    = True
    cfg['training']['apply_post_correction'] = True
    cfg['features']['feature_standardization'] = 'per_tract'  # the one change

    scratch_dir = str(REPO_ROOT / 'output' / 'ablation_01_scratch')
    os.makedirs(scratch_dir, exist_ok=True)

    # load spatial data once
    print('[ablation01] loading spatial data...')
    pipeline_init_cfg = dict(cfg)
    pipeline_init_cfg['data'] = dict(cfg['data'])
    pipeline_init_cfg['data']['target_fips'] = tract_list[0]
    pipeline = GRANITEPipeline(pipeline_init_cfg, output_dir=scratch_dir)
    pipeline.verbose = args.verbose

    try:
        data = pipeline._load_spatial_data()
    except Exception as e:
        print(f'[ablation01] HALT: spatial data load failed: {e}')
        sys.exit(1)

    # IDW and kriging
    print('[ablation01] fitting IDW and kriging baselines...')
    try:
        idw_model, krig_model = _load_idw_kriging(data['tracts'])
    except Exception as e:
        print(f'[ablation01] WARNING: IDW/kriging load failed: {e}')
        idw_model, krig_model = None, None

    # storage
    per_tract_rows = []
    pooled_addr_gdfs = {a: [] for a in ARCHITECTURES}
    pooled_preds     = {a: [] for a in ARCHITECTURES}
    pooled_preds_idw  = []
    pooled_preds_krig = []
    pooled_addr_gdfs_baselines = []
    importance_by_arch = {a: [] for a in ARCHITECTURES}

    # per-tract scaler accumulation (for per_tract_scalers.npz and zero_var_columns.csv)
    all_tract_mu    = {}   # (fips, arch) -> mu array
    all_tract_sigma = {}   # (fips, arch) -> sigma array
    zero_var_records = []  # list of {arch, fips, tract, feature_idx}

    total = len(ARCHITECTURES) * len(tract_list)
    done  = 0

    # -----------------------------------------------------------------------
    # main loop: architecture x tract
    # -----------------------------------------------------------------------
    for arch in ARCHITECTURES:
        arch_label = ARCH_LABELS[arch]
        print(f'\n[ablation01] === architecture: {arch_label} ===')

        cfg['model']['architecture']  = arch
        pipeline.config = dict(cfg)
        pipeline.verbose = args.verbose

        for idx, fips in enumerate(tract_list):
            done += 1
            print(
                f'[ablation01] [{done}/{total}] {arch_label} | '
                f'tract {idx+1}/{len(tract_list)}: {fips}'
            )
            t0 = time.time()

            cfg['data']['target_fips']           = fips
            pipeline.config['data']['target_fips']  = fips
            pipeline.config['model']['architecture'] = arch

            try:
                result = pipeline._process_single_tract(fips, data)
            except Exception as e:
                msg = str(e)[:300]
                print(f'[ablation01]   ERROR: {msg}')
                per_tract_rows.append({
                    'fips': fips, 'architecture': arch, 'n_addresses': 0,
                    'tract_svi': float('nan'), 'constraint_error': float('nan'),
                    'spatial_std': float('nan'), 'morans_i': float('nan'),
                    'bg_r': float('nan'), 'n_bgs': 0, 'runtime_s': 0,
                    'failure': msg,
                })
                continue

            if not result.get('success'):
                msg = result.get('error', 'unknown')[:300]
                print(f'[ablation01]   FAILED: {msg}')
                per_tract_rows.append({
                    'fips': fips, 'architecture': arch, 'n_addresses': 0,
                    'tract_svi': float('nan'), 'constraint_error': float('nan'),
                    'spatial_std': float('nan'), 'morans_i': float('nan'),
                    'bg_r': float('nan'), 'n_bgs': 0, 'runtime_s': 0,
                    'failure': msg,
                })
                continue

            runtime     = time.time() - t0
            address_gdf = result['address_gdf']
            preds_arr   = result['predictions']['mean'].values.astype(float)
            tract_svi   = float(result['tract_svi'])
            n_addresses = len(address_gdf)

            constr_err = abs(np.mean(preds_arr) - tract_svi)
            sp_std     = float(np.std(preds_arr))
            morans_i   = _compute_morans_i(preds_arr, address_gdf)

            # stop conditions
            _check_nan_stop('constraint_error', constr_err, fips, arch)
            _check_nan_stop('spatial_std', sp_std, fips, arch)
            _check_constraint_stop(constr_err, fips, arch)
            if not math.isfinite(morans_i):
                print(f'[ablation01]   WARNING: non-finite morans_i for {fips}/{arch}')
                morans_i = float('nan')

            # BG validation
            try:
                bg_agg = _aggregate_to_bg(validator, address_gdf, preds_arr,
                                          MIN_ADDRESSES_PER_BG)
                bm = _bg_metrics(bg_agg, bg_gdf)
            except Exception as e:
                bm = {'bg_r': float('nan'), 'bg_rmse': float('nan'), 'n_bgs': 0}
                print(f'[ablation01]   WARNING: BG metrics failed: {e}')

            per_tract_rows.append({
                'fips': fips, 'architecture': arch, 'n_addresses': n_addresses,
                'tract_svi': tract_svi,
                'constraint_error': round(constr_err, 8),
                'spatial_std': round(sp_std, 6),
                'morans_i': round(morans_i, 6) if math.isfinite(morans_i) else float('nan'),
                'bg_r': bm['bg_r'], 'n_bgs': bm['n_bgs'],
                'runtime_s': round(runtime, 2),
                'failure': '',
            })

            pooled_addr_gdfs[arch].append(address_gdf.copy())
            pooled_preds[arch].append(preds_arr)

            if arch == ARCHITECTURES[0]:
                pooled_addr_gdfs_baselines.append(address_gdf.copy())
                if idw_model is not None:
                    try:
                        addr_coords = np.array([[g.x, g.y] for g in address_gdf.geometry])
                        idw_p = idw_model.disaggregate(addr_coords, fips, tract_svi)
                        pooled_preds_idw.append(idw_p)
                    except Exception:
                        pooled_preds_idw.append(np.full(n_addresses, tract_svi))
                if krig_model is not None:
                    try:
                        addr_coords = np.array([[g.x, g.y] for g in address_gdf.geometry])
                        krig_p = krig_model.disaggregate(addr_coords, fips, tract_svi)
                        pooled_preds_krig.append(krig_p)
                    except Exception:
                        pooled_preds_krig.append(np.full(n_addresses, tract_svi))

            # collect per-tract scaler info
            scaler = getattr(pipeline, '_stored_feature_scaler', None)
            if isinstance(scaler, dict):
                for tract_key, mu_vals in scaler.get('tract_mu', {}).items():
                    all_tract_mu[(fips, arch)]    = np.array(mu_vals)
                    all_tract_sigma[(fips, arch)] = np.array(
                        scaler['tract_sigma'].get(tract_key, [])
                    )
                for rec in scaler.get('zero_var_log', []):
                    zero_var_records.append({
                        'architecture': arch,
                        'fips': fips,
                        'tract': rec['tract'],
                        'feature_idx': rec['feature_idx'],
                    })

            # feature importance
            try:
                fi = pipeline.analyze_feature_importance(result, n_repeats=5)
                if fi is not None:
                    perm = fi.get('permutation', {})
                    if isinstance(perm, dict) and 'feature_importance' in perm:
                        imp_df = perm['feature_importance'].copy()
                        imp_df['fips']         = fips
                        imp_df['architecture'] = arch
                        importance_by_arch[arch].append(imp_df)
            except Exception as e:
                print(f'[ablation01]   WARNING: feature importance failed for {fips}/{arch}: {e}')

            if args.verbose:
                print(
                    f'[ablation01]   constr_err={constr_err:.2e} '
                    f'sp_std={sp_std:.4f} morans_i={morans_i:.3f} '
                    f'bg_r={bm["bg_r"]:.3f} t={runtime:.1f}s'
                )

    # -----------------------------------------------------------------------
    # save per-tract scaler diagnostics
    # -----------------------------------------------------------------------
    _save_scaler_diagnostics(all_tract_mu, all_tract_sigma, zero_var_records)

    # -----------------------------------------------------------------------
    # per_tract_metrics.csv
    # -----------------------------------------------------------------------
    per_tract_df = pd.DataFrame(per_tract_rows)
    per_tract_df.to_csv(RESULTS_DIR / 'per_tract_metrics.csv', index=False)
    print(f'\n[ablation01] per_tract_metrics.csv written ({len(per_tract_df)} rows)')

    # -----------------------------------------------------------------------
    # aggregate metrics
    # -----------------------------------------------------------------------
    agg_metrics = {}
    for arch in ARCHITECTURES:
        arch_df = per_tract_df[
            (per_tract_df['architecture'] == arch) &
            (per_tract_df['failure'].isna() | (per_tract_df['failure'] == ''))
        ]
        if len(arch_df) == 0:
            agg_metrics[arch] = {}
            continue
        agg_metrics[arch] = {
            'n_tracts': int(len(arch_df)),
            'constraint_error_mean':   float(arch_df['constraint_error'].mean()),
            'constraint_error_median': float(arch_df['constraint_error'].median()),
            'spatial_std_mean':        float(arch_df['spatial_std'].mean()),
            'spatial_std_median':      float(arch_df['spatial_std'].median()),
            'morans_i_mean':   float(arch_df['morans_i'].dropna().mean())
                               if arch_df['morans_i'].notna().any() else float('nan'),
            'morans_i_median': float(arch_df['morans_i'].dropna().median())
                               if arch_df['morans_i'].notna().any() else float('nan'),
            'bg_r_mean':   float(arch_df['bg_r'].dropna().mean())
                           if arch_df['bg_r'].notna().any() else float('nan'),
            'bg_r_median': float(arch_df['bg_r'].dropna().median())
                           if arch_df['bg_r'].notna().any() else float('nan'),
        }

    with open(RESULTS_DIR / 'aggregate_metrics.json', 'w') as f:
        json.dump(agg_metrics, f, indent=2,
                  default=lambda x: None if (isinstance(x, float) and math.isnan(x)) else x)
    print('[ablation01] aggregate_metrics.json written')

    # stop condition: mean spatial_std dropped by more than 0.02 vs baseline
    for arch in ARCHITECTURES:
        a = agg_metrics.get(arch, {})
        b = BASELINE_AGG.get(arch, {})
        if a and b:
            delta = a.get('spatial_std_mean', float('nan')) - b.get('spatial_std_mean', float('nan'))
            if math.isfinite(delta) and delta < -SPATIAL_STD_DROP_WARN:
                print(
                    f'[ablation01] FLAG: {arch} mean spatial_std dropped {abs(delta):.4f} '
                    f'below baseline (>{SPATIAL_STD_DROP_WARN}). Per-tract standardization '
                    'suppressed within-tract variance. Continuing; note in manifest.'
                )

    # -----------------------------------------------------------------------
    # pooled BG validation
    # -----------------------------------------------------------------------
    print('[ablation01] computing pooled BG validation...')
    bg_validation = {}
    per_tract_bg  = {}

    for arch in ARCHITECTURES:
        if not pooled_addr_gdfs[arch]:
            bg_validation[arch] = {'pearson_r': float('nan')}
            continue
        combined_gdf = pd.concat(pooled_addr_gdfs[arch], ignore_index=True)
        if not isinstance(combined_gdf, gpd.GeoDataFrame):
            combined_gdf = gpd.GeoDataFrame(combined_gdf, geometry='geometry', crs='EPSG:4326')
        if combined_gdf.crs is None:
            combined_gdf = combined_gdf.set_crs('EPSG:4326')
        all_p = np.concatenate(pooled_preds[arch])
        try:
            bg_agg_pooled = _aggregate_to_bg(validator, combined_gdf, all_p, MIN_ADDRESSES_POOLED)
            bm_pooled     = _bg_metrics(bg_agg_pooled, bg_gdf)
            svi_lookup    = bg_gdf[['GEOID', 'SVI']].dropna(subset=['SVI'])
            bg_agg_all    = _aggregate_to_bg(validator, combined_gdf, all_p, 1)
            merged_scatter = bg_agg_all.merge(svi_lookup, on='GEOID', how='inner')
        except Exception as e:
            bm_pooled      = {'bg_r': float('nan'), 'bg_rmse': float('nan'), 'n_bgs': 0}
            merged_scatter = pd.DataFrame()
            print(f'[ablation01] WARNING: pooled BG validation failed for {arch}: {e}')

        bg_validation[arch] = {
            'pearson_r': bm_pooled['bg_r'],
            'bg_rmse':   bm_pooled['bg_rmse'],
            'n_bgs':     bm_pooled['n_bgs'],
        }
        per_tract_bg[arch] = merged_scatter

        # stop condition: BG r change > 0.05 (flag only)
        base_r = BASELINE_BG_R.get(arch, float('nan'))
        curr_r = bm_pooled['bg_r']
        if math.isfinite(curr_r) and math.isfinite(base_r):
            delta_r = abs(curr_r - base_r)
            if delta_r > BG_R_CHANGE_WARN:
                print(
                    f'[ablation01] FLAG: {arch} pooled BG r changed {delta_r:.3f} '
                    f'from baseline {base_r:.3f} -> {curr_r:.3f} (>{BG_R_CHANGE_WARN}). '
                    'Noted in manifest; not stopping.'
                )

    for label, preds_list in [('IDW', pooled_preds_idw), ('kriging', pooled_preds_krig)]:
        if not preds_list or not pooled_addr_gdfs_baselines:
            bg_validation[label] = {'pearson_r': BASELINE_BG_R.get(label, float('nan')),
                                    'note': 'copied from baseline; IDW/kriging unaffected by feature standardization'}
            per_tract_bg[label] = pd.DataFrame()
            continue
        combined_gdf = pd.concat(pooled_addr_gdfs_baselines, ignore_index=True)
        if not isinstance(combined_gdf, gpd.GeoDataFrame):
            combined_gdf = gpd.GeoDataFrame(combined_gdf, geometry='geometry', crs='EPSG:4326')
        if combined_gdf.crs is None:
            combined_gdf = combined_gdf.set_crs('EPSG:4326')
        all_p = np.concatenate(preds_list)
        try:
            bg_agg_pooled  = _aggregate_to_bg(validator, combined_gdf, all_p, MIN_ADDRESSES_POOLED)
            bm_pooled      = _bg_metrics(bg_agg_pooled, bg_gdf)
            svi_lookup     = bg_gdf[['GEOID', 'SVI']].dropna(subset=['SVI'])
            bg_agg_all     = _aggregate_to_bg(validator, combined_gdf, all_p, 1)
            merged_scatter = bg_agg_all.merge(svi_lookup, on='GEOID', how='inner')
        except Exception as e:
            bm_pooled      = {'bg_r': float('nan'), 'bg_rmse': float('nan'), 'n_bgs': 0}
            merged_scatter = pd.DataFrame()
            print(f'[ablation01] WARNING: pooled BG validation failed for {label}: {e}')
        bg_validation[label] = {
            'pearson_r': bm_pooled['bg_r'],
            'bg_rmse':   bm_pooled['bg_rmse'],
            'n_bgs':     bm_pooled['n_bgs'],
        }
        per_tract_bg[label] = merged_scatter

    with open(RESULTS_DIR / 'block_group_validation.json', 'w') as f:
        json.dump(bg_validation, f, indent=2,
                  default=lambda x: None if (isinstance(x, float) and math.isnan(x)) else x)
    print('[ablation01] block_group_validation.json written')

    # -----------------------------------------------------------------------
    # feature importance
    # -----------------------------------------------------------------------
    importance_agg = _write_feature_importance(importance_by_arch)

    # -----------------------------------------------------------------------
    # figures: standard 6 (same functions as baseline)
    # -----------------------------------------------------------------------
    print('[ablation01] generating figures...')
    _generate_figures(per_tract_df, bg_validation, per_tract_bg, importance_agg)

    # -----------------------------------------------------------------------
    # figures: 2 comparison figures vs baseline
    # -----------------------------------------------------------------------
    _generate_comparison_figures(per_tract_df)

    # -----------------------------------------------------------------------
    # delta_vs_baseline.json
    # -----------------------------------------------------------------------
    _write_delta(agg_metrics, bg_validation, importance_agg, per_tract_df)

    # -----------------------------------------------------------------------
    # README
    # -----------------------------------------------------------------------
    ts_end = datetime.now()
    _write_readme(per_tract_df, agg_metrics, bg_validation, importance_agg,
                  ts_start, ts_end, tract_list)

    print(f'\n[ablation01] complete: {ts_end.strftime("%Y-%m-%d %H:%M:%S")}')
    elapsed = (ts_end - ts_start).seconds
    print(f'[ablation01] elapsed: {elapsed // 60}m {elapsed % 60}s')


# ---------------------------------------------------------------------------
# scaler diagnostics
# ---------------------------------------------------------------------------

def _save_scaler_diagnostics(all_tract_mu, all_tract_sigma, zero_var_records):
    """Save per_tract_scalers.npz and zero_var_columns.csv."""
    # pack mu / sigma into npz
    npz_data = {}
    for (fips, arch), mu in all_tract_mu.items():
        key_mu    = f'{fips}__{arch}__mu'
        key_sigma = f'{fips}__{arch}__sigma'
        npz_data[key_mu]    = mu
        npz_data[key_sigma] = all_tract_sigma.get((fips, arch), np.array([]))

    npz_path = RESULTS_DIR / 'per_tract_scalers.npz'
    if npz_data:
        np.savez(str(npz_path), **npz_data)
        print(f'[ablation01] per_tract_scalers.npz written ({len(npz_data)//2} tract-arch pairs)')
    else:
        # write empty npz so the file always exists
        np.savez(str(npz_path))
        print('[ablation01] per_tract_scalers.npz written (empty; no per_tract scaler data collected)')

    # zero_var_columns.csv
    zv_path = RESULTS_DIR / 'zero_var_columns.csv'
    if zero_var_records:
        pd.DataFrame(zero_var_records).to_csv(zv_path, index=False)
        print(f'[ablation01] zero_var_columns.csv written ({len(zero_var_records)} entries)')
    else:
        pd.DataFrame(columns=['architecture', 'fips', 'tract', 'feature_idx']).to_csv(
            zv_path, index=False
        )
        print('[ablation01] zero_var_columns.csv written (no zero-variance columns detected)')


# ---------------------------------------------------------------------------
# feature importance aggregation (mirrors baseline)
# ---------------------------------------------------------------------------

def _write_feature_importance(importance_by_arch):
    agg = {}
    for arch, dfs in importance_by_arch.items():
        if not dfs:
            agg[arch] = pd.DataFrame()
            continue
        combined = pd.concat(dfs, ignore_index=True)
        grouped = (
            combined.groupby('feature')['importance']
            .agg(['mean', 'std', 'count'])
            .reset_index()
            .rename(columns={'mean': 'mean_drop', 'std': 'std_drop', 'count': 'n_tracts'})
            .sort_values('mean_drop', ascending=False)
            .reset_index(drop=True)
        )
        grouped['rank'] = grouped.index + 1
        agg[arch] = grouped
        # write with spec name (sage_importance.csv / gcngat_importance.csv)
        short_name = 'sage' if arch == 'sage' else 'gcngat'
        grouped.to_csv(FI_DIR / f'{short_name}_importance.csv', index=False)
        print(f'[ablation01] feature_importance/{short_name}_importance.csv written')
    return agg


# ---------------------------------------------------------------------------
# figure generation: standard 6
# ---------------------------------------------------------------------------

def _generate_figures(per_tract_df, bg_validation, per_tract_bg, importance_agg):
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

    for name, fn, kwargs in [
        ('constraint_error_dist', plot_ablation_constraint_error_dist,
         {'df': success_df, 'output_path': str(FIGURES_DIR / 'constraint_error_dist.png')}),
        ('spatial_std_by_svi', plot_ablation_spatial_std_by_svi,
         {'df': success_df, 'output_path': str(FIGURES_DIR / 'spatial_std_by_svi.png')}),
        ('morans_i_by_tract', plot_ablation_morans_i_by_tract,
         {'df': success_df, 'output_path': str(FIGURES_DIR / 'morans_i_by_tract.png')}),
        ('block_group_scatter', plot_ablation_block_group_scatter,
         {'per_tract_bg': per_tract_bg, 'bg_validation': bg_validation,
          'output_path': str(FIGURES_DIR / 'block_group_scatter.png')}),
    ]:
        try:
            fn(**kwargs)
            print(f'[ablation01] figures/{name}.png written')
        except Exception as e:
            print(f'[ablation01] WARNING: {name} failed: {e}')

    sage_imp   = importance_agg.get('sage', pd.DataFrame())
    gcngat_imp = importance_agg.get('gcn_gat', pd.DataFrame())

    for name, fn, kwargs in [
        ('feature_importance_top20', plot_ablation_feature_importance_top20,
         {'sage_imp': sage_imp, 'gcngat_imp': gcngat_imp,
          'output_path': str(FIGURES_DIR / 'feature_importance_top20.png')}),
        ('architecture_overlap', plot_ablation_architecture_overlap,
         {'sage_imp': sage_imp, 'gcngat_imp': gcngat_imp,
          'output_path': str(FIGURES_DIR / 'architecture_overlap.png')}),
    ]:
        try:
            fn(**kwargs)
            print(f'[ablation01] figures/{name}.png written')
        except Exception as e:
            print(f'[ablation01] WARNING: {name} failed: {e}')


# ---------------------------------------------------------------------------
# figure generation: 2 comparison figures
# ---------------------------------------------------------------------------

def _generate_comparison_figures(per_tract_df):
    """Generate comparison_spatial_std.png and comparison_morans_i.png."""
    baseline_csv = BASELINE_RESULTS / 'per_tract_metrics.csv'
    if not baseline_csv.exists():
        print('[ablation01] WARNING: baseline per_tract_metrics.csv not found; skipping comparison figures')
        return

    base_df = pd.read_csv(baseline_csv)
    curr_df = per_tract_df.copy()

    success_base = base_df[base_df['failure'].isna() | (base_df['failure'] == '')].copy()
    success_curr = curr_df[curr_df['failure'].isna() | (curr_df['failure'] == '')].copy()

    _plot_comparison_spatial_std(success_base, success_curr)
    _plot_comparison_morans_i(success_base, success_curr)


def _plot_comparison_spatial_std(base_df, curr_df):
    """
    Two-panel figure: spatial std vs tract SVI, baseline (gray) vs per-tract (color).
    Linear-fit slope per series shown in legend.
    """
    arch_color = {'sage': '#2196F3', 'gcn_gat': '#FF5722'}
    arch_label = {'sage': 'GRANITE-SAGE', 'gcn_gat': 'GRANITE-GCNGAT'}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, arch in zip(axes, ['sage', 'gcn_gat']):
        b = base_df[base_df['architecture'] == arch]
        c = curr_df[curr_df['architecture'] == arch]

        def _slope_label(sub, label, color, marker, alpha=0.7):
            if len(sub) < 2:
                return
            x = sub['tract_svi'].values
            y = sub['spatial_std'].values
            ax.scatter(x, y, color=color, marker=marker, alpha=alpha, s=50)
            if len(sub) > 2:
                z = np.polyfit(x, y, 1)
                slope = z[0]
                x_line = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_line, np.polyval(z, x_line), color=color,
                        linestyle='--', linewidth=1.2, alpha=0.8,
                        label=f'{label} (slope={slope:.4f})')
            else:
                ax.scatter([], [], color=color, label=label)

        _slope_label(b, f'baseline', '#888888', 'o', alpha=0.5)
        _slope_label(c, f'per-tract std', arch_color[arch], 's')

        ax.set_title(arch_label[arch], fontsize=11)
        ax.set_xlabel('tract SVI', fontsize=10)
        ax.legend(fontsize=8, loc='upper right')
        ax.tick_params(labelsize=9)

    axes[0].set_ylabel('spatial std (within-tract)', fontsize=10)
    fig.suptitle('Spatial Std vs Tract SVI: Baseline vs Per-Tract Standardization', fontsize=12)
    fig.tight_layout()
    out = FIGURES_DIR / 'comparison_spatial_std.png'
    fig.savefig(str(out), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'[ablation01] figures/comparison_spatial_std.png written')


def _plot_comparison_morans_i(base_df, curr_df):
    """
    Side-by-side strip plot of Moran's I: baseline left, per-tract right.
    Same tract ordering (sorted by tract SVI from curr_df).
    """
    tract_order = (
        curr_df.groupby('fips')['tract_svi'].first()
        .sort_values()
        .index.tolist()
    )
    x_pos = {fips: i for i, fips in enumerate(tract_order)}
    n = len(tract_order)

    arch_styles = {
        'sage':    {'color': '#2196F3', 'marker': 'o', 'label': 'GRANITE-SAGE'},
        'gcn_gat': {'color': '#FF5722', 'marker': 's', 'label': 'GRANITE-GCNGAT'},
    }

    fig, axes = plt.subplots(1, 2, figsize=(max(14, n * 0.6), 5), sharey=True)
    titles = ['Baseline (global RobustScaler)', 'Per-Tract Z-Score']

    for ax, df, title in zip(axes, [base_df, curr_df], titles):
        offsets = {'sage': -0.12, 'gcn_gat': 0.12}
        for arch, style in arch_styles.items():
            sub = df[df['architecture'] == arch]
            xs = [x_pos[f] + offsets[arch] for f in sub['fips'] if f in x_pos]
            ys = [sub.loc[sub['fips'] == f, 'morans_i'].values[0]
                  for f in sub['fips'] if f in x_pos]
            valid = [(x, y) for x, y in zip(xs, ys) if math.isfinite(y)]
            if valid:
                vx, vy = zip(*valid)
                ax.scatter(vx, vy, color=style['color'], marker=style['marker'],
                           alpha=0.8, s=60, label=style['label'])
        ax.axhline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
        ax.set_xticks(range(n))
        ax.set_xticklabels([str(t)[-4:] for t in tract_order], fontsize=7, rotation=45)
        ax.set_xlabel('tract (sorted by SVI)', fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Moran's I", fontsize=10)
    fig.suptitle("Moran's I: Baseline vs Per-Tract Standardization", fontsize=12)
    fig.tight_layout()
    out = FIGURES_DIR / 'comparison_morans_i.png'
    fig.savefig(str(out), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'[ablation01] figures/comparison_morans_i.png written')


# ---------------------------------------------------------------------------
# delta_vs_baseline.json
# ---------------------------------------------------------------------------

def _spatial_std_slope(df, arch):
    """Linear slope of spatial_std vs tract_svi for one architecture."""
    sub = df[(df['architecture'] == arch) &
             (df['failure'].isna() | (df['failure'] == ''))].copy()
    if len(sub) < 3:
        return float('nan')
    try:
        z = np.polyfit(sub['tract_svi'].values, sub['spatial_std'].values, 1)
        return float(z[0])
    except Exception:
        return float('nan')


def _fi_spearman_top10(sage_imp, gcngat_imp):
    """Spearman rho between SAGE and GCN-GAT feature importance ranks; top-10 overlap count."""
    if sage_imp.empty or gcngat_imp.empty:
        return float('nan'), 0
    merged = sage_imp[['feature', 'rank']].merge(
        gcngat_imp[['feature', 'rank']], on='feature', suffixes=('_sage', '_gcngat')
    )
    if len(merged) < 2:
        return float('nan'), 0
    rho, _ = stats.spearmanr(merged['rank_sage'], merged['rank_gcngat'])
    sage_top10   = set(sage_imp.head(10)['feature'])
    gcngat_top10 = set(gcngat_imp.head(10)['feature'])
    overlap = len(sage_top10 & gcngat_top10)
    return float(rho), int(overlap)


def _fmt_delta(val, baseline):
    if val is None or not math.isfinite(float(val) if val is not None else float('nan')):
        return '?'
    if baseline is None or not math.isfinite(float(baseline)):
        return '?'
    return round(float(val) - float(baseline), 6)


def _write_delta(agg_metrics, bg_validation, importance_agg, per_tract_df):
    """Write results/delta_vs_baseline.json."""
    # load baseline per_tract_metrics for slope computation
    baseline_csv = BASELINE_RESULTS / 'per_tract_metrics.csv'
    base_df = pd.read_csv(baseline_csv) if baseline_csv.exists() else pd.DataFrame()

    sage_agg   = agg_metrics.get('sage', {})
    gcngat_agg = agg_metrics.get('gcn_gat', {})

    sage_std_pt   = sage_agg.get('spatial_std_mean')
    gcngat_std_pt = gcngat_agg.get('spatial_std_mean')
    sage_mi_pt    = sage_agg.get('morans_i_mean')
    gcngat_mi_pt  = gcngat_agg.get('morans_i_mean')

    sage_bg_pt   = bg_validation.get('sage', {}).get('pearson_r')
    gcngat_bg_pt = bg_validation.get('gcn_gat', {}).get('pearson_r')

    # slopes
    sage_slope_pt   = _spatial_std_slope(per_tract_df, 'sage')
    gcngat_slope_pt = _spatial_std_slope(per_tract_df, 'gcn_gat')
    sage_slope_base   = _spatial_std_slope(base_df, 'sage')   if not base_df.empty else float('nan')
    gcngat_slope_base = _spatial_std_slope(base_df, 'gcn_gat') if not base_df.empty else float('nan')

    # feature importance
    sage_imp   = importance_agg.get('sage', pd.DataFrame())
    gcngat_imp = importance_agg.get('gcn_gat', pd.DataFrame())
    fi_rho, fi_top10 = _fi_spearman_top10(sage_imp, gcngat_imp)

    delta = {
        'mean_spatial_std': {
            'sage':   {'baseline': BASELINE_AGG['sage']['spatial_std_mean'],
                       'per_tract': sage_std_pt,
                       'delta': _fmt_delta(sage_std_pt, BASELINE_AGG['sage']['spatial_std_mean'])},
            'gcngat': {'baseline': BASELINE_AGG['gcn_gat']['spatial_std_mean'],
                       'per_tract': gcngat_std_pt,
                       'delta': _fmt_delta(gcngat_std_pt, BASELINE_AGG['gcn_gat']['spatial_std_mean'])},
        },
        'mean_morans_i': {
            'sage':   {'baseline': BASELINE_AGG['sage']['morans_i_mean'],
                       'per_tract': sage_mi_pt,
                       'delta': _fmt_delta(sage_mi_pt, BASELINE_AGG['sage']['morans_i_mean'])},
            'gcngat': {'baseline': BASELINE_AGG['gcn_gat']['morans_i_mean'],
                       'per_tract': gcngat_mi_pt,
                       'delta': _fmt_delta(gcngat_mi_pt, BASELINE_AGG['gcn_gat']['morans_i_mean'])},
        },
        'block_group_r': {
            'sage':    {'baseline': BASELINE_BG_R['sage'],
                        'per_tract': sage_bg_pt,
                        'delta': _fmt_delta(sage_bg_pt, BASELINE_BG_R['sage'])},
            'gcngat':  {'baseline': BASELINE_BG_R['gcn_gat'],
                        'per_tract': gcngat_bg_pt,
                        'delta': _fmt_delta(gcngat_bg_pt, BASELINE_BG_R['gcn_gat'])},
            'idw':     {'baseline': BASELINE_BG_R['IDW'],
                        'note': 'IDW unaffected by feature standardization; copied from baseline'},
            'kriging': {'baseline': BASELINE_BG_R['kriging'],
                        'note': 'kriging unaffected by feature standardization; copied from baseline'},
        },
        'spatial_std_slope_vs_svi': {
            'sage':   {'baseline': round(sage_slope_base, 6) if math.isfinite(sage_slope_base) else None,
                       'per_tract': round(sage_slope_pt, 6) if math.isfinite(sage_slope_pt) else None,
                       'delta': _fmt_delta(sage_slope_pt, sage_slope_base)},
            'gcngat': {'baseline': round(gcngat_slope_base, 6) if math.isfinite(gcngat_slope_base) else None,
                       'per_tract': round(gcngat_slope_pt, 6) if math.isfinite(gcngat_slope_pt) else None,
                       'delta': _fmt_delta(gcngat_slope_pt, gcngat_slope_base)},
        },
        'feature_importance_spearman': {
            'baseline': BASELINE_FI_SPEARMAN,
            'per_tract': round(fi_rho, 3) if math.isfinite(fi_rho) else None,
            'delta': _fmt_delta(fi_rho, BASELINE_FI_SPEARMAN),
        },
        'top10_overlap': {
            'baseline':  BASELINE_TOP10_OVERLAP,
            'per_tract': fi_top10,
            'delta': fi_top10 - BASELINE_TOP10_OVERLAP,
        },
    }

    with open(RESULTS_DIR / 'delta_vs_baseline.json', 'w') as f:
        json.dump(delta, f, indent=2,
                  default=lambda x: None if (isinstance(x, float) and math.isnan(x)) else x)
    print('[ablation01] delta_vs_baseline.json written')


# ---------------------------------------------------------------------------
# README
# ---------------------------------------------------------------------------

def _write_readme(per_tract_df, agg_metrics, bg_validation, importance_agg,
                  ts_start, ts_end, tract_list):
    git_sha = (ABLATION_DIR / 'git_state.txt').read_text().strip()

    def _fmt(v, fmt='.4f'):
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            return 'n/a'
        return format(v, fmt)

    sage_agg   = agg_metrics.get('sage', {})
    gcn_agg    = agg_metrics.get('gcn_gat', {})
    sage_bg_r  = _fmt(bg_validation.get('sage',    {}).get('pearson_r'))
    gcn_bg_r   = _fmt(bg_validation.get('gcn_gat', {}).get('pearson_r'))
    idw_bg_r   = _fmt(bg_validation.get('IDW',     {}).get('pearson_r'))
    krig_bg_r  = _fmt(bg_validation.get('kriging', {}).get('pearson_r'))
    elapsed    = (ts_end - ts_start).seconds

    # spatial std slope from per_tract_df
    sage_slope   = _fmt(_spatial_std_slope(per_tract_df, 'sage'), '.5f')
    gcngat_slope = _fmt(_spatial_std_slope(per_tract_df, 'gcn_gat'), '.5f')

    # load baseline slopes
    baseline_csv = BASELINE_RESULTS / 'per_tract_metrics.csv'
    if baseline_csv.exists():
        base_df = pd.read_csv(baseline_csv)
        base_sage_slope   = _fmt(_spatial_std_slope(base_df, 'sage'), '.5f')
        base_gcngat_slope = _fmt(_spatial_std_slope(base_df, 'gcn_gat'), '.5f')
    else:
        base_sage_slope   = 'n/a'
        base_gcngat_slope = 'n/a'

    lines = [
        '# 01_per_tract_std',
        '',
        'Ablation step 2a in the GRANITE series.',
        '',
        '## What changed vs 00_baseline',
        '',
        'Single change: feature standardization replaced from global RobustScaler',
        '(fit across all addresses in a tract, using median/IQR) to per-tract z-score',
        '(mean/std computed separately within each tract). All model architecture,',
        'graph construction, training hyperparameters, tract list, and random seed are',
        'identical to 00_baseline. For single-tract mode the practical effect is that',
        'robust scaling (median/IQR) is replaced by z-score (mean/std); multi-tract',
        'runs would additionally normalize each tract independently.',
        '',
        '## Run metadata',
        '',
        f'| field | value |',
        f'|---|---|',
        f'| git SHA | `{git_sha}` |',
        f'| run timestamp | {ts_start.strftime("%Y-%m-%d %H:%M:%S")} |',
        f'| elapsed | {elapsed // 60}m {elapsed % 60}s |',
        f'| seed | {SEED} |',
        f'| tracts | {len(tract_list)} |',
        f'| architectures | sage, gcn_gat |',
        f'| feature_standardization | per_tract (z-score) |',
        '',
        '## Headline metrics',
        '',
        '| metric | GRANITE-SAGE | GRANITE-GCNGAT |',
        '|---|---|---|',
        f'| constraint error (mean) | {_fmt(sage_agg.get("constraint_error_mean"))} | {_fmt(gcn_agg.get("constraint_error_mean"))} |',
        f'| spatial std (mean) | {_fmt(sage_agg.get("spatial_std_mean"))} | {_fmt(gcn_agg.get("spatial_std_mean"))} |',
        f'| spatial std slope vs SVI | {sage_slope} | {gcngat_slope} |',
        f'| spatial std slope (baseline) | {base_sage_slope} | {base_gcngat_slope} |',
        f"| moran's I (mean) | {_fmt(sage_agg.get('morans_i_mean'))} | {_fmt(gcn_agg.get('morans_i_mean'))} |",
        f'| block-group r (pooled) | {sage_bg_r} | {gcn_bg_r} |',
        '',
        '## Block-group validation',
        '',
        '| method | pooled BG r |',
        '|---|---|',
        f'| GRANITE-SAGE | {sage_bg_r} |',
        f'| GRANITE-GCNGAT | {gcn_bg_r} |',
        f'| IDW | {idw_bg_r} |',
        f'| Kriging | {krig_bg_r} |',
        '',
        'Note: IDW and kriging use no learned features and are unaffected by feature',
        'standardization. Values are re-run on the same addresses for completeness.',
        '',
        '## Artifacts',
        '',
        '### Results',
        '- `results/per_tract_metrics.csv`: one row per (tract, architecture)',
        '- `results/aggregate_metrics.json`: mean/median per architecture',
        '- `results/block_group_validation.json`: pooled BG Pearson r per method',
        '- `results/delta_vs_baseline.json`: delta table vs 00_baseline',
        '- `results/per_tract_scalers.npz`: per-tract mu/sigma for reproducibility',
        '- `results/zero_var_columns.csv`: (tract, feature_idx) pairs where std was clamped',
        '- `results/feature_importance/sage_importance.csv`',
        '- `results/feature_importance/gcngat_importance.csv`',
        '',
        '### Figures',
        '- `figures/constraint_error_dist.png`',
        '- `figures/spatial_std_by_svi.png`',
        '- `figures/morans_i_by_tract.png`',
        '- `figures/block_group_scatter.png`',
        '- `figures/feature_importance_top20.png`',
        '- `figures/architecture_overlap.png`',
        '- `figures/comparison_spatial_std.png`: baseline vs per-tract overlay, slopes in legend',
        '- `figures/comparison_morans_i.png`: side-by-side strip plot, same tract ordering',
        '',
        '## Next step',
        '',
        'Step 2b will swap normalization layers (LayerNorm -> BatchNorm or vice versa)',
        'inside the GNN architecture while reverting feature standardization to global.',
        '',
    ]

    readme_path = ABLATION_DIR / 'README.md'
    with open(readme_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('[ablation01] README.md written')


if __name__ == '__main__':
    main()
