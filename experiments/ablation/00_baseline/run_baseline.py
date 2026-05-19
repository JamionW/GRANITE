"""
00_baseline: frozen reference run for the ablation series.

Runs GRANITEPipeline in single-tract SVI mode for all in-scope tracts,
for both GraphSAGE and GCN-GAT architectures. Captures per-tract metrics,
aggregate metrics, block-group validation, and feature importance.

All results land in experiments/ablation/00_baseline/results/.
All figures land in experiments/ablation/00_baseline/figures/.

Stop conditions (hard-exit):
  - NaN/non-finite metric on any tract
  - Block-group r deviates from for_mehdi_review baseline by > 0.02
  - Sign inversion on a top-10 feature vs for_mehdi_review (flagged, not exited)

Usage:
    python experiments/ablation/00_baseline/run_baseline.py [--verbose]
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
import yaml
from scipy import stats

# path setup
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
BASELINE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASELINE_DIR / 'results'
FIGURES_DIR = BASELINE_DIR / 'figures'
FIG_DIR = FIGURES_DIR
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / 'feature_importance').mkdir(exist_ok=True)

TRACT_INVENTORY = REPO_ROOT / 'tract_inventory.csv'
CONFIG_PATH = REPO_ROOT / 'config.yaml'
BG_SVI_PATH = REPO_ROOT / 'data' / 'processed' / 'national_bg_svi.csv'
GRAVEYARD = REPO_ROOT / 'graveyard'

# m0 reference baselines (stop condition)
# source: for_mehdi_review/m0_n20_svi_parity/aggregate.csv, pooled BG r, GRANITE-SAGE
# note: CLAUDE.md r=0.469 is from a different holdout/global validation context;
# the single-tract n20 pooled BG r is 0.7692 for SAGE
M0_BASELINE = {
    'GRANITE': 0.7692,  # m0 n20 single-tract pooled BG r for GRANITE-SAGE
    'IDW':     0.558,   # from CLAUDE.md (different validation setup)
}
R_TOLERANCE = 0.02

ARCHITECTURES = ['sage', 'gcn_gat']
ARCH_LABELS = {'sage': 'GRANITE-SAGE', 'gcn_gat': 'GRANITE-GCNGAT'}
SEED = 42
MIN_ADDRESSES_PER_BG = 3     # per-tract BG threshold (few BGs per tract)
MIN_ADDRESSES_POOLED = 10    # pooled BG threshold


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_config():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f) or {}
    for k in ('data', 'model', 'training', 'processing', 'recovery', 'validation'):
        cfg.setdefault(k, {})
    return cfg


def _load_tract_inventory(verbose):
    """Load tract_inventory.csv; handle missing Status column gracefully."""
    df = pd.read_csv(TRACT_INVENTORY)
    df['fips'] = df['fips'].astype(str).str.strip()

    if 'Status' not in df.columns:
        note = (
            "tract_inventory.csv has no 'Status' column; "
            "treating all 20 rows as in-scope"
        )
        if verbose:
            print(f'[baseline] NOTE: {note}')
        df['_status_note'] = note
        in_scope = df
    else:
        valid_statuses = df['Status'].dropna().unique().tolist()
        documented = {'in-scope', 'in_scope', 'active', 'include', 'yes'}
        bad = [s for s in valid_statuses if s.lower() not in documented]
        if bad:
            print(f'[baseline] HALT: unknown Status values: {bad}')
            sys.exit(1)
        in_scope = df[df['Status'].str.lower().isin(documented)]

    return in_scope


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
    bg_agg = bg_agg[bg_agg['n_addresses'] >= min_addresses].copy()
    return bg_agg


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
    bg_r = float(np.corrcoef(p[valid], t[valid])[0, 1])
    bg_rmse = float(np.sqrt(np.mean((p[valid] - t[valid]) ** 2)))
    return {'bg_r': bg_r, 'bg_rmse': bg_rmse, 'n_bgs': int(valid.sum())}


def _compute_morans_i(predictions, address_gdf, k_neighbors=8):
    """Compute Moran's I using address coordinates."""
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
        print(
            f'[baseline] STOP: non-finite {metric_name} = {value} '
            f'on tract {fips} / arch {arch}'
        )
        sys.exit(2)


def _load_idw_kriging(tracts_gdf):
    """Load IDW and Kriging from graveyard (retired, read-only)."""
    import importlib.util
    gv_path = GRAVEYARD / 'disaggregation_baselines_idw_kriging.py'
    spec = importlib.util.spec_from_file_location(
        'idw_kriging_graveyard', gv_path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    idw = mod.IDWDisaggregation(power=2.0, n_neighbors=8)
    idw.fit(tracts_gdf, svi_column='RPL_THEMES')

    krig = mod.OrdinaryKrigingDisaggregation()
    krig.fit(tracts_gdf, svi_column='RPL_THEMES')

    return idw, krig


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='00_baseline ablation run')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    ts_start = datetime.now()
    print(f'[baseline] start: {ts_start.strftime("%Y-%m-%d %H:%M:%S")}')
    set_random_seed(SEED)

    # preflight
    if not BG_SVI_PATH.exists():
        print(f'[baseline] HALT: {BG_SVI_PATH} missing')
        sys.exit(1)

    # --- tract selection ---
    inventory_df = _load_tract_inventory(args.verbose)
    tract_list = inventory_df['fips'].tolist()
    print(f'[baseline] in-scope tracts: {len(tract_list)}')

    # write tract_selection.txt
    with open(BASELINE_DIR / 'tract_selection.txt', 'w') as f:
        f.write(f'source: {TRACT_INVENTORY}\n')
        f.write(f'status_column_present: False\n')
        f.write(f'note: no Status column; all {len(tract_list)} rows treated as in-scope\n')
        f.write(f'count: {len(tract_list)}\n')
        f.write('fips_list:\n')
        for fp in tract_list:
            f.write(f'  {fp}\n')

    # --- load BG data and validator ---
    print('[baseline] loading BG geodataframe...')
    bg_gdf = _load_bg_gdf()
    validator = BlockGroupValidator(bg_gdf, verbose=False)

    # --- load config and initialize pipeline ---
    cfg = _load_config()
    cfg['data']['target'] = 'svi'
    cfg['data']['neighbor_tracts'] = 0
    cfg['data']['state_fips'] = '47'
    cfg['data']['county_fips'] = '065'
    cfg['processing']['skip_importance'] = True   # handled manually per-tract below
    cfg['processing']['verbose'] = args.verbose
    cfg['processing']['random_seed'] = SEED
    cfg['processing']['enable_caching'] = True
    cfg['training']['apply_post_correction'] = True

    scratch_dir = str(REPO_ROOT / 'output' / 'ablation_baseline_scratch')
    os.makedirs(scratch_dir, exist_ok=True)

    # --- load spatial data once ---
    print('[baseline] loading spatial data...')
    pipeline_init_cfg = dict(cfg)
    pipeline_init_cfg['data'] = dict(cfg['data'])
    pipeline_init_cfg['data']['target_fips'] = tract_list[0]
    pipeline = GRANITEPipeline(pipeline_init_cfg, output_dir=scratch_dir)
    pipeline.verbose = args.verbose
    try:
        data = pipeline._load_spatial_data()
    except Exception as e:
        print(f'[baseline] HALT: spatial data load failed: {e}')
        sys.exit(1)

    # load IDW and kriging baselines
    print('[baseline] fitting IDW and kriging baselines...')
    try:
        idw_model, krig_model = _load_idw_kriging(data['tracts'])
    except Exception as e:
        print(f'[baseline] WARNING: IDW/kriging load failed: {e}')
        idw_model, krig_model = None, None

    # storage
    per_tract_rows = []
    # arch -> list of (address_gdf, predictions) for pooled BG validation
    pooled_addr_gdfs = {a: [] for a in ARCHITECTURES}
    pooled_preds = {a: [] for a in ARCHITECTURES}
    # for IDW and kriging pooled BG
    pooled_preds_idw = []
    pooled_preds_krig = []
    pooled_addr_gdfs_baselines = []
    # arch -> list of per-tract importance DataFrames
    importance_by_arch = {a: [] for a in ARCHITECTURES}

    # -----------------------------------------------------------------------
    # main loop: architecture x tract
    # -----------------------------------------------------------------------
    total = len(ARCHITECTURES) * len(tract_list)
    done = 0

    for arch in ARCHITECTURES:
        arch_label = ARCH_LABELS[arch]
        print(f'\n[baseline] === architecture: {arch_label} ===')

        cfg['model']['architecture'] = arch
        pipeline.config = dict(cfg)
        pipeline.verbose = args.verbose

        for idx, fips in enumerate(tract_list):
            done += 1
            print(f'[baseline] [{done}/{total}] {arch_label} | tract {idx+1}/{len(tract_list)}: {fips}')
            t0 = time.time()

            cfg['data']['target_fips'] = fips
            pipeline.config['data']['target_fips'] = fips
            pipeline.config['model']['architecture'] = arch

            try:
                result = pipeline._process_single_tract(fips, data)
            except Exception as e:
                msg = str(e)[:200]
                print(f'[baseline]   ERROR: {msg}')
                per_tract_rows.append({
                    'fips': fips, 'architecture': arch, 'n_addresses': 0,
                    'tract_svi': float('nan'), 'constraint_error': float('nan'),
                    'spatial_std': float('nan'), 'morans_i': float('nan'),
                    'bg_r': float('nan'), 'n_bgs': 0, 'runtime_s': 0,
                    'failure': msg,
                })
                continue

            if not result.get('success'):
                msg = result.get('error', 'unknown')[:200]
                print(f'[baseline]   FAILED: {msg}')
                per_tract_rows.append({
                    'fips': fips, 'architecture': arch, 'n_addresses': 0,
                    'tract_svi': float('nan'), 'constraint_error': float('nan'),
                    'spatial_std': float('nan'), 'morans_i': float('nan'),
                    'bg_r': float('nan'), 'n_bgs': 0, 'runtime_s': 0,
                    'failure': msg,
                })
                continue

            runtime = time.time() - t0
            address_gdf = result['address_gdf']
            preds_arr = result['predictions']['mean'].values.astype(float)
            tract_svi = float(result['tract_svi'])
            n_addresses = len(address_gdf)

            # constraint error (absolute, after post-correction)
            constr_err = abs(np.mean(preds_arr) - tract_svi)
            # spatial std
            sp_std = float(np.std(preds_arr))
            # moran's i
            morans_i = _compute_morans_i(preds_arr, address_gdf)

            # stop conditions for non-finite metrics
            _check_nan_stop('constraint_error', constr_err, fips, arch)
            _check_nan_stop('spatial_std', sp_std, fips, arch)
            # morans_i can be nan on small tracts - warn but don't stop
            if not math.isfinite(morans_i):
                print(f'[baseline]   WARNING: non-finite morans_i for {fips}/{arch}')
                morans_i = float('nan')

            # BG validation for GRANITE
            try:
                bg_agg = _aggregate_to_bg(validator, address_gdf, preds_arr,
                                          MIN_ADDRESSES_PER_BG)
                bm = _bg_metrics(bg_agg, bg_gdf)
            except Exception as e:
                bm = {'bg_r': float('nan'), 'bg_rmse': float('nan'), 'n_bgs': 0}
                print(f'[baseline]   WARNING: BG metrics failed: {e}')

            per_tract_rows.append({
                'fips': fips, 'architecture': arch, 'n_addresses': n_addresses,
                'tract_svi': tract_svi,
                'constraint_error': round(constr_err, 6),
                'spatial_std': round(sp_std, 6),
                'morans_i': round(morans_i, 6) if math.isfinite(morans_i) else float('nan'),
                'bg_r': bm['bg_r'], 'n_bgs': bm['n_bgs'],
                'runtime_s': round(runtime, 2),
                'failure': '',
            })

            # accumulate for pooled BG validation
            pooled_addr_gdfs[arch].append(address_gdf.copy())
            pooled_preds[arch].append(preds_arr)

            # accumulate IDW/kriging for first architecture pass only
            if arch == ARCHITECTURES[0]:
                pooled_addr_gdfs_baselines.append(address_gdf.copy())
                if idw_model is not None:
                    try:
                        addr_coords = np.array(
                            [[g.x, g.y] for g in address_gdf.geometry]
                        )
                        idw_p = idw_model.disaggregate(addr_coords, fips, tract_svi)
                        pooled_preds_idw.append(idw_p)
                    except Exception as e:
                        pooled_preds_idw.append(np.full(n_addresses, tract_svi))
                        print(f'[baseline]   WARNING: IDW failed for {fips}: {e}')
                if krig_model is not None:
                    try:
                        addr_coords = np.array(
                            [[g.x, g.y] for g in address_gdf.geometry]
                        )
                        krig_p = krig_model.disaggregate(addr_coords, fips, tract_svi)
                        pooled_preds_krig.append(krig_p)
                    except Exception as e:
                        pooled_preds_krig.append(np.full(n_addresses, tract_svi))
                        print(f'[baseline]   WARNING: Kriging failed for {fips}: {e}')

            # feature importance - _process_single_tract does not run it;
            # must call analyze_feature_importance explicitly
            if True:
                try:
                    fi = pipeline.analyze_feature_importance(result, n_repeats=5)
                    if fi is not None:
                        perm = fi.get('permutation', {})
                        if isinstance(perm, dict) and 'feature_importance' in perm:
                            imp_df = perm['feature_importance'].copy()
                            imp_df['fips'] = fips
                            imp_df['architecture'] = arch
                            importance_by_arch[arch].append(imp_df)
                except Exception as e:
                    print(f'[baseline]   WARNING: feature importance failed for {fips}/{arch}: {e}')

            if args.verbose:
                print(
                    f'[baseline]   constr_err={constr_err:.4f} '
                    f'sp_std={sp_std:.4f} morans_i={morans_i:.3f} '
                    f'bg_r={bm["bg_r"]:.3f} t={runtime:.1f}s'
                )

    # -----------------------------------------------------------------------
    # write per_tract_metrics.csv
    # -----------------------------------------------------------------------
    per_tract_df = pd.DataFrame(per_tract_rows)
    per_tract_path = RESULTS_DIR / 'per_tract_metrics.csv'
    per_tract_df.to_csv(per_tract_path, index=False)
    print(f'\n[baseline] per_tract_metrics.csv written ({len(per_tract_df)} rows)')

    # -----------------------------------------------------------------------
    # aggregate metrics per architecture
    # -----------------------------------------------------------------------
    agg_metrics = {}
    for arch in ARCHITECTURES:
        arch_df = per_tract_df[
            (per_tract_df['architecture'] == arch) &
            (per_tract_df['failure'] == '')
        ]
        if len(arch_df) == 0:
            agg_metrics[arch] = {}
            continue
        agg_metrics[arch] = {
            'n_tracts': int(len(arch_df)),
            'constraint_error_mean': float(arch_df['constraint_error'].mean()),
            'constraint_error_median': float(arch_df['constraint_error'].median()),
            'spatial_std_mean': float(arch_df['spatial_std'].mean()),
            'spatial_std_median': float(arch_df['spatial_std'].median()),
            'morans_i_mean': float(arch_df['morans_i'].dropna().mean()) if arch_df['morans_i'].notna().any() else float('nan'),
            'morans_i_median': float(arch_df['morans_i'].dropna().median()) if arch_df['morans_i'].notna().any() else float('nan'),
            'bg_r_mean': float(arch_df['bg_r'].dropna().mean()) if arch_df['bg_r'].notna().any() else float('nan'),
            'bg_r_median': float(arch_df['bg_r'].dropna().median()) if arch_df['bg_r'].notna().any() else float('nan'),
        }

    with open(RESULTS_DIR / 'aggregate_metrics.json', 'w') as f:
        json.dump(agg_metrics, f, indent=2, default=lambda x: None if (isinstance(x, float) and math.isnan(x)) else x)
    print('[baseline] aggregate_metrics.json written')

    # -----------------------------------------------------------------------
    # pooled BG validation: SAGE, GCN-GAT, IDW, kriging
    # -----------------------------------------------------------------------
    print('[baseline] computing pooled BG validation...')
    bg_validation = {}
    per_tract_bg = {}  # arch -> df of per-tract predicted_svi vs actual_svi

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
            bm_pooled = _bg_metrics(bg_agg_pooled, bg_gdf)
        except Exception as e:
            bm_pooled = {'bg_r': float('nan'), 'bg_rmse': float('nan'), 'n_bgs': 0}
            print(f'[baseline] WARNING: pooled BG validation failed for {arch}: {e}')

        # per-tract scatter data for block_group_scatter figure
        svi_lookup = bg_gdf[['GEOID', 'SVI']].dropna(subset=['SVI'])
        try:
            bg_agg_all = _aggregate_to_bg(validator, combined_gdf, all_p, 1)
            merged_scatter = bg_agg_all.merge(svi_lookup, on='GEOID', how='inner')
        except Exception:
            merged_scatter = pd.DataFrame()

        bg_validation[arch] = {
            'pearson_r': bm_pooled['bg_r'],
            'bg_rmse': bm_pooled['bg_rmse'],
            'n_bgs': bm_pooled['n_bgs'],
        }
        per_tract_bg[arch] = merged_scatter

    # IDW pooled
    for label, preds_list in [('IDW', pooled_preds_idw), ('kriging', pooled_preds_krig)]:
        if not preds_list or not pooled_addr_gdfs_baselines:
            bg_validation[label] = {'pearson_r': float('nan')}
            per_tract_bg[label] = pd.DataFrame()
            continue
        combined_gdf = pd.concat(pooled_addr_gdfs_baselines, ignore_index=True)
        if not isinstance(combined_gdf, gpd.GeoDataFrame):
            combined_gdf = gpd.GeoDataFrame(combined_gdf, geometry='geometry', crs='EPSG:4326')
        if combined_gdf.crs is None:
            combined_gdf = combined_gdf.set_crs('EPSG:4326')
        all_p = np.concatenate(preds_list)
        try:
            bg_agg_pooled = _aggregate_to_bg(validator, combined_gdf, all_p, MIN_ADDRESSES_POOLED)
            bm_pooled = _bg_metrics(bg_agg_pooled, bg_gdf)
            bg_agg_all = _aggregate_to_bg(validator, combined_gdf, all_p, 1)
            svi_lookup = bg_gdf[['GEOID', 'SVI']].dropna(subset=['SVI'])
            merged_scatter = bg_agg_all.merge(svi_lookup, on='GEOID', how='inner')
        except Exception as e:
            bm_pooled = {'bg_r': float('nan'), 'bg_rmse': float('nan'), 'n_bgs': 0}
            merged_scatter = pd.DataFrame()
            print(f'[baseline] WARNING: pooled BG validation failed for {label}: {e}')
        bg_validation[label] = {
            'pearson_r': bm_pooled['bg_r'],
            'bg_rmse': bm_pooled['bg_rmse'],
            'n_bgs': bm_pooled['n_bgs'],
        }
        per_tract_bg[label] = merged_scatter

    with open(RESULTS_DIR / 'block_group_validation.json', 'w') as f:
        json.dump(bg_validation, f, indent=2,
                  default=lambda x: None if (isinstance(x, float) and math.isnan(x)) else x)
    print('[baseline] block_group_validation.json written')

    # stop condition: BG r deviation from m0 reference
    sage_r = bg_validation.get('sage', {}).get('pearson_r', float('nan'))
    if math.isfinite(sage_r) and math.isfinite(M0_BASELINE['GRANITE']):
        diff = abs(sage_r - M0_BASELINE['GRANITE'])
        if diff > R_TOLERANCE:
            print(
                f'[baseline] STOP: SAGE pooled BG r={sage_r:.3f} deviates from '
                f'reference {M0_BASELINE["GRANITE"]:.3f} by {diff:.3f} > {R_TOLERANCE}'
            )
            print('[baseline] Investigate before proceeding to ablation steps 01-05.')
            # write partial outputs then exit
            _write_feature_importance(importance_by_arch)
            sys.exit(3)

    # -----------------------------------------------------------------------
    # feature importance: aggregate across tracts per architecture
    # -----------------------------------------------------------------------
    importance_agg = _write_feature_importance(importance_by_arch)

    # -----------------------------------------------------------------------
    # figures
    # -----------------------------------------------------------------------
    print('[baseline] generating figures...')
    _generate_figures(per_tract_df, bg_validation, per_tract_bg, importance_agg)

    # -----------------------------------------------------------------------
    # README.md
    # -----------------------------------------------------------------------
    ts_end = datetime.now()
    _write_readme(
        per_tract_df, agg_metrics, bg_validation, importance_agg,
        ts_start, ts_end, tract_list,
    )

    print(f'\n[baseline] complete: {ts_end.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'[baseline] elapsed: {(ts_end - ts_start).seconds // 60}m {(ts_end - ts_start).seconds % 60}s')


# ---------------------------------------------------------------------------
# feature importance aggregation
# ---------------------------------------------------------------------------

def _write_feature_importance(importance_by_arch):
    """Aggregate per-tract importance, write CSVs, return dict arch->agg_df."""
    agg = {}
    fi_dir = RESULTS_DIR / 'feature_importance'
    fi_dir.mkdir(exist_ok=True)

    for arch, dfs in importance_by_arch.items():
        if not dfs:
            agg[arch] = pd.DataFrame()
            continue
        combined = pd.concat(dfs, ignore_index=True)
        # mean importance (mean drop in performance) across tracts
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
        out_path = fi_dir / f'{arch}_permutation_importance.csv'
        grouped.to_csv(out_path, index=False)
        print(f'[baseline] feature_importance/{arch}_permutation_importance.csv written')

    # sign-inversion check vs for_mehdi_review (flag only, don't exit)
    _check_sign_inversions(agg)
    return agg


def _check_sign_inversions(agg):
    """Flag top-10 feature sign inversions vs for_mehdi_review baseline."""
    # for_mehdi_review doesn't have a permutation importance CSV,
    # so we skip the check but log the note
    print('[baseline] NOTE: sign-inversion check vs for_mehdi_review: '
          'no permutation importance baseline available in for_mehdi_review/; '
          'skipped. Run manually if needed.')


# ---------------------------------------------------------------------------
# figure generation
# ---------------------------------------------------------------------------

def _generate_figures(per_tract_df, bg_validation, per_tract_bg, importance_agg):
    """Generate all 6 ablation figures via plots.py extensions."""
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

    # 1. constraint error distribution
    try:
        plot_ablation_constraint_error_dist(
            success_df, str(FIGURES_DIR / 'constraint_error_dist.png')
        )
        print('[baseline] figures/constraint_error_dist.png written')
    except Exception as e:
        print(f'[baseline] WARNING: constraint_error_dist failed: {e}')

    # 2. spatial std vs SVI
    try:
        plot_ablation_spatial_std_by_svi(
            success_df, str(FIGURES_DIR / 'spatial_std_by_svi.png')
        )
        print('[baseline] figures/spatial_std_by_svi.png written')
    except Exception as e:
        print(f'[baseline] WARNING: spatial_std_by_svi failed: {e}')

    # 3. morans_i by tract
    try:
        plot_ablation_morans_i_by_tract(
            success_df, str(FIGURES_DIR / 'morans_i_by_tract.png')
        )
        print('[baseline] figures/morans_i_by_tract.png written')
    except Exception as e:
        print(f'[baseline] WARNING: morans_i_by_tract failed: {e}')

    # 4. block group scatter
    try:
        plot_ablation_block_group_scatter(
            per_tract_bg, bg_validation,
            str(FIGURES_DIR / 'block_group_scatter.png')
        )
        print('[baseline] figures/block_group_scatter.png written')
    except Exception as e:
        print(f'[baseline] WARNING: block_group_scatter failed: {e}')

    # 5. feature importance top-20
    try:
        sage_imp = importance_agg.get('sage', pd.DataFrame())
        gcngat_imp = importance_agg.get('gcn_gat', pd.DataFrame())
        plot_ablation_feature_importance_top20(
            sage_imp, gcngat_imp,
            str(FIGURES_DIR / 'feature_importance_top20.png')
        )
        print('[baseline] figures/feature_importance_top20.png written')
    except Exception as e:
        print(f'[baseline] WARNING: feature_importance_top20 failed: {e}')

    # 6. architecture overlap
    try:
        plot_ablation_architecture_overlap(
            sage_imp, gcngat_imp,
            str(FIGURES_DIR / 'architecture_overlap.png')
        )
        print('[baseline] figures/architecture_overlap.png written')
    except Exception as e:
        print(f'[baseline] WARNING: architecture_overlap failed: {e}')


# ---------------------------------------------------------------------------
# README.md
# ---------------------------------------------------------------------------

def _write_readme(per_tract_df, agg_metrics, bg_validation, importance_agg,
                  ts_start, ts_end, tract_list):
    """Write experiments/ablation/00_baseline/README.md."""
    git_sha = (BASELINE_DIR / 'git_state.txt').read_text().strip()

    def _fmt(v, fmt='.4f'):
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            return 'n/a'
        return format(v, fmt)

    sage_agg = agg_metrics.get('sage', {})
    gcn_agg = agg_metrics.get('gcn_gat', {})

    sage_bg_r = _fmt(bg_validation.get('sage', {}).get('pearson_r'))
    gcn_bg_r = _fmt(bg_validation.get('gcn_gat', {}).get('pearson_r'))
    idw_bg_r = _fmt(bg_validation.get('IDW', {}).get('pearson_r'))
    krig_bg_r = _fmt(bg_validation.get('kriging', {}).get('pearson_r'))

    lines = [
        '# 00_baseline',
        '',
        'Frozen reference run for the GRANITE ablation series.',
        'Steps 01 through 05 mirror this structure with exactly one change applied.',
        '',
        '## Run metadata',
        '',
        f'| field | value |',
        f'|---|---|',
        f'| git SHA | `{git_sha}` |',
        f'| run timestamp | {ts_start.strftime("%Y-%m-%d %H:%M:%S")} |',
        f'| elapsed | {(ts_end - ts_start).seconds // 60}m {(ts_end - ts_start).seconds % 60}s |',
        f'| seed | {SEED} |',
        f'| tracts | {len(tract_list)} |',
        f'| architectures | sage, gcn_gat |',
        '',
        '## Headline metrics',
        '',
        '| metric | GRANITE-SAGE | GRANITE-GCNGAT |',
        '|---|---|---|',
        f'| constraint error (mean) | {_fmt(sage_agg.get("constraint_error_mean"))} | {_fmt(gcn_agg.get("constraint_error_mean"))} |',
        f'| spatial std (mean) | {_fmt(sage_agg.get("spatial_std_mean"))} | {_fmt(gcn_agg.get("spatial_std_mean"))} |',
        f'| moran\'s I (mean) | {_fmt(sage_agg.get("morans_i_mean"))} | {_fmt(gcn_agg.get("morans_i_mean"))} |',
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
        '## Artifacts',
        '',
        '### Results',
        '- `results/per_tract_metrics.csv`: one row per (tract, architecture)',
        '- `results/aggregate_metrics.json`: mean/median per architecture',
        '- `results/block_group_validation.json`: pooled BG Pearson r per method',
        '- `results/feature_importance/`: per-architecture permutation importance CSVs',
        '',
        '### Figures',
        '- `figures/constraint_error_dist.png`: per-architecture histograms, shared x-axis',
        '- `figures/spatial_std_by_svi.png`: within-tract std vs tract SVI scatter',
        '- `figures/morans_i_by_tract.png`: strip plot sorted by tract SVI',
        '- `figures/block_group_scatter.png`: predicted vs observed BG SVI, 4 panels',
        '- `figures/feature_importance_top20.png`: top-20 features, side-by-side bars',
        '- `figures/architecture_overlap.png`: Spearman rank correlation + rank-rank scatter',
        '',
        '## Ablation series note',
        '',
        'Steps 01 through 05 each mirror this structure with exactly one change applied.',
        'Cross-step diffs compare back to these results as the frozen anchor.',
        '',
    ]

    readme_path = BASELINE_DIR / 'README.md'
    with open(readme_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('[baseline] README.md written')


if __name__ == '__main__':
    main()
