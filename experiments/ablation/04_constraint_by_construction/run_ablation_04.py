"""
04_constraint_by_construction: test constraint-by-construction output formulation.

three modes:
  soft            -- standard soft constraint loss + post-hoc shift (baseline for this step)
  cbc_with_shift  -- cbc in train loop; post-hoc shift still applied
  cbc_no_shift    -- cbc in train loop; post-hoc shift skipped

usage:
    python experiments/ablation/04_constraint_by_construction/run_ablation_04.py --mode soft --subdir 00_baseline_for_step4
    python experiments/ablation/04_constraint_by_construction/run_ablation_04.py --mode cbc_with_shift --subdir 01_cbc_no_constraint_loss
    python experiments/ablation/04_constraint_by_construction/run_ablation_04.py --mode cbc_no_shift --subdir 02_cbc_no_shift
    python experiments/ablation/04_constraint_by_construction/run_ablation_04.py --write-summary
"""
import argparse
import json
import math
import os
import subprocess
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
TRACT_SELECTION = REPO_ROOT / 'experiments' / 'ablation' / '00_baseline' / 'tract_selection.txt'
BASE_CONFIG_PATH = (
    REPO_ROOT / 'experiments' / 'ablation' / '03_smoothness' / '02_default' / 'config_snapshot.yaml'
)
BG_SVI_PATH = REPO_ROOT / 'data' / 'processed' / 'national_bg_svi.csv'
GRAVEYARD = REPO_ROOT / 'graveyard'

ARCHITECTURES = ['sage', 'gcn_gat']
ARCH_LABELS = {'sage': 'GRANITE-SAGE', 'gcn_gat': 'GRANITE-GCNGAT'}
SEED = 42
MIN_ADDRESSES_PER_BG = 3
MIN_ADDRESSES_POOLED = 10

# reference r values from ablation step 2a per-tract standardization
# used to check that '00_baseline_for_step4' reproduces the reference within 1%
STEP4_REFERENCE = {
    'sage': 0.754,
    'gcn_gat': 0.766,
}
STEP4_TOLERANCE = 0.01  # 1% absolute


# ---------------------------------------------------------------------------
# helpers shared across modes
# ---------------------------------------------------------------------------

def _load_tract_list():
    fips = []
    in_list = False
    with open(TRACT_SELECTION) as f:
        for line in f:
            line = line.strip()
            if line == 'fips_list:':
                in_list = True
                continue
            if in_list and line:
                fips.append(line.lstrip('- ').strip())
    return fips


def _load_base_config():
    with open(BASE_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f) or {}
    for k in ('data', 'model', 'training', 'processing', 'features', 'recovery', 'validation', 'norm_layers'):
        cfg.setdefault(k, {})
    # remove smoothness_weight -- it is no longer valid (step 3 finding)
    cfg.get('training', {}).pop('smoothness_weight', None)
    # remove variation_weight if present -- hardcoded in trainer
    cfg.get('training', {}).pop('variation_weight', None)
    return cfg


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
    try:
        coords = np.array([[g.x, g.y] for g in address_gdf.geometry])
        diag = SpatialLearningDiagnostics()
        return diag.compute_spatial_autocorrelation(predictions, coords, k_neighbors=k_neighbors)
    except Exception:
        return float('nan')


def _check_nan_stop(metric_name, value, fips, arch):
    if not math.isfinite(value):
        print(f'[ablation04] STOP: non-finite {metric_name}={value} on tract {fips} / arch {arch}')
        sys.exit(2)


def _load_idw_kriging(tracts_gdf):
    import importlib.util
    gv_path = GRAVEYARD / 'disaggregation_baselines_idw_kriging.py'
    spec = importlib.util.spec_from_file_location('idw_kriging_graveyard', gv_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    idw = mod.IDWDisaggregation(power=2.0, n_neighbors=8)
    idw.fit(tracts_gdf, svi_column='RPL_THEMES')
    krig = mod.OrdinaryKrigingDisaggregation()
    krig.fit(tracts_gdf, svi_column='RPL_THEMES')
    return idw, krig


def _write_preflight(subdir_path, cfg, tract_list):
    """write git_state.txt, config_snapshot.yaml, environment.txt, tract_selection.txt."""
    # git state
    try:
        sha = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=str(REPO_ROOT)
        ).decode().strip()
        diff_stat = subprocess.check_output(
            ['git', 'diff', '--stat'], cwd=str(REPO_ROOT)
        ).decode().strip()
        git_txt = f'sha: {sha}\ndiff_stat:\n{diff_stat}\n'
    except Exception as e:
        git_txt = f'git error: {e}\n'
    (subdir_path / 'git_state.txt').write_text(git_txt)

    # config snapshot
    with open(subdir_path / 'config_snapshot.yaml', 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # environment
    try:
        env_txt = subprocess.check_output(['pip', 'freeze']).decode()
    except Exception:
        env_txt = 'pip freeze failed\n'
    (subdir_path / 'environment.txt').write_text(env_txt)

    # tract selection
    import shutil
    shutil.copy(str(TRACT_SELECTION), str(subdir_path / 'tract_selection.txt'))
    print(f'[ablation04] preflight files written to {subdir_path}')


# ---------------------------------------------------------------------------
# main run
# ---------------------------------------------------------------------------

def run_mode(mode, subdir):
    ts_start = datetime.now()
    print(f'[ablation04] start: mode={mode} subdir={subdir} {ts_start.strftime("%Y-%m-%d %H:%M:%S")}')
    set_random_seed(SEED)

    subdir_path = ABLATION_DIR / subdir
    results_dir = subdir_path / 'results'
    figures_dir = subdir_path / 'figures'
    (results_dir / 'feature_importance').mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # preflight checks
    if not BG_SVI_PATH.exists():
        print(f'[ablation04] HALT: {BG_SVI_PATH} missing')
        sys.exit(1)

    tract_list = _load_tract_list()
    print(f'[ablation04] tracts: {len(tract_list)}')

    # build config
    cfg = _load_base_config()
    cfg['data']['target'] = 'svi'
    cfg['data']['neighbor_tracts'] = 0
    cfg['data']['state_fips'] = '47'
    cfg['data']['county_fips'] = '065'
    cfg['processing']['skip_importance'] = True
    cfg['processing']['verbose'] = False
    cfg['processing']['random_seed'] = SEED
    cfg['processing']['enable_caching'] = True
    cfg['features']['feature_standardization'] = 'per_tract'
    cfg['training']['constraint_mode'] = mode

    # cbc_no_shift: disable post-hoc additive shift
    if mode == 'cbc_no_shift':
        cfg['training']['apply_post_correction'] = False
    else:
        cfg['training']['apply_post_correction'] = True

    _write_preflight(subdir_path, cfg, tract_list)

    # load bg data
    print('[ablation04] loading BG geodataframe...')
    bg_gdf = _load_bg_gdf()
    validator = BlockGroupValidator(bg_gdf, verbose=False)

    # init pipeline
    scratch_dir = str(REPO_ROOT / 'output' / f'ablation04_{subdir}_scratch')
    os.makedirs(scratch_dir, exist_ok=True)
    pipeline_cfg = dict(cfg)
    pipeline_cfg['data'] = dict(cfg['data'])
    pipeline_cfg['data']['target_fips'] = tract_list[0]
    pipeline = GRANITEPipeline(pipeline_cfg, output_dir=scratch_dir)
    pipeline.verbose = False

    # load spatial data once
    print('[ablation04] loading spatial data...')
    try:
        data = pipeline._load_spatial_data()
    except Exception as e:
        print(f'[ablation04] HALT: spatial data load failed: {e}')
        sys.exit(1)

    # IDW/kriging baselines
    print('[ablation04] fitting IDW and kriging baselines...')
    try:
        idw_model, krig_model = _load_idw_kriging(data['tracts'])
    except Exception as e:
        print(f'[ablation04] WARNING: IDW/kriging load failed: {e}')
        idw_model, krig_model = None, None

    # storage
    per_tract_rows = []
    pooled_addr_gdfs = {a: [] for a in ARCHITECTURES}
    pooled_preds = {a: [] for a in ARCHITECTURES}
    pooled_preds_idw = []
    pooled_preds_krig = []
    pooled_addr_gdfs_baselines = []
    importance_by_arch = {a: [] for a in ARCHITECTURES}
    # for tract_mean_diagnostic (cbc_no_shift only)
    tract_mean_diag_rows = []

    total = len(ARCHITECTURES) * len(tract_list)
    done = 0

    for arch in ARCHITECTURES:
        arch_label = ARCH_LABELS[arch]
        print(f'\n[ablation04] === {arch_label} ===')
        cfg['model']['architecture'] = arch
        pipeline.config = dict(cfg)
        pipeline.verbose = False

        for idx, fips in enumerate(tract_list):
            done += 1
            print(f'[ablation04] [{done}/{total}] {arch_label} | tract {idx+1}/{len(tract_list)}: {fips}')
            t0 = time.time()

            cfg['data']['target_fips'] = fips
            pipeline.config['data']['target_fips'] = fips
            pipeline.config['model']['architecture'] = arch

            try:
                result = pipeline._process_single_tract(fips, data)
            except Exception as e:
                msg = str(e)[:300]
                print(f'[ablation04]   ERROR: {msg}')
                traceback.print_exc()
                per_tract_rows.append({
                    'fips': fips, 'architecture': arch, 'n_addresses': 0,
                    'tract_svi': float('nan'), 'constraint_error': float('nan'),
                    'pre_correction_constraint_error': float('nan'),
                    'spatial_std': float('nan'), 'morans_i': float('nan'),
                    'bg_r': float('nan'), 'n_bgs': 0, 'runtime_s': 0, 'failure': msg,
                })
                continue

            if not result.get('success'):
                msg = result.get('error', 'unknown')[:300]
                print(f'[ablation04]   FAILED: {msg}')
                per_tract_rows.append({
                    'fips': fips, 'architecture': arch, 'n_addresses': 0,
                    'tract_svi': float('nan'), 'constraint_error': float('nan'),
                    'pre_correction_constraint_error': float('nan'),
                    'spatial_std': float('nan'), 'morans_i': float('nan'),
                    'bg_r': float('nan'), 'n_bgs': 0, 'runtime_s': 0, 'failure': msg,
                })
                continue

            runtime = time.time() - t0
            address_gdf = result['address_gdf']
            preds_arr = result['predictions']['mean'].values.astype(float)
            tract_svi = float(result['tract_svi'])
            n_addresses = len(address_gdf)

            # constraint error after any post-correction
            constr_err = abs(np.mean(preds_arr) - tract_svi)

            # pre_correction_constraint_error: raw model output mean minus tract target,
            # before post-hoc shift. key is 'raw_predictions' in the single-tract
            # training_result dict (pipeline renames final_predictions -> raw_predictions).
            training_result = result.get('training_result', {})
            raw_preds = training_result.get('raw_predictions', None)
            if raw_preds is not None:
                pre_corr_err = abs(float(np.mean(raw_preds)) - tract_svi)
            else:
                pre_corr_err = float('nan')

            sp_std = float(np.std(preds_arr))
            morans_i = _compute_morans_i(preds_arr, address_gdf)

            # stop conditions
            _check_nan_stop('constraint_error', constr_err, fips, arch)
            _check_nan_stop('spatial_std', sp_std, fips, arch)
            if not math.isfinite(morans_i):
                print(f'[ablation04]   WARNING: non-finite morans_i for {fips}/{arch}')
                morans_i = float('nan')

            # cbc_no_shift stop: constraint error must be < 1e-6
            if mode == 'cbc_no_shift':
                if math.isfinite(constr_err) and constr_err >= 1e-6:
                    print(
                        f'[ablation04] STOP: cbc_no_shift constraint error {constr_err:.2e} >= 1e-6 '
                        f'on tract {fips} / arch {arch}'
                    )
                    sys.exit(4)

            # warnings
            try:
                bg_agg = _aggregate_to_bg(validator, address_gdf, preds_arr, MIN_ADDRESSES_PER_BG)
                bm = _bg_metrics(bg_agg, bg_gdf)
            except Exception as e:
                bm = {'bg_r': float('nan'), 'bg_rmse': float('nan'), 'n_bgs': 0}
                print(f'[ablation04]   WARNING: BG metrics failed: {e}')

            if math.isfinite(bm['bg_r']) and bm['bg_r'] < 0.6:
                print(f'[ablation04]   WARNING: bg_r={bm["bg_r"]:.3f} < 0.6 for {fips}/{arch}')
            if math.isfinite(sp_std) and sp_std > 0.3:
                print(f'[ablation04]   WARNING: spatial_std={sp_std:.4f} > 0.3 for {fips}/{arch}')

            per_tract_rows.append({
                'fips': fips, 'architecture': arch, 'n_addresses': n_addresses,
                'tract_svi': tract_svi,
                'constraint_error': round(constr_err, 8),
                'pre_correction_constraint_error': round(pre_corr_err, 8) if math.isfinite(pre_corr_err) else float('nan'),
                'spatial_std': round(sp_std, 6),
                'morans_i': round(morans_i, 6) if math.isfinite(morans_i) else float('nan'),
                'bg_r': bm['bg_r'], 'n_bgs': bm['n_bgs'],
                'runtime_s': round(runtime, 2), 'failure': '',
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
                    except Exception as e:
                        pooled_preds_idw.append(np.full(n_addresses, tract_svi))
                        print(f'[ablation04]   WARNING: IDW failed for {fips}: {e}')
                if krig_model is not None:
                    try:
                        addr_coords = np.array([[g.x, g.y] for g in address_gdf.geometry])
                        krig_p = krig_model.disaggregate(addr_coords, fips, tract_svi)
                        pooled_preds_krig.append(krig_p)
                    except Exception as e:
                        pooled_preds_krig.append(np.full(n_addresses, tract_svi))
                        print(f'[ablation04]   WARNING: kriging failed for {fips}: {e}')

            # tract_mean_diagnostic for cbc_no_shift
            if mode == 'cbc_no_shift' and raw_preds is not None:
                raw_subset = raw_preds[:n_addresses] if len(raw_preds) >= n_addresses else raw_preds
                raw_mean = float(np.mean(raw_subset))
                dev_mean = float(np.mean(raw_subset - raw_mean))
                tract_mean_diag_rows.append({
                    'tract_fips': fips, 'architecture': arch,
                    'raw_model_output_mean': round(raw_mean, 8),
                    'deviation_mean': round(dev_mean, 8),
                    'final_prediction_mean': round(float(np.mean(preds_arr)), 8),
                    'target_svi': round(tract_svi, 8),
                })

            # feature importance
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
                print(f'[ablation04]   WARNING: feature importance failed for {fips}/{arch}: {e}')

            print(
                f'[ablation04]   constr_err={constr_err:.2e} '
                f'sp_std={sp_std:.4f} morans_i={morans_i:.3f} '
                f'bg_r={bm["bg_r"]:.3f} t={runtime:.1f}s'
            )

    # -----------------------------------------------------------------------
    # write per_tract_metrics.csv
    # -----------------------------------------------------------------------
    per_tract_df = pd.DataFrame(per_tract_rows)
    per_tract_path = results_dir / 'per_tract_metrics.csv'
    per_tract_df.to_csv(per_tract_path, index=False)
    print(f'\n[ablation04] per_tract_metrics.csv written ({len(per_tract_df)} rows)')

    # -----------------------------------------------------------------------
    # aggregate metrics
    # -----------------------------------------------------------------------
    agg_metrics = {}
    for arch in ARCHITECTURES:
        arch_df = per_tract_df[
            (per_tract_df['architecture'] == arch) & (per_tract_df['failure'] == '')
        ]
        if len(arch_df) == 0:
            agg_metrics[arch] = {}
            continue
        agg_metrics[arch] = {
            'n_tracts': int(len(arch_df)),
            'constraint_error_mean': float(arch_df['constraint_error'].mean()),
            'constraint_error_median': float(arch_df['constraint_error'].median()),
            'pre_correction_constraint_error_mean': float(arch_df['pre_correction_constraint_error'].dropna().mean()) if arch_df['pre_correction_constraint_error'].notna().any() else float('nan'),
            'spatial_std_mean': float(arch_df['spatial_std'].mean()),
            'spatial_std_median': float(arch_df['spatial_std'].median()),
            'morans_i_mean': float(arch_df['morans_i'].dropna().mean()) if arch_df['morans_i'].notna().any() else float('nan'),
            'morans_i_median': float(arch_df['morans_i'].dropna().median()) if arch_df['morans_i'].notna().any() else float('nan'),
            'bg_r_mean': float(arch_df['bg_r'].dropna().mean()) if arch_df['bg_r'].notna().any() else float('nan'),
            'bg_r_median': float(arch_df['bg_r'].dropna().median()) if arch_df['bg_r'].notna().any() else float('nan'),
        }

    with open(results_dir / 'aggregate_metrics.json', 'w') as f:
        json.dump(agg_metrics, f, indent=2,
                  default=lambda x: None if (isinstance(x, float) and math.isnan(x)) else x)
    print('[ablation04] aggregate_metrics.json written')

    # -----------------------------------------------------------------------
    # pooled BG validation
    # -----------------------------------------------------------------------
    print('[ablation04] computing pooled BG validation...')
    bg_validation = {}
    per_tract_bg = {}

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
            svi_lookup = bg_gdf[['GEOID', 'SVI']].dropna(subset=['SVI'])
            bg_agg_all = _aggregate_to_bg(validator, combined_gdf, all_p, 1)
            merged_scatter = bg_agg_all.merge(svi_lookup, on='GEOID', how='inner')
        except Exception as e:
            bm_pooled = {'bg_r': float('nan'), 'bg_rmse': float('nan'), 'n_bgs': 0}
            merged_scatter = pd.DataFrame()
            print(f'[ablation04] WARNING: pooled BG validation failed for {arch}: {e}')
        bg_validation[arch] = {
            'pearson_r': bm_pooled['bg_r'],
            'bg_rmse': bm_pooled['bg_rmse'],
            'n_bgs': bm_pooled['n_bgs'],
        }
        per_tract_bg[arch] = merged_scatter

    # IDW / kriging pooled
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
            svi_lookup = bg_gdf[['GEOID', 'SVI']].dropna(subset=['SVI'])
            bg_agg_all = _aggregate_to_bg(validator, combined_gdf, all_p, 1)
            merged_scatter = bg_agg_all.merge(svi_lookup, on='GEOID', how='inner')
        except Exception as e:
            bm_pooled = {'bg_r': float('nan'), 'bg_rmse': float('nan'), 'n_bgs': 0}
            merged_scatter = pd.DataFrame()
            print(f'[ablation04] WARNING: pooled BG validation failed for {label}: {e}')
        bg_validation[label] = {
            'pearson_r': bm_pooled['bg_r'],
            'bg_rmse': bm_pooled['bg_rmse'],
            'n_bgs': bm_pooled['n_bgs'],
        }
        per_tract_bg[label] = merged_scatter

    with open(results_dir / 'block_group_validation.json', 'w') as f:
        json.dump(bg_validation, f, indent=2,
                  default=lambda x: None if (isinstance(x, float) and math.isnan(x)) else x)
    print('[ablation04] block_group_validation.json written')

    # stop condition for baseline_for_step4: bg_r within 1% of reference
    if subdir == '00_baseline_for_step4':
        for arch in ARCHITECTURES:
            ref_r = STEP4_REFERENCE.get(arch)
            actual_r = bg_validation.get(arch, {}).get('pearson_r', float('nan'))
            if ref_r is not None and math.isfinite(actual_r) and math.isfinite(ref_r):
                diff = abs(actual_r - ref_r)
                if diff > STEP4_TOLERANCE:
                    print(
                        f'[ablation04] STOP: {arch} pooled BG r={actual_r:.4f} deviates from '
                        f'step2a reference {ref_r:.4f} by {diff:.4f} > {STEP4_TOLERANCE}'
                    )
                    sys.exit(3)
                else:
                    print(f'[ablation04] bg_r check: {arch} r={actual_r:.4f} ref={ref_r:.4f} diff={diff:.4f} OK')

    # -----------------------------------------------------------------------
    # feature importance
    # -----------------------------------------------------------------------
    fi_dir = results_dir / 'feature_importance'
    fi_dir.mkdir(exist_ok=True)
    importance_agg = {}
    for arch, dfs in importance_by_arch.items():
        if not dfs:
            importance_agg[arch] = pd.DataFrame()
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
        importance_agg[arch] = grouped
        # write with arch-specific name
        arch_label_clean = arch.replace('_', '')
        grouped.to_csv(fi_dir / f'{arch_label_clean}_importance.csv', index=False)
        print(f'[ablation04] feature_importance/{arch_label_clean}_importance.csv written')

    # also write with canonical names expected by summary
    for arch, df in importance_agg.items():
        if len(df) > 0:
            if arch == 'sage':
                df.to_csv(fi_dir / 'sage_importance.csv', index=False)
            else:
                df.to_csv(fi_dir / 'gcngat_importance.csv', index=False)

    # -----------------------------------------------------------------------
    # tract_mean_diagnostic for cbc_no_shift
    # -----------------------------------------------------------------------
    if mode == 'cbc_no_shift' and tract_mean_diag_rows:
        diag_df = pd.DataFrame(tract_mean_diag_rows)
        diag_df.to_csv(results_dir / 'tract_mean_diagnostic.csv', index=False)
        print('[ablation04] tract_mean_diagnostic.csv written')

    # -----------------------------------------------------------------------
    # figures
    # -----------------------------------------------------------------------
    print('[ablation04] generating figures...')
    _generate_figures(per_tract_df, bg_validation, per_tract_bg, importance_agg, figures_dir)

    ts_end = datetime.now()
    elapsed = (ts_end - ts_start).seconds
    print(f'\n[ablation04] complete: {ts_end.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'[ablation04] elapsed: {elapsed // 60}m {elapsed % 60}s')

    # return summary dict for --write-summary
    return {
        'mode': mode,
        'subdir': subdir,
        'agg_metrics': agg_metrics,
        'bg_validation': bg_validation,
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def _generate_figures(per_tract_df, bg_validation, per_tract_bg, importance_agg, figures_dir):
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

    for fn, fname, args in [
        (plot_ablation_constraint_error_dist, 'constraint_error_dist.png', (success_df,)),
        (plot_ablation_spatial_std_by_svi, 'spatial_std_by_svi.png', (success_df,)),
        (plot_ablation_morans_i_by_tract, 'morans_i_by_tract.png', (success_df,)),
    ]:
        try:
            fn(*args, str(figures_dir / fname))
            print(f'[ablation04] figures/{fname} written')
        except Exception as e:
            print(f'[ablation04] WARNING: {fname} failed: {e}')

    try:
        plot_ablation_block_group_scatter(
            per_tract_bg, bg_validation, str(figures_dir / 'block_group_scatter.png')
        )
        print('[ablation04] figures/block_group_scatter.png written')
    except Exception as e:
        print(f'[ablation04] WARNING: block_group_scatter.png failed: {e}')

    try:
        sage_imp = importance_agg.get('sage', pd.DataFrame())
        gcngat_imp = importance_agg.get('gcn_gat', pd.DataFrame())
        assert isinstance(sage_imp, pd.DataFrame), 'sage_imp must be DataFrame'
        assert isinstance(gcngat_imp, pd.DataFrame), 'gcngat_imp must be DataFrame'
        plot_ablation_feature_importance_top20(
            sage_imp, gcngat_imp, str(figures_dir / 'feature_importance_top20.png')
        )
        print('[ablation04] figures/feature_importance_top20.png written')
    except Exception as e:
        print(f'[ablation04] WARNING: feature_importance_top20.png failed: {e}')

    try:
        sage_imp = importance_agg.get('sage', pd.DataFrame())
        gcngat_imp = importance_agg.get('gcn_gat', pd.DataFrame())
        assert isinstance(sage_imp, pd.DataFrame), 'sage_imp must be DataFrame'
        assert isinstance(gcngat_imp, pd.DataFrame), 'gcngat_imp must be DataFrame'
        plot_ablation_architecture_overlap(
            sage_imp, gcngat_imp, str(figures_dir / 'architecture_overlap.png')
        )
        print('[ablation04] figures/architecture_overlap.png written')
    except Exception as e:
        print(f'[ablation04] WARNING: architecture_overlap.png failed: {e}')


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def _load_results(subdir):
    """load aggregate_metrics.json and block_group_validation.json from subdir."""
    results_dir = ABLATION_DIR / subdir / 'results'
    agg = {}
    bg = {}
    agg_path = results_dir / 'aggregate_metrics.json'
    bg_path = results_dir / 'block_group_validation.json'
    if agg_path.exists():
        with open(agg_path) as f:
            agg = json.load(f)
    if bg_path.exists():
        with open(bg_path) as f:
            bg = json.load(f)
    return agg, bg


def write_summary():
    summary_dir = ABLATION_DIR / 'summary'
    summary_dir.mkdir(exist_ok=True)

    subdirs = {
        'soft': '00_baseline_for_step4',
        'cbc_with_shift': '01_cbc_no_constraint_loss',
        'cbc_no_shift': '02_cbc_no_shift',
    }

    results = {}
    for mode, subdir in subdirs.items():
        agg, bg = _load_results(subdir)
        results[mode] = {'agg': agg, 'bg': bg}

    # delta_vs_soft: compare cbc modes against soft
    soft_agg = results.get('soft', {}).get('agg', {})
    soft_bg = results.get('soft', {}).get('bg', {})

    delta = {}
    for mode in ('cbc_with_shift', 'cbc_no_shift'):
        mode_agg = results.get(mode, {}).get('agg', {})
        mode_bg = results.get(mode, {}).get('bg', {})
        delta[mode] = {}
        for arch in ARCHITECTURES:
            soft_r = soft_bg.get(arch, {}).get('pearson_r')
            mode_r = mode_bg.get(arch, {}).get('pearson_r')
            soft_std = soft_agg.get(arch, {}).get('spatial_std_mean')
            mode_std = mode_agg.get(arch, {}).get('spatial_std_mean')
            soft_mi = soft_agg.get(arch, {}).get('morans_i_mean')
            mode_mi = mode_agg.get(arch, {}).get('morans_i_mean')
            delta[mode][arch] = {
                'bg_r_soft': soft_r,
                'bg_r_mode': mode_r,
                'bg_r_delta': (mode_r - soft_r) if (mode_r is not None and soft_r is not None) else None,
                'spatial_std_soft': soft_std,
                'spatial_std_mode': mode_std,
                'spatial_std_delta': (mode_std - soft_std) if (mode_std is not None and soft_std is not None) else None,
                'morans_i_soft': soft_mi,
                'morans_i_mode': mode_mi,
                'morans_i_delta': (mode_mi - soft_mi) if (mode_mi is not None and soft_mi is not None) else None,
            }

    with open(summary_dir / 'delta_vs_soft.json', 'w') as f:
        json.dump(delta, f, indent=2,
                  default=lambda x: None if (isinstance(x, float) and math.isnan(x)) else x)
    print('[ablation04] summary/delta_vs_soft.json written')

    # summary figures
    _write_summary_figures(results, summary_dir, subdirs)

    # README.md
    _write_summary_readme(results, delta, summary_dir)
    print('[ablation04] summary/README.md written')


def _write_summary_figures(results, summary_dir, subdirs):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('[ablation04] WARNING: matplotlib not available; skipping summary figures')
        return

    modes = ['soft', 'cbc_with_shift', 'cbc_no_shift']
    mode_labels = {'soft': 'soft', 'cbc_with_shift': 'cbc+shift', 'cbc_no_shift': 'cbc-shift'}

    # figure 1: cbc_sweep.png -- 3 rows x 2 cols
    # rows: within-tract std, moran's i, bg r; cols: sage, gcn_gat
    try:
        fig, axes = plt.subplots(3, 2, figsize=(10, 9))
        metrics = ['spatial_std_mean', 'morans_i_mean', 'bg_r_mean']
        metric_labels = ['within-tract std (mean)', "moran's i (mean)", 'BG r (mean)']
        for col_idx, arch in enumerate(ARCHITECTURES):
            for row_idx, (metric, ylabel) in enumerate(zip(metrics, metric_labels)):
                ax = axes[row_idx, col_idx]
                vals = []
                xlabels = []
                for mode in modes:
                    agg = results.get(mode, {}).get('agg', {})
                    if metric == 'bg_r_mean':
                        v = results.get(mode, {}).get('bg', {}).get(arch, {}).get('pearson_r')
                    else:
                        v = agg.get(arch, {}).get(metric)
                    vals.append(v if v is not None else float('nan'))
                    xlabels.append(mode_labels[mode])
                ax.bar(xlabels, vals, color=['steelblue', 'darkorange', 'forestgreen'])
                ax.set_ylabel(ylabel, fontsize=9)
                ax.set_title(ARCH_LABELS[arch], fontsize=10)
                ax.set_ylim(bottom=0)
        fig.suptitle('ablation step 4: constraint mode sweep', fontsize=12)
        plt.tight_layout()
        fig.savefig(str(summary_dir / 'cbc_sweep.png'), dpi=150)
        plt.close(fig)
        print('[ablation04] summary/cbc_sweep.png written')
    except Exception as e:
        print(f'[ablation04] WARNING: cbc_sweep.png failed: {e}')

    # figure 2: pre_correction_error.png
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for col_idx, arch in enumerate(ARCHITECTURES):
            ax = axes[col_idx]
            for mode_i, mode in enumerate(modes):
                subdir = subdirs[mode]
                csv_path = ABLATION_DIR / subdir / 'results' / 'per_tract_metrics.csv'
                if not csv_path.exists():
                    continue
                df = pd.read_csv(csv_path)
                arch_df = df[(df['architecture'] == arch) & (df['failure'].fillna('') == '')]
                if 'pre_correction_constraint_error' not in arch_df.columns:
                    continue
                x = np.arange(len(arch_df))
                width = 0.25
                ax.bar(x + mode_i * width, arch_df['pre_correction_constraint_error'].values,
                       width=width, label=mode_labels[mode])
            ax.set_xlabel('tract index')
            ax.set_ylabel('pre-correction constraint error')
            ax.set_title(f'{ARCH_LABELS[arch]}: pre-correction constraint error per tract')
            ax.legend(fontsize=8)
        plt.tight_layout()
        fig.savefig(str(summary_dir / 'pre_correction_error.png'), dpi=150)
        plt.close(fig)
        print('[ablation04] summary/pre_correction_error.png written')
    except Exception as e:
        print(f'[ablation04] WARNING: pre_correction_error.png failed: {e}')

    # figure 3: extreme_tract_followup.png
    try:
        # find tracts with lowest/highest SVI from soft baseline
        soft_subdir = subdirs['soft']
        soft_csv = ABLATION_DIR / soft_subdir / 'results' / 'per_tract_metrics.csv'
        if soft_csv.exists():
            soft_df = pd.read_csv(soft_csv)
            sage_soft = soft_df[(soft_df['architecture'] == 'sage') & (soft_df['failure'].fillna('') == '')]
            if len(sage_soft) >= 2:
                lowest_fips = sage_soft.loc[sage_soft['tract_svi'].idxmin(), 'fips']
                highest_fips = sage_soft.loc[sage_soft['tract_svi'].idxmax(), 'fips']
                extreme_fips = [lowest_fips, highest_fips]
                extreme_labels = [f'{lowest_fips} (low SVI)', f'{highest_fips} (high SVI)']

                fig, axes = plt.subplots(2, 2, figsize=(10, 8))
                for row_i, (fips_val, fips_label) in enumerate(zip(extreme_fips, extreme_labels)):
                    for col_i, metric_col in enumerate(['morans_i', 'spatial_std']):
                        ax = axes[row_i, col_i]
                        for mode in modes:
                            subdir_m = subdirs[mode]
                            csv_path = ABLATION_DIR / subdir_m / 'results' / 'per_tract_metrics.csv'
                            if not csv_path.exists():
                                continue
                            df_m = pd.read_csv(csv_path)
                            row = df_m[
                                (df_m['fips'].astype(str) == str(fips_val)) &
                                (df_m['architecture'] == 'sage')
                            ]
                            if len(row) == 0:
                                continue
                            val = row[metric_col].values[0]
                            ax.bar([mode_labels[mode]], [val if math.isfinite(float(val)) else 0])
                        ax.set_title(f'{fips_label}\n{metric_col}', fontsize=9)
                        ax.set_ylabel(metric_col)
                plt.suptitle('extreme tract followup (SAGE)', fontsize=11)
                plt.tight_layout()
                fig.savefig(str(summary_dir / 'extreme_tract_followup.png'), dpi=150)
                plt.close(fig)
                print('[ablation04] summary/extreme_tract_followup.png written')
    except Exception as e:
        print(f'[ablation04] WARNING: extreme_tract_followup.png failed: {e}')


def _write_summary_readme(results, delta, summary_dir):
    def _fmt(v):
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            return 'n/a'
        return f'{v:.4f}'

    lines = [
        '# ablation step 4 summary: constraint-by-construction',
        '',
        'Three constraint modes compared:',
        '- `soft`: standard soft constraint loss + post-hoc additive shift (baseline)',
        '- `cbc_with_shift`: cbc in training loop; post-hoc shift still applied',
        '- `cbc_no_shift`: cbc in training loop; post-hoc shift skipped',
        '',
        '## pooled BG r by mode and architecture',
        '',
        '| mode | SAGE bg_r | GCN-GAT bg_r |',
        '|---|---|---|',
    ]
    for mode in ('soft', 'cbc_with_shift', 'cbc_no_shift'):
        sage_r = _fmt(results.get(mode, {}).get('bg', {}).get('sage', {}).get('pearson_r'))
        gcn_r = _fmt(results.get(mode, {}).get('bg', {}).get('gcn_gat', {}).get('pearson_r'))
        lines.append(f'| {mode} | {sage_r} | {gcn_r} |')

    lines += [
        '',
        '## delta vs soft (cbc_with_shift)',
        '',
        '| arch | bg_r delta | spatial_std delta | morans_i delta |',
        '|---|---|---|---|',
    ]
    for arch in ARCHITECTURES:
        d = delta.get('cbc_with_shift', {}).get(arch, {})
        lines.append(f'| {arch} | {_fmt(d.get("bg_r_delta"))} | {_fmt(d.get("spatial_std_delta"))} | {_fmt(d.get("morans_i_delta"))} |')

    lines += [
        '',
        '## delta vs soft (cbc_no_shift)',
        '',
        '| arch | bg_r delta | spatial_std delta | morans_i delta |',
        '|---|---|---|---|',
    ]
    for arch in ARCHITECTURES:
        d = delta.get('cbc_no_shift', {}).get(arch, {})
        lines.append(f'| {arch} | {_fmt(d.get("bg_r_delta"))} | {_fmt(d.get("spatial_std_delta"))} | {_fmt(d.get("morans_i_delta"))} |')

    lines += [
        '',
        '## figures',
        '',
        '- `cbc_sweep.png`: within-tract std, moran\'s i, BG r across modes for both architectures',
        '- `pre_correction_error.png`: pre-correction constraint error per tract and mode',
        '- `extreme_tract_followup.png`: moran\'s i and spatial_std for lowest/highest SVI tracts',
        '',
    ]

    (summary_dir / 'README.md').write_text('\n'.join(lines) + '\n')


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='ablation step 4: constraint-by-construction')
    parser.add_argument('--mode', choices=['soft', 'cbc_with_shift', 'cbc_no_shift'],
                        help='constraint mode to run')
    parser.add_argument('--subdir', help='subdirectory name under experiments/ablation/04_...')
    parser.add_argument('--write-summary', action='store_true',
                        help='read all three result sets and write summary/')
    args = parser.parse_args()

    if args.write_summary:
        write_summary()
        return

    if not args.mode or not args.subdir:
        parser.print_help()
        sys.exit(1)

    run_mode(args.mode, args.subdir)


if __name__ == '__main__':
    main()
