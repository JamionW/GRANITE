"""
04b_variation_weight_recalibration: sweep variation_weight under cbc_no_shift.

Determines whether raising variation_loss weight recovers SAGE within-tract spread
that collapsed in step 4 (0.082 soft vs 0.060 cbc). Both architectures are run at
each weight value.

usage:
    python experiments/ablation/04b_variation_weight_recalibration/run_ablation_04b.py --weight 0.8 --subdir 00_w_0p8
    python experiments/ablation/04b_variation_weight_recalibration/run_ablation_04b.py --weight 1.5 --subdir 01_w_1p5
    python experiments/ablation/04b_variation_weight_recalibration/run_ablation_04b.py --weight 2.5 --subdir 02_w_2p5
    python experiments/ablation/04b_variation_weight_recalibration/run_ablation_04b.py --weight 4.0 --subdir 03_w_4p0
    python experiments/ablation/04b_variation_weight_recalibration/run_ablation_04b.py --write-summary
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
STEP4_DIR = REPO_ROOT / 'experiments' / 'ablation' / '04_constraint_by_construction'
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

WEIGHTS = [0.8, 1.5, 2.5, 4.0]
SUBDIRS = ['00_w_0p8', '01_w_1p5', '02_w_2p5', '03_w_4p0']

# reference values from step 4 02_cbc_no_shift (variation_weight=0.8)
# sanity check: 00_w_0p8 must reproduce these within tolerance
STEP4B_SANITY_REFERENCE = {
    'sage': {
        'spatial_std_mean': 0.059486,
        'morans_i_mean': 0.866872,
        'bg_r': 0.751064,
    },
    'gcn_gat': {
        'spatial_std_mean': 0.082960,
        'morans_i_mean': 0.841977,
        'bg_r': 0.748065,
    },
}
SANITY_TOLERANCE = 1e-4  # aggregate metrics must match within this tolerance

# soft mode reference values from step 4 00_baseline_for_step4 for summary figures
SOFT_REFERENCE = {
    'sage': {
        'spatial_std_mean': 0.082321,
        'morans_i_mean': 0.877590,
        'bg_r': 0.753697,
    },
    'gcn_gat': {
        'spatial_std_mean': 0.081425,
        'morans_i_mean': 0.849045,
        'bg_r': 0.766431,
    },
}

# stop conditions
STOP_BG_R_FLOOR = 0.65
STOP_STD_CEILING = 0.25
STOP_CONSTRAINT_ERROR_CEILING = 1e-6


# ---------------------------------------------------------------------------
# helpers
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
    cfg.get('training', {}).pop('smoothness_weight', None)
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
        print(f'[ablation04b] STOP: non-finite {metric_name}={value} on tract {fips} / arch {arch}')
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

    with open(subdir_path / 'config_snapshot.yaml', 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)

    try:
        env_txt = subprocess.check_output(['pip', 'freeze']).decode()
    except Exception:
        env_txt = 'pip freeze failed\n'
    (subdir_path / 'environment.txt').write_text(env_txt)

    import shutil
    shutil.copy(str(TRACT_SELECTION), str(subdir_path / 'tract_selection.txt'))
    print(f'[ablation04b] preflight files written to {subdir_path}')


# ---------------------------------------------------------------------------
# main run
# ---------------------------------------------------------------------------

def run_weight(variation_weight, subdir):
    ts_start = datetime.now()
    print(f'[ablation04b] start: variation_weight={variation_weight} subdir={subdir} '
          f'{ts_start.strftime("%Y-%m-%d %H:%M:%S")}')
    set_random_seed(SEED)

    subdir_path = ABLATION_DIR / subdir
    results_dir = subdir_path / 'results'
    figures_dir = subdir_path / 'figures'
    (results_dir / 'feature_importance').mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not BG_SVI_PATH.exists():
        print(f'[ablation04b] HALT: {BG_SVI_PATH} missing')
        sys.exit(1)

    tract_list = _load_tract_list()
    print(f'[ablation04b] tracts: {len(tract_list)}')

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
    cfg['training']['constraint_mode'] = 'cbc_no_shift'
    cfg['training']['apply_post_correction'] = False
    cfg['training']['variation_weight'] = variation_weight

    _write_preflight(subdir_path, cfg, tract_list)

    print('[ablation04b] loading BG geodataframe...')
    bg_gdf = _load_bg_gdf()
    validator = BlockGroupValidator(bg_gdf, verbose=False)

    scratch_dir = str(REPO_ROOT / 'output' / f'ablation04b_{subdir}_scratch')
    os.makedirs(scratch_dir, exist_ok=True)
    pipeline_cfg = dict(cfg)
    pipeline_cfg['data'] = dict(cfg['data'])
    pipeline_cfg['data']['target_fips'] = tract_list[0]
    pipeline = GRANITEPipeline(pipeline_cfg, output_dir=scratch_dir)
    pipeline.verbose = False

    print('[ablation04b] loading spatial data...')
    try:
        data = pipeline._load_spatial_data()
    except Exception as e:
        print(f'[ablation04b] HALT: spatial data load failed: {e}')
        sys.exit(1)

    print('[ablation04b] fitting IDW and kriging baselines...')
    try:
        idw_model, krig_model = _load_idw_kriging(data['tracts'])
    except Exception as e:
        print(f'[ablation04b] WARNING: IDW/kriging load failed: {e}')
        idw_model, krig_model = None, None

    per_tract_rows = []
    pooled_addr_gdfs = {a: [] for a in ARCHITECTURES}
    pooled_preds = {a: [] for a in ARCHITECTURES}
    pooled_preds_idw = []
    pooled_preds_krig = []
    pooled_addr_gdfs_baselines = []
    importance_by_arch = {a: [] for a in ARCHITECTURES}
    # per-arch variation activation rate accumulator
    variation_activation_rates = {a: [] for a in ARCHITECTURES}

    total = len(ARCHITECTURES) * len(tract_list)
    done = 0

    for arch in ARCHITECTURES:
        arch_label = ARCH_LABELS[arch]
        print(f'\n[ablation04b] === {arch_label} ===')
        cfg['model']['architecture'] = arch
        pipeline.config = dict(cfg)
        pipeline.verbose = False

        for idx, fips in enumerate(tract_list):
            done += 1
            print(f'[ablation04b] [{done}/{total}] {arch_label} | tract {idx+1}/{len(tract_list)}: {fips}')
            t0 = time.time()

            cfg['data']['target_fips'] = fips
            pipeline.config['data']['target_fips'] = fips
            pipeline.config['model']['architecture'] = arch

            try:
                result = pipeline._process_single_tract(fips, data)
            except Exception as e:
                msg = str(e)[:300]
                print(f'[ablation04b]   ERROR: {msg}')
                traceback.print_exc()
                per_tract_rows.append({
                    'fips': fips, 'architecture': arch, 'n_addresses': 0,
                    'tract_svi': float('nan'), 'constraint_error': float('nan'),
                    'spatial_std': float('nan'), 'morans_i': float('nan'),
                    'bg_r': float('nan'), 'n_bgs': 0,
                    'variation_activation_rate': float('nan'),
                    'runtime_s': 0, 'failure': msg,
                })
                continue

            if not result.get('success'):
                msg = result.get('error', 'unknown')[:300]
                print(f'[ablation04b]   FAILED: {msg}')
                per_tract_rows.append({
                    'fips': fips, 'architecture': arch, 'n_addresses': 0,
                    'tract_svi': float('nan'), 'constraint_error': float('nan'),
                    'spatial_std': float('nan'), 'morans_i': float('nan'),
                    'bg_r': float('nan'), 'n_bgs': 0,
                    'variation_activation_rate': float('nan'),
                    'runtime_s': 0, 'failure': msg,
                })
                continue

            runtime = time.time() - t0
            address_gdf = result['address_gdf']
            preds_arr = result['predictions']['mean'].values.astype(float)
            tract_svi = float(result['tract_svi'])
            n_addresses = len(address_gdf)

            constr_err = abs(np.mean(preds_arr) - tract_svi)
            sp_std = float(np.std(preds_arr))
            morans_i = _compute_morans_i(preds_arr, address_gdf)

            # variation_loss_activation_rate from training_result
            training_result = result.get('training_result', {})
            var_act_rate = training_result.get('variation_loss_activation_rate', float('nan'))
            if math.isfinite(var_act_rate):
                variation_activation_rates[arch].append(var_act_rate)

            # stop conditions
            _check_nan_stop('constraint_error', constr_err, fips, arch)
            _check_nan_stop('spatial_std', sp_std, fips, arch)
            if not math.isfinite(morans_i):
                print(f'[ablation04b]   WARNING: non-finite morans_i for {fips}/{arch}')
                morans_i = float('nan')

            # cbc_no_shift: constraint error must be < STOP_CONSTRAINT_ERROR_CEILING
            if math.isfinite(constr_err) and constr_err >= STOP_CONSTRAINT_ERROR_CEILING:
                print(
                    f'[ablation04b] STOP: cbc_no_shift constraint error {constr_err:.2e} >= '
                    f'{STOP_CONSTRAINT_ERROR_CEILING:.0e} on tract {fips} / arch {arch}. '
                    'variation_loss is affecting by-construction mean enforcement; check cbc code path.'
                )
                sys.exit(4)

            try:
                bg_agg = _aggregate_to_bg(validator, address_gdf, preds_arr, MIN_ADDRESSES_PER_BG)
                bm = _bg_metrics(bg_agg, bg_gdf)
            except Exception as e:
                bm = {'bg_r': float('nan'), 'bg_rmse': float('nan'), 'n_bgs': 0}
                print(f'[ablation04b]   WARNING: BG metrics failed: {e}')

            # stop: pooled BG r below floor (checked per-tract as early warning)
            if math.isfinite(bm['bg_r']) and bm['bg_r'] < STOP_BG_R_FLOOR:
                print(f'[ablation04b]   WARNING: tract bg_r={bm["bg_r"]:.3f} < {STOP_BG_R_FLOOR}')

            # stop: pathological std
            if math.isfinite(sp_std) and sp_std > STOP_STD_CEILING:
                print(
                    f'[ablation04b] STOP: within-tract std={sp_std:.4f} > {STOP_STD_CEILING} '
                    f'on tract {fips} / arch {arch}. recalibration produced pathologically noisy predictions.'
                )
                sys.exit(5)

            per_tract_rows.append({
                'fips': fips, 'architecture': arch, 'n_addresses': n_addresses,
                'tract_svi': tract_svi,
                'constraint_error': round(constr_err, 8),
                'spatial_std': round(sp_std, 6),
                'morans_i': round(morans_i, 6) if math.isfinite(morans_i) else float('nan'),
                'bg_r': bm['bg_r'], 'n_bgs': bm['n_bgs'],
                'variation_activation_rate': round(var_act_rate, 4) if math.isfinite(var_act_rate) else float('nan'),
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
                        print(f'[ablation04b]   WARNING: IDW failed for {fips}: {e}')
                if krig_model is not None:
                    try:
                        addr_coords = np.array([[g.x, g.y] for g in address_gdf.geometry])
                        krig_p = krig_model.disaggregate(addr_coords, fips, tract_svi)
                        pooled_preds_krig.append(krig_p)
                    except Exception as e:
                        pooled_preds_krig.append(np.full(n_addresses, tract_svi))
                        print(f'[ablation04b]   WARNING: kriging failed for {fips}: {e}')

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
                print(f'[ablation04b]   WARNING: feature importance failed for {fips}/{arch}: {e}')

            print(
                f'[ablation04b]   constr_err={constr_err:.2e} '
                f'sp_std={sp_std:.4f} morans_i={morans_i:.3f} '
                f'bg_r={bm["bg_r"]:.3f} var_act={var_act_rate:.3f} t={runtime:.1f}s'
            )

    # -----------------------------------------------------------------------
    # per_tract_metrics.csv
    # -----------------------------------------------------------------------
    per_tract_df = pd.DataFrame(per_tract_rows)
    per_tract_path = subdir_path / 'per_tract_metrics.csv'
    per_tract_df.to_csv(per_tract_path, index=False)
    print(f'\n[ablation04b] per_tract_metrics.csv written ({len(per_tract_df)} rows)')

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
        mean_var_act = float(arch_df['variation_activation_rate'].dropna().mean()) \
            if arch_df['variation_activation_rate'].notna().any() else float('nan')
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
            'variation_loss_activation_rate': mean_var_act,
        }

    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / 'aggregate_metrics.json', 'w') as f:
        json.dump(agg_metrics, f, indent=2,
                  default=lambda x: None if (isinstance(x, float) and math.isnan(x)) else x)
    print('[ablation04b] aggregate_metrics.json written')

    # informational comparison for 00_w_0p8 against step 4 02_cbc_no_shift.
    # note: step 4 used variation_weight=1.5 (single-tract hardcoded default); 00_w_0p8 uses 0.8.
    # bit-identical reproduction is not expected. this block logs the delta for the README.
    if subdir == '00_w_0p8':
        print('[ablation04b] 00_w_0p8 comparison vs step 4 02_cbc_no_shift (informational):')
        for arch in ARCHITECTURES:
            ref = STEP4B_SANITY_REFERENCE.get(arch, {})
            got = agg_metrics.get(arch, {})
            for key in ('spatial_std_mean', 'morans_i_mean'):
                ref_val = ref.get(key)
                got_val = got.get(key)
                if ref_val is not None and got_val is not None:
                    diff = got_val - ref_val
                    print(f'  {arch}.{key}: got={got_val:.6f} ref={ref_val:.6f} delta={diff:+.6f}')
        # hard stop: constraint errors must still be machine precision (cbc_no_shift invariant)
        for arch in ARCHITECTURES:
            got = agg_metrics.get(arch, {})
            ce = got.get('constraint_error_mean', float('nan'))
            if math.isfinite(ce) and ce >= STOP_CONSTRAINT_ERROR_CEILING:
                print(
                    f'[ablation04b] STOP: {arch} constraint_error_mean={ce:.2e} >= '
                    f'{STOP_CONSTRAINT_ERROR_CEILING:.0e} in 00_w_0p8. '
                    'cbc_no_shift enforcement broken; investigate before continuing.'
                )
                sys.exit(6)

    # -----------------------------------------------------------------------
    # pooled BG validation
    # -----------------------------------------------------------------------
    print('[ablation04b] computing pooled BG validation...')
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
            print(f'[ablation04b] WARNING: pooled BG validation failed for {arch}: {e}')
        bg_validation[arch] = {
            'pearson_r': bm_pooled['bg_r'],
            'bg_rmse': bm_pooled['bg_rmse'],
            'n_bgs': bm_pooled['n_bgs'],
        }
        per_tract_bg[arch] = merged_scatter

        # pooled BG r stop condition
        pooled_r = bm_pooled['bg_r']
        if math.isfinite(pooled_r) and pooled_r < STOP_BG_R_FLOOR:
            print(
                f'[ablation04b] STOP: {arch} pooled BG r={pooled_r:.4f} < {STOP_BG_R_FLOOR}. '
                'recalibration broke generalization.'
            )
            sys.exit(7)

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
        bg_validation[label] = {
            'pearson_r': bm_pooled['bg_r'],
            'bg_rmse': bm_pooled['bg_rmse'],
            'n_bgs': bm_pooled['n_bgs'],
        }
        per_tract_bg[label] = merged_scatter

    with open(results_dir / 'block_group_validation.json', 'w') as f:
        json.dump(bg_validation, f, indent=2,
                  default=lambda x: None if (isinstance(x, float) and math.isnan(x)) else x)
    print('[ablation04b] block_group_validation.json written')

    if subdir == '00_w_0p8':
        print('[ablation04b] 00_w_0p8 pooled BG r vs step 4 02_cbc_no_shift (informational):')
        for arch in ARCHITECTURES:
            ref_r = STEP4B_SANITY_REFERENCE.get(arch, {}).get('bg_r')
            got_r = bg_validation.get(arch, {}).get('pearson_r', float('nan'))
            if ref_r is not None and math.isfinite(got_r):
                diff = got_r - ref_r
                print(f'  {arch} bg_r: got={got_r:.6f} ref={ref_r:.6f} delta={diff:+.6f}')
        print('[ablation04b] 00_w_0p8 informational check complete '
              '(delta expected: step 4 used variation_weight=1.5, this run uses 0.8)')

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
        arch_label_clean = arch.replace('_', '')
        grouped.to_csv(fi_dir / f'{arch_label_clean}_importance.csv', index=False)
    for arch, df in importance_agg.items():
        if len(df) > 0:
            fname = 'sage_importance.csv' if arch == 'sage' else 'gcngat_importance.csv'
            df.to_csv(fi_dir / fname, index=False)

    # -----------------------------------------------------------------------
    # figures
    # -----------------------------------------------------------------------
    print('[ablation04b] generating per-variant figures...')
    _generate_variant_figures(per_tract_df, bg_validation, per_tract_bg, importance_agg, figures_dir)

    ts_end = datetime.now()
    elapsed = (ts_end - ts_start).seconds
    print(f'\n[ablation04b] complete: {ts_end.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'[ablation04b] elapsed: {elapsed // 60}m {elapsed % 60}s')

    return {
        'variation_weight': variation_weight,
        'subdir': subdir,
        'agg_metrics': agg_metrics,
        'bg_validation': bg_validation,
    }


# ---------------------------------------------------------------------------
# per-variant figures (same six as step 4)
# ---------------------------------------------------------------------------

def _generate_variant_figures(per_tract_df, bg_validation, per_tract_bg, importance_agg, figures_dir):
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
        except Exception as e:
            print(f'[ablation04b] WARNING: {fname} failed: {e}')

    try:
        plot_ablation_block_group_scatter(
            per_tract_bg, bg_validation, str(figures_dir / 'block_group_scatter.png')
        )
    except Exception as e:
        print(f'[ablation04b] WARNING: block_group_scatter.png failed: {e}')

    try:
        sage_imp = importance_agg.get('sage', pd.DataFrame())
        gcngat_imp = importance_agg.get('gcn_gat', pd.DataFrame())
        plot_ablation_feature_importance_top20(
            sage_imp, gcngat_imp, str(figures_dir / 'feature_importance_top20.png')
        )
    except Exception as e:
        print(f'[ablation04b] WARNING: feature_importance_top20.png failed: {e}')

    try:
        sage_imp = importance_agg.get('sage', pd.DataFrame())
        gcngat_imp = importance_agg.get('gcn_gat', pd.DataFrame())
        plot_ablation_architecture_overlap(
            sage_imp, gcngat_imp, str(figures_dir / 'architecture_overlap.png')
        )
    except Exception as e:
        print(f'[ablation04b] WARNING: architecture_overlap.png failed: {e}')


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def _load_results(subdir):
    results_dir = ABLATION_DIR / subdir / 'results'
    agg = {}
    bg = {}
    pt_df = pd.DataFrame()
    if (results_dir / 'aggregate_metrics.json').exists():
        with open(results_dir / 'aggregate_metrics.json') as f:
            agg = json.load(f)
    if (results_dir / 'block_group_validation.json').exists():
        with open(results_dir / 'block_group_validation.json') as f:
            bg = json.load(f)
    pt_path = ABLATION_DIR / subdir / 'per_tract_metrics.csv'
    if pt_path.exists():
        pt_df = pd.read_csv(pt_path)
    return agg, bg, pt_df


def _load_step4_soft():
    """load soft mode reference from step 4 for comparison lines in figures."""
    step4_soft_dir = STEP4_DIR / '00_baseline_for_step4' / 'results'
    agg, bg = {}, {}
    if (step4_soft_dir / 'aggregate_metrics.json').exists():
        with open(step4_soft_dir / 'aggregate_metrics.json') as f:
            agg = json.load(f)
    if (step4_soft_dir / 'block_group_validation.json').exists():
        with open(step4_soft_dir / 'block_group_validation.json') as f:
            bg = json.load(f)
    return agg, bg


def write_summary():
    summary_dir = ABLATION_DIR / 'summary'
    summary_dir.mkdir(exist_ok=True)

    results = {}
    for weight, subdir in zip(WEIGHTS, SUBDIRS):
        agg, bg, pt_df = _load_results(subdir)
        results[weight] = {'agg': agg, 'bg': bg, 'pt_df': pt_df, 'subdir': subdir}

    soft_agg, soft_bg = _load_step4_soft()

    # cbc baseline = 00_w_0p8 (same as step 4 02_cbc_no_shift)
    cbc_base_agg = results.get(0.8, {}).get('agg', {})
    cbc_base_bg = results.get(0.8, {}).get('bg', {})
    soft_sage_r = (soft_bg.get('sage') or {}).get('pearson_r')
    soft_gcn_r = (soft_bg.get('gcn_gat') or {}).get('pearson_r')

    # delta_vs_cbc_baseline.json
    delta = {}
    for weight, subdir in zip(WEIGHTS, SUBDIRS):
        res = results[weight]
        delta[str(weight)] = {}
        for arch in ARCHITECTURES:
            m = res['agg'].get(arch, {})
            cb = cbc_base_agg.get(arch, {})
            sm_agg = soft_agg.get(arch, {})
            bg_r = (res['bg'].get(arch) or {}).get('pearson_r')
            cb_bg_r = (cbc_base_bg.get(arch) or {}).get('pearson_r')
            soft_bg_r = (soft_bg.get(arch) or {}).get('pearson_r')

            def _d(a, b):
                if a is None or b is None:
                    return None
                try:
                    return a - b
                except TypeError:
                    return None

            delta[str(weight)][arch] = {
                'mean_within_tract_std': m.get('spatial_std_mean'),
                'delta_vs_cbc_baseline_std': _d(m.get('spatial_std_mean'), cb.get('spatial_std_mean')),
                'delta_vs_soft_std': _d(m.get('spatial_std_mean'), sm_agg.get('spatial_std_mean')),
                'mean_morans_i': m.get('morans_i_mean'),
                'delta_vs_cbc_baseline_morans_i': _d(m.get('morans_i_mean'), cb.get('morans_i_mean')),
                'delta_vs_soft_morans_i': _d(m.get('morans_i_mean'), sm_agg.get('morans_i_mean')),
                'pooled_block_group_r': bg_r,
                'delta_vs_cbc_baseline_bg_r': _d(bg_r, cb_bg_r),
                'delta_vs_soft_bg_r': _d(bg_r, soft_bg_r),
                'variation_loss_activation_rate': m.get('variation_loss_activation_rate'),
            }

    with open(summary_dir / 'delta_vs_cbc_baseline.json', 'w') as f:
        json.dump(delta, f, indent=2,
                  default=lambda x: None if (isinstance(x, float) and math.isnan(x)) else x)
    print('[ablation04b] summary/delta_vs_cbc_baseline.json written')

    # figures
    _write_summary_figures(results, soft_agg, soft_bg, summary_dir)

    # README
    _write_summary_readme(results, delta, soft_agg, soft_bg, summary_dir)
    print('[ablation04b] summary/README.md written')


def _write_summary_figures(results, soft_agg, soft_bg, summary_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('[ablation04b] WARNING: matplotlib not available; skipping summary figures')
        return

    log_weights = np.log10(WEIGHTS)

    # figure 1: variation_weight_sweep.png -- 3x2 grid
    try:
        fig, axes = plt.subplots(3, 2, figsize=(10, 9))
        metrics = ['spatial_std_mean', 'morans_i_mean', 'bg_r']
        metric_labels = ['within-tract std (mean)', "moran's i (mean)", 'pooled BG r']
        soft_refs = {
            'spatial_std_mean': {
                'sage': (soft_agg.get('sage') or {}).get('spatial_std_mean'),
                'gcn_gat': (soft_agg.get('gcn_gat') or {}).get('spatial_std_mean'),
            },
            'morans_i_mean': {
                'sage': (soft_agg.get('sage') or {}).get('morans_i_mean'),
                'gcn_gat': (soft_agg.get('gcn_gat') or {}).get('morans_i_mean'),
            },
            'bg_r': {
                'sage': (soft_bg.get('sage') or {}).get('pearson_r'),
                'gcn_gat': (soft_bg.get('gcn_gat') or {}).get('pearson_r'),
            },
        }
        for col_idx, arch in enumerate(ARCHITECTURES):
            for row_idx, (metric, ylabel) in enumerate(zip(metrics, metric_labels)):
                ax = axes[row_idx, col_idx]
                vals = []
                for weight in WEIGHTS:
                    if metric == 'bg_r':
                        v = (results.get(weight, {}).get('bg', {}).get(arch) or {}).get('pearson_r')
                    else:
                        v = (results.get(weight, {}).get('agg', {}).get(arch) or {}).get(metric)
                    vals.append(v if v is not None else float('nan'))
                valid_mask = [math.isfinite(v) for v in vals]
                xlog = [log_weights[i] for i, ok in enumerate(valid_mask) if ok]
                yvals = [vals[i] for i, ok in enumerate(valid_mask) if ok]
                ax.plot(xlog, yvals, marker='o', color='steelblue', linewidth=1.5)
                ax.set_xticks(log_weights)
                ax.set_xticklabels([str(w) for w in WEIGHTS], fontsize=8)
                ax.set_xlabel('variation_weight (log scale)', fontsize=8)
                ax.set_ylabel(ylabel, fontsize=9)
                ax.set_title(ARCH_LABELS[arch], fontsize=10)
                ref_val = soft_refs[metric].get(arch)
                if ref_val is not None and math.isfinite(ref_val):
                    ax.axhline(ref_val, color='darkorange', linestyle='--', linewidth=1,
                               label=f'soft mode ({ref_val:.4f})')
                    ax.legend(fontsize=7)
        fig.suptitle('step 4b: variation_weight sweep under cbc_no_shift', fontsize=11)
        plt.tight_layout()
        fig.savefig(str(summary_dir / 'variation_weight_sweep.png'), dpi=150)
        plt.close(fig)
        print('[ablation04b] summary/variation_weight_sweep.png written')
    except Exception as e:
        print(f'[ablation04b] WARNING: variation_weight_sweep.png failed: {e}')

    # figure 2: spread_vs_generalization.png -- 2x1 grid with twin axes
    try:
        fig, axes = plt.subplots(2, 1, figsize=(8, 8))
        for row_idx, arch in enumerate(ARCHITECTURES):
            ax_l = axes[row_idx]
            ax_r = ax_l.twinx()
            std_vals = []
            bg_r_vals = []
            for weight in WEIGHTS:
                sv = (results.get(weight, {}).get('agg', {}).get(arch) or {}).get('spatial_std_mean')
                rv = (results.get(weight, {}).get('bg', {}).get(arch) or {}).get('pearson_r')
                std_vals.append(sv if sv is not None else float('nan'))
                bg_r_vals.append(rv if rv is not None else float('nan'))
            ax_l.plot(WEIGHTS, std_vals, marker='o', color='steelblue', linewidth=1.5, label='within-tract std')
            ax_l.set_xlabel('variation_weight', fontsize=9)
            ax_l.set_ylabel('within-tract std (mean)', fontsize=9, color='steelblue')
            ax_l.tick_params(axis='y', labelcolor='steelblue')
            ax_r.plot(WEIGHTS, bg_r_vals, marker='s', color='darkorange', linewidth=1.5, label='pooled BG r')
            ax_r.set_ylabel('pooled BG r', fontsize=9, color='darkorange')
            ax_r.tick_params(axis='y', labelcolor='darkorange')
            ax_l.set_title(ARCH_LABELS[arch], fontsize=10)
            # add soft mode reference lines
            soft_std = (soft_agg.get(arch) or {}).get('spatial_std_mean')
            soft_r_val = (soft_bg.get(arch) or {}).get('pearson_r')
            if soft_std is not None:
                ax_l.axhline(soft_std, linestyle='--', color='steelblue', alpha=0.5, linewidth=1)
            if soft_r_val is not None:
                ax_r.axhline(soft_r_val, linestyle='--', color='darkorange', alpha=0.5, linewidth=1)
            lines_l, labels_l = ax_l.get_legend_handles_labels()
            lines_r, labels_r = ax_r.get_legend_handles_labels()
            ax_l.legend(lines_l + lines_r, labels_l + labels_r, fontsize=8)
        fig.suptitle('step 4b: spread vs generalization trade-off', fontsize=11)
        plt.tight_layout()
        fig.savefig(str(summary_dir / 'spread_vs_generalization.png'), dpi=150)
        plt.close(fig)
        print('[ablation04b] summary/spread_vs_generalization.png written')
    except Exception as e:
        print(f'[ablation04b] WARNING: spread_vs_generalization.png failed: {e}')

    # figure 3: extreme_tract_recalibration.png -- 2x2 grid
    # identify tracts 1324 and 1900 (low/high SVI extreme tracts from step 4)
    try:
        step4_soft_csv = STEP4_DIR / '00_baseline_for_step4' / 'results' / 'per_tract_metrics.csv'
        extreme_fips = []
        extreme_labels = []
        if step4_soft_csv.exists():
            soft_df = pd.read_csv(step4_soft_csv)
            sage_soft = soft_df[(soft_df['architecture'] == 'sage') & (soft_df['failure'].fillna('') == '')]
            if len(sage_soft) >= 2:
                low_fips = sage_soft.loc[sage_soft['tract_svi'].idxmin(), 'fips']
                high_fips = sage_soft.loc[sage_soft['tract_svi'].idxmax(), 'fips']
                extreme_fips = [low_fips, high_fips]
                extreme_labels = [
                    f'{low_fips} (low SVI)',
                    f'{high_fips} (high SVI)',
                ]

        if extreme_fips:
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            soft_refs_extreme = {}
            if step4_soft_csv.exists():
                for ef in extreme_fips:
                    for mc in ['morans_i', 'spatial_std']:
                        row = sage_soft[sage_soft['fips'].astype(str) == str(ef)]
                        if len(row) > 0:
                            soft_refs_extreme[(ef, mc)] = float(row[mc].values[0])

            for row_i, (fips_val, fips_label) in enumerate(zip(extreme_fips, extreme_labels)):
                for col_i, metric_col in enumerate(['morans_i', 'spatial_std']):
                    ax = axes[row_i, col_i]
                    vals = []
                    for weight, subdir in zip(WEIGHTS, SUBDIRS):
                        pt_df = results.get(weight, {}).get('pt_df', pd.DataFrame())
                        if pt_df.empty:
                            vals.append(float('nan'))
                            continue
                        row = pt_df[
                            (pt_df['fips'].astype(str) == str(fips_val)) &
                            (pt_df['architecture'] == 'sage')
                        ]
                        if len(row) == 0:
                            vals.append(float('nan'))
                        else:
                            vals.append(float(row[metric_col].values[0]))
                    ax.plot(WEIGHTS, vals, marker='o', color='steelblue', linewidth=1.5)
                    ax.set_xticks(WEIGHTS)
                    ax.set_xlabel('variation_weight', fontsize=8)
                    ax.set_ylabel(metric_col, fontsize=9)
                    ax.set_title(f'{fips_label}\n{metric_col}', fontsize=9)
                    ref_val = soft_refs_extreme.get((fips_val, metric_col))
                    if ref_val is not None and math.isfinite(float(ref_val)):
                        ax.axhline(ref_val, color='darkorange', linestyle='--', linewidth=1,
                                   label=f'soft ({ref_val:.4f})')
                        ax.legend(fontsize=7)
            fig.suptitle('step 4b: extreme tract recalibration (SAGE)', fontsize=11)
            plt.tight_layout()
            fig.savefig(str(summary_dir / 'extreme_tract_recalibration.png'), dpi=150)
            plt.close(fig)
            print('[ablation04b] summary/extreme_tract_recalibration.png written')
    except Exception as e:
        print(f'[ablation04b] WARNING: extreme_tract_recalibration.png failed: {e}')


def _write_summary_readme(results, delta, soft_agg, soft_bg, summary_dir):
    try:
        sha = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=str(REPO_ROOT)
        ).decode().strip()
    except Exception:
        sha = 'unknown'

    now = datetime.now().strftime('%Y-%m-%d %H:%M')

    def _fmt(v):
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            return 'n/a'
        return f'{v:.4f}'

    soft_sage_std = (soft_agg.get('sage') or {}).get('spatial_std_mean')
    soft_sage_r = (soft_bg.get('sage') or {}).get('pearson_r')
    target_std = 0.082 * 0.9  # within 10% of soft mode 0.082

    lines = [
        '# ablation step 4b: variation_weight recalibration under cbc_no_shift',
        '',
        f'git sha: {sha}',
        f'seed: 42',
        f'generated: {now}',
        '',
        '## context',
        '',
        'Step 4 found that SAGE within-tract std collapsed under cbc_no_shift (0.060 vs 0.082 soft).',
        'This sweep tests whether raising variation_weight recovers spread.',
        '',
        '## 00_w_0p8 vs step 4 02_cbc_no_shift (informational)',
        '',
        'Step 4 02_cbc_no_shift used variation_weight=1.5 (single-tract hardcoded default).',
        'This sweep uses variation_weight=0.8. Delta is expected and non-zero.',
        '',
    ]

    w0p8 = results.get(0.8, {})
    for arch in ARCHITECTURES:
        got_std = (w0p8.get('agg', {}).get(arch) or {}).get('spatial_std_mean')
        got_r = (w0p8.get('bg', {}).get(arch) or {}).get('pearson_r')
        ref_std = STEP4B_SANITY_REFERENCE.get(arch, {}).get('spatial_std_mean')
        ref_r = STEP4B_SANITY_REFERENCE.get(arch, {}).get('bg_r')
        d_std = (got_std - ref_std) if (got_std is not None and ref_std is not None) else None
        d_r = (got_r - ref_r) if (got_r is not None and ref_r is not None) else None
        lines.append(
            f'- {arch}: spatial_std_mean got={_fmt(got_std)} ref(step4,w=1.5)={_fmt(ref_std)} '
            f'delta={_fmt(d_std)} | bg_r got={_fmt(got_r)} ref={_fmt(ref_r)} delta={_fmt(d_r)}'
        )

    lines += [
        '',
        '## sweep results',
        '',
        '| weight | SAGE std | SAGE BG r | GCN-GAT std | GCN-GAT BG r | SAGE act_rate | GCN-GAT act_rate |',
        '|---|---|---|---|---|---|---|',
    ]
    for weight in WEIGHTS:
        res = results.get(weight, {})
        s_std = _fmt((res.get('agg', {}).get('sage') or {}).get('spatial_std_mean'))
        s_r = _fmt((res.get('bg', {}).get('sage') or {}).get('pearson_r'))
        g_std = _fmt((res.get('agg', {}).get('gcn_gat') or {}).get('spatial_std_mean'))
        g_r = _fmt((res.get('bg', {}).get('gcn_gat') or {}).get('pearson_r'))
        s_act = _fmt((res.get('agg', {}).get('sage') or {}).get('variation_loss_activation_rate'))
        g_act = _fmt((res.get('agg', {}).get('gcn_gat') or {}).get('variation_loss_activation_rate'))
        lines.append(f'| {weight} | {s_std} | {s_r} | {g_std} | {g_r} | {s_act} | {g_act} |')

    # soft mode reference row
    lines.append(
        f'| soft (ref) | {_fmt(soft_sage_std)} | {_fmt(soft_sage_r)} | '
        f'{_fmt((soft_agg.get("gcn_gat") or {}).get("spatial_std_mean"))} | '
        f'{_fmt((soft_bg.get("gcn_gat") or {}).get("pearson_r"))} | - | - |'
    )

    lines += [
        '',
        '## primary question',
        '',
        f'Does any variation_weight in [0.8, 4.0] recover SAGE within-tract std to within 10% of soft mode',
        f'({_fmt(soft_sage_std)})? Target: std >= {target_std:.4f} with pooled BG r not dropping by > 0.02.',
        '',
    ]

    # determine verdict
    best_sage_std = None
    best_weight = None
    bg_r_at_best = None
    for weight in WEIGHTS:
        res = results.get(weight, {})
        sv = (res.get('agg', {}).get('sage') or {}).get('spatial_std_mean')
        rv = (res.get('bg', {}).get('sage') or {}).get('pearson_r')
        if sv is not None and math.isfinite(sv):
            if best_sage_std is None or sv > best_sage_std:
                best_sage_std = sv
                best_weight = weight
                bg_r_at_best = rv

    sage_soft_std = (soft_agg.get('sage') or {}).get('spatial_std_mean')
    sage_soft_r = (soft_bg.get('sage') or {}).get('pearson_r')

    recovered = (
        best_sage_std is not None and
        sage_soft_std is not None and
        best_sage_std >= sage_soft_std * 0.9 and
        bg_r_at_best is not None and
        sage_soft_r is not None and
        (sage_soft_r - bg_r_at_best) <= 0.02
    )

    partial = (
        best_sage_std is not None and
        sage_soft_std is not None and
        best_sage_std > (STEP4B_SANITY_REFERENCE.get('sage', {}).get('spatial_std_mean') or 0) and
        not recovered
    )

    if recovered:
        verdict = 'outcome_A'
        verdict_text = (
            f'**Outcome A**: clean recovery. SAGE within-tract std={_fmt(best_sage_std)} '
            f'at variation_weight={best_weight} (ref {_fmt(sage_soft_std)}); '
            f'pooled BG r={_fmt(bg_r_at_best)} (ref {_fmt(sage_soft_r)}; delta={_fmt(bg_r_at_best - sage_soft_r if bg_r_at_best and sage_soft_r else None)}). '
            f'Recommend cbc_no_shift with variation_weight={best_weight} as step 5 production default.'
        )
    elif partial:
        verdict = 'outcome_B'
        verdict_text = (
            f'**Outcome B**: partial recovery. Best SAGE std={_fmt(best_sage_std)} at weight={best_weight} '
            f'(cbc baseline {_fmt(STEP4B_SANITY_REFERENCE.get("sage", {}).get("spatial_std_mean"))}; '
            f'soft ref {_fmt(sage_soft_std)}). '
            'Spread improved but did not reach soft-mode level or trade-offs were unacceptable. '
            'Document trade-off; choose mode on judgment.'
        )
    else:
        verdict = 'outcome_C'
        verdict_text = (
            '**Outcome C**: no recovery. No variation_weight in [0.8, 4.0] restored SAGE spread '
            'without other damage. '
            'Recommend soft mode as step 5 default. Document cbc spread collapse as a finding.'
        )

    lines += [
        '## verdict',
        '',
        verdict_text,
        '',
        '## step 5 launch decision',
        '',
    ]
    if recovered:
        lines.append(f'Proceed with step 5 using `constraint_mode: cbc_no_shift`, `variation_weight: {best_weight}`.')
    else:
        lines.append('Proceed with step 5 using `constraint_mode: soft`. cbc_no_shift limitation documented.')

    lines += [
        '',
        '## figures',
        '',
        '- `variation_weight_sweep.png`: 3x2 grid; rows=metric, cols=architecture; x-axis=weight (log); soft reference lines',
        '- `spread_vs_generalization.png`: 2x1; twin y-axes (within-tract std, pooled BG r) vs variation_weight',
        '- `extreme_tract_recalibration.png`: 2x2; extreme tracts x metrics, soft reference lines',
        '',
    ]

    (summary_dir / 'README.md').write_text('\n'.join(lines) + '\n')


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='ablation step 4b: variation_weight recalibration')
    parser.add_argument('--weight', type=float, help='variation_weight value to run')
    parser.add_argument('--subdir', help='subdirectory name under experiments/ablation/04b_...')
    parser.add_argument('--write-summary', action='store_true',
                        help='read all four result sets and write summary/')
    args = parser.parse_args()

    if args.write_summary:
        write_summary()
        return

    if args.weight is None or not args.subdir:
        parser.print_help()
        sys.exit(1)

    if args.weight not in WEIGHTS:
        print(f'[ablation04b] WARNING: weight {args.weight} not in canonical list {WEIGHTS}; proceeding anyway')

    run_weight(args.weight, args.subdir)


if __name__ == '__main__':
    main()
