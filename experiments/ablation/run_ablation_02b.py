"""
run_ablation_02b: normalization-layer audit (Step 2b).

Runs three sub-experiments in sequence, each differing from 2a (per-tract
z-score, all other settings frozen) by one normalization-layer change:

  2b-i   drop input LayerNorm     input_layernorm=False, conv_norm_type=batchnorm
  2b-ii  replace BatchNorm with Identity  input_layernorm=True,  conv_norm_type=identity
  2b-iii replace BatchNorm with LayerNorm input_layernorm=True,  conv_norm_type=layernorm

Per-variant artifacts land in:
  experiments/ablation/02b_i_no_input_layernorm/
  experiments/ablation/02b_ii_no_batchnorm/
  experiments/ablation/02b_iii_layernorm_in_hidden/

Cross-variant summary lands in:
  experiments/ablation/02b_summary/

Usage:
    python experiments/ablation/run_ablation_02b.py [--verbose] [--only 2b_i]
"""
import argparse
import json
import math
import os
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
CONFIG_PATH   = REPO_ROOT / 'config.yaml'
BG_SVI_PATH   = REPO_ROOT / 'data' / 'processed' / 'national_bg_svi.csv'
GRAVEYARD     = REPO_ROOT / 'graveyard'
TRACT_INV     = REPO_ROOT / 'tract_inventory.csv'

ARCHITECTURES = ['sage', 'gcn_gat']
ARCH_LABELS   = {'sage': 'GRANITE-SAGE', 'gcn_gat': 'GRANITE-GCNGAT'}
SEED          = 42
MIN_ADDRESSES_PER_BG = 3
MIN_ADDRESSES_POOLED = 10

# 2a reference (per-tract z-score baseline for delta table)
REF_2A = {
    'sage':   {'spatial_std_mean': 0.0823208,  'morans_i_mean': 0.8775895,
               'bg_r': 0.7536965003138819,  'spatial_std_std': None},
    'gcn_gat':{'spatial_std_mean': 0.08142490, 'morans_i_mean': 0.84904515,
               'bg_r': 0.7664314601020286,  'spatial_std_std': None},
}
REF_2A_FI_SPEARMAN = 0.116
REF_2A_TOP10_OVERLAP = 3

# stop thresholds
CONSTRAINT_MAX   = 1e-4
SPATIAL_STD_MAX  = 0.3    # flag only
BG_R_MIN         = 0.6    # stop

# variant definitions: name, dir_suffix, norm_layers cfg
VARIANTS = [
    {
        'key':  '2b_i',
        'dir':  '02b_i_no_input_layernorm',
        'label': '2b-i (no input LN)',
        'norm_layers': {'input_layernorm': False, 'conv_norm_type': 'batchnorm'},
        'description': 'drop input LayerNorm; keep BatchNorm on conv layers',
    },
    {
        'key':  '2b_ii',
        'dir':  '02b_ii_no_batchnorm',
        'label': '2b-ii (Identity conv norm)',
        'norm_layers': {'input_layernorm': True, 'conv_norm_type': 'identity'},
        'description': 'keep input LayerNorm; replace conv BatchNorm with Identity',
    },
    {
        'key':  '2b_iii',
        'dir':  '02b_iii_layernorm_in_hidden',
        'label': '2b-iii (LayerNorm conv norm)',
        'norm_layers': {'input_layernorm': True, 'conv_norm_type': 'layernorm'},
        'description': 'keep input LayerNorm; replace conv BatchNorm with LayerNorm',
    },
]


# ---------------------------------------------------------------------------
# shared helpers (mirrors run_ablation_01.py)
# ---------------------------------------------------------------------------

def _load_config():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f) or {}
    for k in ('data', 'model', 'training', 'processing', 'recovery',
              'validation', 'features', 'norm_layers'):
        cfg.setdefault(k, {})
    return cfg


def _load_tract_list():
    df = pd.read_csv(TRACT_INV)
    df['fips'] = df['fips'].astype(str).str.strip()
    return df['fips'].tolist()


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
        'bg_r':   float(np.corrcoef(p[valid], t[valid])[0, 1]),
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


def _fi_spearman_top10(sage_imp, gcngat_imp):
    if sage_imp.empty or gcngat_imp.empty:
        return float('nan'), 0
    merged = sage_imp[['feature', 'rank']].merge(
        gcngat_imp[['feature', 'rank']], on='feature', suffixes=('_sage', '_gcngat'))
    if len(merged) < 2:
        return float('nan'), 0
    rho, _ = stats.spearmanr(merged['rank_sage'], merged['rank_gcngat'])
    overlap = len(set(sage_imp.head(10)['feature']) & set(gcngat_imp.head(10)['feature']))
    return float(rho), int(overlap)


def _load_idw_kriging(tracts_gdf):
    import importlib.util
    gv = GRAVEYARD / 'disaggregation_baselines_idw_kriging.py'
    spec = importlib.util.spec_from_file_location('idw_kriging_graveyard', gv)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    idw  = mod.IDWDisaggregation(power=2.0, n_neighbors=8)
    idw.fit(tracts_gdf, svi_column='RPL_THEMES')
    krig = mod.OrdinaryKrigingDisaggregation()
    krig.fit(tracts_gdf, svi_column='RPL_THEMES')
    return idw, krig


# ---------------------------------------------------------------------------
# per-variant run
# ---------------------------------------------------------------------------

def run_variant(variant, cfg_base, data, bg_gdf, validator,
                idw_model, krig_model, tract_list, verbose):
    """Run one norm-layer variant. Returns (per_tract_df, agg_metrics, bg_validation, importance_agg)."""
    vdir   = ABLATION_ROOT / variant['dir']
    res_dir = vdir / 'results'
    fig_dir = vdir / 'figures'
    fi_dir  = res_dir / 'feature_importance'

    cfg = {k: dict(v) if isinstance(v, dict) else v for k, v in cfg_base.items()}
    cfg['norm_layers'] = dict(variant['norm_layers'])
    cfg['features']['feature_standardization'] = 'per_tract'

    scratch = str(REPO_ROOT / 'output' / f'ablation_{variant["key"]}_scratch')
    os.makedirs(scratch, exist_ok=True)

    pipeline = GRANITEPipeline(cfg, output_dir=scratch)
    pipeline.verbose = verbose

    per_tract_rows = []
    pooled_addr  = {a: [] for a in ARCHITECTURES}
    pooled_preds = {a: [] for a in ARCHITECTURES}
    pooled_addr_base = []
    pooled_idw  = []
    pooled_krig = []
    imp_by_arch = {a: [] for a in ARCHITECTURES}

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

            cfg['data']['target_fips']           = fips
            pipeline.config['data']['target_fips']  = fips
            pipeline.config['model']['architecture'] = arch

            fail_row = {
                'fips': fips, 'architecture': arch, 'n_addresses': 0,
                'tract_svi': float('nan'), 'constraint_error': float('nan'),
                'spatial_std': float('nan'), 'morans_i': float('nan'),
                'bg_r': float('nan'), 'n_bgs': 0, 'runtime_s': 0, 'failure': '',
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

            constr_err = abs(np.mean(preds_arr) - tract_svi)
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
                'runtime_s': round(runtime, 2),
                'failure': '',
            })

            pooled_addr[arch].append(address_gdf.copy())
            pooled_preds[arch].append(preds_arr)

            if arch == ARCHITECTURES[0]:
                pooled_addr_base.append(address_gdf.copy())
                if idw_model is not None:
                    try:
                        ac = np.array([[g.x, g.y] for g in address_gdf.geometry])
                        pooled_idw.append(idw_model.disaggregate(ac, fips, tract_svi))
                    except Exception:
                        pooled_idw.append(np.full(n_addresses, tract_svi))
                if krig_model is not None:
                    try:
                        ac = np.array([[g.x, g.y] for g in address_gdf.geometry])
                        pooled_krig.append(krig_model.disaggregate(ac, fips, tract_svi))
                    except Exception:
                        pooled_krig.append(np.full(n_addresses, tract_svi))

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
            # per-tract BG r is noisy; stop condition applies to pooled BG r only (checked after loop)

            if verbose:
                print(f'  constr_err={constr_err:.2e} sp_std={sp_std:.4f} '
                      f'morans_i={morans_i:.3f} bg_r={bm["bg_r"]:.3f} t={runtime:.1f}s {flag}')

    # -----------------------------------------------------------------------
    # write per_tract_metrics.csv
    # -----------------------------------------------------------------------
    per_tract_df = pd.DataFrame(per_tract_rows)
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
        agg_metrics[arch] = {
            'n_tracts':               int(len(arch_df)),
            'constraint_error_mean':  float(arch_df['constraint_error'].mean()),
            'constraint_error_median':float(arch_df['constraint_error'].median()),
            'spatial_std_mean':       float(arch_df['spatial_std'].mean()),
            'spatial_std_median':     float(arch_df['spatial_std'].median()),
            'spatial_std_std':        float(arch_df['spatial_std'].std()),
            'morans_i_mean':  float(arch_df['morans_i'].dropna().mean())  if arch_df['morans_i'].notna().any() else float('nan'),
            'morans_i_median':float(arch_df['morans_i'].dropna().median()) if arch_df['morans_i'].notna().any() else float('nan'),
            'morans_i_std':   float(arch_df['morans_i'].dropna().std())    if arch_df['morans_i'].notna().any() else float('nan'),
            'bg_r_mean':  float(arch_df['bg_r'].dropna().mean())  if arch_df['bg_r'].notna().any() else float('nan'),
            'bg_r_median':float(arch_df['bg_r'].dropna().median()) if arch_df['bg_r'].notna().any() else float('nan'),
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

    for label, plist in [('IDW', pooled_idw), ('kriging', pooled_krig)]:
        if not plist or not pooled_addr_base:
            bg_validation[label] = {'pearson_r': float('nan')}
            per_tract_bg[label] = pd.DataFrame()
            continue
        combined = pd.concat(pooled_addr_base, ignore_index=True)
        if not isinstance(combined, gpd.GeoDataFrame):
            combined = gpd.GeoDataFrame(combined, geometry='geometry', crs='EPSG:4326')
        if combined.crs is None:
            combined = combined.set_crs('EPSG:4326')
        all_p = np.concatenate(plist)
        try:
            bm_p = _bg_metrics(_aggregate_to_bg(validator, combined, all_p, MIN_ADDRESSES_POOLED), bg_gdf)
            svi_lk = bg_gdf[['GEOID', 'SVI']].dropna(subset=['SVI'])
            ms = _aggregate_to_bg(validator, combined, all_p, 1).merge(svi_lk, on='GEOID', how='inner')
        except Exception:
            bm_p = {'bg_r': float('nan'), 'bg_rmse': float('nan'), 'n_bgs': 0}
            ms = pd.DataFrame()
        bg_validation[label] = {'pearson_r': bm_p['bg_r'], 'bg_rmse': bm_p['bg_rmse'], 'n_bgs': bm_p['n_bgs']}
        per_tract_bg[label] = ms

    with open(res_dir / 'block_group_validation.json', 'w') as f:
        json.dump(bg_validation, f, indent=2,
                  default=lambda x: None if isinstance(x, float) and math.isnan(x) else x)

    # stop condition: pooled BG r < 0.6 on any GRANITE run
    for arch in ARCHITECTURES:
        pooled_r = bg_validation.get(arch, {}).get('pearson_r', float('nan'))
        if math.isfinite(pooled_r) and pooled_r < BG_R_MIN:
            print(f'[{variant["key"]}] STOP: pooled BG r={pooled_r:.3f} < {BG_R_MIN} '
                  f'for {arch}. Normalization removal may have broken generalization.')
            sys.exit(2)

    # feature importance
    importance_agg = {}
    for arch, dfs in imp_by_arch.items():
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
        short = 'sage' if arch == 'sage' else 'gcngat'
        grouped.to_csv(fi_dir / f'{short}_importance.csv', index=False)

    # figures (standard 6)
    _generate_figures(per_tract_df, bg_validation, per_tract_bg, importance_agg, fig_dir, variant)

    # per-variant README
    _write_variant_readme(variant, vdir, per_tract_df, agg_metrics, bg_validation,
                          importance_agg, tract_list)

    print(f'[{variant["key"]}] done: {len(per_tract_df)} rows written')
    return per_tract_df, agg_metrics, bg_validation, importance_agg


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
                          importance_agg, tract_list):
    git_sha = (vdir / 'git_state.txt').read_text().strip()

    def _f(v, fmt='.4f'):
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            return 'n/a'
        return format(v, fmt)

    sa = agg_metrics.get('sage', {})
    ga = agg_metrics.get('gcn_gat', {})
    nl = variant['norm_layers']
    sage_slope   = _spatial_std_slope(per_tract_df, 'sage')
    gcngat_slope = _spatial_std_slope(per_tract_df, 'gcn_gat')

    lines = [
        f'# {variant["dir"]}',
        '',
        f'Ablation 2b sub-experiment: {variant["description"]}.',
        '',
        '## Norm-layer configuration',
        '',
        f'| key | value |',
        f'|---|---|',
        f'| input_layernorm | {nl["input_layernorm"]} |',
        f'| conv_norm_type  | {nl["conv_norm_type"]} |',
        f'| feature_standardization | per_tract (z-score, same as 2a) |',
        '',
        '## Run metadata',
        '',
        f'| field | value |',
        f'|---|---|',
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
        f'| spatial std slope vs SVI | {_f(sage_slope, ".5f")} | {_f(gcngat_slope, ".5f")} |',
        f"| moran's I (mean) | {_f(sa.get('morans_i_mean'))} | {_f(ga.get('morans_i_mean'))} |",
        f'| block-group r (pooled) | {_f(bg_validation.get("sage", {}).get("pearson_r"))} | {_f(bg_validation.get("gcn_gat", {}).get("pearson_r"))} |',
        '',
        '## Delta vs 2a (per-tract z-score baseline)',
        '',
        '| metric | SAGE delta | GCN-GAT delta |',
        '|---|---|---|',
        f'| spatial std (mean) | {_f(sa.get("spatial_std_mean", float("nan")) - REF_2A["sage"]["spatial_std_mean"] if sa else float("nan"), "+.4f")} | {_f(ga.get("spatial_std_mean", float("nan")) - REF_2A["gcn_gat"]["spatial_std_mean"] if ga else float("nan"), "+.4f")} |',
        f"| moran's I (mean)   | {_f(sa.get('morans_i_mean', float('nan')) - REF_2A['sage']['morans_i_mean'] if sa else float('nan'), '+.4f')} | {_f(ga.get('morans_i_mean', float('nan')) - REF_2A['gcn_gat']['morans_i_mean'] if ga else float('nan'), '+.4f')} |",
        f'| block-group r      | {_f(bg_validation.get("sage", {}).get("pearson_r", float("nan")) - REF_2A["sage"]["bg_r"], "+.4f")} | {_f(bg_validation.get("gcn_gat", {}).get("pearson_r", float("nan")) - REF_2A["gcn_gat"]["bg_r"], "+.4f")} |',
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
    """Generate 02b_summary/norm_layer_sweep.png, delta_vs_2a.json, README.md."""
    summary_dir = ABLATION_ROOT / '02b_summary'
    summary_dir.mkdir(exist_ok=True)

    # build delta table
    delta = {}
    for v in VARIANTS:
        key = v['key']
        per_tract_df, agg_metrics, bg_validation, importance_agg = results_by_variant[key]

        sa = agg_metrics.get('sage', {})
        ga = agg_metrics.get('gcn_gat', {})
        fi_rho, fi_top10 = _fi_spearman_top10(
            importance_agg.get('sage', pd.DataFrame()),
            importance_agg.get('gcn_gat', pd.DataFrame())
        )

        def _delta_dict(arch, ref_key, agg):
            if not agg:
                return {}
            ref = REF_2A[ref_key]
            slope_val = _spatial_std_slope(per_tract_df, arch)
            bg_r_val  = bg_validation.get(arch, {}).get('pearson_r', float('nan'))
            return {
                'mean_spatial_std':           agg.get('spatial_std_mean'),
                'mean_spatial_std_delta':     _safe_delta(agg.get('spatial_std_mean'), ref['spatial_std_mean']),
                'mean_morans_i':              agg.get('morans_i_mean'),
                'mean_morans_i_delta':        _safe_delta(agg.get('morans_i_mean'), ref['morans_i_mean']),
                'block_group_r':              bg_r_val,
                'block_group_r_delta':        _safe_delta(bg_r_val, ref['bg_r']),
                'spatial_std_slope_vs_svi':   slope_val,
                'spatial_std_slope_delta':    _safe_delta(slope_val, None),  # no ref slope for 2a available inline
            }

        delta[key] = {
            'label':              v['label'],
            'norm_layers':        v['norm_layers'],
            'sage':   _delta_dict('sage',    'sage',    sa),
            'gcngat': _delta_dict('gcn_gat', 'gcn_gat', ga),
            'feature_importance_spearman_vs_other_arch': {
                'value': round(fi_rho, 3) if math.isfinite(fi_rho) else None,
                'delta_vs_2a': _safe_delta(fi_rho, REF_2A_FI_SPEARMAN),
            },
            'top10_overlap': {
                'value': fi_top10,
                'delta_vs_2a': fi_top10 - REF_2A_TOP10_OVERLAP,
            },
        }

    # load 2a reference per_tract_metrics for slope
    ref_2a_csv = ABLATION_ROOT / '01_per_tract_std' / 'results' / 'per_tract_metrics.csv'
    if ref_2a_csv.exists():
        ref_df = pd.read_csv(ref_2a_csv)
        for arch_key, arch in [('sage', 'sage'), ('gcngat', 'gcn_gat')]:
            ref_slope = _spatial_std_slope(ref_df, arch)
            for key in delta:
                if arch_key in delta[key]:
                    curr_slope = delta[key][arch_key].get('spatial_std_slope_vs_svi')
                    delta[key][arch_key]['spatial_std_slope_delta'] = _safe_delta(curr_slope, ref_slope)

    with open(summary_dir / 'delta_vs_2a.json', 'w') as f:
        json.dump(delta, f, indent=2,
                  default=lambda x: None if isinstance(x, float) and math.isnan(x) else x)
    print('[02b_summary] delta_vs_2a.json written')

    # norm_layer_sweep.png
    _plot_sweep(results_by_variant, summary_dir)

    # summary README
    _write_summary_readme(delta, summary_dir)


def _safe_delta(a, b):
    try:
        if a is None or b is None:
            return None
        v = float(a) - float(b)
        return round(v, 6) if math.isfinite(v) else None
    except Exception:
        return None


def _plot_sweep(results_by_variant, summary_dir):
    """
    4-panel figure: 2 rows (SAGE, GCN-GAT) x 2 cols (spatial std, Moran's I).
    x-axis: [2a baseline, 2b-i, 2b-ii, 2b-iii].
    y-axis: metric mean across 20 tracts with ±1 std error bars.
    """
    x_labels = ['2a\n(baseline)', '2b-i\n(no input LN)', '2b-ii\n(identity)', '2b-iii\n(LN hidden)']
    x_pos    = np.arange(len(x_labels))

    # 2a reference values from 01_per_tract_std
    ref_2a_csv = ABLATION_ROOT / '01_per_tract_std' / 'results' / 'per_tract_metrics.csv'
    ref_df = pd.read_csv(ref_2a_csv) if ref_2a_csv.exists() else pd.DataFrame()

    def _tract_stats(df, arch, col):
        sub = df[(df['architecture'] == arch) &
                 (df['failure'].isna() | (df['failure'] == ''))][col].dropna()
        if len(sub) == 0:
            return float('nan'), float('nan')
        return float(sub.mean()), float(sub.std())

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey='col')

    arch_info = [
        ('sage',    'GRANITE-SAGE',    '#2196F3'),
        ('gcn_gat', 'GRANITE-GCNGAT',  '#FF5722'),
    ]
    col_info = [
        ('spatial_std', 'spatial std (mean)'),
        ('morans_i',    "Moran's I (mean)"),
    ]

    for row, (arch, arch_label, color) in enumerate(arch_info):
        for col, (metric, ylabel) in enumerate(col_info):
            ax = axes[row, col]

            means = []
            stds  = []

            # 2a reference
            if not ref_df.empty:
                m, s = _tract_stats(ref_df, arch, metric)
            else:
                m, s = REF_2A[arch][f'{metric}_mean'], float('nan')
            means.append(m)
            stds.append(s if math.isfinite(s) else 0)

            # 2b variants in order
            for v in VARIANTS:
                vkey = v['key']
                if vkey in results_by_variant:
                    vdf = results_by_variant[vkey][0]
                    m2, s2 = _tract_stats(vdf, arch, metric)
                else:
                    m2, s2 = float('nan'), float('nan')
                means.append(m2)
                stds.append(s2 if math.isfinite(s2) else 0)

            means = np.array(means, dtype=float)
            stds  = np.array(stds,  dtype=float)

            valid = np.isfinite(means)
            if valid.any():
                ax.errorbar(x_pos[valid], means[valid], yerr=stds[valid],
                            fmt='o-', color=color, capsize=4, linewidth=1.5,
                            markersize=6, label=arch_label)
                # reference line (2a)
                if math.isfinite(means[0]):
                    ax.axhline(means[0], color='gray', linestyle=':', linewidth=0.8, alpha=0.6)

            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels, fontsize=8)
            ax.tick_params(labelsize=8)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            if row == 0:
                ax.set_title(ylabel, fontsize=10)
            ax.set_xlabel('')
            if row == 0 and col == 0:
                ax.legend(fontsize=8)

    rows_labels = ['GRANITE-SAGE', 'GRANITE-GCNGAT']
    for ax, label in zip(axes[:, 0], rows_labels):
        ax.set_ylabel(f'{label}\n{ax.get_ylabel()}', fontsize=9)

    fig.suptitle('Normalization Layer Sweep: spatial std and Moran\'s I\nacross 2a baseline and 2b variants',
                 fontsize=11)
    fig.tight_layout()
    out = summary_dir / 'norm_layer_sweep.png'
    fig.savefig(str(out), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'[02b_summary] norm_layer_sweep.png written')


def _write_summary_readme(delta, summary_dir):
    git_sha = (ABLATION_ROOT / '02b_i_no_input_layernorm' / 'git_state.txt').read_text().strip()

    def _f(v, fmt='.4f'):
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            return 'n/a'
        try:
            return format(float(v), fmt)
        except Exception:
            return 'n/a'

    def _largest_mover(metric_key):
        """Return variant key with largest absolute delta on metric_key for each arch."""
        best = {'sage': (None, 0), 'gcn_gat': (None, 0)}
        for vkey, vdata in delta.items():
            for arch in ('sage', 'gcn_gat'):
                arch_short = 'gcngat' if arch == 'gcn_gat' else arch
                d = vdata.get(arch_short, {}).get(metric_key)
                if d is not None and abs(d) > best[arch][1]:
                    best[arch] = (vkey, abs(d))
        return best

    std_movers = _largest_mover('mean_spatial_std_delta')
    mi_movers  = _largest_mover('mean_morans_i_delta')

    # architecture-sensitivity prediction: GCN-GAT spread > SAGE spread on spatial_std and Moran's I?
    def _arch_spread(arch_short, metric_key):
        vals = []
        for vkey, vdata in delta.items():
            d = vdata.get(arch_short, {}).get(metric_key)
            if d is not None:
                vals.append(abs(d))
        return max(vals) if vals else float('nan')

    sage_std_spread   = _arch_spread('sage',   'mean_spatial_std_delta')
    gcngat_std_spread = _arch_spread('gcngat', 'mean_spatial_std_delta')
    sage_mi_spread    = _arch_spread('sage',   'mean_morans_i_delta')
    gcngat_mi_spread  = _arch_spread('gcngat', 'mean_morans_i_delta')

    arch_sens_std = gcngat_std_spread > sage_std_spread
    arch_sens_mi  = gcngat_mi_spread  > sage_mi_spread
    arch_sens_result = arch_sens_std and arch_sens_mi

    lines = [
        '# 02b_summary: normalization-layer audit',
        '',
        'Cross-variant summary for Step 2b of the GRANITE ablation series.',
        'All three variants build on 2a (per-tract z-score standardization).',
        '',
        '## Architecture-sensitivity prediction outcome',
        '',
        'Hypothesis: GCN-GAT spread across 2b-i to 2b-iii exceeds SAGE spread on',
        'both spatial std and Moran\'s I.',
        '',
        '| metric | SAGE spread (max |delta|) | GCN-GAT spread | GCN-GAT > SAGE? |',
        '|---|---|---|---|',
        f'| spatial std | {_f(sage_std_spread, ".4f")} | {_f(gcngat_std_spread, ".4f")} | {"YES" if arch_sens_std else "NO"} |',
        f"| Moran's I   | {_f(sage_mi_spread, '.4f')} | {_f(gcngat_mi_spread, '.4f')} | {'YES' if arch_sens_mi else 'NO'} |",
        '',
        f'**Prediction: {"CONFIRMED" if arch_sens_result else "NOT CONFIRMED"}** '
        f'(both metrics require GCN-GAT > SAGE).',
        '',
        '## Largest absolute movement per architecture',
        '',
        '| metric | SAGE largest mover | GCN-GAT largest mover |',
        '|---|---|---|',
        f'| spatial std | {std_movers["sage"][0] or "n/a"} | {std_movers["gcn_gat"][0] or "n/a"} |',
        f"| Moran's I   | {mi_movers['sage'][0] or 'n/a'} | {mi_movers['gcn_gat'][0] or 'n/a'} |",
        '',
        '## Delta table vs 2a',
        '',
        '### SAGE (GraphSAGE)',
        '',
        '| variant | spatial std | delta | morans_i | delta | bg_r | delta |',
        '|---|---|---|---|---|---|---|',
    ]
    for v in VARIANTS:
        k = v['key']
        s = delta.get(k, {}).get('sage', {})
        lines.append(
            f'| {v["label"]} | {_f(s.get("mean_spatial_std"))} | {_f(s.get("mean_spatial_std_delta"), "+.4f")} | '
            f'{_f(s.get("mean_morans_i"))} | {_f(s.get("mean_morans_i_delta"), "+.4f")} | '
            f'{_f(s.get("block_group_r"))} | {_f(s.get("block_group_r_delta"), "+.4f")} |'
        )
    lines += [
        '',
        '### GCN-GAT',
        '',
        '| variant | spatial std | delta | morans_i | delta | bg_r | delta |',
        '|---|---|---|---|---|---|---|',
    ]
    for v in VARIANTS:
        k = v['key']
        g = delta.get(k, {}).get('gcngat', {})
        lines.append(
            f'| {v["label"]} | {_f(g.get("mean_spatial_std"))} | {_f(g.get("mean_spatial_std_delta"), "+.4f")} | '
            f'{_f(g.get("mean_morans_i"))} | {_f(g.get("mean_morans_i_delta"), "+.4f")} | '
            f'{_f(g.get("block_group_r"))} | {_f(g.get("block_group_r_delta"), "+.4f")} |'
        )
    lines += [
        '',
        '## Artifacts',
        '',
        '- `delta_vs_2a.json`: structured delta table, all three variants',
        '- `norm_layer_sweep.png`: 2x2 panel figure resolving architecture-sensitivity prediction',
        '',
        '## Next step',
        '',
        'Step 3: cross-tract smoothness loss weight sweep.',
        '',
        f'git SHA: `{git_sha}` | seed: 42 | tracts: 20',
        '',
    ]
    with open(summary_dir / 'README.md', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('[02b_summary] README.md written')


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='02b normalization-layer audit')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--only', default=None,
                        help='run only one variant key: 2b_i | 2b_ii | 2b_iii')
    args = parser.parse_args()

    ts_start = datetime.now()
    print(f'[02b] start: {ts_start.strftime("%Y-%m-%d %H:%M:%S")}')
    set_random_seed(SEED)

    if not BG_SVI_PATH.exists():
        print(f'[02b] HALT: {BG_SVI_PATH} missing')
        sys.exit(1)

    tract_list = _load_tract_list()
    print(f'[02b] tracts: {len(tract_list)}')

    print('[02b] loading BG geodataframe...')
    bg_gdf    = _load_bg_gdf()
    validator = BlockGroupValidator(bg_gdf, verbose=False)

    cfg_base = _load_config()
    cfg_base['data']['target']          = 'svi'
    cfg_base['data']['neighbor_tracts'] = 0
    cfg_base['data']['state_fips']      = '47'
    cfg_base['data']['county_fips']     = '065'
    cfg_base['processing']['skip_importance']   = True
    cfg_base['processing']['verbose']           = args.verbose
    cfg_base['processing']['random_seed']       = SEED
    cfg_base['processing']['enable_caching']    = True
    cfg_base['training']['apply_post_correction'] = True

    # load spatial data once
    print('[02b] loading spatial data...')
    init_cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in cfg_base.items()}
    init_cfg['data'] = dict(cfg_base['data'])
    init_cfg['data']['target_fips'] = tract_list[0]
    scratch0 = str(REPO_ROOT / 'output' / 'ablation_02b_spatial_scratch')
    os.makedirs(scratch0, exist_ok=True)
    p0 = GRANITEPipeline(init_cfg, output_dir=scratch0)
    p0.verbose = args.verbose
    try:
        data = p0._load_spatial_data()
    except Exception as e:
        print(f'[02b] HALT: spatial data load failed: {e}')
        sys.exit(1)

    print('[02b] fitting IDW and kriging...')
    try:
        idw_model, krig_model = _load_idw_kriging(data['tracts'])
    except Exception as e:
        print(f'[02b] WARNING: IDW/kriging failed: {e}')
        idw_model, krig_model = None, None

    # determine which variants to run
    variants_to_run = VARIANTS if args.only is None else [
        v for v in VARIANTS if v['key'] == args.only
    ]
    if not variants_to_run:
        print(f'[02b] HALT: --only "{args.only}" does not match any variant key')
        sys.exit(1)

    results_by_variant = {}
    for variant in variants_to_run:
        print(f'\n[02b] ===== {variant["label"]} =====')
        t0 = time.time()
        res = run_variant(
            variant, cfg_base, data, bg_gdf, validator,
            idw_model, krig_model, tract_list, args.verbose
        )
        results_by_variant[variant['key']] = res
        elapsed = time.time() - t0
        print(f'[02b] {variant["key"]} elapsed: {int(elapsed)//60}m {int(elapsed)%60}s')

    # summary (only if all three ran)
    if len(results_by_variant) == 3:
        print('\n[02b] generating cross-variant summary...')
        generate_summary(results_by_variant, args.verbose)
    else:
        print(f'[02b] skipping summary (only {len(results_by_variant)}/3 variants ran)')

    ts_end = datetime.now()
    total_elapsed = (ts_end - ts_start).seconds
    print(f'\n[02b] complete: {ts_end.strftime("%Y-%m-%d %H:%M:%S")} '
          f'({total_elapsed//60}m {total_elapsed%60}s)')


if __name__ == '__main__':
    main()
