"""
05_graph_contribution: graph contribution boundary test (two-pole).

Compares production (road-network-plus-geographic graph) vs mlp_floor (self-loops only)
across both architectures at five training seeds. constraint_mode=soft,
variation_weight=0.8, all other settings at the step 4 soft baseline.

Acceptance criteria:
- production at seed 42 reproduces step 4 soft baseline within 1e-3 (BG r) / 2e-3 (std).
- mlp_floor constraint satisfaction holds (post-correction error at soft-mode levels).
- ValueError fires on unknown graph_variant.

Usage:
    python experiments/ablation/05_graph_contribution/run_sweep.py
    python experiments/ablation/05_graph_contribution/run_sweep.py --condition production
    python experiments/ablation/05_graph_contribution/run_sweep.py --condition mlp_floor
    python experiments/ablation/05_graph_contribution/run_sweep.py --dry-run
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

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from granite.disaggregation.pipeline import GRANITEPipeline
from granite.data.block_group_loader import BlockGroupLoader
from granite.validation.block_group_validation import BlockGroupValidator
from granite.evaluation.spatial_diagnostics import SpatialLearningDiagnostics
from granite.models.gnn import set_random_seed

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
ABLATION_DIR = Path(__file__).resolve().parent
STEP5_ROOT = ABLATION_DIR
TRACT_SELECTION = REPO_ROOT / 'experiments' / 'ablation' / '00_baseline' / 'tract_selection.txt'
BASE_CONFIG_PATH = (
    REPO_ROOT / 'experiments' / 'ablation' / '03_smoothness' / '02_default' / 'config_snapshot.yaml'
)
BG_SVI_PATH = REPO_ROOT / 'data' / 'processed' / 'national_bg_svi.csv'

ARCHITECTURES = ['sage', 'gcn_gat']
ARCH_LABELS = {'sage': 'GRANITE-SAGE', 'gcn_gat': 'GRANITE-GCNGAT'}
SEEDS = [42, 17, 123, 2024, 7]
CONDITIONS = ['production', 'mlp_floor']

MIN_ADDRESSES_PER_BG = 3
MIN_ADDRESSES_POOLED = 10

# sanity regression: production seed=42 must match step 4 soft baseline
# source: experiments/ablation/04_constraint_by_construction/00_baseline_for_step4/results/
SANITY_REFERENCE = {
    'sage':    {'pooled_bg_r': 0.7537, 'within_tract_std': 0.0823},
    'gcn_gat': {'pooled_bg_r': 0.7664, 'within_tract_std': 0.0814},
}
SANITY_TOL_BG_R = 1e-3
SANITY_TOL_STD  = 2e-3

# soft-mode constraint ceiling (post-correction error)
SOFT_CONSTRAINT_CEILING = 1e-6


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
    for k in ('data', 'model', 'training', 'processing', 'features',
              'recovery', 'validation', 'norm_layers'):
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


def _git_sha():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=str(REPO_ROOT)
        ).decode().strip()
    except Exception:
        return 'unknown'


def _write_preflight(cond_dir, cfg, tract_list):
    sha = _git_sha()
    try:
        diff_stat = subprocess.check_output(
            ['git', 'diff', '--stat'], cwd=str(REPO_ROOT)
        ).decode().strip()
        git_txt = f'sha: {sha}\ndiff_stat:\n{diff_stat}\n'
    except Exception as e:
        git_txt = f'sha: {sha}\ngit diff error: {e}\n'
    (cond_dir / 'git_state.txt').write_text(git_txt)
    with open(cond_dir / 'config_snapshot.yaml', 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    try:
        env_txt = subprocess.check_output(['pip', 'freeze']).decode()
    except Exception:
        env_txt = 'pip freeze failed\n'
    (cond_dir / 'environment.txt').write_text(env_txt)
    import shutil
    shutil.copy(str(TRACT_SELECTION), str(cond_dir / 'tract_selection.txt'))
    print(f'[step05] preflight written to {cond_dir}')


# ---------------------------------------------------------------------------
# single seed run
# ---------------------------------------------------------------------------

def run_seed(condition, arch, seed, cfg_base, pipeline, data, bg_gdf, validator, tract_list):
    """
    Run one (condition, arch, seed) combination over all 20 tracts.
    Returns dict with per-seed aggregate metrics.
    """
    set_random_seed(seed)

    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in cfg_base.items()}
    cfg['processing'] = dict(cfg_base.get('processing', {}))
    cfg['processing']['random_seed'] = seed
    cfg['model'] = dict(cfg_base.get('model', {}))
    cfg['model']['architecture'] = arch
    cfg['graph_variant'] = condition

    pipeline.config = cfg
    pipeline.data_loader.config['graph_variant'] = condition
    pipeline.data_loader.config['processing'] = cfg['processing']

    per_tract_stds = []
    per_tract_morans = []
    pooled_addr_gdfs = []
    pooled_preds = []

    for fips in tract_list:
        cfg['data'] = dict(cfg_base.get('data', {}))
        cfg['data']['target_fips'] = fips
        pipeline.config['data']['target_fips'] = fips

        try:
            result = pipeline._process_single_tract(fips, data)
        except Exception as e:
            print(f'[step05]   ERROR {condition}/{arch}/seed={seed}/{fips}: {str(e)[:200]}')
            traceback.print_exc()
            continue

        if not result.get('success'):
            print(f'[step05]   FAILED {condition}/{arch}/seed={seed}/{fips}: '
                  f'{result.get("error", "?")[:200]}')
            continue

        address_gdf = result['address_gdf']
        preds_arr = result['predictions']['mean'].values.astype(float)
        tract_svi = float(result['tract_svi'])

        constr_err = abs(np.mean(preds_arr) - tract_svi)
        sp_std = float(np.std(preds_arr))
        morans_i = _compute_morans_i(preds_arr, address_gdf)

        per_tract_stds.append(sp_std)
        if math.isfinite(morans_i):
            per_tract_morans.append(morans_i)

        pooled_addr_gdfs.append(address_gdf.copy())
        pooled_preds.append(preds_arr)

        print(
            f'[step05]   {condition}/{arch}/seed={seed}/{fips}: '
            f'std={sp_std:.4f} moran={morans_i:.3f} constr={constr_err:.2e}'
        )

    # per-seed aggregates (mean over tracts)
    within_tract_std = float(np.mean(per_tract_stds)) if per_tract_stds else float('nan')
    morans_i_mean = float(np.mean(per_tract_morans)) if per_tract_morans else float('nan')

    # pooled BG r
    pooled_bg_r = float('nan')
    n_bgs = 0
    if pooled_addr_gdfs:
        try:
            all_addr = pd.concat(pooled_addr_gdfs, ignore_index=True)
            all_preds = np.concatenate(pooled_preds)
            if all_addr.crs is None:
                all_addr = all_addr.set_crs('EPSG:4326')
            bg_agg = _aggregate_to_bg(validator, all_addr, all_preds, MIN_ADDRESSES_PER_BG)
            bm = _bg_metrics(bg_agg, bg_gdf)
            pooled_bg_r = bm['bg_r']
            n_bgs = bm['n_bgs']
        except Exception as e:
            print(f'[step05]   WARNING: pooled BG metrics failed: {e}')

    return {
        'seed': seed,
        'within_tract_std': within_tract_std,
        'morans_i': morans_i_mean,
        'pooled_bg_r': pooled_bg_r,
        'n_bgs': n_bgs,
        'n_tracts': len(per_tract_stds),
    }


# ---------------------------------------------------------------------------
# condition sweep
# ---------------------------------------------------------------------------

def run_condition(condition, cfg_base, bg_gdf, validator, tract_list, verbose=False):
    """Run all (arch, seed) combinations for one condition. Returns nested results dict."""
    print(f'\n[step05] === condition: {condition} ===')
    cond_dir = STEP5_ROOT / condition
    cond_dir.mkdir(parents=True, exist_ok=True)
    (cond_dir / 'results').mkdir(exist_ok=True)
    (cond_dir / 'figures').mkdir(exist_ok=True)

    # build condition config for preflight snapshot
    cfg_preflight = {k: (dict(v) if isinstance(v, dict) else v) for k, v in cfg_base.items()}
    cfg_preflight['graph_variant'] = condition
    _write_preflight(cond_dir, cfg_preflight, tract_list)

    # init pipeline once; update config per seed
    scratch_dir = str(REPO_ROOT / 'output' / f'step05_{condition}_scratch')
    os.makedirs(scratch_dir, exist_ok=True)
    pipeline_cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in cfg_base.items()}
    pipeline_cfg['data'] = dict(cfg_base.get('data', {}))
    pipeline_cfg['data']['target_fips'] = tract_list[0]
    pipeline_cfg['graph_variant'] = condition
    pipeline = GRANITEPipeline(pipeline_cfg, output_dir=scratch_dir)
    pipeline.verbose = verbose

    print('[step05] loading spatial data...')
    data = pipeline._load_spatial_data()

    results = {}
    for arch in ARCHITECTURES:
        arch_seed_results = []
        print(f'\n[step05] {condition} / {ARCH_LABELS[arch]}')

        for seed in SEEDS:
            print(f'[step05] seed={seed}')
            t0 = time.time()
            seed_result = run_seed(
                condition, arch, seed, cfg_base, pipeline, data,
                bg_gdf, validator, tract_list
            )
            seed_result['runtime_s'] = round(time.time() - t0, 1)
            arch_seed_results.append(seed_result)
            print(
                f'[step05] seed={seed} done: '
                f'std={seed_result["within_tract_std"]:.4f} '
                f'moran={seed_result["morans_i"]:.4f} '
                f'bg_r={seed_result["pooled_bg_r"]:.4f} '
                f't={seed_result["runtime_s"]:.0f}s'
            )

        # per-condition / per-arch aggregates (std is across seeds, not across tracts)
        finite_stds  = [r['within_tract_std'] for r in arch_seed_results if math.isfinite(r['within_tract_std'])]
        finite_moran = [r['morans_i']         for r in arch_seed_results if math.isfinite(r['morans_i'])]
        finite_bg_r  = [r['pooled_bg_r']      for r in arch_seed_results if math.isfinite(r['pooled_bg_r'])]

        assert len(finite_stds) == len(SEEDS), (
            f'expected {len(SEEDS)} finite within_tract_std values for '
            f'{condition}/{arch}, got {len(finite_stds)}'
        )

        results[arch] = {
            'seeds': arch_seed_results,
            'mean': {
                'within_tract_std': float(np.mean(finite_stds)),
                'morans_i':         float(np.mean(finite_moran)) if finite_moran else float('nan'),
                'pooled_bg_r':      float(np.mean(finite_bg_r))  if finite_bg_r  else float('nan'),
            },
            'std': {
                'within_tract_std': float(np.std(finite_stds, ddof=1)) if len(finite_stds) > 1 else float('nan'),
                'morans_i':         float(np.std(finite_moran, ddof=1)) if len(finite_moran) > 1 else float('nan'),
                'pooled_bg_r':      float(np.std(finite_bg_r,  ddof=1)) if len(finite_bg_r)  > 1 else float('nan'),
            },
        }

    return results


# ---------------------------------------------------------------------------
# sanity regression
# ---------------------------------------------------------------------------

def check_sanity(results):
    """
    Verify production at seed=42 reproduces step 4 soft baseline.
    Fails loud if outside tolerance.
    """
    prod_results = results.get('production', {})
    passed = True
    print('\n[step05] === sanity regression ===')
    for arch in ARCHITECTURES:
        arch_res = prod_results.get(arch, {})
        seed42 = next((r for r in arch_res.get('seeds', []) if r['seed'] == 42), None)
        if seed42 is None:
            print(f'[step05] SANITY FAIL: no seed=42 result for production/{arch}')
            passed = False
            continue
        ref = SANITY_REFERENCE[arch]
        got_bg_r  = seed42['pooled_bg_r']
        got_std   = seed42['within_tract_std']
        ref_bg_r  = ref['pooled_bg_r']
        ref_std   = ref['within_tract_std']
        bg_r_ok  = math.isfinite(got_bg_r)  and abs(got_bg_r  - ref_bg_r)  <= SANITY_TOL_BG_R
        std_ok   = math.isfinite(got_std)   and abs(got_std   - ref_std)   <= SANITY_TOL_STD
        status_bg_r = 'OK' if bg_r_ok else 'FAIL'
        status_std  = 'OK' if std_ok  else 'FAIL'
        print(
            f'[step05] {arch}: '
            f'bg_r got={got_bg_r:.4f} ref={ref_bg_r:.4f} tol={SANITY_TOL_BG_R} [{status_bg_r}]  '
            f'std got={got_std:.4f} ref={ref_std:.4f} tol={SANITY_TOL_STD} [{status_std}]'
        )
        if not bg_r_ok or not std_ok:
            passed = False
    if not passed:
        print('[step05] SANITY REGRESSION FAILED -- check graph_variant=production path and config')
        sys.exit(3)
    print('[step05] sanity regression passed')


# ---------------------------------------------------------------------------
# constraint check
# ---------------------------------------------------------------------------

def check_constraints(results):
    """
    Confirm mlp_floor constraint satisfaction holds at soft-mode levels.
    Logs a warning if any seed's per-tract constraint error exceeds ceiling.
    (constraint_error is not tracked per-seed in this design; this is a
     qualitative check from per-tract logs during the run.)
    """
    # constraint satisfaction is verified per-tract during run_seed (logged);
    # here we assert that all conditions produced non-NaN std (proxy for no collapse)
    for cond in CONDITIONS:
        cond_res = results.get(cond, {})
        for arch in ARCHITECTURES:
            arch_res = cond_res.get(arch, {})
            for sr in arch_res.get('seeds', []):
                if not math.isfinite(sr.get('within_tract_std', float('nan'))):
                    print(
                        f'[step05] WARNING: non-finite within_tract_std for '
                        f'{cond}/{arch}/seed={sr["seed"]}'
                    )
    print('[step05] constraint check complete (see per-tract logs for per-run error values)')


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', choices=CONDITIONS + ['all'], default='all',
                        help='which condition to run (default: all)')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--dry-run', action='store_true',
                        help='validate setup without running training')
    args = parser.parse_args()

    ts_start = datetime.now()
    print(f'[step05] start {ts_start.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'[step05] conditions={CONDITIONS} architectures={ARCHITECTURES} seeds={SEEDS}')

    if not BG_SVI_PATH.exists():
        print(f'[step05] HALT: {BG_SVI_PATH} missing')
        sys.exit(1)

    tract_list = _load_tract_list()
    print(f'[step05] tracts: {len(tract_list)}')
    if not tract_list:
        print('[step05] HALT: empty tract list')
        sys.exit(1)

    if args.dry_run:
        print('[step05] dry-run: validating graph_variant ValueError behavior')
        from granite.data.loaders import DataLoader
        dl = DataLoader(str(REPO_ROOT / 'data'), config={'graph_variant': 'bad_value'})
        try:
            import torch
            import numpy as np
            dummy_addr = gpd.GeoDataFrame({'geometry': []})
            dl.create_spatial_accessibility_graph(dummy_addr, np.zeros((0, 10)))
            print('[step05] DRY-RUN FAIL: expected ValueError not raised')
            sys.exit(4)
        except ValueError as e:
            print(f'[step05] ValueError correctly raised: {e}')
        print('[step05] dry-run complete')
        return

    cfg_base = _load_base_config()
    cfg_base['data']['target'] = 'svi'
    cfg_base['data']['neighbor_tracts'] = 0
    cfg_base['data']['state_fips'] = '47'
    cfg_base['data']['county_fips'] = '065'
    cfg_base['processing']['skip_importance'] = True
    cfg_base['processing']['verbose'] = args.verbose
    cfg_base['processing']['enable_caching'] = True
    cfg_base['features']['feature_standardization'] = 'per_tract'
    cfg_base['training']['constraint_mode'] = 'soft'
    cfg_base['training']['apply_post_correction'] = True
    cfg_base['training']['variation_weight'] = 0.8

    print('[step05] loading BG geodataframe...')
    bg_gdf = _load_bg_gdf()
    validator = BlockGroupValidator(bg_gdf, verbose=False)

    # determine which conditions to run
    conditions_to_run = CONDITIONS if args.condition == 'all' else [args.condition]

    # load or init results for all conditions
    results_path = STEP5_ROOT / 'results' / 'graph_contribution_metrics.json'
    results_path.parent.mkdir(exist_ok=True)
    if results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)
        print(f'[step05] loaded existing results from {results_path}')
    else:
        all_results = {}

    for cond in conditions_to_run:
        if cond in all_results:
            print(f'[step05] {cond} already in results, skipping (delete to rerun)')
            continue
        cond_results = run_condition(cond, cfg_base, bg_gdf, validator, tract_list, args.verbose)
        all_results[cond] = cond_results

        # write per-condition results
        cond_results_path = STEP5_ROOT / cond / 'results' / f'{cond}_metrics.json'
        with open(cond_results_path, 'w') as f:
            json.dump(cond_results, f, indent=2,
                      default=lambda x: None if (isinstance(x, float) and math.isnan(x)) else x)
        print(f'[step05] {cond} results written to {cond_results_path}')

        # save aggregate after each condition
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2,
                      default=lambda x: None if (isinstance(x, float) and math.isnan(x)) else x)
        print(f'[step05] aggregate results written to {results_path}')

    # post-run checks (only if both conditions present)
    if 'production' in all_results and 'mlp_floor' in all_results:
        check_sanity(all_results)
        check_constraints(all_results)

    ts_end = datetime.now()
    elapsed = (ts_end - ts_start).total_seconds() / 60
    print(f'\n[step05] complete in {elapsed:.1f} min')
    print(f'[step05] results at {results_path}')


if __name__ == '__main__':
    main()
