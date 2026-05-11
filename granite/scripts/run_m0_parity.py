"""
M0 parity driver: GraphSAGE vs Dasymetric vs Pycnophylactic on n20 SVI target.

Runs GRANITEPipeline in single-tract SVI mode for each of 20 stratified tracts,
validates each method at block-group resolution using national ACS SVI as ground
truth, and produces the parity decision artifact.

Usage:
    python granite/scripts/run_m0_parity.py [--config config.yaml] [--verbose]
"""
import argparse
import json
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd
import geopandas as gpd
import yaml
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from granite.disaggregation.pipeline import GRANITEPipeline
from granite.data.block_group_loader import BlockGroupLoader
from granite.validation.block_group_validation import BlockGroupValidator
from granite.evaluation.baselines import DasymetricDisaggregation, PycnophylacticDisaggregation
from granite.models.gnn import set_random_seed

N20_LIST_PATH = 'output/m2_n20_recovery/summary/n20_tract_list.txt'
BG_SVI_PATH = 'data/processed/national_bg_svi.csv'
OUTPUT_DIR = 'data/results/m0_n20_svi_parity'
SCRATCH_DIR = 'output/m0_parity_scratch'
SEED = 42
METHODS = ['GRANITE', 'Dasymetric', 'Pycnophylactic']
MAX_FAILURES_PER_METHOD = 3
MIN_ADDRESSES_PER_BG_PERTRACT = 3   # lowered for per-tract (few BGs per tract)
MIN_ADDRESSES_PER_BG_POOLED = 10    # standard threshold for pooled validation


def _load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    for k in ('data', 'model', 'training', 'processing', 'recovery', 'validation'):
        cfg.setdefault(k, {})
    return cfg


def _load_n20():
    if not os.path.exists(N20_LIST_PATH):
        print(f'[m0] n20 list not found: {N20_LIST_PATH}')
        sys.exit(1)
    with open(N20_LIST_PATH) as f:
        lst = [l.strip() for l in f if l.strip()]
    if len(lst) != 20:
        print(f'[m0] expected 20 tracts in n20 list, got {len(lst)}; halting')
        sys.exit(1)
    return lst


def _load_bg_gdf():
    """Load Hamilton County BG geometries merged with national SVI."""
    loader = BlockGroupLoader(data_dir='./data', verbose=False)
    bg_gdf = loader.get_block_groups_with_demographics('47', '065', svi_ranking_scope='national')
    if bg_gdf is None or len(bg_gdf) == 0:
        raise RuntimeError('BlockGroupLoader returned empty GeoDataFrame')
    if bg_gdf.crs is None:
        bg_gdf = bg_gdf.set_crs('EPSG:4326')
    elif bg_gdf.crs.to_epsg() != 4326:
        bg_gdf = bg_gdf.to_crs('EPSG:4326')
    return bg_gdf


def _aggregate_to_bg(validator, address_gdf, preds, min_addresses):
    """
    Use BlockGroupValidator internals to aggregate address predictions to BG means.
    Returns DataFrame with GEOID, predicted_svi, n_addresses; drops BGs below threshold.
    """
    # ensure CRS match before spatial join
    if address_gdf.crs is None:
        address_gdf = address_gdf.set_crs('EPSG:4326')
    addresses_with_bg = validator._assign_to_block_groups(address_gdf)
    bg_agg = validator._aggregate_predictions(addresses_with_bg, preds)
    bg_agg = bg_agg[bg_agg['n_addresses'] >= min_addresses].copy()
    return bg_agg


def _bg_metrics(bg_agg, bg_gdf):
    """
    Join aggregated predictions to national SVI, compute Pearson r and RMSE.
    Returns dict with bg_r, bg_rmse, n_bgs.
    """
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


def _extract_baseline_preds(baseline_result, fips, tract_svi, address_gdf, data, verbose):
    """
    Pull Dasymetric and Pycnophylactic predictions out of pipeline baseline_comparisons.
    Falls back to direct computation if pipeline baseline failed.
    """
    dasy_preds = None
    pycno_preds = None

    if isinstance(baseline_result, dict) and 'methods' in baseline_result:
        methods = baseline_result['methods']
        d = methods.get('Dasymetric', {})
        p = methods.get('Pycnophylactic', {})
        if isinstance(d, dict) and 'predictions' in d:
            dasy_preds = np.array(d['predictions'], dtype=float)
        if isinstance(p, dict) and 'predictions' in p:
            pycno_preds = np.array(p['predictions'], dtype=float)

    if dasy_preds is None or pycno_preds is None:
        if verbose:
            print(f'[m0]   {fips}: pipeline baseline missing, computing directly')
        address_coords = np.array([[g.x, g.y] for g in address_gdf.geometry])
        if dasy_preds is None:
            dasy = DasymetricDisaggregation(ancillary_column='nlcd_impervious_pct')
            dasy.fit(data['tracts'], svi_column='RPL_THEMES')
            dasy_preds = dasy.disaggregate(address_coords, fips, tract_svi,
                                           address_gdf=address_gdf)
        if pycno_preds is None:
            pycno = PycnophylacticDisaggregation(n_iterations=50, k_neighbors=8)
            pycno.fit(data['tracts'], svi_column='RPL_THEMES')
            pycno_preds = pycno.disaggregate(address_coords, fips, tract_svi,
                                             address_gdf=address_gdf)

    return (np.array(dasy_preds, dtype=float) if dasy_preds is not None else None,
            np.array(pycno_preds, dtype=float) if pycno_preds is not None else None)


def _bootstrap_median_ci(vals, n_boot=1000, seed=42):
    """Bootstrap 95% CI on median. Returns (median, ci_low, ci_high)."""
    rng = np.random.default_rng(seed)
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return float('nan'), float('nan'), float('nan')
    med = float(np.median(v))
    boots = [float(np.median(rng.choice(v, size=len(v), replace=True)))
             for _ in range(n_boot)]
    return med, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _bootstrap_diff_ci(a_vals, b_vals, n_boot=1000, seed=42):
    """Bootstrap 95% CI on median pairwise difference (a - b)."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a_vals, dtype=float)
    b = np.asarray(b_vals, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]; b = b[mask]
    if len(a) < 2:
        return float('nan'), float('nan'), float('nan')
    diffs = a - b
    obs = float(np.median(diffs))
    boots = [float(np.median(rng.choice(diffs, size=len(diffs), replace=True)))
             for _ in range(n_boot)]
    return obs, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _bootstrap_pooled_r_ci(all_gdfs, all_preds_by_method, validator, bg_gdf, n_boot=1000, seed=42):
    """
    Bootstrap CI on pooled BG Pearson r by resampling block groups with replacement.
    Returns dict: method -> (r_obs, ci_low, ci_high, n_bgs).
    """
    rng = np.random.default_rng(seed)
    results = {}

    # build one merged table per method: GEOID, predicted_svi, SVI
    combined_gdf = pd.concat(all_gdfs, ignore_index=True)
    if not hasattr(combined_gdf, 'geometry') or combined_gdf.geometry.name not in combined_gdf.columns:
        combined_gdf = gpd.GeoDataFrame(combined_gdf, geometry='geometry', crs='EPSG:4326')

    svi_lookup = bg_gdf[['GEOID', 'SVI']].dropna(subset=['SVI'])

    for method in METHODS:
        if method not in all_preds_by_method or not all_preds_by_method[method]:
            results[method] = (float('nan'), float('nan'), float('nan'), 0)
            continue
        pooled_preds = np.concatenate(all_preds_by_method[method])
        bg_agg = _aggregate_to_bg(validator, combined_gdf, pooled_preds,
                                   MIN_ADDRESSES_PER_BG_POOLED)
        merged = bg_agg.merge(svi_lookup, on='GEOID', how='inner')
        merged = merged.dropna(subset=['predicted_svi', 'SVI'])
        n_bgs = len(merged)
        if n_bgs < 2:
            results[method] = (float('nan'), float('nan'), float('nan'), n_bgs)
            continue

        p = merged['predicted_svi'].values.astype(float)
        t = merged['SVI'].values.astype(float)
        r_obs = float(np.corrcoef(p, t)[0, 1])

        # bootstrap by resampling BGs
        boots = []
        for _ in range(n_boot):
            idx = rng.integers(0, n_bgs, size=n_bgs)
            ps = p[idx]; ts = t[idx]
            if np.std(ps) < 1e-9 or np.std(ts) < 1e-9:
                continue
            boots.append(float(np.corrcoef(ps, ts)[0, 1]))
        if len(boots) < 10:
            results[method] = (r_obs, float('nan'), float('nan'), n_bgs)
        else:
            results[method] = (r_obs,
                               float(np.percentile(boots, 2.5)),
                               float(np.percentile(boots, 97.5)),
                               n_bgs)
    return results


def main():
    parser = argparse.ArgumentParser(description='M0 parity driver')
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    wall_start = time.time()
    set_random_seed(SEED)

    # --- preflight checks ---
    if not os.path.exists(BG_SVI_PATH):
        print(f'[m0] HALT: {BG_SVI_PATH} missing; cannot proceed')
        sys.exit(1)

    n20_fips = _load_n20()
    print(f'[m0] n20 tracts loaded: {len(n20_fips)}')

    cfg = _load_config(args.config)

    # configure for single-tract SVI mode with GraphSAGE
    cfg['data']['target'] = 'svi'
    cfg['data']['neighbor_tracts'] = 0
    cfg['data']['state_fips'] = '47'
    cfg['data']['county_fips'] = '065'
    cfg['data']['target_fips'] = n20_fips[0]
    cfg['model']['architecture'] = 'sage'
    cfg['processing']['skip_importance'] = True
    cfg['processing']['verbose'] = False
    cfg['processing']['random_seed'] = SEED
    cfg['processing']['enable_caching'] = True

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SCRATCH_DIR, exist_ok=True)

    # load BG geodataframe
    print('[m0] loading BG geodataframe...')
    try:
        bg_gdf = _load_bg_gdf()
        n_complete = int(bg_gdf['svi_complete'].sum()) if 'svi_complete' in bg_gdf.columns else len(bg_gdf)
        print(f'[m0] {len(bg_gdf)} Hamilton County BGs, {n_complete} with complete SVI')
    except Exception as e:
        print(f'[m0] HALT: BG loading failed: {e}')
        sys.exit(1)

    validator = BlockGroupValidator(bg_gdf, verbose=False)

    # initialize pipeline and load spatial data once
    print('[m0] initializing pipeline...')
    pipeline = GRANITEPipeline(cfg, output_dir=SCRATCH_DIR)
    pipeline.verbose = False

    print('[m0] loading spatial data...')
    try:
        data = pipeline._load_spatial_data()
    except Exception as e:
        print(f'[m0] HALT: spatial data loading failed: {e}')
        sys.exit(1)

    # --- main loop ---
    per_tract_rows = []
    all_preds_by_method = {m: [] for m in METHODS}
    all_addr_gdfs = []
    failure_counts = {m: 0 for m in METHODS}

    for idx, fips in enumerate(n20_fips):
        # stop condition: too many failures
        for m in METHODS:
            if failure_counts[m] > MAX_FAILURES_PER_METHOD:
                print(f'[m0] HALT: {m} failed on {failure_counts[m]} tracts '
                      f'(> {MAX_FAILURES_PER_METHOD})')
                sys.exit(1)

        print(f'[m0] tract {idx+1}/20: {fips}')
        t_start = time.time()

        cfg['data']['target_fips'] = fips
        try:
            result = pipeline._process_single_tract(fips, data)
        except Exception as e:
            msg = str(e)[:120]
            print(f'[m0]   ERROR {fips}: {msg}')
            for m in METHODS:
                failure_counts[m] += 1
                per_tract_rows.append({
                    'fips': fips, 'method': m, 'n_addresses': 0, 'n_bgs': 0,
                    'bg_r': float('nan'), 'bg_rmse': float('nan'),
                    'constraint_error': float('nan'), 'spatial_std': float('nan'),
                    'runtime_s': 0, 'failure': msg,
                })
            continue

        if not result.get('success'):
            msg = result.get('error', 'unknown')[:120]
            print(f'[m0]   FAILED {fips}: {msg}')
            for m in METHODS:
                failure_counts[m] += 1
                per_tract_rows.append({
                    'fips': fips, 'method': m, 'n_addresses': 0, 'n_bgs': 0,
                    'bg_r': float('nan'), 'bg_rmse': float('nan'),
                    'constraint_error': float('nan'), 'spatial_std': float('nan'),
                    'runtime_s': 0, 'failure': msg,
                })
            continue

        address_gdf = result['address_gdf']
        granite_preds = result['predictions']['mean'].values.astype(float)
        tract_svi = float(result['tract_svi'])
        runtime = time.time() - t_start
        n_addresses = len(address_gdf)

        # extract baseline predictions
        dasy_preds, pycno_preds = _extract_baseline_preds(
            result.get('baseline_comparisons', {}),
            fips, tract_svi, address_gdf, data, args.verbose,
        )

        preds_by_method = {
            'GRANITE': granite_preds,
            'Dasymetric': dasy_preds,
            'Pycnophylactic': pycno_preds,
        }

        all_addr_gdfs.append(address_gdf.copy())

        for method in METHODS:
            preds = preds_by_method[method]
            if preds is None:
                failure_counts[method] += 1
                per_tract_rows.append({
                    'fips': fips, 'method': method, 'n_addresses': n_addresses, 'n_bgs': 0,
                    'bg_r': float('nan'), 'bg_rmse': float('nan'),
                    'constraint_error': float('nan'), 'spatial_std': float('nan'),
                    'runtime_s': runtime if method == 'GRANITE' else 0,
                    'failure': 'predictions unavailable',
                })
                continue

            all_preds_by_method[method].append(preds)

            # per-tract BG validation (may be low-n for small tracts)
            try:
                bg_agg = _aggregate_to_bg(validator, address_gdf, preds,
                                          MIN_ADDRESSES_PER_BG_PERTRACT)
                bm = _bg_metrics(bg_agg, bg_gdf)
            except Exception as e:
                bm = {'bg_r': float('nan'), 'bg_rmse': float('nan'), 'n_bgs': 0}
                if args.verbose:
                    print(f'[m0]   {fips}/{method} BG metrics failed: {e}')

            constr_err = (abs(np.mean(preds) - tract_svi)
                         / max(abs(tract_svi), 1e-10) * 100.0)
            sp_std = float(np.std(preds))

            per_tract_rows.append({
                'fips': fips, 'method': method,
                'n_addresses': n_addresses, 'n_bgs': bm['n_bgs'],
                'bg_r': bm['bg_r'], 'bg_rmse': bm['bg_rmse'],
                'constraint_error': round(constr_err, 4),
                'spatial_std': round(sp_std, 6),
                'runtime_s': round(runtime, 2) if method == 'GRANITE' else 0,
                'failure': '',
            })
            if args.verbose:
                print(f'[m0]     {method}: bg_r={bm["bg_r"]:.3f}, '
                      f'n_bgs={bm["n_bgs"]}, constr_err={constr_err:.2f}%')

    # --- write per_tract.csv ---
    per_tract_df = pd.DataFrame(per_tract_rows)
    per_tract_df.to_csv(os.path.join(OUTPUT_DIR, 'per_tract.csv'), index=False)
    print(f'[m0] per_tract.csv written ({len(per_tract_df)} rows)')

    # --- pooled BG validation across all 20 tracts ---
    print('[m0] computing pooled BG validation...')
    if not all_addr_gdfs:
        print('[m0] HALT: no tracts succeeded; cannot compute aggregate metrics')
        sys.exit(1)

    combined_gdf = pd.concat(all_addr_gdfs, ignore_index=True)
    if not isinstance(combined_gdf, gpd.GeoDataFrame):
        combined_gdf = gpd.GeoDataFrame(combined_gdf, geometry='geometry', crs='EPSG:4326')
    if combined_gdf.crs is None:
        combined_gdf = combined_gdf.set_crs('EPSG:4326')

    pooled_metrics = {}
    for method in METHODS:
        if not all_preds_by_method[method]:
            pooled_metrics[method] = {'bg_r': float('nan'), 'bg_rmse': float('nan'), 'n_bgs': 0}
            continue
        pooled_preds = np.concatenate(all_preds_by_method[method])
        try:
            bg_agg = _aggregate_to_bg(validator, combined_gdf, pooled_preds,
                                       MIN_ADDRESSES_PER_BG_POOLED)
            pooled_metrics[method] = _bg_metrics(bg_agg, bg_gdf)
        except Exception as e:
            pooled_metrics[method] = {'bg_r': float('nan'), 'bg_rmse': float('nan'), 'n_bgs': 0}
            print(f'[m0] pooled BG metrics failed for {method}: {e}')

    # --- bootstrap CIs ---
    print('[m0] bootstrapping CIs...')
    pooled_boot = _bootstrap_pooled_r_ci(
        all_addr_gdfs, all_preds_by_method, validator, bg_gdf,
        n_boot=1000, seed=SEED,
    )

    # --- aggregate.csv ---
    agg_rows = []
    bg_r_per_tract = {}
    for method in METHODS:
        subset = per_tract_df[per_tract_df['method'] == method]
        vals = subset['bg_r'].values.astype(float)
        bg_r_per_tract[method] = vals
        med, ci_lo, ci_hi = _bootstrap_median_ci(vals, seed=SEED)
        n_ok = int(np.isfinite(vals).sum())
        pm = pooled_metrics.get(method, {})
        pb = pooled_boot.get(method, (float('nan'), float('nan'), float('nan'), 0))
        agg_rows.append({
            'method': method,
            'median_bg_r': round(med, 4) if np.isfinite(med) else float('nan'),
            'ci_low_95': round(ci_lo, 4) if np.isfinite(ci_lo) else float('nan'),
            'ci_high_95': round(ci_hi, 4) if np.isfinite(ci_hi) else float('nan'),
            'n_tracts': n_ok,
            'pooled_bg_r': round(pm.get('bg_r', float('nan')), 4),
            'pooled_bg_r_ci_low': round(pb[1], 4) if np.isfinite(pb[1]) else float('nan'),
            'pooled_bg_r_ci_high': round(pb[2], 4) if np.isfinite(pb[2]) else float('nan'),
            'pooled_n_bgs': pm.get('n_bgs', 0),
        })

    # pairwise difference CIs (per-tract bg_r)
    pairs = [
        ('granite_vs_dasymetric', 'GRANITE', 'Dasymetric'),
        ('granite_vs_pycno', 'GRANITE', 'Pycnophylactic'),
        ('dasymetric_vs_pycno', 'Dasymetric', 'Pycnophylactic'),
    ]
    diff_rows = []
    for label, ma, mb in pairs:
        a_df = (per_tract_df[per_tract_df['method'] == ma][['fips', 'bg_r']]
                .rename(columns={'bg_r': 'ra'}))
        b_df = (per_tract_df[per_tract_df['method'] == mb][['fips', 'bg_r']]
                .rename(columns={'bg_r': 'rb'}))
        paired = a_df.merge(b_df, on='fips').dropna(subset=['ra', 'rb'])
        obs, lo, hi = _bootstrap_diff_ci(paired['ra'].values, paired['rb'].values,
                                         seed=SEED)
        separable = bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0))
        diff_rows.append({
            'pair': label,
            'obs_median_diff': round(obs, 4) if np.isfinite(obs) else float('nan'),
            'ci_low_95': round(lo, 4) if np.isfinite(lo) else float('nan'),
            'ci_high_95': round(hi, 4) if np.isfinite(hi) else float('nan'),
            'n_pairs': len(paired),
            'separable': separable,
        })

    agg_df = pd.DataFrame(agg_rows)
    diff_df = pd.DataFrame(diff_rows)
    agg_df.to_csv(os.path.join(OUTPUT_DIR, 'aggregate.csv'), index=False)
    diff_df.to_csv(os.path.join(OUTPUT_DIR, 'pairwise_diffs.csv'), index=False)
    print(f'[m0] aggregate.csv and pairwise_diffs.csv written')

    # --- determine parity decision ---
    granite_pooled_r = pooled_metrics.get('GRANITE', {}).get('bg_r', float('nan'))
    dasy_pooled_r = pooled_metrics.get('Dasymetric', {}).get('bg_r', float('nan'))
    pycno_pooled_r = pooled_metrics.get('Pycnophylactic', {}).get('bg_r', float('nan'))

    gd_diff = diff_rows[0] if diff_rows else {}
    gp_diff = diff_rows[1] if len(diff_rows) > 1 else {}

    granite_vs_dasy_sep = gd_diff.get('separable', False)
    granite_vs_pycno_sep = gp_diff.get('separable', False)

    if np.isfinite(granite_pooled_r):
        if not granite_vs_dasy_sep and not granite_vs_pycno_sep:
            parity_sentence = (
                f'GRANITE (pooled BG r={granite_pooled_r:.3f}) is not statistically '
                f'separable from Dasymetric (r={dasy_pooled_r:.3f}) or Pycnophylactic '
                f'(r={pycno_pooled_r:.3f}) at BG resolution across the n20 subset; '
                f'parity holds and the Narrative-A parity footnote survives.'
            )
        elif not granite_vs_dasy_sep:
            parity_sentence = (
                f'GRANITE (pooled BG r={granite_pooled_r:.3f}) is not separable from '
                f'Dasymetric (r={dasy_pooled_r:.3f}) but is separable from Pycnophylactic '
                f'(r={pycno_pooled_r:.3f}); partial parity -- verify Narrative-A framing.'
            )
        else:
            better = ('above' if granite_pooled_r > max(dasy_pooled_r if np.isfinite(dasy_pooled_r) else -99,
                                                         pycno_pooled_r if np.isfinite(pycno_pooled_r) else -99)
                      else 'below')
            parity_sentence = (
                f'GRANITE (pooled BG r={granite_pooled_r:.3f}) is statistically '
                f'separable from Dasymetric (r={dasy_pooled_r:.3f}) -- GRANITE is '
                f'{better} the baselines; '
                f'retire or revise the Narrative-A parity footnote.'
            )
    else:
        parity_sentence = (
            'Pooled BG r could not be computed for GRANITE; parity decision deferred.'
        )

    # --- write RESULTS.md ---
    wall_total = time.time() - wall_start
    n_tracts_ok = len(all_addr_gdfs)

    results_md_lines = [
        '# M0 n20 SVI Parity Results',
        '',
        f'**Decision:** {parity_sentence}',
        '',
        f'Run date: 2026-05-09  |  n20 tracts attempted: 20  |  succeeded: {n_tracts_ok}'
        f'  |  wall-clock: {wall_total/60:.1f} min',
        '',
        '---',
        '',
        '## Aggregate metrics',
        '',
        '### Pooled BG r (all 20 tracts combined, min 10 addresses/BG)',
        '',
        '| Method | pooled_bg_r | CI low 95 | CI high 95 | n_BGs |',
        '|--------|-------------|-----------|------------|-------|',
    ]
    for row in agg_rows:
        m = row['method']
        results_md_lines.append(
            f"| {m} | {row['pooled_bg_r']:.3f} | {row['pooled_bg_r_ci_low']:.3f} | "
            f"{row['pooled_bg_r_ci_high']:.3f} | {row['pooled_n_bgs']} |"
        )

    results_md_lines += [
        '',
        '### Per-tract median BG r (bootstrap on per-tract values)',
        '',
        '| Method | median_bg_r | CI low 95 | CI high 95 | n_tracts_with_r |',
        '|--------|-------------|-----------|------------|-----------------|',
    ]
    for row in agg_rows:
        m = row['method']
        results_md_lines.append(
            f"| {m} | {row['median_bg_r']} | {row['ci_low_95']} | "
            f"{row['ci_high_95']} | {row['n_tracts']} |"
        )

    results_md_lines += [
        '',
        '## Pairwise comparisons',
        '',
        '| Pair | obs_median_diff | CI low 95 | CI high 95 | n_pairs | separable |',
        '|------|----------------|-----------|------------|---------|-----------|',
    ]
    for row in diff_rows:
        results_md_lines.append(
            f"| {row['pair']} | {row['obs_median_diff']} | {row['ci_low_95']} | "
            f"{row['ci_high_95']} | {row['n_pairs']} | {row['separable']} |"
        )

    results_md_lines += [
        '',
        '## Discussion',
        '',
        'Primary metric is **pooled BG r**: all 20 tracts combined into one '
        'pool, addresses aggregated to BG means (min 10 addresses/BG), '
        'compared against nationally-ranked ACS BG SVI.',
        '',
        'Per-tract BG r is also reported but has low statistical power: most '
        'tracts contain only 2-5 BGs after the min-address threshold, making '
        'per-tract r unreliable as a standalone metric.',
        '',
        'Separability test: a pair is "separable" when the 95% bootstrap CI '
        'on median difference (from per-tract values) excludes zero.',
        '',
        '### Constraint error sanity check',
        '',
        '| Method | median_constraint_error_pct |',
        '|--------|-----------------------------|',
    ]
    for method in METHODS:
        subset = per_tract_df[per_tract_df['method'] == method]
        med_ce = float(np.nanmedian(subset['constraint_error'].values.astype(float)))
        results_md_lines.append(f'| {method} | {med_ce:.4f} |')

    results_md_lines += [
        '',
        'Dasymetric and Pycnophylactic satisfy the aggregate constraint by '
        'construction (mean-preservation); nonzero values here indicate rounding '
        'or clipping effects only.',
        '',
        'GRANITE constraint error reflects the soft-loss training penalty; values '
        'above 5% indicate the constraint was not well-enforced for that tract.',
    ]

    results_md_path = os.path.join(OUTPUT_DIR, 'RESULTS.md')
    with open(results_md_path, 'w') as f:
        f.write('\n'.join(results_md_lines) + '\n')
    print(f'[m0] RESULTS.md written')

    # --- write or append Research_Status.md ---
    rs_path = 'Research_Status.md'
    m0_entry_lines = [
        '',
        '---',
        '',
        f'## M0: n20 SVI parity (2026-05-09)',
        '',
        f'- n_tracts: {n_tracts_ok}/20',
        f'- GRANITE pooled BG r: {granite_pooled_r:.3f}',
        f'- Dasymetric pooled BG r: {dasy_pooled_r:.3f}',
        f'- Pycnophylactic pooled BG r: {pycno_pooled_r:.3f}',
        f'- Decision: {parity_sentence}',
    ]

    if os.path.exists(rs_path):
        with open(rs_path, 'a') as f:
            f.write('\n'.join(m0_entry_lines) + '\n')
    else:
        header_lines = [
            '# Research Status',
            '',
            'Running log of completed experiments and decisions.',
        ]
        with open(rs_path, 'w') as f:
            f.write('\n'.join(header_lines + m0_entry_lines) + '\n')
    print(f'[m0] Research_Status.md updated')

    # --- print summary ---
    print(f'\n[m0] === M0 SUMMARY ===')
    print(f'[m0] tracts: {n_tracts_ok}/20 succeeded')
    for method in METHODS:
        pm = pooled_metrics.get(method, {})
        print(f'[m0] {method}: pooled BG r = {pm.get("bg_r", float("nan")):.3f}, '
              f'n_bgs = {pm.get("n_bgs", 0)}')
    print(f'[m0] decision: {parity_sentence}')
    print(f'[m0] wall-clock: {wall_total/60:.1f} min')
    print(f'[m0] outputs: {OUTPUT_DIR}/')


if __name__ == '__main__':
    main()
