"""
persist_per_address_predictions.py  --  PATH B (no checkpoint)

Re-runs the m0 parity pipeline for all 20 n20 tracts under the frozen config
(GraphSAGE, seed=42, single-tract SVI mode, apply_post_correction=True).
Captures per-address SVI predictions for GRANITE, Dasymetric, and Pycnophylactic
from the same pipeline run so all three arrays share the same address ordering.

Provenance guard (hard gate):
    Aggregates each array to block groups, computes pooled Pearson r vs national
    ACS SVI. Asserts reproduction within tol=0.005 of frozen reference values:
        GRANITE       0.7692
        Dasymetric    0.8018
        Pycnophylactic 0.7678

Output:
    experiments/recovery/per_address_predictions/granite_m0.parquet
    experiments/recovery/per_address_predictions/dasymetric.parquet
    experiments/recovery/per_address_predictions/pycnophylactic.parquet
    experiments/recovery/per_address_predictions/provenance.json

Each parquet schema: fips (str), address_idx (int), svi_pred (float64)
Row order matches n20_feature_matrix.csv exactly (same get_addresses_for_tract
call order per tract, 0-based address_idx within each tract).

Usage:
    python scripts/persist_per_address_predictions.py [--verbose]
"""
import argparse
import hashlib
import json
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd
import geopandas as gpd
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from granite.disaggregation.pipeline import GRANITEPipeline
from granite.data.block_group_loader import BlockGroupLoader
from granite.validation.block_group_validation import BlockGroupValidator
from granite.evaluation.baselines import DasymetricDisaggregation, PycnophylacticDisaggregation
from granite.models.gnn import set_random_seed

# -----------------------------------------------------------------------
# constants
# -----------------------------------------------------------------------
SEED = 42
N20_LIST_PATH = os.path.join(REPO_ROOT, 'output', 'm2_n20_recovery', 'summary', 'n20_tract_list.txt')
BG_SVI_PATH = os.path.join(REPO_ROOT, 'data', 'processed', 'national_bg_svi.csv')
OUTPUT_DIR = os.path.join(REPO_ROOT, 'experiments', 'recovery', 'per_address_predictions')
SCRATCH_DIR = os.path.join(REPO_ROOT, 'output', 'persist_preds_scratch')
CONFIG_PATH = os.path.join(REPO_ROOT, 'config.yaml')

FROZEN_BG_R = {
    'GRANITE': 0.7692,
    'Dasymetric': 0.8018,
    'Pycnophylactic': 0.7678,
}
REPRO_TOL = 0.005

METHODS = ['GRANITE', 'Dasymetric', 'Pycnophylactic']
# per-tract min for the pooled validation (matches run_m0_parity.py)
MIN_ADDRESSES_PER_BG_POOLED = 10


# -----------------------------------------------------------------------
# helpers (replicating run_m0_parity.py logic exactly)
# -----------------------------------------------------------------------
def _load_n20():
    if not os.path.exists(N20_LIST_PATH):
        print(f'[persist] HALT: n20 list not found at {N20_LIST_PATH}')
        sys.exit(1)
    with open(N20_LIST_PATH) as f:
        lst = [l.strip() for l in f if l.strip()]
    if len(lst) != 20:
        print(f'[persist] HALT: expected 20 tracts, got {len(lst)}')
        sys.exit(1)
    return lst


def _load_bg_gdf():
    loader = BlockGroupLoader(data_dir=os.path.join(REPO_ROOT, 'data'), verbose=False)
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
        return {'bg_r': float('nan'), 'n_bgs': n_bgs}
    p = merged['predicted_svi'].values.astype(float)
    t = merged['SVI'].values.astype(float)
    valid = np.isfinite(p) & np.isfinite(t)
    if valid.sum() < 2:
        return {'bg_r': float('nan'), 'n_bgs': n_bgs}
    return {'bg_r': float(np.corrcoef(p[valid], t[valid])[0, 1]), 'n_bgs': int(valid.sum())}


def _pooled_bg_r(all_addr_gdfs, all_preds_by_method, validator, bg_gdf):
    """Compute pooled Pearson r across all tracts for each method."""
    combined_gdf = pd.concat(all_addr_gdfs, ignore_index=True)
    if not isinstance(combined_gdf, gpd.GeoDataFrame):
        combined_gdf = gpd.GeoDataFrame(combined_gdf, geometry='geometry', crs='EPSG:4326')
    if combined_gdf.crs is None:
        combined_gdf = combined_gdf.set_crs('EPSG:4326')

    results = {}
    for method in METHODS:
        if not all_preds_by_method[method]:
            results[method] = float('nan')
            continue
        pooled = np.concatenate(all_preds_by_method[method])
        try:
            bg_agg = _aggregate_to_bg(validator, combined_gdf, pooled, MIN_ADDRESSES_PER_BG_POOLED)
            bm = _bg_metrics(bg_agg, bg_gdf)
            results[method] = bm['bg_r']
        except Exception as e:
            print(f'[persist] WARNING: pooled BG r failed for {method}: {e}')
            results[method] = float('nan')
    return results


def _extract_baseline_preds(baseline_result, fips, tract_svi, address_gdf, data, verbose):
    """Extract Dasymetric and Pycnophylactic preds from pipeline result or recompute."""
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
            print(f'[persist]   {fips}: pipeline baseline missing, computing directly')
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


def _config_hash(cfg):
    """Stable hash of the config dict for provenance."""
    s = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()[:16]


# -----------------------------------------------------------------------
# main
# -----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    wall_start = time.time()
    set_random_seed(SEED)

    # preflight
    if not os.path.exists(BG_SVI_PATH):
        print(f'[persist] HALT: {BG_SVI_PATH} missing')
        sys.exit(1)

    n20_fips = _load_n20()
    print(f'[persist] n20 tracts: {len(n20_fips)}')

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f) or {}
    for k in ('data', 'model', 'training', 'processing', 'recovery', 'validation'):
        cfg.setdefault(k, {})

    # frozen m0 config: single-tract SVI mode, GraphSAGE, seed 42
    cfg['data']['target'] = 'svi'
    cfg['data']['neighbor_tracts'] = 0
    cfg['data']['state_fips'] = '47'
    cfg['data']['county_fips'] = '065'
    cfg['data']['target_fips'] = n20_fips[0]
    cfg['model']['architecture'] = 'sage'
    cfg['processing']['skip_importance'] = True
    cfg['processing']['verbose'] = args.verbose
    cfg['processing']['random_seed'] = SEED
    cfg['processing']['enable_caching'] = True
    cfg['training']['apply_post_correction'] = True

    cfg_hash = _config_hash(cfg)
    print(f'[persist] config hash: {cfg_hash}')
    print(f'[persist] epochs: {cfg["training"].get("epochs", cfg.get("training", {}).get("epochs", "?"))}')

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SCRATCH_DIR, exist_ok=True)

    print('[persist] loading BG geodataframe...')
    try:
        bg_gdf = _load_bg_gdf()
        print(f'[persist] {len(bg_gdf)} Hamilton County BGs loaded')
    except Exception as e:
        print(f'[persist] HALT: BG load failed: {e}')
        sys.exit(1)
    validator = BlockGroupValidator(bg_gdf, verbose=False)

    print('[persist] initializing pipeline...')
    pipeline = GRANITEPipeline(cfg, output_dir=SCRATCH_DIR)
    pipeline.verbose = args.verbose

    print('[persist] loading spatial data...')
    try:
        data = pipeline._load_spatial_data()
    except Exception as e:
        print(f'[persist] HALT: spatial data load failed: {e}')
        sys.exit(1)

    # accumulation
    per_tract_rows_by_method = {m: [] for m in METHODS}
    all_addr_gdfs = []
    all_preds_by_method = {m: [] for m in METHODS}  # for pooled BG r
    failures = {m: 0 for m in METHODS}
    MAX_FAILURES = 3

    global_address_idx = 0  # running 0-based index across all tracts (matches n20 matrix concat order)

    for tract_idx, fips in enumerate(n20_fips):
        for m in METHODS:
            if failures[m] > MAX_FAILURES:
                print(f'[persist] HALT: {m} exceeded {MAX_FAILURES} failures')
                sys.exit(1)

        print(f'[persist] [{tract_idx+1}/20] {fips}', flush=True)
        t0 = time.time()

        cfg['data']['target_fips'] = fips
        try:
            result = pipeline._process_single_tract(fips, data)
        except Exception as e:
            msg = str(e)[:200]
            print(f'[persist]   ERROR: {msg}')
            traceback.print_exc()
            for m in METHODS:
                failures[m] += 1
            continue

        if not result.get('success'):
            msg = result.get('error', 'unknown')[:200]
            print(f'[persist]   FAILED: {msg}')
            for m in METHODS:
                failures[m] += 1
            continue

        address_gdf = result['address_gdf']
        granite_preds = result['predictions']['mean'].values.astype(float)
        tract_svi = float(result['tract_svi'])
        n_addr = len(address_gdf)
        runtime = time.time() - t0

        dasy_preds, pycno_preds = _extract_baseline_preds(
            result.get('baseline_comparisons', {}),
            fips, tract_svi, address_gdf, data, args.verbose,
        )

        preds_map = {
            'GRANITE': granite_preds,
            'Dasymetric': dasy_preds,
            'Pycnophylactic': pycno_preds,
        }

        all_addr_gdfs.append(address_gdf.copy())

        for method in METHODS:
            preds = preds_map[method]
            if preds is None:
                failures[method] += 1
                print(f'[persist]   WARNING: {method} predictions None for {fips}')
                continue

            if len(preds) != n_addr:
                print(f'[persist]   WARNING: {method} pred len {len(preds)} != n_addr {n_addr} for {fips}')
                failures[method] += 1
                continue

            # address_idx: 0-based within this tract (matches n20 matrix address_idx column)
            rows = pd.DataFrame({
                'fips': fips,
                'address_idx': np.arange(n_addr, dtype=np.int32),
                'svi_pred': preds.astype(np.float64),
            })
            per_tract_rows_by_method[method].append(rows)
            all_preds_by_method[method].append(preds)

        print(f'[persist]   n_addr={n_addr} tract_svi={tract_svi:.4f} t={runtime:.1f}s')

    # -----------------------------------------------------------------------
    # assemble per-address DataFrames and verify row counts
    # -----------------------------------------------------------------------
    arrays = {}
    for method in METHODS:
        if not per_tract_rows_by_method[method]:
            print(f'[persist] HALT: no rows collected for {method}')
            sys.exit(1)
        df = pd.concat(per_tract_rows_by_method[method], ignore_index=True)
        arrays[method] = df
        print(f'[persist] {method}: {len(df)} rows, {df["fips"].nunique()} tracts')

    # all three must have identical (fips, address_idx) sequences
    ref_keys = arrays['GRANITE'][['fips', 'address_idx']].reset_index(drop=True)
    for method in ['Dasymetric', 'Pycnophylactic']:
        keys = arrays[method][['fips', 'address_idx']].reset_index(drop=True)
        if not ref_keys.equals(keys):
            print(f'[persist] HALT: address index mismatch between GRANITE and {method}')
            sys.exit(1)

    total_rows = sum(len(v) for v in per_tract_rows_by_method['GRANITE'])
    if total_rows != len(arrays['GRANITE']):
        print(f'[persist] HALT: row count inconsistency: {total_rows} vs {len(arrays["GRANITE"])}')
        sys.exit(1)

    # cross-check against n20 feature matrix row count
    N20_EXPECTED_ROWS = 39535
    if len(arrays['GRANITE']) != N20_EXPECTED_ROWS:
        print(f'[persist] WARNING: total rows {len(arrays["GRANITE"])} != expected {N20_EXPECTED_ROWS}')
        # not halting; some tracts may have failed -- reported below

    # -----------------------------------------------------------------------
    # provenance guard: compute pooled BG r for all three methods
    # -----------------------------------------------------------------------
    print('\n[persist] computing pooled BG r (provenance guard)...')
    repro_r = _pooled_bg_r(all_addr_gdfs, all_preds_by_method, validator, bg_gdf)

    print('\n[persist] reproduction check:')
    print(f'  {"method":<16} {"frozen":>8} {"repro":>8} {"delta":>8} {"pass?":>6}')
    print(f'  {"-"*16}  {"-"*7}  {"-"*7}  {"-"*7}  {"-"*5}')
    all_pass = True
    repro_deltas = {}
    for method in METHODS:
        frozen = FROZEN_BG_R[method]
        repro = repro_r[method]
        delta = abs(repro - frozen) if np.isfinite(repro) else float('nan')
        passed = np.isfinite(delta) and delta <= REPRO_TOL
        repro_deltas[method] = {'frozen': frozen, 'repro': repro, 'delta': delta, 'passed': passed}
        flag = 'PASS' if passed else 'FAIL'
        print(f'  {method:<16} {frozen:>8.4f} {repro:>8.4f} {delta:>8.4f}  {flag}')
        if not passed:
            all_pass = False

    if not all_pass:
        print('\n[persist] HALT: provenance guard failed -- arrays not canonical; not persisted')
        failed = [m for m in METHODS if not repro_deltas[m]['passed']]
        for m in failed:
            d = repro_deltas[m]
            print(f'  {m}: repro={d["repro"]:.4f} frozen={d["frozen"]:.4f} delta={d["delta"]:.4f} > tol={REPRO_TOL}')
        sys.exit(1)

    print('\n[persist] all three methods pass provenance guard')

    # -----------------------------------------------------------------------
    # write parquets
    # -----------------------------------------------------------------------
    paths = {}
    name_map = {'GRANITE': 'granite_m0', 'Dasymetric': 'dasymetric', 'Pycnophylactic': 'pycnophylactic'}
    for method, name in name_map.items():
        out_path = os.path.join(OUTPUT_DIR, f'{name}.parquet')
        arrays[method].to_parquet(out_path, index=False)
        paths[method] = out_path
        print(f'[persist] written: {out_path}  ({len(arrays[method])} rows)')

    # -----------------------------------------------------------------------
    # provenance.json
    # -----------------------------------------------------------------------
    wall_elapsed = time.time() - wall_start
    provenance = {
        'source_run': 'persist_per_address_predictions.py PATH B re-run',
        'source_reference': 'experiments/ablation/00_baseline_current/results/aggregate.csv',
        'frozen_bg_r': FROZEN_BG_R,
        'reproduced_bg_r': {m: round(float(repro_r[m]), 6) for m in METHODS},
        'deltas': {m: round(float(repro_deltas[m]['delta']), 6) for m in METHODS},
        'all_within_tol': all_pass,
        'tol': REPRO_TOL,
        'config_hash': cfg_hash,
        'config_key_settings': {
            'architecture': cfg['model']['architecture'],
            'seed': SEED,
            'epochs': cfg['training'].get('epochs'),
            'apply_post_correction': cfg['training'].get('apply_post_correction'),
            'neighbor_tracts': cfg['data']['neighbor_tracts'],
            'target': cfg['data']['target'],
        },
        'n20_tracts': n20_fips,
        'total_addresses': len(arrays['GRANITE']),
        'n20_expected_rows': N20_EXPECTED_ROWS,
        'index_alignment': 'fips + address_idx (0-based within tract); matches n20_feature_matrix.csv row order',
        'parquet_paths': {m: os.path.relpath(paths[m], REPO_ROOT) for m in METHODS},
        'wall_time_s': round(wall_elapsed, 1),
    }

    prov_path = os.path.join(OUTPUT_DIR, 'provenance.json')
    with open(prov_path, 'w') as f:
        json.dump(provenance, f, indent=2)
    print(f'[persist] written: {prov_path}')

    # -----------------------------------------------------------------------
    # final report
    # -----------------------------------------------------------------------
    print(f'\n[persist] === DONE in {wall_elapsed:.0f}s ===')
    print(f'  arrays: {len(arrays["GRANITE"])} rows x 3 methods')
    print(f'  index alignment: confirmed (fips + address_idx match across all three)')
    print(f'  provenance guard: PASS (all deltas <= {REPRO_TOL})')
    print(f'  output dir: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
