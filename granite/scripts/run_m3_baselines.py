"""
M3 baseline driver: per-tract ridge and GBM baselines across 3 targets on n20 tracts.
Reads the n20 tract list from M2 output so the tract sets are guaranteed identical.
Writes five summary files and a lift table comparing GRANITE (M2) against baselines.

Usage:
    python granite/scripts/run_m3_baselines.py [--config CONFIG] [--verbose]
                                               [--save-predictions]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from granite.evaluation.recovery_baselines import run_baselines

TARGETS = [
    'log_appvalue',
    'employment_walk_effective_access',
    'nlcd_impervious_pct',
]
ARCHITECTURES = ['sage', 'gcn_gat']
SEED = 42
REFERENCE_FIPS = '47065000600'

M2_SUMMARY_DIR = 'output/m2_n20_recovery/summary'
M2_PIVOT_PATH = os.path.join(M2_SUMMARY_DIR, 'pivot_pearson_r.csv')
N20_LIST_PATH = os.path.join(M2_SUMMARY_DIR, 'n20_tract_list.txt')

SWEEP_BASE = 'output/m3_n20_baselines'
SUMMARY_DIR = os.path.join(SWEEP_BASE, 'summary')


def load_config(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault('data', {})
    cfg.setdefault('model', {})
    cfg.setdefault('training', {})
    cfg.setdefault('processing', {})
    cfg.setdefault('recovery', {})
    return cfg


def load_n20_tracts(path):
    with open(path, 'r') as f:
        fips_list = [line.strip() for line in f if line.strip()]
    if len(fips_list) != 20:
        print(f'[m3] expected 20 tracts in {path}, got {len(fips_list)}')
        sys.exit(1)
    return fips_list


def aggregate_results(all_rows_by_target):
    """
    Build per_tract_metrics.csv (long format, 60 rows) and
    baseline_summary_stats.csv from per-tract metric dicts.
    """
    os.makedirs(SUMMARY_DIR, exist_ok=True)

    long_rows = []
    for target, rows in all_rows_by_target.items():
        for r in rows:
            long_rows.append({
                'target': target,
                'tract_fips': r['tract_fips'],
                'n_addresses': r['n_addresses'],
                'ridge_pearson_r': r['ridge_pearson_r'],
                'gbm_pearson_r': r['gbm_pearson_r'],
                'ridge_rmse': r['ridge_rmse_native'],
                'gbm_rmse': r['gbm_rmse_native'],
            })

    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(os.path.join(SUMMARY_DIR, 'per_tract_metrics.csv'), index=False)

    stat_rows = []
    for target in TARGETS:
        df = long_df[long_df['target'] == target]
        r_ridge = pd.to_numeric(df['ridge_pearson_r'], errors='coerce').dropna()
        r_gbm = pd.to_numeric(df['gbm_pearson_r'], errors='coerce').dropna()
        stat_rows.append({
            'target': target,
            'median_ridge_r': float(r_ridge.median()) if len(r_ridge) > 0 else float('nan'),
            'median_gbm_r': float(r_gbm.median()) if len(r_gbm) > 0 else float('nan'),
            'mean_ridge_r': float(r_ridge.mean()) if len(r_ridge) > 0 else float('nan'),
            'mean_gbm_r': float(r_gbm.mean()) if len(r_gbm) > 0 else float('nan'),
            'n_tracts_ridge_r_gt_05': int((r_ridge > 0.5).sum()),
            'n_tracts_gbm_r_gt_05': int((r_gbm > 0.5).sum()),
        })

    stats_df = pd.DataFrame(stat_rows)
    stats_df.to_csv(os.path.join(SUMMARY_DIR, 'baseline_summary_stats.csv'), index=False)
    return long_df, stats_df


def build_lift_tables(long_df, m2_pivot_path):
    """
    Join M2 pivot_pearson_r against M3 per-tract metrics to produce
    lift_table.csv (120 rows) and lift_summary.csv (6 rows).
    """
    if not os.path.exists(m2_pivot_path):
        print(f'[m3] M2 pivot not found at {m2_pivot_path}; skipping lift tables')
        return None, None

    m2_pivot = pd.read_csv(m2_pivot_path)
    m2_pivot['tract_fips'] = m2_pivot['tract_fips'].astype(str).str.strip()

    # melt M2 pivot to long format
    m2_long = m2_pivot.melt(id_vars='tract_fips', var_name='cell', value_name='granite_r')
    # parse "target__architecture" column names
    split = m2_long['cell'].str.rsplit('__', n=1, expand=True)
    m2_long['target'] = split[0]
    m2_long['architecture'] = split[1]
    m2_long = m2_long.drop(columns=['cell'])

    # M3 lookup: (target, tract_fips) -> ridge_r, gbm_r
    m3_lookup = long_df[['target', 'tract_fips', 'ridge_pearson_r', 'gbm_pearson_r']].copy()
    m3_lookup.columns = ['target', 'tract_fips', 'ridge_r', 'gbm_r']

    lift = m2_long.merge(m3_lookup, on=['target', 'tract_fips'], how='left')
    lift['lift_vs_ridge'] = pd.to_numeric(lift['granite_r'], errors='coerce') - pd.to_numeric(lift['ridge_r'], errors='coerce')
    lift['lift_vs_gbm'] = pd.to_numeric(lift['granite_r'], errors='coerce') - pd.to_numeric(lift['gbm_r'], errors='coerce')

    # reorder columns
    lift = lift[['target', 'architecture', 'tract_fips', 'granite_r', 'ridge_r', 'gbm_r', 'lift_vs_ridge', 'lift_vs_gbm']]
    lift.to_csv(os.path.join(SUMMARY_DIR, 'lift_table.csv'), index=False)

    # lift summary: 6 rows (3 targets x 2 architectures)
    summary_rows = []
    for target in TARGETS:
        for arch in ARCHITECTURES:
            sub = lift[(lift['target'] == target) & (lift['architecture'] == arch)]
            granite_r = pd.to_numeric(sub['granite_r'], errors='coerce').dropna()
            ridge_r = pd.to_numeric(sub['ridge_r'], errors='coerce').dropna()
            gbm_r = pd.to_numeric(sub['gbm_r'], errors='coerce').dropna()
            lift_ridge = pd.to_numeric(sub['lift_vs_ridge'], errors='coerce').dropna()
            lift_gbm = pd.to_numeric(sub['lift_vs_gbm'], errors='coerce').dropna()

            # beats counts: granite_r > baseline_r per tract
            beats_ridge = int(
                (pd.to_numeric(sub['granite_r'], errors='coerce') >
                 pd.to_numeric(sub['ridge_r'], errors='coerce')).sum()
            )
            beats_gbm = int(
                (pd.to_numeric(sub['granite_r'], errors='coerce') >
                 pd.to_numeric(sub['gbm_r'], errors='coerce')).sum()
            )

            summary_rows.append({
                'target': target,
                'architecture': arch,
                'median_granite_r': float(granite_r.median()) if len(granite_r) > 0 else float('nan'),
                'median_ridge_r': float(ridge_r.median()) if len(ridge_r) > 0 else float('nan'),
                'median_gbm_r': float(gbm_r.median()) if len(gbm_r) > 0 else float('nan'),
                'median_lift_vs_ridge': float(lift_ridge.median()) if len(lift_ridge) > 0 else float('nan'),
                'median_lift_vs_gbm': float(lift_gbm.median()) if len(lift_gbm) > 0 else float('nan'),
                'n_tracts_granite_beats_ridge': beats_ridge,
                'n_tracts_granite_beats_gbm': beats_gbm,
            })

    lift_summary = pd.DataFrame(summary_rows)
    lift_summary.to_csv(os.path.join(SUMMARY_DIR, 'lift_summary.csv'), index=False)
    return lift, lift_summary


def write_lift_brief(stats_df, lift_summary, lift_table, long_df):
    """
    Write lift_brief.md addressing the four required points:
    1. baseline ceiling per target
    2. does GRANITE clear ceiling on median, mean, tract count
    3. special attention to employment_walk_effective_access
    4. anomalies, skipped tracts, near-zero targets, ridge vs gbm divergence
    """
    os.makedirs(SUMMARY_DIR, exist_ok=True)

    paras = []

    # paragraph 1: baseline ceilings per target
    lines = []
    for _, row in stats_df.iterrows():
        tgt = row['target']
        med_ridge = row['median_ridge_r']
        med_gbm = row['median_gbm_r']
        ceiling = max(
            med_ridge if np.isfinite(med_ridge) else -np.inf,
            med_gbm if np.isfinite(med_gbm) else -np.inf,
        )
        if ceiling == -np.inf:
            ceiling = float('nan')
        ceiling_model = 'none'
        if np.isfinite(med_ridge) and np.isfinite(med_gbm):
            ceiling_model = 'ridge' if med_ridge >= med_gbm else 'gbm'
        elif np.isfinite(med_ridge):
            ceiling_model = 'ridge'
        elif np.isfinite(med_gbm):
            ceiling_model = 'gbm'
        lines.append(
            f'For {tgt}, the baseline ceiling is median r = {ceiling:.3f} ({ceiling_model}); '
            f'ridge median r = {med_ridge:.3f}, gbm median r = {med_gbm:.3f}.'
        )
    paras.append(' '.join(lines))

    # paragraph 2: does GRANITE clear ceiling per (target, architecture)
    if lift_summary is not None:
        clears = []
        does_not_clear = []
        for _, row in lift_summary.iterrows():
            tgt = row['target']
            arch = row['architecture']
            g_med = row['median_granite_r']
            r_med = row['median_ridge_r']
            gbm_med = row['median_gbm_r']
            beats_ridge = row['n_tracts_granite_beats_ridge']
            beats_gbm = row['n_tracts_granite_beats_gbm']

            # ceiling is max of ridge and gbm medians
            valid_baselines = [x for x in [r_med, gbm_med] if np.isfinite(x)]
            ceiling = max(valid_baselines) if valid_baselines else float('nan')
            clears_ceiling = np.isfinite(g_med) and np.isfinite(ceiling) and g_med > ceiling
            cell_str = (
                f'{tgt}/{arch}: GRANITE median r = {g_med:.3f}, '
                f'ceiling = {ceiling:.3f}, '
                f'beats ridge in {beats_ridge}/20 tracts, '
                f'beats gbm in {beats_gbm}/20 tracts'
            )
            if clears_ceiling:
                clears.append(cell_str + ' (clears ceiling on median)')
            else:
                does_not_clear.append(cell_str + ' (does not clear ceiling on median)')
        all_cells = clears + does_not_clear
        p2 = 'GRANITE vs baseline ceiling by (target, architecture): ' + '; '.join(all_cells) + '.'
    else:
        p2 = 'M2 pivot not available; lift comparison could not be computed.'
    paras.append(p2)

    # paragraph 3: special attention to employment_walk_effective_access
    tgt_emp = 'employment_walk_effective_access'
    emp_stats = stats_df[stats_df['target'] == tgt_emp]
    emp_ridge_med = float(emp_stats['median_ridge_r'].iloc[0]) if len(emp_stats) > 0 else float('nan')
    emp_gbm_med = float(emp_stats['median_gbm_r'].iloc[0]) if len(emp_stats) > 0 else float('nan')

    if lift_summary is not None:
        emp_lift = lift_summary[lift_summary['target'] == tgt_emp]
        emp_parts = []
        for _, row in emp_lift.iterrows():
            arch = row['architecture']
            g_med = row['median_granite_r']
            emp_parts.append(f'{arch}: GRANITE median r = {g_med:.3f}')
        granite_emp_str = '; '.join(emp_parts)
    else:
        granite_emp_str = 'not available'

    p3 = (
        f'For employment_walk_effective_access, ridge alone achieves median r = {emp_ridge_med:.3f} '
        f'and gbm median r = {emp_gbm_med:.3f} without any graph or constraint. '
        f'GRANITE results for this target: {granite_emp_str}. '
        f'If ridge or gbm median r approaches or exceeds the GRANITE median r, the M2 result '
        f'for this target is consistent with feature-correlation recovery rather than '
        f'GRANITE-attributable graph or constraint recovery.'
    )
    paras.append(p3)

    # paragraph 4: anomalies
    anomaly_parts = []

    # tracts with ridge r > 0.7 on any target
    high_r_tracts = []
    for target in TARGETS:
        df_t = long_df[long_df['target'] == target]
        hi = df_t[pd.to_numeric(df_t['ridge_pearson_r'], errors='coerce') > 0.7]
        for _, r in hi.iterrows():
            high_r_tracts.append(f'{r["tract_fips"]} ({target}, ridge r = {r["ridge_pearson_r"]:.3f})')
    if high_r_tracts:
        anomaly_parts.append(
            'Tracts with ridge r > 0.70 (feature-correlation-alone cases): ' + ', '.join(high_r_tracts) + '.'
        )

    # tracts where gbm r is null (skipped)
    gbm_null = long_df[long_df['gbm_pearson_r'].isnull()]
    if len(gbm_null) > 0:
        skipped_str = ', '.join(
            f'{r["tract_fips"]} ({r["target"]}, n = {r["n_addresses"]})'
            for _, r in gbm_null.iterrows()
        )
        anomaly_parts.append(
            f'GBM skipped in {len(gbm_null)} (target, tract) cells due to n < 50: ' + skipped_str + '.'
        )

    # near-zero targets: tracts where both ridge r and gbm r are near zero (|r| < 0.05)
    near_zero = long_df[
        (pd.to_numeric(long_df['ridge_pearson_r'], errors='coerce').abs() < 0.05) &
        (pd.to_numeric(long_df['gbm_pearson_r'], errors='coerce').abs() < 0.05)
    ]
    if len(near_zero) > 0:
        nz_str = ', '.join(
            f'{r["tract_fips"]} ({r["target"]})'
            for _, r in near_zero.iterrows()
        )
        anomaly_parts.append(
            f'{len(near_zero)} (target, tract) cells where both ridge and gbm r < 0.05: ' + nz_str + '.'
        )

    # large ridge vs gbm divergence (|ridge_r - gbm_r| > 0.3)
    df_num = long_df.copy()
    df_num['ridge_pearson_r_n'] = pd.to_numeric(df_num['ridge_pearson_r'], errors='coerce')
    df_num['gbm_pearson_r_n'] = pd.to_numeric(df_num['gbm_pearson_r'], errors='coerce')
    df_num['r_divergence'] = (df_num['ridge_pearson_r_n'] - df_num['gbm_pearson_r_n']).abs()
    div_cases = df_num[df_num['r_divergence'] > 0.3].dropna(subset=['r_divergence'])
    if len(div_cases) > 0:
        div_str = ', '.join(
            f'{r["tract_fips"]} ({r["target"]}, ridge={r["ridge_pearson_r_n"]:.3f}, gbm={r["gbm_pearson_r_n"]:.3f})'
            for _, r in div_cases.iterrows()
        )
        anomaly_parts.append(
            f'{len(div_cases)} (target, tract) cells with |ridge_r - gbm_r| > 0.30: ' + div_str + '.'
        )

    if anomaly_parts:
        p4 = 'Anomalies and flags: ' + ' '.join(anomaly_parts)
    else:
        p4 = (
            'No anomalies detected. All 60 (target, tract) cells produced finite ridge r values, '
            'no ridge r > 0.70 cases, no large ridge-vs-gbm divergences.'
        )
    paras.append(p4)

    brief_path = os.path.join(SUMMARY_DIR, 'lift_brief.md')
    with open(brief_path, 'w') as f:
        f.write('# M3 Baseline Lift Brief\n\n')
        for i, para in enumerate(paras):
            f.write(para + '\n')
            if i < len(paras) - 1:
                f.write('\n')

    return brief_path


def main():
    parser = argparse.ArgumentParser(description='M3 baseline driver')
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--save-predictions', action='store_true',
                        help='write per-address prediction files (default: off)')
    args = parser.parse_args()

    wall_start = time.time()

    config = load_config(args.config)

    # read n20 tract list from M2 output
    if not os.path.exists(N20_LIST_PATH):
        print(f'[m3] n20 tract list not found at {N20_LIST_PATH}')
        print('[m3] run M2 sweep first to generate that file')
        sys.exit(1)
    n20_fips = load_n20_tracts(N20_LIST_PATH)
    print(f'[m3] n20 tracts loaded: {len(n20_fips)} from {N20_LIST_PATH}')

    # build shared config (no arch, no epochs needed for baselines)
    config['processing']['random_seed'] = SEED
    config['processing']['enable_caching'] = True
    config['data']['target_fips'] = REFERENCE_FIPS
    config['data']['state_fips'] = REFERENCE_FIPS[:2]
    config['data']['county_fips'] = REFERENCE_FIPS[2:5]
    config['recovery']['standardize_target'] = True

    all_rows_by_target = {}

    for target in TARGETS:
        run_dir = os.path.join(SWEEP_BASE, target)
        print(f'\n[m3] target={target}')
        print(f'[m3]   output: {run_dir}')

        t0 = time.time()
        try:
            rows = run_baselines(
                config=config,
                target_feature=target,
                tract_list=n20_fips,
                output_dir=run_dir,
                save_predictions=args.save_predictions,
                verbose=args.verbose,
                seed=SEED,
            )
        except Exception as e:
            import traceback as tb
            print(f'[m3]   FAILED: {e}')
            if args.verbose:
                tb.print_exc()
            # record empty rows so downstream aggregation doesn't break
            rows = []

        elapsed = time.time() - t0
        print(f'[m3]   done in {elapsed:.1f}s, {len(rows)} tracts')
        all_rows_by_target[target] = rows

    # aggregation
    print('\n[m3] aggregating results...')
    long_df, stats_df = aggregate_results(all_rows_by_target)

    lift_table, lift_summary = build_lift_tables(long_df, M2_PIVOT_PATH)

    brief_path = write_lift_brief(stats_df, lift_summary, lift_table, long_df)

    wall_elapsed = time.time() - wall_start
    print(f'\n[m3] sweep complete in {wall_elapsed:.1f}s ({wall_elapsed/60:.1f} min)')

    print('\n--- baseline_summary_stats.csv ---')
    print(stats_df.to_string(index=False))

    if lift_summary is not None:
        print('\n--- lift_summary.csv ---')
        print(lift_summary.to_string(index=False))

    print(f'\n--- lift_brief.md ({brief_path}) ---')
    with open(brief_path) as f:
        print(f.read())

    # flag high-ridge-r tracts
    for target in TARGETS:
        rows = all_rows_by_target.get(target, [])
        for r in rows:
            ridge_r = r.get('ridge_pearson_r')
            if ridge_r is not None and np.isfinite(float(ridge_r)) and float(ridge_r) > 0.7:
                print(f'[m3] ridge r > 0.7: {r["tract_fips"]} target={target} ridge_r={ridge_r:.4f}')

    # flag skipped GBM tracts
    for target in TARGETS:
        rows = all_rows_by_target.get(target, [])
        for r in rows:
            if r.get('gbm_pearson_r') is None:
                print(f'[m3] GBM skipped: {r["tract_fips"]} target={target} n={r["n_addresses"]}')


if __name__ == '__main__':
    main()
