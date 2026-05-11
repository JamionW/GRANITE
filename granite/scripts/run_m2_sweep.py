"""
M2 sweep driver: runs recovery harness across 3 targets x 2 architectures
on the n20 stratified tract set. Aggregates results into pivot tables and
a decision brief.

Usage:
    python granite/scripts/run_m2_sweep.py [--config CONFIG] [--verbose]
                                            [--skip-preflight]
"""
import argparse
import json
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from granite.disaggregation.pipeline import GRANITEPipeline
from granite.models.gnn import (
    AccessibilitySVIGNN,
    GraphSAGEAccessibilitySVIGNN,
    MultiTractGNNTrainer,
    normalize_accessibility_features,
    set_random_seed,
)
from granite.disaggregation.recovery_harness import (
    _get_git_sha,
    _compute_per_tract_metrics,
    _write_outputs,
    run_recovery,
)

TARGETS = [
    'log_appvalue',
    'employment_walk_effective_access',
    'nlcd_impervious_pct',
]
ARCHITECTURES = ['sage', 'gcn_gat']
SEED = 42
EPOCHS = 100
REFERENCE_FIPS = '47065000600'
INVENTORY_PATH = 'tract_inventory.csv'
SWEEP_BASE = 'output/m2_n20_recovery'


def load_config(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault('data', {})
    cfg.setdefault('model', {})
    cfg.setdefault('training', {})
    cfg.setdefault('processing', {})
    cfg.setdefault('recovery', {})
    return cfg


def load_n20_tracts(inventory_path):
    """
    Load n20 FIPS list from tract_inventory.csv.

    If a Status column is present, filter on '✓' and verify exactly 20 rows.
    If absent, use all rows and verify exactly 20.
    """
    df = pd.read_csv(inventory_path)
    cols_upper = {c: c.upper() for c in df.columns}
    df = df.rename(columns=cols_upper)

    fips_col = next((c for c in df.columns if 'FIPS' in c), None)
    if fips_col is None:
        # fall back to first column
        fips_col = df.columns[0]

    if 'STATUS' in df.columns:
        unique_vals = df['STATUS'].unique().tolist()
        subset = df[df['STATUS'] == '✓']
        if len(subset) != 20:
            print(f'[m2] Status filtering yielded {len(subset)} rows, not 20.')
            print(f'[m2] Unique Status values: {unique_vals}')
            sys.exit(1)
        fips_list = sorted(subset[fips_col].astype(str).str.strip().tolist())
    else:
        if len(df) != 20:
            print(f'[m2] tract_inventory.csv has {len(df)} rows (expected 20) and no Status column.')
            print(f'[m2] Columns: {list(df.columns)}')
            sys.exit(1)
        fips_list = sorted(df[fips_col].astype(str).str.strip().tolist())

    return fips_list


def _run_recovery_explicit_tracts(
    config,
    target_feature,
    tract_list,
    output_dir,
    verbose=False,
):
    """
    Recovery run over an explicit list of tracts. Replicates recovery_harness
    logic but uses tract_list directly instead of get_neighboring_tracts().

    Returns dict with keys: success, output_dir, overall_constraint_error,
    per_tract_errors, epochs_trained, error (on failure).
    """
    seed = config.get('processing', {}).get('random_seed', SEED)
    set_random_seed(seed)

    arch = config.get('model', {}).get('architecture', 'gcn_gat')
    target_fips = config['data']['target_fips']

    pipeline = GRANITEPipeline(config, output_dir=output_dir)
    pipeline.verbose = verbose

    try:
        data = pipeline._load_spatial_data()
    except Exception as e:
        return {'success': False, 'error': f'data loading failed: {e}'}

    # load addresses for each tract in the explicit list
    all_addresses = []
    skipped = []
    for fips in tract_list:
        fips = str(fips).strip()
        if len(data['tracts'][data['tracts']['FIPS'] == fips]) == 0:
            skipped.append((fips, 'not in tracts GDF'))
            continue
        addrs = pipeline.data_loader.get_addresses_for_tract(fips)
        if len(addrs) == 0:
            skipped.append((fips, 'no addresses'))
            continue
        addrs['tract_fips'] = fips
        all_addresses.append(addrs)

    if skipped and verbose:
        for fips, reason in skipped:
            print(f'[m2] skipped {fips}: {reason}')

    if not all_addresses:
        return {'success': False, 'error': 'no addresses loaded for any tract'}

    tract_addresses = pd.concat(all_addresses, ignore_index=True)
    valid_fips_list = [str(f).strip() for f in tract_list
                       if str(f).strip() not in [s[0] for s in skipped]]

    if verbose:
        print(f'[m2] {len(tract_addresses)} addresses across {len(valid_fips_list)} tracts')

    # compute per-tract so each tract's result maps to an existing (or new) single-tract
    # cache key rather than the never-cached 20-tract combined hash.  travel times are
    # per-address quantities independent of batch composition, so stacking is valid.
    # on the first sweep run each tract's cache entry is written; all five subsequent
    # runs hit those entries and skip osrm entirely.
    per_tract_arrays = []
    for fips in valid_fips_list:
        mask = (tract_addresses['tract_fips'] == fips).values
        tract_slice = tract_addresses[mask].copy().reset_index(drop=True)
        try:
            feats = pipeline._compute_accessibility_features(tract_slice, data)
        except Exception as e:
            return {
                'success': False,
                'error': f'feature computation raised for {fips}: {e}\n{traceback.format_exc()}',
            }
        if feats is None:
            return {'success': False, 'error': f'feature computation returned None for {fips}'}
        per_tract_arrays.append(feats)

    try:
        accessibility_features = np.vstack(per_tract_arrays)
    except Exception as e:
        return {'success': False, 'error': f'feature stacking failed: {e}'}

    if accessibility_features is None or accessibility_features.shape[0] == 0:
        return {'success': False, 'error': 'feature computation returned empty array'}

    feature_names = pipeline._generate_feature_names(accessibility_features.shape[1])

    if target_feature not in feature_names:
        return {
            'success': False,
            'error': (
                f"target_feature '{target_feature}' not found in feature matrix "
                f"({len(feature_names)} features). Available: {feature_names}"
            ),
        }

    feat_idx = feature_names.index(target_feature)
    raw_target_values = accessibility_features[:, feat_idx].copy().astype(float)

    standardize = config.get('recovery', {}).get('standardize_target', True)
    if standardize:
        tgt_mean = float(np.nanmean(raw_target_values))
        tgt_std = float(np.nanstd(raw_target_values))
        if tgt_std < 1e-8:
            tgt_std = 1.0
        target_values_std = (raw_target_values - tgt_mean) / tgt_std
    else:
        tgt_mean = 0.0
        tgt_std = 1.0
        target_values_std = raw_target_values.copy()

    if verbose:
        print(f'[m2] target "{target_feature}": mean={tgt_mean:.4f}, std={tgt_std:.4f}')

    tract_target_means_std = {}
    for fips in valid_fips_list:
        mask = (tract_addresses['tract_fips'] == fips).values
        if mask.sum() > 0:
            tract_target_means_std[fips] = float(np.nanmean(target_values_std[mask]))

    # drop target feature from matrix
    feat_matrix = np.delete(accessibility_features, feat_idx, axis=1)
    feat_names_reduced = [n for i, n in enumerate(feature_names) if i != feat_idx]

    if verbose:
        print(f'[m2] feature matrix: {accessibility_features.shape[1]} -> {feat_matrix.shape[1]}')

    normalized_features, _ = normalize_accessibility_features(feat_matrix)
    normalized_features = pipeline._apply_feature_mode(
        normalized_features,
        tract_addresses,
        mode=config.get('feature_mode', 'full'),
        seed=seed,
    )

    context_features = pipeline.data_loader.create_context_features_for_addresses(
        addresses=tract_addresses,
        svi_data=data['svi'],
    )
    normalized_context, _ = pipeline.data_loader.normalize_context_features(context_features)

    graph_data = pipeline.data_loader.create_spatial_accessibility_graph(
        addresses=tract_addresses,
        accessibility_features=normalized_features,
        context_features=normalized_context,
        state_fips=target_fips[:2],
        county_fips=target_fips[2:5],
    )

    if verbose:
        print(f'[m2] graph: {graph_data.x.shape[0]} nodes, '
              f'{graph_data.edge_index.shape[1]} edges, '
              f'{graph_data.x.shape[1]} node features')

    normed2, _ = normalize_accessibility_features(graph_data.x.numpy())
    graph_data.x = torch.FloatTensor(normed2)

    has_context = hasattr(graph_data, 'context') and graph_data.context is not None
    context_dim = graph_data.context.shape[1] if has_context else 5
    use_gating = config.get('model', {}).get('use_context_gating', True)

    ModelClass = GraphSAGEAccessibilitySVIGNN if arch == 'sage' else AccessibilitySVIGNN
    model = ModelClass(
        accessibility_features_dim=graph_data.x.shape[1],
        context_features_dim=context_dim,
        hidden_dim=config.get('model', {}).get('hidden_dim', 64),
        dropout=config.get('model', {}).get('dropout', 0.3),
        seed=seed,
        use_context_gating=use_gating and has_context,
    )

    tract_masks = {
        fips: (tract_addresses['tract_fips'] == fips).values
        for fips in tract_target_means_std
    }

    training_config = {
        **config.get('training', {}),
        'use_multitask': True,
        'bg_constraint_weight': 0.0,
        'ordering_weight': 0.0,
    }
    trainer = MultiTractGNNTrainer(model, config=training_config, seed=seed)

    epochs = config.get('training', {}).get('epochs',
             config.get('model', {}).get('epochs', EPOCHS))

    if verbose:
        print(f'[m2] training {arch} for {epochs} epochs '
              f'with {len(tract_target_means_std)} tract constraints')

    try:
        training_result = trainer.train(
            graph_data=graph_data,
            tract_svis=tract_target_means_std,
            tract_masks=tract_masks,
            epochs=epochs,
            verbose=verbose,
            feature_names=feat_names_reduced,
            block_group_targets=None,
            block_group_masks=None,
            ordering_values=None,
        )
    except Exception as e:
        return {'success': False, 'error': f'training raised: {e}\n{traceback.format_exc()}'}

    if not training_result.get('success', False):
        return {'success': False, 'error': training_result.get('error', 'training failed')}

    raw_preds = training_result['final_predictions']
    preds_native = raw_preds * tgt_std + tgt_mean

    _write_outputs(
        output_dir=output_dir,
        tract_addresses=tract_addresses,
        target_feature=target_feature,
        arch=arch,
        seed=seed,
        preds_std=raw_preds,
        preds_native=preds_native,
        true_values=raw_target_values,
        tract_target_means_std=tract_target_means_std,
        tgt_mean=tgt_mean,
        tgt_std=tgt_std,
        config=config,
        n_features_after_drop=feat_matrix.shape[1],
        standardize=standardize,
    )

    return {
        'success': True,
        'output_dir': output_dir,
        'overall_constraint_error': training_result['overall_constraint_error'],
        'per_tract_errors': training_result['per_tract_errors'],
        'epochs_trained': training_result['epochs_trained'],
        'skipped_tracts': skipped,
    }


def run_preflight_smoke(config, verbose=False):
    """Smoke-check M1 on single tract 47065000600, log_appvalue, sage, seed 42, 50 epochs."""
    print('[m2] pre-flight: smoke-check M1 (log_appvalue / sage / 47065000600 / 50 epochs)')
    smoke_cfg = {
        **config,
        'data': {
            **config.get('data', {}),
            'target_fips': REFERENCE_FIPS,
            'neighbor_tracts': 0,
        },
        'model': {
            **config.get('model', {}),
            'architecture': 'sage',
        },
        'training': {
            **config.get('training', {}),
            'epochs': 50,
        },
        'processing': {
            **config.get('processing', {}),
            'random_seed': 42,
        },
        'recovery': {
            **config.get('recovery', {}),
            'standardize_target': True,
        },
    }
    smoke_out = os.path.join(SWEEP_BASE, '_preflight_smoke')
    result = run_recovery(
        config=smoke_cfg,
        target_feature='log_appvalue',
        output_dir=smoke_out,
        verbose=verbose,
    )
    if not result.get('success'):
        print(f'[m2] pre-flight FAILED: {result.get("error")}')
        return False
    # read per_tract_metrics.csv and print the reference row
    metrics_path = os.path.join(smoke_out, 'per_tract_metrics.csv')
    if os.path.exists(metrics_path):
        df = pd.read_csv(metrics_path)
        row = df[df['tract_fips'] == REFERENCE_FIPS]
        if not row.empty:
            r = row.iloc[0]
            print(f'[m2] pre-flight OK: tract={REFERENCE_FIPS} '
                  f'pearson_r={r["pearson_r"]:.4f} '
                  f'rmse={r["rmse"]:.4f} '
                  f'constraint_err={r["constraint_error_pct"]:.2f}%')
        else:
            print(f'[m2] pre-flight OK: no row for {REFERENCE_FIPS} in smoke output')
    else:
        print('[m2] pre-flight OK (no metrics file found)')
    return True


def aggregate_results(sweep_base, n20_fips):
    """
    Read per_tract_metrics.csv from each run directory and aggregate into
    pivot tables and summary stats.
    """
    summary_dir = os.path.join(sweep_base, 'summary')
    os.makedirs(summary_dir, exist_ok=True)

    col_keys = []
    run_frames = {}
    run_meta = {}

    for target in TARGETS:
        for arch in ARCHITECTURES:
            col_key = f'{target}__{arch}'
            run_dir = os.path.join(sweep_base, f'{target}_{arch}')
            metrics_path = os.path.join(run_dir, 'per_tract_metrics.csv')
            if os.path.exists(metrics_path):
                df = pd.read_csv(metrics_path)
                df['tract_fips'] = df['tract_fips'].astype(str).str.strip()
                run_frames[col_key] = df
            else:
                run_frames[col_key] = None
            col_keys.append(col_key)

            meta_path = os.path.join(run_dir, 'run_meta.json')
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    run_meta[col_key] = json.load(f)

    # n20_tract_list.txt
    with open(os.path.join(summary_dir, 'n20_tract_list.txt'), 'w') as f:
        for fips in sorted(n20_fips):
            f.write(fips + '\n')

    # pivot_pearson_r.csv and pivot_rmse.csv: 20 rows x 6 columns
    pivot_r = pd.DataFrame({'tract_fips': sorted(n20_fips)})
    pivot_rmse = pd.DataFrame({'tract_fips': sorted(n20_fips)})

    for col_key in col_keys:
        df = run_frames.get(col_key)
        if df is not None:
            r_map = dict(zip(df['tract_fips'], df['pearson_r']))
            rmse_map = dict(zip(df['tract_fips'], df['rmse']))
        else:
            r_map = {}
            rmse_map = {}
        pivot_r[col_key] = pivot_r['tract_fips'].map(r_map)
        pivot_rmse[col_key] = pivot_rmse['tract_fips'].map(rmse_map)

    pivot_r.to_csv(os.path.join(summary_dir, 'pivot_pearson_r.csv'), index=False)
    pivot_rmse.to_csv(os.path.join(summary_dir, 'pivot_rmse.csv'), index=False)

    # summary_stats.csv
    stat_rows = []
    for target in TARGETS:
        for arch in ARCHITECTURES:
            col_key = f'{target}__{arch}'
            df = run_frames.get(col_key)
            if df is None:
                stat_rows.append({
                    'target': target,
                    'architecture': arch,
                    'median_r': float('nan'),
                    'mean_r': float('nan'),
                    'std_r': float('nan'),
                    'n_tracts_r_gt_03': float('nan'),
                    'n_tracts_r_gt_05': float('nan'),
                    'median_constraint_error_pct': float('nan'),
                })
                continue
            r_vals = df['pearson_r'].dropna()
            ce_vals = df['constraint_error_pct'].dropna()
            stat_rows.append({
                'target': target,
                'architecture': arch,
                'median_r': float(r_vals.median()) if len(r_vals) > 0 else float('nan'),
                'mean_r': float(r_vals.mean()) if len(r_vals) > 0 else float('nan'),
                'std_r': float(r_vals.std()) if len(r_vals) > 0 else float('nan'),
                'n_tracts_r_gt_03': int((r_vals > 0.3).sum()),
                'n_tracts_r_gt_05': int((r_vals > 0.5).sum()),
                'median_constraint_error_pct': float(ce_vals.median()) if len(ce_vals) > 0 else float('nan'),
            })

    stats_df = pd.DataFrame(stat_rows)
    stats_df.to_csv(os.path.join(summary_dir, 'summary_stats.csv'), index=False)

    return stats_df, pivot_r, pivot_rmse, run_frames, run_meta


def write_decision_brief(summary_dir, stats_df, pivot_r, run_frames, run_results):
    """
    Write a two-to-four paragraph prose decision brief based on the sweep results.
    """
    BRANCH1_THRESHOLD = 0.5

    # identify best cell by median_r
    valid = stats_df.dropna(subset=['median_r'])
    if len(valid) == 0:
        best_row = None
        best_cell = 'none (all runs failed)'
        best_median_r = float('nan')
    else:
        best_row = valid.loc[valid['median_r'].idxmax()]
        best_cell = f"{best_row['target']} / {best_row['architecture']}"
        best_median_r = best_row['median_r']

    # branch 1 check
    branch1_cells = valid[valid['median_r'] > BRANCH1_THRESHOLD]

    # architecture comparison across targets
    arch_medians = {}
    for arch in ARCHITECTURES:
        rows = valid[valid['architecture'] == arch]['median_r']
        arch_medians[arch] = float(rows.mean()) if len(rows) > 0 else float('nan')

    # anomalies: NaN pearson_r rows, failed runs, negative r
    anomalies = []
    for target in TARGETS:
        for arch in ARCHITECTURES:
            col_key = f'{target}__{arch}'
            rr = run_results.get(col_key, {})
            if not rr.get('success', False):
                anomalies.append(f'run {col_key} failed: {rr.get("error", "unknown")}')
                continue
            df = run_frames.get(col_key)
            if df is not None:
                nan_count = df['pearson_r'].isna().sum()
                if nan_count > 0:
                    anomalies.append(f'{col_key}: {nan_count} tract(s) with NaN pearson_r')
                neg_count = int((df['pearson_r'].dropna() < -0.2).sum())
                if neg_count > 0:
                    neg_fips = df[df['pearson_r'] < -0.2]['tract_fips'].tolist()
                    anomalies.append(f'{col_key}: {neg_count} tract(s) with r < -0.2 ({neg_fips})')

    # check for guard fallbacks (standardized targets near zero)
    guard_notes = []
    for target in TARGETS:
        for arch in ARCHITECTURES:
            col_key = f'{target}__{arch}'
            df = run_frames.get(col_key)
            if df is not None:
                # constraint_error_pct > 500 suggests guard was not triggered or target is near zero
                extreme = df[df['constraint_error_pct'].fillna(0) > 500]
                if len(extreme) > 0:
                    guard_notes.append(f'{col_key}: {len(extreme)} tract(s) with constraint_error_pct > 500%')

    # build prose
    paras = []

    # paragraph 1: best cell and overall landscape
    p1 = (
        f'Across the six (target, architecture) cells in the n20 sweep, the best median '
        f'Pearson r was achieved by {best_cell} at r = {best_median_r:.3f}. '
    )
    if len(valid) == 6:
        r_range = f'{float(valid["median_r"].min()):.3f} to {float(valid["median_r"].max()):.3f}'
        p1 += f'Median r across all six cells ranged from {r_range}. '
    else:
        n_failed = 6 - len(valid)
        p1 += f'{n_failed} of 6 runs did not produce usable results. '

    # add per-target summary
    target_summaries = []
    for target in TARGETS:
        t_rows = valid[valid['target'] == target]
        if len(t_rows) > 0:
            best_arch = t_rows.loc[t_rows['median_r'].idxmax(), 'architecture']
            best_r = t_rows['median_r'].max()
            target_summaries.append(f'{target} peaked at r = {best_r:.3f} ({best_arch})')
    if target_summaries:
        p1 += 'By target: ' + '; '.join(target_summaries) + '.'
    paras.append(p1)

    # paragraph 2: branch 1 threshold
    if len(branch1_cells) > 0:
        cells_str = ', '.join(
            f"{r['target']} / {r['architecture']} (r = {r['median_r']:.3f})"
            for _, r in branch1_cells.iterrows()
        )
        p2 = (
            f'The Branch 1 threshold of median r > {BRANCH1_THRESHOLD} was crossed by '
            f'{len(branch1_cells)} cell(s): {cells_str}. '
            f'This indicates that at least one (target, architecture) combination achieves '
            f'non-trivial address-level recovery under the hard aggregate constraint.'
        )
    else:
        p2 = (
            f'No cell crossed the Branch 1 threshold of median r > {BRANCH1_THRESHOLD}. '
            f'The highest median r observed was {best_median_r:.3f}. '
            f'All results fall below the threshold for declaring address-level recovery '
            f'competitive with the unconstrained baseline.'
        )
    paras.append(p2)

    # paragraph 3: architecture comparison
    sage_mean = arch_medians.get('sage', float('nan'))
    gcn_mean = arch_medians.get('gcn_gat', float('nan'))
    if np.isfinite(sage_mean) and np.isfinite(gcn_mean):
        diff = sage_mean - gcn_mean
        if abs(diff) < 0.02:
            arch_verdict = (
                f'The two architectures performed similarly: sage averaged '
                f'r = {sage_mean:.3f} and gcn_gat averaged r = {gcn_mean:.3f} '
                f'across their three targets (difference {diff:+.3f}), '
                f'which is within the margin of noise.'
            )
        elif diff > 0:
            arch_verdict = (
                f'sage outperformed gcn_gat on average across the three targets '
                f'(sage: mean median r = {sage_mean:.3f}, gcn_gat: {gcn_mean:.3f}, '
                f'difference {diff:+.3f}).'
            )
        else:
            arch_verdict = (
                f'gcn_gat outperformed sage on average across the three targets '
                f'(gcn_gat: mean median r = {gcn_mean:.3f}, sage: {sage_mean:.3f}, '
                f'difference {diff:+.3f}).'
            )
    else:
        arch_verdict = 'Architecture comparison is not available due to failed runs.'
    paras.append(arch_verdict)

    # paragraph 4: anomalies
    all_notes = anomalies + guard_notes
    if all_notes:
        p4 = 'Anomalies and run flags: ' + ' '.join(all_notes)
    else:
        p4 = (
            'No anomalies were detected. All 20 tracts produced finite pearson_r values '
            'in each of the six runs, and no guard fallbacks were triggered by '
            'constraint_error_pct exceeding 500%.'
        )
    paras.append(p4)

    brief_path = os.path.join(summary_dir, 'decision_brief.md')
    with open(brief_path, 'w') as f:
        f.write('# M2 Sweep Decision Brief\n\n')
        for i, para in enumerate(paras):
            f.write(para + '\n')
            if i < len(paras) - 1:
                f.write('\n')

    return brief_path


def _run_external_sweep(args):
    """
    External target sweep: for each {name, path, tract_subset_csv} entry in
    the JSON config, run recovery across ARCHITECTURES and write results to
    runs/m2_external_<timestamp>/.

    Schema: list of {name: str, path: str, tract_subset_csv: str (optional)}
    """
    import datetime
    import json as _json
    from granite.disaggregation.recovery_harness import run_recovery

    wall_start = time.time()

    with open(args.external_targets) as fh:
        ext_targets = _json.load(fh)

    if not isinstance(ext_targets, list):
        print('[m2-ext] --external-targets JSON must be a list of objects')
        sys.exit(1)

    config = load_config(args.config)
    config['processing']['random_seed'] = SEED
    config['processing']['enable_caching'] = True
    config['data']['target_fips'] = REFERENCE_FIPS
    config['data']['state_fips'] = REFERENCE_FIPS[:2]
    config['data']['county_fips'] = REFERENCE_FIPS[2:5]
    config['recovery']['standardize_target'] = True

    # load default tract list for targets without tract_subset_csv
    inventory_path = INVENTORY_PATH
    if not os.path.exists(inventory_path):
        inventory_path = os.path.join('docs', 'tract_inventory.csv')
    n20_fips = load_n20_tracts(inventory_path)

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    sweep_base = os.path.join('runs', f'm2_external_{ts}')
    os.makedirs(sweep_base, exist_ok=True)

    print(f'[m2-ext] external sweep: {len(ext_targets)} targets x {len(ARCHITECTURES)} architectures')
    print(f'[m2-ext] output base: {sweep_base}')

    run_results = {}

    for entry in ext_targets:
        name = entry.get('name') or os.path.splitext(os.path.basename(entry['path']))[0]
        ext_path = entry['path']
        subset_csv = entry.get('tract_subset_csv')

        if subset_csv and os.path.exists(subset_csv):
            tract_list = load_n20_tracts(subset_csv)
        else:
            tract_list = n20_fips

        if not os.path.exists(ext_path):
            print(f'[m2-ext] skipping "{name}": path not found: {ext_path}')
            continue

        for arch in ARCHITECTURES:
            run_key = f'{name}__{arch}'
            run_dir = os.path.join(sweep_base, f'{name}_{arch}')
            print(f'\n[m2-ext] starting run: target={name}, arch={arch}')
            print(f'[m2-ext]   output: {run_dir}')

            run_cfg = {
                **config,
                'data': {
                    **config.get('data', {}),
                    'target_fips': REFERENCE_FIPS,
                    'neighbor_tracts': len(tract_list) - 1,
                },
                'model': {
                    **config.get('model', {}),
                    'architecture': arch,
                },
                'training': {
                    **config.get('training', {}),
                    'epochs': EPOCHS,
                },
                'processing': {
                    **config.get('processing', {}),
                    'random_seed': SEED,
                },
            }

            t0 = time.time()
            try:
                result = run_recovery(
                    config=run_cfg,
                    output_dir=run_dir,
                    verbose=args.verbose,
                    external_target_path=ext_path,
                )
            except Exception as exc:
                result = {'success': False, 'error': str(exc)}

            elapsed = time.time() - t0
            run_results[run_key] = result

            if result.get('success'):
                ce = result.get('overall_constraint_error', float('nan'))
                print(f'[m2-ext]   done in {elapsed:.1f}s, '
                      f'constraint_err={ce:.2f}%, '
                      f'epochs={result.get("epochs_trained")}')
            else:
                print(f'[m2-ext]   FAILED: {result.get("error", "unknown")}')

    wall_elapsed = time.time() - wall_start
    print(f'\n[m2-ext] external sweep complete in {wall_elapsed:.1f}s')

    ok = sum(1 for r in run_results.values() if r.get('success'))
    print(f'[m2-ext] {ok}/{len(run_results)} runs succeeded')


def main():
    parser = argparse.ArgumentParser(description='M2 sweep driver')
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--skip-preflight', action='store_true')
    parser.add_argument(
        '--external-targets', type=str, default=None, metavar='JSON_PATH',
        help=(
            'Path to JSON config for external target sweep. '
            'Schema: list of {name, path, tract_subset_csv (optional)}. '
            'Mutually exclusive with the built-in held-out-feature target list.'
        ),
    )
    args = parser.parse_args()

    if args.external_targets is not None:
        _run_external_sweep(args)
        return

    wall_start = time.time()

    config = load_config(args.config)

    # load n20 tract list
    inventory_path = INVENTORY_PATH
    if not os.path.exists(inventory_path):
        inventory_path = os.path.join('docs', 'tract_inventory.csv')
    n20_fips = load_n20_tracts(inventory_path)
    print(f'[m2] n20 tracts loaded: {len(n20_fips)} ({inventory_path})')

    # pre-flight smoke test
    if not args.skip_preflight:
        ok = run_preflight_smoke(config, verbose=args.verbose)
        if not ok:
            print('[m2] aborting: pre-flight smoke test failed')
            sys.exit(1)
    else:
        print('[m2] pre-flight skipped (--skip-preflight)')

    # set shared processing config
    config['processing']['random_seed'] = SEED
    config['processing']['enable_caching'] = True
    config['data']['target_fips'] = REFERENCE_FIPS
    config['data']['state_fips'] = REFERENCE_FIPS[:2]
    config['data']['county_fips'] = REFERENCE_FIPS[2:5]
    config['recovery']['standardize_target'] = True

    run_results = {}

    for target in TARGETS:
        for arch in ARCHITECTURES:
            run_key = f'{target}__{arch}'
            run_dir = os.path.join(SWEEP_BASE, f'{target}_{arch}')
            print(f'\n[m2] starting run: target={target}, arch={arch}')
            print(f'[m2]   output: {run_dir}')

            run_cfg = {
                **config,
                'data': {
                    **config.get('data', {}),
                    'target_fips': REFERENCE_FIPS,
                },
                'model': {
                    **config.get('model', {}),
                    'architecture': arch,
                },
                'training': {
                    **config.get('training', {}),
                    'epochs': EPOCHS,
                },
                'processing': {
                    **config.get('processing', {}),
                    'random_seed': SEED,
                },
            }

            t0 = time.time()
            result = _run_recovery_explicit_tracts(
                config=run_cfg,
                target_feature=target,
                tract_list=n20_fips,
                output_dir=run_dir,
                verbose=args.verbose,
            )
            elapsed = time.time() - t0

            run_results[run_key] = result
            if result.get('success'):
                ce = result.get('overall_constraint_error', float('nan'))
                print(f'[m2]   done in {elapsed:.1f}s, '
                      f'constraint_err={ce:.2f}%, '
                      f'epochs={result.get("epochs_trained")}')
                skipped = result.get('skipped_tracts', [])
                if skipped:
                    print(f'[m2]   skipped tracts: {skipped}')
            else:
                print(f'[m2]   FAILED: {result.get("error", "unknown error")}')

    # aggregation
    print('\n[m2] aggregating results...')
    stats_df, pivot_r, pivot_rmse, run_frames, run_meta = aggregate_results(SWEEP_BASE, n20_fips)

    brief_path = write_decision_brief(
        os.path.join(SWEEP_BASE, 'summary'),
        stats_df,
        pivot_r,
        run_frames,
        run_results,
    )

    wall_elapsed = time.time() - wall_start

    # report
    print(f'\n[m2] sweep complete in {wall_elapsed:.1f}s ({wall_elapsed/60:.1f} min)')
    print('\n--- summary_stats.csv ---')
    print(stats_df.to_string(index=False))

    print(f'\n--- decision_brief.md ({brief_path}) ---')
    with open(brief_path) as f:
        print(f.read())

    # flag any NaN or failed runs
    for target in TARGETS:
        for arch in ARCHITECTURES:
            col_key = f'{target}__{arch}'
            rr = run_results.get(col_key, {})
            if not rr.get('success', False):
                print(f'[m2] FAILED run: {col_key}')
            df = run_frames.get(col_key)
            if df is not None:
                nan_tracts = df[df['pearson_r'].isna()]['tract_fips'].tolist()
                if nan_tracts:
                    print(f'[m2] NaN pearson_r in {col_key}: {nan_tracts}')


if __name__ == '__main__':
    main()
