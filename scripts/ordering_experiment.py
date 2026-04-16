"""
Pairwise Ordering Constraint Effectiveness Experiment

2x3 design: {GCN, GraphSAGE} x {tract-only, tract+BG, tract+BG+ordering}
Measures whether ordering constraints improve within-block-group agreement
and prediction quality for tract 47065000600.
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats
from datetime import datetime

sys.path.insert(0, '/workspaces/GRANITE')

from granite.models.gnn import (
    AccessibilitySVIGNN, GraphSAGEAccessibilitySVIGNN,
    MultiTractGNNTrainer, set_random_seed, normalize_accessibility_features
)
from granite.data.loaders import DataLoader
from granite.data.block_group_loader import BlockGroupLoader, rescale_block_group_svis

TARGET_FIPS = '47065000600'
SEED = 42
HIDDEN_DIM = 64
EPOCHS = 200
LEARNING_RATE = 0.005
DROPOUT = 0.3
ORDERING_WEIGHT = 0.5
ORDERING_MIN_GAP = 0.5
ORDERING_MARGIN = 0.02

RESULTS_DIR = '/workspaces/GRANITE/results/convergence_experiment'

# prior experiment results for reproducibility checks
PRIOR_TRACT_ONLY = {
    'gcn_tract_error': 8.26,
    'sage_tract_error': 6.71,
    'gcn_spatial_std': 0.1002,
    'sage_spatial_std': 0.1168,
    'gcn_sage_r': 0.420,
    'gcn_sage_rho': 0.327,
}
PRIOR_BG = {
    'gcn_tract_error': 5.01,
    'sage_tract_error': 1.87,
    'gcn_bg_error': 6.04,
    'sage_bg_error': 3.89,
    'gcn_spatial_std': 0.2052,
    'sage_spatial_std': 0.2056,
    'gcn_sage_r': 0.693,
    'gcn_sage_rho': 0.790,
}
PRIOR_PER_BG_AGREEMENT = {
    '470650006001': 0.711,
    '470650006002': -0.010,
    '470650006003': 0.079,
}


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def load_config():
    with open('/workspaces/GRANITE/config.yaml') as f:
        return yaml.safe_load(f)


def load_data_and_features(config):
    """load addresses, compute features, build graph -- shared across all runs."""
    log("Loading spatial data...")
    data_loader = DataLoader('./data', config=config)

    state_fips = config['data']['state_fips']
    county_fips = config['data']['county_fips']
    census_tracts = data_loader.load_census_tracts(state_fips, county_fips)
    county_name = data_loader._get_county_name(state_fips, county_fips)
    svi = data_loader.load_svi_data(state_fips, county_name)
    tracts = census_tracts.merge(svi, on='FIPS', how='inner')

    target_tract = tracts[tracts['FIPS'] == TARGET_FIPS]
    if len(target_tract) == 0:
        raise RuntimeError(f"Tract {TARGET_FIPS} not found")
    tract_svi = float(target_tract.iloc[0]['RPL_THEMES'])
    log(f"Tract {TARGET_FIPS} SVI = {tract_svi:.4f}")

    addresses = data_loader.get_addresses_for_tract(TARGET_FIPS)
    if len(addresses) == 0:
        raise RuntimeError("No addresses found")
    addresses['tract_fips'] = TARGET_FIPS
    log(f"Loaded {len(addresses)} addresses")

    employment = data_loader.create_employment_destinations(use_real_data=True)
    healthcare = data_loader.create_healthcare_destinations(use_real_data=True)
    grocery = data_loader.create_grocery_destinations(use_real_data=True)

    from granite.disaggregation.pipeline import GRANITEPipeline
    pipeline = GRANITEPipeline(config, data_dir='./data', output_dir='/tmp/granite_exp', verbose=True)
    pipeline.data_loader = data_loader

    data = {
        'tracts': tracts,
        'svi': svi,
        'employment_destinations': employment,
        'healthcare_destinations': healthcare,
        'grocery_destinations': grocery,
        'roads': data_loader.load_road_network(state_fips=state_fips, county_fips=county_fips),
        'addresses': data_loader.load_address_points(state_fips, county_fips),
    }

    log("Computing accessibility features (using cache)...")
    accessibility_features = pipeline._compute_accessibility_features(addresses, data)
    if accessibility_features is None:
        raise RuntimeError("Failed to compute accessibility features")

    feature_names = pipeline._generate_feature_names(accessibility_features.shape[1])
    log(f"Feature matrix: {accessibility_features.shape}")

    normalized_features, _ = normalize_accessibility_features(accessibility_features)

    context_features = data_loader.create_context_features_for_addresses(
        addresses=addresses, svi_data=svi
    )
    normalized_context, _ = data_loader.normalize_context_features(context_features)

    log("Building graph...")
    graph_data = data_loader.create_spatial_accessibility_graph(
        addresses=addresses,
        accessibility_features=normalized_features,
        context_features=normalized_context,
        state_fips=state_fips,
        county_fips=county_fips
    )
    log(f"Graph: {graph_data.x.shape[0]} nodes, {graph_data.edge_index.shape[1]} edges")

    return {
        'graph_data': graph_data,
        'addresses': addresses,
        'tract_svi': tract_svi,
        'feature_names': feature_names,
        'data_loader': data_loader,
    }


def load_block_group_data_national(addresses, tract_svi, rescale=True):
    """load block group targets with national SVI ranking, optionally rescaling."""
    log("Loading block group data (national ranking)...")
    bg_loader = BlockGroupLoader(data_dir='./data', verbose=True)
    bg_data = bg_loader.get_block_groups_with_demographics(svi_ranking_scope='national')
    addresses_with_bg = bg_loader.assign_addresses_to_block_groups(addresses, bg_data)

    block_group_targets = {}
    block_group_masks = {}
    bg_address_counts = {}

    for _, bg_row in bg_data.iterrows():
        bg_id = bg_row['GEOID']
        if not bg_row.get('svi_complete', False):
            continue
        svi_val = bg_row.get('SVI', None)
        if svi_val is None or pd.isna(svi_val):
            continue
        bg_tract = bg_row.get('tract_fips', bg_id[:11])
        if bg_tract != TARGET_FIPS:
            continue
        bg_mask = (addresses_with_bg['block_group_id'] == bg_id).values
        n_bg = bg_mask.sum()
        if n_bg < 5:
            continue
        block_group_targets[bg_id] = float(svi_val)
        block_group_masks[bg_id] = bg_mask
        bg_address_counts[bg_id] = int(n_bg)

    raw_targets = dict(block_group_targets)

    if rescale and block_group_targets:
        block_group_targets = rescale_block_group_svis(
            block_group_targets, bg_address_counts, tract_svi
        )

    return block_group_targets, block_group_masks, bg_data, raw_targets, bg_address_counts


def preflight_ordering(addresses, bg_masks):
    """check log_appvalue availability and valid pair counts per block group."""
    log("\n--- PRE-FLIGHT: Ordering data inspection ---")

    if 'log_appvalue' not in addresses.columns:
        log("FATAL: log_appvalue column not found in address data")
        return None, None

    raw_vals = pd.to_numeric(addresses['log_appvalue'], errors='coerce').values.astype(float)
    n_total = len(raw_vals)
    n_valid = int(np.sum(~np.isnan(raw_vals)))
    log(f"Addresses with valid log_appvalue: {n_valid} / {n_total}")

    if n_valid < 2:
        log("FATAL: fewer than 2 addresses with valid log_appvalue")
        return None, None

    print(f"\n{'Block Group':<16s}  {'n_addr':>6s}  {'n_valid':>7s}  {'min':>8s}  {'max':>8s}  {'median':>8s}  {'pairs(gap>=0.5)':>15s}")
    print(f"{'-'*16}  {'-'*6}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*15}")

    bg_pair_counts = {}
    any_sparse = False
    for bg_id in sorted(bg_masks.keys()):
        mask = bg_masks[bg_id]
        n_addr = int(mask.sum())
        bg_vals = raw_vals[mask]
        valid = bg_vals[~np.isnan(bg_vals)]
        n_v = len(valid)

        if n_v < 2:
            bg_pair_counts[bg_id] = 0
            print(f"{bg_id:<16s}  {n_addr:>6d}  {n_v:>7d}  {'n/a':>8s}  {'n/a':>8s}  {'n/a':>8s}  {0:>15d}")
            any_sparse = True
            continue

        sorted_vals = np.sort(valid)
        n_pairs = 0
        for i in range(len(sorted_vals)):
            j_start = np.searchsorted(sorted_vals, sorted_vals[i] + ORDERING_MIN_GAP)
            n_pairs += len(sorted_vals) - j_start

        bg_pair_counts[bg_id] = n_pairs
        if n_pairs < 10:
            any_sparse = True

        print(f"{bg_id:<16s}  {n_addr:>6d}  {n_v:>7d}  {np.min(valid):>8.3f}  {np.max(valid):>8.3f}  {np.median(valid):>8.3f}  {n_pairs:>15d}")

    if any_sparse:
        log("WARNING: one or more block groups have fewer than 10 valid pairs -- ordering signal will be sparse there")

    total_pairs = sum(bg_pair_counts.values())
    log(f"Total valid pairs across all block groups: {total_pairs}")

    return raw_vals, bg_pair_counts


def run_training(label, model_class, graph_data, tract_svi, feature_names,
                 block_group_targets=None, block_group_masks=None,
                 ordering_values=None):
    """run a single training configuration and return results."""
    log(f"=== Run {label} ===")
    set_random_seed(SEED)

    has_context = hasattr(graph_data, 'context') and graph_data.context is not None
    context_dim = graph_data.context.shape[1] if has_context else 5

    model = model_class(
        accessibility_features_dim=graph_data.x.shape[1],
        context_features_dim=context_dim,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
        seed=SEED,
        use_context_gating=has_context,
        use_multitask=True
    )

    trainer_config = {
        'learning_rate': LEARNING_RATE,
        'weight_decay': 1e-4,
        'enforce_constraints': True,
        'constraint_weight': 2.0,
        'use_multitask': True,
    }
    if block_group_targets is not None:
        trainer_config['bg_constraint_weight'] = 1.0
    else:
        trainer_config['bg_constraint_weight'] = 0.0

    if ordering_values is not None:
        trainer_config['ordering_weight'] = ORDERING_WEIGHT
        trainer_config['ordering_min_gap'] = ORDERING_MIN_GAP
        trainer_config['ordering_margin'] = ORDERING_MARGIN
    else:
        trainer_config['ordering_weight'] = 0.0

    trainer = MultiTractGNNTrainer(model, config=trainer_config, seed=SEED)

    tract_svis = {TARGET_FIPS: tract_svi}
    n = graph_data.x.shape[0]
    tract_masks = {TARGET_FIPS: np.ones(n, dtype=bool)}

    try:
        result = trainer.train(
            graph_data=graph_data,
            tract_svis=tract_svis,
            tract_masks=tract_masks,
            epochs=EPOCHS,
            verbose=False,
            feature_names=feature_names,
            block_group_targets=block_group_targets,
            block_group_masks=block_group_masks,
            ordering_values=ordering_values
        )
        predictions = result['final_predictions'].flatten()

        # extract ordering loss from training history
        ordering_losses = result['training_history'].get('ordering_losses', [])
        final_ordering_loss = ordering_losses[-1] if ordering_losses else None

        log(f"  {label}: epochs={result['epochs_trained']}, loss={result['final_loss']:.6f}, "
            f"tract_err={result['overall_constraint_error']:.2f}%, "
            f"std={result['final_spatial_std']:.4f}")
        if 'bg_constraint_error' in result:
            log(f"  BG error: {result['bg_constraint_error']:.2f}%")
        if final_ordering_loss is not None:
            log(f"  Ordering loss: {final_ordering_loss:.4f}")

        return {
            'predictions': predictions,
            'epochs_trained': result['epochs_trained'],
            'final_loss': result['final_loss'],
            'tract_error': result['overall_constraint_error'],
            'bg_error': result.get('bg_constraint_error', None),
            'spatial_std': result['final_spatial_std'],
            'per_bg_errors': result.get('per_bg_errors', {}),
            'final_ordering_loss': final_ordering_loss,
            'ordering_losses': ordering_losses,
            'success': True,
        }
    except Exception as e:
        log(f"  {label} FAILED: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def compute_ordering_compliance(predictions, ordering_values, bg_masks, min_gap=0.5):
    """compute fraction of all valid pairs where ordering is satisfied."""
    raw_vals = ordering_values
    total_correct = 0
    total_pairs = 0

    for bg_id, mask in bg_masks.items():
        bg_preds = predictions[mask]
        bg_vals = raw_vals[mask]
        valid = ~np.isnan(bg_vals)
        if valid.sum() < 2:
            continue

        valid_preds = bg_preds[valid]
        valid_vals = bg_vals[valid]

        # sort by property value
        order = np.argsort(valid_vals)
        sorted_vals = valid_vals[order]
        sorted_preds = valid_preds[order]

        n = len(sorted_vals)
        for i in range(n):
            j_start = np.searchsorted(sorted_vals, sorted_vals[i] + min_gap)
            for j in range(j_start, n):
                total_pairs += 1
                # low property value (i) should predict higher vulnerability than high property value (j)
                if sorted_preds[i] > sorted_preds[j]:
                    total_correct += 1

    if total_pairs == 0:
        return 0.0, 0
    return total_correct / total_pairs, total_pairs


def check_reproducibility(runs):
    """check if tract-only and BG runs match prior experiments."""
    warnings = []
    checks = [
        ('A', 'tract_error', PRIOR_TRACT_ONLY['gcn_tract_error'], 'GCN tract-only tract error'),
        ('B', 'tract_error', PRIOR_TRACT_ONLY['sage_tract_error'], 'SAGE tract-only tract error'),
        ('A', 'spatial_std', PRIOR_TRACT_ONLY['gcn_spatial_std'], 'GCN tract-only spatial std'),
        ('B', 'spatial_std', PRIOR_TRACT_ONLY['sage_spatial_std'], 'SAGE tract-only spatial std'),
        ('C', 'tract_error', PRIOR_BG['gcn_tract_error'], 'GCN +BG tract error'),
        ('D', 'tract_error', PRIOR_BG['sage_tract_error'], 'SAGE +BG tract error'),
        ('C', 'bg_error', PRIOR_BG['gcn_bg_error'], 'GCN +BG BG error'),
        ('D', 'bg_error', PRIOR_BG['sage_bg_error'], 'SAGE +BG BG error'),
    ]
    for run_key, metric, prior_val, desc in checks:
        r = runs.get(run_key, {})
        if not r.get('success'):
            continue
        val = r.get(metric)
        if val is None:
            continue
        if isinstance(prior_val, float) and prior_val < 1:
            # spatial std: relative comparison
            if abs(val - prior_val) / max(prior_val, 1e-6) > 0.05:
                warnings.append(f"WARNING: {desc}: {val:.4f} vs prior {prior_val:.4f} (>5% relative diff)")
        else:
            if abs(val - prior_val) > 1.0:
                warnings.append(f"WARNING: {desc}: {val:.2f}% vs prior {prior_val:.2f}% (>1pp diff)")
    return warnings


def main():
    start = time.time()
    log("Pairwise Ordering Constraint Effectiveness Experiment")
    log("=" * 60)

    config = load_config()

    # step 1: load shared data
    shared = load_data_and_features(config)
    graph_data = shared['graph_data']
    tract_svi = shared['tract_svi']
    feature_names = shared['feature_names']
    addresses = shared['addresses']

    # step 2: load BG data
    bg_targets, bg_masks, bg_data, raw_targets, bg_addr_counts = \
        load_block_group_data_national(addresses, tract_svi, rescale=True)

    log(f"\nRescaled BG targets:")
    for bg_id in sorted(bg_targets.keys()):
        log(f"  {bg_id}: {bg_targets[bg_id]:.4f} ({bg_addr_counts[bg_id]} addresses)")

    # step 3: pre-flight ordering check
    ordering_values, bg_pair_counts = preflight_ordering(addresses, bg_masks)
    if ordering_values is None:
        log("STOPPING: no valid ordering data")
        return

    # step 4: run six training configurations
    log("\n--- TRAINING RUNS (6 configurations) ---")
    runs = {}

    # A: GCN, tract-only
    runs['A'] = run_training('A (GCN, tract-only)', AccessibilitySVIGNN,
                              graph_data, tract_svi, feature_names)

    # B: SAGE, tract-only
    runs['B'] = run_training('B (SAGE, tract-only)', GraphSAGEAccessibilitySVIGNN,
                              graph_data, tract_svi, feature_names)

    # C: GCN, tract + BG
    runs['C'] = run_training('C (GCN, tract+BG)', AccessibilitySVIGNN,
                              graph_data, tract_svi, feature_names,
                              block_group_targets=bg_targets,
                              block_group_masks=bg_masks)

    # D: SAGE, tract + BG
    runs['D'] = run_training('D (SAGE, tract+BG)', GraphSAGEAccessibilitySVIGNN,
                              graph_data, tract_svi, feature_names,
                              block_group_targets=bg_targets,
                              block_group_masks=bg_masks)

    # E: GCN, tract + BG + ordering
    runs['E'] = run_training('E (GCN, tract+BG+ordering)', AccessibilitySVIGNN,
                              graph_data, tract_svi, feature_names,
                              block_group_targets=bg_targets,
                              block_group_masks=bg_masks,
                              ordering_values=ordering_values)

    # F: SAGE, tract + BG + ordering
    runs['F'] = run_training('F (SAGE, tract+BG+ordering)', GraphSAGEAccessibilitySVIGNN,
                              graph_data, tract_svi, feature_names,
                              block_group_targets=bg_targets,
                              block_group_masks=bg_masks,
                              ordering_values=ordering_values)

    # check reproducibility
    warnings = check_reproducibility(runs)
    if warnings:
        log("\n--- REPRODUCIBILITY WARNINGS ---")
        for w in warnings:
            log(w)

    # step 5: cross-architecture correlations
    log("\n--- CROSS-ARCHITECTURE CORRELATIONS ---")
    conditions = [
        ('A', 'B', 'Tract-only'),
        ('C', 'D', '+BG'),
        ('E', 'F', '+BG+ordering'),
    ]
    correlations = {}
    for k1, k2, desc in conditions:
        r1, r2 = runs.get(k1), runs.get(k2)
        if r1 and r1['success'] and r2 and r2['success']:
            pearson_r, _ = stats.pearsonr(r1['predictions'], r2['predictions'])
            spearman_rho, _ = stats.spearmanr(r1['predictions'], r2['predictions'])
            correlations[f'{k1}_vs_{k2}'] = {
                'description': f'GCN vs SAGE ({desc})',
                'pearson_r': float(pearson_r),
                'spearman_rho': float(spearman_rho),
            }
            log(f"  {desc}: r={pearson_r:.3f}, rho={spearman_rho:.3f}")

    # step 6: per-block-group agreement
    log("\n--- PER-BLOCK-GROUP AGREEMENT ---")
    condition_pairs = [('A', 'B'), ('C', 'D'), ('E', 'F')]
    condition_labels = ['Tract-only', '+BG', '+BG+ordering']

    print(f"\n   Per-Block-Group Cross-Architecture Agreement:")
    hdr = f"   {'GEOID':<16s}  {'n_addr':>6s}  {'n_pairs':>7s}"
    for lbl in condition_labels:
        hdr += f"  {lbl:>12s}"
    print(hdr)
    print(f"   {'-'*16}  {'-'*6}  {'-'*7}" + f"  {'-'*12}" * 3)

    bg_agreement = {}
    for bg_id in sorted(bg_masks.keys()):
        mask = bg_masks[bg_id]
        n_addr = int(mask.sum())
        n_pairs = bg_pair_counts.get(bg_id, 0)
        row = f"   {bg_id:<16s}  {n_addr:>6d}  {n_pairs:>7d}"
        bg_agreement[bg_id] = {}

        for (k1, k2), lbl in zip(condition_pairs, condition_labels):
            r1, r2 = runs.get(k1), runs.get(k2)
            if r1 and r1['success'] and r2 and r2['success']:
                a_bg = r1['predictions'][mask]
                b_bg = r2['predictions'][mask]
                if len(a_bg) >= 3 and np.std(a_bg) > 1e-10 and np.std(b_bg) > 1e-10:
                    r_val, _ = stats.pearsonr(a_bg, b_bg)
                    row += f"  {r_val:>12.3f}"
                    bg_agreement[bg_id][lbl] = float(r_val)
                else:
                    row += f"  {'n/a':>12s}"
            else:
                row += f"  {'n/a':>12s}"
        print(row)

    # step 7: ordering compliance
    log("\n--- ORDERING COMPLIANCE ---")
    compliance = {}
    compliance_runs = [('C', 'GCN no ordering'), ('D', 'SAGE no ordering'),
                       ('E', 'GCN with ordering'), ('F', 'SAGE with ordering')]
    for run_key, desc in compliance_runs:
        r = runs.get(run_key)
        if r and r['success']:
            frac, n_pairs = compute_ordering_compliance(
                r['predictions'], ordering_values, bg_masks, min_gap=ORDERING_MIN_GAP
            )
            compliance[run_key] = {'fraction': frac, 'n_pairs': n_pairs}
            log(f"  {desc}: {frac*100:.1f}% ({n_pairs} pairs)")

    print(f"\n   Ordering Compliance (fraction of valid pairs correctly ordered):")
    print(f"   {'':>20s}  {'No ordering loss':>16s}  {'With ordering loss':>18s}")
    gcn_no = compliance.get('C', {}).get('fraction')
    gcn_yes = compliance.get('E', {}).get('fraction')
    sage_no = compliance.get('D', {}).get('fraction')
    sage_yes = compliance.get('F', {}).get('fraction')
    print(f"   {'GCN:':>20s}  {gcn_no*100 if gcn_no is not None else float('nan'):>15.1f}%  {gcn_yes*100 if gcn_yes is not None else float('nan'):>17.1f}%")
    print(f"   {'SAGE:':>20s}  {sage_no*100 if sage_no is not None else float('nan'):>15.1f}%  {sage_yes*100 if sage_yes is not None else float('nan'):>17.1f}%")

    # step 8: rank correlation with property value
    log("\n--- RANK CORRELATION WITH PROPERTY VALUE ---")
    # higher appvalue => lower vulnerability, so we correlate predictions with -log_appvalue
    valid_mask = ~np.isnan(ordering_values)
    inverted_vals = -ordering_values  # higher property value = more negative = lower vulnerability

    print(f"\n   Spearman rho (predictions vs inverted log_appvalue):")
    print(f"   {'':>20s}  {'Tract-only':>10s}  {'+BG':>10s}  {'+BG+ordering':>12s}")
    appvalue_rho = {}
    for arch_label, run_keys in [('GCN:', ['A', 'C', 'E']), ('SAGE:', ['B', 'D', 'F'])]:
        row = f"   {arch_label:>20s}"
        for rk in run_keys:
            r = runs.get(rk)
            if r and r['success']:
                rho, _ = stats.spearmanr(r['predictions'][valid_mask], inverted_vals[valid_mask])
                appvalue_rho[rk] = float(rho)
                row += f"  {rho:>10.3f}"
            else:
                row += f"  {'n/a':>10s}"
        print(row)

    # step 9: count violated pairs at final epoch for E and F
    violated_info = {}
    for rk in ['E', 'F']:
        r = runs.get(rk)
        if r and r['success']:
            preds = r['predictions']
            n_violated = 0
            n_total = 0
            for bg_id, mask in bg_masks.items():
                bg_preds = preds[mask]
                bg_vals = ordering_values[mask]
                valid = ~np.isnan(bg_vals)
                if valid.sum() < 2:
                    continue
                vp = bg_preds[valid]
                vv = bg_vals[valid]
                order = np.argsort(vv)
                sv = vv[order]
                sp = vp[order]
                for i in range(len(sv)):
                    j_start = np.searchsorted(sv, sv[i] + ORDERING_MIN_GAP)
                    for j in range(j_start, len(sv)):
                        n_total += 1
                        if sp[i] <= sp[j]:  # violated: low-value not predicting higher SVI
                            n_violated += 1
            violated_info[rk] = {'violated': n_violated, 'total': n_total}

    # step 10: full summary table
    print("\n" + "=" * 90)
    total_valid_pairs = sum(bg_pair_counts.values())
    print(f"   ORDERING CONSTRAINT EXPERIMENT: Tract {TARGET_FIPS}")
    print(f"   Valid pairs per epoch: ~{min(100, total_valid_pairs) * len(bg_masks)} across {len(bg_masks)} block groups")
    print()

    def nval(run_key, metric, fmt='.2f', suffix='%', default='n/a'):
        r = runs.get(run_key)
        if r and r['success'] and r.get(metric) is not None:
            return f"{r[metric]:{fmt}}{suffix}"
        return default

    print(f"   {'':>30s}  {'Tract-Only':>12s}  {'+BG':>12s}  {'+BG+Ordering':>12s}")
    print(f"   {'':>30s}  {'-'*12:>12s}  {'-'*12:>12s}  {'-'*12:>12s}")
    print(f"   {'GCN tract error:':>30s}  {nval('A','tract_error'):>12s}  {nval('C','tract_error'):>12s}  {nval('E','tract_error'):>12s}")
    print(f"   {'SAGE tract error:':>30s}  {nval('B','tract_error'):>12s}  {nval('D','tract_error'):>12s}  {nval('F','tract_error'):>12s}")
    print(f"   {'GCN BG error:':>30s}  {'n/a':>12s}  {nval('C','bg_error'):>12s}  {nval('E','bg_error'):>12s}")
    print(f"   {'SAGE BG error:':>30s}  {'n/a':>12s}  {nval('D','bg_error'):>12s}  {nval('F','bg_error'):>12s}")

    # ordering loss
    e_ord = nval('E', 'final_ordering_loss', '.4f', '', 'n/a')
    f_ord = nval('F', 'final_ordering_loss', '.4f', '', 'n/a')
    print(f"   {'GCN ordering loss:':>30s}  {'n/a':>12s}  {'n/a':>12s}  {e_ord:>12s}")
    print(f"   {'SAGE ordering loss:':>30s}  {'n/a':>12s}  {'n/a':>12s}  {f_ord:>12s}")

    print(f"   {'GCN spatial std:':>30s}  {nval('A','spatial_std','.4f',''):>12s}  {nval('C','spatial_std','.4f',''):>12s}  {nval('E','spatial_std','.4f',''):>12s}")
    print(f"   {'SAGE spatial std:':>30s}  {nval('B','spatial_std','.4f',''):>12s}  {nval('D','spatial_std','.4f',''):>12s}  {nval('F','spatial_std','.4f',''):>12s}")

    # cross-arch correlations
    for pair_key, label in [('A_vs_B', 'Tract-only'), ('C_vs_D', '+BG'), ('E_vs_F', '+BG+ordering')]:
        c = correlations.get(pair_key, {})
        r_val = f"{c['pearson_r']:.3f}" if 'pearson_r' in c else 'n/a'
        rho_val = f"{c['spearman_rho']:.3f}" if 'spearman_rho' in c else 'n/a'
        if label == 'Tract-only':
            col1, col2, col3 = r_val, '', ''
        elif label == '+BG':
            col1, col2, col3 = '', r_val, ''
        else:
            col1, col2, col3 = '', '', r_val

    # print as single rows spanning all conditions
    ab = correlations.get('A_vs_B', {})
    cd = correlations.get('C_vs_D', {})
    ef = correlations.get('E_vs_F', {})
    print(f"   {'GCN vs SAGE r:':>30s}  {ab.get('pearson_r', float('nan')):>12.3f}  {cd.get('pearson_r', float('nan')):>12.3f}  {ef.get('pearson_r', float('nan')):>12.3f}")
    print(f"   {'GCN vs SAGE rho:':>30s}  {ab.get('spearman_rho', float('nan')):>12.3f}  {cd.get('spearman_rho', float('nan')):>12.3f}  {ef.get('spearman_rho', float('nan')):>12.3f}")

    # compliance
    print(f"   {'GCN ordering compliance:':>30s}  {'n/a':>12s}  {gcn_no*100 if gcn_no is not None else float('nan'):>11.1f}%  {gcn_yes*100 if gcn_yes is not None else float('nan'):>11.1f}%")
    print(f"   {'SAGE ordering compliance:':>30s}  {'n/a':>12s}  {sage_no*100 if sage_no is not None else float('nan'):>11.1f}%  {sage_yes*100 if sage_yes is not None else float('nan'):>11.1f}%")

    # appvalue rho
    print(f"   {'GCN appvalue rho:':>30s}  {appvalue_rho.get('A', float('nan')):>12.3f}  {appvalue_rho.get('C', float('nan')):>12.3f}  {appvalue_rho.get('E', float('nan')):>12.3f}")
    print(f"   {'SAGE appvalue rho:':>30s}  {appvalue_rho.get('B', float('nan')):>12.3f}  {appvalue_rho.get('D', float('nan')):>12.3f}  {appvalue_rho.get('F', float('nan')):>12.3f}")

    # violated pairs
    e_viol = violated_info.get('E', {})
    f_viol = violated_info.get('F', {})
    if e_viol and f_viol:
        print(f"   {'Violated pairs (final):':>30s}  {'n/a':>12s}  {'n/a':>12s}  {e_viol['violated']}/{e_viol['total']} GCN, {f_viol['violated']}/{f_viol['total']} SAGE")

    print("=" * 90)

    # step 11: interpretation
    print("\nInterpretation:")
    ef_r = ef.get('pearson_r', None)
    cd_r = cd.get('pearson_r', None)

    if ef_r is not None and cd_r is not None:
        r_improvement = ef_r - cd_r
        compliance_improvement_gcn = (gcn_yes - gcn_no) * 100 if gcn_yes is not None and gcn_no is not None else 0
        compliance_improvement_sage = (sage_yes - sage_no) * 100 if sage_yes is not None and sage_no is not None else 0
        avg_compliance_improvement = (compliance_improvement_gcn + compliance_improvement_sage) / 2

        # check tract/BG error degradation
        e_tract = runs.get('E', {}).get('tract_error', 0) if runs.get('E', {}).get('success') else 0
        c_tract = runs.get('C', {}).get('tract_error', 0) if runs.get('C', {}).get('success') else 0
        f_tract = runs.get('F', {}).get('tract_error', 0) if runs.get('F', {}).get('success') else 0
        d_tract = runs.get('D', {}).get('tract_error', 0) if runs.get('D', {}).get('success') else 0
        tract_err_increase = max(e_tract - c_tract, f_tract - d_tract)

        if tract_err_increase > 5:
            print("Ordering constraints interfere with aggregate constraint satisfaction. "
                  "Reduce ordering_weight or increase constraint_weight.")
        elif r_improvement > 0.1 and avg_compliance_improvement > 10:
            print("Ordering constraints provide effective within-block-group supervision. "
                  "The pairwise signal resolves allocation ambiguity that aggregate constraints alone cannot.")
        elif r_improvement > 0.1 and avg_compliance_improvement <= 10:
            print("Cross-architecture convergence improved but ordering compliance is marginal. "
                  "The ordering loss may be acting as a regularizer rather than directly enforcing parcel-based ordering.")
        else:
            print("Ordering constraints did not meaningfully improve convergence. "
                  "Possible causes: insufficient valid pairs, min_gap too restrictive, ordering_weight too low, "
                  "or property value is too noisy a proxy for vulnerability in this tract.")

        # check BG2 and BG3 specifically
        bg2_bg3_improved = True
        for bg_id in ['470650006002', '470650006003']:
            bg_ordering_r = bg_agreement.get(bg_id, {}).get('+BG+ordering')
            if bg_ordering_r is None or bg_ordering_r < 0.3:
                bg2_bg3_improved = False

        if bg2_bg3_improved:
            print("\nOrdering constraints specifically resolve the underdetermined allocation in block groups "
                  "where aggregate targets are close together.")

    # step 12: save results
    os.makedirs(RESULTS_DIR, exist_ok=True)

    json_results = {
        'experiment': 'ordering_constraint_effectiveness',
        'tract': TARGET_FIPS,
        'seed': SEED,
        'hyperparameters': {
            'hidden_dim': HIDDEN_DIM, 'epochs': EPOCHS,
            'learning_rate': LEARNING_RATE, 'dropout': DROPOUT,
            'ordering_weight': ORDERING_WEIGHT,
            'ordering_min_gap': ORDERING_MIN_GAP,
            'ordering_margin': ORDERING_MARGIN,
        },
        'preflight': {
            'n_valid_log_appvalue': int(np.sum(~np.isnan(ordering_values))),
            'n_total_addresses': len(ordering_values),
            'bg_pair_counts': {k: int(v) for k, v in bg_pair_counts.items()},
        },
        'runs': {},
        'correlations': correlations,
        'bg_agreement': bg_agreement,
        'compliance': {k: {'fraction': v['fraction'], 'n_pairs': v['n_pairs']}
                       for k, v in compliance.items()},
        'appvalue_rho': appvalue_rho,
        'violated_info': {k: {kk: int(vv) for kk, vv in v.items()}
                          for k, v in violated_info.items()},
        'reproducibility_warnings': warnings,
        'timestamp': datetime.now().isoformat(),
        'elapsed_seconds': time.time() - start,
    }
    for label, r in runs.items():
        entry = {k: v for k, v in r.items() if k not in ('predictions', 'ordering_losses')}
        if r.get('success'):
            entry['predictions'] = r['predictions'].tolist()
            if 'per_bg_errors' in entry:
                entry['per_bg_errors'] = {k: float(v) for k, v in entry['per_bg_errors'].items()}
        json_results['runs'][label] = entry

    with open(os.path.join(RESULTS_DIR, 'ordering_experiment.json'), 'w') as f:
        json.dump(json_results, f, indent=2)
    log(f"\nSaved JSON to {RESULTS_DIR}/ordering_experiment.json")

    # save summary text
    summary_lines = []
    summary_lines.append(f"Ordering Constraint Effectiveness Experiment: {datetime.now().isoformat()}")
    summary_lines.append(f"Tract: {TARGET_FIPS}, Seed: {SEED}, Epochs: {EPOCHS}")
    summary_lines.append(f"Ordering: weight={ORDERING_WEIGHT}, min_gap={ORDERING_MIN_GAP}, margin={ORDERING_MARGIN}")
    summary_lines.append(f"Valid pairs: {total_valid_pairs} across {len(bg_masks)} block groups")
    summary_lines.append("")

    # reproduce the summary table as text
    summary_lines.append(f"{'':>30s}  {'Tract-Only':>12s}  {'+BG':>12s}  {'+BG+Ordering':>12s}")
    summary_lines.append(f"{'':>30s}  {'-'*12:>12s}  {'-'*12:>12s}  {'-'*12:>12s}")
    summary_lines.append(f"{'GCN tract error:':>30s}  {nval('A','tract_error'):>12s}  {nval('C','tract_error'):>12s}  {nval('E','tract_error'):>12s}")
    summary_lines.append(f"{'SAGE tract error:':>30s}  {nval('B','tract_error'):>12s}  {nval('D','tract_error'):>12s}  {nval('F','tract_error'):>12s}")
    summary_lines.append(f"{'GCN BG error:':>30s}  {'n/a':>12s}  {nval('C','bg_error'):>12s}  {nval('E','bg_error'):>12s}")
    summary_lines.append(f"{'SAGE BG error:':>30s}  {'n/a':>12s}  {nval('D','bg_error'):>12s}  {nval('F','bg_error'):>12s}")
    summary_lines.append(f"{'GCN ordering loss:':>30s}  {'n/a':>12s}  {'n/a':>12s}  {e_ord:>12s}")
    summary_lines.append(f"{'SAGE ordering loss:':>30s}  {'n/a':>12s}  {'n/a':>12s}  {f_ord:>12s}")
    summary_lines.append(f"{'GCN spatial std:':>30s}  {nval('A','spatial_std','.4f',''):>12s}  {nval('C','spatial_std','.4f',''):>12s}  {nval('E','spatial_std','.4f',''):>12s}")
    summary_lines.append(f"{'SAGE spatial std:':>30s}  {nval('B','spatial_std','.4f',''):>12s}  {nval('D','spatial_std','.4f',''):>12s}  {nval('F','spatial_std','.4f',''):>12s}")
    summary_lines.append(f"{'GCN vs SAGE r:':>30s}  {ab.get('pearson_r', float('nan')):>12.3f}  {cd.get('pearson_r', float('nan')):>12.3f}  {ef.get('pearson_r', float('nan')):>12.3f}")
    summary_lines.append(f"{'GCN vs SAGE rho:':>30s}  {ab.get('spearman_rho', float('nan')):>12.3f}  {cd.get('spearman_rho', float('nan')):>12.3f}  {ef.get('spearman_rho', float('nan')):>12.3f}")
    summary_lines.append(f"{'GCN ordering compliance:':>30s}  {'n/a':>12s}  {gcn_no*100 if gcn_no is not None else float('nan'):>11.1f}%  {gcn_yes*100 if gcn_yes is not None else float('nan'):>11.1f}%")
    summary_lines.append(f"{'SAGE ordering compliance:':>30s}  {'n/a':>12s}  {sage_no*100 if sage_no is not None else float('nan'):>11.1f}%  {sage_yes*100 if sage_yes is not None else float('nan'):>11.1f}%")
    summary_lines.append(f"{'GCN appvalue rho:':>30s}  {appvalue_rho.get('A', float('nan')):>12.3f}  {appvalue_rho.get('C', float('nan')):>12.3f}  {appvalue_rho.get('E', float('nan')):>12.3f}")
    summary_lines.append(f"{'SAGE appvalue rho:':>30s}  {appvalue_rho.get('B', float('nan')):>12.3f}  {appvalue_rho.get('D', float('nan')):>12.3f}  {appvalue_rho.get('F', float('nan')):>12.3f}")

    with open(os.path.join(RESULTS_DIR, 'ordering_summary.txt'), 'w') as f:
        f.write('\n'.join(summary_lines) + '\n')
    log(f"Saved summary to {RESULTS_DIR}/ordering_summary.txt")

    elapsed = time.time() - start
    log(f"\nExperiment completed in {elapsed:.1f}s")


if __name__ == '__main__':
    main()
