"""
National Block Group Constraint Convergence Experiment

Replaces county-ranked BG SVIs with nationally-ranked (and lightly rescaled)
BG SVIs. Compares results against prior tract-only and county-ranked conditions.
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

RESULTS_DIR = '/workspaces/GRANITE/results/convergence_experiment'

# prior experiment results (hard-coded from saved results)
PRIOR_TRACT_ONLY = {
    'gcn_tract_error': 8.26,
    'sage_tract_error': 6.71,
    'gcn_spatial_std': 0.1002,
    'sage_spatial_std': 0.1168,
    'gcn_sage_r': 0.420,
    'gcn_sage_rho': 0.327,
}
PRIOR_COUNTY_BG = {
    'gcn_tract_error': 5.01,
    'sage_tract_error': 1.87,
    'gcn_bg_error': 6.04,
    'sage_bg_error': 3.89,
    'gcn_spatial_std': 0.2052,
    'sage_spatial_std': 0.2056,
    'gcn_sage_r': 0.693,
    'gcn_sage_rho': 0.790,
    'rescale_shift': -0.1806,
    'bg1_rescaled_svi': 0.000,
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
    rescale_shift = None

    if rescale and block_group_targets:
        total = sum(bg_address_counts.values())
        pre_wm = sum(block_group_targets[g] * bg_address_counts[g] for g in block_group_targets) / total
        rescale_shift = tract_svi - pre_wm

        block_group_targets = rescale_block_group_svis(
            block_group_targets, bg_address_counts, tract_svi
        )

    return block_group_targets, block_group_masks, bg_data, raw_targets, rescale_shift, bg_address_counts


def run_training(label, model_class, graph_data, tract_svi, feature_names,
                 block_group_targets=None, block_group_masks=None):
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
            block_group_masks=block_group_masks
        )
        predictions = result['final_predictions'].flatten()
        log(f"  {label}: epochs={result['epochs_trained']}, loss={result['final_loss']:.6f}, "
            f"tract_err={result['overall_constraint_error']:.2f}%, "
            f"std={result['final_spatial_std']:.4f}")
        if 'bg_constraint_error' in result:
            log(f"  BG error: {result['bg_constraint_error']:.2f}%")
        return {
            'predictions': predictions,
            'epochs_trained': result['epochs_trained'],
            'final_loss': result['final_loss'],
            'tract_error': result['overall_constraint_error'],
            'bg_error': result.get('bg_constraint_error', None),
            'spatial_std': result['final_spatial_std'],
            'per_bg_errors': result.get('per_bg_errors', {}),
            'success': True,
        }
    except Exception as e:
        log(f"  {label} FAILED: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def compute_correlations(runs):
    """compute pairwise prediction correlations."""
    pairs = [
        ('A', 'B', 'GCN vs SAGE (tract-only)'),
        ('C', 'D', 'GCN vs SAGE (national BG)'),
        ('A', 'C', 'GCN: tract-only vs national BG'),
        ('B', 'D', 'SAGE: tract-only vs national BG'),
    ]
    results = {}
    for k1, k2, desc in pairs:
        r1 = runs.get(k1)
        r2 = runs.get(k2)
        if r1 and r1['success'] and r2 and r2['success']:
            p1 = r1['predictions']
            p2 = r2['predictions']
            pearson_r, _ = stats.pearsonr(p1, p2)
            spearman_rho, _ = stats.spearmanr(p1, p2)
            results[f'{k1}_vs_{k2}'] = {
                'description': desc,
                'pearson_r': float(pearson_r),
                'spearman_rho': float(spearman_rho),
            }
            log(f"  {desc}: r={pearson_r:.3f}, rho={spearman_rho:.3f}")
        else:
            results[f'{k1}_vs_{k2}'] = {'description': desc, 'error': 'one or both runs failed'}
    return results


def per_block_group_agreement(runs, block_group_masks):
    """compute within-block-group correlations between GCN and SAGE."""
    lines = []
    lines.append("")
    lines.append("PER-BLOCK-GROUP AGREEMENT (within-BG Pearson r)")
    lines.append(f"{'Block Group':<16s}  {'Addresses':>9s}  {'Unconstrained':>13s}  {'Constrained':>11s}")
    lines.append(f"{'-'*16:<16s}  {'-'*9:>9s}  {'-'*13:>13s}  {'-'*11:>11s}")

    for bg_id in sorted(block_group_masks.keys()):
        mask = block_group_masks[bg_id]
        n_addr = int(mask.sum())

        # unconstrained: A vs B
        unconstrained_r = 'n/a'
        if runs.get('A', {}).get('success') and runs.get('B', {}).get('success'):
            a_bg = runs['A']['predictions'][mask]
            b_bg = runs['B']['predictions'][mask]
            if len(a_bg) >= 3 and np.std(a_bg) > 0 and np.std(b_bg) > 0:
                r_val, _ = stats.pearsonr(a_bg, b_bg)
                unconstrained_r = f"{r_val:.3f}"

        # constrained: C vs D
        constrained_r = 'n/a'
        if runs.get('C', {}).get('success') and runs.get('D', {}).get('success'):
            c_bg = runs['C']['predictions'][mask]
            d_bg = runs['D']['predictions'][mask]
            if len(c_bg) >= 3 and np.std(c_bg) > 0 and np.std(d_bg) > 0:
                r_val, _ = stats.pearsonr(c_bg, d_bg)
                constrained_r = f"{r_val:.3f}"

        lines.append(f"{bg_id:<16s}  {n_addr:>9d}  {unconstrained_r:>13s}  {constrained_r:>11s}")

    return '\n'.join(lines)


def check_reproducibility(runs):
    """check if tract-only runs match prior experiment."""
    warnings = []
    if runs.get('A', {}).get('success'):
        a_err = runs['A']['tract_error']
        if abs(a_err - PRIOR_TRACT_ONLY['gcn_tract_error']) > 1.0:
            warnings.append(f"WARNING: GCN tract error {a_err:.2f}% differs from prior {PRIOR_TRACT_ONLY['gcn_tract_error']:.2f}% by >1%")
        a_std = runs['A']['spatial_std']
        if abs(a_std - PRIOR_TRACT_ONLY['gcn_spatial_std']) / PRIOR_TRACT_ONLY['gcn_spatial_std'] > 0.01:
            warnings.append(f"WARNING: GCN spatial std {a_std:.4f} differs from prior {PRIOR_TRACT_ONLY['gcn_spatial_std']:.4f} by >1%")
    if runs.get('B', {}).get('success'):
        b_err = runs['B']['tract_error']
        if abs(b_err - PRIOR_TRACT_ONLY['sage_tract_error']) > 1.0:
            warnings.append(f"WARNING: SAGE tract error {b_err:.2f}% differs from prior {PRIOR_TRACT_ONLY['sage_tract_error']:.2f}% by >1%")
        b_std = runs['B']['spatial_std']
        if abs(b_std - PRIOR_TRACT_ONLY['sage_spatial_std']) / PRIOR_TRACT_ONLY['sage_spatial_std'] > 0.01:
            warnings.append(f"WARNING: SAGE spatial std {b_std:.4f} differs from prior {PRIOR_TRACT_ONLY['sage_spatial_std']:.4f} by >1%")
    return warnings


def format_comparison_table(runs, correlations, rescale_shift, bg_targets_rescaled):
    """format the full three-condition comparison table."""
    lines = []
    lines.append(f"CONVERGENCE EXPERIMENT COMPARISON: Tract {TARGET_FIPS}")
    lines.append("")

    def nval(run_key, metric, fmt='.2f', suffix='%'):
        r = runs.get(run_key)
        if r and r['success'] and r.get(metric) is not None:
            return f"{r[metric]:{fmt}}{suffix}"
        return 'n/a'

    hdr = f"{'Condition':<22s}  {'Tract-Only':>12s}  {'County BG(rescaled)':>20s}  {'National BG(rescaled)':>22s}"
    sep = f"{'-'*22:<22s}  {'-'*12:>12s}  {'-'*20:>20s}  {'-'*22:>22s}"
    lines.append(hdr)
    lines.append(sep)

    lines.append(f"{'GCN tract error:':<22s}  {PRIOR_TRACT_ONLY['gcn_tract_error']:>11.2f}%  {PRIOR_COUNTY_BG['gcn_tract_error']:>19.2f}%  {nval('C','tract_error'):>22s}")
    lines.append(f"{'SAGE tract error:':<22s}  {PRIOR_TRACT_ONLY['sage_tract_error']:>11.2f}%  {PRIOR_COUNTY_BG['sage_tract_error']:>19.2f}%  {nval('D','tract_error'):>22s}")
    lines.append(f"{'GCN BG error:':<22s}  {'n/a':>12s}  {PRIOR_COUNTY_BG['gcn_bg_error']:>19.2f}%  {nval('C','bg_error'):>22s}")
    lines.append(f"{'SAGE BG error:':<22s}  {'n/a':>12s}  {PRIOR_COUNTY_BG['sage_bg_error']:>19.2f}%  {nval('D','bg_error'):>22s}")
    lines.append(f"{'GCN spatial std:':<22s}  {PRIOR_TRACT_ONLY['gcn_spatial_std']:>12.4f}  {PRIOR_COUNTY_BG['gcn_spatial_std']:>20.4f}  {nval('C','spatial_std','.4f',''):>22s}")
    lines.append(f"{'SAGE spatial std:':<22s}  {PRIOR_TRACT_ONLY['sage_spatial_std']:>12.4f}  {PRIOR_COUNTY_BG['sage_spatial_std']:>20.4f}  {nval('D','spatial_std','.4f',''):>22s}")

    # correlations
    cd = correlations.get('C_vs_D', {})
    cd_r = f"{cd['pearson_r']:.3f}" if 'pearson_r' in cd else 'n/a'
    cd_rho = f"{cd['spearman_rho']:.3f}" if 'spearman_rho' in cd else 'n/a'

    lines.append(f"{'GCN vs SAGE r:':<22s}  {PRIOR_TRACT_ONLY['gcn_sage_r']:>12.3f}  {PRIOR_COUNTY_BG['gcn_sage_r']:>20.3f}  {cd_r:>22s}")
    lines.append(f"{'GCN vs SAGE rho:':<22s}  {PRIOR_TRACT_ONLY['gcn_sage_rho']:>12.3f}  {PRIOR_COUNTY_BG['gcn_sage_rho']:>20.3f}  {cd_rho:>22s}")
    lines.append(f"{'Rescaling shift:':<22s}  {'n/a':>12s}  {PRIOR_COUNTY_BG['rescale_shift']:>20.4f}  {rescale_shift:>22.4f}")

    # find min rescaled BG SVI for comparison to county's 0.000
    min_bg_svi = min(bg_targets_rescaled.values()) if bg_targets_rescaled else float('nan')
    lines.append(f"{'BG1 rescaled SVI:':<22s}  {'n/a':>12s}  {PRIOR_COUNTY_BG['bg1_rescaled_svi']:>20.3f}  {min_bg_svi:>22.3f}")

    # cross-condition correlations
    lines.append("")
    ac = correlations.get('A_vs_C', {})
    bd = correlations.get('B_vs_D', {})
    ac_r = f"r={ac['pearson_r']:.3f}" if 'pearson_r' in ac else 'n/a'
    bd_r = f"r={bd['pearson_r']:.3f}" if 'pearson_r' in bd else 'n/a'
    lines.append(f"Cross-condition (GCN tract-only vs GCN+BG):             {ac_r}")
    lines.append(f"Cross-condition (SAGE tract-only vs SAGE+BG):            {bd_r}")

    return '\n'.join(lines)


def format_interpretation(runs, correlations):
    """generate interpretation text."""
    lines = []
    cd = correlations.get('C_vs_D', {})
    constrained_r = cd.get('pearson_r', None)
    c_tract_err = runs.get('C', {}).get('tract_error') if runs.get('C', {}).get('success') else None
    d_tract_err = runs.get('D', {}).get('tract_error') if runs.get('D', {}).get('success') else None
    c_bg_err = runs.get('C', {}).get('bg_error') if runs.get('C', {}).get('success') else None
    d_bg_err = runs.get('D', {}).get('bg_error') if runs.get('D', {}).get('success') else None

    if constrained_r is not None and c_tract_err is not None and d_tract_err is not None:
        avg_tract_err = (c_tract_err + d_tract_err) / 2
        avg_bg_err = None
        if c_bg_err is not None and d_bg_err is not None:
            avg_bg_err = (c_bg_err + d_bg_err) / 2

        if constrained_r > 0.5 and avg_tract_err < 10 and (avg_bg_err is None or avg_bg_err < 15):
            lines.append("National ranking produces well-calibrated hierarchical constraints. "
                         "Both architectures converge on a consistent spatial pattern while "
                         "satisfying constraints at both geographic scales.")
        elif constrained_r > 0.5:
            lines.append("Architectures converge but constraint satisfaction remains imperfect. "
                         "Weight tuning (bg_constraint_weight) may help.")
        else:
            lines.append("Block group constraints at this scale do not resolve within-tract "
                         "allocation ambiguity. The problem may require finer-grained supervision "
                         "(e.g., block-level constraints) or richer features.")

    # compare national vs county cross-condition correlations
    ac = correlations.get('A_vs_C', {})
    bd = correlations.get('B_vs_D', {})
    ac_r = ac.get('pearson_r', None)
    bd_r = bd.get('pearson_r', None)

    if ac_r is not None and bd_r is not None:
        lines.append("")
        if ac_r > 0.2 and bd_r > 0.2:
            lines.append("National ranking preserves the tract-only spatial pattern while "
                         "refining it, unlike county ranking which inverted it.")
        elif ac_r > 0.2 or bd_r > 0.2:
            lines.append(f"Cross-condition correlations are mixed (GCN: {ac_r:.3f}, SAGE: {bd_r:.3f}). "
                         "National ranking partially preserves the tract-only pattern.")
        else:
            lines.append(f"Cross-condition correlations remain low (GCN: {ac_r:.3f}, SAGE: {bd_r:.3f}). "
                         "National ranking still significantly restructures the spatial allocation.")

    return '\n'.join(lines)


def main():
    start = time.time()
    log("National BG Constraint Convergence Experiment")
    log("=" * 60)

    config = load_config()

    # step 1: load shared data
    shared = load_data_and_features(config)
    graph_data = shared['graph_data']
    tract_svi = shared['tract_svi']
    feature_names = shared['feature_names']
    addresses = shared['addresses']

    # step 2: pre-flight -- load national BG data and inspect
    log("\n--- PRE-FLIGHT: National BG SVI inspection ---")
    bg_targets, bg_masks, bg_data, raw_targets, rescale_shift, bg_addr_counts = \
        load_block_group_data_national(addresses, tract_svi, rescale=True)

    log(f"\nNationally-ranked BG SVIs for tract {TARGET_FIPS}:")
    log(f"{'Block Group':<16s}  {'Raw SVI':>8s}  {'Rescaled SVI':>12s}  {'Addresses':>9s}  {'At boundary?':>12s}")
    for bg_id in sorted(raw_targets.keys()):
        raw = raw_targets[bg_id]
        rescaled = bg_targets.get(bg_id, float('nan'))
        n_addr = bg_addr_counts.get(bg_id, 0)
        at_boundary = "YES" if (rescaled <= 0.001 or rescaled >= 0.999) else "no"
        log(f"{bg_id:<16s}  {raw:>8.4f}  {rescaled:>12.4f}  {n_addr:>9d}  {at_boundary:>12s}")

    log(f"\nRescale shift: {rescale_shift:+.4f}")
    n_bg_used = len(bg_targets)
    n_bg_addresses = sum(m.sum() for m in bg_masks.values())
    log(f"Block groups qualifying: {n_bg_used}")
    log(f"Total addresses covered: {n_bg_addresses}")

    # check pre-flight stop condition
    qualifying = sum(1 for bg_id in bg_targets
                     if bg_addr_counts.get(bg_id, 0) >= 5)
    if qualifying < 2:
        log("STOPPING: fewer than 2 block groups with svi_complete=True and >= 5 addresses.")
        return

    # step 3: run four training configurations
    log("\n--- TRAINING RUNS ---")
    runs = {}

    # A: GCN, no BG constraints
    runs['A'] = run_training('A (GCN, tract-only)', AccessibilitySVIGNN,
                              graph_data, tract_svi, feature_names)

    # B: SAGE, no BG constraints
    runs['B'] = run_training('B (SAGE, tract-only)', GraphSAGEAccessibilitySVIGNN,
                              graph_data, tract_svi, feature_names)

    # C: GCN, with national BG constraints
    runs['C'] = run_training('C (GCN, national BG)', AccessibilitySVIGNN,
                              graph_data, tract_svi, feature_names,
                              block_group_targets=bg_targets,
                              block_group_masks=bg_masks)

    # D: SAGE, with national BG constraints
    runs['D'] = run_training('D (SAGE, national BG)', GraphSAGEAccessibilitySVIGNN,
                              graph_data, tract_svi, feature_names,
                              block_group_targets=bg_targets,
                              block_group_masks=bg_masks)

    # check reproducibility of tract-only runs
    warnings = check_reproducibility(runs)
    for w in warnings:
        log(w)

    # step 4: pairwise correlations
    log("\n--- PAIRWISE CORRELATIONS ---")
    correlations = compute_correlations(runs)

    # step 5: per-block-group agreement
    bg_agreement = per_block_group_agreement(runs, bg_masks)
    print(bg_agreement)

    # step 6: comparison table
    comparison = format_comparison_table(runs, correlations, rescale_shift, bg_targets)
    print("\n" + "=" * 90)
    print(comparison)
    print("=" * 90)

    # step 7: interpretation
    interpretation = format_interpretation(runs, correlations)
    print(f"\nInterpretation:")
    print(interpretation)

    # step 8: save results
    os.makedirs(RESULTS_DIR, exist_ok=True)

    json_results = {
        'tract': TARGET_FIPS,
        'seed': SEED,
        'hyperparameters': {
            'hidden_dim': HIDDEN_DIM, 'epochs': EPOCHS,
            'learning_rate': LEARNING_RATE, 'dropout': DROPOUT,
        },
        'svi_ranking_scope': 'national',
        'rescaling': {
            'shift': float(rescale_shift) if rescale_shift is not None else None,
            'raw_targets': raw_targets,
            'rescaled_targets': {k: v for k, v in bg_targets.items()},
        },
        'block_groups': {
            'qualifying': n_bg_used,
            'addresses_covered': int(n_bg_addresses),
            'address_counts': bg_addr_counts,
        },
        'runs': {},
        'correlations': correlations,
        'prior_tract_only': PRIOR_TRACT_ONLY,
        'prior_county_bg': PRIOR_COUNTY_BG,
        'reproducibility_warnings': warnings,
        'timestamp': datetime.now().isoformat(),
        'elapsed_seconds': time.time() - start,
    }
    for label, r in runs.items():
        entry = {k: v for k, v in r.items() if k != 'predictions'}
        if r['success']:
            entry['predictions'] = r['predictions'].tolist()
            if 'per_bg_errors' in entry:
                entry['per_bg_errors'] = {k: float(v) for k, v in entry['per_bg_errors'].items()}
        json_results['runs'][label] = entry

    with open(os.path.join(RESULTS_DIR, 'national_bg_convergence.json'), 'w') as f:
        json.dump(json_results, f, indent=2)
    log(f"\nSaved JSON to {RESULTS_DIR}/national_bg_convergence.json")

    # save summary text
    summary_text = []
    summary_text.append(f"National BG Convergence Experiment: {datetime.now().isoformat()}")
    summary_text.append(f"Tract: {TARGET_FIPS}, Seed: {SEED}, Epochs: {EPOCHS}")
    summary_text.append(f"SVI ranking scope: national")
    summary_text.append(f"Rescale shift: {rescale_shift:+.4f}" if rescale_shift else "No rescaling")
    summary_text.append("")
    summary_text.append(comparison)
    summary_text.append("")
    summary_text.append(bg_agreement)
    summary_text.append("")
    summary_text.append("Interpretation:")
    summary_text.append(interpretation)

    with open(os.path.join(RESULTS_DIR, 'national_bg_summary.txt'), 'w') as f:
        f.write('\n'.join(summary_text) + '\n')
    log(f"Saved summary to {RESULTS_DIR}/national_bg_summary.txt")

    elapsed = time.time() - start
    log(f"\nExperiment completed in {elapsed:.1f}s")


if __name__ == '__main__':
    main()
