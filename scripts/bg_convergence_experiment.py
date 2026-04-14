"""
Block Group Constraint Convergence Experiment

Tests whether adding block group constraints forces GCN and GraphSAGE
toward the same spatial prediction pattern on tract 47065000600.

2x2 design: {GCN, SAGE} x {tract-only, tract+BG constraints}
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import geopandas as gpd
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
from granite.data.block_group_loader import BlockGroupLoader

TARGET_FIPS = '47065000600'
SEED = 42
HIDDEN_DIM = 64
EPOCHS = 200
LEARNING_RATE = 0.005
DROPOUT = 0.3

RESULTS_DIR = '/workspaces/GRANITE/results/convergence_experiment'


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def load_config():
    with open('/workspaces/GRANITE/config.yaml') as f:
        return yaml.safe_load(f)


def load_data_and_features(config):
    """Load addresses, compute features, build graph -- shared across all runs."""
    log("Loading spatial data...")
    data_loader = DataLoader('./data', config=config)

    # load tracts and SVI
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

    # load addresses
    addresses = data_loader.get_addresses_for_tract(TARGET_FIPS)
    if len(addresses) == 0:
        raise RuntimeError("No addresses found")
    addresses['tract_fips'] = TARGET_FIPS
    log(f"Loaded {len(addresses)} addresses")

    # destinations
    employment = data_loader.create_employment_destinations(use_real_data=True)
    healthcare = data_loader.create_healthcare_destinations(use_real_data=True)
    grocery = data_loader.create_grocery_destinations(use_real_data=True)

    # accessibility features via pipeline helper
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

    # normalize
    normalized_features, _ = normalize_accessibility_features(accessibility_features)

    # context features
    context_features = data_loader.create_context_features_for_addresses(
        addresses=addresses, svi_data=svi
    )
    normalized_context, _ = data_loader.normalize_context_features(context_features)

    # build graph
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


def load_block_group_data(addresses):
    """Load block group targets and masks for the target tract."""
    log("Loading block group data...")
    bg_loader = BlockGroupLoader(data_dir='./data', verbose=True)
    bg_data = bg_loader.get_block_groups_with_demographics(svi_ranking_scope='county')
    addresses_with_bg = bg_loader.assign_addresses_to_block_groups(addresses, bg_data)

    block_group_targets = {}
    block_group_masks = {}

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

    return block_group_targets, block_group_masks, bg_data


def run_training(label, model_class, graph_data, tract_svi, feature_names,
                 block_group_targets=None, block_group_masks=None):
    """Run a single training configuration and return results."""
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
    """Compute pairwise prediction correlations."""
    pairs = [
        ('A', 'B', 'GCN vs SAGE (tract-only)'),
        ('C', 'D', 'GCN vs SAGE (tract+BG)'),
        ('A', 'C', 'GCN: tract-only vs tract+BG'),
        ('B', 'D', 'SAGE: tract-only vs tract+BG'),
    ]
    results = {}
    for k1, k2, desc in pairs:
        r1 = runs.get(k1)
        r2 = runs.get(k2)
        if r1 and r1['success'] and r2 and r2['success']:
            p1 = r1['predictions']
            p2 = r2['predictions']
            pearson_r, pearson_p = stats.pearsonr(p1, p2)
            spearman_rho, spearman_p = stats.spearmanr(p1, p2)
            results[f'{k1}_vs_{k2}'] = {
                'description': desc,
                'pearson_r': float(pearson_r),
                'pearson_p': float(pearson_p),
                'spearman_rho': float(spearman_rho),
                'spearman_p': float(spearman_p),
            }
            log(f"  {desc}: r={pearson_r:.3f}, rho={spearman_rho:.3f}")
        else:
            results[f'{k1}_vs_{k2}'] = {'description': desc, 'error': 'one or both runs failed'}
    return results


def compute_per_bg_agreement(runs, block_group_masks):
    """Per-block-group correlation between GCN and SAGE."""
    table = []
    for bg_id, mask in block_group_masks.items():
        n_addr = mask.sum()
        row = {'block_group': bg_id, 'n_addresses': int(n_addr)}

        # unconstrained: A vs B
        a = runs.get('A')
        b = runs.get('B')
        if a and a['success'] and b and b['success'] and n_addr >= 3:
            r, _ = stats.pearsonr(a['predictions'][mask], b['predictions'][mask])
            row['unconstrained_r'] = float(r)
        else:
            row['unconstrained_r'] = None

        # constrained: C vs D
        c = runs.get('C')
        d = runs.get('D')
        if c and c['success'] and d and d['success'] and n_addr >= 3:
            r, _ = stats.pearsonr(c['predictions'][mask], d['predictions'][mask])
            row['constrained_r'] = float(r)
        else:
            row['constrained_r'] = None

        table.append(row)
    return table


def format_summary(runs, correlations, bg_table, n_bg_used, n_bg_total, n_addresses):
    """Format the summary table."""
    lines = []
    lines.append(f"CONVERGENCE EXPERIMENT: Tract {TARGET_FIPS}")
    lines.append(f"Block groups used: {n_bg_used} (of {n_bg_total} total), {n_addresses} addresses covered")
    lines.append("")
    lines.append(f"{'':30s} {'Tract-Only':>12s}    {'Tract+BG':>12s}")
    lines.append(f"{'':30s} {'----------':>12s}    {'--------':>12s}")

    def val(run_key, metric, fmt='.2f', suffix='%'):
        r = runs.get(run_key)
        if r and r['success'] and r.get(metric) is not None:
            return f"{r[metric]:{fmt}}{suffix}"
        return 'n/a'

    lines.append(f"{'GCN tract error:':30s} {val('A','tract_error'):>12s}    {val('C','tract_error'):>12s}")
    lines.append(f"{'SAGE tract error:':30s} {val('B','tract_error'):>12s}    {val('D','tract_error'):>12s}")
    lines.append(f"{'GCN BG error:':30s} {'n/a':>12s}    {val('C','bg_error'):>12s}")
    lines.append(f"{'SAGE BG error:':30s} {'n/a':>12s}    {val('D','bg_error'):>12s}")
    lines.append(f"{'GCN spatial std:':30s} {val('A','spatial_std','.4f',''):>12s}    {val('C','spatial_std','.4f',''):>12s}")
    lines.append(f"{'SAGE spatial std:':30s} {val('B','spatial_std','.4f',''):>12s}    {val('D','spatial_std','.4f',''):>12s}")

    ab = correlations.get('A_vs_B', {})
    cd = correlations.get('C_vs_D', {})
    ab_r = f"{ab['pearson_r']:.3f}" if 'pearson_r' in ab else 'n/a'
    cd_r = f"{cd['pearson_r']:.3f}" if 'pearson_r' in cd else 'n/a'
    ab_rho = f"{ab['spearman_rho']:.3f}" if 'spearman_rho' in ab else 'n/a'
    cd_rho = f"{cd['spearman_rho']:.3f}" if 'spearman_rho' in cd else 'n/a'

    lines.append(f"{'GCN vs SAGE r:':30s} {ab_r:>12s}    {cd_r:>12s}")
    lines.append(f"{'GCN vs SAGE rho:':30s} {ab_rho:>12s}    {cd_rho:>12s}")

    # interpretation
    lines.append("")
    constrained_r = cd.get('pearson_r', None)
    if constrained_r is not None:
        if constrained_r > 0.5:
            interp = "convergent: block group constraints reduce allocation ambiguity"
        elif constrained_r > 0.1:
            interp = "partial convergence: block group constraints provide some but insufficient supervision"
        else:
            interp = "still underdetermined: block group scale supervision does not resolve within-tract allocation"
        lines.append(f"Interpretation: {interp}")
    else:
        lines.append("Interpretation: cannot determine (run failure)")

    # per-block-group table
    lines.append("")
    lines.append("Per-Block-Group Agreement:")
    lines.append(f"{'GEOID':16s} {'n_addr':>8s} {'unconstrained_r':>16s} {'constrained_r':>14s}")
    for row in bg_table:
        ur = f"{row['unconstrained_r']:.3f}" if row['unconstrained_r'] is not None else 'n/a'
        cr = f"{row['constrained_r']:.3f}" if row['constrained_r'] is not None else 'n/a'
        lines.append(f"{row['block_group']:16s} {row['n_addresses']:>8d} {ur:>16s} {cr:>14s}")

    return '\n'.join(lines)


def main():
    start = time.time()
    log("Block Group Constraint Convergence Experiment")
    log("=" * 60)

    config = load_config()

    # Step 1: Load block group data and check prerequisites
    bg_targets, bg_masks, bg_data = load_block_group_data(
        # need addresses first -- load quickly
        DataLoader('./data', config=config).get_addresses_for_tract(TARGET_FIPS)
    )

    # check how many in the tract total
    tract_bgs = bg_data[bg_data['GEOID'].str.startswith(TARGET_FIPS)]
    n_bg_total = len(tract_bgs)
    n_bg_used = len(bg_targets)
    n_bg_addresses = sum(m.sum() for m in bg_masks.values())

    log(f"Block groups in tract: {n_bg_total} total, {n_bg_used} qualify (svi_complete + >=5 addresses)")
    log(f"Addresses covered by qualifying BGs: {n_bg_addresses}")

    if n_bg_used < 2:
        log("STOPPING: fewer than 2 qualifying block groups. Experiment needs multiple BGs.")
        return

    # Step 2: Load shared data and features
    shared = load_data_and_features(config)
    graph_data = shared['graph_data']
    tract_svi = shared['tract_svi']
    feature_names = shared['feature_names']
    addresses = shared['addresses']

    # Rebuild BG masks with the correct address set (same order as graph)
    bg_targets_final, bg_masks_final, _ = load_block_group_data(addresses)
    n_bg_used = len(bg_targets_final)
    n_bg_addresses = sum(m.sum() for m in bg_masks_final.values())
    log(f"Final BG targets: {n_bg_used} block groups, {n_bg_addresses} addresses")

    if n_bg_used < 2:
        log("STOPPING: fewer than 2 qualifying block groups after re-assignment.")
        return

    # Step 3: Run four training configurations
    runs = {}

    # A: GCN, no BG constraints
    runs['A'] = run_training('A (GCN, tract-only)', AccessibilitySVIGNN,
                              graph_data, tract_svi, feature_names,
                              block_group_targets=None, block_group_masks=None)

    # B: SAGE, no BG constraints
    runs['B'] = run_training('B (SAGE, tract-only)', GraphSAGEAccessibilitySVIGNN,
                              graph_data, tract_svi, feature_names,
                              block_group_targets=None, block_group_masks=None)

    # C: GCN, with BG constraints
    runs['C'] = run_training('C (GCN, tract+BG)', AccessibilitySVIGNN,
                              graph_data, tract_svi, feature_names,
                              block_group_targets=bg_targets_final,
                              block_group_masks=bg_masks_final)

    # D: SAGE, with BG constraints
    runs['D'] = run_training('D (SAGE, tract+BG)', GraphSAGEAccessibilitySVIGNN,
                              graph_data, tract_svi, feature_names,
                              block_group_targets=bg_targets_final,
                              block_group_masks=bg_masks_final)

    # Step 4: Correlations
    log("\nPairwise prediction correlations:")
    correlations = compute_correlations(runs)

    # Step 5: Per-BG agreement
    log("\nPer-block-group agreement:")
    bg_table = compute_per_bg_agreement(runs, bg_masks_final)

    # Step 6: Summary
    summary = format_summary(runs, correlations, bg_table,
                             n_bg_used, n_bg_total, n_bg_addresses)
    print("\n" + "=" * 70)
    print(summary)
    print("=" * 70)

    # Step 7: Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # JSON results (convert numpy arrays to lists)
    json_results = {
        'tract': TARGET_FIPS,
        'seed': SEED,
        'hyperparameters': {
            'hidden_dim': HIDDEN_DIM, 'epochs': EPOCHS,
            'learning_rate': LEARNING_RATE, 'dropout': DROPOUT,
        },
        'block_groups': {
            'total_in_tract': n_bg_total,
            'qualifying': n_bg_used,
            'addresses_covered': int(n_bg_addresses),
            'targets': {k: v for k, v in bg_targets_final.items()},
        },
        'runs': {},
        'correlations': correlations,
        'per_bg_agreement': bg_table,
        'timestamp': datetime.now().isoformat(),
        'elapsed_seconds': time.time() - start,
    }
    for label, r in runs.items():
        entry = {k: v for k, v in r.items() if k != 'predictions'}
        if r['success']:
            entry['predictions'] = r['predictions'].tolist()
            # drop numpy from per_bg_errors
            if 'per_bg_errors' in entry:
                entry['per_bg_errors'] = {k: float(v) for k, v in entry['per_bg_errors'].items()}
        json_results['runs'][label] = entry

    with open(os.path.join(RESULTS_DIR, 'bg_constraint_convergence.json'), 'w') as f:
        json.dump(json_results, f, indent=2)
    log(f"Saved JSON results to {RESULTS_DIR}/bg_constraint_convergence.json")

    with open(os.path.join(RESULTS_DIR, 'summary.txt'), 'w') as f:
        f.write(summary + '\n')
    log(f"Saved summary to {RESULTS_DIR}/summary.txt")

    elapsed = time.time() - start
    log(f"Experiment completed in {elapsed:.1f}s")


if __name__ == '__main__':
    main()
