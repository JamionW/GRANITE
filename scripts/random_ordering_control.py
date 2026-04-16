"""
Random Ordering Control Experiment

Tests whether property value carries signal or acts purely as a consistent
symmetry breaker. Replaces log_appvalue with a fixed random vector and
compares cross-architecture convergence to BG-only and BG+appvalue conditions.
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
PRIOR_RESULTS = os.path.join(RESULTS_DIR, 'ordering_experiment.json')


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def load_config():
    with open('/workspaces/GRANITE/config.yaml') as f:
        return yaml.safe_load(f)


def load_data_and_features(config):
    """load addresses, compute features, build graph."""
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

    if rescale and block_group_targets:
        block_group_targets = rescale_block_group_svis(
            block_group_targets, bg_address_counts, tract_svi
        )

    return block_group_targets, block_group_masks, bg_address_counts


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
            'success': True,
        }
    except Exception as e:
        log(f"  {label} FAILED: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def compute_ordering_compliance(predictions, ordering_values, bg_masks, min_gap=0.5):
    """compute fraction of valid pairs where ordering is satisfied."""
    total_correct = 0
    total_pairs = 0

    for bg_id, mask in bg_masks.items():
        bg_preds = predictions[mask]
        bg_vals = ordering_values[mask]
        valid = ~np.isnan(bg_vals)
        if valid.sum() < 2:
            continue

        valid_preds = bg_preds[valid]
        valid_vals = bg_vals[valid]

        order = np.argsort(valid_vals)
        sorted_vals = valid_vals[order]
        sorted_preds = valid_preds[order]

        n = len(sorted_vals)
        for i in range(n):
            j_start = np.searchsorted(sorted_vals, sorted_vals[i] + min_gap)
            for j in range(j_start, n):
                total_pairs += 1
                if sorted_preds[i] > sorted_preds[j]:
                    total_correct += 1

    if total_pairs == 0:
        return 0.0, 0
    return total_correct / total_pairs, total_pairs


def load_prior_results():
    """load prior ordering experiment results for comparison columns."""
    if not os.path.exists(PRIOR_RESULTS):
        log(f"WARNING: prior results not found at {PRIOR_RESULTS}")
        return None

    with open(PRIOR_RESULTS) as f:
        return json.load(f)


def main():
    start = time.time()
    log("Random Ordering Control Experiment")
    log("=" * 60)

    # load prior results for comparison
    prior = load_prior_results()
    if prior is None:
        log("Cannot proceed without prior ordering experiment results")
        return

    config = load_config()

    # step 1: load shared data
    shared = load_data_and_features(config)
    graph_data = shared['graph_data']
    tract_svi = shared['tract_svi']
    feature_names = shared['feature_names']
    addresses = shared['addresses']
    n_addresses = graph_data.x.shape[0]

    # step 2: load BG data
    bg_targets, bg_masks, bg_addr_counts = \
        load_block_group_data_national(addresses, tract_svi, rescale=True)

    log(f"\nRescaled BG targets:")
    for bg_id in sorted(bg_targets.keys()):
        log(f"  {bg_id}: {bg_targets[bg_id]:.4f} ({bg_addr_counts[bg_id]} addresses)")

    # step 3: generate fixed random ordering vector
    log(f"\nGenerating random ordering vector (seed=42, n={n_addresses})...")
    rng = np.random.RandomState(42)
    random_ordering = rng.uniform(size=n_addresses).astype(np.float64)
    log(f"  range: [{random_ordering.min():.4f}, {random_ordering.max():.4f}], "
        f"mean={random_ordering.mean():.4f}, std={random_ordering.std():.4f}")

    # count valid pairs with random ordering
    total_random_pairs = 0
    bg_random_pair_counts = {}
    for bg_id, mask in bg_masks.items():
        bg_vals = random_ordering[mask]
        sorted_vals = np.sort(bg_vals)
        n_pairs = 0
        for i in range(len(sorted_vals)):
            j_start = np.searchsorted(sorted_vals, sorted_vals[i] + ORDERING_MIN_GAP)
            n_pairs += len(sorted_vals) - j_start
        bg_random_pair_counts[bg_id] = n_pairs
        total_random_pairs += n_pairs

    log(f"  Random vector valid pairs (gap >= {ORDERING_MIN_GAP}): {total_random_pairs}")
    for bg_id in sorted(bg_random_pair_counts.keys()):
        log(f"    {bg_id}: {bg_random_pair_counts[bg_id]} pairs")

    # step 4: run G and H (random ordering)
    log("\n--- TRAINING RUNS (2 configurations) ---")
    runs = {}

    # G: GCN, tract + BG + random ordering
    runs['G'] = run_training('G (GCN, tract+BG+random)', AccessibilitySVIGNN,
                              graph_data, tract_svi, feature_names,
                              block_group_targets=bg_targets,
                              block_group_masks=bg_masks,
                              ordering_values=random_ordering)

    # H: SAGE, tract + BG + random ordering
    runs['H'] = run_training('H (SAGE, tract+BG+random)', GraphSAGEAccessibilitySVIGNN,
                              graph_data, tract_svi, feature_names,
                              block_group_targets=bg_targets,
                              block_group_masks=bg_masks,
                              ordering_values=random_ordering)

    # step 5: compute metrics for random runs
    g = runs.get('G', {})
    h = runs.get('H', {})

    if not (g.get('success') and h.get('success')):
        log("One or both runs failed, cannot compute comparisons")
        return

    # cross-architecture correlation
    gh_r, _ = stats.pearsonr(g['predictions'], h['predictions'])
    gh_rho, _ = stats.spearmanr(g['predictions'], h['predictions'])

    # per-BG agreement
    bg_agreement_random = {}
    for bg_id in sorted(bg_masks.keys()):
        mask = bg_masks[bg_id]
        g_bg = g['predictions'][mask]
        h_bg = h['predictions'][mask]
        if len(g_bg) >= 3 and np.std(g_bg) > 1e-10 and np.std(h_bg) > 1e-10:
            r_val, _ = stats.pearsonr(g_bg, h_bg)
            bg_agreement_random[bg_id] = float(r_val)
        else:
            bg_agreement_random[bg_id] = None

    # random ordering compliance
    g_compliance, g_pairs = compute_ordering_compliance(
        g['predictions'], random_ordering, bg_masks, min_gap=ORDERING_MIN_GAP)
    h_compliance, h_pairs = compute_ordering_compliance(
        h['predictions'], random_ordering, bg_masks, min_gap=ORDERING_MIN_GAP)

    # extract prior values
    prior_bg_r = prior['correlations']['C_vs_D']['pearson_r']
    prior_bg_rho = prior['correlations']['C_vs_D']['spearman_rho']
    prior_app_r = prior['correlations']['E_vs_F']['pearson_r']
    prior_app_rho = prior['correlations']['E_vs_F']['spearman_rho']

    prior_bg_agreement = prior['bg_agreement']
    prior_c = prior['runs']['C']
    prior_d = prior['runs']['D']
    prior_e = prior['runs']['E']
    prior_f = prior['runs']['F']

    # step 6: print comparison table
    print("\n" + "=" * 90)
    print(f"   RANDOM ORDERING CONTROL: Tract {TARGET_FIPS}")
    print()

    print(f"   {'':>30s}  {'+BG (no ordering)':>18s}  {'+BG+appvalue':>14s}  {'+BG+random':>14s}")
    print(f"   {'':>30s}  {'-'*18:>18s}  {'-'*14:>14s}  {'-'*14:>14s}")
    print(f"   {'GCN vs SAGE r:':>30s}  {prior_bg_r:>18.3f}  {prior_app_r:>14.3f}  {gh_r:>14.3f}")
    print(f"   {'GCN vs SAGE rho:':>30s}  {prior_bg_rho:>18.3f}  {prior_app_rho:>14.3f}  {gh_rho:>14.3f}")
    print()

    print(f"   Per-BG agreement:")
    print(f"   {'GEOID':<16s}  {'+BG':>18s}  {'+appvalue':>14s}  {'+random':>14s}")
    for bg_id in sorted(bg_masks.keys()):
        bg_prior = prior_bg_agreement.get(bg_id, {}).get('+BG')
        bg_app = prior_bg_agreement.get(bg_id, {}).get('+BG+ordering')
        bg_rand = bg_agreement_random.get(bg_id)
        bg_str = f"{bg_prior:>18.3f}" if bg_prior is not None else f"{'n/a':>18s}"
        app_str = f"{bg_app:>14.3f}" if bg_app is not None else f"{'n/a':>14s}"
        rand_str = f"{bg_rand:>14.3f}" if bg_rand is not None else f"{'n/a':>14s}"
        print(f"   {bg_id:<16s}  {bg_str}  {app_str}  {rand_str}")
    print()

    print(f"   {'GCN tract error:':>30s}  {prior_c['tract_error']:>17.2f}%  {prior_e['tract_error']:>13.2f}%  {g['tract_error']:>13.2f}%")
    print(f"   {'SAGE tract error:':>30s}  {prior_d['tract_error']:>17.2f}%  {prior_f['tract_error']:>13.2f}%  {h['tract_error']:>13.2f}%")
    print()

    print(f"   {'GCN BG error:':>30s}  {prior_c['bg_error']:>17.2f}%  {prior_e['bg_error']:>13.2f}%  {g['bg_error']:>13.2f}%")
    print(f"   {'SAGE BG error:':>30s}  {prior_d['bg_error']:>17.2f}%  {prior_f['bg_error']:>13.2f}%  {h['bg_error']:>13.2f}%")
    print()

    print(f"   {'GCN spatial std:':>30s}  {prior_c['spatial_std']:>18.4f}  {prior_e['spatial_std']:>14.4f}  {g['spatial_std']:>14.4f}")
    print(f"   {'SAGE spatial std:':>30s}  {prior_d['spatial_std']:>18.4f}  {prior_f['spatial_std']:>14.4f}  {h['spatial_std']:>14.4f}")
    print()

    # ordering compliance for random
    prior_gcn_no = prior['compliance']['C']['fraction']
    prior_sage_no = prior['compliance']['D']['fraction']
    prior_gcn_yes = prior['compliance']['E']['fraction']
    prior_sage_yes = prior['compliance']['F']['fraction']
    print(f"   {'GCN ordering compliance:':>30s}  {prior_gcn_no*100:>17.1f}%  {prior_gcn_yes*100:>13.1f}%  {g_compliance*100:>13.1f}%")
    print(f"   {'SAGE ordering compliance:':>30s}  {prior_sage_no*100:>17.1f}%  {prior_sage_yes*100:>13.1f}%  {h_compliance*100:>13.1f}%")
    print("=" * 90)

    # step 7: interpretation
    print("\nInterpretation:")
    random_convergence = gh_r
    appvalue_convergence = prior_app_r
    bg_only_convergence = prior_bg_r

    random_improvement = random_convergence - bg_only_convergence
    appvalue_improvement = appvalue_convergence - bg_only_convergence

    print(f"  Convergence improvement over BG-only (r={bg_only_convergence:.3f}):")
    print(f"    +appvalue: +{appvalue_improvement:.3f} (to r={appvalue_convergence:.3f})")
    print(f"    +random:   +{random_improvement:.3f} (to r={random_convergence:.3f})")
    print()

    if abs(random_improvement - appvalue_improvement) < 0.05:
        print("  FINDING: Random ordering produces comparable convergence gain to property value.")
        print("  Property value does NOT carry meaningful signal; any consistent ordering vector")
        print("  breaks the symmetry equally well. The ordering loss acts as a pure regularizer.")
    elif random_improvement > appvalue_improvement * 0.7:
        print("  FINDING: Random ordering captures most of the convergence gain.")
        print("  Property value carries marginal additional signal beyond symmetry breaking.")
        print("  The ordering loss is primarily a regularizer with weak content dependence.")
    elif random_improvement > 0.05:
        print("  FINDING: Random ordering provides partial convergence gain, but property value")
        print("  adds substantial additional improvement. The ordering signal is partly content-")
        print("  dependent (property value matters) and partly structural (symmetry breaking).")
    else:
        print("  FINDING: Random ordering does NOT improve convergence. Property value carries")
        print("  genuine within-block-group signal that random noise cannot replicate.")

    # step 8: save results
    json_results = {
        'experiment': 'random_ordering_control',
        'tract': TARGET_FIPS,
        'seed': SEED,
        'hyperparameters': {
            'hidden_dim': HIDDEN_DIM, 'epochs': EPOCHS,
            'learning_rate': LEARNING_RATE, 'dropout': DROPOUT,
            'ordering_weight': ORDERING_WEIGHT,
            'ordering_min_gap': ORDERING_MIN_GAP,
            'ordering_margin': ORDERING_MARGIN,
        },
        'random_ordering': {
            'n_addresses': int(n_addresses),
            'total_valid_pairs': int(total_random_pairs),
            'bg_pair_counts': {k: int(v) for k, v in bg_random_pair_counts.items()},
        },
        'runs': {
            'G': {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                  for k, v in g.items() if k != 'predictions'},
            'H': {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                  for k, v in h.items() if k != 'predictions'},
        },
        'correlations': {
            'G_vs_H': {
                'description': 'GCN vs SAGE (+BG+random)',
                'pearson_r': float(gh_r),
                'spearman_rho': float(gh_rho),
            },
        },
        'bg_agreement_random': bg_agreement_random,
        'compliance': {
            'G': {'fraction': g_compliance, 'n_pairs': g_pairs},
            'H': {'fraction': h_compliance, 'n_pairs': h_pairs},
        },
        'comparison': {
            'bg_only_r': float(bg_only_convergence),
            'appvalue_r': float(appvalue_convergence),
            'random_r': float(random_convergence),
            'appvalue_improvement': float(appvalue_improvement),
            'random_improvement': float(random_improvement),
        },
        'predictions': {
            'G': g['predictions'].tolist(),
            'H': h['predictions'].tolist(),
        },
        'timestamp': datetime.now().isoformat(),
        'elapsed_seconds': time.time() - start,
    }

    # convert per_bg_errors and epochs_trained
    for k in ['G', 'H']:
        if 'per_bg_errors' in json_results['runs'][k]:
            json_results['runs'][k]['per_bg_errors'] = {
                bg: float(v) for bg, v in json_results['runs'][k]['per_bg_errors'].items()
            }
        if 'epochs_trained' in json_results['runs'][k]:
            json_results['runs'][k]['epochs_trained'] = int(json_results['runs'][k]['epochs_trained'])

    with open(os.path.join(RESULTS_DIR, 'random_ordering_control.json'), 'w') as f:
        json.dump(json_results, f, indent=2)
    log(f"\nSaved JSON to {RESULTS_DIR}/random_ordering_control.json")

    elapsed = time.time() - start
    log(f"\nExperiment completed in {elapsed:.1f}s")


if __name__ == '__main__':
    main()
