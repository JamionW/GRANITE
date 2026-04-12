"""Compare GCN vs SAGE architecture outputs for tract 47065000600."""

import os
import sys
import copy
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from granite.models.gnn import set_random_seed
from granite.disaggregation.pipeline import GRANITEPipeline

FIPS = '47065000600'
SEED = 42
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output', 'architecture_comparison')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_config():
    import yaml
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    with open(config_path) as f:
        config = yaml.safe_load(f)
    # single-tract mode
    config['data']['target_fips'] = FIPS
    config['data']['neighbor_tracts'] = 0
    config['processing']['random_seed'] = SEED
    config['processing']['skip_importance'] = True
    return config


def run_architecture(arch_name, config):
    """Run pipeline with a given architecture and return results."""
    cfg = copy.deepcopy(config)
    cfg['model']['architecture'] = arch_name
    set_random_seed(SEED)

    pipeline = GRANITEPipeline(cfg, output_dir=OUTPUT_DIR)
    results = pipeline.run()

    if not results.get('success'):
        print(f"ERROR: {arch_name} run failed: {results.get('error')}")
        sys.exit(1)

    # learned accessibility lives in training_result
    training_result = results.get('training_result', {})
    learned_acc = training_result.get('learned_accessibility')

    # predictions is a DataFrame with 'mean' column
    predictions_df = results.get('predictions')
    if predictions_df is not None and hasattr(predictions_df, 'values'):
        predictions = predictions_df['mean'].values
    else:
        predictions = training_result.get('raw_predictions')

    return learned_acc, predictions


def main():
    config = load_config()

    print("=" * 60)
    print("GRANITE Architecture Comparison: GCN vs SAGE")
    print(f"Tract: {FIPS}  Seed: {SEED}")
    print("=" * 60)

    # run gcn_gat
    print("\n--- Running GCN_GAT ---")
    gcn_acc, gcn_pred = run_architecture('gcn_gat', config)

    # run sage
    print("\n--- Running SAGE ---")
    sage_acc, sage_pred = run_architecture('sage', config)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    # stage 1: learned accessibility
    print("\n1. Stage 1: Learned Accessibility")
    if gcn_acc is not None and sage_acc is not None:
        gcn_acc = np.array(gcn_acc)
        sage_acc = np.array(sage_acc)

        # if multi-dimensional, compute per-address composite (mean across dims)
        if gcn_acc.ndim > 1:
            gcn_composite = gcn_acc.mean(axis=1)
            sage_composite = sage_acc.mean(axis=1)
            print(f"   Raw shape: GCN {gcn_acc.shape}, SAGE {sage_acc.shape}")
        else:
            gcn_composite = gcn_acc
            sage_composite = sage_acc

        print(f"   GCN  - mean: {gcn_composite.mean():.6f}, std: {gcn_composite.std():.6f}, "
              f"min: {gcn_composite.min():.6f}, max: {gcn_composite.max():.6f}")
        print(f"   SAGE - mean: {sage_composite.mean():.6f}, std: {sage_composite.std():.6f}, "
              f"min: {sage_composite.min():.6f}, max: {sage_composite.max():.6f}")

        acc_r, acc_p = stats.pearsonr(gcn_composite, sage_composite)
        print(f"   Pearson r: {acc_r:.4f} (p={acc_p:.2e})")

        np.save(os.path.join(OUTPUT_DIR, 'gcn_accessibility.npy'), gcn_acc)
        np.save(os.path.join(OUTPUT_DIR, 'sage_accessibility.npy'), sage_acc)
    else:
        print("   WARNING: learned accessibility not available for one or both architectures")
        acc_r = None

    # stage 2: SVI predictions
    print("\n2. Stage 2: SVI Predictions")
    gcn_pred = np.array(gcn_pred)
    sage_pred = np.array(sage_pred)

    print(f"   GCN  - mean: {gcn_pred.mean():.6f}, std: {gcn_pred.std():.6f}")
    print(f"   SAGE - mean: {sage_pred.mean():.6f}, std: {sage_pred.std():.6f}")

    pred_r, pred_p = stats.pearsonr(gcn_pred, sage_pred)
    pred_rho, rho_p = stats.spearmanr(gcn_pred, sage_pred)
    print(f"   Pearson r:  {pred_r:.4f} (p={pred_p:.2e})")
    print(f"   Spearman rho: {pred_rho:.4f} (p={rho_p:.2e})")

    np.save(os.path.join(OUTPUT_DIR, 'gcn_predictions.npy'), gcn_pred)
    np.save(os.path.join(OUTPUT_DIR, 'sage_predictions.npy'), sage_pred)

    # summary
    print("\n" + "=" * 60)
    if acc_r is not None:
        match = "yes" if acc_r > 0.9 else "no"
        print(f"Accessibility representations match: {match} (r={acc_r:.4f})")
    else:
        print("Accessibility representations match: UNKNOWN (data unavailable)")
    print(f"Prediction correlation: r={pred_r:.4f}, rho={pred_rho:.4f}")
    print("=" * 60)
    print(f"\nRaw vectors saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
