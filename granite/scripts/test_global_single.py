"""
Quick test: Train on 5 diverse tracts, test on 1 holdout tract
"""
import sys
sys.path.insert(0, '/workspaces/GRANITE')

from granite.disaggregation.pipeline import GRANITEPipeline
from granite.models.gnn import set_random_seed
from granite.data.loaders import DataLoader

def quick_test():
    set_random_seed(42)
    
    loader = DataLoader()
    
    # Select 5 diverse training tracts manually
    # (You can check SVI values to ensure diversity)
    training_tracts = [
        '47065010433',  # Medium SVI (~0.45)
        '47065012400',  # Medium SVI (~0.41)
        '47065000600',  # Low SVI (~0.22)
        '47065010800',  # High SVI (~0.70)
        '47065001300'   # Very High SVI (~0.87)
    ]
    
    # Test on one separate tract
    test_tracts = ['47065011444'] 
    
    print(f"\n{'='*80}")
    print("QUICK GLOBAL TRAINING TEST")
    print(f"{'='*80}")
    print(f"Training tracts: {training_tracts}")
    print(f"Test tract: {test_tracts}")
    
    config = {
        'data': {
            'state_fips': '47',
            'county_fips': '065'
        },
        'model': {
            'epochs': 100,  # Reduced for quick test
            'hidden_dim': 64,
            'dropout': 0.3
        },
        'training': {
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'enforce_constraints': True,
            'constraint_weight': 1.0,
            'use_multitask': True
        },
        'processing': {
            'verbose': True,
            'enable_caching': True,
            'random_seed': 42
        }
    }
    
    pipeline = GRANITEPipeline(config, output_dir='./output/test_global')
    
    # Run global training
    results = pipeline.run_global_training(
        training_fips_list=training_tracts,
        test_fips_list=test_tracts
    )
    
    if results['success']:
        print(f"\n{'='*80}")
        print("TEST RESULTS")
        print(f"{'='*80}")
        
        test_result = results['test_results'][test_tracts[0]]
        print(f"\nTest Tract: {test_tracts[0]}")
        print(f"  Actual SVI: {test_result['actual_svi']:.4f}")
        print(f"  Predicted SVI: {test_result['predicted_mean']:.4f}")
        print(f"  Error: {test_result['mean_error_pct']:.2f}%")
        print(f"  Correlation: {test_result['correlation']:.3f}")
        
        if test_result['mean_error_pct'] < 30:
            print("\n✓ SUCCESS: Error < 30% on diverse training!")
        else:
            print("\n⚠ WARNING: Error > 30%, may need more training tracts")
    else:
        print(f"\n✗ FAILED: {results.get('error', 'Unknown error')}")
    
    return results

if __name__ == '__main__':
    results = quick_test()