# test_hybrid.py
from granite.disaggregation.pipeline import EnhancedGRANITEPipeline
import yaml

# Load config
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Run pipeline on single tract for testing
config['data']['processing_mode'] = 'fips'
config['data']['target_fips'] = '47065001100'  # Example tract

pipeline = EnhancedGRANITEPipeline(config=config, verbose=True)
results = pipeline.run()

# Check for improvement
if results['success']:
    comparison = results['tract_results'][0]['comparison']
    print(f"\nIDM Baseline CV: {comparison['idm_baseline']['cv']:.3f}")
    print(f"Hybrid IDM+GNN CV: {comparison['hybrid']['cv']:.3f}")
    print(f"Improvement Ratio: {comparison['improvement']['variation_ratio']:.2f}x")