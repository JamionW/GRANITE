#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example: Custom GRANITE Analysis

This script demonstrates how to use the GRANITE framework with custom
parameters and configurations for your specific analysis needs.
"""
import os
import sys
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from granite.disaggregation.pipeline import GRANITEPipeline
from granite.utils.config import ConfigManager
from granite.utils.logging import setup_logger, log_section
from granite.models.gnn import create_gnn_model
from granite.visualization.plots import DisaggregationVisualizer


def custom_gnn_model(input_dim):
    """
    Example of creating a custom GNN model
    
    You can define your own architecture here
    """
    return create_gnn_model(
        input_dim=input_dim,
        model_type='hybrid',  # Use hybrid model
        hidden_dim=128,       # Larger hidden dimension
        dropout=0.3           # More dropout
    )


def run_custom_analysis():
    """Run GRANITE with custom configuration"""
    
    # Setup logging
    logger = setup_logger(
        name='GRANITE-Custom',
        level='DEBUG',
        log_file='logs/custom_analysis.log'
    )
    
    log_section(logger, "Custom GRANITE Analysis", "=", 70)
    
    # Load and customize configuration
    config = ConfigManager('config/config.yaml')
    
    # Update configuration
    config.update({
        'model': {
            'epochs': 200,
            'learning_rate': 0.005,
            'hidden_dim': 128
        },
        'data': {
            'n_synthetic_addresses': 2000
        },
        'visualization': {
            'figure_size': [20, 15],
            'dpi': 150
        }
    })
    
    logger.info("Configuration updated:")
    logger.info(f"  - Epochs: {config.get('model.epochs')}")
    logger.info(f"  - Learning rate: {config.get('model.learning_rate')}")
    logger.info(f"  - Addresses: {config.get('data.n_synthetic_addresses')}")
    
    # Create pipeline with custom settings
    pipeline = GRANITEPipeline(
        data_dir=config.get('data.data_dir', './data'),
        output_dir='./output/custom_analysis',
        verbose=True
    )
    
    # Option 1: Run with custom configuration
    logger.info("\nRunning pipeline with custom configuration...")
    
    results = pipeline.run(
        roads_file=None,  # Download automatically
        epochs=config.get('model.epochs'),
        visualize=True
    )
    
    # Option 2: Run step by step for more control
    logger.info("\nAlternatively, you can run steps individually:")
    
    # Load data
    pipeline.load_data()
    
    # Customize data if needed
    # For example, filter to specific area
    if 'census_tracts' in pipeline.data:
        # Filter tracts by some criteria
        high_svi_tracts = pipeline.data['tracts_with_svi'][
            pipeline.data['tracts_with_svi']['RPL_THEMES'] > 0.75
        ]
        logger.info(f"Found {len(high_svi_tracts)} high vulnerability tracts")
    
    # Prepare graphs
    pipeline.prepare_graph_structures()
    
    # Train GNN with custom model
    input_dim = pipeline.data['pyg_data'].x.shape[1]
    custom_model = custom_gnn_model(input_dim)
    
    # You could also use a pre-trained model
    # custom_model.load_state_dict(torch.load('pretrained_model.pth'))
    
    # Learn features
    pipeline.results['gnn_model'] = custom_model
    pipeline.learn_accessibility_features(
        epochs=config.get('model.epochs'),
        learning_rate=config.get('model.learning_rate')
    )
    
    # Disaggregate
    pipeline.disaggregate_svi()
    
    # Custom validation
    pipeline.validate_results()
    
    # Additional analysis
    logger.info("\nPerforming additional custom analysis...")
    
    # Example: Analyze high-uncertainty areas
    high_uncertainty = pipeline.data['addresses'][
        pipeline.data['addresses']['svi_sd'] > 
        pipeline.data['addresses']['svi_sd'].quantile(0.9)
    ]
    
    logger.info(f"High uncertainty addresses: {len(high_uncertainty)}")
    logger.info(f"Average SVI in high uncertainty areas: "
                f"{high_uncertainty['svi_predicted'].mean():.3f}")
    
    # Save custom results
    high_uncertainty.to_csv('output/custom_analysis/high_uncertainty_addresses.csv')
    
    # Create custom visualization
    viz = DisaggregationVisualizer()
    
    # You can create individual plots
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Plot uncertainty vs SVI
    scatter = ax.scatter(
        pipeline.data['addresses']['svi_predicted'],
        pipeline.data['addresses']['svi_sd'],
        c=pipeline.data['addresses']['svi_predicted'],
        cmap='RdYlBu_r',
        alpha=0.6,
        s=20
    )
    
    ax.set_xlabel('Predicted SVI')
    ax.set_ylabel('Prediction Uncertainty (SD)')
    ax.set_title('SVI Prediction Uncertainty Analysis')
    plt.colorbar(scatter, ax=ax, label='SVI Score')
    
    plt.tight_layout()
    plt.savefig('output/custom_analysis/uncertainty_analysis.png', dpi=300)
    plt.close()
    
    # Generate metrics report
    from granite.utils.metrics import (
        calculate_disaggregation_metrics,
        create_metrics_report
    )
    
    if 'validation' in pipeline.results:
        val_df = pipeline.results['validation']
        metrics = calculate_disaggregation_metrics(
            val_df['true_svi'].values,
            val_df['predicted_avg'].values
        )
        
        report = create_metrics_report(metrics)
        logger.info("\n" + report)
        
        # Save report
        with open('output/custom_analysis/metrics_report.txt', 'w') as f:
            f.write(report)
    
    logger.info("\nCustom analysis complete!")
    logger.info(f"Results saved to: output/custom_analysis/")
    
    return results


def batch_analysis_example():
    """
    Example of running multiple analyses with different parameters
    """
    logger = setup_logger('GRANITE-Batch')
    log_section(logger, "Batch Analysis Example", "=", 70)
    
    # Define parameter grid
    param_grid = {
        'epochs': [50, 100, 200],
        'learning_rate': [0.001, 0.01, 0.1],
        'hidden_dim': [32, 64, 128]
    }
    
    results_summary = []
    
    # Run analyses
    for epochs in param_grid['epochs']:
        for lr in param_grid['learning_rate']:
            for hidden_dim in param_grid['hidden_dim']:
                
                logger.info(f"\nRunning: epochs={epochs}, lr={lr}, "
                           f"hidden_dim={hidden_dim}")
                
                # Create unique output directory
                output_dir = f"output/batch/e{epochs}_lr{lr}_h{hidden_dim}"
                
                # Run pipeline
                pipeline = GRANITEPipeline(
                    output_dir=output_dir,
                    verbose=False  # Less verbose for batch
                )
                
                try:
                    results = pipeline.run(
                        epochs=epochs,
                        visualize=False  # Skip viz for batch
                    )
                    
                    # Collect summary metrics
                    if 'validation' in results:
                        val_df = results['validation']
                        summary = {
                            'epochs': epochs,
                            'learning_rate': lr,
                            'hidden_dim': hidden_dim,
                            'mae': val_df['error'].mean(),
                            'correlation': val_df['true_svi'].corr(
                                val_df['predicted_avg']
                            )
                        }
                        results_summary.append(summary)
                        
                except Exception as e:
                    logger.error(f"Failed: {str(e)}")
    
    # Summarize results
    import pandas as pd
    summary_df = pd.DataFrame(results_summary)
    summary_df.to_csv('output/batch/batch_results_summary.csv', index=False)
    
    # Find best parameters
    best_idx = summary_df['mae'].idxmin()
    best_params = summary_df.iloc[best_idx]
    
    logger.info("\nBatch analysis complete!")
    logger.info(f"Best parameters: {best_params.to_dict()}")
    
    return summary_df


if __name__ == "__main__":
    # Run custom analysis
    print("Running custom GRANITE analysis...")
    results = run_custom_analysis()
    
    # Uncomment to run batch analysis
    # print("\nRunning batch analysis...")
    # batch_results = batch_analysis_example()