"""
Visualization functions for GRANITE framework

This module provides plotting functionality for disaggregation results.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Dict, Optional
import seaborn as sns


class DisaggregationVisualizer:
    """Visualization class for GRANITE results"""
    
    def __init__(self):
        """Initialize visualizer with default settings"""
        self.figsize = (12, 8)
        self.dpi = 300
        self.cmap = 'viridis'
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
    
    def create_disaggregation_plot(self, data: Dict, results: Dict, 
                                  output_path: str = None):
        """
        Create main disaggregation visualization
        
        Parameters:
        -----------
        data : Dict
            Input data dictionary
        results : Dict
            Results dictionary
        output_path : str, optional
            Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle('GRANITE SVI Disaggregation Results', fontsize=16, fontweight='bold')
        
        # Plot 1: Original SVI by tract
        self._plot_tract_svi(axes[0, 0], data)
        
        # Plot 2: Disaggregated predictions
        self._plot_predictions(axes[0, 1], results)
        
        # Plot 3: Uncertainty map
        self._plot_uncertainty(axes[1, 0], results)
        
        # Plot 4: Summary statistics
        self._plot_summary(axes[1, 1], data, results)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def _plot_tract_svi(self, ax, data: Dict):
        """Plot original tract-level SVI"""
        if 'tracts_with_svi' in data:
            tracts = data['tracts_with_svi']
            tracts.plot(column='RPL_THEMES', ax=ax, cmap=self.cmap, 
                       legend=True, legend_kwds={'label': 'SVI Score'})
            ax.set_title('Original Tract-Level SVI')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
        else:
            ax.text(0.5, 0.5, 'No tract data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Original Tract-Level SVI')
    
    def _plot_predictions(self, ax, results: Dict):
        """Plot disaggregated predictions"""
        if 'predictions' in results:
            predictions = results['predictions']
            
            # Create scatter plot
            scatter = ax.scatter(
                predictions['longitude'], 
                predictions['latitude'],
                c=predictions.get('predicted_svi', predictions.get('mean', [])),
                cmap=self.cmap,
                s=10,
                alpha=0.6
            )
            
            plt.colorbar(scatter, ax=ax, label='Predicted SVI')
            ax.set_title('Disaggregated SVI Predictions')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
        else:
            ax.text(0.5, 0.5, 'No predictions available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Disaggregated SVI Predictions')
    
    def _plot_uncertainty(self, ax, results: Dict):
        """Plot prediction uncertainty"""
        if 'predictions' in results:
            predictions = results['predictions']
            
            if 'uncertainty' in predictions.columns or 'sd' in predictions.columns:
                uncertainty_col = 'uncertainty' if 'uncertainty' in predictions.columns else 'sd'
                
                scatter = ax.scatter(
                    predictions['longitude'], 
                    predictions['latitude'],
                    c=predictions[uncertainty_col],
                    cmap='Reds',
                    s=10,
                    alpha=0.6
                )
                
                plt.colorbar(scatter, ax=ax, label='Uncertainty')
                ax.set_title('Prediction Uncertainty')
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
            else:
                ax.text(0.5, 0.5, 'No uncertainty data', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Prediction Uncertainty')
        else:
            ax.text(0.5, 0.5, 'No predictions available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Prediction Uncertainty')
    
    def _plot_summary(self, ax, data: Dict, results: Dict):
        """Plot summary statistics"""
        summary_text = []
        
        # Data statistics
        if 'tracts_with_svi' in data:
            n_tracts = len(data['tracts_with_svi'])
            summary_text.append(f"Census tracts: {n_tracts}")
        
        if 'road_network' in data:
            n_nodes = data['road_network'].number_of_nodes()
            n_edges = data['road_network'].number_of_edges()
            summary_text.append(f"Road network: {n_nodes} nodes, {n_edges} edges")
        
        # Results statistics
        if 'predictions' in results:
            predictions = results['predictions']
            n_predictions = len(predictions)
            
            pred_col = 'predicted_svi' if 'predicted_svi' in predictions.columns else 'mean'
            if pred_col in predictions.columns:
                mean_svi = predictions[pred_col].mean()
                std_svi = predictions[pred_col].std()
                
                summary_text.append("")
                summary_text.append(f"Predictions: {n_predictions}")
                summary_text.append(f"Mean SVI: {mean_svi:.3f}")
                summary_text.append(f"Std SVI: {std_svi:.3f}")
        
        if 'validation' in results and isinstance(results['validation'], dict):
            summary_text.append("")
            summary_text.append("Validation metrics:")
            for key, value in results['validation'].items():
                if isinstance(value, (int, float)):
                    summary_text.append(f"  {key}: {value:.3f}")
        
        # Display summary
        ax.text(0.1, 0.9, '\n'.join(summary_text), 
               transform=ax.transAxes,
               verticalalignment='top',
               fontfamily='monospace',
               fontsize=10)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Summary Statistics')
    
    def plot_gnn_features(self, features: np.ndarray, output_path: str = None):
        """
        Plot learned GNN features (SPDE parameters)
        
        Parameters:
        -----------
        features : np.ndarray
            Learned SPDE parameters [n_nodes, 3]
        output_path : str, optional
            Path to save figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Learned SPDE Parameters', fontsize=14, fontweight='bold')
        
        param_names = ['Kappa (Precision)', 'Alpha (Smoothness)', 'Tau (Nugget)']
        
        for i, (ax, name) in enumerate(zip(axes, param_names)):
            ax.hist(features[:, i], bins=30, alpha=0.7, color=f'C{i}')
            ax.set_xlabel(name)
            ax.set_ylabel('Frequency')
            ax.set_title(f'{name} Distribution')
            
            # Add statistics
            mean_val = features[:, i].mean()
            std_val = features[:, i].std()
            ax.axvline(mean_val, color='red', linestyle='--', 
                      label=f'Mean: {mean_val:.3f}')
            ax.legend()
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()