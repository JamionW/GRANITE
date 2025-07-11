"""
Visualization functions for GRANITE framework

This module provides plotting functionality for disaggregation results,
including comparisons between GNN-Whittle-Matérn and baseline methods.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Dict, Optional, Union
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize


class DisaggregationVisualizer:
    """Visualization class for GRANITE results with method comparison"""
    
    def __init__(self):
        """Initialize visualizer with default settings"""
        self.figsize = (12, 8)
        self.dpi = 300
        self.cmap = 'viridis'
        self.cmap_uncertainty = 'Reds'
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
    
    def create_disaggregation_plot(self, predictions=None, data=None, 
                                  results=None, output_path=None, 
                                  comparison_results=None):
        """
        Create main disaggregation visualization with method comparison
        
        Parameters:
        -----------
        predictions : pd.DataFrame, optional
            Direct predictions dataframe
        data : Dict, optional
            Input data dictionary (legacy)
        results : Dict, optional
            Results dictionary (legacy)
        output_path : str, optional
            Path to save figure
        comparison_results : Dict, optional
            Results from baseline method for comparison
        """
        # Handle different input formats
        if predictions is not None:
            pred_df = predictions
        elif results is not None and 'predictions' in results:
            pred_df = results['predictions']
        else:
            raise ValueError("No predictions provided")
            
        # Determine layout based on comparison availability
        if comparison_results is not None:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('GRANITE SVI Disaggregation: GNN-Whittle-Matérn vs Kriging', 
                        fontsize=16, fontweight='bold')
        else:
            fig, axes = plt.subplots(2, 2, figsize=self.figsize)
            axes_flat = axes.flatten()
            fig.suptitle('GRANITE SVI Disaggregation Results', 
                        fontsize=16, fontweight='bold')
        
        if comparison_results is not None:
            # Comparison layout
            # Row 1: Spatial predictions
            self._plot_predictions(axes[0, 0], pred_df, 
                                 title='GNN-Whittle-Matérn Predictions')
            
            comp_pred = comparison_results.get('predictions')
            if comp_pred is not None:
                self._plot_predictions(axes[0, 1], comp_pred,
                                     title='Ordinary Kriging Baseline')
                
                # Difference map
                self._plot_difference_map(axes[0, 2], pred_df, comp_pred)
            
            # Row 2: Analysis
            self._plot_method_comparison(axes[1, 0], pred_df, comp_pred)
            self._plot_uncertainty_comparison(axes[1, 1], pred_df, comp_pred)
            self._plot_comparison_summary(axes[1, 2], results, comparison_results)
        else:
            # Single method layout
            if data is not None and 'tracts' in data:
                self._plot_tract_svi(axes_flat[0], data)
            else:
                axes_flat[0].text(0.5, 0.5, 'Tract data not available', 
                                ha='center', va='center')
                axes_flat[0].set_title('Original Tract-Level SVI')
            
            self._plot_predictions(axes_flat[1], pred_df)
            self._plot_uncertainty(axes_flat[2], pred_df)
            self._plot_summary(axes_flat[3], data, results)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def _plot_predictions(self, ax, predictions, title='Disaggregated Predictions'):
        """Plot spatial predictions"""
        if predictions is None or predictions.empty:
            ax.text(0.5, 0.5, 'No predictions available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return
            
        # Get coordinates and values
        x = predictions.get('x', predictions.get('longitude', []))
        y = predictions.get('y', predictions.get('latitude', []))
        values = predictions.get('mean', predictions.get('predicted_svi', []))
        
        if len(x) == 0 or len(values) == 0:
            ax.text(0.5, 0.5, 'Invalid prediction data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return
        
        # Create scatter plot
        scatter = ax.scatter(x, y, c=values, cmap=self.cmap, 
                           s=20, alpha=0.8, edgecolors='none')
        
        plt.colorbar(scatter, ax=ax, label='Predicted SVI')
        ax.set_title(title)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_aspect('equal', adjustable='box')
    
    def _plot_uncertainty(self, ax, predictions, title='Prediction Uncertainty'):
        """Plot prediction uncertainty"""
        if predictions is None or predictions.empty:
            ax.text(0.5, 0.5, 'No uncertainty data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return
            
        x = predictions.get('x', predictions.get('longitude', []))
        y = predictions.get('y', predictions.get('latitude', []))
        uncertainty = predictions.get('sd', predictions.get('uncertainty', []))
        
        if len(uncertainty) == 0:
            ax.text(0.5, 0.5, 'No uncertainty estimates', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return
        
        scatter = ax.scatter(x, y, c=uncertainty, cmap=self.cmap_uncertainty,
                           s=20, alpha=0.8, edgecolors='none')
        
        plt.colorbar(scatter, ax=ax, label='Uncertainty (SD)')
        ax.set_title(title)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_aspect('equal', adjustable='box')
    
    def _plot_difference_map(self, ax, pred1, pred2):
        """Plot difference between two prediction methods"""
        if pred1 is None or pred2 is None:
            ax.text(0.5, 0.5, 'Cannot compute difference', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Prediction Difference')
            return
        
        # Align predictions by location
        x = pred1.get('x', pred1.get('longitude', []))
        y = pred1.get('y', pred1.get('latitude', []))
        
        values1 = pred1.get('mean', pred1.get('predicted_svi', []))
        values2 = pred2.get('mean', pred2.get('predicted_svi', []))
        
        if len(values1) != len(values2):
            ax.text(0.5, 0.5, 'Mismatched prediction sizes', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Prediction Difference')
            return
        
        # Compute difference
        diff = np.array(values1) - np.array(values2)
        
        # Create diverging colormap centered at 0
        vmax = np.abs(diff).max()
        scatter = ax.scatter(x, y, c=diff, cmap='RdBu_r', 
                           s=20, alpha=0.8, edgecolors='none',
                           vmin=-vmax, vmax=vmax)
        
        plt.colorbar(scatter, ax=ax, label='GNN-WM - Kriging')
        ax.set_title('Prediction Difference Map')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_aspect('equal', adjustable='box')
    
    def _plot_method_comparison(self, ax, pred1, pred2):
        """Scatter plot comparing two methods"""
        if pred1 is None or pred2 is None:
            ax.text(0.5, 0.5, 'Cannot compare methods', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Method Comparison')
            return
        
        values1 = np.array(pred1.get('mean', pred1.get('predicted_svi', [])))
        values2 = np.array(pred2.get('mean', pred2.get('predicted_svi', [])))
        
        # Scatter plot
        ax.scatter(values2, values1, alpha=0.5, s=10)
        
        # Add diagonal line
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
        
        # Add correlation
        if len(values1) > 0 and len(values2) > 0:
            corr = np.corrcoef(values1, values2)[0, 1]
            ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                   transform=ax.transAxes, va='top')
        
        ax.set_xlabel('Kriging Predictions')
        ax.set_ylabel('GNN-WM Predictions')
        ax.set_title('Method Comparison')
        ax.set_aspect('equal', adjustable='box')
    
    def _plot_uncertainty_comparison(self, ax, pred1, pred2):
        """Compare uncertainty between methods"""
        if pred1 is None or pred2 is None:
            ax.text(0.5, 0.5, 'Cannot compare uncertainty', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Uncertainty Comparison')
            return
        
        unc1 = np.array(pred1.get('sd', pred1.get('uncertainty', [])))
        unc2 = np.array(pred2.get('sd', pred2.get('uncertainty', [])))
        
        if len(unc1) == 0 or len(unc2) == 0:
            ax.text(0.5, 0.5, 'No uncertainty data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Uncertainty Comparison')
            return
        
        # Box plots
        data_to_plot = [unc1, unc2]
        bp = ax.boxplot(data_to_plot, labels=['GNN-WM', 'Kriging'])
        
        # Add mean values
        means = [np.mean(unc1), np.mean(unc2)]
        ax.scatter([1, 2], means, marker='o', s=100, c='red', zorder=3)
        
        # Add statistics
        ax.text(0.02, 0.98, f'Mean uncertainty:\nGNN-WM: {means[0]:.4f}\nKriging: {means[1]:.4f}',
               transform=ax.transAxes, va='top', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_ylabel('Uncertainty (SD)')
        ax.set_title('Uncertainty Comparison')
        ax.grid(True, alpha=0.3)
    
    def _plot_comparison_summary(self, ax, results1, results2):
        """Summary statistics comparison table"""
        ax.axis('off')
        
        # Extract diagnostics
        diag1 = results1.get('diagnostics', {}) if results1 else {}
        diag2 = results2.get('diagnostics', {}) if results2 else {}
        
        # Create comparison data
        metrics = [
            ('Method', 'GNN-Whittle-Matérn', 'Ordinary Kriging'),
            ('Constraint Satisfied', 
             '✓' if diag1.get('constraint_satisfied', False) else '✗',
             '✓' if diag2.get('constraint_satisfied', False) else '✗'),
            ('Constraint Error', 
             f"{diag1.get('constraint_error', 0):.2%}",
             f"{diag2.get('constraint_error', 0):.2%}"),
            ('Processing Time',
             f"{results1.get('timing', 0):.2f}s" if results1 else 'N/A',
             f"{results2.get('timing', 0):.2f}s" if results2 else 'N/A'),
        ]
        
        # Add prediction statistics if available
        if results1 and 'predictions' in results1:
            pred1 = results1['predictions']['mean']
            metrics.append(('Mean Prediction', f"{pred1.mean():.3f}", ''))
            metrics.append(('Std Prediction', f"{pred1.std():.3f}", ''))
        
        if results2 and 'predictions' in results2:
            pred2 = results2['predictions']['mean']
            if len(metrics) > 4:
                metrics[-2] = (metrics[-2][0], metrics[-2][1], f"{pred2.mean():.3f}")
                metrics[-1] = (metrics[-1][0], metrics[-1][1], f"{pred2.std():.3f}")
            else:
                metrics.append(('Mean Prediction', '', f"{pred2.mean():.3f}"))
                metrics.append(('Std Prediction', '', f"{pred2.std():.3f}"))
        
        # Create table
        table = ax.table(cellText=[m[1:] for m in metrics[1:]],
                        colLabels=['GNN-WM', 'Kriging'],
                        rowLabels=[m[0] for m in metrics[1:]],
                        cellLoc='center',
                        loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        ax.set_title('Method Comparison Summary', fontsize=12, pad=20)
    
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
    
    def _plot_summary(self, ax, data: Dict, results: Dict):
        """Plot summary statistics"""
        ax.axis('off')
        
        summary_text = "Summary Statistics\n" + "="*30 + "\n\n"
        
        if results and 'summary' in results:
            summary = results['summary']
            summary_text += f"Total addresses: {summary.get('total_addresses', 'N/A')}\n"
            summary_text += f"Mean SVI: {summary.get('mean_svi', 0):.3f}\n"
            summary_text += f"Std SVI: {summary.get('std_svi', 0):.3f}\n"
            summary_text += f"Mean uncertainty: {summary.get('mean_uncertainty', 0):.3f}\n"
        
        if results and 'diagnostics' in results:
            diag = results['diagnostics']
            summary_text += f"\nConstraint satisfied: {'Yes' if diag.get('constraint_satisfied') else 'No'}\n"
            summary_text += f"Constraint error: {diag.get('constraint_error', 0):.2%}\n"
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    def plot_gnn_features(self, gnn_features: np.ndarray, output_path: str = None):
        """
        Visualize learned GNN features (kappa, alpha, tau)
        
        Parameters:
        -----------
        gnn_features : np.ndarray
            Array of shape (n_nodes, 3) with kappa, alpha, tau values
        output_path : str, optional
            Path to save figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('GNN-Learned SPDE Parameters', fontsize=14, fontweight='bold')
        
        param_names = ['κ (kappa)', 'α (alpha)', 'τ (tau)']
        
        for i, (ax, param_name) in enumerate(zip(axes, param_names)):
            values = gnn_features[:, i]
            
            # Histogram
            ax.hist(values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax.axvline(values.mean(), color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {values.mean():.3f}')
            ax.axvline(np.median(values), color='green', linestyle='--', linewidth=2,
                      label=f'Median: {np.median(values):.3f}')
            
            ax.set_xlabel(param_name)
            ax.set_ylabel('Frequency')
            ax.set_title(f'Distribution of {param_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()