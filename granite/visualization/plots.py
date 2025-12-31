"""
GRANITE Visualization (Spatial Version)

Simplified plotting for spatial disaggregation results.
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from typing import Dict, Optional, List
import os


class SpatialVisualizer:
    """
    Visualization suite for GRANITE spatial disaggregation.
    """
    
    def __init__(self, output_dir: str = './output'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Color scheme
        self.cmap = plt.cm.RdYlGn_r  # Red (high SVI) to Green (low SVI)
        self.method_colors = {
            'GNN': '#2E86AB',
            'IDW': '#A23B72', 
            'Kriging': '#F18F01',
            'Naive': '#95190C'
        }
    
    def create_summary_figure(self,
                              results: Dict,
                              tract_geometry=None,
                              save: bool = True) -> plt.Figure:
        """
        Create comprehensive summary figure with all key visualizations.
        
        Args:
            results: Pipeline results dict
            tract_geometry: Optional tract boundary for plotting
            save: Whether to save to file
        
        Returns:
            matplotlib Figure
        """
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        fig.suptitle(f"GRANITE Spatial Disaggregation Results\n"
                     f"Tract: {results.get('tract_fips', 'Unknown')} | "
                     f"SVI: {results.get('tract_svi', 0):.3f}", 
                     fontsize=14, fontweight='bold')
        
        addresses = results.get('addresses')
        predictions = results.get('predictions')
        
        # Panel 1: Spatial prediction map
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_spatial_predictions(ax1, addresses, predictions, tract_geometry)
        
        # Panel 2: Prediction distribution
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_prediction_distribution(ax2, results)
        
        # Panel 3: Method comparison (if baselines available)
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_method_comparison(ax3, results)
        
        # Panel 4: Training history
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_training_history(ax4, results.get('training_history', {}))
        
        # Panel 5: Feature correlations
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_feature_correlations(ax5, results.get('validation', {}))
        
        # Panel 6: GNN vs IDW scatter
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_gnn_vs_baseline(ax6, results)
        
        # Panel 7-9: Summary text
        ax7 = fig.add_subplot(gs[2, :])
        self._plot_summary_text(ax7, results)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'granite_summary.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Summary figure saved to: {filepath}")
        
        return fig
    
    def _plot_spatial_predictions(self, ax, addresses, predictions, tract_geometry=None):
        """Plot address-level predictions on map."""
        if addresses is None or predictions is None:
            ax.text(0.5, 0.5, 'No spatial data', ha='center', va='center')
            ax.set_title('Spatial Predictions')
            return
        
        x = addresses.geometry.x.values
        y = addresses.geometry.y.values
        
        scatter = ax.scatter(x, y, c=predictions, cmap=self.cmap, 
                            s=3, alpha=0.7, vmin=0, vmax=1)
        
        if tract_geometry is not None:
            # Plot tract boundary
            if hasattr(tract_geometry, 'exterior'):
                bx, by = tract_geometry.exterior.xy
                ax.plot(bx, by, 'k-', linewidth=1.5, label='Tract boundary')
        
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Predicted SVI by Address')
        plt.colorbar(scatter, ax=ax, label='Predicted SVI', shrink=0.8)
        ax.set_aspect('equal')
    
    def _plot_prediction_distribution(self, ax, results):
        """Plot histogram of predictions with tract SVI reference."""
        predictions = results.get('predictions')
        tract_svi = results.get('tract_svi', 0)
        
        if predictions is None:
            ax.text(0.5, 0.5, 'No predictions', ha='center', va='center')
            ax.set_title('Prediction Distribution')
            return
        
        ax.hist(predictions, bins=30, color=self.method_colors['GNN'], 
                alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.axvline(tract_svi, color='red', linestyle='--', linewidth=2, 
                   label=f'Tract SVI: {tract_svi:.3f}')
        ax.axvline(np.mean(predictions), color='blue', linestyle='-', linewidth=2,
                   label=f'Pred Mean: {np.mean(predictions):.3f}')
        
        ax.set_xlabel('Predicted SVI')
        ax.set_ylabel('Count')
        ax.set_title('Prediction Distribution')
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)
    
    def _plot_method_comparison(self, ax, results):
        """Bar chart comparing GNN, IDW, Kriging spatial variation."""
        baselines = results.get('baselines', {})
        gnn_std = results.get('summary', {}).get('spatial_variation', 0)
        
        methods = ['GNN']
        stds = [gnn_std]
        colors = [self.method_colors['GNN']]
        
        if 'idw' in baselines and 'std' in baselines['idw']:
            methods.append('IDW')
            stds.append(baselines['idw']['std'])
            colors.append(self.method_colors['IDW'])
        
        if 'kriging' in baselines and 'std' in baselines['kriging']:
            methods.append('Kriging')
            stds.append(baselines['kriging']['std'])
            colors.append(self.method_colors['Kriging'])
        
        # Add naive baseline (zero variation)
        methods.append('Naive')
        stds.append(0)
        colors.append(self.method_colors['Naive'])
        
        bars = ax.bar(methods, stds, color=colors, alpha=0.8, edgecolor='black')
        
        ax.set_ylabel('Spatial Std Dev')
        ax.set_title('Within-Tract Variation by Method')
        ax.set_ylim(0, max(stds) * 1.2 if max(stds) > 0 else 0.2)
        
        # Add value labels
        for bar, std in zip(bars, stds):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{std:.3f}', ha='center', va='bottom', fontsize=9)
    
    def _plot_training_history(self, ax, history):
        """Plot training loss and constraint error over epochs."""
        losses = history.get('losses', [])
        constraint_errors = history.get('constraint_errors', [])
        
        if not losses:
            ax.text(0.5, 0.5, 'No training history', ha='center', va='center')
            ax.set_title('Training History')
            return
        
        epochs = range(len(losses))
        
        ax.plot(epochs, losses, 'b-', label='Loss', linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        
        if constraint_errors:
            ax2 = ax.twinx()
            ax2.plot(epochs, constraint_errors, 'r--', label='Constraint %', linewidth=1.5)
            ax2.set_ylabel('Constraint Error %', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
        
        ax.set_title('Training Convergence')
        ax.legend(loc='upper right')
    
    def _plot_feature_correlations(self, ax, validation):
        """Plot feature-prediction correlations."""
        correlations = validation.get('feature_correlations', {})
        
        if not correlations:
            ax.text(0.5, 0.5, 'No feature correlations', ha='center', va='center')
            ax.set_title('Feature Correlations')
            return
        
        features = list(correlations.keys())
        corrs = list(correlations.values())
        
        # Color by sign
        colors = ['green' if c > 0 else 'red' for c in corrs]
        
        y_pos = np.arange(len(features))
        ax.barh(y_pos, corrs, color=colors, alpha=0.7, edgecolor='black')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features, fontsize=8)
        ax.set_xlabel('Correlation with Predictions')
        ax.set_title('Feature-Prediction Correlations')
        ax.axvline(0, color='black', linewidth=0.5)
        ax.set_xlim(-1, 1)
    
    def _plot_gnn_vs_baseline(self, ax, results):
        """Scatter plot comparing GNN predictions to IDW."""
        predictions = results.get('predictions')
        baselines = results.get('baselines', {})
        
        idw_preds = baselines.get('idw', {}).get('predictions')
        
        if predictions is None or idw_preds is None:
            ax.text(0.5, 0.5, 'No comparison data', ha='center', va='center')
            ax.set_title('GNN vs IDW')
            return
        
        ax.scatter(idw_preds, predictions, alpha=0.3, s=5, c=self.method_colors['GNN'])
        
        # Add diagonal reference
        lims = [0, 1]
        ax.plot(lims, lims, 'k--', alpha=0.5, label='y=x')
        
        # Compute correlation
        r = np.corrcoef(idw_preds, predictions)[0, 1]
        ax.text(0.05, 0.95, f'r = {r:.3f}', transform=ax.transAxes, 
                fontsize=10, verticalalignment='top')
        
        ax.set_xlabel('IDW Predictions')
        ax.set_ylabel('GNN Predictions')
        ax.set_title('GNN vs IDW Comparison')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
    
    def _plot_summary_text(self, ax, results):
        """Display summary statistics as text."""
        ax.axis('off')
        
        summary = results.get('summary', {})
        baselines = results.get('baselines', {})
        validation = results.get('validation', {})
        
        lines = [
            "=" * 80,
            "RESULTS SUMMARY",
            "=" * 80,
            "",
            f"Addresses processed: {summary.get('addresses_processed', 'N/A')}",
            f"Spatial features: {summary.get('spatial_features', 'N/A')}",
            "",
            "METHOD COMPARISON:",
            f"  {'Method':<12} {'Std Dev':<12} {'Constraint Err':<15}",
            f"  {'-'*40}",
            f"  {'GNN':<12} {summary.get('spatial_variation', 0):<12.4f} {summary.get('constraint_error', 0):<15.2f}%",
        ]
        
        if 'idw' in baselines and 'std' in baselines['idw']:
            lines.append(f"  {'IDW':<12} {baselines['idw']['std']:<12.4f} {baselines['idw'].get('constraint_error', 0):<15.2f}%")
        
        if 'kriging' in baselines and 'std' in baselines['kriging']:
            lines.append(f"  {'Kriging':<12} {baselines['kriging']['std']:<12.4f} {baselines['kriging'].get('constraint_error', 0):<15.2f}%")
        
        lines.extend([
            f"  {'Naive':<12} {'0.0000':<12} {'0.00':<15}%",
            "",
            "INTERPRETATION:",
        ])
        
        gnn_std = summary.get('spatial_variation', 0)
        idw_std = baselines.get('idw', {}).get('std', 0)
        
        if gnn_std > idw_std * 1.5:
            lines.append("  + GNN produces significantly MORE spatial variation than IDW")
        elif gnn_std > idw_std:
            lines.append("  ~ GNN produces slightly more variation than IDW")
        elif gnn_std > 0:
            lines.append("  - GNN produces less variation than IDW")
        else:
            lines.append("  ! No spatial variation detected")
        
        if validation.get('constraint_satisfied', False):
            lines.append("  + Tract mean constraint satisfied (< 1% error)")
        
        text = '\n'.join(lines)
        ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace')
    
    def plot_spatial_comparison(self,
                                addresses: gpd.GeoDataFrame,
                                gnn_predictions: np.ndarray,
                                idw_predictions: np.ndarray,
                                kriging_predictions: np.ndarray = None,
                                tract_geometry=None,
                                save: bool = True) -> plt.Figure:
        """
        Side-by-side spatial comparison of methods.
        """
        n_methods = 2 if kriging_predictions is None else 3
        fig, axes = plt.subplots(1, n_methods, figsize=(6*n_methods, 5))
        
        x = addresses.geometry.x.values
        y = addresses.geometry.y.values
        
        methods = [('GNN', gnn_predictions), ('IDW', idw_predictions)]
        if kriging_predictions is not None:
            methods.append(('Kriging', kriging_predictions))
        
        for ax, (name, preds) in zip(axes, methods):
            scatter = ax.scatter(x, y, c=preds, cmap=self.cmap, s=3, alpha=0.7, vmin=0, vmax=1)
            
            if tract_geometry is not None and hasattr(tract_geometry, 'exterior'):
                bx, by = tract_geometry.exterior.xy
                ax.plot(bx, by, 'k-', linewidth=1.5)
            
            ax.set_title(f'{name}\nstd={np.std(preds):.4f}')
            ax.set_aspect('equal')
            plt.colorbar(scatter, ax=ax, shrink=0.8)
        
        plt.suptitle('Spatial Disaggregation Comparison', fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'spatial_comparison.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Spatial comparison saved to: {filepath}")
        
        return fig


def generate_all_plots(results: Dict, output_dir: str = './output'):
    """
    Convenience function to generate all standard plots.
    
    Args:
        results: Pipeline results dictionary
        output_dir: Directory to save plots
    """
    viz = SpatialVisualizer(output_dir=output_dir)
    
    # Get tract geometry if available
    addresses = results.get('addresses')
    tract_geom = None
    
    # Main summary figure
    viz.create_summary_figure(results, tract_geometry=tract_geom, save=True)
    
    # Spatial comparison if baselines available
    baselines = results.get('baselines', {})
    if 'idw' in baselines and 'predictions' in baselines['idw']:
        viz.plot_spatial_comparison(
            addresses=addresses,
            gnn_predictions=results.get('predictions'),
            idw_predictions=baselines['idw']['predictions'],
            kriging_predictions=baselines.get('kriging', {}).get('predictions'),
            save=True
        )
    
    print(f"\nAll plots saved to: {output_dir}/")