"""
GRANITE Disaggregation Visualization Module
Professional visualizations focused on disaggregation quality metrics.

Key visualizations:
1. Constraint satisfaction across methods
2. Spatial variation comparison
3. Accessibility-SVI relationship
4. Baseline comparison dashboard
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict, Optional, List
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')


class DisaggregationVisualizer:
    """Visualization tools for GRANITE disaggregation results."""
    
    def __init__(self, style: str = 'default'):
        plt.style.use(style)
        self.figsize = (14, 10)
        self.dpi = 300
        
        # Color scheme
        self.colors = {
            'gnn': '#2E7D32',       # Dark green
            'idw': '#1565C0',       # Blue
            'kriging': '#7B1FA2',   # Purple
            'naive': '#757575',     # Gray
            'highlight': '#FF6F00'  # Orange
        }
    
    def plot_disaggregation_dashboard(self, 
                                      comparison_results: Dict,
                                      accessibility_features: np.ndarray = None,
                                      output_path: str = None) -> plt.Figure:
        """
        Create comprehensive disaggregation comparison dashboard.
        
        Args:
            comparison_results: Output from DisaggregationComparison.run_comparison()
            accessibility_features: Optional accessibility feature matrix
            output_path: Path to save figure
            
        Returns:
            matplotlib Figure object
        """
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
        
        methods = comparison_results['methods']
        tract_svi = comparison_results['tract_svi']
        
        # 1. Constraint Satisfaction Bar Chart
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_constraint_satisfaction(ax1, methods, tract_svi)
        
        # 2. Spatial Variation Comparison
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_variation_comparison(ax2, methods)
        
        # 3. Method Predictions Distribution
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_prediction_distributions(ax3, methods, tract_svi)
        
        # 4. Accessibility Correlation Comparison
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_accessibility_correlations(ax4, methods)
        
        # 5. GNN vs IDW Scatter
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_gnn_vs_baseline(ax5, methods, baseline='IDW_p2.0')
        
        # 6. Spatial Pattern (if coordinates available)
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_prediction_summary(ax6, methods, comparison_results)
        
        # 7. Summary Metrics Table
        ax7 = fig.add_subplot(gs[2, :])
        self._plot_metrics_table(ax7, comparison_results)
        
        # Title
        fig.suptitle(
            f"GRANITE Disaggregation Comparison\n"
            f"Tract {comparison_results['tract_fips']} | "
            f"Known SVI: {tract_svi:.4f} | "
            f"n={comparison_results['n_addresses']} addresses",
            fontsize=14, fontweight='bold', y=0.98
        )
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"Saved: {output_path}")
        
        return fig
    
    def _plot_constraint_satisfaction(self, ax, methods: Dict, tract_svi: float):
        """Bar chart of constraint satisfaction (mean error %)."""
        
        method_names = []
        errors = []
        colors = []
        
        for name in ['GNN', 'Naive_Uniform', 'IDW_p2.0', 'IDW_p3.0', 'Kriging']:
            if name in methods:
                method_names.append(name.replace('_', '\n'))
                errors.append(methods[name]['constraint_error_pct'])
                colors.append(self._get_method_color(name))
        
        bars = ax.bar(method_names, errors, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, err in zip(bars, errors):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{err:.1f}%', ha='center', va='bottom', fontsize=9)
        
        ax.set_ylabel('Constraint Error (%)', fontsize=10)
        ax.set_title('Constraint Satisfaction\n(Lower = Better)', fontsize=11, fontweight='bold')
        ax.axhline(5, color='green', linestyle='--', alpha=0.5, label='5% threshold')
        ax.set_ylim(0, max(errors) * 1.2 if errors else 10)
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_variation_comparison(self, ax, methods: Dict):
        """Bar chart comparing spatial variation (std)."""
        
        method_names = []
        stds = []
        colors = []
        
        for name in ['GNN', 'Naive_Uniform', 'IDW_p2.0', 'IDW_p3.0', 'Kriging']:
            if name in methods:
                method_names.append(name.replace('_', '\n'))
                stds.append(methods[name]['std'])
                colors.append(self._get_method_color(name))
        
        bars = ax.bar(method_names, stds, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, std in zip(bars, stds):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                   f'{std:.4f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_ylabel('Standard Deviation', fontsize=10)
        ax.set_title('Spatial Variation\n(Higher = More Disaggregation)', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_prediction_distributions(self, ax, methods: Dict, tract_svi: float):
        """KDE plots of prediction distributions."""
        
        for name in ['GNN', 'IDW_p2.0', 'Kriging', 'Naive_Uniform']:
            if name in methods:
                preds = methods[name]['predictions']
                color = self._get_method_color(name)
                label = name.replace('_', ' ')
                
                if np.std(preds) > 1e-6:
                    sns.kdeplot(preds, ax=ax, color=color, label=label, linewidth=2)
                else:
                    # Uniform distribution - show as vertical line
                    ax.axvline(np.mean(preds), color=color, linestyle='--', 
                              label=f'{label} (uniform)', linewidth=2)
        
        ax.axvline(tract_svi, color='red', linestyle='-', linewidth=2, 
                  label=f'Tract SVI ({tract_svi:.3f})')
        
        ax.set_xlabel('Predicted SVI', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title('Prediction Distributions', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
    
    def _plot_accessibility_correlations(self, ax, methods: Dict):
        """Bar chart of accessibility-SVI correlations."""
        
        method_names = []
        correlations = []
        colors = []
        
        for name in ['GNN', 'IDW_p2.0', 'IDW_p3.0', 'Kriging']:
            if name in methods:
                corr = methods[name].get('accessibility_correlation')
                if corr is not None:
                    method_names.append(name.replace('_', '\n'))
                    correlations.append(corr)
                    colors.append(self._get_method_color(name))
        
        if not correlations:
            ax.text(0.5, 0.5, 'No accessibility\ndata available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Accessibility Correlation', fontsize=11, fontweight='bold')
            return
        
        bars = ax.bar(method_names, correlations, color=colors, alpha=0.8, edgecolor='black')
        
        # Color bars by direction (negative = expected equity pattern)
        for bar, corr in zip(bars, correlations):
            if corr < -0.3:
                bar.set_edgecolor('green')
                bar.set_linewidth(2)
        
        ax.set_ylabel('Correlation (r)', fontsize=10)
        ax.set_title('Accessibility-SVI Correlation\n(Negative = Equity Pattern)', 
                    fontsize=11, fontweight='bold')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axhline(-0.3, color='green', linestyle='--', alpha=0.5, label='Strong equity threshold')
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_gnn_vs_baseline(self, ax, methods: Dict, baseline: str = 'IDW_p2.0'):
        """Scatter plot comparing GNN vs baseline predictions."""
        
        if 'GNN' not in methods or baseline not in methods:
            ax.text(0.5, 0.5, f'Missing GNN or {baseline}', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        gnn_preds = methods['GNN']['predictions']
        baseline_preds = methods[baseline]['predictions']
        
        ax.scatter(baseline_preds, gnn_preds, alpha=0.5, s=20, 
                  color=self.colors['gnn'], edgecolors='none')
        
        # Diagonal line
        lims = [
            min(min(baseline_preds), min(gnn_preds)),
            max(max(baseline_preds), max(gnn_preds))
        ]
        ax.plot(lims, lims, 'k--', alpha=0.5, label='1:1 line')
        
        # Correlation
        if len(gnn_preds) > 2:
            corr = np.corrcoef(baseline_preds, gnn_preds)[0, 1]
            ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel(f'{baseline} Predictions', fontsize=10)
        ax.set_ylabel('GNN Predictions', fontsize=10)
        ax.set_title(f'GNN vs {baseline}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_prediction_summary(self, ax, methods: Dict, results: Dict):
        """Summary statistics panel."""
        
        tract_svi = results['tract_svi']
        gnn = methods.get('GNN', {})
        
        summary_text = f"""Disaggregation Summary
{'='*30}

Target Tract SVI: {tract_svi:.4f}
Addresses: {results['n_addresses']:,}

GNN Disaggregation:
  Mean: {gnn.get('mean', 0):.4f}
  Std:  {gnn.get('std', 0):.4f}
  Range: {gnn.get('range', 0):.4f}
  Constraint Error: {gnn.get('constraint_error_pct', 0):.2f}%

Comparison:
  Variation vs Naive: {gnn.get('std', 0) - methods.get('Naive_Uniform', {}).get('std', 0):+.4f}
  Variation vs IDW: {gnn.get('std', 0) - methods.get('IDW_p2.0', methods.get('IDW_p2', {})).get('std', 0):+.4f}
"""
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        ax.axis('off')
        ax.set_title('Summary Statistics', fontsize=11, fontweight='bold')
    
    def _plot_metrics_table(self, ax, results: Dict):
        """Create metrics comparison table."""
        
        methods = results['methods']
        
        # Build table data
        columns = ['Method', 'Mean', 'Std', 'Range', 'Err %', 'Access r']
        data = []
        
        for name in ['GNN', 'Naive_Uniform', 'IDW_p2.0', 'IDW_p3.0', 'Kriging']:
            if name in methods:
                m = methods[name]
                acc_r = f"{m['accessibility_correlation']:.3f}" if m['accessibility_correlation'] else "N/A"
                data.append([
                    name,
                    f"{m['mean']:.4f}",
                    f"{m['std']:.4f}",
                    f"{m['range']:.4f}",
                    f"{m['constraint_error_pct']:.2f}",
                    acc_r
                ])
        
        ax.axis('off')
        
        # Create table
        table = ax.table(
            cellText=data,
            colLabels=columns,
            loc='center',
            cellLoc='center',
            colColours=['lightgray'] * len(columns)
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Highlight GNN row
        for i, cell in enumerate(table.get_celld().values()):
            if i < len(columns):  # Header row
                cell.set_text_props(fontweight='bold')
        
        ax.set_title('Complete Metrics Comparison', fontsize=12, fontweight='bold', pad=20)
    
    def _get_method_color(self, method_name: str) -> str:
        """Get color for method."""
        if 'GNN' in method_name:
            return self.colors['gnn']
        elif 'IDW' in method_name:
            return self.colors['idw']
        elif 'Kriging' in method_name:
            return self.colors['kriging']
        elif 'Naive' in method_name:
            return self.colors['naive']
        return 'gray'
    
    def plot_spatial_disaggregation(self, 
                                    address_gdf,
                                    predictions: np.ndarray,
                                    tract_svi: float,
                                    title: str = "GNN Disaggregation",
                                    output_path: str = None) -> plt.Figure:
        """
        Create spatial map of disaggregated predictions.
        
        Args:
            address_gdf: GeoDataFrame with address points
            predictions: Array of predicted SVI values
            tract_svi: Known tract SVI
            title: Plot title
            output_path: Path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Extract coordinates
        x = address_gdf.geometry.x.values
        y = address_gdf.geometry.y.values
        
        # Left: Predictions
        ax1 = axes[0]
        scatter = ax1.scatter(x, y, c=predictions, cmap='RdYlGn_r', 
                             s=30, alpha=0.7, edgecolors='none')
        plt.colorbar(scatter, ax=ax1, label='Predicted SVI')
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.set_title(f'{title}\n(Tract SVI: {tract_svi:.4f})', fontweight='bold')
        
        # Right: Deviation from tract mean
        ax2 = axes[1]
        deviations = predictions - tract_svi
        max_dev = max(abs(deviations.min()), abs(deviations.max()))
        scatter2 = ax2.scatter(x, y, c=deviations, cmap='coolwarm',
                              vmin=-max_dev, vmax=max_dev,
                              s=30, alpha=0.7, edgecolors='none')
        plt.colorbar(scatter2, ax=ax2, label='Deviation from Tract SVI')
        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
        ax2.set_title('Spatial Variation Pattern\n(Blue=Lower, Red=Higher)', fontweight='bold')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved: {output_path}")
        
        return fig


def create_disaggregation_visualizations(comparison_results: Dict,
                                         address_gdf = None,
                                         accessibility_features: np.ndarray = None,
                                         output_dir: str = './output') -> Dict:
    """
    Create all disaggregation visualizations.
    
    Args:
        comparison_results: Output from DisaggregationComparison
        address_gdf: Optional address GeoDataFrame for spatial plots
        accessibility_features: Optional accessibility features
        output_dir: Output directory for plots
        
    Returns:
        Dict of figure paths
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    viz = DisaggregationVisualizer()
    outputs = {}
    
    # Main dashboard
    dashboard_path = os.path.join(output_dir, 'disaggregation_comparison.png')
    viz.plot_disaggregation_dashboard(
        comparison_results,
        accessibility_features=accessibility_features,
        output_path=dashboard_path
    )
    outputs['dashboard'] = dashboard_path
    
    # Spatial map (if address data available)
    if address_gdf is not None and 'GNN' in comparison_results['methods']:
        spatial_path = os.path.join(output_dir, 'spatial_disaggregation.png')
        gnn_preds = comparison_results['methods']['GNN']['predictions']
        tract_svi = comparison_results['tract_svi']
        
        viz.plot_spatial_disaggregation(
            address_gdf=address_gdf,
            predictions=gnn_preds,
            tract_svi=tract_svi,
            output_path=spatial_path
        )
        outputs['spatial'] = spatial_path
    
    print(f"\nVisualizations saved to {output_dir}")
    return outputs