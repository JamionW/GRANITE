"""
Visualization functions for GRANITE framework

This module provides plotting functionality for disaggregation results,
focusing on clear GNN vs IDM method comparison.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Dict, Optional, Union
import seaborn as sns
from matplotlib.patches import Rectangle, Circle 
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from scipy.interpolate import griddata  


class DisaggregationVisualizer:
    """Simplified visualizer focused on core GRANITE research findings"""
    
    def __init__(self, network_data=None):
        """Initialize visualizer with clean styling"""
        self.figsize = (12, 8)
        self.dpi = 300
        self.cmap = 'viridis_r'
        self.network_data = network_data
        
        plt.style.use('default')
        sns.set_palette("husl")
    
    def create_disaggregation_plot(self, predictions=None, data=None, 
                                  results=None, output_path=None, 
                                  comparison_results=None):
        """
        Create main disaggregation visualization - simplified version
        """
        # Handle different input formats
        if predictions is not None:
            pred_df = predictions
        elif results is not None and 'predictions' in results:
            pred_df = results['predictions']
        else:
            raise ValueError("No predictions provided")
            
        # Simple 2x2 layout for core information
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('GRANITE SVI Disaggregation Results', fontsize=16, fontweight='bold')
        
        # 1. Main predictions
        self._plot_predictions(axes[0, 0], pred_df, 'GNN-Whittle-Matérn Predictions')
        
        # 2. Uncertainty
        self._plot_uncertainty(axes[0, 1], pred_df, 'Prediction Uncertainty')
        
        # 3. Comparison if available
        if comparison_results is not None:
            comp_pred = comparison_results.get('predictions')
            if comp_pred is not None:
                self._plot_predictions(axes[1, 0], comp_pred, 'IDM Baseline')
                self._plot_method_comparison(axes[1, 1], pred_df, comp_pred)
            else:
                self._plot_summary_stats(axes[1, 0], pred_df)
                axes[1, 1].text(0.5, 0.5, 'No baseline comparison', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
        else:
            self._plot_summary_stats(axes[1, 0], pred_df)
            axes[1, 1].text(0.5, 0.5, 'Single method analysis', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def create_clear_method_comparison(self, gnn_predictions, idm_predictions, 
                                    gnn_results=None, idm_results=None, 
                                    output_path=None):
        """
        Create focused GNN vs IDM comparison - THE key visualization
        """
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # Extract data
        gnn_x = gnn_predictions['x'].values
        gnn_y = gnn_predictions['y'].values
        gnn_values = gnn_predictions['mean'].values
        
        idm_x = idm_predictions['x'].values
        idm_y = idm_predictions['y'].values
        idm_values = idm_predictions['mean'].values
        
        # Calculate key metrics
        gnn_std = np.std(gnn_values)
        idm_std = np.std(idm_values)
        variation_ratio = idm_std / gnn_std if gnn_std > 0 else float('inf')
        correlation = np.corrcoef(gnn_values, idm_values)[0, 1]
        
        # Main title with key finding
        fig.suptitle(f'GNN vs IDM Spatial Disaggregation Comparison\n'
                    f'Key Finding: IDM {variation_ratio:.1f}x More Spatially Variable (r={correlation:.3f})', 
                    fontsize=14, fontweight='bold')
        
        # Row 1: Spatial patterns
        # GNN spatial pattern
        ax1 = fig.add_subplot(gs[0, 0])
        scatter1 = ax1.scatter(gnn_x, gnn_y, c=gnn_values, cmap='viridis_r', 
                            s=12, alpha=0.8, edgecolors='white', linewidth=0.2)
        ax1.set_title(f'GNN-Whittle-Matérn\n(Learned Parameters)\nσ = {gnn_std:.6f}', 
                     fontweight='bold')
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        plt.colorbar(scatter1, ax=ax1, fraction=0.046, pad=0.04)
        ax1.set_aspect('equal')
        
        # IDM spatial pattern
        ax2 = fig.add_subplot(gs[0, 1])
        scatter2 = ax2.scatter(idm_x, idm_y, c=idm_values, cmap='viridis_r',
                            s=12, alpha=0.8, edgecolors='white', linewidth=0.2)
        ax2.set_title(f'IDM Fixed Coefficients\n(Land Cover Based)\nσ = {idm_std:.6f}', 
                     fontweight='bold')
        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
        plt.colorbar(scatter2, ax=ax2, fraction=0.046, pad=0.04)
        ax2.set_aspect('equal')
        
        # Difference map
        ax3 = fig.add_subplot(gs[0, 2])
        diff = gnn_values - idm_values
        vmax = max(abs(diff.min()), abs(diff.max()))
        scatter3 = ax3.scatter(gnn_x, gnn_y, c=diff, cmap='RdBu_r',
                            s=12, alpha=0.8, vmin=-vmax, vmax=vmax)
        ax3.set_title(f'Prediction Difference\n(GNN - IDM)\nMax Diff: ±{vmax:.4f}', 
                     fontweight='bold')
        ax3.set_xlabel('Longitude')
        ax3.set_ylabel('Latitude')
        plt.colorbar(scatter3, ax=ax3, fraction=0.046, pad=0.04)
        ax3.set_aspect('equal')
        
        # Row 2: Analysis
        # Method agreement scatter
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.scatter(idm_values, gnn_values, alpha=0.6, s=8, color='steelblue')
        min_val = min(gnn_values.min(), idm_values.min())
        max_val = max(gnn_values.max(), idm_values.max())
        ax4.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
        ax4.set_xlabel('IDM Predictions')
        ax4.set_ylabel('GNN Predictions')
        ax4.set_title(f'Method Agreement\nr = {correlation:.3f}', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Distribution comparison
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.hist(gnn_values, bins=25, alpha=0.6, label=f'GNN (σ={gnn_std:.4f})', 
                color='steelblue', density=True)
        ax5.hist(idm_values, bins=25, alpha=0.6, label=f'IDM (σ={idm_std:.4f})', 
                color='orange', density=True)
        ax5.set_xlabel('SVI Prediction Value')
        ax5.set_ylabel('Density')
        ax5.set_title('Value Distributions', fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Research summary
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        
        # Determine interpretation
        if variation_ratio > 5:
            interpretation = "IDM creates much more\nspatial detail"
            color = "orange"
        elif variation_ratio < 0.5:
            interpretation = "GNN creates more\nspatial detail"  
            color = "steelblue"
        else:
            interpretation = "Similar spatial\nvariation patterns"
            color = "lightgreen"
        
        if abs(correlation) > 0.5:
            agreement = "Methods agree\non patterns"
        elif abs(correlation) > 0.2:
            agreement = "Moderate agreement"
        else:
            agreement = "Methods disagree\non patterns"
        
        summary_text = f"""
RESEARCH FINDINGS

Spatial Variation:
• GNN: {gnn_std:.6f}
• IDM: {idm_std:.6f}
• Ratio: {variation_ratio:.1f}:1

Method Correlation: {correlation:.3f}

Interpretation:
{interpretation}

{agreement}

Recommendation:
{"Use IDM for spatial detail" if variation_ratio > 3 else "Consider hybrid approach"}
        """
        
        ax6.text(0.05, 0.95, summary_text.strip(), transform=ax6.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.8))
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def _plot_predictions(self, ax, predictions, title='Predictions'):
        """Plot spatial predictions with clean styling"""
        
        if predictions is None:
            ax.text(0.5, 0.5, 'No predictions available', 
                ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return
        
        # Handle different prediction formats (DataFrame vs dict)
        if isinstance(predictions, dict):
            # IDM results come as dict with 'predictions' key
            if 'predictions' in predictions:
                pred_df = predictions['predictions']
            else:
                ax.text(0.5, 0.5, 'Invalid IDM format', 
                    ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title)
                return
        else:
            pred_df = predictions
        
        if pred_df is None or (hasattr(pred_df, 'empty') and pred_df.empty):
            ax.text(0.5, 0.5, 'Empty predictions', 
                ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return
        
        # Extract coordinates - try multiple column name variations
        x = None
        y = None
        values = None
        
        # Try different coordinate column names
        for x_col in ['x', 'longitude', 'lon', 'lng']:
            if x_col in pred_df.columns:
                x = pred_df[x_col].values
                break
        
        for y_col in ['y', 'latitude', 'lat']:
            if y_col in pred_df.columns:
                y = pred_df[y_col].values
                break
        
        # Try different value column names
        for val_col in ['mean', 'predicted_svi', 'prediction', 'value']:
            if val_col in pred_df.columns:
                values = pred_df[val_col].values
                break
        
        # Add network background if available
        if self.network_data and 'edges_gdf' in self.network_data:
            self.network_data['edges_gdf'].plot(
                ax=ax, color='lightgray', linewidth=0.3, alpha=0.5, zorder=1
            )
        
        scatter = ax.scatter(x, y, c=values, cmap='viridis_r', s=8, alpha=0.8, 
                           edgecolors='white', linewidth=0.2, zorder=5)
        
        cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('SVI (Vulnerability)', rotation=270, labelpad=15)
        
        ax.set_title(title)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_xlim(x.min() - 0.001, x.max() + 0.001)
        ax.set_ylim(y.min() - 0.001, y.max() + 0.001)
        ax.set_aspect('equal')
    
    def _plot_uncertainty(self, ax, predictions, title='Uncertainty'):
        """Plot prediction uncertainty"""
        
        if predictions is None or predictions.empty:
            ax.text(0.5, 0.5, 'No uncertainty data', 
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
        
        # Add network background
        if self.network_data and 'edges_gdf' in self.network_data:
            self.network_data['edges_gdf'].plot(
                ax=ax, color='lightgray', linewidth=0.3, alpha=0.4, zorder=1
            )
        
        scatter = ax.scatter(x, y, c=uncertainty, cmap='Reds', s=12, alpha=0.8, 
                           edgecolors='white', linewidth=0.2, zorder=5)
        
        cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Uncertainty (SD)', rotation=270, labelpad=15)
        
        ax.set_title(title)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_xlim(x.min() - 0.001, x.max() + 0.001)
        ax.set_ylim(y.min() - 0.001, y.max() + 0.001)
        ax.set_aspect('equal')
    
    def _plot_method_comparison(self, ax, pred1, pred2):
        """Simple scatter plot comparing two methods"""
        
        if pred2 is None:
            ax.text(0.5, 0.5, 'No comparison data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Method Comparison')
            return
        
        # Extract GNN values
        values1 = pred1['mean'].values
        
        # Extract IDM values - handle different formats
        if isinstance(pred2, dict):
            if 'predictions' in pred2:
                pred2_df = pred2['predictions']
                # Try different column names for IDM values
                values2 = None
                for col in ['mean', 'predicted_svi', 'prediction', 'value']:
                    if col in pred2_df.columns:
                        values2 = pred2_df[col].values
                        break
                if values2 is None:
                    ax.text(0.5, 0.5, f'IDM values not found\nColumns: {list(pred2_df.columns)}', 
                           ha='center', va='center', transform=ax.transAxes)
                    return
            else:
                ax.text(0.5, 0.5, 'Invalid IDM format\n(no predictions key)', 
                       ha='center', va='center', transform=ax.transAxes)
                return
        else:
            values2 = pred2['mean'].values
        
        # Ensure same length
        min_len = min(len(values1), len(values2))
        if len(values1) != len(values2):
            print(f"WARNING: Length mismatch - GNN {len(values1)}, IDM {len(values2)}, using {min_len}")
            values1 = values1[:min_len]
            values2 = values2[:min_len]
        
        if min_len == 0:
            ax.text(0.5, 0.5, 'No data to compare', 
                ha='center', va='center', transform=ax.transAxes)
            return
        
        # Compute correlation
        try:
            correlation = np.corrcoef(values1, values2)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        except:
            correlation = 0.0
        
        # Scatter plot
        ax.scatter(values2, values1, alpha=0.6, s=8, color='steelblue')
        
        # Diagonal line
        min_val = min(np.min(values1), np.min(values2))
        max_val = max(np.max(values1), np.max(values2))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        
        ax.set_xlabel('IDM Predictions')
        ax.set_ylabel('GNN Predictions')
        ax.set_title(f'Method Comparison\nr = {correlation:.3f}')
        ax.grid(True, alpha=0.3)
    
    def _plot_summary_stats(self, ax, predictions):
        """Plot simple summary statistics"""
        ax.axis('off')
        
        values = predictions['mean'].values
        uncertainty = predictions.get('sd', predictions.get('uncertainty', np.array([])))
        
        summary_text = f"""
PREDICTION SUMMARY

Total Addresses: {len(values):,}

SVI Statistics:
  Mean: {np.mean(values):.4f}
  Std:  {np.std(values):.4f}
  Min:  {np.min(values):.4f}
  Max:  {np.max(values):.4f}
"""
        
        if len(uncertainty) > 0:
            summary_text += f"""
Uncertainty:
  Mean: {np.mean(uncertainty):.4f}
  Std:  {np.std(uncertainty):.4f}
"""
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    def plot_accessibility_gradients(self, predictions, network_data, 
                                   transit_data=None, output_path=None):
        """
        Simplified accessibility gradient visualization
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Transit Accessibility Gradients', fontsize=14, fontweight='bold')
        
        x = np.array(predictions.get('x', predictions.get('longitude', [])))
        y = np.array(predictions.get('y', predictions.get('latitude', [])))
        values = np.array(predictions.get('mean', predictions.get('predicted_svi', [])))
        
        # 1. Basic spatial gradient
        ax1 = axes[0, 0]
        if len(x) > 0 and len(y) > 0:
            try:
                grid_x, grid_y = np.mgrid[min(x):max(x):30j, min(y):max(y):30j]
                grid_values = griddata((x, y), values, (grid_x, grid_y), method='cubic')
                
                im = ax1.imshow(grid_values.T, extent=[min(x), max(x), min(y), max(y)],
                               origin='lower', cmap='viridis_r', alpha=0.7)
                ax1.scatter(x, y, c=values, cmap='viridis_r', s=4, edgecolors='white', linewidth=0.5)
                
                plt.colorbar(im, ax=ax1, label='Vulnerability Gradient')
                ax1.set_title('Spatial Accessibility Gradient')
                ax1.set_xlabel('Longitude')
                ax1.set_ylabel('Latitude')
            except:
                ax1.scatter(x, y, c=values, cmap='viridis_r', s=8)
                ax1.set_title('Spatial Pattern')
        
        # 2. Network centrality (simplified)
        ax2 = axes[0, 1]
        centrality = np.sqrt((x - x.mean())**2 + (y - y.mean())**2)  # Distance from center
        scatter2 = ax2.scatter(centrality, values, c=values, cmap='viridis_r', s=8, alpha=0.7)
        ax2.set_xlabel('Distance from Center')
        ax2.set_ylabel('Predicted SVI')
        ax2.set_title('Vulnerability vs Network Centrality')
        
        # 3. Value distribution
        ax3 = axes[1, 0]
        ax3.hist(values, bins=25, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.axvline(values.mean(), color='red', linestyle='--', linewidth=2)
        ax3.set_xlabel('Predicted SVI')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Value Distribution')
        ax3.grid(True, alpha=0.3)
        
        # 4. Simple clustering visualization
        ax4 = axes[1, 1]
        if len(values) > 10:
            from sklearn.cluster import KMeans
            features = np.column_stack([x, y, values])
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(features)
            
            scatter4 = ax4.scatter(x, y, c=clusters, cmap='Set1', s=8, alpha=0.8)
            centers = kmeans.cluster_centers_
            ax4.scatter(centers[:, 0], centers[:, 1], c='black', marker='x', 
                       s=100, linewidths=3)
            ax4.set_title('Accessibility Clusters')
            ax4.set_xlabel('Longitude')
            ax4.set_ylabel('Latitude')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_uncertainty_analysis(self, predictions, output_path=None):
        """
        Simplified uncertainty analysis
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Uncertainty Source Analysis', fontsize=14, fontweight='bold')
        
        x = np.array(predictions.get('x', predictions.get('longitude', [])))
        y = np.array(predictions.get('y', predictions.get('latitude', [])))
        uncertainty = np.array(predictions.get('sd', predictions.get('uncertainty', [])))
        values = np.array(predictions.get('mean', predictions.get('predicted_svi', [])))
        
        if len(uncertainty) == 0:
            for ax in axes.flat:
                ax.text(0.5, 0.5, 'No uncertainty data available', 
                    ha='center', va='center', transform=ax.transAxes)
            return
        
        # 1. Uncertainty distribution
        ax1 = axes[0, 0]
        ax1.hist(uncertainty, bins=25, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(uncertainty.mean(), color='red', linestyle='--', 
                   label=f'Mean: {uncertainty.mean():.4f}')
        ax1.set_xlabel('Uncertainty (SD)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Uncertainty Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Uncertainty vs prediction
        ax2 = axes[0, 1]
        scatter2 = ax2.scatter(values, uncertainty, c=uncertainty, cmap='Reds', s=8, alpha=0.7)
        ax2.set_xlabel('Predicted SVI')
        ax2.set_ylabel('Uncertainty')
        ax2.set_title('Uncertainty vs Prediction')
        plt.colorbar(scatter2, ax=ax2)
        
        # 3. Spatial uncertainty
        ax3 = axes[1, 0]
        scatter3 = ax3.scatter(x, y, c=uncertainty, cmap='Reds', s=8, alpha=0.8)
        plt.colorbar(scatter3, ax=ax3, label='Uncertainty')
        ax3.set_title('Spatial Uncertainty Pattern')
        ax3.set_xlabel('Longitude')
        ax3.set_ylabel('Latitude')
        ax3.set_aspect('equal')
        
        # 4. Confidence intervals
        ax4 = axes[1, 1]
        sort_idx = np.argsort(values)
        sorted_values = values[sort_idx]
        sorted_uncertainty = uncertainty[sort_idx]
        
        upper_bound = sorted_values + 1.96 * sorted_uncertainty
        lower_bound = sorted_values - 1.96 * sorted_uncertainty
        
        indices = np.arange(len(sorted_values))
        ax4.fill_between(indices, lower_bound, upper_bound, 
                        alpha=0.3, color='lightblue', label='95% CI')
        ax4.plot(indices, sorted_values, 'b-', linewidth=1, label='Predictions')
        
        ax4.set_xlabel('Sorted Address Index')
        ax4.set_ylabel('SVI Value')
        ax4.set_title('Prediction Confidence Intervals')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_model_interpretability(self, predictions, gnn_features=None, 
                                  network_data=None, output_path=None):
        """
        Simplified model interpretability dashboard
        """
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle('GRANITE Model Interpretability Dashboard', fontsize=14, fontweight='bold')
        
        x = np.array(predictions.get('x', predictions.get('longitude', [])))
        y = np.array(predictions.get('y', predictions.get('latitude', [])))
        values = np.array(predictions.get('mean', predictions.get('predicted_svi', [])))
        
        # 1. Feature importance
        ax1 = axes[0, 0]
        if gnn_features is not None:
            feature_std = np.std(gnn_features, axis=0)
            feature_names = ['κ (Precision)', 'α (Smoothness)', 'τ (Nugget)']
            bars = ax1.bar(feature_names, feature_std, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            ax1.set_title('GNN Feature Importance')
            ax1.set_ylabel('Standard Deviation')
            
            for bar, val in zip(bars, feature_std):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                        f'{val:.4f}', ha='center', va='bottom', fontsize=8)
        else:
            ax1.text(0.5, 0.5, 'No GNN features available', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('GNN Feature Importance')
        
        # 2. Feature correlations
        ax2 = axes[0, 1]
        if gnn_features is not None:
            corr_matrix = np.corrcoef(gnn_features.T)
            im = ax2.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
            ax2.set_xticks(range(3))
            ax2.set_yticks(range(3))
            ax2.set_xticklabels(['κ', 'α', 'τ'])
            ax2.set_yticklabels(['κ', 'α', 'τ'])
            ax2.set_title('Feature Correlations')
            
            for i in range(3):
                for j in range(3):
                    ax2.text(j, i, f'{corr_matrix[i,j]:.2f}', 
                           ha='center', va='center', fontsize=9)
            
            plt.colorbar(im, ax=ax2, fraction=0.046)
        else:
            ax2.text(0.5, 0.5, 'No features available', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Feature Correlations')
        
        # 3. Spatial predictions
        ax3 = axes[0, 2]
        if len(x) > 0:
            scatter3 = ax3.scatter(x, y, c=values, cmap='viridis_r', s=6, alpha=0.8)
            plt.colorbar(scatter3, ax=ax3, label='Predicted SVI')
            ax3.set_title('Predictions on Network Structure')
            ax3.set_xlabel('Longitude')
            ax3.set_ylabel('Latitude')
            ax3.set_aspect('equal')
        
        # 4. Prediction distribution
        ax4 = axes[1, 0]
        ax4.hist(values, bins=25, alpha=0.7, color='skyblue', edgecolor='black')
        ax4.axvline(values.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {values.mean():.4f}')
        ax4.set_xlabel('Predicted SVI')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Prediction Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Uncertainty vs complexity
        ax5 = axes[1, 1]
        uncertainty = np.array(predictions.get('sd', predictions.get('uncertainty', [])))
        if len(uncertainty) > 0:
            complexity = np.sqrt((x - x.mean())**2 + (y - y.mean())**2)
            scatter5 = ax5.scatter(complexity, uncertainty, c=values, 
                                 cmap='viridis', s=6, alpha=0.7)
            plt.colorbar(scatter5, ax=ax5, label='Predicted SVI')
            ax5.set_xlabel('Distance from Center')
            ax5.set_ylabel('Prediction Uncertainty')
            ax5.set_title('Uncertainty vs Spatial Complexity')
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'No uncertainty data', 
                    ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Uncertainty Analysis')
        
        # 6. Summary statistics
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        summary_text = f"""
MODEL SUMMARY

Predictions: {len(values):,}
Mean SVI: {np.mean(values):.4f}
Std SVI: {np.std(values):.4f}
Range: [{np.min(values):.4f}, {np.max(values):.4f}]
"""
        
        if len(uncertainty) > 0:
            summary_text += f"""
Mean Uncertainty: {np.mean(uncertainty):.4f}
Uncertainty Range: [{np.min(uncertainty):.4f}, {np.max(uncertainty):.4f}]
"""
        
        if gnn_features is not None:
            summary_text += f"""
GNN Features: {gnn_features.shape[1]}
Feature Variation:
  κ: {np.std(gnn_features[:, 0]):.4f}
  α: {np.std(gnn_features[:, 1]):.4f}
  τ: {np.std(gnn_features[:, 2]):.4f}
"""
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_uncertainty_sources(self, predictions, network_data=None, output_path=None):
        """
        Alias for plot_uncertainty_analysis - for pipeline compatibility
        """
        return self.plot_uncertainty_analysis(predictions, output_path)
    
    def set_network_data(self, network_data):
        """Set network data for background visualization"""
        self.network_data = network_data