"""
Clean GRANITE Visualization Framework
Professional, focused visualizations without editorial content
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict, Optional, Tuple, List, Union
from scipy.stats import pearsonr, shapiro, probplot
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class GRANITEResearchVisualizer:
    """Clean, professional visualizations for GRANITE research"""
    
    def __init__(self):
        plt.style.use('default')
        self.figsize = (15, 10)
        self.dpi = 300
        
        # Feature names mapping - clear and descriptive
        self.feature_names = {
            'F1': 'Employment_Travel_Time',
            'F2': 'Healthcare_Travel_Time', 
            'F3': 'Grocery_Travel_Time',
            'F4': 'Employment_Count',
            'F5': 'Healthcare_Count',
            'F6': 'Grocery_Count',
            'F7': 'Employment_Gravity',
            'F8': 'Healthcare_Gravity',
            'F9': 'Grocery_Gravity'
        }
        
        # Quality thresholds
        self.excellent_r2 = 0.64
        self.good_r2 = 0.36
        self.moderate_r2 = 0.16

    def create_comprehensive_research_analysis(self, results: Dict, output_dir: str,
                                                diagnostics: bool = False):
        """Create clean research analysis without editorial content.

        Args:
            results: Research results dictionary
            output_dir: Base output directory
            diagnostics: If True, generate stage1/stage2/statistical_summary plots.
                         Default False (only spatial_analysis.png is produced).
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        if diagnostics:
            # 1. Stage 1 Validation
            diag_dir = os.path.join(output_dir, 'visualizations')
            os.makedirs(diag_dir, exist_ok=True)

            stage1_path = os.path.join(diag_dir, 'stage1_validation.png')
            self.plot_accessibility_learning_validation(
                learned_accessibility=results.get('learned_accessibility'),
                traditional_accessibility=results.get('traditional_accessibility'),
                output_path=stage1_path
            )

            # 2. Stage 2 Validation
            stage2_path = os.path.join(diag_dir, 'stage2_validation.png')
            self.plot_svi_prediction_validation(
                gnn_predictions=results.get('gnn_predictions'),
                stage2_metrics=results.get('stage2_metrics', {}),
                tract_svi=results.get('tract_svi', None),
                output_path=stage2_path
            )

            # 3. Statistical Summary
            stats_path = os.path.join(diag_dir, 'statistical_summary.png')
            self.plot_statistical_summary(
                results=results,
                output_path=stats_path
            )

        # 4. Spatial Analysis - always generated
        spatial_path = os.path.join(output_dir, 'spatial_analysis.png')
        self.plot_spatial_analysis(
            gnn_predictions=results.get('gnn_predictions'),
            learned_accessibility=results.get('learned_accessibility'),
            output_path=spatial_path,
            tract_svi=results.get('tract_svi'),
            tract_fips=results.get('tract_fips')
        )

    def plot_accessibility_learning_validation(self, learned_accessibility: np.ndarray,
                                            traditional_accessibility: Optional[np.ndarray],
                                            output_path: str):
        """Stage 1: Does GNN learn meaningful accessibility patterns?

        deprecated: retained for backward compatibility with existing callers.
        """
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Stage 1: Accessibility Learning Validation', fontsize=14, fontweight='bold')
        
        if learned_accessibility is None:
            fig.text(0.5, 0.5, 'No learned accessibility data available', 
                    ha='center', va='center', fontsize=16)
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            return
        
        # Ensure 2D array
        if learned_accessibility.ndim == 1:
            learned_accessibility = learned_accessibility.reshape(-1, 1)
        
        n_addresses, n_features = learned_accessibility.shape
        
        # 1. Learning Validation: Learned vs Traditional
        ax1 = axes[0, 0]
        if traditional_accessibility is not None and traditional_accessibility.shape[0] == n_addresses:
            # Calculate correlation
            if traditional_accessibility.ndim > 1:
                traditional_mean = np.mean(traditional_accessibility, axis=1)
            else:
                traditional_mean = traditional_accessibility
            learned_mean = np.mean(learned_accessibility, axis=1)
            
            correlation, p_value = pearsonr(learned_mean, traditional_mean)
            r_squared = max(0, correlation**2)
            
            # Scatter plot
            ax1.scatter(traditional_mean, learned_mean, alpha=0.6, s=20, color='steelblue')
            
            # Regression line
            z = np.polyfit(traditional_mean, learned_mean, 1)
            p = np.poly1d(z)
            ax1.plot(sorted(traditional_mean), p(sorted(traditional_mean)), "r--", alpha=0.8)
            
            ax1.set_xlabel('Traditional Accessibility')
            ax1.set_ylabel('GNN Learned Accessibility')
            ax1.set_title(f'Learning Validation\nR² = {r_squared:.3f}, p = {p_value:.4f}')
            ax1.grid(True, alpha=0.3)
            
            # Quality assessment
            quality = self._interpret_r2(r_squared)
            color = self._get_r2_color(r_squared)
            ax1.text(0.05, 0.95, f'{quality}\nLearning', transform=ax1.transAxes,
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.7),
                    verticalalignment='top', fontweight='bold')
        else:
            ax1.text(0.5, 0.5, 'No traditional\nbaseline available', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Learning Validation\n(No Baseline)')
            r_squared = 0
        
        # 2. Feature Correlation Matrix with meaningful names
        ax2 = axes[0, 1]
        if n_features > 1:
            corr_matrix = np.corrcoef(learned_accessibility.T)
            im = ax2.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
            ax2.set_title('Feature Independence Matrix')
            
            # Use meaningful feature names
            feature_labels = []
            for i in range(min(n_features, 9)):  # Limit to 9 features
                feature_name = self.feature_names.get(f'F{i+1}', f'F{i+1}')
                # Shorten for display
                short_name = feature_name.split('_')[0][:4] + '_' + feature_name.split('_')[-1][:2]
                feature_labels.append(short_name)
            
            ax2.set_xticks(range(len(feature_labels)))
            ax2.set_yticks(range(len(feature_labels)))
            ax2.set_xticklabels(feature_labels, rotation=45, ha='right', fontsize=8)
            ax2.set_yticklabels(feature_labels, fontsize=8)
            
            # Add correlation values
            for i in range(len(feature_labels)):
                for j in range(len(feature_labels)):
                    if i != j:  # Skip diagonal
                        color = 'white' if abs(corr_matrix[i, j]) > 0.5 else 'black'
                        ax2.text(j, i, f'{corr_matrix[i, j]:.2f}',
                               ha="center", va="center", color=color, fontsize=7)
            
            plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        else:
            ax2.text(0.5, 0.5, 'Single Feature\nNo Correlation\nAnalysis', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Feature Independence')
        
        # 3. Feature Diversity Analysis
        ax3 = axes[0, 2]
        feature_stds = np.std(learned_accessibility, axis=0)
        n_display = min(len(feature_stds), 9)
        
        x_positions = range(n_display)
        bars = ax3.bar(x_positions, feature_stds[:n_display], color='steelblue', alpha=0.7)
        
        # Use meaningful names for x-axis
        x_labels = []
        for i in range(n_display):
            feature_name = self.feature_names.get(f'F{i+1}', f'F{i+1}')
            x_labels.append(feature_name.split('_')[-1][:4])  # Use last part of name
        
        ax3.set_xticks(x_positions)
        ax3.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
        ax3.set_ylabel('Standard Deviation')
        ax3.set_title('Feature Diversity\n(Spatial Variation)')
        ax3.grid(True, alpha=0.3)
        
        # Color bars by diversity
        diversity_threshold = np.median(feature_stds) if len(feature_stds) > 1 else 0
        for i, bar in enumerate(bars):
            if i < len(feature_stds) and feature_stds[i] > diversity_threshold:
                bar.set_color('darkgreen')
            else:
                bar.set_color('orange')
        
        # 4. Residual Analysis
        ax4 = axes[1, 0]
        if traditional_accessibility is not None and len(traditional_mean) == len(learned_mean):
            residuals = learned_mean - traditional_mean
            ax4.scatter(traditional_mean, residuals, alpha=0.6, s=15, color='purple')
            ax4.axhline(0, color='red', linestyle='--')
            ax4.set_xlabel('Traditional Accessibility')
            ax4.set_ylabel('Residuals')
            ax4.set_title('Residual Analysis')
            ax4.grid(True, alpha=0.3)
            
            # Add residual stats
            rmse = np.sqrt(np.mean(residuals**2))
            ax4.text(0.05, 0.95, f'RMSE: {rmse:.4f}', transform=ax4.transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    verticalalignment='top')
        else:
            ax4.text(0.5, 0.5, 'No baseline for\nresidual analysis', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Residual Analysis')
        
        # 5. Multi-Modal Summary
        ax5 = axes[1, 1]
        if n_features >= 9:
            # Group by destination type (employment, healthcare, grocery)
            dest_means = []
            dest_stds = []
            dest_names = ['Employment', 'Healthcare', 'Grocery']
            
            for i in range(3):
                start_idx = i * 3
                end_idx = (i + 1) * 3
                if end_idx <= n_features:
                    dest_data = learned_accessibility[:, start_idx:end_idx]
                    dest_means.append(np.mean(dest_data))
                    dest_stds.append(np.std(dest_data))
            
            if len(dest_means) == 3:
                x_pos = np.arange(3)
                bars = ax5.bar(x_pos, dest_means, yerr=dest_stds, capsize=5, 
                              alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
                ax5.set_xticks(x_pos)
                ax5.set_xticklabels(dest_names)
                ax5.set_ylabel('Learned Accessibility')
                ax5.set_title('Multi-Modal Accessibility\n(Mean ± Std)')
                ax5.grid(True, alpha=0.3)
            else:
                ax5.text(0.5, 0.5, f'Features: {n_features}\n(Expected: 9)', 
                        ha='center', va='center', transform=ax5.transAxes)
                ax5.set_title('Multi-Modal Analysis')
        else:
            ax5.text(0.5, 0.5, f'Features: {n_features}\n(Need 9 for analysis)', 
                    ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Multi-Modal Analysis')
        
        # 6. Summary Statistics
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        # Calculate summary metrics
        diversity_score = np.mean(feature_stds)
        independence_score = 1 - np.max(np.abs(np.corrcoef(learned_accessibility.T) - 
                                               np.eye(n_features))) if n_features > 1 else 1
        
        summary_text = f"""Stage 1 Assessment:
• Samples: {n_addresses:,}
• Features: {n_features}
• R²: {r_squared:.3f} ({self._interpret_r2(r_squared)})
• Diversity: {diversity_score:.4f}
• Independence: {independence_score:.3f}

Status: {self._assess_stage1_status(r_squared, diversity_score)}

Next Steps:
{self._suggest_stage1_improvements(r_squared)}"""
        
        ax6.text(0.05, 0.95, summary_text.strip(), transform=ax6.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

    def plot_svi_prediction_validation(self, gnn_predictions: pd.DataFrame,
                                     stage2_metrics: Dict, tract_svi: float,
                                     output_path: str):
        """Stage 2: Does model predict SVI accurately?"""
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Stage 2: SVI Prediction Validation', fontsize=14, fontweight='bold')
        
        if gnn_predictions is None or gnn_predictions.empty:
            fig.text(0.5, 0.5, 'No GNN predictions available', 
                    ha='center', va='center', fontsize=16)
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            return
        
        # Extract data
        predictions = gnn_predictions['mean'].values
        uncertainty = gnn_predictions.get('sd', np.full(len(predictions), 0.05)).values
        x_coords = gnn_predictions.get('x', np.arange(len(predictions))).values
        y_coords = gnn_predictions.get('y', np.arange(len(predictions))).values
        
        # 1. Spatial Distribution
        ax1 = axes[0, 0]
        if 'x' in gnn_predictions.columns and 'y' in gnn_predictions.columns:
            scatter = ax1.scatter(x_coords, y_coords, c=predictions, 
                                s=15, alpha=0.7, cmap='viridis')
            ax1.set_xlabel('Longitude')
            ax1.set_ylabel('Latitude')
            ax1.set_title('SVI Predictions (Spatial)')
            ax1.set_aspect('equal')
            plt.colorbar(scatter, ax=ax1, label='Predicted SVI')
        else:
            ax1.text(0.5, 0.5, 'No spatial\ncoordinates', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Spatial Distribution')
        
        # 2. Distribution vs Target
        ax2 = axes[0, 1]
        ax2.hist(predictions, bins=30, alpha=0.7, density=True, color='steelblue', edgecolor='black')
        ax2.axvline(tract_svi, color='red', linestyle='--', linewidth=2, 
                   label=f'Target: {tract_svi:.3f}')
        ax2.axvline(np.mean(predictions), color='green', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(predictions):.3f}')
        ax2.set_xlabel('SVI Value')
        ax2.set_ylabel('Density')
        ax2.set_title('Prediction Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Constraint Satisfaction
        ax3 = axes[0, 2]
        predicted_mean = np.mean(predictions)
        constraint_error = abs(predicted_mean - tract_svi) / tract_svi * 100 if tract_svi > 0 else 100
        
        # Simple gauge visualization
        categories = ['Error', 'Satisfied']
        error_pct = min(100, constraint_error)
        satisfied_pct = max(0, 100 - error_pct)
        values = [error_pct, satisfied_pct]
        colors = ['lightcoral', 'lightgreen']
        
        wedges = ax3.pie(values, labels=categories, colors=colors, 
                        autopct='%1.1f%%', startangle=90)
        ax3.set_title(f'Constraint Satisfaction\nTarget: {tract_svi:.3f}')
        
        # 4. Uncertainty Analysis
        ax4 = axes[1, 0]
        ax4.scatter(predictions, uncertainty, alpha=0.6, s=15, color='orange')
        ax4.set_xlabel('Predicted SVI')
        ax4.set_ylabel('Uncertainty')
        ax4.set_title('Prediction Confidence')
        ax4.grid(True, alpha=0.3)
        
        # Add trend line if enough points
        if len(predictions) > 5:
            z = np.polyfit(predictions, uncertainty, 1)
            p = np.poly1d(z)
            ax4.plot(sorted(predictions), p(sorted(predictions)), "r--", alpha=0.8)
        
        # 5. Spatial Variation
        ax5 = axes[1, 1]
        if len(predictions) > 10:
            # Simple spatial autocorrelation
            distances = []
            pred_diffs = []
            
            # Sample for efficiency
            sample_indices = np.random.choice(len(predictions), min(100, len(predictions)), replace=False)
            
            for i, idx1 in enumerate(sample_indices[:-1]):
                for idx2 in sample_indices[i+1:i+6]:  # Limit comparisons
                    geo_dist = np.sqrt((x_coords[idx1] - x_coords[idx2])**2 + 
                                     (y_coords[idx1] - y_coords[idx2])**2)
                    pred_diff = abs(predictions[idx1] - predictions[idx2])
                    distances.append(geo_dist)
                    pred_diffs.append(pred_diff)
            
            if len(distances) > 0:
                ax5.scatter(distances, pred_diffs, alpha=0.5, s=10, color='purple')
                ax5.set_xlabel('Geographic Distance')
                ax5.set_ylabel('Prediction Difference')
                ax5.set_title('Spatial Autocorrelation')
                ax5.grid(True, alpha=0.3)
            else:
                ax5.text(0.5, 0.5, 'Insufficient data\nfor analysis', 
                        ha='center', va='center', transform=ax5.transAxes)
                ax5.set_title('Spatial Autocorrelation')
        else:
            ax5.text(0.5, 0.5, 'Too few predictions\nfor analysis', 
                    ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Spatial Autocorrelation')
        
        # 6. Summary Statistics
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        # Calculate key metrics
        mae = np.mean(np.abs(predictions - tract_svi))
        rmse = np.sqrt(np.mean((predictions - tract_svi)**2))
        spatial_std = np.std(predictions)
        
        summary_text = f"""Stage 2 Assessment:
• Target SVI: {tract_svi:.3f}
• Predicted Mean: {predicted_mean:.3f}
• Constraint Error: {constraint_error:.1f}%
• MAE: {mae:.4f}
• RMSE: {rmse:.4f}
• Spatial Std: {spatial_std:.4f}

Quality: {self._interpret_constraint_satisfaction(constraint_error)}

Status: {self._assess_stage2_status(constraint_error, spatial_std)}"""
        
        quality_color = self._get_constraint_color(constraint_error)
        ax6.text(0.05, 0.95, summary_text.strip(), transform=ax6.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=quality_color, alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

    def plot_statistical_summary(self, results: Dict, output_path: str):
        """Clean statistical summary without editorial content"""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Statistical Summary', fontsize=14, fontweight='bold')
        
        # Extract key data
        learned_accessibility = results.get('learned_accessibility')
        traditional_accessibility = results.get('traditional_accessibility') 
        gnn_predictions = results.get('gnn_predictions')
        tract_svi = results.get('tract_svi', 0)
        
        # Calculate key metrics
        stage1_r2 = 0
        stage2_error = 100
        n_addresses = 0
        
        if learned_accessibility is not None and traditional_accessibility is not None:
            min_len = min(len(learned_accessibility), len(traditional_accessibility))
            learned_mean = np.mean(learned_accessibility[:min_len], axis=1) if learned_accessibility.ndim > 1 else learned_accessibility[:min_len]
            traditional_mean = np.mean(traditional_accessibility[:min_len], axis=1) if traditional_accessibility.ndim > 1 else traditional_accessibility[:min_len]
            
            correlation, _ = pearsonr(learned_mean, traditional_mean)
            stage1_r2 = max(0, correlation**2)
        
        if gnn_predictions is not None and not gnn_predictions.empty:
            predictions = gnn_predictions['mean'].values
            stage2_error = abs(np.mean(predictions) - tract_svi) / tract_svi * 100 if tract_svi > 0 else 100
            n_addresses = len(predictions)
        
        # 1. Method Performance Summary
        ax1 = axes[0, 0]
        components = ['Stage 1\n(Learning)', 'Stage 2\n(Prediction)', 'Overall\n(Combined)']
        stage1_score = min(100, stage1_r2 * 100)
        stage2_score = max(0, 100 - stage2_error)
        overall_score = (stage1_score + stage2_score) / 2
        
        scores = [stage1_score, stage2_score, overall_score]
        colors = [self._get_score_color(s) for s in scores]
        
        bars = ax1.bar(components, scores, color=colors, alpha=0.7)
        ax1.set_ylabel('Performance Score')
        ax1.set_title('Method Performance Summary')
        ax1.set_ylim(0, 100)
        ax1.grid(True, alpha=0.3)
        
        # Add score labels
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{score:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 2. Data Quality Summary
        ax2 = axes[0, 1]
        ax2.axis('off')
        
        quality_text = f"""Data Quality Assessment:

Addresses Processed: {n_addresses:,}
Features Generated: {learned_accessibility.shape[1] if learned_accessibility is not None else 0}

Stage 1 (Accessibility Learning):
• R² with Traditional: {stage1_r2:.3f}
• Learning Quality: {self._interpret_r2(stage1_r2)}

Stage 2 (SVI Prediction):
• Constraint Error: {stage2_error:.1f}%
• Prediction Quality: {self._interpret_constraint_satisfaction(stage2_error)}

Overall Assessment: {self._assess_overall_quality(stage1_r2, stage2_error)}"""
        
        ax2.text(0.05, 0.95, quality_text.strip(), transform=ax2.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        # 3. Diagnostic Information
        ax3 = axes[1, 0]
        ax3.axis('off')
        
        # Diagnose issues
        issues = []
        if stage1_r2 < 0.1:
            issues.append("Stage 1: Very low R² - check data alignment")
        if stage2_error > 50:
            issues.append("Stage 2: High constraint error - check model")
        if n_addresses < 100:
            issues.append("Data: Low sample size may affect reliability")
        
        if not issues:
            issues = ["No critical issues detected"]
        
        diagnostic_text = f"""Diagnostic Summary:

Critical Issues:
{chr(10).join(['• ' + issue for issue in issues])}

Recommendations:
• {"Investigate Stage 1 learning" if stage1_r2 < 0.3 else "Stage 1 learning acceptable"}
• {"Improve constraint satisfaction" if stage2_error > 20 else "Constraint satisfaction good"}
• {"Increase sample size" if n_addresses < 500 else "Sample size adequate"}

Next Steps:
{self._suggest_next_steps(stage1_r2, stage2_error)}"""
        
        ax3.text(0.05, 0.95, diagnostic_text.strip(), transform=ax3.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
        
        # 4. Method Comparison Table
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Create simple comparison table
        table_data = [
            ['Metric', 'Value', 'Assessment'],
            ['Stage 1 R²', f'{stage1_r2:.3f}', self._interpret_r2(stage1_r2)],
            ['Stage 2 Error', f'{stage2_error:.1f}%', self._interpret_constraint_satisfaction(stage2_error)],
            ['Sample Size', f'{n_addresses:,}', 'High' if n_addresses > 1000 else 'Medium' if n_addresses > 100 else 'Low'],
            ['Resolution', 'Address-level', 'High']
        ]
        
        # Simple text table
        table_text = "\n".join([" | ".join(row) for row in table_data])
        ax4.text(0.05, 0.95, f"Performance Summary:\n\n{table_text}", 
                transform=ax4.transAxes, fontsize=9, verticalalignment='top', 
                fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

    def plot_spatial_analysis(self, gnn_predictions: pd.DataFrame,
                            learned_accessibility: np.ndarray = None,
                            output_path: str = None,
                            tract_svi: float = None,
                            tract_fips: str = None):
        """3-panel spatial analysis: SVI predictions, within-tract deviation, prediction distribution"""

        if gnn_predictions is None or gnn_predictions.empty:
            fig = plt.figure(figsize=(18, 6))
            fig.text(0.5, 0.5, 'No spatial data available',
                    ha='center', va='center', fontsize=16)
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            return

        predictions = gnn_predictions['mean'].values
        x_coords = gnn_predictions.get('x', np.arange(len(predictions))).values
        y_coords = gnn_predictions.get('y', np.arange(len(predictions))).values
        n_addresses = len(predictions)

        # use provided tract_svi or fall back to prediction mean
        if tract_svi is None:
            tract_svi = float(np.mean(predictions))
        if tract_fips:
            title = f'Spatial Analysis: Tract {tract_fips} (SVI {tract_svi:.3f}, n={n_addresses})'
        else:
            title = f'Spatial Analysis: (SVI {tract_svi:.3f}, n={n_addresses})'

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(title, fontsize=14, fontweight='bold')

        # 1. SVI Prediction Map (fixed 0-1 colorscale)
        ax1 = axes[0]
        if 'x' in gnn_predictions.columns:
            scatter1 = ax1.scatter(x_coords, y_coords, c=predictions,
                                 s=15, alpha=0.7, cmap='RdYlGn_r',
                                 vmin=0, vmax=1)
            ax1.set_title('SVI Predictions')
            ax1.set_xlabel('Longitude')
            ax1.set_ylabel('Latitude')
            ax1.set_aspect('equal')
            plt.colorbar(scatter1, ax=ax1, label='Predicted SVI')
        else:
            ax1.text(0.5, 0.5, 'No coordinates\navailable',
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('SVI Predictions')

        # 2. Deviation from Tract Mean
        ax2 = axes[1]
        deviations = predictions - tract_svi
        max_dev = max(abs(deviations.min()), abs(deviations.max()), 1e-6)
        if 'x' in gnn_predictions.columns:
            scatter2 = ax2.scatter(x_coords, y_coords, c=deviations,
                                 s=15, alpha=0.7, cmap='coolwarm',
                                 vmin=-max_dev, vmax=max_dev)
            ax2.set_title('Within-Tract Variation')
            ax2.set_xlabel('Longitude')
            ax2.set_ylabel('Latitude')
            ax2.set_aspect('equal')
            plt.colorbar(scatter2, ax=ax2, label='Deviation from Tract SVI')
        else:
            ax2.text(0.5, 0.5, 'No coordinates\navailable',
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Within-Tract Variation')

        # 3. Prediction Distribution
        ax3 = axes[2]
        ax3.hist(predictions, bins=30, alpha=0.7, density=True,
                color='steelblue', edgecolor='black')
        ax3.axvline(tract_svi, color='red', linestyle='--', linewidth=2,
                   label=f'Tract SVI = {tract_svi:.3f}')
        pred_std = np.std(predictions)
        pred_range = np.ptp(predictions)
        ax3.text(0.95, 0.95,
                f'std = {pred_std:.4f}\nrange = {pred_range:.4f}',
                transform=ax3.transAxes, ha='right', va='top',
                fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax3.set_xlabel('Predicted SVI')
        ax3.set_ylabel('Density')
        ax3.set_title('Prediction Distribution')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

    # Helper methods for interpretation
    def _interpret_r2(self, r2: float) -> str:
        if r2 > self.excellent_r2:
            return "Excellent"
        elif r2 > self.good_r2:
            return "Good"
        elif r2 > self.moderate_r2:
            return "Fair"
        else:
            return "Poor"

    def _get_r2_color(self, r2: float) -> str:
        if r2 > self.excellent_r2:
            return 'lightgreen'
        elif r2 > self.good_r2:
            return 'lightyellow'
        elif r2 > self.moderate_r2:
            return 'orange'
        else:
            return 'lightcoral'

    def _interpret_constraint_satisfaction(self, error_pct: float) -> str:
        if error_pct < 5:
            return "Excellent"
        elif error_pct < 15:
            return "Good"
        elif error_pct < 30:
            return "Fair"
        else:
            return "Poor"

    def _get_constraint_color(self, error_pct: float) -> str:
        if error_pct < 5:
            return 'lightgreen'
        elif error_pct < 15:
            return 'lightyellow'
        elif error_pct < 30:
            return 'orange'
        else:
            return 'lightcoral'

    def _get_score_color(self, score: float) -> str:
        if score > 75:
            return 'lightgreen'
        elif score > 50:
            return 'lightyellow'
        elif score > 25:
            return 'orange'
        else:
            return 'lightcoral'

    def _assess_stage1_status(self, r2: float, diversity: float) -> str:
        if r2 > self.good_r2 and diversity > 0.05:
            return "Learning successful"
        elif r2 > self.moderate_r2:
            return "Partial learning achieved"
        else:
            return "Learning needs improvement"

    def _suggest_stage1_improvements(self, r2: float) -> str:
        if r2 < 0.1:
            return "Debug data alignment and model architecture"
        elif r2 < 0.3:
            return "Tune hyperparameters and check feature scaling"
        else:
            return "Consider expanding feature set"

    def _assess_stage2_status(self, error_pct: float, spatial_std: float) -> str:
        if error_pct < 10 and spatial_std > 0.01:
            return "Good prediction quality"
        elif error_pct < 30:
            return "Acceptable with improvements needed"
        else:
            return "Requires methodology refinement"

    def _assess_overall_quality(self, r2: float, error_pct: float) -> str:
        if r2 > self.good_r2 and error_pct < 15:
            return "High quality results"
        elif r2 > self.moderate_r2 and error_pct < 30:
            return "Moderate quality results"
        else:
            return "Quality needs improvement"

    def _suggest_next_steps(self, r2: float, error_pct: float) -> str:
        steps = []
        if r2 < 0.3:
            steps.append("Debug Stage 1 learning")
        if error_pct > 20:
            steps.append("Improve constraint satisfaction")
        if not steps:
            steps.append("Validate on additional data")
        return "; ".join(steps)

    def _assess_spatial_variation(self, predictions: np.ndarray) -> str:
        std = np.std(predictions)
        if std > 0.1:
            return "High"
        elif std > 0.05:
            return "Moderate"
        else:
            return "Low"

    def _assess_spatial_coverage(self, n_addresses: int) -> str:
        if n_addresses > 1000:
            return "High"
        elif n_addresses > 100:
            return "Medium"
        else:
            return "Low"

    def _assess_equity_pattern(self, correlation: float) -> str:
        if correlation < -0.3:
            return "Strong equity pattern detected"
        elif correlation < -0.1:
            return "Moderate equity pattern"
        else:
            return "Weak/no equity pattern"


# Main function to create all visualizations
def create_granite_research_visualizations(results_dict: Dict, output_directory: str = "./output/"):
    """
    Create clean GRANITE research visualizations
    
    Args:
        results_dict: Dictionary with research results
        output_directory: Where to save visualizations
    
    Returns:
        dict: Summary metrics
    """
    import os
    os.makedirs(output_directory, exist_ok=True)
    
    visualizer = GRANITEResearchVisualizer()
    visualizer.create_comprehensive_research_analysis(results_dict, output_directory)
    
    print(f"Clean GRANITE visualizations saved to {output_directory}")
    return visualizer

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
        Create disaggregation comparison dashboard.
        
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
        
        # 1. Constraint-Variation Tradeoff Scatter
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_tradeoff_scatter(ax1, methods)
        
        # 2. Spatial Variation Comparison
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_variation_comparison(ax2, methods)
        
        # 3. Method Predictions Distribution
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_prediction_distributions(ax3, methods, tract_svi)
        
        # 4. Prediction Range by Method
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_prediction_range(ax4, methods)
        
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
    
    def _plot_tradeoff_scatter(self, ax, methods: Dict):
        """Scatter of constraint error % vs spatial variation (std) per method."""

        for name in ['GNN', 'Naive_Uniform', 'IDW_p2.0', 'IDW_p3.0', 'Kriging']:
            if name not in methods:
                continue
            m = methods[name]
            err = m['constraint_error_pct']
            std = m['std']
            color = self._get_method_color(name)
            ax.scatter(err, std, color=color, s=120, edgecolors='black',
                      linewidths=0.8, zorder=5)
            ax.annotate(name.replace('_', '\n'), (err, std),
                       textcoords='offset points', xytext=(8, 4),
                       fontsize=8, color=color)

        ax.set_xlabel('Constraint Error (%)', fontsize=10)
        ax.set_ylabel('Spatial Variation (std)', fontsize=10)
        ax.set_title('Constraint-Variation Tradeoff', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_variation_comparison(self, ax, methods: Dict):
        """Bar chart comparing spatial variation (std)."""

        method_names = []
        stds = []
        colors = []

        for name in ['GNN', 'IDW_p2.0']:
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

        for name in ['GNN', 'IDW_p2.0']:
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
    
    def _plot_prediction_range(self, ax, methods: Dict):
        """Horizontal bar chart of prediction range (max - min) per method."""

        method_names = []
        ranges = []
        colors = []

        for name in ['GNN', 'IDW_p2.0']:
            if name in methods:
                method_names.append(name.replace('_', '\n'))
                ranges.append(methods[name]['range'])
                colors.append(self._get_method_color(name))

        if not ranges:
            ax.text(0.5, 0.5, 'No method data\navailable',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Prediction Range', fontsize=11, fontweight='bold')
            return

        y_pos = np.arange(len(method_names))
        bars = ax.barh(y_pos, ranges, color=colors, alpha=0.8, edgecolor='black')

        for bar, r in zip(bars, ranges):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                   f'{r:.4f}', ha='left', va='center', fontsize=9)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(method_names, fontsize=9)
        ax.set_xlabel('Prediction Range (max - min)', fontsize=10)
        ax.set_title('Prediction Range by Method\n(Higher = More Differentiation)',
                    fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
    
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
        columns = ['Method', 'Mean', 'Std', 'Range', 'Err %', "Moran's I"]
        data = []

        for name in ['GNN', 'Naive_Uniform', 'IDW_p2.0', 'IDW_p3.0', 'Kriging']:
            if name in methods:
                m = methods[name]
                moran_val = m.get('morans_i')
                moran_str = f"{moran_val:.3f}" if moran_val is not None else "n/a"
                data.append([
                    name,
                    f"{m['mean']:.4f}",
                    f"{m['std']:.4f}",
                    f"{m['range']:.4f}",
                    f"{m['constraint_error_pct']:.2f}",
                    moran_str
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
                                    address_gdf=None,
                                    predictions: np.ndarray = None,
                                    tract_svi: float = None,
                                    tract_results: Dict = None,
                                    title: str = None,
                                    output_path: str = None,
                                    multi_tract_data: Dict = None,
                                    tract_boundaries_path: str = None) -> plt.Figure:
        """
        Create spatial map of disaggregated predictions.

        Single-tract mode (when address_gdf is provided):
            Two-panel layout: predictions + deviation from tract mean.

        Multi-tract mode (when tract_results or multi_tract_data is provided):
            Two-panel layout on shared axes: unified SVI colorscale + deviation.

        Args:
            address_gdf: GeoDataFrame with address points (single-tract)
            predictions: Array of predicted SVI values (single-tract)
            tract_svi: Known tract SVI (single-tract)
            tract_results: Dict keyed by FIPS, each value a dict with
                'address_gdf', 'predictions', 'tract_svi' (multi-tract)
            title: Plot title
            output_path: Path to save figure
            multi_tract_data: Deprecated alias for tract_results
            tract_boundaries_path: Path to TIGER shapefile for boundary overlay
        """
        # support both parameter names
        multi = tract_results or multi_tract_data
        if multi is not None:
            return self._plot_multi_tract_disaggregation(
                multi, title, output_path, tract_boundaries_path
            )

        # single-tract mode (original behavior)
        if title is None:
            title = "GNN Disaggregation"
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        x = address_gdf.geometry.x.values
        y = address_gdf.geometry.y.values

        # left: predictions
        ax1 = axes[0]
        scatter = ax1.scatter(x, y, c=predictions, cmap='RdYlGn_r',
                             s=30, alpha=0.7, edgecolors='none',
                             vmin=0, vmax=1)
        plt.colorbar(scatter, ax=ax1, label='Predicted SVI')
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.set_title(f'{title}\n(Tract SVI: {tract_svi:.4f})', fontweight='bold')

        # right: deviation from tract mean
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

    def _plot_multi_tract_disaggregation(self, tract_data, title, output_path,
                                          tract_boundaries_path):
        """multi-tract spatial heatmap across all tracts on shared axes."""
        import os

        n_tracts = len(tract_data)

        # collect all coordinates, predictions, and per-address deviations
        all_x, all_y, all_preds, all_devs = [], [], [], []
        total_addresses = 0

        for fips, tdata in tract_data.items():
            gdf = tdata['address_gdf']
            preds = np.asarray(tdata['predictions'])
            tsvi = tdata['tract_svi']

            x = gdf.geometry.x.values
            y = gdf.geometry.y.values

            all_x.append(x)
            all_y.append(y)
            all_preds.append(preds)
            all_devs.append(preds - tsvi)
            total_addresses += len(x)

        all_x = np.concatenate(all_x)
        all_y = np.concatenate(all_y)
        all_preds = np.concatenate(all_preds)
        all_devs = np.concatenate(all_devs)

        # auto-scale point size
        if total_addresses >= 20000:
            point_size = 3
        elif total_addresses >= 5000:
            point_size = 10
        else:
            point_size = 30

        # default title
        if title is None:
            title = f"GRANITE Disaggregation ({n_tracts} tracts, GCN)"

        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        # try to load tract boundaries for background overlay
        tract_geoms = self._load_tract_boundaries(
            tract_boundaries_path, list(tract_data.keys())
        )

        for ax in axes:
            if tract_geoms is not None:
                tract_geoms.boundary.plot(ax=ax, color='gray', linewidth=0.5, zorder=1)

        # left panel: unified SVI colorscale 0 to 1
        ax1 = axes[0]
        scatter = ax1.scatter(all_x, all_y, c=all_preds, cmap='RdYlGn_r',
                             vmin=0, vmax=1, s=point_size, alpha=0.7,
                             edgecolors='none', zorder=2)
        plt.colorbar(scatter, ax=ax1, label='Predicted SVI', shrink=0.8)
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.set_title(f'{title}\n({n_tracts} tracts, {total_addresses:,} addresses)',
                     fontweight='bold')
        ax1.set_aspect('equal')

        # right panel: deviation from each address's own tract SVI (coolwarm)
        ax2 = axes[1]
        max_dev = max(abs(all_devs.min()), abs(all_devs.max())) if len(all_devs) > 0 else 0.1
        scatter2 = ax2.scatter(all_x, all_y, c=all_devs, cmap='coolwarm',
                              vmin=-max_dev, vmax=max_dev, s=point_size,
                              alpha=0.7, edgecolors='none', zorder=2)
        plt.colorbar(scatter2, ax=ax2, label='Deviation from Tract SVI', shrink=0.8)
        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
        ax2.set_title('Spatial Variation Pattern\n(Blue=Lower, Red=Higher than Tract Mean)',
                     fontweight='bold')
        ax2.set_aspect('equal')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"Saved: {output_path}")

        return fig

    def _load_tract_boundaries(self, shapefile_path, fips_list):
        """load tract boundary polygons for overlay. returns None on failure."""
        try:
            import geopandas as gpd
        except ImportError:
            return None

        if shapefile_path is None:
            # try default TIGER location
            import os
            default_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                'data', 'raw', 'tl_2020_47_tract.shp'
            )
            if os.path.exists(default_path):
                shapefile_path = default_path
            else:
                return None

        try:
            tracts = gpd.read_file(shapefile_path)
            # filter to matching FIPS codes
            matched = tracts[tracts['GEOID'].isin(fips_list)]
            if matched.empty:
                # try without filtering (show all county tracts for context)
                county_fips = fips_list[0][:5] if fips_list else None
                if county_fips:
                    matched = tracts[
                        (tracts['STATEFP'] == county_fips[:2]) &
                        (tracts['COUNTYFP'] == county_fips[2:5])
                    ]
            return matched if not matched.empty else None
        except Exception:
            return None


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