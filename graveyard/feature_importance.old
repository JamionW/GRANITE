"""
Feature importance analysis for GRANITE GNN
"""
import numpy as np
import torch
from typing import Dict, List, Tuple
import pandas as pd
import matplotlib.pyplot as plt

class FeatureImportanceAnalyzer:
    """
    Analyzes which accessibility features drive GNN predictions using
    permutation importance and gradient-based methods.
    """
    
    def __init__(self, model, device='cpu', verbose=True):
        """
        Args:
            model: Trained AccessibilitySVIGNN model
            device: 'cpu' or 'cuda'
            verbose: Print progress
        """
        self.model = model
        self.device = device
        self.verbose = verbose
        self.model.eval()
    
    def permutation_importance(
        self,
        accessibility_features: np.ndarray,
        edge_index: torch.LongTensor,
        true_svi: float,
        feature_names: List[str],
        n_repeats: int = 10
    ) -> Dict:
        """
        Compute permutation importance for each feature.
        
        Args:
            accessibility_features: [N, 54] input features
            edge_index: Graph edges
            true_svi: Ground truth tract SVI
            feature_names: Names of all 54 features
            n_repeats: Number of permutation repeats per feature
            
        Returns:
            Dictionary with importance scores and rankings
        """
        if self.verbose:
            print(f"\nComputing permutation importance ({n_repeats} repeats per feature)...")
        
        n_addresses, n_features = accessibility_features.shape
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(accessibility_features).to(self.device)
        edge_index = edge_index.to(self.device)
        
        # Baseline performance
        with torch.no_grad():
            baseline_preds = self.model(features_tensor, edge_index).cpu().numpy()
            baseline_constraint_error = abs(np.mean(baseline_preds) - true_svi) / true_svi * 100
            baseline_spatial_std = np.std(baseline_preds)
        
        if self.verbose:
            print(f"Baseline: constraint error = {baseline_constraint_error:.2f}%, spatial std = {baseline_spatial_std:.4f}")
        
        # Compute importance for each feature
        importance_scores = np.zeros(n_features)
        importance_std = np.zeros(n_features)
        
        for feat_idx in range(n_features):
            if self.verbose and feat_idx % 10 == 0:
                print(f"  Processing feature {feat_idx+1}/{n_features}...")
            
            repeat_scores = []
            
            for _ in range(n_repeats):
                # Permute this feature
                permuted_features = features_tensor.clone()
                shuffled_idx = torch.randperm(n_addresses)
                permuted_features[:, feat_idx] = permuted_features[shuffled_idx, feat_idx]
                
                # Evaluate with permuted feature
                with torch.no_grad():
                    permuted_preds = self.model(permuted_features, edge_index).cpu().numpy()
                    permuted_error = abs(np.mean(permuted_preds) - true_svi) / true_svi * 100
                
                # Importance = increase in error after permutation
                error_increase = permuted_error - baseline_constraint_error
                repeat_scores.append(error_increase)
            
            importance_scores[feat_idx] = np.mean(repeat_scores)
            importance_std[feat_idx] = np.std(repeat_scores)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores,
            'std': importance_std,
            'importance_normalized': importance_scores / np.sum(np.abs(importance_scores)) if np.sum(np.abs(importance_scores)) > 0 else importance_scores
        })
        
        results_df = results_df.sort_values('importance', ascending=False).reset_index(drop=True)
        results_df['rank'] = range(1, len(results_df) + 1)
        
        # Summary statistics
        n_important = np.sum(importance_scores > 0.1)
        n_negligible = np.sum(np.abs(importance_scores) < 0.01)
        
        summary = {
            'total_features': n_features,
            'important_features': n_important,
            'negligible_features': n_negligible,
            'top_10_features': results_df.head(10)['feature'].tolist(),
            'baseline_error': baseline_constraint_error,
            'baseline_std': baseline_spatial_std
        }
        
        return {
            'feature_importance': results_df,
            'summary': summary,
            'raw_scores': importance_scores
        }
    
    def gradient_importance(
        self,
        accessibility_features: np.ndarray,
        edge_index: torch.LongTensor,
        feature_names: List[str]
    ) -> Dict:
        """
        Compute gradient-based feature importance.
        Measures sensitivity of output to each input feature.
        
        Args:
            accessibility_features: [N, 54] input features
            edge_index: Graph edges
            feature_names: Names of all 54 features
            
        Returns:
            Dictionary with gradient-based importance scores
        """
        if self.verbose:
            print("\nComputing gradient-based importance...")
        
        n_addresses, n_features = accessibility_features.shape
        
        # Convert to tensor and enable gradients
        features_tensor = torch.FloatTensor(accessibility_features).to(self.device)
        features_tensor.requires_grad = True
        edge_index = edge_index.to(self.device)
        
        # Forward pass
        predictions = self.model(features_tensor, edge_index)
        
        # Compute gradient of mean prediction with respect to inputs
        mean_pred = predictions.mean()
        mean_pred.backward()
        
        # Gradient magnitude as importance
        gradients = features_tensor.grad.cpu().numpy()
        importance_scores = np.mean(np.abs(gradients), axis=0)
        
        # Normalize
        importance_normalized = importance_scores / np.sum(importance_scores)
        
        results_df = pd.DataFrame({
            'feature': feature_names,
            'gradient_importance': importance_scores,
            'gradient_normalized': importance_normalized
        })
        
        results_df = results_df.sort_values('gradient_importance', ascending=False).reset_index(drop=True)
        results_df['rank'] = range(1, len(results_df) + 1)
        
        return {
            'feature_importance': results_df,
            'raw_gradients': gradients
        }
    
    def plot_importance(self, importance_results: Dict, output_path: str, top_n: int = 20):
        """
        Create visualization of feature importance.
        
        Args:
            importance_results: Results from permutation_importance()
            output_path: Where to save plot
            top_n: Number of top features to show
        """
        df = importance_results['feature_importance']
        top_features = df.head(top_n)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 8))
        
        # Plot 1: Top features bar chart
        ax1 = axes[0]
        y_pos = np.arange(len(top_features))
        bars = ax1.barh(y_pos, top_features['importance'].values, 
                        xerr=top_features['std'].values, 
                        color='steelblue', alpha=0.7)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(top_features['feature'].values, fontsize=9)
        ax1.invert_yaxis()
        ax1.set_xlabel('Importance (Constraint Error Increase %)', fontsize=11)
        ax1.set_title(f'Top {top_n} Most Important Features', fontsize=12, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        ax1.axvline(0, color='red', linestyle='--', alpha=0.5)
        
        # Plot 2: Cumulative importance
        ax2 = axes[1]
        sorted_importance = np.sort(df['importance_normalized'].values)[::-1]
        cumulative = np.cumsum(sorted_importance)
        ax2.plot(range(1, len(cumulative)+1), cumulative, 'o-', linewidth=2, markersize=4)
        ax2.axhline(0.8, color='red', linestyle='--', alpha=0.7, label='80% threshold')
        ax2.axhline(0.9, color='orange', linestyle='--', alpha=0.7, label='90% threshold')
        ax2.set_xlabel('Number of Features', fontsize=11)
        ax2.set_ylabel('Cumulative Importance', fontsize=11)
        ax2.set_title('Cumulative Feature Importance', fontsize=12, fontweight='bold')
        ax2.grid(alpha=0.3)
        ax2.legend()
        
        # Find how many features explain 80% and 90%
        n_80 = np.argmax(cumulative >= 0.8) + 1
        n_90 = np.argmax(cumulative >= 0.9) + 1
        ax2.text(0.5, 0.15, f'{n_80} features explain 80% of importance\n{n_90} features explain 90% of importance',
                transform=ax2.transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        if self.verbose:
            print(f"Feature importance plot saved to {output_path}")
    
    def generate_report(self, importance_results: Dict, output_path: str):
        """
        Generate text report of feature importance analysis.
        
        Args:
            importance_results: Results from permutation_importance()
            output_path: Where to save report
        """
        df = importance_results['feature_importance']
        summary = importance_results['summary']
        
        report = []
        report.append("=" * 80)
        report.append("GRANITE FEATURE IMPORTANCE ANALYSIS")
        report.append("=" * 80)
        report.append("")
        
        report.append("SUMMARY STATISTICS")
        report.append("-" * 80)
        report.append(f"Total features analyzed: {summary['total_features']}")
        report.append(f"Features with importance > 0.1%: {summary['important_features']}")
        report.append(f"Negligible features (|importance| < 0.01%): {summary['negligible_features']}")
        report.append(f"Baseline constraint error: {summary['baseline_error']:.2f}%")
        report.append("")
        
        report.append("TOP 20 MOST IMPORTANT FEATURES")
        report.append("-" * 80)
        report.append(f"{'Rank':<6} {'Feature':<40} {'Importance':<12} {'Normalized':<12}")
        report.append("-" * 80)
        
        for idx, row in df.head(20).iterrows():
            report.append(f"{row['rank']:<6} {row['feature']:<40} {row['importance']:>10.4f}% {row['importance_normalized']:>10.4f}")
        
        report.append("")
        report.append("BOTTOM 10 FEATURES (Least Important)")
        report.append("-" * 80)
        
        for idx, row in df.tail(10).iterrows():
            report.append(f"{row['rank']:<6} {row['feature']:<40} {row['importance']:>10.4f}% {row['importance_normalized']:>10.4f}")
        
        report.append("")
        report.append("INTERPRETATION GUIDE")
        report.append("-" * 80)
        report.append("Importance Score: Increase in constraint error (%) when feature is permuted")
        report.append("Positive values: Feature helps model maintain constraint")
        report.append("Near-zero values: Feature is ignored by model")
        report.append("Negative values: Feature may introduce noise")
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        with open(output_path, 'w') as f:
            f.write(report_text)
        
        if self.verbose:
            print(report_text)
            print(f"\nReport saved to {output_path}")