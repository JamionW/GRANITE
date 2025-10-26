"""
Updated Baseline Methods for Simplified GRANITE
Direct comparison with GNN accessibility → SVI predictions
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, Tuple, Optional

class AccessibilityBaseline:
    """
    Baseline methods that use the same accessibility features as the GNN
    but apply traditional spatial disaggregation techniques
    """
    
    def __init__(self, config=None):
        self.config = config or {}
    
    def create_baseline_predictions(self, addresses: pd.DataFrame, 
                                  accessibility_features: np.ndarray,
                                  tract_svi: float) -> pd.DataFrame:
        """
        Create baseline SVI predictions using accessibility features
        
        Methods compared:
        1. Linear accessibility-SVI relationship
        2. Inverse distance weighting (IDW) 
        3. Accessibility-weighted spatial interpolation
        
        Args:
            addresses: GeoDataFrame with address locations
            accessibility_features: Array [n_addresses, n_features] 
            tract_svi: Target tract SVI value
            
        Returns:
            DataFrame with baseline predictions
        """
        n_addresses = len(addresses)
        
        # Method 1: Linear accessibility-SVI relationship
        linear_predictions = self._linear_accessibility_model(accessibility_features, tract_svi)
        
        # Method 2: Inverse distance weighting with accessibility
        idw_predictions = self._accessibility_weighted_idw(addresses, accessibility_features, tract_svi)
        
        # Method 3: Accessibility-stratified interpolation
        stratified_predictions = self._accessibility_stratified_interpolation(
            addresses, accessibility_features, tract_svi
        )
        
        # Create comparison DataFrame
        baseline_df = pd.DataFrame({
            'address_id': [addr.get('address_id', i) for i, (_, addr) in enumerate(addresses.iterrows())],
            'x': [addr.geometry.x for _, addr in addresses.iterrows()],
            'y': [addr.geometry.y for _, addr in addresses.iterrows()],
            'linear_baseline': linear_predictions,
            'idw_baseline': idw_predictions,
            'stratified_baseline': stratified_predictions
        })
        
        # Select best performing baseline as primary
        methods = ['linear_baseline', 'idw_baseline', 'stratified_baseline']
        constraint_errors = []
        
        for method in methods:
            pred_mean = baseline_df[method].mean()
            error = abs(pred_mean - tract_svi) / tract_svi * 100
            constraint_errors.append(error)
        
        best_method = methods[np.argmin(constraint_errors)]
        baseline_df['mean'] = baseline_df[best_method]
        baseline_df['method_used'] = best_method
        
        # Add uncertainty estimates
        baseline_df['sd'] = self._estimate_baseline_uncertainty(baseline_df['mean'], tract_svi)
        baseline_df['q025'] = np.clip(baseline_df['mean'] - 1.96 * baseline_df['sd'], 0.0, 1.0)
        baseline_df['q975'] = np.clip(baseline_df['mean'] + 1.96 * baseline_df['sd'], 0.0, 1.0)
        
        return baseline_df
    
    def _linear_accessibility_model(self, accessibility_features: np.ndarray, tract_svi: float) -> np.ndarray:
        """
        Simple linear model: SVI ~ f(accessibility)
        
        Assumes lower accessibility correlates with higher vulnerability
        but includes regularization to prevent systematic bias
        """
        # Compute overall accessibility score for each address
        overall_accessibility = np.mean(accessibility_features, axis=1)
        
        # Normalize accessibility to [0, 1]
        min_access = np.min(overall_accessibility)
        max_access = np.max(overall_accessibility)
        
        if max_access > min_access:
            normalized_access = (overall_accessibility - min_access) / (max_access - min_access)
        else:
            normalized_access = np.full_like(overall_accessibility, 0.5)
        
        # Linear relationship with moderate slope to avoid extreme predictions
        # SVI = tract_mean + accessibility_effect
        accessibility_effect = (0.5 - normalized_access) * 0.2  # Max ±0.1 SVI units
        
        predictions = tract_svi + accessibility_effect
        predictions = np.clip(predictions, 0.0, 1.0)
        
        # Apply constraint: adjust to preserve tract mean
        adjustment = tract_svi - np.mean(predictions)
        predictions += adjustment * 0.8  # Partial adjustment to maintain variation
        predictions = np.clip(predictions, 0.0, 1.0)
        
        return predictions
    
    def _accessibility_weighted_idw(self, addresses: pd.DataFrame, 
                                   accessibility_features: np.ndarray,
                                   tract_svi: float) -> np.ndarray:
        """
        Inverse Distance Weighting with accessibility-based adjustment
        
        Uses spatial interpolation but weights by accessibility patterns
        """
        # Extract coordinates
        coords = np.array([[addr.geometry.x, addr.geometry.y] for _, addr in addresses.iterrows()])
        n_addresses = len(coords)
        
        # Compute accessibility-based "virtual SVI" values
        overall_accessibility = np.mean(accessibility_features, axis=1)
        
        # Create accessibility-based SVI estimates
        normalized_access = (overall_accessibility - np.min(overall_accessibility))
        if np.max(normalized_access) > 0:
            normalized_access = normalized_access / np.max(normalized_access)
        
        # Virtual SVI: higher accessibility → lower SVI (with limits)
        virtual_svi = tract_svi - (normalized_access - 0.5) * 0.15
        virtual_svi = np.clip(virtual_svi, tract_svi - 0.2, tract_svi + 0.2)
        
        # IDW interpolation using virtual SVI values
        predictions = []
        
        for i in range(n_addresses):
            target_point = coords[i]
            
            # Calculate distances to all other points
            distances = np.array([
                np.sqrt(np.sum((target_point - coords[j])**2)) 
                for j in range(n_addresses)
            ])
            
            # Avoid zero distances
            distances = np.maximum(distances, 1e-10)
            
            # IDW weights
            weights = 1 / (distances ** 2)
            weights = weights / np.sum(weights)
            
            # Weighted average
            interpolated_svi = np.sum(weights * virtual_svi)
            predictions.append(interpolated_svi)
        
        predictions = np.array(predictions)
        
        # Apply constraint preservation
        adjustment = tract_svi - np.mean(predictions)
        predictions += adjustment * 0.7
        predictions = np.clip(predictions, 0.0, 1.0)
        
        return predictions
    
    def _accessibility_stratified_interpolation(self, addresses: pd.DataFrame,
                                              accessibility_features: np.ndarray, 
                                              tract_svi: float) -> np.ndarray:
        """
        Stratify addresses by accessibility level and interpolate within strata
        """
        n_addresses = len(addresses)
        
        # Compute accessibility terciles
        overall_accessibility = np.mean(accessibility_features, axis=1)
        terciles = np.percentile(overall_accessibility, [33.33, 66.67])
        
        # Assign accessibility strata
        strata = np.zeros(n_addresses, dtype=int)
        strata[overall_accessibility <= terciles[0]] = 0  # Low accessibility
        strata[(overall_accessibility > terciles[0]) & (overall_accessibility <= terciles[1])] = 1  # Medium
        strata[overall_accessibility > terciles[1]] = 2  # High accessibility
        
        # Assign SVI by stratum (inverse relationship but moderated)
        stratum_svi = {
            0: tract_svi + 0.08,  # Low accessibility → slightly higher SVI
            1: tract_svi,         # Medium accessibility → tract average
            2: tract_svi - 0.08   # High accessibility → slightly lower SVI
        }
        
        predictions = np.array([stratum_svi[stratum] for stratum in strata])
        
        # Add within-stratum variation based on spatial patterns
        coords = np.array([[addr.geometry.x, addr.geometry.y] for _, addr in addresses.iterrows()])
        
        for stratum in [0, 1, 2]:
            stratum_mask = (strata == stratum)
            if np.sum(stratum_mask) > 1:
                stratum_coords = coords[stratum_mask]
                
                # Add spatial variation within stratum
                centroid = np.mean(stratum_coords, axis=0)
                distances_from_centroid = np.array([
                    np.sqrt(np.sum((coord - centroid)**2)) 
                    for coord in stratum_coords
                ])
                
                max_distance = np.max(distances_from_centroid)
                if max_distance > 0:
                    normalized_distances = distances_from_centroid / max_distance
                    spatial_variation = (normalized_distances - 0.5) * 0.04  # ±0.02 SVI units
                    predictions[stratum_mask] += spatial_variation
        
        # Apply constraint preservation
        adjustment = tract_svi - np.mean(predictions)
        predictions += adjustment * 0.9  # Strong constraint preservation
        predictions = np.clip(predictions, 0.0, 1.0)
        
        return predictions
    
    def _estimate_baseline_uncertainty(self, predictions: np.ndarray, tract_svi: float) -> np.ndarray:
        """Estimate uncertainty for baseline predictions"""
        
        # Base uncertainty
        base_uncertainty = 0.06
        
        # Higher uncertainty for predictions far from tract mean
        distance_uncertainty = np.abs(predictions - tract_svi) * 0.2
        
        # Spatial variation component
        spatial_std = np.std(predictions)
        spatial_uncertainty = np.full_like(predictions, spatial_std * 0.5)
        
        # Combined uncertainty
        total_uncertainty = base_uncertainty + distance_uncertainty + spatial_uncertainty
        total_uncertainty = np.clip(total_uncertainty, 0.03, 0.20)
        
        return total_uncertainty

    def compare_with_gnn(self, gnn_predictions: pd.DataFrame, 
                        baseline_predictions: pd.DataFrame,
                        tract_svi: float) -> Dict:
        """
        Comprehensive comparison between GNN and baseline methods
        
        Args:
            gnn_predictions: DataFrame with GNN results
            baseline_predictions: DataFrame with baseline results
            tract_svi: True tract SVI value
            
        Returns:
            Dictionary with comparison metrics
        """
        
        # Ensure same number of predictions
        min_length = min(len(gnn_predictions), len(baseline_predictions))
        gnn_vals = gnn_predictions['mean'].values[:min_length]
        baseline_vals = baseline_predictions['mean'].values[:min_length]
        
        # Basic correlation
        correlation, p_value = stats.pearsonr(gnn_vals, baseline_vals)
        
        # Constraint satisfaction
        gnn_constraint_error = abs(np.mean(gnn_vals) - tract_svi) / tract_svi * 100
        baseline_constraint_error = abs(np.mean(baseline_vals) - tract_svi) / tract_svi * 100
        
        # Spatial variation
        gnn_spatial_std = np.std(gnn_vals)
        baseline_spatial_std = np.std(baseline_vals)
        
        # Prediction ranges
        gnn_range = np.ptp(gnn_vals)
        baseline_range = np.ptp(baseline_vals)
        
        # Method performance assessment
        gnn_performance = self._assess_method_performance(
            gnn_vals, tract_svi, method_name="GNN"
        )
        baseline_performance = self._assess_method_performance(
            baseline_vals, tract_svi, method_name="Baseline"
        )
        
        comparison = {
            'correlation': {
                'pearson_r': correlation,
                'p_value': p_value,
                'r_squared': correlation**2 if not np.isnan(correlation) else 0
            },
            'constraint_satisfaction': {
                'gnn_error_percent': gnn_constraint_error,
                'baseline_error_percent': baseline_constraint_error,
                'gnn_better_constraint': gnn_constraint_error < baseline_constraint_error
            },
            'spatial_variation': {
                'gnn_std': gnn_spatial_std,
                'baseline_std': baseline_spatial_std,
                'gnn_more_variable': gnn_spatial_std > baseline_spatial_std,
                'variation_ratio': baseline_spatial_std / gnn_spatial_std if gnn_spatial_std > 0 else 1
            },
            'prediction_ranges': {
                'gnn_range': gnn_range,
                'baseline_range': baseline_range
            },
            'performance_assessment': {
                'gnn': gnn_performance,
                'baseline': baseline_performance
            },
            'summary': {
                'better_constraint': 'GNN' if gnn_constraint_error < baseline_constraint_error else 'Baseline',
                'more_spatial_detail': 'GNN' if gnn_spatial_std > baseline_spatial_std else 'Baseline',
                'correlation_strength': 'Strong' if abs(correlation) > 0.7 else 
                                      'Moderate' if abs(correlation) > 0.3 else 'Weak'
            }
        }
        
        return comparison
    
    def _assess_method_performance(self, predictions: np.ndarray, tract_svi: float, 
                                 method_name: str) -> Dict:
        """Assess performance of a prediction method"""
        
        constraint_error = abs(np.mean(predictions) - tract_svi) / tract_svi * 100
        spatial_std = np.std(predictions)
        pred_range = np.ptp(predictions)
        
        # Quality scores
        constraint_score = max(0, 100 - constraint_error * 2)  # Penalize constraint violations
        variation_score = min(100, spatial_std * 1000)  # Reward reasonable variation
        range_score = min(100, pred_range * 500)  # Reward reasonable prediction range
        
        overall_score = (constraint_score * 0.5 + variation_score * 0.3 + range_score * 0.2)
        
        performance = {
            'constraint_error_percent': constraint_error,
            'spatial_std': spatial_std,
            'prediction_range': pred_range,
            'constraint_score': constraint_score,
            'variation_score': variation_score,
            'range_score': range_score,
            'overall_score': overall_score,
            'quality_rating': (
                'Excellent' if overall_score > 80 else
                'Good' if overall_score > 60 else
                'Fair' if overall_score > 40 else
                'Poor'
            )
        }
        
        return performance

    def create_comparison_summary(self, comparison_results: Dict) -> str:
        """Create human-readable summary of comparison results"""
        
        corr = comparison_results['correlation']['pearson_r']
        r2 = comparison_results['correlation']['r_squared']
        
        gnn_error = comparison_results['constraint_satisfaction']['gnn_error_percent']
        baseline_error = comparison_results['constraint_satisfaction']['baseline_error_percent']
        
        gnn_std = comparison_results['spatial_variation']['gnn_std']
        baseline_std = comparison_results['spatial_variation']['baseline_std']
        
        summary = f"""
GRANITE GNN vs Baseline Comparison Summary:

Method Correlation:
• Pearson correlation: {corr:.3f}
• R-squared: {r2:.3f}
• Interpretation: {comparison_results['summary']['correlation_strength']} agreement

Constraint Satisfaction:
• GNN tract error: {gnn_error:.2f}%
• Baseline tract error: {baseline_error:.2f}%
• Better constraint: {comparison_results['summary']['better_constraint']}

Spatial Detail:
• GNN spatial variation: {gnn_std:.4f}
• Baseline spatial variation: {baseline_std:.4f}
• More spatial detail: {comparison_results['summary']['more_spatial_detail']}

Overall Assessment:
• GNN Performance: {comparison_results['performance_assessment']['gnn']['quality_rating']}
• Baseline Performance: {comparison_results['performance_assessment']['baseline']['quality_rating']}

Key Finding: {self._interpret_comparison_results(comparison_results)}
"""
        
        return summary.strip()
    
    def _interpret_comparison_results(self, results: Dict) -> str:
        """Provide interpretation of comparison results"""
        
        r2 = results['correlation']['r_squared']
        gnn_better_constraint = results['constraint_satisfaction']['gnn_better_constraint']
        gnn_more_variable = results['spatial_variation']['gnn_more_variable']
        
        if r2 > 0.5 and gnn_better_constraint and gnn_more_variable:
            return "GNN successfully learned accessibility patterns with superior constraint satisfaction and spatial detail"
        elif r2 > 0.5 and gnn_better_constraint:
            return "GNN shows good agreement with baseline but with better constraint preservation"
        elif r2 > 0.5:
            return "GNN learned patterns similar to traditional methods"
        elif gnn_better_constraint:
            return "GNN discovered novel patterns while maintaining better tract-level accuracy"
        else:
            return "GNN learned significantly different patterns - requires further validation"

# Convenience function for running complete baseline comparison
def run_baseline_comparison(addresses: pd.DataFrame, 
                          accessibility_features: np.ndarray,
                          gnn_predictions: pd.DataFrame,
                          tract_svi: float) -> Dict:
    """
    Run complete baseline comparison analysis
    
    Main function called by the pipeline for validation
    """
    baseline_analyzer = AccessibilityBaseline()
    
    # Create baseline predictions
    baseline_predictions = baseline_analyzer.create_baseline_predictions(
        addresses, accessibility_features, tract_svi
    )
    
    # Compare with GNN
    comparison_results = baseline_analyzer.compare_with_gnn(
        gnn_predictions, baseline_predictions, tract_svi
    )
    
    # Generate summary
    comparison_summary = baseline_analyzer.create_comparison_summary(comparison_results)
    
    return {
        'baseline_predictions': baseline_predictions,
        'comparison_results': comparison_results,
        'comparison_summary': comparison_summary,
        'validation_passed': (
            comparison_results['constraint_satisfaction']['gnn_error_percent'] < 10 and
            comparison_results['spatial_variation']['gnn_std'] > 0.01
        )
    }