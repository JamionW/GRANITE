"""
Spatial Learning Diagnostics for GRANITE
Add this as granite/evaluation/spatial_diagnostics.py
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class SpatialLearningDiagnostics:
    """
    Diagnostic tools to evaluate if GNN is learning meaningful spatial patterns
    """
    
    def __init__(self, verbose=True):
        self.verbose = verbose
    
    def _log(self, message):
        if self.verbose:
            print(f"[SpatialDiag] {message}")
    
    def compute_spatial_autocorrelation(self, predictions, coordinates, k_neighbors=8):
        """
        Compute Moran's I spatial autocorrelation
        Values near 1 = strong positive spatial correlation
        Values near -1 = strong negative spatial correlation  
        Values near 0 = no spatial correlation (random)
        """
        n = len(predictions)
        
        # Build spatial weights matrix using k-nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(coordinates)
        distances, indices = nbrs.kneighbors(coordinates)
        
        # Create weights matrix
        W = np.zeros((n, n))
        for i in range(n):
            for j_idx in range(1, len(indices[i])):  # Skip self
                j = indices[i][j_idx]
                distance = distances[i][j_idx]
                if distance > 0:
                    W[i, j] = 1.0 / distance  # Inverse distance weighting
        
        # Row-normalize weights
        row_sums = W.sum(axis=1)
        W = W / row_sums[:, np.newaxis]
        W[np.isnan(W)] = 0
        
        # Compute Moran's I
        predictions_centered = predictions - np.mean(predictions)
        numerator = np.sum(W * np.outer(predictions_centered, predictions_centered))
        denominator = np.sum(predictions_centered**2)
        
        if denominator == 0:
            return 0.0
        
        moran_i = (n / np.sum(W)) * (numerator / denominator)
        return moran_i
    
    def evaluate_accessibility_correlation(self, predictions, accessibility_features):
        """
        Test if predictions correlate with accessibility patterns
        """
        # Overall accessibility score
        overall_accessibility = np.mean(accessibility_features, axis=1)
        
        # Feature-specific correlations
        correlations = {}
        
        # Overall correlation
        overall_corr = np.corrcoef(overall_accessibility, predictions)[0, 1]
        correlations['overall'] = overall_corr
        
        # By destination type (assuming 8 features per type)
        if accessibility_features.shape[1] >= 24:
            dest_types = ['employment', 'healthcare', 'grocery']
            for i, dest_type in enumerate(dest_types):
                start_idx = i * 8
                end_idx = (i + 1) * 8
                dest_accessibility = np.mean(accessibility_features[:, start_idx:end_idx], axis=1)
                dest_corr = np.corrcoef(dest_accessibility, predictions)[0, 1]
                correlations[dest_type] = dest_corr
        
        return correlations
    
    def create_baseline_predictions(self, accessibility_features, coordinates, target_mean):
        """
        Create baseline predictions for comparison
        """
        baselines = {}
        
        # 1. Random predictions (same mean/std as target)
        random_preds = np.random.normal(target_mean, 0.02, len(accessibility_features))
        baselines['random'] = random_preds
        
        # 2. Linear regression on accessibility features
        X = accessibility_features
        y = np.full(len(X), target_mean) + np.random.normal(0, 0.01, len(X))  # Synthetic target
        
        lr = LinearRegression()
        lr.fit(X, y)
        linear_preds = lr.predict(X)
        baselines['linear'] = linear_preds
        
        # 3. Spatial smoothing (inverse distance weighting)
        spatial_preds = self._create_spatial_baseline(coordinates, target_mean)
        baselines['spatial'] = spatial_preds
        
        return baselines
    
    def _create_spatial_baseline(self, coordinates, target_mean):
        """Create predictions based on spatial smoothing"""
        n = len(coordinates)
        
        # Add some spatial structure by distance to centroid
        centroid = np.mean(coordinates, axis=0)
        distances_to_center = np.sqrt(np.sum((coordinates - centroid)**2, axis=1))
        max_distance = np.max(distances_to_center)
        
        # Edge locations have slightly different values
        spatial_component = (distances_to_center / max_distance) * 0.02
        spatial_preds = target_mean + spatial_component + np.random.normal(0, 0.005, n)
        
        return spatial_preds
    
    def comprehensive_evaluation(self, raw_predictions, accessibility_features, 
                                coordinates, target_svi):
        """
        Complete diagnostic evaluation of model learning quality
        """
        self._log("Running comprehensive spatial learning evaluation...")
        
        results = {}
        
        # 1. Spatial autocorrelation
        moran_i = self.compute_spatial_autocorrelation(raw_predictions, coordinates)
        results['spatial_autocorrelation'] = moran_i
        
        # 2. Accessibility correlations
        acc_correlations = self.evaluate_accessibility_correlation(raw_predictions, accessibility_features)
        results['accessibility_correlations'] = acc_correlations
        
        # 3. Prediction characteristics
        results['prediction_stats'] = {
            'mean': np.mean(raw_predictions),
            'std': np.std(raw_predictions),
            'range': np.ptp(raw_predictions),
            'min': np.min(raw_predictions),
            'max': np.max(raw_predictions)
        }
        
        # 4. Baseline comparisons
        baselines = self.create_baseline_predictions(accessibility_features, coordinates, target_svi)
        baseline_results = {}
        
        for baseline_name, baseline_preds in baselines.items():
            baseline_moran = self.compute_spatial_autocorrelation(baseline_preds, coordinates)
            baseline_acc_corr = self.evaluate_accessibility_correlation(baseline_preds, accessibility_features)
            
            baseline_results[baseline_name] = {
                'spatial_autocorr': baseline_moran,
                'accessibility_corr': baseline_acc_corr['overall']
            }
        
        results['baseline_comparisons'] = baseline_results
        
        # 5. Learning quality assessment
        quality_assessment = self._assess_learning_quality(results, target_svi)
        results['quality_assessment'] = quality_assessment
        
        return results
    
    def _assess_learning_quality(self, results, target_svi):
        """
        Assess whether the model is learning meaningful patterns
        """
        assessment = {}
        
        # Check spatial structure
        moran_i = results['spatial_autocorrelation']
        random_moran = results['baseline_comparisons']['random']['spatial_autocorr']
        
        assessment['has_spatial_structure'] = moran_i > random_moran + 0.05
        assessment['spatial_strength'] = 'strong' if moran_i > 0.3 else 'moderate' if moran_i > 0.1 else 'weak'
        
        # Check accessibility relationship
        overall_acc_corr = results['accessibility_correlations']['overall']
        assessment['accessibility_correlation_strength'] = abs(overall_acc_corr)
        assessment['meaningful_accessibility_relationship'] = abs(overall_acc_corr) > 0.1
        
        # Check prediction reasonableness
        pred_mean = results['prediction_stats']['mean']
        pred_std = results['prediction_stats']['std']
        
        assessment['reasonable_scale'] = 0.0 <= pred_mean <= 1.0
        assessment['reasonable_variation'] = 0.005 < pred_std < 0.15
        assessment['mean_bias'] = abs(pred_mean - target_svi) / target_svi
        
        # Overall learning quality
        learning_indicators = [
            assessment['has_spatial_structure'],
            assessment['meaningful_accessibility_relationship'], 
            assessment['reasonable_scale'],
            assessment['reasonable_variation'],
            assessment['mean_bias'] < 0.5  # Not too biased
        ]
        
        assessment['learning_quality'] = sum(learning_indicators) / len(learning_indicators)
        
        if assessment['learning_quality'] > 0.8:
            assessment['overall_verdict'] = 'excellent'
        elif assessment['learning_quality'] > 0.6:
            assessment['overall_verdict'] = 'good'
        elif assessment['learning_quality'] > 0.4:
            assessment['overall_verdict'] = 'poor'
        else:
            assessment['overall_verdict'] = 'failing'
        
        return assessment
    
    def print_diagnostic_report(self, results):
        """
        Print a comprehensive diagnostic report
        """
        print("\n" + "="*60)
        print("SPATIAL LEARNING DIAGNOSTIC REPORT")
        print("="*60)
        
        # Spatial autocorrelation
        moran_i = results['spatial_autocorrelation']
        print(f"\n1. SPATIAL AUTOCORRELATION")
        print(f"   Moran's I: {moran_i:.4f}")
        print(f"   Interpretation: {'Strong' if moran_i > 0.3 else 'Moderate' if moran_i > 0.1 else 'Weak'} spatial clustering")
        
        # Accessibility correlations  
        acc_corr = results['accessibility_correlations']['overall']
        print(f"\n2. ACCESSIBILITY-VULNERABILITY RELATIONSHIP")
        print(f"   Overall correlation: {acc_corr:.4f}")
        print(f"   Expected: Negative correlation (better access → lower vulnerability)")
        print(f"   Strength: {'Strong' if abs(acc_corr) > 0.3 else 'Moderate' if abs(acc_corr) > 0.1 else 'Weak'}")
        
        # Prediction characteristics
        stats = results['prediction_stats']
        print(f"\n3. PREDICTION CHARACTERISTICS")
        print(f"   Mean: {stats['mean']:.4f}")
        print(f"   Std: {stats['std']:.4f}")
        print(f"   Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        
        # Baseline comparisons
        print(f"\n4. BASELINE COMPARISONS")
        baselines = results['baseline_comparisons']
        for name, baseline in baselines.items():
            print(f"   {name.capitalize()}:")
            print(f"     Spatial autocorr: {baseline['spatial_autocorr']:.4f}")
            print(f"     Accessibility corr: {baseline['accessibility_corr']:.4f}")
        
        # Overall assessment
        quality = results['quality_assessment']
        print(f"\n5. LEARNING QUALITY ASSESSMENT")
        print(f"   Overall verdict: {quality['overall_verdict'].upper()}")
        print(f"   Learning quality score: {quality['learning_quality']:.2f}")
        print(f"   Has spatial structure: {quality['has_spatial_structure']}")
        print(f"   Meaningful accessibility relationship: {quality['meaningful_accessibility_relationship']}")
        print(f"   Mean bias: {quality['mean_bias']:.2%}")
        
        # Recommendations
        print(f"\n6. RECOMMENDATIONS")
        if quality['overall_verdict'] == 'failing':
            print("   ❌ Model is not learning meaningful patterns")
            print("   → Consider revising architecture, features, or training approach")
            print("   → Mean adjustment is masking fundamental model failure")
        elif quality['overall_verdict'] == 'poor':
            print("   ⚠️  Model learning is weak but present")
            print("   → Investigate training stability and feature engineering")
            print("   → Mean adjustment may be appropriate but monitor closely")
        else:
            print("   ✅ Model appears to be learning meaningful patterns")
            print("   → Mean adjustment is likely correcting systematic bias")
            print("   → Continue monitoring spatial and accessibility relationships")
        
        print("\n" + "="*60)
        
        return quality['overall_verdict']