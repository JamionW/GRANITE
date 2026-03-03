"""
Spatial Learning Diagnostics for GRANITE

Provides diagnostic tools for evaluating spatial pattern learning quality
including Moran's I, accessibility correlations, and baseline comparisons.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class SpatialLearningDiagnostics:
 """
 Enhanced diagnostic tools to evaluate if GNN is learning meaningful spatial patterns
 Maintains original API while adding diagnostic capabilities
 """
 
 def __init__(self, verbose=True):
 self.verbose = verbose
 self.diagnostic_results = {}
 
 def _log(self, message):
 if self.verbose:
 print(f"[SpatialDiag] {message}")
 
 # ===== ORIGINAL API METHODS (MAINTAINED FOR COMPATIBILITY) =====
 
 def compute_spatial_autocorrelation(self, predictions, coordinates, k_neighbors=8):
 """
 Original method - compute Moran's I spatial autocorrelation
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
 for j_idx in range(1, len(indices[i])): # Skip self
 j = indices[i][j_idx]
 distance = distances[i][j_idx]
 if distance > 0:
 W[i, j] = 1.0 / distance # Inverse distance weighting
 
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
 """Original method - test if predictions correlate with accessibility patterns"""
 
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
 """Original method - create baseline predictions for comparison"""
 
 baselines = {}
 
 # 1. Random predictions (same mean/std as target)
 random_preds = np.random.normal(target_mean, 0.02, len(accessibility_features))
 baselines['random'] = random_preds
 
 # 2. Linear regression on accessibility features
 X = accessibility_features
 y = np.full(len(X), target_mean) + np.random.normal(0, 0.01, len(X)) # Synthetic target
 
 lr = LinearRegression()
 lr.fit(X, y)
 linear_preds = lr.predict(X)
 baselines['linear'] = linear_preds
 
 # 3. Spatial smoothing (inverse distance weighting)
 spatial_preds = self._create_spatial_baseline(coordinates, target_mean)
 baselines['spatial'] = spatial_preds
 
 return baselines
 
 def_evaluation(self, raw_predictions, accessibility_features, 
 coordinates, target_svi):
 """
 Original method - complete diagnostic evaluation of model learning quality
 Enhanced with additional diagnostics
 """
 self._log("Running spatial learning evaluation...")
 
 results = {}
 
 # Store data for enhanced diagnostics
 self.raw_predictions = raw_predictions
 self.accessibility_features = accessibility_features
 self.coordinates = coordinates
 self.target_svi = target_svi
 self.n_addresses = len(raw_predictions)
 
 # 1. Original spatial autocorrelation
 moran_i = self.compute_spatial_autocorrelation(raw_predictions, coordinates)
 results['spatial_autocorrelation'] = moran_i
 
 # 2. Original accessibility correlations
 acc_correlations = self.evaluate_accessibility_correlation(raw_predictions, accessibility_features)
 results['accessibility_correlations'] = acc_correlations
 
 # 3. Original prediction characteristics
 results['prediction_stats'] = {
 'mean': np.mean(raw_predictions),
 'std': np.std(raw_predictions),
 'range': np.ptp(raw_predictions),
 'min': np.min(raw_predictions),
 'max': np.max(raw_predictions)
 }
 
 # 4. Original baseline comparisons
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
 
 # 5. Original learning quality assessment
 quality_assessment = self._assess_learning_quality(results, target_svi)
 results['quality_assessment'] = quality_assessment
 
 self._log("Running enhanced diagnostic analysis...")
 enhanced_results = self._run_enhanced_diagnostics()
 results['enhanced_diagnostics'] = enhanced_results
 
 # Store results for further analysis
 self.diagnostic_results = results
 
 return results
 
 def print_diagnostic_report(self, results):
 """
 Original method - print a diagnostic report
 Enhanced with additional insights
 """
 print("\n" + "="*60)
 print("SPATIAL LEARNING DIAGNOSTIC REPORT")
 print("="*60)
 
 # Original spatial autocorrelation
 moran_i = results['spatial_autocorrelation']
 print(f"\n1. SPATIAL AUTOCORRELATION")
 print(f" Moran's I: {moran_i:.4f}")
 print(f" Interpretation: {'Strong' if moran_i > 0.3 else 'Moderate' if moran_i > 0.1 else 'Weak'} spatial clustering")
 
 # Original accessibility correlations 
 acc_corr = results['accessibility_correlations']['overall']
 print(f"\n2. ACCESSIBILITY-VULNERABILITY RELATIONSHIP")
 print(f" Overall correlation: {acc_corr:.4f}")
 print(f" Expected: Negative correlation (better access -> lower vulnerability)")
 print(f" Strength: {'Strong' if abs(acc_corr) > 0.3 else 'Moderate' if abs(acc_corr) > 0.1 else 'Weak'}")
 
 # Original prediction characteristics
 stats = results['prediction_stats']
 print(f"\n3. PREDICTION CHARACTERISTICS")
 print(f" Mean: {stats['mean']:.4f}")
 print(f" Std: {stats['std']:.4f}")
 print(f" Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
 
 # Original baseline comparisons
 print(f"\n4. BASELINE COMPARISONS")
 baselines = results['baseline_comparisons']
 for name, baseline in baselines.items():
 print(f" {name.capitalize()}:")
 print(f" Spatial autocorr: {baseline['spatial_autocorr']:.4f}")
 print(f" Accessibility corr: {baseline['accessibility_corr']:.4f}")
 
 # Original overall assessment
 quality = results['quality_assessment']
 print(f"\n5. LEARNING QUALITY ASSESSMENT")
 print(f" Overall verdict: {quality['overall_verdict'].upper()}")
 print(f" Learning quality score: {quality['learning_quality']:.2f}")
 print(f" Has spatial structure: {quality['has_spatial_structure']}")
 print(f" Meaningful accessibility relationship: {quality['meaningful_accessibility_relationship']}")
 print(f" Mean bias: {quality['mean_bias']:.2%}")
 
 if 'enhanced_diagnostics' in results:
 self._print_enhanced_diagnostic_section(results['enhanced_diagnostics'])
 
 # Original recommendations
 print(f"\n6. RECOMMENDATIONS")
 if quality['overall_verdict'] == 'failing':
 print(" Model is not learning meaningful patterns")
 print(" -> Consider revising architecture, features, or training approach")
 print(" -> Mean adjustment is masking fundamental model failure")
 elif quality['overall_verdict'] == 'poor':
 print(" Model learning is weak but present")
 print(" -> Investigate training stability and feature engineering")
 print(" -> Mean adjustment may be appropriate but monitor closely")
 else:
 print(" Model appears to be learning meaningful patterns")
 print(" -> Mean adjustment is likely correcting systematic bias")
 print(" -> Continue monitoring spatial and accessibility relationships")
 
 # Enhanced recommendations
 if 'enhanced_diagnostics' in results:
 enhanced_recs = self._get_enhanced_recommendations(results)
 if enhanced_recs:
 print(f"\n7. ENHANCED RECOMMENDATIONS")
 for rec in enhanced_recs:
 print(f" -> {rec}")
 
 print("\n" + "="*60)
 
 return quality['overall_verdict']
 
 # ===== ENHANCED DIAGNOSTIC METHODS (NEW) =====
 
 def _run_enhanced_diagnostics(self) -> dict:
 """Run enhanced diagnostics"""
 
 enhanced_results = {}
 
 # 1. Feature Engineering Analysis
 enhanced_results['feature_analysis'] = self._analyze_accessibility_features()
 
 # 2. Correlation Direction Analysis 
 enhanced_results['correlation_analysis'] = self._analyze_correlation_directions()
 
 # 3. Spatial Pattern Analysis
 enhanced_results['spatial_analysis'] = self._analyze_spatial_patterns()
 
 # 4. Model Behavior Analysis
 enhanced_results['model_analysis'] = self._analyze_model_behavior()
 
 # 5. Root Cause Identification
 enhanced_results['root_cause'] = self._identify_root_causes(enhanced_results)
 
 return enhanced_results
 
 def _analyze_accessibility_features(self):
 """
 FIXED: Analyze accessibility features with flexible feature count
 """
 if self.accessibility_features is None:
 return {
 'error': 'No accessibility features available',
 'suspicious_patterns': {}
 }
 
 try:
 n_addresses, n_features = self.accessibility_features.shape
 
 # Generate flexible feature names if not provided
 if not hasattr(self, 'feature_names') or self.feature_names is None:
 # Create generic feature names to match actual feature count
 feature_names = [f'feature_{i}' for i in range(n_features)]
 else:
 feature_names = self.feature_names
 
 # Ensure feature names match feature count
 if len(feature_names) != n_features:
 self._log(f"WARNING: Feature name count ({len(feature_names)}) != feature count ({n_features})")
 # Adjust feature names to match actual features
 if len(feature_names) > n_features:
 feature_names = feature_names[:n_features]
 else:
 # Extend with generic names
 feature_names = feature_names + [f'feature_{i}' for i in range(len(feature_names), n_features)]
 
 # Calculate feature statistics safely
 feature_stats = pd.DataFrame({
 'feature': feature_names,
 'mean': np.mean(self.accessibility_features, axis=0),
 'std': np.std(self.accessibility_features, axis=0),
 'min': np.min(self.accessibility_features, axis=0),
 'max': np.max(self.accessibility_features, axis=0),
 'range': np.ptp(self.accessibility_features, axis=0)
 })
 
 # Basic feature analysis
 zero_var_features = feature_stats[feature_stats['std'] < 1e-8]['feature'].tolist()
 high_var_features = feature_stats[feature_stats['std'] > feature_stats['std'].median() * 2]['feature'].tolist()
 
 # Feature correlation analysis (handle large matrices efficiently)
 if n_features > 50:
 # Sample features for correlation analysis
 sample_indices = np.random.choice(n_features, 50, replace=False)
 sample_features = self.accessibility_features[:, sample_indices]
 corr_matrix = np.corrcoef(sample_features.T)
 sample_names = [feature_names[i] for i in sample_indices]
 else:
 corr_matrix = np.corrcoef(self.accessibility_features.T)
 sample_names = feature_names
 
 # Find high correlations
 high_correlations = []
 if corr_matrix.shape[0] > 1:
 for i in range(len(sample_names)):
 for j in range(i + 1, len(sample_names)):
 if abs(corr_matrix[i, j]) > 0.9:
 high_correlations.append({
 'feature1': sample_names[i],
 'feature2': sample_names[j],
 'correlation': corr_matrix[i, j]
 })
 
 # CRITICAL: Create suspicious_patterns analysis
 suspicious_patterns = {}
 
 # Check travel time features
 time_features = [i for i, name in enumerate(feature_names) if 'time' in name.lower()]
 if time_features:
 time_values = self.accessibility_features[:, time_features]
 suspicious_patterns['unrealistic_times'] = {
 'extremely_high': bool(np.any(time_values > 300)), # > 5 hours
 'negative_times': bool(np.any(time_values < 0)),
 'zero_variance': bool(np.any(np.std(time_values, axis=0) < 0.01))
 }
 else:
 suspicious_patterns['unrealistic_times'] = {
 'extremely_high': False,
 'negative_times': False,
 'zero_variance': False
 }
 
 # Check count features
 count_features = [i for i, name in enumerate(feature_names) if 'count' in name.lower()]
 if count_features:
 count_values = self.accessibility_features[:, count_features]
 suspicious_patterns['unrealistic_counts'] = {
 'extremely_high': bool(np.any(count_values > 50)), # > 50 destinations
 'negative_counts': bool(np.any(count_values < 0)),
 'all_zeros': bool(np.any(np.all(count_values == 0, axis=0)))
 }
 else:
 suspicious_patterns['unrealistic_counts'] = {
 'extremely_high': False,
 'negative_counts': False,
 'all_zeros': False
 }
 
 # Check for perfect correlations
 if corr_matrix.shape[0] > 1:
 perfect_corr_mask = (np.abs(corr_matrix) > 0.99) & (corr_matrix != 1.0)
 suspicious_patterns['perfect_correlations'] = bool(np.any(perfect_corr_mask))
 else:
 suspicious_patterns['perfect_correlations'] = False
 
 # Check for negative accessibility values
 suspicious_patterns['negative_accessibility'] = bool(np.any(self.accessibility_features < 0))
 
 return {
 'n_features': n_features,
 'feature_stats': feature_stats,
 'zero_variance_features': zero_var_features,
 'high_variance_features': high_var_features,
 'high_correlations': high_correlations,
 'suspicious_patterns': suspicious_patterns, # This key is required by downstream code
 'feature_quality_score': len(zero_var_features) / n_features if n_features > 0 else 0 # Lower is better
 }
 
 except Exception as e:
 self._log(f"Error in feature analysis: {str(e)}")
 return {
 'error': f'Feature analysis failed: {str(e)}',
 'suspicious_patterns': {
 'unrealistic_times': {'extremely_high': False, 'negative_times': False, 'zero_variance': False},
 'unrealistic_counts': {'extremely_high': False, 'negative_counts': False, 'all_zeros': False},
 'perfect_correlations': False,
 'negative_accessibility': False
 },
 'n_features': 0,
 'feature_stats': pd.DataFrame(),
 'zero_variance_features': [],
 'high_variance_features': [],
 'high_correlations': [],
 'feature_quality_score': 1.0
 }
 
 def _analyze_correlation_directions(self):
 """Analyze correlation directions to identify problematic features"""
 
 analysis = {}
 feature_names = self._generate_feature_names(self.accessibility_features.shape[1])
 
 # 1. Individual feature correlations with predictions
 feature_correlations = {}
 for i, name in enumerate(feature_names):
 corr = np.corrcoef(self.accessibility_features[:, i], self.raw_predictions)[0, 1]
 feature_correlations[name] = corr
 
 analysis['individual_correlations'] = feature_correlations
 
 # 2. Expected vs actual correlation directions
 expected_directions = {}
 problematic_features = []
 
 for name, corr in feature_correlations.items():
 if 'time' in name:
 # Travel times should correlate POSITIVELY with vulnerability 
 expected_directions[name] = 'positive'
 if corr < 0:
 problematic_features.append((name, corr, 'should_be_positive'))
 elif 'count' in name or 'score' in name:
 # Counts and scores should correlate NEGATIVELY with vulnerability
 expected_directions[name] = 'negative'
 if corr > 0:
 problematic_features.append((name, corr, 'should_be_negative'))
 
 analysis['expected_directions'] = expected_directions
 analysis['problematic_features'] = problematic_features
 
 # 3. Overall accessibility indices
 time_features = [i for i, name in enumerate(feature_names) if 'time' in name]
 count_features = [i for i, name in enumerate(feature_names) if 'count' in name]
 score_features = [i for i, name in enumerate(feature_names) if 'score' in name]
 
 accessibility_indices = {}
 
 if time_features:
 avg_time = np.mean(self.accessibility_features[:, time_features], axis=1)
 time_corr = np.corrcoef(avg_time, self.raw_predictions)[0, 1]
 accessibility_indices['average_travel_time'] = {
 'values': avg_time,
 'correlation': time_corr,
 'expected_sign': 'positive',
 'problematic': time_corr < 0
 }
 
 if count_features:
 avg_count = np.mean(self.accessibility_features[:, count_features], axis=1)
 count_corr = np.corrcoef(avg_count, self.raw_predictions)[0, 1]
 accessibility_indices['average_destination_count'] = {
 'values': avg_count,
 'correlation': count_corr,
 'expected_sign': 'negative',
 'problematic': count_corr > 0
 }
 
 if score_features:
 avg_score = np.mean(self.accessibility_features[:, score_features], axis=1)
 score_corr = np.corrcoef(avg_score, self.raw_predictions)[0, 1]
 accessibility_indices['average_accessibility_score'] = {
 'values': avg_score,
 'correlation': score_corr,
 'expected_sign': 'negative',
 'problematic': score_corr > 0
 }
 
 analysis['accessibility_indices'] = accessibility_indices
 
 return analysis
 
 def _analyze_spatial_patterns(self):
 """Analyze spatial patterns to identify over-smoothing or confounding"""
 
 analysis = {}
 
 # 1. Distance-based correlation analysis
 analysis['distance_correlation'] = self._analyze_distance_correlations()
 
 # 2. Edge vs center analysis
 analysis['edge_center_analysis'] = self._analyze_edge_center_patterns()
 
 # 3. Spatial clustering analysis
 analysis['clustering_analysis'] = self._analyze_spatial_clustering()
 
 return analysis
 
 def _analyze_model_behavior(self):
 """Analyze what the model might be learning instead of accessibility"""
 
 analysis = {}
 
 # 1. Feature importance via random forest
 analysis['feature_importance'] = self._analyze_feature_importance()
 
 # 2. Principal component analysis
 analysis['pca_analysis'] = self._analyze_principal_components()
 
 # 3. Non-linear relationship analysis
 analysis['nonlinear_analysis'] = self._analyze_nonlinear_relationships()
 
 return analysis
 
 def _identify_root_causes(self, analysis_results):
 """Identify most likely root causes based on all analyses"""
 
 root_causes = []
 confidence_scores = {}
 
 # Analyze feature engineering issues
 feature_analysis = analysis_results['feature_analysis']
 if feature_analysis['suspicious_patterns'].get('unrealistic_times', {}).get('extremely_high', False):
 root_causes.append("Unrealistic travel times detected")
 confidence_scores['data_quality'] = 0.8
 
 # Analyze correlation direction issues
 correlation_analysis = analysis_results['correlation_analysis']
 problematic_features = correlation_analysis['problematic_features']
 
 feature_names = self._generate_feature_names(self.accessibility_features.shape[1])
 if len(problematic_features) > len(feature_names) // 2:
 root_causes.append("Majority of features have wrong correlation direction")
 confidence_scores['feature_sign_reversal'] = 0.9
 
 # Analyze spatial over-smoothing
 spatial_autocorr = self.compute_spatial_autocorrelation(self.raw_predictions, self.coordinates)
 
 if spatial_autocorr > 0.8:
 root_causes.append("Extremely high spatial autocorrelation suggests over-smoothing")
 confidence_scores['spatial_oversmoothing'] = 0.8
 
 # Check baseline comparisons
 baselines = self.create_baseline_predictions(self.accessibility_features, self.coordinates, self.target_svi)
 corrected_baseline = self._create_corrected_baseline()
 
 overall_access = np.mean(self.accessibility_features, axis=1)
 corrected_corr = np.corrcoef(corrected_baseline, overall_access)[0, 1]
 
 if abs(corrected_corr) > 0.3 and corrected_corr < 0:
 root_causes.append("Corrected features show expected negative correlation")
 confidence_scores['feature_correction_works'] = 0.9
 
 # Primary diagnosis
 primary_cause = "Unknown"
 if confidence_scores.get('feature_sign_reversal', 0) > 0.8:
 primary_cause = "Feature sign reversal - GNN learning wrong accessibility directions"
 elif confidence_scores.get('spatial_oversmoothing', 0) > 0.8:
 primary_cause = "Spatial over-smoothing - model ignoring accessibility patterns"
 elif confidence_scores.get('data_quality', 0) > 0.8:
 primary_cause = "Data quality issues - unrealistic accessibility values"
 
 return {
 'identified_causes': root_causes,
 'confidence_scores': confidence_scores,
 'primary_diagnosis': primary_cause,
 'recommended_fixes': self._recommend_fixes(confidence_scores)
 }
 
 # ===== HELPER METHODS =====
 
 def _generate_feature_names(self, n_features):
 """Generate descriptive feature names based on GRANITE structure"""
 base_features = []
 
 # Employment, healthcare, grocery features (8 each)
 for dest_type in ['employment', 'healthcare', 'grocery']:
 base_features.extend([
 f'{dest_type}_min_time',
 f'{dest_type}_mean_time', 
 f'{dest_type}_90th_time',
 f'{dest_type}_count_30min',
 f'{dest_type}_count_60min',
 f'{dest_type}_count_90min',
 f'{dest_type}_transit_share',
 f'{dest_type}_accessibility_score'
 ])
 
 # Derived features (4)
 derived_features = [
 'total_accessibility',
 'accessibility_diversity',
 'avg_transit_dependence', 
 'time_efficiency'
 ]
 
 # Return appropriate subset
 all_features = base_features + derived_features
 return all_features[:n_features]
 
 def _check_feature_directionality(self, feature_names):
 """Check if accessibility features have expected directional relationships"""
 
 # Create synthetic "good accessibility" score
 good_access_components = []
 
 for i, name in enumerate(feature_names):
 feature_values = self.accessibility_features[:, i]
 
 if 'time' in name:
 # Lower times are better - invert
 normalized = 1 / (1 + feature_values)
 good_access_components.append(normalized)
 elif 'count' in name or 'score' in name:
 # Higher counts/scores are better - use as is
 if np.ptp(feature_values) > 0:
 normalized = (feature_values - np.min(feature_values)) / np.ptp(feature_values)
 good_access_components.append(normalized)
 else:
 # Unknown feature - skip
 continue
 
 if good_access_components:
 synthetic_good_access = np.mean(good_access_components, axis=0)
 
 # Check correlations
 correlations = {}
 for i, name in enumerate(feature_names):
 corr = np.corrcoef(self.accessibility_features[:, i], synthetic_good_access)[0, 1]
 correlations[name] = corr
 
 return {
 'synthetic_accessibility': synthetic_good_access,
 'feature_correlations': correlations,
 'expected_negative': [name for name, corr in correlations.items() 
 if 'time' in name and corr > 0],
 'expected_positive': [name for name, corr in correlations.items() 
 if ('count' in name or 'score' in name) and corr < 0]
 }
 
 return {'error': 'Could not create synthetic accessibility score'}
 
 def _analyze_distance_correlations(self):
 """Analyze how predictions correlate with spatial distances"""
 
 # Distance to centroid
 centroid = np.mean(self.coordinates, axis=0)
 distances_to_center = np.sqrt(np.sum((self.coordinates - centroid)**2, axis=1))
 
 # Distance correlations
 center_corr = np.corrcoef(distances_to_center, self.raw_predictions)[0, 1]
 
 return {
 'distance_to_center_correlation': center_corr,
 'center_distance_range': [np.min(distances_to_center), np.max(distances_to_center)]
 }
 
 def _analyze_edge_center_patterns(self):
 """Analyze difference between edge and center predictions"""
 
 # Find center and edge addresses
 centroid = np.mean(self.coordinates, axis=0)
 distances_to_center = np.sqrt(np.sum((self.coordinates - centroid)**2, axis=1))
 
 # Define center and edge groups
 center_threshold = np.percentile(distances_to_center, 33)
 edge_threshold = np.percentile(distances_to_center, 67)
 
 center_mask = distances_to_center <= center_threshold
 edge_mask = distances_to_center >= edge_threshold
 
 center_preds = self.raw_predictions[center_mask]
 edge_preds = self.raw_predictions[edge_mask]
 
 analysis = {
 'center_addresses': np.sum(center_mask),
 'edge_addresses': np.sum(edge_mask),
 'center_mean_prediction': np.mean(center_preds) if len(center_preds) > 0 else np.nan,
 'edge_mean_prediction': np.mean(edge_preds) if len(edge_preds) > 0 else np.nan,
 'center_std': np.std(center_preds) if len(center_preds) > 0 else np.nan,
 'edge_std': np.std(edge_preds) if len(edge_preds) > 0 else np.nan
 }
 
 # Statistical test for difference
 if len(center_preds) > 0 and len(edge_preds) > 0:
 t_stat, p_value = stats.ttest_ind(center_preds, edge_preds)
 analysis['statistical_difference'] = {
 't_statistic': t_stat,
 'p_value': p_value,
 'significant': p_value < 0.05
 }
 
 return analysis
 
 def _analyze_spatial_clustering(self):
 """Analyze spatial clustering patterns in predictions"""
 
 n_clusters = min(5, self.n_addresses // 10)
 if n_clusters < 2:
 return {'error': 'Too few addresses for clustering analysis'}
 
 kmeans = KMeans(n_clusters=n_clusters, random_state=42)
 cluster_labels = kmeans.fit_predict(self.coordinates)
 
 # Analyze predictions by cluster
 cluster_analysis = {}
 for cluster_id in range(n_clusters):
 cluster_mask = cluster_labels == cluster_id
 cluster_preds = self.raw_predictions[cluster_mask]
 
 cluster_analysis[f'cluster_{cluster_id}'] = {
 'size': np.sum(cluster_mask),
 'mean_prediction': np.mean(cluster_preds),
 'std_prediction': np.std(cluster_preds)
 }
 
 # Compute variance components
 total_variance = np.var(self.raw_predictions)
 within_cluster_variance = 0
 
 for cluster_id in range(n_clusters):
 cluster_mask = cluster_labels == cluster_id
 cluster_preds = self.raw_predictions[cluster_mask]
 cluster_weight = np.sum(cluster_mask) / self.n_addresses
 within_cluster_variance += cluster_weight * np.var(cluster_preds)
 
 between_cluster_variance = total_variance - within_cluster_variance
 
 return {
 'clusters': cluster_analysis,
 'total_variance': total_variance,
 'within_cluster_variance': within_cluster_variance,
 'between_cluster_variance': between_cluster_variance,
 'variance_ratio': between_cluster_variance / total_variance
 }
 
 def _analyze_feature_importance(self):
 """FIXED: Feature importance analysis with proper array handling"""
 
 feature_names = self._generate_feature_names(self.accessibility_features.shape[1])
 
 # FIXED: Ensure feature names match feature count
 n_features = self.accessibility_features.shape[1]
 if len(feature_names) != n_features:
 self._log(f"WARNING: Feature name count ({len(feature_names)}) != feature count ({n_features})")
 # Adjust feature names to match
 if len(feature_names) > n_features:
 feature_names = feature_names[:n_features]
 else:
 # Extend with generic names
 feature_names = feature_names + [f'feature_{i}' for i in range(len(feature_names), n_features)]
 
 try:
 # Use random forest to estimate feature importance
 from sklearn.ensemble import RandomForestRegressor
 rf = RandomForestRegressor(n_estimators=100, random_state=42)
 rf.fit(self.accessibility_features, self.raw_predictions)
 
 # FIXED: Ensure arrays have same length
 importance_values = rf.feature_importances_
 
 if len(importance_values) != len(feature_names):
 self.log(f"ERROR: Importance values ({len(importance_values)}) != feature names ({len(feature_names)})")
 # Use minimum length
 min_len = min(len(importance_values), len(feature_names))
 importance_values = importance_values[:min_len]
 feature_names = feature_names[:min_len]
 
 importance_df = pd.DataFrame({
 'feature': feature_names,
 'importance': importance_values
 }).sort_values('importance', ascending=False)
 
 return {
 'feature_importance_ranking': importance_df,
 'top_3_features': importance_df.head(3)['feature'].tolist(),
 'random_forest_score': rf.score(self.accessibility_features, self.raw_predictions)
 }
 
 except Exception as e:
 self.log(f"Error in feature importance analysis: {str(e)}")
 return {
 'error': str(e),
 'feature_importance_ranking': pd.DataFrame(),
 'top_3_features': [],
 'random_forest_score': 0.0
 }
 
 def _analyze_principal_components(self):
 """Analyze principal components of accessibility features"""
 
 # Standardize features
 scaler = StandardScaler()
 features_scaled = scaler.fit_transform(self.accessibility_features)
 
 # PCA analysis
 pca = PCA()
 pca_features = pca.fit_transform(features_scaled)
 
 # Analyze PC correlations with predictions
 pc_correlations = []
 for i in range(min(5, pca_features.shape[1])): # Top 5 PCs
 corr = np.corrcoef(pca_features[:, i], self.raw_predictions)[0, 1]
 pc_correlations.append(corr)
 
 feature_names = self._generate_feature_names(self.accessibility_features.shape[1])
 
 return {
 'explained_variance_ratio': pca.explained_variance_ratio_[:5],
 'cumulative_variance': np.cumsum(pca.explained_variance_ratio_)[:5],
 'pc_correlations': pc_correlations,
 'first_pc_loadings': dict(zip(feature_names, pca.components_[0]))
 }
 
 def _analyze_nonlinear_relationships(self):
 """Check for non-linear relationships that might confuse interpretation"""
 
 # Overall accessibility
 overall_accessibility = np.mean(self.accessibility_features, axis=1)
 
 # Linear correlation
 linear_corr = np.corrcoef(overall_accessibility, self.raw_predictions)[0, 1]
 
 # Quadratic correlation
 quadratic_features = np.column_stack([
 overall_accessibility,
 overall_accessibility**2
 ])
 
 lr = LinearRegression()
 lr.fit(quadratic_features, self.raw_predictions)
 quadratic_r2 = lr.score(quadratic_features, self.raw_predictions)
 
 return {
 'linear_correlation': linear_corr,
 'quadratic_r2': quadratic_r2,
 'nonlinearity_improvement': quadratic_r2 - linear_corr**2
 }
 
 def _create_spatial_baseline(self, coordinates, target_mean):
 """Create predictions based on spatial smoothing"""
 
 # Distance-based prediction
 centroid = np.mean(coordinates, axis=0)
 distances = np.sqrt(np.sum((coordinates - centroid)**2, axis=1))
 
 # Normalize and add noise
 if np.ptp(distances) > 0:
 normalized_distances = (distances - np.min(distances)) / np.ptp(distances)
 spatial_predictions = target_mean + (normalized_distances - 0.5) * 0.02
 else:
 spatial_predictions = np.full(len(coordinates), target_mean)
 
 return spatial_predictions
 
 def _create_corrected_baseline(self):
 """Create baseline with theoretically correct accessibility relationships"""
 
 feature_names = self._generate_feature_names(self.accessibility_features.shape[1])
 
 # Identify time and count features
 time_indices = [i for i, name in enumerate(feature_names) if 'time' in name]
 count_indices = [i for i, name in enumerate(feature_names) if 'count' in name]
 score_indices = [i for i, name in enumerate(feature_names) if 'score' in name]
 
 accessibility_components = []
 
 # Times should contribute positively to vulnerability
 if time_indices:
 avg_time = np.mean(self.accessibility_features[:, time_indices], axis=1)
 if np.ptp(avg_time) > 0:
 time_component = (avg_time - np.min(avg_time)) / np.ptp(avg_time)
 accessibility_components.append(time_component)
 
 # Counts should contribute negatively to vulnerability
 if count_indices:
 avg_count = np.mean(self.accessibility_features[:, count_indices], axis=1)
 if np.ptp(avg_count) > 0:
 count_component = 1 - ((avg_count - np.min(avg_count)) / np.ptp(avg_count))
 accessibility_components.append(count_component)
 
 # Scores should contribute negatively to vulnerability
 if score_indices:
 avg_score = np.mean(self.accessibility_features[:, score_indices], axis=1)
 if np.ptp(avg_score) > 0:
 score_component = 1 - ((avg_score - np.min(avg_score)) / np.ptp(avg_score))
 accessibility_components.append(score_component)
 
 if accessibility_components:
 vulnerability_index = np.mean(accessibility_components, axis=0)
 # Scale to reasonable range around target SVI
 scaled_vulnerability = self.target_svi + (vulnerability_index - 0.5) * 0.2
 return np.clip(scaled_vulnerability, 0, 1)
 else:
 return np.full(self.n_addresses, self.target_svi)
 
 def _recommend_fixes(self, confidence_scores):
 """Recommend specific fixes based on identified issues"""
 
 fixes = []
 
 if confidence_scores.get('feature_sign_reversal', 0) > 0.7:
 fixes.append({
 'issue': 'Feature sign reversal',
 'fix': 'Invert time-based features or modify loss function',
 'implementation': 'Transform time features: -1 * log(time + 1)',
 'priority': 'HIGH'
 })
 
 if confidence_scores.get('spatial_oversmoothing', 0) > 0.7:
 fixes.append({
 'issue': 'Spatial over-smoothing',
 'fix': 'Reduce graph connectivity or add accessibility-specific edges',
 'implementation': 'Limit k-NN to 4-6 neighbors, add accessibility-based edges',
 'priority': 'HIGH'
 })
 
 if confidence_scores.get('data_quality', 0) > 0.7:
 fixes.append({
 'issue': 'Data quality',
 'fix': 'Validate and clean accessibility computations',
 'implementation': 'Cap travel times at 120 min, validate distance calculations',
 'priority': 'CRITICAL'
 })
 
 return fixes
 
 def _print_enhanced_diagnostic_section(self, enhanced_results):
 """Print enhanced diagnostic section"""
 
 print(f"\n--- ENHANCED DIAGNOSTIC ANALYSIS ---")
 
 # Feature analysis
 if 'feature_analysis' in enhanced_results:
 feature_analysis = enhanced_results['feature_analysis']
 
 # Count problematic patterns - handle both dict and direct boolean values
 problematic_patterns = 0
 for pattern_group in feature_analysis['suspicious_patterns'].values():
 if isinstance(pattern_group, dict):
 # Dictionary of patterns
 problematic_patterns += sum(1 for pattern in pattern_group.values() if pattern)
 else:
 # Direct boolean value
 if pattern_group:
 problematic_patterns += 1
 
 print(f" Feature quality issues detected: {problematic_patterns}")
 
 # Correlation analysis
 if 'correlation_analysis' in enhanced_results:
 correlation_analysis = enhanced_results['correlation_analysis']
 problematic_features = len(correlation_analysis['problematic_features'])
 print(f" Features with wrong correlation direction: {problematic_features}")
 
 # Root cause
 if 'root_cause' in enhanced_results:
 root_cause = enhanced_results['root_cause']
 print(f" Primary root cause: {root_cause['primary_diagnosis']}")
 
 def _get_enhanced_recommendations(self, results):
 """Get enhanced recommendations based on diagnostics"""
 
 recommendations = []
 
 if 'enhanced_diagnostics' in results:
 enhanced = results['enhanced_diagnostics']
 
 if 'root_cause' in enhanced:
 fixes = enhanced['root_cause']['recommended_fixes']
 for fix in fixes:
 recommendations.append(f"{fix['priority']}: {fix['implementation']}")
 
 return recommendations
 
 # ===== ORIGINAL HELPER METHODS (MAINTAINED) =====
 
 def _assess_learning_quality(self, results, target_svi):
 """Original method - assess whether the model is learning meaningful patterns"""
 
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
 assessment['mean_bias'] < 0.5
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
 
 # ===== VISUALIZATION METHODS (NEW) =====
 
 def create_enhanced_diagnostic_plots(self, output_dir='./enhanced_diagnostics'):
 """Create diagnostic visualizations"""
 
 import os
 os.makedirs(output_dir, exist_ok=True)
 
 if not hasattr(self, 'diagnostic_results') or not self.diagnostic_results:
 self._log("No diagnostic results available. Run_evaluation first.")
 return
 
 # 1. Feature correlation analysis
 self._plot_feature_correlations(output_dir)
 
 # 2. Spatial pattern analysis
 self._plot_spatial_patterns(output_dir)
 
 # 3. Accessibility relationship analysis
 self._plot_accessibility_relationships(output_dir)
 
 self._log(f"Enhanced diagnostic plots saved to {output_dir}")
 
 def _plot_feature_correlations(self, output_dir):
 """Plot feature correlation analysis"""
 
 if 'enhanced_diagnostics' not in self.diagnostic_results:
 return
 
 enhanced = self.diagnostic_results['enhanced_diagnostics']
 
 fig, axes = plt.subplots(2, 2, figsize=(15, 12))
 fig.suptitle('Enhanced Feature Correlation Analysis', fontsize=16)
 
 # 1. Individual feature correlations
 ax1 = axes[0, 0]
 if 'correlation_analysis' in enhanced:
 correlations = enhanced['correlation_analysis']['individual_correlations']
 
 feature_names_short = [name[:15] for name in correlations.keys()]
 corr_values = list(correlations.values())
 
 colors = ['red' if abs(c) > 0.3 else 'blue' for c in corr_values]
 bars = ax1.barh(feature_names_short, corr_values, color=colors, alpha=0.7)
 ax1.set_xlabel('Correlation with Predictions')
 ax1.set_title('Individual Feature Correlations')
 ax1.axvline(0, color='black', linestyle='-', alpha=0.3)
 ax1.grid(True, alpha=0.3)
 
 # 2. Feature correlation matrix
 ax2 = axes[0, 1]
 corr_matrix = np.corrcoef(self.accessibility_features.T)
 im = ax2.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
 ax2.set_title('Feature Correlation Matrix')
 plt.colorbar(im, ax=ax2)
 
 # 3. Accessibility indices
 ax3 = axes[1, 0]
 if 'correlation_analysis' in enhanced:
 indices = enhanced['correlation_analysis']['accessibility_indices']
 
 index_names = []
 index_corrs = []
 index_colors = []
 
 for name, data in indices.items():
 index_names.append(name.replace('_', ' ').title()[:15])
 index_corrs.append(data['correlation'])
 # Color based on whether it's problematic
 index_colors.append('red' if data['problematic'] else 'green')
 
 if index_names:
 ax3.barh(index_names, index_corrs, color=index_colors, alpha=0.7)
 ax3.set_xlabel('Correlation with Vulnerability')
 ax3.set_title('Accessibility Index Correlations')
 ax3.axvline(0, color='black', linestyle='--', alpha=0.5)
 ax3.grid(True, alpha=0.3)
 
 # 4. Feature importance
 ax4 = axes[1, 1]
 if 'model_analysis' in enhanced:
 importance = enhanced['model_analysis']['feature_importance']['feature_importance_ranking']
 
 top_features = importance.head(10)
 ax4.barh(range(len(top_features)), top_features['importance'], alpha=0.7)
 ax4.set_yticks(range(len(top_features)))
 ax4.set_yticklabels([name[:15] for name in top_features['feature']], fontsize=8)
 ax4.set_xlabel('Feature Importance')
 ax4.set_title('Random Forest Feature Importance')
 ax4.grid(True, alpha=0.3)
 
 plt.tight_layout()
 plt.savefig(f"{output_dir}/enhanced_feature_correlations.png", dpi=300, bbox_inches='tight')
 plt.close()
 
 def _plot_spatial_patterns(self, output_dir):
 """Plot enhanced spatial pattern analysis"""
 
 fig, axes = plt.subplots(2, 2, figsize=(15, 12))
 fig.suptitle('Enhanced Spatial Pattern Analysis', fontsize=16)
 
 x_coords = self.coordinates[:, 0]
 y_coords = self.coordinates[:, 1]
 
 # 1. Spatial distribution
 ax1 = axes[0, 0]
 scatter1 = ax1.scatter(x_coords, y_coords, c=self.raw_predictions, 
 cmap='viridis', s=30, alpha=0.7)
 ax1.set_xlabel('Longitude')
 ax1.set_ylabel('Latitude')
 ax1.set_title('Raw Prediction Distribution')
 ax1.set_aspect('equal')
 plt.colorbar(scatter1, ax=ax1)
 
 # 2. Distance to center analysis
 ax2 = axes[0, 1]
 centroid = np.mean(self.coordinates, axis=0)
 distances = np.sqrt(np.sum((self.coordinates - centroid)**2, axis=1))
 
 ax2.scatter(distances, self.raw_predictions, alpha=0.6, s=20)
 z = np.polyfit(distances, self.raw_predictions, 1)
 p = np.poly1d(z)
 ax2.plot(sorted(distances), p(sorted(distances)), "r--", alpha=0.8)
 
 corr = np.corrcoef(distances, self.raw_predictions)[0, 1]
 ax2.set_xlabel('Distance to Center')
 ax2.set_ylabel('Predictions')
 ax2.set_title(f'Center-Distance Relationship (r={corr:.3f})')
 ax2.grid(True, alpha=0.3)
 
 # 3. Overall accessibility vs predictions
 ax3 = axes[1, 0]
 overall_access = np.mean(self.accessibility_features, axis=1)
 ax3.scatter(overall_access, self.raw_predictions, alpha=0.6, s=20, color='purple')
 
 z = np.polyfit(overall_access, self.raw_predictions, 1)
 p = np.poly1d(z)
 ax3.plot(sorted(overall_access), p(sorted(overall_access)), "r--", alpha=0.8)
 
 corr = np.corrcoef(overall_access, self.raw_predictions)[0, 1]
 ax3.set_xlabel('Overall Accessibility')
 ax3.set_ylabel('Predictions')
 ax3.set_title(f'Accessibility vs Predictions (r={corr:.3f})')
 ax3.grid(True, alpha=0.3)
 
 # Color title based on correctness
 if corr > 0:
 ax3.title.set_color('red')
 else:
 ax3.title.set_color('green')
 
 # 4. Spatial clustering
 ax4 = axes[1, 1]
 if hasattr(self, 'diagnostic_results') and 'enhanced_diagnostics' in self.diagnostic_results:
 enhanced = self.diagnostic_results['enhanced_diagnostics']
 if 'spatial_analysis' in enhanced and 'clustering_analysis' in enhanced['spatial_analysis']:
 clustering = enhanced['spatial_analysis']['clustering_analysis']
 
 if 'clusters' in clustering:
 n_clusters = len(clustering['clusters'])
 kmeans = KMeans(n_clusters=n_clusters, random_state=42)
 cluster_labels = kmeans.fit_predict(self.coordinates)
 
 scatter4 = ax4.scatter(x_coords, y_coords, c=cluster_labels, 
 cmap='tab10', s=30, alpha=0.7)
 ax4.set_xlabel('Longitude')
 ax4.set_ylabel('Latitude')
 ax4.set_title('Spatial Clusters')
 ax4.set_aspect('equal')
 
 plt.tight_layout()
 plt.savefig(f"{output_dir}/enhanced_spatial_patterns.png", dpi=300, bbox_inches='tight')
 plt.close()
 
 def _plot_accessibility_relationships(self, output_dir):
 """Plot detailed accessibility relationship analysis"""
 
 if 'enhanced_diagnostics' not in self.diagnostic_results:
 return
 
 enhanced = self.diagnostic_results['enhanced_diagnostics']
 
 if 'correlation_analysis' not in enhanced:
 return
 
 accessibility_indices = enhanced['correlation_analysis']['accessibility_indices']
 
 n_plots = len(accessibility_indices)
 if n_plots == 0:
 return
 
 # Create appropriate subplot grid
 n_rows = (n_plots + 1) // 2
 fig, axes = plt.subplots(n_rows, 2, figsize=(15, 6*n_rows))
 fig.suptitle('Detailed Accessibility-Vulnerability Relationships', fontsize=16)
 
 if n_rows == 1:
 axes = axes.reshape(1, -1)
 
 plot_idx = 0
 for name, data in accessibility_indices.items():
 if plot_idx >= n_plots:
 break
 
 row = plot_idx // 2
 col = plot_idx % 2
 ax = axes[row, col]
 
 values = data['values']
 correlation = data['correlation']
 expected_sign = data['expected_sign']
 problematic = data['problematic']
 
 # Scatter plot
 ax.scatter(values, self.raw_predictions, alpha=0.6, s=20, 
 color='red' if problematic else 'blue')
 
 # Regression line
 z = np.polyfit(values, self.raw_predictions, 1)
 p = np.poly1d(z)
 ax.plot(sorted(values), p(sorted(values)), 
 "r--" if problematic else "b--", alpha=0.8)
 
 ax.set_xlabel(name.replace('_', ' ').title())
 ax.set_ylabel('Predictions')
 ax.set_title(f'{name.replace("_", " ").title()}\nr={correlation:.3f} (Expected: {expected_sign})')
 ax.grid(True, alpha=0.3)
 
 # Color the title based on correctness
 if problematic:
 ax.title.set_color('red')
 else:
 ax.title.set_color('green')
 
 plot_idx += 1
 
 # Hide unused subplots
 for i in range(plot_idx, n_rows * 2):
 row = i // 2
 col = i % 2
 if row < n_rows:
 axes[row, col].set_visible(False)
 
 plt.tight_layout()
 plt.savefig(f"{output_dir}/enhanced_accessibility_relationships.png", dpi=300, bbox_inches='tight')
 plt.close()


# Convenience function for easy usage
def run_enhanced_spatial_diagnostics(raw_predictions, accessibility_features, coordinates, target_svi, 
 output_dir='./enhanced_diagnostics', create_plots=True):
 """
 Convenience function to run enhanced spatial diagnostics
 Maintains compatibility with original API while adding new features
 """
 
 diagnostics = SpatialLearningDiagnostics(verbose=True)
 
 # Run analysis (includes both original and enhanced diagnostics)
 results = diagnostics.comprehensive_evaluation(
 raw_predictions=raw_predictions,
 accessibility_features=accessibility_features,
 coordinates=coordinates,
 target_svi=target_svi
 )
 
 # Print enhanced diagnostic report
 verdict = diagnostics.print_diagnostic_report(results)
 
 # Create visualizations if requested
 if create_plots:
 diagnostics.create_enhanced_diagnostic_plots(output_dir)
 
 return results, diagnostics, verdict