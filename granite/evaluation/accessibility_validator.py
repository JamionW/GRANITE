"""
Accessibility Feature Validator for GRANITE

Validates accessibility feature computation including travel times,
destination counts, and theoretical consistency.
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cdist
from geopy.distance import geodesic
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class AccessibilityFeatureValidator:
 """
 validator for accessibility feature computation
 Identifies systematic issues in travel time calculation and feature engineering
 """
 
 def __init__(self, verbose=True):
 self.verbose = verbose
 self.validation_results = {}
 
 def log(self, message, level='INFO'):
 if self.verbose:
 print(f"[AccessValidator] {message}")
 
 def validate_accessibility_pipeline(self, addresses: gpd.GeoDataFrame, 
 accessibility_features: np.ndarray,
 destinations: Dict[str, gpd.GeoDataFrame],
 feature_names: List[str] = None,
 tract_svi: float = None) -> Dict:
 """
 Main validation function - analysis of accessibility computation
 
 Args:
 addresses: GeoDataFrame of address points
 accessibility_features: Computed accessibility feature matrix
 destinations: Dict of destination GeoDataFrames by type
 feature_names: List of feature names
 tract_svi: Target SVI for relationship validation
 
 Returns:
 validation results
 """
 
 self.log("Starting accessibility feature validation...")
 
 # Store data
 self.addresses = addresses
 self.accessibility_features = accessibility_features
 self.destinations = destinations
 self.feature_names = feature_names or self._generate_feature_names()
 self.tract_svi = tract_svi
 self.n_addresses = len(addresses)
 
 results = {}
 
 # 1. Travel Time Validation
 self.log("1. Validating travel time computations...")
 results['travel_time_validation'] = self._validate_travel_times()
 
 # 2. Destination Accessibility Validation 
 self.log("2. Validating destination accessibility logic...")
 results['destination_validation'] = self._validate_destination_accessibility()
 
 # 3. Feature Engineering Validation
 self.log("3. Validating feature engineering...")
 results['feature_engineering'] = self._validate_feature_engineering()
 
 # 4. Theoretical Consistency Validation
 self.log("4. Validating theoretical consistency...")
 results['theoretical_validation'] = self._validate_theoretical_consistency()
 
 # 5. Geographic Pattern Validation
 self.log("5. Validating geographic patterns...")
 results['geographic_validation'] = self._validate_geographic_patterns()
 
 # 6. Data Quality Assessment
 self.log("6. Assessing data quality...")
 results['data_quality'] = self._assess_data_quality()
 
 # 7. Root Cause Analysis
 self.log("7. Performing root cause analysis...")
 results['root_cause_analysis'] = self._perform_root_cause_analysis(results)
 
 # Store results
 self.validation_results = results
 
 # Print report
 self._print_validation_report(results)
 
 return results
 
 def _validate_travel_times(self) -> Dict:
 """Validate travel time computation accuracy and realism"""
 
 validation = {}
 
 # Extract travel time features
 time_features = [i for i, name in enumerate(self.feature_names) if 'time' in name]
 
 if not time_features:
 return {'error': 'No travel time features found'}
 
 time_values = self.accessibility_features[:, time_features]
 
 # 1. Realism checks
 validation['realism_checks'] = {
 'negative_times': int(np.sum(time_values < 0)),
 'zero_times': int(np.sum(time_values == 0)),
 'extremely_high_times': int(np.sum(time_values > 180)), # > 3 hours
 'unrealistic_times': int(np.sum(time_values > 300)), # > 5 hours
 'reasonable_range': int(np.sum((time_values >= 1) & (time_values <= 120))) # 1min - 2hrs
 }
 
 # 2. Statistical properties
 validation['statistical_properties'] = {
 'mean_times': np.mean(time_values, axis=0).tolist(),
 'std_times': np.std(time_values, axis=0).tolist(),
 'min_times': np.min(time_values, axis=0).tolist(),
 'max_times': np.max(time_values, axis=0).tolist(),
 'median_times': np.median(time_values, axis=0).tolist()
 }
 
 # 3. Cross-validation with straight-line distances
 validation['distance_validation'] = self._validate_times_vs_distances()
 
 # 4. Internal consistency checks
 validation['consistency_checks'] = self._check_travel_time_consistency(time_values)
 
 return validation
 
 def _validate_times_vs_distances(self) -> Dict:
 """Cross-validate travel times against straight-line distances"""
 
 validation = {}
 
 # Sample addresses for efficiency
 sample_size = min(100, self.n_addresses)
 sample_indices = np.random.choice(self.n_addresses, sample_size, replace=False)
 
 for dest_type, dest_gdf in self.destinations.items():
 
 dest_validation = {}
 time_feature_idx = None
 
 # Find corresponding time feature
 for i, name in enumerate(self.feature_names):
 if dest_type in name and 'min_time' in name:
 time_feature_idx = i
 break
 
 if time_feature_idx is None:
 continue
 
 # Calculate distance vs time relationships for sample
 distance_time_pairs = []
 
 for addr_idx in sample_indices:
 addr_point = self.addresses.iloc[addr_idx].geometry
 computed_time = self.accessibility_features[addr_idx, time_feature_idx]
 
 # Find minimum distance to destinations of this type
 min_distance_km = float('inf')
 for _, dest in dest_gdf.iterrows():
 dist_km = geodesic(
 (addr_point.y, addr_point.x), 
 (dest.geometry.y, dest.geometry.x)
 ).kilometers
 min_distance_km = min(min_distance_km, dist_km)
 
 if min_distance_km != float('inf'):
 distance_time_pairs.append((min_distance_km, computed_time))
 
 if distance_time_pairs:
 distances, times = zip(*distance_time_pairs)
 
 # Expected travel time ranges based on distance
 expected_walk_times = [d / 5.0 * 60 for d in distances] # 5 km/h
 expected_drive_times = [d / 30.0 * 60 for d in distances] # 30 km/h
 
 dest_validation = {
 'correlation': pearsonr(distances, times)[0],
 'mean_distance_km': np.mean(distances),
 'mean_computed_time': np.mean(times),
 'mean_expected_walk_time': np.mean(expected_walk_times),
 'mean_expected_drive_time': np.mean(expected_drive_times),
 'time_distance_ratio': np.mean(times) / np.mean(distances) if np.mean(distances) > 0 else 0,
 'reasonable_ratios': sum(1 for d, t in zip(distances, times) 
 if 0.5 <= (t / (d * 10)) <= 5.0) # Rough reasonableness check
 }
 
 validation[dest_type] = dest_validation
 
 return validation
 
 def _check_travel_time_consistency(self, time_values: np.ndarray) -> Dict:
 """Check internal consistency of travel time features"""
 
 consistency = {}
 
 # Group by destination type
 dest_types = ['employment', 'healthcare', 'grocery']
 
 for i, dest_type in enumerate(dest_types):
 start_idx = i * 10
 if start_idx + 2 < time_values.shape[1]:
 min_times = time_values[:, start_idx] # min_time
 mean_times = time_values[:, start_idx + 1] # mean_time
 p90_times = time_values[:, start_idx + 2] # 90th_time
 
 # Check ordering: min <= mean <= 90th
 valid_ordering = np.sum((min_times <= mean_times) & (mean_times <= p90_times))
 total_addresses = len(min_times)
 
 consistency[dest_type] = {
 'valid_ordering_count': int(valid_ordering),
 'total_addresses': int(total_addresses),
 'ordering_validity_rate': valid_ordering / total_addresses,
 'min_exceeds_mean': int(np.sum(min_times > mean_times)),
 'mean_exceeds_90th': int(np.sum(mean_times > p90_times)),
 'zero_variance': int(np.sum(min_times == p90_times))
 }
 
 return consistency
 
 def _validate_destination_accessibility(self) -> Dict:
 """Validate destination counting and accessibility score logic"""
 
 validation = {}
 
 # Extract count and score features
 count_features = [i for i, name in enumerate(self.feature_names) if 'count' in name]
 score_features = [i for i, name in enumerate(self.feature_names) if 'score' in name]
 
 if count_features:
 count_values = self.accessibility_features[:, count_features]
 
 validation['count_validation'] = {
 'negative_counts': int(np.sum(count_values < 0)),
 'zero_counts': int(np.sum(count_values == 0)),
 'extremely_high_counts': int(np.sum(count_values > 100)),
 'total_destinations_available': {dest_type: len(dest_gdf) 
 for dest_type, dest_gdf in self.destinations.items()},
 'counts_exceed_available': self._check_counts_vs_available_destinations(count_values)
 }
 
 if score_features:
 score_values = self.accessibility_features[:, score_features]
 
 validation['score_validation'] = {
 'negative_scores': int(np.sum(score_values < 0)),
 'zero_scores': int(np.sum(score_values == 0)),
 'infinite_scores': int(np.sum(np.isinf(score_values))),
 'nan_scores': int(np.sum(np.isnan(score_values))),
 'score_distribution': {
 'mean': np.mean(score_values),
 'std': np.std(score_values),
 'min': np.min(score_values),
 'max': np.max(score_values),
 'median': np.median(score_values)
 }
 }
 
 # Validate count progressions (5min <= 10min <= 15min)
 validation['count_progression'] = self._validate_count_progressions()
 
 return validation
 
 def _check_counts_vs_available_destinations(self, count_values: np.ndarray) -> Dict:
 """Check if destination counts exceed actually available destinations"""
 
 issues = {}
 
 dest_types = ['employment', 'healthcare', 'grocery']
 count_types = ['5min', '10min', '15min']
 
 for dest_idx, dest_type in enumerate(dest_types):
 available_destinations = len(self.destinations.get(dest_type, []))
 
 for count_idx, count_type in enumerate(count_types):
 feature_idx = dest_idx * 10 + 3 + count_idx # Count features start at offset 3
 
 if feature_idx < count_values.shape[1]:
 counts = count_values[:, feature_idx]
 exceeds_available = np.sum(counts > available_destinations)
 
 issues[f'{dest_type}_{count_type}'] = {
 'available_destinations': available_destinations,
 'max_computed_count': int(np.max(counts)),
 'addresses_exceeding_available': int(exceeds_available),
 'addresses_with_zero_access': int(np.sum(counts == 0))
 }
 
 return issues
 
 def _validate_count_progressions(self) -> Dict:
 """Validate that count progressions make sense"""
 
 progressions = {}
 
 dest_types = ['employment', 'healthcare', 'grocery']
 
 for dest_idx, dest_type in enumerate(dest_types):
 start_idx = dest_idx * 10 + 3 # Count features start at offset 3
 
 if start_idx + 2 < self.accessibility_features.shape[1]:
 count_5min = self.accessibility_features[:, start_idx]
 count_10min = self.accessibility_features[:, start_idx + 1]
 count_15min = self.accessibility_features[:, start_idx + 2]
 
 # Check progression validity
 valid_5_to_10 = np.sum(count_5min <= count_10min)
 valid_10_to_15 = np.sum(count_10min <= count_15min)
 valid_full = np.sum((count_5min <= count_10min) & (count_10min <= count_15min))
 
 progressions[dest_type] = {
 'valid_5_to_10': int(valid_5_to_10),
 'valid_10_to_15': int(valid_10_to_15),
 'valid_full_progression': int(valid_full),
 'total_addresses': self.n_addresses,
 'progression_validity_rate': valid_full / self.n_addresses,
 'violations_5_to_10': int(np.sum(count_5min > count_10min)),
 'violations_10_to_15': int(np.sum(count_10min > count_15min))
 }
 
 return progressions
 
 def _validate_feature_engineering(self) -> Dict:
 """Validate feature engineering logic and transformations"""
 
 validation = {}
 
 # 1. Feature scaling and normalization
 validation['scaling_analysis'] = self._analyze_feature_scaling()
 
 # 2. Feature correlation analysis
 validation['correlation_analysis'] = self._analyze_feature_correlations()
 
 # 3. Derived feature validation
 validation['derived_features'] = self._validate_derived_features()
 
 # 4. Missing value analysis
 validation['missing_values'] = self._analyze_missing_values()
 
 return validation
 
 def _analyze_feature_scaling(self) -> Dict:
 """Analyze feature scaling and detect issues"""
 
 scaling_analysis = {}
 
 # Calculate feature statistics
 feature_stats = pd.DataFrame({
 'feature': self.feature_names,
 'mean': np.mean(self.accessibility_features, axis=0),
 'std': np.std(self.accessibility_features, axis=0),
 'min': np.min(self.accessibility_features, axis=0),
 'max': np.max(self.accessibility_features, axis=0),
 'range': np.ptp(self.accessibility_features, axis=0),
 'skewness': [self._calculate_skewness(self.accessibility_features[:, i]) 
 for i in range(self.accessibility_features.shape[1])]
 })
 
 scaling_analysis['feature_statistics'] = feature_stats
 
 # Identify scaling issues
 scaling_analysis['scaling_issues'] = {
 'zero_variance': list(feature_stats[feature_stats['std'] < 1e-8]['feature']),
 'extreme_skewness': list(feature_stats[abs(feature_stats['skewness']) > 5]['feature']),
 'extreme_ranges': list(feature_stats[feature_stats['range'] > 1000]['feature']),
 'scale_mismatches': self._identify_scale_mismatches(feature_stats)
 }
 
 return scaling_analysis
 
 def _identify_scale_mismatches(self, feature_stats: pd.DataFrame) -> List[str]:
 """Identify features with very different scales that might cause issues"""
 
 # Group features by type
 time_features = feature_stats[feature_stats['feature'].str.contains('time')]
 count_features = feature_stats[feature_stats['feature'].str.contains('count')]
 score_features = feature_stats[feature_stats['feature'].str.contains('score')]
 
 mismatches = []
 
 # Check within-group scale consistency
 for group_name, group in [('time', time_features), ('count', count_features), ('score', score_features)]:
 if len(group) > 1:
 ranges = group['range'].values
 max_range = np.max(ranges)
 min_range = np.min(ranges[ranges > 0]) # Exclude zero ranges
 
 if min_range > 0 and max_range / min_range > 100: # 100x difference
 mismatches.extend(group['feature'].tolist())
 
 return mismatches
 
 def _analyze_feature_correlations(self) -> Dict:
 """Analyze correlations between accessibility features"""
 
 correlation_matrix = np.corrcoef(self.accessibility_features.T)
 
 # Find highly correlated pairs
 high_correlations = []
 for i in range(len(self.feature_names)):
 for j in range(i + 1, len(self.feature_names)):
 corr = correlation_matrix[i, j]
 if abs(corr) > 0.95: # Very high correlation
 high_correlations.append({
 'feature1': self.feature_names[i],
 'feature2': self.feature_names[j],
 'correlation': corr
 })
 
 # Analyze within-destination correlations
 dest_correlations = {}
 dest_types = ['employment', 'healthcare', 'grocery']
 
 for dest_idx, dest_type in enumerate(dest_types):
 start_idx = dest_idx * 10
 end_idx = start_idx + 10
 
 if end_idx <= len(self.feature_names):
 dest_features = self.accessibility_features[:, start_idx:end_idx]
 dest_corr_matrix = np.corrcoef(dest_features.T)
 
 dest_correlations[dest_type] = {
 'correlation_matrix': dest_corr_matrix,
 'feature_names': self.feature_names[start_idx:end_idx]
 }
 
 return {
 'overall_correlation_matrix': correlation_matrix,
 'high_correlations': high_correlations,
 'destination_correlations': dest_correlations
 }
 
 def _validate_derived_features(self) -> Dict:
 """Validate derived/summary features if they exist"""
 
 # Derived features typically start after the 24 base features
 if len(self.feature_names) > 24:
 derived_start = 24
 derived_features = self.accessibility_features[:, derived_start:]
 derived_names = self.feature_names[derived_start:]
 
 validation = {
 'derived_feature_count': len(derived_names),
 'derived_feature_names': derived_names,
 'derived_statistics': {
 'means': np.mean(derived_features, axis=0).tolist(),
 'stds': np.std(derived_features, axis=0).tolist(),
 'ranges': np.ptp(derived_features, axis=0).tolist()
 }
 }
 
 # Check if derived features are computed correctly
 if 'total_accessibility' in derived_names:
 total_idx = derived_names.index('total_accessibility')
 computed_total = derived_features[:, total_idx]
 
 # Approximate total accessibility from base features
 score_features = [i for i, name in enumerate(self.feature_names[:24]) if 'score' in name]
 if score_features:
 expected_total = np.sum(self.accessibility_features[:, score_features], axis=1)
 correlation = pearsonr(computed_total, expected_total)[0]
 validation['total_accessibility_validation'] = {
 'correlation_with_expected': correlation,
 'computation_appears_correct': correlation > 0.9
 }
 
 return validation
 
 return {'derived_feature_count': 0}
 
 def _analyze_missing_values(self) -> Dict:
 """Analyze missing values and data completeness"""
 
 analysis = {
 'nan_values': int(np.sum(np.isnan(self.accessibility_features))),
 'infinite_values': int(np.sum(np.isinf(self.accessibility_features))),
 'zero_values': int(np.sum(self.accessibility_features == 0)),
 'total_values': int(self.accessibility_features.size)
 }
 
 # Per-feature missing value analysis
 analysis['per_feature_missing'] = {}
 for i, name in enumerate(self.feature_names):
 feature_values = self.accessibility_features[:, i]
 analysis['per_feature_missing'][name] = {
 'nan_count': int(np.sum(np.isnan(feature_values))),
 'inf_count': int(np.sum(np.isinf(feature_values))),
 'zero_count': int(np.sum(feature_values == 0)),
 'missing_rate': (np.sum(np.isnan(feature_values)) + np.sum(np.isinf(feature_values))) / len(feature_values)
 }
 
 return analysis
 
 def _validate_theoretical_consistency(self) -> Dict:
 """Validate against transportation equity theory"""
 
 validation = {}
 
 # 1. Expected correlation directions
 validation['expected_directions'] = self._check_expected_correlation_directions()
 
 # 2. Urban accessibility patterns
 validation['urban_patterns'] = self._validate_urban_accessibility_patterns()
 
 # 3. Cross-destination consistency
 validation['cross_destination_consistency'] = self._check_cross_destination_consistency()
 
 return validation
 
 def _check_expected_correlation_directions(self) -> Dict:
 """Check if features correlate in expected directions with each other"""
 
 directions = {}
 
 # Time vs Count relationships (should be negative)
 time_count_correlations = []
 
 dest_types = ['employment', 'healthcare', 'grocery']
 for dest_idx, dest_type in enumerate(dest_types):
 start_idx = dest_idx * 10
 
 if start_idx + 7 < len(self.feature_names):
 min_time = self.accessibility_features[:, start_idx]
 count_10min = self.accessibility_features[:, start_idx + 4] # 10min count
 
 if np.std(min_time) > 0 and np.std(count_5) > 0:
 corr = pearsonr(min_time, count_5)[0]
 time_count_correlations.append({
 'destination': dest_type,
 'correlation': corr,
 'expected_negative': corr < 0,
 'strength': 'strong' if abs(corr) > 0.5 else 'moderate' if abs(corr) > 0.3 else 'weak'
 })
 
 directions['time_count_correlations'] = time_count_correlations
 
 # Score vs Count relationships (should be positive)
 score_count_correlations = []
 
 for dest_idx, dest_type in enumerate(dest_types):
 start_idx = dest_idx * 10
 
 if start_idx + 7 < len(self.feature_names):
 score = self.accessibility_features[:, start_idx + 7] # accessibility_score
 count_10min = self.accessibility_features[:, start_idx + 4] # 60min count
 
 if np.std(score) > 0 and np.std(count_5) > 0:
 corr = pearsonr(score, count_5)[0]
 score_count_correlations.append({
 'destination': dest_type,
 'correlation': corr,
 'expected_positive': corr > 0,
 'strength': 'strong' if abs(corr) > 0.5 else 'moderate' if abs(corr) > 0.3 else 'weak'
 })
 
 directions['score_count_correlations'] = score_count_correlations
 
 return directions
 
 def _validate_urban_accessibility_patterns(self) -> Dict:
 """Validate expected urban accessibility patterns"""
 
 patterns = {}
 
 # Calculate distance to centroid
 coords = np.array([[addr.geometry.x, addr.geometry.y] for _, addr in self.addresses.iterrows()])
 centroid = np.mean(coords, axis=0)
 distances_to_center = np.sqrt(np.sum((coords - centroid)**2, axis=1))
 
 # Test center-periphery accessibility patterns
 center_periphery_analysis = {}
 
 # Overall accessibility vs distance to center
 overall_accessibility = np.mean(self.accessibility_features, axis=1)
 center_corr = pearsonr(distances_to_center, overall_accessibility)[0]
 
 center_periphery_analysis['overall_accessibility_vs_distance'] = {
 'correlation': center_corr,
 'expected_negative': center_corr < 0, # Center should have better access
 'pattern_strength': 'strong' if abs(center_corr) > 0.5 else 'moderate' if abs(center_corr) > 0.3 else 'weak'
 }
 
 # Test by destination type
 dest_types = ['employment', 'healthcare', 'grocery']
 for dest_idx, dest_type in enumerate(dest_types):
 start_idx = dest_idx * 10
 
 if start_idx + 7 < len(self.feature_names):
 # Use accessibility score as proxy for overall access to this destination type
 dest_accessibility = self.accessibility_features[:, start_idx + 9]
 dest_corr = pearsonr(distances_to_center, dest_accessibility)[0]
 
 center_periphery_analysis[f'{dest_type}_vs_distance'] = {
 'correlation': dest_corr,
 'expected_negative': dest_corr < 0,
 'pattern_strength': 'strong' if abs(dest_corr) > 0.5 else 'moderate' if abs(dest_corr) > 0.3 else 'weak'
 }
 
 patterns['center_periphery_analysis'] = center_periphery_analysis
 
 return patterns
 
 def _check_cross_destination_consistency(self) -> Dict:
 """Check consistency patterns across destination types"""
 
 consistency = {}
 
 dest_types = ['employment', 'healthcare', 'grocery']
 
 # Compare accessibility scores across destination types
 if len(dest_types) >= 2:
 accessibility_scores = []
 for dest_idx, dest_type in enumerate(dest_types):
 start_idx = dest_idx * 10
 if start_idx + 7 < len(self.feature_names):
 score = self.accessibility_features[:, start_idx + 9]
 accessibility_scores.append(score)
 
 if len(accessibility_scores) >= 2:
 # Calculate cross-correlations
 cross_correlations = {}
 for i in range(len(accessibility_scores)):
 for j in range(i + 1, len(accessibility_scores)):
 corr = pearsonr(accessibility_scores[i], accessibility_scores[j])[0]
 cross_correlations[f'{dest_types[i]}_vs_{dest_types[j]}'] = {
 'correlation': corr,
 'expected_positive': corr > 0, # Generally expect positive correlation
 'strength': 'strong' if abs(corr) > 0.5 else 'moderate' if abs(corr) > 0.3 else 'weak'
 }
 
 consistency['cross_destination_correlations'] = cross_correlations
 
 return consistency
 
 def _validate_geographic_patterns(self) -> Dict:
 """Validate geographic patterns and spatial relationships"""
 
 validation = {}
 
 # 1. Spatial autocorrelation of accessibility features
 validation['spatial_autocorrelation'] = self._analyze_accessibility_spatial_autocorr()
 
 # 2. Distance decay patterns
 validation['distance_decay'] = self._analyze_distance_decay_patterns()
 
 # 3. Destination proximity analysis
 validation['proximity_analysis'] = self._analyze_destination_proximity_effects()
 
 return validation
 
 def _analyze_accessibility_spatial_autocorr(self) -> Dict:
 """Analyze spatial autocorrelation of accessibility features"""
 
 from sklearn.neighbors import NearestNeighbors
 
 coords = np.array([[addr.geometry.x, addr.geometry.y] for _, addr in self.addresses.iterrows()])
 
 # Build spatial weights
 k = min(8, len(coords) - 1)
 nbrs = NearestNeighbors(n_neighbors=k+1).fit(coords)
 distances, indices = nbrs.kneighbors(coords)
 
 # Calculate Moran's I for key accessibility features
 autocorr_results = {}
 
 # Test overall accessibility
 overall_accessibility = np.mean(self.accessibility_features, axis=1)
 overall_moran = self._calculate_morans_i(overall_accessibility, coords, distances, indices)
 autocorr_results['overall_accessibility'] = overall_moran
 
 # Test individual feature types
 feature_types = ['time', 'count', 'score']
 for feature_type in feature_types:
 type_features = [i for i, name in enumerate(self.feature_names) if feature_type in name]
 if type_features:
 type_accessibility = np.mean(self.accessibility_features[:, type_features], axis=1)
 type_moran = self._calculate_morans_i(type_accessibility, coords, distances, indices)
 autocorr_results[f'{feature_type}_features'] = type_moran
 
 return autocorr_results
 
 def _calculate_morans_i(self, values, coords, distances, indices):
 """Calculate Moran's I for spatial autocorrelation"""
 
 n = len(values)
 
 # Create weights matrix
 W = np.zeros((n, n))
 for i in range(n):
 for j_idx in range(1, len(indices[i])): # Skip self
 j = indices[i][j_idx]
 dist = distances[i][j_idx]
 if dist > 0:
 W[i, j] = 1.0 / dist
 
 # Row normalize
 row_sums = W.sum(axis=1)
 W = np.divide(W, row_sums[:, np.newaxis], out=np.zeros_like(W), where=row_sums[:, np.newaxis]!=0)
 
 # Compute Moran's I
 values_centered = values - np.mean(values)
 numerator = np.sum(W * np.outer(values_centered, values_centered))
 denominator = np.sum(values_centered**2)
 
 if denominator == 0:
 return 0.0
 
 moran_i = (n / np.sum(W)) * (numerator / denominator) if np.sum(W) > 0 else 0.0
 return moran_i
 
 def _analyze_distance_decay_patterns(self) -> Dict:
 """Analyze distance decay in accessibility features"""
 
 decay_analysis = {}
 
 coords = np.array([[addr.geometry.x, addr.geometry.y] for _, addr in self.addresses.iterrows()])
 
 # For each destination type, analyze distance decay
 for dest_type, dest_gdf in self.destinations.items():
 
 # Find corresponding accessibility score feature
 score_feature_idx = None
 for i, name in enumerate(self.feature_names):
 if dest_type in name and 'score' in name:
 score_feature_idx = i
 break
 
 if score_feature_idx is None:
 continue
 
 # Calculate distances to nearest destination of this type
 min_distances = []
 accessibility_scores = self.accessibility_features[:, score_feature_idx]
 
 for _, addr in self.addresses.iterrows():
 addr_point = (addr.geometry.y, addr.geometry.x)
 
 min_dist = float('inf')
 for _, dest in dest_gdf.iterrows():
 dest_point = (dest.geometry.y, dest.geometry.x)
 dist = geodesic(addr_point, dest_point).kilometers
 min_dist = min(min_dist, dist)
 
 min_distances.append(min_dist)
 
 # Analyze distance-accessibility relationship
 if len(min_distances) > 0:
 correlation = pearsonr(min_distances, accessibility_scores)[0]
 
 decay_analysis[dest_type] = {
 'distance_accessibility_correlation': correlation,
 'expected_negative': correlation < 0, # Further = less accessible
 'mean_distance_km': np.mean(min_distances),
 'distance_range': [np.min(min_distances), np.max(min_distances)],
 'decay_strength': 'strong' if correlation < -0.5 else 'moderate' if correlation < -0.3 else 'weak'
 }
 
 return decay_analysis
 
 def _analyze_destination_proximity_effects(self) -> Dict:
 """Analyze how destination proximity affects accessibility calculations"""
 
 proximity_analysis = {}
 
 for dest_type, dest_gdf in self.destinations.items():
 
 # Find features for this destination type
 dest_features = [i for i, name in enumerate(self.feature_names) if dest_type in name]
 
 if not dest_features or len(dest_gdf) == 0:
 continue
 
 # Calculate destination density (destinations per unit area)
 dest_coords = np.array([[dest.geometry.x, dest.geometry.y] for _, dest in dest_gdf.iterrows()])
 
 if len(dest_coords) > 1:
 # Measure of destination clustering
 pairwise_distances = cdist(dest_coords, dest_coords)
 mean_inter_dest_distance = np.mean(pairwise_distances[pairwise_distances > 0])
 
 proximity_analysis[dest_type] = {
 'destination_count': len(dest_gdf),
 'mean_inter_destination_distance': mean_inter_dest_distance,
 'destination_density': len(dest_gdf) / (mean_inter_dest_distance ** 2) if mean_inter_dest_distance > 0 else 0,
 'spatial_distribution': 'clustered' if mean_inter_dest_distance < 0.01 else 'dispersed'
 }
 
 return proximity_analysis
 
 def _assess_data_quality(self) -> Dict:
 """Comprehensive data quality assessment"""
 
 quality = {}
 
 # 1. Overall quality score
 quality_indicators = []
 
 # Check for unrealistic values
 unrealistic_count = np.sum((self.accessibility_features < 0) | 
 (self.accessibility_features > 1000) |
 np.isnan(self.accessibility_features) |
 np.isinf(self.accessibility_features))
 total_values = self.accessibility_features.size
 realistic_rate = 1 - (unrealistic_count / total_values)
 quality_indicators.append(realistic_rate)
 
 # Check feature completeness
 zero_variance_features = np.sum(np.std(self.accessibility_features, axis=0) < 1e-8)
 completeness_rate = 1 - (zero_variance_features / len(self.feature_names))
 quality_indicators.append(completeness_rate)
 
 # Check theoretical consistency
 time_features = [i for i, name in enumerate(self.feature_names) if 'time' in name]
 count_features = [i for i, name in enumerate(self.feature_names) if 'count' in name]
 
 if time_features and count_features:
 avg_time = np.mean(self.accessibility_features[:, time_features], axis=1)
 avg_count = np.mean(self.accessibility_features[:, count_features], axis=1)
 expected_correlation = pearsonr(avg_time, avg_count)[0]
 consistency_score = max(0, -expected_correlation) # Should be negative
 quality_indicators.append(consistency_score)
 
 overall_quality = np.mean(quality_indicators)
 
 quality['overall_assessment'] = {
 'quality_score': overall_quality,
 'quality_grade': self._get_quality_grade(overall_quality),
 'realistic_values_rate': realistic_rate,
 'feature_completeness_rate': completeness_rate,
 'theoretical_consistency_score': consistency_score if 'consistency_score' in locals() else 0
 }
 
 # 2. Specific quality issues
 quality['specific_issues'] = self._identify_specific_quality_issues()
 
 return quality
 
 def _identify_specific_quality_issues(self) -> List[str]:
 """Identify specific data quality issues"""
 
 issues = []
 
 # Check for extreme values
 if np.any(self.accessibility_features > 500): # > 8+ hours travel time
 issues.append("Extremely high travel times detected (>8 hours)")
 
 if np.any(self.accessibility_features < 0):
 issues.append("Negative accessibility values detected")
 
 # Check for no-variation features
 zero_var_count = np.sum(np.std(self.accessibility_features, axis=0) < 1e-8)
 if zero_var_count > 0:
 issues.append(f"{zero_var_count} features have zero variation")
 
 # Check count vs destination availability
 for dest_type, dest_gdf in self.destinations.items():
 available_destinations = len(dest_gdf)
 count_features = [i for i, name in enumerate(self.feature_names) 
 if dest_type in name and 'count' in name]
 
 if count_features:
 max_counts = np.max(self.accessibility_features[:, count_features], axis=0)
 if np.any(max_counts > available_destinations):
 issues.append(f"{dest_type} counts exceed available destinations")
 
 # Check for perfect correlations (suspicious)
 corr_matrix = np.corrcoef(self.accessibility_features.T)
 perfect_corr_pairs = np.sum((np.abs(corr_matrix) > 0.999) & (corr_matrix != 1.0))
 if perfect_corr_pairs > 0:
 issues.append(f"{perfect_corr_pairs} pairs of features are perfectly correlated")
 
 return issues
 
 def _perform_root_cause_analysis(self, validation_results: Dict) -> Dict:
 """Comprehensive root cause analysis"""
 
 root_causes = []
 confidence_scores = {}
 
 # Analyze travel time issues
 if 'travel_time_validation' in validation_results:
 time_val = validation_results['travel_time_validation']
 realism = time_val.get('realism_checks', {})
 
 if realism.get('unrealistic_times', 0) > 0:
 root_causes.append("Unrealistic travel time computations")
 confidence_scores['travel_time_computation'] = 0.9
 
 if 'distance_validation' in time_val:
 dist_val = time_val['distance_validation']
 poor_correlations = sum(1 for dest_data in dist_val.values() 
 if isinstance(dest_data, dict) and 
 dest_data.get('correlation', 0) < 0.3)
 if poor_correlations > 0:
 root_causes.append("Travel times don't correlate with distances")
 confidence_scores['distance_time_mismatch'] = 0.8
 
 # Analyze destination accessibility issues
 if 'destination_validation' in validation_results:
 dest_val = validation_results['destination_validation']
 
 if 'count_validation' in dest_val:
 count_val = dest_val['count_validation']
 if count_val.get('counts_exceed_available', {}):
 exceed_issues = sum(1 for dest_data in count_val['counts_exceed_available'].values()
 if dest_data.get('addresses_exceeding_available', 0) > 0)
 if exceed_issues > 0:
 root_causes.append("Destination counts exceed available destinations")
 confidence_scores['destination_counting_error'] = 0.9
 
 # Analyze theoretical consistency issues
 if 'theoretical_validation' in validation_results:
 theory_val = validation_results['theoretical_validation']
 
 if 'expected_directions' in theory_val:
 directions = theory_val['expected_directions']
 
 # Check time-count correlations
 if 'time_count_correlations' in directions:
 wrong_directions = sum(1 for corr_data in directions['time_count_correlations']
 if not corr_data.get('expected_negative', True))
 if wrong_directions > 0:
 root_causes.append("Time-count correlations have wrong direction")
 confidence_scores['wrong_correlation_direction'] = 0.8
 
 # Analyze data quality issues
 if 'data_quality' in validation_results:
 quality = validation_results['data_quality']
 overall_score = quality['overall_assessment']['quality_score']
 
 if overall_score < 0.7:
 root_causes.append("Overall data quality is poor")
 confidence_scores['poor_data_quality'] = 1 - overall_score
 
 # Primary diagnosis
 primary_cause = "Unknown"
 max_confidence = 0
 
 for cause, confidence in confidence_scores.items():
 if confidence > max_confidence:
 max_confidence = confidence
 primary_cause = self._map_cause_to_diagnosis(cause)
 
 return {
 'identified_causes': root_causes,
 'confidence_scores': confidence_scores,
 'primary_diagnosis': primary_cause,
 'recommended_actions': self._recommend_actions(confidence_scores),
 'priority_fixes': self._prioritize_fixes(confidence_scores)
 }
 
 def _map_cause_to_diagnosis(self, cause: str) -> str:
 """Map detected cause to diagnosis"""
 
 mapping = {
 'travel_time_computation': 'Travel time calculation errors',
 'distance_time_mismatch': 'Distance-time relationship broken',
 'destination_counting_error': 'Destination accessibility counting logic error',
 'wrong_correlation_direction': 'Feature engineering produces wrong relationships',
 'poor_data_quality': 'Systematic data quality issues'
 }
 
 return mapping.get(cause, cause)
 
 def _recommend_actions(self, confidence_scores: Dict) -> List[str]:
 """Recommend specific actions based on detected issues"""
 
 actions = []
 
 if confidence_scores.get('travel_time_computation', 0) > 0.7:
 actions.extend([
 "Validate travel time calculation methodology",
 "Check units and scaling in travel time computation",
 "Verify distance-to-time conversion factors",
 "Test travel time calculations on known routes"
 ])
 
 if confidence_scores.get('distance_time_mismatch', 0) > 0.7:
 actions.extend([
 "Debug distance calculation (geodesic vs projected)",
 "Verify speed assumptions for different modes",
 "Check for unit conversion errors (degrees vs km vs miles)",
 "Validate geographic coordinate systems"
 ])
 
 if confidence_scores.get('destination_counting_error', 0) > 0.7:
 actions.extend([
 "Review destination counting logic",
 "Verify time threshold definitions (30/60/90 minutes)",
 "Check for double-counting of destinations",
 "Validate destination filtering criteria"
 ])
 
 if confidence_scores.get('wrong_correlation_direction', 0) > 0.7:
 actions.extend([
 "Review feature engineering transformations",
 "Check for inverted or incorrectly scaled features",
 "Validate accessibility score calculations",
 "Test individual feature relationships manually"
 ])
 
 return actions
 
 def _prioritize_fixes(self, confidence_scores: Dict) -> List[Dict]:
 """Prioritize fixes by impact and confidence"""
 
 fixes = []
 
 # Sort by confidence score
 sorted_issues = sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True)
 
 for issue, confidence in sorted_issues:
 if confidence > 0.6:
 priority = 'HIGH' if confidence > 0.8 else 'MEDIUM'
 fixes.append({
 'issue': self._map_cause_to_diagnosis(issue),
 'confidence': confidence,
 'priority': priority,
 'estimated_impact': 'HIGH' if issue in ['travel_time_computation', 'destination_counting_error'] else 'MEDIUM'
 })
 
 return fixes
 
 # ===== HELPER METHODS =====
 
 def _generate_feature_names(self) -> List[str]:
 """Generate feature names based on GRANITE structure"""

 #Index 0: min_time
 #Index 1: mean_time
 #Index 2: median_time
 #Index 3: count_5min
 #Index 4: count_10min
 #Index 5: count_15min
 #Index 6: drive_advantage
 #Index 7: dispersion
 #Index 8: time_range
 #Index 9: accessibility_percentile
 
 base_features = []
 for dest_type in ['employment', 'healthcare', 'grocery']:
 base_features.extend([
 f'{dest_type}_min_time', 
 f'{dest_type}_mean_time', 
 f'{dest_type}_median_time', # <- Changed from 90th_time
 f'{dest_type}_count_5min', 
 f'{dest_type}_count_10min', 
 f'{dest_type}_count_15min',
 f'{dest_type}_drive_advantage', # <- NEW
 f'{dest_type}_dispersion', # <- NEW (not concentration)
 f'{dest_type}_time_range', # <- NEW
 f'{dest_type}_percentile' # <- NEW (not accessibility_score)
 ])
 
 derived_features = ['total_accessibility', 'accessibility_diversity', 
 'avg_transit_dependence', 'time_efficiency']
 
 all_features = base_features + derived_features
 return all_features[:self.accessibility_features.shape[1]]
 
 def _calculate_skewness(self, values):
 """Calculate skewness of a feature"""
 
 mean = np.mean(values)
 std = np.std(values)
 
 if std == 0:
 return 0
 
 skewness = np.mean(((values - mean) / std) ** 3)
 return skewness
 
 def _get_quality_grade(self, score: float) -> str:
 """Convert quality score to grade"""
 
 if score > 0.9:
 return 'A'
 elif score > 0.8:
 return 'B'
 elif score > 0.7:
 return 'C'
 elif score > 0.6:
 return 'D'
 else:
 return 'F'
 
 # ===== REPORTING METHODS =====
 
 def _print_validation_report(self, results: Dict):
 """Print validation report"""
 
 print("\n" + "="*80)
 print("ACCESSIBILITY FEATURE VALIDATION REPORT")
 print("="*80)
 
 # Data Overview
 print(f"\nDATA OVERVIEW")
 print(f" Addresses: {self.n_addresses:,}")
 print(f" Features: {len(self.feature_names)}")
 print(f" Destinations: {sum(len(dest_gdf) for dest_gdf in self.destinations.values())}")
 print(f" Target SVI: {self.tract_svi:.4f}" if self.tract_svi else " Target SVI: Not provided")
 
 # Travel Time Validation
 if 'travel_time_validation' in results:
 time_val = results['travel_time_validation']
 print(f"\n1. TRAVEL TIME VALIDATION")
 
 if 'realism_checks' in time_val:
 realism = time_val['realism_checks']
 print(f" Negative times: {realism.get('negative_times', 0)}")
 print(f" Zero times: {realism.get('zero_times', 0)}")
 print(f" Unrealistic times (>5hrs): {realism.get('unrealistic_times', 0)}")
 print(f" Reasonable times (1min-2hrs): {realism.get('reasonable_range', 0)}")
 
 if 'distance_validation' in time_val:
 print(f" Distance-time correlations:")
 for dest_type, dist_data in time_val['distance_validation'].items():
 if isinstance(dist_data, dict):
 corr = dist_data.get('correlation', 0)
 status = '' if corr > 0.3 else ''
 print(f" {dest_type}: r={corr:.3f} {status}")
 
 # Destination Validation
 if 'destination_validation' in results:
 dest_val = results['destination_validation']
 print(f"\n2. DESTINATION ACCESSIBILITY VALIDATION")
 
 if 'count_validation' in dest_val:
 count_val = dest_val['count_validation']
 print(f" Negative counts: {count_val.get('negative_counts', 0)}")
 print(f" Zero counts: {count_val.get('zero_counts', 0)}")
 print(f" Extremely high counts: {count_val.get('extremely_high_counts', 0)}")
 
 if 'count_progression' in dest_val:
 print(f" Count progression validity:")
 for dest_type, prog_data in dest_val['count_progression'].items():
 rate = prog_data.get('progression_validity_rate', 0)
 status = '' if rate > 0.9 else '' if rate > 0.7 else ''
 print(f" {dest_type}: {rate:.1%} {status}")
 
 # Theoretical Validation
 if 'theoretical_validation' in results:
 theory_val = results['theoretical_validation']
 print(f"\n3. THEORETICAL CONSISTENCY VALIDATION")
 
 if 'expected_directions' in theory_val:
 directions = theory_val['expected_directions']
 
 if 'time_count_correlations' in directions:
 print(f" Time-Count correlations (should be negative):")
 for corr_data in directions['time_count_correlations']:
 dest = corr_data['destination']
 corr = corr_data['correlation']
 expected = corr_data['expected_negative']
 status = '' if expected else ''
 print(f" {dest}: r={corr:.3f} {status}")
 
 # Data Quality
 if 'data_quality' in results:
 quality = results['data_quality']
 print(f"\n4. DATA QUALITY ASSESSMENT")
 
 if 'overall_assessment' in quality:
 overall = quality['overall_assessment']
 grade = overall['quality_grade']
 score = overall['quality_score']
 print(f" Overall Quality: {grade} ({score:.1%})")
 print(f" Realistic Values: {overall['realistic_values_rate']:.1%}")
 print(f" Feature Completeness: {overall['feature_completeness_rate']:.1%}")
 
 if 'specific_issues' in quality:
 issues = quality['specific_issues']
 if issues:
 print(f" Specific Issues:")
 for issue in issues:
 print(f" • {issue}")
 
 # Root Cause Analysis
 if 'root_cause_analysis' in results:
 root_cause = results['root_cause_analysis']
 print(f"\n5. ROOT CAUSE ANALYSIS")
 print(f" Primary Diagnosis: {root_cause['primary_diagnosis']}")
 
 if 'priority_fixes' in root_cause:
 print(f" Priority Fixes:")
 for fix in root_cause['priority_fixes'][:3]: # Top 3
 priority = fix['priority']
 issue = fix['issue']
 confidence = fix['confidence']
 print(f" {priority}: {issue} (confidence: {confidence:.1%})")
 
 if 'recommended_actions' in root_cause:
 print(f"\n6. RECOMMENDED ACTIONS")
 for action in root_cause['recommended_actions'][:5]: # Top 5
 print(f" -> {action}")
 
 print("\n" + "="*80)
 
 # ===== VISUALIZATION METHODS =====
 
 def create_validation_visualizations(self, output_dir: str = './accessibility_validation'):
 """Create validation visualizations"""
 
 import os
 os.makedirs(output_dir, exist_ok=True)
 
 if not self.validation_results:
 self.log("No validation results available. Run validate_accessibility_pipeline first.")
 return
 
 # 1. Travel time analysis plots
 self._plot_travel_time_analysis(output_dir)
 
 # 2. Feature relationship plots
 self._plot_feature_relationships(output_dir)
 
 # 3. Geographic pattern plots
 self._plot_geographic_patterns(output_dir)
 
 # 4. Data quality overview
 self._plot_data_quality_overview(output_dir)
 
 self.log(f"Validation visualizations saved to {output_dir}")
 
 def _plot_travel_time_analysis(self, output_dir: str):
 """Plot travel time validation analysis"""
 
 fig, axes = plt.subplots(2, 2, figsize=(15, 12))
 fig.suptitle('Travel Time Validation Analysis', fontsize=16)
 
 # Extract time features
 time_features = [i for i, name in enumerate(self.feature_names) if 'time' in name]
 time_values = self.accessibility_features[:, time_features]
 time_names = [self.feature_names[i] for i in time_features]
 
 # 1. Travel time distributions
 ax1 = axes[0, 0]
 for i, name in enumerate(time_names[:3]): # First 3 for visibility
 ax1.hist(time_values[:, i], bins=30, alpha=0.7, label=name.split('_')[0], density=True)
 ax1.set_xlabel('Travel Time (minutes)')
 ax1.set_ylabel('Density')
 ax1.set_title('Travel Time Distributions')
 ax1.legend()
 ax1.grid(True, alpha=0.3)
 
 # 2. Time vs distance correlation (if available)
 ax2 = axes[0, 1]
 if 'travel_time_validation' in self.validation_results:
 dist_val = self.validation_results['travel_time_validation'].get('distance_validation', {})
 
 dest_types = []
 correlations = []
 
 for dest_type, data in dist_val.items():
 if isinstance(data, dict) and 'correlation' in data:
 dest_types.append(dest_type)
 correlations.append(data['correlation'])
 
 if dest_types:
 colors = ['green' if c > 0.5 else 'orange' if c > 0.3 else 'red' for c in correlations]
 bars = ax2.bar(dest_types, correlations, color=colors, alpha=0.7)
 ax2.set_ylabel('Distance-Time Correlation')
 ax2.set_title('Distance-Time Relationships')
 ax2.axhline(0.3, color='orange', linestyle='--', alpha=0.5, label='Minimum Expected')
 ax2.axhline(0.5, color='green', linestyle='--', alpha=0.5, label='Good Correlation')
 ax2.legend()
 ax2.grid(True, alpha=0.3)
 
 # 3. Time consistency check
 ax3 = axes[1, 0]
 dest_types = ['employment', 'healthcare', 'grocery']
 
 for i, dest_type in enumerate(dest_types):
 start_idx = i * 10
 if start_idx + 2 < len(time_features):
 min_times = time_values[:, start_idx]
 mean_times = time_values[:, start_idx + 1]
 
 ax3.scatter(min_times, mean_times, alpha=0.6, s=20, label=dest_type)
 
 # Perfect correlation line
 max_time = np.max(time_values)
 ax3.plot([0, max_time], [0, max_time], 'r--', alpha=0.5, label='Perfect Correlation')
 ax3.set_xlabel('Minimum Time')
 ax3.set_ylabel('Mean Time')
 ax3.set_title('Time Consistency (Min vs Mean)')
 ax3.legend()
 ax3.grid(True, alpha=0.3)
 
 # 4. Outlier analysis
 ax4 = axes[1, 1]
 
 # Box plot of travel times
 time_data_for_plot = [time_values[:, i] for i in range(min(len(time_names), 6))]
 plot_names = [name.split('_')[0][:8] for name in time_names[:6]]
 
 ax4.boxplot(time_data_for_plot, labels=plot_names)
 ax4.set_ylabel('Travel Time (minutes)')
 ax4.set_title('Travel Time Outlier Analysis')
 ax4.tick_params(axis='x', rotation=45)
 ax4.grid(True, alpha=0.3)
 
 plt.tight_layout()
 plt.savefig(f"{output_dir}/travel_time_analysis.png", dpi=300, bbox_inches='tight')
 plt.close()
 
 def _plot_feature_relationships(self, output_dir: str):
 """Plot feature relationship analysis"""
 
 fig, axes = plt.subplots(2, 2, figsize=(15, 12))
 fig.suptitle('Feature Relationship Analysis', fontsize=16)
 
 # 1. Time vs Count relationship
 ax1 = axes[0, 0]
 time_features = [i for i, name in enumerate(self.feature_names) if 'min_time' in name]
 count_features = [i for i, name in enumerate(self.feature_names) if 'count_5min' in name]
 
 if time_features and count_features:
 for i, (time_idx, count_idx) in enumerate(zip(time_features, count_features)):
 dest_type = ['employment', 'healthcare', 'grocery'][i] if i < 3 else f'dest_{i}'
 
 times = self.accessibility_features[:, time_idx]
 counts = self.accessibility_features[:, count_idx]
 
 ax1.scatter(times, counts, alpha=0.6, s=20, label=dest_type)
 
 ax1.set_xlabel('Minimum Travel Time (minutes)')
 ax1.set_ylabel('Destination Count (60min)')
 ax1.set_title('Time vs Count Relationship')
 ax1.legend()
 ax1.grid(True, alpha=0.3)
 
 # 2. Cross-destination correlations
 ax2 = axes[0, 1]
 dest_types = ['employment', 'healthcare', 'grocery']
 accessibility_scores = []
 
 for dest_idx, dest_type in enumerate(dest_types):
 start_idx = dest_idx * 10
 if start_idx + 7 < len(self.feature_names):
 score = self.accessibility_features[:, start_idx + 7]
 accessibility_scores.append(score)
 
 if len(accessibility_scores) >= 2:
 ax2.scatter(accessibility_scores[0], accessibility_scores[1], 
 alpha=0.6, s=20, label=f'{dest_types[0]} vs {dest_types[1]}')
 
 if len(accessibility_scores) >= 3:
 ax2.scatter(accessibility_scores[0], accessibility_scores[2], 
 alpha=0.6, s=20, label=f'{dest_types[0]} vs {dest_types[2]}')
 
 ax2.set_xlabel(f'{dest_types[0]} Accessibility')
 ax2.set_ylabel('Other Destination Accessibility')
 ax2.set_title('Cross-Destination Correlations')
 ax2.legend()
 ax2.grid(True, alpha=0.3)
 
 # 3. Feature correlation heatmap
 ax3 = axes[1, 0]
 
 # Sample features for visualization
 n_sample = min(12, len(self.feature_names))
 sample_features = self.accessibility_features[:, :n_sample]
 sample_names = [name[:10] for name in self.feature_names[:n_sample]]
 
 corr_matrix = np.corrcoef(sample_features.T)
 im = ax3.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
 
 ax3.set_xticks(range(len(sample_names)))
 ax3.set_yticks(range(len(sample_names)))
 ax3.set_xticklabels(sample_names, rotation=45, ha='right', fontsize=8)
 ax3.set_yticklabels(sample_names, fontsize=8)
 ax3.set_title('Feature Correlation Matrix')
 
 plt.colorbar(im, ax=ax3)
 
 # 4. Feature scaling comparison
 ax4 = axes[1, 1]
 
 # Show feature ranges
 feature_ranges = np.ptp(self.accessibility_features, axis=0)
 feature_indices = range(min(len(self.feature_names), 12))
 
 bars = ax4.bar(feature_indices, feature_ranges[:len(feature_indices)], alpha=0.7)
 ax4.set_xlabel('Feature Index')
 ax4.set_ylabel('Range (Max - Min)')
 ax4.set_title('Feature Scaling Analysis')
 ax4.grid(True, alpha=0.3)
 
 # Color bars by magnitude
 max_range = np.max(feature_ranges)
 for i, bar in enumerate(bars):
 if i < len(feature_ranges):
 if feature_ranges[i] > max_range * 0.8:
 bar.set_color('red')
 elif feature_ranges[i] > max_range * 0.5:
 bar.set_color('orange')
 else:
 bar.set_color('green')
 
 plt.tight_layout()
 plt.savefig(f"{output_dir}/feature_relationships.png", dpi=300, bbox_inches='tight')
 plt.close()
 
 def _plot_geographic_patterns(self, output_dir: str):
 """Plot geographic pattern analysis"""
 
 fig, axes = plt.subplots(2, 2, figsize=(15, 12))
 fig.suptitle('Geographic Pattern Analysis', fontsize=16)
 
 # Get coordinates
 coords = np.array([[addr.geometry.x, addr.geometry.y] for _, addr in self.addresses.iterrows()])
 x_coords = coords[:, 0]
 y_coords = coords[:, 1]
 
 # 1. Overall accessibility spatial distribution
 ax1 = axes[0, 0]
 overall_accessibility = np.mean(self.accessibility_features, axis=1)
 
 scatter1 = ax1.scatter(x_coords, y_coords, c=overall_accessibility, 
 cmap='viridis', s=15, alpha=0.7)
 ax1.set_xlabel('Longitude')
 ax1.set_ylabel('Latitude')
 ax1.set_title('Overall Accessibility Distribution')
 ax1.set_aspect('equal')
 plt.colorbar(scatter1, ax=ax1)
 
 # 2. Distance to center vs accessibility
 ax2 = axes[0, 1]
 centroid = np.mean(coords, axis=0)
 distances_to_center = np.sqrt(np.sum((coords - centroid)**2, axis=1))
 
 ax2.scatter(distances_to_center, overall_accessibility, alpha=0.6, s=20)
 
 # Regression line
 z = np.polyfit(distances_to_center, overall_accessibility, 1)
 p = np.poly1d(z)
 ax2.plot(sorted(distances_to_center), p(sorted(distances_to_center)), "r--", alpha=0.8)
 
 corr = pearsonr(distances_to_center, overall_accessibility)[0]
 ax2.set_xlabel('Distance to Center')
 ax2.set_ylabel('Overall Accessibility')
 ax2.set_title(f'Center-Periphery Pattern (r={corr:.3f})')
 ax2.grid(True, alpha=0.3)
 
 # 3. Destination locations vs accessibility
 ax3 = axes[1, 0]
 
 # Plot destinations
 colors = ['red', 'blue', 'green']
 for i, (dest_type, dest_gdf) in enumerate(self.destinations.items()):
 if len(dest_gdf) > 0:
 dest_x = [dest.geometry.x for _, dest in dest_gdf.iterrows()]
 dest_y = [dest.geometry.y for _, dest in dest_gdf.iterrows()]
 ax3.scatter(dest_x, dest_y, c=colors[i % len(colors)], 
 s=100, marker='*', label=f'{dest_type} destinations', alpha=0.8)
 
 # Overlay accessibility
 scatter3 = ax3.scatter(x_coords, y_coords, c=overall_accessibility, 
 cmap='viridis', s=10, alpha=0.5)
 ax3.set_xlabel('Longitude')
 ax3.set_ylabel('Latitude')
 ax3.set_title('Destinations vs Accessibility')
 ax3.set_aspect('equal')
 ax3.legend()
 
 # 4. Accessibility clustering
 ax4 = axes[1, 1]
 
 # K-means clustering of accessibility
 n_clusters = min(5, len(coords) // 50)
 if n_clusters >= 2:
 kmeans = KMeans(n_clusters=n_clusters, random_state=42)
 clusters = kmeans.fit_predict(self.accessibility_features)
 
 scatter4 = ax4.scatter(x_coords, y_coords, c=clusters, 
 cmap='tab10', s=15, alpha=0.7)
 ax4.set_xlabel('Longitude')
 ax4.set_ylabel('Latitude')
 ax4.set_title('Accessibility-Based Clusters')
 ax4.set_aspect('equal')
 
 plt.tight_layout()
 plt.savefig(f"{output_dir}/geographic_patterns.png", dpi=300, bbox_inches='tight')
 plt.close()
 
 def _plot_data_quality_overview(self, output_dir: str):
 """Plot data quality overview"""
 
 fig, axes = plt.subplots(2, 2, figsize=(15, 12))
 fig.suptitle('Data Quality Overview', fontsize=16)
 
 # 1. Feature completeness
 ax1 = axes[0, 0]
 
 # Check for missing/problematic values per feature
 missing_rates = []
 feature_display_names = []
 
 for i, name in enumerate(self.feature_names):
 feature_values = self.accessibility_features[:, i]
 missing_count = np.sum(np.isnan(feature_values)) + np.sum(np.isinf(feature_values))
 missing_rate = missing_count / len(feature_values)
 missing_rates.append(missing_rate)
 feature_display_names.append(name[:15]) # Truncate for display
 
 bars = ax1.bar(range(len(missing_rates)), missing_rates, alpha=0.7)
 ax1.set_xlabel('Feature Index')
 ax1.set_ylabel('Missing/Invalid Rate')
 ax1.set_title('Feature Completeness')
 ax1.set_xticks(range(len(feature_display_names)))
 ax1.set_xticklabels(feature_display_names, rotation=45, ha='right', fontsize=8)
 ax1.grid(True, alpha=0.3)
 
 # Color bars by completeness
 for i, (bar, rate) in enumerate(zip(bars, missing_rates)):
 if rate > 0.1:
 bar.set_color('red')
 elif rate > 0.05:
 bar.set_color('orange')
 else:
 bar.set_color('green')
 
 # 2. Value distribution analysis
 ax2 = axes[0, 1]
 
 # Check for outliers
 outlier_counts = []
 
 for i in range(len(self.feature_names)):
 feature_values = self.accessibility_features[:, i]
 Q1 = np.percentile(feature_values, 25)
 Q3 = np.percentile(feature_values, 75)
 IQR = Q3 - Q1
 
 lower_bound = Q1 - 1.5 * IQR
 upper_bound = Q3 + 1.5 * IQR
 
 outliers = np.sum((feature_values < lower_bound) | (feature_values > upper_bound))
 outlier_counts.append(outliers)
 
 bars2 = ax2.bar(range(len(outlier_counts)), outlier_counts, alpha=0.7)
 ax2.set_xlabel('Feature Index')
 ax2.set_ylabel('Outlier Count')
 ax2.set_title('Outlier Analysis')
 ax2.grid(True, alpha=0.3)
 
 # 3. Correlation quality
 ax3 = axes[1, 0]
 
 # Analyze correlation structure
 corr_matrix = np.corrcoef(self.accessibility_features.T)
 
 # Count high correlations
 high_corr_count = np.sum(np.abs(corr_matrix) > 0.9) - len(self.feature_names) # Subtract diagonal
 perfect_corr_count = np.sum(np.abs(corr_matrix) > 0.99) - len(self.feature_names)
 
 categories = ['High Corr\n(>0.9)', 'Perfect Corr\n(>0.99)', 'Total Pairs']
 values = [high_corr_count, perfect_corr_count, len(self.feature_names) * (len(self.feature_names) - 1)]
 
 bars3 = ax3.bar(categories, values, alpha=0.7, color=['orange', 'red', 'blue'])
 ax3.set_ylabel('Count')
 ax3.set_title('Correlation Structure Quality')
 ax3.grid(True, alpha=0.3)
 
 # 4. Overall quality summary
 ax4 = axes[1, 1]
 ax4.axis('off')
 
 # Calculate quality metrics
 if 'data_quality' in self.validation_results:
 quality = self.validation_results['data_quality']['overall_assessment']
 
 quality_text = f"""Data Quality Summary:

Overall Grade: {quality['quality_grade']}
Quality Score: {quality['quality_score']:.1%}

Realistic Values: {quality['realistic_values_rate']:.1%}
Feature Completeness: {quality['feature_completeness_rate']:.1%}
Theoretical Consistency: {quality['theoretical_consistency_score']:.1%}

Issues Detected:
{chr(10).join(['• ' + issue for issue in self.validation_results['data_quality']['specific_issues'][:5]])}"""
 
 # Color background by grade
 grade_colors = {'A': 'lightgreen', 'B': 'lightyellow', 'C': 'orange', 'D': 'lightcoral', 'F': 'red'}
 bg_color = grade_colors.get(quality['quality_grade'], 'lightgray')
 
 ax4.text(0.05, 0.95, quality_text.strip(), transform=ax4.transAxes,
 fontsize=10, verticalalignment='top', fontfamily='monospace',
 bbox=dict(boxstyle='round,pad=0.5', facecolor=bg_color, alpha=0.8))
 
 plt.tight_layout()
 plt.savefig(f"{output_dir}/data_quality_overview.png", dpi=300, bbox_inches='tight')
 plt.close()


# ===== INTEGRATION FUNCTIONS =====

def validate_granite_accessibility_features(addresses, accessibility_features, destinations,
 feature_names=None, tract_svi=None,
 output_dir='./accessibility_validation'):
 """
 Convenience function to validate GRANITE accessibility features
 
 Args:
 addresses: GeoDataFrame of address points
 accessibility_features: Computed accessibility feature matrix 
 destinations: Dict of destination GeoDataFrames by type
 feature_names: List of feature names (optional)
 tract_svi: Target SVI for relationship validation (optional)
 output_dir: Directory for output visualizations
 
 Returns:
 Validation results dictionary
 """
 
 validator = AccessibilityFeatureValidator(verbose=True)
 
 # Run validation
 results = validator.validate_accessibility_pipeline(
 addresses=addresses,
 accessibility_features=accessibility_features,
 destinations=destinations,
 feature_names=feature_names,
 tract_svi=tract_svi
 )
 
 # Create visualizations
 validator.create_validation_visualizations(output_dir)
 
 return results, validator


def integrate_with_spatial_diagnostics(spatial_diagnostics_results, accessibility_validation_results):
 """
 Integrate accessibility validation results with spatial diagnostics
 
 Args:
 spatial_diagnostics_results: Results from SpatialLearningDiagnostics
 accessibility_validation_results: Results from AccessibilityFeatureValidator
 
 Returns:
 Integrated analysis results
 """
 
 integrated = {
 'spatial_analysis': spatial_diagnostics_results,
 'accessibility_validation': accessibility_validation_results,
 'integrated_diagnosis': {}
 }
 
 # Cross-validate findings
 spatial_corr = spatial_diagnostics_results['accessibility_correlations']['overall']
 
 if 'root_cause_analysis' in accessibility_validation_results:
 access_diagnosis = accessibility_validation_results['root_cause_analysis']['primary_diagnosis']
 
 # Determine integrated diagnosis
 if 'travel time' in access_diagnosis.lower() and spatial_corr > 0:
 integrated_diagnosis = "Travel time computation errors causing wrong accessibility-vulnerability relationship"
 elif 'destination' in access_diagnosis.lower() and spatial_corr > 0:
 integrated_diagnosis = "Destination accessibility counting errors leading to inverted relationships"
 elif spatial_corr > 0.5:
 integrated_diagnosis = "Strong positive correlation confirms systematic accessibility feature issues"
 else:
 integrated_diagnosis = "Mixed signals - requires detailed investigation"
 
 integrated['integrated_diagnosis'] = {
 'primary_diagnosis': integrated_diagnosis,
 'spatial_correlation': spatial_corr,
 'accessibility_diagnosis': access_diagnosis,
 'confidence': 'high' if abs(spatial_corr) > 0.5 else 'medium'
 }
 
 return integrated