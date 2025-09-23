"""
Feature Deduplication and Enhanced Validation
Add this to your pipeline after feature computation
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from typing import Tuple, List

def deduplicate_accessibility_features(features: np.ndarray, feature_names: List[str], 
                                     correlation_threshold: float = 0.99, 
                                     verbose: bool = True) -> Tuple[np.ndarray, List[str]]:
    """
    Remove perfectly correlated and redundant features
    
    Args:
        features: Feature matrix [n_addresses, n_features]
        feature_names: List of feature names
        correlation_threshold: Correlation threshold for deduplication
        verbose: Whether to print deduplication info
        
    Returns:
        deduplicated_features, deduplicated_feature_names
    """
    
    if verbose:
        print(f"[FeatureDedup] Starting with {features.shape[1]} features")
    
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(features.T)
    
    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            if abs(corr_matrix[i, j]) >= correlation_threshold:
                high_corr_pairs.append((i, j, corr_matrix[i, j], feature_names[i], feature_names[j]))
    
    if verbose and high_corr_pairs:
        print(f"[FeatureDedup] Found {len(high_corr_pairs)} highly correlated pairs:")
        for i, j, corr, name1, name2 in high_corr_pairs[:5]:  # Show first 5
            print(f"  {name1} <-> {name2}: {corr:.4f}")
    
    # Determine which features to remove
    features_to_remove = set()
    
    for i, j, corr, name1, name2 in high_corr_pairs:
        if i not in features_to_remove and j not in features_to_remove:
            # Decide which feature to remove based on name/importance
            if _should_remove_feature(name1, name2):
                features_to_remove.add(j)
            else:
                features_to_remove.add(i)
    
    # Create indices for features to keep
    features_to_keep = [i for i in range(features.shape[1]) if i not in features_to_remove]
    
    # Apply deduplication
    deduplicated_features = features[:, features_to_keep]
    deduplicated_names = [feature_names[i] for i in features_to_keep]
    
    if verbose:
        removed_count = len(features_to_remove)
        print(f"[FeatureDedup] Removed {removed_count} redundant features")
        print(f"[FeatureDedup] Final feature count: {deduplicated_features.shape[1]}")
        
        if removed_count > 0:
            removed_names = [feature_names[i] for i in sorted(features_to_remove)]
            print(f"[FeatureDedup] Removed features: {removed_names[:3]}...")  # Show first 3
    
    return deduplicated_features, deduplicated_names

def _should_remove_feature(name1: str, name2: str) -> bool:
    """
    Decide which of two highly correlated features to remove
    
    Priority order (keep these):
    1. min_time (most fundamental)
    2. count features (directly interpretable)
    3. mean_time (representative)
    4. Others by alphabetical order
    """
    
    # Priority keywords (higher priority = keep)
    priority_keywords = ['min_time', 'count_5min', 'count_10min', 'mean_time', 'percentile']
    
    name1_priority = _get_feature_priority(name1, priority_keywords)
    name2_priority = _get_feature_priority(name2, priority_keywords)
    
    if name1_priority != name2_priority:
        return name1_priority < name2_priority  # Remove lower priority
    else:
        return name1 > name2  # Alphabetical tiebreaker (remove later alphabetically)

def _get_feature_priority(name: str, priority_keywords: List[str]) -> int:
    """Get priority score for a feature (higher = more important)"""
    for i, keyword in enumerate(priority_keywords):
        if keyword in name:
            return len(priority_keywords) - i
    return 0  # Default priority

def validate_feature_quality(features: np.ndarray, feature_names: List[str], 
                           addresses_count: int, verbose: bool = True) -> dict:
    """
    Comprehensive feature quality validation
    
    Returns:
        validation_results: Dictionary with validation metrics
    """
    
    validation = {}
    
    # Basic validation
    n_addresses, n_features = features.shape
    validation['basic'] = {
        'n_addresses': n_addresses,
        'n_features': n_features,
        'expected_addresses': addresses_count,
        'address_count_match': n_addresses == addresses_count
    }
    
    # Check for problematic values
    validation['data_quality'] = {
        'has_nan': bool(np.any(np.isnan(features))),
        'has_inf': bool(np.any(np.isinf(features))),
        'has_negative': bool(np.any(features < 0)),
        'nan_count': int(np.sum(np.isnan(features))),
        'inf_count': int(np.sum(np.isinf(features))),
        'negative_count': int(np.sum(features < 0))
    }
    
    # Feature variance analysis
    feature_stds = np.std(features, axis=0)
    validation['variance'] = {
        'zero_variance_count': int(np.sum(feature_stds < 1e-8)),
        'low_variance_count': int(np.sum(feature_stds < 0.01)),
        'zero_variance_features': [feature_names[i] for i in range(n_features) 
                                 if feature_stds[i] < 1e-8],
        'min_std': float(np.min(feature_stds)),
        'max_std': float(np.max(feature_stds)),
        'mean_std': float(np.mean(feature_stds))
    }
    
    # Feature correlation analysis (sample if too many features)
    if n_features > 50:
        # Sample features for correlation analysis
        sample_indices = np.random.choice(n_features, 50, replace=False)
        sample_features = features[:, sample_indices]
        sample_names = [feature_names[i] for i in sample_indices]
    else:
        sample_features = features
        sample_names = feature_names
    
    corr_matrix = np.corrcoef(sample_features.T)
    
    # Find high correlations
    high_corr_count = 0
    perfect_corr_count = 0
    
    for i in range(len(sample_names)):
        for j in range(i + 1, len(sample_names)):
            corr = abs(corr_matrix[i, j])
            if corr > 0.9:
                high_corr_count += 1
            if corr > 0.99:
                perfect_corr_count += 1
    
    validation['correlations'] = {
        'high_correlation_pairs': high_corr_count,
        'perfect_correlation_pairs': perfect_corr_count,
        'correlation_density': high_corr_count / (len(sample_names) * (len(sample_names) - 1) / 2)
    }
    
    # Feature type analysis
    feature_type_counts = _analyze_feature_types(feature_names)
    validation['feature_types'] = feature_type_counts
    
    # Overall quality score
    quality_score = _compute_quality_score(validation)
    validation['overall_quality'] = quality_score
    
    if verbose:
        _print_validation_summary(validation)
    
    return validation

def _analyze_feature_types(feature_names: List[str]) -> dict:
    """Analyze the types of features present"""
    
    type_counts = {
        'time_features': 0,
        'count_features': 0,
        'advantage_features': 0,
        'concentration_features': 0,
        'percentile_features': 0,
        'derived_features': 0,
        'unknown_features': 0
    }
    
    for name in feature_names:
        name_lower = name.lower()
        if 'time' in name_lower:
            type_counts['time_features'] += 1
        elif 'count' in name_lower:
            type_counts['count_features'] += 1
        elif 'advantage' in name_lower:
            type_counts['advantage_features'] += 1
        elif 'concentration' in name_lower:
            type_counts['concentration_features'] += 1
        elif 'percentile' in name_lower:
            type_counts['percentile_features'] += 1
        elif any(keyword in name_lower for keyword in ['accessibility', 'flexibility', 'equity', 'geographic']):
            type_counts['derived_features'] += 1
        else:
            type_counts['unknown_features'] += 1
    
    return type_counts

def _compute_quality_score(validation: dict) -> dict:
    """Compute overall feature quality score"""
    
    score_components = []
    
    # Address count matching
    if validation['basic']['address_count_match']:
        score_components.append(100)
    else:
        score_components.append(0)
    
    # Data quality
    data_quality = validation['data_quality']
    total_values = validation['basic']['n_addresses'] * validation['basic']['n_features']
    
    if total_values > 0:
        problem_rate = (data_quality['nan_count'] + data_quality['inf_count']) / total_values
        data_score = max(0, 100 * (1 - problem_rate * 10))  # Heavy penalty for NaN/inf
        score_components.append(data_score)
    
    # Variance quality
    variance_info = validation['variance']
    if validation['basic']['n_features'] > 0:
        zero_var_rate = variance_info['zero_variance_count'] / validation['basic']['n_features']
        variance_score = max(0, 100 * (1 - zero_var_rate))
        score_components.append(variance_score)
    
    # Correlation quality (lower correlation density is better)
    corr_density = validation['correlations']['correlation_density']
    correlation_score = max(0, 100 * (1 - corr_density))
    score_components.append(correlation_score)
    
    overall_score = np.mean(score_components)
    
    return {
        'overall_score': overall_score,
        'component_scores': {
            'address_matching': score_components[0] if len(score_components) > 0 else 0,
            'data_quality': score_components[1] if len(score_components) > 1 else 0,
            'variance_quality': score_components[2] if len(score_components) > 2 else 0,
            'correlation_quality': score_components[3] if len(score_components) > 3 else 0
        },
        'grade': 'A' if overall_score > 90 else 
                'B' if overall_score > 80 else 
                'C' if overall_score > 70 else 
                'D' if overall_score > 60 else 'F'
    }

def _print_validation_summary(validation: dict):
    """Print feature validation summary"""
    
    basic = validation['basic']
    quality = validation['data_quality']
    variance = validation['variance']
    corr = validation['correlations']
    overall = validation['overall_quality']
    
    print(f"\n[FeatureValidation] === FEATURE QUALITY SUMMARY ===")
    print(f"Addresses: {basic['n_addresses']} (expected: {basic['expected_addresses']})")
    print(f"Features: {basic['n_features']}")
    print(f"")
    print(f"Data Quality:")
    print(f"  NaN values: {quality['nan_count']}")
    print(f"  Infinite values: {quality['inf_count']} ")
    print(f"  Negative values: {quality['negative_count']}")
    print(f"")
    print(f"Feature Variance:")
    print(f"  Zero variance: {variance['zero_variance_count']} features")
    print(f"  Low variance: {variance['low_variance_count']} features")
    print(f"  Mean std: {variance['mean_std']:.4f}")
    print(f"")
    print(f"Feature Correlations:")
    print(f"  High correlation pairs: {corr['high_correlation_pairs']}")
    print(f"  Perfect correlation pairs: {corr['perfect_correlation_pairs']}")
    print(f"")
    print(f"Overall Quality: {overall['grade']} ({overall['overall_score']:.1f}%)")
    
    if variance['zero_variance_features']:
        print(f"Zero variance features: {variance['zero_variance_features'][:3]}...")

def enhance_accessibility_features_with_validation(features: np.ndarray, 
                                                  feature_names: List[str],
                                                  addresses_count: int,
                                                  verbose: bool = True) -> Tuple[np.ndarray, List[str], dict]:
    """
    Complete feature enhancement pipeline with validation
    
    Returns:
        enhanced_features, enhanced_feature_names, validation_results
    """
    
    if verbose:
        print(f"[FeatureEnhancement] Starting enhancement pipeline...")
    
    # Step 1: Validate input features
    validation_results = validate_feature_quality(features, feature_names, addresses_count, verbose)
    
    # Step 2: Handle data quality issues
    if validation_results['data_quality']['has_nan'] or validation_results['data_quality']['has_inf']:
        if verbose:
            print(f"[FeatureEnhancement] Cleaning data quality issues...")
        features = np.nan_to_num(features, nan=0.0, posinf=999.0, neginf=-999.0)
    
    # Step 3: Remove perfectly correlated features
    if validation_results['correlations']['perfect_correlation_pairs'] > 0:
        features, feature_names = deduplicate_accessibility_features(
            features, feature_names, correlation_threshold=0.99, verbose=verbose
        )
    
    # Step 4: Final validation
    final_validation = validate_feature_quality(features, feature_names, addresses_count, verbose)
    
    if verbose:
        print(f"[FeatureEnhancement] Enhancement complete!")
        print(f"  Final features: {features.shape[1]} (was {validation_results['basic']['n_features']})")
        print(f"  Final quality: {final_validation['overall_quality']['grade']}")
    
    return features, feature_names, final_validation