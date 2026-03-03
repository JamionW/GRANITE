"""
GRANITE Expert Routing Analysis

Identifies what accessibility features distinguish tracts routed to each expert.
Explains counterintuitive routing (e.g., why low-SVI tract 11900 -> High expert).

Key insight: Expert routing may be based on ACCESSIBILITY PATTERNS, not SVI.
A low-SVI suburb with poor transit access may route to the same expert as
high-SVI urban cores with similar accessibility challenges.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/workspaces/GRANITE')


def analyze_expert_routing(tract_results, accessibility_features, feature_names,
                           verbose=True):
    """
    Analyze what distinguishes tracts routed to each expert.
    
    Args:
        tract_results: Dict of {fips: {'actual_svi': float, 'dominant_expert': str, ...}}
        accessibility_features: Dict of {fips: np.array} with features per tract
        feature_names: List of feature names
        
    Returns:
        dict with expert-level feature profiles and distinguishing characteristics
    """
    
    if verbose:
        print("\n" + "="*70)
        print("EXPERT ROUTING FEATURE ANALYSIS")
        print("="*70)
    
    # Group tracts by dominant expert
    expert_groups = defaultdict(list)
    for fips, result in tract_results.items():
        expert = result.get('dominant_expert', 'Unknown')
        expert_groups[expert].append(fips)
    
    if verbose:
        print("\nTracts per expert:")
        for expert, tracts in sorted(expert_groups.items()):
            print(f"  {expert}: {len(tracts)} tracts")
    
    results = {
        'expert_groups': dict(expert_groups),
        'feature_profiles': {},
        'distinguishing_features': {},
        'counterintuitive_cases': []
    }
    
    # Compute feature profiles per expert
    for expert, tracts in expert_groups.items():
        if len(tracts) == 0:
            continue
        
        # Aggregate features for this expert's tracts
        expert_features = []
        expert_svis = []
        
        for fips in tracts:
            if fips in accessibility_features:
                expert_features.append(accessibility_features[fips])
            expert_svis.append(tract_results[fips]['actual_svi'])
        
        if len(expert_features) == 0:
            continue
        
        expert_features = np.vstack(expert_features)
        
        # Compute mean and std per feature
        feature_means = np.mean(expert_features, axis=0)
        feature_stds = np.std(expert_features, axis=0)
        
        results['feature_profiles'][expert] = {
            'n_tracts': len(tracts),
            'svi_mean': np.mean(expert_svis),
            'svi_std': np.std(expert_svis),
            'svi_range': (min(expert_svis), max(expert_svis)),
            'feature_means': feature_means,
            'feature_stds': feature_stds
        }
        
        if verbose:
            print(f"\n{expert} Expert Profile:")
            print(f"  Tracts: {len(tracts)}")
            print(f"  SVI range: {min(expert_svis):.3f} - {max(expert_svis):.3f}")
            print(f"  SVI mean: {np.mean(expert_svis):.3f}")
    
    # Find distinguishing features between experts
    if verbose:
        print("\n" + "-"*70)
        print("DISTINGUISHING FEATURES BY EXPERT")
        print("-"*70)
    
    experts = list(results['feature_profiles'].keys())
    if len(experts) >= 2:
        # Compare each pair of experts
        for i, expert1 in enumerate(experts):
            for expert2 in experts[i+1:]:
                profile1 = results['feature_profiles'][expert1]
                profile2 = results['feature_profiles'][expert2]
                
                if profile1['n_tracts'] < 2 or profile2['n_tracts'] < 2:
                    continue
                
                # Find features that differ most between these experts
                diff_scores = []
                for j, fname in enumerate(feature_names):
                    mean1 = profile1['feature_means'][j]
                    mean2 = profile2['feature_means'][j]
                    std_pooled = np.sqrt((profile1['feature_stds'][j]**2 + 
                                         profile2['feature_stds'][j]**2) / 2)
                    
                    if std_pooled > 0:
                        effect_size = abs(mean1 - mean2) / std_pooled
                    else:
                        effect_size = 0
                    
                    diff_scores.append((fname, effect_size, mean1, mean2))
                
                # Sort by effect size
                diff_scores.sort(key=lambda x: -x[1])
                
                key = f"{expert1}_vs_{expert2}"
                results['distinguishing_features'][key] = diff_scores[:10]
                
                if verbose:
                    print(f"\n{expert1} vs {expert2}:")
                    print(f"  {'Feature':<45} {'Effect':>8} {expert1:>10} {expert2:>10}")
                    print(f"  {'-'*75}")
                    for fname, effect, m1, m2 in diff_scores[:5]:
                        print(f"  {fname:<45} {effect:>8.2f} {m1:>10.3f} {m2:>10.3f}")
    
    # Identify counterintuitive cases
    if verbose:
        print("\n" + "-"*70)
        print("COUNTERINTUITIVE ROUTING CASES")
        print("-"*70)
    
    for fips, result in tract_results.items():
        svi = result['actual_svi']
        expert = result.get('dominant_expert', 'Unknown')
        
        # Flag counterintuitive: low SVI to High expert or high SVI to Low expert
        is_counterintuitive = False
        reason = ""
        
        if svi < 0.3 and 'High' in expert:
            is_counterintuitive = True
            reason = f"Low SVI ({svi:.3f}) routed to {expert} expert"
        elif svi > 0.7 and 'Low' in expert:
            is_counterintuitive = True
            reason = f"High SVI ({svi:.3f}) routed to {expert} expert"
        
        if is_counterintuitive:
            case = {
                'fips': fips,
                'svi': svi,
                'expert': expert,
                'reason': reason,
                'features': accessibility_features.get(fips)
            }
            results['counterintuitive_cases'].append(case)
            
            if verbose:
                print(f"\n  Tract {fips}: {reason}")
                
                # Explain based on features if available
                if fips in accessibility_features:
                    features = accessibility_features[fips]
                    
                    # Find features that explain routing
                    # (compare to the expected expert's profile)
                    if 'High' in expert and 'High' in results['feature_profiles']:
                        high_profile = results['feature_profiles']['High']
                        similarities = []
                        
                        for j, fname in enumerate(feature_names):
                            if 'min_time' in fname or 'transit' in fname:
                                feat_val = features[j]
                                expert_mean = high_profile['feature_means'][j]
                                similarity = abs(feat_val - expert_mean)
                                similarities.append((fname, feat_val, expert_mean, similarity))
                        
                        similarities.sort(key=lambda x: x[3])
                        
                        print(f"    Possible explanation (features similar to {expert} profile):")
                        for fname, val, exp_mean, _ in similarities[:3]:
                            print(f"      {fname}: {val:.3f} (expert mean: {exp_mean:.3f})")
    
    if len(results['counterintuitive_cases']) == 0 and verbose:
        print("  No counterintuitive routing cases found.")
    
    return results


def explain_single_tract_routing(fips, accessibility_features, expert_profiles, 
                                 feature_names, actual_svi, dominant_expert):
    """
    Explain why a specific tract was routed to its expert.
    
    Args:
        fips: Tract FIPS code
        accessibility_features: Features for this tract
        expert_profiles: Dict of expert feature profiles
        feature_names: List of feature names
        actual_svi: Actual SVI for this tract
        dominant_expert: Which expert this tract was routed to
        
    Returns:
        Explanation dict with key features driving the routing
    """
    
    explanation = {
        'fips': fips,
        'actual_svi': actual_svi,
        'dominant_expert': dominant_expert,
        'routing_factors': []
    }
    
    # Compare tract's features to the dominant expert's profile
    if dominant_expert not in expert_profiles:
        explanation['error'] = f"No profile for expert {dominant_expert}"
        return explanation
    
    profile = expert_profiles[dominant_expert]
    
    # Find features where this tract is closest to expert mean
    similarities = []
    for i, fname in enumerate(feature_names):
        if i >= len(accessibility_features):
            continue
        
        tract_val = accessibility_features[i]
        expert_mean = profile['feature_means'][i]
        expert_std = profile['feature_stds'][i]
        
        if expert_std > 0:
            z_score = abs(tract_val - expert_mean) / expert_std
        else:
            z_score = 0
        
        similarities.append({
            'feature': fname,
            'tract_value': tract_val,
            'expert_mean': expert_mean,
            'z_score': z_score,
            'is_close': z_score < 1.0
        })
    
    # Sort by closeness (low z-score = close to expert mean)
    similarities.sort(key=lambda x: x['z_score'])
    
    # Top factors that align with expert
    explanation['routing_factors'] = [s for s in similarities[:5] if s['is_close']]
    
    # Features that are different from expected
    explanation['deviations'] = [s for s in similarities if s['z_score'] > 2.0][:3]
    
    return explanation


def create_routing_summary_table(tract_results, accessibility_features, feature_names):
    """
    Create a summary table of tracts, their SVIs, and routing explanations.
    """
    
    rows = []
    for fips, result in tract_results.items():
        row = {
            'FIPS': fips,
            'Actual_SVI': result['actual_svi'],
            'Predicted_SVI': result.get('predicted_mean', np.nan),
            'Expert': result.get('dominant_expert', 'Unknown'),
            'Error_Pct': result.get('error_pct', result.get('mean_error_pct', np.nan))
        }
        
        # Add key accessibility features
        if fips in accessibility_features:
            features = accessibility_features[fips]
            
            # Find time and count features
            for i, fname in enumerate(feature_names):
                if i < len(features):
                    if 'employment_min_time' in fname:
                        row['Emp_Min_Time'] = features[i]
                    elif 'healthcare_min_time' in fname:
                        row['Health_Min_Time'] = features[i]
                    elif 'transit_dependence' in fname:
                        row['Transit_Dep'] = features[i]
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.sort_values('Actual_SVI')
    
    return df


def demo_with_mock_data():
    """Demonstrate expert routing analysis with mock data."""
    
    print("\n" + "="*70)
    print("DEMO: Expert Routing Analysis")
    print("="*70)
    
    np.random.seed(42)
    
    # Mock tract results
    tract_results = {
        '47065010100': {'actual_svi': 0.15, 'dominant_expert': 'Low', 'predicted_mean': 0.18},
        '47065010200': {'actual_svi': 0.22, 'dominant_expert': 'Low', 'predicted_mean': 0.25},
        '47065010300': {'actual_svi': 0.35, 'dominant_expert': 'Medium', 'predicted_mean': 0.38},
        '47065010400': {'actual_svi': 0.48, 'dominant_expert': 'Medium', 'predicted_mean': 0.45},
        '47065010500': {'actual_svi': 0.65, 'dominant_expert': 'High', 'predicted_mean': 0.62},
        '47065010600': {'actual_svi': 0.78, 'dominant_expert': 'High', 'predicted_mean': 0.75},
        # Counterintuitive cases
        '47065011900': {'actual_svi': 0.28, 'dominant_expert': 'High', 'predicted_mean': 0.35},  # Low SVI -> High
        '47065012000': {'actual_svi': 0.72, 'dominant_expert': 'Low', 'predicted_mean': 0.68},   # High SVI -> Low
    }
    
    # Mock feature names
    feature_names = [
        'employment_min_time', 'employment_count_5min', 'employment_transit_dependence',
        'healthcare_min_time', 'healthcare_count_5min', 'healthcare_modal_gap',
        'grocery_min_time', 'grocery_count_5min', 'grocery_walk_effective'
    ]
    
    # Mock accessibility features
    # Low SVI tracts: good accessibility
    # High SVI tracts: poor accessibility
    # Counterintuitive: accessibility pattern doesn't match SVI
    accessibility_features = {}
    
    for fips, result in tract_results.items():
        svi = result['actual_svi']
        expert = result['dominant_expert']
        
        if fips == '47065011900':
            # Low SVI but poor accessibility (explains High expert routing)
            features = np.array([25.0, 2, 0.8, 20.0, 1, 0.7, 15.0, 3, 0.2])
        elif fips == '47065012000':
            # High SVI but good accessibility (explains Low expert routing)
            features = np.array([5.0, 15, 0.1, 8.0, 8, 0.2, 4.0, 10, 0.8])
        elif 'Low' in expert:
            # Good accessibility
            features = np.array([5 + np.random.rand()*3, 12 + np.random.rand()*5, 0.1 + np.random.rand()*0.2,
                                8 + np.random.rand()*3, 6 + np.random.rand()*4, 0.15 + np.random.rand()*0.2,
                                4 + np.random.rand()*2, 8 + np.random.rand()*4, 0.7 + np.random.rand()*0.2])
        elif 'Medium' in expert:
            features = np.array([12 + np.random.rand()*5, 6 + np.random.rand()*4, 0.35 + np.random.rand()*0.2,
                                14 + np.random.rand()*4, 3 + np.random.rand()*3, 0.4 + np.random.rand()*0.2,
                                10 + np.random.rand()*4, 4 + np.random.rand()*3, 0.4 + np.random.rand()*0.2])
        else:  # High expert
            # Poor accessibility
            features = np.array([22 + np.random.rand()*8, 2 + np.random.rand()*3, 0.65 + np.random.rand()*0.3,
                                18 + np.random.rand()*6, 1 + np.random.rand()*2, 0.6 + np.random.rand()*0.3,
                                15 + np.random.rand()*5, 2 + np.random.rand()*2, 0.25 + np.random.rand()*0.2])
        
        accessibility_features[fips] = features
    
    # Run analysis
    results = analyze_expert_routing(
        tract_results, 
        accessibility_features, 
        feature_names,
        verbose=True
    )
    
    # Create summary table
    print("\n" + "-"*70)
    print("ROUTING SUMMARY TABLE")
    print("-"*70)
    
    summary_df = create_routing_summary_table(
        tract_results, accessibility_features, feature_names
    )
    print(summary_df.to_string(index=False))
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Expert Routing Analysis')
    parser.add_argument('--demo', action='store_true', help='Run demo with mock data')
    
    args = parser.parse_args()
    
    if args.demo:
        demo_with_mock_data()
    else:
        print("Use --demo to run demonstration with mock data")
        print("For real data, import and call analyze_expert_routing() directly")
