"""
Modal accessibility features that account for vehicle ownership constraints.
Breaks the urban paradox: good spatial access ≠ effective access without a car.
"""

import numpy as np
from typing import Dict, Tuple

def compute_modal_features(
    accessibility_features: np.ndarray,
    feature_names: list,
    tract_svi_data: Dict[str, Dict],
    address_tract_ids: np.ndarray
) -> Tuple[np.ndarray, list]:
    """
    Compute 15 modal accessibility features (5 per destination type).
    
    Parameters
    ----------
    accessibility_features : array, shape (n_addresses, 30)
        Base accessibility features (10 per destination × 3 destinations)
    feature_names : list of str
        Names of the 30 base features
    tract_svi_data : dict
        {fips: {'EP_NOVEH': pct, 'EP_POV150': pct, ...}}
    address_tract_ids : array, shape (n_addresses,)
        FIPS code for each address's tract
        
    Returns
    -------
    modal_features : array, shape (n_addresses, 15)
    modal_names : list of str
    """
    
    n_addresses = len(address_tract_ids)
    
    # Extract vehicle ownership for each address
    pct_no_vehicle = np.array([
        tract_svi_data[fips]['EP_NOVEH'] 
        for fips in address_tract_ids
    ]) / 100.0  # Convert to [0, 1]
    
    modal_features = []
    modal_names = []
    
    # Process each destination type
    for dest_type in ['employment', 'healthcare', 'grocery']:
        
        # Extract relevant base features for this destination
        indices = {
            'count_5min': feature_names.index(f'{dest_type}_count_5min'),
            'min_time': feature_names.index(f'{dest_type}_min_time'),
            'drive_advantage': feature_names.index(f'{dest_type}_drive_advantage'),
        }
        
        count_5min = accessibility_features[:, indices['count_5min']]
        min_time = accessibility_features[:, indices['min_time']]
        drive_advantage = accessibility_features[:, indices['drive_advantage']]
        
        # Walkability rate: inverse of drive advantage
        # High when walking is competitive with driving
        walkability = 1.0 / (1.0 + drive_advantage)
        
        # === 5 MODAL FEATURES PER DESTINATION ===
        
        # 1. Transit dependence: no car AND not walkable
        # High = vulnerable (forced to use inadequate transit)
        transit_dep = pct_no_vehicle * (1.0 - walkability)
        modal_features.append(transit_dep)
        modal_names.append(f'{dest_type}_transit_dependence')
        
        # 2. Car-effective access: destinations weighted by vehicle availability
        # Low when no car despite many nearby destinations
        car_effective = count_5min * (1.0 - pct_no_vehicle)
        modal_features.append(car_effective)
        modal_names.append(f'{dest_type}_car_effective_access')
        
        # 3. Walk-effective access: destinations weighted by walkability
        # Shows whether proximity translates to actual accessibility
        walk_effective = count_5min * walkability
        modal_features.append(walk_effective)
        modal_names.append(f'{dest_type}_walk_effective_access')
        
        # 4. Modal access gap: advantage of car × lack of car
        # High = vulnerable (big benefit from car, but don't have one)
        modal_gap = drive_advantage * pct_no_vehicle
        modal_features.append(modal_gap)
        modal_names.append(f'{dest_type}_modal_access_gap')
        
        # 5. Forced walk burden: travel time × no vehicle rate
        # High = forced to walk/transit despite long distances
        forced_walk = min_time * pct_no_vehicle
        modal_features.append(forced_walk)
        modal_names.append(f'{dest_type}_forced_walk_burden')
    
    modal_features = np.column_stack(modal_features)
    
    return modal_features, modal_names


def validate_modal_correlations(
    modal_features: np.ndarray,
    modal_names: list,
    svi_values: np.ndarray
) -> float:
    """
    Validate modal features correlate correctly with SVI.
    Returns percentage of features with correct direction.
    """
    from scipy.stats import pearsonr
    
    expected_directions = {
        'transit_dependence': 'positive',      # More dependence → higher vulnerability
        'car_effective': 'negative',           # More car access → lower vulnerability
        'walk_effective': 'negative',          # More walk access → lower vulnerability
        'modal_access_gap': 'positive',        # Bigger gap → higher vulnerability
        'forced_walk_burden': 'positive',      # Forced to walk far → higher vulnerability
    }
    
    n_correct = 0
    n_total = 0
    
    print("\n" + "="*80)
    print("MODAL FEATURE VALIDATION")
    print("="*80)
    
    for i, name in enumerate(modal_names):
        corr, pval = pearsonr(modal_features[:, i], svi_values)
        
        # Determine expected direction
        exp_dir = None
        for key, direction in expected_directions.items():
            if key in name:
                exp_dir = direction
                break
        
        if exp_dir and pval < 0.05:
            n_total += 1
            correct = (corr > 0) == (exp_dir == 'positive')
            n_correct += int(correct)
            
            status = "✓" if correct else "✗"
            print(f"{status} {name:45s} r={corr:+.3f} p={pval:.3f} (expected {exp_dir})")
    
    pct_correct = (n_correct / n_total * 100) if n_total > 0 else 0
    print(f"\nCorrectness: {n_correct}/{n_total} ({pct_correct:.1f}%)")
    print("="*80)
    
    return pct_correct