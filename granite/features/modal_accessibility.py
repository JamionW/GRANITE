"""
Modal accessibility features computed per address from driving and walking travel times.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

def compute_modal_features(
    accessibility_features: np.ndarray,
    feature_names: list,
    tract_svi_data: Dict[str, Dict],
    address_tract_ids: np.ndarray,
    per_address_times: Optional[Dict[str, Dict]] = None,
    address_ids: Optional[List] = None
) -> Tuple[np.ndarray, list]:
    """
    Compute 15 modal accessibility features (5 per destination type).

    When per_address_times is provided (dict of dest_type -> {addr_id -> summary}),
    features are computed at the address level using per-address driving and walking
    travel times from OSRM. Otherwise falls back to tract-level approximation from
    base accessibility features.

    Parameters
    ----------
    accessibility_features : array, shape (n_addresses, 30)
        Base accessibility features (10 per destination x 3 destinations)
    feature_names : list of str
        Names of the 30 base features
    tract_svi_data : dict
        {fips: {'EP_NOVEH': pct, ...}}
    address_tract_ids : array, shape (n_addresses,)
        FIPS code for each address's tract
    per_address_times : dict, optional
        {dest_type: {addr_id: {'drive_nearest': float, 'walk_nearest': float,
                               'drive_times': array, 'walk_times': array}}}
    address_ids : list, optional
        Address IDs in row order, required when per_address_times is provided

    Returns
    -------
    modal_features : array, shape (n_addresses, 15)
    modal_names : list of str
    """

    n_addresses = len(address_tract_ids)
    use_per_address = (
        per_address_times is not None
        and address_ids is not None
        and len(per_address_times) > 0
    )

    modal_features = []
    modal_names = []

    for dest_type in ['employment', 'healthcare', 'grocery']:

        if use_per_address and dest_type in per_address_times:
            dest_summaries = per_address_times[dest_type]

            avg_time = np.empty(n_addresses)
            time_std = np.empty(n_addresses)
            access_density = np.empty(n_addresses)
            equity_gap = np.empty(n_addresses)
            car_advantage = np.empty(n_addresses)

            for i, addr_id in enumerate(address_ids):
                s = dest_summaries.get(addr_id)
                if s is None:
                    avg_time[i] = 120.0
                    time_std[i] = 0.0
                    access_density[i] = 0.0
                    equity_gap[i] = 0.0
                    car_advantage[i] = 1.0
                    continue

                dn = s['drive_nearest']
                wn = s['walk_nearest']
                dt = s['drive_times']
                wt = s['walk_times']

                # 1. avg_time: mean of drive and walk time to nearest destination
                avg_time[i] = (dn + wn) / 2.0

                # 2. time_std: std of those two values
                time_std[i] = np.std([dn, wn])

                # 3. access_density: destinations reachable within 10 min by either mode
                drive_reachable = set(np.where(dt <= 10)[0])
                walk_reachable = set(np.where(wt <= 10)[0])
                # union: min of array lengths handles mismatched dest counts
                n_union = len(drive_reachable | walk_reachable)
                access_density[i] = float(n_union)

                # 4. equity_gap: |walk_nearest - drive_nearest|
                equity_gap[i] = abs(wn - dn)

                # 5. car_advantage: walk_nearest / drive_nearest
                if dn > 0:
                    car_advantage[i] = wn / dn
                else:
                    car_advantage[i] = 1.0

        else:
            # fallback: derive from base features (tract-level approximation)
            indices = {
                'count_5min': feature_names.index(f'{dest_type}_count_5min'),
                'count_10min': feature_names.index(f'{dest_type}_count_10min'),
                'min_time': feature_names.index(f'{dest_type}_min_time'),
                'mean_time': feature_names.index(f'{dest_type}_mean_time'),
                'drive_advantage': feature_names.index(f'{dest_type}_drive_advantage'),
            }

            min_t = accessibility_features[:, indices['min_time']]
            mean_t = accessibility_features[:, indices['mean_time']]
            da = accessibility_features[:, indices['drive_advantage']]
            count_10 = accessibility_features[:, indices['count_10min']]

            # reconstruct approximate walk/drive from drive_advantage ratio
            # da = (walk_avg - drive_avg) / walk_avg => drive_avg = walk_avg * (1 - da)
            # use min_time as proxy for drive_nearest, estimate walk from da
            drive_est = min_t
            walk_est = np.where(da < 1.0, min_t / (1.0 - da + 1e-8), min_t * 5.0)

            avg_time = (drive_est + walk_est) / 2.0
            time_std = np.abs(walk_est - drive_est) / 2.0
            access_density = count_10.copy()
            equity_gap = np.abs(walk_est - drive_est)
            car_advantage = np.where(drive_est > 0, walk_est / drive_est, 1.0)

        modal_features.append(avg_time)
        modal_names.append(f'{dest_type}_transit_dependence')

        modal_features.append(time_std)
        modal_names.append(f'{dest_type}_car_effective_access')

        modal_features.append(access_density)
        modal_names.append(f'{dest_type}_walk_effective_access')

        modal_features.append(equity_gap)
        modal_names.append(f'{dest_type}_modal_access_gap')

        modal_features.append(car_advantage)
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
        'transit_dependence': 'positive',      # higher avg_time -> higher vulnerability
        'car_effective': 'negative',           # higher time_std -> mode disparity
        'walk_effective': 'negative',          # more access_density -> lower vulnerability
        'modal_access_gap': 'positive',        # bigger equity gap -> higher vulnerability
        'forced_walk_burden': 'positive',      # higher car_advantage -> car-dependent
    }

    n_correct = 0
    n_total = 0

    print("\n" + "="*80)
    print("MODAL FEATURE VALIDATION")
    print("="*80)

    for i, name in enumerate(modal_names):
        corr, pval = pearsonr(modal_features[:, i], svi_values)

        exp_dir = None
        for key, direction in expected_directions.items():
            if key in name:
                exp_dir = direction
                break

        if exp_dir and pval < 0.05:
            n_total += 1
            correct = (corr > 0) == (exp_dir == 'positive')
            n_correct += int(correct)

            status = "" if correct else ""
            print(f"{status} {name:45s} r={corr:+.3f} p={pval:.3f} (expected {exp_dir})")

    pct_correct = (n_correct / n_total * 100) if n_total > 0 else 0
    print(f"\nCorrectness: {n_correct}/{n_total} ({pct_correct:.1f}%)")
    print("="*80)

    return pct_correct
