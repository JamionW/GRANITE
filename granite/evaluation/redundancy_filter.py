"""
Redundancy filter for GRANITE Door 2 external target candidates.

Determines whether a candidate target is recoverable from the existing
73-feature stack using per-tract ridge and GBM reconstruction tests.

Reference: M3.5 finding that syntactically independent features can be
near-monotone functional proxies within tract geography.
"""
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from granite.evaluation.recovery_baselines import (
    MIN_ADDRESSES_GBM,
    _compute_metrics,
    _fit_gbm_oof,
    _fit_ridge,
)

REDUNDANCY_THRESHOLD = 0.5


@dataclass
class RedundancyFilterResult:
    """Result from run_redundancy_filter."""
    median_ridge_r: float
    median_gbm_r: float
    per_tract: List[Dict]
    # is_admissible=True: target survives filter; use downstream
    is_admissible: bool
    # is_redundant=True: ridge or GBM reconstructed the target; reject for Door 2
    is_redundant: bool
    threshold: float
    n_tracts: int
    tract_ids_used: List[str]


def run_redundancy_filter(
    target_vector: np.ndarray,
    feature_matrix: np.ndarray,
    tract_ids: np.ndarray,
    n_tracts: int = 5,
    seed: int = 42,
    threshold: float = REDUNDANCY_THRESHOLD,
    output_dir: Optional[str] = None,
) -> RedundancyFilterResult:
    """
    Test whether target_vector is recoverable from feature_matrix.

    Selects n_tracts deterministically: sort FIPS ascending, seeded shuffle,
    take first n. Runs per-tract ridge (LOO) and GBM (5-fold OOF). Reports
    median Pearson r across selected tracts for each model.

    is_admissible = max(median_ridge_r, median_gbm_r) < threshold

    Parameters
    ----------
    target_vector : np.ndarray, shape (n_addresses,)
        External target values. NaN for unmatched/missing addresses.
    feature_matrix : np.ndarray, shape (n_addresses, n_features)
        Full feature stack without any column dropped.
    tract_ids : np.ndarray or list, shape (n_addresses,)
        Tract FIPS code for each address.
    n_tracts : int
        Number of tracts to sample for the filter.
    seed : int
        Random seed for deterministic tract shuffle.
    threshold : float
        Redundancy threshold applied to max of medians.
    output_dir : str, optional
        If provided, writes redundancy_filter.json and
        redundancy_filter_per_tract.csv here.

    Returns
    -------
    RedundancyFilterResult
    """
    rng = np.random.default_rng(seed)

    unique_fips = sorted(set(str(t) for t in tract_ids))
    shuffled = list(unique_fips)
    rng.shuffle(shuffled)
    selected_fips = shuffled[:n_tracts]

    per_tract_rows = []
    ridge_rs: List[float] = []
    gbm_rs: List[float] = []

    for fips in selected_fips:
        mask = np.array([str(t) == fips for t in tract_ids])
        n = int(mask.sum())

        X_raw = feature_matrix[mask].astype(float)
        y = target_vector[mask].astype(float)

        # exclude addresses where target is NaN
        valid = np.isfinite(y)
        n_valid = int(valid.sum())

        if n_valid < 2:
            per_tract_rows.append({
                'tract_fips': fips,
                'n_addresses': n,
                'n_valid': n_valid,
                'ridge_r': float('nan'),
                'gbm_r': float('nan'),
                'skipped': True,
            })
            continue

        X_raw = X_raw[valid]
        y = y[valid]

        # drop zero-variance columns within tract
        col_vars = np.var(X_raw, axis=0)
        nonzero_mask = col_vars > 1e-10
        X_filtered = X_raw[:, nonzero_mask]

        # per-tract z-score standardization of predictors
        col_mean = X_filtered.mean(axis=0)
        col_std = X_filtered.std(axis=0)
        col_std = np.where(col_std < 1e-10, 1.0, col_std)
        X_scaled = (X_filtered - col_mean) / col_std

        # z-score standardize target within tract for regression
        y_mean = float(np.nanmean(y))
        y_std = float(np.nanstd(y))
        if y_std < 1e-8:
            y_std = 1.0
        y_std_arr = (y - y_mean) / y_std

        # ridge LOO
        try:
            ridge_preds_std, _ = _fit_ridge(X_scaled, y_std_arr, seed=seed)
            ridge_preds_native = ridge_preds_std * y_std + y_mean
            ridge_m = _compute_metrics(ridge_preds_native, y)
            ridge_r = ridge_m['pearson_r']
        except Exception as exc:
            print(
                f'[redundancy_filter] {fips}: ridge failed: {exc}',
                file=sys.stderr,
            )
            ridge_r = float('nan')

        # gbm OOF
        if n_valid < MIN_ADDRESSES_GBM:
            gbm_r = float('nan')
        else:
            try:
                gbm_preds_std = _fit_gbm_oof(X_scaled, y_std_arr, seed=seed)
                gbm_preds_native = gbm_preds_std * y_std + y_mean
                gbm_m = _compute_metrics(gbm_preds_native, y)
                gbm_r = gbm_m['pearson_r']
            except Exception as exc:
                print(
                    f'[redundancy_filter] {fips}: GBM failed: {exc}',
                    file=sys.stderr,
                )
                gbm_r = float('nan')

        per_tract_rows.append({
            'tract_fips': fips,
            'n_addresses': n,
            'n_valid': n_valid,
            'ridge_r': ridge_r,
            'gbm_r': gbm_r,
            'skipped': False,
        })

        if np.isfinite(ridge_r):
            ridge_rs.append(ridge_r)
        if np.isfinite(gbm_r):
            gbm_rs.append(gbm_r)

    median_ridge_r = float(np.median(ridge_rs)) if ridge_rs else float('nan')
    median_gbm_r = float(np.median(gbm_rs)) if gbm_rs else float('nan')

    # nan-safe max of medians
    candidates = [v for v in (median_ridge_r, median_gbm_r) if np.isfinite(v)]
    max_median = max(candidates) if candidates else float('nan')

    if np.isfinite(max_median):
        is_admissible = bool(max_median < threshold)
    else:
        # all tracts produced NaN fits; treat as admissible but warn
        is_admissible = True
        print(
            '[redundancy_filter] warning: all per-tract fits produced NaN r; '
            'is_admissible defaulting to True',
            file=sys.stderr,
        )

    result = RedundancyFilterResult(
        median_ridge_r=median_ridge_r,
        median_gbm_r=median_gbm_r,
        per_tract=per_tract_rows,
        is_admissible=is_admissible,
        is_redundant=not is_admissible,
        threshold=threshold,
        n_tracts=n_tracts,
        tract_ids_used=selected_fips,
    )

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        meta = {
            'median_ridge_r': median_ridge_r,
            'median_gbm_r': median_gbm_r,
            'is_admissible': is_admissible,
            'is_redundant': not is_admissible,
            'threshold': threshold,
            'n_tracts': n_tracts,
            'tract_ids_used': selected_fips,
        }
        with open(os.path.join(output_dir, 'redundancy_filter.json'), 'w') as fh:
            json.dump(meta, fh, indent=2, default=str)

        pd.DataFrame(per_tract_rows).to_csv(
            os.path.join(output_dir, 'redundancy_filter_per_tract.csv'),
            index=False,
        )

    return result
