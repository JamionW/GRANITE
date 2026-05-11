"""
M3 non-graph leakage baselines for the recovery framework.

Ridge regression (RidgeCV with LOO predictions via hat matrix) and
gradient-boosted regression (5-fold OOF) on the same three targets and
same n20 tracts as M2. No graph, no constraint, no cross-tract pooling.
"""
import json
import os
import traceback

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from typing import Dict, List, Optional, Tuple

from granite.disaggregation.pipeline import GRANITEPipeline
from granite.models.gnn import set_random_seed


MIN_ADDRESSES_GBM = 50

RIDGE_ALPHAS = np.logspace(-3, 3, 13)

GBM_PARAMS = dict(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42,
)


def _ridge_loo_predictions(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """
    Closed-form LOO predictions for ridge regression via the hat matrix.

    For ridge with regularization alpha:
      H = X (X'X + alpha I)^{-1} X'
      LOO residuals: e_loo[i] = (y[i] - y_hat[i]) / (1 - H[i,i])
      LOO prediction: loo_pred[i] = y[i] - e_loo[i]
    """
    # use SVD for numerical stability when n or p is large
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    d = s / (s ** 2 + alpha)
    # hat matrix diagonal: H_ii = sum_j (U_ij * s_j * d_j)^2
    H_diag = np.sum((U * (s * d)[np.newaxis, :]) ** 2, axis=1)
    # fitted values
    y_hat = X @ (Vt.T @ (d * (U.T @ y)))
    denom = 1.0 - H_diag
    # guard near-zero denominators (degenerate leverage points)
    denom = np.where(np.abs(denom) < 1e-10, np.sign(denom + 1e-20) * 1e-10, denom)
    loo_residuals = (y - y_hat) / denom
    return y - loo_residuals


def _fit_ridge(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
) -> Tuple[np.ndarray, float]:
    """
    Select alpha via 5-fold CV, then compute LOO predictions at the selected alpha.

    Returns (loo_predictions, best_alpha).
    """
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    rcv = RidgeCV(alphas=RIDGE_ALPHAS, cv=cv, fit_intercept=True)
    rcv.fit(X, y)
    best_alpha = float(rcv.alpha_)
    loo_preds = _ridge_loo_predictions(X, y, best_alpha)
    return loo_preds, best_alpha


def _fit_gbm_oof(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
) -> np.ndarray:
    """
    5-fold out-of-fold GBM predictions. Returns OOF array aligned with X rows.
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan)
    for train_idx, val_idx in kf.split(X):
        gbm_fold = GradientBoostingRegressor(**GBM_PARAMS)
        gbm_fold.fit(X[train_idx], y[train_idx])
        oof[val_idx] = gbm_fold.predict(X[val_idx])
    return oof


def _compute_metrics(pred: np.ndarray, true: np.ndarray) -> Dict:
    """Pearson r, Spearman rho, RMSE on finite pairs."""
    mask = np.isfinite(pred) & np.isfinite(true)
    if mask.sum() < 2:
        return {
            'pearson_r': float('nan'),
            'spearman_rho': float('nan'),
            'rmse_native': float('nan'),
        }
    p = pred[mask]
    t = true[mask]
    pearson_r = float(np.corrcoef(p, t)[0, 1])
    spearman_rho = float(stats.spearmanr(p, t).correlation)
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    return {'pearson_r': pearson_r, 'spearman_rho': spearman_rho, 'rmse_native': rmse}


def run_baselines(
    config: dict,
    target_feature: Optional[str] = None,
    tract_list: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    save_predictions: bool = False,
    verbose: bool = False,
    seed: int = 42,
    external_target_vector: Optional[np.ndarray] = None,
) -> List[Dict]:
    """
    Run ridge and GBM baselines for one target across all tracts in tract_list.

    Held-out feature path (target_feature is not None):
        Feature computation matches M2 exactly. Target is extracted from the
        feature matrix and the column is dropped before fitting.

    External target path (external_target_vector is not None):
        The supplied vector is used as the target directly and the full
        feature matrix is used as predictors (no column dropped).
        Rows where target is NaN (unmatched addresses) are excluded per tract.

    Exactly one of target_feature or external_target_vector must be provided.

    Feature computation matches M2 exactly (same cache path, same per-tract stacking).
    Target standardization is global across all tracts, matching M2.
    Predictor standardization is per-tract z-score.
    Zero-variance predictor columns are dropped per tract before fitting.

    Returns list of per-tract metric dicts with keys:
      tract_fips, n_addresses, n_features_used,
      ridge_pearson_r, ridge_spearman_rho, ridge_rmse_native, ridge_alpha,
      gbm_pearson_r, gbm_spearman_rho, gbm_rmse_native.
    GBM fields are None when n_addresses < MIN_ADDRESSES_GBM.
    """
    if target_feature is not None and external_target_vector is not None:
        raise ValueError(
            "run_baselines: target_feature and external_target_vector are mutually "
            "exclusive; provide exactly one."
        )
    if target_feature is None and external_target_vector is None:
        raise ValueError(
            "run_baselines: exactly one of target_feature or external_target_vector "
            "must be provided."
        )
    if tract_list is None:
        raise ValueError("run_baselines: tract_list is required.")
    if output_dir is None:
        raise ValueError("run_baselines: output_dir is required.")
    set_random_seed(seed)

    target_fips = config['data']['target_fips']
    pipeline = GRANITEPipeline(config, output_dir=output_dir)
    pipeline.verbose = verbose

    try:
        data = pipeline._load_spatial_data()
    except Exception as e:
        raise RuntimeError(f'data loading failed: {e}') from e

    # load addresses per tract, matching M2 approach
    all_addresses = []
    skipped = []
    for fips in tract_list:
        fips = str(fips).strip()
        if len(data['tracts'][data['tracts']['FIPS'] == fips]) == 0:
            skipped.append((fips, 'not in tracts GDF'))
            continue
        addrs = pipeline.data_loader.get_addresses_for_tract(fips)
        if len(addrs) == 0:
            skipped.append((fips, 'no addresses'))
            continue
        addrs['tract_fips'] = fips
        all_addresses.append(addrs)

    if skipped:
        for fips, reason in skipped:
            print(f'[m3] skipped {fips}: {reason}')

    if not all_addresses:
        raise RuntimeError('no addresses loaded for any tract')

    tract_addresses = pd.concat(all_addresses, ignore_index=True)
    valid_fips = [
        str(f).strip() for f in tract_list
        if str(f).strip() not in [s[0] for s in skipped]
    ]

    if verbose:
        print(f'[m3] {len(tract_addresses)} addresses across {len(valid_fips)} tracts')

    # compute feature matrix per tract (same per-tract cache key logic as M2)
    per_tract_arrays = []
    for fips in valid_fips:
        mask = (tract_addresses['tract_fips'] == fips).values
        tract_slice = tract_addresses[mask].copy().reset_index(drop=True)
        feats = pipeline._compute_accessibility_features(tract_slice, data)
        if feats is None:
            raise RuntimeError(f'feature computation returned None for {fips}')
        per_tract_arrays.append(feats)

    accessibility_features = np.vstack(per_tract_arrays)
    feature_names = pipeline._generate_feature_names(accessibility_features.shape[1])

    # target materialization: diverges by path
    if target_feature is not None:
        if target_feature not in feature_names:
            raise ValueError(
                f"target_feature '{target_feature}' not in feature matrix "
                f"({len(feature_names)} features). Available: {feature_names}"
            )
        feat_idx = feature_names.index(target_feature)
        raw_target_values = accessibility_features[:, feat_idx].copy().astype(float)
        feat_matrix = np.delete(accessibility_features, feat_idx, axis=1)
        _t_label = target_feature
    else:
        # external target path: validate shape and use full feature matrix
        if len(external_target_vector) != len(accessibility_features):
            raise ValueError(
                f"run_baselines: external_target_vector length {len(external_target_vector)} "
                f"does not match feature matrix rows {len(accessibility_features)}"
            )
        raw_target_values = np.array(external_target_vector, dtype=float)
        feat_matrix = accessibility_features
        _t_label = 'external_target'

    # standardize target globally across all tracts (matches M2)
    tgt_mean = float(np.nanmean(raw_target_values))
    tgt_std = float(np.nanstd(raw_target_values))
    if tgt_std < 1e-8:
        tgt_std = 1.0
    target_values_std = (raw_target_values - tgt_mean) / tgt_std

    if verbose:
        print(f'[m3] target "{_t_label}": mean={tgt_mean:.4f}, std={tgt_std:.4f}')

    os.makedirs(output_dir, exist_ok=True)

    rows = []
    all_pred_rows = []
    gbm_skipped_log = []

    for fips in valid_fips:
        mask = (tract_addresses['tract_fips'] == fips).values
        n = int(mask.sum())

        X_raw = feat_matrix[mask].astype(float)
        y_std = target_values_std[mask]
        y_native = raw_target_values[mask]

        # exclude rows with NaN target (unmatched external addresses)
        valid_target = np.isfinite(y_std)
        n_valid = int(valid_target.sum())
        if n_valid < n:
            X_raw = X_raw[valid_target]
            y_std = y_std[valid_target]
            y_native = y_native[valid_target]
            n_fit = n_valid
        else:
            n_fit = n

        # drop zero-variance columns within tract
        col_vars = np.var(X_raw, axis=0)
        nonzero_mask = col_vars > 1e-10
        n_zero_var = int((~nonzero_mask).sum())
        X_filtered = X_raw[:, nonzero_mask]
        n_features_used = int(nonzero_mask.sum())

        if n_zero_var > 0:
            print(f'[m3]   {fips}: dropped {n_zero_var} zero-variance columns, {n_features_used} remain')

        # per-tract z-score standardization of predictors
        col_mean = X_filtered.mean(axis=0)
        col_std = X_filtered.std(axis=0)
        col_std = np.where(col_std < 1e-10, 1.0, col_std)
        X_scaled = (X_filtered - col_mean) / col_std

        # ridge
        try:
            ridge_preds_std, ridge_alpha = _fit_ridge(X_scaled, y_std, seed=seed)
            ridge_preds_native = ridge_preds_std * tgt_std + tgt_mean
            ridge_m = _compute_metrics(ridge_preds_native, y_native)
        except Exception as e:
            print(f'[m3]   {fips}: ridge failed: {e}')
            ridge_preds_native = np.full(n_fit, np.nan)
            ridge_m = {'pearson_r': float('nan'), 'spearman_rho': float('nan'), 'rmse_native': float('nan')}
            ridge_alpha = float('nan')

        # gbm
        if n_fit < MIN_ADDRESSES_GBM:
            gbm_skipped_log.append((fips, n_fit))
            gbm_preds_native = np.full(n_fit, np.nan)
            gbm_m = {'pearson_r': None, 'spearman_rho': None, 'rmse_native': None}
        else:
            try:
                gbm_preds_std = _fit_gbm_oof(X_scaled, y_std, seed=seed)
                gbm_preds_native = gbm_preds_std * tgt_std + tgt_mean
                gbm_m = _compute_metrics(gbm_preds_native, y_native)
            except Exception as e:
                print(f'[m3]   {fips}: GBM failed: {e}')
                gbm_preds_native = np.full(n_fit, np.nan)
                gbm_m = {'pearson_r': float('nan'), 'spearman_rho': float('nan'), 'rmse_native': float('nan')}

        rows.append({
            'tract_fips': fips,
            'n_addresses': n,
            'n_features_used': n_features_used,
            'ridge_pearson_r': ridge_m['pearson_r'],
            'ridge_spearman_rho': ridge_m['spearman_rho'],
            'ridge_rmse_native': ridge_m['rmse_native'],
            'ridge_alpha': ridge_alpha,
            'gbm_pearson_r': gbm_m['pearson_r'],
            'gbm_spearman_rho': gbm_m['spearman_rho'],
            'gbm_rmse_native': gbm_m['rmse_native'],
        })

        if save_predictions:
            if 'address_id' in tract_addresses.columns:
                all_addr_ids = tract_addresses.loc[mask, 'address_id'].values
            else:
                all_addr_ids = tract_addresses.index[mask].values
            # save only valid-target rows when external path excludes some addresses
            valid_addr_ids = all_addr_ids[valid_target] if n_valid < n else all_addr_ids
            for i in range(n_fit):
                all_pred_rows.append({
                    'tract_fips': fips,
                    'address_id': valid_addr_ids[i],
                    'true_value': y_native[i],
                    'ridge_pred': ridge_preds_native[i],
                    'gbm_pred': gbm_preds_native[i],
                })

    if gbm_skipped_log:
        print(f'[m3] GBM skipped (n < {MIN_ADDRESSES_GBM}): '
              + ', '.join(f'{f}(n={n})' for f, n in gbm_skipped_log))

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(os.path.join(output_dir, 'per_tract_metrics.csv'), index=False)

    if save_predictions and all_pred_rows:
        pd.DataFrame(all_pred_rows).to_csv(
            os.path.join(output_dir, 'predictions.csv'), index=False
        )

    meta = {
        'target_feature': _t_label,
        'seed': seed,
        'n_tracts': len(valid_fips),
        'skipped_loading': skipped,
        'gbm_skipped_small': [(f, n) for f, n in gbm_skipped_log],
        'target_mean_native': tgt_mean,
        'target_std_native': tgt_std,
        'min_addresses_gbm': MIN_ADDRESSES_GBM,
        'ridge_alphas_searched': list(RIDGE_ALPHAS),
    }
    with open(os.path.join(output_dir, 'run_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2, default=str)

    if verbose:
        print(f'[m3] outputs written to {output_dir}')

    return rows
