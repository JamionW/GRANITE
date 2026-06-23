"""
M5 synthetic target generator for the GRANITE boundary-surface testbed.

Generates address-level synthetic target variables over the Hamilton County
n20 tract set. Supports latent and feature-based signal sources, three spatial
autocorrelation levels (weak/medium/strong), and three SNR levels.

Spatial autocorrelation is injected per-tract via Gaussian process sampling
with a Matern nu=1.5 kernel. Length scales are calibrated by binary search
to hit target Moran's I values within +/-0.05 tolerance.

Usage:
    gen = SyntheticTargetGenerator(seed=42, params={...})
    result = gen.generate()
"""

import os
import json
import random
import hashlib
import datetime
import subprocess
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.gaussian_process.kernels import Matern

warnings.filterwarnings('ignore')

# canonical paths relative to repo root
_REPO_ROOT = '/workspaces/GRANITE'
_FEATURE_CACHE_RELPATH = 'data/raw/address_features/combined_address_features.csv'
_N20_LIST_A = 'output/m2_n20_recovery/summary/n20_tract_list.txt'
_N20_LIST_B = 'data/results/m0_n20_svi_parity/per_tract.csv'
_PROJ_CRS = 'EPSG:32616'   # UTM zone 16N, Hamilton County TN
_CALIBRATION_PATH = 'data/synthetic/calibration/svi_variance_decomposition.json'
_BG_SVI_PATH = 'data/processed/national_bg_svi.csv'
_HAMILTON_FIPS_PREFIX = '47065'  # first 5 chars of 12-char BG GEOID

# autocorrelation targets (Moran's I on within-tract k-NN adjacency)
_AUTOCORR_TARGETS = {
    'weak':   {'morans_i': 0.10, 'tolerance': 0.05},
    'medium': {'morans_i': 0.40, 'tolerance': 0.05},
    'strong': {'morans_i': 0.70, 'tolerance': 0.05},
}

# noise fraction of total variance by SNR level
# sigma_noise^2 / (sigma_pre^2 + sigma_noise^2) = noise_fraction
_SNR_NOISE_FRACTION = {
    'low':    0.75,
    'medium': 0.50,
    'high':   0.25,
}

# k-NN graph parameters matching _create_geographic_fallback_graph
# use same k and max_dist for calibration and global MI to ensure consistency
_GLOBAL_K = 16
_GLOBAL_MAX_DIST_M = 1000.0
_CALIB_K = 16       # must match _GLOBAL_K for MI-length-scale relationship to transfer
_CALIB_MAX_DIST_M = 1000.0

# numeric columns to exclude from feature matrix (coordinates, identifiers, raw values)
_FEAT_EXCLUDE = frozenset({
    'number', 'postcode', 'id', 'lon', 'lat',
    'APPVALUE', 'ASSVALUE', 'LANDVALUE', 'BUILDVALUE', 'CALCACRES', 'SALE1CONSD',
    'district', 'region',  # 100% NaN in combined_address_features.csv
})


# ---------------------------------------------------------------------------
# path helpers
# ---------------------------------------------------------------------------

def _abspath(relpath):
    return os.path.join(_REPO_ROOT, relpath)


def _stable_hash(s):
    """deterministic hash for a string, not subject to PYTHONHASHSEED."""
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


# ---------------------------------------------------------------------------
# tract list resolution
# ---------------------------------------------------------------------------

def _resolve_tract_list(source):
    if source != 'auto':
        if not os.path.exists(source):
            raise FileNotFoundError(f"tract_list_source path not found: {source}")
        with open(source) as fh:
            return [line.strip() for line in fh if line.strip()]

    path_a = _abspath(_N20_LIST_A)
    if os.path.exists(path_a):
        with open(path_a) as fh:
            return [line.strip() for line in fh if line.strip()]

    path_b = _abspath(_N20_LIST_B)
    if os.path.exists(path_b):
        df = pd.read_csv(path_b, dtype={'fips': str})
        return df['fips'].unique().tolist()

    raise FileNotFoundError(
        "no tract list found at:\n"
        f"  {path_a}\n"
        f"  {path_b}\n"
        "generate n20 outputs first, or pass an explicit tract_list_source path."
    )


# ---------------------------------------------------------------------------
# address loading
# ---------------------------------------------------------------------------

def _load_addresses(tract_list, feat_df):
    """
    Build a GeoDataFrame of addresses for the n20 tracts.

    Uses lon/lat from the feature CSV plus a spatial join against tract
    boundaries. Returns GeoDataFrame with columns:
        hash, fips, geometry (projected to _PROJ_CRS), x, y
    """
    from granite.data.loaders import DataLoader

    loader = DataLoader()
    tracts_gdf = loader.load_census_tracts('47', '065')
    tracts_n20 = tracts_gdf[tracts_gdf['FIPS'].isin(tract_list)][['FIPS', 'geometry']].copy()

    if len(tracts_n20) == 0:
        raise ValueError(
            f"no census tracts matched for {len(tract_list)} FIPS codes; "
            "check that load_census_tracts returns the expected FIPS column."
        )

    # build point GDF from feature CSV lon/lat
    valid = feat_df[feat_df['lon'].notna() & feat_df['lat'].notna()].copy()
    gdf = gpd.GeoDataFrame(
        valid[['hash']].copy(),
        geometry=gpd.points_from_xy(valid['lon'], valid['lat']),
        crs='EPSG:4326',
    )

    # spatial join to assign tract FIPS
    tracts_4326 = tracts_n20.to_crs('EPSG:4326')
    joined = gpd.sjoin(gdf, tracts_4326, how='inner', predicate='within')
    joined = joined.rename(columns={'FIPS': 'fips'})
    joined = joined[['hash', 'fips', 'geometry']].drop_duplicates('hash').copy()

    if len(joined) == 0:
        raise ValueError(
            "spatial join returned no addresses; check tract geometries and "
            "feature CSV lon/lat range."
        )

    # project to UTM for metric distance calculations
    joined = joined.to_crs(_PROJ_CRS)
    joined['x'] = joined.geometry.x
    joined['y'] = joined.geometry.y
    joined = joined.reset_index(drop=True)

    return joined


# ---------------------------------------------------------------------------
# spatial weight matrix
# ---------------------------------------------------------------------------

def _build_knn_weights(xy, k=8, max_dist_m=1000.0):
    """
    Build a symmetrized k-NN weight matrix in COO format.

    Returns (rows, cols, weights) as numpy int/float arrays.
    Weights use the same exponential decay as _create_geographic_fallback_graph.
    """
    n = len(xy)
    if n < 2:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float)

    k_actual = min(k + 1, n)
    nbrs = NearestNeighbors(n_neighbors=k_actual, metric='euclidean').fit(xy)
    distances, indices = nbrs.kneighbors(xy)

    # collect directed edges
    edge_dict = {}
    for i in range(n):
        for j_pos in range(1, k_actual):
            j = indices[i, j_pos]
            d = distances[i, j_pos]
            if d > max_dist_m:
                continue
            w = np.exp(-d / 300.0)
            key = (min(i, j), max(i, j))
            if key in edge_dict:
                edge_dict[key] = (edge_dict[key] + w) / 2.0
            else:
                edge_dict[key] = w

    if not edge_dict:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float)

    rows, cols, weights = [], [], []
    for (i, j), w in edge_dict.items():
        rows.extend([i, j])
        cols.extend([j, i])
        weights.extend([w, w])

    return np.array(rows, dtype=int), np.array(cols, dtype=int), np.array(weights, dtype=float)


# ---------------------------------------------------------------------------
# Moran's I
# ---------------------------------------------------------------------------

def _morans_i(values, rows, cols, weights):
    """
    Compute Moran's I given variable values and a COO weight matrix.
    Returns 0.0 if weights or variance are degenerate.
    """
    n = len(values)
    if len(weights) == 0:
        return 0.0

    xbar = np.mean(values)
    z = values - xbar
    w_sum = np.sum(weights)
    if w_sum == 0:
        return 0.0

    denom = np.sum(z ** 2)
    if denom == 0:
        return 0.0

    numer = np.sum(weights * z[rows] * z[cols])
    return float((n / w_sum) * (numer / denom))


# ---------------------------------------------------------------------------
# Gaussian process sampling
# ---------------------------------------------------------------------------

def _sample_gp_residual(xy, length_scale, rng):
    """
    Draw a zero-mean GP sample at positions xy using Matern nu=1.5 kernel.

    Uses Cholesky factorization with a small jitter for stability.
    Falls back to eigendecomposition if Cholesky fails.

    Returns array of shape (n,).
    """
    n = len(xy)
    kernel = Matern(length_scale=length_scale, nu=1.5)
    K = kernel(xy)
    K += np.eye(n) * 1e-6

    try:
        L = np.linalg.cholesky(K)
        z = rng.standard_normal(n)
        return L @ z
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(K)
        eigvals = np.maximum(eigvals, 1e-8)
        return eigvecs @ (np.sqrt(eigvals) * rng.standard_normal(n))


# ---------------------------------------------------------------------------
# length-scale calibration
# ---------------------------------------------------------------------------

def _calibrate_length_scale(
    xy, target_morans_i, rng_seed,
    max_cal_points=200, n_samples=1, n_bisect=10
):
    """
    Find a Matern nu=1.5 GP length scale that produces a within-tract sample
    whose Moran's I is close to target_morans_i.

    Subsamples xy to at most max_cal_points for speed.
    Returns (length_scale_meters, achieved_morans_i_on_subsample).
    """
    n = len(xy)

    # subsample for speed
    if n > max_cal_points:
        cal_rng = np.random.default_rng(rng_seed + 99999)
        idx = cal_rng.choice(n, size=max_cal_points, replace=False)
        idx.sort()
        xy_cal = xy[idx]
    else:
        xy_cal = xy

    rows, cols, weights = _build_knn_weights(xy_cal, k=_CALIB_K, max_dist_m=_CALIB_MAX_DIST_M)

    if len(rows) == 0:
        # no edges; spatial structure cannot be calibrated
        return 500.0, 0.0

    def estimate_mi(ls):
        values_list = []
        for s_idx in range(n_samples):
            rng = np.random.default_rng(rng_seed + s_idx * 31)
            r = _sample_gp_residual(xy_cal, ls, rng)
            values_list.append(_morans_i(r, rows, cols, weights))
        return float(np.mean(values_list))

    # coarse scan to bracket the target.
    # hamilton county n20 addresses average ~90m apart; MI saturates around ls=200m.
    # the interesting length-scale range is 5-200m.
    ls_grid = [5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0,
               400.0, 800.0, 2000.0, 6000.0]
    mi_grid = [estimate_mi(ls) for ls in ls_grid]

    lo_ls, hi_ls = None, None
    for i in range(len(ls_grid) - 1):
        if mi_grid[i] <= target_morans_i <= mi_grid[i + 1]:
            lo_ls, hi_ls = ls_grid[i], ls_grid[i + 1]
            break

    if lo_ls is None:
        # target outside the scanned range; clamp to nearest extreme
        best_ls = ls_grid[-1] if mi_grid[-1] < target_morans_i else ls_grid[0]
        final_rng = np.random.default_rng(rng_seed)
        r = _sample_gp_residual(xy_cal, best_ls, final_rng)
        return best_ls, _morans_i(r, rows, cols, weights)

    # binary search
    for _ in range(n_bisect):
        mid = (lo_ls + hi_ls) / 2.0
        mi_mid = estimate_mi(mid)
        if mi_mid < target_morans_i:
            lo_ls = mid
        else:
            hi_ls = mid
        if (hi_ls - lo_ls) / (lo_ls + 1.0) < 0.01:
            break

    best_ls = (lo_ls + hi_ls) / 2.0
    final_rng = np.random.default_rng(rng_seed)
    r = _sample_gp_residual(xy_cal, best_ls, final_rng)
    return best_ls, _morans_i(r, rows, cols, weights)


# ---------------------------------------------------------------------------
# SVI variance decomposition calibration
# ---------------------------------------------------------------------------

def compute_svi_variance_decomposition():
    """
    Compute between-tract and within-tract SVI variance for the n20 Hamilton
    County tracts, weighted by block-group population.

    Weighting note: population weighting used here (each BG contributes
    proportionally to its census population). Address-count weighting is an
    alternative that weights by number of parcels in the BG; population
    weighting is preferred for SVI calibration because SVI itself is
    population-based.

    Saves result to data/synthetic/calibration/svi_variance_decomposition.json.
    Returns the dict that was saved.
    """
    n20_path = _abspath(_N20_LIST_A)
    if not os.path.exists(n20_path):
        raise FileNotFoundError(
            f"n20 tract list not found at {_N20_LIST_A}; "
            "generate m2 outputs first."
        )
    with open(n20_path) as fh:
        n20_tracts = set(line.strip() for line in fh if line.strip())

    bg_svi_path = _abspath(_BG_SVI_PATH)
    if not os.path.exists(bg_svi_path):
        raise FileNotFoundError(
            f"national BG SVI not found at {_BG_SVI_PATH}; "
            "run BlockGroupLoader.get_block_groups_with_demographics() first."
        )

    df = pd.read_csv(bg_svi_path, dtype={'GEOID': str})
    # filter to Hamilton County block groups whose tract is in the n20 set
    # GEOID is 12-char: 2 state + 3 county + 6 tract + 1 BG
    # first 5 chars = county FIPS, first 11 chars = tract FIPS
    df = df[df['GEOID'].str.startswith(_HAMILTON_FIPS_PREFIX)].copy()
    df['tract_fips'] = df['GEOID'].str[:11]
    df = df[df['tract_fips'].isin(n20_tracts)].copy()
    df = df[df['SVI'].notna() & df['population'].notna()].copy()

    if len(df) == 0:
        raise ValueError(
            "no BG rows found for n20 tracts in Hamilton County; "
            "check that national_bg_svi.csv has GEOID starting with 47065."
        )

    # per-tract population-weighted mean SVI
    def pop_wtd_mean(sub):
        w = sub['population'].values.astype(float)
        if w.sum() == 0:
            return float(sub['SVI'].mean())
        return float(np.average(sub['SVI'].values, weights=w))

    tract_stats = []
    for tract_fips, grp in df.groupby('tract_fips'):
        tract_pop = float(grp['population'].sum())
        tract_mean_svi = pop_wtd_mean(grp)
        # within-tract variance: population-weighted variance over BGs
        w = grp['population'].values.astype(float)
        svi_vals = grp['SVI'].values
        if w.sum() > 0:
            wtd_var = float(np.average((svi_vals - tract_mean_svi) ** 2, weights=w))
        else:
            wtd_var = float(np.var(svi_vals))
        tract_stats.append({
            'tract_fips': tract_fips,
            'tract_pop': tract_pop,
            'tract_mean_svi': tract_mean_svi,
            'within_var': wtd_var,
        })

    stats_df = pd.DataFrame(tract_stats)
    total_pop = float(stats_df['tract_pop'].sum())
    tract_weights = stats_df['tract_pop'].values / total_pop

    # between-tract variance: population-weighted variance of tract means
    grand_mean = float(np.average(stats_df['tract_mean_svi'].values, weights=tract_weights))
    between_var = float(np.average(
        (stats_df['tract_mean_svi'].values - grand_mean) ** 2,
        weights=tract_weights,
    ))

    # within-tract variance: population-weighted mean of per-tract within-variances
    within_var = float(np.average(stats_df['within_var'].values, weights=tract_weights))

    total_var = between_var + within_var
    ratio_between = between_var / total_var if total_var > 0 else 0.0

    result = {
        'source': 'national_bg_svi',
        'weighting': 'population',
        'n_tracts': int(len(stats_df)),
        'n_bgs': int(len(df)),
        'between_var': between_var,
        'within_var': within_var,
        'total_var': total_var,
        'ratio_between': ratio_between,
        'computed_at': datetime.datetime.utcnow().isoformat() + 'Z',
    }

    out_path = _abspath(_CALIBRATION_PATH)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as fh:
        json.dump(result, fh, indent=2)
    print(f"  saved SVI variance decomposition to {_CALIBRATION_PATH}")
    return result


# ---------------------------------------------------------------------------
# main generator class
# ---------------------------------------------------------------------------

class SyntheticTargetGenerator:
    """
    Generates synthetic address-level target variables for the M5 testbed.

    params keys:
        signal_source:          'latent' (default) or 'features'
        feature_indices:        list[int], used when signal_source='features'
        signal_type:            'linear', 'nonlinear', 'interaction'
        spatial_autocorrelation:'weak', 'medium', or 'strong'
        snr:                    'low', 'medium', or 'high'
        tract_list_source:      path string or 'auto'
    """

    def __init__(self, seed: int, params: dict):
        self.seed = seed
        self.params = params
        cal_path = _abspath(_CALIBRATION_PATH)
        if not os.path.exists(cal_path):
            raise FileNotFoundError(
                f"calibration file not found at {_CALIBRATION_PATH}; "
                "run compute_svi_variance_decomposition() first to generate it."
            )
        with open(cal_path) as fh:
            self._calibration = json.load(fh)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def generate(self) -> dict:
        """
        Run the full synthetic generation pipeline.

        Returns dict with keys:
            addresses, tract_means, diagnostics, params, seed, output_dir
        """
        params = self.params
        seed = self.seed

        # fix all random seeds at the top for full reproducibility
        np.random.seed(seed)
        random.seed(seed)
        try:
            import torch as _torch
            _torch.manual_seed(seed)
        except ImportError:
            pass
        rng = np.random.default_rng(seed)

        signal_source = params.get('signal_source', 'latent')
        signal_type = params.get('signal_type', 'linear')
        autocorr = params.get('spatial_autocorrelation', 'medium')
        snr_level = params.get('snr', 'medium')
        tract_list_source = params.get('tract_list_source', 'auto')
        feature_indices = params.get('feature_indices', [0, 1, 2])

        if autocorr not in _AUTOCORR_TARGETS:
            raise ValueError(f"unknown spatial_autocorrelation '{autocorr}'; "
                             f"choose from {list(_AUTOCORR_TARGETS)}")
        if snr_level not in _SNR_NOISE_FRACTION:
            raise ValueError(f"unknown snr '{snr_level}'; "
                             f"choose from {list(_SNR_NOISE_FRACTION)}")

        # resolve tract list
        tract_list = _resolve_tract_list(tract_list_source)
        print(f"  tract list: {len(tract_list)} tracts from '{tract_list_source}'")

        # load feature CSV (needed for addresses and for mode='features')
        feat_cache = _abspath(_FEATURE_CACHE_RELPATH)
        if not os.path.exists(feat_cache):
            raise FileNotFoundError(
                f"feature cache not found at {_FEATURE_CACHE_RELPATH}; "
                "run granite/scripts/run_granite.py first to generate cached features"
            )
        feat_df = pd.read_csv(feat_cache, dtype={'hash': str})

        # load address coordinates + tract assignments
        print(f"  loading addresses for {len(tract_list)} tracts ...")
        addr_gdf = _load_addresses(tract_list, feat_df)
        n_total = len(addr_gdf)
        print(f"  loaded {n_total} addresses across "
              f"{addr_gdf['fips'].nunique()} tracts")

        xy = addr_gdf[['x', 'y']].values.astype(float)

        # build signal
        print(f"  building signal: source={signal_source}, type={signal_type}")
        s = self._build_signal(
            signal_source, signal_type, feat_df, addr_gdf, feature_indices, rng
        )
        s_std = float(np.std(s))
        if s_std > 0:
            s = (s - float(np.mean(s))) / s_std

        # inject per-tract GP autocorrelation
        target_mi = _AUTOCORR_TARGETS[autocorr]['morans_i']
        print(f"  injecting GP autocorrelation: level={autocorr}, "
              f"target Moran's I={target_mi}")
        r, ls_per_tract = self._inject_autocorrelation(autocorr, addr_gdf, xy)
        r_std = float(np.std(r))
        if r_std > 0:
            r = (r - float(np.mean(r))) / r_std

        # combine signal + residual, normalize to unit variance
        y_pre = s + r
        y_pre_std = float(np.std(y_pre))
        if y_pre_std > 0:
            y_pre = (y_pre - float(np.mean(y_pre))) / y_pre_std

        # add noise according to SNR
        noise_fraction = _SNR_NOISE_FRACTION[snr_level]
        # solve: sigma_noise^2 / (var(y_pre) + sigma_noise^2) = noise_fraction
        # var(y_pre) ~ 1 after normalization
        sigma_noise_sq = noise_fraction / max(1.0 - noise_fraction, 1e-9)
        sigma_noise = float(np.sqrt(sigma_noise_sq))
        eps = rng.normal(0.0, sigma_noise, n_total)
        y_within = y_pre + eps

        # inject between-tract effect calibrated to real SVI variance decomposition
        wtvr_target = 1.0 - float(self._calibration['ratio_between'])
        y_true, sigma_between, tract_effect_var = self._inject_tract_effect(
            y_within, addr_gdf, wtvr_target, rng
        )

        # tract means (address-count-weighted)
        tract_means = {}
        for fips, grp in addr_gdf.groupby('fips'):
            idx = grp.index.tolist()
            tract_means[fips] = float(np.mean(y_true[idx]))

        # within-tract variance ratio
        wtvr = self._within_tract_variance(y_true, addr_gdf)
        total_var = float(np.var(y_true))
        wtvr_ratio = wtvr / total_var if total_var > 0 else 0.0

        if wtvr_ratio < 0.05:
            raise ValueError(
                f"within-tract variance ratio {wtvr_ratio:.4f} < minimum 0.05. "
                f"params: {self.params}"
            )

        if wtvr_ratio > 0.95:
            raise ValueError(
                f"within-tract variance ratio {wtvr_ratio:.4f} > maximum 0.95; "
                f"between-tract variance too low, tract-mean constraint is uninformative. "
                f"params: {params}"
            )

        # global Moran's I on full n20 address set, computed on the GP residuals.
        # the residuals are what the per-tract calibration targets; computing on y_true
        # would always reduce MI below target due to noise and uncorrelated signal mixing.
        # morans_i_achieved reflects the spatial structure actually injected.
        print(f"  computing global Moran's I ({n_total} addresses) ...")
        g_rows, g_cols, g_weights = _build_knn_weights(
            xy, k=_GLOBAL_K, max_dist_m=_GLOBAL_MAX_DIST_M
        )
        achieved_mi = _morans_i(r, g_rows, g_cols, g_weights)
        # also record MI of final y_true for reference
        mi_y_true = _morans_i(y_true, g_rows, g_cols, g_weights)

        # variance decomposition
        signal_var = float(np.var(s))
        residual_var = float(np.var(r))
        noise_var = float(sigma_noise_sq)

        diagnostics = {
            'morans_i_achieved': achieved_mi,    # MI of GP residuals (calibrated component)
            'morans_i_target': target_mi,
            'morans_i_y_true': mi_y_true,        # MI of final y_true (for reference)
            'wtvr_achieved': wtvr_ratio,
            'wtvr_target': wtvr_target,
            'sigma_between': float(sigma_between),
            'tract_effect_variance': float(tract_effect_var),
            'signal_variance': signal_var,
            'residual_variance': residual_var,
            'noise_variance': noise_var,
            'length_scale_per_tract': {k: float(v) for k, v in ls_per_tract.items()},
            'n_addresses': n_total,
            'n_tracts': len(tract_means),
        }

        # assemble output DataFrame
        out_df = pd.DataFrame({
            'fips': addr_gdf['fips'].values,
            'address_hash': addr_gdf['hash'].values,
            'x': xy[:, 0],
            'y': xy[:, 1],
            'y_true': y_true,
        })

        # save to disk
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        out_dir = _abspath(f'data/synthetic/run_{timestamp}')
        self._save_outputs(
            out_df, diagnostics, out_dir, timestamp,
            s, r, eps, xy, params
        )

        return {
            'addresses': out_df,
            'tract_means': tract_means,
            'diagnostics': diagnostics,
            'params': self.params,
            'seed': self.seed,
            'output_dir': out_dir,
        }

    # ------------------------------------------------------------------
    # signal construction
    # ------------------------------------------------------------------

    def _build_signal(self, signal_source, signal_type, feat_df,
                      addr_gdf, feature_indices, rng):
        n = len(addr_gdf)

        if signal_source == 'latent':
            z = rng.standard_normal(n)
            w1 = float(rng.standard_normal())
            w2 = float(rng.standard_normal())
            b = float(rng.standard_normal())

            if signal_type == 'linear':
                return w1 * z + b
            elif signal_type == 'nonlinear':
                return np.tanh(w1 * z) + w2 * z ** 2
            elif signal_type == 'interaction':
                z2 = rng.standard_normal(n)
                return w1 * z * z2 + w2 * (z + z2)
            else:
                raise ValueError(f"unknown signal_type '{signal_type}'")

        elif signal_source == 'features':
            # select numeric columns, excluding coordinates and raw value columns
            numeric_cols = feat_df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [c for c in numeric_cols if c not in _FEAT_EXCLUDE]
            # drop columns that are >99% NaN (provides no signal)
            nan_frac = feat_df[numeric_cols].isna().mean()
            numeric_cols = [c for c in numeric_cols if nan_frac[c] <= 0.99]

            max_idx = max(feature_indices)
            if max_idx >= len(numeric_cols):
                raise IndexError(
                    f"feature_indices max={max_idx} out of range; "
                    f"only {len(numeric_cols)} numeric feature columns available: "
                    f"{numeric_cols}"
                )

            # left-join features onto addr_gdf by hash, preserving row order
            feat_sub = feat_df[['hash'] + numeric_cols].copy()
            merged = addr_gdf[['hash']].merge(feat_sub, on='hash', how='left')

            selected_cols = [numeric_cols[i] for i in feature_indices]
            X_sub = merged[selected_cols].values.astype(float)

            # fill NaN with column means, then standardize
            for col_i in range(X_sub.shape[1]):
                mask = np.isnan(X_sub[:, col_i])
                if mask.any():
                    col_mean = float(np.nanmean(X_sub[:, col_i]))
                    X_sub[mask, col_i] = col_mean
                col_std = float(np.std(X_sub[:, col_i]))
                if col_std > 0:
                    X_sub[:, col_i] = (
                        (X_sub[:, col_i] - float(np.mean(X_sub[:, col_i]))) / col_std
                    )

            n_cols = X_sub.shape[1]

            if signal_type == 'linear':
                w = rng.standard_normal(n_cols)
                return X_sub @ w

            elif signal_type == 'nonlinear':
                w = rng.standard_normal(n_cols)
                return np.tanh(X_sub @ w)

            elif signal_type == 'interaction':
                if n_cols < 2:
                    raise ValueError(
                        "signal_type='interaction' requires at least 2 feature_indices"
                    )
                pairs = [(i, j) for i in range(n_cols) for j in range(i + 1, n_cols)]
                v = rng.standard_normal(len(pairs))
                s = np.zeros(n)
                for k_idx, (i, j) in enumerate(pairs):
                    s += v[k_idx] * X_sub[:, i] * X_sub[:, j]
                return s

            else:
                raise ValueError(f"unknown signal_type '{signal_type}'")

        else:
            raise ValueError(f"unknown signal_source '{signal_source}'; "
                             "choose 'latent' or 'features'")

    # ------------------------------------------------------------------
    # autocorrelation injection
    # ------------------------------------------------------------------

    def _inject_autocorrelation(self, autocorr, addr_gdf, xy):
        target_mi = _AUTOCORR_TARGETS[autocorr]['morans_i']
        n_total = len(addr_gdf)
        residual = np.zeros(n_total)
        ls_per_tract = {}

        for fips, grp in addr_gdf.groupby('fips'):
            idx = grp.index.tolist()
            xy_tract = xy[idx]
            n_tract = len(idx)

            if n_tract < 4:
                # not enough points for meaningful GP; use white noise
                tract_rng = np.random.default_rng(
                    self.seed + _stable_hash(fips) % 1_000_000
                )
                residual[idx] = tract_rng.standard_normal(n_tract)
                ls_per_tract[fips] = 0.0
                continue

            # calibrate length scale using the full tract (no subsampling) so that
            # calibrated MI reflects the actual density of the n20 address set.
            # subsampling at lower density requires a larger length scale that overshoots
            # when applied to the full, denser set.
            cal_seed = self.seed + _stable_hash(fips) % 1_000_000
            ls, _cal_mi = _calibrate_length_scale(
                xy_tract, target_mi, rng_seed=cal_seed,
                max_cal_points=n_tract, n_samples=1, n_bisect=12
            )
            ls_per_tract[fips] = ls

            # sample full GP with calibrated length scale (different seed)
            sample_seed = self.seed + _stable_hash(fips) % 1_000_000 + 1_000_000
            tract_rng = np.random.default_rng(sample_seed)
            r_tract = _sample_gp_residual(xy_tract, ls, tract_rng)

            # normalize within tract to zero mean, unit variance.
            # subtracting the mean is critical: GP samples have non-zero mean by
            # chance; leaving it creates spurious between-tract contrast that inflates
            # the global Moran's I beyond the per-tract calibration target.
            r_tract = r_tract - float(np.mean(r_tract))
            std_r = float(np.std(r_tract))
            if std_r > 0:
                r_tract = r_tract / std_r

            residual[idx] = r_tract

        return residual, ls_per_tract

    # ------------------------------------------------------------------
    # variance diagnostics
    # ------------------------------------------------------------------

    def _within_tract_variance(self, y_true, addr_gdf):
        """weighted average of within-tract variance (weights = tract size / total)."""
        n_total = len(y_true)
        weighted_var = 0.0
        for fips, grp in addr_gdf.groupby('fips'):
            idx = grp.index.tolist()
            n_tract = len(idx)
            wt = n_tract / n_total
            weighted_var += wt * float(np.var(y_true[idx]))
        return weighted_var

    def _inject_tract_effect(self, y_within, addr_gdf, wtvr_target, rng):
        """
        Add a between-tract random effect so the address-count-weighted
        within-tract variance ratio equals wtvr_target.

        Variance accounting (address-count weighting):
            W  = current within-tract variance of y_within
            B0 = current between-tract variance of y_within
            B_target = W * (1 / wtvr_target - 1)
            sigma_between = sqrt(max(B_target - B0, 0))

        Draws one N(0,1) effect per unique tract, centers to mean 0, scales
        empirical std to exactly sigma_between, then broadcasts to addresses.

        Returns:
            (y_within + tract_effect_vector, sigma_between,
             float(np.var(tract_effect_per_tract)))
        """
        n_total = len(y_within)

        # collect per-tract address-count statistics
        tract_info = {}
        for fips, grp in addr_gdf.groupby('fips'):
            idx = grp.index.tolist()
            n_t = len(idx)
            tract_info[fips] = {
                'idx': idx,
                'n': n_t,
                'mean': float(np.mean(y_within[idx])),
            }

        # address-count weighted within-tract variance
        w_var = sum(
            (info['n'] / n_total) * float(np.var(y_within[info['idx']]))
            for info in tract_info.values()
        )

        # address-count weighted between-tract variance
        grand_mean = float(np.mean(y_within))
        b0_var = sum(
            (info['n'] / n_total) * (info['mean'] - grand_mean) ** 2
            for info in tract_info.values()
        )

        # solve for required additional between-tract std
        b_target = w_var * (1.0 / max(wtvr_target, 1e-9) - 1.0)
        sigma_between = float(np.sqrt(max(b_target - b0_var, 0.0)))

        fips_list = list(tract_info.keys())
        n_tracts = len(fips_list)

        # draw, center, scale per-tract raw effects
        raw_effects = rng.standard_normal(n_tracts)
        raw_effects = raw_effects - float(np.mean(raw_effects))
        raw_std = float(np.std(raw_effects))
        if sigma_between > 0 and raw_std > 0:
            raw_effects = raw_effects * (sigma_between / raw_std)

        # broadcast to address vector
        tract_effect_vector = np.zeros(n_total)
        for i, fips in enumerate(fips_list):
            for j in tract_info[fips]['idx']:
                tract_effect_vector[j] = raw_effects[i]

        return (
            y_within + tract_effect_vector,
            sigma_between,
            float(np.var(raw_effects)),
        )

    def _git_commit(self):
        """return current HEAD commit hash, or 'unknown' on any error."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True, text=True, cwd=_REPO_ROOT, timeout=5,
            )
            return result.stdout.strip() if result.returncode == 0 else 'unknown'
        except Exception:
            return 'unknown'

    # ------------------------------------------------------------------
    # output persistence
    # ------------------------------------------------------------------

    def _save_outputs(self, out_df, diagnostics, out_dir, timestamp,
                      s, r, eps, xy, params):
        fig_dir = os.path.join(out_dir, 'figures')
        os.makedirs(fig_dir, exist_ok=True)

        # addresses CSV
        out_df.to_csv(os.path.join(out_dir, 'addresses.csv'), index=False)

        # metadata.json -- records input params, diagnostics, and address_id_column
        # the canonical address identifier column in the raw data is 'hash';
        # we rename it to 'address_hash' in the output CSV.
        metadata = {
            'timestamp': timestamp,
            'seed': self.seed,
            'generator_commit': self._git_commit(),
            'params': params,
            'diagnostics': diagnostics,
            'address_id_column': 'address_hash',
            'raw_address_id_column': 'hash',
            'feature_cache_path': _FEATURE_CACHE_RELPATH,
            'proj_crs': _PROJ_CRS,
            'n20_tract_list_source': _N20_LIST_A,
        }
        with open(os.path.join(out_dir, 'metadata.json'), 'w') as fh:
            json.dump(metadata, fh, indent=2)

        y_true = out_df['y_true'].values
        y_pre = s + r  # pre-noise composite

        # scatter: pre-noise signal vs truth
        fig, ax = plt.subplots(figsize=(6, 5))
        n_plot = min(10000, len(y_pre))
        rng_plot = np.random.default_rng(self.seed + 777)
        plot_idx = rng_plot.choice(len(y_pre), size=n_plot, replace=False)
        r_val = float(np.corrcoef(y_pre[plot_idx], y_true[plot_idx])[0, 1])
        ax.scatter(y_pre[plot_idx], y_true[plot_idx],
                   alpha=0.15, s=2, rasterized=True, color='steelblue')
        ax.set_xlabel('signal + residual (pre-noise)')
        ax.set_ylabel('y_true')
        ax.set_title(f"signal vs truth  (r={r_val:.3f})")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, 'scatter_signal_vs_truth.png'), dpi=120)
        plt.close(fig)

        # spatial heatmap
        vlo = float(np.percentile(y_true, 2))
        vhi = float(np.percentile(y_true, 98))
        fig, ax = plt.subplots(figsize=(8, 7))
        sc = ax.scatter(
            xy[:, 0], xy[:, 1], c=y_true, s=1,
            cmap='RdYlBu_r', vmin=vlo, vmax=vhi, rasterized=True
        )
        plt.colorbar(sc, ax=ax, label='y_true')
        ax.set_title('spatial heatmap of y_true')
        ax.set_xlabel('easting (m)')
        ax.set_ylabel('northing (m)')
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, 'spatial_heatmap.png'), dpi=120)
        plt.close(fig)

        # Moran's I validation text
        achieved = float(diagnostics['morans_i_achieved'])
        target = float(diagnostics['morans_i_target'])
        in_band = abs(achieved - target) <= 0.05
        with open(os.path.join(fig_dir, 'morans_i_validation.txt'), 'w') as fh:
            fh.write("Moran's I validation\n")
            fh.write("=" * 40 + "\n")
            fh.write(f"achieved : {achieved:.4f}\n")
            fh.write(f"target   : {target:.4f}\n")
            fh.write(f"tolerance: +/- 0.05\n")
            fh.write(f"in_band  : {'YES' if in_band else 'NO'}\n")
