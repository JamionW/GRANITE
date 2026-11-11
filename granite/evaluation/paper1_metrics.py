"""
paper 1 metric module (preregistration v3, section 3).

pure computation: no file io, no global state. inputs are pandas objects
keyed by string geoids.

coherence: per-tract output moran's i on fixed row-standardized k-nn
weights built from address coordinates, independent of any training graph,
averaged unweighted over tracts.

agreement: block-group values are the plain mean of address values, kept
for block groups with at least MIN_BG_ADDRESSES addresses. r_T is the
correlation over all scored block groups; r_W is the correlation after
demeaning predictions and reference within each tract over that tract's
scored block groups.

duplicated tract sets (bootstrap draws): bg_predictions and
per_tract_morans_i run once on the address table, where each address appears
once. a draw is applied afterwards by repeating rows (resample_tracts) or
tract labels (run_coherence). every downstream function accepts repeated
tracts.
"""
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from granite.evaluation.spatial_diagnostics import SpatialLearningDiagnostics

MORAN_K = 8
MIN_BG_ADDRESSES = 3
ZERO_SPREAD_TOL = 1e-12   # spread treated as zero variance, relative to max(1, max |x|)

TRACT_COL = 'tract_geoid'
BG_COL = 'bg_geoid'
COORD_COLS = ('x', 'y')
METHODS = ('pearson', 'spearman')


class Paper1MetricWarning(UserWarning):
    """undefined metric (zero variance) returned as nan."""


def _warn(message):
    # importing spatial_diagnostics installs a process-wide 'ignore' filter;
    # keep this module's own category visible.
    with warnings.catch_warnings():
        warnings.simplefilter('always', Paper1MetricWarning)
        warnings.warn(message, Paper1MetricWarning, stacklevel=3)


@dataclass(frozen=True)
class CorrelationResult:
    r: float
    method: str
    n: int                     # block-group rows used
    n_dropped: int             # rows dropped for a missing reference value
    n_tracts_excluded: int = 0  # r_W only: tract labels with < 2 scored block groups
    n_rows_excluded: int = 0    # r_W only: rows in those tracts


# ---------------------------------------------------------------------------
# coherence
# ---------------------------------------------------------------------------

def tract_morans_i(coords, values, k=MORAN_K):
    """global moran's i of values under row-standardized k-nn weights on coords.

    estimator: SpatialLearningDiagnostics.compute_spatial_autocorrelation
    (libpysal KNN.from_array, transform 'r', esda.Moran permutations=0), the
    ruler behind the 5b, crossed, figure 3a/4a and posctrl values. coords are
    used as given (EPSG:4326 lon, lat for the committed sets); duplicate
    coordinates are kept, with no jitter or dedup.
    """
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2 or coords.shape[0] != values.shape[0]:
        raise ValueError(f'coords must be (n, 2) matching values; got {coords.shape}, {values.shape}')
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(coords)):
        raise ValueError('non-finite coords or values')
    if len(values) < k + 1:
        # the shared ruler returns 0.0 here; refuse instead of reporting a fake value
        raise ValueError(f'need at least k + 1 = {k + 1} addresses, got {len(values)}')
    if np.all(values == values[0]):
        _warn("zero-variance values: moran's i undefined, returning nan")
        return float('nan')
    diag = SpatialLearningDiagnostics(verbose=False)
    return float(diag.compute_spatial_autocorrelation(values, coords, k_neighbors=k))


def per_tract_morans_i(df, value_col, tract_col=TRACT_COL, coord_cols=COORD_COLS, k=MORAN_K):
    """tract_morans_i per tract; Series indexed by tract geoid (sorted)."""
    out = {}
    for tract, g in df.groupby(tract_col, sort=True):
        out[str(tract)] = tract_morans_i(g[list(coord_cols)].to_numpy(float),
                                         g[value_col].to_numpy(float), k=k)
    return pd.Series(out, name='morans_i', dtype=float).rename_axis(tract_col)


def run_coherence(per_tract, tracts=None):
    """unweighted mean of per-tract moran's i, optionally over a tract list.

    tracts may repeat (bootstrap draw); each listed entry counts once.
    raises on any nan so no tract drops silently.
    """
    s = pd.Series(per_tract, dtype=float)
    if tracts is not None:
        tracts = [str(t) for t in tracts]
        missing = sorted(set(tracts) - set(s.index.astype(str)))
        if missing:
            raise ValueError(f'tracts not in per_tract: {missing}')
        s = s.loc[tracts]
    if len(s) == 0:
        raise ValueError('no tracts')
    if s.isna().any():
        raise ValueError(f'nan moran\'s i for tracts {sorted(set(s.index[s.isna()]))}')
    return float(s.to_numpy().mean())


# ---------------------------------------------------------------------------
# block groups
# ---------------------------------------------------------------------------

def bg_predictions(df, value_col, min_addresses=MIN_BG_ADDRESSES,
                   tract_col=TRACT_COL, bg_col=BG_COL):
    """plain mean of address values per block group, n_addresses >= min_addresses.

    df holds one row per address with tract and block-group geoids. returns
    columns bg_geoid, tract_geoid, n_addresses, value (sorted by bg_geoid).
    raises if a block group spans more than one tract.
    """
    d = pd.DataFrame({
        BG_COL: df[bg_col].astype(str).to_numpy(),
        TRACT_COL: df[tract_col].astype(str).to_numpy(),
        'value': pd.to_numeric(df[value_col]).to_numpy(float),
    })
    if d[BG_COL].isna().any() or (d[BG_COL] == 'nan').any():
        raise ValueError('addresses without a block group')
    if not np.all(np.isfinite(d['value'].to_numpy())):
        raise ValueError('non-finite address values')
    n_tr = d.groupby(BG_COL)[TRACT_COL].nunique()
    if (n_tr > 1).any():
        raise ValueError(f'block groups spanning tracts: {sorted(n_tr.index[n_tr > 1])[:5]}')
    agg = d.groupby(BG_COL, sort=True).agg(
        tract_geoid=(TRACT_COL, 'first'),
        n_addresses=('value', 'size'),
        value=('value', 'mean'),
    ).reset_index()
    agg = agg[agg['n_addresses'] >= min_addresses].reset_index(drop=True)
    return agg[[BG_COL, TRACT_COL, 'n_addresses', 'value']]


def resample_tracts(frame, tracts, tract_col=TRACT_COL):
    """rows of frame for each listed tract, in list order, repeats kept."""
    tracts = [str(t) for t in tracts]
    groups = {str(k): g for k, g in frame.groupby(tract_col, sort=False)}
    missing = sorted(set(tracts) - set(groups))
    if missing:
        raise ValueError(f'tracts not in frame: {missing}')
    return pd.concat([groups[t] for t in tracts], ignore_index=True)


# ---------------------------------------------------------------------------
# correlation cores (shared by the pandas functions and fast bootstrap statistics)
# ---------------------------------------------------------------------------

def correlation(x, y, method):
    """pearson or spearman r of y against x along the last axis.

    x may be (n,) or (m, n) for m runs scored on the same rows; y is (n,).
    spearman ranks with average ranks for ties. returns nan with a warning
    when either side has zero variance.
    """
    if method not in METHODS:
        raise ValueError(f'method must be one of {METHODS}, got {method!r}')
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape[-1] != y.shape[-1] or y.ndim != 1 or x.ndim not in (1, 2):
        raise ValueError(f'shape mismatch: {x.shape} vs {y.shape}')
    if y.shape[0] < 2:
        raise ValueError('need at least 2 rows')
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError('non-finite input')
    # zero variance is judged on the raw values, before ranking, so float noise
    # left by demeaning a constant is not ranked as signal
    zero = _is_constant(x) | _is_constant(y)
    if method == 'spearman':
        x = stats.rankdata(x, method='average', axis=-1)
        y = stats.rankdata(y, method='average')
    xc = x - x.mean(axis=-1, keepdims=True)
    yc = y - y.mean()
    sxx = (xc * xc).sum(axis=-1)
    syy = (yc * yc).sum()
    sxy = (xc * yc).sum(axis=-1)
    with np.errstate(invalid='ignore', divide='ignore'):
        r = np.where(zero, np.nan, np.clip(sxy / np.sqrt(sxx * syy), -1.0, 1.0))
    if np.any(zero):
        _warn('zero variance: correlation undefined, returning nan')
    return float(r) if r.ndim == 0 else r



def _is_constant(a):
    """spread along the last axis at most ZERO_SPREAD_TOL times max(1, max |a|)."""
    scale = np.maximum(1.0, np.abs(a).max(axis=-1))
    return np.ptp(a, axis=-1) <= ZERO_SPREAD_TOL * scale


def demean_within(values, groups):
    """subtract the unweighted mean of each group along the last axis."""
    values = np.asarray(values, dtype=float)
    codes, inv = np.unique(np.asarray(groups), return_inverse=True)
    counts = np.bincount(inv, minlength=len(codes)).astype(float)
    if values.ndim == 1:
        means = np.bincount(inv, weights=values, minlength=len(codes)) / counts
        return values - means[inv]
    sums = np.zeros((values.shape[0], len(codes)))
    np.add.at(sums.T, inv, values.T)
    return values - (sums / counts)[:, inv]


# ---------------------------------------------------------------------------
# agreement
# ---------------------------------------------------------------------------

def _join_reference(bg_values, reference):
    ref = pd.Series(reference)
    ref.index = ref.index.astype(str)
    if ref.index.duplicated().any():
        raise ValueError('reference has duplicated bg_geoid')
    keys = bg_values[BG_COL].astype(str)
    t = keys.map(ref).to_numpy(float)
    keep = np.isfinite(t)
    joined = bg_values.loc[keep].reset_index(drop=True)
    return joined, t[keep], int((~keep).sum())


def pooled_r(bg_values, reference, method):
    """r_T over all scored block groups.

    bg_values: bg_predictions output (rows may repeat for duplicated tracts).
    reference: Series of reference values indexed by bg_geoid. rows with no
    finite reference value drop; their count is returned.
    """
    joined, t, n_dropped = _join_reference(bg_values, reference)
    p = joined['value'].to_numpy(float)
    r = correlation(p, t, method)
    return CorrelationResult(r=float(r), method=method, n=len(p), n_dropped=n_dropped)


def within_tract_r(bg_values, reference, method):
    """r_W: correlation after demeaning predictions and reference within tract.

    each tract is demeaned by the unweighted mean over its scored block groups,
    the same operator on both sides. tracts with fewer than 2 distinct scored
    block groups are excluded and counted. copies of a duplicated tract give
    the same demeaned values whether grouped jointly or separately.
    """
    joined, t, n_dropped = _join_reference(bg_values, reference)
    tract = joined[TRACT_COL].astype(str).to_numpy()
    n_bg = joined.groupby(TRACT_COL)[BG_COL].nunique()
    small = set(n_bg.index[n_bg < 2].astype(str))
    keep = ~np.isin(tract, list(small))
    p = joined['value'].to_numpy(float)[keep]
    t = t[keep]
    tract = tract[keep]
    if len(p) < 2:
        _warn('fewer than 2 block groups after exclusions: r_W undefined, returning nan')
        r = float('nan')
    else:
        r = correlation(demean_within(p, tract), demean_within(t, tract), method)
    return CorrelationResult(r=float(r), method=method, n=int(len(p)), n_dropped=n_dropped,
                             n_tracts_excluded=len(small), n_rows_excluded=int((~keep).sum()))


# ---------------------------------------------------------------------------
# fisher z
# ---------------------------------------------------------------------------

def _check_r(rs):
    rs = np.asarray(rs, dtype=float)
    if not np.all(np.isfinite(rs)):
        raise ValueError('non-finite r')
    if np.any(np.abs(rs) >= 1.0):
        raise ValueError('|r| = 1: fisher z undefined')
    return rs


def fisher_z_mean(rs):
    """tanh of the mean of artanh(r) over seeds."""
    rs = _check_r(rs)
    if rs.size == 0:
        raise ValueError('no r values')
    return float(np.tanh(np.arctanh(rs).mean()))


def q_difference(r1, r2):
    """cohen's q: artanh(r1) minus artanh(r2); vectorized."""
    q = np.arctanh(_check_r(r1)) - np.arctanh(_check_r(r2))
    return float(q) if np.ndim(q) == 0 else q
