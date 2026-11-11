"""
paper 1 inference engine (preregistration v3, sections 5 and 6).

tract-clustered bootstrap, leave-one-tract-out jackknife, decision helpers,
and the within-tract permutation null. randomness comes only from a
numpy.random.Generator built from the explicit seed.

pairing: tract_bootstrap calls statistic once per draw with the sampled tract
list, and that one call scores every run in every condition on the same
sampled block groups. paired contrasts must be computed inside statistic.

nothing here picks between bootstrap and jackknife; compare_verdicts reports
both (section 6).

resampling indices follow scripts/paper1_bootstrap.py (BOOT-P1):
default_rng(seed).integers(0, n_tracts, size=(n_draws, n_tracts)).
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

BOOTSTRAP_DRAWS = 10_000
QUANTILE_METHOD = 'linear'
JACKKNIFE_Z = 1.645


@dataclass(frozen=True)
class BootstrapResult:
    estimate: object            # statistic on the full tract set (float or 1-d array)
    draws: np.ndarray           # (n_draws,) or (n_draws, k)
    seed: int
    n_draws: int
    numpy_version: str
    quantile_method: str
    n_nonfinite: int            # draws with any non-finite component
    tract_index: np.ndarray = field(repr=False)  # (n_draws, n_tracts) positions into tracts


@dataclass(frozen=True)
class Interval:
    lower: object
    upper: object               # +inf for a lower one-sided bound
    sided: str
    method: str                 # 'percentile' or 'jackknife-normal'
    level: float = None
    z: float = None
    quantile_method: str = None
    quantiles: tuple = None     # quantile levels used (percentile intervals)


@dataclass(frozen=True)
class JackknifeResult:
    estimate: object
    loo: np.ndarray             # (n,) or (n, k), leave-one-tract-out values in tract order
    se: object
    n: int


@dataclass(frozen=True)
class TostDecision:
    verdict: str                # 'equivalent', 'different' or 'inconclusive'
    margin: float
    tightest_margin: float      # max(|lo|, |hi|)


@dataclass(frozen=True)
class PermutationNullResult:
    observed: float
    null: np.ndarray
    p95: float
    exceeds: bool               # observed > p95
    seed: int
    n_perm: int
    quantile_method: str


def _as_stat(value, shape=None):
    a = np.asarray(value, dtype=float)
    if a.ndim > 1:
        raise ValueError(f'statistic must return a scalar or 1-d vector, got shape {a.shape}')
    if shape is not None and a.shape != shape:
        raise ValueError(f'statistic changed shape: {shape} -> {a.shape}')
    return a


def _scalar_or_array(a):
    return float(a) if a.ndim == 0 else a


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------

def tract_bootstrap(tracts, statistic, n_draws=BOOTSTRAP_DRAWS, *, seed, allow_nonfinite=False):
    """tract-clustered bootstrap.

    each draw samples len(tracts) tracts with replacement and calls
    statistic(sampled_tracts) once, sampled_tracts a list of tract labels with
    repeats. raises on any non-finite draw unless allow_nonfinite.
    """
    tracts = list(tracts)
    n = len(tracts)
    if n < 2:
        raise ValueError('need at least 2 tracts')
    if seed is None:
        raise ValueError('seed is required')
    estimate = _as_stat(statistic(list(tracts)))
    if not np.all(np.isfinite(estimate)):
        raise ValueError('statistic is non-finite on the full tract set')
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_draws, n))
    draws = np.empty((n_draws,) + estimate.shape)
    for b in range(n_draws):
        draws[b] = _as_stat(statistic([tracts[i] for i in idx[b]]), estimate.shape)
    bad = ~np.isfinite(draws.reshape(n_draws, -1)).all(axis=1)
    n_nonfinite = int(bad.sum())
    if n_nonfinite and not allow_nonfinite:
        raise ValueError(f'{n_nonfinite} of {n_draws} bootstrap draws are non-finite')
    return BootstrapResult(estimate=_scalar_or_array(estimate), draws=draws, seed=seed,
                           n_draws=n_draws, numpy_version=np.__version__,
                           quantile_method=QUANTILE_METHOD, n_nonfinite=n_nonfinite,
                           tract_index=idx)


def percentile_interval(draws, level, sided='two'):
    """percentile interval along axis 0.

    sided='two': (1 - level)/2 and (1 + level)/2 quantiles.
    sided='lower': the (1 - level) quantile, upper = +inf.
    raises on non-finite draws.
    """
    draws = np.asarray(draws, dtype=float)
    if not 0.0 < level < 1.0:
        raise ValueError('level must be in (0, 1)')
    if not np.all(np.isfinite(draws)):
        raise ValueError('non-finite draws')
    # levels rounded so 1 - 0.95 is exactly the 0.05 quantile
    if sided == 'two':
        qs = (round((1 - level) / 2, 12), round((1 + level) / 2, 12))
        lo, hi = np.quantile(draws, qs, axis=0, method=QUANTILE_METHOD)
    elif sided == 'lower':
        qs = (round(1 - level, 12),)
        lo = np.quantile(draws, qs[0], axis=0, method=QUANTILE_METHOD)
        hi = np.full_like(lo, np.inf)
    else:
        raise ValueError("sided must be 'two' or 'lower'")
    return Interval(lower=_scalar_or_array(np.asarray(lo)), upper=_scalar_or_array(np.asarray(hi)),
                    sided=sided, method='percentile', level=level,
                    quantile_method=QUANTILE_METHOD, quantiles=qs)


# ---------------------------------------------------------------------------
# jackknife
# ---------------------------------------------------------------------------

def tract_jackknife(tracts, statistic):
    """leave-one-tract-out jackknife; SE = sqrt((n-1)/n * sum (theta_i - theta_bar)^2)."""
    tracts = list(tracts)
    n = len(tracts)
    if n < 3:
        raise ValueError('need at least 3 tracts')
    estimate = _as_stat(statistic(list(tracts)))
    loo = np.stack([_as_stat(statistic(tracts[:i] + tracts[i + 1:]), estimate.shape)
                    for i in range(n)])
    if not (np.all(np.isfinite(estimate)) and np.all(np.isfinite(loo))):
        raise ValueError('non-finite jackknife value')
    se = np.sqrt((n - 1) / n * ((loo - loo.mean(axis=0)) ** 2).sum(axis=0))
    return JackknifeResult(estimate=_scalar_or_array(estimate), loo=loo,
                           se=_scalar_or_array(np.asarray(se)), n=n)


def jackknife_interval(estimate, se, z=JACKKNIFE_Z, sided='two'):
    """estimate +/- z * se around the full-sample estimate, no bias correction."""
    est = np.asarray(estimate, dtype=float)
    se = np.asarray(se, dtype=float)
    if np.any(se < 0) or not (np.all(np.isfinite(est)) and np.all(np.isfinite(se))):
        raise ValueError('invalid estimate or se')
    lo = est - z * se
    if sided == 'two':
        hi = est + z * se
    elif sided == 'lower':
        hi = np.full_like(lo, np.inf)
    else:
        raise ValueError("sided must be 'two' or 'lower'")
    return Interval(lower=_scalar_or_array(lo), upper=_scalar_or_array(np.asarray(hi)),
                    sided=sided, method='jackknife-normal', z=z)


# ---------------------------------------------------------------------------
# decisions
# ---------------------------------------------------------------------------

def decide_lower_bound(bound, threshold):
    """'pass' when bound > threshold, else 'fail'."""
    if not (np.isfinite(bound) and np.isfinite(threshold)):
        raise ValueError('non-finite bound or threshold')
    return 'pass' if bound > threshold else 'fail'


def decide_tost(lo, hi, margin):
    """equivalence decision for an interval against (-margin, +margin).

    'equivalent': -margin < lo and hi < margin.
    'different': the interval lies wholly outside, lo >= margin or hi <= -margin.
    'inconclusive': otherwise.
    """
    if not (np.isfinite(lo) and np.isfinite(hi) and np.isfinite(margin)):
        raise ValueError('non-finite input')
    if lo > hi:
        raise ValueError('lo > hi')
    if margin <= 0:
        raise ValueError('margin must be positive')
    if -margin < lo and hi < margin:
        verdict = 'equivalent'
    elif lo >= margin or hi <= -margin:
        verdict = 'different'
    else:
        verdict = 'inconclusive'
    return TostDecision(verdict=verdict, margin=float(margin),
                        tightest_margin=float(max(abs(lo), abs(hi))))


def compare_verdicts(bootstrap_verdict, jackknife_verdict):
    """both verdicts and whether they agree; no automatic choice."""
    return {'bootstrap': bootstrap_verdict, 'jackknife': jackknife_verdict,
            'agree': bootstrap_verdict == jackknife_verdict}


# ---------------------------------------------------------------------------
# within-tract permutation null
# ---------------------------------------------------------------------------

def permute_within_tracts(values, tracts, rng):
    """shuffle values among positions of the same tract; tract multisets kept exactly."""
    values = np.asarray(values)
    codes = pd.factorize(np.asarray(tracts))[0]
    base = np.argsort(codes, kind='stable')
    shuffled = np.lexsort((rng.random(len(codes)), codes))
    out = np.empty_like(values)
    out[base] = values[shuffled]
    return out


def within_tract_permutation_null(df, value_col, score, n_perm=1000, *, seed,
                                  tract_col='tract_geoid'):
    """null of score under shuffles of address values within each tract.

    every tract mean and per-tract value distribution is kept exactly. each
    shuffle is scored with score(permuted_df). returns the observed score,
    the null draws, their 95th percentile and whether observed exceeds it.
    """
    if seed is None:
        raise ValueError('seed is required')
    rng = np.random.default_rng(seed)
    observed = float(score(df))
    values = df[value_col].to_numpy()
    tracts = df[tract_col].to_numpy()
    null = np.empty(n_perm)
    perm_df = df.copy()
    for i in range(n_perm):
        perm_df[value_col] = permute_within_tracts(values, tracts, rng)
        null[i] = float(score(perm_df))
    if not (np.isfinite(observed) and np.all(np.isfinite(null))):
        raise ValueError('non-finite score')
    p95 = float(np.quantile(null, 0.95, method=QUANTILE_METHOD))
    return PermutationNullResult(observed=observed, null=null, p95=p95,
                                 exceeds=bool(observed > p95), seed=seed, n_perm=n_perm,
                                 quantile_method=QUANTILE_METHOD)
