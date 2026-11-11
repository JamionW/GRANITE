"""tests for granite.evaluation.paper1_inference on synthetic, in-memory maps only."""
import numpy as np
import pandas as pd
import pytest

from granite.evaluation import paper1_inference as pi
from granite.evaluation import paper1_metrics as pm


def mean_stat(values):
    values = np.asarray(values, dtype=float)
    return lambda s: float(values[np.asarray(s)].mean())


# ---------------------------------------------------------------------------
# bootstrap basics
# ---------------------------------------------------------------------------

def test_seed_reproducible_and_recorded():
    v = np.random.default_rng(0).normal(size=20)
    a = pi.tract_bootstrap(range(20), mean_stat(v), n_draws=300, seed=11)
    b = pi.tract_bootstrap(range(20), mean_stat(v), n_draws=300, seed=11)
    c = pi.tract_bootstrap(range(20), mean_stat(v), n_draws=300, seed=12)
    assert np.array_equal(a.draws, b.draws)
    assert not np.array_equal(a.draws, c.draws)
    assert a.seed == 11 and c.seed == 12 and a.n_draws == 300
    assert a.numpy_version == np.__version__ and a.quantile_method == 'linear'
    assert a.n_nonfinite == 0 and abs(a.estimate - v.mean()) < 1e-15
    # index scheme matches BOOT-P1: default_rng(seed).integers(0, n, size=(B, n))
    assert np.array_equal(a.tract_index, np.random.default_rng(11).integers(0, 20, size=(300, 20)))


def test_seed_required():
    with pytest.raises(TypeError):
        pi.tract_bootstrap(range(5), mean_stat(np.arange(5.0)), n_draws=10)


def test_vector_statistic():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=15), rng.normal(size=15)
    res = pi.tract_bootstrap(range(15), lambda s: [a[s].mean(), b[s].mean(), (a[s] - b[s]).mean()],
                             n_draws=200, seed=3)
    assert res.draws.shape == (200, 3)
    assert np.allclose(res.draws[:, 0] - res.draws[:, 1], res.draws[:, 2], atol=1e-12)
    iv = pi.percentile_interval(res.draws, 0.9)
    assert iv.lower.shape == (3,)


def test_pairing_zero_width_and_negative_control():
    rng = np.random.default_rng(2)
    a = rng.normal(size=20)
    c = 0.37
    b = a + c
    paired = pi.tract_bootstrap(range(20), lambda s: b[s].mean() - a[s].mean(),
                                n_draws=1000, seed=5)
    assert np.max(np.abs(paired.draws - c)) < 1e-12
    iv = pi.percentile_interval(paired.draws, 0.9)
    assert iv.upper - iv.lower < 1e-12
    # negative control: conditions resampled independently, pairing broken
    ra = pi.tract_bootstrap(range(20), mean_stat(a), n_draws=1000, seed=5)
    rb = pi.tract_bootstrap(range(20), mean_stat(b), n_draws=1000, seed=6)
    iv_broken = pi.percentile_interval(rb.draws - ra.draws, 0.9)
    assert iv_broken.upper - iv_broken.lower > 0.1


def test_pairing_through_block_group_statistic():
    # two conditions scored on the same sampled block groups inside one call
    rng = np.random.default_rng(3)
    tracts = [f'47065{t:06d}' for t in range(12)]
    bg = pd.DataFrame([{'bg_geoid': f'{t}{b}', 'tract_geoid': t, 'n_addresses': 10}
                       for t in tracts for b in range(1, 5)])
    ref = pd.Series(rng.uniform(0, 1, len(bg)), index=bg['bg_geoid'])
    a = bg.assign(value=ref.to_numpy() + rng.normal(0, 0.2, len(bg)))
    off = {t: rng.normal() for t in tracts}
    b = a.assign(value=a['value'] + a['tract_geoid'].map(off))  # per-tract shift leaves r_W

    def stat(s):
        ra = pm.within_tract_r(pm.resample_tracts(a, s), ref, 'pearson').r
        rb = pm.within_tract_r(pm.resample_tracts(b, s), ref, 'pearson').r
        return rb - ra
    res = pi.tract_bootstrap(tracts, stat, n_draws=100, seed=7)
    assert np.max(np.abs(res.draws)) < 1e-12


def test_nonfinite_draw_raises():
    v = np.arange(10.0)

    def stat(s):
        return float('nan') if s.count(0) >= 3 else v[s].mean()
    with pytest.raises(ValueError):
        pi.tract_bootstrap(list(range(10)), stat, n_draws=500, seed=1)
    res = pi.tract_bootstrap(list(range(10)), stat, n_draws=500, seed=1, allow_nonfinite=True)
    assert res.n_nonfinite > 0
    with pytest.raises(ValueError):
        pi.percentile_interval(res.draws, 0.9)


# ---------------------------------------------------------------------------
# intervals and jackknife
# ---------------------------------------------------------------------------

def test_one_sided_lower_bound_is_5th_percentile():
    v = np.random.default_rng(4).normal(size=20)
    res = pi.tract_bootstrap(range(20), mean_stat(v), n_draws=2000, seed=9)
    iv = pi.percentile_interval(res.draws, 0.95, sided='lower')
    assert iv.lower == np.percentile(res.draws, 5)
    assert iv.upper == np.inf and iv.sided == 'lower'
    two = pi.percentile_interval(res.draws, 0.9)
    assert two.lower == np.percentile(res.draws, 5) and two.upper == np.percentile(res.draws, 95)


def test_jackknife_identity_for_mean():
    v = np.random.default_rng(5).normal(size=23)
    jk = pi.tract_jackknife(list(range(23)), mean_stat(v))
    assert abs(jk.se - v.std(ddof=1) / np.sqrt(23)) < 1e-12
    assert abs(jk.estimate - v.mean()) < 1e-15 and jk.loo.shape == (23,)
    iv = pi.jackknife_interval(jk.estimate, jk.se)
    assert abs(iv.lower - (v.mean() - 1.645 * jk.se)) < 1e-15
    assert abs(iv.upper - (v.mean() + 1.645 * jk.se)) < 1e-15
    lo = pi.jackknife_interval(jk.estimate, jk.se, sided='lower')
    assert lo.lower == iv.lower and lo.upper == np.inf


def test_decide_tost_and_compare():
    eq = pi.decide_tost(-0.04, 0.06, 0.10)
    assert eq.verdict == 'equivalent' and abs(eq.tightest_margin - 0.06) < 1e-15
    diff = pi.decide_tost(0.12, 0.30, 0.10)
    assert diff.verdict == 'different' and abs(diff.tightest_margin - 0.30) < 1e-15
    assert pi.decide_tost(-0.40, -0.10, 0.10).verdict == 'different'
    inc = pi.decide_tost(0.02, 0.15, 0.10)   # excludes zero, crosses the margin
    assert inc.verdict == 'inconclusive' and abs(inc.tightest_margin - 0.15) < 1e-15
    assert pi.decide_tost(-0.2, 0.2, 0.10).verdict == 'inconclusive'
    assert pi.decide_lower_bound(0.31, 0.3) == 'pass'
    assert pi.decide_lower_bound(0.3, 0.3) == 'fail'
    cmp_ = pi.compare_verdicts('equivalent', 'inconclusive')
    assert cmp_ == {'bootstrap': 'equivalent', 'jackknife': 'inconclusive', 'agree': False}
    assert pi.compare_verdicts('pass', 'pass')['agree'] is True


# ---------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------

def test_coverage_sanity():
    rng = np.random.default_rng(6)
    mu, n_tracts, covered = 0.4, 20, 0
    for rep in range(200):
        u = rng.normal(0, 0.5, n_tracts)
        sizes = rng.integers(20, 80, n_tracts)
        tract_means = np.array([rng.normal(mu + u[t], 1.0, sizes[t]).mean()
                                for t in range(n_tracts)])
        res = pi.tract_bootstrap(range(n_tracts), mean_stat(tract_means), n_draws=500,
                                 seed=1000 + rep)
        iv = pi.percentile_interval(res.draws, 0.90)
        covered += iv.lower <= mu <= iv.upper
    rate = covered / 200
    assert 0.84 <= rate <= 0.96, rate


# ---------------------------------------------------------------------------
# permutation null
# ---------------------------------------------------------------------------

def make_address_map(rng, n_tracts=8, bgs=4, addr=(15, 40), signal=0.0):
    rows, ref = [], {}
    for t in range(n_tracts):
        tract = f'47065{t:06d}'
        u = rng.normal(0, 0.3)
        for b in range(1, bgs + 1):
            g = f'{tract}{b}'
            ref[g] = rng.uniform(0, 1)
            n = int(rng.integers(*addr))
            vals = u + signal * ref[g] + rng.normal(0, 0.5, n)
            rows += [{'tract_geoid': tract, 'bg_geoid': g, 'pred': v} for v in vals]
    return pd.DataFrame(rows), pd.Series(ref)


def test_permute_within_tracts_keeps_tract_multisets():
    rng = np.random.default_rng(7)
    df, _ = make_address_map(rng)
    out = pi.permute_within_tracts(df['pred'].to_numpy(), df['tract_geoid'].to_numpy(), rng)
    for t, idx in df.groupby('tract_geoid').indices.items():
        assert np.array_equal(np.sort(out[idx]), np.sort(df['pred'].to_numpy()[idx]))
        assert abs(out[idx].mean() - df['pred'].to_numpy()[idx].mean()) < 1e-12
    assert not np.array_equal(out, df['pred'].to_numpy())


def test_permutation_null_detects_planted_signal():
    rng = np.random.default_rng(8)
    df, ref = make_address_map(rng, signal=1.5)

    def score(d):
        return pm.within_tract_r(pm.bg_predictions(d, 'pred'), ref, 'pearson').r
    res = pi.within_tract_permutation_null(df, 'pred', score, n_perm=200, seed=21)
    assert res.exceeds and res.observed > res.p95
    assert res.null.shape == (200,) and res.seed == 21
    assert abs(res.p95 - np.quantile(res.null, 0.95)) < 1e-15


def _fast_rw_score(df, ref):
    # r_W on block-group means using the module's array cores
    bg_codes, bg_keys = pd.factorize(df['bg_geoid'])
    tract_of_bg = pd.Series(df['tract_geoid'].to_numpy(), index=bg_codes).groupby(level=0).first()
    counts = np.bincount(bg_codes)
    t = ref.loc[bg_keys].to_numpy()
    groups = tract_of_bg.sort_index().to_numpy()
    td = pm.demean_within(t, groups)

    def score(d):
        m = np.bincount(bg_codes, weights=d['pred'].to_numpy()) / counts
        return pm.correlation(pm.demean_within(m, groups), td, 'pearson')
    return score


def test_permutation_null_exceedance_rate_without_signal():
    rng = np.random.default_rng(9)
    exceed = 0
    for rep in range(100):
        df, ref = make_address_map(rng, signal=0.0)
        res = pi.within_tract_permutation_null(df, 'pred', _fast_rw_score(df, ref),
                                               n_perm=200, seed=500 + rep)
        exceed += res.exceeds
    rate = exceed / 100
    assert 0.01 <= rate <= 0.10, rate


def test_fast_score_matches_module_rw():
    rng = np.random.default_rng(10)
    df, ref = make_address_map(rng, signal=0.8)
    fast = _fast_rw_score(df, ref)(df)
    slow = pm.within_tract_r(pm.bg_predictions(df, 'pred'), ref, 'pearson').r
    assert abs(fast - slow) < 1e-12
