"""tests for granite.evaluation.paper1_metrics on synthetic, in-memory maps only."""
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from granite.evaluation import paper1_metrics as pm


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def make_map(seed=0, n_tracts=6, bgs_per_tract=4, addr_per_bg=(12, 30)):
    """synthetic address table: tract_geoid, bg_geoid, x, y, plus a bg reference.

    geoids are 11-digit tract and 12-digit bg strings. coordinates cluster by
    block group inside a tract square.
    """
    rng = np.random.default_rng(seed)
    rows, ref = [], {}
    for t in range(n_tracts):
        tract = f'47065{t + 1:06d}'
        ox, oy = (t % 3) * 10.0, (t // 3) * 10.0
        for b in range(bgs_per_tract):
            bg = f'{tract}{b + 1}'
            ref[bg] = float(rng.uniform(0.05, 0.95))
            cx, cy = ox + rng.uniform(1, 9), oy + rng.uniform(1, 9)
            n = int(rng.integers(*addr_per_bg))
            for _ in range(n):
                rows.append({'tract_geoid': tract, 'bg_geoid': bg,
                             'x': cx + rng.normal(0, 0.5), 'y': cy + rng.normal(0, 0.5)})
    return pd.DataFrame(rows), pd.Series(ref, name='svi')


def hand_pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    da, db = a - a.mean(), b - b.mean()
    return float((da * db).sum() / np.sqrt((da * da).sum() * (db * db).sum()))


# ---------------------------------------------------------------------------
# reference-equal map
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('method', ['pearson', 'spearman'])
def test_map_equal_to_reference(method):
    df, ref = make_map(seed=1)
    df['pred'] = df['bg_geoid'].map(ref)
    bg = pm.bg_predictions(df, 'pred')
    rt = pm.pooled_r(bg, ref, method)
    rw = pm.within_tract_r(bg, ref, method)
    assert abs(rt.r - 1.0) < 1e-12
    assert abs(rw.r - 1.0) < 1e-12
    assert rt.n == len(ref) and rt.n_dropped == 0


# ---------------------------------------------------------------------------
# tract-constant map
# ---------------------------------------------------------------------------

def test_tract_constant_map():
    df, ref = make_map(seed=2)
    rng = np.random.default_rng(22)
    tract_val = {t: float(rng.uniform(0.1, 0.9)) for t in df['tract_geoid'].unique()}
    df['pred'] = df['tract_geoid'].map(tract_val)
    bg = pm.bg_predictions(df, 'pred')

    with pytest.warns(pm.Paper1MetricWarning):
        rw = pm.within_tract_r(bg, ref, 'pearson')
    assert np.isnan(rw.r)
    with pytest.warns(pm.Paper1MetricWarning):
        assert np.isnan(pm.within_tract_r(bg, ref, 'spearman').r)

    expected = hand_pearson(bg['tract_geoid'].map(tract_val), bg['bg_geoid'].map(ref))
    assert abs(pm.pooled_r(bg, ref, 'pearson').r - expected) < 1e-12

    with pytest.warns(pm.Paper1MetricWarning):
        mi = pm.per_tract_morans_i(df, 'pred')
    assert mi.isna().all() and len(mi) == df['tract_geoid'].nunique()
    with pytest.raises(ValueError):
        pm.run_coherence(mi)


def test_warning_survives_spatial_diagnostics_global_filter():
    # spatial_diagnostics sets warnings.filterwarnings('ignore') at import
    import granite.evaluation.spatial_diagnostics  # noqa: F401
    with warnings.catch_warnings(record=True) as rec:
        warnings.filterwarnings('ignore')
        pm.correlation(np.ones(5), np.arange(5.0), 'pearson')
    assert any(issubclass(w.category, pm.Paper1MetricWarning) for w in rec)


# ---------------------------------------------------------------------------
# within-tract permutation and per-tract shifts
# ---------------------------------------------------------------------------

def test_within_tract_permuted_map_keeps_tract_means_and_values():
    df, _ = make_map(seed=3)
    rng = np.random.default_rng(33)
    df['pred'] = rng.uniform(0, 1, len(df))
    perm = df.copy()
    for t, idx in perm.groupby('tract_geoid').groups.items():
        perm.loc[idx, 'pred'] = rng.permutation(perm.loc[idx, 'pred'].to_numpy())
    m0 = df.groupby('tract_geoid')['pred'].mean()
    m1 = perm.groupby('tract_geoid')['pred'].mean()
    assert np.max(np.abs(m0 - m1)) < 1e-12
    for t in df['tract_geoid'].unique():
        a = np.sort(df.loc[df.tract_geoid == t, 'pred'].to_numpy())
        b = np.sort(perm.loc[perm.tract_geoid == t, 'pred'].to_numpy())
        assert np.array_equal(a, b)
    assert not np.array_equal(df['pred'].to_numpy(), perm['pred'].to_numpy())


@pytest.mark.parametrize('method', ['pearson', 'spearman'])
def test_per_tract_constant_shift_leaves_rw_unchanged(method):
    df, ref = make_map(seed=4)
    rng = np.random.default_rng(44)
    df['pred'] = df['bg_geoid'].map(ref) + rng.normal(0, 0.2, len(df))
    shift = {t: float(rng.normal(0, 3)) for t in df['tract_geoid'].unique()}
    df['shifted'] = df['pred'] + df['tract_geoid'].map(shift)
    r0 = pm.within_tract_r(pm.bg_predictions(df, 'pred'), ref, method).r
    r1 = pm.within_tract_r(pm.bg_predictions(df, 'shifted'), ref, method).r
    assert abs(r0 - r1) < 1e-12


def test_reference_plus_tract_offsets():
    df, ref = make_map(seed=5)
    rng = np.random.default_rng(55)
    off = {t: float(rng.normal(0, 0.5)) for t in df['tract_geoid'].unique()}
    df['pred'] = df['bg_geoid'].map(ref) + df['tract_geoid'].map(off)
    bg = pm.bg_predictions(df, 'pred')
    for method in ('pearson', 'spearman'):
        assert abs(pm.within_tract_r(bg, ref, method).r - 1.0) < 1e-12
        assert pm.pooled_r(bg, ref, method).r < 1.0 - 1e-3


# ---------------------------------------------------------------------------
# moran's i
# ---------------------------------------------------------------------------

def _direct_moran(coords, values, k=8):
    import esda
    import libpysal.weights as lw
    w = lw.KNN.from_array(np.asarray(coords, float), k=k)
    w.transform = 'r'
    return esda.Moran(np.asarray(values, float), w, permutations=0).I


def test_morans_i_signs_and_parity():
    rng = np.random.default_rng(6)
    gx, gy = np.meshgrid(np.arange(12.0), np.arange(12.0))
    coords = np.column_stack([gx.ravel(), gy.ravel()]) + rng.normal(0, 1e-3, (144, 2))
    gradient = coords[:, 0] + coords[:, 1]
    checker = ((gx.ravel() + gy.ravel()) % 2).astype(float)
    i_grad = pm.tract_morans_i(coords, gradient)
    assert i_grad > 0.5
    assert abs(i_grad - _direct_moran(coords, gradient)) < 1e-12
    # on a square grid k=8 holds 4 rook (opposite) and 4 diagonal (same) neighbours,
    # so the checkerboard sign check uses k=4 (rook)
    i_check = pm.tract_morans_i(coords, checker, k=4)
    assert i_check < -0.5
    assert abs(i_check - _direct_moran(coords, checker, k=4)) < 1e-12
    assert abs(pm.tract_morans_i(coords, checker) - _direct_moran(coords, checker)) < 1e-12


def test_morans_i_parity_with_duplicate_coordinates():
    rng = np.random.default_rng(7)
    coords = rng.uniform(0, 1, (300, 2))
    coords[150:200] = coords[100:150]   # duplicated locations
    vals = rng.normal(size=300) + coords[:, 0]
    assert abs(pm.tract_morans_i(coords, vals) - _direct_moran(coords, vals)) < 1e-12


def test_per_tract_morans_i_and_coherence():
    df, ref = make_map(seed=8)
    df['pred'] = df['bg_geoid'].map(ref) + np.random.default_rng(88).normal(0, 0.05, len(df))
    mi = pm.per_tract_morans_i(df, 'pred')
    for t, g in df.groupby('tract_geoid'):
        assert abs(mi[t] - _direct_moran(g[['x', 'y']].to_numpy(), g['pred'].to_numpy())) < 1e-12
    assert abs(pm.run_coherence(mi) - mi.mean()) < 1e-15
    sub = list(mi.index[:3])
    assert abs(pm.run_coherence(mi, sub) - mi[sub].mean()) < 1e-15
    # a duplicated tract set counts each listed entry
    dup = [sub[0], sub[0], sub[1]]
    assert abs(pm.run_coherence(mi, dup) - (2 * mi[sub[0]] + mi[sub[1]]) / 3) < 1e-15
    with pytest.raises(ValueError):
        pm.run_coherence(mi, ['nope'])


def test_morans_i_refuses_tiny_tract():
    with pytest.raises(ValueError):
        pm.tract_morans_i(np.zeros((5, 2)), np.arange(5.0))


# ---------------------------------------------------------------------------
# block groups and pooled r
# ---------------------------------------------------------------------------

def test_min_block_group_size():
    df, ref = make_map(seed=9)
    df['pred'] = 0.5
    small_bg = '470650000019'
    extra = pd.DataFrame({'tract_geoid': ['47065000001'] * 2, 'bg_geoid': [small_bg] * 2,
                          'x': [0.0, 0.1], 'y': [0.0, 0.1], 'pred': [0.2, 0.3]})
    bg = pm.bg_predictions(pd.concat([df, extra], ignore_index=True), 'pred')
    assert small_bg not in set(bg['bg_geoid'])
    assert (bg['n_addresses'] >= 3).all()
    three = pd.concat([extra, extra.iloc[:1]], ignore_index=True)
    bg3 = pm.bg_predictions(pd.concat([df, three], ignore_index=True), 'pred')
    row = bg3.set_index('bg_geoid').loc[small_bg]
    assert row['n_addresses'] == 3 and abs(row['value'] - 0.7 / 3) < 1e-15


def test_bg_mean_is_plain_mean():
    df, ref = make_map(seed=10)
    df['pred'] = np.random.default_rng(100).uniform(0, 1, len(df))
    bg = pm.bg_predictions(df, 'pred').set_index('bg_geoid')
    plain = df.groupby('bg_geoid')['pred'].mean()
    assert np.max(np.abs(bg['value'] - plain.loc[bg.index])) < 1e-15


def test_bg_spanning_two_tracts_raises():
    df, _ = make_map(seed=11)
    df['pred'] = 0.5
    df.loc[0, 'tract_geoid'] = '47065999999'
    with pytest.raises(ValueError):
        pm.bg_predictions(df, 'pred')


def test_pooled_r_drops_missing_reference():
    df, ref = make_map(seed=12)
    df['pred'] = df['bg_geoid'].map(ref) + np.random.default_rng(120).normal(0, 0.1, len(df))
    bg = pm.bg_predictions(df, 'pred')
    ref2 = ref.copy()
    ref2.iloc[0] = np.nan
    ref2 = ref2.drop(ref2.index[1])
    res = pm.pooled_r(bg, ref2, 'pearson')
    assert res.n_dropped == 2 and res.n == len(bg) - 2
    kept = bg[~bg['bg_geoid'].isin(ref.index[:2])]
    assert abs(res.r - hand_pearson(kept['value'], kept['bg_geoid'].map(ref))) < 1e-12


def test_spearman_ties_match_scipy():
    rng = np.random.default_rng(13)
    n = 60
    bg = pd.DataFrame({'bg_geoid': [f'47065{i // 5:06d}{i % 5 + 1}' for i in range(n)],
                       'tract_geoid': [f'47065{i // 5:06d}' for i in range(n)],
                       'n_addresses': 10,
                       'value': np.clip(rng.normal(0.5, 0.5, n), 0, 1)})
    ref = pd.Series(np.clip(bg['value'].to_numpy() + rng.normal(0, 0.4, n), 0, 1),
                    index=bg['bg_geoid'])
    assert (bg['value'] == 0).sum() > 2 and (bg['value'] == 1).sum() > 2
    assert (ref == 0).sum() > 2 and (ref == 1).sum() > 2
    res = pm.pooled_r(bg, ref, 'spearman')
    expected = stats.spearmanr(bg['value'].to_numpy(), ref.to_numpy()).statistic
    assert abs(res.r - expected) < 1e-12


# ---------------------------------------------------------------------------
# fisher z
# ---------------------------------------------------------------------------

def test_fisher_z_mean_and_q():
    rs = [0.2, 0.5, 0.71, -0.1]
    hand = np.tanh((np.arctanh(0.2) + np.arctanh(0.5) + np.arctanh(0.71) + np.arctanh(-0.1)) / 4)
    assert abs(pm.fisher_z_mean(rs) - hand) < 1e-15
    for bad in ([0.3, 1.0], [-1.0], [0.2, np.nan]):
        with pytest.raises(ValueError):
            pm.fisher_z_mean(bad)
    assert abs(pm.q_difference(0.8, 0.6) - (np.arctanh(0.8) - np.arctanh(0.6))) < 1e-15
    v = pm.q_difference(np.array([0.1, 0.2]), np.array([0.0, 0.1]))
    assert v.shape == (2,)
    with pytest.raises(ValueError):
        pm.q_difference(1.0, 0.5)


# ---------------------------------------------------------------------------
# duplicated tracts and single-bg tracts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('method', ['pearson', 'spearman'])
def test_duplicated_tracts_joint_equals_separate(method):
    df, ref = make_map(seed=14, n_tracts=6)
    df['pred'] = df['bg_geoid'].map(ref) + np.random.default_rng(140).normal(0, 0.2, len(df))
    bg = pm.bg_predictions(df, 'pred')
    tracts = sorted(bg['tract_geoid'].unique())
    draw = [tracts[0], tracts[0], tracts[0], tracts[2], tracts[3], tracts[3]]
    joint = pm.resample_tracts(bg, draw)
    # separate: relabel each copy as its own tract
    parts = []
    for i, t in enumerate(draw):
        p = bg[bg['tract_geoid'] == t].copy()
        p['tract_geoid'] = f'{t}#{i}'
        parts.append(p)
    sep = pd.concat(parts, ignore_index=True)
    r_joint = pm.within_tract_r(joint, ref, method)
    r_sep = pm.within_tract_r(sep, ref, method)
    assert abs(r_joint.r - r_sep.r) < 1e-12
    assert r_joint.n == r_sep.n == len(joint)
    # pooled r on the draw counts each copy
    p_dup = pm.pooled_r(joint, ref, method)
    assert p_dup.n == len(joint)


def test_single_bg_tracts_excluded():
    df, ref = make_map(seed=15)
    df['pred'] = df['bg_geoid'].map(ref) + np.random.default_rng(150).normal(0, 0.2, len(df))
    bg = pm.bg_predictions(df, 'pred')
    singles = pd.DataFrame({'bg_geoid': ['470659000011', '470659000021'],
                            'tract_geoid': ['47065900001', '47065900002'],
                            'n_addresses': [5, 7], 'value': [0.9, 0.1]})
    ref2 = pd.concat([ref, pd.Series({'470659000011': 0.2, '470659000021': 0.8})])
    with_s = pm.within_tract_r(pd.concat([bg, singles], ignore_index=True), ref2, 'pearson')
    without = pm.within_tract_r(bg, ref2, 'pearson')
    assert with_s.n_tracts_excluded == 2 and with_s.n_rows_excluded == 2
    assert without.n_tracts_excluded == 0
    assert with_s.r == without.r
    # a duplicated single-bg tract is still excluded when demeaned jointly
    dup = pd.concat([bg, singles, singles.iloc[:1]], ignore_index=True)
    assert pm.within_tract_r(dup, ref2, 'spearman').r == pm.within_tract_r(bg, ref2, 'spearman').r
