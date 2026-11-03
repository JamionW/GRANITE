"""
Build the Paper 1 figures (FIG-P1 B6) from committed artifacts only.

One command, no network, no pipeline runs:
    python scripts/build_paper1_figures.py

Figures follow docs/paper1/figure_list.md. Output:
  data/results/paper1_figures/figures/fig{1,2,3,4}.png   (gitignored; regenerated)
  data/results/paper1_figures/figure_numbers.json        every plotted or captioned
                                                         number, full precision, with
                                                         its source path and commit

Rounding for display happens once, in round3 (3 decimals, ROUND_HALF_UP on repr(float)).
Tract-clustered intervals: resample the 20 tracts with replacement, 2000 resamples,
percentile 95%, fixed RNG seed BOOT_SEED.
"""

import json
import subprocess
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use('Agg')
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from granite.evaluation.spatial_diagnostics import SpatialLearningDiagnostics

FIG_DIR_REL = 'data/results/paper1_figures'
OUT_DIR = REPO_ROOT / FIG_DIR_REL
PNG_DIR = OUT_DIR / 'figures'
NUMBERS_JSON = OUT_DIR / 'figure_numbers.json'

SRC = {
    'geojson': 'data/results/paper1_figures/hamilton_tracts_svi.geojson',
    'inset': 'data/results/paper1_figures/inset_edges_47065000600.npz',
    'inset_prov': 'data/results/paper1_figures/provenance.json',
    'm0_agg': 'data/results/bg_reference_v2/m0_aggregate_v1_v2.csv',
    'rank': 'data/results/bg_reference_v2/power_analysis_v2_tietol.json',
    'm5b_metrics': 'data/results/m6_topology_5b/topology_specificity_metrics.json',
    'm5b_pred': 'data/results/m6_topology_5b/predictions',
    'crossed_pred': 'data/results/m6_topology_5b_crossed/predictions',
    'crossed_trials': 'data/results/m6_topology_5b_crossed/trials_incremental.csv',
    'cond_means': 'data/results/bg_reference_v2/rescored_condition_means.csv',
    'fig4_tracts': 'data/results/m6_fig4_randomization/tract_rows.csv',
    'fig4_cells': 'data/results/m6_fig4_randomization/cells.csv',
    'grid': 'data/results/m6_recovery_grid/recovery_grid.csv',
    'field': 'data/synthetic/run_20260628_030416/addresses.csv',
    'tract_selection': 'experiments/ablation/00_baseline/tract_selection.txt',
}

ARCHS = ['sage', 'gcn_gat']
ARCH_LABEL = {'sage': 'GraphSAGE', 'gcn_gat': 'GCN-GAT'}
ARCH_COLOR = {'sage': '#2a78d6', 'gcn_gat': '#eb6834'}   # categorical slots 1 and 2
TRAINING_SEEDS = [42, 17, 123, 2024, 7]
GRAPH_DRAW_SEEDS = [42, 17, 123, 2024, 7]
N_BOOT = 2000
BOOT_SEED = 20261101
TEXT = '#0b0b0b'
TEXT2 = '#52514e'
GRID = '#e6e5e1'

plt.rcParams.update({
    'font.size': 9, 'axes.edgecolor': TEXT2, 'axes.labelcolor': TEXT, 'xtick.color': TEXT2,
    'ytick.color': TEXT2, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
    'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb',
})


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def round3(x):
    """display rounding: 3 decimals, ROUND_HALF_UP via decimal on repr(float)."""
    return float(Decimal(repr(float(x))).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP))


_COMMITS = {}


def source_commit(rel):
    if rel not in _COMMITS:
        try:
            _COMMITS[rel] = subprocess.check_output(
                ['git', 'log', '-1', '--format=%H', '--', rel], cwd=str(REPO_ROOT)
            ).decode().strip() or 'untracked'
        except Exception:
            _COMMITS[rel] = 'unknown'
    return _COMMITS[rel]


NUMBERS = []


def record(figure, name, value, source, ci=None, **extra):
    entry = {'figure': figure, 'name': name, 'value': None if value is None else float(value),
             'value_rounded': None if value is None else round3(value),
             'source': source, 'source_commit': source_commit(source)}
    if ci is not None:
        entry['ci95'] = [float(ci[0]), float(ci[1])]
        entry['ci95_rounded'] = [round3(ci[0]), round3(ci[1])]
    entry.update(extra)
    NUMBERS.append(entry)
    return entry


def tract_bootstrap(per_tract, rng):
    """per_tract: 1-D array of tract-level values; returns (mean, lo, hi)."""
    v = np.asarray(per_tract, dtype=float)
    idx = rng.integers(0, len(v), size=(N_BOOT, len(v)))
    boots = v[idx].mean(axis=1)
    return float(v.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def crossed_variance_components(cell_means):
    """
    two-way crossed random-effects ANOVA without replication on cell means
    (training seed x graph draw), balanced moment estimators; negatives truncated at 0.
    cell_means: DataFrame indexed by training_seed with graph_draw_seed columns.
    """
    y = cell_means.to_numpy(dtype=float)
    a, b = y.shape
    grand = y.mean()
    ss_t = b * ((y.mean(axis=1) - grand) ** 2).sum()
    ss_g = a * ((y.mean(axis=0) - grand) ** 2).sum()
    ss_e = ((y - y.mean(axis=1, keepdims=True) - y.mean(axis=0, keepdims=True) + grand) ** 2).sum()
    ms_t, ms_g, ms_e = ss_t / (a - 1), ss_g / (b - 1), ss_e / ((a - 1) * (b - 1))
    raw_t, raw_g = (ms_t - ms_e) / b, (ms_g - ms_e) / a
    return {
        'estimator': 'two-way crossed random effects, no replication, balanced moment estimators',
        'n_training_seeds': a, 'n_graph_draws': b,
        'ms_training_seed': ms_t, 'ms_graph_draw': ms_g, 'ms_residual': ms_e,
        'var_training_seed_raw': raw_t, 'var_graph_draw_raw': raw_g,
        'var_training_seed': max(raw_t, 0.0), 'var_graph_draw': max(raw_g, 0.0),
        'var_residual_interaction': ms_e,
    }


def load_tracts():
    text = (REPO_ROOT / SRC['tract_selection']).read_text()
    return sorted(text.split('fips_list:')[1].split())


# ---------------------------------------------------------------------------
# figure 1
# ---------------------------------------------------------------------------

def figure1():
    gdf = gpd.read_file(REPO_ROOT / SRC['geojson'])
    z = np.load(REPO_ROOT / SRC['inset'])
    prov = json.loads((REPO_ROOT / SRC['inset_prov']).read_text())
    ei, coords, win = z['edge_index'], z['coords'], z['window_idx']
    inset_fips = '47065000600'

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [1.25, 1]},
                                   layout='constrained')
    has = gdf[gdf['RPL_THEMES'].notna()]
    has.plot(column='RPL_THEMES', cmap='Blues', vmin=0, vmax=1, ax=ax1, edgecolor='#fcfcfb',
             linewidth=0.3, legend=True,
             legend_kwds={'label': 'Tract SVI (RPL_THEMES)', 'shrink': 0.6})
    gdf[gdf['RPL_THEMES'].isna()].plot(ax=ax1, color='#f0efec', edgecolor=TEXT2, linewidth=0.3,
                                       hatch='///')
    gdf[gdf['paper1']].boundary.plot(ax=ax1, color=TEXT, linewidth=1.1)
    gdf[gdf['GEOID'] == inset_fips].boundary.plot(ax=ax1, color='#eb6834', linewidth=1.6)
    ax1.set_title('a  Tract SVI; Paper 1 tracts outlined', loc='left', color=TEXT)
    ax1.set_axis_off()

    in_win = np.zeros(len(coords), dtype=bool)
    in_win[win] = True
    keep = in_win[ei[0]] & in_win[ei[1]] & (ei[0] < ei[1])
    segs = np.stack([coords[ei[0, keep]], coords[ei[1, keep]]], axis=1)
    ax2.add_collection(LineCollection(segs, colors='#86b6ef', linewidths=0.6))
    ax2.scatter(coords[win, 0], coords[win, 1], s=9, color='#1c5cab', zorder=3,
                edgecolors='#fcfcfb', linewidths=0.5)
    ax2.set_aspect('equal')
    ax2.set_title(f'b  Production graph, tract {inset_fips}\n    {len(win)}-node window',
                  loc='left', color=TEXT)
    ax2.set_axis_off()
    fig.savefig(PNG_DIR / 'fig1.png', dpi=200)
    plt.close(fig)

    st = prov['outputs'][SRC['inset']]
    n_t = int(len(gdf))
    record(1, 'n_tracts_county', n_t, SRC['geojson'])
    record(1, 'n_tracts_with_svi', int(gdf['RPL_THEMES'].notna().sum()), SRC['geojson'])
    record(1, 'n_tracts_paper1', int(gdf['paper1'].sum()), SRC['geojson'])
    record(1, 'inset_n_nodes', st['n_nodes'], SRC['inset_prov'])
    record(1, 'inset_n_undirected_edges', st['n_undirected_edges'], SRC['inset_prov'])
    record(1, 'inset_full_tract_mean_degree', st['full_tract_mean_degree'], SRC['inset_prov'])
    record(1, 'inset_window_n_nodes', st['window_n_nodes'], SRC['inset_prov'])
    record(1, 'inset_window_induced_mean_degree', st['window_induced_mean_degree'], SRC['inset_prov'])
    # recomputed from the npz as a check on the provenance values
    assert ei.shape[1] // 2 == st['n_undirected_edges'] and len(win) == st['window_n_nodes']


# ---------------------------------------------------------------------------
# figure 2 (panel a only)
# ---------------------------------------------------------------------------

def figure2():
    agg = pd.read_csv(REPO_ROOT / SRC['m0_agg'])
    v2 = agg[agg['reference_version'] == 'v2'].set_index('method')
    methods = ['GRANITE', 'Dasymetric', 'Pycnophylactic']
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for i, m in enumerate(methods):
        r = v2.loc[m]
        ax.errorbar(i, r['pooled_bg_r'], yerr=[[r['pooled_bg_r'] - r['pooled_bg_r_ci_low']],
                                               [r['pooled_bg_r_ci_high'] - r['pooled_bg_r']]],
                    fmt='o', ms=7, color=TEXT2, ecolor=TEXT2, elinewidth=1.5, capsize=0)
        ax.annotate(f"{round3(r['pooled_bg_r']):.3f}", (i, r['pooled_bg_r']), xytext=(8, 0),
                    textcoords='offset points', va='center', color=TEXT)
        record(2, f'pooled_bg_r_{m}', r['pooled_bg_r'], SRC['m0_agg'],
               ci=(r['pooled_bg_r_ci_low'], r['pooled_bg_r_ci_high']),
               n_bgs=int(r['pooled_n_bgs']), reference='v2')
    ax.set_xticks(range(3), methods)
    ax.set_xlim(-0.5, 2.7)
    ax.set_ylabel('Pooled block-group r (67 BGs, v2)')
    ax.set_title('a  Agreement with the block-group reference', loc='left', color=TEXT)
    fig.tight_layout()
    fig.savefig(PNG_DIR / 'fig2.png', dpi=200)
    plt.close(fig)

    rank = json.loads((REPO_ROOT / SRC['rank']).read_text())['rank_based_sensitivity']
    w, s = rank['wilcoxon_signed_rank'], rank['exact_sign_test']
    record(2, 'caption_wilcoxon_statistic', w['statistic'], SRC['rank'], n_nonzero=w['n_nonzero'])
    record(2, 'caption_wilcoxon_p', w['p'], SRC['rank'])
    record(2, 'caption_sign_test_n_pos', s['n_pos'], SRC['rank'])
    record(2, 'caption_sign_test_n_neg', s['n_neg'], SRC['rank'])
    record(2, 'caption_sign_test_p', s['p'], SRC['rank'])


# ---------------------------------------------------------------------------
# figure 3
# ---------------------------------------------------------------------------

def per_tract_morans(pred_dir, condition, arch, cells, tracts):
    """returns array (n_cells, n_tracts) of output Moran's I from committed npz."""
    diag = SpatialLearningDiagnostics(verbose=False)
    out = np.full((len(cells), len(tracts)), np.nan)
    for i, (ts, gs) in enumerate(cells):
        for j, f in enumerate(tracts):
            z = np.load(pred_dir / f'{condition}__{arch}__ts{ts}__gs{gs}__{f}.npz')
            out[i, j] = diag.compute_spatial_autocorrelation(z['predictions'], z['coords'],
                                                             k_neighbors=8)
    return out


def figure3(rng, tracts):
    metrics = json.loads((REPO_ROOT / SRC['m5b_metrics']).read_text())
    trials = pd.read_csv(REPO_ROOT / SRC['crossed_trials'])
    struct_cells = [(ts, 42) for ts in TRAINING_SEEDS]
    cross_cells = [(ts, gs) for ts in TRAINING_SEEDS for gs in GRAPH_DRAW_SEEDS]
    res = {}
    for arch in ARCHS:
        s_mi = per_tract_morans(REPO_ROOT / SRC['m5b_pred'], 'spatial_knn_uniform', arch,
                                struct_cells, tracts)
        c_mi = per_tract_morans(REPO_ROOT / SRC['crossed_pred'], 'randomized', arch,
                                cross_cells, tracts)
        # check: per-cell tract means reproduce the committed metrics
        s_json = [d['morans_i'] for d in metrics['spatial_knn_uniform'][arch]['seeds']]
        assert np.allclose(s_mi.mean(axis=1), s_json, atol=1e-9), (arch, s_mi.mean(axis=1), s_json)
        t = trials[trials['arch'] == arch].set_index(['training_seed', 'graph_draw_seed'])
        c_json = [t.loc[c, 'morans_i'] for c in cross_cells]
        assert np.allclose(c_mi.mean(axis=1), c_json, atol=1e-9), arch
        assert abs(np.mean(s_json) - metrics['spatial_knn_uniform'][arch]['mean']['morans_i']) < 1e-12

        s_est = tract_bootstrap(s_mi.mean(axis=0), rng)
        c_est = tract_bootstrap(c_mi.mean(axis=0), rng)
        cm = pd.DataFrame(c_mi.mean(axis=1).reshape(5, 5), index=TRAINING_SEEDS,
                          columns=GRAPH_DRAW_SEEDS)
        vc = crossed_variance_components(cm)
        s_var_ts = float(np.var(s_mi.mean(axis=1), ddof=1))
        res[arch] = (s_est, c_est)
        record('3a', f'morans_i_structured_{arch}', s_est[0], SRC['m5b_metrics'],
               ci=s_est[1:], ci_source=SRC['m5b_pred'], n_cells=5, n_tracts=len(tracts),
               var_training_seed=s_var_ts, sd_training_seed=s_var_ts ** 0.5)
        record('3a', f'morans_i_randomized_crossed_{arch}', c_est[0], SRC['crossed_trials'],
               ci=c_est[1:], ci_source=SRC['crossed_pred'], n_cells=25, n_tracts=len(tracts),
               variance_components=vc)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4))
    for k, arch in enumerate(ARCHS):
        for j, (label, est) in enumerate(zip(['Structured\n(k-NN, k=10)', 'Randomized\n(5 x 5)'],
                                             res[arch])):
            x = j + (k - 0.5) * 0.22
            ax1.errorbar(x, est[0], yerr=[[est[0] - est[1]], [est[2] - est[0]]], fmt='o',
                         ms=7, color=ARCH_COLOR[arch], elinewidth=1.5, capsize=0,
                         mfc=ARCH_COLOR[arch] if j == 0 else '#fcfcfb', mew=1.8)
            # labels on the outer side of each pair so they clear the other marker
            ax1.annotate(f'{round3(est[0]):.3f}', (x, est[0]), xytext=(-8 if k == 0 else 8, 0),
                         textcoords='offset points', va='center', ha='right' if k == 0 else 'left',
                         color=TEXT, fontsize=8)
        ax1.plot([], [], 'o', color=ARCH_COLOR[arch], label=ARCH_LABEL[arch])
    ax1.set_xticks([0, 1], ['Structured\n(k-NN, k=10)', 'Randomized\n(5 x 5)'])
    ax1.set_xlim(-0.6, 1.6)
    ax1.set_ylabel("Output Moran's I (tract mean)")
    ax1.set_title("a  Output Moran's I", loc='left', color=TEXT)
    ax1.legend(frameon=False, loc='lower left')

    cmeans = pd.read_csv(REPO_ROOT / SRC['cond_means'])
    conds = [('5b', 'production', 'Production'), ('5b', 'road_network_uniform', 'Road k-NN'),
             ('5b', 'spatial_knn_uniform', 'Spatial k-NN'), ('crossed', 'randomized', 'Randomized\n(5 x 5)')]
    vals = []
    for k, arch in enumerate(ARCHS):
        for j, (st, cond, label) in enumerate(conds):
            r = cmeans[(cmeans['set'] == st) & (cmeans['condition'] == cond) & (cmeans['arch'] == arch)].iloc[0]
            x = j + (k - 0.5) * 0.22
            ax2.plot(x, r['mean_v2'], 'o', ms=7, color=ARCH_COLOR[arch])
            vals.append(r['mean_v2'])
            record('3b', f'pooled_bg_r_{cond}_{arch}', r['mean_v2'], SRC['cond_means'],
                   set=st, n_cells=int(r['n_cells']), sd_across_cells=float(r['sd_v2']),
                   n_bgs=int(r['n_bgs_v2']), intervals='absent from source file')
        ax2.plot([], [], 'o', color=ARCH_COLOR[arch], label=ARCH_LABEL[arch])
    record('3b', 'pooled_bg_r_range_min', min(vals), SRC['cond_means'])
    record('3b', 'pooled_bg_r_range_max', max(vals), SRC['cond_means'])
    ax2.set_xticks(range(4), [c[2] for c in conds])
    ax2.set_ylabel('Pooled block-group r (v2)')
    ax2.set_title('b  Pooled block-group r by graph', loc='left', color=TEXT)
    ax2.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(PNG_DIR / 'fig3.png', dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# figure 4
# ---------------------------------------------------------------------------

def figure4(rng, tracts):
    grid = pd.read_csv(REPO_ROOT / SRC['grid'], dtype={'tract_fips': str})
    ceil = grid[(grid['method'] == 'ceiling_gbm') & (grid['autocorr'] == 'strong') &
                (grid['snr'] == 'medium') & (grid['between_tract'] == 'default') &
                (grid['seed'] == 42)]
    assert len(ceil) == 20
    ceiling = float(ceil['recovery_r'].mean())
    record('4a', 'ceiling_gbm_seed42_tract_mean', ceiling, SRC['grid'],
           cell='latent/strong/medium/default', seed=42, n_tracts=20)

    have_4a = (REPO_ROOT / SRC['fig4_tracts']).exists() and (REPO_ROOT / SRC['fig4_cells']).exists()
    status_4a = 'built'
    res = {}
    if have_4a:
        rows = pd.read_csv(REPO_ROOT / SRC['fig4_tracts'], dtype={'tract_fips': str})
        cells = pd.read_csv(REPO_ROOT / SRC['fig4_cells'])
        for arch in ARCHS:
            for cond, n_exp in (('spatial_knn_uniform', 5), ('randomized', 25)):
                c = cells[(cells['arch'] == arch) & (cells['condition'] == cond)]
                r = rows[(rows['arch'] == arch) & (rows['condition'] == cond)]
                if len(c) < n_exp:
                    status_4a = 'incomplete'
                    continue
                for metric in ('recovery_r', 'morans_i_output'):
                    piv = r.pivot_table(index=['training_seed', 'graph_draw_seed'],
                                        columns='tract_fips', values=metric)
                    assert piv.shape == (n_exp, len(tracts))
                    est = tract_bootstrap(piv.mean(axis=0).to_numpy(), rng)
                    extra = {'n_cells': n_exp, 'n_tracts': len(tracts),
                             'runtime_s_total': float(c['runtime_s'].sum()),
                             'runtime_s_mean_per_cell': float(c['runtime_s'].mean())}
                    cm = piv.mean(axis=1)
                    if cond == 'randomized':
                        extra['variance_components'] = crossed_variance_components(
                            cm.unstack('graph_draw_seed').loc[TRAINING_SEEDS, GRAPH_DRAW_SEEDS])
                    else:
                        extra['var_training_seed'] = float(np.var(cm.to_numpy(), ddof=1))
                    record('4a', f'{metric}_{cond}_{arch}', est[0], SRC['fig4_tracts'],
                           ci=est[1:], **extra)
                    res[(arch, cond, metric)] = est
    else:
        status_4a = 'absent'

    # input Moran's I of the seed-42 field: current estimator recomputed from the committed
    # field, next to the committed morans_i_true column (pre-826c61a estimator), 20 tracts
    field = pd.read_csv(REPO_ROOT / SRC['field'], dtype={'fips': str})
    diag = SpatialLearningDiagnostics(verbose=False)
    cur = [diag.compute_spatial_autocorrelation(
        field.loc[field['fips'] == f, 'y_true'].to_numpy(float),
        field.loc[field['fips'] == f, ['x', 'y']].to_numpy(float), k_neighbors=8) for f in tracts]
    old = ceil.set_index('tract_fips').loc[tracts, 'morans_i_true'].to_numpy(float)
    record('4a', 'field_seed42_morans_i_true_current_estimator', float(np.mean(cur)), SRC['field'],
           estimator='current (826c61a and later): libpysal KNN k=8 on generator UTM x, y, '
                     "transform 'r', esda.Moran permutations=0",
           aggregation='unweighted mean over the 20 Paper 1 tracts', n_tracts=len(tracts))
    record('4a', 'field_seed42_morans_i_true_committed_old_estimator', float(np.mean(old)),
           SRC['grid'], estimator='pre-826c61a (inverse-distance k=8, 2dc231f), as committed in '
                                  'recovery_grid.csv morans_i_true; not rescored',
           aggregation='unweighted mean over the 20 Paper 1 tracts', n_tracts=len(tracts))

    sub = grid[(grid['method'] == 'granite') & (grid['feature_mode'] == 'coordinates_only') &
               (grid['arch'] == 'sage')]
    draw = sub.groupby(['autocorr', 'snr', 'between_tract', 'seed'])[['morans_i_output', 'recovery_r']].mean()
    assert len(draw) == 81
    mi_med = float(draw['morans_i_output'].median())
    rr_med = float(draw['recovery_r'].median())
    record('4b', 'morans_i_output_median_of_draw_means', mi_med, SRC['grid'], n_draws=81,
           aggregation='per-draw mean over 20 tracts, median over 81 draws')
    record('4b', 'recovery_r_median_of_draw_means', rr_med, SRC['grid'], n_draws=81,
           aggregation='per-draw mean over 20 tracts, median over 81 draws')

    ncol = 3 if status_4a == 'built' else 1
    fig, axes = plt.subplots(1, ncol, figsize=(4.2 * ncol, 3.4))
    axes = np.atleast_1d(axes)
    if status_4a == 'built':
        for ax, metric, ylab, title in ((axes[0], 'recovery_r', 'Within-tract recovery r',
                                         'a  Recovery against known truth'),
                                        (axes[1], 'morans_i_output', "Output Moran's I",
                                         "a  Output Moran's I")):
            for k, arch in enumerate(ARCHS):
                for j, cond in enumerate(('spatial_knn_uniform', 'randomized')):
                    est = res[(arch, cond, metric)]
                    x = j + (k - 0.5) * 0.22
                    ax.errorbar(x, est[0], yerr=[[est[0] - est[1]], [est[2] - est[0]]], fmt='o',
                                ms=7, color=ARCH_COLOR[arch], elinewidth=1.5, capsize=0,
                                mfc=ARCH_COLOR[arch] if j == 0 else '#fcfcfb', mew=1.8)
                ax.plot([], [], 'o', color=ARCH_COLOR[arch], label=ARCH_LABEL[arch])
            if metric == 'recovery_r':
                ax.axhline(ceiling, color=TEXT2, linestyle='--', linewidth=1)
                ax.annotate(f'coordinate ceiling {round3(ceiling):.3f}', (1.55, ceiling),
                            xytext=(0, 4), textcoords='offset points', ha='right',
                            color=TEXT2, fontsize=8)
                ax.axhline(0, color=TEXT2, linewidth=0.6)
            ax.set_xticks([0, 1], ['Structured', 'Randomized'])
            ax.set_xlim(-0.5, 1.6)
            ax.set_ylabel(ylab)
            ax.set_title(title, loc='left', color=TEXT)
        axes[0].legend(frameon=False, loc='center right')
    axb = axes[-1]
    axb.scatter(draw['recovery_r'], draw['morans_i_output'], s=14, color='#2a78d6',
                edgecolors='#fcfcfb', linewidths=0.6)
    axb.axvline(rr_med, color=TEXT2, linestyle='--', linewidth=0.8)
    axb.axhline(mi_med, color=TEXT2, linestyle='--', linewidth=0.8)
    axb.annotate(f"median I {round3(mi_med):.3f}", (1, mi_med), xycoords=('axes fraction', 'data'),
                 xytext=(0, -10), textcoords='offset points', ha='right', color=TEXT2, fontsize=8)
    axb.set_xlabel('Within-tract recovery r (draw mean)')
    axb.set_ylabel("Output Moran's I (draw mean)")
    axb.set_title('b  Coordinate-only grid\n    GraphSAGE, 81 draws', loc='left', color=TEXT)
    fig.tight_layout()
    fig.savefig(PNG_DIR / 'fig4.png', dpi=200)
    plt.close(fig)
    return status_4a, mi_med, rr_med


def main():
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(BOOT_SEED)
    tracts = load_tracts()
    assert len(tracts) == 20
    figure1()
    figure2()
    figure3(rng, tracts)
    status_4a, mi_med, rr_med = figure4(rng, tracts)

    out = {
        'milestone': 'FIG-P1 B6',
        'command': 'python scripts/build_paper1_figures.py',
        'build_commit': source_commit('.'),
        'rounding': '3 decimals, ROUND_HALF_UP via decimal on repr(float)',
        'bootstrap': {'unit': 'tract (20 tracts, resampled with replacement)', 'n_resamples': N_BOOT,
                      'interval': 'percentile 95%', 'rng': 'numpy default_rng', 'seed': BOOT_SEED},
        'figure_4a_status': status_4a,
        'figure_2b': 'not drawn (equivalence margin pending with Dr. Gao)',
        'numbers': NUMBERS,
    }
    NUMBERS_JSON.write_text(json.dumps(out, indent=2) + '\n')
    for n in NUMBERS:
        ci = f"  [{n['ci95'][0]:.5f}, {n['ci95'][1]:.5f}]" if 'ci95' in n else ''
        print(f"fig {n['figure']:<3} {n['name']:<48} {n['value']!r}{ci}")
    print(f"figure 4a: {status_4a}")
    by_name = {n['name']: n['value'] for n in NUMBERS}
    checks = {
        '4b morans_i_output median of draw means': (mi_med, f'{mi_med:.5f}', '0.94073'),
        '4b recovery_r median of draw means': (rr_med, f'{rr_med:.1e}', '8.5e-05'),
        '3a randomized crossed GraphSAGE': (by_name['morans_i_randomized_crossed_sage'],
                                            f"{by_name['morans_i_randomized_crossed_sage']:.5f}",
                                            '0.50367'),
        '3a randomized crossed GCN-GAT': (by_name['morans_i_randomized_crossed_gcn_gat'],
                                          f"{by_name['morans_i_randomized_crossed_gcn_gat']:.5f}",
                                          '0.07112'),
    }
    for label, (v, shown, expected) in checks.items():
        print(f"check {label}: {v!r} -> {shown} (expected {expected})")
        assert shown == expected, label
    print(f"wrote {NUMBERS_JSON.relative_to(REPO_ROOT)} and {PNG_DIR.relative_to(REPO_ROOT)}/fig1-4.png")


if __name__ == '__main__':
    main()
