"""
Produce four figures for the 20-tract rank consistency experiment.

  a) rho_heatmap_sage.png, rho_heatmap_gcngat.png
  b) rho_distributions.png
  c) feature_importance_per_tract.png
  d) sensitivity_grid.png

All written to results/rank_consistency_n20/figures/.

Usage
-----
    python scripts/make_rank_consistency_figures_n20.py \
        --results-dir results/rank_consistency_n20 \
        --base-dir output/rank_consistency_run \
        --sens1-dir results/rank_consistency_n20_sens1 \
        --sens2-dir results/rank_consistency_n20_sens2
"""
import argparse
import sys
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

sys.path.insert(0, '/workspaces/GRANITE')


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_summary_and_rho(results_dir):
    """Load feature_summary.csv and per_tract_rho.csv from a results directory."""
    rd = Path(results_dir)
    summary = pd.read_csv(rd / 'feature_summary.csv')
    per_tract = pd.read_csv(rd / 'per_tract_rho.csv', dtype={'fips': str})
    return summary, per_tract


def _load_svi_map(base_dir):
    """Build {fips: svi} from all tract directories in base_dir/graphsage/."""
    svi_map = {}
    inv_path = Path('./tract_inventory.csv')
    if inv_path.exists():
        df = pd.read_csv(inv_path, dtype={'fips': str})
        svi_map = dict(zip(df['fips'], df['svi']))
    return svi_map


def _top_features_by_mean_abs_rho(per_tract_df, arch, n=20):
    """Return top-n feature names sorted by mean absolute rho for arch."""
    sub = per_tract_df[per_tract_df['architecture'] == arch]
    feat_mean = sub.groupby('feature')['rho'].apply(lambda x: np.mean(np.abs(x)))
    return feat_mean.nlargest(n).index.tolist()


# ---------------------------------------------------------------------------
# figure a: rho heatmaps
# ---------------------------------------------------------------------------

def make_rho_heatmap(per_tract_df, arch, svi_map, out_path):
    """Heatmap: rows = all 73 features sorted by mean |rho| desc,
       columns = tracts ordered by SVI asc."""
    sub = per_tract_df[per_tract_df['architecture'] == arch].copy()
    if sub.empty:
        print(f"  [{arch}] no data for heatmap")
        return

    # pivot to (feature x fips) rho matrix
    pivot = sub.pivot_table(index='feature', columns='fips', values='rho', aggfunc='mean')

    # order columns by SVI ascending
    fips_in_pivot = list(pivot.columns)
    fips_svi = [(f, svi_map.get(f, 0.5)) for f in fips_in_pivot]
    fips_svi.sort(key=lambda x: x[1])
    ordered_fips = [f for f, _ in fips_svi]
    ordered_svi = [s for _, s in fips_svi]

    pivot = pivot.reindex(columns=ordered_fips)

    # order rows by mean absolute rho descending
    mean_abs = pivot.abs().mean(axis=1)
    pivot = pivot.reindex(mean_abs.sort_values(ascending=False).index)

    n_feat, n_tracts = pivot.shape
    fig_h = max(8, n_feat * 0.22)
    fig_w = max(8, n_tracts * 0.55 + 3)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    data = pivot.values
    vmax = max(0.01, np.nanpercentile(np.abs(data), 95))
    cmap = plt.cm.RdBu_r
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax.imshow(data, aspect='auto', cmap=cmap, norm=norm,
                   interpolation='nearest')

    plt.colorbar(im, ax=ax, label='Spearman rho', fraction=0.02, pad=0.01)

    arch_label = 'GraphSAGE' if 'sage' in arch.lower() else 'GCN-GAT'
    ax.set_title(f'Within-tract rank consistency -- {arch_label}\n'
                 f'Spearman rho (feature vs predicted SVI), {n_tracts} tracts',
                 fontsize=11)

    ax.set_yticks(range(n_feat))
    ax.set_yticklabels(pivot.index.tolist(), fontsize=6)

    ax.set_xticks(range(n_tracts))
    svi_labels = [f'{fips[-4:]}\n{s:.2f}' for fips, s in zip(ordered_fips, ordered_svi)]
    ax.set_xticklabels(svi_labels, fontsize=7, rotation=45, ha='right')
    ax.set_xlabel('Tract (last 4 FIPS digits) / SVI', fontsize=9)
    ax.set_ylabel('Feature (sorted by mean |rho| desc)', fontsize=9)

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# figure b: rho distributions
# ---------------------------------------------------------------------------

def make_rho_distributions(per_tract_df, out_path):
    """Boxplots of per-tract rho for top 20 features, faceted by architecture."""
    archs = per_tract_df['architecture'].unique().tolist()

    # pick top 20 features by combined mean |rho| across both architectures
    feat_score = per_tract_df.groupby('feature')['rho'].apply(
        lambda x: np.mean(np.abs(x))
    )
    top20 = feat_score.nlargest(20).index.tolist()

    sub = per_tract_df[per_tract_df['feature'].isin(top20)].copy()

    n_arch = len(archs)
    fig, axes = plt.subplots(1, n_arch, figsize=(8 * n_arch, 8), sharey=True)
    if n_arch == 1:
        axes = [axes]

    for ax, arch in zip(axes, archs):
        arch_sub = sub[sub['architecture'] == arch]
        # order features by median rho for this arch
        feat_medians = arch_sub.groupby('feature')['rho'].median()
        ordered_feats = feat_medians.reindex(top20).sort_values(ascending=False).index.tolist()

        data_list = []
        for feat in ordered_feats:
            vals = arch_sub[arch_sub['feature'] == feat]['rho'].values
            data_list.append(vals if len(vals) > 0 else np.array([np.nan]))

        bp = ax.boxplot(data_list, vert=False, patch_artist=True,
                        medianprops={'color': 'black', 'linewidth': 1.5},
                        flierprops={'marker': 'o', 'markersize': 4,
                                    'markerfacecolor': 'gray', 'alpha': 0.5})

        for patch in bp['boxes']:
            patch.set(facecolor='steelblue', alpha=0.5)

        # overlay individual tract points
        for i, feat in enumerate(ordered_feats):
            vals = arch_sub[arch_sub['feature'] == feat]['rho'].values
            y = np.full(len(vals), i + 1)
            jitter = np.random.default_rng(42).uniform(-0.2, 0.2, len(vals))
            ax.scatter(vals, y + jitter, color='tomato', alpha=0.6, s=15, zorder=5)

        ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_yticks(range(1, len(ordered_feats) + 1))
        ax.set_yticklabels(ordered_feats, fontsize=7)
        arch_label = 'GraphSAGE' if 'sage' in arch.lower() else 'GCN-GAT'
        ax.set_title(f'{arch_label}\ntop 20 features by mean |rho|', fontsize=10)
        ax.set_xlabel('Spearman rho (per tract)', fontsize=9)
        ax.grid(axis='x', alpha=0.3)

    fig.suptitle('Per-tract rho distributions -- top 20 features', fontsize=12, y=1.01)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# figure c: feature importance per tract
# ---------------------------------------------------------------------------

def _load_permutation_importance(base_dir, arch_dir_name):
    """Load permutation_importance.csv for all tracts in one architecture dir.

    Returns {fips: DataFrame with columns [feature, importance]}.
    """
    arch_path = Path(base_dir) / arch_dir_name
    result = {}
    if not arch_path.is_dir():
        return result
    for td in sorted(arch_path.glob('tract_*')):
        fips = td.name.replace('tract_', '')
        imp_path = td / 'feature_importance' / 'permutation_importance.csv'
        if not imp_path.exists():
            continue
        try:
            df = pd.read_csv(imp_path)
            if 'feature' in df.columns and 'importance' in df.columns:
                result[fips] = df[['feature', 'importance']].copy()
        except Exception:
            pass
    return result


def make_feature_importance_per_tract(base_dir, out_path, svi_map, n_features=10):
    """4x5 grid of bar plots. Each cell = one tract, bars = top-10 features,
       hue = architecture (graphsage vs gcn_gat).

    Reads existing permutation_importance.csv files from output directories.
    """
    sage_imp = _load_permutation_importance(base_dir, 'graphsage')
    gcn_imp = _load_permutation_importance(base_dir, 'gcn_gat')

    all_fips = sorted(set(list(sage_imp.keys()) + list(gcn_imp.keys())),
                      key=lambda f: svi_map.get(f, 0.5))

    n_tracts = len(all_fips)
    if n_tracts == 0:
        print("  no permutation importance data found; skipping figure c")
        return

    n_cols = 5
    n_rows = int(np.ceil(n_tracts / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, 3 * n_rows))
    axes = np.array(axes).flatten()

    colors = {'graphsage': 'steelblue', 'gcn_gat': 'tomato'}

    for idx, fips in enumerate(all_fips):
        ax = axes[idx]
        svi = svi_map.get(fips, float('nan'))

        # collect top features across both architectures
        all_feats = set()
        for df in [sage_imp.get(fips), gcn_imp.get(fips)]:
            if df is not None and not df.empty:
                top = df.nlargest(n_features, 'importance')['feature'].tolist()
                all_feats.update(top)

        if not all_feats:
            ax.set_visible(False)
            continue

        # union of top-n from each architecture; sort by sage importance
        ref_df = sage_imp.get(fips) if fips in sage_imp else gcn_imp.get(fips)
        if ref_df is None:
            ax.set_visible(False)
            continue

        merged = ref_df[ref_df['feature'].isin(all_feats)].copy()
        merged = merged.nlargest(n_features, 'importance')
        feat_order = merged['feature'].tolist()

        x = np.arange(len(feat_order))
        bar_w = 0.35

        for j, (arch_label, arch_key, imp_dict) in enumerate([
            ('SAGE', 'graphsage', sage_imp),
            ('GCN', 'gcn_gat', gcn_imp),
        ]):
            df = imp_dict.get(fips)
            if df is None:
                continue
            vals = []
            for feat in feat_order:
                row = df[df['feature'] == feat]
                vals.append(float(row['importance'].values[0]) if len(row) > 0 else 0.0)
            ax.bar(x + j * bar_w, vals, bar_w, label=arch_label,
                   color=colors[arch_key], alpha=0.75)

        ax.set_xticks(x + bar_w / 2)
        ax.set_xticklabels(feat_order, rotation=70, ha='right', fontsize=5)
        ax.set_title(f'{fips[-6:]}  SVI={svi:.2f}', fontsize=7)
        ax.set_ylabel('importance', fontsize=6)
        ax.tick_params(axis='y', labelsize=5)
        if idx == 0:
            ax.legend(fontsize=6)

    # hide unused panels
    for idx in range(n_tracts, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Permutation feature importance per tract (top 10)\n'
                 'blue=GraphSAGE  red=GCN-GAT', fontsize=11)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# figure d: sensitivity grid
# ---------------------------------------------------------------------------

def _parse_section_counts(summary_txt_path):
    """Parse section A/B/C/D counts from a summary.txt file.

    Returns dict with keys A, B, C, D.
    """
    counts = {'A': None, 'B': None, 'C': None, 'D': None}
    try:
        with open(summary_txt_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith('section A'):
                    counts['A'] = int(line.split()[-1])
                elif line.startswith('section B'):
                    counts['B'] = int(line.split()[-1])
                elif line.startswith('section C'):
                    counts['C'] = int(line.split()[-1])
                elif line.startswith('section D'):
                    counts['D'] = int(line.split()[-1])
    except FileNotFoundError:
        pass
    return counts


def make_sensitivity_grid(primary_dir, sens1_dir, sens2_dir, out_path):
    """Bar chart of section A/B/C/D counts across three sensitivity runs."""
    runs = [
        ('primary\ncv=0.10\nmin_t=12', primary_dir),
        ('sens1\ncv=0.20\nmin_t=10', sens1_dir),
        ('sens2\ncv=0.30\nmin_t=8',  sens2_dir),
    ]

    section_labels = ['A', 'B', 'C', 'D']
    colors_sec = ['steelblue', 'darkorange', 'seagreen', 'crimson']

    all_counts = []
    run_labels = []
    for label, rdir in runs:
        txt = Path(rdir) / 'summary.txt'
        c = _parse_section_counts(txt)
        all_counts.append(c)
        run_labels.append(label)

    x = np.arange(len(runs))
    bar_w = 0.18
    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (sec, color) in enumerate(zip(section_labels, colors_sec)):
        vals = [c[sec] if c[sec] is not None else 0 for c in all_counts]
        ax.bar(x + i * bar_w, vals, bar_w, label=f'Section {sec}', color=color, alpha=0.75)

        for xi, v in zip(x + i * bar_w, vals):
            if v is not None:
                ax.text(xi, v + 0.05, str(v), ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x + bar_w * 1.5)
    ax.set_xticklabels(run_labels, fontsize=9)
    ax.set_ylabel('Feature count', fontsize=10)
    ax.set_title('Section A/B/C/D counts across sensitivity runs\n'
                 '(cv_threshold and min_tracts variation)', fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', default='results/rank_consistency_n20')
    parser.add_argument('--base-dir', default='output/rank_consistency_run')
    parser.add_argument('--sens1-dir', default='results/rank_consistency_n20_sens1')
    parser.add_argument('--sens2-dir', default='results/rank_consistency_n20_sens2')
    args = parser.parse_args()

    fig_dir = Path(args.results_dir) / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading results from {args.results_dir}")
    try:
        summary, per_tract = _load_summary_and_rho(args.results_dir)
    except FileNotFoundError as e:
        sys.exit(f"ERROR: {e} -- run within_tract_rank_consistency.py first")

    svi_map = _load_svi_map(args.base_dir)
    print(f"svi_map: {len(svi_map)} tracts")
    print(f"per_tract rows: {len(per_tract)}")

    print("\nfigure a: rho heatmaps")
    make_rho_heatmap(per_tract, 'graphsage',
                     svi_map, fig_dir / 'rho_heatmap_sage.png')
    make_rho_heatmap(per_tract, 'gcn_gat',
                     svi_map, fig_dir / 'rho_heatmap_gcngat.png')

    print("\nfigure b: rho distributions")
    make_rho_distributions(per_tract, fig_dir / 'rho_distributions.png')

    print("\nfigure c: feature importance per tract")
    make_feature_importance_per_tract(
        args.base_dir, fig_dir / 'feature_importance_per_tract.png', svi_map
    )

    print("\nfigure d: sensitivity grid")
    make_sensitivity_grid(
        args.results_dir, args.sens1_dir, args.sens2_dir,
        fig_dir / 'sensitivity_grid.png'
    )

    print("\nall figures written to:")
    for p in sorted(fig_dir.glob('*.png')):
        print(f"  {p.absolute()}")


if __name__ == '__main__':
    main()
