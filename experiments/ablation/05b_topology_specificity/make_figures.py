"""
05b_topology_specificity: headline figure.

3x2 grid: rows = within_tract_std / morans_i / pooled_bg_r
          cols = SAGE / GCN-GAT
x-axis: spatial_knn_uniform, road_network_uniform, randomized
error bars: across-seed std (5 values)
horizontal reference lines: step-5 production and mlp_floor poles

Usage:
    python experiments/ablation/05b_topology_specificity/make_figures.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[3]
STEP5B_ROOT = Path(__file__).resolve().parent

RESULTS_PATH = STEP5B_ROOT / 'results' / 'topology_specificity_metrics.json'
PARITY_PATH  = STEP5B_ROOT / 'results' / 'degree_parity.json'
OUT_PATH     = STEP5B_ROOT / 'figures' / 'topology_specificity.png'

CONDITIONS   = ['spatial_knn_uniform', 'road_network_uniform', 'randomized']
COND_LABELS  = {
    'spatial_knn_uniform':  'spatial\nkNN',
    'road_network_uniform': 'road\nkNN',
    'randomized':           'randomized',
}
ARCHITECTURES = ['sage', 'gcn_gat']
ARCH_TITLES   = {'sage': 'GRANITE-SAGE', 'gcn_gat': 'GRANITE-GCNGAT'}

METRICS = ['within_tract_std', 'morans_i', 'pooled_bg_r']
METRIC_LABELS = {
    'within_tract_std': 'within-tract std',
    'morans_i':         "Moran's I",
    'pooled_bg_r':      'pooled BG r',
}

# step-5 reference poles (production=distance-weighted hybrid, mlp_floor=no-graph floor)
STEP5_REFS = {
    'sage': {
        'production': {'morans_i': 0.8570, 'within_tract_std': 0.0899, 'pooled_bg_r': 0.7632},
        'mlp_floor':  {'morans_i': 0.6820, 'within_tract_std': 0.0832, 'pooled_bg_r': 0.7714},
    },
    'gcn_gat': {
        'production': {'morans_i': 0.8368, 'within_tract_std': 0.0906, 'pooled_bg_r': 0.7639},
        'mlp_floor':  {'morans_i': 0.6747, 'within_tract_std': 0.0812, 'pooled_bg_r': 0.7660},
    },
}

COLORS = {
    'spatial_knn_uniform':  '#2166ac',
    'road_network_uniform': '#4dac26',
    'randomized':           '#d6604d',
}


def _val(d, keys, default=float('nan')):
    for k in keys:
        if d is None or not isinstance(d, dict):
            return default
        d = d.get(k, None)
    return d if d is not None else default


def main():
    if not RESULTS_PATH.exists():
        print(f'[make_figures] HALT: {RESULTS_PATH} missing; run run_sweep.py first')
        sys.exit(1)

    with open(RESULTS_PATH) as f:
        results = json.load(f)

    missing = [c for c in CONDITIONS if c not in results]
    if missing:
        print(f'[make_figures] WARNING: conditions missing from results: {missing}')
        print('[make_figures] continuing with available conditions')

    # load parity info if available
    parity = {}
    if PARITY_PATH.exists():
        with open(PARITY_PATH) as f:
            parity = json.load(f)

    fig, axes = plt.subplots(
        len(METRICS), len(ARCHITECTURES),
        figsize=(10, 11),
        sharex=False,
    )
    fig.suptitle('Step 5b: Topology Specificity Sweep\n(uniform edge weights, k=10)', fontsize=13)

    x_pos = np.arange(len(CONDITIONS))

    for col_idx, arch in enumerate(ARCHITECTURES):
        for row_idx, metric in enumerate(METRICS):
            ax = axes[row_idx][col_idx]

            means = []
            stds = []
            for cond in CONDITIONS:
                m = _val(results, [cond, arch, 'mean', metric])
                s = _val(results, [cond, arch, 'std',  metric])
                means.append(m if (m is not None and math.isfinite(m)) else float('nan'))
                stds.append(s  if (s is not None and math.isfinite(s)) else 0.0)

            colors = [COLORS[c] for c in CONDITIONS]
            ax.bar(x_pos, means, yerr=stds, color=colors,
                   capsize=5, width=0.55, error_kw={'linewidth': 1.5})

            # step-5 reference lines
            ref = STEP5_REFS[arch]
            prod_val = ref['production'].get(metric, float('nan'))
            mlp_val  = ref['mlp_floor'].get(metric,  float('nan'))
            if math.isfinite(prod_val):
                ax.axhline(prod_val, color='#555555', linestyle='--', linewidth=1.2,
                           label='step-5 production')
            if math.isfinite(mlp_val):
                ax.axhline(mlp_val, color='#aaaaaa', linestyle=':', linewidth=1.2,
                           label='step-5 mlp_floor')

            ax.set_xticks(x_pos)
            ax.set_xticklabels([COND_LABELS[c] for c in CONDITIONS], fontsize=9)
            ax.set_ylabel(METRIC_LABELS[metric], fontsize=9)

            if row_idx == 0:
                ax.set_title(ARCH_TITLES[arch], fontsize=11)

            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=8, loc='lower right')

    # degree / jaccard annotation
    degrees = parity.get('degrees', {})
    jaccard = parity.get('jaccard_spatial_road', float('nan'))
    ann_lines = []
    for cond in CONDITIONS:
        d = degrees.get(cond, float('nan'))
        if d is not None and math.isfinite(d):
            ann_lines.append(f'{cond}: deg={d:.2f}')
    if math.isfinite(jaccard):
        ann_lines.append(f'jaccard(spatial, road)={jaccard:.3f}')

    if ann_lines:
        fig.text(
            0.5, 0.01, '  |  '.join(ann_lines),
            ha='center', va='bottom', fontsize=8, color='#444444'
        )

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
    print(f'[make_figures] saved {OUT_PATH}')

    # print table for README
    print('\n[make_figures] === metrics table ===')
    header = f"{'condition':<22} {'arch':<9} {'moran_mean':>10} {'moran_std':>9} {'bg_r_mean':>9} {'bg_r_std':>8} {'std_mean':>9} {'std_std':>8}"
    print(header)
    for cond in CONDITIONS:
        for arch in ARCHITECTURES:
            mm = _val(results, [cond, arch, 'mean', 'morans_i'])
            ms = _val(results, [cond, arch, 'std',  'morans_i'])
            bm = _val(results, [cond, arch, 'mean', 'pooled_bg_r'])
            bs = _val(results, [cond, arch, 'std',  'pooled_bg_r'])
            sm = _val(results, [cond, arch, 'mean', 'within_tract_std'])
            ss = _val(results, [cond, arch, 'std',  'within_tract_std'])

            def _fmt(v):
                return f'{v:.4f}' if (v is not None and math.isfinite(v)) else '   nan'

            print(f'{cond:<22} {arch:<9} {_fmt(mm):>10} {_fmt(ms):>9} {_fmt(bm):>9} {_fmt(bs):>8} {_fmt(sm):>9} {_fmt(ss):>8}')


if __name__ == '__main__':
    main()
