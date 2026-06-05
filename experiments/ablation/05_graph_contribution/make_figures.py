"""
05_graph_contribution: headline figure.

3x2 grid: rows = within_tract_std / Moran's I / pooled BG r,
cols = SAGE / GCN-GAT, x = {production, mlp_floor}, error bars = across-seed std.
Production band drawn as a horizontal reference span on each panel.

Asserts that across-seed std derives from 5 values per condition, not 20.

Usage:
    python experiments/ablation/05_graph_contribution/make_figures.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ABLATION_DIR = Path(__file__).resolve().parent
RESULTS_PATH = ABLATION_DIR / 'results' / 'graph_contribution_metrics.json'
OUTPUT_PATH  = ABLATION_DIR / 'graph_contribution.png'

ARCHITECTURES = ['sage', 'gcn_gat']
ARCH_LABELS   = {'sage': 'GRANITE-SAGE', 'gcn_gat': 'GRANITE-GCNGAT'}
CONDITIONS    = ['production', 'mlp_floor']
COND_LABELS   = {'production': 'production', 'mlp_floor': 'mlp_floor'}
METRICS       = ['within_tract_std', 'morans_i', 'pooled_bg_r']
METRIC_LABELS = {
    'within_tract_std': 'within-tract std (mean over 20 tracts)',
    'morans_i':         "Moran's I (mean over 20 tracts)",
    'pooled_bg_r':      'pooled BG r (n_bgs=69)',
}
N_SEEDS = 5
COLORS = {'production': '#2166ac', 'mlp_floor': '#d6604d'}


def _load_results():
    if not RESULTS_PATH.exists():
        print(f'[make_figures] HALT: {RESULTS_PATH} not found. run run_sweep.py first.')
        sys.exit(1)
    with open(RESULTS_PATH) as f:
        return json.load(f)


def _extract(results, condition, arch, metric):
    """Return list of per-seed values for (condition, arch, metric)."""
    arch_res = results.get(condition, {}).get(arch, {})
    seeds = arch_res.get('seeds', [])
    vals = [s[metric] for s in seeds if s.get(metric) is not None and math.isfinite(s.get(metric, float('nan')))]
    assert len(vals) == N_SEEDS, (
        f'expected {N_SEEDS} seed values for {condition}/{arch}/{metric}, got {len(vals)}. '
        f'across-seed band must derive from {N_SEEDS} values, not 20 (across-tract).'
    )
    return vals


def main():
    results = _load_results()

    fig, axes = plt.subplots(3, 2, figsize=(10, 11))
    fig.suptitle('Step 5: graph contribution boundary test\nproduction vs mlp_floor (self-loops only)',
                 fontsize=12, y=0.98)

    x_positions = {c: i for i, c in enumerate(CONDITIONS)}

    for col, arch in enumerate(ARCHITECTURES):
        for row, metric in enumerate(METRICS):
            ax = axes[row, col]

            # production reference band (mean +/- std across seeds)
            prod_vals = _extract(results, 'production', arch, metric)
            prod_mean = float(np.mean(prod_vals))
            prod_std  = float(np.std(prod_vals, ddof=1)) if len(prod_vals) > 1 else 0.0
            ax.axhspan(
                prod_mean - prod_std, prod_mean + prod_std,
                alpha=0.15, color=COLORS['production'], label='production seed band'
            )
            ax.axhline(prod_mean, color=COLORS['production'], linewidth=0.8, linestyle='--', alpha=0.6)

            # points + error bars for each condition
            for cond in CONDITIONS:
                vals = _extract(results, cond, arch, metric)
                mean = float(np.mean(vals))
                std  = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                x    = x_positions[cond]
                ax.errorbar(
                    x, mean, yerr=std,
                    fmt='o', color=COLORS[cond], markersize=7,
                    capsize=4, linewidth=1.5,
                    label=f'{COND_LABELS[cond]} (n={N_SEEDS} seeds)'
                )

            ax.set_xticks(list(x_positions.values()))
            ax.set_xticklabels(list(x_positions.keys()), fontsize=9)
            ax.set_xlim(-0.5, len(CONDITIONS) - 0.5)

            if col == 0:
                ax.set_ylabel(METRIC_LABELS[metric], fontsize=8)
            if row == 0:
                ax.set_title(ARCH_LABELS[arch], fontsize=10)
            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc='best')

            ax.grid(axis='y', linestyle=':', alpha=0.5)
            ax.tick_params(labelsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[make_figures] saved {OUTPUT_PATH}')

    # summary table to stdout
    print('\n=== metrics table (mean +/- across-seed std) ===')
    header = f'{"condition":<12} {"arch":<8} {"within_std":>12} {"morans_i":>12} {"pooled_bg_r":>13}'
    print(header)
    print('-' * len(header))
    for cond in CONDITIONS:
        for arch in ARCHITECTURES:
            row_parts = [f'{cond:<12}', f'{arch:<8}']
            for metric in METRICS:
                vals = _extract(results, cond, arch, metric)
                m = np.mean(vals)
                s = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
                row_parts.append(f'{m:.4f}+/-{s:.4f}'.rjust(12 if metric != "pooled_bg_r" else 13))
            print(' '.join(row_parts))

    # verdict paragraph
    print('\n=== verdict ===')
    for arch in ARCHITECTURES:
        prod_vals = {m: _extract(results, 'production', arch, m) for m in METRICS}
        mlp_vals  = {m: _extract(results, 'mlp_floor',  arch, m) for m in METRICS}
        for metric in ['morans_i', 'within_tract_std']:
            prod_m = np.mean(prod_vals[metric])
            prod_s = np.std(prod_vals[metric], ddof=1)
            mlp_m  = np.mean(mlp_vals[metric])
            # within production band if mlp_m is within prod_m +/- prod_s
            in_band = abs(mlp_m - prod_m) <= prod_s
            verdict = 'within production band (GRAPH DECORATIVE)' if in_band else 'outside production band (GRAPH CONTRIBUTES)'
            print(f'{arch} / {metric}: mlp_floor={mlp_m:.4f} prod={prod_m:.4f}+/-{prod_s:.4f} -> {verdict}')


if __name__ == '__main__':
    main()
