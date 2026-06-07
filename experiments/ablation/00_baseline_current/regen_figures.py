"""
regen_figures.py -- regenerate the current-baseline comparison figure.

reads from:
  results/per_tract.csv    (GRANITE, Dasymetric, Pycnophylactic per-tract bg_r)
  results/aggregate.csv    (pooled metrics per method)

writes to:
  figures/bg_r_by_tract.png   (per-tract block-group r, three methods)
  figures/aggregate_summary.png  (pooled bg_r with 95% CI)

source: data/results/m0_n20_svi_parity/ (m0 n20 svi parity run)
current dissertation baselines: Dasymetric (headline) and Pycnophylactic (secondary).

usage:
    python experiments/ablation/00_baseline_current/regen_figures.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / 'results'
FIGURES_DIR = SCRIPT_DIR / 'figures'

_METHOD_ORDER = ['GRANITE', 'Dasymetric', 'Pycnophylactic']
_COLORS = {'GRANITE': '#2166ac', 'Dasymetric': '#d6604d', 'Pycnophylactic': '#4dac26'}


def _load():
    per_tract = pd.read_csv(RESULTS_DIR / 'per_tract.csv')
    aggregate = pd.read_csv(RESULTS_DIR / 'aggregate.csv')
    return per_tract, aggregate


def plot_bg_r_by_tract(per_tract, output_path):
    # drop tracts with fewer than 2 block groups (bg_r undefined)
    df = per_tract[per_tract['n_bgs'] >= 2].copy()
    tracts = sorted(df[df['method'] == 'GRANITE']['fips'].unique())

    fig, ax = plt.subplots(figsize=(12, 5))
    n = len(_METHOD_ORDER)
    width = 0.25
    x = np.arange(len(tracts))

    for i, method in enumerate(_METHOD_ORDER):
        subset = df[df['method'] == method].set_index('fips')
        vals = [subset.loc[t, 'bg_r'] if t in subset.index else np.nan for t in tracts]
        ax.bar(x + (i - 1) * width, vals, width, label=method,
               color=_COLORS[method], alpha=0.85)

    ax.axhline(0, color='black', linewidth=0.6, linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels([str(t)[-4:] for t in tracts], rotation=45, ha='right', fontsize=7)
    ax.set_xlabel('census tract (last 4 digits of FIPS)')
    ax.set_ylabel('block-group Pearson r')
    ax.set_title('per-tract block-group correlation: GRANITE vs current baselines\n'
                 '(n=19 tracts with >=2 block groups, m0 n20 svi parity run)')
    ax.legend(loc='upper right')
    ax.set_ylim(-1.1, 1.1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f'written: {output_path}')


def plot_aggregate_summary(aggregate, output_path):
    df = aggregate.set_index('method').reindex(_METHOD_ORDER)

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(_METHOD_ORDER))
    bars = ax.bar(x, df['pooled_bg_r'], color=[_COLORS[m] for m in _METHOD_ORDER],
                  alpha=0.85, width=0.5)

    # 95% CI error bars
    err_low = df['pooled_bg_r'] - df['pooled_bg_r_ci_low']
    err_high = df['pooled_bg_r_ci_high'] - df['pooled_bg_r']
    ax.errorbar(x, df['pooled_bg_r'],
                yerr=[err_low.values, err_high.values],
                fmt='none', color='black', capsize=5, linewidth=1.5)

    for bar, val in zip(bars, df['pooled_bg_r']):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.015,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(_METHOD_ORDER)
    ax.set_ylabel('pooled block-group Pearson r')
    ax.set_ylim(0, 1.05)
    ax.set_title('pooled block-group r with 95% CI\n'
                 f'(n={int(df["pooled_n_bgs"].iloc[0])} block groups, m0 n20 svi parity run)')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f'written: {output_path}')


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    per_tract, aggregate = _load()
    print(f'loaded per_tract.csv: {len(per_tract)} rows')
    print(f'loaded aggregate.csv: {len(aggregate)} rows')
    print(f'methods: {per_tract["method"].unique().tolist()}')

    print('\n--- bg_r_by_tract ---')
    plot_bg_r_by_tract(per_tract, FIGURES_DIR / 'bg_r_by_tract.png')

    print('\n--- aggregate_summary ---')
    plot_aggregate_summary(aggregate, FIGURES_DIR / 'aggregate_summary.png')

    print('\ndone.')


if __name__ == '__main__':
    main()
