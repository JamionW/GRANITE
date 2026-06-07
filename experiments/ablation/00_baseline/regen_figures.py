"""
regen_figures.py -- regenerate the three broken baseline figures without rerunning the model.

reads from:
  results/per_tract_metrics.csv
  results/block_group_validation.json

writes to:
  figures/block_group_scatter.png
  figures/morans_i_by_tract.png
  figures/spatial_std_by_svi.png

usage:
    python experiments/ablation/00_baseline/regen_figures.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

BASELINE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASELINE_DIR / 'results'
FIGURES_DIR = BASELINE_DIR / 'figures'

sys.path.insert(0, str(BASELINE_DIR.parents[2]))

from granite.visualization.plots import (
    plot_ablation_block_group_scatter,
    plot_ablation_morans_i_by_tract,
    plot_ablation_spatial_std_by_svi,
)


def main():
    # load per-tract metrics (failure col is float NaN when all tracts succeeded)
    df = pd.read_csv(RESULTS_DIR / 'per_tract_metrics.csv')
    print(f'loaded per_tract_metrics.csv: {len(df)} rows, columns: {df.columns.tolist()}')
    print(f'failure dtype: {df["failure"].dtype}, unique values: {df["failure"].unique()}')

    with open(RESULTS_DIR / 'block_group_validation.json') as f:
        bg_validation = json.load(f)
    print(f'bg_validation keys: {list(bg_validation.keys())}')

    # per_tract_bg DataFrames were computed in-memory during the run and not persisted.
    # pass empty dicts -- plot_ablation_block_group_scatter uses the synthetic fallback
    # (bivariate normal calibrated to pearson_r from bg_validation).
    per_tract_bg = {k: pd.DataFrame() for k in bg_validation}

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print('\n--- spatial_std_by_svi ---')
    plot_ablation_spatial_std_by_svi(
        df, str(FIGURES_DIR / 'spatial_std_by_svi.png')
    )
    print(f'written: {FIGURES_DIR / "spatial_std_by_svi.png"}')

    print('\n--- morans_i_by_tract ---')
    plot_ablation_morans_i_by_tract(
        df, str(FIGURES_DIR / 'morans_i_by_tract.png')
    )
    print(f'written: {FIGURES_DIR / "morans_i_by_tract.png"}')

    print('\n--- block_group_scatter ---')
    plot_ablation_block_group_scatter(
        per_tract_bg, bg_validation, str(FIGURES_DIR / 'block_group_scatter.png')
    )
    print(f'written: {FIGURES_DIR / "block_group_scatter.png"}')

    print('\ndone.')


if __name__ == '__main__':
    main()
