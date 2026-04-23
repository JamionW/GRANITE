"""
Cross-condition summary for the coordinate-artifact experiment.

Reads per-tract CSV outputs from coord_artifact_experiment.py and produces:
  output/coord_artifact_test/summary/cross_mode_metrics.csv    (already written by experiment)
  output/coord_artifact_test/summary/prediction_correlations.csv (already written by experiment)
  output/coord_artifact_test/summary/cross_mode_dashboard.png

Dashboard panels
----------------
A - spatial std per mode, grouped by tract
B - Moran's I per mode, grouped by tract
C - average prediction-prediction r across tracts (mode x mode heatmap)
D - per-tract constraint error by mode

Run after coord_artifact_experiment.py completes.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = './output/coord_artifact_test'
SUMMARY_DIR = os.path.join(BASE_DIR, 'summary')
FEATURE_MODES = ['full', 'coordinates_only', 'random_noise', 'coords_plus_noise']
MODE_LABELS = {
    'full': 'Full',
    'coordinates_only': 'Coords only',
    'random_noise': 'Random noise',
    'coords_plus_noise': 'Coords+noise',
}


def load_metrics():
    path = os.path.join(SUMMARY_DIR, 'cross_mode_metrics.csv')
    if not os.path.exists(path):
        # reconstruct from per-tract results_summary.json files if CSV missing
        rows = []
        for mode in FEATURE_MODES:
            mode_dir = os.path.join(BASE_DIR, mode)
            if not os.path.isdir(mode_dir):
                continue
            for tract_dir in os.listdir(mode_dir):
                if not tract_dir.startswith('tract_'):
                    continue
                fips = tract_dir[len('tract_'):]
                summary_path = os.path.join(mode_dir, tract_dir, 'results_summary.json')
                metrics_path = os.path.join(mode_dir, tract_dir, 'metrics.csv')
                if os.path.exists(metrics_path):
                    df = pd.read_csv(metrics_path)
                    row = {
                        'fips': fips,
                        'tract_svi': df.get('tract_svi', [float('nan')])[0]
                            if 'tract_svi' in df.columns else float('nan'),
                        'mode': mode,
                        'mean': df['predicted_mean'].iloc[0]
                            if 'predicted_mean' in df.columns else float('nan'),
                        'std': df['predicted_std'].iloc[0]
                            if 'predicted_std' in df.columns else float('nan'),
                        'range': df['predicted_range'].iloc[0]
                            if 'predicted_range' in df.columns else float('nan'),
                        'constraint_error_pct': df['constraint_error_pct'].iloc[0]
                            if 'constraint_error_pct' in df.columns else float('nan'),
                        'moran_i': df['moran_i'].iloc[0]
                            if 'moran_i' in df.columns else float('nan'),
                    }
                    rows.append(row)
                elif os.path.exists(summary_path):
                    import json
                    with open(summary_path) as f:
                        s = json.load(f)
                    row = {
                        'fips': fips,
                        'tract_svi': s.get('tract_svi', float('nan')),
                        'mode': mode,
                        'mean': s.get('predicted_mean', float('nan')),
                        'std': s.get('predicted_std', float('nan')),
                        'range': s.get('predicted_range', float('nan')),
                        'constraint_error_pct': s.get('constraint_error_pct', float('nan')),
                        'moran_i': s.get('moran_i', float('nan')),
                    }
                    rows.append(row)
        if rows:
            df_out = pd.DataFrame(rows)
            os.makedirs(SUMMARY_DIR, exist_ok=True)
            df_out.to_csv(path, index=False)
            return df_out
        else:
            print(f"ERROR: no metrics found at {path} and no per-tract fallback available.")
            sys.exit(1)
    return pd.read_csv(path)


def load_correlations():
    path = os.path.join(SUMMARY_DIR, 'prediction_correlations.csv')
    if not os.path.exists(path):
        # reconstruct from per-tract address CSVs
        rows = []
        for mode in FEATURE_MODES:
            mode_dir = os.path.join(BASE_DIR, mode)
            if not os.path.isdir(mode_dir):
                continue
            for tract_dir in os.listdir(mode_dir):
                if not tract_dir.startswith('tract_'):
                    continue
                fips = tract_dir[len('tract_'):]
                addr_csv = os.path.join(mode_dir, tract_dir, 'address_predictions.csv')
                if os.path.exists(addr_csv):
                    df = pd.read_csv(addr_csv)
                    if 'mean' in df.columns:
                        pred_vecs[fips][mode] = df['mean'].values
        # compute pairwise correlations
        from scipy.stats import pearsonr
        all_fips = list(pred_vecs.keys())
        for fips in all_fips:
            for i, ma in enumerate(FEATURE_MODES):
                for mb in FEATURE_MODES[i+1:]:
                    va = pred_vecs[fips].get(ma)
                    vb = pred_vecs[fips].get(mb)
                    if va is None or vb is None or len(va) != len(vb):
                        continue
                    r, _ = pearsonr(va, vb)
                    rows.append({'fips': fips, 'mode_a': ma, 'mode_b': mb, 'pearson_r': float(r)})
        if rows:
            df_out = pd.DataFrame(rows)
            os.makedirs(SUMMARY_DIR, exist_ok=True)
            df_out.to_csv(path, index=False)
            return df_out
        return pd.DataFrame(columns=['fips', 'mode_a', 'mode_b', 'pearson_r'])
    return pd.read_csv(path)


def build_corr_matrix(corr_df):
    """Build symmetric NxN mean-r matrix across tracts for all mode pairs."""
    n = len(FEATURE_MODES)
    mat = np.eye(n)
    mode_idx = {m: i for i, m in enumerate(FEATURE_MODES)}
    for _, row in corr_df.iterrows():
        i = mode_idx.get(row['mode_a'])
        j = mode_idx.get(row['mode_b'])
        if i is None or j is None:
            continue
        # accumulate mean r across tracts (fill both triangles symmetrically)
        mat[i, j] = float('nan')
        mat[j, i] = float('nan')

    # compute mean per pair
    for i, ma in enumerate(FEATURE_MODES):
        for j, mb in enumerate(FEATURE_MODES):
            if i == j:
                mat[i, j] = 1.0
                continue
            if i > j:
                mat[i, j] = mat[j, i]
                continue
            subset = corr_df[
                ((corr_df['mode_a'] == ma) & (corr_df['mode_b'] == mb)) |
                ((corr_df['mode_a'] == mb) & (corr_df['mode_b'] == ma))
            ]
            if len(subset) == 0:
                mat[i, j] = float('nan')
            else:
                mat[i, j] = subset['pearson_r'].mean()
    # fill lower triangle
    for i in range(n):
        for j in range(i):
            mat[i, j] = mat[j, i]
    return mat


def make_dashboard(metrics_df, corr_df):
    all_fips = sorted(metrics_df['fips'].unique())
    n_tracts = len(all_fips)
    x = np.arange(n_tracts)
    width = 0.2
    labels = [MODE_LABELS[m] for m in FEATURE_MODES]
    colors = ['#333333', '#1565C0', '#C62828', '#2E7D32']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Coordinate Artifact Experiment: Cross-Mode Comparison', fontsize=13)

    # panel A: spatial std
    ax = axes[0, 0]
    for k, (mode, color) in enumerate(zip(FEATURE_MODES, colors)):
        vals = []
        for fips in all_fips:
            row = metrics_df[(metrics_df['fips'] == fips) & (metrics_df['mode'] == mode)]
            vals.append(row['std'].iloc[0] if len(row) > 0 else float('nan'))
        ax.bar(x + k * width, vals, width, label=labels[k], color=color)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([str(f)[:11] for f in all_fips], rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Spatial std')
    ax.set_title('A. Spatial std by mode and tract')
    ax.legend(fontsize=8)

    # panel B: Moran's I
    ax = axes[0, 1]
    for k, (mode, color) in enumerate(zip(FEATURE_MODES, colors)):
        vals = []
        for fips in all_fips:
            row = metrics_df[(metrics_df['fips'] == fips) & (metrics_df['mode'] == mode)]
            vals.append(row['moran_i'].iloc[0] if len(row) > 0 else float('nan'))
        ax.bar(x + k * width, vals, width, label=labels[k], color=color)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([str(f)[:11] for f in all_fips], rotation=30, ha='right', fontsize=8)
    ax.set_ylabel("Moran's I")
    ax.set_title("B. Moran's I by mode and tract")
    ax.legend(fontsize=8)

    # panel C: cross-mode correlation heatmap (averaged over tracts)
    ax = axes[1, 0]
    if len(corr_df) > 0:
        mat = build_corr_matrix(corr_df)
        im = ax.imshow(mat, vmin=-1, vmax=1, cmap='RdBu_r')
        ax.set_xticks(range(len(FEATURE_MODES)))
        ax.set_yticks(range(len(FEATURE_MODES)))
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        for i in range(len(FEATURE_MODES)):
            for j in range(len(FEATURE_MODES)):
                val = mat[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8,
                            color='white' if abs(val) > 0.6 else 'black')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        ax.text(0.5, 0.5, 'no correlation data', ha='center', va='center',
                transform=ax.transAxes)
    ax.set_title('C. Prediction correlation (mean over tracts)')

    # panel D: constraint error
    ax = axes[1, 1]
    for k, (mode, color) in enumerate(zip(FEATURE_MODES, colors)):
        vals = []
        for fips in all_fips:
            row = metrics_df[(metrics_df['fips'] == fips) & (metrics_df['mode'] == mode)]
            vals.append(row['constraint_error_pct'].iloc[0] if len(row) > 0 else float('nan'))
        ax.bar(x + k * width, vals, width, label=labels[k], color=color)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([str(f)[:11] for f in all_fips], rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Constraint error (%)')
    ax.set_title('D. Constraint error by mode and tract')
    ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(SUMMARY_DIR, 'cross_mode_dashboard.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Dashboard saved to {out_path}")


def print_interpretation(metrics_df, corr_df):
    print("\n--- Interpretation hooks ---")
    if len(corr_df) == 0:
        print("No correlation data available.")
        return

    def mean_r(mode_a, mode_b):
        subset = corr_df[
            ((corr_df['mode_a'] == mode_a) & (corr_df['mode_b'] == mode_b)) |
            ((corr_df['mode_a'] == mode_b) & (corr_df['mode_b'] == mode_a))
        ]
        return subset['pearson_r'].mean() if len(subset) > 0 else float('nan')

    r_full_noise = mean_r('full', 'random_noise')
    r_full_coords = mean_r('full', 'coordinates_only')
    r_noise_coords = mean_r('random_noise', 'coordinates_only')

    print(f"r(full, random_noise)        = {r_full_noise:.4f}"
          " -- high => features decorative, structure is graph-induced")
    print(f"r(full, coordinates_only)    = {r_full_coords:.4f}"
          " -- high => coordinate dominance")
    print(f"r(random_noise, coords_only) = {r_noise_coords:.4f}"
          " -- low => coordinates carry signal beyond graph floor")


def main():
    os.makedirs(SUMMARY_DIR, exist_ok=True)

    print("Loading metrics...")
    metrics_df = load_metrics()
    print(f"  {len(metrics_df)} rows across "
          f"{metrics_df['fips'].nunique()} tracts, "
          f"{metrics_df['mode'].nunique()} modes")

    print("Loading correlations...")
    corr_df = load_correlations()
    print(f"  {len(corr_df)} mode-pair rows")

    print("Rendering dashboard...")
    make_dashboard(metrics_df, corr_df)

    print("\nCross-mode metrics summary:")
    pivot = metrics_df.pivot_table(
        index='mode',
        values=['std', 'constraint_error_pct', 'moran_i'],
        aggfunc='mean'
    ).round(4)
    print(pivot.to_string())

    print_interpretation(metrics_df, corr_df)


if __name__ == '__main__':
    main()
