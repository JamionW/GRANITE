"""
variance_decomposition.py

One-way ANOVA variance partition (between-tract / within-tract) for all 75
analyzed columns in the n20 feature matrix (lat, lon, + 73 named features).

No normalization. Operates on natural-unit values as stored in the CSV.

Outputs (committable):
  experiments/ecological_fallacy/variance_decomposition.csv
  experiments/ecological_fallacy/variance_decomposition_summary.md
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(REPO_ROOT, 'experiments', 'ecological_fallacy', 'n20_feature_matrix.csv')
OUT_CSV    = os.path.join(REPO_ROOT, 'experiments', 'ecological_fallacy', 'variance_decomposition.csv')
OUT_MD     = os.path.join(REPO_ROOT, 'experiments', 'ecological_fallacy', 'variance_decomposition_summary.md')

# canonical class assignment by feature name prefix / exact match
ACCESSIBILITY_FEATURES = [
    f'{d}_{s}'
    for d in ['employment', 'healthcare', 'grocery']
    for s in ['min_time', 'mean_time', 'median_time', 'count_5min', 'count_10min',
              'count_15min', 'drive_advantage', 'dispersion', 'time_range', 'percentile']
]  # 30

MODAL_FEATURES = [
    f'{d}_{s}'
    for d in ['employment', 'healthcare', 'grocery']
    for s in ['transit_dependence', 'car_effective_access', 'walk_effective_access',
              'modal_access_gap', 'forced_walk_burden']
]  # 15

SOCIOECONOMIC_FEATURES = [
    'pct_no_vehicle', 'pct_poverty', 'pct_unemployed', 'pct_no_hs_diploma',
    'pct_uninsured', 'pct_mobile_homes', 'pct_crowded', 'population', 'housing_units',
]  # 9

BUILDING_FEATURES = [
    'log_bldg_footprint_m2', 'bldg_vertex_count', 'in_sfha', 'is_residential',
    'log_appvalue', 'build_to_land_ratio', 'log_acres',
    'lucode_residential', 'lucode_commercial', 'lucode_industrial', 'lucode_other',
    'proptype_residential', 'proptype_apartment_10plus', 'proptype_commercial',
    'proptype_rental_40pct', 'proptype_cha_housing',
    'nlcd_land_cover', 'nlcd_impervious_pct', 'nlcd_tree_canopy_pct',
]  # 19

COORDINATE_FEATURES = ['lat', 'lon']

CLASS_MAP = (
    {f: 'accessibility'    for f in ACCESSIBILITY_FEATURES}
    | {f: 'modal'          for f in MODAL_FEATURES}
    | {f: 'socioeconomic'  for f in SOCIOECONOMIC_FEATURES}
    | {f: 'building'       for f in BUILDING_FEATURES}
    | {f: 'coordinate'     for f in COORDINATE_FEATURES}
)

assert len(ACCESSIBILITY_FEATURES) == 30
assert len(MODAL_FEATURES) == 15
assert len(SOCIOECONOMIC_FEATURES) == 9
assert len(BUILDING_FEATURES) == 19
assert len(COORDINATE_FEATURES) == 2


def partition_variance(series, groups):
    """One-way variance partition.  Returns (eta_sq, within_share, zero_variance)."""
    vals = series.values.astype(float)
    grand_mean = vals.mean()
    ss_total = np.sum((vals - grand_mean) ** 2)
    if ss_total == 0:
        return np.nan, np.nan, True

    ss_between = 0.0
    for g, idx in groups:
        sub = vals[idx]
        ss_between += len(sub) * (sub.mean() - grand_mean) ** 2

    ss_within = ss_total - ss_between
    eta_sq = ss_between / ss_total
    within_share = ss_within / ss_total
    return eta_sq, within_share, False


def within_std_ratio(series, groups, global_std):
    """Median over tracts of (within-tract std) / (global std)."""
    if global_std == 0 or np.isnan(global_std):
        return np.nan
    tract_stds = []
    vals = series.values.astype(float)
    for g, idx in groups:
        tract_stds.append(vals[idx].std())
    return float(np.median(tract_stds)) / global_std


def main():
    print(f"reading {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH, dtype={'fips': str})
    print(f"  shape: {df.shape}")

    analyzed_cols = COORDINATE_FEATURES + list(
        ACCESSIBILITY_FEATURES + MODAL_FEATURES + SOCIOECONOMIC_FEATURES + BUILDING_FEATURES
    )
    # verify all analyzed columns are in the dataframe
    missing_in_df = [c for c in analyzed_cols if c not in df.columns]
    if missing_in_df:
        raise RuntimeError(f"columns missing from matrix: {missing_in_df}")

    # verify all analyzed columns have a class assignment
    unclassified = [c for c in analyzed_cols if c not in CLASS_MAP]
    if unclassified:
        raise RuntimeError(f"unclassified columns: {unclassified}")

    # build group index lists once (used by all columns)
    fips_list = df['fips'].unique()
    groups = [(fips, np.where(df['fips'].values == fips)[0]) for fips in fips_list]

    # tract-level means of tract_svi (constant per tract -- used as coupling target)
    tract_svi_series = df.groupby('fips')['tract_svi'].first()

    # tracts with n >= 100 for sensitivity
    tract_n = df.groupby('fips').size()
    fips_n100 = tract_n[tract_n >= 100].index.tolist()
    groups_n100 = [(fips, np.where(df['fips'].values == fips)[0])
                   for fips in fips_n100]

    rows = []
    for col in analyzed_cols:
        feature_class = CLASS_MAP[col]
        series = df[col]

        # guard: no normalization; operate on stored values
        global_std = float(series.std())

        eta, within, zero_var = partition_variance(series, groups)
        wsr = within_std_ratio(series, groups, global_std)

        # tract-level mean for coupling
        tract_means = df.groupby('fips')[col].mean()
        tract_means_aligned = tract_means.loc[tract_svi_series.index]
        if zero_var:
            r = np.nan
        else:
            r, _ = stats.pearsonr(tract_means_aligned.values, tract_svi_series.values)

        # sensitivity: n >= 100 tracts only
        tract_means_n100 = tract_means.loc[fips_n100]
        tract_svi_n100   = tract_svi_series.loc[fips_n100]
        if zero_var or len(fips_n100) < 3:
            r_n100 = np.nan
        else:
            r_n100, _ = stats.pearsonr(tract_means_n100.values, tract_svi_n100.values)

        rows.append({
            'feature':          col,
            'feature_class':    feature_class,
            'eta_sq':           eta,
            'within_share':     within,
            'within_std_ratio': wsr,
            'zero_variance':    zero_var,
            'tract_svi_r':      r,
            'tract_svi_r_n100': r_n100,
            'tract_svi_r2':     r**2 if (r is not None and not np.isnan(r)) else np.nan,
        })

    result = pd.DataFrame(rows)

    # ---- validation ----
    # (1) socioeconomic must be tract-level constants: median eta_sq >= 0.99
    soc_subset = result[result['feature_class'] == 'socioeconomic']
    soc_med = soc_subset['eta_sq'].median()
    if np.isnan(soc_med) or soc_med < 0.99:
        failing = soc_subset[soc_subset['eta_sq'] < 0.99][['feature', 'eta_sq']].to_string(index=False)
        raise RuntimeError(
            f"VALIDATION FAILED: socioeconomic median eta_sq = {soc_med:.6f} < 0.99\n"
            f"failing columns:\n{failing}"
        )
    print(f"  PASS: socioeconomic median eta_sq = {soc_med:.6f} >= 0.99")

    # (2) report coordinate vs accessibility -- no directional gate
    coord_med  = result[result['feature_class'] == 'coordinate']['eta_sq'].median()
    access_med = result[result['feature_class'] == 'accessibility']['eta_sq'].median()
    print(f"  INFO: coordinate median eta_sq = {coord_med:.4f}, accessibility median eta_sq = {access_med:.4f} (reported, not gated)")

    # (3) modal: no threshold -- report only
    modal_subset = result[result['feature_class'] == 'modal']
    modal_med  = modal_subset['eta_sq'].median()
    modal_min  = modal_subset['eta_sq'].min()
    modal_max  = modal_subset['eta_sq'].max()
    print(f"  INFO: modal eta_sq median={modal_med:.4f} range=[{modal_min:.4f}, {modal_max:.4f}] (per-address OSRM, not gated)")

    # check no unexpected NaN eta_sq
    nan_eta = result[result['eta_sq'].isna() & ~result['zero_variance']]
    if len(nan_eta) > 0:
        raise RuntimeError(f"unexpected NaN eta_sq (non-zero-variance): {nan_eta['feature'].tolist()}")

    zero_var_features = result[result['zero_variance']]['feature'].tolist()

    # ---- write CSV ----
    result.to_csv(OUT_CSV, index=False, float_format='%.8f')
    print(f"  wrote {OUT_CSV} ({len(result)} rows)")

    # ---- coupling correlations ----
    valid = result[~result['eta_sq'].isna() & ~result['tract_svi_r'].isna()]
    eta_sq_vals = valid['eta_sq'].values
    abs_r_vals  = valid['tract_svi_r'].abs().values
    coupling_r, _ = stats.pearsonr(eta_sq_vals, abs_r_vals)
    print(f"  eta_sq vs |tract_svi_r| coupling r = {coupling_r:.4f} (n={len(valid)})")

    # subset: exclude coordinate (2) and socioeconomic (9) -- no-corners-no-constants
    subset_classes = {'accessibility', 'modal', 'building'}
    valid_sub = valid[valid['feature_class'].isin(subset_classes)]
    coupling_r_sub, _ = stats.pearsonr(valid_sub['eta_sq'].values, valid_sub['tract_svi_r'].abs().values)
    print(f"  eta_sq vs |tract_svi_r| coupling r (accessibility+modal+building, n={len(valid_sub)}) = {coupling_r_sub:.4f}")

    # ---- class-level summary ----
    def class_stats(cls):
        sub = result[result['feature_class'] == cls]
        eta = sub['eta_sq'].dropna()
        ar  = sub['tract_svi_r'].abs().dropna()
        q25, q75 = eta.quantile(0.25), eta.quantile(0.75)
        return {
            'n':              len(sub),
            'eta_sq_median':  eta.median(),
            'eta_sq_q25':     q25,
            'eta_sq_q75':     q75,
            'eta_sq_iqr':     q75 - q25,
            'abs_r_median':   ar.median(),
        }

    classes = ['coordinate', 'accessibility', 'modal', 'socioeconomic', 'building']
    summary = {cls: class_stats(cls) for cls in classes}

    # ---- write markdown summary ----
    lines = []
    lines.append("# Variance Decomposition: n20 Feature Matrix")
    lines.append("")
    lines.append("Input: `experiments/ecological_fallacy/n20_feature_matrix.csv` (39535 rows, 20 tracts)")
    lines.append("Method: one-way ANOVA partition by tract FIPS. No normalization applied.")
    lines.append("")
    lines.append("## Class-level summary")
    lines.append("")
    lines.append("| feature_class | n | eta_sq median | eta_sq IQR [Q1, Q3] | median |tract_svi_r| |")
    lines.append("|---|---|---|---|---|")
    for cls in classes:
        s = summary[cls]
        lines.append(
            f"| {cls} | {s['n']} "
            f"| {s['eta_sq_median']:.4f} "
            f"| [{s['eta_sq_q25']:.4f}, {s['eta_sq_q75']:.4f}] "
            f"| {s['abs_r_median']:.4f} |"
        )
    lines.append("")
    lines.append("## Headline numbers")
    lines.append("")
    acc_med  = summary['accessibility']['eta_sq_median']
    coord_med = summary['coordinate']['eta_sq_median']
    build_med = summary['building']['eta_sq_median']
    soc_med  = summary['socioeconomic']['eta_sq_median']
    modal_med = summary['modal']['eta_sq_median']
    lines.append(f"- Accessibility median eta_sq: **{acc_med:.4f}**")
    lines.append(f"- Coordinate median eta_sq: **{coord_med:.4f}**")
    lines.append(f"- Building class median eta_sq: **{build_med:.4f}**")
    lines.append(f"- Socioeconomic median eta_sq: **{soc_med:.6f}** (>= 0.99: {'YES' if soc_med >= 0.99 else 'NO'})")
    lines.append(f"- Modal median eta_sq: **{modal_med:.6f}** (>= 0.99: {'YES' if modal_med >= 0.99 else 'NO'})")
    lines.append("")
    lines.append(f"- eta_sq vs |tract_svi_r| coupling Pearson r (full n=75): **{coupling_r:.4f}**")
    lines.append(f"- eta_sq vs |tract_svi_r| coupling Pearson r (accessibility+modal+building only, n={len(valid_sub)}): **{coupling_r_sub:.4f}**")
    lines.append("  (positive = between-tract features are the SVI-predictive ones;")
    lines.append("  subset excludes coordinate and socioeconomic to remove the two extreme corners)")
    lines.append("")
    lines.append("## Modal features note")
    lines.append("")
    lines.append(
        f"Modal features carry substantial within-tract variance: "
        f"median eta_sq = {modal_med:.4f}, range [{modal_min:.4f}, {modal_max:.4f}]. "
        f"This is consistent with per-address OSRM computation (see `docs/FEATURES.md:52-64`); "
        f"modal features were upgraded from tract-level constants to per-address drive/walk "
        f"time comparisons. Feature class does not partition the effect: modal eta_sq "
        f"overlaps both accessibility and building ranges."
    )
    lines.append("")

    if zero_var_features:
        lines.append("## Zero-variance features")
        lines.append("")
        for f in zero_var_features:
            lines.append(f"- {f}")
        lines.append("")
    else:
        lines.append("## Zero-variance features")
        lines.append("")
        lines.append("None.")
        lines.append("")

    lines.append("## Per-feature detail (top 10 by eta_sq)")
    lines.append("")
    top10 = result.nlargest(10, 'eta_sq')[['feature', 'feature_class', 'eta_sq', 'within_share', 'tract_svi_r']]
    lines.append("| feature | class | eta_sq | within_share | tract_svi_r |")
    lines.append("|---|---|---|---|---|")
    for _, row in top10.iterrows():
        lines.append(
            f"| {row['feature']} | {row['feature_class']} "
            f"| {row['eta_sq']:.4f} | {row['within_share']:.4f} "
            f"| {row['tract_svi_r']:.4f} |"
        )
    lines.append("")
    lines.append("## Per-feature detail (bottom 10 by eta_sq, excluding zero-variance)")
    lines.append("")
    bot10 = result[~result['zero_variance']].nsmallest(10, 'eta_sq')[
        ['feature', 'feature_class', 'eta_sq', 'within_share', 'tract_svi_r']
    ]
    lines.append("| feature | class | eta_sq | within_share | tract_svi_r |")
    lines.append("|---|---|---|---|---|")
    for _, row in bot10.iterrows():
        lines.append(
            f"| {row['feature']} | {row['feature_class']} "
            f"| {row['eta_sq']:.4f} | {row['within_share']:.4f} "
            f"| {row['tract_svi_r']:.4f} |"
        )

    with open(OUT_MD, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  wrote {OUT_MD}")

    # ---- console summary ----
    print("\n=== class-level summary ===")
    print(f"{'class':<15} {'n':>3} {'eta_sq_med':>10} {'eta_sq_iqr':>10} {'|r|_med':>8}")
    for cls in classes:
        s = summary[cls]
        print(f"{cls:<15} {s['n']:>3} {s['eta_sq_median']:>10.4f} {s['eta_sq_iqr']:>10.4f} {s['abs_r_median']:>8.4f}")
    print(f"\nzero-variance features: {zero_var_features if zero_var_features else 'none'}")
    print(f"coupling r (eta_sq vs |tract_svi_r|): {coupling_r:.4f}")


if __name__ == '__main__':
    main()
