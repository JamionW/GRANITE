"""
Block-group validation for the four coord-artifact feature modes.

Reads per-tract prediction CSVs from coord_artifact_experiment.py output,
aggregates to block-group scale, and compares GNN-full / GNN-coords_only /
GNN-noise / GNN-cpn against Dasymetric and Pycnophylactic baselines.

Output:
    output/coord_artifact/bg_validation_summary.csv
    output/coord_artifact/bg_validation_report.txt
    output/coord_artifact/bg_validation_bootstrap.csv
"""
import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import stats

sys.path.insert(0, '/workspaces/GRANITE')

SEED = 42
N_BOOTSTRAP = 1000

# five Mehdi review tracts in SVI order
TRACTS = [
    ('47065000700', 0.114),
    ('47065000600', 0.224),
    ('47065011326', 0.510),
    ('47065011321', 0.696),
    ('47065002400', 0.891),
]
TRACT_FIPS = [t[0] for t in TRACTS]
TRACT_SVI_MAP = {fips: svi for fips, svi in TRACTS}

FEATURE_MODES = ['full', 'coordinates_only', 'random_noise', 'coords_plus_noise']
METHOD_LABELS = {
    'full':              'GNN-full',
    'coordinates_only':  'GNN-coords',
    'random_noise':      'GNN-noise',
    'coords_plus_noise': 'GNN-cpn',
}

INPUT_DIR = './output/coord_artifact_test'
OUTPUT_DIR = './output/coord_artifact'


# =============================================================================
# step 1: confirm prediction CSV path pattern
# =============================================================================

def prediction_csv_path(mode, fips):
    return os.path.join(INPUT_DIR, mode, f'tract_{fips}',
                        f'granite_predictions_tract_{fips}_{mode}.csv')


# =============================================================================
# step 2: load addresses for all 5 tracts in fixed order
# =============================================================================

def load_addresses():
    from granite.data.loaders import DataLoader
    loader = DataLoader()
    parts = []
    for fips in TRACT_FIPS:
        addr = loader.get_addresses_for_tract(fips)
        if len(addr) == 0:
            print(f"WARNING: no addresses for tract {fips}")
            continue
        addr = addr.sort_values('address_id').reset_index(drop=True)
        addr['tract_fips'] = fips
        parts.append(addr)
    combined = gpd.GeoDataFrame(
        pd.concat(parts, ignore_index=True),
        geometry='geometry', crs='EPSG:4326'
    )
    print(f"loaded {len(combined)} addresses across {len(parts)} tracts")
    return combined


# =============================================================================
# step 3: load block groups with national SVI
# =============================================================================

def load_block_groups():
    from granite.data.block_group_loader import BlockGroupLoader
    loader = BlockGroupLoader(data_dir='./data', verbose=False)
    bg = loader.get_block_groups_with_demographics(svi_ranking_scope='national')
    print(f"loaded {len(bg)} block groups (national scope)")
    return bg


# =============================================================================
# step 4: load per-mode GNN predictions aligned to addresses
# =============================================================================

def load_mode_predictions(mode, addresses):
    parts = []
    for fips in TRACT_FIPS:
        path = prediction_csv_path(mode, fips)
        if not os.path.exists(path):
            raise FileNotFoundError(f"missing prediction CSV: {path}")
        df = pd.read_csv(path)
        df = df.sort_values('address_id').reset_index(drop=True)
        df['tract_fips'] = fips
        parts.append(df)
    preds_df = pd.concat(parts, ignore_index=True)

    # align with addresses by (tract_fips, address_id) order
    addr_key = addresses[['tract_fips', 'address_id']].copy()
    merged = addr_key.merge(
        preds_df[['tract_fips', 'address_id', 'mean']],
        on=['tract_fips', 'address_id'],
        how='left'
    )
    preds = merged['mean'].values

    n_missing = np.isnan(preds).sum()
    if n_missing > 0:
        print(f"WARNING: {n_missing} unmatched addresses in mode {mode}")

    assert len(preds) == len(addresses), (
        f"mode {mode}: prediction length {len(preds)} != address count {len(addresses)}"
    )
    return preds


# =============================================================================
# step 5: compute baseline predictions (dasymetric, pycnophylactic)
# =============================================================================

def load_tract_gdf():
    from granite.data.loaders import DataLoader
    loader = DataLoader()
    tracts = loader.load_census_tracts('47', '065')
    svi = loader.load_svi_data('47', 'Hamilton')
    merged = tracts.merge(svi, on='FIPS', how='inner')
    return merged


def compute_baselines(addresses, tract_gdf):
    from granite.evaluation.baselines import (
        DasymetricDisaggregation, PycnophylacticDisaggregation
    )
    n = len(addresses)
    results = {}

    # dasymetric
    print("computing dasymetric predictions...")
    try:
        dasy = DasymetricDisaggregation(ancillary_column='nlcd_impervious_pct')
        dasy.fit(tract_gdf, svi_column='RPL_THEMES')
        dasy_preds = np.zeros(n)
        for fips, tract_svi in TRACTS:
            mask = addresses['tract_fips'] == fips
            if mask.sum() == 0:
                continue
            addr_coords = np.column_stack([
                addresses.loc[mask, 'geometry'].apply(lambda g: g.x).values,
                addresses.loc[mask, 'geometry'].apply(lambda g: g.y).values,
            ])
            tract_preds = dasy.disaggregate(
                addr_coords, fips, tract_svi,
                address_gdf=addresses.loc[mask]
            )
            dasy_preds[mask] = tract_preds
        results['Dasymetric'] = dasy_preds
        print(f"  dasymetric: mean={np.mean(dasy_preds):.3f} std={np.std(dasy_preds):.4f}")
    except Exception as e:
        print(f"  dasymetric failed: {e}")

    # pycnophylactic
    print("computing pycnophylactic predictions...")
    try:
        pycno = PycnophylacticDisaggregation(n_iterations=50, k_neighbors=8)
        pycno.fit(tract_gdf, svi_column='RPL_THEMES')
        pycno_preds = np.zeros(n)
        for fips, tract_svi in TRACTS:
            mask = addresses['tract_fips'] == fips
            if mask.sum() == 0:
                continue
            addr_coords = np.column_stack([
                addresses.loc[mask, 'geometry'].apply(lambda g: g.x).values,
                addresses.loc[mask, 'geometry'].apply(lambda g: g.y).values,
            ])
            tract_preds = pycno.disaggregate(addr_coords, fips, tract_svi)
            pycno_preds[mask] = tract_preds
        results['Pycnophylactic'] = pycno_preds
        print(f"  pycnophylactic: mean={np.mean(pycno_preds):.3f} std={np.std(pycno_preds):.4f}")
    except Exception as e:
        print(f"  pycnophylactic failed: {e}")

    return results


# =============================================================================
# step 6: block-group validation via BlockGroupValidator
# =============================================================================

def run_validation(addresses, all_predictions, block_groups):
    from granite.validation.block_group_validation import BlockGroupValidator
    validator = BlockGroupValidator(block_groups, verbose=False)
    bg_results = {}
    for method_name, preds in all_predictions.items():
        print(f"  validating {method_name}...")
        result = validator.validate(addresses, preds, method_name=method_name)
        bg_results[method_name] = result
        corrs = result['correlations'].get('svi_correlation', {})
        r = corrs.get('pearson_r', float('nan'))
        n_bg = result['n_block_groups']
        print(f"    r={r:.3f}  n_bg={n_bg}")
    return bg_results


# =============================================================================
# step 7: bootstrap Pearson r
# =============================================================================

def bootstrap_all(bg_results, n_bootstrap=N_BOOTSTRAP, seed=SEED):
    np.random.seed(seed)

    # build a common GEOID index from the first method's validation_data
    first_method = list(bg_results.keys())[0]
    base_df = bg_results[first_method]['validation_data'].copy()
    if 'SVI' not in base_df.columns:
        print("WARNING: no SVI column in validation_data; bootstrap skipped")
        return {}, pd.DataFrame()

    base_df = base_df.dropna(subset=['SVI']).set_index('GEOID')

    boot_results = {}
    all_boot_rows = []

    for method_name, result in bg_results.items():
        df = result['validation_data'].copy()
        if 'SVI' not in df.columns or 'predicted_svi' not in df.columns:
            print(f"WARNING: missing columns for {method_name}, skipping bootstrap")
            continue

        df = df.set_index('GEOID')
        common_geoids = base_df.index.intersection(df.index)
        gt = base_df.loc[common_geoids, 'SVI'].values
        pred = df.loc[common_geoids, 'predicted_svi'].values

        valid = ~np.isnan(gt) & ~np.isnan(pred)
        gt_v = gt[valid]
        pred_v = pred[valid]
        n = len(gt_v)

        if n < 5:
            print(f"WARNING: only {n} valid BGs for {method_name}; bootstrap unreliable")
            continue

        r_obs = np.corrcoef(gt_v, pred_v)[0, 1]
        boot_rs = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(n, size=n, replace=True)
            if np.std(gt_v[idx]) < 1e-10 or np.std(pred_v[idx]) < 1e-10:
                boot_rs.append(float('nan'))
                continue
            r_b = np.corrcoef(gt_v[idx], pred_v[idx])[0, 1]
            boot_rs.append(r_b)

        boot_arr = np.array(boot_rs)
        ci_lo = np.nanpercentile(boot_arr, 2.5)
        ci_hi = np.nanpercentile(boot_arr, 97.5)

        boot_results[method_name] = {
            'r': r_obs,
            'ci_lower': ci_lo,
            'ci_upper': ci_hi,
            'n_bg': n,
            'bootstrap_dist': boot_arr,
        }

        for val in boot_arr:
            all_boot_rows.append({'method': method_name, 'r_boot': val})

    boot_df = pd.DataFrame(all_boot_rows)
    return boot_results, boot_df


# =============================================================================
# step 8: build summary, report, and print table
# =============================================================================

def build_summary(bg_results, boot_results, all_predictions, addresses):
    rows = []
    for method_name, result in bg_results.items():
        corrs = result['correlations'].get('svi_correlation', {})
        diag = result['diagnostics']
        r_bg = corrs.get('pearson_r', float('nan'))
        spearman = corrs.get('spearman_rho', float('nan'))
        p_val = corrs.get('p_value', float('nan'))
        n_bg = result['n_block_groups']
        n_addr = len(addresses)
        mean_std = diag.get('mean_within_bg_std', float('nan'))

        boot = boot_results.get(method_name, {})
        ci_lo = boot.get('ci_lower', float('nan'))
        ci_hi = boot.get('ci_upper', float('nan'))

        rows.append({
            'method':             method_name,
            'r_bg':               r_bg,
            'spearman_rho':       spearman,
            'p_value':            p_val,
            'ci_lower':           ci_lo,
            'ci_upper':           ci_hi,
            'n_bg':               n_bg,
            'n_addresses':        n_addr,
            'mean_within_bg_std': mean_std,
        })
    return pd.DataFrame(rows)


def ci_overlap(boot_a, boot_b):
    """return True if 95% CIs overlap between two bootstrap dists."""
    if boot_a is None or boot_b is None:
        return True
    lo_a, hi_a = np.nanpercentile(boot_a, 2.5), np.nanpercentile(boot_a, 97.5)
    lo_b, hi_b = np.nanpercentile(boot_b, 2.5), np.nanpercentile(boot_b, 97.5)
    return not (hi_a < lo_b or hi_b < lo_a)


def build_report(summary_df, boot_results, bg_results):
    lines = []
    lines.append("=" * 75)
    lines.append("COORD ARTIFACT EXPERIMENT: BLOCK-GROUP VALIDATION")
    lines.append("=" * 75)
    lines.append("")
    lines.append("5-tract pilot: 47065000700, 47065000600, 47065011326,")
    lines.append("               47065011321, 47065002400")
    lines.append("")

    # comparison table
    hdr = f"{'method':<18} {'r_bg':>7}  {'ci_95':>20}  {'n_bg':>5}  {'n_addr':>7}"
    lines.append(hdr)
    lines.append("-" * 62)
    for _, row in summary_df.iterrows():
        ci_str = f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]"
        lines.append(
            f"{row['method']:<18} {row['r_bg']:>7.3f}  {ci_str:>20}  "
            f"{int(row['n_bg']):>5}  {int(row['n_addresses']):>7}"
        )
    lines.append("")

    # interpretation
    lines.append("-" * 75)
    lines.append("INTERPRETATION")
    lines.append("-" * 75)
    lines.append("")

    full_r   = summary_df.loc[summary_df['method'] == METHOD_LABELS['full'], 'r_bg']
    coords_r = summary_df.loc[summary_df['method'] == METHOD_LABELS['coordinates_only'], 'r_bg']
    dasy_r   = summary_df.loc[summary_df['method'] == 'Dasymetric', 'r_bg']

    if len(full_r) and len(coords_r):
        r_full   = float(full_r.iloc[0])
        r_coords = float(coords_r.iloc[0])
        delta_fc = r_full - r_coords

        boot_full   = boot_results.get(METHOD_LABELS['full'],   {}).get('bootstrap_dist')
        boot_coords = boot_results.get(METHOD_LABELS['coordinates_only'], {}).get('bootstrap_dist')
        overlap = ci_overlap(boot_full, boot_coords)

        if delta_fc >= 0.05 and not overlap:
            lines.append(
                "Features contribute block-group-detectable vulnerability signal "
                "beyond coordinate structure."
            )
        else:
            lines.append(
                "Features produce spatially structured within-tract variation not "
                "aligned with block-group boundaries. The orthogonal feature signal "
                "operates below block-group scale; block-group validation cannot "
                "resolve it."
            )
        lines.append(f"  r(GNN-full)={r_full:.3f}  r(GNN-coords)={r_coords:.3f}  "
                     f"delta={delta_fc:+.3f}  CIs overlap={overlap}")
        lines.append("")

    if len(full_r) and len(dasy_r):
        r_full = float(full_r.iloc[0])
        r_dasy = float(dasy_r.iloc[0])
        delta_fd = r_full - r_dasy
        lines.append(f"GNN-full vs Dasymetric: delta r = {delta_fd:+.3f} "
                     f"(r_full={r_full:.3f}, r_dasy={r_dasy:.3f})")
        lines.append("")

    # small-n warning
    min_bg = int(summary_df['n_bg'].min()) if len(summary_df) else 0
    if min_bg < 5:
        lines.append(
            f"WARNING: minimum n_bg across methods is {min_bg}. the 5-tract pilot "
            "has limited block-group power. a full 85-tract rerun is the next step."
        )
        lines.append("")

    lines.append("=" * 75)
    return "\n".join(lines)


def print_table(summary_df):
    print("")
    print(f"{'method':<18} {'r_bg':>7}  {'ci_95':>20}  {'n_bg':>5}  {'n_addr':>7}")
    print("-" * 62)
    for _, row in summary_df.iterrows():
        ci_str = f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]"
        print(f"{row['method']:<18} {row['r_bg']:>7.3f}  {ci_str:>20}  "
              f"{int(row['n_bg']):>5}  {int(row['n_addresses']):>7}")
    print("")


# =============================================================================
# main
# =============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # step 2: load addresses in canonical order
    print("\nloading addresses...")
    addresses = load_addresses()

    # step 3: load block groups
    print("\nloading block groups...")
    block_groups = load_block_groups()

    # step 4: load GNN mode predictions
    print("\nloading GNN predictions...")
    all_predictions = {}
    for mode in FEATURE_MODES:
        label = METHOD_LABELS[mode]
        try:
            preds = load_mode_predictions(mode, addresses)
            all_predictions[label] = preds
            print(f"  {label}: loaded {len(preds)} predictions")
        except Exception as e:
            print(f"  {label}: FAILED - {e}")

    # check for identical prediction vectors between modes
    mode_keys = list(all_predictions.keys())
    for i, ma in enumerate(mode_keys):
        for mb in mode_keys[i+1:]:
            if np.allclose(all_predictions[ma], all_predictions[mb], atol=1e-6):
                print(f"WARNING: predictions identical between {ma} and {mb} - upstream bug suspected")

    # step 5: baselines
    print("\nloading tract geometries for baselines...")
    tract_gdf = load_tract_gdf()
    print("computing baselines...")
    baseline_preds = compute_baselines(addresses, tract_gdf)
    all_predictions.update(baseline_preds)

    # step 4d / 4e: block-group validation
    print("\nrunning block-group validation...")
    bg_results = run_validation(addresses, all_predictions, block_groups)

    # step 6: bootstrap
    print("\nbootstrapping Pearson r (1000 iterations)...")
    boot_results, boot_df = bootstrap_all(bg_results)

    # step 7: outputs
    summary_df = build_summary(bg_results, boot_results, all_predictions, addresses)

    summary_path = os.path.join(OUTPUT_DIR, 'bg_validation_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"saved: {summary_path}")

    report_text = build_report(summary_df, boot_results, bg_results)
    report_path = os.path.join(OUTPUT_DIR, 'bg_validation_report.txt')
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(f"saved: {report_path}")

    if len(boot_df):
        boot_path = os.path.join(OUTPUT_DIR, 'bg_validation_bootstrap.csv')
        boot_df.to_csv(boot_path, index=False)
        print(f"saved: {boot_path}")

    # step 8: print table
    print_table(summary_df)

    # verification checks
    print("verification:")
    if summary_df['r_bg'].isna().any():
        print("  FAIL: some methods have NaN r_bg")
    else:
        print("  PASS: all methods have non-NaN r_bg")

    coords_row = summary_df[summary_df['method'] == METHOD_LABELS['coordinates_only']]
    dasy_row   = summary_df[summary_df['method'] == 'Dasymetric']
    if len(coords_row) and len(dasy_row):
        r_c = float(coords_row['r_bg'].iloc[0])
        r_d = float(dasy_row['r_bg'].iloc[0])
        if abs(r_c - r_d) < 1e-6:
            print("  WARN: GNN-coords r_bg == Dasymetric r_bg - check data pipeline")
        else:
            print(f"  PASS: GNN-coords r_bg ({r_c:.3f}) != Dasymetric r_bg ({r_d:.3f})")

    ci_ok = all(
        row['ci_lower'] < row['r_bg'] < row['ci_upper']
        for _, row in summary_df.iterrows()
        if not (np.isnan(row['ci_lower']) or np.isnan(row['r_bg']) or np.isnan(row['ci_upper']))
    )
    print(f"  {'PASS' if ci_ok else 'FAIL'}: ci_lower < r_bg < ci_upper for all rows")

    min_bg = int(summary_df['n_bg'].min()) if len(summary_df) else 0
    if min_bg < 5:
        print(f"\nWARNING: n_bg={min_bg} for at least one method. "
              "5-tract pilot has limited BG power. a full 85-tract rerun is the next step.")


if __name__ == '__main__':
    main()
