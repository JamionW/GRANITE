"""
Stage 4: Multi-tract property-value proxy evaluation.

Runs the property_value proxy across 10 stratified tracts from tract_inventory.csv.
Produces per-tract and aggregate RMSE, MAE, Pearson r against address-level
proxy truth for GRANITE, Dasymetric, and Pycnophylactic.

Proxy definition
----------------
The evaluation target is log-transformed, min-max-normalized Hamilton County
Assessor APPVALUE (assessed property value). No features enter a generative
function and no noise is injected. The transformation is:

    proxy = (log(APPVALUE + 1) - min) / (max - min)

Rationale: using a proxy derived independently of the model's training features
escapes the circularity that would arise from synthesizing a label directly from
the features the model consumes. Eight parcel-derived features (raw APPVALUE,
log_appvalue, appvalue_percentile, appvalue_zscore, appvalue_per_sqft,
is_high_value, is_low_value, value_tier) are excluded from the feature set
when property_value is the target to prevent leakage; leakage audit completed
2026-04, confirmed no residual parcel-value columns enter model input.

Dasymetric framing
------------------
This evaluation follows the dasymetric mapping tradition of disaggregating
areal-unit statistics to sub-unit estimates using ancillary data:

    Mennis, J. (2003). Generating surface models of population using dasymetric
    mapping. The Professional Geographer, 55(1), 31-42.

    Maantay, J. A., & Maroko, A. R. (2009). Mapping urban risk: Flood hazards,
    race, and environmental justice in New York. Applied Geography, 29(1), 111-124.
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime

sys.path.insert(0, '/workspaces/GRANITE')

SEED = 42
OUTPUT_DIR = './output/stage4_property_value_proxy_eval'
TRACT_INVENTORY = './tract_inventory.csv'
EPOCHS = 150
MAX_FAILURES = 3
CONSTRAINT_TOL = 1e-6
MAX_CONSTRAINT_VIOLATIONS = 2


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_tract_inventory():
    df = pd.read_csv(TRACT_INVENTORY, dtype={'fips': str})
    log(f"Loaded {len(df)} tracts from {TRACT_INVENTORY}")
    for _, row in df.iterrows():
        log(f"  {row['fips']}  SVI={row['svi']:.4f}")
    return df


def run_single_tract(fips, config):
    """Run pipeline for one tract with property_value proxy target. Returns results dict."""
    from granite.models.gnn import set_random_seed
    from granite.disaggregation.pipeline import GRANITEPipeline

    set_random_seed(SEED)

    tract_output = os.path.join(OUTPUT_DIR, f'tract_{fips}')
    os.makedirs(tract_output, exist_ok=True)

    tract_config = dict(config)
    tract_config['data'] = dict(config['data'])
    tract_config['data']['target_fips'] = fips
    tract_config['data']['state_fips'] = fips[:2]
    tract_config['data']['county_fips'] = fips[2:5]
    tract_config['data']['target'] = 'property_value'

    pipeline = GRANITEPipeline(tract_config, output_dir=tract_output, verbose=False)

    # suppress count_5 validator bug if it surfaces
    try:
        results = pipeline.run()
    except NameError as e:
        if 'count_5' in str(e):
            log(f"  suppressed count_5 NameError: {e}")
            results = {'success': False, 'error': f'count_5 bug: {e}'}
        else:
            raise

    if results.get('success'):
        pipeline.save_results(results)

    return results


def compute_tract_metrics(fips, results):
    """Compute per-method metrics from pipeline results against proxy truth."""
    if not results.get('success'):
        return None

    address_truth = results.get('address_truth')
    if address_truth is None:
        return None

    predictions_df = results['predictions']
    granite_preds = predictions_df['mean'].values
    truth = address_truth

    # drop addresses with no proxy truth value
    valid = ~np.isnan(truth)
    if valid.sum() < 10:
        return None

    tract_target = results['tract_svi']
    n_addresses = len(granite_preds)

    methods = {}

    # granite
    methods['GRANITE'] = {
        'predictions': granite_preds,
        'constraint_error_abs': abs(np.mean(granite_preds) - tract_target),
        'constraint_error_rel': abs(np.mean(granite_preds) - tract_target) / tract_target * 100
            if tract_target > 0 else 0.0,
    }

    # baselines from saved address_truth.csv or inline results
    baselines = results.get('baseline_comparisons', {}).get('methods', {})
    for method_key, col_name in [('Dasymetric', 'Dasymetric'), ('Pycnophylactic', 'Pycnophylactic')]:
        if method_key in baselines and 'predictions' in baselines[method_key]:
            preds = baselines[method_key]['predictions']
            if len(preds) == n_addresses:
                methods[col_name] = {
                    'predictions': np.array(preds),
                    'constraint_error_abs': abs(np.mean(preds) - tract_target),
                    'constraint_error_rel': abs(np.mean(preds) - tract_target) / tract_target * 100
                        if tract_target > 0 else 0.0,
                }

    rows = []
    for method_name, method_data in methods.items():
        preds = method_data['predictions']
        v = valid & ~np.isnan(preds)
        if v.sum() < 10:
            continue

        p = preds[v]
        t = truth[v]

        rmse = float(np.sqrt(np.mean((p - t) ** 2)))
        mae = float(np.mean(np.abs(p - t)))
        r_val, _ = stats.pearsonr(p, t)
        spatial_std = float(np.std(preds))

        rows.append({
            'fips': fips,
            'tract_svi': tract_target,
            'n_addresses': n_addresses,
            'method': method_name,
            'rmse': rmse,
            'mae': mae,
            'pearson_r': float(r_val),
            'spatial_std': spatial_std,
            'constraint_error_abs': method_data['constraint_error_abs'],
            'constraint_error_rel': method_data['constraint_error_rel'],
        })

    return rows


def compute_tract_metrics_from_csv(fips, tract_svi, csv_path):
    """Fallback: compute metrics from saved address_truth.csv."""
    df = pd.read_csv(csv_path)
    truth = df['truth'].values
    valid_base = ~np.isnan(truth)

    n_addresses = len(df)
    rows = []

    method_cols = {
        'GRANITE': 'granite',
        'Dasymetric': 'dasymetric',
        'Pycnophylactic': 'pycnophylactic',
    }

    for method_name, col in method_cols.items():
        if col not in df.columns:
            continue
        preds = df[col].values
        v = valid_base & ~np.isnan(preds)
        if v.sum() < 10:
            continue

        p = preds[v]
        t = truth[v]

        rmse = float(np.sqrt(np.mean((p - t) ** 2)))
        mae = float(np.mean(np.abs(p - t)))
        r_val, _ = stats.pearsonr(p, t)
        spatial_std = float(np.std(preds))

        constraint_error_abs = abs(np.mean(preds) - tract_svi)
        constraint_error_rel = constraint_error_abs / tract_svi * 100 if tract_svi > 0 else 0.0

        rows.append({
            'fips': fips,
            'tract_svi': tract_svi,
            'n_addresses': n_addresses,
            'method': method_name,
            'rmse': rmse,
            'mae': mae,
            'pearson_r': float(r_val),
            'spatial_std': spatial_std,
            'constraint_error_abs': constraint_error_abs,
            'constraint_error_rel': constraint_error_rel,
        })

    return rows if rows else None


def main():
    start = time.time()
    np.random.seed(SEED)

    log("=" * 70)
    log("STAGE 4: Multi-Tract Property-Value Proxy Evaluation")
    log(f"Seed: {SEED}")
    log(f"Epochs: {EPOCHS}")
    log("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # load config
    import yaml
    with open('./config.yaml') as f:
        config = yaml.safe_load(f)

    config['training']['epochs'] = EPOCHS
    config['model']['epochs'] = EPOCHS
    config['processing']['random_seed'] = SEED
    config['processing']['enable_caching'] = True
    config['processing']['verbose'] = False
    config['training']['apply_post_correction'] = True

    # load tract inventory
    inventory = load_tract_inventory()

    # run per-tract evaluation
    all_rows = []
    failures = []
    constraint_violations = []
    completed_tracts = []

    for idx, inv_row in inventory.iterrows():
        fips = inv_row['fips']
        tract_svi = inv_row['svi']

        log(f"\n--- Tract {idx+1}/{len(inventory)}: {fips} (SVI={tract_svi:.4f}) ---")
        tract_start = time.time()

        try:
            results = run_single_tract(fips, config)
        except Exception as e:
            import traceback
            traceback.print_exc()
            log(f"  FAILED: {e}")
            failures.append({'fips': fips, 'error': str(e)})

            if len(failures) > MAX_FAILURES:
                log(f"\nSTOPPING: {len(failures)} failures exceeds limit of {MAX_FAILURES}")
                break
            continue

        if not results.get('success'):
            error_msg = results.get('error', 'unknown')
            log(f"  FAILED: {error_msg}")
            failures.append({'fips': fips, 'error': error_msg})

            if len(failures) > MAX_FAILURES:
                log(f"\nSTOPPING: {len(failures)} failures exceeds limit of {MAX_FAILURES}")
                break
            continue

        # compute metrics from results
        rows = compute_tract_metrics(fips, results)

        # fallback to CSV if inline extraction failed
        if rows is None:
            csv_path = os.path.join(OUTPUT_DIR, f'tract_{fips}', 'address_truth.csv')
            if os.path.exists(csv_path):
                log(f"  falling back to CSV metrics from {csv_path}")
                # use tract_target from results
                rows = compute_tract_metrics_from_csv(
                    fips, results.get('tract_svi', tract_svi), csv_path
                )

        if rows is None:
            log(f"  FAILED: no metrics could be computed")
            failures.append({'fips': fips, 'error': 'no valid metrics'})
            if len(failures) > MAX_FAILURES:
                log(f"\nSTOPPING: {len(failures)} failures exceeds limit of {MAX_FAILURES}")
                break
            continue

        # check GRANITE constraint error
        granite_rows = [r for r in rows if r['method'] == 'GRANITE']
        if granite_rows:
            ce = granite_rows[0]['constraint_error_abs']
            if ce > CONSTRAINT_TOL:
                log(f"  WARNING: GRANITE constraint error {ce:.2e} exceeds {CONSTRAINT_TOL}")
                constraint_violations.append({'fips': fips, 'constraint_error': ce})

                if len(constraint_violations) > MAX_CONSTRAINT_VIOLATIONS:
                    log(f"\nSTOPPING: {len(constraint_violations)} constraint violations "
                        f"exceeds limit of {MAX_CONSTRAINT_VIOLATIONS}")
                    log("Tracts with violations:")
                    for cv in constraint_violations:
                        log(f"  {cv['fips']}: {cv['constraint_error']:.2e}")
                    break

        all_rows.extend(rows)
        completed_tracts.append(fips)

        elapsed_t = time.time() - tract_start
        for r in rows:
            log(f"  {r['method']}: RMSE={r['rmse']:.4f} MAE={r['mae']:.4f} "
                f"r={r['pearson_r']:.4f} std={r['spatial_std']:.4f} "
                f"CE={r['constraint_error_abs']:.2e}")
        log(f"  completed in {elapsed_t:.0f}s")

    # check if we have enough data to proceed
    if len(completed_tracts) == 0:
        log("\nNo tracts completed successfully. Aborting.")
        return

    # step 2: write per-tract metrics
    per_tract_df = pd.DataFrame(all_rows)
    per_tract_path = os.path.join(OUTPUT_DIR, 'per_tract_metrics.csv')
    per_tract_df.to_csv(per_tract_path, index=False)
    log(f"\nPer-tract metrics saved to {per_tract_path}")

    # step 3: aggregate metrics
    agg = per_tract_df.groupby('method').agg(
        mean_rmse=('rmse', 'mean'),
        median_rmse=('rmse', 'median'),
        mean_mae=('mae', 'mean'),
        median_mae=('mae', 'median'),
        mean_pearson_r=('pearson_r', 'mean'),
        median_pearson_r=('pearson_r', 'median'),
    ).reset_index()

    agg_path = os.path.join(OUTPUT_DIR, 'aggregate_metrics.csv')
    agg.to_csv(agg_path, index=False)
    log(f"Aggregate metrics saved to {agg_path}")

    print()
    print("=" * 70)
    print(f"AGGREGATE RESULTS ({len(completed_tracts)}/{len(inventory)} tracts)")
    print("=" * 70)
    print(agg.to_string(index=False))

    if failures:
        print(f"\nExcluded tracts ({len(failures)}):")
        for f in failures:
            print(f"  {f['fips']}: {f['error']}")

    if constraint_violations:
        print(f"\nConstraint violations ({len(constraint_violations)}):")
        for cv in constraint_violations:
            print(f"  {cv['fips']}: {cv['constraint_error']:.2e}")

    # step 4: stratified metrics
    per_tract_df['svi_bucket'] = pd.cut(
        per_tract_df['tract_svi'],
        bins=[-0.001, 0.3, 0.7, 1.001],
        labels=['low', 'mid', 'high']
    )

    stratified = per_tract_df.groupby(['svi_bucket', 'method']).agg(
        mean_rmse=('rmse', 'mean'),
        median_rmse=('rmse', 'median'),
        mean_mae=('mae', 'mean'),
        median_mae=('mae', 'median'),
        mean_pearson_r=('pearson_r', 'mean'),
        median_pearson_r=('pearson_r', 'median'),
        n_tracts=('fips', 'nunique'),
    ).reset_index()

    strat_path = os.path.join(OUTPUT_DIR, 'stratified_metrics.csv')
    stratified.to_csv(strat_path, index=False)
    log(f"Stratified metrics saved to {strat_path}")

    print()
    print("=" * 70)
    print("STRATIFIED RESULTS (low < 0.3, mid 0.3-0.7, high > 0.7)")
    print("=" * 70)
    print(stratified.to_string(index=False))

    elapsed = time.time() - start
    print()
    print("=" * 70)
    print(f"Stage 4 complete: {elapsed/60:.1f} minutes, seed={SEED}")
    print(f"Completed: {len(completed_tracts)}/{len(inventory)} tracts")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()
