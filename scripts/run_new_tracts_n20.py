"""
Train GraphSAGE and GCN-GAT on the 12 new tracts added to bring the
rank-consistency experiment to 20 tracts total. Skips any tract whose
output directory already contains both required files.

Prints chosen tracts (FIPS, SVI, address count) before training begins.

Usage
-----
    python scripts/run_new_tracts_n20.py
"""
import os
import sys
import time
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

sys.path.insert(0, '/workspaces/GRANITE')

SEED = 42
EPOCHS = 200
BASE_OUTPUT_DIR = './output/rank_consistency_run'
MAX_FAILURES = 6

ARCHITECTURES = [
    ('graphsage', 'sage'),
    ('gcn_gat', 'gcn_gat'),
]

# 12 new tracts -- pre-computed address counts from spatial join
NEW_TRACTS = [
    # fips,          svi,    n_addresses, bin
    ('47065011447', 0.165,   853),    # [0.0, 0.2)
    ('47065011413', 0.2735, 1442),    # [0.2, 0.4)
    ('47065010431', 0.3848, 3394),    # [0.2, 0.4)
    ('47065010433', 0.454,  2307),    # [0.4, 0.6)
    ('47065001800', 0.5471, 1411),    # [0.4, 0.6)
    ('47065011402', 0.5759, 2858),    # [0.4, 0.6)
    ('47065011444', 0.7088, 1934),    # [0.6, 0.8)
    ('47065010435', 0.7153, 2780),    # [0.6, 0.8)
    ('47065011311', 0.7464, 2358),    # [0.6, 0.8)
    ('47065003400', 0.8797, 2559),    # [0.8, 1.0]
    ('47065001200', 0.9066, 1420),    # [0.8, 1.0]
    ('47065001900', 0.9804, 2469),    # [0.8, 1.0]
]

BINS = {
    '[0.0, 0.2)': 4, '[0.2, 0.4)': 4, '[0.4, 0.6)': 4,
    '[0.6, 0.8)': 4, '[0.8, 1.0]': 4,
}


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _needs_run(tract_dir):
    """True if the tract directory is missing predictions or features."""
    td = Path(tract_dir)
    return not (td / 'granite_predictions.csv').exists() or \
           not (td / 'accessibility_features.csv').exists()


def run_single_tract(fips, config, output_dir):
    from granite.models.gnn import set_random_seed
    from granite.disaggregation.pipeline import GRANITEPipeline

    set_random_seed(SEED)
    os.makedirs(output_dir, exist_ok=True)

    tract_config = dict(config)
    tract_config['data'] = dict(config['data'])
    tract_config['data']['target_fips'] = fips
    tract_config['data']['state_fips'] = fips[:2]
    tract_config['data']['county_fips'] = fips[2:5]

    pipeline = GRANITEPipeline(tract_config, output_dir=output_dir, verbose=False)
    results = pipeline.run()

    if results.get('success'):
        pipeline.save_results(results)

    return results


def main():
    start = time.time()
    np.random.seed(SEED)

    log("=" * 70)
    log("RANK CONSISTENCY EXPANSION: 12 NEW TRACTS (n=20 target)")
    log(f"seed: {SEED}  epochs: {EPOCHS}")
    log(f"output: {BASE_OUTPUT_DIR}")
    log("=" * 70)

    # print chosen tracts before training
    log("\nchosen 12 new tracts (svi bin stratification):")
    log(f"  {'FIPS':<14s}  {'SVI':>7s}  {'addresses':>10s}  bin")
    log(f"  {'-'*14}  {'-'*7}  {'-'*10}  ---")
    for fips, svi, n_addr in NEW_TRACTS:
        svi_bin = (
            '[0.0, 0.2)' if svi < 0.2 else
            '[0.2, 0.4)' if svi < 0.4 else
            '[0.4, 0.6)' if svi < 0.6 else
            '[0.6, 0.8)' if svi < 0.8 else
            '[0.8, 1.0]'
        )
        log(f"  {fips:<14s}  {svi:>7.4f}  {n_addr:>10d}  {svi_bin}")
    log("")

    with open('./config.yaml') as f:
        base_config = yaml.safe_load(f)

    base_config['training']['epochs'] = EPOCHS
    base_config['model']['epochs'] = EPOCHS
    base_config['processing']['random_seed'] = SEED
    base_config['processing']['enable_caching'] = True
    base_config['processing']['verbose'] = False
    base_config['training']['apply_post_correction'] = True
    base_config['seed'] = SEED

    failures = []

    for arch_dir, arch_key in ARCHITECTURES:
        log(f"\n{'='*70}")
        log(f"architecture: {arch_dir}  (config key: {arch_key})")
        log(f"{'='*70}")

        config = dict(base_config)
        config['model'] = dict(base_config['model'])
        config['model']['architecture'] = arch_key

        arch_output = os.path.join(BASE_OUTPUT_DIR, arch_dir)
        os.makedirs(arch_output, exist_ok=True)

        for fips, svi, n_addr in NEW_TRACTS:
            tract_dir = os.path.join(arch_output, f'tract_{fips}')

            if not _needs_run(tract_dir):
                log(f"  skipping {fips} -- outputs already exist")
                continue

            log(f"\n  tract {fips} (SVI={svi:.4f}, n_addr={n_addr})")
            t0 = time.time()

            try:
                results = run_single_tract(fips, config, tract_dir)
            except Exception as e:
                import traceback
                traceback.print_exc()
                log(f"  FAILED: {e}")
                failures.append({'arch': arch_dir, 'fips': fips, 'error': str(e)})
                if len(failures) > MAX_FAILURES:
                    log(f"stopping: {len(failures)} failures exceeds limit")
                    sys.exit(1)
                continue

            if not results.get('success'):
                msg = results.get('error', 'unknown')
                log(f"  FAILED: {msg}")
                failures.append({'arch': arch_dir, 'fips': fips, 'error': msg})
                if len(failures) > MAX_FAILURES:
                    log(f"stopping: {len(failures)} failures exceeds limit")
                    sys.exit(1)
                continue

            preds_df = results.get('predictions')
            if preds_df is not None:
                n = len(preds_df)
                mean_svi = float(preds_df['mean'].mean())
                ce = abs(mean_svi - svi) / svi * 100 if svi > 0 else 0.0
                log(f"  n={n}  mean={mean_svi:.4f}  CE={ce:.2f}%  "
                    f"elapsed={time.time()-t0:.0f}s")
            else:
                log(f"  no predictions  elapsed={time.time()-t0:.0f}s")

    # verification
    log(f"\n{'='*70}")
    log("output verification -- all 20 tracts")
    log(f"{'='*70}")

    all_ok = True
    for arch_dir, _ in ARCHITECTURES:
        arch_path = Path(BASE_OUTPUT_DIR) / arch_dir
        tract_dirs = sorted(arch_path.glob('tract_*'))
        log(f"\n{arch_dir}: {len(tract_dirs)} tract dir(s)")
        for td in tract_dirs:
            fips = td.name.replace('tract_', '')
            pred_path = td / 'granite_predictions.csv'
            feat_path = td / 'accessibility_features.csv'
            if pred_path.exists() and feat_path.exists():
                try:
                    feat_header = pd.read_csv(feat_path, nrows=0).columns.tolist()
                    n_cols = len(feat_header)
                    numeric_hdr = all(c.isdigit() for c in feat_header[:5])
                    pred_cols = pd.read_csv(pred_path, nrows=0).columns.tolist()
                    has_raw = 'raw_prediction' in pred_cols or 'mean' in pred_cols
                    status = "OK" if (not numeric_hdr and has_raw and n_cols == 73) else "WARN"
                    log(f"  {fips}: {status}  feat_cols={n_cols}  has_raw={has_raw}")
                    if status != "OK":
                        all_ok = False
                except Exception as e:
                    log(f"  {fips}: ERROR: {e}")
                    all_ok = False
            else:
                log(f"  {fips}: MISSING  preds={pred_path.exists()}  feat={feat_path.exists()}")
                all_ok = False

    if failures:
        log(f"\nfailures ({len(failures)}):")
        for f in failures:
            log(f"  {f['arch']} {f['fips']}: {f['error']}")
    else:
        log("\nno failures")

    elapsed = time.time() - start
    log(f"\nexperiment completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    if all_ok:
        log("verification: all 20 tracts OK")
    else:
        log("verification: WARNINGS -- check output above")


if __name__ == '__main__':
    main()
