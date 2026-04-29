"""
Multi-architecture experiment for within-tract rank-consistency analysis.

Runs GRANITEPipeline for both GraphSAGE and GCN-GAT on the 8 tracts from
tract_inventory.csv. Outputs per-tract accessibility_features.csv and
granite_predictions.csv in the layout expected by within_tract_rank_consistency.py:

  output/rank_consistency_run/graphsage/tract_{fips}/granite_predictions.csv
  output/rank_consistency_run/graphsage/tract_{fips}/accessibility_features.csv
  output/rank_consistency_run/gcn_gat/tract_{fips}/granite_predictions.csv
  output/rank_consistency_run/gcn_gat/tract_{fips}/accessibility_features.csv

Usage
-----
    python scripts/run_rank_consistency_experiment.py
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
MAX_FAILURES = 4

ARCHITECTURES = [
    ('graphsage', 'sage'),
    ('gcn_gat', 'gcn_gat'),
]

INVENTORY_PATH = './tract_inventory.csv'


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_tracts():
    df = pd.read_csv(INVENTORY_PATH, dtype={'fips': str})
    tracts = list(zip(df['fips'].tolist(), df['svi'].tolist()))
    log(f"loaded {len(tracts)} tracts from {INVENTORY_PATH}")
    for fips, svi in tracts:
        log(f"  {fips}  SVI={svi:.4f}")
    return tracts


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
    log("RANK CONSISTENCY MULTI-ARCHITECTURE EXPERIMENT")
    log(f"Seed: {SEED}  Epochs: {EPOCHS}")
    log(f"Output: {BASE_OUTPUT_DIR}")
    log("=" * 70)

    with open('./config.yaml') as f:
        base_config = yaml.safe_load(f)

    base_config['training']['epochs'] = EPOCHS
    base_config['model']['epochs'] = EPOCHS
    base_config['processing']['random_seed'] = SEED
    base_config['processing']['enable_caching'] = True
    base_config['processing']['verbose'] = False
    base_config['training']['apply_post_correction'] = True
    base_config['seed'] = SEED

    tracts = load_tracts()
    failures = []

    for arch_dir, arch_key in ARCHITECTURES:
        log(f"\n{'='*70}")
        log(f"ARCHITECTURE: {arch_dir}  (config key: {arch_key})")
        log(f"{'='*70}")

        config = dict(base_config)
        config['model'] = dict(base_config['model'])
        config['model']['architecture'] = arch_key

        arch_output = os.path.join(BASE_OUTPUT_DIR, arch_dir)
        os.makedirs(arch_output, exist_ok=True)

        for fips, svi in tracts:
            tract_dir = os.path.join(arch_output, f'tract_{fips}')
            log(f"\n  Tract {fips} (SVI={svi:.4f})")
            t0 = time.time()

            try:
                results = run_single_tract(fips, config, tract_dir)
            except Exception as e:
                import traceback
                traceback.print_exc()
                log(f"  FAILED: {e}")
                failures.append({'arch': arch_dir, 'fips': fips, 'error': str(e)})
                if len(failures) > MAX_FAILURES:
                    log(f"Stopping: {len(failures)} failures exceeds limit")
                    sys.exit(1)
                continue

            if not results.get('success'):
                log(f"  FAILED: {results.get('error', 'unknown')}")
                failures.append({'arch': arch_dir, 'fips': fips, 'error': results.get('error', 'unknown')})
                if len(failures) > MAX_FAILURES:
                    log(f"Stopping: {len(failures)} failures exceeds limit")
                    sys.exit(1)
                continue

            preds_df = results.get('predictions')
            if preds_df is not None:
                n = len(preds_df)
                mean_svi = float(preds_df['mean'].mean())
                constraint_err = abs(mean_svi - svi) / svi * 100 if svi > 0 else 0.0
                log(f"  n={n}  mean={mean_svi:.4f}  CE={constraint_err:.2f}%  "
                    f"elapsed={time.time()-t0:.0f}s")
            else:
                log(f"  no predictions  elapsed={time.time()-t0:.0f}s")

    # verify output
    log(f"\n{'='*70}")
    log("OUTPUT VERIFICATION")
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
            pred_ok = pred_path.exists()
            feat_ok = feat_path.exists()
            if pred_ok and feat_ok:
                try:
                    feat_header = pd.read_csv(feat_path, nrows=0).columns.tolist()
                    # check for numeric-only headers
                    numeric_headers = all(c.isdigit() for c in feat_header[:5])
                    pred_cols = pd.read_csv(pred_path, nrows=0).columns.tolist()
                    has_raw = 'raw_prediction' in pred_cols or 'mean' in pred_cols
                    status = "OK" if (not numeric_headers and has_raw) else "WARN"
                    log(f"  {fips}: {status}  feat_cols={len(feat_header)}  "
                        f"numeric_hdr={numeric_headers}  has_raw={has_raw}")
                    if status == "WARN":
                        all_ok = False
                except Exception as e:
                    log(f"  {fips}: ERROR reading CSVs: {e}")
                    all_ok = False
            else:
                log(f"  {fips}: MISSING  preds={pred_ok}  feat={feat_ok}")
                all_ok = False

    if failures:
        log(f"\nFAILURES ({len(failures)}):")
        for f in failures:
            log(f"  {f['arch']} {f['fips']}: {f['error']}")
    else:
        log("\nno failures")

    elapsed = time.time() - start
    log(f"\nexperiment completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    log(f"output: {BASE_OUTPUT_DIR}")
    if all_ok:
        log("verification: all tracts OK")
    else:
        log("verification: WARNINGS -- check output above before running rank_consistency")


if __name__ == '__main__':
    main()
