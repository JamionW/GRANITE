"""
Coordinate-artifact fair-comparison experiment.

Holds graph construction, loss, training schedule, seed, and post-hoc
correction fixed. Varies only the feature matrix across four conditions
on the five Mehdi review tracts.

Conditions
----------
full              - 73-feature accessibility + parcel + NLCD matrix (baseline)
coordinates_only  - z-scored lat/lon in cols 0-1, zeros elsewhere
random_noise      - all d columns drawn i.i.d. N(0, 1) with fixed seed
coords_plus_noise - z-scored lat/lon in cols 0-1, N(0, 1) elsewhere

Usage
-----
    python scripts/coord_artifact_experiment.py
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, '/workspaces/GRANITE')

SEED = 42
BASE_OUTPUT_DIR = './output/coord_artifact_test'
EPOCHS = 200
MAX_FAILURES = 3
CONSTRAINT_TOL = 1e-4

# five Mehdi review tracts in SVI order
TRACTS = [
    ('47065000700', 0.114),
    ('47065000600', 0.224),
    ('47065011326', 0.510),
    ('47065011321', 0.696),
    ('47065002400', 0.891),
]

FEATURE_MODES = ['full', 'coordinates_only', 'random_noise', 'coords_plus_noise']


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


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


def collect_metrics(fips, tract_svi, mode, results):
    if not results.get('success'):
        return None

    predictions_df = results.get('predictions')
    if predictions_df is None:
        return None

    preds = predictions_df['mean'].values
    return {
        'fips': fips,
        'tract_svi': tract_svi,
        'mode': mode,
        'mean': float(np.mean(preds)),
        'std': float(np.std(preds)),
        'range': float(np.max(preds) - np.min(preds)),
        'constraint_error_pct': float(abs(np.mean(preds) - tract_svi) / tract_svi * 100
                                      if tract_svi > 0 else 0.0),
        'moran_i': results.get('moran_i', float('nan')),
    }


def main():
    start = time.time()
    np.random.seed(SEED)

    log("=" * 70)
    log("COORDINATE ARTIFACT EXPERIMENT")
    log(f"Seed: {SEED}  Epochs: {EPOCHS}")
    log(f"Modes: {FEATURE_MODES}")
    log(f"Tracts: {[t[0] for t in TRACTS]}")
    log("=" * 70)

    import yaml
    with open('./config.yaml') as f:
        config = yaml.safe_load(f)

    config['training']['epochs'] = EPOCHS
    config['model']['epochs'] = EPOCHS
    config['processing']['random_seed'] = SEED
    config['processing']['enable_caching'] = True
    config['processing']['verbose'] = False
    config['training']['apply_post_correction'] = True
    config['seed'] = SEED

    all_metrics = []
    # store raw prediction vectors for cross-mode correlation analysis
    # keyed by (fips, mode)
    pred_vectors = {}

    failures = []

    for mode in FEATURE_MODES:
        config['feature_mode'] = mode
        mode_dir = os.path.join(BASE_OUTPUT_DIR, mode)
        os.makedirs(mode_dir, exist_ok=True)

        log(f"\n{'='*70}")
        log(f"FEATURE MODE: {mode}")
        log(f"{'='*70}")

        for fips, tract_svi in TRACTS:
            tract_dir = os.path.join(mode_dir, f'tract_{fips}')
            log(f"\n  Tract {fips} (SVI={tract_svi:.3f})")
            t0 = time.time()

            try:
                results = run_single_tract(fips, config, tract_dir)
            except Exception as e:
                import traceback
                traceback.print_exc()
                log(f"  FAILED: {e}")
                failures.append({'mode': mode, 'fips': fips, 'error': str(e)})
                if len(failures) > MAX_FAILURES:
                    log(f"Stopping: {len(failures)} failures exceeds limit")
                    break
                continue

            if not results.get('success'):
                log(f"  FAILED: {results.get('error', 'unknown')}")
                failures.append({'mode': mode, 'fips': fips, 'error': results.get('error')})
                if len(failures) > MAX_FAILURES:
                    log(f"Stopping: {len(failures)} failures exceeds limit")
                    break
                continue

            row = collect_metrics(fips, tract_svi, mode, results)
            if row:
                all_metrics.append(row)
                # store prediction vector
                preds_df = results.get('predictions')
                if preds_df is not None:
                    pred_vectors[(fips, mode)] = preds_df['mean'].values.copy()

                ce = row['constraint_error_pct']
                log(f"  std={row['std']:.4f}  range={row['range']:.4f}  "
                    f"CE={ce:.3f}%  moran_i={row['moran_i']:.4f}")
            else:
                log(f"  WARNING: no metrics collected")

            log(f"  elapsed: {time.time()-t0:.0f}s")

    # write metrics CSV
    summary_dir = os.path.join(BASE_OUTPUT_DIR, 'summary')
    os.makedirs(summary_dir, exist_ok=True)

    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        metrics_path = os.path.join(summary_dir, 'cross_mode_metrics.csv')
        metrics_df.to_csv(metrics_path, index=False)
        log(f"\nMetrics saved to {metrics_path}")

    # write prediction correlations CSV
    if pred_vectors:
        corr_rows = []
        all_fips = list({fips for fips, _ in pred_vectors})
        for fips in all_fips:
            for i, mode_a in enumerate(FEATURE_MODES):
                for mode_b in FEATURE_MODES[i+1:]:
                    va = pred_vectors.get((fips, mode_a))
                    vb = pred_vectors.get((fips, mode_b))
                    if va is None or vb is None or len(va) != len(vb):
                        continue
                    from scipy.stats import pearsonr
                    r, _ = pearsonr(va, vb)
                    corr_rows.append({
                        'fips': fips,
                        'mode_a': mode_a,
                        'mode_b': mode_b,
                        'pearson_r': float(r),
                    })
        corr_df = pd.DataFrame(corr_rows)
        corr_path = os.path.join(summary_dir, 'prediction_correlations.csv')
        corr_df.to_csv(corr_path, index=False)
        log(f"Correlations saved to {corr_path}")

    elapsed = time.time() - start
    log(f"\nDone in {elapsed/60:.1f} minutes")
    if failures:
        log(f"Failures ({len(failures)}):")
        for f in failures:
            log(f"  {f['mode']} / {f['fips']}: {f['error']}")


if __name__ == '__main__':
    main()
