"""
M5 smoke driver -- validates the synthetic target generator over three
representative parameter configurations.

Runs:
    Run 1: latent/linear, autocorr=medium, snr=medium
    Run 2: latent/nonlinear, autocorr=strong, snr=high
    Run 3: features/linear, feature_indices=[0,1,2], autocorr=weak, snr=low

Per run, prints: parameter dict, Moran's I (achieved vs target, in-band),
within-tract variance ratio, variance breakdown, output path.

After all three runs, prints a summary table. Emits a calibration warning
if any run falls outside the target Moran's I band.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))

from granite.synthetic.generator import SyntheticTargetGenerator, _AUTOCORR_TARGETS

_SEED = 42

_CONFIGS = [
    {
        'run_id': 1,
        'params': {
            'signal_source': 'latent',
            'signal_type': 'linear',
            'spatial_autocorrelation': 'medium',
            'snr': 'medium',
            'tract_list_source': 'auto',
        },
    },
    {
        'run_id': 2,
        'params': {
            'signal_source': 'latent',
            'signal_type': 'nonlinear',
            'spatial_autocorrelation': 'strong',
            'snr': 'high',
            'tract_list_source': 'auto',
        },
    },
    {
        'run_id': 3,
        'params': {
            'signal_source': 'features',
            'feature_indices': [0, 1, 2],
            'signal_type': 'linear',
            'spatial_autocorrelation': 'weak',
            'snr': 'low',
            'tract_list_source': 'auto',
        },
    },
]


def _in_band(achieved, target, tol=0.05):
    return abs(achieved - target) <= tol


def run_single(cfg):
    run_id = cfg['run_id']
    params = cfg['params']
    print(f"\n{'='*60}")
    print(f"RUN {run_id}")
    print(f"{'='*60}")
    print(f"params: {params}")

    gen = SyntheticTargetGenerator(seed=_SEED, params=params)
    result = gen.generate()

    diag = result['diagnostics']
    achieved_mi = float(diag['morans_i_achieved'])
    target_mi = float(diag['morans_i_target'])
    tol = 0.05
    ib = _in_band(achieved_mi, target_mi, tol)

    wtvr = float(diag['within_tract_variance_ratio'])
    sig_v = float(diag['signal_variance'])
    res_v = float(diag['residual_variance'])
    noi_v = float(diag['noise_variance'])

    print(f"  achieved Moran's I : {achieved_mi:.4f}")
    print(f"  target  Moran's I  : {target_mi:.4f}  +/- {tol}")
    print(f"  in-band            : {'YES' if ib else 'NO'}")
    print(f"  within-tract var ratio : {wtvr:.4f}")
    print(f"  signal variance    : {sig_v:.4f}")
    print(f"  residual variance  : {res_v:.4f}")
    print(f"  noise variance     : {noi_v:.4f}")
    print(f"  output path        : {result['output_dir']}")

    # verify metadata.json echoes input params
    import json
    meta_path = os.path.join(result['output_dir'], 'metadata.json')
    with open(meta_path) as fh:
        meta = json.load(fh)
    params_match = meta['params'] == params and meta['seed'] == _SEED
    print(f"  metadata.json matches input params: {'YES' if params_match else 'NO'}")

    return {
        'run_id': run_id,
        'achieved_mi': achieved_mi,
        'target_mi': target_mi,
        'in_band': ib,
        'wtvr': wtvr,
        'output_dir': result['output_dir'],
        'params_match': params_match,
    }


def main():
    print("M5 synthetic generator smoke test")
    print(f"seed: {_SEED}")

    summaries = []
    for cfg in _CONFIGS:
        summary = run_single(cfg)
        summaries.append(summary)

    # summary table
    print(f"\n{'='*60}")
    print("SUMMARY TABLE")
    print(f"{'='*60}")
    header = f"{'Run':<5} {'achieved MI':>12} {'target MI':>10} {'tol':>6} {'in-band':>8} {'WTVR':>7}"
    print(header)
    print('-' * len(header))

    any_out_of_band = False
    for s in summaries:
        ib_str = 'YES' if s['in_band'] else 'NO'
        print(
            f"{s['run_id']:<5} "
            f"{s['achieved_mi']:>12.4f} "
            f"{s['target_mi']:>10.4f} "
            f"{'0.05':>6} "
            f"{ib_str:>8} "
            f"{s['wtvr']:>7.4f}"
        )
        if not s['in_band']:
            any_out_of_band = True

    print()
    if any_out_of_band:
        print("CALIBRATION WARNING: length-scale tuning needed before M6 grid run")
    else:
        print("all runs within Moran's I target band.")


if __name__ == '__main__':
    main()
