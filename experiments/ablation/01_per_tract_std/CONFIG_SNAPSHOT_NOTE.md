# CONFIG_SNAPSHOT_NOTE.md

The `config_snapshot.yaml` in this directory shows `feature_standardization: "global"`, but the
actual run used `per_tract` z-score standardization. This is a preflight artifact capture bug.

## What happened

At the time of the 2a run (2026-05-21, SHA `486279248c872616d5574d1917a4027b3c3a4575`), the
preflight artifacts (git_state.txt, environment.txt, config_snapshot.yaml) were created from
the on-disk `config.yaml` file rather than from the runtime cfg dict. The runtime override
`cfg['features']['feature_standardization'] = 'per_tract'` was applied to the in-memory dict
after the snapshot was written, so the on-disk snapshot never reflected it.

## Evidence that the run used per_tract scaling

- `experiments/ablation/01_per_tract_std/README.md` run metadata table:
  `feature_standardization | per_tract (z-score)`
- `results/per_tract_scalers.npz` (68 KB): per-tract mu/sigma saved only under per-tract scaling
- `results/zero_var_columns.csv` (18 KB): (tract, feature_idx) pairs clamped to std=1.0,
  produced only by the per-tract z-score path
- Numerical results are identical to the step 4 `00_baseline_for_step4` run, which explicitly
  sets `feature_standardization: per_tract` and produces a correct config snapshot

## Fix applied

The snapshot-capture bug was fixed in `run_ablation_03.py` and `run_ablation_04.py` at SHA
`e988de442bbe8cf7bb587263be4817fb6a5b464b` (step 4 script) and in the fixup commit that lands
this note. In both fixed scripts, all runtime config mutations are applied before `_write_preflight`
is called, so the snapshot on disk matches what actually ran.

This note is the authoritative record of the discrepancy. The `config_snapshot.yaml` file in this
directory has not been regenerated to avoid altering committed history.
