#!/usr/bin/env python3
"""
variance components for the study 1 sizing criterion (milestone A).

decomposes the spread of the granite-minus-dasymetric per-tract paired
recovery_r difference, on the m6 synthetic recovery grid, into a tract
component (heterogeneity of the true per-tract effect across the 20 n20
tracts) and a seed component (draw-to-draw noise for a fixed tract, pooled
across the 81 draws: autocorr x snr x between_tract x seed).

input: data/results/m6_recovery_grid/recovery_grid.csv
  - granite rows restricted to feature_mode=coordinates_only, arch=sage
    (the primary 81-draw x 20-tract grid; dasymetric has no feature_mode/arch
    split, so this is the comparable subset -- 1620 rows each side)
  - paired on (autocorr, snr, between_tract, seed, tract_fips)

estimator: one-way random-effects ANOVA (balanced, n=81 draws per tract,
k=20 tracts), classic moment estimator:
  sigma_seed^2  = MS_within                     (residual variance, "seed")
  sigma_tract^2 = max((MS_between - MS_within) / n, 0)

"seed" here pools all non-tract draw-level variation (true seed plus the
autocorr/snr/between_tract cell factors), not literal reseeding of a fixed
cell -- the grid does not hold cell fixed while only varying seed across all
81 draws. this is a simplification requested by the two-component framing;
see README.md alongside the output JSON.

MDE convention: two-sided alpha 0.05, 80% power target, z-approximation
(z_0.975 + z_0.80 = 2.8016), matching scripts/power_analysis_parity.py's
docstring convention for interval/sample-size figures. MDE(s) at n tracts,
each tract's diff averaged over s seeds:
  MDE(s) = (z_alpha + z_beta) * sqrt(sigma_tract^2 + sigma_seed^2/s) / sqrt(n)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ARTIFACT = Path("data/results/m6_recovery_grid/recovery_grid.csv")
OUT_JSON = Path("data/results/variance_components/study1_variance_components.json")

ALPHA = 0.05
POWER_TARGET = 0.80
N_TRACTS_TARGET = 85
SEED_COUNTS = [3, 5, 9, 15]
SEED_SHARE_THRESHOLD = 0.25
MDE_TARGET = 0.15
SIGMA_TRACT_THRESHOLD_REFERENCE = 0.494  # stated in the milestone spec; verified below


def load_paired_diff(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    g = df[(df["method"] == "granite") & (df["feature_mode"] == "coordinates_only") & (df["arch"] == "sage")]
    d = df[df["method"] == "dasymetric"]
    merged = g.merge(
        d, on=["autocorr", "snr", "between_tract", "seed", "tract_fips"], suffixes=("_g", "_d")
    )
    merged["diff"] = merged["recovery_r_g"] - merged["recovery_r_d"]
    return merged


def one_way_variance_components(merged: pd.DataFrame):
    counts = merged.groupby("tract_fips").size()
    if counts.nunique() != 1:
        raise RuntimeError(f"unbalanced design across tracts: {counts.unique()}")
    n = int(counts.iloc[0])
    k = int(counts.shape[0])
    N = int(len(merged))

    grand_mean = merged["diff"].mean()
    tract_means = merged.groupby("tract_fips")["diff"].mean()
    ss_between = float((counts * (tract_means - grand_mean) ** 2).sum())
    ss_within = float(
        sum(((sub["diff"] - sub["diff"].mean()) ** 2).sum() for _, sub in merged.groupby("tract_fips"))
    )

    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (N - k)

    sigma_seed2 = ms_within
    sigma_tract2 = max((ms_between - ms_within) / n, 0.0)

    return {
        "n_draws_per_tract": n,
        "n_tracts": k,
        "n_rows": N,
        "grand_mean_diff": float(grand_mean),
        "ms_between": ms_between,
        "ms_within": ms_within,
        "sigma_tract2": sigma_tract2,
        "sigma_seed2": sigma_seed2,
        "sigma_tract": float(np.sqrt(sigma_tract2)),
        "sigma_seed": float(np.sqrt(sigma_seed2)),
        "combined_sd": float(np.sqrt(sigma_tract2 + sigma_seed2)),
    }


def mde(sigma_tract2: float, sigma_seed2: float, s: int, n_tracts: int, mult: float) -> float:
    se = np.sqrt(sigma_tract2 + sigma_seed2 / s) / np.sqrt(n_tracts)
    return float(mult * se)


def seed_share(sigma_tract2: float, sigma_seed2: float, s: int) -> float:
    per_tract_seed_var = sigma_seed2 / s
    return float(per_tract_seed_var / (sigma_tract2 + per_tract_seed_var))


def crossover_seed_count(sigma_tract2: float, sigma_seed2: float, threshold: float) -> float:
    """smallest continuous s where seed_share(s) < threshold."""
    if sigma_tract2 <= 0:
        return float("inf")
    # seed_share(s) < t  <=>  sigma_seed2/s < t/(1-t) * sigma_tract2  <=>  s > sigma_seed2*(1-t)/(t*sigma_tract2)
    return float(sigma_seed2 * (1 - threshold) / (threshold * sigma_tract2))


def main() -> None:
    merged = load_paired_diff(ARTIFACT)
    vc = one_way_variance_components(merged)

    z_alpha = stats.norm.ppf(1 - ALPHA / 2)
    z_beta = stats.norm.ppf(POWER_TARGET)
    mult = z_alpha + z_beta

    mde_by_seeds = {
        s: {
            "mde": mde(vc["sigma_tract2"], vc["sigma_seed2"], s, N_TRACTS_TARGET, mult),
            "seed_share": seed_share(vc["sigma_tract2"], vc["sigma_seed2"], s),
        }
        for s in SEED_COUNTS
    }

    s_cross = crossover_seed_count(vc["sigma_tract2"], vc["sigma_seed2"], SEED_SHARE_THRESHOLD)

    sigma_tract_threshold = MDE_TARGET * np.sqrt(N_TRACTS_TARGET) / mult

    result = {
        "artifact": str(ARTIFACT),
        "n_pairs": int(len(merged)),
        "n_tracts": vc["n_tracts"],
        "n_draws_per_tract": vc["n_draws_per_tract"],
        "estimator": "one-way random-effects ANOVA, balanced, moment estimator",
        "mde_convention": {
            "alpha": ALPHA,
            "power_target": POWER_TARGET,
            "z_alpha": float(z_alpha),
            "z_beta": float(z_beta),
            "multiplier": float(mult),
            "approximation": "normal (z), matching power_analysis_parity.py convention",
        },
        "grand_mean_diff": vc["grand_mean_diff"],
        "ms_between": vc["ms_between"],
        "ms_within": vc["ms_within"],
        "sigma_tract": vc["sigma_tract"],
        "sigma_seed": vc["sigma_seed"],
        "combined_sd": vc["combined_sd"],
        "n_tracts_target": N_TRACTS_TARGET,
        "mde_by_seed_count": {
            str(s): {"mde": v["mde"], "seed_variance_share": v["seed_share"]}
            for s, v in mde_by_seeds.items()
        },
        "seed_count_seed_share_below_0.25": {
            "continuous": s_cross,
            "smallest_integer": int(np.ceil(s_cross)) if np.isfinite(s_cross) else None,
        },
        "sigma_tract_unreachability_check": {
            "mde_target": MDE_TARGET,
            "n_tracts": N_TRACTS_TARGET,
            "threshold_sigma_tract": float(sigma_tract_threshold),
            "reference_threshold_in_spec": SIGMA_TRACT_THRESHOLD_REFERENCE,
            "observed_sigma_tract": vc["sigma_tract"],
            "sigma_tract_exceeds_threshold": bool(vc["sigma_tract"] > sigma_tract_threshold),
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)

    print(f"wrote {OUT_JSON}")
    print(f"sigma_tract={vc['sigma_tract']:.6f}  sigma_seed={vc['sigma_seed']:.6f}  combined_sd={vc['combined_sd']:.6f}")
    print(f"threshold sigma_tract for 0.15 MDE @ n=85: {sigma_tract_threshold:.6f} (spec reference: {SIGMA_TRACT_THRESHOLD_REFERENCE})")
    print(f"sigma_tract exceeds threshold: {vc['sigma_tract'] > sigma_tract_threshold}")
    for s, v in mde_by_seeds.items():
        print(f"  seeds={s:2d}  MDE={v['mde']:.5f}  seed_share={v['seed_share']:.4f}")
    print(f"seed count where seed share < 0.25: continuous={s_cross:.3f}, smallest int={int(np.ceil(s_cross))}")


if __name__ == "__main__":
    main()
