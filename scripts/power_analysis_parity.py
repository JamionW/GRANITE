#!/usr/bin/env python3
"""
power analysis for the section 4 real-data parity comparison.

reproduces every power/effect-size figure the proposal reports for the
framework-minus-dasymetric per-tract difference, from the committed artifact
data/results/m0_n20_svi_parity/per_tract.csv. no figure in section 4's power
paragraph is hand-entered; this script is the source.

unit of inference is the tract. per-tract bg_r is defined only where a tract
holds at least two block groups, which leaves 19 of the 20 parity tracts.
the effect size is the conventional mean-based paired cohen's d; the headline
per-tract point estimate reported in table 2 is the median, carried here for
the descriptive interval only.

conventions: two-sided alpha 0.05, normal (z) approximation for interval and
sample-size figures to match the proposal text, noncentral-t for power and mde.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ARTIFACT = Path("data/results/m0_n20_svi_parity/per_tract.csv")
OUTPUT_JSON = Path("data/results/m0_n20_svi_parity/power_analysis.json")
# block-group reference version: v1 is the frozen m0 artifact; v2 is the
# REF-FIX rescore from persisted predictions (scripts/rescore_bg_reference.py)
REFERENCE_PATHS = {
    "v1": (ARTIFACT, OUTPUT_JSON),
    "v2": (Path("data/results/bg_reference_v2/m0_per_tract_v2.csv"),
           Path("data/results/bg_reference_v2/power_analysis_v2.json")),
}
ALPHA = 0.05
MIN_BGS = 2  # per-tract correlation is undefined below two block groups
TIE_TOL = 1e-12  # paired differences at or below this magnitude count as exact ties


def load_paired_differences(path: Path) -> pd.Series:
    """framework-minus-dasymetric per-tract bg_r over tracts with >= 2 bgs."""
    df = pd.read_csv(path)
    bg_r = df.pivot_table(index="fips", columns="method", values="bg_r")
    n_bgs = df.pivot_table(index="fips", columns="method", values="n_bgs")
    sufficient = n_bgs["GRANITE"] >= MIN_BGS
    pair = bg_r.loc[sufficient, ["GRANITE", "Dasymetric"]].dropna()
    return pair["GRANITE"] - pair["Dasymetric"]


def power_paired(d: float, n: int, alpha: float = ALPHA) -> float:
    """two-sided power of a one-sample (paired) t-test at effect size d."""
    df = n - 1
    ncp = abs(d) * np.sqrt(n)
    tcrit = stats.t.ppf(1 - alpha / 2, df)
    return float(
        1 - stats.nct.cdf(tcrit, df, ncp) + stats.nct.cdf(-tcrit, df, ncp)
    )


def mde_at_power(n: int, target: float = 0.80, alpha: float = ALPHA) -> float:
    """smallest effect size d detectable at target power for n tracts."""
    lo, hi = 1e-4, 5.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if power_paired(mid, n, alpha) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def n_for_power(d: float, target: float = 0.80, alpha: float = ALPHA) -> int:
    """smallest tract count reaching target power at effect size d."""
    n = 3
    while power_paired(d, n, alpha) < target and n < 100000:
        n += 1
    return n


def z_ci(center: float, sd: float, n: int) -> tuple:
    """normal-approximation 95 percent interval about a location estimate."""
    half = stats.norm.ppf(1 - ALPHA / 2) * sd / np.sqrt(n)
    return center - half, center + half


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference-version", choices=sorted(REFERENCE_PATHS), default="v1",
                    help="block-group reference; sets default --artifact and --output")
    ap.add_argument("--artifact", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    default_artifact, default_output = REFERENCE_PATHS[args.reference_version]
    args.artifact = args.artifact or default_artifact
    args.output = args.output or default_output

    diff = load_paired_differences(args.artifact)
    n = len(diff)
    mean_d = diff.mean()
    median_d = diff.median()
    sd = diff.std(ddof=1)
    d = mean_d / sd  # conventional paired cohen's d, mean based

    print(f"artifact              {args.artifact}")
    print(f"tracts (>= {MIN_BGS} bgs)      {n}")
    print(f"mean paired diff      {mean_d:+.4f}")
    print(f"median paired diff    {median_d:+.4f}")
    print(f"sd of differences     {sd:.4f}")
    print(f"cohen's d (mean)      {d:+.4f}  |d| = {abs(d):.4f}")
    print()

    p19 = power_paired(d, 19)
    p85 = power_paired(d, 85)
    mde19 = mde_at_power(19)
    n80 = n_for_power(d)
    lo_mean, hi_mean = z_ci(mean_d, sd, 85)
    lo_med, hi_med = z_ci(median_d, sd, 85)
    lo19_med, hi19_med = z_ci(median_d, sd, 19)

    print(f"power at n=19         {p19*100:4.1f} %")
    print(f"power at n=85         {p85*100:4.1f} %")
    print(f"mde at n=19, 80% pwr  {mde19*sd:.3f}  (d={mde19:.3f})")
    print(f"  vs median diff      {abs(median_d):.3f}  -> {mde19*sd/abs(median_d):.1f}x")
    print(f"n for 80% power       {n80}")
    print()
    print(f"projected 95% CI at n=85, on the mean effect   "
          f"[{lo_mean:+.2f}, {hi_mean:+.2f}]  width {hi_mean-lo_mean:.2f}  "
          f"({'excludes' if hi_mean < 0 or lo_mean > 0 else 'spans'} zero)")
    print(f"descriptive 95% CI at n=85, on the median      "
          f"[{lo_med:+.2f}, {hi_med:+.2f}]  width {hi_med-lo_med:.2f}")
    print(f"observed 95% CI at n=19, on the median         "
          f"[{lo19_med:+.2f}, {hi19_med:+.2f}]  (table 2 reports a "
          f"tract-clustered bootstrap of the median)")
    print()

    # rank-based sensitivity: distribution-free alternatives to the mean-based
    # cohen's d / t-test power analysis above. wilcoxon drops exact-zero
    # differences (zero_method='wilcox' default); n_nonzero records the count
    # actually used by the test statistic. float ties (two-bg tracts score
    # r = +/-1 for both methods) are snapped to zero first.
    rank_diff = diff.where(diff.abs() > TIE_TOL, 0.0)
    n_zero = int((rank_diff == 0).sum())
    n_nonzero = n - n_zero
    wilcoxon_stat, wilcoxon_p = stats.wilcoxon(rank_diff, alternative="two-sided")
    n_pos = int((rank_diff > 0).sum())
    n_neg = int((rank_diff < 0).sum())
    sign_test = stats.binomtest(n_pos, n_pos + n_neg, p=0.5, alternative="two-sided")

    print(f"wilcoxon signed-rank (two-sided)  statistic={wilcoxon_stat:.4f}  "
          f"p={wilcoxon_p:.4f}  n_nonzero={n_nonzero}  (n_zero_dropped={n_zero})")
    print(f"exact sign test (two-sided)       n_pos={n_pos}  n_neg={n_neg}  "
          f"p={sign_test.pvalue:.4f}")

    results = {
        "artifact": str(args.artifact),
        "n_tracts": n,
        "mean_paired_diff": mean_d,
        "median_paired_diff": median_d,
        "sd_of_differences": sd,
        "cohens_d_mean": d,
        "power_at_n19": p19,
        "power_at_n85": p85,
        "mde_at_n19_80pct_power": {"d": mde19, "diff_units": mde19 * sd},
        "n_for_80pct_power": n80,
        "projected_ci_n85_mean": [lo_mean, hi_mean],
        "descriptive_ci_n85_median": [lo_med, hi_med],
        "observed_ci_n19_median": [lo19_med, hi19_med],
        "rank_based_sensitivity": {
            "wilcoxon_signed_rank": {
                "alternative": "two-sided",
                "zero_method": "wilcox",
                "statistic": float(wilcoxon_stat),
                "p": float(wilcoxon_p),
                "n_total": n,
                "n_zero_dropped": n_zero,
                "n_nonzero": n_nonzero,
            },
            "exact_sign_test": {
                "alternative": "two-sided",
                "n_pos": n_pos,
                "n_neg": n_neg,
                "n_zero": n_zero,
                "p": float(sign_test.pvalue),
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
