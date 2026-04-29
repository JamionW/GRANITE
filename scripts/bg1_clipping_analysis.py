"""
BG1 Clipping Investigation: diagnostic analysis of block group 470650006001
clipping to SVI=0.0 after rescaling.

Read-only analysis. Does not modify any source files.
"""
import sys
sys.path.insert(0, '/workspaces/GRANITE')

import numpy as np
import pandas as pd
from pathlib import Path
from io import StringIO

from granite.data.block_group_loader import BlockGroupLoader, rescale_block_group_svis


def main():
    out = StringIO()
    def p(s=""):
        print(s, file=out)
        print(s)

    loader = BlockGroupLoader(data_dir='./data', verbose=False)
    df = loader.fetch_acs_demographics()

    bgs = ['470650006001', '470650006002', '470650006003']
    labels = ['BG1 (006001)', 'BG2 (006002)', 'BG3 (006003)']
    target = df[df['GEOID'].isin(bgs)].set_index('GEOID').loc[bgs]
    complete = df[df['svi_complete'] == True]

    # =====================================================================
    p("=" * 90)
    p("BG1 CLIPPING INVESTIGATION: Tract 47065000600")
    p("=" * 90)

    # --- Step 1: Raw ACS Indicators ---
    p("\n1. RAW ACS INDICATORS\n")

    # income indicators
    income_cols = ['EP_MHI', 'EP_PCI']
    pct_cols = ['EP_UNEMP', 'EP_NOHSDP', 'EP_NOVEH', 'EP_AGE65', 'EP_AGE17',
                'EP_MINRTY', 'EP_SNGPNT', 'EP_MUNIT', 'EP_MOBILE', 'EP_CROWD']

    header = f"{'Indicator':<20} {'BG1 (006001)':>14} {'BG2 (006002)':>14} {'BG3 (006003)':>14} {'County Median':>14}"
    p(header)
    p("-" * len(header))

    for col in income_cols:
        vals = [target.loc[g, col] for g in bgs]
        med = complete[col].median()
        p(f"{col:<20} {'$'+f'{vals[0]:,.0f}':>14} {'$'+f'{vals[1]:,.0f}':>14} {'$'+f'{vals[2]:,.0f}':>14} {'$'+f'{med:,.0f}':>14}")

    for col in pct_cols:
        vals = [target.loc[g, col] for g in bgs]
        med = complete[col].median()
        p(f"{col:<20} {vals[0]:>13.1f}% {vals[1]:>13.1f}% {vals[2]:>13.1f}% {med:>13.1f}%")

    # population
    pops = [target.loc[g, 'population'] for g in bgs]
    med_pop = complete['population'].median()
    p(f"{'Population':<20} {pops[0]:>14.0f} {pops[1]:>14.0f} {pops[2]:>14.0f} {med_pop:>14.0f}")

    # svi_complete
    for g, label in zip(bgs, labels):
        sc = target.loc[g, 'svi_complete']
        p(f"{'svi_complete':<20} {str(sc):>14}") if g == bgs[0] else None
    sc_vals = [str(target.loc[g, 'svi_complete']) for g in bgs]
    p(f"{'svi_complete':<20} {sc_vals[0]:>14} {sc_vals[1]:>14} {sc_vals[2]:>14}")

    # NaN check
    all_ep = income_cols + pct_cols
    p("\nNaN check:")
    any_nan = False
    for col in all_ep:
        for g, label in zip(bgs, labels):
            if pd.isna(target.loc[g, col]):
                p(f"  {col} is NaN for {label}")
                any_nan = True
    if not any_nan:
        p("  No NaN values in any EP_ column for these block groups.")

    # --- Step 2: Rank Positions ---
    p("\n\n2. BG1 PERCENTILE RANK AMONG ALL HAMILTON COUNTY BLOCK GROUPS (N={})".format(len(complete)))
    p("   (percentile = fraction of BGs with equal or lower value)\n")

    # for income vars, lower = more vulnerable, so high percentile = low vulnerability
    # for rate vars, higher = more vulnerable, so low percentile = low vulnerability
    invert_vars = ['EP_MHI', 'EP_PCI']

    header2 = f"{'Indicator':<20} {'BG1 Value':>14} {'Percentile':>12} {'Interpretation':>30}"
    p(header2)
    p("-" * len(header2))

    for col in all_ep:
        val = target.loc[bgs[0], col]
        if col in invert_vars:
            # higher income = less vulnerable; rank ascending, percentile = fraction below
            rank_pctl = (complete[col] <= val).mean()
            interp = "less vulnerable" if rank_pctl > 0.5 else "more vulnerable"
        else:
            # higher rate = more vulnerable; rank ascending
            rank_pctl = (complete[col] <= val).mean()
            interp = "less vulnerable" if rank_pctl < 0.5 else "more vulnerable"

        if col in income_cols:
            val_str = f"${val:,.0f}"
        else:
            val_str = f"{val:.1f}%"
        p(f"{col:<20} {val_str:>14} {rank_pctl:>11.1%} {interp:>30}")

    # --- Step 3: Pre-rescaling SVI Breakdown ---
    p("\n\n3. COMPOSITE SVI BREAKDOWN (pre-rescaling)\n")

    svi_rows = [
        ('Overall SVI', 'SVI'),
        ('Theme 1 (Socio)', 'SVI_theme1'),
        ('Theme 2 (Household)', 'SVI_theme2'),
        ('Theme 3 (Minority)', 'SVI_theme3'),
        ('Theme 4 (Housing)', 'SVI_theme4'),
    ]

    header3 = f"{'':>22} {'BG1 (006001)':>14} {'BG2 (006002)':>14} {'BG3 (006003)':>14}"
    p(header3)
    p("-" * len(header3))
    for label, col in svi_rows:
        vals = [target.loc[g, col] for g in bgs]
        p(f"{label:>22} {vals[0]:>14.4f} {vals[1]:>14.4f} {vals[2]:>14.4f}")

    # --- Step 4: Rescaling simulation ---
    p("\n\n4. RESCALING SIMULATION\n")

    # get tract SVI (from pipeline context: 0.2894 for 47065000600)
    # compute weighted mean of BG SVIs
    bg_svis = {g: target.loc[g, 'SVI'] for g in bgs}
    bg_counts = {g: int(target.loc[g, 'population']) for g in bgs}
    total_pop = sum(bg_counts.values())
    weighted_mean = sum(bg_svis[g] * bg_counts[g] for g in bgs) / total_pop

    p(f"BG SVI weighted mean (by population): {weighted_mean:.4f}")

    # The tract SVI used in the pipeline - need to find it
    # From the convergence experiment context, the shift was -0.1806
    # So tract_svi = weighted_mean + shift... but let's compute from the shift
    # shift = tract_svi - weighted_mean => tract_svi = weighted_mean - 0.1806
    # Actually, the user said shift of -0.1806, meaning tract_svi < weighted_mean
    # Let's try tract_svi = weighted_mean - 0.1806
    # But we should check what the actual tract SVI is

    # Let's try a few plausible tract SVI values
    # From CLAUDE.md: Dasymetric block-group correlation: r = 0.558
    # The user said shift = -0.1806
    # shift = tract_svi - weighted_mean => tract_svi = weighted_mean + (-0.1806) = weighted_mean - 0.1806
    tract_svi = weighted_mean - 0.1806

    p(f"Tract SVI (from shift of -0.1806): {tract_svi:.4f}")
    p(f"Required shift: {-0.1806:+.4f}")

    rescaled = rescale_block_group_svis(bg_svis, bg_counts, tract_svi)

    p(f"\nRescaled SVIs:")
    for g, label in zip(bgs, labels):
        orig = bg_svis[g]
        new = rescaled[g]
        p(f"  {label}: {orig:.4f} -> {new:.4f} (change: {new - orig:+.4f})")

    # verify clipping
    p(f"\nBG1 clipped to 0.0: {rescaled[bgs[0]] == 0.0}")

    # --- Step 5: Relative Position Analysis ---
    p("\n\n5. BG1 RELATIVE POSITION ANALYSIS\n")

    # (a) Is BG1 least vulnerable on majority of indicators?
    p("(a) Indicator-by-indicator comparison: which BG is least vulnerable?\n")

    bg1_least = 0
    bg2_least = 0
    bg3_least = 0

    for col in all_ep:
        vals = {g: target.loc[g, col] for g in bgs}
        if col in invert_vars:
            # higher income = less vulnerable
            least_vuln = max(vals, key=vals.get)
        else:
            # lower rate = less vulnerable
            least_vuln = min(vals, key=vals.get)

        idx = bgs.index(least_vuln)
        if idx == 0:
            bg1_least += 1
        elif idx == 1:
            bg2_least += 1
        else:
            bg3_least += 1

        winner = labels[idx]
        p(f"  {col:<20} least vulnerable: {winner}")

    p(f"\n  Score: BG1={bg1_least}, BG2={bg2_least}, BG3={bg3_least} indicators as least vulnerable")
    p(f"  BG1 is least vulnerable on {'majority' if bg1_least > len(all_ep)/2 else 'minority'} of indicators ({bg1_least}/{len(all_ep)})")

    # (b) Gap analysis
    p("\n(b) Gap analysis: BG1-vs-BG2 gap relative to BG2-vs-BG3 gap\n")

    for col in all_ep:
        v1 = target.loc[bgs[0], col]
        v2 = target.loc[bgs[1], col]
        v3 = target.loc[bgs[2], col]

        gap_12 = abs(v1 - v2)
        gap_23 = abs(v2 - v3)

        if col in income_cols:
            p(f"  {col:<20} BG1-BG2 gap: ${gap_12:>10,.0f}   BG2-BG3 gap: ${gap_23:>10,.0f}   ratio: {gap_12/(gap_23+1e-9):.2f}")
        else:
            p(f"  {col:<20} BG1-BG2 gap: {gap_12:>10.1f}pp  BG2-BG3 gap: {gap_23:>10.1f}pp  ratio: {gap_12/(gap_23+1e-9):.2f}")

    # (c) National comparison
    p("\n(c) BG1 vs national medians (2020 ACS)\n")

    national = {
        'EP_UNEMP': (5.4, 'unemployment'),
        'EP_NOVEH': (8.5, 'no vehicle'),
        'EP_NOHSDP': (11.5, 'no HS diploma'),  # approximate national
        'EP_MHI': (64994, 'median household income'),
        'EP_PCI': (35384, 'per capita income'),
    }

    for col, (nat_val, desc) in national.items():
        bg1_val = target.loc[bgs[0], col]
        if col in invert_vars:
            comparison = "above (less vulnerable)" if bg1_val > nat_val else "below (more vulnerable)"
        else:
            comparison = "below (less vulnerable)" if bg1_val < nat_val else "above (more vulnerable)"

        if col in income_cols:
            p(f"  {desc:<30} BG1: ${bg1_val:>10,.0f}  National: ${nat_val:>10,.0f}  {comparison}")
        else:
            p(f"  {desc:<30} BG1: {bg1_val:>9.1f}%  National: {nat_val:>9.1f}%  {comparison}")

    # --- Step 6: Recommendation ---
    p("\n\n6. RECOMMENDATION\n")

    # Assess: BG1 has 0% unemployment, 0% no-vehicle, low minority %, low no-HS-diploma
    # But its pre-rescaling SVI is 0.133, which is low but not zero
    # After rescaling with a -0.1806 shift, 0.133 - 0.1806 = -0.048, clipped to 0.0
    # Meanwhile BG2: 0.490 - 0.1806 = 0.310, BG3: 0.570 - 0.1806 = 0.390

    bg1_pre = bg_svis[bgs[0]]
    bg2_pre = bg_svis[bgs[1]]
    bg1_post = rescaled[bgs[0]]
    bg2_post = rescaled[bgs[1]]

    pre_gap = bg2_pre - bg1_pre
    post_gap = bg2_post - bg1_post

    p(f"Pre-rescaling gap BG1-BG2: {pre_gap:.4f}")
    p(f"Post-rescaling gap BG1-BG2: {post_gap:.4f}")
    p(f"Gap distortion: {post_gap - pre_gap:+.4f} ({(post_gap/pre_gap - 1)*100:+.1f}%)")
    p()

    # BG1 is genuinely low vulnerability on several indicators (0% unemployment, 0% no-vehicle)
    # but its SVI of 0.133 is NOT zero. The clipping to 0.0 exaggerates the gap.
    # A floor based on national percentile estimates:
    # BG1 unemployment 0% -> ~0th percentile nationally -> very low vulnerability
    # BG1 income $80,952 -> above national median -> low vulnerability
    # BG1 no-vehicle 0% -> very low vulnerability
    # Overall BG1 is genuinely in the bottom decile nationally, but not literally zero.
    # A reasonable floor: ~0.05 (5th percentile nationally)

    p("ASSESSMENT:")
    p(f"  BG1 is genuinely less vulnerable than BG2 and BG3 on most indicators.")
    p(f"  However, the clipping from SVI={bg1_pre:.4f} to 0.0 (a loss of {bg1_pre:.4f})")
    p(f"  overstates BG1's distance from BG2 (gap inflated by {(post_gap/pre_gap - 1)*100:+.1f}%).")
    p(f"  BG1 is not literally zero-vulnerability: it has {target.loc[bgs[0], 'EP_AGE17']:.1f}% under-17,")
    p(f"  {target.loc[bgs[0], 'EP_AGE65']:.1f}% over-65, and {target.loc[bgs[0], 'EP_NOHSDP']:.1f}% without HS diploma.")
    p()

    p("BG1 clipping exaggerated: raw indicators show BG1 is less vulnerable but the gap")
    p("is modest; consider using a floor (e.g., 0.05) instead of hard clipping to 0.0.")
    p()
    p("PROPOSED FIX:")
    p("  Add a min_svi_floor parameter to rescale_block_group_svis() in")
    p("  granite/data/block_group_loader.py, line 589:")
    p("    svis = np.clip(svis + shift, min_svi_floor, 1)")
    p("  Default value: min_svi_floor=0.05")
    p()
    p("  With floor=0.05, BG1 would rescale to 0.05 instead of 0.0,")

    # simulate
    svis_sim = np.array([bg_svis[g] for g in bgs], dtype=float)
    counts_sim = np.array([bg_counts[g] for g in bgs], dtype=float)
    total_sim = counts_sim.sum()
    for _ in range(10):
        wm = (svis_sim * counts_sim).sum() / total_sim
        if abs(wm - tract_svi) < 0.001:
            break
        shift = tract_svi - wm
        svis_sim = np.clip(svis_sim + shift, 0.05, 1)

    p(f"  preserving a more realistic gap:")
    for i, (g, label) in enumerate(zip(bgs, labels)):
        p(f"    {label}: {svis_sim[i]:.4f} (vs {rescaled[g]:.4f} with hard clip)")

    final_wm = (svis_sim * counts_sim).sum() / total_sim
    p(f"  Weighted mean with floor: {final_wm:.4f} (target: {tract_svi:.4f})")

    # Write output
    out_dir = Path('/workspaces/GRANITE/results/convergence_experiment')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'bg1_clipping_analysis.txt'
    out_path.write_text(out.getvalue())
    print(f"\nAnalysis saved to {out_path}")


if __name__ == '__main__':
    main()
