"""
Consistency check: national vs county block group SVI rankings for tract 47065000600.
"""
import sys
import os
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, '/workspaces/GRANITE')

from granite.data.loaders import DataLoader
from granite.data.block_group_loader import BlockGroupLoader

TARGET_FIPS = '47065000600'


def main():
    with open('/workspaces/GRANITE/config.yaml') as f:
        config = yaml.safe_load(f)

    # load tract SVI from CDC data
    data_loader = DataLoader('./data', config=config)
    state_fips = config['data']['state_fips']
    county_fips = config['data']['county_fips']
    county_name = data_loader._get_county_name(state_fips, county_fips)
    svi = data_loader.load_svi_data(state_fips, county_name)
    tract_row = svi[svi['FIPS'] == TARGET_FIPS]
    if len(tract_row) == 0:
        print(f"Tract {TARGET_FIPS} not found in SVI data")
        return
    tract_svi = float(tract_row.iloc[0]['RPL_THEMES'])

    bg_loader = BlockGroupLoader(data_dir='./data', verbose=True)

    # load both national and county rankings
    bg_national = bg_loader.get_block_groups_with_demographics(svi_ranking_scope='national')
    bg_county = bg_loader.get_block_groups_with_demographics(svi_ranking_scope='county')

    # load addresses and assign to block groups
    addresses = data_loader.get_addresses_for_tract(TARGET_FIPS)
    addresses_with_bg = bg_loader.assign_addresses_to_block_groups(addresses, bg_national)

    # filter to target tract
    tract_bgs_national = bg_national[bg_national['GEOID'].str.startswith(TARGET_FIPS)].copy()
    tract_bgs_county = bg_county[bg_county['GEOID'].str.startswith(TARGET_FIPS)].copy()

    # build per-BG data
    rows = []
    for _, bg_row in tract_bgs_national.iterrows():
        bg_id = bg_row['GEOID']
        nat_svi = bg_row.get('SVI', np.nan)
        n_addr = int((addresses_with_bg['block_group_id'] == bg_id).sum())

        county_row = tract_bgs_county[tract_bgs_county['GEOID'] == bg_id]
        county_svi = float(county_row.iloc[0]['SVI']) if len(county_row) > 0 else np.nan

        rows.append({
            'GEOID': bg_id,
            'n_addr': n_addr,
            'national_svi': float(nat_svi) if not pd.isna(nat_svi) else np.nan,
            'county_svi': county_svi,
        })

    valid = [r for r in rows if not np.isnan(r['national_svi']) and r['n_addr'] > 0]
    total_addr = sum(r['n_addr'] for r in valid)
    weighted_mean = sum(r['national_svi'] * r['n_addr'] for r in valid) / total_addr if total_addr > 0 else np.nan
    discrepancy = abs(weighted_mean - tract_svi)
    pct_discrepancy = discrepancy / tract_svi * 100 if tract_svi > 0 else 0
    rescaling_shift = tract_svi - weighted_mean

    # count national BGs for context
    national_svi_df = bg_loader.compute_national_svi()
    n_national = len(national_svi_df)

    lines = []
    lines.append(f"NATIONAL RANKING CONSISTENCY CHECK: Tract {TARGET_FIPS}")
    lines.append("")
    lines.append(f"Ranking scope:             National (~{n_national // 1000}K block groups)")
    lines.append(f"Tract SVI (CDC):           {tract_svi:.4f}")
    lines.append(f"BG weighted mean SVI:      {weighted_mean:.4f}")
    lines.append(f"Discrepancy:               {discrepancy:.4f} ({pct_discrepancy:.1f}%)")
    lines.append("")
    lines.append("Per-block-group:")
    lines.append(f"{'GEOID':20s} {'n_addr':>8s} {'National SVI':>14s} {'County SVI (old)':>18s} {'Difference':>12s}")

    for r in sorted(rows, key=lambda x: x['GEOID']):
        diff = r['national_svi'] - r['county_svi'] if not np.isnan(r['national_svi']) and not np.isnan(r['county_svi']) else np.nan
        diff_str = f"{diff:+.4f}" if not np.isnan(diff) else "N/A"
        nat_str = f"{r['national_svi']:.4f}" if not np.isnan(r['national_svi']) else "N/A"
        cty_str = f"{r['county_svi']:.4f}" if not np.isnan(r['county_svi']) else "N/A"
        lines.append(f"{r['GEOID']:20s} {r['n_addr']:>8d} {nat_str:>14s} {cty_str:>18s} {diff_str:>12s}")

    lines.append("")
    lines.append(f"Rescaling shift needed:    {rescaling_shift:+.4f} (vs -0.1806 with county ranking)")
    lines.append("")

    # check BG1 specifically
    bg1 = [r for r in rows if r['GEOID'].endswith('1')]
    if bg1:
        bg1_svi = bg1[0]['national_svi']
        if not np.isnan(bg1_svi):
            if bg1_svi > 0.05:
                lines.append(f"BG1 nationally-ranked SVI: {bg1_svi:.4f} (above 0.05 floor)")
                lines.append("The clipping problem is resolved without needing an artificial floor.")
            else:
                lines.append(f"BG1 nationally-ranked SVI: {bg1_svi:.4f} (still below 0.05 floor)")
                lines.append("BG1 clipping may still be an issue even with national ranking.")

    report = '\n'.join(lines)
    print('\n' + report)

    # save report
    os.makedirs('results/convergence_experiment', exist_ok=True)
    with open('results/convergence_experiment/national_ranking_consistency.txt', 'w') as f:
        f.write(report + '\n')
    print(f"\nReport saved to results/convergence_experiment/national_ranking_consistency.txt")


if __name__ == '__main__':
    main()
