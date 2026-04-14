"""
Diagnose constraint inconsistency between block group SVIs and tract SVI.
"""
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
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

    # load block group SVIs
    bg_loader = BlockGroupLoader(data_dir='./data', verbose=False)
    bg_data = bg_loader.get_block_groups_with_demographics(svi_ranking_scope='county')

    # load addresses and assign to block groups
    addresses = data_loader.get_addresses_for_tract(TARGET_FIPS)
    addresses_with_bg = bg_loader.assign_addresses_to_block_groups(addresses, bg_data)

    # filter to block groups in this tract
    tract_bgs = bg_data[bg_data['GEOID'].str.startswith(TARGET_FIPS)].copy()

    print(f"\nCONSTRAINT CONSISTENCY CHECK: Tract {TARGET_FIPS}\n")
    print(f"Tract SVI (CDC):           {tract_svi:.4f}")

    # per-block-group breakdown
    rows = []
    for _, bg_row in tract_bgs.iterrows():
        bg_id = bg_row['GEOID']
        svi_val = bg_row.get('SVI', np.nan)
        complete = bg_row.get('svi_complete', False)
        n_addr = (addresses_with_bg['block_group_id'] == bg_id).sum()
        rows.append({
            'GEOID': bg_id,
            'n_addr': int(n_addr),
            'BG_SVI': float(svi_val) if not pd.isna(svi_val) else np.nan,
            'complete': complete,
        })

    # compute weighted mean using only complete BGs with addresses
    valid = [r for r in rows if not np.isnan(r['BG_SVI']) and r['n_addr'] > 0]
    total_addr = sum(r['n_addr'] for r in valid)
    weighted_mean = sum(r['BG_SVI'] * r['n_addr'] for r in valid) / total_addr if total_addr > 0 else np.nan
    discrepancy = abs(weighted_mean - tract_svi)
    pct_discrepancy = discrepancy / tract_svi * 100 if tract_svi > 0 else 0

    print(f"BG weighted mean SVI:      {weighted_mean:.4f}")
    print(f"Discrepancy:               {discrepancy:.4f} ({pct_discrepancy:.1f}%)")

    print(f"\nPer-block-group breakdown:")
    print(f"{'GEOID':20s} {'n_addr':>8s} {'BG SVI':>10s} {'Contribution':>14s} {'Complete':>10s}")
    for r in sorted(rows, key=lambda x: x['GEOID']):
        contrib = r['BG_SVI'] * r['n_addr'] / total_addr if total_addr > 0 and not np.isnan(r['BG_SVI']) else 0
        print(f"{r['GEOID']:20s} {r['n_addr']:>8d} {r['BG_SVI']:>10.4f} {contrib:>14.4f} {str(r['complete']):>10s}")

    if pct_discrepancy < 2.0:
        print("\nConstraints are approximately consistent; rescaling unnecessary")
    else:
        print(f"\nConstraints are INCONSISTENT ({pct_discrepancy:.1f}% > 2% threshold)")
        print("Rescaling is needed to make block group SVIs consistent with tract SVI")

        # show what the shift would be
        shift = tract_svi - weighted_mean
        print(f"\nRequired additive shift: {shift:+.4f}")
        print(f"Rescaled BG SVIs would be:")
        for r in sorted(valid, key=lambda x: x['GEOID']):
            rescaled = np.clip(r['BG_SVI'] + shift, 0, 1)
            print(f"  {r['GEOID']}: {r['BG_SVI']:.4f} -> {rescaled:.4f}")


if __name__ == '__main__':
    main()
