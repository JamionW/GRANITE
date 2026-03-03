"""
Tract inventory: List all available tracts with address counts and SVI values.
"""
import sys
sys.path.insert(0, '/workspaces/GRANITE')

import pandas as pd
from granite.data.loaders import DataLoader

def inventory_tracts():
 """Generate complete tract inventory for manual curation"""
 
 loader = DataLoader()
 
 print("\nLoading tract data...")
 tracts = loader.load_census_tracts('47', '065')
 svi = loader.load_svi_data('47', 'Hamilton')
 
 # Merge to get SVI values
 tract_data = tracts.merge(svi, on='FIPS', how='inner')
 tract_data = tract_data[tract_data['RPL_THEMES'].notna()].copy()
 
 # Sort by SVI for easy selection
 tract_data = tract_data.sort_values('RPL_THEMES')
 
 print(f"\n{'='*80}")
 print("HAMILTON COUNTY TRACT INVENTORY")
 print(f"{'='*80}")
 print(f"Total tracts with valid SVI: {len(tract_data)}\n")
 
 # Collect results
 results = []
 
 for idx, row in tract_data.iterrows():
 fips = row['FIPS']
 svi = row['RPL_THEMES']
 
 try:
 addresses = loader.get_addresses_for_tract(fips)
 n_addresses = len(addresses)
 status = "" if n_addresses >= 50 else "" if n_addresses > 0 else ""
 except Exception as e:
 n_addresses = 0
 status = ""
 
 results.append({
 'FIPS': fips,
 'SVI': svi,
 'Addresses': n_addresses,
 'Status': status
 })
 
 # Create DataFrame for nice display
 results_df = pd.DataFrame(results)
 
 # Print full list
 print(f"{'Status':<8} {'FIPS':<15} {'SVI':<8} {'Addresses':<12}")
 print("-" * 80)
 
 for _, row in results_df.iterrows():
 print(f"{row['Status']:<8} {row['FIPS']:<15} {row['SVI']:.4f} {row['Addresses']:<12,}")
 
 # Summary statistics
 print(f"\n{'='*80}")
 print("SUMMARY")
 print(f"{'='*80}")
 
 good_tracts = results_df[results_df['Addresses'] >= 50]
 some_tracts = results_df[(results_df['Addresses'] > 0) & (results_df['Addresses'] < 50)]
 no_tracts = results_df[results_df['Addresses'] == 0]
 
 print(f" Good tracts (≥50 addresses): {len(good_tracts)} tracts")
 print(f" Sparse tracts (1-49 addresses): {len(some_tracts)} tracts")
 print(f" Empty tracts (0 addresses): {len(no_tracts)} tracts")
 
 print(f"\nGood tracts SVI range: {good_tracts['SVI'].min():.3f} - {good_tracts['SVI'].max():.3f}")
 print(f"Good tracts address range: {good_tracts['Addresses'].min():,} - {good_tracts['Addresses'].max():,}")
 
 # Print SVI quintile breakdown
 print(f"\n{'='*80}")
 print("GOOD TRACTS BY SVI QUINTILE")
 print(f"{'='*80}")
 
 good_tracts['SVI_Quintile'] = pd.qcut(
 good_tracts['SVI'], 
 q=5, 
 labels=['Very Low (0-20%)', 'Low (20-40%)', 'Medium (40-60%)', 'High (60-80%)', 'Very High (80-100%)'],
 duplicates='drop'
 )
 
 for quintile in good_tracts['SVI_Quintile'].unique():
 quintile_tracts = good_tracts[good_tracts['SVI_Quintile'] == quintile]
 print(f"\n{quintile}: {len(quintile_tracts)} tracts")
 print(f" SVI range: {quintile_tracts['SVI'].min():.3f} - {quintile_tracts['SVI'].max():.3f}")
 print(f" Example tracts:")
 for _, tract in quintile_tracts.head(3).iterrows():
 print(f" {tract['FIPS']}: SVI={tract['SVI']:.3f}, {tract['Addresses']:,} addresses")
 
 # Save to CSV for easy reference
 output_file = './tract_inventory.csv'
 results_df.to_csv(output_file, index=False)
 print(f"\n{'='*80}")
 print(f"Full inventory saved to: {output_file}")
 print(f"{'='*80}\n")
 
 return results_df

if __name__ == '__main__':
 inventory = inventory_tracts()