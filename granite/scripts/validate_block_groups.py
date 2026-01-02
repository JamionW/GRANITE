"""
GRANITE Block Group Validation Runner

Validates spatial disaggregation by comparing address-level predictions
aggregated to block groups against actual ACS-derived SVI.

Usage:
    python -m granite.scripts.validate_block_groups --fips 47065000600 -v
"""
import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import stats
from typing import Dict, Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class BlockGroupValidation:
    """
    Validates disaggregation quality using Census block group data.
    
    The key insight: if address-level predictions capture real vulnerability
    gradients, then aggregating them to block groups should correlate with
    actual block-group-level SVI computed from ACS demographics.
    """
    
    def __init__(self, data_dir: str = './data', verbose: bool = False):
        self.data_dir = data_dir
        self.verbose = verbose
        self.block_groups = None
        self.bg_svi = None
    
    def _log(self, msg: str):
        if self.verbose:
            print(f"[BlockGroupValidation] {msg}")
    
    def load_block_groups(self, state_fips: str = '47', county_fips: str = '065') -> gpd.GeoDataFrame:
        """Load block group geometries."""
        bg_file = os.path.join(self.data_dir, 'raw', f'tl_2020_{state_fips}_bg.shp')
        
        if not os.path.exists(bg_file):
            raise FileNotFoundError(
                f"Block group shapefile not found: {bg_file}\n"
                f"Download from: https://www.census.gov/cgi-bin/geo/shapefiles/index.php\n"
                f"Select Year=2020, Layer=Block Groups, State=Tennessee"
            )
        
        self._log(f"Loading block groups from {bg_file}")
        bg_gdf = gpd.read_file(bg_file)
        
        # Filter to county
        county_bg = bg_gdf[bg_gdf['COUNTYFP'] == county_fips].copy()
        county_bg['GEOID'] = county_bg['GEOID'].astype(str)
        county_bg['tract_fips'] = state_fips + county_fips + county_bg['TRACTCE'].astype(str)
        
        if county_bg.crs is None:
            county_bg.set_crs(epsg=4326, inplace=True)
        elif county_bg.crs != 'EPSG:4326':
            county_bg = county_bg.to_crs(epsg=4326)
        
        self._log(f"Loaded {len(county_bg)} block groups for county {county_fips}")
        self.block_groups = county_bg
        return county_bg
    
    def load_block_group_svi(self) -> pd.DataFrame:
        """Load precomputed block group SVI from ACS data."""
        svi_file = os.path.join(self.data_dir, 'processed', 'acs_block_groups_svi.csv')
        
        if not os.path.exists(svi_file):
            raise FileNotFoundError(
                f"Block group SVI file not found: {svi_file}\n"
                f"Run block_group_loader.py with Census API key to generate."
            )
        
        self._log(f"Loading block group SVI from {svi_file}")
        svi_df = pd.read_csv(svi_file, dtype={'GEOID': str})
        
        # Check for required columns
        required = ['GEOID', 'SVI']
        missing = [c for c in required if c not in svi_df.columns]
        if missing:
            raise ValueError(f"Missing columns in SVI file: {missing}")
        
        valid_svi = svi_df['SVI'].notna().sum()
        self._log(f"Loaded SVI for {valid_svi}/{len(svi_df)} block groups")
        
        self.bg_svi = svi_df
        return svi_df
    
    def assign_addresses_to_block_groups(self, 
                                         addresses: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Spatially join addresses to block groups."""
        if self.block_groups is None:
            raise ValueError("Block groups not loaded. Call load_block_groups() first.")
        
        self._log(f"Assigning {len(addresses)} addresses to block groups...")
        
        # Ensure consistent CRS
        if addresses.crs != self.block_groups.crs:
            addresses = addresses.to_crs(self.block_groups.crs)
        
        # Spatial join
        joined = gpd.sjoin(
            addresses,
            self.block_groups[['GEOID', 'geometry']],
            how='left',
            predicate='within'
        )
        
        joined = joined.rename(columns={'GEOID': 'block_group_id'})
        
        # Report coverage
        n_matched = joined['block_group_id'].notna().sum()
        self._log(f"Matched {n_matched}/{len(addresses)} addresses to block groups")
        
        if 'index_right' in joined.columns:
            joined = joined.drop(columns=['index_right'])
        
        return joined
    
    def aggregate_to_block_groups(self,
                                  addresses: gpd.GeoDataFrame,
                                  predictions: np.ndarray,
                                  method_name: str = 'GNN') -> pd.DataFrame:
        """Aggregate address-level predictions to block group means."""
        
        # Assign addresses to block groups if not already done
        if 'block_group_id' not in addresses.columns:
            addresses = self.assign_addresses_to_block_groups(addresses)
        
        df = pd.DataFrame({
            'block_group_id': addresses['block_group_id'],
            'prediction': predictions
        })
        
        # Drop addresses without block group assignment
        df = df.dropna(subset=['block_group_id'])
        
        # Aggregate
        agg = df.groupby('block_group_id').agg(
            predicted_svi=('prediction', 'mean'),
            prediction_std=('prediction', 'std'),
            n_addresses=('prediction', 'count')
        ).reset_index()
        
        agg.columns = ['GEOID', f'{method_name}_predicted_svi', 
                       f'{method_name}_std', f'{method_name}_n']
        
        self._log(f"Aggregated {method_name} to {len(agg)} block groups")
        return agg
    
    def validate(self,
                 addresses: gpd.GeoDataFrame,
                 gnn_predictions: np.ndarray,
                 idw_predictions: np.ndarray,
                 kriging_predictions: np.ndarray = None,
                 tract_fips: str = None) -> Dict:
        """
        Run full validation comparing methods against ground truth.
        
        Args:
            addresses: GeoDataFrame with address points
            gnn_predictions: GNN predicted SVI values
            idw_predictions: IDW predicted SVI values
            kriging_predictions: Optional Kriging predictions
            tract_fips: Optional tract filter
        
        Returns:
            Dict with validation results for each method
        """
        # Load ground truth if not already loaded
        if self.block_groups is None:
            self.load_block_groups()
        if self.bg_svi is None:
            self.load_block_group_svi()
        
        # Filter to tract if specified
        if tract_fips:
            tract_bgs = self.block_groups[self.block_groups['tract_fips'] == tract_fips]
            self._log(f"Filtering to {len(tract_bgs)} block groups in tract {tract_fips}")
            bg_geoids = set(tract_bgs['GEOID'].values)
        else:
            bg_geoids = set(self.block_groups['GEOID'].values)
        
        # Aggregate predictions
        gnn_agg = self.aggregate_to_block_groups(addresses.copy(), gnn_predictions, 'GNN')
        idw_agg = self.aggregate_to_block_groups(addresses.copy(), idw_predictions, 'IDW')
        
        if kriging_predictions is not None:
            kriging_agg = self.aggregate_to_block_groups(addresses.copy(), kriging_predictions, 'Kriging')
        
        # Merge with ground truth
        merged = self.bg_svi[['GEOID', 'SVI']].copy()
        merged = merged[merged['GEOID'].isin(bg_geoids)]
        merged = merged.merge(gnn_agg, on='GEOID', how='inner')
        merged = merged.merge(idw_agg, on='GEOID', how='inner')
        
        if kriging_predictions is not None:
            merged = merged.merge(kriging_agg, on='GEOID', how='inner')
        
        # Filter to block groups with valid SVI
        merged = merged[merged['SVI'].notna()]
        
        self._log(f"Validating on {len(merged)} block groups with complete data")
        
        if len(merged) < 3:
            return {
                'success': False,
                'error': f'Only {len(merged)} block groups with valid data - need at least 3',
                'n_block_groups': len(merged)
            }
        
        # Compute correlations
        results = {
            'success': True,
            'n_block_groups': len(merged),
            'ground_truth_svi_range': [float(merged['SVI'].min()), float(merged['SVI'].max())],
            'methods': {}
        }
        
        for method in ['GNN', 'IDW'] + (['Kriging'] if kriging_predictions is not None else []):
            pred_col = f'{method}_predicted_svi'
            
            # Pearson correlation
            r, p = stats.pearsonr(merged['SVI'], merged[pred_col])
            
            # Spearman correlation (rank-based, more robust)
            rho, p_spearman = stats.spearmanr(merged['SVI'], merged[pred_col])
            
            results['methods'][method] = {
                'pearson_r': float(r),
                'pearson_p': float(p),
                'spearman_rho': float(rho),
                'spearman_p': float(p_spearman),
                'predicted_range': [
                    float(merged[pred_col].min()),
                    float(merged[pred_col].max())
                ],
                'predicted_std': float(merged[pred_col].std()),
                'n_addresses': int(merged[f'{method}_n'].sum())
            }
        
        # Store merged data for plotting
        results['validation_data'] = merged
        
        return results
    
    def print_report(self, results: Dict):
        """Print formatted validation report."""
        if not results.get('success'):
            print(f"\nValidation failed: {results.get('error')}")
            return
        
        print("\n" + "=" * 70)
        print("BLOCK GROUP VALIDATION REPORT")
        print("=" * 70)
        print(f"\nBlock groups validated: {results['n_block_groups']}")
        print(f"Ground truth SVI range: {results['ground_truth_svi_range'][0]:.3f} - {results['ground_truth_svi_range'][1]:.3f}")
        
        print(f"\n{'Method':<12} {'Pearson r':>12} {'p-value':>12} {'Spearman ρ':>12} {'Addresses':>12}")
        print("-" * 60)
        
        for method, stats in results['methods'].items():
            r = stats['pearson_r']
            p = stats['pearson_p']
            rho = stats['spearman_rho']
            n = stats['n_addresses']
            
            # Significance indicator
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            
            print(f"{method:<12} {r:>11.3f}{sig} {p:>12.4f} {rho:>12.3f} {n:>12,}")
        
        print("\n" + "-" * 70)
        print("INTERPRETATION")
        print("-" * 70)
        
        gnn_r = results['methods']['GNN']['pearson_r']
        idw_r = results['methods']['IDW']['pearson_r']
        
        if gnn_r > idw_r + 0.05:
            print(f"✓ GNN outperforms IDW (r={gnn_r:.3f} vs r={idw_r:.3f})")
            print("  → GNN captures real vulnerability gradients that IDW misses")
        elif idw_r > gnn_r + 0.05:
            print(f"✗ IDW outperforms GNN (r={idw_r:.3f} vs r={gnn_r:.3f})")
            print("  → GNN's extra variation may be noise, not signal")
        else:
            print(f"~ GNN and IDW perform similarly (r={gnn_r:.3f} vs r={idw_r:.3f})")
            print("  → Both capture similar spatial patterns")
        
        if gnn_r > 0.5:
            print("\n★ Strong validation: GNN predictions align well with ground truth")
        elif gnn_r > 0.3:
            print("\n◉ Moderate validation: GNN captures meaningful patterns")
        elif gnn_r > 0.15:
            print("\n○ Weak validation: Some signal detected but limited")
        else:
            print("\n✗ No meaningful correlation with ground truth")
        
        print("=" * 70)
    
    def create_validation_plots(self, 
                                results: Dict,
                                output_dir: str = './output') -> plt.Figure:
        """Create validation visualization."""
        if not results.get('success'):
            print("Cannot create plots - validation failed")
            return None
        
        merged = results['validation_data']
        
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        fig.suptitle('Block Group Validation: Predicted vs Ground Truth SVI', 
                     fontsize=14, fontweight='bold')
        
        methods = list(results['methods'].keys())
        colors = {'GNN': '#2E86AB', 'IDW': '#A23B72', 'Kriging': '#F18F01'}
        
        # Scatter plots for each method
        for i, method in enumerate(methods[:3]):
            ax = fig.add_subplot(gs[0, i])
            
            pred_col = f'{method}_predicted_svi'
            r = results['methods'][method]['pearson_r']
            p = results['methods'][method]['pearson_p']
            
            ax.scatter(merged['SVI'], merged[pred_col], 
                      alpha=0.6, s=50, c=colors.get(method, 'gray'), edgecolor='white')
            
            # Regression line
            slope, intercept = np.polyfit(merged['SVI'], merged[pred_col], 1)
            x_line = np.array([merged['SVI'].min(), merged['SVI'].max()])
            ax.plot(x_line, slope * x_line + intercept, 'k--', alpha=0.5)
            
            # Diagonal reference
            ax.plot([0, 1], [0, 1], 'gray', linestyle=':', alpha=0.5)
            
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            ax.set_title(f'{method}\nr = {r:.3f}{sig}')
            ax.set_xlabel('Ground Truth SVI')
            ax.set_ylabel('Predicted SVI')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
        
        # Bar chart comparing correlations
        ax4 = fig.add_subplot(gs[1, 0])
        method_names = list(results['methods'].keys())
        correlations = [results['methods'][m]['pearson_r'] for m in method_names]
        bars = ax4.bar(method_names, correlations, 
                      color=[colors.get(m, 'gray') for m in method_names],
                      edgecolor='black', alpha=0.8)
        ax4.axhline(0, color='black', linewidth=0.5)
        ax4.set_ylabel('Pearson Correlation (r)')
        ax4.set_title('Method Comparison')
        ax4.set_ylim(min(0, min(correlations) - 0.1), max(correlations) + 0.1)
        
        # Add value labels
        for bar, corr in zip(bars, correlations):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{corr:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Residual plot for GNN
        ax5 = fig.add_subplot(gs[1, 1])
        gnn_pred = merged['GNN_predicted_svi']
        residuals = merged['SVI'] - gnn_pred
        ax5.scatter(gnn_pred, residuals, alpha=0.6, s=30, c=colors['GNN'])
        ax5.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax5.set_xlabel('GNN Predicted SVI')
        ax5.set_ylabel('Residual (Truth - Predicted)')
        ax5.set_title('GNN Residuals')
        
        # Summary text
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        
        summary_text = [
            f"Block Groups: {results['n_block_groups']}",
            f"",
            f"Ground Truth Range:",
            f"  {results['ground_truth_svi_range'][0]:.3f} - {results['ground_truth_svi_range'][1]:.3f}",
            f"",
            "Method Performance:",
        ]
        
        for method, stats in results['methods'].items():
            r = stats['pearson_r']
            sig = '***' if stats['pearson_p'] < 0.001 else '**' if stats['pearson_p'] < 0.01 else '*' if stats['pearson_p'] < 0.05 else ''
            summary_text.append(f"  {method}: r = {r:.3f}{sig}")
        
        summary_text.extend([
            "",
            "Significance: * p<0.05, ** p<0.01, *** p<0.001"
        ])
        
        ax6.text(0.1, 0.9, '\n'.join(summary_text), transform=ax6.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        # Save
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, 'block_group_validation.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Validation plot saved to: {filepath}")
        
        return fig


def run_validation_from_results(results: Dict, 
                                data_dir: str = './data',
                                output_dir: str = './output',
                                verbose: bool = True) -> Dict:
    """
    Run block group validation from pipeline results.
    
    Args:
        results: Output from GRANITEPipeline.run()
        data_dir: Data directory with block group files
        output_dir: Output directory for plots
        verbose: Print progress
    
    Returns:
        Validation results dict
    """
    if not results.get('success'):
        print("Cannot validate - pipeline did not succeed")
        return {'success': False, 'error': 'Pipeline failed'}
    
    validator = BlockGroupValidation(data_dir=data_dir, verbose=verbose)
    
    addresses = results['addresses']
    gnn_preds = results['predictions']
    
    # Get baseline predictions
    baselines = results.get('baselines', {})
    idw_preds = baselines.get('idw', {}).get('predictions')
    kriging_preds = baselines.get('kriging', {}).get('predictions')
    
    if idw_preds is None:
        print("Warning: IDW predictions not available - using uniform")
        idw_preds = np.full(len(gnn_preds), results['tract_svi'])
    
    # Get tract FIPS from addresses
    tract_fips = None
    if 'tract_fips' in addresses.columns:
        tract_fips = addresses['tract_fips'].iloc[0]
    
    # Run validation
    validation_results = validator.validate(
        addresses=addresses,
        gnn_predictions=gnn_preds,
        idw_predictions=idw_preds,
        kriging_predictions=kriging_preds,
        tract_fips=tract_fips
    )
    
    # Print report
    validator.print_report(validation_results)
    
    # Create plots
    if validation_results.get('success'):
        validator.create_validation_plots(validation_results, output_dir)
    
    return validation_results


# CLI entry point
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='GRANITE Block Group Validation')
    parser.add_argument('--fips', type=str, required=True, help='Tract FIPS code')
    parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--output', type=str, default='./output', help='Output directory')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("GRANITE Block Group Validation")
    print("=" * 60)
    print(f"Tract: {args.fips}")
    print("=" * 60)
    
    # First run the pipeline
    print("\nStep 1: Running GRANITE pipeline...")
    from granite.disaggregation.pipeline import GRANITEPipeline
    from granite.models.gnn import set_random_seed
    
    config = {
        'data': {
            'target_fips': args.fips,
            'state_fips': args.fips[:2],
            'county_fips': args.fips[2:5],
        },
        'model': {'hidden_dim': 32, 'k_neighbors': 8},
        'training': {'epochs': 100},
        'processing': {'verbose': args.verbose, 'random_seed': 42}
    }
    
    set_random_seed(42)
    pipeline = GRANITEPipeline(config, data_dir=args.data_dir, output_dir=args.output)
    results = pipeline.run()
    
    if not results.get('success'):
        print(f"Pipeline failed: {results.get('error')}")
        return 1
    
    # Run validation
    print("\nStep 2: Running block group validation...")
    validation_results = run_validation_from_results(
        results, 
        data_dir=args.data_dir,
        output_dir=args.output,
        verbose=args.verbose
    )
    
    return 0 if validation_results.get('success') else 1


if __name__ == '__main__':
    sys.exit(main())