"""
GRANITE County-Wide Block Group Validation

Runs GRANITE on ALL valid tracts and aggregates block group validation
across the entire county for robust statistical conclusions.

This is the definitive test: Does GNN outperform IDW when validated
against ground truth across ~200+ block groups?

Usage:
    python -m granite.scripts.validate_county -v
    python -m granite.scripts.validate_county --max-tracts 20 -v  # Quick test
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import stats
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class CountyWideValidation:
    """
    Validates GRANITE across all tracts in a county.
    
    Aggregates predictions to block groups and correlates with
    ground truth ACS-derived SVI for definitive method comparison.
    """
    
    def __init__(self, data_dir: str = './data', output_dir: str = './output', verbose: bool = False):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.verbose = verbose
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Will be populated during run
        self.block_groups = None
        self.bg_svi = None
        self.all_results = []
        
    def _log(self, msg: str, level: str = 'INFO'):
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {level}: {msg}")
    
    def load_tract_inventory(self, min_addresses: int = 50) -> pd.DataFrame:
        """Load tract inventory and filter to usable tracts."""
        inventory_file = os.path.join(self.data_dir, '..', 'tract_inventory.csv')
        
        if not os.path.exists(inventory_file):
            # Try alternate location
            inventory_file = './tract_inventory.csv'
        
        if not os.path.exists(inventory_file):
            # Create from data
            self._log("Tract inventory not found, will discover tracts from data")
            return None
        
        inventory = pd.read_csv(inventory_file, dtype={'FIPS': str})
        
        # Filter to usable tracts
        usable = inventory[inventory['Addresses'] >= min_addresses].copy()
        self._log(f"Loaded {len(usable)} tracts with ≥{min_addresses} addresses")
        
        return usable
    
    def load_block_groups(self, state_fips: str = '47', county_fips: str = '065'):
        """Load block group geometries and SVI."""
        # Load geometries
        bg_file = os.path.join(self.data_dir, 'raw', f'tl_2020_{state_fips}_bg.shp')
        if not os.path.exists(bg_file):
            raise FileNotFoundError(f"Block group shapefile not found: {bg_file}")
        
        self._log(f"Loading block group geometries...")
        bg_gdf = gpd.read_file(bg_file)
        county_bg = bg_gdf[bg_gdf['COUNTYFP'] == county_fips].copy()
        county_bg['GEOID'] = county_bg['GEOID'].astype(str)
        county_bg['tract_fips'] = state_fips + county_fips + county_bg['TRACTCE'].astype(str)
        
        if county_bg.crs is None:
            county_bg.set_crs(epsg=4326, inplace=True)
        elif county_bg.crs != 'EPSG:4326':
            county_bg = county_bg.to_crs(epsg=4326)
        
        self.block_groups = county_bg
        self._log(f"Loaded {len(county_bg)} block group geometries")
        
        # Load SVI
        svi_file = os.path.join(self.data_dir, 'processed', 'acs_block_groups_svi.csv')
        if not os.path.exists(svi_file):
            raise FileNotFoundError(f"Block group SVI file not found: {svi_file}")
        
        self.bg_svi = pd.read_csv(svi_file, dtype={'GEOID': str})
        valid_svi = self.bg_svi['SVI'].notna().sum()
        self._log(f"Loaded SVI for {valid_svi}/{len(self.bg_svi)} block groups")
    
    def run_single_tract(self, fips: str) -> Optional[Dict]:
        """Run GRANITE pipeline on a single tract."""
        from granite.disaggregation.pipeline import GRANITEPipeline
        from granite.models.gnn import set_random_seed
        
        config = {
            'data': {
                'target_fips': fips,
                'state_fips': fips[:2],
                'county_fips': fips[2:5],
            },
            'model': {'hidden_dim': 32, 'k_neighbors': 8, 'dropout': 0.2},
            'training': {'epochs': 100, 'learning_rate': 0.001, 'constraint_weight': 2.0},
            'processing': {'verbose': False, 'random_seed': 42}
        }
        
        set_random_seed(42)
        
        try:
            pipeline = GRANITEPipeline(config, data_dir=self.data_dir, output_dir=self.output_dir)
            results = pipeline.run()
            
            if results.get('success'):
                return results
            else:
                self._log(f"Tract {fips} failed: {results.get('error')}", level='WARN')
                return None
        except Exception as e:
            self._log(f"Tract {fips} error: {str(e)}", level='ERROR')
            return None
    
    def assign_addresses_to_block_groups(self, addresses: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Spatial join addresses to block groups."""
        if addresses.crs != self.block_groups.crs:
            addresses = addresses.to_crs(self.block_groups.crs)
        
        joined = gpd.sjoin(
            addresses,
            self.block_groups[['GEOID', 'geometry']],
            how='left',
            predicate='within'
        )
        
        joined = joined.rename(columns={'GEOID': 'block_group_id'})
        
        if 'index_right' in joined.columns:
            joined = joined.drop(columns=['index_right'])
        
        return joined
    
    def run_county_validation(self, 
                              tract_list: List[str] = None,
                              max_tracts: int = None) -> Dict:
        """
        Run GRANITE on multiple tracts and aggregate validation.
        
        Args:
            tract_list: Optional list of FIPS codes (default: all usable tracts)
            max_tracts: Optional limit on number of tracts to process
        
        Returns:
            Comprehensive validation results
        """
        start_time = time.time()
        
        # Load block groups
        self.load_block_groups()
        
        # Get tract list
        if tract_list is None:
            inventory = self.load_tract_inventory(min_addresses=50)
            if inventory is not None:
                tract_list = inventory['FIPS'].tolist()
            else:
                raise ValueError("No tract list provided and inventory not found")
        
        if max_tracts:
            tract_list = tract_list[:max_tracts]
        
        self._log(f"Processing {len(tract_list)} tracts...")
        print(f"\n{'='*70}")
        print(f"GRANITE COUNTY-WIDE VALIDATION")
        print(f"{'='*70}")
        print(f"Tracts to process: {len(tract_list)}")
        print(f"{'='*70}\n")
        
        # Collect all predictions
        all_bg_data = []
        successful_tracts = 0
        failed_tracts = []
        
        for i, fips in enumerate(tract_list):
            progress = f"[{i+1}/{len(tract_list)}]"
            print(f"{progress} Processing tract {fips}...", end=" ", flush=True)
            
            results = self.run_single_tract(fips)
            
            if results is None:
                print("FAILED")
                failed_tracts.append(fips)
                continue
            
            # Get predictions
            addresses = results['addresses'].copy()
            gnn_preds = results['predictions']
            
            baselines = results.get('baselines', {})
            idw_preds = baselines.get('idw', {}).get('predictions')
            kriging_preds = baselines.get('kriging', {}).get('predictions')
            
            if idw_preds is None:
                idw_preds = np.full(len(gnn_preds), results['tract_svi'])
            if kriging_preds is None:
                kriging_preds = np.full(len(gnn_preds), results['tract_svi'])
            
            # Assign to block groups
            addresses = self.assign_addresses_to_block_groups(addresses)
            
            # Store per-address data
            for j in range(len(addresses)):
                bg_id = addresses.iloc[j].get('block_group_id')
                if pd.notna(bg_id):
                    all_bg_data.append({
                        'tract_fips': fips,
                        'block_group_id': bg_id,
                        'gnn_pred': gnn_preds[j],
                        'idw_pred': idw_preds[j],
                        'kriging_pred': kriging_preds[j],
                        'tract_svi': results['tract_svi']
                    })
            
            successful_tracts += 1
            n_addr = len(addresses)
            gnn_std = np.std(gnn_preds)
            print(f"OK ({n_addr} addresses, GNN std={gnn_std:.3f})")
        
        print(f"\n{'='*70}")
        print(f"PROCESSING COMPLETE")
        print(f"{'='*70}")
        print(f"Successful: {successful_tracts}/{len(tract_list)} tracts")
        if failed_tracts:
            print(f"Failed: {failed_tracts[:5]}{'...' if len(failed_tracts) > 5 else ''}")
        
        if not all_bg_data:
            return {'success': False, 'error': 'No data collected'}
        
        # Create DataFrame
        df = pd.DataFrame(all_bg_data)
        
        # Aggregate to block groups
        print(f"\nAggregating to block groups...")
        bg_agg = df.groupby('block_group_id').agg(
            gnn_mean=('gnn_pred', 'mean'),
            gnn_std=('gnn_pred', 'std'),
            idw_mean=('idw_pred', 'mean'),
            idw_std=('idw_pred', 'std'),
            kriging_mean=('kriging_pred', 'mean'),
            kriging_std=('kriging_pred', 'std'),
            tract_svi=('tract_svi', 'first'),
            n_addresses=('gnn_pred', 'count')
        ).reset_index()
        
        bg_agg.columns = ['GEOID', 'GNN_pred', 'GNN_std', 'IDW_pred', 'IDW_std',
                         'Kriging_pred', 'Kriging_std', 'tract_svi', 'n_addresses']
        
        # Merge with ground truth
        merged = bg_agg.merge(
            self.bg_svi[['GEOID', 'SVI']], 
            on='GEOID', 
            how='inner'
        )
        
        # Filter to valid SVI
        merged = merged[merged['SVI'].notna()]
        
        print(f"Block groups with ground truth: {len(merged)}")
        print(f"Total addresses covered: {merged['n_addresses'].sum():,}")
        
        # Compute correlations
        results = self._compute_validation_stats(merged)
        results['processing_time'] = time.time() - start_time
        results['n_tracts_processed'] = successful_tracts
        results['n_tracts_failed'] = len(failed_tracts)
        results['validation_data'] = merged
        results['raw_data'] = df
        
        return results
    
    def _compute_validation_stats(self, merged: pd.DataFrame) -> Dict:
        """Compute comprehensive validation statistics."""
        results = {
            'success': True,
            'n_block_groups': len(merged),
            'n_addresses': int(merged['n_addresses'].sum()),
            'ground_truth_svi_range': [
                float(merged['SVI'].min()),
                float(merged['SVI'].max())
            ],
            'ground_truth_svi_mean': float(merged['SVI'].mean()),
            'methods': {}
        }
        
        for method in ['GNN', 'IDW', 'Kriging']:
            pred_col = f'{method}_pred'
            
            # Pearson correlation
            r, p = stats.pearsonr(merged['SVI'], merged[pred_col])
            
            # Spearman correlation
            rho, p_spearman = stats.spearmanr(merged['SVI'], merged[pred_col])
            
            # R-squared
            r_squared = r ** 2
            
            # Root mean squared error
            rmse = np.sqrt(np.mean((merged['SVI'] - merged[pred_col]) ** 2))
            
            # Mean absolute error
            mae = np.mean(np.abs(merged['SVI'] - merged[pred_col]))
            
            results['methods'][method] = {
                'pearson_r': float(r),
                'pearson_p': float(p),
                'spearman_rho': float(rho),
                'spearman_p': float(p_spearman),
                'r_squared': float(r_squared),
                'rmse': float(rmse),
                'mae': float(mae),
                'predicted_mean': float(merged[pred_col].mean()),
                'predicted_std': float(merged[pred_col].std()),
                'predicted_range': [
                    float(merged[pred_col].min()),
                    float(merged[pred_col].max())
                ]
            }
        
        # Comparative stats
        gnn_r = results['methods']['GNN']['pearson_r']
        idw_r = results['methods']['IDW']['pearson_r']
        kriging_r = results['methods']['Kriging']['pearson_r']
        
        results['comparison'] = {
            'gnn_vs_idw_r_diff': float(gnn_r - idw_r),
            'gnn_vs_kriging_r_diff': float(gnn_r - kriging_r),
            'best_method': max(['GNN', 'IDW', 'Kriging'], 
                              key=lambda m: results['methods'][m]['pearson_r']),
            'worst_method': min(['GNN', 'IDW', 'Kriging'],
                               key=lambda m: results['methods'][m]['pearson_r'])
        }
        
        # Statistical test: is GNN significantly better than IDW?
        # Using Fisher z-transformation for comparing correlations
        n = len(merged)
        z_gnn = np.arctanh(gnn_r)
        z_idw = np.arctanh(idw_r)
        se_diff = np.sqrt(2 / (n - 3))
        z_stat = (z_gnn - z_idw) / se_diff
        p_diff = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        results['comparison']['gnn_vs_idw_z_stat'] = float(z_stat)
        results['comparison']['gnn_vs_idw_p_value'] = float(p_diff)
        results['comparison']['gnn_significantly_better'] = (gnn_r > idw_r) and (p_diff < 0.05)
        
        return results
    
    def print_report(self, results: Dict):
        """Print comprehensive validation report."""
        if not results.get('success'):
            print(f"\nValidation failed: {results.get('error')}")
            return
        
        print(f"\n{'='*75}")
        print("COUNTY-WIDE BLOCK GROUP VALIDATION REPORT")
        print(f"{'='*75}")
        
        print(f"\nSAMPLE SIZE:")
        print(f"  Tracts processed: {results['n_tracts_processed']}")
        print(f"  Block groups validated: {results['n_block_groups']}")
        print(f"  Total addresses: {results['n_addresses']:,}")
        print(f"  Processing time: {results['processing_time']:.1f} seconds")
        
        print(f"\nGROUND TRUTH DISTRIBUTION:")
        print(f"  SVI range: {results['ground_truth_svi_range'][0]:.3f} - {results['ground_truth_svi_range'][1]:.3f}")
        print(f"  SVI mean: {results['ground_truth_svi_mean']:.3f}")
        
        print(f"\n{'-'*75}")
        print("METHOD COMPARISON")
        print(f"{'-'*75}")
        print(f"{'Method':<10} {'Pearson r':>12} {'p-value':>12} {'Spearman ρ':>12} {'RMSE':>10} {'R²':>10}")
        print("-" * 68)
        
        for method in ['GNN', 'IDW', 'Kriging']:
            s = results['methods'][method]
            sig = '***' if s['pearson_p'] < 0.001 else '**' if s['pearson_p'] < 0.01 else '*' if s['pearson_p'] < 0.05 else ''
            print(f"{method:<10} {s['pearson_r']:>11.4f}{sig} {s['pearson_p']:>12.4f} "
                  f"{s['spearman_rho']:>12.4f} {s['rmse']:>10.4f} {s['r_squared']:>10.4f}")
        
        print(f"\n{'-'*75}")
        print("STATISTICAL COMPARISON: GNN vs IDW")
        print(f"{'-'*75}")
        
        comp = results['comparison']
        gnn_r = results['methods']['GNN']['pearson_r']
        idw_r = results['methods']['IDW']['pearson_r']
        
        print(f"  GNN r = {gnn_r:.4f}")
        print(f"  IDW r = {idw_r:.4f}")
        print(f"  Difference: {comp['gnn_vs_idw_r_diff']:+.4f}")
        print(f"  Z-statistic: {comp['gnn_vs_idw_z_stat']:.3f}")
        print(f"  p-value: {comp['gnn_vs_idw_p_value']:.4f}")
        
        print(f"\n{'-'*75}")
        print("CONCLUSION")
        print(f"{'-'*75}")
        
        if comp['gnn_significantly_better']:
            print(f"  ★ GNN SIGNIFICANTLY OUTPERFORMS IDW (p < 0.05)")
            print(f"    → The extra spatial variation captured by GNN is SIGNAL, not noise")
            print(f"    → GNN disaggregation captures real vulnerability gradients")
        elif gnn_r > idw_r + 0.02:
            print(f"  ◉ GNN performs better than IDW, but not statistically significant")
            print(f"    → Suggestive evidence for GNN advantage, need more data")
        elif abs(gnn_r - idw_r) <= 0.02:
            print(f"  ~ GNN and IDW perform similarly")
            print(f"    → Both methods capture similar spatial patterns")
        else:
            print(f"  ✗ IDW outperforms GNN")
            print(f"    → GNN's extra variation may be noise, not signal")
        
        # Interpretation of absolute performance
        best_r = max(gnn_r, idw_r)
        if best_r > 0.7:
            print(f"\n  ★★ Excellent overall performance (r > 0.7)")
        elif best_r > 0.5:
            print(f"\n  ★ Good overall performance (r > 0.5)")
        elif best_r > 0.3:
            print(f"\n  ◉ Moderate performance (r > 0.3)")
        elif best_r > 0.15:
            print(f"\n  ○ Weak but detectable signal (r > 0.15)")
        else:
            print(f"\n  ✗ No meaningful correlation with ground truth")
        
        print(f"\n{'='*75}")
    
    def create_validation_plots(self, results: Dict) -> plt.Figure:
        """Create comprehensive validation visualization."""
        if not results.get('success'):
            return None
        
        merged = results['validation_data']
        
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
        
        fig.suptitle(f'County-Wide Block Group Validation\n'
                     f'{results["n_block_groups"]} Block Groups | {results["n_addresses"]:,} Addresses | '
                     f'{results["n_tracts_processed"]} Tracts',
                     fontsize=14, fontweight='bold')
        
        colors = {'GNN': '#2E86AB', 'IDW': '#A23B72', 'Kriging': '#F18F01'}
        
        # Row 1: Scatter plots for each method
        for i, method in enumerate(['GNN', 'IDW', 'Kriging']):
            ax = fig.add_subplot(gs[0, i])
            
            pred_col = f'{method}_pred'
            stats_dict = results['methods'][method]
            r = stats_dict['pearson_r']
            p = stats_dict['pearson_p']
            
            ax.scatter(merged['SVI'], merged[pred_col],
                      alpha=0.5, s=20, c=colors[method], edgecolor='white', linewidth=0.3)
            
            # Regression line
            slope, intercept = np.polyfit(merged['SVI'], merged[pred_col], 1)
            x_line = np.array([0, 1])
            ax.plot(x_line, slope * x_line + intercept, 'k--', alpha=0.7, linewidth=2)
            
            # Diagonal
            ax.plot([0, 1], [0, 1], 'gray', linestyle=':', alpha=0.5, linewidth=1)
            
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            ax.set_title(f'{method}\nr = {r:.3f}{sig}', fontsize=12)
            ax.set_xlabel('Ground Truth SVI')
            ax.set_ylabel('Predicted SVI')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
        
        # Row 2, Col 1: Bar chart comparing correlations
        ax4 = fig.add_subplot(gs[1, 0])
        methods = ['GNN', 'IDW', 'Kriging']
        correlations = [results['methods'][m]['pearson_r'] for m in methods]
        bars = ax4.bar(methods, correlations, 
                      color=[colors[m] for m in methods],
                      edgecolor='black', alpha=0.8)
        ax4.axhline(0, color='black', linewidth=0.5)
        ax4.set_ylabel('Pearson Correlation (r)')
        ax4.set_title('Method Comparison: Correlation')
        ax4.set_ylim(min(0, min(correlations) - 0.1), max(correlations) + 0.15)
        
        for bar, corr in zip(bars, correlations):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{corr:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Row 2, Col 2: RMSE comparison
        ax5 = fig.add_subplot(gs[1, 1])
        rmses = [results['methods'][m]['rmse'] for m in methods]
        bars = ax5.bar(methods, rmses,
                      color=[colors[m] for m in methods],
                      edgecolor='black', alpha=0.8)
        ax5.set_ylabel('Root Mean Squared Error')
        ax5.set_title('Method Comparison: RMSE (lower is better)')
        
        for bar, rmse in zip(bars, rmses):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{rmse:.3f}', ha='center', va='bottom', fontsize=11)
        
        # Row 2, Col 3: Residual distribution
        ax6 = fig.add_subplot(gs[1, 2])
        for method in methods:
            residuals = merged['SVI'] - merged[f'{method}_pred']
            ax6.hist(residuals, bins=30, alpha=0.5, label=method, color=colors[method])
        ax6.axvline(0, color='black', linestyle='--', alpha=0.7)
        ax6.set_xlabel('Residual (Truth - Predicted)')
        ax6.set_ylabel('Count')
        ax6.set_title('Residual Distributions')
        ax6.legend()
        
        # Row 3, Col 1: GNN vs IDW scatter
        ax7 = fig.add_subplot(gs[2, 0])
        ax7.scatter(merged['IDW_pred'], merged['GNN_pred'], 
                   alpha=0.5, s=20, c='purple', edgecolor='white', linewidth=0.3)
        ax7.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        r_gnn_idw = np.corrcoef(merged['IDW_pred'], merged['GNN_pred'])[0, 1]
        ax7.set_xlabel('IDW Predicted SVI')
        ax7.set_ylabel('GNN Predicted SVI')
        ax7.set_title(f'GNN vs IDW Agreement\nr = {r_gnn_idw:.3f}')
        ax7.set_xlim(0, 1)
        ax7.set_ylim(0, 1)
        ax7.set_aspect('equal')
        
        # Row 3, Col 2: SVI distribution by method
        ax8 = fig.add_subplot(gs[2, 1])
        data_to_plot = [merged['SVI'], merged['GNN_pred'], merged['IDW_pred'], merged['Kriging_pred']]
        bp = ax8.boxplot(data_to_plot, labels=['Ground\nTruth', 'GNN', 'IDW', 'Kriging'],
                        patch_artist=True)
        bp['boxes'][0].set_facecolor('lightgray')
        bp['boxes'][1].set_facecolor(colors['GNN'])
        bp['boxes'][2].set_facecolor(colors['IDW'])
        bp['boxes'][3].set_facecolor(colors['Kriging'])
        for box in bp['boxes']:
            box.set_alpha(0.7)
        ax8.set_ylabel('SVI Value')
        ax8.set_title('Distribution Comparison')
        
        # Row 3, Col 3: Summary text
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')
        
        comp = results['comparison']
        gnn_r = results['methods']['GNN']['pearson_r']
        idw_r = results['methods']['IDW']['pearson_r']
        
        summary = [
            "SUMMARY STATISTICS",
            "=" * 35,
            f"Block Groups: {results['n_block_groups']}",
            f"Addresses: {results['n_addresses']:,}",
            f"",
            f"Ground Truth SVI:",
            f"  Range: {results['ground_truth_svi_range'][0]:.2f} - {results['ground_truth_svi_range'][1]:.2f}",
            f"  Mean: {results['ground_truth_svi_mean']:.3f}",
            f"",
            "CORRELATION WITH GROUND TRUTH:",
            f"  GNN:     r = {gnn_r:.3f}",
            f"  IDW:     r = {idw_r:.3f}",
            f"  Diff:    {comp['gnn_vs_idw_r_diff']:+.3f}",
            f"",
            "STATISTICAL TEST:",
            f"  z = {comp['gnn_vs_idw_z_stat']:.2f}, p = {comp['gnn_vs_idw_p_value']:.4f}",
            f"",
        ]
        
        if comp['gnn_significantly_better']:
            summary.append("★ GNN SIGNIFICANTLY BETTER")
        elif gnn_r > idw_r:
            summary.append("◉ GNN better (not significant)")
        else:
            summary.append("~ Methods perform similarly")
        
        ax9.text(0.05, 0.95, '\n'.join(summary), transform=ax9.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        # Save
        filepath = os.path.join(self.output_dir, 'county_validation.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"\nValidation plot saved to: {filepath}")
        
        return fig
    
    def save_results(self, results: Dict):
        """Save detailed results to CSV."""
        if not results.get('success'):
            return
        
        # Save block group level results
        merged = results['validation_data']
        bg_file = os.path.join(self.output_dir, 'county_validation_block_groups.csv')
        merged.to_csv(bg_file, index=False)
        print(f"Block group results saved to: {bg_file}")
        
        # Save summary statistics
        summary_rows = []
        for method, stats in results['methods'].items():
            summary_rows.append({
                'method': method,
                'pearson_r': stats['pearson_r'],
                'pearson_p': stats['pearson_p'],
                'spearman_rho': stats['spearman_rho'],
                'r_squared': stats['r_squared'],
                'rmse': stats['rmse'],
                'mae': stats['mae'],
                'predicted_mean': stats['predicted_mean'],
                'predicted_std': stats['predicted_std']
            })
        
        summary_df = pd.DataFrame(summary_rows)
        summary_file = os.path.join(self.output_dir, 'county_validation_summary.csv')
        summary_df.to_csv(summary_file, index=False)
        print(f"Summary statistics saved to: {summary_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='GRANITE County-Wide Validation')
    parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--output', type=str, default='./output', help='Output directory')
    parser.add_argument('--max-tracts', type=int, default=None, 
                        help='Limit number of tracts (for testing)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    validator = CountyWideValidation(
        data_dir=args.data_dir,
        output_dir=args.output,
        verbose=args.verbose
    )
    
    # Run validation
    results = validator.run_county_validation(max_tracts=args.max_tracts)
    
    if results.get('success'):
        # Print report
        validator.print_report(results)
        
        # Create plots
        validator.create_validation_plots(results)
        
        # Save results
        validator.save_results(results)
        
        return 0
    else:
        print(f"\nValidation failed: {results.get('error')}")
        return 1


if __name__ == '__main__':
    sys.exit(main())