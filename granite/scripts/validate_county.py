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
    
    def run_single_tract(self, fips: str, training_bg_ids: List[str] = None) -> Optional[Dict]:
        """Run GRANITE pipeline on a single tract with optional BG holdout."""
        from granite.disaggregation.pipeline import GRANITEPipeline
        from granite.models.gnn import set_random_seed
        
        config = {
            'data': {
                'target_fips': fips,
                'state_fips': fips[:2],
                'county_fips': fips[2:5],
            },
            'model': {'hidden_dim': 32, 'k_neighbors': 8, 'dropout': 0.2, 'use_road_network': True},
            'training': {
                'epochs': 100, 
                'learning_rate': 0.001, 
                'constraint_weight': 0.0,  # deprecated
                'bg_weight': 0.2,
                'coherence_weight': 0.0,
                'discrimination_weight': 0.0,
                'smoothness_weight': 2.0,
                'variation_weight': 1.5
            },
            'processing': {'verbose': self.verbose, 'random_seed': 42}
        }
        
        set_random_seed(42)
        
        try:
            pipeline = GRANITEPipeline(config, data_dir=self.data_dir, output_dir=self.output_dir)
            
            # Load data with block groups
            data = pipeline._load_data()
            
            # Run with training BG subset
            results = pipeline._run_single_tract(fips, data, training_bg_ids=training_bg_ids)
            
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
                            max_tracts: int = None,
                            holdout_fraction: float = 0.2,
                            seed: int = 42) -> Dict:
        """
        Run GRANITE on multiple tracts with block group holdout validation.
        
        Args:
            tract_list: Optional list of FIPS codes (default: all usable tracts)
            max_tracts: Optional limit on number of tracts to process
            holdout_fraction: Fraction of block groups to hold out for validation
            seed: Random seed for holdout split
        
        Returns:
            Comprehensive validation results
        """
        start_time = time.time()
        
        # Load block groups
        self.load_block_groups()
        
        # Create holdout split
        training_bg_ids, validation_bg_ids = self.create_holdout_split(
            holdout_fraction=holdout_fraction,
            seed=seed
        )
        
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
        print(f"GRANITE COUNTY-WIDE VALIDATION (Block Group Supervision)")
        print(f"{'='*70}")
        print(f"Tracts to process: {len(tract_list)}")
        print(f"Training block groups: {len(training_bg_ids)}")
        print(f"Validation block groups: {len(validation_bg_ids)}")
        print(f"{'='*70}\n")
        
        # Collect all predictions
        all_bg_data = []
        successful_tracts = 0
        failed_tracts = []
        
        for i, fips in enumerate(tract_list):
            progress = f"[{i+1}/{len(tract_list)}]"
            print(f"{progress} Processing tract {fips}...", end=" ", flush=True)
            
            # Pass training BGs to pipeline
            results = self.run_single_tract(fips, training_bg_ids=training_bg_ids)
            
            if results is None:
                failed_tracts.append(fips)
                print("FAILED")
                continue
            
            successful_tracts += 1
            n_bg = results.get('n_bg_constraints', 0)
            print(f"OK ({n_bg} BG constraints)")
            
            # Collect address-level predictions
            addresses = results['addresses']
            addresses = addresses.copy()
            addresses['GNN_pred'] = results['predictions']
            
            if 'idw' in results.get('baselines', {}):
                idw = results['baselines']['idw']
                if 'predictions' in idw:
                    addresses['IDW_pred'] = idw['predictions']
            
            if 'kriging' in results.get('baselines', {}):
                kriging = results['baselines']['kriging']
                if 'predictions' in kriging:
                    addresses['Kriging_pred'] = kriging['predictions']
            
            addresses['tract_fips'] = fips
            all_bg_data.append(addresses)
        
        if len(all_bg_data) == 0:
            return {'success': False, 'error': 'No tracts processed successfully'}
        
        # Combine all addresses
        all_addresses = pd.concat(all_bg_data, ignore_index=True)
        all_addresses = gpd.GeoDataFrame(all_addresses, crs='EPSG:4326')
        
        print(f"\n{'='*70}")
        print(f"Aggregating to block groups...")
        print(f"{'='*70}")
        
        # Aggregate to validation block groups ONLY
        validation_results = self._aggregate_to_validation_bgs(
            all_addresses, validation_bg_ids
        )
        
        # Also compute training set metrics (to detect overfitting)
        training_results = self._aggregate_to_validation_bgs(
            all_addresses, training_bg_ids
        )
        
        elapsed = time.time() - start_time
        
        return {
            'success': True,
            'validation': validation_results,
            'training': training_results,
            'n_tracts': successful_tracts,
            'n_failed': len(failed_tracts),
            'failed_tracts': failed_tracts,
            'n_training_bgs': len(training_bg_ids),
            'n_validation_bgs': len(validation_bg_ids),
            'elapsed_time': elapsed,
            'holdout_fraction': holdout_fraction
        }


    def _aggregate_to_validation_bgs(self, 
                                    addresses: gpd.GeoDataFrame,
                                    bg_ids: List[str]) -> Dict:
        """
        Aggregate address predictions to specified block groups and compute metrics.
        
        Args:
            addresses: GeoDataFrame with predictions and block_group_id
            bg_ids: List of block group IDs to aggregate to
        
        Returns:
            Dict with aggregated predictions and metrics
        """
        # Filter to specified BGs
        bg_set = set(bg_ids)
        
        if 'block_group_id' not in addresses.columns:
            # Spatial join if needed
            addresses = self.assign_addresses_to_block_groups(addresses)
        
        addr_in_bgs = addresses[addresses['block_group_id'].isin(bg_set)].copy()
        
        if len(addr_in_bgs) == 0:
            return {'success': False, 'error': 'No addresses in specified block groups'}
        
        # Aggregate predictions by block group
        agg_dict = {'GNN_pred': 'mean'}
        if 'IDW_pred' in addr_in_bgs.columns:
            agg_dict['IDW_pred'] = 'mean'
        if 'Kriging_pred' in addr_in_bgs.columns:
            agg_dict['Kriging_pred'] = 'mean'
        agg_dict['geometry'] = 'count'  # count addresses
        
        bg_agg = addr_in_bgs.groupby('block_group_id').agg(agg_dict).reset_index()
        bg_agg = bg_agg.rename(columns={'geometry': 'n_addresses'})
        
        # Merge with ground truth SVI
        bg_agg = bg_agg.merge(
            self.bg_svi[['GEOID', 'SVI']], 
            left_on='block_group_id', 
            right_on='GEOID',
            how='inner'
        )
        
        # Filter to BGs with valid SVI
        bg_agg = bg_agg[bg_agg['SVI'].notna()].copy()
        
        if len(bg_agg) < 5:
            return {'success': False, 'error': f'Only {len(bg_agg)} BGs with valid SVI'}
        
        # Compute metrics for each method
        methods = {}
        
        for method in ['GNN', 'IDW', 'Kriging']:
            pred_col = f'{method}_pred'
            if pred_col not in bg_agg.columns:
                continue
            
            valid = bg_agg[[pred_col, 'SVI']].dropna()
            if len(valid) < 5:
                continue
            
            r, p = stats.pearsonr(valid['SVI'], valid[pred_col])
            rho, _ = stats.spearmanr(valid['SVI'], valid[pred_col])
            rmse = np.sqrt(np.mean((valid['SVI'] - valid[pred_col])**2))
            mae = np.mean(np.abs(valid['SVI'] - valid[pred_col]))
            
            methods[method] = {
                'pearson_r': r,
                'pearson_p': p,
                'spearman_rho': rho,
                'rmse': rmse,
                'mae': mae,
                'n_block_groups': len(valid)
            }
        
        return {
            'success': True,
            'methods': methods,
            'n_block_groups': len(bg_agg),
            'n_addresses': int(bg_agg['n_addresses'].sum()),
            'validation_data': bg_agg
        }
    
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
        """Print comprehensive validation report with train/val split."""
        if not results.get('success'):
            print(f"Validation failed: {results.get('error')}")
            return
        
        print(f"\n{'='*70}")
        print("GRANITE COUNTY-WIDE VALIDATION REPORT")
        print(f"{'='*70}")
        
        print(f"\nData Summary:")
        print(f"  Tracts processed: {results['n_tracts']}")
        print(f"  Training block groups: {results['n_training_bgs']}")
        print(f"  Validation block groups: {results['n_validation_bgs']}")
        print(f"  Holdout fraction: {results['holdout_fraction']:.0%}")
        print(f"  Elapsed time: {results['elapsed_time']:.1f}s")
        
        # Validation set results (PRIMARY)
        print(f"\n{'='*70}")
        print("VALIDATION SET RESULTS (Held-out block groups)")
        print(f"{'='*70}")
        
        val = results['validation']
        if val.get('success'):
            print(f"\n  Block groups: {val['n_block_groups']}")
            print(f"  Addresses: {val['n_addresses']:,}")
            print(f"\n  {'Method':<12} {'Pearson r':<12} {'RMSE':<12} {'MAE':<12}")
            print(f"  {'-'*48}")
            
            for method in ['GNN', 'IDW', 'Kriging']:
                if method in val['methods']:
                    m = val['methods'][method]
                    sig = '***' if m['pearson_p'] < 0.001 else '**' if m['pearson_p'] < 0.01 else '*' if m['pearson_p'] < 0.05 else ''
                    print(f"  {method:<12} {m['pearson_r']:.4f}{sig:<6} {m['rmse']:.4f}       {m['mae']:.4f}")
        else:
            print(f"  Error: {val.get('error')}")
        
        # Training set results (for overfitting check)
        print(f"\n{'='*70}")
        print("TRAINING SET RESULTS (for overfitting check)")
        print(f"{'='*70}")
        
        train = results['training']
        if train.get('success'):
            print(f"\n  Block groups: {train['n_block_groups']}")
            print(f"\n  {'Method':<12} {'Pearson r':<12} {'RMSE':<12}")
            print(f"  {'-'*36}")
            
            for method in ['GNN', 'IDW', 'Kriging']:
                if method in train['methods']:
                    m = train['methods'][method]
                    print(f"  {method:<12} {m['pearson_r']:.4f}       {m['rmse']:.4f}")
        
        # Compare GNN vs IDW on validation set
        if val.get('success') and 'GNN' in val['methods'] and 'IDW' in val['methods']:
            gnn_r = val['methods']['GNN']['pearson_r']
            idw_r = val['methods']['IDW']['pearson_r']
            diff = gnn_r - idw_r
            
            print(f"\n{'='*70}")
            print("STATISTICAL COMPARISON (Validation Set)")
            print(f"{'='*70}")
            print(f"\n  GNN r = {gnn_r:.4f}")
            print(f"  IDW r = {idw_r:.4f}")
            print(f"  Difference = {diff:+.4f}")
            
            # Fisher z-test for correlation comparison
            n = val['n_block_groups']
            z_gnn = 0.5 * np.log((1 + gnn_r) / (1 - gnn_r))
            z_idw = 0.5 * np.log((1 + idw_r) / (1 - idw_r))
            se = np.sqrt(2 / (n - 3))
            z_stat = (z_gnn - z_idw) / se
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            
            print(f"\n  Fisher z-test: z = {z_stat:.3f}, p = {p_value:.4f}")
            
            if p_value < 0.05 and gnn_r > idw_r:
                print(f"\n  ★ GNN SIGNIFICANTLY BETTER THAN IDW (p < 0.05)")
            elif p_value < 0.05 and idw_r > gnn_r:
                print(f"\n  ★ IDW SIGNIFICANTLY BETTER THAN GNN (p < 0.05)")
            else:
                print(f"\n  ~ No significant difference between methods")
    
    def create_validation_plots(self, results: Dict) -> plt.Figure:
        """Create comprehensive validation visualization."""
        if not results.get('success'):
            return None

        # Adapt to new nested structure
        if 'validation' in results:
            val = results['validation']
            merged = val.get('validation_data')
            
            # Compute comparison stats on the fly
            gnn_r = val['methods']['GNN']['pearson_r']
            idw_r = val['methods']['IDW']['pearson_r']
            n = val['n_block_groups']
            
            # Fisher z-test
            z_gnn = 0.5 * np.log((1 + gnn_r) / (1 - gnn_r))
            z_idw = 0.5 * np.log((1 + idw_r) / (1 - idw_r))
            se = np.sqrt(2 / (n - 3))
            z_stat = (z_gnn - z_idw) / se
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            
            results = {
                **results,
                'n_block_groups': val.get('n_block_groups'),
                'n_addresses': val.get('n_addresses'),
                'n_tracts_processed': results.get('n_tracts'),
                'methods': val.get('methods', {}),
                'validation_data': merged,
                'ground_truth_svi_range': [merged['SVI'].min(), merged['SVI'].max()],
                'ground_truth_svi_mean': merged['SVI'].mean(),
                'comparison': {
                    'gnn_vs_idw_r_diff': gnn_r - idw_r,
                    'gnn_vs_idw_z_stat': z_stat,
                    'gnn_vs_idw_p_value': p_value,
                    'gnn_significantly_better': p_value < 0.05 and gnn_r > idw_r
                }
            }
        
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

    def create_holdout_split(self, 
                            holdout_fraction: float = 0.2,
                            seed: int = 42) -> Tuple[List[str], List[str]]:
        """
        Split block groups into training and validation sets.
        
        Stratifies by tract to ensure each tract has both training and
        validation block groups where possible.
        
        Args:
            holdout_fraction: Fraction of BGs to hold out for validation
            seed: Random seed for reproducibility
        
        Returns:
            Tuple of (training_bg_ids, validation_bg_ids)
        """
        np.random.seed(seed)
        
        if self.block_groups is None or self.bg_svi is None:
            raise ValueError("Must load block groups before creating split")
        
        # Get BGs with valid SVI
        valid_bg = self.bg_svi[self.bg_svi['SVI'].notna()]['GEOID'].tolist()
        
        # Group by tract
        bg_to_tract = dict(zip(
            self.block_groups['GEOID'],
            self.block_groups['tract_fips']
        ))
        
        tract_to_bgs = {}
        for bg_id in valid_bg:
            if bg_id in bg_to_tract:
                tract = bg_to_tract[bg_id]
                if tract not in tract_to_bgs:
                    tract_to_bgs[tract] = []
                tract_to_bgs[tract].append(bg_id)
        
        # Stratified split within each tract
        training_bgs = []
        validation_bgs = []
        
        for tract, bgs in tract_to_bgs.items():
            n_bgs = len(bgs)
            if n_bgs == 1:
                # Single BG tract: assign to training
                training_bgs.extend(bgs)
            elif n_bgs == 2:
                # Two BG tract: one train, one validate
                np.random.shuffle(bgs)
                training_bgs.append(bgs[0])
                validation_bgs.append(bgs[1])
            else:
                # 3+ BGs: proportional split
                np.random.shuffle(bgs)
                n_holdout = max(1, int(n_bgs * holdout_fraction))
                validation_bgs.extend(bgs[:n_holdout])
                training_bgs.extend(bgs[n_holdout:])
        
        self._log(f"Holdout split: {len(training_bgs)} training, {len(validation_bgs)} validation BGs")
        
        return training_bgs, validation_bgs
    
    def save_results(self, results: Dict):
        """Save detailed results to CSV."""
        if not results.get('success'):
            return
        
        # Handle nested structure
        if 'validation' in results:
            val = results['validation']
            merged = val.get('validation_data')
            methods = val.get('methods', {})
        else:
            merged = results.get('validation_data')
            methods = results.get('methods', {})
        
        if merged is None:
            print("No validation data to save")
            return
        bg_file = os.path.join(self.output_dir, 'county_validation_block_groups.csv')
        merged.to_csv(bg_file, index=False)
        print(f"Block group results saved to: {bg_file}")
        
        # Save summary statistics
        summary_rows = []
        for method, stats in methods.items():
            summary_rows.append({
                'method': method,
                'pearson_r': stats.get('pearson_r'),
                'pearson_p': stats.get('pearson_p'),
                'spearman_rho': stats.get('spearman_rho'),
                'r_squared': stats.get('pearson_r', 0)**2,  # compute from r
                'rmse': stats.get('rmse'),
                'mae': stats.get('mae'),
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