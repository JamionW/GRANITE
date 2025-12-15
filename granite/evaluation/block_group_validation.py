"""
Block Group Validation for GRANITE

Validates address-level predictions by aggregating to block groups
and correlating with ACS demographic indicators.
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import stats
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
import os


class BlockGroupValidator:
    """
    Validates disaggregation quality using Census block group demographics.
    
    Core insight: If address-level predictions are meaningful (not arbitrary),
    then aggregating them to block groups should correlate with actual
    block-group-level demographics from ACS.
    """
    
    def __init__(self, block_group_data, verbose: bool = False):
        """
        Args:
            block_group_data: Either:
                - GeoDataFrame with GEOID, geometry, and demographic columns
                - Tuple of (geometries_gdf, demographics_df) from load_block_groups_for_validation
        """
        self.verbose = verbose
        self.results = None
        
        # Handle tuple input from load_block_groups_for_validation
        if isinstance(block_group_data, tuple):
            geometries_gdf, demographics_df = block_group_data
            # Merge geometries with demographics
            self.block_groups = geometries_gdf.merge(demographics_df, on='GEOID', how='left')
            self._log(f"Merged {len(geometries_gdf)} geometries with {len(demographics_df)} demographic records")
        else:
            # Assume it's already a complete GeoDataFrame
            self.block_groups = block_group_data
        
        # Ensure CRS is set
        if self.block_groups.crs is None:
            self.block_groups = self.block_groups.set_crs('EPSG:4326')
        
        self._log(f"Initialized with {len(self.block_groups)} block groups")
    
    def _log(self, msg: str):
        if self.verbose:
            print(f"[BlockGroupValidator] {msg}")
    
    def validate(self,
                 addresses: gpd.GeoDataFrame,
                 predictions: np.ndarray,
                 method_name: str = 'GRANITE') -> Dict:
        """
        Validate predictions against block group demographics.
        
        Args:
            addresses: GeoDataFrame with address points
            predictions: Array of address-level vulnerability predictions
            method_name: Name for this method in results
            
        Returns:
            Dict with validation metrics and analysis
        """
        self._log(f"Validating {method_name} predictions...")
        
        # Step 1: Assign addresses to block groups
        addresses_bg = self._assign_to_block_groups(addresses)
        
        # Step 2: Aggregate predictions to block group level
        bg_predictions = self._aggregate_predictions(addresses_bg, predictions)
        
        # Step 3: Merge with actual demographics
        validation_df = self._merge_with_demographics(bg_predictions)
        
        # Step 4: Compute correlation metrics
        correlations = self._compute_correlations(validation_df)
        
        # Step 5: Compute additional diagnostics
        diagnostics = self._compute_diagnostics(validation_df)
        
        self.results = {
            'method': method_name,
            'n_addresses': len(addresses),
            'n_block_groups': len(validation_df),
            'correlations': correlations,
            'diagnostics': diagnostics,
            'validation_data': validation_df
        }
        
        return self.results
    
    def _assign_to_block_groups(self, addresses: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
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
        
        n_unassigned = joined['block_group_id'].isna().sum()
        if n_unassigned > 0:
            self._log(f"Warning: {n_unassigned} addresses not in any block group")
        
        return joined
    
    def _aggregate_predictions(self, addresses: gpd.GeoDataFrame,
                                predictions: np.ndarray) -> pd.DataFrame:
        """Aggregate address predictions to block group means."""
        
        df = pd.DataFrame({
            'block_group_id': addresses['block_group_id'].values,
            'prediction': predictions
        })
        
        # Remove unassigned addresses
        df = df.dropna(subset=['block_group_id'])
        
        # Aggregate with multiple statistics
        agg = df.groupby('block_group_id').agg(
            predicted_vulnerability=('prediction', 'mean'),
            prediction_std=('prediction', 'std'),
            prediction_range=('prediction', lambda x: x.max() - x.min()),
            n_addresses=('prediction', 'count')
        ).reset_index()
        
        agg = agg.rename(columns={'block_group_id': 'GEOID'})
        
        return agg
    
    def _merge_with_demographics(self, bg_predictions: pd.DataFrame) -> pd.DataFrame:
        """Merge predictions with actual demographics."""
        
        # Identify available demographic columns
        available_cols = ['GEOID']
        demographic_cols = ['poverty_rate', 'no_vehicle_rate', 'no_hs_rate', 'population']
        
        for col in demographic_cols:
            if col in self.block_groups.columns:
                available_cols.append(col)
            else:
                self._log(f"Warning: {col} not found in block group data")
        
        demographics = self.block_groups[available_cols].copy()
        
        merged = bg_predictions.merge(demographics, on='GEOID', how='inner')
        
        # Require minimum addresses per block group for reliability
        min_addresses = 5
        merged = merged[merged['n_addresses'] >= min_addresses]
        
        self._log(f"Merged {len(merged)} block groups with sufficient addresses")
        
        return merged
    
    def _compute_correlations(self, df: pd.DataFrame) -> Dict:
        """Compute correlations between predictions and demographics."""
        
        results = {}
        
        # Primary validation: predicted vulnerability vs poverty rate
        # Rationale: SVI is heavily weighted toward poverty indicators
        valid_mask = df['poverty_rate'].notna() & df['predicted_vulnerability'].notna()
        if valid_mask.sum() >= 10:
            r, p = stats.pearsonr(
                df.loc[valid_mask, 'predicted_vulnerability'],
                df.loc[valid_mask, 'poverty_rate']
            )
            results['poverty_correlation'] = {
                'r': float(r),
                'p_value': float(p),
                'n': int(valid_mask.sum()),
                'significant': p < 0.05
            }
            self._log(f"Poverty correlation: r={r:.3f}, p={p:.4f}")
        
        # Secondary: predicted vulnerability vs no vehicle rate
        valid_mask = df['no_vehicle_rate'].notna() & df['predicted_vulnerability'].notna()
        if valid_mask.sum() >= 10:
            r, p = stats.pearsonr(
                df.loc[valid_mask, 'predicted_vulnerability'],
                df.loc[valid_mask, 'no_vehicle_rate']
            )
            results['no_vehicle_correlation'] = {
                'r': float(r),
                'p_value': float(p),
                'n': int(valid_mask.sum()),
                'significant': p < 0.05
            }
            self._log(f"No-vehicle correlation: r={r:.3f}, p={p:.4f}")
        
        # Tertiary: predicted vulnerability vs education
        valid_mask = df['no_hs_rate'].notna() & df['predicted_vulnerability'].notna()
        if valid_mask.sum() >= 10:
            r, p = stats.pearsonr(
                df.loc[valid_mask, 'predicted_vulnerability'],
                df.loc[valid_mask, 'no_hs_rate']
            )
            results['education_correlation'] = {
                'r': float(r),
                'p_value': float(p),
                'n': int(valid_mask.sum()),
                'significant': p < 0.05
            }
            self._log(f"Education correlation: r={r:.3f}, p={p:.4f}")
        
        # Composite score: weighted average of absolute correlations
        weights = {'poverty_correlation': 0.5, 'no_vehicle_correlation': 0.3, 
                   'education_correlation': 0.2}
        
        composite = 0
        total_weight = 0
        for key, weight in weights.items():
            if key in results:
                composite += abs(results[key]['r']) * weight
                total_weight += weight
        
        results['composite_score'] = composite / total_weight if total_weight > 0 else 0
        
        return results
    
    def _compute_diagnostics(self, df: pd.DataFrame) -> Dict:
        """Compute additional diagnostic metrics."""
        
        return {
            'n_block_groups_validated': len(df),
            'mean_addresses_per_bg': df['n_addresses'].mean(),
            'median_addresses_per_bg': df['n_addresses'].median(),
            'mean_prediction': df['predicted_vulnerability'].mean(),
            'mean_within_bg_std': df['prediction_std'].mean(),
            'prediction_range': df['predicted_vulnerability'].max() - df['predicted_vulnerability'].min()
        }
    
    def create_validation_report(self, output_dir: str, 
                                  comparison_results: Dict[str, Dict] = None):
        """
        Create visualization comparing methods against block group ground truth.
        
        Args:
            output_dir: Directory for output files
            comparison_results: Dict of {method_name: validation_results}
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Use comparison_results if provided, otherwise use self.results
        if comparison_results:
            all_results = comparison_results
        elif self.results is not None:
            all_results = {self.results['method']: self.results}
        else:
            self._log("No results to report")
            return
        
        # Filter to results with validation_data
        valid_results = {k: v for k, v in all_results.items() 
                        if 'validation_data' in v and v['validation_data'] is not None}
        
        if not valid_results:
            self._log("No valid results with validation data to visualize")
            return
        
        # Create multi-panel figure
        n_methods = len(valid_results)
        fig, axes = plt.subplots(2, max(n_methods, 1), figsize=(5*max(n_methods, 1), 10))
        
        if n_methods == 1:
            axes = axes.reshape(2, 1)
        
        fig.suptitle('Block Group Validation: Predicted Vulnerability vs ACS Demographics', 
                     fontsize=14, fontweight='bold')
        
        for col, (method_name, results) in enumerate(valid_results.items()):
            df = results['validation_data']
            corrs = results['correlations']
            
            # Top row: Prediction vs Poverty Rate
            ax1 = axes[0, col]
            valid = df['poverty_rate'].notna()
            ax1.scatter(df.loc[valid, 'poverty_rate'], 
                       df.loc[valid, 'predicted_vulnerability'],
                       alpha=0.6, s=30)
            
            # Add regression line
            if valid.sum() > 2:
                z = np.polyfit(df.loc[valid, 'poverty_rate'], 
                              df.loc[valid, 'predicted_vulnerability'], 1)
                p = np.poly1d(z)
                x_line = np.linspace(df.loc[valid, 'poverty_rate'].min(),
                                     df.loc[valid, 'poverty_rate'].max(), 100)
                ax1.plot(x_line, p(x_line), 'r-', alpha=0.7, linewidth=2)
            
            poverty_r = corrs.get('poverty_correlation', {}).get('r', np.nan)
            ax1.set_xlabel('Block Group Poverty Rate (%)')
            ax1.set_ylabel('Mean Predicted Vulnerability')
            ax1.set_title(f'{method_name}\nr = {poverty_r:.3f}')
            
            # Bottom row: Prediction vs No Vehicle Rate
            ax2 = axes[1, col]
            valid = df['no_vehicle_rate'].notna()
            ax2.scatter(df.loc[valid, 'no_vehicle_rate'],
                       df.loc[valid, 'predicted_vulnerability'],
                       alpha=0.6, s=30, color='green')
            
            if valid.sum() > 2:
                z = np.polyfit(df.loc[valid, 'no_vehicle_rate'],
                              df.loc[valid, 'predicted_vulnerability'], 1)
                p = np.poly1d(z)
                x_line = np.linspace(df.loc[valid, 'no_vehicle_rate'].min(),
                                     df.loc[valid, 'no_vehicle_rate'].max(), 100)
                ax2.plot(x_line, p(x_line), 'r-', alpha=0.7, linewidth=2)
            
            vehicle_r = corrs.get('no_vehicle_correlation', {}).get('r', np.nan)
            ax2.set_xlabel('Block Group No-Vehicle Rate (%)')
            ax2.set_ylabel('Mean Predicted Vulnerability')
            ax2.set_title(f'r = {vehicle_r:.3f}')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'block_group_validation.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        
        # Create summary table
        self._create_summary_table(valid_results, output_dir)
        
        self._log(f"Validation report saved to {output_dir}")
    
    def _create_summary_table(self, all_results: Dict, output_dir: str):
        """Create summary comparison table."""
        
        rows = []
        for method_name, results in all_results.items():
            corrs = results['correlations']
            diag = results['diagnostics']
            
            rows.append({
                'Method': method_name,
                'Poverty r': corrs.get('poverty_correlation', {}).get('r', np.nan),
                'Vehicle r': corrs.get('no_vehicle_correlation', {}).get('r', np.nan),
                'Education r': corrs.get('education_correlation', {}).get('r', np.nan),
                'Composite': corrs.get('composite_score', np.nan),
                'Block Groups': diag.get('n_block_groups_validated', 0),
                'Mean BG Std': diag.get('mean_within_bg_std', np.nan)
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(output_dir, 'block_group_validation_summary.csv'), 
                  index=False)
        
        # Also create text report
        report_lines = [
            "=" * 70,
            "BLOCK GROUP VALIDATION SUMMARY",
            "=" * 70,
            "",
            "Methodology:",
            "  - Address-level predictions aggregated to block group means",
            "  - Correlated with ACS 5-year demographic estimates",
            "  - Higher correlation = more meaningful within-tract variation",
            "",
            "-" * 70,
            f"{'Method':<15} {'Poverty r':>12} {'Vehicle r':>12} {'Educ r':>12} {'Composite':>12}",
            "-" * 70
        ]
        
        for _, row in df.iterrows():
            report_lines.append(
                f"{row['Method']:<15} {row['Poverty r']:>12.3f} {row['Vehicle r']:>12.3f} "
                f"{row['Education r']:>12.3f} {row['Composite']:>12.3f}"
            )
        
        report_lines.extend([
            "-" * 70,
            "",
            "Interpretation:",
            "  r > 0.3: Strong evidence of meaningful disaggregation",
            "  r > 0.15: Moderate evidence",
            "  r < 0.15: Weak evidence (may be no better than random)",
            "",
            "=" * 70
        ])
        
        with open(os.path.join(output_dir, 'block_group_validation_report.txt'), 'w') as f:
            f.write('\n'.join(report_lines))


def run_block_group_validation(
    addresses: gpd.GeoDataFrame,
    predictions: np.ndarray,
    block_groups: gpd.GeoDataFrame,
    method_name: str = 'GRANITE',
    output_dir: str = './output/validation',
    verbose: bool = True
) -> Dict:
    """
    Convenience function to run full block group validation.
    
    Args:
        addresses: GeoDataFrame with address points
        predictions: Address-level vulnerability predictions
        block_groups: GeoDataFrame with block group geometries and demographics
        method_name: Name for this method
        output_dir: Output directory for reports
        verbose: Print progress
        
    Returns:
        Validation results dict
    """
    validator = BlockGroupValidator(block_groups, verbose=verbose)
    results = validator.validate(addresses, predictions, method_name)
    validator.create_validation_report(output_dir)
    
    return results