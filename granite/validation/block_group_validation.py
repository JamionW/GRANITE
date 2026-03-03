"""
Block Group Validation for GRANITE

Validates address-level predictions by aggregating to block groups
and correlating with computed block-group SVI.

Key insight: We compute SVI at block group level using CDC methodology
(with available variables), then test if predicted SVI correlates with
this ground truth. This is the proper validation for spatial disaggregation.
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import stats
from typing import Dict, Tuple, Optional, List
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import os


class BlockGroupValidator:
    """
    Validates disaggregation quality using Census block group SVI.
    
    Core insight: If address-level predictions capture meaningful spatial
    variation in vulnerability, then aggregating them to block groups should
    correlate with actual block-group-level SVI computed from ACS data.
    """
    
    def __init__(self, block_group_data, verbose: bool = False):
        """
        Args:
            block_group_data: GeoDataFrame with GEOID, geometry, SVI, and component columns
        """
        self.verbose = verbose
        self.results = {}
        
        if isinstance(block_group_data, tuple):
            geometries_gdf, demographics_df = block_group_data
            self.block_groups = geometries_gdf.merge(demographics_df, on='GEOID', how='left')
        else:
            self.block_groups = block_group_data
        
        if self.block_groups.crs is None:
            self.block_groups = self.block_groups.set_crs('EPSG:4326')
        
        self._log(f"Initialized with {len(self.block_groups)} block groups")
        
        # Check for SVI column
        if 'SVI' not in self.block_groups.columns:
            self._log("Warning: SVI column not found - validation will be limited")
    
    def _log(self, msg: str):
        if self.verbose:
            print(f"[BlockGroupValidator] {msg}")
    
    def validate(self,
                 addresses: gpd.GeoDataFrame,
                 predictions: np.ndarray,
                 method_name: str = 'GRANITE') -> Dict:
        """
        Validate predictions against block group SVI.
        
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
        
        # Step 3: Merge with block group SVI
        validation_df = self._merge_with_svi(bg_predictions)
        
        # Step 4: Compute correlation metrics
        correlations = self._compute_correlations(validation_df)
        
        # Step 5: Compute diagnostics
        diagnostics = self._compute_diagnostics(validation_df)
        
        result = {
            'method': method_name,
            'n_addresses': len(addresses),
            'n_block_groups': len(validation_df),
            'correlations': correlations,
            'diagnostics': diagnostics,
            'validation_data': validation_df
        }
        
        self.results[method_name] = result
        
        return result
    
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
        
        df = df.dropna(subset=['block_group_id'])
        
        agg = df.groupby('block_group_id').agg(
            predicted_svi=('prediction', 'mean'),
            prediction_std=('prediction', 'std'),
            prediction_min=('prediction', 'min'),
            prediction_max=('prediction', 'max'),
            prediction_range=('prediction', lambda x: x.max() - x.min()),
            n_addresses=('prediction', 'count')
        ).reset_index()
        
        agg = agg.rename(columns={'block_group_id': 'GEOID'})
        
        return agg
    
    def _merge_with_svi(self, bg_predictions: pd.DataFrame) -> pd.DataFrame:
        """Merge predictions with block group SVI and components."""
        
        # Columns to include from block groups
        svi_cols = ['GEOID', 'SVI', 'svi_complete', 'population',
                    'SVI_theme1', 'SVI_theme2', 'SVI_theme3', 'SVI_theme4',
                    'EP_MHI', 'EP_PCI', 'EP_UNEMP', 'EP_NOHSDP', 'EP_NOVEH',
                    'EP_AGE65', 'EP_AGE17', 'EP_MINRTY']
        
        available_cols = [c for c in svi_cols if c in self.block_groups.columns]
        demographics = self.block_groups[available_cols].copy()
        
        merged = bg_predictions.merge(demographics, on='GEOID', how='inner')
        
        # Require minimum addresses and complete SVI
        min_addresses = 5
        merged = merged[merged['n_addresses'] >= min_addresses]
        
        if 'svi_complete' in merged.columns:
            complete_count = merged['svi_complete'].sum()
            self._log(f"Block groups with complete SVI: {complete_count}/{len(merged)}")
        
        self._log(f"Merged {len(merged)} block groups with sufficient addresses")
        
        return merged
    
    def _compute_correlations(self, df: pd.DataFrame) -> Dict:
        """Compute correlations between predictions and block group SVI."""
        
        results = {}
        
        # Primary validation: predicted SVI vs computed block-group SVI
        if 'SVI' in df.columns:
            valid_mask = df['SVI'].notna() & df['predicted_svi'].notna()
            if valid_mask.sum() >= 10:
                r, p = stats.pearsonr(
                    df.loc[valid_mask, 'predicted_svi'],
                    df.loc[valid_mask, 'SVI']
                )
                rho, p_spearman = stats.spearmanr(
                    df.loc[valid_mask, 'predicted_svi'],
                    df.loc[valid_mask, 'SVI']
                )
                results['svi_correlation'] = {
                    'pearson_r': float(r),
                    'spearman_rho': float(rho),
                    'p_value': float(p),
                    'n': int(valid_mask.sum()),
                    'significant': p < 0.05
                }
                self._log(f"SVI correlation: r={r:.3f}, rho={rho:.3f}, p={p:.4f}")
        
        # Theme-level correlations
        for theme_num in [1, 2, 3, 4]:
            theme_col = f'SVI_theme{theme_num}'
            if theme_col in df.columns:
                valid_mask = df[theme_col].notna() & df['predicted_svi'].notna()
                if valid_mask.sum() >= 10:
                    r, p = stats.pearsonr(
                        df.loc[valid_mask, 'predicted_svi'],
                        df.loc[valid_mask, theme_col]
                    )
                    results[f'theme{theme_num}_correlation'] = {
                        'r': float(r),
                        'p_value': float(p),
                        'n': int(valid_mask.sum())
                    }
        
        # Component correlations (for diagnostic insight)
        component_cols = ['EP_MHI', 'EP_NOVEH', 'EP_UNEMP', 'EP_NOHSDP', 'EP_MINRTY']
        for col in component_cols:
            if col in df.columns:
                valid_mask = df[col].notna() & df['predicted_svi'].notna()
                if valid_mask.sum() >= 10:
                    # For income, invert direction expectation
                    pred = df.loc[valid_mask, 'predicted_svi']
                    actual = df.loc[valid_mask, col]
                    r, p = stats.pearsonr(pred, actual)
                    results[f'{col}_correlation'] = {
                        'r': float(r),
                        'p_value': float(p),
                        'n': int(valid_mask.sum())
                    }
        
        # Composite score (weighted average of absolute correlations)
        if 'svi_correlation' in results:
            results['composite_score'] = abs(results['svi_correlation']['pearson_r'])
        else:
            results['composite_score'] = 0.0
        
        return results
    
    def _compute_diagnostics(self, df: pd.DataFrame) -> Dict:
        """Compute diagnostic metrics."""
        
        diagnostics = {
            'n_block_groups_validated': len(df),
            'mean_addresses_per_bg': df['n_addresses'].mean(),
            'median_addresses_per_bg': df['n_addresses'].median(),
            'total_addresses': df['n_addresses'].sum(),
            'mean_predicted_svi': df['predicted_svi'].mean(),
            'std_predicted_svi': df['predicted_svi'].std(),
            'mean_within_bg_std': df['prediction_std'].mean(),
            'mean_within_bg_range': df['prediction_range'].mean(),
        }
        
        if 'SVI' in df.columns:
            diagnostics['mean_actual_svi'] = df['SVI'].mean()
            diagnostics['std_actual_svi'] = df['SVI'].std()
        
        return diagnostics
    
    def compare_methods(self, 
                        addresses: gpd.GeoDataFrame,
                        method_predictions: Dict[str, np.ndarray]) -> Dict[str, Dict]:
        """
        Compare multiple disaggregation methods.
        
        Args:
            addresses: GeoDataFrame with address points
            method_predictions: Dict of {method_name: predictions_array}
            
        Returns:
            Dict of validation results for each method
        """
        results = {}
        for method_name, predictions in method_predictions.items():
            results[method_name] = self.validate(addresses, predictions, method_name)
        
        return results
    
    def create_validation_report(self, output_dir: str):
        """Create validation report with visualizations."""
        
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.results:
            self._log("No results to report")
            return
        
        # Filter to results with validation data
        valid_results = {k: v for k, v in self.results.items() 
                        if 'validation_data' in v and v['validation_data'] is not None
                        and len(v['validation_data']) > 0}
        
        if not valid_results:
            self._log("No valid results with validation data")
            return
        
        # Create visualizations
        self._create_svi_scatter_plot(valid_results, output_dir)
        self._create_method_comparison_bar(valid_results, output_dir)
        self._create_theme_correlation_heatmap(valid_results, output_dir)
        self._create_spatial_residual_plot(valid_results, output_dir)
        self._create_summary_table(valid_results, output_dir)
        self._create_text_report(valid_results, output_dir)
        
        self._log(f"Validation report saved to {output_dir}")
    
    def _create_svi_scatter_plot(self, results: Dict, output_dir: str):
        """Scatter plot of predicted vs actual SVI for each method."""
        
        n_methods = len(results)
        fig, axes = plt.subplots(1, max(n_methods, 1), 
                                  figsize=(6*max(n_methods, 1), 5))
        
        if n_methods == 1:
            axes = [axes]
        
        fig.suptitle('Predicted vs Block Group SVI', fontsize=14, fontweight='bold')
        
        colors = plt.cm.Set2(np.linspace(0, 1, n_methods))
        
        for idx, (method_name, result) in enumerate(results.items()):
            ax = axes[idx]
            df = result['validation_data']
            corrs = result['correlations']
            
            if 'SVI' not in df.columns:
                ax.text(0.5, 0.5, 'SVI not available', ha='center', va='center')
                continue
            
            valid = df['SVI'].notna() & df['predicted_svi'].notna()
            x = df.loc[valid, 'SVI']
            y = df.loc[valid, 'predicted_svi']
            
            # Scatter with size by address count
            sizes = np.clip(df.loc[valid, 'n_addresses'], 10, 100)
            scatter = ax.scatter(x, y, c=colors[idx], alpha=0.6, s=sizes, 
                                edgecolors='white', linewidth=0.5)
            
            # Regression line
            if len(x) > 2:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                x_line = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_line, p(x_line), 'k--', alpha=0.7, linewidth=2, 
                       label=f'Linear fit')
            
            # Perfect agreement line
            lims = [min(x.min(), y.min()), max(x.max(), y.max())]
            ax.plot(lims, lims, 'gray', linestyle=':', alpha=0.5, label='Perfect agreement')
            
            # Correlation annotation
            svi_corr = corrs.get('svi_correlation', {})
            r = svi_corr.get('pearson_r', np.nan)
            rho = svi_corr.get('spearman_rho', np.nan)
            n = svi_corr.get('n', 0)
            
            ax.annotate(f'r = {r:.3f}\nρ = {rho:.3f}\nn = {n}',
                       xy=(0.05, 0.95), xycoords='axes fraction',
                       fontsize=11, fontweight='bold',
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            ax.set_xlabel('Block Group SVI (Ground Truth)', fontsize=11)
            ax.set_ylabel('Mean Predicted SVI', fontsize=11)
            ax.set_title(method_name, fontsize=12, fontweight='bold')
            ax.legend(loc='lower right', fontsize=9)
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'svi_validation_scatter.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_method_comparison_bar(self, results: Dict, output_dir: str):
        """Bar chart comparing method performance."""
        
        methods = list(results.keys())
        metrics = []
        
        for method in methods:
            corrs = results[method]['correlations']
            svi_r = corrs.get('svi_correlation', {}).get('pearson_r', 0)
            metrics.append(svi_r)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        colors = ['#2ecc71' if m > 0.3 else '#f1c40f' if m > 0.15 else '#e74c3c' 
                  for m in metrics]
        
        bars = ax.bar(methods, metrics, color=colors, edgecolor='black', linewidth=1.2)
        
        # Add value labels
        for bar, val in zip(bars, metrics):
            height = bar.get_height()
            ax.annotate(f'{val:.3f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords='offset points',
                       ha='center', fontsize=11, fontweight='bold')
        
        # Add threshold lines
        ax.axhline(y=0.3, color='green', linestyle='--', alpha=0.7, label='Strong (r>0.3)')
        ax.axhline(y=0.15, color='orange', linestyle='--', alpha=0.7, label='Moderate (r>0.15)')
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        
        ax.set_ylabel('Correlation with Block Group SVI (r)', fontsize=11)
        ax.set_xlabel('Method', fontsize=11)
        ax.set_title('Disaggregation Quality: Correlation with Ground Truth SVI', 
                    fontsize=13, fontweight='bold')
        ax.legend(loc='upper right')
        ax.set_ylim(-0.5, 1.0)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'method_comparison_bar.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_theme_correlation_heatmap(self, results: Dict, output_dir: str):
        """Heatmap of correlations by SVI theme for each method."""
        
        methods = list(results.keys())
        themes = ['Overall SVI', 'Theme 1\n(Socioeconomic)', 'Theme 2\n(Household)', 
                  'Theme 3\n(Minority)', 'Theme 4\n(Housing)']
        theme_keys = ['svi_correlation', 'theme1_correlation', 'theme2_correlation',
                      'theme3_correlation', 'theme4_correlation']
        
        # Build correlation matrix
        data = []
        for method in methods:
            corrs = results[method]['correlations']
            row = []
            for key in theme_keys:
                if key == 'svi_correlation':
                    r = corrs.get(key, {}).get('pearson_r', np.nan)
                else:
                    r = corrs.get(key, {}).get('r', np.nan)
                row.append(r)
            data.append(row)
        
        data = np.array(data)
        
        fig, ax = plt.subplots(figsize=(10, 4 + 0.4*len(methods)))
        
        # Custom colormap: red (negative) -> white (zero) -> green (positive)
        cmap = mcolors.LinearSegmentedColormap.from_list(
            'rg', ['#e74c3c', 'white', '#2ecc71'])
        
        im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=-0.5, vmax=0.5)
        
        # Labels
        ax.set_xticks(np.arange(len(themes)))
        ax.set_yticks(np.arange(len(methods)))
        ax.set_xticklabels(themes, fontsize=10)
        ax.set_yticklabels(methods, fontsize=11)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=0, ha='center')
        
        # Add correlation values as text
        for i in range(len(methods)):
            for j in range(len(themes)):
                val = data[i, j]
                if not np.isnan(val):
                    color = 'black' if abs(val) < 0.3 else 'white'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                           color=color, fontsize=11, fontweight='bold')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Pearson Correlation (r)', fontsize=10)
        
        ax.set_title('Correlation with Block Group SVI by Theme', 
                    fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'theme_correlation_heatmap.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_spatial_residual_plot(self, results: Dict, output_dir: str):
        """Plot prediction residuals to identify systematic spatial bias."""
        
        # Use first method with geometry
        for method_name, result in results.items():
            df = result['validation_data']
            
            if 'SVI' not in df.columns:
                continue
            
            # Compute residuals
            valid = df['SVI'].notna() & df['predicted_svi'].notna()
            df_valid = df[valid].copy()
            df_valid['residual'] = df_valid['predicted_svi'] - df_valid['SVI']
            
            # Merge with geometry
            df_geo = self.block_groups[['GEOID', 'geometry']].merge(
                df_valid[['GEOID', 'residual', 'SVI', 'predicted_svi']], 
                on='GEOID', how='inner'
            )
            
            if len(df_geo) < 5:
                continue
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Left: Actual SVI
            gdf = gpd.GeoDataFrame(df_geo, geometry='geometry')
            gdf.plot(column='SVI', cmap='RdYlGn_r', legend=True, ax=axes[0],
                    legend_kwds={'label': 'SVI', 'shrink': 0.7})
            axes[0].set_title('Block Group SVI\n(Ground Truth)', fontsize=11, fontweight='bold')
            axes[0].axis('off')
            
            # Middle: Predicted SVI
            gdf.plot(column='predicted_svi', cmap='RdYlGn_r', legend=True, ax=axes[1],
                    legend_kwds={'label': 'Predicted SVI', 'shrink': 0.7})
            axes[1].set_title(f'{method_name}\nPredicted SVI', fontsize=11, fontweight='bold')
            axes[1].axis('off')
            
            # Right: Residuals
            vmax = max(abs(df_geo['residual'].min()), abs(df_geo['residual'].max()))
            gdf.plot(column='residual', cmap='RdBu_r', legend=True, ax=axes[2],
                    vmin=-vmax, vmax=vmax,
                    legend_kwds={'label': 'Residual (Pred - Actual)', 'shrink': 0.7})
            axes[2].set_title('Prediction Residuals\n(Red=Over, Blue=Under)', 
                             fontsize=11, fontweight='bold')
            axes[2].axis('off')
            
            plt.suptitle(f'Spatial Validation: {method_name}', fontsize=13, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'spatial_residuals_{method_name}.png'),
                        dpi=150, bbox_inches='tight')
            plt.close()
            
            break  # Only do first method for now
    
    def _create_summary_table(self, results: Dict, output_dir: str):
        """Create CSV summary of all methods."""
        
        rows = []
        for method_name, result in results.items():
            corrs = result['correlations']
            diag = result['diagnostics']
            
            svi_corr = corrs.get('svi_correlation', {})
            
            rows.append({
                'Method': method_name,
                'SVI_r': svi_corr.get('pearson_r', np.nan),
                'SVI_rho': svi_corr.get('spearman_rho', np.nan),
                'SVI_p': svi_corr.get('p_value', np.nan),
                'Theme1_r': corrs.get('theme1_correlation', {}).get('r', np.nan),
                'Theme2_r': corrs.get('theme2_correlation', {}).get('r', np.nan),
                'Theme3_r': corrs.get('theme3_correlation', {}).get('r', np.nan),
                'Theme4_r': corrs.get('theme4_correlation', {}).get('r', np.nan),
                'EP_MHI_r': corrs.get('EP_MHI_correlation', {}).get('r', np.nan),
                'EP_NOVEH_r': corrs.get('EP_NOVEH_correlation', {}).get('r', np.nan),
                'N_BlockGroups': diag.get('n_block_groups_validated', 0),
                'N_Addresses': diag.get('total_addresses', 0),
                'Mean_BG_Std': diag.get('mean_within_bg_std', np.nan),
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(output_dir, 'block_group_validation_summary.csv'), 
                  index=False)
    
    def _create_text_report(self, results: Dict, output_dir: str):
        """Create human-readable text report."""
        
        lines = [
            "=" * 75,
            "BLOCK GROUP VALIDATION REPORT",
            "=" * 75,
            "",
            "METHODOLOGY:",
            "  1. Compute block-group SVI using CDC methodology with available ACS variables",
            "  2. Aggregate address-level predictions to block group means",
            "  3. Correlate predicted means with computed block-group SVI",
            "  4. Positive correlation = meaningful spatial disaggregation",
            "",
            "GROUND TRUTH: Block-group SVI computed from:",
            "  - Theme 1 (Socioeconomic): Median HH income, per capita income, unemployment, education",
            "  - Theme 2 (Household): Age 65+, Age 17-, single parent households",
            "  - Theme 3 (Minority): Minority status",
            "  - Theme 4 (Housing): Multi-unit, mobile homes, crowding, no vehicle",
            "",
            "-" * 75,
            "RESULTS SUMMARY",
            "-" * 75,
            "",
            f"{'Method':<15} {'SVI r':>10} {'SVI ρ':>10} {'p-value':>12} {'N BGs':>8}",
            "-" * 55
        ]
        
        for method_name, result in results.items():
            corrs = result['correlations']
            diag = result['diagnostics']
            
            svi_corr = corrs.get('svi_correlation', {})
            r = svi_corr.get('pearson_r', np.nan)
            rho = svi_corr.get('spearman_rho', np.nan)
            p = svi_corr.get('p_value', np.nan)
            n = diag.get('n_block_groups_validated', 0)
            
            lines.append(f"{method_name:<15} {r:>10.3f} {rho:>10.3f} {p:>12.4f} {n:>8}")
        
        lines.extend([
            "",
            "-" * 75,
            "INTERPRETATION GUIDE",
            "-" * 75,
            "",
            "Correlation Strength:",
            "  r > 0.5   : Strong positive - excellent spatial learning",
            "  r > 0.3   : Moderate positive - meaningful patterns captured",
            "  r > 0.15  : Weak positive - some signal detected",
            "  |r| < 0.15: No meaningful correlation",
            "  r < -0.15 : Negative correlation - systematic bias",
            "",
            "Statistical Significance:",
            "  p < 0.05  : Correlation is statistically significant",
            "  p < 0.01  : Highly significant",
            "",
            "Key Insight:",
            "  A positive correlation indicates that addresses predicted to have",
            "  higher vulnerability tend to be in block groups that actually have",
            "  higher SVI according to ACS demographics. This validates that the",
            "  learned spatial disaggregation captures real vulnerability gradients.",
            "",
            "=" * 75
        ])
        
        with open(os.path.join(output_dir, 'block_group_validation_report.txt'), 'w') as f:
            f.write('\n'.join(lines))


def run_block_group_validation(
    addresses: gpd.GeoDataFrame,
    predictions: np.ndarray,
    block_groups: gpd.GeoDataFrame,
    method_name: str = 'GRANITE',
    output_dir: str = './output/validation',
    verbose: bool = True
) -> Dict:
    """
    Convenience function for single-method validation.
    """
    validator = BlockGroupValidator(block_groups, verbose=verbose)
    results = validator.validate(addresses, predictions, method_name)
    validator.create_validation_report(output_dir)
    return results


def run_comparative_validation(
    addresses: gpd.GeoDataFrame,
    method_predictions: Dict[str, np.ndarray],
    block_groups: gpd.GeoDataFrame,
    output_dir: str = './output/validation',
    verbose: bool = True
) -> Dict[str, Dict]:
    """
    Convenience function for multi-method comparison.
    
    Args:
        addresses: GeoDataFrame with address points
        method_predictions: Dict of {method_name: predictions_array}
        block_groups: GeoDataFrame with block group geometries and SVI
        output_dir: Output directory
        verbose: Print progress
        
    Returns:
        Dict of validation results per method
    """
    validator = BlockGroupValidator(block_groups, verbose=verbose)
    results = validator.compare_methods(addresses, method_predictions)
    validator.create_validation_report(output_dir)
    return results