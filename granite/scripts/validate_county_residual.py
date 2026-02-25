"""
GRANITE County-Wide Residual Learning Validation

Tests whether GNN can learn systematic corrections to IDW baseline
across the entire county with proper holdout validation.

Key question: Does IDW + GNN_residual outperform IDW alone?

Usage:
    python -m granite.scripts.validate_county_residual -v
    python -m granite.scripts.validate_county_residual --max-tracts 10 -v
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

sys.path.insert(0, '/workspaces/GRANITE')

from granite.models.gnn import set_random_seed, SpatialDisaggregationGNN
from granite.data.loaders import DataLoader
from granite.features.spatial_features import SpatialFeatureComputer, normalize_spatial_features
from granite.evaluation.baselines import IDWDisaggregation, OrdinaryKrigingDisaggregation

import torch
import torch.nn.functional as F


class ResidualGNNTrainer:
    """Trainer for residual learning - predicts deviation from IDW baseline."""
    
    def __init__(self, model, learning_rate=0.001, seed=42,
                 bg_weight=1.0, smoothness_weight=1.0):
        
        self.model = model
        self.seed = seed
        self.bg_weight = bg_weight
        self.smoothness_weight = smoothness_weight
        
        set_random_seed(seed)
        
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.8, min_lr=1e-5
        )
    
    def train_residual(self, graph_data, bg_masks, bg_residuals,
                       epochs=100, verbose=False):
        """Train GNN to predict residuals from IDW baseline."""
        set_random_seed(self.seed)
        self.model.train()
        
        # Convert to tensors
        bg_masks_tensor = {}
        bg_targets_tensor = {}
        for bg_id, mask in bg_masks.items():
            if bg_id in bg_residuals:
                bg_masks_tensor[bg_id] = torch.BoolTensor(mask)
                bg_targets_tensor[bg_id] = torch.FloatTensor([bg_residuals[bg_id]])
        
        if len(bg_masks_tensor) == 0:
            # No training signal
            self.model.eval()
            with torch.no_grad():
                preds = self.model(graph_data.x, graph_data.edge_index)
            return {'raw_predictions': np.zeros(len(preds)), 'epochs_trained': 0}
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            predicted_residuals = self.model(graph_data.x, graph_data.edge_index)
            
            # BG residual matching loss
            bg_losses = []
            for bg_id, mask in bg_masks_tensor.items():
                if mask.sum() < 3:
                    continue
                bg_pred_residual = predicted_residuals[mask].mean()
                bg_target_residual = bg_targets_tensor[bg_id]
                bg_losses.append(F.mse_loss(bg_pred_residual.unsqueeze(0), bg_target_residual))
            
            if bg_losses:
                bg_loss = torch.stack(bg_losses).mean()
            else:
                bg_loss = torch.tensor(0.0, requires_grad=True)
            
            # Smoothness loss
            src, dst = graph_data.edge_index[0], graph_data.edge_index[1]
            residual_diff = predicted_residuals[src] - predicted_residuals[dst]
            smoothness_loss = torch.mean(residual_diff ** 2)
            
            # Magnitude regularization
            magnitude_loss = torch.mean(predicted_residuals ** 2)
            
            total_loss = (
                self.bg_weight * bg_loss +
                self.smoothness_weight * smoothness_loss +
                0.1 * magnitude_loss
            )
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step(total_loss)
            
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= 15:
                break
        
        self.model.eval()
        with torch.no_grad():
            final_residuals = self.model(graph_data.x, graph_data.edge_index)
        
        return {
            'raw_predictions': final_residuals.numpy(),
            'epochs_trained': epoch + 1,
        }


class CountyResidualValidation:
    """
    County-wide validation of residual learning approach.
    
    Compares:
    - IDW alone
    - IDW + GNN residual
    - Kriging (for reference)
    """
    
    def __init__(self, data_dir: str = './data', output_dir: str = './output',
                 verbose: bool = False):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.verbose = verbose
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.block_groups = None
        self.bg_svi = None
        self.tracts = None
        self.loader = None
    
    def _log(self, msg: str, level: str = 'INFO'):
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {level}: {msg}")
    
    def load_data(self, state_fips: str = '47', county_fips: str = '065'):
        """Load all required data."""
        self._log("Loading county data...")
        
        self.loader = DataLoader(self.data_dir)
        
        # Load tracts
        tracts = self.loader.load_census_tracts(state_fips, county_fips)
        county_name = self.loader._get_county_name(state_fips, county_fips)
        svi = self.loader.load_svi_data(state_fips, county_name)
        self.tracts = tracts.merge(svi, on='FIPS', how='inner')
        
        # Load block groups
        bg_file = os.path.join(self.data_dir, 'raw', f'tl_2020_{state_fips}_bg.shp')
        if not os.path.exists(bg_file):
            raise FileNotFoundError(f"Block group shapefile not found: {bg_file}")
        
        bg_gdf = gpd.read_file(bg_file)
        county_bg = bg_gdf[bg_gdf['COUNTYFP'] == county_fips].copy()
        county_bg['GEOID'] = county_bg['GEOID'].astype(str)
        county_bg['tract_fips'] = state_fips + county_fips + county_bg['TRACTCE'].astype(str)
        
        if county_bg.crs is None:
            county_bg.set_crs(epsg=4326, inplace=True)
        elif county_bg.crs != 'EPSG:4326':
            county_bg = county_bg.to_crs(epsg=4326)
        
        self.block_groups = county_bg
        
        # Load BG SVI
        svi_file = os.path.join(self.data_dir, 'processed', 'acs_block_groups_svi.csv')
        if not os.path.exists(svi_file):
            raise FileNotFoundError(f"Block group SVI not found: {svi_file}")
        
        self.bg_svi = pd.read_csv(svi_file, dtype={'GEOID': str})
        
        self._log(f"Loaded {len(self.tracts)} tracts, {len(self.block_groups)} block groups")
    
    def load_tract_inventory(self, min_addresses: int = 50) -> pd.DataFrame:
        """Load tract inventory."""
        inventory_file = os.path.join(self.data_dir, '..', 'tract_inventory.csv')
        if not os.path.exists(inventory_file):
            inventory_file = './tract_inventory.csv'
        
        if os.path.exists(inventory_file):
            inventory = pd.read_csv(inventory_file, dtype={'FIPS': str})
            usable = inventory[inventory['Addresses'] >= min_addresses].copy()
            self._log(f"Loaded {len(usable)} tracts with >= {min_addresses} addresses")
            return usable
        
        return None
    
    def create_holdout_split(self, holdout_fraction: float = 0.2,
                            seed: int = 42) -> Tuple[List[str], List[str]]:
        """Create stratified holdout split of block groups."""
        np.random.seed(seed)
        
        valid_bg = self.bg_svi[self.bg_svi['SVI'].notna()]['GEOID'].tolist()
        
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
        
        training_bgs = []
        validation_bgs = []
        
        for tract, bgs in tract_to_bgs.items():
            n_bgs = len(bgs)
            if n_bgs == 1:
                training_bgs.extend(bgs)
            elif n_bgs == 2:
                np.random.shuffle(bgs)
                training_bgs.append(bgs[0])
                validation_bgs.append(bgs[1])
            else:
                np.random.shuffle(bgs)
                n_holdout = max(1, int(n_bgs * holdout_fraction))
                validation_bgs.extend(bgs[:n_holdout])
                training_bgs.extend(bgs[n_holdout:])
        
        self._log(f"Holdout split: {len(training_bgs)} training, {len(validation_bgs)} validation BGs")
        
        return training_bgs, validation_bgs
    
    def run_single_tract_residual(self, fips: str, training_bg_ids: List[str],
                                   seed: int = 42) -> Optional[Dict]:
        """Run residual learning on a single tract."""
        set_random_seed(seed)
        
        tract_data = self.tracts[self.tracts['FIPS'] == fips]
        if len(tract_data) == 0:
            return None
        
        tract_svi = float(tract_data.iloc[0]['RPL_THEMES'])
        tract_geom = tract_data.geometry.iloc[0]
        
        # Load addresses
        try:
            addresses = self.loader.get_addresses_for_tract(fips)
        except Exception as e:
            self._log(f"Failed to load addresses for {fips}: {e}", level='WARN')
            return None
        
        if len(addresses) < 20:
            return None
        
        # Get tract's block groups
        tract_bgs = self.block_groups[self.block_groups['tract_fips'] == fips].copy()
        
        # Assign addresses to BGs
        if addresses.crs != tract_bgs.crs:
            addresses = addresses.to_crs(tract_bgs.crs)
        
        joined = gpd.sjoin(
            addresses,
            tract_bgs[['GEOID', 'geometry']],
            how='left',
            predicate='within'
        )
        joined = joined.rename(columns={'GEOID': 'block_group_id'})
        if 'index_right' in joined.columns:
            joined = joined.drop(columns=['index_right'])
        addresses = joined
        
        # Compute IDW baseline
        address_coords = np.column_stack([
            addresses.geometry.x.values,
            addresses.geometry.y.values
        ])
        
        idw = IDWDisaggregation(power=2.0, n_neighbors=8)
        idw.fit(self.tracts, svi_column='RPL_THEMES')
        idw_predictions = idw.disaggregate(address_coords, fips, tract_svi)
        
        # Compute Kriging baseline
        try:
            kriging = OrdinaryKrigingDisaggregation()
            kriging.fit(self.tracts, svi_column='RPL_THEMES')
            kriging_predictions = kriging.disaggregate(address_coords, fips, tract_svi)
        except:
            kriging_predictions = idw_predictions.copy()
        
        # Compute residual targets for training BGs
        svi_lookup = dict(zip(self.bg_svi['GEOID'], self.bg_svi['SVI']))
        
        bg_masks = {}
        bg_residuals = {}
        n_training_bgs = 0
        
        for bg_id in addresses['block_group_id'].dropna().unique():
            # Only use training BGs
            if bg_id not in training_bg_ids:
                continue
            if bg_id not in svi_lookup or pd.isna(svi_lookup[bg_id]):
                continue
            
            mask = (addresses['block_group_id'] == bg_id).values
            if mask.sum() < 5:
                continue
            
            bg_true_svi = svi_lookup[bg_id]
            bg_idw_mean = idw_predictions[mask].mean()
            bg_residual = bg_true_svi - bg_idw_mean
            
            bg_masks[bg_id] = mask
            bg_residuals[bg_id] = bg_residual
            n_training_bgs += 1
        
        # Compute spatial features
        feature_computer = SpatialFeatureComputer(verbose=False)
        features, feature_names = feature_computer.compute_features(
            addresses, tract_geom, data_loader=self.loader
        )
        normalized_features, _ = normalize_spatial_features(features)
        
        # Build graph
        graph_data = self.loader.create_spatial_graph(addresses, normalized_features)
        
        # Train residual model
        model = SpatialDisaggregationGNN(
            input_dim=normalized_features.shape[1],
            hidden_dim=32,
            dropout=0.2
        )
        
        trainer = ResidualGNNTrainer(
            model,
            learning_rate=0.001,
            seed=seed,
            bg_weight=1.0,
            smoothness_weight=1.0
        )
        
        result = trainer.train_residual(
            graph_data=graph_data,
            bg_masks=bg_masks,
            bg_residuals=bg_residuals,
            epochs=100,
            verbose=False
        )
        
        # Compute final predictions: IDW + residual
        predicted_residuals = result['raw_predictions']
        residual_predictions = idw_predictions + predicted_residuals
        residual_predictions = np.clip(residual_predictions, 0, 1)
        
        # Apply tract mean correction
        correction = tract_svi - np.mean(residual_predictions)
        residual_predictions = np.clip(residual_predictions + correction, 0, 1)
        
        return {
            'success': True,
            'addresses': addresses,
            'tract_svi': tract_svi,
            'idw_predictions': idw_predictions,
            'kriging_predictions': kriging_predictions,
            'residual_predictions': residual_predictions,
            'raw_residuals': predicted_residuals,
            'n_training_bgs': n_training_bgs,
        }
    
    def run_county_validation(self, tract_list: List[str] = None,
                              max_tracts: int = None,
                              holdout_fraction: float = 0.2,
                              seed: int = 42) -> Dict:
        """Run residual learning validation across county."""
        start_time = time.time()
        
        # Load data
        self.load_data()
        
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
                raise ValueError("No tract list provided")
        
        if max_tracts:
            tract_list = tract_list[:max_tracts]
        
        print(f"\n{'='*70}")
        print("GRANITE COUNTY-WIDE RESIDUAL LEARNING VALIDATION")
        print(f"{'='*70}")
        print(f"Tracts to process: {len(tract_list)}")
        print(f"Training block groups: {len(training_bg_ids)}")
        print(f"Validation block groups: {len(validation_bg_ids)}")
        print(f"{'='*70}\n")
        
        # Process all tracts
        all_addresses = []
        successful_tracts = 0
        failed_tracts = []
        
        for i, fips in enumerate(tract_list):
            progress = f"[{i+1}/{len(tract_list)}]"
            print(f"{progress} Processing tract {fips}...", end=" ", flush=True)
            
            results = self.run_single_tract_residual(fips, training_bg_ids, seed=seed)
            
            if results is None:
                failed_tracts.append(fips)
                print("FAILED")
                continue
            
            successful_tracts += 1
            n_bg = results['n_training_bgs']
            print(f"OK ({n_bg} training BGs)")
            
            # Collect predictions
            addresses = results['addresses'].copy()
            addresses['IDW_pred'] = results['idw_predictions']
            addresses['Kriging_pred'] = results['kriging_predictions']
            addresses['Residual_pred'] = results['residual_predictions']
            addresses['raw_residual'] = results['raw_residuals']
            addresses['tract_fips'] = fips
            
            all_addresses.append(addresses)
        
        if len(all_addresses) == 0:
            return {'success': False, 'error': 'No tracts processed'}
        
        # Combine
        combined = pd.concat(all_addresses, ignore_index=True)
        combined = gpd.GeoDataFrame(combined, crs='EPSG:4326')
        
        print(f"\n{'='*70}")
        print("Aggregating to block groups...")
        print(f"{'='*70}")
        
        # Aggregate to validation BGs
        validation_results = self._aggregate_to_bgs(combined, validation_bg_ids)
        training_results = self._aggregate_to_bgs(combined, training_bg_ids)
        
        elapsed = time.time() - start_time
        
        # Compute statistics
        val_stats = self._compute_method_stats(validation_results)
        train_stats = self._compute_method_stats(training_results)
        
        # Statistical comparison
        comparison = self._compare_methods(validation_results)
        
        return {
            'success': True,
            'validation': {
                'data': validation_results,
                'stats': val_stats,
                'n_block_groups': len(validation_results),
            },
            'training': {
                'data': training_results,
                'stats': train_stats,
                'n_block_groups': len(training_results),
            },
            'comparison': comparison,
            'n_tracts': successful_tracts,
            'failed_tracts': failed_tracts,
            'n_addresses': len(combined),
            'elapsed_time': elapsed,
            'training_bg_ids': training_bg_ids,
            'validation_bg_ids': validation_bg_ids,
        }
    
    def _aggregate_to_bgs(self, addresses: gpd.GeoDataFrame,
                          bg_ids: List[str]) -> pd.DataFrame:
        """Aggregate address predictions to block group level."""
        svi_lookup = dict(zip(self.bg_svi['GEOID'], self.bg_svi['SVI']))
        
        results = []
        for bg_id in bg_ids:
            if bg_id not in svi_lookup or pd.isna(svi_lookup[bg_id]):
                continue
            
            mask = addresses['block_group_id'] == bg_id
            n_addr = mask.sum()
            
            if n_addr < 3:
                continue
            
            bg_data = addresses[mask]
            
            results.append({
                'GEOID': bg_id,
                'SVI': svi_lookup[bg_id],
                'n_addresses': n_addr,
                'IDW_pred': bg_data['IDW_pred'].mean(),
                'Kriging_pred': bg_data['Kriging_pred'].mean(),
                'Residual_pred': bg_data['Residual_pred'].mean(),
                'mean_raw_residual': bg_data['raw_residual'].mean(),
            })
        
        return pd.DataFrame(results)
    
    def _compute_method_stats(self, df: pd.DataFrame) -> Dict:
        """Compute correlation stats for each method."""
        if len(df) < 3:
            return {}
        
        methods = ['IDW', 'Kriging', 'Residual']
        stats_dict = {}
        
        for method in methods:
            col = f'{method}_pred'
            if col not in df.columns:
                continue
            
            valid = df[['SVI', col]].dropna()
            if len(valid) < 3:
                continue
            
            r, p = stats.pearsonr(valid['SVI'], valid[col])
            rmse = np.sqrt(np.mean((valid['SVI'] - valid[col])**2))
            mae = np.mean(np.abs(valid['SVI'] - valid[col]))
            
            stats_dict[method] = {
                'pearson_r': r,
                'pearson_p': p,
                'rmse': rmse,
                'mae': mae,
            }
        
        return stats_dict
    
    def _compare_methods(self, df: pd.DataFrame) -> Dict:
        """Statistical comparison of IDW vs Residual."""
        if len(df) < 5:
            return {}
        
        valid = df[['SVI', 'IDW_pred', 'Residual_pred']].dropna()
        
        r_idw = np.corrcoef(valid['SVI'], valid['IDW_pred'])[0, 1]
        r_res = np.corrcoef(valid['SVI'], valid['Residual_pred'])[0, 1]
        
        # Fisher z-transformation for comparing correlations
        n = len(valid)
        z_idw = 0.5 * np.log((1 + r_idw) / (1 - r_idw))
        z_res = 0.5 * np.log((1 + r_res) / (1 - r_res))
        
        se = np.sqrt(2 / (n - 3))
        z_diff = (z_res - z_idw) / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_diff)))
        
        return {
            'idw_r': r_idw,
            'residual_r': r_res,
            'r_diff': r_res - r_idw,
            'z_stat': z_diff,
            'p_value': p_value,
            'significant': p_value < 0.05,
        }
    
    def print_report(self, results: Dict):
        """Print validation report."""
        print(f"\n{'='*70}")
        print("GRANITE COUNTY-WIDE RESIDUAL LEARNING REPORT")
        print(f"{'='*70}")
        
        print(f"\nData Summary:")
        print(f"  Tracts processed: {results['n_tracts']}")
        print(f"  Total addresses: {results['n_addresses']:,}")
        print(f"  Elapsed time: {results['elapsed_time']:.1f}s")
        
        val = results['validation']
        print(f"\n{'='*70}")
        print(f"VALIDATION SET RESULTS ({val['n_block_groups']} held-out block groups)")
        print(f"{'='*70}")
        
        stats = val['stats']
        print(f"\n  {'Method':<15} {'Pearson r':<12} {'RMSE':<12} {'MAE':<12}")
        print(f"  {'-'*50}")
        
        for method in ['IDW', 'Residual', 'Kriging']:
            if method in stats:
                s = stats[method]
                sig = '***' if s['pearson_p'] < 0.001 else '**' if s['pearson_p'] < 0.01 else '*' if s['pearson_p'] < 0.05 else ''
                print(f"  {method:<15} {s['pearson_r']:.4f}{sig:<4} {s['rmse']:<12.4f} {s['mae']:<12.4f}")
        
        # Training set (overfitting check)
        train = results['training']
        print(f"\n{'='*70}")
        print(f"TRAINING SET RESULTS ({train['n_block_groups']} training block groups)")
        print(f"{'='*70}")
        
        train_stats = train['stats']
        print(f"\n  {'Method':<15} {'Pearson r':<12} {'RMSE':<12}")
        print(f"  {'-'*40}")
        
        for method in ['IDW', 'Residual', 'Kriging']:
            if method in train_stats:
                s = train_stats[method]
                print(f"  {method:<15} {s['pearson_r']:.4f}       {s['rmse']:.4f}")
        
        # Comparison
        comp = results['comparison']
        if comp:
            print(f"\n{'='*70}")
            print("STATISTICAL COMPARISON (Validation Set)")
            print(f"{'='*70}")
            print(f"\n  IDW r = {comp['idw_r']:.4f}")
            print(f"  IDW + Residual r = {comp['residual_r']:.4f}")
            print(f"  Difference = {comp['r_diff']:+.4f}")
            print(f"\n  Fisher z-test: z = {comp['z_stat']:.3f}, p = {comp['p_value']:.4f}")
            
            if comp['significant']:
                if comp['r_diff'] > 0:
                    print(f"\n  ★ RESIDUAL LEARNING SIGNIFICANTLY BETTER (p < 0.05)")
                else:
                    print(f"\n  ✗ IDW significantly better (p < 0.05)")
            else:
                if comp['r_diff'] > 0:
                    print(f"\n  ~ Residual better but not significant")
                else:
                    print(f"\n  ~ No significant difference")
    
    def create_validation_plot(self, results: Dict) -> plt.Figure:
        """Create validation visualization."""
        val_data = results['validation']['data']
        val_stats = results['validation']['stats']
        comp = results['comparison']
        
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        colors = {'IDW': '#A23B72', 'Residual': '#2E86AB', 'Kriging': '#F18F01'}
        
        # Row 1: Scatter plots
        for i, (method, color) in enumerate([('IDW', colors['IDW']), 
                                              ('Residual', colors['Residual']),
                                              ('Kriging', colors['Kriging'])]):
            ax = fig.add_subplot(gs[0, i])
            col = f'{method}_pred'
            
            if col in val_data.columns:
                ax.scatter(val_data['SVI'], val_data[col], alpha=0.6, s=40, c=color)
                ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
                
                # Regression line
                valid = val_data[['SVI', col]].dropna()
                if len(valid) > 2:
                    z = np.polyfit(valid['SVI'], valid[col], 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(0, 1, 100)
                    ax.plot(x_line, p(x_line), color=color, linestyle=':', alpha=0.8)
                
                r = val_stats.get(method, {}).get('pearson_r', 0)
                ax.set_title(f'{method}\nr = {r:.3f}***')
            
            ax.set_xlabel('Ground Truth SVI')
            ax.set_ylabel('Predicted SVI')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
        
        # Row 2: Method comparison bars
        ax4 = fig.add_subplot(gs[1, 0])
        methods = ['IDW', 'Residual', 'Kriging']
        r_vals = [val_stats.get(m, {}).get('pearson_r', 0) for m in methods]
        bars = ax4.bar(methods, r_vals, color=[colors[m] for m in methods], alpha=0.8)
        ax4.set_ylabel('Pearson Correlation (r)')
        ax4.set_title('Method Comparison: Correlation')
        ax4.set_ylim(0, 1)
        for bar, r in zip(bars, r_vals):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{r:.3f}', ha='center', fontsize=10, fontweight='bold')
        
        ax5 = fig.add_subplot(gs[1, 1])
        rmse_vals = [val_stats.get(m, {}).get('rmse', 0) for m in methods]
        bars = ax5.bar(methods, rmse_vals, color=[colors[m] for m in methods], alpha=0.8)
        ax5.set_ylabel('RMSE (lower is better)')
        ax5.set_title('Method Comparison: RMSE')
        for bar, rmse in zip(bars, rmse_vals):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{rmse:.3f}', ha='center', fontsize=10)
        
        # Residual distribution
        ax6 = fig.add_subplot(gs[1, 2])
        if 'mean_raw_residual' in val_data.columns:
            ax6.hist(val_data['mean_raw_residual'], bins=20, color=colors['Residual'],
                    alpha=0.7, edgecolor='black')
            ax6.axvline(0, color='black', linestyle='--', linewidth=1.5)
            ax6.set_xlabel('Mean Predicted Residual (per BG)')
            ax6.set_ylabel('Count')
            ax6.set_title('Residual Distribution\n(0 = IDW was correct)')
        
        # Row 3: IDW vs Residual comparison, distribution, summary
        ax7 = fig.add_subplot(gs[2, 0])
        ax7.scatter(val_data['IDW_pred'], val_data['Residual_pred'],
                   alpha=0.6, s=40, c='steelblue')
        ax7.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        r_agree = np.corrcoef(val_data['IDW_pred'].dropna(), 
                              val_data['Residual_pred'].dropna())[0, 1]
        ax7.set_xlabel('IDW Predicted SVI')
        ax7.set_ylabel('IDW + Residual Predicted SVI')
        ax7.set_title(f'Method Agreement\nr = {r_agree:.3f}')
        ax7.set_xlim(0, 1)
        ax7.set_ylim(0, 1)
        ax7.set_aspect('equal')
        
        # Distribution comparison
        ax8 = fig.add_subplot(gs[2, 1])
        data_to_plot = [val_data['SVI'].dropna(),
                       val_data['IDW_pred'].dropna(),
                       val_data['Residual_pred'].dropna()]
        bp = ax8.boxplot(data_to_plot, labels=['Ground\nTruth', 'IDW', 'IDW+\nResidual'],
                        patch_artist=True)
        bp['boxes'][0].set_facecolor('lightgray')
        bp['boxes'][1].set_facecolor(colors['IDW'])
        bp['boxes'][2].set_facecolor(colors['Residual'])
        for box in bp['boxes']:
            box.set_alpha(0.7)
        ax8.set_ylabel('SVI Value')
        ax8.set_title('Distribution Comparison')
        
        # Summary text
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')
        
        summary_lines = [
            "SUMMARY STATISTICS",
            "=" * 35,
            f"Validation Block Groups: {results['validation']['n_block_groups']}",
            f"Total Addresses: {results['n_addresses']:,}",
            "",
            "CORRELATION WITH GROUND TRUTH:",
            f"  IDW:          r = {comp.get('idw_r', 0):.3f}",
            f"  IDW+Residual: r = {comp.get('residual_r', 0):.3f}",
            f"  Difference:   {comp.get('r_diff', 0):+.3f}",
            "",
            "STATISTICAL TEST:",
            f"  z = {comp.get('z_stat', 0):.2f}, p = {comp.get('p_value', 1):.4f}",
            "",
        ]
        
        if comp.get('significant'):
            if comp.get('r_diff', 0) > 0:
                summary_lines.append("★ RESIDUAL LEARNING WINS")
            else:
                summary_lines.append("✗ IDW wins")
        else:
            summary_lines.append("~ No significant difference")
        
        ax9.text(0.05, 0.95, '\n'.join(summary_lines), transform=ax9.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        n_val = results['validation']['n_block_groups']
        n_addr = results['n_addresses']
        n_tracts = results['n_tracts']
        
        fig.suptitle(f"County-Wide Residual Learning Validation\n"
                    f"{n_val} Block Groups | {n_addr:,} Addresses | {n_tracts} Tracts",
                    fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, 'county_residual_validation.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"\nValidation plot saved to: {filepath}")
        
        return fig
    
    def save_results(self, results: Dict):
        """Save results to CSV."""
        val_data = results['validation']['data']
        val_file = os.path.join(self.output_dir, 'county_residual_validation_bgs.csv')
        val_data.to_csv(val_file, index=False)
        print(f"Block group results saved to: {val_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='GRANITE County Residual Validation')
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--output', type=str, default='./output')
    parser.add_argument('--max-tracts', type=int, default=None)
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    validator = CountyResidualValidation(
        data_dir=args.data_dir,
        output_dir=args.output,
        verbose=args.verbose
    )
    
    results = validator.run_county_validation(
        max_tracts=args.max_tracts,
        seed=args.seed
    )
    
    if results['success']:
        validator.print_report(results)
        validator.create_validation_plot(results)
        validator.save_results(results)
        return 0
    else:
        print(f"\nValidation failed: {results.get('error')}")
        return 1


if __name__ == '__main__':
    sys.exit(main())