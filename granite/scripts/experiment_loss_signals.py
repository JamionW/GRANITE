"""
GRANITE Loss Signal Experiments

Tests alternative supervision strategies:

1. NO_BG_SIGNAL: Tract constraint + smoothness only, no block group info during training
2. CENTROID_ANCHORS: Point supervision at BG centroids (knows values, not boundaries)
3. CURRENT: Standard BG mean matching (baseline for comparison)

All validated against held-out block groups.

Usage:
    python ./granite/scripts/experiment_loss_signals.py -v
    python ./granite/scripts/experiment_loss_signals.py --max-tracts 10 -v
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import stats
from scipy.spatial import cKDTree
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/workspaces/GRANITE')

from granite.models.gnn import set_random_seed, SpatialDisaggregationGNN
from granite.data.loaders import DataLoader
from granite.features.spatial_features import SpatialFeatureComputer, normalize_spatial_features
from granite.evaluation.baselines import IDWDisaggregation


class NoBGSignalTrainer:
    """
    Trainer using ONLY tract constraint + spatial smoothness.
    No block group information during training.
    """
    
    def __init__(self, model, learning_rate=0.001, seed=42,
                 smoothness_weight=2.0, variation_weight=1.0):
        self.model = model
        self.seed = seed
        self.smoothness_weight = smoothness_weight
        self.variation_weight = variation_weight
        
        set_random_seed(seed)
        
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=1e-4
        )
    
    def train(self, graph_data, tract_svi, epochs=100, verbose=False):
        """Train with tract constraint and smoothness only."""
        set_random_seed(self.seed)
        self.model.train()
        
        target_mean = torch.FloatTensor([tract_svi])
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            predictions = self.model(graph_data.x, graph_data.edge_index)
            
            # Loss 1: Tract mean constraint
            pred_mean = predictions.mean()
            tract_loss = F.mse_loss(pred_mean.unsqueeze(0), target_mean)
            
            # Loss 2: Spatial smoothness
            src, dst = graph_data.edge_index[0], graph_data.edge_index[1]
            diff = predictions[src] - predictions[dst]
            smoothness_loss = torch.mean(diff ** 2)
            
            # Loss 3: Variation (prevent collapse)
            pred_std = predictions.std()
            variation_loss = F.relu(0.02 - pred_std)
            
            total_loss = (
                tract_loss +
                self.smoothness_weight * smoothness_loss +
                self.variation_weight * variation_loss
            )
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
        
        self.model.eval()
        with torch.no_grad():
            final_preds = self.model(graph_data.x, graph_data.edge_index)
        
        return {'raw_predictions': final_preds.numpy()}


class CentroidAnchorTrainer:
    """
    Trainer using BG centroid locations as point supervision.
    Knows SVI values at specific points, but not BG boundaries.
    """
    
    def __init__(self, model, learning_rate=0.001, seed=42,
                 anchor_weight=1.0, smoothness_weight=1.5, variation_weight=1.0):
        self.model = model
        self.seed = seed
        self.anchor_weight = anchor_weight
        self.smoothness_weight = smoothness_weight
        self.variation_weight = variation_weight
        
        set_random_seed(seed)
        
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=1e-4
        )
    
    def train(self, graph_data, tract_svi, anchor_indices, anchor_values,
              epochs=100, verbose=False):
        """
        Train with point supervision at anchor locations.
        
        Args:
            graph_data: PyG Data object
            tract_svi: Tract-level SVI for mean constraint
            anchor_indices: List of address indices near BG centroids
            anchor_values: List of BG SVI values for those anchors
        """
        set_random_seed(self.seed)
        self.model.train()
        
        target_mean = torch.FloatTensor([tract_svi])
        anchor_idx_tensor = torch.LongTensor(anchor_indices)
        anchor_val_tensor = torch.FloatTensor(anchor_values)
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            predictions = self.model(graph_data.x, graph_data.edge_index)
            
            # Loss 1: Tract mean constraint
            pred_mean = predictions.mean()
            tract_loss = F.mse_loss(pred_mean.unsqueeze(0), target_mean)
            
            # Loss 2: Anchor point supervision
            anchor_preds = predictions[anchor_idx_tensor]
            anchor_loss = F.mse_loss(anchor_preds, anchor_val_tensor)
            
            # Loss 3: Spatial smoothness
            src, dst = graph_data.edge_index[0], graph_data.edge_index[1]
            diff = predictions[src] - predictions[dst]
            smoothness_loss = torch.mean(diff ** 2)
            
            # Loss 4: Variation
            pred_std = predictions.std()
            variation_loss = F.relu(0.02 - pred_std)
            
            total_loss = (
                tract_loss +
                self.anchor_weight * anchor_loss +
                self.smoothness_weight * smoothness_loss +
                self.variation_weight * variation_loss
            )
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
        
        self.model.eval()
        with torch.no_grad():
            final_preds = self.model(graph_data.x, graph_data.edge_index)
        
        return {'raw_predictions': final_preds.numpy()}


class StandardBGTrainer:
    """Standard BG mean matching (current approach) for comparison."""
    
    def __init__(self, model, learning_rate=0.001, seed=42,
                 bg_weight=0.2, smoothness_weight=2.0, variation_weight=1.5):
        self.model = model
        self.seed = seed
        self.bg_weight = bg_weight
        self.smoothness_weight = smoothness_weight
        self.variation_weight = variation_weight
        
        set_random_seed(seed)
        
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=1e-4
        )
    
    def train(self, graph_data, tract_svi, bg_masks, bg_svis,
              epochs=100, verbose=False):
        """Train with BG mean constraints."""
        set_random_seed(self.seed)
        self.model.train()
        
        target_mean = torch.FloatTensor([tract_svi])
        
        bg_masks_tensor = {bg: torch.BoolTensor(m) for bg, m in bg_masks.items()}
        bg_svis_tensor = {bg: torch.FloatTensor([s]) for bg, s in bg_svis.items()}
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            predictions = self.model(graph_data.x, graph_data.edge_index)
            
            # Loss 1: Tract mean
            pred_mean = predictions.mean()
            tract_loss = F.mse_loss(pred_mean.unsqueeze(0), target_mean)
            
            # Loss 2: BG mean matching
            bg_losses = []
            for bg_id, mask in bg_masks_tensor.items():
                if mask.sum() < 3:
                    continue
                bg_pred_mean = predictions[mask].mean()
                bg_target = bg_svis_tensor[bg_id]
                bg_losses.append(F.mse_loss(bg_pred_mean.unsqueeze(0), bg_target))
            
            if bg_losses:
                bg_loss = torch.stack(bg_losses).mean()
            else:
                bg_loss = torch.tensor(0.0)
            
            # Loss 3: Smoothness
            src, dst = graph_data.edge_index[0], graph_data.edge_index[1]
            diff = predictions[src] - predictions[dst]
            smoothness_loss = torch.mean(diff ** 2)
            
            # Loss 4: Variation
            pred_std = predictions.std()
            variation_loss = F.relu(0.02 - pred_std)
            
            total_loss = (
                tract_loss +
                self.bg_weight * bg_loss +
                self.smoothness_weight * smoothness_loss +
                self.variation_weight * variation_loss
            )
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
        
        self.model.eval()
        with torch.no_grad():
            final_preds = self.model(graph_data.x, graph_data.edge_index)
        
        return {'raw_predictions': final_preds.numpy()}


class LossSignalExperiment:
    """Run experiments comparing loss signal approaches."""
    
    def __init__(self, data_dir: str = './data', output_dir: str = './output',
                 verbose: bool = False):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.verbose = verbose
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.loader = None
        self.tracts = None
        self.block_groups = None
        self.bg_svi = None
    
    def _log(self, msg: str):
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {msg}")
    
    def load_data(self, state_fips: str = '47', county_fips: str = '065'):
        """Load all required data."""
        self._log("Loading data...")
        
        self.loader = DataLoader(self.data_dir)
        
        tracts = self.loader.load_census_tracts(state_fips, county_fips)
        county_name = self.loader._get_county_name(state_fips, county_fips)
        svi = self.loader.load_svi_data(state_fips, county_name)
        self.tracts = tracts.merge(svi, on='FIPS', how='inner')
        
        bg_file = os.path.join(self.data_dir, 'raw', f'tl_2020_{state_fips}_bg.shp')
        bg_gdf = gpd.read_file(bg_file)
        county_bg = bg_gdf[bg_gdf['COUNTYFP'] == county_fips].copy()
        county_bg['GEOID'] = county_bg['GEOID'].astype(str)
        county_bg['tract_fips'] = state_fips + county_fips + county_bg['TRACTCE'].astype(str)
        if county_bg.crs != 'EPSG:4326':
            county_bg = county_bg.to_crs(epsg=4326)
        self.block_groups = county_bg
        
        svi_file = os.path.join(self.data_dir, 'processed', 'acs_block_groups_svi.csv')
        self.bg_svi = pd.read_csv(svi_file, dtype={'GEOID': str})
        
        self._log(f"Loaded {len(self.tracts)} tracts, {len(self.block_groups)} block groups")
    
    def create_holdout_split(self, holdout_fraction: float = 0.2,
                             seed: int = 42) -> Tuple[List[str], List[str]]:
        """Create stratified BG holdout split."""
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
            n = len(bgs)
            if n == 1:
                training_bgs.extend(bgs)
            elif n == 2:
                np.random.shuffle(bgs)
                training_bgs.append(bgs[0])
                validation_bgs.append(bgs[1])
            else:
                np.random.shuffle(bgs)
                n_holdout = max(1, int(n * holdout_fraction))
                validation_bgs.extend(bgs[:n_holdout])
                training_bgs.extend(bgs[n_holdout:])
        
        return training_bgs, validation_bgs
    
    def find_centroid_anchors(self, addresses: gpd.GeoDataFrame,
                               tract_bgs: gpd.GeoDataFrame,
                               training_bg_ids: List[str]) -> Tuple[List[int], List[float]]:
        """
        Find address nearest to each training BG centroid.
        Returns (anchor_indices, anchor_svi_values).
        """
        svi_lookup = dict(zip(self.bg_svi['GEOID'], self.bg_svi['SVI']))
        
        addr_coords = np.column_stack([
            addresses.geometry.x.values,
            addresses.geometry.y.values
        ])
        tree = cKDTree(addr_coords)
        
        anchor_indices = []
        anchor_values = []
        
        for _, bg_row in tract_bgs.iterrows():
            bg_id = bg_row['GEOID']
            
            if bg_id not in training_bg_ids:
                continue
            if bg_id not in svi_lookup or pd.isna(svi_lookup[bg_id]):
                continue
            
            centroid = bg_row.geometry.centroid
            centroid_coords = np.array([[centroid.x, centroid.y]])
            
            _, idx = tree.query(centroid_coords, k=1)
            
            anchor_indices.append(int(idx[0]))
            anchor_values.append(svi_lookup[bg_id])
        
        return anchor_indices, anchor_values
    
    def run_single_tract(self, fips: str, training_bg_ids: List[str],
                          seed: int = 42) -> Optional[Dict]:
        """Run all three approaches on a single tract."""
        set_random_seed(seed)
        
        state_fips = fips[:2]
        county_fips = fips[2:5]
        
        tract_data = self.tracts[self.tracts['FIPS'] == fips]
        if len(tract_data) == 0:
            return None
        
        tract_svi = float(tract_data.iloc[0]['RPL_THEMES'])
        tract_geom = tract_data.geometry.iloc[0]
        
        addresses = self.loader.get_addresses_for_tract(fips)
        if len(addresses) < 20:
            return None
        
        tract_bgs = self.block_groups[self.block_groups['tract_fips'] == fips].copy()
        
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
        
        # Compute features
        feature_computer = SpatialFeatureComputer(verbose=False)
        features, _ = feature_computer.compute_features(
            addresses, tract_geom, data_loader=self.loader
        )
        normalized_features, _ = normalize_spatial_features(features)
        
        # Build road network graph
        graph_data = self.loader.create_road_network_graph(
            addresses, normalized_features, state_fips, county_fips, k_neighbors=8
        )
        
        # Prepare BG masks for standard approach
        svi_lookup = dict(zip(self.bg_svi['GEOID'], self.bg_svi['SVI']))
        
        bg_masks = {}
        bg_svis = {}
        for bg_id in addresses['block_group_id'].dropna().unique():
            if bg_id not in training_bg_ids:
                continue
            if bg_id not in svi_lookup or pd.isna(svi_lookup[bg_id]):
                continue
            mask = (addresses['block_group_id'] == bg_id).values
            if mask.sum() < 5:
                continue
            bg_masks[bg_id] = mask
            bg_svis[bg_id] = svi_lookup[bg_id]
        
        # Find centroid anchors
        anchor_indices, anchor_values = self.find_centroid_anchors(
            addresses, tract_bgs, training_bg_ids
        )
        
        results = {}
        
        # 1. NO_BG_SIGNAL
        set_random_seed(seed)
        model1 = SpatialDisaggregationGNN(
            input_dim=normalized_features.shape[1], hidden_dim=32, dropout=0.2
        )
        trainer1 = NoBGSignalTrainer(model1, seed=seed)
        res1 = trainer1.train(graph_data, tract_svi, epochs=100)
        preds1 = res1['raw_predictions']
        preds1 = preds1 + (tract_svi - preds1.mean())
        results['no_bg_signal'] = np.clip(preds1, 0, 1)
        
        # 2. CENTROID_ANCHORS
        if len(anchor_indices) >= 1:
            set_random_seed(seed)
            model2 = SpatialDisaggregationGNN(
                input_dim=normalized_features.shape[1], hidden_dim=32, dropout=0.2
            )
            trainer2 = CentroidAnchorTrainer(model2, seed=seed)
            res2 = trainer2.train(graph_data, tract_svi, anchor_indices, anchor_values, epochs=100)
            preds2 = res2['raw_predictions']
            preds2 = preds2 + (tract_svi - preds2.mean())
            results['centroid_anchors'] = np.clip(preds2, 0, 1)
        else:
            results['centroid_anchors'] = results['no_bg_signal'].copy()
        
        # 3. STANDARD (BG mean matching)
        if len(bg_masks) >= 1:
            set_random_seed(seed)
            model3 = SpatialDisaggregationGNN(
                input_dim=normalized_features.shape[1], hidden_dim=32, dropout=0.2
            )
            trainer3 = StandardBGTrainer(model3, seed=seed)
            res3 = trainer3.train(graph_data, tract_svi, bg_masks, bg_svis, epochs=100)
            preds3 = res3['raw_predictions']
            preds3 = preds3 + (tract_svi - preds3.mean())
            results['standard_bg'] = np.clip(preds3, 0, 1)
        else:
            results['standard_bg'] = results['no_bg_signal'].copy()
        
        # IDW baseline
        addr_coords = np.column_stack([
            addresses.geometry.x.values, addresses.geometry.y.values
        ])
        idw = IDWDisaggregation(power=2.0, n_neighbors=8)
        idw.fit(self.tracts, svi_column='RPL_THEMES')
        results['idw'] = idw.disaggregate(addr_coords, fips, tract_svi)
        
        return {
            'success': True,
            'addresses': addresses,
            'results': results,
            'n_training_bgs': len(bg_masks),
            'n_anchors': len(anchor_indices),
        }
    
    def run_county_experiment(self, max_tracts: int = None,
                               holdout_fraction: float = 0.2,
                               seed: int = 42) -> Dict:
        """Run experiment across county."""
        start_time = time.time()
        
        self.load_data()
        
        training_bg_ids, validation_bg_ids = self.create_holdout_split(
            holdout_fraction, seed
        )
        
        inventory_file = './tract_inventory.csv'
        if not os.path.exists(inventory_file):
            inventory_file = os.path.join(self.data_dir, '..', 'tract_inventory.csv')
        
        inventory = pd.read_csv(inventory_file, dtype={'FIPS': str})
        tract_list = inventory[inventory['Addresses'] >= 50]['FIPS'].tolist()
        
        if max_tracts:
            tract_list = tract_list[:max_tracts]
        
        print(f"\n{'='*70}")
        print("GRANITE LOSS SIGNAL EXPERIMENT")
        print(f"{'='*70}")
        print(f"Tracts: {len(tract_list)}")
        print(f"Training BGs: {len(training_bg_ids)}, Validation BGs: {len(validation_bg_ids)}")
        print(f"{'='*70}\n")
        
        all_addresses = []
        
        for i, fips in enumerate(tract_list):
            print(f"[{i+1}/{len(tract_list)}] {fips}...", end=" ", flush=True)
            
            result = self.run_single_tract(fips, training_bg_ids, seed)
            
            if result is None:
                print("FAILED")
                continue
            
            print(f"OK ({result['n_training_bgs']} BGs, {result['n_anchors']} anchors)")
            
            addresses = result['addresses'].copy()
            for method, preds in result['results'].items():
                addresses[f'{method}_pred'] = preds
            addresses['tract_fips'] = fips
            
            all_addresses.append(addresses)
        
        if not all_addresses:
            return {'success': False}
        
        combined = pd.concat(all_addresses, ignore_index=True)
        combined = gpd.GeoDataFrame(combined, crs='EPSG:4326')
        
        # Aggregate to validation BGs
        validation_results = self._aggregate_to_bgs(combined, validation_bg_ids)
        
        elapsed = time.time() - start_time
        
        # Compute stats
        method_stats = {}
        for method in ['no_bg_signal', 'centroid_anchors', 'standard_bg', 'idw']:
            col = f'{method}_pred'
            if col not in validation_results.columns:
                continue
            
            valid = validation_results[['SVI', col]].dropna()
            if len(valid) >= 3:
                r, p = stats.pearsonr(valid['SVI'], valid[col])
                rmse = np.sqrt(np.mean((valid['SVI'] - valid[col])**2))
                method_stats[method] = {'r': r, 'p': p, 'rmse': rmse}
        
        return {
            'success': True,
            'validation_data': validation_results,
            'method_stats': method_stats,
            'n_validation_bgs': len(validation_results),
            'elapsed': elapsed,
        }
    
    def _aggregate_to_bgs(self, addresses, bg_ids):
        """Aggregate to BG level."""
        svi_lookup = dict(zip(self.bg_svi['GEOID'], self.bg_svi['SVI']))
        
        pred_cols = [c for c in addresses.columns if c.endswith('_pred')]
        
        results = []
        for bg_id in bg_ids:
            if bg_id not in svi_lookup or pd.isna(svi_lookup[bg_id]):
                continue
            
            mask = addresses['block_group_id'] == bg_id
            if mask.sum() < 3:
                continue
            
            row = {'GEOID': bg_id, 'SVI': svi_lookup[bg_id], 'n_addresses': mask.sum()}
            for col in pred_cols:
                row[col] = addresses.loc[mask, col].mean()
            results.append(row)
        
        return pd.DataFrame(results)
    
    def print_report(self, results: Dict):
        """Print results."""
        print(f"\n{'='*70}")
        print("LOSS SIGNAL EXPERIMENT RESULTS")
        print(f"{'='*70}")
        print(f"Validation BGs: {results['n_validation_bgs']}")
        print(f"Elapsed: {results['elapsed']:.1f}s")
        
        print(f"\n{'Method':<20} {'Pearson r':<12} {'RMSE':<12}")
        print("-" * 45)
        
        sorted_stats = sorted(
            results['method_stats'].items(),
            key=lambda x: x[1]['r'],
            reverse=True
        )
        
        for method, stats in sorted_stats:
            print(f"{method:<20} {stats['r']:.4f}       {stats['rmse']:.4f}")
        
        # Comparison
        print(f"\n{'='*70}")
        idw_r = results['method_stats'].get('idw', {}).get('r', 0)
        
        for method, stats in sorted_stats:
            if method != 'idw':
                diff = stats['r'] - idw_r
                print(f"{method} vs IDW: {diff:+.4f}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-tracts', type=int, default=None)
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--output', type=str, default='./output')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    experiment = LossSignalExperiment(
        data_dir=args.data_dir,
        output_dir=args.output,
        verbose=args.verbose
    )
    
    results = experiment.run_county_experiment(
        max_tracts=args.max_tracts,
        seed=args.seed
    )
    
    if results['success']:
        experiment.print_report(results)
    else:
        print("Experiment failed")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())