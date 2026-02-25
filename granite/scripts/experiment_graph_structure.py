"""
GRANITE Graph Structure Experiments

Tests whether different graph connectivity patterns improve GNN performance:

1. k-NN (current): Connect k nearest neighbors by Euclidean distance
2. Road Network: Connect addresses reachable via road network
3. Cross-BG Edges: Explicitly connect addresses across block group boundaries
4. Distance-Weighted: Weight edges by inverse distance
5. Hybrid: Combine road network + cross-BG connections

Hypothesis: The current k-NN graph may create "echo chambers" within BGs,
preventing the GNN from learning cross-boundary patterns.

Usage:
    python ./granite/scripts/experiment_graph_structure.py --fips 47065000600 -v
    python ./granite/scripts/experiment_graph_structure.py --county -v  # All tracts
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from scipy import stats
import torch
from torch_geometric.data import Data
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/workspaces/GRANITE')

from granite.models.gnn import set_random_seed, SpatialDisaggregationGNN, SpatialGNNTrainer
from granite.data.loaders import DataLoader
from granite.features.spatial_features import SpatialFeatureComputer, normalize_spatial_features
from granite.evaluation.baselines import IDWDisaggregation


class GraphStructureExperiment:
    """
    Experiments with different graph construction methods.
    """
    
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
        
        # Tracts
        tracts = self.loader.load_census_tracts(state_fips, county_fips)
        county_name = self.loader._get_county_name(state_fips, county_fips)
        svi = self.loader.load_svi_data(state_fips, county_name)
        self.tracts = tracts.merge(svi, on='FIPS', how='inner')
        
        # Block groups
        bg_file = os.path.join(self.data_dir, 'raw', f'tl_2020_{state_fips}_bg.shp')
        bg_gdf = gpd.read_file(bg_file)
        county_bg = bg_gdf[bg_gdf['COUNTYFP'] == county_fips].copy()
        county_bg['GEOID'] = county_bg['GEOID'].astype(str)
        county_bg['tract_fips'] = state_fips + county_fips + county_bg['TRACTCE'].astype(str)
        if county_bg.crs != 'EPSG:4326':
            county_bg = county_bg.to_crs(epsg=4326)
        self.block_groups = county_bg
        
        # BG SVI
        svi_file = os.path.join(self.data_dir, 'processed', 'acs_block_groups_svi.csv')
        self.bg_svi = pd.read_csv(svi_file, dtype={'GEOID': str})
        
        self._log(f"Loaded {len(self.tracts)} tracts, {len(self.block_groups)} block groups")
    
    # =========================================================================
    # GRAPH CONSTRUCTION METHODS
    # =========================================================================
    
    def create_knn_graph(self, addresses: gpd.GeoDataFrame, features: np.ndarray,
                         k: int = 8) -> Data:
        """Standard k-NN graph (current method)."""
        n = len(addresses)
        coords = np.column_stack([
            addresses.geometry.x.values,
            addresses.geometry.y.values
        ])
        
        tree = cKDTree(coords)
        _, indices = tree.query(coords, k=min(k + 1, n))
        
        edge_list = []
        for i in range(n):
            for j in indices[i, 1:]:
                if j < n:
                    edge_list.append([i, j])
                    edge_list.append([j, i])
        
        edge_set = set(tuple(e) for e in edge_list)
        edge_list = list(edge_set)
        
        edge_index = torch.LongTensor(edge_list).t().contiguous()
        x = torch.FloatTensor(features)
        
        return Data(x=x, edge_index=edge_index)
    
    def create_distance_weighted_graph(self, addresses: gpd.GeoDataFrame,
                                        features: np.ndarray,
                                        k: int = 12,
                                        sigma: float = 0.001) -> Data:
        """
        k-NN with distance-weighted edges.
        Edge weight = exp(-distance^2 / sigma^2)
        """
        n = len(addresses)
        coords = np.column_stack([
            addresses.geometry.x.values,
            addresses.geometry.y.values
        ])
        
        tree = cKDTree(coords)
        distances, indices = tree.query(coords, k=min(k + 1, n))
        
        edge_list = []
        edge_weights = []
        
        for i in range(n):
            for idx, j in enumerate(indices[i, 1:]):
                if j < n:
                    dist = distances[i, idx + 1]
                    weight = np.exp(-dist**2 / sigma**2)
                    
                    edge_list.append([i, j])
                    edge_weights.append(weight)
                    edge_list.append([j, i])
                    edge_weights.append(weight)
        
        edge_index = torch.LongTensor(edge_list).t().contiguous()
        edge_attr = torch.FloatTensor(edge_weights).unsqueeze(1)
        x = torch.FloatTensor(features)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    def create_cross_bg_graph(self, addresses: gpd.GeoDataFrame,
                               features: np.ndarray,
                               k_within: int = 6,
                               k_cross: int = 4,
                               max_cross_dist: float = 0.005) -> Data:
        """
        Graph with explicit cross-block-group connections.
        
        - k_within neighbors from same BG
        - k_cross neighbors from different BGs (within distance threshold)
        
        This encourages information flow across BG boundaries.
        """
        n = len(addresses)
        coords = np.column_stack([
            addresses.geometry.x.values,
            addresses.geometry.y.values
        ])
        
        bg_ids = addresses['block_group_id'].values
        
        edge_list = []
        
        # For each address, find within-BG and cross-BG neighbors
        tree = cKDTree(coords)
        distances, indices = tree.query(coords, k=min(k_within + k_cross + 5, n))
        
        for i in range(n):
            my_bg = bg_ids[i]
            
            within_count = 0
            cross_count = 0
            
            for idx, j in enumerate(indices[i, 1:]):
                if j >= n:
                    continue
                
                neighbor_bg = bg_ids[j]
                dist = distances[i, idx + 1]
                
                if pd.isna(my_bg) or pd.isna(neighbor_bg):
                    # Unknown BG - treat as within
                    if within_count < k_within:
                        edge_list.append([i, j])
                        edge_list.append([j, i])
                        within_count += 1
                elif my_bg == neighbor_bg:
                    # Same BG
                    if within_count < k_within:
                        edge_list.append([i, j])
                        edge_list.append([j, i])
                        within_count += 1
                else:
                    # Different BG - only if within distance threshold
                    if cross_count < k_cross and dist < max_cross_dist:
                        edge_list.append([i, j])
                        edge_list.append([j, i])
                        cross_count += 1
        
        edge_set = set(tuple(e) for e in edge_list)
        edge_list = list(edge_set)
        
        edge_index = torch.LongTensor(edge_list).t().contiguous()
        x = torch.FloatTensor(features)
        
        self._log(f"Cross-BG graph: {n} nodes, {edge_index.shape[1]} edges")
        
        return Data(x=x, edge_index=edge_index)
    
    def create_boundary_focused_graph(self, addresses: gpd.GeoDataFrame,
                                       features: np.ndarray,
                                       tract_bgs: gpd.GeoDataFrame,
                                       k_base: int = 6,
                                       boundary_boost: int = 4) -> Data:
        """
        Extra connections for addresses near BG boundaries.
        
        Addresses within X meters of a BG boundary get additional
        cross-boundary neighbors.
        """
        n = len(addresses)
        coords = np.column_stack([
            addresses.geometry.x.values,
            addresses.geometry.y.values
        ])
        
        # Compute distance to nearest BG boundary for each address
        boundary_distances = []
        for idx, row in addresses.iterrows():
            point = row.geometry
            min_dist = float('inf')
            for _, bg in tract_bgs.iterrows():
                if bg.geometry is not None and hasattr(bg.geometry, 'boundary'):
                    dist = point.distance(bg.geometry.boundary)
                    min_dist = min(min_dist, dist)
            boundary_distances.append(min_dist)
        
        boundary_distances = np.array(boundary_distances)
        
        # Addresses near boundary (within ~100m in degrees)
        near_boundary = boundary_distances < 0.001
        
        self._log(f"Addresses near BG boundary: {near_boundary.sum()}/{n}")
        
        # Build graph with extra edges for boundary addresses
        tree = cKDTree(coords)
        
        edge_list = []
        
        for i in range(n):
            # Base k-NN
            k = k_base + boundary_boost if near_boundary[i] else k_base
            _, indices = tree.query(coords[i], k=min(k + 1, n))
            
            for j in indices[1:]:
                if j < n:
                    edge_list.append([i, j])
                    edge_list.append([j, i])
        
        edge_set = set(tuple(e) for e in edge_list)
        edge_list = list(edge_set)
        
        edge_index = torch.LongTensor(edge_list).t().contiguous()
        x = torch.FloatTensor(features)
        
        return Data(x=x, edge_index=edge_index)
    
    def create_road_network_graph(self, addresses: gpd.GeoDataFrame,
                                   features: np.ndarray,
                                   state_fips: str = '47',
                                   county_fips: str = '065') -> Data:
        """Use actual road network connectivity."""
        return self.loader.create_road_network_graph(
            addresses, features, state_fips, county_fips,
            k_neighbors=8, max_path_length=1500.0
        )
    
    # =========================================================================
    # EXPERIMENT RUNNER
    # =========================================================================
    
    def run_single_tract_experiment(self, fips: str, training_bg_ids: List[str],
                                     seed: int = 42) -> Dict:
        """
        Run all graph structure experiments on a single tract.
        """
        set_random_seed(seed)
        
        state_fips = fips[:2]
        county_fips = fips[2:5]
        
        tract_data = self.tracts[self.tracts['FIPS'] == fips]
        if len(tract_data) == 0:
            return None
        
        tract_svi = float(tract_data.iloc[0]['RPL_THEMES'])
        tract_geom = tract_data.geometry.iloc[0]
        
        # Load addresses
        addresses = self.loader.get_addresses_for_tract(fips)
        if len(addresses) < 20:
            return None
        
        # Assign to block groups
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
        features, feature_names = feature_computer.compute_features(
            addresses, tract_geom, data_loader=self.loader
        )
        normalized_features, _ = normalize_spatial_features(features)
        
        # Create BG training targets
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
        
        # Create different graphs
        graphs = {
            'kNN_k8': self.create_knn_graph(addresses, normalized_features, k=8),
            'kNN_k12': self.create_knn_graph(addresses, normalized_features, k=12),
            'kNN_k16': self.create_knn_graph(addresses, normalized_features, k=16),
            'cross_BG': self.create_cross_bg_graph(addresses, normalized_features,
                                                    k_within=6, k_cross=4),
            'boundary_focus': self.create_boundary_focused_graph(
                addresses, normalized_features, tract_bgs, k_base=6, boundary_boost=4
            ),
        }
        
        # Try road network if available
        try:
            road_graph = self.create_road_network_graph(
                addresses, normalized_features, state_fips, county_fips
            )
            graphs['road_network'] = road_graph
        except Exception as e:
            self._log(f"Road network graph failed: {e}")
        
        # Train with each graph structure
        results = {}
        
        for graph_name, graph_data in graphs.items():
            self._log(f"  Training with {graph_name} graph...")
            
            set_random_seed(seed)
            
            model = SpatialDisaggregationGNN(
                input_dim=normalized_features.shape[1],
                hidden_dim=32,
                dropout=0.2
            )
            
            trainer = SpatialGNNTrainer(
                model,
                learning_rate=0.001,
                bg_weight=0.2,
                coherence_weight=0.0,
                discrimination_weight=0.0,
                smoothness_weight=2.0,
                variation_weight=1.5,
                seed=seed
            )
            
            train_result = trainer.train(
                graph_data=graph_data,
                tract_svi=tract_svi,
                epochs=100,
                verbose=False,
                bg_masks=bg_masks,
                bg_svis=bg_svis
            )
            
            predictions = train_result['raw_predictions']
            correction = tract_svi - np.mean(predictions)
            predictions = np.clip(predictions + correction, 0, 1)
            
            results[graph_name] = {
                'predictions': predictions,
                'n_edges': graph_data.edge_index.shape[1],
            }
        
        # Also compute IDW baseline
        address_coords = np.column_stack([
            addresses.geometry.x.values,
            addresses.geometry.y.values
        ])
        idw = IDWDisaggregation(power=2.0, n_neighbors=8)
        idw.fit(self.tracts, svi_column='RPL_THEMES')
        idw_predictions = idw.disaggregate(address_coords, fips, tract_svi)
        
        results['IDW'] = {
            'predictions': idw_predictions,
            'n_edges': 0,
        }
        
        return {
            'success': True,
            'addresses': addresses,
            'tract_svi': tract_svi,
            'results': results,
            'n_training_bgs': len(bg_masks),
        }
    
    def run_county_experiment(self, max_tracts: int = None,
                               holdout_fraction: float = 0.2,
                               seed: int = 42) -> Dict:
        """Run graph structure experiments across county."""
        start_time = time.time()
        
        # Load data
        self.load_data()
        
        # Create holdout split
        training_bg_ids, validation_bg_ids = self._create_holdout_split(
            holdout_fraction, seed
        )
        
        # Get tract list
        inventory_file = os.path.join(self.data_dir, '..', 'tract_inventory.csv')
        if not os.path.exists(inventory_file):
            inventory_file = './tract_inventory.csv'
        
        inventory = pd.read_csv(inventory_file, dtype={'FIPS': str})
        tract_list = inventory[inventory['Addresses'] >= 50]['FIPS'].tolist()
        
        if max_tracts:
            tract_list = tract_list[:max_tracts]
        
        print(f"\n{'='*70}")
        print("GRANITE GRAPH STRUCTURE EXPERIMENT")
        print(f"{'='*70}")
        print(f"Tracts to process: {len(tract_list)}")
        print(f"Training BGs: {len(training_bg_ids)}")
        print(f"Validation BGs: {len(validation_bg_ids)}")
        print(f"{'='*70}\n")
        
        # Collect all predictions
        all_addresses = []
        
        for i, fips in enumerate(tract_list):
            print(f"[{i+1}/{len(tract_list)}] Processing {fips}...", end=" ", flush=True)
            
            result = self.run_single_tract_experiment(fips, training_bg_ids, seed)
            
            if result is None:
                print("FAILED")
                continue
            
            print(f"OK ({result['n_training_bgs']} BGs)")
            
            # Collect predictions
            addresses = result['addresses'].copy()
            for graph_name, graph_result in result['results'].items():
                addresses[f'{graph_name}_pred'] = graph_result['predictions']
            addresses['tract_fips'] = fips
            
            all_addresses.append(addresses)
        
        if not all_addresses:
            return {'success': False, 'error': 'No tracts processed'}
        
        # Combine
        combined = pd.concat(all_addresses, ignore_index=True)
        combined = gpd.GeoDataFrame(combined, crs='EPSG:4326')
        
        # Aggregate to validation BGs
        validation_results = self._aggregate_to_bgs(combined, validation_bg_ids)
        
        elapsed = time.time() - start_time
        
        # Compute stats for each graph type
        graph_stats = {}
        pred_cols = [c for c in validation_results.columns if c.endswith('_pred')]
        
        for col in pred_cols:
            graph_name = col.replace('_pred', '')
            valid = validation_results[['SVI', col]].dropna()
            
            if len(valid) >= 3:
                r, p = stats.pearsonr(valid['SVI'], valid[col])
                rmse = np.sqrt(np.mean((valid['SVI'] - valid[col])**2))
                
                graph_stats[graph_name] = {
                    'pearson_r': r,
                    'pearson_p': p,
                    'rmse': rmse,
                }
        
        return {
            'success': True,
            'validation_data': validation_results,
            'graph_stats': graph_stats,
            'n_tracts': len(all_addresses),
            'n_validation_bgs': len(validation_results),
            'elapsed_time': elapsed,
        }
    
    def _create_holdout_split(self, holdout_fraction: float,
                               seed: int) -> Tuple[List[str], List[str]]:
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
    
    def _aggregate_to_bgs(self, addresses: gpd.GeoDataFrame,
                          bg_ids: List[str]) -> pd.DataFrame:
        """Aggregate predictions to BG level."""
        svi_lookup = dict(zip(self.bg_svi['GEOID'], self.bg_svi['SVI']))
        
        pred_cols = [c for c in addresses.columns if c.endswith('_pred')]
        
        results = []
        for bg_id in bg_ids:
            if bg_id not in svi_lookup or pd.isna(svi_lookup[bg_id]):
                continue
            
            mask = addresses['block_group_id'] == bg_id
            if mask.sum() < 3:
                continue
            
            row = {
                'GEOID': bg_id,
                'SVI': svi_lookup[bg_id],
                'n_addresses': mask.sum(),
            }
            
            for col in pred_cols:
                row[col] = addresses.loc[mask, col].mean()
            
            results.append(row)
        
        return pd.DataFrame(results)
    
    def print_report(self, results: Dict):
        """Print experiment report."""
        print(f"\n{'='*70}")
        print("GRAPH STRUCTURE EXPERIMENT RESULTS")
        print(f"{'='*70}")
        
        print(f"\nValidation Block Groups: {results['n_validation_bgs']}")
        print(f"Tracts Processed: {results['n_tracts']}")
        print(f"Elapsed Time: {results['elapsed_time']:.1f}s")
        
        print(f"\n{'Graph Type':<20} {'Pearson r':<12} {'RMSE':<12}")
        print("-" * 45)
        
        # Sort by correlation
        sorted_stats = sorted(
            results['graph_stats'].items(),
            key=lambda x: x[1]['pearson_r'],
            reverse=True
        )
        
        for graph_name, stats in sorted_stats:
            r = stats['pearson_r']
            rmse = stats['rmse']
            print(f"{graph_name:<20} {r:.4f}       {rmse:.4f}")
        
        # Best graph
        best = sorted_stats[0]
        idw_stats = results['graph_stats'].get('IDW', {})
        
        print(f"\n{'='*70}")
        print("COMPARISON")
        print(f"{'='*70}")
        print(f"Best Graph: {best[0]} (r = {best[1]['pearson_r']:.4f})")
        if idw_stats:
            diff = best[1]['pearson_r'] - idw_stats['pearson_r']
            print(f"IDW Baseline: r = {idw_stats['pearson_r']:.4f}")
            print(f"Improvement over IDW: {diff:+.4f}")
    
    def create_comparison_plot(self, results: Dict) -> plt.Figure:
        """Create visualization comparing graph structures."""
        val_data = results['validation_data']
        stats = results['graph_stats']
        
        # Get graph types (excluding IDW for the GNN comparison)
        gnn_graphs = [g for g in stats.keys() if g != 'IDW']
        
        n_graphs = len(gnn_graphs) + 1  # +1 for IDW
        n_cols = min(4, n_graphs)
        n_rows = (n_graphs + n_cols - 1) // n_cols + 1  # Extra row for bar chart
        
        fig = plt.figure(figsize=(5 * n_cols, 4 * n_rows))
        
        # Scatter plots for each method
        for i, graph_name in enumerate(['IDW'] + gnn_graphs):
            ax = fig.add_subplot(n_rows, n_cols, i + 1)
            
            col = f'{graph_name}_pred'
            if col not in val_data.columns:
                continue
            
            valid = val_data[['SVI', col]].dropna()
            
            ax.scatter(valid['SVI'], valid[col], alpha=0.6, s=30)
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            
            r = stats.get(graph_name, {}).get('pearson_r', 0)
            ax.set_title(f'{graph_name}\nr = {r:.3f}')
            ax.set_xlabel('Ground Truth SVI')
            ax.set_ylabel('Predicted SVI')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
        
        # Bar chart comparing all methods
        ax_bar = fig.add_subplot(n_rows, 1, n_rows)
        
        methods = list(stats.keys())
        r_vals = [stats[m]['pearson_r'] for m in methods]
        
        colors = ['#A23B72' if m == 'IDW' else '#2E86AB' for m in methods]
        
        bars = ax_bar.bar(methods, r_vals, color=colors, alpha=0.8)
        ax_bar.set_ylabel('Pearson Correlation (r)')
        ax_bar.set_title('Graph Structure Comparison')
        ax_bar.set_ylim(0, 1)
        
        # Rotate labels
        ax_bar.set_xticklabels(methods, rotation=45, ha='right')
        
        for bar, r in zip(bars, r_vals):
            ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{r:.3f}', ha='center', fontsize=9)
        
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, 'graph_structure_comparison.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {filepath}")
        
        return fig


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='GRANITE Graph Structure Experiment')
    parser.add_argument('--fips', type=str, default=None,
                        help='Single tract FIPS (omit for county-wide)')
    parser.add_argument('--county', action='store_true',
                        help='Run county-wide experiment')
    parser.add_argument('--max-tracts', type=int, default=None)
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--output', type=str, default='./output')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    experiment = GraphStructureExperiment(
        data_dir=args.data_dir,
        output_dir=args.output,
        verbose=args.verbose
    )
    
    if args.county or args.fips is None:
        # County-wide experiment
        results = experiment.run_county_experiment(
            max_tracts=args.max_tracts,
            seed=args.seed
        )
        
        if results['success']:
            experiment.print_report(results)
            experiment.create_comparison_plot(results)
        else:
            print(f"Experiment failed: {results.get('error')}")
            return 1
    else:
        # Single tract experiment
        experiment.load_data()
        
        # Use all BGs for single tract (no holdout)
        all_bgs = experiment.bg_svi[experiment.bg_svi['SVI'].notna()]['GEOID'].tolist()
        
        result = experiment.run_single_tract_experiment(args.fips, all_bgs, args.seed)
        
        if result:
            print(f"\nTract {args.fips} Results:")
            print(f"{'Graph Type':<20} {'Std Dev':<12}")
            print("-" * 35)
            for name, data in result['results'].items():
                std = np.std(data['predictions'])
                print(f"{name:<20} {std:.4f}")
        else:
            print(f"Failed to process tract {args.fips}")
            return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())