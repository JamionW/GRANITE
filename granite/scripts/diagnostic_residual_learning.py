"""
GRANITE Residual Learning Diagnostic

Instead of predicting raw SVI, the GNN learns to predict:
    residual = true_SVI - IDW_prediction

This forces the GNN to learn ONLY what IDW misses, not rediscover
spatial interpolation. Any positive signal is genuine value-add.

The final prediction is: IDW_baseline + GNN_residual
"""
import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/workspaces/GRANITE')

from granite.models.gnn import set_random_seed, SpatialDisaggregationGNN, SpatialGNNTrainer
from granite.data.loaders import DataLoader
from granite.features.spatial_features import SpatialFeatureComputer, normalize_spatial_features
from granite.evaluation.baselines import IDWDisaggregation


def run_residual_model(fips: str, data_dir: str, training_bg_ids: list = None,
                       seed: int = 42, verbose: bool = False,
                       loss_config: dict = None) -> dict:
    """
    Run GRANITE with residual learning - predict deviation from IDW.
    
    Args:
        fips: Tract FIPS code
        data_dir: Data directory path
        training_bg_ids: If provided, only use these BGs for training
        seed: Random seed
        verbose: Print progress
        loss_config: Dict of loss weights
    
    Returns:
        Dict with predictions, IDW baseline, residuals, etc.
    """
    set_random_seed(seed)
    
    if loss_config is None:
        loss_config = {
            'bg_weight': 1.0,
            'coherence_weight': 0.0,
            'discrimination_weight': 0.0,
            'smoothness_weight': 1.0,
            'variation_weight': 0.5,
        }
    
    state_fips = fips[:2]
    county_fips = fips[2:5]
    
    # Load data
    loader = DataLoader(data_dir)
    tracts = loader.load_census_tracts(state_fips, county_fips)
    county_name = loader._get_county_name(state_fips, county_fips)
    svi = loader.load_svi_data(state_fips, county_name)
    tracts = tracts.merge(svi, on='FIPS', how='inner')
    
    tract_data = tracts[tracts['FIPS'] == fips]
    if len(tract_data) == 0:
        raise ValueError(f"Tract {fips} not found")
    
    tract_svi = float(tract_data.iloc[0]['RPL_THEMES'])
    tract_geom = tract_data.geometry.iloc[0]
    
    # Load addresses
    addresses = loader.get_addresses_for_tract(fips)
    if verbose:
        print(f"  Loaded {len(addresses)} addresses")
    
    # Load block groups
    bg_file = os.path.join(data_dir, 'raw', f'tl_2020_{state_fips}_bg.shp')
    bg_gdf = gpd.read_file(bg_file)
    county_bg = bg_gdf[bg_gdf['COUNTYFP'] == county_fips].copy()
    county_bg['GEOID'] = county_bg['GEOID'].astype(str)
    county_bg['tract_fips'] = state_fips + county_fips + county_bg['TRACTCE'].astype(str)
    if county_bg.crs != 'EPSG:4326':
        county_bg = county_bg.to_crs(epsg=4326)
    
    tract_bgs = county_bg[county_bg['tract_fips'] == fips].copy()
    
    # Load BG SVI
    svi_file = os.path.join(data_dir, 'processed', 'acs_block_groups_svi.csv')
    bg_svi = pd.read_csv(svi_file, dtype={'GEOID': str})
    svi_lookup = dict(zip(bg_svi['GEOID'], bg_svi['SVI']))
    
    # Assign addresses to block groups
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
    
    # Step 1: Compute IDW baseline for all addresses
    address_coords = np.column_stack([
        addresses.geometry.x.values,
        addresses.geometry.y.values
    ])
    
    idw = IDWDisaggregation(power=2.0, n_neighbors=8)
    idw.fit(tracts, svi_column='RPL_THEMES')
    idw_predictions = idw.disaggregate(address_coords, fips, tract_svi)
    
    if verbose:
        print(f"  IDW baseline: mean={idw_predictions.mean():.3f}, std={idw_predictions.std():.4f}")
    
    # Step 2: Compute residual targets for each BG
    # residual = BG_true_SVI - mean(IDW for addresses in BG)
    bg_masks = {}
    bg_residuals = {}  # These are the training targets
    bg_true_svis = {}
    training_bgs_used = []
    
    for bg_id in addresses['block_group_id'].dropna().unique():
        if training_bg_ids is not None and bg_id not in training_bg_ids:
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
        bg_residuals[bg_id] = bg_residual  # Target is the residual
        bg_true_svis[bg_id] = bg_true_svi
        training_bgs_used.append(bg_id)
        
        if verbose:
            print(f"    BG {bg_id}: true={bg_true_svi:.3f}, idw={bg_idw_mean:.3f}, residual={bg_residual:+.3f}")
    
    if verbose:
        print(f"  Using {len(training_bgs_used)} BGs for residual training")
        residual_vals = list(bg_residuals.values())
        print(f"  Residual range: [{min(residual_vals):.3f}, {max(residual_vals):.3f}]")
    
    # Step 3: Compute features
    feature_computer = SpatialFeatureComputer(verbose=verbose)
    features, feature_names = feature_computer.compute_features(
        addresses, tract_geom, data_loader=loader
    )
    normalized_features, _ = normalize_spatial_features(features)
    
    # Build graph
    graph_data = loader.create_spatial_graph(addresses, normalized_features)
    
    # Step 4: Train GNN to predict residuals
    model = SpatialDisaggregationGNN(
        input_dim=normalized_features.shape[1],
        hidden_dim=32,
        dropout=0.2
    )
    
    trainer = ResidualGNNTrainer(
        model,
        learning_rate=0.001,
        seed=seed,
        **loss_config
    )
    
    result = trainer.train_residual(
        graph_data=graph_data,
        idw_baseline=idw_predictions,
        bg_masks=bg_masks,
        bg_residuals=bg_residuals,
        epochs=100,
        verbose=verbose
    )
    
    # Step 5: Final prediction = IDW + residual
    predicted_residuals = result['raw_predictions']
    final_predictions = idw_predictions + predicted_residuals
    final_predictions = np.clip(final_predictions, 0, 1)
    
    # Apply tract mean correction
    correction = tract_svi - np.mean(final_predictions)
    final_predictions = np.clip(final_predictions + correction, 0, 1)
    
    return {
        'predictions': final_predictions,
        'idw_baseline': idw_predictions,
        'predicted_residuals': predicted_residuals,
        'addresses': addresses,
        'tract_svi': tract_svi,
        'tract_geom': tract_geom,
        'tract_bgs': tract_bgs,
        'bg_svi': bg_svi,
        'bg_residuals': bg_residuals,
        'bg_true_svis': bg_true_svis,
        'training_bgs': training_bgs_used,
    }


class ResidualGNNTrainer:
    """
    Trainer for residual learning - predicts deviation from IDW baseline.
    """
    
    def __init__(self, model, learning_rate=0.001, seed=42,
                 bg_weight=1.0, coherence_weight=0.0, discrimination_weight=0.0,
                 smoothness_weight=1.0, variation_weight=0.5):
        
        import torch
        import torch.nn.functional as F
        
        self.model = model
        self.seed = seed
        self.bg_weight = bg_weight
        self.coherence_weight = coherence_weight
        self.discrimination_weight = discrimination_weight
        self.smoothness_weight = smoothness_weight
        self.variation_weight = variation_weight
        
        set_random_seed(seed)
        
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.8, min_lr=1e-5
        )
    
    def train_residual(self, graph_data, idw_baseline, bg_masks, bg_residuals,
                       epochs=100, verbose=True):
        """
        Train GNN to predict residuals from IDW baseline.
        
        Args:
            graph_data: PyG Data object
            idw_baseline: IDW predictions for all addresses
            bg_masks: {bg_id: boolean mask}
            bg_residuals: {bg_id: target residual} - what IDW got wrong
            epochs: Training epochs
            verbose: Print progress
        """
        import torch
        import torch.nn.functional as F
        
        set_random_seed(self.seed)
        self.model.train()
        
        # Convert to tensors
        idw_tensor = torch.FloatTensor(idw_baseline)
        
        bg_masks_tensor = {}
        bg_targets_tensor = {}
        for bg_id, mask in bg_masks.items():
            if bg_id in bg_residuals:
                bg_masks_tensor[bg_id] = torch.BoolTensor(mask)
                bg_targets_tensor[bg_id] = torch.FloatTensor([bg_residuals[bg_id]])
        
        if verbose:
            print(f"  Training residual model with {len(bg_masks_tensor)} BG constraints")
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            # GNN predicts residuals (unbounded, can be positive or negative)
            predicted_residuals = self.model(graph_data.x, graph_data.edge_index)
            
            # Loss 1: BG residual matching
            # Mean predicted residual in each BG should match true residual
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
            
            # Loss 2: Smoothness of residuals
            src, dst = graph_data.edge_index[0], graph_data.edge_index[1]
            residual_diff = predicted_residuals[src] - predicted_residuals[dst]
            smoothness_loss = torch.mean(residual_diff ** 2)
            
            # Loss 3: Residual magnitude regularization
            # Penalize very large residuals (IDW shouldn't be THAT wrong)
            magnitude_loss = torch.mean(predicted_residuals ** 2)
            
            # Combined loss
            total_loss = (
                self.bg_weight * bg_loss +
                self.smoothness_weight * smoothness_loss +
                0.1 * magnitude_loss  # Light regularization
            )
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step(total_loss)
            
            # Early stopping
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= 15:
                if verbose:
                    print(f"  Early stopping at epoch {epoch}")
                break
            
            if verbose and epoch % 20 == 0:
                residual_std = float(predicted_residuals.std())
                print(f"  Epoch {epoch:3d}: Loss={total_loss.item():.6f}, "
                      f"BG={bg_loss.item():.4f}, Residual std={residual_std:.4f}")
        
        # Final predictions
        self.model.eval()
        with torch.no_grad():
            final_residuals = self.model(graph_data.x, graph_data.edge_index)
        
        return {
            'raw_predictions': final_residuals.numpy(),
            'epochs_trained': epoch + 1,
        }


def create_residual_diagnostic_plot(fips: str, data_dir: str = './data',
                                    output_dir: str = './output', seed: int = 42):
    """
    Create diagnostic comparing standard GNN vs residual learning GNN.
    """
    print(f"\n{'='*70}")
    print(f"GRANITE Residual Learning Diagnostic: Tract {fips}")
    print(f"{'='*70}")
    
    state_fips = fips[:2]
    county_fips = fips[2:5]
    
    # Load block groups for holdout split
    bg_file = os.path.join(data_dir, 'raw', f'tl_2020_{state_fips}_bg.shp')
    bg_gdf = gpd.read_file(bg_file)
    county_bg = bg_gdf[bg_gdf['COUNTYFP'] == county_fips].copy()
    county_bg['GEOID'] = county_bg['GEOID'].astype(str)
    county_bg['tract_fips'] = state_fips + county_fips + county_bg['TRACTCE'].astype(str)
    
    tract_bgs = county_bg[county_bg['tract_fips'] == fips]['GEOID'].tolist()
    
    svi_file = os.path.join(data_dir, 'processed', 'acs_block_groups_svi.csv')
    bg_svi = pd.read_csv(svi_file, dtype={'GEOID': str})
    valid_bgs = [bg for bg in tract_bgs if bg in bg_svi[bg_svi['SVI'].notna()]['GEOID'].values]
    
    print(f"\nTract has {len(tract_bgs)} block groups, {len(valid_bgs)} with valid SVI")
    
    # Create holdout split
    np.random.seed(seed)
    np.random.shuffle(valid_bgs)
    n_holdout = max(1, len(valid_bgs) // 5)
    holdout_bgs = valid_bgs[:n_holdout]
    training_bgs = valid_bgs[n_holdout:]
    
    print(f"Training BGs: {len(training_bgs)}, Holdout BGs: {len(holdout_bgs)}")
    
    # Loss config for residual model
    loss_config = {
        'bg_weight': 1.0,
        'smoothness_weight': 1.0,
    }
    
    # Run overfitted residual model (all BGs)
    print(f"\n[1/4] Running OVERFITTED residual model (all {len(valid_bgs)} BGs)...")
    overfitted = run_residual_model(fips, data_dir, training_bg_ids=valid_bgs,
                                     seed=seed, verbose=True, loss_config=loss_config)
    
    # Run holdout residual model
    print(f"\n[2/4] Running HOLDOUT residual model ({len(training_bgs)} training BGs)...")
    holdout = run_residual_model(fips, data_dir, training_bg_ids=training_bgs,
                                  seed=seed, verbose=True, loss_config=loss_config)
    
    # IDW baseline is already computed
    idw_predictions = overfitted['idw_baseline']
    
    print(f"\n[3/4] Analyzing results...")
    
    # Create figure
    fig = plt.figure(figsize=(20, 14))
    
    cmap = plt.cm.RdYlGn_r
    norm = Normalize(vmin=0, vmax=1)
    residual_norm = Normalize(vmin=-0.3, vmax=0.3)
    residual_cmap = plt.cm.coolwarm
    
    addresses = overfitted['addresses']
    x = addresses.geometry.x.values
    y = addresses.geometry.y.values
    tract_bgs_gdf = overfitted['tract_bgs']
    bg_svi_df = overfitted['bg_svi']
    tract_svi = overfitted['tract_svi']
    
    # Row 1: Final predictions
    ax1 = fig.add_subplot(2, 4, 1)
    scatter1 = ax1.scatter(x, y, c=overfitted['predictions'], cmap=cmap,
                           s=3, alpha=0.7, norm=norm)
    _add_bg_boundaries(ax1, tract_bgs_gdf, bg_svi_df)
    ax1.set_title(f"Residual Model (Overfitted)\nFinal SVI, std={np.std(overfitted['predictions']):.4f}")
    ax1.set_aspect('equal')
    plt.colorbar(scatter1, ax=ax1, shrink=0.7, label='Predicted SVI')
    
    ax2 = fig.add_subplot(2, 4, 2)
    scatter2 = ax2.scatter(x, y, c=holdout['predictions'], cmap=cmap,
                           s=3, alpha=0.7, norm=norm)
    _add_bg_boundaries(ax2, tract_bgs_gdf, bg_svi_df)
    ax2.set_title(f"Residual Model (Holdout)\nFinal SVI, std={np.std(holdout['predictions']):.4f}")
    ax2.set_aspect('equal')
    plt.colorbar(scatter2, ax=ax2, shrink=0.7, label='Predicted SVI')
    
    ax3 = fig.add_subplot(2, 4, 3)
    scatter3 = ax3.scatter(x, y, c=idw_predictions, cmap=cmap,
                           s=3, alpha=0.7, norm=norm)
    _add_bg_boundaries(ax3, tract_bgs_gdf, bg_svi_df)
    ax3.set_title(f"IDW Baseline\nstd={np.std(idw_predictions):.4f}")
    ax3.set_aspect('equal')
    plt.colorbar(scatter3, ax=ax3, shrink=0.7, label='Predicted SVI')
    
    # Row 1, Panel 4: Ground truth
    ax4 = fig.add_subplot(2, 4, 4)
    _plot_bg_ground_truth(ax4, addresses, tract_bgs_gdf, bg_svi_df, holdout_bgs)
    ax4.set_title(f"Block Group Ground Truth\n(Red outline = holdout BGs)")
    ax4.set_aspect('equal')
    
    # Row 2: Residuals and analysis
    ax5 = fig.add_subplot(2, 4, 5)
    scatter5 = ax5.scatter(x, y, c=overfitted['predicted_residuals'], cmap=residual_cmap,
                           s=3, alpha=0.7, norm=residual_norm)
    _add_bg_boundaries(ax5, tract_bgs_gdf, bg_svi_df, show_svi=False)
    ax5.set_title(f"Predicted Residuals (Overfitted)\nstd={np.std(overfitted['predicted_residuals']):.4f}")
    ax5.set_aspect('equal')
    plt.colorbar(scatter5, ax=ax5, shrink=0.7, label='Residual (SVI - IDW)')
    
    ax6 = fig.add_subplot(2, 4, 6)
    scatter6 = ax6.scatter(x, y, c=holdout['predicted_residuals'], cmap=residual_cmap,
                           s=3, alpha=0.7, norm=residual_norm)
    _add_bg_boundaries(ax6, tract_bgs_gdf, bg_svi_df, show_svi=False)
    ax6.set_title(f"Predicted Residuals (Holdout)\nstd={np.std(holdout['predicted_residuals']):.4f}")
    ax6.set_aspect('equal')
    plt.colorbar(scatter6, ax=ax6, shrink=0.7, label='Residual (SVI - IDW)')
    
    # Overfitted vs Holdout comparison
    ax7 = fig.add_subplot(2, 4, 7)
    ax7.scatter(overfitted['predictions'], holdout['predictions'], alpha=0.3, s=5, c='steelblue')
    ax7.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    r = np.corrcoef(overfitted['predictions'], holdout['predictions'])[0, 1]
    ax7.set_xlabel('Overfitted Predictions')
    ax7.set_ylabel('Holdout Predictions')
    ax7.set_title(f'Overfitted vs Holdout\nr = {r:.3f}')
    ax7.set_xlim(0, 1)
    ax7.set_ylim(0, 1)
    ax7.set_aspect('equal')
    
    # BG-level validation
    ax8 = fig.add_subplot(2, 4, 8)
    _plot_bg_validation(ax8, addresses, overfitted['predictions'], holdout['predictions'],
                        idw_predictions, bg_svi_df, holdout_bgs)
    
    fig.suptitle(f"GRANITE Residual Learning: Tract {fips} | SVI: {tract_svi:.3f}\n"
                 f"GNN learns: residual = true_SVI - IDW_baseline", fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f'diagnostic_residual_{fips}.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"\nDiagnostic plot saved to: {filepath}")
    
    # Print summary
    _print_residual_summary(overfitted, holdout, idw_predictions, addresses, bg_svi_df,
                           holdout_bgs, training_bgs)
    
    return fig


def _add_bg_boundaries(ax, tract_bgs_gdf, bg_svi_df, show_svi=True):
    """Add block group boundaries."""
    svi_lookup = dict(zip(bg_svi_df['GEOID'], bg_svi_df['SVI']))
    
    for idx, row in tract_bgs_gdf.iterrows():
        geom = row.geometry
        if hasattr(geom, 'exterior'):
            bx, by = geom.exterior.xy
            ax.plot(bx, by, 'k-', linewidth=1.0, alpha=0.7)
            
            if show_svi:
                bg_id = row['GEOID']
                svi = svi_lookup.get(bg_id, np.nan)
                if not pd.isna(svi):
                    cx, cy = geom.centroid.x, geom.centroid.y
                    ax.text(cx, cy, f'{svi:.2f}', fontsize=7, ha='center', va='center',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))


def _plot_bg_ground_truth(ax, addresses, tract_bgs_gdf, bg_svi_df, holdout_bgs):
    """Plot addresses colored by their BG's actual SVI."""
    svi_lookup = dict(zip(bg_svi_df['GEOID'], bg_svi_df['SVI']))
    
    bg_svi_values = []
    for bg_id in addresses['block_group_id']:
        if pd.isna(bg_id) or bg_id not in svi_lookup:
            bg_svi_values.append(np.nan)
        else:
            bg_svi_values.append(svi_lookup[bg_id])
    
    bg_svi_values = np.array(bg_svi_values)
    
    cmap = plt.cm.RdYlGn_r
    norm = Normalize(vmin=0, vmax=1)
    
    x = addresses.geometry.x.values
    y = addresses.geometry.y.values
    
    scatter = ax.scatter(x, y, c=bg_svi_values, cmap=cmap, s=3, alpha=0.7, norm=norm)
    
    for idx, row in tract_bgs_gdf.iterrows():
        geom = row.geometry
        bg_id = row['GEOID']
        
        if hasattr(geom, 'exterior'):
            bx, by = geom.exterior.xy
            if bg_id in holdout_bgs:
                ax.plot(bx, by, 'r-', linewidth=2.5, alpha=0.9)
            else:
                ax.plot(bx, by, 'k-', linewidth=1.0, alpha=0.7)
            
            svi = svi_lookup.get(bg_id, np.nan)
            if not pd.isna(svi):
                cx, cy = geom.centroid.x, geom.centroid.y
                ax.text(cx, cy, f'{svi:.2f}', fontsize=8, ha='center', va='center',
                       fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    plt.colorbar(scatter, ax=ax, shrink=0.7, label='BG Ground Truth SVI')


def _plot_bg_validation(ax, addresses, overfitted_preds, holdout_preds, idw_preds,
                        bg_svi_df, holdout_bgs):
    """Plot BG-level validation."""
    svi_lookup = dict(zip(bg_svi_df['GEOID'], bg_svi_df['SVI']))
    
    bg_results = []
    for bg_id in addresses['block_group_id'].dropna().unique():
        if bg_id not in svi_lookup or pd.isna(svi_lookup[bg_id]):
            continue
        
        mask = addresses['block_group_id'] == bg_id
        n_addresses = mask.sum()
        if n_addresses < 3:
            continue
        
        bg_results.append({
            'bg_id': bg_id,
            'ground_truth': svi_lookup[bg_id],
            'overfitted': overfitted_preds[mask].mean(),
            'holdout': holdout_preds[mask].mean(),
            'idw': idw_preds[mask].mean(),
            'is_holdout': bg_id in holdout_bgs,
        })
    
    if not bg_results:
        ax.text(0.5, 0.5, 'No BG data', ha='center', va='center', transform=ax.transAxes)
        return
    
    df = pd.DataFrame(bg_results)
    
    train_df = df[~df['is_holdout']]
    holdout_df = df[df['is_holdout']]
    
    # Plot
    if len(train_df) > 0:
        ax.scatter(train_df['ground_truth'], train_df['overfitted'],
                  c='steelblue', marker='o', s=80, alpha=0.7, label='Residual (train BGs)')
        ax.scatter(train_df['ground_truth'], train_df['holdout'],
                  c='green', marker='s', s=80, alpha=0.7, label='Residual (train BGs)')
    
    if len(holdout_df) > 0:
        ax.scatter(holdout_df['ground_truth'], holdout_df['overfitted'],
                  c='steelblue', marker='o', s=120, alpha=0.9, edgecolors='red', linewidths=2)
        ax.scatter(holdout_df['ground_truth'], holdout_df['holdout'],
                  c='green', marker='s', s=120, alpha=0.9, edgecolors='red', linewidths=2,
                  label='Residual (held-out BGs)')
    
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    
    if len(df) > 2:
        r_over = np.corrcoef(df['ground_truth'], df['overfitted'])[0, 1]
        r_hold = np.corrcoef(df['ground_truth'], df['holdout'])[0, 1]
        r_idw = np.corrcoef(df['ground_truth'], df['idw'])[0, 1]
        
        ax.text(0.05, 0.95, f'Overfitted r: {r_over:.3f}\n'
               f'Holdout r: {r_hold:.3f}\n'
               f'IDW r: {r_idw:.3f}',
               transform=ax.transAxes, fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Ground Truth BG SVI')
    ax.set_ylabel('Predicted Mean SVI')
    ax.set_title('Block Group Validation\n(Red edge = held-out BGs)')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc='lower right', fontsize=7)
    ax.set_aspect('equal')


def _print_residual_summary(overfitted, holdout, idw_preds, addresses, bg_svi_df,
                           holdout_bgs, training_bgs):
    """Print summary statistics."""
    svi_lookup = dict(zip(bg_svi_df['GEOID'], bg_svi_df['SVI']))
    
    print(f"\n{'='*70}")
    print("RESIDUAL LEARNING SUMMARY")
    print(f"{'='*70}")
    
    print(f"\n{'Method':<25} {'Std Dev':<12} {'Residual Std':<15}")
    print("-" * 55)
    print(f"{'Residual GNN (Overfitted)':<25} {np.std(overfitted['predictions']):<12.4f} "
          f"{np.std(overfitted['predicted_residuals']):<15.4f}")
    print(f"{'Residual GNN (Holdout)':<25} {np.std(holdout['predictions']):<12.4f} "
          f"{np.std(holdout['predicted_residuals']):<15.4f}")
    print(f"{'IDW Baseline':<25} {np.std(idw_preds):<12.4f} {'N/A':<15}")
    
    # BG-level validation
    print(f"\n{'='*70}")
    print("BLOCK GROUP VALIDATION (Residual Learning)")
    print(f"{'='*70}")
    
    for bg_type, bg_ids in [('Training BGs', training_bgs), ('Holdout BGs', holdout_bgs)]:
        gt_vals, over_vals, hold_vals, idw_vals = [], [], [], []
        
        for bg_id in bg_ids:
            if bg_id not in svi_lookup or pd.isna(svi_lookup[bg_id]):
                continue
            
            mask = addresses['block_group_id'] == bg_id
            if mask.sum() < 3:
                continue
            
            gt_vals.append(svi_lookup[bg_id])
            over_vals.append(overfitted['predictions'][mask].mean())
            hold_vals.append(holdout['predictions'][mask].mean())
            idw_vals.append(idw_preds[mask].mean())
        
        if len(gt_vals) < 2:
            print(f"\n{bg_type}: Insufficient data")
            continue
        
        r_over = np.corrcoef(gt_vals, over_vals)[0, 1]
        r_hold = np.corrcoef(gt_vals, hold_vals)[0, 1]
        r_idw = np.corrcoef(gt_vals, idw_vals)[0, 1]
        
        print(f"\n{bg_type} ({len(gt_vals)} block groups):")
        print(f"  Overfitted Residual GNN r: {r_over:.3f}")
        print(f"  Holdout Residual GNN r:    {r_hold:.3f}")
        print(f"  IDW Baseline r:            {r_idw:.3f}")
    
    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print(f"{'='*70}")
    
    # Check if residual learning adds value over IDW
    residual_std = np.std(overfitted['predicted_residuals'])
    if residual_std > 0.02:
        print(f"+ GNN learns non-trivial residuals (std={residual_std:.4f})")
        print("  The model found systematic patterns IDW missed.")
    else:
        print(f"- GNN residuals are near-zero (std={residual_std:.4f})")
        print("  IDW already captures the spatial pattern; nothing to learn.")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='GRANITE Residual Learning Diagnostic')
    parser.add_argument('--fips', type=str, default='47065000600',
                        help='Tract FIPS code')
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--output', type=str, default='./output')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    create_residual_diagnostic_plot(
        fips=args.fips,
        data_dir=args.data_dir,
        output_dir=args.output,
        seed=args.seed
    )