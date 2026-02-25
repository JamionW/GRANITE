"""
GRANITE Block Group Diagnostic Visualization

Compares:
1. Overfitted model (trained with ALL block groups as constraints)
2. Holdout model (trained with ~80% of BGs, validates against held-out BGs)
3. Block group boundaries with actual SVI values

This reveals whether GNN is memorizing BG patterns vs learning transferable spatial relationships.
"""
import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/workspaces/GRANITE')

from granite.models.gnn import set_random_seed, SpatialDisaggregationGNN, SpatialGNNTrainer
from granite.data.loaders import DataLoader
from granite.features.spatial_features import SpatialFeatureComputer, normalize_spatial_features
from granite.evaluation.baselines import IDWDisaggregation


def run_single_tract_model(fips: str, data_dir: str, training_bg_ids: list = None, 
                           seed: int = 42, verbose: bool = False) -> dict:
    """
    Run GRANITE on a single tract with optional BG holdout.
    
    Args:
        fips: Tract FIPS code
        data_dir: Data directory path
        training_bg_ids: If provided, only use these BGs for training (holdout mode)
        seed: Random seed
        verbose: Print progress
    
    Returns:
        Dict with predictions, addresses, training info
    """
    set_random_seed(seed)
    
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
    
    # Filter to this tract's BGs
    tract_bgs = county_bg[county_bg['tract_fips'] == fips].copy()
    
    # Load BG SVI
    svi_file = os.path.join(data_dir, 'processed', 'acs_block_groups_svi.csv')
    bg_svi = pd.read_csv(svi_file, dtype={'GEOID': str})
    
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
    
    # Compute features
    feature_computer = SpatialFeatureComputer(verbose=verbose)
    features, feature_names = feature_computer.compute_features(
        addresses, tract_geom, data_loader=loader
    )
    normalized_features, _ = normalize_spatial_features(features)
    
    # Build graph
    graph_data = loader.create_spatial_graph(addresses, normalized_features)
    
    # Create BG masks and targets
    svi_lookup = dict(zip(bg_svi['GEOID'], bg_svi['SVI']))
    
    bg_masks = {}
    bg_svis = {}
    training_bgs_used = []
    
    for bg_id in addresses['block_group_id'].dropna().unique():
        if training_bg_ids is not None and bg_id not in training_bg_ids:
            continue
        if bg_id not in svi_lookup or pd.isna(svi_lookup[bg_id]):
            continue
        
        mask = (addresses['block_group_id'] == bg_id).values
        if mask.sum() < 5:
            continue
        
        bg_masks[bg_id] = mask
        bg_svis[bg_id] = svi_lookup[bg_id]
        training_bgs_used.append(bg_id)
    
    if verbose:
        print(f"  Using {len(training_bgs_used)} BGs for training")
    
    # Build model
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
    
    # Train
    result = trainer.train(
        graph_data=graph_data,
        tract_svi=tract_svi,
        epochs=100,
        verbose=verbose,
        bg_masks=bg_masks,
        bg_svis=bg_svis
    )
    
    # Apply constraint correction
    predictions = result['raw_predictions']
    correction = tract_svi - np.mean(predictions)
    predictions = np.clip(predictions + correction, 0, 1)
    
    return {
        'predictions': predictions,
        'addresses': addresses,
        'tract_svi': tract_svi,
        'tract_geom': tract_geom,
        'tract_bgs': tract_bgs,
        'bg_svi': bg_svi,
        'training_bgs': training_bgs_used,
        'training_history': result.get('training_history', {}).get('losses', []),
    }


def create_diagnostic_plot(fips: str, data_dir: str = './data', 
                           output_dir: str = './output', seed: int = 42):
    """
    Create diagnostic visualization comparing overfitted vs holdout models.
    """
    print(f"\n{'='*70}")
    print(f"GRANITE Block Group Diagnostic: Tract {fips}")
    print(f"{'='*70}")
    
    # Load block groups first to create holdout split
    state_fips = fips[:2]
    county_fips = fips[2:5]
    
    bg_file = os.path.join(data_dir, 'raw', f'tl_2020_{state_fips}_bg.shp')
    bg_gdf = gpd.read_file(bg_file)
    county_bg = bg_gdf[bg_gdf['COUNTYFP'] == county_fips].copy()
    county_bg['GEOID'] = county_bg['GEOID'].astype(str)
    county_bg['tract_fips'] = state_fips + county_fips + county_bg['TRACTCE'].astype(str)
    
    tract_bgs = county_bg[county_bg['tract_fips'] == fips]['GEOID'].tolist()
    
    # Load BG SVI to filter to valid BGs
    svi_file = os.path.join(data_dir, 'processed', 'acs_block_groups_svi.csv')
    bg_svi = pd.read_csv(svi_file, dtype={'GEOID': str})
    valid_bgs = [bg for bg in tract_bgs if bg in bg_svi[bg_svi['SVI'].notna()]['GEOID'].values]
    
    print(f"\nTract has {len(tract_bgs)} block groups, {len(valid_bgs)} with valid SVI")
    
    # Create holdout split (stratified within tract)
    np.random.seed(seed)
    np.random.shuffle(valid_bgs)
    n_holdout = max(1, len(valid_bgs) // 5)  # 20% holdout
    holdout_bgs = valid_bgs[:n_holdout]
    training_bgs = valid_bgs[n_holdout:]
    
    print(f"Training BGs: {len(training_bgs)}, Holdout BGs: {len(holdout_bgs)}")
    
    # Run overfitted model (ALL BGs)
    print(f"\n[1/3] Running OVERFITTED model (all {len(valid_bgs)} BGs)...")
    overfitted = run_single_tract_model(fips, data_dir, training_bg_ids=valid_bgs, 
                                         seed=seed, verbose=False)
    
    # Run holdout model (training BGs only)
    print(f"[2/3] Running HOLDOUT model ({len(training_bgs)} training BGs)...")
    holdout = run_single_tract_model(fips, data_dir, training_bg_ids=training_bgs,
                                      seed=seed, verbose=False)
    
    # Run IDW baseline
    print(f"[3/3] Running IDW baseline...")
    loader = DataLoader(data_dir)
    tracts = loader.load_census_tracts(state_fips, county_fips)
    county_name = loader._get_county_name(state_fips, county_fips)
    svi = loader.load_svi_data(state_fips, county_name)
    tracts = tracts.merge(svi, on='FIPS', how='inner')
    
    idw = IDWDisaggregation(power=2.0, n_neighbors=8)
    idw.fit(tracts, svi_column='RPL_THEMES')
    
    address_coords = np.column_stack([
        overfitted['addresses'].geometry.x.values,
        overfitted['addresses'].geometry.y.values
    ])
    idw_predictions = idw.disaggregate(address_coords, fips, overfitted['tract_svi'])
    
    # Create figure
    fig = plt.figure(figsize=(18, 14))
    
    # Top row: spatial predictions (3 methods)
    # Bottom row: BG overlay, scatter comparisons, summary
    
    cmap = plt.cm.RdYlGn_r
    norm = Normalize(vmin=0, vmax=1)
    
    addresses = overfitted['addresses']
    x = addresses.geometry.x.values
    y = addresses.geometry.y.values
    tract_bgs_gdf = overfitted['tract_bgs']
    bg_svi_df = overfitted['bg_svi']
    tract_svi = overfitted['tract_svi']
    
    # Panel 1: Overfitted model
    ax1 = fig.add_subplot(2, 3, 1)
    scatter1 = ax1.scatter(x, y, c=overfitted['predictions'], cmap=cmap, 
                           s=3, alpha=0.7, norm=norm)
    _add_bg_boundaries(ax1, tract_bgs_gdf, bg_svi_df, show_svi=False)
    ax1.set_title(f"OVERFITTED Model\n(All {len(valid_bgs)} BGs as constraints)\nstd={np.std(overfitted['predictions']):.4f}")
    ax1.set_aspect('equal')
    plt.colorbar(scatter1, ax=ax1, shrink=0.7, label='Predicted SVI')
    
    # Panel 2: Holdout model
    ax2 = fig.add_subplot(2, 3, 2)
    scatter2 = ax2.scatter(x, y, c=holdout['predictions'], cmap=cmap,
                           s=3, alpha=0.7, norm=norm)
    _add_bg_boundaries(ax2, tract_bgs_gdf, bg_svi_df, show_svi=False)
    ax2.set_title(f"HOLDOUT Model\n({len(training_bgs)} training, {len(holdout_bgs)} held out)\nstd={np.std(holdout['predictions']):.4f}")
    ax2.set_aspect('equal')
    plt.colorbar(scatter2, ax=ax2, shrink=0.7, label='Predicted SVI')
    
    # Panel 3: IDW baseline
    ax3 = fig.add_subplot(2, 3, 3)
    scatter3 = ax3.scatter(x, y, c=idw_predictions, cmap=cmap,
                           s=3, alpha=0.7, norm=norm)
    _add_bg_boundaries(ax3, tract_bgs_gdf, bg_svi_df, show_svi=False)
    ax3.set_title(f"IDW Baseline\nstd={np.std(idw_predictions):.4f}")
    ax3.set_aspect('equal')
    plt.colorbar(scatter3, ax=ax3, shrink=0.7, label='Predicted SVI')
    
    # Panel 4: Block group ground truth with boundaries
    ax4 = fig.add_subplot(2, 3, 4)
    _plot_bg_ground_truth(ax4, addresses, tract_bgs_gdf, bg_svi_df, holdout_bgs)
    ax4.set_title(f"Block Group Ground Truth\n(Red outline = holdout BGs)")
    ax4.set_aspect('equal')
    
    # Panel 5: Overfitted vs Holdout scatter
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.scatter(overfitted['predictions'], holdout['predictions'], alpha=0.3, s=5, c='steelblue')
    ax5.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    r = np.corrcoef(overfitted['predictions'], holdout['predictions'])[0, 1]
    ax5.set_xlabel('Overfitted Predictions')
    ax5.set_ylabel('Holdout Predictions')
    ax5.set_title(f'Overfitted vs Holdout\nr = {r:.3f}')
    ax5.set_xlim(0, 1)
    ax5.set_ylim(0, 1)
    ax5.set_aspect('equal')
    
    # Panel 6: BG-level validation
    ax6 = fig.add_subplot(2, 3, 6)
    _plot_bg_validation(ax6, addresses, overfitted['predictions'], holdout['predictions'],
                       idw_predictions, bg_svi_df, holdout_bgs)
    
    # Title
    fig.suptitle(f"GRANITE Diagnostic: Tract {fips} | SVI: {tract_svi:.3f}\n"
                 f"Comparing Overfitted vs Holdout Models with Block Group Ground Truth",
                 fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f'diagnostic_bg_{fips}.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"\nDiagnostic plot saved to: {filepath}")
    
    # Print summary
    _print_summary(overfitted, holdout, idw_predictions, addresses, bg_svi_df, 
                  holdout_bgs, training_bgs)
    
    return fig


def _add_bg_boundaries(ax, tract_bgs_gdf, bg_svi_df, show_svi=True):
    """Add block group boundaries to axes."""
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
    """Plot addresses colored by their block group's actual SVI."""
    svi_lookup = dict(zip(bg_svi_df['GEOID'], bg_svi_df['SVI']))
    
    # Color addresses by their BG's SVI
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
    
    # Add BG boundaries
    for idx, row in tract_bgs_gdf.iterrows():
        geom = row.geometry
        bg_id = row['GEOID']
        
        if hasattr(geom, 'exterior'):
            bx, by = geom.exterior.xy
            
            # Highlight holdout BGs
            if bg_id in holdout_bgs:
                ax.plot(bx, by, 'r-', linewidth=2.5, alpha=0.9)
            else:
                ax.plot(bx, by, 'k-', linewidth=1.0, alpha=0.7)
            
            # Show SVI value
            svi = svi_lookup.get(bg_id, np.nan)
            if not pd.isna(svi):
                cx, cy = geom.centroid.x, geom.centroid.y
                ax.text(cx, cy, f'{svi:.2f}', fontsize=8, ha='center', va='center',
                       fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    plt.colorbar(scatter, ax=ax, shrink=0.7, label='BG Ground Truth SVI')


def _plot_bg_validation(ax, addresses, overfitted_preds, holdout_preds, idw_preds,
                       bg_svi_df, holdout_bgs):
    """Plot BG-level aggregated predictions vs ground truth."""
    svi_lookup = dict(zip(bg_svi_df['GEOID'], bg_svi_df['SVI']))
    
    # Aggregate predictions to BG level
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
            'n_addresses': n_addresses
        })
    
    if not bg_results:
        ax.text(0.5, 0.5, 'No BG data', ha='center', va='center', transform=ax.transAxes)
        return
    
    df = pd.DataFrame(bg_results)
    
    # Split by holdout status
    train_df = df[~df['is_holdout']]
    holdout_df = df[df['is_holdout']]
    
    # Plot training BGs
    if len(train_df) > 0:
        ax.scatter(train_df['ground_truth'], train_df['overfitted'], 
                  c='steelblue', marker='o', s=80, alpha=0.7, label='Overfitted (train BGs)')
        ax.scatter(train_df['ground_truth'], train_df['holdout'],
                  c='green', marker='s', s=80, alpha=0.7, label='Holdout (train BGs)')
    
    # Plot holdout BGs
    if len(holdout_df) > 0:
        ax.scatter(holdout_df['ground_truth'], holdout_df['overfitted'],
                  c='steelblue', marker='o', s=120, alpha=0.9, edgecolors='red', linewidths=2)
        ax.scatter(holdout_df['ground_truth'], holdout_df['holdout'],
                  c='green', marker='s', s=120, alpha=0.9, edgecolors='red', linewidths=2,
                  label='Holdout (held-out BGs)')
    
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    
    # Compute correlations
    if len(df) > 2:
        r_over = np.corrcoef(df['ground_truth'], df['overfitted'])[0, 1]
        r_hold = np.corrcoef(df['ground_truth'], df['holdout'])[0, 1]
        
        # Holdout-only correlations
        if len(holdout_df) > 1:
            r_hold_only = np.corrcoef(holdout_df['ground_truth'], holdout_df['holdout'])[0, 1]
            ax.text(0.05, 0.95, f'Overfitted r: {r_over:.3f}\n'
                   f'Holdout r: {r_hold:.3f}\n'
                   f'Holdout (on held-out): {r_hold_only:.3f}',
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.text(0.05, 0.95, f'Overfitted r: {r_over:.3f}\n'
                   f'Holdout r: {r_hold:.3f}',
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Ground Truth BG SVI')
    ax.set_ylabel('Predicted Mean SVI')
    ax.set_title('Block Group Validation\n(Red edge = held-out BGs)')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc='lower right', fontsize=7)
    ax.set_aspect('equal')


def _print_summary(overfitted, holdout, idw_preds, addresses, bg_svi_df, 
                  holdout_bgs, training_bgs):
    """Print summary statistics."""
    svi_lookup = dict(zip(bg_svi_df['GEOID'], bg_svi_df['SVI']))
    
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}")
    
    print(f"\n{'Method':<20} {'Std Dev':<12} {'Constraint Err':<15}")
    print("-" * 50)
    print(f"{'Overfitted GNN':<20} {np.std(overfitted['predictions']):<12.4f} "
          f"{abs(np.mean(overfitted['predictions']) - overfitted['tract_svi']) / overfitted['tract_svi'] * 100:.2f}%")
    print(f"{'Holdout GNN':<20} {np.std(holdout['predictions']):<12.4f} "
          f"{abs(np.mean(holdout['predictions']) - holdout['tract_svi']) / holdout['tract_svi'] * 100:.2f}%")
    print(f"{'IDW':<20} {np.std(idw_preds):<12.4f} "
          f"{abs(np.mean(idw_preds) - overfitted['tract_svi']) / overfitted['tract_svi'] * 100:.2f}%")
    
    # BG-level validation
    print(f"\n{'='*70}")
    print("BLOCK GROUP VALIDATION")
    print(f"{'='*70}")
    
    # Aggregate to BG level
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
        print(f"  Overfitted GNN r: {r_over:.3f}")
        print(f"  Holdout GNN r:    {r_hold:.3f}")
        print(f"  IDW r:            {r_idw:.3f}")
    
    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print(f"{'='*70}")
    
    over_std = np.std(overfitted['predictions'])
    hold_std = np.std(holdout['predictions'])
    
    if over_std > hold_std * 1.3:
        print("! Overfitted model shows MORE variation than holdout model")
        print("  This suggests the GNN may be memorizing BG boundaries rather than")
        print("  learning transferable spatial patterns.")
    elif over_std < hold_std * 0.7:
        print("? Holdout model shows MORE variation than overfitted model")
        print("  Unusual pattern - may indicate training instability.")
    else:
        print("~ Similar variation between overfitted and holdout models")
        print("  This suggests the learned patterns are relatively stable.")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='GRANITE Block Group Diagnostic')
    parser.add_argument('--fips', type=str, default='47065000600',
                        help='Tract FIPS code (default: 47065000600)')
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--output', type=str, default='./output')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    create_diagnostic_plot(
        fips=args.fips,
        data_dir=args.data_dir,
        output_dir=args.output,
        seed=args.seed
    )