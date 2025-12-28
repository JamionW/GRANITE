"""
GRANITE Fast Ablation Study

Uses Euclidean distances instead of OSRM routing for speed.
Answers the core question in ~5-10 minutes instead of hours.

Core question: Do accessibility features carry any signal for SVI prediction,
or does constraint enforcement do all the work?

Scientific note: Euclidean distances are a proxy for network distances.
If Euclidean accessibility correlates with SVI, network accessibility will too.
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from scipy.spatial import cKDTree
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/workspaces/GRANITE')


def compute_euclidean_accessibility(addresses_gdf, destinations_gdf, dest_type):
    """
    Compute accessibility features using Euclidean distances.
    Much faster than OSRM (~1000x), still captures spatial relationships.
    
    Returns 10 features per destination type.
    """
    # Get coordinates
    addr_coords = np.column_stack([
        addresses_gdf.geometry.x.values,
        addresses_gdf.geometry.y.values
    ])
    
    dest_coords = np.column_stack([
        destinations_gdf.geometry.x.values,
        destinations_gdf.geometry.y.values
    ])
    
    # Build KD-tree for fast nearest neighbor queries
    tree = cKDTree(dest_coords)
    
    n_addresses = len(addr_coords)
    
    # Query distances to k nearest destinations
    k = min(20, len(dest_coords))
    distances, indices = tree.query(addr_coords, k=k)
    
    # Convert degrees to approximate km (at ~35°N latitude)
    km_per_degree = 111 * np.cos(np.radians(35))
    distances_km = distances * km_per_degree
    
    # Compute features
    features = np.zeros((n_addresses, 10))
    
    # Distance-based features
    features[:, 0] = distances_km[:, 0]  # min_distance (nearest)
    features[:, 1] = np.mean(distances_km, axis=1)  # mean_distance
    features[:, 2] = np.median(distances_km, axis=1)  # median_distance
    features[:, 3] = np.std(distances_km, axis=1)  # distance_std
    
    # Count-based features (destinations within threshold)
    features[:, 4] = np.sum(distances_km < 1.0, axis=1)  # count_1km
    features[:, 5] = np.sum(distances_km < 3.0, axis=1)  # count_3km
    features[:, 6] = np.sum(distances_km < 5.0, axis=1)  # count_5km
    features[:, 7] = np.sum(distances_km < 10.0, axis=1)  # count_10km
    
    # Concentration features
    features[:, 8] = distances_km[:, 0] / (distances_km[:, -1] + 0.01)  # concentration_ratio
    features[:, 9] = np.max(distances_km, axis=1) - np.min(distances_km, axis=1)  # distance_range
    
    return features


def load_data_fast():
    """Load spatial data without heavy computation."""
    
    print("Loading spatial data...")
    
    from granite.data.loaders import DataLoader
    
    loader = DataLoader()
    
    # Load tract data
    census_tracts = loader.load_census_tracts('47', '065')
    svi = loader.load_svi_data('47', 'Hamilton')
    tracts = census_tracts.merge(svi, on='FIPS', how='inner')
    
    # Load destinations
    employment = loader.create_employment_destinations(use_real_data=True)
    healthcare = loader.create_healthcare_destinations(use_real_data=True)
    grocery = loader.create_grocery_destinations(use_real_data=True)
    
    print(f"  Tracts: {len(tracts)}")
    print(f"  Employment destinations: {len(employment)}")
    print(f"  Healthcare destinations: {len(healthcare)}")
    print(f"  Grocery destinations: {len(grocery)}")
    
    return {
        'loader': loader,
        'tracts': tracts,
        'employment': employment,
        'healthcare': healthcare,
        'grocery': grocery
    }


def compute_tract_features_fast(addresses, data):
    """Compute all accessibility features using Euclidean distances."""
    
    emp_features = compute_euclidean_accessibility(
        addresses, data['employment'], 'employment'
    )
    health_features = compute_euclidean_accessibility(
        addresses, data['healthcare'], 'healthcare'
    )
    grocery_features = compute_euclidean_accessibility(
        addresses, data['grocery'], 'grocery'
    )
    
    return np.hstack([emp_features, health_features, grocery_features])


def run_fast_ablation(
    results_dir='./output/global_validation',
    output_dir=None,
    n_train_tracts=10,
    epochs=100,
    seed=42
):
    """
    Run fast ablation study using Euclidean distances.
    Expected runtime: 5-10 minutes.
    """
    
    print("\n" + "="*70)
    print("GRANITE FAST ABLATION STUDY")
    print("="*70)
    print("\nUsing Euclidean distances (fast proxy for network distances)")
    print("Question: Do accessibility features carry signal for SVI prediction?")
    print("="*70)
    
    start_total = time.time()
    
    if output_dir is None:
        output_dir = os.path.join(results_dir, 'ablation')
    os.makedirs(output_dir, exist_ok=True)
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load data
    data = load_data_fast()
    tracts = data['tracts']
    loader = data['loader']
    
    # Split tracts
    validation_csv = os.path.join(results_dir, 'validation_results.csv')
    if os.path.exists(validation_csv):
        val_df = pd.read_csv(validation_csv)
        test_fips = [str(f) for f in val_df['fips'].tolist()]
        print(f"\nUsing {len(test_fips)} test tracts from validation results")
    else:
        all_fips = tracts['FIPS'].tolist()
        np.random.shuffle(all_fips)
        test_fips = all_fips[:10]
    
    train_fips = [f for f in tracts['FIPS'].tolist() if f not in test_fips]
    train_fips = train_fips[:n_train_tracts]
    
    print(f"Training tracts: {len(train_fips)}")
    print(f"Test tracts: {len(test_fips)}")
    
    # Compute features for all tracts (fast with Euclidean)
    print("\nComputing Euclidean accessibility features...")
    
    tract_data = {}
    feature_time_start = time.time()
    
    all_fips_needed = train_fips + test_fips
    for i, fips in enumerate(all_fips_needed):
        addresses = loader.get_addresses_for_tract(fips)
        if len(addresses) < 10:
            continue
        
        tract_svi = tracts[tracts['FIPS'] == fips]['RPL_THEMES'].values[0]
        features = compute_tract_features_fast(addresses, data)
        
        tract_data[fips] = {
            'features': features,
            'svi': tract_svi,
            'n_addresses': len(addresses)
        }
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(all_fips_needed)} tracts")
    
    feature_time = time.time() - feature_time_start
    print(f"Feature computation: {feature_time:.1f} seconds")
    
    n_features = 30  # 10 features × 3 destination types
    
    # Run configurations
    configs = {
        'accessibility_only': {
            'use_random': False,
            'description': 'Euclidean accessibility features'
        },
        'random_baseline': {
            'use_random': True,
            'description': 'Random features (should fail)'
        }
    }
    
    results = {}
    
    for config_name, config in configs.items():
        print(f"\n{'='*70}")
        print(f"CONFIGURATION: {config_name}")
        print(f"{'='*70}")
        
        config_start = time.time()
        
        # Prepare training data
        X_train_list = []
        y_train_list = []
        
        for fips in train_fips:
            if fips not in tract_data:
                continue
            
            td = tract_data[fips]
            
            if config['use_random']:
                features = np.random.randn(td['n_addresses'], n_features)
            else:
                features = td['features']
            
            X_train_list.append(features)
            y_train_list.append(np.full(td['n_addresses'], td['svi']))
        
        X_train = np.vstack(X_train_list)
        y_train = np.concatenate(y_train_list)
        
        print(f"Training data: {X_train.shape[0]} addresses, {X_train.shape[1]} features")
        
        # Handle NaN/Inf
        nan_mask = np.isnan(X_train) | np.isinf(X_train)
        if nan_mask.any():
            print(f"  Replacing {nan_mask.sum()} NaN/Inf values")
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=10.0, neginf=-10.0)
        
        # Normalize
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train simple MLP
        model = SimpleMLP(n_features, hidden_dim=32, output_dim=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.MSELoss()
        
        X_tensor = torch.FloatTensor(X_train_scaled)
        y_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        
        print(f"Training for {epochs} epochs...")
        model.train()
        
        batch_size = 256
        n_batches = (len(X_tensor) + batch_size - 1) // batch_size
        
        for epoch in range(epochs):
            epoch_loss = 0
            indices = torch.randperm(len(X_tensor))
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(X_tensor))
                batch_indices = indices[start_idx:end_idx]
                
                X_batch = X_tensor[batch_indices]
                y_batch = y_tensor[batch_indices]
                
                optimizer.zero_grad()
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                
                if torch.isnan(loss):
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if (epoch + 1) % 25 == 0:
                avg_loss = epoch_loss / n_batches
                print(f"  Epoch {epoch+1}: loss = {avg_loss:.4f}")
        
        # Evaluate on test tracts
        print(f"\nEvaluating on test tracts...")
        model.eval()
        
        test_actual = []
        test_predicted = []
        
        with torch.no_grad():
            for fips in test_fips:
                if fips not in tract_data:
                    continue
                
                td = tract_data[fips]
                
                if config['use_random']:
                    features = np.random.randn(td['n_addresses'], n_features)
                else:
                    features = td['features']
                
                features = np.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0)
                features_scaled = scaler.transform(features)
                
                X_test = torch.FloatTensor(features_scaled)
                preds = model(X_test).numpy().flatten()
                
                test_actual.append(td['svi'])
                test_predicted.append(np.mean(preds))
        
        test_actual = np.array(test_actual)
        test_predicted = np.array(test_predicted)
        
        # Compute metrics (RAW - no constraint enforcement)
        if len(test_actual) >= 3:
            correlation = np.corrcoef(test_actual, test_predicted)[0, 1]
            mae = np.mean(np.abs(test_actual - test_predicted))
        else:
            correlation = np.nan
            mae = np.nan
        
        config_time = time.time() - config_start
        
        results[config_name] = {
            'test_correlation': correlation,
            'mae': mae,
            'n_test': len(test_actual),
            'elapsed_seconds': config_time
        }
        
        print(f"\nResults for {config_name}:")
        print(f"  Test correlation (raw): r = {correlation:.3f}")
        print(f"  Mean absolute error: {mae:.3f}")
        print(f"  Time: {config_time:.1f} seconds")
    
    # Generate report
    total_time = time.time() - start_total
    generate_fast_ablation_report(results, output_dir, total_time)
    
    return results


class SimpleMLP(nn.Module):
    """Simple MLP for ablation."""
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)


def generate_fast_ablation_report(results, output_dir, total_time):
    """Generate ablation report with interpretation."""
    
    print("\n" + "="*70)
    print("ABLATION STUDY RESULTS")
    print("="*70)
    
    acc_r = results.get('accessibility_only', {}).get('test_correlation', np.nan)
    rand_r = results.get('random_baseline', {}).get('test_correlation', np.nan)
    
    print(f"\n{'Configuration':<25} {'Correlation':>15} {'MAE':>10}")
    print("-"*55)
    
    for name, res in results.items():
        r = res.get('test_correlation', np.nan)
        mae = res.get('mae', np.nan)
        r_str = f"{r:.3f}" if not np.isnan(r) else "N/A"
        mae_str = f"{mae:.3f}" if not np.isnan(mae) else "N/A"
        print(f"{name:<25} {r_str:>15} {mae_str:>10}")
    
    print("\n" + "-"*70)
    print("INTERPRETATION")
    print("-"*70)
    
    interpretation = "UNKNOWN"
    
    if not np.isnan(acc_r):
        diff = acc_r - rand_r if not np.isnan(rand_r) else acc_r
        
        if acc_r > 0.5:
            interpretation = "STRONG_SIGNAL"
            print(f"\nFINDING: Accessibility features carry STRONG signal (r = {acc_r:.3f})")
            print("The model learns meaningful patterns from accessibility data.")
            print("\n=> Use 'Methodological Contribution' narrative for defense")
            
        elif acc_r > 0.3:
            interpretation = "MODERATE_SIGNAL"
            print(f"\nFINDING: Accessibility features carry MODERATE signal (r = {acc_r:.3f})")
            print("Some learning from accessibility, but limited predictive power.")
            print("\n=> Use 'Methodological Contribution' narrative with caveats")
            
        else:
            interpretation = "WEAK_SIGNAL"
            print(f"\nFINDING: Accessibility features carry WEAK signal (r = {acc_r:.3f})")
            print("Model relies primarily on constraint enforcement.")
            print("\n=> Use 'Valuable Negative Result' narrative for defense")
        
        if not np.isnan(rand_r):
            print(f"\nAccessibility r={acc_r:.3f} vs Random r={rand_r:.3f}")
            if acc_r > rand_r + 0.15:
                print("=> Accessibility features outperform random (features contain signal)")
            elif acc_r > rand_r + 0.05:
                print("=> Accessibility slightly better than random (weak signal)")
            else:
                print("=> Accessibility similar to random (no detectable signal)")
    
    print(f"\nTotal runtime: {total_time/60:.1f} minutes")
    print("="*70)
    
    # Save report
    lines = [
        "="*70,
        "GRANITE FAST ABLATION STUDY REPORT",
        "="*70,
        "",
        "Method: Euclidean distances (fast proxy for network routing)",
        f"Runtime: {total_time/60:.1f} minutes",
        "",
        "-"*70,
        "RESULTS",
        "-"*70,
        "",
        f"{'Configuration':<25} {'Correlation':>15} {'MAE':>10}",
        "-"*55
    ]
    
    for name, res in results.items():
        r = res.get('test_correlation', np.nan)
        mae = res.get('mae', np.nan)
        r_str = f"{r:.3f}" if not np.isnan(r) else "N/A"
        mae_str = f"{mae:.3f}" if not np.isnan(mae) else "N/A"
        lines.append(f"{name:<25} {r_str:>15} {mae_str:>10}")
    
    lines.extend([
        "",
        "-"*70,
        "INTERPRETATION",
        "-"*70,
        "",
        f"Signal strength: {interpretation}",
        f"Accessibility correlation: {acc_r:.3f}" if not np.isnan(acc_r) else "N/A",
        f"Random baseline: {rand_r:.3f}" if not np.isnan(rand_r) else "N/A",
        "",
        "="*70
    ])
    
    report_path = os.path.join(output_dir, 'fast_ablation_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"\nReport saved: {report_path}")
    
    # Save CSV
    results_df = pd.DataFrame([
        {'configuration': name, **res}
        for name, res in results.items()
    ])
    csv_path = os.path.join(output_dir, 'fast_ablation_results.csv')
    results_df.to_csv(csv_path, index=False)
    
    return interpretation


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='GRANITE Fast Ablation Study')
    parser.add_argument('--results-dir', type=str, default='./output/global_validation')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--n-train-tracts', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    run_fast_ablation(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        n_train_tracts=args.n_train_tracts,
        epochs=args.epochs,
        seed=args.seed
    )