# DIAGNOSTIC SCRIPT: Debug GNN vs IDM Comparison Issues
# Run this to identify what's causing the suspicious similarity

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def diagnose_comparison_issues(gnn_predictions, idm_predictions, tract_svi):
    """
    Comprehensive diagnosis of GNN vs IDM comparison issues
    """
    print("🔍 DIAGNOSTIC ANALYSIS: GNN vs IDM Comparison")
    print("="*60)
    
    # Extract prediction values
    gnn_values = gnn_predictions['mean'].values
    idm_values = idm_predictions['mean'].values
    
    gnn_x = gnn_predictions['x'].values
    gnn_y = gnn_predictions['y'].values
    idm_x = idm_predictions['x'].values  
    idm_y = idm_predictions['y'].values
    
    # 1. DATA ALIGNMENT CHECK
    print("\n1. DATA ALIGNMENT CHECK")
    print("-" * 30)
    print(f"GNN predictions: {len(gnn_values)}")
    print(f"IDM predictions: {len(idm_values)}")
    print(f"Same length: {len(gnn_values) == len(idm_values)}")
    
    # Check if coordinates match
    if len(gnn_x) == len(idm_x):
        coord_diff_x = np.abs(gnn_x - idm_x)
        coord_diff_y = np.abs(gnn_y - idm_y)
        print(f"Max coordinate difference X: {coord_diff_x.max():.8f}")
        print(f"Max coordinate difference Y: {coord_diff_y.max():.8f}")
        print(f"Coordinates aligned: {coord_diff_x.max() < 1e-6 and coord_diff_y.max() < 1e-6}")
    
    # 2. CONSTRAINT SATISFACTION CHECK
    print("\n2. CONSTRAINT SATISFACTION CHECK")
    print("-" * 30)
    gnn_mean = np.mean(gnn_values)
    idm_mean = np.mean(idm_values)
    gnn_error = abs(gnn_mean - tract_svi) / tract_svi
    idm_error = abs(idm_mean - tract_svi) / tract_svi
    
    print(f"Tract SVI target: {tract_svi:.6f}")
    print(f"GNN mean: {gnn_mean:.6f} (error: {gnn_error:.1%})")
    print(f"IDM mean: {idm_mean:.6f} (error: {idm_error:.1%})")
    print(f"Both perfectly constrained: {gnn_error < 0.001 and idm_error < 0.001}")
    
    if gnn_error < 0.001 and idm_error < 0.001:
        print("⚠️  WARNING: Perfect constraint satisfaction detected!")
        print("   This reduces spatial variation and causes similarity")
    
    # 3. SPATIAL VARIATION ANALYSIS
    print("\n3. SPATIAL VARIATION ANALYSIS")
    print("-" * 30)
    gnn_std = np.std(gnn_values)
    idm_std = np.std(idm_values)
    gnn_range = gnn_values.max() - gnn_values.min()
    idm_range = idm_values.max() - idm_values.min()
    
    print(f"GNN spatial std: {gnn_std:.6f}")
    print(f"IDM spatial std: {idm_std:.6f}")
    print(f"Std deviation ratio (IDM/GNN): {idm_std/gnn_std:.1f}")
    print(f"GNN range: {gnn_range:.6f}")
    print(f"IDM range: {idm_range:.6f}")
    
    if gnn_std < 0.01:
        print("⚠️  WARNING: GNN has very low spatial variation!")
        print("   Check GNN spatial smoothness loss weight")
    
    # 4. CORRELATION ANALYSIS
    print("\n4. CORRELATION ANALYSIS")
    print("-" * 30)
    if len(gnn_values) == len(idm_values):
        correlation, p_value = pearsonr(gnn_values, idm_values)
        print(f"Pearson correlation: {correlation:.6f} (p={p_value:.2e})")
        
        if abs(correlation) > 0.8:
            print("⚠️  WARNING: Very high correlation detected!")
            print("   Methods may be too similar or have implementation issues")
        elif abs(correlation) < 0.1:
            print("✓ Good: Low correlation suggests different methods")
        
        # Check for systematic bias
        diff = gnn_values - idm_values
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)
        
        print(f"Mean difference (GNN - IDM): {mean_diff:.6f}")
        print(f"Std of differences: {std_diff:.6f}")
        
        if abs(mean_diff) > 0.01:
            print("⚠️  WARNING: Systematic bias detected!")
            print("   One method consistently higher than other")
    
    # 5. VISUAL PATTERN CHECK
    print("\n5. VISUAL PATTERN CHECK")
    print("-" * 30)
    
    # Check for spatial autocorrelation in differences
    if len(gnn_values) == len(idm_values) and len(gnn_x) == len(idm_x):
        diff = gnn_values - idm_values
        
        # Simple spatial autocorrelation check
        x_trend = np.corrcoef(gnn_x, diff)[0, 1]
        y_trend = np.corrcoef(gnn_y, diff)[0, 1]
        
        print(f"X-coordinate vs difference correlation: {x_trend:.3f}")
        print(f"Y-coordinate vs difference correlation: {y_trend:.3f}")
        
        if abs(x_trend) > 0.3 or abs(y_trend) > 0.3:
            print("⚠️  WARNING: Systematic spatial trend in differences!")
            print("   Suggests coordinate transformation or indexing issue")
    
    # 6. VALUE DISTRIBUTION CHECK  
    print("\n6. VALUE DISTRIBUTION CHECK")
    print("-" * 30)
    print(f"GNN values - Min: {gnn_values.min():.6f}, Max: {gnn_values.max():.6f}")
    print(f"IDM values - Min: {idm_values.min():.6f}, Max: {idm_values.max():.6f}")
    
    # Check for identical values
    n_identical = np.sum(np.abs(gnn_values - idm_values) < 1e-10)
    print(f"Identical values: {n_identical}/{len(gnn_values)}")
    
    if n_identical > len(gnn_values) * 0.1:
        print("⚠️  WARNING: Many identical values detected!")
        print("   Possible data sharing between methods")
    
    # 7. SUMMARY AND RECOMMENDATIONS
    print("\n7. SUMMARY AND RECOMMENDATIONS")
    print("-" * 30)
    
    issues_found = []
    
    if gnn_error < 0.001 and idm_error < 0.001:
        issues_found.append("Perfect constraint satisfaction reducing variation")
    
    if abs(correlation) > 0.8:
        issues_found.append("Suspiciously high correlation between methods")
        
    if gnn_std < 0.01:
        issues_found.append("GNN spatial variation too low")
        
    if abs(x_trend) > 0.3 or abs(y_trend) > 0.3:
        issues_found.append("Systematic spatial bias in differences")
    
    if n_identical > len(gnn_values) * 0.05:
        issues_found.append("Too many identical predictions")
    
    if issues_found:
        print("🚨 ISSUES FOUND:")
        for i, issue in enumerate(issues_found, 1):
            print(f"   {i}. {issue}")
        
        print("\n💡 RECOMMENDED FIXES:")
        if "Perfect constraint satisfaction" in str(issues_found):
            print("   - Relax constraint tolerance from 0.01 to 0.05")
            print("   - Use soft constraint satisfaction")
        
        if "high correlation" in str(issues_found):
            print("   - Check for data sharing between methods")
            print("   - Verify methods are truly independent")
            
        if "spatial variation too low" in str(issues_found):
            print("   - Reduce GNN spatial smoothness loss weight")
            print("   - Increase spatial parameter diversity")
            
        if "spatial bias" in str(issues_found):
            print("   - Check coordinate alignment between methods")
            print("   - Verify same prediction locations")
            
        if "identical predictions" in str(issues_found):
            print("   - Check for accidental data copying")
            print("   - Ensure independent implementations")
    else:
        print("✅ NO MAJOR ISSUES DETECTED")
        print("   Methods appear to be working correctly")
    
    return {
        'gnn_std': gnn_std,
        'idm_std': idm_std,
        'correlation': correlation if len(gnn_values) == len(idm_values) else None,
        'constraint_errors': (gnn_error, idm_error),
        'issues_found': issues_found
    }

def create_diagnostic_plots(gnn_predictions, idm_predictions, tract_svi):
    """
    Create diagnostic plots to visualize the issues
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('DIAGNOSTIC PLOTS: GNN vs IDM Issues', fontsize=16, fontweight='bold')
    
    gnn_values = gnn_predictions['mean'].values
    idm_values = idm_predictions['mean'].values
    gnn_x = gnn_predictions['x'].values
    gnn_y = gnn_predictions['y'].values
    
    # 1. Distribution comparison
    ax1 = axes[0, 0]
    ax1.hist(gnn_values, bins=30, alpha=0.7, label=f'GNN (σ={np.std(gnn_values):.6f})', density=True)
    ax1.hist(idm_values, bins=30, alpha=0.7, label=f'IDM (σ={np.std(idm_values):.6f})', density=True)
    ax1.axvline(tract_svi, color='red', linestyle='--', label=f'Target: {tract_svi:.3f}')
    ax1.set_title('Value Distributions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Scatter plot
    ax2 = axes[0, 1]
    if len(gnn_values) == len(idm_values):
        ax2.scatter(idm_values, gnn_values, alpha=0.6, s=10)
        min_val = min(gnn_values.min(), idm_values.min())
        max_val = max(gnn_values.max(), idm_values.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        correlation = np.corrcoef(gnn_values, idm_values)[0, 1]
        ax2.set_title(f'Method Correlation: {correlation:.3f}')
        ax2.set_xlabel('IDM Predictions')
        ax2.set_ylabel('GNN Predictions')
    ax2.grid(True, alpha=0.3)
    
    # 3. Difference spatial pattern
    ax3 = axes[0, 2]
    if len(gnn_values) == len(idm_values):
        diff = gnn_values - idm_values
        scatter = ax3.scatter(gnn_x, gnn_y, c=diff, cmap='RdBu_r', s=8, alpha=0.8)
        plt.colorbar(scatter, ax=ax3, label='GNN - IDM')
        ax3.set_title('Spatial Difference Pattern')
        ax3.set_xlabel('Longitude')
        ax3.set_ylabel('Latitude')
    
    # 4. Constraint satisfaction check
    ax4 = axes[1, 0]
    methods = ['GNN', 'IDM', 'Target']
    means = [np.mean(gnn_values), np.mean(idm_values), tract_svi]
    colors = ['steelblue', 'orange', 'red']
    bars = ax4.bar(methods, means, color=colors, alpha=0.7)
    ax4.set_title('Mean Value Comparison')
    ax4.set_ylabel('SVI Value')
    
    # Add error annotations
    gnn_error = abs(means[0] - tract_svi) / tract_svi
    idm_error = abs(means[1] - tract_svi) / tract_svi
    ax4.text(0, means[0] + 0.01, f'Err: {gnn_error:.1%}', ha='center')
    ax4.text(1, means[1] + 0.01, f'Err: {idm_error:.1%}', ha='center')
    ax4.grid(True, alpha=0.3)
    
    # 5. Spatial variation comparison
    ax5 = axes[1, 1]
    variations = [np.std(gnn_values), np.std(idm_values)]
    ax5.bar(['GNN', 'IDM'], variations, color=['steelblue', 'orange'], alpha=0.7)
    ax5.set_title('Spatial Variation Comparison')
    ax5.set_ylabel('Standard Deviation')
    ratio = variations[1] / variations[0] if variations[0] > 0 else 0
    ax5.text(0.5, max(variations) * 0.8, f'Ratio: {ratio:.1f}:1', 
            ha='center', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. Residual analysis
    ax6 = axes[1, 2]
    if len(gnn_values) == len(idm_values):
        diff = gnn_values - idm_values
        ax6.scatter(np.mean([gnn_values, idm_values], axis=0), diff, alpha=0.6, s=10)
        ax6.axhline(0, color='red', linestyle='--')
        ax6.set_title('Residual Analysis')
        ax6.set_xlabel('Mean Prediction')
        ax6.set_ylabel('GNN - IDM Difference')
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# Example usage:
# diagnostic_results = diagnose_comparison_issues(gnn_predictions, idm_predictions, tract_svi)
# diagnostic_fig = create_diagnostic_plots(gnn_predictions, idm_predictions, tract_svi)