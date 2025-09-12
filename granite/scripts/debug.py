"""
Debug R² = 0.000 Issue in GRANITE
Systematic diagnosis of Stage 1 learning problems
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

def diagnose_r_squared_issue(learned_accessibility, traditional_accessibility, verbose=True):
    """
    Comprehensive diagnosis of why R² might be zero
    """
    print("=== GRANITE R² DIAGNOSIS ===\n")
    
    if learned_accessibility is None:
        print("❌ CRITICAL: learned_accessibility is None")
        return
    
    if traditional_accessibility is None:
        print("❌ CRITICAL: traditional_accessibility is None")
        return
    
    print(f"✓ Data loaded successfully")
    print(f"  Learned shape: {learned_accessibility.shape}")
    print(f"  Traditional shape: {traditional_accessibility.shape}")
    
    # Check for shape mismatches
    if len(learned_accessibility.shape) != len(traditional_accessibility.shape):
        print(f"⚠️  WARNING: Shape dimension mismatch")
        print(f"    Learned: {len(learned_accessibility.shape)}D")
        print(f"    Traditional: {len(traditional_accessibility.shape)}D")
    
    # Align shapes for comparison
    min_len = min(len(learned_accessibility), len(traditional_accessibility))
    learned = learned_accessibility[:min_len]
    traditional = traditional_accessibility[:min_len]
    
    print(f"✓ Aligned to {min_len} samples")
    
    # Convert to 1D for correlation if needed
    if len(learned.shape) > 1:
        learned_1d = np.mean(learned, axis=1)
        print(f"  Learned collapsed: {learned.shape} -> {learned_1d.shape}")
    else:
        learned_1d = learned
    
    if len(traditional.shape) > 1:
        traditional_1d = np.mean(traditional, axis=1)
        print(f"  Traditional collapsed: {traditional.shape} -> {traditional_1d.shape}")
    else:
        traditional_1d = traditional
    
    # Basic statistics
    print(f"\n=== BASIC STATISTICS ===")
    print(f"Learned:")
    print(f"  Mean: {np.mean(learned_1d):.6f}")
    print(f"  Std:  {np.std(learned_1d):.6f}")
    print(f"  Min:  {np.min(learned_1d):.6f}")
    print(f"  Max:  {np.max(learned_1d):.6f}")
    print(f"  Range: {np.ptp(learned_1d):.6f}")
    
    print(f"\nTraditional:")
    print(f"  Mean: {np.mean(traditional_1d):.6f}")
    print(f"  Std:  {np.std(traditional_1d):.6f}")
    print(f"  Min:  {np.min(traditional_1d):.6f}")
    print(f"  Max:  {np.max(traditional_1d):.6f}")
    print(f"  Range: {np.ptp(traditional_1d):.6f}")
    
    # Check for constant values
    if np.std(learned_1d) < 1e-10:
        print(f"❌ CRITICAL: Learned features are constant (std = {np.std(learned_1d):.2e})")
        print(f"   This explains R² = 0. GNN is not learning variation.")
        return "constant_learned"
    
    if np.std(traditional_1d) < 1e-10:
        print(f"❌ CRITICAL: Traditional features are constant (std = {np.std(traditional_1d):.2e})")
        return "constant_traditional"
    
    # Check for NaN/Inf
    learned_bad = np.isnan(learned_1d).sum() + np.isinf(learned_1d).sum()
    traditional_bad = np.isnan(traditional_1d).sum() + np.isinf(traditional_1d).sum()
    
    if learned_bad > 0:
        print(f"⚠️  WARNING: {learned_bad} bad values in learned features")
    if traditional_bad > 0:
        print(f"⚠️  WARNING: {traditional_bad} bad values in traditional features")
    
    # Clean data for correlation
    mask = ~(np.isnan(learned_1d) | np.isnan(traditional_1d) | 
             np.isinf(learned_1d) | np.isinf(traditional_1d))
    
    learned_clean = learned_1d[mask]
    traditional_clean = traditional_1d[mask]
    
    print(f"\n=== CORRELATION ANALYSIS ===")
    print(f"Clean samples: {len(learned_clean)}/{len(learned_1d)} ({100*len(learned_clean)/len(learned_1d):.1f}%)")
    
    if len(learned_clean) < 10:
        print(f"❌ CRITICAL: Too few clean samples for correlation")
        return "insufficient_data"
    
    # Calculate correlation
    correlation, p_value = pearsonr(learned_clean, traditional_clean)
    r_squared = correlation**2
    
    print(f"Correlation: {correlation:.6f}")
    print(f"R-squared: {r_squared:.6f}")
    print(f"P-value: {p_value:.6f}")
    
    # Diagnose the correlation result
    if abs(correlation) < 0.01:
        print(f"\n❌ DIAGNOSIS: Near-zero correlation detected")
        print(f"   Possible causes:")
        print(f"   1. GNN learning different patterns than traditional methods")
        print(f"   2. Feature scaling/normalization mismatch")
        print(f"   3. Address subset mismatch between methods")
        print(f"   4. GNN overfitting to noise rather than accessibility")
        
        # Check if data looks random
        if np.corrcoef(learned_clean[:-1], learned_clean[1:])[0,1] < 0.1:
            print(f"   5. Learned features appear random (low autocorrelation)")
            
    elif abs(correlation) < 0.3:
        print(f"\n⚠️  DIAGNOSIS: Weak correlation")
        print(f"   GNN is learning some patterns but not traditional accessibility")
        
    else:
        print(f"\n✓ DIAGNOSIS: Reasonable correlation found")
    
    # Visualize if possible
    if verbose and len(learned_clean) > 0:
        create_diagnostic_plot(learned_clean, traditional_clean, correlation, r_squared)
    
    return {
        'correlation': correlation,
        'r_squared': r_squared,
        'p_value': p_value,
        'n_clean': len(learned_clean),
        'learned_stats': {
            'mean': np.mean(learned_clean),
            'std': np.std(learned_clean),
            'range': np.ptp(learned_clean)
        },
        'traditional_stats': {
            'mean': np.mean(traditional_clean),
            'std': np.std(traditional_clean),
            'range': np.ptp(traditional_clean)
        }
    }

def create_diagnostic_plot(learned, traditional, correlation, r_squared):
    """Create diagnostic visualization"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Scatter plot
    axes[0].scatter(traditional, learned, alpha=0.6, s=20)
    axes[0].set_xlabel('Traditional Accessibility')
    axes[0].set_ylabel('Learned Accessibility')
    axes[0].set_title(f'Correlation: {correlation:.3f}\nR² = {r_squared:.3f}')
    axes[0].grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(traditional, learned, 1)
    p = np.poly1d(z)
    axes[0].plot(traditional, p(traditional), "r--", alpha=0.8)
    
    # Histograms
    axes[1].hist(learned, bins=30, alpha=0.7, label='Learned', density=True)
    axes[1].hist(traditional, bins=30, alpha=0.7, label='Traditional', density=True)
    axes[1].set_xlabel('Accessibility Value')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Distribution Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Residuals
    residuals = learned - traditional
    axes[2].scatter(traditional, residuals, alpha=0.6, s=20)
    axes[2].axhline(y=0, color='r', linestyle='--')
    axes[2].set_xlabel('Traditional Accessibility')
    axes[2].set_ylabel('Residuals')
    axes[2].set_title('Residual Analysis')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./debug_r_squared_diagnostic.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\n✓ Diagnostic plot saved to './debug_r_squared_diagnostic.png'")

def suggest_fixes(diagnosis_result):
    """Suggest specific fixes based on diagnosis"""
    if isinstance(diagnosis_result, str):
        if diagnosis_result == "constant_learned":
            print(f"\n=== SUGGESTED FIXES ===")
            print(f"1. Check GNN training loss - is it actually decreasing meaningfully?")
            print(f"2. Verify accessibility targets are not constant")
            print(f"3. Increase learning rate or reduce regularization")
            print(f"4. Check if model is getting stuck in local minimum")
            
        elif diagnosis_result == "constant_traditional":
            print(f"\n=== SUGGESTED FIXES ===")
            print(f"1. Verify traditional accessibility calculation")
            print(f"2. Check if all addresses have same accessibility")
            print(f"3. Ensure destination variety in calculation")
            
        return
    
    correlation = diagnosis_result['correlation']
    
    print(f"\n=== SUGGESTED FIXES ===")
    
    if abs(correlation) < 0.1:
        print(f"HIGH PRIORITY FIXES:")
        print(f"1. Verify address alignment between learned and traditional")
        print(f"2. Check feature scaling - normalize both before comparison")
        print(f"3. Inspect GNN loss curve - is it learning anything meaningful?")
        print(f"4. Compare accessibility target computation with traditional method")
        print(f"5. Try different GNN architecture or hyperparameters")
        
    elif abs(correlation) < 0.3:
        print(f"MODERATE PRIORITY FIXES:")
        print(f"1. Feature engineering - try different accessibility metrics")
        print(f"2. Regularization tuning in GNN training")
        print(f"3. Verify destination sets match between methods")
        
    print(f"\nDEBUGGING STEPS:")
    print(f"1. Print first 10 values of each array to inspect manually")
    print(f"2. Verify GNN is using same addresses as traditional calculation")
    print(f"3. Check if GNN predictions change across training epochs")
    print(f"4. Try simplest possible GNN (linear layer) as baseline")

# Example usage function
def debug_granite_r_squared(results_dict):
    """
    Main function to debug R² issues in GRANITE results
    
    Args:
        results_dict: Dictionary from GRANITE pipeline containing:
            - learned_accessibility: np.array from Stage 1 GNN
            - traditional_accessibility: np.array from baseline calculation
    """
    learned = results_dict.get('learned_accessibility')
    traditional = results_dict.get('traditional_accessibility')
    
    diagnosis = diagnose_r_squared_issue(learned, traditional, verbose=True)
    suggest_fixes(diagnosis)
    
    return diagnosis