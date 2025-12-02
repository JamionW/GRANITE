"""
Feature Encoding Diagnostic: Test if accessibility features correctly predict vulnerability
Creates synthetic addresses with known accessibility-vulnerability relationships
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

class FeatureEncodingDiagnostic:
    """
    Test if accessibility features encode the transportation-vulnerability relationship correctly
    """
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.test_results = {}
    
    def log(self, message):
        if self.verbose:
            print(f"[FeatureDiagnostic] {message}")
    
    def create_synthetic_test_data(self, n_addresses=500) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        Create synthetic addresses and destinations with known spatial patterns
        
        Pattern: Addresses in center have good access (low SVI)
                 Addresses on periphery have poor access (high SVI)
        """
        self.log(f"Creating {n_addresses} synthetic addresses with known patterns...")
        
        # Create addresses in a grid pattern
        center = (-85.3, 35.05)
        grid_size = int(np.sqrt(n_addresses))
        
        addresses_data = []
        for i in range(grid_size):
            for j in range(grid_size):
                # Position in grid
                x = center[0] + (i - grid_size/2) * 0.01  # longitude
                y = center[1] + (j - grid_size/2) * 0.01  # latitude
                
                # Distance from center
                dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
                
                # TRUE RELATIONSHIP: farther from center = worse accessibility = higher SVI
                true_svi = 0.2 + (dist_from_center / 0.05) * 0.6  # Maps distance to SVI 0.2-0.8
                true_svi = np.clip(true_svi, 0.0, 1.0)
                
                addresses_data.append({
                    'address_id': len(addresses_data),
                    'geometry': Point(x, y),
                    'true_svi': true_svi,
                    'distance_from_center': dist_from_center
                })
        
        addresses_gdf = gpd.GeoDataFrame(addresses_data, crs='EPSG:4326')
        
        # Create destinations (clustered near center)
        destinations_data = []
        for i in range(15):
            # Destinations clustered near center with some noise
            angle = np.random.uniform(0, 2*np.pi)
            radius = np.random.uniform(0.002, 0.015)  # Closer to center
            
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            
            destinations_data.append({
                'dest_id': i,
                'dest_type': ['employment', 'healthcare', 'grocery'][i % 3],
                'geometry': Point(x, y)
            })
        
        destinations_gdf = gpd.GeoDataFrame(destinations_data, crs='EPSG:4326')
        
        self.log(f"Created {len(addresses_gdf)} addresses and {len(destinations_gdf)} destinations")
        
        return addresses_gdf, destinations_gdf
    
    def compute_features_from_synthetic_data(self, addresses: gpd.GeoDataFrame, 
                                            destinations: gpd.GeoDataFrame) -> np.ndarray:
        """
        Use your ACTUAL feature computation pipeline on synthetic data
        """
        self.log("Computing features using GRANITE pipeline...")
        
        # Import your actual feature computation
        from ..data.enhanced_accessibility import EnhancedAccessibilityComputer
        
        computer = EnhancedAccessibilityComputer(verbose=False)
        
        # Group destinations by type
        dest_dict = {
            'employment': destinations[destinations['dest_type'] == 'employment'],
            'healthcare': destinations[destinations['dest_type'] == 'healthcare'],
            'grocery': destinations[destinations['dest_type'] == 'grocery']
        }
        
        all_features = []
        
        for dest_type, dest_gdf in dest_dict.items():
            if len(dest_gdf) == 0:
                continue
            
            # Calculate travel times using your actual pipeline
            travel_times = computer.calculate_realistic_travel_times(
                origins=addresses,
                destinations=dest_gdf,
                time_period='morning'
            )
            
            # Extract features using your actual pipeline
            features = computer.extract_enhanced_accessibility_features(
                addresses=addresses,
                travel_times=travel_times,
                dest_type=dest_type
            )
            
            all_features.append(features)
        
        # Combine features
        feature_matrix = np.column_stack(all_features)
        
        # Add derived features
        derived = computer.compute_enhanced_derived_features(feature_matrix)
        final_features = np.column_stack([feature_matrix, derived])
        
        self.log(f"Computed {final_features.shape[1]} features for {final_features.shape[0]} addresses")
        
        return final_features
    
    def test_feature_correlation_directions(self, features: np.ndarray, 
                                           true_svi: np.ndarray,
                                           feature_names: list = None) -> Dict:
        """
        Test if each feature correlates with true SVI in the expected direction
        """
        self.log("Testing feature correlation directions against known ground truth...")
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(features.shape[1])]
        
        results = {
            'correct': [],
            'incorrect': [],
            'weak': [],
            'correlations': {}
        }
        
        for i, name in enumerate(feature_names):
            if i >= features.shape[1]:
                break
            
            values = features[:, i]
            
            # Skip zero-variance features
            if np.std(values) < 1e-8:
                results['weak'].append(name)
                continue
            
            correlation = np.corrcoef(values, true_svi)[0, 1]
            results['correlations'][name] = correlation
            
            # Determine expected direction
            if any(keyword in name.lower() for keyword in ['time', 'drive_advantage', 'range']):
                expected_positive = True
                expected_sign = 'positive'
            elif any(keyword in name.lower() for keyword in ['count', 'percentile', 'accessibility', 'equity', 'advantage']):
                expected_positive = False
                expected_sign = 'negative'
            else:
                continue  # Unknown feature type
            
            # Check if correlation matches expectation
            is_correct = (correlation > 0.1 if expected_positive else correlation < -0.1)
            
            status = {
                'name': name,
                'correlation': correlation,
                'expected_sign': expected_sign,
                'is_correct': is_correct,
                'strength': 'strong' if abs(correlation) > 0.5 else 'moderate' if abs(correlation) > 0.25 else 'weak'
            }
            
            if is_correct:
                results['correct'].append(status)
            else:
                results['incorrect'].append(status)
        
        return results
    
    def run_complete_diagnostic(self, n_addresses=500) -> Dict:
        """
        Run complete diagnostic test
        """
        self.log("="*60)
        self.log("FEATURE ENCODING DIAGNOSTIC TEST")
        self.log("="*60)
        
        # Step 1: Create synthetic data with known relationships
        addresses, destinations = self.create_synthetic_test_data(n_addresses)
        true_svi = addresses['true_svi'].values
        
        # Step 2: Compute features using actual pipeline
        try:
            features = self.compute_features_from_synthetic_data(addresses, destinations)
        except Exception as e:
            self.log(f"ERROR: Feature computation failed: {str(e)}")
            return {'success': False, 'error': str(e)}
        
        # Step 3: Test correlation directions
        feature_names = self._generate_feature_names(features.shape[1])
        direction_results = self.test_feature_correlation_directions(features, true_svi, feature_names)
        
        # Step 4: Overall assessment
        n_correct = len(direction_results['correct'])
        n_incorrect = len(direction_results['incorrect'])
        n_total = n_correct + n_incorrect
        
        correctness_rate = n_correct / n_total if n_total > 0 else 0
        
        # Step 5: Print results
        self._print_diagnostic_report(direction_results, correctness_rate)
        
        # Step 6: Create visualizations
        self._create_diagnostic_plots(addresses, features, true_svi, direction_results)
        
        # Final verdict
        if correctness_rate >= 0.8:
            verdict = "PASS"
            message = "Feature encoding is correct. Issues likely in data quality or model architecture."
        elif correctness_rate >= 0.6:
            verdict = "MARGINAL"
            message = "Feature encoding partially correct. Review incorrect features."
        else:
            verdict = "FAIL"
            message = "Feature encoding is fundamentally broken. Fix feature computation before proceeding."
        
        self.log("="*60)
        self.log(f"DIAGNOSTIC VERDICT: {verdict}")
        self.log(f"Correctness Rate: {correctness_rate:.1%}")
        self.log(f"Assessment: {message}")
        self.log("="*60)
        
        return {
            'success': True,
            'verdict': verdict,
            'correctness_rate': correctness_rate,
            'direction_results': direction_results,
            'n_correct': n_correct,
            'n_incorrect': n_incorrect,
            'message': message
        }
    
    def _print_diagnostic_report(self, results: Dict, correctness_rate: float):
        """Print detailed diagnostic report"""
        
        self.log("\n" + "="*60)
        self.log("FEATURE DIRECTION TEST RESULTS")
        self.log("="*60)
        
        # Correct features
        if results['correct']:
            self.log(f"\nCORRECT FEATURES ({len(results['correct'])}):")
            for feat in sorted(results['correct'], key=lambda x: abs(x['correlation']), reverse=True)[:10]:
                self.log(f"  ✓ {feat['name'][:30]:30} r={feat['correlation']:6.3f} "
                        f"(expected {feat['expected_sign']}, {feat['strength']})")
        
        # Incorrect features
        if results['incorrect']:
            self.log(f"\nINCORRECT FEATURES ({len(results['incorrect'])}):")
            for feat in results['incorrect']:
                self.log(f"  ✗ {feat['name'][:30]:30} r={feat['correlation']:6.3f} "
                        f"(expected {feat['expected_sign']}, got {'positive' if feat['correlation'] > 0 else 'negative'})")
        
        # Summary
        self.log(f"\n{'='*60}")
        self.log(f"OVERALL CORRECTNESS: {correctness_rate:.1%}")
        self.log(f"Correct: {len(results['correct'])}")
        self.log(f"Incorrect: {len(results['incorrect'])}")
        self.log(f"Weak/Skipped: {len(results['weak'])}")
    
    def _create_diagnostic_plots(self, addresses: gpd.GeoDataFrame, 
                                 features: np.ndarray,
                                 true_svi: np.ndarray,
                                 results: Dict):
        """Create diagnostic visualization"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Feature Encoding Diagnostic Results', fontsize=16, fontweight='bold')
        
        # 1. True SVI spatial pattern
        ax1 = axes[0, 0]
        scatter1 = ax1.scatter(addresses.geometry.x, addresses.geometry.y, 
                              c=true_svi, cmap='RdYlGn_r', s=20, alpha=0.7)
        ax1.set_title('True SVI Pattern\n(Ground Truth)')
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.set_aspect('equal')
        plt.colorbar(scatter1, ax=ax1)
        
        # 2. Feature correlation with true SVI
        ax2 = axes[0, 1]
        
        correlations = list(results['correlations'].values())
        names = list(results['correlations'].keys())
        
        # Color by correctness
        colors = []
        for name in names:
            if any(f['name'] == name for f in results['correct']):
                colors.append('green')
            elif any(f['name'] == name for f in results['incorrect']):
                colors.append('red')
            else:
                colors.append('gray')
        
        y_pos = np.arange(len(names))
        ax2.barh(y_pos, correlations, color=colors, alpha=0.7)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([n[:20] for n in names], fontsize=8)
        ax2.set_xlabel('Correlation with True SVI')
        ax2.set_title('Feature Correlations\n(Green=Correct, Red=Wrong)')
        ax2.axvline(0, color='black', linewidth=0.5)
        ax2.grid(axis='x', alpha=0.3)
        
        # 3. Example feature scatter (strongest correct feature)
        ax3 = axes[1, 0]
        
        if results['correct']:
            strongest_correct = max(results['correct'], key=lambda x: abs(x['correlation']))
            feature_idx = list(results['correlations'].keys()).index(strongest_correct['name'])
            
            feature_values = features[:, feature_idx]
            ax3.scatter(feature_values, true_svi, alpha=0.5, s=20, color='green')
            
            # Regression line
            z = np.polyfit(feature_values, true_svi, 1)
            p = np.poly1d(z)
            ax3.plot(sorted(feature_values), p(sorted(feature_values)), "r--", alpha=0.8)
            
            ax3.set_xlabel(strongest_correct['name'][:30])
            ax3.set_ylabel('True SVI')
            ax3.set_title(f'Strongest Correct Feature\nr={strongest_correct["correlation"]:.3f}')
            ax3.grid(True, alpha=0.3)
        
        # 4. Summary statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        n_correct = len(results['correct'])
        n_incorrect = len(results['incorrect'])
        n_total = n_correct + n_incorrect
        correctness_rate = n_correct / n_total if n_total > 0 else 0
        
        summary_text = f"""DIAGNOSTIC SUMMARY

Test Dataset:
- Addresses: {len(addresses)}
- Known Pattern: Center (good access) → Edge (poor access)
- True SVI Range: [{true_svi.min():.2f}, {true_svi.max():.2f}]

Feature Test Results:
- Total Features Tested: {n_total}
- Correct Direction: {n_correct} ({n_correct/n_total*100:.0f}%)
- Wrong Direction: {n_incorrect} ({n_incorrect/n_total*100:.0f}%)
- Weak/Skipped: {len(results['weak'])}

Verdict:
{'✓ PASS - Features encode correctly' if correctness_rate >= 0.8 else
 '⚠ MARGINAL - Some issues found' if correctness_rate >= 0.6 else
 '✗ FAIL - Major encoding problems'}

Recommendation:
{'Proceed to test real data' if correctness_rate >= 0.8 else
 'Fix incorrect features before proceeding' if correctness_rate >= 0.6 else
 'Fundamental feature computation issues - fix before using real data'}
"""
        
        verdict_color = 'lightgreen' if correctness_rate >= 0.8 else 'lightyellow' if correctness_rate >= 0.6 else 'lightcoral'
        
        ax4.text(0.05, 0.95, summary_text.strip(), transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=verdict_color, alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('./output/feature_encoding_diagnostic.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.log("Diagnostic visualization saved to ./output/feature_encoding_diagnostic.png")
    
    def _generate_feature_names(self, n_features):
        """Generate feature names matching GRANITE structure"""
        base_features = []
        for dest_type in ['employment', 'healthcare', 'grocery']:
            base_features.extend([
                f'{dest_type}_min_time',
                f'{dest_type}_mean_time',
                f'{dest_type}_median_time',
                f'{dest_type}_count_5min',
                f'{dest_type}_count_10min',
                f'{dest_type}_count_15min',
                f'{dest_type}_drive_advantage',
                f'{dest_type}_concentration',
                f'{dest_type}_time_range',
                f'{dest_type}_percentile'
            ])
        
        derived_features = [
            'local_accessibility_index',
            'modal_flexibility',
            'accessibility_equity',
            'geographic_advantage'
        ]
        
        all_features = base_features + derived_features
        return all_features[:n_features]


# Convenience function
def run_feature_encoding_diagnostic(n_addresses=500):
    """
    Run the diagnostic test
    
    Returns: Dictionary with test results including verdict
    """
    diagnostic = FeatureEncodingDiagnostic(verbose=True)
    results = diagnostic.run_complete_diagnostic(n_addresses=n_addresses)
    
    return results


if __name__ == "__main__":
    results = run_feature_encoding_diagnostic(n_addresses=500)