
def create_research_summary(results):
    """
    Create clean, research-focused summary of GRANITE results
    """
    print("\n" + "="*60)
    print("🎯 GRANITE RESEARCH RESULTS")
    print("="*60)
    
    # Core metrics
    total_addresses = results.get('summary', {}).get('total_addresses', 0)
    global_correlation = results.get('summary', {}).get('global_correlation', 0)
    
    print(f"📊 Analysis Scale: {total_addresses:,} real address points")
    print(f"🔗 GNN-IDM Correlation: {global_correlation:.3f}")
    
    # Method comparison
    if 'tract_results' in results:
        gnn_stds = []
        idm_stds = []
        
        for tract_result in results['tract_results']:
            if tract_result.get('status') == 'success':
                # Extract spatial variation metrics
                validation = tract_result.get('validation_metrics', {})
                if 'gnn_spatial_std' in validation:
                    gnn_stds.append(validation['gnn_spatial_std'])
                if 'idm_spatial_std' in validation:
                    idm_stds.append(validation['idm_spatial_std'])
        
        if gnn_stds and idm_stds:
            avg_gnn_std = sum(gnn_stds) / len(gnn_stds)
            avg_idm_std = sum(idm_stds) / len(idm_stds)
            variation_ratio = avg_idm_std / avg_gnn_std if avg_gnn_std > 0 else float('inf')
            
            print(f"\n📈 SPATIAL VARIATION COMPARISON:")
            print(f"   GNN (Learned Parameters): σ = {avg_gnn_std:.6f}")
            print(f"   IDM (Fixed Coefficients): σ = {avg_idm_std:.6f}")
            print(f"   Variation Ratio (IDM:GNN): {variation_ratio:.1f}:1")
            
            # Research interpretation
            print(f"\n🔬 RESEARCH FINDING:")
            if variation_ratio > 5:
                print(f"   → Land cover coefficients create {variation_ratio:.1f}x more spatial detail")
                print(f"   → GNN parameters emphasize spatial smoothness")
                print(f"   → Different approaches optimize for different spatial properties")
            else:
                print(f"   → Methods show similar spatial variation patterns")
                print(f"   → Approaches may be converging on similar solutions")
    
    print("\n" + "="*60)

def add_to_pipeline_class():
    """Add the summarizer method to GRANITEPipeline class"""
    return """
    def _create_research_summary(self):
        """Create research-focused summary"""
        create_research_summary(self.results)
    """
