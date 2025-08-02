#!/usr/bin/env python
"""
GRANITE Logging Cleanup Script
Systematically reduces verbose logging to focus on key research results
"""
import os
import re
import sys
from pathlib import Path

def clean_dataloader_logging(file_path):
    """Remove excessive NLCD diagnostic logging from DataLoader"""
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace verbose NLCD diagnostics with concise summary
    nlcd_diagnostic_pattern = r'self\._log\(f?"🔍 NLCD FILE DIAGNOSTICS:"\).*?self\._log\(f?"  County/raster overlap confirmed"\)'
    replacement = '''if self.verbose:
            self._log(f"NLCD data loaded: {nlcd_path} -> {len(addresses)} addresses")'''
    
    content = re.sub(nlcd_diagnostic_pattern, replacement, content, flags=re.DOTALL)
    
    # Simplify quality reporting
    quality_pattern = r'self\._log\(f?"NLCD Extraction Quality Report:"\).*?self\._log\(f?"    Vulnerability std: \{vuln_std:.4f\}"\)'
    quality_replacement = '''self._log(f"NLCD features extracted: {len(features_df)} addresses, {len(unique_classes)} land cover classes")
        if self.verbose:
            for nlcd_class in sorted(unique_classes):
                count = class_counts.get(nlcd_class, 0)
                percentage = (count / len(features_df)) * 100
                class_name = class_names.get(nlcd_class, f"Unknown({nlcd_class})")
                self._log(f"  {nlcd_class}: {class_name} - {count} addresses ({percentage:.1f}%)")'''
    
    content = re.sub(quality_pattern, quality_replacement, content, flags=re.DOTALL)
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"✓ Cleaned DataLoader logging: {file_path}")

def clean_metricgraph_logging(file_path):
    """Reduce R callback verbosity in MetricGraph interface"""
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Suppress R callback messages during initialization
    r_suppress_pattern = r'import rpy2\.robjects as ro'
    r_suppress_replacement = '''# Suppress R output during MetricGraph operations
import os
os.environ['R_HOME'] = os.environ.get('R_HOME', '')
os.environ['R_LIBS_SITE'] = ''
with open(os.devnull, 'w') as devnull:
    import contextlib
    with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
        import rpy2.robjects as ro'''
    
    content = re.sub(r_suppress_pattern, r_suppress_replacement, content)
    
    # Add method to temporarily suppress R output
    r_context_manager = '''
    @contextlib.contextmanager
    def _suppress_r_output(self):
        """Temporarily suppress R console output"""
        if not self.verbose:
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                    yield
        else:
            yield
'''
    
    # Add context manager before class definition
    content = re.sub(r'class MetricGraphInterface:', f'{r_context_manager}\nclass MetricGraphInterface:', content)
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"✓ Cleaned MetricGraph logging: {file_path}")

def create_results_summarizer():
    """Create a focused results summary function"""
    
    summarizer_code = '''
def create_research_summary(results):
    """
    Create clean, research-focused summary of GRANITE results
    """
    print("\\n" + "="*60)
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
            
            print(f"\\n📈 SPATIAL VARIATION COMPARISON:")
            print(f"   GNN (Learned Parameters): σ = {avg_gnn_std:.6f}")
            print(f"   IDM (Fixed Coefficients): σ = {avg_idm_std:.6f}")
            print(f"   Variation Ratio (IDM:GNN): {variation_ratio:.1f}:1")
            
            # Research interpretation
            print(f"\\n🔬 RESEARCH FINDING:")
            if variation_ratio > 5:
                print(f"   → Land cover coefficients create {variation_ratio:.1f}x more spatial detail")
                print(f"   → GNN parameters emphasize spatial smoothness")
                print(f"   → Different approaches optimize for different spatial properties")
            else:
                print(f"   → Methods show similar spatial variation patterns")
                print(f"   → Approaches may be converging on similar solutions")
    
    print("\\n" + "="*60)

def add_to_pipeline_class():
    """Add the summarizer method to GRANITEPipeline class"""
    return """
    def _create_research_summary(self):
        \"\"\"Create research-focused summary\"\"\"
        create_research_summary(self.results)
    """
'''
    
    with open('granite_research_summary.py', 'w') as f:
        f.write(summarizer_code)
    
    print("✓ Created research summary module: granite_research_summary.py")

def main():
    """Run logging cleanup on GRANITE codebase"""
    
    print("🧹 GRANITE Logging Cleanup Starting...")
    
    # Find GRANITE source files
    granite_dir = Path('granite')
    if not granite_dir.exists():
        print("❌ GRANITE directory not found. Run from project root.")
        sys.exit(1)
    
    # Clean DataLoader
    dataloader_path = granite_dir / 'data' / 'loaders.py'
    if dataloader_path.exists():
        clean_dataloader_logging(str(dataloader_path))
    
    # Clean MetricGraph interface
    metricgraph_path = granite_dir / 'metricgraph' / 'interface.py'
    if metricgraph_path.exists():
        clean_metricgraph_logging(str(metricgraph_path))
    
    # Create research summarizer
    create_results_summarizer()
    
    # Update config to ensure verbose=False is respected
    config_path = Path('config') / 'config.yaml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config_content = f.read()
        
        # Ensure verbose is clearly set to False
        if 'verbose: false' not in config_content:
            config_content = re.sub(r'verbose:\s*true', 'verbose: false', config_content)
            config_content = re.sub(r'verbose:\s*True', 'verbose: false', config_content)
            
            with open(config_path, 'w') as f:
                f.write(config_content)
            
            print(f"✓ Updated config verbose setting: {config_path}")
    
    print("\\n✅ GRANITE Logging Cleanup Complete!")
    print("\\nNext steps:")
    print("1. Update pipeline.py __init__ method with verbose fix")
    print("2. Import granite_research_summary in your pipeline")
    print("3. Call self._create_research_summary() at end of run()")

if __name__ == "__main__":
    main()