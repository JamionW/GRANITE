#!/usr/bin/env python3
"""
Fixed GRANITE Spatial Smoothness Diagnostic
Save this as: granite_diagnostic_fixed.py
Run with: python granite_diagnostic_fixed.py
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import yaml

# Add granite to path
if os.path.exists('./granite'):
    sys.path.insert(0, '.')

def run_granite_diagnostic():
    """Run diagnostic tests on GRANITE framework - Fixed version"""
    
    print("🧪 GRANITE Spatial Smoothness Diagnostic (Fixed)")
    print("=" * 50)
    
    try:
        # Import GRANITE modules with correct function names
        from granite.data.loaders import load_hamilton_county_data
        from granite.models.gnn import (
            create_gnn_model, 
            prepare_graph_data_topological,  # This exists
            prepare_graph_data_with_nlcd,    # This exists (corrected name)
            prepare_graph_data               # This is an alias
        )
        from granite.models.training import train_accessibility_gnn
        from torch_geometric.nn import GCNConv
        
        print("✅ GRANITE modules imported successfully")
        
        # Load config
        config_path = 'config/config.yaml'
        if not os.path.exists(config_path):
            print(f"❌ Config file not found: {config_path}")
            print("Available files:", os.listdir('.'))
            return
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✅ Config loaded")
        
        # Load data
        print("\n📊 Loading data...")
        try:
            data = load_hamilton_county_data(data_dir='./data')
            print(f"✅ Loaded {len(data)} datasets")
        except Exception as e:
            print(f"❌ Data loading failed: {e}")
            return
        
        # Create graph using your actual functions
        print("\n🔗 Creating graph...")
        model_config = config.get('model', {})
        
        try:
            # Try to use the features specified in config
            feature_type = model_config.get('features', {}).get('type', 'topological')
            
            if feature_type == 'nlcd':
                print("   Attempting NLCD features...")
                try:
                    # Get first tract for testing
                    test_tract = data['tracts_with_svi'].iloc[0:1]
                    
                    # Try to create NLCD features (this might fail, which is fine)
                    nlcd_features = None  # You'd need NLCD data here
                    
                    graph_data, _ = prepare_graph_data_with_nlcd(
                        data['road_network'],
                        nlcd_features if nlcd_features is not None else test_tract
                    )
                    feature_type_used = "NLCD"
                except Exception as nlcd_error:
                    print(f"   NLCD failed ({nlcd_error}), falling back to topological")
                    graph_data, _ = prepare_graph_data_topological(data['road_network'])
                    feature_type_used = "Topological (NLCD fallback)"
            else:
                print("   Using topological features...")
                graph_data, _ = prepare_graph_data_topological(data['road_network'])
                feature_type_used = "Topological"
            
            print(f"✅ Graph created with {feature_type_used} features")
            print(f"   Nodes: {graph_data.x.shape[0]}")
            print(f"   Edges: {graph_data.edge_index.shape[1]}")
            print(f"   Features: {graph_data.x.shape[1]}")
            
        except Exception as e:
            print(f"❌ Graph creation failed: {e}")
            print("   Creating minimal fallback graph for testing...")
            
            # Create minimal synthetic graph for testing
            num_nodes = 100
            x = torch.randn(num_nodes, 5)  # 5 features
            edge_list = []
            for i in range(num_nodes - 1):
                edge_list.append([i, i+1])
                edge_list.append([i+1, i])
            
            from torch_geometric.data import Data
            edge_index = torch.LongTensor(edge_list).t().contiguous()
            graph_data = Data(x=x, edge_index=edge_index)
            feature_type_used = "Synthetic (fallback)"
            
            print(f"✅ Created synthetic graph: {num_nodes} nodes")
        
        # Create model
        print("\n🤖 Creating model...")
        model = create_gnn_model(
            input_dim=graph_data.x.shape[1],
            hidden_dim=model_config.get('hidden_dim', 64),
            output_dim=3
        )
        
        print(f"✅ Model created: {graph_data.x.shape[1]} → 64 → 3")
        
        # Quick training for diagnostic
        print("\n🎯 Training model (5 epochs for diagnostic)...")
        try:
            trained_model, features, metrics = train_accessibility_gnn(
                graph_data,
                model=model,
                epochs=5,
                lr=model_config.get('learning_rate', 0.001),
                spatial_weight=model_config.get('spatial_weight', 0.01),
                verbose=False  # Reduced verbosity for cleaner output
            )
            print("✅ Training complete")
        except Exception as e:
            print(f"❌ Training failed: {e}")
            print("   Using untrained model for architectural analysis...")
            trained_model = model
        
        # Run diagnostic tests
        print("\n" + "=" * 60)
        print("🔍 RUNNING DIAGNOSTIC TESTS")
        print("=" * 60)
        
        # TEST 1: Config verification
        print("\n1️⃣ CONFIG VERIFICATION")
        print("-" * 30)
        
        spatial_weight = model_config.get('spatial_weight', 'NOT_FOUND')
        print(f"spatial_weight in config: {spatial_weight}")
        
        if spatial_weight == 'NOT_FOUND':
            print("🚨 SPATIAL_WEIGHT NOT IN CONFIG")
            config_verdict = "NO_SPATIAL_WEIGHT"
        elif spatial_weight == 0:
            print("🚨 SPATIAL_WEIGHT IS ZERO")
            config_verdict = "ZERO_SPATIAL_WEIGHT"
        elif spatial_weight > 0.1:
            print("⚠️  HIGH SPATIAL_WEIGHT (may over-smooth)")
            config_verdict = "HIGH_SPATIAL_WEIGHT"
        else:
            print("✅ Spatial weight reasonable")
            config_verdict = "CONFIG_OK"
        
        # TEST 2: Input feature variation
        print("\n2️⃣ INPUT FEATURE ANALYSIS")
        print("-" * 30)
        
        input_features = graph_data.x.cpu().numpy()
        feature_stds = np.std(input_features, axis=0)
        
        print(f"Feature dimensions: {input_features.shape[1]}")
        print(f"Feature variations: {feature_stds}")
        print(f"Mean variation: {np.mean(feature_stds):.6f}")
        print(f"Min variation:  {np.min(feature_stds):.6f}")
        print(f"Max variation:  {np.max(feature_stds):.6f}")
        
        if np.mean(feature_stds) < 0.001:
            print("🚨 INPUT FEATURES TOO UNIFORM")
            input_verdict = "UNIFORM_FEATURES"
        elif np.min(feature_stds) == 0:
            print("⚠️  Some features are constant")
            input_verdict = "SOME_CONSTANT_FEATURES"
        else:
            print("✅ Input features have reasonable variation")
            input_verdict = "FEATURES_OK"
        
        # TEST 3: Model output variation
        print("\n3️⃣ MODEL OUTPUT ANALYSIS")
        print("-" * 30)
        
        trained_model.eval()
        with torch.no_grad():
            output = trained_model(graph_data.x, graph_data.edge_index)
        
        output_np = output.cpu().numpy()
        
        print(f"Output shape: {output_np.shape}")
        for i, param_name in enumerate(['kappa', 'alpha', 'tau']):
            param_std = np.std(output_np[:, i])
            param_range = np.max(output_np[:, i]) - np.min(output_np[:, i])
            param_mean = np.mean(output_np[:, i])
            print(f"{param_name}: std={param_std:.6f}, range={param_range:.6f}, mean={param_mean:.3f}")
        
        overall_std = np.mean(np.std(output_np, axis=0))
        print(f"Overall spatial std: {overall_std:.6f}")
        
        if overall_std < 0.01:
            print("🚨 OUTPUT SHOWS VERY LOW SPATIAL VARIATION")
            output_verdict = "LOW_VARIATION"
        else:
            print("✅ Output shows reasonable spatial variation")
            output_verdict = "VARIATION_OK"
        
        # TEST 4: Layer-by-layer analysis
        print("\n4️⃣ LAYER-BY-LAYER SMOOTHING ANALYSIS")
        print("-" * 40)
        
        # Hook to capture activations
        activations = {}
        def hook_fn(name):
            def hook(module, input, output):
                activations[name] = output.detach().cpu().numpy()
            return hook
        
        # Register hooks on GNN layers
        hooks = []
        layer_names = []
        for attr_name in ['conv1', 'conv2', 'conv3', 'conv4']:
            if hasattr(trained_model, attr_name):
                layer = getattr(trained_model, attr_name)
                hooks.append(layer.register_forward_hook(hook_fn(attr_name)))
                layer_names.append(attr_name)
        
        # Forward pass to collect activations
        with torch.no_grad():
            final_output = trained_model(graph_data.x, graph_data.edge_index)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Analyze layer variations
        if activations:
            layer_variations = []
            for layer_name in layer_names:
                if layer_name in activations:
                    layer_std = np.std(activations[layer_name], axis=0).mean()
                    layer_variations.append(layer_std)
                    print(f"{layer_name}: spatial_std = {layer_std:.6f}")
            
            # Add final output
            final_std = torch.std(final_output, dim=0).mean().item()
            layer_variations.append(final_std)
            print(f"final_output: spatial_std = {final_std:.6f}")
            
            if len(layer_variations) > 1:
                first_std = layer_variations[0]
                last_std = layer_variations[-1]
                reduction_ratio = last_std / first_std if first_std > 0 else 0
                
                print(f"Variation reduction ratio: {reduction_ratio:.3f}")
                
                # Check if each layer reduces variation
                smoothing_trend = all(layer_variations[i] >= layer_variations[i+1] 
                                    for i in range(len(layer_variations)-1))
                
                if smoothing_trend and reduction_ratio < 0.5:
                    print("🚨 SIGNIFICANT SMOOTHING ACROSS LAYERS")
                    layer_verdict = "LAYER_SMOOTHING"
                else:
                    print("✅ Layers preserve reasonable variation")
                    layer_verdict = "LAYERS_OK"
            else:
                print("⚠️  Insufficient layer data")
                layer_verdict = "INSUFFICIENT_DATA"
        else:
            print("⚠️  No layer activations captured")
            layer_verdict = "NO_ACTIVATIONS"
        
        # TEST 5: No-graph baseline comparison
        print("\n5️⃣ NO-GRAPH MLP BASELINE")
        print("-" * 30)
        
        class SimpleMLPBaseline(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, hidden_dim//2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim//2, output_dim)
                )
            
            def forward(self, x):
                return self.net(x)
        
        mlp = SimpleMLPBaseline(graph_data.x.shape[1], 64, 3)
        
        with torch.no_grad():
            mlp_output = mlp(graph_data.x)
            gnn_output = trained_model(graph_data.x, graph_data.edge_index)
        
        mlp_std = torch.std(mlp_output, dim=0).mean().item()
        gnn_std = torch.std(gnn_output, dim=0).mean().item()
        
        print(f"MLP spatial_std: {mlp_std:.6f}")
        print(f"GNN spatial_std: {gnn_std:.6f}")
        
        if gnn_std > 0:
            ratio = mlp_std / gnn_std
            print(f"Ratio (MLP/GNN): {ratio:.2f}")
            
            if ratio > 5.0:
                print("🚨 GRAPH CONVOLUTION CAUSES MAJOR SMOOTHING")
                baseline_verdict = "GRAPH_SMOOTHING"
            elif ratio > 2.0:
                print("⚠️  Graph convolution shows some smoothing")
                baseline_verdict = "MODERATE_GRAPH_SMOOTHING"
            else:
                print("✅ Graph structure not primary smoothing cause")
                baseline_verdict = "NOT_GRAPH_SMOOTHING"
        else:
            print("⚠️  GNN output has zero variation")
            baseline_verdict = "ZERO_GNN_VARIATION"
        
        # FINAL VERDICT
        print("\n" + "=" * 60)
        print("🎯 FINAL DIAGNOSTIC VERDICT")
        print("=" * 60)
        
        print(f"Config check:       {config_verdict}")
        print(f"Input features:     {input_verdict}")
        print(f"Output variation:   {output_verdict}")
        print(f"Layer smoothing:    {layer_verdict}")
        print(f"Graph vs no-graph:  {baseline_verdict}")
        
        # Count evidence for architectural vs implementation
        architectural_evidence = sum([
            layer_verdict in ["LAYER_SMOOTHING"],
            baseline_verdict in ["GRAPH_SMOOTHING", "MODERATE_GRAPH_SMOOTHING"],
            output_verdict == "LOW_VARIATION" and input_verdict == "FEATURES_OK"
        ])
        
        implementation_evidence = sum([
            config_verdict in ["NO_SPATIAL_WEIGHT", "ZERO_SPATIAL_WEIGHT"],
            input_verdict in ["UNIFORM_FEATURES", "SOME_CONSTANT_FEATURES"],
            output_verdict == "LOW_VARIATION" and input_verdict != "FEATURES_OK"
        ])
        
        print(f"\nEvidence summary:")
        print(f"📊 Architectural issues: {architectural_evidence}/3")
        print(f"🔧 Implementation issues: {implementation_evidence}/3")
        
        if architectural_evidence >= 2:
            print(f"\n🚨 VERDICT: LIKELY ARCHITECTURAL LIMITATION")
            print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(f"Your multi-layer GNN architecture is causing spatial smoothing.")
            print(f"This appears to be a fundamental property of graph convolutions.")
            print(f"\n🎯 IMMEDIATE TESTS TO CONFIRM:")
            print(f"1. Try single-layer GNN (replace 4 layers with 1)")
            print(f"2. Test GraphSAGE instead of GCN layers")
            print(f"3. Add skip connections to bypass smoothing")
            print(f"\n🔬 RESEARCH SIGNIFICANCE:")
            print(f"You may have discovered a fundamental limitation of multi-layer")
            print(f"GNNs for spatial parameter learning - this could be a major finding!")
        elif implementation_evidence >= 2:
            print(f"\n🔧 VERDICT: LIKELY IMPLEMENTATION ISSUE")
            print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(f"Focus on fixing:")
            if config_verdict in ["NO_SPATIAL_WEIGHT", "ZERO_SPATIAL_WEIGHT"]:
                print(f"• Config: spatial_weight parameter")
            if input_verdict in ["UNIFORM_FEATURES"]:
                print(f"• Features: input features lack spatial variation")
        else:
            print(f"\n❓ VERDICT: MIXED EVIDENCE - NEED MORE INVESTIGATION")
            print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(f"Some evidence for both architectural and implementation issues.")
            print(f"Run additional tests to isolate the primary cause.")
        
        print(f"\n✅ Diagnostic complete!")
        
        return {
            'config': config_verdict,
            'input': input_verdict,
            'output': output_verdict,
            'layers': layer_verdict,
            'baseline': baseline_verdict,
            'architectural_evidence': architectural_evidence,
            'implementation_evidence': implementation_evidence
        }
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're in the GRANITE root directory")
        print("Available directories:", [d for d in os.listdir('.') if os.path.isdir(d)])
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = run_granite_diagnostic()
    if results:
        print(f"\n📋 Summary: {results['architectural_evidence']} architectural, {results['implementation_evidence']} implementation issues detected")