# GIN Integration Audit

Date: 2026-04-19

---

## 1. Current class structure

### `AccessibilitySVIGNN` (GCN-GAT hybrid, default)

Defined in `granite/models/gnn.py:36`. Convolution stack:

```
spatial_conv1:    GCNConv(hidden_dim, hidden_dim)
attention_conv:   GATConv(hidden_dim, hidden_dim//2, heads=2, concat=True)
spatial_conv2:    GCNConv(hidden_dim, hidden_dim//2)
```

Three layers, mixed aggregation. `attention_conv` output feeds directly into `spatial_conv2`; no residual skip. Final representation dimension is `hidden_dim//2`.

### `GraphSAGEAccessibilitySVIGNN`

Defined in `granite/models/gnn.py:214`. Convolution stack:

```
spatial_conv1:    SAGEConv(hidden_dim, hidden_dim)
spatial_conv2:    SAGEConv(hidden_dim, hidden_dim)
spatial_conv3:    SAGEConv(hidden_dim, hidden_dim//2)
```

Three homogeneous SAGEConv layers. Dropout between layers 1-2 and 2-3. Final dimension is `hidden_dim//2`, identical to GCN-GAT.

### Shared components (both classes)

- `ContextGatedFeatureModulator`: softmax attention over accessibility features conditioned on context (socioeconomic) features. Shared submodule, instantiated if `use_context_gating=True`.
- `feature_encoder`: two-layer MLP before any graph convolution.
- `accessibility_learner`, `svi_predictor`, optional multi-task heads: identical in both classes, operating on the `hidden_dim//2` representation.
- `forward` signature: `(accessibility_features, edge_index, context_features=None, return_accessibility=False, return_all_tasks=False)` -- identical in both.

---

## 2. Aggregation primitives and torch_geometric dependencies

| Layer | Class | Aggregation | torch_geometric symbol |
|-------|-------|-------------|------------------------|
| GCNConv | GCN-GAT | symmetric normalized sum | `torch_geometric.nn.GCNConv` |
| GATConv | GCN-GAT | attention-weighted sum | `torch_geometric.nn.GATConv` |
| SAGEConv | GraphSAGE | mean concat | `torch_geometric.nn.SAGEConv` |
| BatchNorm | both | -- | `torch_geometric.nn.BatchNorm` |

All four symbols are imported at the top of `gnn.py:10`:
```python
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, BatchNorm
```

**GINConv availability:** `torch_geometric.nn.GINConv` is present in the installed version (confirmed via import test). GINConv wraps an arbitrary MLP as the aggregation function (`nn(h_v + sum(h_u))`) and requires no additional torch_geometric data fields beyond `edge_index`.

---

## 3. Architecture dispatch in the pipeline

The dispatch pattern is identical at four call sites in `granite/disaggregation/pipeline.py`:

```python
arch = self.config.get('model', {}).get('architecture', 'gcn_gat')
ModelClass = GraphSAGEAccessibilitySVIGNN if arch == 'sage' else AccessibilitySVIGNN
```

Lines: 684-685, 2018-2019, 2354-2355, 4400-4401.

The `arch` value is a config string. Default is `'gcn_gat'`; the only alternative currently dispatched is `'sage'`. All other values fall through to `AccessibilitySVIGNN`.

The trainers (`AccessibilityGNNTrainer`, `MultiTractGNNTrainer`) call the model exclusively through its `forward` signature and never inspect the model's internal layer types. There are no hard-coded layer type checks in trainer logic.

---

## 4. Diff-style plan

### 4a. Add GIN as a third architecture option

**File: `granite/models/gnn.py`**

```diff
-from torch_geometric.nn import GCNConv, GATConv, SAGEConv, BatchNorm
+from torch_geometric.nn import GCNConv, GATConv, SAGEConv, BatchNorm, GINConv
```

Add class `GINAccessibilitySVIGNN` after `GraphSAGEAccessibilitySVIGNN` (~line 360):

```diff
+class GINAccessibilitySVIGNN(nn.Module):
+    """
+    GIN variant of AccessibilitySVIGNN.
+
+    Replaces the GCN+GAT+GCN stack with three GINConv layers. GIN uses
+    sum aggregation with a learned MLP, giving it strictly greater
+    expressive power than GCN or mean-aggregation SAGE (Xu et al., 2019).
+    All other components are unchanged.
+    """
+    def __init__(self, accessibility_features_dim, context_features_dim=5,
+                 hidden_dim=64, dropout=0.3, seed=42, use_context_gating=True,
+                 use_multitask=True):
+        super(GINAccessibilitySVIGNN, self).__init__()
+        set_random_seed(seed)
+        self.accessibility_features_dim = accessibility_features_dim
+        self.context_features_dim = context_features_dim
+        self.hidden_dim = hidden_dim
+        self.dropout_rate = dropout
+        self.use_context_gating = use_context_gating
+        if use_context_gating:
+            self.context_gate = ContextGatedFeatureModulator(
+                accessibility_dim=accessibility_features_dim,
+                context_dim=context_features_dim,
+                hidden_dim=32
+            )
+        self.use_multitask = use_multitask
+        self.input_norm = nn.LayerNorm(accessibility_features_dim)
+        self.feature_encoder = nn.Sequential(
+            nn.Linear(accessibility_features_dim, hidden_dim),
+            nn.ReLU(),
+            nn.Dropout(dropout * 0.5),
+            nn.Linear(hidden_dim, hidden_dim),
+            nn.ReLU(),
+        )
+        # GINConv requires an explicit MLP per layer
+        gin_mlp1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
+                                  nn.Linear(hidden_dim, hidden_dim))
+        gin_mlp2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
+                                  nn.Linear(hidden_dim, hidden_dim))
+        gin_mlp3 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
+                                  nn.Linear(hidden_dim // 2, hidden_dim // 2))
+        self.spatial_conv1 = GINConv(gin_mlp1)
+        self.spatial_norm1 = BatchNorm(hidden_dim)
+        self.spatial_conv2 = GINConv(gin_mlp2)
+        self.spatial_norm2 = BatchNorm(hidden_dim)
+        self.spatial_conv3 = GINConv(gin_mlp3)
+        self.spatial_norm3 = BatchNorm(hidden_dim // 2)
+        # downstream heads identical to GraphSAGE variant
+        self.accessibility_learner = nn.Sequential(
+            nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.ReLU(),
+            nn.Dropout(dropout * 0.5),
+            nn.Linear(hidden_dim // 4, accessibility_features_dim), nn.ReLU()
+        )
+        self.svi_predictor = nn.Sequential(
+            nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.ReLU(),
+            nn.Dropout(dropout), nn.Linear(hidden_dim // 4, 1)
+        )
+        if use_multitask:
+            self.accessibility_classifier = nn.Sequential(
+                nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.ReLU(),
+                nn.Dropout(dropout * 0.5), nn.Linear(hidden_dim // 4, 5)
+            )
+            self.vehicle_predictor = nn.Sequential(
+                nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.ReLU(),
+                nn.Dropout(dropout * 0.5), nn.Linear(hidden_dim // 4, 1)
+            )
+            self.employment_classifier = nn.Sequential(
+                nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.ReLU(),
+                nn.Dropout(dropout * 0.5), nn.Linear(hidden_dim // 4, 3)
+            )
+        self._initialize_weights(seed)
+        self.dropout = nn.Dropout(dropout)
+
+    def _initialize_weights(self, seed):
+        torch.manual_seed(seed)
+        for module in self.modules():
+            if isinstance(module, nn.Linear):
+                nn.init.xavier_normal_(module.weight, gain=1.0)
+                if module.bias is not None:
+                    nn.init.constant_(module.bias, 0)
+
+    def forward(self, accessibility_features, edge_index, context_features=None,
+                return_accessibility=False, return_all_tasks=False):
+        attention_weights = None
+        if self.use_context_gating and context_features is not None:
+            modulated_features, attention_weights = self.context_gate(
+                accessibility_features, context_features
+            )
+            x = self.input_norm(modulated_features)
+        else:
+            x = self.input_norm(accessibility_features)
+        x = self.feature_encoder(x)
+        x = self.spatial_conv1(x, edge_index)
+        x = self.spatial_norm1(x)
+        x = F.relu(x)
+        x = self.dropout(x)
+        x = self.spatial_conv2(x, edge_index)
+        x = self.spatial_norm2(x)
+        x = F.relu(x)
+        x = self.dropout(x)
+        x = self.spatial_conv3(x, edge_index)
+        x = self.spatial_norm3(x)
+        x = F.relu(x)
+        learned_accessibility = self.accessibility_learner(x)
+        svi_predictions = self.svi_predictor(x)
+        svi_predictions = torch.sigmoid(svi_predictions.squeeze())
+        if not self.use_multitask or not return_all_tasks:
+            if return_accessibility:
+                return svi_predictions, learned_accessibility, attention_weights
+            return svi_predictions
+        return {
+            'svi': svi_predictions,
+            'accessibility_quintile_logits': self.accessibility_classifier(x),
+            'vehicle_ownership': torch.sigmoid(self.vehicle_predictor(x).squeeze()),
+            'employment_category_logits': self.employment_classifier(x),
+            'learned_accessibility': learned_accessibility,
+            'attention_weights': attention_weights,
+            'embeddings': x
+        }
```

**File: `granite/disaggregation/pipeline.py`** (4 call sites, identical change each):

```diff
-from ..models.gnn import AccessibilitySVIGNN, GraphSAGEAccessibilitySVIGNN, ...
+from ..models.gnn import AccessibilitySVIGNN, GraphSAGEAccessibilitySVIGNN, GINAccessibilitySVIGNN, ...

 arch = self.config.get('model', {}).get('architecture', 'gcn_gat')
-ModelClass = GraphSAGEAccessibilitySVIGNN if arch == 'sage' else AccessibilitySVIGNN
+if arch == 'sage':
+    ModelClass = GraphSAGEAccessibilitySVIGNN
+elif arch == 'gin':
+    ModelClass = GINAccessibilitySVIGNN
+else:
+    ModelClass = AccessibilitySVIGNN
```

### 4b. Decouple standalone GCN from the GCN-GAT hybrid

`AccessibilitySVIGNN` is a GCN-GAT hybrid, not a pure GCN. Decoupling options:

**Option 1 -- rename the hybrid, add a pure GCN class (recommended).**

Add `GCNAccessibilitySVIGNN` alongside the existing hybrid, replacing the three-layer GCN+GAT+GCN stack with three pure `GCNConv` layers (same structure as the SAGE variant but with GCNConv). The existing `AccessibilitySVIGNN` remains untouched. Pipeline dispatch gains a fourth branch: `arch == 'gcn'`.

```diff
 arch = self.config.get('model', {}).get('architecture', 'gcn_gat')
 if arch == 'sage':
     ModelClass = GraphSAGEAccessibilitySVIGNN
 elif arch == 'gin':
     ModelClass = GINAccessibilitySVIGNN
+elif arch == 'gcn':
+    ModelClass = GCNAccessibilitySVIGNN
 else:  # 'gcn_gat' default
     ModelClass = AccessibilitySVIGNN
```

**Option 2 -- refactor the hybrid to accept a `conv_type` parameter.** Parameterize the layer choices inside `AccessibilitySVIGNN.__init__`. Avoids class proliferation but makes the class harder to read and conflicts with the convention of modifying functions in place rather than adding configurability.

Option 1 is preferred. No refactoring of the hybrid is required; the GCN path is a clean addition.

---

## 5. What GINConv requires that the current pipeline does not already produce

GINConv's signature is `GINConv(nn, eps=0, train_eps=False)`. It expects:
- `x`: node feature tensor -- already produced by the feature pipeline.
- `edge_index`: COO edge index tensor -- already constructed by the graph builder.

GINConv does **not** require edge weights, edge attributes, or any data field beyond `(x, edge_index)`. The current pipeline already produces both. No feature pipeline changes are needed.

The GIN MLPs add learnable parameters. With `hidden_dim=64`:
- `gin_mlp1`: 64x64 + 64 + 64x64 + 64 = ~8,320 params
- `gin_mlp2`: same, ~8,320 params
- `gin_mlp3`: 64x32 + 32 + 32x32 + 32 = ~3,168 params
- Total additional GIN-specific params: ~19,808 vs. GCNConv which uses ~4,224 (3 layers).

This is manageable given typical tract sizes (500-3,000 addresses).

---

## 6. GCN-GAT decoupling analysis

The hybrid's `attention_conv` (GATConv) does not output to an `attention_weights` tensor that the rest of the pipeline depends on -- `attention_weights` in the return dict comes from `ContextGatedFeatureModulator`, not `GATConv`. Removing the GAT layer from a pure-GCN variant requires no downstream changes. The `AccessibilityGNNTrainer` and `MultiTractGNNTrainer` never inspect `GATConv` outputs directly.

Introducing a clean `GCNAccessibilitySVIGNN` class (Option 1) therefore requires no refactoring of the hybrid and no changes to trainer logic.

---

## 7. Estimated scope

| Change | Files touched | Lines changed (approx) |
|--------|---------------|------------------------|
| Add `GINAccessibilitySVIGNN` class | `granite/models/gnn.py` | +~120 |
| Add `GINConv` import | `granite/models/gnn.py` | +1 |
| Update 4 dispatch sites (GIN branch) | `granite/disaggregation/pipeline.py` | +12 (4 sites x 3 lines) |
| Update 4 import lines (GIN) | `granite/disaggregation/pipeline.py` | +4 |
| Add `GCNAccessibilitySVIGNN` class | `granite/models/gnn.py` | +~100 |
| Update 4 dispatch sites (GCN branch) | `granite/disaggregation/pipeline.py` | +4 (one line each) |
| Update 4 import lines (GCN) | `granite/disaggregation/pipeline.py` | +4 |
| **Total** | **2 files** | **~245 lines** |

No changes to trainers, feature pipeline, data loaders, or evaluation code are required. Cache keys are unaffected (they hash feature values and tract FIPS, not model architecture).
