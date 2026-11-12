# GRANITE Import Convention

**TL;DR: Always use absolute imports from the `granite` package root.**

## Project Structure

```
/workspaces/GRANITE/
├── granite/                          ← Package root
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── gnn.py
│   │   └── mixture_of_experts.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loaders.py
│   │   └── enhanced_accessibility.py
│   ├── disaggregation/
│   │   ├── __init__.py
│   │   └── pipeline.py
│   └── ... (other modules)
├── scripts/
│   └── ... (run_holdout_validation.py lives at granite/scripts/run_holdout_validation.py)
└── ... (other files)
```

## Import Rules

### ✓ CORRECT: Absolute imports from `granite` package

```python
# In any file (models, scripts, anywhere):
from granite.models.gnn import AccessibilitySVIGNN, set_random_seed
from granite.models.mixture_of_experts import MixtureOfExpertsGNN
from granite.data.loaders import DataLoader
from granite.disaggregation.pipeline import GRANITEPipeline
```

### ✗ WRONG: Bare imports (don't use)
```python
from gnn import AccessibilitySVIGNN  # ✗ FAILS
from models.gnn import ...           # ✗ FAILS
```

### ✗ WRONG: Relative imports at module level (don't use)
```python
from .gnn import AccessibilitySVIGNN  # ✗ FAILS when loaded directly
```

## When Code is Generated

When generating new code that imports from GRANITE modules:

1. **Always check the full path** from project root
2. **Always use absolute imports** starting with `granite.`
3. **Never use relative imports** (`.module`) except within `__init__.py`
4. **Never use bare imports** (just the module name)

## Examples

### ✓ Correct Example: New MoE File

File: `granite/models/mixture_of_experts.py`

```python
import torch
import torch.nn as nn

# Absolute import - works everywhere
from granite.models.gnn import AccessibilitySVIGNN, set_random_seed

class MixtureOfExpertsGNN(nn.Module):
    def __init__(self):
        # Uses gnn classes successfully
        self.expert = AccessibilitySVIGNN(...)
```

### ✓ Correct Example: New Script

File: `granite/scripts/validate_moe.py`

```python
import sys
from pathlib import Path

# Add project root to path if running from elsewhere
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Absolute import - works whether run from root or scripts dir
from granite.models.mixture_of_experts import MixtureOfExpertsGNN
from granite.data.loaders import DataLoader
```

### ✓ Correct Example: __init__.py (only place relative imports are OK)

File: `granite/models/__init__.py`

```python
# In __init__.py, relative imports are acceptable
from .gnn import AccessibilitySVIGNN, set_random_seed
from .mixture_of_experts import MixtureOfExpertsGNN

__all__ = ['AccessibilitySVIGNN', 'set_random_seed', 'MixtureOfExpertsGNN']
```

Then in code you can do:
```python
from granite.models import AccessibilitySVIGNN
```

## Testing Imports

When generating a new module, verify imports work like this:

```bash
cd /workspaces/GRANITE

# Test 1: Import works from project root
python -c "from granite.models.mixture_of_experts import MixtureOfExpertsGNN; print('✓ Works')"

# Test 2: Import works from anywhere
cd /tmp
python -c "import sys; sys.path.insert(0, '/workspaces/GRANITE'); from granite.models.mixture_of_experts import MixtureOfExpertsGNN; print('✓ Works')"
```

## Summary for Code Generation

When generating code for GRANITE:

| Situation | Rule | Example |
|-----------|------|---------|
| New module in `granite/models/` | Use absolute imports from `granite` | `from granite.models.gnn import ...` |
| New module in `granite/data/` | Use absolute imports from `granite` | `from granite.data.loaders import ...` |
| New script in `granite/scripts/` | Use absolute imports from `granite` | `from granite.models.gnn import ...` |
| In `__init__.py` files | Use relative imports | `from .gnn import ...` |
| Standalone test file | Can use absolute imports | `from granite.models.gnn import ...` |

## Quick Checklist

When code is generated, check:

- [ ] Imports start with `granite.`
- [ ] No bare module names (e.g., not `from gnn import`)
- [ ] No relative imports (`.something`) except in `__init__.py`
- [ ] Can import successfully: `python -c "from granite.X.Y import Z"`
