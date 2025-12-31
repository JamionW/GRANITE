import pickle
import numpy as np

with open('./granite_cache/absolute/cac27375632ccbb631a25093c6cff74c.pkl', 'rb') as f:
    features = pickle.load(f)

# Check drive_advantage columns (6, 16, 26)
for col, name in [(6, 'employment_drive_advantage'), (16, 'healthcare_drive_advantage'), (26, 'grocery_drive_advantage')]:
    vals = features[:, col]
    print(f"{name}:")
    print(f"  min={vals.min():.6f}, max={vals.max():.6f}")
    print(f"  mean={vals.mean():.6f}, std={vals.std():.6f}")
    print(f"  unique values: {len(np.unique(vals.round(6)))}")
    print()