"""Minimal test to check which variables are available at block group level."""
import requests

API_KEY = "5a543fddd0caa7caa19763c16b0e0faad29f761c"

# Test specific variables one at a time
test_vars = [
    ('B17001_001E', 'Poverty universe'),
    ('B17001_002E', 'Below poverty'),
    ('B06012_001E', 'Poverty status universe (alt)'),
    ('B06012_002E', 'Below 100% poverty (alt)'),
    ('B19013_001E', 'Median household income'),
    ('B25044_001E', 'Vehicle universe'),
]

print("Testing variable availability at block group level...")
print("="*60)

for var, desc in test_vars:
    url = (
        f"https://api.census.gov/data/2020/acs/acs5"
        f"?get=NAME,{var}"
        f"&for=block%20group:*"
        f"&in=state:47%20county:065"
        f"&key={API_KEY}"
    )
    
    response = requests.get(url, timeout=30)
    
    if response.status_code == 200:
        data = response.json()
        # Check first few values
        vals = [row[1] for row in data[1:4]]
        print(f"OK   {var}: {desc}")
        print(f"     Sample values: {vals}")
    else:
        print(f"FAIL {var}: {desc}")
        print(f"     Error: {response.text[:100]}")
    print()