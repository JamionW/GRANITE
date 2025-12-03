import requests
import json
import pandas as pd

# Hamilton County bounding box [south, west, north, east]
bbox = "34.9, -85.4, 35.2, -85.1"  # Approximate Hamilton County bounds

overpass_url = "http://overpass-api.de/api/interpreter"

# Query for grocery stores
overpass_query = f"""
[out:json][timeout:60];
(
  node["shop"="supermarket"]({bbox});
  way["shop"="supermarket"]({bbox});
  node["shop"="convenience"]({bbox});
  way["shop"="convenience"]({bbox});
  node["shop"="grocery"]({bbox});
  way["shop"="grocery"]({bbox});
);
out center;
"""

response = requests.get(overpass_url, params={'data': overpass_query})
data = response.json()

# Parse results
stores = []
for element in data['elements']:
    if element['type'] == 'node':
        lat, lon = element['lat'], element['lon']
    elif 'center' in element:
        lat, lon = element['center']['lat'], element['center']['lon']
    else:
        continue
    
    stores.append({
        'osm_id': element['id'],
        'name': element.get('tags', {}).get('name', 'Unnamed'),
        'type': element.get('tags', {}).get('shop', 'unknown'),
        'lat': lat,
        'lon': lon,
        'brand': element.get('tags', {}).get('brand', None),
        'operator': element.get('tags', {}).get('operator', None)
    })

# Save to CSV
df = pd.DataFrame(stores)
df.to_csv('osm_grocery/hamilton_county_grocery_stores.csv', index=False)
print(f"Downloaded {len(df)} grocery stores")