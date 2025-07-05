#!/bin/bash
# GRANITE Data Setup Script
# Quick one-command setup for GRANITE framework data

echo "🚀 GRANITE Data Setup Starting..."

# Check if we're in the right directory
if [ ! -d "granite" ]; then
    echo "❌ Error: Please run this from the GRANITE project root directory"
    exit 1
fi

# Install required Python packages for downloading (if needed)
echo "📦 Checking Python dependencies..."
pip install requests pandas geopandas --quiet

# Run the download script
echo "📥 Starting automated data download..."
python granite_download_data.py

# Check if downloads were successful
if [ -f "data/raw/SVI2020_US_tract.csv" ] || [ -f "data/raw/SVI_2020_US.csv" ]; then
    echo "✅ SVI data ready"
else
    echo "⚠️  SVI data not found - will use mock data"
fi

if [ -f "data/raw/tl_2020_47_tract.shp" ]; then
    echo "✅ Census tracts ready"
else
    echo "⚠️  Census tracts not found - will use mock data"
fi

if [ -f "data/raw/tl_2023_47065_roads.shp" ] || [ -f "data/raw/tl_2022_47065_roads.shp" ]; then
    echo "✅ Roads data ready"
else
    echo "⚠️  Roads data not found - will use mock data"
fi

echo ""
echo "🎯 Setup complete! Try running:"
echo "   python scripts/run_granite.py --epochs 10"
echo ""