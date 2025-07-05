#!/usr/bin/env python3
"""
GRANITE Automated Data Downloader
Downloads and organizes all required data files for GRANITE framework
"""

import os
import sys
import requests
import zipfile
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import time


def print_banner():
    print("=" * 80)
    print("📥 GRANITE Automated Data Downloader")
    print("Downloading and organizing data for Hamilton County, TN analysis")
    print("=" * 80)


def create_directories(base_path="data"):
    """Create necessary directory structure"""
    dirs = [
        f"{base_path}/raw",
        f"{base_path}/processed"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")


def download_file(url, destination, description="file", timeout=300):
    """Download a file with progress indication"""
    print(f"\n📥 Downloading {description}...")
    print(f"   From: {url[:70]}{'...' if len(url) > 70 else ''}")
    print(f"   To: {destination}")
    
    try:
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
        
        # Get file size if available
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination, 'wb') as f:
            if total_size == 0:
                # No content-length header
                f.write(response.content)
                print(f"   ✓ Downloaded {description}")
            else:
                # Show progress
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        percent = (downloaded / total_size) * 100
                        print(f"\r   Progress: {percent:.1f}% ({downloaded}/{total_size} bytes)", end="")
                
                print(f"\n   ✓ Downloaded {description} ({total_size} bytes)")
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Failed to download {description}: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Error downloading {description}: {e}")
        return False


def extract_zip(zip_path, extract_to, description="archive"):
    """Extract ZIP file to destination"""
    print(f"\n📦 Extracting {description}...")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        
        # List extracted files
        extracted_files = []
        for root, dirs, files in os.walk(extract_to):
            for file in files:
                extracted_files.append(os.path.join(root, file))
        
        print(f"   ✓ Extracted {len(extracted_files)} files")
        for file in extracted_files[:5]:  # Show first 5 files
            print(f"     - {os.path.basename(file)}")
        if len(extracted_files) > 5:
            print(f"     ... and {len(extracted_files) - 5} more files")
        
        return extracted_files
        
    except Exception as e:
        print(f"   ❌ Failed to extract {description}: {e}")
        return []


def download_svi_data(data_dir="data"):
    """Download CDC Social Vulnerability Index data"""
    print("\n" + "🏥 CDC SOCIAL VULNERABILITY INDEX DATA" + "=" * 45)
    
    # Try multiple SVI URLs
    svi_urls = [
        {
            "url": "https://svi.cdc.gov/Documents/Data/2020/csv/SVI2020_US_tract.csv",
            "filename": "SVI2020_US_tract.csv"
        },
        {
            "url": "https://www.atsdr.cdc.gov/placeandhealth/svi/data/SviDataCsv/SVI_2020_US.csv", 
            "filename": "SVI_2020_US.csv"
        },
        {
            "url": "https://svi.cdc.gov/Documents/Data/2020/csv/SVI_2020_US.csv",
            "filename": "SVI_2020_US.csv"
        }
    ]
    
    for i, source in enumerate(svi_urls):
        destination = os.path.join(data_dir, "raw", source["filename"])
        
        if download_file(source["url"], destination, f"SVI 2020 data (source {i+1})"):
            # Verify the file is a valid CSV
            try:
                import pandas as pd
                df = pd.read_csv(destination, nrows=5)
                if len(df.columns) > 10:  # SVI should have many columns
                    print(f"   ✓ Verified CSV with {len(df.columns)} columns")
                    return True
                else:
                    print(f"   ⚠️  Invalid CSV - only {len(df.columns)} columns")
                    os.remove(destination)
            except Exception as e:
                print(f"   ⚠️  Could not verify CSV: {e}")
                os.remove(destination)
    
    print("\n💡 Manual SVI Download Instructions:")
    print("   1. Go to: https://svi.cdc.gov/dataDownloads/data-download.html")
    print("   2. Select: United States > 2020 > Census Tracts > CSV File")
    print("   3. Save as: data/raw/SVI2020_US_tract.csv")
    
    return False


def download_census_tracts(data_dir="data"):
    """Download Tennessee census tracts"""
    print("\n" + "🗺️  TENNESSEE CENSUS TRACTS" + "=" * 48)
    
    # Try direct download URLs
    tract_urls = [
        "https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_47_tract.zip",
        "https://census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_47_tract.zip"
    ]
    
    for i, url in enumerate(tract_urls):
        temp_zip = os.path.join(tempfile.gettempdir(), "tl_2020_47_tract.zip")
        
        if download_file(url, temp_zip, f"TN Census Tracts (source {i+1})"):
            # Extract to data/raw
            extracted_files = extract_zip(temp_zip, os.path.join(data_dir, "raw"), "Tennessee census tracts")
            
            if extracted_files:
                # Clean up temp file
                os.remove(temp_zip)
                
                # Verify we got the shapefile components
                shp_files = [f for f in extracted_files if f.endswith('.shp')]
                if shp_files:
                    print(f"   ✓ Successfully extracted census tracts shapefile")
                    return True
    
    print("\n💡 Manual Census Tracts Download Instructions:")
    print("   1. Go to: https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2020&layergroup=Census+Tracts")
    print("   2. Select: Tennessee")
    print("   3. Download and extract to: data/raw/")
    
    return False


def download_hamilton_roads(data_dir="data"):
    """Download Hamilton County roads"""
    print("\n" + "🛣️  HAMILTON COUNTY ROADS" + "=" * 50)
    
    # Try direct download URLs  
    roads_urls = [
        "https://www2.census.gov/geo/tiger/TIGER2023/ROADS/tl_2023_47065_roads.zip",
        "https://www2.census.gov/geo/tiger/TIGER2022/ROADS/tl_2022_47065_roads.zip",
        "https://census.gov/geo/tiger/TIGER2023/ROADS/tl_2023_47065_roads.zip"
    ]
    
    for i, url in enumerate(roads_urls):
        temp_zip = os.path.join(tempfile.gettempdir(), f"tl_roads_{i}.zip")
        
        if download_file(url, temp_zip, f"Hamilton County Roads (source {i+1})"):
            # Extract to data/raw
            extracted_files = extract_zip(temp_zip, os.path.join(data_dir, "raw"), "Hamilton County roads")
            
            if extracted_files:
                # Clean up temp file
                os.remove(temp_zip)
                
                # Verify we got the shapefile components
                shp_files = [f for f in extracted_files if f.endswith('.shp')]
                if shp_files:
                    print(f"   ✓ Successfully extracted roads shapefile")
                    return True
    
    print("\n💡 Manual Roads Download Instructions:")
    print("   1. Go to: https://catalog.data.gov/dataset/tiger-line-shapefile-current-county-hamilton-county-tn-all-roads")
    print("   2. Download the shapefile")
    print("   3. Extract to: data/raw/")
    
    return False


def verify_downloads(data_dir="data"):
    """Verify all required files are present and valid"""
    print("\n" + "✅ VERIFICATION" + "=" * 65)
    
    required_files = {
        "SVI Data": [
            "SVI2020_US_tract.csv",
            "SVI_2020_US.csv"  # Alternative name
        ],
        "TN Census Tracts": [
            "tl_2020_47_tract.shp",
            "tl_2020_47_tract.shx", 
            "tl_2020_47_tract.dbf",
            "tl_2020_47_tract.prj"
        ],
        "Hamilton Roads": [
            "tl_2023_47065_roads.shp",
            "tl_2022_47065_roads.shp"  # Alternative year
        ]
    }
    
    success_count = 0
    total_categories = len(required_files)
    
    for category, files in required_files.items():
        print(f"\n📋 Checking {category}:")
        
        found_files = []
        for filename in files:
            file_path = os.path.join(data_dir, "raw", filename)
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                print(f"   ✓ {filename} ({file_size:,} bytes)")
                found_files.append(filename)
        
        if found_files:
            success_count += 1
            print(f"   ✅ {category}: READY")
        else:
            print(f"   ❌ {category}: MISSING")
            print(f"      Need at least one of: {', '.join(files)}")
    
    print(f"\n📊 Summary: {success_count}/{total_categories} data categories ready")
    
    if success_count == total_categories:
        print("🎉 All data downloaded successfully!")
        print("\n🚀 Ready to run GRANITE:")
        print("   python scripts/run_granite.py --epochs 20")
        return True
    else:
        print("⚠️  Some data missing - see manual download instructions above")
        print("   GRANITE will use mock data for missing files")
        return False


def create_config_file(data_dir="data"):
    """Create/update config file with local file paths"""
    print("\n⚙️  Updating configuration...")
    
    config_content = f"""# GRANITE Framework Configuration - Auto-generated
# Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

data:
  state: "Tennessee"
  state_fips: "47"
  county: "Hamilton"
  county_fips: "065"
  census_year: 2020
  
  # Local data files (downloaded by granite_download_data.py)
  svi_file: "{data_dir}/raw/SVI2020_US_tract.csv"
  roads_file: "{data_dir}/raw/tl_2023_47065_roads.shp"
  census_tracts_file: "{data_dir}/raw/tl_2020_47_tract.shp"
  
  # Fallback settings
  n_synthetic_addresses: 1000
  bbox: [-85.5, 35.0, -85.0, 35.5]

# GNN model settings
model:
  type: "standard"
  input_dim: 5
  hidden_dim: 64
  output_dim: 3
  dropout: 0.2
  epochs: 100
  learning_rate: 0.01

# MetricGraph settings  
metricgraph:
  alpha: 1.5
  mesh_resolution: 0.01
  formula: "y ~ gnn_kappa + gnn_alpha + gnn_tau"

# Output settings
output:
  save_predictions: true
  save_features: true
  save_validation: true
  save_plots: true
"""
    
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    config_file = config_dir / "config.yaml"
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    print(f"   ✓ Updated {config_file}")


def main():
    """Main execution function"""
    print_banner()
    
    # Check if we're in the right directory
    if not os.path.exists("granite"):
        print("❌ Error: Please run this script from the GRANITE project root directory")
        print("   (The directory containing the 'granite' folder)")
        sys.exit(1)
    
    start_time = time.time()
    
    # Create directory structure
    print("\n📁 Setting up directory structure...")
    create_directories()
    
    # Download data
    downloads_successful = []
    
    # SVI Data
    svi_success = download_svi_data()
    downloads_successful.append(("SVI Data", svi_success))
    
    # Census Tracts  
    tracts_success = download_census_tracts()
    downloads_successful.append(("Census Tracts", tracts_success))
    
    # Roads
    roads_success = download_hamilton_roads()
    downloads_successful.append(("Roads", roads_success))
    
    # Verify downloads
    all_ready = verify_downloads()
    
    # Create/update config
    create_config_file()
    
    # Summary
    end_time = time.time()
    print("\n" + "=" * 80)
    print(f"📊 DOWNLOAD SUMMARY (completed in {end_time - start_time:.1f}s)")
    print("=" * 80)
    
    for category, success in downloads_successful:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"   {category:.<20} {status}")
    
    if all_ready:
        print("\n🎉 GRANITE is ready for analysis!")
        print("🚀 Next steps:")
        print("   1. python scripts/run_granite.py --epochs 20")
        print("   2. Check output/ directory for results")
    else:
        print("\n⚠️  Some downloads failed")
        print("💡 Next steps:")
        print("   1. Follow manual download instructions above")
        print("   2. Or run GRANITE with mock data: python scripts/run_granite.py --epochs 5")
    
    print("=" * 80)


if __name__ == "__main__":
    main()