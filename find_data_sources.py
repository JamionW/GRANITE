#!/usr/bin/env python3
"""
GRANITE Data Source Finder
Searches for and validates working data source URLs for GRANITE framework
"""

import requests
import urllib.request
import urllib.error
import pandas as pd
import geopandas as gpd
import time
from datetime import datetime
import json
import yaml


def print_banner():
    print("=" * 80)
    print("🔍 GRANITE Data Source Finder")
    print("Finding and validating real data sources...")
    print("=" * 80)


def test_url(url, timeout=30, description=""):
    """Test if a URL is accessible and returns data"""
    try:
        print(f"  Testing: {description}")
        print(f"    URL: {url[:60]}{'...' if len(url) > 60 else ''}")
        
        # Try HEAD request first (faster)
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        
        if response.status_code == 200:
            # For CSV files, try to read a few lines
            if url.endswith('.csv'):
                df = pd.read_csv(url, nrows=5)
                print(f"    ✅ WORKING - CSV with {len(df.columns)} columns")
                return True, f"CSV with {len(df.columns)} columns, sample: {list(df.columns[:3])}"
            
            # For ZIP files, check if they exist
            elif url.endswith('.zip'):
                print(f"    ✅ WORKING - ZIP file accessible")
                return True, "ZIP file accessible"
            
            else:
                print(f"    ✅ WORKING - HTTP {response.status_code}")
                return True, f"HTTP {response.status_code}"
                
        else:
            print(f"    ❌ FAILED - HTTP {response.status_code}")
            return False, f"HTTP {response.status_code}"
            
    except requests.exceptions.Timeout:
        print(f"    ❌ FAILED - Timeout after {timeout}s")
        return False, "Timeout"
    except Exception as e:
        print(f"    ❌ FAILED - {str(e)[:50]}")
        return False, str(e)[:50]


def find_svi_data():
    """Find working SVI data sources"""
    print("\n📊 Testing SVI Data Sources...")
    
    svi_sources = [
        {
            "name": "CDC ATSDR SVI 2020 (Primary)",
            "url": "https://www.atsdr.cdc.gov/placeandhealth/svi/data/SviDataCsv/SVI_2020_US.csv",
            "description": "Official CDC SVI 2020 data"
        },
        {
            "name": "CDC SVI Portal 2020", 
            "url": "https://svi.cdc.gov/Documents/Data/2020/csv/SVI_2020_US.csv",
            "description": "CDC SVI portal download"
        },
        {
            "name": "CDC Alternative Download",
            "url": "https://www.atsdr.cdc.gov/placeandhealth/svi/data/SviDataCsv/SVI2020_US.csv",
            "description": "Alternative CDC download location"
        },
        {
            "name": "Census Bureau SVI Mirror",
            "url": "https://www2.census.gov/programs-surveys/acs/data/svi/SVI_2020_US.csv", 
            "description": "Census Bureau mirror of SVI data"
        },
        {
            "name": "FEMA SVI Data",
            "url": "https://hazards.fema.gov/nri/data-resources/SVI_2020_US.csv",
            "description": "FEMA National Risk Index SVI data"
        }
    ]
    
    working_sources = []
    
    for source in svi_sources:
        success, details = test_url(source["url"], description=source["name"])
        if success:
            working_sources.append({
                **source,
                "status": "WORKING",
                "details": details,
                "tested_at": datetime.now().isoformat()
            })
    
    return working_sources


def find_census_data():
    """Find working Census TIGER data sources"""
    print("\n🗺️ Testing Census TIGER Data Sources...")
    
    # Hamilton County, TN: State FIPS 47, County FIPS 065
    years = [2023, 2022, 2021, 2020]
    
    census_sources = []
    working_sources = []
    
    # Test different years for different data types
    for year in years:
        census_sources.extend([
            {
                "name": f"Census Tracts {year}",
                "url": f"https://www2.census.gov/geo/tiger/TIGER{year}/TRACT/tl_{year}_47_tract.zip",
                "type": "census_tracts",
                "year": year
            },
            {
                "name": f"Hamilton County Roads {year}",
                "url": f"https://www2.census.gov/geo/tiger/TIGER{year}/ROADS/tl_{year}_47065_roads.zip",
                "type": "roads", 
                "year": year
            }
        ])
    
    # Test each source
    for source in census_sources:
        success, details = test_url(source["url"], description=source["name"])
        if success:
            working_sources.append({
                **source,
                "status": "WORKING",
                "details": details,
                "tested_at": datetime.now().isoformat()
            })
    
    return working_sources


def find_alternative_data():
    """Find alternative data sources"""
    print("\n🔄 Testing Alternative Data Sources...")
    
    alternative_sources = [
        {
            "name": "OpenStreetMap Hamilton County",
            "url": "https://download.geofabrik.de/north-america/us/tennessee-latest.osm.pbf",
            "type": "osm_roads",
            "description": "OpenStreetMap data for Tennessee"
        },
        {
            "name": "GTFS Transit Data",
            "url": "https://transitland.org/api/v2/feeds",
            "type": "transit",
            "description": "Transitland GTFS feeds API"
        }
    ]
    
    working_sources = []
    
    for source in alternative_sources:
        success, details = test_url(source["url"], description=source["name"])
        if success:
            working_sources.append({
                **source,
                "status": "WORKING", 
                "details": details,
                "tested_at": datetime.now().isoformat()
            })
    
    return working_sources


def generate_config_updates(working_sources):
    """Generate config file updates"""
    print("\n⚙️ Generating Configuration Updates...")
    
    # Organize sources by type
    svi_sources = [s for s in working_sources if 'svi' in s['name'].lower() or 'SVI' in s['url']]
    tract_sources = [s for s in working_sources if s.get('type') == 'census_tracts']
    road_sources = [s for s in working_sources if s.get('type') == 'roads']
    
    config_update = {
        "data_sources": {
            "svi": {
                "primary_url": svi_sources[0]["url"] if svi_sources else None,
                "backup_urls": [s["url"] for s in svi_sources[1:3]] if len(svi_sources) > 1 else [],
                "last_updated": datetime.now().isoformat()
            },
            "census_tracts": {
                "primary_url": tract_sources[0]["url"] if tract_sources else None,
                "backup_urls": [s["url"] for s in tract_sources[1:3]] if len(tract_sources) > 1 else [],
                "last_updated": datetime.now().isoformat()
            },
            "roads": {
                "primary_url": road_sources[0]["url"] if road_sources else None,
                "backup_urls": [s["url"] for s in road_sources[1:3]] if len(road_sources) > 1 else [],
                "last_updated": datetime.now().isoformat()
            }
        }
    }
    
    return config_update


def save_results(all_sources, config_update):
    """Save results to files"""
    print("\n💾 Saving Results...")
    
    # Save detailed results
    with open('data_source_test_results.json', 'w') as f:
        json.dump(all_sources, f, indent=2)
    print("  ✓ Detailed results: data_source_test_results.json")
    
    # Save config update
    with open('config_data_sources.yaml', 'w') as f:
        yaml.dump(config_update, f, default_flow_style=False)
    print("  ✓ Config update: config_data_sources.yaml")
    
    # Save quick reference
    with open('working_urls.txt', 'w') as f:
        f.write("GRANITE Framework - Working Data Source URLs\n")
        f.write("=" * 50 + "\n\n")
        
        # SVI URLs
        svi_urls = [s for s in all_sources if 'svi' in s['name'].lower() or 'SVI' in s['url']]
        if svi_urls:
            f.write("SVI Data Sources:\n")
            for i, source in enumerate(svi_urls[:3], 1):
                f.write(f"{i}. {source['name']}\n")
                f.write(f"   {source['url']}\n\n")
        
        # Census URLs
        census_urls = [s for s in all_sources if s.get('type') in ['census_tracts', 'roads']]
        if census_urls:
            f.write("Census TIGER Data:\n")
            for source in census_urls:
                f.write(f"- {source['name']}: {source['url']}\n")
        
        f.write(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print("  ✓ Quick reference: working_urls.txt")


def main():
    """Main execution function"""
    print_banner()
    
    start_time = time.time()
    all_working_sources = []
    
    # Test all data sources
    try:
        svi_sources = find_svi_data()
        all_working_sources.extend(svi_sources)
        
        census_sources = find_census_data()
        all_working_sources.extend(census_sources)
        
        alt_sources = find_alternative_data()
        all_working_sources.extend(alt_sources)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Testing interrupted by user")
        
    # Generate config updates
    config_update = generate_config_updates(all_working_sources)
    
    # Save results
    save_results(all_working_sources, config_update)
    
    # Summary
    end_time = time.time()
    print("\n" + "=" * 80)
    print(f"🎯 Data Source Discovery Complete ({end_time - start_time:.1f}s)")
    print("=" * 80)
    
    if all_working_sources:
        print(f"✅ Found {len(all_working_sources)} working data sources:")
        
        # Group by type
        svi_count = len([s for s in all_working_sources if 'svi' in s['name'].lower() or 'SVI' in s['url']])
        census_count = len([s for s in all_working_sources if s.get('type') in ['census_tracts', 'roads']])
        alt_count = len(all_working_sources) - svi_count - census_count
        
        print(f"   📊 SVI Data: {svi_count} sources")
        print(f"   🗺️ Census Data: {census_count} sources") 
        print(f"   🔄 Alternative: {alt_count} sources")
        
        print(f"\n📋 Next Steps:")
        print(f"   1. Review: working_urls.txt")
        print(f"   2. Update config/config.yaml with URLs from config_data_sources.yaml")
        print(f"   3. Test: python scripts/run_granite.py")
        
        # Show top recommendations
        svi_sources = [s for s in all_working_sources if 'svi' in s['name'].lower() or 'SVI' in s['url']]
        if svi_sources:
            print(f"\n🎯 Recommended SVI URL:")
            print(f"   {svi_sources[0]['url']}")
        
        road_sources = [s for s in all_working_sources if s.get('type') == 'roads']
        if road_sources:
            print(f"\n🛣️ Recommended Roads URL:")
            print(f"   {road_sources[0]['url']}")
            
    else:
        print("❌ No working data sources found!")
        print("   • Check your internet connection")
        print("   • Data sources may have moved - check CDC and Census websites manually")
        print("   • Consider using local data files instead")
    
    print("=" * 80)


if __name__ == "__main__":
    main()