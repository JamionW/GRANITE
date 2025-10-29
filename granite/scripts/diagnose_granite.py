#!/usr/bin/env python3
"""
GRANITE Diagnostic Tool
Identifies issues with OSRM and data setup, suggests fixes
"""
import os
import subprocess
import sys
from pathlib import Path
import requests

def check_osrm_servers():
    """Check if OSRM servers are running"""
    print("\n" + "="*60)
    print("CHECKING OSRM SERVERS")
    print("="*60)
    
    servers = [
        ("Driving", "http://localhost:5000/route/v1/driving/-85.3,35.0;-85.2,35.0"),
        ("Walking", "http://localhost:5001/route/v1/foot/-85.3,35.0;-85.2,35.0")
    ]
    
    issues = []
    
    for name, url in servers:
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                print(f"✓ {name} OSRM server responding on {url.split('/')[2]}")
            else:
                print(f"✗ {name} OSRM server returned status {response.status_code}")
                issues.append(f"{name} server not working properly")
        except requests.exceptions.ConnectionError:
            print(f"✗ {name} OSRM server not reachable")
            issues.append(f"{name} server not running")
        except Exception as e:
            print(f"✗ {name} OSRM server check failed: {e}")
            issues.append(f"{name} server error: {e}")
    
    if issues:
        print("\n⚠ OSRM Issues Found:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nSuggested fixes:")
        print("  1. Check if Docker is running: docker ps")
        print("  2. Start OSRM: bash scripts/start_osrm.sh")
        print("  3. Check logs: docker logs osrm-backend-car")
        return False
    else:
        print("\n✓ All OSRM servers working correctly")
        return True

def check_osrm_scripts():
    """Check if OSRM startup scripts exist and are executable"""
    print("\n" + "="*60)
    print("CHECKING OSRM SCRIPTS")
    print("="*60)
    
    scripts = [
        "scripts/start_osrm.sh",
        "scripts/setup_osrm.sh"
    ]
    
    issues = []
    
    for script in scripts:
        path = Path(script)
        if not path.exists():
            print(f"✗ Missing: {script}")
            issues.append(f"{script} not found")
        else:
            is_executable = os.access(path, os.X_OK)
            if is_executable:
                print(f"✓ Found and executable: {script}")
            else:
                print(f"⚠ Found but not executable: {script}")
                issues.append(f"{script} needs execute permission")
    
    if issues:
        print("\n⚠ Script Issues Found:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nSuggested fixes:")
        print("  chmod +x scripts/start_osrm.sh")
        print("  chmod +x scripts/setup_osrm.sh")
        return False
    else:
        print("\n✓ All OSRM scripts present and executable")
        return True

def check_data_structure():
    """Check if data directory structure is correct"""
    print("\n" + "="*60)
    print("CHECKING DATA DIRECTORY STRUCTURE")
    print("="*60)
    
    expected_structure = {
        "data/raw/lehd": ["tn_wac_S000_JT00_2021.csv"],
        "data/raw/healthcare": ["hamilton_county_healthcare.csv"],
        "data/raw/osm_grocery": ["hamilton_county_grocery_stores.csv"]
    }
    
    issues = []
    
    for directory, expected_files in expected_structure.items():
        path = Path(directory)
        
        if not path.exists():
            print(f"✗ Missing directory: {directory}")
            issues.append(f"Create directory: mkdir -p {directory}")
        else:
            print(f"✓ Found directory: {directory}")
            
            for expected_file in expected_files:
                file_path = path / expected_file
                if not file_path.exists():
                    print(f"  ✗ Missing file: {expected_file}")
                    issues.append(f"Missing data file: {directory}/{expected_file}")
                else:
                    size = file_path.stat().st_size
                    print(f"  ✓ Found file: {expected_file} ({size:,} bytes)")
    
    if issues:
        print("\n⚠ Data Structure Issues Found:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nSuggested fixes:")
        print("  1. Create missing directories: mkdir -p data/raw/{lehd,healthcare,osm_grocery}")
        print("  2. Check if data files exist elsewhere: find . -name '*.csv' | grep -E '(lehd|healthcare|grocery)'")
        print("  3. Move files to correct location if found")
        return False
    else:
        print("\n✓ Data directory structure correct")
        return True

def check_data_file_locations():
    """Search for data files that might be in wrong locations"""
    print("\n" + "="*60)
    print("SEARCHING FOR MISPLACED DATA FILES")
    print("="*60)
    
    search_patterns = {
        "LEHD Employment": "*wac_S000*.csv",
        "Healthcare": "*healthcare*.csv",
        "Grocery": "*grocery*.csv"
    }
    
    found_files = {}
    
    for name, pattern in search_patterns.items():
        print(f"\nSearching for {name} files ({pattern})...")
        try:
            result = subprocess.run(
                ["find", ".", "-name", pattern, "-type", "f"],
                capture_output=True,
                text=True,
                timeout=10
            )
            files = [f for f in result.stdout.strip().split('\n') if f]
            
            if files:
                print(f"✓ Found {len(files)} file(s):")
                for file in files:
                    print(f"  - {file}")
                found_files[name] = files
            else:
                print(f"✗ No files found matching {pattern}")
                
        except Exception as e:
            print(f"⚠ Search failed: {e}")
    
    if found_files:
        print("\n" + "="*60)
        print("FILES FOUND IN UNEXPECTED LOCATIONS")
        print("="*60)
        print("\nYou may need to move these files to the correct locations:")
        print("  LEHD files → data/raw/lehd/")
        print("  Healthcare files → data/raw/healthcare/")
        print("  Grocery files → data/raw/osm_grocery/")
        return True
    else:
        print("\n⚠ No data files found anywhere")
        return False

def run_diagnostics():
    """Run all diagnostic checks"""
    print("\n" + "="*60)
    print("GRANITE DIAGNOSTIC TOOL")
    print("Checking OSRM and data configuration")
    print("="*60)
    
    results = {
        "OSRM Scripts": check_osrm_scripts(),
        "OSRM Servers": check_osrm_servers(),
        "Data Structure": check_data_structure()
    }
    
    check_data_file_locations()
    
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)
    
    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} - {check}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL CHECKS PASSED")
        print("System appears to be configured correctly")
        print("\nYou can now run: python test_real_data_integration_strict.py")
    else:
        print("✗ ISSUES DETECTED")
        print("Review the output above and apply suggested fixes")
        print("\nAfter fixing issues, run this diagnostic again")
    print("="*60)
    
    return all_passed

if __name__ == "__main__":
    success = run_diagnostics()
    sys.exit(0 if success else 1)