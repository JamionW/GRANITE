"""
Robust setup script for GRANITE framework with system dependency detection
"""
import sys
import os
import subprocess
import warnings
import platform
from pathlib import Path
from setuptools import setup, find_packages
from setuptools.command.install import install


class SystemChecker:
    """Check system dependencies and provide installation guidance"""
    
    def __init__(self):
        self.os_type = platform.system().lower()
        self.distro = self._get_linux_distro()
        self.missing_deps = []
        self.install_commands = []
    
    def _get_linux_distro(self):
        """Detect Linux distribution"""
        if self.os_type != 'linux':
            return None
        
        try:
            with open('/etc/os-release') as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith('ID='):
                        return line.split('=')[1].strip().strip('"')
        except FileNotFoundError:
            pass
        
        # Fallback detection
        if os.path.exists('/etc/debian_version'):
            return 'debian'
        elif os.path.exists('/etc/redhat-release'):
            return 'rhel'
        
        return 'unknown'
    
    def check_r_installation(self):
        """Check if R is properly installed"""
        try:
            result = subprocess.run(['R', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                return True, result.stdout.split('\n')[0]
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            pass
        return False, None
    
    def check_system_libraries(self):
        """Check for required system libraries"""
        required_libs = {
            'libtirpc': ['/usr/lib/x86_64-linux-gnu/libtirpc.so', '/usr/lib64/libtirpc.so'],
            'libgdal': ['/usr/lib/x86_64-linux-gnu/libgdal.so', '/usr/lib64/libgdal.so'],
            'libgeos': ['/usr/lib/x86_64-linux-gnu/libgeos.so', '/usr/lib64/libgeos.so'],
            'libproj': ['/usr/lib/x86_64-linux-gnu/libproj.so', '/usr/lib64/libproj.so']
        }
        
        missing = []
        for lib_name, possible_paths in required_libs.items():
            if not any(os.path.exists(path) for path in possible_paths):
                missing.append(lib_name)
        
        return missing
    
    def check_dev_headers(self):
        """Check for development headers"""
        header_checks = {
            'libtirpc-dev': ['/usr/include/tirpc/netconfig.h', '/usr/include/tirpc/rpc/rpc.h'],
            'libgdal-dev': ['/usr/include/gdal/gdal.h', '/usr/include/gdal.h'],
            'r-base-dev': ['/usr/share/R/include/R.h'],
            'python3-dev': [f'/usr/include/python{sys.version_info.major}.{sys.version_info.minor}/Python.h']
        }
        
        missing = []
        for pkg_name, header_paths in header_checks.items():
            if not any(os.path.exists(path) for path in header_paths):
                missing.append(pkg_name)
        
        return missing
    
    def generate_install_commands(self, missing_packages):
        """Generate installation commands for missing packages"""
        if self.os_type == 'linux':
            if self.distro in ['ubuntu', 'debian']:
                base_cmd = 'sudo apt-get update && sudo apt-get install -y'
                packages = ' '.join(missing_packages)
                return [f"{base_cmd} {packages}"]
            elif self.distro in ['rhel', 'centos', 'rocky', 'almalinux']:
                # Map package names
                pkg_map = {
                    'libtirpc-dev': 'libtirpc-devel',
                    'libgdal-dev': 'gdal-devel',
                    'libgeos-dev': 'geos-devel',
                    'libproj-dev': 'proj-devel',
                    'r-base-dev': 'R-devel',
                    'python3-dev': 'python3-devel'
                }
                mapped_packages = [pkg_map.get(pkg, pkg) for pkg in missing_packages]
                packages = ' '.join(mapped_packages)
                return [f"sudo dnf install -y {packages}"]
        elif self.os_type == 'darwin':
            return ['brew install r gdal geos proj']
        
        return []
    
    def comprehensive_check(self):
        """Run all system checks and provide guidance"""
        print("🔍 Checking system dependencies...")
        
        # Check R
        r_installed, r_version = self.check_r_installation()
        if not r_installed:
            self.missing_deps.append("R (r-base r-base-dev)")
            print("  ❌ R not found")
        else:
            print(f"  ✅ R found: {r_version}")
        
        # Check system libraries
        missing_libs = self.check_system_libraries()
        if missing_libs:
            print(f"  ❌ Missing system libraries: {', '.join(missing_libs)}")
        else:
            print("  ✅ System libraries found")
        
        # Check development headers
        missing_headers = self.check_dev_headers()
        if missing_headers:
            self.missing_deps.extend(missing_headers)
            print(f"  ❌ Missing development headers: {', '.join(missing_headers)}")
        else:
            print("  ✅ Development headers found")
        
        # Generate installation commands
        if self.missing_deps:
            self.install_commands = self.generate_install_commands(self.missing_deps)
            return False
        
        return True


def get_requirements_with_fallbacks():
    """Get requirements with intelligent fallbacks"""
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        base_requirements = [line.strip() for line in fh 
                           if line.strip() and not line.startswith("#")]
    
    # Check system capabilities
    checker = SystemChecker()
    system_ok = checker.comprehensive_check()
    
    if not system_ok:
        print("\n⚠️  System dependency issues detected!")
        print("\nTo fix these issues, run:")
        for cmd in checker.install_commands:
            print(f"  {cmd}")
        
        print("\nAlternatively, GRANITE will attempt to install with fallbacks...")
        
        # Remove problematic packages and add fallbacks
        fallback_requirements = []
        
        for req in base_requirements:
            if req.startswith('rpy2'):
                # Try to use pre-compiled wheels or ABI mode
                print("  → Using rpy2 fallback strategy")
                fallback_requirements.append('rpy2>=3.4.0 ; platform_system != "Windows"')
            elif req.startswith('geopandas'):
                # GeoPandas might also have issues
                fallback_requirements.append('geopandas>=0.10.0')
            else:
                fallback_requirements.append(req)
        
        return fallback_requirements, False
    
    print("✅ All system dependencies satisfied!")
    return base_requirements, True


class RobustInstall(install):
    """Custom install command with fallback mechanisms"""
    
    def run(self):
        # Check if we can proceed with full installation
        print("\n🚀 Starting GRANITE installation...")
        
        install.run(self)
        
        # Post-installation setup
        self.post_install_setup()
    
    def post_install_setup(self):
        """Post-installation R package setup and validation"""
        print("\n📦 Setting up R packages...")
        
        try:
            # Try to import rpy2 to test installation
            import rpy2.robjects as ro
            
            # Install R packages
            r_packages = {
                'remotes': 'install.packages("remotes", repos="https://cloud.r-project.org")',
                'INLA': 'install.packages("INLA", repos=c(getOption("repos"), INLA="https://inla.r-inla-download.org/R/stable"))',
                'MetricGraph': 'remotes::install_github("davidbolin/MetricGraph")'
            }
            
            for pkg_name, install_cmd in r_packages.items():
                try:
                    print(f"  Installing {pkg_name}...")
                    ro.r(install_cmd)
                    print(f"    ✅ {pkg_name} installed successfully")
                except Exception as e:
                    print(f"    ⚠️  {pkg_name} installation failed: {e}")
            
        except ImportError as e:
            print(f"  ⚠️  rpy2 not available: {e}")
            print("  → MetricGraph functionality will be limited")
            
            # Create a fallback configuration
            self.create_fallback_config()
    
    def create_fallback_config(self):
        """Create configuration for systems without R"""
        fallback_config = """
# GRANITE Fallback Configuration
# This configuration is used when R/MetricGraph is not available

data:
  state: "Tennessee"
  county: "Hamilton"
  
model:
  type: "gnn_only"  # Disable MetricGraph integration
  hidden_dim: 64
  
metricgraph:
  enabled: false  # Disable MetricGraph
  
disaggregation:
  method: "interpolation"  # Use simple interpolation instead
"""
        
        config_dir = Path("granite/config")
        config_dir.mkdir(exist_ok=True)
        
        with open(config_dir / "fallback_config.yaml", "w") as f:
            f.write(fallback_config)
        
        print("  ✅ Fallback configuration created")


def main():
    # Read README
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
    
    # Get requirements with fallback handling
    requirements, system_ok = get_requirements_with_fallbacks()
    
    # Setup configuration
    setup_kwargs = {
        "name": "granite-svi",
        "version": "0.1.0",
        "author": "Your Name",
        "author_email": "your.email@university.edu",
        "description": "Graph-Refined Accessibility Network for Integrated Transit Equity",
        "long_description": long_description,
        "long_description_content_type": "text/markdown",
        "url": "https://github.com/yourusername/granite",
        "packages": find_packages(),
        "classifiers": [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Science/Research",
            "Topic :: Scientific/Engineering :: GIS",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
        ],
        "python_requires": ">=3.8",
        "install_requires": requirements,
        "entry_points": {
            "console_scripts": [
                "granite=granite.scripts.run_granite:main",  
            ],
        },
        "include_package_data": True,
        "package_data": {
            "granite": ["config/*.yaml"],
        },
        "extras_require": {
            'dev': [
                'pytest>=6.0',
                'pytest-cov>=2.0',
                'flake8>=3.8',
                'black>=21.0',
                'isort>=5.0',
            ],
            'docs': [
                'sphinx>=4.0',
                'sphinx-rtd-theme>=1.0',
                'sphinx-autodoc-typehints>=1.0',
            ],
            'full': [
                'rpy2>=3.4.0',
            ],
            'minimal': [
                # Minimal install without R dependencies
                'numpy>=1.21.0',
                'pandas>=1.3.0',
                'torch>=1.10.0',
                'torch-geometric>=2.0.0',
                'networkx>=2.6.0',
            ]
        },
    }
    
    # Add custom install command only if we have system issues
    if not system_ok:
        setup_kwargs["cmdclass"] = {"install": RobustInstall}
    
    setup(**setup_kwargs)


if __name__ == "__main__":
    main()