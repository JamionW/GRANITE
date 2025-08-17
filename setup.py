#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Setup script for GRANITE with integrated R installation
"""

from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop
import os
import subprocess
import sys

def run_r_setup():
    """Run R setup script during installation"""
    print("🔧 Setting up R environment for GRANITE...")
    
    # Find the setup script
    here = os.path.abspath(os.path.dirname(__file__))
    setup_script = os.path.join(here, 'granite', 'scripts', 'setup_metricgraph.sh')
    
    if not os.path.exists(setup_script):
        print(f"⚠️  R setup script not found at {setup_script}")
        print("⚠️  Please run: ./granite/scripts/setup_metricgraph.sh")
        return False
    
    try:
        # Make script executable
        os.chmod(setup_script, 0o755)
        
        # Run setup script
        result = subprocess.run(['bash', setup_script], 
                              capture_output=True, text=True, 
                              timeout=600)  # 10 minute timeout
        
        if result.returncode == 0:
            print("✅ R environment setup completed successfully")
            return True
        else:
            print(f"❌ R setup failed with code {result.returncode}")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ R setup timed out (took >10 minutes)")
        return False
    except Exception as e:
        print(f"❌ R setup failed: {str(e)}")
        return False

class PostInstallCommand(install):
    """Custom install command that runs R setup"""
    def run(self):
        # Run normal install first
        install.run(self)
        
        # Then run R setup
        if not run_r_setup():
            print("\n⚠️  R setup failed. MetricGraph features may not work.")
            print("💡 You can manually run: ./granite/scripts/setup_metricgraph.sh")

class PostDevelopCommand(develop):
    """Custom develop command that runs R setup"""
    def run(self):
        # Run normal develop first
        develop.run(self)
        
        # Then run R setup
        if not run_r_setup():
            print("\n⚠️  R setup failed. MetricGraph features may not work.")
            print("💡 You can manually run: ./granite/scripts/setup_metricgraph.sh")

# Read version and other metadata
def get_version():
    here = os.path.abspath(os.path.dirname(__file__))
    init_file = os.path.join(here, 'granite', '__init__.py')
    try:
        with open(init_file, encoding='utf-8') as f:
            for line in f:
                if line.startswith('__version__'):
                    return line.split('=')[1].strip().strip('"').strip("'")
    except FileNotFoundError:
        pass
    return "0.1.0"

# Core dependencies (always required)
INSTALL_REQUIRES = [
    # Core scientific computing
    "numpy>=1.21.0",
    "pandas>=1.3.0", 
    "scipy>=1.9.0",
    "scikit-learn>=1.0.0",
    
    # Geospatial libraries
    "geopandas>=0.10.0",
    "shapely>=1.8.0",
    "fiona>=1.8.0",
    "pyproj>=3.0.0",
    "rasterio>=1.2.0",
    
    # Deep learning
    "torch>=1.10.0",
    "torch-geometric>=2.0.0",
    
    # Network analysis
    "networkx>=2.6.0",
    
    # Visualization
    "matplotlib>=3.5.0",
    "seaborn>=0.11.0",
    
    # Configuration and utilities
    "PyYAML>=5.4.0",
    "tqdm>=4.62.0",
    "requests>=2.25.0",
]

# R interface - made optional for initial install
R_DEPENDENCIES = [
    "rpy2>=3.4.0",
]

# Optional dependencies
EXTRAS_REQUIRE = {
    'r': R_DEPENDENCIES,  # For manual R installation
    'dev': [
        "pytest>=6.0",
        "pytest-cov>=2.0", 
        "black>=21.0",
        "flake8>=3.9",
        "mypy>=0.910",
        "pre-commit>=2.15",
    ],
    'jupyter': [
        "jupyter>=1.0.0",
        "ipykernel>=6.0.0",
        "ipywidgets>=7.6.0",
    ],
    'all': R_DEPENDENCIES + [
        "pytest>=6.0", "jupyter>=1.0.0", "ipykernel>=6.0.0",
    ]
}

# Try to install rpy2 by default, but don't fail if R isn't available
try:
    import subprocess
    result = subprocess.run(['R', '--version'], capture_output=True)
    if result.returncode == 0:
        # R is available, include rpy2 in default install
        INSTALL_REQUIRES.extend(R_DEPENDENCIES)
        print("✅ R detected, including rpy2 in installation")
    else:
        print("⚠️  R not detected, rpy2 will be installed during post-install")
except:
    print("⚠️  R not detected, rpy2 will be installed during post-install")

setup(
    name="granite-svi",
    version=get_version(),
    author="Jamion Williams",
    description="Graph-Refined Accessibility Network for Integrated Transit Equity",
    long_description="GNN framework for SVI disaggregation with MetricGraph integration",
    
    packages=find_packages(),
    include_package_data=True,
    
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    python_requires=">=3.8",
    
    # Custom install commands
    cmdclass={
        'install': PostInstallCommand,
        'develop': PostDevelopCommand,
    },
    
    entry_points={
        'console_scripts': [
            'granite=granite.scripts.run_granite:main',
            'granite-setup=granite.scripts.setup_granite:main',  # New setup command
        ],
    },
    
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research", 
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    
    zip_safe=False,
)