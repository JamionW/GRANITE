#!/bin/bash
# GRANITE Installation and Fix Script
# This script fixes the MetricGraph installation issues

set -e  # Exit on any error

echo "🚀 GRANITE Installation and Fix Script"
echo "======================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo "📋 Checking prerequisites..."

if ! command_exists R; then
    echo "❌ R is not installed. Installing R..."
    if command_exists apt-get; then
        sudo apt-get update
        sudo apt-get install -y r-base r-base-dev
    elif command_exists yum; then
        sudo yum install -y R R-devel
    elif command_exists brew; then
        brew install r
    else
        echo "❌ Cannot install R automatically. Please install R manually."
        exit 1
    fi
else
    echo "✅ R is installed"
fi

if ! command_exists python3; then
    echo "❌ Python 3 is not installed"
    exit 1
else
    echo "✅ Python 3 is installed"
fi

# Install Python dependencies if needed
echo "📦 Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "⚠️  requirements.txt not found, installing core dependencies"
    pip install rpy2 numpy pandas torch torch-geometric networkx geopandas
fi

# Install/reinstall GRANITE
echo "📦 Installing GRANITE..."
pip install -e .

# Install R packages manually
echo "📦 Installing R packages..."
Rscript -e "
# Set options
options(repos = c(CRAN = 'https://cloud.r-project.org'))

# Install remotes if needed
if (!requireNamespace('remotes', quietly = TRUE)) {
    install.packages('remotes')
}

# Install Matrix if needed  
if (!requireNamespace('Matrix', quietly = TRUE)) {
    install.packages('Matrix')
}

# Install INLA
if (!requireNamespace('INLA', quietly = TRUE)) {
    install.packages('INLA', 
        repos = c(getOption('repos'), 
                 INLA = 'https://inla.r-inla-download.org/R/stable'))
}

# Install MetricGraph from GitHub
if (!requireNamespace('MetricGraph', quietly = TRUE)) {
    cat('Installing MetricGraph from GitHub...\\n')
    remotes::install_github('davidbolin/MetricGraph', dependencies = TRUE)
}

# Test loading
library(MetricGraph)
library(Matrix) 
library(INLA)

cat('✅ All R packages installed successfully!\\n')
"

# Test the installation
echo "🧪 Testing GRANITE installation..."
python3 -c "
try:
    import rpy2.robjects as ro
    ro.r('library(MetricGraph)')
    print('✅ MetricGraph loads successfully via rpy2')
    
    from granite.metricgraph.interface import MetricGraphInterface
    mg_interface = MetricGraphInterface(verbose=False)
    if mg_interface.mg is not None:
        print('✅ GRANITE MetricGraph interface working')
    else:
        print('❌ GRANITE MetricGraph interface failed')
        exit(1)
        
except Exception as e:
    print(f'❌ Test failed: {e}')
    exit(1)
"

echo ""
echo "🎉 Installation complete!"
echo "Try running: granite --fips 47065000600 --epochs 1"