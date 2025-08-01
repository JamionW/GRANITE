#!/bin/bash
# MetricGraph Setup Script for GitHub Codespace
# Run this script to install all R dependencies for GRANITE

echo "🔧 Setting up MetricGraph R environment in GitHub Codespace..."

# 1. Update system and install R dependencies
sudo apt-get update -y
sudo apt-get install -y \
    r-base \
    r-base-dev \
    libglu1-mesa-dev \
    libeigen3-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libgdal-dev \
    libproj-dev \
    libgeos-dev \
    libudunits2-dev

echo "✅ System dependencies installed"

# 2. Create R script for package installation
cat > /tmp/install_metricgraph.R << 'EOF'
# MetricGraph R Package Installation Script
cat("🔧 Installing R packages for MetricGraph...\n")

# Set CRAN mirror
options(repos = c(CRAN = "https://cran.r-project.org/"))

# Install base dependencies first
base_packages <- c(
    "remotes", 
    "devtools", 
    "Matrix", 
    "igraph", 
    "ggplot2", 
    "dplyr", 
    "tidyr", 
    "R6", 
    "Rcpp",
    "RANN",
    "broom",
    "zoo",
    "ggnewscale",
    "lifecycle",
    "magrittr"
)

cat("📦 Installing base dependencies...\n")
install.packages(base_packages, dependencies = TRUE)

# Add INLA repository (required for MetricGraph)
cat("📦 Adding INLA repository...\n")
options(repos = c(
    INLA = "https://inla.r-inla-download.org/R/testing",
    CRAN = "https://cran.r-project.org/"
))

# Install INLA
cat("📦 Installing INLA...\n")
install.packages("INLA", dependencies = TRUE)

# Install rSPDE from GitHub (required dependency)
cat("📦 Installing rSPDE from GitHub...\n")
remotes::install_github("davidbolin/rSPDE", ref = "devel")

# Install MetricGraph from GitHub (stable version)
cat("📦 Installing MetricGraph from GitHub...\n")
remotes::install_github("davidbolin/MetricGraph", ref = "stable")

# Verify installation
cat("🔍 Verifying installation...\n")
tryCatch({
    library(MetricGraph)
    library(INLA)
    library(rSPDE)
    cat("✅ All packages loaded successfully!\n")
    cat("MetricGraph version:", packageVersion("MetricGraph"), "\n")
    cat("INLA version:", packageVersion("INLA"), "\n")
    cat("rSPDE version:", packageVersion("rSPDE"), "\n")
}, error = function(e) {
    cat("❌ Error loading packages:", e$message, "\n")
    quit(status = 1)
})

cat("🎉 MetricGraph setup complete!\n")
EOF

# 3. Run R installation script
echo "📦 Installing R packages (this may take several minutes)..."
Rscript /tmp/install_metricgraph.R

# 4. Test the installation with Python
echo "🐍 Testing rpy2 integration..."
python3 << 'EOF'
import warnings
warnings.filterwarnings('ignore')

try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    import rpy2.robjects.packages as rpackages
    
    # Test MetricGraph import
    print("Testing MetricGraph import...")
    mg = rpackages.importr('MetricGraph')
    print("✅ MetricGraph imported successfully")
    
    # Test INLA import  
    print("Testing INLA import...")
    inla = rpackages.importr('INLA')
    print("✅ INLA imported successfully")
    
    # Test basic functionality
    print("Testing basic MetricGraph functionality...")
    ro.r('''
    library(MetricGraph)
    # Create simple test graph
    edge_df <- data.frame(from = c(1, 2), to = c(2, 3))
    vertex_df <- data.frame(x = c(0, 1, 2), y = c(0, 0, 0))
    test_graph <- metric_graph$new(edges = edge_df, V = vertex_df)
    cat("✅ MetricGraph test successful\\n")
    ''')
    
    print("🎉 All tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Please check the R package installation")
EOF

echo "🔧 Creating R environment file..."
cat > ~/.Renviron << 'EOF'
R_LIBS_USER="~/R/library"
R_MAX_NUM_DLLS=150
EOF

echo "✅ MetricGraph setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Restart your Python kernel if running in a notebook"
echo "2. Your GRANITE MetricGraph interface should now work"
echo "3. Test with: python -c 'from granite.metricgraph.interface import MetricGraphInterface; mg = MetricGraphInterface()'"