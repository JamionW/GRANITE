GRANITE Framework
Graph-Refined Accessibility Network for Integrated Transit Equity
A novel framework for Social Vulnerability Index (SVI) disaggregation using Graph Neural Networks integrated with MetricGraph's Whittle-Matérn spatial models.
Overview
GRANITE addresses the critical challenge of disaggregating census tract-level Social Vulnerability Index (SVI) data to address-level predictions while respecting road network topology and transit accessibility patterns. This framework combines:
Graph Neural Networks (GNN) for learning transit accessibility features
MetricGraph for rigorous spatial modeling on networks
Whittle-Matérn fields for uncertainty-aware disaggregation
Key Features
🗺️ Network-Aware: Respects road topology rather than assuming Euclidean distances
🧠 Data-Driven: Learns accessibility patterns from data instead of hand-crafting features
📊 Uncertainty Quantification: Provides confidence intervals for all predictions
🚀 Scalable: Efficient implementation suitable for metropolitan-scale analysis
Installation
Prerequisites
Python 3.8+
R 4.0+ with MetricGraph package
CUDA-capable GPU (optional but recommended)
Setup
Clone the repository:
bash
git clone https://github.com/yourusername/granite.git
cd granite
Create a virtual environment:
bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install Python dependencies:
bash
pip install -r requirements.txt
Install R dependencies:
R
install.packages("MetricGraph")
install.packages("INLA", repos=c(getOption("repos"), INLA="https://inla.r-inla-download.org/R/stable"))
Quick Start
Basic Usage
bash
# Run with default settings (downloads data automatically)
python scripts/run_granite.py

# Use local road network file
python scripts/run_granite.py --roads_file data/tl_2023_47065_roads.shp

# Customize training
python scripts/run_granite.py --epochs 200 --learning_rate 0.005
Python API
python
from granite.disaggregation.pipeline import GRANITEPipeline

# Create pipeline
pipeline = GRANITEPipeline(data_dir='./data', output_dir='./output')

# Run complete analysis
results = pipeline.run(
    roads_file='path/to/roads.shp',  # Optional
    epochs=100,
    visualize=True
)

# Access results
predictions = results['predictions']  # Address-level SVI
gnn_features = results['gnn_features']  # Learned accessibility features
Project Structure
granite/
├── granite/                  # Main package
│   ├── data/                # Data loading and preprocessing
│   │   ├── loaders.py      # Data loading functions
│   │   ├── preprocessors.py # Data preprocessing
│   │   └── generators.py   # Synthetic data generation
│   ├── models/             # GNN models
│   │   ├── gnn.py         # GNN architectures
│   │   └── training.py    # Training utilities
│   ├── metricgraph/        # MetricGraph interface
│   │   ├── interface.py   # R interface
│   │   └── modeling.py    # Spatial modeling
│   ├── disaggregation/     # Main pipeline
│   │   └── pipeline.py    # Disaggregation pipeline
│   └── visualization/      # Plotting functions
│       └── plots.py       # Visualization utilities
├── scripts/                # Executable scripts
│   └── run_granite.py     # Main execution script
├── data/                   # Data directory
├── output/                 # Results directory
└── tests/                  # Unit tests
Data Requirements
GRANITE can automatically download required data, but you can also provide:
Census Tract Geometries: TIGER/Line shapefiles
SVI Data: CDC Social Vulnerability Index CSV
Road Network: TIGER/Line roads shapefile (e.g., tl_2023_47065_roads.shp)
Transit Stops (optional): GTFS stops.txt or shapefile
Address Points (optional): Address point shapefile
Methodology
1. Graph Construction
Convert road network to graph structure
Create both NetworkX and MetricGraph representations
2. GNN Feature Learning
Train GNN to learn spatially-varying SPDE parameters (κ, α, τ)
Use spatial smoothness loss to ensure coherent patterns
Extract accessibility features for each network node
3. Spatial Disaggregation
Use tract-level SVI as observations
Incorporate GNN features as covariates in MetricGraph
Fit Whittle-Matérn model with learned parameters
Predict at address locations with uncertainty
4. Validation
Ensure mass preservation (tract averages match)
Cross-validation on hold-out tracts
Uncertainty calibration checks
Output Files
After running GRANITE, you'll find:
granite_predictions.csv: Address-level SVI predictions with uncertainty
gnn_features.npy: Learned accessibility features
validation_results.csv: Tract-level validation metrics
granite_visualization.png: Summary visualization
Example Results
GRANITE provides:
Point Predictions: SVI score for each address
Uncertainty Bounds: 95% confidence intervals
Accessibility Features: Interpretable transit access patterns
Validation Metrics: MAE, correlation, mass preservation
Citation
If you use GRANITE in your research, please cite:
bibtex
@inproceedings{yourname2026granite,
  title={Graph Neural Network Feature Learning for Transit Accessibility Assessment in Social Vulnerability Index Disaggregation},
  author={Your Name and Collaborators},
  booktitle={Transportation Research Board Annual Meeting},
  year={2026}
}
Contributing
We welcome contributions! Please see CONTRIBUTING.md for guidelines.
License
This project is licensed under the MIT License - see LICENSE file for details.
Acknowledgments
Dr. Mehdi Khaleghian (UTC) for transportation expertise
David Bolin for the MetricGraph R package
CDC for Social Vulnerability Index data
U.S. Census Bureau for TIGER/Line data
Contact
For questions or collaboration opportunities, please contact:
Email: your.email@university.edu
GitHub Issues: Create an issue
Note: This is research software under active development. Please report any issues or bugs.
