# GRANITE: Graph-Refined Accessibility Network for Transportation Equity

GNN-based spatial disaggregation of tract-level Social Vulnerability Index (SVI) to address-level resolution using multi-modal transportation accessibility features.

## Research Question

Can Graph Neural Networks leverage transportation network topology and multi-modal accessibility patterns to disaggregate tract-level Social Vulnerability Index values to address-level resolution, and how does this approach compare to traditional spatial interpolation methods?

## Overview

GRANITE accepts known tract-level SVI values as hard constraints and learns how vulnerability distributes spatially within each tract based on accessibility patterns. The system computes real travel times via OSRM routing to employment, healthcare, and grocery destinations, then uses graph convolution over a road-network-derived spatial graph to produce address-level SVI estimates.

The architecture treats disaggregation as an allocation problem: the tract mean is fixed, and the GNN learns the within-tract distribution.

## Study Area

Hamilton County, Tennessee (FIPS 47065): 87 census tracts, approximately 100,000 addresses.

**Data sources:**
- LEHD Workplace Area Characteristics (employment locations)
- CMS Hospital Compare (healthcare facilities)
- OpenStreetMap (grocery stores)
- CDC Social Vulnerability Index 2020
- TIGER/Line census tract boundaries and road networks
- OSRM routing engine (driving and walking profiles)

## Features

54 features per address, organized in three groups:

- **30 base accessibility features** (10 per destination type: employment, healthcare, grocery). Temporal metrics (min, mean, median travel time), count metrics (destinations reachable within 5/10/15 min), and equity metrics (drive advantage ratio, dispersion, time range, percentile rank).
- **15 modal features** (5 per destination type). Cross-mode comparisons including average time, time standard deviation, access density, equity gap, and car advantage.
- **9 socioeconomic controls.** Tract-level ACS variables: no vehicle, poverty, unemployment, no high school diploma, age 65+, age 17 and under, disability, single parent, minority.

See `docs/FEATURES.md` for the complete feature reference.

## Architecture

Hybrid GCN/GAT graph neural network (3 layers, ~50K parameters):

1. Feature encoder: 54 -> 64 dimensions (2-layer MLP)
2. GCNConv: spatial aggregation from road network neighbors
3. GATConv: attention-weighted neighbor aggregation (2 heads)
4. GCNConv: final spatial aggregation with dimensionality reduction
5. Prediction head: SVI output with sigmoid activation

Training uses a multi-component loss: constraint satisfaction (tract mean preservation), spatial smoothness, and prediction variance. Post-training additive correction enforces exact constraint satisfaction.

## Installation

```bash
# clone and install
git clone https://github.com/JamionW/GRANITE.git
cd GRANITE
pip install -e .

# OSRM setup (required for accessibility computation)
# driving profile on port 5000, walking profile on port 5001
docker run -t -i -p 5000:5000 osrm/osrm-backend osrm-routed --algorithm mld /data/tennessee-latest.osrm
docker run -t -i -p 5001:5001 osrm/osrm-backend osrm-routed --algorithm mld /data/tennessee-latest-foot.osrm
```

**Requirements:** Python 3.8+, PyTorch, PyTorch Geometric, GeoPandas. See `requirements.txt` for full dependencies.

## Usage

```bash
# single-tract analysis
granite --fips 47065010100 --verbose

# multi-tract (target + neighbors for training diversity)
granite --fips 47065010100 --neighbor-tracts 3 --verbose

# custom epochs and output
granite --fips 47065010100 --epochs 200 --output ./results

# skip baseline comparisons for faster iteration
granite --fips 47065010100 --skip-baselines
```

## Project Structure

```
granite/
 data/ # data loading, graph construction
 disaggregation/ # main pipeline orchestration
 evaluation/ # ablation, baselines, spatial diagnostics
 features/ # modal accessibility computation
 models/ # GNN architecture, mixture of experts
 routing/ # OSRM interface
 scripts/ # CLI entry points
 validation/ # block group validation
 visualization/ # research plots
config.yaml # system configuration
docs/ # feature reference, architecture notes
```

## Configuration

All parameters live in `config.yaml`. CLI arguments override config values. Key sections: geographic scope, accessibility computation, OSRM routing endpoints, model architecture, training hyperparameters, and validation settings.

## Caching

First runs compute OSRM travel times for all origin/destination pairs (can take 30+ minutes for multi-tract runs). Subsequent runs with the same addresses and destinations hit the cache and complete in under 5 minutes. Cache location: `./granite_cache/`.

## Author

Jamion Williams, PhD Candidate, University of Tennessee at Chattanooga

## License

Research use. See repository for details.