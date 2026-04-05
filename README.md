# GRANITE: Graph-Refined Accessibility Network for Transportation Equity

GNN-driven spatial disaggregation of the CDC's tract-level Social Vulnerability Index (SVI) to address-level resolution using multi-modal transportation accessibility features and address-level property attributes.

## Research Question

Can Graph Neural Networks leverage transportation network topology and multi-modal accessibility patterns to disaggregate tract-level Social Vulnerability Index values to address-level resolution, and how does this approach compare to traditional spatial interpolation methods?

The evolved core question: Under what conditions does hard aggregate constraint enforcement improve or degrade the accuracy of learned spatial disaggregation models, and what feature classes survive constraint correction?

## Overview

GRANITE accepts known tract-level SVI values as hard constraints and learns how vulnerability distributes spatially within each tract based on accessibility patterns and address-level property characteristics. The system computes real travel times via OSRM routing to employment, healthcare, and grocery destinations, then uses graph convolution over a road-network-derived spatial graph to produce address-level SVI estimates.

Disaggregation is treated as an allocation problem: the tract mean is fixed, and the GNN learns the within-tract distribution.

## Current Study Area

Hamilton County, Tennessee (FIPS 47065): 85 tracts with valid SVI data, 102,647 address points (66 tracts with sufficient address coverage for analysis).

**Data sources:**
- LEHD Workplace Area Characteristics 2021 (259 employment locations)
- CMS Hospital Compare (12 healthcare facilities)
- OpenStreetMap (181 grocery stores)
- CDC Social Vulnerability Index 2020
- TIGER/Line census tract boundaries and road networks
- OSRM routing engine (driving and walking profiles)
- Hamilton County tax assessor parcel data (appraised value, land use, property type, acreage)
- Microsoft Building Footprints (footprint area, vertex count)
- FEMA National Flood Hazard Layer (SFHA status, flood zone classification)
- OpenStreetMap building tags (building type, levels)

## Features

70 features per address (when all address-level data is available), organized in four groups:

**30 base accessibility features** (10 per destination type: employment, healthcare, grocery). Temporal metrics (min, mean, median travel time), count metrics (destinations reachable within 5/10/15 min), and equity metrics (drive advantage ratio, dispersion, time range, percentile rank).

**15 modal features** (5 per destination type). Vehicle-ownership-weighted accessibility: transit dependence, car-effective access, walk-effective access, modal access gap, and forced walk burden.

**9 socioeconomic controls.** Tract-level ACS variables: no vehicle, poverty, unemployment, no high school diploma, uninsured, mobile homes, crowded housing, population, housing units.

**16 address-level features.** Building footprint (log area, vertex count), FEMA flood zone (SFHA binary, zone classification), OSM building type (residential binary), parcel data (log appraised value, build-to-land ratio, log acres), land use code (4-category one-hot), property type (top-5 one-hot).

See `docs/FEATURES.md` for the complete feature reference.

## Architecture

Hybrid GCN/GAT graph neural network (3 layers, ~50K parameters):

1. Optional context-gated feature modulation (socioeconomic context gates accessibility features)
2. Feature encoder: input dim to 64 dimensions (2-layer MLP)
3. GCNConv: spatial aggregation from road network neighbors
4. GATConv: attention-weighted neighbor aggregation (2 heads)
5. GCNConv: final spatial aggregation with dimensionality reduction
6. Prediction head: SVI output with sigmoid activation

Training uses a multi-component loss: constraint satisfaction (tract mean preservation, weight configurable), spatial variation encouragement, bounds enforcement, and accessibility consistency. Post-training additive correction is available via config but currently disabled for the constraint ablation experiment.

## Installation

```bash
# clone and install
git clone https://github.com/JamionW/GRANITE.git
cd GRANITE
pip install -e .

# OSRM setup (required for accessibility computation)
# driving profile on port 5000, walking profile on port 5001
bash granite/scripts/setup_osrm.sh
bash granite/scripts/process_driving_profile.sh
bash granite/scripts/process_foot_profile.sh

# Fetch LEHD, Healthcare, and Grocery data
bash granite/scripts/setup_data.sh

# Start OSRM servers (add to devcontainer.json for auto-start)
bash granite/scripts/start_osrm.sh

# Acquire address-level features (building footprints, flood zones, parcel data)
python granite/scripts/acquire_address_features.py
# Then run the spatial join:
python data/raw/address_features/join_to_addresses.py
```

**Requirements:** Python 3.8+, PyTorch, PyTorch Geometric, GeoPandas. See `requirements.txt` for full dependencies.

## Usage

```bash
# single-tract analysis
granite --fips 47065000600 --verbose

# multi-tract (target + neighbors for training diversity)
granite --fips 47065000600 --neighbor-tracts 3 --verbose

# unconstrained training (constraint ablation)
granite --fips 47065000600 --no-constraints --verbose

# custom epochs and output
granite --fips 47065000600 --epochs 200 --output ./results

# skip baseline comparisons for faster iteration
granite --fips 47065000600 --skip-baselines

# skip feature importance for faster iteration
granite --fips 47065000600 --skip-importance

# global MoE training with curated train/test split
granite --global-training --verbose
```

## Project Structure

```
granite/
    data/                   # data loading, graph construction, address feature join
    disaggregation/         # main pipeline orchestration
    evaluation/             # baselines (IDW, kriging, naive), spatial diagnostics
    features/               # modal accessibility computation
    models/                 # GNN architecture, mixture of experts
    routing/                # OSRM interface
    scripts/                # CLI entry, address feature acquisition
    validation/             # block group validation
    visualization/          # research plots, disaggregation comparison
config.yaml                 # system configuration
docs/                       # feature reference, architecture notes
graveyard/                  # deprecated code (preserved for recovery)
output/                     # visualizations and raw data output
```

## Configuration

All parameters live in `config.yaml`. CLI arguments override config values. Key sections: geographic scope, accessibility computation, OSRM routing endpoints, model architecture, training hyperparameters (including constraint weight and enforcement toggle), and validation settings.

## Caching

First runs compute OSRM travel times for all origin/destination pairs (can take 30+ minutes for multi-tract runs). Subsequent runs with the same addresses and destinations hit the cache and complete in under 5 minutes. Address-level features (parcel, building, flood) are not cached; they are re-extracted from the CSV on each run. Cache location: `./granite_cache/`.

## Key Research Findings

Accessibility features (OSRM-based travel times) proved counterproductive for within-tract disaggregation through rigorous ablation; raw spatial coordinates outperformed them. Constraint enforcement dominates learning: post-hoc correction to satisfy tract-level aggregate constraints does most of the predictive work. Features that correlate strongly with SVI at tract level (e.g., grocery count within 3km) actively hurt address-level disaggregation due to ecological fallacy effects.

## Author

Jamion Williams, PhD Candidate, University of Tennessee at Chattanooga

## License

Research use. See repository for details.