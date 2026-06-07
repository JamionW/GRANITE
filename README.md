# GRANITE

Constraint-preserving graph neural network for spatial disaggregation of
the CDC Social Vulnerability Index from census tract resolution to individual
addresses in Hamilton County, Tennessee (FIPS 47065).

First-time setup: see [STARTUP.md](STARTUP.md).

## Research Question

Under what conditions does hard aggregate constraint enforcement improve or
degrade the accuracy of learned spatial disaggregation models, and which
feature classes survive constraint correction?

## Overview

GRANITE accepts known tract-level SVI values as hard constraints and learns how vulnerability distributes spatially within each tract based on 73 address-level features (canonical full configuration): parcel attributes, building footprints, flood zones, land cover, multi-modal accessibility, and socioeconomic controls. Graph convolution over a road-network-derived spatial graph produces address-level SVI estimates.

Disaggregation is treated as an allocation problem: the tract mean is fixed, and the GNN learns the within-tract distribution.

## Current Study Area

Hamilton County, Tennessee (FIPS 47065): 85 tracts with valid SVI data, 102,647 address points (66 tracts with sufficient address coverage for analysis).

**Data sources:**
- LEHD Workplace Area Characteristics 2021 (1,329 employment locations)
- CMS Hospital Compare (12 healthcare facilities)
- OpenStreetMap (189 grocery stores)
- CDC Social Vulnerability Index 2020
- TIGER/Line census tract boundaries and road networks
- OSRM routing engine (driving and walking profiles)
- Hamilton County tax assessor parcel data (appraised value, land use, property type, acreage)
- Microsoft Building Footprints (footprint area, vertex count)
- FEMA National Flood Hazard Layer (SFHA status)
- OpenStreetMap building tags (building type)
- USGS National Land Cover Database (land cover class, impervious surface, tree canopy)

## Features

73 features per address in the canonical full configuration, organized in four groups:

**30 base accessibility features** (10 per destination type: employment, healthcare, grocery). Temporal metrics (min, mean, median travel time), count metrics (destinations reachable within 5/10/15 min), and equity metrics (drive advantage ratio, dispersion, time range, percentile rank).

**15 modal features** (5 per destination type). Computed per address from OSRM driving and walking travel times: avg time to nearest destination (mean of drive and walk), time std (mode disparity), access density (destinations reachable within 10 min by either mode), equity gap (|walk - drive| to nearest), and car advantage (walk/drive ratio to nearest).

**9 socioeconomic controls.** Tract-level ACS variables from CDC SVI: no vehicle (EP_NOVEH), poverty (EP_POV150), unemployment (EP_UNEMP), no high school diploma (EP_NOHSDP), uninsured (EP_UNINSUR), mobile homes (EP_MOBILE), crowded housing (EP_CROWD), population (E_TOTPOP), housing units (E_HU). Tract-level constants.

**19 address-level features** (canonical full configuration; count is lower when source columns are absent). Building footprint (log area, vertex count), FEMA flood zone (SFHA binary), OSM building type (residential binary), parcel data (log appraised value, build-to-land ratio, log acres), land use code (4-category one-hot), property type (5-category one-hot, all columns emitted even if a type is absent from a tract), NLCD land cover (classification code, impervious surface %, tree canopy %). The four groups sum to 30+15+9+19=73 in the canonical run.

See `docs/FEATURES.md` for the complete feature reference.

## Architecture

Two GNN architectures, selectable via `--architecture {gcn_gat|sage}`:

**gcn_gat (default):** Hybrid GCN/GAT graph neural network (3 convolution layers, ~50K parameters):

1. Optional context-gated feature modulation (socioeconomic context gates accessibility features)
2. Feature encoder: input dim to 64 dimensions (2-layer MLP)
3. GCNConv: spatial aggregation from road network neighbors
4. GATConv: attention-weighted neighbor aggregation (2 heads)
5. GCNConv: final spatial aggregation with dimensionality reduction
6. Prediction head: SVI output with sigmoid activation

**sage:** GraphSAGE variant. Same encoder, context gating, and prediction heads, but replaces the GCN+GAT+GCN convolution stack with three SAGEConv layers with batch normalization. Uses neighborhood sampling instead of fixed graph structure.

Training uses a multi-component loss: constraint satisfaction (tract mean preservation, weight configurable), optional block group mean constraints (intermediate supervision between tract and address level, weight configurable via `bg_constraint_weight`), spatial variation encouragement, bounds enforcement, and accessibility consistency. Block group SVI targets are computed independently from ACS block-group-level estimates via CDC percentile methodology, not from disaggregated tract data. Post-training additive correction is available via config but currently disabled for the constraint ablation experiment.

## Baselines

The headline baseline is **Dasymetric disaggregation** (single-attribute, NLCD impervious surface ancillary, following Nguyen et al. 2021). It allocates tract SVI proportionally to impervious surface percentage, satisfying the aggregate constraint by construction. The secondary mass-preserving baseline is **Pycnophylactic disaggregation** (Tobler 1979), which iteratively smooths the predicted surface while preserving tract totals.

IDW and Kriging are retired to `graveyard/` and are not used in the current pipeline. Under single-centroid interpolation (one point per tract), point interpolation methods collapse to the tract mean and contribute no within-tract signal; they function as a degenerate proximity floor rather than a spatial disaggregation method. They remain in `graveyard/` as frozen artifacts for ablation runs that recorded them.

The primary validation metric is **pooled BG r**: Pearson r between predicted block-group-mean SVI and nationally-ranked ACS block-group SVI, pooled across all n20 evaluation tracts (69 qualifying block groups, min 10 addresses each). This metric is grounded in `data/results/m0_n20_svi_parity/aggregate.csv`.

The **per-tract BG r** is a secondary diagnostic: the same correlation computed independently within each tract's block groups. It isolates within-tract allocation skill (pooled BG r is dominated by between-tract variance from tract-mean preservation) and is the metric on which Dasymetric and GRANITE visibly separate, though with wide confidence intervals due to small per-tract BG counts (typically 2-5 qualifying block groups per tract).

## Installation

```bash
# clone and install
git clone https://github.com/JamionW/GRANITE.git
cd GRANITE
pip install -r requirements.txt
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

**Requirements:** Python 3.11, PyTorch (CPU build), PyTorch Geometric, GeoPandas. See `requirements.txt` for full dependencies.

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
    evaluation/             # baselines (Dasymetric, Pycnophylactic), spatial diagnostics; IDW/Kriging retired to graveyard/
    features/               # modal accessibility computation
    models/                 # GNN architecture, mixture of experts
    routing/                # OSRM interface
    scripts/                # CLI entry, address feature acquisition, data setup
    validation/             # block group validation (post-hoc, independent of training constraints)
    visualization/          # research plots, disaggregation comparison
config.yaml                 # system configuration
docs/                       # feature reference, architecture notes
graveyard/                  # deprecated code (preserved for recovery)
output/                     # visualizations and raw data output
```

## Configuration

All parameters live in `config.yaml`. CLI arguments override config values. Key sections: geographic scope, accessibility computation, OSRM routing endpoints, model architecture, training hyperparameters (including constraint weight and enforcement toggle), and validation settings.

`feature_mode` controls feature matrix substitution for ablation experiments. Default `'full'` passes the normalized 73-feature matrix unchanged. Other values: `'coordinates_only'` (z-scored lat/lon only), `'random_noise'` (i.i.d. Gaussian), `'coords_plus_noise'` (coordinates + noise). Used by `scripts/coord_artifact_experiment.py` to test whether coordinate information alone drives GNN predictions.

## Caching

First runs compute OSRM travel times for all origin/destination pairs (can take 30+ minutes for multi-tract runs). Subsequent runs with the same addresses and destinations hit the cache and complete in under 5 minutes. Address-level features (parcel, building, flood, socioeconomic) are not cached; they are re-extracted on each run. The cache supports version tags for invalidation when destination data changes, and a `cache.invalidate(older_than_days=N)` API for TTL-based cleanup. Cache location: `./granite_cache/`.

## Key Research Findings

Ablation results indicate that accessibility features (OSRM-based travel times) may be counterproductive for within-tract disaggregation, with raw spatial coordinates outperforming them in initial runs. A confirmatory coordinate-artifact re-run is pending before this is treated as a settled finding. Constraint enforcement dominates learning: post-hoc correction to satisfy tract-level aggregate constraints does most of the predictive work. Features that correlate strongly with SVI at tract level (e.g., employment count reachable within 10 minutes) appear to hurt address-level disaggregation in ablation, consistent with ecological fallacy effects, though this direction also awaits confirmation.

## Author

Jamion Williams, PhD Candidate, University of Tennessee at Chattanooga

## License

Research use. See repository for details.