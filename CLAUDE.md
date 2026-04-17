# GRANITE

Constraint-preserving graph neural network for spatial disaggregation of
the CDC Social Vulnerability Index from census tract resolution to individual
addresses in Hamilton County, Tennessee (FIPS 47065).

## Primary research question

Under what conditions does hard aggregate constraint enforcement improve or
degrade the accuracy of learned spatial disaggregation models, and which
feature classes survive constraint correction?

## Contribution

1. Constraint-preserving GNN architecture treating tract-level SVI as a
   hard aggregate constraint rather than a prediction target.
2. 73-feature address-level input: parcel attributes, Microsoft building
   footprints, FEMA flood zones, NLCD land cover (impervious, canopy, land
   cover class), multi-modal accessibility, socioeconomic controls.
3. Evaluation against IDW and kriging mapping boundary conditions of learned
   disaggregation versus proximity-weighted baselines.
4. Dual-architecture comparison (GraphSAGE vs. GCN-GAT) showing constraint
   enforcement interacts with model inductive bias to determine which
   features survive. GraphSAGE currently stronger.

## Framing note

The acronym GRANITE is a project codename. Its original expansion
("Graph-Refined Accessibility Network for Transportation Equity") is
retired; do not reproduce it in documentation or comments. Accessibility
is one of several feature classes, not the primary driver of disaggregation.

PDFM (Agarwal et al., 2024) is positioned as the unconstrained complement.

## Empirical framing

IDW outperforms GRANITE on block-group validation (r=0.558 vs r=0.469).
This null result is a defensible dissertation contribution establishing
boundary conditions under which constrained GNN disaggregation collapses
toward proximity-weighted allocation.

## Critical constraint: aggregate preservation

Address-level predictions must average back to the known tract-level SVI value. This is a hard constraint, not a regularization term. Do not remove, weaken, or reroute around this logic. It is the methodological core of the framework.

## Repo structure

```
granite/
  models/gnn.py                   # GNN architecture (AccessibilitySVIGNN)
  disaggregation/pipeline.py      # Main pipeline
  features/enhanced_accessibility.py
  features/osrm_router.py         # OSRM interface (driving: 5000, walking: 5001)
  data/loaders.py
  evaluation/validators.py
scripts/run_granite.py            # CLI entry point
config.yaml
data/raw/                         # Not in git; includes chattanooga.geojson
granite_cache/                    # Not in git; OSRM routes cached here
graveyard/                        # Deprecated code; move here instead of deleting
```

## Active branch

`main`

## Environment

GitHub Codespaces. All paths relative to `/workspaces/GRANITE/`. Do not use absolute paths or reference `/mnt/` directories.

## OSRM routing servers

Two local Docker containers:
- Driving: `localhost:5000`
- Walking: `localhost:5001`

OSRM routing is the dominant runtime cost (~96% of a cold run). The cache at `granite_cache/` reduces subsequent runs from ~76 minutes to under 5 minutes. Preserve cache keys when modifying feature or routing logic.

## Running the pipeline

```bash
# Standard multi-tract run
granite --fips 47065000600 --neighbor-tracts 3 --epochs 200 --verbose

# Debug (small, fast)
granite --fips 47065000600 --epochs 50 --verbose

# Skip cache (debugging only)
granite --fips 47065000600 --no-cache
```

## Feature matrix

72+ features per address: 30 base accessibility features, 15 modal features (now per-address from OSRM drive/walk times), 9 socioeconomic features, 18+ address-level attributes (building, parcel, flood, NLCD). Of these, 9 are tract-level constants (socioeconomic controls).

Accessibility features (travel times, destination counts, modal gaps) show near-zero within-tract variance. Raw spatial coordinates (r~0.67) dramatically outperform accessibility features (r~0.03) as predictors. This is a validated empirical finding; do not treat it as a data quality problem.

## Code conventions

- No new function names for updates; modify functions in place.
- Deprecated code goes to `graveyard/` with a `.old` extension, not deletion.
- Comments are lowercase, minimal, and descriptive. No emojis or hyperbole.
- No em dashes in any generated text or comments.
- Caching is load-bearing; any change that invalidates cache keys should be flagged explicitly.

## Validation ground truth

Block-group-level ACS-derived SVI (11 variables across 4 CDC SVI themes). This is derived independently from ACS components, not from pipeline predictions. Do not substitute IDW-interpolated values as ground truth.

National BG SVI data (242,335 block groups, 239,346 with complete SVI) is cached at `data/processed/national_bg_acs_raw.csv` and `data/processed/national_bg_svi.csv`. These are fetched from Census ACS 5-year estimates and ranked nationally. Use `svi_ranking_scope='national'` in `BlockGroupLoader.get_block_groups_with_demographics()` to rank Hamilton County BGs against the full US distribution instead of county-only.

## Key result reference points

- IDW block-group correlation: r = 0.558
- GRANITE block-group correlation: r = 0.469
- Spatial coordinates alone: r ~ 0.67
- Accessibility features alone: r ~ 0.03

## Session logging

After any session that changes pipeline logic, data loading, or feature extraction, append a brief summary to `SESSION_LOG.md` in the repo root. Include: date, files changed, what changed and why, any cache invalidation notes.