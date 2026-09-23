# GRANITE

Constraint-preserving graph neural network for spatial disaggregation of
the CDC Social Vulnerability Index from census tract resolution to individual
addresses in Hamilton County, Tennessee (FIPS 47065).

## Primary research question

Under what conditions does a constraint-preserving graph neural network recover
the within-tract variation a tract average hides, and by what mechanism does it
fail when it does not; and which feature classes survive constraint correction
across architectures?

## Contribution

1. Constraint-preserving GNN architecture treating tract-level SVI as an
   aggregate constraint rather than a prediction target, enforced as a soft
   training penalty plus exact mean reconciliation at inference.
2. 73-feature address-level input: parcel attributes, Microsoft building
   footprints, FEMA flood zones, NLCD land cover (impervious, canopy, land
   cover class), multi-modal accessibility, socioeconomic controls.
3. A synthetic recovery testbed and a ceiling-referenced validation protocol
   for disaggregation where no target-resolution truth exists, with dasymetric
   and pycnophylactic interpolation as the classical comparison baselines.
4. Dual-architecture comparison (GraphSAGE vs. GCN-GAT) showing constraint
   enforcement interacts with model inductive bias to determine which
   features survive. The observed feature-survival asymmetry (GraphSAGE retains
   none at any screening threshold; the GCN-GAT hybrid retains a threshold-dependent
   handful, between five and fourteen features) is treated as architecture-dependent
   pending the Study 4 artifact controls.

## Framing note

The acronym GRANITE is a project codename. Its original expansion
("Graph-Refined Accessibility Network for Transportation Equity") is
retired; do not reproduce it in documentation or comments. Accessibility
is one of several feature classes, not the primary driver of disaggregation.

PDFM (Agarwal et al., 2024) is positioned as the unconstrained complement.

## Empirical framing

IDW and kriging were retired to `graveyard/` (2026-04-18) as a degenerate
proximity floor; the legacy r=0.558/0.469 pair is retired with them and must
not be cited (unknown holdout, legacy context; see
`experiments/audits/baseline_metric_provenance.md`). The current empirical
spine: pooled block-group parity is a tie (framework 0.769, dasymetric 0.802,
pycnophylactic 0.768) because the tract-mean constraint settles aggregate
agreement before any learning; methods separate only within tracts, where the
20-tract comparison is descriptive and underpowered. On the coordinate-only
synthetic grid, within-tract recovery is near zero against a supervised ceiling
of r=0.13, while output coherence stays high (Moran's I ~0.94), which locates a
boundary rather than a bare null. The null-as-boundary contribution rests on
the synthetic testbed and the ceiling-referenced protocol, not on the retired
baseline comparison.

## Critical constraint: aggregate preservation

Address-level predictions must reconcile to the known tract-level SVI value. As shipped this is enforced as a soft training penalty on tract-mean deviation plus an exact mean reconciliation at inference (`constraint_mode='soft'`, `apply_post_correction=True`), not as a hard architectural constraint and not as an ordinary regularization term. The aggregate-preservation logic is the methodological core of the framework: do not remove, weaken, or reroute around it. Note that the inference-time reconciliation is a per-tract additive shift followed by a clip to the unit interval (`pipeline.py`, iterated bounded projection): rank-preserving within a tract wherever the clip does not bind, so on interior tracts it cannot change the primary within-tract correlation, but at tracts whose mean lies near 0 or 1 the clip can tie boundary addresses and alter within-tract order (the primary metric is reported on the pre-clip ordering).

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

73 features per address: 30 base accessibility features, 15 modal features (now per-address from OSRM drive/walk times), 9 socioeconomic features, 19 address-level attributes (building, parcel, flood, NLCD). Of these, 9 are tract-level constants (socioeconomic controls).

## Code conventions

- No new function names for updates; modify functions in place.
- Deprecated code goes to `graveyard/` with a `.old` extension, not deletion.
- Comments are lowercase, minimal, and descriptive. No emojis or hyperbole.
- No em dashes in any generated text or comments.
- Caching is load-bearing; any change that invalidates cache keys should be flagged explicitly.

## Validation ground truth

Block-group-level ACS-derived SVI (12 variables across 4 CDC SVI themes). This is derived independently from ACS components, not from pipeline predictions. Do not substitute IDW-interpolated values as ground truth.

National BG SVI data (242,335 block groups, 239,346 with complete SVI) is cached at `data/processed/national_bg_acs_raw.csv` and `data/processed/national_bg_svi.csv`. These are fetched from Census ACS 5-year estimates and ranked nationally. Use `svi_ranking_scope='national'` in `BlockGroupLoader.get_block_groups_with_demographics()` to rank Hamilton County BGs against the full US distribution instead of county-only.

## Key result reference points

Current (proposal-aligned; verify against committed artifacts before citing):

- Pooled block-group parity (69 BGs): framework 0.769, dasymetric 0.802,
  pycnophylactic 0.768; bootstrap CIs overlap (constraint pins aggregate agreement)
- Per-tract median r (19 tracts): framework 0.390, dasymetric 0.787,
  pycnophylactic 0.208; paired diff median -0.121, 95% CI [-0.536, 0.108]
- Coordinate-only synthetic ceiling: r = 0.13 (0.23 under strong autocorrelation)
- Output coherence on the coordinate grid: Moran's I median ~0.94
- Power (framework vs dasymetric): mean d 0.29, ~22% at n=19, ~76% at n=85
  (reproduce with `scripts/power_analysis_parity.py`)

Retired, do not cite: IDW r = 0.558, GRANITE r = 0.469 (legacy holdout, see
`experiments/audits/baseline_metric_provenance.md`).

## Session logging

After any session that changes pipeline logic, data loading, or feature extraction, append a brief summary to `SESSION_LOG.md` in the repo root. Include: date, files changed, what changed and why, any cache invalidation notes.