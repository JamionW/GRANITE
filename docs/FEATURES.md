# GRANITE Feature Reference

Complete inventory of features available to the GNN, organized by implementation status and source.

## Implemented Features (54 total)

### Base Accessibility (30 features: 10 per destination type)

Computed per address from real OSRM travel times to each destination type (employment, healthcare, grocery).

**Temporal (3 per type):**

| # | Feature | Formula | Expected SVI Correlation |
|---|---------|---------|--------------------------|
| 1 | `{type}_min_time` | min driving time to any destination | Positive |
| 2 | `{type}_mean_time` | mean driving time to all destinations | Positive |
| 3 | `{type}_median_time` | median driving time | Positive |

**Count (3 per type):**

| # | Feature | Formula | Expected SVI Correlation |
|---|---------|---------|--------------------------|
| 4 | `{type}_count_5min` | destinations reachable in 5 min driving | Negative |
| 5 | `{type}_count_10min` | destinations reachable in 10 min driving | Negative |
| 6 | `{type}_count_15min` | destinations reachable in 15 min driving | Negative |

**Equity/Mode (4 per type):**

| # | Feature | Formula | Expected SVI Correlation |
|---|---------|---------|--------------------------|
| 7 | `{type}_drive_advantage` | mean walk time / mean drive time | Positive |
| 8 | `{type}_dispersion` | CV of drive times (std/mean) | Positive |
| 9 | `{type}_time_range` | max drive time - min drive time | Positive |
| 10 | `{type}_percentile` | rank percentile within tract | Positive |

**Destination types and data sources:**
- `employment`: LEHD Workplace Area Characteristics 2021 (259 locations in Hamilton County)
- `healthcare`: CMS Hospital Compare (12 facilities)
- `grocery`: OpenStreetMap supermarket tags (181 stores)

### Modal Features (15 features: 5 per destination type)

Cross-mode comparisons computed from driving and walking travel times.

| # | Feature | Description |
|---|---------|-------------|
| 1 | `modal_{type}_avg_time` | Average across both modes |
| 2 | `modal_{type}_time_std` | Standard deviation across modes |
| 3 | `modal_{type}_access_density` | Destination density metric |
| 4 | `modal_{type}_equity_gap` | Walk time minus drive time gap |
| 5 | `modal_{type}_car_advantage` | Ratio of walk to drive accessibility |

**Known limitation:** Modal features are currently computed at tract level. All addresses in a tract receive identical modal values, reducing within-tract variation.

### Socioeconomic Controls (9 features)

Tract-level ACS variables from CDC SVI, applied uniformly to all addresses within a tract.

| # | Feature | ACS Source | Description |
|---|---------|-----------|-------------|
| 1 | `no_vehicle` | E_NOVEH | Households without vehicle access |
| 2 | `poverty` | E_POV150 | Population below 150% poverty line |
| 3 | `unemployment` | E_UNEMP | Unemployed population |
| 4 | `no_highschool` | E_NOHSDP | No high school diploma |
| 5 | `age65_plus` | E_AGE65 | Population 65 and older |
| 6 | `age17_under` | E_AGE17 | Population under 17 |
| 7 | `disability` | E_DISABL | Population with disabilities |
| 8 | `single_parent` | E_SNGPNT | Single-parent households |
| 9 | `minority` | E_MINRTY | Minority population |

**Known limitation:** These are tract-level aggregates. They do not vary within a tract and therefore cannot drive within-tract disaggregation on their own. Their role is to provide socioeconomic context to the GNN.

## Candidate Features (Not Yet Implemented)

Identified through literature review and feature pipeline analysis as potential improvements.

### Transit Accessibility (via GTFS)
- Transit travel times to each destination type (requires CARTA GTFS feed)
- Transit frequency and coverage metrics
- Transit/drive ratio per address
- Walk-to-transit time (distance to nearest stop)

### Temporal Variation
- Peak vs. off-peak accessibility differences
- Morning commute vs. midday access patterns
- Weekend vs. weekday accessibility

### Interaction Terms
- `accessibility x no_vehicle`: effective accessibility given vehicle ownership
- `drive_advantage x poverty`: mode dependency weighted by economic vulnerability
- Vehicle-weighted accessibility metrics (weight travel times by tract vehicle ownership rates)

### Address-Level Refinements
- Compute `drive_advantage` per address instead of per tract
- Housing type indicators (from parcel data if available)
- Distance to nearest transit stop per address
- Land use classification per address (NLCD data available in `data/`)

## Feature Quality Notes

From validation runs on 4-tract configurations:

- Time/count anti-correlation: r = -0.956 (correct; shorter times correlate with higher counts)
- 73.3% of testable features show expected correlation directions with SVI
- `dispersion` features show inverse correlations (expected positive, observed negative); warrants investigation
- `drive_advantage` shows inverse correlation in some runs; may reflect the urban accessibility paradox (vulnerable populations in spatially accessible urban cores)
- Zero-variance features: 0
- NaN values: 0

## File Locations

- Feature extraction: `granite/data/enhanced_accessibility.py`
- Modal computation: `granite/features/modal_accessibility.py`
- OSRM routing: `granite/routing/osrm_router.py`
- Feature caching: `granite/cache.py` (stored in `./granite_cache/`)