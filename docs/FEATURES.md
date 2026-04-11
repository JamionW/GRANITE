# GRANITE Feature Reference

Complete inventory of features used by the GNN, organized by source and group.

## Summary

72+ features per address (variable depending on property type coverage), organized in four groups:

| Group | Count | Varies within tract? | Source |
|---|---|---|---|
| Base accessibility | 30 | Yes | OSRM travel times |
| Modal accessibility | 15 | Yes | Per-address OSRM drive + walk times |
| Socioeconomic controls | 9 | No (tract-level) | CDC SVI / ACS |
| Address-level attributes | 18 (variable) | Yes | Parcel, building, flood, land cover |
| **Total** | **72+** | | |

Of these, 9 are tract-level constants (socioeconomic controls). Modal features were previously tract-level constants but are now computed per address from OSRM driving and walking travel times, giving them within-tract variance.

---

## Group 1: Base Accessibility (30 features)

10 features per destination type, computed per address from real OSRM travel times.

### Per destination type

| # | Feature | Description | Expected SVI correlation |
|---|---------|-------------|--------------------------|
| 1 | `{type}_min_time` | Min driving time to any destination | Positive |
| 2 | `{type}_mean_time` | Mean driving time across all destinations | Positive |
| 3 | `{type}_median_time` | Median driving time | Positive |
| 4 | `{type}_count_5min` | Destinations reachable in 5 min driving | Negative |
| 5 | `{type}_count_10min` | Destinations reachable in 10 min driving | Negative |
| 6 | `{type}_count_15min` | Destinations reachable in 15 min driving | Negative |
| 7 | `{type}_drive_advantage` | (walk_avg - drive_avg) / walk_avg, clipped to [-0.2, 1.0] | Positive |
| 8 | `{type}_dispersion` | CV of drive times (std/mean), clipped to [0, 1] | Positive |
| 9 | `{type}_time_range` | (max - min) / mean drive time, clipped to [0, 2] | Positive |
| 10 | `{type}_percentile` | Rank percentile of mean_time within tract | Positive |

### Destination types

| Type | Data source | Count |
|---|---|---|
| `employment` | LEHD Workplace Area Characteristics 2021 | 1,329 locations |
| `healthcare` | CMS Hospital Compare | 12 facilities |
| `grocery` | OpenStreetMap supermarket/grocery tags | 189 stores |

### Empirical finding

Accessibility features show near-zero within-tract variance (r ~ 0.03 with SVI predictions). Raw spatial coordinates (r ~ 0.67) dramatically outperform them as predictors. This is a validated empirical finding, not a data quality problem. In Hamilton County, addresses within a single tract are close enough that travel-time differences are negligible.

---

## Group 2: Modal Accessibility (15 features)

5 features per destination type, computed per address from OSRM driving and walking travel times. Per-address travel time summaries (drive/walk nearest, full time arrays) are extracted during base feature computation and cached alongside per-destination features.

| # | Feature | Formula | Description |
|---|---------|---------|-------------|
| 1 | `{type}_transit_dependence` | (drive_nearest + walk_nearest) / 2 | Avg time to nearest destination across both modes |
| 2 | `{type}_car_effective_access` | std(drive_nearest, walk_nearest) | Mode disparity for nearest destination |
| 3 | `{type}_walk_effective_access` | \|{drive reachable in 10min} union {walk reachable in 10min}\| | Destinations reachable by either mode |
| 4 | `{type}_modal_access_gap` | \|walk_nearest - drive_nearest\| | Absolute equity gap between modes |
| 5 | `{type}_forced_walk_burden` | walk_nearest / drive_nearest | How much worse walking is than driving |

Feature names are kept from the original implementation for downstream compatibility; the semantics have changed from tract-level vehicle-ownership-weighted features to per-address multi-modal travel time comparisons.

When per-address travel times are unavailable (e.g., partial cache from an older run), the function falls back to a tract-level approximation derived from base accessibility features.

**Source:** `granite/features/modal_accessibility.py`

---

## Group 3: Socioeconomic Controls (9 features)

Tract-level ACS variables from the CDC Social Vulnerability Index, applied uniformly to all addresses within a tract.

| # | Feature | SVI field | Description |
|---|---------|-----------|-------------|
| 1 | `pct_no_vehicle` | EP_NOVEH | % households without vehicle access |
| 2 | `pct_poverty` | EP_POV150 | % population below 150% poverty line |
| 3 | `pct_unemployed` | EP_UNEMP | % civilian labor force unemployed |
| 4 | `pct_no_hs_diploma` | EP_NOHSDP | % age 25+ without high school diploma |
| 5 | `pct_uninsured` | EP_UNINSUR | % civilian noninstitutionalized without health insurance |
| 6 | `pct_mobile_homes` | EP_MOBILE | % housing units that are mobile homes |
| 7 | `pct_crowded` | EP_CROWD | % occupied housing with more people than rooms |
| 8 | `population` | E_TOTPOP | Total population |
| 9 | `housing_units` | E_HU | Total housing units |

**Known limitation:** Same as modal features: tract-level constants that cannot drive within-tract disaggregation.

**Source:** `granite/data/loaders.py` (`get_tract_socioeconomic_features`)

---

## Group 4: Address-Level Attributes (up to 18 features)

Extracted from `combined_address_features.csv`. Feature count varies per tract depending on data availability (particularly PROPTYPE coverage).

### Building (2 features)

| Feature | Description |
|---------|-------------|
| `log_bldg_footprint_m2` | log(1 + building footprint area in m2) |
| `bldg_vertex_count` | Building polygon vertex count (shape complexity) |

**Source:** Microsoft Building Footprints

### Flood (1 feature)

| Feature | Description |
|---------|-------------|
| `in_sfha` | Binary: address in FEMA Special Flood Hazard Area |

**Source:** FEMA National Flood Hazard Layer

### Building Type (1 feature)

| Feature | Description |
|---------|-------------|
| `is_residential` | Binary: OSM building type is residential/house/apartments/detached/etc. |

**Source:** OpenStreetMap building tags

### Parcel (3 features)

| Feature | Description |
|---------|-------------|
| `log_appvalue` | log of county-appraised property value |
| `build_to_land_ratio` | Building area / lot area |
| `log_acres` | log of parcel acreage |

**Source:** Hamilton County tax assessor parcel data

### Land Use Code (4 features, one-hot)

| Feature | LUCODE range |
|---------|-------------|
| `lucode_residential` | 100-199 |
| `lucode_commercial` | 200-299 |
| `lucode_industrial` | 300-399 |
| `lucode_other` | All other codes |

**Source:** Hamilton County tax assessor

### Property Type (up to 5 features, one-hot)

Only property types present in the tract's addresses are encoded, so the count varies (0-5).

| Feature | Code | Description |
|---------|------|-------------|
| `proptype_residential` | 22.0 | Single-family residential |
| `proptype_apartment_10plus` | 40.0 | Apartment 10+ units |
| `proptype_commercial` | 8.0 | Commercial |
| `proptype_rental_40pct` | 32.0 | Rental 40%+ |
| `proptype_cha_housing` | 11.0 | Chattanooga Housing Authority |

**Source:** Hamilton County tax assessor

### NLCD Land Cover (3 features)

| Feature | Description |
|---------|-------------|
| `nlcd_land_cover` | NLCD classification code (raw numeric) |
| `nlcd_impervious_pct` | Percent impervious surface |
| `nlcd_tree_canopy_pct` | Percent tree canopy cover |

**Source:** USGS National Land Cover Database

---

## Context Features (5 features, separate from feature matrix)

Used by the GNN's context-gating mechanism to modulate accessibility features. These are passed to the model separately and do not appear in the feature matrix column count.

| # | Feature | Derivation |
|---|---------|------------|
| 1 | pct_no_vehicle | Normalized from socioeconomic group |
| 2 | pct_poverty | Normalized from socioeconomic group |
| 3 | pct_unemployed | Normalized from socioeconomic group |
| 4 | pct_no_hs_diploma | Normalized from socioeconomic group |
| 5 | population | E_TOTPOP / 10000 |

**Source:** `granite/data/loaders.py` (`create_context_features_for_addresses`)

---

## Feature Quality Notes

From validation runs on multi-tract configurations:

- Time/count anti-correlation: r = -0.956 (correct; shorter times correlate with higher counts)
- `dispersion` features show inverse correlations in some tracts (expected positive, observed negative)
- `drive_advantage` shows inverse correlation in some runs; may reflect the urban accessibility paradox (vulnerable populations in spatially accessible urban cores)
- Zero-variance features: 11 per tract typical (9 socioeconomic constants + count features where all addresses reach the same number of destinations within the threshold). Previously 24+ when modal features were also tract-level constants.

## File Locations

| Component | File |
|---|---|
| Base accessibility extraction | `granite/data/enhanced_accessibility.py` |
| Modal feature computation | `granite/features/modal_accessibility.py` |
| Socioeconomic extraction | `granite/data/loaders.py` |
| Building/flood/NLCD extraction | `granite/disaggregation/pipeline.py` (`_extract_building_features`) |
| Per-address travel time summaries | `granite/disaggregation/pipeline.py` (`_summarize_travel_times`) |
| Feature assembly + caching | `granite/disaggregation/pipeline.py` (`_compute_accessibility_features`) |
| Feature name generation | `granite/disaggregation/pipeline.py` (`_generate_feature_names`) |
| OSRM routing | `granite/routing/osrm_router.py` |
