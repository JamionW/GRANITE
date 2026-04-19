# Property Value Target: Feature Leakage Audit

Date: 2026-04-18

## Purpose

When the disaggregation target is `property_value` (normalized
log assessed value), any feature derived from assessor records or
building geometry that plausibly correlates with assessed value
constitutes target leakage and must be excluded from the input
feature matrix. This audit classifies every feature active under
the SVI target path.

## Classification key

| Category | Definition |
|---|---|
| parcel_derived_leaky | Derived from assessor records or building geometry; plausibly correlates with assessed value |
| spatially_derived | From remote sensing, FEMA, OSRM routing, road topology, or LEHD; no assessor link |
| socioeconomic_control | Tract-level ACS/SVI demographic rate; constant across addresses in a tract |
| ambiguous | Could go either way; reasoning documented |

## Feature inventory

### Address-level building/parcel features (from `_extract_building_features`)

| Feature | Classification | Rationale |
|---|---|---|
| log_appvalue | parcel_derived_leaky | This IS the target variable (log assessed value). Already excluded in Session 1. |
| build_to_land_ratio | parcel_derived_leaky | Ratio of improvement value to land value from assessor records. Already excluded in Session 1. |
| bldg_footprint_m2 | parcel_derived_leaky | Building footprint area from OSM, but larger footprint directly predicts higher assessed value. |
| log_acres | parcel_derived_leaky | Parcel acreage from assessor records. Larger lots command higher values. |
| LUCODE | parcel_derived_leaky | Hamilton County land use code from assessor. Commercial/industrial codes correlate with value tiers. |
| PROPTYPE | parcel_derived_leaky | Property type code from assessor (residential, commercial, apartment, etc.). Directly tied to value bands. |
| osm_building_type | parcel_derived_leaky | OSM building type (house, apartments, commercial, etc.). Building type is a strong value predictor even though sourced from OSM rather than assessor. |
| bldg_vertex_count | ambiguous | OSM building polygon complexity. More vertices may indicate irregular/larger/custom construction (higher value), but the signal is weak and indirect. Could also just reflect OSM mapper detail. |
| in_sfha | spatially_derived | FEMA Special Flood Hazard Area designation. Federal spatial dataset independent of assessor records. |
| nlcd_land_cover | spatially_derived | NLCD land cover class from satellite imagery. No assessor connection. |
| nlcd_impervious_pct | spatially_derived | NLCD impervious surface percentage from satellite. No assessor connection. |
| nlcd_tree_canopy_pct | spatially_derived | NLCD tree canopy percentage from satellite. No assessor connection. |

### Features not currently active but relevant

| Feature | Classification | Rationale |
|---|---|---|
| osm_levels | parcel_derived_leaky | OSM building levels (number of stories). Not currently in the `expected` list in `_extract_building_features`, so not active. If added, would be leaky: story count strongly predicts value. |
| flood_zone | spatially_derived | FEMA flood zone code. Loaded in `loaders.py` but not in `_extract_building_features` expected list, so not active. If added, would be spatially_derived. |

### Base accessibility features (30 features: 3 dest types x 10)

All spatially_derived. Travel times and destination counts computed
from OSRM routing and destination locations. No assessor connection.

| Feature pattern | Classification | Count |
|---|---|---|
| {employment,healthcare,grocery}_{min,mean,median}_time | spatially_derived | 9 |
| {employment,healthcare,grocery}_count_{5,10,15}min | spatially_derived | 9 |
| {employment,healthcare,grocery}_drive_advantage | spatially_derived | 3 |
| {employment,healthcare,grocery}_dispersion | spatially_derived | 3 |
| {employment,healthcare,grocery}_time_range | spatially_derived | 3 |
| {employment,healthcare,grocery}_percentile | spatially_derived | 3 |

### Modal accessibility features (15 features: 3 dest types x 5)

All spatially_derived. Per-address OSRM drive/walk time summaries.

| Feature pattern | Classification | Count |
|---|---|---|
| {employment,healthcare,grocery}_transit_dependence | spatially_derived | 3 |
| {employment,healthcare,grocery}_car_effective_access | spatially_derived | 3 |
| {employment,healthcare,grocery}_walk_effective_access | spatially_derived | 3 |
| {employment,healthcare,grocery}_modal_access_gap | spatially_derived | 3 |
| {employment,healthcare,grocery}_forced_walk_burden | spatially_derived | 3 |

### Socioeconomic controls (9 features, tract-level constants)

All socioeconomic_control. ACS-derived rates from CDC SVI data,
constant for all addresses within a tract.

| Feature | Classification |
|---|---|
| pct_no_vehicle | socioeconomic_control |
| pct_poverty | socioeconomic_control |
| pct_unemployed | socioeconomic_control |
| pct_no_hs_diploma | socioeconomic_control |
| pct_uninsured | socioeconomic_control |
| pct_mobile_homes | socioeconomic_control |
| pct_crowded | socioeconomic_control |
| population | socioeconomic_control |
| housing_units | socioeconomic_control |

## Proposed exclusion list for `target=property_value`

### Exclude (parcel_derived_leaky)

1. `log_appvalue` (already excluded)
2. `build_to_land_ratio` (already excluded)
3. `bldg_footprint_m2`
4. `log_acres`
5. `LUCODE`
6. `PROPTYPE`
7. `osm_building_type`

### Exclude (ambiguous, confirmed leaky)

8. `bldg_vertex_count` -- weak signal but only path to leakage is
   through building size/complexity, which correlates with value.
   Excluding is conservative and costs little predictive power.
   **Decision: exclude.** Confirmed 2026-04-18.

### Retain (spatially_derived)

- in_sfha
- nlcd_land_cover, nlcd_impervious_pct, nlcd_tree_canopy_pct
- All 30 base accessibility features
- All 15 modal accessibility features

### Retain (socioeconomic_control)

- All 9 tract-level ACS features

## Expected feature count after exclusions

| Category | SVI target | property_value target |
|---|---|---|
| Base accessibility | 30 | 30 |
| Modal accessibility | 15 | 15 |
| Socioeconomic controls | 9 | 9 |
| Building/parcel features | ~18* | 4 (in_sfha + 3 NLCD) |
| **Total** | **~72** | **~58** |

*Building feature count varies due to one-hot encoding of LUCODE (4 dummies)
and PROPTYPE (up to 5 dummies).

## Status

Exclusion list approved 2026-04-18. All 8 features excluded.
`bldg_vertex_count` confirmed excluded (ambiguous, leaning leaky).
