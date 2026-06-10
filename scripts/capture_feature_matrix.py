"""
capture_feature_matrix.py

Assembles the 73-feature matrix for all 20 n20 tracts in natural (un-standardized)
units and writes it to experiments/ecological_fallacy/n20_feature_matrix.csv.

No GNN training. No feature standardization. No feature-mode substitution.
Reuses existing pipeline feature-assembly methods.

Output columns (in order):
  fips, address_idx, lat, lon, tract_svi, <73 feature columns>

Usage:
  python scripts/capture_feature_matrix.py
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd

# ensure repo root is on sys.path when run from scripts/
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from granite.disaggregation.pipeline import GRANITEPipeline

# hamilton county lat range (rough bounding box); used to catch accidental z-scoring
LAT_MIN, LAT_MAX = 34.9, 35.3

N20_FIPS = [
    '47065000600', '47065000700', '47065001200', '47065001800', '47065001900',
    '47065002400', '47065003400', '47065010431', '47065010433', '47065010435',
    '47065011311', '47065011321', '47065011324', '47065011325', '47065011326',
    '47065011402', '47065011413', '47065011444', '47065011447', '47065011900',
]

OUTPUT_PATH = os.path.join(REPO_ROOT, 'experiments', 'ecological_fallacy', 'n20_feature_matrix.csv')


def load_config():
    config_path = os.path.join(REPO_ROOT, 'config.yaml')
    with open(config_path) as f:
        config = yaml.safe_load(f)
    # disable feature-mode substitution and pruning for this capture run
    config['feature_mode'] = 'full'
    config.setdefault('processing', {})['prune_features_path'] = None
    return config


def main():
    config = load_config()

    pipeline = GRANITEPipeline(
        config=config,
        data_dir=os.path.join(REPO_ROOT, 'data'),
        output_dir=os.path.join(REPO_ROOT, 'output', 'capture_feature_matrix'),
        verbose=True,
    )

    print("Loading spatial data...")
    data = pipeline._load_spatial_data()

    all_frames = []
    per_tract_counts = {}
    canonical_feature_names = None

    for fips in N20_FIPS:
        print(f"\n--- tract {fips} ---")

        addresses = pipeline.data_loader.get_addresses_for_tract(fips)
        if len(addresses) == 0:
            raise RuntimeError(f"zero rows for tract {fips}")

        addresses = addresses.copy()
        addresses['tract_fips'] = fips

        # get tract SVI (RPL_THEMES from CDC SVI data)
        tract_svi = pipeline.data_loader.get_tract_target_value(fips, data['svi'], target='svi')

        # compute feature matrix in natural units; no normalization applied here
        raw_features = pipeline._compute_accessibility_features(addresses, data)
        if raw_features is None:
            raise RuntimeError(f"_compute_accessibility_features returned None for {fips}")

        n_features = raw_features.shape[1]
        feature_names = pipeline._generate_feature_names(n_features)

        print(f"  addresses: {len(addresses)}, features: {n_features}")
        print(f"  feature names: {feature_names[:5]} ... {feature_names[-3:]}")

        # verify no per-tract standardization: feature matrix must be in natural units
        # (socioeco features are raw CDC values, accessibility are real travel times/counts)
        # lat/lon from geometry (raw coordinates, not z-scored)
        lat = addresses.geometry.y.values
        lon = addresses.geometry.x.values

        # guard: lat must be in hamilton county plausible range
        if lat.min() < LAT_MIN or lat.max() > LAT_MAX:
            raise RuntimeError(
                f"lat out of expected range for {fips}: min={lat.min():.4f} max={lat.max():.4f}. "
                f"Expected [{LAT_MIN}, {LAT_MAX}]. Possible accidental z-scoring."
            )

        # establish canonical feature name list from first tract; enforce consistency
        if canonical_feature_names is None:
            canonical_feature_names = feature_names
        elif feature_names != canonical_feature_names:
            missing = [n for n in canonical_feature_names if n not in feature_names]
            extra = [n for n in feature_names if n not in canonical_feature_names]
            raise RuntimeError(
                f"feature name mismatch for {fips}.\n"
                f"  missing vs canonical: {missing}\n"
                f"  extra vs canonical: {extra}"
            )

        # assert one unique svi value per fips (it is a tract-level constant)
        assert isinstance(tract_svi, float) and not np.isnan(tract_svi), \
            f"tract_svi is NaN or not float for {fips}"

        frame = pd.DataFrame(raw_features, columns=feature_names)
        frame.insert(0, 'fips', fips)
        frame.insert(1, 'address_idx', np.arange(len(addresses)))
        frame.insert(2, 'lat', lat)
        frame.insert(3, 'lon', lon)
        frame.insert(4, 'tract_svi', tract_svi)

        all_frames.append(frame)
        per_tract_counts[fips] = len(addresses)
        print(f"  tract_svi: {tract_svi:.4f}, lat range: [{lat.min():.4f}, {lat.max():.4f}]")

    combined = pd.concat(all_frames, ignore_index=True)

    # final assertions before write
    assert combined['fips'].nunique() == 20, \
        f"expected 20 distinct fips, got {combined['fips'].nunique()}"

    # per-fips uniqueness of tract_svi
    svi_per_fips = combined.groupby('fips')['tract_svi'].nunique()
    bad = svi_per_fips[svi_per_fips > 1]
    if len(bad) > 0:
        raise RuntimeError(f"multiple tract_svi values within a fips: {bad.to_dict()}")

    total_rows = sum(per_tract_counts.values())
    assert len(combined) == total_rows, \
        f"row count mismatch: combined={len(combined)}, expected={total_rows}"

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    combined.to_csv(OUTPUT_PATH, index=False)

    print(f"\n=== done ===")
    print(f"output: {OUTPUT_PATH}")
    print(f"shape: {combined.shape}")
    print(f"columns ({len(combined.columns)}): {list(combined.columns)}")
    print(f"\nper-tract address counts:")
    for fips, count in per_tract_counts.items():
        svi_val = combined[combined['fips'] == fips]['tract_svi'].iloc[0]
        print(f"  {fips}: {count:5d} addresses  svi={svi_val:.4f}")
    print(f"\ntotal addresses: {total_rows}")
    print(f"feature columns: {len(canonical_feature_names)} ({canonical_feature_names[0]} .. {canonical_feature_names[-1]})")


if __name__ == '__main__':
    main()
