"""
Input-feature Moran's I probe (FIG-P1 B4).

For each of the 73 address features in the n20 feature matrix, compute Moran's I
per tract with the same estimator and coordinates as output Moran's I
(SpatialLearningDiagnostics.compute_spatial_autocorrelation: k=8 k-NN on (lon, lat),
row-standardized, permutations=0), then take the unweighted mean across tracts.
Zero-variance feature-tract pairs are skipped and counted. lat and lon are scored
as reference rows outside the 73.

Outputs (data/results/paper1_figures/):
  input_feature_morans_i.csv           feature, group, mean_morans_i, tracts_scored
  input_feature_morans_i_summary.json  group medians, lat/lon reference values, and
                                       I for the top 10 SAGE permutation-importance
                                       features

Usage:
    python scripts/input_feature_morans_probe.py
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from granite.evaluation.spatial_diagnostics import SpatialLearningDiagnostics

MATRIX = REPO_ROOT / 'experiments' / 'ecological_fallacy' / 'n20_feature_matrix.csv'
IMPORTANCE = (REPO_ROOT / 'experiments' / 'ablation' / '00_baseline' / 'results'
              / 'feature_importance' / 'sage_permutation_importance.csv')
OUT_DIR = REPO_ROOT / 'data' / 'results' / 'paper1_figures'
OUT_CSV = OUT_DIR / 'input_feature_morans_i.csv'
OUT_JSON = OUT_DIR / 'input_feature_morans_i_summary.json'

NON_FEATURE = ('fips', 'address_idx', 'lat', 'lon', 'tract_svi')
DESTS = ('employment', 'healthcare', 'grocery')
ACCESS_SUFFIX = ('min_time', 'mean_time', 'median_time', 'count_5min', 'count_10min',
                 'count_15min', 'drive_advantage', 'dispersion', 'time_range', 'percentile')
MODAL_SUFFIX = ('transit_dependence', 'car_effective_access', 'walk_effective_access',
                'modal_access_gap', 'forced_walk_burden')
GROUPS = {
    'accessibility': [f'{d}_{s}' for d in DESTS for s in ACCESS_SUFFIX],
    'modal': [f'{d}_{s}' for d in DESTS for s in MODAL_SUFFIX],
    'socioeconomic': ['pct_no_vehicle', 'pct_poverty', 'pct_unemployed', 'pct_no_hs_diploma',
                      'pct_uninsured', 'pct_mobile_homes', 'pct_crowded', 'population',
                      'housing_units'],
    'building': ['log_bldg_footprint_m2', 'bldg_vertex_count'],
    'flood': ['in_sfha'],
    'land_cover': ['nlcd_land_cover', 'nlcd_impervious_pct', 'nlcd_tree_canopy_pct'],
    'parcel': ['is_residential', 'log_appvalue', 'build_to_land_ratio', 'log_acres',
               'lucode_residential', 'lucode_commercial', 'lucode_industrial', 'lucode_other',
               'proptype_residential', 'proptype_apartment_10plus', 'proptype_commercial',
               'proptype_rental_40pct', 'proptype_cha_housing'],
}
EXPECTED_GROUP_SIZES = {'building': 2, 'flood': 1, 'land_cover': 3, 'parcel': 13,
                        'accessibility': 30, 'modal': 15, 'socioeconomic': 9}


def _git_sha():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(REPO_ROOT)).decode().strip()


def main():
    df = pd.read_csv(MATRIX, dtype={'fips': str})
    features = [c for c in df.columns if c not in NON_FEATURE]
    group_of = {f: g for g, fs in GROUPS.items() for f in fs}
    assert {g: len(fs) for g, fs in GROUPS.items()} == EXPECTED_GROUP_SIZES
    assert sorted(features) == sorted(group_of), set(features) ^ set(group_of)
    assert len(features) == 73

    diag = SpatialLearningDiagnostics(verbose=False)
    tracts = sorted(df['fips'].unique())
    scored = {f: [] for f in features + ['lat', 'lon']}
    skipped = 0
    n_nan_values = int(df[features].isna().sum().sum())

    for fips in tracts:
        sub = df[df['fips'] == fips]
        coords = sub[['lon', 'lat']].to_numpy(dtype=float)
        for f in features + ['lat', 'lon']:
            v = sub[f].to_numpy(dtype=float)
            if not np.isfinite(v).all():
                raise ValueError(f'non-finite values in {f} for {fips}')
            # exact constant test; np.var leaves float residue on constant columns
            if np.ptp(v) == 0.0:
                if f in group_of:
                    skipped += 1
                continue
            scored[f].append(float(diag.compute_spatial_autocorrelation(v, coords, k_neighbors=8)))

    rows = []
    for f in features + ['lat', 'lon']:
        vals = scored[f]
        rows.append({
            'feature': f,
            'group': group_of.get(f, 'reference'),
            'mean_morans_i': float(np.mean(vals)) if vals else float('nan'),
            'tracts_scored': len(vals),
        })
    out = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False, float_format='%.17g')

    feat = out[out['group'] != 'reference']
    group_medians = {}
    for g in GROUPS:
        med = feat.loc[feat['group'] == g, 'mean_morans_i'].median()
        group_medians[g] = None if pd.isna(med) else float(med)
    group_unscored = {g: int(feat.loc[feat['group'] == g, 'mean_morans_i'].isna().sum())
                      for g in GROUPS}
    imp = pd.read_csv(IMPORTANCE).sort_values('rank').head(10)
    lookup = out.set_index('feature')
    top10 = [{
        'rank': int(r['rank']), 'feature': r['feature'], 'group': group_of[r['feature']],
        'mean_drop': float(r['mean_drop']),
        'mean_morans_i': float(lookup.loc[r['feature'], 'mean_morans_i']),
        'tracts_scored': int(lookup.loc[r['feature'], 'tracts_scored']),
    } for _, r in imp.iterrows()]

    summary = {
        'milestone': 'FIG-P1 B4',
        'command': 'python scripts/input_feature_morans_probe.py',
        'commit': _git_sha(),
        'input': str(MATRIX.relative_to(REPO_ROOT)),
        'estimator': ('granite.evaluation.spatial_diagnostics.SpatialLearningDiagnostics.'
                      'compute_spatial_autocorrelation; libpysal KNN k=8 on (lon, lat), '
                      "transform 'r', esda.Moran permutations=0"),
        'aggregation': 'unweighted mean of per-tract I across tracts; median of feature means by group',
        'n_tracts': len(tracts),
        'n_features': len(features),
        'feature_tract_pairs': len(features) * len(tracts),
        'zero_variance_pairs_skipped': skipped,
        'zero_variance_rule': 'max equals min within the tract (np.ptp == 0)',
        'nan_feature_values': n_nan_values,
        'group_sizes': {g: len(fs) for g, fs in GROUPS.items()},
        'group_median_mean_morans_i': group_medians,
        'group_features_never_scored': group_unscored,
        'reference_rows': {c: {'mean_morans_i': float(lookup.loc[c, 'mean_morans_i']),
                               'tracts_scored': int(lookup.loc[c, 'tracts_scored'])}
                           for c in ('lat', 'lon')},
        'top10_sage_permutation_importance': {
            'source': str(IMPORTANCE.relative_to(REPO_ROOT)),
            'flag': ('importance ranking computed on the production graph (00_baseline run); '
                     "Moran's I here uses fixed k=8 k-NN weights on coordinates"),
            'features': top10,
        },
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: summary[k] for k in (
        'zero_variance_pairs_skipped', 'feature_tract_pairs', 'group_median_mean_morans_i',
        'group_features_never_scored', 'reference_rows')}, indent=2))
    for t in top10:
        print(f"  {t['rank']:>2} {t['feature']:<32} {t['group']:<14} I={t['mean_morans_i']:.5f}")


if __name__ == '__main__':
    main()
