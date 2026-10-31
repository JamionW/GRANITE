#!/usr/bin/env python3
"""
manifest and v1-vs-v2 comparison for the block-group SVI reference (REF-FIX).

reads the national caches for both reference versions (gitignored, sha256
anchored here) and the committed 5b npz address coordinates. writes only
data/results/bg_reference_v2/:
  manifest.json                 row counts, per-indicator ranking pools, svi
                                counts, hamilton totals, scored bg counts
  hamilton_v1_v2_comparison.csv per-bg svi under each version and the difference

scored block groups: hamilton bgs with non-null svi holding at least the
pooled address threshold of the n20 address set (3 for the 5b sweeps, 10 for
m0). the address set is identical across 5b cells; coordinates come from the
production sage ts42 npz files.

usage:
    python scripts/bg_reference_v2_manifest.py
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from granite.data.block_group_loader import BlockGroupLoader
from granite.validation.block_group_validation import BlockGroupValidator

OUT_DIR = REPO_ROOT / 'data' / 'results' / 'bg_reference_v2'
PROCESSED = REPO_ROOT / 'data' / 'processed'
PRED_DIR = REPO_ROOT / 'data' / 'results' / 'm6_topology_5b' / 'predictions'
COORD_CELL = 'production__sage__ts42__gs42__'
NAMED_BGS = ['470650104314', '470650104333']
THRESHOLDS = {'5b_min3': 3, 'm0_min10': 10}
EP_VARS = ['EP_MHI', 'EP_PCI', 'EP_UNEMP', 'EP_NOHSDP', 'EP_AGE65', 'EP_AGE17',
           'EP_SNGPNT', 'EP_MINRTY', 'EP_MUNIT', 'EP_MOBILE', 'EP_CROWD', 'EP_NOVEH']


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _bg_address_counts(bg_gdf):
    """address count per bg over the n20 address set, via the validator join."""
    files = sorted(PRED_DIR.glob(f'{COORD_CELL}*.npz'))
    coords = np.concatenate([np.load(f)['coords'] for f in files])
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(coords[:, 0], coords[:, 1]),
                           crs='EPSG:4326')
    validator = BlockGroupValidator(bg_gdf, verbose=False)
    joined = validator._assign_to_block_groups(pts)
    counts = joined['block_group_id'].dropna().value_counts()
    return counts, len(files), len(coords)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    loader = BlockGroupLoader(data_dir=str(REPO_ROOT / 'data'), verbose=False)

    manifest = {
        'milestone': 'REF-FIX',
        'reference_versions': {
            'v1': 'frozen; census sentinels unhandled; EP_MINRTY = non-white-alone '
                  '(B02001) + hispanic (B03003), capped at 100',
            'v2': 'sentinels (<= -100000000) nulled before rates; EP_MINRTY = '
                  '100 * (1 - B03002_003E / B03002_001E) (cdc svi 2020 DP05 '
                  'minority sum, block-group equivalent)',
        },
        'cache_files': {},
        'versions': {},
    }

    svi_by_version = {}
    raw_by_version = {}
    for v in ('v1', 'v2'):
        raw_path = PROCESSED / loader.NATIONAL_CACHE_FILES[v]['raw']
        svi_path = PROCESSED / loader.NATIONAL_CACHE_FILES[v]['svi']
        manifest['cache_files'][v] = {
            str(raw_path.relative_to(REPO_ROOT)): _sha256(raw_path),
            str(svi_path.relative_to(REPO_ROOT)): _sha256(svi_path),
        }
        raw = pd.read_csv(raw_path, dtype={'GEOID': str})
        svi = loader.compute_national_svi(reference_version=v)
        raw_by_version[v] = raw
        svi_by_version[v] = svi

        est_cols = [c for c in raw.columns if c.endswith('E') and c[:1] in 'BC' and '_' in c]
        ep_neg = {c: int((svi[c] < 0).sum()) for c in EP_VARS}
        ep_sentinel = {c: int((svi[c] <= loader.CENSUS_SENTINEL_MAX).sum()) for c in EP_VARS}
        raw_sentinel = {c: int((raw[c] <= loader.CENSUS_SENTINEL_MAX).sum())
                        for c in est_cols if (raw[c] <= loader.CENSUS_SENTINEL_MAX).any()}

        ham = svi[svi['GEOID'].str.startswith('47065')]
        manifest['versions'][v] = {
            'row_counts': {
                'raw_rows': int(len(raw)),
                'raw_unique_geoid': int(raw['GEOID'].nunique()),
                'svi_rows': int(len(svi)),
                'svi_non_null': int(svi['SVI'].notna().sum()),
            },
            'indicator_non_null_ranking_pool': {c: int(svi[c].notna().sum()) for c in EP_VARS},
            'ep_negative_counts': ep_neg,
            'ep_sentinel_counts': ep_sentinel,
            'raw_sentinel_counts_by_column': raw_sentinel,
            'ep_minrty_at_100': int((svi['EP_MINRTY'] >= 100).sum()),
            'hamilton_county': {
                'geoid_prefix': '47065',
                'total_bgs': int(len(ham)),
                'svi_non_null': int(ham['SVI'].notna().sum()),
            },
        }

    # scored bg counts per version and threshold
    geoms = loader.load_block_group_geometries('47', '065')
    counts, n_files, n_addr = _bg_address_counts(geoms)
    manifest['scored_bg_address_source'] = {
        'npz_glob': f'data/results/m6_topology_5b/predictions/{COORD_CELL}*.npz',
        'n_files': n_files,
        'n_addresses': int(n_addr),
        'n_bgs_with_addresses': int(len(counts)),
    }
    for v in ('v1', 'v2'):
        ham = svi_by_version[v]
        ham = ham[ham['GEOID'].str.startswith('47065')].set_index('GEOID')['SVI']
        scored = {}
        for label, thr in THRESHOLDS.items():
            eligible = counts[counts >= thr].index
            scored[label] = {
                'threshold': thr,
                'bgs_meeting_threshold': int(len(eligible)),
                'scored_bgs': int(ham.reindex(eligible).notna().sum()),
            }
        manifest['versions'][v]['hamilton_county']['scored'] = scored

    # per-bg comparison
    keep = ['GEOID', 'SVI', 'EP_MHI', 'EP_PCI', 'EP_MINRTY',
            'SVI_theme1', 'SVI_theme3']
    h1 = svi_by_version['v1'][svi_by_version['v1']['GEOID'].str.startswith('47065')][keep]
    h2 = svi_by_version['v2'][svi_by_version['v2']['GEOID'].str.startswith('47065')][keep]
    comp = h1.merge(h2, on='GEOID', how='outer', suffixes=('_v1', '_v2'), validate='one_to_one')
    comp['SVI_diff_v2_minus_v1'] = comp['SVI_v2'] - comp['SVI_v1']
    comp['n_addresses_n20'] = comp['GEOID'].map(counts).fillna(0).astype(int)
    comp = comp.sort_values('GEOID').reset_index(drop=True)
    comp.to_csv(OUT_DIR / 'hamilton_v1_v2_comparison.csv', index=False)

    def _corr(df):
        both = df.dropna(subset=['SVI_v1', 'SVI_v2'])
        return {
            'n_bgs': int(len(both)),
            'pearson': float(stats.pearsonr(both['SVI_v1'], both['SVI_v2'])[0]),
            'spearman': float(stats.spearmanr(both['SVI_v1'], both['SVI_v2'])[0]),
            'mean_abs_diff': float(both['SVI_diff_v2_minus_v1'].abs().mean()),
            'max_abs_diff': float(both['SVI_diff_v2_minus_v1'].abs().max()),
        }

    manifest['hamilton_v1_v2'] = {
        'all_bgs_scored_by_both': _corr(comp),
        'scoring_set_min3_scored_by_both': _corr(comp[comp['n_addresses_n20'] >= 3]),
        'scoring_set_min10_scored_by_both': _corr(comp[comp['n_addresses_n20'] >= 10]),
        'named_bgs': comp[comp['GEOID'].isin(NAMED_BGS)][
            ['GEOID', 'SVI_v1', 'SVI_v2', 'EP_MHI_v1', 'EP_MHI_v2']
        ].astype(object).where(lambda d: d.notna(), None).to_dict(orient='records'),
    }

    with open(OUT_DIR / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
