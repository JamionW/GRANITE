"""
Build the committed Figure 1 inputs for Paper 1 (FIG-P1 B1).

Writes to data/results/paper1_figures/:
  hamilton_tracts_svi.geojson  Hamilton County (47065) tract polygons, EPSG:4326,
                               GEOID, RPL_THEMES (null where SVI is missing or -999),
                               paper1 flag for the 20 tracts in tract_selection.txt
  inset_edges_<fips>.npz       full production edge list for the inset tract
                               (road-network hybrid, granite.data.loaders), node
                               coordinates, and a display window of nodes
  provenance.json              sha256 of each raw source, the command, the commit

Usage:
    python scripts/build_figure1_artifacts.py --inset-fips 47065000600
"""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from granite.data.loaders import DataLoader

OUT_DIR = REPO_ROOT / 'data' / 'results' / 'paper1_figures'
RAW = REPO_ROOT / 'data' / 'raw'
TRACT_SELECTION = REPO_ROOT / 'experiments' / 'ablation' / '00_baseline' / 'tract_selection.txt'
TRACT_SHP = RAW / 'tl_2020_47_tract.shp'
SVI_CSV = RAW / 'SVI_2020_US.csv'

# raw files read by this script or by the production graph builder it calls
RAW_SOURCES = [
    RAW / 'tl_2020_47_tract.shp', RAW / 'tl_2020_47_tract.shx',
    RAW / 'tl_2020_47_tract.dbf', RAW / 'tl_2020_47_tract.prj',
    RAW / 'SVI_2020_US.csv',
    RAW / 'chattanooga.geojson',
    RAW / 'address_features' / 'combined_address_features.csv',
    RAW / 'tl_2023_47065_roads.shp', RAW / 'tl_2023_47065_roads.shx',
    RAW / 'tl_2023_47065_roads.dbf', RAW / 'tl_2023_47065_roads.prj',
]

WINDOW_N = 200  # display window size, inside the approved 150 to 250 range


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _git_sha():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(REPO_ROOT)).decode().strip()


def _package_versions():
    # the production graph's k-NN tie-breaking differs between scikit-learn 1.8.0 and 1.9.1
    import importlib.metadata as md
    return {p: md.version(p) for p in
            ('scikit-learn', 'numpy', 'pandas', 'geopandas', 'shapely', 'pyogrio', 'scipy', 'networkx')}


def _load_tract_list():
    text = TRACT_SELECTION.read_text()
    return text.split('fips_list:')[1].split()


def build_geojson(paper1):
    tracts = gpd.read_file(TRACT_SHP)
    tracts = tracts[tracts['COUNTYFP'] == '065'].copy()
    tracts['GEOID'] = tracts['GEOID'].astype(str).str.strip()
    svi = pd.read_csv(SVI_CSV, dtype={'FIPS': str}, usecols=['FIPS', 'RPL_THEMES'])
    svi['RPL_THEMES'] = svi['RPL_THEMES'].where(svi['RPL_THEMES'] != -999)
    out = tracts[['GEOID', 'geometry']].merge(
        svi.rename(columns={'FIPS': 'GEOID'}), on='GEOID', how='left')
    out['paper1'] = out['GEOID'].isin(set(paper1))
    out = out.to_crs('EPSG:4326').sort_values('GEOID').reset_index(drop=True)
    path = OUT_DIR / 'hamilton_tracts_svi.geojson'
    out[['GEOID', 'RPL_THEMES', 'paper1', 'geometry']].to_file(path, driver='GeoJSON')
    return path, {
        'n_tracts': int(len(out)),
        'n_with_svi': int(out['RPL_THEMES'].notna().sum()),
        'n_paper1': int(out['paper1'].sum()),
        'missing_svi_geoids': out.loc[out['RPL_THEMES'].isna(), 'GEOID'].tolist(),
    }


def build_inset(fips):
    dl = DataLoader(str(REPO_ROOT / 'data'), config={})
    dl.config['graph_variant'] = 'production'
    addresses = dl.get_addresses_for_tract(fips)
    n = len(addresses)
    g = dl.create_spatial_accessibility_graph(addresses, np.zeros((n, 10), dtype=np.float32))
    edge_index = g.edge_index.numpy().astype(np.int64)
    edge_weight = g.edge_attr.numpy().astype(np.float64)
    coords = np.array([[p.x, p.y] for p in addresses.geometry], dtype=np.float64)

    # window: the WINDOW_N nodes nearest the node closest to the median coordinate
    med = np.median(coords, axis=0)
    center = int(np.argmin(((coords - med) ** 2).sum(axis=1)))
    d2 = ((coords - coords[center]) ** 2).sum(axis=1)
    window = np.sort(np.argsort(d2, kind='stable')[:WINDOW_N]).astype(np.int64)

    n_undirected = edge_index.shape[1] // 2
    in_win = np.zeros(n, dtype=bool)
    in_win[window] = True
    keep = in_win[edge_index[0]] & in_win[edge_index[1]]
    win_directed = int(keep.sum())
    full_deg = np.bincount(edge_index[0], minlength=n)

    path = OUT_DIR / f'inset_edges_{fips}.npz'
    np.savez(path, edge_index=edge_index, edge_weight=edge_weight, coords=coords,
             window_idx=window, center_idx=np.int64(center))
    stats = {
        'fips': fips,
        'n_nodes': int(n),
        'n_directed_edges': int(edge_index.shape[1]),
        'n_undirected_edges': int(n_undirected),
        'full_tract_mean_degree': 2.0 * n_undirected / n,
        'coords_crs': str(addresses.crs),
        'center_idx': center,
        'window_n_nodes': int(len(window)),
        'window_induced_undirected_edges': win_directed // 2,
        'window_induced_mean_degree': win_directed / len(window),
        'window_full_graph_mean_degree': float(full_deg[window].mean()),
    }
    return path, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inset-fips', required=True)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paper1 = _load_tract_list()
    assert len(paper1) == 20 and args.inset_fips in paper1

    gj_path, gj_stats = build_geojson(paper1)
    npz_path, inset_stats = build_inset(args.inset_fips)

    prov = {
        'milestone': 'FIG-P1 B1',
        'command': f'python scripts/build_figure1_artifacts.py --inset-fips {args.inset_fips}',
        'commit': _git_sha(),
        'commit_note': 'HEAD at build time; this script is committed together with its outputs',
        'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'tract_selection': str(TRACT_SELECTION.relative_to(REPO_ROOT)),
        'inset_rule': 'paper1 tract nearest the median address count (2011.5); '
                      'tie between 47065000600 and 47065011444 resolved to 47065000600 by the PI',
        'packages': _package_versions(),
        'raw_sources_sha256': {str(p.relative_to(REPO_ROOT)): _sha256(p) for p in RAW_SOURCES},
        'outputs': {
            str(gj_path.relative_to(REPO_ROOT)): {'sha256': _sha256(gj_path), **gj_stats},
            str(npz_path.relative_to(REPO_ROOT)): {'sha256': _sha256(npz_path), **inset_stats},
        },
    }
    (OUT_DIR / 'provenance.json').write_text(json.dumps(prov, indent=2) + '\n')
    print(json.dumps({'geojson': gj_stats, 'inset': inset_stats}, indent=2))


if __name__ == '__main__':
    main()
