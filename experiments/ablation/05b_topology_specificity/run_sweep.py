"""
05b_topology_specificity: discriminate road-network wiring vs generic within-tract smoothing.

Three conditions, uniform edge weights across all three (neighbor selection is the only
moving part):
  spatial_knn_uniform  -- euclidean k-NN, k=10, w=1.0
  road_network_uniform -- road-snapped shortest-path k-NN, k=10, w=1.0
  randomized           -- degree-preserving double-edge swap on spatial_knn_uniform

Both architectures (SAGE, GCN-GAT). constraint_mode=soft, variation_weight=0.8.
Primary metric: Moran's I (computed on fixed k=8 address-coordinate weights, not the
message-passing graph).

Seed handling (--design, default single_axis):
  structured conditions: 5 training seeds [42, 17, 123, 2024, 7], one fixed graph
  randomized (single_axis): training_seed=42, 5 graph_draw_seeds [42, 17, 123, 2024, 7]
  randomized (crossed): Cartesian product of 5 training seeds x 5 graph draws = 25
    cells, paired on training seed with the structured spatial_knn_uniform arm
    (Dr. Gao design, 2026-10-21). structured conditions are unaffected by --design.
    crossed writes to data/results/m6_topology_5b_crossed/ (override with --out-dir);
    never to the frozen data/results/m6_topology_5b/.

Usage:
    python experiments/ablation/05b_topology_specificity/run_sweep.py
    python experiments/ablation/05b_topology_specificity/run_sweep.py --condition spatial_knn_uniform
    python experiments/ablation/05b_topology_specificity/run_sweep.py --dry-run
    python experiments/ablation/05b_topology_specificity/run_sweep.py --check-only
    python experiments/ablation/05b_topology_specificity/run_sweep.py --condition randomized --design crossed

Swap guard (SWAP-GUARD): every randomized cell is preceded by a graph-only swap check
over all tracts at the pending graph draws. A tract whose randomized edge list equals
its input raises GraphRandomizationError unless named in --allow-unswapped FIPS[,FIPS];
an allowlisted tract trains on its unswapped graph and is recorded randomized=False.
Per-tract swap records go to swap_provenance.csv beside trials_incremental.csv.
"""
import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from granite.disaggregation.pipeline import GRANITEPipeline
from granite.data.block_group_loader import BlockGroupLoader
from granite.validation.block_group_validation import BlockGroupValidator
from granite.evaluation.spatial_diagnostics import SpatialLearningDiagnostics
from granite.models.gnn import set_random_seed
from granite.errors import GraphRandomizationError

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
ABLATION_DIR = Path(__file__).resolve().parent
STEP5B_ROOT = ABLATION_DIR
TRACT_SELECTION = REPO_ROOT / 'experiments' / 'ablation' / '00_baseline' / 'tract_selection.txt'
BASE_CONFIG_PATH = (
    REPO_ROOT / 'experiments' / 'ablation' / '03_smoothness' / '02_default' / 'config_snapshot.yaml'
)
BG_SVI_PATH = REPO_ROOT / 'data' / 'processed' / 'national_bg_svi.csv'

# canonical output paths (committed, not scratch)
CANONICAL_DIR         = REPO_ROOT / 'data' / 'results' / 'm6_topology_5b'
PRED_DIR              = CANONICAL_DIR / 'predictions'
CANONICAL_METRICS     = CANONICAL_DIR / 'topology_specificity_metrics.json'
CANONICAL_TRIALS_CSV  = CANONICAL_DIR / 'trials_incremental.csv'

ARCHITECTURES = ['sage', 'gcn_gat']
ARCH_LABELS = {'sage': 'GRANITE-SAGE', 'gcn_gat': 'GRANITE-GCNGAT'}
TRAINING_SEEDS = [42, 17, 123, 2024, 7]
GRAPH_DRAW_SEEDS = [42, 17, 123, 2024, 7]
CONDITIONS = ['spatial_knn_uniform', 'road_network_uniform', 'randomized', 'production']
# mlp_floor (self-loops only, loaders.py) runs only when named; 'all' keeps the four above
STRUCTURED_CONDITIONS = ['spatial_knn_uniform', 'road_network_uniform']

MIN_ADDRESSES_PER_BG = 3
MIN_ADDRESSES_POOLED = 10

# incremental csv columns; the two swap flags survive resume (SWAP-GUARD)
TRIAL_FIELDS = [
    'condition', 'arch', 'training_seed', 'graph_draw_seed',
    'morans_i', 'within_tract_std', 'pooled_bg_r',
    'n_tracts', 'runtime_s', 'under_mixed_tracts', 'n_unswapped_tracts',
]
SWAP_FLAG_FIELDS = ['under_mixed_tracts', 'n_unswapped_tracts']
PROVENANCE_FIELDS = [
    'condition', 'arch', 'training_seed', 'graph_draw_seed', 'tract_fips',
    'n_edges', 'edges_preserved', 'share_preserved', 'randomized',
    'swaps_requested', 'swap_outcome', 'allowlisted',
]

# degree parity tolerance: mean node degree must agree within this value
DEGREE_PARITY_TOL = 0.5

# step 5 reference values for figure horizontal lines
# source: experiments/ablation/05_graph_contribution/results/graph_contribution_metrics.json
STEP5_REFS = {
    'sage': {
        'production': {'morans_i': 0.8570, 'within_tract_std': 0.0899, 'pooled_bg_r': 0.7632},
        'mlp_floor':  {'morans_i': 0.6820, 'within_tract_std': 0.0832, 'pooled_bg_r': 0.7714},
    },
    'gcn_gat': {
        'production': {'morans_i': 0.8368, 'within_tract_std': 0.0906, 'pooled_bg_r': 0.7639},
        'mlp_floor':  {'morans_i': 0.6747, 'within_tract_std': 0.0812, 'pooled_bg_r': 0.7660},
    },
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_tract_list():
    fips = []
    in_list = False
    with open(TRACT_SELECTION) as f:
        for line in f:
            line = line.strip()
            if line == 'fips_list:':
                in_list = True
                continue
            if in_list and line:
                fips.append(line.lstrip('- ').strip())
    return fips


def _load_base_config():
    with open(BASE_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f) or {}
    for k in ('data', 'model', 'training', 'processing', 'features',
              'recovery', 'validation', 'norm_layers'):
        cfg.setdefault(k, {})
    cfg.get('training', {}).pop('smoothness_weight', None)
    return cfg


def _load_bg_gdf():
    loader = BlockGroupLoader(data_dir=str(REPO_ROOT / 'data'), verbose=False)
    bg_gdf = loader.get_block_groups_with_demographics('47', '065', svi_ranking_scope='national')
    if bg_gdf is None or len(bg_gdf) == 0:
        raise RuntimeError('BlockGroupLoader returned empty GeoDataFrame')
    if bg_gdf.crs is None:
        bg_gdf = bg_gdf.set_crs('EPSG:4326')
    elif bg_gdf.crs.to_epsg() != 4326:
        bg_gdf = bg_gdf.to_crs('EPSG:4326')
    return bg_gdf


def _aggregate_to_bg(validator, address_gdf, preds, min_addresses):
    if address_gdf.crs is None:
        address_gdf = address_gdf.set_crs('EPSG:4326')
    addresses_with_bg = validator._assign_to_block_groups(address_gdf)
    bg_agg = validator._aggregate_predictions(addresses_with_bg, preds)
    return bg_agg[bg_agg['n_addresses'] >= min_addresses].copy()


def _bg_metrics(bg_agg, bg_gdf):
    svi_lookup = bg_gdf[['GEOID', 'SVI']].dropna(subset=['SVI'])
    merged = bg_agg.merge(svi_lookup, on='GEOID', how='inner')
    n = len(merged)
    if n < 2:
        return {'bg_r': float('nan'), 'bg_rmse': float('nan'), 'n_bgs': n}
    p = merged['predicted_svi'].values.astype(float)
    t = merged['SVI'].values.astype(float)
    valid = np.isfinite(p) & np.isfinite(t)
    if valid.sum() < 2:
        return {'bg_r': float('nan'), 'bg_rmse': float('nan'), 'n_bgs': n}
    return {
        'bg_r':    float(np.corrcoef(p[valid], t[valid])[0, 1]),
        'bg_rmse': float(np.sqrt(np.mean((p[valid] - t[valid]) ** 2))),
        'n_bgs':   int(valid.sum()),
    }


def _compute_morans_i(predictions, address_gdf, k=8):
    try:
        coords = np.array([[g.x, g.y] for g in address_gdf.geometry])
        return SpatialLearningDiagnostics().compute_spatial_autocorrelation(
            predictions, coords, k_neighbors=k)
    except Exception:
        return float('nan')


def _git_sha():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=str(REPO_ROOT)
        ).decode().strip()
    except Exception:
        return 'unknown'


def _pred_path(condition, arch, training_seed, graph_draw_seed, fips):
    """npz path for persisted (predictions, coords) for one trial×fips cell."""
    return PRED_DIR / (
        f'{condition}__{arch}__ts{training_seed}__gs{graph_draw_seed}__{fips}.npz'
    )


def _save_pred_atomic(path, preds, coords, raw=None):
    """atomically write predictions, EPSG:4326 coords and, if given, the raw trainer output."""
    import os
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.npz')
    arrays = {'predictions': np.asarray(preds, dtype=float),
              'coords': np.asarray(coords, dtype=float)}
    if raw is not None:
        arrays['raw_prediction'] = np.asarray(raw, dtype=float)
    try:
        np.savez(tmp, **arrays)
        os.replace(str(tmp), str(path))
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def _write_preflight(cond_dir, cfg, tract_list):
    sha = _git_sha()
    try:
        diff_stat = subprocess.check_output(
            ['git', 'diff', '--stat'], cwd=str(REPO_ROOT)
        ).decode().strip()
        git_txt = f'sha: {sha}\ndiff_stat:\n{diff_stat}\n'
    except Exception as e:
        git_txt = f'sha: {sha}\ngit diff error: {e}\n'
    (cond_dir / 'git_state.txt').write_text(git_txt)
    with open(cond_dir / 'config_snapshot.yaml', 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    try:
        env_txt = subprocess.check_output(['pip', 'freeze']).decode()
    except Exception:
        env_txt = 'pip freeze failed\n'
    (cond_dir / 'environment.txt').write_text(env_txt)
    import shutil
    shutil.copy(str(TRACT_SELECTION), str(cond_dir / 'tract_selection.txt'))
    print(f'[step05b] preflight written to {cond_dir}')


# ---------------------------------------------------------------------------
# preflight: degree parity, jaccard, swap-guard (all 20 tracts)
# ---------------------------------------------------------------------------

def _neighbor_sets_from_edge_index(edge_index_tensor, n_nodes):
    """return list of sets, one per node, containing neighbor indices"""
    sets = [set() for _ in range(n_nodes)]
    ei = edge_index_tensor.numpy()
    for col in range(ei.shape[1]):
        u, v = int(ei[0, col]), int(ei[1, col])
        sets[u].add(v)
    return sets


def _jaccard_tract(ei_a, ei_b, n_nodes):
    """mean per-node Jaccard between two edge_index tensors for one tract"""
    sets_a = _neighbor_sets_from_edge_index(ei_a, n_nodes)
    sets_b = _neighbor_sets_from_edge_index(ei_b, n_nodes)
    scores = []
    for sa, sb in zip(sets_a, sets_b):
        union = sa | sb
        if not union:
            continue
        scores.append(len(sa & sb) / len(union))
    return float(np.mean(scores)) if scores else float('nan')


def run_preflight_check(cfg_base, tract_list):
    """
    Build graphs for all three conditions on all 20 in-scope tracts.

    Prints (in order):
      - per-tract row: fips, n_nodes, edges per condition, road fallback count, jaccard
      - per-condition degree table (mean degree, total edges, n_tracts)
      - under_mixed_tracts count (randomized swap failures)
      - jaccard distribution: mean / median / min / max / per-tract vector

    Assertions (fail-loud, abort on failure):
      - |road_mean_degree - spatial_mean_degree| <= DEGREE_PARITY_TOL
      - randomized total_edges == spatial total_edges (exact; swap preserves degree)

    Returns dict with all evidence for the results json. Does not write files.
    """
    from granite.data.loaders import DataLoader

    dl = DataLoader(str(REPO_ROOT / 'data'), config=dict(cfg_base))

    # accumulators
    # per condition: list of (n_nodes, n_undirected_edges)
    tract_data = {cond: [] for cond in CONDITIONS}
    road_fallback_per_tract = []   # (fips, n_fallback)
    jaccard_per_tract = []         # float per tract, spatial vs road
    under_mixed = 0

    print(f'\n[step05b] preflight: building graphs for {len(tract_list)} tracts...')
    print(
        f'{"fips":<14} {"n":>5} '
        f'{"spatial_e":>9} {"road_e":>7} {"rand_e":>7} '
        f'{"fallback":>8} {"jaccard":>8}'
    )

    for fips in tract_list:
        try:
            addresses = dl.get_addresses_for_tract(fips)
        except Exception as e:
            print(f'[step05b]   {fips}: address load failed: {e}')
            continue

        n = len(addresses)
        if n < 2:
            print(f'[step05b]   {fips}: only {n} addresses, skipping')
            continue

        dummy_features = np.zeros((n, 10), dtype=np.float32)
        ei_by_cond = {}
        edges_by_cond = {}
        fallback_count = 0

        for condition in CONDITIONS:
            dl.config['graph_variant'] = condition
            dl.config['graph_draw_seed'] = 42
            if condition == 'randomized':
                dl._under_mixed_tracts = 0
            try:
                g = dl.create_spatial_accessibility_graph(addresses, dummy_features)
                ei = g.edge_index
                n_undirected = ei.shape[1] // 2
                ei_by_cond[condition] = ei
                edges_by_cond[condition] = n_undirected
                tract_data[condition].append((n, n_undirected))
                if condition == 'road_network_uniform':
                    fallback_count = getattr(dl, '_last_road_fallback', 0)
                if condition == 'randomized':
                    under_mixed += getattr(dl, '_under_mixed_tracts', 0)
            except GraphRandomizationError:
                raise
            except Exception as e:
                print(f'[step05b]   {fips}/{condition}: graph failed: {e}')
                ei_by_cond[condition] = None
                edges_by_cond[condition] = 0
                tract_data[condition].append((n, 0))

        road_fallback_per_tract.append((fips, fallback_count))

        # per-tract jaccard
        ei_spat = ei_by_cond.get('spatial_knn_uniform')
        ei_road = ei_by_cond.get('road_network_uniform')
        j = float('nan')
        if ei_spat is not None and ei_road is not None:
            j = _jaccard_tract(ei_spat, ei_road, n)
            jaccard_per_tract.append(j)

        print(
            f'{fips:<14} {n:>5} '
            f'{edges_by_cond.get("spatial_knn_uniform", 0):>9} '
            f'{edges_by_cond.get("road_network_uniform", 0):>7} '
            f'{edges_by_cond.get("randomized", 0):>7} '
            f'{fallback_count:>8} '
            f'{j:>8.3f}'
        )

    # -- degree table --
    print('\n[step05b] === degree table (summed across all tracts) ===')
    print(f'{"condition":<22} {"mean_degree":>12} {"total_edges":>12} {"n_tracts":>9}')
    cond_stats = {}
    for cond in CONDITIONS:
        pts = tract_data[cond]
        if not pts:
            cond_stats[cond] = {'mean_degree': float('nan'), 'total_edges': 0, 'n_tracts': 0}
            print(f'{cond:<22} {"nan":>12} {"0":>12} {"0":>9}')
            continue
        total_nodes = sum(nn for nn, _ in pts)
        total_edges = sum(ne for _, ne in pts)
        # mean degree = 2 * undirected_edges / nodes (bidirectional representation)
        mean_degree = 2.0 * total_edges / total_nodes if total_nodes > 0 else float('nan')
        cond_stats[cond] = {
            'mean_degree': mean_degree,
            'total_edges': total_edges,
            'n_tracts': len(pts),
        }
        print(f'{cond:<22} {mean_degree:>12.3f} {total_edges:>12} {len(pts):>9}')

    # -- road fallback summary --
    print('\n[step05b] road_network_uniform fallback counts per tract:')
    for fips, fb in road_fallback_per_tract:
        if fb > 0:
            print(f'  {fips}: {fb} unsnapped nodes fell back to euclidean')
    total_fb = sum(fb for _, fb in road_fallback_per_tract)
    print(f'  total fallback nodes across all tracts: {total_fb}')

    # -- under_mixed --
    print(f'\n[step05b] under_mixed_tracts (randomized swap failures): {under_mixed}')

    # -- jaccard distribution --
    print('\n[step05b] === jaccard(spatial, road) across tracts ===')
    jaccard_stats = {}
    if jaccard_per_tract:
        j_arr = np.array([j for j in jaccard_per_tract if math.isfinite(j)])
        if len(j_arr) > 0:
            jaccard_stats = {
                'mean':      float(np.mean(j_arr)),
                'median':    float(np.median(j_arr)),
                'min':       float(np.min(j_arr)),
                'max':       float(np.max(j_arr)),
                'per_tract': j_arr.tolist(),
                'n_tracts':  len(j_arr),
            }
            print(
                f'  n={len(j_arr)}  mean={jaccard_stats["mean"]:.3f}  '
                f'median={jaccard_stats["median"]:.3f}  '
                f'min={jaccard_stats["min"]:.3f}  max={jaccard_stats["max"]:.3f}'
            )
            print(f'  per-tract: {[round(x, 3) for x in j_arr.tolist()]}')
            if jaccard_stats['mean'] > 0.85:
                print(
                    '  WARNING: high mean Jaccard; spatial-vs-road tie is uninformative'
                )
    else:
        print('  no Jaccard values computed')

    # -- assertions --
    print('\n[step05b] === degree parity assertions ===')
    spat_deg   = cond_stats.get('spatial_knn_uniform', {}).get('mean_degree', float('nan'))
    road_deg   = cond_stats.get('road_network_uniform', {}).get('mean_degree', float('nan'))
    spat_edges = cond_stats.get('spatial_knn_uniform', {}).get('total_edges', -2)
    rand_edges = cond_stats.get('randomized', {}).get('total_edges', -1)

    passed = True

    # fix 1: road vs spatial (not road vs k)
    if math.isfinite(spat_deg) and math.isfinite(road_deg):
        gap = abs(road_deg - spat_deg)
        ok = gap <= DEGREE_PARITY_TOL
        print(
            f'  road vs spatial: road={road_deg:.3f} spatial={spat_deg:.3f} '
            f'gap={gap:.3f} tol={DEGREE_PARITY_TOL} [{"OK" if ok else "FAIL"}]'
        )
        if not ok:
            passed = False
    else:
        print(f'  road vs spatial: cannot assert (road={road_deg} spatial={spat_deg})')
        passed = False

    # randomized must have exactly the same total edge count as spatial
    if rand_edges == spat_edges:
        print(f'  randomized total_edges == spatial: {rand_edges} [OK]')
    else:
        print(
            f'  randomized total_edges={rand_edges} != spatial total_edges={spat_edges} [FAIL]'
        )
        passed = False

    if not passed:
        print('\n[step05b] DEGREE PARITY FAILED -- confound detected; aborting')
        sys.exit(5)

    print('[step05b] degree parity passed')

    return {
        'degrees': {c: cond_stats[c]['mean_degree'] for c in CONDITIONS},
        'total_edges': {c: cond_stats[c]['total_edges'] for c in CONDITIONS},
        'jaccard_spatial_road': jaccard_stats,
        'under_mixed_tracts': under_mixed,
        'road_fallback_total': total_fb,
    }


# ---------------------------------------------------------------------------
# single seed run
# ---------------------------------------------------------------------------

def run_seed(condition, arch, training_seed, graph_draw_seed,
             cfg_base, pipeline, data, bg_gdf, validator, tract_list,
             persist_preds=False):
    """
    Run one (condition, arch, seed) combination over all tracts.
    Returns dict with per-seed aggregate metrics.
    For structured conditions, the varying seed is training_seed.
    For randomized, training_seed=42 and graph_draw_seed varies.
    """
    set_random_seed(training_seed)

    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in cfg_base.items()}
    cfg['processing'] = dict(cfg_base.get('processing', {}))
    cfg['processing']['random_seed'] = training_seed
    cfg['model'] = dict(cfg_base.get('model', {}))
    cfg['model']['architecture'] = arch
    cfg['graph_variant'] = condition
    cfg['graph_draw_seed'] = graph_draw_seed

    pipeline.config = cfg
    pipeline.data_loader.config['graph_variant'] = condition
    pipeline.data_loader.config['graph_draw_seed'] = graph_draw_seed
    pipeline.data_loader.config['graph_knn_k'] = cfg.get('graph_knn_k', 10)
    pipeline.data_loader.config['road_knn_k'] = cfg.get('road_knn_k', cfg.get('graph_knn_k', 10))
    pipeline.data_loader.config['processing'] = cfg['processing']
    pipeline.data_loader.config['allow_unswapped'] = list(cfg_base.get('allow_unswapped', []))

    # reset under_mixed counter and swap records for this seed run
    pipeline.data_loader._under_mixed_tracts = 0
    pipeline.data_loader._swap_records = []

    per_tract_stds = []
    per_tract_morans = []
    pooled_addr_gdfs = []
    pooled_preds = []

    for fips in tract_list:
        cfg['data'] = dict(cfg_base.get('data', {}))
        cfg['data']['target_fips'] = fips
        pipeline.config['data']['target_fips'] = fips

        try:
            result = pipeline._process_single_tract(fips, data)
        except GraphRandomizationError:
            # unswapped, non-allowlisted tract: abort the run, never skip the tract
            raise
        except Exception as e:
            print(f'[step05b]   ERROR {condition}/{arch}/tseed={training_seed}/gseed={graph_draw_seed}/{fips}: {str(e)[:200]}')
            traceback.print_exc()
            continue

        if not result.get('success'):
            print(f'[step05b]   FAILED {condition}/{arch}/tseed={training_seed}/{fips}: '
                  f'{result.get("error", "?")[:200]}')
            continue

        address_gdf = result['address_gdf']
        preds_arr = result['predictions']['mean'].values.astype(float)
        tract_svi = float(result['tract_svi'])

        if persist_preds:
            pp = _pred_path(condition, arch, training_seed, graph_draw_seed, fips)
            if not pp.exists():
                # raw_prediction: trainer output before the inference shift and clip
                raw_arr = None
                if 'raw_prediction' in result['predictions'].columns:
                    raw_arr = result['predictions']['raw_prediction'].values.astype(float)
                _save_pred_atomic(
                    pp, preds_arr,
                    np.array([[g.x, g.y] for g in address_gdf.geometry]),
                    raw=raw_arr,
                )

        constr_err = abs(np.mean(preds_arr) - tract_svi)
        sp_std = float(np.std(preds_arr))
        morans_i = _compute_morans_i(preds_arr, address_gdf)

        per_tract_stds.append(sp_std)
        if math.isfinite(morans_i):
            per_tract_morans.append(morans_i)

        pooled_addr_gdfs.append(address_gdf.copy())
        pooled_preds.append(preds_arr)

        print(
            f'[step05b]   {condition}/{arch}/tseed={training_seed}/gseed={graph_draw_seed}/{fips}: '
            f'std={sp_std:.4f} moran={morans_i:.3f} constr={constr_err:.2e}'
        )

    # swap records exist only for the randomized variant; structured cells report 0
    swap_records = list(getattr(pipeline.data_loader, '_swap_records', []))
    for r in swap_records:
        if not r['randomized']:
            print(f'[step05b]   UNSWAPPED {condition}/{arch}/tseed={training_seed}/'
                  f'gseed={graph_draw_seed}/{r["tract_fips"]}: randomized=False '
                  f'(allowlisted={r["allowlisted"]}, outcome={r["swap_outcome"]})')

    within_tract_std = float(np.mean(per_tract_stds)) if per_tract_stds else float('nan')
    morans_i_mean = float(np.mean(per_tract_morans)) if per_tract_morans else float('nan')

    pooled_bg_r = float('nan')
    n_bgs = 0
    if pooled_addr_gdfs:
        try:
            all_addr = pd.concat(pooled_addr_gdfs, ignore_index=True)
            all_preds = np.concatenate(pooled_preds)
            if all_addr.crs is None:
                all_addr = all_addr.set_crs('EPSG:4326')
            bg_agg = _aggregate_to_bg(validator, all_addr, all_preds, MIN_ADDRESSES_PER_BG)
            bm = _bg_metrics(bg_agg, bg_gdf)
            pooled_bg_r = bm['bg_r']
            n_bgs = bm['n_bgs']
        except Exception as e:
            print(f'[step05b]   WARNING: pooled BG metrics failed: {e}')

    return {
        'training_seed': training_seed,
        'graph_draw_seed': graph_draw_seed,
        'within_tract_std': within_tract_std,
        'morans_i': morans_i_mean,
        'pooled_bg_r': pooled_bg_r,
        'n_bgs': n_bgs,
        'n_tracts': len(per_tract_stds),
        'under_mixed_tracts': int(getattr(pipeline.data_loader, '_under_mixed_tracts', 0)),
        'n_unswapped_tracts': sum(1 for r in swap_records if not r['randomized']),
        '_swap_records': swap_records,
    }


# ---------------------------------------------------------------------------
# condition sweep
# ---------------------------------------------------------------------------

def _load_completed_trials(csv_path):
    """read incremental CSV; return dict keyed by (condition, arch, training_seed, graph_draw_seed)."""
    if csv_path is None or not csv_path.exists():
        return {}
    completed = {}
    missing = []
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            key = (row['condition'], row['arch'],
                   int(row['training_seed']), int(row['graph_draw_seed']))
            # swap flags: carried through as read; absent or empty is None, never 0
            flags = {}
            for col in SWAP_FLAG_FIELDS:
                v = row.get(col)
                flags[col] = int(v) if v not in (None, '') else None
            if any(v is None for v in flags.values()):
                missing.append((key, [c for c, v in flags.items() if v is None]))
            completed[key] = {
                'training_seed':    int(row['training_seed']),
                'graph_draw_seed':  int(row['graph_draw_seed']),
                'morans_i':         float(row['morans_i']),
                'within_tract_std': float(row['within_tract_std']),
                'pooled_bg_r':      float(row['pooled_bg_r']),
                'n_tracts':         int(row['n_tracts']),
                'runtime_s':        float(row['runtime_s']),
                **flags,
            }
    if missing:
        print(f'[step05b] resume: {len(missing)} completed trial(s) in {csv_path.name} '
              f'missing swap flags (reported as missing, not filled):')
        for key, cols in missing:
            print(f'[step05b]   MISSING {"/".join(str(k) for k in key)}: {cols}')
    return completed


def _check_csv_header(csv_path, fields):
    """refuse to append rows to an existing csv whose header lacks fields."""
    if csv_path is None or not csv_path.exists():
        return
    with open(csv_path, newline='') as f:
        header = next(csv.reader(f), [])
    absent = [c for c in fields if c not in header]
    if absent:
        raise RuntimeError(
            f'{csv_path} predates SWAP-GUARD (header lacks {absent}); new cells must '
            f'go to a new --out-dir, not an existing results directory'
        )


def _append_trial_csv(csv_path, condition, arch, seed_result):
    """append one trial row to the incremental CSV; creates with header if new."""
    _check_csv_header(csv_path, TRIAL_FIELDS)
    is_new = not csv_path.exists()
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=TRIAL_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow({
            'condition':        condition,
            'arch':             arch,
            'training_seed':    seed_result.get('training_seed'),
            'graph_draw_seed':  seed_result.get('graph_draw_seed'),
            'morans_i':         seed_result.get('morans_i'),
            'within_tract_std': seed_result.get('within_tract_std'),
            'pooled_bg_r':      seed_result.get('pooled_bg_r'),
            'n_tracts':         seed_result.get('n_tracts'),
            'runtime_s':        seed_result.get('runtime_s'),
            'under_mixed_tracts': seed_result.get('under_mixed_tracts'),
            'n_unswapped_tracts': seed_result.get('n_unswapped_tracts'),
        })


def _append_swap_provenance(path, condition, arch, training_seed, graph_draw_seed, records):
    """append one row per tract swap record for a randomized cell."""
    is_new = not path.exists()
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=PROVENANCE_FIELDS, extrasaction='ignore')
        if is_new:
            writer.writeheader()
        for r in records:
            writer.writerow({**r, 'condition': condition, 'arch': arch,
                             'training_seed': training_seed,
                             'graph_draw_seed': graph_draw_seed})


def run_condition(condition, cfg_base, bg_gdf, validator, tract_list, verbose=False,
                  incremental_csv=None, archs=None, completed_trials=None,
                  persist_preds=False, design='single_axis', out_dir=None):
    """Run all (arch, seed) combinations for one condition. Returns nested results dict."""
    if archs is None:
        archs = ARCHITECTURES
    if completed_trials is None:
        completed_trials = {}
    print(f'\n[step05b] === condition: {condition} ===')
    # crossed runs keep the frozen single-axis provenance tree untouched: write
    # the per-condition preflight/figures under the crossed out dir instead
    cond_dir = (out_dir / condition) if out_dir is not None else (STEP5B_ROOT / condition)
    cond_dir.mkdir(parents=True, exist_ok=True)
    (cond_dir / 'results').mkdir(exist_ok=True)
    (cond_dir / 'figures').mkdir(exist_ok=True)

    cfg_preflight = {k: (dict(v) if isinstance(v, dict) else v) for k, v in cfg_base.items()}
    cfg_preflight['graph_variant'] = condition
    _write_preflight(cond_dir, cfg_preflight, tract_list)

    scratch_dir = str(REPO_ROOT / 'output' / f'step05b_{condition}_scratch')
    os.makedirs(scratch_dir, exist_ok=True)
    pipeline_cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in cfg_base.items()}
    pipeline_cfg['data'] = dict(cfg_base.get('data', {}))
    pipeline_cfg['data']['target_fips'] = tract_list[0]
    pipeline_cfg['graph_variant'] = condition
    pipeline = GRANITEPipeline(pipeline_cfg, output_dir=scratch_dir)
    pipeline.verbose = verbose

    print('[step05b] loading spatial data...')
    data = pipeline._load_spatial_data()

    # determine seed iteration strategy
    if condition == 'randomized':
        if design == 'crossed':
            # cross training seeds with graph draws; paired on training seed with
            # the structured spatial_knn_uniform arm (Dr. Gao design, 2026-10-21)
            seed_pairs = [(ts, gs) for ts in TRAINING_SEEDS for gs in GRAPH_DRAW_SEEDS]
        else:
            # training_seed fixed at 42; iterate over graph_draw_seeds
            seed_pairs = [(42, gs) for gs in GRAPH_DRAW_SEEDS]
        seed_label = 'graph_draw_seed'
    else:
        # training_seed varies; graph_draw_seed irrelevant (set to 42)
        seed_pairs = [(ts, 42) for ts in TRAINING_SEEDS]
        seed_label = 'training_seed'

    pending = [(a, ts, gs) for a in archs for ts, gs in seed_pairs
               if (condition, a, ts, gs) not in completed_trials]
    provenance_csv = None
    if pending:
        _check_csv_header(incremental_csv, TRIAL_FIELDS)
    if condition == 'randomized' and pending:
        if incremental_csv is not None:
            provenance_csv = incremental_csv.parent / 'swap_provenance.csv'
        # graph-only swap check at every pending graph draw; raises before training
        dl = pipeline.data_loader
        dl.config['graph_knn_k'] = cfg_base.get('graph_knn_k', 10)
        dl.config['allow_unswapped'] = list(cfg_base.get('allow_unswapped', []))
        pending_gs = sorted({gs for _, _, gs in pending})
        recs = dl.check_randomization(tract_list, pending_gs)
        n_unswapped = sum(1 for r in recs if not r['randomized'])
        print(f'[step05b] swap check: {len(tract_list)} tracts x {len(pending_gs)} graph draws, '
              f'{n_unswapped} unswapped, allowlisted: '
              f'{sorted({r["tract_fips"] for r in recs if not r["randomized"]})}')

    results = {}
    for arch in archs:
        arch_seed_results = []
        print(f'\n[step05b] {condition} / {ARCH_LABELS[arch]}')

        for training_seed, graph_draw_seed in seed_pairs:
            varying_seed = graph_draw_seed if condition == 'randomized' else training_seed
            if condition == 'randomized' and design == 'crossed':
                cell_label = f'tseed={training_seed}/gseed={graph_draw_seed}'
            else:
                cell_label = f'{seed_label}={varying_seed}'
            key = (condition, arch, training_seed, graph_draw_seed)
            if key in completed_trials:
                print(f'[step05b] {cell_label} already in CSV, skipping')
                arch_seed_results.append(completed_trials[key])
                continue
            print(f'[step05b] {cell_label}')
            t0 = time.time()
            seed_result = run_seed(
                condition, arch, training_seed, graph_draw_seed,
                cfg_base, pipeline, data, bg_gdf, validator, tract_list,
                persist_preds=persist_preds
            )
            seed_result['runtime_s'] = round(time.time() - t0, 1)
            swap_records = seed_result.pop('_swap_records', [])
            if provenance_csv is not None:
                _append_swap_provenance(provenance_csv, condition, arch,
                                        training_seed, graph_draw_seed, swap_records)
            arch_seed_results.append(seed_result)
            if incremental_csv is not None:
                _append_trial_csv(incremental_csv, condition, arch, seed_result)
                print(f'[step05b] trial row appended to {incremental_csv.name}')
            print(
                f'[step05b] {cell_label} done: '
                f'std={seed_result["within_tract_std"]:.4f} '
                f'moran={seed_result["morans_i"]:.4f} '
                f'bg_r={seed_result["pooled_bg_r"]:.4f} '
                f't={seed_result["runtime_s"]:.0f}s'
            )

        # per-condition / per-arch aggregates; across-seed std from 5 values, not 20 tracts
        finite_stds  = [r['within_tract_std'] for r in arch_seed_results if math.isfinite(r['within_tract_std'])]
        finite_moran = [r['morans_i']         for r in arch_seed_results if math.isfinite(r['morans_i'])]
        finite_bg_r  = [r['pooled_bg_r']      for r in arch_seed_results if math.isfinite(r['pooled_bg_r'])]

        # acceptance criterion 2: band from the seed cells for this condition
        # (5 for single-axis; 25 for the crossed randomized arm), not 20 tract values
        assert len(finite_stds) == len(seed_pairs), (
            f'expected {len(seed_pairs)} finite within_tract_std values for '
            f'{condition}/{arch}, got {len(finite_stds)}'
        )

        results[arch] = {
            'seeds': arch_seed_results,
            'mean': {
                'within_tract_std': float(np.mean(finite_stds)),
                'morans_i':         float(np.mean(finite_moran)) if finite_moran else float('nan'),
                'pooled_bg_r':      float(np.mean(finite_bg_r))  if finite_bg_r  else float('nan'),
            },
            'std': {
                'within_tract_std': float(np.std(finite_stds, ddof=1)) if len(finite_stds) > 1 else float('nan'),
                'morans_i':         float(np.std(finite_moran, ddof=1)) if len(finite_moran) > 1 else float('nan'),
                'pooled_bg_r':      float(np.std(finite_bg_r,  ddof=1)) if len(finite_bg_r)  > 1 else float('nan'),
            },
        }

    return results


# ---------------------------------------------------------------------------
# acceptance criterion checks
# ---------------------------------------------------------------------------

def check_constraints(results):
    """confirm non-finite std doesn't appear (proxy for constraint collapse)"""
    for cond in CONDITIONS:
        cond_res = results.get(cond, {})
        for arch in ARCHITECTURES:
            arch_res = cond_res.get(arch, {})
            for sr in arch_res.get('seeds', []):
                if not math.isfinite(sr.get('within_tract_std', float('nan'))):
                    print(
                        f'[step05b] WARNING: non-finite within_tract_std for '
                        f'{cond}/{arch}/seed={sr.get("training_seed","?")}'
                    )
    print('[step05b] constraint check complete (see per-tract logs for per-run error values)')


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', choices=CONDITIONS + ['mlp_floor', 'all'], default='all',
                        help='which condition to run (default: all)')
    parser.add_argument('--sage-only', action='store_true',
                        help='restrict to SAGE architecture only')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--design', choices=['single_axis', 'crossed'], default='single_axis',
                        help=(
                            'seed design. single_axis (default): randomized iterates graph '
                            'draws at training_seed=42 (reproduces committed results). crossed: '
                            'randomized iterates the Cartesian product TRAINING_SEEDS x '
                            'GRAPH_DRAW_SEEDS (25 cells). structured conditions are unaffected.'
                        ))
    parser.add_argument('--out-dir', default=None,
                        help=(
                            'output base dir for metrics/csv/predictions. crossed design '
                            'defaults to data/results/m6_topology_5b_crossed/. never '
                            'data/results/m6_topology_5b/ (frozen).'
                        ))
    parser.add_argument('--persist-preds', action='store_true',
                        help=(
                            'with --out-dir: save per-tract npz files under '
                            '<out-dir>/predictions/ and write per-condition provenance under '
                            '<out-dir>/<condition>/ instead of the frozen 05b tree.'
                        ))
    parser.add_argument('--allow-unswapped', default='',
                        help=(
                            'comma-separated tract FIPS allowed to train on an unswapped '
                            'randomized graph (recorded randomized=False, warned). any other '
                            'unswapped tract raises GraphRandomizationError.'
                        ))
    parser.add_argument('--dry-run', action='store_true',
                        help='validate setup without running training')
    parser.add_argument('--check-only', action='store_true',
                        help='run preflight (degree/jaccard/swap-guard) on all tracts and exit')
    parser.add_argument('--rerun-fresh', action='store_true',
                        help=(
                            'rerun all conditions under the fixed esda estimator; '
                            'persist per-trial predictions and coords to '
                            'data/results/m6_topology_5b/predictions/ (committed); '
                            'write corrected metrics to data/results/m6_topology_5b/'
                        ))
    args = parser.parse_args()

    ts_start = datetime.now()
    print(f'[step05b] start {ts_start.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'[step05b] conditions={CONDITIONS} architectures={ARCHITECTURES}')
    print(f'[step05b] training_seeds={TRAINING_SEEDS}')
    print(f'[step05b] graph_draw_seeds={GRAPH_DRAW_SEEDS}')

    if not BG_SVI_PATH.exists():
        print(f'[step05b] HALT: {BG_SVI_PATH} missing')
        sys.exit(1)

    tract_list = _load_tract_list()
    print(f'[step05b] tracts: {len(tract_list)}')
    if not tract_list:
        print('[step05b] HALT: empty tract list')
        sys.exit(1)

    if args.dry_run:
        print('[step05b] dry-run: validating ValueError on unknown variant')
        from granite.data.loaders import DataLoader
        dl = DataLoader(str(REPO_ROOT / 'data'), config={'graph_variant': 'bad_value'})
        try:
            dummy_addr = gpd.GeoDataFrame({'geometry': []})
            dl.create_spatial_accessibility_graph(dummy_addr, np.zeros((0, 10)))
            print('[step05b] DRY-RUN FAIL: expected ValueError not raised')
            sys.exit(4)
        except ValueError as e:
            print(f'[step05b] ValueError correctly raised: {e}')

        # enumerate the cells this invocation would run (condition, arch, tseed, gseed)
        conds_dry = CONDITIONS if args.condition == 'all' else [args.condition]
        archs_dry = ['sage'] if args.sage_only else ARCHITECTURES
        cells = []
        for cond in conds_dry:
            if cond == 'randomized':
                if args.design == 'crossed':
                    pairs = [(ts, gs) for ts in TRAINING_SEEDS for gs in GRAPH_DRAW_SEEDS]
                else:
                    pairs = [(42, gs) for gs in GRAPH_DRAW_SEEDS]
            else:
                pairs = [(ts, 42) for ts in TRAINING_SEEDS]
            for arch in archs_dry:
                for ts, gs in pairs:
                    cells.append((cond, arch, ts, gs))
        print(f'[step05b] dry-run design={args.design}: {len(cells)} cell(s)')
        n_rand = sum(1 for c in cells if c[0] == 'randomized')
        print(f'[step05b] dry-run: {n_rand} randomized cell(s) '
              f'({n_rand // max(1, len(archs_dry))} per architecture)')
        for cond, arch, ts, gs in cells:
            print(f'[step05b]   cell: {cond}/{arch}/tseed={ts}/gseed={gs}')
        print('[step05b] dry-run complete')
        return

    cfg_base = _load_base_config()
    with open(REPO_ROOT / 'config.yaml') as _f:
        _live = yaml.safe_load(_f) or {}
    cfg_base['graph_knn_k'] = _live.get('graph_knn_k', 10)
    cfg_base['road_knn_k'] = _live.get('road_knn_k', cfg_base['graph_knn_k'])
    print(f'[step05b] graph_knn_k={cfg_base["graph_knn_k"]} road_knn_k={cfg_base["road_knn_k"]}')
    cfg_base['data']['target'] = 'svi'
    cfg_base['data']['neighbor_tracts'] = 0
    cfg_base['data']['state_fips'] = '47'
    cfg_base['data']['county_fips'] = '065'
    cfg_base['processing']['skip_importance'] = True
    cfg_base['processing']['verbose'] = args.verbose
    cfg_base['processing']['enable_caching'] = True
    cfg_base['features']['feature_standardization'] = 'per_tract'
    cfg_base['training']['constraint_mode'] = 'soft'
    cfg_base['training']['apply_post_correction'] = True
    cfg_base['training']['variation_weight'] = 0.8
    cfg_base['allow_unswapped'] = [f.strip() for f in args.allow_unswapped.split(',') if f.strip()]
    if cfg_base['allow_unswapped']:
        print(f'[step05b] allow-unswapped: {cfg_base["allow_unswapped"]}')

    if args.check_only:
        print('[step05b] check-only mode: running preflight validation')
        run_preflight_check(cfg_base, tract_list)
        print('\n[step05b] check-only complete, exit 0')
        sys.exit(0)

    print('[step05b] loading BG geodataframe...')
    bg_gdf = _load_bg_gdf()
    validator = BlockGroupValidator(bg_gdf, verbose=False)

    if args.sage_only:
        archs_to_run = ['sage']
        print('[step05b] --sage-only: restricting to SAGE architecture')
    else:
        archs_to_run = ARCHITECTURES

    conditions_to_run = CONDITIONS if args.condition == 'all' else [args.condition]

    # per-condition provenance dir; redirected away from the frozen tree for crossed
    cond_out_dir = None

    if args.design == 'crossed':
        # crossed arm: redirect every output to a separate base dir; never the
        # frozen canonical dir. mirror the fixed-estimator / persisted-prediction
        # behavior of rerun_fresh so the (42,42) cell reproduces the committed value.
        global PRED_DIR
        out_base = Path(args.out_dir) if args.out_dir else (
            REPO_ROOT / 'data' / 'results' / 'm6_topology_5b_crossed'
        )
        if out_base.resolve() == CANONICAL_DIR.resolve():
            print('[step05b] HALT: crossed design must not write to canonical '
                  f'{CANONICAL_DIR}')
            sys.exit(2)
        out_base.mkdir(parents=True, exist_ok=True)
        cond_out_dir = out_base
        PRED_DIR = out_base / 'predictions'
        results_path = out_base / 'topology_specificity_metrics.json'
        incremental_csv = out_base / 'trials_incremental.csv'
        persist_preds = True
        if results_path.exists():
            with open(results_path) as f:
                all_results = json.load(f)
            print(f'[step05b] crossed: loaded existing results from {results_path}')
        else:
            all_results = {}
        print(f'[step05b] crossed: output -> {results_path}')
        print(f'[step05b] crossed: predictions -> {PRED_DIR}')
    elif args.rerun_fresh:
        results_path = CANONICAL_METRICS
        results_path.parent.mkdir(parents=True, exist_ok=True)
        incremental_csv = CANONICAL_TRIALS_CSV
        all_results = {}
        persist_preds = True
        print(f'[step05b] rerun-fresh: output -> {results_path}')
        print(f'[step05b] rerun-fresh: predictions -> {PRED_DIR}')
    elif args.out_dir:
        out_base = Path(args.out_dir)
        if out_base.resolve() == CANONICAL_DIR.resolve():
            print(f'[step05b] HALT: --out-dir must not be canonical {CANONICAL_DIR}')
            sys.exit(2)
        out_base.mkdir(parents=True, exist_ok=True)
        results_path = out_base / 'topology_specificity_metrics.json'
        incremental_csv = out_base / 'trials_incremental.csv'
        persist_preds = args.persist_preds
        if persist_preds:
            # keep provenance and npz under out_base; the 05b tree stays frozen
            cond_out_dir = out_base
            PRED_DIR = out_base / 'predictions'
            print(f'[step05b] persist-preds: predictions -> {PRED_DIR}')
        if results_path.exists():
            with open(results_path) as f:
                all_results = json.load(f)
            print(f'[step05b] loaded existing results from {results_path}')
        else:
            all_results = {}
    else:
        results_path = STEP5B_ROOT / 'results' / 'topology_specificity_metrics.json'
        results_path.parent.mkdir(exist_ok=True)
        incremental_csv = STEP5B_ROOT / 'results' / 'trials_incremental.csv'
        persist_preds = False
        if results_path.exists():
            with open(results_path) as f:
                all_results = json.load(f)
            print(f'[step05b] loaded existing results from {results_path}')
        else:
            all_results = {}

    if args.rerun_fresh and incremental_csv.exists():
        completed_trials = _load_completed_trials(incremental_csv)
        print(f'[step05b] rerun-fresh resume: {len(completed_trials)} trial(s) already done')
    else:
        completed_trials = _load_completed_trials(incremental_csv)
        if completed_trials:
            print(f'[step05b] {len(completed_trials)} completed trial(s) found in CSV; will skip')

    # preflight gate: run before sweep, write once, abort on failure
    parity_path = STEP5B_ROOT / 'results' / 'preflight.json'
    if not parity_path.exists():
        parity_result = run_preflight_check(cfg_base, tract_list)
        with open(parity_path, 'w') as f:
            json.dump(parity_result, f, indent=2,
                      default=lambda x: None if (isinstance(x, float) and math.isnan(x)) else x)
        print(f'[step05b] preflight result written to {parity_path}')
    else:
        print(f'[step05b] preflight already passed (loaded {parity_path}); skipping re-check')

    for cond in conditions_to_run:
        # skip only if every requested arch is already present in the JSON
        cond_done_archs = set(all_results.get(cond, {}).keys())
        archs_needed = [a for a in archs_to_run if a not in cond_done_archs]
        if not archs_needed:
            print(f'[step05b] {cond} already in results for all archs, skipping (delete to rerun)')
            continue
        if cond_done_archs:
            print(f'[step05b] {cond} partial ({sorted(cond_done_archs)} done); running {archs_needed}')
        cond_results = run_condition(
            cond, cfg_base, bg_gdf, validator, tract_list, args.verbose,
            incremental_csv=incremental_csv,
            archs=archs_needed,
            completed_trials=completed_trials,
            persist_preds=persist_preds,
            design=args.design,
            out_dir=cond_out_dir,
        )
        # merge new arch results with any already-persisted arches for this condition
        merged = dict(all_results.get(cond, {}))
        merged.update(cond_results)
        all_results[cond] = merged

        if not args.rerun_fresh and args.design != 'crossed' and cond_out_dir is None:
            cond_results_path = STEP5B_ROOT / cond / 'results' / f'{cond}_metrics.json'
            with open(cond_results_path, 'w') as f:
                json.dump(merged, f, indent=2,
                          default=lambda x: None if (isinstance(x, float) and math.isnan(x)) else x)
            print(f'[step05b] {cond} results written to {cond_results_path}')

        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2,
                      default=lambda x: None if (isinstance(x, float) and math.isnan(x)) else x)
        print(f'[step05b] aggregate results written to {results_path}')

    if all(c in all_results for c in CONDITIONS):
        check_constraints(all_results)

    ts_end = datetime.now()
    elapsed = (ts_end - ts_start).total_seconds() / 60
    print(f'\n[step05b] complete in {elapsed:.1f} min')
    print(f'[step05b] results at {results_path}')


if __name__ == '__main__':
    main()
