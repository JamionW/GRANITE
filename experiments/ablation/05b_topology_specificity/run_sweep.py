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

Seed handling:
  structured conditions: 5 training seeds [42, 17, 123, 2024, 7], one fixed graph
  randomized: training_seed=42, 5 graph_draw_seeds [42, 17, 123, 2024, 7]

Usage:
    python experiments/ablation/05b_topology_specificity/run_sweep.py
    python experiments/ablation/05b_topology_specificity/run_sweep.py --condition spatial_knn_uniform
    python experiments/ablation/05b_topology_specificity/run_sweep.py --dry-run
    python experiments/ablation/05b_topology_specificity/run_sweep.py --check-only
"""
import argparse
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

ARCHITECTURES = ['sage', 'gcn_gat']
ARCH_LABELS = {'sage': 'GRANITE-SAGE', 'gcn_gat': 'GRANITE-GCNGAT'}
TRAINING_SEEDS = [42, 17, 123, 2024, 7]
GRAPH_DRAW_SEEDS = [42, 17, 123, 2024, 7]
CONDITIONS = ['spatial_knn_uniform', 'road_network_uniform', 'randomized']
STRUCTURED_CONDITIONS = ['spatial_knn_uniform', 'road_network_uniform']
GRAPH_KNN_K = 10

MIN_ADDRESSES_PER_BG = 3
MIN_ADDRESSES_POOLED = 10

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
    dl.config['graph_knn_k'] = GRAPH_KNN_K

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
             cfg_base, pipeline, data, bg_gdf, validator, tract_list):
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
    cfg['graph_knn_k'] = GRAPH_KNN_K
    cfg['graph_draw_seed'] = graph_draw_seed

    pipeline.config = cfg
    pipeline.data_loader.config['graph_variant'] = condition
    pipeline.data_loader.config['graph_knn_k'] = GRAPH_KNN_K
    pipeline.data_loader.config['graph_draw_seed'] = graph_draw_seed
    pipeline.data_loader.config['processing'] = cfg['processing']

    # reset under_mixed counter for this seed run
    pipeline.data_loader._under_mixed_tracts = 0

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
    }


# ---------------------------------------------------------------------------
# condition sweep
# ---------------------------------------------------------------------------

def run_condition(condition, cfg_base, bg_gdf, validator, tract_list, verbose=False):
    """Run all (arch, seed) combinations for one condition. Returns nested results dict."""
    print(f'\n[step05b] === condition: {condition} ===')
    cond_dir = STEP5B_ROOT / condition
    cond_dir.mkdir(parents=True, exist_ok=True)
    (cond_dir / 'results').mkdir(exist_ok=True)
    (cond_dir / 'figures').mkdir(exist_ok=True)

    cfg_preflight = {k: (dict(v) if isinstance(v, dict) else v) for k, v in cfg_base.items()}
    cfg_preflight['graph_variant'] = condition
    cfg_preflight['graph_knn_k'] = GRAPH_KNN_K
    _write_preflight(cond_dir, cfg_preflight, tract_list)

    scratch_dir = str(REPO_ROOT / 'output' / f'step05b_{condition}_scratch')
    os.makedirs(scratch_dir, exist_ok=True)
    pipeline_cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in cfg_base.items()}
    pipeline_cfg['data'] = dict(cfg_base.get('data', {}))
    pipeline_cfg['data']['target_fips'] = tract_list[0]
    pipeline_cfg['graph_variant'] = condition
    pipeline_cfg['graph_knn_k'] = GRAPH_KNN_K
    pipeline = GRANITEPipeline(pipeline_cfg, output_dir=scratch_dir)
    pipeline.verbose = verbose

    print('[step05b] loading spatial data...')
    data = pipeline._load_spatial_data()

    # determine seed iteration strategy
    if condition == 'randomized':
        # training_seed fixed at 42; iterate over graph_draw_seeds
        seed_pairs = [(42, gs) for gs in GRAPH_DRAW_SEEDS]
        seed_label = 'graph_draw_seed'
    else:
        # training_seed varies; graph_draw_seed irrelevant (set to 42)
        seed_pairs = [(ts, 42) for ts in TRAINING_SEEDS]
        seed_label = 'training_seed'

    results = {}
    for arch in ARCHITECTURES:
        arch_seed_results = []
        print(f'\n[step05b] {condition} / {ARCH_LABELS[arch]}')

        for training_seed, graph_draw_seed in seed_pairs:
            varying_seed = graph_draw_seed if condition == 'randomized' else training_seed
            print(f'[step05b] {seed_label}={varying_seed}')
            t0 = time.time()
            seed_result = run_seed(
                condition, arch, training_seed, graph_draw_seed,
                cfg_base, pipeline, data, bg_gdf, validator, tract_list
            )
            seed_result['runtime_s'] = round(time.time() - t0, 1)
            arch_seed_results.append(seed_result)
            print(
                f'[step05b] {seed_label}={varying_seed} done: '
                f'std={seed_result["within_tract_std"]:.4f} '
                f'moran={seed_result["morans_i"]:.4f} '
                f'bg_r={seed_result["pooled_bg_r"]:.4f} '
                f't={seed_result["runtime_s"]:.0f}s'
            )

        # per-condition / per-arch aggregates; across-seed std from 5 values, not 20 tracts
        finite_stds  = [r['within_tract_std'] for r in arch_seed_results if math.isfinite(r['within_tract_std'])]
        finite_moran = [r['morans_i']         for r in arch_seed_results if math.isfinite(r['morans_i'])]
        finite_bg_r  = [r['pooled_bg_r']      for r in arch_seed_results if math.isfinite(r['pooled_bg_r'])]

        # acceptance criterion 2: band from 5 seed values, not 20 tract values
        assert len(finite_stds) == len(TRAINING_SEEDS), (
            f'expected {len(TRAINING_SEEDS)} finite within_tract_std values for '
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
    parser.add_argument('--condition', choices=CONDITIONS + ['all'], default='all',
                        help='which condition to run (default: all)')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--dry-run', action='store_true',
                        help='validate setup without running training')
    parser.add_argument('--check-only', action='store_true',
                        help='run preflight (degree/jaccard/swap-guard) on all tracts and exit')
    args = parser.parse_args()

    ts_start = datetime.now()
    print(f'[step05b] start {ts_start.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'[step05b] conditions={CONDITIONS} architectures={ARCHITECTURES}')
    print(f'[step05b] training_seeds={TRAINING_SEEDS}')
    print(f'[step05b] graph_draw_seeds={GRAPH_DRAW_SEEDS}')
    print(f'[step05b] graph_knn_k={GRAPH_KNN_K}')

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
        print('[step05b] dry-run complete')
        return

    cfg_base = _load_base_config()
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

    if args.check_only:
        print('[step05b] check-only mode: running preflight validation')
        run_preflight_check(cfg_base, tract_list)
        print('\n[step05b] check-only complete, exit 0')
        sys.exit(0)

    print('[step05b] loading BG geodataframe...')
    bg_gdf = _load_bg_gdf()
    validator = BlockGroupValidator(bg_gdf, verbose=False)

    conditions_to_run = CONDITIONS if args.condition == 'all' else [args.condition]

    results_path = STEP5B_ROOT / 'results' / 'topology_specificity_metrics.json'
    results_path.parent.mkdir(exist_ok=True)
    if results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)
        print(f'[step05b] loaded existing results from {results_path}')
    else:
        all_results = {}

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
        if cond in all_results:
            print(f'[step05b] {cond} already in results, skipping (delete to rerun)')
            continue
        cond_results = run_condition(
            cond, cfg_base, bg_gdf, validator, tract_list, args.verbose
        )
        all_results[cond] = cond_results

        cond_results_path = STEP5B_ROOT / cond / 'results' / f'{cond}_metrics.json'
        with open(cond_results_path, 'w') as f:
            json.dump(cond_results, f, indent=2,
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
