"""
Figure 4 synthetic randomization run (FIG-P1 B5).

One recovery-grid cell, the synthetic field held fixed at the committed seed-42 field
(data/synthetic/run_20260628_030416, loaded, not regenerated: the current generator
no longer reproduces it), GRANITE on coordinates_only features under two graphs:
  structured  spatial_knn_uniform k=10, 5 training seeds (graph draw seed 42)
  randomized  degree-preserving randomization of that graph, 5 training seeds x
              5 graph draws
Both architectures: 10 structured + 50 randomized cells, 20 tracts each. Seed lists
match the 5b crossed run (experiments/ablation/05b_topology_specificity/run_sweep.py).

Batch 1: structured (sage, gcn_gat) and randomized sage. Batch 2: randomized gcn_gat.

Per tract: recovery_r against known truth (run_grid._recovery_r) and output Moran's I
(run_grid._compute_morans_i_output, the B4 estimator). Per cell: unweighted tract
means and runtime_s.

Outputs (data/results/m6_fig4_randomization/):
  predictions/{condition}__{arch}__ts{ts}__gs{gs}__ds{dataseed}__{fips}.npz
      arrays predictions, coords (EPSG:4326), y_true (rescaled, aligned)
  tract_rows.csv   one row per (cell, tract)
  cells.csv        one row per completed cell; resume skips cells listed here;
                   under_mixed_tracts and n_unswapped_tracts per cell (SWAP-GUARD)
  swap_provenance.csv  one row per (randomized cell, tract) swap record
  run_meta.json    cell, seeds, packages, commit, generator diagnostics

Usage:
    python experiments/m6_recovery_grid/run_fig4.py --batch 1
    python experiments/m6_recovery_grid/run_fig4.py --batch 2
    ... --allow-unswapped 47065011324   train that tract on its unswapped graph,
                                        recorded randomized=False
"""

import argparse
import csv
import importlib.metadata as md
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))

import run_grid
from granite.disaggregation.pipeline import GRANITEPipeline
from granite.models.gnn import set_random_seed

OUT_DIR = REPO_ROOT / 'data' / 'results' / 'm6_fig4_randomization'
PRED_DIR = OUT_DIR / 'predictions'
TRACT_CSV = OUT_DIR / 'tract_rows.csv'
CELL_CSV = OUT_DIR / 'cells.csv'
META_JSON = OUT_DIR / 'run_meta.json'
PROVENANCE_CSV = OUT_DIR / 'swap_provenance.csv'

CELL = {'signal_source': 'latent', 'autocorr': 'strong', 'snr': 'medium',
        'between_tract': 'default'}
DATA_SEED = run_grid.SEEDS[0]          # first committed data-generation seed of the cell (42)
FIELD_DIR = REPO_ROOT / 'data' / 'synthetic' / 'run_20260628_030416'
FEATURE_MODE = 'coordinates_only'
GRAPH_KNN_K = 10
TRAINING_SEEDS = [42, 17, 123, 2024, 7]    # 5b run_sweep.py TRAINING_SEEDS
GRAPH_DRAW_SEEDS = [42, 17, 123, 2024, 7]  # 5b run_sweep.py GRAPH_DRAW_SEEDS
STRUCTURED_GS = 42

TRACT_FIELDS = ['condition', 'arch', 'training_seed', 'graph_draw_seed', 'data_seed',
                'tract_fips', 'n_addresses', 'recovery_r', 'morans_i_output']
CELL_FIELDS = ['condition', 'arch', 'training_seed', 'graph_draw_seed', 'data_seed',
               'n_tracts', 'recovery_r_mean', 'morans_i_output_mean', 'runtime_s',
               'finished_at', 'under_mixed_tracts', 'n_unswapped_tracts']
PROVENANCE_FIELDS = ['condition', 'arch', 'training_seed', 'graph_draw_seed', 'data_seed',
                     'tract_fips', 'n_edges', 'edges_preserved', 'share_preserved',
                     'randomized', 'swaps_requested', 'swap_outcome', 'allowlisted']


def cells_for_batch(batch):
    structured = [('spatial_knn_uniform', a, ts, STRUCTURED_GS)
                  for a in ('sage', 'gcn_gat') for ts in TRAINING_SEEDS]
    rand = {a: [('randomized', a, ts, gs) for ts in TRAINING_SEEDS for gs in GRAPH_DRAW_SEEDS]
            for a in ('sage', 'gcn_gat')}
    if batch == 1:
        return structured + rand['sage']
    if batch == 2:
        return rand['gcn_gat']
    raise ValueError(batch)


def _cell_key(condition, arch, ts, gs):
    return (condition, arch, int(ts), int(gs), int(DATA_SEED))


def _pred_path(condition, arch, ts, gs, fips):
    return PRED_DIR / f'{condition}__{arch}__ts{ts}__gs{gs}__ds{DATA_SEED}__{fips}.npz'


def _completed_cells():
    if not CELL_CSV.exists():
        return set()
    with open(CELL_CSV, newline='') as f:
        return {_cell_key(r['condition'], r['arch'], r['training_seed'], r['graph_draw_seed'])
                for r in csv.DictReader(f) if int(r['data_seed']) == DATA_SEED}


def _drop_incomplete_tract_rows(done):
    """remove tract rows of cells that never reached cells.csv (interrupted run)."""
    if not TRACT_CSV.exists():
        return
    df = pd.read_csv(TRACT_CSV, dtype={'tract_fips': str})
    keep = [(_cell_key(r.condition, r.arch, r.training_seed, r.graph_draw_seed) in done)
            for r in df.itertuples()]
    if not all(keep):
        print(f'[fig4] dropping {len(keep) - sum(keep)} tract rows from an interrupted cell')
        df[keep].to_csv(TRACT_CSV, index=False, float_format='%.17g')


def _check_header(path, fields):
    """refuse to append to an existing csv whose header lacks fields (pre-SWAP-GUARD)."""
    if not path.exists():
        return
    with open(path, newline='') as f:
        header = next(csv.reader(f), [])
    absent = [c for c in fields if c not in header]
    if absent:
        raise RuntimeError(f'{path} predates SWAP-GUARD (header lacks {absent}); '
                           f'new cells must not be appended to an existing results directory')


def _append(path, fields, rows):
    is_new = not path.exists()
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if is_new:
            w.writeheader()
        for r in rows:
            w.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in r.items()})


def _write_meta(gen_result, y_min, y_max, wtvr, batch):
    meta = json.loads(META_JSON.read_text()) if META_JSON.exists() else {'batches': {}}
    meta.update({
        'milestone': 'FIG-P1 B5',
        'cell': CELL,
        'data_seed': DATA_SEED,
        'feature_mode': FEATURE_MODE,
        'graph_knn_k': GRAPH_KNN_K,
        'training_seeds': TRAINING_SEEDS,
        'graph_draw_seeds': GRAPH_DRAW_SEEDS,
        'structured_graph_draw_seed_label': STRUCTURED_GS,
        'field_dir': str(FIELD_DIR.relative_to(REPO_ROOT)),
        'field_generator_commit': gen_result.get('generator_commit_stored'),
        'base_config': str(run_grid.BASE_CONFIG_PATH.relative_to(REPO_ROOT)),
        'tract_selection': str(run_grid.TRACT_SELECTION.relative_to(REPO_ROOT)),
        'generator': {
            'wtvr_achieved': wtvr,
            'morans_i_achieved': float(gen_result['diagnostics']['morans_i_achieved']),
            'ytrue_rescale_min': y_min,
            'ytrue_rescale_max': y_max,
        },
        'packages': {p: md.version(p) for p in (
            'scikit-learn', 'numpy', 'pandas', 'geopandas', 'shapely', 'scipy', 'torch',
            'torch-geometric', 'libpysal', 'esda', 'networkx')},
    })
    meta['batches'][str(batch)] = {'commit': run_grid._git_sha(),
                                   'started_at': datetime.now().isoformat(timespec='seconds')}
    META_JSON.write_text(json.dumps(meta, indent=2) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', type=int, choices=[1, 2], required=True)
    ap.add_argument('--max-cells', type=int, default=None, help='stop after N new cells')
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--allow-unswapped', default='',
                    help='comma-separated tract FIPS allowed to train on an unswapped '
                         'randomized graph (recorded randomized=False, warned)')
    args = ap.parse_args()
    allow = [f.strip() for f in args.allow_unswapped.split(',') if f.strip()]

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    done = _completed_cells()
    _drop_incomplete_tract_rows(done)
    todo = [c for c in cells_for_batch(args.batch) if _cell_key(*c) not in done]
    print(f'[fig4] batch {args.batch}: {len(cells_for_batch(args.batch))} cells, '
          f'{len(todo)} to run, data seed {DATA_SEED}')
    if not todo:
        return
    _check_header(CELL_CSV, CELL_FIELDS)

    tract_list = run_grid._load_tract_list()
    cfg_base = run_grid._load_base_config()
    cfg_init = run_grid._make_run_cfg(cfg_base, tract_list[0], FEATURE_MODE, 'sage')
    cfg_init['processing']['verbose'] = args.verbose
    pipeline = GRANITEPipeline(cfg_init, output_dir=str(run_grid.PIPELINE_SCRATCH))
    pipeline.verbose = args.verbose
    data = pipeline._load_spatial_data()
    pipeline.data_loader.config['graph_knn_k'] = GRAPH_KNN_K
    pipeline.data_loader.config['allow_unswapped'] = allow
    if allow:
        print(f'[fig4] allow-unswapped: {allow}')

    # graph-only swap check at every pending graph draw; raises before any training
    rand_gs = sorted({gs for condition, _, _, gs in todo if condition == 'randomized'})
    if rand_gs:
        recs = pipeline.data_loader.check_randomization(tract_list, rand_gs)
        print(f'[fig4] swap check: {len(tract_list)} tracts x {len(rand_gs)} graph draws, '
              f'{sum(1 for r in recs if not r["randomized"])} unswapped, allowlisted: '
              f'{sorted({r["tract_fips"] for r in recs if not r["randomized"]})}')

    draw_spec = dict(CELL, seed=DATA_SEED)
    gen_result, addr_df, tract_means, y_min, y_max, wtvr = run_grid._generate_draw(
        draw_spec, field_dir=FIELD_DIR)
    gen_result['generator_commit_stored'] = json.loads(
        (FIELD_DIR / 'metadata.json').read_text())['generator_commit']
    _write_meta(gen_result, y_min, y_max, wtvr, args.batch)

    n_new = 0
    for condition, arch, ts, gs in todo:
        if args.max_cells is not None and n_new >= args.max_cells:
            break
        t0 = time.time()
        graph_opts = {'graph_variant': condition, 'graph_draw_seed': gs, 'training_seed': ts}
        set_random_seed(ts)
        pipeline.data_loader._under_mixed_tracts = 0
        pipeline.data_loader._swap_records = []
        rows = []
        for fips in tract_list:
            run_cfg = run_grid._make_run_cfg(cfg_base, fips, FEATURE_MODE, arch, graph_opts)
            run_grid._configure_pipeline(pipeline, run_cfg)
            result = pipeline._process_single_tract(
                fips, data, svi_override=float(tract_means[fips]))
            if not result.get('success'):
                raise RuntimeError(f'{condition}/{arch}/ts{ts}/gs{gs}/{fips}: '
                                   f'{result.get("error", "?")}')
            preds = result['predictions']['mean'].values.astype(float)
            gdf = result['address_gdf']
            coords = np.array([[g.x, g.y] for g in gdf.geometry])
            y_true = run_grid._align_ytrue(addr_df, fips, gdf)
            rr = run_grid._recovery_r(preds, y_true)
            mi = run_grid._compute_morans_i_output(preds, gdf)
            path = _pred_path(condition, arch, ts, gs, fips)
            tmp = path.with_name(path.stem + '.tmp.npz')
            np.savez(tmp, predictions=preds, coords=coords, y_true=y_true)
            tmp.replace(path)
            rows.append({'condition': condition, 'arch': arch, 'training_seed': ts,
                         'graph_draw_seed': gs, 'data_seed': DATA_SEED, 'tract_fips': fips,
                         'n_addresses': len(preds), 'recovery_r': rr, 'morans_i_output': mi})
            print(f'[fig4]   {condition}/{arch}/ts{ts}/gs{gs}/{fips}: rr={rr:.4f} mi={mi:.4f}')
        runtime = time.time() - t0
        swap_records = list(pipeline.data_loader._swap_records)
        for r in swap_records:
            if not r['randomized']:
                print(f'[fig4]   UNSWAPPED {condition}/{arch}/ts{ts}/gs{gs}/{r["tract_fips"]}: '
                      f'randomized=False (allowlisted={r["allowlisted"]}, '
                      f'outcome={r["swap_outcome"]})')
        if condition == 'randomized':
            _append(PROVENANCE_CSV, PROVENANCE_FIELDS, [
                {**{k: r[k] for k in PROVENANCE_FIELDS if k in r},
                 'condition': condition, 'arch': arch, 'training_seed': ts,
                 'graph_draw_seed': gs, 'data_seed': DATA_SEED}
                for r in swap_records])
        _append(TRACT_CSV, TRACT_FIELDS, rows)
        _append(CELL_CSV, CELL_FIELDS, [{
            'condition': condition, 'arch': arch, 'training_seed': ts, 'graph_draw_seed': gs,
            'data_seed': DATA_SEED, 'n_tracts': len(rows),
            'recovery_r_mean': float(np.mean([r['recovery_r'] for r in rows])),
            'morans_i_output_mean': float(np.mean([r['morans_i_output'] for r in rows])),
            'runtime_s': runtime, 'finished_at': datetime.now().isoformat(timespec='seconds'),
            'under_mixed_tracts': int(getattr(pipeline.data_loader, '_under_mixed_tracts', 0)),
            'n_unswapped_tracts': sum(1 for r in swap_records if not r['randomized']),
        }])
        n_new += 1
        print(f'[fig4] cell {condition}/{arch}/ts{ts}/gs{gs} done in {runtime:.0f}s')


if __name__ == '__main__':
    main()
