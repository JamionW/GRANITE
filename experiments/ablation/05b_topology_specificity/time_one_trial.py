"""
single-trial timing check for step05b.
runs spatial_knn_uniform / sage / training_seed=42 over all 20 tracts.
reports wall-clock and per-tract moran's i. no results written.
"""
import importlib.util
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

spec = importlib.util.spec_from_file_location(
    "run_sweep",
    str(Path(__file__).parent / "run_sweep.py"),
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

import numpy as np
import yaml
import geopandas as gpd

from granite.disaggregation.pipeline import GRANITEPipeline
from granite.data.block_group_loader import BlockGroupLoader
from granite.validation.block_group_validation import BlockGroupValidator
from granite.evaluation.spatial_diagnostics import SpatialLearningDiagnostics
from granite.models.gnn import set_random_seed
import math
import traceback

CONDITION = 'spatial_knn_uniform'
ARCH = 'sage'
TRAINING_SEED = 42
GRAPH_DRAW_SEED = 42

def main():
    t_start = time.time()

    tract_list = mod._load_tract_list()
    print(f'tracts: {len(tract_list)}')

    cfg_base = mod._load_base_config()
    with open(REPO_ROOT / 'config.yaml') as f:
        _live = yaml.safe_load(f) or {}
    cfg_base['graph_knn_k'] = _live.get('graph_knn_k', 10)
    cfg_base['data']['target'] = 'svi'
    cfg_base['data']['neighbor_tracts'] = 0
    cfg_base['data']['state_fips'] = '47'
    cfg_base['data']['county_fips'] = '065'
    cfg_base['processing']['skip_importance'] = True
    cfg_base['processing']['verbose'] = False
    cfg_base['processing']['enable_caching'] = True
    cfg_base['features']['feature_standardization'] = 'per_tract'
    cfg_base['training']['constraint_mode'] = 'soft'
    cfg_base['training']['apply_post_correction'] = True
    cfg_base['training']['variation_weight'] = 0.8

    print('loading BG geodataframe...')
    bg_gdf = mod._load_bg_gdf()
    validator = BlockGroupValidator(bg_gdf, verbose=False)

    scratch_dir = str(REPO_ROOT / 'output' / f'step05b_{CONDITION}_scratch')
    os.makedirs(scratch_dir, exist_ok=True)

    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in cfg_base.items()}
    cfg['data'] = dict(cfg_base.get('data', {}))
    cfg['data']['target_fips'] = tract_list[0]
    cfg['graph_variant'] = CONDITION
    pipeline = GRANITEPipeline(cfg, output_dir=scratch_dir)
    pipeline.verbose = False

    print('loading spatial data...')
    data = pipeline._load_spatial_data()

    set_random_seed(TRAINING_SEED)

    cfg2 = {k: (dict(v) if isinstance(v, dict) else v) for k, v in cfg_base.items()}
    cfg2['processing'] = dict(cfg_base.get('processing', {}))
    cfg2['processing']['random_seed'] = TRAINING_SEED
    cfg2['model'] = dict(cfg_base.get('model', {}))
    cfg2['model']['architecture'] = ARCH
    cfg2['graph_variant'] = CONDITION
    cfg2['graph_draw_seed'] = GRAPH_DRAW_SEED

    pipeline.config = cfg2
    pipeline.data_loader.config['graph_variant'] = CONDITION
    pipeline.data_loader.config['graph_draw_seed'] = GRAPH_DRAW_SEED
    pipeline.data_loader.config['processing'] = cfg2['processing']
    pipeline.data_loader._under_mixed_tracts = 0

    print(f'\nrunning {CONDITION}/{ARCH}/tseed={TRAINING_SEED} over {len(tract_list)} tracts...\n')

    per_tract_morans = []
    for fips in tract_list:
        cfg2['data'] = dict(cfg_base.get('data', {}))
        cfg2['data']['target_fips'] = fips
        pipeline.config['data']['target_fips'] = fips
        try:
            result = pipeline._process_single_tract(fips, data)
        except Exception as e:
            print(f'  ERROR {fips}: {str(e)[:200]}')
            traceback.print_exc()
            continue
        if not result.get('success'):
            print(f'  FAILED {fips}: {result.get("error","?")[:200]}')
            continue
        address_gdf = result['address_gdf']
        preds_arr = result['predictions']['mean'].values.astype(float)
        morans_i = mod._compute_morans_i(preds_arr, address_gdf)
        constr_err = abs(np.mean(preds_arr) - float(result['tract_svi']))
        print(f'  {fips}: moran={morans_i:.4f}  std={np.std(preds_arr):.4f}  constr={constr_err:.2e}')
        if math.isfinite(morans_i):
            per_tract_morans.append(morans_i)

    elapsed = time.time() - t_start
    mean_moran = float(np.mean(per_tract_morans)) if per_tract_morans else float('nan')
    print(f'\n--- timing trial complete ---')
    print(f'condition:      {CONDITION}')
    print(f'arch:           {ARCH}')
    print(f'training_seed:  {TRAINING_SEED}')
    print(f'tracts run:     {len(per_tract_morans)} / {len(tract_list)}')
    print(f'mean moran I:   {mean_moran:.4f}')
    print(f'wall-clock:     {elapsed:.1f}s  ({elapsed/60:.1f} min)')


if __name__ == '__main__':
    main()
