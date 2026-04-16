"""
Multi-Tract Generalization Experiment

Tests whether single-tract findings generalize across tracts spanning the full
Hamilton County SVI distribution. Three supervision conditions x two architectures
per tract.
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats
from datetime import datetime
from pathlib import Path

sys.path.insert(0, '/workspaces/GRANITE')

from granite.models.gnn import (
    AccessibilitySVIGNN, GraphSAGEAccessibilitySVIGNN,
    MultiTractGNNTrainer, set_random_seed, normalize_accessibility_features
)
from granite.data.loaders import DataLoader
from granite.data.block_group_loader import BlockGroupLoader, rescale_block_group_svis

SEED = 42
HIDDEN_DIM = 64
EPOCHS = 200
LEARNING_RATE = 0.005
DROPOUT = 0.3
ORDERING_WEIGHT = 0.5
ORDERING_MIN_GAP = 0.5
ORDERING_MARGIN = 0.02
CONTINUITY_FIPS = '47065000600'

RESULTS_DIR = '/workspaces/GRANITE/results/multi_tract_experiment'


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_config():
    with open('/workspaces/GRANITE/config.yaml') as f:
        return yaml.safe_load(f)


def select_tracts(n_tracts=10):
    """select tracts at percentile intervals across county SVI distribution."""
    svi_path = '/workspaces/GRANITE/data/raw/SVI_2020_US.csv'
    df = pd.read_csv(svi_path, dtype={'FIPS': str, 'STCNTY': str})
    # hamilton county: STCNTY = 47065
    hamilton = df[df['STCNTY'] == '47065'].copy()
    # remove missing SVI
    hamilton = hamilton[hamilton['RPL_THEMES'] >= 0].copy()
    hamilton = hamilton.sort_values('RPL_THEMES').reset_index(drop=True)

    n = len(hamilton)
    percentiles = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
    selected_idx = []
    for p in percentiles:
        idx = int(round((p / 100.0) * (n - 1)))
        idx = max(0, min(n - 1, idx))
        selected_idx.append(idx)

    selected = hamilton.iloc[selected_idx][['FIPS', 'RPL_THEMES']].drop_duplicates(subset='FIPS').copy()
    selected = selected.reset_index(drop=True)

    # add continuity tract if not present
    if CONTINUITY_FIPS not in selected['FIPS'].values:
        row = hamilton[hamilton['FIPS'] == CONTINUITY_FIPS]
        if len(row) > 0:
            selected = pd.concat([selected, row[['FIPS', 'RPL_THEMES']]], ignore_index=True)
            log(f"Added continuity tract {CONTINUITY_FIPS}")

    selected = selected.sort_values('RPL_THEMES').reset_index(drop=True)
    log(f"Selected {len(selected)} tracts")
    for _, row in selected.iterrows():
        log(f"  {row['FIPS']}  SVI={row['RPL_THEMES']:.4f}")
    return selected


def load_shared_data(config):
    """load county-level destinations, roads, all addresses -- shared across tracts."""
    log("Loading shared county data...")
    data_loader = DataLoader('./data', config=config)
    state_fips = config['data']['state_fips']
    county_fips = config['data']['county_fips']

    census_tracts = data_loader.load_census_tracts(state_fips, county_fips)
    county_name = data_loader._get_county_name(state_fips, county_fips)
    svi = data_loader.load_svi_data(state_fips, county_name)
    tracts = census_tracts.merge(svi, on='FIPS', how='inner')

    employment = data_loader.create_employment_destinations(use_real_data=True)
    healthcare = data_loader.create_healthcare_destinations(use_real_data=True)
    grocery = data_loader.create_grocery_destinations(use_real_data=True)
    roads = data_loader.load_road_network(state_fips=state_fips, county_fips=county_fips)
    all_addresses = data_loader.load_address_points(state_fips, county_fips)

    return {
        'data_loader': data_loader,
        'tracts': tracts,
        'svi': svi,
        'employment': employment,
        'healthcare': healthcare,
        'grocery': grocery,
        'roads': roads,
        'all_addresses': all_addresses,
        'state_fips': state_fips,
        'county_fips': county_fips,
    }


def load_tract_data(fips, shared, config):
    """load addresses, features, and graph for one tract."""
    from granite.disaggregation.pipeline import GRANITEPipeline

    data_loader = shared['data_loader']
    tracts = shared['tracts']
    svi = shared['svi']

    tract_row = tracts[tracts['FIPS'] == fips]
    if len(tract_row) == 0:
        raise RuntimeError(f"Tract {fips} not in merged tracts")
    tract_svi = float(tract_row.iloc[0]['RPL_THEMES'])

    addresses = data_loader.get_addresses_for_tract(fips)
    if len(addresses) == 0:
        raise RuntimeError(f"No addresses for tract {fips}")
    addresses['tract_fips'] = fips

    pipeline = GRANITEPipeline(config, data_dir='./data',
                                output_dir='/tmp/granite_mt', verbose=False)
    pipeline.data_loader = data_loader

    data = {
        'tracts': tracts,
        'svi': svi,
        'employment_destinations': shared['employment'],
        'healthcare_destinations': shared['healthcare'],
        'grocery_destinations': shared['grocery'],
        'roads': shared['roads'],
        'addresses': shared['all_addresses'],
    }

    accessibility_features = pipeline._compute_accessibility_features(addresses, data)
    if accessibility_features is None:
        raise RuntimeError(f"Failed to compute features for {fips}")

    feature_names = pipeline._generate_feature_names(accessibility_features.shape[1])
    normalized_features, _ = normalize_accessibility_features(accessibility_features)

    context_features = data_loader.create_context_features_for_addresses(
        addresses=addresses, svi_data=svi
    )
    normalized_context, _ = data_loader.normalize_context_features(context_features)

    graph_data = data_loader.create_spatial_accessibility_graph(
        addresses=addresses,
        accessibility_features=normalized_features,
        context_features=normalized_context,
        state_fips=shared['state_fips'],
        county_fips=shared['county_fips'],
    )

    return {
        'tract_svi': tract_svi,
        'addresses': addresses,
        'graph_data': graph_data,
        'feature_names': feature_names,
    }


def load_bg_data_for_tract(fips, addresses, tract_svi):
    """load nationally-ranked rescaled BG targets for a single tract."""
    bg_loader = BlockGroupLoader(data_dir='./data', verbose=False)
    bg_data = bg_loader.get_block_groups_with_demographics(svi_ranking_scope='national')
    addresses_with_bg = bg_loader.assign_addresses_to_block_groups(addresses, bg_data)

    block_group_targets = {}
    block_group_masks = {}
    bg_address_counts = {}
    bg_svs_raw = {}

    for _, bg_row in bg_data.iterrows():
        bg_id = bg_row['GEOID']
        if not bg_row.get('svi_complete', False):
            continue
        svi_val = bg_row.get('SVI', None)
        if svi_val is None or pd.isna(svi_val):
            continue
        bg_tract = bg_row.get('tract_fips', bg_id[:11])
        if bg_tract != fips:
            continue
        bg_mask = (addresses_with_bg['block_group_id'] == bg_id).values
        n_bg = int(bg_mask.sum())
        if n_bg < 5:
            continue
        block_group_targets[bg_id] = float(svi_val)
        block_group_masks[bg_id] = bg_mask
        bg_address_counts[bg_id] = n_bg
        bg_svs_raw[bg_id] = float(svi_val)

    rescale_shift = None
    if block_group_targets:
        total = sum(bg_address_counts.values())
        pre_wm = sum(block_group_targets[g] * bg_address_counts[g]
                     for g in block_group_targets) / total
        rescale_shift = tract_svi - pre_wm
        block_group_targets = rescale_block_group_svis(
            block_group_targets, bg_address_counts, tract_svi
        )

    # rescaled values
    bg_svs_rescaled = dict(block_group_targets)

    return {
        'targets': block_group_targets,
        'masks': block_group_masks,
        'address_counts': bg_address_counts,
        'raw_svs': bg_svs_raw,
        'rescaled_svs': bg_svs_rescaled,
        'rescale_shift': rescale_shift,
    }


def build_tract_inventory_row(fips, tract_svi, addresses, bg_info):
    """compute inventory metrics for one tract."""
    n_addr = len(addresses)
    n_BG = len(bg_info['targets'])

    if n_BG > 0:
        rescaled = list(bg_info['rescaled_svs'].values())
        bg_spread = max(rescaled) - min(rescaled)
        # address-count-weighted variance
        counts = bg_info['address_counts']
        total = sum(counts.values())
        wm = sum(bg_info['rescaled_svs'][g] * counts[g] for g in counts) / total
        bg_wt_var = sum(counts[g] * (bg_info['rescaled_svs'][g] - wm) ** 2
                        for g in counts) / total
    else:
        bg_spread = float('nan')
        bg_wt_var = float('nan')

    rescale_shift = bg_info['rescale_shift'] if bg_info['rescale_shift'] is not None else float('nan')

    # count addresses with valid log_appvalue
    if 'log_appvalue' in addresses.columns:
        n_appvalue = int((~addresses['log_appvalue'].isna()).sum())
    else:
        n_appvalue = 0

    return {
        'fips': fips,
        'svi': tract_svi,
        'n_addr': n_addr,
        'n_BG': n_BG,
        'bg_spread': bg_spread,
        'bg_wt_var': bg_wt_var,
        'rescale_shift': rescale_shift,
        'n_appvalue': n_appvalue,
    }


def run_single(label, fips, model_class, graph_data, tract_svi, feature_names,
               block_group_targets=None, block_group_masks=None,
               ordering_values=None):
    """run one training configuration."""
    set_random_seed(SEED)

    has_context = hasattr(graph_data, 'context') and graph_data.context is not None
    context_dim = graph_data.context.shape[1] if has_context else 5

    model = model_class(
        accessibility_features_dim=graph_data.x.shape[1],
        context_features_dim=context_dim,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
        seed=SEED,
        use_context_gating=has_context,
        use_multitask=True,
    )

    trainer_config = {
        'learning_rate': LEARNING_RATE,
        'weight_decay': 1e-4,
        'enforce_constraints': True,
        'constraint_weight': 2.0,
        'use_multitask': True,
        'bg_constraint_weight': 1.0 if block_group_targets is not None else 0.0,
        'ordering_weight': ORDERING_WEIGHT if ordering_values is not None else 0.0,
        'ordering_min_gap': ORDERING_MIN_GAP,
        'ordering_margin': ORDERING_MARGIN,
    }

    trainer = MultiTractGNNTrainer(model, config=trainer_config, seed=SEED)
    n = graph_data.x.shape[0]

    try:
        result = trainer.train(
            graph_data=graph_data,
            tract_svis={fips: tract_svi},
            tract_masks={fips: np.ones(n, dtype=bool)},
            epochs=EPOCHS,
            verbose=False,
            feature_names=feature_names,
            block_group_targets=block_group_targets,
            block_group_masks=block_group_masks,
            ordering_values=ordering_values,
        )
        preds = result['final_predictions'].flatten()
        if not np.isfinite(result['final_loss']):
            raise RuntimeError(f"NaN/inf loss: {result['final_loss']}")
        return {
            'predictions': preds,
            'epochs_trained': int(result['epochs_trained']),
            'final_loss': float(result['final_loss']),
            'tract_error': float(result['overall_constraint_error']),
            'bg_error': float(result['bg_constraint_error']) if 'bg_constraint_error' in result else None,
            'spatial_std': float(result['final_spatial_std']),
            'success': True,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def run_tract_experiments(tract_idx, fips, tract_svi, tract_data, bg_info, inventory_row):
    """run all 6 conditions for one tract. returns per-condition results."""
    graph_data = tract_data['graph_data']
    addresses = tract_data['addresses']
    feature_names = tract_data['feature_names']

    bg_targets = bg_info['targets'] if bg_info['targets'] else None
    bg_masks = bg_info['masks'] if bg_info['masks'] else None

    # random ordering vector: seed = 42 + tract_index
    n = graph_data.x.shape[0]
    rng = np.random.RandomState(42 + tract_idx)
    random_ordering = rng.uniform(size=n).astype(np.float64)

    no_appvalue = inventory_row['n_appvalue'] == 0

    results = {}

    # condition 1: tract-only (GCN and SAGE)
    results['cond1_gcn'] = run_single('tract-only GCN', fips, AccessibilitySVIGNN,
                                       graph_data, tract_svi, feature_names)
    results['cond1_sage'] = run_single('tract-only SAGE', fips, GraphSAGEAccessibilitySVIGNN,
                                        graph_data, tract_svi, feature_names)

    # condition 2: tract+BG (GCN and SAGE)
    if bg_targets:
        results['cond2_gcn'] = run_single('tract+BG GCN', fips, AccessibilitySVIGNN,
                                           graph_data, tract_svi, feature_names,
                                           block_group_targets=bg_targets,
                                           block_group_masks=bg_masks)
        results['cond2_sage'] = run_single('tract+BG SAGE', fips, GraphSAGEAccessibilitySVIGNN,
                                            graph_data, tract_svi, feature_names,
                                            block_group_targets=bg_targets,
                                            block_group_masks=bg_masks)
    else:
        log(f"  Skipping cond2 (no qualifying BGs)")
        results['cond2_gcn'] = {'success': False, 'error': 'no qualifying BGs'}
        results['cond2_sage'] = {'success': False, 'error': 'no qualifying BGs'}

    # condition 3: tract+BG+random ordering
    if bg_targets:
        results['cond3_gcn'] = run_single('tract+BG+random GCN', fips, AccessibilitySVIGNN,
                                           graph_data, tract_svi, feature_names,
                                           block_group_targets=bg_targets,
                                           block_group_masks=bg_masks,
                                           ordering_values=random_ordering)
        results['cond3_sage'] = run_single('tract+BG+random SAGE', fips, GraphSAGEAccessibilitySVIGNN,
                                            graph_data, tract_svi, feature_names,
                                            block_group_targets=bg_targets,
                                            block_group_masks=bg_masks,
                                            ordering_values=random_ordering)
    else:
        log(f"  Skipping cond3 (no qualifying BGs)")
        results['cond3_gcn'] = {'success': False, 'error': 'no qualifying BGs'}
        results['cond3_sage'] = {'success': False, 'error': 'no qualifying BGs'}

    return results


def corr_pair(r1, r2):
    """compute pearson r and spearman rho between two prediction arrays."""
    if (r1 is None or r2 is None or
            not r1.get('success') or not r2.get('success')):
        return float('nan'), float('nan')
    p1 = r1['predictions']
    p2 = r2['predictions']
    if np.std(p1) < 1e-10 or np.std(p2) < 1e-10:
        return float('nan'), float('nan')
    pr, _ = stats.pearsonr(p1, p2)
    sr, _ = stats.spearmanr(p1, p2)
    return float(pr), float(sr)


def compute_convergence_row(fips, tract_svi, inv_row, tract_results):
    """compute convergence metrics for one tract."""
    r1 = tract_results.get('cond1_gcn', {})
    r2 = tract_results.get('cond1_sage', {})
    r3 = tract_results.get('cond2_gcn', {})
    r4 = tract_results.get('cond2_sage', {})
    r5 = tract_results.get('cond3_gcn', {})
    r6 = tract_results.get('cond3_sage', {})

    r_t, rho_t = corr_pair(r1, r2)
    r_bg, rho_bg = corr_pair(r3, r4)
    r_rnd, rho_rnd = corr_pair(r5, r6)

    return {
        'fips': fips,
        'svi': tract_svi,
        'bg_spread': inv_row['bg_spread'],
        'r_tract': r_t,
        'r_bg': r_bg,
        'r_bg_rnd': r_rnd,
        'rho_tract': rho_t,
        'rho_bg': rho_bg,
        'rho_bg_rnd': rho_rnd,
    }


def fmt_f(v, fmt='.3f'):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 'n/a'
    return f'{v:{fmt}}'


def fmt_e(r, key):
    if r is None or not r.get('success'):
        return 'n/a'
    v = r.get(key)
    if v is None or np.isnan(v):
        return 'n/a'
    return f'{v:.1f}'


def print_inventory(rows):
    print()
    print('   TRACT INVENTORY')
    print()
    hdr = (f"   {'FIPS':<14s}  {'SVI':>6s}  {'n_addr':>6s}  {'n_BG':>4s}  "
           f"{'BG_spread':>9s}  {'BG_wt_var':>9s}  {'rescale_shift':>13s}  {'n_appvalue':>10s}")
    print(hdr)
    print('   ' + '-' * (len(hdr) - 3))
    flags = []
    for row in rows:
        flag = ''
        if row['n_BG'] == 0:
            flag = ' [no BGs]'
            flags.append(f"  FLAGGED: {row['fips']} has 0 qualifying block groups (tract-only condition only)")
        if row['n_appvalue'] == 0:
            flag += ' [no appvalue]'
            flags.append(f"  FLAGGED: {row['fips']} has 0 addresses with valid log_appvalue (excluded from ordering)")
        print(f"   {row['fips']:<14s}  {row['svi']:>6.3f}  {row['n_addr']:>6d}  "
              f"{row['n_BG']:>4d}  {fmt_f(row['bg_spread']):>9s}  "
              f"{fmt_f(row['bg_wt_var'], '.4f'):>9s}  "
              f"{fmt_f(row['rescale_shift'], '.4f'):>13s}  "
              f"{row['n_appvalue']:>10d}{flag}")
    for f in flags:
        print(f)
    print()


def print_convergence_table(conv_rows):
    print()
    print('   MULTI-TRACT CONVERGENCE RESULTS')
    print()
    hdr = (f"   {'FIPS':<14s}  {'SVI':>5s}  {'BG_spr':>6s}  "
           f"{'r(tract)':>8s}  {'r(+BG)':>6s}  {'r(+BG+rnd)':>10s}  "
           f"{'rho(tract)':>10s}  {'rho(+BG)':>8s}  {'rho(+BG+rnd)':>12s}")
    print(hdr)
    print('   ' + '-' * (len(hdr) - 3))

    r_t_vals, r_bg_vals, r_rnd_vals = [], [], []
    rho_t_vals, rho_bg_vals, rho_rnd_vals = [], [], []

    for row in conv_rows:
        print(f"   {row['fips']:<14s}  {row['svi']:>5.3f}  {fmt_f(row['bg_spread']):>6s}  "
              f"{fmt_f(row['r_tract']):>8s}  {fmt_f(row['r_bg']):>6s}  "
              f"{fmt_f(row['r_bg_rnd']):>10s}  "
              f"{fmt_f(row['rho_tract']):>10s}  {fmt_f(row['rho_bg']):>8s}  "
              f"{fmt_f(row['rho_bg_rnd']):>12s}")
        for arr, val in [(r_t_vals, row['r_tract']), (r_bg_vals, row['r_bg']),
                          (r_rnd_vals, row['r_bg_rnd']), (rho_t_vals, row['rho_tract']),
                          (rho_bg_vals, row['rho_bg']), (rho_rnd_vals, row['rho_bg_rnd'])]:
            if not np.isnan(val):
                arr.append(val)

    def stats_line(label, vals):
        if vals:
            return (f"   {label:<14s}  {'':>5s}  {'':>6s}  "
                    f"{np.mean(vals):>8.3f}  {np.mean(r_bg_vals) if r_bg_vals else float('nan'):>6.3f}  "
                    f"{np.mean(r_rnd_vals) if r_rnd_vals else float('nan'):>10.3f}  "
                    f"{np.mean(rho_t_vals) if rho_t_vals else float('nan'):>10.3f}  "
                    f"{np.mean(rho_bg_vals) if rho_bg_vals else float('nan'):>8.3f}  "
                    f"{np.mean(rho_rnd_vals) if rho_rnd_vals else float('nan'):>12.3f}")
        return ''

    print()
    if r_t_vals:
        print(f"   {'Mean:':<14s}  {'':>5s}  {'':>6s}  "
              f"{np.mean(r_t_vals):>8.3f}  {np.mean(r_bg_vals) if r_bg_vals else float('nan'):>6.3f}  "
              f"{np.mean(r_rnd_vals) if r_rnd_vals else float('nan'):>10.3f}  "
              f"{np.mean(rho_t_vals) if rho_t_vals else float('nan'):>10.3f}  "
              f"{np.mean(rho_bg_vals) if rho_bg_vals else float('nan'):>8.3f}  "
              f"{np.mean(rho_rnd_vals) if rho_rnd_vals else float('nan'):>12.3f}")
        print(f"   {'Std:':<14s}  {'':>5s}  {'':>6s}  "
              f"{np.std(r_t_vals):>8.3f}  {np.std(r_bg_vals) if r_bg_vals else float('nan'):>6.3f}  "
              f"{np.std(r_rnd_vals) if r_rnd_vals else float('nan'):>10.3f}  "
              f"{np.std(rho_t_vals) if rho_t_vals else float('nan'):>10.3f}  "
              f"{np.std(rho_bg_vals) if rho_bg_vals else float('nan'):>8.3f}  "
              f"{np.std(rho_rnd_vals) if rho_rnd_vals else float('nan'):>12.3f}")
        r_t_m = np.median(r_t_vals) if r_t_vals else float('nan')
        r_bg_m = np.median(r_bg_vals) if r_bg_vals else float('nan')
        r_rnd_m = np.median(r_rnd_vals) if r_rnd_vals else float('nan')
        rho_t_m = np.median(rho_t_vals) if rho_t_vals else float('nan')
        rho_bg_m = np.median(rho_bg_vals) if rho_bg_vals else float('nan')
        rho_rnd_m = np.median(rho_rnd_vals) if rho_rnd_vals else float('nan')
        print(f"   {'Median:':<14s}  {'':>5s}  {'':>6s}  "
              f"{r_t_m:>8.3f}  {r_bg_m:>6.3f}  {r_rnd_m:>10.3f}  "
              f"{rho_t_m:>10.3f}  {rho_bg_m:>8.3f}  {rho_rnd_m:>12.3f}")
    print()


def print_constraint_table(conv_rows, all_results):
    print()
    print('   CONSTRAINT SATISFACTION')
    print()
    hdr = (f"   {'FIPS':<14s}  {'SVI':>5s}  "
           f"{'Tract_err(t)':>16s}  {'Tract_err(BG)':>16s}  {'Tract_err(rnd)':>16s}  "
           f"{'BG_err(BG)':>14s}  {'BG_err(rnd)':>14s}")
    print(hdr)
    sub = f"   {'':14s}  {'':5s}  {'GCN / SAGE':>16s}  {'GCN / SAGE':>16s}  {'GCN / SAGE':>16s}  {'GCN / SAGE':>14s}  {'GCN / SAGE':>14s}"
    print(sub)
    print('   ' + '-' * (len(hdr) - 3))

    for row in conv_rows:
        fips = row['fips']
        r = all_results.get(fips, {})
        r1g = r.get('cond1_gcn', {})
        r1s = r.get('cond1_sage', {})
        r2g = r.get('cond2_gcn', {})
        r2s = r.get('cond2_sage', {})
        r3g = r.get('cond3_gcn', {})
        r3s = r.get('cond3_sage', {})

        t1 = f"{fmt_e(r1g,'tract_error')} / {fmt_e(r1s,'tract_error')}"
        t2 = f"{fmt_e(r2g,'tract_error')} / {fmt_e(r2s,'tract_error')}"
        t3 = f"{fmt_e(r3g,'tract_error')} / {fmt_e(r3s,'tract_error')}"
        bg2 = f"{fmt_e(r2g,'bg_error')} / {fmt_e(r2s,'bg_error')}"
        bg3 = f"{fmt_e(r3g,'bg_error')} / {fmt_e(r3s,'bg_error')}"

        print(f"   {fips:<14s}  {row['svi']:>5.3f}  {t1:>16s}  {t2:>16s}  "
              f"{t3:>16s}  {bg2:>14s}  {bg3:>14s}")
    print()


def safe_corr(xs, ys):
    """pearson r between two lists, excluding nan pairs."""
    pairs = [(x, y) for x, y in zip(xs, ys)
             if not np.isnan(x) and not np.isnan(y)]
    if len(pairs) < 3:
        return float('nan')
    xa, ya = zip(*pairs)
    r, _ = stats.pearsonr(xa, ya)
    return float(r)


def print_density_analysis(conv_rows, inventory_rows):
    inv = {r['fips']: r for r in inventory_rows}
    n = len(conv_rows)

    bg_spreads = [inv[r['fips']]['bg_spread'] for r in conv_rows]
    bg_wtvars = [inv[r['fips']]['bg_wt_var'] for r in conv_rows]
    n_bgs = [inv[r['fips']]['n_BG'] for r in conv_rows]
    svils = [r['svi'] for r in conv_rows]
    n_addrs = [inv[r['fips']]['n_addr'] for r in conv_rows]

    bg_impr = [r['r_bg'] - r['r_tract'] for r in conv_rows]
    rnd_impr = [r['r_bg_rnd'] - r['r_bg'] for r in conv_rows]

    corr_spread_bg = safe_corr(bg_spreads, bg_impr)
    corr_spread_rnd = safe_corr(bg_spreads, rnd_impr)
    corr_wtvar_bg = safe_corr(bg_wtvars, bg_impr)
    corr_nbg_bg = safe_corr(n_bgs, bg_impr)
    corr_svi_base = safe_corr(svils, [r['r_tract'] for r in conv_rows])
    corr_naddr_base = safe_corr(n_addrs, [r['r_tract'] for r in conv_rows])

    print()
    print(f'   SUPERVISION DENSITY ANALYSIS (correlations across {n} tracts)')
    print()
    print(f"   {'Predictor':<32s}  {'vs BG improvement':>18s}  {'vs ordering improvement':>23s}")
    print(f"   {'-'*32:<32s}  {'-'*18:>18s}  {'-'*23:>23s}")
    print(f"   {'BG SVI spread':<32s}  {'r='+fmt_f(corr_spread_bg):>18s}  {'r='+fmt_f(corr_spread_rnd):>23s}")
    print(f"   {'BG SVI wt variance':<32s}  {'r='+fmt_f(corr_wtvar_bg):>18s}  {'':>23s}")
    print(f"   {'n qualifying BGs':<32s}  {'r='+fmt_f(corr_nbg_bg):>18s}  {'':>23s}")
    print(f"   {'Tract SVI level':<32s}  {'vs baseline r: r='+fmt_f(corr_svi_base):>18s}")
    print(f"   {'n_addresses':<32s}  {'vs baseline r: r='+fmt_f(corr_naddr_base):>18s}")
    print()

    return {
        'n_tracts': n,
        'bg_spread_vs_bg_improvement': corr_spread_bg,
        'bg_spread_vs_ordering_improvement': corr_spread_rnd,
        'bg_wtvar_vs_bg_improvement': corr_wtvar_bg,
        'n_bgs_vs_bg_improvement': corr_nbg_bg,
        'tract_svi_vs_baseline': corr_svi_base,
        'n_addr_vs_baseline': corr_naddr_base,
    }


def print_cases(conv_rows, inventory_rows):
    inv = {r['fips']: r for r in inventory_rows}

    strong = [r for r in conv_rows if not np.isnan(r['r_bg_rnd']) and r['r_bg_rnd'] > 0.7]
    under = [r for r in conv_rows if not np.isnan(r['r_bg_rnd']) and r['r_bg_rnd'] < 0.2]

    print()
    print('   CASE IDENTIFICATION')
    print()
    print('   Strong convergence (r > 0.7):')
    if strong:
        for r in strong:
            iv = inv[r['fips']]
            print(f"     {r['fips']}: SVI={r['svi']:.3f}, BG_spread={fmt_f(iv['bg_spread'])}, "
                  f"n_BG={iv['n_BG']}, n_addr={iv['n_addr']}")
    else:
        print('     No strong cases found')

    print()
    print('   Underdetermined (r < 0.2):')
    if under:
        for r in under:
            iv = inv[r['fips']]
            print(f"     {r['fips']}: SVI={r['svi']:.3f}, BG_spread={fmt_f(iv['bg_spread'])}, "
                  f"n_BG={iv['n_BG']}, n_addr={iv['n_addr']}")
    else:
        print('     No underdetermined cases found')
    print()


def print_random_ordering_cost(conv_rows, all_results):
    gcn_bg_errs = []
    sage_bg_errs = []
    gcn_rnd_errs = []
    sage_rnd_errs = []
    n_worse_gcn = 0
    n_worse_sage = 0
    n_with_bg = 0

    for row in conv_rows:
        fips = row['fips']
        r = all_results.get(fips, {})
        r2g = r.get('cond2_gcn', {})
        r2s = r.get('cond2_sage', {})
        r3g = r.get('cond3_gcn', {})
        r3s = r.get('cond3_sage', {})

        if (r2g.get('success') and r3g.get('success') and
                r2g.get('bg_error') is not None and r3g.get('bg_error') is not None):
            gcn_bg_errs.append(r2g['bg_error'])
            gcn_rnd_errs.append(r3g['bg_error'])
            if r3g['bg_error'] > r2g['bg_error']:
                n_worse_gcn += 1

        if (r2s.get('success') and r3s.get('success') and
                r2s.get('bg_error') is not None and r3s.get('bg_error') is not None):
            sage_bg_errs.append(r2s['bg_error'])
            sage_rnd_errs.append(r3s['bg_error'])
            if r3s['bg_error'] > r2s['bg_error']:
                n_worse_sage += 1

        if r2g.get('success') and r2g.get('bg_error') is not None:
            n_with_bg += 1

    mean_gcn_bg = np.mean(gcn_bg_errs) if gcn_bg_errs else float('nan')
    mean_sage_bg = np.mean(sage_bg_errs) if sage_bg_errs else float('nan')
    mean_gcn_rnd = np.mean(gcn_rnd_errs) if gcn_rnd_errs else float('nan')
    mean_sage_rnd = np.mean(sage_rnd_errs) if sage_rnd_errs else float('nan')
    delta_gcn = mean_gcn_rnd - mean_gcn_bg
    delta_sage = mean_sage_rnd - mean_sage_bg

    print()
    print('   RANDOM ORDERING COST ANALYSIS')
    print()
    print(f"   Mean BG error (+BG only):         {fmt_f(mean_gcn_bg, '.2f')}% (GCN) / {fmt_f(mean_sage_bg, '.2f')}% (SAGE)")
    print(f"   Mean BG error (+BG+random):       {fmt_f(mean_gcn_rnd, '.2f')}% (GCN) / {fmt_f(mean_sage_rnd, '.2f')}% (SAGE)")
    print(f"   Mean BG error increase:           {fmt_f(delta_gcn, '.2f')} pp (GCN) / {fmt_f(delta_sage, '.2f')} pp (SAGE)")
    print(f"   Tracts where random worsened BG error: {n_worse_gcn}/{n_with_bg} (GCN) / {n_worse_sage}/{n_with_bg} (SAGE)")
    print()

    return {
        'mean_gcn_bg_error': float(mean_gcn_bg),
        'mean_sage_bg_error': float(mean_sage_bg),
        'mean_gcn_rnd_error': float(mean_gcn_rnd),
        'mean_sage_rnd_error': float(mean_sage_rnd),
        'delta_gcn': float(delta_gcn),
        'delta_sage': float(delta_sage),
        'n_worse_gcn': n_worse_gcn,
        'n_worse_sage': n_worse_sage,
        'n_with_bg': n_with_bg,
    }


def print_summary(conv_rows, density_corrs, ordering_cost):
    r_bg_vals = [r['r_bg'] for r in conv_rows if not np.isnan(r['r_bg'])]
    r_t_vals = [r['r_tract'] for r in conv_rows if not np.isnan(r['r_tract'])]
    r_rnd_vals = [r['r_bg_rnd'] for r in conv_rows if not np.isnan(r['r_bg_rnd'])]
    bg_impr = [r['r_bg'] - r['r_tract'] for r in conv_rows
               if not np.isnan(r['r_bg']) and not np.isnan(r['r_tract'])]

    mean_bg_impr = np.mean(bg_impr) if bg_impr else float('nan')
    mean_base = np.mean(r_t_vals) if r_t_vals else float('nan')
    mean_bg = np.mean(r_bg_vals) if r_bg_vals else float('nan')
    mean_rnd = np.mean(r_rnd_vals) if r_rnd_vals else float('nan')

    spread_corr = density_corrs.get('bg_spread_vs_bg_improvement', float('nan'))
    rnd_spread_corr = density_corrs.get('bg_spread_vs_ordering_improvement', float('nan'))

    print()
    print('   SUMMARY INTERPRETATION')
    print()
    print(f"   BG constraints improve cross-architecture convergence on average by "
          f"{fmt_f(mean_bg_impr)} r units (baseline r={fmt_f(mean_base)}, "
          f"with BG r={fmt_f(mean_bg)}). Variance across tracts is "
          f"{'high' if len(r_bg_vals) > 1 and np.std(r_bg_vals) > 0.15 else 'moderate'}.")

    if not np.isnan(spread_corr):
        if abs(spread_corr) > 0.4:
            print(f"   The supervision density hypothesis is supported: BG SVI spread "
                  f"correlates with convergence improvement (r={spread_corr:.3f}). "
                  f"Tracts with wider BG spread benefit more from BG constraints.")
        else:
            print(f"   The supervision density hypothesis is not clearly supported: BG SVI spread "
                  f"correlates weakly with convergence improvement (r={spread_corr:.3f}).")

    if not np.isnan(rnd_spread_corr):
        print(f"   Random ordering improvement correlates with BG spread at r={rnd_spread_corr:.3f}.")

    if not np.isnan(ordering_cost.get('delta_gcn', float('nan'))):
        delta = ordering_cost['delta_gcn']
        if delta > 0.5:
            print(f"   Random ordering trades BG constraint satisfaction (+{delta:.2f}pp BG error) "
                  f"for convergence, consistent with the single-tract finding.")
        else:
            print(f"   Random ordering has minimal impact on BG constraint satisfaction "
                  f"(+{delta:.2f}pp BG error on average).")

    max_r = max(r_rnd_vals) if r_rnd_vals else float('nan')
    print(f"   Best convergence achieved: r={fmt_f(max_r)} (tract+BG+random, best tract).")

    if not np.isnan(mean_rnd) and mean_rnd < 0.4:
        resolution = 'tract'
    elif not np.isnan(mean_rnd) and mean_rnd < 0.6:
        resolution = 'block group'
    else:
        resolution = 'sub-block-group'

    print()
    print(f"   The disaggregation resolution limit for Hamilton County under the current "
          f"feature set falls at [{resolution}] level based on cross-architecture convergence analysis.")
    print()


def main():
    start = time.time()
    log("Multi-Tract Generalization Experiment")
    log("=" * 70)

    timeout = 30 * 60  # 30 minutes

    config = load_config()

    # part 1: select tracts
    log("\n--- PART 1: Tract Selection ---")
    selected = select_tracts()
    fips_list = selected['FIPS'].tolist()
    svi_map = dict(zip(selected['FIPS'], selected['RPL_THEMES']))
    n_tracts = len(fips_list)

    # load shared county data once
    log("\n--- Loading shared county data ---")
    shared = load_shared_data(config)

    # parts 1+2 interleaved: load data and run experiments per tract
    log("\n--- PARTS 1+2: Per-tract data loading and experiments ---")
    inventory_rows = []
    all_results = {}
    n_success = 0
    n_total = 0
    failures = []

    for i, fips in enumerate(fips_list):
        if time.time() - start > timeout:
            log(f"TIMEOUT: reached 30-minute limit at tract {i+1}/{n_tracts}, stopping")
            break

        tract_start = time.time()
        log(f"\nTract {i+1}/{n_tracts}: {fips} (SVI={svi_map[fips]:.3f}) ...")

        # load tract data (may trigger OSRM routing if not cached)
        try:
            td = load_tract_data(fips, shared, config)
        except Exception as e:
            log(f"  DATA LOAD FAILED: {e}")
            import traceback
            traceback.print_exc()
            inventory_rows.append({
                'fips': fips, 'svi': svi_map[fips], 'n_addr': 0,
                'n_BG': 0, 'bg_spread': float('nan'), 'bg_wt_var': float('nan'),
                'rescale_shift': float('nan'), 'n_appvalue': 0,
            })
            continue

        # check timeout again after potentially long data load
        if time.time() - start > timeout:
            log(f"TIMEOUT after data load for tract {i+1}/{n_tracts}")
            inventory_rows.append(build_tract_inventory_row(
                fips, td['tract_svi'], td['addresses'],
                {'targets': {}, 'masks': {}, 'rescale_shift': None,
                 'rescaled_svs': {}, 'address_counts': {}}))
            break

        bg_info = load_bg_data_for_tract(fips, td['addresses'], td['tract_svi'])
        inv_row = build_tract_inventory_row(fips, td['tract_svi'], td['addresses'], bg_info)
        inventory_rows.append(inv_row)

        # run 6 training configurations
        tract_results = {}
        bg_targets = bg_info['targets'] if bg_info['targets'] else None
        bg_masks = bg_info['masks'] if bg_info['masks'] else None
        n = td['graph_data'].x.shape[0]
        rng = np.random.RandomState(42 + i)
        random_ordering = rng.uniform(size=n).astype(np.float64)

        for cond, gcn_key, sage_key, use_bg, use_rnd in [
            (1, 'cond1_gcn', 'cond1_sage', False, False),
            (2, 'cond2_gcn', 'cond2_sage', True, False),
            (3, 'cond3_gcn', 'cond3_sage', True, True),
        ]:
            cond_bg = bg_targets if use_bg else None
            cond_masks = bg_masks if use_bg else None
            cond_ordering = random_ordering if use_rnd else None

            if use_bg and not bg_targets:
                for key in [gcn_key, sage_key]:
                    tract_results[key] = {'success': False, 'error': 'no qualifying BGs'}
                n_total += 2
                continue

            for key, model_class, arch_name in [
                (gcn_key, AccessibilitySVIGNN, 'GCN'),
                (sage_key, GraphSAGEAccessibilitySVIGNN, 'SAGE'),
            ]:
                n_total += 1
                r = run_single(f"cond{cond}_{arch_name}", fips, model_class,
                               td['graph_data'], td['tract_svi'], td['feature_names'],
                               block_group_targets=cond_bg,
                               block_group_masks=cond_masks,
                               ordering_values=cond_ordering)
                tract_results[key] = r
                if r['success']:
                    n_success += 1
                    log(f"  cond{cond} {arch_name}: tract_err={r['tract_error']:.1f}%, "
                        f"bg_err={r.get('bg_error') or 'n/a'}, std={r['spatial_std']:.4f}")
                else:
                    failures.append({'fips': fips, 'condition': cond, 'arch': arch_name,
                                     'error': r.get('error', 'unknown')})
                    log(f"  cond{cond} {arch_name}: FAILED ({r.get('error', '')})")

        all_results[fips] = tract_results
        elapsed_t = time.time() - tract_start
        log(f"Tract {i+1}/{n_tracts}: {fips} (SVI={svi_map[fips]:.3f}) ... done ({elapsed_t:.0f}s)")

    print(f"\nCompleted {n_success}/{n_total} runs successfully, {len(failures)} failures")
    if failures:
        for f in failures:
            print(f"  FAIL: {f['fips']} cond{f['condition']} {f['arch']}: {f['error']}")

    # part 3: analysis
    log("\n--- PART 3: Analysis ---")
    inv_map = {r['fips']: r for r in inventory_rows}

    conv_rows = []
    for row in inventory_rows:
        fips = row['fips']
        if fips not in all_results:
            continue
        conv_row = compute_convergence_row(fips, row['svi'], row, all_results[fips])
        conv_rows.append(conv_row)

    # capture all output for saving
    import io
    from contextlib import redirect_stdout

    output_lines = []

    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, data):
            for f in self.files:
                f.write(data)
        def flush(self):
            for f in self.files:
                f.flush()

    buf = io.StringIO()
    tee = Tee(sys.stdout, buf)

    import builtins
    real_print = builtins.print

    def capturing_print(*args, **kwargs):
        kwargs.setdefault('file', tee)
        real_print(*args, **kwargs)

    builtins.print = capturing_print

    try:
        print_convergence_table(conv_rows)
        print_constraint_table(conv_rows, all_results)
        density_corrs = print_density_analysis(conv_rows, inventory_rows)
        print_cases(conv_rows, inventory_rows)
        ordering_cost = print_random_ordering_cost(conv_rows, all_results)
        print_summary(conv_rows, density_corrs, ordering_cost)
    finally:
        builtins.print = real_print

    summary_text = buf.getvalue()

    # part 5: save results
    log("\n--- PART 5: Saving ---")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # tract_inventory.csv
    inv_df = pd.DataFrame(inventory_rows)
    inv_df.to_csv(os.path.join(RESULTS_DIR, 'tract_inventory.csv'), index=False)
    log("Saved tract_inventory.csv")

    # convergence_results.csv
    conv_df = pd.DataFrame(conv_rows)
    conv_df.to_csv(os.path.join(RESULTS_DIR, 'convergence_results.csv'), index=False)
    log("Saved convergence_results.csv")

    # supervision_density.json
    with open(os.path.join(RESULTS_DIR, 'supervision_density.json'), 'w') as f:
        json.dump(density_corrs, f, indent=2)
    log("Saved supervision_density.json")

    # multi_tract_summary.txt
    # rebuild full output with inventory table
    inv_buf = io.StringIO()
    import sys as _sys
    _old_stdout = _sys.stdout
    _sys.stdout = inv_buf
    print_inventory(inventory_rows)
    _sys.stdout = _old_stdout
    full_summary = (
        f"Multi-Tract Experiment: {datetime.now().isoformat()}\n"
        f"Tracts: {n_tracts}, Seed: {SEED}, Epochs: {EPOCHS}\n\n"
        + inv_buf.getvalue()
        + summary_text
    )
    with open(os.path.join(RESULTS_DIR, 'multi_tract_summary.txt'), 'w') as f:
        f.write(full_summary)
    log("Saved multi_tract_summary.txt")

    # full_results.json
    def safe_serialize(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, float) and np.isnan(obj):
            return None
        return obj

    full_json = {
        'experiment': 'multi_tract_generalization',
        'timestamp': datetime.now().isoformat(),
        'elapsed_seconds': time.time() - start,
        'hyperparameters': {
            'seed': SEED, 'hidden_dim': HIDDEN_DIM, 'epochs': EPOCHS,
            'learning_rate': LEARNING_RATE, 'dropout': DROPOUT,
            'ordering_weight': ORDERING_WEIGHT,
            'ordering_min_gap': ORDERING_MIN_GAP,
            'ordering_margin': ORDERING_MARGIN,
        },
        'tracts': fips_list,
        'inventory': inventory_rows,
        'results': {},
        'convergence': conv_rows,
        'density_correlations': density_corrs,
        'ordering_cost': ordering_cost,
        'failures': failures,
        'n_success': n_success,
        'n_total': n_total,
    }

    for fips, tract_res in all_results.items():
        full_json['results'][fips] = {}
        for key, r in tract_res.items():
            entry = {k: safe_serialize(v) for k, v in r.items()
                     if k != 'predictions'}
            if r.get('success') and 'predictions' in r:
                entry['predictions'] = r['predictions'].tolist()
            full_json['results'][fips][key] = entry

    # fix nan floats in inventory and convergence
    def fix_nans(obj):
        if isinstance(obj, dict):
            return {k: fix_nans(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [fix_nans(v) for v in obj]
        if isinstance(obj, float) and np.isnan(obj):
            return None
        return obj

    full_json = fix_nans(full_json)

    with open(os.path.join(RESULTS_DIR, 'full_results.json'), 'w') as f:
        json.dump(full_json, f, indent=2)
    log("Saved full_results.json")

    elapsed = time.time() - start
    log(f"\nExperiment completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == '__main__':
    main()
