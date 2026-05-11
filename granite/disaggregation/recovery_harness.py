"""
Held-out feature recovery harness for GRANITE (roadmap M1).

Hides one address-level feature, uses its per-tract mean as the soft
constraint (replacing the per-tract SVI), trains the GNN, and records
address-level recovery metrics.

This module does not alter the default SVI pipeline path. The harness
reuses GRANITEPipeline's data loading and feature methods, then calls
MultiTractGNNTrainer directly with modified constraints.
"""
import json
import os
import subprocess
import traceback

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Optional, Tuple

from granite.data.external_targets import load_external_target
from granite.disaggregation.pipeline import GRANITEPipeline
from granite.models.gnn import (
    AccessibilitySVIGNN,
    GraphSAGEAccessibilitySVIGNN,
    MultiTractGNNTrainer,
    normalize_accessibility_features,
    set_random_seed,
)


def _get_git_sha() -> Optional[str]:
    """Return current HEAD SHA, or None if unavailable."""
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def _compute_per_tract_metrics(
    tract_addresses: pd.DataFrame,
    preds_std: np.ndarray,
    preds_native: np.ndarray,
    true_values: np.ndarray,
    tract_target_means_std: Dict[str, float],
    tgt_mean: float,
    tgt_std: float,
) -> pd.DataFrame:
    """Compute per-tract recovery metrics."""
    rows = []
    for fips in tract_addresses['tract_fips'].unique():
        mask = (tract_addresses['tract_fips'] == fips).values
        n = int(mask.sum())
        if n == 0:
            continue

        p_native = preds_native[mask]
        t_native = true_values[mask]

        finite_mask = np.isfinite(p_native) & np.isfinite(t_native)
        p_f = p_native[finite_mask]
        t_f = t_native[finite_mask]

        if len(p_f) < 2:
            pearson_r = float('nan')
            spearman_rho = float('nan')
        else:
            pearson_r = float(np.corrcoef(p_f, t_f)[0, 1])
            spearman_rho = float(stats.spearmanr(p_f, t_f).correlation)

        rmse = float(np.sqrt(np.mean((p_f - t_f) ** 2))) if len(p_f) > 0 else float('nan')

        # constraint error: predicted tract mean vs target tract mean (both native)
        predicted_tract_mean = float(np.mean(p_native[np.isfinite(p_native)])) if np.any(np.isfinite(p_native)) else float('nan')
        tract_target_native = tract_target_means_std.get(fips, float('nan')) * tgt_std + tgt_mean

        if abs(tract_target_native) > 1e-10:
            constraint_err_pct = abs(predicted_tract_mean - tract_target_native) / abs(tract_target_native) * 100.0
        else:
            constraint_err_pct = float('nan')

        rows.append({
            'tract_fips': fips,
            'n_addresses': n,
            'pearson_r': pearson_r,
            'spearman_rho': spearman_rho,
            'rmse': rmse,
            'constraint_error_pct': constraint_err_pct,
        })
    return pd.DataFrame(rows)


def _write_outputs(
    output_dir: str,
    tract_addresses: pd.DataFrame,
    target_feature: str,
    arch: str,
    seed: int,
    preds_std: np.ndarray,
    preds_native: np.ndarray,
    true_values: np.ndarray,
    tract_target_means_std: Dict[str, float],
    tgt_mean: float,
    tgt_std: float,
    config: dict,
    n_features_after_drop: int,
    standardize: bool,
    target_mode: str = 'held_out_feature',
    target_name: Optional[str] = None,
    target_source: Optional[str] = None,
    n_addresses_matched: Optional[int] = None,
    n_addresses_missing: Optional[int] = None,
) -> None:
    """Write predictions.csv, per_tract_metrics.csv, and run_meta.json."""
    os.makedirs(output_dir, exist_ok=True)

    # build address_id column: use existing column if present, else DataFrame index
    if 'address_id' in tract_addresses.columns:
        addr_ids = tract_addresses['address_id'].values
    else:
        addr_ids = tract_addresses.index.values

    # tract_target_mean in native units for each address row
    tract_target_mean_native = np.array([
        tract_target_means_std.get(fips, float('nan')) * tgt_std + tgt_mean
        for fips in tract_addresses['tract_fips'].values
    ])

    predictions_df = pd.DataFrame({
        'tract_fips': tract_addresses['tract_fips'].values,
        'address_id': addr_ids,
        'target_feature': target_feature,
        'architecture': arch,
        'seed': seed,
        'prediction_standardized': preds_std,
        'prediction_destandardized': preds_native,
        'true_value': true_values,
        'tract_target_mean': tract_target_mean_native,
    })
    predictions_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)

    metrics_df = _compute_per_tract_metrics(
        tract_addresses=tract_addresses,
        preds_std=preds_std,
        preds_native=preds_native,
        true_values=true_values,
        tract_target_means_std=tract_target_means_std,
        tgt_mean=tgt_mean,
        tgt_std=tgt_std,
    )
    metrics_df.to_csv(os.path.join(output_dir, 'per_tract_metrics.csv'), index=False)

    # run metadata
    meta = {
        'target_feature': target_feature,
        'target_mode': target_mode,
        'target_name': target_name if target_name is not None else target_feature,
        'target_source': target_source,
        'n_addresses_matched': n_addresses_matched,
        'n_addresses_missing': n_addresses_missing,
        'architecture': arch,
        'seed': seed,
        'n_features_after_drop': n_features_after_drop,
        'standardize_target': standardize,
        'target_mean_native': tgt_mean,
        'target_std_native': tgt_std,
        'git_sha': _get_git_sha(),
        'config': config,
    }
    with open(os.path.join(output_dir, 'run_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2, default=str)


def run_recovery(
    config: dict,
    target_feature: Optional[str] = None,
    output_dir: Optional[str] = None,
    verbose: bool = False,
    external_target_path: Optional[str] = None,
) -> dict:
    """
    Run held-out feature recovery or external target experiment.

    Held-out feature path (target_feature is not None):
        Drops target_feature from the input feature matrix, computes
        per-tract means as constraints, trains the GNN, and records
        address-level recovery metrics.

    External target path (external_target_path is not None):
        Loads an externally-supplied CSV as the target, uses the full
        73-feature matrix as input (no column dropped), and proceeds
        through the identical downstream pipeline.

    Exactly one of target_feature or external_target_path must be non-None.

    Parameters
    ----------
    config : dict
        Full GRANITE config dict. Must include data.target_fips.
    target_feature : str, optional
        Name of the feature column to hold out.
    output_dir : str, optional
        Directory for output files.
    verbose : bool
        Enable progress logging.
    external_target_path : str, optional
        Path to CSV with columns address_id, target_value (and optionally
        tract_fips). Gzipped CSV is supported.

    Returns
    -------
    dict with keys: success, output_dir, per_tract_errors, error (on failure)
    """
    if target_feature is not None and external_target_path is not None:
        raise ValueError(
            "run_recovery: target_feature and external_target_path are mutually "
            "exclusive; provide exactly one."
        )
    if target_feature is None and external_target_path is None:
        raise ValueError(
            "run_recovery: exactly one of target_feature or external_target_path "
            "must be provided."
        )
    if output_dir is None:
        raise ValueError("run_recovery: output_dir is required.")
    seed = config.get('processing', {}).get('random_seed', 42)
    set_random_seed(seed)

    arch = config.get('model', {}).get('architecture', 'gcn_gat')
    target_fips = config.get('data', {}).get('target_fips')
    n_neighbors = config.get('data', {}).get('neighbor_tracts', 0)

    if not target_fips:
        return {'success': False, 'error': 'data.target_fips not set in config'}

    # instantiate pipeline to reuse data loading and feature methods
    pipeline = GRANITEPipeline(config, output_dir=output_dir)
    pipeline.verbose = verbose

    # load spatial data
    try:
        data = pipeline._load_spatial_data()
    except Exception as e:
        return {'success': False, 'error': f'data loading failed: {e}'}

    # build tract list
    if n_neighbors > 0:
        try:
            tract_list = pipeline.data_loader.get_neighboring_tracts(target_fips, n_neighbors)
        except Exception as e:
            return {'success': False, 'error': f'get_neighboring_tracts failed: {e}'}
    else:
        tract_list = [target_fips]

    # load and combine addresses
    all_addresses = []
    for fips in tract_list:
        fips = str(fips).strip()
        if len(data['tracts'][data['tracts']['FIPS'] == fips]) == 0:
            if verbose:
                print(f'[recovery] skipping {fips}: not in tracts GDF')
            continue
        addrs = pipeline.data_loader.get_addresses_for_tract(fips)
        if len(addrs) == 0:
            if verbose:
                print(f'[recovery] skipping {fips}: no addresses')
            continue
        addrs['tract_fips'] = fips
        all_addresses.append(addrs)

    if not all_addresses:
        return {'success': False, 'error': 'no addresses loaded for any tract'}

    tract_addresses = pd.concat(all_addresses, ignore_index=True)
    valid_fips_list = [str(f).strip() for f in tract_list
                       if len(data['tracts'][data['tracts']['FIPS'] == str(f).strip()]) > 0]

    if verbose:
        print(f'[recovery] {len(tract_addresses)} addresses across {len(valid_fips_list)} tracts')

    # compute full accessibility feature matrix (pre-normalization)
    try:
        accessibility_features = pipeline._compute_accessibility_features(tract_addresses, data)
    except Exception as e:
        return {'success': False, 'error': f'feature computation raised: {e}\n{traceback.format_exc()}'}

    if accessibility_features is None:
        return {'success': False, 'error': 'feature computation returned None'}

    feature_names = pipeline._generate_feature_names(accessibility_features.shape[1])

    # --- target materialization: diverges by path, converges below ---

    ext_meta: Optional[dict] = None

    if target_feature is not None:
        # held-out feature path: validate, extract, drop
        if target_feature not in feature_names:
            return {
                'success': False,
                'error': (
                    f"target_feature '{target_feature}' not found in computed feature matrix "
                    f"({len(feature_names)} features). "
                    f"Available: {feature_names}"
                ),
            }

        feat_idx = feature_names.index(target_feature)
        raw_target_values = accessibility_features[:, feat_idx].copy().astype(float)

        feat_matrix = np.delete(accessibility_features, feat_idx, axis=1)
        feat_names_reduced = [n for i, n in enumerate(feature_names) if i != feat_idx]

        if verbose:
            print(f'[recovery] feature matrix: {accessibility_features.shape[1]} -> '
                  f'{feat_matrix.shape[1]} (dropped "{target_feature}")')

        _t_mode = 'held_out_feature'
        _t_name = target_feature
        _t_source = None
        _n_matched = len(tract_addresses)
        _n_missing = 0

    else:
        # external target path: load CSV, use full feature matrix
        if 'address_id' in tract_addresses.columns:
            addr_index = pd.Index(tract_addresses['address_id'].values)
        else:
            addr_index = tract_addresses.index

        try:
            raw_target_values, ext_meta = load_external_target(
                path=external_target_path,
                address_index=addr_index,
            )
        except Exception as e:
            return {'success': False, 'error': f'load_external_target failed: {e}'}

        feat_matrix = accessibility_features
        feat_names_reduced = feature_names

        if verbose:
            print(
                f'[recovery] external target "{ext_meta["target_name"]}": '
                f'matched {ext_meta["n_matched"]}/{len(tract_addresses)} addresses, '
                f'missing {ext_meta["n_missing"]}'
            )

        _t_mode = 'external'
        _t_name = ext_meta['target_name']
        _t_source = external_target_path
        _n_matched = ext_meta['n_matched']
        _n_missing = ext_meta['n_missing']

    # --- convergence point: identical downstream from here ---

    # standardize across all addresses in the run
    standardize = config.get('recovery', {}).get('standardize_target', True)
    if standardize:
        tgt_mean = float(np.nanmean(raw_target_values))
        tgt_std = float(np.nanstd(raw_target_values))
        if tgt_std < 1e-8:
            tgt_std = 1.0
        target_values_std = (raw_target_values - tgt_mean) / tgt_std
    else:
        tgt_mean = 0.0
        tgt_std = 1.0
        target_values_std = raw_target_values.copy()

    if verbose:
        tname = target_feature if target_feature else _t_name
        print(f'[recovery] target "{tname}": '
              f'mean={tgt_mean:.4f}, std={tgt_std:.4f}, standardize={standardize}')

    # compute per-tract means on the standardized target (these become the constraints)
    # for external targets, NaN entries (unmatched addresses) are excluded from the mean
    tract_target_means_std: Dict[str, float] = {}
    for fips in valid_fips_list:
        mask = (tract_addresses['tract_fips'] == fips).values
        vals = target_values_std[mask]
        finite_vals = vals[np.isfinite(vals)]
        if len(finite_vals) > 0:
            tract_target_means_std[fips] = float(np.mean(finite_vals))

    if verbose:
        for fips, v in tract_target_means_std.items():
            print(f'[recovery]   tract {fips} target mean (std): {v:.4f}')

    # normalize (first pass - matches pipeline._process_single_tract behavior)
    normalized_features, _ = normalize_accessibility_features(feat_matrix)
    normalized_features = pipeline._apply_feature_mode(
        normalized_features,
        tract_addresses,
        mode=config.get('feature_mode', 'full'),
        seed=seed,
    )

    # context features
    context_features = pipeline.data_loader.create_context_features_for_addresses(
        addresses=tract_addresses,
        svi_data=data['svi'],
    )
    normalized_context, _ = pipeline.data_loader.normalize_context_features(context_features)

    # build graph
    import torch
    graph_data = pipeline.data_loader.create_spatial_accessibility_graph(
        addresses=tract_addresses,
        accessibility_features=normalized_features,
        context_features=normalized_context,
        state_fips=target_fips[:2],
        county_fips=target_fips[2:5],
    )

    if verbose:
        print(f'[recovery] graph: {graph_data.x.shape[0]} nodes, '
              f'{graph_data.edge_index.shape[1]} edges, '
              f'{graph_data.x.shape[1]} node features')

    # second normalization pass - replicates _train_multi_tract_gnn behavior
    normed2, _ = normalize_accessibility_features(graph_data.x.numpy())
    graph_data.x = torch.FloatTensor(normed2)

    # model construction
    has_context = hasattr(graph_data, 'context') and graph_data.context is not None
    context_dim = graph_data.context.shape[1] if has_context else 5
    use_gating = config.get('model', {}).get('use_context_gating', True)

    ModelClass = GraphSAGEAccessibilitySVIGNN if arch == 'sage' else AccessibilitySVIGNN
    model = ModelClass(
        accessibility_features_dim=graph_data.x.shape[1],
        context_features_dim=context_dim,
        hidden_dim=config.get('model', {}).get('hidden_dim', 64),
        dropout=config.get('model', {}).get('dropout', 0.3),
        seed=seed,
        use_context_gating=use_gating and has_context,
    )

    # tract masks
    tract_masks = {
        fips: (tract_addresses['tract_fips'] == fips).values
        for fips in tract_target_means_std
    }

    # trainer config: disable block-group and ordering constraints for M1
    training_config = {
        **config.get('training', {}),
        'use_multitask': True,
        'bg_constraint_weight': 0.0,
        'ordering_weight': 0.0,
    }
    trainer = MultiTractGNNTrainer(model, config=training_config, seed=seed)

    epochs = config.get('training', {}).get('epochs',
             config.get('model', {}).get('epochs', 100))

    if verbose:
        print(f'[recovery] training {arch} for {epochs} epochs '
              f'with {len(tract_target_means_std)} tract constraints')

    try:
        training_result = trainer.train(
            graph_data=graph_data,
            tract_svis=tract_target_means_std,
            tract_masks=tract_masks,
            epochs=epochs,
            verbose=verbose,
            feature_names=feat_names_reduced,
            block_group_targets=None,
            block_group_masks=None,
            ordering_values=None,
        )
    except Exception as e:
        return {'success': False, 'error': f'training raised: {e}\n{traceback.format_exc()}'}

    if not training_result.get('success', False):
        return {'success': False, 'error': training_result.get('error', 'training failed')}

    raw_preds = training_result['final_predictions']
    preds_native = raw_preds * tgt_std + tgt_mean

    _write_outputs(
        output_dir=output_dir,
        tract_addresses=tract_addresses,
        target_feature=target_feature or _t_name,
        arch=arch,
        seed=seed,
        preds_std=raw_preds,
        preds_native=preds_native,
        true_values=raw_target_values,
        tract_target_means_std=tract_target_means_std,
        tgt_mean=tgt_mean,
        tgt_std=tgt_std,
        config=config,
        n_features_after_drop=feat_matrix.shape[1],
        standardize=standardize,
        target_mode=_t_mode,
        target_name=_t_name,
        target_source=_t_source,
        n_addresses_matched=_n_matched,
        n_addresses_missing=_n_missing,
    )

    if verbose:
        print(f'[recovery] outputs written to {output_dir}')
        print(f'[recovery] overall constraint error: '
              f'{training_result["overall_constraint_error"]:.2f}%')

    return {
        'success': True,
        'output_dir': output_dir,
        'overall_constraint_error': training_result['overall_constraint_error'],
        'per_tract_errors': training_result['per_tract_errors'],
        'epochs_trained': training_result['epochs_trained'],
    }
