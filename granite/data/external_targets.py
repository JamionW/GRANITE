"""
External target loader for GRANITE Door 2 evaluation path.

Reads a CSV with columns address_id, target_value, optional tract_fips.
Returns a numpy array aligned to address_index with NaN for unmatched
addresses, plus a metadata dict.
"""
import gzip
import os
import sys
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def load_external_target(
    path: str,
    address_index: pd.Index,
    tract_filter: Optional[List[str]] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Load external target CSV and align to address_index.

    Parameters
    ----------
    path : str
        Path to CSV (plain or gzipped) with columns address_id, target_value.
        Optional column: tract_fips.
    address_index : pd.Index
        Address identifiers to align to (address_id values or integer index).
    tract_filter : list of str, optional
        If provided and the CSV has a tract_fips column, restrict to matching rows.

    Returns
    -------
    target_array : np.ndarray, shape (len(address_index),)
        Values aligned to address_index. NaN for unmatched addresses.
    metadata : dict
        source: file path
        n_total: rows in CSV (after tract_filter if applied)
        n_matched: addresses found in address_index
        n_missing: len(address_index) - n_matched
        target_name: CSV column header (non-address_id) or filename stem
        value_range: dict with min, max, mean (over matched values)

    Raises
    ------
    ValueError
        If zero addresses match between the CSV and address_index.
    """
    if path.endswith('.gz'):
        with gzip.open(path, 'rt') as fh:
            df = pd.read_csv(fh)
    else:
        df = pd.read_csv(path)

    if 'address_id' not in df.columns:
        raise ValueError(
            f"CSV at '{path}' must contain an 'address_id' column; "
            f"found columns: {list(df.columns)}"
        )

    # determine target value column and name
    value_col = 'target_value'
    if value_col not in df.columns:
        candidate_cols = [
            c for c in df.columns
            if c not in ('address_id', 'tract_fips')
        ]
        if not candidate_cols:
            raise ValueError(
                f"cannot find target value column in '{path}': "
                f"columns={list(df.columns)}"
            )
        value_col = candidate_cols[0]
        target_name = value_col
    else:
        # use filename stem when column is the generic 'target_value'
        stem = os.path.basename(path)
        if stem.endswith('.gz'):
            stem = os.path.splitext(stem)[0]
        target_name = os.path.splitext(stem)[0]

    # apply optional tract filter
    if tract_filter is not None and 'tract_fips' in df.columns:
        str_filter = [str(t) for t in tract_filter]
        df = df[df['tract_fips'].astype(str).isin(str_filter)]

    n_total = len(df)

    # build lookup: address_id -> target_value
    df_indexed = df.set_index('address_id')[value_col]

    target_array = np.full(len(address_index), np.nan)
    n_matched = 0
    for i, addr_id in enumerate(address_index):
        if addr_id in df_indexed.index:
            val = df_indexed.loc[addr_id]
            # handle duplicate keys: take first value
            if hasattr(val, '__len__'):
                val = val.iloc[0] if hasattr(val, 'iloc') else val[0]
            if pd.notna(val):
                target_array[i] = float(val)
                n_matched += 1

    n_missing = len(address_index) - n_matched

    if n_matched == 0:
        raise ValueError(
            f"load_external_target: zero addresses matched between "
            f"'{path}' and the provided address_index "
            f"(CSV n_total={n_total}, address_index size={len(address_index)}). "
            f"Check that address_id values are consistent."
        )

    matched_frac = n_matched / len(address_index)
    if matched_frac < 0.80:
        msg = (
            f"load_external_target: matched fraction {matched_frac:.3f} is below 0.80 "
            f"({n_matched}/{len(address_index)} addresses matched from '{path}')"
        )
        warnings.warn(msg, stacklevel=2)
        print(f'[external_target] warning: {msg}', file=sys.stderr)

    matched_vals = target_array[np.isfinite(target_array)]
    value_range = {
        'min': float(matched_vals.min()) if len(matched_vals) > 0 else float('nan'),
        'max': float(matched_vals.max()) if len(matched_vals) > 0 else float('nan'),
        'mean': float(matched_vals.mean()) if len(matched_vals) > 0 else float('nan'),
    }

    metadata = {
        'source': path,
        'n_total': n_total,
        'n_matched': n_matched,
        'n_missing': n_missing,
        'target_name': target_name,
        'value_range': value_range,
    }
    return target_array, metadata
