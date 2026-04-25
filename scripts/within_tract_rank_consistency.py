"""
Within-tract feature-to-prediction rank consistency analysis.

Tests whether within-tract Spearman correlations between each feature and
predicted SVI are consistent in sign across tracts, separately for GraphSAGE
(sage) and GCN-GAT (gcn_gat).

This script requires raw pre-correction predictions. Post-correction predictions
are clipped to [0,1] and produce attenuated Spearman correlations in tracts with
SVI near 0 or 1 because the clip creates ties in the prediction vector. The
additive mean-shift that implements constraint correction preserves Spearman rank
order, but the subsequent [0,1] clip does not -- ranks at the boundaries collapse
to tied values and rho magnitudes are underestimated. Use the raw_prediction
column (pre-clip, pre-shift GNN output) whenever it is available.

Expected input layout (at least one of):

  Layout A: per-architecture subdirectories under a shared base dir
    {base_dir}/graphsage/tract_{fips}/granite_predictions*.csv
    {base_dir}/graphsage/tract_{fips}/accessibility_features.csv
    {base_dir}/gcn_gat/tract_{fips}/granite_predictions*.csv
    {base_dir}/gcn_gat/tract_{fips}/accessibility_features.csv

  Layout B: separate dirs passed via --sage-dir / --gcn-dir
    {sage_dir}/{fips}/granite_predictions*.csv   (or tract_{fips}/)
    {sage_dir}/{fips}/accessibility_features.csv
    {gcn_dir}/...

Predictions file: prefers raw_prediction column (pre-constraint-correction);
falls back to mean (post-correction, clipped). See raw-prediction note above.

Usage
-----
    python scripts/within_tract_rank_consistency.py \\
        --sage-dir output/run_sage \\
        --gcn-dir  output/run_gcn

    python scripts/within_tract_rank_consistency.py \\
        --base-dir output/architecture_run \\
        --cv-threshold 0.10 \\
        --min-tracts 5 \\
        --min-addresses 50 \\
        --results-dir results/rank_consistency
"""

import argparse
import os
import sys
import glob
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

# ---------------------------------------------------------------------------
# feature name registry
# ---------------------------------------------------------------------------

def read_feature_names(csv_path):
    """Read feature column names from the header row of accessibility_features.csv.

    Parameters
    ----------
    csv_path : str or Path
        Path to an accessibility_features.csv file.

    Returns
    -------
    list of str
        Column names in CSV header order.
    """
    with open(csv_path, 'r') as f:
        header_line = f.readline().strip()
    names = [c.strip() for c in header_line.split(',')]
    if not names or names == ['']:
        raise ValueError(f"empty or unreadable header in {csv_path}")
    return names


# binary (0/1) one-hot features. for these, CV is undefined or degenerate when
# the column is near-constant within a tract. instead of CV, use minority-class
# fraction threshold: signal-bearing iff fraction >= onehot_min_fraction (0.05).
ONE_HOT_FEATURES = frozenset({
    'in_sfha', 'is_residential',
    'lucode_residential', 'lucode_commercial', 'lucode_industrial', 'lucode_other',
    'proptype_residential', 'proptype_apartment_10plus', 'proptype_commercial',
    'proptype_rental_40pct', 'proptype_cha_housing',
})


# ---------------------------------------------------------------------------
# data discovery
# ---------------------------------------------------------------------------

def _find_tract_dirs_in(base):
    """Return {fips: (pred_path, feat_path)} for all tract dirs under base.

    Accepts subdirectory names of the form:
      tract_{fips}   e.g. tract_47065000600
      {fips}         e.g. 47065000600
    """
    result = {}
    base = Path(base)
    if not base.is_dir():
        return result

    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        # extract fips from directory name
        if name.startswith('tract_'):
            fips = name[len('tract_'):]
        elif name.isdigit() and len(name) in (10, 11):
            fips = name
        else:
            continue

        feat_path = d / 'accessibility_features.csv'
        if not feat_path.exists():
            continue

        # match granite_predictions*.csv (may have tract/mode suffix)
        pred_candidates = sorted(d.glob('granite_predictions*.csv'))
        if not pred_candidates:
            continue
        pred_path = pred_candidates[0]  # take first match

        result[fips] = (str(pred_path), str(feat_path))

    return result


def discover_inputs(args):
    """Return (sage_tracts, gcn_tracts) where each is {fips: (pred, feat)}.

    Raises SystemExit with a descriptive message if discovery fails.
    """
    if args.base_dir:
        base = Path(args.base_dir)
        sage_tracts = _find_tract_dirs_in(base / 'graphsage')
        gcn_tracts = _find_tract_dirs_in(base / 'gcn_gat')
        # also accept sage / gcn subdirectory names
        if not sage_tracts:
            sage_tracts = _find_tract_dirs_in(base / 'sage')
        if not gcn_tracts:
            gcn_tracts = _find_tract_dirs_in(base / 'gcn')
    else:
        sage_tracts = _find_tract_dirs_in(args.sage_dir) if args.sage_dir else {}
        gcn_tracts = _find_tract_dirs_in(args.gcn_dir) if args.gcn_dir else {}

    if not sage_tracts and not gcn_tracts:
        sys.exit(
            "ERROR: no tract directories found. Provide --base-dir "
            "(containing graphsage/ and gcn_gat/ subdirs) or --sage-dir / "
            "--gcn-dir pointing to per-tract output directories."
        )

    n_sage = len(sage_tracts)
    n_gcn = len(gcn_tracts)
    print(f"discovered {n_sage} GraphSAGE tract(s), {n_gcn} GCN-GAT tract(s)")

    # warn if the two architecture sets cover different tracts
    sage_fips = set(sage_tracts)
    gcn_fips = set(gcn_tracts)
    only_sage = sage_fips - gcn_fips
    only_gcn = gcn_fips - sage_fips
    if only_sage:
        print(f"  warning: {len(only_sage)} tract(s) have SAGE but not GCN: "
              f"{sorted(only_sage)}")
    if only_gcn:
        print(f"  warning: {len(only_gcn)} tract(s) have GCN but not SAGE: "
              f"{sorted(only_gcn)}")

    return sage_tracts, gcn_tracts


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------

def load_tract_data(pred_path, feat_path, arch_label=None, do_diagnostic=False):
    """Load predictions vector and raw feature matrix for one tract.

    Prediction source preference:
      1. raw_prediction column (pre-constraint-correction, pre-clip)
      2. mean column (post-correction; additive shift preserves Spearman rank
         order, but the [0,1] clip applied at boundary-SVI tracts creates ties
         that attenuate Spearman correlations)

    See module docstring for why raw pre-correction predictions are required.

    Parameters
    ----------
    pred_path : str
    feat_path : str
    arch_label : str or None
        Architecture label used in diagnostic output.
    do_diagnostic : bool
        If True, print column selection diagnostics to stderr (once per arch).

    Returns (predictions: np.ndarray shape (n,), features: np.ndarray shape (n, d)).
    Raises ValueError if files cannot be read, shapes are inconsistent, or
    post-correction predictions are detected when raw are available.
    """
    preds_df = pd.read_csv(pred_path)
    if 'raw_prediction' in preds_df.columns:
        col_name = 'raw_prediction'
    elif 'mean' in preds_df.columns:
        col_name = 'mean'
    else:
        raise ValueError(f"{pred_path}: neither 'raw_prediction' nor 'mean' column found")

    predictions = preds_df[col_name].values.astype(float)

    if do_diagnostic:
        # runs once per architecture; greppable via DIAGNOSTIC prefix
        other_pred_cols = [
            c for c in preds_df.columns
            if c != col_name and any(
                kw in c.lower()
                for kw in ('pred', 'mean', 'svi', 'score', 'output', 'corrected')
            )
        ]
        label = arch_label or 'unknown'
        print(f"DIAGNOSTIC [{label}] first-tract prediction column selection:", file=sys.stderr)
        print(
            f"DIAGNOSTIC   selected: {col_name}  "
            f"min={predictions.min():.4f}  max={predictions.max():.4f}  "
            f"mean={predictions.mean():.4f}",
            file=sys.stderr,
        )
        for oc in other_pred_cols:
            ov = preds_df[oc].values.astype(float)
            print(
                f"DIAGNOSTIC   other '{oc}':  "
                f"min={ov.min():.4f}  max={ov.max():.4f}  mean={ov.mean():.4f}",
                file=sys.stderr,
            )

    # detect likely post-correction (clipped) predictions loaded instead of raw
    p_min = float(predictions.min())
    p_max = float(predictions.max())
    if p_min >= 0.0 and p_max <= 1.0 and col_name != 'raw_prediction':
        if 'raw_prediction' in preds_df.columns:
            raw = preds_df['raw_prediction'].values.astype(float)
            if float(raw.min()) < 0.0 or float(raw.max()) > 1.0:
                raise ValueError(
                    f"{pred_path}: selected column '{col_name}' is entirely within [0,1] "
                    f"but 'raw_prediction' has values outside [0,1] "
                    f"(raw min={raw.min():.4f}, max={raw.max():.4f}). "
                    "this script may be loading post-correction predictions instead of raw; "
                    "see module docstring."
                )

    feat_df = pd.read_csv(feat_path, header=0)
    features = feat_df.values.astype(float)

    if len(predictions) != len(features):
        raise ValueError(
            f"shape mismatch: {pred_path} has {len(predictions)} rows, "
            f"{feat_path} has {len(features)} rows"
        )

    return predictions, features


# ---------------------------------------------------------------------------
# CV and one-hot signal detection
# ---------------------------------------------------------------------------

def is_signal_bearing(col_values, feature_name, cv_threshold, onehot_min_fraction=0.05):
    """Decide whether a feature column carries signal within one tract.

    Parameters
    ----------
    col_values : np.ndarray, shape (n,)
        Feature values for addresses within the tract.
    feature_name : str
        Used to detect one-hot encoded binary features.
    cv_threshold : float
        Minimum coefficient of variation required for continuous features.
    onehot_min_fraction : float
        Minimum minority-class fraction for binary one-hot features (default 0.05).
        Replaces CV for features in ONE_HOT_FEATURES where CV is undefined or
        degenerate when the column is near-constant.

    Returns
    -------
    (signal : bool, metric : float)
        metric is the minority fraction (for one-hot) or CV (for continuous).
    """
    n = len(col_values)
    if n == 0:
        return False, 0.0

    if feature_name in ONE_HOT_FEATURES:
        # binary column: count minority class fraction
        n_pos = float(np.sum(col_values > 0.5))
        minority = min(n_pos, n - n_pos) / n
        return minority >= onehot_min_fraction, minority

    std = float(np.std(col_values))
    if std < 1e-10:
        # constant within tract; no within-tract rank signal
        return False, 0.0

    mean = float(np.mean(col_values))
    if abs(mean) < 1e-8:
        # near-zero mean: std/|mean| would blow up. treat as signal-bearing
        # whenever std is nonzero (infinite CV), since any variation relative
        # to a zero baseline represents substantial relative spread.
        return True, float('inf')

    cv = std / abs(mean)
    return cv >= cv_threshold, cv


# ---------------------------------------------------------------------------
# per-tract Spearman computation
# ---------------------------------------------------------------------------

def compute_tract_rhos(predictions, features, feature_names, fips,
                       cv_threshold, min_addresses=50, onehot_min_fraction=0.05):
    """Compute Spearman rho between each signal-bearing feature and predictions.

    Parameters
    ----------
    predictions : np.ndarray, shape (n,)
    features : np.ndarray, shape (n, d)
    feature_names : list of str, length d
    fips : str
    cv_threshold : float
    min_addresses : int
        Tracts with fewer than this many addresses are skipped (insufficient
        rank resolution for reliable Spearman estimates).
    onehot_min_fraction : float

    Returns
    -------
    list of dicts with keys: fips, feature, rho, cv, n_addresses
    """
    n = len(predictions)
    if n < min_addresses:
        return []

    d = features.shape[1]
    if d != len(feature_names):
        warnings.warn(
            f"tract {fips}: feature matrix has {d} columns but expected "
            f"{len(feature_names)} feature names; using numeric fallback names"
        )
        names = [f'feature_{i}' for i in range(d)]
    else:
        names = feature_names

    rows = []
    for i, fname in enumerate(names):
        col = features[:, i]
        signal, metric = is_signal_bearing(col, fname, cv_threshold, onehot_min_fraction)
        if not signal:
            continue

        # spearmanr returns NaN if one array is constant; guard defensively
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            rho, _ = stats.spearmanr(col, predictions)

        if not np.isfinite(rho):
            continue

        rows.append({
            'fips': fips,
            'feature': fname,
            'rho': float(rho),
            'cv': float(metric) if np.isfinite(metric) else float('nan'),
            'n_addresses': n,
        })

    return rows


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

def aggregate_rhos(per_tract_rows, feature_names, min_tracts=5):
    """Aggregate per-tract rho values to per-feature summary statistics.

    Only features with >= min_tracts signal-bearing tract observations are
    included in the summary.

    consistent_flag = True when:
      - sign-test p < 0.05 (sign is non-random across tracts), AND
      - median |rho| >= 0.10 (effect is not negligible in magnitude).

    Returns
    -------
    pd.DataFrame with columns:
      feature, n_tracts_signal_bearing, median_rho, iqr_rho, mean_rho,
      n_positive, n_negative, sign_test_p, wilcoxon_p, consistent_flag
    """
    if not per_tract_rows:
        return pd.DataFrame()

    df = pd.DataFrame(per_tract_rows)
    rows = []

    for fname in feature_names:
        subset = df[df['feature'] == fname]['rho'].dropna().values
        n = len(subset)
        if n < min_tracts:
            continue

        n_pos = int(np.sum(subset > 0))
        n_neg = int(np.sum(subset < 0))
        n_nonzero = n_pos + n_neg

        median_rho = float(np.median(subset))
        iqr_rho = float(np.percentile(subset, 75) - np.percentile(subset, 25))
        mean_rho = float(np.mean(subset))

        # two-sided sign test via binomtest
        if n_nonzero > 0:
            binom_result = stats.binomtest(n_pos, n_nonzero, p=0.5, alternative='two-sided')
            sign_test_p = float(binom_result.pvalue)
        else:
            sign_test_p = 1.0

        # wilcoxon signed-rank test (requires >= 10 observations for reliability;
        # suppress warning for small n which scipy handles gracefully)
        if n >= 5 and np.any(subset != 0):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                try:
                    _, wilcoxon_p = stats.wilcoxon(subset, alternative='two-sided')
                    wilcoxon_p = float(wilcoxon_p)
                except ValueError:
                    # all values identical -- edge case
                    wilcoxon_p = 1.0
        else:
            wilcoxon_p = float('nan')

        consistent_flag = (sign_test_p < 0.05) and (abs(median_rho) >= 0.10)

        rows.append({
            'feature': fname,
            'n_tracts_signal_bearing': n,
            'median_rho': median_rho,
            'iqr_rho': iqr_rho,
            'mean_rho': mean_rho,
            'n_positive': n_pos,
            'n_negative': n_neg,
            'sign_test_p': sign_test_p,
            'wilcoxon_p': wilcoxon_p,
            'consistent_flag': consistent_flag,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# output writing
# ---------------------------------------------------------------------------

def write_summary_txt(sage_summary, gcn_summary, out_path):
    """Write text report with four labeled sections of consistently-signed features.

    Section A: consistent under GraphSAGE only (sage True, gcn False)
    Section B: consistent under GCN-GAT only (gcn True, sage False)
    Section C: consistent under both, same sign
    Section D: consistent under both, opposite signs -- headline finding
    """
    sage_consistent = set()
    gcn_consistent = set()

    if not sage_summary.empty:
        sage_consistent = set(sage_summary.loc[sage_summary['consistent_flag'], 'feature'])
    if not gcn_summary.empty:
        gcn_consistent = set(gcn_summary.loc[gcn_summary['consistent_flag'], 'feature'])

    sage_only = sage_consistent - gcn_consistent
    gcn_only = gcn_consistent - sage_consistent
    both = sage_consistent & gcn_consistent

    sage_idx = sage_summary.set_index('feature') if not sage_summary.empty else pd.DataFrame()
    gcn_idx = gcn_summary.set_index('feature') if not gcn_summary.empty else pd.DataFrame()

    # split "both" into same-sign and sign-flip (Section C vs D)
    both_same_sign = set()
    both_flip_sign = set()
    for f in both:
        s_sign = np.sign(sage_idx.loc[f, 'median_rho']) if f in sage_idx.index else 0
        g_sign = np.sign(gcn_idx.loc[f, 'median_rho']) if f in gcn_idx.index else 0
        if s_sign != 0 and g_sign != 0 and s_sign != g_sign:
            both_flip_sign.add(f)
        else:
            both_same_sign.add(f)

    def _fmt_single(features_set, summary_idx):
        """Format features from one-architecture sections (A or B)."""
        if not features_set:
            return ['  (none)']
        rows_list = []
        for f in features_set:
            if f in summary_idx.index:
                rows_list.append((
                    f,
                    float(summary_idx.loc[f, 'median_rho']),
                    int(summary_idx.loc[f, 'n_tracts_signal_bearing']),
                ))
        rows_list.sort(key=lambda x: abs(x[1]), reverse=True)
        out = [
            f"  {'feature':<40s}  {'median_rho':>10s}  {'n_tracts':>8s}",
            f"  {'-'*40}  {'-'*10}  {'-'*8}",
        ]
        for fname, rho, n in rows_list:
            out.append(f"  {fname:<40s}  {rho:>10.4f}  {n:>8d}")
        return out

    def _fmt_both(features_set):
        """Format features consistent under both architectures (Sections C and D)."""
        if not features_set:
            return ['  (none)']
        rows_list = []
        for f in features_set:
            s_rho = float(sage_idx.loc[f, 'median_rho']) if f in sage_idx.index else float('nan')
            g_rho = float(gcn_idx.loc[f, 'median_rho']) if f in gcn_idx.index else float('nan')
            s_n = int(sage_idx.loc[f, 'n_tracts_signal_bearing']) if f in sage_idx.index else 0
            g_n = int(gcn_idx.loc[f, 'n_tracts_signal_bearing']) if f in gcn_idx.index else 0
            rows_list.append((f, s_rho, g_rho, s_n, g_n))
        rows_list.sort(
            key=lambda x: abs(x[1] if np.isfinite(x[1]) else 0)
                        + abs(x[2] if np.isfinite(x[2]) else 0),
            reverse=True,
        )
        out = [
            f"  {'feature':<40s}  {'SAGE_rho':>10s}  {'GCN_rho':>10s}  {'SAGE_n':>6s}  {'GCN_n':>6s}",
            f"  {'-'*40}  {'-'*10}  {'-'*10}  {'-'*6}  {'-'*6}",
        ]
        for fname, s_rho, g_rho, s_n, g_n in rows_list:
            out.append(
                f"  {fname:<40s}  {s_rho:>10.4f}  {g_rho:>10.4f}  {s_n:>6d}  {g_n:>6d}"
            )
        return out

    lines = []
    lines.append("=" * 72)
    lines.append("WITHIN-TRACT RANK CONSISTENCY REPORT")
    lines.append("=" * 72)
    lines.append("")
    # four-line preamble: one count per section
    lines.append(f"section A (GraphSAGE only):              {len(sage_only)}")
    lines.append(f"section B (GCN-GAT only):                {len(gcn_only)}")
    lines.append(f"section C (both, same sign):             {len(both_same_sign)}")
    lines.append(f"section D (both, sign-flip) [HEADLINE]:  {len(both_flip_sign)}")
    lines.append("")

    lines.append("=" * 72)
    lines.append("SECTION A: consistent under GraphSAGE only")
    lines.append("  sage consistent_flag=True, gcn_gat consistent_flag=False")
    lines.append("-" * 72)
    lines.extend(_fmt_single(sage_only, sage_idx))
    lines.append("")

    lines.append("=" * 72)
    lines.append("SECTION B: consistent under GCN-GAT only")
    lines.append("  gcn_gat consistent_flag=True, sage consistent_flag=False")
    lines.append("-" * 72)
    lines.extend(_fmt_single(gcn_only, gcn_idx))
    lines.append("")

    lines.append("=" * 72)
    lines.append("SECTION C: consistent under both architectures, same sign")
    lines.append("  both consistent_flag=True, sign(median_rho_sage) == sign(median_rho_gcn)")
    lines.append("-" * 72)
    lines.extend(_fmt_both(both_same_sign))
    lines.append("")

    lines.append("=" * 72)
    lines.append("SECTION D: sign-flippers across architectures  [HEADLINE FINDING]")
    lines.append("  features consistent in both architectures with opposite median_rho signs")
    lines.append("  both consistent_flag=True, sign(median_rho_sage) != sign(median_rho_gcn)")
    lines.append("-" * 72)
    lines.extend(_fmt_both(both_flip_sign))
    lines.append("")
    lines.append("=" * 72)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    return sage_only, gcn_only, both_same_sign, both_flip_sign


def print_terminal_counts(sage_only, gcn_only, both_same_sign, both_flip_sign):
    print()
    print("=" * 60)
    print("RANK CONSISTENCY SUMMARY")
    print("=" * 60)
    print(f"  section A (SAGE only):               {len(sage_only)}")
    print(f"  section B (GCN-GAT only):             {len(gcn_only)}")
    print(f"  section C (both, same sign):          {len(both_same_sign)}")
    print(f"  section D (both, sign-flip) HEADLINE: {len(both_flip_sign)}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# per-architecture pipeline
# ---------------------------------------------------------------------------

def run_architecture(arch_label, tract_dict, cv_threshold,
                     min_addresses, onehot_min_fraction, min_tracts):
    """Load data, compute per-tract rhos, and aggregate for one architecture.

    Discovers feature names from the header of the first accessibility_features.csv
    encountered. Asserts that all subsequent CSVs in this architecture have an
    identical header; exits with the offending tract FIPS if not.

    Returns (per_tract_df, summary_df).
    """
    all_rows = []
    n_loaded = 0
    n_failed = 0
    reference_header = None
    reference_feat_path = None
    feature_names = None
    diagnostic_done = False

    for fips, (pred_path, feat_path) in sorted(tract_dict.items()):
        # read and enforce header consistency across all tracts in this architecture
        current_header = read_feature_names(feat_path)
        if reference_header is None:
            reference_header = current_header
            reference_feat_path = feat_path
            feature_names = current_header
            print(f"  [{arch_label}] reading feature names from header: "
                  f"{len(feature_names)} columns ({feat_path})")
        else:
            if current_header != reference_header:
                sys.exit(
                    f"ERROR: accessibility_features.csv header mismatch in tract {fips}.\n"
                    f"  reference ({reference_feat_path}): {len(reference_header)} columns\n"
                    f"  offending ({feat_path}): {len(current_header)} columns\n"
                    f"  offending tract FIPS: {fips}"
                )

        do_diag = not diagnostic_done
        try:
            predictions, features = load_tract_data(
                pred_path, feat_path,
                arch_label=arch_label,
                do_diagnostic=do_diag,
            )
            diagnostic_done = True
        except Exception as e:
            print(f"  [{arch_label}] {fips}: load error -- {e}")
            n_failed += 1
            continue

        rows = compute_tract_rhos(
            predictions, features, feature_names, fips,
            cv_threshold=cv_threshold,
            min_addresses=min_addresses,
            onehot_min_fraction=onehot_min_fraction,
        )
        all_rows.extend(rows)
        n_loaded += 1

    print(f"  [{arch_label}] loaded {n_loaded} tract(s), "
          f"{n_failed} failed, "
          f"{len(all_rows)} (feature, tract) rho values computed")

    if not all_rows:
        return pd.DataFrame(), pd.DataFrame()

    if feature_names is None:
        feature_names = []

    per_tract_df = pd.DataFrame(
        [{'architecture': arch_label, **r} for r in all_rows]
    )[['architecture', 'fips', 'feature', 'rho', 'cv', 'n_addresses']]

    summary_df = aggregate_rhos(all_rows, feature_names, min_tracts=min_tracts)
    if not summary_df.empty:
        summary_df.insert(0, 'architecture', arch_label)

    return per_tract_df, summary_df


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            'Test within-tract feature-to-prediction rank correlation consistency '
            'across tracts for GraphSAGE and GCN-GAT architectures.'
        )
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        '--base-dir', metavar='DIR',
        help=(
            'Base directory containing graphsage/ and gcn_gat/ subdirectories, '
            'each holding per-tract subdirectories with predictions and features.'
        )
    )

    parser.add_argument(
        '--sage-dir', metavar='DIR',
        help='Directory containing per-tract subdirs for GraphSAGE predictions.'
    )
    parser.add_argument(
        '--gcn-dir', metavar='DIR',
        help='Directory containing per-tract subdirs for GCN-GAT predictions.'
    )

    parser.add_argument(
        '--cv-threshold', type=float, default=0.10, metavar='FLOAT',
        help=(
            'Minimum coefficient of variation for a continuous feature to be '
            'considered signal-bearing within a tract (default: 0.10). '
            'For one-hot features, the threshold is minority-class fraction >= 0.05.'
        )
    )
    parser.add_argument(
        '--min-tracts', type=int, default=5, metavar='INT',
        help=(
            'Minimum number of signal-bearing tracts required to include a '
            'feature in the aggregated summary (default: 5).'
        )
    )
    parser.add_argument(
        '--min-addresses', type=int, default=50, metavar='INT',
        help=(
            'Tracts with fewer than this many addresses are skipped '
            '(insufficient rank resolution; default: 50).'
        )
    )
    parser.add_argument(
        '--results-dir', default='results/rank_consistency', metavar='DIR',
        help='Output directory for CSV and text results (default: results/rank_consistency).'
    )
    parser.add_argument(
        '--onehot-min-fraction', type=float, default=0.05, metavar='FLOAT',
        help=(
            'Minimum minority-class fraction for binary one-hot features to be '
            'considered signal-bearing within a tract (default: 0.05).'
        )
    )

    args = parser.parse_args()

    # validate: need at least one of --base-dir, --sage-dir / --gcn-dir
    if not args.base_dir and not args.sage_dir and not args.gcn_dir:
        parser.error(
            'provide --base-dir or at least one of --sage-dir / --gcn-dir'
        )

    # if sage/gcn dirs are given alongside base-dir that would be ambiguous;
    # base-dir takes precedence (handled in discover_inputs)
    if args.base_dir and (args.sage_dir or args.gcn_dir):
        print("note: --base-dir takes precedence over --sage-dir / --gcn-dir")

    print("within-tract rank consistency analysis")
    print(f"  cv_threshold:       {args.cv_threshold}")
    print(f"  min_tracts:         {args.min_tracts}")
    print(f"  min_addresses:      {args.min_addresses}")
    print(f"  onehot_min_frac:    {args.onehot_min_fraction}")
    print(f"  results_dir:        {args.results_dir}")
    print()

    sage_tracts, gcn_tracts = discover_inputs(args)
    print()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # run per architecture
    all_per_tract = []
    all_summary = []

    if sage_tracts:
        print("computing GraphSAGE rhos ...")
        sage_pt, sage_sum = run_architecture(
            'graphsage', sage_tracts,
            args.cv_threshold, args.min_addresses,
            args.onehot_min_fraction, args.min_tracts,
        )
        if not sage_pt.empty:
            all_per_tract.append(sage_pt)
        if not sage_sum.empty:
            all_summary.append(sage_sum)
    else:
        sage_sum = pd.DataFrame()

    if gcn_tracts:
        print("computing GCN-GAT rhos ...")
        gcn_pt, gcn_sum = run_architecture(
            'gcn_gat', gcn_tracts,
            args.cv_threshold, args.min_addresses,
            args.onehot_min_fraction, args.min_tracts,
        )
        if not gcn_pt.empty:
            all_per_tract.append(gcn_pt)
        if not gcn_sum.empty:
            all_summary.append(gcn_sum)
    else:
        gcn_sum = pd.DataFrame()

    # write per-tract rho CSV
    per_tract_out = results_dir / 'per_tract_rho.csv'
    if all_per_tract:
        per_tract_df = pd.concat(all_per_tract, ignore_index=True)
        per_tract_df.to_csv(per_tract_out, index=False)
        print(f"\nwrote {len(per_tract_df)} rows -> {per_tract_out}")
    else:
        print("\nno per-tract rho data to write")

    # write feature summary CSV
    feature_sum_out = results_dir / 'feature_summary.csv'
    if all_summary:
        summary_df = pd.concat(all_summary, ignore_index=True)
        summary_df.to_csv(feature_sum_out, index=False)
        print(f"wrote {len(summary_df)} rows -> {feature_sum_out}")
    else:
        print("no feature summary data to write")
        summary_df = pd.DataFrame()

    # write text summary
    summary_txt_out = results_dir / 'summary.txt'
    sage_sum_final = summary_df[summary_df['architecture'] == 'graphsage'] \
        if not summary_df.empty else pd.DataFrame()
    gcn_sum_final = summary_df[summary_df['architecture'] == 'gcn_gat'] \
        if not summary_df.empty else pd.DataFrame()

    sage_only, gcn_only, both, sign_flip = write_summary_txt(
        sage_sum_final, gcn_sum_final, summary_txt_out
    )
    print(f"wrote text report -> {summary_txt_out}")

    print_terminal_counts(sage_only, gcn_only, both, sign_flip)


if __name__ == '__main__':
    main()
