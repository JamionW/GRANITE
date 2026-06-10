"""
Delinquency convergent validity analysis for GRANITE per-address predictions.

Pre-registered hypothesis (recorded before results):
  Higher within-tract allocated SVI -> more delinquency-prone addresses.
  Expected sign: POSITIVE partial Spearman of (method_svi, n_delinq_years)
  controlling log_appvalue, per tract.

Rationale: SVI captures socioeconomic vulnerability; delinquency is an
independent distress proxy. If the GNN allocates SVI meaningfully
within tracts, higher-SVI addresses should be more delinquency-prone
after controlling for property value.

Escrow ceiling: mortgaged owner-occupied homes pay via escrow and
under-report delinquency. This attenuates the test toward null; a positive
result is therefore conservative. A null cannot separate true non-separation
from proxy attenuation.

Power limit: n=16 primary tracts. Between-method comparison is underpowered.
The primary (vs-zero) test is load-bearing; between-method is secondary.
"""

import json
import os
import sys
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
REPO = "/workspaces/GRANITE"
PRED_DIR = os.path.join(REPO, "experiments/recovery/per_address_predictions")
N20_CSV = os.path.join(REPO, "experiments/ecological_fallacy/n20_feature_matrix.csv")
DELQ_CSV = os.path.join(REPO, "data/raw/CTRUDELQCSV.csv")
PARCELS_SHP = os.path.join(REPO, "data/raw/parcels/parcels.shp")
OUT_DIR = os.path.join(REPO, "experiments/recovery/delinquency_convergent_validity")

# 16 primary (non-sparse) and 4 sparse tracts
PRIMARY_TRACTS = [
    "47065000600", "47065000700", "47065001200", "47065001800", "47065001900",
    "47065002400", "47065003400", "47065010431", "47065010433", "47065010435",
    "47065011311", "47065011321", "47065011326", "47065011402", "47065011413",
    "47065011444",
]
SPARSE_TRACTS = [
    "47065011324", "47065011900", "47065011325", "47065011447",
]
ALL_N20 = PRIMARY_TRACTS + SPARSE_TRACTS


def make_parcel_key(m, g, p):
    """build MAP|GROUP_|PARCEL key; blank group -> empty string."""
    m = str(m).strip().upper() if pd.notna(m) else ""
    g = str(g).strip().upper() if pd.notna(g) and str(g).strip() != "" else ""
    p = str(p).strip().upper() if pd.notna(p) else ""
    return f"{m}|{g}|{p}"


def partial_spearman(x, y, z):
    """partial Spearman r_xy.z controlling for z.

    formula: r_xy.z = (r_xy - r_xz*r_yz) / sqrt((1-r_xz^2)(1-r_yz^2))

    returns NaN if either denominator factor is zero or if x is constant.
    """
    if x.std() == 0 or y.std() == 0 or z.std() == 0:
        return np.nan
    r_xy, _ = stats.spearmanr(x, y)
    r_xz, _ = stats.spearmanr(x, z)
    r_yz, _ = stats.spearmanr(y, z)
    denom = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
    if denom == 0:
        return np.nan
    return (r_xy - r_xz * r_yz) / denom


def wilcoxon_vs_zero(arr, label):
    """one-sample wilcoxon signed-rank vs 0; return dict."""
    arr = np.array([v for v in arr if not np.isnan(v)])
    n = len(arr)
    if n < 2:
        return {"label": label, "n": n, "statistic": np.nan, "p": np.nan, "median": np.nan}
    stat, p = stats.wilcoxon(arr, alternative="greater")
    return {
        "label": label,
        "n": n,
        "statistic": float(stat),
        "p": float(p),
        "median": float(np.median(arr)),
    }


def wilcoxon_paired(a, b, label_a, label_b):
    """paired wilcoxon of a vs b (a > b alternative); return dict."""
    pairs = [(x, y) for x, y in zip(a, b) if not np.isnan(x) and not np.isnan(y)]
    if len(pairs) < 2:
        return {
            "label": f"{label_a} vs {label_b}",
            "n_pairs": len(pairs),
            "statistic": np.nan,
            "p": np.nan,
        }
    aa = np.array([p[0] for p in pairs])
    bb = np.array([p[1] for p in pairs])
    diff = aa - bb
    if np.all(diff == 0):
        return {
            "label": f"{label_a} vs {label_b}",
            "n_pairs": len(pairs),
            "statistic": np.nan,
            "p": np.nan,
        }
    stat, p = stats.wilcoxon(diff, alternative="greater")
    return {
        "label": f"{label_a} vs {label_b}",
        "n_pairs": len(pairs),
        "statistic": float(stat),
        "p": float(p),
    }


# ---------------------------------------------------------------------------
# 1. load predictions and assert index alignment
# ---------------------------------------------------------------------------
print("loading predictions...")
df_granite = pd.read_parquet(os.path.join(PRED_DIR, "granite_m0.parquet"))
df_dasy = pd.read_parquet(os.path.join(PRED_DIR, "dasymetric.parquet"))
df_pycno = pd.read_parquet(os.path.join(PRED_DIR, "pycnophylactic.parquet"))

# assert identical (fips, address_idx) sequence
assert (df_granite["fips"].values == df_dasy["fips"].values).all(), "fips mismatch granite/dasy"
assert (df_granite["fips"].values == df_pycno["fips"].values).all(), "fips mismatch granite/pycno"
assert (df_granite["address_idx"].values == df_dasy["address_idx"].values).all(), "address_idx mismatch granite/dasy"
assert (df_granite["address_idx"].values == df_pycno["address_idx"].values).all(), "address_idx mismatch granite/pycno"
print(f"  index alignment confirmed: {len(df_granite):,} rows, identical (fips, address_idx) sequence across all three files")

# ---------------------------------------------------------------------------
# 2. load n20 feature matrix
# ---------------------------------------------------------------------------
print("loading n20 feature matrix...")
n20 = pd.read_csv(N20_CSV)
n20["fips"] = n20["fips"].astype(str)

# assert alignment with predictions
assert len(n20) == len(df_granite), f"row count mismatch: n20={len(n20)} predictions={len(df_granite)}"
assert (n20["fips"].values == df_granite["fips"].values).all(), "fips mismatch n20/predictions"
assert (n20["address_idx"].values == df_granite["address_idx"].values).all(), "address_idx mismatch n20/predictions"
print(f"  n20 alignment confirmed: {len(n20):,} rows")

# building features present check
building_cols = ["build_to_land_ratio", "log_acres", "log_bldg_footprint_m2"]
for c in building_cols:
    if c not in n20.columns:
        print(f"  WARNING: {c} not in n20 -- will be NaN in completeness table")

# ---------------------------------------------------------------------------
# 3. assemble delinquency proxy via parcels.shp sjoin
# ---------------------------------------------------------------------------
print("building delinquency join...")

# load parcels
print("  loading parcels shapefile...")
parcels = gpd.read_file(PARCELS_SHP)
print(f"  parcels loaded: {len(parcels):,} rows, crs={parcels.crs}")

# build parcel key on parcels
parcels["parcel_key"] = parcels.apply(
    lambda r: make_parcel_key(r["MAP"], r["GROUP_"], r["PARCEL"]), axis=1
)

# load delinquency CSV
print("  loading delinquency CSV...")
delq_raw = pd.read_csv(DELQ_CSV, dtype={"Bill Year": str})
# filter bill year [2000, 2026]
delq_raw["bill_year_int"] = pd.to_numeric(delq_raw["Bill Year"], errors="coerce")
delq_filt = delq_raw[delq_raw["bill_year_int"].between(2000, 2026)].copy()
print(f"  delinquency records after year filter: {len(delq_filt):,} (raw: {len(delq_raw):,})")

# build parcel key on delinquency
delq_filt["parcel_key"] = delq_filt.apply(
    lambda r: make_parcel_key(r["Map"], r["Group"], r["Parcel"]), axis=1
)

# aggregate: n_delinq_years = count of distinct bill years per parcel_key
delq_agg = (
    delq_filt.groupby("parcel_key")["bill_year_int"]
    .nunique()
    .reset_index()
    .rename(columns={"bill_year_int": "n_delinq_years"})
)
print(f"  unique delinquent parcels: {len(delq_agg):,}")

# spatial join: n20 addresses -> parcels (within)
print("  spatial join n20 addresses -> parcels...")
n20_gdf = gpd.GeoDataFrame(
    n20[["fips", "address_idx", "lat", "lon", "log_appvalue"]].copy(),
    geometry=gpd.points_from_xy(n20["lon"], n20["lat"]),
    crs="EPSG:4326",
).to_crs(parcels.crs)

# keep only parcel polygon index + parcel_key
parcels_slim = parcels[["parcel_key", "geometry"]].copy()

joined = gpd.sjoin(n20_gdf, parcels_slim, how="left", predicate="within")
# if multiple matches take first
joined = joined[~joined.index.duplicated(keep="first")]
print(f"  sjoin match rate: {joined['parcel_key'].notna().sum():,}/{len(joined):,} ({100*joined['parcel_key'].notna().mean():.1f}%)")

# join delinquency counts
joined = joined.merge(delq_agg, on="parcel_key", how="left")
# non-delinquent = 0
joined["n_delinq_years"] = joined["n_delinq_years"].fillna(0).astype(int)
joined["binary_delinq"] = (joined["n_delinq_years"] > 0).astype(int)

print(f"  addresses with any delinquency: {(joined['n_delinq_years']>0).sum():,} ({100*(joined['n_delinq_years']>0).mean():.1f}%)")

# ---------------------------------------------------------------------------
# 4. assemble analysis frame
# ---------------------------------------------------------------------------
print("assembling analysis frame...")
frame = n20[["fips", "address_idx", "log_appvalue"]].copy()

# add building features for completeness check
for c in building_cols:
    if c in n20.columns:
        frame[c] = n20[c].values

# add predictions (index-aligned)
frame["svi_granite"] = df_granite["svi_pred"].values
frame["svi_dasy"] = df_dasy["svi_pred"].values
frame["svi_pycno"] = df_pycno["svi_pred"].values

# add delinquency (must re-align on original index after sjoin)
# joined preserves original DataFrame index from n20_gdf (0..N-1)
delq_series = joined["n_delinq_years"].values
binary_series = joined["binary_delinq"].values
frame["n_delinq_years"] = delq_series
frame["binary_delinq"] = binary_series

# final alignment check
assert len(frame) == 39535, f"unexpected frame length: {len(frame)}"
print(f"  analysis frame: {len(frame):,} rows")

# ---------------------------------------------------------------------------
# 5. per-tract partial Spearman (primary: 16 non-sparse tracts)
# ---------------------------------------------------------------------------
print("\ncomputing per-tract partial Spearman...")

methods = [
    ("granite", "svi_granite"),
    ("dasymetric", "svi_dasy"),
    ("pycnophylactic", "svi_pycno"),
]

records = []
for tract in ALL_N20:
    sub = frame[frame["fips"] == tract].copy()
    is_primary = tract in PRIMARY_TRACTS
    row = {"fips": tract, "n_addresses": len(sub), "is_primary": is_primary}
    for mname, mcol in methods:
        rho = partial_spearman(sub[mcol], sub["n_delinq_years"], sub["log_appvalue"])
        row[f"partial_rho_{mname}"] = rho
    records.append(row)

df_per_tract = pd.DataFrame(records)
print(df_per_tract[["fips", "n_addresses", "partial_rho_granite", "partial_rho_dasymetric", "partial_rho_pycnophylactic"]].to_string(index=False))

# ---------------------------------------------------------------------------
# 6. binary robustness
# ---------------------------------------------------------------------------
print("\nbinary robustness (binary_delinq)...")
binary_records = []
for tract in ALL_N20:
    sub = frame[frame["fips"] == tract].copy()
    is_primary = tract in PRIMARY_TRACTS
    row = {"fips": tract, "is_primary": is_primary}
    for mname, mcol in methods:
        rho = partial_spearman(sub[mcol], sub["binary_delinq"], sub["log_appvalue"])
        row[f"binary_rho_{mname}"] = rho
    binary_records.append(row)
df_binary = pd.DataFrame(binary_records)

# ---------------------------------------------------------------------------
# 7. completeness: within-tract Spearman of n_delinq_years vs building features
# ---------------------------------------------------------------------------
print("\ncompleteness check (building features vs delinquency)...")
comp_records = []
for tract in PRIMARY_TRACTS:
    sub = frame[frame["fips"] == tract].copy()
    row = {"fips": tract}
    for c in building_cols:
        if c in sub.columns and sub[c].std() > 0:
            r, _ = stats.spearmanr(sub[c], sub["n_delinq_years"])
        else:
            r = np.nan
        row[f"spearman_{c}"] = r
    comp_records.append(row)
df_comp = pd.DataFrame(comp_records)

for c in building_cols:
    col = f"spearman_{c}"
    if col in df_comp.columns:
        vals = df_comp[col].dropna()
        print(f"  {c}: median={vals.median():.3f}, IQR=[{vals.quantile(0.25):.3f},{vals.quantile(0.75):.3f}], n={len(vals)}")

# ---------------------------------------------------------------------------
# 8. Wilcoxon tests (primary tracts only, n=16)
# ---------------------------------------------------------------------------
print("\nWilcoxon tests (n=16 primary tracts)...")

primary_mask = df_per_tract["is_primary"]
rho_g = df_per_tract.loc[primary_mask, "partial_rho_granite"].values
rho_d = df_per_tract.loc[primary_mask, "partial_rho_dasymetric"].values
rho_p = df_per_tract.loc[primary_mask, "partial_rho_pycnophylactic"].values

w_g0 = wilcoxon_vs_zero(rho_g, "granite vs 0")
w_d0 = wilcoxon_vs_zero(rho_d, "dasymetric vs 0")
w_p0 = wilcoxon_vs_zero(rho_p, "pycnophylactic vs 0")
w_gd = wilcoxon_paired(rho_g, rho_d, "granite", "dasymetric")
w_gp = wilcoxon_paired(rho_g, rho_p, "granite", "pycnophylactic")

wilcoxon_results = [w_g0, w_d0, w_p0, w_gd, w_gp]

print(f"  NOTE: 3 vs-zero + 2 between-method tests (not corrected; flag only)")
for w in wilcoxon_results:
    print(f"  {w['label']}: n={w.get('n', w.get('n_pairs'))}, W={w['statistic']:.3f}, p={w['p']:.4f}")

# ---------------------------------------------------------------------------
# 9. full n20 secondary (all 20 tracts)
# ---------------------------------------------------------------------------
print("\nfull n20 secondary (all 20 tracts)...")
all_mask = pd.Series([True] * len(df_per_tract))
rho_g_all = df_per_tract["partial_rho_granite"].values
rho_d_all = df_per_tract["partial_rho_dasymetric"].values
rho_p_all = df_per_tract["partial_rho_pycnophylactic"].values

w_g0_all = wilcoxon_vs_zero(rho_g_all, "granite vs 0 (all 20)")
w_d0_all = wilcoxon_vs_zero(rho_d_all, "dasymetric vs 0 (all 20)")
w_p0_all = wilcoxon_vs_zero(rho_p_all, "pycnophylactic vs 0 (all 20)")

print(f"  deltas (all-20 median vs primary-16 median):")
for mname, r16, r20 in [
    ("granite", np.nanmedian(rho_g), np.nanmedian(rho_g_all)),
    ("dasymetric", np.nanmedian(rho_d), np.nanmedian(rho_d_all)),
    ("pycnophylactic", np.nanmedian(rho_p), np.nanmedian(rho_p_all)),
]:
    print(f"    {mname}: primary-16={r16:.4f}, all-20={r20:.4f}, delta={r20-r16:+.4f}")

# ---------------------------------------------------------------------------
# 10. summary statistics
# ---------------------------------------------------------------------------
primary_df = df_per_tract[primary_mask]

def summarize(arr, label):
    arr = np.array([v for v in arr if not np.isnan(v)])
    return {
        "method": label,
        "n_tracts": len(arr),
        "median": float(np.median(arr)) if len(arr) else np.nan,
        "q25": float(np.percentile(arr, 25)) if len(arr) else np.nan,
        "q75": float(np.percentile(arr, 75)) if len(arr) else np.nan,
        "n_positive": int((arr > 0).sum()),
        "n_negative": int((arr < 0).sum()),
    }

summaries = [
    summarize(rho_g, "granite"),
    summarize(rho_d, "dasymetric"),
    summarize(rho_p, "pycnophylactic"),
]

print("\nmedian partial rho (primary 16 tracts):")
for s in summaries:
    print(f"  {s['method']}: median={s['median']:.4f}, IQR=[{s['q25']:.4f},{s['q75']:.4f}], +:{s['n_positive']} -:{s['n_negative']}")

binary_primary = df_binary[df_binary["is_primary"]]
print("\nbinary robustness median partial rho (primary 16):")
for mname, _ in methods:
    col = f"binary_rho_{mname}"
    vals = binary_primary[col].dropna().values
    print(f"  {mname}: median={np.nanmedian(vals):.4f}, IQR=[{np.nanpercentile(vals,25):.4f},{np.nanpercentile(vals,75):.4f}]")

# ---------------------------------------------------------------------------
# 11. verdict
# ---------------------------------------------------------------------------
g_p = w_g0["p"]
g_med = summaries[0]["median"]
d_p = w_d0["p"]
d_med = summaries[1]["median"]
p_p = w_p0["p"]
p_med = summaries[2]["median"]

def verdict(p_val, med):
    if p_val < 0.05 and med > 0:
        return "supported (p<0.05, positive median)"
    elif p_val < 0.10 and med > 0:
        return "marginal (p<0.10, positive median)"
    elif med > 0:
        return f"null/inconclusive (p={p_val:.3f}, positive median, n=16 power limit)"
    else:
        return f"null (p={p_val:.3f}, negative median)"

print("\nverdicts:")
for label, pv, med in [("granite", g_p, g_med), ("dasymetric", d_p, d_med), ("pycnophylactic", p_p, p_med)]:
    print(f"  {label}: {verdict(pv, med)}")

# ---------------------------------------------------------------------------
# 12. write outputs
# ---------------------------------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)

# per_tract_partial.csv (16 primary tracts)
out_per_tract = primary_df[
    ["fips", "n_addresses", "partial_rho_granite", "partial_rho_dasymetric", "partial_rho_pycnophylactic"]
].copy()
out_per_tract.to_csv(os.path.join(OUT_DIR, "per_tract_partial.csv"), index=False)
print(f"\nwrote per_tract_partial.csv ({len(out_per_tract)} rows)")

# results.json
results = {
    "index_alignment": "confirmed: identical (fips, address_idx) across granite_m0, dasymetric, pycnophylactic, and n20 feature matrix",
    "n_addresses": int(len(frame)),
    "n_primary_tracts": 16,
    "n_sparse_tracts_excluded_primary": 4,
    "pre_registered_direction": "positive partial Spearman (higher svi_pred -> more delinquency-prone)",
    "primary_partial_rho": {
        m["method"]: {
            "median": round(m["median"], 6),
            "q25": round(m["q25"], 6),
            "q75": round(m["q75"], 6),
            "n_positive": m["n_positive"],
            "n_negative": m["n_negative"],
            "n_tracts": m["n_tracts"],
        }
        for m in summaries
    },
    "wilcoxon_vs_zero": [
        {k: (round(v, 6) if isinstance(v, float) else v) for k, v in w.items()}
        for w in [w_g0, w_d0, w_p0]
    ],
    "wilcoxon_paired": [
        {k: (round(v, 6) if isinstance(v, float) else v) for k, v in w.items()}
        for w in [w_gd, w_gp]
    ],
    "binary_robustness_median_partial_rho": {
        mname: round(float(np.nanmedian(binary_primary[f"binary_rho_{mname}"].dropna().values)), 6)
        for mname, _ in methods
    },
    "full_n20_secondary": {
        "n_tracts": 20,
        "median_partial_rho": {
            "granite": round(float(np.nanmedian(rho_g_all)), 6),
            "dasymetric": round(float(np.nanmedian(rho_d_all)), 6),
            "pycnophylactic": round(float(np.nanmedian(rho_p_all)), 6),
        },
        "wilcoxon_vs_zero": [
            {k: (round(v, 6) if isinstance(v, float) else v) for k, v in w.items()}
            for w in [w_g0_all, w_d0_all, w_p0_all]
        ],
    },
    "completeness_building_vs_delinquency": {
        c: {
            "median_spearman": round(float(df_comp[f"spearman_{c}"].dropna().median()), 6),
            "q25": round(float(df_comp[f"spearman_{c}"].dropna().quantile(0.25)), 6),
            "q75": round(float(df_comp[f"spearman_{c}"].dropna().quantile(0.75)), 6),
        }
        for c in building_cols
        if f"spearman_{c}" in df_comp.columns
    },
    "verdicts": {
        "granite": verdict(g_p, g_med),
        "dasymetric": verdict(d_p, d_med),
        "pycnophylactic": verdict(p_p, p_med),
    },
    "caveats": [
        "n=16 primary tracts; between-method comparison underpowered",
        "3 vs-zero + 2 between-method tests; no correction applied; counts flagged",
        "escrow ceiling: mortgaged owner-occupied under-reports delinquency; attenuates toward null",
        "convergent validity against independent distress proxy; not recovery of true address-level SVI",
    ],
}

with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
    json.dump(results, f, indent=2)
print("wrote results.json")

# summary.txt
lines = [
    "DELINQUENCY CONVERGENT VALIDITY -- SUMMARY",
    "=" * 60,
    "",
    "Index alignment: confirmed across all 3 prediction files and n20 matrix.",
    f"Analysis frame: {len(frame):,} addresses, {len(primary_df)} primary tracts, 4 sparse excluded.",
    "",
    "Pre-registered direction: POSITIVE partial Spearman (higher svi_pred -> more delinquency-prone).",
    "",
    "Per-tract partial Spearman rho (n=16 primary tracts):",
    f"  granite:        median={summaries[0]['median']:+.4f}, IQR=[{summaries[0]['q25']:+.4f},{summaries[0]['q75']:+.4f}]  (+{summaries[0]['n_positive']}/-{summaries[0]['n_negative']})",
    f"  dasymetric:     median={summaries[1]['median']:+.4f}, IQR=[{summaries[1]['q25']:+.4f},{summaries[1]['q75']:+.4f}]  (+{summaries[1]['n_positive']}/-{summaries[1]['n_negative']})",
    f"  pycnophylactic: median={summaries[2]['median']:+.4f}, IQR=[{summaries[2]['q25']:+.4f},{summaries[2]['q75']:+.4f}]  (+{summaries[2]['n_positive']}/-{summaries[2]['n_negative']})",
    "",
    "Wilcoxon signed-rank (one-sided, vs 0, n=16):",
    f"  granite:        W={w_g0['statistic']:.1f}, p={w_g0['p']:.4f}",
    f"  dasymetric:     W={w_d0['statistic']:.1f}, p={w_d0['p']:.4f}",
    f"  pycnophylactic: W={w_p0['statistic']:.1f}, p={w_p0['p']:.4f}",
    "",
    "Wilcoxon paired (one-sided, GRANITE > baseline, n=16 pairs):",
    f"  granite vs dasymetric:     W={w_gd['statistic']:.1f}, p={w_gd['p']:.4f}  (n_pairs={w_gd['n_pairs']})",
    f"  granite vs pycnophylactic: W={w_gp['statistic']:.1f}, p={w_gp['p']:.4f}  (n_pairs={w_gp['n_pairs']})",
    "  NOTE: 3 vs-zero + 2 between-method tests counted; no correction applied.",
    "",
    "Binary robustness (binary_delinq, partial Spearman, n=16):",
]

for mname, _ in methods:
    col = f"binary_rho_{mname}"
    vals = binary_primary[col].dropna().values
    lines.append(f"  {mname}: median={np.nanmedian(vals):+.4f}, IQR=[{np.nanpercentile(vals,25):+.4f},{np.nanpercentile(vals,75):+.4f}]")

lines += [
    "",
    "Completeness (building features vs n_delinq_years, n=16):",
]
for c in building_cols:
    col = f"spearman_{c}"
    if col in df_comp.columns:
        vals = df_comp[col].dropna()
        lines.append(f"  {c}: median={vals.median():+.3f}, IQR=[{vals.quantile(0.25):+.3f},{vals.quantile(0.75):+.3f}]")

lines += [
    "",
    "Full n20 secondary (all 20 tracts, sparse included):",
    f"  granite:        median={np.nanmedian(rho_g_all):+.4f}  (delta from primary: {np.nanmedian(rho_g_all)-np.nanmedian(rho_g):+.4f})",
    f"  dasymetric:     median={np.nanmedian(rho_d_all):+.4f}  (delta from primary: {np.nanmedian(rho_d_all)-np.nanmedian(rho_d):+.4f})",
    f"  pycnophylactic: median={np.nanmedian(rho_p_all):+.4f}  (delta from primary: {np.nanmedian(rho_p_all)-np.nanmedian(rho_p):+.4f})",
    "",
    "Verdicts:",
    f"  granite:        {verdict(g_p, g_med)}",
    f"  dasymetric:     {verdict(d_p, d_med)}",
    f"  pycnophylactic: {verdict(p_p, p_med)}",
    "",
    "Interpretation:",
    "  Primary (vs-zero) test is load-bearing for the recovery question.",
    "  Between-method comparison is secondary and underpowered at n=16.",
    "  Escrow ceiling: mortgaged owner-occupied under-reports delinquency; attenuates toward null.",
    "  A positive result (if observed) is conservative given attenuation.",
    "  A null is ambiguous: cannot separate 'GNN allocates noise' from 'proxy too attenuated'.",
    "  This test establishes convergent validity against an independent distress proxy,",
    "  not recovery of true address-level SVI (no such ground truth exists).",
]

with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("wrote summary.txt")
print("\nDONE.")
