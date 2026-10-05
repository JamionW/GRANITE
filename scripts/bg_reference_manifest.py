#!/usr/bin/env python3
"""
block-group SVI reference manifest.

records provenance for the national ACS/SVI reference used as validation
ground truth (A5): ACS vintage and source URL template, row counts at every
completeness definition, sha256 of both national CSVs (both gitignored,
never committed -- see docs/paper1/number_trace.md), the Hamilton County
(GEOID prefix 47065) coverage counts behind the 69-BG pooled m0 figure, the
EP_* variable direction table, and the tract-level SVI vintage used as
training supervision (a separate file from the BG validation reference).

read-only against data/processed/*.csv and granite/data/block_group_loader.py;
writes only data/results/bg_reference_manifest/manifest.json.
"""
import hashlib
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = REPO_ROOT / "data" / "processed" / "national_bg_acs_raw.csv"
SVI_PATH = REPO_ROOT / "data" / "processed" / "national_bg_svi.csv"
TRACT_SVI_PATH = REPO_ROOT / "data" / "raw" / "SVI_2020_US.csv"
OUTPUT_PATH = REPO_ROOT / "data" / "results" / "bg_reference_manifest" / "manifest.json"

# EP_* variable direction, transcribed from granite/data/block_group_loader.py:424-445.
# "ascending" means higher raw value -> higher percentile (more vulnerable);
# "inverted" (ascending=False) means lower raw value -> higher percentile.
EP_DIRECTION = {
    "EP_MHI":    {"theme": "theme1_socioeconomic", "direction": "inverted",  "line": "block_group_loader.py:441-442"},
    "EP_PCI":    {"theme": "theme1_socioeconomic", "direction": "inverted",  "line": "block_group_loader.py:441-442"},
    "EP_UNEMP":  {"theme": "theme1_socioeconomic", "direction": "ascending", "line": "block_group_loader.py:443-445"},
    "EP_NOHSDP": {"theme": "theme1_socioeconomic", "direction": "ascending", "line": "block_group_loader.py:443-445"},
    "EP_AGE65":  {"theme": "theme2_household",     "direction": "ascending", "line": "block_group_loader.py:443-445"},
    "EP_AGE17":  {"theme": "theme2_household",     "direction": "ascending", "line": "block_group_loader.py:443-445"},
    "EP_SNGPNT": {"theme": "theme2_household",     "direction": "ascending", "line": "block_group_loader.py:443-445"},
    "EP_MINRTY": {"theme": "theme3_minority",       "direction": "ascending", "line": "block_group_loader.py:443-445"},
    "EP_MUNIT":  {"theme": "theme4_housing",        "direction": "ascending", "line": "block_group_loader.py:443-445"},
    "EP_MOBILE": {"theme": "theme4_housing",        "direction": "ascending", "line": "block_group_loader.py:443-445"},
    "EP_CROWD":  {"theme": "theme4_housing",        "direction": "ascending", "line": "block_group_loader.py:443-445"},
    "EP_NOVEH":  {"theme": "theme4_housing",        "direction": "ascending", "line": "block_group_loader.py:443-445"},
}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    svi = pd.read_csv(SVI_PATH, dtype={"GEOID": str})
    raw_rows = sum(1 for _ in open(RAW_PATH)) - 1  # minus header

    svi_non_null = int(svi["SVI"].notna().sum())
    svi_complete_true = int((svi["svi_complete"] == True).sum())
    complete_flag_but_svi_nan = int(((svi["svi_complete"] == True) & svi["SVI"].isna()).sum())

    hamilton = svi[svi["GEOID"].str.startswith("47065")]
    hamilton_total = int(len(hamilton))
    hamilton_svi_non_null = int(hamilton["SVI"].notna().sum())
    hamilton_complete_true = int((hamilton["svi_complete"] == True).sum())
    hamilton_complete_but_nan = int(
        ((hamilton["svi_complete"] == True) & hamilton["SVI"].isna()).sum()
    )

    manifest = {
        "acs_vintage": {
            "year": 2020,
            "estimate_type": "5-year",
            "source": "block_group_loader.py:28 comment, fetch_national_acs_data default year=2020 (block_group_loader.py:491)",
        },
        "acs_source_url_template": "https://api.census.gov/data/{year}/acs/acs5",
        "url_template_citation": "block_group_loader.py:237-238 (_fetch_raw_acs_data), reused at block_group_loader.py:516-517 (fetch_national_acs_data)",
        "row_counts": {
            "fetched_raw": raw_rows,
            "svi_non_null_ranked": svi_non_null,
            "svi_complete_flag_true": svi_complete_true,
            "svi_complete_flag_true_but_svi_nan": complete_flag_but_svi_nan,
        },
        "svi_complete_flag_caveat": (
            "svi_complete (block_group_loader.py:474-475) checks only 4 of the 12 EP_* "
            "variables (EP_MHI, EP_UNEMP, EP_NOHSDP, EP_NOVEH). themes_sum uses "
            "skipna=False across all 4 themes (block_group_loader.py:462), so any BG "
            "missing one of the other 8 variables still gets SVI=NaN despite "
            "svi_complete=True. This is known and unpatched; see README.md."
        ),
        "national_csv_sha256": {
            "data/processed/national_bg_acs_raw.csv": sha256_of(RAW_PATH),
            "data/processed/national_bg_svi.csv": sha256_of(SVI_PATH),
        },
        "national_csv_git_status": "both gitignored, never committed (see docs/paper1/number_trace.md A5)",
        "hamilton_county_c2": {
            "geoid_prefix_filter": "47065",
            "total_bgs": hamilton_total,
            "svi_non_null": hamilton_svi_non_null,
            "svi_complete_flag_true": hamilton_complete_true,
            "svi_complete_flag_true_but_svi_nan": hamilton_complete_but_nan,
            "m0_pooled_bg_filter": {
                "citation": "granite/scripts/run_m0_parity.py:76-95 (_aggregate_to_bg, _bg_metrics)",
                "filter_column": "n_addresses (aggregated address count per BG from the n20 subset), threshold MIN_ADDRESSES_PER_BG_POOLED=10 (run_m0_parity.py:40,196)",
                "svi_null_guard": "_bg_metrics drops SVI-null rows before merge: svi_lookup = bg_gdf[['GEOID','SVI']].dropna(subset=['SVI']) (run_m0_parity.py:89)",
                "pooled_n_bgs": 69,
                "all_69_have_non_null_svi": True,
            },
        },
        "ep_variable_direction_c3": EP_DIRECTION,
        "tract_level_svi_supervision_c4": {
            "role": "training/validation supervision target (RPL_THEMES), distinct from the BG-level ACS-derived SVI reference above",
            "vintage": "CDC SVI 2020",
            "source_path": "data/raw/SVI_2020_US.csv",
            "column": "RPL_THEMES",
            "load_citation": "granite/data/loaders.py:93 (load_svi_data), used at loaders.py:119,123,138,202,251-256,313,816,822",
            "git_status": "gitignored, never committed (data/raw/ not tracked per CLAUDE.md)",
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {OUTPUT_PATH}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
