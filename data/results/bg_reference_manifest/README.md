# BG reference manifest (TRACE-P1-CLOSE B3)

`manifest.json` is produced by `scripts/bg_reference_manifest.py` (read-only against
`data/processed/national_bg_acs_raw.csv`, `data/processed/national_bg_svi.csv`, and
`data/raw/SVI_2020_US.csv`; writes only this directory).

## Known, unpatched: `svi_complete` is a 4-of-12-variable flag

`granite/data/block_group_loader.py:474-475` defines:

```python
core_vars = ['EP_MHI', 'EP_UNEMP', 'EP_NOHSDP', 'EP_NOVEH']
result['svi_complete'] = ~result[core_vars].isna().any(axis=1)
```

This checks only 4 of the 12 `EP_*` variables that actually feed the SVI computation. The real
completeness requirement is all 12, because `themes_sum` (`block_group_loader.py:462`) and each
theme's own sum (`block_group_loader.py:451`) use `skipna=False`: a block group missing any one
of the 8 variables *not* in `core_vars` (`EP_PCI`, `EP_AGE65`, `EP_AGE17`, `EP_SNGPNT`,
`EP_MINRTY`, `EP_MUNIT`, `EP_MOBILE`, `EP_CROWD`) still gets `SVI = NaN`, even though
`svi_complete` reports `True`.

Measured effect, nationally: `svi_complete=True` for 239,346 block groups, but only 233,338 of
those actually carry a non-null `SVI` -- 6,008 block groups pass the 4-variable flag and still
end up with `NaN` SVI. The Hamilton County subset (`manifest.json: hamilton_county_c2`) shows the
same pattern at smaller scale (`svi_complete_flag_true_but_svi_nan`).

This is flagged here as a known discrepancy in the completeness flag's definition, not fixed.
`granite/data/block_group_loader.py` is out of scope for this milestone (TRACE-P1-CLOSE
instructions: "Do not modify granite/data/block_group_loader.py"). Any Paper 1 or downstream code
that needs a "will this BG have a usable SVI" predicate should check `SVI.notna()` directly
rather than `svi_complete`.

## Why this exists

TRACE-P1 (`docs/paper1/number_trace.md`) found `data/processed/national_bg_svi.csv` and
`national_bg_acs_raw.csv` both gitignored and never committed, so the national row counts (A5)
were UNTRACKED-ONLY. This manifest does not change that status -- the two source CSVs remain
gitignored -- but it anchors every number cited in the A5 re-verdict to a sha256 of the exact
file on disk, a committed script, and a committed JSON artifact, so a future session can detect
if the underlying (still-untracked) CSVs ever change.
