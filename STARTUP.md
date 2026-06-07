# STARTUP: first-time setup guide

This guide targets a committee member opening GRANITE in GitHub Codespaces for the first
time. It leads to one successful canonical run that produces visible output from committed
data files, with no external data or OSRM routing servers required.

---

## What is available in a fresh clone

A fresh clone contains the granite Python package source, configuration, and two sets of
frozen experiment results:

- `experiments/ablation/00_baseline_current/results/` -- GRANITE vs Dasymetric vs
  Pycnophylactic per-tract and pooled block-group metrics from the m0 n20 SVI parity run.
  These are the current dissertation baselines. The canonical run reads these files.

- `experiments/ablation/00_baseline/results/` -- an earlier frozen artifact from the
  original n20 ablation run. Its `block_group_validation.json` carries keys for IDW and
  kriging, which are retired baselines retained only for reproducibility. Do not use these
  for committee presentation.

Nothing in `data/raw/`, `data/processed/`, `granite_cache/`, or `output/` is committed.
The main `granite` CLI requires all of those and is covered only in the optional section.

---

## 1. Open in GitHub Codespaces

From the repository page on GitHub, click **Code > Codespaces > Create codespace on main**.

The devcontainer runs two commands automatically:

- `postCreateCommand` (on creation): installs all Python dependencies and the granite
  package. This is the slow step on first creation; see step 2 for timing details.

- `postStartCommand` (on every start): attempts to start OSRM routing servers. If OSRM
  data files are absent a postStart error is printed. That error is expected and does not
  affect the canonical run.

After the Codespace finishes creating, proceed to step 3 to verify the install.

---

## 2. Install dependencies (fallback, if postCreateCommand was skipped)

If you opened the repo by cloning rather than through Codespaces, or if the postCreate
step did not run, install manually from the repo root:

```bash
pip install -r requirements.txt
pip install -e .
```

**Timing note (measured in this environment):** A clean virtualenv install with a
semi-warm pip cache (some packages cached, some downloaded fresh) completed in
**1 minute 7 seconds**. A fully cold install with no prior pip cache will be longer;
a fully warm cache (packages already on disk) completes in under 10 seconds.

`requirements.txt` pins `torch==2.10.0+cpu` from the PyTorch CPU wheel index. No
nvidia or cuda runtime libraries are pulled. No compiler step is required; pip installs
pre-built wheels throughout. Approximately 80 packages are installed including scipy,
geopandas, libpysal, and torch-geometric.

---

## 3. Verify the package imports

```bash
python -c "from granite.models.gnn import AccessibilitySVIGNN; print('ok')"
```

Expected output:

```
ok
```

If this fails, see Troubleshooting below.

---

## 4. Run the canonical command

```bash
python experiments/ablation/00_baseline_current/regen_figures.py
```

This reads two committed result files (GRANITE, Dasymetric, and Pycnophylactic metrics
from the m0 n20 SVI parity run) and writes two PNG figures.

**Expected output (pasted from a verified run in a clean virtualenv):**

```
loaded per_tract.csv: 60 rows
loaded aggregate.csv: 3 rows
methods: ['GRANITE', 'Dasymetric', 'Pycnophylactic']

--- bg_r_by_tract ---
written: /workspaces/GRANITE/experiments/ablation/00_baseline_current/figures/bg_r_by_tract.png

--- aggregate_summary ---
written: /workspaces/GRANITE/experiments/ablation/00_baseline_current/figures/aggregate_summary.png

done.
```

**Runtime:** approximately 1-2 seconds total (measured 1.2 seconds in a clean virtualenv
with the CPU torch build and warm pip cache).

---

## 5. Where the output lands

The script resolves output paths from its own location regardless of working directory.
Both figures are written to:

```
experiments/ablation/00_baseline_current/figures/bg_r_by_tract.png
experiments/ablation/00_baseline_current/figures/aggregate_summary.png
```

Open them from the VS Code file explorer (they will open inline).

- `bg_r_by_tract.png`: per-tract block-group Pearson r for GRANITE, Dasymetric, and
  Pycnophylactic across 19 tracts with at least 2 block groups.

- `aggregate_summary.png`: pooled block-group Pearson r with 95% CI for all three methods
  (n=69 block groups). Pooled values: GRANITE 0.769, Dasymetric 0.802,
  Pycnophylactic 0.768.

---

## Data dependencies of the canonical run

| File | Committed | Size |
|------|-----------|------|
| `experiments/ablation/00_baseline_current/results/per_tract.csv` | yes | 5 KB |
| `experiments/ablation/00_baseline_current/results/aggregate.csv` | yes | 292 bytes |

Source: `data/results/m0_n20_svi_parity/` (m0 n20 SVI parity run, not committed due to
size limits, but the summary CSVs above are committed). No external downloads, no OSRM,
no access to `data/raw/` required.

**Note on the 00_baseline artifact:** `experiments/ablation/00_baseline/results/` contains
an earlier run whose `block_group_validation.json` includes IDW and kriging keys. Those
are retired baselines kept for reproducibility only. The current dissertation baselines are
Dasymetric (headline) and Pycnophylactic (secondary), shown in the canonical run above.

---

## 6. Troubleshooting

**ModuleNotFoundError: No module named 'granite'**

The package is not installed. Run:

```bash
pip install -e .
```

Then retry step 3.

**ModuleNotFoundError: No module named 'torch'** or **'torch_geometric'**

torch or torch-geometric did not install. Run:

```bash
pip install -r requirements.txt
```

`requirements.txt` already targets the CPU-only torch wheel, so no separate CPU override
is needed. If torch-geometric still fails independently, run `pip install torch-geometric`
after torch is installed.

**FileNotFoundError for per_tract.csv or aggregate.csv**

The committed result files are missing. Verify the clone is complete:

```bash
git ls-files experiments/ablation/00_baseline_current/results/
```

Expected: two lines, `aggregate.csv` and `per_tract.csv`. If missing, re-clone or run:

```bash
git checkout HEAD -- experiments/ablation/00_baseline_current/
```

---

## Optional: OSRM standup and full pipeline run

**This section is unverified in a fresh-clone environment.** The steps below require data
files that are not committed to the repository.

### What the full pipeline requires

The `granite` CLI needs:

- `data/raw/chattanooga.geojson` (28 MB, address points)
- `data/raw/SVI_2020_US.csv` (58 MB, CDC SVI)
- `data/raw/tl_2020_47_tract.*` (24 MB, census tract shapefiles)
- `data/raw/tl_2020_47_bg.*` (39 MB, census block group shapefiles)
- `data/raw/address_features/combined_address_features.csv` (address-level features)
- OSRM routing servers on localhost:5000 (driving) and localhost:5001 (walking), or
  a populated `granite_cache/` directory (976 MB) that bypasses live routing

These files are gitignored due to size and licensing. Contact the repository owner for
access to the data bundle or the cache snapshot.

### Starting OSRM servers (requires data/raw/osrm/ files)

```bash
bash granite/scripts/start_osrm.sh
```

The script checks for `data/raw/osrm/tennessee-car.osrm` and
`data/raw/osrm/tennessee-foot.osrm`. Both must be present and fully processed before the
script will start the Docker containers.

### Running the full pipeline (unverified)

```bash
granite --fips 47065000600 --epochs 50 --verbose
```

This runs a single-tract analysis. With a cold OSRM cache, runtime is approximately 76
minutes; with a populated cache, under 5 minutes. Output is written to `output/`.
