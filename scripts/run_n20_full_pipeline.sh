#!/usr/bin/env bash
# Full n=20 rank-consistency pipeline: training, analysis, figures, report.
# Designed to run autonomously without user interaction.
# Usage: bash scripts/run_n20_full_pipeline.sh 2>&1 | tee /tmp/n20_pipeline.log

set -euo pipefail
cd /workspaces/GRANITE

# ---------------------------------------------------------------------------
# keepalive: echo every 4 minutes so codespaces does not idle-disconnect
# ---------------------------------------------------------------------------
keepalive() {
    while true; do
        sleep 240
        echo "[keepalive] $(date '+%H:%M:%S') still running..."
    done
}
keepalive &
KEEPALIVE_PID=$!
trap "kill $KEEPALIVE_PID 2>/dev/null || true" EXIT

log() {
    echo "[$(date '+%H:%M:%S')] $*"
}

log "====================================================================="
log "RANK CONSISTENCY n=20 FULL PIPELINE"
log "====================================================================="

# ---------------------------------------------------------------------------
# step 1 / step 2: training for missing tracts
# run_new_tracts_n20.py prints chosen 12 tracts and skips existing outputs
# ---------------------------------------------------------------------------
log ""
log "STEP 1+2: training -- 12 new tracts (skips already-cached tracts)"
log "---------------------------------------------------------------------"
python scripts/run_new_tracts_n20.py
log "training complete"

# ---------------------------------------------------------------------------
# step 3: verification
# ---------------------------------------------------------------------------
log ""
log "STEP 3: verification -- all 20 tracts, 73 columns"
log "---------------------------------------------------------------------"
python - <<'PYEOF'
import sys
from pathlib import Path
import pandas as pd

BASE = Path('output/rank_consistency_run')
ARCHS = ['graphsage', 'gcn_gat']
all_ok = True

for arch in ARCHS:
    arch_path = BASE / arch
    dirs = sorted(arch_path.glob('tract_*'))
    print(f"{arch}: {len(dirs)} tract dir(s)")
    for td in dirs:
        fips = td.name.replace('tract_', '')
        feat = td / 'accessibility_features.csv'
        pred = td / 'granite_predictions.csv'
        if not feat.exists() or not pred.exists():
            print(f"  MISSING {fips}: feat={feat.exists()} pred={pred.exists()}")
            all_ok = False
            continue
        try:
            cols = pd.read_csv(feat, nrows=0).columns.tolist()
            n_cols = len(cols)
            numeric_hdr = all(c.isdigit() for c in cols[:5])
            pred_cols = pd.read_csv(pred, nrows=0).columns.tolist()
            has_raw = 'raw_prediction' in pred_cols or 'mean' in pred_cols
            if n_cols == 73 and not numeric_hdr and has_raw:
                print(f"  OK {fips}: cols={n_cols} has_raw={has_raw}")
            else:
                print(f"  WARN {fips}: cols={n_cols} numeric_hdr={numeric_hdr} has_raw={has_raw}")
                all_ok = False
        except Exception as e:
            print(f"  ERROR {fips}: {e}")
            all_ok = False

if not all_ok:
    print("\nVERIFICATION FAILED -- check output above")
    sys.exit(1)
else:
    print("\nverification: all tracts OK")
PYEOF
log "verification passed"

# ---------------------------------------------------------------------------
# step 4: rank-consistency analysis
# ---------------------------------------------------------------------------
log ""
log "STEP 4: rank-consistency analysis"
log "---------------------------------------------------------------------"

log "  primary run (cv=0.10, min-tracts=12)"
python scripts/within_tract_rank_consistency.py \
    --base-dir output/rank_consistency_run \
    --cv-threshold 0.10 \
    --min-tracts 12 \
    --min-addresses 50 \
    --results-dir results/rank_consistency_n20 2>&1 \
    | tee /tmp/rank_consistency_n20.log
log "  primary run complete"

log "  sensitivity run 1 (cv=0.20, min-tracts=10)"
python scripts/within_tract_rank_consistency.py \
    --base-dir output/rank_consistency_run \
    --cv-threshold 0.20 \
    --min-tracts 10 \
    --min-addresses 50 \
    --results-dir results/rank_consistency_n20_sens1
log "  sensitivity run 1 complete"

log "  sensitivity run 2 (cv=0.30, min-tracts=8)"
python scripts/within_tract_rank_consistency.py \
    --base-dir output/rank_consistency_run \
    --cv-threshold 0.30 \
    --min-tracts 8 \
    --min-addresses 50 \
    --results-dir results/rank_consistency_n20_sens2
log "  sensitivity run 2 complete"

# ---------------------------------------------------------------------------
# step 5: figures
# ---------------------------------------------------------------------------
log ""
log "STEP 5: visualizations"
log "---------------------------------------------------------------------"
python scripts/make_rank_consistency_figures_n20.py \
    --results-dir results/rank_consistency_n20 \
    --base-dir output/rank_consistency_run \
    --sens1-dir results/rank_consistency_n20_sens1 \
    --sens2-dir results/rank_consistency_n20_sens2
log "figures complete"

# ---------------------------------------------------------------------------
# step 7: report
# ---------------------------------------------------------------------------
log ""
log "====================================================================="
log "STEP 7: FINAL REPORT"
log "====================================================================="

python - <<'PYEOF'
from pathlib import Path
import re

# section counts from primary summary
summary_path = Path('results/rank_consistency_n20/summary.txt')
if summary_path.exists():
    text = summary_path.read_text()
    print("=" * 72)
    print("PRIMARY RUN SECTION COUNTS")
    print("=" * 72)
    for line in text.splitlines():
        if line.startswith('section '):
            print(" ", line)

    # section D verbatim
    print()
    print("=" * 72)
    print("SECTION D (verbatim)")
    print("=" * 72)
    in_d = False
    for line in text.splitlines():
        if 'SECTION D' in line:
            in_d = True
        if in_d:
            print(line)
            if line.startswith('=') and in_d and 'SECTION D' not in line:
                break
else:
    print("ERROR: summary.txt not found")

# sensitivity grid text table
print()
print("=" * 72)
print("SENSITIVITY GRID")
print("=" * 72)

runs = [
    ('primary (cv=0.10, min_t=12)', 'results/rank_consistency_n20/summary.txt'),
    ('sens1   (cv=0.20, min_t=10)', 'results/rank_consistency_n20_sens1/summary.txt'),
    ('sens2   (cv=0.30, min_t=8) ', 'results/rank_consistency_n20_sens2/summary.txt'),
]

def parse_counts(path):
    counts = {'A': '-', 'B': '-', 'C': '-', 'D': '-'}
    try:
        for line in Path(path).read_text().splitlines():
            for sec in 'ABCD':
                if line.startswith(f'section {sec}'):
                    m = re.search(r'\d+$', line.strip())
                    if m:
                        counts[sec] = m.group()
    except FileNotFoundError:
        pass
    return counts

hdr = f"{'run':<35}  {'A':>5}  {'B':>5}  {'C':>5}  {'D':>5}"
print(hdr)
print("-" * len(hdr))
for label, path in runs:
    c = parse_counts(path)
    print(f"{label:<35}  {c['A']:>5}  {c['B']:>5}  {c['C']:>5}  {c['D']:>5}")

# figure paths
print()
print("=" * 72)
print("GENERATED FIGURES")
print("=" * 72)
fig_dir = Path('results/rank_consistency_n20/figures')
if fig_dir.exists():
    for p in sorted(fig_dir.glob('*.png')):
        print(f"  {p.absolute()}")
else:
    print("  (figures directory not found)")
PYEOF

log "pipeline complete"
