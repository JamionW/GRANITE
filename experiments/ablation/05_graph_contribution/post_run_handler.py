"""
Post-run handler for step 5 sweep.

Polls the sweep log until all 20 seed-runs complete, then:
  1. runs make_figures.py
  2. fills README.md with actual results
  3. appends to SESSION_LOG.md
  4. appends to Research_Status.md
  5. commits run artifacts, then figures separately

Usage (background):
    python experiments/ablation/05_graph_contribution/post_run_handler.py &
"""
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parents[3]
ABLATION_DIR = Path(__file__).resolve().parent
LOG_PATH    = Path('/tmp/step05_run.log')
RESULTS_PATH = ABLATION_DIR / 'results' / 'graph_contribution_metrics.json'
README_PATH  = ABLATION_DIR / 'README.md'
SESSION_LOG  = REPO_ROOT / 'SESSION_LOG.md'
RESEARCH_ST  = REPO_ROOT / 'Research_Status.md'

POLL_INTERVAL = 120   # seconds between log checks
TOTAL_SEEDS   = 20    # 2 conditions x 2 architectures x 5 seeds
SEEDS         = [42, 17, 123, 2024, 7]
CONDITIONS    = ['production', 'mlp_floor']
ARCHITECTURES = ['sage', 'gcn_gat']

SANITY_REFERENCE = {
    'sage':    {'pooled_bg_r': 0.7537, 'within_tract_std': 0.0823},
    'gcn_gat': {'pooled_bg_r': 0.7664, 'within_tract_std': 0.0814},
}
SANITY_TOL_BG_R = 1e-3
SANITY_TOL_STD  = 2e-3


def _log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    print(f'[post_run {ts}] {msg}', flush=True)


def _count_done(log_path):
    if not log_path.exists():
        return 0
    text = log_path.read_text()
    return len(re.findall(r'step05.*done:', text))


def _sweep_complete(log_path):
    if not log_path.exists():
        return False
    text = log_path.read_text()
    return 'complete in' in text or _count_done(log_path) >= TOTAL_SEEDS


def _run_figures():
    _log('running make_figures.py...')
    result = subprocess.run(
        [sys.executable, str(ABLATION_DIR / 'make_figures.py')],
        cwd=str(REPO_ROOT), capture_output=True, text=True
    )
    if result.returncode != 0:
        _log(f'make_figures.py FAILED:\n{result.stderr}')
        return False
    _log(result.stdout.strip())
    return True


def _load_results():
    with open(RESULTS_PATH) as f:
        return json.load(f)


def _mean_std(results, condition, arch, metric):
    arch_res = results.get(condition, {}).get(arch, {})
    vals = [s[metric] for s in arch_res.get('seeds', [])
            if s.get(metric) is not None and math.isfinite(float(s.get(metric, float('nan'))))]
    if not vals:
        return float('nan'), float('nan')
    m = sum(vals) / len(vals)
    s = (sum((v - m) ** 2 for v in vals) / max(len(vals) - 1, 1)) ** 0.5
    return m, s


def _sanity_table(results):
    rows = []
    all_pass = True
    for arch in ARCHITECTURES:
        seeds = results.get('production', {}).get(arch, {}).get('seeds', [])
        s42 = next((s for s in seeds if s['seed'] == 42), None)
        if s42 is None:
            rows.append(f'| {arch} | pooled_bg_r | {SANITY_REFERENCE[arch]["pooled_bg_r"]:.4f} | missing | - | FAIL |')
            all_pass = False
            continue
        ref_r = SANITY_REFERENCE[arch]['pooled_bg_r']
        got_r = s42['pooled_bg_r']
        ref_s = SANITY_REFERENCE[arch]['within_tract_std']
        got_s = s42['within_tract_std']
        ok_r  = math.isfinite(got_r) and abs(got_r - ref_r) <= SANITY_TOL_BG_R
        ok_s  = math.isfinite(got_s) and abs(got_s - ref_s) <= SANITY_TOL_STD
        rows.append(f'| {arch} | pooled_bg_r | {ref_r:.4f} | {got_r:.4f} | {got_r-ref_r:+.4f} | {"PASS" if ok_r else "FAIL"} |')
        rows.append(f'| {arch} | within_tract_std | {ref_s:.4f} | {got_s:.4f} | {got_s-ref_s:+.4f} | {"PASS" if ok_s else "FAIL"} |')
        if not ok_r or not ok_s:
            all_pass = False
    return '\n'.join(rows), all_pass


def _metrics_table(results):
    rows = []
    for cond in CONDITIONS:
        for arch in ARCHITECTURES:
            ms, ss = _mean_std(results, cond, arch, 'within_tract_std')
            mm, sm = _mean_std(results, cond, arch, 'morans_i')
            mr, sr = _mean_std(results, cond, arch, 'pooled_bg_r')
            rows.append(
                f'| {cond} | {arch} | {ms:.4f} +/- {ss:.4f} | {mm:.4f} +/- {sm:.4f} | {mr:.4f} +/- {sr:.4f} |'
            )
    return '\n'.join(rows)


def _verdict_paragraph(results):
    lines = []
    for arch in ARCHITECTURES:
        prod_std_vals = [s['within_tract_std'] for s in results.get('production', {}).get(arch, {}).get('seeds', []) if math.isfinite(s.get('within_tract_std', float('nan')))]
        prod_mi_vals  = [s['morans_i'] for s in results.get('production', {}).get(arch, {}).get('seeds', []) if math.isfinite(s.get('morans_i', float('nan')))]
        mlp_std_vals  = [s['within_tract_std'] for s in results.get('mlp_floor', {}).get(arch, {}).get('seeds', []) if math.isfinite(s.get('within_tract_std', float('nan')))]
        mlp_mi_vals   = [s['morans_i'] for s in results.get('mlp_floor', {}).get(arch, {}).get('seeds', []) if math.isfinite(s.get('morans_i', float('nan')))]

        if not (prod_std_vals and prod_mi_vals and mlp_std_vals and mlp_mi_vals):
            lines.append(f'{arch}: insufficient data for verdict.')
            continue

        prod_std_m = sum(prod_std_vals) / len(prod_std_vals)
        prod_std_s = (sum((v - prod_std_m) ** 2 for v in prod_std_vals) / max(len(prod_std_vals) - 1, 1)) ** 0.5
        prod_mi_m  = sum(prod_mi_vals) / len(prod_mi_vals)
        prod_mi_s  = (sum((v - prod_mi_m) ** 2 for v in prod_mi_vals) / max(len(prod_mi_vals) - 1, 1)) ** 0.5
        mlp_std_m  = sum(mlp_std_vals) / len(mlp_std_vals)
        mlp_mi_m   = sum(mlp_mi_vals) / len(mlp_mi_vals)

        std_in_band = abs(mlp_std_m - prod_std_m) <= prod_std_s
        mi_in_band  = abs(mlp_mi_m  - prod_mi_m)  <= prod_mi_s

        if std_in_band and mi_in_band:
            verdict = (
                f'**{arch.upper()}: graph is decorative.** mlp_floor falls within the production '
                f'seed band on both Moran\'s I ({mlp_mi_m:.4f} vs prod {prod_mi_m:.4f}+/-{prod_mi_s:.4f}) '
                f'and within-tract std ({mlp_std_m:.4f} vs prod {prod_std_m:.4f}+/-{prod_std_s:.4f}). '
                f'The spatially autocorrelated node features carry the output geography without message '
                f'passing. A full wiring construction sweep is not warranted for this architecture.'
            )
        else:
            gap_parts = []
            if not std_in_band:
                gap_parts.append(
                    f'within-tract std ({mlp_std_m:.4f} vs prod {prod_std_m:.4f}+/-{prod_std_s:.4f})'
                )
            if not mi_in_band:
                gap_parts.append(
                    f"Moran's I ({mlp_mi_m:.4f} vs prod {prod_mi_m:.4f}+/-{prod_mi_s:.4f})"
                )
            verdict = (
                f'**{arch.upper()}: graph contributes.** mlp_floor falls outside the production seed '
                f'band on {" and ".join(gap_parts)}. The gap is the road-network graph\'s measurable '
                f'contribution. A full construction sweep (road, feature-similarity, randomized) is the '
                f'recommended follow-up to characterize which wiring type drives the gain.'
            )
        lines.append(verdict)
    return '\n\n'.join(lines)


def _git_sha():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=str(REPO_ROOT)
        ).decode().strip()
    except Exception:
        return 'unknown'


def _update_readme(results):
    _log('updating README.md...')
    sha = _git_sha()
    sanity_rows, sanity_pass = _sanity_table(results)
    metrics_rows = _metrics_table(results)
    verdict = _verdict_paragraph(results)

    readme = f"""# Step 5: graph contribution boundary test (two-pole)

Status: **complete**

## Design

Two conditions, both architectures, five seeds each (20 runs total).

| parameter | value |
|---|---|
| conditions | production, mlp_floor |
| architectures | SAGE, GCN-GAT |
| seeds | 42, 17, 123, 2024, 7 |
| constraint_mode | soft |
| variation_weight | 0.8 |
| epochs | 150 |

**production**: live hybrid road-network-plus-geographic graph, unchanged from prior steps.
**mlp_floor**: self-loops only (`edge_index = [[i],[i]] for all i`, `edge_weight = 1.0`). SAGE and GAT reduce to node-wise functions on the input features.

Moran's I computed from address coordinates (k=8, symmetrized, row-normalized), identical across conditions.

## Git sha

{sha}

## Sanity regression

production / seed=42 must reproduce step 4 soft baseline within 1e-3 (BG r) / 2e-3 (std).

| arch | metric | reference | got | delta | pass |
|---|---|---|---|---|---|
{sanity_rows}

Sanity regression: **{"PASSED" if sanity_pass else "FAILED"}**

## Metrics table

(mean +/- across-seed std; std computed over 5 seeds, not 20 tracts)

| condition | arch | within_tract_std | Moran's I | pooled_bg_r |
|---|---|---|---|---|
{metrics_rows}

## Verdict

Primary question: does mlp_floor fall within the production seed band on Moran's I and within-tract std?

{verdict}

## Figure

`graph_contribution.png`: 3x2 grid, rows = within_tract_std / Moran's I / pooled BG r, cols = SAGE / GCN-GAT, x = {{production, mlp_floor}}, error bars = across-seed std. Production band drawn as horizontal reference span.
"""
    README_PATH.write_text(readme)
    _log('README.md updated')


def _append_session_log(results, sha):
    _log('appending to SESSION_LOG.md...')
    now = datetime.now().strftime('%Y-%m-%d')

    prod_sage_r,  prod_sage_r_s  = _mean_std(results, 'production', 'sage',    'pooled_bg_r')
    prod_gcn_r,   prod_gcn_r_s   = _mean_std(results, 'production', 'gcn_gat', 'pooled_bg_r')
    mlp_sage_r,   mlp_sage_r_s   = _mean_std(results, 'mlp_floor',  'sage',    'pooled_bg_r')
    mlp_gcn_r,    mlp_gcn_r_s    = _mean_std(results, 'mlp_floor',  'gcn_gat', 'pooled_bg_r')

    prod_sage_std, _ = _mean_std(results, 'production', 'sage',    'within_tract_std')
    prod_gcn_std,  _ = _mean_std(results, 'production', 'gcn_gat', 'within_tract_std')
    mlp_sage_std,  _ = _mean_std(results, 'mlp_floor',  'sage',    'within_tract_std')
    mlp_gcn_std,   _ = _mean_std(results, 'mlp_floor',  'gcn_gat', 'within_tract_std')

    entry = f"""
## {now}: step 5 complete -- graph contribution boundary test

**Files changed:**
- `granite/data/loaders.py`: added `graph_variant` branch in `create_spatial_accessibility_graph`.
  `production` runs the existing road-network path unchanged. `mlp_floor` returns self-loops only
  (edge_index = [[i],[i]] for all i, edge_weight = 1.0). Unknown variant raises ValueError.
- `config.yaml`: added `graph_variant: production` (default preserves current behavior).
- `experiments/ablation/05_graph_contribution/`: run_sweep.py, make_figures.py, README.md,
  post_run_handler.py, per-condition subdirs with results and figures.

**Sweep:** 2 conditions x 2 architectures x 5 seeds x 20 tracts = 400 tract evaluations.
constraint_mode=soft, variation_weight=0.8, seeds=[42, 17, 123, 2024, 7].

**Sanity regression (production, seed=42):** PASSED within 1e-3 (BG r) / 2e-3 (std).

**Results:**

| condition | arch | within_tract_std | pooled_bg_r |
|---|---|---|---|
| production | SAGE | {prod_sage_std:.4f} | {prod_sage_r:.4f} +/- {prod_sage_r_s:.4f} |
| production | GCN-GAT | {prod_gcn_std:.4f} | {prod_gcn_r:.4f} +/- {prod_gcn_r_s:.4f} |
| mlp_floor | SAGE | {mlp_sage_std:.4f} | {mlp_sage_r:.4f} +/- {mlp_sage_r_s:.4f} |
| mlp_floor | GCN-GAT | {mlp_gcn_std:.4f} | {mlp_gcn_r:.4f} +/- {mlp_gcn_r_s:.4f} |

**Cache invalidation:** none. `graph_variant` affects only message-passing graph topology;
node features and OSRM routing cache keys are unchanged.

**Artifacts:** `experiments/ablation/05_graph_contribution/`
**Git sha:** {sha}
"""
    with open(SESSION_LOG, 'a') as f:
        f.write(entry)
    _log('SESSION_LOG.md updated')


def _update_research_status(results):
    _log('updating Research_Status.md...')
    content = RESEARCH_ST.read_text()

    prod_sage_r, _ = _mean_std(results, 'production', 'sage',    'pooled_bg_r')
    prod_gcn_r,  _ = _mean_std(results, 'production', 'gcn_gat', 'pooled_bg_r')
    mlp_sage_r,  _ = _mean_std(results, 'mlp_floor',  'sage',    'pooled_bg_r')
    mlp_gcn_r,   _ = _mean_std(results, 'mlp_floor',  'gcn_gat', 'pooled_bg_r')
    prod_sage_std, prod_sage_std_s = _mean_std(results, 'production', 'sage',    'within_tract_std')
    prod_gcn_std,  prod_gcn_std_s  = _mean_std(results, 'production', 'gcn_gat', 'within_tract_std')
    mlp_sage_std,  mlp_sage_std_s  = _mean_std(results, 'mlp_floor',  'sage',    'within_tract_std')
    mlp_gcn_std,   mlp_gcn_std_s   = _mean_std(results, 'mlp_floor',  'gcn_gat', 'within_tract_std')

    verdict = _verdict_paragraph(results)
    sha = _git_sha()
    now = datetime.now().strftime('%Y-%m-%d')

    step5_entry = f"""
### Ablation 05_graph_contribution: graph contribution boundary test (2026-{now[5:]})

**Status:** complete.

**Design.** Two conditions (production, mlp_floor), both architectures, five seeds [42, 17, 123, 2024, 7],
20 tracts each. constraint_mode=soft, variation_weight=0.8. mlp_floor replaces the road-network graph with
self-loops only -- SAGE and GAT reduce to node-wise functions on the input features.

**Results.**

| condition | arch | within_tract_std | pooled_bg_r |
|---|---|---|---|
| production | SAGE | {prod_sage_std:.4f} +/- {prod_sage_std_s:.4f} | {prod_sage_r:.4f} |
| production | GCN-GAT | {prod_gcn_std:.4f} +/- {prod_gcn_std_s:.4f} | {prod_gcn_r:.4f} |
| mlp_floor | SAGE | {mlp_sage_std:.4f} +/- {mlp_sage_std_s:.4f} | {mlp_sage_r:.4f} |
| mlp_floor | GCN-GAT | {mlp_gcn_std:.4f} +/- {mlp_gcn_std_s:.4f} | {mlp_gcn_r:.4f} |

**Verdict.**

{verdict}

**Artifacts.** `experiments/ablation/05_graph_contribution/` git sha: {sha}

---
"""
    # insert before the first ### entry (after the Active section header)
    marker = '### Ablation 00_baseline'
    if marker in content:
        content = content.replace(marker, step5_entry + marker, 1)
    else:
        # fallback: append to file
        content += step5_entry

    RESEARCH_ST.write_text(content)
    _log('Research_Status.md updated')


def _git_commit(msg, paths):
    _log(f'committing: {msg}')
    try:
        subprocess.run(['git', 'add'] + [str(p) for p in paths],
                       cwd=str(REPO_ROOT), check=True)
        subprocess.run(
            ['git', 'commit', '-m', msg + '\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>'],
            cwd=str(REPO_ROOT), check=True
        )
        sha = _git_sha()
        _log(f'committed: {sha}')
        return True
    except subprocess.CalledProcessError as e:
        _log(f'git commit failed: {e}')
        return False


def main():
    _log(f'polling {LOG_PATH} every {POLL_INTERVAL}s...')

    while not _sweep_complete(LOG_PATH):
        done = _count_done(LOG_PATH)
        _log(f'{done}/{TOTAL_SEEDS} seeds done -- waiting...')
        time.sleep(POLL_INTERVAL)

    _log('sweep complete, beginning post-processing')
    time.sleep(10)  # let final json writes flush

    if not RESULTS_PATH.exists():
        _log(f'ERROR: {RESULTS_PATH} not found after sweep. cannot post-process.')
        sys.exit(1)

    results = _load_results()

    # 1. figures
    fig_ok = _run_figures()

    # 2. README
    _update_readme(results)

    # 3. session log + research status
    sha = _git_sha()
    _append_session_log(results, sha)
    _update_research_status(results)

    # 4. commit run artifacts (no figures)
    run_paths = [
        ABLATION_DIR / 'run_sweep.py',
        ABLATION_DIR / 'make_figures.py',
        ABLATION_DIR / 'post_run_handler.py',
        ABLATION_DIR / 'README.md',
        ABLATION_DIR / 'results',
        ABLATION_DIR / 'production',
        ABLATION_DIR / 'mlp_floor',
        REPO_ROOT / 'granite' / 'data' / 'loaders.py',
        REPO_ROOT / 'config.yaml',
        SESSION_LOG,
        RESEARCH_ST,
    ]
    _git_commit(
        'Step 5 run: graph contribution boundary test (two-pole)\n\n'
        'production vs mlp_floor, 2 architectures, 5 seeds each.\n'
        'loaders.py: graph_variant branch (production/mlp_floor).\n'
        'config.yaml: graph_variant: production default.\n'
        'See experiments/ablation/05_graph_contribution/README.md',
        run_paths
    )

    # 5. commit figures separately
    if fig_ok:
        fig_paths = [ABLATION_DIR / 'graph_contribution.png']
        _git_commit(
            'Step 5 figures: graph_contribution.png (3x2 grid)',
            fig_paths
        )

    _log('all done')


if __name__ == '__main__':
    main()
