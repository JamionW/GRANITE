#!/usr/bin/env python3
"""
per-tract clip counts for paper 1 (preregistration v4.1, section 7).

counts saved npz prediction values exactly at 0 and exactly at 1. scores
nothing. per (set, architecture, graph condition, tract) for the frozen 5b
set and the crossed set separately, then totals and per-tract shares for
four bases:

  frozen_5b     every frozen 5b file (800)
  crossed       every crossed file (1,000)
  analysis_set  frozen structured conditions plus the crossed set (1,600);
                randomized cells read the crossed set only, so its ts42
                files, identical to the frozen randomized files, count once.
                the registered base.
  ts42_twice    every file in both sets (1,800)

gates (exit 1 on failure): base totals equal the settled TAG-B recon table;
analysis-set share rounds to 0.65% and 47065001900 holds 84.78% of clipped
values; 47065010431 leads every structured GraphSAGE condition and clips
nothing under randomization; exactly six tracts with SVI <= 0.51 (two
decimals) clip at 1.

usage:
    python scripts/paper1_clip_counts.py
writes data/results/paper1_prereg_artifacts/clip_counts.json
"""
import hashlib
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / 'data' / 'results'
SETS = {'frozen_5b': RESULTS / 'm6_topology_5b' / 'predictions',
        'crossed': RESULTS / 'm6_topology_5b_crossed' / 'predictions'}
FEATURE_MATRIX = REPO_ROOT / 'experiments' / 'ecological_fallacy' / 'n20_feature_matrix.csv'
OUT_PATH = RESULTS / 'paper1_prereg_artifacts' / 'clip_counts.json'
STRUCTURED = ('spatial_knn_uniform', 'road_network_uniform', 'production')
NAME = re.compile(r'^(?P<cond>[a-z_]+?)__(?P<arch>sage|gcn_gat)__ts(?P<ts>\d+)__gs(?P<gs>\d+)'
                  r'__(?P<fips>\d{11})\.npz$')

# settled at the TAG-B recon (Nov 11, ccbc3ee): clipped / values
EXPECTED = {'frozen_5b': (11481, 1581400), 'crossed': (14420, 1976750),
            'analysis_set': (20586, 3162800), 'ts42_twice': (25901, 3558150)}
EXPECTED_ANALYSIS_SHARE_PCT = 0.65
EXPECTED_1900_SHARE = 0.8478
INTERIOR_SVI_MAX = 0.51
EXPECTED_INTERIOR_AT_1 = {'47065000600', '47065011413', '47065011900',
                          '47065010431', '47065010433', '47065011326'}


def _git(*args):
    return subprocess.check_output(['git', *args], cwd=str(REPO_ROOT)).decode().strip()


def read_files():
    rows = []
    for set_name, d in SETS.items():
        for p in sorted(d.glob('*.npz')):
            m = NAME.match(p.name)
            if not m:
                raise RuntimeError(f'unexpected file name {p.name}')
            v = np.asarray(np.load(p)['predictions'], dtype=float)
            rows.append({'set': set_name, 'condition': m['cond'], 'arch': m['arch'],
                         'training_seed': int(m['ts']), 'graph_draw_seed': int(m['gs']),
                         'tract': m['fips'], 'path': str(p.relative_to(REPO_ROOT)),
                         'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                         'n_values': int(v.size), 'n_at_0': int((v == 0).sum()),
                         'n_at_1': int((v == 1).sum())})
    df = pd.DataFrame(rows)
    df['n_clipped'] = df['n_at_0'] + df['n_at_1']
    return df


def bases(df):
    f, c = df[df.set == 'frozen_5b'], df[df.set == 'crossed']
    analysis = pd.concat([f[f.condition.isin(STRUCTURED)], c])
    # crossed ts42 equals the frozen randomized arm file for file
    fr = f[f.condition == 'randomized'].set_index(['arch', 'graph_draw_seed', 'tract'])['sha256']
    c42 = c[c.training_seed == 42].set_index(['arch', 'graph_draw_seed', 'tract'])['sha256']
    if not fr.sort_index().equals(c42.sort_index()):
        raise RuntimeError('crossed ts42 files differ from the frozen randomized files')
    alt = pd.concat([f, c[c.training_seed != 42]])
    if alt[['n_values', 'n_clipped']].sum().tolist() != analysis[['n_values', 'n_clipped']].sum().tolist():
        raise RuntimeError('analysis set totals depend on which ts42 copy is kept')
    return {'frozen_5b': f, 'crossed': c, 'analysis_set': analysis, 'ts42_twice': df}


def per_cell_table(df):
    g = df.groupby(['arch', 'condition', 'tract'])[['n_values', 'n_at_0', 'n_at_1', 'n_clipped']].sum()
    n_files = df.groupby(['arch', 'condition', 'tract']).size()
    out = []
    for (arch, cond, tract), r in g.iterrows():
        out.append({'arch': arch, 'condition': cond, 'tract': tract,
                    'n_files': int(n_files[(arch, cond, tract)]),
                    **{k: int(r[k]) for k in ('n_values', 'n_at_0', 'n_at_1', 'n_clipped')}})
    return out


def base_summary(d, svi):
    t = d.groupby('tract')[['n_values', 'n_at_0', 'n_at_1', 'n_clipped']].sum()
    total_c, total_n = int(t.n_clipped.sum()), int(t.n_values.sum())
    tracts = {}
    for tract, r in t.iterrows():
        tracts[tract] = {'svi': float(svi[tract]),
                         **{k: int(r[k]) for k in ('n_values', 'n_at_0', 'n_at_1', 'n_clipped')},
                         'share_of_tract_values': float(r.n_clipped / r.n_values),
                         'share_of_clipped': float(r.n_clipped / total_c) if total_c else 0.0}
    return {'n_files': int(len(d)), 'n_values': total_n, 'n_at_0': int(t.n_at_0.sum()),
            'n_at_1': int(t.n_at_1.sum()), 'n_clipped': total_c,
            'share': total_c / total_n, 'share_pct': 100.0 * total_c / total_n, 'tracts': tracts}


def run_gates(summaries, df):
    g = []
    for b, (ec, en) in EXPECTED.items():
        s = summaries[b]
        g.append({'gate': f'{b} totals equal settled table', 'expected': [ec, en],
                  'observed': [s['n_clipped'], s['n_values']],
                  'pass': (s['n_clipped'], s['n_values']) == (ec, en)})
    a = summaries['analysis_set']
    g.append({'gate': 'analysis-set share rounds to 0.65%', 'expected': EXPECTED_ANALYSIS_SHARE_PCT,
              'observed': a['share_pct'], 'pass': round(a['share_pct'], 2) == EXPECTED_ANALYSIS_SHARE_PCT})
    s1900 = a['tracts']['47065001900']['share_of_clipped']
    g.append({'gate': '47065001900 holds 84.78% of analysis-set clipped values (rounds to 85%)',
              'expected': EXPECTED_1900_SHARE, 'observed': s1900,
              'pass': round(s1900, 4) == EXPECTED_1900_SHARE and round(100 * s1900) == 85})
    f = df[(df.set == 'frozen_5b') & (df.arch == 'sage')]
    for cond in STRUCTURED:
        t = f[f.condition == cond].groupby('tract').n_clipped.sum().sort_values(ascending=False)
        g.append({'gate': f'47065010431 leads frozen sage {cond}', 'expected': '47065010431',
                  'observed': [t.index[0], int(t.iloc[0])],
                  'pass': t.index[0] == '47065010431' and t.iloc[0] > t.iloc[1]})
    for set_name in ('frozen_5b', 'crossed'):
        r = df[(df.set == set_name) & (df.arch == 'sage') & (df.condition == 'randomized')
               & (df.tract == '47065010431')]
        g.append({'gate': f'47065010431 clips nothing under randomization (sage, {set_name})',
                  'expected': 0, 'observed': int(r.n_clipped.sum()),
                  'pass': len(r) > 0 and int(r.n_clipped.sum()) == 0})
    at1 = {t for t, r in a['tracts'].items() if r['n_at_1'] > 0 and round(r['svi'], 2) <= INTERIOR_SVI_MAX}
    g.append({'gate': 'exactly six tracts with SVI <= 0.51 clip at 1 (analysis set)',
              'expected': sorted(EXPECTED_INTERIOR_AT_1), 'observed': sorted(at1),
              'pass': at1 == EXPECTED_INTERIOR_AT_1})
    for x in g:
        x['pass'] = bool(x['pass'])
    return g


def main():
    df = read_files()
    fm = pd.read_csv(FEATURE_MATRIX, usecols=['fips', 'tract_svi'], dtype={'fips': str})
    svi = fm.groupby('fips')['tract_svi'].first()
    summaries = {b: base_summary(d, svi) for b, d in bases(df).items()}
    summaries['analysis_set']['registered'] = True
    gates = run_gates(summaries, df)
    file_list = '\n'.join(sorted(df['path'])) + '\n'
    payload = {
        'milestone': 'TAG-B',
        'artifact': 'per-tract clip counts (preregistration v4.1, section 7)',
        'definition': 'values exactly equal to 0.0 or 1.0 in the saved npz predictions; nothing scored',
        'registered_base': 'analysis_set',
        'bases': {
            'frozen_5b': 'every frozen 5b file',
            'crossed': 'every crossed file',
            'analysis_set': ('frozen structured conditions (spatial_knn_uniform, road_network_uniform, '
                             'production) plus the crossed set; randomized cells read the crossed set only, '
                             'so the crossed training-seed-42 files, byte-identical to the frozen randomized '
                             'files (checked by sha256), are counted once'),
            'ts42_twice': 'every file in both sets; the training-seed-42 randomized files count twice',
        },
        'note_073': ('0.73% is the frozen 5b set alone (0.726%) or the ts42 double count (0.728%); '
                     'the registered share is the analysis set'),
        'git_head': _git('rev-parse', 'HEAD'),
        'input_commits': {k: _git('log', '-1', '--format=%H', '--', str(v.relative_to(REPO_ROOT)))
                          for k, v in SETS.items()},
        'svi_source': str(FEATURE_MATRIX.relative_to(REPO_ROOT)),
        'n_files': int(len(df)),
        'file_list_sha256': hashlib.sha256(file_list.encode()).hexdigest(),
        'file_list_hash_rule': 'sha256 of the sorted repo-relative npz paths, one per line, trailing newline',
        'gates': gates,
        'all_gates_pass': all(x['pass'] for x in gates),
        'totals': {b: {k: v for k, v in s.items() if k != 'tracts'} for b, s in summaries.items()},
        'per_tract': {b: s['tracts'] for b, s in summaries.items()},
        'by_set_arch_condition_tract': {s: per_cell_table(df[df.set == s]) for s in SETS},
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=1, sort_keys=False) + '\n')
    for b, s in payload['totals'].items():
        print(f"{b:13s} files {s['n_files']:5d} clipped {s['n_clipped']:6d} / {s['n_values']:8d} "
              f"= {s['share_pct']:.4f}%")
    for x in gates:
        print(f"[{'PASS' if x['pass'] else 'FAIL'}] {x['gate']}: {x['observed']}")
    print(f'wrote {OUT_PATH.relative_to(REPO_ROOT)}')
    return 0 if payload['all_gates_pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
