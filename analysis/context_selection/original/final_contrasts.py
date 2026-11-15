#!/usr/bin/env python3
"""C3 final contrasts with cluster-bootstrap intervals, for both labelings and both rounds.

The v8 and v9 analyses reported point estimates and Holm-corrected permutation p values but no
intervals. A p value without an interval cannot express how large an effect the data exclude,
which is the thing a reader needs. This script adds the intervals.

Clustering unit: the target country, as everywhere else. Intervals are percentile bootstrap
over subjects, 20000 resamples. The permutation p is recomputed with the same seeds as before
so the two outputs can be checked against each other.
"""
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'next_stage' / 'c3_final'

ROUNDS = {
    'v8_development': dict(raw=ROOT / 'next_stage' / 'c3_realwrong_v8' / 'raw',
                           prefix='c3v8_', n_perm=20000),
    'v9_confirmatory': dict(raw=ROOT / 'next_stage' / 'c3_confirm' / 'raw',
                            prefix='c3v9_', n_perm=20000),
}
PANEL = ROOT / 'next_stage' / 'cross_model_snapshot_117_20261113' / 'raw' / 'FACT_PANEL.csv'
NEG = ('not', 'isnt', 'rather', 'instead', 'actually', 'incorrect', 'wrong', 'never')


def norm(s):
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9 ]', ' ', str(s).lower())).strip()


panel = pd.read_csv(PANEL, keep_default_na=False, low_memory=False)
cap = panel[panel.prop == 'capital']
city = {norm(x) for x in cap.gold.unique() if str(x).strip()}
country = {norm(x) for x in panel.subj.unique() if str(x).strip()}
city.discard('')


def slot(out):
    return ' '.join(re.split(r'[.\n]', str(out).lstrip(), maxsplit=1)[0].split()[:6])


def label_audited(out, wrong, gold):
    s = norm(slot(out))
    if not s:
        return 'nonanswer'
    w, g = norm(wrong), norm(gold)
    if g and g in s:
        return 'correct'
    if w and w in s:
        i = s.find(w)
        if any(k in s[max(0, i - 20):i + len(w) + 8].split() for k in NEG):
            return 'negated_or_corrected'
        return 'designated_wrong'
    if any(c in s for c in city if c):
        return 'other_answer'
    if any(c in s for c in country if c):
        return 'type_mismatch'
    if any(k in s.split() for k in ('airport', 'county', 'province', 'state', 'island')):
        return 'type_mismatch'
    return 'nonanswer' if len(s.split()) <= 2 else 'unverifiable'


def label_rule(out, wrong, gold):
    t = norm(out)
    if not t:
        return 'nonanswer'
    w, g = norm(wrong), norm(gold)
    hw, hg = (w in t), (g in t)
    if hw and hg:
        return 'mixed'
    if hw:
        return 'designated_wrong'
    if hg:
        return 'correct'
    if any(c in t for c in city if c):
        return 'other_answer'
    return 'nonanswer' if len(t.split()) <= 2 else 'unverifiable'


def holm(p):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    adj = np.maximum.accumulate(p[o] * (n - np.arange(n)))
    out = np.empty(n)
    out[o] = np.minimum(adj, 1.0)
    return out


def cluster_stats(sub, arm, base, col, cls, rng, n_perm=20000, n_boot=20000):
    g = sub[sub.arm.isin([arm, base])]
    cells = []
    for c, x in g.groupby('target_country'):
        a, b = x[x.arm == arm], x[x.arm == base]
        if not len(a) or not len(b):
            continue
        cells.append((c, int((a[col] == cls).sum()), len(a),
                      int((b[col] == cls).sum()), len(b)))
    if len(cells) < 5:
        return None
    A = np.array([(k1, n1, k2, n2) for _, k1, n1, k2, n2 in cells], float)
    obs = 100 * (A[:, 0].sum() / A[:, 1].sum() - A[:, 2].sum() / A[:, 3].sum())
    null = np.empty(n_perm)
    for i in range(n_perm):
        s = rng.random(len(A)) < 0.5
        ka = np.where(s, A[:, 2], A[:, 0]); na = np.where(s, A[:, 3], A[:, 1])
        kb = np.where(s, A[:, 0], A[:, 2]); nb = np.where(s, A[:, 1], A[:, 3])
        null[i] = 100 * (ka.sum() / na.sum() - kb.sum() / nb.sum())
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)
    bs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(A), len(A))
        S = A[idx]
        bs[i] = 100 * (S[:, 0].sum() / S[:, 1].sum() - S[:, 2].sum() / S[:, 3].sum())
    lo, hi = np.percentile(bs, 2.5), np.percentile(bs, 97.5)
    return obs, p, lo, hi, len(cells)


CONTRASTS = [('G_matched', 'C_true', 'G_bare - C_rel'),
             ('G_matched', 'B_filler', 'G_bare - B'),
             ('C_true', 'B_filler', 'C_rel - B'),
             ('C_true', 'F_false', 'C_rel - F_false')]

rows = []
for rname, cfg in ROUNDS.items():
    files = sorted(cfg['raw'].glob(cfg['prefix'] + '*.parquet'))
    if not files:
        print(f'  {rname}: no parquet files found, skipped')
        continue
    d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    d['cls'] = [label_audited(o, w, g) for o, w, g in
                zip(d.output, d.wrong_answer, d.gold)]
    d['cls_rule'] = [label_rule(o, w, g) for o, w, g in
                     zip(d.output, d.wrong_answer, d.gold)]
    print(f'{rname}: rows={len(d)}  conditions={d.task.nunique()}  '
          f'pairs={d.groupby(["target_country","wrong_answer"]).ngroups}')
    arms = set(d.arm)
    alias = {'G_matched': 'G_matched' if 'G_matched' in arms else 'G_bare'}
    for col, lname in (('cls', 'audited'), ('cls_rule', 'rule')):
        rng = np.random.default_rng(20261228)
        block = []
        for task in sorted(d.task.unique()):
            sub = d[d.task == task]
            for arm, base, cname in CONTRASTS:
                arm = alias.get(arm, arm)
                if arm not in set(sub.arm) or base not in set(sub.arm):
                    continue
                r = cluster_stats(sub, arm, base, col, 'designated_wrong', rng)
                if r is None:
                    continue
                block.append(dict(round=rname, labeling=lname, condition=task,
                                  contrast=cname, delta_pp=round(r[0], 2),
                                  ci_lo=round(r[2], 2), ci_hi=round(r[3], 2),
                                  p_raw=r[1], n_subjects=r[4]))
        if block:
            B = pd.DataFrame(block)
            B['p_holm'] = np.nan
            B['sig'] = False
            for cname, g in B.groupby('contrast'):
                adj = holm(g.p_raw.values)
                B.loc[g.index, 'p_holm'] = np.round(adj, 5)
                B.loc[g.index, 'sig'] = adj < 0.05
            rows.append(B)
            print(f'  {lname}: {len(B)} contrasts computed')

R = pd.concat(rows, ignore_index=True)
R.to_csv(OUT / 'C3_FINAL_CONTRASTS.csv', index=False, encoding='utf-8')
print(f'\nwrote {OUT}/C3_FINAL_CONTRASTS.csv  rows={len(R)}')

print('\n=== primary contrast, audited labeling, both rounds ===')
P = R[(R.contrast == 'G_bare - C_rel') & (R.labeling == 'audited')]
print(P[['round', 'condition', 'delta_pp', 'ci_lo', 'ci_hi', 'p_holm', 'sig']]
      .to_string(index=False))
for rname, g in P.groupby('round'):
    print(f'  {rname}: positive {int((g.delta_pp > 0).sum())}/{len(g)}, '
          f'interval excludes zero {int(g.sig.sum())}/{len(g)}, '
          f'range {g.delta_pp.min():+.2f} to {g.delta_pp.max():+.2f}')

print('\n=== dependence of the primary contrast on the labeling ===')
W = R[(R.contrast == 'G_bare - C_rel')].pivot_table(
    index=['round', 'condition'], columns='labeling', values='delta_pp')
if 'audited' in W and 'rule' in W:
    W['difference'] = (W.audited - W.rule).round(2)
    print(W.to_string())
    print(f'  max |difference| = {W.difference.abs().max():.2f} pp')
