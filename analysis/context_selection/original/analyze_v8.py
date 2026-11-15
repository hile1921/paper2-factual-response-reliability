#!/usr/bin/env python3
"""C3 v8 analysis: is the bare-mention advantage explained by length?

If G_matched (bare mention, length matched to the relation sentence) still produces more
designated wrong answers than C_true, then length is excluded as the explanation.
C_padded and G_short bound how much padding alone moves the outcome.
"""
import re
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'next_stage' / 'c3_realwrong_v8' / 'raw'
PANEL = ROOT / 'next_stage' / 'cross_model_snapshot_117_20261113' / 'raw' / 'FACT_PANEL.csv'
INV = ROOT / 'next_stage' / 'c2_wrong_relations' / 'wrong_answer_pair_inventory.csv'
OUT = ROOT / 'next_stage' / 'c3_realwrong_v8'

NEG = ('not', 'isnt', 'rather', 'instead', 'actually', 'incorrect', 'wrong', 'never')


def norm(s):
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9 ]', ' ', str(s).lower())).strip()


panel = pd.read_csv(PANEL, keep_default_na=False, low_memory=False)
lex = {norm(x) for x in panel[panel.prop == 'capital'].gold.unique() if str(x).strip()}
inv = pd.read_csv(INV, keep_default_na=False, low_memory=False)
lex |= {norm(x) for x in inv.generated_wrong_answer.unique() if str(x).strip()}
lex.discard('')


def classify(out, wrong, correct):
    t = norm(out)
    if not t:
        return 'nonanswer'
    w, c = norm(wrong), norm(correct)
    hw, hc = (w in t), (c in t)
    if hw and hc:
        return 'mixed'
    if hw:
        i = t.find(w)
        win = t[max(0, i - 24):i + len(w) + 10]
        return 'negated_or_corrected' if any(k in win.split() for k in NEG) else 'designated_wrong'
    if hc:
        return 'correct'
    for cand in lex:
        if cand and cand not in (w, c) and cand in t:
            return 'other_answer'
    if len(t.split()) <= 2:
        return 'nonanswer'
    return 'unverifiable'


d = pd.concat([pd.read_parquet(f) for f in sorted(RAW.glob('c3v8_*.parquet'))],
              ignore_index=True)
d['cls'] = [classify(o, w, g) for o, w, g in zip(d.output, d.wrong_answer, d.gold)]
print(f'rows={len(d)}  models={d.task.nunique()}  subjects={d.target_country.nunique()}')

order = ['correct', 'designated_wrong', 'mixed', 'other_answer',
         'negated_or_corrected', 'unverifiable', 'nonanswer']
print('\n=== outcome by arm ===')
print(pd.crosstab(d.arm, d.cls, normalize='index').reindex(columns=order).round(4).to_string())


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.minimum(1.0, p[o] * (len(p) - np.arange(len(p))))
    out = np.empty_like(p); out[o] = adj
    return out


def cluster_boot(sub, arm, base, cls, n_boot=4000, seed=20261209):
    def agg(a):
        return a.groupby('target_country').apply(
            lambda x: pd.Series({'k': (x.cls == cls).sum(), 'n': len(x)}),
            include_groups=False)
    A, B = agg(sub[sub.arm == arm]), agg(sub[sub.arm == base])
    idx = sorted(set(A.index) & set(B.index))
    if not idx:
        return None
    A, B = A.reindex(idx).fillna(0), B.reindex(idx).fillna(0)
    obs = A.k.sum() / A.n.sum() - B.k.sum() / B.n.sum()
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(idx), len(idx))
        sa, sb = A.iloc[i], B.iloc[i]
        if sa.n.sum() and sb.n.sum():
            bs.append(sa.k.sum() / sa.n.sum() - sb.k.sum() / sb.n.sum())
    return obs, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(idx)


CONTRASTS = [('G_matched', 'C_true', 'G_matched_minus_C_true'),
             ('G_short', 'G_matched', 'G_short_minus_G_matched'),
             ('C_padded', 'C_true', 'C_padded_minus_C_true'),
             ('G_matched', 'B_filler', 'G_matched_minus_B'),
             ('C_true', 'B_filler', 'C_minus_B')]

rows = []
for task in sorted(d.task.unique()):
    sub = d[d.task == task]
    for arm, base, label in CONTRASTS:
        for cls in ('designated_wrong', 'correct'):
            r = cluster_boot(sub, arm, base, cls)
            if r is None:
                continue
            rows.append(dict(model=task, contrast=label, outcome=cls,
                             delta_pp=round(100 * r[0], 2),
                             ci_lo=round(100 * r[1], 2), ci_hi=round(100 * r[2], 2),
                             n_subjects=r[3],
                             excludes_zero=bool(r[1] > 0 or r[2] < 0)))
R = pd.DataFrame(rows)
R['p_holm'] = np.nan
R['sig_holm'] = False
for (label, cls), g in R.groupby(['contrast', 'outcome']):
    se = (g.ci_hi - g.ci_lo) / 3.92
    p = np.array([2 * (1 - 0.5 * (1 + erf(abs(x) / (s * sqrt(2)))))
                  for x, s in zip(g.delta_pp, se)])
    adj = holm(np.clip(p, 0, 1))
    R.loc[g.index, 'p_holm'] = np.round(adj, 4)
    R.loc[g.index, 'sig_holm'] = adj < 0.05
R.to_csv(OUT / 'V8_CONTRASTS.csv', index=False, encoding='utf-8')
print('\n=== all contrasts ===')
print(R.to_string(index=False))

print('\n=== decisive: does bare mention still beat the relation at matched length? ===')
piv = R[R.contrast == 'G_matched_minus_C_true'].pivot_table(
    index='model', columns='outcome', values='delta_pp')
print(piv.to_string())
print('\n=== padding controls ===')
for c in ('G_short_minus_G_matched', 'C_padded_minus_C_true'):
    piv = R[R.contrast == c].pivot_table(index='model', columns='outcome', values='delta_pp')
    print(f'\n{c}:'); print(piv.to_string())
print(f'\nwrote {OUT}')
