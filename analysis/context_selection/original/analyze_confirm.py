#!/usr/bin/env python3
"""C3 v9 confirmatory analysis.

Primary contrast, fixed before generation: G_bare minus C_true on designated_wrong, per
checkpoint condition, tested by exchanging condition labels within subject and Holm
corrected across the four conditions.

Secondary: C_true minus B_filler (relation vs neutral) and C_true minus F_false (relation
truth), reported with the same machinery but not used to decide replication.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'next_stage' / 'c3_confirm' / 'raw'
PANEL = ROOT / 'next_stage' / 'cross_model_snapshot_117_20261113' / 'raw' / 'FACT_PANEL.csv'
INV = ROOT / 'next_stage' / 'c2_wrong_relations' / 'wrong_answer_pair_inventory.csv'
PROTO = ROOT / 'next_stage' / 'c3_confirm' / 'C3_CONFIRM_V9_PROTOCOL.json'
OUT = ROOT / 'next_stage' / 'c3_confirm'

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


d = pd.concat([pd.read_parquet(f) for f in sorted(RAW.glob('c3v9_*.parquet'))],
              ignore_index=True)
d['cls'] = [classify(o, w, g) for o, w, g in zip(d.output, d.wrong_answer, d.gold)]
print(f'rows={len(d)}  conditions={d.task.nunique()}  subjects={d.target_country.nunique()} '
      f'pairs={d.groupby(["target_country","wrong_answer"]).ngroups}')

order = ['correct', 'designated_wrong', 'mixed', 'other_answer',
         'negated_or_corrected', 'unverifiable', 'nonanswer']
print('\n=== outcome by arm ===')
print(pd.crosstab(d.arm, d.cls, normalize='index').reindex(columns=order).round(4).to_string())


def holm(p):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    adj = np.maximum.accumulate(p[o] * (n - np.arange(n)))
    out = np.empty(n); out[o] = np.minimum(adj, 1.0)
    return out


def cluster_perm(sub, arm, base, cls, n_perm=20000, seed=20261213):
    g = sub[sub.arm.isin([arm, base])]
    groups = []
    for _, x in g.groupby('target_country'):
        a, b = x[x.arm == arm], x[x.arm == base]
        if not len(a) or not len(b):
            continue
        groups.append((int((a.cls == cls).sum()), len(a),
                       int((b.cls == cls).sum()), len(b)))
    if len(groups) < 5:
        return None
    G = np.array(groups, float)
    obs = 100 * (G[:, 0].sum() / G[:, 1].sum() - G[:, 2].sum() / G[:, 3].sum())
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        s = rng.random(len(G)) < 0.5
        ka = np.where(s, G[:, 2], G[:, 0]); na = np.where(s, G[:, 3], G[:, 1])
        kb = np.where(s, G[:, 0], G[:, 2]); nb = np.where(s, G[:, 1], G[:, 3])
        null[i] = 100 * (ka.sum() / na.sum() - kb.sum() / nb.sum())
    return obs, (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1), len(G)


CONTRASTS = [('G_bare', 'C_true', 'PRIMARY_G_minus_C'),
             ('C_true', 'B_filler', 'C_minus_B'),
             ('C_true', 'F_false', 'C_minus_F')]

rows = []
for task in sorted(d.task.unique()):
    sub = d[d.task == task]
    for arm, base, label in CONTRASTS:
        for cls in ('designated_wrong', 'correct'):
            r = cluster_perm(sub, arm, base, cls)
            if r is None:
                continue
            rows.append(dict(condition=task, contrast=label, outcome=cls,
                             delta_pp=round(r[0], 2), p_raw=r[1], n_subjects=r[2]))
R = pd.DataFrame(rows)
R['p_holm'] = np.nan
R['sig_holm'] = False
for (label, cls), g in R.groupby(['contrast', 'outcome']):
    adj = holm(g.p_raw.values)
    R.loc[g.index, 'p_holm'] = np.round(adj, 5)
    R.loc[g.index, 'sig_holm'] = adj < 0.05
R.to_csv(OUT / 'V9_CONTRASTS.csv', index=False, encoding='utf-8')

print('\n\n=== PRIMARY: bare mention minus true relation, city-final, words matched ===')
P = R[(R.contrast == 'PRIMARY_G_minus_C') & (R.outcome == 'designated_wrong')]
print(P.to_string(index=False))
n_dir = int((P.delta_pp > 0).sum())
n_sig = int(P.sig_holm.sum())
print(f'\n  direction consistent: {n_dir}/{len(P)} positive')
print(f'  Holm-significant: {n_sig}/{len(P)}')

print('\n=== secondary contrasts ===')
for label in ('C_minus_B', 'C_minus_F'):
    Q = R[(R.contrast == label) & (R.outcome == 'designated_wrong')]
    print(f'\n{label}:'); print(Q.to_string(index=False))

print('\n=== VERDICT against the pre-registered stopping rule ===')
rule = ('report per-condition primary contrast; if direction is consistent and intervals '
        'exclude zero the effect replicates; otherwise report the inconsistency')
print(f'  rule: {rule}')
if n_dir == len(P) and n_sig == len(P):
    print('  REPLICATES: all four conditions positive and Holm-significant.')
elif n_dir == len(P):
    print('  DIRECTION CONSISTENT but not all significant: partial replication.')
else:
    print('  NOT REPLICATED: direction inconsistent across conditions.')
print(f'\nwrote {OUT}')
