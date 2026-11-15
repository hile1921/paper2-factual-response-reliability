#!/usr/bin/env python3
"""C1 final analysis on the 117-checkpoint snapshot.

Adds what the earlier pass did not do:
  (b) paired cluster bootstrap: the SAME subjects are resampled for early and late, so the
      comparison respects the pairing instead of treating the two periods independently;
  (c) block-length sensitivity for the trend (blocks 3, 5, 8, 10);
  (d) A / B / E plus an explicit bracket for B under the two extreme readings of the
      unverified outputs (all correct vs all wrong);
  (e) capital and country analysed separately, so a change in task mix cannot be mistaken
      for a change in accuracy;
  (f) temporal dynamics per fact: first appearance, persistence, disappearance,
      reappearance, and replacement of one wrong answer by another.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SNAP = sys.argv[1] if len(sys.argv) > 1 else 'cross_model_snapshot_117_20261113'
T = ROOT / 'next_stage' / SNAP / 'tables'
OUT = ROOT / 'next_stage' / 'c1_final' / SNAP
OUT.mkdir(parents=True, exist_ok=True)

d = pd.read_csv(T / 'C1_AUDITED_OUTPUTS.csv', keep_default_na=False, low_memory=False)
d['step'] = d.revision.str.extract(r'step(\d+)').astype(float)
# AMBER uses ckpt_000/179/main; give them a synthetic ordering
ck_order = {c: i for i, c in enumerate(
    ['ckpt_000', 'ckpt_179', 'main', 'step15000-tokens62B',
     'step369000-tokens1547B', 'step738020-tokens3094B'])}
d['order'] = d.step.fillna(d.revision.map(ck_order).fillna(-1))
print(f'rows={len(d)} checkpoints={d.task_id.nunique()} facts={d.uid.nunique()}')

d['is_correct'] = (d.audit_label == 'correct_verified').astype(int)
d['is_wrong'] = (d.audit_label == 'wrong_confirmed').astype(int)
d['is_unver'] = (d.audit_label == 'unverified').astype(int)
d['is_ans'] = d.is_correct + d.is_wrong + d.is_unver      # produced a candidate answer
d['is_scorable'] = d.is_correct + d.is_wrong               # answer could be verified


def aggregates(g):
    n = len(g)
    c, w, u = g.is_correct.sum(), g.is_wrong.sum(), g.is_unver.sum()
    s = c + w
    return pd.Series(dict(
        n=n, n_correct=c, n_wrong=w, n_unverified=u,
        A_scorable=s / n,
        B_cond=w / s if s else np.nan,
        E=w / n,
        # bracket for B under the two extreme readings of unverified outputs
        B_lo=w / (s + u) if (s + u) else np.nan,
        B_hi=(w + u) / (s + u) if (s + u) else np.nan,
    ))


# ---------------------------------------------------------------- (e) by relation type
print('\n=== (e) capital vs country, per model (all checkpoints pooled) ===')
byprop = d.groupby(['model', 'prop']).apply(aggregates, include_groups=False).reset_index()
byprop.to_csv(OUT / 'ABE_BY_RELATION.csv', index=False, encoding='utf-8')
print(byprop.round(4).to_string(index=False))

print('\n=== (e2) same, but split early (<=20k) vs late (>=100k) ===')
per_cp = d.groupby(['model', 'prop', 'task_id', 'order']).apply(
    aggregates, include_groups=False).reset_index()
per_cp.to_csv(OUT / 'ABE_BY_CHECKPOINT.csv', index=False, encoding='utf-8')
for model in sorted(d.model.unique()):
    for prop in ('capital', 'country'):
        g = per_cp[(per_cp.model == model) & (per_cp.prop == prop)]
        e = g[g.order <= 20000]; l = g[g.order >= 100000]
        if not len(e) or not len(l):
            continue
        ae = e.n_correct.sum() + e.n_wrong.sum()
        al = l.n_correct.sum() + l.n_wrong.sum()
        print(f'  {model:12s} {prop:8s} early A={ae/e.n.sum():.3f} '
              f'B={e.n_wrong.sum()/ae if ae else float("nan"):.3f}   '
              f'late A={al/l.n.sum():.3f} B={l.n_wrong.sum()/al if al else float("nan"):.3f}')


# ---------------------------------------------------------------- (b) paired bootstrap
def paired_cluster_boot(sub, n_boot=4000, seed=20261123):
    """Resample SUBJECTS once; apply the same draw to early and late."""
    early = sub[sub.order <= 20000]
    late = sub[sub.order >= 100000]
    if not len(early) or not len(late):
        return None
    E = early.groupby('subj')[['is_correct', 'is_wrong', 'is_unver']].sum()
    L = late.groupby('subj')[['is_correct', 'is_wrong', 'is_unver']].sum()
    subs = sorted(set(E.index) | set(L.index))
    E = E.reindex(subs).fillna(0); L = L.reindex(subs).fillna(0)

    def b_of(x):
        s = float((x.is_correct + x.is_wrong).sum())
        return float(x.is_wrong.sum() / s) if s > 0 else np.nan

    obs_e, obs_l = b_of(E), b_of(L)
    if not (np.isfinite(obs_e) and np.isfinite(obs_l)):
        return None
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(subs), len(subs))
        be, bl = b_of(E.iloc[idx]), b_of(L.iloc[idx])
        if np.isfinite(be) and np.isfinite(bl):
            diffs.append(be - bl)
    return (float(obs_e), float(obs_l), float(obs_e - obs_l),
            float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)),
            len(subs))


print('\n=== (b) paired early-late comparison (same subjects resampled for both) ===')
rows = []
for model in sorted(d.model.unique()):
    for prop in ('capital', 'country', 'all'):
        sub = d[d.model == model] if prop == 'all' else d[(d.model == model) & (d.prop == prop)]
        r = paired_cluster_boot(sub)
        if r is None:
            continue
        rows.append(dict(model=model, prop=prop, B_early=round(r[0], 4),
                         B_late=round(r[1], 4), diff=round(r[2], 4),
                         ci_lo=round(r[3], 4), ci_hi=round(r[4], 4),
                         n_subjects=r[5],
                         excludes_zero=bool(r[3] > 0 or r[4] < 0)))
PAIR = pd.DataFrame(rows)
PAIR.to_csv(OUT / 'PAIRED_EARLY_LATE.csv', index=False, encoding='utf-8')
print(PAIR.to_string(index=False))


# ---------------------------------------------------------------- (c) block sensitivity
def block_rho(x, y, block, n_boot=3000, seed=20261124):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 6:
        return np.nan, np.nan, np.nan
    def rho(a, b):
        if np.std(a) == 0 or np.std(b) == 0:
            return np.nan
        return float(np.corrcoef(pd.Series(a).rank(), pd.Series(b).rank())[0, 1])
    obs = rho(x, y)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(len(x) / block))
    bs = []
    for _ in range(n_boot):
        st = rng.integers(0, len(x), nb)
        idx = np.concatenate([np.arange(s, min(s + block, len(x))) for s in st])[:len(x)]
        v = rho(x[idx], y[idx])
        if np.isfinite(v):
            bs.append(v)
    return obs, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


print('\n=== (c) block-length sensitivity ===')
sens = []
for model in sorted(d.model.unique()):
    g = per_cp[(per_cp.model == model) & (per_cp.prop == 'all')] if False else \
        per_cp[per_cp.model == model].groupby('order', as_index=False).agg(
            {'n': 'sum', 'n_correct': 'sum', 'n_wrong': 'sum', 'n_unverified': 'sum'})
    if len(g) < 6:
        continue
    g = g.sort_values('order')
    s = g.n_correct + g.n_wrong
    g['A'] = s / g.n
    g['B'] = g.n_wrong / s.replace(0, np.nan)
    g['E'] = g.n_wrong / g.n
    for var in ('A', 'B', 'E'):
        for blk in (3, 5, 8, 10):
            r, lo, hi = block_rho(g['order'], g[var], blk)
            sens.append(dict(model=model, variable=var, block=blk,
                             rho=None if np.isnan(r) else round(r, 3),
                             ci_lo=None if np.isnan(lo) else round(lo, 3),
                             ci_hi=None if np.isnan(hi) else round(hi, 3),
                             excludes_zero=bool(np.isfinite(lo) and np.isfinite(hi)
                                                and (lo > 0 or hi < 0))))
SENS = pd.DataFrame(sens)
SENS.to_csv(OUT / 'BLOCK_SENSITIVITY.csv', index=False, encoding='utf-8')
print(SENS.pivot_table(index=['model', 'variable'], columns='block',
                       values='excludes_zero').to_string())
print('\nrho by block:')
print(SENS.pivot_table(index=['model', 'variable'], columns='block', values='rho').to_string())


# ---------------------------------------------------------------- (f) temporal dynamics
print('\n=== (f) temporal dynamics per fact ===')
d['wrong_answer'] = d['span'].where(d.is_wrong == 1)
rows = []
for (model, prop, uid, subj), g in d.groupby(['model', 'prop', 'uid', 'subj']):
    g = g.sort_values('order')
    steps = g.order.values
    wrong_by_step = g.groupby('order').is_wrong.sum().reindex(steps).fillna(0).values
    correct_by_step = g.groupby('order').is_correct.sum().reindex(steps).fillna(0).values
    hit = wrong_by_step > 0
    if not hit.any():
        rows.append(dict(model=model, prop=prop, subj=subj, uid=uid, ever_wrong=0,
                         first=-1, last=-1, n_wrong_cp=0, n_cp=len(steps),
                         persistence=np.nan, disappears=0, reappears=0,
                         n_distinct_wrong_answers=0, answers=''))
        continue
    idx = np.where(hit)[0]
    first, last = int(steps[idx[0]]), int(steps[idx[-1]])
    # reappearance: a zero stretch flanked by ones
    re = 0
    seen_hit = False
    for k in range(len(hit)):
        if hit[k]:
            if seen_hit and not hit[k - 1]:
                re += 1
            seen_hit = True
    span = hit[idx[0]:idx[-1] + 1]
    pers = float(span.mean()) if len(span) else np.nan
    ans = g.loc[g.is_wrong == 1, 'wrong_answer'].dropna().unique()
    rows.append(dict(model=model, prop=prop, subj=subj, uid=uid, ever_wrong=1,
                     first=first, last=last, n_wrong_cp=int(hit.sum()),
                     n_cp=len(steps), persistence=pers,
                     disappears=int(last < steps[-1]), reappears=re,
                     n_distinct_wrong_answers=len(ans),
                     answers=';'.join(sorted(str(a) for a in ans))[:120]))
DYN = pd.DataFrame(rows)
DYN.to_csv(OUT / 'TEMPORAL_DYNAMICS.csv', index=False, encoding='utf-8')
print(f'  facts analysed: {len(DYN)}')
for model, g in DYN.groupby('model'):
    ev = g[g.ever_wrong == 1]
    if not len(ev):
        continue
    print(f'  {model:12s} facts={len(g):4d} ever_wrong={len(ev):4d} '
          f'({len(ev)/len(g):.1%})  median_persistence={ev.persistence.median():.2f}  '
          f'disappear_before_end={ev.disappears.mean():.1%}  '
          f'reappear_any={int((ev.reappears>0).sum())}  '
          f'multi_wrong_answer={int((ev.n_distinct_wrong_answers>1).sum())}')
print(f'\nwrote {OUT}')
