#!/usr/bin/env python3
"""C1 A/B/E decomposition, built from per-output audit labels.

Definitions (mutually exclusive and exhaustive over all outputs):
    A  scorable share      = (correct_verified + wrong_confirmed) / all outputs
    B  conditional error   = wrong_confirmed / (correct_verified + wrong_confirmed)
    E  confirmed error     = wrong_confirmed / all outputs            == A * B

The remaining labels stay separate and are never counted as right or wrong:
    nonanswer, unverified, type_mismatch

Inference handles two dependences that the user flagged:
  * adjacent checkpoints are nested in training, so a moving-block bootstrap is used for
    any trend against step;
  * outputs cluster by subject/country, so the early-vs-late comparison bootstraps over
    subjects, not over outputs.

Usage:  python c1_abe.py <snapshot_dir_name>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SNAP = sys.argv[1] if len(sys.argv) > 1 else 'cross_model_snapshot_91_20261113'
T = ROOT / 'next_stage' / SNAP / 'tables'
OUT = ROOT / 'next_stage' / 'c1_abe' / SNAP
OUT.mkdir(parents=True, exist_ok=True)

LABELS = ['correct_verified', 'wrong_confirmed', 'unverified', 'nonanswer',
          'type_mismatch']

d = pd.read_csv(T / 'C1_AUDITED_OUTPUTS.csv', keep_default_na=False, low_memory=False)
d['step'] = d.revision.str.extract(r'step(\d+)').astype(float)
print(f'rows={len(d)}  checkpoints={d.task_id.nunique()}  '
      f'labels={d.audit_label.value_counts().to_dict()}')

# ------------------------------------------------------------------ per checkpoint A/B/E
rows = []
for (task, model, rev, step), g in d.groupby(['task_id', 'model', 'revision', 'step'],
                                              dropna=False):
    n = len(g)
    cnt = g.audit_label.value_counts()
    c = int(cnt.get('correct_verified', 0))
    w = int(cnt.get('wrong_confirmed', 0))
    r = dict(task_id=task, model=model, revision=rev,
             step=int(step) if pd.notna(step) else -1, outputs=n,
             n_correct=c, n_wrong=w, n_unverified=int(cnt.get('unverified', 0)),
             n_nonanswer=int(cnt.get('nonanswer', 0)),
             n_type=int(cnt.get('type_mismatch', 0)))
    r['A_scorable'] = (c + w) / n if n else np.nan
    r['B_cond_error'] = w / (c + w) if (c + w) else np.nan
    r['E_confirmed'] = w / n if n else np.nan
    r['E_from_AB'] = r['A_scorable'] * (r['B_cond_error'] if r['B_cond_error'] == r['B_cond_error'] else 0)
    rows.append(r)
cp = pd.DataFrame(rows).sort_values(['model', 'step']).reset_index(drop=True)
cp.to_csv(OUT / 'C1_ABE_BY_CHECKPOINT.csv', index=False, encoding='utf-8')

# ------------------------------------------------------------------ exact identity check
bad = cp[(cp.E_confirmed - cp.E_from_AB).abs() > 1e-12]
print(f'\n=== identity E = A*B ===')
print(f'  checkpoints checked: {len(cp)}   violations: {len(bad)}')
print(f'  max |E - A*B| = {(cp.E_confirmed - cp.E_from_AB).abs().max():.2e}')

for model in cp.model.unique():
    g = cp[cp.model == model]
    print(f'\n=== {model} (n={len(g)}) ===')
    print(f"{'step':>7} {'A':>7} {'B':>7} {'E':>7} {'nonans':>7} {'unverif':>8} {'type':>6}")
    for _, r in g.iterrows():
        b = f"{r.B_cond_error:.3f}" if pd.notna(r.B_cond_error) else '   --'
        print(f"{r.step:>7} {r.A_scorable:>7.4f} {b:>7} {r.E_confirmed:>7.4f} "
              f"{r.n_nonanswer/r.outputs:>7.3f} {r.n_unverified/r.outputs:>8.3f} "
              f"{r.n_type/r.outputs:>6.3f}")


# ------------------------------------------------------------------ helpers
def moving_block_spearman(x, y, block=5, n_boot=4000, seed=20261121):
    """Block bootstrap over consecutive checkpoints, preserving local dependence."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 6:
        return np.nan, np.nan, np.nan, n

    def rho(a, b):
        if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
            return np.nan
        return float(np.corrcoef(pd.Series(a).rank(), pd.Series(b).rank())[0, 1])

    obs = rho(x, y)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    boots = []
    for _ in range(n_boot):
        starts = rng.integers(0, n, nb)
        idx = np.concatenate([np.arange(s, min(s + block, n)) for s in starts])[:n]
        v = rho(x[idx], y[idx])
        if np.isfinite(v):
            boots.append(v)
    if not boots:
        return obs, np.nan, np.nan, n
    return obs, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), n


def cluster_boot_diff(k1, n1, k2, n2, clusters1, clusters2, n_boot=4000, seed=20261122):
    """Bootstrap the early-late difference over subject clusters."""
    def agg(group_keys, k, n):
        df = pd.DataFrame(dict(cl=group_keys, k=k, n=n))
        g = df.groupby('cl')[['k', 'n']].sum()
        return g
    a = agg(clusters1, k1, n1)
    b = agg(clusters2, k2, n2)
    obs = a.k.sum() / a.n.sum() - b.k.sum() / b.n.sum()
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        ia = rng.integers(0, len(a), len(a))
        ib = rng.integers(0, len(b), len(b))
        sa, sb = a.iloc[ia], b.iloc[ib]
        da = sa.k.sum() / sa.n.sum() if sa.n.sum() else np.nan
        db = sb.k.sum() / sb.n.sum() if sb.n.sum() else np.nan
        if np.isfinite(da) and np.isfinite(db):
            boots.append(da - db)
    return obs, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


print('\n\n=== trends (moving-block bootstrap, block=5 checkpoints) ===')
trend = []
for model in cp.model.unique():
    g = cp[cp.model == model]
    for var in ('A_scorable', 'B_cond_error', 'E_confirmed'):
        o, lo, hi, n = moving_block_spearman(g.step, g[var])
        trend.append(dict(model=model, variable=var, n=n,
                          spearman=None if np.isnan(o) else round(o, 3),
                          ci_lo=None if np.isnan(lo) else round(lo, 3),
                          ci_hi=None if np.isnan(hi) else round(hi, 3),
                          excludes_zero=bool(np.isfinite(lo) and np.isfinite(hi)
                                             and (lo > 0 or hi < 0))))
TR = pd.DataFrame(trend)
TR.to_csv(OUT / 'C1_ABE_TRENDS.csv', index=False, encoding='utf-8')
print(TR.to_string(index=False))

# ------------------------------------------------------------------ early vs late B
print('\n=== early vs late conditional error B (cluster bootstrap over subjects) ===')
d['scorable'] = d.audit_label.isin(['correct_verified', 'wrong_confirmed']).astype(int)
d['wrong'] = (d.audit_label == 'wrong_confirmed').astype(int)
comp = []
for model in d.model.unique():
    g = d[d.model == model]
    if g.step.isna().all():
        continue
    early = g[g.step <= 20000]
    late = g[g.step >= 100000]
    if not len(early) or not len(late):
        continue
    e_s, e_w = early.scorable.sum(), early.wrong.sum()
    l_s, l_w = late.scorable.sum(), late.wrong.sum()
    obs, lo, hi = cluster_boot_diff(
        early.wrong.values, early.scorable.values,
        late.wrong.values, late.scorable.values,
        early.subj.values, late.subj.values)
    comp.append(dict(model=model,
                     early_wrong=int(e_w), early_scorable=int(e_s),
                     early_B=round(e_w / e_s, 4) if e_s else np.nan,
                     late_wrong=int(l_w), late_scorable=int(l_s),
                     late_B=round(l_w / l_s, 4) if l_s else np.nan,
                     diff=round(float(obs), 4),
                     ci_lo=round(lo, 4), ci_hi=round(hi, 4)))
CMP = pd.DataFrame(comp)
CMP.to_csv(OUT / 'C1_ABE_EARLY_LATE.csv', index=False, encoding='utf-8')
print(CMP.to_string(index=False))
print(f'\nwrote {OUT}')
