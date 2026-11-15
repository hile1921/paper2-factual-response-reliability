"""Post-processing for frozen C2 OOF: vectorized bootstrap, sign-flip, report."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

OUT = Path(__file__).resolve().parent
OOF = pd.read_csv(OUT / "C2_OOF_PREDICTIONS_FORWARD.csv", keep_default_na=False)
OOF["predicted_probability"] = pd.to_numeric(OOF.predicted_probability, errors="coerce")
OOF["target_y"] = pd.to_numeric(OOF.target_y, errors="coerce")
OOF["target_n"] = pd.to_numeric(OOF.target_n, errors="coerce")
assert OOF.predicted_probability.notna().all()
subjects = sorted(OOF.subject.unique())
S = len(subjects)
assert S == 42
draws = np.load(OUT / "C2_FORWARD_BOOTSTRAP_DRAWS.npy")
assert draws.shape == (20000, 42)
EPS = 1e-15


def observed_metrics(df, model, metric):
    g = df[df.model_name == model]
    p = np.clip(g.predicted_probability.to_numpy(float), EPS, 1 - EPS)
    y = g.target_y.to_numpy(float); n = g.target_n.to_numpy(float)
    if metric == "logloss":
        sub = {}
        for s, x in g.groupby("subject"):
            pp = np.clip(x.predicted_probability.to_numpy(float), EPS, 1 - EPS)
            yy = x.target_y.to_numpy(float); nn = x.target_n.to_numpy(float)
            sub[s] = (-np.sum(yy * np.log(pp) + (nn - yy) * np.log(1 - pp)), nn.sum())
        num = np.array([sub[s][0] for s in subjects]); den = np.array([sub[s][1] for s in subjects])
        return float(num.sum() / den.sum()), float(np.mean(num / den)), (num, den)
    if metric == "brier":
        sub = {}
        for s, x in g.groupby("subject"):
            pp = np.clip(x.predicted_probability.to_numpy(float), EPS, 1 - EPS)
            yy = x.target_y.to_numpy(float); nn = x.target_n.to_numpy(float)
            sub[s] = (np.sum(yy * (1 - pp) ** 2 + (nn - yy) * pp ** 2), nn.sum())
        num = np.array([sub[s][0] for s in subjects]); den = np.array([sub[s][1] for s in subjects])
        return float(num.sum() / den.sum()), float(np.mean(num / den)), (num, den)
    if metric == "AP":
        ys, ps = [], []
        for yi, ni, pi in zip(y, n, p):
            yi, ni = int(round(yi)), int(round(ni))
            ys.append(np.r_[np.ones(yi), np.zeros(ni - yi)])
            ps.append(np.full(ni, pi))
        yy, pp = np.concatenate(ys), np.concatenate(ps)
        return (float(average_precision_score(yy, pp)) if yy.sum() else np.nan), np.nan, None
    raise ValueError(metric)


# Precompute subject sufficient arrays for fast LL/Brier.
def prep(model):
    g = OOF[OOF.model_name == model]
    s = {x: g[g.subject == x] for x in subjects}
    ll_num = np.zeros(S); br_num = np.zeros(S); nn = np.zeros(S)
    expands = []
    for i, sub in enumerate(subjects):
        x = s[sub]
        p = np.clip(x.predicted_probability.to_numpy(float), EPS, 1 - EPS)
        y = x.target_y.to_numpy(float); n = x.target_n.to_numpy(float)
        ll_num[i] = -np.sum(y * np.log(p) + (n - y) * np.log(1 - p))
        br_num[i] = np.sum(y * (1 - p) ** 2 + (n - y) * p ** 2)
        nn[i] = n.sum()
        ys, ps = [], []
        for yi, ni, pi in zip(y, n, p):
            yi, ni = int(round(yi)), int(round(ni))
            ys.append(np.r_[np.ones(yi), np.zeros(ni - yi)])
            ps.append(np.full(ni, pi))
        expands.append((np.concatenate(ys), np.concatenate(ps)))
    return dict(ll=ll_num, br=br_num, n=nn, exp=expands)


models = ["M0", "M1-E", "M1-Direct", "M1-Cooc", "M1-EU"]
P = {m: prep(m) for m in models}
comps = [("M1-E", "M0", "primary M1-E - M0"),
         ("M1-Direct", "M0", "secondary M1-Direct - M0"),
         ("M1-Cooc", "M0", "secondary M1-Cooc - M0"),
         ("M1-EU", "M0", "secondary M1-EU - M0")]
rows = []
for met, col in (("logloss", "ll"), ("brier", "br")):
    for a, b, label in comps:
        pa, pb = P[a], P[b]
        num_a = pa[col][draws].sum(1); den_a = pa["n"][draws].sum(1)
        num_b = pb[col][draws].sum(1); den_b = pb["n"][draws].sum(1)
        sign = 1
        mic = sign * num_a / den_a - sign * num_b / den_b
        ma = sign * (pa[col] / pa["n"])[draws].mean(1) - sign * (pb[col] / pb["n"])[draws].mean(1)
        for kind, arr, obs in (("micro", mic,
                                 sign * pa[col].sum() / pa["n"].sum() - sign * pb[col].sum() / pb["n"].sum()),
                               ("macro", ma,
                                 sign * np.mean(pa[col] / pa["n"]) - sign * np.mean(pb[col] / pb["n"]))):
            v = arr[np.isfinite(arr)]
            lo, hi = np.percentile(v, [2.5, 97.5])
            rows.append(dict(metric=met, comparison=label, aggregation=kind,
                             observed_delta=float(obs), bootstrap_mean=float(np.mean(v)),
                             CI_low=float(lo), CI_high=float(hi),
                             valid_replicates=len(v), invalid_replicates=20000 - len(v)))
# AP bootstrap only for primary comparison (secondary AP point estimates are retained).
for a, b, label in comps[:1]:
    vals = []
    ea, eb = P[a]["exp"], P[b]["exp"]
    for r in range(20000):
        ys = []; ps = []; ys2 = []; ps2 = []
        for j in draws[r]:
            ys.append(ea[j][0]); ps.append(ea[j][1]); ys2.append(eb[j][0]); ps2.append(eb[j][1])
        y1, p1 = np.concatenate(ys), np.concatenate(ps)
        y2, p2 = np.concatenate(ys2), np.concatenate(ps2)
        if y1.sum() == 0 or y2.sum() == 0:
            vals.append(np.nan); continue
        vals.append(average_precision_score(y1, p1) - average_precision_score(y2, p2))
    vals = np.asarray(vals, float); v = vals[np.isfinite(vals)]
    obs = observed_metrics(OOF, a, "AP")[0] - observed_metrics(OOF, b, "AP")[0]
    rows.append(dict(metric="AP", comparison=label, aggregation="micro", observed_delta=float(obs),
                     bootstrap_mean=float(np.mean(v)), CI_low=float(np.percentile(v, 2.5)),
                     CI_high=float(np.percentile(v, 97.5)), valid_replicates=len(v),
                     invalid_replicates=20000 - len(v)))
R = pd.DataFrame(rows)
R.to_csv(OUT / "C2_PAIRED_SUBJECT_BOOTSTRAP.csv", index=False, encoding="utf-8")

def macro_ll(df, model):
    g = df[df.model_name == model]
    vals = []
    for _, x in g.groupby("subject"):
        p = np.clip(x.predicted_probability.to_numpy(float), EPS, 1 - EPS)
        y = x.target_y.to_numpy(float); n = x.target_n.to_numpy(float)
        vals.append(-np.sum(y * np.log(p) + (n - y) * np.log(1 - p)) / n.sum())
    return float(np.mean(vals))


# Sign-flip on subject-macro Log Loss for primary.
d = []
for s in subjects:
    d.append(macro_ll(OOF[OOF.subject == s], "M1-E") - macro_ll(OOF[OOF.subject == s], "M0"))
d = np.asarray(d)
T = d.mean()
rng = np.random.default_rng(20261113)
signs = rng.choice([-1.0, 1.0], size=(100000, S))
p = (1 + np.sum(np.abs((signs * d).mean(1)) >= abs(T))) / 100001
pd.DataFrame([dict(comparison="M1-E vs M0", metric="logloss_macro", observed=float(T),
                   p_monte_carlo=float(p), B=100000, seed=20261113,
                   note="subject-level sign-flip; exchangeability/symmetry assumption")]).to_csv(
    OUT / "C2_SIGNFLIP_TESTS.csv", index=False, encoding="utf-8")

# Leave-one-subject-out diagnostics.
loo = []
for drop in subjects:
    sub = OOF[OOF.subject != drop]
    loo.append(dict(dropped_subject=drop, n_rows=len(sub),
                    delta_logloss_macro=macro_ll(sub, "M1-E") - macro_ll(sub, "M0")))
pd.DataFrame(loo).to_csv(OUT / "C2_LEAVE_ONE_SUBJECT_OUT_DIAGNOSTICS.csv", index=False, encoding="utf-8")

print(R.to_string(index=False))
print("signflip", T, p)
