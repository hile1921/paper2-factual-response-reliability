"""H-G: standalone C2 subject-cluster bootstrap from the raw predictions."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "next_stage" / "c2_c3_phaseH_closure_20261113"
PRED = ROOT / "next_stage" / "c2_final_audit" / "PREDICTIONS_ALL_MODELS.csv"
B = 20000
SEED = 20261113
EPS = 1e-15
MODELS = {"M0": "M0", "M0+P": "M0+P", "M0+P+E": "M0+P+evidence",
          "M0+E": "M1 = M0+evidence (primary)"}
COMPARISONS = [("M0+E", "M0", "primary  M0+E - M0"),
               ("M0+P+E", "M0+P", "secondary M0+P+E - M0+P"),
               ("M0+P", "M0", "secondary M0+P - M0"),
               ("M0+P+E", "M0+E", "secondary M0+P+E - M0+E")]

df = pd.read_csv(PRED, keep_default_na=False, low_memory=False)
for c in ("y", "n", "p"):
    df[c] = pd.to_numeric(df[c], errors="coerce")
subjects = sorted(df.subject.unique())
S = len(subjects)
sidx = {s: i for i, s in enumerate(subjects)}
assert S == 42


def ll_vec(y, n, p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y, n = np.asarray(y, float), np.asarray(n, float)
    return -np.sum(y * np.log(p) + (n - y) * np.log(1 - p)) / np.sum(n)


def br_vec(y, n, p):
    y, n, p = np.asarray(y, float), np.asarray(n, float), np.asarray(p, float)
    return np.sum(y * (1 - p) ** 2 + (n - y) * p ** 2) / np.sum(n)


# Per-subject sufficient statistics and expanded arrays.
LL = {}; BR = {}; NN = {}; EXP = {}
for short, full in MODELS.items():
    g = df[df.model == full]
    a = np.zeros(S); b = np.zeros(S); nn = np.zeros(S); exp = [None] * S
    for s, gg in g.groupby("subject"):
        i = sidx[s]
        y = gg.y.values.astype(float); n = gg.n.values.astype(float)
        p = np.clip(gg.p.values.astype(float), EPS, 1 - EPS)
        a[i] = np.sum(y * np.log(p) + (n - y) * np.log(1 - p))
        b[i] = np.sum(y * (1 - p) ** 2 + (n - y) * p ** 2)
        nn[i] = n.sum()
        ys, ps = [], []
        for yi, ni, pi in zip(y, n, p):
            yi, ni = int(round(yi)), int(round(ni))
            ys.append(np.r_[np.ones(yi), np.zeros(ni - yi)])
            ps.append(np.full(ni, pi))
        exp[i] = (np.concatenate(ys), np.concatenate(ps))
    LL[short], BR[short], NN[short], EXP[short] = a, b, nn, exp

rng = np.random.default_rng(SEED)
draws = rng.integers(0, S, size=(B, S))
np.save(OUT / "C2_FULL_SAMPLE_BOOTSTRAP_DRAWS.npy", draws)

rows = []
for met in ("logloss", "brier", "AP"):
    for am, bm, label in COMPARISONS:
        if met in ("logloss", "brier"):
            num_a = (LL if met == "logloss" else BR)[am]
            num_b = (LL if met == "logloss" else BR)[bm]
            raw_a = num_a[draws].sum(1); raw_b = num_b[draws].sum(1)
            den_a = NN[am][draws].sum(1); den_b = NN[bm][draws].sum(1)
            sign = -1.0 if met == "logloss" else 1.0
            micro = sign * raw_a / den_a - sign * raw_b / den_b
            per_a = sign * num_a / NN[am]; per_b = sign * num_b / NN[bm]
            macro = per_a[draws].mean(1) - per_b[draws].mean(1)
            obs_micro = sign * num_a.sum() / NN[am].sum() - sign * num_b.sum() / NN[bm].sum()
            obs_macro = per_a.mean() - per_b.mean()
            for kind, arr, obs in (("micro", micro, obs_micro), ("macro", macro, obs_macro)):
                v = arr[np.isfinite(arr)]
                lo, hi = np.percentile(v, [2.5, 97.5])
                rows.append(dict(metric=met, comparison=label, aggregation=kind,
                                 observed_delta=round(float(obs), 9),
                                 bootstrap_mean_delta=round(float(np.mean(v)), 9),
                                 bootstrap_bias=round(float(np.mean(v) - obs), 9),
                                 CI_low=round(float(lo), 9), CI_high=round(float(hi), 9),
                                 valid_replicates=len(v), invalid_replicates=B - len(v)))
        else:
            ea, eb = EXP[am], EXP[bm]
            vals = np.empty(B)
            for r in range(B):
                ind = draws[r]
                ya = np.concatenate([ea[i][0] for i in ind])
                pa = np.concatenate([ea[i][1] for i in ind])
                yb = np.concatenate([eb[i][0] for i in ind])
                pb = np.concatenate([eb[i][1] for i in ind])
                vals[r] = average_precision_score(ya, pa) - average_precision_score(yb, pb)
            # observed pooled AP
            yya = np.concatenate([ea[i][0] for i in range(S)])
            ppa = np.concatenate([ea[i][1] for i in range(S)])
            yyb = np.concatenate([eb[i][0] for i in range(S)])
            ppb = np.concatenate([eb[i][1] for i in range(S)])
            obs = average_precision_score(yya, ppa) - average_precision_score(yyb, ppb)
            v = vals[np.isfinite(vals)]
            lo, hi = np.percentile(v, [2.5, 97.5])
            rows.append(dict(metric="AP", comparison=label, aggregation="micro",
                             observed_delta=round(float(obs), 9),
                             bootstrap_mean_delta=round(float(np.mean(v)), 9),
                             bootstrap_bias=round(float(np.mean(v) - obs), 9),
                             CI_low=round(float(lo), 9), CI_high=round(float(hi), 9),
                             valid_replicates=len(v), invalid_replicates=B - len(v)))

R = pd.DataFrame(rows)
R.to_csv(OUT / "C2_FULL_SAMPLE_BOOTSTRAP_REPRODUCED.csv", index=False, encoding="utf-8")
print(R.to_string(index=False))
print(json.dumps({"subjects": S, "draws": str(OUT / "C2_FULL_SAMPLE_BOOTSTRAP_DRAWS.npy"),
                  "rows": len(R)}, indent=2))
