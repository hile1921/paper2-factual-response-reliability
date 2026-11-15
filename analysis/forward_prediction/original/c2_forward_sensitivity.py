"""C2 forward sensitivity: upper-bound outcome and lambda grid."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import average_precision_score
from pathlib import Path

OUT = Path(__file__).resolve().parent
V = pd.read_csv(OUT / "C2_FORWARD_TRANSITIONS_VALID.csv", keep_default_na=False)
B = ["initial_difficulty", "log_subject_source", "log_answer_source", "log_shared_source"]
E = ["direct_state_source", "log_direct_source", "cooc_state_source", "log_cooc_source"]
TARGETS = [340000, 500000, 658032]
EPS = 1e-15


def fit(train, test, feats, lam):
    X = train[feats].to_numpy(float); Xt = test[feats].to_numpy(float)
    mu = X.mean(0); sd = X.std(0, ddof=1); const = sd == 0; sd[const] = 1
    A = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    At = np.column_stack([np.ones(len(Xt)), (Xt - mu) / sd])
    A[:, np.where(const)[0] + 1] = 0; At[:, np.where(const)[0] + 1] = 0
    y = train.target_y.to_numpy(float); n = train.target_n.to_numpy(float)
    pen = np.full(A.shape[1], lam); pen[0] = 0

    def f(b):
        eta = np.clip(A @ b, -30, 30)
        nll = np.sum(n * np.logaddexp(0, eta) - y * eta)
        p = expit(eta)
        return nll + 0.5 * lam * np.sum(b[1:] ** 2), A.T @ (n * p - y) + pen * b
    r = minimize(f, np.zeros(A.shape[1]), jac=True, method="BFGS", options={"gtol": 1e-5, "maxiter": 10000})
    return expit(np.clip(At @ r.x, -30, 30)), bool(r.success), float(np.linalg.norm(f(r.x)[1]))


def run(data, feats, lam):
    rec = []
    for tgt in TARGETS:
        test = data[(data.target_step == tgt) & data.target_outcome_available]
        for uid in sorted(test.uid.unique()):
            te = test[test.uid == uid]
            tr = data[(data.target_step < tgt) & (data.uid != uid) & data.target_outcome_available]
            if not len(tr) or not len(te):
                continue
            p, ok, gn = fit(tr, te, feats, lam)
            rec.append(dict(uid=uid, target_step=tgt, y=float(te.target_y.iloc[0]),
                            n=float(te.target_n.iloc[0]), p=float(p[0]), ok=ok, gn=gn))
    return pd.DataFrame(rec)


def met(d):
    y = d.y.to_numpy(float); n = d.n.to_numpy(float); p = np.clip(d.p.to_numpy(float), EPS, 1 - EPS)
    ll = -np.sum(y * np.log(p) + (n - y) * np.log(1 - p)) / n.sum()
    br = np.sum(y * (1 - p) ** 2 + (n - y) * p ** 2) / n.sum()
    ys, ps = [], []
    for yi, ni, pi in zip(y, n, p):
        yi, ni = int(round(yi)), int(round(ni))
        ys.append(np.r_[np.ones(yi), np.zeros(ni - yi)]); ps.append(np.full(ni, pi))
    yy, pp = np.concatenate(ys), np.concatenate(ps)
    ap = average_precision_score(yy, pp) if yy.sum() else np.nan
    return float(ll), float(br), float(ap)

# Error upper-bound.
up = V.copy(); up["target_y"] = up.target_y_upper
rows = []
for name, feats in (("M0", B), ("M1-E", B + E)):
    d = run(up, feats, 1.0)
    ll, br, ap = met(d)
    rows.append(dict(model=name, outcome="error_upper", n_rows=len(d),
                     logloss_micro=ll, brier_micro=br, AP=ap,
                     optimizer_failures=int((~d.ok).sum()),
                     max_gradient_norm=float(d.gn.max())))
pd.DataFrame(rows).to_csv(OUT / "C2_ERROR_UPPER_SENSITIVITY.csv", index=False, encoding="utf-8")

# Lambda grid.
rows = []
for lam in (0.1, 1.0, 10.0):
    for name, feats in (("M0", B), ("M1-E", B + E)):
        d = run(V, feats, lam)
        ll, br, ap = met(d)
        rows.append(dict(lambda_=lam, model=name, n_rows=len(d),
                         logloss_micro=ll, brier_micro=br, AP=ap,
                         optimizer_failures=int((~d.ok).sum()),
                         max_gradient_norm=float(d.gn.max())))
pd.DataFrame(rows).to_csv(OUT / "C2_LAMBDA_SENSITIVITY.csv", index=False, encoding="utf-8")
print(pd.DataFrame(rows).to_string(index=False))
