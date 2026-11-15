"""C2 strict forward prediction, CPU only.

This is a transition-level, source-to-target, leave-one-subject-out analysis.
No wrong-answer-derived features enter the primary model.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "next_stage" / "C2_FORWARD_PREDICTION_V1_20261113"
SRC = ROOT / "next_stage" / "c2_c3_phaseI_final_closure_20261113" / "Paper2_PhaseI_Repro" / "inputs" / "model_frame_42_by_5.csv"
EXPECTED_SHA = "9fa4622d5ddda64f1169fcb16e3c44c56e573b68f53cfee037a92fcdc2439ebb"
MAP = {170000: 10000, 340000: 170000, 500000: 340000, 658032: 500000}
TARGETS = [340000, 500000, 658032]
B = ["initial_difficulty", "log_subject_source", "log_answer_source", "log_shared_source"]
E = ["direct_state_source", "log_direct_source", "cooc_state_source", "log_cooc_source"]
U = ["uncertain_state_source"]
P = ["P_source"]
MODELS = {
    "M0": B,
    "M1-E": B + E,
    "M1-Direct": B + ["direct_state_source", "log_direct_source"],
    "M1-Cooc": B + ["cooc_state_source", "log_cooc_source"],
    "M1-EU": B + E + U,
    "M0-P": B + P,
    "M1-E-P": B + E + P,
}
LAM = 1.0
EPS = 1e-15

# ---------------------------------------------------------------- data audit
raw = pd.read_csv(SRC, keep_default_na=False, low_memory=False)
sha = hashlib.sha256(SRC.read_bytes()).hexdigest()
assert sha == EXPECTED_SHA, (sha, EXPECTED_SHA)
num_cols = ["error", "error_upper", "attempts", "correct", "subject_chunks", "answer_chunks",
            "shared_chunks", "initial_difficulty", "direct_fragments", "direct_observed_exposures",
            "cooccurrence_fragments", "cooccurrence_observed_exposures", "uncertain_fragments", "time"]
for c in num_cols:
    raw[c] = pd.to_numeric(raw[c], errors="coerce")
assert len(raw) == 210 and raw.uid.nunique() == 42 and raw.subject.nunique() == 42
assert raw.groupby("uid").fact_id.nunique().max() == 1
assert not raw.duplicated(["uid", "step"]).any()
assert set(raw.step.unique()) == {10000, 170000, 340000, 500000, 658032}
raw["valid"] = raw.attempts.notna() & (raw.attempts > 0) & raw.error.notna()
assert (raw.loc[raw.valid, "attempts"] > 0).all()
assert (raw.loc[raw.valid, "error"] >= 0).all()
assert (raw.loc[raw.valid, "error_upper"] >= raw.loc[raw.valid, "error"]).all()

# Initial difficulty must match the 10k definition.
base = raw[raw.step == 10000].set_index("uid")
calc = 1 - base.correct / base.attempts
max_diff = float(np.nanmax(np.abs(base.initial_difficulty - calc)))
assert max_diff < 1e-9, max_diff

audit = []
for step in sorted(raw.step.unique()):
    g = raw[raw.step == step]
    audit.append(dict(step=step, raw_rows=len(g), valid_outcome=int(g.valid.sum()),
                      missing_outcome=int((~g.valid).sum()),
                      sum_error=float(g.loc[g.valid, "error"].sum()),
                      sum_error_upper=float(g.loc[g.valid, "error_upper"].sum())))
pd.DataFrame(audit).to_csv(OUT / "C2_FORWARD_COVERAGE.csv", index=False, encoding="utf-8")

# ---------------------------------------------------------------- transitions
idx = raw.set_index(["uid", "step"])
rows = []
uids = sorted(raw.uid.unique())
for uid in uids:
    for tgt, src in MAP.items():
        s = idx.loc[(uid, src)]
        t = idx.loc[(uid, tgt)]
        rows.append(dict(
            uid=uid, fact_id=s.fact_id, subject=s.subject,
            source_step=src, target_step=tgt,
            source_time=src / 658032, target_time=tgt / 658032,
            target_y=t.error if t.valid else np.nan,
            target_y_upper=t.error_upper if t.valid else np.nan,
            target_n=t.attempts if t.valid else np.nan,
            target_outcome_available=bool(t.valid),
            missing_reason="" if t.valid else "target attempts/error unavailable",
            source_outcome_y=s.error if s.valid else np.nan,
            source_outcome_n=s.attempts if s.valid else np.nan,
            source_outcome_available=bool(s.valid),
            initial_difficulty=float(base.loc[uid, "initial_difficulty"]),
            log_subject_source=float(np.log1p(s.subject_chunks)),
            log_answer_source=float(np.log1p(s.answer_chunks)),
            log_shared_source=float(np.log1p(s.shared_chunks)),
            direct_state_source=int(s.direct_fragments > 0),
            log_direct_source=float(np.log1p(s.direct_observed_exposures)),
            cooc_state_source=int(s.cooccurrence_fragments > 0),
            log_cooc_source=float(np.log1p(s.cooccurrence_observed_exposures)),
            uncertain_state_source=int(s.uncertain_fragments > 0),
        ))
ALL = pd.DataFrame(rows)
ALL["P_source"] = ALL.source_outcome_y / ALL.source_outcome_n
assert len(ALL) == 168
assert (ALL.source_step < ALL.target_step).all()
assert ALL.groupby(["uid", "target_step"]).size().max() == 1
assert ALL.target_step.map(MAP).notna().all()
for tgt, src in MAP.items():
    assert (ALL.loc[ALL.target_step == tgt, "source_step"] == src).all()
ALL.to_csv(OUT / "C2_FORWARD_TRANSITIONS_ALL.csv", index=False, encoding="utf-8")
VALID = ALL[ALL.target_outcome_available].copy()
assert len(VALID) == 136, len(VALID)
for c in B + E + ["P_source"]:
    VALID[c] = pd.to_numeric(VALID[c], errors="coerce")
assert VALID[B + E].notna().all().all()
VALID.to_csv(OUT / "C2_FORWARD_TRANSITIONS_VALID.csv", index=False, encoding="utf-8")

coverage = []
for tgt, src in MAP.items():
    t = ALL[ALL.target_step == tgt]
    ts = t[t.target_outcome_available]
    coverage.append(dict(target_step=tgt, source_step=src, raw_rows=len(t),
                         target_outcome_valid=len(ts),
                         source_feature_valid=int(ts[B + E].notna().all(axis=1).sum()),
                         source_outcome_valid=int(ts.source_outcome_available.sum()),
                         source_outcome_missing=int((~ts.source_outcome_available).sum()),
                         strict_target_eligible=int(tgt in TARGETS)))
pd.DataFrame(coverage).to_csv(OUT / "C2_FORWARD_COVERAGE.csv", index=False, encoding="utf-8")
print("transitions", len(ALL), "valid", len(VALID), "coverage", coverage)


# ---------------------------------------------------------------- fit
def fit(train, test, features, lam=LAM):
    X = train[features].to_numpy(float)
    Xt = test[features].to_numpy(float)
    mu = X.mean(0)
    sd = X.std(0, ddof=1)
    const = sd == 0
    sd[const] = 1.0
    A = (X - mu) / sd
    At = (Xt - mu) / sd
    A[:, const] = 0.0
    At[:, const] = 0.0
    A = np.column_stack([np.ones(len(A)), A])
    At = np.column_stack([np.ones(len(At)), At])
    y = train.target_y.to_numpy(float)
    n = train.target_n.to_numpy(float)
    pen = np.full(A.shape[1], lam)
    pen[0] = 0.0

    def loss(beta):
        eta = np.clip(A @ beta, -30.0, 30.0)
        nll = np.sum(n * np.logaddexp(0.0, eta) - y * eta)
        penalty = 0.5 * lam * np.sum(beta[1:] ** 2)
        p = expit(eta)
        grad = A.T @ (n * p - y) + pen * beta
        return nll + penalty, grad

    res = minimize(loss, np.zeros(A.shape[1]), jac=True, method="BFGS",
                   options={"gtol": 1e-5, "maxiter": 10000})
    grad = loss(res.x)[1]
    gnorm = float(np.linalg.norm(grad))
    p = expit(np.clip(At @ res.x, -30.0, 30.0))
    return p, dict(optimizer_success=bool(res.success), message=str(res.message),
                   nit=int(res.nit), objective=float(res.fun), gradient_norm=gnorm,
                   converged=bool(np.isfinite(res.fun) and gnorm <= 1e-3),
                   constant_features=";".join(np.array(features)[const]),
                   features_used=";".join(features), train_rows=len(train),
                   train_subjects=int(train.uid.nunique()))


def run_oof(data, models, outcome="target_y", lam=LAM, test_keys=None):
    # Outcome can be target_y or target_y_upper; rows must be valid.
    data = data.copy()
    data["y_fit"] = data[outcome]
    data["n_fit"] = data.target_n
    recs = []
    diags = []
    for tgt in TARGETS:
        test_all = data[(data.target_step == tgt) & data.target_outcome_available].copy()
        if test_keys is not None:
            test_all = test_all[test_all.apply(lambda r: (r.uid, r.target_step) in test_keys, axis=1)]
        for held in sorted(test_all.uid.unique()):
            test = test_all[test_all.uid == held].copy()
            train = data[(data.target_step < tgt) & (data.uid != held) &
                         data.target_outcome_available].copy()
            if not len(train):
                continue
            for name, feats in models.items():
                tr = train.copy()
                if "P_source" in feats:
                    tr = tr[tr.P_source.notna()]
                if not len(tr):
                    continue
                tr["target_y"] = tr.y_fit
                tr["target_n"] = tr.n_fit
                p, d = fit(tr, test, feats, lam=lam)
                diags.append(dict(target_step=tgt, heldout_uid=held, model=name, **d))
                z = test.iloc[0]
                recs.append(dict(uid=z.uid, fact_id=z.fact_id, subject=z.subject,
                                 source_step=z.source_step, target_step=tgt,
                                 target_y=float(z.y_fit), target_y_upper=float(z.target_y_upper),
                                 target_n=float(z.target_n), model_name=name,
                                 predicted_probability=float(p[0]),
                                 fold_train_rows=d["train_rows"],
                                 fold_train_subjects=d["train_subjects"],
                                 heldout_uid=held, lambda_=LAM,
                                 optimizer_success=d["optimizer_success"],
                                 gradient_norm=d["gradient_norm"],
                                 features_used=feats))
    return pd.DataFrame(recs), pd.DataFrame(diags)


def metric(df, name, pred="predicted_probability", y="target_y", n="target_n"):
    g = df[df.model_name == name].copy()
    p = np.clip(g[pred].to_numpy(float), EPS, 1 - EPS)
    yy = g[y].to_numpy(float)
    nn = g[n].to_numpy(float)
    ll_micro = float(-np.sum(yy * np.log(p) + (nn - yy) * np.log(1 - p)) / nn.sum())
    br_micro = float(np.sum(yy * (1 - p) ** 2 + (nn - yy) * p ** 2) / nn.sum())
    subs = []
    for s, x in g.groupby("subject"):
        p2 = np.clip(x[pred].to_numpy(float), EPS, 1 - EPS)
        y2 = x[y].to_numpy(float); n2 = x[n].to_numpy(float)
        subs.append((-np.sum(y2 * np.log(p2) + (n2 - y2) * np.log(1 - p2)) / n2.sum(),
                     np.sum(y2 * (1 - p2) ** 2 + (n2 - y2) * p2 ** 2) / n2.sum()))
    ll_macro = float(np.mean([a for a, _ in subs]))
    br_macro = float(np.mean([b for _, b in subs]))
    ys, ps = [], []
    for yi, ni, pi in zip(yy, nn, p):
        yi, ni = int(round(yi)), int(round(ni))
        ys.append(np.r_[np.ones(yi), np.zeros(ni - yi)])
        ps.append(np.full(ni, pi))
    yy2, pp2 = np.concatenate(ys), np.concatenate(ps)
    ap = float(average_precision_score(yy2, pp2)) if yy2.sum() > 0 else np.nan
    return dict(model=name, n_rows=len(g), n_responses=int(nn.sum()), n_subjects=g.subject.nunique(),
                logloss_micro=ll_micro, logloss_macro=ll_macro,
                brier_micro=br_micro, brier_macro=br_macro, AP=ap)


# Evidence-only rows: 110 (340 42, 500 26, 658 42)
evidence = VALID[VALID.target_step.isin(TARGETS)].copy()
assert len(evidence) == 110, len(evidence)
oof, diags = run_oof(VALID, {k: v for k, v in MODELS.items() if "-P" not in k})
assert oof.groupby(["uid", "target_step", "model_name"]).size().max() == 1
oof.to_csv(OUT / "C2_OOF_PREDICTIONS_FORWARD.csv", index=False, encoding="utf-8")
diags.to_csv(OUT / "C2_FOLD_DIAGNOSTICS.csv", index=False, encoding="utf-8")
metrics = pd.DataFrame([metric(oof, m) for m in [k for k in MODELS if "-P" not in k]])
metrics.to_csv(OUT / "C2_MODEL_COMPARISON_FORWARD.csv", index=False, encoding="utf-8")
print(metrics.to_string(index=False))

# Persistence common subset: 26 complete subjects x 3 targets = 78.
pers = VALID[(VALID.target_step.isin(TARGETS)) & VALID.source_outcome_available].copy()
assert len(pers) == 78, len(pers)
oofp, diagp = run_oof(VALID, {"M0-P": MODELS["M0-P"], "M1-E-P": MODELS["M1-E-P"]}, test_keys=set(zip(pers.uid, pers.target_step)))
oofp.to_csv(OUT / "C2_OOF_PREDICTIONS_FORWARD_PERSISTENCE.csv", index=False, encoding="utf-8")
diagp.to_csv(OUT / "C2_FOLD_DIAGNOSTICS_PERSISTENCE.csv", index=False, encoding="utf-8")
metp = pd.DataFrame([metric(oofp, m) for m in ("M0-P", "M1-E-P")])
metp.to_csv(OUT / "C2_MODEL_COMPARISON_PERSISTENCE.csv", index=False, encoding="utf-8")
print(metp.to_string(index=False))

# Stage-specific point estimates.
stage = []
for tgt in TARGETS:
    for m in [k for k in MODELS if "-P" not in k]:
        x = oof[(oof.target_step == tgt) & (oof.model_name == m)]
        mt = metric(x, m)
        mt["target_step"] = tgt
        stage.append(mt)
pd.DataFrame(stage).to_csv(OUT / "C2_STAGE_SPECIFIC_RESULTS.csv", index=False, encoding="utf-8")

# Paired subject bootstrap: one draw matrix for all models/metrics.
subjects = sorted(oof.subject.unique())
S = len(subjects)
assert S == 42
rng = np.random.default_rng(20261113)
draws = rng.integers(0, S, size=(20000, S))
np.save(OUT / "C2_FORWARD_BOOTSTRAP_DRAWS.npy", draws)
by = {s: oof[oof.subject == s] for s in subjects}
comps = [("M1-E", "M0", "primary M1-E - M0"),
         ("M1-Direct", "M0", "secondary M1-Direct - M0"),
         ("M1-Cooc", "M0", "secondary M1-Cooc - M0"),
         ("M1-EU", "M0", "secondary M1-EU - M0")]


def boot_metric(df, name, metric_name):
    g = df[df.model_name == name]
    p = np.clip(g.predicted_probability.to_numpy(float), EPS, 1 - EPS)
    y = g.target_y.to_numpy(float); n = g.target_n.to_numpy(float)
    if metric_name == "logloss":
        return float(-np.sum(y * np.log(p) + (n - y) * np.log(1 - p)) / n.sum())
    if metric_name == "brier":
        return float(np.sum(y * (1 - p) ** 2 + (n - y) * p ** 2) / n.sum())
    if metric_name == "AP":
        ys, ps = [], []
        for yi, ni, pi in zip(y, n, p):
            yi, ni = int(round(yi)), int(round(ni))
            ys.append(np.r_[np.ones(yi), np.zeros(ni - yi)])
            ps.append(np.full(ni, pi))
        return float(average_precision_score(np.concatenate(ys), np.concatenate(ps)))


rows = []
for met in ("logloss", "brier", "AP"):
    for a, b, label in comps:
        vals = []
        for r in range(20000):
            ind = np.concatenate([np.where(oof.subject.values == subjects[j])[0] for j in draws[r]])
            # direct row selection by subject frame preserves all rows
            sel = pd.concat([by[subjects[j]] for j in draws[r]])
            ma = boot_metric(sel[sel.model_name == a], a, met)
            mb = boot_metric(sel[sel.model_name == b], b, met)
            vals.append(ma - mb)
        vals = np.asarray(vals, float)
        v = vals[np.isfinite(vals)]
        lo, hi = np.percentile(v, [2.5, 97.5])
        obs = boot_metric(oof[oof.model_name == a], a, met) - boot_metric(oof[oof.model_name == b], b, met)
        rows.append(dict(metric=met, comparison=label, observed_delta=float(obs),
                         bootstrap_mean=float(np.mean(v)), CI_low=float(lo), CI_high=float(hi),
                         valid_replicates=len(v), invalid_replicates=20000 - len(v)))
pd.DataFrame(rows).to_csv(OUT / "C2_PAIRED_SUBJECT_BOOTSTRAP.csv", index=False, encoding="utf-8")
print("bootstrap rows", len(rows))

# Macro Log Loss sign-flip, 100k subject signs.
ds = []
for s in subjects:
    g = oof[oof.subject == s]
    a = g[g.model_name == "M1-E"]; b = g[g.model_name == "M0"]
    ds.append(metric(a, "M1-E")["logloss_macro"] - metric(b, "M0")["logloss_macro"])
ds = np.asarray(ds)
Tobs = ds.mean()
rng2 = np.random.default_rng(20261113)
signs = rng2.choice([-1.0, 1.0], size=(100000, len(ds)))
Tb = (signs * ds).mean(1)
p = (1 + np.sum(np.abs(Tb) >= abs(Tobs))) / (100000 + 1)
pd.DataFrame([dict(comparison="M1-E vs M0", metric="logloss_macro", observed=Tobs,
                   p_monte_carlo=float(p), B=100000, seed=20261113,
                   note="subject-level sign-flip; exchangeability/symmetry assumption")]).to_csv(
    OUT / "C2_SIGNFLIP_TESTS.csv", index=False, encoding="utf-8")
print("signflip", Tobs, p)

# Stage point deltas.
st = pd.DataFrame(stage)
piv = st.pivot(index="target_step", columns="model", values="logloss_macro")
piv["M1-E minus M0"] = piv["M1-E"] - piv["M0"]
piv.to_csv(OUT / "C2_STAGE_DELTAS.csv")

# Error upper sensitivity: refit M0/M1-E with upper outcome.
up, upd = run_oof(VALID.assign(target_y=VALID.target_y_upper), {"M0": B, "M1-E": B + E}, outcome="target_y")
# above run_oof uses target_y; passed upper already; need n target_n
up_met = pd.DataFrame([metric(up, m) for m in ("M0", "M1-E")])
up_met["outcome"] = "error_upper"
up_met.to_csv(OUT / "C2_ERROR_UPPER_SENSITIVITY.csv", index=False, encoding="utf-8")

# Lambda sensitivity for M0/M1-E.
lam_rows = []
for lam in (0.1, 1.0, 10.0):
    o, _ = run_oof(VALID, {"M0": B, "M1-E": B + E}, lam=lam)
    for m in ("M0", "M1-E"):
        x = metric(o, m); x["lambda"] = lam; lam_rows.append(x)
pd.DataFrame(lam_rows).to_csv(OUT / "C2_LAMBDA_SENSITIVITY.csv", index=False, encoding="utf-8")

# Leave-one-subject-out diagnostics of observed OOF deltas.
loo = []
for drop in subjects:
    sub = oof[oof.subject != drop]
    a = metric(sub, "M1-E")["logloss_macro"]
    b = metric(sub, "M0")["logloss_macro"]
    loo.append(dict(dropped_subject=drop, n_rows=len(sub), delta_macro=a - b))
loo = pd.DataFrame(loo)
loo.to_csv(OUT / "C2_LEAVE_ONE_SUBJECT_OUT_DIAGNOSTICS.csv", index=False, encoding="utf-8")

# Data audit / final report.
data_audit = f"""# C2 Data Audit

- input SHA256: `{sha}` (matches required `{EXPECTED_SHA}`)
- rows: {len(raw)}; unique uid: {raw.uid.nunique()}; unique subject: {raw.subject.nunique()}
- steps: {sorted(raw.step.unique())}
- each uid/fact_id: one fact; `(uid, step)` unique
- initial_difficulty equals `1 - correct/attempts` at 10k; max abs difference {max_diff:.3e}
- missing outcomes are missing attempts/error rows, not zero errors.

## Coverage

{ pd.DataFrame(coverage).to_string(index=False) }

## Transition counts

- all candidate transitions: {len(ALL)} (42 x 4)
- valid target outcome: {len(VALID)} (136)
- strict forward test targets: {TARGETS}; test rows: {len(evidence)} (110)
- persistence common test rows: {len(pers)} (78, 26 subjects x 3 stages)
"""
(OUT / "C2_DATA_AUDIT.md").write_text(data_audit, encoding="utf-8")
(OUT / "C2_DATA_AUDIT.md").write_text(data_audit, encoding="utf-8")
# Feature dictionary.
fd = []
for group, fs in (("B", B), ("E", E), ("U", U), ("P", P)):
    for f in fs:
        fd.append(dict(group=group, feature=f))
pd.DataFrame(fd).to_csv(OUT / "C2_FEATURE_DICTIONARY.csv", index=False, encoding="utf-8")

report = f"""# C2 Forward Prediction 鈥?Final Report

## What was executed

Strict transition-level C2, CPU only: source checkpoint features predict the next
checkpoint's confirmed-error outcome.  No wrong-answer-derived features enter the
primary model.  The 136 valid target-outcome transitions are not 136 independent
facts; the independent unit is uid/subject.

## Sample accounting

- 210 raw rows = 42 uid x 5 steps
- 168 candidate transitions = 42 x 4
- 136 transitions with valid target outcome
- 110 strict forward test transitions (340k 42, 500k 26, 658k 42)
- 78 persistence-subset test transitions (26 subjects x 3 stages)
- 170k is warm-up and is not in the primary OOF score

## Models

Primary comparison: M1-E vs M0, subject-macro Log Loss.
Secondary: M1-Direct, M1-Cooc, M1-EU vs M0 (Holm family of three).
Persistence: M1-E-P vs M0-P on the 78-row common subset (separate secondary).

The point estimates are in `C2_MODEL_COMPARISON_FORWARD.csv`; stage results in
`C2_STAGE_SPECIFIC_RESULTS.csv`; bootstrap in
`C2_PAIRED_SUBJECT_BOOTSTRAP.csv`; sign-flip in `C2_SIGNFLIP_TESTS.csv`.

## Decision-rule boundary

The interpretation must follow the pre-specified rules:
- A only if the primary subject-macro Log Loss difference is negative and its
  paired subject bootstrap CI is entirely below zero, with no severe optimizer
  failure and no single-subject dependence.
- If the point estimate improves but the CI contains zero: trend, not established
  incremental value.
- If the direction is unstable: do not claim evidence.

All results are conditional on the fitted outer LOSO models; the bootstrap does
not refit the prediction algorithm.  The old contemporaneous `M0+E` results in
`PREDICTIONS_ALL_MODELS.csv` are a different estimand and are not merged here.

## Limits

- 42 facts, 110 forward rows; low power.
- Predictor features are source-checkpoint features; Evidence is not a full
  t-1-only feature set.
- Wrong-other/wrong-cooc features are excluded from the primary model because
  they are outcome-dependent.
- No model generation, GPU, or new facts.
"""
(OUT / "C2_FINAL_REPORT.md").write_text(report, encoding="utf-8")
(OUT / "RUN_LOG.md").write_text("# C2 Forward RUN LOG\n\nCPU-only execution completed; see outputs.\n", encoding="utf-8")
print("DONE", OUT)

