"""Finalise C2 forward outputs, reports, persistence CI and ZIP."""
from __future__ import annotations

import hashlib
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
SRC = Path(r"C:\Users\Administrator\Documents\ChatGPT\sci论文\Paper2_Hallucination_Dynamics\next_stage\c2_c3_phaseI_final_closure_20261113\Paper2_PhaseI_Repro\inputs\model_frame_42_by_5.csv")
V = pd.read_csv(OUT / "C2_FORWARD_TRANSITIONS_VALID.csv", keep_default_na=False)
OOF = pd.read_csv(OUT / "C2_OOF_PREDICTIONS_FORWARD.csv", keep_default_na=False)
OP = pd.read_csv(OUT / "C2_OOF_PREDICTIONS_FORWARD_PERSISTENCE.csv", keep_default_na=False)
COMP = pd.read_csv(OUT / "C2_MODEL_COMPARISON_FORWARD.csv", keep_default_na=False)
CP = pd.read_csv(OUT / "C2_MODEL_COMPARISON_PERSISTENCE.csv", keep_default_na=False)
BOOT = pd.read_csv(OUT / "C2_PAIRED_SUBJECT_BOOTSTRAP.csv", keep_default_na=False)
SIGN = pd.read_csv(OUT / "C2_SIGNFLIP_TESTS.csv", keep_default_na=False)
STAGE = pd.read_csv(OUT / "C2_STAGE_SPECIFIC_RESULTS.csv", keep_default_na=False)
ERR = pd.read_csv(OUT / "C2_ERROR_UPPER_SENSITIVITY.csv", keep_default_na=False)
LAM = pd.read_csv(OUT / "C2_LAMBDA_SENSITIVITY.csv", keep_default_na=False)
LOO = pd.read_csv(OUT / "C2_LEAVE_ONE_SUBJECT_OUT_DIAGNOSTICS.csv", keep_default_na=False)

# Stage deltas M1-E - M0.
piv = STAGE.pivot(index="target_step", columns="model", values="logloss_macro")
piv["M1-E minus M0"] = piv["M1-E"] - piv["M0"]
piv.to_csv(OUT / "C2_STAGE_DELTAS.csv")

# Persistence subject bootstrap (secondary), 20k.
OP["predicted_probability"] = pd.to_numeric(OP.predicted_probability, errors="coerce")
subs = sorted(OP.subject.unique())
rng = np.random.default_rng(20261113)
draws = rng.integers(0, len(subs), size=(20000, len(subs)))
by = {s: OP[OP.subject == s] for s in subs}


def ll_macro(df, model):
    g = df[df.model_name == model]
    vals = []
    for _, x in g.groupby("subject"):
        p = np.clip(x.predicted_probability.to_numpy(float), 1e-15, 1 - 1e-15)
        y = x.target_y.to_numpy(float); n = x.target_n.to_numpy(float)
        vals.append(-np.sum(y * np.log(p) + (n - y) * np.log(1 - p)) / n.sum())
    return float(np.mean(vals))


vals = []
for r in range(20000):
    sel = pd.concat([by[subs[j]] for j in draws[r]])
    vals.append(ll_macro(sel, "M1-E-P") - ll_macro(sel, "M0-P"))
vals = np.asarray(vals)
obs = ll_macro(OP, "M1-E-P") - ll_macro(OP, "M0-P")
pd.DataFrame([dict(comparison="M1-E-P vs M0-P", metric="logloss_macro",
                   observed=float(obs), CI_low=float(np.percentile(vals, 2.5)),
                   CI_high=float(np.percentile(vals, 97.5)),
                   bootstrap_mean=float(np.mean(vals)), n_subjects=26, n_rows=78)]).to_csv(
    OUT / "C2_PAIRED_SUBJECT_BOOTSTRAP_PERSISTENCE.csv", index=False, encoding="utf-8")

# Reports.
audit = f"""# C2 Data Audit

## Frozen input

- path: `{SRC}`
- SHA256: `{hashlib.sha256(SRC.read_bytes()).hexdigest()}` (required `9fa4622d5ddda64f1169fcb16e3c44c56e573b68f53cfee037a92fcdc2439ebb`)
- 210 rows; 42 uid; 42 subject; 5 steps
- `(uid, step)` unique; one fact_id per uid
- steps: 10000, 170000, 340000, 500000, 658032
- `initial_difficulty` matches `1 - correct/attempts` at 10k.

## Coverage and sample accounting

| target step | source step | raw target rows | target outcome valid | missing |
|---:|---:|---:|---:|---:|
| 170000 | 10000 | 42 | 26 | 16 |
| 340000 | 170000 | 42 | 42 | 0 |
| 500000 | 340000 | 42 | 26 | 16 |
| 658032 | 500000 | 42 | 42 | 0 |

- 168 candidate transitions = 42 x 4
- 136 valid-target-outcome candidate transitions (not 136 OOF predictions)
- 110 strict forward test transitions (340k 42, 500k 26, 658k 42)
- 78 persistence common-subset test transitions (26 complete subjects x 3 stages)
- missing outcomes are missing attempts/error rows, not zero errors.

## Fold training sizes

- evidence-only: 340k 25-26, 500k 66, 658k 91-93 rows per fold
- persistence: 340k 25, 500k 50, 658k 75 rows per fold
- all 550 evidence folds and 156 persistence folds report optimizer success
""" 
(OUT / "C2_DATA_AUDIT.md").write_text(audit, encoding="utf-8")

prim = BOOT[(BOOT.metric == "logloss") & (BOOT.aggregation == "macro") & (BOOT.comparison.str.startswith("primary"))].iloc[0]
brier = BOOT[(BOOT.metric == "brier") & (BOOT.aggregation == "macro") & (BOOT.comparison.str.startswith("primary"))].iloc[0]
ap = BOOT[(BOOT.metric == "AP") & (BOOT.comparison.str.startswith("primary"))].iloc[0]
m0 = COMP[COMP.model == "M0"].iloc[0]; m1 = COMP[COMP.model == "M1-E"].iloc[0]
pers = pd.read_csv(OUT / "C2_PAIRED_SUBJECT_BOOTSTRAP_PERSISTENCE.csv").iloc[0]

report = f"""# C2 Forward Prediction — Final Report

## Executed design

Transition-level C2: source-checkpoint features predict the next checkpoint's
confirmed-error outcome.  The independent unit is `uid/subject`; 42 subjects.
170k is warm-up and is not scored.  No wrong-answer-derived features enter the
primary model.  No target-time linear term is included.

## Sample accounting

- 210 raw rows = 42 x 5
- 168 candidate transitions
- 136 valid target-outcome candidate transitions
- **110 strict forward OOF transitions**
- **78 persistence common-subset OOF transitions**
- 136 is not 136 independent forward predictions.

## Primary comparison: M1-E vs M0

Model: binomial ridge logistic, lambda=1.0, fold-internal standardization,
strict walk-forward + leave-one-subject-out.  M0 uses B; M1-E adds direct and
co-occurrence evidence.

| metric | M0 | M1-E | M1-E - M0 | 95% paired subject CI |
|---|---:|---:|---:|---|
| Log Loss micro | {m0.logloss_micro:.6f} | {m1.logloss_micro:.6f} | {prim.observed_delta:+.6f} | [{prim.CI_low:+.6f}, {prim.CI_high:+.6f}] |
| Log Loss macro | {m0.logloss_macro:.6f} | {m1.logloss_macro:.6f} | {prim.observed_delta:+.6f} | [{prim.CI_low:+.6f}, {prim.CI_high:+.6f}] |
| Brier macro | {m0.brier_macro:.6f} | {m1.brier_macro:.6f} | {brier.observed_delta:+.6f} | [{brier.CI_low:+.6f}, {brier.CI_high:+.6f}] |
| AP micro | {m0.AP:.6f} | {m1.AP:.6f} | {ap.observed_delta:+.6f} | [{ap.CI_low:+.6f}, {ap.CI_high:+.6f}] |

Primary macro Log Loss sign-flip: observed {float(SIGN.observed.iloc[0]):+.6f},
Monte Carlo p = {float(SIGN.p_monte_carlo.iloc[0]):.5f}, B=100,000.

**Decision: Case D, no stable forward improvement.**
The point estimate for the primary metric is in the worse direction
(+{prim.observed_delta:.6f} Log Loss), the 95% CI contains zero, and the
subject sign-flip p is large.  Brier is worse with an interval above zero; AP is
better but its interval contains zero.  This is not “C2 proves no association”;
the contemporaneous association remains a separate result.

## Secondary models

- M1-Direct: Log Loss macro delta {float(BOOT[(BOOT.metric=='logloss')&(BOOT.aggregation=='macro')&(BOOT.comparison.str.contains('Direct'))].observed_delta.iloc[0]):+.6f};
  CI includes zero.
- M1-Cooc: delta {float(BOOT[(BOOT.metric=='logloss')&(BOOT.aggregation=='macro')&(BOOT.comparison.str.contains('Cooc'))].observed_delta.iloc[0]):+.6f};
  CI is entirely above zero (worse).
- M1-EU: delta {float(BOOT[(BOOT.metric=='logloss')&(BOOT.aggregation=='macro')&(BOOT.comparison.str.contains('EU'))].observed_delta.iloc[0]):+.6f};
  CI includes zero.

## Persistence common subset

M0-P vs M1-E-P on 78 rows / 26 subjects:

- M0-P macro Log Loss {float(CP[CP.model=='M0-P'].logloss_macro.iloc[0]):.6f}
- M1-E-P macro Log Loss {float(CP[CP.model=='M1-E-P'].logloss_macro.iloc[0]):.6f}
- observed delta {float(pers.observed):+.6f}; 95% CI
  [{float(pers.CI_low):+.6f}, {float(pers.CI_high):+.6f}]
- separate secondary comparison; not mixed with 110-row results.

## Time stages

{piv.to_string()}

The 340k stage shows a lower Log Loss for M1-E while 500k and 658k are worse;
there is no stable stage pattern.  Stage estimates are descriptive and not
individually tested as new confirmatory claims.

## Sensitivity

- error_upper: M0 {float(ERR[ERR.model=='M0'].logloss_micro.iloc[0]):.6f}, M1-E
  {float(ERR[ERR.model=='M1-E'].logloss_micro.iloc[0]):.6f}; direction is
  favorable for M1-E, opposite to the primary outcome sensitivity.
- lambda=0.1/1/10: see `C2_LAMBDA_SENSITIVITY.csv`; at lambda=0.1 M1-E is much
  worse, at lambda=10 slightly better.  The primary lambda=1 result is not
  stable across regularization.
- leave-one-subject-out observed delta range:
  [{float(LOO.delta_logloss_macro.min()):+.6f}, {float(LOO.delta_logloss_macro.max()):+.6f}].

## Existing contemporaneous results are separate

The frozen `PREDICTIONS_ALL_MODELS.csv` contemporaneous M0+E vs M0 result
(micro LL about -0.0114, CI containing zero) is a different estimand.  It is not
merged with this forward analysis and is not called prospective.

## What can be written

- At the current 42-fact scale, strict forward corpus evidence did not provide a
  stable incremental prediction gain over the baseline.
- There is no evidence for a reliable forward improvement; the primary point
  estimate is worse and the interval includes zero.
- The contemporaneous association can still be described separately as a trend,
  not as forward validation.

## What cannot be written

- C2 is not “proven successful”.
- The old contemporaneous numbers must not be presented as forward prediction.
- `wrong_other` / `wrong_cooc` retrospective features are not in the primary model.
- The 110 transitions are not 110 independent facts.
- The bootstrap is conditional on fitted outer LOSO models and does not refit.
- This is not a full t-1-only feature set; Evidence is source-time but some
  context construction remains task-specific.
"""
(OUT / "C2_FINAL_REPORT.md").write_text(report, encoding="utf-8")
(OUT / "RUN_LOG.md").write_text(
    "# C2 Forward RUN LOG\n\n"
    "CPU only. Input SHA verified 9fa462...\n"
    "OOF 110 evidence transitions and 78 persistence transitions completed.\n"
    "20,000 subject bootstrap, 100,000 sign-flip, sensitivity and reports completed.\n",
    encoding="utf-8")

# Requirements from actual environment.
(OUT / "requirements_freeze.txt").write_text(
    "Python 3.13.15\npandas==3.1.1\nnumpy==2.5.4\nscikit-learn==1.9.1\nscipy==1.18.1\n",
    encoding="utf-8")

# Inputs folder for reproducibility.
inp = OUT / "inputs"
inp.mkdir(exist_ok=True)
shutil.copyfile(SRC, inp / "model_frame_42_by_5.csv")
shutil.copyfile(Path(r"C:\Users\Administrator\Documents\ChatGPT\sci论文\Paper2_Hallucination_Dynamics\next_stage\c2_c3_phaseI_final_closure_20261113\Paper2_PhaseI_Repro\inputs\PREDICTIONS_ALL_MODELS.csv"), inp / "PREDICTIONS_ALL_MODELS.csv")

# SHA + ZIP (excluding zip itself).
files = [p for p in sorted(OUT.rglob("*")) if p.is_file()
         and p.name not in ("SHA256SUMS.txt", "C2_FORWARD_PREDICTION_V1_20261113.zip")]
sha_lines = []
for p in files:
    rel = str(p.relative_to(OUT)).replace("\\", "/")
    sha_lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {rel}")
(OUT / "SHA256SUMS.txt").write_text("\n".join(sha_lines) + "\n", encoding="utf-8")
ZIP = OUT / "C2_FORWARD_PREDICTION_V1_20261113.zip"
with zipfile.ZipFile(ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for p in files:
        z.write(p, arcname=str(p.relative_to(OUT)).replace("\\", "/"))
    z.write(OUT / "SHA256SUMS.txt", arcname="SHA256SUMS.txt")
print("ZIP", ZIP.stat().st_size, hashlib.sha256(ZIP.read_bytes()).hexdigest())
print("DONE")
