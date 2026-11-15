"""I-B/I-C: corrected strict-lag selection flow and common-step selection diagnostics."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "next_stage" / "c2_c3_phaseI_final_closure_20261113"
FRAME = ROOT / "next_stage" / "c2_wrong_relations" / "model_frame_42_by_5.csv"
EXPECTED = {170000: 10000, 340000: 170000, 500000: 340000, 658032: 500000}
COMMON = [10000, 340000, 658032]

raw = pd.read_csv(FRAME, keep_default_na=False, low_memory=False)
for c in raw.columns:
    if c not in ("uid", "fact_id", "subject", "scope_grade"):
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
assert len(raw) == 210, len(raw)
assert raw.uid.nunique() == 42
assert raw.groupby("uid").size().eq(5).all()
assert raw.duplicated(["uid", "step"]).sum() == 0
# subject/fact_id must be stable within uid
assert raw.groupby("uid").subject.nunique().eq(1).all()
assert raw.groupby("uid").fact_id.nunique().eq(1).all()

raw["available"] = raw.attempts.notna() & raw.error.notna() & (raw.attempts > 0)

# Corrected flow from the unfiltered 210-row frame.
flow = []
strict_rows = []
for tgt, src_step in EXPECTED.items():
    target = raw[raw.step == tgt].copy()
    source = raw[raw.step == src_step].copy()
    tkeys = set(target.uid); skeys = set(source.uid)
    assert len(tkeys) == 42 and len(skeys) == 42
    t_ok = target[target.available].copy()
    s_ok = source[source.available].copy()
    merged = t_ok.merge(s_ok[["uid", "step", "error", "attempts"]], on="uid",
                        how="inner", suffixes=("_target", "_source"), validate="one_to_one")
    strict_pairs = len(merged)
    flow.append(dict(
        target_step=tgt, source_step=src_step,
        raw_target_rows=len(target), target_outcome_available=int(target.available.sum()),
        target_outcome_missing=int((~target.available).sum()),
        raw_source_rows=len(source), source_outcome_available=int(source.available.sum()),
        source_outcome_missing=int((~source.available).sum()),
        strict_pairs_available=strict_pairs,
        missing_target_outcome=int((~target.available).sum()),
        missing_source_outcome=int((~source.available).sum()),
        target_rows_actually_absent=int(42 - len(tkeys)),
        source_rows_actually_absent=int(42 - len(skeys)),
    ))
    strict_rows.append(merged.assign(target_step=tgt, source_step=src_step))

F = pd.DataFrame(flow)
S = pd.concat(strict_rows, ignore_index=True)
# Reconstruct strict dataset rows (target features + source lag).
# Keep explicit source/target outcomes and all model-frame fields.
target_all = raw[raw.step.isin(EXPECTED.keys())].copy()
rebuilt = []
for tgt, src in EXPECTED.items():
    t = raw[(raw.step == tgt) & raw.available].copy()
    tmp = raw[(raw.step == src) & raw.available][["uid", "error", "attempts"]].copy()
    tmp["prev_rate_strict"] = tmp["error"] / tmp["attempts"]
    r = tmp.rename(columns={"error": "source_error", "attempts": "source_attempts"})
    x = t.merge(r, on="uid", how="inner", validate="one_to_one")
    x["target_step"] = tgt
    x["source_step"] = src
    x["target_error"] = x.error
    x["target_attempts"] = x.attempts
    rebuilt.append(x)
RB = pd.concat(rebuilt, ignore_index=True)
assert len(S) == len(RB) == 104
assert RB.uid.nunique() == 26
assert RB.groupby("uid").size().eq(4).all()

F.to_csv(OUT / "C2_STRICT_LAG_SELECTION_FLOW_CORRECTED.csv", index=False, encoding="utf-8")
RB.to_csv(OUT / "C2_STRICT_LAG_DATASET_REBUILT_V2.csv", index=False, encoding="utf-8")
excluded = sorted(set(raw.uid) - set(RB.uid))
complete = sorted(set(RB.uid))
assert len(excluded) == 16

# Common-step counts (same windows for both groups).
common = raw[raw.step.isin(COMMON)]
rows = []
for name, ids in (("complete_26", set(complete)), ("excluded_16", set(excluded))):
    x = common[common.uid.isin(ids)]
    for step, g in x.groupby("step"):
        rows.append(dict(group=name, step=int(step), subjects=int(g.uid.nunique()), rows=len(g),
                         errors=float(g.error.sum()), attempts=float(g.attempts.sum()),
                         error_rate=float(g.error.sum() / g.attempts.sum())))
C = pd.DataFrame(rows)
C.to_csv(OUT / "C2_STRICT_LAG_COMMON_STEP_COUNTS.csv", index=False, encoding="utf-8")

# Subject-level diagnostic at common steps and initial difficulty.
subj = []
for uid, g in common.groupby("uid"):
    group = "complete_26" if uid in set(complete) else "excluded_16"
    subj.append(dict(uid=uid, group=group, rows=len(g),
                     errors=float(g.error.sum()), attempts=float(g.attempts.sum()),
                     subject_error_rate=float(g.error.sum() / g.attempts.sum()),
                     initial_difficulty=float(g.initial_difficulty.iloc[0])))
Dsub = pd.DataFrame(subj)


def smd(a, b):
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled else float("nan")


diag = []
for name in ("complete_26", "excluded_16"):
    x = Dsub[Dsub.group == name]
    diag.append(dict(group=name, n_subjects=len(x),
                     n_rows=int(x.rows.sum()), errors=float(x.errors.sum()),
                     attempts=float(x.attempts.sum()),
                     pooled_error_rate=float(x.errors.sum() / x.attempts.sum()),
                     subject_error_mean=float(x.subject_error_rate.mean()),
                     subject_error_sd=float(x.subject_error_rate.std(ddof=1)),
                     initial_difficulty_mean=float(x.initial_difficulty.mean()),
                     initial_difficulty_sd=float(x.initial_difficulty.std(ddof=1))))
D = pd.DataFrame(diag)
D["initial_difficulty_smd_complete_minus_excluded"] = np.nan
D["subject_error_rate_smd_complete_minus_excluded"] = np.nan
a = Dsub[Dsub.group == "complete_26"]; b = Dsub[Dsub.group == "excluded_16"]
D.loc[D.group == "complete_26", "initial_difficulty_smd_complete_minus_excluded"] = smd(
    a.initial_difficulty.values, b.initial_difficulty.values)
D.loc[D.group == "complete_26", "subject_error_rate_smd_complete_minus_excluded"] = smd(
    a.subject_error_rate.values, b.subject_error_rate.values)
D.to_csv(OUT / "C2_STRICT_LAG_SELECTION_DIAGNOSTICS.csv", index=False, encoding="utf-8")
Dsub.to_csv(OUT / "C2_STRICT_LAG_SUBJECT_DIAGNOSTICS.csv", index=False, encoding="utf-8")

report = f"""# C2 Strict-Lag Selection Bias (I-B/I-C)

## What was wrong

Phase H's selection flow calculated `expected_subjects` after filtering to valid
attempts.  That made 170k and 500k look like they had only 26 raw rows.  The raw
`model_frame_42_by_5.csv` actually has **210 rows = 42 subjects x 5 steps**; all
42 rows exist at every step.  At 170k and 500k, **16 rows have missing/invalid
outcome attempts**, not absent rows.

Corrected flow:

{ F.to_string(index=False) }

- complete strict-lag subjects: **26**
- excluded subjects: **16**
- strict rows: **104**
- every complete subject has four target steps
- merge used `validate="one_to_one"`; subject/fact_id were stable within uid

## Common-step comparison

The old baseline comparison mixed five valid steps for the complete group with
three valid steps for the excluded group.  This report uses only the common
steps **10000, 340000, 658032**.

{ C.to_string(index=False) }

Subject-level diagnostics:

{ D.to_string(index=False) }

Interpretation:

- complete_26 common-step error rate is about 7.69%;
- excluded_16 common-step error rate is about 13.96%;
- initial difficulty is lower (easier) in complete_26 (about 0.877) than in
  excluded_16 (about 0.944);
- initial-difficulty SMD is about {float(D.loc[D.group == 'complete_26', 'initial_difficulty_smd_complete_minus_excluded'].iloc[0]):.3f};
- subject-level error-rate SMD is about {float(D.loc[D.group == 'complete_26', 'subject_error_rate_smd_complete_minus_excluded'].iloc[0]):.3f}.

These are **descriptive selection diagnostics**, not causal evidence.  No
inverse-probability weighting was applied; there is no credible missingness model
here.

## Scope of the strict-lag result

Only 26 of 42 subjects have complete valid observations.  `prev_rate_strict` is a
strict lag feature, but the Evidence features remain target-step t.  This is a
**strict-lag persistence-feature sensitivity analysis**, not a fully prospective
forecasting validation.
"""
(OUT / "C2_STRICT_LAG_SELECTION_BIAS_REPORT.md").write_text(report, encoding="utf-8")
print(json.dumps({"flow": F.to_dict("records"), "diag": D.to_dict("records"),
                  "common": C.to_dict("records")}, indent=2, default=str))
