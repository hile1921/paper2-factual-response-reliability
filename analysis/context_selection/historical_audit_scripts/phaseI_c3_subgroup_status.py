"""I-D: correct exploratory status and build 8-test subgroup Holm families."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "next_stage" / "c2_c3_phaseI_final_closure_20261113"
H = ROOT / "next_stage" / "c2_c3_phaseH_closure_20261113"
sub = pd.read_csv(H / "C3_GC_NEW_OVERLAP_SUBGROUPS.csv", keep_default_na=False)
# Primary reference Holm values from Phase H / G, not the subgroup raw p.
primary = pd.read_csv(ROOT / "next_stage" / "c2_c3_phaseG_audit_20261113" / "C3_HOLM_FAMILIES.csv",
                      keep_default_na=False)
prim = primary[(primary.family == "primary_v9_G-C_heuristic") & (primary.estimand == "micro")]
pmap = dict(zip(prim.checkpoint, prim.p_holm))
praw = dict(zip(prim.checkpoint, prim.p_raw))


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    a = np.maximum.accumulate(p[o] * (len(p) - np.arange(len(p))))
    out = np.empty(len(p)); out[o] = np.minimum(a, 1.0)
    return out


S = sub.copy()
S["exploratory"] = S.subset.isin(["new_countries", "overlap_countries"])
S["analysis_status"] = np.where(S.subset == "all_v9", "primary_analysis_reference",
                                "exploratory_subgroup")
S["primary_reference_raw_p"] = np.nan
S["primary_reference_holm_p"] = np.nan
for i, r in S.iterrows():
    if r.subset == "all_v9":
        S.loc[i, "primary_reference_raw_p"] = praw.get(r.checkpoint, np.nan)
        S.loc[i, "primary_reference_holm_p"] = pmap.get(r.checkpoint, np.nan)
S.to_csv(OUT / "C3_GC_SUBGROUP_STATUS_CORRECTED.csv", index=False, encoding="utf-8")

# 8-test exploratory families: 2 subgroups x 4 checkpoints, separately micro/macro.
rows = []
for estimand, pcol in (("micro", "micro_p"), ("macro", "macro_p")):
    x = S[S.subset.isin(["new_countries", "overlap_countries"])].copy()
    adj = holm(x[pcol].values)
    for (_, r), a in zip(x.iterrows(), adj):
        rows.append(dict(estimand=estimand, subset=r.subset, checkpoint=r.checkpoint,
                         n_countries=int(r.n_countries), p_raw=float(r[pcol]),
                         p_holm=float(a), family="8-test exploratory (2 subgroups x 4 checkpoints)"))
Hh = pd.DataFrame(rows)
Hh.to_csv(OUT / "C3_GC_SUBGROUP_HOLM_EXPLORATORY.csv", index=False, encoding="utf-8")

report = f"""# C3 Subgroup Inference (I-D)

## Status correction

`all_v9` is **not** an exploratory subgroup.  It is the existing primary
confirmatory analysis.  In `C3_GC_SUBGROUP_STATUS_CORRECTED.csv` it is marked
`primary_analysis_reference` and the original primary raw and Holm p values are
copied into the reference columns; its subgroup-table raw p must not be used as a
new confirmatory result.

`new_countries` and `overlap_countries` are both **exploratory**.  The new-country
subgroup has 10 country clusters and is not an independent external validation.
The overlap subgroup has 19 clusters and is not a new validation either.

## Eight-test exploratory Holm families

Micro and Macro are handled separately, each with 2 subgroups x 4 checkpoints = 8
tests:

{ Hh.to_string(index=False) }

## Interpretation

Bootstrap CIs and exact/permutation sign-flip p values can disagree when there
are only 10 clusters.  Report both and do not select whichever is favourable.
All new/overlap subgroup claims are exploratory; no new confirmatory family is
created.

## Limits

- v9 is a new country-wrong-answer pair holdout, not a fully independent
  country/template validation.
- The original G-C primary family is unchanged.
"""
(OUT / "C3_GC_SUBGROUP_INFERENCE_REPORT.md").write_text(report, encoding="utf-8")
print(S[S.subset.isin(["new_countries", "overlap_countries"])][
    ["subset", "checkpoint", "micro_p", "macro_p", "n_countries"]].to_string(index=False))
print(Hh.to_string(index=False))
