"""I-A: F_false valid-country subset, separately by checkpoint."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "next_stage" / "c2_c3_phaseI_final_closure_20261113"
H = ROOT / "next_stage" / "c2_c3_phaseH_closure_20261113"
PANEL = pd.read_csv(ROOT / "next_stage" / "cross_model_snapshot_117_20261113" / "raw" / "FACT_PANEL.csv",
                    keep_default_na=False, low_memory=False)
RAW = pd.concat([pd.read_parquet(f) for f in sorted((ROOT / "next_stage" / "c3_confirm" / "raw").glob("c3v9_*.parquet"))],
                ignore_index=True)
VALID = pd.read_csv(H / "C3_F_FALSE_VALID_COUNTRY_SUBSET.csv", keep_default_na=False)
SOURCE = pd.read_csv(H / "C3_F_FALSE_ENTITY_AUDIT.csv", keep_default_na=False)
B = 20000
N_PERM_ALL46 = 20000
SEED = 20261113
NEG = ("not", "isnt", "rather", "instead", "actually", "incorrect", "wrong", "never")
cap = PANEL[PANEL.prop == "capital"]
city = {re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(x).lower())).strip() for x in cap.gold.unique()}
country = {re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(x).lower())).strip() for x in PANEL.subj.unique()}
city.discard(""); country.discard("")


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(s).lower())).strip()


def first_six(out):
    return " ".join(re.split(r"[.\n]", str(out).lstrip(), maxsplit=1)[0].split()[:6])


def heuristic(out, wrong, gold):
    s = norm(first_six(out))
    if not s:
        return "nonanswer"
    w, g = norm(wrong), norm(gold)
    if g and g in s:
        return "correct"
    if w and w in s:
        i = s.find(w)
        if any(k in s[max(0, i - 20):i + len(w) + 8].split() for k in NEG):
            return "negated_or_corrected"
        return "designated_wrong"
    if any(c in s for c in city if c):
        return "other_answer"
    if any(c in s for c in country if c):
        return "type_mismatch"
    if any(k in s.split() for k in ("airport", "county", "province", "state", "island")):
        return "type_mismatch"
    return "nonanswer" if len(s.split()) <= 2 else "unverifiable"


RAW["probe_key"] = RAW.probe_id.astype(str).str.rsplit("|", n=1).str[0]
RAW["lab"] = [heuristic(o, w, g) for o, w, g in zip(RAW.output, RAW.wrong_answer, RAW.gold)]

valid_keys = set(zip(VALID.target_country, VALID.wrong_answer))
assert len(valid_keys) == 18
# All 46 keys for the exploratory mixed-entity companion.
all_keys = set(zip(SOURCE.target_country, SOURCE.wrong_answer))
assert len(all_keys) == 46


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.maximum.accumulate(p[o] * (len(p) - np.arange(len(p))))
    out = np.empty(len(p))
    out[o] = np.minimum(adj, 1.0)
    return out


def analyse(keys, tag, exact=True):
    sub = RAW[RAW[["target_country", "wrong_answer"]].apply(tuple, axis=1).isin(keys)].copy()
    counts = []
    contrast = []
    boot = []
    sign = []
    level = []
    for ck in sorted(sub.task.unique()):
        dt = sub[sub.task == ck]
        # Pair audit / country cells.
        h = dt[dt.arm == "C_true"].set_index(["probe_key", "template"])
        l = dt[dt.arm == "F_false"].set_index(["probe_key", "template"])
        common = h.index.intersection(l.index)
        h, l = h.loc[common], l.loc[common]
        cells = []
        for c in sorted(set(h.target_country) | set(l.target_country)):
            hh, ll = h[h.target_country == c], l[l.target_country == c]
            if not len(hh) or not len(ll):
                continue
            cells.append((int((hh.lab == "designated_wrong").sum()), len(hh),
                          int((ll.lab == "designated_wrong").sum()), len(ll)))
            level.append(dict(analysis=tag, checkpoint=ck, target_country=c,
                              k_C=int((hh.lab == "designated_wrong").sum()), N_C=len(hh),
                              k_F=int((ll.lab == "designated_wrong").sum()), N_F=len(ll)))
        A = np.asarray(cells, float)
        NA, NB = A[:, 1].sum(), A[:, 3].sum()
        g = 100 * (A[:, 0] / NA - A[:, 2] / NB)
        d = 100 * (A[:, 0] / A[:, 1] - A[:, 2] / A[:, 3])
        obs_micro, obs_macro = float(g.sum()), float(d.mean())
        rng = np.random.default_rng(SEED)
        idx = rng.integers(0, len(A), size=(B, len(A)))
        S = A[idx]
        mic = 100 * (S[:, :, 0].sum(1) / S[:, :, 1].sum(1)
                     - S[:, :, 2].sum(1) / S[:, :, 3].sum(1))
        mac = 100 * ((S[:, :, 0] / S[:, :, 1]) - (S[:, :, 2] / S[:, :, 3])).mean(1)
        # Exact enumeration of 2^18 sign patterns.
        n = len(A)
        if exact and n <= 18:
            bits = ((np.arange(2 ** n)[:, None] >> np.arange(n)[None, :]) & 1) * 2 - 1
            micro_null = (bits * g).sum(1)
            macro_null = (bits * d).mean(1)
            p_micro = (np.sum(np.abs(micro_null) >= abs(obs_micro)) + 1) / (2 ** n + 1)
            p_macro = (np.sum(np.abs(macro_null) >= abs(obs_macro)) + 1) / (2 ** n + 1)
            n_patterns = 2 ** n
        else:
            rng2 = np.random.default_rng(SEED)
            bits = rng2.choice([-1.0, 1.0], size=(N_PERM_ALL46, n))
            micro_null = (bits * g).sum(1)
            macro_null = (bits * d).mean(1)
            p_micro = (np.sum(np.abs(micro_null) >= abs(obs_micro)) + 1) / (N_PERM_ALL46 + 1)
            p_macro = (np.sum(np.abs(macro_null) >= abs(obs_macro)) + 1) / (N_PERM_ALL46 + 1)
            n_patterns = N_PERM_ALL46
        for c in sorted(set(h.target_country) | set(l.target_country)):
            pass
        contrast.append(dict(analysis=tag, checkpoint=ck, n_country_clusters=n,
                             k_C=int(A[:, 0].sum()), N_C=int(NA),
                             k_F=int(A[:, 2].sum()), N_F=int(NB),
                             micro_delta_pp=obs_micro,
                             macro_delta_pp=obs_macro,
                             micro_p=p_micro, macro_p=p_macro,
                             complete_pairs=int(len(common)),
                             exploratory=(tag != "valid_country_subset_primary")))
        boot.append(dict(analysis=tag, checkpoint=ck, estimand="micro",
                         CI_low=float(np.percentile(mic, 2.5)),
                         CI_high=float(np.percentile(mic, 97.5)),
                         bootstrap_mean=float(np.mean(mic)),
                         valid_replicates=int(np.isfinite(mic).sum()),
                         invalid_replicates=int((~np.isfinite(mic)).sum())))
        boot.append(dict(analysis=tag, checkpoint=ck, estimand="macro",
                         CI_low=float(np.percentile(mac, 2.5)),
                         CI_high=float(np.percentile(mac, 97.5)),
                         bootstrap_mean=float(np.mean(mac)),
                         valid_replicates=int(np.isfinite(mac).sum()),
                         invalid_replicates=int((~np.isfinite(mac)).sum())))
        sign.append(dict(analysis=tag, checkpoint=ck, estimand="micro", observed=obs_micro,
                         p_exact=float(p_micro), n_patterns=int(n_patterns),
                         exact=bool(exact and n <= 18)))
        sign.append(dict(analysis=tag, checkpoint=ck, estimand="macro", observed=obs_macro,
                         p_exact=float(p_macro), n_patterns=int(n_patterns),
                         exact=bool(exact and n <= 18)))
        print(tag, ck, "countries", n, "N", NA, "micro", obs_micro, "p", p_micro,
              "macro", obs_macro, "p", p_macro, flush=True)
    return contrast, boot, sign, level


# Valid subset per checkpoint.
valid_contrast, valid_boot, valid_sign, valid_level = analyse(valid_keys, "valid_country_subset")
# Full 46 mixed-entity companion.
all_contrast, all_boot, all_sign, all_level = analyse(all_keys, "mixed_entity_type_F_control", exact=False)

# Holm separately for 4 micro and 4 macro exploratory tests.
valid_holm = []
for est, key in (("micro", "micro_p"), ("macro", "macro_p")):
    pp = [r[key] for r in valid_contrast]
    adj = holm(pp)
    for r, a in zip(valid_contrast, adj):
        valid_holm.append(dict(analysis="valid_country_subset", estimand=est,
                               checkpoint=r["checkpoint"], p_raw=r[key], p_holm=float(a)))
    pp = [r[key] for r in all_contrast]
    adj = holm(pp)
    for r, a in zip(all_contrast, adj):
        valid_holm.append(dict(analysis="mixed_entity_type_F_control", estimand=est,
                               checkpoint=r["checkpoint"], p_raw=r[key], p_holm=float(a)))

V = pd.DataFrame(valid_contrast); VB = pd.DataFrame(valid_boot); VS = pd.DataFrame(valid_sign)
VL = pd.DataFrame(valid_level); A = pd.DataFrame(all_contrast); AB = pd.DataFrame(all_boot)
AS = pd.DataFrame(all_sign); H = pd.DataFrame(valid_holm)
V.to_csv(OUT / "C3_C_TRUE_F_FALSE_BY_CHECKPOINT.csv", index=False, encoding="utf-8")
VL.to_csv(OUT / "C3_C_TRUE_F_FALSE_COUNTRY_LEVEL.csv", index=False, encoding="utf-8")
VB.to_csv(OUT / "C3_C_TRUE_F_FALSE_BOOTSTRAP.csv", index=False, encoding="utf-8")
VS.to_csv(OUT / "C3_C_TRUE_F_FALSE_EXACT_SIGNFLIP.csv", index=False, encoding="utf-8")
H.to_csv(OUT / "C3_C_TRUE_F_FALSE_HOLM.csv", index=False, encoding="utf-8")
A.to_csv(OUT / "C3_C_TRUE_F_FALSE_ALL46_BY_CHECKPOINT.csv", index=False, encoding="utf-8")
AB.to_csv(OUT / "C3_C_TRUE_F_FALSE_ALL46_BOOTSTRAP.csv", index=False, encoding="utf-8")
AS.to_csv(OUT / "C3_C_TRUE_F_FALSE_ALL46_EXACT_SIGNFLIP.csv", index=False, encoding="utf-8")

# Country counts for every checkpoint and arm, valid subset and all46.
rows = []
for tag, keys in (("valid_country_subset", valid_keys), ("mixed_entity_type_F_control", all_keys)):
    sub = RAW[RAW[["target_country", "wrong_answer"]].apply(tuple, axis=1).isin(keys)]
    for (ck, arm, c), g in sub.groupby(["task", "arm", "target_country"]):
        if arm not in ("C_true", "F_false"):
            continue
        rows.append(dict(analysis=tag, checkpoint=ck, arm=arm, target_country=c,
                         k=int((g.lab == "designated_wrong").sum()), N=len(g)))
CC = pd.DataFrame(rows)
CC[CC.analysis == "valid_country_subset"].to_csv(
    OUT / "C3_F_FALSE_VALID_SUBSET_COUNTRY_COUNTS_BY_CHECKPOINT.csv", index=False, encoding="utf-8")
CC[CC.analysis == "mixed_entity_type_F_control"].to_csv(
    OUT / "C3_F_FALSE_COUNTRY_COUNTS_BY_CHECKPOINT.csv", index=False, encoding="utf-8")

# Source audit: entity classifications with checkpoint-level evidence.
src = SOURCE.copy()
src["checkpoint_count"] = 4
src["source_status_note"] = ("entity classification from Phase H frozen audit; "
                             "not a new web lookup")
src.to_csv(OUT / "C3_F_FALSE_SOURCE_AUDIT.csv", index=False, encoding="utf-8")

# Pair balance audit.
bal = []
for ck in sorted(RAW.task.unique()):
    dt = RAW[(RAW.task == ck) & RAW[["target_country", "wrong_answer"]].apply(tuple, axis=1).isin(valid_keys)]
    for arm in ("C_true", "F_false"):
        x = dt[dt.arm == arm]
        bal.append(dict(checkpoint=ck, arm=arm, rows=len(x),
                        pairs=x[["target_country", "wrong_answer"]].drop_duplicates().shape[0],
                        templates=x.template.nunique(),
                        n_per_pair_template=x.groupby(["target_country", "wrong_answer", "template"]).size().unique().tolist()))
pd.DataFrame(bal).to_csv(OUT / "C3_F_FALSE_VALID_PAIR_BALANCE.csv", index=False, encoding="utf-8")

report = f"""# C3 F_false by Checkpoint (I-A)

## Frozen subset

The Phase H frozen valid-country subset is used unchanged: **18 pairs / 18
target countries**.  The two templates are retained.  No case was added or
removed based on effect size.

## Expected structure checks

For each checkpoint and arm, the expected valid-subset structure is 18 countries
x 2 templates x 100 generations = **3,600 generation rows**.  The actual counts
and pair/template balance are in `C3_F_FALSE_VALID_PAIR_BALANCE.csv` and
`C3_F_FALSE_VALID_SUBSET_COUNTRY_COUNTS_BY_CHECKPOINT.csv`.

## Per-checkpoint results

| checkpoint | C k/N | F k/N | Micro Δpp | Micro 95% CI | Micro exact p | Macro Δpp | Macro 95% CI | Macro exact p | Micro Holm |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
"""
for r in V.itertuples():
    bm = VB[(VB.checkpoint == r.checkpoint) & (VB.estimand == "micro")].iloc[0]
    bM = VB[(VB.checkpoint == r.checkpoint) & (VB.estimand == "macro")].iloc[0]
    hm = H[(H.analysis == "valid_country_subset") & (H.estimand == "micro") & (H.checkpoint == r.checkpoint)].iloc[0]
    report += (f"| {r.checkpoint} | {r.k_C}/{r.N_C} | {r.k_F}/{r.N_F} | "
               f"{r.micro_delta_pp:+.3f} | [{bm.CI_low:+.3f}, {bm.CI_high:+.3f}] | "
               f"{r.micro_p:.5g} | {r.macro_delta_pp:+.3f} | "
               f"[{bM.CI_low:+.3f}, {bM.CI_high:+.3f}] | {r.macro_p:.5g} | {hm.p_holm:.5g} |\n")
report += f"""
Exact sign-flip enumerates all **2^18 = 262,144** country sign patterns separately
for Micro and Macro.  It is a cluster-level sign-flip test and relies on the
exchangeability/symmetry assumption for country-cluster differences under the
null; it is not an unconditional exact randomization test.

The four checkpoint Micro p-values are a separate **exploratory 4-test Holm
family**; the four Macro p-values are another exploratory 4-test family.  Neither
is added to the original G-C confirmatory family.

## All 46 mixed-entity companion

`C3_C_TRUE_F_FALSE_ALL46_BY_CHECKPOINT.csv` is explicitly
`mixed_entity_type_F_control`; it is descriptive and is not a clean
country-vs-country control.

## Source audit

`C3_F_FALSE_SOURCE_AUDIT.csv` retains the Phase H classification: 6 sovereign
countries, 6 subnational entities, 2 supranational organizations, 0 unverified.
It records that entity classification was not a new web lookup.

## Limits

- These are automatic first-six labels, not human audited.
- The subset is exploratory; no claim of a clean F control.
- Four checkpoints are analysed separately; the old pooled +0.139 pp estimate is
  not used to claim all models have zero effect.
"""
(OUT / "C3_F_FALSE_PHASEI_REPORT.md").write_text(report, encoding="utf-8")
print(json.dumps({"valid": V.to_dict("records"), "holm": H.to_dict("records"),
                  "balance": bal}, indent=2, default=str))
