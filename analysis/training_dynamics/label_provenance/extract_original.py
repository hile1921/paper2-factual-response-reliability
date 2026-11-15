#!/usr/bin/env python3
"""Rebuild the eligible set WITH subject/object QIDs, then re-extract."""
import re
from pathlib import Path

import pandas as pd

ROOT = Path("/home/ubuntu04/gpu_workspace/p2hall")
raw = pd.read_parquet(ROOT / "FACT_COMPLETION_RAW.parquet")
panel = pd.read_csv(ROOT / "DATA_PANEL/PANEL_P1.csv")

# uid is subj_id_prop_id_obj_id
ids = panel.assign(
    uid=panel["subj_id"].astype(str) + "_" + panel["prop_id"].astype(str) + "_"
        + panel["obj_id"].astype(str))
elig = ids[ids["prop"].isin(["capital", "country"])][
    ["uid", "prop", "subj", "gold" if "gold" in ids else "obj", "s_uri", "o_uri",
     "num_shared_chunks"]].rename(columns={"obj": "gold"})
print(f"eligible facts: {len(elig)}  subjects: {elig['subj'].nunique()}")
print(elig["s_uri"].notna().sum(), "with s_uri;", elig["o_uri"].notna().sum(), "with o_uri")
elig.to_csv(ROOT / "gate/P1_ELIGIBLE_FACTS.csv", index=False)

sub = raw[raw["uid"].isin(elig["uid"])].merge(
    elig[["uid", "s_uri", "o_uri", "num_shared_chunks"]], on="uid", how="left")
print(f"outputs: {len(sub)}  missing s_uri: {sub['s_uri'].isna().sum()}")

TITLE = re.compile(r"^(mr|mrs|ms|dr|sir|lord|lady|prof|st|saint|jr|sr)\.?$", re.I)
INIT = re.compile(r"^([A-Z]\.?\s*){1,4}$")
ART = re.compile(r"^(a|an|the)\s+\w+")


def norm(x):
    x = str(x).lower()
    x = re.sub(r"[^a-z0-9 ]", " ", x)
    return re.sub(r"\s+", " ", x).strip()


def extract(out, subj, prop):
    raw_o = (out or "").strip()
    if not raw_o:
        return "", "unparseable"
    first = raw_o.splitlines()[0].strip().strip(' "\'“”‘’')
    first = re.sub(r"^[\d,.\-–\s]+", "", first)
    if not first:
        return "", "nonanswer"
    span = re.split(r"[.;,\n]| which | who | that ", first, maxsplit=1)[0].strip()
    n = norm(span)
    if not n:
        return span, "unparseable"
    if len(span.split()) > 7 or len(span) < 3 or TITLE.match(span) or INIT.match(span):
        return span, "nonanswer"
    if ART.match(n) or n.startswith(norm(prop)):
        return span, "nonanswer"
    if n.startswith(norm(subj)):
        return span, "echo_or_template"
    if not re.match(r"^[A-Z0-9]", span):
        return span, "nonanswer"
    return span, "target_relation_assertion"


rows = []
for r in sub.to_dict("records"):
    span, st = extract(r["output"], r["subj"], r["prop"])
    rows.append({**r, "extracted_span": span, "assertion_status": st})
out = pd.DataFrame(rows)
out.to_csv(ROOT / "gate/P1_EXTRACTED_3420.csv", index=False)
print(out["assertion_status"].value_counts().to_string())
print("assertions with s_uri:", out[(out.assertion_status == "target_relation_assertion")]
      ["s_uri"].notna().sum())
