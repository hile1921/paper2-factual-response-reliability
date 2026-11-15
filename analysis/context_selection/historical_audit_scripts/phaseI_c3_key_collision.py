"""I-F: reproducible key-collision audit with output hashes, without Phase G."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "next_stage" / "c2_c3_phaseI_final_closure_20261113"
H = ROOT / "next_stage" / "c2_c3_phaseH_closure_20261113"
KEY = pd.read_csv(H / "C3_ANNOTATION_KEY_PRIVATE.csv", keep_default_na=False, low_memory=False)
RAW = pd.concat([
    pd.concat([pd.read_parquet(f) for f in sorted((ROOT / "next_stage" / "c3_realwrong_v8" / "raw").glob("c3v8_*.parquet"))], ignore_index=True).assign(round="v8"),
    pd.concat([pd.read_parquet(f) for f in sorted((ROOT / "next_stage" / "c3_confirm" / "raw").glob("c3v9_*.parquet"))], ignore_index=True).assign(round="v9"),
], ignore_index=True)
JOIN = ["round", "task", "arm", "target_country", "wrong_answer", "output", "template"]
BASE = ["round", "task", "arm", "target_country", "wrong_answer", "template"]
BASE_NO_TEMPLATE = ["round", "task", "arm", "target_country", "wrong_answer", "output"]


def sha(x):
    return hashlib.sha256(str(x).encode("utf-8")).hexdigest()


# Corrected audit: full join includes template and stores only output hash.
g = (RAW.groupby(JOIN, dropna=False)
     .agg(n_raw=("probe_id", "size"),
          n_probe_id=("probe_id", "nunique"),
          n_template=("template", "nunique"))
     .reset_index())
g["output_sha256"] = g.output.map(sha)
g["join_key_sha256"] = g[JOIN].astype(str).agg("|".join, axis=1).map(sha)
g["collision"] = ((g.n_probe_id > 1) | (g.n_template > 1)).astype(int)
g = g.drop(columns=["output"])
g.to_csv(OUT / "C3_KEY_RECOVERY_COLLISION_AUDIT_CORRECTED.csv", index=False, encoding="utf-8")

# Reproduce the old ambiguity without template.
b = (RAW.groupby(BASE_NO_TEMPLATE, dropna=False)
     .agg(n_raw=("probe_id", "size"),
          n_distinct_probe_id=("probe_id", "nunique"),
          n_distinct_template=("template", "nunique"),
          probe_ids=("probe_id", lambda x: ";".join(sorted(set(map(str, x))))),
          templates=("template", lambda x: ";".join(sorted(set(map(str, x))))))
     .reset_index())
amb = b[(b.n_distinct_probe_id > 1) | (b.n_distinct_template > 1)].copy()
key_base = KEY[BASE_NO_TEMPLATE].astype(str).agg("|".join, axis=1)
base_counts = key_base.value_counts()
amb["base_key"] = ["|".join(map(str, r)) for r in amb[BASE_NO_TEMPLATE].itertuples(index=False, name=None)]
amb["affected_key_records"] = amb.base_key.map(base_counts).fillna(0).astype(int)
amb.to_csv(OUT / "C3_KEY_COLLISIONS_WITHOUT_TEMPLATE.csv", index=False, encoding="utf-8")

# Verify canonical key has 643 rows and can be matched with full key.
assert len(KEY) == 643 and KEY.queue_id.is_unique
K = KEY.copy()
K["join_key_sha256"] = K[JOIN].astype(str).agg("|".join, axis=1).map(sha)
matched = K.merge(g[["join_key_sha256", "n_raw", "n_probe_id", "collision"]],
                  on="join_key_sha256", how="left", validate="many_to_one")
missing = int(matched.n_raw.isna().sum())
assert missing == 0, missing
assert int(matched.collision.sum()) == 0, int(matched.collision.sum())

# Script/output consistency: the delivered script must be able to read the CSV it wrote.
re_read = pd.read_csv(OUT / "C3_KEY_RECOVERY_COLLISION_AUDIT_CORRECTED.csv", keep_default_na=False)
for c in ("output_sha256", "join_key_sha256", "n_raw", "n_probe_id", "collision"):
    assert c in re_read.columns, c
assert len(re_read) == len(g)

report = f"""# Key Recovery Script/Output Audit (I-F)

## Corrected CSV schema

`C3_KEY_RECOVERY_COLLISION_AUDIT_CORRECTED.csv` contains:

```
{', '.join(re_read.columns)}
```

It stores `output_sha256`, not the full output text.  The full `output` is used
only in memory for grouping.  The delivered script reads this exact schema.

## Corrected collisions

- raw join keys (with template): **{len(g)}**
- collision groups: **{int(g.collision.sum())}**
- key rows: **{len(KEY)}**
- key rows with a match: **{len(KEY) - missing}**
- key rows with >1 distinct probe_id for a join key: **{int((matched.n_probe_id > 1).sum())}**

## Old ambiguity without template

`C3_KEY_COLLISIONS_WITHOUT_TEMPLATE.csv` reproduces the old join.  Result:

- ambiguous groups without template: **{len(amb)}**
- affected key records: **{int(amb.affected_key_records.sum()) if len(amb) else 0}**
- distinct probe IDs involved in the ambiguous groups: **{len(set(";".join(amb.probe_ids.tolist()).split(";"))) if len(amb) else 0}**

The old join omitted `template`; an output string shared by two templates could
therefore map to two different probe IDs.  The corrected join includes
`template` and has zero collisions.

## Limit

This is a key-to-raw audit only; it does not create human labels.
"""
(OUT / "C3_KEY_RECOVERY_SCRIPT_OUTPUT_AUDIT.md").write_text(report, encoding="utf-8")
print({"raw_keys": len(g), "collisions": int(g.collision.sum()),
       "ambiguous_without_template": len(amb), "missing": missing})
