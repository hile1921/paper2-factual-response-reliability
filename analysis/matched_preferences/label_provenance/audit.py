"""Produce a separate machine-supported crosswalk and a blinded review sample."""
import hashlib
import json
from itertools import combinations, product
from pathlib import Path
import re
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
P1 = PROJECT / "p1_confirmation"
KEY = ["uid", "step", "template", "mode"]
STEPS = [10000, 340000, 658032]
SEED = 20261110
LABELS = {"correct_verified": "correct_supported", "wrong_confirmed": "false_supported",
          "type_mismatch": "type_or_granularity_mismatch", "nonanswer": "nonassertion",
          "unverified": "unknown"}


def evidence_checks(rows,evidence):
    ref=P1/"references"
    with zipfile.ZipFile(ref/"cities15000.zip") as z:
        entries={f[0]:f for line in z.read("cities15000.txt").decode().splitlines()
                 if (f:=line.split("\t"))}
    countries={}
    for line in (ref/"countryInfo.txt").read_text(encoding="utf8").splitlines():
        if line and not line.startswith("#"):
            f=line.split("\t")
            countries[f[16]]=f
    checks={}
    for r in rows[rows.audit_label.eq("false_supported")].itertuples():
        ent=evidence[r.s_uri]
        city_id=r.output_entity_url.rsplit("/",1)[-1]
        geo=entries[city_id]
        assert geo[6]=="P" and r.prop=="capital"
        assert str(ent["lastrevid"]) in r.subject_revision_url
        if r.statement_id:
            claims=[x for x in ent["claims"]["P36"] if x["id"]==r.statement_id]
            assert len(claims)==1 and claims[0]["rank"]!="deprecated"
            assert not ({"P582","P585"}&claims[0].get("qualifiers",{}).keys())
            expected=countries[r.independent_reference_url.rsplit("/",1)[-1]]
            assert expected[0]!=geo[8]
            proof="wikidata_revision_and_cross_country_city_record"
        else:
            assert r.independent_reference_url.startswith("https://")
            assert r.s_uri in {"Q816","Q759","Q1581"}
            expected_code={"Q816":"AZ","Q759":"NH","Q1581":"MO"}[r.s_uri]
            assert geo[8]!="US" or geo[10]!=expected_code
            proof="official_state_capital_and_out_of_state_city_record"
        caution=bool(re.search(r"(?i)\b(?:the capital city is|its capital is now|capital, however|"
                               r"though most of the capital|actual administrative cent[er]+)\b",r.output))
        checks[r.review_id]=(proof,caution)
    rows["evidence_crosscheck"]=[checks.get(k,("",False))[0] for k in rows.review_id]
    rows["whole_output_scope_flag"]=[checks.get(k,("",False))[1] for k in rows.review_id]
    rows["audit_scope_sensitivity_label"]=rows.audit_label.mask(rows.whole_output_scope_flag,"unknown")
    return rows


def load(name, cohort):
    frame = pd.read_csv(P1 / "results" / name, keep_default_na=False)
    expected = set(product(frame.uid.unique(), STEPS, ["t1", "t2"], ["greedy", "s0", "s1", "s2", "s3"]))
    assert len(frame) == len(expected) and set(frame[KEY].itertuples(index=False, name=None)) == expected
    assert frame.groupby("uid").s_uri.nunique().eq(1).all()
    return frame.assign(cohort=cohort)


def stats(rows):
    records = []
    for cohort, cohort_rows in rows.groupby("cohort"):
        for prop, relation in cohort_rows.groupby("prop"):
            strata = [("all", relation)]
            for field in ("mode", "template"):
                strata += [(f"{field}={k}", v) for k, v in relation.groupby(field)]
            for scope, group in strata:
                y = group.assign(**{s: group.audit_label.eq(s).astype(int) for s in LABELS.values()},
                                 n=1, assertion=group.assertion_status.eq("target_relation_assertion").astype(int))
                y["verified"] = y.correct_supported + y.false_supported
                cols = ["n", "assertion", "verified", *LABELS.values()]
                m = y.groupby(["s_uri", "step"])[cols].sum().unstack().sort_index()
                a = {c: m[c].reindex(columns=STEPS).to_numpy() for c in cols}
                assert not any(np.isnan(v).any() for v in a.values())
                draw = np.random.default_rng(SEED).integers(0, len(m), (5000, len(m)))
                p = {c: v.sum(0).astype(float) for c, v in a.items()}
                b = {c: v[draw].sum(1).astype(float) for c, v in a.items()}
                metrics = {"correct_per_attempt": ("correct_supported", "n"),
                           "false_per_attempt": ("false_supported", "n"),
                           "assertion_per_attempt": ("assertion", "n"),
                           "verified_per_assertion": ("verified", "assertion"),
                           "unknown_per_attempt": ("unknown", "n"),
                           "nonassertion_per_attempt": ("nonassertion", "n"),
                           "false_given_verified": ("false_supported", "verified")}
                for label, (num, den) in metrics.items():
                    with np.errstate(invalid="ignore", divide="ignore"):
                        v, q = p[num]/p[den], b[num]/b[den]
                    common = dict(cohort=cohort, relation=prop, scope=scope, metric=label, subjects=len(m))
                    for i, step in enumerate(STEPS):
                        finite = q[:, i][np.isfinite(q[:, i])]
                        lo, hi = np.quantile(finite,[.025,.975]) if len(finite) else (np.nan,np.nan)
                        records.append(dict(**common, kind="stage", step=step, value=v[i],
                                            numerator=int(p[num][i]), denominator=int(p[den][i]),
                                            ci_low=lo, ci_high=hi))
                    for i,j in combinations(range(3),2):
                        d = q[:,j]-q[:,i]
                        finite = d[np.isfinite(d)]
                        lo,hi = np.quantile(finite,[.025,.975]) if len(finite) else (np.nan,np.nan)
                        extra = {}
                        if label == "false_per_attempt":
                            extra = {"unknown_low":v[j]-v[i]-p["unknown"][i]/p["n"][i],
                                     "unknown_high":v[j]-v[i]+p["unknown"][j]/p["n"][j]}
                        records.append(dict(**common, kind="paired", early=STEPS[i], late=STEPS[j],
                                            value=v[j]-v[i], ci_low=lo, ci_high=hi,
                                            valid_draws=len(finite), **extra))
    pd.DataFrame(records).to_csv(ROOT / "P1_REPRODUCED_STATISTICS.csv", index=False)


def main():
    a = load("old_p1_same_gate_rows.csv", "original")
    b = load("confirmation_rows.csv", "confirmation")
    assert not set(a.s_uri)&set(b.s_uri)
    rows = pd.concat([a,b],ignore_index=True)
    history = pd.read_csv(P1 / "results/old_94_wrong_evidence.csv", usecols=KEY+["old_truth_status"])
    rows = rows.merge(history,on=KEY,how="left",validate="one_to_one")
    rows["audit_label"] = rows.truth_status.map(LABELS)
    rows["review_id"] = [hashlib.sha256(("next-audit|"+"|".join(map(str,k))).encode()).hexdigest()[:20]
                         for k in rows[KEY].itertuples(index=False,name=None)]
    rows["human_label"] = ""
    assert rows.review_id.is_unique and not rows.audit_label.isna().any()
    evidence = json.loads((P1/"references/entities.json").read_text(encoding="utf8"))["entities"]
    rows=evidence_checks(rows,evidence)
    rows.to_csv(ROOT/"P1_AUDIT_ADJUDICATION.csv",index=False,encoding="utf-8-sig")
    eligible = rows.audit_label.eq("false_supported") | rows.old_truth_status.eq("wrong_confirmed")
    sampled = [rows[eligible].assign(inclusion_probability=1.)]
    rest = rows[~eligible & rows.audit_label.ne("nonassertion")]
    rng = np.random.default_rng(SEED)
    for _,g in rest.groupby(["cohort","prop","step","audit_label"],sort=True):
        n = min(8,len(g))
        sampled.append(g.loc[rng.choice(g.index,n,replace=False)].assign(inclusion_probability=n/len(g)))
    queue = pd.concat(sampled).sample(frac=1,random_state=SEED)
    queue[["review_id","prop","subj","prompt","output"]].assign(
        adjudicator="",judgment="",evidence_url="",evidence_as_of="",reason=""
    ).to_csv(ROOT/"P1_AUDIT_QUEUE.csv",index=False,encoding="utf-8-sig")
    queue[["review_id","cohort",*KEY,"audit_label","inclusion_probability","subject_revision_url",
           "statement_id","output_entity_url","independent_reference_url","reason"]].to_csv(
               ROOT/"audit_private_key.csv",index=False)
    inventory=[]
    for r in rows.drop_duplicates("uid").itertuples(index=False):
        e = evidence.get(r.s_uri,{})
        prop = "P36" if r.prop=="capital" else "P17"
        claims = [c for c in e.get("claims",{}).get(prop,[]) if c["rank"]!="deprecated"]
        values = {c["mainsnak"].get("datavalue",{}).get("value",{}).get("id","") for c in claims}
        values.discard("")
        inventory.append(dict(cohort=r.cohort,uid=r.uid,relation=r.prop,subject=r.subj,
                              s_uri=r.s_uri,o_uri=r.o_uri,revision=e.get("lastrevid"),
                              n_distinct_reference_values=len(values),multiple_reference_values=len(values)>1,
                              time_qualified=any({"P580","P582","P585"} & c.get("qualifiers",{}).keys()
                                                 for c in claims)))
    pd.DataFrame(inventory).to_csv(ROOT/"fact_inventory.csv",index=False)
    stats(rows)
    print(len(rows),len(queue),rows.audit_label.value_counts().to_dict())


if __name__ == "__main__":
    main()
