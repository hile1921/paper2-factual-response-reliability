"""Reassess the original 364 cases with frozen public sources and conservative gates."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
STAGE = ROOT.parent
PROJECT = STAGE.parent
sys.path[:0] = [str(PROJECT / "p1_confirmation"), str(PROJECT / "p1_audit")]
from verify_analyze import References, claim_value, names, normalize
from run_p1 import candidate

LABELS = ["verified_correct", "verified_wrong", "type_mismatch", "unverified"]


def read(path):
    return pd.read_csv(path, keep_default_na=False)


def check(row, refs):
    status, reason, subject, answer, statement, independent = refs.verify(
        row, row.span, row.assertion_status)
    rel = "P36" if row.prop == "capital" else "P17"
    entity = refs.items.get(row.s_uri, {})
    claims, _ = refs.current(entity, rel)
    citations = []
    for c in claims:
        if c["id"] == statement:
            for ref in c.get("references", []):
                citations.extend(claim_value({"mainsnak": s}) for s in ref.get("snaks", {}).get("P854", []))
    source_urls = sorted({u for u in citations if isinstance(u, str) and u.startswith(("https://", "http://"))})
    extra = ""
    if status == "correct_verified" and not independent:
        if row.prop == "capital":
            iso = {claim_value(c) for c in entity.get("claims", {}).get("P297", []) if c["rank"] != "deprecated"}
            matching = [refs.country_by_code[k] for k in iso & refs.country_by_code.keys()]
            compatible = [c for c in claims if c["id"] == statement and
                          normalize(row.span) in names(refs.items.get(claim_value(c), {}))]
            if len(matching) == len(compatible) == 1 and len({claim_value(c) for c in claims}) == 1:
                _, cap, geo = matching[0]
                if cap and geo and normalize(cap) in names(refs.items.get(claim_value(compatible[0]), {})):
                    independent = f"https://www.geonames.org/{geo}"
                    extra = "Wikidata statement and GeoNames country capital concordant"
        if not independent:
            status = "unverified"
            extra = "No independently corroborating evidence for the extracted target relation"
    if status in {"correct_verified", "wrong_confirmed"} and row.whole_output_scope_flag:
        status = "unverified"
        extra = "Later capital language creates a competing scope reading"
    if status == "wrong_confirmed" and (not independent or not answer or not row.evidence_crosscheck):
        status = "unverified"
        extra = "Exclusive contradiction proof incomplete"
    strict = {"correct_verified": "verified_correct", "wrong_confirmed": "verified_wrong",
              "type_mismatch": "type_mismatch"}.get(status, "unverified")
    alt_span, alt_state = candidate(row.output, row.subj) if row.output.strip() else ("", "empty")
    agrees = alt_state == "candidate" and normalize(row.span) == normalize(alt_span)
    alternate = strict if agrees or strict == "type_mismatch" else "unverified"
    return dict(label=strict, span_agreement_label=alternate, new_reason=extra or reason,
                subject_revision=subject, answer_entity=answer, claim_id=statement,
                cross_reference=independent, claim_reference_urls="|".join(source_urls),
                alternate_span=alt_span, alternate_extractor_state=alt_state,
                extractors_exact_agreement=agrees)


def summarize(rows):
    results = []
    by_step = [("all", rows), *rows.groupby(["cohort", "prop", "step"])]
    for label_col in ("label", "span_agreement_label"):
        for mode in ("raw_sample", "weighted_eligible_frame"):
            for scope, part in by_step:
                c, r, step = scope if isinstance(scope, tuple) else ("all", "all", "all")
                w = np.ones(len(part)) if mode == "raw_sample" else 1 / part.inclusion_probability.to_numpy()
                flags = np.column_stack([part[label_col].eq(k) for k in LABELS]).astype(float)
                wc = flags * w[:, None]
                sums = wc.sum(0)
                coverage, good, bad, mismatch, unverified = metrics(sums)
                per_subj = pd.DataFrame(wc).assign(subject=part.s_uri.to_numpy()).groupby("subject").sum().to_numpy()
                picks = np.random.default_rng(20261112).integers(0, len(per_subj), (5000, len(per_subj)))
                sim = metrics(per_subj[picks].sum(1))
                ci = np.quantile(sim[:3], [.025, .975], axis=1)
                results.append(dict(cohort=c, relation=r, step=step, gate=label_col, weighting=mode,
                                    sampled=len(part), subjects=len(per_subj), effective_frame_weight=w.sum(),
                                    **dict(zip(LABELS, sums)), coverage=coverage, correct_rate=good,
                                    wrong_rate=bad, coverage_ci_low=ci[0,0], coverage_ci_high=ci[1,0],
                                    correct_ci_low=ci[0,1], correct_ci_high=ci[1,1],
                                    wrong_ci_low=ci[0,2], wrong_ci_high=ci[1,2],
                                    correct_bound_low=good, correct_bound_high=good+unverified,
                                    wrong_bound_low=bad, wrong_bound_high=bad+unverified,
                                    correct_given_verified=good/coverage if coverage else np.nan,
                                    wrong_given_verified=bad/coverage if coverage else np.nan))
    return pd.DataFrame(results)


def metrics(sums):
    good, bad, mismatch, unverified = (sums / sums.sum(axis=-1, keepdims=True)).T
    return np.array([good+bad, good, bad, mismatch, unverified])


def main():
    old = read(STAGE / "c1/P1_AUDIT_ADJUDICATION.csv")
    keys = read(STAGE / "c1/audit_private_key.csv")[["review_id", "inclusion_probability"]]
    rows = keys.merge(old, on="review_id", validate="one_to_one")
    assert len(rows) == 364 and rows.inclusion_probability.gt(0).all()
    refs = References()
    rows = rows.rename(columns={"label": "frozen_raw_label"}).drop(
        columns=["human_label", "independent_human_review"])
    rows = rows.join(pd.DataFrame([check(r, refs) for r in rows.itertuples(index=False)]))
    rows["audit_mode"] = "machine_assisted_public_source_verification"
    rows.to_csv(ROOT / "adjudicated_364.csv", index=False, encoding="utf-8-sig")
    stats = summarize(rows)
    stats.to_csv(ROOT / "coverage_and_sensitivity.csv", index=False, encoding="utf-8-sig")
    pd.crosstab(rows.audit_label, rows.label).to_csv(ROOT / "label_transitions.csv")
    rows[rows.label.eq("unverified")].to_csv(ROOT / "unverified_cases.csv", index=False, encoding="utf-8-sig")
    print(stats[stats.cohort.eq("all")].to_string(index=False))


if __name__ == "__main__":
    main()
