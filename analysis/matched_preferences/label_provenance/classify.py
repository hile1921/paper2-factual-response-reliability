"""Apply the user's two-axis taxonomy to existing generations with traceable evidence."""
from pathlib import Path
import hashlib
import json
import re
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
STAGE = ROOT.parent
sys.path[:0] = [str(STAGE.parent / "p1_confirmation"), str(STAGE / "c1"), str(STAGE / "c1_machine")]
import verify_analyze as verifier
import audit
import verify as strict

SEED = 20261115
SECOND = re.compile(r"(?:[.!?]\s+|\n\s*)(?:The capital(?: city)?|Its capital) is\s+([^.;\n]+)", re.I)


def secondary_claims(data):
    refs = verifier.References()
    candidates = []
    for r in data[data.prop.eq("capital")].itertuples():
        for i, hit in enumerate(SECOND.finditer(r.output)):
            span = hit.group(1).strip()
            if len(span.split()) > 7:
                continue
            # These patterns inherit the prompted subject. Other named subjects require separate review.
            status, reason, subject, entity, claim, reference = refs.verify(r._replace(output=span), span, "target_relation_assertion")
            candidates.append(dict(review_id=r.review_id + f"|claim{i+1}", parent_id=r.review_id,
                s_uri=r.s_uri, prop=r.prop, subj=r.subj, span=span, output=span,
                assertion_status="target_relation_assertion", truth_status=status,
                reason=reason, subject_revision_url=subject, output_entity_url=entity, statement_id=claim,
                independent_reference_url=reference, audit_label=audit.LABELS[status]))
    if not candidates:
        return pd.DataFrame(columns=["parent_id", "label"])
    frame = audit.evidence_checks(pd.DataFrame(candidates), refs.items)
    decisions = pd.DataFrame([strict.check(r, refs) for r in frame.itertuples()])
    frame["label"] = decisions.label
    frame["proof_reference"] = decisions.cross_reference
    return frame


def classify():
    data = pd.read_parquet(STAGE / "empirical_revision/c1_dual_axis_records.parquet")
    assert len(data) == 11370 and data.review_id.is_unique
    data["prior_revision_label"] = data.revision_label
    data["error_confirmed"] = data.revision_label.eq("wrong_confirmed")
    # A positive erroneous initial assertion is retained even when later language adds uncertainty.
    recover = data.baseline_first_span_label.eq("wrong_confirmed") & ~data.error_confirmed
    assert data.loc[recover, "evidence_crosscheck"].ne("").all()
    data.loc[recover, "error_confirmed"] = True
    data["first_claim_correct"] = data.revision_label.eq("correct_verified")
    more = secondary_claims(data)
    more.to_csv(ROOT / "secondary_proposition_evidence.csv", index=False, encoding="utf-8-sig")
    wrong_parents = set(more.loc[more.label.eq("verified_wrong"), "parent_id"])
    correct_parents = set(more.loc[more.label.eq("verified_correct"), "parent_id"])
    data["error_confirmed"] |= data.review_id.isin(wrong_parents)
    data["contains_verified_correct_proposition"] = data.first_claim_correct | data.review_id.isin(correct_parents)
    conflict = set()
    for parent, f in more.groupby("parent_id"):
        r = data.set_index("review_id").loc[parent]
        if r.error_confirmed and f.label.isin(["verified_correct", "verified_wrong"]).any():
            other = set(f.loc[f.label.isin(["verified_correct", "verified_wrong"]), "span"].map(verifier.normalize))
            if verifier.normalize(r.revision_evidence_span) not in other:
                conflict.add(parent)
    text = data.output.str.strip()
    refusal = text.str.match(r"(?i)^(?:I (?:refuse|decline)|I (?:cannot|can't|won't) (?:provide|assist|help)|I am unable to provide)\b")
    no_answer = text.str.match(r"(?i)^(?:I (?:do not|don't) know|I (?:cannot|can't) (?:determine|answer)|I am not sure|unknown\b)")
    incomplete = text.str.match(r"(?i)^(?:the (?:city|capital|country) of|located in|part of)\s*$")
    data["substantive_factual_answer"] = (data.error_confirmed | data.first_claim_correct |
        data.revision_label.eq("type_mismatch") | data.assertion_status.eq("target_relation_assertion") |
        data.alternate_extractor_state.eq("candidate")) & ~refusal & ~no_answer
    data["substantive_denominator_uncertain"] = ~data.substantive_factual_answer & data.revision_label.isin(["unverified", "incomplete_unresolved"])
    data["output_state"] = np.select(
        [data.review_id.isin(conflict), data.error_confirmed & data.contains_verified_correct_proposition,
         data.error_confirmed, refusal, no_answer, incomplete, data.first_claim_correct,
         data.revision_label.isin(["nonanswer", "type_mismatch"])],
        ["Contradictory", "Mixed", "Incorrect", "Refusal", "No Answer", "Incomplete", "Correct", "Non-responsive"], default="Unverifiable")
    data["primary_error_type"] = np.where(data.error_confirmed, "F1", "")
    data["secondary_error_types"] = ""
    data["error_type_evidence"] = np.where(data.error_confirmed, "sourced_exclusive_target_relation_and_distinct_real_city", "")
    data["full_output_additional_claims_unverified"] = text.str.contains(r"[.!?]\s+\S|\n\s*\S|\b(?:but|though|however|and|or)\b", regex=True)
    # A verified target span alone cannot establish that every extra assertion is correct.
    data.loc[data.output_state.eq("Correct") & data.full_output_additional_claims_unverified, "output_state"] = "Unverifiable"
    data["possible_error_unresolved"] = ~data.error_confirmed & data.output_state.isin(["Unverifiable", "Incomplete"])
    data["task_failure_confirmed"] = data.error_confirmed | data.output_state.isin(["Refusal", "No Answer", "Non-responsive", "Incomplete"])
    data["task_success_unresolved"] = ~data.task_failure_confirmed & ~data.output_state.eq("Correct")
    data["rule_version"] = "C1-taxonomy-v1.0"
    data.to_parquet(ROOT / "classified_records.parquet", index=False)
    data[data.error_confirmed | recover].to_csv(ROOT / "confirmed_error_events.csv", index=False, encoding="utf-8-sig")
    data[data.output_state.isin(["Mixed", "Contradictory"]) | recover].to_csv(ROOT / "multi_claim_cases.csv", index=False, encoding="utf-8-sig")
    data.groupby(["cohort", "prop", "step", "output_state"]).size().rename("answers").reset_index().to_csv(ROOT / "output_states_by_stage.csv", index=False)
    type_counts = []
    for key, f in data.groupby(["cohort", "prop", "step"]):
        for kind in ("F1", "F2", "F3", "F4", "F5", "F6"):
            type_counts.append(dict(zip(["cohort", "relation", "step"], key), error_type=kind,
                                    primary_count=int(f.primary_error_type.eq(kind).sum()),
                                    interpretation="confirmed" if kind == "F1" else "not_established_by_current_evidence"))
    pd.DataFrame(type_counts).to_csv(ROOT / "error_types_by_stage.csv", index=False)
    return data


def statistics(data):
    columns = ["error_confirmed", "substantive_factual_answer", "substantive_denominator_uncertain",
               "possible_error_unresolved", "task_failure_confirmed", "task_success_unresolved"]
    results = []
    for key, f in data.groupby(["cohort", "prop", "step"]):
        groups = f.groupby("s_uri")
        values = groups[columns].sum().to_numpy(dtype=float)
        values = np.column_stack([groups.size().to_numpy(), values])
        totals = values.sum(0)
        n,e,s,q,u,a,v = totals
        picks = np.random.default_rng(SEED).integers(0,len(values),(5000,len(values)))
        sim = values[picks].sum(1)
        estimates = {"FEI":(e/n,sim[:,1]/sim[:,0]),
                     "CFER":(e/s if s else np.nan,np.divide(sim[:,1],sim[:,2],out=np.full(5000,np.nan),where=sim[:,2]>0)),
                     "AFR_confirmed":(a/n,sim[:,5]/sim[:,0])}
        for metric,(point,draw) in estimates.items():
            good = draw[np.isfinite(draw)]
            lo,hi = np.quantile(good,[.025,.975]) if len(good) else (np.nan,np.nan)
            # Unknown factual respondents are already in s. q is uncertain denominator membership.
            unknown_in_s = float(f.loc[f.substantive_factual_answer & f.possible_error_unresolved].shape[0])
            lower = e/(s+q) if s+q else np.nan
            upper = min(1,(e+unknown_in_s+q)/(s+q)) if s+q else np.nan
            bounds = {"FEI":(e/n,(e+u)/n), "CFER":(lower,upper), "AFR_confirmed":(a/n,(a+v)/n)}[metric]
            results.append(dict(zip(["cohort","relation","step"],key),metric=metric,subjects=len(values),
                answers=int(n),confirmed_error_answers=int(e),substantive_answers=int(s),uncertain_denominator=int(q),
                value=point,ci_low=lo,ci_high=hi,valid_bootstrap_draws=len(good),identification_lower=bounds[0],identification_upper=bounds[1]))
    pd.DataFrame(results).to_csv(ROOT / "metrics_by_stage.csv",index=False)


if __name__ == "__main__":
    statistics(classify())
