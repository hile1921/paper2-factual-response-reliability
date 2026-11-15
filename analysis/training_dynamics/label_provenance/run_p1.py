"""Offline reproduction and versioned sensitivity audit of the existing P1 outputs."""
import ast
import hashlib
import json
import platform
import re
import tempfile
import unicodedata
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow


ROOT = Path(__file__).resolve().parent
HANDOFF = ROOT.parent / "handoff"
ARCHIVE = HANDOFF / "RUN20_DELIVERABLES"
OUT = ROOT / "results"
KEY = ["uid", "step", "template", "mode"]
STEPS = [10000, 340000, 658032]
TEMPLATES = ["t1", "t2"]
MODES = ["greedy", "s0", "s1", "s2", "s3"]
CITY = {"Q515", "Q3957", "Q15284", "Q1549591", "Q486972", "Q589282",
        "Q1093829", "Q5119", "Q2087181"}
COUNTRY = "Q6256"
SEED = 20261110


def read_csv(path):
    return pd.read_csv(path, keep_default_na=False)


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse(path):
    return ast.parse(path.read_text(encoding="utf8"), filename=str(path))


def inherited_inputs(cache_claims):
    raw_path = HANDOFF / "FACT_COMPLETION_RAW.parquet"
    raw = pd.read_parquet(raw_path)
    panel_path = HANDOFF / "p1_final/DATA_PANEL/PANEL_P1.csv"
    panel = read_csv(panel_path)
    panel["uid"] = panel[["subj_id", "prop_id", "obj_id"]].astype(str).agg("_".join, axis=1)
    generation = HANDOFF / "src/fact_completion.py"
    template_code = next(n for n in parse(generation).body
                         if isinstance(n, ast.Assign) and n.targets[0].id == "TEMPLATES")
    templates = ast.literal_eval(template_code.value)
    assert panel.uid.is_unique
    expected = panel[panel.prop.isin(templates)].head(1000)
    expected_grid = pd.MultiIndex.from_tuples(
        product(expected.uid, STEPS, TEMPLATES, MODES), names=KEY)
    assert set(raw.model) == {"LMEnt-1B-6E"}
    assert not raw.isna().any().any() and not raw.output.str.strip().eq("").any()
    raw_grid = pd.MultiIndex.from_frame(raw[KEY])
    assert raw_grid.is_unique and not len(expected_grid.difference(raw_grid))
    assert not len(raw_grid.difference(expected_grid))
    identity = raw.merge(panel[["uid", "prop", "subj", "obj"]], on="uid",
                         suffixes=("", "_panel"), validate="many_to_one")
    for left, right in [("prop", "prop_panel"), ("subj", "subj_panel"), ("gold", "obj")]:
        assert identity[left].equals(identity[right])
    expected_prompts = [templates[r.prop][TEMPLATES.index(r.template)].format(s=r.subj)
                        for r in raw.itertuples()]
    assert raw.prompt.tolist() == expected_prompts
    eligible = panel[panel.prop.isin(["capital", "country"])][
        ["uid", "prop", "subj", "obj", "s_uri", "o_uri", "num_shared_chunks"]
    ].rename(columns={"obj": "gold"})
    pd.testing.assert_frame_equal(eligible.reset_index(drop=True), read_csv(HANDOFF / "P1_ELIGIBLE_FACTS.csv"))
    selected = raw.merge(eligible[["uid", "s_uri", "o_uri", "num_shared_chunks"]],
                         on="uid", validate="many_to_one")
    assert len(selected) == 3420 and selected.s_uri.str.fullmatch(r"Q\d+").all()
    write_json("matrix.json", {
        "raw_rows": len(raw), "facts": raw.uid.nunique(), "unique_cells": len(raw_grid),
        "missing_cells": 0, "duplicate_cells": 0, "null_or_blank_outputs": 0,
        "prompt_mismatches_to_no_trailing_space_templates": 0,
        "steps": STEPS, "templates": TEMPLATES, "modes": MODES,
        "selection": "First 1000 supported relations in PANEL_P1 order, then capital/country only",
        "panel_relations": eligible.prop.value_counts().to_dict(),
        "generated_relations": selected.drop_duplicates("uid").prop.value_counts().to_dict(),
        "not_in_this_generation_matrix": eligible.loc[~eligible.uid.isin(selected.uid)].prop.value_counts().to_dict(),
        "missing_cached_subject_property_uids": sorted(
            r.uid for r in selected.drop_duplicates("uid").itertuples()
            if not (cache_claims.get(r.s_uri) or {}).get({"capital": "P36", "country": "P17"}[r.prop])
        ),
        "num_shared_chunks_min": int(selected.num_shared_chunks.min()),
        "num_shared_chunks_zero_count": int(selected.drop_duplicates("uid").num_shared_chunks.eq(0).sum()),
        "subjects_by_panel_id": int(selected.uid.str.split("_").str[0].nunique()),
        "subjects_by_wikidata_id": int(selected.s_uri.nunique()),
        "files_sha256": {str(p.relative_to(HANDOFF)): digest(p) for p in [
            raw_path, panel_path, generation, HANDOFF / "patch_fc.py",
            HANDOFF / "server_snapshot/fact_completion.py",
            HANDOFF / "server_snapshot/fact_completion.log",
            HANDOFF / "P1_ELIGIBLE_FACTS.csv", HANDOFF / "p1_verify_v3.py",
            HANDOFF / "p1_wikidata_cache.json", HANDOFF / "src/p1_extract2.py",
            ARCHIVE / "P1_EXTRACTED_3420.csv", ARCHIVE / "P1_VERIFIED_3420.csv"
        ]},
        "audit_sha256": {"script": digest(Path(__file__)), "capital_references": digest(ROOT / "capital_references.csv")},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "pandas": pd.__version__, "pyarrow": pyarrow.__version__}
    })
    return selected


def reproduce_legacy(selected):
    parser_file = HANDOFF / "src/p1_extract2.py"
    parser_tree = parse(parser_file)
    # Reuse the reviewed regex definitions and extraction functions without the
    # source file's machine-specific I/O.
    parser_tree.body = [
        n for n in parser_tree.body if
        (isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
         and n.targets[0].id in {"TITLE", "INIT", "ART"})
        or (isinstance(n, ast.FunctionDef) and n.name in {"norm", "extract"})
    ]
    ns = {"re": re}
    exec(compile(parser_tree, str(parser_file), "exec"), ns)
    labels = [ns["extract"](r.output, r.subj, r.prop) for r in selected.itertuples()]
    extracted = selected.assign(extracted_span=[r[0] for r in labels],
                                assertion_status=[r[1] for r in labels])
    compare(extracted, read_csv(ARCHIVE / "P1_EXTRACTED_3420.csv"))

    verifier_file = HANDOFF / "p1_verify_v3.py"
    tree = parse(verifier_file)
    with tempfile.TemporaryDirectory(dir=OUT) as tmp:
        tmp = Path(tmp)
        source, verified = tmp / "extracted.csv", tmp / "verified.csv"
        extracted.to_csv(source, index=False)
        network_misses = []

        def offline_get(url):
            network_misses.append(url)
            return None

        # Leave the truth rules untouched. Supply local paths, use the archived
        # cache read-only and intercept every cache miss before any network I/O.
        tree.body = [n for n in tree.body if not (
            (isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
             and n.targets[0].id in {"BASE", "DEL", "SRC", "OUT", "CACHE"})
            or (isinstance(n, ast.FunctionDef) and n.name in {"get", "save"}))]
        exec(compile(tree, str(verifier_file), "exec"), {
            "SRC": str(source), "OUT": str(verified),
            "CACHE": str(HANDOFF / "p1_wikidata_cache.json"),
            "get": offline_get, "save": lambda: None,
        })
        rebuilt = read_csv(verified)
    original = read_csv(ARCHIVE / "P1_VERIFIED_3420.csv")
    a = rebuilt.set_index(KEY).sort_index()
    b = original.set_index(KEY).sort_index()
    assert a.index.equals(b.index)
    differences = a.ne(b)
    write_json("legacy_reproduction.json", {
        "extracted_rows_matching_archive": len(extracted),
        "verified_rows": len(a), "truth_status_differences": int(differences.truth_status.sum()),
        "fields_with_differences": {k: int(v) for k, v in differences.sum().items() if v},
        "intercepted_cache_misses": len(network_misses),
        "cache_miss_urls": sorted(set(network_misses)),
    })
    assert not differences.truth_status.any(), "Legacy truth labels differ from archive"
    return extracted.merge(rebuilt[KEY + ["truth_status", "answer_set", "current_set",
                                          "resolved_qid", "verify_note"]],
                           on=KEY, how="left", validate="one_to_one")


def compare(left, right):
    pd.testing.assert_frame_equal(left.set_index(KEY).sort_index(),
                                  right.set_index(KEY).sort_index(), check_dtype=False)


def normalize(text):
    # Keep accents and distinguish letters instead of collapsing unrelated names.
    text = unicodedata.normalize("NFC", text).casefold().replace("’", "'")
    return " ".join(text.split())


def candidate(raw, subject):
    first = raw.strip().splitlines()[0].strip().lstrip("\"'“‘")
    prefix = re.match(r"(?i)^the (?:city|town|capital city) of\s+", first)
    name_text = first[prefix.end():] if prefix else first
    name = re.split(r"[.;,!?]|\s+(?:which|who|that|where|and|but)\s+",
                    name_text, maxsplit=1, flags=re.I)[0].strip(" \"'“”‘’")
    if not name or not name[0].isalpha() or re.match(
        r"(?i)^(?:a|an|the|one|it|in|to|as|of|not|known|said|considered|now|located|part)\s+",
        re.sub(r"(?i)^the (?=United (?:States|Kingdom)\b)", "", name)
    ):
        return name, "invalid_continuation"
    if normalize(name) == normalize(subject):
        return name, "echo_or_ambiguous_self_reference"
    if not name[0].isupper() or len(name.split()) > 7 or "?" in first:
        return name, "unresolved_assertion"
    return name, "candidate"


def audit_truth(row, cache, references):
    name, state = candidate(row.output, row.subj)
    ref = references.get(row.s_uri, {}) if row.prop == "capital" else {}
    url = ref.get("reference", "")

    def record(status, reason, qid=""):
        return name, status, reason, qid, url

    if state == "invalid_continuation":
        return record("invalid_continuation", state)
    if state != "candidate":
        return record("unresolved", state)
    if re.search(r"(?i)\b(?:former(?:ly)?|until|moved|historical(?:ly)?|used to|proposed)\b",
                 row.output):
        return record("unresolved", "temporal reading or scope may differ")
    if re.search(r"(?i)\b(?:and|but)\b[^.!?\n]{0,50}\bcapital(?: city)? is\b",
                 row.output.splitlines()[0]):
        return record("unresolved", "competing statement in first clause")
    norm = normalize(name).removeprefix("the ")
    exact = {qid for key in {name, name.removeprefix("the ")}
             for qid, label in cache["hits"].get(key, []) if normalize(label) == norm}
    qid = next(iter(exact)) if len(exact) == 1 else ""
    types = set(cache["types"].get(qid, []))
    if ref and norm in {normalize(x) for x in ref["accepted"].split("|")}:
        return record("correct_machine", "literal match to positive public capital reference", qid)
    if ref and norm in {normalize(x) for x in ref["hold"].split("|")}:
        return record("unresolved", "historical or competing geographic reading", qid)
    if row.prop == "country":
        if types & CITY and COUNTRY not in types:
            return record("type_or_granularity", "city does not answer country; broad prompt may be literally true", qid)
        return record("unresolved", "prompt does not specify sovereign-country granularity", qid)
    if COUNTRY in types and not (types & CITY):
        return record("type_or_granularity", "exact country name in capital-city slot", qid)
    # A mutually exclusive, positively sourced capital is required. Wikidata
    # search only supplies a candidate identity; its omissions cannot imply falsity.
    accepted = ref.get("accepted", "").split("|")
    shares_name = any(norm in normalize(a) or normalize(a) in norm for a in accepted if a)
    if ref and not shares_name and qid and types & CITY and COUNTRY not in types:
        return record("wrong_machine_candidate", "distinct exact-linked settlement conflicts with sourced current capital; review required", qid)
    return record("unresolved", "missing exclusive reference, clean span or exact typed entity", qid)


def rates(group):
    n = len(group)
    candidate_count = int(group.assertion.sum())
    verified = int((group.correct + group.wrong).sum())
    return {
        "attempts": n, "subjects": group.s_uri.nunique(), "candidates": candidate_count,
        "correct": int(group.correct.sum()), "wrong": int(group.wrong.sum()),
        "type_or_granularity": int(group.type.sum()), "unresolved": int(group.unresolved.sum()),
        "invalid": int((1 - group.assertion).sum()), "candidate_per_attempt": candidate_count / n,
        "verified_valid_per_attempt": verified / n,
        "truth_coverage_of_candidates": verified / candidate_count if candidate_count else None,
        "wrong_per_attempt": group.wrong.mean(),
        "wrong_given_verified": group.wrong.sum() / verified if verified else None,
        "wrong_plus_unresolved_per_attempt": (group.wrong + group.unresolved).mean(),
    }


def summarize(df, label):
    scopes = [("all", df)] + [
        (f"{column}={value}", group) for column in ("prop", "template", "mode")
        for value, group in df.groupby(column, sort=True)
    ] + [(f"prop={p}/template={t}/mode={m}", group)
         for (p, t, m), group in df.groupby(["prop", "template", "mode"], sort=True)]
    table, contrasts = [], []
    for name, frame in scopes:
        stage = {int(s): rates(g) for s, g in frame.groupby("step")}
        table.extend({"scope": name, "step": s, **stats} for s, stats in stage.items())
        paired = frame.groupby(["s_uri", "step"])[["wrong", "correct", "assertion", "unresolved"]].sum()
        count = frame.groupby(["s_uri", "step"]).size().unstack()
        assert count.nunique(axis=1).eq(1).all()
        weight = count[STEPS[0]].to_numpy()
        # Resample subjects with all their paired attempts, preserving their size.
        rng = np.random.default_rng(SEED)
        indices = rng.integers(0, len(weight), size=(5000, len(weight)))
        total = weight[indices].sum(axis=1)
        for early, late in combinations(STEPS, 2):
            conditional_low = (stage[late]["wrong_per_attempt"]
                               - stage[early]["wrong_plus_unresolved_per_attempt"])
            conditional_high = (stage[late]["wrong_plus_unresolved_per_attempt"]
                                - stage[early]["wrong_per_attempt"])
            for metric in ("wrong", "correct", "assertion"):
                pivot = paired[metric].unstack().reindex(count.index)
                delta = (pivot[late] - pivot[early]).to_numpy()
                lo, hi = np.percentile(delta[indices].sum(axis=1) / total, [2.5, 97.5])
                contrasts.append({
                    "scope": name, "from_step": early, "to_step": late, "metric": metric,
                    "subjects": len(weight), "delta": delta.sum() / weight.sum(),
                    "ci_low": lo, "ci_high": hi,
                    "unresolved_only_low": conditional_low if metric == "wrong" else None,
                    "unresolved_only_high": conditional_high if metric == "wrong" else None,
                })
    pd.DataFrame(table).to_csv(OUT / f"{label}_rates.csv", index=False)
    pd.DataFrame(contrasts).to_csv(OUT / f"{label}_paired.csv", index=False)


def make_queue(rows):
    review = rows.copy()
    review["review_id"] = [
        hashlib.sha256(("P1 audit blind 20261110|" + "|".join(map(str, k))).encode()).hexdigest()[:18]
        for k in review[KEY].itertuples(index=False, name=None)]
    assert review.review_id.is_unique
    mandatory = review.truth_status.eq("wrong_confirmed") | review.strict_status.eq("wrong_machine_candidate")
    rest = review.loc[~mandatory]
    parts = [review.loc[mandatory].assign(inclusion_probability=1.0)]
    rng = np.random.default_rng(SEED)
    for _, group in rest.groupby(["prop", "step", "strict_status"], sort=True):
        n = min(len(group), 12)
        parts.append(group.loc[rng.choice(group.index, n, replace=False)].assign(
            inclusion_probability=n / len(group)))
    sample = pd.concat(parts)
    assert sample.review_id.is_unique
    sample[["review_id", *KEY, "prop", "s_uri", "truth_status", "strict_status",
            "inclusion_probability"]].to_csv(OUT / "review_key_private.csv", index=False)
    fields = ["review_id", "prop", "subj", "prompt", "output"]
    blind = sample[fields].sample(frac=1, random_state=SEED).assign(
        human_assertion_status="", human_truth_status="", human_evidence_url="",
        human_evidence_note="", reviewer_id="")
    blind.to_csv(OUT / "review_blinded.csv", index=False, encoding="utf-8-sig")


def transitions(rows):
    summary = []
    for version, field in [("legacy", "truth_status"), ("strict", "strict_status")]:
        for relation, group in rows.groupby("prop"):
            pairs = group.pivot(index=["uid", "template", "mode"], columns="step", values=field)
            assert not pairs.isna().any().any()
            for early, late in combinations(STEPS, 2):
                counts = pairs.groupby([early, late]).size()
                summary.extend(dict(protocol=version, relation=relation, from_step=early, to_step=late,
                                    from_label=a, to_label=b, pairs=int(n))
                               for (a, b), n in counts.items())
    pd.DataFrame(summary).to_csv(OUT / "matched_label_transitions.csv", index=False)


def main():
    OUT.mkdir(exist_ok=True)
    cache = json.loads((HANDOFF / "p1_wikidata_cache.json").read_text(encoding="utf8"))
    rows = reproduce_legacy(inherited_inputs(cache["claims"]))
    rows["truth_status"] = rows.truth_status.fillna("not_an_assertion")
    refs = read_csv(ROOT / "capital_references.csv").set_index("s_uri").to_dict("index")
    strict = pd.DataFrame([audit_truth(r, cache, refs) for r in rows.itertuples()],
                          columns=["strict_span", "strict_status", "strict_reason",
                                   "strict_qid", "strict_reference"], index=rows.index)
    rows = rows.join(strict).assign(legacy_reviewer_type="machine_rule_plus_wikidata",
                                   strict_reviewer_type="machine_rules_with_source_references",
                                   human_review_status="pending")
    managua = (rows.s_uri.eq("Q811") & rows.truth_status.eq("wrong_confirmed")
               & rows.strict_status.eq("correct_machine"))
    rows["legacy_managua_only_sensitivity"] = rows.truth_status.mask(managua, "correct_verified")
    rows.to_csv(OUT / "attempts.csv", index=False, encoding="utf-8-sig")
    for label, column, states in [
        ("legacy", "truth_status", ["correct_verified", "wrong_confirmed", "type_mismatch", "unverified", "not_an_assertion"]),
        ("legacy_managua_only", "legacy_managua_only_sensitivity",
         ["correct_verified", "wrong_confirmed", "type_mismatch", "unverified", "not_an_assertion"]),
        ("strict", "strict_status", ["correct_machine", "wrong_machine_candidate", "type_or_granularity", "unresolved", "invalid_continuation"]),
    ]:
        assert set(rows[column]).issubset(states)
        table = rows.assign(**{key: rows[column].eq(value).astype(int)
                               for key, value in zip(("correct", "wrong", "type", "unresolved"), states)},
                            assertion=rows[column].ne(states[-1]).astype(int))
        summarize(table, label)
        if label == "strict":
            summarize(table[table.prop.eq("capital") & table.s_uri.isin(refs)], "strict_reference_panel")
    transitions(rows)
    make_queue(rows)


if __name__ == "__main__":
    main()
