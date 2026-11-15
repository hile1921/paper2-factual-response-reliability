"""Apply the frozen P1 candidate extractor and public evidence gate offline."""
import ast
from collections import defaultdict
import hashlib
from itertools import combinations, product
import json
from pathlib import Path
import re
import unicodedata
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
HANDOFF = ROOT.parent / "handoff"
REF = ROOT / "references"
OUT = ROOT / "results"
KEY = ["uid", "step", "template", "mode"]
STEPS = [10000, 340000, 658032]
TEMPLATES = ["t1", "t2"]
MODES = ["greedy", "s0", "s1", "s2", "s3"]
STATUS = ["correct_verified", "wrong_confirmed", "type_mismatch", "nonanswer", "unverified"]
SEED = 20261110
N_BOOT = 5000
TEMPORAL = re.compile(r"\b(?:former(?:ly)?|historical(?:ly)?|used to|proposed|moved|until)\b", re.I)
CONFLICT = re.compile(r"\b(?:and|but)\b[^.!?\n]{0,50}\bcapital(?: city)? is\b|\bnot\b", re.I)


def read(path):
    return pd.read_csv(path, keep_default_na=False)


def check_freeze():
    for line in (ROOT / "FREEZE.sha256").read_text(encoding="ascii").splitlines():
        sha, name = line.split("  ", 1)
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha, name


def source_extractor():
    path = ROOT / "extract_original.py"
    tree = ast.parse(path.read_text(encoding="utf8"), filename=str(path))
    tree.body = [n for n in tree.body if
                 (isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                  and n.targets[0].id in {"TITLE", "INIT", "ART"})
                 or (isinstance(n, ast.FunctionDef) and n.name in {"norm", "extract"})]
    ns = {"re": re}
    exec(compile(tree, str(path), "exec"), ns)
    return ns["extract"]


def normalize(value):
    value = unicodedata.normalize("NFC", value).casefold().replace("’", "'")
    return " ".join(value.split())


def claim_value(claim, field="id"):
    value = claim["mainsnak"].get("datavalue", {}).get("value", {})
    return value.get(field, "") if isinstance(value, dict) else value


def names(entity):
    label = entity.get("labels", {}).get("en", {}).get("value", "")
    return {normalize(n) for n in [label, *[x["value"] for x in entity.get("aliases", {}).get("en", [])]] if n}


def link(entity):
    if not entity.get("id") or not entity.get("lastrevid"):
        return ""
    return f"https://www.wikidata.org/w/index.php?title={entity['id']}&oldid={entity['lastrevid']}"


class References:
    def __init__(self):
        self.items = json.loads((REF / "entities.json").read_text(encoding="utf8"))["entities"]
        self.official = read(ROOT / "capital_references.csv").set_index("s_uri").to_dict("index")
        self.countries = defaultdict(set)
        self.country_by_code = {}
        self.cities = defaultdict(dict)
        self.city_by_id = {}
        for line in (REF / "countryInfo.txt").read_text(encoding="utf-8-sig").splitlines():
            if line and not line.startswith("#"):
                fields = line.split("\t")
                code, country, capital, geo_id = fields[0], fields[4], fields[5], fields[16]
                self.country_by_code[code] = (country, capital, geo_id)
                self.countries[normalize(country)].add(code)
        with zipfile.ZipFile(REF / "cities15000.zip") as archive:
            with archive.open("cities15000.txt") as data:
                for line in data:
                    f = line.decode("utf8").rstrip("\n").split("\t")
                    city = (f[0], f[1], f[7], f[8], f[10])
                    self.city_by_id[f[0]] = city
                    # Restrict alternative names to source-confirmed Wikidata aliases.
                    for label in {normalize(f[1]), normalize(f[2])}:
                        if label:
                            self.cities[label][city[0]] = city
        for e in self.items.values():
            code = {claim_value(c) for c in e.get("claims", {}).get("P297", [])
                    if c["rank"] != "deprecated"} & self.country_by_code.keys()
            if len(code) == 1:
                for label in names(e):
                    self.countries[label].update(code)

    def current(self, subject, prop):
        current, historical = [], []
        for claim in subject.get("claims", {}).get(prop, []):
            if claim["rank"] == "deprecated" or not claim_value(claim):
                continue
            qualifiers = claim.get("qualifiers", {})
            # Unqualified present assertions are the fixed estimand. Any explicit
            # start/end/point-in-time is retained in the snapshot for adjudication.
            if {"P582", "P585"} & qualifiers.keys():
                historical.append(claim)
            else:
                current.append(claim)
        return current, historical

    def verify(self, row, span, assertion):
        if assertion != "target_relation_assertion":
            return ("nonanswer", "original P1 extractor: " + assertion, "", "", "", "")
        subject = self.items.get(row.s_uri, {})
        prop = "P36" if row.prop == "capital" else "P17"
        claims, past = self.current(subject, prop)
        good = {claim_value(c): c for c in claims}
        known = {claim_value(c): c for c in past}
        norm = normalize(span)
        official = self.official.get(row.s_uri, {}) if row.prop == "capital" else {}
        official_names = {normalize(x) for x in official.get("accepted", "").split("|") if x}
        historical_names = {normalize(x) for x in official.get("hold", "").split("|") if x}
        country_codes = {claim_value(c) for c in subject.get("claims", {}).get("P297", [])
                         if c["rank"] != "deprecated"} & self.country_by_code.keys()
        iso = next(iter(country_codes)) if len(country_codes) == 1 else ""
        geo_country = self.country_by_code.get(iso, ("", "", ""))
        geo_ref = f"https://www.geonames.org/{geo_country[2]}" if geo_country[2] else ""
        city_map = self.cities.get(norm, {})

        def result(status, reason, obj="", claim=None, extra=""):
            return (status, reason, link(subject), obj, claim.get("id", "") if claim else "", extra)

        if TEMPORAL.search(row.output) or CONFLICT.search(row.output.splitlines()[0]):
            return result("unverified", "temporal, negative or competing first-line reading")
        matches = [q for q in good if norm in names(self.items.get(q, {}))]
        past_hits = [q for q in known if norm in names(self.items.get(q, {}))]
        if norm in official_names:
            return result("correct_verified", "exact alias in independently sourced current-capital table",
                          extra=official["reference"])
        if len(matches) == 1:
            q = matches[0]
            return result("correct_verified", "exact label/alias of current relation statement",
                          link(self.items.get(q, {})), good[q])
        if past_hits or norm in historical_names:
            return result("unverified", "historical or competing capital name")
        if len(matches) > 1:
            return result("unverified", "multiple compatible entity links")
        if row.prop == "country":
            if city_map and norm not in self.countries:
                return result("type_mismatch", "place name below country granularity; may be true literally",
                              extra="|".join(f"https://www.geonames.org/{cid}" for cid in sorted(city_map)))
            return result("unverified", "no positive country match or no exclusive contradiction for broad prompt")
        if norm in self.countries and not city_map:
            return result("type_mismatch", "country name rather than capital city")
        if len(city_map) != 1:
            return result("unverified", "no unique exact city in fixed gazetteer")
        city = next(iter(city_map.values()))
        city_url = f"https://www.geonames.org/{city[0]}"
        if len(good) == 1:
            q, claim = next(iter(good.items()))
            target = self.items.get(q, {})
            target_names = names(target)
            expected = normalize(geo_country[1])
            if expected and norm == expected and expected in target_names:
                return result("correct_verified", "matching capital in independent country register",
                              link(target), claim, geo_ref)
            if (expected and expected in target_names and city[3] != iso
                    and not any(norm in a or a in norm for a in target_names if a)
                    and norm not in self.countries and not known):
                return result("wrong_confirmed", "distinct exact city in another country conflicts with concordant current capital sources",
                              city_url, claim, geo_ref)
        if official_names and not any(norm in a or a in norm for a in official_names):
            # Listed US state cases require that the linked city be in another
            # state or country. City-names that may denote a country are withheld.
            state = {"Q816": "AZ", "Q759": "NH", "Q1581": "MO"}.get(row.s_uri)
            if state and norm not in self.countries and (city[3] != "US" or city[4] != state):
                return result("wrong_confirmed", "exact city outside state contradicts sourced current state capital",
                              city_url, extra=official["reference"])
        return result("unverified", "insufficient evidence of an exclusive factual contradiction", city_url)


def assess(frame, refs, extract):
    rows = []
    for row in frame.itertuples(index=False):
        span, assertion = extract(row.output, row.subj, row.prop)
        rows.append((span, assertion, *refs.verify(row, span, assertion)))
    fields = ["span", "assertion_status", "truth_status", "reason", "subject_revision_url",
              "output_entity_url", "statement_id", "independent_reference_url"]
    return frame.join(pd.DataFrame(rows, columns=fields, index=frame.index)).assign(
        reviewer_type="machine_rule_plus_public_references", independent_human_review="pending")


def paired_panel(frame):
    count = frame.groupby(["s_uri", "step"]).size().unstack().reindex(columns=STEPS)
    assert not count.isna().any().any() and count.nunique(axis=1).eq(1).all()
    weights = count.iloc[:, 0].to_numpy()
    indices = np.random.default_rng(SEED).integers(0, len(weights), size=(N_BOOT, len(weights)))
    denominator = weights[indices].sum(axis=1)[:, None]
    arrays = {}
    flags = {x: frame.truth_status.eq(x) for x in STATUS}
    flags["candidate"] = frame.assertion_status.eq("target_relation_assertion")
    flags["valid_verified"] = flags["correct_verified"] | flags["wrong_confirmed"]
    for name, flag in flags.items():
        values = frame.assign(flag=flag).groupby(["s_uri", "step"]).flag.sum().unstack()
        arrays[name] = values.reindex(index=count.index, columns=STEPS).to_numpy()
    samples = {name: values[indices].sum(axis=1) / denominator for name, values in arrays.items()}
    samples["coverage"] = np.divide(samples["valid_verified"], samples["candidate"],
                                    out=np.full_like(samples["candidate"], np.nan),
                                    where=samples["candidate"] > 0)
    return {k: v.sum(axis=0) for k, v in arrays.items()}, samples


def summaries(frame, cohort):
    scope = [("all", frame)]
    for key in ("prop", "template", "mode"):
        scope += [(f"{key}={k}", x) for k, x in frame.groupby(key)]
    scope += [(f"{p}/{t}/{m}", x) for (p,t,m), x in frame.groupby(["prop", "template", "mode"])]
    scope += [(f"{p}/template={t}", x) for (p,t), x in frame.groupby(["prop", "template"])]
    scope += [(f"{p}/mode={m}", x) for (p,m), x in frame.groupby(["prop", "mode"])]
    stage_rows, delta_rows = [], []
    for label, f in scope:
        counts, samples = paired_panel(f)
        n = f.groupby("step").size().reindex(STEPS).to_numpy()
        rates = {k: v / n for k, v in counts.items()}
        rates["coverage"] = np.divide(counts["valid_verified"], counts["candidate"],
                                      out=np.full(3, np.nan), where=counts["candidate"] > 0)
        for i, s in enumerate(STEPS):
            stage_rows.append({"cohort": cohort, "scope": label, "step": s, "n": n[i],
                               "subjects": f.s_uri.nunique(), **{k: v[i] for k,v in counts.items()},
                               **{k + "_rate": v[i] for k,v in rates.items()}})
        for a,b in combinations(range(3), 2):
            for name in ("candidate", "valid_verified", "coverage", "correct_verified", "wrong_confirmed"):
                draws = samples[name][:, b] - samples[name][:, a]
                finite = draws[np.isfinite(draws)]
                lo, hi = np.percentile(finite, [2.5, 97.5]) if len(finite) else (np.nan, np.nan)
                delta_rows.append({"cohort": cohort, "scope": label, "early": STEPS[a],
                                   "late": STEPS[b], "metric": name, "subjects": f.s_uri.nunique(),
                                   "delta": rates[name][b] - rates[name][a], "ci_low": lo, "ci_high": hi,
                                   "bootstrap_valid_draws": len(finite),
                                   "unverified_low": rates[name][b] - rates[name][a] - rates["unverified"][a]
                                   if name == "wrong_confirmed" else np.nan,
                                   "unverified_high": rates[name][b] - rates[name][a] + rates["unverified"][b]
                                   if name == "wrong_confirmed" else np.nan})
    return pd.DataFrame(stage_rows), pd.DataFrame(delta_rows)


def main():
    check_freeze()
    OUT.mkdir(exist_ok=True)
    panel = read(ROOT / "FROZEN_PANEL.csv")
    facts = read(HANDOFF / "P1_ELIGIBLE_FACTS.csv")
    old_labels = read(HANDOFF / "RUN20_DELIVERABLES/P1_VERIFIED_3420.csv")
    old_keys = read(HANDOFF / "RUN20_DELIVERABLES/P1_EXTRACTED_3420.csv")[KEY]
    raw = pd.concat([pd.read_parquet(ROOT / f"raw_{s}.parquet") for s in STEPS], ignore_index=True)
    expected = set(product(panel.uid, STEPS, TEMPLATES, MODES))
    assert len(raw) == len(expected) == 4770 and set(raw[KEY].itertuples(index=False,name=None)) == expected
    assert set(raw.model) == {"LMEnt-1B-6E"} and raw.output.notna().all()
    assert not set(raw.uid) & set(old_keys.uid)
    prompt_map = {"capital": ["The capital of {s} is", "{s} has its capital at"],
                  "country": ["{s} is located in", "{s} is a part of"]}
    assert all(r.prompt == prompt_map[r.prop][TEMPLATES.index(r.template)].format(s=r.subj)
               for r in raw.itertuples())
    ident = raw.merge(panel[["uid", "prop", "subj", "obj"]], on="uid",
                      suffixes=("", "_panel"), validate="many_to_one")
    assert (ident.prop == ident.prop_panel).all() and (ident.subj == ident.subj_panel).all()
    assert (ident.gold == ident.obj).all()
    old = pd.read_parquet(HANDOFF / "FACT_COMPLETION_RAW.parquet").merge(old_keys, on=KEY, validate="one_to_one")
    refs, extract = References(), source_extractor()
    assert sum(bool(refs.items.get(q, {}).get("lastrevid")) for q in panel.s_uri) >= 0.9 * len(panel)
    result, contrast = [], []
    for cohort, f in [("confirmation", raw), ("old_p1_same_gate", old)]:
        f = f.merge(facts[["uid", "s_uri", "o_uri", "num_shared_chunks"]], on="uid", validate="many_to_one")
        f = assess(f, refs, extract)
        f.to_csv(OUT / f"{cohort}_rows.csv", index=False, encoding="utf-8-sig")
        a,b = summaries(f, cohort)
        result.append(a)
        contrast.append(b)
        if cohort == "old_p1_same_gate":
            audit = old_labels[old_labels.truth_status.eq("wrong_confirmed")][
                KEY + ["truth_status", "answer_set", "resolved_qid", "claim_labels", "verify_note"]]
            audit = audit.rename(columns={"truth_status":"old_truth_status", "resolved_qid":"old_resolved_qid"})
            audit = audit.merge(f, on=KEY, validate="one_to_one")
            assert len(audit) == 94
            audit.to_csv(OUT / "old_94_wrong_evidence.csv", index=False, encoding="utf-8-sig")
        else:
            f[f.truth_status.eq("wrong_confirmed")].to_csv(
                OUT / "new_wrong_evidence.csv", index=False, encoding="utf-8-sig")
    pd.concat(result).to_csv(OUT / "stage_and_strata.csv", index=False)
    d = pd.concat(contrast)
    d.to_csv(OUT / "paired_contrasts.csv", index=False)
    primary = d[d.cohort.eq("confirmation") & d.scope.eq("all") & d.early.eq(STEPS[0])
                & d.late.eq(STEPS[-1]) & d.metric.eq("correct_verified")].iloc[0]
    (OUT / "primary.json").write_text(json.dumps({
        "delta": float(primary.delta), "ci95": [float(primary.ci_low), float(primary.ci_high)],
        "replicated": bool(primary.delta > 0 and primary.ci_low > 0),
        "raw_sha256": {str(s): hashlib.sha256((ROOT / f"raw_{s}.parquet").read_bytes()).hexdigest()
                       for s in STEPS},
        "reference_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(REF.iterdir()) if p.is_file()},
        "independent_human_reviews": 0,
    }, indent=2) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
