#!/usr/bin/env python3
"""Track III: open-vocabulary relational fact completion.

Templates are relational STATEMENT stubs (not QA), frozen before running:
  "The capital of France is"      -> model must emit an entity
Two lexical variants per relation, both frozen; no per-test template selection.

Decoding: greedy + 4 seeded samples. Labels are conservative:
  gold_alias_hit   - output starts with an accepted alias (needs factual confirmation)
  nonanswer        - relation paraphrase / prompt echo / no entity
  other_entity     - a named entity that is not an accepted alias (candidate wrong)
  empty            - nothing emitted
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path("/home/ubuntu04/gpu_workspace/p2hall")
CKPT_ROOT = Path("/dev/shm/lment")

# frozen template pairs, keyed by relation
TEMPLATES = {
    "capital": ["The capital of {s} is", "{s} has its capital at"],
    "country": ["{s} is located in", "{s} is a part of"],
    "place of birth": ["{s} was born in", "{s} was born at"],
    "father": ["The father of {s} was", "{s}'s father was"],
    "mother": ["The mother of {s} was", "{s}'s mother was"],
    "director": ["The director of {s} was", "{s} was directed by"],
    "producer": ["The producer of {s} was", "{s} was produced by"],
    "screenwriter": ["The screenwriter of {s} was", "{s} was written by"],
    "composer": ["The composer of {s} was", "{s} was composed by"],
    "author": ["The author of {s} was", "{s} was written by"],
    "occupation": ["{s} worked as a", "{s} was a"],
    "sport": ["{s} played", "{s} was a player of"],
    "genre": ["{s} is a", "{s} belongs to the genre of"],
    "religion": ["{s} followed", "{s} practiced"],
}


def norm(s):
    s = str(s).lower().strip()
    s = re.sub(r"^(the|a|an)\s+", "", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def classify(out, fact):
    raw = (out or "").strip()
    if not raw:
        return "empty"
    first = raw.splitlines()[0].strip().strip(' "\'“”‘’')
    n = norm(first)
    if not n:
        return "empty"
    # prompt echo / relation paraphrase
    s_n, p_n = norm(fact["subj"]), norm(fact["prop"])
    if s_n and s_n in n and len(n.split()) <= len(s_n.split()) + 2:
        return "nonanswer"
    if p_n and n.startswith(p_n):
        return "nonanswer"
    if "?" in first or re.search(r"\b(is|was|were)\s+(a|the)\s+\w+\?$", first):
        return "nonanswer"
    toks = n.split()
    head = " ".join(toks[:4])
    for a in fact["aliases"]:
        na = norm(a)
        if not na:
            continue
        if n.startswith(na) or head.startswith(na) or n == na:
            return "gold_alias_hit"
    # a capitalised span that is not an accepted alias
    if re.match(r"^[A-Z0-9]", first) and len(first.split()) <= 6:
        return "other_entity"
    if len(first.split()) > 8:
        return "nonanswer"
    return "other_entity"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="LMEnt-1B-6E")
    ap.add_argument("--steps", default="10000,340000,658032")
    ap.add_argument("--panel", default="DATA_PANEL/PANEL_P1.csv")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", default="FACT_COMPLETION_RAW.parquet")
    args = ap.parse_args()

    panel = pd.read_csv(ROOT / args.panel)
    panel = panel[panel["prop"].isin(TEMPLATES)]
    panel = panel.head(args.n).reset_index(drop=True)
    print(f"panel facts: {len(panel)}  relations: {panel['prop'].nunique()}")

    facts = []
    for r in panel.to_dict("records"):
        al = r.get("possible_answers")
        try:
            al = list(al) if al is not None else []
        except TypeError:
            al = []
        al = [a for a in al if isinstance(a, str) and a.strip()]
        if str(r.get("obj")) not in al:
            al.append(str(r["obj"]))
        facts.append({"uid": f"{r['subj_id']}_{r['prop_id']}_{r['obj_id']}",
                      "subj": r["subj"], "prop": r["prop"], "gold": r["obj"],
                      "aliases": al, "freq_bin": str(r.get("num_shared_chunks", ""))})

    rows = []
    for step in [int(s) for s in args.steps.split(",")]:
        ck = CKPT_ROOT / f"{args.model}-step{step}"
        if not (ck / "model-00001-of-00002.safetensors").exists() and \
           not (ck / "model.safetensors").exists():
            print(f"MISSING {ck}", flush=True)
            continue
        tok = AutoTokenizer.from_pretrained(str(ck), padding_side="left")
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            str(ck), dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()

        for ti, tname in enumerate(["t1", "t2"]):
            prompts, metas = [], []
            for f in facts:
                if f["prop"] not in TEMPLATES:
                    continue
                tpl = TEMPLATES[f["prop"]][ti]
                # no trailing space: with one, the model emits numbers
                prompts.append(tpl.format(s=f["subj"]))
                metas.append(f)
            for mode in ["greedy"] + [f"s{i}" for i in range(args.samples)]:
                if mode == "greedy":
                    torch.manual_seed(0)
                    kw = dict(do_sample=False)
                else:
                    torch.manual_seed(int(mode[1:]) + 101)
                    kw = dict(do_sample=True, temperature=0.8, top_p=0.95)
                for b0 in range(0, len(prompts), args.batch):
                    pb = prompts[b0:b0 + args.batch]
                    mb = metas[b0:b0 + args.batch]
                    enc = tok(pb, return_tensors="pt", padding=True).to("cuda")
                    with torch.no_grad():
                        g = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                           pad_token_id=tok.eos_token_id, **kw)
                    dec = tok.batch_decode(g[:, enc["input_ids"].shape[1]:],
                                           skip_special_tokens=True)
                    for f, p, o in zip(mb, pb, dec):
                        rows.append({
                            "model": args.model, "step": step, "template": tname,
                            "mode": mode, "uid": f["uid"], "prop": f["prop"],
                            "subj": f["subj"], "gold": f["gold"],
                            "prompt": p, "output": o, "label": classify(o, f),
                        })
            print(f"  step{step} {tname} done ({len(rows)} rows)", flush=True)
        del model
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_parquet(ROOT / args.out, index=False)
    print(f"\nwrote {ROOT/args.out}: {len(df)} rows")
    print(df.groupby(["step", "mode", "label"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
