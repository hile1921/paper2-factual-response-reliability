#!/usr/bin/env python3
"""C2 v2 GPU step 2: gold-answer log-probability and candidate rank.

No generation.  For each fact x checkpoint, score the gold capital and all
unique capital candidates under the frozen LMEnt checkpoint.
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path("/home/ubuntu04/gpu_workspace/p2hall")
OUT = ROOT / "next_stage" / "c2_gpu_expansion_20261113"
OUT.mkdir(parents=True, exist_ok=True)
PANEL = OUT / "C2_V2_FACT_PANEL_SERVER.csv"
CKPT = Path("/dev/shm/lment")
STEPS = [10000, 170000, 340000, 500000, 658032]
BATCH = 16

torch.cuda.set_per_process_memory_fraction(0.20)
torch.set_num_threads(4)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

panel = pd.read_csv(PANEL, keep_default_na=False).drop_duplicates("uid")
assert len(panel) == 90, len(panel)
candidates = sorted({str(x).strip() for x in panel.gold if str(x).strip()})
assert len(candidates) > 50
print("facts", len(panel), "candidates", len(candidates), flush=True)


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()


rows = []
for step in STEPS:
    ckpt = CKPT / f"LMEnt-1B-6E-step{step}"
    tok = AutoTokenizer.from_pretrained(ckpt, padding_side="left", local_files_only=True)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        ckpt, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True
    ).to("cuda:0").eval()
    print("loaded", step, "free_gib", torch.cuda.mem_get_info()[0] / 1024**3, flush=True)
    jobs = []
    for r in panel.itertuples(index=False):
        prompt = f"The capital of {r.subj} is"
        prefix = tok.encode(prompt, add_special_tokens=False)
        for cand in candidates:
            full = tok.encode(prompt + " " + cand, add_special_tokens=False)
            if len(full) <= len(prefix) or full[:len(prefix)] != prefix:
                continue
            jobs.append((r.uid, r.subj, r.gold, cand, prompt, prefix, full))
    print("jobs", step, len(jobs), flush=True)
    step_rows = []
    with torch.inference_mode():
        for i in range(0, len(jobs), BATCH):
            batch = jobs[i:i + BATCH]
            width = max(len(x[6]) for x in batch)
            ids = torch.tensor([[tok.pad_token_id] * (width - len(x[6])) + x[6] for x in batch],
                               device="cuda:0")
            mask = torch.tensor([[0] * (width - len(x[6])) + [1] * len(x[6]) for x in batch],
                                device="cuda:0")
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits.float()
            logsm = torch.log_softmax(logits, dim=-1)
            for j, (uid, subj, gold, cand, prompt, prefix, full) in enumerate(batch):
                n = len(full) - len(prefix)
                lp = logsm[j, width - n - 1:width - 1].gather(
                    1, ids[j, width - n:, None]).flatten()
                step_rows.append(dict(uid=uid, subject=subj, gold=gold, candidate=cand,
                                      step=step, n_answer_tokens=n,
                                      sum_logprob=float(lp.sum()),
                                      mean_logprob=float(lp.mean())))
            if i % (BATCH * 10) == 0:
                print("step", step, "batch", i, "/", len(jobs), flush=True)
    D = pd.DataFrame(step_rows)
    D.to_csv(OUT / f"c2_v2_logprob_step{step}.csv", index=False, encoding="utf-8")
    rows.extend(step_rows)
    del model
    torch.cuda.empty_cache()

D = pd.DataFrame(rows)
D.to_csv(OUT / "C2_V2_KNOWLEDGE_PROBS.csv", index=False, encoding="utf-8")
summary = []
for (uid, step), g in D.groupby(["uid", "step"]):
    gold = panel[panel.uid == uid].gold.iloc[0]
    gg = g[g.candidate == gold]
    if not len(gg):
        continue
    gl = float(gg.sum_logprob.iloc[0])
    rank = int((g.sum_logprob > gl).sum() + 1)
    top = g.sort_values("sum_logprob", ascending=False).iloc[0]
    # softmax over the candidate set
    z = np.exp(g.sum_logprob - g.sum_logprob.max())
    p_gold = float(z[g.candidate == gold].iloc[0] / z.sum())
    summary.append(dict(uid=uid, subject=panel[panel.uid == uid].subj.iloc[0], step=step,
                        gold_logprob=gl, gold_rank=rank, n_candidates=len(g),
                        gold_prob=float(p_gold), top_candidate=str(top.candidate),
                        top_logprob=float(top.sum_logprob)))
pd.DataFrame(summary).to_csv(OUT / "C2_V2_GOLD_PROBS.csv", index=False, encoding="utf-8")
(OUT / "V2_LOGP_DONE.json").write_text(json.dumps({"rows": len(D), "steps": STEPS,
    "facts": len(panel), "candidates": len(candidates)}, indent=2), encoding="utf-8")
print("DONE", len(D), flush=True)
