"""I-E: distinguish prompt-instance rows from global unique prompt strings."""
from __future__ import annotations

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "next_stage" / "c2_c3_phaseI_final_closure_20261113"
H = ROOT / "next_stage" / "c2_c3_phaseH_closure_20261113"
P = pd.read_csv(H / "C3_TOKEN_POSTPROCESSOR_ALL_PROMPTS.csv", keep_default_na=False)
L = pd.read_csv(H / "C3_TOKEN_LENGTH_AUDIT_CORRECTED.csv", keep_default_na=False)

counts = []
for (r, k), g in P.groupby(["round", "tokenizer"]):
    counts.append(dict(round=r, tokenizer=k,
                       n_checkpoint_arm_prompt_instances=len(g),
                       n_global_unique_prompt_strings=g.prompt.nunique(),
                       added_min=int(g.added_tokens.min()),
                       added_max=int(g.added_tokens.max()),
                       added_mean=round(float(g.added_tokens.mean()), 4)))
C = pd.DataFrame(counts)
C.to_csv(OUT / "C3_TOKEN_POSTPROCESSOR_COUNTS_CORRECTED.csv", index=False, encoding="utf-8")

matched = L[((L.checkpoint.str.contains("pythia", case=False)) & (L.tokenizer == "Pythia"))
            | ((L.checkpoint.str.contains("olmo", case=False)) & (L.tokenizer == "OLMo"))].copy()
matched.to_csv(OUT / "C3_TOKEN_LENGTH_MODEL_MATCHED_ONLY.csv", index=False, encoding="utf-8")
L.to_csv(OUT / "C3_TOKEN_LENGTH_CROSS_TOKENIZER_SENSITIVITY.csv", index=False, encoding="utf-8")

report = f"""# C3 Token Reporting (I-E)

## Prompt-instance rows vs global unique strings

| round | tokenizer | checkpoint-arm prompt instances | global unique prompt strings |
|---|---|---:|---:|
""" + "\n".join(
    f"| {r.round} | {r.tokenizer} | {r.n_checkpoint_arm_prompt_instances} | "
    f"{r.n_global_unique_prompt_strings} |" for r in C.itertuples()
) + f"""

The 1936/1568 values are **checkpoint-arm prompt-instance rows**, not global
unique prompt strings.  The global unique strings are 484/392.

## Token length tables

- `C3_TOKEN_LENGTH_MODEL_MATCHED_ONLY.csv`: Pythia checkpoints with Pythia
  tokenizer; OLMo checkpoints with OLMo tokenizer.  This is the model-matched
  result.
- `C3_TOKEN_LENGTH_CROSS_TOKENIZER_SENSITIVITY.csv`: all tokenizer x checkpoint
  cross counts; sensitivity only.

`n_generation_rows` and `n_unique_prompts` are separate fields.  For the G/C
comparison, v8 has 96 unique prompts and v9 has 92; the raw generation rows are
9,600 and 9,200 respectively.  A_none/B_filler have fewer unique prompt strings
because their context is empty; their generation rows are still 9,600/9,200.

## PostProcessor audit

Every unique prompt was checked with default encoding and
`add_special_tokens=False`.  OLMo adds exactly 1 token (EOS); Pythia adds 0.
See `C3_TOKEN_POSTPROCESSOR_COUNTS_CORRECTED.csv`.

## Runtime limit

**runtime tokenization not verified.**  The original v8/v9 generation wrapper was
not found.  These are reconstructed-prompt token counts.
"""
(OUT / "C3_TOKEN_REPORT_PHASEI.md").write_text(report, encoding="utf-8")
print(C.to_string(index=False))
