"""I-H: source fields for Stage 2 and a guarded Stage-1 -> Stage-2 builder."""
from __future__ import annotations

import pandas as pd
from pathlib import Path
from datetime import date

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "next_stage" / "c2_c3_phaseI_final_closure_20261113"
H = ROOT / "next_stage" / "c2_c3_phaseH_closure_20261113"
FIELDS = ["annotation_id", "target_country", "extracted_answer", "verified_gold",
          "correctness_status", "confidence_1to3", "note",
          "verification_source_name", "verification_source_url",
          "verification_access_date", "alias_resolution_note"]
for coder in ("A", "B"):
    s1 = pd.read_csv(H / f"C3_BLIND_CODER_{coder}_STAGE1.csv", keep_default_na=False)
    t = s1[["annotation_id", "target_country"]].copy()
    for c in FIELDS[2:]:
        t[c] = ""
    t = t[FIELDS]
    t.to_csv(OUT / f"C3_FACT_CHECK_{coder}_TEMPLATE_PHASEI.csv", index=False, encoding="utf-8")

protocol = """# C3 Annotation Source Protocol (Phase I)

## Stage 1

Use the frozen files:

- `C3_BLIND_CODER_A_STAGE1.csv`
- `C3_BLIND_CODER_B_STAGE1.csv`

Do not re-sample, do not regenerate annotation IDs, and do not send the private
key to either coder.  The question is always
`What is the capital of <target_country>?`.

Stage 1 fields are `annotation_id`, `target_country`, `neutral_question`,
`output`, `extracted_answer`, `assertion_status`, `confidence_1to3`, `note`.
`wrong_answer`, `gold`, `context`, treatment identifiers and automatic labels are
absent.

## Stage 2

Stage 2 must wait until both stage-1 files are locked.  The Phase I templates add:

```
verification_source_name
verification_source_url
verification_access_date
alias_resolution_note
```

For every extracted answer, record the source used to verify the capital.  If an
alias, historical capital, administrative seat, or disputed political status
makes the answer uncertain, use `uncertain_alias` or send it to adjudication.
Do not force a wrong label when the source is ambiguous.

`verified_gold` may show the correct capital; `wrong_answer` must never be shown
in Stage 2.  `sampling_weight` is not a coder field.

## Builder

`phaseI_build_stage2_from_locked_stage1.py` copies only `extracted_answer` from a
locked Stage 1 into the Stage 2 template.  It refuses a Stage 1 that is not
locked/complete and never reads the private key for hidden answers.

## Post-annotation analysis (not performed here)

After real human labels exist, compute raw and sampling-weight-weighted agreement,
Cohen's kappa, class confusion, adjudication, and automatic-rule precision,
recall and specificity.  Do not use `1 / rounded_inclusion_probability` in place
of the frozen `sampling_weight`.  No labels were generated in this stage.
"""
(OUT / "C3_ANNOTATION_SOURCE_PROTOCOL.md").write_text(protocol, encoding="utf-8")

builder = '''"""Copy locked Stage-1 extracted answers into Stage-2 templates.

This script deliberately refuses empty/incomplete Stage 1.  It never reads
wrong_answer, gold, context, arm, task, round or the private key.
"""
from __future__ import annotations

import argparse
import pandas as pd
from pathlib import Path

FIELDS = ["annotation_id", "target_country", "extracted_answer", "verified_gold",
          "correctness_status", "confidence_1to3", "note",
          "verification_source_name", "verification_source_url",
          "verification_access_date", "alias_resolution_note"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coder", choices=("A", "B"), required=True)
    ap.add_argument("--base", type=Path, required=True,
                    help="root containing the Stage 1 and Stage 2 template files")
    args = ap.parse_args()
    base = args.base
    s1 = pd.read_csv(base / f"C3_BLIND_CODER_{args.coder}_STAGE1.csv",
                     keep_default_na=False)
    s2 = pd.read_csv(base / f"C3_FACT_CHECK_{args.coder}_TEMPLATE_PHASEI.csv",
                     keep_default_na=False)
    if s1.extracted_answer.eq("").all() or s1.assertion_status.eq("").all():
        raise SystemExit("REFUSED: Stage 1 is empty/unlocked; complete and lock it first")
    if not s1.annotation_id.is_unique or set(s1.annotation_id) != set(s2.annotation_id):
        raise SystemExit("REFUSED: Stage 1/2 annotation_id mismatch")
    if (s1.annotation_id != s2.annotation_id).any():
        raise SystemExit("REFUSED: Stage 1/2 row order does not align")
    s2["extracted_answer"] = s1.extracted_answer.values
    s2.to_csv(base / f"C3_FACT_CHECK_{args.coder}_PHASEI.csv", index=False, encoding="utf-8")
    if any(x in s2.columns for x in ("wrong_answer", "gold", "arm", "task", "round",
                                     "context", "probe_id", "template", "stratum")):
        raise SystemExit("REFUSED: forbidden field in Stage 2")
    print(f"wrote C3_FACT_CHECK_{args.coder}_PHASEI.csv; no labels inferred")


if __name__ == "__main__":
    main()
'''
(OUT / "phaseI_build_stage2_from_locked_stage1.py").write_text(builder, encoding="utf-8")
print("wrote stage2 templates/source protocol/builder")
