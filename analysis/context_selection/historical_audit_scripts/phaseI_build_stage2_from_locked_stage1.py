"""Copy locked Stage-1 extracted answers into Stage-2 templates.

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
