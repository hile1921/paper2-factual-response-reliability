# CPU reproducibility

## Environment and commands

Tested with Python 3.13.15 and the package versions pinned in `requirements.txt`. Install them in a dedicated environment with `python -m pip install -r requirements.txt`. MATLAB R2026a is used for figures separately. PyArrow is optional for inspecting the archived Parquet snapshots and is not required by the verified commands below.

From the repository root:

```text
python scripts/verify_files.py
python scripts/verify_generated_records.py
python scripts/reproduce.py all
```

The statistics command can also take one panel name: `training_dynamics`, `matched_preferences`, `context_selection`, `forward_prediction`, or `positive_control`. It uses no GPU, downloads or model API. It writes to ignored `reproduction_outputs/`, preserving frozen inputs and expected tables. The five recorded panel runs together took approximately two minutes on the preparation workstation; runtime depends on the machine. Replaying every original evidence field is a separate command. A rerun is a replay of existing observations, not new model generation or new human annotation.

## Evidence-to-result map

| Panel | Observations and inputs | Recomputed evidence |
|---|---|---|
| Training dynamics | 117 `data/raw/*.json.gz` task files, 319,410 responses, 273 queries; `C1_AUDITED_OUTPUTS.csv.gz`; cached references | Scorability/error decomposition, paired windows, block sensitivity, unresolved brackets and per-fact dynamics |
| Matched preference/generation | Canonical generations, 85-candidate likelihood records, fixed scope/identity registry, input freeze and output-form diagnostics | Candidate scores, stage means, country bootstrap changes/correlations, wrong-event exclusions, coverage accounting |
| Contextual answer selection | 230,400 development outputs plus 184,000 extension outputs in two losslessly compressed CSVs | 56 final contrasts covering both detector definitions, subject-cluster 20,000-draw intervals and Holm families |
| Forward prediction | Frozen 210-row fact-by-stage frame; archived evidence timelines and inspected training excerpts | Original LOSO/walk-forward fits, OOF predictions, primary subject bootstrap, sign-flip and sensitivity tables |
| Single-case positive control | Baseline/evaluation JSON, six completed continuation-cell records, frozen configuration and label function | Original margins, generation metrics, 192 training-update records, pairing checks and failure cases |

## Original sources and portable adapters

`analysis/*/original/` retains original source bytes. `docs/ARCHIVE_FILE_PROVENANCE.json` maps every imported item to its archive-relative source and original/stored SHA256. Lossless gzip avoids GitHub's ordinary single-file limit without discarding fields or rows. `analysis/**` is explicitly exempted from Git text-line-ending normalisation because the original freeze checks hash raw bytes.

The portable runner changes only file locations, compressed-input reading, and orchestration. The matched-panel analysis and positive-control analysis run unchanged in scratch folders. Training analysis reads the same audited table from gzip. The final contextual analysis loads retained CSV exports rather than unavailable original Parquet wrappers, retaining row order, detector rules, seeds, contrast definitions and multiplicity families. The forward runner executes the original fitting portion followed by the archived vectorised postprocessor and sensitivity script. It excludes a superseded slow bootstrap section and uses the published postprocessor's macro/micro results. The exact original code remains available for inspection.

## Verified scope and retained historical diagnostics

The checked-in `docs/reproduction_checks/*.json` list every recomputed table and every archived table not recomputed by the main runner. Historical exploratory/annotation/token-length diagnostics remain in `expected/` and `historical_audit_scripts/` for provenance; they are not silently counted as newly verified outputs. No independent human annotation was performed. The matched national result uses frozen machine-assisted correctness labels; its taxonomy source and precursor snapshot are included for provenance, while the runnable analysis starts from the frozen canonical-generation table.

Original v8/v9 runtime-wrapper logs were not recovered. Retained CSVs preserve the available model task, probe, template, arm, subject, candidate, context and output fields; these cannot reconstruct missing token-level or runtime metadata. No claim is made that this repository can regenerate the original pretrained models or reproduce the full pretraining corpus. Training exposure counts retain their inspected/indexed scope and are not full-corpus frequencies.

## Encoding discrepancy in the single-case control

The old Windows CSV exporter decoded UTF-8 text as GBK in a subset of outputs (for example, `Almería` became `Almer铆a`). Original JSON and old CSV bytes are both retained. The UTF-8 replay changes only the `text` column in 65/1,170 output rows and 38/788 failure rows. Candidate scores, labels, metrics and training checks match. The comparison checker explicitly verifies the exact reversible UTF-8-to-GBK mapping; it does not tolerate arbitrary text changes. See the recorded report for authoritative row counts.

## Private access and publication

Keep the repository Private. No public licence, DOI or public release is asserted. A private repository URL alone does not provide anonymous reviewer access; the journal's review access route remains to be settled. The anonymous manuscript should not be given an identifying repository URL during review without following the journal's procedure.
