# Factual Response Reliability

Companion data and code for *Factual Response Reliability during Language Model Training and Contextual Answer Selection*.

## Status

Private preparation repository. Private visibility was directly verified in the signed-in GitHub interface; no public release is authorised. The repository now includes the retained raw longitudinal and context generations, frozen audit/evidence records, original analysis scripts, portable CPU reproduction entry points, expected result tables and figure sources.

Five CPU analysis chains have been executed and compared against 55 archived tables. All scientific/statistical comparisons passed. A reversible text-encoding discrepancy is recorded for two single-case-control CSVs; numerical results and labels agree. All 319,410 longitudinal raw responses were joined to the archived output table and replayed through the original offline evidence gate with zero field differences. These checks replay machine-assisted rules and are not independent human verification. See `docs/REPRODUCIBILITY.md` and `docs/reproduction_checks/`.

## Contents

- `matlab/data/source_data/`: CSV tables used by manuscript figures, including checkpoint summaries, matched-fact records and context rates.
- `matlab/data/*.mat`: saved inputs loaded by the MATLAB entry point.
- `matlab/design/` and `matlab/src/`: plotting implementations and shared rendering utilities.
- `protocols/`: frozen development/extension protocols and national-capital scope table.
- `figures/reference_exports/`: seven reference vector PDFs from the current manuscript.
- `scripts/verify_files.py`: integrity and CSV readability check.
- `docs/RELEASE_SCOPE.md`: included evidence, excluded material and outstanding reproducibility work.
- `analysis/`: raw observations, frozen inputs, original analysis sources and expected result tables for five panels.
- `scripts/reproduce.py`: portable CPU statistics entry points.
- `scripts/verify_generated_records.py`: raw-response join and offline label/evidence replay.
- `FILE_MANIFEST.csv`: SHA256 checksums and file sizes for this snapshot.

## Reproduce the figures

In MATLAB, set the working folder to `matlab`, then run:

```matlab
run_paper_figures;
```

To render a subset: `run_paper_figures([2 4 5]);`. Results go to `matlab/output/`. The source was used with MATLAB R2026a in the manuscript workflow; no claim of compatibility testing on other MATLAB versions is made. A MATLAB licence is required. Figures use saved measurements and estimates; this command does not retrain models or recompute all inferential statistics. New rendering has not been performed while preparing this repository.

To verify the distributed files with Python 3 (standard library only):

```text
python scripts/verify_files.py
```

## Interpretation and units

Sampled responses, fact queries, target subjects and model checkpoints are different units. Conditional confirmed-error rate uses scorable responses as its denominator; all-output confirmed-error rate uses all responses. Candidate-set-normalised scores refer to the fixed candidate set. Automatic designated-entity detections are distinct from full factual verification. Context tables contain repeated pairs within subjects; inferential analyses must preserve subject clustering. Frozen protocols and manuscript definitions determine the analysis, not exploratory regrouping of the CSV files.

## Access and licensing

This private preparation snapshot has no open-source or open-data licence selected yet. Cached third-party reference snapshots are retained solely for private reproducibility preparation; their terms and attribution must be reviewed before a public release. Third-party models, training corpora and external reference data retain their original terms. Full model weights, complete corpora, retrieval indexes, author identity files and manuscript drafts are not included. A private GitHub URL alone does not provide anonymous reviewers access. Repository identity, release scope and archival DOI will be settled before the availability statement is updated.
