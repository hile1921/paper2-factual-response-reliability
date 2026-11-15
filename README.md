# Factual Response Reliability

Companion data and code for *Factual Response Reliability during Language Model Training and Contextual Answer Selection*.

## Status

Initial local repository, prepared on 2026-11-15 for private GitHub publication. This snapshot contains processed figure inputs, MATLAB figure code and frozen context protocols. It is not the complete raw-generation or statistical-analysis archive. No public release or DOI is claimed. Keep the GitHub repository private during preparation and handle reviewer access separately from public release.

## Contents

- `matlab/data/source_data/`: CSV tables used by manuscript figures, including checkpoint summaries, matched-fact records and context rates.
- `matlab/data/*.mat`: saved inputs loaded by the MATLAB entry point.
- `matlab/design/` and `matlab/src/`: plotting implementations and shared rendering utilities.
- `protocols/`: frozen development/extension protocols and national-capital scope table.
- `figures/reference_exports/`: seven reference vector PDFs from the current manuscript.
- `scripts/verify_files.py`: integrity and CSV readability check.
- `docs/RELEASE_SCOPE.md`: included evidence, excluded material and outstanding reproducibility work.
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

This private preparation snapshot has no open-source or open-data licence selected yet. Third-party models, training corpora and external reference data retain their original terms. Full model weights, complete corpora, retrieval indexes, author identity files and manuscript drafts are not included. A private GitHub URL alone does not provide anonymous reviewers access. Repository identity, release scope and archival DOI will be settled before the availability statement is updated.
