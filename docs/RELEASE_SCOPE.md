# Release scope and evidence inventory

| Material | Current location | Access status |
|---|---|---|
| Figure source CSVs and saved MATLAB inputs | matlab/data | Included in this local snapshot; private publication intended |
| MATLAB figure code | matlab/design, matlab/src, matlab/run_paper_figures.m | Included |
| Frozen development and extension context protocols | protocols | Included |
| Figure reference PDFs | figures/reference_exports | Included |
| Complete longitudinal raw generations and audit records | Original project archive | Not included yet |
| Complete candidate-scoring raw outputs and analysis scripts | Original project archive | Not included yet |
| Complete context-generation raw outputs and clustered-bootstrap/Holm analysis scripts | Original project archive | Not included yet |
| Full pretrained weights and training corpora | Original providers | Not redistributed |
| Third-party national-capital source workbook | Original project/provider | Not redistributed in this snapshot; scope derivative included |
| Manuscript, author page, private account and server files | Original project | Excluded |

## Before a scientific public release

1. Locate and add the exact scripts and raw records underlying manuscript estimates, with dependencies and commands. Do not describe figure-only reproduction as full statistical reproduction.
2. Verify raw-output IDs against frozen protocols and figure source tables; retain failed and uncertain cases.
3. Document permitted redistribution and selected licences for original code/data, with third-party sources attributed separately.
4. Confirm a reviewer access route suitable for double-anonymised review.
5. Create a stable version/tag and, if appropriate, an archival deposit with a DOI after release scope is approved.
6. Update the paper's availability statement only after actual access is verified.

Original scientific values in copied MAT/CSV inputs have not been changed. Machine-specific absolute paths in the provenance JSON have been replaced with `project_archive/` references in this repository copy only. The original archive is unchanged.
