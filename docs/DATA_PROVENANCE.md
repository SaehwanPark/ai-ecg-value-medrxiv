# Data provenance and scope

## What is included

- **The `ecg_rep` analysis pipeline** — the code that constructs the cohort,
  extracts the CIIS-derived morphology features and the frozen D-BETA
  embeddings, fits the ridge-logistic probes, runs the evaluation, uncertainty,
  transport, and selective-processing analyses, and generates the displays —
  with its pinned seeds and the prespecified `configs/*.yaml` parameters.
- Main and supplementary table CSVs as committed with the manuscript.
- The aggregate JSON/CSV inputs required by the five main figure functions.
- A small aggregate cohort-counts JSON for Figure 1.
- The aggregate-only population-identity receipt used by the cart add-on
  contrast gate.
- Renderer/build code and release manifests.

## What is excluded

- MIMIC-IV, MIMIC-IV-ECG, CODE-15%, and PTB-XL source records or waveforms.
- Patient identifiers, split assignments, embeddings, predictions, and models.
- Rendered manuscript DOCX/PDF files.

The pipeline reads the source datasets from the locations documented in
[DATA_LAYOUT.md](DATA_LAYOUT.md); each reader obtains the datasets from their
providers under the providers' access terms.

## Interpretation

MIMIC-IV is the pretraining data source for D-BETA, so the main comparisons are
in-domain downstream representation analyses, not external validation. CODE-15%
is pretraining-disjoint but uses a one-year horizon and is reported as transport,
not same-horizon validation. Review-list capture is a prioritization measure, not
prevented mortality or clinical benefit.

## Integrity

`data/manifests/checksums.sha256` lists the SHA-256 of every distributed input and
code file. `data/manifests/release.json` records the release version and scope.
`scripts/check_values.py` rechecks the headline table values after any copy.
