# AI-ECG incremental value — medRxiv replication package

Replication code and aggregate results for the medRxiv preprint:

> **Evaluating the Incremental Value of AI-Derived ECG Features Beyond Automated
> ECG Reports and Clinical Data for 30-Day Mortality: A MIMIC-IV Cohort Study**

Release `v3.0.11` (2026-10-24).

This package is the code companion of the preprint: it carries the **analysis
pipeline** that produced the study's results — cohort construction, CIIS-derived
ECG feature extraction, frozen D-BETA embedding extraction, probe fitting and
evaluation, uncertainty analyses, and display generation — together with the
committed aggregate inputs so every table and figure can also be rebuilt
without any source data.

## What this package contains

- **The `ecg_rep` analysis pipeline** (the same code, seeds, and prespecified
  configuration that produced the manuscript's numbers) with its
  `ecg-rep` command-line interface and the `configs/*.yaml` parameter files.
- **Main Tables 1–4** as committed aggregate CSV files, with a Markdown renderer.
- **Main Figures 1–5** as reproducible matplotlib code and their aggregate inputs.
- **Supplementary table CSVs** for the reported supplementary tables.
- A release manifest with SHA-256 checksums and display-to-source mappings.
- [DATA_LAYOUT.md](docs/DATA_LAYOUT.md) — the required directory organization
  for the source datasets.
- [REPLICATION.md](docs/REPLICATION.md) — the stage-to-command map for
  re-running the analysis from the raw data.

It **does not** contain patient-level data, ECG waveforms, model weights,
embeddings, patient predictions, or rendered manuscript DOCX/PDF files. The
source datasets are obtained by each reader from their providers under the
providers' access terms (see [DATA_LAYOUT.md](docs/DATA_LAYOUT.md)); the
previously trained D-BETA checkpoint is a third-party Hugging Face model gated
behind its own terms.

## Quick start

With [uv](https://docs.astral.sh/uv/) (Python ≥ 3.13; first sync downloads the
pipeline's scientific dependencies, including PyTorch):

```bash
uv sync
uv run python scripts/build_all.py       # rebuild every table/figure from committed aggregates
```

To re-run the analysis from the source datasets, follow
[REPLICATION.md](docs/REPLICATION.md) after arranging the data as described in
[DATA_LAYOUT.md](docs/DATA_LAYOUT.md) — for example:

```bash
export MIMIC_ROOT=~/data/mimiciv/3.1
export MIMIC_ECG_ROOT=~/data/mimic-iv-ecg/1.0
uv run ecg-rep preflight
uv run ecg-rep cohort
uv run ecg-rep representations
uv run ecg-rep evaluate
```

Outputs of the aggregate build:

```text
outputs/
  figures/                       # Figure 1–5, PNG + SVG
  tables/                        # Markdown renders of every CSV table
  figure_collision_report.md     # rendered-layout audit for the five figures
  tables/index.md                # table inventory
```

Run only one aggregate stage:

```bash
uv run python scripts/build_tables.py
uv run python scripts/build_figures.py
uv run python scripts/check_values.py
```

## Reproducibility contract

- The pipeline code, seeds, and configuration shipped here are the study's own;
  the stage commands re-run the prespecified analysis on the source datasets
  obtained under their access terms.
- Inputs to the aggregate display build are aggregate-only files copied from
  the frozen manuscript evidence. Figure code reads committed CSVs/JSON only;
  it performs no fitting, tuning, or resampling. Figure 1 reads a small
  aggregate counts file instead of the patient-level cohort table.
- Table CSVs retain full stored precision; Markdown renders show the stored
  values without recomputation or rounding.
- `scripts/check_values.py` asserts the headline estimates in Tables 2 and 3 so
  a broken input copy fails loudly.

## Scope and interpretation

The MIMIC-IV evaluation is **in-domain** for D-BETA because D-BETA was pretrained
on MIMIC-IV-ECG. CODE-15% is a pretraining-disjoint cohort with a different
(one-year) outcome horizon; it is not same-horizon external validation. Captured
review-list deaths describe prioritization at a fixed workload, not prevented
mortality or clinical benefit. See the manuscript and supplement for the full
estimands, uncertainty conventions, and limitations.

## Data and code terms

MIMIC-IV, MIMIC-IV-ECG, CODE-15%, and PTB-XL remain subject to their original
access terms; this repository distributes only code and aggregate results. No
reuse license has been assigned to this replication package. Contact the
corresponding author before redistribution or reuse.

## Repository layout

```text
data/
  tables/              # main and supplementary CSV tables
  clinical_tables/     # clinical-extension aggregate input (attenuation)
  artifacts/           # frozen aggregate JSON/CSV inputs for figures
  manifests/           # release, display, and checksum manifests
src/ecg_rep/           # the analysis pipeline (cohort → features → probes → evaluation → displays)
src/ai_ecg_value/      # aggregate figure renderer
configs/               # prespecified parameter files used by the pipeline
scripts/               # aggregate build and value-check entry points
docs/                  # display map, provenance notes, DATA_LAYOUT, REPLICATION
```

## Citation

Cite the medRxiv preprint and this replication release. Author list, affiliations,
and the final article DOI are reported in the manuscript.
