# AI-ECG incremental value — replication package

Code and aggregate results for the medRxiv preprint **"Evaluating the
Incremental Value of AI-Derived ECG Features Beyond Automated ECG Reports and
Clinical Data for 30-Day Mortality: A MIMIC-IV Cohort Study"** — release
`v3.0.14` (2026-10-28).

## What is inside

- `src/ecg_rep` — the analysis pipeline (cohort construction, CIIS-derived
  feature extraction, frozen D-BETA embedding extraction, probe fitting,
  evaluation, and display generation) with the `ecg-rep` command-line
  interface and the prespecified `configs/`.
- `data/` — the aggregate tables and figure inputs behind every displayed
  number, with checksum manifests.
- `scripts/` — rebuilds all tables and figures from the committed aggregates
  and re-checks the headline values.

No source data is distributed here. You obtain the datasets yourself under
their providers' access terms; see [docs/DATA_LAYOUT.md](docs/DATA_LAYOUT.md).

## Quick start

```bash
uv sync                                          # Python ≥ 3.13
uv run python scripts/build_all.py               # rebuild every table and figure
uv run python scripts/check_values.py            # verify headline estimates
```

To re-run the analysis from the raw data, follow
[docs/REPLICATION.md](docs/REPLICATION.md):

```bash
export MIMIC_ROOT=~/data/mimiciv/3.1
export MIMIC_ECG_ROOT=~/data/mimic-iv-ecg/1.0
uv run ecg-rep preflight
uv run ecg-rep cohort
uv run ecg-rep representations
uv run ecg-rep evaluate
```

## Documentation

- [docs/REPLICATION.md](docs/REPLICATION.md) — stage-by-stage analysis map.
- [docs/DATA_LAYOUT.md](docs/DATA_LAYOUT.md) — required dataset layout and access terms.
- [docs/DATA_PROVENANCE.md](docs/DATA_PROVENANCE.md) — what ships and what does not.
- [docs/DISPLAYS.md](docs/DISPLAYS.md) — display-to-source mapping.

## Scope

The MIMIC-IV evaluation is in-domain for D-BETA (it was pretrained on
MIMIC-IV-ECG); CODE-15% is a different-horizon transport cohort, not
same-horizon validation. Captured review-list deaths measure prioritization,
not prevented mortality. See the preprint for the full estimands and
limitations.

## Terms and citation

The source datasets remain under their original access terms; this repository
distributes only code and aggregate results. Cite the medRxiv preprint and
this release; author list and DOI are reported in the preprint.
