# Required dataset layout

The replication code reads the source datasets from the locations below. All
paths can be overridden with command-line flags or environment variables; the
layout itself mirrors the official release distributions, so no renaming is
required. The code never redistributes any of these datasets — obtain each one
from its provider under its own access terms.

## Directory organization

Place the downloaded releases under a common data root (the examples use
`~/data`; any root works if you pass the flags shown in
[REPLICATION.md](REPLICATION.md)):

```text
~/data/
  mimiciv/3.1/                  # MIMIC-IV v3.1 (PhysioNet, credentialed)
    hosp/
      patients.csv.gz
      admissions.csv.gz
      diagnoses_icd.csv.gz
      labevents.csv.gz
      transfers.csv.gz
    icu/
      icustays.csv.gz
      chartevents.csv.gz
  mimic-iv-ecg/1.0/             # MIMIC-IV-ECG v1.0 (PhysioNet, credentialed)
    record_list.csv
    files/                      # WFDB signal files (<subject_id>/<study_id>.dat/.hea)
  code15pct/                    # CODE-15% v1.0.0 (Zenodo record 4916206, open)
    exams.csv
    exams_part0.hdf5 … exams_partN.hdf5   # as distributed (also accepts .hdf5.zip)
  ptbxl/                        # PTB-XL v1.0.3 (PhysioNet, open)
    ptbxl_database.csv
    records100/ records500/     # WFDB signal folders, as distributed
  ptbxl_plus/                   # PTB-XL+ v1.0.1 (PhysioNet, open)
    12sl_features.csv           # GE 12SL measurement export
```

## Access terms

- **MIMIC-IV and MIMIC-IV-ECG** require PhysioNet credentialing, completion of
  the required training, and a data-use agreement. The study used them under
  those terms; the same terms govern any replication.
- **CODE-15%** is distributed publicly on Zenodo under its own license.
- **PTB-XL and PTB-XL+** are distributed publicly on PhysioNet under their own
  licenses.
- **D-BETA checkpoint** `Manhph2211/D-BETA` (revision
  `20ff3ccce1759d7d629171e15befafa9a424d2ca`) is a third-party model hosted on
  Hugging Face. It is publicly discoverable but gates the model files behind
  acceptance of its terms; the adapter (`ecg-rep representations …`)
  downloads it through `huggingface_hub` at the pinned revision, and continued
  availability rests with that repository's maintainers.

## Path configuration

| Dataset | CLI flags (per command) | Environment variables | Default |
|---|---|---|---|
| MIMIC-IV | `--mimic-root` | `MIMIC_ROOT` | `~/data/mimiciv/3.1` |
| MIMIC-IV-ECG | `--ecg-root` | `MIMIC_ECG_ROOT` (or `ECG_ROOT`) | `~/data/mimic-iv-ecg/1.0` |
| CODE-15% | `--data-dir` (external commands) | — | `~/data/code15pct/` |
| PTB-XL | `--ptbxl-root` | — | `~/data/ptbxl` |
| PTB-XL+ 12SL | `--features-12sl-path` | — | `~/data/ptbxl_plus/12sl_features.csv` |
| D-BETA checkpoint | pinned model id/revision | Hugging Face cache | `Manhph2211/D-BETA` |

Outputs are written under the working directory (`artifacts/`, `outputs/`,
`docs/`) by each command's `--output-dir` / `--report-path` flags, which
default to the layout used in [REPLICATION.md](REPLICATION.md).

## Integrity

After downloading, record the source-provenance state before any analysis:
`ecg-rep external-code15 audit --data-dir ~/data/code15pct/` inventories the
CODE-15% release (checksums optional), and the MIMIC commands write the record
list/patient table hashes into their manifests. The study's frozen-evidence
hashes are recorded in the release manifests and the manuscript's
Supplementary Data S1 ledger; a replication that obtains the same dataset
releases should reproduce them.
