# Replicating the analysis

The package ships the same `ecg_rep` pipeline code that produced the
manuscript's results (pinned seeds, frozen extraction, prespecified
configuration). Given the [required dataset layout](DATA_LAYOUT.md), the
stages below re-run the analysis end to end. Every stage is a single
`ecg-rep` command with defaults that implement the study's directory
organization; run them from the repository root after `uv sync`.

Set the data roots once per shell (or pass the flags on every command):

```bash
export MIMIC_ROOT=~/data/mimiciv/3.1
export MIMIC_ECG_ROOT=~/data/mimic-iv-ecg/1.0
```

## Stage map

| Stage | Command (defaults shown) | Consumes → produces |
|---|---|---|
| 0. Environment preflight | `ecg-rep preflight` | data roots → acceptance report under `artifacts/preflight/` |
| 1. Cohort and splits | `ecg-rep cohort` | MIMIC-IV + MIMIC-IV-ECG → `artifacts/cohort/primary_cohort.parquet`, split assignments, cohort-flow report (seed 42) |
| 2. Representations | `ecg-rep representations` | cohort + waveforms + D-BETA checkpoint → `artifacts/representations/` (15 CIIS features; frozen 768-d D-BETA embeddings) |
| 3. Care setting | `ecg-rep clinical-setting` | cohort + transfers/icustays → `artifacts/clinical/setting/` |
| 4. Acute context (Cacute) | `ecg-rep clinical-acute-context` | cohort + setting + labs → `artifacts/clinical/context/` |
| 5. Broad context (Cbroad) | `ecg-rep cmax-feasibility` then `ecg-rep cmax-evaluate` | cohort + labs (+ `configs/cmax_context_v1.yaml`) → `artifacts/clinical/cmax/` |
| 6. Probes and paired evaluation | `ecg-rep evaluate` | representations + context → `artifacts/probe/` (locked ridge-logistic probes, paired AUROC/AUPRC/proper scores, 1,000-resample intervals) |
| 7. Scientific analysis | `ecg-rep analyze` | probe artifacts → `artifacts/analysis/` |
| 8. Controls and extensions | `ecg-rep random-control`, `pretraining-label-efficiency`, `pretraining-ami-control`, `clinical-subgroups`, `subgroup-multiplicity`, `refit-uncertainty`, `bootstrap-estimand-sensitivity`, `clinical-dca` | representations/probes → per-analysis artifact and report directories |
| 9. CODE-15% transport | `ecg-rep external-code15 audit`, `… prepare`, `… extract`, then `ecg-rep transport` and `ecg-rep code15-recalibrate` (+ `configs/code15_recalibration_v1.yaml`) | CODE-15% → `artifacts/external/code15pct/`, `artifacts/external/code15/` (seed 1912) |
| 10. Selective-processing analyses | `ecg-rep probability-cascade-run` (plus `-fit-routing` / `-replay`) | frozen ECG-only scores → cascade/DCA artifacts |
| 11. Measurement reference | `ecg-rep ptbxl-measurement-validity` | PTB-XL/PTB-XL+ → `artifacts/external/ptbxl/` |
| 12. Manuscript displays | `ecg-rep manuscript-ledger`, `manuscript-reporting`, `submission-figures` | study artifacts → committed CSVs/JSON, figures |
| 13. Aggregate display build | `uv run python scripts/build_all.py` | committed aggregate inputs → `outputs/figures`, `outputs/tables` |

Each command accepts explicit `--output-dir` / `--report-path` flags; the
defaults above are the layout the study used. Inspect any stage with
`ecg-rep <command> --help`.

## Notes for replicators

- **Seeds and prespecification.** Cohort/split seed 42, CODE seed 1912, probe
  seed 42, and the remaining resampling seeds are pinned in the code and
  configs; Methods S6 of the manuscript records the full seed/provenance map.
  Validation tuning selects the probe's regularization on validation log loss
  only, and the test split is evaluated once.
- **Configuration files.** `configs/*.yaml` carry the prespecified
  parameters (context definitions, phenotype dictionary, label protocols,
  recalibration policy). Pass them with the `--config` flags shown by each
  command's help.
- **Compute.** D-BETA extraction runs on CPU, Apple MPS, or CUDA
  (`--device auto`); CIIS extraction parallelizes with `--ciis-workers`.
  The frozen D-BETA checkpoint downloads from Hugging Face at the pinned
  revision on first use.
- **Audit and forensics commands.** The CLI also carries the study's
  audit/forensics commands (`ascertainment-audit`, `paired-valid-audit`,
  `history-sensitivity`, deployment/provenance checks, and related
  `manuscript-*` reporting stages). They are part of the study record but are
  not required for the analysis stages above.
- **Scope discipline.** The code re-runs the study's analyses; it does not
  redistribute any data. Results depend on the exact dataset releases listed
  in [DATA_LAYOUT.md](DATA_LAYOUT.md); the release manifests and the
  manuscript's Supplementary Data S1 ledger provide the hash anchors for
  confirming that the right source data were used.

## What is *not* reproduced from raw data

The committed aggregate inputs (`data/`) and the display builders reproduce
every table and figure without re-running the pipeline, which is the fastest
verification path. The full raw-data pipeline re-derives those aggregates from
the source datasets; it requires the credentialed MIMIC releases and the
compute noted above.
