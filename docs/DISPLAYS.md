# Display map

The public package regenerates the five main figures and renders every table CSV
to Markdown. Display IDs below mirror the medRxiv manuscript. Supplementary
tables are distributed as their frozen CSV inputs; see `data/manifests/tables.json`
for the complete inventory.

## Main figures

| Display | Regenerated file | Source inputs |
|---|---|---|
| Figure 1 | `outputs/figures/figure1_cohort_evidence_flow.{png,svg}` | `data/artifacts/cohort/cohort_counts.json` |
| Figure 2 | `outputs/figures/figure2_discrimination_attenuation.{png,svg}` | `table2_overall_performance.csv`, `table3_incremental_value_and_controls.csv`, `table6_limitation_mitigation.csv`, `attenuation_comparison.csv` |
| Figure 3 | `outputs/figures/figure3_review_capacity_5_10.{png,svg}` | `scientific_analysis_results.json`, `supp_risk_strata.csv` |
| Figure 4 | `outputs/figures/figure4_subgroup_heterogeneity.{png,svg}` | `table4_subgroup_gain.csv`, `table3_incremental_value_and_controls.csv` |
| Figure 5 | `outputs/figures/figure5_pretraining_label_efficiency.{png,svg}` | `label_efficiency_long.csv`, `supp_label_efficiency.csv`, `supp_random_encoder_control.csv` |

## Main tables

| Display | CSV | Notes |
|---|---|---|
| Table 1 | `table1_cohort_characteristics.csv` | Cohort characteristics by split |
| Table 2 | `table2_overall_performance.csv` | Six prespecified models; AUROC/AUPRC/Brier/log loss with intervals |
| Table 3 | `table3_incremental_value_and_controls.csv` | Paired contrasts, add-on increments, controls |
| Table 4 | `table4_subgroup_gain.csv` | Subgroup gains underlying Figure 4 (the manuscript's Table 4 also draws on the clinical-extension CSV) |

## v3.1.0 revision displays (medRxiv revision v2)

The v3.1.0 manuscript splits the preprint-era Table 4 into Table 4
(discrimination) and Table 5 (event capture), moves the PPV/sensitivity
companions to Supplementary Table S8, and adds the fourth Figure 3 panel.
Sources:

| Display | Source inputs | Notes |
|---|---|---|
| Table 5 rows 1–2 (ECG-only; with `C_basic`) | `data/receipts/primary_capacity_operating_points.json` | Prespecified ECG-only row; post hoc context-adjusted row |
| Table 5 row 3 (separately refitted expanded report) | `data/receipts/lm1b_paired_lists_receipt.json` + `data/tables/lm1b_paired_lists.csv` | Post hoc; paired intervals + flagged-patient/captured-death identity |
| Table 5 row 4 (broad context, `C_broad`) | `data/tables/table6_limitation_mitigation.csv`, rows `L93_capacity_0.05*` and `L93_capacity_0.1*` | Post hoc; **not** sourced from the morphology risk-strata file |
| Supplementary Table S8 (rows 1–6) | both receipts (PPV/sensitivity fields) | Per-list descriptive shares; the `C_broad` row is intentionally absent (its ledger carries capture counts only) |
| Figure 3 panels a–b | `data/receipts/primary_capacity_operating_points.json` | Unchanged values from v3.0.14 |
| Figure 3 panel c | `data/tables/supp_risk_strata.csv` (all row) | Morphology-vs-AI 5% death exchange (126/136/95) |
| Figure 3 panel d | `data/receipts/lm1b_paired_lists_receipt.json` | Post hoc refitted report exchange (172/90/89 deaths) with the 654/1,272 flagged-patient annotation |

Rebuild: `uv run python scripts/build_figure3_medrxiv_v2.py` (verifies
`data/manifests/figure3_source_manifest.json` fail-closed, then draws with
zero-violation layout QA). Built images are semantic replays (values, panels,
labels, layout), not byte-identical copies across environments.

## Supplementary inputs

All committed supplementary table CSVs are copied under `data/tables/`. Their
display numbers and titles are listed in `data/manifests/tables.json` and in the
manuscript supplement. The public package does not regenerate the supplementary
figures from raw study data; their aggregate source CSVs remain available in the
same directory where applicable.

## Status conventions

- `prespecified` / `post hoc` labels travel in the table columns and in the
  figure artwork.
- Bootstrap intervals, seed ranges, and descriptive point estimates are distinct
  quantities; the source columns name the interval type and must not be relabelled.
- Figure 1 uses aggregate counts only. The private study pipeline remains the
  source of the patient-level cohort, splits, and predictions.
