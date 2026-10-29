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
