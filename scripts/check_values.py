#!/usr/bin/env python3
"""Fail loudly if the copied aggregate tables no longer carry the headline values.

These checks pin only the estimates quoted in the abstract/main text and the
v3.1.0 (medRxiv revision v2) additions: all four fixed-capacity Table 5 rows
(including the broad-context row from the frozen limitation-mitigation ledger),
the Supplementary Table S8 operating points, the LM1B companion ledger against
its receipt, and the panel-c death-identity consistency. They are a
copy-integrity guard, not a re-analysis.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

TABLES = Path("data/tables")
RECEIPTS = Path("data/receipts")
MANIFESTS = Path("data/manifests")


def _rows(name: str) -> list[dict[str, str]]:
  with (TABLES / name).open(newline="", encoding="utf-8") as handle:
    return list(csv.DictReader(handle))


def _close(actual: float, expected: float, tolerance: float = 5e-4) -> bool:
  return math.isclose(actual, expected, abs_tol=tolerance)


def main() -> int:
  failures: list[str] = []

  table2 = {row["model"]: row for row in _rows("table2_overall_performance.csv")}
  for model, expected in (("M_A_ECG", 0.737), ("M_B_ECG", 0.812),
                          ("M_A", 0.784), ("M_B", 0.837)):
    actual = float(table2[model]["auroc"])
    if not _close(actual, expected):
      failures.append(f"Table 2 {model} AUROC: {actual:.4f}, expected {expected:.3f}")

  table3 = {(row["contrast"], row["metric"]): row
            for row in _rows("table3_incremental_value_and_controls.csv")}
  for contrast, expected in (("ECG-only: B - A", 0.075),
                             ("Context-adjusted: B - A", 0.053)):
    actual = float(table3[(contrast, "auroc")]["estimate"])
    if not _close(actual, expected):
      failures.append(f"Table 3 {contrast}: {actual:.4f}, expected {expected:.3f}")

  table1 = {row["variable"]: row for row in _rows("table1_cohort_characteristics.csv")}
  patients = table1["Patients"]
  if patients["test"] != "25,760" or patients["test_analytic"] != "25,430":
    failures.append("Table 1 test denominators no longer match 25,760 / 25,430")

  # ---- v3.1.0 (medRxiv revision v2) fixed-capacity checks -------------------
  primary = json.loads((RECEIPTS / "primary_capacity_operating_points.json").read_text(encoding="utf-8"))
  lm1b = json.loads((RECEIPTS / "lm1b_paired_lists_receipt.json").read_text(encoding="utf-8"))
  if primary["population"]["test_population_sha256"] != lm1b["population"]["test_population_sha256"]:
    failures.append("capacity receipts describe different locked-test populations")
  if (primary["population"]["n_test"], primary["population"]["n_events"]) != (25430, 924):
    failures.append("primary capacity receipt is not the 25,430/924 locked test")

  # Table 5 rows 1-2: prespecified ECG-only and post hoc C_basic (from the primary receipt).
  expected_capacity = {
    ("ecg_only", "0.05"): (163, 236, 73.0, 43.0, 96.02499999999998),
    ("ecg_only", "0.1"): (311, 389, 78.0, 46.975, 110.0),
    ("with_c_basic", "0.05"): (221, 262, 41.0, 10.975000000000001, 67.0),
    ("with_c_basic", "0.1"): (364, 422, 58.0, 25.0, 90.0),
  }
  for (family, key), (a, b, delta, lo, hi) in expected_capacity.items():
    row = primary["capacities"][family][key]
    if (int(row["a_events_captured"]), int(row["b_events_captured"])) != (a, b):
      failures.append(f"Table 5 {family}@{key}: capture counts drifted")
    if not _close(float(row["delta_events_captured"]), delta, 1e-9):
      failures.append(f"Table 5 {family}@{key}: paired delta drifted")
    if [float(row["delta_events_ci"][0]), float(row["delta_events_ci"][1])] != [lo, hi]:
      failures.append(f"Table 5 {family}@{key}: interval bounds drifted")

  # Table 5 row 3: separately refitted expanded report (LM1B receipt).
  cap05, cap10 = lm1b["capacities"]["0.05"], lm1b["capacities"]["0.1"]
  if (cap05["b_events_captured"], cap05["a2_events_captured"]) != (262, 261) or \
      (cap10["b_events_captured"], cap10["a2_events_captured"]) != (422, 408):
    failures.append("Table 5 refitted report row capture counts drifted")
  if [round(cap05["delta_events_ci"][0]), round(cap05["delta_events_ci"][1])] != [-25, 22]:
    failures.append("Table 5 refitted report 5% interval drifted")
  if [round(cap10["delta_events_ci"][0]), round(cap10["delta_events_ci"][1])] != [-11, 41]:
    failures.append("Table 5 refitted report 10% interval drifted")
  identity05, identity10 = cap05["identity"], cap10["identity"]
  if identity05["flagged_list_intersection"] != 654 or cap05["n_flagged_per_list"] != 1272:
    failures.append("LM1B 5% flagged-list overlap drifted")
  if identity10["flagged_list_intersection"] != 1546 or cap10["n_flagged_per_list"] != 2543:
    failures.append("LM1B 10% flagged-list overlap drifted")
  if (identity05["captured_by_both"], identity05["captured_by_b_only"], identity05["captured_by_a2_only"]) != (172, 90, 89):
    failures.append("LM1B 5% captured-death identity drifted")
  if (identity10["captured_by_both"], identity10["captured_by_b_only"], identity10["captured_by_a2_only"]) != (309, 113, 99):
    failures.append("LM1B 10% captured-death identity drifted")
  if int(identity05["flagged_list_union"]) != 2 * cap05["n_flagged_per_list"] - identity05["flagged_list_intersection"]:
    failures.append("LM1B 5% flagged-list union identity inconsistent")

  # Table 5 row 4: broad context comes from the frozen limitation-mitigation
  # ledger (L93_capacity_* rows) — NOT from the morphology risk-strata file.
  ledger = {row["id"]: row for row in _rows("table6_limitation_mitigation.csv")}
  l93_expectations = {
    "L93_capacity_0.05": (25.0, -2.02, 44.0),
    "L93_capacity_0.05_Cmax+A": (296.0, None, None),
    "L93_capacity_0.05_Cmax+B": (321.0, None, None),
    "L93_capacity_0.1": (28.0, -0.02, 49.0),
    "L93_capacity_0.1_Cmax+A": (459.0, None, None),
    "L93_capacity_0.1_Cmax+B": (487.0, None, None),
  }
  for key, (estimate, _lo, _hi) in l93_expectations.items():
    row = ledger.get(key)
    if row is None:
      failures.append(f"broad-context capacity row missing from the ledger: {key}")
      continue
    if not _close(float(row["estimate"]), estimate, 1e-9):
      failures.append(f"broad-context capacity row {key}: estimate drifted")

  # Supplementary Table S8 operating points (rows 1-4 primary receipt, 5-6 LM1B).
  s8_expectations = [
    ("ecg_only", "0.05", "b_ppv", 18.6), ("ecg_only", "0.05", "a_ppv", 12.8),
    ("ecg_only", "0.1", "b_ppv", 15.3), ("ecg_only", "0.1", "a_ppv", 12.2),
    ("with_c_basic", "0.05", "b_ppv", 20.6), ("with_c_basic", "0.05", "a_ppv", 17.4),
    ("with_c_basic", "0.1", "b_ppv", 16.6), ("with_c_basic", "0.1", "a_ppv", 14.3),
    ("with_c_basic", "0.05", "b_sensitivity", 28.4), ("with_c_basic", "0.05", "a_sensitivity", 23.9),
    ("with_c_basic", "0.1", "b_sensitivity", 45.7), ("with_c_basic", "0.1", "a_sensitivity", 39.4),
  ]
  for family, key, field, expected in s8_expectations:
    row = primary["capacities"][family][key]
    if not _close(round(float(row[field]) * 100, 1), expected, 5e-2):
      failures.append(f"S8 {family}@{key} {field}: drifted")
  for key, (b_ppv, a2_ppv, b_sens, a2_sens) in (("0.05", (20.6, 20.5, 28.4, 28.2)), ("0.1", (16.6, 16.0, 45.7, 44.2))):
    row = lm1b["capacities"][key]
    for field, expected in (("b_ppv", b_ppv), ("a2_ppv", a2_ppv), ("b_sensitivity", b_sens), ("a2_sensitivity", a2_sens)):
      if not _close(round(float(row[field]) * 100, 1), expected, 5e-2):
        failures.append(f"S8 LM1B@{key} {field}: drifted")

  # The LM1B companion ledger must reconcile with its receipt.
  lm1b_rows = {row["id"]: row for row in _rows("lm1b_paired_lists.csv")}
  ledger_expectations = {
    "L91B_capacity_0.05_delta_events": (1.0, -25.025, 22.024999999999977),
    "L91B_capacity_0.1_delta_events": (14.0, -11.0, 41.0),
    "L91B_capacity_0.05_flagged_list_intersection": (654.0, None, None),
    "L91B_capacity_0.1_flagged_list_intersection": (1546.0, None, None),
  }
  for key, (estimate, lo, hi) in ledger_expectations.items():
    row = lm1b_rows.get(key)
    if row is None:
      failures.append(f"LM1B ledger row missing: {key}")
      continue
    if not _close(float(row["estimate"]), estimate, 1e-9):
      failures.append(f"LM1B ledger row {key}: estimate drifted")
    if lo is not None and (float(row["interval_lower"]), float(row["interval_upper"])) != (lo, hi):
      failures.append(f"LM1B ledger row {key}: interval drifted")

  # Panel-c death identity: the strata all-row must stay consistent with the
  # capacity receipt's with-C_basic 5% row (126 shared + 136/95 exclusive).
  strata_all = next(row for row in _rows("supp_risk_strata.csv") if row["stratum"] == "all")
  captured_by_b = int(strata_all["capacity_events_captured_by_b"])
  b_only = int(strata_all["capacity_events_b_only"])
  a_only = int(strata_all["capacity_events_a_only"])
  shared = captured_by_b - b_only
  if (shared, b_only, a_only) != (126, 136, 95):
    failures.append(f"panel-c death identity drifted: shared/b_only/a_only = {shared}/{b_only}/{a_only}")
  ctx05 = primary["capacities"]["with_c_basic"]["0.05"]
  if (int(ctx05["a_events_captured"]), int(ctx05["b_events_captured"])) != (
      int(strata_all["capacity_events_captured_by_a"]), captured_by_b):
    failures.append("panel-c strata all-row disagrees with the capacity receipt")

  # The committed figure-source manifest must still hash the real inputs.
  manifest = json.loads((MANIFESTS / "figure3_source_manifest.json").read_text(encoding="utf-8"))
  for entry in manifest["inputs"]:
    path = Path(entry["path"])
    if not path.is_file():
      failures.append(f"figure-source input missing: {entry['path']}")
    elif hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
      failures.append(f"figure-source input hash mismatch: {entry['path']}")

  if failures:
    print("Headline-value checks failed:")
    for failure in failures:
      print(f"  - {failure}")
    return 1
  print("Headline-value checks passed.")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
