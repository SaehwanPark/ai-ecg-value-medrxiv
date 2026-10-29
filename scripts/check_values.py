#!/usr/bin/env python3
"""Fail loudly if the copied aggregate tables no longer carry the headline values.

These checks pin only the estimates quoted in the abstract/main text. They are a
copy-integrity guard, not a re-analysis.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

TABLES = Path("data/tables")


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

  if failures:
    print("Headline-value checks failed:")
    for failure in failures:
      print(f"  - {failure}")
    return 1
  print("Headline-value checks passed.")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
