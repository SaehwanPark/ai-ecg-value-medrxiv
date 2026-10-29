#!/usr/bin/env python3
"""Render the five main manuscript figures from aggregate inputs.

Inputs live under ``data/``; outputs are written under ``outputs/``. The renderer
performs no fitting or resampling. It is a path-configured copy of the study's
frozen submission-figure renderer.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_ecg_value.figures import run_submission_figures  # noqa: E402


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--artifacts-root", type=Path, default=ROOT / "data/artifacts")
  parser.add_argument("--tables-dir", type=Path, default=ROOT / "data/tables")
  parser.add_argument("--clinical-tables-dir", type=Path, default=ROOT / "data/clinical_tables")
  parser.add_argument("--figures-dir", type=Path, default=ROOT / "outputs/figures")
  parser.add_argument("--qa-report", type=Path, default=ROOT / "outputs/figure_collision_report.md")
  args = parser.parse_args()

  result = run_submission_figures(
    figures_dir=args.figures_dir,
    tables_dir=args.tables_dir,
    artifacts_root=args.artifacts_root,
    clinical_tables_dir=args.clinical_tables_dir,
    qa_report_path=args.qa_report,
  )
  for name, paths in result["figures"].items():
    print(f"{name}: {len(paths)} file(s)")
  print(f"QA report: {result['qa_report_path']}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
