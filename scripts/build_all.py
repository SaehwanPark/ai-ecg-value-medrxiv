#!/usr/bin/env python3
"""Build tables and figures (including the v3.1.0 four-panel Figure 3), then run the value checks."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(script: str) -> None:
  subprocess.run([sys.executable, str(ROOT / "scripts" / script)], cwd=ROOT, check=True)


def main() -> int:
  _run("build_tables.py")
  _run("build_figures.py")
  _run("build_figure3_medrxiv_v2.py")
  _run("check_values.py")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
