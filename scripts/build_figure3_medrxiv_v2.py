#!/usr/bin/env python3
"""Render the v3.1.0 (medRxiv revision v2) four-panel Figure 3 from aggregates.

Inputs are the aggregate-only receipts and tables committed in this release
(``data/receipts/``, ``data/tables/``); outputs are written under ``outputs/figures/``.
The renderer performs no fitting and no resampling. Before drawing, it verifies
every input against the committed figure-source manifest
(``data/manifests/figure3_source_manifest.json``) and re-derives every displayed
quantity through a fail-closed validator (counts, intervals, PPV/sensitivity,
cross-receipt population identity, list-identity sums) — any mismatch raises.

The rendered image is a semantic replay: byte-identity across environments is
not guaranteed (fonts/matplotlib builds), so the release contract is the
fail-closed value validation plus the renderer's zero-violation layout QA, not
byte equality. The committed figure-source manifest hashes the *inputs* and the
*renderer*; the built image inherits their values by construction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_ecg_value import figures  # noqa: E402

MANIFEST = ROOT / "data/manifests/figure3_source_manifest.json"


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_inputs_against_manifest() -> None:
  """Fail closed when any figure input or renderer no longer matches the manifest."""
  manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
  if manifest.get("schema_version") != "figure3_source_manifest_public_v1":
    raise ValueError(f"{MANIFEST}: unexpected manifest schema")
  for entry in manifest["inputs"]:
    path = ROOT / entry["path"]
    if not path.is_file():
      raise FileNotFoundError(f"figure input missing: {entry['path']}")
    actual = _sha256(path)
    if actual != entry["sha256"]:
      raise ValueError(f"figure input hash mismatch (stale or tampered release?): {entry['path']}")


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--tables-dir", type=Path, default=ROOT / "data/tables")
  parser.add_argument("--receipts-dir", type=Path, default=ROOT / "data/receipts")
  parser.add_argument("--figures-dir", type=Path, default=ROOT / "outputs/figures")
  args = parser.parse_args()
  verify_inputs_against_manifest()
  args.figures_dir.mkdir(parents=True, exist_ok=True)
  figures.apply_style()
  paths = figures.figure3_review_capacity_5_10_medrxiv_v2(
    args.figures_dir, args.tables_dir, args.receipts_dir,
    stem="figure3_review_capacity_5_10_medrxiv_v2")
  for path in paths:
    print(f"wrote {path}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
