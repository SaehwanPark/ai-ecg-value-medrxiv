#!/usr/bin/env python3
"""Render Markdown tables from the committed aggregate CSV files.

The CSV values are copied verbatim; this script only changes presentation. It
also writes ``outputs/tables/index.md`` so every table is discoverable.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _escape(value: str) -> str:
  return value.replace("|", r"\|").replace("\n", " ").strip()


def render_csv(path: Path) -> str:
  with path.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.reader(handle))
  if not rows:
    return f"# {path.stem}\n\n_Empty table._\n"
  width = len(rows[0])
  if any(len(row) != width for row in rows):
    raise ValueError(f"{path.name}: ragged CSV rows")
  header = "| " + " | ".join(_escape(cell) for cell in rows[0]) + " |"
  divider = "|" + "|".join("---" for _ in rows[0]) + "|"
  body = ["| " + " | ".join(_escape(cell) for cell in row) + " |" for row in rows[1:]]
  return "\n".join([f"# {path.stem}", "", header, divider, *body, ""])


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--tables-dir", type=Path, default=Path("data/tables"))
  parser.add_argument("--output-dir", type=Path, default=Path("outputs/tables"))
  args = parser.parse_args()

  csv_files = sorted(args.tables_dir.glob("*.csv"))
  if not csv_files:
    raise SystemExit(f"No CSV files found under {args.tables_dir}")
  args.output_dir.mkdir(parents=True, exist_ok=True)
  index = ["# Table inventory", "", f"Source directory: `{args.tables_dir}`", ""]
  for path in csv_files:
    rendered = render_csv(path)
    (args.output_dir / f"{path.stem}.md").write_text(rendered, encoding="utf-8")
    with path.open(newline="", encoding="utf-8") as handle:
      rows = sum(1 for _ in csv.reader(handle)) - 1
    index.append(f"- [{path.stem}]({path.stem}.md) — {rows} data rows")
  (args.output_dir / "index.md").write_text("\n".join(index) + "\n", encoding="utf-8")
  print(f"Rendered {len(csv_files)} tables into {args.output_dir}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
