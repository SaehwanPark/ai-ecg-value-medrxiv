"""Results ledger: every manuscript number resolved from a named artifact key.

The ledger spec (`docs/manuscript/ledger_spec.csv`) lists each reportable quantity with
its source file and a key path. This module resolves the values, renders
`results_ledger.md` and `results_ledger.csv`, and in check mode fails when a committed
ledger no longer matches the artifacts. Values are never typed by hand.

Key paths are dot-separated. A bracketed selector picks the single list element (or
CSV row) whose fields match, e.g. `analyses.rq5_capacity_comparison[capacity_fraction=0.05].delta_events_captured`
or `[model=M_B_ECG].auroc`. Selector values compare numerically when both sides parse
as numbers, otherwise as strings.
"""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Any

_TOKEN = re.compile(r"\[([^\]]+)\]|([^.\[\]]+)")
LEDGER_COLUMNS: list[str] = [
  "id", "section", "quantity", "value", "source", "key", "claims",
  "key_lower", "key_upper", "workstream", "population", "analysis_status",
  "interval_type", "n", "n_events", "denominator_source", "n_key", "events_key",
]


@dataclass(frozen=True)
class SpecRow:
  """One reportable quantity and where it comes from."""

  id: str
  section: str
  quantity: str
  source: str
  key: str
  key_lower: str
  key_upper: str
  scale: float
  decimals: int
  unit: str
  claims: str
  workstream: str = ""
  population: str = ""
  analysis_status: str = ""
  interval_type: str = ""
  denominator_source: str = ""
  n_key: str = ""
  events_key: str = ""


def _as_number(value: Any) -> float | None:
  if isinstance(value, bool):
    return None
  if isinstance(value, (int, float)):
    return float(value)
  try:
    return float(str(value))
  except ValueError:
    return None


def _matches(element: Any, selector: str) -> bool:
  if not isinstance(element, dict):
    return False
  for clause in selector.split(","):
    field, _, expected = clause.partition("=")
    field, expected = field.strip(), expected.strip()
    if field not in element:
      return False
    actual = element[field]
    a_num, e_num = _as_number(actual), _as_number(expected)
    if a_num is not None and e_num is not None:
      if abs(a_num - e_num) > 1e-9:
        return False
    elif str(actual) != expected:
      return False
  return True


def resolve_key(data: Any, key: str) -> Any:
  """Follow a dot/selector key path through nested dicts and lists."""
  current = data
  for selector, field in _TOKEN.findall(key):
    if selector:
      if not isinstance(current, list):
        raise KeyError(f"selector [{selector}] applied to non-list in {key!r}")
      hits = [e for e in current if _matches(e, selector)]
      if len(hits) != 1:
        raise KeyError(f"selector [{selector}] matched {len(hits)} elements in {key!r}")
      current = hits[0]
    elif isinstance(current, dict):
      if field not in current:
        raise KeyError(f"field {field!r} missing in {key!r}")
      current = current[field]
    elif isinstance(current, list) and field.isdigit():
      current = current[int(field)]
    else:
      raise KeyError(f"cannot read {field!r} in {key!r}")
  return current


def load_source(path: Path) -> Any:
  """JSON documents load as-is; CSV files load as a list of row dicts."""
  if path.suffix == ".csv":
    with path.open(newline="", encoding="utf-8") as f:
      return list(csv.DictReader(f))
  return json.loads(path.read_text(encoding="utf-8"))


def read_spec(path: Path) -> list[SpecRow]:
  """Parse the ledger spec CSV."""
  with path.open(newline="", encoding="utf-8") as f:
    return [
      SpecRow(
        id=r["id"],
        section=r["section"],
        quantity=r["quantity"],
        source=r["source"],
        key=r["key"],
        key_lower=r.get("key_lower", "") or "",
        key_upper=r.get("key_upper", "") or "",
        scale=float(r.get("scale") or 1.0),
        decimals=int(r.get("decimals") or 3),
        unit=r.get("unit", "") or "",
        claims=r.get("claims", "") or "",
        workstream=r.get("workstream", "") or "",
        population=r.get("population", "") or "",
        analysis_status=r.get("analysis_status", "") or "",
        interval_type=r.get("interval_type", "") or "",
        denominator_source=r.get("denominator_source", "") or "",
        n_key=r.get("n_key", "") or "",
        events_key=r.get("events_key", "") or "",
      )
      for r in csv.DictReader(f)
    ]


def _fmt(value: Any, row: SpecRow) -> str:
  number = _as_number(value)
  if number is None:
    raise ValueError(f"{row.id}: non-numeric value {value!r}")
  return f"{number * row.scale:,.{row.decimals}f}"


def format_value(row: SpecRow, data: Any) -> str:
  """Point estimate with optional interval, scaled and rounded as the spec says."""
  text = _fmt(resolve_key(data, row.key), row)
  if row.key_lower and row.key_upper:
    lo = _fmt(resolve_key(data, row.key_lower), row)
    hi = _fmt(resolve_key(data, row.key_upper), row)
    text = f"{text} ({lo} to {hi})"
  return f"{text}{row.unit}"


def build_ledger(spec: Sequence[SpecRow], root: Path) -> list[dict[str, str]]:
  """Resolve every spec row against files under ``root``."""
  cache: dict[str, Any] = {}
  rows: list[dict[str, str]] = []
  for s in spec:
    if s.source not in cache:
      cache[s.source] = load_source(root / s.source)
    denominator_source = s.denominator_source or s.source
    if s.n_key or s.events_key:
      if denominator_source not in cache:
        cache[denominator_source] = load_source(root / denominator_source)
    rows.append({
      "id": s.id,
      "section": s.section,
      "quantity": s.quantity,
      "value": format_value(s, cache[s.source]),
      "source": s.source,
      "key": s.key,
      "claims": s.claims,
      "key_lower": s.key_lower,
      "key_upper": s.key_upper,
      "workstream": s.workstream,
      "population": s.population,
      "analysis_status": s.analysis_status,
      "interval_type": s.interval_type,
      "n": str(resolve_key(cache[denominator_source], s.n_key)) if s.n_key else "",
      "n_events": str(resolve_key(cache[denominator_source], s.events_key)) if s.events_key else "",
      "denominator_source": denominator_source if s.n_key or s.events_key else "",
      "n_key": s.n_key,
      "events_key": s.events_key,
    })
  return rows


def render_markdown(rows: Sequence[dict[str, str]]) -> str:
  """Group ledger rows by section into markdown tables."""
  lines = [
    "# Results ledger",
    "",
    "Generated by `uv run ecg-rep manuscript-ledger` from [`ledger_spec.csv`](ledger_spec.csv); do not edit by hand. "
    "Each value is read from the named source and key. `artifacts/` files are local and gitignored; "
    "`docs/` tables are committed. `uv run ecg-rep manuscript-ledger --check` fails if any value has drifted. "
    "Claim IDs refer to [`claims_register.md`](claims_register.md).",
    "",
  ]
  for section, group in groupby(rows, key=lambda r: r["section"]):
    lines += [f"## {section}", "", "| ID | Quantity | Value | Source | Key | Claims |", "|---|---|---|---|---|---|"]
    section_rows = list(group)
    for r in section_rows:
      key = r["key"].replace("|", "\\|")
      lines.append(f"| {r['id']} | {r['quantity']} | {r['value']} | `{r['source']}` | `{key}` | {r['claims']} |")
    for r in section_rows:
      if r.get("workstream"):
        lines.append(
          f"\n{r['id']}: {r['workstream']}; {r['analysis_status']}; {r['population']}; "
          f"n={r['n']}, events={r['n_events'] or 'not applicable'}; {r['interval_type']}. "
          f"Interval keys: `{r['key_lower']}` / `{r['key_upper']}`.\n"
        )
    lines.append("")
  return "\n".join(lines)


def write_ledger(rows: Sequence[dict[str, str]], md_path: Path, csv_path: Path) -> None:
  """Write the markdown and CSV ledgers."""
  md_path.write_text(render_markdown(rows), encoding="utf-8")
  with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=LEDGER_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


def ledger_drift(rows: Sequence[dict[str, str]], committed_csv: Path) -> list[str]:
  """IDs whose values or provenance/denominator metadata drift, or are new/removed."""
  if not committed_csv.exists():
    return ["<no committed ledger>"]
  with committed_csv.open(newline="", encoding="utf-8") as f:
    committed = {r["id"]: tuple(r.get(c, "") for c in LEDGER_COLUMNS) for r in csv.DictReader(f)}
  current = {r["id"]: tuple(r.get(c, "") for c in LEDGER_COLUMNS) for r in rows}
  return sorted(i for i in committed.keys() | current.keys() if committed.get(i) != current.get(i))
