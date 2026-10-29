"""Submission-facing manuscript exhibits (v2.18 five-figure lineup).

Round-5 audit (20261021, round05): renders the five clinician-facing main figures
as one styled batch from committed outputs only — no fitting, no resampling, no
cohort change, no overwrite of frozen upstream evidence or the legacy
``manuscript_figures`` outputs (historical files stay for provenance).

Round-6 audit (20261021, round06): every canvas is drawn at its actual
   target publication width (``PUB_WIDTH`` inches, double column) so that the
rendered point sizes are the printed point sizes — no post-hoc fitting shrink.
Long methodological paragraphs moved out of the artwork into the captions in
``docs/manuscript/texts/ehjdh-exhibits.md``. Figure 1 separates the prespecified
selective-deployment (cascade) question from the later post hoc
probability-routing/branch policies. Figure 5 panels read in natural order
a, b / c, d.

Round-7 audit (20261021, round07): Figure 1's three separate evaluation
populations are source-labeled cells with their own evidence-status annotation;
thin grey connectors show *data source only* and the independent Brazilian
CODE-15% cohort receives no connector from the MIMIC flow (P0). Figure 5's
entirely post hoc label-efficiency panels b/c draw open markers and state the
status in their titles, so the filled=prespecified contract is reserved for the
status-coded contrast panels where it is deliberately used (Figure 2c, Figure
5d). Every explicit font override is >= ``MIN_FONT_PT`` (6.5) — enforced by
``tests/test_release_contract.py`` — and footer text is kept inside the
   publication-width canvas so the saved physical width matches ``PUB_WIDTH``
and the printed point sizes are the nominal ones.

Lineup: Figure 1 cohort/evidence flow, Figure 2 discrimination and
comparator attenuation, Figure 3 fixed-capacity prioritization (5% and 10%),
Figure 4 subgroup heterogeneity, Figure 5 pretraining and label efficiency.
A single style module (``apply_style`` plus the colour/semantic constants)
enforces one typography, panel-letter, marker and colour convention across all
   panels: prespecified = filled marker / solid border, post hoc = hollow marker /
dashed border (never colour alone) in the status-coded panels; seed/subset
ranges = thick bands that are explicitly *not* confidence intervals.

Round-8 audit (20261021, round08, author-authorized visual-economy pass §2A):
in-figure text is reduced and key numerical interpretation lives in Results and
the tables, not on the canvas. Figure 1 keeps only intrinsic cohort-flow counts
plus succinct category labels in the analysis lanes and a one-line
source/status legend. Figure 2c drops its per-point ΔAUROC annotations and
per-row denominators (caption carries populations/status). Figure 3 drops the
Δ/CI strings above the bars, simplifies the PPV/sensitivity companion to two
aligned rows, and moves the 26-of-136 tertile note to Results/S7. Figure 4
replaces its prose footer with a compact marker key. Figure 5 shortens the
schematic and drops the narrative footer. No estimand, sample, metric, status
or lineup change.

Round-9 audit (20261021, round09, reader-facing editorial pass — no refitting,
no estimate change): reader-facing display labels adopt the canonical LaTeX
information-set notation with upright descriptive subscripts —
``A_morph``/``A_cart``/``A_report``/``B_AI``/``C_basic``/``C_acute``/``
C_broad`` rendered as matplotlib mathtext raw literals (e.g.
``r"$C_{{\\mathrm{{basic}}}}+B_{{\\mathrm{{AI}}}}"``). Internal result IDs, table CSV
contrast strings, ledger keys and model identifiers are unchanged provenance
aliases. Figure 2c row labels become short mathtext contrasts; the 202-column
feature total moves to the caption/model key (round-09 §1). The CODE-15%
population cell is reduced to source/population/horizon/count; its
independence and update-status qualifiers live in the caption (round-09 §2).
Every ``_box`` call now registers its rectangle and text handles, and each
figure runs ``_audit_figure_layout`` after drawing: text extents must fit
inside their own rectangles with padding, and text–text overlaps outside the
whitelist fail the render; ``run_submission_figures`` writes a one-page
collision report for all five figures (round-09 §2). Fonts are never shrunk
below ``MIN_FONT_PT`` to fix containment — labels are shortened instead.

Round-10 audit (20261021, round10, verified-artwork corrective pass — no
refitting, no estimate change): the round-9 box/whitelist QA reported zero
violations while three reader-visible conflicts survived, because unboxed
transition annotations, tick labels and legend-vs-data geometry were outside
its measured classes. (1) Figure 1's cohort-flow transition labels are now
short, registered notes with a hard right-edge bound inside the cohort column
(``_transition_note``), so they can neither cross into the analysis panel nor
run under the grey source connectors; the ascertainment-order qualifier moved
to the caption. (2) Figure 2 panels a/b replace six overlapping rotated math
tick labels with the audit's shared numbered model key (figure-level line above
the panels, canonical symbols defined there and in the caption), and the
top-of-axes group headers are removed — the separator plus key carry the
grouping. (3) Figure 5 panels b/c replace their two data-obscuring legends
with one shared model-family legend placed outside the plotted data regions;
panel d keeps its distinct status/provenance key. The QA itself is tightened
(round-10 P1): the left-title artist bug in ``_text_overlaps`` is fixed
(``ax.title`` is the center-title artist and measured nothing), tick–tick and
tick→adjacent-panel collisions and legend-vs-plotted-data overlaps are checked
in separate report classes, and the collision report distinguishes measured
classes from an explicit not-evaluated list instead of claiming an overall
clean-figure certificate. Final page-level DOCX/PDF inspection remains a
required human gate.

Round-11 audit (20261021, round11, figure-semantics and caption-economy pass — no
refitting, no estimate change): (1) Figure 1's two grey connectors from cohort
stages to the prespecified/post hoc boxes are removed — those boxes are now an
unconnected analysis-status key under an "Analyses reported below" header,
because analysis status is not a population partition; the single remaining
grey connector marks actual derivation (locked test → MIMIC-restricted refit
population) and CODE-15% stays unlinked. The "paired-evaluable surface" jargon
is replaced with "patients with both ECG feature sets available". (2) Figure 2c
regroups the seven unchanged estimates by the dimension of added information —
$C_{\\mathrm{basic}}$-adjusted ECG comparator breadth (25,430/924), restricted
care-context refits (25,181/915), and parallel ECG-only contrasts — separated
by small in-panel group captions. (3) Figure 2a/b's long model-key line shrinks
to a compact cue over panels a/b; the exact 1–6 mapping lives in the caption;
the Figure 2 whisker wording distinguishes ordinary patient bootstrapping
(panels a/b) from paired outcome-stratified resampling (panel c). Figure 1/2/5
captions are trimmed toward the editorial ~120–180-word target while keeping
populations, status and uncertainty semantics. (4) The legend-vs-data QA class
is renamed "legend over plotted points" with an explicit point-coordinate
scope note.
"""

from __future__ import annotations
   

import json
import math
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "ecg-rep"
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

#: Actual target publication width (inches) for the wide exhibits — the
#: double-column text block of the target journal (~180 mm). Rendering at this
#: size makes the plotted point sizes the printed point sizes (round-6 audit P0).
PUB_WIDTH = 7.05

#: Minimum explicit font size (points) anywhere in the artwork (round-7 audit
#: P1). The renderer keeps footer prose inside the publication-width canvas so
#: the saved physical width stays at ``PUB_WIDTH`` and these printed point
#: sizes are the displayed ones. Enforced by ``tests/test_release_contract.py``.
MIN_FONT_PT = 6.5

#: Required clearance between a box's text extents and its own rectangle
#: border, in points (round-9 audit §2; converted to pixels with the figure
#: dpi so the check is dpi-independent). Text merely staying inside the page
#: is not enough: each ``_box`` text must sit inside its rectangle with this
#: internal padding, measured on the rendered canvas.
BOX_PAD_PT = 1.0

#: Minimum intersection width *and* height for two text artists to count as
#: an overlap, in points (round-9 audit §2; dpi-independent). Smaller touches
#: are sub-pixel rendering artefacts, not collisions.
TEXT_OVERLAP_PT = 0.75

#: Justified text–text overlaps, whitelisted for human review (round-9 audit
#: §2). Each entry is ``(figure_stem, first_text, second_text)`` with the
#: rationale recorded here; anything else fails the render.
TEXT_OVERLAP_WHITELIST: frozenset[tuple[str, str, str]] = frozenset()

#: Canonical reader-facing information-set symbols (round-9 audit §1).
#: Matplotlib mathtext raw literals with upright descriptive subscripts; the
#: internal result IDs and CSV contrast strings remain unchanged aliases.
SYM_MORPH = r"$A_{\mathrm{morph}}$"
SYM_CART = r"$A_{\mathrm{cart}}$"
SYM_REPORT = r"$A_{\mathrm{report}}$"
SYM_AI = r"$B_{\mathrm{AI}}$"
SYM_BASIC = r"$C_{\mathrm{basic}}$"
SYM_ACUTE = r"$C_{\mathrm{acute}}$"
SYM_BROAD = r"$C_{\mathrm{broad}}$"

#: Combined contrast/comparison labels: one mathtext group per expression so
#: operators render inside the math (audit §1 example ``$C_{\mathrm{basic}}+
#: B_{\mathrm{AI}}$``), not as text between two math groups.
CON_AI_MORPH = r"$B_{\mathrm{AI}}-A_{\mathrm{morph}}$"
CON_AI_CART = r"$B_{\mathrm{AI}}-A_{\mathrm{cart}}$"
CON_AI_REPORT = r"$B_{\mathrm{AI}}-A_{\mathrm{report}}$"
SUM_BASIC_MORPH = r"$C_{\mathrm{basic}}+A_{\mathrm{morph}}$"
SUM_BASIC_AI = r"$C_{\mathrm{basic}}+B_{\mathrm{AI}}$"
SUM_BASIC_MORPH_AI = r"$C_{\mathrm{basic}}+A_{\mathrm{morph}}+B_{\mathrm{AI}}$"

#: Box and text artist handles registered by ``_box`` for the layout audit,
#: keyed by figure id (round-9 audit §2).
_BOX_REGISTRY: dict[int, list[tuple[Any, Any]]] = {}

#: Per-figure QA results from the most recent ``run_submission_figures`` render
#: (round-9 audit §2); consumed for the one-page collision report.
_QA_REPORTS: list[dict[str, Any]] = []

#: Unboxed transition annotations registered by ``_transition_note`` (round-10
#: audit P1/R10-01), keyed by figure id: ``(axes, text_artist, max_right)``.
#: ``max_right`` is the right edge — in the axes' *data* coordinates — that the
#: rendered text extent may not cross, so prose between cohort boxes cannot run
#: into the analysis panel or under the grey source connectors again.
_TRANSITION_NOTES: dict[int, list[tuple[Any, Any, float]]] = {}

#: Geometry classes explicitly outside the rendered-geometry QA (round-10 audit
#: P1). The collision report names these rather than implying a mechanical
#: clean-figure certificate; they stay at the documented manual visual gate.
QA_NOT_EVALUATED: tuple[str, ...] = (
    "grey connector routing against text and boxes (manual visual gate)",
    "colour contrast and aesthetic quality (manual visual gate)",
    "final DOCX/PDF page composition and caption adjacency (manual page gate)",
)

# One house style for every submission-facing display (audit §7). Sizes are
# absolute points at publication width; nothing below MIN_FONT_PT appears in artwork.
STYLE: dict[str, Any] = {
  "font.family": "sans-serif",
  "font.size": 7.5,
  "axes.titlesize": 8.0,
  "axes.labelsize": 7.5,
  "axes.linewidth": 0.6,
  "xtick.labelsize": 7.0,
  "ytick.labelsize": 7.0,
  "legend.fontsize": 6.5,
  "lines.linewidth": 1.1,
  "figure.dpi": 300,
  "savefig.dpi": 300,
}

#: Neutral grey-blue: CIIS-derived morphology measurements (A).
COLOR_MORPH = "#6f7d8c"
#: Fixed dark blue: D-BETA-derived features (B).
COLOR_DBETA = "#1a4e8a"
#: Restrained green: models carrying basic clinical information (C0 context).
COLOR_CONTEXT = "#3d7a4f"
#: Contrasting orange: random-encoder control.
COLOR_RANDOM = "#c2661d"
COLOR_HIGHLIGHT_EDGE = "#8a1c1c"
#: Thin grey used for source connectors — a derivation relationship, never an
#: analysis-status signal (round-7 audit P0).
SOURCE_CONNECTOR = "#8a8a8a"

#: Separate evaluation populations (round-7 audit P0). Each cell carries its
#: own data-source and evidence-status label; ``source_arrow`` names the
#: left-flow element a thin grey source connector may be drawn from, or
#: ``None`` for a population that must never receive a connector. CODE-15% is
#: an independent Brazilian cohort (one-year endpoint) — no MIMIC connector.
EVALUATION_POPULATIONS: tuple[dict[str, Any], ...] = (
  {
    "key": "cacute_cmax",
    # Round-9 §1: reader-facing cell labels use the canonical information-set
    # symbols; the internal Cacute/Cmax aliases remain in provenance and code.
    "text": f"{SYM_ACUTE} / {SYM_BROAD} — MIMIC restricted, refitted\n25,181 patients / 915 deaths · post hoc",
    "source_arrow": "locked_test",
    "cohort": "MIMIC",
  },
  {
    "key": "ami",
    "text": "Discharge-coded AMI — MIMIC endpoint\n13,175 encounters / 475 events · exploratory",
    "source_arrow": None,
    "cohort": "MIMIC",
  },
  {
    # Round-9 §2: the CODE cell is minimized to source/population/horizon/count;
    # its pretraining independence and update-status qualifiers live in the
    # figure caption, not on the canvas.
    "key": "code15",
    "text": "Brazilian CODE-15% cohort\n44,768 patients · 577 one-year deaths",
    "source_arrow": None,
    "cohort": "CODE-15% (independent Brazilian cohort)",
  },
)


def apply_style() -> None:
  matplotlib.rcParams.update(STYLE)  # type: ignore[reportCallIssue,reportArgumentType]  # matplotlib stubs: RcKeyType is private; str keys valid at runtime


def _save(fig: Any, figures_dir: Path, stem: str) -> list[str]:
  paths: list[str] = []
  for ext in ("png", "svg"):
    path = figures_dir / f"{stem}.{ext}"
    fig.savefig(path, dpi=300 if ext == "png" else None, bbox_inches="tight", pad_inches=0.02, metadata={"Date": None} if ext == "svg" else None)
    paths.append(str(path))
  plt.close(fig)
  return paths


def _panel_letter(ax: Any, letter: str) -> None:
  ax.set_title(f"{letter}", loc="left", fontsize=8.5, fontweight="bold")


def _load(path: Path) -> Any:
  return json.loads(path.read_text(encoding="utf-8"))


def _box(
  ax: Any,
  x: float,
  y: float,
  w: float,
  h: float,
  text: str,
  face: str = "#f4f4f4",
  edge: str = "black",
  dashed: bool = False,
  lw: float = 1.0,
  fontsize: float = 6.8,
  weight: str = "normal",
) -> tuple[Any, Any]:
  """Draw a labelled rectangle and register both handles for the layout audit.

  Round-9 audit §2: ``Rectangle`` + centred ``Text`` with no wrap or fit
  measurement can silently overflow a small box while page-level checks still
  pass, so every call registers its rectangle and text for
  ``_audit_figure_layout`` to measure after drawing.
  """
  rect = Rectangle((x, y), w, h, facecolor=face, edgecolor=edge, lw=lw, ls="--" if dashed else "-")
  ax.add_patch(rect)
  artist = ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, weight=weight, linespacing=1.3)
  _BOX_REGISTRY.setdefault(id(artist.figure), []).append((rect, artist))
  return rect, artist


def _transition_note(ax: Any, x: float, y: float, text: str, max_right: float) -> Any:
  """Draw an unboxed cohort-flow transition label bounded to its column.

  Round-10 audit R10-01: prose transition labels between Figure 1's cohort
  boxes previously had no containment at all — their rendered extents ran
  across the left column's right boundary into the analysis panel and under
  the grey source connectors. Every transition note is registered with a hard
  right-edge bound (``max_right``, in the axes' data coordinates) that
  ``_audit_figure_layout`` measures on the rendered canvas.
  """
  artist = ax.text(x, y, text, ha="left", va="center", fontsize=6.5, color="#333333")
  _TRANSITION_NOTES.setdefault(id(artist.figure), []).append((ax, artist, max_right))
  return artist


def _artist_display_points(artist: Any) -> list[tuple[float, float]]:
  """Return an artist's data points in display (pixel) coordinates.

  Used by the round-10 legend-vs-data check: ``Line2D`` contributes
  ``get_xydata()``, scatter/collection artists contribute ``get_offsets()``.
  Failures degrade to an empty list (the collision report then under-reports
  rather than fabricating geometry).
  """
  try:
    if hasattr(artist, "get_xydata"):
      points = artist.get_xydata()
    else:
      points = artist.get_offsets()
    if points is None or len(points) == 0:
      return []
    transformed = artist.get_transform().transform(np.asarray(points))
    return [(float(px), float(py)) for px, py in transformed]
  except Exception:  # noqa: BLE001 — QA probe must never break a render
    return []


def _text_overlaps(
  ax: Any,
  renderer: Any,
  threshold: float,
  extra_artists: list[tuple[str, Any]] | None = None,
) -> list[tuple[str, str, float, float]]:
  """Report suspicious pairwise intersections among an axes' text artists.

  Tick labels are excluded here: they are checked pairwise by
  ``_tick_overlaps`` and against neighbouring panels by ``_tick_crossings``
  (round-10 audit P1) as separate report classes.
  """
  artists: list[tuple[str, Any]] = [(t.get_text(), t) for t in ax.texts]
  # Round-10 audit P1 bug fix: ``_panel_letter`` writes a *left* title, which
  # matplotlib stores separately from ``ax.title`` (the center-title artist).
  # The previous code read ``ax.get_title(loc="left")`` for the string but
  # measured ``ax.title`` — an empty center artist — so left titles were never
  # actually measured against other text.
  for title_artist in (getattr(ax, "_left_title", None), getattr(ax, "_right_title", None), ax.title):
    if title_artist is not None and title_artist.get_text():
      artists.append((title_artist.get_text(), title_artist))
  for label in (ax.xaxis.label, ax.yaxis.label):
    if label.get_text():
      artists.append((label.get_text(), label))
  legend = ax.get_legend()
  if legend is not None:
    for text in legend.get_texts():
      artists.append((text.get_text(), text))
  if extra_artists:
    artists.extend(extra_artists)
  overlaps: list[tuple[str, str, float, float]] = []
  for i in range(len(artists)):
    for j in range(i + 1, len(artists)):
      (t1, a1), (t2, a2) = artists[i], artists[j]
      b1, b2 = a1.get_window_extent(renderer), a2.get_window_extent(renderer)
      ix = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
      iy = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
      if ix > threshold and iy > threshold:
        overlaps.append((t1, t2, ix, iy))
  return overlaps


def _tick_overlaps(ax: Any, renderer: Any, threshold: float) -> list[dict[str, Any]]:
  """Report pairwise tick-label collisions within one axes (round-10 audit P1).

  Separate report class from panel text: tick labels are compared only against
  other tick labels of the same axis. High confidence by construction — a
  finding requires intersection in *both* dimensions, so adjacent category
  labels that share a baseline or gridline row but do not actually overlap
  never fire.
  """
  findings: list[dict[str, Any]] = []
  for axis_name in ("xaxis", "yaxis"):
    labels = [label for label in getattr(ax, axis_name).get_ticklabels() if label.get_text()]
    boxes = [label.get_window_extent(renderer) for label in labels]
    for i in range(len(boxes)):
      for j in range(i + 1, len(boxes)):
        b1, b2 = boxes[i], boxes[j]
        ix = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
        iy = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
        if ix > threshold and iy > threshold:
          findings.append({
            "axis": axis_name[0],
            "first": labels[i].get_text(),
            "second": labels[j].get_text(),
            "overlap_px": [round(ix, 1), round(iy, 1)],
          })
  return findings


def _tick_crossings(fig: Any, renderer: Any, threshold: float) -> list[dict[str, Any]]:
  """Report tick labels that reach into a *neighbouring* panel (round-10 P1).

  Figure-level check: a tick label whose extent intersects another axes' area
  by more than ``threshold`` in both dimensions is flagged. This catches the
  narrow-panel crowd-out pattern that survived the round-9 text-only check
  (Figure 2a/b rotated labels ran under panel c's data region).
  """
  panels = [(ax, ax.get_window_extent(renderer)) for ax in fig.axes]
  findings: list[dict[str, Any]] = []
  for ax, _ in panels:
    for axis_name in ("xaxis", "yaxis"):
      for label in getattr(ax, axis_name).get_ticklabels():
        text = label.get_text()
        if not text:
          continue
        b = label.get_window_extent(renderer)
        for other, other_box in panels:
          if other is ax:
            continue
          ix = min(b.x1, other_box.x1) - max(b.x0, other_box.x0)
          iy = min(b.y1, other_box.y1) - max(b.y0, other_box.y0)
          if ix > threshold and iy > threshold:
            findings.append({
              "label": text,
              "axis": axis_name[0],
              "into_panel": fig.axes.index(other) + 1,
              "overlap_px": [round(ix, 1), round(iy, 1)],
            })
  return findings


def _legend_data_overlaps(fig: Any, renderer: Any, threshold: float) -> list[dict[str, Any]]:
  """Report legends whose box sits over plotted data points (round-10 audit P1).

  Checks every axes legend and every figure-level legend against the display
  coordinates of all plotted points (lines via ``get_xydata``, collections via
  ``get_offsets``). The legend box is padded by ``threshold`` so a legend that
  merely grazes data still fires — the Figure 5 b/c pattern this check exists
  for hid curve segments under an in-axes legend. Scope note (round-11 R11-04):
  this inspects *point* coordinates only — it is a legend-over-plotted-points
  check, not a geometric test of full line segments, interval bars, connectors
  or final-page composition, which remain manual visual gates.
  """
  legends: list[tuple[str, Any]] = [
    (f"axes {fig.axes.index(ax) + 1}", ax.get_legend()) for ax in fig.axes if ax.get_legend() is not None
  ]
  for fig_legend in getattr(fig, "legends", []):
    legends.append(("figure", fig_legend))
  findings: list[dict[str, Any]] = []
  for owner, legend in legends:
    lb = legend.get_window_extent(renderer)
    padded = (
      lb.x0 - threshold,
      lb.y0 - threshold,
      lb.x1 + threshold,
      lb.y1 + threshold,
    )
    for ax in fig.axes:
      for artist in [*ax.lines, *ax.collections]:
        hidden = [
          (px, py) for px, py in _artist_display_points(artist)
          if padded[0] <= px <= padded[2] and padded[1] <= py <= padded[3]
        ]
        if hidden:
          findings.append({
            "legend": owner,
            "hidden_points": len(hidden),
            "sample_point_px": [round(hidden[0][0], 1), round(hidden[0][1], 1)],
          })
  return findings


def _transition_note_violations(fig: Any, renderer: Any, threshold: float) -> list[dict[str, Any]]:
  """Measure registered transition notes against their bounds (round-10 P1).

  Two checks per note: the rendered text extent's right edge must stay inside
  the note's ``max_right`` bound (data coordinates), and the extent must not
  intersect any registered ``_box`` rectangle.
  """
  findings: list[dict[str, Any]] = []
  entries = _TRANSITION_NOTES.get(id(fig), [])
  box_rects = [rect for rect, _ in _BOX_REGISTRY.get(id(fig), [])]
  for ax, artist, max_right in entries:
    extent = artist.get_window_extent(renderer)
    left_data, right_data = ax.transData.inverted().transform([(extent.x0, extent.y0), (extent.x1, extent.y1)])
    if right_data[0] > max_right:
      findings.append({
        "text": artist.get_text(),
        "right_edge": round(right_data[0], 4),
        "max_right": max_right,
      })
    artist_box = (extent.x0, extent.y0, extent.x1, extent.y1)
    for rect in box_rects:
      rb = rect.get_window_extent(renderer)
      ix = min(artist_box[2], rb.x1) - max(artist_box[0], rb.x0)
      iy = min(artist_box[3], rb.y1) - max(artist_box[1], rb.y0)
      if ix > threshold and iy > threshold:
        findings.append({
          "text": artist.get_text(),
          "intersects_box": True,
          "overlap_px": [round(ix, 1), round(iy, 1)],
        })
  return findings


def _audit_figure_layout(fig: Any, stem: str) -> dict[str, Any]:
  """Measure rendered text containment and multi-class collisions on one figure.

  Round-9 audit §2 introduced box containment and panel-text overlaps; the
  round-10 audit (P1) splits and extends the measured classes — panel text
  (with the left-title artist actually measured), tick–tick collisions,
  tick-label crossings into neighbouring panels, legend-vs-plotted-data
  overlaps, and registered transition notes against their right-edge bounds
  and registered boxes. Raises on any violation; never shrinks fonts.
  Remaining manual classes (connector routing, colour contrast, final page
  composition) are listed in ``QA_NOT_EVALUATED`` rather than silently skipped.
  """
  fig.canvas.draw()
  renderer = fig.canvas.get_renderer()
  pad = BOX_PAD_PT * fig.dpi / 72.0
  entries = _BOX_REGISTRY.pop(id(fig), [])
  box_violations: list[dict[str, Any]] = []
  for rect, text in entries:
    t_box = text.get_window_extent(renderer)
    r_box = rect.get_window_extent(renderer)
    if (t_box.x0 < r_box.x0 + pad or t_box.x1 > r_box.x1 - pad
        or t_box.y0 < r_box.y0 + pad or t_box.y1 > r_box.y1 - pad):
      box_violations.append({
        "text": text.get_text(),
        "text_box": [round(t_box.x0, 1), round(t_box.y0, 1), round(t_box.x1, 1), round(t_box.y1, 1)],
        "rect_box": [round(r_box.x0, 1), round(r_box.y0, 1), round(r_box.x1, 1), round(r_box.y1, 1)],
      })
  threshold = TEXT_OVERLAP_PT * fig.dpi / 72.0
  panel_text_overlaps: list[dict[str, Any]] = []
  for ax_index, ax in enumerate(fig.axes):
    # Figure-level texts (e.g. Figure 2's shared model key) are paired with the
    # first axes' panel text; pairing them with every axes would duplicate
    # report rows for the same geometry.
    extras = [(t.get_text(), t) for t in fig.texts] if ax_index == 0 else []
    for t1, t2, ix, iy in _text_overlaps(ax, renderer, threshold, extra_artists=extras):
      if (stem, t1, t2) in TEXT_OVERLAP_WHITELIST or (stem, t2, t1) in TEXT_OVERLAP_WHITELIST:
        continue
      panel_text_overlaps.append({"first": t1, "second": t2, "overlap_px": [round(ix, 1), round(iy, 1)]})
  tick_overlaps = [f for ax in fig.axes for f in _tick_overlaps(ax, renderer, threshold)]
  tick_crossings = _tick_crossings(fig, renderer, threshold)
  legend_overlaps = _legend_data_overlaps(fig, renderer, threshold)
  transition_violations = _transition_note_violations(fig, renderer, threshold)
  violations = [
    ("box overflow", box_violations),
    ("panel text overlap", panel_text_overlaps),
    ("tick overlap", tick_overlaps),
    ("tick crossing into neighbouring panel", tick_crossings),
    ("legend over plotted points", legend_overlaps),
    ("transition note out of bounds", transition_violations),
  ]
  failed = {name: found for name, found in violations if found}
  if failed:
    raise ValueError(f"Figure layout QA failed for {stem}: {failed}")
  report = {
    "stem": stem,
    "boxes_checked": len(entries),
    "box_violations": 0,
    "panel_text_overlaps": 0,
    "tick_overlaps": 0,
    "tick_crossings": 0,
    "legend_data_overlaps": 0,
    "transition_notes_checked": len(_TRANSITION_NOTES.get(id(fig), [])),
    "transition_note_violations": 0,
    "figure_texts_checked": len(getattr(fig, "texts", [])),
  }
  _QA_REPORTS.append(report)
  return report


def collision_report_markdown(reports: list[dict[str, Any]]) -> str:
  """Round-10 per-class collision report across the five submission figures.

  Every measured class is reported separately; classes the rendered-geometry
  QA cannot certify are listed as explicitly not evaluated (manual visual
  gate) instead of implying an overall clean-figure certificate.
  """
  lines = [
    "# Round-10 figure layout QA (rendered-geometry collision report)",
    "",
    "Measured on the rendered canvas after `fig.canvas.draw()`, in separate",
    "classes: box containment (every `_box` text fits its rectangle with",
    f"{BOX_PAD_PT}-pt internal padding), panel-text overlaps (titles, annotations,",
    f"axis labels, legend entries, figure-level texts; >{TEXT_OVERLAP_PT} pt in both",
    "dimensions), pairwise tick-label overlaps per axis, tick labels crossing",
    "into a neighbouring panel, legends covering plotted line/collection",
    "points, and registered transition notes against their right-edge bounds",
    "and registered boxes. Fonts are never shrunk below the",
    f"{MIN_FONT_PT} pt floor; containment is fixed by shorter labels or larger",
    "containers. A clean row certifies only the measured classes above.",
    "",
    "| Figure | Boxes checked | Box overflows | Panel-text overlaps | Tick overlaps | Tick-panel crossings | Legend over plotted points | Transition notes (checked / violations) |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
  ]
  for report in reports:
    lines.append(
      f"| {report['stem']} | {report['boxes_checked']} | {report['box_violations']} | {report['panel_text_overlaps']} "
      f"| {report['tick_overlaps']} | {report['tick_crossings']} | {report['legend_data_overlaps']} "
      f"| {report['transition_notes_checked']} / {report['transition_note_violations']} |"
    )
  lines += ["", "Not evaluated by this report (documented manual visual gate):"]
  lines += [f"- {item}" for item in QA_NOT_EVALUATED]
  lines.append("")
  return "\n".join(lines)


def _arrow(ax: Any, x0: float, y0: float, x1: float, y1: float, dashed: bool = False, color: str = "black", lw: float = 0.9, connectionstyle: str | None = None) -> None:
  options: dict[str, Any] = {"connectionstyle": connectionstyle} if connectionstyle else {}
  ax.add_patch(
    FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=lw, ls="--" if dashed else "-", color=color,
                    **options)
  )


#: Aggregate cohort denominators consumed by Figure 1. The public replication
#: package ships this small JSON instead of the patient-level cohort parquet;
#: the private analysis tree keeps using its frozen sources below.
FIGURE1_COUNT_KEYS = (
  "ecgs", "patients_initial", "eligible_adults", "ascertainment_excluded",
  "cohort", "dev", "val", "test", "dev_analytic", "val_analytic",
  "test_analytic", "test_events",
)


def _counts_for_figure1(tables_dir: Path, artifacts_root: Path) -> dict[str, int]:
  """Cohort denominators from frozen sources.

  Initially-identified, eligible-adult, ascertainment and cohort counts come
  from the frozen cohort flow report; split sizes from the frozen cohort
  parquet (as in the legacy renderer); paired-evaluable surface sizes per split
  from the LM1 surface artifact (``n_surface_by_split``), the machine-readable
  source of the 76,590 / 25,526 / 25,430 denominators quoted in Supplement S1.

  When ``artifacts/cohort/cohort_counts.json`` exists it is the authoritative
  aggregate source. The public replication package uses that route because it
  must not receive the patient-level cohort parquet or the private LM1 tree.
  """
  counts_path = artifacts_root / "cohort" / "cohort_counts.json"
  if counts_path.exists():
    aggregate = _load(counts_path)
    missing = [key for key in FIGURE1_COUNT_KEYS if key not in aggregate]
    if missing:
      raise KeyError(f"{counts_path} lacks required counts: {', '.join(missing)}")
    return {key: int(aggregate[key]) for key in FIGURE1_COUNT_KEYS}

  text = (artifacts_root / "cohort" / "cohort_flow.md").read_text(encoding="utf-8")

  def cell(step: int, column: int) -> int:
    row = next(line for line in text.splitlines() if line.startswith(f"| **{step}** |"))
    return int(row.split("|")[column].strip().replace(",", ""))

  cohort = pl.read_parquet(artifacts_root / "cohort" / "primary_cohort.parquet")
  split_counts = cohort.group_by("split").len()
  splits = dict(zip(split_counts["split"].to_list(), split_counts["len"].to_list()))
  lm1 = _load(artifacts_root / "limitation_mitigation" / "lm1" / "cart_report_models.json")
  surface = lm1["n_surface_by_split"]
  return {
    "ecgs": cell(1, 3),
    "patients_initial": cell(1, 4),
    "eligible_adults": cell(4, 4),
    "ascertainment_excluded": cell(6, 6),
    "cohort": cell(6, 4),
    "dev": int(splits.get("dev", 0)),
    "val": int(splits.get("val", 0)),
    "test": int(splits.get("test", 0)),
    "dev_analytic": int(surface["dev"]),
    "val_analytic": int(surface["val"]),
    "test_analytic": int(surface["test"]),
    "test_events": 924,
  }


def figure1_cohort_evidence_flow(figures_dir: Path, counts: dict[str, int]) -> list[str]:
  """Submission Figure 1: cohort/evidence flow with paired-evaluable denominators in all three splits.

  Round-5 audit §2: two distinct exclusion stages (outcome ascertainment before
  splitting; representation availability within each split), prespecified and
  post hoc analysis lanes with a visible legend, and a visually separate lane
  for the three non-attrition evaluation populations.
  Round-6 audit: the prespecified lane names the original prespecified
  selective-deployment (cascade) question (RQ6) while the post hoc lane keeps
  only the later probability-routing/branch policies and decision-curve
  implementations (round-06 P1), and the in-figure note no longer points to a
  supplementary analytic-population flow that does not exist (round-06 P1).
  """
  dev_excl = counts["dev"] - counts["dev_analytic"]
  val_excl = counts["val"] - counts["val_analytic"]
  test_excl = counts["test"] - counts["test_analytic"]
  surface_total = counts["dev_analytic"] + counts["val_analytic"] + counts["test_analytic"]

  fig, ax = plt.subplots(figsize=(PUB_WIDTH, 5.9))
  ax.set_axis_off()
  ax.set_xlim(0, 1)
  ax.set_ylim(0, 1)

  # Left column: cohort flow (two exclusion stages made explicit). Inter-box
  # notes are two-line and are registered transition notes with a hard
  # right-edge bound inside the left column (column ends at x = 0.555; the
  # right lane starts at x = 0.585), so they can neither cross into the
  # analysis panel nor run under the grey source connectors (round-10 R10-01).
  _box(ax, 0.01, 0.902, 0.545, 0.088, f"MIMIC-IV + MIMIC-IV-ECG\n{counts['ecgs']:,} ECGs · {counts['patients_initial']:,} patients", face="#e8edf2", weight="bold", fontsize=7.0)
  _arrow(ax, 0.2825, 0.902, 0.2825, 0.852)
  _transition_note(ax, 0.31, 0.872, "Adults · one earliest\neligible index ECG", max_right=0.55)
  _box(ax, 0.01, 0.769, 0.545, 0.078, f"Waveform-eligible index ECGs: {counts['eligible_adults']:,} patients", fontsize=7.0)
  _arrow(ax, 0.2825, 0.769, 0.2825, 0.719)
  # Ascertainment-order qualifier ("ascertainment first") lives in the caption
  # (round-10 R10-01); the in-figure note keeps only the intrinsic count.
  _transition_note(ax, 0.31, 0.739, f"exclude {counts['ascertainment_excluded']:,} without\nobservable 30-day outcome", max_right=0.55)
  _box(ax, 0.01, 0.636, 0.545, 0.078, f"Eligible cohort: {counts['cohort']:,} patients", face="#e8edf2", fontsize=7.0)
  _arrow(ax, 0.2825, 0.636, 0.2825, 0.576)
  _box(ax, 0.01, 0.476, 0.545, 0.10,
       "Patient-disjoint 60/20/20 assignment (seed 42)\n"
       f"Development {counts['dev']:,} · Validation {counts['val']:,}\nLocked test {counts['test']:,}", fontsize=7.0)
  _arrow(ax, 0.2825, 0.476, 0.2825, 0.426)
  _transition_note(ax, 0.31, 0.446, "both ECG representations\n"
                   f"required (excl. {dev_excl}/{val_excl}/{test_excl})", max_right=0.55)
  _box(ax, 0.01, 0.316, 0.545, 0.10,
       f"Patients with both ECG feature sets available: {surface_total:,}\n"
       f"Development {counts['dev_analytic']:,} · Validation {counts['val_analytic']:,} · Test {counts['test_analytic']:,}", fontsize=7.0)
  _arrow(ax, 0.2825, 0.316, 0.2825, 0.261)
  _box(ax, 0.01, 0.131, 0.545, 0.125,
       f"Primary locked test · both ECG feature sets available\n{counts['test_analytic']:,} patients · {counts['test_events']} deaths within 30 days",
       face="#dde6f0", edge=COLOR_HIGHLIGHT_EDGE, lw=1.3, weight="bold", fontsize=7.0)

  # Right column: analysis-status key (round-11 R11-01: unconnected to the
  # cohort flow — prespecified/post hoc encodes the *question/analysis
  # status*, never a population partition or data source; the cohort stages
  # above feed every MIMIC analysis). Exact example lists live in the caption.
  lane_x, lane_w = 0.585, 0.405
  ax.text(lane_x + lane_w / 2, 0.985, "Analyses reported below",
          ha="center", va="top", fontsize=6.8, fontweight="bold")
  _box(ax, lane_x, 0.715, lane_w, 0.23,
       f"Prespecified analyses\n\nPrimary representation comparisons\n({SYM_MORPH} vs {SYM_AI}, without and with {SYM_BASIC})\nTraining controls",
       face="#eef3ee", lw=1.1, fontsize=6.8)
  _box(ax, lane_x, 0.52, lane_w, 0.18,
       "Post hoc extensions\n\nRicher comparators · context refits ·\nrouting and decision-curve policies",
       face="#ffffff", dashed=True, lw=1.1, fontsize=6.8)

  # Round-7 audit (P0); round-11 R11-01: the single remaining grey connector
  # marks actual population derivation only — the MIMIC-restricted refit of
  # the locked test — and the independent Brazilian CODE-15% cohort receives
  # no connector from the MIMIC flow (independent cohort, different one-year
  # endpoint/horizon).
  pop_x, pop_w = lane_x, lane_w
  ax.text(pop_x + pop_w / 2, 0.472, "Separate evaluation populations\n(not serial attrition)",
          ha="center", va="top", fontsize=6.8, fontweight="bold")
  pop_geo = {"cacute_cmax": (0.345, 0.075), "ami": (0.245, 0.07), "code15": (0.115, 0.07)}
  for pop in EVALUATION_POPULATIONS:
    y0, h = pop_geo[pop["key"]]
    _box(ax, pop_x, y0, pop_w, h, pop["text"], face="#fbfbfb", edge="#777777", lw=0.9, fontsize=6.5)
    if pop["source_arrow"] == "locked_test":
      # MIMIC-restricted refit of the paired-evaluable locked test.
      _arrow(ax, 0.555, 0.225, pop_x, y0 + h / 2, color=SOURCE_CONNECTOR, lw=0.8,
             connectionstyle="angle,angleA=0,angleB=90")

  # Round-08 P0 (§2A): the footer shrinks to one minimal key; populations,
  # status semantics and Methods pointers live in the caption. Round-11
  # R11-01: only one grey connector remains, so the key states derivation.
  ax.text(
    0.01, 0.095,
    "Vertical arrows: cohort attrition · grey connector: population derived from the locked test · solid = prespecified, dashed = post hoc.",
    ha="left", va="top", fontsize=6.5, linespacing=1.4,
  )
  _audit_figure_layout(fig, "figure1_cohort_evidence_flow")
  return _save(fig, figures_dir, "figure1_cohort_evidence_flow")


def figure2_discrimination_attenuation(figures_dir: Path, tables_dir: Path, clinical_tables_dir: Path) -> list[str]:
  """Submission Figure 2: absolute primary performance plus the comparator-attenuation panel.

  Panels a/b: the six prespecified models' locked-test AUROC and AUPRC
  (Table 2), numbered 1–6 with a compact cue (the full mapping lives in the
  caption). Panel c: paired ΔAUROC grouped by the dimension of added
  information (round-11 R11-02) — the same-population $C_{\\mathrm{basic}}$-adjusted
  ECG comparator breadth, the restricted care-context refits, and the
  parallel ECG-only contrasts — separate comparison conditions, never a
  nested trajectory. Canvas at publication width.
  """
  t2 = pl.read_csv(tables_dir / "table2_overall_performance.csv")
  t3 = pl.read_csv(tables_dir / "table3_incremental_value_and_controls.csv")
  t6 = pl.read_csv(tables_dir / "table6_limitation_mitigation.csv")
  attenuation = pl.read_csv(clinical_tables_dir / "attenuation_comparison.csv")

  fig, (ax_auc, ax_prc, ax_b) = plt.subplots(1, 3, figsize=(PUB_WIDTH, 3.4), gridspec_kw={"width_ratios": [0.85, 0.85, 1.8]})
  order = (("M_A_ECG", COLOR_MORPH), ("M_B_ECG", COLOR_DBETA), ("M_C", COLOR_CONTEXT), ("M_A", COLOR_CONTEXT), ("M_B", COLOR_CONTEXT), ("M_AB", COLOR_CONTEXT))
  # Round-11 R11-03: the full canonical-symbol 1–6 mapping (1 A_morph,
  # 2 B_AI, 3 C_basic, 4 C_basic+A_morph, 5 C_basic+B_AI,
  # 6 C_basic+A_morph+B_AI) lives in the external caption; the artwork
  # carries only numbered ticks and a compact cue.
  model_numbers = tuple(str(i + 1) for i in range(len(order)))
  frozen = {r["model"]: r for r in t2.to_dicts()}
  for ax, metric, ylabel in ((ax_auc, "auroc", "AUROC"), (ax_prc, "auprc", "AUPRC")):
    for i, (label, color) in enumerate(order):
      r = frozen[label]
      ax.errorbar(i, r[metric], yerr=[[r[metric] - r[f"{metric}_ci_lower"]], [r[f"{metric}_ci_upper"] - r[metric]]],
                  fmt="o", color=color, capsize=2, ms=3.8)
    ax.set_xticks(range(len(order)), list(model_numbers), fontsize=6.5)
    ax.set_ylabel(f"{ylabel} (95% CI)")
    ax.set_xlim(-0.5, 5.5)
    ax.axvline(1.5, color="grey", lw=0.5, ls=":")
  fig.text(0.30, 0.975, f"Models:  1–2 ECG-only  ·  3–6 with {SYM_BASIC}", ha="center", va="center", fontsize=6.5)
  _panel_letter(ax_auc, "a  AUROC, six models")
  _panel_letter(ax_prc, "b  AUPRC, six models")

  # Panel c: paired ΔAUROC by dimension of added information (round-11 R11-02).
  # The seven stored estimates are unchanged; rows are grouped so the
  # same-population, C_basic-adjusted ECG comparator breadth (the main
  # attenuation sequence) reads separately from the restricted care-context
  # refits and from the parallel ECG-only contrasts — never one nested
  # trajectory. Small in-panel group captions separate the groups (no
  # background boxes, no second legend, no connecting line between groups).
  by_name = {r["contrast"]: r for r in t3.to_dicts() if r["metric"] == "auroc"}
  cacute = attenuation.filter(pl.col("metric") == "auroc").row(0, named=True)

  def t6_row(result_id: str) -> dict[str, Any]:
    return t6.filter(pl.col("id") == result_id).row(0, named=True)

  # (label, (point, lo, hi), post_hoc, y position) — top to bottom.
  rows: list[tuple[str, tuple[float, float, float], bool, float]] = [
    (f"{CON_AI_MORPH} · with {SYM_BASIC}", (by_name["Context-adjusted: B - A"]["estimate"], by_name["Context-adjusted: B - A"]["interval_lower"], by_name["Context-adjusted: B - A"]["interval_upper"]), False, 8.0),
    (f"{CON_AI_CART} · with {SYM_BASIC}", (by_name["Context-adjusted: B - A'"]["estimate"], by_name["Context-adjusted: B - A'"]["interval_lower"], by_name["Context-adjusted: B - A'"]["interval_upper"]), True, 7.0),
    (f"{CON_AI_REPORT} · with {SYM_BASIC}", tuple(t6_row("L91_c_b_minus_c_a2_auroc")[c] for c in ("estimate", "interval_lower", "interval_upper")), True, 6.0),
    (f"{CON_AI_MORPH} · with {SYM_ACUTE}", (float(cacute["delta_cacute"]), float(cacute["delta_cacute_ci_low"]), float(cacute["delta_cacute_ci_high"])), True, 3.6),
    (f"{CON_AI_MORPH} · with {SYM_BROAD}", tuple(t6_row("L93_Cmax+B_minus_Cmax+A_auroc")[c] for c in ("estimate", "interval_lower", "interval_upper")), True, 2.6),
    (f"{CON_AI_MORPH} · ECG only", (by_name["ECG-only: B - A"]["estimate"], by_name["ECG-only: B - A"]["interval_lower"], by_name["ECG-only: B - A"]["interval_upper"]), False, 0.6),
    (f"{CON_AI_REPORT} · ECG only", tuple(t6_row("L91_b_minus_a2_ecg_auroc")[c] for c in ("estimate", "interval_lower", "interval_upper")), True, -0.4),
  ]
  for label, (point, lo, hi), post_hoc, y in rows:
    ax_b.errorbar(point, y, xerr=[[point - lo], [hi - point]], fmt="o", color=COLOR_DBETA, mfc="white" if post_hoc else COLOR_DBETA, capsize=2, ms=4.0)
  # Round-12 R12-01: the zero reference is drawn only across the rows that
  # contain data, with deliberate gaps at the three in-panel group captions
  # (y ≈ 9.3/4.6/1.6), so the line can no longer run through their text. No
  # result row is truncated; the line is never obscured with boxes.
  for y0, y1 in ((-0.7, 0.9), (2.35, 3.85), (5.75, 8.35)):
    ax_b.plot([0, 0], [y0, y1], color="black", lw=0.7, solid_capstyle="butt")
  ax_b.set_yticks([r[3] for r in rows], [r[0] for r in rows], fontsize=6.5)
  for tick, post_hoc in zip(ax_b.get_yticklabels(), [r[2] for r in rows]):
    tick.set_color("#555555" if post_hoc else "black")
  ax_b.tick_params(axis="y", length=0, pad=2)
  ax_b.text(-0.028, 9.3, f"With {SYM_BASIC}: comparator breadth\n(same 25,430/924 test)",
            ha="left", va="center", fontsize=6.5, color="#555555")
  ax_b.text(-0.028, 4.6, "Care-context refits: clinical information\n(restricted 25,181/915)",
            ha="left", va="center", fontsize=6.5, color="#555555")
  ax_b.text(-0.028, 1.6, "ECG-only contrasts\n(same 25,430/924 test)",
            ha="left", va="center", fontsize=6.5, color="#555555")
  ax_b.set_xlim(-0.03, 0.105)
  ax_b.set_ylim(-1.1, 10.0)
  ax_b.set_xlabel("Paired ΔAUROC (95% CI)")
  _panel_letter(ax_b, f"c  {SYM_AI} − comparator")
  # Round-08 P0 (§2A): no footer prose; populations, status and interval
  # semantics live in the caption. Round-9 §2: rendered-layout QA before save.
  # Round-10 R10-02: the top margin now carries the shared model key line
  # above the panel letters instead of in-axes group headers.
  fig.subplots_adjust(left=0.072, right=0.99, top=0.84, bottom=0.15, wspace=0.88)
  _audit_figure_layout(fig, "figure2_discrimination_attenuation")
  return _save(fig, figures_dir, "figure2_discrimination_attenuation")


def figure3_review_capacity_5_10(figures_dir: Path, tables_dir: Path, artifacts_root: Path) -> list[str]:
  """Submission Figure 3: fixed-capacity prioritization at the 5% and 10% operating points.

  Event capture bars keep only their death counts; panel b is a compact
  paired-difference/CI graphic (the PPV/sensitivity companion lives in
  Table 4C); the 5% C0 identity panel shows the 126/136/95 exchange. Review
  capacity is a ranked share, never a predicted-risk threshold and never an
  AI-routing fraction (audit §4). Round-08 P0 (§2A): Δ/CI strings above the
  bars, the in-figure PPV/sensitivity table, the 26-of-136 tertile overlay
  and the footer prose moved to Results/Table 4/caption.
  """
  analyses = _load(artifacts_root / "analysis" / "scientific_analysis_results.json")["analyses"]
  strata = pl.read_csv(tables_dir / "supp_risk_strata.csv")

  def cap_rows(key: str) -> dict[float, dict[str, Any]]:
    return {round(float(r["capacity_fraction"]), 4): r for r in analyses[key]}

  ecg = cap_rows("rq5_capacity_comparison")
  ctx = cap_rows("rq5_capacity_comparison_context")
  groups: list[tuple[str, dict[str, Any]]] = [
    ("5%, ECG-only", ecg[0.05]),
    ("10%, ECG-only", ecg[0.1]),
    (f"5%, with {SYM_BASIC}", ctx[0.05]),
    (f"10%, with {SYM_BASIC}", ctx[0.1]),
  ]
  all_row = strata.filter(pl.col("stratum") == "all").row(0, named=True)
  shared = int(all_row["capacity_events_captured_by_b"]) - int(all_row["capacity_events_b_only"])
  b_only = int(all_row["capacity_events_b_only"])
  a_only = int(all_row["capacity_events_a_only"])

  fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(PUB_WIDTH, 3.1), gridspec_kw={"width_ratios": [1.25, 0.9, 1.0]})
  xs = np.arange(len(groups))
  width = 0.34
  a_counts = [int(g[1]["capacity_a"]["n_events_captured"]) for g in groups]
  b_counts = [int(g[1]["capacity_b"]["n_events_captured"]) for g in groups]
  ax_a.bar(xs - width / 2, a_counts, width, color=COLOR_MORPH, label=f"Morphology measurements ({SYM_MORPH})")
  ax_a.bar(xs + width / 2, b_counts, width, color=COLOR_DBETA, label=f"AI-derived features ({SYM_AI})")
  # Round-08 P0 (§2A): paired Δ/CI strings above the bars moved to Results
  # §1.3/Table 4; bars keep only their death counts.
  for x, (label, r) in zip(xs, groups):
    ax_a.annotate(f"{a_counts[x]}", (x - width / 2, a_counts[x]), textcoords="offset points", xytext=(0, 2), ha="center", fontsize=6.5)
    # Round-9 §2 QA fix: the white count sits at its bar's base (never near the
    # bar top), so it cannot touch the neighbouring above-bar count whatever
    # the two bars' heights.
    ax_a.annotate(f"{b_counts[x]}", (x + width / 2, 0), textcoords="offset points", xytext=(0, 3), ha="center", va="bottom", fontsize=6.5, color="white")
  ax_a.set_xticks(xs, [g[0].replace("%, ", "%\n") for g in groups], fontsize=6.8)
  ax_a.set_ylabel("Deaths among flagged patients")
  ax_a.set_ylim(top=max(b_counts) * 1.14)
  ax_a.legend(fontsize=6.5, loc="upper left")
  _panel_letter(ax_a, "a  Deaths captured")

  # Panel b: compact paired-difference/CI graphic (round-08 P0 §2A); the
  # PPV/sensitivity companion moved to Table 4C.
  for i, (label, r) in enumerate(groups):
    d, lo, hi = float(r["delta_events_captured"]), float(r["delta_events_ci_lower"]), float(r["delta_events_ci_upper"])
    ax_b.errorbar(d, i, xerr=[[d - lo], [hi - d]], fmt="o", color=COLOR_DBETA, capsize=2, ms=4.0)
  ax_b.axvline(0, color="black", lw=0.7)
  ax_b.set_yticks(range(len(groups)), [g[0].replace("%, ", "%\n") for g in groups], fontsize=6.8)
  ax_b.invert_yaxis()
  ax_b.set_xlim(-12, 122)
  ax_b.set_xlabel(f"Additional deaths identified, {CON_AI_MORPH} (95% CI)")
  _panel_letter(ax_b, f"b  Paired Δ, {CON_AI_MORPH}")

  bars = [shared, b_only, a_only]
  labels_c = ["Both lists", "B only", "A only"]
  ax_c.bar([0, 1, 2], bars, width=0.62, color=["#9aa5ad", COLOR_DBETA, COLOR_MORPH])
  for x, v in zip([0, 1, 2], bars):
    ax_c.annotate(str(v), (x, v), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7.0)
  # Round-08 P0 (§2A): the 26-of-136 tertile annotation lives in Results §1.3
  # and Supplementary Table S7, not overlaid on the bars.
  ax_c.set_xticks([0, 1, 2], labels_c, fontsize=6.5)
  ax_c.set_ylim(top=max(bars) * 1.18)
  ax_c.set_ylabel(f"Deaths (5% lists, with {SYM_BASIC})")
  _panel_letter(ax_c, f"c  5% {SYM_BASIC}-list identity")
  fig.subplots_adjust(left=0.068, right=0.99, top=0.90, bottom=0.15, wspace=0.32)
  _audit_figure_layout(fig, "figure3_review_capacity_5_10")
  return _save(fig, figures_dir, "figure3_review_capacity_5_10")


def figure4_subgroup_heterogeneity(figures_dir: Path, tables_dir: Path) -> list[str]:
  """Submission Figure 4: subgroup heterogeneity (patient-level prioritization moved to Figure 3).

  Nominal within-stratum paired intervals are clearly labeled as such; the post
  hoc within-dimension simultaneous bands and pairwise contrasts stay in the
  supplement (audit §5).
  Round-6 audit: the risk-tertile range-restriction caveat — a caveat about
  conditioning on C0+A risk strata — is no longer applied to demographic
  subgroups (round-06 P0). Estimand distinction uses marker shape (circle =
  with C0, triangle = ECG-only), keeping fill reserved for analysis status
  (round-06 P1).
  """
  subgroup = pl.read_csv(tables_dir / "table4_subgroup_gain.csv")
  ctx = [r for r in subgroup.to_dicts() if r["estimand"] == "context" and r["metric"] == "auroc"]
  ecg = {(r["variable"], r["level"]): r for r in subgroup.to_dicts() if r["estimand"] == "ecg_only" and r["metric"] == "auroc"}
  overall = next(
    r["estimate"] for r in (pl.read_csv(tables_dir / "table3_incremental_value_and_controls.csv").to_dicts())
    if r["contrast"] == "Context-adjusted: B - A" and r["metric"] == "auroc"
  )

  fig, ax = plt.subplots(figsize=(PUB_WIDTH, 0.26 * len(ctx) + 1.15))
  for i, r in enumerate(reversed(ctx)):
    ax.errorbar(r["gain_b_minus_a"], i, xerr=[[r["gain_b_minus_a"] - r["ci_lower"]], [r["ci_upper"] - r["gain_b_minus_a"]]],
                fmt="o", color=COLOR_DBETA, capsize=2, ms=4.0)
    e = ecg[(r["variable"], r["level"])]
    ax.errorbar(e["gain_b_minus_a"], i - 0.22, xerr=[[e["gain_b_minus_a"] - e["ci_lower"]], [e["ci_upper"] - e["gain_b_minus_a"]]],
                fmt="^", color=COLOR_DBETA, capsize=2, ms=3.6)
  ax.set_yticks(range(len(ctx)), [f"{r['label']} ({r['n_events']}/{r['n']:,})" for r in reversed(ctx)], fontsize=7.0)
  ax.axvline(0, color="black", lw=0.7)
  ax.axvline(float(overall), color=COLOR_DBETA, lw=0.7, ls="--")
  ax.set_xlabel(f"Paired ΔAUROC, {CON_AI_MORPH} (nominal 95% CI)")
  # Round-08 P0 (§2A): the resample/simultaneous-band/adjusted-contrast footer
  # moved to the caption (the adjusted female−male contrast is in Results
  # §2.1); only the compact marker key stays in the artwork.
  ax.legend(handles=[Line2D([], [], marker="o", color=COLOR_DBETA, ls="none", label=f"With {SYM_BASIC}"),
                     Line2D([], [], marker="^", color=COLOR_DBETA, ls="none", label="ECG-only")],
            fontsize=6.5, loc="lower right", frameon=False)
  fig.subplots_adjust(left=0.21, right=0.975, top=0.965, bottom=0.12)
  _audit_figure_layout(fig, "figure4_subgroup_heterogeneity")
  return _save(fig, figures_dir, "figure4_subgroup_heterogeneity")


def figure5_pretraining_label_efficiency(figures_dir: Path, tables_dir: Path, artifacts_root: Path) -> list[str]:
  """Submission Figure 5: what was controlled, absolute performance, and when pretraining helps.

  Panel a: matched two-stream schematic. Panels b/c: absolute AUROC by label
  fraction (ECG-only companion, with-C0 primary) from the post hoc
  label-efficiency pipeline. Panel d: pretrained-minus-random means with
  seed/subset ranges (not confidence intervals) plus the prespecified
  full-label per-initialization dots from the separate random-control run —
  the +0.0091 vs +0.0096 provenance distinction is drawn, not pooled (audit §6).
  Round-6 audit: panels are laid out in natural reading order a, b / c, d
  (round-06 P1).
  Round-7 audit (P1): panels b/c are an entirely post hoc label-efficiency
  experiment — all three series draw open markers and the panel titles state
  the status directly, so the filled=prespecified contract is reserved for the
  status-coded contrast panels (Figure 2c and panel d, where the post hoc
  label-efficiency lines are hollow and the prespecified full-label control is
  filled). Model identity stays blue/orange/grey; surface stays b ECG-only /
  c with C0.
  """
  long = pl.read_csv(artifacts_root / "probe" / "label_efficiency" / "label_efficiency_long.csv")
  summary = pl.read_csv(tables_dir / "supp_label_efficiency.csv")
  control = pl.read_csv(tables_dir / "supp_random_encoder_control.csv")
  fractions = [0.025, 0.05, 0.1, 0.25, 0.5, 1.0]

  def absolutes(surface: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {"A": [], "B": [], "RAND": []}
    for frac in fractions:
      sub = long.filter((pl.col("surface") == surface) & (pl.col("fraction") == frac) & (pl.col("status") == "ok"))
      rand_reps = sub.filter(pl.col("representation").str.starts_with("B_RAND"))
      seed_means = [cast(float, rand_reps.filter(pl.col("representation") == rep)["auroc"].mean()) for rep in rand_reps["representation"].unique().to_list()]
      out["A"].append(cast(float, sub.filter(pl.col("representation") == "A")["auroc"].mean()))
      out["B"].append(cast(float, sub.filter(pl.col("representation") == "B")["auroc"].mean()))
      out["RAND"].append(float(np.mean(seed_means)))
    return out

  ecg_abs = absolutes("ecg_only")
  ctx_abs = absolutes("context")

  fig, axes = plt.subplots(2, 2, figsize=(PUB_WIDTH, 6.9), gridspec_kw={"height_ratios": [0.9, 1.0]})
  ax_s, ax_e = axes[0]
  ax_c, ax_d = axes[1]
  ax_s.set_axis_off()

  # Panel a: matched two-stream schematic with short labels (round-08 P0 §2A;
  # round-09 §3 clinician-first wording — "previously trained network whose
  # weights were not updated" instead of "frozen encoder").
  _box(ax_s, 0.03, 0.64, 0.46, 0.23, "Previously trained model\n(D-BETA; 768 features;\nweights unchanged)", face="#dbe6f2", edge=COLOR_DBETA, lw=1.0, fontsize=6.8)
  _box(ax_s, 0.52, 0.64, 0.46, 0.23, "Ten random networks\n(same architecture;\nweights unchanged)", face="#fdf0e4", edge=COLOR_RANDOM, lw=1.0, fontsize=6.8)
  _arrow(ax_s, 0.25, 0.64, 0.25, 0.50)
  _arrow(ax_s, 0.75, 0.64, 0.75, 0.50)
  _box(ax_s, 0.03, 0.30, 0.94, 0.20, "Separately fitted logistic regression predictors\n(same development and evaluation procedures)", face="#f4f4f4", fontsize=6.8)
  _arrow(ax_s, 0.25, 0.30, 0.25, 0.17)
  _arrow(ax_s, 0.75, 0.30, 0.75, 0.17)
  _box(ax_s, 0.03, 0.07, 0.94, 0.10, "Same locked-test evaluation (25,430 / 924)", face="#e8f0e8", fontsize=6.8)
  _panel_letter(ax_s, "a  Matched design")

  x = [100 * f for f in fractions]
  xlim = (10 ** math.log10(2.0), 120.0)
  # Round-10 audit R10-03: one shared model-family legend for panels b/c,
  # placed below the panels outside the plotted data regions — the previous
  # per-axes `upper left` legends covered the high-label-fraction end of the
  # curves (the B series starts near 0.83 at the 2.5% tick). Handles are
  # captured from panel b and reused for both panels.
  shared_handles: list[Any] = []
  shared_labels: list[str] = []
  for ax, abs_data, title in ((ax_e, ecg_abs, "b  ECG-only · post hoc label-efficiency"),
                              (ax_c, ctx_abs, f"c  With {SYM_BASIC} · post hoc label-efficiency")):
    # Entirely post hoc experiment (round-07 P1): every series is drawn open and
    # the panel title states the status; model identity is colour, surface is
    # the panel, so the filled=prespecified contract is never contradicted.
    ax.plot(x, abs_data["B"], "o-", color=COLOR_DBETA, lw=1.2, mfc="white", label=f"AI-derived ECG features ({SYM_AI})")
    ax.plot(x, abs_data["RAND"], "s--", color=COLOR_RANDOM, lw=1.0, mfc="white", label="Random networks (10-init mean)")
    ax.plot(x, abs_data["A"], "^:", color=COLOR_MORPH, lw=0.9, mfc="white", label=f"Morphology measurements ({SYM_MORPH})")
    ax.set_xscale("log")
    ax.set_xticks(x, [f"{v:g}%" for v in x], fontsize=6.5)
    ax.set_xlim(*xlim)
    ax.set_xlabel("Available development/validation outcome labels")
    ax.set_ylabel("Locked-test AUROC")
    if not shared_handles:
      shared_handles, shared_labels = ax.get_legend_handles_labels()
    _panel_letter(ax, title)

  # Panel d: pretrained-minus-random contrasts by fraction (post hoc pipeline,
  # hollow markers per the fill = analysis-status contract), plus the
  # prespecified full-label control's per-initialization differences (filled).
  for surface, color, marker, label in (("ecg_only", COLOR_DBETA, "o", "ECG-only · post hoc runs"), ("context", COLOR_CONTEXT, "s", f"With {SYM_BASIC} · post hoc runs")):
    sub = summary.filter((pl.col("surface") == surface) & (pl.col("comparison") == "B_minus_random_mean_over_seeds")).sort("fraction")
    fr = [100 * float(v) for v in sub["fraction"].to_list()]
    mean = [float(v) for v in sub["mean_delta_auroc"].to_list()]
    lo = [float(v) for v in sub["min_seed_mean"].to_list()]
    hi = [float(v) for v in sub["max_seed_mean"].to_list()]
    ax_d.plot(fr, mean, marker, color=color, lw=1.1, mfc="white", label=label)
    for f, l, h in zip(fr, lo, hi):
      ax_d.vlines(f, l, h, color=color, lw=3.0, alpha=0.35)
  ctrl = control.filter(pl.col("estimand") == "ecg_only")["pretrained_minus_random_auroc"].cast(pl.Float64).to_numpy()
  rng = np.random.default_rng(42)
  jitter = rng.uniform(-1.7, 1.7, size=ctrl.size)
  ax_d.scatter(np.full(ctrl.size, 100.0) + jitter, ctrl, marker="D", s=14, facecolor=COLOR_DBETA, edgecolor=COLOR_DBETA, lw=0.5,
               label="Prespecified control (separate run)")
  ax_d.axhline(0, color="black", lw=0.7)
  ax_d.set_xscale("log")
  ax_d.set_xticks(x, [f"{v:g}%" for v in x], fontsize=6.5)
  ax_d.set_xlim(*xlim)
  ax_d.set_ylim(-0.014, 0.064)
  ax_d.set_xlabel("Available development/validation outcome labels")
  ax_d.set_ylabel("Pretrained − random ΔAUROC")
  ax_d.legend(fontsize=6.5, loc="upper right")
  _panel_letter(ax_d, "d  Pretrained − random")
  # Round-08 P0 (§2A): the multi-line narrative footer moved to the caption;
  # the seed-range/provenance numbers live in Results §3.1 and Table S3.
  # Round-10 R10-03: the shared b/c model-family legend sits in a reserved
  # bottom band (tight_layout rect), outside every plotted data region; panel
  # d keeps its own status/provenance key.
  fig.legend(shared_handles, shared_labels, loc="lower left", bbox_to_anchor=(0.015, 0.004),
             ncols=3, frameon=False, fontsize=6.5)
  fig.tight_layout(rect=(0, 0.055, 1, 1))
  _audit_figure_layout(fig, "figure5_pretraining_label_efficiency")
  return _save(fig, figures_dir, "figure5_pretraining_label_efficiency")


def run_submission_figures(figures_dir: Path, tables_dir: Path, artifacts_root: Path, clinical_tables_dir: Path = Path("docs/clinical_extensions/tables"), qa_report_path: Path | None = None) -> dict[str, Any]:
  """Render the five submission-facing figures from committed outputs (no fitting).

  Round-9 audit §2: every figure passes the rendered-geometry layout audit and
  a one-page collision report is written to ``qa_report_path`` (default
  ``outputs/figure_collision_report.md``) in addition to the PNG/SVG exports.
  """
  apply_style()
  figures_dir.mkdir(parents=True, exist_ok=True)
  _QA_REPORTS.clear()
  counts = _counts_for_figure1(tables_dir, artifacts_root)
  figures = {
    "figure1": figure1_cohort_evidence_flow(figures_dir, counts),
    "figure2": figure2_discrimination_attenuation(figures_dir, tables_dir, clinical_tables_dir),
    "figure3": figure3_review_capacity_5_10(figures_dir, tables_dir, artifacts_root),
    "figure4": figure4_subgroup_heterogeneity(figures_dir, tables_dir),
    "figure5": figure5_pretraining_label_efficiency(figures_dir, tables_dir, artifacts_root),
  }
  report_path = qa_report_path if qa_report_path is not None else Path("outputs/figure_collision_report.md")
  report_path.parent.mkdir(parents=True, exist_ok=True)
  report_path.write_text(collision_report_markdown(list(_QA_REPORTS)), encoding="utf-8")
  return {"status": "complete", "counts": counts, "figures": figures, "qa_reports": list(_QA_REPORTS), "qa_report_path": str(report_path)}
