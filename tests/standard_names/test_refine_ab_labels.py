"""The blind refine scores rest on labels that hold their own contract.

The refine A/B is judged by the labels a fresh worker wrote against the blind
sheet. Those labels are the measurement, so their shape is checked here rather
than trusted: one label per sheet row inside the domain, the counts the labels
fragment states, and one criterion per DD source — every set of rows that share
a source must be resolved by the single criterion the labels fragment records
for that source.

The fragment files are committed, so the check loads them directly. The labels
path is overridable through ``SRTJ_REFINE_AB_LABELS`` so a mutated copy can be
run against the same assertions.
"""

from __future__ import annotations

import html
import json
import os
import re
from pathlib import Path

_FRAGMENT_DIR = (
    Path(__file__).resolve().parents[2]
    / "docs/evidence/fragments/sn-review-through-jev"
)
_SHEET = _FRAGMENT_DIR / "srtj-refine-ab-label-sheet.json"
_LABELS = Path(
    os.environ.get(
        "SRTJ_REFINE_AB_LABELS",
        str(_FRAGMENT_DIR / "srtj-refine-ab-labels.json"),
    )
)
_LABELS_HTML = _FRAGMENT_DIR / "srtj-refine-ab-labels.html"


def _sheet() -> list[dict]:
    return json.loads(_SHEET.read_text())


def _labels() -> list[dict]:
    return json.loads(_LABELS.read_text())


def _fragment_text() -> str:
    return _LABELS_HTML.read_text()


def label_domain_findings(sheet: list[dict], labels: list[dict]) -> list[str]:
    """One label per sheet row, inside the domain, with a non-empty reason."""
    findings: list[str] = []
    sheet_ids = [row["sheet_id"] for row in sheet]
    label_ids = [row["sheet_id"] for row in labels]
    if len(set(sheet_ids)) != len(sheet_ids):
        findings.append("sheet carries duplicate sheet_id values")
    if len(set(label_ids)) != len(label_ids):
        findings.append("labels carry duplicate sheet_id values")
    if set(sheet_ids) != set(label_ids):
        findings.append("labels do not cover exactly the sheet's rows")
    for row in labels:
        if row["label"] not in (0, 1):
            findings.append(
                f"{row['sheet_id']}: label {row['label']!r} outside {{0, 1}}"
            )
        if not str(row.get("reason") or "").strip():
            findings.append(f"{row['sheet_id']}: empty reason")
    return findings


_COUNT_ROW = re.compile(
    r"<tr><td><code>(\d)</code></td><td>[^<]*</td><td>(\d+)</td></tr>"
)


def stated_counts(text: str) -> dict[int, int]:
    """The per-label row counts the fragment's label-counts table states."""
    counts = {int(label): int(rows) for label, rows in _COUNT_ROW.findall(text)}
    assert counts, "the fragment states no label counts"
    return counts


def source_groups(
    sheet: list[dict], labels: list[dict]
) -> dict[str, list[tuple[str, int]]]:
    """Rows and labels grouped by DD source path."""
    label_of = {row["sheet_id"]: row["label"] for row in labels}
    groups: dict[str, list[tuple[str, int]]] = {}
    for row in sheet:
        groups.setdefault(row["source_path"], []).append(
            (row["sheet_id"], label_of[row["sheet_id"]])
        )
    return groups


_CONSISTENCY_HEADING = 'id="srtj-refine-ab-labels-consistency"'


def criterion_table(text: str) -> list[tuple[str, list[tuple[str, int]], str]]:
    """The fragment's per-source criterion table.

    Each entry is the source path as recorded (some are truncated with an
    ellipsis), the rows and labels the fragment states for it, and the single
    criterion it says resolves the group.
    """
    anchor = text.index(_CONSISTENCY_HEADING)
    table = text[text.index("<table>", anchor) : text.index("</table>", anchor)]
    entries = []
    for chunk in table.split("<tr>")[1:]:
        cells = re.findall(r"<td>(.*?)</td>", chunk, re.S)
        if len(cells) != 3:
            continue
        token = re.search(r"<code>(.*?)</code>", cells[0], re.S)
        if token is None:
            continue
        rows = re.findall(r"(\d{3})\s*<code>(\d)</code>", cells[1])
        criterion = html.unescape(re.sub(r"<[^>]+>", "", cells[2])).strip()
        entries.append(
            (
                html.unescape(token.group(1)),
                [(f"row-{number}", int(label)) for number, label in rows],
                criterion,
            )
        )
    assert entries, "the fragment records no per-source criterion table"
    return entries


def _resolve_token(token: str, groups: dict[str, list]) -> str:
    """The data source path a (possibly truncated) fragment path names."""
    tail = token.split("…")[-1].removeprefix("dd:")
    matches = [path for path in groups if tail in path.removeprefix("dd:")]
    assert len(matches) == 1, f"fragment path {token!r} matches {len(matches)} sources"
    return matches[0]


# The criterion the labels fragment records for each source that carries mixed
# labels, reduced to the token test that decides each row. Every predicate is
# read off the criterion the fragment states for that source; it must accept the
# rows the fragment labels 1 and reject the rows it labels 0.
_CRITERIA: dict[str, tuple[str, object]] = {
    "dd:equilibrium/time_slice/boundary_secondary_separatrix/distance_inner_outer": (
        "a separation in the major-radius direction, so the name states the "
        "radial direction",
        lambda name: (
            "radial" in name
            and any(token in name for token in ("gap", "distance", "extent"))
        ),
    ),
    "dd:edge_profiles/ggd/ion/state/ionization_potential/values": (
        "the per-state ionization potential, not an undirected charge name",
        lambda name: "potential" in name,
    ),
    "dd:equilibrium/time_slice/profiles_1d/gm9": (
        "the metric coefficient gm9 = <1/R> itself",
        lambda name: "major_radius" in name,
    ),
    "dd:gyrokinetics/fluxes_integrated_norm/energy_phi_potential": (
        "the source's cause, the E cross B drift flux",
        lambda name: "e_cross_b" in name,
    ),
    "dd:magnetics/b_field_pol_probe/non_linear_response/b_field_non_linear": (
        "the poloidal field probe as the owner",
        lambda name: "probe" in name,
    ),
    "dd:spectrometer_visible/channel/grating_spectrometer/wavelength_calibration/gain": (
        "the pixel-to-wavelength calibration factor",
        lambda name: "calibration" in name,
    ),
    "dd:wall/global_quantities/electrons/gas_puff": (
        "the electron count the source counts",
        lambda name: "electron" in name,
    ),
    "dd:wall/global_quantities/neutral/particle_flux_from_plasma": (
        "no cause the source does not state",
        lambda name: "due_to" not in name,
    ),
}


def test_labels_are_one_per_sheet_row_with_a_reason():
    findings = label_domain_findings(_sheet(), _labels())
    assert findings == [], "labels violate their own contract: " + "; ".join(findings)


def test_label_counts_match_the_fragment():
    labels = _labels()
    counts = {
        0: sum(1 for row in labels if row["label"] == 0),
        1: sum(1 for row in labels if row["label"] == 1),
    }
    text = _fragment_text()
    stated = stated_counts(text)
    for label, rows in stated.items():
        assert counts[label] == rows, (
            f"label {label}: {counts[label]} written against {rows} stated"
        )
    assert sum(stated.values()) == len(labels)
    assert f"<td>{len(labels)}</td>" in text, "the fragment's total is stale"


def test_sources_are_grouped_and_the_fragment_names_every_mixed_one():
    groups = source_groups(_sheet(), _labels())
    mixed = {path for path, rows in groups.items() if len({lab for _, lab in rows}) > 1}
    homogeneous = set(groups) - mixed
    text = _fragment_text()
    flat = re.sub(r"\s+", " ", text)
    assert f"<strong>{len(groups)} distinct source paths</strong>" in flat
    assert f"<strong>{len(homogeneous)} groups are homogeneous</strong>" in flat
    assert f"<strong>{len(mixed)} groups carry mixed labels</strong>" in flat

    entries = criterion_table(text)
    recorded = {
        _resolve_token(token, groups): (rows, criterion)
        for token, rows, criterion in entries
    }
    assert set(recorded) == mixed, "the criterion table and the mixed groups disagree"
    for path, (rows, criterion) in recorded.items():
        assert criterion, f"{path}: the fragment records no criterion"
        assert sorted(rows) == sorted(groups[path]), f"{path}: row coverage differs"


def test_each_mixed_source_follows_its_recorded_criterion():
    sheet = _sheet()
    labels = _labels()
    name_of = {row["sheet_id"]: row["refined_name"] for row in sheet}
    groups = source_groups(sheet, labels)
    mixed = {path for path, rows in groups.items() if len({lab for _, lab in rows}) > 1}
    assert set(_CRITERIA) == mixed, (
        "a criterion is declared for a source the labels do not split, or a split "
        "source has none"
    )
    for path, rows in groups.items():
        if path not in mixed:
            assert len({lab for _, lab in rows}) == 1, f"{path}: not homogeneous"
            continue
        _, predicate = _CRITERIA[path]
        verdicts = []
        for sheet_id, label in sorted(rows):
            verdict = bool(predicate(name_of[sheet_id]))
            verdicts.append(verdict)
            assert verdict == (label == 1), (
                f"{sheet_id} ({name_of[sheet_id]}) is labelled {label}, but the "
                f"criterion for {path} reads {verdict}"
            )
        assert len(set(verdicts)) == 2, (
            f"{path}: the criterion does not decide the group — its rows are "
            "separated by something else"
        )
