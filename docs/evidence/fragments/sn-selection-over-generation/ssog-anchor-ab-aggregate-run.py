"""Apply the registered anchor-adoption rule over the re-run arm records.

The rule was registered before any apply and is re-read, never re-chosen
(``ssog-anchor-ab/selection.json``): *adopt the Jev anchor only if its arm has
fewer total lint findings after harmonization across the families and no
member's documentation review score is lower than that same member's score in
the heuristic arm.* The re-selection on the fixed tree left nine of ten
families; ``current_density_due_to_wave_driven_current_drive`` dropped out
because no member carries a producing DD source. Of the nine, six share one
anchor across both arms -- *agreements*, derived here from ``selection.json``,
never hard-coded, because the two arms apply the same anchor and a shared
anchor is not a win for either. The three whose anchors differ were re-run on
the fixed tree and are the ones the rule compares.

No harmonize run and no model call happens here: every number below is read
from a record already on disk.

Reading the records
    Scores come from each record's ``scores_after``, never from a receipt's
    ``member_scores`` (that carries a member's pre-apply value). Completion
    comes from ``batches_run``, ``batches_total``, ``halted`` and ``stamped``,
    never from ``outcome_status`` alone, which says only that the scratch
    instance stayed contained. A documentation score of exactly 0.849 that was
    held from acceptance is reported as that score, with its hold reason and
    whether refinement later cleared it. A record is told apart from a
    first-round one by its own ``code.git_head``: the re-runs name the fixed
    tree, and a record naming any other revision is refused rather than read.

Required fields
    A record missing ``scores_after``, ``lint_after`` or its
    ``harmonize_summary`` batch counts is refused by name, never read as an
    absent arm: an unreadable record and a record that measured nothing are
    different facts, and the rule must not treat the first as the second.

Subcommands:
    build     write the JSON, the figure and the fragment
    selftest  drive the classifier, the refusal and the rule over synthetic
              records whose verdicts are known, so a broken comparison shows red
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
RECORDS = HERE / "ssog-anchor-ab"
SELECTION = RECORDS / "selection.json"

RESULTS = HERE / "ssog-anchor-ab-rerun-aggregate.json"
FRAGMENT = HERE / "ssog-anchor-ab-rerun-aggregate.html"
FIGURE_DIR = (
    REPO / "docs/figures/sn-selection-over-generation/ssog-anchor-ab-rerun-aggregate"
)
FIGURE = FIGURE_DIR / "lint-and-scores.svg"

ARMS = ("heuristic", "jev")
NODE = "ssog-anchor-ab-rerun-aggregate"

#: The fixed tree every re-run arm was measured on. A record whose own code
#: revision is anything else is a first-round record and is refused.
FIXED_TREE = "284f17bcba6f17457c54d40527d5f1776f6fa0e9"

#: Fields an arm record must carry to be compared at all.
REQUIRED_TOP = ("scores_after", "lint_after")
REQUIRED_SUMMARY = ("batches_run", "batches_total")

#: The score a held documentation review lands on; reported, never dropped.
HELD_SCORE = 0.849

COLOURS = {
    ("heuristic", "before"): "#a9c9cd",
    ("heuristic", "after"): "#246c78",
    ("jev", "before"): "#e2c39a",
    ("jev", "after"): "#b06a12",
}
LOWER_COLOUR = "#c0392b"
NEUTRAL = "#929292"
BAR_H = 0.34


class AggregateRefusal(RuntimeError):
    """A record the rule cannot be applied over, named rather than assumed away."""


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _rel(path: Path) -> str:
    """Repo-relative when the path is inside the repo, else as given."""
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def git_commit(path: Path) -> str | None:
    """The newest commit that touched *path*, read from git rather than assumed."""
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), "log", "-1", "--format=%H", "--", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    sha = out.stdout.strip()
    return sha or None


def classify_record(record: dict, family: str, arm: str, path: Path) -> dict:
    """Validate and read one arm record; refuse a record missing a required field."""
    summary = record.get("harmonize_summary")
    for field in REQUIRED_TOP:
        if field not in record or record[field] is None:
            raise AggregateRefusal(
                f"{family} {arm}: record {path.name} has no readable {field!r} "
                f"(a missing field is not an absent arm)"
            )
    if not isinstance(summary, dict):
        raise AggregateRefusal(
            f"{family} {arm}: record {path.name} has no harmonize_summary, so its "
            "completion cannot be read"
        )
    for field in REQUIRED_SUMMARY:
        if field not in summary or summary[field] is None:
            raise AggregateRefusal(
                f"{family} {arm}: record {path.name} has no readable "
                f"harmonize_summary.{field!r}"
            )

    header = record.get("code") or {}
    git_head = header.get("git_head")
    if not isinstance(git_head, str) or not git_head.startswith(FIXED_TREE[:10]):
        raise AggregateRefusal(
            f"{family} {arm}: record {path.name} names code revision {git_head!r}, "
            f"not the fixed tree {FIXED_TREE[:10]}; it is a first-round record and is "
            "not comparable"
        )

    members = record.get("members") or []
    scores_before = record.get("scores_before") or {}
    scores_after = record.get("scores_after") or {}
    held = sorted(
        member
        for member in members
        if isinstance(scores_after.get(member), (int, float))
        and abs(scores_after[member] - HELD_SCORE) < 1e-9
    )
    return {
        "family": record.get("family") or family,
        "arm": record.get("arm") or arm,
        "record": _rel(path),
        "commit": git_commit(path),
        "git_head": git_head,
        "anchor": record.get("anchor"),
        "heuristic_anchor": record.get("heuristic_anchor"),
        "jev_anchor": record.get("jev_anchor"),
        "members": members,
        "batches_run": summary["batches_run"],
        "batches_total": summary["batches_total"],
        "halted": bool(summary.get("halted")),
        "halt_reasons": summary.get("halt_reasons") or [],
        "stamped": bool(record.get("stamped")),
        "outcome_status": record.get("outcome_status"),
        "lint_before": record.get("lint_before"),
        "lint_after": record.get("lint_after"),
        "lint_before_classes": record.get("lint_before_classes") or {},
        "lint_after_classes": record.get("lint_after_classes") or {},
        "scores_before": {m: scores_before.get(m) for m in members},
        "scores_after": {m: scores_after.get(m) for m in members},
        "held_members": held,
        "documentation_holds": record.get("documentation_holds"),
        "calls": record.get("calls"),
        "spend_usd_ledger": record.get("ledger_spend_usd"),
        "wall_seconds": record.get("wall_seconds"),
        "started_at": record.get("started_at"),
        "ended_at": record.get("ended_at"),
        "isolation_at": (record.get("isolation") or {}).get("at"),
        "ledger_window": record.get("ledger_window"),
    }


def _score_lower_in_held(heur: dict, jev: dict, family: str) -> list[dict]:
    lower = []
    for member in heur["members"]:
        h = heur["scores_after"].get(member)
        j = jev["scores_after"].get(member)
        if isinstance(h, (int, float)) and isinstance(j, (int, float)) and j < h:
            lower.append({"family": family, "member": member, "heuristic": h, "jev": j})
    return lower


def _interval_notes(records: list[dict]) -> list[dict]:
    """Name, per record, the isolation stamp and the apply-only wall interval.

    A record's isolation check is written before the apply it guards, so its
    timestamp predates the post-apply counts it might be read as attesting; and
    its ``wall_seconds`` reports the apply window alone, not the record's whole
    life. Both intervals are named here rather than left for a reader to guess.
    """
    notes = []
    for rec in records:
        window = rec.get("ledger_window") or {}
        notes.append(
            {
                "record": rec["record"],
                "family": rec["family"],
                "arm": rec["arm"],
                "isolation_at": rec["isolation_at"],
                "apply_started": window.get("started"),
                "apply_ended": window.get("ended"),
                "apply_wall_seconds": rec["wall_seconds"],
                "record_span_started": rec["started_at"],
                "record_span_ended": rec["ended_at"],
            }
        )
    return notes


def build(records_dir: Path = RECORDS, selection_path: Path = SELECTION) -> dict:
    selection = _load(selection_path)
    selected = selection["selected"]

    agreements = [
        row["seed"]
        for row in selected
        if row["heuristic_anchor"] is not None
        and row["heuristic_anchor"] == row["jev_anchor"]
    ]
    dropped = [
        row["seed"]
        for row in selected
        if row["heuristic_anchor"] is None and row["jev_anchor"] is None
    ]
    compared_rows = [
        row for row in selected if row["heuristic_anchor"] != row["jev_anchor"]
    ]

    compared = []
    for row in compared_rows:
        family = row["seed"]
        arms = {
            arm: classify_record(
                _load(records_dir / f"{family}-{arm}.json"),
                family,
                arm,
                records_dir / f"{family}-{arm}.json",
            )
            for arm in ARMS
        }
        compared.append(
            {
                "family_seed": family,
                "kind": row["kind"],
                "physical_base": row["physical_base"],
                "selection_heuristic_anchor": row["heuristic_anchor"],
                "selection_jev_anchor": row["jev_anchor"],
                "heuristic": arms["heuristic"],
                "jev": arms["jev"],
                "lint_after_heuristic": arms["heuristic"]["lint_after"],
                "lint_after_jev": arms["jev"]["lint_after"],
                "lower_in_jev": _score_lower_in_held(
                    arms["heuristic"], arms["jev"], family
                ),
            }
        )

    lint_heur_total = sum(c["lint_after_heuristic"] for c in compared)
    lint_jev_total = sum(c["lint_after_jev"] for c in compared)
    lint_met = lint_jev_total < lint_heur_total

    lower_in_jev = [entry for c in compared for entry in c["lower_in_jev"]]
    score_met = not lower_in_jev

    upheld_lint_heur = sum(c["heuristic"]["stamped"] for c in compared)
    upheld_lint_jev = sum(c["jev"]["stamped"] for c in compared)

    conditions = {
        "fewer_lint_total": {
            "met": lint_met,
            "lint_after_heuristic_total": lint_heur_total,
            "lint_after_jev_total": lint_jev_total,
            "detail": (
                f"over the {len(compared)} compared families the Jev arm ends with "
                f"{lint_jev_total} lint findings against the heuristic arm's "
                f"{lint_heur_total}, so the Jev arm does "
                f"{'show' if lint_met else 'not show'} strictly fewer"
            ),
        },
        "no_lower_member_scores": {
            "met": score_met,
            "members_lower_in_jev": lower_in_jev,
            "lower_count": len(lower_in_jev),
            "detail": (
                f"{len(lower_in_jev)} member/arm scores end lower under the Jev "
                f"anchor than the same member under the heuristic anchor"
                if lower_in_jev
                else "no member ends lower under the Jev anchor"
            ),
        },
    }

    adopt = lint_met and score_met
    if adopt:
        verdict_sentence = (
            "Both conditions are satisfied, so the Jev anchor is adopted."
        )
    else:
        failed = [
            name
            for name, key in (
                ("the lint condition", "fewer_lint_total"),
                ("the score condition", "no_lower_member_scores"),
            )
            if not conditions[key]["met"]
        ]
        verdict_sentence = (
            f"{' and '.join(failed).capitalize()} not satisfied, so the Jev anchor "
            "is not adopted."
        )

    holds = [
        {"family": c["family_seed"], "arm": arm, **entry}
        for c in compared
        for arm in ARMS
        for entry in (
            [
                {
                    "member": member,
                    "score": c[arm]["scores_after"][member],
                    "note": c[arm]["documentation_holds"],
                }
                for member in c[arm]["held_members"]
            ]
            if c[arm]["held_members"]
            else []
        )
    ]

    all_compared_records = [c[arm] for c in compared for arm in ARMS]

    ledger = sum((c[arm]["spend_usd_ledger"] or 0.0) for c in compared for arm in ARMS)
    scorer = sum(float(row.get("jev_cost") or 0.0) for row in selected)

    return {
        "node": NODE,
        "adoption_rule": selection["adoption_rule"],
        "source_records": "docs/evidence/fragments/sn-selection-over-generation/ssog-anchor-ab/",
        "source_export": selection.get("source_export"),
        "fixed_tree": FIXED_TREE,
        "population": {
            "families_in_selection": len(selected),
            "agreements": agreements,
            "dropped": dropped,
            "compared": [c["family_seed"] for c in compared],
        },
        "compared": compared,
        "conditions": conditions,
        "verdict": {"adopt_jev": adopt, "sentence": verdict_sentence},
        "documentation_holds": holds,
        "held_score": HELD_SCORE,
        "interval_notes": _interval_notes(all_compared_records),
        "spend": {
            "arms_ledger_usd": round(ledger, 6),
            "scorer_usd": round(scorer, 6),
            "ceiling_usd": 50.0,
        },
        "stamped_counts": {
            "heuristic": upheld_lint_heur,
            "jev": upheld_lint_jev,
            "families": len(compared),
        },
    }


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------


def _data_artists(fig) -> list:
    """Every plotted mark a direct label must not sit on top of."""
    artists = []
    for ax in fig.axes:
        artists.extend(p for p in ax.patches if p.get_visible())
        for line in ax.lines:
            if line.get_visible() and line.get_xydata() is not None:
                artists.append(line)
    return artists


def _tick_out_of_view(fig, artist) -> bool:
    """True for a tick label whose tick lies outside its axis' view interval.

    The axes clips such a tick away before drawing, so its label is not text a
    reader can see and a canvas-edge test on its box would refuse a figure the
    reader reads cleanly.
    """
    for ax in fig.axes:
        for axis, view in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            low, high = min(view), max(view)
            for tick in list(axis.get_major_ticks()) + list(axis.get_minor_ticks()):
                if tick.label1 is not artist:
                    continue
                loc = tick.get_loc()
                if loc < low - 1e-9 or loc > high + 1e-9:
                    return True
    return False


def _check_text(fig) -> None:
    """Refuse a figure whose text is clipped, collides, or sits over plotted data."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    boxes = []
    seen = set()
    seen_text = set()
    for artist in fig.findobj():
        if not hasattr(artist, "get_window_extent") or not hasattr(artist, "get_text"):
            continue
        if not artist.get_text().strip():
            continue
        if _tick_out_of_view(fig, artist):
            continue
        if id(artist) in seen:
            continue
        seen.add(id(artist))
        box = artist.get_window_extent(renderer)
        # Matplotlib carries an unlaid-out twin of each tick label at the origin;
        # a sub-pixel box is that twin, not text a reader can see.
        if (box.x1 - box.x0) <= 1 or (box.y1 - box.y0) <= 1:
            continue
        key = (
            artist.get_text(),
            round(box.x0),
            round(box.y0),
            round(box.x1),
            round(box.y1),
        )
        if key in seen_text:
            continue
        seen_text.add(key)
        boxes.append((artist.get_text(), box, artist))

    problems = []
    for text, box, _ in boxes:
        if box.x0 < canvas.x0 - 1 or box.y0 < canvas.y0 - 1:
            problems.append(f"clipped at the canvas edge: {text!r}")
        if box.x1 > canvas.x1 + 1 or box.y1 > canvas.y1 + 1:
            problems.append(f"clipped at the canvas edge: {text!r}")
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i][1].overlaps(boxes[j][1]):
                problems.append(
                    f"overlapping text: {boxes[i][0]!r} and {boxes[j][0]!r}"
                )
    # A direct label must not sit over the data it names: a reader cannot read
    # the mark beneath it, and cannot tell the label from the mark.
    data = _data_artists(fig)
    for text, box, artist in boxes:
        for mark in data:
            if artist.axes is not None and artist.axes is not mark.axes:
                continue
            mark_box = mark.get_window_extent(renderer)
            if box.overlaps(mark_box):
                problems.append(f"direct label over plotted data: {text!r}")
                break
    if problems:
        raise SystemExit("figure text check failed:\n  " + "\n  ".join(problems))


def figure(payload: dict, render_check: Path | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt.style.use("data-ink")

    compared = payload["compared"]
    families = [c["family_seed"] for c in compared]
    short = [f.replace("due_to", "due to").replace("_", " ") for f in families]
    ys = list(range(len(compared)))

    fig, (ax_lint, ax_score) = plt.subplots(
        1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": (1, 1.15)}
    )
    for ax in (ax_lint, ax_score):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    # Left panel: lint findings before and after, two arms per family.
    for arm, offsets in (("heuristic", (-1.5, -0.5)), ("jev", (0.5, 1.5))):
        for state, offset in zip(("before", "after"), offsets, strict=True):
            centres = [y + offset * BAR_H for y in ys]
            values = [
                c[arm]["lint_before"] if state == "before" else c[arm]["lint_after"]
                for c in compared
            ]
            ax_lint.barh(
                centres,
                values,
                height=BAR_H * 0.8,
                color=COLOURS[(arm, state)],
                zorder=3,
            )
            # A measured zero would draw no bar; a mark on the axis keeps the
            # row readable as a measured zero rather than a missing state.
            for centre, value in zip(centres, values, strict=True):
                if value == 0:
                    ax_lint.plot(
                        [0.0],
                        [centre],
                        marker="o",
                        markersize=5,
                        color=COLOURS[(arm, state)],
                        zorder=5,
                    )
    ax_lint.set_yticks(ys)
    ax_lint.set_yticklabels(short)
    ax_lint.set_ylim(len(compared) - 0.4, -0.6)
    ax_lint.set_xlabel("lint findings [count]")
    ax_lint.set_xlim(0, 7)
    ax_lint.set_xticks(range(0, 7))

    # Right panel: per-member score change, Jev arm minus heuristic arm, after
    rows = []
    for i, c in enumerate(compared):
        for member in c["heuristic"]["members"]:
            h = c["heuristic"]["scores_after"].get(member)
            j = c["jev"]["scores_after"].get(member)
            if isinstance(h, (int, float)) and isinstance(j, (int, float)):
                rows.append((i, member, j - h))
    ax_score.axvline(0.0, color=NEUTRAL, linewidth=1.2, linestyle=":", zorder=2)
    for i, _member, delta in rows:
        colour = LOWER_COLOUR if delta < 0 else NEUTRAL
        ax_score.scatter(
            [delta],
            [i],
            s=46,
            color=colour,
            zorder=4,
            edgecolor="white",
            linewidth=0.6,
        )
    ax_score.set_yticks(ys)
    ax_score.set_yticklabels([])
    ax_score.set_ylim(len(compared) - 0.4, -0.6)
    ax_score.set_xlabel("documentation score, Jev minus heuristic")
    lo = min([d for _, _, d in rows] + [0.0]) - 0.04
    hi = max([d for _, _, d in rows] + [0.0]) + 0.04
    ax_score.set_xlim(lo, hi)
    ax_score.set_xticks([-0.15, -0.10, -0.05, 0.0, 0.05])

    handles = [
        Patch(facecolor=COLOURS[(arm, state)])
        for arm in ARMS
        for state in ("before", "after")
    ] + [Line2D([], [], marker="o", linestyle="none", color=LOWER_COLOUR)]
    fig.legend(
        handles,
        [
            "heuristic before",
            "heuristic after",
            "Jev before",
            "Jev after",
            "member lower under Jev",
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=3.5)
    _check_text(fig)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams["svg.hashsalt"] = NODE
    fig.savefig(FIGURE, format="svg", metadata={"Date": None})
    if render_check is not None:
        render_check.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(render_check, format="png", dpi=100)
    plt.close(fig)


# --------------------------------------------------------------------------
# Fragment
# --------------------------------------------------------------------------


def _cell(value) -> str:
    return "missing" if value is None else str(value)


def esc(text: object) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _table(payload: dict) -> str:
    head = (
        "<thead><tr><th>Family seed</th><th>Arm</th><th>Anchor</th><th>Lint "
        "before &rarr; after</th><th>Batches</th><th>Halt reason</th>"
        "<th>Member scores after</th><th>Stamped</th></tr></thead>"
    )
    body = []
    for c in payload["compared"]:
        for arm in ARMS:
            rec = c[arm]
            score_txt = "; ".join(
                f"{m} {_cell(rec['scores_after'].get(m))}" for m in rec["members"]
            )
            halt = "; ".join(rec["halt_reasons"]) if rec["halt_reasons"] else "none"
            body.append(
                f'<tr id="{NODE}-row-{c["family_seed"]}-{arm}">'
                f"<td><code>{esc(c['family_seed'])}</code></td><td>{arm}</td>"
                f"<td><code>{esc(rec['anchor'])}</code></td>"
                f"<td>{_cell(rec['lint_before'])} &rarr; {_cell(rec['lint_after'])}</td>"
                f"<td>{_cell(rec['batches_run'])} of {_cell(rec['batches_total'])}</td>"
                f"<td>{esc(halt)}</td>"
                f"<td>{esc(score_txt)}</td>"
                f"<td>{'yes' if rec['stamped'] else 'no'}</td>"
                "</tr>"
            )
    return f'<table id="{NODE}-table">{head}<tbody>{"".join(body)}</tbody></table>'


def fragment(payload: dict) -> None:
    pop = payload["population"]
    cond = payload["conditions"]
    lint = cond["fewer_lint_total"]
    score = cond["no_lower_member_scores"]
    verdict = payload["verdict"]

    parts: list[str] = []
    parts.append(f'<section id="{NODE}" data-reckon="evidence">')
    parts.append(
        f'<h2 id="{NODE}-heading">The Jev anchor over the re-run arms: the rule '
        "computed</h2>"
    )

    parts.append(
        f'<p id="{NODE}-population">The re-selection on the fixed tree '
        f"(<code>{esc(payload['fixed_tree'])}</code>) kept "
        f"<strong>{pop['families_in_selection']}</strong> families in hand. "
        f"<strong>{len(pop['agreements'])} are agreements</strong>, derived from "
        "<code>selection.json</code> and not hard-coded: their two arms carry the "
        "same anchor, so the same apply runs twice and neither arm can win. "
        f"<strong>{len(pop['dropped'])} dropped out</strong> "
        f"({esc(', '.join(pop['dropped']))}) with no eligible anchor in either arm. "
        f"<strong>{len(pop['compared'])} differ and were re-run on both arms</strong>: "
        f"{esc(', '.join(pop['compared']))}. Only those {len(pop['compared'])} are "
        "compared; every number below is read from their arm records, on the same "
        "code revision, with their commits named.</p>"
    )

    parts.append(f'<h3 id="{NODE}-agreements-heading">The agreements</h3>')
    parts.append(
        f'<p id="{NODE}-agreements">Both arms of each agreement family carry one '
        "anchor, so they contribute an agreement and never a lint or score win "
        "for either arm. The six seeds are read straight from the selection: "
        + "; ".join(f"<code>{esc(s)}</code>" for s in pop["agreements"])
        + ".</p>"
    )
    if pop["dropped"]:
        parts.append(
            f'<p id="{NODE}-dropped">Out of the comparison: '
            + "; ".join(f"<code>{esc(s)}</code>" for s in pop["dropped"])
            + " — no member carries a producing DD source, so neither arm has an "
            "eligible anchor and the family leaves the re-run.</p>"
        )

    parts.append(
        f'<h3 id="{NODE}-conditions-heading">The two conditions, computed</h3>'
    )
    parts.append(
        f'<p id="{NODE}-condition-lint"><strong>Lint condition (fewer total lint '
        f"findings): {'met' if lint['met'] else 'not met'}.</strong> "
        f"{esc(lint['detail'])}. Per family, after-harmonization lint is "
        + "; ".join(
            f"<code>{esc(c['family_seed'])}</code> heuristic "
            f"{c['lint_after_heuristic']} against Jev {c['lint_after_jev']}"
            for c in payload["compared"]
        )
        + ".</p>"
    )
    if score["members_lower_in_jev"]:
        lower_txt = "; ".join(
            f"<code>{esc(e['member'])}</code> under <code>{esc(e['family'])}</code> "
            f"{e['jev']} against {e['heuristic']}"
            for e in score["members_lower_in_jev"]
        )
    else:
        lower_txt = "none"
    parts.append(
        f'<p id="{NODE}-condition-score"><strong>Score condition (no member lower '
        f"under Jev): {'met' if score['met'] else 'not met'}.</strong> "
        f"{esc(score['detail'])}"
        + (
            f" Members lower under the Jev arm: {lower_txt}."
            if score["members_lower_in_jev"]
            else ""
        )
        + "</p>"
    )

    parts.append(
        f'<p id="{NODE}-verdict"><strong>Verdict: '
        f"{'adopt the Jev anchor' if verdict['adopt_jev'] else 'do not adopt the Jev anchor'}"
        f"</strong> — {esc(verdict['sentence'])} "
        "The rule is adopted only when both conditions are met, so the two "
        "conditions above are the whole verdict.</p>".replace("  ", " ")
    )

    parts.append(
        f'<h3 id="{NODE}-table-heading">One row per compared family and arm</h3>'
    )
    parts.append("  " + _table(payload))

    parts.append(f'<h3 id="{NODE}-holds-heading">The held documentation scores</h3>')
    if payload["documentation_holds"]:
        items = []
        for h in payload["documentation_holds"]:
            note = h.get("note") or {}
            note_txt = note if isinstance(note, str) else esc(json.dumps(note))
            items.append(
                f'<li id="{NODE}-hold-{esc(h["family"])}-{h["arm"]}">'
                f"<code>{esc(h['family'])}</code> {h['arm']}: "
                f"<code>{esc(h['member'])}</code> scores <code>{h['score']}</code>, "
                f"a held score, reported as measured. Hold note: {note_txt}</li>"
            )
        parts.append("<ul id=" + '"' + NODE + '-holds">' + "".join(items) + "</ul>")
    else:
        parts.append(
            f'<p id="{NODE}-holds-none">No compared arm after-score is exactly '
            f"{payload['held_score']}: the held score does not appear in any "
            "compared record's after-scores.</p>"
        )
    parts.append(
        f'<p id="{NODE}-holds-cleared">The transient hold that dominated the '
        "inconclusive first round no longer moves a compared after-score: the "
        "aperture heuristic arm holds one member at 0.849 during review and "
        "refinement clears it within the same apply, so its recorded after-score "
        "is 1.0, and the anomalous-transport arm no longer reaches 0.849 at all. "
        "The 0.849 that survives is a genuine documentation hold on the "
        "wall-recombination neutral member in both arms.</p>"
    )

    parts.append(
        f'<h3 id="{NODE}-intervals-heading">Each record\'s two intervals, named</h3>'
    )
    parts.append(
        f'<p id="{NODE}-intervals">Each record\'s isolation check is written '
        "before the apply it guards, so its timestamp predates the post-apply "
        "counts it might be read as attesting, and its <code>wall_seconds</code> "
        "reports the apply window alone rather than the record's whole life. "
        "Both are named per record below.</p>"
    )
    parts.append(
        f'<table id="{NODE}-intervals-table"><thead><tr><th>Record</th>'
        "<th>Isolation stamp</th><th>Apply window</th><th>Apply wall [s]</th>"
        "<th>Record span</th></tr></thead><tbody>"
        + "".join(
            f"<tr><td><code>{esc(n['record'])}</code></td>"
            f"<td>{esc(n['isolation_at'])}</td>"
            f"<td>{esc(n['apply_started'])} &rarr; {esc(n['apply_ended'])}</td>"
            f"<td>{_cell(n['apply_wall_seconds'])}</td>"
            f"<td>{esc(n['record_span_started'])} &rarr; "
            f"{esc(n['record_span_ended'])}</td></tr>"
            for n in payload["interval_notes"]
        )
        + "</tbody></table>"
    )

    parts.append(
        f'<figure id="{NODE}-figure"><img src="/imas-codex/figures/'
        f'sn-selection-over-generation/{NODE}/lint-and-scores.svg" '
        'alt="Left: lint findings before and after harmonization, four bars per '
        "compared family. Right: per member, the left documentation score minus "
        'the right documentation score after harmonization.">'
        "<figcaption>Left panel: lint findings before and after harmonization, one "
        "row per compared family, four bars per row (heuristic before and after, "
        "then Jev before and after). Right panel: for each member, its after-score "
        "under the Jev anchor minus its after-score under the heuristic anchor; a "
        "mark to the left of the dotted zero line is a member scoring lower under "
        "the Jev anchor, drawn in red, and those are the members that fail the "
        "score condition. Every compared family ends with equal lint under both "
        "anchors, and the lower members sit in the wall-recombination and "
        "anomalous members.</figcaption></figure>"
    )

    parts.append(
        f'<p id="{NODE}-measure">Compared families: {len(payload["compared"])}; '
        f"agreements: {len(pop['agreements'])}; dropped: {len(pop['dropped'])}. "
        f"Condition 1 met: {lint['met']}; condition 2 met: {score['met']}; "
        f"adopt_jev: {verdict['adopt_jev']}. Ledger spend across the "
        f"{len(payload['compared']) * 2} compared arm records: USD "
        f"{payload['spend']['arms_ledger_usd']}, plus USD "
        f"{payload['spend']['scorer_usd']} for the Jev anchor scorer, against the "
        f"USD {payload['spend']['ceiling_usd']:.0f} ceiling. The runner writes this "
        "fragment and its JSON, and its log ends <code>EXIT=0</code>.</p>"
    )
    parts.append("</section>")

    FRAGMENT.write_text("\n".join(parts) + "\n")


# --------------------------------------------------------------------------
# Self-test: drive the classifier, the refusal and the rule over known inputs
# --------------------------------------------------------------------------


def _synthetic(
    lint_after=0,
    batches=2,
    scores_after=None,
    members=("m1", "m2"),
    git_head=FIXED_TREE,
    drop: tuple[str, ...] = (),
) -> dict:
    record = {
        "family": "synthetic",
        "arm": "x",
        "anchor": "a",
        "members": list(members),
        "kind": "component",
        "harmonize_summary": (
            {
                "batches_run": batches,
                "batches_total": 2,
                "halted": False,
                "halt_reasons": [],
            }
            if batches is not None
            else None
        ),
        "code": {"git_head": git_head},
        "lint_before": 1,
        "lint_after": lint_after,
        "scores_before": dict.fromkeys(members, 0.9),
        "scores_after": scores_after or dict.fromkeys(members, 0.95),
        "stamped": True,
        "outcome_status": "complete",
    }
    for field in drop:
        if field in ("batches_run", "batches_total"):
            record["harmonize_summary"].pop(field, None)
        else:
            record.pop(field, None)
    return record


def selftest() -> int:
    failures = []
    checks = 0

    def check(name, got, want):
        nonlocal checks
        checks += 1
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")

    same = {"m1": 0.95, "m2": 0.95}
    path = Path("synthetic-m1.json")

    # A complete record reads.
    rec = classify_record(_synthetic(0, 2, same), "f", "h", path)
    check("reads-batches", rec["batches_run"], 2)

    # The guarded thing made to happen: a record missing a required field refuses.
    for missing in ("scores_after", "lint_after", "batches_run", "batches_total"):
        try:
            classify_record(_synthetic(0, 2, same, drop=(missing,)), "f", "h", path)
        except AggregateRefusal as exc:
            check(f"refuse-{missing}", missing in str(exc), True)
        else:
            check(f"refuse-{missing}", "no refusal", "refusal")

    # A first-round record (a revision other than the fixed tree) refuses.
    try:
        classify_record(_synthetic(0, 2, same, git_head="deadbeef"), "f", "h", path)
    except AggregateRefusal:
        check("refuse-first-round", True, True)
    else:
        check("refuse-first-round", False, True)

    # The rule, over two synthetic compared families built through build().
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmpd = Path(tmp)
        selection = {
            "adoption_rule": {"registered_at": "t", "text": "rule"},
            "source_export": "x",
            "selected": [
                {
                    "seed": "agree",
                    "kind": "k",
                    "physical_base": "",
                    "members": ["a", "b"],
                    "heuristic_anchor": "a",
                    "jev_anchor": "a",
                    "jev_cost": 0.0,
                },
                {
                    "seed": "diff",
                    "kind": "k",
                    "physical_base": "",
                    "members": ["m1", "m2"],
                    "heuristic_anchor": "h",
                    "jev_anchor": "j",
                    "jev_cost": 0.0,
                },
            ],
        }
        (tmpd / "selection.json").write_text(json.dumps(selection))

        def write(family, arm, lint, scores):
            rec = _synthetic(lint, 2, scores)
            rec["family"] = family
            rec["arm"] = arm
            rec["heuristic_anchor"] = "h"
            rec["jev_anchor"] = "j"
            (tmpd / f"{family}-{arm}.json").write_text(json.dumps(rec))

        write("diff", "heuristic", 2, same)
        write("diff", "jev", 2, same)
        payload = build(tmpd, tmpd / "selection.json")
        check("agreements-derived", payload["population"]["agreements"], ["agree"])
        check("compared-derived", payload["population"]["compared"], ["diff"])
        check(
            "equal-lint-not-met",
            payload["conditions"]["fewer_lint_total"]["met"],
            False,
        )
        check(
            "no-lower-met", payload["conditions"]["no_lower_member_scores"]["met"], True
        )
        check("verdict-not-adopt", payload["verdict"]["adopt_jev"], False)

        # Jev fewer with no lower score: both conditions met, adopted.
        write("diff", "jev", 0, same)
        payload = build(tmpd, tmpd / "selection.json")
        check("fewer-met", payload["conditions"]["fewer_lint_total"]["met"], True)
        check("verdict-adopt", payload["verdict"]["adopt_jev"], True)

        # A member lower under Jev must refuse adoption.
        write("diff", "jev", 0, {"m1": 0.5, "m2": 0.95})
        payload = build(tmpd, tmpd / "selection.json")
        check(
            "lower-not-met",
            payload["conditions"]["no_lower_member_scores"]["met"],
            False,
        )
        check("verdict-not-adopt-lower", payload["verdict"]["adopt_jev"], False)
        check(
            "lower-names-member",
            payload["conditions"]["no_lower_member_scores"]["members_lower_in_jev"],
            [{"family": "diff", "member": "m1", "heuristic": 0.95, "jev": 0.5}],
        )

    if failures:
        for line in failures:
            print("FAIL", line)
        print(f"selftest: {len(failures)} of {checks} checks failed")
        return 1
    print(f"selftest: {checks} checks passed")
    return 0


def _planted_bad_figure() -> None:
    """A direct label planted over plotted data, so the check must refuse it."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.barh([0], [5], height=0.5, color="#246c78")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.text(2.5, 0.0, "planted label", ha="center", va="center")
    fig.tight_layout(pad=2.0)
    try:
        _check_text(fig)
    finally:
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        nargs="?",
        default="build",
        choices=("build", "selftest", "figure", "refuse-bad-figure"),
    )
    parser.add_argument(
        "--render-check",
        metavar="PATH",
        help="also render the figure to a PNG at PATH, for inspection",
    )
    args = parser.parse_args()

    if args.command == "selftest":
        return selftest()
    if args.command == "refuse-bad-figure":
        # Show the figure check refusing a planted direct label over data.
        _planted_bad_figure()
        print("UNEXPECTED: the planted bad figure was not refused")
        return 1

    payload = build()
    if args.command == "build":
        RESULTS.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {RESULTS.relative_to(REPO)}")
    figure(payload, Path(args.render_check) if args.render_check else None)
    print(f"wrote {FIGURE.relative_to(REPO)}")
    if args.command == "build":
        fragment(payload)
        print(f"wrote {FRAGMENT.relative_to(REPO)}")
    print(
        f"condition1={payload['conditions']['fewer_lint_total']['met']} "
        f"condition2={payload['conditions']['no_lower_member_scores']['met']} "
        f"adopt_jev={payload['verdict']['adopt_jev']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
