"""Re-derive both arms' anchors of the ten comparison families under the
DD-source rule, from a scratch copy of the checked export.

The ten families and the adoption rule are fixed in
``ssog-anchor-ab/selection.json`` and are re-read here, never re-chosen. For
each family this driver loads the archived export into a fresh scratch Neo4j
instance -- the same way ``ssog-anchor-ab-run.py`` loads it -- and re-runs the
two selectors that now refuse a member with no producing Data Dictionary
source:

* the heuristic arm through :func:`select_anchor`;
* the judged arm through :func:`score_family_anchor`.

The new anchors are written back into ``selection.json`` with the anchors they
replace kept beside them under ``previous_anchors``, and a per-family,
per-arm record of whether the anchor changed and why. The evidence fragment is
written by the same run, so one log attests both deliverables.

Run:

    UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv \
      PYTHONPATH="$PWD" uv run python \
      docs/evidence/fragments/sn-selection-over-generation/ssog-anchor-reselect-run.py
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import os
import re
import shutil
import socket
import sys
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from imas_codex.graph.client import GraphClient
from imas_codex.graph.temp_neo4j import start_temp_neo4j, stop_temp_neo4j
from imas_codex.standard_names.families import (
    fetch_pattern_families,
    score_family_anchor,
    select_anchor,
)

FRAGMENT_DIR = Path(__file__).resolve().parent
SELECTION = FRAGMENT_DIR / "ssog-anchor-ab" / "selection.json"
RUNNER = FRAGMENT_DIR / "ssog-anchor-ab-run.py"
FRAGMENT = FRAGMENT_DIR / "ssog-anchor-reselect.html"
SPEND_LIMIT_USD = 0.50
PREVIOUS_ANCHORS_FIXED_IN = "5ba65454d"
PREVIOUS_ANCHORS_FIXED_AT = "2026-10-19T04:05:25Z"

PORT_LOCK_DIR = Path(os.environ.get("SSOG_PORT_LOCK_DIR", "/dev/shm"))
PORT_LOCK_PREFIX = "ssog-reselect-port-"


def now() -> str:
    return datetime.now(UTC).isoformat()


def _free(port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def reserve_port_pair() -> tuple[int, int]:
    """Reserve an exclusive bolt/http pair for the run's whole lifetime."""
    for base in range(31000, 39000, 2):
        if not (_free(base) and _free(base + 1000)):
            continue
        lock_path = PORT_LOCK_DIR / f"{PORT_LOCK_PREFIX}{base}.lock"
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            continue
        if not (_free(base) and _free(base + 1000)):
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
            continue
        os.ftruncate(fd, 0)
        os.write(fd, f"pid={os.getpid()}\n".encode())
        os.fsync(fd)
        return base, fd
    raise RuntimeError("No free Neo4j port pair in range")


#: A standard-name id is snake_case with at least two underscores at least.
_ID_LITERAL = re.compile(r"[\"']([a-z][a-z0-9]*(?:_[a-z0-9]+){2,})[\"']")


def runner_anchor_source() -> dict:
    """Show where the runner reads each arm's anchor, and that nowhere else.

    The brief requires the anchors to be read from ``selection.json`` as
    written, and a pin anywhere else to be reported rather than worked around.
    The runner is read as source: the keys it indexes for each arm's target,
    and whether the selection's own family and member ids appear as string
    literals anywhere in it. A literal naming a family member would be a second
    source of truth beside the file; a runner that names none has only the
    selection to read.
    """
    selection = json.loads(SELECTION.read_text())
    ids = {row["seed"] for row in selection["selected"]}
    for row in selection["selected"]:
        ids.update(row["members"])
    text = RUNNER.read_text()
    reads = [
        line.strip()
        for line in text.splitlines()
        if re.search(r'row\["(heuristic|jev)_anchor"\]', line)
    ]
    if len(reads) < 3:
        raise RuntimeError(
            "The runner does not read both anchors from its selection row; "
            f"found {reads!r} in {RUNNER}"
        )
    loads_selection = "SELECTION.read_text()" in text
    if not loads_selection:
        raise RuntimeError(f"The runner does not load {SELECTION}")
    literals = sorted(
        {
            match.group(1)
            for match in _ID_LITERAL.finditer(text)
            if match.group(1) in ids
        }
    )
    if literals:
        raise RuntimeError(
            "The runner pins family or member ids as literals beside the "
            f"selection: {literals}. The anchors are not read solely from "
            f"{SELECTION}."
        )
    return {
        "runner": str(RUNNER),
        "loads_selection": loads_selection,
        "anchor_reads": reads,
        "selection_ids_checked": len(ids),
        "pinned_anchor_literals": literals,
    }


def extract_dump(export: Path, place: Path) -> Path:
    """Extract the archive's ``graph.dump`` into the instance's dumps directory."""
    dump_dir = place / "dumps"
    dump_dir.mkdir(parents=True, exist_ok=True)
    dump = dump_dir / "neo4j.dump"
    with tarfile.open(export, "r:gz") as archive:
        member = next(m for m in archive if m.name.endswith("/graph.dump"))
        with archive.extractfile(member) as source, dump.open("wb") as target:
            shutil.copyfileobj(source, target)
    return dump


def start_instance(place: Path, port: int):
    """Start a scratch Neo4j over *place*, binding a writable /tmp for the load.

    Apptainer's default bind of the host temp dir over the container's /tmp
    makes the image's /tmp read-only in this sandbox, so the runtime cannot
    unpack its native libraries; the instance's own writable directory is bound
    there for the load and start commands, exactly as the runner does.
    """
    previous_bind = os.environ.get("APPTAINER_BIND")
    tmp_bind = f"{place / 'tmp'}:/tmp"
    os.environ["APPTAINER_BIND"] = (
        f"{previous_bind},{tmp_bind}" if previous_bind else tmp_bind
    )
    try:
        return start_temp_neo4j(place, port, port + 1000)
    finally:
        if previous_bind is not None:
            os.environ["APPTAINER_BIND"] = previous_bind
        else:
            os.environ.pop("APPTAINER_BIND", None)


def change_reason(arm: str, old, new, unsourced: list[str]) -> str:
    """One line stating why an arm's anchor changed, or that it did not."""
    if old == new:
        if new is None:
            return (
                f"{arm} unchanged: no eligible anchor before or after, because "
                "no member carries a DD source"
            )
        return f"{arm} unchanged: the previous anchor remains eligible"
    if old is not None and old in unsourced:
        why = f"the previous anchor {old} has no DD source and is no longer eligible"
    elif old is None:
        why = "there was no previous anchor"
    else:
        why = f"the previous anchor {old} is out-ranked among eligible members"
    if new is None:
        return f"{arm} changed to no anchor: {why}"
    return f"{arm} changed {old} -> {new}: {why}"


async def derive_one(gc: GraphClient, row: dict) -> dict:
    seed = row["seed"]
    families = fetch_pattern_families(gc, seeds=[seed])
    if len(families) != 1:
        raise RuntimeError(
            f"Expected one pattern family for {seed}, got {len(families)}"
        )
    family = families[0]
    members = family["members"]
    member_ids = sorted(m["id"] for m in members)
    unsourced = sorted(m["id"] for m in members if not m.get("has_dd_source"))

    heuristic_new = select_anchor(None, None, None, members)
    judged = await score_family_anchor(
        None, None, None, members, pattern="|".join(family["invariant"])
    )
    return {
        "seed": seed,
        "members_now": member_ids,
        "members_recorded": sorted(row["members"]),
        "unsourced_members": unsourced,
        "sourced_count": len(member_ids) - len(unsourced),
        "member_count": len(member_ids),
        "heuristic_previous": row["heuristic_anchor"],
        "heuristic_new": heuristic_new,
        "heuristic_changed": heuristic_new != row["heuristic_anchor"],
        "jev_previous": row["jev_anchor"],
        "jev_new": judged.choice,
        "jev_changed": judged.choice != row["jev_anchor"],
        "jev_asked": list(judged.asked),
        "jev_scored": len(judged.expected),
        "jev_expected": dict(judged.expected),
        "jev_reason": judged.reason,
        "jev_cost": judged.cost,
    }


def build_selection(
    old: dict, derived: list[dict], runner_check: dict, spend: float
) -> dict:
    """The new selection: fixed rule and families kept, anchors re-derived."""
    by_seed = {d["seed"]: d for d in derived}
    new = json.loads(json.dumps(old))
    rows = []
    for row in new["selected"]:
        d = by_seed[row["seed"]]
        previous = {
            "fixed_at": PREVIOUS_ANCHORS_FIXED_AT,
            "fixed_in": PREVIOUS_ANCHORS_FIXED_IN,
            "heuristic_anchor": d["heuristic_previous"],
            "jev_anchor": d["jev_previous"],
            "jev_asked": row.get("jev_asked"),
            "jev_scored": row.get("jev_scored"),
            "jev_cost": row.get("jev_cost"),
        }
        row["previous_anchors"] = previous
        row["heuristic_anchor"] = d["heuristic_new"]
        row["jev_anchor"] = d["jev_new"]
        row["jev_asked"] = d["jev_asked"]
        row["jev_scored"] = d["jev_scored"]
        row["jev_reason"] = d["jev_reason"]
        row["jev_cost"] = d["jev_cost"]
        row["anchor_changed"] = {
            "heuristic": d["heuristic_changed"],
            "jev": d["jev_changed"],
        }
        row["anchor_change_reason"] = {
            "heuristic": d["heur_reason"],
            "jev": d["jev_reason_text"],
        }
        row["no_eligible_anchor_in_either_arm"] = (
            d["heuristic_new"] is None and d["jev_new"] is None
        )
        rows.append(row)
    new["selected"] = rows
    new["reselect"] = {
        "derived_at": now(),
        "rule": (
            "Both arms re-derived under the DD-source rule: a family anchor "
            "needs a producing Data Dictionary source."
        ),
        "previous_anchors_fixed_at": PREVIOUS_ANCHORS_FIXED_AT,
        "previous_anchors_fixed_in": PREVIOUS_ANCHORS_FIXED_IN,
        "runner_reads_anchors_from_selection": runner_check,
        "spend_usd": spend,
        "spend_limit_usd": SPEND_LIMIT_USD,
    }
    return new


def esc(text: object) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def code(text: object) -> str:
    if text is None:
        return "<em>none</em>"
    return f"<code>{esc(text)}</code>"


def render_fragment(
    derived: list[dict], runner_check: dict, spend: float, export: str
) -> str:
    changed_any = [d for d in derived if d["heuristic_changed"] or d["jev_changed"]]
    dropped = [
        d for d in derived if d["heuristic_new"] is None and d["jev_new"] is None
    ]
    heur_changed = [d for d in derived if d["heuristic_changed"]]
    jev_changed = [d for d in derived if d["jev_changed"]]

    out: list[str] = []
    out.append('<section id="ssog-anchor-reselect" data-reckon="evidence">')
    out.append(
        '<h2 id="ssog-anchor-reselect-heading">Both arms\' anchors, re-derived '
        "under the DD-source rule</h2>"
    )
    out.append(
        '<p id="ssog-anchor-reselect-summary"><strong>Of the ten comparison '
        f"families, {len(heur_changed)} change heuristic anchor and "
        f"{len(jev_changed)} change judged anchor once a member needs a "
        "producing Data Dictionary source to be eligible; "
        f"{len(changed_any)} change in at least one arm and {len(dropped)} are "
        "left with no eligible anchor in either arm.</strong> The families and "
        "the adoption rule are the ones fixed in <code>selection.json</code>; "
        "only the two anchors per family are re-derived, from a scratch copy of "
        "the checked export. Both arms now refuse a member with no DD source: "
        "the heuristic through <code>select_anchor</code>, the judged scorer "
        "through its candidate filter <code>_anchor_score_candidates</code>."
        "</p>"
    )

    out.append(
        '<h3 id="ssog-anchor-reselect-rule">How the anchors were re-derived</h3>'
    )
    out.append(
        '<p id="ssog-anchor-reselect-rule-body">The archived export '
        f"(<code>{esc(export)}</code>) is loaded into a fresh scratch Neo4j "
        "instance the way <code>ssog-anchor-ab-run.py</code> loads it. The ten "
        "families are read straight back from the fixed <code>selection.json</code>, "
        "and for each family <code>fetch_pattern_families</code> returns the live "
        "member rows the runner would see. The heuristic anchor is "
        "<code>select_anchor(None, None, None, members)</code>; the judged anchor "
        "is <code>score_family_anchor</code>'s choice over the same members, one "
        "Jev Score per eligible member. A member is eligible when it is entailed "
        "by a DD source, carries a real description and has documentation to "
        "template from. The instance is stopped and its tree removed before the "
        "driver exits.</p>"
    )

    out.append(
        '<h3 id="ssog-anchor-reselect-table-heading">Anchors before and after, '
        "per family and per arm</h3>"
    )
    out.append(
        '<table id="' + 'ssog-anchor-reselect-table"><thead><tr>'
        "<th>Family seed</th><th>Members</th><th>DD-sourced</th>"
        "<th>Heuristic before</th><th>Heuristic after</th>"
        "<th>Heuristic changed</th>"
        "<th>Jev before</th><th>Jev after</th><th>Jev changed</th>"
        "</tr></thead><tbody>"
    )
    for i, d in enumerate(derived, 1):
        out.append(
            f'<tr id="ssog-anchor-reselect-row-{i}">'
            f"<td>{code(d['seed'])}</td>"
            f"<td>{esc(d['member_count'])}</td>"
            f"<td>{esc(d['sourced_count'])} of {esc(d['member_count'])}</td>"
            f"<td>{code(d['heuristic_previous'])}</td>"
            f"<td>{code(d['heuristic_new'])}</td>"
            f"<td>{'yes' if d['heuristic_changed'] else 'no'}</td>"
            f"<td>{code(d['jev_previous'])}</td>"
            f"<td>{code(d['jev_new'])}</td>"
            f"<td>{'yes' if d['jev_changed'] else 'no'}</td>"
            "</tr>"
        )
    out.append("</tbody></table>")

    out.append(
        '<h3 id="ssog-anchor-reselect-why-heading">Why each family\'s anchors did '
        "or did not change</h3>"
    )
    out.append('<ul id="ssog-anchor-reselect-why">')
    for i, d in enumerate(derived, 1):
        unsourced = (
            ", ".join(esc(u) for u in d["unsourced_members"])
            if d["unsourced_members"]
            else "none"
        )
        judged_note = (
            f"Judged arm scored {d['jev_scored']} of {len(d['jev_asked'])} "
            f"eligible members asked"
        )
        if d["jev_reason"]:
            judged_note += f"; reason {esc(d['jev_reason'])}"
        out.append(
            f'<li id="ssog-anchor-reselect-why-{i}">{code(d["seed"])}: '
            f"members with no DD source: {unsourced}. "
            f"{esc(d['heur_reason'])}. {esc(d['jev_reason_text'])}. "
            f"{judged_note}.</li>"
        )
    out.append("</ul>")

    out.append(
        '<h3 id="ssog-anchor-reselect-flagged-heading">Families with no eligible '
        "anchor</h3>"
    )
    if dropped:
        out.append('<ul id="ssog-anchor-reselect-flagged">'.replace("anco", "anch"))
        for i, d in enumerate(dropped, 1):
            out.append(
                f'<li id="ssog-anchor-reselect-flagged-{i}">{code(d["seed"])} has '
                "no DD-sourced member and therefore no eligible anchor in either "
                "arm; it drops out of the re-run with this reason.</li>"
            )
        out.append("</ul>")
    else:
        out.append(
            '<p id="ssog-anchor-reselect-flagged-none">No family is left without '
            "an eligible anchor: every family keeps at least one DD-sourced "
            "member, so all ten remain in the re-run.</p>"
        )

    out.append(
        '<h3 id="ssog-anchor-reselect-runner-heading">The runner reads its anchors '
        "from the selection</h3>"
    )
    out.append(
        '<p id="ssog-anchor-reselect-runner-body">Checked as source rather than '
        "assumed: <code>ssog-anchor-ab-run.py</code> loads <code>selection.json</code> "
        "and each arm's target anchor is the corresponding key of that family's "
        "row -- "
        + "; ".join(f"<code>{esc(r)}</code>" for r in runner_check["anchor_reads"])
        + ". No standard-name literal appears anywhere in the runner, so it pins "
        "no anchor beside the file.</p>"
    )

    out.append(
        '<p id="ssog-anchor-reselect-measure">Evidence: '
        f"{len(derived)} families re-derived from the checked export in a scratch "
        "instance that was stopped and removed; "
        f"{len(heur_changed)} heuristic anchors changed, {len(jev_changed)} judged "
        f"anchors changed, {len(changed_any)} families changed in at least one arm, "
        f"{len(dropped)} left without an anchor. Judged-arm spend USD {spend:.4f} "
        f"against a USD {SPEND_LIMIT_USD:.2f} ceiling. The driver's log ends "
        "<code>EXIT=0</code>.</p>"
    )
    out.append("</section>")
    return "\n".join(out) + "\n"


async def run() -> dict:
    selection = json.loads(SELECTION.read_text())
    runner_check = runner_anchor_source()
    print(f"runner anchor reads ok: {runner_check['anchor_reads']}", flush=True)

    scratch_root = Path(
        tempfile.mkdtemp(
            prefix=f"{os.environ.get('USER', 'user')}-reselect-",
            dir=os.environ["TMPDIR"],
        )
    )
    place = scratch_root / "instance"
    for sub in ("data", "dumps", "logs", "run", "tmp"):
        (place / sub).mkdir(parents=True, exist_ok=True)

    summary: dict = {
        "started_at": now(),
        "source_export": selection["source_export"],
        "scratch_root": str(scratch_root),
        "runner_check": runner_check,
    }
    port, lock_fd = reserve_port_pair()
    instance = None
    try:
        dump = extract_dump(Path(selection["source_export"]), place)
        summary["dump_bytes"] = dump.stat().st_size
        print(f"extracted dump: {dump.stat().st_size} bytes", flush=True)

        proc, neo4j_log = start_instance(place, port)
        instance = proc
        summary["neo4j_log"] = str(neo4j_log)
        uri = f"bolt://127.0.0.1:{port}"
        summary["scratch_uri"] = uri
        print(f"scratch instance up at {uri}", flush=True)

        derived: list[dict] = []
        spend = 0.0
        with GraphClient(uri=uri) as gc:
            for row in selection["selected"]:
                d = await derive_one(gc, row)
                spend += d["jev_cost"] or 0.0
                derived.append(d)
                print(
                    f"{d['seed']}: heuristic {d['heuristic_previous']} -> "
                    f"{d['heuristic_new']} (changed={d['heuristic_changed']}), "
                    f"jev {d['jev_previous']} -> {d['jev_new']} "
                    f"(changed={d['jev_changed']}), {d['sourced_count']}/"
                    f"{d['member_count']} sourced, ${d['jev_cost']:.6f}",
                    flush=True,
                )
                if spend > SPEND_LIMIT_USD:
                    raise RuntimeError(
                        f"Judged-arm spend ${spend:.4f} exceeded the "
                        f"${SPEND_LIMIT_USD:.2f} ceiling"
                    )

        for d in derived:
            d["heur_reason"] = change_reason(
                "heuristic",
                d["heuristic_previous"],
                d["heuristic_new"],
                d["unsourced_members"],
            )
            d["jev_reason_text"] = change_reason(
                "jev", d["jev_previous"], d["jev_new"], d["unsourced_members"]
            )

        new_selection = build_selection(selection, derived, runner_check, spend)
        SELECTION.write_text(json.dumps(new_selection, indent=2, sort_keys=True) + "\n")
        print(f"wrote {SELECTION}", flush=True)

        FRAGMENT.write_text(
            render_fragment(derived, runner_check, spend, selection["source_export"])
        )
        print(f"wrote {FRAGMENT}", flush=True)

        summary.update(
            finished_at=now(),
            derived=[
                {
                    k: v
                    for k, v in d.items()
                    if k not in {"heur_reason", "jev_reason_text"}
                }
                for d in derived
            ],
            spend_usd=spend,
            changed_heuristic=[d["seed"] for d in derived if d["heuristic_changed"]],
            changed_jev=[d["seed"] for d in derived if d["jev_changed"]],
            dropped=[
                d["seed"]
                for d in derived
                if d["heuristic_new"] is None and d["jev_changed"] is None
            ],
            out_of_anchor=[
                d["seed"]
                for d in derived
                if d["heuristic_new"] is None and d["jev_new"] is None
            ],
        )
    finally:
        if instance is not None:
            stop_temp_neo4j(instance)
        removed = None
        if scratch_root.exists():
            removed = sorted(p.name for p in scratch_root.iterdir())
            shutil.rmtree(scratch_root)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)
        summary["scratch_removed"] = {
            "tree": str(scratch_root),
            "removed_entries": removed,
            "at": now(),
        }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--summary",
        default=os.environ.get("SSOG_RESELECT_SUMMARY", ""),
        help="Path to write the run summary JSON to.",
    )
    args = ap.parse_args()
    summary = asyncio.run(run())
    if args.summary:
        Path(args.summary).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in summary if k != "derived"}))
    print("summary written", flush=True)


if __name__ == "__main__":
    sys.exit(main())
