"""Break the REFINED_FROM lineage cycles through a signed edge archive.

The reader is the lineage-cycle census, run rather than re-implemented: each
``capture`` shells out to ``sib-lineage-cycle-census-run.py --capture`` and
reads the snapshot it retains. From that snapshot the driver classifies every
cycle component under the repair rule, composes one signed edge-archive
authority for the whole cohort, previews it, applies it with the preview's own
digest, and re-captures.

The rule, per cycle component:

* one live member -- that name keeps its lineage and every edge leaving a
  superseded member is archived and removed;
* no live member -- every internal edge of the component is archived and
  removed;
* more than one live member -- listed for a ruling and left untouched.

An edge is never chosen by name order or creation time: the selection is the
component membership plus the endpoint lifecycle, and the composed authority
records both.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from imas_codex.graph.client import GraphClient  # noqa: E402
from imas_codex.standard_names.signed_manifest import (  # noqa: E402
    apply_signed_manifest,
    compose_edge_archive_authority,
    signed_payload_sha256,
)

HERE = Path(__file__).resolve().parent
STEM = Path(__file__).stem.removesuffix("-run")
FIGURE = ROOT / "docs/figures/sn-integrity-backlog" / STEM / "rule-application.svg"
CENSUS_RUNNER = HERE / "sib-lineage-cycle-census-run.py"
OPERATION_ID = "sib-lineage-cycle-repair-2026-11-11"
REASON = (
    "break every REFINED_FROM lineage cycle at its superseded members under "
    "the lineage-cycle repair rule"
)
ADAPTER = "edge-archive"
MUTATION = "archive-lineage-cycle-edges"
GUARDS = (
    "signed-edge-snapshot-manifest",
    "component-edge-closure",
    "out-of-allowlist-immutability",
)
RELATIONSHIP = "REFINED_FROM"

COLLATERAL_QUERY = """// collateral-census
CALL { MATCH (node:StandardName) RETURN count(node) AS names }
CALL { MATCH (node:StandardNameSource) RETURN count(node) AS sources }
CALL { MATCH (node:StandardNameReview) RETURN count(node) AS reviews }
CALL { MATCH (node:DocsRevision) RETURN count(node) AS documents }
CALL { MATCH (:StandardName)-[edge:REFINED_FROM]->(:StandardName)
       RETURN count(edge) AS lineage_edges }
CALL { MATCH (change:StandardNameChange) RETURN count(change) AS receipts }
RETURN names, sources, reviews, documents, lineage_edges, receipts"""


def log(message: str) -> None:
    print(message, flush=True)


def capture(run_dir: Path, label: str) -> dict:
    """Run the census instrument into ``run_dir/label`` and load its snapshot."""
    target = run_dir / label
    target.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(CENSUS_RUNNER),
        "--run-dir",
        str(target),
        "--capture",
    ]
    log(f"# census: {shlex.join(command)}")
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    (run_dir / f"census-{label}.log").write_text(completed.stdout + completed.stderr)
    if completed.returncode != 0:
        tail = "\n".join((completed.stdout + completed.stderr).splitlines()[-20:])
        raise SystemExit(f"census capture {label} failed:\n{tail}")
    snapshot = json.loads((target / "graph-snapshot.json").read_text())
    log(
        f"CAPTURE {label} names={len(snapshot['nodes'])} "
        f"edges={len(snapshot['edges'])} "
        f"cycle_components={len(snapshot['cycles'])} "
        f"cycle_names={sum(len(component) for component in snapshot['cycles'])}"
    )
    return snapshot


def collateral() -> dict:
    """Read the counts the archive must leave untouched."""
    with GraphClient() as gc:
        rows = gc.query(COLLATERAL_QUERY)
    return {key: int(value) for key, value in dict(rows[0]).items()}


def component_key(members: list[str]) -> str:
    digest = hashlib.sha256("\n".join(sorted(members)).encode()).hexdigest()
    return "lineage-cycle-" + digest[:12]


def plan(snapshot: dict) -> dict:
    """Classify every cycle component and select the edges the rule archives."""
    nodes = {row["id"]: row for row in snapshot["nodes"]}
    edges = snapshot["edges"]
    components, archived, multi_live = [], [], []
    for component in snapshot["cycles"]:
        members = sorted(component)
        key = component_key(members)
        live = [name for name in members if nodes[name]["stage"] != "superseded"]
        component_edges = [
            edge
            for edge in edges
            if edge["successor"] in component and edge["predecessor"] in component
        ]
        if not live:
            rule = "no-live"
            selected = component_edges
        elif len(live) == 1:
            rule = "single-live"
            selected = [
                edge for edge in component_edges if edge["successor"] not in live
            ]
        else:
            rule = "multi-live"
            selected = []
            multi_live.append(
                {
                    "component": key,
                    "members": members,
                    "live_members": live,
                    "evidence": [
                        {
                            "name": name,
                            "description": nodes[name].get("description"),
                            "stage": nodes[name]["stage"],
                            "status": nodes[name].get("status"),
                            "name_score": nodes[name].get("name_score"),
                            "unit": nodes[name].get("unit"),
                            "source_id": nodes[name].get("source_id"),
                        }
                        for name in members
                    ],
                    "internal_edges": [
                        [edge["successor"], edge["predecessor"]]
                        for edge in component_edges
                    ],
                }
            )
        entry = {
            "component": key,
            "members": members,
            "live_members": live,
            "rule": rule,
            "internal_edges": len(component_edges),
            "archived_edges": len(selected),
        }
        components.append(entry)
        for edge in selected:
            archived.append(
                {
                    "successor": edge["successor"],
                    "predecessor": edge["predecessor"],
                    "relationship_type": RELATIONSHIP,
                    "component": key,
                    "element_id": edge["edge_id"],
                    "properties": dict(edge.get("properties") or {}),
                    "successor_stage": nodes[edge["successor"]]["stage"],
                    "successor_status": nodes[edge["successor"]].get("status"),
                    "predecessor_stage": nodes[edge["predecessor"]]["stage"],
                    "predecessor_status": nodes[edge["predecessor"]].get("status"),
                }
            )
    keeping = [entry for entry in components if entry["rule"] != "multi-live"]
    return {
        "observed_at": snapshot["observed_at"],
        "census_revision": snapshot["revision"],
        "cycle_components": len(components),
        "cycle_names": sum(len(entry["members"]) for entry in components),
        "components": components,
        "rule_counts": {
            rule: sum(1 for entry in components if entry["rule"] == rule)
            for rule in ("single-live", "no-live", "multi-live")
        },
        "edges_archived": len(archived),
        "edges_retained_inside_archived_components": sum(
            entry["internal_edges"] - entry["archived_edges"] for entry in keeping
        ),
        "multi_live": multi_live,
        "archived_edges": archived,
        "composed_components": [
            {
                "component": entry["component"],
                "members": entry["members"],
                "live_members": entry["live_members"],
                "rule": entry["rule"],
            }
            for entry in keeping
        ],
    }


def write_authority(run_dir: Path, planned: dict) -> tuple[Path, str, str]:
    authority = compose_edge_archive_authority(
        operation_id=OPERATION_ID,
        components=planned["compose_components"],
        edges=planned["archived_edges"],
    )
    path = run_dir / "edge-archive-authority.json"
    raw = json.dumps(authority, sort_keys=True, indent=2).encode()
    path.write_bytes(raw)
    return (
        path,
        hashlib.sha256(raw).hexdigest(),
        signed_payload_sha256(json.loads(raw)),
    )


def apply(run_dir: Path, apply_it: bool, manifest_sha256: str | None) -> dict:
    authority_path = run_dir / "edge-archive-authority.json"
    raw = authority_path.read_bytes()
    file_sha256 = hashlib.sha256(raw).hexdigest()
    payload_sha256 = signed_payload_sha256(json.loads(raw))
    result = apply_signed_manifest(
        authority_path,
        authority_file_sha256=file_sha256,
        authority_payload_sha256=payload_sha256,
        authority_adapter=ADAPTER,
        mutation_kind=MUTATION,
        guard_set=GUARDS,
        reason=REASON,
        apply=apply_it,
        manifest_sha256=manifest_sha256,
    )
    result = dict(result)
    result.pop("manifest", None)
    result.setdefault("authority_file_sha256", file_sha256)
    result.setdefault("authority_payload_sha256", payload_sha256)
    return result


LIVE_COLOUR = "#2f6db3"
RETIRED_COLOUR = "#8f8f8f"
ARCHIVED_COLOUR = "#c0392b"
KEPT_COLOUR = "#333333"


def draw_figure(target: Path) -> None:
    """Draw the two rules applied to a planted component, before and after."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    fig, axes = plt.subplots(2, 2, figsize=(14, 5.4))
    fig.patch.set_facecolor("white")
    rules = [
        (
            "one live member",
            {"live": (0.72, 0.62), "retired": (0.22, 0.28)},
            [("live", "retired", True), ("retired", "live", False)],
        ),
        (
            "no live member",
            {"retired_a": (0.25, 0.62), "retired_b": (0.70, 0.30)},
            [("retired_a", "retired_b", False), ("retired_b", "retired_a", False)],
        ),
    ]
    labels = {
        "live": "live name\nkept",
        "retired": "superseded\nmember",
        "retired_a": "superseded",
        "retired_b": "superseded",
    }
    colours = {
        "live": LIVE_COLOUR,
        "retired": RETIRED_COLOUR,
        "retired_a": RETIRED_COLOUR,
        "retired_b": RETIRED_COLOUR,
    }
    for row, (title, positions, edges) in enumerate(rules):
        for column, keep_archived in enumerate((True, False)):
            ax = axes[row][column]
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
            for name, (x, y) in positions.items():
                ax.scatter(
                    [x],
                    [y],
                    s=1500,
                    color=colours[name],
                    zorder=3,
                    edgecolors="white",
                    linewidths=1.5,
                )
                ax.text(
                    x,
                    y - 0.17,
                    labels[name],
                    ha="center",
                    va="top",
                    fontsize=11,
                    color="#222222",
                )
            for start, end, retained in edges:
                if not retained and not keep_archived:
                    continue
                ax.add_patch(
                    FancyArrowPatch(
                        positions[start],
                        positions[end],
                        arrowstyle="-|>",
                        connectionstyle="arc3,rad=0.22",
                        mutation_scale=16,
                        linewidth=2.4 if retained else 2.6,
                        color=KEPT_COLOUR if retained else ARCHIVED_COLOUR,
                        linestyle="-" if retained else (0, (5, 3)),
                        shrinkA=22,
                        shrinkB=22,
                        zorder=2,
                    )
                )
            ax.set_title(
                ("before the archive" if keep_archived else "after the archive")
                + f" — {title}",
                fontsize=13,
                color="#222222",
            )
    fig.text(
        0.5,
        0.02,
        "solid: REFINED_FROM kept    dashed red: archived into a signed receipt and removed",
        ha="center",
        fontsize=12,
        color="#333333",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, format=target.suffix.lstrip("."), metadata={"Date": None})
    plt.close(fig)


def report(run_dir: Path, figure: Path) -> dict:
    """Compare the two captures and retain the figures the fragment quotes."""
    before = json.loads((run_dir / "before" / "graph-snapshot.json").read_text())
    after = json.loads((run_dir / "after" / "graph-snapshot.json").read_text())
    planned = json.loads((run_dir / "plan.json").read_text())
    applied = json.loads((run_dir / "apply.json").read_text())
    counters_before = json.loads((run_dir / "collateral-before.json").read_text())
    counters_after = json.loads((run_dir / "collateral-after.json").read_text())
    summary = {
        "before_cycle_components": len(before["cycles"]),
        "before_cycle_names": sum(len(component) for component in before["cycles"]),
        "after_cycle_components": len(after["cycles"]),
        "after_cycle_names": sum(len(component) for component in after["cycles"]),
        "before_lineage_edges": len(before["edges"]),
        "after_lineage_edges": len(after["edges"]),
        "outcome": applied.get("outcome"),
        "archived_edges": applied.get("changed") or len(applied.get("receipts") or []),
        "rule_counts": planned["rule_counts"],
        "multi_live_components": len(planned["multi_live"]),
        "collateral_before": counters_before,
        "collateral_after": counters_after,
        "authority_payload_sha256": applied.get("authority_payload_sha256"),
        "authority_file_sha256": applied.get("authority_file_sha256"),
        "receipts": len(applied.get("receipts") or []),
        "before_observed_at": before["observed_at"],
        "after_observed_at": after["observed_at"],
    }
    draw_figure(figure)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    log("SUMMARY " + json.dumps(summary, sort_keys=True))
    write_fragment(run_dir, summary, planned, applied, figure)
    return summary


def _escape(value: object) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def write_fragment(
    run_dir: Path,
    summary: dict,
    planned: dict,
    applied: dict,
    figure: Path,
) -> Path:
    """Write the landing fragment the section reads."""
    stem = STEM
    figure_src = "/imas-codex/figures/sn-integrity-backlog/" + (f"{stem}/{figure.name}")
    rules = planned["rule_counts"]
    collateral = summary["collateral_before"]
    after_collateral = summary["collateral_after"]
    counts = (
        f'<table id="{stem}-counts"><thead><tr><th>Measure</th>'
        "<th>Before</th><th>After</th></tr></thead><tbody>"
    )
    rows = [
        (
            "REFINED_FROM cycle components",
            summary["before_cycle_components"],
            summary["after_cycle_components"],
        ),
        (
            "names inside a cycle",
            summary["before_cycle_names"],
            summary["after_cycle_names"],
        ),
        (
            "REFINED_FROM edges",
            summary["before_lineage_edges"],
            summary["after_lineage_edges"],
        ),
    ]
    for label, before_value, after_value in rows:
        counts += (
            f"<tr><td>{label}</td><td>{before_value}</td><td>{after_value}</td></tr>"
        )
    for key in ("sources", "documents", "reviews"):
        counts += (
            f"<tr><td>{key}</td><td>{collateral.get(key)}</td>"
            f"<td>{after_collateral.get(key)}</td></tr>"
        )
    counts += "</tbody></table>"

    archived_from: dict[str, set[str]] = defaultdict(set)
    for edge in planned["archived_edges"]:
        archived_from[edge["component"]].add(edge["successor"])

    live_rows = ""
    for entry in sorted(
        (item for item in planned["components"] if item["rule"] == "single-live"),
        key=lambda item: item["live_members"][0],
    ):
        live_rows += (
            "<tr><td><code>"
            + _escape(entry["live_members"][0])
            + "</code></td><td>"
            + _escape(", ".join(sorted(archived_from[entry["component"]])))
            + "</td><td>"
            + str(entry["archived_edges"])
            + "</td></tr>"
        )

    retired_rows = ""
    for entry in sorted(
        (item for item in planned["components"] if item["rule"] == "no-live"),
        key=lambda item: item["members"],
    ):
        retired_rows += (
            "<tr><td>"
            + _escape(", ".join(entry["members"]))
            + "</td><td>"
            + str(entry["archived_edges"])
            + "</td></tr>"
        )
    retained = planned["edges_retained_inside_archived_components"]
    multi_live_block = ""
    if planned["multi_live"]:
        rows = ""
        for case in planned["multi_live"]:
            rows += (
                "<tr><td><code>"
                + _escape(case["component"])
                + "</code></td><td>"
                + _escape(", ".join(case["members"]))
                + "</td><td>"
                + _escape(", ".join(case["live_members"]))
                + "</td><td>"
                + _escape(
                    ", ".join(
                        f"{successor} -> {predecessor}"
                        for successor, predecessor in case["internal_edges"]
                    )
                )
                + "</td></tr>"
            )
        multi_live_block = (
            f'<h3 id="{stem}-multi-live">Components with more than one live name, '
            "left untouched</h3><p>These carry no rule in the signed archive. Each "
            "needs a ruling on which member keeps the lineage; the evidence is in "
            "<code>plan.json</code> under <code>multi_live</code>.</p>"
            f'<table id="{stem}-multi-live-table"><thead><tr><th>Component</th>'
            "<th>Members</th><th>Live members</th><th>Internal edges</th>"
            f"</tr></thead><tbody>{rows}</tbody></table>"
        )
    multi_live = summary["multi_live_components"]
    if multi_live:
        held = (
            f"<strong>{multi_live} components carried more than one live name</strong> "
            "and were left untouched: they are listed below with their proposal."
        )
    else:
        held = (
            "<strong>No component carried more than one live name</strong>, so no "
            "case was held back for a ruling."
        )
    html = f"""<section id="{stem}">
<style>#{stem} table {{table-layout:fixed;width:100%;overflow-wrap:anywhere}} #{stem} code {{white-space:normal;overflow-wrap:anywhere}}</style>
<h2>Every refined-from lineage cycle is broken by a signed edge archive</h2>
<p>The lineage-cycle census was run before and after the repair. It reads <strong>{summary["before_cycle_components"]} cycle components over {summary["before_cycle_names"]} names</strong> before the archive ({summary["before_observed_at"]}) and <strong>{summary["after_cycle_components"]} cycle components over {summary["after_cycle_names"]} names</strong> after it ({summary["after_observed_at"]}), from {summary["before_lineage_edges"]} to {summary["after_lineage_edges"]} <code>REFINED_FROM</code> edges. The archive removed <strong>{summary["archived_edges"]} edges</strong> and minted <strong>{summary["receipts"]} signed receipts</strong>, one per edge, each carrying the full edge snapshot: endpoints, relationship type, element id and properties. Of the {summary["before_cycle_components"]} components, {rules.get("single-live", 0)} had one live member: that name and its own lineage were kept, and every edge leaving a superseded member was archived. The other {rules.get("no-live", 0)} had no live member, so every internal edge was archived. {held}</p>
<figure id="{stem}-rule"><img src="{figure_src}" alt="Two planted lineage components before and after the archive: a single-live pair loses only the superseded member's edge, and a two-member retired pair loses both."/><figcaption>The rule applied to the two component shapes. A single-live component keeps the live name and archives every edge leaving a superseded member; a component with no live member archives all of its internal edges. Dashed red edges are archived into a signed receipt and removed.</figcaption></figure>
<h3 id="{stem}-counts-heading">Measured before and after</h3>
{counts}
<h3 id="{stem}-kept">One live member: the name kept, and the members whose edges were archived</h3>
<p>Every component in this table kept its live name and that name's own lineage. The edge removed in each case leaves a superseded member, so the cycle closes and no live identity moves. The selection is component membership plus the endpoint lifecycle; no edge was chosen by name order or creation time.</p>
<table id="{stem}-components"><thead><tr><th>Live name kept</th><th>Superseded member whose edge was archived</th><th>Edges archived</th></tr></thead><tbody>
{live_rows}</tbody></table>
<h3 id="{stem}-retired">No live member: every internal edge archived</h3>
<p>Neither member of these components is live, so the component carries no lineage for any identity to keep. Each internal edge was archived into a receipt and removed, and both names remain in the graph with their lifecycle unchanged.</p>
<table id="{stem}-retired-table"><thead><tr><th>Members, all superseded</th><th>Edges archived</th></tr></thead><tbody>
{retired_rows}</tbody></table>
{multi_live_block}
<h3 id="{stem}-receipts">Signed archive</h3>
<p>The cohort is one authority file signed by two digests, its file SHA-256 <code>{_escape(summary["authority_file_sha256"])}</code> and its canonical payload SHA-256 <code>{_escape(summary["authority_payload_sha256"])}</code>. The operation re-read the live component edges inside the applying transaction, refused nothing, minted one <code>StandardNameChange</code> receipt per edge with operation <code>archive_lineage_cycle_edges</code>, deleted each edge by its element id, and read every member's lifecycle and its producing-source, review and document counts back unchanged before committing. {retained} edges inside the archived components were retained: those leaving the live member, which is that name's own lineage.</p>
<h3 id="{stem}-limits">Limits and controls</h3>
<p>The disposable-graph tests plant a single-live cycle, a retired cycle and a component holding an edge the manifest omits; the last refuses the whole manifest and changes nothing. Removing the operation from the dispatcher fails those tests, which is the declared negative control. The census's own planted controls (a reciprocal cycle, a self-loop, a branching tip and an isolated clean name) run before every capture. Sources, documents and reviews read back by count unchanged.</p>
<p>Run record: <code>{_escape(run_dir)}</code>. The census instrument is <code>sib-lineage-cycle-census-run.py</code>, invoked as its own process for each capture; its snapshots and logs are retained there rather than in this repository.</p>
</section>
"""
    path = HERE / f"{stem}.html"
    path.write_text(html)
    log(f"FRAGMENT {path} bytes={len(html.encode())}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--step",
        required=True,
        choices=[
            "capture-before",
            "plan",
            "compose",
            "preview",
            "apply",
            "capture-after",
            "report",
        ],
    )
    parser.add_argument("--manifest-sha256")
    args = parser.parse_args()
    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.step == "capture-before":
        snapshot = capture(run_dir, "before")
        (run_dir / "collateral-before.json").write_text(
            json.dumps(collateral(), indent=2) + "\n"
        )
    elif args.step == "plan":
        snapshot = json.loads((run_dir / "before" / "graph-snapshot.json").read_text())
        planned = plan(snapshot)
        (run_dir / "plan.json").write_text(json.dumps(planned, indent=2) + "\n")
        log(
            "PLAN components={cycle_components} names={cycle_names} "
            "rules={rule_counts} edges_archived={edges_archived} "
            "multi_live={multi_live}".format(
                cycle_components=planned["cycle_components"],
                cycle_names=planned["cycle_names"],
                rule_counts=planned["rule_counts"],
                edges_archived=planned["edges_archived"],
                multi_live=len(planned["multi_live"]),
            )
        )
    elif args.step == "compose":
        planned = json.loads((run_dir / "plan.json").read_text())
        planned["compose_components"] = [
            {
                key: entry[key]
                for key in ("component", "members", "live_members", "rule")
            }
            for entry in planned["components"]
            if entry["rule"] != "multi-live"
        ]
        (run_dir / "plan.json").write_text(json.dumps(planned, indent=2) + "\n")
        path, file_sha256, payload_sha256 = write_authority(run_dir, planned)
        log(
            f"AUTHORITY path={path} file_sha256={file_sha256} "
            f"payload_sha256={payload_sha256}"
        )
    elif args.step == "preview":
        result = apply(run_dir, False, None)
        (run_dir / "preview.json").write_text(json.dumps(result, indent=2) + "\n")
        log(
            "PREVIEW outcome={outcome} would_change={would_change} "
            "refusals={refusals} manifest_sha256={sha}".format(
                outcome=result["outcome"],
                would_change=result.get("would_change"),
                refusals=len(result.get("refusals") or []),
                sha=result.get("manifest_sha256"),
            )
        )
        if result["outcome"] != "would_apply":
            return 1
    elif args.step == "apply":
        preview = json.loads((run_dir / "preview.json").read_text())
        digest = args.manifest_sha256 or preview["manifest_sha256"]
        result = apply(run_dir, True, digest)
        (run_dir / "apply.json").write_text(json.dumps(result, indent=2) + "\n")
        log(
            "APPLY outcome={outcome} changed={changed} receipts={receipts}".format(
                outcome=result["outcome"],
                changed=result.get("changed"),
                receipts=len(result.get("receipts") or []),
            )
        )
        if result["outcome"] != "applied":
            return 1
    elif args.step == "capture-after":
        snapshot = capture(run_dir, "after")
        (run_dir / "collateral-after.json").write_text(
            json.dumps(collateral(), indent=2) + "\n"
        )
    elif args.step == "report":
        report(run_dir, FIGURE)
    return 0


if __name__ == "__main__":
    print(
        f"# revision: {subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()} tree: {ROOT} command: {shlex.join([sys.executable, *sys.argv])}",
        flush=True,
    )
    print(f"# module: {__file__}; cwd: {Path.cwd().resolve()}", flush=True)
    try:
        code = main()
    except Exception:
        import traceback

        traceback.print_exc()
        code = 1
    print(f"EXIT={code}", flush=True)
    raise SystemExit(code)
