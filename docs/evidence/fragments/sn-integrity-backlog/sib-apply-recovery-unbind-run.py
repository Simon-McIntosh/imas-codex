"""Remove signed recovery bindings after checking their live graph closure."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any

from imas_codex.graph import GraphClient
from imas_codex.graph.offsite import live_graph_activity
from imas_codex.standard_names.graph_ops import classify_sourceless_standard_names
from imas_codex.standard_names.signed_manifest import apply_signed_manifest

HERE = Path(__file__).resolve().parent
INPUT = HERE / "sib-restore-lineage-sources.json"
AUTHORITY = Path(__file__).with_name("sib-apply-recovery-unbind-authority.json")
RECORD = Path(__file__).with_name("sib-apply-recovery-unbind.json")
FRAGMENT = Path(__file__).with_name("sib-apply-recovery-unbind.html")
RUN_DIR = Path(os.environ["RECOVERY_UNBIND_RUN_DIR"])
BEFORE_SNAPSHOTS = RUN_DIR / "recovery-unbind-before-snapshots.json"
AFTER_SNAPSHOTS = RUN_DIR / "recovery-unbind-after-snapshots.json"
REASON = "Remove recovery bindings below the 0.8 choice-probability or 0.5 confidence floor, as ruled by the lead."
GUARDS = ("exhausted-target-live-source", "out-of-allowlist-immutability")


def save(record: dict[str, Any]) -> None:
    record["recorded_at"] = datetime.now(UTC).isoformat()
    RECORD.write_text(json.dumps(record, indent=2, sort_keys=True, default=str) + "\n")


def read() -> dict[str, Any]:
    return json.loads(RECORD.read_text())


def query(graph: GraphClient, cypher: str, **params: Any) -> list[dict[str, Any]]:
    started = time.monotonic()
    rows = graph.query(cypher, **params)
    elapsed = time.monotonic() - started
    if elapsed >= 10:
        raise RuntimeError(f"bounded graph query exceeded ten seconds ({elapsed:.2f})")
    return rows


def snapshot(graph: GraphClient, name_id: str, source_id: str) -> dict[str, Any]:
    participants = query(
        graph,
        """OPTIONAL MATCH (name:StandardName {id: $name_id})
        OPTIONAL MATCH (source:StandardNameSource {id: $source_id})
        OPTIONAL MATCH (source)-[binding:PRODUCED_NAME]->(name)
        RETURN properties(name) AS name, properties(source) AS source,
               collect(elementId(binding)) AS binding_ids""",
        name_id=name_id,
        source_id=source_id,
    )
    if len(participants) != 1:
        raise RuntimeError(
            f"participant read returned {len(participants)} rows for {name_id}"
        )
    state = participants[0]
    for label, identifier in (
        ("StandardName", name_id),
        ("StandardNameSource", source_id),
    ):
        state[label + "_edges"] = query(
            graph,
            f"""MATCH (node:{label} {{id: $identifier}})-[edge]-(other)
            RETURN elementId(edge) AS edge_id, type(edge) AS kind,
                   elementId(other) AS other_element_id, other.id AS other_id,
                   labels(other) AS other_labels,
                   properties(edge) AS properties, startNode(edge) = node AS outgoing
            ORDER BY edge_id LIMIT 500""",
            identifier=identifier,
        )
    return json.loads(json.dumps(state, sort_keys=True, default=str))


def classifier(graph: GraphClient) -> dict[str, Any]:
    rows = classify_sourceless_standard_names(graph)
    names = sorted(str(row["id"]) for row in rows)
    if len(names) != len(set(names)):
        raise RuntimeError("source classifier returned duplicate names")
    return {
        "count": len(rows),
        "counts_by_kind": dict(Counter(str(row["population"]) for row in rows)),
        "names": names,
    }


def authority_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name_id": row["name"],
            "source_id": row["source_id"],
            "reason": "A recovery binding was made below the later approval floor; the lead ruled to unbind it.",
            "lineage_evidence": {
                "bound_dd_path": row["candidate"],
                "choice_probability": row["probability"],
                "confidence": row["confidence"],
                "original_reason": row["original_reason"],
            },
        }
        for row in rows
    ]


def prepare() -> None:
    recorded = json.loads(INPUT.read_text())["bound_below_confidence_floor"]
    if len(recorded) != 16 or len({row["name"] for row in recorded}) != 16:
        raise RuntimeError("the recorded recovery cohort is not sixteen distinct names")
    result: dict[str, Any] = {
        "input": str(INPUT),
        "rows": [],
        "authority_schema": "imas-codex.recovery-binding-unbind.v1",
    }
    snapshots: dict[str, Any] = {}
    with GraphClient() as graph:
        baseline = classifier(graph)
        result["classifier_before"] = baseline
        for original in recorded:
            name_id = original["name"]
            source_id = "dd:" + original["candidate"]
            before = snapshot(graph, name_id, source_id)
            snapshots[name_id] = before
            reasons = []
            if not before["name"]:
                reasons.append("name absent")
            elif before["name"].get("name_stage") != "exhausted":
                reasons.append("name is not exhausted")
            if not before["source"]:
                reasons.append("source absent")
            elif before["source"].get("status") == "stale":
                reasons.append("source is stale")
            if len(before["binding_ids"]) != 1 or before["binding_ids"][0] is None:
                reasons.append("recorded binding absent or non-unique")
            result["rows"].append(
                {
                    "name": name_id,
                    "candidate": original["candidate"],
                    "source_id": source_id,
                    "probability": original["probability"],
                    "confidence": original["confidence"],
                    "original_reason": original["reason"],
                    "match": {
                        "binding_ids": before["binding_ids"],
                        "source_status": None
                        if before["source"] is None
                        else before["source"].get("status"),
                        "name_stage": None
                        if before["name"] is None
                        else before["name"].get("name_stage"),
                        "reasons": reasons,
                    },
                    "action": "leave_alone" if reasons else "unbind",
                    "before_snapshot": str(BEFORE_SNAPSHOTS),
                }
            )
    BEFORE_SNAPSHOTS.write_text(json.dumps(snapshots, indent=2, sort_keys=True) + "\n")
    matched = [row for row in result["rows"] if row["action"] == "unbind"]
    if matched:
        rows = authority_rows(matched)
        canonical = (
            json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            + "\n"
        )
        authority = {
            "schema": "imas-codex.recovery-binding-unbind.v1",
            "rows": rows,
            "signature": {
                "algorithm": "sha256",
                "canonicalization": "jq -cS '.rows'",
                "scope": "rows",
                "digest": hashlib.sha256(canonical.encode()).hexdigest(),
            },
        }
        AUTHORITY.write_text(json.dumps(authority, indent=2, ensure_ascii=False) + "\n")
        result["authority_path"] = str(AUTHORITY)
        result["authority_file_sha256"] = hashlib.sha256(
            AUTHORITY.read_bytes()
        ).hexdigest()
    result["matched_count"] = len(matched)
    result["left_alone_count"] = 16 - len(matched)
    save(result)
    print(
        json.dumps(
            {
                "matched": len(matched),
                "left_alone": 16 - len(matched),
                "classifier_before": baseline["count"],
            },
            sort_keys=True,
        )
    )


def run_writer(*, apply: bool, manifest_sha256: str | None = None) -> dict[str, Any]:
    record = read()
    names = [row["name"] for row in record["rows"] if row["action"] == "unbind"]
    if not names:
        raise RuntimeError("no matching rows to sign")
    return apply_signed_manifest(
        AUTHORITY,
        authority_adapter="recovery-binding-unbind",
        mutation_kind="detach",
        guard_set=GUARDS,
        name_ids=names,
        reason=REASON,
        apply=apply,
        manifest_sha256=manifest_sha256,
        run_id="recovery-unbind",
    )


def preview() -> None:
    record = read()
    outcome = run_writer(apply=False)
    record["preview"] = outcome
    save(record)
    if (
        outcome["outcome"] != "would_apply"
        or outcome["would_change"] != record["matched_count"]
    ):
        raise RuntimeError(
            f"writer preview refused or omitted rows: {outcome['outcome']}"
        )
    print(
        json.dumps(
            {
                "outcome": outcome["outcome"],
                "would_change": outcome["would_change"],
                "manifest_sha256": outcome["manifest_sha256"],
            },
            sort_keys=True,
        )
    )


def apply() -> None:
    record = read()
    expected = record["matched_count"]
    if record.get("preview", {}).get("would_change") != expected:
        raise RuntimeError("no complete writer preview is recorded")
    activity = live_graph_activity(window_seconds=6.0, interval_seconds=3.0)
    quiet = {
        "window_seconds": activity.window_seconds,
        "foreign_transactions_started": activity.transactions_started,
        "foreign_open_queries": activity.open_queries,
        "quiet": not activity.busy,
        "checked_at": datetime.now(UTC).isoformat(),
    }
    if activity.busy:
        record["quiet_graph"] = quiet
        save(record)
        raise RuntimeError("graph is not quiet; refusing recovery unbind")
    receipt = run_writer(
        apply=True, manifest_sha256=record["preview"]["manifest_sha256"]
    )
    record["quiet_graph"] = quiet
    record["receipt"] = receipt
    save(record)
    if receipt.get("outcome") != "applied" or receipt.get("changed") != expected:
        raise RuntimeError(f"writer did not apply all previewed rows: {receipt}")
    print(json.dumps({"quiet": True, "receipt": receipt}, sort_keys=True))


def verify() -> None:
    record = read()
    receipt = record.get("receipt") or {}
    changed = receipt.get("changed")
    if receipt.get("outcome") != "applied" or changed != record["matched_count"]:
        raise RuntimeError("successful apply receipt is required")
    before_snapshots = json.loads(BEFORE_SNAPSHOTS.read_text())
    after_snapshots: dict[str, Any] = {}
    with GraphClient() as graph:
        after_classifier = classifier(graph)
        record["classifier_after"] = after_classifier
        baseline = record["classifier_before"]
        unbound = {row["name"] for row in record["rows"] if row["action"] == "unbind"}
        before_names = set(baseline["names"])
        after_names = set(after_classifier["names"])
        assert not (unbound & before_names), (
            "an intended unbind was already sourceless at baseline"
        )
        assert len(after_names) == len(before_names) + changed, (
            "sourceless count did not rise by the applied count"
        )
        assert after_names - before_names == unbound, (
            "post-apply sourceless entrants differ from applied rows"
        )
        for row in record["rows"]:
            if row["action"] != "unbind":
                continue
            before = before_snapshots[row["name"]]
            after = snapshot(graph, row["name"], row["source_id"])
            after_snapshots[row["name"]] = after
            assert after["binding_ids"] == [], f"binding remains for {row['name']}"
            assert after["name"] == before["name"], (
                f"name properties changed for {row['name']}"
            )
            assert after["source"] == before["source"], (
                f"source properties or status changed for {row['name']}"
            )
            receipt_ids = set(receipt["change_ids"])
            name_before = before["StandardName_edges"]
            name_after = [
                edge
                for edge in after["StandardName_edges"]
                if not (
                    edge["kind"] == "HAS_INTERNAL_CHANGE"
                    and edge["other_id"] in receipt_ids
                )
            ]
            binding_id = before["binding_ids"][0]
            assert name_after == [
                edge for edge in name_before if edge["edge_id"] != binding_id
            ], f"other name edges changed for {row['name']}"
            assert after["StandardNameSource_edges"] == [
                edge
                for edge in before["StandardNameSource_edges"]
                if edge["edge_id"] != binding_id
            ], f"source edges changed for {row['name']}"
            row["readback"] = {
                "binding_gone": True,
                "name_unchanged": True,
                "stage_unchanged": True,
                "other_edges_unchanged": True,
                "source_unchanged": True,
                "source_status_unchanged": True,
                "source_status": after["source"].get("status"),
                "sourceless": row["name"] in after_names,
                "after_snapshot": str(AFTER_SNAPSHOTS),
            }
        paths = [
            row["candidate"] for row in record["rows"] if row["action"] == "unbind"
        ]
        recorder_control = query(
            graph,
            """MATCH (name:StandardName)
            RETURN count(name.source_path) AS with_source_path,
                   count(CASE WHEN size(coalesce(name.source_paths, [])) > 0 THEN 1 END)
                   AS with_source_paths""",
        )[0]
        if (
            not recorder_control["with_source_path"]
            or not recorder_control["with_source_paths"]
        ):
            raise RuntimeError("live name path-property query has no positive control")
        record["path_recorder_instrument"] = recorder_control
        candidates = query(
            graph,
            """UNWIND $paths AS path
            MATCH (other:StandardName)
            WHERE NOT other.status IN ['deprecated', 'superseded']
              AND (path = other.source_path OR path IN coalesce(other.source_paths, [])
                   OR 'dd:' + path IN coalesce(other.source_paths, []))
            RETURN path, other.id AS name, other.name_stage AS stage, other.status AS status
            ORDER BY path, name LIMIT 500""",
            paths=paths,
        )
        by_path: dict[str, list[dict[str, Any]]] = {path: [] for path in paths}
        for candidate in candidates:
            by_path[candidate["path"]].append(
                {k: candidate[k] for k in ("name", "stage", "status")}
            )
        for row in record["rows"]:
            if row["action"] != "unbind":
                continue
            holders = query(
                graph,
                """MATCH (source:StandardNameSource {id: $source_id})
                OPTIONAL MATCH (source)-[:PRODUCED_NAME]->(holder:StandardName)
                RETURN [name IN collect(holder.id) WHERE name IS NOT NULL] AS holders""",
                source_id=row["source_id"],
            )
            assert not holders[0]["holders"], f"DD path remains bound for {row['name']}"
            assert (
                after_snapshots[row["name"]]["source"]["dd_path"] == row["candidate"]
            ), f"DD source path identity changed for {row['name']}"
            other_recorders = [
                candidate
                for candidate in by_path[row["candidate"]]
                if candidate["name"] != row["name"]
            ]
            row["freed_path"] = {
                "dd_path": row["candidate"],
                "remaining_holders": holders[0]["holders"],
                "available_in_graph": not holders[0]["holders"],
                "other_live_recorders": other_recorders,
                "other_live_name_could_take": not holders[0]["holders"],
                "source_dd_path_positive_control": True,
                "reassignment_condition": "A different live name may bind this free path after semantic review; no other live name has it in source_path or source_paths"
                if not other_recorders
                else "A different live name may bind this free path after semantic review; other live path recorders are listed",
            }
    AFTER_SNAPSHOTS.write_text(
        json.dumps(after_snapshots, indent=2, sort_keys=True) + "\n"
    )
    record["verification"] = {
        "applied": changed,
        "sourceless_before": baseline["count"],
        "sourceless_after": after_classifier["count"],
        "increase": after_classifier["count"] - baseline["count"],
        "all_rows_checked": True,
    }
    save(record)
    print(json.dumps(record["verification"], sort_keys=True))


def refusal_control() -> None:
    record = read()
    outcome = run_writer(apply=False)
    record["post_apply_refusal_control"] = {
        "outcome": outcome["outcome"],
        "would_change": outcome["would_change"],
        "counts": outcome["counts"],
        "refusals": outcome["refusals"],
    }
    save(record)
    expected_names = {
        row["name"] for row in record["rows"] if row["action"] == "unbind"
    }
    refused_names = {row["row_id"] for row in outcome["refusals"]}
    if (
        outcome["outcome"] != "refused"
        or outcome["would_change"] != 0
        or refused_names != expected_names
    ):
        raise RuntimeError(
            "absent-edge refusal control did not fire for every applied row"
        )
    if {row["reason"] for row in outcome["refusals"]} != {
        "recovery binding does not exist"
    }:
        raise RuntimeError("absent-edge refusal control produced an unexpected reason")
    print(json.dumps(record["post_apply_refusal_control"], sort_keys=True))


def render() -> None:
    record = read()
    verified = record.get("verification") or {}
    if not verified.get("all_rows_checked") or verified.get("increase") != 16:
        raise RuntimeError("complete graph verification is required before rendering")
    if (
        record.get("post_apply_refusal_control", {}).get("counts", {}).get("refused")
        != 16
    ):
        raise RuntimeError(
            "writer's absent-edge refusal control is required before rendering"
        )
    prefix = "sib-apply-recovery-unbind"
    authority = json.loads(AUTHORITY.read_text())
    preview_result = record["preview"]
    receipt = record["receipt"]
    quiet = record["quiet_graph"]
    rows = []
    for index, row in enumerate(record["rows"], 1):
        match = row["match"]
        readback = row["readback"]
        freed = row["freed_path"]
        if not all(
            readback[key]
            for key in (
                "binding_gone",
                "name_unchanged",
                "stage_unchanged",
                "other_edges_unchanged",
                "source_unchanged",
                "source_status_unchanged",
                "sourceless",
            )
        ):
            raise RuntimeError(f"row {row['name']} has an incomplete read-back")
        if match["reasons"] or not freed["available_in_graph"]:
            raise RuntimeError(
                f"row {row['name']} is not matched or its path remains bound"
            )
        rows.append(
            f'<tr id="{prefix}-row-{index}"><td><code>{escape(row["name"])}</code></td>'
            f"<td><code>{escape(row['candidate'])}</code></td>"
            f"<td>{row['probability']:.2f} / {row['confidence']:.2f}</td>"
            f"<td>edge present; stage <code>{escape(match['name_stage'])}</code>; "
            f"source <code>{escape(match['source_status'])}</code>; no refusals</td>"
            f"<td>{escape(row['action'])}; edge gone; name, stage, other edges and "
            f"source including status unchanged (<code>{escape(readback['source_status'])}</code>); sourceless</td>"
            f"<td>free; another live name may take it after semantic review; "
            f"{len(freed['other_live_recorders'])} other live path recorders</td></tr>"
        )
    table_rows = "\n".join(rows)
    html = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="reckon-type" content="evidence-fragment">
<meta name="plan-slug" content="sn-integrity-backlog">
<meta name="node-id" content="{prefix}">
<title>Sixteen exhausted-name recovery bindings removed through signed authority</title></head>
<body><main>
<h1 id="{prefix}-title">Sixteen sub-floor recovery bindings removed</h1>
<p id="{prefix}-summary">All 16 recorded bindings still existed on exhausted names with non-stale
sources when re-derived from the live graph. The signed writer preview admitted all 16 with
zero refusals. A quiet-graph probe immediately preceded the apply, and the writer removed
16 edges and recorded 16 change receipts. The independent classifier rose from
{verified["sourceless_before"]} to {verified["sourceless_after"]}, exactly 16, with each
unbound name appearing in the new sourceless set. No LLM call was made; spend USD 0.</p>
<h2 id="{prefix}-authority">Authority and preview</h2>
<p>The authority uses this program's own <code>{escape(authority["schema"])}</code>
schema. Its loader requires a SHA-256 signature over canonical rows rather than the
canonical repair-authority envelope, so this operation follows the program-specific
contract. <a href="/imas-codex/evidence/fragments/sn-integrity-backlog/sib-apply-recovery-unbind-authority.json">Signed authority</a>;
<a href="/imas-codex/evidence/fragments/sn-integrity-backlog/sib-apply-recovery-unbind.json">full machine record</a>.</p>
<pre id="{prefix}-authority-data">{escape(json.dumps({"authority_rows": len(authority["rows"]), "signature": authority["signature"], "authority_file_sha256": record["authority_file_sha256"], "preview_outcome": preview_result["outcome"], "preview_would_change": preview_result["would_change"], "preview_refusals": preview_result["refusals"], "preview_manifest_sha256": preview_result["manifest_sha256"]}, indent=2))}</pre>
<h2 id="{prefix}-apply">Quiet graph and signed receipt</h2>
<pre id="{prefix}-receipt">{escape(json.dumps({"quiet_graph": quiet, "receipt": {"outcome": receipt["outcome"], "changed": receipt["changed"], "mutations": receipt["mutations"], "receipt_rows": receipt["receipt_rows"], "manifest_sha256": receipt["manifest_sha256"], "authority_payload_sha256": receipt["authority_payload_sha256"]}}, indent=2))}</pre>
<h2 id="{prefix}-rows">Every binding and freed DD path</h2>
<p>The match column records the pre-apply edge, stage and source status. Each read-back
compares full pre- and post-apply name and source properties and every other edge; the
only new name edge is the writer's signed change receipt. All 16 DD paths have no
remaining producer. A free path can be assigned to another live name only after its
quantity is reviewed; none of these paths appears in another live name's
<code>source_path</code> or <code>source_paths</code> record. The path-property
query has a positive control across the live name population:
{record["path_recorder_instrument"]["with_source_path"]} names have
<code>source_path</code>, and {record["path_recorder_instrument"]["with_source_paths"]}
have nonempty <code>source_paths</code>. Each freed source's own <code>dd_path</code>
still equals the listed path.</p>
<table id="{prefix}-table"><thead><tr><th>Name</th><th>DD path freed</th><th>P(choice) / confidence</th><th>Live match</th><th>Action and read-back</th><th>Other live use</th></tr></thead>
<tbody>{table_rows}</tbody></table>
<h2 id="{prefix}-classifier">Classifier check</h2>
<p>The same <code>classify_sourceless_standard_names(gc)</code> ran before and after:
{verified["sourceless_before"]} + {receipt["changed"]} = {verified["sourceless_after"]}.
The set difference equals the 16 signed names, with no other entrant. The
<code>verify-final.log</code> run ended at <code>EXIT=0</code> after every assertion.
The writer's post-apply preview then refused all 16 absent edges with
<code>recovery binding does not exist</code> and <code>would_change: 0</code>
(<code>refusal-control.log</code>, <code>EXIT=0</code>).</p>
</main></body></html>
'''
    FRAGMENT.write_text(html)
    print(
        json.dumps(
            {
                "fragment": str(FRAGMENT),
                "rows": len(rows),
                "sourceless_increase": verified["increase"],
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=("prepare", "preview", "apply", "verify", "refusal", "render")
    )
    mode = parser.parse_args().mode
    {
        "prepare": prepare,
        "preview": preview,
        "apply": apply,
        "verify": verify,
        "refusal": refusal_control,
        "render": render,
    }[mode]()


if __name__ == "__main__":
    main()
