"""One-time provenance rebuild — recover the ledger to fresh-parity.

The export→import round-trip strips pipeline provenance, so names re-enter the
graph (``origin='catalog_edit'``) with no ``StandardNameSource``. Of the live
names, a large fraction carry no ``PRODUCED_NAME`` source at all. This module
recovers 100% provenance by replaying the **deterministic** half of a fresh
build against the **existing** names — it never regenerates names or docs:

- **dd sources** — rebind each name to its ``StandardNameSource(dd)`` +
  ``FROM_DD_PATH`` using an authoritative ISNC recovery commit's ``sources:``
  blocks as the map; the DD graph closes gaps deterministically.
- **derived / parent structure** — reconstructed by the existing grammar
  fixpoint (:func:`graph_ops.rederive_structural_edges`).
- **change history** — a live name whose latest recorded predecessor still has
  real semantic sources inherits those existing sources.
- **residue** — any live name without deterministic or historical evidence is
  reported unresolved. No fallback source is fabricated.

Recoverable link topology (``StandardNameSource`` + ``PRODUCED_NAME`` +
``FROM_DD_PATH`` + ``HAS_PARENT``) is produced by the same deterministic
routines as a fresh build. The report retains an explicit unresolved set for
names whose evidence is insufficient, and an idempotent re-run never invents
new provenance for that residue.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

import yaml

from imas_codex.graph.client import GraphClient
from imas_codex.standard_names.attachment_audit import (
    AttachmentAuditResult,
    audit_attachments,
    reconcile_attachment_consistency,
)
from imas_codex.standard_names.graph_ops import (
    classify_orphan_parent_source_candidates,
    find_orphan_parent_source_candidates,
    normalize_derived_parent_lifecycle,
    reconcile_orphan_parent_sources,
    reconcile_standard_name_sources,
    rederive_structural_edges,
    seed_parent_sources,
    structural_accept_derived_parents,
)
from imas_codex.standard_names.ledger import (
    find_edge_scalar_desyncs,
    find_provenance_orphans,
    reattach_produced_name_edges,
)
from imas_codex.standard_names.provenance_lifecycle import (
    DELETION_OPERATIONS,
    bind_sources_exclusively,
    find_semantic_source_invariant_violations,
    retire_unrecoverable_provenance_orphans,
)
from imas_codex.standard_names.source_paths import parse_source_path

logger = logging.getLogger(__name__)

DD_PREFIX = "dd:"

#: The ISNC commit that still carried near-complete ``sources:`` blocks
#: (2026-07-03) — the authoritative recovery map for the bulk of names before
#: the round-trip erosion. Overridable per-call.
DEFAULT_RECOVERY_REF = "a2f8831"

#: MERGE a batch of recovery source specs for one live name, gating every write
#: on the ``StandardName`` existing (MATCH-before-SET) so a missing name never
#: mints an orphan source. FROM_DD_PATH is linked only when the ``IMASNode``
#: still exists (a stale DD path leaves the source without a leaf link, never
#: fabricates one). ``produced_sn_id`` mirrors the edge for recoverability.
_BIND_SOURCES = """
    MATCH (sn:StandardName {id: $name_id})
    UNWIND $specs AS spec
    MERGE (sns:StandardNameSource {id: spec.id})
      ON CREATE SET sns.created_at = datetime(), sns.attempt_count = 0
    SET sns.source_type = spec.source_type,
        sns.source_id = spec.source_id,
        sns.status = spec.status,
        sns.produced_sn_id = sn.id,
        sns.composed_at = coalesce(sns.composed_at, datetime()),
        sns.claimed_at = null,
        sns.claim_token = null
    MERGE (sns)-[:PRODUCED_NAME]->(sn)
    WITH sns, spec
    OPTIONAL MATCH (imas:IMASNode {id: spec.dd_path})
    FOREACH (_ IN CASE WHEN imas IS NULL THEN [] ELSE [1] END |
        MERGE (sns)-[:FROM_DD_PATH]->(imas))
    WITH sns, spec
    OPTIONAL MATCH (sig:FacilitySignal {id: spec.signal_id})
    FOREACH (_ IN CASE WHEN sig IS NULL THEN [] ELSE [1] END |
        MERGE (sns)-[:FROM_SIGNAL]->(sig))
    RETURN count(DISTINCT sns) AS bound
"""


def _source_type_from_id(source_id: str) -> str:
    """Infer a source's type from its URI id.

    ``dd:`` → ``'dd'``; anything else (facility-prefixed signal id) → ``'signals'``.
    """
    return "dd" if source_id.startswith(DD_PREFIX) else "signals"


def recovery_sources_from_entries(
    entries: list[dict[str, Any]],
    *,
    include_catalog_entries: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """Extract the provenance recovery map from parsed catalog entries.

    Returns ``{name: [source_spec, ...]}`` for every entry that carries a
    non-empty ``sources:`` block. Each ``source_spec`` is normalised to the
    fields the rebuild needs to reconstruct a ``StandardNameSource``:
    ``{id, source_type, dd_path?, signal_id?, status}``.
    """
    recovered: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        name = entry.get("name")
        sources = entry.get("sources")
        if not name:
            continue
        if not sources:
            if include_catalog_entries:
                recovered[name] = [
                    {
                        "id": f"catalog:{name}",
                        "source_type": "catalog",
                        "source_id": name,
                        "status": "attached",
                    }
                ]
            continue
        specs: list[dict[str, Any]] = []
        for src in sources:
            source_id = src.get("id")
            if not source_id and src.get("dd_path"):
                source_id = f"dd:{src['dd_path']}"
            elif not source_id and src.get("signal_id"):
                source_id = f"signals:{src['signal_id']}"
            if not source_id:
                continue
            source_type = _source_type_from_id(source_id)
            spec: dict[str, Any] = {
                "id": source_id,
                "source_type": source_type,
                "source_id": (
                    src.get("dd_path") if source_type == "dd" else src.get("signal_id")
                ),
                "status": src.get("status", "attached"),
            }
            if src.get("dd_path"):
                spec["dd_path"] = src["dd_path"]
            elif source_type == "dd":
                spec["dd_path"] = source_id[len(DD_PREFIX) :]
            if src.get("signal_id"):
                spec["signal_id"] = src["signal_id"]
            specs.append(spec)
        if specs:
            recovered[name] = specs
    return recovered


def _git_lines(cwd: Path, *args: str) -> list[str]:
    """Run a read-only git command in *cwd*; return stdout lines ([] on error)."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:  # git missing / not executable
        logger.warning("git invocation failed in %s: %s", cwd, exc)
        return []
    if result.returncode != 0:
        logger.warning("git %s failed: %s", args[0], result.stderr.strip())
        return []
    return result.stdout.splitlines()


def _git_show(cwd: Path, ref: str, path: str) -> str | None:
    """Return the content of *path* at *ref* (None on error)."""
    try:
        result = subprocess.run(
            ["git", "show", f"{ref}:{path}"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def load_recovery_map(
    isnc_dir: str | Path,
    ref: str = DEFAULT_RECOVERY_REF,
) -> dict[str, list[dict[str, Any]]]:
    """Extract the ``{name: [source_spec]}`` recovery map from a catalog commit.

    Reads the ``standard_names/*.yml`` tree at *ref* (a git commit-ish, NOT the
    working tree) and parses each entry's ``sources:`` block. Returns an empty
    map if the ref/tree cannot be read (e.g. unknown ref, not a git repo).
    """
    isnc_dir = Path(isnc_dir)
    names = _git_lines(isnc_dir, "ls-tree", "-r", "--name-only", ref, "standard_names/")
    yaml_paths = [n for n in names if n.endswith((".yml", ".yaml"))]
    if not yaml_paths:
        return {}
    entries: list[dict[str, Any]] = []
    for path in yaml_paths:
        content = _git_show(isnc_dir, ref, path)
        if not content:
            continue
        try:
            docs = yaml.safe_load(content)
        except yaml.YAMLError as exc:
            logger.warning("failed to parse %s@%s: %s", path, ref, exc)
            continue
        if isinstance(docs, list):
            entries.extend(d for d in docs if isinstance(d, dict))
    return recovery_sources_from_entries(entries, include_catalog_entries=True)


def bind_recovery_sources(
    name_id: str,
    specs: list[dict[str, Any]],
    *,
    gc: GraphClient | None = None,
) -> int:
    """MERGE recovery source specs for one live name and link them.

    For each spec, MERGEs the ``StandardNameSource`` (by ``id``), sets its
    scalar fields + ``produced_sn_id`` mirror, MERGEs ``PRODUCED_NAME`` to the
    name, and links ``FROM_DD_PATH`` / ``FROM_SIGNAL`` where the upstream
    entity still exists. Every write is gated on the ``StandardName`` existing.
    Returns the number of sources bound (0 if the name is absent or no specs).
    """
    if not name_id or not specs:
        return 0
    owns = gc is None
    gc = gc or GraphClient()
    try:
        rows = gc.query(_BIND_SOURCES, name_id=name_id, specs=specs)
        return int(rows[0]["bound"]) if rows else 0
    finally:
        if owns:
            gc.close()


def _fetch_pending_source_names(gc: GraphClient, ids: list[str]) -> set[str]:
    """Return the subset of *ids* that have a claimable PENDING dd source.

    A live name may be a provenance orphan (no ``PRODUCED_NAME`` edge yet) not
    because its provenance is lost, but because the pipeline has not finished
    composing it: a ``StandardNameSource(status='extracted'|'drafted')`` destined
    for the name (``produced_sn_id`` names it) sits in the GENERATE_NAME queue.
    Such a name must NOT be given a synthesized ``derived`` / ``manual`` source —
    that would pre-empt the pipeline and pin a fabricated fallback over the real
    dd source about to be composed. It is excluded from the fallback and left
    for the pipeline to source. (The deterministic desync reattach in
    :mod:`ledger` deliberately ignores pending sources, so they survive to be
    caught here.)
    """
    if not ids:
        return set()
    rows = gc.query(
        """
        MATCH (sns:StandardNameSource)
        WHERE coalesce(sns.status, '') IN ['extracted', 'drafted']
          AND sns.produced_sn_id IS NOT NULL
          AND sns.produced_sn_id IN $ids
          AND NOT (sns)-[:PRODUCED_NAME]->(:StandardName)
        RETURN DISTINCT sns.produced_sn_id AS id
        """,
        ids=ids,
    )
    return {r["id"] for r in rows if r.get("id")}


def _fetch_change_history_sources(
    gc: GraphClient,
    ids: list[str],
) -> dict[str, list[str]]:
    """Recover source ids by walking each orphan's recorded predecessor chain.

    This uses only durable evidence already in the graph: a non-deletion
    ``StandardNameChange`` names each predecessor, and an existing composed or
    attached source still points to one of those predecessors by edge or scalar
    mirror. No source node or upstream DD/signal identity is invented.
    """
    if not ids:
        return {}
    change_rows = gc.query(
        """
        MATCH (change:StandardNameChange)
        WHERE NOT (coalesce(change.operation, '') IN $deletion_operations)
          AND change.from_name IS NOT NULL
          AND change.to_name IS NOT NULL
        RETURN change.from_name AS from_name,
               change.to_name AS to_name,
               change.changed_at AS changed_at
        ORDER BY change.changed_at DESC
        """,
        deletion_operations=sorted(DELETION_OPERATIONS),
    )
    predecessors_by_target: dict[str, list[dict[str, Any]]] = {}
    for row in change_rows:
        predecessors_by_target.setdefault(row["to_name"], []).append(dict(row))

    chains: dict[str, list[str]] = {}
    predecessor_ids: set[str] = set()
    for orphan_id in ids:
        chain: list[str] = []
        current = orphan_id
        visited = {current}
        for _ in range(100):
            candidates = predecessors_by_target.get(current, [])
            if not candidates:
                break
            latest = candidates[0]
            predecessor = latest.get("from_name")
            if not predecessor or predecessor in visited:
                break
            chain.append(predecessor)
            predecessor_ids.add(predecessor)
            visited.add(predecessor)
            current = predecessor
        chains[orphan_id] = chain

    if not predecessor_ids:
        return {}
    source_rows = gc.query(
        """
        MATCH (source:StandardNameSource)
        WHERE source.status IN ['composed', 'attached']
          AND NOT EXISTS {
            MATCH (source)-[:PRODUCED_NAME]->(live_target:StandardName)
            WHERE NOT (coalesce(live_target.name_stage, '') IN
                       ['superseded', 'exhausted', 'contested'])
          }
          AND NOT EXISTS {
            MATCH (scalar_target:StandardName)
            WHERE scalar_target.id = source.produced_sn_id
              AND NOT (coalesce(scalar_target.name_stage, '') IN
                       ['superseded', 'exhausted', 'contested'])
          }
          AND (
            source.produced_sn_id IN $predecessor_ids
            OR EXISTS {
              MATCH (source)-[:PRODUCED_NAME]->(prior:StandardName)
              WHERE prior.id IN $predecessor_ids
            }
          )
        OPTIONAL MATCH (source)-[:PRODUCED_NAME]->(target:StandardName)
        RETURN source.id AS source_id,
               source.produced_sn_id AS produced_sn_id,
               collect(DISTINCT target.id) AS target_ids
        """,
        predecessor_ids=sorted(predecessor_ids),
    )
    sources_by_target: dict[str, set[str]] = {}
    for row in source_rows:
        source_id = row.get("source_id")
        if not source_id:
            continue
        targets = set(row.get("target_ids") or [])
        if row.get("produced_sn_id"):
            targets.add(row["produced_sn_id"])
        for target in targets:
            if target in predecessor_ids:
                sources_by_target.setdefault(target, set()).add(source_id)

    recovered: dict[str, list[str]] = {}
    for orphan_id, chain in chains.items():
        for predecessor in chain:
            if source_ids := sources_by_target.get(predecessor):
                recovered[orphan_id] = sorted(source_ids)
                break
    return recovered


def _fetch_dd_source_paths(
    gc: GraphClient, ids: list[str]
) -> dict[str, list[dict[str, Any]]]:
    """For names carrying a surviving ``source_paths`` scalar, build source specs.

    The ``source_paths`` scalar (``dd:<path>`` / facility signal URIs) is a
    surviving in-graph anchor from the original composition — an authoritative,
    deterministic recovery of the real dd/signal source, superior to a manual
    fallback. Returns ``{name: [spec, ...]}`` only for names with a non-empty
    ``source_paths``.
    """
    if not ids:
        return {}
    rows = gc.query(
        """
        MATCH (sn:StandardName)
        WHERE sn.id IN $ids AND sn.source_paths IS NOT NULL
        RETURN sn.id AS id, sn.source_paths AS source_paths
        """,
        ids=ids,
    )
    out: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        specs: list[dict[str, Any]] = []
        for path in r["source_paths"] or []:
            source_type, source_id = parse_source_path(path)
            if source_type == "dd":
                specs.append(
                    {
                        "id": f"{DD_PREFIX}{source_id}",
                        "source_type": "dd",
                        "dd_path": source_id,
                        "status": "attached",
                    }
                )
            else:
                specs.append(
                    {
                        "id": path,
                        "source_type": "signals",
                        "signal_id": source_id,
                        "status": "attached",
                    }
                )
        if specs:
            out[r["id"]] = specs
    return out


def _run_deterministic_fixpoints() -> None:
    """Replay the deterministic half of a fresh build.

    These are the exact idempotent routines every ``sn run`` executes, so the
    resulting link topology (HAS_PARENT + derived StandardNameSource +
    FROM_DD_PATH) equals a fresh from-scratch build by construction. Crucially
    ``seed_parent_sources`` materialises a ``derived`` StandardNameSource for
    every admissible derived parent — the composed-from-children provenance.
    """
    rederive_structural_edges()
    seed_parent_sources()
    normalize_derived_parent_lifecycle()
    structural_accept_derived_parents()
    reconcile_standard_name_sources("dd")
    reconcile_standard_name_sources("signals")


_ADJUDICATION_KEYS = frozenset({"attachment_violations", "semantic_source_violations"})


def _attachment_violation_rows(result: AttachmentAuditResult) -> list[dict[str, Any]]:
    """Serialize attachment findings for exact adjudication and reporting."""
    return [
        {
            "source_node_id": verdict.source_node_id,
            "dd_path": verdict.dd_path,
            "sn_id": verdict.sn_id,
            "name_stage": verdict.name_stage,
            "reason": verdict.reason,
            "other_live_names": verdict.other_live_names,
        }
        for verdict in result.rejected
    ]


def _canonical_row(row: dict[str, Any]) -> str:
    """Return a stable exact-match key for a graph consistency finding."""
    return json.dumps(row, sort_keys=True, separators=(",", ":"), default=str)


def _unadjudicated_rows(
    rows: list[dict[str, Any]],
    manifest_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split current findings from stale entries in an exact-row manifest."""
    current = {_canonical_row(row): row for row in rows}
    manifested = {_canonical_row(row): row for row in manifest_rows}
    unresolved = [row for key, row in current.items() if key not in manifested]
    stale = [row for key, row in manifested.items() if key not in current]
    return unresolved, stale


def _reconcile_recovery_consistency(
    gc: GraphClient,
    *,
    dry_run: bool,
    adjudication_manifest: dict[str, list[dict[str, Any]]] | None,
) -> dict[str, Any]:
    """Run canonical post-replay consistency checks in dependency order.

    Attachment reconciliation runs first because it may detach a rejected
    pairing and update its scalar/projection mirrors. The one-live-source audit
    must therefore observe the post-attachment graph. Findings may be excluded
    from completion only by an exact-row manifest; partial matches and stale
    manifest rows keep completion fail-closed.
    """
    manifest = adjudication_manifest or {}
    unknown_keys = sorted(set(manifest) - _ADJUDICATION_KEYS)
    if unknown_keys:
        raise ValueError(
            "unknown recovery adjudication manifest fields: " + ", ".join(unknown_keys)
        )
    for key in _ADJUDICATION_KEYS:
        value = manifest.get(key, [])
        if not isinstance(value, list) or any(
            not isinstance(row, dict) for row in value
        ):
            raise TypeError(
                f"recovery adjudication manifest {key!r} must be a list of rows"
            )

    attachment_result = reconcile_attachment_consistency(gc=gc, dry_run=dry_run)
    attachment_postcheck = attachment_result if dry_run else audit_attachments(gc=gc)
    attachment_rows = _attachment_violation_rows(attachment_postcheck)
    semantic_rows = find_semantic_source_invariant_violations(gc)

    unresolved_attachments, stale_attachments = _unadjudicated_rows(
        attachment_rows,
        manifest.get("attachment_violations", []),
    )
    unresolved_semantic, stale_semantic = _unadjudicated_rows(
        semantic_rows,
        manifest.get("semantic_source_violations", []),
    )
    stale_manifest = {
        "attachment_violations": stale_attachments,
        "semantic_source_violations": stale_semantic,
    }
    return {
        "attachment_reconcile": attachment_result.as_dict(),
        "attachment_postcheck": attachment_postcheck.as_dict(),
        "attachment_violation_rows": attachment_rows,
        "semantic_source_violation_rows": semantic_rows,
        "unresolved_attachment_rows": unresolved_attachments,
        "unresolved_semantic_source_rows": unresolved_semantic,
        "stale_adjudication_rows": stale_manifest,
        "consistent": not (
            unresolved_attachments
            or unresolved_semantic
            or stale_attachments
            or stale_semantic
        ),
    }


def rebuild_provenance(
    *,
    gc: GraphClient | None = None,
    isnc_dir: str | Path | None = None,
    ref: str = DEFAULT_RECOVERY_REF,
    recovery_map: dict[str, list[dict[str, Any]]] | None = None,
    dry_run: bool = False,
    retire_unresolved: bool = False,
    include_accepted_retirement: bool = False,
    adjudication_manifest: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Rebuild provenance for every orphaned live name to fresh-parity.

    Runs the deterministic fresh-build fixpoints FIRST (reattach true sources,
    rederive structure, materialise derived-parent sources, relink DD/signal),
    so derived parents and reattachable names are sourced natively. Then, for
    any live name STILL without a source, a conservative decision tree binds:
    (1) ISNC recovery map → dd/signal; (2) surviving ``source_paths`` scalar →
    dd/signal (an authoritative in-graph anchor); (3) latest non-deletion change
    predecessor → that predecessor's existing semantic sources. Childful
    structural parents are repaired by :func:`reconcile_orphan_parent_sources`.
    Residue without evidence stays unresolved. After all replay bindings, the
    canonical attachment reconcile runs before the one-live-source invariant
    audit. Completion is false when either check still has an exact finding not
    covered by *adjudication_manifest*, or when that manifest is stale. The
    manifest accepts two keys, ``attachment_violations`` and
    ``semantic_source_violations``; values must be full rows copied from the
    corresponding consistency result. A dry run reports ``would_complete`` but
    never claims applied completion. Content (name/description/docs/stage) is
    never touched.
    """
    owns = gc is None
    gc = gc or GraphClient()
    try:
        if recovery_map is None:
            recovery_map = load_recovery_map(isnc_dir, ref) if isnc_dir else {}

        # Deterministic fixpoints: reattach edge/scalar
        # desyncs to their TRUE source, rederive HAS_PARENT, and materialise a
        # derived StandardNameSource for every admissible derived parent, then
        # relink FROM_DD_PATH. Run before classification so parents drop out of
        # the orphan set with their real composed-from-children provenance.
        initial_orphans = find_provenance_orphans(gc=gc)
        initial_orphan_ids = {row["sn_id"] for row in initial_orphans}
        desync_ids = {d["sn_id"] for d in find_edge_scalar_desyncs(gc=gc)}
        reattached = 0
        parent_sources_reconciled = 0
        if not dry_run:
            reattached = reattach_produced_name_edges(gc=gc)
            _run_deterministic_fixpoints()
        parent_candidate_rows = find_orphan_parent_source_candidates(gc=gc)
        parent_classification = classify_orphan_parent_source_candidates(
            gc,
            parent_candidate_rows,
        )
        repairable_parent_ids = {
            row["parent_id"]
            for row in parent_classification["repairable"]
            if row.get("parent_id") in initial_orphan_ids
        }
        rejected_parent_ids = {
            row["parent_id"]
            for row in parent_classification["rejected_derived"]
            if row.get("parent_id") in initial_orphan_ids
        }
        if not dry_run:
            parent_sources_reconciled = reconcile_orphan_parent_sources(
                gc=gc,
                classification=parent_classification,
            )

        orphans = find_provenance_orphans(gc=gc)
        orphan_ids = [
            orphan["sn_id"]
            for orphan in orphans
            if orphan["sn_id"] not in desync_ids
            and orphan["sn_id"] not in repairable_parent_ids
        ]

        # Classify the remainder by descending anchor authority.
        not_mapped = [i for i in orphan_ids if i not in recovery_map]
        scalar_specs = _fetch_dd_source_paths(gc, not_mapped)
        not_scalar = [i for i in not_mapped if i not in scalar_specs]
        history_sources = _fetch_change_history_sources(gc, not_scalar)
        not_historical = [i for i in not_scalar if i not in history_sources]
        # Exclude-pending guard: an orphan whose real dd source is still pending
        # (extracted/drafted) in the GENERATE_NAME queue must not be given a
        # fabricated fallback — the pipeline will source it.
        pending_names = _fetch_pending_source_names(gc, not_historical)

        summary: dict[str, Any] = {
            "orphans_before": len(initial_orphans),
            "reattached": reattached if not dry_run else len(desync_ids),
            "parent_source_candidates": len(repairable_parent_ids),
            "parent_source_candidate_names": sorted(repairable_parent_ids),
            "parent_source_rejected": len(rejected_parent_ids),
            "parent_source_rejected_names": sorted(rejected_parent_ids),
            "parent_sources_reconciled": parent_sources_reconciled,
            "bound_from_map": 0,
            "bound_from_scalar": 0,
            "bound_from_history": 0,
            "history_recoverable_names": sorted(history_sources),
            "excluded_pending": 0,
            "unresolved": 0,
            "unresolved_names": [],
            "retired_unresolved": 0,
            "retired_unresolved_names": [],
            "dry_run": dry_run,
        }
        for name_id in orphan_ids:
            if name_id in recovery_map:
                specs = recovery_map[name_id]
                summary["bound_from_map"] += 1
            elif name_id in scalar_specs:
                specs = scalar_specs[name_id]
                summary["bound_from_scalar"] += 1
            elif name_id in history_sources:
                summary["bound_from_history"] += 1
                if not dry_run:
                    bind_sources_exclusively(
                        gc,
                        name_id,
                        history_sources[name_id],
                        enforce_consistency=False,
                    )
                continue
            elif name_id in pending_names:
                # Real dd source pending in the queue — leave it for the
                # pipeline; never fabricate a fallback over a claimable source.
                summary["excluded_pending"] += 1
                continue
            else:
                summary["unresolved"] += 1
                summary["unresolved_names"].append(name_id)
                continue
            if not dry_run:
                bind_recovery_sources(name_id, specs, gc=gc)

        if not dry_run:
            summary["orphans_after"] = len(find_provenance_orphans(gc=gc))
            if retire_unresolved and summary["unresolved_names"]:
                retired = retire_unrecoverable_provenance_orphans(
                    gc,
                    summary["unresolved_names"],
                    include_accepted=include_accepted_retirement,
                )
                summary["retired_unresolved"] = len(retired)
                summary["retired_unresolved_names"] = retired
                remaining = find_provenance_orphans(gc=gc)
                summary["orphans_after"] = len(remaining)
                summary["unresolved"] = len(remaining)
                summary["unresolved_names"] = [row["sn_id"] for row in remaining]

        consistency = _reconcile_recovery_consistency(
            gc,
            dry_run=dry_run,
            adjudication_manifest=adjudication_manifest,
        )
        summary["consistency"] = consistency
        summary["would_complete"] = bool(
            consistency["consistent"] and summary["unresolved"] == 0
        )
        summary["completed"] = bool(not dry_run and summary["would_complete"])
        summary["completion_status"] = (
            "dry_run"
            if dry_run
            else ("complete" if summary["completed"] else "incomplete")
        )
        if not summary["would_complete"]:
            logger.error(
                "rebuild_provenance completion refused: unresolved_names=%s, "
                "attachment_rows=%s, semantic_source_rows=%s, stale_manifest=%s",
                summary["unresolved_names"],
                consistency["unresolved_attachment_rows"],
                consistency["unresolved_semantic_source_rows"],
                consistency["stale_adjudication_rows"],
            )

        logger.info("rebuild_provenance: %s", summary)
        return summary
    finally:
        if owns:
            gc.close()
