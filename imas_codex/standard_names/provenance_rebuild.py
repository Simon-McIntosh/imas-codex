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
- **residue** — any live name with no deterministic anchor gets an explicit
  ``source_type='manual'`` source (auditable; never a fabricated DD path).

The final link topology (``StandardNameSource`` + ``PRODUCED_NAME`` +
``FROM_DD_PATH`` + ``HAS_PARENT``) is identical to a fresh from-scratch build
because it is produced by the same deterministic routines — proven by an empty
orphan set + idempotent re-run.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

import yaml

from imas_codex.graph.client import GraphClient
from imas_codex.standard_names.graph_ops import (
    normalize_derived_parent_lifecycle,
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
        if not name or not sources:
            continue
        specs: list[dict[str, Any]] = []
        for src in sources:
            source_id = src.get("id")
            if not source_id:
                continue
            source_type = _source_type_from_id(source_id)
            spec: dict[str, Any] = {
                "id": source_id,
                "source_type": source_type,
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
    return recovery_sources_from_entries(entries)


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


def _synth_spec(name_id: str, source_type: str) -> dict[str, Any]:
    """Build an explicit non-DD source spec (``derived`` or ``manual``).

    Never carries a ``dd_path`` — these are the conservative fallbacks for
    names with no deterministic DD anchor, so a DD path is never fabricated.
    """
    prefix = "derived" if source_type == "derived" else "manual"
    status = "composed" if source_type == "derived" else "attached"
    return {
        "id": f"{prefix}:{name_id}",
        "source_type": source_type,
        "source_id": name_id,
        "status": status,
    }


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


def _classify_derived_parents(gc: GraphClient, ids: list[str]) -> set[str]:
    """Return the subset of *ids* that are DERIVED PARENTS.

    A derived parent is a name that other names point at via ``HAS_PARENT``
    (i.e. it has children) or is explicitly ``origin='derived'``. Its provenance
    is the composed-from child group — the plan's "derived — a parent/group of
    SNs" — so it gets a ``derived`` source, not ``manual``. (A pure child with a
    lost DD anchor is NOT derived here — its source was a DD leaf.)
    """
    if not ids:
        return set()
    rows = gc.query(
        """
        MATCH (sn:StandardName) WHERE sn.id IN $ids
        RETURN sn.id AS id,
               (exists { (:StandardName)-[:HAS_PARENT]->(sn) }
                OR coalesce(sn.origin, '') = 'derived') AS is_parent
        """,
        ids=ids,
    )
    return {r["id"] for r in rows if r.get("is_parent")}


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
    """Replay the deterministic half of a fresh build (loop B2 + B3b).

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


def rebuild_provenance(
    *,
    gc: GraphClient | None = None,
    isnc_dir: str | Path | None = None,
    ref: str = DEFAULT_RECOVERY_REF,
    recovery_map: dict[str, list[dict[str, Any]]] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Rebuild provenance for every orphaned live name to fresh-parity.

    Runs the deterministic fresh-build fixpoints FIRST (reattach true sources,
    rederive structure, materialise derived-parent sources, relink DD/signal),
    so derived parents and reattachable names are sourced natively. Then, for
    any live name STILL without a source, a conservative decision tree binds:
    (1) ISNC recovery map → dd/signal; (2) surviving ``source_paths`` scalar →
    dd/signal (an authoritative in-graph anchor); (3) derived parent (has
    children) → an explicit ``derived`` source (composed-from-children);
    (4) residue with no anchor → an explicit ``manual`` source. Never fabricates
    a DD path. Content (name/description/docs/stage) is never touched.
    """
    owns = gc is None
    gc = gc or GraphClient()
    try:
        if recovery_map is None:
            recovery_map = load_recovery_map(isnc_dir, ref) if isnc_dir else {}

        # Deterministic fixpoints (fresh-build B2+B3b): reattach edge/scalar
        # desyncs to their TRUE source, rederive HAS_PARENT, and materialise a
        # derived StandardNameSource for every admissible derived parent, then
        # relink FROM_DD_PATH. Run before classification so parents drop out of
        # the orphan set with their real composed-from-children provenance.
        desync_ids = {d["sn_id"] for d in find_edge_scalar_desyncs(gc=gc)}
        reattached = 0
        if not dry_run:
            reattached = reattach_produced_name_edges(gc=gc)
            _run_deterministic_fixpoints()

        orphans = find_provenance_orphans(gc=gc)
        orphan_ids = [o["sn_id"] for o in orphans if o["sn_id"] not in desync_ids]

        # Classify the remainder by descending anchor authority.
        not_mapped = [i for i in orphan_ids if i not in recovery_map]
        scalar_specs = _fetch_dd_source_paths(gc, not_mapped)
        not_scalar = [i for i in not_mapped if i not in scalar_specs]
        # Exclude-pending guard: an orphan whose real dd source is still pending
        # (extracted/drafted) in the GENERATE_NAME queue must not be given a
        # synthesized derived/manual fallback — the pipeline will source it.
        pending_names = _fetch_pending_source_names(gc, not_scalar)
        parent_ids = _classify_derived_parents(
            gc, [i for i in not_scalar if i not in pending_names]
        )

        summary: dict[str, Any] = {
            "orphans_before": len(orphan_ids),
            "reattached": reattached if not dry_run else len(desync_ids),
            "bound_from_map": 0,
            "bound_from_scalar": 0,
            "bound_derived": 0,
            "bound_manual": 0,
            "excluded_pending": 0,
            "dry_run": dry_run,
        }
        for name_id in orphan_ids:
            if name_id in recovery_map:
                specs = recovery_map[name_id]
                summary["bound_from_map"] += 1
            elif name_id in scalar_specs:
                specs = scalar_specs[name_id]
                summary["bound_from_scalar"] += 1
            elif name_id in pending_names:
                # Real dd source pending in the queue — leave it for the
                # pipeline; never synthesize a fallback over a claimable source.
                summary["excluded_pending"] += 1
                continue
            elif name_id in parent_ids:
                specs = [_synth_spec(name_id, "derived")]
                summary["bound_derived"] += 1
            else:
                specs = [_synth_spec(name_id, "manual")]
                summary["bound_manual"] += 1
            if not dry_run:
                bind_recovery_sources(name_id, specs, gc=gc)

        if not dry_run:
            summary["orphans_after"] = len(find_provenance_orphans(gc=gc))

        logger.info("rebuild_provenance: %s", summary)
        return summary
    finally:
        if owns:
            gc.close()
