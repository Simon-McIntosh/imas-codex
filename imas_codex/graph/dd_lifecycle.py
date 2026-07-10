"""Reconcile IMASNode lifecycle against the current Data Dictionary.

The graph accumulates IMASNode entries across DD builds, so paths that a new
major DD version removed or renamed (e.g. the DDv3 ``*_tor`` family renamed
to ``*_phi`` in DD 4.0) linger with a stale ``lifecycle_status``. Standard
names must be grounded strictly in current-DD semantics, coordinates, COCOS
and definitions — a node absent from the current DD must never seed a
StandardNameSource.

This module derives, from the packaged DD XML for the version imas-codex is
built against:

- the set of valid machine paths (``<ids>/<path>``), and
- the old→new rename map from ``change_nbc_previous_name`` metadata
  (propagated through ancestors, so a structure rename maps its whole
  subtree);

and stamps every IMASNode absent from that set with
``lifecycle_status='removed'`` (plus ``renamed_to`` when the rename map
resolves it). Nodes present in the DD that were previously stamped are
restored. Idempotent; run after every graph build and safe to run any time.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from xml.etree import ElementTree as ET

logger = logging.getLogger(__name__)

#: lifecycle_status value for nodes absent from the current DD.
LIFECYCLE_REMOVED = "removed"


@lru_cache(maxsize=2)
def dd_path_index(
    dd_version: str | None = None,
) -> tuple[frozenset[str], dict[str, str]]:
    """Return ``(valid_paths, old_to_new)`` for *dd_version*.

    ``valid_paths`` holds every machine path (``<ids>/<path>``) plus the bare
    IDS names. ``old_to_new`` maps pre-rename paths to their current spelling,
    derived from ``change_nbc_previous_name`` and propagated to descendants
    (a renamed structure moves its whole subtree).
    """
    import imas_data_dictionaries as idd

    from imas_codex import dd_version as current_dd_version

    version = dd_version or current_dd_version
    root = ET.fromstring(idd.get_dd_xml(version))

    valid: set[str] = set()
    old_to_new: dict[str, str] = {}

    def walk(
        el: ET.Element, new_segs: list[str], old_segs: list[str], ids: str
    ) -> None:
        for f in el.findall("field"):
            name = f.get("name")
            if not name:
                continue
            prev = f.get("change_nbc_previous_name")
            ns = [*new_segs, name]
            os_ = [*old_segs, *(prev.split("/") if prev else [name])]
            new_path = f"{ids}/{'/'.join(ns)}"
            valid.add(new_path)
            if os_ != ns:
                old_to_new[f"{ids}/{'/'.join(os_)}"] = new_path
            walk(f, ns, os_, ids)

    for ids_el in root.iter("IDS"):
        ids_name = ids_el.get("name")
        if not ids_name:
            continue
        valid.add(ids_name)
        walk(ids_el, [], [], ids_name)

    return frozenset(valid), old_to_new


def reconcile_node_lifecycle(
    *, gc=None, dd_version: str | None = None
) -> dict[str, int]:
    """Stamp IMASNodes absent from the current DD as removed; restore returnees.

    Returns counters: ``marked_removed``, ``renamed_annotated``, ``restored``,
    ``total_nodes``.
    """
    from imas_codex.graph.client import GraphClient

    valid, old_to_new = dd_path_index(dd_version)
    owns = gc is None
    gc = gc or GraphClient()
    try:
        rows = gc.query(
            "MATCH (n:IMASNode) RETURN n.id AS id, n.lifecycle_status AS ls"
        )
        absent = [r["id"] for r in rows if r["id"] not in valid]
        returned = [
            r["id"] for r in rows if r["id"] in valid and r["ls"] == LIFECYCLE_REMOVED
        ]
        renames = [{"id": p, "to": old_to_new[p]} for p in absent if p in old_to_new]
        if absent:
            gc.query(
                """
                UNWIND $ids AS nid
                MATCH (n:IMASNode {id: nid})
                SET n.lifecycle_status = $removed
                """,
                ids=absent,
                removed=LIFECYCLE_REMOVED,
            )
        if renames:
            gc.query(
                """
                UNWIND $pairs AS pr
                MATCH (n:IMASNode {id: pr.id})
                SET n.renamed_to = pr.to
                """,
                pairs=renames,
            )
        if returned:
            gc.query(
                """
                UNWIND $ids AS nid
                MATCH (n:IMASNode {id: nid})
                SET n.lifecycle_status = null, n.renamed_to = null
                """,
                ids=returned,
            )
        result = {
            "total_nodes": len(rows),
            "marked_removed": len(absent),
            "renamed_annotated": len(renames),
            "restored": len(returned),
        }
        logger.info("dd_lifecycle reconcile: %s", result)
        return result
    finally:
        if owns:
            gc.close()
