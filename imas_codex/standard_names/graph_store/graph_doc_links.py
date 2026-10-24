"""Graph storage operations for the standard-name pipeline."""

from __future__ import annotations

import re
import uuid
from typing import Any


def claim_unresolved_links(limit: int = 20, *, services: Any) -> list[dict[str, Any]]:
    """Claim StandardName nodes with unresolved links for resolution.

    Uses age-weighted random selection: nodes not checked recently have
    higher priority, preventing spin on temporarily unresolvable links.

    Returns list of dicts with ``id``, ``links``, ``link_retry_count``.
    """
    token = str(uuid.uuid4())
    with services.GraphClient() as gc:
        gc.query(
            "\n            MATCH (sn:StandardName)\n            WHERE sn.link_status = 'unresolved'\n              AND sn.claimed_at IS NULL\n              AND coalesce(sn.link_retry_count, 0) < $max_retries\n            WITH sn,\n                 duration.between(\n                     coalesce(sn.link_checked_at, datetime('2020-01-01')),\n                     datetime()\n                 ).minutes + 1.0 AS age_minutes\n            ORDER BY rand() * age_minutes DESC\n            LIMIT $limit\n            SET sn.updated_at = datetime(), sn.claimed_at = datetime(), sn.claim_token = $token\n            ",
            limit=limit,
            max_retries=services._MAX_LINK_RETRIES,
            token=token,
        )
        rows = list(
            gc.query(
                "\n                MATCH (sn:StandardName {claim_token: $token})\n                RETURN sn.id AS id, sn.links AS links,\n                       coalesce(sn.link_retry_count, 0) AS retry_count\n                ",
                token=token,
            )
        )
    return [dict(r) for r in rows]


def resolve_links_batch(
    items: list[dict[str, Any]],
    *,
    override: bool = False,
    override_names: set[str] | None = None,
    services: Any,
) -> dict[str, Any]:
    """Resolve dd: links to name: links for a batch of names.

    For each ``dd:path`` link, checks if a StandardName exists that was
    generated from that path. If found, replaces with ``name:sn_id``.

    Parameters
    ----------
    override:
        When ``True``, bypass pipeline protection on catalog-edited names.
    override_names:
        Selective override — set of name IDs that should bypass protection
        even if they have ``origin='catalog_edit'``.
    """
    from imas_codex.standard_names.protection import filter_protected

    items, skipped = filter_protected(
        items, override=override, override_names=override_names
    )
    if skipped:
        services.logger.warning(
            "resolve_links_batch: stripped protected fields from %d name(s): %s",
            len(skipped),
            ", ".join(skipped[:5]),
        )
    resolved_count = 0
    still_unresolved = 0
    failed_count = 0
    with services.GraphClient() as gc:
        all_dd_paths = set()
        for item in items:
            for link in item.get("links") or []:
                if link.startswith("dd:"):
                    all_dd_paths.add(link[3:])
        path_to_name: dict[str, str] = {}
        if all_dd_paths:
            rows = gc.query(
                "\n                UNWIND $paths AS path\n                MATCH (n:IMASNode {id: path})-[:HAS_STANDARD_NAME]->(sn:StandardName)\n                RETURN path AS dd_path, sn.id AS sn_id\n                ",
                paths=list(all_dd_paths),
            )
            for r in rows:
                path_to_name[r["dd_path"]] = r["sn_id"]
        for item in items:
            links = list(item.get("links") or [])
            new_links = []
            any_unresolved = False
            for link in links:
                if link.startswith("dd:"):
                    dd_path = link[3:]
                    if dd_path in path_to_name:
                        new_links.append(f"name:{path_to_name[dd_path]}")
                    else:
                        new_links.append(link)
                        any_unresolved = True
                else:
                    new_links.append(link)
            retry_count = item.get("retry_count", 0) + 1
            if any_unresolved and retry_count >= services._MAX_LINK_RETRIES:
                status = "failed"
                failed_count += 1
            elif any_unresolved:
                status = "unresolved"
                still_unresolved += 1
            else:
                status = "resolved"
                resolved_count += 1
            gc.query(
                "\n                MATCH (sn:StandardName {id: $id})\n                SET sn.updated_at = datetime(), sn.links = $links,\n                    sn.link_status = $status,\n                    sn.link_checked_at = datetime(),\n                    sn.link_retry_count = $retry_count,\n                    sn.claimed_at = null,\n                    sn.claim_token = null\n                ",
                id=item["id"],
                links=new_links,
                status=status,
                retry_count=retry_count,
            )
    return {
        "resolved": resolved_count,
        "unresolved": still_unresolved,
        "failed": failed_count,
    }


_NAME_LINK_RE = re.compile(r"\(name:([^)]+)\)")

_JSON_ESCAPE_CTRL = {"\b": r"\b", "\t": r"\t", "\f": r"\f", "\r": r"\r", "\v": r"\v"}

_STRIP_CTRL_RE = re.compile(r"[\x00-\x08\x0b-\x1f]")  # all control chars except \n


def _sanitize_doc_text(text: str | None, *, services: Any) -> str | None:
    """Repair JSON-escape-mangled LaTeX control characters in LLM doc text."""
    if not text:
        return text
    for ctrl, repl in services._JSON_ESCAPE_CTRL.items():
        if ctrl in text:
            text = text.replace(ctrl, repl)
    if services._STRIP_CTRL_RE.search(text):
        text = services._STRIP_CTRL_RE.sub("", text)
    return text


_DOC_LINK_RE = re.compile(r"\[([^\]]+)\]\(name:([a-z0-9_]+)\)")

_DOC_TEXT_UNSET = object()


def _doc_link_mismatches(
    gc: Any, documentation: str | None, *, services: Any
) -> list[str]:
    """Return ``[label](name:target)`` mismatches in *documentation*.

    A link is a mismatch when its snake_cased label is itself an EXISTING
    StandardName id different from the link target — the text promises one
    quantity while the href resolves to another. Human-readable labels that
    are not themselves ids are fine. Returns human-readable findings like
    ``"[radial_current_density] -> name:radial_total_current_density"``.
    """
    if not documentation:
        return []
    pairs = [
        (label.strip().lower().replace(" ", "_").replace("-", "_"), target, label)
        for label, target in services._DOC_LINK_RE.findall(documentation)
    ]
    candidates = {lid for lid, target, _ in pairs if lid != target}
    if not candidates:
        return []
    rows = gc.query(
        "MATCH (x:StandardName) WHERE x.id IN $ids RETURN x.id AS id",
        ids=sorted(candidates),
    )
    existing = {r["id"] for r in rows or []}
    return [
        f"[{label}] -> name:{target}"
        for lid, target, label in pairs
        if lid != target and lid in existing
    ]


def _normalize_bare_doc_links(
    gc: Any,
    sn_id: str | None = None,
    documentation: str | None | object = _DOC_TEXT_UNSET,
    *,
    dry_run: bool = False,
    services: Any,
) -> int | list[dict[str, Any]]:
    """Convert bare prose brackets to exact-name links or plain text.

    A bare bracketed phrase links only when its snake-case form is an exact live
    StandardName id other than the documented name. Otherwise its brackets are
    stripped. A ``name:`` link to the documented name becomes plain text.
    Idempotent. Return the number updated, or per-document changes in dry-run mode.

    When *sn_id* is given the pass is scoped to that single node (the cheap
    accept-path variant — see :func:`persist_reviewed_docs`): only the tokens
    present in that one doc are checked for liveness, so no full-catalogue scan
    is performed.  When *sn_id* is ``None`` (the post-drain reconcile) it scans
    every doc and resolves token liveness against the whole catalogue. The
    accept path passes its already-read *documentation* so docs without bare
    candidates need no second graph read.
    """
    if documentation is not services._DOC_TEXT_UNSET:
        if sn_id is None:
            raise ValueError("documentation requires a scoped standard name id")
        if not documentation or "[" not in documentation:
            return [] if dry_run else 0
        rows = [{"id": sn_id, "docs": documentation}]
    elif sn_id is not None:
        rows = list(
            gc.query(
                "MATCH (sn:StandardName {id: $id}) WHERE sn.documentation IS NOT NULL AND sn.documentation CONTAINS '[' RETURN sn.id AS id, sn.documentation AS docs",
                id=sn_id,
            )
        )
    else:
        rows = list(
            gc.query(
                "MATCH (sn:StandardName) WHERE sn.documentation IS NOT NULL AND sn.documentation CONTAINS '[' RETURN sn.id AS id, sn.documentation AS docs"
            )
        )
    if not rows:
        return [] if dry_run else 0
    bare_phrases = {
        row["id"]: tuple(
            dict.fromkeys(
                (
                    *services.find_prose_bracket_spans(row["docs"]),
                    *(
                        (reference.name, reference.start, reference.end)
                        for reference in services.find_name_references(
                            row["docs"], known=None
                        )
                        if reference.syntax == "bare_bracket"
                    ),
                )
            )
        )
        for row in rows
    }

    def candidate_id(phrase: str) -> str:
        return re.sub("[^a-z0-9]+", "_", phrase.strip().lower()).strip("_")

    if sn_id is not None:
        candidate_tokens = {
            candidate_id(phrase)
            for phrases in bare_phrases.values()
            for phrase, _, _ in phrases
        }
        names = (
            {
                r["id"]
                for r in gc.query(
                    "MATCH (s:StandardName) WHERE s.id IN $toks AND NOT coalesce(s.name_stage, '') IN ['superseded', 'exhausted', 'contested'] RETURN s.id AS id",
                    toks=list(candidate_tokens),
                )
            }
            if candidate_tokens
            else set()
        )
    else:
        names = {
            r["id"]
            for r in gc.query(
                "MATCH (s:StandardName) WHERE NOT coalesce(s.name_stage, '') IN ['superseded', 'exhausted', 'contested'] RETURN s.id AS id"
            )
        }
    updates: list[dict[str, services.Any]] = []
    for r in rows:
        before = r["docs"]
        new = before
        changes: list[tuple[int, int, str, dict[str, str]]] = []
        for phrase, start, end in bare_phrases[r["id"]]:
            target = candidate_id(phrase)
            linked = target in names and target != r["id"]
            replacement = f"[{phrase}](name:{target})" if linked else phrase
            changes.append(
                (
                    start,
                    end,
                    replacement,
                    {
                        "action": "linked" if linked else "unbracketed",
                        "phrase": phrase,
                        **({"target": target} if linked else {}),
                    },
                )
            )
        for link in services.find_name_links(before):
            if link.target == r["id"]:
                changes.append(
                    (
                        link.start,
                        link.end,
                        link.label,
                        {"action": "self-link removed", "phrase": link.label},
                    )
                )
        for start, end, replacement, _ in sorted(changes, reverse=True):
            new = new[:start] + replacement + new[end:]
        if new != r["docs"]:
            updates.append(
                {
                    "id": r["id"],
                    "before": before,
                    "after": new,
                    "actions": [change[3] for change in sorted(changes)],
                }
            )
    if dry_run:
        return updates
    for i in range(0, len(updates), 200):
        gc.query(
            "UNWIND $items AS it MATCH (sn:StandardName {id: it.id}) SET sn.updated_at = datetime(), sn.documentation = it.doc",
            items=[
                {"id": update["id"], "doc": update["after"]}
                for update in updates[i : i + 200]
            ],
        )
    if updates:
        services.logger.info(
            "resolve_doc_links: normalized bare [name] brackets in %d doc(s)%s",
            len(updates),
            f" (scoped to {sn_id})" if sn_id else "",
        )
    return len(updates)


def resolve_doc_links(gc: Any | None = None, *, services: Any) -> dict[str, int]:
    """Rewrite stale (name:xxx) references in documentation text.

    Scans accepted StandardName documentation for ``(name:xxx)`` markdown
    links.  When a referenced name is superseded or exhausted:

    1. **Superseded with accepted successor** (via ``REFINED_FROM`` chain):
       rewrite the link to point to the accepted successor.
    2. **Dead-end** (exhausted/superseded with no accepted successor):
       remove the hyperlink, keeping the descriptive text intact.

    Also updates the cached ``links`` list and ``link_status`` to match
    the cleaned documentation.

    Parameters
    ----------
    gc:
        Optional active GraphClient.  A new one is opened if None.

    Returns
    -------
    Dict with keys: ``resolved`` (rewritten to successor), ``removed``
    (dead link stripped), ``unchanged`` (already valid).
    """
    _own_gc = gc is None
    if _own_gc:
        gc = services.GraphClient().__enter__()
    try:
        services._normalize_bare_doc_links(gc)
        rows = list(
            gc.query(
                "\n                MATCH (sn:StandardName)\n                WHERE sn.name_stage IN ['accepted', 'approved']\n                  AND sn.documentation IS NOT NULL\n                RETURN sn.id AS id, sn.documentation AS docs, sn.links AS links\n                "
            )
        )
        if not rows:
            return {"resolved": 0, "removed": 0, "unchanged": 0}
        valid_targets = {
            r["id"]
            for r in gc.query(
                "MATCH (s:StandardName) WHERE NOT coalesce(s.name_stage, '') IN ['superseded', 'exhausted', 'contested'] RETURN s.id AS id"
            )
        }
        superseded_ids: set[str] = set()
        for row in rows:
            docs = row.get("docs") or ""
            for match in services._NAME_LINK_RE.finditer(docs):
                target = match.group(1)
                if target not in valid_targets:
                    superseded_ids.add(target)
        if not superseded_ids:
            return {"resolved": 0, "removed": 0, "unchanged": 0}
        resolution_map: dict[str, str | None] = {}
        for target in superseded_ids:
            successors = gc.query(
                "\n                MATCH (new:StandardName)-[:REFINED_FROM*]->(old:StandardName {id: $target})\n                WHERE new.name_stage IN ['accepted', 'approved']\n                RETURN new.id AS successor\n                LIMIT 1\n                ",
                target=target,
            )
            resolution_map[target] = successors[0]["successor"] if successors else None
        resolved = 0
        removed = 0
        unchanged = 0
        for row in rows:
            docs = row.get("docs") or ""
            original_docs = docs

            def _replace_link(m: re.Match) -> str:
                nonlocal resolved, removed
                target = m.group(1)
                if target in valid_targets:
                    return m.group(0)
                successor = resolution_map.get(target)
                if successor:
                    resolved += 1
                    return f"(name:{successor})"
                else:
                    removed += 1
                    return ""

            new_docs = services._NAME_LINK_RE.sub(_replace_link, docs)
            if new_docs != original_docs:
                self_link = f"name:{row['id']}"
                seen: set[str] = set()
                new_links: list[str] = []
                for m in services._NAME_LINK_RE.finditer(new_docs):
                    ref = f"name:{m.group(1)}"
                    if ref == self_link or ref in seen:
                        continue
                    seen.add(ref)
                    new_links.append(ref)
                gc.query(
                    "\n                    MATCH (sn:StandardName {id: $id})\n                    SET sn.updated_at = datetime(), sn.documentation = $docs,\n                        sn.links = $links,\n                        sn.link_status = 'resolved'\n                    ",
                    id=row["id"],
                    docs=new_docs,
                    links=new_links,
                )
            else:
                unchanged += 1
        services.logger.info(
            "resolve_doc_links: resolved=%d rewritten, %d removed, %d unchanged",
            resolved,
            removed,
            unchanged,
        )
        return {"resolved": resolved, "removed": removed, "unchanged": unchanged}
    finally:
        if _own_gc:
            gc.__exit__(None, None, None)
