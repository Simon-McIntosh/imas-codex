"""``sn merge`` — fold a reviewed catalog PR back into the graph-ledger.

Reads the catalog-entry diff of a reviewed ISNC pull request against a base
git ref, matches each changed entry to its graph ``StandardName`` **by id**,
and re-plays the human edit through the SAME steered-proposal path as
``sn edit`` (:func:`~imas_codex.standard_names.edit.apply_edit`): the changed
field becomes the candidate and the human intent becomes the ``reason``.

The re-attached proposal is then scored by the FULL review pipeline with
**no refine step** — a human-reviewed wording must never be silently
rewritten afterwards.  The score decides the outcome:

* ``score >= threshold`` → **ACCEPT**: the edit lands and the name reaches
  the accepted state via ``persist_reviewed_name`` / ``persist_reviewed_docs``
  (a NAME edit's accept also fires the descendant rename cascade).
* ``score <  threshold`` → **QUARANTINE + FLAG**: the existing quarantine
  signal (``validation_status='quarantined'``) is set and the proposal is
  surfaced for human attention.  It is never accepted, never refined, never
  mutated — the human's exact wording is preserved on the node.

A NAME change rides ``apply_edit``'s **rename mode**, which carries the
producing-source (``PRODUCED_NAME``) provenance through the rename cascade —
never delete-and-recreate.

The "no refine" guarantee is structural: :func:`run_merge` runs only the
review scorer and immediately transitions each proposal to *accepted* or
*quarantined*.  It never leaves a proposal in the refine-eligible
``'reviewed'`` state and never invokes any refine pool.  Attaching with
``refine=False`` additionally stamps the durable review-only marker on the
node (see :func:`~imas_codex.standard_names.edit.apply_edit`).
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from imas_codex.graph.client import GraphClient
from imas_codex.standard_names.defaults import DEFAULT_MIN_SCORE
from imas_codex.standard_names.edit import apply_edit
from imas_codex.standard_names.graph_ops import (
    persist_reviewed_docs,
    persist_reviewed_name,
)

logger = logging.getLogger(__name__)

#: Catalog fields whose change is a docs-axis edit.
_DOCS_FIELDS = ("documentation", "description")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MergeChange:
    """A single catalog-entry edit extracted from a reviewed PR diff.

    Attributes
    ----------
    sn_id:
        The graph ``StandardName`` id the edit targets — for a docs edit the
        entry's ``name``; for a rename the *old* name (the id that still
        lives in the graph).
    axis:
        ``"docs"`` (documentation/description replacement) or ``"name"``
        (a rename).
    new_value:
        The replacement documentation (docs axis) or the new name (name axis).
    old_value:
        The prior value, for the reason/audit trail.  Optional.
    """

    sn_id: str
    axis: str
    new_value: str
    old_value: str | None = None


@dataclass
class MergeOutcome:
    """Per-proposal outcome record."""

    sn_id: str
    axis: str
    decision: str  # accepted | quarantined | blocked | unmatched | planned
    target_id: str | None = None  # the reviewed node (rename successor / target)
    score: float | None = None
    reason: str = ""


@dataclass
class MergeReport:
    """Summary of a :func:`run_merge` invocation."""

    threshold: float = DEFAULT_MIN_SCORE
    dry_run: bool = False
    changes_seen: int = 0
    accepted: list[str] = field(default_factory=list)
    quarantined: list[dict[str, Any]] = field(default_factory=list)
    contested: list[dict[str, Any]] = field(default_factory=list)
    auto_approved: list[str] = field(default_factory=list)
    blocked: list[dict[str, Any]] = field(default_factory=list)
    unmatched: list[str] = field(default_factory=list)
    outcomes: list[MergeOutcome] = field(default_factory=list)


# ---------------------------------------------------------------------------
# PR diff reader
# ---------------------------------------------------------------------------


def _git(args: list[str], cwd: Path) -> str | None:
    """Run a git command in *cwd*; return stdout, or ``None`` on failure."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.warning("git %s failed: %s", args[0] if args else "?", exc)
        return None
    if result.returncode != 0:
        logger.debug("git %s: %s", args, result.stderr.strip())
        return None
    return result.stdout


def _parse_entries(text: str | None) -> dict[str, dict[str, Any]]:
    """Parse a per-domain catalog YAML list into ``{name: entry}``."""
    if not text:
        return {}
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        return {}
    if not isinstance(data, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for entry in data:
        if isinstance(entry, dict) and entry.get("name"):
            out[str(entry["name"])] = entry
    return out


def _norm(v: Any) -> str:
    return v.strip() if isinstance(v, str) else ("" if v is None else str(v))


def read_pr_changes(isnc_dir: str | Path, base_ref: str) -> list[MergeChange]:
    """Extract catalog-entry edits changed between *base_ref* and the worktree.

    Compares each changed ``standard_names/<domain>.y[a]ml`` file against its
    *base_ref* revision, matching entries by their ``name`` (the graph id):

    * an entry present in both whose ``documentation``/``description`` differs
      yields a ``docs`` :class:`MergeChange`;
    * a removed name paired 1:1 with an added name sharing the same ``unit``
      and ``kind`` (best-effort rename detection) yields a ``name``
      :class:`MergeChange`.

    Reads the *working tree* for the head side, so a reviewed PR that is
    checked out (committed or not) is compared correctly.
    """
    isnc = Path(isnc_dir)
    listing = _git(["diff", "--name-only", base_ref, "--", "standard_names"], isnc)
    if not listing:
        return []
    files = [
        line.strip()
        for line in listing.splitlines()
        if line.strip().endswith((".yml", ".yaml"))
    ]

    changes: list[MergeChange] = []
    for rel in files:
        base_text = _git(["show", f"{base_ref}:{rel}"], isnc)
        head_path = isnc / rel
        head_text = head_path.read_text() if head_path.exists() else None
        base_entries = _parse_entries(base_text)
        head_entries = _parse_entries(head_text)

        # Docs edits — same id in both sides, docs/description differs.
        for name, head in head_entries.items():
            base = base_entries.get(name)
            if base is None:
                continue
            for fld in _DOCS_FIELDS:
                new_v = _norm(head.get(fld))
                if new_v and new_v != _norm(base.get(fld)):
                    changes.append(
                        MergeChange(
                            sn_id=name,
                            axis="docs",
                            new_value=head.get(fld),
                            old_value=base.get(fld),
                        )
                    )
                    break

        # Rename edits — best-effort 1:1 pairing of removed↔added ids by
        # matching unit + kind (the fields that survive a rename).
        removed = [n for n in base_entries if n not in head_entries]
        added = [n for n in head_entries if n not in base_entries]
        for old in removed:
            b = base_entries[old]
            candidates = [
                a
                for a in added
                if _norm(head_entries[a].get("unit")) == _norm(b.get("unit"))
                and _norm(head_entries[a].get("kind")) == _norm(b.get("kind"))
            ]
            if len(candidates) == 1:
                changes.append(
                    MergeChange(
                        sn_id=old,
                        axis="name",
                        new_value=candidates[0],
                        old_value=old,
                    )
                )
    return changes


# ---------------------------------------------------------------------------
# Review scorer — FULL review, NO refine
# ---------------------------------------------------------------------------

#: Fields the review scorer needs from a StandardName node.
_REVIEW_NODE_FIELDS = (
    "id",
    "name",
    "description",
    "documentation",
    "kind",
    "unit",
    "physics_domain",
    "source_paths",
    "physical_base",
    "tags",
)


def _load_review_node(sn_id: str, gc: GraphClient) -> dict[str, Any] | None:
    """Load the fields the review scorer needs for *sn_id*."""
    rows = gc.query(
        """
        // MERGE_LOAD_REVIEW_NODE
        MATCH (sn:StandardName {id: $id})
        OPTIONAL MATCH (sn)-[:HAS_UNIT]->(u:Unit)
        RETURN sn.id AS id, sn.id AS name, sn.description AS description,
               sn.documentation AS documentation, sn.kind AS kind,
               coalesce(u.id, sn.unit) AS unit,
               sn.physics_domain AS physics_domain,
               sn.source_paths AS source_paths,
               sn.physical_base AS physical_base, sn.tags AS tags
        """,
        id=sn_id,
    )
    if not rows:
        return None
    return {k: rows[0].get(k) for k in _REVIEW_NODE_FIELDS}


def _score_proposal(
    sn_id: str,
    *,
    axis: str,
    gc: GraphClient,
    models: list[str] | None = None,
) -> float:
    """Score a merged proposal with the FULL (refine-free) review scorer.

    Runs the review pipeline's RD-quorum scorer over the single attached
    node for the configured reviewer models and returns the mean normalised
    score (0–1).  This is a pure scoring pass — it neither transitions the
    node's stage nor enters any refine pool; the accept/quarantine decision
    is owned by :func:`run_merge`.
    """
    import asyncio

    from imas_codex.settings import (
        get_sn_review_docs_models,
        get_sn_review_names_models,
    )
    from imas_codex.standard_names.review.pipeline import (
        _get_compose_context_for_review,
        _get_grammar_enums,
        _review_single_batch,
    )

    target = "docs" if axis == "docs" else "names"
    if models is None:
        models = (
            get_sn_review_docs_models()
            if axis == "docs"
            else get_sn_review_names_models()
        )
    node = _load_review_node(sn_id, gc)
    if node is None:
        return 0.0

    grammar_enums = _get_grammar_enums()
    compose_ctx = _get_compose_context_for_review()
    wlog = logging.LoggerAdapter(logger, {})

    scores: list[float] = []
    for model in (models or [])[:3]:
        try:
            result = asyncio.run(
                _review_single_batch(
                    names=[dict(node)],
                    model=model,
                    grammar_enums=grammar_enums,
                    compose_ctx=compose_ctx,
                    batch_context="sn-merge",
                    neighborhood=[],
                    audit_findings=[],
                    wlog=wlog,
                    target=target,
                )
            )
        except Exception:
            logger.debug(
                "merge review scorer failed for %s (%s)", sn_id, model, exc_info=True
            )
            continue
        items = result.get("_items", [])
        if items:
            scores.append(float(items[0].get("reviewer_score") or 0.0))

    if not scores:
        return 0.0
    return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# Accept / quarantine transitions
# ---------------------------------------------------------------------------


def _clear_claim(sn_id: str, gc: GraphClient) -> None:
    """Clear any stale claim so the accept persist's token guard matches."""
    gc.query(
        """
        // MERGE_CLEAR_CLAIM
        MATCH (sn:StandardName {id: $id})
        SET sn.claim_token = null, sn.claimed_at = null
        """,
        id=sn_id,
    )


def _accept(
    review_target: str,
    *,
    axis: str,
    score: float,
    threshold: float,
    run_id: str | None,
    gc: GraphClient,
) -> None:
    """Promote a scored proposal to the accepted state.

    Reuses the pipeline's own accept path (``persist_reviewed_name`` /
    ``persist_reviewed_docs``) so a NAME accept also fires the descendant
    rename cascade exactly as a normal reviewed rename would.
    """
    _clear_claim(review_target, gc)
    if axis == "docs":
        persist_reviewed_docs(
            sn_id=review_target,
            claim_token="",
            score=score,
            model="sn-merge",
            min_score=threshold,
            run_id=run_id,
            skip_review_node=True,
        )
    else:
        persist_reviewed_name(
            sn_id=review_target,
            claim_token="",
            score=score,
            model="sn-merge",
            min_score=threshold,
            run_id=run_id,
            skip_review_node=True,
        )


def _quarantine(
    review_target: str,
    *,
    axis: str,
    score: float,
    reason: str,
    gc: GraphClient,
) -> None:
    """Flag a below-threshold proposal for human attention.

    Sets the existing ``validation_status='quarantined'`` signal, records the
    merge reason + score, and moves the reviewed axis stage out of both the
    review (``'drafted'``) and refine (``'reviewed'``) claim windows so the
    human's wording is never re-reviewed or refined.  The wording itself is
    left untouched.
    """
    stage_set = (
        "sn.name_stage = CASE WHEN sn.name_stage IN ['drafted','reviewed'] "
        "THEN 'exhausted' ELSE sn.name_stage END,"
        if axis == "name"
        else "sn.docs_stage = CASE WHEN sn.docs_stage IN ['drafted','reviewed'] "
        "THEN 'exhausted' ELSE sn.docs_stage END,"
    )
    score_field = "reviewer_score_name" if axis == "name" else "reviewer_score_docs"
    gc.query(
        f"""
        // MERGE_QUARANTINE
        MATCH (sn:StandardName {{id: $id}})
        SET sn.validation_status = 'quarantined',
            sn.edit_status = 'rejected',
            {stage_set}
            sn.{score_field} = $score,
            sn.merge_quarantine_reason = $reason,
            sn.merge_quarantine_at = $ts
        """,
        id=review_target,
        score=score,
        reason=reason,
        ts=datetime.now(UTC).isoformat(),
    )


def _contest(
    review_target: str,
    *,
    axis: str,
    score: float,
    threshold: float,
    reason: str,
    gc: GraphClient,
) -> None:
    """Move a reviewer edit that failed the compliance re-review to 'contested'.

    A human deliberately changed the wording but the edited form did not pass
    the rubric, so it is neither published (approved) nor silently reverted:
    ``name_stage='contested'`` freezes it (pool-excluded) pending human
    adjudication via sn edit / sn approve --override / sn revert.
    """
    score_field = "reviewer_score_name" if axis == "name" else "reviewer_score_docs"
    detail = (
        f"{axis} edit failed compliance re-review "
        f"(score {score:.3f} < {threshold:.3f}): {reason}"
    )
    gc.query(
        f"""
        // MERGE_CONTEST
        MATCH (sn:StandardName {{id: $id}})
        SET sn.name_stage = 'contested',
            sn.edit_status = 'rejected',
            sn.{score_field} = $score,
            sn.contested_reason = $reason,
            sn.contested_at = $ts,
            sn.contested_resolution = null,
            sn.claim_token = null,
            sn.claimed_at = null
        """,
        id=review_target,
        score=score,
        reason=detail,
        ts=datetime.now(UTC).isoformat(),
    )


def list_contested(gc: GraphClient | None = None) -> list[dict[str, Any]]:
    """Return all names in the 'contested' stage with their failing verdict."""
    owns = gc is None
    if gc is None:
        gc = GraphClient()
    try:
        rows = gc.query(
            """
            MATCH (sn:StandardName {name_stage: 'contested'})
            RETURN sn.id AS id, sn.contested_reason AS reason,
                   sn.contested_at AS at
            ORDER BY sn.id
            """
        )
        return [dict(r) for r in (rows or [])]
    finally:
        if owns:
            gc.close()


def override_approve_contested(
    name: str, *, reason: str, gc: GraphClient | None = None
) -> bool:
    """Force a contested name to 'approved' over the rubric, on the record.

    Human authority beats the machine rubric, but only deliberately: the
    justification is stored in ``contested_resolution``.
    """
    owns = gc is None
    if gc is None:
        gc = GraphClient()
    try:
        rows = gc.query(
            """
            MATCH (sn:StandardName {id: $name, name_stage: 'contested'})
            SET sn.name_stage = 'approved',
                sn.contested_resolution = $reason,
                sn.catalog_approved_at = coalesce(sn.catalog_approved_at, datetime()),
                sn.origin = 'catalog_edit'
            RETURN sn.id AS id
            """,
            name=name,
            reason=reason,
        )
        return bool(rows)
    finally:
        if owns:
            gc.close()


def revert_contested(name: str, *, reason: str, gc: GraphClient | None = None) -> bool:
    """Drop a contested name back to 'accepted', re-opening it for a later batch."""
    owns = gc is None
    if gc is None:
        gc = GraphClient()
    try:
        rows = gc.query(
            """
            MATCH (sn:StandardName {id: $name, name_stage: 'contested'})
            SET sn.name_stage = 'accepted',
                sn.contested_resolution = $reason,
                sn.edit_status = null
            RETURN sn.id AS id
            """,
            name=name,
            reason=reason,
        )
        return bool(rows)
    finally:
        if owns:
            gc.close()


def _name_exists(sn_id: str, gc: GraphClient) -> bool:
    rows = gc.query(
        "// MERGE_MATCH_BY_ID\nMATCH (sn:StandardName {id: $id}) RETURN count(sn) AS n",
        id=sn_id,
    )
    return bool(rows and rows[0].get("n"))


def _reason_for(change: MergeChange) -> str:
    axis_word = "name" if change.axis == "name" else "documentation"
    return (
        f"human catalog PR edit — reviewer-approved {axis_word} change folded "
        "back into the ledger; score the wording as-is (do not revert to the "
        "prior text)."
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_merge(
    *,
    isnc_dir: str | Path,
    base_ref: str,
    threshold: float | None = None,
    catalog_pr_number: int | None = None,
    catalog_pr_url: str | None = None,
    catalog_merge_commit_sha: str | None = None,
    dry_run: bool = False,
    batch: list[str] | None = None,
    gc: GraphClient | None = None,
) -> MergeReport:
    """Fold a reviewed catalog PR back into the graph-ledger.

    Parameters
    ----------
    isnc_dir:
        Path to the ISNC catalog git checkout (the reviewed PR branch).
    base_ref:
        Git ref the PR is diffed against (e.g. ``origin/main``).
    threshold:
        Accept threshold in 0–1.  Defaults to
        :data:`~imas_codex.standard_names.defaults.DEFAULT_MIN_SCORE`.
    dry_run:
        When ``True``, report the planned matches only — attach/review/accept
        are all skipped and nothing is written.
    gc:
        Optional open :class:`GraphClient`.  When omitted, one is opened for
        the call.

    Returns
    -------
    MergeReport
        The accept / quarantine / blocked / unmatched breakdown.
    """
    thr = DEFAULT_MIN_SCORE if threshold is None else float(threshold)
    report = MergeReport(threshold=thr, dry_run=dry_run)

    approval_values = (
        catalog_pr_number,
        catalog_pr_url,
        catalog_merge_commit_sha,
    )
    if any(value is not None for value in approval_values) and not all(
        value is not None and value != "" for value in approval_values
    ):
        raise ValueError(
            "catalog approval requires PR number, PR URL, and merge commit SHA"
        )

    changes = read_pr_changes(isnc_dir, base_ref)
    report.changes_seen = len(changes)
    # With no edits AND no batch there is nothing to do. A batch with no edits
    # is the common case (reviewers approved as-is) — fall through to the
    # untouched auto-approve below.
    if not changes and not batch:
        return report

    owns_gc = gc is None
    if gc is None:
        gc = GraphClient()
    try:
        for change in changes:
            # ── Match by id ──────────────────────────────────────────────
            if not _name_exists(change.sn_id, gc):
                report.unmatched.append(change.sn_id)
                report.outcomes.append(
                    MergeOutcome(
                        sn_id=change.sn_id, axis=change.axis, decision="unmatched"
                    )
                )
                continue

            if dry_run:
                report.outcomes.append(
                    MergeOutcome(
                        sn_id=change.sn_id, axis=change.axis, decision="planned"
                    )
                )
                continue

            # ── Attach the human edit exactly like `sn edit` ─────────────
            reason = _reason_for(change)
            if change.axis == "name":
                plan = apply_edit(
                    target=change.sn_id,
                    rename=change.new_value,
                    reason=reason,
                    origin="human",
                    refine=False,
                    gc=gc,
                )
            else:
                plan = apply_edit(
                    target=change.sn_id,
                    docs=change.new_value,
                    reason=reason,
                    origin="human",
                    refine=False,
                    gc=gc,
                )

            if getattr(plan, "blocked", None):
                report.blocked.append({"sn_id": change.sn_id, "reason": plan.blocked})
                report.outcomes.append(
                    MergeOutcome(
                        sn_id=change.sn_id,
                        axis=change.axis,
                        decision="blocked",
                        reason=plan.blocked,
                    )
                )
                continue

            # For a rename the reviewed node is the drafted successor; for a
            # docs edit it is the target itself.
            review_target = plan.successor if change.axis == "name" else change.sn_id
            if not review_target:
                report.blocked.append(
                    {
                        "sn_id": change.sn_id,
                        "reason": "apply_edit produced no successor for rename",
                    }
                )
                report.outcomes.append(
                    MergeOutcome(
                        sn_id=change.sn_id,
                        axis=change.axis,
                        decision="blocked",
                        reason="no successor",
                    )
                )
                continue

            # ── FULL review, NO refine ───────────────────────────────────
            score = _score_proposal(review_target, axis=change.axis, gc=gc)

            # ── Accept ≥ threshold else quarantine + flag ────────────────
            if score >= thr:
                _accept(
                    review_target,
                    axis=change.axis,
                    score=score,
                    threshold=thr,
                    run_id=plan.run_id,
                    gc=gc,
                )
                if all(value is not None for value in approval_values):
                    mark_catalog_name_approved(
                        review_target,
                        catalog_pr_number=int(catalog_pr_number),
                        catalog_pr_url=str(catalog_pr_url),
                        catalog_merge_commit_sha=str(catalog_merge_commit_sha),
                        gc=gc,
                    )
                report.accepted.append(review_target)
                report.outcomes.append(
                    MergeOutcome(
                        sn_id=change.sn_id,
                        axis=change.axis,
                        decision="accepted",
                        target_id=review_target,
                        score=score,
                    )
                )
            else:
                # A reviewer edit that fails re-review is neither approved nor
                # silently reverted — it moves to the 'contested' holding state.
                _contest(
                    review_target,
                    axis=change.axis,
                    score=score,
                    threshold=thr,
                    reason=reason,
                    gc=gc,
                )
                report.contested.append(
                    {"sn_id": change.sn_id, "target_id": review_target, "score": score}
                )
                report.outcomes.append(
                    MergeOutcome(
                        sn_id=change.sn_id,
                        axis=change.axis,
                        decision="contested",
                        target_id=review_target,
                        score=score,
                        reason=reason,
                    )
                )

        # ── Untouched batch names auto-promote accepted → approved ──────
        # The human approved the batch by merging; a name the reviewers left
        # unchanged carries an implicit compliance rubber-stamp, so it is
        # promoted directly (no re-review). Only with complete PR metadata.
        if batch and not dry_run and all(v is not None for v in approval_values):
            edited_ids = {c.sn_id for c in changes}
            for nid in batch:
                if nid in edited_ids:
                    continue
                if mark_catalog_name_approved(
                    nid,
                    catalog_pr_number=int(catalog_pr_number),
                    catalog_pr_url=str(catalog_pr_url),
                    catalog_merge_commit_sha=str(catalog_merge_commit_sha),
                    gc=gc,
                ):
                    report.auto_approved.append(nid)
                    report.outcomes.append(
                        MergeOutcome(sn_id=nid, axis="name", decision="auto_approved")
                    )
        return report
    finally:
        if owns_gc:
            gc.close()


@dataclass(frozen=True)
class ResolvedPr:
    """Merged-PR metadata resolved from a PR URL via the gh CLI."""

    number: int
    url: str
    merge_commit: str
    head_ref: str
    base_ref: str


def resolve_merged_pr(pr_url: str) -> ResolvedPr:
    """Resolve a merged PR's number, merge commit, and branch refs from its URL.

    The URL is the only input the maintainer should need: the PR number, the
    merge-commit SHA (base for the content diff is ``<merge_commit>^1``), and
    the head branch (``review/<rc>`` — which locates the frozen batch artifact)
    are all recorded on the PR itself.

    Raises ValueError when gh fails, the PR is not merged, or no merge commit
    is recorded.
    """
    import json

    result = subprocess.run(
        [
            "gh",
            "pr",
            "view",
            pr_url,
            "--json",
            "number,url,state,mergeCommit,headRefName,baseRefName",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise ValueError(f"gh pr view failed: {result.stderr.strip()}")
    data = json.loads(result.stdout)
    if data.get("state") != "MERGED":
        raise ValueError(
            f"PR is not merged (state={data.get('state')}) — sn merge runs only "
            "from an accepted (merged) PR"
        )
    oid = (data.get("mergeCommit") or {}).get("oid")
    if not oid:
        raise ValueError("merged PR records no merge commit")
    return ResolvedPr(
        number=int(data["number"]),
        url=data.get("url") or pr_url,
        merge_commit=oid,
        head_ref=data.get("headRefName") or "",
        base_ref=data.get("baseRefName") or "main",
    )


@dataclass
class UndoReport:
    """Summary of an :func:`undo_merge` invocation."""

    pr_number: int = 0
    demoted: list[str] = field(default_factory=list)
    contested_reverted: list[str] = field(default_factory=list)


def undo_merge(
    *,
    pr_number: int,
    batch: list[str] | None = None,
    gc: GraphClient | None = None,
) -> UndoReport:
    """Unwind the graph promotions of a previously folded merge.

    The reverse of :func:`run_merge`'s *promotions* — a property-level revert,
    not a checkout:

    * names ``approved`` by this PR (``catalog_pr_number`` matches) drop back
      to ``accepted`` with the catalog provenance fields cleared;
    * ``contested`` names in *batch* (the frozen artifact list) drop back to
      ``accepted`` with the contested fields cleared.

    What it deliberately does NOT undo: accepted human *edits*. A merged rename
    or docs change is permanent graph history (``REFINED_FROM`` chains,
    ``DocsRevision`` snapshots) — reverting wording is a new ``sn edit``, never
    node surgery. Full-state rollback is a graph-archive restore
    (``imas-codex graph export`` / ``graph load``), the checkout analogue.
    """
    report = UndoReport(pr_number=pr_number)
    owns = gc is None
    if gc is None:
        gc = GraphClient()
    try:
        resolution = f"merge of catalog PR {pr_number} unwound"
        rows = gc.query(
            """
            MATCH (sn:StandardName {name_stage: 'approved'})
            WHERE sn.catalog_pr_number = $pr
            SET sn.name_stage = 'accepted',
                sn.catalog_pr_number = null,
                sn.catalog_pr_url = null,
                sn.catalog_merge_commit_sha = null,
                sn.catalog_approved_at = null
            RETURN sn.id AS id ORDER BY id
            """,
            pr=pr_number,
        )
        report.demoted = [r["id"] for r in (rows or [])]
        if batch:
            rows = gc.query(
                """
                MATCH (sn:StandardName {name_stage: 'contested'})
                WHERE sn.id IN $batch
                SET sn.name_stage = 'accepted',
                    sn.contested_reason = null,
                    sn.contested_at = null,
                    sn.contested_resolution = $resolution,
                    sn.edit_status = null
                RETURN sn.id AS id ORDER BY id
                """,
                batch=batch,
                resolution=resolution,
            )
            report.contested_reverted = [r["id"] for r in (rows or [])]
        return report
    finally:
        if owns:
            gc.close()


def mark_catalog_name_approved(
    name: str,
    *,
    catalog_pr_number: int,
    catalog_pr_url: str,
    catalog_merge_commit_sha: str,
    gc: GraphClient,
) -> bool:
    """Promote an accepted name using complete merged-PR evidence only."""
    if catalog_pr_number <= 0 or not catalog_pr_url or not catalog_merge_commit_sha:
        raise ValueError("complete merged catalog PR metadata is required")
    rows = gc.query(
        """
        MATCH (sn:StandardName {id: $name})
        WHERE sn.name_stage IN ['accepted', 'approved']
          AND sn.docs_stage = 'accepted'
        SET sn.name_stage = 'approved',
            sn.catalog_pr_number = $pr_number,
            sn.catalog_pr_url = $pr_url,
            sn.catalog_merge_commit_sha = $merge_commit,
            sn.catalog_approved_at = coalesce(sn.catalog_approved_at, datetime()),
            sn.origin = 'catalog_edit'
        RETURN sn.id AS id
        """,
        name=name,
        pr_number=catalog_pr_number,
        pr_url=catalog_pr_url,
        merge_commit=catalog_merge_commit_sha,
    )
    return bool(rows)


# ---------------------------------------------------------------------------
# The fold-back receipt — a version tag on the merge commit
# ---------------------------------------------------------------------------
#
# Merging the catalog PR is durably recorded by GitHub; folding it back into the
# graph-ledger was recorded nowhere durable, so a merged-but-not-folded release
# looked identical to a folded one. The receipt closes that gap: after a
# successful fold-back the merge commit is tagged with a deterministic contract
# block whose presence means "catalog and graph are in sync for this version".
# A grounded human summary is appended below the block; it is never parsed and
# never blocks the fold-back.

#: First token of the machine-readable contract line. A tag whose message
#: starts with this marker certifies that its version has been folded back.
CONTRACT_MARKER = "graph-merged:"

#: Separates the deterministic contract block from the human prose below it.
_NOTES_SEPARATOR = "---"


@dataclass
class FoldBackTagReport:
    """Outcome of writing the fold-back receipt tag."""

    tag: str = ""
    created: bool = False
    pushed: bool = False
    notes_included: bool = False
    error: str | None = None


def _git_cp(args: list[str], cwd: str | Path) -> subprocess.CompletedProcess[str]:
    """Run git in *cwd* returning the full result (returncode + stderr).

    Distinct from :func:`_git` (which swallows failures to ``None``): tag
    creation and pushes need the return code and stderr to report failures.
    """
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=30,
    )


def merge_tag_name(head_ref: str) -> str | None:
    """Derive the version tag from a PR's head branch.

    Both the batch flow (``review/<rc>``) and the locked plain-final flow
    (``release/<version>``) name the branch after the version being released,
    so the tag is exactly the branch suffix. Any other branch yields ``None``
    (there is no version to certify).
    """
    for prefix in ("review/", "release/"):
        if head_ref.startswith(prefix):
            return head_ref[len(prefix) :].strip("/") or None
    return None


def build_contract_block(
    *,
    pr_number: int | None,
    pr_url: str | None,
    batch_artifact: str | None,
    report: MergeReport,
    timestamp: str | None = None,
) -> str:
    """The deterministic, machine-readable contract lines.

    Line 1 is ``graph-merged: <iso-ts>`` — the idempotency guard parses only
    this. The remaining lines carry the PR reference, the frozen batch artifact,
    and the fold-back outcome counts for the human record.
    """
    ts = timestamp or datetime.now(UTC).isoformat()
    return "\n".join(
        [
            f"{CONTRACT_MARKER} {ts}",
            f"pr: #{pr_number} {pr_url}",
            f"batch: {batch_artifact or '(none)'}",
            (
                f"outcomes: approved={len(report.accepted)} "
                f"auto_approved={len(report.auto_approved)} "
                f"contested={len(report.contested)}"
            ),
        ]
    )


def build_merge_tag_message(contract_block: str, notes: str = "") -> str:
    """Assemble the tag message: contract block first, prose (if any) below.

    The contract block is always the head of the message so the idempotency
    check reads a stable prefix regardless of whether a summary was written.
    """
    if notes and notes.strip():
        return f"{contract_block}\n\n{_NOTES_SEPARATOR}\n\n{notes.strip()}"
    return contract_block


def has_contract_tag(isnc_dir: str | Path, tag: str) -> bool:
    """True when *tag* exists locally and carries the fold-back contract.

    Reads the annotated tag's message; a tag whose message begins with the
    contract marker certifies the version has already been folded back.
    """
    contents = _git(["tag", "-l", tag, "--format=%(contents)"], Path(isnc_dir))
    return bool(contents) and contents.lstrip().startswith(CONTRACT_MARKER)


def create_fold_back_tag(
    isnc_dir: str | Path,
    *,
    tag: str,
    merge_commit: str,
    message: str,
    remote: str,
) -> tuple[bool, str | None]:
    """Create the annotated tag on the merge commit and push it to *remote*.

    On a push failure the local tag is rolled back so the repo never carries a
    local receipt with no remote counterpart. Returns ``(ok, error)``.
    """
    isnc = Path(isnc_dir)
    made = _git_cp(["tag", "-a", tag, merge_commit, "-m", message], isnc)
    if made.returncode != 0:
        return False, f"failed to create tag {tag}: {made.stderr.strip()}"
    pushed = _git_cp(["push", remote, tag], isnc)
    if pushed.returncode != 0:
        _git_cp(["tag", "-d", tag], isnc)  # roll back the local tag
        return False, f"failed to push tag {tag} to {remote}: {pushed.stderr.strip()}"
    return True, None


def delete_fold_back_tag(
    isnc_dir: str | Path, *, tag: str, remote: str
) -> tuple[bool, str | None]:
    """Delete the fold-back tag locally and on *remote* (the ``--undo`` inverse).

    A missing local tag is not an error (undo may run after a fresh checkout);
    a remote-delete failure is reported. Returns ``(ok, error)``.
    """
    isnc = Path(isnc_dir)
    errors: list[str] = []
    local = _git_cp(["tag", "-d", tag], isnc)
    if local.returncode != 0 and "not found" not in local.stderr.lower():
        errors.append(local.stderr.strip())
    remote_del = _git_cp(["push", remote, "--delete", tag], isnc)
    if remote_del.returncode != 0:
        errors.append(remote_del.stderr.strip())
    return (not errors), ("; ".join(e for e in errors if e) or None)


def resolve_tag_remote(
    isnc_dir: str | Path, pr_url: str, *, default: str = "origin"
) -> str:
    """The checkout remote whose github repo matches the PR URL's owner/repo.

    The receipt tag is pushed to the PR's target repo — the fork for a batch RC,
    upstream for a final. Both are remotes of the ISNC checkout, so match the
    PR URL's ``owner/repo`` against each remote's github slug.
    """
    m = re.search(r"github\.com[:/]([\w.-]+)/([\w.-]+?)(?:\.git)?/pull/", pr_url)
    if not m:
        return default
    want = (m[1], m[2])

    from imas_codex.standard_names.catalog_release import _github_slug

    for remote in ("upstream", "origin"):
        if _github_slug(Path(isnc_dir), remote) == want:
            return remote
    return default


def fetch_pr_evidence(pr_url: str) -> dict[str, Any]:
    """Gather the merge summary's evidence from the PR itself, via ``gh``.

    One call returns the PR description, the full conversation (comments +
    reviews), and the commit list (whose first entry locates the review-delta
    base). Never raises — a gh failure returns ``{}`` so the summary degrades
    to the deterministic block alone.
    """
    import json

    try:
        result = subprocess.run(
            [
                "gh",
                "pr",
                "view",
                pr_url,
                "--json",
                "body,comments,reviews,commits",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.warning("gh pr view (evidence) failed: %s", exc)
        return {}
    if result.returncode != 0:
        logger.warning("gh pr view (evidence) failed: %s", result.stderr.strip())
        return {}
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def review_delta_diff(
    isnc_dir: str | Path,
    *,
    base_oid: str | None,
    merge_commit: str | None,
    max_chars: int = 20000,
) -> str:
    """The diff of what reviewers changed, scoped to ``standard_names/``.

    Compares the PR's original content (``base_oid``) against the merged state
    (``merge_commit``); truncated to *max_chars* to keep the prompt bounded.
    Returns ``""`` when either ref is missing or the diff is empty.
    """
    if not base_oid or not merge_commit:
        return ""
    out = _git(["diff", base_oid, merge_commit, "--", "standard_names"], Path(isnc_dir))
    return (out or "")[:max_chars]


def _conversation_from_evidence(evidence: dict[str, Any]) -> list[dict[str, str]]:
    """Flatten PR comments + reviews into ``{author, kind, body}`` records."""
    out: list[dict[str, str]] = []
    for comment in evidence.get("comments") or []:
        body = (comment.get("body") or "").strip()
        if body:
            out.append(
                {
                    "author": (comment.get("author") or {}).get("login", ""),
                    "kind": "comment",
                    "body": body,
                }
            )
    for review in evidence.get("reviews") or []:
        body = (review.get("body") or "").strip()
        if body:
            out.append(
                {
                    "author": (review.get("author") or {}).get("login", ""),
                    "kind": f"review ({review.get('state', '')})".strip(),
                    "body": body,
                }
            )
    return out


def _commit_messages_from_evidence(evidence: dict[str, Any]) -> list[str]:
    """Every commit message that went into the PR (headline + body)."""
    out: list[str] = []
    for commit in evidence.get("commits") or []:
        headline = (commit.get("messageHeadline") or "").strip()
        body = (commit.get("messageBody") or "").strip()
        msg = f"{headline}\n{body}".strip()
        if msg:
            out.append(msg)
    return out


def _default_merge_notes(**kwargs: Any) -> str:
    """Bind the grounded merge-summary synthesizer (lazy import)."""
    from imas_codex.standard_names.release_notes import build_merge_notes

    return build_merge_notes(**kwargs)


def tag_fold_back(
    *,
    isnc_dir: str | Path,
    head_ref: str,
    merge_commit: str,
    pr_number: int | None,
    pr_url: str | None,
    batch_artifact: str | None,
    report: MergeReport,
    remote: str,
    include_notes: bool = True,
    pr_evidence: dict[str, Any] | None = None,
    notes_builder: Any | None = None,
    timestamp: str | None = None,
) -> FoldBackTagReport:
    """Write the fold-back receipt after a successful non-dry merge.

    Builds the deterministic contract block, optionally appends a grounded human
    summary synthesized from the PR (description + conversation + commit messages
    + review-delta diff), then tags the merge commit and pushes it to *remote*.

    A notes-synthesis failure never blocks the fold-back — the tag is written
    with the deterministic block alone. ``pr_evidence`` / ``notes_builder`` are
    injectable so the flow is testable with no live GitHub and no live LLM.
    """
    out = FoldBackTagReport()
    tag = merge_tag_name(head_ref)
    if not tag:
        out.error = f"cannot derive a version tag from head ref {head_ref!r}"
        return out
    out.tag = tag

    contract = build_contract_block(
        pr_number=pr_number,
        pr_url=pr_url,
        batch_artifact=batch_artifact,
        report=report,
        timestamp=timestamp,
    )

    notes = ""
    if include_notes:
        evidence = (
            pr_evidence if pr_evidence is not None else fetch_pr_evidence(pr_url or "")
        )
        commits = evidence.get("commits") or []
        base_oid = commits[0].get("oid") if commits else None
        delta = review_delta_diff(
            isnc_dir, base_oid=base_oid, merge_commit=merge_commit
        )
        builder = notes_builder or _default_merge_notes
        try:
            notes = (
                builder(
                    pr_description=evidence.get("body") or "",
                    conversation=_conversation_from_evidence(evidence),
                    commit_messages=_commit_messages_from_evidence(evidence),
                    review_delta=delta,
                )
                or ""
            )
        except Exception:
            logger.warning(
                "merge-notes builder raised — writing the deterministic tag block "
                "alone",
                exc_info=True,
            )
            notes = ""

    message = build_merge_tag_message(contract, notes)
    ok, err = create_fold_back_tag(
        isnc_dir,
        tag=tag,
        merge_commit=merge_commit,
        message=message,
        remote=remote,
    )
    out.created = ok
    out.pushed = ok
    out.notes_included = bool(notes and notes.strip())
    out.error = err
    return out
