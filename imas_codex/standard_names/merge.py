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
    dry_run: bool = False,
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

    changes = read_pr_changes(isnc_dir, base_ref)
    report.changes_seen = len(changes)
    if not changes:
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
                _quarantine(
                    review_target,
                    axis=change.axis,
                    score=score,
                    reason=reason,
                    gc=gc,
                )
                report.quarantined.append(
                    {"sn_id": change.sn_id, "target_id": review_target, "score": score}
                )
                report.outcomes.append(
                    MergeOutcome(
                        sn_id=change.sn_id,
                        axis=change.axis,
                        decision="quarantined",
                        target_id=review_target,
                        score=score,
                        reason=reason,
                    )
                )
        return report
    finally:
        if owns_gc:
            gc.close()
