"""Budgeted, resumable documentation-refinement campaigns for accepted names.

A campaign selects accepted standard names by a *defect predicate* — banned
prose in their documentation (typical values, estimator recipes, procedural
padding) and/or deterministic audit findings — materialises a reviewed
manifest, then drains the selection through the NORMAL docs pools in fixed,
budgeted batches.

Each batch:

1. lifts audit quarantine on its members so the fix can become visible
   (the docs pools skip quarantined names — an accepted-but-quarantined name
   would otherwise stay invisible after refinement);
2. snapshots every member's current docs to a :class:`DocsRevision` and resets
   the docs pipeline, stamping a fresh scope ``run_id`` — reusing the same
   mechanism the family-curative wave uses (:func:`mark_members_for_regen`),
   so every refinement is reversible;
3. drains ONLY that scope through the six-pool orchestrator
   (``sn run --docs-only --flush --scope-run-id <id>``) via an injected
   callable, so this module carries no pool machinery;
4. re-validates the touched names against the banned-prose policy — a name
   whose refreshed docs still trip the grep-audit is re-quarantined, a clean
   name is confirmed valid;
5. records a campaign change event per touched name for traceability;
6. is checked against a convergence gate — acceptance below threshold,
   banned prose reintroduced, or name-identity drift halts the campaign for
   root-causing rather than churning.

Nothing here composes new names or renames anything: docs campaigns never
touch name identity (renames stay individual ``sn edit`` decisions), and there
is no full-catalog regeneration mode. All accepts drain through the normal
review pools; the campaign only selects, snapshots, and gates.

The selection logic is pure (:func:`match_target`) so it is unit-testable
without a graph, and every graph interaction in the runner is an injectable
callable so the orchestration is testable against a mock graph.
"""

from __future__ import annotations

import json
import logging
import random
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from imas_codex.standard_names.prose_policy import (
    BANNED_PROSE_PATTERNS,
    banned_prose_findings,
)

logger = logging.getLogger(__name__)

# ── Predicate vocabulary ────────────────────────────────────────────────────
#
# A campaign spec is a comma-separated list of predicate tokens:
#
#   prose                 → any banned-prose class in the documentation
#   prose:<class>         → one banned-prose class (typical_values,
#                           estimator_recipe, procedural_padding)
#   audit                 → carries any deterministic audit finding
#   audit:<substr>        → an audit finding whose check name contains <substr>
#                           (e.g. audit:decomposition, audit:latex,
#                           audit:consistency)
#   quarantined           → accepted but validation_status='quarantined',
#                           regardless of which finding quarantined it
#   all                   → prose + audit + quarantined
#
# The banned-prose classes are the single source of truth in
# ``prose_policy.BANNED_PROSE_PATTERNS`` — the same grep-audit the docs seat
# is benchmarked against — so selection and evaluation never diverge.

PROSE_CLASSES: tuple[str, ...] = tuple(BANNED_PROSE_PATTERNS.keys())

_PROSE = "prose"
_AUDIT = "audit"
_QUARANTINED = "quarantined"
_ALL = "all"
_AUDIT_TAG_PREFIX = "audit:"


@dataclass(frozen=True)
class CampaignSpec:
    """A parsed campaign defect-predicate selector.

    ``prose_classes`` is a subset of :data:`PROSE_CLASSES` (empty → no prose
    predicate).  ``audit_categories`` holds substrings matched against audit
    check names; the empty string ``""`` means "any audit finding".
    ``include_quarantined`` selects accepted-but-quarantined names on their
    own.  At least one predicate is always active after :meth:`parse`.
    """

    prose_classes: tuple[str, ...] = ()
    audit_categories: tuple[str, ...] = ()
    include_quarantined: bool = False
    raw: str = ""

    @classmethod
    def parse(cls, spec: str) -> CampaignSpec:
        """Parse a spec string into a :class:`CampaignSpec`.

        Raises :class:`ValueError` on an unknown token or an unknown prose
        class, so a typo fails loudly rather than selecting nothing.
        """
        if not spec or not spec.strip():
            raise ValueError("campaign spec is empty")

        prose: set[str] = set()
        audit: set[str] = set()
        quarantined = False

        for token in (t.strip() for t in spec.split(",")):
            if not token:
                continue
            low = token.lower()
            if low == _ALL:
                prose.update(PROSE_CLASSES)
                audit.add("")
                quarantined = True
            elif low == _PROSE:
                prose.update(PROSE_CLASSES)
            elif low.startswith(f"{_PROSE}:"):
                cls_name = low.split(":", 1)[1]
                if cls_name not in BANNED_PROSE_PATTERNS:
                    raise ValueError(
                        f"unknown prose class {cls_name!r}; valid: "
                        f"{', '.join(PROSE_CLASSES)}"
                    )
                prose.add(cls_name)
            elif low == _AUDIT:
                audit.add("")
            elif low.startswith(f"{_AUDIT}:"):
                audit.add(low.split(":", 1)[1])
            elif low == _QUARANTINED:
                quarantined = True
            else:
                raise ValueError(
                    f"unknown campaign predicate {token!r}; valid tokens: "
                    "prose[:class], audit[:substr], quarantined, all"
                )

        if not prose and not audit and not quarantined:
            raise ValueError(f"campaign spec {spec!r} selects no predicate")

        return cls(
            prose_classes=tuple(sorted(prose)),
            audit_categories=tuple(sorted(audit)),
            include_quarantined=quarantined,
            raw=spec,
        )

    def describe(self) -> str:
        """A short human description of what this spec selects."""
        parts: list[str] = []
        if self.prose_classes:
            parts.append("banned prose (" + ", ".join(self.prose_classes) + ")")
        for cat in self.audit_categories:
            parts.append("any audit finding" if cat == "" else f"audit:{cat}")
        if self.include_quarantined:
            parts.append("audit-quarantined")
        return "; ".join(parts)


# ── Selection ───────────────────────────────────────────────────────────────


@dataclass
class CampaignTarget:
    """One accepted name matched by a campaign spec, with defect evidence."""

    id: str
    name: str
    matched_predicates: dict[str, list[str]] = field(default_factory=dict)
    quarantined: bool = False
    physics_domain: str = ""

    @property
    def predicate_keys(self) -> list[str]:
        return sorted(self.matched_predicates)


@dataclass
class CampaignSelection:
    """The full set of accepted names a campaign spec selects."""

    spec: CampaignSpec
    targets: list[CampaignTarget] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.targets)

    @property
    def per_predicate(self) -> dict[str, int]:
        counter: Counter[str] = Counter()
        for t in self.targets:
            for key in t.matched_predicates:
                counter[key] += 1
        return dict(sorted(counter.items()))

    @property
    def quarantined_count(self) -> int:
        return sum(1 for t in self.targets if t.quarantined)

    @property
    def per_domain(self) -> dict[str, int]:
        counter: Counter[str] = Counter(
            t.physics_domain or "(none)" for t in self.targets
        )
        return dict(sorted(counter.items()))

    @property
    def ids(self) -> list[str]:
        return [t.id for t in self.targets]


def _audit_check_names(validation_issues: Any) -> list[str]:
    """Extract audit check names from a ``validation_issues`` value.

    Each issue is a tagged string ``"audit:<check_name>: <detail>"``; returns
    the ``<check_name>`` for each audit-tagged issue.
    """
    if not validation_issues:
        return []
    checks: list[str] = []
    for issue in validation_issues:
        text = str(issue)
        if text.startswith(_AUDIT_TAG_PREFIX):
            rest = text[len(_AUDIT_TAG_PREFIX) :]
            checks.append(rest.split(":", 1)[0].strip())
    return checks


def match_target(row: dict[str, Any], spec: CampaignSpec) -> CampaignTarget | None:
    """Match one accepted-name row against a spec — pure, no graph.

    ``row`` needs ``id``, ``name``, ``description``, ``documentation``,
    ``validation_issues`` (list) and ``validation_status``.  Returns a
    :class:`CampaignTarget` with per-predicate evidence when at least one
    predicate matches, else ``None``.
    """
    matched: dict[str, list[str]] = {}
    quarantined = (row.get("validation_status") or "") == "quarantined"

    # Banned-prose predicate — grep the combined description + documentation.
    if spec.prose_classes:
        text = f"{row.get('description') or ''}\n{row.get('documentation') or ''}"
        findings = banned_prose_findings(text)
        for cls_name in spec.prose_classes:
            count = findings.get(cls_name, 0)
            if count > 0:
                matched[f"prose:{cls_name}"] = [f"{count} match(es)"]

    # Audit-finding predicate — match check names against requested categories.
    if spec.audit_categories:
        checks = _audit_check_names(row.get("validation_issues"))
        for cat in spec.audit_categories:
            hits = [c for c in checks if cat in c] if cat else list(checks)
            if hits:
                key = "audit" if cat == "" else f"audit:{cat}"
                matched[key] = sorted(set(hits))

    # Quarantine predicate — selects on its own.
    if spec.include_quarantined and quarantined:
        matched.setdefault(
            "quarantined", [str(row.get("quarantine_reason") or "quarantined")]
        )

    if not matched:
        return None

    return CampaignTarget(
        id=row["id"],
        name=row.get("name") or row["id"],
        matched_predicates=matched,
        quarantined=quarantined,
        physics_domain=row.get("physics_domain") or "",
    )


_SELECT_QUERY = """
MATCH (sn:StandardName)
WHERE sn.name_stage = 'accepted'
RETURN sn.id AS id,
       coalesce(sn.name, sn.id) AS name,
       sn.description AS description,
       sn.documentation AS documentation,
       sn.validation_issues AS validation_issues,
       sn.validation_status AS validation_status,
       sn.quarantine_reason AS quarantine_reason,
       sn.physics_domain AS physics_domain,
       sn.docs_stage AS docs_stage
ORDER BY sn.id
"""


def select_targets(
    gc: Any,
    spec: CampaignSpec,
    *,
    limit: int | None = None,
) -> CampaignSelection:
    """Read-only selection of accepted names matching *spec*.

    Performs no pipeline mutation — safe to run at any time, including as the
    ``--dry-run`` manifest generator.
    """
    rows = gc.query(_SELECT_QUERY) or []
    targets: list[CampaignTarget] = []
    for row in rows:
        target = match_target(dict(row), spec)
        if target is not None:
            targets.append(target)
            if limit is not None and len(targets) >= limit:
                break
    return CampaignSelection(spec=spec, targets=targets)


def stratified_pilot(
    selection: CampaignSelection,
    n: int,
    *,
    seed: int = 0,
) -> CampaignSelection:
    """Reduce *selection* to a deterministic stratified pilot of *n* targets.

    Round-robins across physics domains so every domain is represented before
    any is doubled up; within a domain, rotates across defect (predicate)
    classes so the pilot carries a mixed defect profile.  Deterministic for a
    given selection and seed, so a ``--dry-run`` pilot manifest and the
    subsequent live pilot run pick exactly the same members.
    """
    if n <= 0:
        raise ValueError("pilot size must be positive")
    if n >= selection.total:
        return selection

    rng = random.Random(seed)
    # domain → defect class → shuffled targets
    domains: dict[str, dict[str, list[CampaignTarget]]] = {}
    for t in selection.targets:
        cls = t.predicate_keys[0] if t.predicate_keys else ""
        domains.setdefault(t.physics_domain or "", {}).setdefault(cls, []).append(t)
    for classes in domains.values():
        for bucket in classes.values():
            rng.shuffle(bucket)

    domain_keys = sorted(domains)
    class_cursor = dict.fromkeys(domain_keys, 0)
    picked: list[CampaignTarget] = []
    while len(picked) < n:
        progressed = False
        for dom in domain_keys:
            classes = domains[dom]
            keys = sorted(k for k, bucket in classes.items() if bucket)
            if not keys:
                continue
            key = keys[class_cursor[dom] % len(keys)]
            class_cursor[dom] += 1
            picked.append(classes[key].pop())
            progressed = True
            if len(picked) >= n:
                break
        if not progressed:
            break
    return CampaignSelection(spec=selection.spec, targets=picked)


# ── Manifest ─────────────────────────────────────────────────────────────────


def build_manifest(
    selection: CampaignSelection,
    *,
    sample_size: int = 20,
    batch_size: int = 100,
    seed: int = 0,
    pilot_from: int | None = None,
) -> dict[str, Any]:
    """Materialise a reviewable manifest for a selection.

    The manifest carries the total count, the per-predicate and per-domain
    breakdowns, the batch plan, and a deterministic random sample (default 20)
    of matched names with their matched-defect evidence — the object the lead
    approves before a live campaign runs.  For a pilot selection pass
    ``pilot_from`` = the size of the full selection it was drawn from; the
    manifest then carries a ``pilot`` marker.
    """
    rng = random.Random(seed)
    sample_targets = list(selection.targets)
    rng.shuffle(sample_targets)
    sample = [
        {
            "id": t.id,
            "name": t.name,
            "matched_predicates": t.matched_predicates,
            "quarantined": t.quarantined,
            "physics_domain": t.physics_domain,
        }
        for t in sample_targets[:sample_size]
    ]
    n_batches = (selection.total + batch_size - 1) // batch_size if batch_size else 0
    manifest = {
        "spec": selection.spec.raw,
        "spec_describe": selection.spec.describe(),
        "generated_at": datetime.now(UTC).isoformat(),
        "total": selection.total,
        "per_predicate": selection.per_predicate,
        "per_domain": selection.per_domain,
        "quarantined_count": selection.quarantined_count,
        "batch_plan": {"batch_size": batch_size, "n_batches": n_batches},
        "sample": sample,
    }
    if pilot_from is not None:
        manifest["pilot"] = {"n": selection.total, "from_total": pilot_from}
    return manifest


def write_manifest(manifest: dict[str, Any], path: str | Path) -> Path:
    """Write *manifest* as pretty JSON to *path* and return the path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n")
    return out


# ── Batching & budget ────────────────────────────────────────────────────────


@dataclass
class CampaignBudget:
    """Batch sizing and cost ceilings for a campaign.

    ``per_batch_cost_cap`` is passed to the scoped drain as its pool
    ``cost_limit``; ``campaign_cost_ceiling`` halts the campaign between
    batches once cumulative measured spend would exceed it.
    """

    batch_size: int = 100
    per_batch_cost_cap: float = 10.0
    campaign_cost_ceiling: float = 100.0


def plan_batches(ids: Sequence[str], batch_size: int) -> list[list[str]]:
    """Split *ids* into fixed-size batches (last batch may be short)."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [list(ids[i : i + batch_size]) for i in range(0, len(ids), batch_size)]


# ── Convergence gate ──────────────────────────────────────────────────────────


@dataclass
class ConvergenceThresholds:
    """Per-batch convergence gate (anti-churn).

    Defaults: at least 90% of touched docs accepted within the rotation cap,
    zero banned-prose reintroduction, and zero name-axis drift (docs-only
    campaigns never change name identity).
    """

    min_docs_accept_rate: float = 0.90
    max_banned_prose_reintroduced: int = 0
    allow_name_drift: bool = False


@dataclass
class BatchOutcome:
    """Measured result of one drained batch."""

    batch_index: int
    requested: int
    touched: int
    accepted: int
    reintroduced_ids: list[str] = field(default_factory=list)
    name_drift: list[str] = field(default_factory=list)
    cost: float = 0.0
    # Deterministic-audit deltas from re-stamping the refreshed batch: how many
    # members a fresh ISN audit re-quarantined (genuine defect) vs cleared.
    audit_requarantined: int = 0
    audit_cleared: int = 0
    # Docs the prose grep flagged but the LLM adjudicator cleared as legitimate
    # definitional writing (recorded for review, do NOT count as reintroduction).
    prose_adjudicated_clear: list[str] = field(default_factory=list)

    @property
    def accept_rate(self) -> float:
        return self.accepted / self.touched if self.touched else 1.0

    @property
    def banned_prose_reintroduced(self) -> int:
        return len(self.reintroduced_ids)


def evaluate_convergence(
    outcome: BatchOutcome,
    thresholds: ConvergenceThresholds,
) -> tuple[bool, list[str]]:
    """Return ``(converged, reasons)`` for a batch outcome.

    ``reasons`` is empty when the batch converged and otherwise lists every
    failed threshold — the campaign halts on the first non-converged batch.
    """
    reasons: list[str] = []
    if outcome.accept_rate < thresholds.min_docs_accept_rate:
        reasons.append(
            f"docs acceptance {outcome.accept_rate:.2%} < "
            f"{thresholds.min_docs_accept_rate:.2%} "
            f"({outcome.accepted}/{outcome.touched} touched)"
        )
    if outcome.banned_prose_reintroduced > thresholds.max_banned_prose_reintroduced:
        reasons.append(
            f"banned prose reintroduced in {outcome.banned_prose_reintroduced} "
            f"doc(s) (cap {thresholds.max_banned_prose_reintroduced}): "
            + ", ".join(outcome.reintroduced_ids[:10])
        )
    if outcome.name_drift and not thresholds.allow_name_drift:
        reasons.append(
            f"name-axis drift on {len(outcome.name_drift)} name(s) in a "
            "docs-only campaign: " + ", ".join(outcome.name_drift[:10])
        )
    return (not reasons), reasons


def measure_batch(
    refreshed_rows: list[dict[str, Any]],
    batch_ids: Sequence[str],
    *,
    batch_index: int,
    cost: float = 0.0,
    adjudicate: Callable[[Sequence[tuple[str, str, dict[str, int]]]], list[bool]]
    | None = None,
) -> BatchOutcome:
    """Compute a :class:`BatchOutcome` from post-drain graph rows — pure.

    ``refreshed_rows`` needs ``id``, ``name_stage``, ``docs_stage``,
    ``description`` and ``documentation`` for each still-present member.  A
    batch id absent from the rows, or present with a non-accepted name stage,
    counts as name-axis drift.

    The banned-prose grep is a cheap, over-inclusive pre-filter.  A doc it flags
    is a *candidate* reintroduction; when ``adjudicate`` is supplied, the flagged
    candidates are passed to a light LLM that decides which are genuine banned
    prose versus legitimate definitional writing (mathematical definitions,
    provenance/derivation among catalogued quantities, taxonomy links).  Only
    genuine flags count as reintroduction and can halt the gate; the rest are
    recorded in ``prose_adjudicated_clear`` for review.  Without ``adjudicate``
    every grep flag counts as reintroduction (grep-only fallback).
    """
    by_id = {r["id"]: r for r in refreshed_rows}
    accepted = 0
    flagged: list[tuple[str, str, dict[str, int]]] = []
    drift: list[str] = []
    touched = 0

    for sid in batch_ids:
        row = by_id.get(sid)
        if row is None:
            drift.append(sid)
            continue
        if (row.get("name_stage") or "") != "accepted":
            drift.append(sid)
            continue
        touched += 1
        if (row.get("docs_stage") or "") == "accepted":
            accepted += 1
        text = f"{row.get('description') or ''}\n{row.get('documentation') or ''}"
        findings = banned_prose_findings(text)
        if any(v > 0 for v in findings.values()):
            flagged.append((sid, text, findings))

    if flagged and adjudicate is not None:
        verdicts = adjudicate(flagged)
        reintroduced = [
            sid
            for (sid, _, _), genuine in zip(flagged, verdicts, strict=True)
            if genuine
        ]
        adjudicated_clear = [
            sid
            for (sid, _, _), genuine in zip(flagged, verdicts, strict=True)
            if not genuine
        ]
    else:
        reintroduced = [sid for sid, _, _ in flagged]
        adjudicated_clear = []

    return BatchOutcome(
        batch_index=batch_index,
        requested=len(batch_ids),
        touched=touched,
        accepted=accepted,
        reintroduced_ids=reintroduced,
        name_drift=drift,
        cost=cost,
        prose_adjudicated_clear=adjudicated_clear,
    )


# ── Default graph operations (injectable) ─────────────────────────────────────


def default_clear_quarantine(gc: Any, ids: Sequence[str]) -> int:
    """Lift audit quarantine on accepted names so the docs pools can claim them.

    The docs review/accept path excludes ``validation_status='quarantined'``,
    so an accepted-but-quarantined name would never surface a refined doc.
    Reset to ``'pending'`` (an honest transient) before the drain; the
    post-drain re-validation restores 'valid' or re-quarantines.
    """
    if not ids:
        return 0
    rows = gc.query(
        """
        UNWIND $ids AS sid
        MATCH (sn:StandardName {id: sid})
        WHERE sn.name_stage = 'accepted'
          AND coalesce(sn.validation_status, '') = 'quarantined'
        SET sn.validation_status = 'pending',
            sn.quarantine_reason = null
        RETURN count(sn) AS n
        """,
        ids=list(ids),
    )
    return int(rows[0]["n"]) if rows else 0


_REFRESH_QUERY = """
UNWIND $ids AS sid
MATCH (sn:StandardName {id: sid})
RETURN sn.id AS id,
       coalesce(sn.name, sn.id) AS name,
       sn.name_stage AS name_stage,
       sn.docs_stage AS docs_stage,
       sn.description AS description,
       sn.documentation AS documentation
"""


def default_fetch_refreshed(gc: Any, ids: Sequence[str]) -> list[dict[str, Any]]:
    """Read post-drain state for the batch members."""
    if not ids:
        return []
    rows = gc.query(_REFRESH_QUERY, ids=list(ids)) or []
    return [dict(r) for r in rows]


def default_revalidate(
    gc: Any,
    reintroduced_ids: Sequence[str],
    clean_ids: Sequence[str],
) -> dict[str, int]:
    """Re-quarantine names whose docs still trip the policy; confirm the rest.

    Keeps campaign fixes visible: a name lifted from quarantine whose refined
    docs are clean is confirmed ``'valid'``; one that still carries banned
    prose is re-quarantined so the defect stays tracked.
    """
    requarantined = 0
    confirmed = 0
    if reintroduced_ids:
        rows = gc.query(
            """
            UNWIND $ids AS sid
            MATCH (sn:StandardName {id: sid})
            SET sn.validation_status = 'quarantined',
                sn.quarantine_reason = 'campaign: banned prose persisted after refine'
            RETURN count(sn) AS n
            """,
            ids=list(reintroduced_ids),
        )
        requarantined = int(rows[0]["n"]) if rows else 0
    if clean_ids:
        rows = gc.query(
            """
            UNWIND $ids AS sid
            MATCH (sn:StandardName {id: sid})
            WHERE coalesce(sn.validation_status, '') <> 'quarantined'
            SET sn.validation_status = 'valid'
            RETURN count(sn) AS n
            """,
            ids=list(clean_ids),
        )
        confirmed = int(rows[0]["n"]) if rows else 0
    return {"requarantined": requarantined, "confirmed": confirmed}


def _run_scoped_validation_drain(ids: Sequence[str]) -> dict[str, Any]:
    """Bridge to the LLM-free scoped ISN audit drain (lazy import + run).

    Isolated so the campaign default can wire the real drain while tests inject
    a fake, keeping this module free of a hard dependency on the worker/graph
    machinery at import time.
    """
    from imas_codex.cli.utils import run_async
    from imas_codex.standard_names.workers import drain_validation_for_ids

    return run_async(drain_validation_for_ids(ids))


def default_audit_revalidate(
    gc: Any,
    ids: Sequence[str],
    *,
    drain_fn: Callable[[Sequence[str]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Clear the validation stamp on *ids* then re-run the deterministic ISN
    audit scoped to them, re-stamping validation status.

    The fast banned-prose grep confirms 'valid' too eagerly — it never re-runs
    the deterministic checks (latex definitions, spelling, length, unit
    consistency), so a refreshed doc carrying a genuine defect would wash to
    'valid'. This clears ``validated_at`` on every batch member (so the scoped
    drain re-claims them) and runs the full audit; a member whose fresh audit
    finds a defect re-quarantines instead. Returns ``{"cleared": n,
    "requarantined_ids": [...], "valid_ids": [...]}``.
    """
    ids = list(ids)
    if not ids:
        return {"cleared": 0, "requarantined_ids": [], "valid_ids": []}
    rows = gc.query(
        """
        UNWIND $ids AS sid
        MATCH (sn:StandardName {id: sid})
        SET sn.validated_at = null, sn.claimed_at = null, sn.claim_token = null
        RETURN count(sn) AS n
        """,
        ids=ids,
    )
    cleared = int(rows[0]["n"]) if rows else 0
    drained = (drain_fn or _run_scoped_validation_drain)(ids) or {}
    return {
        "cleared": cleared,
        "requarantined_ids": list(drained.get("requarantined_ids", [])),
        "valid_ids": list(drained.get("cleared_ids", [])),
    }


def default_record_change(
    gc: Any,
    ids: Sequence[str],
    run_id: str,
    spec: CampaignSpec,
) -> int:
    """Emit a campaign change event per touched name for traceability.

    Reuses the internal change-event emitter; ``from``/``to`` are identical
    because docs campaigns never change name identity — the event marks a
    reversible docs refinement, whose prior text lives in the DocsRevision.
    """
    if not ids:
        return 0
    from imas_codex.standard_names.provenance_lifecycle import (
        record_standard_name_change,
    )

    for sid in ids:
        record_standard_name_change(
            gc,
            sid,
            sid,
            operation="campaign_docs_refine",
            reason=spec.describe(),
            origin="campaign",
            run_id=run_id,
        )
    return len(ids)


def default_cost(run_id: str) -> float:  # noqa: ARG001 - overridden in the CLI
    """Cost hook stub. The CLI injects ``aggregate_spend_for_run``."""
    return 0.0


# ── Orchestrator ──────────────────────────────────────────────────────────────


@dataclass
class CampaignResult:
    """Outcome of a campaign run (or resumed segment)."""

    spec: CampaignSpec
    total_selected: int
    batches_total: int
    batches_run: int = 0
    total_touched: int = 0
    total_accepted: int = 0
    total_cost: float = 0.0
    total_audit_requarantined: int = 0
    total_audit_cleared: int = 0
    total_prose_adjudicated_clear: int = 0
    run_ids: list[str] = field(default_factory=list)
    outcomes: list[BatchOutcome] = field(default_factory=list)
    halted: bool = False
    halt_reasons: list[str] = field(default_factory=list)
    resume_from: int | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "spec": self.spec.raw,
            "total_selected": self.total_selected,
            "batches_total": self.batches_total,
            "batches_run": self.batches_run,
            "total_touched": self.total_touched,
            "total_accepted": self.total_accepted,
            "total_cost": round(self.total_cost, 4),
            "total_audit_requarantined": self.total_audit_requarantined,
            "total_audit_cleared": self.total_audit_cleared,
            "halted": self.halted,
            "halt_reasons": self.halt_reasons,
            "resume_from": self.resume_from,
            "run_ids": list(self.run_ids),
        }


class CampaignRunner:
    """Drives a selection through budgeted, gated, resumable batches.

    Every graph interaction is an injected callable so the loop is testable
    against a mock graph and mock pools.  The drain itself
    (``drain_fn(run_id, cost_limit)``) runs the normal six-pool orchestrator —
    this runner never accepts a name directly.  A drain that measured its own
    spend returns it (USD); returning ``None`` falls back to aggregating
    ``LLMCost`` by the scope ``run_id``.
    """

    def __init__(
        self,
        spec: CampaignSpec,
        budget: CampaignBudget,
        thresholds: ConvergenceThresholds,
    ) -> None:
        self.spec = spec
        self.budget = budget
        self.thresholds = thresholds

    def run(
        self,
        *,
        gc: Any,
        target_ids: Sequence[str],
        drain_fn: Callable[[str, float], float | None],
        mark_fn: Callable[..., dict[str, Any]] | None = None,
        clear_quarantine_fn: Callable[
            [Any, Sequence[str]], int
        ] = default_clear_quarantine,
        fetch_refreshed_fn: Callable[
            [Any, Sequence[str]], list[dict[str, Any]]
        ] = default_fetch_refreshed,
        revalidate_fn: Callable[
            [Any, Sequence[str], Sequence[str]], dict[str, int]
        ] = default_revalidate,
        audit_fn: Callable[[Any, Sequence[str]], dict[str, Any]] | None = None,
        record_change_fn: Callable[
            [Any, Sequence[str], str, CampaignSpec], int
        ] = default_record_change,
        cost_fn: Callable[[str], float] = default_cost,
        start_batch: int = 0,
        abort_check: Callable[[], bool] | None = None,
        on_batch: Callable[[BatchOutcome], None] | None = None,
        adjudicate_prose_fn: Callable[
            [Sequence[tuple[str, str, dict[str, int]]]], list[bool]
        ]
        | None = None,
    ) -> CampaignResult:
        """Run the campaign from *start_batch*, halting on the first failure.

        The batch loop, per batch: lift quarantine → snapshot+reset+stamp a
        fresh scope ``run_id`` → drain that scope → measure → re-run the
        deterministic ISN audit on the refreshed docs (``audit_fn``) →
        re-quarantine banned-prose reintroductions → record the change event →
        convergence gate.  A failed gate, an exceeded cost ceiling, or an abort
        request halts between batches and records ``resume_from`` so a later run
        resumes cleanly.
        """
        if mark_fn is None:
            from imas_codex.standard_names.harmonize import mark_members_for_regen

            mark_fn = mark_members_for_regen

        batches = plan_batches(target_ids, self.budget.batch_size)
        result = CampaignResult(
            spec=self.spec,
            total_selected=len(target_ids),
            batches_total=len(batches),
        )

        for index in range(start_batch, len(batches)):
            batch = batches[index]

            if abort_check is not None and abort_check():
                result.resume_from = index
                break

            if result.total_cost >= self.budget.campaign_cost_ceiling:
                result.halted = True
                result.halt_reasons.append(
                    f"campaign cost ceiling ${self.budget.campaign_cost_ceiling:.2f} "
                    f"reached (spent ${result.total_cost:.2f})"
                )
                result.resume_from = index
                break

            # 1. Lift audit quarantine so the docs pools can claim these names.
            clear_quarantine_fn(gc, batch)

            # 2. Snapshot current docs to DocsRevisions, reset the docs
            #    pipeline, and stamp a fresh scope run_id (reused mechanism).
            mark_out = mark_fn(batch, dry_run=False)
            run_id = mark_out.get("run_id")
            if run_id:
                result.run_ids.append(run_id)

            # 3. Drain ONLY this scope through the normal six-pool orchestrator.
            drained_cost = drain_fn(run_id, self.budget.per_batch_cost_cap)

            # 4. Measure the drained batch.  The pool session bills its
            #    LLMCost under its own session run_id, not the campaign
            #    scope run_id, so a drain that measured its own spend is
            #    authoritative; the scope-run aggregation is the fallback.
            if drained_cost is not None:
                cost = float(drained_cost)
            else:
                cost = float(cost_fn(run_id)) if run_id else 0.0
            refreshed = fetch_refreshed_fn(gc, batch)
            outcome = measure_batch(
                refreshed,
                batch,
                batch_index=index,
                cost=cost,
                adjudicate=adjudicate_prose_fn,
            )

            touched_ids = [r["id"] for r in refreshed]

            # 5. Re-run the deterministic ISN audit on the refreshed batch:
            #    clear the validation stamp and re-stamp validation status, so a
            #    genuine defect (e.g. a unit inconsistency) re-quarantines
            #    instead of the fast prose grep washing it to 'valid'.
            if audit_fn is not None:
                audit_deltas = audit_fn(gc, touched_ids)
                outcome.audit_requarantined = len(
                    audit_deltas.get("requarantined_ids", [])
                )
                outcome.audit_cleared = len(audit_deltas.get("valid_ids", []))

            # 6. Re-quarantine banned-prose reintroductions on top of the audit
            #    stamp — the prose grep is a campaign-specific signal the ISN
            #    audit does not carry, and it must not overwrite an audit
            #    quarantine (default_revalidate only confirms non-quarantined
            #    members to 'valid').
            clean_ids = [
                sid for sid in touched_ids if sid not in set(outcome.reintroduced_ids)
            ]
            revalidate_fn(gc, outcome.reintroduced_ids, clean_ids)

            # 7. Record a campaign change event per touched name.
            record_change_fn(gc, touched_ids, run_id or "", self.spec)

            # Accounting.
            result.batches_run += 1
            result.total_touched += outcome.touched
            result.total_accepted += outcome.accepted
            result.total_cost += cost
            result.total_audit_requarantined += outcome.audit_requarantined
            result.total_audit_cleared += outcome.audit_cleared
            result.total_prose_adjudicated_clear += len(outcome.prose_adjudicated_clear)
            result.outcomes.append(outcome)
            if outcome.prose_adjudicated_clear:
                logger.info(
                    "prose adjudicator cleared %d grep-flagged doc(s) as "
                    "legitimate this batch: %s",
                    len(outcome.prose_adjudicated_clear),
                    ", ".join(outcome.prose_adjudicated_clear[:10]),
                )
            if on_batch is not None:
                on_batch(outcome)

            # 7. Convergence gate — halt on the first non-converged batch.
            converged, reasons = evaluate_convergence(outcome, self.thresholds)
            if not converged:
                result.halted = True
                result.halt_reasons = reasons
                result.resume_from = index + 1
                break

        return result


__all__ = [
    "PROSE_CLASSES",
    "CampaignSpec",
    "CampaignTarget",
    "CampaignSelection",
    "match_target",
    "select_targets",
    "stratified_pilot",
    "build_manifest",
    "write_manifest",
    "CampaignBudget",
    "plan_batches",
    "ConvergenceThresholds",
    "BatchOutcome",
    "evaluate_convergence",
    "measure_batch",
    "default_clear_quarantine",
    "default_fetch_refreshed",
    "default_revalidate",
    "default_audit_revalidate",
    "default_record_change",
    "CampaignResult",
    "CampaignRunner",
]
