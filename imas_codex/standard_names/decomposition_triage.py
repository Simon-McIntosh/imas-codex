"""Deterministic (LLM-free) triage of stale ``decomposition_audit`` findings.

The ``decomposition_audit`` surfaces a finding whenever a closed-vocabulary
token appears as an underscore-delimited substring of a standard name's *raw
id* (see :func:`imas_codex.standard_names.decomposition.find_absorbed_closed_tokens`).
It does NOT consult the parse, so a name whose token is correctly slotted by
the grammar (``ion_current_density`` → ``subject=ion``) is flagged exactly the
same as one whose token genuinely leaked into ``physical_base``
(``reference_magnetic_field``). Most stored findings on accepted names are
therefore *stale*, not real.

This module re-parses each flagged name under the CURRENT grammar and buckets
it deterministically — no LLM, no cost:

    drain    the parse slots every flagged token into a real segment and the
             parsed ``physical_base`` carries no leaked closed token; the
             finding is stale → clear it (and re-stamp the segment columns).
    suppress a closed token remains inside the parsed ``physical_base`` but the
             ``physical_base`` is a grammar-registered atomic base (a
             lexicalised compound such as ``convection_velocity`` or
             ``diffusion_coefficient``); the token is legitimately part of the
             base → clear the finding.
    rename   a closed token remains inside a ``physical_base`` the grammar does
             NOT accept as an atomic base — a genuine decomposition failure
             (``reference_magnetic_field``, ``vacuum_magnetic_vector_potential``);
             left untouched and queued for a rename rotation.

Names the grammar rejects outright are reported separately (``parse_fail`` /
``non_canonical``) as migration backlog rather than silently dropped.

The bucket rule is the same one the *production* audit should have used: scan
the PARSED physical_base, and exempt a physical_base the grammar registers as
an atomic base. The triage drains the stored stale findings; making
``decomposition_audit_check`` parse-aware is the durable companion fix that
stops them recurring (queued alongside the apply pass).
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Triage bucket labels.
DRAIN = "drain"  # (a) stale — re-parse slots the token; clear + re-stamp
SUPPRESS = "suppress"  # (b) lexicalised compound base — clear (legitimate)
RENAME = "rename"  # (c) genuine absorption — queue for rename rotation
PARSE_FAIL = "parse_fail"  # grammar rejects the name outright
NON_CANONICAL = "non_canonical"  # parses but in non-canonical order

#: Buckets whose stored decomposition_audit findings are safe to clear with no
#: LLM call and no name-string change (the free drain).
CLEARABLE_BUCKETS = frozenset({DRAIN, SUPPRESS})

_AUDIT_TAG = "decomposition_audit"
# Aliased / open segments that decomposition_audit_check excludes from the
# closed vocabulary (mirror that construction exactly).
_ALIAS_SEGMENTS = frozenset({"coordinate", "object", "position"})


@dataclass
class TriageEntry:
    """One flagged name's deterministic verdict."""

    name: str
    bucket: str
    physical_base: str | None = None
    #: Closed tokens still embedded in the PARSED physical_base (bucket b / c).
    leaked_tokens: list[str] = field(default_factory=list)
    #: Suggested segmentation for a rename candidate (bucket c only).
    suggestion: str | None = None
    #: Reason string for parse_fail / non_canonical.
    detail: str | None = None


def build_closed_vocab() -> dict[str, list[str]]:
    """Return the closed-vocabulary dict used by the decomposition audit.

    Mirrors :func:`imas_codex.standard_names.audits.decomposition_audit_check`:
    every ``SEGMENT_TOKEN_MAP`` segment except the open ``physical_base`` and
    the aliased segments, with non-empty token lists.
    """
    from imas_standard_names.grammar.constants import SEGMENT_TOKEN_MAP

    closed: dict[str, list[str]] = {}
    for seg, toks in SEGMENT_TOKEN_MAP.items():
        if seg in _ALIAS_SEGMENTS or seg == "physical_base" or not toks:
            continue
        closed[seg] = list(toks)
    return closed


def load_registered_bases() -> frozenset[str]:
    """Return the set of tokens the grammar accepts as atomic bases/carriers.

    A ``physical_base`` in this set is a lexicalised compound the grammar owns
    (``convection_velocity``, ``diffusion_coefficient``, ``safety_factor``); a
    closed token embedded in it is legitimate, not an absorption.
    """
    from imas_standard_names.grammar.parser import load_default_vocabularies

    vocabs = load_default_vocabularies()
    return frozenset(set(vocabs.bases) | set(vocabs.carriers))


def _suggest_segmentation(
    physical_base: str, leaked: list[tuple[str, str]], bases: frozenset[str]
) -> str | None:
    """Naive rename hint for a bucket-c name.

    If stripping a single leaked token from the compound leaves a
    grammar-registered base, suggest ``<token>[segment] + <base>``. Purely a
    reviewer aid — the rename rotation composes the real name.
    """
    for token, segment in leaked:
        for candidate in (
            physical_base.replace(f"{token}_", "", 1),
            physical_base.replace(f"_{token}", "", 1),
        ):
            if candidate != physical_base and candidate in bases:
                return f"{token} → {segment}; base → {candidate}"
    tokens = ", ".join(f"{t} ({s})" for t, s in leaked)
    return f"absorbed: {tokens}"


def classify_name(
    name: str,
    *,
    closed_vocab: dict[str, list[str]],
    bases: frozenset[str],
) -> TriageEntry:
    """Bucket one flagged name via a re-parse under the current grammar.

    Pure function — no graph, no LLM. See the module docstring for the rule.
    """
    from imas_standard_names.grammar import parse_standard_name
    from imas_standard_names.grammar.model import NonCanonicalNameError

    from imas_codex.standard_names.decomposition import find_absorbed_closed_tokens

    try:
        model = parse_standard_name(name)
    except NonCanonicalNameError as exc:
        return TriageEntry(
            name=name,
            bucket=NON_CANONICAL,
            detail=f"canonical form: {exc.canonical_form}",
        )
    except Exception as exc:  # noqa: BLE001 — any grammar rejection is backlog
        return TriageEntry(
            name=name, bucket=PARSE_FAIL, detail=f"{type(exc).__name__}: {exc}"
        )

    physical_base = getattr(model, "physical_base", None) or ""
    leaked = find_absorbed_closed_tokens(physical_base, closed_vocab)

    if not leaked:
        # Every flagged token was slotted into a real segment; the parsed base
        # is clean. The stored finding is stale.
        return TriageEntry(name=name, bucket=DRAIN, physical_base=physical_base)

    leaked_tokens = [tok for tok, _ in leaked]
    if physical_base in bases:
        # The base is a grammar-registered atomic/lexicalised compound; the
        # embedded token is legitimate.
        return TriageEntry(
            name=name,
            bucket=SUPPRESS,
            physical_base=physical_base,
            leaked_tokens=leaked_tokens,
        )

    return TriageEntry(
        name=name,
        bucket=RENAME,
        physical_base=physical_base,
        leaked_tokens=leaked_tokens,
        suggestion=_suggest_segmentation(physical_base, leaked, bases),
    )


def fetch_flagged_names(gc: Any, *, limit: int | None = None) -> list[str]:
    """Return the ids of live (accepted) names carrying a decomposition_audit finding."""
    query = f"""
        MATCH (sn:StandardName)
        WHERE sn.name_stage = 'accepted'
          AND sn.validation_issues IS NOT NULL
          AND any(x IN sn.validation_issues WHERE x CONTAINS '{_AUDIT_TAG}')
        RETURN sn.id AS id
        ORDER BY sn.id
        {f"LIMIT {int(limit)}" if limit else ""}
    """
    return [r["id"] for r in gc.query(query) if r.get("id")]


def triage(
    names: list[str],
    *,
    closed_vocab: dict[str, list[str]] | None = None,
    bases: frozenset[str] | None = None,
) -> list[TriageEntry]:
    """Classify every name; pure over the grammar (no graph)."""
    closed_vocab = closed_vocab or build_closed_vocab()
    bases = bases if bases is not None else load_registered_bases()
    return [classify_name(n, closed_vocab=closed_vocab, bases=bases) for n in names]


def build_manifest(
    entries: list[TriageEntry], *, rename_review_cost_per_name: float = 0.08
) -> dict[str, Any]:
    """Aggregate triage entries into a review manifest with bucket counts."""
    counts = Counter(e.bucket for e in entries)
    rename_queue = [
        {
            "name": e.name,
            "physical_base": e.physical_base,
            "absorbed_tokens": e.leaked_tokens,
            "suggestion": e.suggestion,
        }
        for e in entries
        if e.bucket == RENAME
    ]
    backlog = [
        {"name": e.name, "bucket": e.bucket, "detail": e.detail}
        for e in entries
        if e.bucket in (PARSE_FAIL, NON_CANONICAL)
    ]
    n_rename = counts[RENAME]
    return {
        "total": len(entries),
        "buckets": {
            "drain": counts[DRAIN],
            "suppress": counts[SUPPRESS],
            "rename": n_rename,
            "parse_fail": counts[PARSE_FAIL],
            "non_canonical": counts[NON_CANONICAL],
        },
        "clearable_free": counts[DRAIN] + counts[SUPPRESS],
        "rename_queue": rename_queue,
        "rename_queue_projected_review_cost_usd": round(
            n_rename * rename_review_cost_per_name, 2
        ),
        "grammar_backlog": backlog,
    }


def apply_drain(gc: Any, entries: list[TriageEntry]) -> dict[str, int]:
    """Clear stale ``decomposition_audit`` findings for drain/suppress buckets.

    Removes only the ``decomposition_audit`` lines from ``validation_issues``
    (other findings are preserved); an emptied list becomes ``null``. Bucket-c
    (rename) and grammar-backlog names are left untouched. Segment columns are
    re-stamped separately by the always-on ``rederive_structural_edges`` pass,
    which the caller should run after this drain.
    """
    clearable = [e.name for e in entries if e.bucket in CLEARABLE_BUCKETS]
    if not clearable:
        return {"cleared": 0}
    result = gc.query(
        """
        UNWIND $names AS nm
        MATCH (sn:StandardName {id: nm})
        WHERE sn.validation_issues IS NOT NULL
        WITH sn, [x IN sn.validation_issues WHERE NOT x CONTAINS $tag] AS kept
        SET sn.validation_issues = CASE WHEN size(kept) = 0 THEN null ELSE kept END
        RETURN count(sn) AS n
        """,
        {"names": clearable, "tag": _AUDIT_TAG},
    )
    cleared = result[0]["n"] if result else 0
    return {"cleared": cleared}
