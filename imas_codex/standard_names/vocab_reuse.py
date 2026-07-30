"""Mechanical reuse detection — a registered token wearing a different spelling.

:mod:`vocab_semantic_dedup` answers the reuse question with an embedding and
lets the composer overrule it: a proposal re-emitted after being shown a near
neighbour is stamped ``distinct_confirmed``.  That adjudication is right for a
judgement call (``electron`` vs ``ion``) and wrong for a spelling, and the stored
population shows the failure — ``lower_hybrid_antenna_module`` against the
registered ``lower_hybrid_antenna`` at 0.97 and ``methane_deuterated`` against
``deuterated_methane`` at 0.96 both survived as ``distinct_confirmed``.

The checks here are not similarity: each one *derives* the registered token the
proposal spells, so a hit is reuse by construction and overrides any embedding
verdict.  Four mechanisms:

- **advisory_alias** — ISN publishes a segment-scoped source spelling and the
  registered canonical token to use instead.  These are retry hints, not parser
  aliases: consuming one never makes the source spelling grammatical.
- **structural_suffix** — the DD subdivides an assembly by appending a
  structural noun (``_module``, ``_component``, ``_channel``, ``_element``,
  ``_image``).  The subdivision is DD structure, not a distinct quantity, so the
  parent token is the one to use.
- **word_order** — the proposal's words are a permutation of a registered
  token's words.  Word order is the grammar's, not the composer's; a
  permutation is the same token misspelled.  Safe only because the registered
  vocabulary itself contains no two tokens sharing a word multiset — a test
  asserts that, and the check must be reconsidered if ISN ever adds such a pair.
- **settled_synonym** — a synonym relation that no amount of ISN introspection
  can derive because the two spellings share no words.  Each entry is a
  decision already settled in review, and is dropped at runtime when its target
  is not registered: a mapping onto a token ISN no longer carries must fall
  silent rather than send the composer at a name that does not exist.

Every registered token comes from :func:`grammar_token_index` at runtime — the
operator registry included — so nothing here carries a vocabulary snapshot.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

#: Nouns by which the DD names a subdivision of an assembly.  A proposal that
#: is a registered token plus one of these asks for the subdivision's own token,
#: but the subdivision carries the parent's quantity — the axis that
#: distinguishes one module or channel from another is the DD path structure,
#: never a vocabulary token.
_STRUCTURAL_SUFFIXES: tuple[str, ...] = (
    "module",
    "component",
    "channel",
    "element",
    "image",
)

#: Synonym pairs settled in review, ``proposal -> registered token``.  Kept
#: minimal and individually justified; verified against the live vocabulary by
#: :func:`settled_synonyms`, which drops any entry whose target is unregistered.
#:
#: - ``optical_depth`` -> ``opacity``: the dimensionless line-of-sight
#:   attenuation exponent.  ISN carries it as ``opacity``; the reviewer settled
#:   the pair twice and the composer has re-proposed ``optical_depth`` since.
_SETTLED_SYNONYM_CANDIDATES: dict[str, str] = {
    "optical_depth": "opacity",
}


@dataclass(frozen=True, slots=True)
class ReuseFinding:
    """A proposed token resolved to the registered token it spells.

    Attributes:
        token: The proposed token.
        segment: The grammar class the proposal was reported against.
        target: The registered token to use instead.
        target_segments: Every grammar class admitting ``target`` — the slot
            guidance names, which need not be ``segment``.
        mechanism: ``advisory_alias``, ``structural_suffix``, ``word_order`` or
            ``settled_synonym``.
        detail: One clause explaining the derivation, for retry guidance.
    """

    token: str
    segment: str
    target: str
    target_segments: tuple[str, ...]
    mechanism: str
    detail: str


@dataclass(frozen=True, slots=True)
class StoredVerdictAudit:
    """A stored gap whose dedup verdict a mechanical check contradicts or fills in.

    Attributes:
        id: The stored node id, when the record carried one.
        segment: The gap's grammar class.
        token: The gap's token.
        stored_decision: The dedup verdict on record — ``distinct_confirmed``,
            ``unchecked``, or ``None`` when never stamped.
        finding: The mechanical resolution that supersedes it.
    """

    id: str | None
    segment: str
    token: str
    stored_decision: str | None
    finding: ReuseFinding


def _registered() -> dict[str, tuple[str, ...]]:
    """Token -> admitting classes, operators included.  Empty when ISN is absent."""
    from imas_codex.standard_names.segments import grammar_token_index

    return grammar_token_index()


@lru_cache(maxsize=1)
def _advisory_aliases() -> Mapping[str, Any]:
    """Return ISN's segment-scoped retry aliases, or an empty mapping.

    Older supported ISN releases do not expose this optional contract.  The
    absence of the key therefore disables only this mechanism and leaves the
    existing mechanical checks unchanged.
    """
    try:
        from imas_standard_names import get_grammar_context
    except ImportError:
        return {}

    try:
        aliases = get_grammar_context()["grammar"].get("advisory_aliases", {})
    except (AttributeError, KeyError, TypeError):
        return {}
    return aliases if isinstance(aliases, Mapping) else {}


def _advisory_alias(
    segment: str,
    token: str,
    registered: Mapping[str, tuple[str, ...]],
) -> tuple[str, str] | None:
    """Resolve one exact ISN advisory alias within its declared segment."""
    segment_aliases = _advisory_aliases().get(segment)
    if not isinstance(segment_aliases, Mapping):
        return None
    definition = segment_aliases.get(token)
    if not isinstance(definition, Mapping):
        return None

    target = definition.get("canonical")
    reason = definition.get("reason")
    if (
        not isinstance(target, str)
        or not target
        or not isinstance(reason, str)
        or not reason
        or target not in registered
        or segment not in registered[target]
    ):
        return None
    return target, reason


@lru_cache(maxsize=1)
def settled_synonyms() -> dict[str, str]:
    """The settled synonym map, restricted to entries whose target is registered."""
    registered = _registered()
    if not registered:
        return {}
    return {
        source: target
        for source, target in _SETTLED_SYNONYM_CANDIDATES.items()
        if target in registered and source not in registered
    }


@lru_cache(maxsize=1)
def _tokens_by_word_multiset() -> dict[tuple[str, ...], tuple[str, ...]]:
    """Registered multi-word tokens grouped by their sorted word multiset."""
    grouped: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for token in _registered():
        words = token.split("_")
        if len(words) > 1:
            grouped[tuple(sorted(words))].append(token)
    return {key: tuple(sorted(members)) for key, members in grouped.items()}


def _structural_suffix_target(token: str) -> tuple[str, str] | None:
    """Resolve ``token`` as a registered token plus a DD subdivision noun.

    Returns ``(target, suffix)``.  Both directions are tried: the proposal may
    append the noun to a registered token, or drop one the registered spelling
    carries.
    """
    registered = _registered()
    words = token.split("_")
    if len(words) > 1 and words[-1] in _STRUCTURAL_SUFFIXES:
        stem = "_".join(words[:-1])
        if stem in registered:
            return stem, words[-1]
    for suffix in _STRUCTURAL_SUFFIXES:
        candidate = f"{token}_{suffix}"
        if candidate in registered:
            return candidate, suffix
    return None


def registered_reuse(segment: str, token: str) -> ReuseFinding | None:
    """Resolve ``token`` to the registered token it spells, or ``None``.

    Pure and free — no embedding, no graph, no LLM.  Returns ``None`` for a
    token that is itself registered (the caller's classifier already calls that
    a false positive), for a genuinely novel proposal, and whenever ISN is
    unavailable, so an installation without the grammar reports no reuse rather
    than a wrong one.
    """
    if not token:
        return None
    registered = _registered()
    if not registered or token in registered:
        return None

    alias_hit = _advisory_alias(segment, token, registered)
    if alias_hit is not None:
        target, reason = alias_hit
        return ReuseFinding(
            token=token,
            segment=segment,
            target=target,
            target_segments=registered[target],
            mechanism="advisory_alias",
            detail=reason,
        )

    target = settled_synonyms().get(token)
    if target is not None:
        return ReuseFinding(
            token=token,
            segment=segment,
            target=target,
            target_segments=registered[target],
            mechanism="settled_synonym",
            detail=f"'{token}' is the settled synonym of the registered '{target}'",
        )

    suffix_hit = _structural_suffix_target(token)
    if suffix_hit is not None:
        target, suffix = suffix_hit
        return ReuseFinding(
            token=token,
            segment=segment,
            target=target,
            target_segments=registered[target],
            mechanism="structural_suffix",
            detail=(
                f"'{token}' is the registered '{target}' plus the DD structural "
                f"noun '{suffix}', which names a subdivision of the same quantity "
                f"rather than a distinct one"
            ),
        )

    words = token.split("_")
    if len(words) > 1:
        variants = [
            other
            for other in _tokens_by_word_multiset().get(tuple(sorted(words)), ())
            if other != token
        ]
        if variants:
            target = variants[0]
            return ReuseFinding(
                token=token,
                segment=segment,
                target=target,
                target_segments=registered[target],
                mechanism="word_order",
                detail=(
                    f"'{token}' carries exactly the words of the registered "
                    f"'{target}' in another order; word order is the grammar's"
                ),
            )

    return None


def reuse_guidance(finding: ReuseFinding) -> str:
    """One line of retry feedback naming the registered token to use instead."""
    where = " or ".join(finding.target_segments) or "the grammar"
    return (
        f"{finding.detail} — use the registered {where} token "
        f"'{finding.target}' instead of requesting '{finding.token}'."
    )


def audit_stored_verdicts(
    records: Iterable[Mapping[str, Any]],
) -> list[StoredVerdictAudit]:
    """Re-audit stored gap records, returning only those a mechanical check resolves.

    Read-only over its input, so it is safe to dry-run against a dump before any
    graph write.  ``records`` need only carry ``segment`` and ``token``; ``id``
    and the dedup verdict (under either ``dedup_decision`` — the stored property
    — or the abbreviated ``dedup``) are reported back when present so a caller
    can see which stored verdict each finding supersedes.
    """
    audits: list[StoredVerdictAudit] = []
    for record in records:
        segment = record.get("segment") or ""
        token = record.get("token") or ""
        finding = registered_reuse(segment, token)
        if finding is None:
            continue
        stored = record.get("dedup_decision", record.get("dedup"))
        audits.append(
            StoredVerdictAudit(
                id=record.get("id"),
                segment=segment,
                token=token,
                stored_decision=stored,
                finding=finding,
            )
        )
    return audits


def apply_reuse_verdicts(
    gc: Any,
    audits: Iterable[StoredVerdictAudit],
    *,
    dry_run: bool = False,
) -> int:
    """Stamp mechanically-derived reuse verdicts onto stored ``VocabGap`` nodes.

    A mechanical hit derives the registered token the proposal spells, so the
    verdict supersedes any embedding adjudication on record.  Writes the
    verdict, the registered target, the deriving mechanism and its one-clause
    detail, and a reconciliation stamp.  Nothing is deleted — the node stays as
    the record of what the composer proposed and where it was sent instead.

    Returns the number of nodes stamped (or that would be, under ``dry_run``).
    """
    items = [
        {
            "id": a.id,
            "target": a.finding.target,
            "mechanism": a.finding.mechanism,
            "detail": a.finding.detail,
        }
        for a in audits
        if a.id
    ]
    if not items or dry_run:
        return len(items)
    gc.query(
        """
        UNWIND $items AS it
        MATCH (g:VocabGap {id: it.id})
        SET g.dedup_decision = 'reuse_confirmed',
            g.reuse_target = it.target,
            g.reuse_mechanism = it.mechanism,
            g.reuse_detail = it.detail,
            g.reconciled_at = datetime()
        """,
        items=items,
    )
    return len(items)


def clear_reuse_caches() -> None:
    """Drop the cached views of the registered vocabulary held in this module."""
    _advisory_aliases.cache_clear()
    settled_synonyms.cache_clear()
    _tokens_by_word_multiset.cache_clear()


__all__ = [
    "ReuseFinding",
    "StoredVerdictAudit",
    "apply_reuse_verdicts",
    "audit_stored_verdicts",
    "clear_reuse_caches",
    "registered_reuse",
    "reuse_guidance",
    "settled_synonyms",
]
