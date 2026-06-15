"""Semantic near-duplicate detection for proposed vocabulary tokens.

The lexical filter in :mod:`vocab_token_filter` catches exact / plural
duplicates, but it cannot see that ``field_line_length`` and
``connection_length`` mean the same thing, or that ``poloidal_flux`` is a
near-synonym of an existing ``poloidal_magnetic_flux`` base.  Without a
semantic check the ISN vocabulary slowly accretes redundant tokens, each
splitting the source population that should map to one canonical term.

This module embeds each proposed gap token (as a short descriptive phrase)
and compares it by cosine similarity against the existing tokens of the
*same* grammar segment.  A near neighbour above ``DUPLICATE_THRESHOLD`` is
surfaced as a **triage annotation**, never an auto-reject: the self-refine
measurement (2026-06-14) showed fuzzy LLM/embedding signals degrade quality
when used as a hard gate.  The human (or LLM) curating the ISN PR sees the
nearest existing token and its score and decides whether the proposal is a
genuine new concept or a synonym to fold in.

Embeddings are produced by the shared local/remote :class:`Encoder` and are
free (no LLM cost).  The check is best-effort: if the embedding backend is
unavailable the candidates pass through un-annotated rather than blocking
the gap report.

Usage::

    from imas_codex.standard_names.vocab_semantic_dedup import annotate_duplicates
    annotated = annotate_duplicates(gap_records)
    for r in annotated:
        if r.get("nearest_existing"):
            print(r["token"], "~", r["nearest_existing"], r["nearest_similarity"])
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

#: Cosine similarity at/above which a proposed token is flagged as a likely
#: semantic duplicate of an existing one.  Tuned to flag clear synonyms while
#: leaving genuinely distinct neighbours (e.g. ``electron`` vs ``ion``) unflagged.
#: Surfaced as advice — a flagged token is reviewed, not auto-rejected.
DUPLICATE_THRESHOLD: float = 0.82

#: A token whose nearest neighbour is below this is "clearly novel" — no note.
NOVEL_THRESHOLD: float = 0.70


def _describe_token(token: str, segment: str) -> str:
    """Render a proposed token as a short phrase for embedding.

    A bare snake_case token embeds poorly (the model sees an underscored
    identifier, not language).  Expanding to a phrase with light segment
    context sharpens the semantic signal so synonyms cluster.
    """
    words = token.replace("_", " ").strip()
    # Light, segment-appropriate framing — keep it short so the token words
    # dominate the embedding rather than the boilerplate.
    seg = segment.replace("_", " ")
    return f"{words} ({seg})"


def _cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity of every row of ``a`` against every row of ``b``.

    Inputs are L2-normalised defensively so the result is a true cosine even
    if the encoder did not normalise.
    """
    import numpy as np

    def _norm(m: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(m, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return m / n

    return _norm(a) @ _norm(b).T


def _existing_tokens_by_segment() -> dict[str, list[str]]:
    """Map each closed grammar segment to its registered token list.

    Returns an empty dict when ISN is unavailable (check is then a no-op).
    """
    try:
        from imas_standard_names.grammar.constants import SEGMENT_TOKEN_MAP

        return {
            seg: sorted(tokens) for seg, tokens in SEGMENT_TOKEN_MAP.items() if tokens
        }
    except Exception:  # ImportError or any parsing error
        logger.debug("ISN SEGMENT_TOKEN_MAP unavailable — semantic dedup is a no-op")
        return {}


def annotate_duplicates(
    records: list[dict[str, Any]],
    *,
    threshold: float = DUPLICATE_THRESHOLD,
    encoder: Any | None = None,
) -> list[dict[str, Any]]:
    """Annotate gap records with their nearest existing same-segment token.

    For every record, embeds the proposed ``token`` and compares it against
    the existing vocabulary of its ``segment``.  Adds three keys in-place:

    - ``nearest_existing``: the closest existing token (or ``None``)
    - ``nearest_similarity``: cosine similarity to it (float, or ``None``)
    - ``likely_duplicate``: ``True`` when similarity >= ``threshold``

    Records whose segment has no existing tokens, or when the embedding
    backend is unavailable, are returned unchanged (keys set to ``None`` /
    ``False``).  Never raises on backend failure — annotation is advisory.

    Args:
        records: Flat gap records from
            :func:`~imas_codex.standard_names.gap_harvest.harvest_vocab_gaps`.
        threshold: Cosine similarity at/above which ``likely_duplicate`` is set.
        encoder: Optional pre-built :class:`Encoder` (mainly for tests); a
            shared instance is created on demand otherwise.

    Returns:
        The same list, with annotation keys added to each record.
    """
    # Initialise keys so downstream formatting can rely on their presence.
    for r in records:
        r.setdefault("nearest_existing", None)
        r.setdefault("nearest_similarity", None)
        r.setdefault("likely_duplicate", False)

    if not records:
        return records

    existing_by_seg = _existing_tokens_by_segment()
    if not existing_by_seg:
        return records

    # Only segments that have BOTH proposed tokens and existing vocab are worth
    # embedding.  Group proposed tokens by segment to batch the encoder calls.
    proposed_by_seg: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        seg = r.get("segment")
        if seg and seg in existing_by_seg:
            proposed_by_seg.setdefault(seg, []).append(r)

    if not proposed_by_seg:
        return records

    try:
        if encoder is None:
            from imas_codex.embeddings.encoder import Encoder

            encoder = Encoder()
    except Exception as exc:  # noqa: BLE001 — advisory check, never block
        logger.warning("Semantic dedup skipped — encoder unavailable: %s", exc)
        return records

    import numpy as np

    for seg, recs in proposed_by_seg.items():
        existing = existing_by_seg[seg]
        proposed_tokens = [r["token"] for r in recs]
        try:
            # Embed existing + proposed together so they share the same call /
            # normalisation; existing-token phrases use the same framing.
            existing_texts = [_describe_token(t, seg) for t in existing]
            proposed_texts = [_describe_token(t, seg) for t in proposed_tokens]
            existing_emb = np.asarray(encoder.embed_texts(existing_texts))
            proposed_emb = np.asarray(encoder.embed_texts(proposed_texts))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Semantic dedup skipped for segment %s — embed failed: %s", seg, exc
            )
            continue

        sims = _cosine_matrix(proposed_emb, existing_emb)  # (n_proposed, n_existing)
        for i, rec in enumerate(recs):
            row = sims[i]
            # Guard: a proposed token may already equal an existing token
            # (lexical compound under-reported elsewhere) — exclude exact match
            # so we surface the nearest *different* neighbour.
            best_j = int(np.argmax(row))
            best_sim = float(row[best_j])
            best_tok = existing[best_j]
            if best_tok == rec["token"] and len(row) > 1:
                masked = row.copy()
                masked[best_j] = -1.0
                best_j = int(np.argmax(masked))
                best_sim = float(masked[best_j])
                best_tok = existing[best_j]
            rec["nearest_existing"] = best_tok
            rec["nearest_similarity"] = round(best_sim, 4)
            rec["likely_duplicate"] = best_sim >= threshold

    flagged = sum(1 for r in records if r.get("likely_duplicate"))
    if flagged:
        logger.info(
            "Semantic dedup flagged %d/%d proposed tokens as likely duplicates",
            flagged,
            len(records),
        )
    return records


__all__ = [
    "DUPLICATE_THRESHOLD",
    "NOVEL_THRESHOLD",
    "annotate_duplicates",
]
