"""Runtime token-reuse check for proposed vocabulary tokens.

The lexical filter in :mod:`vocab_token_filter` catches exact / plural
duplicates, but it cannot see that ``field_line_length`` and
``connection_length`` mean the same thing, or that ``poloidal_flux`` is a
near-synonym of an existing ``poloidal_magnetic_flux`` base.  Without a
semantic check the ISN vocabulary slowly accretes redundant tokens, each
splitting the source population that should map to one canonical term.

This is the **component-token** dedup axis, complementary to the existing
whole-description name-similarity check injected into the reviewer prompt
(``fetch_review_neighbours``).  The two axes catch different failures: a novel
*assembly* of valid tokens (name-similarity) vs a duplicate *token*
(this module).

The check is wired into the compose retry loop: when the LLM emits a
``vocab_gap`` for a needed-but-unregistered token, that token is embedded as a
short descriptive phrase and compared by cosine similarity against the
registered tokens of its *same* grammar segment.  A near neighbour at/above the
configured threshold is fed back into the next compose attempt as advisory
context — the agent then either reuses the registered token (dedup achieved) or
re-emits the gap (confirming the concept is genuinely distinct).  It is never a
hard reject: the 2026-06-14 self-refine measurement showed fuzzy
LLM/embedding signals degrade quality when used as a hard gate.

Embeddings are produced by the shared local/remote :class:`Encoder` and are
free (no LLM cost).  The check is best-effort: if the embedding backend is
unavailable the candidate passes through un-flagged rather than blocking the
compose pipeline.

Usage::

    from imas_codex.standard_names.vocab_semantic_dedup import (
        nearest_registered_token,
    )
    hit = nearest_registered_token("field_line_length", "physical_base")
    if hit:
        print(hit.proposed_token, "~", hit.nearest_token, hit.similarity)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

#: Cosine similarity at/above which a proposed token is treated as a likely
#: synonym of a registered one.  Tuned to flag clear synonyms while leaving
#: genuinely distinct neighbours (e.g. ``electron`` vs ``ion``) unflagged.
#: Surfaced as advice — a flagged token is re-prompted, not auto-rejected.
DUPLICATE_THRESHOLD: float = 0.82

#: A token whose nearest neighbour is below this is "clearly novel" — no note.
NOVEL_THRESHOLD: float = 0.70


@dataclass(frozen=True)
class TokenNeighbour:
    """Nearest registered same-segment token for a proposed new token.

    Attributes:
        proposed_token: The candidate-new token the LLM wanted.
        segment: The grammar segment the token was proposed against.
        nearest_token: The closest registered token in that segment.
        similarity: Cosine similarity (0..1) between the two.
    """

    proposed_token: str
    segment: str
    nearest_token: str
    similarity: float


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
        logger.debug("ISN SEGMENT_TOKEN_MAP unavailable — token-reuse check is a no-op")
        return {}


def _get_encoder(encoder: Any | None) -> Any | None:
    """Return a usable Encoder, or ``None`` if the backend is unavailable.

    Best-effort: never raises — the token-reuse check is advisory.
    """
    if encoder is not None:
        return encoder
    try:
        from imas_codex.embeddings.encoder import Encoder

        return Encoder()
    except Exception as exc:  # noqa: BLE001 — advisory check, never block
        logger.warning("Token-reuse check skipped — encoder unavailable: %s", exc)
        return None


def _nearest_in_segment(
    tokens: list[str],
    segment: str,
    existing: list[str],
    encoder: Any,
) -> dict[str, TokenNeighbour]:
    """Embed ``tokens`` + the segment's ``existing`` vocab once and pair each.

    Embeds the full existing-vocab list a single time so it is not re-embedded
    per proposed token.  Returns a map ``proposed_token -> TokenNeighbour``;
    tokens for which no neighbour could be resolved are omitted.  Never raises.
    """
    import numpy as np

    try:
        existing_texts = [_describe_token(t, segment) for t in existing]
        proposed_texts = [_describe_token(t, segment) for t in tokens]
        existing_emb = np.asarray(encoder.embed_texts(existing_texts))
        proposed_emb = np.asarray(encoder.embed_texts(proposed_texts))
    except Exception as exc:  # noqa: BLE001 — advisory check, never block
        logger.warning(
            "Token-reuse check skipped for segment %s — embed failed: %s",
            segment,
            exc,
        )
        return {}

    sims = _cosine_matrix(proposed_emb, existing_emb)  # (n_proposed, n_existing)
    out: dict[str, TokenNeighbour] = {}
    for i, token in enumerate(tokens):
        row = sims[i]
        best_j = int(np.argmax(row))
        best_sim = float(row[best_j])
        best_tok = existing[best_j]
        # A proposed token may already equal a registered one (lexical compound
        # under-reported elsewhere) — exclude exact self so we surface the
        # nearest *different* neighbour.
        if best_tok == token and len(row) > 1:
            masked = row.copy()
            masked[best_j] = -1.0
            best_j = int(np.argmax(masked))
            best_sim = float(masked[best_j])
            best_tok = existing[best_j]
        elif best_tok == token:
            # Single-element vocab and it IS the token — nothing distinct to
            # compare against.
            continue
        out[token] = TokenNeighbour(
            proposed_token=token,
            segment=segment,
            nearest_token=best_tok,
            similarity=round(best_sim, 4),
        )
    return out


def nearest_registered_token(
    token: str,
    segment: str,
    *,
    encoder: Any | None = None,
    threshold: float = DUPLICATE_THRESHOLD,
) -> TokenNeighbour | None:
    """Return the nearest registered same-segment token at/above ``threshold``.

    Embeds ``token`` (as a short phrase) and compares it by cosine similarity
    against the registered vocabulary of ``segment``.  Returns a
    :class:`TokenNeighbour` only when a near neighbour scores at/above
    ``threshold``; otherwise ``None``.

    Returns ``None`` (no flag) when:

    - ISN is unavailable, or the segment is open / pseudo / has no vocabulary;
    - the token is already the sole registered token in its segment;
    - the embedding backend is unavailable or fails (best-effort — never raises);
    - the nearest neighbour is below ``threshold`` (genuinely distinct token).

    Args:
        token: The candidate-new token the LLM proposed.
        segment: The grammar segment it was proposed against.
        encoder: Optional pre-built :class:`Encoder` (mainly for tests); a
            shared instance is created on demand otherwise.
        threshold: Cosine similarity at/above which the hit is returned.

    Returns:
        A :class:`TokenNeighbour` when a likely-synonym registered token
        exists, else ``None``.
    """
    existing_by_seg = _existing_tokens_by_segment()
    existing = existing_by_seg.get(segment)
    if not existing:
        return None

    enc = _get_encoder(encoder)
    if enc is None:
        return None

    hit = _nearest_in_segment([token], segment, existing, enc).get(token)
    if hit is None or hit.similarity < threshold:
        return None
    return hit


def nearest_registered_tokens(
    items: list[tuple[str, str]],
    *,
    encoder: Any | None = None,
    threshold: float = DUPLICATE_THRESHOLD,
) -> dict[tuple[str, str], TokenNeighbour]:
    """Batch variant of :func:`nearest_registered_token`.

    Groups ``(token, segment)`` pairs by segment so each segment's registered
    vocabulary is embedded only once — important for throughput when a compose
    batch reports several gaps.

    Args:
        items: ``(token, segment)`` pairs to check.
        encoder: Optional pre-built :class:`Encoder` (shared on demand).
        threshold: Cosine similarity at/above which a hit is included.

    Returns:
        Map ``(token, segment) -> TokenNeighbour`` for every pair whose nearest
        registered same-segment token scored at/above ``threshold``.  Pairs
        with no hit are omitted.  Never raises — advisory.
    """
    if not items:
        return {}

    existing_by_seg = _existing_tokens_by_segment()
    if not existing_by_seg:
        return {}

    # Group proposed tokens by segment; only segments with registered vocab
    # are worth embedding.
    by_seg: dict[str, list[str]] = {}
    for token, segment in items:
        if segment in existing_by_seg and existing_by_seg[segment]:
            by_seg.setdefault(segment, []).append(token)
    if not by_seg:
        return {}

    enc = _get_encoder(encoder)
    if enc is None:
        return {}

    results: dict[tuple[str, str], TokenNeighbour] = {}
    for segment, tokens in by_seg.items():
        # De-dup tokens within a segment so we embed each once.
        unique_tokens = sorted(set(tokens))
        neighbours = _nearest_in_segment(
            unique_tokens, segment, existing_by_seg[segment], enc
        )
        for token, hit in neighbours.items():
            if hit.similarity >= threshold:
                results[(token, segment)] = hit

    if results:
        logger.info(
            "Token-reuse check flagged %d/%d proposed tokens as likely synonyms",
            len(results),
            len(items),
        )
    return results


__all__ = [
    "DUPLICATE_THRESHOLD",
    "NOVEL_THRESHOLD",
    "TokenNeighbour",
    "nearest_registered_token",
    "nearest_registered_tokens",
]
