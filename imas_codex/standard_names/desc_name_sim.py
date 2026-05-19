"""Desc-name similarity gate for the Phase 5 REFINE_DOCS routing.

Computes the cosine similarity between the grammar-expanded name embedding
and the description embedding.  Used in the REVIEW_NAME pre-step for
``origin='derived'`` items: below ``desc_name_similarity_threshold`` the
item is routed to REFINE_DOCS instead of completing name scoring, keeping
description-quality failures off the name-failure axis.

The compute function wraps the existing ``semantic_similarity_check`` logic
from ``imas_codex.standard_names.audits`` but returns only the bare float
(or ``None`` on embed failure) without the issue-string side-channel —
callers decide what to do with the score.
"""

from __future__ import annotations

import logging

import numpy as np

from imas_codex.embeddings.description import embed_descriptions_batch
from imas_codex.settings import get_sn_desc_name_similarity_threshold

logger = logging.getLogger(__name__)


def compute_desc_name_similarity(
    name: str,
    description: str | None,
) -> float | None:
    """Compute cosine similarity between the grammar-expanded name and description.

    Both texts are embedded fresh using the project embedding server.
    The name string is humanised (underscores → spaces) before embedding,
    mirroring what ``semantic_similarity_check`` does.

    Args:
        name: Standard name string, e.g. ``"electron_temperature"``.
        description: Short description text.  ``None`` or empty → returns ``None``.

    Returns:
        Cosine similarity in ``[0, 1]``, or ``None`` when either embedding
        fails or the description is absent.  The caller should treat ``None``
        as "gate not applicable" (do not route to REFINE_DOCS on failure).
    """
    if not description or not description.strip():
        return None

    name_text = name.replace("_", " ")
    desc_text = description[:500]

    try:
        items = [
            {"id": "name", "_text": name_text},
            {"id": "desc", "_text": desc_text},
        ]
        embed_descriptions_batch(items, text_field="_text")
        name_emb = items[0].get("embedding")
        desc_emb = items[1].get("embedding")
        if name_emb is None or desc_emb is None:
            logger.debug(
                "compute_desc_name_similarity: embed returned None for %s", name
            )
            return None
    except Exception:
        logger.debug(
            "compute_desc_name_similarity: embed failed for %s", name, exc_info=True
        )
        return None

    name_vec = np.asarray(name_emb, dtype=np.float32)
    desc_vec = np.asarray(desc_emb, dtype=np.float32)
    norm_n = float(np.linalg.norm(name_vec))
    norm_d = float(np.linalg.norm(desc_vec))
    if norm_n < 1e-8 or norm_d < 1e-8:
        return None

    return float(np.dot(name_vec, desc_vec) / (norm_n * norm_d))


def should_route_to_refine_docs(
    sim: float | None,
    threshold: float | None = None,
) -> bool:
    """Return True if this similarity score warrants REFINE_DOCS routing.

    ``None`` similarity means the gate could not run (embed failure) — the
    caller should fall through to normal name scoring rather than silently
    routing to REFINE_DOCS.

    Args:
        sim: Cosine similarity from :func:`compute_desc_name_similarity`.
        threshold: Override for the config-derived threshold; defaults to
            :func:`~imas_codex.settings.get_sn_desc_name_similarity_threshold`.
    """
    if sim is None:
        return False
    if threshold is None:
        threshold = get_sn_desc_name_similarity_threshold()
    return sim < threshold
