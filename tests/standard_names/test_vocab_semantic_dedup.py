"""Tests for the runtime component-token reuse check.

The check embeds a proposed (candidate-new) grammar token and compares it by
cosine similarity against the registered tokens of its same segment, returning
the nearest neighbour only when it scores at/above the threshold.  A stub
encoder maps known phrases to fixed unit vectors so the test controls the
similarity geometry without a live embedding backend.
"""

from __future__ import annotations

import numpy as np
import pytest

from imas_codex.standard_names import vocab_semantic_dedup as vsd
from imas_codex.standard_names.vocab_semantic_dedup import (
    DUPLICATE_THRESHOLD,
    TokenNeighbour,
    nearest_registered_token,
    nearest_registered_tokens,
)


class _StubEncoder:
    """Deterministic encoder mapping known phrases to fixed unit vectors.

    Phrases are matched by a substring key so the test controls similarity
    geometry without a live embedding backend.
    """

    def __init__(self, vectors: dict[str, list[float]]):
        self._vectors = {k: np.asarray(v, dtype=float) for k, v in vectors.items()}
        dim = len(next(iter(self._vectors.values())))
        self._default = np.zeros(dim, dtype=float)

    def embed_texts(self, texts, *, prompt_name=None, **kwargs):  # noqa: ANN001
        rows = []
        for text in texts:
            vec = None
            for key, v in self._vectors.items():
                if key in text:
                    vec = v
                    break
            rows.append(vec if vec is not None else self._default)
        return np.vstack(rows)


@pytest.fixture
def patch_existing(monkeypatch):
    """Force a known per-segment registered-vocabulary map."""

    def _apply(mapping: dict[str, list[str]]):
        monkeypatch.setattr(
            vsd,
            "_existing_tokens_by_segment",
            lambda: {k: sorted(v) for k, v in mapping.items()},
        )

    return _apply


def test_flags_near_synonym(patch_existing):
    patch_existing({"physical_base": ["connection_length"]})
    # Proposed token is geometrically almost identical to the registered one.
    enc = _StubEncoder(
        {
            "field line length": [1.0, 0.05, 0.0],
            "connection length": [0.99, 0.1, 0.0],
        }
    )
    hit = nearest_registered_token("field_line_length", "physical_base", encoder=enc)
    assert isinstance(hit, TokenNeighbour)
    assert hit.nearest_token == "connection_length"
    assert hit.proposed_token == "field_line_length"
    assert hit.segment == "physical_base"
    assert hit.similarity >= DUPLICATE_THRESHOLD


def test_distinct_token_not_flagged(patch_existing):
    patch_existing({"subject": ["electron"]})
    enc = _StubEncoder(
        {
            "ion": [0.0, 1.0, 0.0],
            "electron": [1.0, 0.0, 0.0],
        }
    )
    # Orthogonal vectors → similarity 0 < threshold → no hit.
    hit = nearest_registered_token("ion", "subject", encoder=enc)
    assert hit is None


def test_segment_without_vocab_is_none(patch_existing):
    patch_existing({"physical_base": ["temperature"]})
    enc = _StubEncoder({"temperature": [1.0, 0.0], "foo": [0.0, 1.0]})
    # Proposed token is on a segment absent from the registered map.
    hit = nearest_registered_token("foo_device", "device", encoder=enc)
    assert hit is None


def test_excludes_exact_self_match(patch_existing):
    # When the proposed token already equals a registered token, surface the
    # nearest *different* neighbour rather than a self-similarity of 1.0.
    patch_existing({"physical_base": ["temperature", "pressure"]})
    enc = _StubEncoder(
        {
            "temperature": [1.0, 0.0, 0.0],
            "pressure": [0.0, 1.0, 0.0],
        }
    )
    # pressure is orthogonal to temperature → below threshold → no hit, but the
    # self-match (similarity 1.0) must NOT be returned.
    hit = nearest_registered_token("temperature", "physical_base", encoder=enc)
    assert hit is None


def test_sole_registered_token_equal_to_proposed_is_none(patch_existing):
    # Single-element vocab that IS the proposed token: nothing distinct to
    # compare against.
    patch_existing({"physical_base": ["temperature"]})
    enc = _StubEncoder({"temperature": [1.0, 0.0]})
    hit = nearest_registered_token("temperature", "physical_base", encoder=enc)
    assert hit is None


def test_threshold_override_disables(patch_existing):
    # A threshold above 1.0 can never be met → check is effectively off
    # (used for the OFF arm of the A/B).
    patch_existing({"physical_base": ["connection_length"]})
    enc = _StubEncoder(
        {
            "field line length": [1.0, 0.05, 0.0],
            "connection length": [0.99, 0.1, 0.0],
        }
    )
    hit = nearest_registered_token(
        "field_line_length", "physical_base", encoder=enc, threshold=1.01
    )
    assert hit is None


def test_encoder_failure_is_advisory(patch_existing):
    patch_existing({"subject": ["electron"]})

    class _Boom:
        def embed_texts(self, *a, **k):  # noqa: ANN002, ANN003
            raise RuntimeError("backend down")

    # Must not raise — the check is best-effort.
    hit = nearest_registered_token("positron", "subject", encoder=_Boom())
    assert hit is None


def test_isn_unavailable_is_none(monkeypatch):
    monkeypatch.setattr(vsd, "_existing_tokens_by_segment", dict)
    hit = nearest_registered_token("whatever", "physical_base")
    assert hit is None


# ---------------------------------------------------------------------------
# Batch variant
# ---------------------------------------------------------------------------


def test_batch_groups_and_flags(patch_existing):
    patch_existing(
        {
            "physical_base": ["connection_length"],
            "subject": ["electron"],
        }
    )
    enc = _StubEncoder(
        {
            "field line length": [1.0, 0.05, 0.0],
            "connection length": [0.99, 0.1, 0.0],
            "ion": [0.0, 1.0, 0.0],
            "electron": [1.0, 0.0, 0.0],
        }
    )
    items = [
        ("field_line_length", "physical_base"),  # near-synonym → flagged
        ("ion", "subject"),  # orthogonal → not flagged
    ]
    out = nearest_registered_tokens(items, encoder=enc)
    assert ("field_line_length", "physical_base") in out
    assert out[("field_line_length", "physical_base")].nearest_token == (
        "connection_length"
    )
    assert ("ion", "subject") not in out


def test_batch_empty_returns_empty():
    assert nearest_registered_tokens([]) == {}


def test_batch_skips_unknown_segments(patch_existing):
    patch_existing({"physical_base": ["temperature"]})
    enc = _StubEncoder({"temperature": [1.0, 0.0], "foo": [0.0, 1.0]})
    out = nearest_registered_tokens([("foo_device", "device")], encoder=enc)
    assert out == {}


def test_batch_encoder_failure_is_advisory(patch_existing):
    patch_existing({"subject": ["electron"]})

    class _Boom:
        def embed_texts(self, *a, **k):  # noqa: ANN002, ANN003
            raise RuntimeError("backend down")

    out = nearest_registered_tokens([("positron", "subject")], encoder=_Boom())
    assert out == {}
