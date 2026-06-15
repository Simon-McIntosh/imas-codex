"""Tests for semantic near-duplicate annotation of proposed vocab tokens."""

from __future__ import annotations

import numpy as np
import pytest

from imas_codex.standard_names import vocab_semantic_dedup as vsd
from imas_codex.standard_names.vocab_semantic_dedup import (
    DUPLICATE_THRESHOLD,
    annotate_duplicates,
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
    """Force a known per-segment existing-vocabulary map."""

    def _apply(mapping: dict[str, list[str]]):
        monkeypatch.setattr(
            vsd,
            "_existing_tokens_by_segment",
            lambda: {k: sorted(v) for k, v in mapping.items()},
        )

    return _apply


def test_flags_near_synonym(patch_existing):
    patch_existing({"physical_base": ["connection_length"]})
    # Proposed token is geometrically almost identical to the existing one.
    enc = _StubEncoder(
        {
            "field line length": [1.0, 0.05, 0.0],
            "connection length": [0.99, 0.1, 0.0],
        }
    )
    records = [{"segment": "physical_base", "token": "field_line_length"}]
    out = annotate_duplicates(records, encoder=enc)
    assert out[0]["nearest_existing"] == "connection_length"
    assert out[0]["nearest_similarity"] >= DUPLICATE_THRESHOLD
    assert out[0]["likely_duplicate"] is True


def test_distinct_token_not_flagged(patch_existing):
    patch_existing({"subject": ["electron"]})
    enc = _StubEncoder(
        {
            "ion": [0.0, 1.0, 0.0],
            "electron": [1.0, 0.0, 0.0],
        }
    )
    records = [{"segment": "subject", "token": "ion"}]
    out = annotate_duplicates(records, encoder=enc)
    assert out[0]["nearest_existing"] == "electron"
    assert out[0]["nearest_similarity"] < DUPLICATE_THRESHOLD
    assert out[0]["likely_duplicate"] is False


def test_segment_without_existing_vocab_is_noop(patch_existing):
    patch_existing({"physical_base": ["temperature"]})
    enc = _StubEncoder({"temperature": [1.0, 0.0], "foo": [0.0, 1.0]})
    # Proposed token is on a segment that has no existing vocab in the map.
    records = [{"segment": "device", "token": "foo_device"}]
    out = annotate_duplicates(records, encoder=enc)
    assert out[0]["nearest_existing"] is None
    assert out[0]["likely_duplicate"] is False


def test_excludes_exact_self_match(patch_existing):
    # When the proposed token already equals an existing token, surface the
    # nearest *different* neighbour rather than a self-similarity of 1.0.
    patch_existing({"physical_base": ["temperature", "pressure"]})
    enc = _StubEncoder(
        {
            "temperature": [1.0, 0.0, 0.0],
            "pressure": [0.0, 1.0, 0.0],
        }
    )
    records = [{"segment": "physical_base", "token": "temperature"}]
    out = annotate_duplicates(records, encoder=enc)
    assert out[0]["nearest_existing"] == "pressure"
    assert out[0]["nearest_similarity"] < DUPLICATE_THRESHOLD


def test_empty_records_returns_empty():
    assert annotate_duplicates([]) == []


def test_backend_failure_is_advisory(patch_existing, monkeypatch):
    patch_existing({"subject": ["electron"]})

    class _Boom:
        def embed_texts(self, *a, **k):  # noqa: ANN002, ANN003
            raise RuntimeError("backend down")

    records = [{"segment": "subject", "token": "positron"}]
    # Must not raise — annotation is best-effort.
    out = annotate_duplicates(records, encoder=_Boom())
    assert out[0]["nearest_existing"] is None
    assert out[0]["likely_duplicate"] is False


def test_keys_present_even_when_isn_unavailable(monkeypatch):
    monkeypatch.setattr(vsd, "_existing_tokens_by_segment", dict)
    records = [{"segment": "physical_base", "token": "whatever"}]
    out = annotate_duplicates(records)
    assert out[0]["nearest_existing"] is None
    assert out[0]["nearest_similarity"] is None
    assert out[0]["likely_duplicate"] is False
