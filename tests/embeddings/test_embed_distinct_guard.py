"""Encoder refuses a batch where different texts share one vector.

A backend that returns the same vector for every input makes each affected
name unfindable: its vector equals another name's, so semantic search and
near-duplicate detection cannot tell them apart. These tests drive that
backend shape through ``Encoder.embed_texts`` and assert the refusal, and
that a healthy backend still passes through.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from imas_codex.embeddings.config import EmbeddingBackend, EncoderConfig
from imas_codex.embeddings.encoder import DegenerateEmbeddingError, Encoder


@pytest.fixture
def local_encoder() -> Encoder:
    """An encoder whose model load is stubbed out."""
    config = EncoderConfig(
        model_name="test-model",
        batch_size=8,
        use_rich=False,
        backend=EmbeddingBackend.LOCAL,
    )
    with patch.object(Encoder, "_load_model"):
        enc = Encoder(config=config)
    enc._model = MagicMock()
    enc._model.device = "cpu"
    return enc


class TestIdenticalVectorRefusal:
    def test_distinct_texts_one_vector_is_refused(self, local_encoder):
        """Two different texts returning one vector raise the named error."""
        local_encoder._model.encode.return_value = np.ones((2, 256), dtype=np.float32)

        with pytest.raises(DegenerateEmbeddingError):
            local_encoder.embed_texts(["alpha quantity", "beta quantity"])

    def test_refusal_names_both_texts(self, local_encoder):
        """The refusal identifies the pair so the cause is readable."""
        local_encoder._model.encode.return_value = np.zeros((2, 16), dtype=np.float32)

        with pytest.raises(DegenerateEmbeddingError) as excinfo:
            local_encoder.embed_texts(["first text", "second text"])

        message = str(excinfo.value)
        assert "first text" in message
        assert "second text" in message

    def test_distinct_vectors_pass_through(self, local_encoder):
        """A healthy backend is returned unchanged."""
        vectors = np.stack([np.eye(4)[0], np.eye(4)[1]], axis=0)
        local_encoder._model.encode.return_value = vectors

        result = local_encoder.embed_texts(["one", "two"])

        assert result.shape == (2, 4)
        np.testing.assert_array_equal(result, vectors)

    def test_single_text_is_not_checked(self, local_encoder):
        """One text cannot collide with another in the same call."""
        local_encoder._model.encode.return_value = np.zeros((1, 256), dtype=np.float32)

        result = local_encoder.embed_texts(["only text"])

        assert result.shape == (1, 256)

    def test_repeated_identical_text_is_allowed(self, local_encoder):
        """The same text twice legitimately yields the same vector."""
        local_encoder._model.encode.return_value = np.ones((2, 8), dtype=np.float32)

        result = local_encoder.embed_texts(["same text", "same text"])

        assert result.shape == (2, 8)

    def test_three_texts_one_vector_is_refused(self, local_encoder):
        """A constant batch of three distinct texts is refused."""
        local_encoder._model.encode.return_value = np.full(
            (3, 32), 0.5, dtype=np.float32
        )

        with pytest.raises(DegenerateEmbeddingError):
            local_encoder.embed_texts(["a", "b", "c"])
