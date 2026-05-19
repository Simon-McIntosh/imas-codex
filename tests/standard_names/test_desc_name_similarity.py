"""Tests for Phase 5 desc-name similarity gate.

Covers:
- compute_desc_name_similarity returns ~1.0 for a well-matched (name, desc) pair.
- compute_desc_name_similarity returns a low value for unrelated text.
- should_route_to_refine_docs predicate: below threshold → True, at/above → False.
- Threshold is read from settings (honours pyproject.toml / env override).
- Embedding call is mocked so tests are offline.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

# ── helpers ───────────────────────────────────────────────────────────────────


def _make_embed_mock(pairs: dict[str, list[float]]):
    """Return a mock for ``embed_descriptions_batch`` that fills embeddings in-place.

    ``pairs`` maps the ``_text`` field value to the embedding vector.
    Items whose text doesn't match any key get a zero vector.
    """

    def _fake_embed(items: list[dict], *, text_field: str = "_text") -> None:
        for item in items:
            text = item.get(text_field, "")
            vec = pairs.get(text, [0.0] * 8)
            item["embedding"] = vec

    return _fake_embed


# ── compute_desc_name_similarity ──────────────────────────────────────────────


_EMBED_PATCH = "imas_codex.standard_names.desc_name_sim.embed_descriptions_batch"


def test_compute_high_similarity() -> None:
    """Well-matched name and description produce similarity near 1.0."""
    from imas_codex.standard_names.desc_name_sim import compute_desc_name_similarity

    # Identical direction → cosine = 1.0
    vec = [1.0, 0.5, 0.3, 0.8, 0.2, 0.7, 0.1, 0.9]
    mock = _make_embed_mock(
        {
            "electron temperature": vec,
            "Electron temperature is the thermal energy per electron.": vec,
        }
    )
    with patch(_EMBED_PATCH, side_effect=mock):
        sim = compute_desc_name_similarity(
            "electron_temperature",
            "Electron temperature is the thermal energy per electron.",
        )

    assert sim is not None
    assert sim == pytest.approx(1.0, abs=1e-5)


def test_compute_low_similarity() -> None:
    """Orthogonal embeddings produce similarity near 0.0."""
    from imas_codex.standard_names.desc_name_sim import compute_desc_name_similarity

    vec_name = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    vec_desc = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    mock = _make_embed_mock(
        {
            "electron temperature": vec_name,
            "The quick brown fox jumps over the lazy dog.": vec_desc,
        }
    )
    with patch(_EMBED_PATCH, side_effect=mock):
        sim = compute_desc_name_similarity(
            "electron_temperature",
            "The quick brown fox jumps over the lazy dog.",
        )

    assert sim is not None
    assert sim == pytest.approx(0.0, abs=1e-5)


def test_compute_none_on_empty_description() -> None:
    """None description → returns None without calling embed."""
    from imas_codex.standard_names.desc_name_sim import compute_desc_name_similarity

    with patch(_EMBED_PATCH) as mock_embed:
        sim = compute_desc_name_similarity("electron_temperature", None)
        mock_embed.assert_not_called()

    assert sim is None


def test_compute_none_on_blank_description() -> None:
    """Whitespace-only description → returns None."""
    from imas_codex.standard_names.desc_name_sim import compute_desc_name_similarity

    with patch(_EMBED_PATCH) as mock_embed:
        sim = compute_desc_name_similarity("electron_temperature", "   ")
        mock_embed.assert_not_called()

    assert sim is None


def test_compute_none_on_embed_failure() -> None:
    """Embed failure → returns None (gate not applicable)."""
    from imas_codex.standard_names.desc_name_sim import compute_desc_name_similarity

    with patch(_EMBED_PATCH, side_effect=RuntimeError("server unreachable")):
        sim = compute_desc_name_similarity(
            "electron_temperature", "Electron temperature."
        )

    assert sim is None


def test_compute_none_on_zero_norm() -> None:
    """Zero embedding vector → returns None (degenerate case)."""
    from imas_codex.standard_names.desc_name_sim import compute_desc_name_similarity

    zero_vec = [0.0] * 8
    mock = _make_embed_mock(
        {
            "electron temperature": zero_vec,
            "Electron temperature.": zero_vec,
        }
    )
    with patch(_EMBED_PATCH, side_effect=mock):
        sim = compute_desc_name_similarity(
            "electron_temperature", "Electron temperature."
        )

    assert sim is None


# ── should_route_to_refine_docs ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "sim, threshold, expected",
    [
        (0.40, 0.55, True),  # clearly below
        (0.54, 0.55, True),  # just below
        (0.55, 0.55, False),  # at threshold → not routed
        (0.70, 0.55, False),  # well above
        (None, 0.55, False),  # embed failure → not routed
    ],
)
def test_should_route_to_refine_docs(
    sim: float | None, threshold: float, expected: bool
) -> None:
    from imas_codex.standard_names.desc_name_sim import should_route_to_refine_docs

    assert should_route_to_refine_docs(sim, threshold=threshold) is expected


def test_should_route_uses_settings_default() -> None:
    """Threshold defaults to the value from settings (0.55)."""
    from imas_codex.standard_names.desc_name_sim import should_route_to_refine_docs

    with patch(
        "imas_codex.standard_names.desc_name_sim.get_sn_desc_name_similarity_threshold",
        return_value=0.55,
    ) as _mock:
        assert should_route_to_refine_docs(0.40) is True
        assert should_route_to_refine_docs(0.60) is False
        assert _mock.call_count == 2


# ── settings accessor ─────────────────────────────────────────────────────────


def test_get_sn_desc_name_similarity_threshold_default() -> None:
    """Default threshold resolves to 0.55."""
    from imas_codex.settings import get_sn_desc_name_similarity_threshold

    # No env override — read from config (or fall back to default).
    val = get_sn_desc_name_similarity_threshold()
    assert isinstance(val, float)
    assert val == pytest.approx(0.55)


def test_get_sn_desc_name_similarity_threshold_env_override(monkeypatch) -> None:
    """IMAS_CODEX_SN_DESC_NAME_SIM_THRESHOLD env var overrides config."""
    import importlib

    import imas_codex.settings as settings_mod

    monkeypatch.setenv("IMAS_CODEX_SN_DESC_NAME_SIM_THRESHOLD", "0.70")
    importlib.reload(settings_mod)  # ensure module-level state is refreshed

    from imas_codex.settings import get_sn_desc_name_similarity_threshold

    val = get_sn_desc_name_similarity_threshold()
    assert val == pytest.approx(0.70)
