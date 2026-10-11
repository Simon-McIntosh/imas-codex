"""An error signal's bound is read from whole words, not substrings."""

from unittest.mock import MagicMock

import pytest

from imas_codex.ids.mapping import classify_error_signals


def _classify(description: str, group_key: str = "jt-60sa:src") -> str:
    gc = MagicMock()
    gc.query.return_value = [
        {
            "id": "jt-60sa:src",
            "group_key": group_key,
            "description": description,
            "physics_domain": "general",
            "rep_name": "",
        }
    ]
    (signal,) = classify_error_signals("jt-60sa", gc=gc)
    return signal["probable_error_type"]


@pytest.mark.parametrize(
    "description",
    [
        "plasma flow measurement error estimate",
        "uncertainty of the gas flow below the valve",
        "error of the slow acquisition chain",
        "error bar of the highlighted channel",
    ],
)
def test_a_bound_word_inside_another_word_names_no_bound(description):
    assert _classify(description) == "symmetric"


@pytest.mark.parametrize(
    ("description", "group_key", "bound"),
    [
        ("lower error of the electron temperature", "jt-60sa:te", "lower"),
        ("electron temperature error", "jt-60sa:te_min_err", "lower"),
        ("upper uncertainty of the density", "jt-60sa:ne", "upper"),
        ("high-side error of the density", "jt-60sa:ne", "upper"),
    ],
)
def test_a_bound_named_as_a_word_is_read(description, group_key, bound):
    assert _classify(description, group_key) == bound
