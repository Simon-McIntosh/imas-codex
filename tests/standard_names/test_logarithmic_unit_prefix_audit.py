"""Regression coverage for logarithm operators over logarithmic units."""

from __future__ import annotations

from imas_codex.standard_names import audits

_DECIBEL_RATIO_NAME = (
    "logarithm_of_spectral_signal_to_noise_ratio_of_spectrometer_channel"
)


def test_decibel_quantity_with_logarithm_prefix_is_quarantined() -> None:
    candidate = {"id": _DECIBEL_RATIO_NAME, "unit": "dB"}

    issues = audits.run_audits(candidate)
    matching = [issue for issue in issues if "logarithmic_unit_prefix_check" in issue]

    assert len(matching) == 1
    assert audits.has_critical_audit_failure(matching)
    assert "unit already encodes a logarithm" in matching[0]


def test_logarithm_prefix_over_non_logarithmic_unit_passes() -> None:
    candidate = {"id": "logarithm_of_electron_density", "unit": "m^-3"}

    assert audits.logarithmic_unit_prefix_check(candidate) == []


def test_logarithm_prefix_tokens_follow_isn_grammar(monkeypatch) -> None:
    import imas_standard_names

    context = {
        "grammar": {
            "vocabularies": {
                "operators": {
                    "natural_logarithm": {"kind": "unary_prefix"},
                    "logarithm": {"kind": "binary"},
                    "average": {"kind": "unary_prefix"},
                }
            }
        }
    }
    audits._isn_logarithm_prefix_tokens.cache_clear()
    monkeypatch.setattr(imas_standard_names, "get_grammar_context", lambda: context)
    try:
        assert audits._isn_logarithm_prefix_tokens() == {"natural_logarithm"}
    finally:
        audits._isn_logarithm_prefix_tokens.cache_clear()


def test_logarithmic_units_follow_pint_converter_metadata() -> None:
    from imas_codex.units import unit_registry
    from imas_codex.units.dd_unit_exceptions import canonical_or_none

    expected = {
        canonical
        for spelling, definition in unit_registry._units.items()
        if getattr(getattr(definition, "converter", None), "is_logarithmic", False)
        if (canonical := canonical_or_none(str(spelling))) is not None
    }

    assert audits._logarithmic_unit_symbols() == expected
    assert canonical_or_none("dB") in expected
    assert canonical_or_none("1") not in expected
