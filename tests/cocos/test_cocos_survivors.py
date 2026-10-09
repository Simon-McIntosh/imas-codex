"""Convention identification from measured signs and explicit declarations."""

import pytest

from imas_codex.cocos import (
    ConventionObservation,
    dd_cocos_class,
    determine_cocos,
    score_cocos_survivors,
)


def mast_observations() -> list[ConventionObservation]:
    """Opposite flux response and q signs at each current polarity."""
    return [
        ConventionObservation(101, 1, "flux_loop_response", -1),
        ConventionObservation(101, 1, "q", -1, 1),
        ConventionObservation(102, -1, "flux_loop_response", 1),
        ConventionObservation(102, -1, "q", 1, 1),
    ]


def test_handedness_declaration_reduces_mast_survivors() -> None:
    observations = mast_observations()
    without_flux_units = score_cocos_survivors(observations)
    assert without_flux_units.survivors == {3, 4, 13, 14}
    assert without_flux_units.coefficient_basis["e_bp"] == "requires-declaration"
    unresolved = score_cocos_survivors(observations, e_bp=0)
    assert unresolved.survivors == {3, 4}
    assert unresolved.cocos is None
    assert unresolved.coefficient_basis["sigma_bp"] == "measured"
    assert unresolved.coefficient_basis["sigma_r_phi_z"] == "requires-declaration"
    assert unresolved.factor_for_class("psi_like", cocos_out=17) == pytest.approx(
        2 * 3.141592653589793
    )
    assert unresolved.factor_for_class("ip_like", cocos_out=17) is None

    declared = score_cocos_survivors(observations, e_bp=0, sigma_r_phi_z=1)
    assert declared.survivors == {3}
    assert declared.cocos == 3
    assert declared.declarations == {"e_bp": 0, "sigma_r_phi_z": 1}


def test_missing_current_polarity_lowers_confidence() -> None:
    complete = score_cocos_survivors(mast_observations(), e_bp=0)
    incomplete = score_cocos_survivors(mast_observations()[:2], e_bp=0)
    assert complete.status == "consistent"
    assert incomplete.status == "polarity-incomplete"
    assert incomplete.confidence < complete.confidence


def test_contradictory_signs_are_not_cocos_expressible() -> None:
    observations = mast_observations()
    observations.append(ConventionObservation(103, 1, "flux_loop_response", 1))
    result = score_cocos_survivors(observations, e_bp=0)
    assert result.survivors == frozenset()
    assert result.status == "not-cocos-expressible"
    assert result.cocos is None
    assert result.factor_for_class("psi_like", cocos_out=17) is None


def test_determine_uses_unique_survivor_and_names_assumptions() -> None:
    result = determine_cocos(0.0, -0.2, 1e6, 5.0, q=-2.0)
    assert result.cocos == 3
    assert result.assumed_declarations == {"e_bp": 0, "sigma_r_phi_z": 1}
    assert tuple(result) == (result.cocos, result.confidence)
    assert (
        result.confidence
        < score_cocos_survivors(mast_observations(), e_bp=0, sigma_r_phi_z=1).confidence
    )


def test_dd_cocos_class_inherits_flux_label_only() -> None:
    assert dd_cocos_class("magnetics/flux_loop/flux/data") == "psi_like"
    assert dd_cocos_class("magnetics/flux_loop/flux") == "psi_like"
    assert dd_cocos_class("magnetics/nonexistent/data") is None


def test_dd_cocos_class_reads_ancestor_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from imas_codex.cocos import transforms

    monkeypatch.setattr(
        transforms,
        "_dd_cocos_labels",
        lambda dd_version: {
            "equilibrium/time_slice/global_quantities/psi_axis": "psi_like"
        },
    )
    assert (
        dd_cocos_class("equilibrium/time_slice/global_quantities/psi_axis/data")
        == "psi_like"
    )
    assert dd_cocos_class("equilibrium/time_slice/global_quantities/ip") is None
