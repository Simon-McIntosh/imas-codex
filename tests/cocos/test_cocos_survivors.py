"""Convention identification from measured signs and explicit declarations."""

import pytest

from imas_codex.cocos import (
    SignObservation,
    ValidationResult,
    dd_cocos_class,
    determine_cocos,
    score_cocos_survivors,
    validate_cocos_from_data,
)


def mast_observations() -> list[SignObservation]:
    """Opposite flux response and q signs at each current polarity."""
    return [
        SignObservation(101, 1, "flux_loop_response", -1),
        SignObservation(101, 1, "q", -1, 1),
        SignObservation(102, -1, "flux_loop_response", 1),
        SignObservation(102, -1, "q", 1, 1),
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
    observations.append(SignObservation(103, 1, "flux_loop_response", 1))
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


def test_validator_reports_contradictory_signs() -> None:
    result = validate_cocos_from_data(
        declared_cocos=17,
        psi_axis=0.0,
        psi_edge=1.0,
        ip=1e6,
        b0=5.0,
        dp_dpsi=1e3,
    )
    assert isinstance(result, ValidationResult)
    assert result.is_consistent is False
    assert result.calculated_cocos is None
    assert result.confidence == 0.0
    assert any("not-cocos-expressible" in error for error in result.inconsistencies)
    assert any("ψ gradient" in error for error in result.inconsistencies)


def test_validator_reports_zero_psi_gradient() -> None:
    result = validate_cocos_from_data(
        declared_cocos=11,
        psi_axis=0.0,
        psi_edge=0.0,
        ip=1e6,
        b0=5.0,
    )
    assert isinstance(result, ValidationResult)
    assert result.is_consistent is False
    assert result.calculated_cocos is None
    assert result.confidence == 0.0
    assert any("zero psi gradient" in error for error in result.inconsistencies)


def mock_dd_graph(
    monkeypatch: pytest.MonkeyPatch, labels: dict[str, str]
) -> list[list[str]]:
    """Supply graph labels while retaining the query's requested path prefixes."""
    from imas_codex.graph import client

    calls: list[list[str]] = []

    class FakeGraph:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def query(self, cypher: str, *, paths: list[dict[str, int | str]]):
            assert "ORDER BY candidate.depth DESC" in cypher
            assert "LIMIT 1" in cypher
            calls.append([str(item["id"]) for item in paths])
            labelled = [item for item in paths if item["id"] in labels]
            if not labelled:
                return []
            deepest = max(labelled, key=lambda item: int(item["depth"]))
            return [{"label": labels[str(deepest["id"])]}]

    monkeypatch.setattr(client, "GraphClient", FakeGraph)
    dd_cocos_class.cache_clear()
    return calls


def test_dd_cocos_class_inherits_graph_label(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = mock_dd_graph(monkeypatch, {"magnetics/flux_loop/flux": "psi_like"})
    path = "magnetics/flux_loop/flux/data"
    assert dd_cocos_class(path) == "psi_like"
    assert calls == [
        ["magnetics", "magnetics/flux_loop", "magnetics/flux_loop/flux", path]
    ]
    assert dd_cocos_class(path) == "psi_like"
    assert len(calls) == 1


def test_dd_cocos_class_prefers_deepest_label(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_dd_graph(
        monkeypatch,
        {
            "equilibrium/time_slice": "ip_like",
            "equilibrium/time_slice/global_quantities/psi_axis": "psi_like",
        },
    )
    assert (
        dd_cocos_class("equilibrium/time_slice/global_quantities/psi_axis/data")
        == "psi_like"
    )


def test_dd_cocos_class_returns_none_without_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = mock_dd_graph(monkeypatch, {})
    assert dd_cocos_class("magnetics/nonexistent/data") is None
    assert len(calls) == 1
