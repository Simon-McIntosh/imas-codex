"""The identity-token collision gate's process-quantity exemption.

A process token spells a cause.  "Due to X" names the quantity X as its cause,
so an identity the installed grammar parses with a physical base names that
quantity and is not the process token standing in for a name.  Every other
collision is still reported.
"""

from __future__ import annotations

from imas_codex.standard_names.export import _identity_token_collision_gate


def _gate(names: list[str]):
    return _identity_token_collision_gate(
        {"equilibrium": [{"name": name} for name in names]}
    )


def test_process_token_parsed_as_quantity_passes() -> None:
    """halo_current is a process token and also the quantity halo current."""
    result = _gate(["halo_current"])

    assert result.passed
    assert result.issues == []


def test_component_token_collision_still_reported() -> None:
    """toroidal is a component token, not a quantity, so it still collides."""
    result = _gate(["toroidal"])

    assert not result.passed
    assert {issue["segment"] for issue in result.issues} == {"component", "coordinate"}
    assert {issue["identity"] for issue in result.issues} == {"toroidal"}
    assert all(issue["type"] == "identity_token_collision" for issue in result.issues)


def test_locus_token_collision_still_reported() -> None:
    """A locus token names no quantity, so publishing it still collides."""
    from imas_standard_names import get_grammar_context

    locus_token = sorted(get_grammar_context()["grammar"]["vocabularies"]["locus_registry"])[0]

    result = _gate([locus_token])

    assert not result.passed
    assert [issue["segment"] for issue in result.issues] == ["locus"]
    assert result.issues[0]["registry_entry"].startswith("locus_registry.yml:")


def test_process_token_that_is_not_a_quantity_still_reported() -> None:
    """anomalous_transport is a process token the grammar cannot parse."""
    result = _gate(["anomalous_transport"])

    assert not result.passed
    assert len(result.issues) == 1
    issue = result.issues[0]
    assert issue["identity"] == "anomalous_transport"
    assert issue["segment"] == "process"
    assert "identity_token_collision" == issue["type"]


def test_exempt_identity_does_not_exempt_its_neighbours() -> None:
    """The exemption is per identity, not a blanket pass over the population."""
    result = _gate(["halo_current", "anomalous_transport"])

    assert not result.passed
    assert [issue["identity"] for issue in result.issues] == ["anomalous_transport"]