"""Finalize a drained family only after its current members pass the guard."""

from __future__ import annotations

from typing import Any

import pytest
from click.testing import CliRunner

from imas_codex.standard_names import graph_ops, harmonize
from imas_codex.standard_names.graph_store import (
    graph_docs_persistence as _owning_graph_docs_persistence,
    graph_source_bindings as _owning_graph_source_bindings,
)
from tests.standard_names.test_harmonize_reverse_links import (
    _ANCHOR,
    _BOUNDARY,
    _PROBE,
    _family_nodes,
    _findings,
    _LinkGraph,
)


def _fixture(monkeypatch: Any) -> _LinkGraph:
    from imas_codex.graph import client as graph_client
    from imas_codex.standard_names import protection

    graph = _LinkGraph(_family_nodes())
    monkeypatch.setattr(graph_client, "GraphClient", lambda *a, **k: graph)
    monkeypatch.setattr(protection, "_frozen_roster_requests", lambda *_a: {})
    monkeypatch.setattr(
        _owning_graph_source_bindings,
        "write_enrichment_results",
        graph.write_enrichment_results,
    )

    def stamp(families: list[dict[str, Any]]) -> int:
        for family in families:
            for member_id in family["members"]:
                graph.node(member_id)["harmonized_group_signature"] = family[
                    "signature"
                ]
        return len(families)

    monkeypatch.setattr(
        _owning_graph_docs_persistence, "stamp_harmonized_families", stamp
    )
    monkeypatch.setattr(harmonize, "_clear_family_anchor", lambda *_a: None)
    return graph


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("docs_stage", "reviewed", "docs_stage='reviewed'"),
        ("documentation", "It is obtained from a diagnostic.", "banned prose"),
        ("description", "In practice, the estimate varies.", "banned prose"),
        ("validation_status", "quarantined", "validation_status='quarantined'"),
    ],
)
def test_refuses_unclean_member_before_any_write(
    monkeypatch: Any, field: str, value: str, reason: str
) -> None:
    graph = _fixture(monkeypatch)
    graph.node(_BOUNDARY)[field] = value

    with pytest.raises(harmonize.HarmonizeFinalizeError) as exc:
        harmonize.finalize_families(graph, seeds=[_ANCHOR])

    assert _BOUNDARY in str(exc.value)
    assert reason in str(exc.value)
    assert graph.written == []
    assert graph.node(_ANCHOR)["harmonized_group_signature"] is None


def test_quarantined_member_cannot_gain_a_stamp(monkeypatch: Any) -> None:
    graph = _fixture(monkeypatch)
    graph.node(_BOUNDARY)["validation_status"] = "quarantined"

    try:
        harmonize.finalize_families(graph, seeds=[_ANCHOR])
    except harmonize.HarmonizeFinalizeError as exc:
        assert _BOUNDARY in str(exc)
    else:
        pytest.fail(
            "quarantined member was finalized: "
            f"stamped={graph.node(_BOUNDARY)['harmonized_group_signature'] is not None}"
        )


def test_clean_family_reconciles_stamps_and_plans_no_calls(
    monkeypatch: Any,
) -> None:
    from imas_codex.cli import sn as sn_module
    from imas_codex.cli.discover import common
    from imas_codex.discovery.base import llm

    graph = _fixture(monkeypatch)
    calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        common, "setup_logging", lambda *args, **_kw: calls.append(args)
    )

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("a model call is forbidden during finalization")

    monkeypatch.setattr(llm, "call_llm_structured", forbidden)
    monkeypatch.setattr(llm, "acall_llm_structured", forbidden)
    monkeypatch.setattr(sn_module, "_docs_drain_factory", forbidden)

    preview = CliRunner().invoke(
        sn_module.sn, ["harmonize", "--family", _ANCHOR, "--finalize", "--dry-run"]
    )
    assert preview.exit_code == 0, preview.output
    assert "ready" in preview.output
    assert "missing_reverse_pairs=2" in preview.output
    assert graph.written == []

    result = CliRunner().invoke(
        sn_module.sn, ["harmonize", "--family", _ANCHOR, "--finalize"]
    )
    assert result.exit_code == 0, result.output
    assert "stamped" in result.output
    assert "0 model calls" in result.output
    assert calls == [("sn", "sn-harmonize"), ("sn", "sn-harmonize")]
    assert _findings(graph, "missing_reverse") == []
    assert f"name:{_ANCHOR}" in graph.node(_BOUNDARY)["links"]
    assert f"name:{_PROBE}" in graph.node(_ANCHOR)["links"]
    signature = harmonize.group_signature(
        [graph.node(member_id) for member_id in (_ANCHOR, _BOUNDARY, _PROBE)]
    )
    assert all(
        graph.node(member_id)["harmonized_group_signature"] == signature
        for member_id in (_ANCHOR, _BOUNDARY, _PROBE)
    )
    plans = harmonize.plan_family_applies(graph, seeds=[_ANCHOR])
    assert len(plans) == 1
    assert plans[0]["decision"] == "harmonized"
    assert plans[0]["planned_calls"] == 0


def test_stamp_readback_requires_every_member_current_signature(
    monkeypatch: Any,
) -> None:
    graph = _fixture(monkeypatch)

    def incomplete_stamp(families: list[dict[str, Any]]) -> int:
        family = families[0]
        for member_id in family["members"][:-1]:
            graph.node(member_id)["harmonized_group_signature"] = family["signature"]
        return 1

    monkeypatch.setattr(
        _owning_graph_docs_persistence, "stamp_harmonized_families", incomplete_stamp
    )

    with pytest.raises(harmonize.HarmonizeFinalizeError) as exc:
        harmonize.finalize_families(graph, seeds=[_ANCHOR])

    assert "Current family signature missing" in str(exc.value)
    assert _PROBE in str(exc.value)
    assert graph.node(_PROBE)["harmonized_group_signature"] is None
    # The planner ignores null stamps when comparing the stored signature set.
    # Its zero-call result alone therefore cannot prove every member was stamped.
    assert (
        harmonize.plan_family_applies(graph, seeds=[_ANCHOR])[0]["planned_calls"] == 0
    )


def test_cli_refusal_names_member_and_reason(monkeypatch: Any) -> None:
    from imas_codex.cli.discover import common
    from imas_codex.cli.sn import sn

    graph = _fixture(monkeypatch)
    monkeypatch.setattr(common, "setup_logging", lambda *_a, **_kw: None)
    graph.node(_BOUNDARY)["validation_status"] = "quarantined"

    result = CliRunner().invoke(sn, ["harmonize", "--family", _ANCHOR, "--finalize"])

    assert result.exit_code != 0
    assert _BOUNDARY in result.output
    assert "quarantined" in result.output
    assert graph.written == []


@pytest.mark.parametrize("dry_run", [True, False])
def test_catalog_protected_member_refuses_before_any_write(
    monkeypatch: Any, dry_run: bool
) -> None:
    graph = _fixture(monkeypatch)
    graph.node(_PROBE)["name_stage"] = "approved"

    with pytest.raises(harmonize.HarmonizeFinalizeError) as exc:
        harmonize.finalize_families(graph, seeds=[_ANCHOR], dry_run=dry_run)

    assert _PROBE in str(exc.value)
    assert "protected" in str(exc.value)
    assert graph.written == []
    assert graph.node(_ANCHOR)["harmonized_group_signature"] is None


def test_late_protection_refuses_all_reconciliation_writes(monkeypatch: Any) -> None:
    from imas_codex.standard_names import protection

    graph = _fixture(monkeypatch)
    graph.node(_PROBE)["name_stage"] = "approved"
    monkeypatch.setattr(protection, "recommended_name_protections", lambda *_a: {})

    with pytest.raises(harmonize.HarmonizeFinalizeError) as exc:
        harmonize.finalize_families(graph, seeds=[_ANCHOR])

    assert _PROBE in str(exc.value)
    assert graph.written == []
    assert graph.node(_ANCHOR)["harmonized_group_signature"] is None


def test_open_review_roster_refuses_dry_run(monkeypatch: Any) -> None:
    from imas_codex.standard_names import protection

    graph = _fixture(monkeypatch)
    monkeypatch.setattr(
        protection,
        "_frozen_roster_requests",
        lambda *_a: {_PROBE: "candidate request is open"},
    )

    with pytest.raises(harmonize.HarmonizeFinalizeError) as exc:
        harmonize.finalize_families(graph, seeds=[_ANCHOR], dry_run=True)

    assert _PROBE in str(exc.value)
    assert "candidate request is open" in str(exc.value)
    assert graph.written == []
