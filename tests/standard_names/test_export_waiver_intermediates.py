"""A waiver refusal sees every identity the waived source's chain publishes.

A cut publishes whatever the pipeline accepted, and a manifest source row
carries only its chain's terminal identity. A declared-waived source whose
chain accepted an identity at an intermediate stage therefore has a published
identity the terminal-only refusal never sees. The refusal must consider the
whole chain: a published intermediate is exactly the contradiction a waiver
cannot hold quietly, and a chain whose non-terminal identities never reached
the cut must still pass as waived.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("imas_standard_names")

from imas_codex.standard_names.export import (  # noqa: E402
    _WAIVED_CHAIN_DEPTH_BOUND,
    _WAIVED_SOURCES_DECLARATION,
    WaivedSourceDeclarationError,
    _load_waived_source_paths,
    _source_disposition,
    run_export,
)
from tests.standard_names.conftest import clean_source_audit_graph

# A declared path from the shipped declaration, restated so the test asserts
# membership rather than reading the declaration back out of itself.
_DECLARED_WAIVED = "equilibrium/time_slice/time"

# The identity an intermediate stage of the waived source's chain carries. It is
# deliberately distinct from the terminal so a terminal-only refusal misses it.
_INTERMEDIATE_IDENTITY = "time_coordinate_of_equilibrium_slice"
_TERMINAL_IDENTITY = "equilibrium_slice_time_terminal"

# A chain with more members than the traversal bound can read. An exportable
# identity sits past the bound (in the cut but beyond the truncated read) and the
# terminal that follows it is not itself exported, so only the bound check can
# catch the published identity, never the terminal path.
_DEEP_PUBLISHED_IDENTITY = "replacement_intermediate_published_beyond_the_bound"


def _over_long_chain() -> list[str]:
    """A successor chain one link longer than the bound, ending in a passing terminal."""
    bound = _WAIVED_CHAIN_DEPTH_BOUND
    chain = [f"superseded_successor_{index}" for index in range(bound + 1)]
    chain.append(_DEEP_PUBLISHED_IDENTITY)
    chain.append(_TERMINAL_IDENTITY)
    return chain


def _waived_paths() -> frozenset[str]:
    return _load_waived_source_paths(_WAIVED_SOURCES_DECLARATION)


def test_a_waived_source_whose_chain_publishes_an_intermediate_refuses() -> None:
    """An intermediate identity in the cut is the contradiction the waiver owes."""
    with pytest.raises(WaivedSourceDeclarationError) as excinfo:
        _source_disposition(
            source_path=_DECLARED_WAIVED,
            waived_paths=_waived_paths(),
            non_nameable_reason="",
            standard_name_id=_TERMINAL_IDENTITY,
            exported_ids={_INTERMEDIATE_IDENTITY},
            exclusion_reason=None,
            chain_identities={_INTERMEDIATE_IDENTITY, _TERMINAL_IDENTITY},
        )

    message = str(excinfo.value)
    assert _DECLARED_WAIVED in message
    assert _INTERMEDIATE_IDENTITY in message


def test_a_waived_source_whose_chain_publishes_no_identity_stays_waived() -> None:
    """A chain that never reached the cut leaves the source a settled waiver."""
    disposition, reason = _source_disposition(
        source_path=_DECLARED_WAIVED,
        waived_paths=_waived_paths(),
        non_nameable_reason="",
        standard_name_id=_TERMINAL_IDENTITY,
        exported_ids={"an_identity_from_an_unrelated_source"},
        exclusion_reason=None,
        chain_identities={_INTERMEDIATE_IDENTITY, _TERMINAL_IDENTITY},
    )

    assert (disposition, reason) == ("waived", "")


@clean_source_audit_graph
class _ChainGraphClient:
    """Answer the successor-chain lookup; every other query returns no rows.

    The chain is a fixture: an ordered identity list from the source's produced
    name to the far end of its successor chain. The mock mirrors the bounded
    traversal the export runs, so a fixture chain longer than the bound is
    truncated to the members the graph would return, and the member at the mock
    chain's bound is reported as the boundary identity when it still has an
    outgoing successor. That is the fact the real query reads from the graph, so
    a fixture reaching the bound exercises the same refusal the live read raises.
    """

    def __init__(self, chains: dict[str, list[str]]) -> None:
        self._chains = chains

    def __enter__(self) -> _ChainGraphClient:
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def query(self, cypher: str, **params):
        if "HAS_SUCCESSOR" in cypher:
            bound = _WAIVED_CHAIN_DEPTH_BOUND
            rows = []
            for path in params.get("paths", []):
                chain = list(self._chains.get(path, []))
                if len(chain) > bound + 1:
                    rows.append(
                        {
                            "source_path": path,
                            "identities": chain[: bound + 1],
                            "boundary_identities": [chain[bound]],
                        }
                    )
                else:
                    rows.append(
                        {
                            "source_path": path,
                            "identities": chain,
                            "boundary_identities": [],
                        }
                    )
            return rows
        return []


def _candidate(name: str) -> dict:
    return {
        "id": name,
        "name_stage": "accepted",
        "status": "draft",
        "validation_status": "valid",
        "_validation_observed_at": "2026-09-09T00:00:00Z",
        "review_quorum_shortfall": None,
        "docs_stage": "accepted",
        "_has_docs_review": True,
        "_has_winning_docs_review": True,
        "docs_review_quorum_shortfall": None,
        "_has_dd_source_binding": True,
        "_has_derived_producer": False,
        "_has_non_derived_producer": True,
        "_is_parent": False,
        "_has_live_child": False,
        "reviewerless": False,
        "reviewer_score_name": 0.95,
        "description": f"Description for {name}.",
        "documentation": f"Documentation for {name}.",
        "kind": "scalar",
        "unit": "1",
        "physics_domain": "general",
        "links": [],
    }


def _run_fixture_export(
    staging_dir: Path,
    population: list[dict],
    chains: dict[str, list[str]],
    manifest_sources: list[dict],
):
    with (
        patch(
            "imas_codex.standard_names.export._fetch_export_population",
            return_value=population,
        ),
        patch(
            "imas_codex.graph.client.GraphClient",
            return_value=_ChainGraphClient(chains),
        ),
        patch(
            "imas_codex.standard_names.export._validate_entry",
            side_effect=lambda entry: entry,
        ),
        patch(
            "imas_codex.standard_names.export._fetch_deprecation_stubs",
            return_value=[],
        ),
        patch(
            "imas_codex.standard_names.export._fetch_ordering_edges_for_domain",
            return_value=([], set()),
        ),
        patch("imas_codex.standard_names.export._write_domain_yaml"),
    ):
        return run_export(
            staging_dir,
            min_score=0.85,
            skip_gate=True,
            force=True,
            include_sources=False,
            manifest_sources=manifest_sources,
        )


def _waived_manifest_row() -> dict:
    return {
        "source_path": _DECLARED_WAIVED,
        "terminal_stage": None,
        "source_status": "produced",
        "standard_name_id": _TERMINAL_IDENTITY,
        "non_nameable_reason": "",
    }


def test_the_cut_refuses_a_waived_source_whose_intermediate_is_published(
    tmp_path: Path,
) -> None:
    """The export leg itself reads the chain and refuses the contradicted cut."""
    population = [
        _candidate("an_identity_from_an_unrelated_source"),
        _candidate(_INTERMEDIATE_IDENTITY),
    ]
    chains = {_DECLARED_WAIVED: [_INTERMEDIATE_IDENTITY, _TERMINAL_IDENTITY]}

    with pytest.raises(WaivedSourceDeclarationError):
        _run_fixture_export(tmp_path, population, chains, [_waived_manifest_row()])


def test_the_cut_carries_a_waived_source_whose_chain_reached_nothing(
    tmp_path: Path,
) -> None:
    """A waived source whose chain published no identity is recorded waived."""
    population = [_candidate("an_identity_from_an_unrelated_source")]
    chains = {_DECLARED_WAIVED: [_TERMINAL_IDENTITY]}

    report = _run_fixture_export(
        tmp_path,
        population,
        chains,
        [_waived_manifest_row()],
    )

    records = {
        record.source_path: record.disposition
        for record in report.source_disposition_records
    }
    assert records == {_DECLARED_WAIVED: "waived"}


def test_the_cut_refuses_a_waived_source_whose_chain_exceeds_the_bound(
    tmp_path: Path,
) -> None:
    """A chain past the read bound refuses, naming the source and the bound identity."""
    chain = _over_long_chain()
    population = [
        _candidate("an_identity_from_an_unrelated_source"),
        _candidate(_DEEP_PUBLISHED_IDENTITY),
    ]
    chains = {_DECLARED_WAIVED: chain}

    with pytest.raises(WaivedSourceDeclarationError) as excinfo:
        _run_fixture_export(tmp_path, population, chains, [_waived_manifest_row()])

    message = str(excinfo.value)
    assert _DECLARED_WAIVED in message
    assert chain[_WAIVED_CHAIN_DEPTH_BOUND] in message


def test_the_cut_reads_a_chain_reaching_exactly_the_bound_without_refusing(
    tmp_path: Path,
) -> None:
    """A chain that ends at the bound is read in full and leaves the source waived."""
    bound = _WAIVED_CHAIN_DEPTH_BOUND
    chain = [f"superseded_successor_{index}" for index in range(bound + 1)]
    population = [_candidate("an_identity_from_an_unrelated_source")]
    chains = {_DECLARED_WAIVED: chain}

    report = _run_fixture_export(
        tmp_path,
        population,
        chains,
        [_waived_manifest_row()],
    )

    records = {
        record.source_path: record.disposition
        for record in report.source_disposition_records
    }
    assert records == {_DECLARED_WAIVED: "waived"}


@pytest.fixture(autouse=True)
def _pin_export_provenance(pinned_export_provenance):
    """Drive this module's exports with archive provenance pinned."""
    return pinned_export_provenance
