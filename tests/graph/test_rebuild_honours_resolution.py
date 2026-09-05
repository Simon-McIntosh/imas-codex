"""DD path rebuilds preserve exact-version reviewed unit authority."""

from __future__ import annotations

import json

import pytest

from imas_codex.graph.build_dd import _batch_create_path_nodes
from imas_codex.graph.models import (
    DDResolutionField,
    DDResolutionStatus,
    DDResolutionValueKind,
)

_PATH = "equilibrium/time_slice/constraints/n_e/reconstructed"
_VERSION = "4.1.1"


class _RebuildProbe:
    """Minimal graph port that records the scalar and unit-edge writes."""

    def __init__(self, resolutions: list[dict[str, str]]) -> None:
        self.resolutions = resolutions
        self.scalar: str | None = None
        self.edge: str | None = None

    def query(self, cypher: str, **parameters: object) -> list[dict[str, object]]:
        if "MATCH (resolution:DDResolution)" in cypher:
            assert "resolution.dd_version = $version" in cypher
            assert "resolution.status = $status" in cypher
            assert "resolution.field = $field" in cypher
            path_ids = parameters["path_ids"]
            assert isinstance(path_ids, list)
            return [
                {
                    "path": resolution["path"],
                    "effective_kind": resolution["effective_kind"],
                    "effective_value": resolution["effective_value"],
                }
                for resolution in self.resolutions
                if resolution["path"] in path_ids
                and resolution["dd_version"] == parameters["version"]
                and resolution["status"] == parameters["status"]
                and resolution["field"] == parameters["field"]
            ]
        if "SET path.name = p.name" in cypher:
            paths = parameters["paths"]
            assert isinstance(paths, list)
            self.scalar = paths[0]["unit"]
        if "MERGE (path)-[:HAS_UNIT]->(u)" in cypher:
            paths = parameters["paths"]
            assert isinstance(paths, list)
            self.edge = paths[0]["unit"]
        return []


def _resolution(
    *,
    status: DDResolutionStatus = DDResolutionStatus.active,
    version: str = _VERSION,
) -> dict[str, str]:
    return {
        "path": _PATH,
        "dd_version": version,
        "field": DDResolutionField.unit.value,
        "status": status.value,
        "effective_kind": DDResolutionValueKind.string.value,
        "effective_value": json.dumps("m^-3"),
    }


def _rebuild(probe: _RebuildProbe) -> tuple[str | None, str | None]:
    _batch_create_path_nodes(
        probe,
        {
            _PATH: {
                "name": "reconstructed",
                "documentation": "Reconstructed electron density.",
                "data_type": "FLT_0D",
                "node_type": "dynamic",
                "units": "1",
            }
        },
        _VERSION,
    )
    return probe.scalar, probe.edge


def test_rebuild_persists_active_resolution_on_scalar_and_edge() -> None:
    probe = _RebuildProbe([_resolution()])

    assert _rebuild(probe) == ("m^-3", "m^-3")


@pytest.mark.parametrize(
    "resolution",
    [
        _resolution(status=DDResolutionStatus.retired_upstream),
        _resolution(version="4.1.0"),
    ],
    ids=["retired", "different-version"],
)
def test_rebuild_ignores_resolution_without_exact_active_authority(
    resolution: dict[str, str],
) -> None:
    probe = _RebuildProbe([resolution])

    assert _rebuild(probe) == ("1", "1")
