"""The IDS domain refresh applies the map and never disturbs a classification.

``refresh_physics_domains`` moves a changed IDS node's domain and, on that
IDS's paths, writes the IDS default only where the path carries no
classification. Both facts are asserted against a model of the graph state
built from the recorded statement, so a statement that drops the
``domain_source IS NULL`` guard overwrites a classified path and reddens.
"""

from __future__ import annotations

import pytest

from imas_codex.core.physics_categorization import physics_categorizer
from imas_codex.graph.build_dd import refresh_physics_domains
from tests.recording_graph_client import RecordingGraphClient

# The twenty-five IDS nodes the reviewed map moves, keyed to their target.
RULED_IDS_DOMAINS = {
    "amns_data": "computational_workflow",
    "calorimetry": "mechanical_measurement_diagnostics",
    "charge_exchange": "spectroscopy",
    "core_instant_changes": "core_plasma_physics",
    "core_profiles": "core_plasma_physics",
    "distribution_sources": "fast_particles",
    "distributions": "fast_particles",
    "ece": "radiation_measurement_diagnostics",
    "gas_injection": "fueling",
    "gyrokinetics": "gyrokinetics",
    "gyrokinetics_local": "gyrokinetics",
    "mse": "spectroscopy",
    "neutron_diagnostic": "neutronics",
    "pellets": "fueling",
    "plasma_initiation": "plasma_initiation",
    "plasma_profiles": "core_plasma_physics",
    "radiation": "core_plasma_physics",
    "runaway_electrons": "runaway_electrons",
    "spectrometer_mass": "spectroscopy",
    "spectrometer_uv": "spectroscopy",
    "spectrometer_visible": "spectroscopy",
    "spectrometer_x_ray_crystal": "spectroscopy",
    "spi": "fueling",
    "thomson_scattering": "particle_measurement_diagnostics",
    "waves": "waves",
}


class _DomainRefreshingClient(RecordingGraphClient):
    """Model the IDS and path nodes the refresh reads and writes.

    The recorded statement decides whether the path write is guarded: a
    statement that drops ``domain_source IS NULL`` overwrites a classified
    path, so the assertions on the model fail.
    """

    def __init__(self) -> None:
        super().__init__()
        self.ids = {
            "core_profiles": {"physics_domain": "transport"},
            "equilibrium": {"physics_domain": "equilibrium"},
        }
        self.paths = {
            "core_profiles/classified": {
                "ids": "core_profiles",
                "physics_domain": "general",
                "domain_source": "llm_classified",
            },
            "core_profiles/unclassified": {
                "ids": "core_profiles",
                "physics_domain": "transport",
                "domain_source": None,
            },
            "core_profiles/already_mapped": {
                "ids": "core_profiles",
                "physics_domain": "core_plasma_physics",
                "domain_source": None,
            },
        }
        self.ids_upsert = ""
        self.path_upsert = ""

    def query(self, cypher: str, **params: object) -> list[dict]:
        self.calls.append((cypher, params))
        if "MATCH (i:IDS)" in cypher and "RETURN i.id AS id" in cypher:
            return [
                {"id": name, "domain": node["physics_domain"]}
                for name, node in self.ids.items()
            ]
        if "SET i.physics_domain = row.after" in cypher:
            self.ids_upsert = cypher
            rows = params["rows"]
            assert isinstance(rows, list)
            for row in rows:
                self.ids[row["id"]]["physics_domain"] = row["after"]
            return []
        if "MATCH (p:IMASNode {ids: $ids})" in cypher:
            guard = "domain_source IS NULL" in cypher
            ids_name = params["ids"]
            domain = params["domain"]
            matched = [
                pid
                for pid, node in self.paths.items()
                if node["ids"] == ids_name
                and (not guard or node["domain_source"] is None)
                and node["physics_domain"] != domain
            ]
            if "RETURN p.id AS id" in cypher:
                return [
                    {"id": pid, "before": self.paths[pid]["physics_domain"]}
                    for pid in matched
                ]
            self.path_upsert = cypher
            for pid in matched:
                self.paths[pid]["physics_domain"] = domain
            return [{"changed": len(matched)}]
        return []


def test_refresh_writes_only_the_domain_and_spares_classified_paths() -> None:
    client = _DomainRefreshingClient()
    classified_before = dict(client.paths["core_profiles/classified"])

    result = refresh_physics_domains(client)

    # The IDS node whose stored domain differs is moved; the one already at its
    # mapped domain is not written, and the IDS statement sets one property.
    assert client.ids_upsert
    assert "i.physics_domain = row.after" in client.ids_upsert
    assert client.ids_upsert.count(" SET ") == 1
    assert client.ids["core_profiles"]["physics_domain"] == "core_plasma_physics"
    assert {row["id"] for row in result["ids_changed"]} == {"core_profiles"}

    # The path statement carries the build's guard and writes one property.
    assert client.path_upsert
    assert "domain_source IS NULL" in client.path_upsert
    assert client.path_upsert.count(" SET ") == 1
    assert client.path_upsert.count("p.physics_domain =") == 1

    # A classified path keeps its own domain and provenance; an unclassified
    # path takes the IDS default; a path already at the default is untouched.
    assert client.paths["core_profiles/classified"] == classified_before
    assert (
        client.paths["core_profiles/unclassified"]["physics_domain"]
        == "core_plasma_physics"
    )
    assert (
        client.paths["core_profiles/already_mapped"]["physics_domain"]
        == "core_plasma_physics"
    )

    assert result["path_changed"] == [
        {
            "id": "core_profiles/unclassified",
            "ids": "core_profiles",
            "before": "transport",
            "after": "core_plasma_physics",
        }
    ]
    assert result["path_changed_count"] == 1
    assert result["applied_path_count"] == 1


def test_dry_run_reports_changes_without_writing() -> None:
    client = _DomainRefreshingClient()
    ids_before = {name: dict(node) for name, node in client.ids.items()}
    paths_before = {pid: dict(node) for pid, node in client.paths.items()}

    result = refresh_physics_domains(client, dry_run=True)

    assert result["dry_run"] is True
    assert len(result["ids_changed"]) == 1
    assert result["path_changed_count"] == 1
    assert client.ids == ids_before
    assert client.paths == paths_before
    assert client.ids_upsert == ""
    assert client.path_upsert == ""


@pytest.mark.graph
def test_dry_run_reads_the_ruled_ids_nodes_from_the_graph() -> None:
    """The dry run reads every IDS node and the twenty-five carry their target.

    The graph resolves by discovery from a compute step, so this runs in
    place. The map is applied in the same node, so each ruled IDS node must
    already carry its mapped domain and the dry run reports no change for it.
    """
    from imas_codex.graph import GraphClient

    with GraphClient() as gc:
        result = refresh_physics_domains(gc, dry_run=True)
        stored = {
            row["id"]: row["domain"]
            for row in gc.query(
                "MATCH (i:IDS) RETURN i.id AS id, i.physics_domain AS domain"
            )
        }

    assert result["ids_total"] == len(stored)
    assert result["ids_total"] >= len(RULED_IDS_DOMAINS)
    changed = {row["id"]: row["after"] for row in result["ids_changed"]}

    for ids_name, target in RULED_IDS_DOMAINS.items():
        assert ids_name in stored, f"{ids_name} is not an IDS node in the graph"
        assert physics_categorizer.get_domain_for_ids(ids_name).value == target
        assert stored[ids_name] == target, (ids_name, stored[ids_name], target)
        assert changed.get(ids_name) is None, (ids_name, changed)
