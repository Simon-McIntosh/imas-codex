"""A DD path rebuild preserves domains assigned by classification."""

import re

from imas_codex.graph.build_dd import _batch_create_path_nodes


class _RecordingClient:
    """Record path upserts and apply their domain assignment to fixture nodes."""

    def __init__(self) -> None:
        self.nodes = {
            "core_profiles/classified": {
                "physics_domain": "core_plasma_physics",
                "domain_source": "llm",
                "domain_model": "classifier",
                "domain_classified_at": "2026-06-18T00:00:00Z",
            },
            "core_profiles/unclassified": {
                "physics_domain": "general",
                "domain_source": None,
            },
        }
        self.upsert = ""

    def query(self, cypher: str, **parameters: object) -> list[dict]:
        if "MERGE (path:IMASNode {id: p.id})" not in cypher:
            return []
        self.upsert = cypher
        match = re.search(
            r"path\.physics_domain\s*=\s*(.*?),\s*path\.maxoccur", cypher, re.S
        )
        assert match, "path upsert must assign the default domain conditionally"
        assignment = " ".join(match.group(1).split())
        paths = parameters["paths"]
        assert isinstance(paths, list)
        for path in paths:
            node = self.nodes[path["id"]]
            if assignment == "p.physics_domain":
                node["physics_domain"] = path["physics_domain"]
            elif assignment == (
                "CASE WHEN path.domain_source IS NULL THEN p.physics_domain "
                "ELSE path.physics_domain END"
            ):
                if node["domain_source"] is None:
                    node["physics_domain"] = path["physics_domain"]
            else:
                raise AssertionError(f"Unrecognized domain assignment: {assignment}")
        return []


def test_path_upsert_keeps_classified_domain_and_defaults_unclassified_path() -> None:
    client = _RecordingClient()
    classified_before = client.nodes["core_profiles/classified"].copy()

    _batch_create_path_nodes(
        client,
        {
            "core_profiles/classified": {"name": "classified", "node_type": "dynamic"},
            "core_profiles/unclassified": {
                "name": "unclassified",
                "node_type": "dynamic",
            },
        },
        "4.1.1",
    )

    assert client.upsert
    assert client.nodes["core_profiles/classified"] == classified_before
    assert client.nodes["core_profiles/unclassified"]["physics_domain"] == "transport"
    for property_name in ("domain_source", "domain_model", "domain_classified_at"):
        assert f"path.{property_name} =" not in client.upsert
