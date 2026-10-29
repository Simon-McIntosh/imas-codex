"""A catalog import binds every published DD path an entry carries.

An imported entry records its DD sources as a union list, and a catalog-typed
entry has no single originating source at all. Binding only the first path
leaves the rest of a published name's sources standing behind a node that never
learns about them, and a container path is not a signal: it binds through its
leaf. A path that cannot be bound is reported on the write result.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from imas_codex.graph.client import GraphClient

_NAME = "electron_temperature"
_SCALAR = "import/bound"
_ARRAY = "import/array"
_STRUCTURE = "import/structure"
_MISSING = "import/missing"
_ORPHAN_ARRAY = "import/orphan_array"


@pytest.fixture()
def planted_graph(disposable_neo4j_uri: str) -> Iterator[GraphClient]:
    """A scalar leaf, a container array, a container structure, no missing path."""
    client = GraphClient(
        uri=disposable_neo4j_uri,
        username="neo4j",
        password=os.environ.get("IMAS_CODEX_TEST_NEO4J_PASSWORD", ""),
        graph_name="catalog-import-fixture",
    )
    client.query(
        """UNWIND [
          {id:$scalar, type:'FLT_1D'},
          {id:$array, type:'STRUCT_ARRAY'},
          {id:$array_values, type:'FLT_1D'},
          {id:$structure, type:'STRUCTURE'},
          {id:$component, type:'FLT_0D'},
          {id:$orphan, type:'STRUCT_ARRAY'}
        ] AS row
        CREATE (:IMASNode {id:row.id, data_type:row.type})""",
        scalar=_SCALAR,
        array=_ARRAY,
        array_values=f"{_ARRAY}/values",
        structure=_STRUCTURE,
        component=f"{_STRUCTURE}/component",
        orphan=_ORPHAN_ARRAY,
    )
    client.query(
        """MATCH (array:IMASNode {id:$array}), (values:IMASNode {id:$values})
        CREATE (array)-[:HAS_CHILD]->(values)
        WITH DISTINCT array
        MATCH (structure:IMASNode {id:$structure}),
              (component:IMASNode {id:$component})
        CREATE (structure)-[:HAS_CHILD]->(component)""",
        array=_ARRAY,
        values=f"{_ARRAY}/values",
        structure=_STRUCTURE,
        component=f"{_STRUCTURE}/component",
    )
    try:
        yield client
    finally:
        client.query(
            "MATCH (node:IMASNode) WHERE node.id STARTS WITH 'import/' "
            "DETACH DELETE node"
        )
        client.close()


def _entry() -> dict:
    return {
        "id": _NAME,
        "source_types": ["catalog"],
        "source_paths": [
            f"dd:{_SCALAR}",
            f"dd:{_ARRAY}",
            f"dd:{_ARRAY}/values",
            f"dd:{_STRUCTURE}",
            f"dd:{_MISSING}",
        ],
        "unit": "eV",
    }


@pytest.mark.graph
def test_import_binds_every_published_leaf_path(planted_graph: GraphClient) -> None:
    from imas_codex.standard_names.graph_store.graph_catalog import (
        write_standard_names,
    )

    result = write_standard_names([_entry()], gc=planted_graph)

    bound = {
        row["path"]
        for row in planted_graph.query(
            """MATCH (path:IMASNode)-[:HAS_STANDARD_NAME]->(
                   name:StandardName {id: $name})
            RETURN path.id AS path""",
            name=_NAME,
        )
    }
    assert bound == {_SCALAR, f"{_ARRAY}/values", f"{_STRUCTURE}/component"}
    # A catalog-typed name takes the published paths it carries.
    assert result == 1
    assert list(result.unbound_paths) == [
        {"name": _NAME, "path": _MISSING, "reason": "no_bindable_leaf"}
    ]


@pytest.mark.graph
def test_container_path_does_not_bind_the_container(
    planted_graph: GraphClient,
) -> None:
    from imas_codex.standard_names.graph_store.graph_catalog import (
        write_standard_names,
    )

    write_standard_names([_entry()], gc=planted_graph)

    containers = planted_graph.query(
        """MATCH (path:IMASNode)-[:HAS_STANDARD_NAME]->(
               name:StandardName {id: $name})
        WHERE path.data_type IN ['STRUCTURE', 'STRUCT_ARRAY']
        RETURN path.id AS path""",
        name=_NAME,
    )
    assert containers == []


@pytest.mark.graph
def test_container_whose_leaf_is_absent_is_reported(
    planted_graph: GraphClient,
) -> None:
    """A container with no leaf node is reported, never quietly left unbound."""
    from imas_codex.standard_names.graph_store.graph_catalog import (
        write_standard_names,
    )

    entry = {
        "id": _NAME,
        "source_types": ["catalog"],
        "source_paths": [f"dd:{_ORPHAN_ARRAY}"],
        "unit": "eV",
    }

    result = write_standard_names([entry], gc=planted_graph)

    assert list(result.unbound_paths) == [
        {"name": _NAME, "path": _ORPHAN_ARRAY, "reason": "leaf_not_found"}
    ]
    assert (
        planted_graph.query(
            """MATCH (path:IMASNode)-[:HAS_STANDARD_NAME]->(
                   name:StandardName {id: $name})
            RETURN path.id AS path""",
            name=_NAME,
        )
        == []
    )
