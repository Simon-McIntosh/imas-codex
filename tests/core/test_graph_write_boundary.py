"""The generic graph tool cannot mutate the governed Standard Name graph."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from imas_codex.llm import server as server_module
from imas_codex.llm.server import AgentsServer

STANDARD_NAME_NODE_TYPES = (
    "StandardName",
    "StandardNameSource",
    "StandardNameChange",
    "StandardNameReview",
    "GrammarSegment",
    "GrammarToken",
    "GrammarTemplate",
    "ISNGrammarVersion",
    "VocabGap",
)


class _ValidatedItem:
    @classmethod
    def model_validate(cls, item: dict[str, object]) -> dict[str, object]:
        return item


class _SchemaView:
    def __init__(self, source_by_label: dict[str, str]) -> None:
        self._source_by_label = source_by_label

    def get_class(self, label: str) -> SimpleNamespace:
        return SimpleNamespace(from_schema=self._source_by_label[label])


class _GraphSchema:
    def __init__(self, source_by_label: dict[str, str]) -> None:
        self.node_labels = list(source_by_label)
        self._view = _SchemaView(source_by_label)
        self.validated_labels: list[str] = []

    def get_private_slots(self, label: str) -> list[str]:
        return []

    def get_model(self, label: str) -> type[_ValidatedItem]:
        self.validated_labels.append(label)
        return _ValidatedItem


def _registered_add_to_graph(server: AgentsServer):
    components = server.mcp._local_provider._components
    return next(
        component.fn
        for key, component in components.items()
        if key.startswith("tool:add_to_graph")
    )


def _install_graph_fakes(
    monkeypatch: pytest.MonkeyPatch,
    source_by_label: dict[str, str],
) -> tuple[_GraphSchema, MagicMock]:
    schema = _GraphSchema(source_by_label)
    graph_client = MagicMock(name="GraphClient")
    client = graph_client.return_value.__enter__.return_value
    client.create_nodes.return_value = {"processed": 1, "relationships": {}}

    monkeypatch.setattr(server_module, "_require_graph_only", lambda: None)
    monkeypatch.setattr(server_module, "get_schema", lambda: schema)
    monkeypatch.setattr(server_module, "GraphClient", graph_client)
    monkeypatch.setattr(server_module, "to_cypher_props", lambda item: item)
    return schema, graph_client


@pytest.mark.parametrize("node_type", STANDARD_NAME_NODE_TYPES)
def test_standard_name_node_type_is_refused_before_graph_write(
    monkeypatch: pytest.MonkeyPatch,
    node_type: str,
) -> None:
    source_name = (
        "grammar_graph" if node_type.startswith(("Grammar", "ISN")) else "standard_name"
    )
    _, graph_client = _install_graph_fakes(
        monkeypatch,
        {node_type: f"https://imas.iter.org/schemas/{source_name}"},
    )
    add_to_graph = _registered_add_to_graph(
        AgentsServer(read_only=False, dd_only=False)
    )

    with pytest.raises(ValueError, match=node_type):
        add_to_graph(node_type, {"id": "governed-node"})

    graph_client.assert_not_called()


def test_future_node_from_standard_name_schema_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node_type = "FutureStandardNameNode"
    _, graph_client = _install_graph_fakes(
        monkeypatch,
        {node_type: "https://imas.iter.org/schemas/standard_name"},
    )
    add_to_graph = _registered_add_to_graph(
        AgentsServer(read_only=False, dd_only=False)
    )

    with pytest.raises(ValueError, match=node_type):
        add_to_graph(node_type, {"id": "future-node"})

    graph_client.assert_not_called()


@pytest.mark.parametrize("node_type", ("SourceFile", "FacilityPath"))
def test_facility_node_type_still_validates_and_writes(
    monkeypatch: pytest.MonkeyPatch,
    node_type: str,
) -> None:
    schema, graph_client = _install_graph_fakes(
        monkeypatch,
        {node_type: "https://imas.iter.org/schemas/facility"},
    )
    add_to_graph = _registered_add_to_graph(
        AgentsServer(read_only=False, dd_only=False)
    )

    result = add_to_graph(node_type, {"id": "facility-node"})

    assert schema.validated_labels == [node_type]
    graph_client.return_value.__enter__.return_value.create_nodes.assert_called_once_with(
        label=node_type,
        items=[{"id": "facility-node"}],
        batch_size=50,
        create_relationships=True,
    )
    assert result == {
        "processed": 1,
        "relationships": {},
        "skipped": 0,
        "errors": [],
    }
