"""Environment-loading invariants for the pytest process."""

from unittest.mock import Mock

import conftest

from imas_codex.graph import client as graph_client_module


def _collected_item(*marker_names):
    item = Mock()
    item.get_closest_marker.side_effect = lambda marker_name: (
        marker_name if marker_name in marker_names else None
    )
    return item


def test_dotenv_only_fills_unset_process_values(tmp_path, monkeypatch):
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "NEO4J_PASSWORD=dotenv-password\nNEO4J_USERNAME=dotenv-user\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("NEO4J_PASSWORD", "process-password")
    monkeypatch.delenv("NEO4J_USERNAME", raising=False)

    conftest._load_test_environment(dotenv_path)

    assert conftest.os.environ["NEO4J_PASSWORD"] == "process-password"
    assert conftest.os.environ["NEO4J_USERNAME"] == "dotenv-user"


def test_collection_avoids_graph_probe_without_graph_items(monkeypatch):
    probe = Mock()
    monkeypatch.setattr(conftest, "_check_neo4j", probe)

    conftest.pytest_collection_modifyitems(None, [_collected_item()])

    probe.assert_not_called()


def test_collection_probes_once_and_skips_only_graph_items(monkeypatch):
    probe = Mock(return_value=False)
    monkeypatch.setattr(conftest, "_check_neo4j", probe)
    unit_item = _collected_item()
    graph_item = _collected_item("graph")

    conftest.pytest_collection_modifyitems(None, [unit_item, graph_item])

    probe.assert_called_once_with()
    unit_item.add_marker.assert_not_called()
    graph_item.add_marker.assert_called_once()


def test_explicit_test_graph_probe_bypasses_project_resolution(monkeypatch):
    test_uri = "bolt://127.0.0.1:17687"
    monkeypatch.setenv("IMAS_CODEX_TEST_NEO4J_URI", test_uri)
    monkeypatch.setenv("NEO4J_USERNAME", "test-user")
    monkeypatch.setenv("NEO4J_PASSWORD", "test-password")
    resolve_profile = Mock(side_effect=AssertionError("project profile resolved"))
    monkeypatch.setattr("imas_codex.graph.profiles.resolve_neo4j", resolve_profile)
    resolve_name = Mock(side_effect=AssertionError("project graph name resolved"))
    monkeypatch.setattr("imas_codex.graph.profiles.get_active_graph_name", resolve_name)
    driver = Mock()
    driver.close = Mock()
    create_driver = Mock(return_value=driver)
    monkeypatch.setattr(graph_client_module.GraphDatabase, "driver", create_driver)
    monkeypatch.setattr(graph_client_module, "get_schema", Mock(return_value=Mock()))

    client = conftest._neo4j_probe_client()
    client.close()

    create_driver.assert_called_once_with(
        test_uri,
        auth=("test-user", "test-password"),
        **graph_client_module.GraphClient._driver_kwargs(test_uri),
    )
    assert client.graph_name == "pytest-explicit-endpoint"
    resolve_profile.assert_not_called()
    resolve_name.assert_not_called()


def test_explicit_test_graph_collection_probes_once(monkeypatch):
    monkeypatch.setenv("IMAS_CODEX_TEST_NEO4J_URI", "bolt://127.0.0.1:17687")
    monkeypatch.setattr(conftest, "_neo4j_available", None)
    client = Mock()
    monkeypatch.setattr(conftest, "_neo4j_probe_client", Mock(return_value=client))
    graph_item = _collected_item("graph")
    integration_item = _collected_item("integration")

    conftest.pytest_collection_modifyitems(None, [graph_item, integration_item])

    conftest._neo4j_probe_client.assert_called_once_with()
    client.get_stats.assert_called_once_with()
    client.close.assert_called_once_with()
    graph_item.add_marker.assert_not_called()
    integration_item.add_marker.assert_not_called()


def test_unavailable_explicit_endpoint_skips_only_graph_items(monkeypatch):
    test_uri = "bolt://127.0.0.1:17687"
    monkeypatch.setenv("IMAS_CODEX_TEST_NEO4J_URI", test_uri)
    monkeypatch.setattr(conftest, "_neo4j_available", None)
    client = Mock()
    client.get_stats.side_effect = OSError("connection refused")
    monkeypatch.setattr(conftest, "_neo4j_probe_client", Mock(return_value=client))
    unit_item = _collected_item()
    graph_item = _collected_item("requires_graph")

    conftest.pytest_collection_modifyitems(None, [unit_item, graph_item])

    unit_item.add_marker.assert_not_called()
    graph_item.add_marker.assert_called_once()
    marker = graph_item.add_marker.call_args.args[0]
    assert marker.kwargs["reason"] == (
        f"Explicit Neo4j test endpoint is not available: {test_uri}"
    )
    client.close.assert_called_once_with()


def test_graph_probe_without_test_uri_uses_project_client(monkeypatch):
    monkeypatch.delenv("IMAS_CODEX_TEST_NEO4J_URI", raising=False)
    client_type = Mock()
    monkeypatch.setattr(graph_client_module, "GraphClient", client_type)

    conftest._neo4j_probe_client()

    client_type.assert_called_once_with()
