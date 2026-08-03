"""Environment-loading invariants for the pytest process."""

from unittest.mock import Mock

import conftest


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
