"""Signal and pulse imports share the archive endpoint persistence owner."""

from unittest.mock import Mock

from iter_codex.discovery.signals.archive_pulses import import_pulses
from iter_codex.discovery.signals.scanners import codac_uda as catalogue
from iter_codex.graph.models import ArchiveEndpoint


def archive():
    return ArchiveEndpoint(
        id="demo:codac_uda:slow",
        facility_id="demo",
        data_system="codac_uda",
        role="slow",
        excluded=False,
    )


def test_pulse_import_delegates_archive_endpoint_write(monkeypatch):
    graph = Mock()
    graph.session.return_value.__enter__ = Mock(return_value=Mock())
    graph.session.return_value.__exit__ = Mock(return_value=False)
    graph.session.return_value.__enter__.return_value.execute_write.return_value = 0
    writer = Mock(return_value=1)
    monkeypatch.setattr(catalogue, "persist_archive_endpoints", writer, raising=False)
    endpoint = archive()
    receipt = import_pulses(graph, endpoint, [], [])
    writer.assert_called_once_with(graph, endpoint.facility_id, [endpoint])
    assert receipt["nodes_created"] == 1
    graph.session.return_value.__enter__.return_value.execute_write.assert_not_called()


def test_signal_scanner_delegates_archive_endpoint_write(monkeypatch):
    graph = Mock()
    graph.query.return_value = [{"written": 1}]
    manager = Mock()
    manager.__enter__ = Mock(return_value=graph)
    manager.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(catalogue, "GraphClient", Mock(return_value=manager))
    writer = Mock(return_value=1)
    monkeypatch.setattr(catalogue, "persist_archive_endpoints", writer, raising=False)
    from iter_codex.discovery.signals import parallel

    monkeypatch.setattr(parallel, "persist_data_access", Mock())
    endpoint = archive()
    catalogue._persist_archives(endpoint.facility_id, [endpoint])
    writer.assert_called_once_with(graph, endpoint.facility_id, [endpoint])
