"""Code progress queries and refresh error visibility."""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

from imas_codex.cli.discover import common
from imas_codex.discovery.code.parallel import get_code_discovery_stats


def test_code_progress_queries_have_single_cypher_braces():
    queries = []
    graph = MagicMock()

    def capture(query, **_params):
        queries.append(query)
        return []

    graph.query.side_effect = capture
    context = MagicMock()
    context.__enter__.return_value = graph

    with patch("imas_codex.discovery.code.parallel.GraphClient", return_value=context):
        get_code_discovery_stats(
            "jt-60sa",
            min_relevance=0.5,
            min_ingest_relevance=0.5,
            min_facet_relevance=0.5,
        )

    assert len(queries) == 8
    for query in queries:
        assert "{{" not in query
        assert "}}" not in query


def test_graph_refresh_logs_each_distinct_error_and_keeps_running():
    records = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    capture = Capture()
    common.logger.addHandler(capture)
    reached = asyncio.Event()
    calls = 0

    async def refresh():
        nonlocal calls
        calls += 1
        if calls >= 3:
            reached.set()
            raise RuntimeError("graph unavailable")
        raise ValueError("bad Cypher")

    async def main(_stop_event, _monitor):
        await asyncio.wait_for(reached.wait(), timeout=1)
        return {"elapsed_seconds": 0}

    monitor = MagicMock()
    monitor.__aenter__ = AsyncMock(return_value=monitor)
    monitor.__aexit__ = AsyncMock(return_value=None)
    config = common.DiscoveryConfig(
        domain="code",
        facility="jt-60sa",
        facility_config={},
        model_section="discovery-relevance",
        display=MagicMock(),
        graph_refresh_interval=0,
        graph_refresh_fn=refresh,
    )
    try:
        with (
            patch.object(common, "create_discovery_monitor", return_value=monitor),
            patch("imas_codex.cli.shutdown.safe_asyncio_run", side_effect=asyncio.run),
            patch("imas_codex.cli.shutdown.install_shutdown_handlers"),
        ):
            common.run_discovery(config, main, on_complete=lambda _result: None)
    finally:
        common.logger.removeHandler(capture)

    warnings = [record.getMessage() for record in records if record.levelno == 30]
    assert warnings == [
        "Graph progress refresh failed: bad Cypher",
        "Graph progress refresh failed: graph unavailable",
    ]
    assert calls >= 3
