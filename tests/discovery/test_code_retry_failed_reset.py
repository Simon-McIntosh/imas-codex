"""A failed code fetch can be retried without reprocessing ingested files."""

from imas_codex.discovery.base.reset import (
    CODE_RESET_SPECS,
    reset_to_status,
)


def test_retry_failed_resets_only_failed_files(monkeypatch):
    files = [
        {"status": "failed", "error": "fetch failed", "path": "/home/a/code.py"},
        {"status": "ingested", "error": None, "path": "/home/a/done.py"},
    ]

    class Graph:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def query(self, cypher, **params):
            assert "n.status IN $source_statuses" in cypher
            assert "n.error = null" in cypher
            assert params["path_prefixes"] == ["/home/"]
            selected = [
                file for file in files if file["status"] in params["source_statuses"]
            ]
            for file in selected:
                file["status"] = params["target_status"]
                file["error"] = None
            return [{"reset_count": len(selected)}]

    monkeypatch.setattr("imas_codex.graph.GraphClient", Graph)
    assert (
        reset_to_status(
            CODE_RESET_SPECS["retry-failed"], "jt-60sa", path_prefixes=["/home/"]
        )
        == 1
    )
    assert files == [
        {"status": "scored", "error": None, "path": "/home/a/code.py"},
        {"status": "ingested", "error": None, "path": "/home/a/done.py"},
    ]
