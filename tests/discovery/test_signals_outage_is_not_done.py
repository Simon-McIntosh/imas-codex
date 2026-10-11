"""An unreachable graph never reads as an empty signals queue.

The pending-work predicates and the enrichment claim decide whether the
signals phases end. Read as "no work", a refused connection ends the run and
reports it completed while signals remain; raised, it reaches the worker
supervisor, which retries infrastructure errors with backoff.
"""

from unittest.mock import patch

import pytest
from neo4j.exceptions import ServiceUnavailable

from imas_codex.discovery.signals import parallel


class _DownGraph:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def query(self, *args, **kwargs):
        raise ServiceUnavailable("Couldn't connect to 98dci4-gpu-0002:7687")


@pytest.mark.parametrize(
    "predicate", [parallel.has_pending_enrich_work, parallel.has_pending_check_work]
)
def test_pending_work_raises_when_the_graph_is_down(predicate):
    with patch.object(parallel, "GraphClient", _DownGraph):
        with pytest.raises(ServiceUnavailable):
            predicate("jt-60sa")


def test_enrichment_claim_raises_when_the_graph_is_down():
    with patch.object(parallel, "GraphClient", _DownGraph):
        with pytest.raises(ServiceUnavailable):
            parallel.claim_signals_for_enrichment("jt-60sa")
