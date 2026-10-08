"""The parallel wiki runner admits pages at the calibrated judgment cutoff."""

from inspect import signature

from imas_codex.discovery.wiki.graph_ops import CONTENT_INGEST_THRESHOLD
from imas_codex.discovery.wiki.parallel import run_parallel_wiki_discovery


def test_parallel_runner_uses_judgment_cutoff_by_default():
    threshold = signature(run_parallel_wiki_discovery).parameters["min_score"].default
    assert threshold == CONTENT_INGEST_THRESHOLD
