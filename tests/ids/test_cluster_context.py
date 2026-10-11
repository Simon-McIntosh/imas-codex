"""Cross-IDS cluster siblings reach the mapping prompt."""

from imas_codex.ids.candidates import CLUSTER_ARM
from imas_codex.ids.mapping import _cluster_context


def test_cluster_siblings_are_listed_apart_from_retrieval_hits():
    candidates = [
        {
            "path": "magnetics/ip/data",
            "ids": "magnetics",
            "arms": ["semantic"],
            "documentation": "Plasma current",
        },
        {
            "path": "equilibrium/time_slice/global_quantities/ip",
            "ids": "equilibrium",
            "arms": [CLUSTER_ARM],
            "documentation": "Plasma current",
        },
    ]
    text = _cluster_context(candidates)
    assert "equilibrium/time_slice/global_quantities/ip" in text
    assert "magnetics/ip/data" not in text


def test_no_cluster_sibling_renders_nothing():
    assert _cluster_context([{"path": "magnetics/ip/data", "arms": ["semantic"]}]) == ""
