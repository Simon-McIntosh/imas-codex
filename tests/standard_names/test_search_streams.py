"""Plan 40 Phase 2 — three-stream RRF fusion + mode kwarg tests.

These tests exercise the public ``search_standard_names`` API with a
mock GraphClient. Vector/keyword/grammar streams are stubbed to return
deterministic rows; the fusion logic and tier policy are the units
under test.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from imas_codex.standard_names import search as sn_search


def _stub_gc(rows_by_signature: dict[str, list[dict]] | None = None) -> MagicMock:
    """Build a MagicMock GraphClient whose .query returns rows by Cypher fingerprint."""
    rows_by_signature = rows_by_signature or {}

    def _query(cypher: str, **params: object) -> list[dict]:
        for sig, rows in rows_by_signature.items():
            if sig in cypher:
                return rows
        return []

    gc = MagicMock()
    gc.query.side_effect = _query
    gc.close = MagicMock()
    return gc


# ---------------------------------------------------------------------------
# rrf_fuse — pure unit test
# ---------------------------------------------------------------------------


def test_rrf_fuse_combines_streams() -> None:
    """T3 — RRF fusion sums 1/(k+rank+1) across streams."""
    s1 = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    s2 = [{"id": "b"}, {"id": "a"}]
    fused = sn_search.rrf_fuse([s1, s2], k=60)
    fused_ids = [sn_id for sn_id, _ in fused]
    # Both 'a' and 'b' appear in both streams; 'c' only in s1.
    assert fused_ids[:2] in (["a", "b"], ["b", "a"])
    assert "c" in fused_ids
    assert fused_ids.index("c") > fused_ids.index("a")
    assert fused_ids.index("c") > fused_ids.index("b")


def test_rrf_fuse_empty_streams() -> None:
    assert sn_search.rrf_fuse([]) == []
    assert sn_search.rrf_fuse([[], [], []]) == []


def test_rrf_fuse_handles_string_rows() -> None:
    """Defensive — fuse accepts plain id strings as well as dicts."""
    fused = sn_search.rrf_fuse([[{"id": "a"}], [{"id": "a"}]])
    assert fused[0][0] == "a"


# ---------------------------------------------------------------------------
# vector_stream / keyword_stream / grammar_stream — direct calls
# ---------------------------------------------------------------------------


def test_vector_stream_calls_index() -> None:
    gc = _stub_gc(
        {
            "db.index.vector.queryNodes": [
                {"id": "sn_a", "score": 0.9},
                {"id": "sn_b", "score": 0.7},
            ]
        }
    )
    rows = sn_search.vector_stream([0.0] * 768, gc, k_candidates=5)
    assert rows == [{"id": "sn_a", "score": 0.9}, {"id": "sn_b", "score": 0.7}]


def test_keyword_stream_uses_substring() -> None:
    gc = _stub_gc({"toLower(sn.id) CONTAINS": [{"id": "sn_temp"}]})
    rows = sn_search.keyword_stream("temperature", gc)
    assert rows == [{"id": "sn_temp"}]


def test_keyword_stream_empty_query() -> None:
    gc = _stub_gc()
    assert sn_search.keyword_stream("", gc) == []
    assert sn_search.keyword_stream("   ", gc) == []


def test_grammar_stream_admits_anchored_with_vk() -> None:
    """Grammar stream filters via tier policy and ranks by RRF mass."""
    gc = MagicMock()

    def _query(cypher: str, **params: object) -> list[dict]:
        if "sn.physical_base IN $tokens" in cypher:
            return [{"id": "sn_anchor"}]
        if "sn.subject IN $tokens" in cypher:
            return [{"id": "sn_anchor"}]
        return []

    gc.query.side_effect = _query
    rows = sn_search.grammar_stream(
        ["temperature", "electron"],
        gc,
        vector_hits={"sn_anchor"},
        keyword_hits=set(),
    )
    assert len(rows) == 1
    assert rows[0]["id"] == "sn_anchor"
    assert rows[0]["score"] > 0


def test_grammar_stream_filters_t2_flood() -> None:
    """§9.4 worked example — Tier-2-only floods are dropped at stream level."""
    gc = MagicMock()

    def _query(cypher: str, **params: object) -> list[dict]:
        if "sn.component IN $tokens" in cypher:
            return [{"id": f"decoy_{i}"} for i in range(50)]
        if "sn.physical_base IN $tokens" in cypher:
            return [{"id": "true_match"}]
        return []

    gc.query.side_effect = _query
    rows = sn_search.grammar_stream(
        ["x", "temperature"],
        gc,
        vector_hits={"true_match"},
        keyword_hits=set(),
    )
    ids = {r["id"] for r in rows}
    assert ids == {"true_match"}, f"unexpected admits: {ids - {'true_match'}}"


# ---------------------------------------------------------------------------
# search_standard_names — end-to-end with mocked components
# ---------------------------------------------------------------------------


def test_mode_vector_skips_keyword_and_grammar() -> None:
    """T4 — mode='vector' must not call keyword or grammar streams."""
    gc = _stub_gc(
        {
            "db.index.vector.queryNodes": [{"id": "sn_a", "score": 0.9}],
            "UNWIND $ids": [
                {
                    "name": "sn_a",
                    "description": "x",
                    "documentation": "",
                    "kind": "scalar",
                    "unit": "K",
                    "pipeline_status": "active",
                    "cocos_transformation_type": None,
                    "cocos": None,
                    "physical_base": "temperature",
                    "subject": "electron",
                }
            ],
        }
    )
    with patch.object(sn_search, "_embed", return_value=[0.0] * 8):
        with (
            patch.object(sn_search, "keyword_stream") as ks,
            patch.object(sn_search, "grammar_stream") as gs,
        ):
            results = sn_search.search_standard_names(
                "electron temperature", k=5, mode="vector", gc=gc
            )
            ks.assert_not_called()
            gs.assert_not_called()
    assert results and results[0]["name"] == "sn_a"


def test_search_empty_query_returns_empty() -> None:
    """Empty / whitespace queries short-circuit to []."""
    gc = MagicMock()
    assert sn_search.search_standard_names("", gc=gc) == []
    assert sn_search.search_standard_names("   ", gc=gc) == []
    gc.query.assert_not_called()


def test_search_post_filter_kind() -> None:
    """kind/pipeline_status/cocos_type filter the enriched rows."""
    gc = _stub_gc(
        {
            "db.index.vector.queryNodes": [
                {"id": "sn_a", "score": 0.9},
                {"id": "sn_b", "score": 0.7},
            ],
            "UNWIND $ids": [
                {
                    "name": "sn_a",
                    "description": "",
                    "documentation": "",
                    "kind": "scalar",
                    "unit": "K",
                    "pipeline_status": "active",
                    "cocos_transformation_type": None,
                    "cocos": None,
                    "physical_base": "temperature",
                    "subject": "electron",
                },
                {
                    "name": "sn_b",
                    "description": "",
                    "documentation": "",
                    "kind": "vector",
                    "unit": "T",
                    "pipeline_status": "active",
                    "cocos_transformation_type": None,
                    "cocos": None,
                    "physical_base": "magnetic_flux_density",
                    "subject": "",
                },
            ],
        }
    )
    with patch.object(sn_search, "_embed", return_value=[0.0] * 8):
        out = sn_search.search_standard_names(
            "anything", k=5, mode="vector", kind="vector", gc=gc
        )
    assert [r["name"] for r in out] == ["sn_b"]


def test_search_empty_query_with_filters_uses_catalog_query() -> None:
    """Empty query still works when filters narrow the catalog query path."""
    gc = _stub_gc(
        {
            "MATCH (sn:StandardName)": [
                {
                    "name": "electron_temperature",
                    "description": "x",
                    "documentation": "",
                    "kind": "scalar",
                    "unit": "K",
                    "pipeline_status": "active",
                    "cocos_transformation_type": None,
                    "cocos": None,
                    "physics_domain": "transport",
                    "source_domains": ["transport"],
                    "physical_base": "temperature",
                    "subject": "electron",
                    "score": 1.0,
                }
            ]
        }
    )
    results = sn_search.search_standard_names(
        "",
        k=5,
        segment_filters={"physical_base": "temperature"},
        physics_domain="transport",
        gc=gc,
    )
    assert results and results[0]["name"] == "electron_temperature"


def test_fetch_return_fields_override_include_flags() -> None:
    """return_fields takes precedence over include_documentation/include_grammar flags."""
    gc = _stub_gc(
        {
            "UNWIND $names": [
                {
                    "name": "electron_temperature",
                    "documentation": "doc",
                    "physical_base": "temperature",
                }
            ]
        }
    )
    rows = sn_search.fetch_standard_names(
        ["electron_temperature"],
        include_documentation=False,
        include_grammar=False,
        return_fields=["name", "documentation", "physical_base"],
        gc=gc,
    )
    assert rows == [
        {
            "name": "electron_temperature",
            "documentation": "doc",
            "physical_base": "temperature",
        }
    ]


def test_fetch_return_fields_neighbours_only() -> None:
    """Neighbour-only projections still fetch neighbours via the grouping key."""
    gc = MagicMock()

    def _query(cypher: str, **params: object) -> list[dict]:
        if "MATCH (sn:StandardName {id: name_id})" in cypher:
            return [{"name": "electron_temperature"}]
        if "MATCH (sn:StandardName {id: sn_id})" in cypher:
            return [
                {
                    "name": "electron_temperature",
                    "predecessors": ["legacy_temperature"],
                    "successors": [],
                    "refined_from": [],
                }
            ]
        return []

    gc.query.side_effect = _query
    gc.close = MagicMock()

    rows = sn_search.fetch_standard_names(
        ["electron_temperature"],
        return_fields=["name", "neighbours"],
        gc=gc,
    )

    assert rows == [
        {
            "name": "electron_temperature",
            "neighbours": {
                "predecessors": ["legacy_temperature"],
                "successors": [],
                "refined_from": [],
            },
        }
    ]


def test_fetch_aggregate_only_return_fields_preserve_per_name_rows() -> None:
    """Aggregate-only projections still group by name and return one row per input."""
    gc = MagicMock()
    gc.query.return_value = [
        {"name": "electron_temperature", "source_ids": ["cp/electron_temperature"]},
        {"name": "ion_temperature", "source_ids": ["cp/ion_temperature"]},
    ]
    gc.close = MagicMock()

    rows = sn_search.fetch_standard_names(
        ["electron_temperature", "ion_temperature"],
        return_fields=["source_ids"],
        gc=gc,
    )

    assert rows == [
        {"source_ids": ["cp/electron_temperature"]},
        {"source_ids": ["cp/ion_temperature"]},
    ]
    cypher = gc.query.call_args[0][0]
    assert "sn.id AS name" in cypher
