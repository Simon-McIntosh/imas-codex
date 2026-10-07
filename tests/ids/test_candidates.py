"""Tests for deterministic DD candidate retrieval.

Retrieval is exercised against a fake graph and a fake ``hybrid_dd_search`` so
the arm merge, the unscoped path and the batch embedding are measured without a
live Neo4j or a real encoder.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from imas_codex.models.constants import SearchMode
from imas_codex.search.search_strategy import SearchHit


def _hit(path: str, score: float) -> SearchHit:
    return SearchHit(
        path=path,
        ids_name=path.split("/", 1)[0],
        documentation=f"documentation for {path}",
        score=score,
        rank=1,
        search_mode=SearchMode.AUTO,
    )


class _FakeGraph:
    """Graph client that answers only the parent-documentation lookup."""

    def __init__(self, node_ids: set[str]):
        self.node_ids = node_ids
        self.queries: list[tuple[str, dict]] = []

    def query(self, cypher: str, **params):
        self.queries.append((cypher, params))
        if "parent_documentation" in cypher:
            return [
                {"id": pid, "parent_documentation": f"parent of {pid}"}
                for pid in params["paths"]
                if pid in self.node_ids
            ]
        return []


class _FakeEncoder:
    """Encoder replacement recording every batch of texts it is asked for."""

    calls: list[list[str]] = []

    def __init__(self) -> None:
        pass

    def embed_texts(self, texts, *, prompt_name=None, **kwargs):
        type(self).calls.append(list(texts))
        return np.zeros((len(texts), 4))


@pytest.fixture(autouse=True)
def fake_encoder(monkeypatch):
    _FakeEncoder.calls = []
    monkeypatch.setattr("imas_codex.embeddings.encoder.Encoder", _FakeEncoder)
    return _FakeEncoder


def _patch_hybrid(monkeypatch, hits_by_arm: dict, calls: list):
    def fake_hybrid(
        gc,
        query,
        *,
        ids_filter=None,
        dd_version=None,
        k=20,
        embedding=None,
        **kwargs,
    ):
        calls.append({"query": query, "ids_filter": ids_filter, "k": k})
        return list(hits_by_arm.get(ids_filter, []))

    monkeypatch.setattr(
        "imas_codex.ids.candidates.hybrid_dd_search", fake_hybrid, raising=True
    )


def test_every_returned_path_is_a_graph_node_id(monkeypatch):
    from imas_codex.ids.candidates import retrieve_candidates

    node_ids = {
        "equilibrium/time_slice/profiles_1d/psi",
        "equilibrium/time_slice/profiles_1d/psi_error_upper",
    }
    gc = _FakeGraph(node_ids)
    calls: list[dict] = []
    _patch_hybrid(
        monkeypatch,
        {
            "equilibrium": [
                _hit("equilibrium/time_slice/profiles_1d/psi", 0.8),
                _hit("equilibrium/time_slice/profiles_1d/psi_error_upper", 0.4),
            ]
        },
        calls,
    )

    result = retrieve_candidates(
        {"src-1": "plasma poloidal flux"},
        {"src-1": ["equilibrium"]},
        gc=gc,
    )

    candidates = result["src-1"]
    assert candidates
    assert {cand.hit.path for cand in candidates} <= node_ids
    # each path was read back from the fake graph for its parent documentation
    assert candidates[0].parent_documentation == f"parent of {candidates[0].hit.path}"


def test_hits_from_several_ids_merge_by_score_without_duplicating_a_path(monkeypatch):
    from imas_codex.ids.candidates import retrieve_candidates

    shared_path = "equilibrium/time_slice/profiles_1d/psi"
    gc = _FakeGraph({shared_path})
    calls: list[dict] = []
    _patch_hybrid(
        monkeypatch,
        {
            "equilibrium": [
                _hit(shared_path, 0.5),
                _hit("equilibrium/time_slice/global_quantities/ip", 0.7),
            ],
            "core_profiles": [
                _hit(shared_path, 0.7),
                _hit("core_profiles/profiles_1d/electrons/temperature", 0.9),
            ],
        },
        calls,
    )

    result = retrieve_candidates(
        {"src-1": "electron temperature and flux"},
        {"src-1": ["equilibrium", "core_profiles"]},
        gc=gc,
    )

    candidates = result["src-1"]
    paths = [cand.hit.path for cand in candidates]
    assert len(paths) == len(set(paths)), f"duplicate path in merge: {paths}"
    assert paths.count(shared_path) == 1

    merged = next(cand for cand in candidates if cand.hit.path == shared_path)
    assert merged.hit.score == 0.7
    assert merged.arms == frozenset({"equilibrium", "core_profiles"})
    # candidates are ordered by descending score
    assert paths[0] == "core_profiles/profiles_1d/electrons/temperature"


def test_empty_ids_list_runs_unscoped(monkeypatch):
    from imas_codex.ids.candidates import UNSCOPED_ARM, retrieve_candidates

    path = "magnetics/flux_loop/flux"
    gc = _FakeGraph({path})
    calls: list[dict] = []
    _patch_hybrid(monkeypatch, {None: [_hit(path, 0.6)]}, calls)

    result = retrieve_candidates(
        {"src-1": "flux loop signal"},
        {"src-1": []},
        gc=gc,
    )

    assert calls[0]["ids_filter"] is None
    assert result["src-1"][0].arms == frozenset({UNSCOPED_ARM})


def test_one_encoder_call_embeds_a_batch(monkeypatch):
    from imas_codex.ids.candidates import retrieve_candidates

    gc = _FakeGraph({"pf_active/coil/current"})
    calls: list[dict] = []
    _patch_hybrid(
        monkeypatch, {"pf_active": [_hit("pf_active/coil/current", 0.5)]}, calls
    )

    retrieve_candidates(
        {"src-1": "coil current one", "src-2": "coil current two"},
        {"src-1": ["pf_active"], "src-2": ["pf_active"]},
        gc=gc,
    )

    assert len(_FakeEncoder.calls) == 1
    assert _FakeEncoder.calls[0] == ["coil current one", "coil current two"]


def test_gather_ids_context_reads_candidates_from_edges(monkeypatch):
    """The IDS context reads each source's shortlist from its candidate edges.

    Retrieval is no longer called here: the shortlist comes from the stored
    MAPPING_CANDIDATE edges through ``read_candidates``.
    """
    import imas_codex.ids.candidates as candidates_mod
    import imas_codex.ids.mapping as mapping

    sentinel = {"src-1": [{"path": "equilibrium/time_slice/time"}]}
    captured: dict = {}

    def fake_read(source_ids, *args, **kwargs):
        captured["source_ids"] = list(source_ids)
        return sentinel

    def fail_retrieve(*args, **kwargs):
        raise AssertionError("gather_ids_context must not call retrieve_candidates")

    monkeypatch.setattr(mapping, "read_candidates", fake_read)
    monkeypatch.setattr(candidates_mod, "retrieve_candidates", fail_retrieve)
    monkeypatch.setattr(mapping, "fetch_imas_subtree", lambda *a, **k: [])
    monkeypatch.setattr(mapping, "search_imas_semantic", lambda *a, **k: [])
    monkeypatch.setattr(mapping, "search_existing_mappings", lambda *a, **k: {})
    monkeypatch.setattr(mapping, "fetch_cross_facility_mappings", lambda *a, **k: [])

    shared = {
        "dd_version": 4,
        "dd_cocos": 11,
        "groups": [],
        "source_descs": [("src-1", "plasma current")],
        "semantic_match_matrix": {},
        "ids_domains": {"equilibrium": ["equilibrium"]},
        "wiki_context": [],
        "code_context": [],
    }

    ctx = mapping.gather_ids_context("jet", "equilibrium", shared, gc=object())

    assert ctx["source_candidates"] is sentinel
    assert captured["source_ids"] == ["src-1"]


def test_section_prompt_lists_candidate_edge_dicts(monkeypatch):
    """The section prompt reads candidate edge dicts from ``read_candidates``.

    ``gather_ids_context`` stores the edge dicts returned by
    ``read_candidates`` (``path`` / ``retrieval_score`` / ``documentation``),
    so ``_prepare_section_context`` must read them as mappings rather than the
    retrieved ``Candidate`` objects it used to receive.
    """
    import imas_codex.ids.mapping as mapping
    from imas_codex.ids.mapping import _prepare_section_context
    from imas_codex.ids.models import TargetAssignment, TargetType

    monkeypatch.setattr(mapping, "fetch_imas_fields", lambda *a, **k: [])
    monkeypatch.setattr(mapping, "fetch_imas_subtree", lambda *a, **k: [])
    monkeypatch.setattr(mapping, "fetch_source_code_refs", lambda *a, **k: [])

    class _NoVersionTool:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr("imas_codex.tools.version_tool.VersionTool", _NoVersionTool)

    assignment = TargetAssignment(
        source_id="src-1",
        imas_target_path="pf_active/coil",
        target_type=TargetType.STRUCT_ARRAY,
        confidence=0.9,
        reasoning="coil geometry",
    )
    context = {
        "groups": [{"id": "src-1"}],
        "cocos_paths": [],
        "existing": {},
        "dd_cocos": None,
        "source_candidates": {
            "src-1": [
                {
                    "path": "pf_active/coil/r",
                    "retrieval_score": 0.83,
                    "documentation": "Coil R position",
                },
                {
                    "path": "pf_active/coil/z",
                    "retrieval_score": None,
                    "documentation": None,
                },
            ]
        },
        "wiki_context": [],
        "code_context": [],
        "semantic_match_matrix": {},
    }

    result = _prepare_section_context(
        "jet", "pf_active", assignment, context, gc=object()
    )

    prompt = result["prompt"]
    assert "pf_active/coil/r" in prompt
    assert "pf_active/coil/z" in prompt
    assert "0.83" in prompt
    assert "Coil R position" in prompt


def test_mapping_module_no_longer_imports_cluster_searcher():
    import imas_codex.ids.mapping as mapping

    assert "ClusterSearcher" not in Path(mapping.__file__).read_text()
