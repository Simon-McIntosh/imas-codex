"""Code references resolve to existing DD paths and IDS roots."""

import pytest

from imas_codex.ingestion.extractors.ids import extract_imas_path_references
from imas_codex.ingestion.graph import (
    link_chunks_to_ids_roots,
    link_chunks_to_imas_paths,
)
from imas_codex.ingestion.pipeline import _split_and_extract


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "equilibrium.time_slice[0].profiles_2d[0].psi",
            "equilibrium/time_slice/profiles_2d/psi",
        ),
        (
            'ids_factory.new("equilibrium"); ids%time_slice(1)%profiles_2d(1)%psi',
            "equilibrium/time_slice/profiles_2d/psi",
        ),
        (
            'path = "equilibrium/time_slice/profiles_2d/psi"',
            "equilibrium/time_slice/profiles_2d/psi",
        ),
    ],
)
def test_extracts_dd_path_notation(source, expected):
    assert expected in extract_imas_path_references(source)


def test_generic_fortran_variable_needs_one_ids_name():
    source = 'new("equilibrium"); new("core_profiles"); ids%time_slice(1)%psi'
    assert "equilibrium/time_slice/psi" not in extract_imas_path_references(source)


class _ReferenceGraph:
    def __init__(self, *, paths=(), names=()):
        self.chunk = {
            "imas_paths": list(paths),
            "related_ids": list(names),
        }
        self.targets = {
            "IMASNode": {"equilibrium/time_slice/profiles_2d/psi"},
            "IDS": {"equilibrium"},
        }
        self.links = set()

    def query(self, cypher, **params):
        if "AS mentions" in cypher:
            prop = "imas_paths" if "c.imas_paths" in cypher else "related_ids"
            return [{"mentions": len(self.chunk[prop])}]
        if "AS linked" in cypher:
            prop, label, relation = (
                ("imas_paths", "IMASNode", "REFERENCES_IMAS")
                if "c.imas_paths" in cypher
                else ("related_ids", "IDS", "REFERENCES_IDS")
            )
            if f"MATCH (target:{label} {{id: reference}})" not in cypher:
                return [{"linked": 0}]
            for value in self.chunk[prop]:
                if value in self.targets[label] and f":{relation}" in cypher:
                    self.links.add((relation, label, value))
            return [{"linked": len(self.links)}]
        return []


def test_dd_path_links_only_to_existing_imas_node():
    graph = _ReferenceGraph(
        paths=(
            "equilibrium/time_slice/profiles_2d/psi",
            "equilibrium/time_slice/guessed_path",
        )
    )
    assert link_chunks_to_imas_paths(graph, ["example"]) == 1
    assert graph.links == {
        (
            "REFERENCES_IMAS",
            "IMASNode",
            "equilibrium/time_slice/profiles_2d/psi",
        )
    }


def test_bare_ids_name_links_to_ids_root():
    graph = _ReferenceGraph(names=("equilibrium",))
    assert link_chunks_to_ids_roots(graph, ["example"]) == 1
    assert graph.links == {("REFERENCES_IDS", "IDS", "equilibrium")}


def test_named_ids_without_a_root_raises():
    graph = _ReferenceGraph(names=("missing",))
    with pytest.raises(ValueError, match="No IDS roots linked"):
        link_chunks_to_ids_roots(graph, ["example"])


def test_one_valid_ids_name_does_not_hide_a_missing_name():
    graph = _ReferenceGraph(names=("equilibrium", "missing"))
    with pytest.raises(ValueError, match="1 of 2 named references"):
        link_chunks_to_ids_roots(graph, ["example"])


def test_ingestion_keeps_both_reference_kinds(monkeypatch):
    from types import SimpleNamespace

    text = 'new("equilibrium"); equilibrium.time_slice[0].profiles_2d[0].psi'
    chunk = SimpleNamespace(text=text, start_line=1, end_line=1)
    monkeypatch.setattr(
        "imas_codex.ingestion.pipeline.chunk_code", lambda *a, **k: [chunk]
    )
    result = _split_and_extract(text, "python", {"facility_id": "jt-60sa"})
    assert result[0]["related_ids"] == ["equilibrium"]
    assert result[0]["imas_paths"] == ["equilibrium/time_slice/profiles_2d/psi"]
