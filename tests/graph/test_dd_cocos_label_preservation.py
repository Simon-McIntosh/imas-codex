"""A DD build must not clear a COCOS label it did not re-derive.

DD builds are run sharded by IDS (``imas dd build --ids <ids>``) and
incrementally by version (``--from-version``), so one invocation extracts only
part of the dictionary. The stale-label clear in ``phase_build`` nulls
``cocos_transformation_type`` on every node the current build did not label,
which — run unscoped — strips the labels of every other shard.

These tests run the real build path on the disposable graph: a label survives
unless the build covered the path and a version that can declare labels shows
the DD no longer does.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.graph

V3 = "3.42.2"
V4 = "4.1.1"


def _path(name: str, label: str | None = None) -> dict:
    return {
        "name": name,
        "documentation": "",
        "units": "",
        "data_type": "FLT_0D",
        "ndim": 0,
        "node_type": "quantity",
        "maxoccur": 1,
        "parent_path": None,
        "cocos_transformation_type": label,
    }


def _version(paths: dict[str, dict]) -> dict:
    return {"paths": paths, "ids_info": {}, "units": set()}


def _run_build(gc, version_data: dict[str, dict]) -> None:
    """Run the build phase over exactly the given scope."""
    from imas_codex.graph.build_dd import phase_build

    phase_build(
        gc,
        sorted(version_data.keys()),
        version_data,
        set(),
        dry_run=False,
    )


def _labels(gc) -> dict[str, dict]:
    rows = gc.query(
        """
        MATCH (p:IMASNode)
        WHERE p.cocos_transformation_type IS NOT NULL
        RETURN p.id AS id, p.cocos_transformation_type AS label,
               p.cocos_label_source AS source
        """
    )
    return {r["id"]: {"label": r["label"], "source": r["source"]} for r in rows}


def test_full_sharded_build_keeps_every_shards_labels(disposable_graph):
    """Each IDS shard of a full build keeps the labels the others derived."""
    from imas_codex.graph.client import GraphClient

    psi = "equilibrium/time_slice/profiles_1d/psi"
    temp = "core_profiles/profiles_1d/electrons/temperature"
    eq = {
        V3: _version({psi: _path("psi", "psi_like")}),
        V4: _version({psi: _path("psi")}),
    }
    cp = {
        V3: _version({temp: _path("temperature", "psi_like")}),
        V4: _version({temp: _path("temperature")}),
    }

    with GraphClient() as gc:
        _run_build(gc, eq)  # shard 1
        _run_build(gc, cp)  # shard 2
        labels = _labels(gc)

    assert labels.get(psi), "the equilibrium shard's forward-ported label was cleared"
    assert labels.get(temp), (
        "the core_profiles shard's forward-ported label was missing"
    )
    assert labels[psi]["label"] == "psi_like"
    assert labels[psi]["source"] == "inferred_forward"


def test_scoped_build_leaves_out_of_scope_labels_intact(disposable_graph):
    """A build over one IDS must not touch a label outside that IDS."""
    from imas_codex.graph.client import GraphClient

    outside = "equilibrium/time_slice/profiles_1d/psi"
    inside = "core_profiles/profiles_1d/electrons/temperature"
    scope = {
        V3: _version({inside: _path("temperature", "psi_like")}),
        V4: _version({inside: _path("temperature")}),
    }

    with GraphClient() as gc:
        gc.query(
            "MERGE (p:IMASNode {id: $id}) "
            "SET p.cocos_transformation_type = 'psi_like', "
            "p.cocos_label_source = 'xml'",
            id=outside,
        )
        _run_build(gc, scope)
        labels = _labels(gc)

    assert labels.get(outside, {}).get("label") == "psi_like", (
        "a scoped build cleared a label outside its scope"
    )
    assert labels.get(inside, {}).get("label") == "psi_like"


def test_full_rebuild_is_idempotent_on_labels(disposable_graph):
    """Rebuilding twice yields the same labels, including 3.x-only paths."""
    from imas_codex.graph.client import GraphClient

    psi = "equilibrium/time_slice/profiles_1d/psi"
    only_3x = "equilibrium/time_slice/profiles_1d/psi_removed_in_4x"
    version_data = {
        V3: _version(
            {
                psi: _path("psi", "psi_like"),
                only_3x: _path("psi_removed_in_4x", "psi_like"),
            }
        ),
        V4: _version({psi: _path("psi")}),
    }

    with GraphClient() as gc:
        _run_build(gc, version_data)
        first = _labels(gc)
        _run_build(gc, version_data)
        second = _labels(gc)

    assert first.get(psi, {}).get("label") == "psi_like"
    assert first.get(only_3x, {}).get("label") == "psi_like", (
        "a 3.x-declared label was cleared although a version in the build declares it"
    )
    assert first == second, f"labels changed on rebuild: {first} -> {second}"
