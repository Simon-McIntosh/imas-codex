"""The DD's own rename annotation outranks derived and packaged renames.

A rename record is assembled from three sources. The packaged path map is a
snapshot of an earlier mapping run, so it can name a target that the current
DD has since moved past. The container-rename detector infers a match from
subtree shape. The current DD's ``change_nbc_previous_name`` annotation is the
authority, and a lower-precedence source must not shadow it: a stale packaged
target written into the ground would point a removed path's successor at the
wrong node.
"""

from __future__ import annotations

from imas_codex.graph.build_dd import collect_renamed_to_mappings


def test_stale_packaged_entry_does_not_shadow_the_dd_annotation():
    mappings = collect_renamed_to_mappings(
        version_renames={},
        path_mappings={
            "old_to_new": {"summary/total_energy": {"new_path": "summary/energy_old"}}
        },
        dd_renames={"summary/total_energy": "summary/energy"},
    )

    assert mappings["summary/total_energy"]["new_path"] == "summary/energy"


def test_dd_annotation_outranks_a_container_rename():
    mappings = collect_renamed_to_mappings(
        version_renames={"old_ids/field": "inferred_ids/field"},
        path_mappings={"old_to_new": {}},
        dd_renames={"old_ids/field": "annotated_ids/field"},
    )

    assert mappings["old_ids/field"]["new_path"] == "annotated_ids/field"


def test_container_rename_outranks_the_packaged_path_map():
    mappings = collect_renamed_to_mappings(
        version_renames={"ec_launchers/launcher/r": "ec_launchers/beam/r"},
        path_mappings={
            "old_to_new": {
                "ec_launchers/launcher/r": {"new_path": "ec_launchers/stale/r"}
            }
        },
        dd_renames={},
    )

    assert mappings["ec_launchers/launcher/r"]["new_path"] == "ec_launchers/beam/r"


def test_packaged_entry_survives_where_no_other_source_names_the_path():
    mappings = collect_renamed_to_mappings(
        version_renames={},
        path_mappings={"old_to_new": {"packaged/only": {"new_path": "packaged/new"}}},
        dd_renames={},
    )

    assert mappings["packaged/only"]["new_path"] == "packaged/new"


def test_every_source_contributes_its_own_paths():
    mappings = collect_renamed_to_mappings(
        version_renames={"container/only": "container/new"},
        path_mappings={"old_to_new": {"packaged/only": {"new_path": "packaged/new"}}},
        dd_renames={"annotated/only": "annotated/new"},
    )

    assert {key: value["new_path"] for key, value in mappings.items()} == {
        "packaged/only": "packaged/new",
        "container/only": "container/new",
        "annotated/only": "annotated/new",
    }
