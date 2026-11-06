"""Container-level DD renames reach the graph as RENAMED_TO edges.

A DD release can rename a whole IDS (``gyrokinetics`` -> ``gyrokinetics_local``)
or an array of structures (``ec_launchers/launcher`` -> ``ec_launchers/beam``).
Neither is a field rename, so a lookup keyed on the field's
``change_nbc_previous_name`` never sees it: the old path is gone, the successor
exists, and nothing connects them. Source reconciliation then treats a name
grounded in the old path as having no successor at all.

The rename record therefore has two sources beyond the packaged path map:
the IDS-level ``change_nbc_previous_name`` annotation, and a structural match
between a removed and an added structure node across a version pair.
"""

from __future__ import annotations

from imas_codex.graph.dd_lifecycle import dd_path_index


def test_ids_rename_maps_every_descendant_path():
    _, old_to_new = dd_path_index()
    assert (
        old_to_new["gyrokinetics/normalizing_quantities/r"]
        == "gyrokinetics_local/normalizing_quantities/r"
    )


def test_struct_array_rename_maps_every_shared_descendant_path():
    from imas_codex.graph.build_dd import (
        _detect_container_renames,
        extract_paths_for_version,
    )

    old = extract_paths_for_version("3.36.0")["paths"]
    new = extract_paths_for_version("3.37.0")["paths"]
    renamed = _detect_container_renames(old, new)
    assert (
        renamed["ec_launchers/launcher/launching_position/r"]
        == "ec_launchers/beam/launching_position/r"
    )


def test_rename_record_builder_carries_both_container_renames():
    from imas_codex.graph.build_dd import (
        _detect_container_renames,
        collect_renamed_to_mappings,
        extract_paths_for_version,
    )

    old = extract_paths_for_version("3.36.0")["paths"]
    new = extract_paths_for_version("3.37.0")["paths"]
    container_renames = _detect_container_renames(old, new)
    _, dd_renames = dd_path_index()

    mappings = collect_renamed_to_mappings(
        version_renames=container_renames,
        path_mappings={"old_to_new": {}},
        dd_renames=dd_renames,
    )

    assert (
        mappings["ec_launchers/launcher/launching_position/r"]["new_path"]
        == "ec_launchers/beam/launching_position/r"
    )
    assert (
        mappings["gyrokinetics/normalizing_quantities/r"]["new_path"]
        == "gyrokinetics_local/normalizing_quantities/r"
    )
