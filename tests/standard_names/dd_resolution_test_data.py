"""Synthetic test data only; production DD resolution truth lives in the graph.

This deliberately tiny row exists solely to give mock-only tests a valid,
nonempty authority shape.  Its synthetic IDS and resolution identities do not
mirror, enumerate, or stand in for the live production records.
"""

SYNTHETIC_GRAPH_RESOLUTION_ROWS = (
    {
        "properties": {
            "id": "dd_resolution:synthetic-test-only",
            "path": "synthetic_ids/channel/value",
            "dd_version": "4.1.1",
            "field": "unit",
            "published_kind": "string",
            "published_value": '"synthetic-published-unit"',
            "effective_kind": "string",
            "effective_value": '"synthetic-effective-unit"',
            "reason": "Exercise the typed graph-authority boundary in mock-only tests.",
            "recorded_by": "synthetic-test-maintainer",
            "recorded_at": "2026-08-19T00:00:00Z",
            "upstream_reference": "none-yet",
            "upstream_commit_reference": None,
            "retiring_release": "none-yet",
            "status": "active",
        },
        "source_paths": ["synthetic_ids/channel/value"],
        "gap_ids": ["dd_gap:synthetic_ids/channel/value:unit_defect"],
        "version_ids": ["4.1.1"],
    },
)

SYNTHETIC_ACTIVE_DIRECTION_ROW = {
    "properties": {
        "id": "dd_resolution:synthetic-active-direction",
        "path": "camera_ir/channel/camera/direction/x",
        "dd_version": "4.1.1",
        "field": "unit",
        "published_kind": "string",
        "published_value": '"m"',
        "effective_kind": "string",
        "effective_value": '"1"',
        "reason": "Exercise exact legacy-authority retirement in a local fixture.",
        "recorded_by": "synthetic-test-maintainer",
        "recorded_at": "2026-08-19T00:00:00Z",
        "upstream_reference": "none-yet",
        "upstream_commit_reference": None,
        "retiring_release": "none-yet",
        "status": "active",
    },
    "source_paths": ["camera_ir/channel/camera/direction/x"],
    "gap_ids": ["dd_gap:camera_ir/channel/camera/direction/x:unit_defect"],
    "version_ids": ["4.1.1"],
}


def load_synthetic_resolution_authority(*rows):
    """Validate explicit synthetic rows through the public graph-loader seam."""
    from imas_codex.standard_names.dd_resolutions import load_dd_resolution_manifest

    class _Reader:
        def read_resolutions(self):
            return rows

    return load_dd_resolution_manifest(graph_reader=_Reader())
