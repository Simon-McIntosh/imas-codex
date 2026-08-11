"""Source identities that refuse to be rebound to a fresh exact-name scope.

Two populations of ``StandardNameSource`` must never have their produced names
re-scoped by ``sn run --name``:

* the shipped WEST production source manifest, whose names carry a released
  catalog identity, and
* the persistent test fixtures that live in the graph alongside real data.

Both are read by the exact-name preflight in
:func:`~imas_codex.standard_names.graph_ops.scope_exact_standard_names`, which
refuses the whole invocation when a requested name's ``HAS_PARENT`` lineage is
produced by either.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

PROTECTED_SOURCE_MANIFEST = (
    Path(__file__).parent / "manifests" / "west_production_dd_paths.yaml"
)

# Fixture rows are seeded with this id prefix so a live graph can carry them
# beside real sources without either population claiming the other's names.
FIXTURE_SOURCE_ID_PREFIX = "dd:test_review_entry__"
FIXTURE_SOURCE_PATH_PREFIX = "test/"


@cache
def protected_source_ids() -> frozenset[str]:
    """Return the ``StandardNameSource`` ids the shipped manifest owns."""
    from imas_codex.standard_names.sources_manifest import load_sources_file

    return frozenset(
        f"dd:{path}" for path in load_sources_file(PROTECTED_SOURCE_MANIFEST)
    )
