"""Hold the graph's active ISN grammar snapshot to the pyproject pin.

The graph serves one grammar snapshot at a time, and the ``sn`` pipeline
syncs it from the installed ISN package, which the pyproject pin holds. So
the pin, the installed distribution and the active snapshot are three
statements of one version, and the hand-verified pin bumps this compares are
the reason the comparison is a test rather than a convention.

The snapshot is written and rotated by
:mod:`imas_codex.standard_names.grammar_sync`; this test reads it. The pin is
read through the default-tier pin-agreement test so both tiers compare
against one extraction rather than two.
"""

from __future__ import annotations

from typing import Any

import pytest
from packaging.version import InvalidVersion, Version

from tests.core.test_isn_pin_agreement import pinned_isn_version

pytestmark = [pytest.mark.graph, pytest.mark.integration]


@pytest.fixture(scope="module")
def active_snapshot_version(graph_client) -> str:
    """The ``version`` of the one ``ISNGrammarVersion`` node marked active.

    A missing property returns zero rows rather than raising, so each of the
    three ways this query can fail to name a version (no active node, several
    active nodes, a node without a ``version``) is asserted separately: a
    silently empty result would otherwise read as a passing comparison.
    """
    from imas_codex.standard_names import grammar_sync

    rows: list[dict[str, Any]] = graph_client.query(
        "MATCH (v:ISNGrammarVersion {active: true}) "
        "RETURN v.version AS version, v.content_digest AS content_digest"
    )

    assert rows, (
        "the graph carries no active ISNGrammarVersion node, so no grammar "
        f"snapshot is in service; {grammar_sync.__name__} rotates the active flag"
    )
    assert len(rows) == 1, (
        f"the graph marks {len(rows)} ISNGrammarVersion nodes active; exactly one "
        "grammar snapshot must be in service"
    )

    version = rows[0].get("version")
    assert version, "the active ISNGrammarVersion node carries no version"
    return str(version)


def test_active_grammar_snapshot_matches_the_pin(active_snapshot_version: str) -> None:
    pinned = pinned_isn_version()
    try:
        snapshot = Version(active_snapshot_version)
    except InvalidVersion:  # pragma: no cover - graph data fault
        pytest.fail(
            f"the active grammar snapshot version {active_snapshot_version!r} "
            f"is not a PEP 440 version, so it cannot be compared with {pinned}"
        )

    assert snapshot == pinned, (
        f"pyproject pins imas-standard-names=={pinned} but the graph's active "
        f"grammar snapshot is {snapshot}; re-sync the grammar or bump the pin"
    )
