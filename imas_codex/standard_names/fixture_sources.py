"""Fixture-source identities that live in the graph beside real data.

Persistent test fixtures are seeded as ``StandardNameSource`` rows with these
prefixes so a live graph can carry them alongside real sources without either
population claiming the other's names.  The exact-name preflight in
:func:`~imas_codex.standard_names.graph_ops.scope_exact_standard_names`
refuses an invocation when a requested name's ``HAS_PARENT`` lineage is
produced by a fixture source, so ``sn run --name`` can never bind fixture
lineage into a production run scope.
"""

from __future__ import annotations

FIXTURE_SOURCE_ID_PREFIX = "dd:test_review_entry__"
FIXTURE_SOURCE_PATH_PREFIX = "test/"
