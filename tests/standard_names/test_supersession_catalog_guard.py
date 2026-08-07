"""Published catalog content is ineligible for automatic supersession.

``supersede_prior_source_names`` retires the predecessor name on a source when
a regen pass composes a new one. A name promoted through a merged catalog pull
request is published, governed content: it carries ``name_stage='approved'``
and a ``catalog_pr_number``, both stamped by the same merge write. Retiring
such a name from an unattended compose pass would revoke published content
with no human in the loop — withdrawal is a catalog act.

Every query site that selects a predecessor must therefore exclude it: the
preflight that plans the supersession and the finalize that performs it. A
guard on only one leaves the other free to act on a row the first refused.

Behavioural contract test — the emitted Cypher is captured through a mocked
graph client, so no live Neo4j is needed.
"""

from __future__ import annotations

import re
from unittest.mock import patch

from imas_codex.standard_names.attachment_audit import AttachmentPairingGuardResult
from imas_codex.standard_names.graph_ops import supersede_prior_source_names

# graph_ops binds GraphClient at module import time, so patch that namespace.
_GC_PATH = "imas_codex.standard_names.graph_ops.GraphClient"

_STAGE_EXCLUSION = re.compile(r"NOT coalesce\(old\.name_stage, ''\) IN \[([^\]]*)\]")


def _emitted_cypher() -> list[str]:
    """Drive one supersession, returning every statement it emitted."""
    statements: list[str] = []

    class _Tx:
        def run(self, cypher, **kwargs):
            statements.append(cypher)
            if "GENERATED_SUPERSESSION_PREFLIGHT" in cypher:
                return [
                    {
                        "requested_source_id": "some/dd/path",
                        "new_name": "new_name",
                        "requested_source_exists": True,
                        "successor_exists": True,
                        "old_name": "old_name",
                        "old_stage": "accepted",
                        "source_ids": ["dd:some/dd/path"],
                    }
                ]
            plan = kwargs["plans"][0]
            return [{"old_name": plan["old_name"], "new_name": plan["new_name"]}]

        def commit(self):
            return None

        def rollback(self):
            return None

    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def begin_transaction(self):
            return _Tx()

    class _GC:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def session(self):
            return _Session()

    with (
        patch(_GC_PATH, return_value=_GC()),
        patch(
            "imas_codex.standard_names.attachment_audit.guard_source_pairings",
            return_value=AttachmentPairingGuardResult(("dd:some/dd/path",), ()),
        ),
        patch(
            "imas_codex.standard_names.provenance_lifecycle."
            "retarget_standard_name_sources",
            return_value=1,
        ),
    ):
        supersede_prior_source_names(
            [{"new_name": "new_name", "source_id": "some/dd/path"}]
        )

    return statements


def _predecessor_selections() -> list[str]:
    """Whitespace-normalised statements that select a predecessor name."""
    selections = [
        " ".join(statement.split())
        for statement in _emitted_cypher()
        if "old.name_stage" in statement
    ]
    # Both the planning and the performing site select a predecessor; pinning
    # the count means a third site added later cannot slip past unguarded.
    assert len(selections) == 2, selections
    return selections


def test_predecessor_selection_excludes_published_catalog_content() -> None:
    for statement in _predecessor_selections():
        excluded = _STAGE_EXCLUSION.search(statement)
        assert excluded is not None, statement
        stages = set(re.findall(r"'([^']*)'", excluded.group(1)))
        assert "approved" in stages, statement
        assert "old.catalog_pr_number IS NULL" in statement, statement


def test_predecessor_selection_still_excludes_terminal_stages() -> None:
    """The catalog guard is additive — the terminal stages stay excluded."""
    for statement in _predecessor_selections():
        excluded = _STAGE_EXCLUSION.search(statement)
        assert excluded is not None, statement
        stages = set(re.findall(r"'([^']*)'", excluded.group(1)))
        assert {"superseded", "exhausted", "contested"} <= stages, statement


def test_structural_parents_stay_ineligible() -> None:
    """A derived parent is owned by the admission gate, not by one source."""
    for statement in _predecessor_selections():
        assert "coalesce(old.origin, 'pipeline') <> 'derived'" in statement, statement
