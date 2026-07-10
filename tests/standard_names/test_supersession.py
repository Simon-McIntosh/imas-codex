"""Supersession events record the predecessor's stage so the export boundary
can decide whether an accepted name warrants a deprecation stub.

Both supersede sites in ``graph_ops`` must stamp ``superseded_from_stage``:

  * ``supersede_prior_source_names`` supersedes a node still carrying its live
    name-axis stage, so it records that stage verbatim (``o_prior_stage``);
  * ``persist_refined_name`` runs after the caller has flipped the predecessor
    to ``'refining'``, so it records the durable published signal (docs_stage)
    instead — a name only reaches consumers once docs review accepts it.

These are behavioural contract tests: they capture the emitted Cypher through a
mocked graph client (no live Neo4j).
"""

from __future__ import annotations

from unittest.mock import patch

from imas_codex.standard_names.graph_ops import (
    persist_refined_name,
    supersede_prior_source_names,
)

# graph_ops binds GraphClient at module import time, so patch that namespace.
_GC_PATH = "imas_codex.standard_names.graph_ops.GraphClient"


class TestSupersedePriorSourceNamesRecording:
    def test_records_live_stage(self):
        captured: dict[str, str] = {}

        class _GC:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def query(self, cypher, **kwargs):
                captured["cypher"] = cypher
                return []

        with patch(_GC_PATH, return_value=_GC()):
            supersede_prior_source_names(
                [{"new_name": "new_name", "source_id": "some/dd/path"}]
            )

        cypher = captured["cypher"]
        assert "superseded_from_stage" in cypher
        # The verbatim live-stage capture is threaded through a WITH binding.
        assert "o_prior_stage" in cypher


class TestPersistRefinedNameRecording:
    def test_records_published_signal(self):
        captured: dict[str, str] = {}

        class _Tx:
            closed = False

            def run(self, cypher, **kwargs):
                captured["cypher"] = cypher
                return [
                    {"new_name": kwargs["new_name"], "old_name": kwargs["old_name"]}
                ]

            def commit(self):
                return None

            def close(self):
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

            def query(self, *args, **kwargs):
                return []

        with patch(_GC_PATH, return_value=_GC()):
            # Passing edit_status explicitly skips the predecessor edit-field
            # lookup branch, keeping the test focused on the supersede write.
            persist_refined_name(
                old_name="old_name",
                new_name="new_name",
                description="desc",
                edit_status="open",
            )

        cypher = captured["cypher"]
        assert "superseded_from_stage" in cypher
        # The pre-refining name-axis stage is gone by persist time; the durable
        # published signal is docs_stage.
        assert "docs_stage" in cypher
