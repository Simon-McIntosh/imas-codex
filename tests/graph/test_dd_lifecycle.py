"""DD-version lifecycle gate: path index, node stamping, extraction gating.

Standard names are grounded strictly in the current DD's semantics; a path
removed or renamed away in a newer major DD version must never seed a
StandardNameSource. The path index and rename map derive from the packaged
DD XML; the extraction queries exclude ``lifecycle_status='removed'`` nodes.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from imas_codex.graph.dd_lifecycle import (
    LIFECYCLE_REMOVED,
    dd_path_index,
    reconcile_node_lifecycle,
)


class TestDDPathIndex:
    def test_current_paths_present(self):
        valid, _ = dd_path_index()
        assert "summary/local/separatrix/momentum_phi/value" in valid
        assert "core_profiles/profiles_1d/electrons/temperature" in valid

    def test_ddv3_only_paths_absent(self):
        valid, _ = dd_path_index()
        assert "summary/local/separatrix/momentum_tor/value" not in valid
        assert "summary/local/separatrix_average/velocity_tor/argon/value" not in valid

    def test_rename_map_propagates_structure_renames(self):
        _, old2new = dd_path_index()
        # momentum_tor is a structure rename — the child /value path maps too
        assert (
            old2new["summary/local/separatrix/momentum_tor/value"]
            == "summary/local/separatrix/momentum_phi/value"
        )
        assert (
            old2new["summary/local/separatrix_average/velocity_tor/argon/value"]
            == "summary/local/separatrix_average/velocity_phi/argon/value"
        )


class TestReconcileNodeLifecycle:
    def test_stamps_absent_and_restores_returned(self):
        gc = MagicMock()
        gc.query.side_effect = [
            [
                {"id": "summary/local/separatrix/momentum_tor/value", "ls": None},
                {
                    "id": "summary/local/separatrix/momentum_phi/value",
                    "ls": LIFECYCLE_REMOVED,
                },
            ],
            None,  # stamp absent
            None,  # annotate renames
            None,  # restore returned
        ]
        result = reconcile_node_lifecycle(gc=gc)
        assert result["marked_removed"] == 1
        assert result["renamed_annotated"] == 1
        assert result["restored"] == 1
        gc.close.assert_not_called()


class TestExtractionGate:
    def test_dd_extractor_where_clause_gates_removed(self):
        import inspect

        from imas_codex.standard_names.sources import dd as dd_mod

        src = inspect.getsource(dd_mod.extract_dd_candidates)
        assert "coalesce(n.lifecycle_status, '') <> 'removed'" in src

    def test_graph_ops_candidates_gate_removed(self):
        import inspect

        from imas_codex.standard_names import graph_ops

        src = inspect.getsource(graph_ops.get_extraction_candidates_dd)
        assert "coalesce(n.lifecycle_status, '') <> 'removed'" in src

    def test_domain_seed_sweep_gates_removed(self):
        import inspect

        from imas_codex.standard_names import loop

        src = inspect.getsource(loop._list_physics_domains_with_extractable_paths)
        assert "coalesce(n.lifecycle_status, '') <> 'removed'" in src


class TestReconcileSourcesGate:
    def test_reconcile_never_revives_removed_node_sources(self):
        # The revive step must skip sources whose IMASNode is stamped removed,
        # and the stale step must catch sources on removed source_id paths —
        # otherwise the startup reconcile resurrects DDv3 sources into the
        # compose queue on every sn run.
        import inspect

        from imas_codex.standard_names import graph_ops

        src = inspect.getsource(graph_ops.reconcile_standard_name_sources)
        assert "lifecycle_status = 'removed'" in src, "stale step must gate on removed"
        assert "coalesce(imas.lifecycle_status, '') <> 'removed'" in src, (
            "revive step must exclude removed nodes"
        )
