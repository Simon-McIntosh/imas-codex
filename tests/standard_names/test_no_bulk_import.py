"""Lock the removal of the bulk catalog-import feature.

Bulk catalog import (``run_import`` / ``_write_import_entries`` and the
``sn import`` CLI verb) was removed because it dropped provenance on every
round-trip.  It is superseded by:

- ``sn merge`` — folds reviewed curator edits from a merged catalog PR back
  into the ledger; and
- ``catalog_reconcile.reconcile_catalog`` — the diff-by-id reconciler that
  restores the graph from a published catalog (replaying each entry's
  ``sources:`` block).

These tests assert the bulk-import surface stays gone so it can never be
reintroduced by accident, while the read-only ``check_catalog`` divergence
report keeps working.
"""

from __future__ import annotations

import pytest


class TestBulkImportRemoved:
    """The node-recreating bulk-import surface must not exist."""

    @pytest.mark.parametrize(
        "attr",
        [
            "run_import",
            "_write_import_entries",
            "_write_catalog_entries",
            "ImportReport",
            "_fetch_graph_state",
            "_protected_fields_differ",
            "_entry_is_unchanged",
            "_validate_unit_against_graph",
        ],
    )
    def test_symbol_gone(self, attr: str) -> None:
        from imas_codex.standard_names import catalog_import

        assert not hasattr(catalog_import, attr), (
            f"{attr} was removed with bulk import — it must not be re-added. "
            "Use `sn merge` (PR acceptance) or catalog_reconcile (graph restore)."
        )

    def test_import_verb_gone_from_cli(self) -> None:
        """The `sn import` command must no longer be registered."""
        from imas_codex.cli.sn import sn

        assert "import" not in sn.commands, (
            "`sn import` was removed — the CLI must not expose a bulk-import verb."
        )
        # The pipeline verbs must still be present.
        for verb in ("run", "release", "edit"):
            assert verb in sn.commands, f"`sn {verb}` unexpectedly missing"


class TestCheckCatalogSurvives:
    """The read-only divergence report must keep working."""

    def test_check_catalog_importable(self) -> None:
        from imas_codex.standard_names.catalog_import import CheckResult, check_catalog

        assert callable(check_catalog)
        # CheckResult is the report dataclass check_catalog returns.
        result = CheckResult()
        assert result.only_in_catalog == []
        assert result.in_sync == 0
