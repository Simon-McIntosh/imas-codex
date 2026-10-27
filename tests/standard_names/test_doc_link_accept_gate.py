"""Acceptance must normalize bare documentation references fail-closed."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from imas_codex.standard_names.graph_store import (
    graph_audits as _owning_graph_audits,
    graph_catalog as _owning_graph_catalog,
    graph_doc_links as _owning_graph_doc_links,
    graph_foundation as _owning_graph_foundation,
    graph_review_persistence as _owning_graph_review_persistence,
)


def _graph_client(responses: list[list[dict[str, Any]]]) -> MagicMock:
    client = MagicMock()
    client.__enter__ = MagicMock(return_value=client)
    client.__exit__ = MagicMock(return_value=False)
    client.query = MagicMock(side_effect=responses)
    return client


def test_normalization_failure_refuses_docs_acceptance() -> None:
    """An unreadable reference authority cannot permit docs acceptance."""
    import imas_codex.standard_names.graph_ops as graph_ops
    from imas_codex.standard_names.review import holds

    client = _graph_client([])
    transaction = MagicMock()
    transaction.run.return_value = [
        {
            "docs_chain_length": 0,
            "documentation": "See [poloidal_magnetic_flux].",
            "edit_status": None,
        }
    ]
    client.session.return_value.__enter__.return_value.begin_transaction.return_value = transaction

    with (
        patch.object(_owning_graph_foundation, "GraphClient", return_value=client),
        patch.object(holds, "clear_holds_not_raised", return_value=0),
        patch.object(_owning_graph_foundation, "docs_accepted", return_value=True),
        patch.object(
            _owning_graph_doc_links,
            "_normalize_bare_doc_links",
            side_effect=RuntimeError("name authority unavailable"),
        ),
        patch.object(_owning_graph_catalog, "write_reviews"),
        patch.object(_owning_graph_audits, "bump_sn_run_counter"),
        pytest.raises(RuntimeError, match="name authority unavailable"),
    ):
        _owning_graph_review_persistence.persist_reviewed_docs(
            sn_id="area_at_plasma_boundary",
            claim_token="claim-token",
            score=0.95,
            model="test/model",
            min_score=0.85,
            rotation_cap=3,
            skip_review_node=True,
            resolution_method="quorum_consensus",
            reviewer_chain_size=3,
        )

    assert transaction.run.call_count == 1
    transaction.rollback.assert_called_once()
    transaction.commit.assert_not_called()


def test_accept_normalizer_uses_shared_reference_parser() -> None:
    """Bare-reference normalization has no graph-local parsing pattern."""
    import imas_codex.standard_names.graph_ops as graph_ops
    from imas_codex.standard_names.doc_links import find_name_references

    assert not hasattr(graph_ops, "_BARE_DOC_LINK_RE")

    client = _graph_client(
        [
            [
                {
                    "id": "area_at_plasma_boundary",
                    "docs": (
                        r"The coefficient is $$C[flux_q]$$; compare "
                        "[poloidal_magnetic_flux]."
                    ),
                }
            ],
            [{"id": "poloidal_magnetic_flux"}],
            [],
        ]
    )
    parser = MagicMock(wraps=find_name_references)

    with patch.object(_owning_graph_foundation, "find_name_references", parser):
        updated = _owning_graph_doc_links._normalize_bare_doc_links(
            client, sn_id="area_at_plasma_boundary"
        )

    assert updated == 1
    assert parser.call_count >= 1
    written = client.query.call_args_list[-1].kwargs["items"][0]["doc"]
    assert "$$C[flux_q]$$" in written
    assert "[poloidal_magnetic_flux](name:poloidal_magnetic_flux)" in written
