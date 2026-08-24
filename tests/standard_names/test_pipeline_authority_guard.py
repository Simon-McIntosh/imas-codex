"""Catalog writers must preserve pipeline-owned review authority."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from imas_codex.standard_names.catalog_import import guard_catalog_write_payloads
from imas_codex.standard_names.catalog_reconcile import reconcile_catalog
from imas_codex.standard_names.protection import (
    PIPELINE_AUTHORITY_FIELDS,
    PIPELINE_AUTHORITY_RELATIONSHIPS,
    PipelineAuthorityError,
    refuse_pipeline_authority_loss,
)


def _catalog(tmp_path: Path, entry: dict) -> Path:
    root = tmp_path / "catalog"
    directory = root / "standard_names"
    directory.mkdir(parents=True)
    (directory / "equilibrium.yml").write_text(
        yaml.safe_dump([entry]), encoding="utf-8"
    )
    return root


def _graph_client(row: dict) -> tuple[MagicMock, list[str]]:
    statements: list[str] = []

    def query(cypher: str, **params):  # noqa: ANN001
        statements.append(cypher)
        if "UNWIND $ids AS id" in cypher:
            return [row]
        return []

    client = MagicMock()
    client.query = MagicMock(side_effect=query)
    return client, statements


def test_reconcile_refuses_score_and_model_clear_on_scored_name(
    tmp_path: Path,
) -> None:
    """A catalog payload cannot erase an existing names-axis verdict."""
    name = "normalized_collisionality"
    root = _catalog(
        tmp_path,
        {
            "name": name,
            "description": "Catalog wording.",
            "documentation": "Catalog documentation.",
            "unit": "1",
            "reviewer_score_name": None,
            "reviewer_model_name": None,
        },
    )
    client, statements = _graph_client(
        {
            "id": name,
            "description": "Graph wording.",
            "documentation": "Graph documentation.",
            "unit": "1",
            "reviewer_score_name": 0.96,
            "reviewer_model_name": "openrouter/reviewer",
            "reviews": [f"{name}:name:review"],
            "structural_authorities": [],
        }
    )

    with pytest.raises(PipelineAuthorityError, match="pipeline-authoritative"):
        reconcile_catalog(root, gc=client)

    assert not any("SET sn.description" in statement for statement in statements)


@pytest.mark.parametrize(
    ("field", "existing", "proposed"),
    [
        ("reviewer_score_name", 0.94, 0.71),
        ("reviewer_model_name", "names-reviewer", "catalog-reviewer"),
        ("reviewer_score_docs", 0.91, None),
        ("reviewer_model_docs", "docs-reviewer", None),
    ],
)
def test_guard_refuses_same_axis_scalar_replacement(
    field: str,
    existing: object,
    proposed: object,
) -> None:
    with pytest.raises(PipelineAuthorityError, match=field):
        refuse_pipeline_authority_loss(
            [{"id": "plasma_current", field: proposed}],
            current_by_id={"plasma_current": {field: existing}},
        )


@pytest.mark.parametrize(
    ("field", "existing"),
    [
        ("reviews", ["plasma_current:name:terminal"]),
        ("structural_authorities", ["plasma_current:children:digest"]),
    ],
)
def test_guard_refuses_authority_record_removal(
    field: str,
    existing: list[str],
) -> None:
    with pytest.raises(PipelineAuthorityError, match=field):
        refuse_pipeline_authority_loss(
            [{"id": "plasma_current", field: []}],
            current_by_id={"plasma_current": {field: existing}},
        )


def test_guard_is_non_mutating_and_allows_omitted_authority() -> None:
    proposed = [{"id": "plasma_current", "description": "Revised wording."}]
    original = [dict(item) for item in proposed]

    guarded = refuse_pipeline_authority_loss(
        proposed,
        current_by_id={
            "plasma_current": {
                "reviewer_score_name": 0.95,
                "reviewer_model_name": "names-reviewer",
                "reviews": ["plasma_current:name:terminal"],
                "structural_authorities": [],
            }
        },
    )

    assert proposed == original
    assert guarded == proposed
    assert guarded is not proposed
    assert guarded[0] is not proposed[0]


def test_catalog_adapter_uses_central_authority_registry() -> None:
    assert PIPELINE_AUTHORITY_FIELDS == {
        "reviewer_score_name",
        "reviewer_model_name",
        "reviewer_score_docs",
        "reviewer_model_docs",
    }
    assert PIPELINE_AUTHORITY_RELATIONSHIPS == {
        "reviews",
        "structural_authorities",
    }

    with pytest.raises(PipelineAuthorityError, match="reviews"):
        guard_catalog_write_payloads(
            [{"name": "plasma_current", "reviews": []}],
            current_by_id={
                "plasma_current": {"reviews": ["plasma_current:name:terminal"]}
            },
        )


def test_graph_read_failure_refuses_reconcile(tmp_path: Path) -> None:
    root = _catalog(
        tmp_path,
        {
            "name": "plasma_current",
            "description": "Catalog wording.",
            "documentation": "Catalog documentation.",
            "unit": "A",
        },
    )
    client = MagicMock()
    client.query = MagicMock(side_effect=OSError("graph unavailable"))

    with pytest.raises(PipelineAuthorityError, match="could not be read"):
        reconcile_catalog(root, gc=client)

    assert client.query.call_count == 1


def test_reconcile_positive_allow_list_excludes_authority_fields(
    tmp_path: Path,
) -> None:
    name = "plasma_current"
    root = _catalog(
        tmp_path,
        {
            "name": name,
            "description": "Catalog wording.",
            "documentation": "Catalog documentation.",
            "unit": "A",
        },
    )
    client, statements = _graph_client(
        {
            "id": name,
            "description": "Graph wording.",
            "documentation": "Graph documentation.",
            "unit": "A",
            "reviewer_score_name": 0.96,
            "reviewer_model_name": "names-reviewer",
            "reviewer_score_docs": 0.93,
            "reviewer_model_docs": "docs-reviewer",
            "reviews": [f"{name}:name:terminal", f"{name}:docs:terminal"],
            "structural_authorities": [],
        }
    )

    report = reconcile_catalog(root, gc=client)

    assert report.updated == 1
    write = next(
        statement for statement in statements if "SET sn.description" in statement
    )
    for field in PIPELINE_AUTHORITY_FIELDS:
        assert f"sn.{field}" not in write
    assert "HAS_REVIEW" not in write
    assert "HAS_STRUCTURAL_AUTHORITY" not in write
