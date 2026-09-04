"""Catalog import receipts cannot overwrite editorial identity state."""

from __future__ import annotations

import re
from copy import deepcopy
from unittest.mock import MagicMock

import pytest

from imas_codex.standard_names.catalog_import import (
    record_catalog_import_provenance,
)


def test_import_receipt_preserves_origin_and_status(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An import records its receipt without carrying editorial fields."""
    identity = {
        "id": "electron_density",
        "origin": "pipeline",
        "status": "active",
        "imported_at": None,
        "catalog_commit_sha": "prior-commit",
    }
    incoming = [
        {
            "id": identity["id"],
            "origin": None,
            "status": None,
            "imported_at": "1900-01-01T00:00:00Z",
            "catalog_commit_sha": "catalog-commit",
        }
    ]
    original = deepcopy(incoming)
    writes: list[tuple[str, dict]] = []

    def query(cypher: str, **params):  # noqa: ANN001
        writes.append((cypher, params))
        write = params["batch"][0]
        for field_name in re.findall(r"sn\.(\w+)\s*=", cypher):
            if field_name == "imported_at":
                identity[field_name] = "database-time"
            elif field_name == "catalog_commit_sha" and write[field_name] is not None:
                identity[field_name] = write[field_name]
            elif field_name in write:
                identity[field_name] = write[field_name]
        return [{"updated": 1}]

    client = MagicMock()
    client.query = MagicMock(side_effect=query)

    updated = record_catalog_import_provenance(incoming, gc=client)

    assert updated == 1
    assert incoming == original
    assert identity["origin"] == "pipeline", "catalog import overwrote origin"
    assert identity["status"] == "active", "catalog import overwrote status"
    assert identity["imported_at"] == "database-time"
    assert identity["catalog_commit_sha"] == "catalog-commit"
    assert len(writes) == 1
    cypher, params = writes[0]
    assert "MATCH (sn:StandardName {id: b.id})" in cypher
    assert "MERGE (sn:StandardName" not in cypher
    assigned_properties = set(re.findall(r"sn\.(\w+)\s*=", cypher))
    assert assigned_properties == {"imported_at", "catalog_commit_sha"}
    assert params["batch"] == [
        {"id": "electron_density", "catalog_commit_sha": "catalog-commit"}
    ]
    assert "Refused catalog import origin/status fields" in caplog.text
