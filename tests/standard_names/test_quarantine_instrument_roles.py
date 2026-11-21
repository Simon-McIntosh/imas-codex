from __future__ import annotations

from typing import Any

import pytest

from iter_codex.standard_names.campaign import default_revalidate


class _ValidationGraph:
    def __init__(self, statuses: dict[str, str]) -> None:
        self.statuses = statuses
        self.writes: list[tuple[str, dict[str, Any]]] = []

    def query(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        ids = list(params.get("ids", []))
        if "collect(sn.id) AS quarantined_ids" in cypher:
            return [
                {
                    "quarantined_ids": [
                        sid for sid in ids if self.statuses.get(sid) == "quarantined"
                    ]
                }
            ]
        if "SET sn.validation_status = 'valid'" in cypher:
            self.writes.append((cypher, params))
            for sid in ids:
                if sid in self.statuses:
                    self.statuses[sid] = "valid"
            return [{"n": len(ids)}]
        raise AssertionError(f"unexpected query: {cypher}")


def test_confirm_instrument_refuses_quarantined_identity_by_name() -> None:
    gc = _ValidationGraph(
        {
            "still_quarantined_density": "quarantined",
            "clean_temperature": "valid",
        }
    )

    with pytest.raises(ValueError, match="still_quarantined_density"):
        default_revalidate(
            gc,
            reintroduced_ids=[],
            clean_ids=["clean_temperature", "still_quarantined_density"],
        )

    assert gc.statuses["still_quarantined_density"] == "quarantined"
    assert gc.writes == []


def test_confirm_instrument_marks_non_quarantined_batch_valid() -> None:
    gc = _ValidationGraph(
        {
            "already_valid_density": "valid",
            "pending_temperature": "pending",
        }
    )

    outcome = default_revalidate(
        gc,
        reintroduced_ids=[],
        clean_ids=["already_valid_density", "pending_temperature"],
    )

    assert outcome == {"requarantined": 0, "confirmed": 2}
    assert gc.statuses == {
        "already_valid_density": "valid",
        "pending_temperature": "valid",
    }
    assert len(gc.writes) == 1
    cypher, params = gc.writes[0]
    assert "WHERE coalesce(sn.validation_status, '') <> 'quarantined'" not in cypher
    assert params["ids"] == ["already_valid_density", "pending_temperature"]
