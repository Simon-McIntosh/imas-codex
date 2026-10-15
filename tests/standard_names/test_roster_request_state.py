"""A frozen review roster protects its members only while its request is open.

The roster's recorded candidate request is read once per call through the
shared promote reader. An open request protects its members; a merged or
closed one contributes nothing, because a name a merged request approved is
protected by ``name_stage=approved``; and a state that cannot be read protects,
with evidence naming it — an unreadable state is never permission.

Each test fails when its mechanism is removed: restoring protection for any
recorded request makes the closed and merged tests fail.
"""

from __future__ import annotations

from typing import Any

import pytest

from imas_codex.standard_names import promote
from imas_codex.standard_names.protection import recommended_name_protections

_MEMBER = "flux_surface_averaged_electron_density_at_plasma_boundary"
_PR_URL = "https://github.com/Simon-McIntosh/imas-standard-names-catalog/pull/3"


class _StageGraph:
    """GraphClient stand-in answering the guard's ``name_stage`` read."""

    def __init__(self, stages: dict[str, str] | None = None):
        self.stages = dict(stages or {})

    def query(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        return [
            {"id": name, "name_stage": self.stages.get(name)}
            for name in params.get("names") or []
        ]


def _roster(tmp_path: Any, *, pr_url: Any, pr_number: Any = None) -> str:
    """Write one frozen review roster recording a candidate request."""
    import yaml

    doc: dict[str, Any] = {"names": [_MEMBER]}
    if pr_url is not None:
        doc["pr_url"] = pr_url
    if pr_number is not None:
        doc["pr_number"] = pr_number
    path = tmp_path / "v0.1.0rc1+unit.sn_names.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return str(tmp_path)


def _stub_request(monkeypatch: Any, status: int, payload: Any) -> None:
    """Answer every pull-request read with one recorded reply."""
    monkeypatch.setattr(
        promote, "_pull_request_call", lambda repo, path: (status, payload)
    )


def test_open_request_protects_its_members(monkeypatch: Any, tmp_path: Any) -> None:
    """An open candidate request protects the roster's members."""
    _stub_request(monkeypatch, 200, {"state": "open"})

    evidence = recommended_name_protections(
        _StageGraph(),
        [_MEMBER],
        reviews_dir=_roster(tmp_path, pr_url=_PR_URL),
    )

    assert _MEMBER in evidence
    assert "candidate_request=" in evidence[_MEMBER]
    assert "(open)" in evidence[_MEMBER]


def test_closed_request_does_not_protect(monkeypatch: Any, tmp_path: Any) -> None:
    """A closed candidate request contributes no protection."""
    _stub_request(monkeypatch, 200, {"state": "closed"})

    evidence = recommended_name_protections(
        _StageGraph(),
        [_MEMBER],
        reviews_dir=_roster(tmp_path, pr_url=_PR_URL),
    )

    assert evidence == {}


def test_merged_request_does_not_protect(monkeypatch: Any, tmp_path: Any) -> None:
    """A merged request protects nothing on its own, even with no fold-back."""
    _stub_request(
        monkeypatch,
        200,
        {"state": "closed", "merged": True, "merged_at": "2026-10-11T00:00:00Z"},
    )

    evidence = recommended_name_protections(
        _StageGraph(),
        [_MEMBER],
        reviews_dir=_roster(tmp_path, pr_url=_PR_URL),
    )

    assert evidence == {}


def test_unreadable_state_protects_with_evidence(
    monkeypatch: Any, tmp_path: Any
) -> None:
    """A non-200 reply protects, and the evidence says the state was unreadable."""
    _stub_request(monkeypatch, 502, {"message": "bad gateway"})

    evidence = recommended_name_protections(
        _StageGraph(),
        [_MEMBER],
        reviews_dir=_roster(tmp_path, pr_url=_PR_URL),
    )

    assert _MEMBER in evidence
    assert "candidate_request=" in evidence[_MEMBER]
    assert "state unreadable" in evidence[_MEMBER]


def test_recorded_number_without_url_protects(monkeypatch: Any, tmp_path: Any) -> None:
    """A recorded ``pr_number`` with no readable URL protects, naming it.

    No request read is attempted: with no URL there is no repository to read,
    so the state is unreadable and the name stays protected.
    """
    calls: list[tuple[str, str]] = []

    def _boom(repo: str, path: str) -> tuple[int, Any]:
        calls.append((repo, path))
        raise RuntimeError("no URL to read")

    monkeypatch.setattr(promote, "_pull_request_call", _boom)

    evidence = recommended_name_protections(
        _StageGraph(),
        [_MEMBER],
        reviews_dir=_roster(tmp_path, pr_url=None, pr_number=3),
    )

    assert _MEMBER in evidence
    assert "pr_number=3" in evidence[_MEMBER]
    assert "state unreadable" in evidence[_MEMBER]
    assert calls == []


def test_malformed_url_protects(monkeypatch: Any, tmp_path: Any) -> None:
    """A recorded URL that is not a pull-request URL protects."""
    _stub_request(monkeypatch, 200, {"state": "open"})

    evidence = recommended_name_protections(
        _StageGraph(),
        [_MEMBER],
        reviews_dir=_roster(tmp_path, pr_url="not-a-pr-url"),
    )

    assert _MEMBER in evidence
    assert "state unreadable" in evidence[_MEMBER]


def test_approved_name_stays_protected_whatever_the_roster(
    monkeypatch: Any, tmp_path: Any
) -> None:
    """A name at ``name_stage=approved`` stays protected though its roster is closed."""
    _stub_request(monkeypatch, 200, {"state": "closed"})

    evidence = recommended_name_protections(
        _StageGraph(stages={_MEMBER: "approved"}),
        [_MEMBER],
        reviews_dir=_roster(tmp_path, pr_url=_PR_URL),
    )

    assert evidence[_MEMBER] == "name_stage=approved"
