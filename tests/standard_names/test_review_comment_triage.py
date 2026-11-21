"""Review comments become explicit pipeline input or explicit adjudication."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from iter_codex.standard_names.review_triage import (
    CommentClass,
    CommentDisposition,
    ingest_and_triage,
    route_triage,
)


@pytest.fixture
def catalog_root(tmp_path: Path) -> Path:
    catalog = tmp_path / "standard_names"
    catalog.mkdir()
    (catalog / "core.yaml").write_text(
        "- name: electron_temperature\n"
        "  description: Electron temperature.\n"
        "  documentation: The electron temperature.\n"
        "  unit: eV\n"
        "  kind: scalar\n"
        "- name: ion_density\n"
        "  description: Ion density.\n"
        "  documentation: The ion density.\n"
        "  unit: m^-3\n"
        "  kind: scalar\n"
    )
    return tmp_path


def test_ingestion_reads_line_and_request_comment_surfaces(catalog_root: Path) -> None:
    calls: list[str] = []

    def api(method: str, path: str):
        assert method == "GET"
        calls.append(path)
        if "/pulls/7/comments" in path:
            return 200, [
                {
                    "id": 11,
                    "body": "Rename to `electron_temperature_corrected`.",
                    "path": "standard_names/core.yaml",
                    "line": 1,
                    "html_url": "https://example.test/comments/11",
                    "user": {"login": "reviewer"},
                }
            ]
        return 200, [{"id": 12, "body": "Does this belong in the batch?"}]

    report = ingest_and_triage(
        "https://github.com/example/catalog/pull/7",
        catalog_root=catalog_root,
        github_call=api,
    )

    assert len(calls) == 2
    assert report.comments[0].resolved.standard_name_id == "electron_temperature"
    assert report.comments[0].classification is CommentClass.NAME
    assert report.comments[0].disposition is CommentDisposition.PROPOSAL
    assert report.comments[1].resolved.standard_name_id is None
    assert report.comments[1].disposition is CommentDisposition.BATCH_ADJUDICATION
    assert report.disposition_counts == {
        "proposal": 1,
        "contested": 0,
        "batch_adjudication": 1,
        "closed": 0,
        "adjudication": 0,
    }


def test_unknown_or_unresolved_comment_requires_adjudication(
    catalog_root: Path,
) -> None:
    report = ingest_and_triage(
        "https://github.com/example/catalog/pull/8",
        catalog_root=catalog_root,
        github_call=lambda _method, path: (
            200,
            [
                {
                    "id": 21,
                    "body": "Please improve this.",
                    "path": "standard_names/missing.yaml",
                    "line": 2,
                }
            ]
            if "/pulls/" in path
            else [],
        ),
    )

    item = report.comments[0]
    assert item.disposition is CommentDisposition.ADJUDICATION
    assert item.resolved.resolution_error == "annotated catalog file is not present"
    assert report.disposition_counts["adjudication"] == 1


def test_wording_comment_preserves_text_and_comment_provenance(
    catalog_root: Path,
) -> None:
    report = ingest_and_triage(
        "https://github.com/example/catalog/pull/9",
        catalog_root=catalog_root,
        github_call=lambda _method, path: (
            200,
            [
                {
                    "id": 31,
                    "body": 'Description should read "Electron temperature at the measurement location."',
                    "path": "standard_names/core.yaml",
                    "line": 2,
                    "html_url": "https://example.test/comments/31",
                    "user": {"login": "physicist"},
                }
            ]
            if "/pulls/" in path
            else [],
        ),
    )
    graph = MagicMock()
    plan = SimpleNamespace(blocked=None)

    with patch(
        "iter_codex.standard_names.review_triage.apply_edit", return_value=plan
    ) as apply:
        route_triage(report, gc=graph)

    kwargs = apply.call_args.kwargs
    assert kwargs["target"] == "electron_temperature"
    assert kwargs["docs"] == "Electron temperature at the measurement location."
    assert kwargs["origin"] == "human"
    assert kwargs["refine"] is False
    assert "comment 31" in kwargs["reason"]
    assert "https://example.test/comments/31" in kwargs["reason"]
    assert report.routed[0]["disposition"] == "staged"


def test_physics_objection_reuses_contested_state(catalog_root: Path) -> None:
    report = ingest_and_triage(
        "https://github.com/example/catalog/pull/10",
        catalog_root=catalog_root,
        github_call=lambda _method, path: (
            200,
            [
                {
                    "id": 41,
                    "body": "This is the wrong physical quantity for the stated diagnostic.",
                    "path": "standard_names/core.yaml",
                    "line": 2,
                    "user": {"login": "physicist"},
                }
            ]
            if "/pulls/" in path
            else [],
        ),
    )
    graph = MagicMock()

    with patch("iter_codex.standard_names.review_triage._contest") as contest:
        route_triage(report, gc=graph)

    contest.assert_called_once()
    assert contest.call_args.args == ("electron_temperature",)
    assert contest.call_args.kwargs["reason"].startswith("catalog review comment 41")
    assert report.routed[0]["disposition"] == "contested"


def test_acknowledgement_closes_without_graph_write(catalog_root: Path) -> None:
    report = ingest_and_triage(
        "https://github.com/example/catalog/pull/11",
        catalog_root=catalog_root,
        github_call=lambda _method, path: (
            200,
            [
                {
                    "id": 51,
                    "body": "LGTM, no action needed.",
                    "path": "standard_names/core.yaml",
                    "line": 2,
                }
            ]
            if "/pulls/" in path
            else [],
        ),
    )

    assert report.comments[0].classification is CommentClass.ACKNOWLEDGEMENT
    assert report.comments[0].disposition is CommentDisposition.CLOSED
    graph = MagicMock()
    route_triage(report, gc=graph)
    graph.query.assert_not_called()
