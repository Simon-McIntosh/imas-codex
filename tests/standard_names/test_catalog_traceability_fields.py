"""The catalog approval and publication receipts survive a full transport."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import yaml

from imas_codex.standard_names.export import CATALOG_EDGE_MODEL_VERSION
from imas_codex.standard_names.promote import mark_catalog_name_approved
from imas_codex.standard_names.publish import run_publish


class RecordingGraph:
    """Small graph double that applies the two receipt writes and reads back."""

    def __init__(self) -> None:
        self.state: dict[str, Any] = {
            "name_stage": "accepted",
            "docs_stage": "accepted",
            "status": "draft",
            "validation_status": "valid",
        }

    def query(self, statement: str, **parameters: Any) -> list[dict[str, Any]]:
        if "catalog_pr_number = $pr_number" in statement:
            self.state.update(
                catalog_pr_number=parameters["pr_number"],
                catalog_pr_url=parameters["pr_url"],
                catalog_merge_commit_sha=parameters["merge_commit"],
                catalog_reviewer_actor=parameters["reviewer_actor"],
                catalog_approved_at="2026-09-08T10:00:00Z",
                name_stage="approved",
                status="active",
            )
            return [{"id": parameters["name"]}]
        if "SET sn.exported_at = datetime($exported_at)" in statement:
            self.state["exported_at"] = parameters["exported_at"]
            return [{"updated": len(parameters["names"])}]
        if "RETURN sn.catalog_pr_number" in statement:
            return [dict(self.state)]
        raise AssertionError(f"unexpected graph query: {statement}")


def _staging(tmp_path: Path) -> Path:
    staging = tmp_path / "staging"
    (staging / "standard_names").mkdir(parents=True)
    (staging / "standard_names" / "equilibrium.yml").write_text(
        yaml.safe_dump(
            [
                {
                    "name": "electron_temperature",
                    "description": "Electron temperature.",
                    "documentation": "Temperature of electrons.",
                    "unit": "eV",
                }
            ]
        ),
        encoding="utf-8",
    )
    manifest = {
        "catalog_name": "imas-standard-names-catalog",
        "cocos_convention": 17,
        "grammar_version": "0.7.0",
        "isn_model_version": "0.7.0",
        "dd_version_lineage": [],
        "generated_by": "test",
        "generated_at": "2026-09-08T10:00:00Z",
        "exported_at": "2026-09-08T10:00:00Z",
        "candidate_count": 1,
        "published_count": 1,
        "excluded_below_score_count": 0,
        "excluded_unreviewed_count": 0,
        "edge_model_version": CATALOG_EDGE_MODEL_VERSION,
        "export_scope": "domain",
        "domains_included": ["equilibrium"],
        "names": {
            "electron_temperature": {
                "kind": "scalar",
                "status": "active",
                "physics_domain": "equilibrium",
                "links": [],
                "sources": [],
            }
        },
    }
    (staging / "catalog.yml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    return staging


def _isnc_repo(tmp_path: Path) -> Path:
    isnc = tmp_path / "isnc"
    isnc.mkdir()
    subprocess.run(["git", "init"], cwd=isnc, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=isnc,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=isnc,
        check=True,
        capture_output=True,
    )
    (isnc / "README.md").write_text("# catalog\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=isnc, check=True)
    subprocess.run(
        ["git", "commit", "-m", "init"], cwd=isnc, check=True, capture_output=True
    )
    return isnc


def test_accepted_edit_readback_contains_author_request_and_export_receipt(
    tmp_path: Path,
) -> None:
    graph = RecordingGraph()
    mark_catalog_name_approved(
        "electron_temperature",
        catalog_pr_number=12,
        catalog_pr_url="https://github.com/o/r/pull/12",
        catalog_merge_commit_sha="abc123",
        catalog_reviewer_actor="physics-reviewer",
        editorial_outcome="content_edit",
        gc=graph,
    )

    report = run_publish(
        _staging(tmp_path),
        _isnc_repo(tmp_path),
        graph_client=graph,
    )

    assert report.errors == []
    assert report.graph_receipt_count == 1
    row = graph.query(
        """
        MATCH (sn:StandardName {id: $id})
        RETURN sn.catalog_pr_number, sn.catalog_pr_url,
               sn.catalog_reviewer_actor, sn.catalog_approved_at,
               sn.catalog_merge_commit_sha, sn.exported_at
        """,
        id="electron_temperature",
    )[0]
    assert row["catalog_reviewer_actor"] == "physics-reviewer"
    assert row["catalog_pr_number"] == 12
    assert row["catalog_pr_url"] == "https://github.com/o/r/pull/12"
    assert row["catalog_approved_at"] == "2026-09-08T10:00:00Z"
    assert row["catalog_merge_commit_sha"] == "abc123"
    assert row["exported_at"] == "2026-09-08T10:00:00Z"


class _StageGuardedGraph:
    """Enforce the approval writer's declared stage preconditions.

    :class:`RecordingGraph` applies whatever write it is handed, so it cannot
    show that the write is refused when an identity is in neither an accepted
    name stage nor an accepted docs stage. This double refuses where the
    Cypher ``WHERE`` clause refuses, so a guard that stopped guarding would
    fail the test rather than pass it silently.
    """

    def __init__(self, *, name_stage: str, docs_stage: str) -> None:
        self.properties: dict[str, Any] = {
            "name_stage": name_stage,
            "docs_stage": docs_stage,
            "status": "draft",
            "validation_status": "valid",
        }

    def query(self, statement: str, **parameters: Any) -> list[dict[str, Any]]:
        if "catalog_pr_number = $pr_number" not in statement:
            raise AssertionError(f"unexpected graph query: {statement}")
        if self.properties["name_stage"] not in {"accepted", "approved"}:
            return []
        if self.properties["docs_stage"] != "accepted":
            return []
        if self.properties["validation_status"] == "quarantined":
            return []
        self.properties.update(
            catalog_pr_number=parameters["pr_number"],
            catalog_pr_url=parameters["pr_url"],
            catalog_merge_commit_sha=parameters["merge_commit"],
            catalog_reviewer_actor=parameters["reviewer_actor"],
            catalog_approved_at="2026-09-08T10:00:00Z",
            name_stage="approved",
            status="active",
        )
        return [{"id": parameters["name"]}]

    def close(self) -> None:
        return None


_RECEIPT_FIELDS = (
    "catalog_pr_number",
    "catalog_pr_url",
    "catalog_merge_commit_sha",
    "catalog_reviewer_actor",
    "catalog_approved_at",
)


def _approve(graph: Any, name: str = "electron_temperature") -> bool:
    return mark_catalog_name_approved(
        name,
        catalog_pr_number=12,
        catalog_pr_url="https://github.com/o/r/pull/12",
        catalog_merge_commit_sha="abc123",
        catalog_reviewer_actor="physics-reviewer",
        editorial_outcome="content_edit",
        gc=graph,
    )


def test_the_receipt_tuple_is_written_together_on_one_identity() -> None:
    graph = _StageGuardedGraph(name_stage="accepted", docs_stage="accepted")

    assert _approve(graph)

    written = {field for field in _RECEIPT_FIELDS if field in graph.properties}
    assert written == set(_RECEIPT_FIELDS)
    assert graph.properties["catalog_pr_number"] == 12
    assert graph.properties["catalog_pr_url"] == "https://github.com/o/r/pull/12"
    assert graph.properties["catalog_merge_commit_sha"] == "abc123"
    assert graph.properties["catalog_reviewer_actor"] == "physics-reviewer"


def test_the_write_is_refused_when_a_stage_precondition_is_unmet() -> None:
    for name_stage, docs_stage in (
        ("drafted", "accepted"),
        ("contested", "accepted"),
        ("accepted", "drafted"),
        ("accepted", "pending"),
    ):
        graph = _StageGuardedGraph(name_stage=name_stage, docs_stage=docs_stage)

        assert _approve(graph) is False

        assert graph.properties["name_stage"] == name_stage
        assert graph.properties["docs_stage"] == docs_stage
        assert not [field for field in _RECEIPT_FIELDS if field in graph.properties]
