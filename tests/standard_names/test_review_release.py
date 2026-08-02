"""The single review-batch release orchestrator (``run_review_release``).

Exercised against a LOCAL bare repo with the export/publish/PR steps injected —
no live graph (sn-names focus) and no live GitHub call. A separate graph-marked
test drives the sn-sources → mint path.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from imas_codex.graph.client import GraphClient
from imas_codex.standard_names.catalog_release import run_review_release


def _git(*args, cwd):
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    )


@pytest.fixture
def isnc_repo(tmp_path):
    """A local ISNC checkout on 'main' with a bare 'origin' remote."""
    bare = tmp_path / "origin.git"
    _git("init", "--bare", "-b", "main", str(bare), cwd=tmp_path)
    work = tmp_path / "isnc"
    work.mkdir()
    _git("init", "-b", "main", cwd=work)
    _git("config", "user.email", "t@t", cwd=work)
    _git("config", "user.name", "t", cwd=work)
    _git("remote", "add", "origin", str(bare), cwd=work)
    (work / "README.md").write_text("isnc\n")
    _git("add", "README.md", cwd=work)
    _git("commit", "-m", "init", cwd=work)
    _git("push", "origin", "main", cwd=work)
    return work


def _stub_exporter(record):
    def exporter(*, staging_dir, force, review_batch, **kw):
        record["review_batch"] = review_batch
        sd = Path(staging_dir)
        (sd / "standard_names").mkdir(parents=True, exist_ok=True)
        (sd / "catalog.yml").write_text("catalog_name: t\n")
        return SimpleNamespace(exported_count=len(review_batch))

    return exporter


def _stub_publisher(isnc):
    def publisher(*, staging_dir, isnc_path, push, allow_dirty):
        # Simulate a publish commit so there is something to push.
        (Path(isnc_path) / "catalog.yml").write_text("catalog_name: t\n")
        _git("add", "catalog.yml", cwd=isnc_path)
        _git("commit", "-m", "publish", cwd=isnc_path)
        return SimpleNamespace(errors=[], commit_sha="deadbeef", files_copied=1)

    return publisher


def _stub_pr():
    def pr_creator(*, branch, base, title, body, repo, head_owner):
        return 42, f"https://github.com/{repo}/pull/42"

    return pr_creator


# The PR target/fork owner are derived from the checkout's github remotes; the
# bare-repo fixture has none, so tests pass them explicitly.
_PR_TARGET = {
    "upstream_repo": "example-org/example-catalog",
    "fork_owner": "example-fork",
}


@pytest.fixture(autouse=True)
def _isolate_release_tests_from_live_dd_gaps(monkeypatch):
    """Release orchestration tests must not read the operator's live graph."""
    monkeypatch.setattr(
        "imas_codex.standard_names.dd_gaps.list_dd_gaps",
        lambda **_kwargs: [],
        raising=False,
    )


def _write_names_focus(tmp_path, *, name="demo-batch", filename="batch.yaml"):
    p = tmp_path / filename
    p.write_text(
        "kind: sn_names\n"
        "schema_version: 1\n"
        f"name: {name}\n"
        "names:\n"
        "  - poloidal_flux\n"
        "  - plasma_current\n",
        encoding="utf-8",
    )
    return p


def test_review_release_full_flow(isnc_repo, tmp_path):
    focus = _write_names_focus(tmp_path)
    reviews = tmp_path / "reviews"
    record: dict = {}

    report = run_review_release(
        isnc_repo,
        focus,
        "Review batch demo",
        staging_dir=tmp_path / "staging",
        bump="minor",
        reviews_dir=reviews,
        exporter=_stub_exporter(record),
        publisher=_stub_publisher(isnc_repo),
        pr_creator=_stub_pr(),
        **_PR_TARGET,
    )

    assert report.errors == [], report.errors
    # The batch identity rides the version as semver build metadata.
    assert report.rc_version == "v0.1.0rc1+demo-batch"
    assert report.batch_label == "demo-batch"
    assert report.batch_size == 2
    # Export received the sorted batch (additive review export).
    assert record["review_batch"] == ["plasma_current", "poloidal_flux"]
    # Branch created and pushed to the fork remote.
    assert report.branch == "review/v0.1.0rc1+demo-batch"
    assert report.pushed is True
    # PR opened and back-filled.
    assert report.pr_number == 42
    assert report.pr_url.endswith("/pull/42")

    # Artifact frozen and back-filled.
    artifact = Path(report.artifact_path)
    assert artifact.name == "v0.1.0rc1+demo-batch.sn_names.yaml"
    doc = yaml.safe_load(artifact.read_text())
    assert doc["kind"] == "sn_names"
    assert doc["names"] == ["plasma_current", "poloidal_flux"]
    assert doc["rc_version"] == "v0.1.0rc1+demo-batch"
    assert doc["batch_label"] == "demo-batch"
    assert doc["pr_number"] == 42
    assert doc["pr_url"].endswith("/pull/42")

    # The review branch exists in the checkout.
    branches = _git("branch", cwd=isnc_repo).stdout
    assert "review/v0.1.0rc1+demo-batch" in branches


def test_review_release_dry_run_no_push_no_pr(isnc_repo, tmp_path):
    focus = _write_names_focus(tmp_path)
    reviews = tmp_path / "reviews"
    record: dict = {}

    report = run_review_release(
        isnc_repo,
        focus,
        "Review batch demo",
        staging_dir=tmp_path / "staging",
        bump="minor",
        dry_run=True,
        reviews_dir=reviews,
        exporter=_stub_exporter(record),
        publisher=_stub_publisher(isnc_repo),
        pr_creator=_stub_pr(),
        **_PR_TARGET,
    )

    assert report.errors == []
    assert report.pushed is False
    assert report.pr_number is None
    # Export still ran (staging built), artifact still frozen (no PR fields).
    assert record.get("review_batch") == ["plasma_current", "poloidal_flux"]
    doc = yaml.safe_load(Path(report.artifact_path).read_text())
    assert doc["pr_number"] is None
    # No review branch created on a dry run.
    assert "review/" not in _git("branch", cwd=isnc_repo).stdout


def test_review_release_empty_focus_errors(isnc_repo, tmp_path):
    empty = tmp_path / "empty.yaml"
    empty.write_text("kind: sn_names\nschema_version: 1\nname: x\nnames: []\n")
    report = run_review_release(
        isnc_repo,
        empty,
        "x",
        staging_dir=tmp_path / "staging",
        bump="minor",
        reviews_dir=tmp_path / "reviews",
        exporter=_stub_exporter({}),
        publisher=_stub_publisher(isnc_repo),
        pr_creator=_stub_pr(),
        **_PR_TARGET,
    )
    # An empty sn_names list fails schema validation (minItems) → focus error.
    assert report.errors
    assert any("focus" in e for e in report.errors)


# ── graph-marked: sn-sources focus is minted through run_review_release ────

PREFIX = "__revreltest__"
LEAF = f"{PREFIX}/leaf1"


@pytest.fixture
def mint_source(tmp_path):
    with GraphClient() as gc:
        gc.query("MATCH (n) WHERE n.id STARTS WITH $p DETACH DELETE n", p=PREFIX)
        gc.query(
            """
            MERGE (n:StandardName {id: $nid})
              SET n.name_stage='accepted', n.source_paths=[$leaf]
            """,
            nid=f"{PREFIX}_name",
            leaf=LEAF,
        )
    yield
    with GraphClient() as gc:
        gc.query("MATCH (n) WHERE n.id STARTS WITH $p DETACH DELETE n", p=PREFIX)


@pytest.mark.graph
def test_review_release_mints_from_sources(isnc_repo, tmp_path, mint_source):
    ids, leaf = LEAF.split("/", 1)
    focus = tmp_path / "src.yaml"
    focus.write_text(
        "kind: sn_sources\n"
        "schema_version: 1\n"
        "name: demo-sources\n"
        "sources:\n"
        f"  {ids}:\n"
        f"    - {leaf}\n",
        encoding="utf-8",
    )
    record: dict = {}
    report = run_review_release(
        isnc_repo,
        focus,
        "Minted batch",
        staging_dir=tmp_path / "staging",
        bump="minor",
        reviews_dir=tmp_path / "reviews",
        exporter=_stub_exporter(record),
        publisher=_stub_publisher(isnc_repo),
        pr_creator=_stub_pr(),
        **_PR_TARGET,
    )
    assert report.errors == [], report.errors
    assert f"{PREFIX}_name" in report.names
    assert record["review_batch"] == report.names


# ── PR target derivation from the checkout's remotes ───────────────────────


def test_pr_target_derived_from_remotes(isnc_repo, tmp_path):
    """With github-style remotes, repo/owner come from the checkout itself."""
    _git(
        "remote",
        "add",
        "upstream",
        "git@github.com:example-org/example-catalog.git",
        cwd=isnc_repo,
    )
    _git(
        "remote",
        "set-url",
        "origin",
        "https://github.com/example-fork/example-catalog.git",
        cwd=isnc_repo,
    )
    # origin now points at github, so push must be stubbed out via dry_run=False
    # is not possible; instead verify derivation directly.
    from imas_codex.standard_names.catalog_release import _github_slug

    assert _github_slug(isnc_repo, "upstream") == ("example-org", "example-catalog")
    assert _github_slug(isnc_repo, "origin") == ("example-fork", "example-catalog")
    assert _github_slug(isnc_repo, "nosuch") is None


def test_pr_target_fork_uses_origin_slug(isnc_repo, tmp_path):
    """pr_target='fork' derives the PR repo from origin, not upstream."""
    _git(
        "remote",
        "add",
        "upstream",
        "git@github.com:example-org/example-catalog.git",
        cwd=isnc_repo,
    )
    focus = _write_names_focus(tmp_path)
    seen: dict = {}

    def pr_creator(*, branch, base, title, body, repo, head_owner):
        seen["repo"] = repo
        return 1, f"https://github.com/{repo}/pull/1"

    # origin is the local bare path (not github) → fork derivation must fail
    # loudly rather than fall back to upstream.
    report = run_review_release(
        isnc_repo,
        focus,
        "x",
        staging_dir=tmp_path / "staging",
        bump="minor",
        reviews_dir=tmp_path / "reviews",
        exporter=_stub_exporter({}),
        publisher=_stub_publisher(isnc_repo),
        pr_creator=pr_creator,
        fork_owner="example-fork",
        pr_target="fork",
    )
    assert report.errors and "pr_target=fork" in report.errors[0]
    assert "repo" not in seen

    # With a github origin, the fork slug is used as the PR repo.
    _git(
        "remote",
        "set-url",
        "origin",
        "git@github.com:example-fork/example-catalog.git",
        cwd=isnc_repo,
    )
    # Pushing to a fake github URL would fail — pre-create the branch push
    # target locally by re-pointing origin back after derivation is not
    # possible mid-call, so assert via dry_run=False is skipped; instead
    # verify the derivation helper directly.
    from imas_codex.standard_names.catalog_release import _github_slug

    assert _github_slug(isnc_repo, "origin") == ("example-fork", "example-catalog")


# ── grounded PR notes ───────────────────────────────────────────────────────


def test_collect_catalog_changes(isnc_repo):
    """Per-domain added/changed/removed entry names vs the base branch."""
    from imas_codex.standard_names.release_notes import collect_catalog_changes

    sn_dir = isnc_repo / "standard_names"
    sn_dir.mkdir()
    (sn_dir / "equilibrium.yml").write_text(
        "- name: poloidal_flux\n  unit: Wb\n- name: old_entry\n  unit: m\n"
    )
    _git("add", "standard_names", cwd=isnc_repo)
    _git("commit", "-m", "base catalog", cwd=isnc_repo)

    # Worktree: add one, change one, remove one; add a new domain file.
    (sn_dir / "equilibrium.yml").write_text(
        "- name: poloidal_flux\n  unit: Wb\n  documentation: revised\n"
        "- name: plasma_current\n  unit: A\n"
    )
    (sn_dir / "transport.yml").write_text("- name: heat_flux\n  unit: W.m^-2\n")
    # The real flow diffs AFTER publish committed the files — stage them so the
    # new domain file is tracked (untracked files never appear in git diff).
    _git("add", "standard_names", cwd=isnc_repo)

    changes = collect_catalog_changes(isnc_repo, base_ref="HEAD")

    by_domain = {c["domain"]: c for c in changes}
    assert by_domain["equilibrium"]["added"] == ["plasma_current"]
    assert by_domain["equilibrium"]["changed"] == ["poloidal_flux"]
    assert by_domain["equilibrium"]["removed"] == ["old_entry"]
    assert by_domain["transport"]["added"] == ["heat_flux"]


def test_build_pr_notes_falls_back_on_llm_failure(monkeypatch):
    from imas_codex.standard_names import release_notes

    def _boom(**kw):
        raise RuntimeError("no model")

    monkeypatch.setattr("imas_codex.discovery.base.llm.call_llm_structured", _boom)
    title, body = release_notes.build_pr_notes(
        message="WEST batch",
        rc_version="v0.1.0rc1",
        batch_size=3,
        minted_from="west_task_2e.yaml",
    )
    assert title == "WEST batch"
    assert "v0.1.0rc1" in body and "3 standard name(s)" in body
    assert "No linked DD defects were reported" in body


def test_dd_gap_release_summary_is_warning_only_and_lifecycle_complete():
    from imas_codex.standard_names.release_notes import summarize_dd_gap_facts

    facts = [
        {
            "id": "dd_gap:equilibrium/a:unit_defect",
            "path": "equilibrium/a",
            "kind": "unit_defect",
            "status": "flagged",
            "source_paths": ["equilibrium/a"],
        },
        {
            "id": "dd_gap:equilibrium/b:type_wiring",
            "path": "equilibrium/b",
            "kind": "type_wiring",
            "status": "upstream_issue",
            "source_paths": ["equilibrium/b", "equilibrium/b"],
            "upstream_url": "https://example.invalid/dd/12",
        },
        {
            "id": "dd_gap:equilibrium/c:unit_defect",
            "path": "equilibrium/*/c",
            "kind": "unit_defect",
            "status": "resolved_upstream",
            "source_paths": ["equilibrium/0/c"],
            "registry_backend": "dd_unit_exceptions",
            "resolved_dd_version": "4.2.0",
        },
        {
            "id": "dd_gap:equilibrium/d:doc_mismatch",
            "path": "equilibrium/d",
            "kind": "doc_mismatch",
            "status": "rejected",
            "source_paths": ["equilibrium/d"],
        },
    ]

    summary = summarize_dd_gap_facts(facts)

    assert summary["total"] == 4
    assert summary["open_count"] == 1
    assert summary["triaged_count"] == 1
    assert summary["unresolved_count"] == 2
    assert summary["retired_count"] == 2
    assert summary["stale_registry_count"] == 1
    assert summary["by_kind"] == {
        "doc_mismatch": 1,
        "type_wiring": 1,
        "unit_defect": 2,
    }
    assert summary["warning_only"] is True
    assert summary["blocks_release"] is False
    upstream = next(
        fact for fact in summary["facts"] if fact["status"] == "upstream_issue"
    )
    assert upstream["exact_paths"] == ["equilibrium/b"]
    assert upstream["upstream_url"] == "https://example.invalid/dd/12"


def test_static_notes_list_exact_dd_paths_without_blocking_release():
    from imas_codex.standard_names.release_notes import (
        static_pr_notes,
        summarize_dd_gap_facts,
    )

    summary = summarize_dd_gap_facts(
        [
            {
                "id": "dd_gap:equilibrium/path:type_wiring",
                "path": "equilibrium/path",
                "kind": "type_wiring",
                "status": "upstream_issue",
                "source_paths": ["equilibrium/path"],
                "upstream_url": "https://example.invalid/dd/27",
            }
        ]
    )
    _title, body = static_pr_notes(
        message="Batch",
        rc_version="v0.1.0rc1",
        batch_size=1,
        minted_from="batch.yaml",
        dd_gaps=summary,
    )

    assert "`equilibrium/path`" in body
    assert "https://example.invalid/dd/27" in body
    assert "do not suppress sources or block this release" in body


def test_release_notes_prompt_receives_structured_dd_gap_evidence(monkeypatch):
    from imas_codex.standard_names import release_notes

    seen: dict = {}

    def _ok(**kw):
        seen["messages"] = kw["messages"]
        return release_notes.PrNotes(title="Batch", body="Body"), 0.0, {}

    monkeypatch.setattr("imas_codex.discovery.base.llm.call_llm_structured", _ok)
    summary = release_notes.summarize_dd_gap_facts(
        [
            {
                "id": "dd_gap:equilibrium/path:unit_defect",
                "path": "equilibrium/path",
                "kind": "unit_defect",
                "status": "registered_exception",
                "source_paths": ["equilibrium/path"],
                "registry_backend": "dd_unit_exceptions",
            }
        ]
    )

    title, body = release_notes.build_pr_notes(
        message="Batch",
        rc_version="v0.1.0rc1",
        batch_size=1,
        minted_from="batch.yaml",
        dd_gaps=summary,
    )

    assert title == "Batch"
    assert body.startswith("Body\n\n## Data Dictionary caveats")
    assert body.count("## Data Dictionary caveats") == 1
    assert "`equilibrium/path`" in body
    assert "registered_exception" in body
    assert "evidence_token" not in body
    system, user = (message["content"] for message in seen["messages"])
    assert "DD defects stay visible and observational" in system
    assert "equilibrium/path" in user
    assert "registered_exception" in user
    assert "Release-blocking: no" in user


def test_model_authored_dd_caveats_are_replaced_not_duplicated(monkeypatch):
    from imas_codex.standard_names import release_notes

    model_body = (
        "Summary.\n\n"
        "## Data Dictionary caveats\n\nNo linked DD defects were reported.\n\n"
        "## Data Dictionary caveats\n\nRelease-blocking: yes.\n\n"
        "## How to review\n\nInspect the catalog diff."
    )

    def _misleading(**_kwargs):
        return release_notes.PrNotes(title="Batch", body=model_body), 0.0, {}

    monkeypatch.setattr(
        "imas_codex.discovery.base.llm.call_llm_structured", _misleading
    )
    summary = release_notes.summarize_dd_gap_facts(
        [
            {
                "id": "dd_gap:equilibrium/path:type_wiring",
                "path": "equilibrium/path",
                "kind": "type_wiring",
                "status": "upstream_issue",
                "source_paths": ["equilibrium/z", "equilibrium/a"],
                "upstream_url": "https://example.invalid/dd/27",
                "evidence_token": "must-not-leak",
            }
        ]
    )

    _title, body = release_notes.build_pr_notes(
        message="Batch",
        rc_version="v0.1.0rc1",
        batch_size=1,
        minted_from="batch.yaml",
        dd_gaps=summary,
    )

    assert body.count("## Data Dictionary caveats") == 1
    assert "No linked DD defects were reported" not in body
    assert "Release-blocking: yes" not in body
    assert "## How to review" in body
    assert "`equilibrium/a`, `equilibrium/z`" in body
    assert "https://example.invalid/dd/27" in body
    assert "must-not-leak" not in body


def test_model_cannot_invent_dd_caveats_for_empty_fact_set(monkeypatch):
    from imas_codex.standard_names import release_notes

    def _invented(**_kwargs):
        return (
            release_notes.PrNotes(
                title="Batch",
                body=(
                    "Summary.\n\n## Data Dictionary caveats\n\n99 unresolved defects."
                ),
            ),
            0.0,
            {},
        )

    monkeypatch.setattr("imas_codex.discovery.base.llm.call_llm_structured", _invented)
    _title, body = release_notes.build_pr_notes(
        message="Batch",
        rc_version="v0.1.0rc1",
        batch_size=1,
        minted_from="batch.yaml",
        dd_gaps=release_notes.summarize_dd_gap_facts([]),
    )

    assert body.count("## Data Dictionary caveats") == 1
    assert "99 unresolved defects" not in body
    assert "No linked DD defects were reported" in body


def test_build_merge_notes_falls_back_to_empty_on_llm_failure(monkeypatch):
    """A merge-summary synthesis failure yields '' so the deterministic tag
    block is written alone — the fold-back is never blocked by the notes model."""
    from imas_codex.standard_names import release_notes

    def _boom(**kw):
        raise RuntimeError("no model")

    monkeypatch.setattr("imas_codex.discovery.base.llm.call_llm_structured", _boom)
    notes = release_notes.build_merge_notes(
        pr_description="Review batch demo",
        conversation=[{"author": "rev", "kind": "review", "body": "approve"}],
        commit_messages=["publish batch"],
        review_delta="--- a/standard_names/equilibrium.yml\n+++ b/...\n",
    )
    assert notes == ""


def test_build_merge_notes_returns_model_summary(monkeypatch):
    """When the model succeeds, its grounded summary is returned verbatim."""
    from imas_codex.standard_names import release_notes

    def _ok(**kw):
        return release_notes.MergeNotes(summary="Reviewers renamed one entry."), 0.0, {}

    monkeypatch.setattr("imas_codex.discovery.base.llm.call_llm_structured", _ok)
    notes = release_notes.build_merge_notes(
        pr_description="d",
        conversation=[],
        commit_messages=[],
        review_delta="",
    )
    assert notes == "Reviewers renamed one entry."


def test_review_release_uses_injected_notes_builder(isnc_repo, tmp_path):
    """The notes builder receives the batch evidence; its output titles the PR."""
    focus = _write_names_focus(tmp_path)
    seen: dict = {}

    def notes_builder(**kw):
        seen.update(kw)
        return "Custom title", "Custom body"

    def pr_creator(*, branch, base, title, body, repo, head_owner):
        seen["pr_title"] = title
        seen["pr_body"] = body
        return 5, f"https://github.com/{repo}/pull/5"

    report = run_review_release(
        isnc_repo,
        focus,
        "msg",
        staging_dir=tmp_path / "staging",
        bump="minor",
        reviews_dir=tmp_path / "reviews",
        exporter=_stub_exporter({}),
        publisher=_stub_publisher(isnc_repo),
        pr_creator=pr_creator,
        notes_builder=notes_builder,
        **_PR_TARGET,
    )
    assert report.errors == [], report.errors
    assert seen["rc_version"] == "v0.1.0rc1+demo-batch"
    assert seen["batch_size"] == 2
    assert seen["pr_title"] == "Custom title"
    assert seen["pr_body"] == "Custom body"


def test_review_release_scopes_dd_caveats_to_batch_names(isnc_repo, tmp_path):
    """The release reads exact batch facts once and never mutates graph state."""
    focus = _write_names_focus(tmp_path)
    seen: dict = {"reader_calls": []}

    def gap_reader(**kwargs):
        seen["reader_calls"].append(kwargs)
        return [
            {
                "id": "dd_gap:equilibrium/path:type_wiring",
                "path": "equilibrium/path",
                "kind": "type_wiring",
                "status": "upstream_issue",
                "source_paths": ["equilibrium/path"],
                "affected_name_ids": ["plasma_current"],
                "upstream_url": "https://example.invalid/dd/27",
            }
        ]

    def pr_creator(*, branch, base, title, body, repo, head_owner):
        seen["pr_body"] = body
        return 5, f"https://github.com/{repo}/pull/5"

    report = run_review_release(
        isnc_repo,
        focus,
        "msg",
        staging_dir=tmp_path / "staging",
        bump="minor",
        reviews_dir=tmp_path / "reviews",
        exporter=_stub_exporter({}),
        publisher=_stub_publisher(isnc_repo),
        pr_creator=pr_creator,
        dd_gap_reader=gap_reader,
        **_PR_TARGET,
    )

    assert report.errors == []
    assert seen["reader_calls"] == [
        {
            "name_ids": ["plasma_current", "poloidal_flux"],
            "gc": None,
        }
    ]
    assert report.dd_gap_summary["unresolved_count"] == 1
    assert report.dd_gap_summary["blocks_release"] is False
    assert "`equilibrium/path`" in seen["pr_body"]
    assert "https://example.invalid/dd/27" in seen["pr_body"]


def test_unavailable_dd_gap_read_is_visible_but_not_release_blocking(
    isnc_repo, tmp_path, monkeypatch
):
    from imas_codex.standard_names import release_notes

    focus = _write_names_focus(tmp_path)
    seen: dict = {}

    def gap_reader(**_kwargs):
        raise RuntimeError("graph unavailable")

    def _misleading(**_kwargs):
        return (
            release_notes.PrNotes(
                title="Batch",
                body=(
                    "Summary.\n\n## Data Dictionary caveats\n\n"
                    "No linked DD defects were reported."
                ),
            ),
            0.0,
            {},
        )

    def pr_creator(*, branch, base, title, body, repo, head_owner):
        seen["body"] = body
        return 5, f"https://github.com/{repo}/pull/5"

    monkeypatch.setattr(
        "imas_codex.discovery.base.llm.call_llm_structured", _misleading
    )

    report = run_review_release(
        isnc_repo,
        focus,
        "msg",
        staging_dir=tmp_path / "staging",
        bump="minor",
        reviews_dir=tmp_path / "reviews",
        exporter=_stub_exporter({}),
        publisher=_stub_publisher(isnc_repo),
        pr_creator=pr_creator,
        dd_gap_reader=gap_reader,
        notes_builder=release_notes.build_pr_notes,
        **_PR_TARGET,
    )

    assert report.errors == []
    assert report.dd_gap_summary["available"] is False
    assert report.dd_gap_summary["read_error"] == "graph unavailable"
    assert report.dd_gap_summary["blocks_release"] is False
    assert seen["body"].count("## Data Dictionary caveats") == 1
    assert "could not be read (graph unavailable)" in seen["body"]
    assert "No linked DD defects were reported" not in seen["body"]


# ── the batch label in the version (semver build metadata) ─────────────────


def test_batch_label_falls_back_to_the_manifest_filename_stem(isnc_repo, tmp_path):
    """A manifest with no usable name is labelled from its filename."""
    focus = tmp_path / "no_name_batch.yaml"
    focus.write_text(
        "kind: sn_names\nschema_version: 1\nname: x\nnames:\n  - plasma_current\n",
        encoding="utf-8",
    )
    # 'name: x' is usable, so override it away to exercise the fallback.
    import yaml as _yaml

    doc = _yaml.safe_load(focus.read_text())
    del doc["name"]
    focus.write_text(_yaml.safe_dump(doc), encoding="utf-8")

    report = run_review_release(
        isnc_repo,
        focus,
        "x",
        staging_dir=tmp_path / "staging",
        bump="minor",
        dry_run=True,
        reviews_dir=tmp_path / "reviews",
        exporter=_stub_exporter({}),
        publisher=_stub_publisher(isnc_repo),
        pr_creator=_stub_pr(),
        **_PR_TARGET,
    )
    # The sn_names schema requires 'name', so the focus load fails first —
    # but the label derivation itself is independent of that and is asserted
    # directly here.
    from imas_codex.standard_names.catalog_release import batch_build_metadata

    assert batch_build_metadata(focus) == "no-name-batch"
    assert report.errors  # schema still enforces 'name' on sn_names files


def test_dry_run_version_and_artifact_carry_the_label(isnc_repo, tmp_path):
    focus = _write_names_focus(tmp_path, name="west-task-2e")
    report = run_review_release(
        isnc_repo,
        focus,
        "x",
        staging_dir=tmp_path / "staging",
        bump="minor",
        dry_run=True,
        reviews_dir=tmp_path / "reviews",
        exporter=_stub_exporter({}),
        publisher=_stub_publisher(isnc_repo),
        pr_creator=_stub_pr(),
        **_PR_TARGET,
    )
    assert report.errors == [], report.errors
    assert report.rc_version == "v0.1.0rc1+west-task-2e"
    assert report.branch == "review/v0.1.0rc1+west-task-2e"
    assert Path(report.artifact_path).name == "v0.1.0rc1+west-task-2e.sn_names.yaml"


def test_batch_rc_counter_continues_past_a_labelled_tag(isnc_repo, tmp_path):
    """The RC counter must see a batch tag — otherwise it silently reuses it."""
    focus = _write_names_focus(tmp_path, name="second-batch")
    _git("tag", "v0.1.0rc1+first-batch", cwd=isnc_repo)
    report = run_review_release(
        isnc_repo,
        focus,
        "x",
        staging_dir=tmp_path / "staging",
        dry_run=True,
        reviews_dir=tmp_path / "reviews",
        exporter=_stub_exporter({}),
        publisher=_stub_publisher(isnc_repo),
        pr_creator=_stub_pr(),
        **_PR_TARGET,
    )
    assert report.errors == [], report.errors
    # rc2, and the NEW batch's label — never the superseded tag's.
    assert report.rc_version == "v0.1.0rc2+second-batch"


def test_frozen_artifact_is_schema_valid_with_the_label(isnc_repo, tmp_path):
    """The label field must not break the sn_names schema the merge side loads."""
    from imas_codex.standard_names.sources_manifest import load_names_file

    focus = _write_names_focus(tmp_path, name="west-task-2e")
    report = run_review_release(
        isnc_repo,
        focus,
        "x",
        staging_dir=tmp_path / "staging",
        bump="minor",
        dry_run=True,
        reviews_dir=tmp_path / "reviews",
        exporter=_stub_exporter({}),
        publisher=_stub_publisher(isnc_repo),
        pr_creator=_stub_pr(),
        **_PR_TARGET,
    )
    assert load_names_file(report.artifact_path) == [
        "plasma_current",
        "poloidal_flux",
    ]
