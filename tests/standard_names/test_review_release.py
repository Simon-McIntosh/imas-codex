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
from imas_codex.standard_names.catalog_release import (
    _assert_approved_entries_unchanged,
    run_release,
    run_review_release,
)
from imas_codex.standard_names.export import (
    approved_baseline_delta,
    assemble_review_catalog,
)


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


def _add_upstream_remote(isnc_repo: Path, tmp_path: Path) -> Path:
    upstream = tmp_path / "upstream.git"
    _git("init", "--bare", "-b", "main", str(upstream), cwd=tmp_path)
    _git("remote", "add", "upstream", str(upstream), cwd=isnc_repo)
    _git("push", "upstream", "main", cwd=isnc_repo)
    return upstream


def _write_domain_catalog(root: Path, text: str) -> None:
    standard_names = root / "standard_names"
    standard_names.mkdir(parents=True, exist_ok=True)
    (standard_names / "equilibrium.yml").write_text(text, encoding="utf-8")


def test_review_release_full_flow(isnc_repo, tmp_path):
    focus = _write_names_focus(tmp_path)
    reviews = tmp_path / "reviews"
    record: dict = {}

    base_exporter = _stub_exporter(record)

    def exporter(**kwargs):
        result = base_exporter(**kwargs)
        staging = Path(kwargs["staging_dir"])
        _write_domain_catalog(
            staging,
            "- name: plasma_current\n  unit: A\n- name: poloidal_flux\n  unit: Wb\n",
        )
        return result

    report = run_review_release(
        isnc_repo,
        focus,
        "Review batch demo",
        staging_dir=tmp_path / "staging",
        bump="minor",
        reviews_dir=reviews,
        exporter=exporter,
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

    subject = _git("log", "-1", "--format=%s", cwd=isnc_repo).stdout.strip()
    body = _git("log", "-1", "--format=%b", cwd=isnc_repo).stdout.strip()
    assert subject == "sn: add Review batch demo"
    assert report.commit_sha == _git("rev-parse", "HEAD", cwd=isnc_repo).stdout.strip()
    assert "Published 2 entries" in body
    assert "withheld 0" in body
    assert "v0.1.0rc1+demo-batch" in body
    assert "plasma_current" not in subject + body
    assert "poloidal_flux" not in subject + body
    assert ".yml" not in subject + body


def test_review_branch_transport_ignores_upstream_remote(isnc_repo, tmp_path):
    """An upstream PR target cannot redirect branch transport upstream."""
    upstream = _add_upstream_remote(isnc_repo, tmp_path)
    focus = _write_names_focus(tmp_path)

    report = run_review_release(
        isnc_repo,
        focus,
        "Review batch demo",
        staging_dir=tmp_path / "staging",
        bump="minor",
        remote="upstream",
        reviews_dir=tmp_path / "reviews",
        exporter=_stub_exporter({}),
        publisher=_stub_publisher(isnc_repo),
        pr_creator=_stub_pr(),
        pr_target="upstream",
        **_PR_TARGET,
    )

    assert report.errors == [], report.errors
    assert report.remote == "origin"
    assert (
        report.branch
        in _git("ls-remote", "--heads", "origin", report.branch, cwd=isnc_repo).stdout
    )
    assert (
        _git(
            "ls-remote", "--heads", str(upstream), report.branch, cwd=isnc_repo
        ).stdout.strip()
        == ""
    )


@pytest.mark.parametrize(
    "approved_entry",
    [
        "- name: approved_name\n  unit: A\n",
        "- name: approved_name\n  unit: kA\n",
    ],
)
def test_review_export_restores_approved_entry_bytes(
    isnc_repo, tmp_path, approved_entry
):
    """Fresh graph serialization cannot replace a non-batch approved mapping."""
    baseline = "- name: approved_name\n  unit: A\n"
    _write_domain_catalog(isnc_repo, baseline)
    _git("add", "standard_names/equilibrium.yml", cwd=isnc_repo)
    _git("commit", "-m", "approved baseline", cwd=isnc_repo)
    _git("push", "origin", "main", cwd=isnc_repo)
    focus = _write_names_focus(tmp_path)

    def exporter(*, staging_dir, force, review_batch, **_kwargs):
        root = Path(staging_dir)
        _write_domain_catalog(
            root,
            approved_entry + "- name: plasma_current\n  unit: A\n",
        )
        (root / "catalog.yml").write_text("catalog_name: t\n", encoding="utf-8")
        return SimpleNamespace(exported_count=2)

    report = run_review_release(
        isnc_repo,
        focus,
        "Review batch demo",
        staging_dir=tmp_path / "staging",
        bump="minor",
        dry_run=True,
        reviews_dir=tmp_path / "reviews",
        exporter=exporter,
        publisher=_stub_publisher(isnc_repo),
        pr_creator=_stub_pr(),
        **_PR_TARGET,
    )

    assert report.errors == [], report.errors
    assert approved_baseline_delta(
        isnc_repo,
        tmp_path / "staging",
        batch_names=["plasma_current", "poloidal_flux"],
    ).unchanged == ("approved_name",)


@pytest.mark.parametrize(
    ("candidate", "expected_error"),
    [
        ("", "missing=['approved_name']"),
        (
            "- name: approved_name\n  unit: kA\n",
            "byte_changed=['approved_name']",
        ),
    ],
)
def test_approved_baseline_guard_refuses_non_batch_drift(
    tmp_path, candidate, expected_error
):
    approved = tmp_path / "approved"
    staging = tmp_path / "staging"
    _write_domain_catalog(approved, "- name: approved_name\n  unit: A\n")
    _write_domain_catalog(staging, candidate)

    with pytest.raises(ValueError, match="approved catalog baseline changed") as exc:
        _assert_approved_entries_unchanged(
            approved,
            staging,
            batch_names=["batch_name"],
        )

    assert expected_error in str(exc.value)


def test_approved_baseline_guard_requires_every_approved_identity(tmp_path):
    approved = tmp_path / "approved"
    staging = tmp_path / "staging"
    _write_domain_catalog(approved, "- name: batch_name\n  unit: A\n")
    _write_domain_catalog(staging, "")

    with pytest.raises(ValueError, match=r"missing=\['batch_name'\]"):
        _assert_approved_entries_unchanged(
            approved,
            staging,
            batch_names=["batch_name"],
        )


def test_review_assembly_retains_withheld_approved_batch_overlaps(tmp_path):
    approved = tmp_path / "approved"
    staging = tmp_path / "staging"
    approved_names = [f"approved_{index:03d}" for index in range(229)]
    emitted_names = approved_names[:206]
    baseline = "".join(
        f"- name: {name}\n  description: approved {name}\n" for name in approved_names
    )
    fresh = "".join(
        f"- name: {name}\n  description: fresh {name}\n" for name in emitted_names
    )
    _write_domain_catalog(approved, baseline)
    _write_domain_catalog(staging, fresh)
    (staging / "catalog.yml").write_text(
        "catalog_name: t\n"
        "candidate_count: 229\n"
        "published_count: 206\n"
        "domains_included:\n"
        "- equilibrium\n",
        encoding="utf-8",
    )

    before = approved_baseline_delta(approved, staging)
    assert len(before.missing) == 23
    assert len(before.byte_changed) == 206
    assert len(before.unchanged) == 0

    assembly = assemble_review_catalog(
        approved,
        staging,
        batch_names=approved_names,
    )

    assert assembly.baseline_count == 229
    assert assembly.staged_count_before == 206
    assert assembly.staged_count_after == 229
    assert assembly.batch_entries_written == 206
    assert assembly.baseline_entries_added == 23
    assert assembly.emitted_batch_names == tuple(emitted_names)

    after = approved_baseline_delta(approved, staging)
    assert len(after.missing) == 0
    assert len(after.byte_changed) == 206
    assert len(after.unchanged) == 23
    protected_after = approved_baseline_delta(
        approved,
        staging,
        batch_names=list(assembly.emitted_batch_names),
    )
    assert len(protected_after.missing) == 0
    assert len(protected_after.byte_changed) == 0
    assert len(protected_after.unchanged) == 23
    _assert_approved_entries_unchanged(
        approved,
        staging,
        batch_names=list(assembly.emitted_batch_names),
    )

    manifest = yaml.safe_load((staging / "catalog.yml").read_text(encoding="utf-8"))
    assert manifest["candidate_count"] == 229
    assert manifest["published_count"] == 229


def test_review_assembly_reproduces_and_closes_sparse_baseline_failure(
    isnc_repo, tmp_path
):
    """A sparse batch export carries the complete approved baseline forward."""
    approved_names = [f"approved_{index:04d}" for index in range(2223)]
    batch_names = approved_names[:206]
    baseline = "".join(
        f"- name: {name}\n  description: baseline {name}\n" for name in approved_names
    )
    _write_domain_catalog(isnc_repo, baseline)
    _git("add", "standard_names/equilibrium.yml", cwd=isnc_repo)
    _git("commit", "-m", "approved baseline", cwd=isnc_repo)
    _git("push", "origin", "main", cwd=isnc_repo)

    focus = tmp_path / "sparse-batch.yaml"
    focus.write_text(
        yaml.safe_dump(
            {
                "kind": "sn_names",
                "schema_version": 1,
                "name": "sparse-batch",
                "names": batch_names,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    observed = {}

    def exporter(*, staging_dir, force, review_batch, **_kwargs):
        root = Path(staging_dir)
        fresh_batch = "".join(
            f"- name: {name}\n  description: fresh {name}\n" for name in review_batch
        )
        _write_domain_catalog(root, fresh_batch)
        (root / "catalog.yml").write_text(
            "catalog_name: t\n"
            "candidate_count: 206\n"
            "published_count: 206\n"
            "domains_included:\n"
            "- equilibrium\n",
            encoding="utf-8",
        )
        observed["before"] = approved_baseline_delta(isnc_repo, root)
        return SimpleNamespace(exported_count=len(review_batch))

    report = run_review_release(
        isnc_repo,
        focus,
        "Review sparse batch",
        staging_dir=tmp_path / "staging",
        bump="minor",
        dry_run=True,
        reviews_dir=tmp_path / "reviews",
        exporter=exporter,
        publisher=_stub_publisher(isnc_repo),
        pr_creator=_stub_pr(),
        **_PR_TARGET,
    )

    assert report.errors == [], report.errors
    before = observed["before"]
    assert len(before.missing) == 2017
    assert len(before.byte_changed) == 206
    assert len(before.unchanged) == 0

    after = approved_baseline_delta(isnc_repo, tmp_path / "staging")
    assert len(after.missing) == 0
    assert len(after.byte_changed) == 206
    assert len(after.unchanged) == 2017
    protected_after = approved_baseline_delta(
        isnc_repo,
        tmp_path / "staging",
        batch_names=batch_names,
    )
    assert len(protected_after.missing) == 0
    assert len(protected_after.byte_changed) == 0
    assert len(protected_after.unchanged) == 2017

    manifest = yaml.safe_load(
        (tmp_path / "staging" / "catalog.yml").read_text(encoding="utf-8")
    )
    assert manifest["candidate_count"] == 2223
    assert manifest["published_count"] == 2223
    assert manifest["domains_included"] == ["equilibrium"]


def test_plain_final_uses_fork_branch_and_defers_tag(isnc_repo, tmp_path):
    """A final opens an upstream PR without pushing main or creating a tag."""
    upstream = _add_upstream_remote(isnc_repo, tmp_path)
    _git("tag", "v0.1.0rc1", cwd=isnc_repo)
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "catalog.yml").write_text("catalog_name: t\n", encoding="utf-8")
    upstream_main_before = _git(
        "rev-parse", "refs/remotes/upstream/main", cwd=isnc_repo
    ).stdout.strip()

    def publisher(*, staging_dir, isnc_path, push, dry_run, allow_dirty):
        (Path(isnc_path) / "catalog.yml").write_text(
            "catalog_name: final\n", encoding="utf-8"
        )
        _git("add", "catalog.yml", cwd=isnc_path)
        _git("commit", "-m", "publish final candidate", cwd=isnc_path)
        sha = _git("rev-parse", "HEAD", cwd=isnc_path).stdout.strip()
        return SimpleNamespace(errors=[], commit_sha=sha, files_copied=1)

    report = run_release(
        isnc_repo,
        "Final catalog review",
        staging_dir=staging,
        final=True,
        remote="upstream",
        skip_export=True,
        publisher=publisher,
        pr_creator=_stub_pr(),
        **_PR_TARGET,
    )

    assert report.errors == [], report.errors
    assert report.branch == "release/v0.1.0"
    assert report.remote == "origin"
    assert report.pr_url == "https://github.com/example-org/example-catalog/pull/42"
    assert (
        report.branch
        in _git("ls-remote", "--heads", "origin", report.branch, cwd=isnc_repo).stdout
    )
    assert (
        _git(
            "ls-remote", "--heads", str(upstream), report.branch, cwd=isnc_repo
        ).stdout.strip()
        == ""
    )
    assert (
        _git(
            "ls-remote", str(upstream), "refs/heads/main", cwd=isnc_repo
        ).stdout.split()[0]
        == upstream_main_before
    )
    assert "v0.1.0" not in _git("tag", cwd=isnc_repo).stdout.splitlines()
    assert (
        _git("ls-remote", "--tags", "origin", "v0.1.0", cwd=isnc_repo).stdout.strip()
        == ""
    )


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
        rc_version="v0.1.0rc1+west-task-2e",
        batch_size=3,
        minted_from="west_task_2e.yaml",
        changes=[
            {
                "domain": "equilibrium",
                "added": ["a", "b"],
                "changed": ["c"],
                "removed": [],
            }
        ],
    )
    assert title == "WEST equilibrium review batch"
    assert "3 standard names" in body
    assert "2 additions, 1 change, and 0 removals" in body
    assert "v0.1.0rc1" not in title + body
    assert "\n" not in body


def test_review_title_uses_domain_only_for_single_domain():
    from imas_codex.standard_names.release_notes import review_pr_title

    single_domain = [
        {
            "domain": "equilibrium",
            "added": ["plasma_current"],
            "changed": [],
            "removed": [],
        }
    ]
    multi_domain = [
        *single_domain,
        {
            "domain": "transport",
            "added": ["energy_flux"],
            "changed": [],
            "removed": [],
        },
    ]

    assert (
        review_pr_title(rc_version="v0.1.0rc1+west-task-2e", changes=single_domain)
        == "WEST equilibrium review batch"
    )
    multi_title = review_pr_title(
        rc_version="v0.1.0rc1+west-task-2e", changes=multi_domain
    )
    assert multi_title == "WEST review batch"
    assert "equilibrium" not in multi_title
    assert "transport" not in multi_title


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


def test_static_notes_summarize_dd_caveats_without_enumerating_entries():
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
        rc_version="v0.1.0rc1+west-task-2e",
        batch_size=1,
        minted_from="batch.yaml",
        changes=[
            {
                "domain": "equilibrium",
                "added": ["plasma_current"],
                "changed": [],
                "removed": [],
            }
        ],
        dd_gaps=summary,
    )

    assert "1 unresolved and 0 retired caveats" in body
    assert "equilibrium/path" not in body
    assert "https://example.invalid/dd/27" not in body
    assert "\n" not in body


def test_release_notes_prompt_receives_structured_dd_gap_evidence(monkeypatch):
    from imas_codex.standard_names import release_notes

    seen: dict = {}

    def _ok(**kw):
        seen["messages"] = kw["messages"]
        return (
            release_notes.PrNotes(
                title="WEST review batch",
                body=(
                    "This WEST review batch contains one standard name. "
                    "The catalog diff contains zero additions, changes, and removals. "
                    "Review the fixed batch view before approving."
                ),
            ),
            0.0,
            {},
        )

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
        rc_version="v0.1.0rc1+west-task-2e",
        batch_size=1,
        minted_from="batch.yaml",
        dd_gaps=summary,
    )

    assert title == "WEST review batch"
    assert body.startswith("This WEST review batch")
    assert "equilibrium/path" not in body
    assert "registered_exception" not in body
    assert "evidence_token" not in body
    system, user = (message["content"] for message in seen["messages"])
    assert "No headings, bullets, tables" in system
    assert "equilibrium/path" not in user
    assert "one short prose-paragraph" in user


def test_enumerating_model_body_is_rejected_before_publish(monkeypatch):
    from imas_codex.standard_names import release_notes

    model_body = "Summary.\n- plasma_current\nReview the batch."

    def _misleading(**_kwargs):
        return (
            release_notes.PrNotes(
                title="WEST equilibrium review batch", body=model_body
            ),
            0.0,
            {},
        )

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

    title, body = release_notes.build_pr_notes(
        message="Batch",
        rc_version="v0.1.0rc1+west-task-2e",
        batch_size=1,
        minted_from="batch.yaml",
        changes=[
            {
                "domain": "equilibrium",
                "added": ["plasma_current"],
                "changed": [],
                "removed": [],
            }
        ],
        dd_gaps=summary,
    )

    assert title == "WEST equilibrium review batch"
    assert "plasma_current" not in body
    assert "\n" not in body
    assert "must-not-leak" not in body


def test_model_cannot_invent_dd_caveats_for_empty_fact_set(monkeypatch):
    from imas_codex.standard_names import release_notes

    def _invented(**_kwargs):
        return (
            release_notes.PrNotes(
                title="Standard names review batch",
                body="This batch has one name. It has 99 unresolved defects.",
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

    assert "\n" not in body
    assert "99 unresolved defects" not in body


def test_missing_model_title_is_rejected_before_publish(monkeypatch):
    from imas_codex.standard_names import release_notes

    class MissingTitle:
        body = (
            "This WEST batch contains one standard name. "
            "The catalog diff contains one addition. "
            "Review the fixed batch view before approving."
        )

    monkeypatch.setattr(
        "imas_codex.discovery.base.llm.call_llm_structured",
        lambda **_kwargs: (MissingTitle(), 0.0, {}),
    )
    title, body = release_notes.build_pr_notes(
        message="WEST batch",
        rc_version="v0.1.0rc1+west-task-2e",
        batch_size=1,
        minted_from="batch.yaml",
        changes=[
            {
                "domain": "equilibrium",
                "added": ["plasma_current"],
                "changed": [],
                "removed": [],
            }
        ],
    )

    assert title == "WEST equilibrium review batch"
    assert body.startswith("This WEST equilibrium review batch")


def test_build_approval_notes_falls_back_to_empty_on_llm_failure(monkeypatch):
    """An approval-summary synthesis failure yields '' so the deterministic tag
    block is written alone — the fold-back is never blocked by the notes model."""
    from imas_codex.standard_names import release_notes

    def _boom(**kw):
        raise RuntimeError("no model")

    monkeypatch.setattr("imas_codex.discovery.base.llm.call_llm_structured", _boom)
    notes = release_notes.build_approval_notes(
        pr_description="Review batch demo",
        conversation=[{"author": "rev", "kind": "review", "body": "approve"}],
        commit_messages=["publish batch"],
        review_delta="--- a/standard_names/equilibrium.yml\n+++ b/...\n",
    )
    assert notes == ""


def test_build_approval_notes_returns_model_summary(monkeypatch):
    """When the model succeeds, its grounded summary is returned verbatim."""
    from imas_codex.standard_names import release_notes

    def _ok(**kw):
        return (
            release_notes.ApprovalNotes(summary="Reviewers renamed one entry."),
            0.0,
            {},
        )

    monkeypatch.setattr("imas_codex.discovery.base.llm.call_llm_structured", _ok)
    notes = release_notes.build_approval_notes(
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
    assert "1 unresolved and 0 retired caveats" in seen["pr_body"]
    assert "equilibrium/path" not in seen["pr_body"]
    assert "https://example.invalid/dd/27" not in seen["pr_body"]


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
                title="Wrong title",
                body="This batch has one name. Review the batch before approving.",
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
    assert "caveat evidence could not be read" in seen["body"]
    assert "\n" not in seen["body"]


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
