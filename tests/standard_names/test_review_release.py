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


def _write_names_focus(tmp_path):
    p = tmp_path / "batch.yaml"
    p.write_text(
        "kind: sn_names\n"
        "schema_version: 1\n"
        "name: demo-batch\n"
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
    assert report.rc_version == "v0.1.0rc1"
    assert report.batch_size == 2
    # Export received the sorted batch (additive review export).
    assert record["review_batch"] == ["plasma_current", "poloidal_flux"]
    # Branch created and pushed to the fork remote.
    assert report.branch == "review/v0.1.0rc1"
    assert report.pushed is True
    # PR opened and back-filled.
    assert report.pr_number == 42
    assert report.pr_url.endswith("/pull/42")

    # Artifact frozen and back-filled.
    artifact = Path(report.artifact_path)
    assert artifact.name == "v0.1.0rc1.sn_names.yaml"
    doc = yaml.safe_load(artifact.read_text())
    assert doc["kind"] == "sn_names"
    assert doc["names"] == ["plasma_current", "poloidal_flux"]
    assert doc["rc_version"] == "v0.1.0rc1"
    assert doc["pr_number"] == 42
    assert doc["pr_url"].endswith("/pull/42")

    # The review branch exists in the checkout.
    branches = _git("branch", cwd=isnc_repo).stdout
    assert "review/v0.1.0rc1" in branches


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
