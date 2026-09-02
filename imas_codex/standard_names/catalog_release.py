"""Release workflow for the ISNC (imas-standard-names-catalog).

Orchestrates the full release cycle: export → publish → tag → push.
The state machine follows the same two-state pattern as codex and ISN
releases (Stable ↔ RC mode).

State machine:
    Stable (v1.0.0) ──bump──→ RC (v1.1.0rc1) ──rc──→ (v1.1.0rc2) ──final──→ Stable (v1.1.0)
"""

from __future__ import annotations

import logging
import re
import shlex
import subprocess
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Only match clean semver tags (ignore legacy hyphen-suffixed tags of the
# form ``v0.3.0-rc1-<label>``). The optional ``+<build>`` suffix is semver build
# metadata carrying the review-batch identity of a batch RC — see
# :func:`sanitize_build_metadata` for the grammar it must satisfy.
_SEMVER_RE = re.compile(
    r"^v(\d+)\.(\d+)\.(\d+)(?:rc(\d+))?(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$"
)

_RC_REMOTE = "origin"
_FINAL_REMOTE = "upstream"


class ReviewPreviewLinkInvariantError(RuntimeError):
    """The created review PR does not expose its exact Pages preview address."""


class _GitHubClient:
    """Small GitHub CLI boundary for creating and verifying review PRs."""

    def create_pull_request(
        self,
        *,
        branch: str,
        base: str,
        title: str,
        body: str,
        repo: str,
        head_owner: str,
    ) -> tuple[int | None, str | None]:
        return _gh_pr_create(
            branch=branch,
            base=base,
            title=title,
            body=body,
            repo=repo,
            head_owner=head_owner,
        )

    def update_pull_request_body(self, *, repo: str, number: int, body: str) -> None:
        result = subprocess.run(
            ["gh", "pr", "edit", str(number), "--repo", repo, "--body", body],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise ReviewPreviewLinkInvariantError(
                f"could not write the preview address to {repo}#{number}: "
                f"{result.stderr.strip()}"
            )

    def read_pull_request_body(self, *, repo: str, number: int) -> str:
        result = subprocess.run(
            [
                "gh",
                "pr",
                "view",
                str(number),
                "--repo",
                repo,
                "--json",
                "body",
                "--jq",
                ".body",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise ReviewPreviewLinkInvariantError(
                f"could not read back the body of {repo}#{number}: "
                f"{result.stderr.strip()}"
            )
        return result.stdout


def _review_preview_url(repo: str, pr_number: int) -> str:
    """Return the deterministic Pages address for a same-repository review PR."""
    owner, catalog = repo.split("/", 1)
    return f"https://{owner}.github.io/{catalog}/pr-{pr_number}/"


def _write_and_verify_review_preview_link(
    github_client: Any,
    *,
    repo: str,
    pr_number: int,
    body: str,
) -> str:
    """Write the exact preview address and require GitHub to return it."""
    preview_url = _review_preview_url(repo, pr_number)
    body_with_preview = f"{body.rstrip()}\n\nPreview: {preview_url}\n"
    github_client.update_pull_request_body(
        repo=repo,
        number=pr_number,
        body=body_with_preview,
    )
    persisted_body = github_client.read_pull_request_body(
        repo=repo,
        number=pr_number,
    )
    if preview_url not in persisted_body:
        raise ReviewPreviewLinkInvariantError(
            f"read-back body for {repo}#{pr_number} lacks exact preview address "
            f"{preview_url}"
        )
    return preview_url


# =============================================================================
# Report model
# =============================================================================


@dataclass
class ReleaseReport:
    """Result of a catalog release operation."""

    version: str = ""
    git_tag: str = ""
    remote: str = ""
    export_count: int = 0
    files_copied: int = 0
    commit_sha: str | None = None
    branch: str = ""
    branch_reclaimed_from: str | None = None
    pr_number: int | None = None
    pr_url: str | None = None
    pushed: bool = False
    dry_run: bool = False
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "git_tag": self.git_tag,
            "remote": self.remote,
            "export_count": self.export_count,
            "files_copied": self.files_copied,
            "commit_sha": self.commit_sha,
            "branch": self.branch,
            "branch_reclaimed_from": self.branch_reclaimed_from,
            "pr_number": self.pr_number,
            "pr_url": self.pr_url,
            "pushed": self.pushed,
            "dry_run": self.dry_run,
            "errors": self.errors,
        }


# =============================================================================
# Git helpers (operate on ISNC checkout)
# =============================================================================


def _run_git(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Run a git command in the ISNC checkout."""
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _closed_pr_heads(isnc_path: Path, branch: str) -> set[str]:
    """Return closed or merged PR heads recorded for *branch*.

    Both catalog remotes are queried because rehearsal PRs live on the fork
    while final review PRs live upstream. A failed lookup contributes no proof:
    branch reclamation then falls back to ancestry and otherwise refuses.
    """
    import json

    repos = {
        f"{owner}/{repo}"
        for remote in (_RC_REMOTE, _FINAL_REMOTE)
        if (slug := _github_slug(isnc_path, remote)) is not None
        for owner, repo in [slug]
    }
    heads: set[str] = set()
    for repo in sorted(repos):
        try:
            result = subprocess.run(
                [
                    "gh",
                    "pr",
                    "list",
                    "--repo",
                    repo,
                    "--state",
                    "closed",
                    "--head",
                    branch,
                    "--limit",
                    "100",
                    "--json",
                    "headRefName,headRefOid,state",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.warning(
                "Could not inspect closed pull requests for %s in %s: %s",
                branch,
                repo,
                exc,
            )
            continue
        if result.returncode != 0:
            logger.warning(
                "Could not inspect closed pull requests for %s in %s: %s",
                branch,
                repo,
                result.stderr.strip(),
            )
            continue
        try:
            rows = json.loads(result.stdout)
        except json.JSONDecodeError:
            logger.warning("gh returned invalid PR data for %s in %s", branch, repo)
            continue
        heads.update(
            row["headRefOid"]
            for row in rows
            if row.get("headRefName") == branch
            and row.get("state") in {"CLOSED", "MERGED"}
            and row.get("headRefOid")
        )
    return heads


def _prepare_release_branch(
    isnc_path: Path,
    branch: str,
    *,
    base_ref: str = "main",
    pr_head_reader: Any | None = None,
) -> tuple[str | None, str | None]:
    """Create *branch* or safely move a stale local copy onto *base_ref*.

    Returns ``(reclaimed_from, error)``. A branch is reclaimable only when its
    head is already contained in the base or exactly matches the recorded head
    of a closed or merged pull request with the same branch identity.
    """
    existing = _run_git("rev-parse", "--verify", f"refs/heads/{branch}", cwd=isnc_path)
    if existing.returncode != 0:
        created = _run_git("checkout", "-b", branch, base_ref, cwd=isnc_path)
        if created.returncode != 0:
            return None, f"failed to create branch {branch}: {created.stderr.strip()}"
        return None, None

    old_head = existing.stdout.strip()
    contained = (
        _run_git(
            "merge-base", "--is-ancestor", old_head, base_ref, cwd=isnc_path
        ).returncode
        == 0
    )
    reader = pr_head_reader or _closed_pr_heads
    pr_heads = set() if contained else set(reader(isnc_path, branch))
    if not contained and old_head not in pr_heads:
        repo = shlex.quote(str(isnc_path))
        branch_arg = shlex.quote(branch)
        base_arg = shlex.quote(base_ref)
        inspect = f"git -C {repo} log --oneline {base_arg}..{branch_arg}"
        delete = f"git -C {repo} branch -D {branch_arg}"
        return None, (
            f"refusing to reclaim local branch {branch} at {old_head}: its commits "
            "are not contained in the current base and no closed or merged pull "
            f"request records that exact head. Inspect with: {inspect}. If those "
            f"commits should be discarded, delete explicitly with: {delete}"
        )

    moved = _run_git("checkout", "-B", branch, base_ref, cwd=isnc_path)
    if moved.returncode != 0:
        return None, (
            f"failed to reclaim branch {branch} from {old_head}: {moved.stderr.strip()}"
        )
    return old_head, None


def sanitize_build_metadata(label: str) -> str:
    """Coerce *label* into legal semver build metadata.

    Semver permits build metadata to be a dot-separated series of identifiers
    made only of ``[0-9A-Za-z-]`` — underscores and every other character are
    illegal. So each run of illegal characters collapses to a single hyphen,
    dots are kept as identifier separators, the result is lower-cased, and
    leading/trailing separators are stripped.

    Raises ValueError when nothing usable survives: a batch release must never
    be cut with a silently-empty label.
    """
    if not label or not label.strip():
        raise ValueError("build metadata label is empty")
    body = re.sub(r"[^0-9a-z.]+", "-", label.strip().lower())
    body = re.sub(r"-{2,}", "-", body)
    # Drop empty identifiers (e.g. from a doubled dot) and edge separators.
    parts = [p.strip("-") for p in body.split(".")]
    body = ".".join(p for p in parts if p)
    if not body:
        raise ValueError(f"no legal semver build metadata in label: {label!r}")
    return body


def batch_build_metadata(manifest: str | Path) -> str:
    """The build-metadata label identifying a review batch, from its manifest.

    Derived from the manifest's own ``name`` field; when that is absent the
    filename stem is used (any ``.sn_names`` inner suffix dropped). Either way
    the value passes through :func:`sanitize_build_metadata`, so a manifest
    whose identity yields no legal label raises rather than cutting an
    unlabelled batch release.
    """
    import yaml

    path = Path(manifest)
    declared: str | None = None
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(doc, dict) and isinstance(doc.get("name"), str):
            declared = doc["name"]
    except (OSError, yaml.YAMLError):
        declared = None
    if declared and declared.strip():
        return sanitize_build_metadata(declared)
    stem = path.name.split(".", 1)[0]
    try:
        return sanitize_build_metadata(stem)
    except ValueError as exc:
        raise ValueError(
            f"{path}: cannot derive a batch label — the manifest declares no "
            "usable 'name' and its filename yields no legal semver build "
            "metadata"
        ) from exc


def _format_tag(
    major: int,
    minor: int,
    patch: int,
    rc: int | None,
    *,
    build: str | None = None,
) -> str:
    """Format version components as a git tag (v1.0.0, v1.0.0rc1, v1.0.0rc1+label).

    Build metadata is only ever carried by a release candidate — a stable
    release is the catalog's authoritative version and must not be qualified
    by the batch it happened to be cut from.
    """
    base = f"v{major}.{minor}.{patch}"
    tag = f"{base}rc{rc}" if rc else base
    if not build:
        return tag
    if rc is None:
        raise ValueError(
            "build metadata is only carried by a release candidate, "
            f"not by the stable tag {tag}"
        )
    return f"{tag}+{sanitize_build_metadata(build)}"


def _parse_version(tag: str) -> tuple[int, int, int, int | None]:
    """Parse a version tag into (major, minor, patch, rc_number|None).

    Handles: v1.0.0, v1.0.0rc1, v1.0.0rc1+build-label. Build metadata does not
    participate in precedence (semver), so it is deliberately absent from the
    returned tuple — read it with :func:`_parse_build`.
    """
    match = _SEMVER_RE.match(tag)
    if not match:
        raise ValueError(f"Cannot parse version tag: {tag}")
    major, minor, patch = int(match[1]), int(match[2]), int(match[3])
    rc = int(match[4]) if match[4] else None
    return major, minor, patch, rc


def _parse_build(tag: str) -> str | None:
    """The build-metadata label of a version tag, or None when it carries none."""
    match = _SEMVER_RE.match(tag)
    if not match:
        raise ValueError(f"Cannot parse version tag: {tag}")
    return match[5]


def _tag_exists(tag: str, *, cwd: Path | None = None) -> bool:
    result = _run_git("tag", "-l", tag, cwd=cwd)
    return bool(result.stdout.strip())


def _commits_since_tag(tag: str, *, cwd: Path | None = None) -> int:
    result = _run_git("rev-list", f"{tag}..HEAD", "--count", cwd=cwd)
    return int(result.stdout.strip()) if result.returncode == 0 else 0


# =============================================================================
# State detection
# =============================================================================


def _tag_sort_key(tag: str) -> tuple[int, int, int, int]:
    """Ordering key for a version tag — **build metadata excluded by design**.

    Two tags differing only in build metadata are equal in precedence (semver),
    so a labelled batch RC and the bare RC of the same number rank identically
    and the RC counter continues from either.

    An unsuffixed tag sorts *below* the RCs of the same M.m.p, mirroring the
    ordering git's version sort produced here before: the state machine leaves
    RC mode via an explicit ``--bump``, never by tag ordering.
    """
    major, minor, patch, rc = _parse_version(tag)
    return (major, minor, patch, rc if rc is not None else -1)


def _get_semver_tags(cwd: Path | None = None) -> list[str]:
    """Get all clean semver tags, ordered by version (descending).

    Ordering is computed in Python rather than delegated to git's
    ``--sort=-v:refname``, which orders a ``+build`` suffix lexically and has no
    notion of build metadata being precedence-neutral.
    """
    result = _run_git("tag", cwd=cwd)
    if result.returncode != 0:
        return []
    tags = [
        tag.strip()
        for tag in result.stdout.strip().splitlines()
        if _SEMVER_RE.match(tag.strip())
    ]
    return sorted(tags, key=_tag_sort_key, reverse=True)


def detect_state(isnc_path: Path, *, fetch_remote: str | None = None) -> dict:
    """Detect current release state from ISNC git tags.

    Parameters
    ----------
    isnc_path:
        Path to the ISNC git checkout.
    fetch_remote:
        If provided, fetch tags from this remote before detecting state.

    Returns
    -------
    Dict with keys: state, tag, major, minor, patch, rc, build, commits_since.
    ``build`` is the batch label of a batch RC (``None`` for a plain release).
    """
    if fetch_remote:
        _run_git("fetch", "--tags", fetch_remote, cwd=isnc_path)

    tags = _get_semver_tags(cwd=isnc_path)
    if not tags:
        return {
            "state": None,
            "tag": None,
            "major": 0,
            "minor": 0,
            "patch": 0,
            "rc": None,
            "build": None,
            "commits_since": 0,
        }

    latest = tags[0]
    major, minor, patch, rc = _parse_version(latest)
    state = "rc" if rc is not None else "stable"
    commits = _commits_since_tag(latest, cwd=isnc_path)

    return {
        "state": state,
        "tag": latest,
        "major": major,
        "minor": minor,
        "patch": patch,
        "rc": rc,
        "build": _parse_build(latest),
        "commits_since": commits,
    }


def _get_latest_stable_tag(cwd: Path | None = None) -> str | None:
    """Get the most recent stable (non-RC) tag."""
    for tag in _get_semver_tags(cwd=cwd):
        _, _, _, rc = _parse_version(tag)
        if rc is None:
            return tag
    return None


def _apply_bump(major: int, minor: int, patch: int, bump: str) -> tuple[int, int, int]:
    if bump == "major":
        return major + 1, 0, 0
    if bump == "minor":
        return major, minor + 1, 0
    return major, minor, patch + 1


def compute_next_version(
    isnc_path: Path,
    bump: str | None,
    *,
    final: bool = False,
    build: str | None = None,
) -> tuple[str, str]:
    """Compute next version tag from current ISNC state.

    Returns (git_tag, version_string) e.g. ("v1.0.0rc1", "1.0.0rc1").

    *build* attaches semver build metadata identifying the review batch the RC
    is cut against (``v0.2.0rc66+west-task-2e``). It is precedence-neutral: the
    RC counter continues from the latest RC whether or not that RC carried a
    label, and the label of the tag being superseded is never inherited — each
    batch RC carries its own.

    Raises
    ------
    ValueError
        If on stable and no bump specified, if *build* is combined with
        *final*, or other invalid transitions.
    """
    if final and build:
        raise ValueError(
            "a stable release carries no build metadata — the catalog's "
            "authoritative version must not be qualified by a review batch"
        )

    info = detect_state(isnc_path)
    state = info["state"]
    major, minor, patch = info["major"], info["minor"], info["patch"]

    if state is None:
        # No tags at all — start fresh
        if bump:
            m, n, p = _apply_bump(0, 0, 0, bump)
        else:
            m, n, p = 1, 0, 0  # Default to v1.0.0
        rc = None if final else 1
        tag = _format_tag(m, n, p, rc, build=build)
        return tag, tag.lstrip("v")

    if state == "stable":
        if not bump:
            raise ValueError(
                f"On stable release {info['tag']}. "
                "Specify --bump (major|minor|patch) to start a new release."
            )
        m, n, p = _apply_bump(major, minor, patch, bump)
        rc = None if final else 1
        tag = _format_tag(m, n, p, rc, build=build)
        return tag, tag.lstrip("v")

    # RC mode
    if bump:
        # Abandon current RC, start new series from latest stable
        stable = _get_latest_stable_tag(cwd=isnc_path)
        if stable:
            s_maj, s_min, s_pat, _ = _parse_version(stable)
        else:
            s_maj, s_min, s_pat = major, minor, patch
        m, n, p = _apply_bump(s_maj, s_min, s_pat, bump)
        rc = None if final else 1
        tag = _format_tag(m, n, p, rc, build=build)
        return tag, tag.lstrip("v")

    if final:
        # Finalize: v1.0.0rc2 → v1.0.0
        tag = _format_tag(major, minor, patch, None)
        return tag, tag.lstrip("v")

    # Increment RC: v1.0.0rc1 → v1.0.0rc2
    next_rc = info["rc"] + 1
    tag = _format_tag(major, minor, patch, next_rc, build=build)
    return tag, tag.lstrip("v")


# =============================================================================
# Pre-flight checks
# =============================================================================


def _check_on_main(isnc_path: Path) -> None:
    result = _run_git("branch", "--show-current", cwd=isnc_path)
    branch = result.stdout.strip()
    if branch != "main":
        raise ValueError(
            f"ISNC not on main branch (current: {branch}). "
            f"Switch first: cd {isnc_path} && git checkout main"
        )


def _check_clean_tree(isnc_path: Path, *, strict: bool = True) -> list[str]:
    """Check if ISNC working tree is clean.

    Returns list of warning strings (empty if clean).
    Raises ValueError if strict and dirty.
    """
    result = _run_git("status", "--porcelain", cwd=isnc_path)
    dirty_lines = [
        line
        for line in result.stdout.strip().splitlines()
        if line.strip() and ".sn-publish.lock" not in line
    ]
    if dirty_lines:
        if strict:
            raise ValueError(
                f"ISNC working tree has {len(dirty_lines)} uncommitted change(s). "
                "Commit changes first."
            )
        return [
            f"Working tree has {len(dirty_lines)} uncommitted change(s) "
            "(allowed for RC)"
        ]
    return []


def _restore_main_after_review_release(func: Any) -> Any:
    """Return the shared catalog checkout to ``main`` after a review cut."""

    @wraps(func)
    def wrapped(isnc_path: Path, *args: Any, **kwargs: Any) -> Any:
        path = Path(isnc_path)
        report = None
        try:
            report = func(path, *args, **kwargs)
            return report
        finally:
            branch = _run_git("branch", "--show-current", cwd=path).stdout.strip()
            if branch and branch != "main":
                restored = _run_git("checkout", "main", cwd=path)
                if restored.returncode != 0:
                    message = (
                        "failed to restore ISNC checkout to main: "
                        f"{restored.stderr.strip()}"
                    )
                    if report is not None:
                        report.errors.append(message)
                    else:
                        logger.error(message)

    return wrapped


def _check_synced(isnc_path: Path, remote: str, *, strict: bool = True) -> list[str]:
    """Check if ISNC is synced with the target remote.

    Returns list of warning strings.
    Raises ValueError if strict and out of sync.
    """
    _run_git("fetch", remote, "main", cwd=isnc_path)
    result = _run_git(
        "rev-list",
        "--left-right",
        "--count",
        f"main...{remote}/main",
        cwd=isnc_path,
    )
    if result.returncode != 0:
        return [f"Could not check sync with {remote}/main"]

    parts = result.stdout.strip().split()
    if len(parts) != 2:
        return []

    ahead, behind = int(parts[0]), int(parts[1])
    warnings = []

    if behind > 0:
        msg = (
            f"ISNC is {behind} commits behind {remote}/main. "
            f"Pull first: cd {isnc_path} && git pull {remote} main"
        )
        if strict:
            raise ValueError(msg)
        warnings.append(msg)

    if ahead > 0:
        msg = (
            f"ISNC is {ahead} commits ahead of {remote}/main. "
            f"Push first: cd {isnc_path} && git push {remote} main"
        )
        if strict:
            raise ValueError(msg)
        warnings.append(msg)

    return warnings


# =============================================================================
# Release status display
# =============================================================================


def get_release_status(isnc_path: Path) -> dict[str, Any]:
    """Get ISNC release status for display.

    Returns dict with state info, available commands, ISN dep version, etc.
    """
    info = detect_state(isnc_path, fetch_remote="origin")

    # Get ISN dependency version from ISNC pyproject.toml
    isn_version = _get_isn_dep_version(isnc_path)

    # Get remotes
    remotes = {}
    for name in ("origin", "upstream"):
        result = _run_git("remote", "get-url", name, cwd=isnc_path)
        if result.returncode == 0:
            remotes[name] = result.stdout.strip()

    # Check GitHub Pages
    pages_enabled = _check_pages_status(isnc_path)

    return {
        **info,
        "isnc_path": str(isnc_path),
        "isn_version": isn_version,
        "remotes": remotes,
        "pages_enabled": pages_enabled,
    }


def _get_isn_dep_version(isnc_path: Path) -> str | None:
    """Extract ISN dependency version from ISNC pyproject.toml."""
    pyproject = isnc_path / "pyproject.toml"
    if not pyproject.is_file():
        return None
    try:
        content = pyproject.read_text(encoding="utf-8")
        # Look for the ISN git dependency tag
        match = re.search(r"imas-standard-names.*@(v[\d.]+(?:rc\d+)?)", content)
        if match:
            return match.group(1)
        # Fallback: look for version specifier
        match = re.search(r"imas-standard-names[>=<~!]*\s*([\d.]+(?:rc\d+)?)", content)
        return match.group(1) if match else None
    except Exception:
        return None


def _check_pages_status(isnc_path: Path) -> bool | None:
    """Check if gh-pages branch exists (proxy for GitHub Pages setup)."""
    result = _run_git("ls-remote", "--heads", "origin", "gh-pages", cwd=isnc_path)
    if result.returncode != 0:
        return None
    return bool(result.stdout.strip())


# =============================================================================
# Main release function
# =============================================================================


def run_release(
    isnc_path: Path,
    message: str,
    *,
    staging_dir: Path | None = None,
    bump: str | None = None,
    final: bool = False,
    remote: str | None = None,
    dry_run: bool = False,
    skip_export: bool = False,
    export_kwargs: dict[str, Any] | None = None,
    exporter: Any | None = None,
    publisher: Any | None = None,
    pr_creator: Any | None = None,
    upstream_repo: str | None = None,
    fork_owner: str | None = None,
    pr_head_reader: Any | None = None,
) -> ReleaseReport:
    """Run the full catalog release workflow.

    Steps:
    1. Pre-flight checks on ISNC checkout
    2. Auto-export (graph → staging) unless skip_export
    3. Copy staging → ISNC (publish)
    4. Compute next version tag
    5. Git commit in ISNC
    6. Create git tag
    7. Push commit + tag to remote

    Parameters
    ----------
    isnc_path:
        Path to the ISNC git checkout.
    message:
        Release message (used for git tag annotation and commit).
    staging_dir:
        Staging directory. If None, uses default from settings.
    bump:
        Version bump type (major, minor, patch). Required for first
        release or when on a stable tag.
    final:
        If True, finalize current RC to stable release.
    remote:
        Git remote to push to. Default: origin for RC, upstream for final.
    dry_run:
        Validate and report without making changes.
    skip_export:
        Skip the export step (use existing staging content).
    export_kwargs:
        Additional kwargs for run_export (e.g., min_score, domain).

    Returns
    -------
    ReleaseReport with version, tag, commit SHA, and any errors.
    """
    report = ReleaseReport(dry_run=dry_run)

    # ── Resolve paths ──────────────────────────────────────
    if staging_dir is None:
        from imas_codex.settings import get_sn_staging_dir

        staging_dir = get_sn_staging_dir()

    is_rc = not final
    version_remote = remote or (_FINAL_REMOTE if final else _RC_REMOTE)
    effective_remote = _RC_REMOTE if final else version_remote
    report.remote = effective_remote
    exporter = exporter or _default_exporter
    publisher = publisher or _default_publisher
    pr_creator = pr_creator or _gh_pr_create

    # ── Pre-flight checks ──────────────────────────────────
    logger.info("Pre-flight checks on %s", isnc_path)

    try:
        _check_on_main(isnc_path)
    except ValueError as exc:
        report.errors.append(str(exc))
        return report

    try:
        warnings = _check_clean_tree(isnc_path, strict=not is_rc)
        for w in warnings:
            logger.warning(w)
    except ValueError as exc:
        report.errors.append(str(exc))
        return report

    try:
        warnings = _check_synced(isnc_path, version_remote, strict=not dry_run)
        for w in warnings:
            logger.warning(w)
    except ValueError as exc:
        report.errors.append(str(exc))
        return report

    # ── Compute version ────────────────────────────────────
    # Fetch tags from remote before computing version
    _run_git("fetch", "--tags", version_remote, cwd=isnc_path)

    try:
        git_tag, version = compute_next_version(isnc_path, bump, final=final)
    except ValueError as exc:
        report.errors.append(str(exc))
        return report

    report.git_tag = git_tag
    report.version = version

    if _tag_exists(git_tag, cwd=isnc_path):
        report.errors.append(f"Tag {git_tag} already exists")
        return report

    logger.info("Next version: %s (tag: %s)", version, git_tag)

    # ── Auto-export ────────────────────────────────────────
    if not skip_export:
        staging_dir.mkdir(parents=True, exist_ok=True)

        kwargs: dict[str, Any] = {
            "staging_dir": staging_dir,
            "force": True,  # Overwrite existing staging
            "final": final,  # Strict gates for final releases
            **(export_kwargs or {}),
        }

        logger.info("Exporting to %s", staging_dir)
        try:
            export_report = exporter(**kwargs)
            report.export_count = export_report.exported_count

            if not export_report.all_gates_passed:
                failed = [g.gate for g in export_report.gate_results if not g.passed]
                report.errors.append(
                    f"Export quality gates failed: {', '.join(failed)}. "
                    "Fix issues or pass --skip-export to bypass."
                )
                return report
        except Exception as exc:
            report.errors.append(f"Export failed: {exc}")
            return report
    else:
        # Validate existing staging
        catalog_yml = staging_dir / "catalog.yml"
        if not catalog_yml.is_file():
            report.errors.append(
                f"No catalog.yml found at {staging_dir}. "
                "Run 'sn export' first, or remove --skip-export."
            )
            return report

    if final:
        report.branch = f"release/{git_tag}"
        if not dry_run:
            reclaimed_from, branch_error = _prepare_release_branch(
                isnc_path,
                report.branch,
                pr_head_reader=pr_head_reader,
            )
            if branch_error:
                report.errors.append(branch_error)
                return report
            report.branch_reclaimed_from = reclaimed_from

    logger.info("Publishing to %s", isnc_path)
    pub_report = publisher(
        staging_dir=str(staging_dir),
        isnc_path=str(isnc_path),
        push=False,  # We handle push ourselves (with tag)
        dry_run=dry_run,
        # Honour the same RC policy the release-layer clean-tree check applied
        # above (strict=not is_rc): an RC the release path admitted with a dirty
        # tree must not then be hard-blocked by publish's own clean-tree gate.
        allow_dirty=is_rc,
    )

    if pub_report.errors:
        report.errors.extend(pub_report.errors)
        return report

    report.files_copied = pub_report.files_copied
    report.commit_sha = pub_report.commit_sha

    if dry_run:
        if final:
            logger.info(
                "[dry-run] Would push %s to the fork and open an upstream PR",
                report.branch,
            )
        else:
            logger.info(
                "[dry-run] Would tag %s and push to %s",
                git_tag,
                effective_remote,
            )
        return report

    if final:
        push_result = _run_git("push", effective_remote, report.branch, cwd=isnc_path)
        if push_result.returncode != 0:
            report.errors.append(
                f"Failed to push {report.branch} to {effective_remote}: "
                f"{push_result.stderr}"
            )
            return report
        report.pushed = True

        if upstream_repo is None:
            slug = _github_slug(isnc_path, _FINAL_REMOTE)
            if slug is None:
                report.errors.append(
                    "cannot derive the upstream PR repository from the "
                    "ISNC checkout's 'upstream' remote"
                )
                return report
            upstream_repo = f"{slug[0]}/{slug[1]}"
        if fork_owner is None:
            slug = _github_slug(isnc_path, _RC_REMOTE)
            if slug is None:
                report.errors.append(
                    "cannot derive the fork owner from the ISNC checkout's "
                    "'origin' remote"
                )
                return report
            fork_owner = slug[0]
        try:
            report.pr_number, report.pr_url = pr_creator(
                branch=report.branch,
                base="main",
                title=message,
                body=(
                    f"Catalog release candidate for `{git_tag}`. The version "
                    "tag is created only after the merged catalog is folded "
                    "back into the graph."
                ),
                repo=upstream_repo,
                head_owner=fork_owner,
            )
        except Exception as exc:
            report.errors.append(f"gh pr create failed: {exc}")
        return report

    # If publish created no commit (no changes), still tag and push
    if report.commit_sha is None:
        logger.info("No changes to commit — tagging current HEAD")

    # ── Create tag ─────────────────────────────────────────
    tag_result = _run_git("tag", "-a", git_tag, "-m", message, cwd=isnc_path)
    if tag_result.returncode != 0:
        report.errors.append(f"Failed to create tag: {tag_result.stderr}")
        return report
    logger.info("Created tag %s", git_tag)

    # ── Push commit + tag ──────────────────────────────────
    # Push main branch first (if there's a new commit)
    if report.commit_sha:
        push_result = _run_git("push", effective_remote, "main", cwd=isnc_path)
        if push_result.returncode != 0:
            # Roll back the tag on push failure
            _run_git("tag", "-d", git_tag, cwd=isnc_path)
            report.errors.append(
                f"Failed to push to {effective_remote}: {push_result.stderr}"
            )
            return report

    # Push tag
    tag_push_result = _run_git("push", effective_remote, git_tag, cwd=isnc_path)
    if tag_push_result.returncode != 0:
        # Roll back the tag on push failure
        _run_git("tag", "-d", git_tag, cwd=isnc_path)
        report.errors.append(
            f"Failed to push tag to {effective_remote}: {tag_push_result.stderr}"
        )
        return report

    report.pushed = True
    logger.info("Pushed %s to %s", git_tag, effective_remote)

    return report


# =============================================================================
# Review-batch release — mint → freeze → export → branch → push → PR → back-fill
# =============================================================================


def _github_slug(isnc_path: Path, remote: str) -> tuple[str, str] | None:
    """Parse a github ``owner/repo`` pair from a remote URL of the ISNC checkout.

    Handles both SSH (``git@github.com:owner/repo.git``) and HTTPS forms.
    Returns None when the remote is missing or not a github URL — the caller
    decides whether that is an error.
    """
    result = _run_git("remote", "get-url", remote, cwd=isnc_path)
    if result.returncode != 0:
        return None
    m = re.search(
        r"github\.com[:/]([\w.-]+)/([\w.-]+?)(?:\.git)?/?$", result.stdout.strip()
    )
    return (m[1], m[2]) if m else None


@dataclass
class ReviewReleaseReport:
    """Result of a review-batch release."""

    dry_run: bool = False
    rc_version: str = ""
    batch_label: str = ""
    batch_size: int = 0
    names: list[str] = field(default_factory=list)
    unmatched_sources: list[str] = field(default_factory=list)
    manifest_size: int = 0
    source_reconciliation: dict[str, Any] = field(default_factory=dict)
    artifact_path: str | None = None
    commit_sha: str | None = None
    branch: str = ""
    branch_reclaimed_from: str | None = None
    remote: str = ""
    pushed: bool = False
    tag_created: bool = False
    tag_pushed: bool = False
    pr_number: int | None = None
    pr_url: str | None = None
    dd_gap_summary: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dry_run": self.dry_run,
            "rc_version": self.rc_version,
            "batch_label": self.batch_label,
            "batch_size": self.batch_size,
            "unmatched_sources": self.unmatched_sources,
            "manifest_size": self.manifest_size,
            "source_reconciliation": self.source_reconciliation,
            "artifact_path": self.artifact_path,
            "commit_sha": self.commit_sha,
            "branch": self.branch,
            "branch_reclaimed_from": self.branch_reclaimed_from,
            "remote": self.remote,
            "pushed": self.pushed,
            "tag_created": self.tag_created,
            "tag_pushed": self.tag_pushed,
            "pr_number": self.pr_number,
            "pr_url": self.pr_url,
            "dd_gap_summary": self.dd_gap_summary,
            "errors": self.errors,
        }


def default_reviews_dir() -> Path:
    """The committed home for frozen review-batch artifacts (imas-codex repo)."""
    return Path(__file__).parent / "manifests" / "reviews"


def _slug_from_rc(rc_version: str) -> str:
    """Kebab-case batch name derived from an RC version tag (e.g. v0.2.0rc65)."""
    body = re.sub(r"[^a-z0-9]+", "-", rc_version.lower()).strip("-")
    return f"review-{body}"


def _review_commit_message(
    message: str,
    *,
    batch_label: str,
    published_count: int,
    withheld_count: int,
    rc_version: str,
) -> tuple[str, str]:
    """Compose concise, deterministic prose for a review-branch commit."""
    operator_phrase = " ".join(message.split())
    subject = f"sn: add {operator_phrase.rstrip('.')}"
    body = (
        f"Published {published_count} entries for the {batch_label} review batch "
        f"and withheld {withheld_count} from this release.\n"
        f"The frozen review cohort is identified by {rc_version}."
    )
    return subject, body


def _freeze_review_artifact(
    reviews_dir: Path,
    *,
    rc_version: str,
    names: list[str],
    minted_from: str,
    unmatched: list[str],
    manifest_sources: list[dict[str, Any]] | None = None,
    batch_label: str | None = None,
) -> Path:
    """Materialise the frozen sn-names batch record, tagged by the RC version.

    The artifact is the reproducible batch identity carried through export → PR
    → merge; ``pr_number``/``pr_url``/``merge_commit`` are written null here and
    back-filled once the PR exists.
    """
    from datetime import UTC, datetime

    import yaml

    reviews_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "kind": "sn_names",
        "schema_version": 1,
        "name": _slug_from_rc(rc_version),
        "rc_version": rc_version,
        "batch_label": batch_label or _parse_build(rc_version),
        "minted_from": minted_from,
        "minted_at": datetime.now(UTC).isoformat(),
        "names": sorted(names),
        "unmatched_sources": sorted(unmatched),
        "pr_number": None,
        "pr_url": None,
        "merge_commit": None,
    }
    if manifest_sources is not None:
        doc["manifest_sources"] = sorted(
            manifest_sources, key=lambda row: row["source_path"]
        )
    path = reviews_dir / f"{rc_version}.sn_names.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    from imas_codex.standard_names.sources_manifest import load_names_file

    load_names_file(path)
    return path


def backfill_review_artifact(
    path: Path,
    *,
    pr_number: int | None = None,
    pr_url: str | None = None,
    merge_commit: str | None = None,
) -> None:
    """Write PR provenance into a frozen artifact as it becomes known.

    The PR number/URL land after ``gh pr create``; the merge commit lands when
    ``sn approve`` folds the merged PR back. Only the provided fields are written.
    """
    import yaml

    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    if pr_number is not None:
        doc["pr_number"] = pr_number
    if pr_url is not None:
        doc["pr_url"] = pr_url
    if merge_commit is not None:
        doc["merge_commit"] = merge_commit
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")


def _gh_pr_create(
    *, branch: str, base: str, title: str, body: str, repo: str, head_owner: str
) -> tuple[int | None, str | None]:
    """Open a PR via the gh CLI; return (pr_number, pr_url).

    Injected as ``pr_creator`` in tests so no live GitHub call is made.
    """
    result = subprocess.run(
        [
            "gh",
            "pr",
            "create",
            "--repo",
            repo,
            "--base",
            base,
            "--head",
            f"{head_owner}:{branch}",
            "--title",
            title,
            "--body",
            body,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gh pr create failed: {result.stderr.strip()}")
    url = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else None
    number = None
    if url and "/pull/" in url:
        try:
            number = int(url.rsplit("/", 1)[-1])
        except ValueError:
            number = None
    return number, url


def _assert_approved_entries_unchanged(
    isnc_path: Path,
    staging_dir: Path,
    *,
    batch_names: list[str],
) -> None:
    """Fail when an additive review export changes its approved baseline."""
    from imas_codex.standard_names.export import approved_baseline_delta

    delta = approved_baseline_delta(
        isnc_path,
        staging_dir,
        batch_names=batch_names,
    )
    if delta.missing or delta.byte_changed:
        details = []
        if delta.missing:
            details.append(f"missing={list(delta.missing)}")
        if delta.byte_changed:
            details.append(f"byte_changed={list(delta.byte_changed)}")
        raise ValueError(
            "approved catalog baseline changed during additive review export: "
            + "; ".join(details)
        )


@_restore_main_after_review_release
def run_review_release(
    isnc_path: Path,
    focus_file: str | Path,
    message: str,
    *,
    staging_dir: Path | None = None,
    bump: str | None = None,
    remote: str | None = None,
    dry_run: bool = False,
    export_kwargs: dict[str, Any] | None = None,
    reviews_dir: Path | None = None,
    gc: object | None = None,
    exporter: Any | None = None,
    publisher: Any | None = None,
    pr_creator: Any | None = None,
    github_client: Any | None = None,
    upstream_repo: str | None = None,
    fork_owner: str | None = None,
    pr_target: str = "upstream",
    notes_builder: Any | None = None,
    llm_notes: bool = False,
    dd_gap_reader: Any | None = None,
    open_pr: bool = True,
    pr_title: str | None = None,
    pr_body: str | None = None,
    pr_head_reader: Any | None = None,
) -> ReviewReleaseReport:
    """Mint → freeze → export → branch → tag → optional PR, in one call.

    A single orchestrating step so the frozen sn-names artifact, the pushed RC
    catalog, and the PR stay in lock-step. The focus file drives the batch: an
    sn-sources file is minted live to the SN set (:func:`mint_sn_list`), an
    sn-names file is used directly (schema dispatch). The resolved set is frozen
    under ``manifests/reviews/<rc>.sn_names.yaml`` (RC-tagged, the traceable key)
    and the PR number/URL back-filled after ``gh pr create`` when ``open_pr``
    is true. The RC tag is always created and pushed to the fork at cut time.

    The RC version carries the batch identity as semver build metadata
    (``v0.2.0rc66+<label>``, derived from the manifest — see
    :func:`batch_build_metadata`), and that one string names the tag, the
    ``review/<rc>`` branch, and the frozen artifact, so a batch RC is
    recognisable as such from its version alone.

    ``exporter``/``publisher``/``pr_creator``/``github_client`` are injectable
    so the flow is testable against a local bare repo with no live GitHub call.
    """
    from imas_codex.standard_names.minting import mint_sn_list
    from imas_codex.standard_names.sources_manifest import load_focus_file

    report = ReviewReleaseReport(dry_run=dry_run)
    isnc_path = Path(isnc_path)
    reviews_dir = reviews_dir or default_reviews_dir()
    if (pr_title is None) != (pr_body is None):
        report.errors.append("PR title and body must be supplied together")
        return report
    if pr_title is not None and not open_pr:
        report.errors.append(
            "PR title and body cannot be used with PR creation disabled"
        )
        return report
    if pr_title is not None and pr_body is not None:
        from imas_codex.standard_names.release_notes import validate_pr_text

        try:
            validate_pr_text(pr_title, pr_body)
        except ValueError as exc:
            report.errors.append(f"PR text validation failed: {exc}")
            return report
    if staging_dir is None:
        from imas_codex.settings import get_sn_staging_dir

        staging_dir = get_sn_staging_dir()
    staging_dir = Path(staging_dir)
    effective_remote = _RC_REMOTE
    report.remote = effective_remote
    if remote and remote != _RC_REMOTE:
        logger.warning(
            "Ignoring review-branch transport override %r; branches always "
            "push to the fork remote %r",
            remote,
            _RC_REMOTE,
        )

    exporter = exporter or _default_exporter
    publisher = publisher or _default_publisher
    if github_client is None and pr_creator is None:
        github_client = _GitHubClient()
    pr_creator = pr_creator or github_client.create_pull_request

    # ── 1. Resolve the focus file to the batch SN set ──────────────────
    try:
        kind, items = load_focus_file(focus_file)
    except Exception as exc:
        report.errors.append(f"focus file: {exc}")
        return report

    manifest_sources: list[dict[str, Any]] | None = None
    if kind == "sn_sources":
        mint = mint_sn_list(items, gc=gc)
        from imas_codex.standard_names.graph_ops import (
            fetch_manifest_source_release_rows,
        )

        manifest_sources = fetch_manifest_source_release_rows(items, gc=gc)
        terminal_ids = [
            row["standard_name_id"]
            for row in manifest_sources
            if row.get("standard_name_id")
        ]
        names = list(dict.fromkeys([*mint.names, *terminal_ids]))
        unmatched = [
            row["source_path"]
            for row in manifest_sources
            if not row.get("standard_name_id")
        ]
    else:
        names, unmatched = list(dict.fromkeys(items)), []

    if not names:
        report.errors.append("focus resolved to zero standard names")
        return report
    report.names = sorted(names)
    report.batch_size = len(report.names)
    report.unmatched_sources = sorted(unmatched)
    report.manifest_size = len(items) if kind == "sn_sources" else 0

    # DD-gap lifecycle evidence is a read-only release caveat, never an export
    # gate. The canonical reader owns graph queries and exact batch-name
    # filtering; this orchestrator only normalizes its projection for reports.
    from imas_codex.standard_names.release_notes import (
        summarize_dd_gap_facts,
        unavailable_dd_gap_summary,
    )

    if dd_gap_reader is None:
        try:
            from imas_codex.standard_names.dd_gaps import list_dd_gaps

            dd_gap_reader = list_dd_gaps
        except Exception as exc:
            report.dd_gap_summary = unavailable_dd_gap_summary(str(exc))
    if dd_gap_reader is not None:
        try:
            gap_facts = dd_gap_reader(name_ids=report.names, gc=gc)
            report.dd_gap_summary = summarize_dd_gap_facts(gap_facts)
        except Exception as exc:
            logger.warning(
                "DD-gap release evidence could not be read; reporting the "
                "unavailable state without blocking the release",
                exc_info=True,
            )
            report.dd_gap_summary = unavailable_dd_gap_summary(str(exc))

    # ── 2. Pre-flight ISNC + compute the RC version ────────────────────
    try:
        _check_on_main(isnc_path)
        _check_clean_tree(isnc_path)
    except ValueError as exc:
        report.errors.append(str(exc))
        return report
    _run_git("fetch", "--tags", effective_remote, cwd=isnc_path)
    # The batch's identity rides the version as semver build metadata, so a
    # batch RC is distinguishable from a full candidate release by its version
    # alone — and the same string names the branch, the tag, and the artifact.
    try:
        batch_label = batch_build_metadata(focus_file)
    except ValueError as exc:
        report.errors.append(f"batch label: {exc}")
        return report
    try:
        git_tag, _version = compute_next_version(
            isnc_path, bump, final=False, build=batch_label
        )
    except ValueError as exc:
        report.errors.append(str(exc))
        return report
    report.rc_version = git_tag
    report.batch_label = batch_label
    report.branch = f"review/{git_tag}"

    # ── 3. Freeze the batch artifact (pre-PR fields) ───────────────────
    artifact = _freeze_review_artifact(
        reviews_dir,
        rc_version=git_tag,
        names=report.names,
        minted_from=str(focus_file),
        unmatched=report.unmatched_sources,
        manifest_sources=manifest_sources,
        batch_label=batch_label,
    )
    report.artifact_path = str(artifact)

    # ── 4. Export approved ∪ batch (review_batch stamped) ──────────────
    staging_dir.mkdir(parents=True, exist_ok=True)
    try:
        export_report = exporter(
            staging_dir=staging_dir,
            force=True,
            review_batch=report.names,
            manifest_sources=manifest_sources,
            **(export_kwargs or {}),
        )
        if manifest_sources is not None and hasattr(export_report, "to_dict"):
            report.source_reconciliation = export_report.to_dict().get(
                "source_reconciliation", {}
            )
    except Exception as exc:
        report.errors.append(f"export failed: {exc}")
        return report
    try:
        from imas_codex.standard_names.export import assemble_review_catalog

        assembly = assemble_review_catalog(
            isnc_path,
            staging_dir,
            batch_names=report.names,
        )
        _assert_approved_entries_unchanged(
            isnc_path,
            staging_dir,
            batch_names=list(assembly.emitted_batch_names),
        )
    except Exception as exc:
        report.errors.append(f"approved baseline check failed: {exc}")
        return report

    if dry_run:
        logger.info(
            "[dry-run] would branch %s, publish, and tag on %s%s",
            report.branch,
            effective_remote,
            ", then open a PR" if open_pr else " without opening a PR",
        )
        return report

    # ── 5. Branch, publish (copy + commit), push to the fork ───────────
    reclaimed_from, branch_error = _prepare_release_branch(
        isnc_path,
        report.branch,
        pr_head_reader=pr_head_reader,
    )
    if branch_error:
        report.errors.append(branch_error)
        return report
    report.branch_reclaimed_from = reclaimed_from
    try:
        pub = publisher(
            staging_dir=str(staging_dir),
            isnc_path=str(isnc_path),
            push=False,
            allow_dirty=True,
        )
    except Exception as exc:
        report.errors.append(f"publish failed: {exc}")
        return report
    if getattr(pub, "errors", None):
        report.errors.extend(pub.errors)
        return report

    if getattr(pub, "commit_sha", None):
        subject, body = _review_commit_message(
            message,
            batch_label=batch_label,
            published_count=assembly.batch_entries_written,
            withheld_count=report.batch_size - assembly.batch_entries_written,
            rc_version=git_tag,
        )
        amended = _run_git(
            "commit",
            "--amend",
            "-m",
            subject,
            "-m",
            body,
            cwd=isnc_path,
        )
        if amended.returncode != 0:
            report.errors.append(
                f"failed to compose review commit message: {amended.stderr}"
            )
            return report
        report.commit_sha = _run_git("rev-parse", "HEAD", cwd=isnc_path).stdout.strip()

    push = _run_git("push", effective_remote, report.branch, cwd=isnc_path)
    if push.returncode != 0:
        report.errors.append(f"failed to push {report.branch}: {push.stderr}")
        return report
    report.pushed = True

    # A candidate becomes a historic build identity when it is cut. Approval
    # may later replace this ref with the graph-fold receipt; undo restores the
    # exact annotated tag object created here.
    tag = _run_git("tag", "-a", git_tag, "-m", message, cwd=isnc_path)
    if tag.returncode != 0:
        report.errors.append(f"failed to create RC tag {git_tag}: {tag.stderr}")
        return report
    report.tag_created = True
    tag_push = _run_git("push", effective_remote, f"refs/tags/{git_tag}", cwd=isnc_path)
    if tag_push.returncode != 0:
        _run_git("tag", "-d", git_tag, cwd=isnc_path)
        report.tag_created = False
        report.errors.append(
            f"failed to push RC tag {git_tag} to {effective_remote}: {tag_push.stderr}"
        )
        return report
    report.tag_pushed = True

    if not open_pr:
        return report

    # ── 6. Open the PR and back-fill the artifact ──────────────────────
    # The PR repo and fork owner are derived from the ISNC checkout's own
    # remotes — never hardcoded — so the tool follows whatever catalog repo
    # the checkout actually tracks. pr_target='fork' raises the PR within the
    # fork itself (origin) — the full gh review/merge flow with no upstream
    # noise (rehearsals); 'upstream' (default) targets the real catalog.
    if upstream_repo is None:
        if pr_target == "fork":
            slug = _github_slug(isnc_path, "origin")
        else:
            slug = _github_slug(isnc_path, "upstream") or _github_slug(
                isnc_path, "origin"
            )
        if slug is None:
            report.errors.append(
                f"cannot derive the PR target repo (pr_target={pr_target}): the "
                "ISNC checkout has no matching github remote — pass upstream_repo"
            )
            return report
        upstream_repo = f"{slug[0]}/{slug[1]}"
    if fork_owner is None:
        slug = _github_slug(isnc_path, "origin")
        if slug is None:
            report.errors.append(
                "cannot derive the fork owner: the ISNC checkout has no github "
                "'origin' remote — pass fork_owner"
            )
            return report
        fork_owner = slug[0]

    # PR description: grounded LLM synthesis (release message + batch record +
    # per-domain catalog diff) when enabled; deterministic static body
    # otherwise. notes_builder is injectable for tests; the LLM path never
    # raises (it falls back to the static form internally).
    from imas_codex.standard_names.release_notes import (
        build_pr_notes,
        collect_catalog_changes,
        static_pr_notes,
    )

    if pr_title is not None and pr_body is not None:
        title, body = pr_title, pr_body
    elif notes_builder is None and llm_notes:
        notes_builder = build_pr_notes
    if pr_title is not None and pr_body is not None:
        pass
    elif notes_builder is not None:
        changes = collect_catalog_changes(isnc_path, base_ref="main")
        title, body = notes_builder(
            message=message,
            rc_version=git_tag,
            batch_size=report.batch_size,
            minted_from=str(focus_file),
            unmatched_count=len(report.unmatched_sources),
            changes=changes,
            dd_gaps=report.dd_gap_summary,
        )
    else:
        title, body = static_pr_notes(
            message=message,
            rc_version=git_tag,
            batch_size=report.batch_size,
            minted_from=str(focus_file),
            dd_gaps=report.dd_gap_summary,
        )
    try:
        pr_number, pr_url = pr_creator(
            branch=report.branch,
            base="main",
            title=title,
            body=body,
            repo=upstream_repo,
            head_owner=fork_owner,
        )
    except Exception as exc:
        report.errors.append(f"gh pr create failed: {exc}")
        return report
    report.pr_number = pr_number
    report.pr_url = pr_url
    target_owner = upstream_repo.split("/", 1)[0]
    if target_owner.casefold() == fork_owner.casefold():
        if pr_number is None:
            report.errors.append(
                "ReviewPreviewLinkInvariantError: GitHub did not return a PR "
                "number, so the preview address cannot be constructed"
            )
            return report
        if github_client is not None:
            try:
                _write_and_verify_review_preview_link(
                    github_client,
                    repo=upstream_repo,
                    pr_number=pr_number,
                    body=body,
                )
            except ReviewPreviewLinkInvariantError as exc:
                report.errors.append(f"{type(exc).__name__}: {exc}")
                return report
    backfill_review_artifact(artifact, pr_number=pr_number, pr_url=pr_url)

    return report


def _default_exporter(**kwargs: Any) -> Any:
    from imas_codex.standard_names.export import run_export

    return run_export(**kwargs)


def _default_publisher(**kwargs: Any) -> Any:
    from imas_codex.standard_names.publish import run_publish

    return run_publish(**kwargs)
