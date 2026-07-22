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
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Only match clean semver tags (ignore legacy suffixed tags like v0.3.0-rc1-w40-corpus)
_SEMVER_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)(?:rc(\d+))?$")

_RC_REMOTE = "origin"
_FINAL_REMOTE = "upstream"


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


def _format_tag(major: int, minor: int, patch: int, rc: int | None) -> str:
    """Format version components as a git tag (v1.0.0 or v1.0.0rc1)."""
    base = f"v{major}.{minor}.{patch}"
    return f"{base}rc{rc}" if rc else base


def _parse_version(tag: str) -> tuple[int, int, int, int | None]:
    """Parse a version tag into (major, minor, patch, rc_number|None).

    Handles: v1.0.0, v1.0.0rc1
    """
    match = _SEMVER_RE.match(tag)
    if not match:
        raise ValueError(f"Cannot parse version tag: {tag}")
    major, minor, patch = int(match[1]), int(match[2]), int(match[3])
    rc = int(match[4]) if match[4] else None
    return major, minor, patch, rc


def _tag_exists(tag: str, *, cwd: Path | None = None) -> bool:
    result = _run_git("tag", "-l", tag, cwd=cwd)
    return bool(result.stdout.strip())


def _commits_since_tag(tag: str, *, cwd: Path | None = None) -> int:
    result = _run_git("rev-list", f"{tag}..HEAD", "--count", cwd=cwd)
    return int(result.stdout.strip()) if result.returncode == 0 else 0


# =============================================================================
# State detection
# =============================================================================


def _get_semver_tags(cwd: Path | None = None) -> list[str]:
    """Get all clean semver tags, sorted by version (descending)."""
    result = _run_git("tag", "--sort=-v:refname", cwd=cwd)
    if result.returncode != 0:
        return []
    return [
        tag.strip()
        for tag in result.stdout.strip().splitlines()
        if _SEMVER_RE.match(tag.strip())
    ]


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
    Dict with keys: state, tag, major, minor, patch, rc, commits_since.
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
) -> tuple[str, str]:
    """Compute next version tag from current ISNC state.

    Returns (git_tag, version_string) e.g. ("v1.0.0rc1", "1.0.0rc1").

    Raises
    ------
    ValueError
        If on stable and no bump specified, or other invalid transitions.
    """
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
        tag = _format_tag(m, n, p, rc)
        return tag, tag.lstrip("v")

    if state == "stable":
        if not bump:
            raise ValueError(
                f"On stable release {info['tag']}. "
                "Specify --bump (major|minor|patch) to start a new release."
            )
        m, n, p = _apply_bump(major, minor, patch, bump)
        rc = None if final else 1
        tag = _format_tag(m, n, p, rc)
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
        tag = _format_tag(m, n, p, rc)
        return tag, tag.lstrip("v")

    if final:
        # Finalize: v1.0.0rc2 → v1.0.0
        tag = _format_tag(major, minor, patch, None)
        return tag, tag.lstrip("v")

    # Increment RC: v1.0.0rc1 → v1.0.0rc2
    next_rc = info["rc"] + 1
    tag = _format_tag(major, minor, patch, next_rc)
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
    effective_remote = remote or (_FINAL_REMOTE if final else _RC_REMOTE)
    report.remote = effective_remote

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
        warnings = _check_synced(isnc_path, effective_remote, strict=not dry_run)
        for w in warnings:
            logger.warning(w)
    except ValueError as exc:
        report.errors.append(str(exc))
        return report

    # ── Compute version ────────────────────────────────────
    # Fetch tags from remote before computing version
    _run_git("fetch", "--tags", effective_remote, cwd=isnc_path)

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
        from imas_codex.standard_names.export import run_export

        staging_dir.mkdir(parents=True, exist_ok=True)

        kwargs: dict[str, Any] = {
            "staging_dir": staging_dir,
            "force": True,  # Overwrite existing staging
            "final": final,  # Strict gates for final releases
            **(export_kwargs or {}),
        }

        logger.info("Exporting to %s", staging_dir)
        try:
            export_report = run_export(**kwargs)
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

    # ── Publish (copy staging → ISNC) ─────────────────────
    from imas_codex.standard_names.publish import run_publish

    logger.info("Publishing to %s", isnc_path)
    pub_report = run_publish(
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
        logger.info(
            "[dry-run] Would tag %s and push to %s",
            git_tag,
            effective_remote,
        )
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
    batch_size: int = 0
    names: list[str] = field(default_factory=list)
    unmatched_sources: list[str] = field(default_factory=list)
    artifact_path: str | None = None
    branch: str = ""
    remote: str = ""
    pushed: bool = False
    pr_number: int | None = None
    pr_url: str | None = None
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dry_run": self.dry_run,
            "rc_version": self.rc_version,
            "batch_size": self.batch_size,
            "unmatched_sources": self.unmatched_sources,
            "artifact_path": self.artifact_path,
            "branch": self.branch,
            "remote": self.remote,
            "pushed": self.pushed,
            "pr_number": self.pr_number,
            "pr_url": self.pr_url,
            "errors": self.errors,
        }


def default_reviews_dir() -> Path:
    """The committed home for frozen review-batch artifacts (imas-codex repo)."""
    return Path(__file__).parent / "manifests" / "reviews"


def _slug_from_rc(rc_version: str) -> str:
    """Kebab-case batch name derived from an RC version tag (e.g. v0.2.0rc65)."""
    body = re.sub(r"[^a-z0-9]+", "-", rc_version.lower()).strip("-")
    return f"review-{body}"


def _freeze_review_artifact(
    reviews_dir: Path,
    *,
    rc_version: str,
    names: list[str],
    minted_from: str,
    unmatched: list[str],
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
        "minted_from": minted_from,
        "minted_at": datetime.now(UTC).isoformat(),
        "names": sorted(names),
        "unmatched_sources": sorted(unmatched),
        "pr_number": None,
        "pr_url": None,
        "merge_commit": None,
    }
    path = reviews_dir / f"{rc_version}.sn_names.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
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
    ``sn merge`` folds the merged PR back. Only the provided fields are written.
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
    upstream_repo: str | None = None,
    fork_owner: str | None = None,
    pr_target: str = "upstream",
) -> ReviewReleaseReport:
    """Mint → freeze → export → branch → push → PR → back-fill, in one call.

    A single orchestrating step so the frozen sn-names artifact, the pushed RC
    catalog, and the PR stay in lock-step. The focus file drives the batch: an
    sn-sources file is minted live to the SN set (:func:`mint_sn_list`), an
    sn-names file is used directly (schema dispatch). The resolved set is frozen
    under ``manifests/reviews/<rc>.sn_names.yaml`` (RC-tagged, the traceable key)
    and the PR number/URL back-filled after ``gh pr create``.

    ``exporter``/``publisher``/``pr_creator`` are injectable so the flow is
    testable against a local bare repo with no live GitHub call.
    """
    from imas_codex.standard_names.minting import mint_sn_list
    from imas_codex.standard_names.sources_manifest import load_focus_file

    report = ReviewReleaseReport(dry_run=dry_run)
    isnc_path = Path(isnc_path)
    reviews_dir = reviews_dir or default_reviews_dir()
    if staging_dir is None:
        from imas_codex.settings import get_sn_staging_dir

        staging_dir = get_sn_staging_dir()
    staging_dir = Path(staging_dir)
    effective_remote = remote or _RC_REMOTE
    report.remote = effective_remote

    exporter = exporter or _default_exporter
    publisher = publisher or _default_publisher
    pr_creator = pr_creator or _gh_pr_create

    # ── 1. Resolve the focus file to the batch SN set ──────────────────
    try:
        kind, items = load_focus_file(focus_file)
    except Exception as exc:
        report.errors.append(f"focus file: {exc}")
        return report

    if kind == "sn_sources":
        mint = mint_sn_list(items, gc=gc)
        names, unmatched = mint.names, mint.unmatched_paths
    else:
        names, unmatched = list(dict.fromkeys(items)), []

    if not names:
        report.errors.append("focus resolved to zero standard names")
        return report
    report.names = sorted(names)
    report.batch_size = len(report.names)
    report.unmatched_sources = sorted(unmatched)

    # ── 2. Pre-flight ISNC + compute the RC version ────────────────────
    try:
        _check_on_main(isnc_path)
    except ValueError as exc:
        report.errors.append(str(exc))
        return report
    _run_git("fetch", "--tags", effective_remote, cwd=isnc_path)
    try:
        git_tag, _version = compute_next_version(isnc_path, bump, final=False)
    except ValueError as exc:
        report.errors.append(str(exc))
        return report
    report.rc_version = git_tag
    report.branch = f"review/{git_tag}"

    # ── 3. Freeze the batch artifact (pre-PR fields) ───────────────────
    artifact = _freeze_review_artifact(
        reviews_dir,
        rc_version=git_tag,
        names=report.names,
        minted_from=str(focus_file),
        unmatched=report.unmatched_sources,
    )
    report.artifact_path = str(artifact)

    # ── 4. Export approved ∪ batch (review_batch stamped) ──────────────
    staging_dir.mkdir(parents=True, exist_ok=True)
    try:
        exporter(
            staging_dir=staging_dir,
            force=True,
            review_batch=report.names,
            **(export_kwargs or {}),
        )
    except Exception as exc:
        report.errors.append(f"export failed: {exc}")
        return report

    if dry_run:
        logger.info(
            "[dry-run] would branch %s, publish, push to %s, and open a PR",
            report.branch,
            effective_remote,
        )
        return report

    # ── 5. Branch, publish (copy + commit), push to the fork ───────────
    br = _run_git("checkout", "-b", report.branch, cwd=isnc_path)
    if br.returncode != 0:
        report.errors.append(f"failed to create branch {report.branch}: {br.stderr}")
        return report
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

    push = _run_git("push", effective_remote, report.branch, cwd=isnc_path)
    if push.returncode != 0:
        report.errors.append(f"failed to push {report.branch}: {push.stderr}")
        return report
    report.pushed = True

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

    title = message or f"Standard-name review batch {git_tag}"
    body = (
        f"Review batch **{git_tag}** — {report.batch_size} standard name(s) for "
        f"first human review.\n\nMinted from `{focus_file}`."
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
    backfill_review_artifact(artifact, pr_number=pr_number, pr_url=pr_url)

    return report


def _default_exporter(**kwargs: Any) -> Any:
    from imas_codex.standard_names.export import run_export

    return run_export(**kwargs)


def _default_publisher(**kwargs: Any) -> Any:
    from imas_codex.standard_names.publish import run_publish

    return run_publish(**kwargs)
