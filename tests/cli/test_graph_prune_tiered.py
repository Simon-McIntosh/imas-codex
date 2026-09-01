"""Tiered registry retention for graph package versions."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from click.testing import CliRunner

from imas_codex.cli.graph import registry


def _version(
    sequence: int,
    created_at: datetime,
    *tags: str,
) -> registry.RegistryVersion:
    return registry.RegistryVersion(
        id=1_000_000_000 - sequence,
        name=f"sha256:{sequence:064x}",
        created_at=created_at,
        tags=tags,
    )


def _registry_census() -> list[registry.RegistryVersion]:
    """Return 47 records with the release, test, and untagged live shape."""
    newest = datetime(2026, 9, 1, 12, tzinfo=UTC)
    versions: list[registry.RegistryVersion] = []

    release_days = [0, 3, 9, 15, 23, 35, 48, 70, 105, 160]
    for sequence, days in enumerate(release_days):
        tags = (f"v5.{9 - sequence}.0-rc1",)
        if sequence == 0:
            tags += ("latest",)
        versions.append(_version(sequence, newest - timedelta(days=days), *tags))

    for offset in range(8):
        sequence = 10 + offset
        versions.append(_version(sequence, newest - timedelta(days=11 + offset * 13)))

    versions.extend(
        [
            _version(18, newest - timedelta(days=1), "test-push"),
            _version(19, newest - timedelta(days=80), "test-manual-push"),
        ]
    )

    for offset in range(27):
        sequence = 20 + offset
        created_at = newest - timedelta(days=offset * 8 + 2)
        versions.append(
            _version(sequence, created_at, f"5.9.0.dev{900 - offset}-gabc-r1")
        )

    assert len(versions) == 47
    return versions


def test_tiered_selector_classifies_live_shaped_census() -> None:
    decisions = registry._select_tiered_retention(_registry_census())

    assert len(decisions) == 47
    assert sum(decision.tier == "weekly" for decision in decisions) == 4
    assert sum(decision.tier == "monthly" for decision in decisions) == 3
    assert sum(decision.tier == "delete-untagged" for decision in decisions) == 8
    assert sum(decision.tier == "delete-test" for decision in decisions) == 2

    release_decisions = [
        decision
        for decision in decisions
        if registry._is_release_version(decision.version)
    ]
    assert len(release_decisions) == 10
    assert all(decision.keep for decision in release_decisions)
    assert all(
        decision.keep for decision in decisions if "latest" in decision.version.tags
    )
    assert all(
        not decision.keep
        for decision in decisions
        if not decision.version.tags
        or any(tag.startswith("test-") for tag in decision.version.tags)
    )


def test_tiered_dry_run_lists_every_version_without_deleting(monkeypatch) -> None:
    versions = _registry_census()
    deleted_tags: list[str] = []
    deleted_ids: list[int] = []

    monkeypatch.setattr(registry, "get_git_info", lambda: {})
    monkeypatch.setattr(
        registry, "get_registry", lambda git_info, override: "ghcr.io/example"
    )
    monkeypatch.setattr(
        registry,
        "_list_registry_versions",
        lambda target, package, token: versions,
    )
    monkeypatch.setattr(
        registry,
        "_delete_tag",
        lambda target, tag, token, pkg_name: deleted_tags.append(tag) or True,
    )
    monkeypatch.setattr(
        registry,
        "_delete_untagged_version",
        lambda target, package, version_id, token: (
            deleted_ids.append(version_id) or True
        ),
    )

    result = CliRunner().invoke(registry.graph_prune, ["--dry-run"])

    assert result.exit_code == 0, result.output
    classified = [
        line
        for line in result.output.splitlines()
        if line.lstrip().startswith(("KEEP", "DELETE"))
    ]
    assert len(classified) == 47
    assert "dry-run — no changes made" in result.output
    assert deleted_tags == []
    assert deleted_ids == []


def test_tiered_apply_uses_tag_and_version_deletion_primitives(monkeypatch) -> None:
    now = datetime(2026, 9, 1, tzinfo=UTC)
    versions = [
        _version(1, now, "latest"),
        _version(2, now - timedelta(days=1), "test-push"),
        _version(3, now - timedelta(days=2)),
    ]
    deleted_tags: list[str] = []
    deleted_ids: list[int] = []

    monkeypatch.setattr(registry, "get_git_info", lambda: {})
    monkeypatch.setattr(
        registry, "get_registry", lambda git_info, override: "ghcr.io/example"
    )
    monkeypatch.setattr(
        registry,
        "_list_registry_versions",
        lambda target, package, token: versions,
    )
    monkeypatch.setattr(
        registry,
        "_delete_tag",
        lambda target, tag, token, pkg_name: deleted_tags.append(tag) or True,
    )
    monkeypatch.setattr(
        registry,
        "_delete_untagged_version",
        lambda target, package, version_id, token: (
            deleted_ids.append(version_id) or True
        ),
    )

    result = CliRunner().invoke(registry.graph_prune, ["--force"])

    assert result.exit_code == 0, result.output
    assert deleted_tags == ["test-push"]
    assert deleted_ids == [versions[2].id]
