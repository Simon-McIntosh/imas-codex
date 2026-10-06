"""JT-60SA path-exclusion configuration.

The facility's absolute exclusions must be subtree prefixes, not basename
directories, and user home directories must not be a discovery root. A
basename entry never excludes a subtree (``should_exclude('/work/edas/x')``
matched nothing), so the absolute entries belong in ``path_prefixes``.
"""

from __future__ import annotations

from imas_codex.config.discovery_config import get_exclusion_config_for_facility
from imas_codex.discovery.base.facility import get_facility


def test_home_subtree_excluded() -> None:
    """A path under /home is excluded as a path prefix."""
    config = get_exclusion_config_for_facility("jt-60sa")
    should_exclude, reason = config.should_exclude("/home/u/x")
    assert should_exclude is True
    assert reason == "path_prefix:/home"


def test_work_edas_subtree_excluded() -> None:
    """A path under /work/edas is excluded as a path prefix."""
    config = get_exclusion_config_for_facility("jt-60sa")
    should_exclude, reason = config.should_exclude("/work/edas/x")
    assert should_exclude is True
    assert reason == "path_prefix:/work/edas"


def test_analysis_src_not_excluded() -> None:
    """An analysis source path is not excluded."""
    config = get_exclusion_config_for_facility("jt-60sa")
    should_exclude, _ = config.should_exclude("/analysis/src/x")
    assert should_exclude is False


def test_home_is_not_a_discovery_root() -> None:
    """User home directories are no longer a discovery root."""
    roots = get_facility("jt-60sa").get("discovery_roots") or []
    assert "/home" not in roots
