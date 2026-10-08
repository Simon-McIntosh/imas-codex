"""JT-60SA path-exclusion configuration.

The facility's absolute exclusions must be subtree prefixes, not basename
directories, and user home directories must not be a discovery root. A
basename entry never excludes a subtree (``should_exclude('/work/edas/x')``
matched nothing), so the absolute entries belong in ``path_prefixes``.
"""

from __future__ import annotations

from urllib.parse import urlsplit

import pytest

from imas_codex.config.discovery_config import get_exclusion_config_for_facility
from imas_codex.discovery.base.facility import get_facility

# Data-access directories that must each be seeded as their own discovery root.
# The parent /analysis/src is deliberately never named: a seed of the parent
# walks every unseeded child instead of only the intended subtrees.
DATA_ACCESS_ROOTS = (
    "/analysis/src/eddb",
    "/analysis/src/uddb",
    "/analysis/src/pmdb",
    "/analysis/src/mbdb",
    "/analysis/src/lcdbWrapper",
    "/analysis/src/safledd",
    "/analysis/src/edas_inf",
    "/analysis/lib/JT60SAEQDB",
    "/analysis/src/MagLine2022",
    "/analysis/src/offlineTCCS",
    "/analysis/src/offlineTCCS.MAIN",
    "/analysis/src/SAsetequ",
    "/analysis/src/setequ",
    "/analysis/src/SAselene",
    "/analysis/CCStable",
    "/analysis/SAdata",
    "/analysis/src/slice",
    "/analysis/src/adam3",
    "/analysis/src/adamwin-v4",
    "/analysis/src/OFMC",
    "/analysis/src/accome",
    "/analysis/src/toolc",
    "/analysis/src/gdlib",
    "/analysis/src/getseldata_v4.1",
)


def test_analysis_src_parent_is_not_a_discovery_root() -> None:
    """The bare parent is never seeded; only its subtrees are."""
    roots = get_facility("jt-60sa").get("discovery_roots") or []
    assert "/analysis/src" not in roots


@pytest.mark.parametrize("root", DATA_ACCESS_ROOTS)
def test_data_access_root_is_seeded_and_unexcluded(root: str) -> None:
    """Every data-access directory is a discovery root no exclude covers."""
    roots = get_facility("jt-60sa").get("discovery_roots") or []
    assert root in roots, f"{root} is not a discovery root"

    config = get_exclusion_config_for_facility("jt-60sa")
    covering = [p for p in config.path_prefixes if root.startswith(p)]
    assert not covering, f"{root} is covered by exclude prefix(es) {covering}"
    should_exclude, reason = config.should_exclude(f"{root}/child")
    assert should_exclude is False, f"{root}/child excluded by {reason}"


def test_home_subtree_not_excluded() -> None:
    """A path under /home remains available for targeted discovery."""
    config = get_exclusion_config_for_facility("jt-60sa")
    should_exclude, _ = config.should_exclude("/home/u/x")
    assert should_exclude is False


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


def test_browser_services_match_port_forwards() -> None:
    """Each browser URL uses the TLS port assigned to its named service."""
    facility = get_facility("jt-60sa")
    forwards = {item["name"]: item for item in facility["port_forwards"]["forwards"]}
    services = facility["browser"]["services"]

    assert {item["name"] for item in services} == set(forwards)
    for service in services:
        forward = forwards[service["name"]]
        url = urlsplit(service["url"])
        assert url.hostname == "localhost"
        assert url.port == forward["local_port"]
        assert url.scheme == forward["protocol"]

    assert {item["name"]: item["url"] for item in services} == {
        "twiki_code": "https://localhost:8800/wiki/WebHome.html",
        "server_docs": "https://localhost:8801/",
    }
