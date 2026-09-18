"""Live-graph regression test for ``_list_physics_domains_with_extractable_paths``.

Caught a real production bug: the Cypher used ``(IDS)-[:HAS_PATH*]->(IMASNode)``
but the actual schema relationship is ``(IMASNode)-[:IN_IDS]->(IDS)``. Result: 0
domains seeded after ``sn clear``, and a smoke run exited with 0 SNs created.

Mocked unit tests in ``test_seed_all_domains.py`` did not catch this because
they stub the function; this test hits the real graph.

Marked ``requires_graph`` — only runs when a Neo4j with DD content is available.
Both tests bind the verdict of :func:`_probe_live_graph`, which is evaluated once
at collection time. That verdict is bounded by ``PROBE_TIMEOUT_SECONDS`` and names
which condition it met — no route, no credential, or a reachable-but-empty graph —
so a skip receipt says why the tests did not run instead of reporting one string
for three different faults. The classification is exercised by
``test_seed_live_graph_probe.py``, which injects each condition against a stub
client rather than needing a broken graph to hand.
"""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest

from imas_codex.graph.client import GraphClient

pytestmark = pytest.mark.requires_graph

#: Longest a collection-time probe may wait for the graph to answer. A live-graph
#: read that exceeds this is reported as no route rather than awaited: pytest
#: cannot start until the probe returns, so an unbounded probe lets one unreachable
#: host hold a whole session open before the first test is collected.
PROBE_TIMEOUT_SECONDS = 10.0

# Verdicts. Exactly one is returned per probe, and the reason text names it.
READY = "ready"
NO_ROUTE = "no-route"
NO_CREDENTIAL = "no-credential"
EMPTY = "empty"
PROBE_FAILED = "probe-failed"

#: Existence, not cardinality: the probe asks whether any extractable IMASNode
#: exists. A ``count`` over the same predicate scans every matching node, which is
#: what made the collection-time probe cost minutes on the login node.
_DD_CONTENT_QUERY = (
    "MATCH (n:IMASNode) WHERE n.physics_domain IS NOT NULL RETURN n.id AS id LIMIT 1"
)

_UNREACHABLE_TOKENS = (
    "connection refused",
    "connection reset",
    "defunct connection",
    "socketdeadlineexceeded",
    "timed out",
    "timeout",
    "unable to retrieve routing information",
    "unreachable",
)


@dataclass(frozen=True)
class LiveGraphProbe:
    """What the collection-time probe found, and which condition it met."""

    verdict: str
    reason: str

    @property
    def dd_content_available(self) -> bool:
        return self.verdict == READY


def _classify_failure(exc: Exception) -> LiveGraphProbe:
    """Name the fault the probe hit instead of reporting that it hit one."""
    from neo4j.exceptions import AuthError, ServiceUnavailable

    detail = str(exc).strip() or type(exc).__name__
    lowered = detail.lower()
    if isinstance(exc, AuthError) or "unauthorized" in lowered:
        return LiveGraphProbe(
            NO_CREDENTIAL,
            f"no credential: the live graph rejected the connection ({detail})",
        )
    if isinstance(exc, ServiceUnavailable | ConnectionError | OSError) or any(
        token in lowered for token in _UNREACHABLE_TOKENS
    ):
        return LiveGraphProbe(
            NO_ROUTE,
            f"no route: the live graph did not answer within "
            f"{PROBE_TIMEOUT_SECONDS:g}s ({detail})",
        )
    return LiveGraphProbe(
        PROBE_FAILED,
        f"probe failed: {type(exc).__name__}: {detail}",
    )


_CONFTEST_PATH = Path(__file__).resolve().parents[1] / "conftest.py"


def _loaded_conftest():
    """The root conftest module object, when a pytest session loaded it."""
    for module in list(sys.modules.values()):
        module_file = getattr(module, "__file__", None)
        if module_file and Path(module_file).resolve() == _CONFTEST_PATH:
            return module
    return None


def _session_graph_gate() -> tuple[str | None, str]:
    """The session's own verdict on the route, reused rather than re-derived.

    The root conftest already answers the two questions this probe has about
    reaching the graph, and it answers them once per session: whether a
    credential is configured at all, and whether the host responds. It decides
    the credential question without constructing a client, because
    authenticating with the packaged placeholder against the project graph is a
    failed login that trips the server's own failed-attempt limiter, and it runs
    its connection attempt on a daemon thread with a bounded join so an
    unreachable host cannot stall collection. Re-deriving either answer here
    would be a second source of truth for conditions the session already names.

    Returns a verdict only when the graph is not reachable; the caller still has
    to ask a reachable graph whether it holds DD content. ``(None, "")`` is the
    reachable case, and it is also what a module imported outside a pytest
    session sees, since no session gate is loaded to consult.
    """
    module = _loaded_conftest()
    if module is None or module._check_neo4j():
        return None, ""
    if not module._graph_credential_is_configured():
        return NO_CREDENTIAL, module._neo4j_unavailable_reason()
    return NO_ROUTE, module._neo4j_unavailable_reason()


def _probe_live_graph(timeout: float | None = None) -> LiveGraphProbe:
    """Probe for DD content, bounded by ``timeout`` (default the module bound).

    The read runs on a daemon worker so the bound holds even when the client
    blocks inside a socket read: the caller stops waiting, and the worker cannot
    hold interpreter shutdown open.
    """
    limit = PROBE_TIMEOUT_SECONDS if timeout is None else timeout
    gate_verdict, gate_reason = _session_graph_gate()
    if gate_verdict is not None:
        return LiveGraphProbe(gate_verdict, f"{gate_verdict}: {gate_reason}")

    outcome: list[LiveGraphProbe] = []

    def read() -> None:
        try:
            with GraphClient() as graph:
                rows = graph.query(_DD_CONTENT_QUERY)
                host = getattr(graph, "uri", None)
                if rows:
                    outcome.append(
                        LiveGraphProbe(
                            READY,
                            f"live graph at {host or 'the resolved route'} holds DD content",
                        )
                    )
                else:
                    outcome.append(
                        LiveGraphProbe(
                            EMPTY,
                            f"reachable-but-empty: the live graph at "
                            f"{host or 'the resolved route'} answered but holds no "
                            f"DD content",
                        )
                    )
        except Exception as exc:  # noqa: BLE001 — classified, never re-raised
            outcome.append(_classify_failure(exc))

    worker = threading.Thread(target=read, name="dd-content-probe", daemon=True)
    worker.start()
    worker.join(limit)
    if not outcome:
        return LiveGraphProbe(
            NO_ROUTE,
            f"no route: the live graph did not answer within {limit:g}s",
        )
    return outcome[0]


_PROBE = _probe_live_graph()

# One marker reused by both tests, so the module pays one probe per collection
# rather than one per decorated test.
_REQUIRES_DD_CONTENT = pytest.mark.skipif(
    not _PROBE.dd_content_available,
    reason=f"DD content unavailable ({_PROBE.reason})",
)


@_REQUIRES_DD_CONTENT
def test_seed_query_finds_real_domains():
    """The Cypher used by ``_list_physics_domains_with_extractable_paths`` must
    return at least 10 domains against a DD-loaded graph.

    A previous bug used ``HAS_PATH*`` and returned 0 rows, silently breaking
    auto-seed.
    """
    from imas_codex.standard_names.loop import (
        _list_physics_domains_with_extractable_paths,
    )

    domains = _list_physics_domains_with_extractable_paths("dd")

    # DD has ~30 physics domains; a working query must return many of them.
    assert len(domains) >= 10, (
        f"Expected ≥10 domains from DD-loaded graph, got {len(domains)}: {domains}. "
        "Likely the IDS↔IMASNode relationship pattern is wrong "
        "(canonical: (n:IMASNode)-[:IN_IDS]->(ids:IDS))."
    )

    # Sanity: well-known domains must be present.
    expected_subset = {"equilibrium", "transport", "magnetohydrodynamics"}
    missing = expected_subset - set(domains)
    assert not missing, f"Expected domains missing: {missing} (got {len(domains)})"


@_REQUIRES_DD_CONTENT
def test_seed_query_returns_empty_for_non_dd_source():
    from imas_codex.standard_names.loop import (
        _list_physics_domains_with_extractable_paths,
    )

    assert _list_physics_domains_with_extractable_paths("signals") == []
    assert _list_physics_domains_with_extractable_paths("manual") == []
