"""
Test configuration and fixtures for the MCP-based architecture.
"""

import os
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _load_test_environment(dotenv_path=None) -> bool:
    """Load developer defaults without replacing explicit process values."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False
    return load_dotenv(dotenv_path=dotenv_path, override=False)


_load_test_environment()

from imas_codex.clusters.search import ClusterSearchResult  # noqa: E402
from imas_codex.embeddings.encoder import Encoder  # noqa: E402
from imas_codex.search.document_store import (  # noqa: E402
    Document,
    DocumentMetadata,
    DocumentStore,
)
from imas_codex.search.engines.base_engine import MockSearchEngine  # noqa: E402
from imas_codex.tools import Tools  # noqa: E402


def pytest_addoption(parser):
    parser.addoption(
        "--embedding-model",
        action="store",
        default=None,
        help="Embedding model to use for tests (default: from settings)",
    )


def pytest_configure(config):
    """Configure pytest."""
    pass


# ── Auto-skip graph/integration tests when Neo4j is unreachable ──────────
_neo4j_available: bool | None = None


def _neo4j_probe_client():
    """Build a client for the selected test endpoint without profile resolution."""
    from imas_codex.graph.client import GraphClient

    if test_uri := os.environ.get("IMAS_CODEX_TEST_NEO4J_URI"):
        return GraphClient(
            uri=test_uri,
            username=os.environ.get("NEO4J_USERNAME", "neo4j"),
            password=os.environ.get("NEO4J_PASSWORD", ""),
            graph_name="pytest-explicit-endpoint",
        )
    return GraphClient()


def _graph_credential_is_configured() -> bool:
    """Whether a real credential exists for the endpoint the probe would use.

    An explicit test endpoint is a deliberate opt-in and cannot reach the
    project graph, so it always qualifies — including an auth-disabled local
    server that legitimately has no password.

    Otherwise the probe would resolve the active profile, which for a remote
    location is the shared project graph.  A checkout that supplies no
    credential resolves to the packaged development placeholder, and
    authenticating against a shared server with a placeholder is a failed
    login: repeat it once per pytest session across a few sessions and the
    server's failed-attempt limiter locks the account out from under whoever
    is using it for real.  Resolution here is metadata-only —
    ``auto_tunnel=False`` is the documented side-effect-free path — so asking
    this question never opens a connection.
    """
    if os.environ.get("IMAS_CODEX_TEST_NEO4J_URI"):
        return True
    try:
        from imas_codex.graph.profiles import DEFAULT_PASSWORD, resolve_neo4j

        credential = resolve_neo4j(auto_tunnel=False).password or ""
    except Exception:
        return False
    return bool(credential) and credential != DEFAULT_PASSWORD


def _neo4j_unavailable_reason() -> str:
    """Describe the selected graph endpoint without triggering resolution."""
    if test_uri := os.environ.get("IMAS_CODEX_TEST_NEO4J_URI"):
        return f"Explicit Neo4j test endpoint is not available: {test_uri}"
    if not _graph_credential_is_configured():
        return (
            "No Neo4j credential is configured for this checkout, so the "
            "reachability probe was skipped rather than authenticating with "
            "the packaged placeholder against the project graph"
        )
    return "Configured project Neo4j graph is not available"


def _check_neo4j() -> bool:
    """Probe whether Neo4j is reachable (cached per session).

    A checkout with no configured credential reports unreachable without
    connecting at all — see :func:`_graph_credential_is_configured`.

    Without an explicit test URI, constructing a ``GraphClient`` resolves the
    active graph profile, which for a remote location establishes an SSH
    tunnel. An explicit test URI bypasses that resolution entirely. When the
    selected host is unreachable, connection setup can block longer than the
    repo's ``faulthandler_timeout``, which would SIGSEGV the whole pytest
    session at collection time. To keep collection bounded regardless of
    reachability, the probe runs on a daemon thread with a bounded join: if it
    hasn't answered within the timeout the host is treated as unavailable and
    the still-blocked thread is abandoned (it dies with the process, so pytest
    can still exit and the main thread never stalls long enough to trip
    faulthandler). The generous default comfortably covers a cold-tunnel
    reachable case while staying well under any faulthandler timeout.
    """
    global _neo4j_available
    if _neo4j_available is not None:
        return _neo4j_available

    # Short-circuit BEFORE any client is built: with no credential the only
    # thing a probe can do is fail authentication against the project graph.
    if not _graph_credential_is_configured():
        _neo4j_available = False
        return _neo4j_available

    import threading

    timeout_s = float(os.environ.get("IMAS_CODEX_TEST_NEO4J_PROBE_TIMEOUT", "20"))
    result: dict[str, bool] = {}

    def _probe() -> None:
        try:
            client = _neo4j_probe_client()
            try:
                client.get_stats()
                result["ok"] = True
            finally:
                client.close()
        except Exception:
            result["ok"] = False

    thread = threading.Thread(target=_probe, name="neo4j-probe", daemon=True)
    thread.start()
    thread.join(timeout_s)

    _neo4j_available = bool(result.get("ok", False))
    return _neo4j_available


def _explicit_graph_marker_selection(config: pytest.Config | None) -> bool:
    """Whether the invocation explicitly selected only graph-marked tests."""
    if config is None:
        return False
    try:
        marker_expression = config.getoption("markexpr", default="")
    except (AttributeError, ValueError):
        return False
    return marker_expression.strip() == "graph"


def pytest_collection_modifyitems(config, items):  # noqa: ARG001
    """Auto-skip graph/integration-marked tests when Neo4j is unreachable."""
    graph_items = [
        item
        for item in items
        if (
            item.get_closest_marker("graph")
            or item.get_closest_marker("integration")
            or item.get_closest_marker("requires_graph")
        )
    ]
    if not graph_items or _check_neo4j():
        return
    unavailable_reason = _neo4j_unavailable_reason()
    if _explicit_graph_marker_selection(config):
        pytest.exit(
            f"Graph-marked test selection was not run: {unavailable_reason}",
            returncode=pytest.ExitCode.NO_TESTS_COLLECTED,
        )
    skip_marker = pytest.mark.skip(reason=unavailable_reason)
    for item in graph_items:
        item.add_marker(skip_marker)


@pytest.fixture(scope="session")
def embedding_model_name(request):
    """Get the embedding model name from command line option."""
    return request.config.getoption("--embedding-model")


@pytest.fixture(scope="session", autouse=True)
def configure_embedding_model(embedding_model_name):
    """Configure the embedding model environment variable.

    Forces local backend with all-MiniLM-L6-v2 for all tests unless
    explicitly overridden via --embedding-model CLI option.

    Uses direct assignment (not setdefault) to ensure deterministic test values
    regardless of developer defaults loaded from ``.env``.
    """
    os.environ["IMAS_CODEX_EMBEDDING_LOCATION"] = "local"
    if embedding_model_name:
        os.environ["IMAS_CODEX_EMBEDDING_MODEL"] = embedding_model_name
    else:
        os.environ["IMAS_CODEX_EMBEDDING_MODEL"] = "all-MiniLM-L6-v2"


# Standard test IDS set for consistency across all tests
# This avoids re-embedding and ensures consistent performance
STANDARD_TEST_IDS_SET = {"equilibrium", "core_profiles"}


def create_mock_document(path_id: str, ids_name: str = "core_profiles") -> Document:
    """Create a mock document for testing."""
    metadata = DocumentMetadata(
        path_id=path_id,
        ids_name=ids_name,
        path_name=path_id.split("/")[-1],
        units="m",
        data_type="float",
        coordinates=("rho_tor_norm",),
        physics_domain="transport",
        physics_phenomena=("transport", "plasma"),
    )

    return Document(
        metadata=metadata,
        documentation=f"Mock documentation for {path_id}",
        relationships={},
        raw_data={"data_type": "float", "units": "m"},
    )


def create_mock_documents() -> list[Document]:
    """Create a set of mock documents for testing."""
    return [
        create_mock_document("core_profiles/profiles_1d/electrons/temperature"),
        create_mock_document("core_profiles/profiles_1d/electrons/density"),
        create_mock_document("equilibrium/time_slice/profiles_1d/psi", "equilibrium"),
        create_mock_document(
            "equilibrium/time_slice/profiles_2d/b_field_r", "equilibrium"
        ),
        create_mock_document("equilibrium/time_slice/boundary/psi", "equilibrium"),
        create_mock_document("equilibrium/time_slice/boundary/psi_norm", "equilibrium"),
        create_mock_document("equilibrium/time_slice/boundary/type", "equilibrium"),
    ]


def create_mock_clusters() -> list[dict]:
    """Create mock cluster data for testing."""
    return [
        {
            "id": 0,
            "label": "Electron Temperature Profiles",
            "description": "Temperature measurements for electrons",
            "is_cross_ids": False,
            "ids_names": ["core_profiles"],
            "paths": [
                "core_profiles/profiles_1d/electrons/temperature",
                "core_profiles/profiles_1d/electrons/temperature_fit",
            ],
            "similarity_score": 0.95,
            "cluster_similarity": 0.87,
        },
        {
            "id": 1,
            "label": "Magnetic Field Components",
            "description": "Magnetic field measurements and derived quantities",
            "is_cross_ids": True,
            "ids_names": ["equilibrium", "core_profiles"],
            "paths": [
                "equilibrium/time_slice/profiles_2d/b_field_r",
                "equilibrium/time_slice/profiles_2d/b_field_z",
            ],
            "similarity_score": 0.88,
            "cluster_similarity": 0.82,
        },
        {
            "id": 2,
            "label": "Boundary Conditions",
            "description": "Plasma boundary and separatrix data",
            "is_cross_ids": False,
            "ids_names": ["equilibrium"],
            "paths": [
                "equilibrium/time_slice/boundary/psi",
                "equilibrium/time_slice/boundary/psi_norm",
                "equilibrium/time_slice/boundary/type",
            ],
            "similarity_score": 0.92,
            "cluster_similarity": 0.79,
        },
    ]


def create_mock_cluster_search_results(query: str) -> list[ClusterSearchResult]:
    """Create mock cluster search results for testing."""
    mock_clusters = create_mock_clusters()
    return [
        ClusterSearchResult(
            cluster_id=c["id"],
            label=c["label"],
            description=c["description"],
            is_cross_ids=c["is_cross_ids"],
            ids_names=c["ids_names"],
            paths=c["paths"],
            similarity_score=c["similarity_score"],
            cluster_similarity=c["cluster_similarity"],
        )
        for c in mock_clusters[:2]  # Return first 2 clusters
    ]


@pytest.fixture(autouse=True)
def temporary_embedding_cache_dir(tmp_path_factory, monkeypatch):
    """Keep embedding cache files isolated per test."""
    temp_dir = tmp_path_factory.mktemp("embedding_cache")
    monkeypatch.setattr(
        Encoder,
        "_get_cache_directory",
        lambda self, _temp_dir=temp_dir: _temp_dir,
    )
    yield


@pytest.fixture(autouse=True)
def isolated_reviewer_profile_env():
    """Confine a reviewer-profile selection to the test that made it.

    ``IMAS_CODEX_SN_REVIEW_PROFILE`` is the environment channel a reviewer
    profile can arrive on, and it is process-wide. A ``CliRunner`` invocation
    runs inside this process, so without this restore a single test that sets
    it re-seats the reviewer chain for every later test. A one-reviewer
    profile's model carries no checked-in pricing row, so the damage surfaces
    far from its cause as unpriceable routes and empty quorums.
    """
    name = "IMAS_CODEX_SN_REVIEW_PROFILE"
    before = os.environ.get(name)
    yield
    if before is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = before


@pytest.fixture(autouse=True)
def disable_caching():
    """Automatically disable caching for all tests by making cache always miss."""
    # Patch the cache get method to always return None (cache miss)
    with patch("imas_codex.search.decorators.cache._cache.get", return_value=None):
        # Also patch the set method to do nothing
        with patch("imas_codex.search.decorators.cache._cache.set"):
            yield


@pytest.fixture(scope="session", autouse=True)
def mock_heavy_operations():
    """Mock heavy operations that slow down tests.

    Mocks Encoder model loading, DocumentStore I/O, and search engines
    for tests that directly use those classes (not via the MCP server).
    """
    mock_documents = create_mock_documents()

    mock_st_cls = MagicMock()
    mock_st_model = MagicMock()
    mock_st_model.device = "cpu"
    mock_st_model.encode.return_value = np.zeros((len(mock_documents), 384))
    mock_st_cls.return_value = mock_st_model

    with (
        patch.object(
            Encoder, "_import_sentence_transformers", return_value=mock_st_cls
        ),
        patch.multiple(
            DocumentStore,
            _ensure_loaded=MagicMock(),
            _ensure_ids_loaded=MagicMock(),
            _load_ids_documents=MagicMock(),
            _load_identifier_catalog_documents=MagicMock(),
            load_all_documents=MagicMock(),
            _build_sqlite_fts_index=MagicMock(),
            _should_rebuild_fts_index=MagicMock(return_value=False),
            get_all_documents=MagicMock(return_value=mock_documents),
            get_document=MagicMock(
                side_effect=lambda path_id: next(
                    (doc for doc in mock_documents if doc.metadata.path_id == path_id),
                    None,
                )
            ),
            get_documents_by_ids=MagicMock(
                side_effect=lambda ids_name: [
                    doc for doc in mock_documents if doc.metadata.ids_name == ids_name
                ]
            ),
            get_available_ids=MagicMock(return_value=list(STANDARD_TEST_IDS_SET)),
            __len__=MagicMock(return_value=len(mock_documents)),
            search_full_text=MagicMock(return_value=mock_documents[:2]),
            search_by_keywords=MagicMock(return_value=mock_documents[:2]),
            search_by_physics_domain=MagicMock(return_value=mock_documents[:2]),
            search_by_units=MagicMock(return_value=mock_documents[:2]),
            get_statistics=MagicMock(
                return_value={
                    "total_documents": len(mock_documents),
                    "total_ids": len(STANDARD_TEST_IDS_SET),
                    "physics_domains": 2,
                    "unique_units": 1,
                    "coordinate_systems": 1,
                    "documentation_terms": 100,
                    "path_segments": 50,
                }
            ),
            get_identifier_schemas=MagicMock(return_value=[]),
            get_identifier_paths=MagicMock(return_value=[]),
            get_identifier_schema_by_name=MagicMock(return_value=None),
        ),
    ):
        mock_engine = MockSearchEngine()
        with (
            patch(
                "imas_codex.search.engines.semantic_engine.SemanticSearchEngine.search",
                side_effect=mock_engine.search,
            ),
            patch(
                "imas_codex.search.engines.lexical_engine.LexicalSearchEngine.search",
                side_effect=mock_engine.search,
            ),
            patch(
                "imas_codex.search.engines.hybrid_engine.HybridSearchEngine.search",
                side_effect=mock_engine.search,
            ),
        ):
            mock_clusters = create_mock_clusters()
            with patch("imas_codex.core.clusters.Clusters") as mock_clusters_class:
                mock_clusters_instance = MagicMock()
                mock_clusters_instance.is_available.return_value = True
                mock_clusters_instance.get_clusters.return_value = mock_clusters
                mock_clusters_class.return_value = mock_clusters_instance

                yield


def _create_mock_graph_client():
    """Create a mock GraphClient for test fixtures."""
    mock_gc = MagicMock()

    def _query(cypher, **kwargs):
        if "MATCH (i:IDS)" in cypher:
            return [
                {
                    "name": "equilibrium",
                    "description": "Equilibrium quantities",
                    "physics_domain": "magnetics",
                    "lifecycle_status": "active",
                    "path_count": 5,
                },
                {
                    "name": "core_profiles",
                    "description": "Core plasma profiles",
                    "physics_domain": "core_transport",
                    "lifecycle_status": "active",
                    "path_count": 4,
                },
            ]
        if "DDVersion" in cypher and "is_current" in cypher:
            if "AS version" in cypher:
                return [{"version": "4.0.0"}]
            return [{"v.id": "4.0.0"}]
        if "RETURN 1" in cypher:
            return [{"1": 1}]
        if "IMASNode" in cypher and "count" in cypher.lower():
            return [{"paths": 9, "ids_count": 2}]
        return []

    mock_gc.query = MagicMock(side_effect=_query)
    return mock_gc


@pytest.fixture(scope="session")
def tools() -> Tools:
    """Session-scoped tools fixture with mock GraphClient."""
    mock_gc = _create_mock_graph_client()
    return Tools(ids_set=STANDARD_TEST_IDS_SET, graph_client=mock_gc)


@pytest.fixture
def sample_search_results() -> dict[str, Any]:
    """Sample search results for testing."""
    return {
        "results": [
            {
                "path": "core_profiles/profiles_1d/electrons/temperature",
                "ids_name": "core_profiles",
                "score": 0.95,
                "documentation": "Electron temperature profile",
            },
            {
                "path": "equilibrium/time_slice/profiles_1d/psi",
                "ids_name": "equilibrium",
                "score": 0.88,
                "documentation": "Poloidal flux profile",
            },
        ],
        "total_results": 2,
    }


@pytest.fixture
def mcp_test_context():
    """Test context for MCP protocol testing."""
    return {
        "test_query": "plasma temperature",
        "test_ids": "core_profiles",
        "expected_tools": [
            ("path_tool", "check_dd_paths"),
            ("path_tool", "fetch_dd_paths"),
            ("overview_tool", "get_dd_catalog"),
            ("identifiers_tool", "get_dd_identifiers"),
            ("list_tool", "list_dd_paths"),
            ("clusters_tool", "search_dd_clusters"),
            ("search_tool", "search_dd_paths"),
        ],
    }


@pytest.fixture
def workflow_test_data():
    """Test data for workflow testing."""
    return {
        "search_query": "core plasma transport",
        "analysis_target": "core_profiles",
        "export_domain": "transport",
        "concept_to_explain": "equilibrium",
    }
