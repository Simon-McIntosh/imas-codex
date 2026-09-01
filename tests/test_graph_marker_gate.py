"""Regression coverage for explicit graph-test selection."""

from pathlib import Path
from unittest.mock import Mock

import conftest
import pytest

pytest_plugins = ("pytester",)


def _install_project_conftest(
    pytester: pytest.Pytester, *, credential_configured: bool | None = None
) -> None:
    source = Path(conftest.__file__).read_text(encoding="utf-8")
    if credential_configured is not None:
        source += (
            f"\n_graph_credential_is_configured = lambda: {credential_configured!r}\n"
        )
    pytester.makeconftest(source)


def _write_mixed_test_file(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        test_selection="""
        import pytest

        @pytest.mark.graph
        def test_requires_graph():
            pass

        def test_without_graph():
            pass
        """
    )


def test_missing_credential_explicit_graph_selection_exits_not_run(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_project_conftest(pytester, credential_configured=False)
    _write_mixed_test_file(pytester)
    monkeypatch.delenv("IMAS_CODEX_TEST_NEO4J_URI", raising=False)

    result = pytester.runpytest_subprocess("-m", "graph", "-q")

    assert result.ret == pytest.ExitCode.NO_TESTS_COLLECTED
    result.stdout.fnmatch_lines(
        ["*Graph-marked test selection was not run: No Neo4j credential is configured*"]
    )


def test_default_credentialless_selection_stays_green(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_project_conftest(pytester, credential_configured=False)
    _write_mixed_test_file(pytester)
    monkeypatch.delenv("IMAS_CODEX_TEST_NEO4J_URI", raising=False)

    result = pytester.runpytest_subprocess("-m", "not slow and not graph", "-q")

    assert result.ret == pytest.ExitCode.OK
    result.assert_outcomes(passed=1, deselected=1)


def test_explicit_graph_selection_with_available_graph_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = Mock()
    config.getoption.return_value = "graph"
    graph_item = Mock()
    graph_item.get_closest_marker.side_effect = lambda name: (
        name if name == "graph" else None
    )
    monkeypatch.setattr(conftest, "_check_neo4j", Mock(return_value=True))

    conftest.pytest_collection_modifyitems(config, [graph_item])

    graph_item.add_marker.assert_not_called()
