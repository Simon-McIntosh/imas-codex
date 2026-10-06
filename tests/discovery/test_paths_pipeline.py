"""Tests for paths discovery pipeline behaviour.

The ``--reset-to`` reset must be scoped by ``--root``: a root-scoped
reprocess clears only the given root and its descendants, so the reset does
not touch every path at the facility.
"""

from __future__ import annotations

import importlib
from unittest.mock import patch

import pytest

# The package namespace exposes the ``paths`` click command; import the module
# itself so the internal discovery function can be called directly.
paths_cli = importlib.import_module("imas_codex.cli.discover.paths")


class _StopAtReset(Exception):
    """Raised by the patched reset to stop the pipeline at the reset call."""


def _capture_reset(root_filter):
    """Run the pipeline up to its reset call and capture the reset arguments."""
    captured: dict = {}

    def fake_reset(
        spec,
        facility,
        *,
        path_prefixes=None,
        extra_filter="",
        extra_params=None,
    ):
        captured["spec"] = spec
        captured["facility"] = facility
        captured["path_prefixes"] = path_prefixes
        captured["extra_filter"] = extra_filter
        captured["extra_params"] = extra_params
        raise _StopAtReset

    with (
        patch(
            "imas_codex.discovery.get_discovery_stats",
            return_value={"total": 0},
        ),
        patch(
            "imas_codex.cli.discover.common.use_rich_output",
            return_value=False,
        ),
        patch(
            "imas_codex.cli.discover.common.setup_logging",
            return_value=None,
        ),
        patch(
            "imas_codex.cli.discover.common.make_log_print",
            return_value=lambda *a, **k: None,
        ),
        patch(
            "imas_codex.discovery.base.reset.reset_to_status",
            side_effect=fake_reset,
        ),
        pytest.raises(_StopAtReset),
    ):
        paths_cli._run_iterative_discovery(
            facility="jt-60sa",
            budget=5.0,
            path_limit=None,
            focus=None,
            threshold=0.7,
            root_filter=root_filter,
            reset_to="scanned",
        )
    return captured


def test_reset_is_root_scoped_when_root_given():
    captured = _capture_reset(["/analysis/src"])
    assert captured["spec"] is not None
    assert captured["path_prefixes"] == ["/analysis/src"]
    assert not captured["extra_filter"]
    assert not captured["extra_params"]


def test_reset_matches_root_and_descendants():
    """The scope matches paths under the root; the reset owns the clause text."""
    from imas_codex.graph.query_builder import render_path_prefix_clause

    captured = _capture_reset(["/analysis/src"])
    assert captured["path_prefixes"] == ["/analysis/src"]
    clause = render_path_prefix_clause("n", "path_prefixes")
    assert "n.path STARTS WITH prefix" in clause


def test_reset_is_unscoped_without_root():
    captured = _capture_reset(None)
    assert captured["path_prefixes"] is None
    assert captured["extra_filter"] == ""
    assert not captured["extra_params"]
