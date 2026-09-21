"""`sn review --target groups` reports drifting families without writing.

Grouped review is the cohort axis: it reports the sibling families
``harmonize.build_worklist`` already detects, carrying each family's parent,
member count, drift and anchor, and it stops there. The read-only decision is
the mutation ban, so one test makes the four harmonization apply helpers raise
and shows the command still succeeds.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pytest
from click.testing import CliRunner

_PARENT = "alpha_base"
_MEMBERS = [
    {
        "id": "alpha_base_at_boundary",
        "description": "alpha base at plasma boundary",
        "documentation": "Alpha base at the plasma boundary.",
        "docs_stage": "drafted",
        "operator_kind": "qualifier",
        "operator": None,
    },
    {
        "id": "alpha_base_at_axis",
        "description": "beta gamma delta",
        "documentation": "A different opening entirely.",
        "docs_stage": "accepted",
        "operator_kind": "qualifier",
        "operator": None,
    },
    {
        "id": "alpha_base_of_probe",
        "description": "epsilon zeta eta",
        "documentation": "Yet another opening.",
        "docs_stage": "drafted",
        "operator_kind": "coordinate",
        "operator": None,
    },
]

_FAMILY_QUERY_MARKER = "MATCH (c:StandardName)-[r:HAS_PARENT]->(p:StandardName)"


class _FamilyGraph:
    """GraphClient stand-in serving build_worklist's family query only."""

    def __enter__(self) -> _FamilyGraph:
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def close(self) -> None:
        return None

    def query(self, cypher: str, **_params: Any) -> list[dict[str, Any]]:
        if _FAMILY_QUERY_MARKER in cypher:
            return [
                {
                    "parent_id": _PARENT,
                    "parent_docs_stage": "accepted",
                    "parent_description": "Alpha base quantity.",
                    "harmonized_group_signature": None,
                    "members": [dict(member) for member in _MEMBERS],
                }
            ]
        return []


def _install_family_graph(monkeypatch: Any) -> None:
    from imas_codex.graph import client as graph_client

    monkeypatch.setattr(graph_client, "GraphClient", _FamilyGraph)


def _worklist_from_output(output: str) -> list[dict[str, Any]]:
    match = re.search(r"\[\{.*\}\]", output, re.DOTALL)
    assert match is not None, output
    return json.loads(match.group(0))


def test_grouped_target_reports_parent_members_drift_and_anchor(
    monkeypatch: Any,
) -> None:
    """The four family fields reach the command's output."""
    from imas_codex.cli.sn import sn

    _install_family_graph(monkeypatch)

    result = CliRunner().invoke(sn, ["review", "--target", "groups"])

    assert result.exit_code == 0, result.output
    worklist = _worklist_from_output(result.output)
    assert len(worklist) > 0, result.output

    family = worklist[0]
    assert family["parent"] == _PARENT
    assert family["n"] == len(_MEMBERS)
    assert family["drift"] == 1.0 - 1.0 / len(_MEMBERS)
    assert family["anchor"] is not None

    # The rendered line carries the same four fields, not only the JSON.
    line = re.search(
        r"parent=(\S+)\s+members=(\d+)\s+drift=([\d.]+)\s+anchor=(\S+)",
        result.output,
    )
    assert line is not None, result.output
    assert line.group(1) == _PARENT
    assert int(line.group(2)) == len(_MEMBERS)
    assert float(line.group(3)) == pytest.approx(family["drift"], abs=1e-3)
    assert line.group(4) == family["anchor"]


def test_grouped_target_calls_none_of_the_apply_helpers(monkeypatch: Any) -> None:
    """The read-only decision is a four-name mutation ban, and it holds."""
    from imas_codex.cli.sn import sn
    from imas_codex.standard_names import harmonize

    _install_family_graph(monkeypatch)

    def _refuse(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("grouped review must not mutate the graph")

    for name in (
        "mark_members_for_regen",
        "stamp_harmonized",
        "mark_families_for_regen",
        "restamp_harmonized_families",
    ):
        monkeypatch.setattr(harmonize, name, _refuse)

    # Negative control: the ban is live, so a helper reached through the
    # module the command imports from would raise rather than pass quietly.
    for name in (
        "mark_members_for_regen",
        "stamp_harmonized",
        "mark_families_for_regen",
        "restamp_harmonized_families",
    ):
        assert callable(getattr(harmonize, name))
        with pytest.raises(AssertionError, match="must not mutate"):
            getattr(harmonize, name)()

    result = CliRunner().invoke(sn, ["review", "--target", "groups"])

    assert result.exit_code == 0, result.output
    assert len(_worklist_from_output(result.output)) > 0
