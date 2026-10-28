"""An export over an empty candidate set refuses instead of exiting zero.

An export that filtered every candidate used to exit zero: the population
gate described the empty population but the command still returned success,
so a caller could not tell "nothing to publish" from "published". The refusal
must surface as a non-zero exit from the command, not only as a report field.
"""

from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from imas_codex.cli.sn import sn


class _ReadOnlyGraphClient:
    """A read-only client over an empty graph.

    The export's source audit reads an independent population count before it
    reads any row and refuses when the two disagree, so an empty graph still has
    to answer a count query with the row a count query returns. Answering that
    with no rows at all reads as a broken read, not as an empty population.
    """

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def query(self, cypher: str, **params):
        if "AS total" in cypher:
            return [{"total": 0}]
        if "AS names" in cypher and "AS projections" in cypher:
            return [{"names": 0, "relations": 0, "changes": 0, "projections": 0}]
        return []


def test_export_over_empty_candidate_set_exits_non_zero(tmp_path: Path) -> None:
    runner = CliRunner()
    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "imas_codex.standard_names.export._fetch_export_population",
                return_value=[],
            )
        )
        stack.enter_context(
            patch(
                "imas_codex.graph.client.GraphClient",
                return_value=_ReadOnlyGraphClient(),
            )
        )
        stack.enter_context(
            patch(
                "imas_codex.standard_names.export._fetch_deprecation_stubs",
                return_value=[],
            )
        )
        stack.enter_context(
            patch(
                "imas_codex.standard_names.export._fetch_ordering_edges_for_domain",
                return_value=([], set()),
            )
        )
        result = runner.invoke(
            sn,
            [
                "release",
                "--export-only",
                "--staging",
                str(tmp_path),
                "--skip-gate",
                "--force",
            ],
        )

    assert result.exit_code != 0, result.output
    assert not (tmp_path / "catalog.yml").exists()

    persisted = json.loads(
        (tmp_path / ".export_report.json").read_text(encoding="utf-8")
    )
    population_gate = next(
        gate for gate in persisted["gates"] if gate["gate"] == "export_population"
    )
    assert population_gate["passed"] is False
    assert [issue["type"] for issue in population_gate["issues"]] == [
        "empty_input_population"
    ]
