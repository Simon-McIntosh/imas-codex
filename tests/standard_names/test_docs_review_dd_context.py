"""Docs review must survive a StandardName that has no bound DD source.

A live name with ``source_paths = []`` is a valid state, not an error: the
review still has a name, a description, and documentation to score. The DD
context attachment must therefore degrade to attaching nothing rather than
reaching for a first source path that is not there.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from imas_codex.standard_names.workers import _enrich_docs_review_dd_context


@pytest.mark.parametrize("source_paths", [[], None])
def test_sourceless_name_attaches_no_dd_context_and_opens_no_graph(
    source_paths: list[str] | None,
) -> None:
    item: dict[str, object] = {
        "id": "electron_temperature",
        "source_paths": source_paths,
    }

    with patch("imas_codex.graph.client.GraphClient") as client:
        _enrich_docs_review_dd_context(item)

    client.assert_not_called()
    assert item == {"id": "electron_temperature", "source_paths": source_paths}


def test_first_bound_source_still_supplies_the_dd_context() -> None:
    item: dict[str, object] = {
        "id": "electron_temperature",
        "source_paths": ["dd:core_profiles/profiles_1d/electrons/temperature", "other"],
    }

    with (
        patch("imas_codex.graph.client.GraphClient"),
        patch("imas_codex.standard_names.workers._enrich_dd_path_context") as enrich,
    ):
        _enrich_docs_review_dd_context(item)

    enrich.assert_called_once()
    assert enrich.call_args.args[2] == "core_profiles/profiles_1d/electrons/temperature"
