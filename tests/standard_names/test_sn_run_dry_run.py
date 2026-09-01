"""The Standard Names run preview must stay outside every write-capable path."""

from __future__ import annotations

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

from click.testing import CliRunner

from imas_codex.cli.sn import sn


def test_dry_run_prints_plan_without_writes_claims_or_llm_calls() -> None:
    queries: list[str] = []

    class ReadOnlyGraphClient:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def query(self, query: str, **_params):
            queries.append(query)
            return [
                {
                    "generate_name": 3,
                    "generate_name_done": 11,
                    "review_name": 2,
                    "review_name_done": 9,
                    "refine_name": 1,
                    "refine_name_done": 4,
                    "generate_docs": 5,
                    "generate_docs_done": 8,
                    "review_docs": 6,
                    "review_docs_done": 7,
                    "refine_docs": 1,
                    "refine_docs_done": 2,
                    "enrich_parents": 0,
                    "enrich_parents_done": 3,
                }
            ]

    claim_names = (
        "claim_generate_name_batch",
        "claim_review_name_batch",
        "claim_refine_name_batch",
        "claim_generate_docs_batch",
        "claim_review_docs_batch",
        "claim_refine_docs_batch",
        "claim_enrich_parents_batch",
    )
    with ExitStack() as stack:
        stack.enter_context(
            patch("imas_codex.graph.client.GraphClient", ReadOnlyGraphClient)
        )
        stack.enter_context(
            patch(
                "imas_codex.standard_names.graph_ops.get_existing_standard_names",
                return_value=set(),
            )
        )
        stack.enter_context(
            patch(
                "imas_codex.standard_names.sources.dd.extract_dd_candidates",
                return_value=[
                    SimpleNamespace(
                        items=[
                            {"path": "equilibrium/time_slice/a"},
                            {"path": "equilibrium/time_slice/b"},
                        ]
                    )
                ],
            )
        )
        llm = stack.enter_context(
            patch("imas_codex.discovery.base.llm.call_llm_structured")
        )
        claims = [
            stack.enter_context(
                patch(f"imas_codex.standard_names.graph_ops.{claim_name}")
            )
            for claim_name in claim_names
        ]

        result = CliRunner().invoke(
            sn,
            [
                "run",
                "--skip-clear-gate",
                "--dry-run",
                "--domain",
                "equilibrium",
            ],
            catch_exceptions=False,
        )

    assert result.exit_code == 0
    assert "Standard Names dry-run plan" in result.output
    assert "Extraction candidates: 2 (dd)" in result.output
    assert "generate_name (pending=3)" in result.output
    assert "Graph writes: 0; claims: 0; LLM calls: 0" in result.output
    assert queries
    write_words = (" CREATE ", " MERGE ", " SET ", " DELETE ", " REMOVE ")
    assert all(
        not any(word in f" {query.upper()} " for word in write_words)
        for query in queries
    )
    assert all(claim.call_count == 0 for claim in claims)
    assert llm.call_count == 0
