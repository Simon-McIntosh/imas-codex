"""Exact existing-name scope and enrich-only routing contracts."""

from __future__ import annotations

import json
import os
import uuid
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from imas_codex.cli.sn import sn
from imas_codex.standard_names.graph_ops import (
    ExactNameScopeConflict,
    claim_generate_name_batch,
    claim_review_name_batch,
    release_review_names_claims,
    scope_exact_standard_names,
)
from imas_codex.standard_names.turn import skip_flags_from_only


def _candidate(name_id: str, **overrides: object) -> dict[str, object]:
    candidate: dict[str, object] = {
        "id": name_id,
        "name_stage": "drafted",
        "status": "draft",
        "claimed_at": None,
        "claim_token": None,
        "run_id": None,
        "drain_scope_id": None,
        "drain_scope_claimed_at": None,
        "drain_claim_scope_id": None,
    }
    candidate.update(overrides)
    return candidate


def _preflight_row(
    name_id: str,
    *,
    matches: list[dict[str, object]] | None = None,
    protected: list[str] | None = None,
) -> dict[str, object]:
    return {
        "requested_name": name_id,
        "matches": [_candidate(name_id)] if matches is None else matches,
        "protected_producers": protected or [],
    }


class _Transaction:
    def __init__(
        self,
        preflight_rows: list[dict[str, object]],
        stamped_ids: list[str] | None = None,
    ) -> None:
        self.preflight_rows = preflight_rows
        self.stamped_ids = stamped_ids
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.closed = False
        self.committed = False

    def run(self, query: str, **params: object) -> list[dict[str, object]]:
        self.calls.append((query, params))
        if "EXACT_NAME_SCOPE_PREFLIGHT" in query:
            return self.preflight_rows
        if "EXACT_NAME_SCOPE_STAMP" in query:
            return [{"stamped_ids": self.stamped_ids or []}]
        raise AssertionError(f"unexpected query: {query}")

    def commit(self) -> None:
        self.committed = True
        self.closed = True

    def close(self) -> None:
        self.closed = True


class _Session:
    def __init__(self, transaction: _Transaction) -> None:
        self.transaction = transaction
        self.closed = False

    def begin_transaction(self) -> _Transaction:
        return self.transaction

    def __enter__(self) -> _Session:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        self.closed = True


class _Client:
    def __init__(self, transaction: _Transaction) -> None:
        self._session = _Session(transaction)

    def session(self) -> _Session:
        return self._session


def _scope(
    names: list[str],
    transaction: _Transaction,
    *,
    dry_run: bool = False,
) -> dict[str, object]:
    with patch(
        "imas_codex.standard_names.protected_sources.protected_source_ids",
        return_value=frozenset({"dd:west/protected"}),
    ):
        return scope_exact_standard_names(
            names, "scope-run", dry_run=dry_run, gc=_Client(transaction)
        )


def test_dry_run_reads_complete_set_once_and_never_writes() -> None:
    transaction = _Transaction([_preflight_row("alpha"), _preflight_row("beta")])

    result = _scope(["beta", "alpha"], transaction, dry_run=True)

    assert result == {
        "name_ids": ["alpha", "beta"],
        "run_id": "scope-run",
        "dry_run": True,
        "stamped": 0,
    }
    assert len(transaction.calls) == 1
    query, params = transaction.calls[0]
    assert "HAS_PARENT*0.." in query
    assert "HAS_PARENT*1.." in query
    assert params["names"] == ["alpha", "beta"]
    assert params["west_source_ids"] == ["dd:west/protected"]
    assert transaction.committed is False


def test_live_scope_stamps_exact_cardinality_in_one_write() -> None:
    transaction = _Transaction(
        [_preflight_row("alpha"), _preflight_row("beta")],
        stamped_ids=["beta", "alpha"],
    )

    result = _scope(["alpha", "beta"], transaction)

    assert result["stamped"] == 2
    assert transaction.committed is True
    assert len(transaction.calls) == 2
    stamp_query, stamp_params = transaction.calls[1]
    assert "MATCH (name:StandardName" in stamp_query
    assert "SET name.run_id = $run_id" in stamp_query
    assert "SET protected_source" not in stamp_query
    assert stamp_params["names"] == ["alpha", "beta"]
    assert stamp_params["run_id"] == "scope-run"


def test_duplicate_request_refuses_before_graph_access() -> None:
    transaction = _Transaction([])

    with pytest.raises(ExactNameScopeConflict, match="duplicate exact"):
        _scope(["alpha", "alpha"], transaction)

    assert transaction.calls == []


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (_preflight_row("missing", matches=[]), "missing StandardName"),
        (
            _preflight_row(
                "ambiguous", matches=[_candidate("ambiguous"), _candidate("ambiguous")]
            ),
            "ambiguous StandardName",
        ),
        (
            _preflight_row(
                "terminal", matches=[_candidate("terminal", name_stage="exhausted")]
            ),
            "terminal StandardName",
        ),
        (
            _preflight_row(
                "claimed", matches=[_candidate("claimed", claim_token="occupied")]
            ),
            "current worker claim",
        ),
        (
            _preflight_row(
                "draining",
                matches=[_candidate("draining", drain_scope_id="active-drain")],
            ),
            "current drain scope",
        ),
        (
            _preflight_row("protected", protected=["dd:west/protected"]),
            "HAS_PARENT lineage",
        ),
        (
            _preflight_row("fixture", protected=["dd:test_review_entry__persistent"]),
            "protected source",
        ),
    ],
)
def test_any_preflight_refusal_is_atomic(row: dict[str, object], message: str) -> None:
    transaction = _Transaction([row], stamped_ids=[str(row["requested_name"])])

    with pytest.raises(ExactNameScopeConflict, match=message):
        _scope([str(row["requested_name"])], transaction)

    assert len(transaction.calls) == 1
    assert transaction.committed is False


def test_stamp_cardinality_drift_rolls_back() -> None:
    transaction = _Transaction([_preflight_row("alpha")], stamped_ids=[])

    with pytest.raises(ExactNameScopeConflict, match="changed between"):
        _scope(["alpha"], transaction)

    assert transaction.committed is False
    assert transaction.closed is True


def test_durable_edit_provenance_is_replaced_by_exact_scope() -> None:
    transaction = _Transaction(
        [
            _preflight_row(
                "edited_name",
                matches=[_candidate("edited_name", run_id="edit-provenance")],
            )
        ],
        stamped_ids=["edited_name"],
    )

    result = _scope(["edited_name"], transaction)

    assert result["stamped"] == 1
    assert transaction.committed is True
    assert len(transaction.calls) == 2
    stamp_query = transaction.calls[1][0]
    assert "name.run_id IS NULL" not in stamp_query
    assert "SET name.run_id = $run_id" in stamp_query


@pytest.mark.parametrize("cohort_size", [1, 40])
def test_exact_scope_uses_a_constant_two_queries(cohort_size: int) -> None:
    names = [f"bounded_name_{index}" for index in range(cohort_size)]
    sorted_names = sorted(names)
    transaction = _Transaction(
        [_preflight_row(name_id) for name_id in sorted_names],
        stamped_ids=sorted_names,
    )

    result = _scope(names, transaction)

    assert result["stamped"] == cohort_size
    assert len(transaction.calls) == 2
    assert all(
        call_params["names"] == sorted_names
        for _query, call_params in transaction.calls
    )


def test_exact_name_dry_run_splits_values_and_uses_zero_write_helper() -> None:
    scoped = {
        "name_ids": ["alpha", "beta", "gamma"],
        "run_id": "scope-run",
        "dry_run": True,
        "stamped": 0,
    }
    with patch(
        "imas_codex.standard_names.graph_ops.scope_exact_standard_names",
        return_value=scoped,
    ) as helper:
        result = CliRunner().invoke(
            sn,
            [
                "run",
                "--name",
                "alpha beta",
                "--name",
                "gamma",
                "--dry-run",
                "--skip-global-maintenance",
            ],
        )

    assert result.exit_code == 0, result.output
    helper.assert_called_once()
    assert helper.call_args.args[0] == ["alpha", "beta", "gamma"]
    assert helper.call_args.kwargs["dry_run"] is True
    assert "no graph writes performed" in result.output


@pytest.mark.parametrize(
    "conflicting_args",
    [
        ["--focus", "equilibrium/time_slice/psi"],
        ["--batch", "west_production_dd_paths"],
        ["--scope-run-id", "other-run"],
        ["--families", "flux"],
    ],
)
def test_exact_name_rejects_other_scope_selectors(
    conflicting_args: list[str],
) -> None:
    result = CliRunner().invoke(
        sn, ["run", "--name", "alpha", "--dry-run", *conflicting_args]
    )

    assert result.exit_code == 2
    assert "--name is mutually exclusive" in result.output


@pytest.mark.parametrize(
    "retry_flag",
    ["--retry-quarantined", "--retry-skipped", "--retry-vocab-gap"],
)
def test_exact_name_rejects_retry_selectors(retry_flag: str) -> None:
    result = CliRunner().invoke(sn, ["run", "--name", "alpha", "--dry-run", retry_flag])

    assert result.exit_code == 2
    assert f"--name is mutually exclusive with {retry_flag}" in result.output


@pytest.mark.parametrize("skip_global", [False, True])
def test_live_exact_scope_routes_to_ordinary_pools_with_edit_intersection(
    skip_global: bool,
) -> None:
    helper_result = {
        "name_ids": ["poloidal_flux"],
        "run_id": "scope-run",
        "dry_run": False,
        "stamped": 1,
    }
    args = [
        "run",
        "--name",
        "poloidal_flux",
        "--only",
        "review",
        "--names-only",
        "--edits",
    ]
    if skip_global:
        args.append("--skip-global-maintenance")
    with (
        patch("imas_codex.cli.sn._require_embed_ready"),
        patch("imas_codex.cli.sn._auto_sync_grammar"),
        patch("imas_codex.cli.sn._note_pipeline_version_drift"),
        patch(
            "imas_codex.standard_names.graph_ops.scope_exact_standard_names",
            return_value=helper_result,
        ) as helper,
        patch("imas_codex.cli.sn._run_sn_cmd") as run,
    ):
        result = CliRunner().invoke(sn, args)

    assert result.exit_code == 0, result.output
    helper.assert_called_once()
    scope_id = helper.call_args.args[1]
    assert run.call_args.kwargs["scope_run_id"] == scope_id
    assert run.call_args.kwargs["edits_only"] is True
    assert run.call_args.kwargs["skip_global_maintenance"] is skip_global
    assert run.call_args.kwargs["skip_generate"] is True


def test_exact_name_review_action_routes_to_one_pool() -> None:
    helper_result = {
        "name_ids": ["poloidal_flux"],
        "run_id": "scope-run",
        "dry_run": False,
        "stamped": 1,
    }
    with (
        patch("imas_codex.cli.sn._require_embed_ready"),
        patch("imas_codex.cli.sn._auto_sync_grammar"),
        patch("imas_codex.cli.sn._note_pipeline_version_drift"),
        patch(
            "imas_codex.standard_names.graph_ops.scope_exact_standard_names",
            return_value=helper_result,
        ),
        patch("imas_codex.cli.sn._run_sn_cmd") as run,
    ):
        result = CliRunner().invoke(
            sn,
            [
                "run",
                "--name",
                "poloidal_flux",
                "--only",
                "review_name",
                "--names-only",
                "--edits",
                "--skip-global-maintenance",
            ],
        )

    assert result.exit_code == 0, result.output
    kwargs = run.call_args.kwargs
    assert kwargs["only"] == "review_name"
    assert kwargs["names_only"] is True
    assert kwargs["edits_only"] is True
    assert kwargs["skip_global_maintenance"] is True
    assert kwargs["skip_generate"] is True
    assert kwargs["scope_size_hint"] == 1


def test_exact_parent_enrich_routes_only_enrichment_pool_flags() -> None:
    helper_result = {
        "name_ids": ["flux"],
        "run_id": "scope-run",
        "dry_run": False,
        "stamped": 1,
    }
    with (
        patch("imas_codex.cli.sn._require_embed_ready"),
        patch(
            "imas_codex.standard_names.graph_ops.scope_exact_standard_names",
            return_value=helper_result,
        ),
        patch("imas_codex.cli.sn._run_sn_cmd") as run,
    ):
        result = CliRunner().invoke(
            sn,
            [
                "run",
                "--name",
                "flux",
                "--only",
                "enrich",
                "--names-only",
                "--skip-global-maintenance",
            ],
        )

    assert result.exit_code == 0, result.output
    kwargs = run.call_args.kwargs
    assert kwargs["only"] == "enrich"
    assert kwargs["skip_generate"] is True
    assert kwargs["skip_review"] is True
    assert kwargs["names_only"] is True
    assert kwargs["docs_only"] is False


def test_generate_phases_keep_enrichment_while_enrich_is_independent() -> None:
    compose = skip_flags_from_only("compose")
    enrich = skip_flags_from_only("enrich")

    assert compose["skip_generate"] is False
    assert compose["skip_enrich"] is False
    assert enrich["skip_generate"] is True
    assert enrich["skip_enrich"] is False
    assert enrich["skip_review"] is True


def test_shared_claim_query_ands_exact_scope_with_edits() -> None:
    from imas_codex.standard_names.graph_ops import _claim_sn_atomic

    transaction = MagicMock()
    transaction.run.return_value = []
    transaction.closed = False
    session = MagicMock()
    session.begin_transaction.return_value = transaction
    graph = MagicMock()
    graph.__enter__.return_value = graph
    graph.__exit__.return_value = False

    @contextmanager
    def _session_context():
        yield session

    graph.session = _session_context

    with patch("imas_codex.standard_names.graph_ops.GraphClient", return_value=graph):
        claimed = _claim_sn_atomic(
            eligibility_where="sn.name_stage = 'drafted'",
            query_params={},
            batch_size=1,
            scope_run_id="scope-run",
            edits_only=True,
        )

    assert claimed == []
    seed_query = transaction.run.call_args.args[0]
    assert "sn.run_id = $scope_run_id" in seed_query
    assert "coalesce(sn.edit_status, '') = 'open'" in seed_query


@pytest.mark.graph
def test_disposable_graph_keeps_unrelated_names_and_sources_unclaimable() -> None:
    """A live graph proves exact stamping and both claim boundaries."""
    from imas_codex.graph.client import GraphClient

    uri = os.environ.get("IMAS_CODEX_TEST_NEO4J_URI")
    if not uri:
        pytest.skip("IMAS_CODEX_TEST_NEO4J_URI is not configured")
    if os.environ.get("IMAS_CODEX_TEST_NEO4J_EPHEMERAL") != "1":
        pytest.fail("exact-name scope graph test requires an ephemeral graph")
    password = os.environ.get("IMAS_CODEX_TEST_NEO4J_PASSWORD", "")
    suffix = uuid.uuid4().hex
    selected = f"exact_scope_selected_{suffix}"
    unrelated = f"exact_scope_unrelated_{suffix}"
    source_id = f"dd:exact_scope_unrelated/{suffix}"
    first_scope_id = str(uuid.uuid4())
    second_scope_id = str(uuid.uuid4())
    ids = [selected, unrelated]
    with GraphClient(
        uri=uri,
        username=os.environ.get("NEO4J_USERNAME", "neo4j"),
        password=password,
        graph_name="ephemeral-exact-name-scope",
    ) as gc:
        gc.query(
            """
            UNWIND $ids AS name_id
            CREATE (name:StandardName {
              id: name_id, name: name_id, name_stage: 'drafted',
              docs_stage: 'pending', status: 'draft',
              validation_status: 'valid', origin: 'pipeline',
              run_id: CASE WHEN name_id = $selected THEN $edit_run_id ELSE null END,
              edit_status: CASE WHEN name_id = $selected THEN 'open' ELSE null END,
              description: 'Disposable exact-scope regression identity.'
            })
            WITH count(*) AS ignored
            CREATE (:StandardNameSource {
              id: $source_id, source_type: 'dd', source_id: $source_id,
              status: 'extracted'
            })
            """,
            ids=ids,
            selected=selected,
            edit_run_id=f"edit:{suffix}",
            source_id=source_id,
        )
        try:
            first_result = scope_exact_standard_names([selected], first_scope_id, gc=gc)
            second_result = scope_exact_standard_names(
                [selected], second_scope_id, gc=gc
            )
            assert first_result["stamped"] == 1
            assert second_result["stamped"] == 1
            rows = gc.query(
                """
                MATCH (name:StandardName) WHERE name.id IN $ids
                OPTIONAL MATCH (source:StandardNameSource {id: $source_id})
                RETURN name.id AS id, name.run_id AS run_id,
                       source.run_id AS source_run_id
                ORDER BY id
                """,
                ids=ids,
                source_id=source_id,
            )
            by_id = {row["id"]: row for row in rows}
            assert by_id[selected]["run_id"] == second_scope_id
            assert by_id[unrelated]["run_id"] is None
            assert all(row["source_run_id"] is None for row in rows)

            def _ephemeral_client() -> GraphClient:
                return GraphClient(
                    uri=uri,
                    username=os.environ.get("NEO4J_USERNAME", "neo4j"),
                    password=password,
                    graph_name="ephemeral-exact-name-scope-claim",
                )

            with patch(
                "imas_codex.standard_names.graph_ops.GraphClient",
                side_effect=_ephemeral_client,
            ):
                reviewed = claim_review_name_batch(
                    scope_run_id=second_scope_id,
                    edits_only=True,
                    batch_size=10,
                )
                assert {item["id"] for item in reviewed} == {selected}
                if reviewed:
                    release_review_names_claims(
                        sn_ids=[item["id"] for item in reviewed],
                        claim_token=str(reviewed[0]["claim_token"]),
                    )
                assert (
                    claim_generate_name_batch(
                        scope_run_id=second_scope_id, batch_size=10
                    )
                    == []
                )
        finally:
            gc.query(
                "MATCH (node) WHERE node.id IN $ids DETACH DELETE node",
                ids=[*ids, source_id],
            )


def _plan_operator_types(plan: object) -> list[str]:
    """Flatten Neo4j plan operators without depending on driver internals."""
    if plan is None:
        return []
    if isinstance(plan, dict):
        operator = plan.get("operatorType") or plan.get("operator_type")
        children = plan.get("children", [])
    else:
        operator = getattr(plan, "operator_type", None)
        children = getattr(plan, "children", [])
    operators = [str(operator)] if operator else []
    for child in children:
        operators.extend(_plan_operator_types(child))
    return operators


@pytest.mark.graph
def test_disposable_graph_exact_scope_query_plans_are_index_bounded() -> None:
    """Cohorts of one and forty retain indexed starts and bounded expansion."""
    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names.graph_ops import (
        _EXACT_NAME_SCOPE_PREFLIGHT_QUERY,
        _EXACT_NAME_SCOPE_STAMP_QUERY,
        _EXACT_SCOPE_TERMINAL_NAME_STAGES,
        _EXACT_SCOPE_TERMINAL_STATUSES,
    )

    uri = os.environ.get("IMAS_CODEX_TEST_NEO4J_URI")
    if not uri:
        pytest.skip("IMAS_CODEX_TEST_NEO4J_URI is not configured")
    if os.environ.get("IMAS_CODEX_TEST_NEO4J_EPHEMERAL") != "1":
        pytest.fail("exact-name scope graph test requires an ephemeral graph")
    password = os.environ.get("IMAS_CODEX_TEST_NEO4J_PASSWORD", "")
    suffix = uuid.uuid4().hex
    ids = [f"exact_scope_plan_{suffix}_{index}" for index in range(40)]
    plans: dict[str, dict[str, list[str]]] = {}
    with GraphClient(
        uri=uri,
        username=os.environ.get("NEO4J_USERNAME", "neo4j"),
        password=password,
        graph_name="ephemeral-exact-name-scope-plans",
    ) as gc:
        gc.query(
            "CREATE CONSTRAINT exact_scope_name_id IF NOT EXISTS "
            "FOR (name:StandardName) REQUIRE name.id IS UNIQUE"
        )
        gc.query(
            "UNWIND $ids AS name_id CREATE (:StandardName {"
            "id: name_id, name_stage: 'drafted', status: 'draft'})",
            ids=ids,
        )
        try:
            for cohort_size in (1, 40):
                names = ids[:cohort_size]
                common_params = {
                    "names": names,
                    "west_source_ids": [],
                    "fixture_source_id_prefix": "dd:test_review_entry__",
                }
                query_specs = {
                    "preflight": (
                        _EXACT_NAME_SCOPE_PREFLIGHT_QUERY,
                        common_params,
                    ),
                    "stamp": (
                        _EXACT_NAME_SCOPE_STAMP_QUERY,
                        {
                            **common_params,
                            "run_id": "explain-only",
                            "terminal_name_stages": sorted(
                                _EXACT_SCOPE_TERMINAL_NAME_STAGES
                            ),
                            "terminal_statuses": sorted(_EXACT_SCOPE_TERMINAL_STATUSES),
                        },
                    ),
                }
                plans[str(cohort_size)] = {}
                with gc.session() as session:
                    for label, (query, params) in query_specs.items():
                        result = session.run("EXPLAIN " + query, **params)
                        operators = _plan_operator_types(result.consume().plan)
                        plans[str(cohort_size)][label] = operators
                        assert "AllNodesScan" not in operators
                        assert not any("AllRelationshipsScan" in op for op in operators)
            print("EXACT_SCOPE_PLAN_EVIDENCE=" + json.dumps(plans, sort_keys=True))
        finally:
            gc.query("MATCH (node) WHERE node.id IN $ids DETACH DELETE node", ids=ids)
