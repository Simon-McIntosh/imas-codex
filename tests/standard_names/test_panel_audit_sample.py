"""The paid-panel audit sample kept beside the deciding Jev route."""

import json
import os
from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest

from imas_codex.standard_names.graph_store import (
    graph_catalog as _owning_graph_catalog,
    graph_foundation as _owning_graph_foundation,
)
from imas_codex.standard_names.review import cascade, panel_audit
from imas_codex.standard_names.worker_tasks import (
    worker_review_names as _owning_worker_review_names,
)

_JEV_DECISION_SCORE = 0.9
_PANEL_SCORE = 0.2


def _route(axis, route="accept", decision_score=_JEV_DECISION_SCORE):
    from imas_codex.standard_names.review.checks import (
        DOCS_DIMENSIONS,
        NAME_DIMENSIONS,
    )

    dimensions = NAME_DIMENSIONS if axis == "name" else DOCS_DIMENSIONS
    return cascade.CascadeRoute(
        route,
        0.9,
        (),
        dict.fromkeys(dimensions, 20 * decision_score),
        selected_score=0.9,
        decision_score=decision_score,
    )


class _Lease:
    def charge_event(self, *_args, **_kwargs):
        pass

    def release_unused(self):
        pass


class _Manager:
    def __init__(self, *, refuse_audit=False):
        self.refuse_audit = refuse_audit
        self.run_id = "run-audit"
        self.phases = []

    def reserve(self, _amount, phase=""):
        self.phases.append(phase)
        if phase.startswith("review_audit_") and self.refuse_audit:
            return None
        return _Lease()


class _AuditCapture:
    """Records the audit rows the cascade writes, and the panel runs it makes."""

    def __init__(self):
        self.rows = []
        self.panel_calls = []

    def install(self, monkeypatch, panel_result, *, declare=True):
        async def store(item, axis, **kwargs):
            self.rows.append(
                {"item": item, "axis": axis, "skipped_reason": None, **kwargs}
            )

        async def incumbent(*, models):
            self.panel_calls.append(list(models))
            return panel_result

        if declare:
            # A stub declares itself non-persisting; the audit refuses any
            # incumbent whose non-persisting behaviour it cannot prove.
            incumbent.non_persisting = True
        monkeypatch.setattr(panel_audit, "store_panel_audit", store)
        return incumbent


class _AggregateGraph:
    """The aggregate query's review set, in memory."""

    def __init__(self):
        self.reviews = [
            {"judge_kind": "incumbent", "is_canonical": True, "score": 0.4},
            {"judge_kind": "incumbent", "is_canonical": True, "score": 0.5},
        ]
        self.aggregate = None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def query(self, statement, **params):
        head = statement.split("WITH sn, count(r)", 1)[0]
        excludes = (
            "coalesce(r.judge_kind, '') = 'jev'" in head
            and "coalesce(r.is_canonical, false) = false" in head
        )
        reviews = [
            review
            for review in self.reviews
            if not (
                excludes
                and review["judge_kind"] == "jev"
                and not review["is_canonical"]
            )
        ]
        scores = [review["score"] for review in reviews if review["score"] is not None]
        self.aggregate = (
            len(reviews),
            sum(scores) / len(scores) if scores else None,
            len(reviews) > 1 and max(scores) - min(scores) >= params["threshold"],
        )
        return [{"id": "temperature"}]


def _install_cascade(monkeypatch, *, axis, rate, panel_result, declare=True):
    from imas_codex import settings

    capture = _AuditCapture()
    incumbent = capture.install(monkeypatch, panel_result, declare=declare)
    routes = []

    async def judgment(_item, _model):
        return _route(axis), {}, 0.0

    async def store(*_args):
        routes.append(True)

    monkeypatch.setattr(settings, "get_model", lambda _seat: "judge-model")
    monkeypatch.setattr(
        "imas_codex.discovery.base.judgment.decisions_key_present", lambda: True
    )
    monkeypatch.setattr(
        cascade, "_name_judgment" if axis == "name" else "_docs_judgment", judgment
    )
    monkeypatch.setattr(cascade, "_store_route", store)
    monkeypatch.setattr(cascade, "deciding_enabled", lambda _axis: True)
    monkeypatch.setattr(panel_audit, "panel_audit_rate", lambda: rate)
    return capture, incumbent, routes


def test_sample_is_deterministic_and_near_the_rate():
    ids = [f"sn_{index}" for index in range(20000)]
    picked = {
        name for name in ids if panel_audit.audit_selected("name", name, "run-1", 0.05)
    }
    replay = {
        name for name in ids if panel_audit.audit_selected("name", name, "run-1", 0.05)
    }
    assert picked == replay, "a replay must pick the same items"
    assert 0.04 <= len(picked) / len(ids) <= 0.06
    other_run = {
        name for name in ids if panel_audit.audit_selected("name", name, "run-2", 0.05)
    }
    assert picked != other_run, "a different run samples differently"


def test_rate_bounds_select_nothing_or_everything():
    ids = [f"sn_{index}" for index in range(500)]
    assert not any(
        panel_audit.audit_selected("name", name, "run-1", 0.0) for name in ids
    )
    assert all(panel_audit.audit_selected("name", name, "run-1", 1.0) for name in ids)


def test_rate_setting_defaults_to_five_percent(monkeypatch):
    from imas_codex import settings

    monkeypatch.setattr(settings, "_get_section", lambda _name: {})
    assert settings.get_sn_review_panel_audit_rate() == 0.05
    monkeypatch.setattr(
        settings, "_get_section", lambda _name: {"panel-audit-rate": 2.0}
    )
    with pytest.raises(ValueError, match="panel-audit-rate"):
        settings.get_sn_review_panel_audit_rate()


@pytest.mark.asyncio
@pytest.mark.parametrize("axis", ["name", "docs"])
async def test_audited_item_keeps_jev_stage_score_route(monkeypatch, axis):
    panel_result = {
        "winning_score": _PANEL_SCORE,
        "resolution_method": "quorum_consensus",
    }
    capture, incumbent, _routes = _install_cascade(
        monkeypatch, axis=axis, rate=1.0, panel_result=panel_result
    )
    manager = _Manager()
    result = await cascade.adjudicate_review(
        axis, {"id": "temperature"}, ["primary", "secondary"], incumbent, mgr=manager
    )
    assert capture.panel_calls == [["primary", "secondary"]], "the sample ran the panel"
    assert result["winning_score"] == _JEV_DECISION_SCORE, "score must stay Jev's"
    assert result["resolution_method"] == "jev_cascade_accept", "route must stay Jev's"
    assert len(capture.rows) == 1
    audit = capture.rows[0]
    assert audit["panel_result"] is panel_result
    assert audit["jev_route"] == "accept"
    assert audit["skipped_reason"] is None
    stored = panel_audit._audit_record(
        {"id": "temperature"},
        axis,
        panel_result=panel_result,
        jev_route="accept",
        skipped_reason=None,
        min_score=0.85,
    )
    assert stored["resolution_role"] == "audit"
    assert stored["is_canonical"] is False
    assert stored["judge_kind"] == "jev"
    comments = json.loads(stored["comments"])
    assert comments["jev_route"] == "accept"
    assert comments["panel_route"] == "refine"


@pytest.mark.asyncio
async def test_audit_row_is_excluded_from_the_review_aggregates(monkeypatch):
    from imas_codex.standard_names import graph_ops

    graph = _AggregateGraph()
    monkeypatch.setattr(_owning_graph_foundation, "GraphClient", lambda: graph)
    assert _owning_graph_catalog.update_review_aggregates(["temperature"]) == 1
    before = graph.aggregate

    capture, incumbent, _routes = _install_cascade(
        monkeypatch,
        axis="name",
        rate=1.0,
        panel_result={"winning_score": _PANEL_SCORE},
    )

    async def store_into_graph(item, axis, **kwargs):
        capture.rows.append({"item": item, "axis": axis, **kwargs})
        graph.reviews.append(
            panel_audit._audit_record(
                item,
                axis,
                panel_result=kwargs["panel_result"],
                jev_route=kwargs["jev_route"],
                skipped_reason=kwargs.get("skipped_reason"),
                min_score=kwargs["min_score"],
            )
        )

    monkeypatch.setattr(panel_audit, "store_panel_audit", store_into_graph)
    await cascade.adjudicate_review(
        "name",
        {"id": "temperature"},
        ["primary", "secondary"],
        incumbent,
        mgr=_Manager(),
    )
    assert len(capture.rows) == 1
    assert _owning_graph_catalog.update_review_aggregates(["temperature"]) == 1
    assert graph.aggregate == before == (2, 0.45, False)


@pytest.mark.asyncio
async def test_exhausted_budget_skips_the_audit_without_failing_the_decision(
    monkeypatch,
):
    capture, incumbent, _routes = _install_cascade(
        monkeypatch, axis="name", rate=1.0, panel_result={"winning_score": _PANEL_SCORE}
    )
    manager = _Manager(refuse_audit=True)
    result = await cascade.adjudicate_review(
        "name", {"id": "temperature"}, ["primary", "secondary"], incumbent, mgr=manager
    )
    assert result["winning_score"] == _JEV_DECISION_SCORE
    assert capture.panel_calls == [], "an exhausted pool must not run the panel"
    assert len(capture.rows) == 1
    skipped = capture.rows[0]
    assert skipped["panel_result"] is None
    assert "lease refused" in skipped["skipped_reason"]


@pytest.mark.asyncio
async def test_audit_runs_the_quorum_without_persisting_its_rows(monkeypatch):
    """The real persisting seam runs the quorum but writes only the audit row."""
    from functools import partial

    from imas_codex import settings
    from imas_codex.standard_names import graph_ops, workers

    written = []
    canonical_row = {
        "judge_kind": "incumbent",
        "is_canonical": True,
        "score": 0.44,
        "model": "rd-primary",
    }

    async def cycles(**_kwargs):
        return {
            "records": [canonical_row],
            "winning_score": 0.44,
            "canonical_model": "rd-primary",
        }

    monkeypatch.setattr(_owning_worker_review_names, "_run_rd_quorum_cycles", cycles)
    monkeypatch.setattr(
        _owning_graph_catalog, "write_reviews", lambda rows: written.extend(rows)
    )

    async def judgment(_item, _model):
        return _route("name"), {}, 0.0

    async def store(*_args):
        pass

    monkeypatch.setattr(settings, "get_model", lambda _seat: "judge-model")
    monkeypatch.setattr(
        "imas_codex.discovery.base.judgment.decisions_key_present", lambda: True
    )
    monkeypatch.setattr(cascade, "_name_judgment", judgment)
    monkeypatch.setattr(cascade, "_store_route", store)
    monkeypatch.setattr(cascade, "deciding_enabled", lambda _axis: True)
    monkeypatch.setattr(panel_audit, "panel_audit_rate", lambda: 1.0)

    incumbent = partial(
        _owning_worker_review_names._persisting_quorum_cycles,
        sn_id="temperature",
        review_axis="name",
        phase="review_name",
    )
    result = await cascade.adjudicate_review(
        "name",
        {"id": "temperature"},
        ["primary", "secondary"],
        incumbent,
        mgr=_Manager(),
    )

    assert result["winning_score"] == _JEV_DECISION_SCORE
    assert [row["resolution_role"] for row in written] == ["audit"]
    assert not [row for row in written if row.get("judge_kind") == "incumbent"]

    graph = _AggregateGraph()
    graph.reviews.extend(written)
    monkeypatch.setattr(_owning_graph_foundation, "GraphClient", lambda: graph)
    assert _owning_graph_catalog.update_review_aggregates(["temperature"]) == 1
    assert graph.aggregate == (2, 0.45, False)


def test_report_pairs_jev_and_panel_routes(monkeypatch):
    from imas_codex.graph import client as graph_client

    rows = [
        {
            "axis": "name",
            "comments": json.dumps({"jev_route": "accept", "panel_route": "accept"}),
        },
        {
            "axis": "name",
            "comments": json.dumps({"jev_route": "accept", "panel_route": "refine"}),
        },
        {
            "axis": "name",
            "comments": json.dumps({"jev_route": "refine", "panel_route": "accept"}),
        },
        {
            "axis": "name",
            "comments": json.dumps(
                {"jev_route": "accept", "audit_skipped": "review audit lease refused"}
            ),
        },
        {
            "axis": "name",
            "comments": json.dumps({"jev_route": "accept"}),
        },
        {
            "axis": "name",
            "comments": "not a payload",
        },
        {
            "axis": "docs",
            "comments": json.dumps({"jev_route": "accept", "panel_route": "accept"}),
        },
    ]

    class _Graph:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def query(self, _statement, **_params):
            return rows

    monkeypatch.setattr(graph_client, "GraphClient", lambda: _Graph())
    report = panel_audit.panel_audit_report(since="2026-01-01T00:00:00+00:00")
    name = report["name"]
    assert (name.sample_size, name.agreement_count) == (3, 1)
    assert name.agreement_rate == pytest.approx(1 / 3)
    assert name.disagreement_counts == {
        "jev_accept_panel_refine": 1,
        "jev_refine_panel_accept": 1,
    }
    assert name.skipped_count == 3
    assert name.skipped_reasons == {
        "review audit lease refused": 1,
        "audit row carries no panel_route": 1,
        "audit row comments are not readable JSON": 1,
    }
    docs = report["docs"]
    assert (docs.sample_size, docs.agreement_count) == (1, 1)
    assert docs.agreement_rate == 1.0


@pytest.mark.asyncio
async def test_rate_zero_audits_nothing(monkeypatch):
    capture, incumbent, _routes = _install_cascade(
        monkeypatch, axis="name", rate=0.0, panel_result={"winning_score": _PANEL_SCORE}
    )
    manager = _Manager()
    result = await cascade.adjudicate_review(
        "name", {"id": "temperature"}, ["primary", "secondary"], incumbent, mgr=manager
    )
    assert result["winning_score"] == _JEV_DECISION_SCORE
    assert capture.panel_calls == []
    assert capture.rows == []
    assert not [phase for phase in manager.phases if phase.startswith("review_audit_")]


def test_seam_refuses_an_incumbent_it_cannot_prove_non_persisting():
    async def incumbent(**_kwargs):
        return None

    with pytest.raises(panel_audit.UnrecognisedIncumbent) as refusal:
        panel_audit.quorum_without_persisting(incumbent)
    assert repr(incumbent) in str(refusal.value)


def test_seam_returns_a_declared_non_persisting_callable_unchanged():
    async def incumbent(**_kwargs):
        return None

    incumbent.non_persisting = True
    assert panel_audit.quorum_without_persisting(incumbent) is incumbent


def _incumbent_arguments_in(source: str) -> list:
    """The expression each ``adjudicate_review`` call in *source* passes.

    The site is the call, so an argument that wraps the partial in anything
    appears here as that wrapper rather than as the partial.
    """
    import ast

    arguments = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        if not (
            isinstance(node.func, ast.Name) and node.func.id == "adjudicate_review"
        ):
            continue
        arguments.append(node.args[3] if len(node.args) > 3 else None)
    return arguments


def _worker_module_sources() -> list[str]:
    """The source of the workers entrypoint and every worker task module.

    The quorum construction sites move between files as the worker tasks are
    extracted, so the scan walks the whole set rather than one file: a site
    that changes file is still counted, and one that is deleted is still
    missed. Modules are read through their imports, so the check follows the
    code the process would run rather than a path that may name another tree.
    """
    import importlib
    import inspect
    import pkgutil

    import imas_codex.standard_names.worker_tasks as worker_tasks
    from imas_codex.standard_names import workers

    sources = [inspect.getsource(workers)]
    for info in pkgutil.iter_modules(worker_tasks.__path__):
        module = importlib.import_module(f"{worker_tasks.__name__}.{info.name}")
        sources.append(inspect.getsource(module))
    return sources


def _quorum_incumbent_arguments() -> list:
    """The incumbent expression at each quorum call site in the worker modules."""
    arguments = []
    for source in _worker_module_sources():
        arguments.extend(_incumbent_arguments_in(source))
    return arguments


def _nested_persisting_partials(source: str) -> list:
    """Every persisting partial in *source*, at any nesting depth.

    This is the count-based expression the construction-site guard used to
    accept: it finds the inner partial even under a wrapper, which is why it
    could not fail on the wrapper case the guard exists to catch.
    """
    import ast

    return [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "partial"
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "_persisting_quorum_cycles"
    ]


def _is_persisting_partial(node) -> bool:
    """Whether this expression is exactly ``partial(_persisting_quorum_cycles, …)``.

    A decorator or a further ``partial`` layer makes this false, which is the
    case the audit's seam refuses.
    """
    import ast

    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "partial"
        and bool(node.args)
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "_persisting_quorum_cycles"
        and bool(node.keywords)
    )


def test_the_site_guard_rejects_a_wrapped_partial():
    """The wrapper case the guard exists to catch must fail it.

    A count-based expression still finds the inner partial under a wrapper and
    reports the site as sound; the guard asserts the site's own expression, so
    a wrapped site is rejected here.
    """
    wrapped = (
        "async def review(item):\n"
        "    return await adjudicate_review(\n"
        "        'name',\n"
        "        item,\n"
        "        models,\n"
        "        _traced(partial(_persisting_quorum_cycles, sn_id=item['id'])),\n"
        "    )\n"
    )
    assert len(_nested_persisting_partials(wrapped)) == 1, (
        "the inner partial stays findable under a wrapper, which is why a "
        "count-based guard cannot fail on this case"
    )
    sites = _incumbent_arguments_in(wrapped)
    assert len(sites) == 1
    assert not _is_persisting_partial(sites[0]), (
        "a site whose argument wraps the partial must be rejected"
    )


def test_every_quorum_construction_site_builds_the_recognised_partial():
    """A call site that stops building this exact partial must fail here.

    The audit recognises the persisting wrapper by what it is: a partial of
    ``_persisting_quorum_cycles`` carrying the quorum's arguments. A decorator
    or a further partial layer around it at either construction site would
    leave the audit unable to prove the callable writes no canonical rows, and
    the audit refuses such a callable — so the loss of the seam is caught here
    rather than only in production. The assertion is on the site's own
    expression, not on any partial nested inside one.
    """
    import ast

    sites = _quorum_incumbent_arguments()
    assert len(sites) == 2, (
        "both quorum call sites must build partial(_persisting_quorum_cycles, "
        f"**kwargs); found {len(sites)}"
    )
    for site in sites:
        assert _is_persisting_partial(site), (
            "each quorum call site must pass partial(_persisting_quorum_cycles, "
            "**kwargs) as the incumbent argument itself: a decorator or a "
            "further partial layer breaks the audit's seam, which is read by "
            f"the partial's keywords; found {ast.dump(site) if site else None}"
        )


@pytest.mark.asyncio
async def test_unprovable_incumbent_is_skipped_without_running_the_panel(monkeypatch):
    """An incumbent the audit cannot prove non-persisting is never called."""
    capture, incumbent, _routes = _install_cascade(
        monkeypatch,
        axis="name",
        rate=1.0,
        panel_result={"winning_score": _PANEL_SCORE},
        declare=False,
    )
    result = await cascade.adjudicate_review(
        "name",
        {"id": "temperature"},
        ["primary", "secondary"],
        incumbent,
        mgr=_Manager(),
    )
    assert result["winning_score"] == _JEV_DECISION_SCORE
    assert result["resolution_method"] == "jev_cascade_accept", "Jev still decides"
    assert capture.panel_calls == [], "an unprovable callable must not run"
    assert len(capture.rows) == 1
    skipped = capture.rows[0]
    assert skipped["panel_result"] is None
    assert "UnrecognisedIncumbent" in skipped["skipped_reason"]


class _Scores:
    def __init__(self, score):
        self.score = score

    def model_dump(self):
        return {"grammar": self.score}


class _ReviewItem:
    """The reviewer response shape the quorum's cycle parses."""

    def __init__(self, score):
        self.scores = _Scores(score)
        self.reasoning = "reads as a temperature"
        self.comments = None
        self.dd_gaps = []


class _Batch:
    def __init__(self, item):
        self.reviews = [item]


@pytest.mark.asyncio
async def test_audit_row_carries_the_group_id_its_own_cycle_rows_carry(
    monkeypatch,
):
    """The audit row's group id is the one its cycle's rows carry.

    Only the model call is stubbed: the real ``_run_rd_quorum_cycles`` builds
    the rows and returns the group id the audit row must join them under.
    """
    from imas_codex.standard_names import workers

    async def stub_call(**_kwargs):
        return _Batch(_ReviewItem(0.44)), 0.0, (0, 0)

    result = await _owning_worker_review_names._run_rd_quorum_cycles(
        sn_id="temperature",
        review_axis="name",
        response_model=object,
        user_prompt="review this name",
        system_prompt="you are a reviewer",
        models=["rd-primary"],
        disagreement_threshold=0.20,
        rubric_dims=("grammar",),
        phase="review_name",
        acall_llm_structured=stub_call,
        budget_manager=None,
        run_id="run-audit",
    )

    group = result["review_group_id"]
    assert group, "the quorum's result must carry the group its rows share"
    assert [row["review_group_id"] for row in result["records"]] == [group]

    record = panel_audit._audit_record(
        {"id": "temperature"},
        "name",
        panel_result=result,
        jev_route="accept",
        skipped_reason=None,
        min_score=0.85,
    )
    assert record["review_group_id"] == group
    assert record["review_group_id"] == result["records"][0]["review_group_id"]


def test_the_group_id_assertion_detects_a_row_that_invents_its_own_group(
    monkeypatch,
):
    """The equality is falsifiable: a row ignoring its cycle's group differs.

    Without this the assertion could be satisfied by the row minting a group
    id of its own, which is exactly the defect it exists to catch.
    """
    import uuid

    group = "6f1f0c2e-0000-4000-8000-00000000beef"
    monkeypatch.setattr(
        panel_audit, "_audit_group_id", lambda _result: str(uuid.uuid4())
    )
    record = panel_audit._audit_record(
        {"id": "temperature"},
        "name",
        panel_result={"winning_score": _PANEL_SCORE, "review_group_id": group},
        jev_route="accept",
        skipped_reason=None,
        min_score=0.85,
    )
    assert record["review_group_id"] != group


def test_skipped_audit_row_carries_a_review_group_id():
    record = panel_audit._audit_record(
        {"id": "temperature"},
        "name",
        panel_result=None,
        jev_route="accept",
        skipped_reason="review audit lease refused",
        min_score=0.85,
    )
    assert record["review_group_id"]
    assert record["resolution_method"] == panel_audit.AUDIT_SKIPPED_METHOD


@pytest.fixture
def disposable_graph_uri(disposable_neo4j_empty: str) -> Iterator[str]:
    """This test's empty graph.

    The audit row must land through the real ``write_reviews``, which builds
    its own ``GraphClient()`` from the active profile. The shared fixture has
    already pointed that profile at the disposable endpoint and refused the
    project graph's address, so the write cannot reach production.
    """
    yield disposable_neo4j_empty


@pytest.mark.graph
def test_the_real_writer_stores_the_audit_row_and_the_aggregate_ignores_it(
    disposable_graph_uri: str,
):
    """The audit row on a real graph, written unstubbed and read back whole."""
    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names.graph_store.graph_catalog import (
        update_review_aggregates,
        write_reviews,
    )
    from tests.standard_names.conftest import neo4j_test_password

    sn_id = "electron_temperature"
    group_id = "2c0e6a5e-0000-4000-8000-00000000fa11"
    client = GraphClient(
        uri=disposable_graph_uri,
        username="neo4j",
        password=neo4j_test_password(),
        graph_name="panel-audit-row",
    )
    try:
        _run_audit_row_contract(
            client,
            sn_id,
            group_id,
            write_reviews,
            update_review_aggregates,
        )
    finally:
        client.close()


def _run_audit_row_contract(
    client, sn_id, group_id, write_reviews, update_review_aggregates
):
    client.query(
        "CREATE (:StandardName {id: $id, name_stage: 'accepted', status: 'draft'})",
        id=sn_id,
    )
    canonical = {
        "id": f"{sn_id}:rd-primary:canonical",
        "standard_name_id": sn_id,
        "model": "rd-primary",
        "reviewer_model": "rd-primary",
        "model_family": "other",
        "is_canonical": True,
        "judge_kind": "incumbent",
        "score": 0.44,
        "scores_json": "{}",
        "tier": "inadequate",
        "reviewed_at": "2026-10-20T00:00:00+00:00",
        "review_axis": "name",
        "cycle_index": 0,
        "review_group_id": "1a2b3c4d-0000-4000-8000-000000000001",
        "resolution_role": "primary",
        "resolution_method": "quorum_consensus",
    }
    assert write_reviews([canonical]) == 1, "the canonical baseline landed"

    record = panel_audit._audit_record(
        {"id": sn_id},
        "name",
        panel_result={"winning_score": _PANEL_SCORE, "review_group_id": group_id},
        jev_route="refine",
        skipped_reason=None,
        min_score=0.85,
    )
    assert write_reviews([record]) == 1, "the audit row was accepted"

    rows = client.query(
        """
        MATCH (:StandardName {id: $id})-[:HAS_REVIEW]->(r:StandardNameReview)
        RETURN r.review_group_id AS review_group_id,
               r.is_canonical AS is_canonical,
               r.judge_kind AS judge_kind,
               r.resolution_role AS resolution_role,
               r.review_axis AS review_axis,
               r.cycle_index AS cycle_index,
               r.comments AS comments
        """,
        id=sn_id,
    )
    assert len(rows) == 2, f"expected the baseline and the audit row, saw {rows}"
    audit_rows = [row for row in rows if row["resolution_role"] == "audit"]
    assert len(audit_rows) == 1
    stored = audit_rows[0]
    assert stored["review_group_id"] == group_id
    assert stored["is_canonical"] is False
    assert stored["judge_kind"] == "jev"
    assert stored["review_axis"] == "name"
    assert stored["cycle_index"] == -1
    comments = json.loads(stored["comments"])
    assert comments["jev_route"] == "refine"
    assert comments["panel_route"] == "refine"

    assert update_review_aggregates([sn_id]) == 1
    aggregates = client.query(
        """
        MATCH (sn:StandardName {id: $id})
        RETURN sn.review_count AS n, sn.review_mean_score AS mean,
               sn.review_disagreement AS disagreement
        """,
        id=sn_id,
    )[0]
    assert aggregates["n"] == 1, "only the canonical row is aggregated"
    assert aggregates["mean"] == pytest.approx(0.44)
    assert aggregates["disagreement"] is False
