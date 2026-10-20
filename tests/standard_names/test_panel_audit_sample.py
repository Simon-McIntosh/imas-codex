"""The paid-panel audit sample kept beside the deciding Jev route."""

import json
from unittest.mock import MagicMock

import pytest

from imas_codex.standard_names.review import cascade, panel_audit

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

    def install(self, monkeypatch, panel_result):
        async def store(item, axis, **kwargs):
            self.rows.append(
                {"item": item, "axis": axis, "skipped_reason": None, **kwargs}
            )

        async def incumbent(*, models):
            self.panel_calls.append(list(models))
            return panel_result

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


def _install_cascade(monkeypatch, *, axis, rate, panel_result):
    from imas_codex import settings

    capture = _AuditCapture()
    incumbent = capture.install(monkeypatch, panel_result)
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
    monkeypatch.setattr(cascade, "deciding_enabled", lambda: True)
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
    monkeypatch.setattr(graph_ops, "GraphClient", lambda: graph)
    assert graph_ops.update_review_aggregates(["temperature"]) == 1
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
    assert graph_ops.update_review_aggregates(["temperature"]) == 1
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

    monkeypatch.setattr(workers, "_run_rd_quorum_cycles", cycles)
    monkeypatch.setattr(graph_ops, "write_reviews", lambda rows: written.extend(rows))

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
    monkeypatch.setattr(cascade, "deciding_enabled", lambda: True)
    monkeypatch.setattr(panel_audit, "panel_audit_rate", lambda: 1.0)

    incumbent = partial(
        workers._persisting_quorum_cycles,
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
    monkeypatch.setattr(graph_ops, "GraphClient", lambda: graph)
    assert graph_ops.update_review_aggregates(["temperature"]) == 1
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
