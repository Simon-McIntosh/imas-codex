"""The review cascade's threshold boundaries and execution modes."""

import pytest

from imas_codex.standard_names.review import cascade
from imas_codex.standard_names.review.cascade import (
    CascadeRoute,
    route_docs,
    route_name,
)


@pytest.mark.asyncio
async def test_shadow_review_record_is_noncanonical(monkeypatch):
    import json

    from imas_codex.standard_names import graph_ops

    stored = []
    monkeypatch.setattr(graph_ops, "write_reviews", lambda rows: stored.extend(rows))
    await cascade._store_route(
        {"id": "temperature"},
        "name",
        "judge-model",
        CascadeRoute("refine", 0.2, ("source_fidelity_unsupported_qualifier",)),
        {"check:source_fidelity_unsupported_qualifier": {"noul": 0.9}},
        0.0,
    )
    assert len(stored) == 1
    assert stored[0]["judge_kind"] == "jev"
    assert stored[0]["is_canonical"] is False
    assert stored[0]["score"] is None
    assert json.loads(stored[0]["judge_answers_json"])["cascade:route"] == "refine"


def test_failed_checks_feed_existing_refine_trigger():
    import json

    from imas_codex.standard_names.fanout.trigger import should_trigger_fanout
    from imas_codex.standard_names.review.checks import load_review_checks

    route = CascadeRoute("refine", 0.1, ("source_fidelity_unsupported_qualifier",))
    result = cascade._direct_result(
        route, "judge-model", 0.0, {"id": "temperature"}, "name"
    )
    item = {
        "reviewer_comments_per_dim_name": json.dumps(result["winning_comments_per_dim"])
    }
    checks = cascade.failed_check_instructions(item, "name")
    assert checks[0]["id"] == "source_fidelity_unsupported_qualifier"
    fires, _ = should_trigger_fanout(
        reviewer_comments_per_dim=None,
        chain_length=1,
        chain_history=[{}],
        keywords=(),
        dims=(),
        char_cap=0,
        trigger_source="checks",
        checks=[check for check in load_review_checks()[0] if check.axis == "name"],
        check_results={
            check["id"]: {"answered": True, "failure_probability": 1.0}
            for check in checks
        },
        check_threshold=0.5,
    )
    assert fires


@pytest.mark.asyncio
async def test_contradiction_note_uses_named_reason(monkeypatch):
    from imas_codex import settings
    from imas_codex.standard_names import workers
    from imas_codex.standard_names.review import jev_review

    class Lease:
        def charge_event(self, *_args):
            pass

        def release_unused(self):
            pass

    class Manager:
        def reserve(self, *_args, **_kwargs):
            return Lease()

    monkeypatch.setattr(settings, "get_model", lambda _seat: "mock")
    monkeypatch.setattr(
        jev_review,
        "build_docs_state",
        lambda _item: {"documentation_claims": ["generic claim"]},
    )
    monkeypatch.setattr(
        jev_review,
        "docs_question_table",
        lambda *_args: {"check:claim_contradicts_source:0": {}},
    )
    monkeypatch.setattr(jev_review, "_default_checks", lambda: [])

    async def decide(*_args, **_kwargs):
        return [({"check:claim_contradicts_source:0": {}}, 0.0)], 0.0

    monkeypatch.setattr(jev_review, "decide_batch", decide)
    monkeypatch.setattr(
        jev_review,
        "aggregate_docs_review",
        lambda *_args: {
            "contradiction_veto": True,
            "contradiction_reason": "contradicts source: named sentence",
            "claim_answers": [{"choice": "contradicts"}],
        },
    )
    note = await workers._documentation_contradiction_note(
        {
            "id": "temperature",
            "dd_source_docs": [
                {"id": "profiles/temperature", "documentation": "DD text"}
            ],
        },
        Manager(),
    )
    assert "named sentence" in note


def test_docs_audit_calls_house_style_owner(monkeypatch):
    from imas_codex.standard_names.context import conventions_context_for
    from imas_codex.standard_names.review import audits, jev_review

    name_id = "flux_surface_averaged_argon_density_at_plasma_boundary"
    marker = audits.LintFinding(
        name_id=name_id, finding_type="convention_violation", detail="owner called"
    )
    monkeypatch.setattr(audits, "run_house_style_checks", lambda _items: [marker])
    monkeypatch.setattr(audits, "run_documentation_gate_checks", lambda _items: [])
    monkeypatch.setattr(audits, "run_link_integrity", lambda *_args, **_kwargs: [])
    item = {"id": name_id, "conventions": conventions_context_for(name_id)}
    report = jev_review.build_docs_audit_report(
        item, catalog_names=[item], known_ids={name_id}
    )
    assert marker in report.lint_findings


@pytest.mark.parametrize("route,axis", [(route_name, "name"), (route_docs, "docs")])
def test_accept_at_threshold(route, axis):
    judgment = {"checks": {"clean": {"answered": True, "failure_probability": 0.1}}}
    threshold = cascade._calibration()[axis]["accept_threshold"]
    assert route(judgment, probability=threshold).route == "accept"
    assert route(judgment, probability=threshold - 0.000001).route == "chat_review"


@pytest.mark.parametrize("route,axis", [(route_name, "name"), (route_docs, "docs")])
def test_refine_at_check_threshold(route, axis):
    check_id, threshold = next(
        iter(cascade._calibration()[axis]["check_thresholds"].items())
    )
    judgment = {
        "checks": {check_id: {"answered": True, "failure_probability": threshold}}
    }
    decision = route(judgment, probability=0.9)
    assert decision.route == "refine"
    assert decision.failed_checks == (check_id,)
    judgment["checks"][check_id]["failure_probability"] = threshold - 0.000001
    assert route(judgment, probability=0.9).route == "accept"


@pytest.mark.parametrize("route,axis", [(route_name, "name"), (route_docs, "docs")])
def test_uncertain_without_failed_check_calls_chat(route, axis):
    judgment = {"checks": {"clean": {"answered": True, "failure_probability": 0.1}}}
    threshold = cascade._calibration()[axis]["accept_threshold"]
    assert route(judgment, probability=threshold - 0.01).route == "chat_review"


@pytest.mark.asyncio
async def test_shadow_records_route_and_keeps_incumbent_result(monkeypatch):
    from imas_codex import settings

    records = []
    calls = []

    async def judgment(_item, _model):
        return CascadeRoute("accept", 0.95), {}, 0.0

    async def store(_item, _axis, _model, route, _answers, _cost):
        records.append(route.route)

    async def incumbent(*, models):
        calls.append(list(models))
        return {"winning_score": 0.2, "resolution_method": "quorum_consensus"}

    class Lease:
        def charge_event(self, *_args):
            pass

        def release_unused(self):
            pass

    class Manager:
        def reserve(self, *_args, **_kwargs):
            return Lease()

    monkeypatch.setattr(settings, "get_model", lambda _seat: "judge-model")
    monkeypatch.setattr(cascade, "_name_judgment", judgment)
    monkeypatch.setattr(cascade, "_store_route", store)
    monkeypatch.setattr(cascade, "deciding_enabled", lambda: False)
    result = await cascade.adjudicate_review(
        "name",
        {"id": "temperature"},
        ["primary", "secondary"],
        incumbent,
        mgr=Manager(),
    )
    assert records == ["accept"]
    assert calls == [["primary", "secondary"]]
    assert result["winning_score"] == 0.2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "route,expected_calls,expected_score",
    [
        ("accept", [], 1.0),
        ("refine", [], 0.0),
        ("chat_review", [["primary"]], 0.3),
    ],
)
@pytest.mark.parametrize("axis", ["name", "docs"])
async def test_deciding_routes_use_one_chat_seat(
    monkeypatch, route, expected_calls, expected_score, axis
):
    from imas_codex import settings

    calls = []
    check_id = (
        "advisory_alias_collapsed" if axis == "name" else "claim_contradicts_source"
    )

    async def judgment(_item, _model):
        return (
            CascadeRoute(route, 0.5, (check_id,) if route == "refine" else ()),
            {},
            0.0,
        )

    async def store(*_args):
        pass

    async def incumbent(*, models):
        calls.append(list(models))
        return {"winning_score": 0.3}

    class Lease:
        def charge_event(self, *_args):
            pass

        def release_unused(self):
            pass

    class Manager:
        def reserve(self, *_args, **_kwargs):
            return Lease()

    monkeypatch.setattr(settings, "get_model", lambda _seat: "judge-model")
    monkeypatch.setattr(
        cascade, "_name_judgment" if axis == "name" else "_docs_judgment", judgment
    )
    monkeypatch.setattr(cascade, "_store_route", store)
    monkeypatch.setattr(cascade, "deciding_enabled", lambda: True)
    result = await cascade.adjudicate_review(
        axis,
        {"id": "temperature"},
        ["primary", "secondary"],
        incumbent,
        mgr=Manager(),
    )
    assert calls == expected_calls
    assert result["winning_score"] == expected_score
    if route == "refine":
        assert result["winning_comments_per_dim"]["failed_checks"][0]["id"] == check_id
