"""The review cascade's threshold boundaries and execution modes."""

import asyncio
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from imas_codex.standard_names.review import cascade
from imas_codex.standard_names.review.cascade import (
    route_docs,
    route_name,
)

_CALIBRATION_SHA256 = "8fd2b51e30887ab1dbed55688f38a154c1633985fb81717382b611fd254c7090"


def _decision(route, probability, failed_checks=()):
    from imas_codex.standard_names.review.checks import NAME_DIMENSIONS

    total = 0.85 if route == "accept" else 0.60
    return cascade.CascadeRoute(
        route,
        probability,
        failed_checks,
        dict.fromkeys(NAME_DIMENSIONS, 20 * total),
    )


def _assert_calibration_digest(artifact):
    assert artifact["content_sha256"] == _CALIBRATION_SHA256
    content = {key: value for key, value in artifact.items() if key != "content_sha256"}
    encoded = json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(encoded).hexdigest() == artifact["content_sha256"]


def test_calibration_artifact_digest_recomputes_from_committed_content():
    artifact = json.loads(Path(cascade._THRESHOLDS).read_text())
    assert artifact["source_split"] == "calibration"
    for axis in ("name", "docs"):
        assert artifact[axis]["unjudged_count"] == 0
        assert sum(artifact[axis]["route_counts"].values()) == artifact[axis]["n"]
    _assert_calibration_digest(artifact)


@pytest.mark.parametrize("field", ["accept_threshold", "score_weight"])
def test_calibration_digest_rejects_unrederived_edit(field):
    artifact = json.loads(Path(cascade._THRESHOLDS).read_text())
    altered = deepcopy(artifact)
    if field == "accept_threshold":
        altered["docs"]["accept_threshold"] += 0.01
    else:
        key = next(iter(altered["name"]["score_weights"]))
        altered["name"]["score_weights"][key] += 0.01
    with pytest.raises(AssertionError):
        _assert_calibration_digest(altered)


class _AggregateGraph:
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
        before_aggregate = statement.split("WITH sn, count(r)", 1)[0]
        excludes_shadow = (
            "coalesce(r.judge_kind, '') = 'jev'" in before_aggregate
            and "coalesce(r.is_canonical, false) = false" in before_aggregate
        )
        reviews = [
            review
            for review in self.reviews
            if not (
                excludes_shadow
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


@pytest.mark.parametrize("shadow_score", [None, 0.99])
def test_shadow_jev_row_does_not_change_review_aggregates(monkeypatch, shadow_score):
    from imas_codex.standard_names import graph_ops

    graph = _AggregateGraph()
    monkeypatch.setattr(graph_ops, "GraphClient", lambda: graph)
    assert graph_ops.update_review_aggregates(["temperature"]) == 1
    incumbent_aggregate = graph.aggregate
    graph.reviews.append(
        {"judge_kind": "jev", "is_canonical": False, "score": shadow_score}
    )
    assert graph_ops.update_review_aggregates(["temperature"]) == 1
    assert graph.aggregate == incumbent_aggregate == (2, 0.45, False)


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
        _decision("refine", 0.2, ("source_fidelity_unsupported_qualifier",)),
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

    route = _decision("refine", 0.1, ("source_fidelity_unsupported_qualifier",))
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


@pytest.fixture
def planted_thresholds(monkeypatch):
    thresholds = {
        axis: {
            "accept_threshold": 0.8,
            "reject_threshold": 0.2,
            "check_thresholds": {"named_failure": 0.6},
            "rubric": {"band_width": 0.05},
        }
        for axis in ("name", "docs")
    }
    monkeypatch.setattr(cascade, "_calibration", lambda: thresholds)
    from imas_codex.standard_names.review.checks import DOCS_DIMENSIONS, NAME_DIMENSIONS

    monkeypatch.setattr(
        cascade,
        "rubric_scores",
        lambda axis, judgment, fit: {
            dimension: judgment.get("quality", 0.9) * 20
            for dimension in (NAME_DIMENSIONS if axis == "name" else DOCS_DIMENSIONS)
        },
    )
    return thresholds


@pytest.mark.parametrize("route", [route_name, route_docs])
def test_accept_at_threshold(route, planted_thresholds):
    judgment = {"checks": {}}
    assert (
        route({**judgment, "quality": 0.75}, probability=0.2, min_score=0.75).route
        == "accept"
    )
    assert (
        route(
            {**judgment, "quality": 0.75 - 0.000001}, probability=0.99, min_score=0.75
        ).route
        == "chat_review"
    )


@pytest.mark.parametrize("route", [route_name, route_docs])
def test_refine_at_check_threshold(route, planted_thresholds):
    check_id, threshold = "named_failure", 0.6
    judgment = {
        "checks": {check_id: {"answered": True, "failure_probability": threshold}}
    }
    decision = route(judgment, probability=0.9)
    assert decision.route == "refine"
    assert decision.failed_checks == (check_id,)
    judgment["checks"][check_id]["failure_probability"] = threshold - 0.000001
    assert route(judgment, probability=0.9).route == "accept"


@pytest.mark.parametrize("route", [route_name, route_docs])
def test_below_band_refines(route, planted_thresholds):
    judgment = {"checks": {}}
    assert (
        route({**judgment, "quality": 0.70}, probability=0.2, min_score=0.75).route
        == "chat_review"
    )
    assert (
        route(
            {**judgment, "quality": 0.70 - 0.000001}, probability=0.99, min_score=0.75
        ).route
        == "refine"
    )
    assert (
        route({**judgment, "quality": 0.74}, probability=0.99, min_score=0.75).route
        == "chat_review"
    )


@pytest.mark.parametrize("route", [route_name, route_docs])
def test_default_threshold_is_the_persistence_consumers_default(
    route, planted_thresholds
):
    from imas_codex.standard_names.defaults import DEFAULT_MIN_SCORE

    assert (
        route({"quality": DEFAULT_MIN_SCORE, "checks": {}}, probability=0.2).route
        == "accept"
    )
    assert (
        route(
            {"quality": DEFAULT_MIN_SCORE - 0.01, "checks": {}}, probability=0.99
        ).route
        == "chat_review"
    )


def test_unsupported_claim_is_not_a_contradiction_veto(planted_thresholds):
    planted_thresholds["docs"]["check_thresholds"] = {"claim_contradicts_source": 1.0}
    judgment = {
        "quality": 0.96,
        "contradiction_veto": False,
        "checks": {
            "claim_contradicts_source": {"answered": True, "failure_probability": 1.0}
        },
    }
    assert route_docs(judgment, probability=0.9).route == "accept"
    judgment.update(
        contradiction_veto=True,
        contradiction_reason="source states the opposite sign",
        raw_claim_answers=[
            {"choice": "supports", "probabilities": {"contradicts": 0.3}}
        ],
    )
    decision = route_docs(judgment, probability=0.9)
    assert decision.route == "refine"
    assert decision.check_details["claim_contradicts_source"]["probability"] == 0.3
    assert decision.check_details["claim_contradicts_source"]["cutoff"] == 0.3


@pytest.mark.parametrize("quality,expected_route", [(0.52, "refine"), (0.96, "accept")])
def test_failed_check_is_recorded_even_when_it_is_not_a_veto(
    planted_thresholds, quality, expected_route
):
    check_id = "source_fidelity_unsupported_qualifier"
    planted_thresholds["name"]["check_thresholds"] = {check_id: 0.95}
    judgment = {
        "quality": quality,
        "checks": {
            check_id: {
                "answered": True,
                "failure_probability": 0.8,
                "answer": {"noul": 0.8},
            }
        },
    }
    route = route_name(judgment, probability=0.9)
    assert route.route == expected_route
    assert not route.failed_checks
    result = cascade._direct_result(
        route, "judge-model", 0.0, {"id": "temperature"}, "name"
    )
    failure = result["winning_comments_per_dim"]["failed_checks"][0]
    assert failure["id"] == check_id
    assert failure["probability"] == 0.8 and failure["cutoff"] == 0.5
    assert failure["veto"] is False
    assert failure["judgment_evidence"] == {"noul": 0.8}


@pytest.mark.parametrize("axis", ["name", "docs"])
@pytest.mark.parametrize("veto,quality", [(False, 0.86), (False, 0.52), (True, 0.96)])
def test_decided_review_stores_dimension_scores_and_their_mean(
    axis, veto, quality, planted_thresholds
):
    from imas_codex.standard_names.review.checks import DOCS_DIMENSIONS, NAME_DIMENSIONS

    check_id = (
        "source_fidelity_unsupported_qualifier"
        if axis == "name"
        else "description_false_of_some_binding"
    )
    planted_thresholds[axis]["check_thresholds"] = {check_id: 0.6}
    judgment = {
        "quality": quality,
        "checks": {
            check_id: {
                "answered": True,
                "failure_probability": 0.8 if veto else 0.1,
                "answer": {"noul": 0.8 if veto else 0.1},
            },
        },
    }
    route = cascade._route(axis, judgment, 0.9, min_score=0.75)
    item = {
        "id": "temperature",
        "source_paths": ["core_profiles/profiles_1d/electrons/temperature"],
    }
    result = cascade._direct_result(route, "judge-model", 0.0, item, axis)
    expected = NAME_DIMENSIONS if axis == "name" else DOCS_DIMENSIONS
    assert set(result["winning_scores"]) == expected
    assert all(0 <= value <= 20 for value in result["winning_scores"].values())
    assert result["winning_score"] == pytest.approx(
        sum(result["winning_scores"].values()) / 80
    )
    assert 0 < result["winning_score"] < 1
    assert (result["winning_score"] >= 0.75) == (route.route == "accept")
    if veto:
        failure = result["winning_comments_per_dim"]["failed_checks"][0]
        assert failure["probability"] == 0.8
        assert failure["cutoff"] == 0.6
        assert failure["judgment_evidence"] == {"noul": 0.8}
        assert failure["evidence"]
        assert route.raw_scores != route.scores


@pytest.mark.parametrize("axis", ["name", "docs"])
@pytest.mark.parametrize("kind", ["accept", "low", "veto"])
@pytest.mark.parametrize("min_score", [0.65, 0.75, 0.85])
def test_consumer_predicates_agree_with_rubric_route(
    monkeypatch, planted_thresholds, axis, kind, min_score
):
    """Execute persistence/export and the score predicate emitted by each claim."""
    import operator
    import re

    from imas_codex.standard_names import graph_ops
    from imas_codex.standard_names.export import _run_gate_c

    judgment = {
        "quality": min_score + 0.10 if kind != "low" else min_score - 0.10,
        "checks": {
            "named_failure": {
                "answered": True,
                "failure_probability": 0.9 if kind == "veto" else 0.1,
            }
        },
    }
    route = cascade._route(axis, judgment, 0.95, min_score=min_score)
    result = cascade._direct_result(
        route, "judge-model", 0.0, {"id": "temperature"}, axis
    )
    score = result["winning_score"]
    assert score == pytest.approx(sum(result["winning_scores"].values()) / 80)
    expected_accept = kind == "accept"
    assert (route.route == "accept") == expected_accept

    gc = MagicMock()
    gc.__enter__.return_value = gc
    gc.query.side_effect = [
        [{"chain_length": 0, "docs_chain_length": 0, "documentation": ""}],
        [{"id": "temperature"}],
    ]
    monkeypatch.setattr(graph_ops, "GraphClient", lambda: gc)
    monkeypatch.setattr(graph_ops, "write_reviews", lambda _rows: None)
    monkeypatch.setattr(graph_ops, "_doc_link_mismatches", lambda *_args: [])
    persist = getattr(graph_ops, "persist_reviewed_" + axis)
    stage = persist(
        sn_id="temperature",
        claim_token="token",
        score=score,
        scores=result["winning_scores"],
        model="judge-model",
        min_score=min_score,
        resolution_method=result["resolution_method"],
        reviewer_chain_size=3,
    )
    assert stage == ("accepted" if expected_accept else "reviewed")

    captured = {}

    def claim(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(graph_ops, "_claim_sn_atomic", claim)
    getattr(graph_ops, "claim_refine_" + axis + "_batch")(min_score=min_score)
    comparator = re.search(
        rf"sn\.reviewer_score_{axis}\s*(<|>=)\s*\$min_score",
        captured["eligibility_where"],
    )
    assert comparator is not None
    eligible = {"<": operator.lt, ">=": operator.ge}[comparator[1]](
        score, captured["query_params"]["min_score"]
    )
    assert eligible == (not expected_accept)
    _, exported, below, _ = _run_gate_c(
        [{"id": "temperature", "reviewer_score_name": score}],
        min_score,
        False,
        None,
    )
    assert bool(exported) == expected_accept
    assert below == int(not expected_accept)


def test_rubric_uses_only_a_dimensions_own_answers():
    from imas_codex.standard_names.review.checks import NAME_DIMENSIONS

    fit = {
        "offset": 0.0,
        "dimensions": {
            dim: {"intercept": 0.0, "weights": {"category": 1.0}}
            for dim in NAME_DIMENSIONS
        },
    }
    fit["dimensions"]["semantic"]["weights"][
        "source_fidelity_unsupported_qualifier"
    ] = 1.0
    from imas_codex.standard_names.review.jev_review import _DIMENSION_LEVELS

    judgment = {
        "categories": {
            dim: {"probabilities": {levels[0]: 0.6, levels[1]: 0.3, levels[2]: 0.1}}
            for dim, levels in _DIMENSION_LEVELS.items()
        },
        "checks": {
            "source_fidelity_unsupported_qualifier": {
                "answered": True,
                "failure_probability": 0.1,
            }
        },
    }
    first = cascade.rubric_scores("name", judgment, fit)
    judgment["checks"]["source_fidelity_unsupported_qualifier"][
        "failure_probability"
    ] = 0.9
    second = cascade.rubric_scores("name", judgment, fit)
    assert second["semantic"] < first["semantic"]
    assert {k: v for k, v in second.items() if k != "semantic"} == {
        k: v for k, v in first.items() if k != "semantic"
    }
    judgment["categories"]["grammar"] = None
    with pytest.raises(ValueError, match="incomplete rubric answers"):
        cascade.rubric_scores("name", judgment, fit)


@pytest.mark.asyncio
async def test_shadow_records_route_and_keeps_incumbent_result(monkeypatch):
    from imas_codex import settings

    records = []
    calls = []

    async def judgment(_item, _model):
        return _decision("accept", 0.95), {}, 0.0

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
        ("accept", [], 0.85),
        ("refine", [], 0.60),
        ("chat_review", [["primary"]], 0.3),
        ("chat_below_reject", [["primary"]], 0.3),
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
            _decision(route, 0.5, (check_id,) if route == "refine" else ()),
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
    if route.startswith("chat"):
        assert result["resolution_method"] == "jev_cascade_" + route


@pytest.mark.parametrize(
    "route,expected_stage", [("accept", "accepted"), ("refine", "reviewed")]
)
def test_deciding_result_reaches_name_stage(monkeypatch, route, expected_stage):
    from imas_codex.standard_names import graph_ops

    gc = MagicMock()
    gc.__enter__.return_value = gc
    gc.query.side_effect = [[{"chain_length": 0}], [{"id": "temperature"}]]
    monkeypatch.setattr(graph_ops, "GraphClient", lambda: gc)
    monkeypatch.setattr(graph_ops, "write_reviews", lambda _rows: None)
    decision = cascade._direct_result(
        _decision(
            route, 0.9, ("advisory_alias_collapsed",) if route == "refine" else ()
        ),
        "judge-model",
        0.0,
        {"id": "temperature"},
        "name",
    )
    stage = graph_ops.persist_reviewed_name(
        sn_id="temperature",
        claim_token="token",
        score=decision["winning_score"],
        comments_per_dim=decision["winning_comments_per_dim"],
        model="judge-model",
        min_score=0.75,
        resolution_method=decision["resolution_method"],
        reviewer_chain_size=3,
    )
    assert stage == expected_stage
    assert gc.query.call_args_list[1].kwargs["quorum_shortfall"] is None
    if route == "refine":
        assert (
            "sn.review_quorum_shortfall IS NULL"
            in graph_ops.REFINE_NAME_ELIGIBILITY_WHERE
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["lease", "judgment"])
@pytest.mark.parametrize("deciding", [False, True])
async def test_unjudged_review_keeps_full_incumbent(monkeypatch, failure, deciding):
    from imas_codex import settings

    stored = []
    calls = []

    async def fail_judgment(_item, _model):
        raise RuntimeError("Jev unavailable")

    async def store(*args, **kwargs):
        stored.append((args, kwargs))

    async def incumbent(*, models):
        calls.append(list(models))
        return {"winning_score": 0.3}

    class Manager:
        def reserve(self, *_args, **_kwargs):
            return None if failure == "lease" else MagicMock()

    monkeypatch.setattr(settings, "get_model", lambda _seat: "judge-model")
    monkeypatch.setattr(cascade, "_name_judgment", fail_judgment)
    monkeypatch.setattr(cascade, "_store_route", store)
    monkeypatch.setattr(cascade, "deciding_enabled", lambda: deciding)
    await cascade.adjudicate_review(
        "name",
        {"id": "temperature"},
        ["primary", "secondary"],
        incumbent,
        mgr=Manager(),
    )
    assert len(stored) == 1
    assert stored[0][0][3].route == "unjudged"
    assert ("lease" if failure == "lease" else "Jev unavailable") in stored[0][1][
        "reason"
    ]
    assert calls == [["primary", "secondary"]]


@pytest.mark.asyncio
@pytest.mark.parametrize("axis", ["name", "docs"])
@pytest.mark.parametrize("failure", ["lease", "call"])
async def test_unjudged_shadow_row_preserves_incumbent_and_aggregate(
    monkeypatch, axis, failure
):
    from imas_codex import settings
    from imas_codex.standard_names import graph_ops

    graph = _AggregateGraph()
    stored = []
    calls = []
    incumbent_result = {"winning_score": 0.4, "resolution_method": "quorum_consensus"}

    def write_reviews(rows):
        stored.extend(rows)
        graph.reviews.extend(rows)

    async def fail_judgment(_item, _model):
        raise RuntimeError("Jev unavailable")

    async def incumbent(*, models):
        calls.append(list(models))
        return incumbent_result

    class Manager:
        def reserve(self, *_args, **_kwargs):
            return None if failure == "lease" else MagicMock()

    monkeypatch.setattr(settings, "get_model", lambda _seat: "judge-model")
    monkeypatch.setattr(graph_ops, "GraphClient", lambda: graph)
    monkeypatch.setattr(graph_ops, "write_reviews", write_reviews)
    monkeypatch.setattr(
        cascade, "_name_judgment" if axis == "name" else "_docs_judgment", fail_judgment
    )
    monkeypatch.setattr(cascade, "deciding_enabled", lambda: False)
    graph_ops.update_review_aggregates(["temperature"])
    before = graph.aggregate
    result = await cascade.adjudicate_review(
        axis,
        {"id": "temperature"},
        ["primary", "secondary"],
        incumbent,
        mgr=Manager(),
    )
    assert result is incumbent_result
    assert calls == [["primary", "secondary"]]
    assert len(stored) == 1
    row = stored[0]
    assert row["judge_kind"] == "jev" and row["is_canonical"] is False
    assert row["score"] is None
    assert json.loads(row["judge_answers_json"])["cascade:route"] == "unjudged"
    reason = json.loads(row["comments"])["failure_reason"]
    assert ("lease" if failure == "lease" else "Jev unavailable") in reason
    assert "unjudged" not in cascade._calibration()[axis]["route_shares"]
    graph_ops.update_review_aggregates(["temperature"])
    assert graph.aggregate == before == (2, 0.45, False)


@pytest.mark.asyncio
async def test_empty_docs_judgment_is_unjudged(monkeypatch):
    from imas_codex.standard_names.review import jev_review

    monkeypatch.setattr(jev_review, "_default_checks", lambda: [])
    monkeypatch.setattr(
        jev_review,
        "build_docs_state",
        lambda *_args, **_kwargs: {"documentation_claims": []},
    )
    monkeypatch.setattr(jev_review, "docs_question_table", lambda *_args: {})

    async def empty(*_args, **_kwargs):
        return [], 0.0

    monkeypatch.setattr(jev_review, "decide_batch", empty)
    with pytest.raises(ValueError, match="no usable judgment"):
        await cascade._docs_judgment({"id": "temperature"}, "judge-model")


@pytest.mark.asyncio
async def test_empty_name_judgment_is_unjudged(monkeypatch):
    from types import SimpleNamespace

    from imas_codex.standard_names.review import jev_review

    async def unanswered(*_args, **_kwargs):
        return SimpleNamespace(items=[{"answered": False}], cost=0.0)

    monkeypatch.setattr(jev_review, "review_names", unanswered)
    with pytest.raises(ValueError, match="no usable judgment"):
        await cascade._name_judgment({"id": "temperature"}, "judge-model")


@pytest.mark.asyncio
async def test_refine_worker_receives_named_checks(monkeypatch):
    from imas_codex.standard_names import fanout, workers
    from imas_codex.standard_names.models import RefinedName

    route = _decision("refine", 0.1, ("source_fidelity_unsupported_qualifier",))
    result = cascade._direct_result(
        route, "judge-model", 0.0, {"id": "test_name"}, "name"
    )
    item = {
        "id": "test_name",
        "description": "A test quantity",
        "kind": "scalar",
        "unit": "eV",
        "physics_domain": ["core_profiles"],
        "source_paths": ["core_profiles/profiles_1d/electrons/temperature"],
        "claim_token": "token",
        "name_stage": "refining",
        "chain_length": 0,
        "chain_history": [],
        "reviewer_comments_per_dim_name": json.dumps(
            result["winning_comments_per_dim"]
        ),
    }
    context = {}
    trigger_calls = []

    def capture_trigger(**kwargs):
        trigger_calls.append(kwargs)
        return False, ""

    def capture_prompt(_template, values):
        context.update(values)
        return "rendered prompt"

    gc = MagicMock()
    gc.__enter__.return_value = gc
    gc.query.return_value = []
    manager = MagicMock()
    manager.reserve.return_value = MagicMock(reserved=1.0)
    refined = RefinedName(
        base_token="temperature",
        base_kind="quantity",
        description="d",
        kind="scalar",
        reason="better",
    )

    async def llm(*_args, **_kwargs):
        return refined, 0.0, {}

    monkeypatch.setattr(fanout, "should_trigger_fanout", capture_trigger)
    monkeypatch.setattr("imas_codex.llm.prompt_loader.render_prompt", capture_prompt)
    monkeypatch.setattr("imas_codex.discovery.base.llm.acall_llm_structured", llm)
    monkeypatch.setattr("imas_codex.graph.client.GraphClient", lambda: gc)
    monkeypatch.setattr("imas_codex.settings.get_model", lambda _seat: "mock")
    monkeypatch.setattr(workers, "_hybrid_search_neighbours", lambda *_args: [])
    monkeypatch.setattr(
        "imas_codex.standard_names.graph_ops.persist_refined_name",
        lambda **_kwargs: {"new_name": "new_name", "old_name": "test_name"},
    )
    await workers.process_refine_name_batch([item], manager, asyncio.Event())
    assert context["failed_checks"][0]["id"] == "source_fidelity_unsupported_qualifier"
    assert trigger_calls[0]["trigger_source"] == "checks"
    assert (
        trigger_calls[0]["check_results"]["source_fidelity_unsupported_qualifier"][
            "failure_probability"
        ]
        == 1.0
    )


class _OrderingManager:
    """Budget manager whose reservations always succeed."""

    def reserve(self, *_args, **_kwargs):
        return MagicMock()


def _ordering_incumbent(writes, *, produces):
    """An incumbent that persists its own rows, as the review pool does."""

    async def incumbent(*, models):
        if not produces:
            return None
        writes.append(
            [
                {"id": "temperature:incumbent-0", "judge_kind": "incumbent"},
                {"id": "temperature:incumbent-1", "judge_kind": "incumbent"},
            ]
        )
        return {"winning_score": 0.2, "resolution_method": "quorum_consensus"}

    return incumbent


@pytest.mark.asyncio
async def test_shadow_row_is_written_after_the_incumbents_rows(monkeypatch):
    from imas_codex import settings
    from imas_codex.standard_names import graph_ops

    writes = []

    async def judgment(_item, _model):
        return _decision("accept", 0.95), {}, 0.0

    monkeypatch.setattr(settings, "get_model", lambda _seat: "judge-model")
    monkeypatch.setattr(
        "imas_codex.discovery.base.judgment.decisions_key_present", lambda: True
    )
    monkeypatch.setattr(cascade, "_name_judgment", judgment)
    monkeypatch.setattr(cascade, "deciding_enabled", lambda: False)
    monkeypatch.setattr(
        graph_ops, "write_reviews", lambda rows, **_: writes.append(list(rows))
    )
    result = await cascade.adjudicate_review(
        "name",
        {"id": "temperature"},
        ["primary", "secondary"],
        _ordering_incumbent(writes, produces=True),
        mgr=_OrderingManager(),
    )
    assert [len(call) for call in writes] == [2, 1]
    assert all(row["judge_kind"] == "incumbent" for row in writes[0])
    assert writes[1][0]["judge_kind"] == "jev"
    assert writes[1][0]["resolution_role"] == "shadow"
    assert result["winning_score"] == 0.2


@pytest.mark.asyncio
async def test_failed_incumbent_cycle_writes_no_shadow_row(monkeypatch):
    from imas_codex import settings
    from imas_codex.standard_names import graph_ops

    writes = []

    async def judgment(_item, _model):
        return _decision("accept", 0.95), {}, 0.0

    monkeypatch.setattr(settings, "get_model", lambda _seat: "judge-model")
    monkeypatch.setattr(
        "imas_codex.discovery.base.judgment.decisions_key_present", lambda: True
    )
    monkeypatch.setattr(cascade, "_name_judgment", judgment)
    monkeypatch.setattr(cascade, "deciding_enabled", lambda: False)
    monkeypatch.setattr(
        graph_ops, "write_reviews", lambda rows, **_: writes.append(list(rows))
    )
    result = await cascade.adjudicate_review(
        "name",
        {"id": "temperature"},
        ["primary", "secondary"],
        _ordering_incumbent(writes, produces=False),
        mgr=_OrderingManager(),
    )
    assert result is None
    assert writes == []
