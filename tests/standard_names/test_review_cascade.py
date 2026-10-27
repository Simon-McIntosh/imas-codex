"""The review cascade's threshold boundaries and execution modes."""

import asyncio
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from imas_codex.standard_names.graph_store import (
    graph_catalog as _owning_graph_catalog,
    graph_cost_and_claims as _owning_graph_cost_and_claims,
    graph_doc_links as _owning_graph_doc_links,
    graph_foundation as _owning_graph_foundation,
    graph_refinement as _owning_graph_refinement,
    graph_review_persistence as _owning_graph_review_persistence,
    resolve_owned_name,
)
from imas_codex.standard_names.review import cascade
from imas_codex.standard_names.review.cascade import (
    route_docs,
    route_name,
)
from imas_codex.standard_names.worker_tasks import (
    worker_extraction_and_enrichment as _owning_worker_extraction_and_enrichment,
    worker_refine_names as _owning_worker_refine_names,
    worker_review_docs as _owning_worker_review_docs,
)

_CALIBRATION_SHA256 = "4e47220f157edee78fc6b0c1c98994606fce42be6a620a68f9f09980eccba428"


def _decision(route, probability, failed_checks=()):
    from imas_codex.standard_names.review.checks import NAME_DIMENSIONS

    total = 0.85 if route == "accept" else 0.60
    return cascade.CascadeRoute(
        route,
        probability,
        failed_checks,
        dict.fromkeys(NAME_DIMENSIONS, 20 * total),
        selected_score=probability,
        decision_score=total,
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
    monkeypatch.setattr(_owning_graph_foundation, "GraphClient", lambda: graph)
    assert _owning_graph_catalog.update_review_aggregates(["temperature"]) == 1
    incumbent_aggregate = graph.aggregate
    graph.reviews.append(
        {"judge_kind": "jev", "is_canonical": False, "score": shadow_score}
    )
    assert _owning_graph_catalog.update_review_aggregates(["temperature"]) == 1
    assert graph.aggregate == incumbent_aggregate == (2, 0.45, False)


@pytest.mark.asyncio
async def test_shadow_review_record_is_noncanonical(monkeypatch):
    import json

    from imas_codex.standard_names import graph_ops

    stored = []
    monkeypatch.setattr(
        _owning_graph_catalog, "write_reviews", lambda rows: stored.extend(rows)
    )
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
    note = await _owning_worker_review_docs._documentation_contradiction_note(
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
            "accept_threshold": 0.4,
            "score_kind": "logistic" if axis == "name" else "rubric",
            "rubric": {},
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


@pytest.mark.parametrize("axis", ["name", "docs"])
def test_mapped_score_crosses_exactly_at_matched_cutoff(axis, planted_thresholds):
    import math

    cutoff = planted_thresholds[axis]["accept_threshold"]
    for score in (math.nextafter(cutoff, 0), cutoff, math.nextafter(cutoff, 1)):
        route = cascade._route(axis, {"quality": score, "checks": {}}, score)
        result = cascade._direct_result(route, "mock", 0, {"id": "temperature"}, axis)
        assert (result["winning_score"] >= 0.85) == (score >= cutoff)
        assert (route.route == "accept") == (score >= cutoff)
        if score == cutoff:
            assert result["winning_score"] == 0.85


@pytest.mark.parametrize("cutoff", [0, 0.1, 0.4, 0.99, 1])
def test_decision_map_is_monotone_and_bounded(cutoff):
    import math

    scores = sorted(
        {
            0,
            0.1,
            0.5,
            0.9,
            1,
            cutoff,
            max(0, math.nextafter(cutoff, 0)),
            min(1, math.nextafter(cutoff, 1)),
        }
    )
    mapped = [cascade.map_decision_score(score, cutoff) for score in scores]
    assert mapped == sorted(mapped)
    assert all(0 <= value <= 1 for value in mapped)
    assert cascade.map_decision_score(cutoff, cutoff) == 0.85
    assert all(
        (value >= 0.85) == (score >= cutoff)
        for score, value in zip(scores, mapped, strict=True)
    )


@pytest.mark.parametrize("axis", ["name", "docs"])
def test_other_failed_checks_are_recorded_without_veto(axis, planted_thresholds):
    check_id = (
        "source_fidelity_unsupported_qualifier"
        if axis == "name"
        else "description_false_of_some_binding"
    )
    # A stale learned cutoff must not regain decision authority.
    planted_thresholds[axis]["check_thresholds"] = {check_id: 0.6}
    judgment = {
        "quality": 0.9,
        "checks": {
            check_id: {
                "answered": True,
                "failure_probability": 0.9,
                "answer": {"noul": 0.9},
            }
        },
    }
    route = cascade._route(axis, judgment, 0.9)
    assert route.route == "accept" and not route.failed_checks
    result = cascade._direct_result(
        route, "mock", 0, {"id": "temperature", "description": "Temperature"}, axis
    )
    failure = result["winning_comments_per_dim"]["failed_checks"][0]
    assert failure["id"] == check_id and failure["probability"] == 0.9
    assert failure["veto"] is False and failure["cutoff"] == 0.5


def test_unsupported_claim_is_not_a_contradiction_veto(planted_thresholds):
    judgment = {
        "quality": 0.96,
        "contradiction_veto": False,
        "checks": {
            "claim_contradicts_source": {"answered": True, "failure_probability": 1.0},
        },
    }
    clear = route_docs(judgment, probability=0.9)
    assert clear.route == "accept"
    judgment.update(
        contradiction_veto=True,
        contradiction_reason="source states the opposite sign",
        raw_claim_answers=[
            {"choice": "supports", "probabilities": {"contradicts": 0.3}}
        ],
    )
    decision = route_docs(judgment, probability=0.9)
    assert decision.route == "refine"
    assert decision.decision_score >= 0.85
    assert [hold["kind"] for hold in decision.holds] == ["cascade_contradiction"]
    assert decision.scores == clear.scores == decision.raw_scores
    assert decision.check_details["claim_contradicts_source"]["probability"] == 0.3
    assert decision.check_details["claim_contradicts_source"]["cutoff"] == 0.3


@pytest.mark.parametrize("axis", ["name", "docs"])
@pytest.mark.parametrize("veto,quality", [(False, 0.86), (False, 0.2), (True, 0.96)])
def test_decided_review_preserves_rubric_and_failed_checks(
    axis, veto, quality, planted_thresholds
):
    from imas_codex.standard_names.review.checks import DOCS_DIMENSIONS, NAME_DIMENSIONS

    check_id = (
        "source_fidelity_unsupported_qualifier"
        if axis == "name"
        else "description_false_of_some_binding"
    )
    judgment = {
        "quality": quality,
        "contradiction_veto": veto,
        "checks": {
            check_id: {
                "answered": True,
                "failure_probability": 0.8 if veto else 0.1,
                "answer": {"noul": 0.8 if veto else 0.1},
            }
        },
    }
    route = cascade._route(axis, judgment, quality, min_score=0.75)
    item = {
        "id": "temperature",
        "description": "Temperature",
        "source_paths": ["core_profiles/profiles_1d/electrons/temperature"],
    }
    result = cascade._direct_result(route, "mock", 0, item, axis)
    dimensions = NAME_DIMENSIONS if axis == "name" else DOCS_DIMENSIONS
    assert result["winning_scores"] == dict.fromkeys(dimensions, quality * 20)
    assert route.scores == route.raw_scores
    # A decided route accepts on its judged score only with no veto hold: a
    # contradiction veto refuses the item at any score, and the score it judged
    # is kept.
    assert (route.route == "accept") == (
        result["winning_score"] >= 0.75 and not route.holds
    )
    if veto:
        failures = {
            f["id"]: f for f in result["winning_comments_per_dim"]["failed_checks"]
        }
        assert failures[check_id]["probability"] == 0.8
        assert failures[check_id]["judgment_evidence"] == {"noul": 0.8}
        assert failures[check_id]["evidence"]


@pytest.mark.parametrize("axis", ["name", "docs"])
@pytest.mark.parametrize("kind", ["accept", "low", "veto"])
@pytest.mark.parametrize("min_score", [0.65, 0.75, 0.85])
def test_consumer_predicates_agree_with_rubric_route(
    monkeypatch, planted_thresholds, axis, kind, min_score
):
    """Execute persistence/export and the score predicate emitted by each claim."""
    import re
    from contextlib import contextmanager

    from imas_codex.standard_names import graph_ops
    from imas_codex.standard_names.export import _run_gate_c

    raw = 0.1 if kind == "low" else 0.4
    judgment = {"quality": raw, "contradiction_veto": False, "checks": {}}
    judgment["contradiction_veto"] = kind == "veto"
    route = cascade._route(axis, judgment, raw, min_score=min_score)
    result = cascade._direct_result(route, "mock", 0, {"id": "temperature"}, axis)
    score = result["winning_score"]
    expected_accept = kind == "accept" or (axis == "name" and kind == "veto")
    assert (route.route == "accept") == expected_accept

    # Acceptance is decided by the product predicate, never by a second copy of
    # the rule written here: the docs axis only accepts a passing score with no
    # open documentation hold, so a contradiction veto refuses the item however
    # high its judged score is.
    from imas_codex.standard_names.review import holds as review_holds

    class _Holds:
        def __init__(self, rows):
            self._rows = rows

        def query(self, *_args, **_kwargs):
            return self._rows

    open_rows = (
        [{"hold": {"id": "hold", "axis": "docs", "status": "open"}}]
        if route.holds
        else []
    )
    if axis == "docs":
        accepted = review_holds.docs_accepted(
            score, min_score, "temperature", graph=_Holds(open_rows)
        )
    else:
        accepted = score >= min_score
    assert accepted == expected_accept

    class _PersistGraph:
        """Answer both query boundaries the two persistence writers use.

        The name writer reads and writes at the client level; the docs writer
        runs its claim check, its hold reads and its score write inside one
        ``gc.session()`` transaction, so a client-level ``query`` side effect
        never reaches it. This answers by statement, and it is the open-hold
        read that settles a passing-but-vetoed document.
        """

        def __init__(self, holds):
            self._holds = holds

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        @contextmanager
        def session(self):
            yield self

        def begin_transaction(self):
            return self

        def commit(self):
            pass

        def rollback(self):
            return None

        def close(self):
            pass

        def _answer(self, statement):
            if "AS cleared" in statement:
                return [{"cleared": 0}]
            if "HAS_HOLD" in statement:
                return list(self._holds)
            if "AS chain_length" in statement or "AS docs_chain_length" in statement:
                return [
                    {
                        "chain_length": 0,
                        "docs_chain_length": 0,
                        "documentation": "",
                    }
                ]
            return [{"id": "temperature"}]

        def query(self, statement, **_params):
            return self._answer(statement)

        def run(self, statement, **_params):
            return self._answer(statement)

    gc = _PersistGraph(open_rows)
    monkeypatch.setattr(_owning_graph_foundation, "GraphClient", lambda: gc)
    monkeypatch.setattr(_owning_graph_catalog, "write_reviews", lambda _rows: None)
    monkeypatch.setattr(
        _owning_graph_doc_links, "_doc_link_mismatches", lambda *_args: []
    )
    stage = resolve_owned_name("persist_reviewed_" + axis)(
        sn_id="temperature",
        claim_token="token",
        score=score,
        scores=result["winning_scores"],
        model="mock",
        min_score=min_score,
        resolution_method=result["resolution_method"],
        reviewer_chain_size=3,
    )
    assert stage == ("accepted" if expected_accept else "reviewed")
    captured = {}

    def claim(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(_owning_graph_cost_and_claims, "_claim_sn_atomic", claim)
    resolve_owned_name("claim_refine_" + axis + "_batch")(min_score=min_score)
    comparator = re.search(
        rf"sn\.reviewer_score_{axis}\s*(<|>=)\s*\$min_score",
        captured["eligibility_where"],
    )
    assert comparator is not None
    # A held document keeps its judged score, so the score clause alone would
    # never re-offer a passing-but-vetoed document. The docs claim predicate
    # admits an open hold through the hold owner's own clause instead of a
    # second copy of the rule written here.
    if axis == "docs":
        assert review_holds.has_open_docs_hold_where() in captured["eligibility_where"]
    _, exported, below, _ = _run_gate_c(
        [{"id": "temperature", "reviewer_score_name": score}], min_score, False, None
    )
    # Gate C is a score gate: it refuses exactly the scores below the bound and
    # keeps a held document's judged score. The export withholds that document
    # at its stage gate, which is the stage the persistence arm above stored.
    assert bool(exported) == (score >= min_score)
    assert below == int(score < min_score)

    from imas_codex.standard_names.export import _classify_export_population

    published, _ = _classify_export_population(
        [
            {
                "id": "temperature",
                "name_stage": "accepted",
                "validation_status": "valid",
                "_validation_observed_at": "2026-10-23T00:00:00Z",
                "review_quorum_shortfall": None,
                "docs_stage": stage,
                "docs_review_quorum_shortfall": None,
                "physics_domain": "general",
                "_has_derived_producer": True,
            }
        ],
        domain=None,
        names_only=False,
    )
    assert bool(published) == expected_accept


@pytest.mark.asyncio
async def test_deciding_name_pass_does_not_reconcile_docs_holds(monkeypatch):
    """The deciding entrypoint passes its axis to hold reconciliation."""
    from imas_codex import settings
    from imas_codex.standard_names.review import holds

    route = _decision("accept", 0.9)
    reconciled = []

    async def judgment(_item, _model):
        return route, {}, 0.0

    async def store(*_args):
        pass

    async def audit(*_args, **_kwargs):
        pass

    async def chat_reviewer(*, models):
        pytest.fail(f"deciding route called the chat reviewer: {models}")

    def sync(*args, **kwargs):
        reconciled.append((args, kwargs))
        return []

    monkeypatch.setattr(
        "imas_codex.discovery.base.judgment.decisions_key_present", lambda: True
    )
    monkeypatch.setattr(settings, "get_model", lambda _seat: "judge-model")
    monkeypatch.setattr(cascade, "_name_judgment", judgment)
    monkeypatch.setattr(cascade, "_store_route", store)
    monkeypatch.setattr(cascade, "_run_panel_audit", audit)
    monkeypatch.setattr(cascade, "deciding_enabled", lambda _axis: True)
    monkeypatch.setattr(holds, "sync_holds", sync)
    manager = MagicMock()
    manager.reserve.return_value = MagicMock()

    result = await cascade.adjudicate_review(
        "name", {"id": "temperature"}, ["incumbent"], chat_reviewer, mgr=manager
    )
    assert result is not None
    assert reconciled == []


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

    async def chat_reviewer(*, models):
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
    monkeypatch.setattr(cascade, "deciding_enabled", lambda _axis: False)
    result = await cascade.adjudicate_review(
        "name",
        {"id": "temperature"},
        ["primary", "secondary"],
        chat_reviewer,
        mgr=Manager(),
    )
    assert records == ["accept"]
    assert calls == [["primary"]]
    assert result["winning_score"] == 0.2


@pytest.mark.asyncio
@pytest.mark.parametrize("axis", ["name", "docs"])
@pytest.mark.parametrize("raw,expected_route", [(0.4, "accept"), (0.39, "refine")])
async def test_deciding_never_calls_chat(
    monkeypatch, planted_thresholds, axis, raw, expected_route
):
    from imas_codex import settings

    calls = []

    async def judgment(_item, _model):
        return cascade._route(axis, {"quality": raw, "checks": {}}, raw), {}, 0.0

    async def store(*_args):
        pass

    async def chat_reviewer(*, models):
        calls.append(models)
        return {"winning_score": 0.1}

    class Manager:
        def reserve(self, *_args, **_kwargs):
            return MagicMock()

    monkeypatch.setattr(settings, "get_model", lambda _seat: "judge-model")
    monkeypatch.setattr(
        "imas_codex.discovery.base.judgment.decisions_key_present", lambda: True
    )
    monkeypatch.setattr(
        cascade, "_name_judgment" if axis == "name" else "_docs_judgment", judgment
    )
    monkeypatch.setattr(cascade, "_store_route", store)
    monkeypatch.setattr(cascade, "deciding_enabled", lambda _axis: True)
    result = await cascade.adjudicate_review(
        axis,
        {"id": "temperature"},
        ["primary", "secondary"],
        chat_reviewer,
        mgr=Manager(),
    )
    assert calls == []
    assert result["resolution_method"] == "jev_cascade_" + expected_route
    assert (result["winning_score"] >= 0.85) == (expected_route == "accept")


@pytest.mark.parametrize(
    "route,expected_stage", [("accept", "accepted"), ("refine", "reviewed")]
)
def test_deciding_result_reaches_name_stage(monkeypatch, route, expected_stage):
    from imas_codex.standard_names import graph_ops

    gc = MagicMock()
    gc.__enter__.return_value = gc
    gc.query.side_effect = [[{"chain_length": 0}], [{"id": "temperature"}]]
    monkeypatch.setattr(_owning_graph_foundation, "GraphClient", lambda: gc)
    monkeypatch.setattr(_owning_graph_catalog, "write_reviews", lambda _rows: None)
    decision = cascade._direct_result(
        _decision(
            route, 0.9, ("advisory_alias_collapsed",) if route == "refine" else ()
        ),
        "judge-model",
        0.0,
        {"id": "temperature"},
        "name",
    )
    stage = _owning_graph_review_persistence.persist_reviewed_name(
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
            in _owning_graph_refinement.REFINE_NAME_ELIGIBILITY_WHERE
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

    async def chat_reviewer(*, models):
        calls.append(list(models))
        return {"winning_score": 0.3}

    class Manager:
        def reserve(self, *_args, **_kwargs):
            return None if failure == "lease" else MagicMock()

    monkeypatch.setattr(settings, "get_model", lambda _seat: "judge-model")
    monkeypatch.setattr(cascade, "_name_judgment", fail_judgment)
    monkeypatch.setattr(cascade, "_store_route", store)
    monkeypatch.setattr(cascade, "deciding_enabled", lambda _axis: deciding)
    await cascade.adjudicate_review(
        "name",
        {"id": "temperature"},
        ["primary", "secondary"],
        chat_reviewer,
        mgr=Manager(),
    )
    assert len(stored) == 1
    assert stored[0][0][3].route == "unjudged"
    assert ("lease" if failure == "lease" else "Jev unavailable") in stored[0][1][
        "reason"
    ]
    assert calls == [["primary"]]


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

    async def chat_reviewer(*, models):
        calls.append(list(models))
        return incumbent_result

    class Manager:
        def reserve(self, *_args, **_kwargs):
            return None if failure == "lease" else MagicMock()

    monkeypatch.setattr(settings, "get_model", lambda _seat: "judge-model")
    monkeypatch.setattr(_owning_graph_foundation, "GraphClient", lambda: graph)
    monkeypatch.setattr(_owning_graph_catalog, "write_reviews", write_reviews)
    monkeypatch.setattr(
        cascade, "_name_judgment" if axis == "name" else "_docs_judgment", fail_judgment
    )
    monkeypatch.setattr(cascade, "deciding_enabled", lambda _axis: False)
    _owning_graph_catalog.update_review_aggregates(["temperature"])
    before = graph.aggregate
    result = await cascade.adjudicate_review(
        axis,
        {"id": "temperature"},
        ["primary", "secondary"],
        chat_reviewer,
        mgr=Manager(),
    )
    assert result is incumbent_result
    assert calls == [["primary"]]
    assert len(stored) == 1
    row = stored[0]
    assert row["judge_kind"] == "jev" and row["is_canonical"] is False
    assert row["score"] is None
    assert json.loads(row["judge_answers_json"])["cascade:route"] == "unjudged"
    reason = json.loads(row["comments"])["failure_reason"]
    assert ("lease" if failure == "lease" else "Jev unavailable") in reason
    assert "unjudged" not in cascade._calibration()[axis]["route_shares"]
    _owning_graph_catalog.update_review_aggregates(["temperature"])
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
    monkeypatch.setattr(
        _owning_worker_extraction_and_enrichment,
        "_hybrid_search_neighbours",
        lambda *_args: [],
    )
    monkeypatch.setattr(
        "imas_codex.standard_names.graph_store.graph_refinement.persist_refined_name",
        lambda **_kwargs: {"new_name": "new_name", "old_name": "test_name"},
    )
    await _owning_worker_refine_names.process_refine_name_batch(
        [item], manager, asyncio.Event()
    )
    assert context["failed_checks"][0]["id"] == "source_fidelity_unsupported_qualifier"
    assert set(trigger_calls[0]) == {"checks", "check_results", "check_threshold"}
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

    async def chat_reviewer(*, models):
        if not produces:
            return None
        writes.append(
            [
                {"id": "temperature:incumbent-0", "judge_kind": "incumbent"},
                {"id": "temperature:incumbent-1", "judge_kind": "incumbent"},
            ]
        )
        return {"winning_score": 0.2, "resolution_method": "quorum_consensus"}

    return chat_reviewer


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
    monkeypatch.setattr(cascade, "deciding_enabled", lambda _axis: False)
    monkeypatch.setattr(
        _owning_graph_catalog,
        "write_reviews",
        lambda rows, **_: writes.append(list(rows)),
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
    monkeypatch.setattr(cascade, "deciding_enabled", lambda _axis: False)
    monkeypatch.setattr(
        _owning_graph_catalog,
        "write_reviews",
        lambda rows, **_: writes.append(list(rows)),
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
