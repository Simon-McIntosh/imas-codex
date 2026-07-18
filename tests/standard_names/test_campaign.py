"""Tests for the documentation-refinement campaign engine.

Selection is pure (no graph); the runner's graph interactions are injected
callables, so the whole engine is exercised without a live Neo4j or LLM.
"""

from __future__ import annotations

import json

import pytest

from imas_codex.standard_names.campaign import (
    PROSE_CLASSES,
    BatchOutcome,
    CampaignBudget,
    CampaignRunner,
    CampaignSelection,
    CampaignSpec,
    CampaignTarget,
    ConvergenceThresholds,
    build_manifest,
    default_audit_revalidate,
    default_clear_quarantine,
    default_fetch_refreshed,
    default_revalidate,
    evaluate_convergence,
    match_target,
    measure_batch,
    plan_batches,
    select_targets,
    stratified_pilot,
    write_manifest,
)

# ── Spec parsing ──────────────────────────────────────────────────────────────


class TestCampaignSpecParse:
    def test_prose_selects_all_prose_classes(self):
        spec = CampaignSpec.parse("prose")
        assert set(spec.prose_classes) == set(PROSE_CLASSES)
        assert spec.audit_categories == ()
        assert spec.include_quarantined is False

    def test_prose_class_selects_one(self):
        spec = CampaignSpec.parse("prose:typical_values")
        assert spec.prose_classes == ("typical_values",)

    def test_unknown_prose_class_raises(self):
        with pytest.raises(ValueError, match="unknown prose class"):
            CampaignSpec.parse("prose:not_a_class")

    def test_audit_any_uses_empty_marker(self):
        spec = CampaignSpec.parse("audit")
        assert spec.audit_categories == ("",)

    def test_audit_substring(self):
        spec = CampaignSpec.parse("audit:decomposition")
        assert spec.audit_categories == ("decomposition",)

    def test_quarantined(self):
        spec = CampaignSpec.parse("quarantined")
        assert spec.include_quarantined is True
        assert spec.prose_classes == ()

    def test_all_selects_everything(self):
        spec = CampaignSpec.parse("all")
        assert set(spec.prose_classes) == set(PROSE_CLASSES)
        assert spec.audit_categories == ("",)
        assert spec.include_quarantined is True

    def test_composite_spec(self):
        spec = CampaignSpec.parse("prose:typical_values,audit:latex,quarantined")
        assert spec.prose_classes == ("typical_values",)
        assert spec.audit_categories == ("latex",)
        assert spec.include_quarantined is True

    def test_empty_spec_raises(self):
        with pytest.raises(ValueError):
            CampaignSpec.parse("")

    def test_unknown_token_raises(self):
        with pytest.raises(ValueError, match="unknown campaign predicate"):
            CampaignSpec.parse("bogus")

    def test_describe_is_human_readable(self):
        spec = CampaignSpec.parse("prose:typical_values,audit:latex")
        text = spec.describe()
        assert "typical_values" in text
        assert "latex" in text


# ── match_target (pure selection) ─────────────────────────────────────────────


def _row(**kw):
    base = {
        "id": "electron_temperature",
        "name": "electron_temperature",
        "description": "Electron temperature.",
        "documentation": "The electron temperature.",
        "validation_issues": None,
        "validation_status": "valid",
        "quarantine_reason": None,
        "docs_stage": "accepted",
    }
    base.update(kw)
    return base


class TestMatchTarget:
    def test_typical_values_prose_matches(self):
        row = _row(
            documentation="Typically on the order of 100 eV in the core.",
        )
        target = match_target(row, CampaignSpec.parse("prose:typical_values"))
        assert target is not None
        assert "prose:typical_values" in target.matched_predicates

    def test_estimator_recipe_prose_matches(self):
        row = _row(
            documentation="This quantity is computed as the ratio of A to B.",
        )
        target = match_target(row, CampaignSpec.parse("prose:estimator_recipe"))
        assert target is not None
        assert "prose:estimator_recipe" in target.matched_predicates

    def test_clean_docs_do_not_match_prose(self):
        row = _row(documentation="The electron temperature scalar field.")
        assert match_target(row, CampaignSpec.parse("prose")) is None

    def test_audit_category_matches_check_name(self):
        row = _row(
            validation_issues=[
                "audit:decomposition_audit: physical_base contains closed-vocab token",
                "audit:latex_def_check: symbol $x$ lacks a definition",
            ],
        )
        target = match_target(row, CampaignSpec.parse("audit:decomposition"))
        assert target is not None
        assert target.matched_predicates["audit:decomposition"] == [
            "decomposition_audit"
        ]

    def test_audit_any_matches_all_findings(self):
        row = _row(
            validation_issues=["audit:latex_def_check: symbol lacks def"],
        )
        target = match_target(row, CampaignSpec.parse("audit"))
        assert target is not None
        assert "audit" in target.matched_predicates

    def test_quarantined_selects_on_its_own(self):
        row = _row(
            validation_status="quarantined",
            quarantine_reason="latex_def_check failure",
        )
        target = match_target(row, CampaignSpec.parse("quarantined"))
        assert target is not None
        assert target.quarantined is True
        assert "quarantined" in target.matched_predicates

    def test_physics_domain_flows_to_target(self):
        row = _row(documentation="Typically 3 T.", physics_domain="equilibrium")
        target = match_target(row, CampaignSpec.parse("prose"))
        assert target.physics_domain == "equilibrium"

    def test_no_predicate_match_returns_none(self):
        row = _row(validation_issues=["audit:latex_def_check: x"])
        assert match_target(row, CampaignSpec.parse("audit:decomposition")) is None

    def test_multiple_predicates_recorded(self):
        row = _row(
            documentation="Typically ~5 keV.",
            validation_issues=["audit:latex_def_check: x"],
            validation_status="quarantined",
        )
        target = match_target(row, CampaignSpec.parse("all"))
        assert set(target.matched_predicates) >= {
            "prose:typical_values",
            "audit",
            "quarantined",
        }


# ── select_targets against a mock graph ───────────────────────────────────────


class _FakeGraph:
    """Minimal GraphClient double: canned SELECT rows + recorded writes."""

    def __init__(self, rows):
        self._rows = rows
        self.writes: list[tuple[str, dict]] = []

    def query(self, cypher, **params):
        if (
            "WHERE sn.name_stage = 'accepted'" in cypher
            and "RETURN sn.id AS id" in cypher
        ):
            return list(self._rows)
        self.writes.append((cypher, params))
        return [{"n": len(params.get("ids", []))}]


class TestSelectTargets:
    def test_selects_and_counts_per_predicate(self):
        rows = [
            _row(id="a", documentation="Typically 3 T."),
            _row(id="b", validation_issues=["audit:latex_def_check: x"]),
            _row(id="c", documentation="A clean field."),
        ]
        gc = _FakeGraph(rows)
        sel = select_targets(gc, CampaignSpec.parse("all"))
        assert sel.total == 2
        assert set(sel.ids) == {"a", "b"}
        assert sel.per_predicate.get("prose:typical_values") == 1
        assert sel.per_predicate.get("audit") == 1

    def test_limit_caps_selection(self):
        rows = [_row(id=f"n{i}", documentation="Typically hot.") for i in range(5)]
        gc = _FakeGraph(rows)
        sel = select_targets(gc, CampaignSpec.parse("prose"), limit=2)
        assert sel.total == 2


# ── Manifest ───────────────────────────────────────────────────────────────────


class TestManifest:
    def _selection(self, n=50):
        targets = [
            CampaignTarget(
                id=f"n{i}",
                name=f"n{i}",
                matched_predicates={"prose:typical_values": ["1 match(es)"]},
            )
            for i in range(n)
        ]
        return CampaignSelection(spec=CampaignSpec.parse("prose"), targets=targets)

    def test_manifest_counts_and_sample(self):
        m = build_manifest(self._selection(50), sample_size=20, batch_size=10)
        assert m["total"] == 50
        assert m["batch_plan"] == {"batch_size": 10, "n_batches": 5}
        assert len(m["sample"]) == 20
        assert m["spec"] == "prose"

    def test_manifest_sample_is_deterministic(self):
        sel = self._selection(50)
        a = build_manifest(sel, seed=7)
        b = build_manifest(sel, seed=7)
        assert [s["id"] for s in a["sample"]] == [s["id"] for s in b["sample"]]

    def test_manifest_carries_per_domain_and_pilot_marker(self):
        targets = [
            _target(i, domain, "audit")
            for domain in ["equilibrium", "transport"]
            for i in range(2)
        ]
        sel = CampaignSelection(spec=CampaignSpec.parse("all"), targets=targets)
        m = build_manifest(sel, sample_size=4, batch_size=10, pilot_from=2332)
        assert m["per_domain"] == {"equilibrium": 2, "transport": 2}
        assert m["pilot"] == {"n": 4, "from_total": 2332}
        assert all("physics_domain" in s for s in m["sample"])

    def test_write_manifest_roundtrip(self, tmp_path):
        m = build_manifest(self._selection(3), sample_size=3)
        path = write_manifest(m, tmp_path / "sub" / "manifest.json")
        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded["total"] == 3


# ── Stratified pilot ─────────────────────────────────────────────────────────


def _target(i, domain, cls):
    return CampaignTarget(
        id=f"{domain}_{cls}_{i}",
        name=f"{domain}_{cls}_{i}",
        matched_predicates={cls: ["1 match(es)"]},
        physics_domain=domain,
    )


class TestStratifiedPilot:
    def _selection(self):
        targets = []
        classes = ["prose:typical_values", "prose:estimator_recipe", "audit"]
        for domain in ["equilibrium", "transport", "mhd", "heating", "edge"]:
            for cls in classes:
                for i in range(4):
                    targets.append(_target(i, domain, cls))
        return CampaignSelection(spec=CampaignSpec.parse("all"), targets=targets)

    def test_every_domain_represented(self):
        pilot = stratified_pilot(self._selection(), 25)
        assert pilot.total == 25
        domains = {t.physics_domain for t in pilot.targets}
        assert domains == {"equilibrium", "transport", "mhd", "heating", "edge"}

    def test_defect_classes_mixed_within_domain(self):
        pilot = stratified_pilot(self._selection(), 25)
        eq = [t for t in pilot.targets if t.physics_domain == "equilibrium"]
        classes = {t.predicate_keys[0] for t in eq}
        assert len(classes) > 1

    def test_deterministic_for_seed(self):
        a = stratified_pilot(self._selection(), 25, seed=3)
        b = stratified_pilot(self._selection(), 25, seed=3)
        assert a.ids == b.ids

    def test_n_at_least_total_returns_selection(self):
        sel = CampaignSelection(
            spec=CampaignSpec.parse("all"),
            targets=[_target(0, "mhd", "audit")],
        )
        assert stratified_pilot(sel, 25) is sel

    def test_nonpositive_n_raises(self):
        with pytest.raises(ValueError):
            stratified_pilot(self._selection(), 0)

    def test_preserves_spec(self):
        pilot = stratified_pilot(self._selection(), 5)
        assert pilot.spec.raw == "all"


# ── Batching ────────────────────────────────────────────────────────────────────


class TestPlanBatches:
    def test_even_split(self):
        assert plan_batches(["a", "b", "c", "d"], 2) == [["a", "b"], ["c", "d"]]

    def test_short_last_batch(self):
        assert plan_batches(["a", "b", "c"], 2) == [["a", "b"], ["c"]]

    def test_zero_batch_size_raises(self):
        with pytest.raises(ValueError):
            plan_batches(["a"], 0)


# ── Convergence gate ─────────────────────────────────────────────────────────────


class TestConvergence:
    def test_converged_batch_passes(self):
        outcome = BatchOutcome(batch_index=0, requested=10, touched=10, accepted=10)
        ok, reasons = evaluate_convergence(outcome, ConvergenceThresholds())
        assert ok is True
        assert reasons == []

    def test_low_acceptance_halts(self):
        outcome = BatchOutcome(batch_index=0, requested=10, touched=10, accepted=5)
        ok, reasons = evaluate_convergence(outcome, ConvergenceThresholds())
        assert ok is False
        assert any("acceptance" in r for r in reasons)

    def test_banned_prose_reintroduction_halts(self):
        outcome = BatchOutcome(
            batch_index=0,
            requested=10,
            touched=10,
            accepted=10,
            reintroduced_ids=["bad_name"],
        )
        ok, reasons = evaluate_convergence(outcome, ConvergenceThresholds())
        assert ok is False
        assert any("banned prose" in r for r in reasons)

    def test_name_drift_halts_docs_only(self):
        outcome = BatchOutcome(
            batch_index=0,
            requested=10,
            touched=9,
            accepted=9,
            name_drift=["renamed_away"],
        )
        ok, reasons = evaluate_convergence(outcome, ConvergenceThresholds())
        assert ok is False
        assert any("drift" in r for r in reasons)

    def test_name_drift_allowed_when_configured(self):
        outcome = BatchOutcome(
            batch_index=0,
            requested=10,
            touched=10,
            accepted=10,
            name_drift=["x"],
        )
        thresholds = ConvergenceThresholds(allow_name_drift=True)
        ok, _ = evaluate_convergence(outcome, thresholds)
        assert ok is True


# ── measure_batch (pure) ──────────────────────────────────────────────────────


class TestMeasureBatch:
    def test_counts_accepted_and_reintroduction_and_drift(self):
        refreshed = [
            {
                "id": "ok",
                "name_stage": "accepted",
                "docs_stage": "accepted",
                "description": "Clean.",
                "documentation": "A clean scalar field.",
            },
            {
                "id": "reintro",
                "name_stage": "accepted",
                "docs_stage": "accepted",
                "description": "Typically ~5 keV.",
                "documentation": "Typically hot.",
            },
            {
                "id": "drifted",
                "name_stage": "superseded",
                "docs_stage": "accepted",
                "description": "x",
                "documentation": "y",
            },
        ]
        # "missing" is in the batch but absent from refreshed rows → drift.
        outcome = measure_batch(
            refreshed, ["ok", "reintro", "drifted", "missing"], batch_index=2
        )
        assert outcome.batch_index == 2
        assert outcome.requested == 4
        assert outcome.touched == 2  # ok + reintro
        assert outcome.accepted == 2
        assert outcome.reintroduced_ids == ["reintro"]
        assert set(outcome.name_drift) == {"drifted", "missing"}
        assert outcome.accept_rate == 1.0

    def test_empty_touched_accept_rate_is_one(self):
        outcome = measure_batch([], ["gone"], batch_index=0)
        assert outcome.touched == 0
        assert outcome.accept_rate == 1.0
        assert outcome.name_drift == ["gone"]


# ── Default graph writes ───────────────────────────────────────────────────────


class TestDefaultGraphOps:
    def test_clear_quarantine_issues_expected_write(self):
        gc = _FakeGraph([])
        n = default_clear_quarantine(gc, ["a", "b"])
        assert n == 2
        cypher, params = gc.writes[-1]
        assert "SET sn.validation_status = 'pending'" in cypher
        assert params["ids"] == ["a", "b"]

    def test_clear_quarantine_noop_on_empty(self):
        gc = _FakeGraph([])
        assert default_clear_quarantine(gc, []) == 0
        assert gc.writes == []

    def test_revalidate_requarantines_and_confirms(self):
        gc = _FakeGraph([])
        out = default_revalidate(gc, ["bad"], ["good1", "good2"])
        assert out["requarantined"] == 1
        assert out["confirmed"] == 2
        joined = " ".join(c for c, _ in gc.writes)
        assert "SET sn.validation_status = 'quarantined'" in joined
        assert "SET sn.validation_status = 'valid'" in joined

    def test_fetch_refreshed_noop_on_empty(self):
        gc = _FakeGraph([])
        assert default_fetch_refreshed(gc, []) == []

    def test_audit_revalidate_clears_stamp_before_scoped_drain(self):
        gc = _FakeGraph([])
        calls: dict[str, list[str]] = {}

        def fake_drain(ids):
            # The stamp must already be cleared when the drain re-claims: assert
            # the clear write landed before the drain runs.
            calls["ids"] = list(ids)
            calls["writes_before_drain"] = [c for c, _ in gc.writes]
            return {"requarantined_ids": ["bad"], "cleared_ids": ["good"]}

        out = default_audit_revalidate(gc, ["bad", "good"], drain_fn=fake_drain)

        clear_cypher, clear_params = gc.writes[-1]
        assert "SET sn.validated_at = null" in clear_cypher
        assert clear_params["ids"] == ["bad", "good"]
        # Drain saw the batch ids, and the clear happened first.
        assert calls["ids"] == ["bad", "good"]
        assert any(
            "SET sn.validated_at = null" in c for c in calls["writes_before_drain"]
        )
        assert out["cleared"] == 2
        assert out["requarantined_ids"] == ["bad"]
        assert out["valid_ids"] == ["good"]

    def test_audit_revalidate_noop_on_empty(self):
        gc = _FakeGraph([])
        out = default_audit_revalidate(gc, [], drain_fn=lambda ids: {})
        assert out == {"cleared": 0, "requarantined_ids": [], "valid_ids": []}
        assert gc.writes == []


# ── Runner orchestration (fully injected) ────────────────────────────────────


class _RunnerHarness:
    """Records the order and arguments of every injected campaign operation."""

    def __init__(self, refreshed_by_batch):
        self._refreshed_by_batch = refreshed_by_batch
        self.events: list[str] = []
        self.drained_run_ids: list[str] = []
        self.marked_batches: list[list[str]] = []
        self.recorded: list[tuple[list[str], str]] = []
        self.audited_ids: list[list[str]] = []
        # ids whose fresh deterministic audit finds a genuine defect.
        self.defective: set[str] = set()
        self._batch_calls = 0

    def mark_fn(self, batch, dry_run=False):
        self.events.append(f"mark:{','.join(batch)}")
        self.marked_batches.append(list(batch))
        return {
            "run_id": f"run-{len(self.marked_batches)}",
            "eligible": len(batch),
            "reset": len(batch),
        }

    def clear_quarantine_fn(self, gc, batch):
        self.events.append(f"clear:{','.join(batch)}")
        return len(batch)

    def drain_fn(self, run_id, cost_limit):
        self.events.append(f"drain:{run_id}:{cost_limit}")
        self.drained_run_ids.append(run_id)

    def fetch_refreshed_fn(self, gc, batch):
        self.events.append(f"fetch:{','.join(batch)}")
        idx = self._batch_calls
        self._batch_calls += 1
        return self._refreshed_by_batch[idx]

    def revalidate_fn(self, gc, reintroduced, clean):
        self.events.append(f"revalidate:{len(reintroduced)}:{len(clean)}")
        return {"requarantined": len(reintroduced), "confirmed": len(clean)}

    def audit_fn(self, gc, ids):
        self.events.append(f"audit:{','.join(ids)}")
        self.audited_ids.append(list(ids))
        req = [i for i in ids if i in self.defective]
        val = [i for i in ids if i not in self.defective]
        return {"requarantined_ids": req, "valid_ids": val}

    def record_change_fn(self, gc, ids, run_id, spec):
        self.events.append(f"record:{','.join(ids)}:{run_id}")
        self.recorded.append((list(ids), run_id))
        return len(ids)

    def cost_fn(self, run_id):
        return 1.0


def _accepted(sid):
    return {
        "id": sid,
        "name_stage": "accepted",
        "docs_stage": "accepted",
        "description": "Clean.",
        "documentation": "A clean scalar field.",
    }


class TestCampaignRunner:
    def test_full_run_all_batches_converge(self):
        target_ids = ["a", "b", "c"]
        refreshed = [[_accepted("a"), _accepted("b")], [_accepted("c")]]
        harness = _RunnerHarness(refreshed)
        runner = CampaignRunner(
            CampaignSpec.parse("prose"),
            CampaignBudget(
                batch_size=2, per_batch_cost_cap=3.0, campaign_cost_ceiling=100.0
            ),
            ConvergenceThresholds(),
        )
        result = runner.run(
            gc=None,
            target_ids=target_ids,
            drain_fn=harness.drain_fn,
            mark_fn=harness.mark_fn,
            clear_quarantine_fn=harness.clear_quarantine_fn,
            fetch_refreshed_fn=harness.fetch_refreshed_fn,
            revalidate_fn=harness.revalidate_fn,
            record_change_fn=harness.record_change_fn,
            cost_fn=harness.cost_fn,
        )
        assert result.halted is False
        assert result.batches_run == 2
        assert result.batches_total == 2
        assert result.total_touched == 3
        assert result.total_accepted == 3
        assert result.run_ids == ["run-1", "run-2"]
        assert result.total_cost == pytest.approx(2.0)

    def test_drain_measured_cost_preferred_over_cost_fn(self):
        # The drain returns its own measured spend (the pool session bills
        # under a different run_id than the campaign scope); the runner must
        # use it instead of the scope-run aggregation.
        harness = _RunnerHarness([[_accepted("a")]])

        def measuring_drain(run_id, cost_limit):
            harness.drain_fn(run_id, cost_limit)
            return 4.25

        runner = CampaignRunner(
            CampaignSpec.parse("prose"),
            CampaignBudget(batch_size=10),
            ConvergenceThresholds(),
        )
        result = runner.run(
            gc=None,
            target_ids=["a"],
            drain_fn=measuring_drain,
            mark_fn=harness.mark_fn,
            clear_quarantine_fn=harness.clear_quarantine_fn,
            fetch_refreshed_fn=harness.fetch_refreshed_fn,
            revalidate_fn=harness.revalidate_fn,
            record_change_fn=harness.record_change_fn,
            cost_fn=harness.cost_fn,
        )
        assert result.total_cost == pytest.approx(4.25)

    def test_operation_order_within_a_batch(self):
        harness = _RunnerHarness([[_accepted("a")]])
        runner = CampaignRunner(
            CampaignSpec.parse("prose"),
            CampaignBudget(batch_size=10),
            ConvergenceThresholds(),
        )
        runner.run(
            gc=None,
            target_ids=["a"],
            drain_fn=harness.drain_fn,
            mark_fn=harness.mark_fn,
            clear_quarantine_fn=harness.clear_quarantine_fn,
            fetch_refreshed_fn=harness.fetch_refreshed_fn,
            revalidate_fn=harness.revalidate_fn,
            record_change_fn=harness.record_change_fn,
            cost_fn=harness.cost_fn,
        )
        # clear before mark, mark before drain, drain before fetch/measure,
        # revalidate before record.
        assert harness.events == [
            "clear:a",
            "mark:a",
            "drain:run-1:10.0",
            "fetch:a",
            "revalidate:0:1",
            "record:a:run-1",
        ]

    def test_halts_on_low_acceptance_and_sets_resume(self):
        # Batch 0 converges; batch 1 has a non-accepted doc → acceptance 0.5.
        refreshed = [
            [_accepted("a"), _accepted("b")],
            [
                _accepted("c"),
                {
                    "id": "d",
                    "name_stage": "accepted",
                    "docs_stage": "reviewed",  # not accepted
                    "description": "x",
                    "documentation": "y",
                },
            ],
            [_accepted("e")],
        ]
        harness = _RunnerHarness(refreshed)
        runner = CampaignRunner(
            CampaignSpec.parse("prose"),
            CampaignBudget(batch_size=2, campaign_cost_ceiling=100.0),
            ConvergenceThresholds(min_docs_accept_rate=0.90),
        )
        result = runner.run(
            gc=None,
            target_ids=["a", "b", "c", "d", "e"],
            drain_fn=harness.drain_fn,
            mark_fn=harness.mark_fn,
            clear_quarantine_fn=harness.clear_quarantine_fn,
            fetch_refreshed_fn=harness.fetch_refreshed_fn,
            revalidate_fn=harness.revalidate_fn,
            record_change_fn=harness.record_change_fn,
            cost_fn=harness.cost_fn,
        )
        assert result.halted is True
        assert result.batches_run == 2
        assert result.resume_from == 2  # third batch never ran
        assert any("acceptance" in r for r in result.halt_reasons)
        # Third batch was never drained.
        assert "run-3" not in result.run_ids

    def test_resume_from_skips_completed_batches(self):
        refreshed = [[_accepted("c")]]  # only the third batch runs
        harness = _RunnerHarness(refreshed)
        runner = CampaignRunner(
            CampaignSpec.parse("prose"),
            CampaignBudget(batch_size=1),
            ConvergenceThresholds(),
        )
        result = runner.run(
            gc=None,
            target_ids=["a", "b", "c"],
            drain_fn=harness.drain_fn,
            mark_fn=harness.mark_fn,
            clear_quarantine_fn=harness.clear_quarantine_fn,
            fetch_refreshed_fn=harness.fetch_refreshed_fn,
            revalidate_fn=harness.revalidate_fn,
            record_change_fn=harness.record_change_fn,
            cost_fn=harness.cost_fn,
            start_batch=2,
        )
        assert result.batches_run == 1
        assert harness.marked_batches == [["c"]]

    def test_cost_ceiling_halts_between_batches(self):
        refreshed = [[_accepted("a")], [_accepted("b")]]
        harness = _RunnerHarness(refreshed)
        runner = CampaignRunner(
            CampaignSpec.parse("prose"),
            CampaignBudget(batch_size=1, campaign_cost_ceiling=1.0),
            ConvergenceThresholds(),
        )
        result = runner.run(
            gc=None,
            target_ids=["a", "b"],
            drain_fn=harness.drain_fn,
            mark_fn=harness.mark_fn,
            clear_quarantine_fn=harness.clear_quarantine_fn,
            fetch_refreshed_fn=harness.fetch_refreshed_fn,
            revalidate_fn=harness.revalidate_fn,
            record_change_fn=harness.record_change_fn,
            cost_fn=harness.cost_fn,
        )
        # First batch spends 1.0, reaching the ceiling; second batch is halted.
        assert result.batches_run == 1
        assert result.halted is True
        assert result.resume_from == 1
        assert any("ceiling" in r for r in result.halt_reasons)

    def test_abort_check_stops_cleanly(self):
        refreshed = [[_accepted("a")], [_accepted("b")]]
        harness = _RunnerHarness(refreshed)
        calls = {"n": 0}

        def abort_after_first():
            calls["n"] += 1
            return calls["n"] > 1  # allow batch 0, abort before batch 1

        runner = CampaignRunner(
            CampaignSpec.parse("prose"),
            CampaignBudget(batch_size=1),
            ConvergenceThresholds(),
        )
        result = runner.run(
            gc=None,
            target_ids=["a", "b"],
            drain_fn=harness.drain_fn,
            mark_fn=harness.mark_fn,
            clear_quarantine_fn=harness.clear_quarantine_fn,
            fetch_refreshed_fn=harness.fetch_refreshed_fn,
            revalidate_fn=harness.revalidate_fn,
            record_change_fn=harness.record_change_fn,
            cost_fn=harness.cost_fn,
            abort_check=abort_after_first,
        )
        assert result.batches_run == 1
        assert result.halted is False  # abort is clean, not a failure
        assert result.resume_from == 1

    def test_audit_fn_reruns_on_every_touched_id(self):
        # The re-audit must cover the whole refreshed batch so no member keeps a
        # stale validated_at from before the drain.
        refreshed = [[_accepted("a"), _accepted("b")], [_accepted("c")]]
        harness = _RunnerHarness(refreshed)
        runner = CampaignRunner(
            CampaignSpec.parse("prose"),
            CampaignBudget(batch_size=2, campaign_cost_ceiling=100.0),
            ConvergenceThresholds(),
        )
        result = runner.run(
            gc=None,
            target_ids=["a", "b", "c"],
            drain_fn=harness.drain_fn,
            mark_fn=harness.mark_fn,
            clear_quarantine_fn=harness.clear_quarantine_fn,
            fetch_refreshed_fn=harness.fetch_refreshed_fn,
            revalidate_fn=harness.revalidate_fn,
            audit_fn=harness.audit_fn,
            record_change_fn=harness.record_change_fn,
            cost_fn=harness.cost_fn,
        )
        assert harness.audited_ids == [["a", "b"], ["c"]]
        assert result.total_audit_cleared == 3
        assert result.total_audit_requarantined == 0

    def test_audit_fn_requarantines_genuine_defect(self):
        # A refreshed doc whose fresh deterministic audit finds a genuine defect
        # (e.g. a unit inconsistency) ends re-quarantined, not washed to 'valid'.
        refreshed = [[_accepted("a"), _accepted("b")]]
        harness = _RunnerHarness(refreshed)
        harness.defective = {"b"}
        runner = CampaignRunner(
            CampaignSpec.parse("prose"),
            CampaignBudget(batch_size=2),
            ConvergenceThresholds(),
        )
        result = runner.run(
            gc=None,
            target_ids=["a", "b"],
            drain_fn=harness.drain_fn,
            mark_fn=harness.mark_fn,
            clear_quarantine_fn=harness.clear_quarantine_fn,
            fetch_refreshed_fn=harness.fetch_refreshed_fn,
            revalidate_fn=harness.revalidate_fn,
            audit_fn=harness.audit_fn,
            record_change_fn=harness.record_change_fn,
            cost_fn=harness.cost_fn,
        )
        outcome = result.outcomes[0]
        assert outcome.audit_requarantined == 1
        assert outcome.audit_cleared == 1
        assert result.total_audit_requarantined == 1
        assert result.total_audit_cleared == 1
        assert result.summary()["total_audit_requarantined"] == 1

    def test_audit_runs_before_prose_requarantine(self):
        # Ordering: the deterministic re-audit stamps the whole batch first, then
        # the prose-grep re-quarantine lands on top (campaign-specific signal).
        harness = _RunnerHarness([[_accepted("a")]])
        runner = CampaignRunner(
            CampaignSpec.parse("prose"),
            CampaignBudget(batch_size=10),
            ConvergenceThresholds(),
        )
        runner.run(
            gc=None,
            target_ids=["a"],
            drain_fn=harness.drain_fn,
            mark_fn=harness.mark_fn,
            clear_quarantine_fn=harness.clear_quarantine_fn,
            fetch_refreshed_fn=harness.fetch_refreshed_fn,
            revalidate_fn=harness.revalidate_fn,
            audit_fn=harness.audit_fn,
            record_change_fn=harness.record_change_fn,
            cost_fn=harness.cost_fn,
        )
        assert harness.events == [
            "clear:a",
            "mark:a",
            "drain:run-1:10.0",
            "fetch:a",
            "audit:a",
            "revalidate:0:1",
            "record:a:run-1",
        ]
