"""Tests for the per-role SN seat benchmarks.

All scoring arithmetic and corpus shaping is pure, so these tests run with no
live graph or LLM — corpus loaders take an injected fake GraphClient.
"""

from __future__ import annotations

from imas_codex.standard_names import benchmark_roles as br


class TestSpearmanRho:
    def test_perfect_positive(self):
        assert br.spearman_rho([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0

    def test_perfect_negative(self):
        assert br.spearman_rho([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0

    def test_ties_are_averaged(self):
        # constant vector → no correlation sentinel
        assert br.spearman_rho([1, 1, 1, 1], [1, 2, 3, 4]) == 0.0

    def test_degenerate_length(self):
        assert br.spearman_rho([1.0], [2.0]) == 0.0
        assert br.spearman_rho([], []) == 0.0

    def test_partial_correlation_in_range(self):
        rho = br.spearman_rho([1, 2, 3, 4, 5], [1, 3, 2, 5, 4])
        assert -1.0 <= rho <= 1.0
        assert rho > 0.0


class TestVerdictFlipQuality:
    def test_all_flips_correct(self):
        # candidate accepts where pair rejects, and outcome is accept → correct
        q, n = br.verdict_flip_quality([0.9], [0.5], [True])
        assert q == 1.0 and n == 1

    def test_flip_wrong(self):
        q, n = br.verdict_flip_quality([0.9], [0.5], [False])
        assert q == 0.0 and n == 1

    def test_no_flips(self):
        # candidate and pair agree → no flip counted
        q, n = br.verdict_flip_quality([0.9, 0.5], [0.85, 0.5], [True, False])
        assert n == 0 and q == 0.0

    def test_mixed(self):
        # 3 flips: (accept vs reject, outcome accept → correct),
        #          (accept vs reject, outcome reject → wrong),
        #          (reject vs accept, outcome accept → wrong) → 1/3 correct.
        q, n = br.verdict_flip_quality(
            [0.9, 0.9, 0.5], [0.5, 0.5, 0.9], [True, False, True]
        )
        assert n == 3
        assert q == round(1 / 3, 4)


class TestBannedProse:
    def test_typical_values(self):
        f = br.banned_prose_findings("Values are typically around 5 keV.")
        assert f["typical_values"] >= 1

    def test_estimator_recipe(self):
        f = br.banned_prose_findings("This is computed as the ratio of A to B.")
        assert f["estimator_recipe"] >= 1

    def test_procedural_padding(self):
        f = br.banned_prose_findings("Note that in practice this varies.")
        assert f["procedural_padding"] >= 2

    def test_clean_text(self):
        f = br.banned_prose_findings(
            "The poloidal component of the magnetic field on the plasma boundary."
        )
        assert sum(f.values()) == 0

    def test_empty(self):
        assert sum(br.banned_prose_findings("").values()) == 0
        assert sum(br.banned_prose_findings(None).values()) == 0


class TestExactMatch:
    def test_all_correct(self):
        acc, c, t = br.exact_match_accuracy({"a": "x", "b": "y"}, {"a": "x", "b": "y"})
        assert (acc, c, t) == (1.0, 2, 2)

    def test_missing_counts_wrong(self):
        acc, c, t = br.exact_match_accuracy({"a": "x"}, {"a": "x", "b": "y"})
        assert (c, t) == (1, 2)
        assert acc == 0.5

    def test_empty_expected(self):
        assert br.exact_match_accuracy({}, {}) == (0.0, 0, 0)


class TestStratifiedSample:
    def test_balances_across_key(self):
        rows = [{"physics_domain": "equilibrium", "id": i} for i in range(10)] + [
            {"physics_domain": "transport", "id": 100 + i} for i in range(10)
        ]
        out = br._stratified_sample(rows, sample=4, seed=1)
        domains = [r["physics_domain"] for r in out]
        assert domains.count("equilibrium") == 2
        assert domains.count("transport") == 2

    def test_deterministic(self):
        rows = [{"physics_domain": d, "id": i} for i, d in enumerate("abcabcabc")]
        a = br._stratified_sample(rows, 5, seed=7)
        b = br._stratified_sample(rows, 5, seed=7)
        assert [r["id"] for r in a] == [r["id"] for r in b]

    def test_respects_sample_size(self):
        rows = [{"physics_domain": "x", "id": i} for i in range(3)]
        assert len(br._stratified_sample(rows, 10, seed=1)) == 3


class _FakeGC:
    """Minimal GraphClient stand-in returning canned rows per query."""

    def __init__(self, rows):
        self._rows = rows
        self.closed = False

    def query(self, cypher, **kwargs):
        return self._rows

    def close(self):
        self.closed = True


class TestLoaders:
    def test_refine_corpus_parses_json_critique(self):
        rows = [
            {
                "sn_id": "a_b",
                "prior_name": "a",
                "prior_score": 0.5,
                "critique": '{"grammar": "bad token"}',
                "source_paths": ["eq/x"],
                "description": "d",
                "unit": "T",
                "data_type": "scalar",
                "physics_domain": "equilibrium",
            }
        ]
        gc = _FakeGC(rows)
        out = br.load_refine_corpus(5, seed=1, gc=gc)
        assert len(out) == 1
        assert out[0]["critique"] == {"grammar": "bad token"}
        assert out[0]["path"] == "eq/x"

    def test_breaker_corpus_requires_both_pair_scores(self):
        rows = [
            {
                "sn_id": "n1",
                "scores": [
                    {"m": br.BLIND_PAIR[0], "s": 0.8},
                    {"m": br.BLIND_PAIR[1], "s": 0.6},
                ],
                "name_stage": "accepted",
                "description": "d",
                "unit": "T",
                "data_type": "scalar",
                "physics_domain": "equilibrium",
                "source_paths": ["eq/x"],
                "documentation": "doc",
            }
        ]
        out = br.load_breaker_corpus(5, seed=1, axis="names", gc=_FakeGC(rows))
        assert len(out) == 1
        assert out[0]["pair_scores"] == {"qwen": 0.8, "minimax": 0.6}
        assert out[0]["final_accepted"] is True

    def test_docs_sample_shape(self):
        rows = [
            {
                "name": "n1",
                "description": "d",
                "unit": "T",
                "kind": "scalar",
                "physics_domain": "equilibrium",
                "source_paths": ["eq/x"],
            }
        ]
        out = br.load_docs_sample(5, seed=1, gc=_FakeGC(rows))
        assert out[0]["name"] == "n1" and out[0]["kind"] == "scalar"


class TestReportRoundTrip:
    def test_json_round_trip(self, tmp_path):
        rep = br.RoleBenchReport(
            role="breaker-names",
            results=[
                br.RoleModelResult(
                    model="openrouter/openai/gpt-5.6-luna",
                    n=10,
                    cost=0.5,
                    metrics={"independence_rho": 0.8},
                )
            ],
            incumbent="openrouter/openai/gpt-5.5",
            axis="names",
        )
        text = rep.to_json()
        back = br.RoleBenchReport.from_json(text)
        assert back.role == "breaker-names"
        assert back.results[0].metrics["independence_rho"] == 0.8
        assert back.results[0].cost_per_item == round(0.5 / 10, 6)

        p = tmp_path / "r.json"
        rep.save_atomic(str(p))
        assert br.RoleBenchReport.from_json(p.read_text()).incumbent == (
            "openrouter/openai/gpt-5.5"
        )

    def test_gold_set_loads(self):
        gold = br.load_classifier_gold()
        assert len(gold) >= 100
        assert {"path", "expected_domain"} <= set(gold[0])


class TestRefinePromptContext:
    """The refine bench must feed the same grammar vocabulary as production.

    Without merging the compose context, the refine system prompt's grammar
    block renders empty and candidates invent unregistered tokens — a bench
    that measures a grammar-failure artifact rather than refine quality.
    """

    _CASE = {
        "sn_id": "safety_factor",
        "prior_name": "safety_factor",
        "prior_score": 0.6,
        "critique": {"grammar": "use the q-profile token"},
        "path": "equilibrium/time_slice/profiles_1d/q",
        "description": "safety factor",
        "unit": "-",
        "data_type": "scalar",
        "physics_domain": "equilibrium",
    }

    def test_merges_compose_context_vocabulary(self):
        compose_context = {
            "vocabulary_sections": [{"segment": "base", "tokens": ["q"]}],
            "closed_vocab_full": {"base": ["q"]},
        }
        ctx = br.build_refine_prompt_context(self._CASE, compose_context, rules=[])
        # The vocabulary the grammar-reference include renders must survive.
        assert ctx["vocabulary_sections"] == compose_context["vocabulary_sections"]
        assert ctx["closed_vocab_full"] == compose_context["closed_vocab_full"]

    def test_wires_case_and_chain(self):
        ctx = br.build_refine_prompt_context(self._CASE, {}, rules=["r1"])
        assert ctx["item"]["path"] == self._CASE["path"]
        assert ctx["item"]["ids_name"] == "equilibrium"
        assert ctx["chain_length"] == 1
        assert ctx["chain_history"][0]["name"] == "safety_factor"
        assert ctx["chain_history"][0]["reviewer_comments_per_dim"] == {
            "grammar": "use the q-profile token"
        }
        assert ctx["composition_rules"] == ["r1"]

    def test_scored_examples_carry_into_context(self):
        # The examples are the only place the refine prompt shows a valid entry
        # kind (kind=<scalar|vector|metadata>); they must reach the context.
        examples = [{"id": "q", "kind": "scalar", "reviewer_score": 0.9}]
        ctx = br.build_refine_prompt_context(
            self._CASE, {}, rules=[], scored_examples=examples
        )
        assert ctx["compose_scored_examples"] == examples

    def test_scored_examples_default_empty(self):
        ctx = br.build_refine_prompt_context(self._CASE, {}, rules=[])
        assert ctx["compose_scored_examples"] == []


class TestClassifyRefineFailure:
    def test_kind_enum(self):
        exc = ValueError("kind must be one of {'scalar'}, got 'standard_name'")
        assert br._classify_refine_failure(exc) == "kind_enum"

    def test_grammar_token(self):
        exc = ValueError("qualifier 'perturbed' is not a registered grammar token")
        assert br._classify_refine_failure(exc) == "grammar_token"

    def test_other(self):
        assert br._classify_refine_failure(RuntimeError("provider timeout")) == "other"
