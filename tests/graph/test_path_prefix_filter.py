"""Tests for the path-prefix clause-and-parameters helper.

The helper exists so a caller cannot render the prefix predicate and then
forget to bind the parameter it names. These tests pin the pair it returns:
the clause for the node alias, the parameter dict that fills it, and the
empty result that leaves a query untouched when no scope is offered.
"""

from imas_codex.graph.query_builder import (
    build_path_prefix_filter,
    render_path_prefix_clause,
)


class TestBuildPathPrefixFilter:
    def test_returns_clause_and_parameter_together(self):
        clause, params = build_path_prefix_filter("sf", ["/analysis/src/SAeqread"])
        assert clause == render_path_prefix_clause("sf", "prefixes")
        assert params == {"prefixes": ["/analysis/src/SAeqread"]}

    def test_clause_targets_the_given_alias(self):
        clause, _ = build_path_prefix_filter("p", ["/analysis/src"])
        assert "p.path STARTS WITH prefix" in clause
        assert "sf.path" not in clause

    def test_parameter_name_matches_the_clause(self):
        clause, params = build_path_prefix_filter("sf", ["/a"])
        (param_name,) = params
        assert f"${param_name}" in clause

    def test_empty_list_yields_empty_clause_and_no_parameters(self):
        assert build_path_prefix_filter("sf", []) == ("", {})

    def test_none_yields_empty_clause_and_no_parameters(self):
        assert build_path_prefix_filter("sf", None) == ("", {})

    def test_parameters_are_a_copy_of_the_input(self):
        prefixes = ["/a"]
        _, params = build_path_prefix_filter("sf", prefixes)
        prefixes.append("/b")
        assert params["prefixes"] == ["/a"]
