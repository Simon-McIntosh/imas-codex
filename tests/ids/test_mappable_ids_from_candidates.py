from unittest.mock import MagicMock

from imas_codex.ids.tools import discover_mappable_ids


def test_selected_candidate_home_extends_default_ids():
    gc = MagicMock()
    gc.query.side_effect = [
        [{"domain": "equilibrium", "cnt": 1}],
        [],
        [{"ids_name": "magnetics", "domains": ["magnetic_field_systems"]}],
    ]

    result = discover_mappable_ids("jet", gc=gc)
    assert [target["ids_name"] for target in result["ids_targets"]] == ["magnetics"]
    candidate_query = gc.query.call_args
    assert "MAPPING_CANDIDATE" in candidate_query.args[0]
    assert "r.route = true" in candidate_query.args[0]
    assert "sg.candidate_route = 'escalated'" in candidate_query.args[0]
    assert candidate_query.kwargs["facility"] == "jet"


def test_domain_filter_includes_only_covered_sources_candidate_homes():
    gc = MagicMock()
    non_candidate_rows = iter(
        [
            [{"domain": "equilibrium", "cnt": 2}],
            [{"ids_name": "equilibrium", "domains": ["equilibrium"]}],
        ]
    )
    candidate_homes = [
        (
            "equilibrium",
            {"ids_name": "magnetics", "domains": ["magnetic_field_systems"]},
        ),
        ("plasma_control", {"ids_name": "pf_active", "domains": ["plasma_control"]}),
    ]

    def query(statement, **params):
        if "MAPPING_CANDIDATE" in statement:
            return [
                home
                for source_domain, home in candidate_homes
                if source_domain in params["filter_domains"]
            ]
        return next(non_candidate_rows)

    gc.query.side_effect = query

    result = discover_mappable_ids("jet", gc=gc, domains=["equilibrium"])

    assert [target["ids_name"] for target in result["ids_targets"]] == [
        "equilibrium",
        "magnetics",
    ]
    assert "pf_active" not in [target["ids_name"] for target in result["ids_targets"]]
    candidate_query = gc.query.call_args
    assert "sg.physics_domain IN $filter_domains" in candidate_query.args[0]
    assert candidate_query.kwargs["filter_domains"] == ["equilibrium"]
    assert candidate_query.kwargs["facility"] == "jet"


def test_candidate_homes_survive_missing_source_physics_domains():
    gc = MagicMock()
    gc.query.side_effect = [
        [],
        [{"ids_name": "magnetics", "domains": ["magnetic_field_systems"]}],
    ]

    result = discover_mappable_ids("jet", gc=gc)

    assert [target["ids_name"] for target in result["ids_targets"]] == ["magnetics"]
    assert "MAPPING_CANDIDATE" in gc.query.call_args.args[0]


def test_explicit_ids_filter_does_not_add_candidate_homes():
    gc = MagicMock()
    gc.query.side_effect = [
        [],
        [{"ids_name": "pf_active", "domains": ["magnetic_field_systems"]}],
    ]

    result = discover_mappable_ids("jet", gc=gc, ids_filter=["pf_active"])

    assert [target["ids_name"] for target in result["ids_targets"]] == ["pf_active"]
    assert gc.query.call_count == 2
