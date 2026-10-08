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
