"""Typed GET record reads preserve identity, bounds and explicit failure outcomes."""

import json
from unittest.mock import Mock

import pytest
import requests

from imas_codex.discovery.icp import (
    ICPClient,
    ICPNotFound,
    ICPPermissionDenied,
    ICPReadLimit,
    ICPRecords,
    ICPResponseError,
)

BASE = "/api/machine-component"


def plant(uid="plant-root", code="11", title="Synthetic plant", parent=None):
    return {"uid": uid, "pbsId": code, "title": title, "parent": parent}


def ref(uid="plant-root", text="Synthetic reference", kind="Pbs"):
    return {"uid": uid, "referenceText": text, "contentType": {"name": kind}}


def page(items=(), more=False):
    return {"items": list(items), "haveMore": more, "startIndex": 0, "endIndex": 0}


def branch(*children, uid="plant-root"):
    return {
        "uid": "tree-display-identity",
        "navigationTarget": {"content": ref(uid)},
        "hasChildren": bool(children),
        "childrenLoaded": True,
        "children": [
            {
                "uid": "display-" + uid,
                "hasChildren": expand,
                "navigationTarget": {"content": ref(uid)},
            }
            for uid, expand in children
        ],
    }


@pytest.fixture
def records():
    client = Mock(spec=ICPClient)
    client.origin = "https://integration.example.test"
    routes = {}

    def get(endpoint, **kwargs):
        params = kwargs.get("params", {})
        key = (endpoint, params.get("start", 0)) if params else endpoint
        assert key in routes, f"Unexpected synthetic request: {key}"
        value = routes[key]
        if isinstance(value, Exception):
            raise value
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(value).encode()
        response.url = client.origin + endpoint
        return response

    client.get.side_effect = get
    return ICPRecords(client), routes


def test_plant_code_lookup_resolves_parent_by_content_identity(records):
    reader, routes = records
    routes[(BASE + "/pbs/type/search", 0)] = page(
        [ref("plant-child"), ref("plant-other")]
    )
    routes[BASE + "/pbs/content/plant-child"] = plant(
        "plant-child",
        "11.20",
        "Child plant",
        {"uid": "plant-root", "referenceText": "Misleading display label"},
    )
    routes[BASE + "/pbs/content/plant-root"] = plant()
    routes[BASE + "/pbs/content/plant-other"] = plant("plant-other", "11.21")
    result = reader.plant_by_code("11.20")
    assert (result.code, result.title, result.parent_uid, result.parent_code) == (
        "11.20",
        "Child plant",
        "plant-root",
        "11",
    )
    assert result.source.origin == reader.client.origin
    assert result.source.endpoint == BASE + "/pbs/content/plant-child"
    assert result.source.uid == result.uid
    assert result.source.content_type == "Pbs"
    assert reader.client.get.call_args_list[0].kwargs["params"] == {
        "query": "11.20",
        "start": 0,
        "tot": 50,
    }


def test_unknown_code_is_explicit_not_found(records):
    reader, routes = records
    routes[(BASE + "/pbs/type/search", 0)] = page()
    with pytest.raises(ICPNotFound, match="exact code"):
        reader.plant_by_code("does-not-name-a-node")
    assert reader.client.get.call_count == 1


def test_fuzzy_match_is_not_an_exact_code_success(records):
    reader, routes = records
    routes[(BASE + "/pbs/type/search", 0)] = page([ref()])
    routes[BASE + "/pbs/content/plant-root"] = plant(code="11.20")
    with pytest.raises(ICPNotFound):
        reader.plant_by_code("11.2")


@pytest.mark.parametrize(
    "value", ({}, {"items": [], "haveMore": None}, {"items": None, "haveMore": False})
)
def test_missing_page_outcome_is_never_empty_success(records, value):
    reader, routes = records
    routes[(BASE + "/pbs/type/search", 0)] = value
    with pytest.raises(ICPResponseError):
        reader.plant_by_code("does-not-name-a-node")


def test_search_pages_advance_by_item_count_not_unpopulated_end_index(records):
    reader, routes = records
    routes[(BASE + "/pbs/type/search", 0)] = page([ref("plant-first")], True)
    routes[(BASE + "/pbs/type/search", 1)] = page([ref("plant-second")])
    pages = list(reader.search_plants("", page_size=1))
    assert [p.start for p in pages] == [0, 1]
    assert [p.items[0].uid for p in pages] == ["plant-first", "plant-second"]
    assert [p.have_more for p in pages] == [True, False]


@pytest.mark.parametrize(
    "bad_page",
    (page([], True), page([ref(), ref("plant-second")]), page([ref(), ref()], True)),
)
def test_inconsistent_pages_are_refused(records, bad_page):
    reader, routes = records
    routes[(BASE + "/pbs/type/search", 0)] = bad_page
    with pytest.raises(ICPResponseError):
        list(reader.search_plants("", page_size=1))


def test_repeated_search_page_is_refused(records):
    reader, routes = records
    routes[(BASE + "/pbs/type/search", 0)] = page([ref()], True)
    routes[(BASE + "/pbs/type/search", 1)] = page([ref()], False)
    with pytest.raises(ICPResponseError, match="repeated"):
        list(reader.search_plants("", page_size=1))


def test_exhausted_page_budget_is_explicit(records):
    reader, routes = records
    routes[(BASE + "/pbs/type/search", 0)] = page([ref()], True)
    with pytest.raises(ICPReadLimit):
        list(reader.search_plants("", max_pages=1))
    assert reader.client.get.call_count == 1


def test_subtree_reads_every_advertised_child_using_native_identity(records):
    reader, routes = records
    routes[BASE + "/pbs/content/plant-root"] = plant()
    routes[BASE + "/pbs/content/plant-child"] = plant(
        "plant-child", "11.20", parent={"uid": "plant-root"}
    )
    routes[BASE + "/pbs/content/plant-leaf"] = plant(
        "plant-leaf", "11.20.30", parent={"uid": "plant-child"}
    )
    routes[BASE + "/pbs/tree/plant-root"] = branch(("plant-child", True))
    routes[BASE + "/pbs/tree/plant-child"] = branch(
        ("plant-leaf", False), uid="plant-child"
    )
    result = reader.subtree("plant-root")
    assert [n.uid for n in result.nodes] == ["plant-root", "plant-child", "plant-leaf"]
    assert [n.parent_code for n in result.nodes] == [None, "11", "11.20"]
    assert result.denied_uids == ()
    assert result.branch_reads == 2
    assert reader.client.get.call_count == 5


@pytest.mark.parametrize(
    "alter", ("not_loaded", "missing_children", "missing_has_children")
)
def test_unloaded_or_inconsistent_branch_is_not_complete(records, alter):
    reader, routes = records
    value = branch(("plant-child", False))
    if alter == "not_loaded":
        value["childrenLoaded"] = False
    elif alter == "missing_children":
        value["children"] = None
    else:
        del value["children"][0]["hasChildren"]
    routes[BASE + "/pbs/tree/plant-root"] = value
    with pytest.raises(ICPResponseError):
        reader.children("plant-root")


def test_subtree_budget_refuses_partial_success(records):
    reader, routes = records
    routes[BASE + "/pbs/content/plant-root"] = plant()
    routes[BASE + "/pbs/tree/plant-root"] = branch(("plant-child", False))
    with pytest.raises(ICPReadLimit):
        reader.subtree("plant-root", max_nodes=1)
    assert reader.client.get.call_count == 2


def test_tree_cycle_refused(records):
    reader, routes = records
    routes[BASE + "/pbs/content/plant-root"] = plant()
    routes[BASE + "/pbs/tree/plant-root"] = branch(("plant-root", True))
    with pytest.raises(ICPResponseError, match="cycle"):
        reader.subtree("plant-root")


def http_error(status):
    response = requests.Response()
    response.status_code = status
    return requests.HTTPError("Synthetic refusal", response=response)


@pytest.mark.parametrize(
    ("status", "expected"),
    ((404, ICPNotFound), (403, ICPPermissionDenied), (500, requests.HTTPError)),
)
def test_http_outcomes_are_explicit(records, status, expected):
    reader, routes = records
    routes[BASE + "/pbs/content/plant-root"] = http_error(status)
    with pytest.raises(expected):
        reader.plant_by_uid("plant-root")


def test_subtree_counts_denied_records_and_continues_other_children(records):
    reader, routes = records
    routes[BASE + "/pbs/content/plant-root"] = plant()
    routes[BASE + "/pbs/tree/plant-root"] = branch(
        ("plant-denied", False), ("plant-readable", False)
    )
    routes[BASE + "/pbs/content/plant-denied"] = http_error(403)
    routes[BASE + "/pbs/content/plant-readable"] = plant("plant-readable", "11.20")
    result = reader.subtree("plant-root")
    assert result.denied_uids == ("plant-denied",)
    assert [n.uid for n in result.nodes] == ["plant-root", "plant-readable"]


@pytest.mark.parametrize("field", ("uid", "pbsId", "title"))
def test_plant_identity_and_required_text_are_checked(records, field):
    reader, routes = records
    value = plant()
    value[field] = "other-identity" if field == "uid" else None
    routes[BASE + "/pbs/content/plant-root"] = value
    with pytest.raises(ICPResponseError):
        reader.plant_by_uid("plant-root")


def test_function_category_read_and_bounded_search(records):
    reader, routes = records
    routes[(BASE + "/ttt-code/type/search", 0)] = page(
        [ref("category-main", "VAL", "TttCode")]
    )
    routes[BASE + "/ttt-code/content/category-main"] = {
        "uid": "category-main",
        "tttCodeId": "VAL",
        "title": "Synthetic valve",
        "componentType": "valve",
    }
    found = list(reader.search_function_categories())
    result = reader.function_category(found[0].items[0].uid)
    assert (result.code, result.title, result.component_type) == (
        "VAL",
        "Synthetic valve",
        "valve",
    )
    assert result.source.content_type == "TttCode"


def test_component_reads_by_reference_and_filters_actual_plant_link(records):
    reader, routes = records
    routes[(BASE + "/type/search", 0)] = page(
        [
            ref("component-main", "component-main", "MachineComponent"),
            ref("component-other", "component-other", "MachineComponent"),
        ]
    )
    for uid, parent in (
        ("component-main", "plant-root"),
        ("component-other", "plant-other"),
    ):
        routes[BASE + "/content/by-reference/" + uid] = {
            "uid": uid,
            "referenceText": uid,
            "pbs": {"uid": parent},
            "tttCode": {"uid": "category-main"},
            "functionReference": "Synthetic reference",
        }
    found = list(reader.components_for_plant("plant-root"))
    assert len(found) == 1
    assert found[0].uid == "component-main"
    assert found[0].function_category_uid == "category-main"
    assert found[0].source.endpoint == BASE + "/content/by-reference/component-main"
    assert reader.client.get.call_args_list[0].kwargs["params"]["query"] == ""


def test_reference_is_encoded_as_one_path_segment(records):
    reader, routes = records
    routes[BASE + "/content/by-reference/part%2Fwith%3Fpunctuation"] = {
        "uid": "component-main",
        "referenceText": "part/with?punctuation",
    }
    assert (
        reader.component_by_reference("part/with?punctuation").uid == "component-main"
    )


def test_mismatched_component_reference_is_refused(records):
    reader, routes = records
    routes[BASE + "/content/by-reference/component-main"] = {
        "uid": "component-main",
        "referenceText": "some-other-component",
    }
    with pytest.raises(ICPResponseError):
        reader.component_by_reference("component-main")
