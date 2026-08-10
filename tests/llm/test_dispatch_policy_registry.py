"""Registry ownership and disjoint typed-policy resource contracts."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

import imas_codex.llm.dispatch_policy_registry as policy_registry
from imas_codex.llm.dispatch_policy_registry import (
    DISPATCH_POLICY_REGISTRY,
    DispatchPolicyRegistryError,
    load_dispatch_policy_registry,
    load_dispatch_registry,
    policy_registry_closure,
    resolve_dispatch_policy,
)


class RegistryResponse(BaseModel):
    label: str


def _count_request(request: dict[str, object]) -> int:
    return 10


def _candidate_models() -> tuple[str, ...]:
    return ("openrouter/openai/gpt-5.6-luna",)


def _invalid_candidate_source() -> str:
    return "openrouter/openai/gpt-5.6-luna"


@pytest.fixture(autouse=True)
def registered_response_contract(monkeypatch):
    monkeypatch.setattr(
        policy_registry,
        "get_callsite_registration",
        lambda callsite_id: SimpleNamespace(
            response_model_identity=f"{__name__}:RegistryResponse"
        ),
    )


def _resource(callsite_id: str) -> dict[str, object]:
    return {
        "policy_id": f"{callsite_id}.policy",
        "source_version": "policy-release",
        "callsite_id": callsite_id,
        "service": "data-dictionary",
        "seat": "dd-enrichment",
        "task_kind": "cluster_labeling",
        "templates": [
            {
                "role": "system",
                "name": "clusters/labeler",
                "source_version": "template-release",
            }
        ],
        "response_model": f"{__name__}:RegistryResponse",
        "model_section": "dd-enrichment",
        "tokenizer": f"{__name__}:_count_request",
        "tokenizer_key": "cluster-provider-wire",
        "identifier_pattern": "[A-Za-z0-9._:/+-]+",
        "channels": {
            name: {"kinds": [], "scopes": ["exact_item"]}
            for name in (
                "source_facts",
                "approved_resolutions",
                "reviewer_intent",
                "comparators",
                "provenance",
                "batch_comparators",
            )
        },
        "required_obligations": [],
        "static_providers": [],
        "max_input_tokens": 1000,
        "max_output_tokens": 100,
        "max_attempts": 2,
        "max_context_bytes": 10000,
        "maximum_cost_exposure": 1.0,
    }


def test_checked_in_registry_has_no_invented_production_policies() -> None:
    assert not DISPATCH_POLICY_REGISTRY
    with pytest.raises(DispatchPolicyRegistryError, match="no typed-ready"):
        resolve_dispatch_policy("dd.cluster-labeling")
    with pytest.raises(DispatchPolicyRegistryError, match="typed-unsupported"):
        resolve_dispatch_policy("discovery.image-scoring")


def test_domain_resources_load_without_shared_registry_edits(tmp_path) -> None:
    first = _resource("dd.cluster-labeling")
    second = _resource("discovery.image-scoring")
    (tmp_path / "data_dictionary.json").write_text(json.dumps({"policies": [first]}))
    (tmp_path / "facility_discovery.json").write_text(
        json.dumps({"policies": [second]})
    )

    registry = load_dispatch_policy_registry(tmp_path)

    assert tuple(registry) == ("dd.cluster-labeling", "discovery.image-scoring")
    with pytest.raises(TypeError):
        registry["another"] = registry["dd.cluster-labeling"]
    with pytest.raises(FrozenInstanceError):
        registry["dd.cluster-labeling"].max_attempts = 99


def test_typed_expression_without_policy_fails_closure() -> None:
    call = SimpleNamespace(transition_kind="typed", callsite_id="missing.policy")

    with pytest.raises(DispatchPolicyRegistryError, match="has no policy"):
        policy_registry_closure((call,), registry={})


def test_atomic_registry_rejects_active_blocker_collision(tmp_path) -> None:
    resource = _resource("discovery.image-scoring")
    (tmp_path / "facility_discovery.json").write_text(
        json.dumps({"policies": [resource]})
    )
    (tmp_path / "facility_discovery.blocked.json").write_text(
        json.dumps(
            {
                "unsupported": [
                    {
                        "callsite_id": "discovery.image-scoring",
                        "closure_blocker": "Exact image pricing is unavailable",
                    }
                ]
            }
        )
    )

    with pytest.raises(DispatchPolicyRegistryError, match="both active and blocked"):
        load_dispatch_registry(tmp_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.__setitem__("suported", False), "extra_forbidden"),
        (
            lambda value: value["attachments"].__setitem__("max_wdith", 10),
            "extra_forbidden",
        ),
    ],
)
def test_policy_resources_reject_unknown_fields(tmp_path, mutation, message) -> None:
    resource = _resource("dd.cluster-labeling")
    resource["attachments"] = {}
    mutation(resource)
    (tmp_path / "data_dictionary.json").write_text(json.dumps({"policies": [resource]}))

    with pytest.raises(DispatchPolicyRegistryError, match=message):
        load_dispatch_registry(tmp_path)


def test_registry_rejects_duplicate_policy_and_template_identities(tmp_path) -> None:
    first = _resource("dd.cluster-labeling")
    second = _resource("discovery.image-scoring")
    second["policy_id"] = first["policy_id"]
    (tmp_path / "duplicate_policy.json").write_text(
        json.dumps({"policies": [first, second]})
    )
    with pytest.raises(DispatchPolicyRegistryError, match="Duplicate typed policy id"):
        load_dispatch_registry(tmp_path)

    first["templates"].append(dict(first["templates"][0]))
    (tmp_path / "duplicate_policy.json").write_text(json.dumps({"policies": [first]}))
    with pytest.raises(
        DispatchPolicyRegistryError, match="templates contains duplicates"
    ):
        load_dispatch_registry(tmp_path)


def test_registry_rejects_duplicate_json_object_keys(tmp_path) -> None:
    resource = json.dumps(_resource("dd.cluster-labeling"))
    (tmp_path / "duplicate_key.json").write_text(
        f'{{"policies": [{resource}], "policies": []}}'
    )

    with pytest.raises(DispatchPolicyRegistryError, match="duplicate JSON key"):
        load_dispatch_registry(tmp_path)


def test_candidate_source_is_finite_unique_and_enforced(tmp_path) -> None:
    resource = _resource("dd.cluster-labeling")
    resource["candidate_source"] = f"{__name__}:_candidate_models"
    (tmp_path / "data_dictionary.json").write_text(json.dumps({"policies": [resource]}))
    registry = load_dispatch_policy_registry(tmp_path)

    resolved = resolve_dispatch_policy(
        "dd.cluster-labeling",
        candidate_model="openrouter/openai/gpt-5.6-luna",
        registry=registry,
    )
    assert resolved.model == "openrouter/openai/gpt-5.6-luna"
    with pytest.raises(DispatchPolicyRegistryError, match="outside"):
        resolve_dispatch_policy(
            "dd.cluster-labeling",
            candidate_model="openrouter/other/model",
            registry=registry,
        )

    resource["candidate_source"] = f"{__name__}:_invalid_candidate_source"
    (tmp_path / "data_dictionary.json").write_text(json.dumps({"policies": [resource]}))
    with pytest.raises(DispatchPolicyRegistryError, match="finite sequence"):
        load_dispatch_registry(tmp_path)


def test_unsupported_policy_cannot_satisfy_typed_closure(tmp_path) -> None:
    resource = _resource("dd.cluster-labeling")
    resource["supported"] = False
    resource["closure_blocker"] = "Route remains unsupported"
    (tmp_path / "data_dictionary.json").write_text(json.dumps({"policies": [resource]}))
    registry = load_dispatch_policy_registry(tmp_path)
    call = SimpleNamespace(transition_kind="typed", callsite_id="dd.cluster-labeling")

    with pytest.raises(DispatchPolicyRegistryError, match="remains unsupported"):
        policy_registry_closure((call,), registry=registry)
