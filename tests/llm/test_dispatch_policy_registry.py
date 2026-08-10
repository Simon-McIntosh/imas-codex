"""Registry ownership and disjoint typed-policy resource contracts."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from imas_codex.llm.dispatch_policy_registry import (
    DISPATCH_POLICY_REGISTRY,
    DispatchPolicyRegistryError,
    load_dispatch_policy_registry,
    policy_registry_closure,
    resolve_dispatch_policy,
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
        "response_model": "imas_codex.clusters.labeler:ClusterLabelBatch",
        "model_section": "dd-enrichment",
        "tokenizer": "imas_codex.llm.tokenizers:count_cluster_request",
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
