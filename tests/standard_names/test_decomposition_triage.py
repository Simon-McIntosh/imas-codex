"""Deterministic bucketing tests for decomposition-audit triage.

Pure over the ISN grammar — no graph. Anchors the drain/suppress/rename split
against real catalog names so a grammar or rule regression is caught.
"""

from __future__ import annotations

import pytest

from imas_codex.standard_names.decomposition_triage import (
    DRAIN,
    NON_CANONICAL,
    PARSE_FAIL,
    RENAME,
    SUPPRESS,
    build_closed_vocab,
    build_manifest,
    classify_name,
    load_registered_bases,
    triage,
)


@pytest.fixture(scope="module")
def rule_inputs():
    return {"closed_vocab": build_closed_vocab(), "bases": load_registered_bases()}


def _bucket(name: str, rule_inputs) -> str:
    return classify_name(name, **rule_inputs).bucket


@pytest.mark.parametrize(
    "name",
    [
        # A closed token appears in the raw name but the parse slots it into a
        # real segment; the parsed physical_base is clean.
        "ion_current_density",  # subject=ion
        "krypton_density",  # subject=krypton
        "total_ion_temperature",  # aggregation + subject
        "plasma_pressure",  # channel_qualifier=plasma
        "momentum_damping_rate",  # channel=momentum
        "power_due_to_collisions",  # process locus
    ],
)
def test_drain_bucket(name, rule_inputs):
    """Stale findings: the token is correctly slotted, base is clean."""
    assert _bucket(name, rule_inputs) == DRAIN


@pytest.mark.parametrize(
    "name",
    [
        # Closed token embedded in a grammar-registered lexicalised base.
        "energy_diffusion_coefficient",  # base=diffusion_coefficient
        "energy_convection_velocity",  # base=convection_velocity
    ],
)
def test_suppress_bucket(name, rule_inputs):
    """The token is legitimately part of a registered atomic base."""
    entry = classify_name(name, **rule_inputs)
    assert entry.bucket == SUPPRESS
    assert entry.physical_base in load_registered_bases()
    assert entry.leaked_tokens  # a token IS embedded — just a legitimate one


@pytest.mark.parametrize(
    "name",
    [
        # Closed token genuinely absorbed into a non-base compound.
        "reference_magnetic_field",  # 'reference' leaked
        "vacuum_magnetic_vector_potential",  # 'vacuum' leaked
    ],
)
def test_rename_bucket(name, rule_inputs):
    """Genuine decomposition failure: absorbed into a non-registered base."""
    entry = classify_name(name, **rule_inputs)
    assert entry.bucket == RENAME
    assert entry.leaked_tokens
    assert entry.suggestion  # a reviewer hint is always emitted


def test_clearable_buckets_never_rename(rule_inputs):
    """The free-drain set must exclude every rename/backlog name."""
    names = [
        "ion_current_density",
        "energy_diffusion_coefficient",
        "reference_magnetic_field",
    ]
    entries = triage(names, **rule_inputs)
    manifest = build_manifest(entries)
    # drain + suppress are cleared; rename stays queued.
    assert manifest["clearable_free"] == 2
    assert manifest["buckets"]["rename"] == 1
    assert len(manifest["rename_queue"]) == 1
    assert manifest["rename_queue"][0]["name"] == "reference_magnetic_field"


def test_manifest_projects_review_cost(rule_inputs):
    entries = triage(
        ["reference_magnetic_field", "vacuum_magnetic_vector_potential"],
        **rule_inputs,
    )
    manifest = build_manifest(entries, rename_review_cost_per_name=0.10)
    assert manifest["buckets"]["rename"] == 2
    assert manifest["rename_queue_projected_review_cost_usd"] == pytest.approx(0.20)


def test_grammar_rejection_is_backlog_not_drain(rule_inputs):
    """A name the grammar rejects must never be silently drained."""
    entry = classify_name("definitely__not_a_valid_grammar_name", **rule_inputs)
    assert entry.bucket in (PARSE_FAIL, NON_CANONICAL)
    assert entry.detail
