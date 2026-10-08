"""Typed transform decisions compose to expressions accepted by the executor."""

from __future__ import annotations

import math

import pytest

from imas_codex.ids.models import (
    BindingTransformSlots,
    CocosLabel,
    IndexLayout,
    SignalMappingEntry,
    SlotResolution,
    TransformSlot,
    UnitConversion,
    ValidatedSignalMapping,
)
from imas_codex.ids.transforms import compose_transform, execute_transform

EXACT_UNIT_PAIRS = [
    ("%", "1"),
    ("A", "A"),
    ("DEG", "1"),
    ("DEG", "rad"),
    ("K", "K"),
    ("KA", "A"),
    ("KV", "V"),
    ("Pa", "Pa"),
    ("Pa*m3/s", "Pa.m^3.s^-1"),
    ("Pam3/s", "Pa.m^3.s^-1"),
    ("Pam3/s", "W"),
    ("V", "V"),
    ("au", "m"),
    ("count", "rad"),
    ("day", "s"),
    ("degree", "rad"),
    ("degreeC", "K"),
    ("g/s", "kg.s^-1"),
    ("kA", "A"),
    ("kV", "V"),
    ("m", "m"),
    ("m3", "m^3"),
    ("minute", "s"),
    ("mm", "m"),
    ("ms", "s"),
    ("msec", "s"),
    ("oC", "K"),
    ("sec", "s"),
    ("uST", "1"),
]


def settled(value, authority=SlotResolution.CODE):
    return TransformSlot(value=value, settled_by=authority, evidence="source and DD")


def slots(**changes):
    values = {
        "sign": settled(1),
        "scale_factor": settled(1.0),
        "unit_conversion": settled(UnitConversion()),
        "cocos_label": settled(CocosLabel.NONE),
        "index_layout": settled(IndexLayout(kind="scalar")),
    }
    values.update(changes)
    return BindingTransformSlots(**values)


def binding(transform_slots):
    return ValidatedSignalMapping(
        source_id="source",
        target_id="target",
        confidence=1,
        mapping_type="direct",
        transform_slots=transform_slots,
    )


@pytest.mark.parametrize(
    ("sign", "factor", "expected"),
    [
        (1, 1.0, 2.0),
        (-1, 1.0, -2.0),
        (1, 2 * math.pi, 4 * math.pi),
        (-1, 1 / (2 * math.pi), -1 / math.pi),
    ],
)
def test_sign_and_scale_compose(sign, factor, expected):
    decision = slots(
        sign=settled(sign, SlotResolution.JEV),
        scale_factor=settled(factor, SlotResolution.LOCAL),
    )
    result = binding(decision)
    assert result.transform_expression == compose_transform(decision)
    assert execute_transform(2.0, result.transform_expression) == pytest.approx(
        expected
    )
    assert decision.sign.evidence == "source and DD"
    assert decision.sign.settled_by == SlotResolution.JEV
    assert decision.scale_factor.settled_by == SlotResolution.LOCAL


@pytest.mark.parametrize(
    ("label", "cocos_factor"),
    [
        ("ip_like", 1.0),
        ("b0_like", 1.0),
        ("tor_angle_like", 1.0),
        ("pol_angle_like", -1.0),
        ("q_like", -1.0),
        ("psi_like", 2 * math.pi),
        ("dodpsi_like", 1 / (2 * math.pi)),
        ("one_like", 1.0),
    ],
)
def test_cocos_label_uses_cocos_sign(label, cocos_factor):
    decision = slots(
        cocos_label=settled(CocosLabel(label)),
        cocos_in=3,
        cocos_out=17,
    )
    expression = compose_transform(decision)
    assert "cocos_sign(" in expression
    assert execute_transform(2.0, expression) == pytest.approx(2 * cocos_factor)


def test_psi_source_sign_flip_multiplies_cocos_factor():
    decision = slots(
        sign=settled(-1),
        cocos_label=settled(CocosLabel.PSI),
        cocos_in=3,
        cocos_out=17,
    )
    expression = binding(decision).transform_expression
    assert (
        expression
        == "((value * -1.0) * cocos_sign('psi_like', cocos_in=3, cocos_out=17))"
    )
    assert execute_transform(2.0, expression) == pytest.approx(-4 * math.pi)


def test_ip_source_identity_sign_uses_cocos_alone():
    decision = slots(
        sign=settled(1),
        cocos_label=settled(CocosLabel.IP),
        cocos_in=11,
        cocos_out=17,
    )
    expression = binding(decision).transform_expression
    assert expression == "(value * cocos_sign('ip_like', cocos_in=11, cocos_out=17))"
    assert execute_transform(2.0, expression) == -2.0


@pytest.mark.parametrize(
    ("source", "target", "input_value", "expected"),
    [
        ("kA", "A", 2.0, 2000.0),
        ("degree", "rad", 180.0, math.pi),
        ("degreeC", "K", 0.0, 273.15),
        ("m", "m", 2.0, 2.0),
    ],
)
def test_exact_unit_pair_composes(source, target, input_value, expected):
    decision = slots(
        unit_conversion=settled(UnitConversion(source_unit=source, target_unit=target))
    )
    expression = compose_transform(decision)
    assert execute_transform(input_value, expression) == pytest.approx(expected)


@pytest.mark.parametrize(("source", "target"), EXACT_UNIT_PAIRS)
def test_all_census_exact_unit_pairs_execute(source, target):
    from imas_codex.units import unit_registry

    decision = slots(
        unit_conversion=settled(UnitConversion(source_unit=source, target_unit=target))
    )
    expected = unit_registry.Quantity(2.0, source).to(target).magnitude
    assert execute_transform(2.0, compose_transform(decision)) == pytest.approx(
        expected
    )


def test_index_unit_and_sign_compose_together():
    decision = slots(
        sign=settled(-1),
        unit_conversion=settled(UnitConversion(source_unit="kA", target_unit="A")),
        index_layout=settled(IndexLayout(kind="index", index=1)),
    )
    assert execute_transform([1.0, 2.0], compose_transform(decision)) == -2000.0


def test_each_slot_retains_value_authority_and_evidence():
    decision = slots(sign=settled(-1, SlotResolution.JEV))
    persisted = BindingTransformSlots.model_validate(decision.model_dump(mode="json"))
    for name in (
        "sign",
        "scale_factor",
        "unit_conversion",
        "cocos_label",
        "index_layout",
    ):
        slot = getattr(persisted, name)
        assert slot.value is not None
        assert slot.settled_by is not None
        assert slot.evidence == "source and DD"


@pytest.mark.parametrize(
    ("layout", "expected"),
    [
        (IndexLayout(kind="scalar"), [2, 3, 4]),
        (IndexLayout(kind="index", index=1), 3),
        (IndexLayout(kind="slice", start=1, stop=3), [3, 4]),
    ],
)
def test_scalar_and_indexed_layouts(layout, expected):
    decision = slots(index_layout=settled(layout))
    assert execute_transform([2, 3, 4], compose_transform(decision)) == expected


@pytest.mark.parametrize(
    "open_name",
    ["sign", "scale_factor", "unit_conversion", "cocos_label", "index_layout"],
)
def test_open_slot_refuses_binding(open_name):
    decision = slots(**{open_name: TransformSlot()})
    with pytest.raises(ValueError, match=open_name):
        binding(decision)


def test_escalated_slot_refuses_binding_even_with_candidate_value():
    decision = slots(sign=settled(-1, SlotResolution.ESCALATED))
    with pytest.raises(ValueError, match="sign"):
        compose_transform(decision)


def test_settled_slot_requires_evidence():
    with pytest.raises(ValueError, match="evidence"):
        TransformSlot(value=-1, settled_by=SlotResolution.CODE)


def test_constant_scale_multiplies_cocos_factor():
    decision = slots(
        scale_factor=settled(2.0),
        cocos_label=settled(CocosLabel.PSI),
        cocos_in=3,
        cocos_out=17,
    )
    expression = compose_transform(decision)
    assert (
        expression
        == "((value * 2.0) * cocos_sign('psi_like', cocos_in=3, cocos_out=17))"
    )
    assert execute_transform(2.0, expression) == pytest.approx(8 * math.pi)


def test_mapping_entry_stores_composed_expression():
    decision = slots(sign=settled(-1))
    entry = SignalMappingEntry(
        source_id="source", target_id="target", confidence=1, transform_slots=decision
    )
    assert entry.transform_expression == "(value * -1.0)"
    assert execute_transform(2, entry.transform_expression) == -2
