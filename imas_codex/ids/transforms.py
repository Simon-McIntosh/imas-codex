"""Transform execution engine for IDS field mappings.

Provides safe evaluation of transform_expression expressions stored on
IMASMapping nodes, with access to math, numpy, and imas_codex
unit/COCOS utilities.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from imas_codex.ids.models import BindingTransformSlots

logger = logging.getLogger(__name__)

# Lazy numpy import — not always available
_numpy = None


def _get_numpy():
    global _numpy
    if _numpy is None:
        try:
            import numpy

            _numpy = numpy
        except ImportError:
            _numpy = False
    return _numpy if _numpy is not False else None


def convert_units(value: float, from_unit: str, to_unit: str) -> float:
    """Convert a value between units using pint."""
    from imas_codex.units import unit_registry

    q = unit_registry.Quantity(value, from_unit)
    return q.to(to_unit).magnitude


def cocos_sign(label: str, *, cocos_in: int, cocos_out: int) -> int | float:
    """Compute the COCOS sign/scale factor for a given label.

    Uses the COCOS parameter decomposition from Sauter & Medvedev (2013)
    to compute the transformation factor for a given cocos_transformation_type.

    Supported labels:
        ip_like:        σ_RφZ_out · σ_Bp_out / (σ_RφZ_in · σ_Bp_in)
        b0_like:        σ_RφZ_out / σ_RφZ_in
        tor_angle_like: σ_RφZ_out / σ_RφZ_in
        pol_angle_like: σ_ρθφ_out / σ_ρθφ_in
        q_like:         (σ_ρθφ_out · σ_RφZ_out) / (σ_ρθφ_in · σ_RφZ_in)
        psi_like:       σ_Bp_out · (2π)^e_Bp_out / (σ_Bp_in · (2π)^e_Bp_in)
        dodpsi_like:    1 / psi_like
        one_like:       1

    Args:
        label: COCOS transformation label (e.g., 'ip_like', 'b0_like').
        cocos_in: Source COCOS convention.
        cocos_out: Target COCOS convention.

    Returns:
        Sign factor (+1 or -1) for simple cases, or a float for psi_like
        when e_Bp differs.
    """
    from imas_codex.cocos.calculator import cocos_to_parameters

    if cocos_in == cocos_out:
        return 1

    p_in = cocos_to_parameters(cocos_in)
    p_out = cocos_to_parameters(cocos_out)

    if label == "one_like":
        return 1
    elif label == "ip_like":
        return (p_out.sigma_r_phi_z * p_out.sigma_bp) // (
            p_in.sigma_r_phi_z * p_in.sigma_bp
        )
    elif label in ("b0_like", "tor_angle_like"):
        return p_out.sigma_r_phi_z // p_in.sigma_r_phi_z
    elif label == "pol_angle_like":
        return p_out.sigma_rho_theta_phi // p_in.sigma_rho_theta_phi
    elif label == "q_like":
        return (p_out.sigma_rho_theta_phi * p_out.sigma_r_phi_z) // (
            p_in.sigma_rho_theta_phi * p_in.sigma_r_phi_z
        )
    elif label == "psi_like":
        factor = p_out.sigma_bp / p_in.sigma_bp
        two_pi_factor = (2 * math.pi) ** (p_out.e_bp - p_in.e_bp)
        return factor * two_pi_factor
    elif label == "dodpsi_like":
        psi = cocos_sign("psi_like", cocos_in=cocos_in, cocos_out=cocos_out)
        return 1 / psi
    else:
        logger.warning("Unknown COCOS label '%s', returning 1", label)
        return 1


def compose_transform(slots: BindingTransformSlots) -> str:
    """Compose a binding expression only after every typed slot is settled."""
    from imas_codex.ids.models import CocosLabel, SlotResolution

    names = ("sign", "scale_factor", "unit_conversion", "cocos_label", "index_layout")
    open_slots = [
        name
        for name in names
        if (slot := getattr(slots, name)).value is None
        or slot.settled_by
        not in (
            SlotResolution.CODE,
            SlotResolution.JEV,
            SlotResolution.LOCAL,
        )
        or not slot.evidence
    ]
    if open_slots:
        raise ValueError(f"open transform slot(s): {', '.join(open_slots)}")

    layout = slots.index_layout.value
    expression = "value"
    if layout.kind == "index":
        expression += f"[{layout.index}]"
    elif layout.kind == "slice":
        bounds = [layout.start, layout.stop, layout.step]
        end = 3 if layout.step is not None else 2
        expression += (
            "[" + ":".join("" if x is None else str(x) for x in bounds[:end]) + "]"
        )

    units = slots.unit_conversion.value
    if units.source_unit is not None and units.source_unit != units.target_unit:
        convert_units(1.0, units.source_unit, units.target_unit)
        expression = (
            f"convert_units({expression}, {units.source_unit!r}, {units.target_unit!r})"
        )

    label = slots.cocos_label.value
    sign = slots.sign.value
    scale = slots.scale_factor.value
    if label != CocosLabel.NONE:
        if slots.cocos_in is None or slots.cocos_out is None:
            raise ValueError("cocos_label requires cocos_in and cocos_out")
        factor = cocos_sign(
            label.value, cocos_in=slots.cocos_in, cocos_out=slots.cocos_out
        )
        if sign != (-1 if factor < 0 else 1) or not math.isclose(
            scale, abs(factor), rel_tol=1e-12
        ):
            raise ValueError("sign or scale_factor disagrees with cocos_label")
        expression = (
            f"({expression} * cocos_sign({label.value!r}, "
            f"cocos_in={slots.cocos_in}, cocos_out={slots.cocos_out}))"
        )
    elif sign != 1 or scale != 1:
        multiplier = sign * scale
        expression = f"({expression} * {multiplier!r})"
    return expression


def execute_transform(value: Any, transform_expression: str | None) -> Any:
    """Execute a mapping's transform_expression.

    The transform_expression is a Python expression evaluated with 'value' as
    the input variable. The expression has access to math, numpy, and
    imas_codex utility functions.

    This follows the same trust model as DataAccess templates — the code
    is authored by agents/humans through the mapping lifecycle, not from
    untrusted external input.

    Args:
        value: Input value to transform.
        transform_expression: Python expression string. If None or "value",
            returns the input unchanged (identity transform).

    Returns:
        Transformed value.
    """
    if not transform_expression or transform_expression == "value":
        return value

    np = _get_numpy()
    context: dict[str, Any] = {
        "value": value,
        "math": math,
        "abs": abs,
        "float": float,
        "int": int,
        "str": str,
        "len": len,
        "convert_units": convert_units,
        "cocos_sign": cocos_sign,
    }
    if np is not None:
        context["np"] = np

    return eval(transform_expression, {"__builtins__": {}}, context)  # noqa: S307


def set_nested(obj: Any, dotted_path: str, value: Any) -> None:
    """Set a value on an imas-python object using a dotted path.

    Handles nested structures like 'geometry.rectangle.r' by traversing
    getattr chain and setting the final attribute. Also supports array
    indexing like 'position[0].r'.
    """
    import re

    parts = re.split(r"\.", dotted_path)
    current = obj
    for part in parts[:-1]:
        # Handle array indices like 'position[0]'
        match = re.match(r"(\w+)\[(\d+)\]", part)
        if match:
            attr_name, idx = match.group(1), int(match.group(2))
            current = getattr(current, attr_name)[idx]
        else:
            current = getattr(current, part)

    # Set the final attribute, handling array index on last part
    last = parts[-1]
    match = re.match(r"(\w+)\[(\d+)\]", last)
    if match:
        attr_name, idx = match.group(1), int(match.group(2))
        getattr(current, attr_name)[idx] = value
    else:
        setattr(current, last, value)
