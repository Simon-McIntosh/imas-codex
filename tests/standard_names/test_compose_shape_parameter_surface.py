"""Deterministic surface injection for dimensionless shape-parameter leaves.

Plasma shape descriptors (triangularity, elongation, squareness) are only
meaningful *of* a surface. When the composer emits a bare leaf, the worker
injects the surface locus deterministically from the source DD path so the
name is always surface-explicit and the boundary/profile siblings de-conflate
into distinct names. These tests exercise the pure path→surface map, the
mutation/guard contract, and the end-to-end composed name. They do not call
the LLM.
"""

from __future__ import annotations

import inspect
import re

import pytest

from imas_codex.standard_names import workers as _workers
from imas_codex.standard_names.models import GrammarSegments, StandardNameCandidate
from imas_codex.standard_names.workers import (
    _SHAPE_PARAMETER_BASES,
    _inject_shape_parameter_surface,
    _is_attachment_consistent,
    _shape_parameter_surface,
    normalize_spelling,
)


def _candidate(
    base: str, *, kind: str = "quantity", qualifiers: list[str] | None = None
) -> StandardNameCandidate:
    seg = GrammarSegments(base_token=base, base_kind=kind, qualifiers=qualifiers or [])
    return StandardNameCandidate(source_id="x", segments=seg, reason="t")


# ---------------------------------------------------------------------------
# Pure path → surface mapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path,expected",
    [
        ("equilibrium/time_slice/boundary/triangularity", "plasma_boundary"),
        ("equilibrium/time_slice/boundary/triangularity_upper", "plasma_boundary"),
        (
            "equilibrium/time_slice/boundary_separatrix/triangularity_outer",
            "plasma_boundary",
        ),
        ("equilibrium/time_slice/profiles_1d/triangularity_lower", "flux_surface"),
        ("equilibrium/time_slice/profiles_1d/elongation", "flux_surface"),
        ("gyrokinetics/flux_surface/elongation", "flux_surface"),
        # Control targets and unrecognised paths fall to the boundary default.
        ("pulse_schedule/position_control/triangularity", "plasma_boundary"),
        ("", "plasma_boundary"),
        (None, "plasma_boundary"),
    ],
)
def test_surface_from_path(path, expected):
    assert _shape_parameter_surface(path) == expected


# ---------------------------------------------------------------------------
# Injection contract — fires, skips, and preserves
# ---------------------------------------------------------------------------


def test_injects_boundary_surface_on_bare_shape_param():
    c = _candidate("triangularity")
    applied = _inject_shape_parameter_surface(
        c, "equilibrium/time_slice/boundary/triangularity"
    )
    assert applied is True
    assert c.segments.locus_token == "plasma_boundary"
    assert c.segments.locus_relation == "of"
    assert c.segments.locus_type == "geometry"


def test_injects_flux_surface_for_profiles_path():
    c = _candidate("elongation")
    applied = _inject_shape_parameter_surface(
        c, "equilibrium/time_slice/profiles_1d/elongation"
    )
    assert applied is True
    assert c.segments.locus_token == "flux_surface"


def test_preserves_composer_supplied_locus():
    c = _candidate("triangularity")
    c.segments.locus_token = "magnetic_axis"
    c.segments.locus_relation = "of"
    c.segments.locus_type = "entity"
    applied = _inject_shape_parameter_surface(
        c, "equilibrium/time_slice/boundary/triangularity"
    )
    assert applied is False
    assert c.segments.locus_token == "magnetic_axis"


def test_skips_non_shape_parameter_base():
    c = _candidate("temperature")
    applied = _inject_shape_parameter_surface(
        c, "core_profiles/profiles_1d/electrons/temperature"
    )
    assert applied is False
    assert c.segments.locus_token is None


def test_idempotent():
    c = _candidate("squareness", qualifiers=["lower", "inner"])
    path = "equilibrium/time_slice/boundary/squareness_lower_inner"
    assert _inject_shape_parameter_surface(c, path) is True
    # Second call is a no-op — locus already set.
    assert _inject_shape_parameter_surface(c, path) is False
    assert c.segments.locus_token == "plasma_boundary"


def test_closed_set_membership():
    assert _SHAPE_PARAMETER_BASES == frozenset(
        {"triangularity", "elongation", "squareness"}
    )


# ---------------------------------------------------------------------------
# End-to-end — injected candidates compose to surface-explicit leaves
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "base,qualifiers,path,expected_name",
    [
        (
            "triangularity",
            [],
            "equilibrium/time_slice/boundary/triangularity",
            "triangularity_of_plasma_boundary",
        ),
        (
            "triangularity",
            ["upper"],
            "equilibrium/time_slice/boundary/triangularity_upper",
            "upper_triangularity_of_plasma_boundary",
        ),
        (
            "triangularity",
            ["upper"],
            "equilibrium/time_slice/profiles_1d/triangularity_upper",
            "upper_triangularity_of_flux_surface",
        ),
        (
            "squareness",
            ["lower", "inner"],
            "equilibrium/time_slice/boundary/squareness_lower_inner",
            "lower_inner_squareness_of_plasma_boundary",
        ),
        (
            "elongation",
            [],
            "equilibrium/time_slice/profiles_1d/elongation",
            "elongation_of_flux_surface",
        ),
    ],
)
def test_end_to_end_composed_name(base, qualifiers, path, expected_name):
    c = _candidate(base, qualifiers=qualifiers)
    _inject_shape_parameter_surface(c, path)
    assert normalize_spelling(c.compose_name()) == expected_name


# ---------------------------------------------------------------------------
# Attachment surface-consistency guard — the de-conflation must survive the
# attachment step (a flux-surface source must not attach to a boundary name)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source_id,sn_name,ok",
    [
        # flux-surface source must NOT attach to a boundary name (the bug)
        (
            "equilibrium/time_slice/profiles_1d/squareness_lower_inner",
            "lower_inner_squareness_of_plasma_boundary",
            False,
        ),
        # boundary source -> boundary name: consistent
        (
            "equilibrium/time_slice/boundary/squareness_lower_inner",
            "lower_inner_squareness_of_plasma_boundary",
            True,
        ),
        # separatrix IS the boundary -> consistent with a boundary name
        (
            "equilibrium/time_slice/boundary_separatrix/squareness_upper_outer",
            "upper_outer_squareness_of_plasma_boundary",
            True,
        ),
        # profile source -> flux-surface name: consistent
        (
            "equilibrium/time_slice/profiles_1d/triangularity_upper",
            "upper_triangularity_of_flux_surface",
            True,
        ),
        # boundary source must NOT attach to a flux-surface name
        (
            "equilibrium/time_slice/boundary/triangularity_upper",
            "upper_triangularity_of_flux_surface",
            False,
        ),
    ],
)
def test_attachment_surface_consistency(source_id, sn_name, ok):
    consistent, reason = _is_attachment_consistent(source_id, sn_name)
    assert consistent is ok, reason


def test_attachment_guard_ignores_non_shape_names():
    # A non-shape name carries no surface constraint here.
    consistent, _ = _is_attachment_consistent(
        "core_profiles/profiles_1d/electrons/temperature",
        "electron_temperature",
    )
    assert consistent is True


# ---------------------------------------------------------------------------
# Regression guard — the injection must be wired into EVERY compose persist
# path. (The bug: it was only in the linear path, not the pooled path `sn run`
# uses, so it never fired in production.)
# ---------------------------------------------------------------------------


def test_injection_precedes_every_compose_persist_site():
    """Every ``name_id = normalize_spelling(c.compose_name())`` persist site
    must be preceded by an ``_inject_shape_parameter_surface(c, ...)`` call in
    the same function body, so both compose paths force the surface locus."""
    src = inspect.getsource(_workers)
    lines = src.splitlines()
    persist_re = re.compile(r"name_id\s*=\s*normalize_spelling\(c\.compose_name\(\)\)")
    persist_lines = [i for i, ln in enumerate(lines) if persist_re.search(ln)]
    assert len(persist_lines) >= 2, (
        "expected at least two compose persist sites (linear + pooled); "
        f"found {len(persist_lines)}"
    )
    for i in persist_lines:
        window = "\n".join(lines[max(0, i - 12) : i])
        assert "_inject_shape_parameter_surface(c," in window, (
            f"compose persist site at source line ~{i + 1} is not preceded by "
            "_inject_shape_parameter_surface — a compose path would emit bare "
            "shape-parameter leaves"
        )


def test_boundary_and_profile_siblings_deconflate():
    """The same qualified leaf from boundary vs profiles_1d → distinct names."""
    boundary = _candidate("triangularity", qualifiers=["upper"])
    profile = _candidate("triangularity", qualifiers=["upper"])
    _inject_shape_parameter_surface(
        boundary, "equilibrium/time_slice/boundary/triangularity_upper"
    )
    _inject_shape_parameter_surface(
        profile, "equilibrium/time_slice/profiles_1d/triangularity_upper"
    )
    n_boundary = normalize_spelling(boundary.compose_name())
    n_profile = normalize_spelling(profile.compose_name())
    assert n_boundary != n_profile
    assert n_boundary == "upper_triangularity_of_plasma_boundary"
    assert n_profile == "upper_triangularity_of_flux_surface"
