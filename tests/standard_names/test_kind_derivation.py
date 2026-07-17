"""Tests for structural ``kind`` derivation from a standard-name string.

``derive_kind`` must defer to ISN's authoritative base classification: a
``physical_base`` declared ``vector`` in the ISN registry yields ``vector``,
a projected component of it yields ``scalar``, and a tensor base yields
``tensor``.  Kind mirrors the ISN catalog ``Kind`` enum exactly — semantic
categories (eigenfunction / spectrum) are structurally scalar; only the
complex-part pattern layers on top of the structural classification.

Regression: before the ISN rewrite, ``derive_kind`` never returned
``vector`` (its hand-maintained base list was stale and no code path
returned it), so the ``MAGNITUDE_OF`` graph edge — which matches on
``v.kind = 'vector'`` — could never link a magnitude to its parent vector.
"""

from __future__ import annotations

import pytest

pytest.importorskip("imas_standard_names")

from imas_codex.standard_names.kind_derivation import (  # noqa: E402
    derive_kind,
    to_isn_kind,
)


class TestVectorBases:
    """ISN-declared vector bases classify as ``vector``."""

    @pytest.mark.parametrize(
        "name",
        [
            "magnetic_field",
            "electric_field",
            "electron_velocity",
            "ion_velocity",
            "current_density",
            "force",
            "angular_velocity",
            "wave_vector",
            "magnetic_vector_potential",
        ],
    )
    def test_vector_base_is_vector(self, name: str) -> None:
        assert derive_kind(name) == "vector"


class TestProjectionIsScalar:
    """A projected component / coordinate of a vector is a scalar."""

    @pytest.mark.parametrize(
        "name",
        [
            "radial_magnetic_field",
            "toroidal_magnetic_field",
            "poloidal_current_density",
            "parallel_electron_velocity",
            "toroidal_magnetic_field_at_magnetic_axis",
            # Coordinate axis of a point (geometric base) is scalar too.
            "radial_coordinate_of_magnetic_axis",
            "vertical_coordinate_of_magnetic_axis",
        ],
    )
    def test_projection_is_scalar(self, name: str) -> None:
        assert derive_kind(name) == "scalar"

    def test_magnitude_of_vector_is_scalar(self) -> None:
        # A magnitude reduction collapses a vector to a scalar.
        assert derive_kind("magnetic_field_magnitude") == "scalar"
        assert derive_kind("velocity_magnitude") == "scalar"


class TestScalarBases:
    """ISN-declared scalar bases classify as ``scalar``."""

    @pytest.mark.parametrize(
        "name",
        [
            "plasma_current",
            "electron_temperature",
            "electron_density",
            "safety_factor",
            "poloidal_magnetic_flux",
            "total_plasma_pressure",
        ],
    )
    def test_scalar_base_is_scalar(self, name: str) -> None:
        assert derive_kind(name) == "scalar"


class TestTensor:
    def test_isn_tensor_base(self) -> None:
        # metric_tensor is the one ISN-registered tensor base.
        assert derive_kind("metric_tensor") == "tensor"

    @pytest.mark.parametrize(
        "name",
        [
            "reynolds_stress_tensor_real_part",
            "maxwell_stress_tensor",
            "pressure_tensor",
        ],
    )
    def test_codex_tensor_compound(self, name: str) -> None:
        # ISN does not register these as bases; the `_tensor` heuristic keeps
        # them classified as tensors (and tensor precedes complex, so the
        # real_part variant is still a tensor).
        assert derive_kind(name) == "tensor"


class TestCodexExtendedKinds:
    def test_eigenfunction_is_structurally_scalar(self) -> None:
        assert derive_kind("plasma_displacement_eigenfunction") == "scalar"
        assert derive_kind("eigenfunction") == "scalar"

    def test_spectrum_is_structurally_scalar(self) -> None:
        assert derive_kind("magnetic_fluctuation_spectrum") == "scalar"
        assert derive_kind("density_fluctuation_spectrum") == "scalar"

    @pytest.mark.parametrize(
        "name",
        [
            "perturbed_electrostatic_potential_real_part",
            "perturbed_mass_density_imaginary_part",
            # A complex part takes precedence over the projected-component
            # scalar rule (matches pre-rewrite behaviour).
            "radial_magnetic_field_real_part",
        ],
    )
    def test_complex_part(self, name: str) -> None:
        assert derive_kind(name) == "complex"


class TestFallback:
    def test_unparseable_defaults_scalar(self) -> None:
        # A prefix magnitude form does not parse in ISN; it is a scalar, and
        # the default catches it.
        assert derive_kind("magnitude_of_electron_velocity") == "scalar"

    def test_empty_defaults_scalar(self) -> None:
        assert derive_kind("") == "scalar"


class TestToIsnKind:
    @pytest.mark.parametrize(
        ("local", "expected"),
        [
            ("scalar", "scalar"),
            ("vector", "vector"),
            ("tensor", "tensor"),
            ("complex", "complex"),
            ("metadata", "metadata"),
            # Legacy graph values from the retired extended vocabulary
            # still collapse to scalar.
            ("eigenfunction", "scalar"),
            ("spectrum", "scalar"),
            (None, "scalar"),
            ("bogus", "scalar"),
        ],
    )
    def test_mapping(self, local: str | None, expected: str) -> None:
        assert to_isn_kind(local) == expected


class TestKindVocabularyConsistency:
    """One kind vocabulary, three carriers — LinkML is the generated source.

    The LLM-facing ``EntryKind`` Literal (kept flat so structured-output
    schemas carry a plain enum, no $ref), the LinkML-generated
    ``StandardNameKind`` graph enum, and the ISN catalog ``Kind`` enum must
    never drift apart.
    """

    def test_three_way_equality(self) -> None:
        from typing import get_args

        from imas_standard_names.models import Kind as IsnKind

        from imas_codex.graph.models import StandardNameKind
        from imas_codex.standard_names.models import EntryKind

        literal = set(get_args(EntryKind))
        graph = {e.value for e in StandardNameKind}
        isn = {k.value for k in IsnKind}
        assert literal == graph == isn
