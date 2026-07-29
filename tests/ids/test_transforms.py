"""Tests for IDS transform execution engine."""

from __future__ import annotations

import math

import pytest

from imas_codex.ids.transforms import (
    cocos_sign,
    convert_units,
    execute_transform,
    set_nested,
)


class TestExecuteTransform:
    def test_identity_none(self):
        assert execute_transform(42, None) == 42

    def test_identity_value(self):
        assert execute_transform(42, "value") == 42

    def test_scale(self):
        assert execute_transform(5.0, "value * 1000.0") == 5000.0

    def test_negate(self):
        assert execute_transform(3.0, "-value") == -3.0

    def test_float_cast(self):
        assert execute_transform("42", "float(value)") == 42.0

    def test_str_cast(self):
        assert execute_transform(42, "str(value)") == "42"

    def test_math_sqrt(self):
        result = execute_transform(16.0, "math.sqrt(value)")
        assert result == pytest.approx(4.0)

    def test_abs_builtin(self):
        assert execute_transform(-5, "abs(value)") == 5

    def test_complex_expression(self):
        result = execute_transform(2.0, "value ** 2 + 1")
        assert result == pytest.approx(5.0)

    def test_no_builtins_access(self):
        with pytest.raises(NameError):
            execute_transform(1, "__import__('os')")


class TestSetNested:
    def test_simple_attr(self):
        class Obj:
            pass

        obj = Obj()
        set_nested(obj, "name", "test")
        assert obj.name == "test"

    def test_dotted_path(self):
        class Inner:
            pass

        class Mid:
            def __init__(self):
                self.inner = Inner()

        class Obj:
            def __init__(self):
                self.mid = Mid()

        obj = Obj()
        set_nested(obj, "mid.inner.value", 42)
        assert obj.mid.inner.value == 42


class TestConvertUnits:
    def test_meters_to_cm(self):
        result = convert_units(1.0, "m", "cm")
        assert result == pytest.approx(100.0)

    def test_same_units(self):
        result = convert_units(5.0, "m", "m")
        assert result == pytest.approx(5.0)


class TestCocosSign:
    """Tests for cocos_sign factor computation."""

    @pytest.mark.parametrize(
        "label",
        [
            "ip_like",
            "b0_like",
            "tor_angle_like",
            "pol_angle_like",
            "q_like",
            "psi_like",
            "dodpsi_like",
            "one_like",
        ],
    )
    def test_same_cocos_returns_one(self, label):
        assert cocos_sign(label, cocos_in=17, cocos_out=17) == 1

    @pytest.mark.parametrize(
        ("label", "expected"),
        [
            pytest.param("ip_like", 1, id="plasma-current"),
            pytest.param("b0_like", 1, id="toroidal-field-or-f"),
            pytest.param("tor_angle_like", 1, id="toroidal-angle"),
            pytest.param("pol_angle_like", -1, id="poloidal-angle"),
            pytest.param("q_like", -1, id="safety-factor"),
            pytest.param("psi_like", 2 * math.pi, id="poloidal-flux"),
            pytest.param(
                "dodpsi_like",
                1 / (2 * math.pi),
                id="poloidal-flux-derivative",
            ),
            pytest.param("one_like", 1, id="invariant"),
        ],
    )
    def test_cocos_3_to_17_factor_matrix(self, label, expected):
        assert cocos_sign(label, cocos_in=3, cocos_out=17) == pytest.approx(expected)

    @pytest.mark.parametrize(
        "label",
        [
            "ip_like",
            "b0_like",
            "tor_angle_like",
            "pol_angle_like",
            "q_like",
            "psi_like",
            "dodpsi_like",
            "one_like",
        ],
    )
    def test_cocos_3_and_17_factors_are_reciprocal(self, label):
        forward = cocos_sign(label, cocos_in=3, cocos_out=17)
        inverse = cocos_sign(label, cocos_in=17, cocos_out=3)

        assert forward * inverse == pytest.approx(1)

    def test_probe_orientation_angles_3_to_17(self):
        assert cocos_sign("pol_angle_like", cocos_in=3, cocos_out=17) == -1
        assert cocos_sign("tor_angle_like", cocos_in=3, cocos_out=17) == 1

    def test_ip_like_11_to_17(self):
        # σ_Bp flips between COCOS 11 and 17, so ip_like = -1
        assert cocos_sign("ip_like", cocos_in=11, cocos_out=17) == -1

    def test_ip_like_17_to_11(self):
        assert cocos_sign("ip_like", cocos_in=17, cocos_out=11) == -1

    def test_b0_like_11_to_17(self):
        # σ_RφZ is the same for COCOS 11 and 17, so b0_like = +1
        assert cocos_sign("b0_like", cocos_in=11, cocos_out=17) == 1

    def test_tor_angle_like_11_to_17(self):
        assert cocos_sign("tor_angle_like", cocos_in=11, cocos_out=17) == 1

    def test_pol_angle_like_11_to_17(self):
        assert cocos_sign("pol_angle_like", cocos_in=11, cocos_out=17) == 1

    def test_q_like_11_to_17(self):
        assert cocos_sign("q_like", cocos_in=11, cocos_out=17) == 1

    def test_one_like(self):
        assert cocos_sign("one_like", cocos_in=1, cocos_out=17) == 1

    def test_psi_like_11_to_17(self):
        # σ_Bp flips, same e_Bp → factor = -1
        assert cocos_sign("psi_like", cocos_in=11, cocos_out=17) == -1

    def test_psi_like_1_to_11(self):
        # e_Bp changes (0→1), σ_Bp is unchanged.
        result = cocos_sign("psi_like", cocos_in=1, cocos_out=11)
        assert result == pytest.approx(2 * math.pi)

    def test_dodpsi_like_11_to_17(self):
        # Inverse of psi_like(11→17) = inverse of -1 = -1
        assert cocos_sign("dodpsi_like", cocos_in=11, cocos_out=17) == pytest.approx(
            -1.0
        )

    def test_ip_like_2_to_17(self):
        # COCOS 2: σ_RφZ=-1, σ_Bp=+1
        # COCOS 17: σ_RφZ=+1, σ_Bp=-1
        # ip_like = (+1*-1) / (-1*+1) = -1 / -1 = 1
        assert cocos_sign("ip_like", cocos_in=2, cocos_out=17) == 1

    def test_unknown_label_returns_one(self):
        assert cocos_sign("unknown_label", cocos_in=11, cocos_out=17) == 1


class TestCocosSignInTransform:
    """Test cocos_sign is accessible inside execute_transform."""

    def test_cocos_sign_in_transform_expression(self):
        result = execute_transform(
            100.0, "value * cocos_sign('ip_like', cocos_in=11, cocos_out=17)"
        )
        assert result == pytest.approx(-100.0)

    def test_cocos_sign_identity(self):
        result = execute_transform(
            100.0, "value * cocos_sign('b0_like', cocos_in=11, cocos_out=17)"
        )
        assert result == pytest.approx(100.0)


class TestSetNestedArrayIndex:
    """Test set_nested with indexed array paths like position[0].r."""

    def test_simple_array_index(self):
        class Inner:
            r = 0.0
            z = 0.0

        class Outer:
            position = [Inner()]

        obj = Outer()
        set_nested(obj, "position[0].r", 1.5)
        assert obj.position[0].r == 1.5

    def test_array_index_preserves_other_fields(self):
        class Inner:
            r = 0.0
            z = 0.0

        class Outer:
            position = [Inner()]

        obj = Outer()
        set_nested(obj, "position[0].r", 1.5)
        set_nested(obj, "position[0].z", 2.5)
        assert obj.position[0].r == 1.5
        assert obj.position[0].z == 2.5
