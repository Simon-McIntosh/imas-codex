"""COCOS calculation from equilibrium quantities.

Based on Sauter & Medvedev, "Tokamak coordinate conventions: COCOS",
Computer Physics Communications, 184(2):293-302, February 2013.
DOI: 10.1016/j.cpc.2012.09.010

Table I defines 16 COCOS values (1-8, 11-18) via four parameters:
- σBp: Poloidal flux sign convention (+1 or -1)
- eBp: Whether ψ is divided by 2π (0) or full ψ (1)
- σRφZ: Cylindrical coordinate handedness (+1 for R,φ,Z; -1 for R,Z,φ)
- σρθφ: Poloidal coordinate handedness (+1 or -1)

The module provides composable functions to:
1. Calculate COCOS from physics quantities (Eq. 23 in paper)
2. Convert between COCOS conventions
3. Validate COCOS consistency

Known COCOS values from the paper:
- CHEASE, ONETWO: 2
- ITER/IMAS DD v3.x: 11
- IMAS DD v4.x: 17
- TCV/LIUQE: 17
- ORB5: 3 (with negative Ip normalization)
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

# Valid COCOS values per Sauter paper Table I
VALID_COCOS = frozenset({1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18})


@dataclass(frozen=True)
class COCOSParameters:
    """The four COCOS-determining parameters from Table I.

    These parameters fully specify a COCOS convention and enable
    transformations between conventions.

    Attributes:
        sigma_bp: Poloidal flux sign convention (+1 or -1).
            +1 if ψ increases from axis to edge with positive Ip.
        e_bp: Whether ψ is divided by 2π (0) or full ψ (1).
            COCOS 1-8 have e_bp=0, COCOS 11-18 have e_bp=1.
        sigma_r_phi_z: Cylindrical coordinate handedness.
            +1 for (R,φ,Z) right-handed, -1 for (R,Z,φ) right-handed.
        sigma_rho_theta_phi: Poloidal coordinate handedness.
            +1 for (ρ,θ,φ) right-handed, -1 otherwise.
    """

    sigma_bp: Literal[-1, 1]
    e_bp: Literal[0, 1]
    sigma_r_phi_z: Literal[-1, 1]
    sigma_rho_theta_phi: Literal[-1, 1]

    @property
    def cocos(self) -> int:
        """Compute COCOS value from parameters via table lookup.

        The mapping follows Table I of Sauter & Medvedev.
        Uses reverse lookup for correctness.
        """
        # Reverse lookup from parameters to COCOS
        for cocos_val, (s_bp, e_bp, s_rpz, s_rtp) in _COCOS_TABLE.items():
            if (
                self.sigma_bp == s_bp
                and self.e_bp == e_bp
                and self.sigma_r_phi_z == s_rpz
                and self.sigma_rho_theta_phi == s_rtp
            ):
                return cocos_val
        # Should never reach here if _COCOS_TABLE is complete
        msg = f"No COCOS found for parameters: {self}"
        raise ValueError(msg)


# Table I lookup: COCOS -> (σBp, eBp, σRφZ, σρθφ)
_COCOS_TABLE: dict[int, tuple[int, int, int, int]] = {
    # σBp, eBp, σRφZ, σρθφ  — Table I of Sauter & Medvedev, CPC 184 (2013)
    1: (+1, 0, +1, +1),
    2: (+1, 0, -1, +1),
    3: (-1, 0, +1, -1),
    4: (-1, 0, -1, -1),
    5: (+1, 0, +1, -1),
    6: (+1, 0, -1, -1),
    7: (-1, 0, +1, +1),
    8: (-1, 0, -1, +1),
    11: (+1, 1, +1, +1),
    12: (+1, 1, -1, +1),
    13: (-1, 1, +1, -1),
    14: (-1, 1, -1, -1),
    15: (+1, 1, +1, -1),
    16: (+1, 1, -1, -1),
    17: (-1, 1, +1, +1),
    18: (-1, 1, -1, +1),
}


def cocos_to_parameters(cocos: int) -> COCOSParameters:
    """Convert COCOS integer to parameter object.

    Args:
        cocos: COCOS value (1-8 or 11-18)

    Returns:
        COCOSParameters with the four determining parameters

    Raises:
        ValueError: If cocos is not a valid value
    """
    if cocos not in VALID_COCOS:
        msg = f"Invalid COCOS value {cocos}. Valid values: 1-8, 11-18"
        raise ValueError(msg)

    sigma_bp, e_bp, sigma_r_phi_z, sigma_rho_theta_phi = _COCOS_TABLE[cocos]
    return COCOSParameters(
        sigma_bp=sigma_bp,
        e_bp=e_bp,
        sigma_r_phi_z=sigma_r_phi_z,
        sigma_rho_theta_phi=sigma_rho_theta_phi,
    )


def cocos_from_dd_version(version: str) -> int:
    """Get COCOS from IMAS Data Dictionary version.

    First attempts to read the COCOS value from the DD XML directly.
    Falls back to version-based heuristic: COCOS 11 for DD < 4.0.0,
    COCOS 17 for DD >= 4.0.0.

    Args:
        version: DD version string (e.g., "3.39.0", "4.0.0")

    Returns:
        COCOS value (11 for DD < 4.0.0, 17 for DD >= 4.0.0)
    """
    try:
        import imas.dd_zip

        tree = imas.dd_zip.dd_etree(version)
        root = tree.getroot() if hasattr(tree, "getroot") else tree
        cocos_el = root.find("cocos")
        if cocos_el is not None:
            return int(cocos_el.text)
    except Exception:
        pass

    # Fallback for versions without <cocos> in XML (pre-3.35.0)
    from packaging.version import Version

    v = Version(version)
    return 17 if v >= Version("4.0.0") else 11


@dataclass(frozen=True)
class SignObservation:
    """A measured sign from one shot, before any convention is assigned.

    ``kind`` identifies the relation being measured. Flux and psi gradients
    have sign σIp σBp; pressure gradients have its opposite. The q sign is
    σIp σB0 σρθφ. A zero or missing sign is not a measurement.
    """

    shot: int
    ip_sign: Literal[-1, 1]
    kind: Literal["flux_loop_response", "psi_gradient", "dp_dpsi", "q"]
    sign: Literal[-1, 1]
    b0_sign: Literal[-1, 1] | None = None

    def __post_init__(self) -> None:
        if self.ip_sign not in (-1, 1) or self.sign not in (-1, 1):
            raise ValueError("Observed signs and plasma-current signs must be ±1")
        if self.kind == "q" and self.b0_sign not in (-1, 1):
            raise ValueError("A q observation needs a measured toroidal-field sign")


@dataclass(frozen=True)
class COCOSSurvivorResult:
    """Conventions consistent with every observation and declaration."""

    survivors: frozenset[int]
    status: Literal["consistent", "polarity-incomplete", "not-cocos-expressible"]
    confidence: float
    coefficient_basis: dict[str, str]
    declarations: dict[str, int]

    @property
    def cocos(self) -> int | None:
        """The convention only when all evidence identifies one."""
        return next(iter(self.survivors)) if len(self.survivors) == 1 else None

    def factor_for_class(self, label: str, *, cocos_out: int) -> int | float | None:
        """Return a conversion factor only when every survivor agrees."""
        from imas_codex.ids.transforms import cocos_sign

        factors = {
            cocos_sign(label, cocos_in=cocos, cocos_out=cocos_out)
            for cocos in self.survivors
        }
        return next(iter(factors)) if len(factors) == 1 else None


def score_cocos_survivors(
    observations: list[SignObservation],
    *,
    e_bp: Literal[0, 1] | None = None,
    sigma_r_phi_z: Literal[-1, 1] | None = None,
    sigma_rho_theta_phi: Literal[-1, 1] | None = None,
) -> COCOSSurvivorResult:
    """Filter all canonical conventions using measured signs and declarations.

    Flux normalisation and toroidal handedness require declarations; values
    cannot establish either. An empty survivor set is reported explicitly.
    """
    if e_bp not in (None, 0, 1):
        raise ValueError("e_bp must be 0 or 1")
    if sigma_r_phi_z not in (None, -1, 1):
        raise ValueError("sigma_r_phi_z must be ±1")
    if sigma_rho_theta_phi not in (None, -1, 1):
        raise ValueError("sigma_rho_theta_phi must be ±1")

    declarations = {
        name: value
        for name, value in (
            ("e_bp", e_bp),
            ("sigma_r_phi_z", sigma_r_phi_z),
            ("sigma_rho_theta_phi", sigma_rho_theta_phi),
        )
        if value is not None
    }
    survivors = set(VALID_COCOS)
    for cocos in tuple(survivors):
        params = cocos_to_parameters(cocos)
        if any(getattr(params, name) != value for name, value in declarations.items()):
            survivors.discard(cocos)
            continue
        for obs in observations:
            if obs.kind == "q":
                expected = obs.ip_sign * obs.b0_sign * params.sigma_rho_theta_phi
            else:
                expected = obs.ip_sign * params.sigma_bp
                if obs.kind == "dp_dpsi":
                    expected = -expected
            if obs.sign != expected:
                survivors.discard(cocos)
                break

    measured = {obs.kind for obs in observations}
    basis = {
        "sigma_bp": "measured"
        if measured & {"flux_loop_response", "psi_gradient", "dp_dpsi"}
        else "unmeasured",
        "sigma_rho_theta_phi": "measurable-as-product"
        if "q" in measured
        else "unmeasured",
        "e_bp": "declared" if e_bp is not None else "requires-declaration",
        "sigma_r_phi_z": "declared"
        if sigma_r_phi_z is not None
        else "requires-declaration",
    }
    polarities = {obs.ip_sign for obs in observations}
    if not survivors:
        status = "not-cocos-expressible"
        confidence = 0.0
    elif len(polarities) < 2:
        status = "polarity-incomplete"
        confidence = 0.75
    else:
        status = "consistent"
        confidence = 0.85
    if survivors:
        confidence -= 0.1 * (e_bp is None)
        confidence -= 0.1 * (sigma_r_phi_z is None)
        confidence -= 0.1 * (not measured & {"q"} and sigma_rho_theta_phi is None)
    return COCOSSurvivorResult(
        frozenset(survivors), status, max(0.0, confidence), basis, declarations
    )


@dataclass(frozen=True)
class COCOSDetermination:
    """A unique convention with the assumptions used to obtain it.

    Iteration preserves the existing ``cocos, confidence`` call pattern.
    """

    cocos: int
    confidence: float
    assumed_declarations: dict[str, int]

    def __iter__(self) -> Iterator[int | float]:
        yield self.cocos
        yield self.confidence


def determine_cocos(
    psi_axis: float,
    psi_edge: float,
    ip: float,
    b0: float,
    q: float | None = None,
    dp_dpsi: float | None = None,
    *,
    assume_r_phi_z_right_handed: bool = True,
    e_bp: Literal[0, 1] | None = None,
) -> COCOSDetermination:
    """Identify one convention while exposing declarations that data cannot make.

    The optional ``e_bp`` is an external declaration. For older callers that
    omit it, the magnitude heuristic is retained as a named, low-confidence
    assumption. Toroidal handedness is likewise an explicit assumption.
    """
    values = (psi_axis, psi_edge, ip, b0)
    if not all(math.isfinite(value) for value in values) or ip == 0 or b0 == 0:
        raise ValueError(
            "Finite nonzero plasma-current and field references are required"
        )
    gradient = psi_edge - psi_axis
    if gradient == 0:
        raise ValueError("A zero psi gradient cannot determine its sign")
    ip_sign = 1 if ip > 0 else -1
    b0_sign = 1 if b0 > 0 else -1
    observations = [
        SignObservation(0, ip_sign, "psi_gradient", 1 if gradient > 0 else -1)
    ]
    if dp_dpsi is not None and math.isfinite(dp_dpsi) and dp_dpsi != 0:
        observations.append(
            SignObservation(0, ip_sign, "dp_dpsi", 1 if dp_dpsi > 0 else -1)
        )
    assumed: dict[str, int] = {}
    if e_bp is None:
        e_bp = 1 if max(abs(psi_axis), abs(psi_edge)) > 0.5 else 0
        assumed["e_bp"] = e_bp
    handedness = 1 if assume_r_phi_z_right_handed else -1
    assumed["sigma_r_phi_z"] = handedness
    rho_handedness = None
    if q is not None and math.isfinite(q) and q != 0:
        observations.append(
            SignObservation(0, ip_sign, "q", 1 if q > 0 else -1, b0_sign)
        )
    else:
        rho_handedness = 1
        assumed["sigma_rho_theta_phi"] = rho_handedness
    result = score_cocos_survivors(
        observations,
        e_bp=e_bp,
        sigma_r_phi_z=handedness,
        sigma_rho_theta_phi=rho_handedness,
    )
    if result.cocos is None:
        raise ValueError(f"Equilibrium observations are {result.status}")
    confidence = max(0.0, result.confidence - 0.1 * len(assumed))
    return COCOSDetermination(result.cocos, confidence, assumed)


def validate_cocos_consistency(
    cocos: int,
    psi_axis: float,
    psi_edge: float,
    ip: float,
    b0: float,
    q: float | None = None,
    dp_dpsi: float | None = None,
) -> list[str]:
    """Validate COCOS against physics quantities.

    Checks that the provided COCOS is consistent with the equilibrium
    data according to Eq. 23 from the Sauter paper.

    Args:
        cocos: COCOS value to validate
        psi_axis: Poloidal flux at magnetic axis
        psi_edge: Poloidal flux at plasma edge
        ip: Plasma current [A]
        b0: Toroidal field at axis [T]
        q: Safety factor (optional)
        dp_dpsi: Pressure gradient (optional)

    Returns:
        List of inconsistency messages (empty if consistent)

    Example:
        >>> errors = validate_cocos_consistency(
        ...     cocos=11,
        ...     psi_axis=0.5, psi_edge=-0.2,
        ...     ip=-1e6, b0=-5.0, q=3.0
        ... )
        >>> if errors:
        ...     print("Inconsistencies found:", errors)
    """
    errors: list[str] = []

    if cocos not in VALID_COCOS:
        errors.append(f"Invalid COCOS value {cocos}")
        return errors

    params = cocos_to_parameters(cocos)
    sigma_ip = 1 if ip >= 0 else -1
    sigma_b0 = 1 if b0 >= 0 else -1

    # Check psi gradient (Eq. 23, line 5)
    expected_psi_sign = sigma_ip * params.sigma_bp
    actual_psi_sign = 1 if (psi_edge - psi_axis) >= 0 else -1
    if expected_psi_sign != actual_psi_sign:
        direction = "increasing" if actual_psi_sign == 1 else "decreasing"
        expected_dir = "increasing" if expected_psi_sign == 1 else "decreasing"
        errors.append(
            f"ψ gradient: expected {expected_dir}, got {direction} "
            f"(σIp={sigma_ip}, σBp={params.sigma_bp})"
        )

    # Check dp/dpsi if provided (Eq. 23, line 6)
    if dp_dpsi is not None and not math.isnan(dp_dpsi):
        expected_dp_sign = -sigma_ip * params.sigma_bp
        actual_dp_sign = 1 if dp_dpsi >= 0 else -1
        if expected_dp_sign != actual_dp_sign:
            errors.append(
                f"dp/dψ sign: expected {expected_dp_sign:+d}, got {actual_dp_sign:+d}"
            )

    # Check q sign if provided (Eq. 23, line 8)
    if q is not None and not math.isnan(q):
        expected_q_sign = sigma_ip * sigma_b0 * params.sigma_rho_theta_phi
        actual_q_sign = 1 if q >= 0 else -1
        if expected_q_sign != actual_q_sign:
            errors.append(
                f"q sign: expected {expected_q_sign:+d}, got {actual_q_sign:+d} "
                f"(σIp={sigma_ip}, σB0={sigma_b0}, σρθφ={params.sigma_rho_theta_phi})"
            )

    return errors


@dataclass(frozen=True)
class ValidationResult:
    """Result of COCOS validation against physics data.

    Attributes:
        is_consistent: Whether declared COCOS matches physics data
        declared_cocos: The COCOS value being validated
        calculated_cocos: COCOS inferred from physics quantities, or None when
            the observations cannot identify a convention
        confidence: Confidence in calculated COCOS (0.0-1.0)
        inconsistencies: List of specific physics inconsistencies
    """

    is_consistent: bool
    declared_cocos: int
    calculated_cocos: int | None
    confidence: float
    inconsistencies: list[str]


def validate_cocos_from_data(
    declared_cocos: int,
    *,
    # Required physics quantities
    psi_axis: float,
    psi_edge: float,
    ip: float,
    b0: float,
    # Optional - improve confidence
    q: float | None = None,
    dp_dpsi: float | None = None,
) -> ValidationResult:
    """Validate declared COCOS against physics data.

    This is deterministic - uses Eq. 23 from Sauter paper.
    No LLM involvement. Data source agnostic.

    The caller is responsible for loading physics quantities from their
    data source (MDSplus, IMAS IDS, EQDSK, HDF5, etc.).

    Args:
        declared_cocos: The COCOS value to validate
        psi_axis: Poloidal flux at magnetic axis [Wb]
        psi_edge: Poloidal flux at plasma edge [Wb]
        ip: Plasma current [A] (sign matters)
        b0: Toroidal field at axis [T] (sign matters)
        q: Safety factor at mid-radius (optional, improves confidence)
        dp_dpsi: Pressure gradient dp/dψ (optional, validates σBp)

    Returns:
        ValidationResult with consistency check, calculated COCOS,
        confidence score, and list of specific inconsistencies.

    Example:
        >>> result = validate_cocos_from_data(
        ...     declared_cocos=17,
        ...     psi_axis=0.5, psi_edge=-0.2,
        ...     ip=-1e6, b0=-5.0, q=3.0,
        ... )
        >>> if not result.is_consistent:
        ...     print(f"COCOS mismatch: declared {result.declared_cocos}, "
        ...           f"calculated {result.calculated_cocos}")
        ...     for error in result.inconsistencies:
        ...         print(f"  - {error}")
    """
    # Validate declared COCOS against physics
    inconsistencies = validate_cocos_consistency(
        cocos=declared_cocos,
        psi_axis=psi_axis,
        psi_edge=psi_edge,
        ip=ip,
        b0=b0,
        q=q,
        dp_dpsi=dp_dpsi,
    )
    try:
        calculated, confidence = determine_cocos(
            psi_axis=psi_axis,
            psi_edge=psi_edge,
            ip=ip,
            b0=b0,
            q=q,
            dp_dpsi=dp_dpsi,
        )
    except ValueError as exc:
        calculated, confidence = None, 0.0
        inconsistencies.append(str(exc))

    return ValidationResult(
        is_consistent=len(inconsistencies) == 0,
        declared_cocos=declared_cocos,
        calculated_cocos=calculated,
        confidence=confidence,
        inconsistencies=inconsistencies,
    )


# Known COCOS values for common codes (from Sauter paper Table IV and Appendix A)
KNOWN_CODE_COCOS: dict[str, int] = {
    "CHEASE": 2,
    "ONETWO": 2,
    "LIUQE": 17,
    "LIUQE.M": 17,
    "LIUQE02": 17,
    "LIUQE03": 17,
    "FBTE": 17,  # TCV convention
    "ORB5": 3,  # Uses negative Ip normalization
    # ITER/IMAS defaults
    "IMAS_DD_V3": 11,
    "IMAS_DD_V4": 17,
}
