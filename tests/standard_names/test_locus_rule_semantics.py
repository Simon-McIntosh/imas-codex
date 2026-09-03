"""The locus/device guard reads what a DD path MEASURES, not just its spelling.

``workers._is_attachment_consistent`` rejects a hardware-locus name from a
path whose segments share no literal device token with the locus — a rule
that only sees spelling. ``attachment_audit._attachment_consistency`` wraps
that guard and, on exactly that rejection, reads the DD path's own
``dd_documentation`` text as independent evidence: a line-integrated density
constraint spelled ``n_e_line`` shares no token with an
``interferometer_beam`` locus, but its documentation names interferometry
outright, so a semantically exact pairing survives instead of being detached.
A genuinely foreign device — one whose documentation never names the locus's
device either — is still refused.
"""

from imas_codex.standard_names.attachment_audit import _attachment_consistency


def test_documented_interferometry_path_accepts_beam_locus() -> None:
    """The path's own documentation names interferometry, so a hardware
    locus with zero token overlap in the path segments is still accepted."""
    ok, reason = _attachment_consistency(
        "equilibrium/time_slice/constraints/n_e_line/measured",
        "line_integrated_electron_density_of_interferometer_beam",
        dd_documentation=(
            "Interferometry constraint: line integrated electron density "
            "along the diagnostic sight line, used to constrain the "
            "equilibrium reconstruction."
        ),
    )
    assert ok, reason
    assert "overridden" in reason.lower()


def test_undocumented_foreign_device_still_rejected() -> None:
    """A camera path sourcing a strain-gauge-locus name is refused when
    nothing — not the segments, not the documentation — names the gauge."""
    ok, reason = _attachment_consistency(
        "camera_ir/channel/camera/direction/y",
        "y_direction_unit_vector_of_strain_gauge_sensor",
        dd_documentation=(
            "Y component of the unit vector giving the camera's viewing "
            "direction in the machine reference frame."
        ),
    )
    assert not ok
    assert "locus" in reason.lower()


def test_missing_documentation_still_rejected() -> None:
    """No documentation at all is the pre-existing spelling-only behaviour —
    the override never fires without evidence to read."""
    ok, reason = _attachment_consistency(
        "camera_ir/channel/camera/direction/y",
        "y_direction_unit_vector_of_strain_gauge_sensor",
        dd_documentation=None,
    )
    assert not ok
    assert "locus" in reason.lower()
