"""Coverage for the audited absence of numeric unit fabrication."""

from inspect import signature

from imas_codex.standard_names import legacy_authority


def test_numeric_missing_unit_fallback_is_audited_absent() -> None:
    audit = legacy_authority.find_shadow_authorities()
    numeric = next(
        item
        for item in audit.carrier_results
        if item.carrier == "numeric_missing_unit_fallback"
    )

    assert numeric.status is legacy_authority.ShadowAuditStatus.audited
    assert numeric.residual_count == 0
    assert numeric.reason is None


def test_public_audit_boundaries_accept_no_applicability_evidence() -> None:
    assert set(signature(legacy_authority.find_shadow_authorities).parameters) == {
        "manifest",
        "dd_version",
    }
    assert set(
        signature(legacy_authority.assert_zero_audited_shadow_authorities).parameters
    ) == {"manifest", "dd_version"}
