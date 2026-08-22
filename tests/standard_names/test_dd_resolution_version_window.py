"""A reviewed resolution applies across its whole retirement window.

``equilibrium/time_slice/constraints/n_e/reconstructed`` declares a
dimensionless sentinel (``1``) for a quantity whose ``measured`` twin is
``m^-3`` — a self-contradiction ``dd_unit_exceptions.yaml`` flags
``correct_in_graph``. A reviewed :class:`DDResolutionRecord` bridges the
defect for exactly one ``dd_version`` label. The DD version most callers
request comes from :func:`imas_codex.settings.get_dd_version`, whose stale
literal default (``settings.py``) and any DD extraction that pinned a
``dd_version`` before the project's dictionary bump can both diverge from the
label the reviewed record carries, even though the underlying DD declaration
never changed. Gating strictly on label equality made every such divergence a
hard :class:`DDResolutionVersionMismatch` refusal instead of a graceful apply.

A resolution already carries a ``retiring_release`` — the DD version at which
the upstream defect is expected to be fixed (or the ``"none-yet"`` sentinel
while it remains open). That field is the correct authority for how long a
reviewed observed/effective pair applies, so any requested version inside
``[*, retiring_release)`` should resolve on content match alone, regardless of
which exact label the review was recorded under.
"""

from __future__ import annotations

from datetime import UTC, datetime

from imas_codex.graph.models import DDResolutionField, DDResolutionValueKind
from imas_codex.standard_names.dd_resolutions import (
    DDResolutionAmbiguity,
    DDResolutionManifest,
    DDResolutionRecord,
    DDResolutionValue,
    DDResolutionVersionMismatch,
    resolve_dd_field,
    resolve_dd_row,
)

_PATH = "equilibrium/time_slice/constraints/n_e/reconstructed"
_REVIEWED_VERSION = "4.1.1"
_RETIRING_RELEASE = "4.2.0"


def _unit(value: str) -> DDResolutionValue:
    return DDResolutionValue(kind=DDResolutionValueKind.string, value=value)


def _manifest(*, retiring_release: str = _RETIRING_RELEASE) -> DDResolutionManifest:
    return DDResolutionManifest(
        resolutions=(
            DDResolutionRecord(
                id="dd_resolution:" + "0" * 64,
                gap_id=f"dd_gap:{_PATH}:self_contradiction",
                path=_PATH,
                dd_version=_REVIEWED_VERSION,
                field=DDResolutionField.unit,
                observed=_unit("1"),
                effective=_unit("m^-3"),
                reason="The measured twin declares m^-3.",
                recorded_by="synthetic-test-maintainer",
                recorded_at=datetime(2026, 8, 21, tzinfo=UTC),
                upstream_reference="none-yet",
                retiring_release=retiring_release,
                state="active",
            ),
        )
    )


def test_a_requested_version_before_the_reviewed_label_still_applies() -> None:
    manifest = _manifest()

    resolved = resolve_dd_field(
        path=_PATH,
        dd_version="4.1.0",
        field=DDResolutionField.unit,
        raw_value=_unit("1"),
        manifest=manifest,
    )

    assert resolved.applied
    assert resolved.effective.value == "m^-3"
    assert resolved.resolution_id == manifest.resolutions[0].id


def test_a_requested_version_after_the_reviewed_label_still_applies() -> None:
    manifest = _manifest()

    resolved = resolve_dd_field(
        path=_PATH,
        dd_version="4.1.9",
        field=DDResolutionField.unit,
        raw_value=_unit("1"),
        manifest=manifest,
    )

    assert resolved.applied
    assert resolved.effective.value == "m^-3"


def test_the_manifest_drain_read_path_resolves_the_same_window() -> None:
    manifest = _manifest()

    context = resolve_dd_row(
        {"path": _PATH, "unit": "1"},
        dd_version="4.1.0",
        manifest=manifest,
    )

    assert context.unit == "m^-3"
    assert manifest.resolutions[0].id in context.applied_resolution_ids


def test_a_requested_version_at_or_past_retirement_still_refuses() -> None:
    manifest = _manifest()

    try:
        resolve_dd_field(
            path=_PATH,
            dd_version=_RETIRING_RELEASE,
            field=DDResolutionField.unit,
            raw_value=_unit("1"),
            manifest=manifest,
        )
    except DDResolutionVersionMismatch as exc:
        assert "reviewed only" in str(exc)
    else:
        raise AssertionError(
            "a version at/after the retiring release must not silently apply"
        )


def test_an_open_ended_window_never_expires() -> None:
    manifest = _manifest(retiring_release="none-yet")

    resolved = resolve_dd_field(
        path=_PATH,
        dd_version="9.9.9",
        field=DDResolutionField.unit,
        raw_value=_unit("1"),
        manifest=manifest,
    )

    assert resolved.applied
    assert resolved.effective.value == "m^-3"


def test_content_that_never_matches_the_bug_still_refuses() -> None:
    manifest = _manifest()

    try:
        resolve_dd_field(
            path=_PATH,
            dd_version="4.1.0",
            field=DDResolutionField.unit,
            raw_value=_unit("kg"),
            manifest=manifest,
        )
    except DDResolutionVersionMismatch as exc:
        assert "reviewed only" in str(exc)
    else:
        raise AssertionError("content that matches neither side must still refuse")


def test_two_windowed_candidates_with_different_remedies_are_ambiguous() -> None:
    first = _manifest().resolutions[0]
    second = first.model_copy(
        update={
            "id": "dd_resolution:" + "1" * 64,
            "dd_version": "4.1.5",
            "effective": _unit("m^-2"),
        }
    )
    manifest = DDResolutionManifest(resolutions=(first, second))

    try:
        resolve_dd_field(
            path=_PATH,
            dd_version="4.1.0",
            field=DDResolutionField.unit,
            raw_value=_unit("1"),
            manifest=manifest,
        )
    except DDResolutionAmbiguity:
        pass
    else:
        raise AssertionError(
            "two active resolutions disagreeing on the effective value must "
            "not silently pick one"
        )
