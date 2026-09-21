"""A settled source exclusion is waived only by an exact declared path.

A manifest cut owes every source a name, so a source the pipeline correctly
refuses forever would otherwise render a cut permanently ungenerable. The
authority settles such a source by enumerating its path in
``imas_codex/standard_names/waived_sources.yaml``; the export leg reads that
enumeration and marks the row ``waived``.

Everything here is about the boundary of that enumeration. Membership is the
whole test: a source is waived because its full path appears in the list, never
because it ends in a segment an entry shares, never because it resembles one, and
never because a refusal reason or a category suggests it. The declaration's own
``not_waived`` list is load-bearing rather than commentary -- a path recorded
there must not be waived, and a path in both lists refuses the load rather than
letting the reader choose. A declaration that is absent or internally
inconsistent grants nothing and refuses loudly, because a silent empty set is
indistinguishable from a declaration that waives nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytest.importorskip("imas_standard_names")

from imas_codex.standard_names.export import (  # noqa: E402
    _WAIVED_SOURCES_DECLARATION,
    WaivedSourceDeclarationError,
    _load_waived_source_paths,
    _source_disposition,
)

# The eleven paths the declaration enumerates, restated so the test asserts the
# declaration rather than reading its own membership back out of it.
_DECLARED_WAIVED = (
    "camera_x_rays/detector_humidity/time",
    "camera_x_rays/detector_temperature/time",
    "hard_x_rays/emissivity_profile_1d/time",
    "camera_x_rays/frame/time",
    "core_profiles/profiles_1d/time",
    "equilibrium/time_slice/time",
    "equilibrium/time_slice/constraints/b_field_pol_probe/weight",
    "equilibrium/time_slice/constraints/flux_loop/weight",
    "equilibrium/time_slice/constraints/faraday_angle/weight",
    "equilibrium/time_slice/constraints/n_e_line/weight",
    "equilibrium/time_slice/convergence/iterations_n",
)

# Recorded as deliberately not waived: a measured event time with a trailing
# segment three declared paths also carry.
_TIME_SEGMENT_NEAR_MISS = "summary/disruption/time/value"

# Resembles a declared path closely enough to tempt a prefix rule, and is not
# one.
_UNDECLARED_NEIGHBOUR = "equilibrium/time_slice/constraints/flux_loop/weight_scale"


def _document() -> dict:
    """A minimal declaration whose shape satisfies every load-time check."""
    return {
        "schema": 1,
        "criteria": {
            "coordinate_axis": "The node is an axis rather than a quantity.",
        },
        "waived": [
            {
                "path": "equilibrium/time_slice/time",
                "criterion": "coordinate_axis",
                "dd_evidence": "FLT_0D, units s, Coordinates 1...N",
            }
        ],
        "not_waived": [
            {
                "path": _TIME_SEGMENT_NEAR_MISS,
                "reason": "A measured event time wants a standard name.",
            }
        ],
    }


def _write_declaration(tmp_path: Path, document: dict) -> Path:
    path = tmp_path / "waived_sources.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def _classify(path: str, waived_paths: frozenset[str]) -> tuple[str, str]:
    """Run one source through the export leg's disposition decision."""
    return _source_disposition(
        source_path=path,
        waived_paths=waived_paths,
        non_nameable_reason="dd_node_category_ineligible: fit_artifact",
        standard_name_id=None,
        exported_ids=set(),
        exclusion_reason=None,
    )


def test_every_declared_path_is_waived() -> None:
    """All eleven enumerated paths produce a waived row."""
    waived_paths = _load_waived_source_paths(_WAIVED_SOURCES_DECLARATION)

    assert isinstance(waived_paths, frozenset)
    assert len(_DECLARED_WAIVED) == 11
    assert set(_DECLARED_WAIVED) <= waived_paths
    for path in _DECLARED_WAIVED:
        assert _classify(path, waived_paths) == ("waived", ""), path


def test_a_shared_segment_does_not_waive_a_measured_event_time() -> None:
    """A refusal reason and a shared path segment grant no waiver."""
    waived_paths = _load_waived_source_paths(_WAIVED_SOURCES_DECLARATION)

    # The source shares the segment ``time`` with declared entries, so a rule
    # keyed on that segment would waive it and membership must not.
    shared_segment = [path for path in waived_paths if "time" in path.split("/")]
    assert len(shared_segment) >= 3
    assert _TIME_SEGMENT_NEAR_MISS not in waived_paths
    assert _classify(_TIME_SEGMENT_NEAR_MISS, waived_paths) == (
        "documented_non_nameable",
        "dd_node_category_ineligible: fit_artifact",
    )


def test_a_path_absent_from_the_declaration_is_not_waived() -> None:
    """Resemblance to a declared path is not membership of the list."""
    waived_paths = _load_waived_source_paths(_WAIVED_SOURCES_DECLARATION)

    assert _UNDECLARED_NEIGHBOUR not in waived_paths
    assert _classify(_UNDECLARED_NEIGHBOUR, waived_paths)[0] != "waived"


def test_a_missing_declaration_refuses_and_names_the_file(tmp_path: Path) -> None:
    """An absent declaration grants nothing and refuses by name."""
    absent = tmp_path / "absent-waived-sources.yaml"

    with pytest.raises(WaivedSourceDeclarationError) as excinfo:
        _load_waived_source_paths(absent)

    assert str(absent) in str(excinfo.value)


def test_an_entry_missing_its_evidence_is_refused(tmp_path: Path) -> None:
    """A waiver that does not state its data-dictionary evidence is refused."""
    document = _document()
    del document["waived"][0]["dd_evidence"]
    path = _write_declaration(tmp_path, document)

    with pytest.raises(WaivedSourceDeclarationError) as excinfo:
        _load_waived_source_paths(path)

    assert "dd_evidence" in str(excinfo.value)


def test_an_entry_citing_an_undefined_criterion_is_refused(tmp_path: Path) -> None:
    """A waiver that does not cite a defined criterion is refused."""
    document = _document()
    document["waived"][0]["criterion"] = "no_such_criterion"
    path = _write_declaration(tmp_path, document)

    with pytest.raises(WaivedSourceDeclarationError) as excinfo:
        _load_waived_source_paths(path)

    assert "no_such_criterion" in str(excinfo.value)


def test_a_path_in_both_lists_is_refused(tmp_path: Path) -> None:
    """A contested path refuses the load rather than being resolved silently."""
    document = _document()
    waivpath = document["waived"][0]["path"]
    document["not_waived"].append({"path": waivpath, "reason": "contested"})
    path = _write_declaration(tmp_path, document)

    with pytest.raises(WaivedSourceDeclarationError) as excinfo:
        _load_waived_source_paths(path)

    assert waivpath in str(excinfo.value)
