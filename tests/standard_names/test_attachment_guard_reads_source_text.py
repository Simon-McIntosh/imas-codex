"""The source-binding guard reads the sources' own stated operations.

``_is_attachment_consistent`` judged a pairing from path spelling, unit and the
name's own wording, so two sources whose stated descriptions define different
operations were admitted to one name and the definition stage then had to write
a single definition from both. Thirteen equilibrium members are defective that
way. The guard now takes the candidate's and the already-bound sources' text and
rejects a source whose stated operation neither the name nor a companion source
carries.

The rejection fixtures use the recorded source descriptions of those members;
the sound fixtures are members of the same group the adjudication judged sound,
so a rule that renamed what was right would fail here too.
"""

from __future__ import annotations

from iter_codex.standard_names import attachment_audit
from iter_codex.standard_names.worker_tasks.worker_extraction_and_enrichment import (
    _is_attachment_consistent,
)

# --- cause members: the recorded source text of each conflicting binding -----

#: ``plasma_resistance`` is defined by the equilibrium source as a volume
#: integral; the summary source states a ratio of two different quantities.
_RESISTANCE_INTEGRAL_PATH = "equilibrium/time_slice/global_quantities/plasma_resistance"
_RESISTANCE_INTEGRAL_TEXT = (
    "Plasma electrical resistance (R), defined as the volume integral of E.J "
    "divided by Ip squared, representing the effective resistive impedance of "
    "the plasma column."
)
_RESISTANCE_RATIO_PATH = "summary/global_quantities/resistance/value"
_RESISTANCE_RATIO_TEXT = (
    "Plasma electrical resistance (R), the ratio of loop voltage to plasma "
    "current, characterizing resistive dissipation in the plasma."
)

#: ``toroidal_magnetic_field`` is a point component; the probe source states a
#: value averaged over the sensor volume.
_PROBE_FIELD_PATH = "magnetics/b_field_phi_probe/field"
_PROBE_FIELD_TEXT = (
    "Magnetic field component measured by a toroidal field (B_phi) probe, "
    "averaged over the sensor volume defined by area and length."
)

#: ``vertical_coordinate_of_measurement_position`` is a point coordinate; the
#: position_average leaf states an average over several collectors.
_AVERAGE_POSITION_PATH = "langmuir_probes/reciprocating/plunge/position_average/z"
_AVERAGE_POSITION_TEXT = (
    "Vertical (Z) coordinate of position average of the measurements derived "
    "from multiple collectors."
)

# --- sound fixtures: members of the same group judged sound ------------------

#: The arithmetic midpoint of the boundary's extents is called an average in
#: its source, and it is one quantity, not an aggregation over samples. The
#: sound member ``vertical_coordinate_of_geometric_axis`` must still bind.
_GEOMETRIC_AXIS_Z_PATH = "summary/boundary/geometric_axis_z/value"
_GEOMETRIC_AXIS_Z_TEXT = (
    "Vertical position (Z) of the plasma geometric axis, defined as the "
    "average of the maximum and minimum Z extents of the plasma boundary."
)

#: A plain point quantity: neither text states an operation.
_PLAIN_DENSITY_PATH = "core_profiles/profiles_1d/electrons/density"
_PLAIN_DENSITY_TEXT = "Electron density of the plasma."


def test_a_ratio_source_is_rejected_from_an_integral_quantity() -> None:
    ok, reason = _is_attachment_consistent(
        _RESISTANCE_RATIO_PATH,
        "plasma_resistance",
        dd_text=_RESISTANCE_RATIO_TEXT,
        existing_texts=(_RESISTANCE_INTEGRAL_TEXT,),
    )

    assert not ok
    assert reason.startswith("stated operation mismatch")
    assert "ratio of loop voltage to plasma current" in reason
    assert "plasma_resistance" in reason


def test_a_volume_averaged_source_is_rejected_from_a_point_quantity() -> None:
    ok, reason = _is_attachment_consistent(
        _PROBE_FIELD_PATH,
        "toroidal_magnetic_field",
        dd_text=_PROBE_FIELD_TEXT,
        existing_texts=(
            "Toroidal component, one scalar value per element in the grid subset.",
        ),
    )

    assert not ok
    assert reason.startswith("stated operation mismatch")
    assert "averaged over the sensor volume" in reason


def test_averaged_position_is_rejected_from_a_point_coordinate() -> None:
    ok, reason = _is_attachment_consistent(
        _AVERAGE_POSITION_PATH,
        "vertical_coordinate_of_measurement_position",
        dd_text=_AVERAGE_POSITION_TEXT,
        existing_texts=("Vertical (Z) coordinate of position.",),
    )

    assert not ok
    assert reason.startswith("stated operation mismatch")


def test_the_same_source_is_admitted_when_the_name_states_the_operation() -> None:
    ok, reason = _is_attachment_consistent(
        _AVERAGE_POSITION_PATH,
        "average_vertical_coordinate_of_measurement_position",
        dd_text=_AVERAGE_POSITION_TEXT,
        existing_texts=("Vertical (Z) coordinate of position.",),
    )

    assert ok, reason


def test_the_same_source_is_admitted_when_a_companion_states_it() -> None:
    ok, reason = _is_attachment_consistent(
        _AVERAGE_POSITION_PATH,
        "vertical_coordinate_of_measurement_position",
        dd_text=_AVERAGE_POSITION_TEXT,
        existing_texts=(_AVERAGE_POSITION_TEXT,),
    )

    assert ok, reason


def test_the_midpoint_of_two_extents_is_not_an_aggregation() -> None:
    """Sound member: ``average`` in an arithmetic midpoint states no operation."""
    ok, reason = _is_attachment_consistent(
        _GEOMETRIC_AXIS_Z_PATH,
        "vertical_coordinate_of_geometric_axis",
        dd_text=_GEOMETRIC_AXIS_Z_TEXT,
        existing_texts=("Vertical (Z) coordinate of geometric axis.",),
    )

    assert ok, reason


def test_a_plain_quantity_is_unchanged_by_the_text_rule() -> None:
    ok, reason = _is_attachment_consistent(
        _PLAIN_DENSITY_PATH,
        "electron_density",
        dd_text=_PLAIN_DENSITY_TEXT,
        existing_texts=(_PLAIN_DENSITY_TEXT,),
    )

    assert ok, reason


# --- wiring: the pairing guard reads the graph's text -----------------------


def _guard_rows() -> list[dict]:
    return [
        {
            "source_id": "src-integral",
            "source_type": "dd",
            "dd_path": _RESISTANCE_INTEGRAL_PATH,
            "dd_unit": "ohm",
            "sn_unit": "ohm",
            "already_bound": False,
            "existing_dd_paths": [],
            "existing_dd_texts": [],
            "name_stage": "drafted",
            "dd_documentation": "",
            "dd_description": _RESISTANCE_INTEGRAL_TEXT,
        },
        {
            "source_id": "src-ratio",
            "source_type": "dd",
            "dd_path": _RESISTANCE_RATIO_PATH,
            "dd_unit": "ohm",
            "sn_unit": "ohm",
            "already_bound": False,
            "existing_dd_paths": [],
            "existing_dd_texts": [],
            "name_stage": "drafted",
            "dd_documentation": "",
            "dd_description": _RESISTANCE_RATIO_TEXT,
        },
    ]


class _RowsClient:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        self.queried: tuple[str, dict] | None = None

    def query(self, cypher: str, **params: object) -> list[dict]:
        self.queried = (cypher, params)
        return self.rows


def test_the_pairing_guard_reads_source_text_and_records_the_rejection() -> None:
    gc = _RowsClient(_guard_rows())

    result = attachment_audit.guard_source_pairings(
        gc, "plasma_resistance", ["src-integral", "src-ratio"]
    )

    assert result.accepted_source_ids == ("src-integral",)
    assert [verdict.rule for verdict in result.rejected] == [
        "stated operation mismatch"
    ]
    assert result.rejected[0].dd_path == _RESISTANCE_RATIO_PATH
    # The query hands the guard the candidate's text and the bound sources'
    # text, which the spelling-only rule could not read.
    assert gc.queried is not None
    assert "dd.description AS dd_description" in gc.queried[0]
    assert "existing_dd_texts" in gc.queried[0]
