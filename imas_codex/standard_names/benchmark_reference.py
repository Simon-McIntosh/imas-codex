"""Reference dataset of known-good standard names for benchmarking.

Each entry maps a DD source path to its expected standard name and
the grammar fields used to compose it.  Every name in this set must
pass a round-trip through ``parse_standard_name`` → ``compose_standard_name``.

The dataset covers a representative range of grammar features:
simple physical bases, subject-qualified quantities, component-qualified
vector quantities, positional variants, compound physical bases, and
geometric quantities.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, TypedDict

import yaml
from imas_standard_names.grammar import (
    Component,
    Object,
    Position,
    Process,
    StandardName,
    Subject,
    compose_standard_name,
)
from imas_standard_names.grammar.model_types import Region


class DocsHoldoutRow(TypedDict):
    """A DD path paired with an immutable catalog documentation snapshot."""

    split_key: str
    dd_path: str
    declared_unit: str | None
    cocos_transformation_type: str | None
    catalog_name: str
    catalog_description: str
    catalog_documentation: str
    catalog_source: str
    catalog_commit: str


class DocsHoldoutAuthority(TypedDict):
    """Unit and COCOS authority bound to one DD path in the graph."""

    declared_unit: str | None
    cocos_transformation_type: str | None


DOCS_HOLDOUT_PATH = (
    Path(__file__).parents[2]
    / "tests"
    / "standard_names"
    / "eval_sets"
    / "docs_holdout.json"
)
CURATED_EXAMPLES_PATH = Path(__file__).with_name("examples_curated.yaml")
DOCS_HOLDOUT_FIELDS = frozenset(DocsHoldoutRow.__required_keys__)
DOCS_HOLDOUT_NULLABLE_FIELDS = frozenset({"declared_unit", "cocos_transformation_type"})

_DOCS_HOLDOUT_AUTHORITY_QUERY = """
UNWIND $dd_paths AS dd_path
OPTIONAL MATCH (node:IMASNode {id: dd_path})
OPTIONAL MATCH (node)-[:HAS_UNIT]->(unit:Unit)
RETURN dd_path,
       count(DISTINCT node) AS node_count,
       collect(DISTINCT unit.id) AS unit_ids,
       collect(DISTINCT node.unit) AS scalar_units,
       collect(DISTINCT node.cocos_transformation_type) AS cocos_types
ORDER BY dd_path
"""

# ---------------------------------------------------------------------------
# Helper: build a reference entry from grammar fields
# ---------------------------------------------------------------------------


def _ref(fields: dict) -> dict:
    """Build a reference entry dict with name string and fields.

    Composes the standard name from the fields at import time so any
    grammar error is caught immediately.
    """
    sn = StandardName(**fields)
    name = compose_standard_name(sn)
    # Store string-valued fields for JSON serialization
    str_fields = {}
    for k, v in fields.items():
        if hasattr(v, "value"):
            str_fields[k] = v.value
        else:
            str_fields[k] = v
    return {"name": name, "fields": str_fields}


# ---------------------------------------------------------------------------
# Reference dataset
# ---------------------------------------------------------------------------

REFERENCE_NAMES: dict[str, dict] = {
    # --- Simple physical bases ---
    "equilibrium/time_slice/profiles_1d/q": _ref({"physical_base": "safety_factor"}),
    "equilibrium/time_slice/global_quantities/magnetic_axis/b_field_tor": _ref(
        {"physical_base": "magnetic_field", "component": Component.TOROIDAL}
    ),
    # shape-param-locus decision: shape parameters are always _of a surface,
    # never bare. profiles_1d → flux_surface.
    "equilibrium/time_slice/profiles_1d/elongation": _ref(
        {"physical_base": "elongation", "geometry": "flux_surface"}
    ),
    "equilibrium/time_slice/boundary/triangularity": _ref(
        {"physical_base": "triangularity", "geometry": "plasma_boundary"}
    ),
    "equilibrium/time_slice/profiles_1d/magnetic_shear": _ref(
        {"physical_base": "magnetic_shear"}
    ),
    "equilibrium/time_slice/global_quantities/beta_pol": _ref(
        {"physical_base": "beta"}
    ),
    # --- Subject-qualified quantities ---
    "core_profiles/profiles_1d/electrons/temperature": _ref(
        {"physical_base": "temperature", "subject": Subject.ELECTRON}
    ),
    "core_profiles/profiles_1d/ion/temperature": _ref(
        {"physical_base": "temperature", "subject": Subject.ION}
    ),
    "core_profiles/profiles_1d/electrons/density": _ref(
        {"physical_base": "density", "subject": Subject.ELECTRON}
    ),
    "core_profiles/profiles_1d/ion/density": _ref(
        {"physical_base": "density", "subject": Subject.ION}
    ),
    "core_profiles/profiles_1d/electrons/pressure": _ref(
        {"physical_base": "pressure", "subject": Subject.ELECTRON}
    ),
    "core_profiles/profiles_1d/ion/pressure": _ref(
        {"physical_base": "pressure", "subject": Subject.ION}
    ),
    # --- Component-qualified vector quantities ---
    "equilibrium/time_slice/profiles_1d/j_tor": _ref(
        {"physical_base": "current_density", "component": Component.TOROIDAL}
    ),
    "equilibrium/time_slice/profiles_1d/j_parallel": _ref(
        {"physical_base": "current_density", "component": Component.PARALLEL}
    ),
    "magnetics/b_field_pol_probe/field/data": _ref(
        {"physical_base": "magnetic_field", "component": Component.POLOIDAL}
    ),
    "magnetics/b_field_tor_probe/field/data": _ref(
        {"physical_base": "magnetic_field", "component": Component.TOROIDAL}
    ),
    # --- Additional magnetics entries ---
    # constraint-instrument-locus decision: name the physics, NOT the instrument
    # (the specific loop/coil is a per-element metadata locus, dropped from the
    # name). poloidal_magnetic_flux = magnetic_flux base + poloidal component.
    "magnetics/flux_loop/flux/data": _ref(
        {"physical_base": "magnetic_flux", "component": Component.POLOIDAL}
    ),
    "magnetics/rogowski_coil/current/data": _ref({"physical_base": "plasma_current"}),
    "magnetics/ip/data": _ref({"physical_base": "plasma_current"}),
    "magnetics/diamagnetic_flux/data": _ref(
        {"physical_base": "magnetic_flux", "component": Component.POLOIDAL}
    ),
    "core_profiles/profiles_1d/rotation_frequency_tor_sonic": _ref(
        {
            "physical_base": "velocity",
            "component": Component.TOROIDAL,
            "subject": Subject.ION,
        }
    ),
    # --- Additional core_profiles entries ---
    "core_profiles/profiles_1d/e_field/parallel": _ref(
        {"physical_base": "electric_field", "component": Component.PARALLEL}
    ),
    "core_profiles/profiles_1d/j_bootstrap": _ref(
        {
            "physical_base": "current_density",
            "component": Component.PARALLEL,
            "process": Process.BOOTSTRAP_CURRENT_DRIVE,
        }
    ),
    "core_profiles/profiles_1d/j_ohmic": _ref(
        {
            "physical_base": "current_density",
            "component": Component.PARALLEL,
            "process": Process.OHMIC_CURRENT_DRIVE,
        }
    ),
    "core_profiles/profiles_1d/ion/velocity/toroidal": _ref(
        {
            "physical_base": "velocity",
            "subject": Subject.ION,
            "component": Component.TOROIDAL,
        }
    ),
    # --- Position-qualified quantities ---
    # A point's R/Z coordinate is <axis>_coordinate_of_<X> (geometry base),
    # NOT major_radius_of_X / vertical_position_of_X.
    "equilibrium/time_slice/global_quantities/magnetic_axis/r": _ref(
        {
            "geometric_base": "radial_coordinate",
            "geometry": "magnetic_axis",
        }
    ),
    "equilibrium/time_slice/profiles_1d/psi": _ref(
        {"physical_base": "magnetic_flux", "component": Component.POLOIDAL}
    ),
    "equilibrium/time_slice/global_quantities/magnetic_axis/z": _ref(
        {
            "geometric_base": "coordinate",
            "coordinate": Component.VERTICAL,
            "geometry": "magnetic_axis",
        }
    ),
    # --- Compound physical bases (generic terms qualified via compounding) ---
    "equilibrium/time_slice/global_quantities/ip": _ref(
        {"physical_base": "plasma_current"}
    ),
    "equilibrium/time_slice/global_quantities/psi_axis": _ref(
        {
            "physical_base": "magnetic_flux",
            "component": Component.POLOIDAL,
            "position": Position.MAGNETIC_AXIS,
        }
    ),
    "equilibrium/time_slice/global_quantities/psi_boundary": _ref(
        {
            "physical_base": "magnetic_flux",
            "component": Component.POLOIDAL,
            "position": Position.PLASMA_BOUNDARY,
        }
    ),
    "equilibrium/time_slice/global_quantities/energy_mhd": _ref(
        {"physical_base": "stored_energy"}
    ),
    "summary/global_quantities/v_loop/value": _ref(
        {"physical_base": "voltage", "object": Object.FLUX_LOOP}
    ),
    "summary/global_quantities/li/value": _ref({"physical_base": "li"}),
    # --- Additional summary entries ---
    "summary/global_quantities/beta_tor/value": _ref({"physical_base": "beta"}),
    "summary/global_quantities/tau_energy/value": _ref(
        {"physical_base": "confinement_time"}
    ),
    # --- Geometric bases (paths that exist in DD 4.1) ---
    "equilibrium/time_slice/boundary/minor_radius": _ref(
        {"physical_base": "minor_radius", "geometry": "plasma_boundary"}
    ),
    # --- core_transport ---
    "core_transport/model/profiles_1d/electrons/energy/flux": _ref(
        {"physical_base": "heat_flux", "subject": Subject.ELECTRON}
    ),
    "core_transport/model/profiles_1d/electrons/particles/flux": _ref(
        {"physical_base": "particle_flux", "subject": Subject.ELECTRON}
    ),
    "core_transport/model/profiles_1d/ion/energy/flux": _ref(
        {"physical_base": "heat_flux", "subject": Subject.ION}
    ),
    "core_transport/model/profiles_1d/ion/particles/flux": _ref(
        {"physical_base": "particle_flux", "subject": Subject.ION}
    ),
    # --- mhd_linear ---
    "mhd_linear/time_slice/toroidal_mode/growthrate": _ref(
        {"physical_base": "growth_rate"}
    ),
    "mhd_linear/time_slice/toroidal_mode/frequency": _ref(
        {"physical_base": "frequency", "subject": Subject.ELECTRON}
    ),
    # --- nbi ---
    "nbi/unit/power_launched/data": _ref(
        {"physical_base": "power", "object": Object.NEUTRAL_BEAM_INJECTOR}
    ),
    "nbi/unit/energy/data": _ref(
        {"physical_base": "energy", "object": Object.NEUTRAL_BEAM_INJECTOR}
    ),
    # --- edge_profiles ---
    "edge_profiles/profiles_1d/electrons/temperature": _ref(
        {
            "physical_base": "temperature",
            "subject": Subject.ELECTRON,
            "region": Region.EDGE_REGION,
        }
    ),
    "edge_profiles/profiles_1d/electrons/density": _ref(
        {
            "physical_base": "density",
            "subject": Subject.ELECTRON,
            "region": Region.EDGE_REGION,
        }
    ),
}
"""Map of DD source_path → {name: str, fields: dict}.

Each entry is a known-good standard name that passes grammar round-trip.
"""


def load_docs_holdout(
    path: Path = DOCS_HOLDOUT_PATH,
) -> list[DocsHoldoutRow]:
    """Load and validate the tracked catalog-documentation holdout."""
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Documentation holdout must be a JSON array")

    rows: list[DocsHoldoutRow] = []
    for index, raw_row in enumerate(payload):
        if not isinstance(raw_row, dict):
            raise ValueError(f"Documentation holdout row {index} must be an object")
        missing = DOCS_HOLDOUT_FIELDS - raw_row.keys()
        if missing:
            fields = ", ".join(sorted(missing))
            raise ValueError(f"Documentation holdout row {index} lacks: {fields}")
        unexpected = raw_row.keys() - DOCS_HOLDOUT_FIELDS
        if unexpected:
            fields = ", ".join(sorted(unexpected))
            raise ValueError(
                f"Documentation holdout row {index} has unexpected fields: {fields}"
            )
        for field in DOCS_HOLDOUT_FIELDS:
            value = raw_row[field]
            if field in DOCS_HOLDOUT_NULLABLE_FIELDS:
                if value is not None and (
                    not isinstance(value, str) or not value.strip()
                ):
                    raise ValueError(
                        f"Documentation holdout row {index} has invalid {field}"
                    )
                continue
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Documentation holdout row {index} has empty {field}")
        if raw_row["split_key"] != raw_row["dd_path"]:
            raise ValueError(
                f"Documentation holdout row {index} split key is not its DD path"
            )
        rows.append(raw_row)
    return rows


def load_docs_holdout_authority(
    dd_paths: Iterable[str],
) -> dict[str, DocsHoldoutAuthority]:
    """Read each holdout path's unit and COCOS authority from the graph.

    Unit authority is fail-closed: each path must bind exactly one IMAS node,
    exactly one ``HAS_UNIT`` target, and the target must agree with the node's
    scalar mirror. A null COCOS transformation property is retained as the
    authoritative declaration that the path is not COCOS-sensitive.
    """
    from imas_codex.graph.client import GraphClient

    paths = tuple(dict.fromkeys(dd_paths))
    if not paths:
        return {}
    if any(not isinstance(path, str) or not path.strip() for path in paths):
        raise ValueError("Documentation holdout DD paths must be non-empty strings")

    with GraphClient() as client:
        graph_rows = list(client.query(_DOCS_HOLDOUT_AUTHORITY_QUERY, dd_paths=paths))

    authority: dict[str, DocsHoldoutAuthority] = {}
    for graph_row in graph_rows:
        row: Mapping[str, Any] = graph_row
        dd_path = row["dd_path"]
        if row["node_count"] != 1:
            raise ValueError(
                f"Documentation holdout path {dd_path!r} binds "
                f"{row['node_count']} IMAS nodes"
            )

        unit_ids = row["unit_ids"] or []
        scalar_units = row["scalar_units"] or []
        if len(unit_ids) != 1:
            raise ValueError(
                f"Documentation holdout path {dd_path!r} has "
                f"{len(unit_ids)} HAS_UNIT authorities"
            )
        if scalar_units != unit_ids:
            raise ValueError(
                f"Documentation holdout path {dd_path!r} has inconsistent "
                "unit scalar and HAS_UNIT authority"
            )

        cocos_types = row["cocos_types"] or []
        if len(cocos_types) > 1:
            raise ValueError(
                f"Documentation holdout path {dd_path!r} has multiple COCOS types"
            )
        authority[dd_path] = {
            "declared_unit": unit_ids[0],
            "cocos_transformation_type": cocos_types[0] if cocos_types else None,
        }

    missing = set(paths) - authority.keys()
    if missing:
        raise ValueError(
            "Documentation holdout graph query omitted paths: "
            + ", ".join(sorted(missing))
        )
    return authority


def curated_example_names(
    path: Path = CURATED_EXAMPLES_PATH,
) -> frozenset[str]:
    """Return the standard-name identities present in the prompt examples."""
    payload: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Curated examples must be grouped by quality tier")

    return frozenset(
        row["id"]
        for tier_rows in payload.values()
        if isinstance(tier_rows, list)
        for row in tier_rows
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    )


def curated_example_split_keys(
    path: Path = CURATED_EXAMPLES_PATH,
) -> frozenset[str]:
    """Return known DD-path split keys represented by prompt examples.

    A single identity may correspond to more than one DD source, so the
    reference map expands every matching example to all known DD paths. The
    holdout also excludes curated identities wholesale, covering DD paths that
    are added to the reference map later.
    """
    curated_names = curated_example_names(path)
    return frozenset(
        dd_path
        for dd_path, reference in REFERENCE_NAMES.items()
        if reference["name"] in curated_names
    )
