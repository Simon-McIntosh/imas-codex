"""Pins the shipped IDS-to-physics-domain map.

``imas_codex/definitions/physics/ids_domains.json`` is read at DD build time:
each IDS node and each IMASNode takes its default ``physics_domain`` from it, so
a silent drift in one row reclassifies every path beneath that IDS. These tests
read the shipped file and pin the entries set by review, the domain vocabulary,
and the summary tally.
"""

import json
import os
from collections import Counter
from pathlib import Path

import pytest

from imas_codex.core.data_model import PhysicsDomain
from imas_codex.definitions.physics import IDS_DOMAINS_FILE

# IDS names whose domain was set by review, and the domain each must carry.
# Kept as a literal table so a change to the map is visible as a change here.
RULED_DOMAINS = {
    "amns_data": "computational_workflow",
    "calorimetry": "mechanical_measurement_diagnostics",
    "charge_exchange": "spectroscopy",
    "core_instant_changes": "core_plasma_physics",
    "core_profiles": "core_plasma_physics",
    "distribution_sources": "fast_particles",
    "distributions": "fast_particles",
    "ece": "radiation_measurement_diagnostics",
    "gas_injection": "fueling",
    "gyrokinetics": "gyrokinetics",
    "gyrokinetics_local": "gyrokinetics",
    "mse": "spectroscopy",
    "neutron_diagnostic": "neutronics",
    "pellets": "fueling",
    "plasma_initiation": "plasma_initiation",
    "plasma_profiles": "core_plasma_physics",
    "radiation": "core_plasma_physics",
    "runaway_electrons": "runaway_electrons",
    "spectrometer_mass": "spectroscopy",
    "spectrometer_uv": "spectroscopy",
    "spectrometer_visible": "spectroscopy",
    "spectrometer_x_ray_crystal": "spectroscopy",
    "spi": "fueling",
    "thomson_scattering": "particle_measurement_diagnostics",
    "waves": "waves",
}


def _map_path() -> Path:
    """The map under test, overridable to point at a scratch copy."""
    override = os.environ.get("IDS_DOMAINS_MAP_PATH")
    return Path(override) if override else IDS_DOMAINS_FILE


def _load_map() -> dict:
    return json.loads(_map_path().read_text(encoding="utf-8"))


def test_every_mapped_domain_is_a_physics_domain_member():
    """No row may name a domain outside the PhysicsDomain vocabulary."""
    mappings = _load_map()["ids_domain_mappings"]
    unknown = {
        ids_name: domain
        for ids_name, domain in mappings.items()
        if domain not in {member.value for member in PhysicsDomain}
    }
    assert unknown == {}


def test_tally_equals_mapping_counts():
    """The summary tally is exactly the count of the mappings it summarises."""
    data = _load_map()
    mappings = data["ids_domain_mappings"]
    declared = data["physics_domains"]
    assert declared == dict(Counter(mappings.values()))
    assert sum(declared.values()) == len(mappings)


@pytest.mark.parametrize(
    ("ids_name", "domain"), sorted(RULED_DOMAINS.items()), ids=sorted(RULED_DOMAINS)
)
def test_ruled_ids_carry_ruled_domain(ids_name: str, domain: str):
    """Each reviewed IDS carries the domain review set for it."""
    mappings = _load_map()["ids_domain_mappings"]
    assert ids_name in mappings, f"{ids_name} is absent from the map"
    assert mappings[ids_name] == domain
