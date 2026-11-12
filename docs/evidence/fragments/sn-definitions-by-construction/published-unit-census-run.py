"""List names whose unit disagrees with the quantity the name states."""

import json
import re
import sys

rows = json.load(open(sys.argv[1]))


def show(pred, label, n=40):
    v = [r for r in rows if pred(r)]
    print("\n##", label, len(v))
    for r in v[:n]:
        d = re.sub(r"\s+", " ", r["description"])[:170]
        print(f"  {r['name']} [{r['unit']}] ({r['domain']}) :: {d}")


def H(h):
    return lambda r: re.search(rf"(^|_){h}($|_at_|_of_|_due_to_|_over_)", r["name"])


show(lambda r: H("current")(r) and r["unit"] != "A", "current not A")
show(lambda r: H("area")(r) and r["unit"] != "m^2", "area not m^2")
show(lambda r: H("coordinate")(r) and r["unit"] not in ("m", "rad", "1"), "coordinate odd")
show(lambda r: H("source_rate")(r) and r["unit"] == "Hz", "source_rate Hz")
show(lambda r: H("particle_flux")(r) and r["unit"] == "s^-1", "particle_flux s^-1")
show(lambda r: H("temperature")(r) and r["unit"] != "eV", "temperature not eV")
show(lambda r: H("energy")(r) and r["unit"] in ("1", "W"), "energy 1/W")
show(lambda r: H("torque")(r) and r["unit"] != "N.m", "torque unit")
show(lambda r: H("torque_density")(r) and r["unit"] != "N.m^-2", "torque density unit", 4)
show(lambda r: H("frequency")(r), "frequency")
show(lambda r: H("current_density")(r) and r["unit"] == "1", "current density 1")
show(lambda r: H("pressure")(r) and r["unit"] == "1", "pressure 1")
show(lambda r: H("time")(r) and r["unit"] == "1", "time 1")
show(lambda r: H("magnetic_field")(r) and r["unit"] == "1", "B 1")
show(lambda r: H("momentum_flux")(r) and r["unit"] == "1", "momflux 1")
show(lambda r: H("emissivity")(r), "emissivity")
GEN = (
    "neon_density tritium_velocity heat_flux torque_density safety_factor plasma_temperature "
    "volumetric_force surface_temperature field_line_length ion_torque radial_conductivity "
    "electron_emissivity power_at_inside_flux_surface straight_field_line_angle minimum_gas_flow "
    "voltage_of_valve energy_of_detector_pixel total_xenon_fraction"
).split()
show(lambda r: r["name"] in GEN, "generic or oddly placed")
