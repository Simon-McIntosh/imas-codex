"""Sibling-family consistency: domain splits, description templates, units by head noun."""

import collections
import json
import re
import sys

rows = json.load(open(sys.argv[1]))
SPECIES = [
    "helium_3", "helium_4", "deuterium_tritium", "deuterium", "tritium", "hydrogen", "lithium",
    "beryllium", "boron", "carbon", "nitrogen", "oxygen", "neon", "argon", "krypton", "xenon",
    "iron", "tungsten", "total_hydrogenic",
]
SP_WORDS = [
    "helium-3", "helium-4", "deuterium-tritium", "deuterium–tritium", "deuterium", "tritium",
    "hydrogenic", "hydrogen", "lithium", "beryllium", "boron", "carbon", "nitrogen", "oxygen",
    "neon", "argon", "krypton", "xenon", "iron", "tungsten",
]
sp_re = re.compile(r"(?<![a-z])(" + "|".join(SPECIES) + r")(?![a-z])")
spw_re = re.compile("|".join(SP_WORDS), re.I)


def tmpl(n):
    return sp_re.sub("<X>", n)


fam = collections.defaultdict(list)
for r in rows:
    fam[tmpl(r["name"])].append(r)
sfam = {k: v for k, v in fam.items() if "<X>" in k and len(v) >= 3}
print("species families (>=3 members):", len(sfam), "members", sum(len(v) for v in sfam.values()))

split = {k: v for k, v in sfam.items() if len({r["domain"] for r in v}) > 1}
print("\n### families split across domains:", len(split))
for k, v in sorted(split.items(), key=lambda kv: -len(kv[1])):
    print(f"  {k} n={len(v)} {dict(collections.Counter(r['domain'] for r in v))}")

print("\n### description-template variants inside one family")
nvar = 0
for k, v in sorted(sfam.items(), key=lambda kv: -len(kv[1])):
    t = collections.Counter(re.sub(r"\s+", " ", spw_re.sub("<X>", r["description"])).strip() for r in v)
    u = collections.Counter(r["unit"] for r in v)
    if len(t) > 1 or len(u) > 1:
        nvar += 1
        print(f"\n  {k} n={len(v)} templates={len(t)} units={dict(u)}")
        for d, c in t.most_common(4):
            print(f"     [{c}] {d[:230]}")
print("families with >1 description template or unit:", nvar, "of", len(sfam))

print("\n### unit by trailing/head noun")
heads = [
    "pressure", "temperature", "density", "velocity", "frequency", "energy", "power", "torque",
    "current", "voltage", "angle", "flux", "count", "rate", "time", "diffusivity",
    "diffusion_coefficient", "momentum", "coordinate", "conductivity", "resistivity", "area",
    "fraction", "emissivity", "current_density", "power_density", "torque_density", "energy_flux",
    "particle_flux", "momentum_flux", "source_rate", "magnetic_field", "magnetic_flux", "safety_factor",
]
for h in heads:
    sel = [r for r in rows if re.search(rf"(^|_){h}($|_at_|_of_|_due_to_|_over_)", r["name"])]
    u = collections.Counter(r["unit"] for r in sel)
    if len(u) > 1:
        print(f"  {h} n={len(sel)}: {dict(u.most_common(9))}")
