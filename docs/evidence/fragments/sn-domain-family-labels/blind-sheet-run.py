"""Build a blind domain label sheet from published standard names.

Reads the published name records (name, description, unit, domain) and the
grammar library's domain descriptions. Writes a sheet that carries no stored
domain, and a separate key that does. Families are names that differ only in a
species or component token; single names share their pattern with no other.
"""

import collections
import json
import random
import re
import sys

rows = json.load(open(sys.argv[1]))
sys.path.insert(0, sys.argv[2])
from imas_standard_names.grammar.tag_types import PHYSICS_DOMAIN_DESCRIPTIONS  # noqa: E402

out_sheet, out_key = sys.argv[3], sys.argv[4]
rng = random.Random(20261112)

SPECIES = [
    "helium_3", "helium_4", "deuterium_tritium", "deuterium", "tritium", "hydrogen",
    "lithium", "beryllium", "boron", "carbon", "nitrogen", "oxygen", "neon", "argon",
    "krypton", "xenon", "iron", "tungsten", "total_hydrogenic",
]
COMPONENT = ["radial", "toroidal", "poloidal", "vertical", "parallel", "perpendicular", "diamagnetic"]
token = re.compile(r"(?<![a-z])(" + "|".join(SPECIES + COMPONENT) + r")(?![a-z])")

fam = collections.defaultdict(list)
for r in rows:
    fam[token.sub("<X>", r["name"])].append(r)

multi = {k: v for k, v in fam.items() if "<X>" in k and len(v) >= 3}
split = sorted(k for k, v in multi.items() if len({r["domain"] for r in v}) > 1)
whole = sorted(k for k in multi if k not in split)
rng.shuffle(whole)
families = split + whole[: max(0, 40 - len(split))]

single = [v[0] for v in fam.values() if len(v) == 1]
by_domain = collections.defaultdict(list)
for r in single:
    by_domain[r["domain"]].append(r)
named = ["safety_factor", "greenwald_density", "voltage_of_valve"]
drawn = []
total = len(single)
for dom, members in sorted(by_domain.items()):
    k = max(1, round(60 * len(members) / total))
    drawn += rng.sample(members, min(k, len(members)))
drawn = [r for r in drawn if r["name"] not in named]
named_rows = [r for r in rows if r["name"] in named]

items, key = [], []
for k in families:
    members = sorted(multi[k], key=lambda r: r["name"])
    shown = rng.sample(members, min(3, len(members)))
    items.append({
        "kind": "family",
        "pattern": k,
        "member_count": len(members),
        "examples": [{"name": r["name"], "description": r["description"], "unit": r["unit"]} for r in shown],
    })
    key.append({"pattern": k, "members": {r["name"]: r["domain"] for r in members}})
for r in drawn + named_rows:
    items.append({"kind": "single", "name": r["name"], "description": r["description"], "unit": r["unit"]})
    key.append({"name": r["name"], "domain": r["domain"], "named_in_plan": r["name"] in named})

order = list(range(len(items)))
rng.shuffle(order)
items = [dict(items[i], id=f"item-{n + 1:03d}") for n, i in enumerate(order)]
key = [dict(key[i], id=f"item-{n + 1:03d}") for n, i in enumerate(order)]

json.dump({"domains": PHYSICS_DOMAIN_DESCRIPTIONS, "items": items}, open(out_sheet, "w"), indent=1)
json.dump(key, open(out_key, "w"), indent=1)
print("families", len(families), "of which split", len(split), "| singles drawn", len(drawn), "| named", len(named_rows))
print("multi-member families available", len(multi), "| single-pattern names", total)
