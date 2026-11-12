"""Registry coverage and conformance of the published catalog."""

import collections
import json
import sys

from imas_standard_names.grammar.terms import select_conventions

from imas_codex.standard_names.docs_gates import house_style_findings

rows = json.load(open(sys.argv[1]))
cov = collections.Counter()
viol = collections.Counter()
concept_use = collections.Counter()
nviol = 0
parse_fail = []
ex = collections.defaultdict(list)
for r in rows:
    try:
        sel = select_conventions(r["name"])
    except Exception as e:  # noqa: BLE001
        parse_fail.append((r["name"], type(e).__name__))
        continue
    kinds = []
    if sel.concepts:
        kinds.append("concept")
    if sel.notation:
        kinds.append("notation")
    if sel.templates:
        kinds.append("template")
    cov["+".join(kinds) or "NONE"] += 1
    for c in sel.concepts:
        concept_use[c.token] += 1
    f = house_style_findings(sel, description=r["description"], documentation=r["documentation"])
    if f:
        nviol += 1
    for x in f:
        k = f"{x.finding_class.value}|{x.registry_entry}"
        viol[k] += 1
        if len(ex[k]) < 3:
            ex[k].append(r["name"])
print("names", len(rows), "parse failures", len(parse_fail), parse_fail[:8])
print("registry coverage:", cov.most_common())
print("concepts selected:", concept_use.most_common(40))
print("names with >=1 finding:", nviol)
for k, v in viol.most_common(60):
    print(f"  {v:4d} {k}  e.g. {ex[k]}")
sel = select_conventions("flux_surface_averaged_parallel_current_density")
print("\nFSA selection:", sel.model_dump_json(indent=1)[:2500])
for n in ["volume_averaged_argon_density", "line_averaged_electron_density", "neon_density", "ion_torque_density", "toroidal_ion_velocity"]:
    s = select_conventions(n)
    print(n, "->", [c.token for c in s.concepts], [t.base for t in s.templates], [x.token for x in s.notation])
