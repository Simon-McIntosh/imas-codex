"""Mechanical census of the standard-name catalog: domains, operators, notation."""

import collections
import glob
import json
import os
import re
import sys

import yaml

ROOT = sys.argv[1]
OUT = sys.argv[2]
rows = []
for f in sorted(glob.glob(os.path.join(ROOT, "standard_names", "*.yml"))):
    dom = os.path.basename(f)[:-4]
    for e in yaml.safe_load(open(f)) or []:
        e["domain"] = dom
        rows.append(e)
json.dump(rows, open(OUT, "w"))
print("names", len(rows))
c = collections.Counter(r["domain"] for r in rows)
print("per domain", dict(c.most_common()))
print("keys", collections.Counter(k for r in rows for k in r))
print("units top", collections.Counter(str(r.get("unit")) for r in rows).most_common(40))

ops = [
    "flux_surface_averaged", "volume_averaged", "line_averaged", "line_integrated",
    "surface_integrated", "volume_integrated", "time_derivative", "derivative_of",
    "gradient", "normalized", "magnitude_of", "square_of", "ratio_of", "product_of",
    "tendency", "change_in", "maximum", "minimum", "standard_deviation", "_per_",
    "toroidal_component", "poloidal_component", "radial_component", "parallel_component",
    "real_part", "imaginary_part", "amplitude", "phase", "flux_surface_average",
]
for op in ops:
    sel = [r for r in rows if op in r["name"]]
    print(f"\n== {op}: {len(sel)}  domains={dict(collections.Counter(r['domain'] for r in sel).most_common(6))}")

# flux-surface-average notation styles
print("\n#### flux surface average notation")
fsa = [r for r in rows if "flux_surface_averaged" in r["name"] or "flux_surface_average" in r["name"]]
style = collections.Counter()
ex = collections.defaultdict(list)
for r in fsa:
    d = r.get("documentation") or ""
    s = []
    if re.search(r"\\langle.*\\rangle_?\{?\\?(psi|mathrm\{fs\}|text\{fs\}|rho)", d): s.append("langle_sub")
    elif "\\langle" in d: s.append("langle_plain")
    if "\\oint" in d: s.append("oint")
    if re.search(r"\\partial\s*V|dV|\\mathrm\{d\}V|d\^3", d): s.append("dV")
    if "\\nabla" in d and ("|\\nabla" in d or "\\vert\\nabla" in d or "\\left|\\nabla" in d): s.append("gradpsi_weight")
    if "B_p" in d or "B_\\theta" in d or "B_{\\mathrm{p" in d or "B_{\\theta" in d: s.append("Bp_weight")
    if "\\sqrt{g}" in d or "\\mathcal{J}" in d or "Jacobian" in d or "jacobian" in d: s.append("jacobian")
    if "$$" not in d: s.append("NO_EQUATION")
    k = "+".join(s) or "none"
    style[k] += 1
    ex[k].append(r["name"])
print(len(fsa), style.most_common())
for k, v in ex.items():
    print("  ", k, v[:4])

# docs with no equation at all; missing docs
noeq = [r for r in rows if "$$" not in (r.get("documentation") or "")]
print("\nno display equation:", len(noeq), dict(collections.Counter(r["domain"] for r in noeq).most_common(8)))
print("no documentation:", sum(1 for r in rows if not r.get("documentation")))
# name tokens absent from description
print("\n#### operator token in name but not reflected in description+documentation")
checks = {
    "flux_surface_averaged": r"flux[- ]surface[- ]averag",
    "volume_averaged": r"volume[- ]averag",
    "line_averaged": r"line[- ](of[- ]sight[- ])?averag",
    "line_integrated": r"line[- ](of[- ]sight[- ])?integra|integra\w+ along",
    "time_derivative": r"time derivative|rate of change|temporal derivative",
    "normalized": r"normali[sz]",
    "magnitude_of": r"magnitude|modulus|norm",
    "toroidal_component": r"toroidal",
    "poloidal_component": r"poloidal",
    "parallel_component": r"parallel",
    "radial_component": r"radial",
}
for tok, pat in checks.items():
    bad = [r["name"] for r in rows if tok in r["name"] and not re.search(pat, (r.get("description") or ""), re.I)]
    tot = sum(1 for r in rows if tok in r["name"])
    print(f"{tok}: {len(bad)}/{tot} descriptions lack it; e.g. {bad[:5]}")
