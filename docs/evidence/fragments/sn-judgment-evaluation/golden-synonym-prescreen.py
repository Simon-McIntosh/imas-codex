"""Write the coordinator's synonym pre-screen and the lead's shortlist from the candidate pool."""

import json
import random

D = "/home/ITER/mcintos/Code/imas-codex/docs/evidence/fragments/sn-judgment-evaluation/"
pool = json.load(open(D + "golden-synonym-candidates.json"))["pairs"]
PLAUSIBLE = {
    4: "total adds nothing to an unresolved neutral internal-state population",
    6: "energy density of a population is its kinetic energy density unless another channel is named",
    19: "neutral and neutral_state may name the same unresolved population",
    21: "neutral and neutral_state may name the same unresolved population",
    40: "bulk adds nothing unless a non-bulk population is defined",
    106: "forward and launched antenna power may be the same quantity",
    140: "neutral_species and neutral_state may name the same population",
    142: "a state-resolved diffusivity may be the general particle diffusivity",
    175: "major_radius may already mean the reference major radius",
    249: "ion and total_ion may both mean all ions",
    269: "momentum and total_momentum may both mean the summed momentum",
    270: "momentum and total_momentum may both mean the summed momentum",
    302: "effective may add nothing to a convection velocity",
    305: "neutral and neutral_state may name the same unresolved population",
    384: "ion_charge_state and ion_state may be one population",
    449: "bulk adds nothing unless a non-bulk population is defined",
    452: "ion and total_ion may both mean all ions",
    552: "total adds nothing to the thermal ion population",
    581: "average_charge_number may already mean the ion charge-state average",
    409: "the normalized toroidal flux coordinate at the LCFS, whichever name it carries",
    632: "aperture and diagnostic_aperture may be the same object",
    633: "aperture and diagnostic_aperture may be the same object",
    557: "the stray breakdown field may only ever be stored as a magnitude",
    558: "the pedestal-top field may only ever be stored as a magnitude",
    656: "neutral_species density and total neutral number density may coincide",
    69: "a conductor element's coordinate and its cross-section's may coincide",
    135: "the divertor's power may be the sum over its targets",
    166: "the divertor's power may be the sum over its targets",
    532: "the divertor's power may be the sum over its targets",
    167: "the unlocated power may already be the wall power",
    534: "the unlocated power may already be the wall power",
    344: "ion charge and the charge-state average charge number may coincide",
    345: "ion charge and the average charge number may coincide",
}
CONTROLS = [0, 16, 27, 99, 150, 268, 286, 569, 643, 256, 141, 15, 1, 5]
rows = []
for i, pair in enumerate(pool):
    shortlisted = i in PLAUSIBLE or i in CONTROLS
    rows.append(
        {
            "pair": i,
            "left": pair["left"],
            "right": pair["right"],
            "verdict": "plausible_synonym" if i in PLAUSIBLE else "distinct",
            "shortlisted": shortlisted,
            "basis": PLAUSIBLE.get(i)
            or "the divergent segment names a different population, component, locus, operator or base",
        }
    )
order = sorted(PLAUSIBLE) + CONTROLS
random.Random(20261013).shuffle(order)
out = {
    "method": (
        "Pairs differing only in species or component were screened by their token swap; "
        "the other 136 pairs were read individually. Plausible synonyms are shortlisted with "
        "14 clearly distinct pairs as controls, in a fixed shuffled order the lead rules on blind."
    ),
    "shortlist_order": order,
    "rows": rows,
}
open(D + "golden-synonym-prescreen.json", "w").write(
    json.dumps(out, ensure_ascii=False, indent=1) + "\n"
)
print(
    len(rows),
    sum(r["shortlisted"] for r in rows),
    sum(r["verdict"] == "plausible_synonym" for r in rows),
)
