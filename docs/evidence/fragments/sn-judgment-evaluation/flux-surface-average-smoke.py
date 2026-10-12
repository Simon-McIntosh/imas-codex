import json, time
from imas_codex.graph.client import GraphClient
from imas_codex.discovery.base.llm import call_decisions

ids = ['flux_surface_averaged_carbon_density_at_plasma_boundary',
       'flux_surface_averaged_argon_density_at_plasma_boundary',
       'flux_surface_averaged_beryllium_density_at_plasma_boundary',
       'flux_surface_averaged_electron_density_at_plasma_boundary']
with GraphClient() as gc:
    rows = gc.query("MATCH (sn:StandardName) WHERE sn.id IN $ids RETURN sn.id AS id, sn.description AS d, sn.documentation AS doc", ids=ids)
questions = {
  "fsa_form": {"type": "choice",
    "instructions": "The documentation in `entry.documentation` may write an equation defining the flux-surface average. Which form does that equation take?",
    "criteria": {
      "surface": "A surface integral weighted by dS/|grad psi|: the integral of f dS/|grad psi| divided by the integral of dS/|grad psi|.",
      "line_poloidal": "A poloidal line integral weighted by dl/B_p, using the poloidal magnetic field B_p (also written B_pol or B_theta).",
      "line_total_field": "A line integral weighted by dl/B where B is the total magnetic field magnitude, not the poloidal field.",
      "none": "No equation defining the flux-surface average appears."}},
  "fsa_wrong_field": {"type": "noul",
    "instructions": "Does the flux-surface-average equation in `entry.documentation` divide the line element by the total magnetic field B where the poloidal field B_p belongs?",
    "criteria": {"true": "The weight is dl/B or dl over |B|, with no poloidal subscript.",
                 "false": "The weight uses the poloidal field B_p, uses dS/|grad psi|, or there is no such equation."}},
  "boundary_gloss": {"type": "choice",
    "instructions": "How does `entry.description` together with `entry.documentation` describe the plasma boundary?",
    "criteria": {"lcfs_only": "As the last closed flux surface, without calling it the separatrix.",
                 "separatrix_equated": "It equates the plasma boundary with the separatrix.",
                 "not_glossed": "It does not say what the plasma boundary is."}},
}
total = 0.0
for r in rows:
    t = time.time()
    ans, cost = call_decisions("typesafe/jev-1.13", {"entry": {"name": r["id"], "description": r["d"], "documentation": r["doc"]}}, questions, service="standard-names")
    total += cost
    print(r["id"].replace('flux_surface_averaged_','').replace('_density_at_plasma_boundary',''),
          "| fsa_form:", ans["fsa_form"]["choice"], {k: round(v,2) for k,v in ans["fsa_form"]["probabilities"].items()},
          "| wrong_field:", round(ans["fsa_wrong_field"]["noul"],2),
          "| gloss:", ans["boundary_gloss"]["choice"], round(ans["boundary_gloss"].get("confidence",0),2),
          "| %.1fs" % (time.time()-t))
print("total USD", round(total, 6))
