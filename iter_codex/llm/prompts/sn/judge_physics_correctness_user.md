---
name: sn/judge_physics_correctness_user
description: Dynamic per-candidate user prompt for the physics-correctness judge
used_by: iter_codex.standard_names.physics_judge
task: judging
dynamic: true
schema_needs: []
---

Judge whether the following proposed IMAS standard name is **physically
faithful** to its source quantity. Score every dimension and return the
structured `PhysicsVerdict`.

This call supplies no closed vocabulary, parser projection, grammar error text,
enriched source description, or parent context. Judge only the physical
faithfulness supported by the candidate and source evidence below; do not treat
an unfamiliar legal grammar token as a defect.

## Candidate

- **Proposed standard name:** `{{ name }}`
- **Source DD path:** `{{ path }}`
- **Unit:** {{ unit | default('dimensionless', true) }} *(DD-authoritative)*
- **Documentation:** {{ documentation }}

Assess `base_correct`, `measurement_principle_correct`,
`qualifiers_preserved`, `no_over_qualification`, and `valid`. Set
`faithful` true only if all five hold, and give a one-sentence `reason`
citing the physics for any failure.
