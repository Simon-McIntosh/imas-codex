# Deterministic parent provenance seed

## Outcome

**PASS — the ordinary parent-source reconcile ran once and closed the exact
transient childful cohort.** The live preflight recovered the same nine
childful bare parents recorded by the preceding census, admitted all nine
through the current structural classifier, and passed that exact classification
to `reconcile_orphan_parent_sources`. The operation seeded nine deterministic
`derived:<parent>` sources and left zero eligible or rejected rows afterward.

| Required measure | Before | After | Result |
|---|---:|---:|---|
| Childful bare structural parents | **9** | **0** | PASS |
| Classifier-eligible parent sources | **9** | **0** | PASS |
| Parent sources seeded | — | **9** | Equals the complete childful cohort |
| Unseeded childful parents | — | **0** | No refusal reasons required |
| Childless bare names | **20** | **20** | Untouched |
| `StandardNameChange` | **7,780** | **7,780** | Delta **0** from the 7,780 baseline |
| All `PRODUCED_NAME` relationships | **5,770** | **5,779** | Expected **+9** structural-provenance links |
| `LLMCost` rows | **27,631** | **27,631** | **0 provider calls** |
| Recorded LLM spend | **$1,366.843569** | **$1,366.843569** | **USD 0.00** spent |

The global relationship increase is exactly the intended result, not restored
DD authority: each new relationship starts at a `derived:` structural source,
ends at its parent, and has no `FROM_DD_PATH` edge.

## Seeded cohort

The nine census rows were seeded one for one:

| Parent | Deterministic source |
|---|---|
| `area_of_langmuir_probe` | `derived:area_of_langmuir_probe` |
| `electrostatic_potential_imaginary_part` | `derived:electrostatic_potential_imaginary_part` |
| `momentum_source` | `derived:momentum_source` |
| `neutral_species_energy_convection_velocity` | `derived:neutral_species_energy_convection_velocity` |
| `neutral_state_particle_diffusivity` | `derived:neutral_state_particle_diffusivity` |
| `normalized_perturbed_current_density` | `derived:normalized_perturbed_current_density` |
| `outer_squareness_of_flux_surface` | `derived:outer_squareness_of_flux_surface` |
| `parallel_normalized_perturbed_current_density` | `derived:parallel_normalized_perturbed_current_density` |
| `volume_averaged_runaway_electron_current_density` | `derived:volume_averaged_runaway_electron_current_density` |

For every row, postflight found source type `derived`, source status `composed`,
`produced_sn_id` equal to the parent identity, exactly one `PRODUCED_NAME`
relationship to that parent, and zero DD realization paths. The ordinary
classifier returned zero repairable and zero rejected-derived candidates after
the operation, so `unseeded=0` is a measured result rather than an inference
from the write count.

## Removed DD bindings stayed removed

The census attributes the nine parents to nine distinct causal DD sources; two
parents share the same gyrokinetic source, while the Langmuir-probe parent has
two source rows. Those nine source nodes carried **10** current
`PRODUCED_NAME` bindings before the parent reconcile and the same **10**
afterward. Their normalized source/status/scalar/target snapshot stayed
byte-equivalent:

```text
before sha256 6c6f676e0b240971e1f299b1658ceb6d0866a77bf52ae37e0650f65277671f8c
after  sha256 6c6f676e0b240971e1f299b1658ceb6d0866a77bf52ae37e0650f65277671f8c
```

The wider signed reconciliation authority was checked as a second boundary.
All **23** source rows retained exactly **28** bindings before and after, with
the same complete normalized snapshot:

```text
before sha256 eb44a8e6fc0d795764969008405752925e22b95ed59a63a3b278a5d6751fd8ee
after  sha256 eb44a8e6fc0d795764969008405752925e22b95ed59a63a3b278a5d6751fd8ee
```

Thus none of the broad DD interpretations removed by the signed source-target
reconcile was restored. The new provenance is structural only.

## Childless partition stayed untouched

The exact 20-name childless partition remained bare and childless. Its
normalized identity/lifecycle/unit/children/producer snapshot was identical on
both sides of the reconcile:

```text
before sha256 3462c315807e5a06f57e23f864deb71dcec13ee36a86dd2e419781695951342b
after  sha256 3462c315807e5a06f57e23f864deb71dcec13ee36a86dd2e419781695951342b
```

This preserves the census's named reasons and governed recovery conditions for
all 20 genuine gaps. The parent-source reconcile did not reinterpret a
childless provenance gap as a structural parent.

## Execution and evidence record

- Live plan authority: `imas-codex:sn-graph-wide-integrity`, version **240**.
- Source checkout at operation time: `40953841`.
- Baseline census:
  `docs/evidence/sn-graph-wide-integrity/bare-structural-census.md` and its
  machine record at
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T225117730640-barecensus/logs/bare-structural-census.json`.
- Signed reconciliation authority used only to identify and verify the prior
  source cohort:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T204309912344-dualapply/dual-bound-source-target-authority.json`.
- Machine-readable preflight, operation result, and postflight:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T001634452777-parentseed/logs/parent-provenance-seed.json`.
- Captured stderr and exit status:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T001634452777-parentseed/logs/parent-provenance-seed.stderr.log`
  and `parent-provenance-seed.exit` in the same directory.

One earlier harness attempt stopped before constructing `GraphClient`: its
distinct-source preflight expected ten causal sources, while the census rows
collapse to nine because two parents share one source. It invoked no reconcile
and made no graph contact. That failed preflight is retained as
`parent-provenance-seed.preflight-failed.stderr.log` with exit status `1` in
the same log directory. The corrected production invocation retained every
semantic assertion, connected once, and called
`reconcile_orphan_parent_sources` exactly once.

No provider-backed pool, review draw, raw Cypher mutation, plan-state edit, or
DD-source reattachment was performed.
