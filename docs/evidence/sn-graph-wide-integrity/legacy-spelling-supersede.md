# Legacy spelling supersede apply

## Outcome

One builder-emitted, signed authority enumerated exactly six redundant
predecessor-to-canonical pairs. The production preview and apply partitioned the
same signed cohort as **5 admitted + 1 refused = 6 rows**. Five predecessors
were transitioned to `name_stage=superseded` and `status=superseded`; the
structural umbrella `area_of_flux_surface` refused fail-closed with the verbatim
reason **`target has a live structural child`**. No child or lineage edge was
removed.

The authority file SHA-256 is
`8a845b6c230e30a1b73dac6333ab7c5079f73eba7127e7f9a3b7b82e3b5f6148`,
its signed-payload SHA-256 is
`81ffb34ca4616215b875808645f2a56ea503095a4289a5b7b627863af7f1e438`,
and the exact live manifest SHA-256 is
`e53316d77d3a72b221461829789b4fcfe607a3ae474986581b087ffb92916aac`.

## Exact identities and dispositions

Every canonical target was read as `accepted` and `valid` before the authority
was emitted. The spelling on the left is the predecessor; the spelling on the
right is the accepted identity that retains the intended meaning.

| Predecessor | Accepted, valid target beforehand | Predecessor before → after | Live children before → after | Disposition |
|---|---|---|---:|---|
| `minimum_of_safety_factor` | `minimum_safety_factor` | `reviewed / null` → `superseded / superseded` | 0 → 0 | Applied. The canonical name denotes the signed safety-factor value on the surface where its magnitude is smallest; the rejected spelling's description incorrectly called the value a location. |
| `line_integrated_electron_density` | `line_integrated_electron_number_density` | `drafted / null` → `superseded / superseded` | 0 → 0 | Applied. `number_density` states the measured carrier explicitly; the shorter spelling was redundant. |
| `poloidal_straight_field_line_angle` | `straight_field_line_angle` | `drafted / null` → `superseded / superseded` | 0 → 0 | Applied. The straight-field-line angle is poloidal by construction, so the extra axis token did not add meaning. |
| `parallel_effective_neutral_internal_state_velocity_due_to_diamagnetic_drift` | `parallel_neutral_state_velocity_due_to_diamagnetic_drift` | `accepted / null` → `superseded / superseded` | 0 → 0 | Applied. The DD path is a physical drift velocity, not an effective coefficient; the canonical spelling retains the parallel neutral-state and diamagnetic-drift semantics. |
| `toroidal_neutral_state_momentum_diffusivity` | `toroidal_neutral_internal_state_momentum_diffusion_coefficient` | `accepted / null` → `superseded / superseded` | 0 → 0 | Applied. The target was derived in the applying invocation from the prior dual-bound closure described below. |
| `area_of_flux_surface` | `poloidal_plane_cross_sectional_area_of_flux_surface` | `accepted / null` → unchanged | 3 → 3 | **Refused:** `target has a live structural child`. The operator did not strip a child to force the chained supersede. |

The toroidal survivor was not supplied as a fixed spelling. The invocation read
the predecessor's historical source suffix
`model/ggd/neutral/state/momentum/d/phi`, found the single current source whose
complete target closure contained both a terminal predecessor and exactly one
live accepted-valid scalar-selected target, and derived
`toroidal_neutral_internal_state_momentum_diffusion_coefficient` from
`dd:plasma_transport/model/ggd/neutral/state/momentum/d/phi`. The signed closure
contained that survivor plus the already-superseded
`toroidal_momentum_diffusivity`.

## Receipt arithmetic and replay

The applying invocation read the live ledger baseline immediately before the
transaction and re-read it immediately after:

| Measure | Before | After apply | After replay | Delta |
|---|---:|---:|---:|---:|
| `StandardNameChange` | 7,836 | 7,841 | 7,841 | **+5** |
| `LLMCost` | 27,631 | 27,631 | 27,631 | 0 |
| Mutated authority rows | — | **5** | 0 | 5 |
| Receipt rows | 0 for this manifest | **5** | 5 retained | **+5** |
| Persistent writes reported by the operator | — | **10** (5 mutations + 5 receipts) | **0** | 10 |

Thus `receipt_rows = changed = StandardNameChange delta = 5`. The immediate
second apply returned `outcome=already_applied`, `changed=0`, and
`persistent_writes=0`; both live counters were byte-for-byte unchanged across
the replay.

Each changed predecessor owns one new `HAS_INTERNAL_CHANGE` receipt under
operation `supersede_legacy_spelling`. The receipt `row_id` preserves the full
before/after identity pair. The five receipt ids are present in the machine
result and are all bound to the same manifest digest.

## Structural and lineage preservation

All direct incoming and outgoing `REFINED_FROM` element ids were captured before
the apply and compared after it. Every captured chain was unchanged, including:

- `line_integrated_electron_density_of_interferometer_beam` retaining its
  existing `REFINED_FROM` edge to `line_integrated_electron_density`;
- `parallel_effective_neutral_internal_state_velocity_due_to_diamagnetic_drift`
  retaining its existing edge to
  `parallel_neutral_internal_state_velocity_due_to_diamagnetic_drift`;
- `area_of_flux_surface` retaining the single existing `REFINED_FROM` edge to
  `cross_section_of_flux_surface`, with the same Neo4j relationship element id
  before and after.

`area_of_flux_surface` retained all three live structural children:
`surface_area_of_flux_surface`,
`derivative_with_respect_to_toroidal_flux_coordinate_of_area_of_flux_surface`,
and
`derivative_with_respect_to_normalized_poloidal_flux_coordinate_of_area_of_flux_surface`.
The refused row remained accepted and valid. This is an intentional incomplete
supersede cohort: structural legitimacy is stronger authority than the desired
spelling cleanup.

## Durable artifacts

- Builder-emitted authority:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T085559805386-supersede6/legacy-spelling-supersede-authority.json`
- Machine result with before/after state, receipt ids, assertions, and replay:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T085559805386-supersede6/legacy-spelling-supersede-result.json`
- Complete production log:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T085559805386-supersede6/production-supersede.log`
- Read-only live preflight and survivor derivation logs:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T085559805386-supersede6/live-preflight.log`,
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T085559805386-supersede6/dual-survivor.log`, and
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T085559805386-supersede6/toroidal-derivation-2.log`.

The first toroidal suffix probe attempted an unavailable APOC string helper and
made no write; its diagnostic is retained as `toroidal-derivation.log`. The
corrected native-Cypher derivation is the one used by the successful applying
invocation.
