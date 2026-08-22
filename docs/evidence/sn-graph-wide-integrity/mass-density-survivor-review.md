# Mass-density survivor and scalar-mirror review

## Verdict

**KEEP-BOTH-WITH-DISTINCT-SEMANTICS — reason: the three-rung aggregation
ladder.** The accepted catalog already distinguishes a generic quantity,
an unqualified plasma quantity, and an explicitly total plasma aggregate:
`mass_density` → `plasma_mass_density` → `total_plasma_mass_density`.
The closed ISN aggregation segment makes `total` identity-bearing, while the
unqualified form remains available for a quantity whose aggregation scope is
unspecified or selected by surrounding source semantics.

The earlier survivor decision is therefore **qualified, not affirmed as a
catalog-wide fold**:

- Keeping `mass_density` as the live target of the three already-reconciled DD
  sources is the conservative source-level choice because their DD text says
  only “Mass density” or “One scalar value is provided per element in the grid
  subset.” None states an all-species aggregation.
- Treating `total_plasma_mass_density` as redundant, or retiring it, was
  incorrect. `total` is a closed grammar token and the catalog uses it to
  distinguish an explicit aggregate from an unqualified counterpart.
- The remaining dual-bound `plasma_profiles` source must stay fail-closed until
  DD authority says whether it is the generic mass-density quantity or the
  all-charged-species aggregate. Keeping both *identities* does not authorize
  keeping two live targets on one source.
- The current `mass_density` description itself claims a charged-species sum.
  Under the recommended distinction, that description needs governed catalog
  clarification so the generic name does not silently carry `plasma` and
  `total` semantics only in prose. This review does not edit catalog text.

No graph mutation, lifecycle change, source retarget, supersede, delete, or
provider call occurred.

## Four-source edge/scalar comparison

Live means a `PRODUCED_NAME` target whose `name_stage` is not `superseded`,
`exhausted`, or `contested`. Every source currently has
`produced_sn_id=mass_density`; the table places the live edge closure beside
membership in each identity's `source_paths` scalar.

| DD source | Live `PRODUCED_NAME` target(s) | In `mass_density.source_paths` | In `total_plasma_mass_density.source_paths` | Edge/scalar agreement |
|---|---|---:|---:|---|
| `dd:edge_profiles/ggd/mass_density/values` | `mass_density` | yes | yes | `mass_density` yes; `total_plasma_mass_density` **no — scalar only** |
| `dd:equilibrium/time_slice/profiles_1d/mass_density` | `mass_density` | yes | yes | `mass_density` yes; `total_plasma_mass_density` **no — scalar only** |
| `dd:mhd/ggd/mass_density/values` | `mass_density` | yes | yes | `mass_density` yes; `total_plasma_mass_density` **no — scalar only** |
| `dd:plasma_profiles/ggd/mass_density/values` | `mass_density`; `total_plasma_mass_density` | yes | yes | both agree |

The complete identity closures make the desynchronization explicit:

| Accepted, valid catalog identity | `source_paths` scalar list | Live incoming source-edge closure | Scalar-only entries | Agrees? |
|---|---|---|---|---:|
| `mass_density` | all four sources listed above | all four sources listed above | none | **yes** |
| `total_plasma_mass_density` | all four sources listed above | `dd:plasma_profiles/ggd/mass_density/values` only | `edge_profiles`, `equilibrium`, and `mhd` paths above | **no** |

**Measured mismatch count: 1 of 2 identities.** At source/identity membership
granularity, **3 of 8 comparisons disagree**, all in the same direction:
`total_plasma_mass_density.source_paths` retains three paths whose live
`PRODUCED_NAME` edges were removed. The live view is not stale: it is internally
consistent across each source's edge closure and `produced_sn_id`, and the
identity-level incoming-edge query independently returns the same 4-versus-1
closure.

The DD-side `HAS_STANDARD_NAME` projection still lists all four sources for
both identities. That corroborates a denormalized/projection cleanup gap; it
does not restore the missing `PRODUCED_NAME` edges or change the mismatch count
requested here.

## ISN aggregation authority

The installed public ISN context is `imas-standard-names 0.8.0rc66`. Its
`vocabulary_sections` entry quotes the complete closed aggregation vocabulary
as:

> aggregation tokens: `net`, `total`

The same public context describes the segment as:

> “Population/species/contribution reduction (total, net) — summed or netted
> over sub-populations, species, or additive contributions.”

It further places aggregation outermost in the rendered identity. Therefore
`total_plasma_mass_density` is grammatically meaningful, not a verbose alias
that can be rejected merely because a DD leaf does not spell out `total`.

## Accepted catalog convention

The live graph contains the following six pairs where both the unqualified and
`total_` identities are `name_stage=accepted`, `validation_status=valid`, and
`origin=catalog_edit`. In every row, the total form carries
`aggregation=total`.

| Unqualified accepted identity | Total-aggregated accepted identity | Distinction stated by the catalog descriptions |
|---|---|---|
| `current_density` | `total_current_density` | Unqualified electric-charge flux density versus a sum over current-carrying mechanisms and plasma species. |
| `ion_density` | `total_ion_density` | A specified ion species, charge state, or population versus all represented ion species, charge states, and thermal/non-thermal populations. |
| `ion_energy_flux` | `total_ion_energy_flux` | A signed energy-flux component versus the all-species total including conductive and convective channels. |
| `neutral_density` | `total_neutral_density` | An unresolved neutral population versus an explicit full aggregate over species, energy populations, internal states, and parent-ion-associated states. |
| `plasma_mass_density` | `total_plasma_mass_density` | Unqualified plasma inertial mass density versus the explicit sum of electrons, main ions, and impurity charge states, excluding neutrals. |
| `power_of_neutral_beam_injector` | `total_power_of_neutral_beam_injector` | Power from one injector versus the non-negative sum over all injectors. |

The fifth row is decisive for this family: the catalog already accepts the
direct unqualified counterpart to `total_plasma_mass_density`. The broader
`mass_density` identity is also accepted, valid, and catalog-edited, but its
current prose overlaps the total-plasma meaning. The correct response is to
preserve the identities and govern their descriptions/source assignments, not
to collapse the explicit aggregation token.

The catalog is not perfectly self-disambiguating: some unqualified
descriptions themselves mention broad aggregation. That is a documentation
quality caveat, not evidence that `total` is redundant. Coexistence of these
accepted pairs demonstrates that aggregation scope is intended to remain
identity-bearing when explicitly asserted.

## Source-level disposition

| Source class | Current edge disposition | Review disposition |
|---|---|---|
| `edge_profiles`, `equilibrium`, `mhd` mass-density paths | sole live target `mass_density` | **Keep provisionally.** Their DD text does not establish total aggregation, so there is no authority to move them to the total identity. |
| `plasma_profiles/ggd/mass_density/values` | live targets `mass_density` and `total_plasma_mass_density` | **Keep fail-closed residue.** Obtain DD aggregation semantics, then select exactly one target through the governed source-target operator. |
| `mass_density` catalog identity | four live producers; scalar synchronized | **Keep**, but clarify it as generic/unqualified mass density through catalog governance rather than encoding plasma-total semantics only in prose. |
| `total_plasma_mass_density` catalog identity | one live producer; scalar claims four | **Keep**, with explicit all-charged-species aggregation; separately reconcile its stale scalar and DD projections after the source decisions are authorized. |

This disposition preserves the one-live-name-per-source invariant while
rejecting the false dichotomy that only one identity may exist in the catalog.

## Read-only production receipt

The entire measurement ran in one process through `GraphClient` using only
`MATCH`, `OPTIONAL MATCH`, `WITH`, `UNWIND`, and `RETURN` clauses.

- Receipt `run_id`:
  `r-20260822T053317889668-massdensity`
- Read-only evidence manifest SHA-256:
  `e6807dad573cbbf963c3823babbe936be5034de0e4eacb72f87e5ce038105cbf`
- Live plan version read before execution: **245**
- Source commit:
  `066fcd17a8c930aa723a4c8b980a38ab44b698ad`

Receipt attribution was sampled before and after the evidence queries:

| Receipt query | Before | After | Delta |
|---|---:|---:|---:|
| `StandardNameChange` with exact `run_id` + manifest SHA-256 | **0** | **0** | **0** |
| `StandardNameChange` with this `run_id` under any manifest | **0** | **0** | **0** |
| `StandardNameChange` with this manifest under any run | **0** | **0** | **0** |

The production mutation counters were identical:

| Persistent graph measure | Before | After | Delta |
|---|---:|---:|---:|
| `StandardNameChange` nodes | **7,787** | **7,787** | **0** |
| `PRODUCED_NAME` relationships | **5,780** | **5,780** | **0** |

These exact receipt queries returning zero, together with both counters holding
byte-for-byte at the measured integer values, prove that this node made **no
production graph mutation**.

Machine-readable evidence:

- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T053317889668-massdensity/mass-density-readonly-receipt.json`
  (SHA-256
  `77fc8ac62810b4bcc6b71af7c3b0e7383dc8036684cd7e7828f27b05edc3e564`)
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T053317889668-massdensity/mass-density-readonly-query.log`
  (SHA-256
  `a1d68c8e0e3b3b8eaae41b0289b29df7ae8492b140aa7dcd8c8ff2c28d1b60e5`)
